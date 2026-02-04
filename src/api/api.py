"""FastAPI service for GPX segment risk/evac prediction.

What it does
------------
- Loads the latest trained models from ./models (risk + evac).
- Finds nearest labeled segments in Postgres (segment_labels) to a given coordinate.
- Builds a feature matrix from segment_labels.features JSONB + cluster_id.
- Runs both models and returns predictions + probabilities.
- Shows continuous-learning heartbeat: every 30 seconds logs DB signature and (optionally)
  triggers the training agent if new data is detected.

Run (dev)
---------
python -m uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload

Notes
-----
- Designed to be cross-platform (macOS/Windows/Linux).
- Does NOT require PostGIS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import joblib
except Exception as e:  # pragma: no cover
    raise RuntimeError("joblib is required. Add it to requirements.txt") from e

# Project DB helpers (expected to exist in your repo)
try:
    from src.common.db import DBConfig, get_connection, set_search_path, test_connection
except Exception as e:  # pragma: no cover
    # If someone runs only the API without the repo layout, give a clear error.
    raise RuntimeError(
        "Cannot import src.common.db. Make sure you run from project root and that src/common/db.py exists."
    ) from e


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("mlbox.api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# -----------------------------------------------------------------------------
# Constants: feature columns must match the training agent
# -----------------------------------------------------------------------------
FEATURE_COLS: List[str] = [
    "distance_m",
    "duration_s",
    "avg_speed_mps",
    "elev_gain_m",
    "elev_loss_m",
    "elev_min_m",
    "elev_max_m",
    "slope",
    "points",
    "temp_c",
    "nearby_waterway_total",
    "nearby_highway_total",
    "nearby_natural_total",
    "nearby_landuse_total",
    "nearby_building_total",
    "cluster_id",
]

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"  # project_root/models


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_dict(x: Any) -> Dict[str, Any]:
    """JSONB in psycopg2 can come as dict, and on some setups as a str."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def ctx_get(d: Dict[str, Any], k: str, default: Any = np.nan) -> Any:
    ctx = d.get("ctx", {}) if isinstance(d.get("ctx", {}), dict) else {}
    return ctx.get(k, default)


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two coordinates in meters."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})\.(joblib|json)$")


def _extract_ts(path: Path) -> Optional[datetime]:
    m = _TIMESTAMP_RE.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def find_latest_artifacts(task: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (joblib_path, json_path) for the newest model of a task."""
    if not MODELS_DIR.exists():
        return None, None

    joblibs = sorted(MODELS_DIR.glob(f"{task}_*.joblib"))
    if not joblibs:
        return None, None

    # Prefer timestamped filenames; fallback to mtime
    def key(p: Path):
        ts = _extract_ts(p)
        return (ts is not None, ts or datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc))

    best_joblib = sorted(joblibs, key=key)[-1]
    best_json = best_joblib.with_suffix(".json")
    if not best_json.exists():
        best_json = None

    return best_joblib, best_json


@dataclass
class LoadedModel:
    task: str
    model_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    model: Any = None
    meta: Dict[str, Any] = None

    def is_loaded(self) -> bool:
        return self.model is not None and self.model_path is not None


class ModelRegistry:
    def __init__(self):
        self._cache: Dict[str, LoadedModel] = {}

    def load(self, task: str) -> LoadedModel:
        joblib_path, meta_path = find_latest_artifacts(task)
        cached = self._cache.get(task)

        if cached and cached.model_path == joblib_path and cached.is_loaded():
            return cached

        lm = LoadedModel(task=task, model_path=joblib_path, meta_path=meta_path, model=None, meta={})

        if joblib_path is None:
            self._cache[task] = lm
            return lm

        lm.model = joblib.load(joblib_path)
        if meta_path and meta_path.exists():
            try:
                lm.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                lm.meta = {}

        self._cache[task] = lm
        logger.info(
            "Loaded model | task=%s | model=%s",
            task,
            str(joblib_path).replace(str(MODELS_DIR.parent), "."),
        )
        return lm


registry = ModelRegistry()


# -----------------------------------------------------------------------------
# DB queries
# -----------------------------------------------------------------------------


def db_signature(cfg: DBConfig) -> Dict[str, Any]:
    """A small signature to detect new data.

    We don't assume that segment_labels has an `updated_at` column.
    Different student DB schemas may have either `updated_at`, `created_at`, or neither.

    Signature fields are kept stable so the continuous-learning loop can compare them.
    """

    # Detect which timestamp column exists (updated_at preferred, then created_at)
    ts_col: Optional[str] = None
    with get_connection() as conn:
        set_search_path(conn)
        q = text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = 'segment_labels'
              AND column_name IN ('updated_at', 'created_at')
            ORDER BY CASE column_name WHEN 'updated_at' THEN 0 WHEN 'created_at' THEN 1 ELSE 2 END
            LIMIT 1;
            """
        )
        res = conn.execute(q, {"schema": cfg.schema}).fetchone()
        if res:
            ts_col = str(res[0])

    # Build the signature query using the existing timestamp column (if any)
    if ts_col:
        sql = f"""
        SELECT
          COUNT(*)::bigint AS rows,
          COALESCE(MAX(segment_id), 0)::bigint AS max_segment_id,
          COALESCE(MAX({ts_col}), TIMESTAMP '1970-01-01') AS max_ts
        FROM "{cfg.schema}"."segment_labels";
        """
    else:
        sql = f"""
        SELECT
          COUNT(*)::bigint AS rows,
          COALESCE(MAX(segment_id), 0)::bigint AS max_segment_id,
          TIMESTAMP '1970-01-01' AS max_ts
        FROM "{cfg.schema}"."segment_labels";
        """

    with get_connection() as conn:
        set_search_path(conn)
        df = pd.read_sql(sql, conn)

    row = df.iloc[0].to_dict()

    # Keep it JSON-serializable
    row["max_ts"] = str(row.get("max_ts"))
    row["ts_col"] = ts_col

    return row


def fetch_candidate_segments(
    cfg: DBConfig,
    lat: float,
    lon: float,
    radius_m: float,
    limit: int,
) -> pd.DataFrame:
    """Fetch candidates using a bounding box, then exact distance computed in Python."""
    # Rough bbox degrees
    # 1 deg lat ~ 111_320 m
    dlat = radius_m / 111_320.0
    # 1 deg lon depends on latitude
    dlon = radius_m / (111_320.0 * max(0.1, float(np.cos(np.radians(lat)))))

    sql = f"""
    SELECT
      segment_id,
      track_id,
      segment_index,
      lat_center,
      lon_center,
      cluster_id,
      risk_label,
      evac_label,
      features
    FROM "{cfg.schema}"."segment_labels"
    WHERE lat_center BETWEEN :lat_min AND :lat_max
      AND lon_center BETWEEN :lon_min AND :lon_max
    LIMIT :limit;
    """

    params = {
        "lat_min": lat - dlat,
        "lat_max": lat + dlat,
        "lon_min": lon - dlon,
        "lon_max": lon + dlon,
        "limit": int(max(50, limit * 20)),  # fetch extra; filter precisely later
    }

    with get_connection() as conn:
        set_search_path(conn)
        df = pd.read_sql(text(sql), conn, params=params)

    if df.empty:
        return df

    # Exact distance
    df["distance_to_query_m"] = df.apply(
        lambda r: haversine_m(lat, lon, float(r["lat_center"]), float(r["lon_center"])), axis=1
    )
    df = df[df["distance_to_query_m"] <= float(radius_m)].copy()
    df = df.sort_values("distance_to_query_m").head(int(limit)).reset_index(drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feats = df["features"].apply(to_dict)

    out = pd.DataFrame(index=df.index)

    # Base segment features
    out["distance_m"] = feats.apply(lambda d: safe_float(d.get("distance_m")))
    out["duration_s"] = feats.apply(lambda d: safe_float(d.get("duration_s")))
    out["avg_speed_mps"] = feats.apply(lambda d: safe_float(d.get("avg_speed_mps")))
    out["elev_gain_m"] = feats.apply(lambda d: safe_float(d.get("elev_gain_m")))
    out["elev_loss_m"] = feats.apply(lambda d: safe_float(d.get("elev_loss_m")))
    out["elev_min_m"] = feats.apply(lambda d: safe_float(d.get("elev_min_m")))
    out["elev_max_m"] = feats.apply(lambda d: safe_float(d.get("elev_max_m")))
    out["slope"] = feats.apply(lambda d: safe_float(d.get("slope")))
    out["points"] = feats.apply(lambda d: safe_float(d.get("points")))

    # Context features nested in ctx
    out["temp_c"] = feats.apply(lambda d: safe_float(ctx_get(d, "temp_c")))
    out["nearby_waterway_total"] = feats.apply(lambda d: safe_float(ctx_get(d, "nearby_waterway_total", 0.0)))
    out["nearby_highway_total"] = feats.apply(lambda d: safe_float(ctx_get(d, "nearby_highway_total", 0.0)))
    out["nearby_natural_total"] = feats.apply(lambda d: safe_float(ctx_get(d, "nearby_natural_total", 0.0)))
    out["nearby_landuse_total"] = feats.apply(lambda d: safe_float(ctx_get(d, "nearby_landuse_total", 0.0)))
    out["nearby_building_total"] = feats.apply(lambda d: safe_float(ctx_get(d, "nearby_building_total", 0.0)))

    # Cluster
    out["cluster_id"] = df["cluster_id"].astype(float)

    return out[FEATURE_COLS].copy()


# -----------------------------------------------------------------------------
# API schemas
# -----------------------------------------------------------------------------


class PredictRequest(BaseModel):
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    radius_m: float = Field(500.0, ge=10.0, le=50_000.0, description="Search radius in meters")
    top_k: int = Field(10, ge=1, le=100, description="Max segments to return")


class TrainRequest(BaseModel):
    task: str = Field("both", description="risk | evac | both")


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI(title="MLBox API", version="1.0")

 # -----------------------------------------------------------------------------
 # Config
 # -----------------------------------------------------------------------------

def _load_env_file(path: Path) -> None:
    """Minimal .env loader (no external deps).

    Loads KEY=VALUE pairs into os.environ only if the key is not already set.
    Supports quoted values and ignores comments/blank lines.
    """
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        # If .env is malformed, we fail later with a clearer message.
        return


def _env_get(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        val = os.getenv(k)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return default


# Load .env from project root (repo_root/.env)
_PROJECT_ROOT = MODELS_DIR.parent
_load_env_file(_PROJECT_ROOT / ".env")


def load_db_config() -> DBConfig:
    """Create DBConfig from environment variables.

    Expected keys (any of these will work):
    - host: DB_HOST, PGHOST
    - port: DB_PORT, PGPORT
    - db:   DB_NAME, DB_DB, PGDATABASE
    - user: DB_USER, PGUSER
    - password: DB_PASSWORD, PGPASSWORD
    - schema: DB_SCHEMA (optional, default=public)
    """
    host = _env_get("DB_HOST", "PGHOST")
    port_s = _env_get("DB_PORT", "PGPORT")
    db = _env_get("DB_NAME", "DB_DB", "PGDATABASE")
    user = _env_get("DB_USER", "PGUSER")
    password = _env_get("DB_PASSWORD", "PGPASSWORD")
    schema = _env_get("DB_SCHEMA", default="public")

    missing = [k for k, v in [("host", host), ("port", port_s), ("db", db), ("user", user), ("password", password)] if v is None]
    if missing:
        raise RuntimeError(
            "DBConfig is missing required values: "
            + ", ".join(missing)
            + ". Set them in .env (project root) or as environment variables."
        )

    try:
        port = int(port_s)  # type: ignore[arg-type]
    except Exception as e:
        raise RuntimeError(f"Invalid DB_PORT/PGPORT value: {port_s!r}") from e

    return DBConfig(host=host, port=port, db=db, user=user, password=password, schema=schema)  # type: ignore[arg-type]



# Global config
# Prefer using helper constructors from src.common.db (if they exist), otherwise fall back to env-based loader here.

def make_db_config() -> DBConfig:
    # Some repos define DBConfig() with no args (it reads env inside), or DBConfig.from_env().
    # We support all variants to be robust across macOS/Windows/Linux.
    ctor = getattr(DBConfig, "from_env", None)
    if callable(ctor):
        return ctor()  # type: ignore[misc]

    # If DBConfig has a no-arg constructor, use it.
    try:
        return DBConfig()  # type: ignore[call-arg]
    except TypeError:
        # Otherwise use our local .env/environment loader.
        return load_db_config()


cfg = make_db_config()

# Continuous learning state
_last_sig: Optional[Dict[str, Any]] = None
_last_tick: Optional[str] = None


async def continuous_learning_loop(interval_seconds: int = 30) -> None:
    """Every N seconds prints what happened.

    Strategy:
    - Read DB signature.
    - If it changed (new rows / new max_segment_id / updated_at), run training agent.
    - Reload models after training.

    This is a SIMPLE imitation of continuous learning (enough for the assignment).
    """
    global _last_sig, _last_tick

    while True:
        try:
            sig = db_signature(cfg)
            tick = now_utc()
            changed = sig != _last_sig

            # Load current models (cached)
            risk_m = registry.load("risk")
            evac_m = registry.load("evac")

            logger.info(
                "[CL] tick=%s | sig=%s | changed=%s | models: risk=%s evac=%s",
                tick,
                sig,
                changed,
                "yes" if risk_m.is_loaded() else "no",
                "yes" if evac_m.is_loaded() else "no",
            )

            # If DB changed, trigger training
            if changed and _last_sig is not None:
                logger.info("[CL] Detected new/updated data. Trigger training agent...")
                _run_training_subprocess(task="both")
                # Reload models after potential update
                registry.load("risk")
                registry.load("evac")

            _last_sig = sig
            _last_tick = tick

        except Exception as e:
            logger.exception("[CL] loop error: %s", e)

        await asyncio.sleep(interval_seconds)


def _run_training_subprocess(task: str = "both") -> str:
    """Run training agent as a subprocess (cross-platform). Returns stdout+stderr."""
    cmd = [sys.executable, "-m", "src.agents.train_agent", "--task", task]
    p = subprocess.run(
        cmd,
        cwd=str(MODELS_DIR.parent),
        capture_output=True,
        text=True,
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        logger.error("Training agent failed (rc=%s). Output:\n%s", p.returncode, out[-4000:])
        raise RuntimeError(f"Training agent failed with code {p.returncode}")
    logger.info("Training agent finished successfully.")
    return out


@app.on_event("startup")
async def on_startup() -> None:
    # DB connectivity check
    ok_msg = test_connection(cfg)
    logger.info("%s", ok_msg)

    # Load models if exist
    registry.load("risk")
    registry.load("evac")

    # Start continuous learning heartbeat
    interval = int(os.getenv("CL_INTERVAL_SECONDS", "30"))
    asyncio.create_task(continuous_learning_loop(interval_seconds=interval))


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        sig = db_signature(cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    risk_m = registry.load("risk")
    evac_m = registry.load("evac")

    return {
        "status": "ok",
        "time_utc": now_utc(),
        "db": {
            "host": cfg.host,
            "port": cfg.port,
            "db": cfg.db,
            "schema": cfg.schema,
            "signature": sig,
        },
        "models": {
            "risk": {
                "loaded": risk_m.is_loaded(),
                "model_path": str(risk_m.model_path) if risk_m.model_path else None,
                "meta": risk_m.meta or {},
            },
            "evac": {
                "loaded": evac_m.is_loaded(),
                "model_path": str(evac_m.model_path) if evac_m.model_path else None,
                "meta": evac_m.meta or {},
            },
        },
        "continuous_learning": {
            "last_tick": _last_tick,
            "last_signature": _last_sig,
        },
    }


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    # Load models
    risk_m = registry.load("risk")
    evac_m = registry.load("evac")
    if not (risk_m.is_loaded() and evac_m.is_loaded()):
        raise HTTPException(
            status_code=500,
            detail="Models are not loaded yet. Train agent first (POST /train) or place artifacts in ./models.",
        )

    # Fetch candidates
    cand = fetch_candidate_segments(cfg, req.lat, req.lon, req.radius_m, req.top_k)
    if cand.empty:
        return {
            "query": req.model_dump(),
            "found": 0,
            "message": "No segments found in radius. Try larger radius_m.",
            "segments": [],
        }

    X = build_feature_matrix(cand)

    # Predict
    risk_pred = risk_m.model.predict(X)
    evac_pred = evac_m.model.predict(X)

    # Proba if available
    def proba_or_none(model, X_):
        if hasattr(model, "predict_proba"):
            try:
                return model.predict_proba(X_)
            except Exception:
                return None
        return None

    risk_proba = proba_or_none(risk_m.model, X)
    evac_proba = proba_or_none(evac_m.model, X)

    segments: List[Dict[str, Any]] = []
    for i in range(len(cand)):
        row = cand.iloc[i]
        item = {
            "segment_id": int(row["segment_id"]),
            "track_id": int(row["track_id"]),
            "segment_index": int(row["segment_index"]),
            "distance_to_query_m": float(row["distance_to_query_m"]),
            "lat_center": float(row["lat_center"]),
            "lon_center": float(row["lon_center"]),
            "cluster_id": int(row["cluster_id"]) if row["cluster_id"] is not None else None,
            "db_labels": {
                "risk_label": row.get("risk_label"),
                "evac_label": row.get("evac_label"),
            },
            "pred": {
                "risk_label": str(risk_pred[i]),
                "evac_label": str(evac_pred[i]),
            },
        }

        if risk_proba is not None and hasattr(risk_m.model, "classes_"):
            item["pred"]["risk_proba"] = {
                str(cls): float(risk_proba[i, j]) for j, cls in enumerate(risk_m.model.classes_)
            }
        if evac_proba is not None and hasattr(evac_m.model, "classes_"):
            item["pred"]["evac_proba"] = {
                str(cls): float(evac_proba[i, j]) for j, cls in enumerate(evac_m.model.classes_)
            }

        segments.append(item)

    # Summary: worst cases
    evac_order = {"easy": 0, "medium": 1, "hard": 2}
    worst_evac = sorted((s["pred"]["evac_label"] for s in segments), key=lambda x: evac_order.get(x, -1))[-1]

    risk_priority = {"none": 0, "flood": 1, "fire": 2}
    worst_risk = sorted((s["pred"]["risk_label"] for s in segments), key=lambda x: risk_priority.get(x, -1))[-1]

    return {
        "query": req.model_dump(),
        "found": len(segments),
        "summary": {
            "worst_risk": worst_risk,
            "worst_evac": worst_evac,
        },
        "segments": segments,
    }


@app.post("/train")
def train(req: TrainRequest) -> Dict[str, Any]:
    task = req.task.strip().lower()
    if task not in {"risk", "evac", "both"}:
        raise HTTPException(status_code=400, detail="task must be one of: risk, evac, both")

    try:
        out = _run_training_subprocess(task=task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Reload models after training
    registry.load("risk")
    registry.load("evac")

    return {
        "status": "ok",
        "task": task,
        "time_utc": now_utc(),
        "output_tail": out[-4000:],
    }


@app.get("/models")
def models() -> Dict[str, Any]:
    risk_m = registry.load("risk")
    evac_m = registry.load("evac")
    return {
        "time_utc": now_utc(),
        "models_dir": str(MODELS_DIR),
        "risk": {
            "model_path": str(risk_m.model_path) if risk_m.model_path else None,
            "meta_path": str(risk_m.meta_path) if risk_m.meta_path else None,
            "meta": risk_m.meta or {},
        },
        "evac": {
            "model_path": str(evac_m.model_path) if evac_m.model_path else None,
            "meta_path": str(evac_m.meta_path) if evac_m.meta_path else None,
            "meta": evac_m.meta or {},
        },
    }

# python -m uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload
# Что смотреть/тестировать
# 	•	Проверка состояния:
# 	•	GET http://localhost:8000/health
# 	•	Посмотреть какие модели подхватились:
# 	•	GET http://localhost:8000/models
# 	•	Предсказать по координатам:
# 	•	POST http://localhost:8000/predict