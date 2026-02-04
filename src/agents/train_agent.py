

"""Training Agent (Module C)

This script acts as an "agent" that:
1) reads labeled segments from PostgreSQL (segment_labels)
2) builds a training dataset (X, y)
3) trains at least 3 classification algorithms
4) evaluates models (macro F1)
5) saves the best model + preprocessing to ./models
6) writes a run record to model_runs and (optionally) saves predictions

Run examples:
  python -m src.agents.train_agent --task risk
  python -m src.agents.train_agent --task evac
  python -m src.agents.train_agent --task both --loop-minutes 30
  python -m src.agents.train_agent --task both --ensure-tables

Notes:
- Connection settings are taken from .env via src.common.db.DBConfig
- The dataset is built from segment_labels.features JSONB plus cluster_id
- Missing values are handled with SimpleImputer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except Exception as e:  # pragma: no cover
    raise RuntimeError("joblib is required. Install it via requirements.txt") from e

# Project DB helpers
from src.common.db import DBConfig, get_connection, set_search_path, test_connection


# -------------------------
# Utilities
# -------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def to_dict(x: Any) -> Dict[str, Any]:
    """Parse JSONB value from DB.

    On macOS it often arrives as dict, on Windows sometimes as str.
    """
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


def safe_get(d: Dict[str, Any], key: str, default: Any = np.nan) -> Any:
    try:
        return d.get(key, default)
    except Exception:
        return default


def ctx_get(d: Dict[str, Any], key: str, default: Any = np.nan) -> Any:
    ctx = d.get("ctx", {}) if isinstance(d.get("ctx", {}), dict) else {}
    return ctx.get(key, default)


# -------------------------
# Flexible DB helpers for model_runs
# -------------------------

def _get_table_columns(cfg: DBConfig, table_name: str) -> List[Dict[str, Any]]:
    """Return column metadata for a table.

    This is used to make the agent compatible with DBs where `model_runs`
    already exists with a different schema (earlier modules / different OS).
    """
    sql = text(
        """
        SELECT column_name, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
        ORDER BY ordinal_position;
        """
    )
    with get_connection() as conn:
        set_search_path(conn)
        rows = conn.execute(sql, {"schema": cfg.schema, "table": table_name}).mappings().all()
    return [dict(r) for r in rows]


def _build_flexible_insert(cfg: DBConfig, task: str, algo: str, metrics: Dict[str, Any], params: Dict[str, Any], artifacts: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Build an INSERT statement that adapts to the existing `model_runs` schema.

    Some competition DBs have extra NOT NULL columns (e.g. `run_tag`).
    We detect required columns and provide reasonable values.
    """
    cols = _get_table_columns(cfg, "model_runs")
    colnames = [c["column_name"] for c in cols]

    ts = _now_utc().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{task}_{algo}_{ts}"

    payload: Dict[str, Any] = {
        "task": task,
        "algorithm": algo,
        "metrics": json.dumps(metrics, ensure_ascii=False),
        "params": json.dumps(params, ensure_ascii=False),
        "artifacts": json.dumps(artifacts, ensure_ascii=False),
        "created_at": _now_utc().isoformat(),
        "run_tag": run_tag,
        "run_name": run_tag,
        "run_version": "1",
        "task_type": "classification",
        "target_name": task,
    }

    # Determine which columns are required (NOT NULL without a default)
    required = []
    for c in cols:
        name = c["column_name"]
        if name == "model_run_id":
            continue
        is_nullable = (c.get("is_nullable") or "YES")
        default = c.get("column_default")
        if is_nullable == "NO" and default is None:
            required.append(name)

    # If the existing DB requires fields we don't know, at least try to set them.
    # We only set keys we have. Unknown required columns will be handled by a clear error.
    insert_cols = [name for name in colnames if name in payload and name != "model_run_id"]

    # Ensure all required columns are present in insert.
    for req in required:
        if req in payload and req not in insert_cols:
            insert_cols.append(req)

    cols_sql = ", ".join(f'"{c}"' for c in insert_cols)
    vals_sql_parts = []
    for c in insert_cols:
        if c in ("metrics", "params", "artifacts"):
            vals_sql_parts.append(f"CAST(:{c} AS jsonb)")
        elif c == "created_at":
            # Let DB default handle it if present; otherwise provide.
            vals_sql_parts.append(f"CAST(:{c} AS timestamptz)")
        else:
            vals_sql_parts.append(f":{c}")

    vals_sql = ", ".join(vals_sql_parts)

    sql = text(
        f"""
        INSERT INTO \"{cfg.schema}\".\"model_runs\" ({cols_sql})
        VALUES ({vals_sql})
        RETURNING model_run_id;
        """
    )

    exec_params = {k: v for k, v in payload.items() if k in insert_cols}

    # If there are required columns we still can't satisfy, crash with a helpful message.
    missing_required = [r for r in required if r not in exec_params]
    if missing_required:
        raise RuntimeError(
            "model_runs has required columns without defaults that the agent cannot fill: "
            + ", ".join(missing_required)
        )

    return sql, exec_params


def ensure_tables(cfg: DBConfig) -> None:
    """Ensure minimal tables for model run tracking exist.

    Important: the repository may already include a `model_runs` table created
    in earlier modules with a different set of columns. For cross-platform and
    "works on existing DB" robustness, we:
      1) CREATE TABLE IF NOT EXISTS with the expected structure
      2) ALTER TABLE ADD COLUMN IF NOT EXISTS for required columns

    This makes the agent safe to run against an existing competition DB.
    """

    # 1) Create tables if they do not exist (fresh DB)
    create_model_runs = f"""
    CREATE TABLE IF NOT EXISTS "{cfg.schema}"."model_runs" (
        model_run_id   BIGSERIAL PRIMARY KEY,
        created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        task           TEXT NOT NULL,
        algorithm      TEXT NOT NULL,
        task_type      TEXT,
        target_name    TEXT,
        run_tag        TEXT,
        metrics        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        params         JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        artifacts      JSONB NOT NULL DEFAULT '{{}}'::jsonb
    );
    """

    create_segment_predictions = f"""
    CREATE TABLE IF NOT EXISTS "{cfg.schema}"."segment_predictions" (
        pred_id        BIGSERIAL PRIMARY KEY,
        model_run_id   BIGINT NOT NULL REFERENCES "{cfg.schema}"."model_runs"(model_run_id) ON DELETE CASCADE,
        segment_id     BIGINT NOT NULL,
        y_true         TEXT NULL,
        y_pred         TEXT NOT NULL,
        y_proba        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    # 2) If tables already exist, migrate them by adding missing columns.
    #    (Your DB might have `model_runs` from earlier steps, without `task`, etc.)
    alter_model_runs = f"""
    ALTER TABLE "{cfg.schema}"."model_runs"
        ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        ADD COLUMN IF NOT EXISTS task TEXT,
        ADD COLUMN IF NOT EXISTS algorithm TEXT,
        ADD COLUMN IF NOT EXISTS task_type TEXT,
        ADD COLUMN IF NOT EXISTS target_name TEXT,
        ADD COLUMN IF NOT EXISTS metrics JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        ADD COLUMN IF NOT EXISTS params JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        ADD COLUMN IF NOT EXISTS artifacts JSONB NOT NULL DEFAULT '{{}}'::jsonb;
    """

    alter_model_runs_extra = f"""
    ALTER TABLE \"{cfg.schema}\".\"model_runs\"
        ADD COLUMN IF NOT EXISTS run_tag TEXT;

    -- If run_tag exists and has no default, set a safe default (does not change existing values).
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = '{cfg.schema}'
              AND table_name = 'model_runs'
              AND column_name = 'run_tag'
              AND column_default IS NULL
        ) THEN
            EXECUTE 'ALTER TABLE "{cfg.schema}"."model_runs" ALTER COLUMN run_tag SET DEFAULT ''manual''';
        END IF;
    END $$;
    """ 

    with get_connection() as conn:
        set_search_path(conn)
        conn.execute(text(create_model_runs))
        conn.execute(text(create_segment_predictions))
        # Migrate existing table schema (no-op if columns already exist)
        conn.execute(text(alter_model_runs))
        conn.execute(text(alter_model_runs_extra))
        conn.commit()


def read_df(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    with get_connection() as conn:
        cfg = DBConfig.from_env()
        set_search_path(conn)
        return pd.read_sql(text(sql), conn, params=params)


# -------------------------
# Data signature and last run helpers
# -------------------------

def get_current_data_signature(limit: Optional[int] = None) -> Dict[str, int]:
    """Compute a simple signature of the labeled dataset.

    We use it to decide whether retraining is needed.
    """
    sql = text(
        """
        SELECT
          COUNT(*)::bigint AS row_count,
          COALESCE(MAX(segment_id), 0)::bigint AS max_segment_id
        FROM segment_labels
        WHERE risk_label IS NOT NULL
          AND evac_label IS NOT NULL;
        """
    )
    with get_connection() as conn:
        set_search_path(conn)
        row = conn.execute(sql).mappings().one()
    return {"row_count": int(row["row_count"]), "max_segment_id": int(row["max_segment_id"]) }


def _table_has_column(cfg: DBConfig, table: str, column: str) -> bool:
    cols = _get_table_columns(cfg, table)
    return any(c.get("column_name") == column for c in cols)


def _parse_json_like(x: Any) -> Dict[str, Any]:
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


def get_last_run_for_task(cfg: DBConfig, task: str) -> Optional[Dict[str, Any]]:
    """Return latest model_runs row for task or None.

    Supports DBs where the task column may be `task` or `target_name`.
    """
    has_task = _table_has_column(cfg, "model_runs", "task")
    has_target = _table_has_column(cfg, "model_runs", "target_name")

    if not has_task and not has_target:
        return None

    if has_task:
        where_col = "task"
    else:
        where_col = "target_name"

    sql = text(
        f"""
        SELECT model_run_id, created_at, algorithm, metrics, params, artifacts
        FROM \"{cfg.schema}\".\"model_runs\"
        WHERE {where_col} = :task
        ORDER BY created_at DESC, model_run_id DESC
        LIMIT 1;
        """
    )

    with get_connection() as conn:
        set_search_path(conn)
        row = conn.execute(sql, {"task": task}).mappings().first()

    if row is None:
        return None

    out = dict(row)
    out["metrics_dict"] = _parse_json_like(out.get("metrics"))
    out["params_dict"] = _parse_json_like(out.get("params"))
    out["artifacts_dict"] = _parse_json_like(out.get("artifacts"))
    return out


# -------------------------
# Dataset
# -------------------------

def load_labeled_segments(limit: Optional[int] = None) -> pd.DataFrame:
    sql = """
    SELECT
      segment_id,
      track_id,
      segment_index,
      cluster_id,
      risk_label,
      evac_label,
      features
    FROM segment_labels
    WHERE risk_label IS NOT NULL
      AND evac_label IS NOT NULL
    ORDER BY segment_id
    """
    if limit is not None:
        sql += " LIMIT :limit"
        df = read_df(sql, {"limit": int(limit)})
    else:
        df = read_df(sql)

    return df


def build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    df = df.copy()
    df["features_dict"] = df["features"].apply(to_dict)

    # segment-level features
    df["distance_m"] = df["features_dict"].apply(lambda d: safe_get(d, "distance_m"))
    df["duration_s"] = df["features_dict"].apply(lambda d: safe_get(d, "duration_s"))
    df["avg_speed_mps"] = df["features_dict"].apply(lambda d: safe_get(d, "avg_speed_mps"))
    df["elev_gain_m"] = df["features_dict"].apply(lambda d: safe_get(d, "elev_gain_m"))
    df["elev_loss_m"] = df["features_dict"].apply(lambda d: safe_get(d, "elev_loss_m"))
    df["elev_min_m"] = df["features_dict"].apply(lambda d: safe_get(d, "elev_min_m"))
    df["elev_max_m"] = df["features_dict"].apply(lambda d: safe_get(d, "elev_max_m"))
    df["slope"] = df["features_dict"].apply(lambda d: safe_get(d, "slope"))
    df["points"] = df["features_dict"].apply(lambda d: safe_get(d, "points"))

    # ctx features (optional)
    df["temp_c"] = df["features_dict"].apply(lambda d: ctx_get(d, "temp_c"))
    df["nearby_waterway_total"] = df["features_dict"].apply(lambda d: ctx_get(d, "nearby_waterway_total", 0.0))
    df["nearby_highway_total"] = df["features_dict"].apply(lambda d: ctx_get(d, "nearby_highway_total", 0.0))
    df["nearby_natural_total"] = df["features_dict"].apply(lambda d: ctx_get(d, "nearby_natural_total", 0.0))
    df["nearby_landuse_total"] = df["features_dict"].apply(lambda d: ctx_get(d, "nearby_landuse_total", 0.0))
    df["nearby_building_total"] = df["features_dict"].apply(lambda d: ctx_get(d, "nearby_building_total", 0.0))

    feature_cols = [
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

    X = df[feature_cols].copy()
    y_risk = df["risk_label"].astype(str)
    y_evac = df["evac_label"].astype(str)
    return X, y_risk, y_evac, feature_cols


# -------------------------
# Training
# -------------------------

def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    # All current features are numeric. Keep transformer for future additions.
    numeric_features = feature_cols

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_features)],
        remainder="drop",
    )


def candidate_models(random_state: int) -> List[Tuple[str, Any]]:
    return [
        (
            "logreg",
            LogisticRegression(
                max_iter=2000,
                n_jobs=None,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto",
                random_state=random_state,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
        ),
        (
            "gboost",
            GradientBoostingClassifier(random_state=random_state),
        ),
    ]


def train_and_select(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    random_state: int,
    test_size: float,
    verbose: bool,
) -> Tuple[str, Pipeline, Dict[str, Any]]:
    """Train >=3 algorithms and select the best by macro F1."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    pre = make_preprocessor(feature_cols)

    results: List[Dict[str, Any]] = []
    best_name = ""
    best_pipe: Optional[Pipeline] = None
    best_f1 = -1.0

    for name, model in candidate_models(random_state=random_state):
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])

        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_s = time.time() - t0

        y_pred = pipe.predict(X_test)
        f1m = float(f1_score(y_test, y_pred, average="macro")) if y_test.nunique() > 1 else 0.0

        rec = {
            "algorithm": name,
            "f1_macro": f1m,
            "train_seconds": float(train_s),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        results.append(rec)

        if verbose:
            _print_header(f"Model: {name}")
            print(f"Train seconds: {train_s:.3f}")
            print(f"F1_macro: {f1m:.4f}")
            print(classification_report(y_test, y_pred, digits=4))

        if f1m > best_f1:
            best_f1 = f1m
            best_name = name
            best_pipe = pipe

    if best_pipe is None:
        raise RuntimeError("No models were trained")

    meta = {
        "best_f1_macro": best_f1,
        "all_results": results,
        "label_distribution": y.value_counts().to_dict(),
    }

    return best_name, best_pipe, meta


def save_artifacts(
    models_dir: Path,
    task: str,
    algo: str,
    pipeline: Pipeline,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)

    ts = _now_utc().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"{task}_{algo}_{ts}.joblib"
    meta_path = models_dir / f"{task}_{algo}_{ts}.json"

    joblib.dump(pipeline, model_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "created_at": _now_utc().isoformat(),
    }


def insert_model_run(cfg: DBConfig, task: str, algo: str, metrics: Dict[str, Any], params: Dict[str, Any], artifacts: Dict[str, Any]) -> int:
    # Build an INSERT that matches the existing DB schema (handles required columns like run_tag).
    sql, exec_params = _build_flexible_insert(
        cfg=cfg,
        task=task,
        algo=algo,
        metrics=metrics,
        params=params,
        artifacts=artifacts,
    )

    with get_connection() as conn:
        set_search_path(conn)
        res = conn.execute(sql, exec_params)
        model_run_id = int(res.scalar_one())
        conn.commit()
    return model_run_id


def save_predictions_to_db(cfg: DBConfig, model_run_id: int, segment_ids: np.ndarray, y_true: Optional[pd.Series], y_pred: np.ndarray, y_proba: Optional[np.ndarray], classes_: Optional[np.ndarray]) -> int:
    """Save per-segment predictions for traceability."""

    insert_sql = text(
        f"""
        INSERT INTO "{cfg.schema}"."segment_predictions" (model_run_id, segment_id, y_true, y_pred, y_proba)
        VALUES (:model_run_id, :segment_id, :y_true, :y_pred, CAST(:y_proba AS jsonb));
        """
    )

    rows = 0
    with get_connection() as conn:
        set_search_path(conn)
        for i in range(len(segment_ids)):
            proba_obj: Dict[str, float] = {}
            if y_proba is not None and classes_ is not None:
                proba_obj = {str(classes_[j]): float(y_proba[i, j]) for j in range(len(classes_))}

            conn.execute(
                insert_sql,
                {
                    "model_run_id": int(model_run_id),
                    "segment_id": int(segment_ids[i]),
                    "y_true": str(y_true.iloc[i]) if y_true is not None else None,
                    "y_pred": str(y_pred[i]),
                    "y_proba": json.dumps(proba_obj, ensure_ascii=False),
                },
            )
            rows += 1
        conn.commit()
    return rows


# -------------------------
# Main agent loop
# -------------------------

def run_once(args: argparse.Namespace) -> None:
    _print_header("Training agent start")

    test_connection(verbose=True)
    cfg = DBConfig.from_env()
    print(f"DB schema: {cfg.schema}")

    if args.ensure_tables:
        ensure_tables(cfg)

    _print_header("Step 1. Load labeled segments")
    df = load_labeled_segments(limit=args.limit)
    print(f"Loaded rows: {len(df)}")
    if df.empty:
        print("No labeled data found in segment_labels. Stop.")
        return

    data_sig = get_current_data_signature(limit=args.limit)
    print(f"Data signature: rows={data_sig['row_count']}, max_segment_id={data_sig['max_segment_id']}")

    _print_header("Step 2. Build dataset")
    X, y_risk, y_evac, feature_cols = build_dataset(df)
    print(f"X shape: {X.shape}")
    print(f"Features: {feature_cols}")
    print("NaN by feature:")
    print(X.isna().sum().to_string())
    print("Label distribution (risk):")
    print(y_risk.value_counts().to_string())
    print("Label distribution (evac):")
    print(y_evac.value_counts().to_string())

    tasks: List[Tuple[str, pd.Series]] = []
    if args.task in ("risk", "both"):
        tasks.append(("risk", y_risk))
    if args.task in ("evac", "both"):
        tasks.append(("evac", y_evac))

    models_dir = Path(args.models_dir)

    for task_name, y in tasks:
        # Decide whether we need to retrain
        last = get_last_run_for_task(cfg, task_name)
        last_sig = None
        last_best_f1 = None
        if last is not None:
            last_sig = (last.get("params_dict") or {}).get("data_signature")
            try:
                last_best_f1 = float((last.get("metrics_dict") or {}).get("best_f1_macro"))
            except Exception:
                last_best_f1 = None

        if (not args.force) and last_sig == data_sig:
            _print_header(f"Step 3. Train models for task: {task_name}")
            print("Skip training: no new labeled data since last run.")
            if last is not None:
                print(f"Last model_run_id={last.get('model_run_id')} | algo={last.get('algorithm')} | best_f1_macro={last_best_f1}")
            continue

        _print_header(f"Step 3. Train models for task: {task_name}")

        # Train and select best
        best_algo, best_pipe, meta = train_and_select(
            X=X,
            y=y,
            feature_cols=feature_cols,
            random_state=args.random_state,
            test_size=args.test_size,
            verbose=args.verbose,
        )

        print("Summary (all algorithms):")
        for r in meta["all_results"]:
            print(f"  {r['algorithm']}: f1_macro={r['f1_macro']:.4f}, train_seconds={r['train_seconds']:.3f}")

        print(f"Best algorithm: {best_algo}")
        print(f"Best F1_macro: {meta['best_f1_macro']:.4f}")

        # Compare to previous best and decide whether to persist
        improved = True
        if last_best_f1 is not None:
            improved = (meta["best_f1_macro"] - last_best_f1) >= float(args.min_improve)

        if not improved:
            _print_header("Step 4. Save artifacts")
            print(
                "No improvement vs last saved model. "
                f"last_best_f1_macro={last_best_f1:.4f} | new_best_f1_macro={meta['best_f1_macro']:.4f} | min_improve={args.min_improve}"
            )
            print("Keeping previous model. Nothing saved to models/ and no DB run inserted.")
            continue

        _print_header("Step 4. Save artifacts")
        artifacts = save_artifacts(
            models_dir=models_dir,
            task=task_name,
            algo=best_algo,
            pipeline=best_pipe,
            meta=meta,
        )
        print("Saved:")
        print(json.dumps(artifacts, ensure_ascii=False, indent=2))

        _print_header("Step 5. Write run info to DB")
        params = {
            "task": task_name,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "feature_cols": feature_cols,
            "limit": args.limit,
            "data_signature": data_sig,
            "min_improve": float(args.min_improve),
        }
        metrics = {
            "best_f1_macro": meta["best_f1_macro"],
            "all_results": meta["all_results"],
            "label_distribution": meta["label_distribution"],
            "previous_best_f1_macro": last_best_f1,
        }

        model_run_id = insert_model_run(
            cfg=cfg,
            task=task_name,
            algo=best_algo,
            metrics=metrics,
            params=params,
            artifacts=artifacts,
        )
        print(f"Inserted model_runs row. model_run_id={model_run_id}")

        if args.save_predictions:
            _print_header("Step 6. Save predictions (optional)")
            # Predict for all rows (not only test) for traceability
            y_pred_all = best_pipe.predict(X)

            y_proba_all = None
            classes_ = None
            model = best_pipe.named_steps.get("model")
            if hasattr(model, "predict_proba"):
                try:
                    y_proba_all = best_pipe.predict_proba(X)
                    classes_ = getattr(model, "classes_", None)
                except Exception:
                    y_proba_all = None
                    classes_ = None

            rows = save_predictions_to_db(
                cfg=cfg,
                model_run_id=model_run_id,
                segment_ids=df["segment_id"].to_numpy(),
                y_true=y,
                y_pred=y_pred_all,
                y_proba=y_proba_all,
                classes_=classes_,
            )
            print(f"Inserted segment_predictions rows: {rows}")

    _print_header("Training agent finished")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training agent for Module C (risk/evac classification)")

    p.add_argument(
        "--task",
        choices=["risk", "evac", "both"],
        default="both",
        help="Which target to train: risk, evac or both",
    )
    p.add_argument("--models-dir", default=str(Path("models")), help="Directory to save .joblib artifacts")
    p.add_argument("--test-size", type=float, default=0.25, help="Test split ratio")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if dataset signature did not change",
    )
    p.add_argument(
        "--min-improve",
        type=float,
        default=0.0005,
        help="Minimal F1_macro improvement to replace the previous saved model",
    )
    p.add_argument("--limit", type=int, default=None, help="Limit rows for quick debug")
    p.add_argument(
        "--save-predictions",
        action="store_true",
        help="If set, save per-segment predictions into segment_predictions",
    )
    p.add_argument(
        "--ensure-tables",
        action="store_true",
        help="If set, create/migrate model_runs and segment_predictions tables (safe to run multiple times)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print full classification reports for each candidate model",
    )
    p.add_argument(
        "--loop-minutes",
        type=int,
        default=0,
        help="If >0, run training in a loop every N minutes (continuous learning imitation)",
    )

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.loop_minutes and args.loop_minutes > 0:
        _print_header("Continuous training mode")
        print(f"Loop interval minutes: {args.loop_minutes}")
        while True:
            try:
                run_once(args)
            except KeyboardInterrupt:
                print("Interrupted by user. Stop.")
                return
            except Exception as e:
                print("Agent run failed:")
                print(str(e))

            sleep_s = int(args.loop_minutes * 60)
            print(f"Sleep seconds: {sleep_s}")
            time.sleep(sleep_s)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
# Обычный запуск (не трогает схемы таблиц)
# python -m src.agents.train_agent --task both          # will skip if no new data
# python -m src.agents.train_agent --task both --force  # always retrain
# python -m src.agents.train_agent --task both --loop-minutes 30  # periodic check; trains only if new data and better
# Если нужно создать/мигрировать таблицы model_runs/segment_predictions
# python -m src.agents.train_agent --task both --ensure-tables
# сохранить  вбазу предсказания
# python -m src.agents.train_agent --task both --save-predictions