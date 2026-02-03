

"""src.build_track_features

Универсальный расчёт агрегированных признаков для GPX-треков.

Что делает:
- читает `tracks` и `track_points`
- считает агрегаты по каждому треку (distance/duration/speed/elevation/stop-time/...)
- пишет результат в `track_features` (UPSERT по track_id)

Зачем:
- для кластеризации и ML удобнее работать не с миллионами точек, а с 1 строкой на трек.
- для дашборда/аналитики нужен быстрый слой признаков.

Важно про универсальность:
- если в исходных GPX нет времени -> duration/avg_speed/max_speed будут NULL
- если нет высоты -> elev_* будут NULL
- если нет speed -> max_speed будет NULL, stop_time будет считаться по геометрии (если есть time)
- скрипт НЕ падает из‑за NULL: аккуратно пропускает недоступные расчёты.

Запуск:
  python -m src.build_track_features
  python -m src.build_track_features --limit 10
  python -m src.build_track_features --force  (пересчитать все)

Что может понадобиться поменять на соревнованиях:
- SCHEMA (если дадут не public) -> через .env PG_SCHEMA или флаг --schema
- stop-speed-threshold (порог «стопа») -> флаг --stop-speed
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

# В проекте уже есть общий модуль подключения.
# Мы опираемся на то, что у тебя ранее работали:
#   from src.common.db import get_engine
from src.common.db import get_engine


# -----------------------------
# Конфигурация / аргументы
# -----------------------------

@dataclass
class BuildCfg:
    schema: str
    limit: Optional[int]
    force: bool
    stop_speed_mps: float
    min_move_m: float
    verbose: bool


def parse_args() -> BuildCfg:
    p = argparse.ArgumentParser(description="Build per-track features into track_features")
    p.add_argument(
        "--schema",
        default=None,
        help="PostgreSQL schema (default: from PG_SCHEMA in .env or 'public')",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N tracks (useful for quick testing)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute features even if track_features already has a row",
    )
    p.add_argument(
        "--stop-speed",
        type=float,
        default=0.5,
        help="Speed threshold (m/s) below which we count time as 'stop' when speed is available",
    )
    p.add_argument(
        "--min-move-m",
        type=float,
        default=1.0,
        help="If speed is not available, we treat segment as stop when delta distance < this (meters)",
    )
    p.add_argument("--verbose", action="store_true", help="More logs")

    a = p.parse_args()

    # Схему лучше брать из PG_SCHEMA, но чтобы не зависеть от конкретной функции,
    # сначала попробуем прочитать через SQL (current_schema) если не передали.
    schema = a.schema
    if schema is None:
        # fallback (часто в .env у тебя PG_SCHEMA=public)
        schema = "public"

    return BuildCfg(
        schema=schema,
        limit=a.limit,
        force=a.force,
        stop_speed_mps=a.stop_speed,
        min_move_m=a.min_move_m,
        verbose=a.verbose,
    )


# -----------------------------
# Гео‑утилиты
# -----------------------------

EARTH_R = 6371000.0  # meters


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Векторизованный haversine, метры."""
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_R * c


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


# -----------------------------
# SQL helpers
# -----------------------------


def qname(schema: str, table: str) -> str:
    """Кавычим идентификаторы, чтобы не зависеть от регистра/символов."""
    return f'"{schema}"."{table}"'


# -----------------------------
# Основная логика
# -----------------------------


def fetch_track_ids(engine, schema: str, limit: Optional[int], force: bool) -> List[int]:
    """Берём список track_id, которые надо обработать."""
    tracks_t = qname(schema, "tracks")
    feats_t = qname(schema, "track_features")

    if force:
        sql = f"""
        SELECT track_id
        FROM {tracks_t}
        ORDER BY track_id
        """
        params: Dict[str, Any] = {}
    else:
        # обрабатываем только те треки, для которых ещё нет features
        sql = f"""
        SELECT t.track_id
        FROM {tracks_t} t
        LEFT JOIN {feats_t} f ON f.track_id = t.track_id
        WHERE f.track_id IS NULL
        ORDER BY t.track_id
        """
        params = {}

    if limit is not None and limit > 0:
        sql += " LIMIT :limit"
        params["limit"] = limit

    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [int(r[0]) for r in rows]


def fetch_track_points(engine, schema: str, track_id: int) -> pd.DataFrame:
    """Достаём точки трека в правильном порядке."""
    tp = qname(schema, "track_points")

    # Мы используем ORDER BY segment_index, seq.
    # Если где-то segment_index NULL/0 — всё равно порядок сохраняется.
    sql = f"""
    SELECT
        track_id,
        segment_index,
        seq,
        lat,
        lon,
        ele,
        time,
        speed,
        heading
    FROM {tp}
    WHERE track_id = :track_id
    ORDER BY segment_index ASC, seq ASC;
    """

    df = pd.read_sql(text(sql), engine, params={"track_id": track_id})
    return df


def compute_features(df: pd.DataFrame, cfg: BuildCfg) -> Dict[str, Any]:
    """Считаем признаки по точкам одного трека."""
    # обязательные поля
    lat = df["lat"].astype(float)
    lon = df["lon"].astype(float)

    # distance
    if len(df) < 2:
        distance_m = 0.0
        seg_dist = np.array([], dtype=float)
    else:
        seg_dist = haversine_m(lat.values[:-1], lon.values[:-1], lat.values[1:], lon.values[1:])
        # грязные данные: если вдруг nan или inf
        seg_dist = np.nan_to_num(seg_dist, nan=0.0, posinf=0.0, neginf=0.0)
        distance_m = float(seg_dist.sum())

    # time / duration
    has_time = df["time"].notna().any()
    duration_s: Optional[float] = None
    start_time = None
    end_time = None
    dt_s: Optional[np.ndarray] = None

    if has_time:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
        # если всё оказалось NaT -> считаем, что времени нет
        if t.notna().any():
            start_time = t.dropna().iloc[0]
            end_time = t.dropna().iloc[-1]
            if start_time is not pd.NaT and end_time is not pd.NaT:
                duration_s = float((end_time - start_time).total_seconds())

            # интервалы времени между соседними точками
            if len(df) >= 2:
                t1 = t.iloc[:-1]
                t2 = t.iloc[1:]
                dt = (t2.values.astype("datetime64[ns]") - t1.values.astype("datetime64[ns]"))
                dt_s = dt.astype("timedelta64[s]").astype(float)
                dt_s = np.nan_to_num(dt_s, nan=0.0, posinf=0.0, neginf=0.0)

    # elevation
    has_ele = df["ele"].notna().any()
    elev_min_m: Optional[float] = None
    elev_max_m: Optional[float] = None
    elev_gain_m: Optional[float] = None
    elev_loss_m: Optional[float] = None

    if has_ele:
        ele = pd.to_numeric(df["ele"], errors="coerce")
        if ele.notna().any():
            elev_min_m = float(ele.min())
            elev_max_m = float(ele.max())
            if len(df) >= 2:
                de = ele.values[1:] - ele.values[:-1]
                de = np.nan_to_num(de, nan=0.0, posinf=0.0, neginf=0.0)
                elev_gain_m = float(de[de > 0].sum())
                elev_loss_m = float((-de[de < 0]).sum())

    # speed
    has_speed = df["speed"].notna().any()
    avg_speed_mps: Optional[float] = None
    max_speed_mps: Optional[float] = None

    if has_time and duration_s and duration_s > 0 and distance_m >= 0:
        avg_speed_mps = float(distance_m / duration_s)

    if has_speed:
        sp = pd.to_numeric(df["speed"], errors="coerce")
        if sp.notna().any():
            max_speed_mps = float(sp.max())

    # stop time
    stop_time_s: Optional[float] = None
    stop_ratio: Optional[float] = None

    if has_time and dt_s is not None and len(dt_s) == max(len(df) - 1, 0):
        if has_speed:
            sp = pd.to_numeric(df["speed"], errors="coerce")
            sp1 = sp.iloc[:-1].values
            sp1 = np.nan_to_num(sp1, nan=np.nan)  # оставим nan
            is_stop = np.less_equal(sp1, cfg.stop_speed_mps)
            # nan speed -> считаем как не стоп (безопаснее, чем завышать stop)
            is_stop = np.where(np.isnan(sp1), False, is_stop)
            stop_time_s = float((dt_s * is_stop).sum())
        else:
            # если speed нет — берём «стоп» как очень маленькое перемещение
            if len(seg_dist) == len(dt_s):
                is_stop = seg_dist < cfg.min_move_m
                stop_time_s = float((dt_s * is_stop).sum())

        if duration_s and duration_s > 0 and stop_time_s is not None:
            stop_ratio = float(stop_time_s / duration_s)

    # point density
    point_density_per_km: Optional[float] = None
    if distance_m > 0:
        point_density_per_km = float(len(df) / (distance_m / 1000.0))

    # Итог
    out: Dict[str, Any] = {
        "distance_m": distance_m,
        "duration_s": duration_s,
        "elev_min_m": elev_min_m,
        "elev_max_m": elev_max_m,
        "elev_gain_m": elev_gain_m,
        "elev_loss_m": elev_loss_m,
        "avg_speed_mps": avg_speed_mps,
        "max_speed_mps": max_speed_mps,
        "stop_time_s": stop_time_s,
        "stop_ratio": stop_ratio,
        "point_density_per_km": point_density_per_km,
        "updated_at": datetime.utcnow(),
        # extra — место под всё нестандартное/дополнительное (не ломает схему)
        "extra": {
            "has_time": bool(has_time),
            "has_ele": bool(has_ele),
            "has_speed": bool(has_speed),
            "points": int(len(df)),
        },
    }

    return out


def upsert_features(engine, schema: str, track_id: int, feats: Dict[str, Any]) -> None:
    tf = qname(schema, "track_features")

    sql = f"""
    INSERT INTO {tf} (
        track_id,
        distance_m,
        duration_s,
        elev_min_m,
        elev_max_m,
        elev_gain_m,
        elev_loss_m,
        avg_speed_mps,
        max_speed_mps,
        stop_time_s,
        stop_ratio,
        point_density_per_km,
        updated_at,
        extra
    ) VALUES (
        :track_id,
        :distance_m,
        :duration_s,
        :elev_min_m,
        :elev_max_m,
        :elev_gain_m,
        :elev_loss_m,
        :avg_speed_mps,
        :max_speed_mps,
        :stop_time_s,
        :stop_ratio,
        :point_density_per_km,
        :updated_at,
        :extra
    )
    ON CONFLICT (track_id) DO UPDATE SET
        distance_m = EXCLUDED.distance_m,
        duration_s = EXCLUDED.duration_s,
        elev_min_m = EXCLUDED.elev_min_m,
        elev_max_m = EXCLUDED.elev_max_m,
        elev_gain_m = EXCLUDED.elev_gain_m,
        elev_loss_m = EXCLUDED.elev_loss_m,
        avg_speed_mps = EXCLUDED.avg_speed_mps,
        max_speed_mps = EXCLUDED.max_speed_mps,
        stop_time_s = EXCLUDED.stop_time_s,
        stop_ratio = EXCLUDED.stop_ratio,
        point_density_per_km = EXCLUDED.point_density_per_km,
        updated_at = EXCLUDED.updated_at,
        extra = EXCLUDED.extra;
    """

    params = {"track_id": track_id, **feats}
    # jsonb: sqlalchemy/psycopg2 понимают dict, но безопаснее явно
    params["extra"] = json.dumps(feats.get("extra", {}) or {})

    with engine.begin() as conn:
        conn.execute(text(sql), params)


def main() -> int:
    cfg = parse_args()
    engine = get_engine()

    # Если пользователь не передал schema и в БД current_schema не public —
    # можно адаптироваться. Но чтобы не усложнять: используем cfg.schema.

    track_ids = fetch_track_ids(engine, cfg.schema, cfg.limit, cfg.force)
    if not track_ids:
        print("Nothing to do: no tracks to process")
        return 0

    print(f"BUILD: schema={cfg.schema} tracks={len(track_ids)} force={cfg.force} limit={cfg.limit}")

    ok = 0
    fail = 0

    for i, track_id in enumerate(track_ids, start=1):
        try:
            df = fetch_track_points(engine, cfg.schema, track_id)
            if df.empty:
                # трек без точек — это странно, но бывает при неполной загрузке
                if cfg.verbose:
                    print(f"SKIP: track_id={track_id} (no points)")
                continue

            feats = compute_features(df, cfg)
            upsert_features(engine, cfg.schema, track_id, feats)
            ok += 1

            if cfg.verbose:
                print(
                    f"OK: track_id={track_id} points={len(df)} dist={feats['distance_m']:.1f}m "
                    f"dur={feats['duration_s']} avg_speed={feats['avg_speed_mps']}"
                )
            elif i % 10 == 0:
                print(f"... processed {i}/{len(track_ids)}")

        except Exception as e:
            fail += 1
            print(f"ERROR: track_id={track_id} -> {type(e).__name__}: {e}")
            # важно: продолжаем дальше, чтобы не терять остальные треки
            continue

    print(f"DONE: ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

# python -m src.build_track_features