"""src.enrich_context

Универсальное обогащение (enrichment) GPX‑треков данными из открытых источников.

Что делает скрипт
-----------------
1) Берёт точки маршрута из БД (таблица track_points).
2) Для части точек (downsample) запрашивает:
   • «что рядом» по OpenStreetMap через Overpass API
   • «погоду» (температуру) по координатам+дате через Open‑Meteo (archive API)
3) Складывает результат в одну таблицу context_time_series:
   (context_id, source_id, time, lat, lon, values)
   где values — JSON (jsonb) со структурой {track_id, point, nearby, weather}.

Где берём данные (открытые источники)
------------------------------------
• OpenStreetMap (OSM): используем Overpass API — запросы «в радиусе N метров от точки».
• Погода: используем Open‑Meteo Archive API (без ключа) — temperature_2m по дате.

Важно про API
-------------
Да, скрипт ходит в интернет по HTTP (REST API). Тебе НЕ нужно ничего сверх дописывать:
просто запускаешь этот файл. Если на соревновании попросят другой источник — обычно
достаточно заменить 2 функции: `overpass_nearby()` и `open_meteo_temp()`.

Идемпотентность / повторные запуски
----------------------------------
Добавляем UNIQUE ключ (source_id, time, lat, lon) и пишем через UPSERT (ON CONFLICT).
Повторный запуск НЕ создаёт дубликаты, а аккуратно «дольёт» новые поля в JSON.

Как запускать
-------------
export PG_SCHEMA=public  # или ваша схема на соревнованиях
python -m src.enrich_context --radius-m 250 --point-step 50 --only-missing

# также поддерживается переменная DB_SCHEMA (фолбэк, если кто-то уже привык)

# или явно через CLI (перекрывает env)
python -m src.enrich_context --schema public --radius-m 250 --point-step 50 --only-missing

Подсказка по параметрам
-----------------------
• --point-step 50  : берём каждую 50‑ю точку (экономит запросы)
• --radius-m 250   : радиус поиска объектов в OSM
• --only-missing   : пропускаем треки, которые уже обогащены (есть записи в context_time_series)

"""

from __future__ import annotations

import argparse
import json
import random
import time as time_mod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
import traceback

import requests
from sqlalchemy import text

# Используем уже существующие модули подключения к БД (у тебя они уже есть в проекте)
from src.common.db import get_engine, test_connection


# Публичные эндпоинты (открытые источники)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@dataclass
class EnrichConfig:
    """Параметры работы enrichment."""

    schema: str
    radius_m: int = 250
    point_step: int = 50
    only_missing: bool = True
    limit_tracks: Optional[int] = None
    track_id: Optional[int] = None

    # «бережность» к публичным API
    sleep_s: float = 0.15
    timeout_s: float = 30.0
    max_retries: int = 4
    verbose_errors: bool = True
    fail_fast: bool = False


# --------------------------------------------------------------------------------------
# Helper: получение схемы из env / CLI (совместимо с src.common.db)
# --------------------------------------------------------------------------------------

import os


def get_schema_from_env(default: str = "public") -> str:
    """Берём имя схемы из переменных окружения.

    В проекте `src.common.db` схема читается из `PG_SCHEMA` (см. .env).
    Чтобы все модули работали одинаково, здесь делаем совместимую логику:

    Приоритет:
      1) PG_SCHEMA  (основной вариант в проекте)
      2) DB_SCHEMA  (фолбэк, если кто-то уже привык)
      3) default    (обычно public)
    """
    for key in ("PG_SCHEMA", "DB_SCHEMA"):
        s = os.getenv(key)
        if s and str(s).strip():
            return str(s).strip()
    return default


# --------------------------------------------------------------------------------------
# Вспомогательные утилиты
# --------------------------------------------------------------------------------------

def _utc(dt: datetime) -> datetime:
    """Гарантируем, что datetime в UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _round_coord(lat: float, lon: float, decimals: int = 4) -> Tuple[float, float]:
    # Округление для кеша: 4 знака ≈ ~11 м по широте. Сильно снижает число повторных запросов.
    return (round(lat, decimals), round(lon, decimals))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _requests_get_json(url: str, params: Dict[str, Any], *, timeout_s: float, max_retries: int) -> Dict[str, Any]:
    """GET JSON с повторами (retry) и backoff."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout_s,
                headers={
                    "User-Agent": "mlbox_2-gpx-enricher/1.0 (https://openstreetmap.org; open-meteo.com)"
                },
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            # backoff + небольшой jitter
            sleep = (2 ** attempt) * 0.6 + random.random() * 0.2
            time_mod.sleep(sleep)
    raise RuntimeError(f"HTTP ошибка после {max_retries} попыток: {url} | last={last_err}")


def _requests_post_text(url: str, data: str, *, timeout_s: float, max_retries: int) -> str:
    """POST текст (Overpass) с повторами (retry) и backoff."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                url,
                data=data.encode("utf-8"),
                timeout=timeout_s,
                headers={
                    "User-Agent": "mlbox_2-gpx-enricher/1.0 (https://openstreetmap.org; open-meteo.com)",
                    "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                },
            )
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            sleep = (2 ** attempt) * 0.6 + random.random() * 0.2
            time_mod.sleep(sleep)
    raise RuntimeError(f"HTTP ошибка после {max_retries} попыток: {url} | last={last_err}")


# --------------------------------------------------------------------------------------
# DB: подготовка и чтение
# --------------------------------------------------------------------------------------

def ensure_context_constraints(engine, schema: str) -> None:
    """Подготовка таблицы context_time_series для безопасных повторных запусков.

    Идея: мы храним и «nearby» и «weather» в одном JSON для ключа (source_id,time,lat,lon).
    Тогда можно поставить UNIQUE и делать UPSERT.
    """
    sql = f"""
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_context_time_series_key'
          AND conrelid = '"{schema}"."context_time_series"'::regclass
      ) THEN
        ALTER TABLE "{schema}"."context_time_series"
          ADD CONSTRAINT uq_context_time_series_key
          UNIQUE (source_id, time, lat, lon);
      END IF;
    END $$;

    ALTER TABLE "{schema}"."context_time_series"
      ALTER COLUMN values SET DEFAULT '{{}}'::jsonb;

    CREATE INDEX IF NOT EXISTS ix_context_time_series_source_time
      ON "{schema}"."context_time_series"(source_id, time);

    CREATE INDEX IF NOT EXISTS ix_context_time_series_lat_lon
      ON "{schema}"."context_time_series"(lat, lon);
    """
    with engine.begin() as conn:
        conn.execute(text(sql))


def iter_tracks(engine, schema: str, *, limit: Optional[int] = None, track_id: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    where = ""
    params: Dict[str, Any] = {}
    if track_id is not None:
        where = "WHERE t.track_id = :track_id"
        params["track_id"] = int(track_id)

    lim = ""
    if limit is not None:
        lim = "LIMIT :limit"
        params["limit"] = int(limit)

    q = f"""
    SELECT
      t.track_id,
      t.source_id,
      t.track_name,
      t.start_time,
      t.end_time,
      t.created_at
    FROM "{schema}"."tracks" t
    {where}
    ORDER BY t.track_id
    {lim}
    """

    with engine.connect() as conn:
        rows = conn.execute(text(q), params).mappings().all()
    for r in rows:
        yield dict(r)


def fetch_sample_points(engine, schema: str, track_id: int, point_step: int) -> List[Dict[str, Any]]:
    """Берём подвыборку точек: каждую N‑ю по seq."""
    # ВАЖНО:
    # В разных источниках `seq` может начинаться с 1 и быть небольшим (например, 45 точек).
    # Тогда условие `seq % step = 0` при большом step (например, 300) вернёт 0 строк.
    # Для универсальности гарантируем минимум 1 точку: всегда берём первую точку трека.

    step = max(1, int(point_step))

    q = f"""
    SELECT track_id, segment_index, seq, lat, lon, ele, time
    FROM "{schema}"."track_points"
    WHERE track_id = :track_id
      AND (
        seq = (
          SELECT MIN(seq)
          FROM "{schema}"."track_points"
          WHERE track_id = :track_id
        )
        OR (seq % :step = 0)
      )
    ORDER BY segment_index, seq
    """

    with engine.connect() as conn:
        pts = conn.execute(text(q), {"track_id": int(track_id), "step": step}).mappings().all()

    return [dict(p) for p in pts]


def context_exists(engine, schema: str, source_id: int) -> bool:
    """Проверяем, есть ли уже хоть одна запись контекста для этого source_id."""
    q = f"""
    SELECT 1
    FROM "{schema}"."context_time_series"
    WHERE source_id = :source_id
    LIMIT 1
    """
    with engine.connect() as conn:
        r = conn.execute(text(q), {"source_id": int(source_id)}).first()
    return r is not None


def upsert_context_row(
    engine,
    schema: str,
    *,
    source_id: int,
    time: datetime,
    lat: float,
    lon: float,
    values: Dict[str, Any],
) -> None:
    """UPSERT: вставить или обновить (склеить JSON).

    ВАЖНО:
    - Не используем конструкцию `:values::jsonb` (PostgreSQL ругается на двоеточие).
    - Передаём JSON как строку; PostgreSQL сам приведёт её к jsonb, так как колонка `values` имеет тип jsonb.
    """

    q = f"""
    INSERT INTO "{schema}"."context_time_series"(source_id, time, lat, lon, values)
    VALUES (:source_id, :time, :lat, :lon, :values)
    ON CONFLICT (source_id, time, lat, lon)
    DO UPDATE SET values = "{schema}"."context_time_series".values || EXCLUDED.values;
    """

    with engine.begin() as conn:
        conn.execute(
            text(q),
            {
                "source_id": int(source_id),
                "time": _utc(time),
                "lat": float(lat),
                "lon": float(lon),
                "values": json.dumps(values, ensure_ascii=False),
            },
        )


# --------------------------------------------------------------------------------------
# Enrichment: OSM / Overpass
# --------------------------------------------------------------------------------------

def overpass_nearby(
    lat: float,
    lon: float,
    radius_m: int,
    *,
    timeout_s: float,
    max_retries: int,
) -> Dict[str, Any]:
    """Запрос Overpass вокруг точки и возврат компактной сводки."""

    # Запрашиваем ограниченный набор тегов, чтобы ответ был небольшим.
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius_m},{lat},{lon})["natural"];
      way(around:{radius_m},{lat},{lon})["natural"];
      relation(around:{radius_m},{lat},{lon})["natural"];

      node(around:{radius_m},{lat},{lon})["water"];
      way(around:{radius_m},{lat},{lon})["water"];

      node(around:{radius_m},{lat},{lon})["waterway"];
      way(around:{radius_m},{lat},{lon})["waterway"];

      node(around:{radius_m},{lat},{lon})["landuse"];
      way(around:{radius_m},{lat},{lon})["landuse"];

      node(around:{radius_m},{lat},{lon})["building"];
      way(around:{radius_m},{lat},{lon})["building"];

      node(around:{radius_m},{lat},{lon})["highway"];
      way(around:{radius_m},{lat},{lon})["highway"];

      node(around:{radius_m},{lat},{lon})["amenity"];
      way(around:{radius_m},{lat},{lon})["amenity"];

      node(around:{radius_m},{lat},{lon})["leisure"];
      way(around:{radius_m},{lat},{lon})["leisure"];
    );
    out tags center qt;
    """.strip()

    raw = _requests_post_text(OVERPASS_URL, query, timeout_s=timeout_s, max_retries=max_retries)
    data = json.loads(raw)

    counts: Dict[str, Dict[str, int]] = {}
    examples: List[Dict[str, Any]] = []

    for el in data.get("elements", [])[:500]:
        tags = el.get("tags") or {}
        # Берём первый «главный» тег из списка
        for key in ("natural", "water", "waterway", "landuse", "building", "highway", "amenity", "leisure"):
            if key in tags:
                val = str(tags.get(key))
                counts.setdefault(key, {})
                counts[key][val] = counts[key].get(val, 0) + 1
                # Пару примеров для дашборда/отчёта
                if len(examples) < 12:
                    examples.append({"tag": key, "value": val, "name": tags.get("name")})
                break

    # Сжимаем статистику: оставляем TOP‑5 значений для каждого тега.
    compact_counts: Dict[str, List[Tuple[str, int]]] = {}
    for k, d in counts.items():
        top = sorted(d.items(), key=lambda x: x[1], reverse=True)[:5]
        compact_counts[k] = top

    return {"radius_m": int(radius_m), "counts": compact_counts, "examples": examples}


# --------------------------------------------------------------------------------------
# Enrichment: Weather (Open‑Meteo)
# --------------------------------------------------------------------------------------

def open_meteo_temp(
    lat: float,
    lon: float,
    t: datetime,
    *,
    timeout_s: float,
    max_retries: int,
) -> Dict[str, Any]:
    """Температура (temperature_2m) для ближайшего часа к времени t."""

    t = _utc(t)
    date_str = t.date().isoformat()

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }

    data = _requests_get_json(OPEN_METEO_ARCHIVE_URL, params, timeout_s=timeout_s, max_retries=max_retries)
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []

    if not times or not temps or len(times) != len(temps):
        return {"ok": False, "reason": "нет hourly temperature"}

    # Выбираем ближайший час к timestamp.
    target = t.replace(minute=0, second=0, microsecond=0)

    best_i = 0
    best_dt: Optional[datetime] = None
    for i, ts in enumerate(times):
        try:
            dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if best_dt is None or abs((dt - target).total_seconds()) < abs((best_dt - target).total_seconds()):
            best_dt = dt
            best_i = i

    temp = _safe_float(temps[best_i])
    return {"ok": True, "temp_c": temp, "hour_utc": times[best_i], "source": "open-meteo-archive"}


# --------------------------------------------------------------------------------------
# Главный пайплайн
# --------------------------------------------------------------------------------------

def enrich_track(engine, cfg: EnrichConfig, track: Dict[str, Any]) -> Tuple[int, int, int]:
    """Обогащаем один трек. Возвращаем (добавлено_строк, пропущено_источников, ошибок)."""

    track_id = int(track["track_id"])
    source_id = int(track["source_id"])

    # Если only_missing и по этому source_id уже есть контекст — пропускаем.
    if cfg.only_missing and context_exists(engine, cfg.schema, source_id):
        return (0, 1, 0)

    pts = fetch_sample_points(engine, cfg.schema, track_id, cfg.point_step)
    if not pts:
        return (0, 0, 1)

    # Время может отсутствовать в GPX (нет <time> в точках).
    # Для БД нам всё равно нужен стабильный timestamp-ключ: берём start_time трека,
    # иначе created_at, иначе текущее время.
    fallback_time: datetime = track.get("start_time") or track.get("created_at") or datetime.now(timezone.utc)
    fallback_time = _utc(fallback_time)

    # Если у трека нет start_time и у точки нет time — погоду корректно получить нельзя.
    # В таком случае пишем только nearby, а weather помечаем как недоступный.
    track_has_time = track.get("start_time") is not None

    ok = 0
    fail = 0

    # Кеш в рамках одного запуска (чтобы не дергать API по одинаковым координатам).
    osm_cache: Dict[Tuple[float, float, int], Dict[str, Any]] = {}
    wx_cache: Dict[Tuple[float, float, str], Dict[str, Any]] = {}

    for p in pts:
        lat = float(p["lat"])
        lon = float(p["lon"])

        # `t_db` — время, которое мы пишем в БД (стабильный ключ).
        # `t_weather` — время для запроса погоды (только если реально есть время трека/точки).
        raw_t = p.get("time")

        t_db: datetime = fallback_time
        t_weather: Optional[datetime] = None

        if raw_t is not None:
            # В точке есть время — используем его и для БД, и для погоды.
            t_db = raw_t
        elif track_has_time:
            # В точке времени нет, но есть start_time у трека — для погоды используем start_time.
            t_db = fallback_time
        else:
            # Времени нет вообще — погоду не запрашиваем.
            t_db = fallback_time

        # Парсим строку времени, если нужно
        if isinstance(t_db, str):
            try:
                t_db = datetime.fromisoformat(t_db)
            except Exception:
                t_db = fallback_time

        if isinstance(t_db, datetime):
            t_db = _utc(t_db)
        else:
            t_db = fallback_time

        # Определяем время для погоды
        if raw_t is not None:
            t_weather = t_db
        elif track_has_time:
            t_weather = fallback_time
        else:
            t_weather = None

        # ключи кеша
        rlat, rlon = _round_coord(lat, lon, decimals=4)
        osm_key = (rlat, rlon, int(cfg.radius_m))

        try:
            # OSM nearby
            if osm_key in osm_cache:
                nearby = osm_cache[osm_key]
            else:
                nearby = overpass_nearby(lat, lon, cfg.radius_m, timeout_s=cfg.timeout_s, max_retries=cfg.max_retries)
                osm_cache[osm_key] = nearby

            # Weather (на день) — запрашиваем ТОЛЬКО если можем корректно определить дату.
            if t_weather is None:
                weather = {"ok": False, "reason": "no_time"}
            else:
                day_key = t_weather.date().isoformat()
                wx_key = (rlat, rlon, day_key)
                if wx_key in wx_cache:
                    weather = wx_cache[wx_key]
                else:
                    weather = open_meteo_temp(lat, lon, t_weather, timeout_s=cfg.timeout_s, max_retries=cfg.max_retries)
                    wx_cache[wx_key] = weather

            values = {
                "track_id": track_id,
                "point": {
                    "segment_index": int(p.get("segment_index") or 0),
                    "seq": int(p.get("seq") or 0),
                    "ele_m": _safe_float(p.get("ele")),
                },
                "nearby": nearby,
                "weather": weather,
            }

            upsert_context_row(engine, cfg.schema, source_id=source_id, time=t_db, lat=lat, lon=lon, values=values)
            ok += 1

            # Бережно к публичным сервисам: небольшая пауза между запросами.
            if cfg.sleep_s > 0:
                time_mod.sleep(cfg.sleep_s)

        except Exception as e:
            fail += 1
            if cfg.verbose_errors:
                print(
                    "WARN: enrich point failed | "
                    f"track_id={track_id} source_id={source_id} "
                    f"lat={lat} lon={lon} time={t_db} | err={type(e).__name__}: {e}"
                )
                # Печатаем короткий traceback, чтобы сразу было понятно где упало
                tb = traceback.format_exc(limit=3)
                print(tb)
            if cfg.fail_fast:
                raise

    return (ok, 0, fail)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich: OSM nearby + weather -> context_time_series"
    )

    # Схему НЕ заставляем задавать в CLI: берём из .env через src.common.db (PG_SCHEMA).
    # Но если нужно — можно явно указать.
    parser.add_argument(
        "--schema",
        default=None,
        help="Схема БД. По умолчанию берётся из .env (PG_SCHEMA), иначе public.",
    )

    parser.add_argument(
        "--radius-m",
        type=int,
        default=250,
        help="Радиус поиска объектов (Overpass) в метрах",
    )
    parser.add_argument(
        "--point-step",
        type=int,
        default=50,
        help="Использовать каждую N-ю точку (seq % N == 0)",
    )

    # По умолчанию работаем безопасно: НЕ трогаем то, что уже обогащено.
    parser.add_argument(
        "--only-missing",
        action="store_true",
        default=True,
        help="Пропускать источники, которые уже обогащены (по умолчанию включено)",
    )
    parser.add_argument(
        "--all",
        dest="only_missing",
        action="store_false",
        help="Обрабатывать даже если уже есть контекст",
    )

    parser.add_argument(
        "--limit-tracks",
        type=int,
        default=None,
        help="Обработать только первые N треков",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Обработать только один track_id",
    )

    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.15,
        help="Пауза между запросами к публичным API",
    )

    # Логи/отладка
    parser.add_argument(
        "--quiet-errors",
        action="store_true",
        help="Не печатать детали ошибок по точкам",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Остановиться на первой ошибке (для отладки)",
    )

    args = parser.parse_args()

    # Проверяем, что подключение к БД корректно (использует настройки проекта)
    test_connection(verbose=True)

    # Схема берётся из .env (PG_SCHEMA). CLI --schema только если явно указали.
    schema = args.schema or get_schema_from_env()
    cfg = EnrichConfig(
        schema=schema,
        radius_m=int(args.radius_m),
        point_step=int(args.point_step),
        only_missing=bool(args.only_missing),
        limit_tracks=args.limit_tracks,
        track_id=args.track_id,
        sleep_s=float(args.sleep_s),
        verbose_errors=not bool(args.quiet_errors),
        fail_fast=bool(args.fail_fast),
    )

    engine = get_engine()

    print(
        f"ENRICH: схема={cfg.schema} радиус={cfg.radius_m}м шаг_точек={cfg.point_step} "
        f"only_missing={cfg.only_missing}"
    )

    ensure_context_constraints(engine, cfg.schema)

    tracks = list(iter_tracks(engine, cfg.schema, limit=cfg.limit_tracks, track_id=cfg.track_id))
    print(f"FOUND: треков={len(tracks)}")

    ok_total = 0
    skip_total = 0
    fail_total = 0

    for i, tr in enumerate(tracks, start=1):
        ok, skipped, fail = enrich_track(engine, cfg, tr)
        ok_total += ok
        skip_total += skipped
        fail_total += fail

        if cfg.limit_tracks is not None or len(tracks) <= 5:
            print(f"TRACK {i}/{len(tracks)} | track_id={tr['track_id']} source_id={tr['source_id']} -> ok_points={ok} skipped_sources={skipped} fail_points={fail}")

        if i % 5 == 0 or i == len(tracks):
            print(
                f"... обработано {i}/{len(tracks)} | добавлено_строк={ok_total} "
                f"пропущено_источников={skip_total} ошибок={fail_total}"
            )

    print(f"DONE: добавлено_строк={ok_total} пропущено_источников={skip_total} ошибок={fail_total}")


if __name__ == "__main__":
    main()


#python -m src.enrich_context --limit-tracks 7 --point-step 300  --radius-m 200 --sleep-s 