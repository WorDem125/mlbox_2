

"""mlbox_2 — GPX ingest (универсальный шаблон)

Зачем этот скрипт:
- Берёт все *.gpx из папки data/raw/gpx (или указанной через CLI)
- Грузит данные в PostgreSQL в 2 главные таблицы:
    1) tracks        — 1 строка = 1 маршрут/трек
    2) track_points  — много строк = точки этого маршрута
- Пишет источник файла в data_sources (1 строка = 1 файл)
- Повторный запуск НЕ создаёт дубли (по file_hash в data_sources)

Важно про универсальность:
- GPX бывают разные: с time/без time, с ele/без ele, с несколькими <trk>, иногда только <rte>.
- Этот шаблон НЕ ломается, если time/ele отсутствуют — просто кладёт NULL.

Что на соревнованиях почти всегда нужно менять:
1) .env (доступ к серверу): PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA
2) Папку с входными файлами (по умолчанию data/raw/gpx)
3) При желании: как формируем ext_track_id (внешний идентификатор маршрута)

Запуск:
- Из корня проекта:
    python -m src.ingest_gpx
- С параметрами:
    python -m src.ingest_gpx --dir data/raw/gpx --limit 10

Подсказка по проверке результата:
- SELECT COUNT(*) FROM public.tracks;
- SELECT COUNT(*) FROM public.track_points;

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import gpxpy
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values


# -----------------------------
# 0) Конфиг PostgreSQL из .env
# -----------------------------

@dataclass(frozen=True)
class PgConfig:
    host: str
    port: int
    db: str
    user: str
    password: str
    schema: str


def load_pg_config() -> PgConfig:
    """Читает параметры подключения из .env.

    На соревнованиях тебе просто дадут значения, ты вставишь их в .env.
    Код менять НЕ надо.
    """
    load_dotenv()
    return PgConfig(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        db=os.getenv("PG_DB", "postgres"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        schema=os.getenv("PG_SCHEMA", "public"),
    )


def connect(cfg: PgConfig):
    """Создаёт соединение с PostgreSQL."""
    return psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.db,
        user=cfg.user,
        password=cfg.password,
    )


def qname(schema: str, name: str) -> str:
    """Schema-qualified имя таблицы.

    Пример:
      qname('public', 'tracks') -> public."tracks"

    Почему кавычки:
    - безопаснее, если вдруг имя совпадёт с ключевым словом.
    """
    return f'{schema}."{name}"'


# -----------------------------
# 1) Входные файлы и file_hash
# -----------------------------

def sha256_file(path: Path) -> str:
    """Хэш файла для защиты от повторной загрузки (анти-дубликаты)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_gpx_files(folder: Path) -> Iterator[Path]:
    """Итерирует GPX файлы в папке."""
    yield from sorted(folder.glob("*.gpx"))


# -----------------------------
# 2) Парсинг GPX → набор треков
# -----------------------------

@dataclass
class PointRow:
    segment_index: int
    seq: int
    lat: float
    lon: float
    ele: Optional[float]
    time: Optional[datetime]
    # мы НЕ считаем speed/heading на этапе ingest — это позже (features/analytics)


@dataclass
class TrackParsed:
    track_name: str
    track_type: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    segment_count: int
    point_count: int
    min_lat: Optional[float]
    min_lon: Optional[float]
    max_lat: Optional[float]
    max_lon: Optional[float]
    points: list[PointRow]


def parse_gpx_file(path: Path) -> list[TrackParsed]:
    """Парсит GPX и возвращает список треков.

    Универсальность:
    - если в GPX несколько <trk> → вернём несколько TrackParsed
    - если <trk> нет, но есть <rte> → считаем route как трек (без segment-ов)
    - если нет time/ele → будут None

    Важно:
    - здесь мы НЕ считаем distance/скорость и т.п. — это отдельный этап (track_features).
    """

    # Иногда файлы бывают не UTF-8 — пробуем мягко.
    raw_text: str
    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = path.read_text(encoding="latin-1")

    try:
        gpx = gpxpy.parse(raw_text)
    except Exception:
        # GPX не парсится — считаем файл бесполезным (например, битый)
        return []

    out: list[TrackParsed] = []

    # --- 2.1) <trk> (обычная запись трекером) ---
    for trk in gpx.tracks:
        name = trk.name or path.stem
        track_type = "unknown"  # на соревнованиях редко дают явный тип

        points: list[PointRow] = []
        segment_count = len(trk.segments)

        min_lat = min_lon = max_lat = max_lon = None
        start_time = end_time = None

        for seg_idx, seg in enumerate(trk.segments):
            for seq_idx, pt in enumerate(seg.points, start=1):
                lat = float(pt.latitude)
                lon = float(pt.longitude)
                ele = None if pt.elevation is None else float(pt.elevation)
                tm = pt.time  # может быть None

                if min_lat is None:
                    min_lat = max_lat = lat
                    min_lon = max_lon = lon
                else:
                    min_lat = min(min_lat, lat)
                    max_lat = max(max_lat, lat)
                    min_lon = min(min_lon, lon)
                    max_lon = max(max_lon, lon)

                if tm is not None:
                    if start_time is None or tm < start_time:
                        start_time = tm
                    if end_time is None or tm > end_time:
                        end_time = tm

                points.append(
                    PointRow(
                        segment_index=seg_idx,
                        seq=seq_idx,
                        lat=lat,
                        lon=lon,
                        ele=ele,
                        time=tm,
                    )
                )

        out.append(
            TrackParsed(
                track_name=name,
                track_type=track_type,
                start_time=start_time,
                end_time=end_time,
                segment_count=segment_count,
                point_count=len(points),
                min_lat=min_lat,
                min_lon=min_lon,
                max_lat=max_lat,
                max_lon=max_lon,
                points=points,
            )
        )

    # --- 2.2) <rte> (иногда GPX содержит только route) ---
    if not out and gpx.routes:
        for rte in gpx.routes:
            name = rte.name or path.stem
            points: list[PointRow] = []
            min_lat = min_lon = max_lat = max_lon = None

            for seq_idx, pt in enumerate(rte.points, start=1):
                lat = float(pt.latitude)
                lon = float(pt.longitude)
                ele = None if pt.elevation is None else float(pt.elevation)

                if min_lat is None:
                    min_lat = max_lat = lat
                    min_lon = max_lon = lon
                else:
                    min_lat = min(min_lat, lat)
                    max_lat = max(max_lat, lat)
                    min_lon = min(min_lon, lon)
                    max_lon = max(max_lon, lon)

                points.append(
                    PointRow(
                        segment_index=0,
                        seq=seq_idx,
                        lat=lat,
                        lon=lon,
                        ele=ele,
                        time=None,  # в rte обычно нет времени
                    )
                )

            out.append(
                TrackParsed(
                    track_name=name,
                    track_type="unknown",
                    start_time=None,
                    end_time=None,
                    segment_count=1,
                    point_count=len(points),
                    min_lat=min_lat,
                    min_lon=min_lon,
                    max_lat=max_lat,
                    max_lon=max_lon,
                    points=points,
                )
            )

    return out


# -----------------------------
# 3) Запись в БД (3 таблицы)
# -----------------------------

def upsert_data_source(conn, cfg: PgConfig, *, source_name: str, source_path: str, file_hash: str) -> Tuple[int, bool]:
    """Создаёт запись в data_sources, либо возвращает существующую.

    Возвращает:
      (source_id, is_new)

    Универсальный приём для соревнований:
    - защита от дублей делается именно на уровне data_sources.file_hash
    """
    received_at = datetime.now(timezone.utc)

    sql_insert = f"""
        INSERT INTO {qname(cfg.schema, 'data_sources')}
            (source_type, source_name, source_path, file_hash, received_at, extra)
        VALUES
            (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (file_hash)
        DO NOTHING
        RETURNING source_id
    """

    with conn.cursor() as cur:
        cur.execute(
            sql_insert,
            (
                "gpx",
                source_name,
                source_path,
                file_hash,
                received_at,
                # всегда JSON-объект с начальным payload
                json.dumps({"status": "new", "notes": "ingest started"}),
            ),
        )
        row = cur.fetchone()
        if row:
            return int(row[0]), True

        # если не вставилось — значит уже есть
        cur.execute(
            f"SELECT source_id FROM {qname(cfg.schema,'data_sources')} WHERE file_hash = %s",
            (file_hash,),
        )
        source_id = int(cur.fetchone()[0])
        return source_id, False


# Универсальный patcher для data_sources.extra (jsonb)
def update_data_source_extra(conn, cfg: PgConfig, *, source_id: int, patch: dict) -> None:
    """Merges fields into data_sources.extra (jsonb).

    Это нужно, чтобы:
    - помечать файлы как skipped/error/ok
    - сохранять причину пропуска
    - сохранять статистику (сколько треков/точек)

    На соревнованиях это удобно: быстро видно, какие файлы были непригодны.
    """
    sql = f"""
        UPDATE {qname(cfg.schema,'data_sources')}
        SET extra = COALESCE(extra, '{{}}'::jsonb) || %s::jsonb
        WHERE source_id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (json.dumps(patch), source_id))


def insert_track(conn, cfg: PgConfig, *, source_id: int, ext_track_id: str, tr: TrackParsed) -> int:
    """Создаёт 1 запись в tracks и возвращает track_id."""

    # В нашей схеме есть ext_object_id/ext_driver_id — для GPX обычно неизвестны → NULL
    sql = f"""
        INSERT INTO {qname(cfg.schema, 'tracks')}
            (source_id, ext_track_id, ext_object_id, ext_driver_id,
             track_name, track_type,
             start_time, end_time,
             segment_count, point_count,
             min_lat, min_lon, max_lat, max_lon,
             extra)
        VALUES
            (%s, %s, %s, %s,
             %s, %s,
             %s, %s,
             %s, %s,
             %s, %s, %s, %s,
             %s)
        RETURNING track_id
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                source_id,
                ext_track_id,
                None,
                None,
                tr.track_name,
                tr.track_type,
                tr.start_time,
                tr.end_time,
                tr.segment_count,
                tr.point_count,
                tr.min_lat,
                tr.min_lon,
                tr.max_lat,
                tr.max_lon,
                json.dumps({
                    # полезная метка: что было в файле
                    "has_time": tr.start_time is not None,
                    "has_ele": any(p.ele is not None for p in tr.points),
                }),
            ),
        )
        return int(cur.fetchone()[0])


def insert_track_points(conn, cfg: PgConfig, *, track_id: int, points: list[PointRow], batch_size: int = 5000) -> int:
    """Вставляет точки в track_points пакетно (быстро) и возвращает количество вставленных."""

    if not points:
        return 0

    # В нашей таблице есть x/y/speed/heading/extra.
    # На ingest мы их не считаем → NULL, extra всегда пустой JSON (никогда NULL)
    values = [
        (
            track_id,
            p.seq,
            p.segment_index,
            p.lat,
            p.lon,
            None,   # x
            None,   # y
            p.ele,
            p.time,
            None,   # speed
            None,   # heading
            json.dumps({}),   # extra всегда JSON, не NULL
        )
        for p in points
    ]

    sql = f"""
        INSERT INTO {qname(cfg.schema, 'track_points')}
            (track_id, seq, segment_index, lat, lon, x, y, ele, time, speed, heading, extra)
        VALUES %s
        ON CONFLICT (track_id, seq, segment_index) DO NOTHING
    """

    inserted = 0
    with conn.cursor() as cur:
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            execute_values(cur, sql, batch, page_size=len(batch))
            inserted += len(batch)

    return inserted


# -----------------------------
# 4) Ingest папки: файл → source → треки → точки
# -----------------------------

def ingest_folder(cfg: PgConfig, folder: Path, *, limit: Optional[int] = None, dry_run: bool = False) -> None:
    """Основной сценарий: прогоняем папку и грузим в БД."""

    if not folder.exists():
        raise FileNotFoundError(f"GPX folder not found: {folder.resolve()}")

    files = list(iter_gpx_files(folder))
    if limit is not None:
        files = files[:limit]

    print(f"FOUND: {len(files)} GPX files in {folder}")

    with connect(cfg) as conn:
        conn.autocommit = False

        for fp in files:
            source_id: Optional[int] = None  # определим заранее, чтобы использовать в except
            try:
                file_hash = sha256_file(fp)

                if dry_run:
                    print(f"DRY-RUN: {fp.name} | hash={file_hash[:10]}...")
                    continue

                source_id, is_new = upsert_data_source(
                    conn,
                    cfg,
                    source_name=fp.name,
                    source_path=str(fp.resolve()),
                    file_hash=file_hash,
                )

                if not is_new:
                    print(f"SKIP (already loaded): {fp.name}")
                    # Обновим статус в extra, чтобы видеть в UI/SQL
                    update_data_source_extra(conn, cfg, source_id=source_id, patch={
                        "status": "skipped",
                        "reason": "already_loaded"
                    })
                    conn.commit()
                    continue

                tracks = parse_gpx_file(fp)
                if not tracks:
                    update_data_source_extra(conn, cfg, source_id=source_id, patch={
                        "status": "skipped",
                        "reason": "no_trk_or_rte",
                        "file": fp.name
                    })
                    print(f"WARN: no <trk> or <rte> found: {fp.name} (source_id={source_id})")
                    conn.commit()
                    continue

                total_points = 0
                for idx, tr in enumerate(tracks, start=1):
                    # Универсальный внешний id (удобно для дебага/сводок)
                    # На соревнованиях можно поменять формат строки.
                    ext_track_id = f"{fp.stem}#{idx}"

                    track_id = insert_track(conn, cfg, source_id=source_id, ext_track_id=ext_track_id, tr=tr)
                    pts = insert_track_points(conn, cfg, track_id=track_id, points=tr.points)
                    total_points += pts
                    print(f"OK: {fp.name} -> track_id={track_id} points={pts}")

                # После успешной загрузки обновляем summary в extra
                update_data_source_extra(conn, cfg, source_id=source_id, patch={
                    "status": "ok",
                    "tracks": len(tracks),
                    "points": total_points,
                    "ingested_at": datetime.now(timezone.utc).isoformat()
                })
                conn.commit()

            except Exception as e:
                conn.rollback()
                print(f"ERROR: {fp.name} -> {e}")
                # Best-effort update: пометить ошибку в data_sources.extra
                if source_id is not None:
                    try:
                        with connect(cfg) as conn2:
                            update_data_source_extra(conn2, cfg, source_id=source_id, patch={
                                "status": "error",
                                "error": str(e),
                                "file": fp.name
                            })
                            conn2.commit()
                    except Exception:
                        pass  # даже если не получилось — ничего страшного


# -----------------------------
# 5) CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest GPX files into PostgreSQL")
    p.add_argument("--dir", type=str, default="data/raw/gpx", help="Folder with .gpx files")
    p.add_argument("--limit", type=int, default=None, help="Process only first N files")
    p.add_argument("--dry-run", action="store_true", help="Do not write to DB, only print actions")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_pg_config()

    folder = Path(args.dir)
    ingest_folder(cfg, folder, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

#python -m src.ingest_gpx --dir data/raw/gpx
#Визуально посотреть 
#python -m src.ingest_gpx --dir data/raw/gpx --dry-run