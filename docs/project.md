# ======================================================================================
# MLBOX_2 — ШПАРГАЛКА ДЛЯ JUPYTER (один файл/одна "записка")
# ======================================================================================
# Идея: ты открываешь ноутбук (например sql/db_inspection.ipynb) и просто идёшь по шагам.
# Все команды ниже можно копировать в ячейки Jupyter.
#
# Важно:
# 1) На соревнованиях почти всегда меняется ТОЛЬКО .env (доступы к PostgreSQL + схема).
# 2) Порядок пайплайна:
#      (A) init_db -> (B) ingest_gpx -> (C) build_track_features -> (D) enrich_context -> (E) проверки
# 3) Если ловишь 429/504 от Overpass — это НЕ ошибка кода, это лимиты API. Решение: sleep + шаг точек.
# ======================================================================================


PG_HOST=localhost
PG_PORT=55432
PG_DB=gpx_db
PG_USER=gpx_user
PG_PASSWORD=gpx_pass
PG_SCHEMA=public

содержание файла .env

# ======================================================================================
# 0) ПЕРЕД СТАРТОМ: ЧТО ГДЕ ЛЕЖИТ (быстро)
# ======================================================================================
# src/common/db.py               - единая точка подключения к PostgreSQL (читает .env)
# src/init_db.py                 - прогоняет sql/001_schema.sql + 002_indexes.sql + 003_views.sql
# src/ingest_gpx.py              - агент загрузки GPX -> data_sources / tracks / track_points
# src/build_track_features.py    - агрегаты по треку -> track_features
# src/enrich_context.py          - обогащение OSM+Weather -> context_time_series (jsonb values)
# sql/db_inspection.ipynb        - ноутбук для проверок и диагностики
# data/raw/gpx/                  - сюда кидаешь GPX (или скачанные открытые треки)


# ======================================================================================
# 1) НА СОРЕВНОВАНИЯХ: ЧТО МЕНЯТЬ
# ======================================================================================
# Меняешь ТОЛЬКО .env в корне проекта:
#   PG_HOST=...
#   PG_PORT=...
#   PG_DB=...
#   PG_USER=...
#   PG_PASSWORD=...
#   PG_SCHEMA=public   (или другая схема — иногда схема = логин)
#
# ⚠️ Никаких "public" в коде хардкодить не надо — schema берётся из .env через src/common/db.py


# ======================================================================================
# 2) АКТИВАЦИЯ ОКРУЖЕНИЯ / ЗАВИСИМОСТИ (если работаешь из терминала)
# ======================================================================================
# В Jupyter обычно venv уже выбран ядром. Если нет — делай в терминале:
#   source venv/bin/activate
#   pip install -r requirements.txt
#
# Внутри Jupyter можно проверить версию Python и пакеты:
import sys, platform
print("Python:", sys.version)
print("Platform:", platform.platform())


# ======================================================================================
# 3) ШАГ A: РАЗВЕРНУТЬ СТРУКТУРУ БД (таблицы/индексы/вьюхи)
# ======================================================================================
# 3.1 В терминале:
#   python -m src.init_db
#
# 3.2 В Jupyter можно так (если ноутбук открыт в корне проекта):
# (если команда не запускается, проверь текущую папку через: !pwd)
# !python -m src.init_db


# ======================================================================================
# 4) ШАГ B: INGEST — ЗАГРУЗИТЬ GPX -> (data_sources, tracks, track_points)
# ======================================================================================
# Кладёшь GPX в data/raw/gpx/
# Запуск:
#   python -m src.ingest_gpx --dir data/raw/gpx
# или из Jupyter:
# !python -m src.ingest_gpx --dir data/raw/gpx
#
# Повторный запуск должен быть ИДЕМПОТЕНТНЫМ (без дублей).


# ======================================================================================
# 5) ШАГ C: FEATURES — ПОСТРОИТЬ track_features
# ======================================================================================
# Запуск:
#   python -m src.build_track_features
# или из Jupyter:
# !python -m src.build_track_features
#
# Правила корректности:
# - нет time -> duration_s / avg_speed_mps / stop_time_s / stop_ratio = NULL (это нормально)
# - нет ele  -> elev_* = NULL (это нормально)


# ======================================================================================
# 6) ШАГ D: ENRICH — "что рядом" + погода -> context_time_series
# ======================================================================================
# Тестовый прогон (быстро и безопасно, чтобы не словить 429/504):
#   python -m src.enrich_context --limit-tracks 1 --point-step 300 --radius-m 200 --sleep-s 0.5 --only-missing
#
# Полный прогон (плотнее):
#   python -m src.enrich_context --point-step 50 --radius-m 200 --sleep-s 0.7 --only-missing
#
# Подсказки:
# - --point-step увеличивай, если Overpass ругается (меньше запросов)
# - --sleep-s увеличивай при 429/504 (даёшь API "дышать")
# - --only-missing = повторный запуск не переписывает уже обогащённые точки
#
# Важно:
# - если в треке нет time -> weather будет {ok:false, reason:"no_time"}, но nearby (OSM) заполнится
# - если Overpass временно недоступен -> часть точек пропустится (это норма для открытых API)
#
# Примеры запуска прямо в Jupyter:
# !python -m src.enrich_context --limit-tracks 1 --point-step 300 --radius-m 200 --sleep-s 0.5 --only-missing


# ======================================================================================
# 7) БЫСТРЫЕ ПРОВЕРКИ В JUPYTER (аудит после каждого шага)
# ======================================================================================
# Ниже — универсальные проверки через SQLAlchemy, используя твой src/common/db.py.
# Эти проверки не ломают данные, только читают.
#
# Если хочешь "тренироваться" с разными point-step и начинать с нуля —
# смотри в конце секцию 9 (очистка context_time_series).
from sqlalchemy import text
from src.common.db import get_engine, test_connection, set_search_path, DBConfig

engine = get_engine()
test_connection(verbose=True)

cfg = DBConfig.from_env()
schema = cfg.schema

def q(sql: str, **params):
    with engine.connect() as conn:
        set_search_path(conn)
        return conn.execute(text(sql), params).fetchall()

print("\n=== CHECK: объём основных таблиц ===")
rows = q("""
SELECT 'data_sources' AS tbl, COUNT(*)::bigint AS rows FROM data_sources
UNION ALL SELECT 'tracks', COUNT(*)::bigint FROM tracks
UNION ALL SELECT 'track_points', COUNT(*)::bigint FROM track_points
UNION ALL SELECT 'track_features', COUNT(*)::bigint FROM track_features
UNION ALL SELECT 'context_time_series', COUNT(*)::bigint FROM context_time_series
ORDER BY tbl;
""")
for r in rows:
    print(r)

print("\n=== CHECK: связность tracks -> track_points (нет ли точек без трека) ===")
res = q("""
SELECT
  (SELECT COUNT(*) FROM tracks)::bigint AS tracks,
  (SELECT COUNT(*) FROM track_points)::bigint AS points,
  (SELECT COUNT(*) FROM track_points p LEFT JOIN tracks t USING(track_id) WHERE t.track_id IS NULL)::bigint AS points_without_track;
""")
print(res[0])

print("\n=== CHECK: координаты (NULL/0/out-of-range) ===")
res = q("""
SELECT
  SUM(CASE WHEN lat IS NULL OR lon IS NULL THEN 1 ELSE 0 END)::bigint AS null_latlon,
  SUM(CASE WHEN lat=0 OR lon=0 THEN 1 ELSE 0 END)::bigint AS zero_latlon,
  SUM(CASE WHEN lat<-90 OR lat>90 THEN 1 ELSE 0 END)::bigint AS lat_out,
  SUM(CASE WHEN lon<-180 OR lon>180 THEN 1 ELSE 0 END)::bigint AS lon_out
FROM track_points;
""")
print(res[0])

print("\n=== CHECK: наличие времени в точках ===")
res = q("""
SELECT
  SUM(CASE WHEN time IS NOT NULL THEN 1 ELSE 0 END)::bigint AS with_time,
  SUM(CASE WHEN time IS NULL THEN 1 ELSE 0 END)::bigint AS without_time
FROM track_points;
""")
print(res[0])

print("\n=== CHECK: sanity track_features (покрытие) ===")
res = q("""
SELECT
  COUNT(*)::bigint AS tracks_total,
  SUM(CASE WHEN distance_m IS NOT NULL THEN 1 ELSE 0 END)::bigint AS with_distance,
  SUM(CASE WHEN duration_s IS NOT NULL THEN 1 ELSE 0 END)::bigint AS with_duration,
  SUM(CASE WHEN avg_speed_mps IS NOT NULL THEN 1 ELSE 0 END)::bigint AS with_avg_speed,
  SUM(CASE WHEN max_speed_mps IS NOT NULL THEN 1 ELSE 0 END)::bigint AS with_max_speed,
  SUM(CASE WHEN elev_min_m IS NOT NULL THEN 1 ELSE 0 END)::bigint AS with_elev_min,
  SUM(CASE WHEN elev_gain_m IS NOT NULL THEN 1 ELSE 0 END)::bigint AS with_elev_gain,
  SUM(CASE WHEN stop_time_s IS NOT NULL THEN 1 ELSE 0 END)::bigint AS with_stop_time,
  SUM(CASE WHEN stop_ratio IS NOT NULL THEN 1 ELSE 0 END)::bigint AS with_stop_ratio
FROM track_features;
""")
print(res[0])

print("\n=== CHECK: аномалии в track_features (подозрительные значения) ===")
anom = q("""
SELECT track_id, distance_m, duration_s, avg_speed_mps, stop_time_s, stop_ratio
FROM track_features
WHERE
  (duration_s IS NOT NULL AND duration_s < 0)
  OR (avg_speed_mps IS NOT NULL AND (avg_speed_mps < 0 OR avg_speed_mps > 25))
  OR (stop_ratio IS NOT NULL AND (stop_ratio < 0 OR stop_ratio > 1))
ORDER BY track_id
LIMIT 50;
""")
print("аномалий найдено:", len(anom))
for r in anom[:10]:
    print(r)

print("\n=== CHECK: краткий аудит context_time_series ===")
res = q("""
SELECT
  COUNT(*)::bigint AS rows,
  COUNT(DISTINCT source_id)::bigint AS sources,
  MIN(time) AS min_time,
  MAX(time) AS max_time
FROM context_time_series;
""")
print(res[0])

print("\n=== CHECK: какие ключи в values(jsonb) есть и сколько раз ===")
keys = q("""
SELECT key, COUNT(*)::bigint AS cnt
FROM (
  SELECT jsonb_object_keys(values) AS key
  FROM context_time_series
) t
GROUP BY key
ORDER BY cnt DESC, key;
""")
for r in keys:
    print(r)

print("\n=== CHECK: покрытие nearby/weather (сколько строк содержит) ===")
res = q("""
SELECT
  SUM(CASE WHEN values ? 'nearby' THEN 1 ELSE 0 END)::bigint AS with_nearby,
  SUM(CASE WHEN values ? 'weather' THEN 1 ELSE 0 END)::bigint AS with_weather,
  SUM(CASE WHEN (values->'weather'->>'ok')='true' THEN 1 ELSE 0 END)::bigint AS weather_ok,
  SUM(CASE WHEN (values->'weather'->>'ok')='false' THEN 1 ELSE 0 END)::bigint AS weather_fail
FROM context_time_series;
""")
print(res[0])

print("\n=== Последние 10 строк context_time_series (коротко) ===")
last = q("""
SELECT
  context_id, source_id, time, lat, lon,
  (values->'weather'->>'ok') AS weather_ok,
  (values->'weather'->>'temp_c') AS temp_c,
  (values->'nearby'->'counts') AS nearby_counts
FROM context_time_series
ORDER BY context_id DESC
LIMIT 10;
""")
for r in last:
    print(r)


# ======================================================================================
# 8) ТИПОВЫЕ ПРОБЛЕМЫ НА СОРЕВНОВАНИЯХ И КАК ЛЕЧИТЬ
# ======================================================================================
# [A] Не подключается к БД:
#   - проверь .env (host/port/db/user/password/schema)
#   - проверь, что сервер доступен и порт открыт
#
# [B] Таблицы не находятся:
#   - 99% проблема в PG_SCHEMA (не public)
#   - выставь PG_SCHEMA правильно
#
# [C] Overpass 429 Too Many Requests / 504 Gateway Timeout:
#   - увеличь --sleep-s (0.7..1.5)
#   - увеличь --point-step (100..300)
#   - уменьшай радиус --radius-m (150..250)
#   - запускай частями: --limit-tracks 3, потом ещё
#
# [D] Погода не заполняется:
#   - если нет time у трека -> weather ok:false reason:no_time (это нормально)
#   - если time есть, но сервис недоступен -> будет ok:false reason:...
#
# [E] “Хочу поменять point-step и пересчитать всё заново”
#   - или запускай без only-missing (перезатрёт значения по ключу)
#   - или очисти context_time_series полностью (см. ниже секцию 9)


# ======================================================================================
# 9) ОЧИСТКА context_time_series (если тренировался и хочешь начать заново)
# ======================================================================================
# ВАЖНО: Это удалит ВСЁ обогащение (погода+nearby) и ты будешь заполнять заново.
# Делай только если действительно нужно.
#
# Запуск в Jupyter:
def clear_context_time_series():
    with engine.begin() as conn:
        set_search_path(conn)
        conn.execute(text("TRUNCATE TABLE context_time_series RESTART IDENTITY;"))
    print("OK: context_time_series очищена полностью (TRUNCATE + reset id).")

# Пример:
# clear_context_time_series()


# ======================================================================================
# 10) РЕКОМЕНДУЕМЫЙ “БОЕВОЙ” ПЛАН НА СОРЕВНОВАНИЯХ (коротко)
# ======================================================================================
# 1) Вписать выданные доступы в .env
# 2) python -m src.init_db
# 3) Скинуть GPX в data/raw/gpx/
# 4) python -m src.ingest_gpx --dir data/raw/gpx
# 5) python -m src.build_track_features
# 6) python -m src.enrich_context --point-step 100 --radius-m 200 --sleep-s 1.0 --only-missing
# 7) Открыть ноутбук db_inspection.ipynb и прогнать проверки (из секции 7)
# 8) Если всё ок — уменьшать point-step (50) для более плотного контекста
# ======================================================================================




mlbox_2/
├─ app.py                      # точка входа (удобно для запуска “одной командой”)
├─ .env                        # твои локальные секреты (лучше НЕ пушить)
├─ .env.example                # шаблон переменных (в GitHub пушим)
├─ requirements.txt
├─ README.md

├─ src/
│  ├─ common/
│  │  ├─ db.py                 # DBConfig, get_connection, read_df, set_search_path
│  │  └─ utils.py              # to_dict/json safe helpers и т.п.
│  ├─ agent/
│  │  ├─ train_agent.py        # “агент”: собрать датасет → обучить 3+ моделей → выбрать лучшую → сохранить → записать model_runs
│  │  └─ schedule.py           # (опц.) запуск каждые 30 минут / защита от параллельного обучения
│  ├─ api/
│  │  ├─ main.py               # FastAPI/Flask приложение: /health /predict /predict_batch
│  │  └─ schemas.py            # Pydantic модели (если FastAPI)
│  └─ streamlit_app/
│     └─ dashboard.py          # UI, который зовёт API (НЕ обучает)

├─ notebooks/
│  ├─ module_b_update.ipynb    # разметка/кластеризация сегментов (как ты делал)
│  └─ module_c_train.ipynb     # эксперименты обучения (перед переносом в agent)

├─ models/                     # сюда сохраняем *.joblib (пушить можно, но лучше через релиз/артефакты)
├─ data/                       # (опц.) локальные файлы
└─ logs/                       # логи агента/апи