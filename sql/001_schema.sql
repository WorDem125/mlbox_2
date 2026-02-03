

-- sql/001_schema.sql
-- ==============================================================
-- Универсальная схема БД для соревнований (трековые/телеметрические данные)
-- ==============================================================
-- Цель: один раз создать «каркас» таблиц, который подходит под разные входные
-- форматы (GPX-треки, телеметрия датчиков, любые time-series + координаты).
--
-- Как это использовать на соревнованиях:
-- 1) Вам выдают доступы к PostgreSQL (host/port/db/user/password, иногда schema).
-- 2) Вы меняете ТОЛЬКО .env (PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASSWORD/PG_SCHEMA).
-- 3) Запускаете init_db (который выполнит этот SQL).
-- 4) Потом скрипт загрузки данных (ingest) наполняет таблицы.
--
-- Что возможно придется менять на соревнованиях:
-- - НИЧЕГО, если вы используете универсальные поля lat/lon/time и кладете нестандартные
--   атрибуты в JSONB `extra`.
-- - Если эксперты требуют хранить дополнительные атрибуты отдельными колонками
--   (например, speed/height/engine_speed), вы можете ДОБАВИТЬ колонки в track_points
--   или track_features (не удалять!).
-- - Если на сервере запрещено CREATE SCHEMA, просто удалите блок CREATE SCHEMA.
--
-- Важно про «схему» PostgreSQL:
-- - Таблицы создаются в текущем search_path.
-- - В вашем проекте search_path выставляется в db.py через PG_SCHEMA.
--
-- ==============================================================

BEGIN;

-- (Опционально) Создание схемы. Может быть запрещено правами на соревнованиях.
-- Если получите ошибку "permission denied to create schema" — удалите этот блок.
DO $$
BEGIN
  IF current_schema() IS NULL THEN
    -- no-op
    NULL;
  END IF;
END $$;

-- ==============================================================
-- 0) Источники данных (файлы/пакеты)
-- ==============================================================
-- Зачем:
-- - не допускать дублей при повторном запуске ingest (по file_hash)
-- - хранить метаданные об источнике
CREATE TABLE IF NOT EXISTS data_sources (
    source_id      BIGSERIAL PRIMARY KEY,
    source_type    TEXT NOT NULL,               -- 'gpx', 'parquet', 'csv', 'api', ...
    source_name    TEXT NOT NULL,               -- имя файла или логическое имя набора
    source_path    TEXT,                        -- путь/URL (если применимо)
    file_hash      TEXT UNIQUE,                 -- sha256/md5 (если есть файл)
    received_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    extra          JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE data_sources IS 'Источники данных: файлы/пакеты. Используется для антидубликатов и аудита загрузки.';
COMMENT ON COLUMN data_sources.file_hash IS 'Хэш файла (например sha256). Если заполнен — помогает избегать дублей.';

-- ==============================================================
-- 1) Треки/рейсы/маршруты (верхний уровень)
-- ==============================================================
-- Сущность "трек" универсальна:
-- - для GPX: один маршрут (trk), возможно несколько сегментов (trkseg)
-- - для телеметрии: один рейс/поездка/смена (trip_id)
CREATE TABLE IF NOT EXISTS tracks (
    track_id       BIGSERIAL PRIMARY KEY,

    -- Ссылка на источник (файл/набор)
    source_id      BIGINT REFERENCES data_sources(source_id) ON DELETE SET NULL,

    -- Внешние идентификаторы (если есть в исходных данных)
    ext_track_id   TEXT,                        -- например gpx track name/id, tripid и т.п.
    ext_object_id  TEXT,                        -- например vehicle_id/objectid
    ext_driver_id  TEXT,                        -- если есть водитель/пользователь

    track_name     TEXT,
    track_type     TEXT,                        -- 'hike', 'bike', 'truck_trip', ... (опционально)

    -- Временные границы трека (если есть time)
    start_time     TIMESTAMPTZ,
    end_time       TIMESTAMPTZ,

    -- Быстрые агрегаты/сводки (можно обновлять после загрузки точек)
    segment_count  INT NOT NULL DEFAULT 0,
    point_count    INT NOT NULL DEFAULT 0,

    -- Bounding box для быстрых фильтров/карт
    min_lat        DOUBLE PRECISION,
    min_lon        DOUBLE PRECISION,
    max_lat        DOUBLE PRECISION,
    max_lon        DOUBLE PRECISION,

    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    extra          JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE tracks IS 'Треки/рейсы/маршруты. Один трек содержит много точек (track_points).';
COMMENT ON COLUMN tracks.ext_track_id IS 'Внешний идентификатор трека из исходных данных (например tripid или имя трека из GPX).';
COMMENT ON COLUMN tracks.ext_object_id IS 'Внешний идентификатор объекта (самосвал/устройство/спортсмен/трекер и т.п.).';
COMMENT ON COLUMN tracks.extra IS 'Любые дополнительные метаданные трека (JSONB), если структура входа расширенная.';

-- ==============================================================
-- 2) Точки трека (сырая телеметрия / GPS точки)
-- ==============================================================
-- Это центральная таблица. Здесь храним:
-- - координаты (lat/lon) — почти всегда есть
-- - время точки (time) — часто есть, но может отсутствовать
-- - высоту (ele/height) — может отсутствовать
-- - любые сенсорные поля можно хранить как отдельные колонки И/ИЛИ в JSONB extra
--
-- Почему делаем extra JSONB:
-- - универсальность: если на соревнованиях появятся новые поля — вы не ломаете схему.
CREATE TABLE IF NOT EXISTS track_points (
    point_id       BIGSERIAL PRIMARY KEY,
    track_id       BIGINT NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,

    -- Порядок точки внутри трека
    seq            INT NOT NULL,

    -- Сегмент (для GPX trkseg). Для обычных данных можно оставить 0.
    segment_index  INT NOT NULL DEFAULT 0,

    -- География
    lat            DOUBLE PRECISION,
    lon            DOUBLE PRECISION,

    -- Плоские координаты (например UTM x/y) — если исходные данные в метрах
    x              DOUBLE PRECISION,
    y              DOUBLE PRECISION,

    -- Высота / отметка
    ele            DOUBLE PRECISION,

    -- Время точки
    time           TIMESTAMPTZ,

    -- Часто встречающиеся сенсорные атрибуты (опционально; можно не заполнять)
    speed          DOUBLE PRECISION,
    heading        DOUBLE PRECISION,

    -- Любые дополнительные датчики/поля (JSON)
    extra          JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE track_points IS 'Сырые точки трека (GPS/телеметрия). Основной слой данных для анализа и построения карт.';
COMMENT ON COLUMN track_points.seq IS 'Порядковый номер точки в треке (0..N). Делает трек восстанавливаемым в правильном порядке.';
COMMENT ON COLUMN track_points.segment_index IS 'Номер сегмента (GPX trkseg). Если не используется — оставлять 0.';
COMMENT ON COLUMN track_points.extra IS 'Расширенные атрибуты точки: датчики, качество GPS, ускорения и т.п. (JSONB).';

-- Уникальность (трек + порядок). Защищает от дублей точек при перезапуске ingest.
CREATE UNIQUE INDEX IF NOT EXISTS ux_track_points_track_seq
    ON track_points(track_id, seq);

-- ==============================================================
-- 3) Признаки по треку (агрегаты для EDA/кластеризации/ML)
-- ==============================================================
-- Таблица признаков — это «витрина» для анализа.
-- Вы можете добавлять новые колонки по мере необходимости.
CREATE TABLE IF NOT EXISTS track_features (
    track_id              BIGINT PRIMARY KEY REFERENCES tracks(track_id) ON DELETE CASCADE,

    -- Базовые признаки (универсальные)
    distance_m            DOUBLE PRECISION,     -- длина трека в метрах
    duration_s            DOUBLE PRECISION,     -- длительность (если есть time)

    elev_min_m            DOUBLE PRECISION,
    elev_max_m            DOUBLE PRECISION,
    elev_gain_m           DOUBLE PRECISION,
    elev_loss_m           DOUBLE PRECISION,

    avg_speed_mps         DOUBLE PRECISION,
    max_speed_mps         DOUBLE PRECISION,

    stop_time_s           DOUBLE PRECISION,
    stop_ratio            DOUBLE PRECISION,     -- stop_time_s / duration_s

    point_density_per_km  DOUBLE PRECISION,     -- point_count / distance

    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    extra                 JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE track_features IS 'Агрегированные признаки по треку. Используются для EDA, проверки распределений, кластеризации и моделей.';
COMMENT ON COLUMN track_features.extra IS 'Любые дополнительные признаки (JSONB), если нужно быстро расшириться без ALTER TABLE.';

-- ==============================================================
-- 4) Контекстные данные (например погода) — универсально по времени
-- ==============================================================
-- Если на соревновании дадут «второй файл» с контекстом (погода/события),
-- удобнее хранить его отдельной таблицей. Потом делаете JOIN по времени.
CREATE TABLE IF NOT EXISTS context_time_series (
    context_id     BIGSERIAL PRIMARY KEY,
    source_id      BIGINT REFERENCES data_sources(source_id) ON DELETE SET NULL,

    -- Временная метка контекста (например каждый час)
    time           TIMESTAMPTZ NOT NULL,

    -- Если контекст зависит от места — можно сохранить координаты
    lat            DOUBLE PRECISION,
    lon            DOUBLE PRECISION,

    -- Сами значения (JSONB, чтобы не зависеть от набора колонок)
    values         JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE context_time_series IS 'Контекстные временные ряды (погода/индикаторы). Значения хранятся в JSONB values.';

-- ==============================================================
-- 5) Кластеризация / результаты аналитики (версирование запусков)
-- ==============================================================
-- Отделяем «запуск» кластеризации от «меток на треках».
-- Это позволяет сравнивать разные методы/настройки и хранить качество.
CREATE TABLE IF NOT EXISTS cluster_runs (
    cluster_run_id      BIGSERIAL PRIMARY KEY,
    run_tag             TEXT NOT NULL,          -- например 'kmeans_v1', 'dbscan_eps_0.2'
    method              TEXT NOT NULL,          -- 'kmeans', 'dbscan', 'iforest', ...
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Метрики качества на весь запуск (если рассчитаны)
    silhouette          DOUBLE PRECISION,
    calinski_harabasz   DOUBLE PRECISION,
    davies_bouldin      DOUBLE PRECISION,

    params              JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE cluster_runs IS 'Запуски кластеризации/разметки. Содержит метрики качества и параметры запуска.';

-- Метки кластеров для каждого трека в рамках конкретного запуска
CREATE TABLE IF NOT EXISTS track_cluster_labels (
    cluster_run_id   BIGINT NOT NULL REFERENCES cluster_runs(cluster_run_id) ON DELETE CASCADE,
    track_id         BIGINT NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
    cluster_id       INT NOT NULL,

    PRIMARY KEY (cluster_run_id, track_id)
);

COMMENT ON TABLE track_cluster_labels IS 'Результат кластеризации: к какому кластеру отнесён каждый трек в рамках cluster_run_id.';

-- ==============================================================
-- 6) (Опционально) Предсказания моделей (если будет модуль ML)
-- ==============================================================
-- Универсально для регрессии/классификации: хранить результаты предсказаний
-- для треков или для точек.
CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id   BIGSERIAL PRIMARY KEY,
    run_tag        TEXT NOT NULL,               -- 'lgbm_v1', 'catboost_v2', ...
    task_type      TEXT NOT NULL,               -- 'regression', 'classification'
    target_name    TEXT NOT NULL,               -- например 'speed'
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metrics        JSONB NOT NULL DEFAULT '{}'::jsonb,
    params         JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS track_predictions (
    model_run_id   BIGINT NOT NULL REFERENCES model_runs(model_run_id) ON DELETE CASCADE,
    track_id       BIGINT NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,

    -- Результат может быть числом (регрессия) или классом (классификация)
    y_pred_num     DOUBLE PRECISION,
    y_pred_class   TEXT,
    y_pred_proba   JSONB NOT NULL DEFAULT '{}'::jsonb,

    PRIMARY KEY (model_run_id, track_id)
);

COMMENT ON TABLE track_predictions IS 'Предсказания модели на уровне трека (регрессия/классификация).';

-- ==============================================================
-- Конец схемы
-- ==============================================================

COMMIT;