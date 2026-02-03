

-- sql/002_indexes.sql
-- ==============================================================
-- Универсальные индексы для ускорения запросов (дашборд/EDA/ML)
-- ==============================================================
-- Зачем это нужно:
-- - Индексы ускоряют SELECT/JOIN/FILTER по большим таблицам (особенно track_points).
-- - На соревнованиях дашборд часто «тормозит» не из-за Python, а из-за медленных запросов.
--
-- Как использовать:
-- - Выполняйте после 001_schema.sql.
-- - Индексы безопасны: CREATE INDEX IF NOT EXISTS.
--
-- Что возможно придется менять на соревнованиях:
-- - Обычно НИЧЕГО.
-- - Если правами запрещено CREATE INDEX — удалите/пропустите этот файл.
-- - Если данных мало, индексы не критичны, но наличие индексов — плюс к качеству решения.
--
-- Примечание:
-- - Мы НЕ делаем GiST/SP-GiST индексы на геометрию, чтобы не требовать PostGIS.
-- - Если PostGIS разрешён и нужен быстрый spatial-поиск — это можно расширить отдельно.
-- ==============================================================

BEGIN;

-- ==============================================================
-- 1) Индексы для tracks
-- ==============================================================
-- Часто фильтруют по объекту/водителю и времени.
CREATE INDEX IF NOT EXISTS ix_tracks_ext_object_id
    ON tracks(ext_object_id);

CREATE INDEX IF NOT EXISTS ix_tracks_ext_driver_id
    ON tracks(ext_driver_id);

CREATE INDEX IF NOT EXISTS ix_tracks_start_time
    ON tracks(start_time);

CREATE INDEX IF NOT EXISTS ix_tracks_end_time
    ON tracks(end_time);

-- Если вам часто нужно находить трек по внешнему id (например tripid)
CREATE INDEX IF NOT EXISTS ix_tracks_ext_track_id
    ON tracks(ext_track_id);

-- ==============================================================
-- 2) Индексы для track_points (самая «тяжёлая» таблица)
-- ==============================================================
-- Быстрый доступ к точкам конкретного трека (часто нужен для построения карты/графиков)
CREATE INDEX IF NOT EXISTS ix_track_points_track_id
    ON track_points(track_id);

-- Быстрый доступ к точкам по времени (если строите графики «по времени»)
CREATE INDEX IF NOT EXISTS ix_track_points_time
    ON track_points(time);

-- Композитный индекс для частого кейса: выбрать точки трека в порядке времени
-- (подходит и для GPX, и для телеметрии)
CREATE INDEX IF NOT EXISTS ix_track_points_track_time
    ON track_points(track_id, time);

-- Композитный индекс для кейса: выбрать точки трека в порядке seq
-- (у вас уже есть уникальный индекс ux_track_points_track_seq в 001_schema.sql,
--  но отдельный обычный индекс иногда помогает планировщику — оставляем не всегда)
-- Если хотите минимизировать количество индексов — этот можно удалить.
CREATE INDEX IF NOT EXISTS ix_track_points_track_seq
    ON track_points(track_id, seq);

-- Индексы по координатам (ускоряют грубые фильтры типа bbox)
-- Важно: это НЕ пространственный индекс, но на практике помогает.
CREATE INDEX IF NOT EXISTS ix_track_points_lat
    ON track_points(lat);

CREATE INDEX IF NOT EXISTS ix_track_points_lon
    ON track_points(lon);

-- Если часто строите bbox-фильтр (min_lat/max_lat/min_lon/max_lon) —
-- можно ускорить композитным индексом (не всегда нужен).
-- Если данных мало или запросов bbox нет — можно удалить.
CREATE INDEX IF NOT EXISTS ix_track_points_lat_lon
    ON track_points(lat, lon);

-- ==============================================================
-- 3) Индексы для track_features
-- ==============================================================
-- Здесь обычно 1 строка на трек, первичный ключ track_id уже индекс.
-- Дополнительно полезны индексы по «частым фильтрам» для кластеризации/EDA.
CREATE INDEX IF NOT EXISTS ix_track_features_distance
    ON track_features(distance_m);

CREATE INDEX IF NOT EXISTS ix_track_features_duration
    ON track_features(duration_s);

CREATE INDEX IF NOT EXISTS ix_track_features_avg_speed
    ON track_features(avg_speed_mps);

-- ==============================================================
-- 4) Индексы для context_time_series
-- ==============================================================
-- Контекст обычно джоинят по времени (например погода почасовая).
CREATE INDEX IF NOT EXISTS ix_context_time_series_time
    ON context_time_series(time);

-- ==============================================================
-- 5) Индексы для кластеризации и предсказаний
-- ==============================================================
-- Быстрый доступ к меткам по треку и по запуску
CREATE INDEX IF NOT EXISTS ix_track_cluster_labels_track_id
    ON track_cluster_labels(track_id);

CREATE INDEX IF NOT EXISTS ix_track_cluster_labels_run_id
    ON track_cluster_labels(cluster_run_id);

-- Быстрый доступ к предсказаниям по треку и по запуску модели
CREATE INDEX IF NOT EXISTS ix_track_predictions_track_id
    ON track_predictions(track_id);

CREATE INDEX IF NOT EXISTS ix_track_predictions_model_run_id
    ON track_predictions(model_run_id);

COMMIT;