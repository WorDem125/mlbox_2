

-- sql/003_views.sql
-- ==============================================================
-- Универсальные VIEW (витрины) для дашборда и EDA
-- ==============================================================
-- Что такое VIEW:
-- - VIEW = сохранённый SELECT. ДАННЫЕ не копируются.
-- - Удобно: дашборд подключается к «готовым витринам», а не к сырым таблицам.
--
-- Зачем этот файл:
-- - Спрятать JOIN/агрегации/вычисления за именем представления.
-- - Упростить код Streamlit/FastAPI/аналитики: SELECT * FROM v_tracks_overview;
-- - Сделать «единый интерфейс» даже если у вас разные входные источники.
--
-- Что возможно придется менять на соревнованиях:
-- - Обычно НИЧЕГО: витрины опираются на универсальные таблицы.
-- - Если эксперты потребуют конкретные метрики/графики, вы ДОБАВИТЕ новые VIEW
--   или расширите существующие (не ломая базовые).
-- - Если у входных данных нет времени (time) — некоторые поля будут NULL.
-- - Если у вас нет рассчитанных features (таблица track_features пустая) —
--   витрины всё равно работают, просто поля features будут NULL.
--
-- Важно про схему PostgreSQL:
-- - VIEW создаются в текущем search_path (вы задаёте PG_SCHEMA в .env).
-- ==============================================================

BEGIN;

-- ==============================================================
-- 1) Витрина «одна строка = один трек»
-- ==============================================================
-- Использование:
-- - Главный источник для таблицы/фильтров на дашборде.
-- - Показывает основные поля трека + признаки + последняя метка кластера.
--
-- Почему «последняя метка кластера»:
-- - Кластеризацию могут запускать несколько раз.
-- - Для дашборда часто нужна «актуальная» версия (самый свежий cluster_run).
CREATE OR REPLACE VIEW v_tracks_overview AS
WITH latest_cluster_run AS (
    SELECT cr.cluster_run_id
    FROM cluster_runs cr
    ORDER BY cr.created_at DESC
    LIMIT 1
)
SELECT
    t.track_id,
    t.source_id,
    t.ext_track_id,
    t.ext_object_id,
    t.ext_driver_id,
    t.track_name,
    t.track_type,
    t.start_time,
    t.end_time,
    t.segment_count,
    t.point_count,
    t.min_lat,
    t.min_lon,
    t.max_lat,
    t.max_lon,
    t.created_at,

    -- Признаки (могут быть NULL, если ещё не рассчитаны)
    f.distance_m,
    f.duration_s,
    f.elev_min_m,
    f.elev_max_m,
    f.elev_gain_m,
    f.elev_loss_m,
    f.avg_speed_mps,
    f.max_speed_mps,
    f.stop_time_s,
    f.stop_ratio,
    f.point_density_per_km,
    f.updated_at AS features_updated_at,

    -- Кластер (могут быть NULL, если кластеризация не запускалась)
    cr.cluster_run_id AS latest_cluster_run_id,
    lbl.cluster_id     AS latest_cluster_id
FROM tracks t
LEFT JOIN track_features f
    ON f.track_id = t.track_id
LEFT JOIN latest_cluster_run lcr
    ON TRUE
LEFT JOIN track_cluster_labels lbl
    ON lbl.track_id = t.track_id
   AND lbl.cluster_run_id = lcr.cluster_run_id
LEFT JOIN cluster_runs cr
    ON cr.cluster_run_id = lcr.cluster_run_id;

COMMENT ON VIEW v_tracks_overview IS 'Витрина: одна строка = один трек (tracks + features + последняя метка кластера).';

-- ==============================================================
-- 2) Витрина «точки трека для карты» (легкий слой)
-- ==============================================================
-- Зачем:
-- - Дашборду нужно быстро получать точки конкретного трека для отрисовки линии.
-- - Мы не делаем тяжелые вычисления, только отдаём упорядоченные точки.
--
-- На соревнованиях:
-- - Обычно не менять.
-- - Если вам нужны дополнительные колонки (например speed/ele/time) — они уже есть.
CREATE OR REPLACE VIEW v_track_points_map AS
SELECT
    p.track_id,
    p.seq,
    p.segment_index,
    p.lat,
    p.lon,
    p.x,
    p.y,
    p.ele,
    p.time,
    p.speed,
    p.heading
FROM track_points p
WHERE p.lat IS NOT NULL AND p.lon IS NOT NULL;

COMMENT ON VIEW v_track_points_map IS 'Витрина для карты: точки трека (упорядочивание делается запросом ORDER BY seq/time в приложении).';

-- ==============================================================
-- 3) Витрина «почасовая статистика скорости»
-- ==============================================================
-- Требование типового задания:
-- - средняя скорость по часам суток (для всех треков или по объекту)
--
-- Как использовать:
-- - SELECT * FROM v_speed_by_hour;
-- - В дашборде можно фильтровать по ext_object_id через JOIN с tracks.
--
-- Примечание:
-- - Если time отсутствует, то hour будет NULL и строк не будет.
CREATE OR REPLACE VIEW v_speed_by_hour AS
SELECT
    EXTRACT(HOUR FROM p.time)::INT AS hour_of_day,
    AVG(p.speed)                  AS avg_speed,
    COUNT(*)                      AS points_cnt
FROM track_points p
WHERE p.time IS NOT NULL
  AND p.speed IS NOT NULL
GROUP BY EXTRACT(HOUR FROM p.time)
ORDER BY hour_of_day;

COMMENT ON VIEW v_speed_by_hour IS 'Почасовая средняя скорость (по всем точкам). Удобно для графика avg speed vs hour.';

-- ==============================================================
-- 4) Витрина «дистанция по трекам» (если distance_m уже рассчитана)
-- ==============================================================
-- Зачем:
-- - Быстро показать топ/распределение дистанций.
-- - Используется для EDA, кластеризации, фильтров.
CREATE OR REPLACE VIEW v_track_distances AS
SELECT
    t.track_id,
    t.ext_object_id,
    t.ext_driver_id,
    t.start_time,
    t.end_time,
    f.distance_m,
    f.duration_s,
    f.avg_speed_mps,
    f.max_speed_mps
FROM tracks t
JOIN track_features f
  ON f.track_id = t.track_id
WHERE f.distance_m IS NOT NULL;

COMMENT ON VIEW v_track_distances IS 'Дистанции и базовые фичи по трекам (только там, где features уже рассчитаны).';

-- ==============================================================
-- 5) Витрина «метрики качества кластеризации по запускам»
-- ==============================================================
-- Зачем:
-- - Показать экспертам сравнение методов/параметров.
-- - Быстро построить таблицу качества.
CREATE OR REPLACE VIEW v_cluster_run_quality AS
SELECT
    cr.cluster_run_id,
    cr.run_tag,
    cr.method,
    cr.created_at,
    cr.silhouette,
    cr.calinski_harabasz,
    cr.davies_bouldin,
    cr.params
FROM cluster_runs cr
ORDER BY cr.created_at DESC;

COMMENT ON VIEW v_cluster_run_quality IS 'Качество кластеризации по запускам (silhouette, CH, DB).';

-- ==============================================================
-- 6) Витрина «распределение треков по кластерам (последний запуск)»
-- ==============================================================
-- Зачем:
-- - Быстрый график/таблица: сколько треков в каждом кластере.
-- - Очень полезно для отчёта и проверки результата.
CREATE OR REPLACE VIEW v_cluster_distribution_latest AS
WITH latest_cluster_run AS (
    SELECT cluster_run_id
    FROM cluster_runs
    ORDER BY created_at DESC
    LIMIT 1
)
SELECT
    lbl.cluster_id,
    COUNT(*) AS tracks_cnt
FROM track_cluster_labels lbl
JOIN latest_cluster_run lcr
  ON lbl.cluster_run_id = lcr.cluster_run_id
GROUP BY lbl.cluster_id
ORDER BY lbl.cluster_id;

COMMENT ON VIEW v_cluster_distribution_latest IS 'Распределение треков по кластерам для последнего запуска кластеризации.';

-- ==============================================================
-- 7) Витрина «контекст по времени» (для JOIN в модели/EDA)
-- ==============================================================
-- Универсально: значения в JSONB values.
-- Пример использования:
-- - брать ближайшую по времени погоду к start_time трека
-- - строить графики индикаторов
CREATE OR REPLACE VIEW v_context_time_series AS
SELECT
    c.context_id,
    c.source_id,
    c.time,
    c.lat,
    c.lon,
    c.values
FROM context_time_series c;

COMMENT ON VIEW v_context_time_series IS 'Контекстные временные ряды (погода/индикаторы) в удобном виде.';

COMMIT;