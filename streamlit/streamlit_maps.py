

"""streamlit/streamlit_maps.py

Streamlit-дешборд для визуализации GPX-треков из PostgreSQL.

Что показывает:
- список треков (tracks)
- карту с треком (по точкам из track_points)
- базовую статистику (track_features)
- контекст (context_time_series): что рядом (OSM/Overpass) и погода (Open-Meteo archive)

Запуск (из корня проекта):
    streamlit run streamlit/streamlit_maps.py

Важно про схему:
- Схема БД берётся из .env (PG_SCHEMA). Мы НЕ хардкодим public.
- На каждом соединении выставляем `SET search_path`.

Если чего-то нет (например, context_time_series), приложение не падает,
а выводит предупреждение.
"""

from __future__ import annotations

# --- ВАЖНО для Streamlit ---
# Streamlit запускает файл как скрипт, и иногда корень проекта не попадает в PYTHONPATH.
# Поэтому импорт `from src...` может падать с `ModuleNotFoundError: No module named 'src'`.
# Решение: добавить корень проекта в sys.path.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../mlbox_2
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import math

import numpy as np
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# Карта: folium + streamlit-folium (если не установлен, просто покажем таблицы)
try:
    import folium
    from branca.colormap import LinearColormap
    HAS_FOLIUM = True
except Exception:
    folium = None
    HAS_FOLIUM = False

try:
    from streamlit_folium import st_folium

    HAS_ST_FOLIUM = True
except Exception:
    st_folium = None
    HAS_ST_FOLIUM = False

from sqlalchemy import text

from src.common.db import DBConfig, get_connection, set_search_path, test_connection


# -----------------------------
# UI helpers
# -----------------------------

def _badge(label: str, value: Any) -> None:
    """Маленькая карточка-метрика."""
    st.metric(label, value)


# Красивое форматирование чисел для UI (например, температуры)
def _fmt_float(x: Any, ndigits: int = 2) -> str:
    """Красивое форматирование чисел для UI."""
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x))):
            return "—"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "—"


def _safe_json(v: Any) -> Dict[str, Any]:
    """values в БД может быть dict/json/строка. Приводим к dict."""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


# -----------------------------
# Человеческая интерпретация контекста OSM (nearby) и погоды
# -----------------------------

# Перевод категорий OSM (ключи верхнего уровня)
TAG_RU = {
    "highway": "Дороги/пути",
    "landuse": "Тип местности",
    "natural": "Природа",
    "waterway": "Вода (реки/каналы)",
    "building": "Застройка",
    "amenity": "Инфраструктура",
    "leisure": "Досуг/спорт",
}

# Перевод часто встречающихся значений. Можно расширять.
VALUE_RU = {
    # highway
    "footway": "пешеходная дорожка/тротуар",
    "path": "тропа",
    "steps": "лестница",
    "residential": "жилая улица",
    "service": "служебный проезд",
    "primary": "главная дорога",
    "secondary": "второстепенная дорога",
    "tertiary": "дорога местного значения",
    "cycleway": "велодорожка",
    "crossing": "пешеходный переход",
    "traffic_signals": "светофор",
    "turning_circle": "разворотная площадка",
    "street_lamp": "фонарь",

    # landuse
    "grass": "газон/трава",
    "meadow": "луг",
    "industrial": "промзона",
    "construction": "стройка",
    "retail": "торговая зона",
    "allotments": "дачи/огороды",

    # natural
    "wood": "лес",
    "scrub": "кустарник",
    "water": "водоём",
    "cliff": "обрыв",
    "tree": "дерево",

    # building
    "yes": "здание",
    "apartments": "многоквартирные дома",
    "detached": "частные дома",
    "school": "школа",
    "shed": "хозпостройка/сарай",
    "hut": "домик/хижина",

    # amenity
    "parking": "парковка",
    "bench": "лавочка",
    "pub": "паб/бар",
    "post_box": "почтовый ящик",
    "post_office": "почта",
    "bus_stop": "остановка",
    "bicycle_parking": "велопарковка",
    "waste_basket": "урна",
    "parking_entrance": "въезд на парковку",
}


def _ru_value(v: str) -> str:
    return VALUE_RU.get(v, v)


def _counts_to_rows(counts: dict) -> list:
    """counts формата {"highway": [["residential", 16], ...], ...} -> плоский список."""
    rows = []
    if not isinstance(counts, dict):
        return rows
    for tag, arr in counts.items():
        if not isinstance(arr, list):
            continue
        for item in arr:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                val, n = item[0], item[1]
                try:
                    n = int(n)
                except Exception:
                    continue
                rows.append(
                    {
                        "tag": str(tag),
                        "tag_ru": TAG_RU.get(str(tag), str(tag)),
                        "value": str(val),
                        "value_ru": _ru_value(str(val)),
                        "count": n,
                    }
                )
    return rows


def summarize_route_nearby(ctx_df: pd.DataFrame) -> pd.DataFrame:
    """Агрегируем nearby.counts по всем строкам контекста выбранного трека."""
    if ctx_df is None or ctx_df.empty or "values" not in ctx_df.columns:
        return pd.DataFrame(columns=["tag_ru", "value_ru", "count", "tag", "value"])

    vals = ctx_df["values"].apply(_safe_json)
    all_rows = []
    for d in vals:
        nearby = d.get("nearby", {}) if isinstance(d, dict) else {}
        counts = nearby.get("counts", {}) if isinstance(nearby, dict) else {}
        all_rows.extend(_counts_to_rows(counts))

    if not all_rows:
        return pd.DataFrame(columns=["tag_ru", "value_ru", "count", "tag", "value"])

    df = pd.DataFrame(all_rows)
    df = df.groupby(["tag", "tag_ru", "value", "value_ru"], as_index=False)["count"].sum()
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    # Удобный порядок колонок
    return df[["tag_ru", "value_ru", "count", "tag", "value"]]


def make_human_summary(agg_df: pd.DataFrame) -> str:
    """Короткий текст: что чаще всего встречается рядом по маршруту."""
    if agg_df is None or agg_df.empty:
        return "Контекст по местности пока не собран или пуст."

    top3 = agg_df.head(3)
    top_line = ", ".join(
        [f"{r['tag_ru']}: {r['value_ru']}×{int(r['count'])}" for _, r in top3.iterrows()]
    )

    # Стабильный набор блоков
    blocks = []
    for tag in ["highway", "natural", "waterway", "landuse", "building", "amenity"]:
        sub = agg_df[agg_df["tag"] == tag].head(3)
        if sub.empty:
            continue
        title = TAG_RU.get(tag, tag)
        items = ", ".join([f"{r['value_ru']} ({int(r['count'])})" for _, r in sub.iterrows()])
        blocks.append(f"- **{title}:** {items}")

    return (
        "### Итог по маршруту (контекст рядом)\n"
        f"- **ТОП-3:** {top_line}\n" + "\n".join(blocks)
    )


def weather_summary(ctx_df: pd.DataFrame) -> dict:
    """Сводка по погоде. Важно: сейчас в БД сохраняется только temp_c (если есть время)."""
    out = {
        "rows": 0,
        "ok": 0,
        "fail": 0,
        "no_time": 0,
        "temp_avg": None,
        "temp_min": None,
        "temp_max": None,
    }
    if ctx_df is None or ctx_df.empty or "values" not in ctx_df.columns:
        return out

    vals = ctx_df["values"].apply(_safe_json)
    out["rows"] = int(len(vals))

    temps = []
    for d in vals:
        w = d.get("weather", {}) if isinstance(d, dict) else {}
        if not isinstance(w, dict):
            continue
        ok = w.get("ok")
        if ok is True:
            out["ok"] += 1
        elif ok is False:
            out["fail"] += 1
            if w.get("reason") == "no_time":
                out["no_time"] += 1
        # temp
        t = w.get("temp_c")
        try:
            if t is not None:
                temps.append(float(t))
        except Exception:
            pass

    if temps:
        out["temp_avg"] = float(np.mean(temps))
        out["temp_min"] = float(np.min(temps))
        out["temp_max"] = float(np.max(temps))

    return out


def extract_weather_fields(values: Dict[str, Any]) -> Dict[str, Any]:
    """Достаём удобные поля погоды из JSON values."""
    w = values.get("weather") if isinstance(values, dict) else None
    if not isinstance(w, dict):
        return {
            "weather_ok": None,
            "temp_c": None,
            "weather_reason": None,
            "hour_utc": None,
            "weather_source": None,
        }
    return {
        "weather_ok": w.get("ok"),
        "temp_c": w.get("temp_c"),
        "weather_reason": w.get("reason"),
        "hour_utc": w.get("hour_utc"),
        "weather_source": w.get("source"),
    }


def aggregate_nearby_counts(ctx_df: pd.DataFrame, top_n: int = 20, include_tech: bool = False) -> pd.DataFrame:
    """ТОП рядом по маршруту, с русскими названиями.

    Возвращает таблицу:
      - Категория | Значение | Кол-во

    Если include_tech=True, добавляет технические поля tag/value (удобно для отладки).
    """
    agg = summarize_route_nearby(ctx_df)
    if agg.empty:
        base_cols = ["Категория", "Значение", "Кол-во"]
        if include_tech:
            base_cols += ["tag", "value"]
        return pd.DataFrame(columns=base_cols)

    out = agg.head(top_n).copy()
    out = out.rename(columns={"tag_ru": "Категория", "value_ru": "Значение", "count": "Кол-во"})

    if include_tech:
        return out[["Категория", "Значение", "Кол-во", "tag", "value"]]
    return out[["Категория", "Значение", "Кол-во"]]


def build_context_summary(ctx_df: pd.DataFrame) -> Dict[str, Any]:
    """Короткая сводка по контексту (погода/время)."""
    if ctx_df is None or ctx_df.empty:
        return {"rows": 0}

    vals = ctx_df["values"].apply(_safe_json)
    w = vals.apply(extract_weather_fields)
    w_df = pd.DataFrame(list(w))

    rows = int(len(ctx_df))
    ok_mask = w_df["weather_ok"] == True
    fail_mask = w_df["weather_ok"] == False
    no_time_mask = w_df["weather_reason"] == "no_time"

    temps = pd.to_numeric(w_df["temp_c"], errors="coerce")

    summary = {
        "rows": rows,
        "weather_ok": int(ok_mask.sum()),
        "weather_fail": int(fail_mask.sum()),
        "no_time": int(no_time_mask.sum()),
        "temp_avg": float(temps.mean()) if temps.notna().any() else None,
        "temp_min": float(temps.min()) if temps.notna().any() else None,
        "temp_max": float(temps.max()) if temps.notna().any() else None,
    }
    return summary


def _try_query_df(sql: str, params: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """Безопасный запрос: если таблицы/колонки нет — возвращаем None и пишем предупреждение."""
    try:
        with get_connection() as conn:
            set_search_path(conn)
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as e:
        st.warning(f"Не удалось выполнить запрос. Возможно, таблицы/колонки нет или нет прав. Ошибка: {e}")
        return None


# -----------------------------
# Data access
# -----------------------------

@st.cache_data(ttl=30)
def load_tracks() -> pd.DataFrame:
    """Список треков."""
    sql = """
    SELECT
        track_id,
        source_id,
        track_name,
        start_time,
        end_time,
        segment_count,
        point_count,
        min_lat, min_lon, max_lat, max_lon
    FROM tracks
    ORDER BY track_id;
    """
    df = _try_query_df(sql)
    return df if df is not None else pd.DataFrame()


@st.cache_data(ttl=30)
def load_track_features(track_id: int) -> Optional[pd.DataFrame]:
    sql = """
    SELECT *
    FROM track_features
    WHERE track_id = :track_id
    LIMIT 1;
    """
    return _try_query_df(sql, {"track_id": track_id})


@st.cache_data(ttl=30)
def load_track_points(track_id: int) -> pd.DataFrame:
    """Точки трека. Сортируем по segment_index, seq."""
    sql = """
    SELECT track_id, segment_index, seq, lat, lon, ele, time, speed
    FROM track_points
    WHERE track_id = :track_id
    ORDER BY segment_index, seq;
    """
    df = _try_query_df(sql, {"track_id": track_id})
    return df if df is not None else pd.DataFrame()


@st.cache_data(ttl=30)
def load_context_for_source(source_id: int) -> pd.DataFrame:
    """Контекст для source_id (в context_time_series лежит source_id, а не track_id)."""
    sql = """
    SELECT context_id, source_id, time, lat, lon, values
    FROM context_time_series
    WHERE source_id = :source_id
    ORDER BY time NULLS LAST, context_id;
    """
    df = _try_query_df(sql, {"source_id": source_id})
    return df if df is not None else pd.DataFrame()


# -----------------------------
# Map building
# -----------------------------


def _center_from_bbox(row: pd.Series) -> Tuple[float, float]:
    """Центр карты: из bbox, иначе из первой точки."""
    try:
        lat = (float(row["min_lat"]) + float(row["max_lat"])) / 2.0
        lon = (float(row["min_lon"]) + float(row["max_lon"])) / 2.0
        if pd.isna(lat) or pd.isna(lon):
            raise ValueError("bbox is NaN")
        return lat, lon
    except Exception:
        return 0.0, 0.0


def build_map(
    track_row: pd.Series,
    points_df: pd.DataFrame,
    ctx_df: Optional[pd.DataFrame],
    sample_points_step: int,
    sample_ctx_step: int,
) -> "folium.Map":
    """Собираем folium карту с треком и (опционально) контекстом."""

    # Центр: bbox трека, иначе по первой точке
    center = _center_from_bbox(track_row)
    if center == (0.0, 0.0) and not points_df.empty:
        center = (float(points_df.iloc[0]["lat"]), float(points_df.iloc[0]["lon"]))

    m = folium.Map(location=center, zoom_start=12, control_scale=True, tiles=None)

    # Базовые тайлы (OSM). Это и есть "подтянуть карту из открытых источников".
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(m)

    # Доп. слой: рельеф/топо (может быть недоступен в некоторых сетях, но обычно работает)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="© OpenTopoMap (CC-BY-SA)",
        name="OpenTopoMap",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # Линия трека
    if not points_df.empty:
        # Чтобы карта не тормозила, рисуем полилинию по сэмплированным точкам
        pts = points_df.iloc[:: max(1, sample_points_step)][["lat", "lon"]].dropna()
        latlon = pts.values.tolist()
        if len(latlon) >= 2:
            folium.PolyLine(latlon, weight=4, opacity=0.9, tooltip="Трек").add_to(m)

        # Маркеры старт/финиш
        start = points_df.iloc[0]
        end = points_df.iloc[-1]
        folium.Marker(
            [float(start["lat"]), float(start["lon"])],
            tooltip="Старт",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)
        folium.Marker(
            [float(end["lat"]), float(end["lon"])],
            tooltip="Финиш",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(m)

    # Контекстные точки: показываем nearby и температуру (цветом)
    if ctx_df is not None and not ctx_df.empty:
        # Подготовим диапазон температур для цветовой шкалы
        vals_all = ctx_df["values"].apply(_safe_json)
        temps_all = vals_all.apply(lambda d: d.get("weather", {}).get("temp_c") if isinstance(d, dict) else None)
        temps_num = pd.to_numeric(pd.Series(list(temps_all)), errors="coerce")

        tmin = float(temps_num.min()) if temps_num.notna().any() else None
        tmax = float(temps_num.max()) if temps_num.notna().any() else None

        colormap = None
        if tmin is not None and tmax is not None and tmin != tmax:
            colormap = LinearColormap(["blue", "green", "orange", "red"], vmin=tmin, vmax=tmax)
            colormap.caption = "Температура (°C)"
            colormap.add_to(m)

        grp_ctx = folium.FeatureGroup(name="Контекст (nearby/weather)", show=True)

        sampled = ctx_df.iloc[:: max(1, sample_ctx_step)].copy()
        for _, r in sampled.iterrows():
            lat = float(r["lat"])
            lon = float(r["lon"])
            vals = _safe_json(r.get("values"))

            weather = vals.get("weather", {}) if isinstance(vals, dict) else {}
            w_ok = weather.get("ok") if isinstance(weather, dict) else None
            temp = weather.get("temp_c") if isinstance(weather, dict) else None
            w_reason = weather.get("reason") if isinstance(weather, dict) else None

            nearby = vals.get("nearby", {}) if isinstance(vals, dict) else {}
            counts = (nearby.get("counts") or {}) if isinstance(nearby, dict) else {}

            # Топ категорий по частоте (очень коротко)
            top_lines = []
            try:
                for k, arr in list(counts.items())[:6]:
                    if isinstance(arr, list) and arr:
                        v0 = arr[0]
                        if isinstance(v0, (list, tuple)) and len(v0) == 2:
                            top_lines.append(f"{k}: {v0[0]}×{v0[1]}")
                        else:
                            top_lines.append(f"{k}: {len(arr)}")
                    else:
                        top_lines.append(f"{k}: 0")
            except Exception:
                top_lines = []

            t = r.get("time")
            t_str = str(t) if pd.notna(t) else "(no_time)"

            popup_html = """
            <div style="font-size: 12px;">
              <b>Контекст точки</b><br/>
              <b>time:</b> {t}<br/>
              <b>temp:</b> {temp}<br/>
              <b>weather_ok:</b> {wok}<br/>
              <b>weather_reason:</b> {wreason}<br/>
              <hr style="margin:6px 0;"/>
              <b>nearby (top):</b><br/>
              {tops}
            </div>
            """.format(
                t=t_str,
                temp=(f"{temp} °C" if temp is not None else "—"),
                wok=("true" if w_ok else "false" if w_ok is not None else "—"),
                wreason=(w_reason or "—"),
                tops=("<br/>".join(top_lines) if top_lines else "—"),
            )

            # Цвет точки: если есть температура — по шкале, иначе по статусу погоды
            if colormap is not None and temp is not None:
                try:
                    color = colormap(float(temp))
                except Exception:
                    color = "blue"
            else:
                # fallback
                if w_ok is True:
                    color = "blue"
                elif w_reason == "no_time":
                    color = "gray"
                elif w_ok is False:
                    color = "orange"
                else:
                    color = "blue"

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=360),
            ).add_to(grp_ctx)

        grp_ctx.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


# -----------------------------
# App
# -----------------------------


def main() -> None:
    st.set_page_config(page_title="GPX карты", layout="wide")
    st.title("🗺️ GPX-треки: карта, характеристики и контекст")

    # Проверка подключения
    try:
        test_connection(verbose=True)
        cfg = DBConfig.from_env()
        st.caption(f"Схема: `{cfg.schema}` (из .env PG_SCHEMA)")
    except Exception as e:
        st.error(f"Нет подключения к БД. Проверь .env и сервер PostgreSQL. Ошибка: {e}")
        st.stop()

    # Сайдбар: параметры
    st.sidebar.header("Настройки отображения")
    sample_points_step = st.sidebar.slider("Точки трека на карте: каждая N-я", 1, 1000, 10, 1)
    sample_ctx_step = st.sidebar.slider("Точки контекста на карте: каждая N-я", 1, 1000, 1, 1)
    show_tech_cols = st.sidebar.checkbox("Показывать технические поля (tag/value)", value=False)

    tracks = load_tracks()
    if tracks.empty:
        st.warning("Таблица tracks пуста или недоступна.")
        st.stop()

    # Выбор трека
    tracks_display = tracks.copy()
    tracks_display["label"] = tracks_display.apply(
        lambda r: f"{int(r['track_id'])} | {str(r['track_name'])}", axis=1
    )

    selected_label = st.sidebar.selectbox("Выбери трек", tracks_display["label"].tolist())
    selected_row = tracks_display.loc[tracks_display["label"] == selected_label].iloc[0]
    track_id = int(selected_row["track_id"])
    source_id = int(selected_row["source_id"])

    # Грузим данные
    pts = load_track_points(track_id)
    feats = load_track_features(track_id)

    # Контекст может отсутствовать
    ctx = load_context_for_source(source_id)

    # Верхние метрики
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _badge("ID трека", track_id)
    with c2:
        _badge("Точек трека", int(selected_row.get("point_count", 0) or 0))
    with c3:
        _badge("Точек контекста", 0 if ctx is None else int(len(ctx)))
    with c4:
        _badge("Начало", str(selected_row.get("start_time")))
    with c5:
        _badge("Конец", str(selected_row.get("end_time")))

    left, right = st.columns([2, 1])

    # Карта
    with left:
        st.subheader("Карта")
        if not (HAS_FOLIUM and HAS_ST_FOLIUM):
            st.warning(
                "Для карты нужен `folium` и `streamlit-folium`. "
                "Установи: pip install folium streamlit-folium"
            )
        else:
            m = build_map(
                track_row=selected_row,
                points_df=pts,
                ctx_df=ctx,
                sample_points_step=sample_points_step,
                sample_ctx_step=sample_ctx_step,
            )
            st_folium(m, height=650, width=None)

    # Правая панель: фичи + контекст summary
    with right:
        st.subheader("Фичи трека")
        if feats is None or feats.empty:
            st.info("track_features для этого трека нет (или таблица недоступна).")
        else:
            r = feats.iloc[0]
            show_cols = [
                ("distance_m", "Дистанция", "м"),
                ("duration_s", "Длительность", "с"),
                ("avg_speed_mps", "Средняя скорость", "м/с"),
                ("max_speed_mps", "Макс. скорость", "м/с"),
                ("elev_min_m", "Мин. высота", "м"),
                ("elev_max_m", "Макс. высота", "м"),
                ("elev_gain_m", "Набор высоты", "м"),
                ("elev_loss_m", "Сброс высоты", "м"),
                ("stop_time_s", "Время остановок", "с"),
                ("stop_ratio", "Доля остановок", ""),
                ("point_density_per_km", "Плотность точек", "точек/км"),
            ]
            data_ru = {}
            for key, title, unit in show_cols:
                if key in feats.columns:
                    val = r.get(key)
                    # stop_ratio красивее как %
                    if key == "stop_ratio" and val is not None:
                        try:
                            data_ru[f"{title}"] = f"{float(val) * 100:.1f}%"
                            continue
                        except Exception:
                            pass
                    # остальные — числа
                    if isinstance(val, (int, float)):
                        data_ru[f"{title}{(' (' + unit + ')') if unit else ''}"] = float(val)
                    else:
                        data_ru[f"{title}{(' (' + unit + ')') if unit else ''}"] = val
            st.json(data_ru, expanded=False)

        st.subheader("Контекст маршрута (местность и погода)")
        st.caption(
            "Смысл контекста: не просто нарисовать линию по координатам, "
            "а объяснить *по какой местности* проходил маршрут (дороги/тропы/лес/вода/застройка) "
            "и какая была *погода* в момент прохождения. "
            "Это позволяет сравнивать треки между собой и делать выводы."
        )

        if ctx is None or ctx.empty:
            st.info(
                "Контекста нет (context_time_series пуст или недоступен). "
                "Сначала запусти: python -m src.enrich_context"
            )
        else:
            # 1) Сводка по погоде (человечески)
            ws = weather_summary(ctx)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Строк контекста", ws.get("rows", 0))
            with m2:
                st.metric("Погода получена", ws.get("ok", 0))
            with m3:
                st.metric("Погода недоступна", ws.get("fail", 0))
            with m4:
                st.metric("Нет времени в GPX", ws.get("no_time", 0))

            if ws.get("temp_avg") is not None:
                st.markdown("**Температура по доступным точкам (°C):**")
                t1, t2, t3 = st.columns(3)
                with t1:
                    st.metric("Средняя", _fmt_float(ws.get("temp_avg"), 1))
                with t2:
                    st.metric("Минимальная", _fmt_float(ws.get("temp_min"), 1))
                with t3:
                    st.metric("Максимальная", _fmt_float(ws.get("temp_max"), 1))
            else:
                st.info(
                    "Температуры нет: чаще всего это означает, что в GPX нет времени (time), "
                    "или погодный источник временно недоступен."
                )

            st.divider()

            # 2) Итог по местности (главная часть задания!)
            agg_ru = summarize_route_nearby(ctx)
            st.markdown(make_human_summary(agg_ru))

            st.caption("ТОП-20 объектов рядом с маршрутом (сводка по тегам OSM; удобно вставлять в отчёт)")
            top_nearby = aggregate_nearby_counts(ctx, top_n=20, include_tech=show_tech_cols)
            if top_nearby.empty:
                st.info(
                    "nearby.counts пуст — возможно, Overpass не ответил (429/504) "
                    "или контекст ещё не успел собраться. "
                    "Решение: попробуй увеличить задержку --sleep-s (например, 1–2 сек) "
                    "и/или уменьшить частоту точек (--point-step).")
            else:
                st.dataframe(top_nearby, use_container_width=True, height=320)

                # Небольшой график: только человеко-понятные подписи
                chart_df = top_nearby.copy()
                chart_df["label"] = chart_df.apply(lambda r: f"{r['Категория']}: {r['Значение']}", axis=1)
                chart_df = chart_df.set_index("label")[["Кол-во"]]
                st.bar_chart(chart_df)

            st.divider()

            # 3) Температура по времени (если есть)
            vals = ctx["values"].apply(_safe_json)
            weather_rows = pd.DataFrame(list(vals.apply(extract_weather_fields)))
            temp_series = pd.to_numeric(weather_rows["temp_c"], errors="coerce")
            if temp_series.notna().sum() >= 2:
                st.caption("Температура по точкам контекста")
                tdf = pd.DataFrame({"time": ctx["time"], "temp_c": temp_series}).dropna()
                tdf = tdf.sort_values("time")
                st.line_chart(tdf.set_index("time")["temp_c"])

            st.caption("Последние 5 строк контекста (для контроля / дебага)")
            view = ctx.tail(5).copy()
            view["values"] = view["values"].apply(_safe_json)
            st.dataframe(view[["context_id", "time", "lat", "lon", "values"]], use_container_width=True, height=260)

    # Ниже: точки/таблицы
    st.divider()
    st.subheader("Данные")
    tabs = st.tabs(["Треки", "Точки трека", "Контекст (OSM/погода)"])

    with tabs[0]:
        st.dataframe(tracks.drop(columns=["label"], errors="ignore"), use_container_width=True, height=320)

    with tabs[1]:
        if pts.empty:
            st.info("Точек нет")
        else:
            st.caption("Первые 200 точек (для контроля)")
            st.dataframe(pts.head(200), use_container_width=True, height=320)

    with tabs[2]:
        if ctx is None or ctx.empty:
            st.info("Контекста нет")
        else:
            st.caption("Первые 200 строк контекста (для контроля)")
            cview = ctx.copy()
            cview["values"] = cview["values"].apply(_safe_json)
            st.dataframe(cview.head(200), use_container_width=True, height=320)


if __name__ == "__main__":
    main()