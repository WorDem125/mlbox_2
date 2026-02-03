# LIBRARIES.md

Эта шпаргалка — **что за библиотека**, **зачем она нужна**, и **как быстро применить** на соревновании по трекам GPX + PostgreSQL + кластеризация + статистика распределений + карты + дашборд.

---

# Важно по зависимостям (если карта не покажется)

pip install folium streamlit-folium

## 0) Быстрый ориентир по ТЗ → какие блоки закрываем

**ТЗ:**
- Парсинг `.gpx` → гео/время/атрибуты
- Топографические карты из открытых источников
- Только PostgreSQL (есть сервер и учётки)
- Поиск/загрузка дополнительных треков (внешние источники)
- Кластерный анализ атрибутов + выделение значимых признаков
- Проверка нормальности, меры скошенности, тип распределения
- Дашборд с активным подключением к PostgreSQL
- Кластеризация + оценка качества

**Пакеты по блокам:**
- GPX + гео: `gpxpy`, `geopandas`, `shapely`, `pyproj`, `folium`
- Данные + статистика: `pandas`, `numpy`, `scipy`, `statsmodels`
- ML/кластеризация: `scikit-learn`, (опционально `catboost`)
- Хранилище/версии: `joblib`
- БД: `sqlalchemy`, `psycopg2-binary`
- Визуализация: `matplotlib`, `seaborn`, `plotly`, `folium`
- API/дашборд: `streamlit`, `flask`/`fastapi`, `uvicorn`, `pydantic`
- Утилиты: `requests`, `python-dotenv`, `tqdm`, `jupyter`, `ipykernel`

---

## 1) Базовые вычисления и табличные данные

### numpy
**Зачем:** быстрые массивы и математика.
**Где в ТЗ:** сезонность (sin/cos), лаги, нормализация, расчёт производных метрик.

**Мини-шаблоны:**
```python
import numpy as np
# сезонность по месяцу (1..12)
m = df['month'].dt.month
sin_m = np.sin(2*np.pi*m/12)
cos_m = np.cos(2*np.pi*m/12)
```

### pandas
**Зачем:** DataFrame, группировки, джойны, агрегации, time-series.
**Где в ТЗ:** сбор признаков треков в таблицу, агрегации, подготовка данных к кластеризации, отчёты.

**Мини-шаблоны:**
```python
import pandas as pd
# агрегация метрик по месяцам
monthly = (df.groupby('month')
             .agg(total=('value','sum'), cnt=('value','size'))
             .reset_index())

# джойн результатов кластеров назад
out = df.merge(labels_df[['track_id','cluster']], on='track_id', how='left')
```

### tqdm
**Зачем:** прогресс-бар для длинных операций.
**Где в ТЗ:** парсинг сотен GPX, вычисление атрибутов, загрузка в PostgreSQL.

**Мини-шаблоны:**
```python
from tqdm import tqdm
for path in tqdm(gpx_files):
    parse_one(path)
```

---

## 2) Статистика распределений и нормальность

### scipy
**Зачем:** статистические тесты, распределения.
**Где в ТЗ:** нормальность, скошенность, проверка распределений.

**Мини-шаблоны:**
```python
from scipy import stats
x = df['feature'].dropna().values

# скошенность/эксцесс
skew = stats.skew(x)
kurt = stats.kurtosis(x, fisher=True)

# Shapiro (лучше для небольших выборок)
W, p = stats.shapiro(x[:5000])  # ограничиваем для скорости

# D’Agostino (больше данных)
K2, p2 = stats.normaltest(x)
```

### statsmodels
**Зачем:** расширенная статистика (в т.ч. диагностические штуки).
**Где в ТЗ:** подтверждать выводы по распределениям/рядам, статистическая часть отчёта.

**Мини-шаблоны:**
```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors

x = df['feature'].dropna().values
stat, p = lilliefors(x)  # тест Лиллиефорса на нормальность
```

**Важно:** на соревновании не надо «идеально», надо **стабильно и объяснимо**: 
- показываем skew/kurt + 1–2 теста
- делаем вывод: нормально/не нормально, какой тип распределения похож

---

## 3) Машинное обучение: кластеризация и оценка качества

### scikit-learn
**Зачем:** всё для препроцессинга, кластеризации и метрик.
**Где в ТЗ:** основной ML блок.

**Ключевые штуки:**
- Предобработка: `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`
- Кластеры: `KMeans`, `DBSCAN`, `AgglomerativeClustering`
- Качество: `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`
- Отбор признаков/важность: `permutation_importance` (и корреляции через pandas)

**Мини-шаблон (универсальный пайплайн):**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

num = ['distance_km','elev_gain_m','avg_speed']
cat = ['surface_type']

pre = ColumnTransformer([
    ('num', StandardScaler(), num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
])

pipe = Pipeline([
    ('pre', pre),
    ('kmeans', KMeans(n_clusters=5, random_state=42, n_init='auto'))
])

X = df[num + cat]
labels = pipe.fit_predict(X)

# оценка
X_pre = pipe.named_steps['pre'].transform(X)
score = silhouette_score(X_pre, labels)
print('silhouette:', score)
```

### catboost
**Зачем:** сильная модель для табличных данных.
**Где в ТЗ:** пригодится, если дадут **задачу классификации/регрессии** (не только кластеры), либо для дополнительных сравнений.

**Мини-шаблон:**
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    depth=6, learning_rate=0.1, iterations=500,
    loss_function='MultiClass', verbose=100
)
model.fit(X_train, y_train, cat_features=cat_feature_indices)
```

### joblib
**Зачем:** сохранять модели/пайплайны/артефакты.
**Где в ТЗ:** версионирование моделей кластеров и прогнозов.

**Мини-шаблон:**
```python
import joblib
joblib.dump(pipe, 'models/cluster_pipe.joblib')
pipe = joblib.load('models/cluster_pipe.joblib')
```

---

## 4) GPX, геометрия и карты

### gpxpy
**Зачем:** читать `.gpx` (точки, треки, время, высота).
**Где в ТЗ:** «работаете с туристическими треками GPX».

**Мини-шаблон парсинга:**
```python
import gpxpy

with open(path, 'r', encoding='utf-8') as f:
    gpx = gpxpy.parse(f)

points = []
for track in gpx.tracks:
    for seg in track.segments:
        for p in seg.points:
            points.append((p.latitude, p.longitude, p.elevation, p.time))
```

### shapely
**Зачем:** геометрия (LineString) и операции.
**Где в ТЗ:** длина трека, геометрические операции, подготовка к GeoPandas.

**Мини-шаблон:**
```python
from shapely.geometry import LineString
line = LineString([(lon, lat) for lat, lon, *_ in points])
```

### pyproj
**Зачем:** проекции, чтобы длина считалась в метрах (а не «в градусах»).
**Где в ТЗ:** корректные расстояния/площади/анализ.

**Мини-шаблон (WGS84 → WebMercator):**
```python
from pyproj import Transformer
tr = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
coords_m = [tr.transform(lon, lat) for lat, lon, *_ in points]
line_m = LineString(coords_m)
distance_m = line_m.length
```

### geopandas
**Зачем:** GeoDataFrame (таблица + геометрия) и работа со слоями.
**Где в ТЗ:** хранение треков/слоёв, подготовка к картам.

**Мини-шаблон:**
```python
import geopandas as gpd
from shapely.geometry import LineString

gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
```

### folium
**Зачем:** интерактивные карты Leaflet.
**Где в ТЗ:** «рисовать топографические карты треков» (как минимум визуализация трека и слоёв).

**Мини-шаблон:**
```python
import folium
m = folium.Map(location=[lat0, lon0], zoom_start=12, tiles='OpenStreetMap')
folium.PolyLine([(lat, lon) for lat, lon, *_ in points], weight=4).add_to(m)
# в Streamlit: st.components.v1.html(m._repr_html_(), height=600)
```

**Про «топографические карты из открытых источников»:**
- На практике часто используют публичные тайлы (OSM/Topo) как фон.
- Если нужен «топо-стиль», ищут доступные tile-слои, но на соревновании главное: **трек на карте + корректные метрики**.

---

## 5) PostgreSQL (единственная СУБД по условиям)

### sqlalchemy
**Зачем:** единый способ подключиться к PostgreSQL и делать запросы.
**Где в ТЗ:** загрузка сырых треков/атрибутов/результатов кластеризации, чтение для дашборда.

**Мини-шаблон:**
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql+psycopg2://user:pass@host:5432/db')

df = pd.read_sql('SELECT * FROM tracks_features LIMIT 10', engine)
# df.to_sql('tracks_features', engine, if_exists='append', index=False)
```

### psycopg2-binary
**Зачем:** драйвер PostgreSQL (под капотом у sqlalchemy) и быстрые ручные запросы.
**Где в ТЗ:** когда надо выполнить DDL/insert быстро, или делать COPY.

---

## 6) API и дашборд

### streamlit
**Зачем:** быстрый UI (загрузка CSV/GPX, кнопки, таблицы, графики).
**Где в ТЗ:** дашборд с подключением к PostgreSQL и визуализациями.

**Мини-шаблон:**
```python
import streamlit as st
import pandas as pd

st.title('Track Dashboard')
file = st.file_uploader('Upload CSV', type=['csv'])
if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())
```

### streamlit-autorefresh
**Зачем:** автообновление страницы.
**Где в ТЗ:** если дашборд должен показывать «живые» данные из PostgreSQL.

### flask
**Зачем:** простой API (если нужно быстро).
**Где в ТЗ:** если потребуется API слой (например, прогноз/кластеризация по запросу).

### fastapi
**Зачем:** современный API с типами и автодоками.
**Где в ТЗ:** альтернатива Flask; удобно для конкурсной документации.

### pydantic
**Зачем:** схемы входных данных, валидация.
**Где в ТЗ:** фиксирует формат запросов/ответов (меньше ошибок).

### uvicorn
**Зачем:** ASGI сервер для FastAPI.

---

## 7) Внешние источники треков / загрузка данных

### requests
**Зачем:** HTTP-запросы.
**Где в ТЗ:** «ищите источники доп. треков» → быстро скачать по URL, дернуть API, загрузить данные.

**Мини-шаблон:**
```python
import requests
r = requests.get(url, timeout=30)
r.raise_for_status()
open('track.gpx','wb').write(r.content)
```

### python-dotenv
**Зачем:** хранить секреты (PG_HOST/PG_USER/PG_PASS) в `.env`, не в коде.
**Где в ТЗ:** чтобы безопасно подключаться к PostgreSQL.

**Мини-шаблон:**
```python
from dotenv import load_dotenv
import os
load_dotenv()
PG_HOST = os.getenv('PG_HOST')
```

---

## 8) Jupyter / разработка

### jupyter, ipykernel
**Зачем:** ноутбуки для исследования и отчётов.
**Где в ТЗ:** EDA, тестирование гипотез, построение графиков, прототипирование.

---

## 9) Визуализация (когда что выбрать)

- **matplotlib** — быстрые базовые графики (и в отчёт).
- **seaborn** — распределения/корреляции/boxplot «красиво и быстро».
- **plotly** — интерактивка (особенно в Streamlit).
- **folium** — интерактивная карта.

---

## 10) Чек-лист «чтобы не проиграть»

1) **Всегда приводим координаты к метрам** перед длиной (`pyproj`).
2) **Храним результаты в PostgreSQL**: сырые треки, признаки, кластеры, метрики.
3) Для кластеров считаем минимум 2 метрики качества (например, silhouette + Davies–Bouldin).
4) Для статистики: skew/kurt + 1–2 теста нормальности (и вывод словами).
5) В дашборде показываем:
   - карту трека,
   - таблицу признаков,
   - распределения,
   - кластеры и их описание,
   - качество кластеризации.

---

Если хочешь — следующим шагом сделаем **шаблон структуры проекта** под GPX→PostgreSQL→Features→Clustering→Dashboard, чтобы можно было копировать на любые соревнования.
