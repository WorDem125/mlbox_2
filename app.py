

"""Streamlit UI for MLBox_2

Goal:
- Simple college-grade UI that demonstrates API usage.
- Calls FastAPI service for health, models info, and predictions.

Run:
  streamlit run app.py

API should be running, e.g.:
  python -m uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st


# -------------------------
# Settings
# -------------------------
DEFAULT_API_BASE = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_S = 10


@dataclass
class ApiResponse:
    ok: bool
    status_code: int
    data: Any
    error: Optional[str] = None


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text


def api_get(base: str, path: str, params: Optional[Dict[str, Any]] = None) -> ApiResponse:
    url = f"{base.rstrip('/')}{path}"
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        data = _safe_json(r)
        return ApiResponse(ok=r.ok, status_code=r.status_code, data=data, error=None if r.ok else str(data))
    except Exception as e:
        return ApiResponse(ok=False, status_code=0, data=None, error=str(e))


def api_post(base: str, path: str, payload: Dict[str, Any]) -> ApiResponse:
    url = f"{base.rstrip('/')}{path}"
    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
        data = _safe_json(r)
        return ApiResponse(ok=r.ok, status_code=r.status_code, data=data, error=None if r.ok else str(data))
    except Exception as e:
        return ApiResponse(ok=False, status_code=0, data=None, error=str(e))


@st.cache_data(ttl=10)
def cached_health(base: str) -> ApiResponse:
    return api_get(base, "/health")


@st.cache_data(ttl=10)
def cached_models(base: str) -> ApiResponse:
    # optional endpoint; not all APIs have it
    return api_get(base, "/models")


def try_predict(base: str, lat: float, lon: float, radius_m: Optional[int]) -> Tuple[ApiResponse, Dict[str, Any]]:
    """Try a few common payload schemas so the UI survives minor API differences."""
    candidates = [
        {"lat": lat, "lon": lon},
        {"latitude": lat, "longitude": lon},
        {"lat": lat, "lon": lon, "radius_m": radius_m},
        {"latitude": lat, "longitude": lon, "radius_m": radius_m},
    ]

    last = ApiResponse(ok=False, status_code=0, data=None, error="No request made")
    used: Dict[str, Any] = {}

    for payload in candidates:
        # remove None fields to avoid validation errors
        payload = {k: v for k, v in payload.items() if v is not None}
        res = api_post(base, "/predict", payload)
        used = payload
        last = res
        if res.ok:
            return res, used

        # if endpoint doesn't exist, stop early
        if res.status_code == 404:
            return res, used

    return last, used


def pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="GPX Risk Demo", layout="wide")

st.title("GPX Risk Demo (API + Streamlit)")

with st.sidebar:
    st.header("API")
    api_base = st.text_input("API base URL", value=DEFAULT_API_BASE, help="Example: http://127.0.0.1:8000")

    st.markdown("---")
    st.header("Input")
    lat = st.number_input("Latitude", value=53.5680, format="%.6f")
    lon = st.number_input("Longitude", value=9.8962, format="%.6f")
    use_radius = st.checkbox("Use radius", value=False)
    radius_m = None
    if use_radius:
        radius_m = int(st.slider("Radius (m)", min_value=50, max_value=2000, value=400, step=50))

    st.markdown("---")
    auto_refresh = st.checkbox("Auto refresh health", value=True)
    refresh_s = int(st.slider("Refresh interval (s)", min_value=5, max_value=60, value=10, step=5))


# Top status row
col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

with col1:
    st.subheader("Service status")
    if auto_refresh:
        # trigger periodic refresh
        st.cache_data.clear()
        time.sleep(0.05)
    health = cached_health(api_base)
    if health.ok:
        st.success(f"UP (HTTP {health.status_code})")
    else:
        st.error(f"DOWN ({health.error})")

    with st.expander("/health response", expanded=False):
        st.code(pretty(health.data), language="json")

with col2:
    st.subheader("Models")
    models = cached_models(api_base)
    if models.ok:
        st.success("Loaded")
        with st.expander("/models response", expanded=False):
            st.code(pretty(models.data), language="json")
    else:
        # Not fatal: endpoint may not exist
        st.info("Endpoint /models not available or error")
        with st.expander("Details", expanded=False):
            st.code(pretty({"status": models.status_code, "error": models.error}), language="json")

with col3:
    st.subheader("How it works")
    st.write(
        "This UI sends coordinates to the FastAPI service. "
        "The service uses the latest saved ML models to predict: "
        "(1) risk_label (none/flood/fire) and (2) evac_label (easy/medium/hard). "
        "Continuous learning happens on the API side; Streamlit only calls endpoints."
    )

st.markdown("---")

# Prediction section
st.header("Prediction")

btn_col1, btn_col2 = st.columns([1, 3])

with btn_col1:
    do_predict = st.button("Predict", type="primary")

if do_predict:
    with st.spinner("Calling /predict ..."):
        res, used_payload = try_predict(api_base, float(lat), float(lon), radius_m)

    st.caption(f"Request payload used: {used_payload}")

    if res.ok:
        st.success("OK")
        st.code(pretty(res.data), language="json")

        # Try to show key fields if present
        if isinstance(res.data, dict):
            r = res.data.get("risk", res.data.get("risk_label", res.data.get("risk_prediction")))
            e = res.data.get("evac", res.data.get("evac_label", res.data.get("evac_prediction")))
            conf = res.data.get("confidence", res.data.get("proba"))

            k1, k2, k3 = st.columns(3)
            k1.metric("Risk", "-" if r is None else str(r))
            k2.metric("Evac", "-" if e is None else str(e))
            k3.metric("Confidence", "-" if conf is None else str(conf))

    else:
        st.error(f"Request failed (HTTP {res.status_code})")
        st.code(pretty({"error": res.error, "response": res.data}), language="json")


# Auto refresh note
if auto_refresh:
    st.caption(f"Auto refresh enabled: every ~{refresh_s}s (manual refresh occurs when page reruns)")
    time.sleep(min(refresh_s, 5))
    st.rerun()