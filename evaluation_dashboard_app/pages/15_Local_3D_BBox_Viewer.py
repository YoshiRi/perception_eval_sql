"""Integrated local parquet-backed 3D bbox viewer."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from backend.local_bbox_api import DEFAULT_PORT, ensure_background_server
from lib.page_chrome import inject_app_page_styles, render_page_hero


st.set_page_config(
    layout="wide",
    page_title="Local 3D BBox Viewer",
    page_icon="▦",
    initial_sidebar_state="collapsed",
)
inject_app_page_styles()


def _running_behind_docker_nginx() -> bool:
    data_root = os.environ.get("EVAL_DASHBOARD_DATA_ROOT", "")
    return data_root.startswith("/app/") or Path("/app/docker-entrypoint.sh").exists()


def _api_base_url() -> str:
    configured = os.environ.get("LOCAL_BBOX_API_BROWSER_BASE_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    if _running_behind_docker_nginx():
        return "/bbox-api"
    return ensure_background_server("127.0.0.1", DEFAULT_PORT).rstrip("/")


render_page_hero(
    kicker="Local parquet viewer",
    title="Local BBox Viewer",
    description=(
        "A parquet-backed 3D bounding box viewer with scenario search, filters, playback, "
        "and camera controls."
    ),
    mode="Local",
)

api_base = _api_base_url()
direct_url = "/bbox-viewer/" if _running_behind_docker_nginx() else f"{api_base}/viewer"
explorer_url = "/bbox-explorer/" if _running_behind_docker_nginx() else f"{api_base}/explorer"

st.info(
    "The viewer opens in a separate page so the canvas, fullscreen controls, and keyboard/mouse input work normally."
)
col_a, col_b = st.columns(2)
with col_a:
    st.link_button("Open Dataset BBox Summary", explorer_url, type="primary")
with col_b:
    st.link_button("Open Local BBox Viewer", direct_url)
st.caption(f"Explorer URL: `{explorer_url}` · Viewer URL: `{direct_url}` · API: `{api_base}`")
