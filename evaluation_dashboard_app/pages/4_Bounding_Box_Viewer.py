import html
import duckdb
import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from lib.path_utils import path_display
from lib.parquet_schema import schema_flags
from lib.page_chrome import inject_app_page_styles, render_loaded_data_section, render_page_hero
from lib.overview_url_hydrate import try_hydrate_session_from_overview_query_params
from lib.ui.bounding_box_viewer_ui import bev_overlay_line_and_status_legend_markup, bev_status_legend_markup
from lib.ui.theme import apply_plotly_theme, is_dark, pick, tokens
from lib.t4_dataset_embed import t4_share_query_params
from lib.t4_three_layers import resolve_t4_dataset_id, resolve_t4_scenario
from lib.t4_visualizer_client import (
    DEFAULT_BASE_URL,
    ENV_BASE_URL,
    RenderRequest,
    TargetObjectIn,
    T4VisualizerClient,
    T4VisualizerError,
    browser_base_url,
    format_t4_visualizer_error,
    target_object_from_gt_row,
)

st.set_page_config(
    layout="wide",
    page_title="Bounding Box Viewer",
    page_icon="🖼️",
    initial_sidebar_state="expanded",
)
inject_app_page_styles()

# Pre-dark-theme light palette: the exact colors this page's charts used before the dark
# theme existed. Light mode must keep rendering these; dark uses the token-derived values.
_LEGACY_BOX_FALLBACK = "#999999"
_LEGACY_IFRAME_BG = "#e2e8f0"
_LEGACY_VELOCITY_LINE = "rgba(100,100,100,0.7)"
_LEGACY_EGO_LINE = "black"
_LEGACY_EGO_FILL = "gray"
_LEGACY_RUN_PROXY_LINE = "#555555"
_LEGACY_MARKER_OUTLINE_WHITE = "white"
_LEGACY_VLINE = "black"
_LEGACY_TRAJ_LINE = "gray"
_LEGACY_TRAJ_CURRENT = "red"
_LEGACY_TRAJ_FN = "orange"
_LEGACY_TRAJ_TP = "green"
_LEGACY_TRAJ_POINT_OUTLINE = "black"


def _theme_chart(fig):
    """Token Plotly theme on dark; on light leave the figure with its pre-dark-theme defaults."""
    if is_dark():
        apply_plotly_theme(fig)
    return fig


# =============================
# Session state from Overview (run path)
# =============================
try_hydrate_session_from_overview_query_params()
if "runA" not in st.session_state:
    st.warning("Please load data from the **Overview** page first (select mode and run(s)).")
    st.stop()

runA = st.session_state["runA"]
mode = st.session_state.get("mode", "Single Mode")
# Respect mode: in Single Mode always show one run; only use compare state when explicitly in Compare Mode
if mode == "Compare Mode":
    all_runs = st.session_state.get("all_runs")
    run_labels_state = st.session_state.get("run_labels")
    if all_runs and run_labels_state and len(all_runs) >= 2:
        runs = all_runs
        run_labels_list = run_labels_state
    else:
        runB = st.session_state.get("runB")
        runs = [runA] if runB is None else [runA, runB]
        run_labels_list = ["A"] if len(runs) == 1 else ["A", "B"]
else:
    runs = [runA]
    run_labels_list = ["A"]


def list_parquets_in_run(run_path) -> List[str]:
    """Return sorted list of absolute paths to .parquet files in the run directory."""
    p = Path(run_path)
    if not p.is_dir():
        return []
    return sorted([str(f.resolve()) for f in p.glob("*.parquet")])


# Parquet files from the run(s) designated on Overview
parquet_lists = [list_parquets_in_run(r["path"]) for r in runs]
for i, (r, pl) in enumerate(zip(runs, parquet_lists)):
    if not pl:
        lbl = run_labels_list[i] if i < len(run_labels_list) else str(i)
        st.error(
            f"No parquet files in run ({lbl}): {path_display(r['path'])}. "
            "Add a .parquet file or generate one from the Download page."
        )
        st.stop()

multi_run = len(runs) >= 2

# ----------------------------
# Loaded Runs (from Overview) + hero
# ----------------------------
_ld_entries = []
for i, r in enumerate(runs):
    lbl = run_labels_list[i] if i < len(run_labels_list) else str(i)
    if lbl == "A":
        _ltitle = "Baseline · A"
    else:
        _ltitle = f"Candidate · {lbl}"
    _ld_entries.append((_ltitle, path_display(r["path"])))
render_loaded_data_section(_ld_entries)
render_page_hero(
    kicker="Bounding boxes",
    title="Bounding box & BEV viewer",
    description=(
        "Inspect frames from parquet: camera overlays, BEV view, and optional multi-run comparison "
        "when several evaluations are loaded from Overview."
    ),
    mode=mode,
)

# ----------------------------
# Sidebar (Filters)
# ----------------------------
with st.sidebar:
    st.markdown("##### Filters")
    st.caption("Parquet file, suite, scenario, frame — narrow down what you visualize.")

    if multi_run:
        runs_to_show = st.multiselect(
            "Runs to show",
            run_labels_list,
            default=run_labels_list,
            key="bbox_viewer_runs_to_show",
        )
        if not runs_to_show:
            st.warning("Select at least one run.")
            st.stop()
    else:
        runs_to_show = run_labels_list

    # Parquet file selection per run (only for runs that are shown)
    selected_files = {}
    for i, lbl in enumerate(run_labels_list):
        if lbl not in runs_to_show:
            continue
        pl = parquet_lists[i]
        if len(pl) == 1:
            selected_files[lbl] = pl[0]
        else:
            selected_files[lbl] = st.selectbox(
                f"File (Run {lbl})",
                pl,
                format_func=os.path.basename,
                key=f"bbox_viewer_file_{lbl}",
            )

    # Primary file for building filter options (suite, scenario, topic, labels)
    first_shown = runs_to_show[0] if runs_to_show else run_labels_list[0]
    filter_file = selected_files.get(first_shown) or parquet_lists[run_labels_list.index(first_shown)][0]

# DuckDB connection (no cache = 安定優先)
con = duckdb.connect()

# --- Columns (for visibility existence check) — use filter_file for filter options
cols = con.execute("DESCRIBE SELECT * FROM parquet_scan(?)", [filter_file]).df()["column_name"].tolist()
has_visibility = "visibility" in cols
has_suite_name = "suite_name" in cols
has_scenario_name = "scenario_name" in cols
has_t4dataset_name = "t4dataset_name" in cols
schema = schema_flags(con, filter_file)
# Optional columns for hover (z, height, vx, vy, confidence, pointcloud_num)
hover_extra_cols = [c for c in ["z", "height", "vx", "vy", "confidence", "pointcloud_num"] if c in cols]


# --- Scene selection: one suite + one scenario (when columns exist)
scene_where = "1=1"
scene_params: List[str] = [filter_file]

if has_suite_name:
    suite_list = con.execute(
        "SELECT DISTINCT suite_name AS v FROM parquet_scan(?) WHERE suite_name IS NOT NULL ORDER BY v",
        [filter_file]
    ).df()["v"].dropna().astype(str).tolist()
else:
    suite_list = []

# Apply deep-link from Detection Stats (suite / scenario / t4dataset) before selectboxes render
if "bbox_viewer_link_suite" in st.session_state:
    _lsu = st.session_state.pop("bbox_viewer_link_suite", None)
    if suite_list and _lsu is not None and str(_lsu) in suite_list:
        st.session_state["bbox_viewer_suite"] = str(_lsu)

with st.sidebar:
    selected_suite = None
    selected_scenario = None
    if suite_list:
        selected_suite = st.selectbox(
            "Suite name",
            suite_list,
            key="bbox_viewer_suite",
        )
    if has_scenario_name:
        if selected_suite is not None:
            scenario_list = con.execute(
                "SELECT DISTINCT scenario_name AS v FROM parquet_scan(?) WHERE suite_name = ? AND scenario_name IS NOT NULL ORDER BY v",
                [filter_file, selected_suite]
            ).df()["v"].dropna().astype(str).tolist()
        else:
            scenario_list = con.execute(
                "SELECT DISTINCT scenario_name AS v FROM parquet_scan(?) WHERE scenario_name IS NOT NULL ORDER BY v",
                [filter_file]
            ).df()["v"].dropna().astype(str).tolist()
        if scenario_list:
            if "bbox_viewer_link_scenario" in st.session_state:
                _lsc = st.session_state.pop("bbox_viewer_link_scenario", None)
                if _lsc is not None and str(_lsc) in scenario_list:
                    st.session_state["bbox_viewer_scenario"] = str(_lsc)
            selected_scenario = st.selectbox(
                "Scenario name",
                scenario_list,
                key="bbox_viewer_scenario",
            )
    # Only offer t4dataset_name filter when column exists and has more than one distinct value
    # Filter t4dataset options by selected_suite and selected_scenario when set
    t4dataset_list: List[str] = []
    if has_t4dataset_name:
        t4_where_parts = ["t4dataset_name IS NOT NULL"]
        t4_params: List[Any] = [filter_file]
        if selected_suite is not None:
            t4_where_parts.insert(0, "suite_name = ?")
            t4_params.append(selected_suite)
        if selected_scenario is not None:
            t4_where_parts.insert(0, "scenario_name = ?")
            t4_params.insert(1, selected_scenario)
        t4_where = " AND ".join(t4_where_parts)
        t4dataset_list = con.execute(
            f"SELECT DISTINCT t4dataset_name AS v FROM parquet_scan(?) WHERE {t4_where} ORDER BY v",
            t4_params,
        ).df()["v"].dropna().astype(str).tolist()
    has_multiple_t4dataset = len(t4dataset_list) > 1
    selected_t4dataset = None
    if has_multiple_t4dataset and t4dataset_list:
        if "bbox_viewer_link_t4dataset" in st.session_state:
            _lt4 = st.session_state.pop("bbox_viewer_link_t4dataset", None)
            if _lt4 is not None and str(_lt4) in t4dataset_list:
                st.session_state["bbox_viewer_t4dataset"] = str(_lt4)
        selected_t4dataset = st.selectbox(
            "t4dataset_name",
            t4dataset_list,
            key="bbox_viewer_t4dataset",
        )

# Build scene filter for queries (one scene = one suite + one scenario)
if selected_suite is not None:
    scene_where = "suite_name = ?"
    scene_params = [filter_file, selected_suite]
if selected_scenario is not None:
    scene_where = scene_where + " AND scenario_name = ?" if scene_where != "1=1" else "scenario_name = ?"
    scene_params = scene_params + [selected_scenario]
if selected_t4dataset is not None:
    scene_where = scene_where + " AND t4dataset_name = ?" if scene_where != "1=1" else "t4dataset_name = ?"
    scene_params = scene_params + [selected_t4dataset]
if scene_where == "1=1":
    scene_params = [filter_file]

# Build a load-scene WHERE that excludes t4dataset_name so each run's parquet is
# queried without the single-dataset restriction. t4dataset_name differs between
# baseline and candidate when comparing different releases.
_load_scene_where = scene_where
_load_scene_params = list(scene_params)
if selected_t4dataset is not None:
    # Remove the "t4dataset_name = ?" clause (always the last filter added).
    if _load_scene_where == "t4dataset_name = ?":
        _load_scene_where = "1=1"
        _load_scene_params = [filter_file]
    else:
        _load_scene_where = _load_scene_where.replace(" AND t4dataset_name = ?", "")
        _load_scene_params = _load_scene_params[:-1]

# --- topic_name（単一選択）
topic_names = con.execute(
    f"SELECT DISTINCT topic_name AS v FROM parquet_scan(?) WHERE {scene_where} ORDER BY v",
    scene_params
).df()["v"].dropna().tolist()
if not topic_names:
    # Clear scene from link so selectbox falls back to first available (avoids "No topic_name" loop)
    for key in (
        "bbox_viewer_scenario",
        "bbox_viewer_suite",
        "bbox_viewer_link_suite",
        "bbox_viewer_link_scenario",
        "bbox_viewer_link_t4dataset",
    ):
        if key in st.session_state:
            del st.session_state[key]
    st.warning(
        "No topic_name for the selected scene (from Detection Stats link). "
        "Cleared scene selection; please choose a scene from the sidebar."
    )
    st.rerun()

with st.sidebar:
    selected_topic = st.selectbox("topic_name (single)", topic_names)

# --- label（複数選択）
labels = con.execute(
    f"SELECT DISTINCT label AS v FROM parquet_scan(?) WHERE {scene_where} AND topic_name=? ORDER BY v",
    scene_params + [selected_topic]
).df()["v"].dropna().tolist()
if not labels:
    st.warning("No label for selected topic.")
    st.stop()

with st.sidebar:
    selected_labels = st.multiselect("label(s)", labels, default=labels)

# --- visibility（列があるときだけ。NULLは UNKNOWN で扱う）
selected_visibility = None
if has_visibility:
    vis_list = con.execute(
        f"SELECT DISTINCT COALESCE(visibility,'UNKNOWN') AS v FROM parquet_scan(?) WHERE {scene_where} AND topic_name=? ORDER BY v",
        scene_params + [selected_topic]
    ).df()["v"].tolist()
    with st.sidebar:
        if vis_list:
            selected_visibility = st.multiselect("visibility", vis_list, default=vis_list)
        else:
            st.info("No visibility values found — skipping.")
else:
    with st.sidebar:
        st.info("No 'visibility' column found — skipping visibility filter.")

# Guard
if not selected_labels:
    st.warning("No label selected.")
    st.stop()

# --- invalidオブジェクト表示オプション ---
with st.sidebar:
    show_invalid = st.checkbox("Show invalid (zero-size) objects", value=False)
    show_velocity_arrows = False
    if schema.get("has_velocity"):
        show_velocity_arrows = st.checkbox("Show velocity vectors", value=False, help="Draw arrows for vx, vy (scale: 2 s)")

# --- Comparison view mode (when multiple runs) ---
compare_view_mode = "side_by_side"
if multi_run and len(runs_to_show) >= 2:
    compare_view_mode = st.sidebar.radio(
        "Comparison view",
        ["Side by side", "Overlay (both runs on one BEV)"],
        index=0,
        key="bbox_compare_view_mode",
        help="Side by side: two BEV plots for direct comparison. Overlay: single BEV with both runs drawn together to see differences at a glance.",
    )
    compare_view_mode = "overlay" if "Overlay" in compare_view_mode else "side_by_side"

# --- T4 visualizer (base URL + preview mode in sidebar)
with st.sidebar:
    st.markdown("##### T4 visualizer")
    st.caption("Uses **GET /datasets/{id}/availability** first; preview runs only if the server reports the dataset is available.")
    if "bbox_t4_base_url" not in st.session_state:
        st.session_state["bbox_t4_base_url"] = (
            (os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        )
    st.text_input(
        "T4 server base URL",
        key="bbox_t4_base_url",
        help=f"Server-side API URL. Default from env `{ENV_BASE_URL}`. Browser iframes use `T4_VISUALIZER_BROWSER_BASE_URL` when set.",
    )
    _t4_mode = st.radio(
        "T4 preview",
        ["html_iframe", "post_png"],
        format_func=lambda m: (
            "HTML iframe (/render/html)" if m == "html_iframe" else "POST /render (PNGs here)"
        ),
        key="bbox_t4_preview_mode",
        horizontal=True,
    )
    if _t4_mode == "post_png":
        _t4p1, _t4p2 = st.columns(2)
        with _t4p1:
            st.checkbox("Crop cameras", value=True, key="bbox_t4_crop_cameras")
            st.checkbox("Show dataset annotations", value=True, key="bbox_t4_show_ann")
        with _t4p2:
            st.checkbox("Draw GT rows as target boxes", value=True, key="bbox_t4_overlay_gt")


# ----------------------------
# Build query safely & load data
# ----------------------------
# Use _load_scene_where (without t4dataset_name) so each run's parquet is
# queried independently — baseline and candidate may have different dataset names.
where = [_load_scene_where, "topic_name = ?"]  # topic_name は単一選択
params = _load_scene_params + [selected_topic]

# label IN (...)
where.append(f"label IN ({','.join(['?']*len(selected_labels))})")
params.extend(selected_labels)

# visibility（ある場合のみ、NULLは UNKNOWN で比較）
select_vis = ", visibility" if has_visibility else ""
if has_visibility and selected_visibility:
    where.append(f"COALESCE(visibility,'UNKNOWN') IN ({','.join(['?']*len(selected_visibility))})")
    params.extend(selected_visibility)

select_extras = (", " + ", ".join(hover_extra_cols)) if hover_extra_cols else ""
# Optional columns for T4 server overlay (z/height) and resolving dataset / scenario per row
_geom_for_t4 = [c for c in ("z", "height") if c in cols and c not in hover_extra_cols]
_geom_select = (", " + ", ".join(_geom_for_t4)) if _geom_for_t4 else ""
_t4_meta_cols = [c for c in ("t4dataset_id", "t4dataset_name", "scenario_name") if c in cols]
_t4_meta_select = (", " + ", ".join(_t4_meta_cols)) if _t4_meta_cols else ""
sql = f"""
SELECT frame_index, x, y, length, width, yaw, label, topic_name, source, status, uuid
{select_vis}{select_extras}{_geom_select}{_t4_meta_select}
FROM parquet_scan(?)
WHERE {" AND ".join(where)}
ORDER BY frame_index
"""

# Build list of (file, run_label) to load
files_to_load: List[Tuple[str, str]] = [(selected_files[lbl], lbl) for lbl in runs_to_show if lbl in selected_files]

# Base params after the file (suite, scenario, topic, labels, visibility)
base_params = _load_scene_params[1:] + [selected_topic] + list(selected_labels)
if has_visibility and selected_visibility:
    base_params = base_params + list(selected_visibility)

dfs = []
for file_path, run_label in files_to_load:
    params = [file_path] + base_params
    df_part = con.execute(sql, params).df()
    if not df_part.empty:
        df_part = df_part.copy()
        df_part["run"] = run_label
        dfs.append(df_part)

if not dfs:
    st.warning("No data matches the selected filters.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# Debug: show loaded data summary per run and source
with st.expander("Data load debug", expanded=False):
    for _rn in df["run"].unique() if "run" in df.columns else ["(single)"]:
        _rdf = df[df["run"] == _rn] if "run" in df.columns else df
        _gt = int((_rdf["source"] == "GT").sum()) if "source" in _rdf.columns else 0
        _est = int((_rdf["source"] == "EST").sum()) if "source" in _rdf.columns else 0
        _src_vals = _rdf["source"].unique().tolist() if "source" in _rdf.columns else []
        st.write(f"Run **{_rn}**: {len(_rdf)} rows, GT={_gt}, EST={_est}, source_values={_src_vals}, frames={_rdf['frame_index'].nunique() if 'frame_index' in _rdf.columns else 'N/A'}")
# When only one run, drop "run" column so rest of code unchanged (optional; we can keep it as "A" or "B")
if len(files_to_load) == 1:
    df["run"] = df["run"].iloc[0]  # keep column for uniform legend logic

# frame_index を int に（比較安定化）
if "frame_index" in df.columns and not np.issubdtype(df["frame_index"].dtype, np.integer):
    df["frame_index"] = (
        pd.to_numeric(df["frame_index"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

# Use full df for stats; when both runs are shown, stats are computed per run and displayed side by side
df_stats = df

# ----------------------------
# Color map
# ----------------------------
# Box hues stay fixed in both themes: they mirror the 3D/BEV viewers (static/bbox_viewer.css --sw-*)
# and the status legend chips in lib/ui/bounding_box_viewer_ui.py. Everything else on the page uses tokens.
color_map = {
    ("GT", "TP"): "#00cc66",   # 緑
    ("GT", "FN"): "#ff9933",   # オレンジ
    ("EST", "TP"): "#66b3ff",  # 青
    ("EST", "FP"): "#ff6666",  # 赤
}
def get_color(source, status): return color_map.get((source, status), pick(_LEGACY_BOX_FALLBACK, tokens()["muted"]))

# ----------------------------
# Currently showing & comparison hint
# ----------------------------
if len(files_to_load) == 1:
    st.info(f"**Currently showing:** Run {files_to_load[0][1]} only")
else:
    run_names = [f[1] for f in files_to_load]
    st.info(f"**Currently showing:** Runs {', '.join(run_names)}")
    if compare_view_mode == "side_by_side":
        left_lbl = run_names[0]
        right_lbl = run_names[1] if len(run_names) > 1 else ""
        hint = f"**Comparison:** Left = **Run {left_lbl}** (baseline), Right = **Run {right_lbl}** (candidate). Same frame and scale — compare object counts and positions."
        if len(run_names) > 2:
            hint = f"**Comparison:** Columns from left to right = **Run {', '.join(run_names)}**. Same frame and scale — compare object counts and positions."
        st.markdown(f":information_source: {hint}")
    else:
        line_legend = " — ".join(f"**{s}** = Run {run_names[i]}" for i, s in enumerate(["solid", "dashed", "dot", "dashdot"][: len(run_names)]))
        st.markdown(f":information_source: **Overlay view:** All runs on one BEV: {line_legend}. Differences show where detections disagree.")

# ----------------------------
# Frame slider (support pre-set from Detection Stats link)
# ----------------------------
if "bbox_viewer_frame_index" in st.session_state:
    try:
        requested = int(st.session_state["bbox_viewer_frame_index"])
        st.session_state["bbox_viewer_frame"] = max(
            int(df.frame_index.min()),
            min(int(df.frame_index.max()), requested),
        )
    except (TypeError, ValueError):
        st.session_state["bbox_viewer_frame"] = int(df.frame_index.min())
    del st.session_state["bbox_viewer_frame_index"]
f_min, f_max = int(df.frame_index.min()), int(df.frame_index.max())
frame = st.slider(
    "Frame index",
    f_min,
    f_max,
    value=st.session_state.get("bbox_viewer_frame", f_min),
    step=1,
    key="bbox_viewer_frame",
)
df_frame = df[df.frame_index == frame]

total_records = len(df_frame)
valid_records = int(((df_frame["length"] > 0) & (df_frame["width"] > 0)).sum())

# Current-frame KPI strip (TP / FN / FP for this frame)
gt_frame = df_frame[df_frame["source"] == "GT"]
est_frame = df_frame[df_frame["source"] == "EST"]
tp_count = int(((gt_frame["status"] == "TP").sum()))
fn_count = int(((gt_frame["status"] == "FN").sum()))
fp_count = int(((est_frame["status"] == "FP").sum()))
tp_est_count = int(((est_frame["status"] == "TP").sum()))
gt_total = tp_count + fn_count
tpr_frame = (tp_count / gt_total) if gt_total > 0 else None
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("TP (this frame)", tp_count)
with k2: st.metric("FN (this frame)", fn_count)
with k3: st.metric("FP (this frame)", fp_count)
with k4: st.metric("TP (EST)", tp_est_count)
with k5: st.metric("TPR", f"{tpr_frame:.2%}" if tpr_frame is not None else "—")

# ----------------------------
# T4 visualizer (HTTP server): camera PNGs for current frame
# ----------------------------
def _bbox_t4_request_key(
    ds: str,
    sc: str,
    frame_idx: int,
    base_url: str,
    crop: bool,
    show_ann: bool,
    overlay_gt: bool,
) -> Tuple[Any, ...]:
    return (
        str(ds),
        str(sc),
        int(frame_idx),
        str(base_url).rstrip("/"),
        bool(crop),
        bool(show_ann),
        bool(overlay_gt),
    )


_t4_preview_mode = st.session_state.get("bbox_t4_preview_mode", "html_iframe")

base_url_t4 = (st.session_state.get("bbox_t4_base_url") or "").strip() or DEFAULT_BASE_URL
browser_url_t4 = browser_base_url(base_url_t4)

_ds_t4 = resolve_t4_dataset_id(df_frame)
if not _ds_t4 and selected_t4dataset is not None:
    _ds_t4 = str(selected_t4dataset)
_sc_t4 = resolve_t4_scenario(df_frame, selected_scenario)

if not _ds_t4:
    for _k in (
        "bbox_t4_last_images",
        "bbox_t4_last_meta",
        "bbox_t4_success_key",
        "bbox_t4_error_key",
        "bbox_t4_error_msg",
        "bbox_t4_availability",
    ):
        st.session_state.pop(_k, None)
    st.caption("T4 camera preview is not available for this scene.")
    with st.expander("Details", expanded=False):
        st.markdown(
            "Needs parquet **t4dataset_id** or **t4dataset_name** (or **t4dataset_name** in the sidebar when "
            "multiple datasets exist). "
            "The Tier4 HTTP visualizer (`t4-server`) must serve that dataset. "
            f"Set **T4 server base URL** in the sidebar or `{ENV_BASE_URL}`."
        )
else:
    _t4_avail_cache_key = f"{base_url_t4.rstrip('/')}|{_ds_t4}"
    _cached_av = st.session_state.get("bbox_t4_availability")
    _need_avail_fetch = _cached_av is None or _cached_av.get("cache_key") != _t4_avail_cache_key
    if _need_avail_fetch:
        try:
            with st.spinner("Checking T4 dataset on the server…"):
                _av_client = T4VisualizerClient(base_url=base_url_t4, timeout=30.0)
                _av_data = _av_client.dataset_availability(_ds_t4)
            st.session_state["bbox_t4_availability"] = {
                "cache_key": _t4_avail_cache_key,
                "ok": True,
                "available": bool(_av_data.get("available")),
                "data": _av_data,
                "error": None,
            }
        except T4VisualizerError as ex:
            st.session_state["bbox_t4_availability"] = {
                "cache_key": _t4_avail_cache_key,
                "ok": False,
                "available": False,
                "data": None,
                "error": format_t4_visualizer_error(ex),
            }
        except (OSError, requests.RequestException) as ex:
            st.session_state["bbox_t4_availability"] = {
                "cache_key": _t4_avail_cache_key,
                "ok": False,
                "available": False,
                "data": None,
                "error": f"Network error: {ex}",
            }
        except Exception as ex:
            st.session_state["bbox_t4_availability"] = {
                "cache_key": _t4_avail_cache_key,
                "ok": False,
                "available": False,
                "data": None,
                "error": f"Availability check failed: {ex}",
            }

    _av = st.session_state.get("bbox_t4_availability") or {}

    if not _av.get("ok"):
        st.caption("T4 preview skipped — could not verify dataset on the visualizer server.")
        with st.expander("Details", expanded=False):
            st.markdown(_av.get("error") or "Unknown error.")
    elif not _av.get("available"):
        st.caption("T4 preview skipped — this dataset is not on the visualizer server host.")
        with st.expander("Details", expanded=False):
            _d = _av.get("data")
            if isinstance(_d, dict) and _d:
                st.json(_d)
            else:
                st.markdown(
                    "The server reported **available: false** (no local dataset path for this id on the machine "
                    "running `t4-server`)."
                )
    else:
        _q_three = t4_share_query_params(_ds_t4, _sc_t4, int(frame))
        _viewer_three_url = f"{browser_url_t4.rstrip('/')}/viewer/three?{_q_three}"
        st.caption("**3D viewer** (Three.js, GT / pred / matched layers) lives on a dedicated page.")
        c3d_a, c3d_b = st.columns([1, 2])
        with c3d_a:
            st.page_link("pages/5_T4_3D_Viewer.py", label="Open T4 3D Viewer", icon="🧊")
        with c3d_b:
            st.markdown(f"[Open `/viewer/three` in new tab]({_viewer_three_url})")

    if not _av.get("ok") or not _av.get("available"):
        pass
    elif _t4_preview_mode == "html_iframe":
        _q = t4_share_query_params(_ds_t4, _sc_t4, int(frame))
        _render_html_url = f"{browser_url_t4.rstrip('/')}/render/html?{_q}"
        st.markdown(f"[Open in new tab]({_render_html_url})")
        _iframe_h = 900
        # Iframe shell: neutral surface while the document loads (avoid a hard dark fill — it reads as a black
        # box for ~2s until the large /render/html response paints; inner page still sets its own background).
        # The iframe is its own document, so the token value is inlined instead of using var(--t4-*).
        components.html(
            f'<iframe src="{html.escape(_render_html_url, quote=True)}" '
            f'width="100%" height="{_iframe_h}" style="border:none;border-radius:8px;background:{pick(_LEGACY_IFRAME_BG, tokens()["surface_3"])}" '
            f'loading="lazy" title="T4 camera render" referrerpolicy="no-referrer-when-downgrade"></iframe>',
            height=_iframe_h + 24,
            scrolling=True,
        )
    elif not _sc_t4:
        st.caption("POST /render mode needs **scenario_name** (sidebar or parquet) for this scene.")
        with st.expander("Details", expanded=False):
            st.markdown(
                "Pick a **Scenario name** in the sidebar or ensure parquet includes **scenario_name**. "
                "Alternatively switch to **HTML iframe** mode if the server accepts an empty scenario for your dataset."
            )
    else:
        t4_crop = bool(st.session_state.get("bbox_t4_crop_cameras", True))
        t4_show_ann = bool(st.session_state.get("bbox_t4_show_ann", True))
        t4_overlay_gt = bool(st.session_state.get("bbox_t4_overlay_gt", True))

        _req_key = _bbox_t4_request_key(
            _ds_t4,
            _sc_t4,
            int(frame),
            base_url_t4,
            t4_crop,
            t4_show_ann,
            t4_overlay_gt,
        )
        _ok_key = st.session_state.get("bbox_t4_success_key")
        _bad_key = st.session_state.get("bbox_t4_error_key")

        _should_fetch = _req_key != _ok_key and _req_key != _bad_key

        if _should_fetch:
            try:
                with st.spinner("Loading T4 camera renders… (usually ~2 seconds)"):
                    client = T4VisualizerClient(
                        base_url=base_url_t4,
                        timeout=120.0,
                    )
                    targets = []
                    if t4_overlay_gt:
                        for _, row in df_frame[df_frame["source"] == "GT"].iterrows():
                            d = target_object_from_gt_row(row.to_dict())
                            targets.append(TargetObjectIn(**d))
                    req = RenderRequest(
                        t4dataset_id=_ds_t4,
                        scenario_name=_sc_t4,
                        frame_index=int(frame),
                        target_objects=targets,
                        crop_cameras=t4_crop,
                        show_annotations=t4_show_ann,
                    )
                    t4_res = client.render(req)
                    _imgs = t4_res.decode_all_images()
                if not _imgs:
                    st.session_state.pop("bbox_t4_last_images", None)
                    st.session_state.pop("bbox_t4_last_meta", None)
                    st.session_state["bbox_t4_error_key"] = _req_key
                    st.session_state["bbox_t4_error_msg"] = (
                        "T4 server returned no camera images for this frame. "
                        "Check that the dataset and scenario exist on the server and the frame index is valid."
                    )
                    st.session_state.pop("bbox_t4_success_key", None)
                else:
                    st.session_state["bbox_t4_last_images"] = _imgs
                    st.session_state["bbox_t4_last_meta"] = {
                        "sample_token": t4_res.sample_token,
                        "timestamp_us": t4_res.timestamp_us,
                        "frame_index": int(frame),
                        "t4dataset_id": _ds_t4,
                        "scenario_name": _sc_t4,
                    }
                    st.session_state["bbox_t4_success_key"] = _req_key
                    st.session_state.pop("bbox_t4_error_key", None)
                    st.session_state.pop("bbox_t4_error_msg", None)
            except T4VisualizerError as ex:
                st.session_state.pop("bbox_t4_last_images", None)
                st.session_state.pop("bbox_t4_last_meta", None)
                st.session_state.pop("bbox_t4_success_key", None)
                st.session_state["bbox_t4_error_key"] = _req_key
                st.session_state["bbox_t4_error_msg"] = format_t4_visualizer_error(ex)
            except (OSError, requests.RequestException) as ex:
                st.session_state.pop("bbox_t4_last_images", None)
                st.session_state.pop("bbox_t4_last_meta", None)
                st.session_state.pop("bbox_t4_success_key", None)
                st.session_state["bbox_t4_error_key"] = _req_key
                st.session_state["bbox_t4_error_msg"] = f"Network error: {ex}"
            except Exception as ex:
                st.session_state.pop("bbox_t4_last_images", None)
                st.session_state.pop("bbox_t4_last_meta", None)
                st.session_state.pop("bbox_t4_success_key", None)
                st.session_state["bbox_t4_error_key"] = _req_key
                st.session_state["bbox_t4_error_msg"] = f"T4 render failed: {ex}"

        _meta = st.session_state.get("bbox_t4_last_meta")
        _imgs = st.session_state.get("bbox_t4_last_images")
        _show_err = st.session_state.get("bbox_t4_error_msg")

        st.caption(
            f"**Request:** t4dataset_id `{_ds_t4}` · scenario_name `{_sc_t4}` · frame_index `{frame}`"
        )
        if _req_key == st.session_state.get("bbox_t4_error_key") and _show_err:
            st.caption("T4 camera preview could not be loaded.")
            with st.expander("Details", expanded=False):
                st.caption(
                    f"t4dataset_id `{_ds_t4}` · scenario_name `{_sc_t4}` · frame_index `{frame}` · "
                    f"server `{base_url_t4}`"
                )
                st.markdown(_show_err)
        elif _meta and _imgs:
            st.caption(
                f"**sample_token** `{_meta.get('sample_token', '')}` · "
                f"**timestamp_us** `{_meta.get('timestamp_us', '')}`"
            )
            _nc = min(3, max(1, len(_imgs)))
            for _row_start in range(0, len(_imgs), _nc):
                _cols_img = st.columns(_nc)
                for _j, _k in enumerate(range(_row_start, min(_row_start + _nc, len(_imgs)))):
                    _lbl, _png = _imgs[_k]
                    with _cols_img[_j]:
                        st.caption(_lbl)
                        st.image(_png, use_container_width=True)

# ----------------------------
# Quick view: switch between "All (comparison)" and single-run view
# ----------------------------
solo_run: str | None = None
if len(files_to_load) > 1:
    run_names = [f[1] for f in files_to_load]
    quick_options = ["All (comparison)"] + [f"Run {lbl} only" for lbl in run_names]
    quick_view = st.radio(
        "**Quick view** — switch which run(s) are shown:",
        quick_options,
        key="bbox_quick_view",
        horizontal=True,
        help="Choose one run to see it alone in a single BEV, or 'All' to see side-by-side or overlay comparison.",
    )
    if quick_view != "All (comparison)":
        solo_run = quick_view.replace("Run ", "").replace(" only", "").strip()
    if solo_run is not None:
        st.caption(f"Showing **Run {solo_run}** only. Select *All (comparison)* above to compare runs again.")

# ----------------------------
# Geometry (yaw補正: x前方, y左方 → +π/2)
# ----------------------------
def rotated_rect(
    x: float, y: float,
    length: float, width: float,
    yaw: float,
    step_depth_ratio: float = 0.25,
    step_width_ratio: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    前方左側に段差（凹み）を入れて向きを表す矩形Polylineを返す。
    - yaw: ラジアン
    - step_depth_ratio: 凹みの「奥行き」（length比）
    - step_width_ratio: 凹みの「横幅」（width比）
    """
    if length < width:
        # something is wrong, fix size
        length, width = max(length, width), min(length, width)

    dx, dy = length / 2.0, width / 2.0
    step_depth = length * step_depth_ratio
    step_width = width * step_width_ratio

    # 頂点順序（時計回り）
    # 後ろ左 → 前左(手前側) → 凹み奥 → 前中央左 → 前右 → 後右 → 後ろ左
    corners = np.array([
        [-dx, -dy],                      # 後ろ左
        [ dx, -dy],                      # 前左端
        [ dx, 0],         # 段差上部
        [ dx - step_depth, 0],  # 凹み奥左
        [dx, 0],
        [dx,  dy],                      # 前右端
        [-dx,  dy],                      # 後右
        [-dx, -dy]                       # 戻る
    ])

    # 回転 (+π/2 でBEV向き調整)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    rotated = corners @ rot.T

    xs, ys = rotated[:, 0] + x, rotated[:, 1] + y
    return xs, ys


def _build_one_bev_figure(
    df_fr: pd.DataFrame,
    plot_title: str,
    show_inv: bool,
    x_range: Tuple[float, float] | None = None,
    y_range: Tuple[float, float] | None = None,
    hover_extra_cols: Optional[List[str]] = None,
    show_velocity_arrows: bool = False,
) -> go.Figure:
    """Build one BEV figure from a single run's frame data. Optional x_range, y_range for consistent side-by-side scale."""
    if hover_extra_cols is None:
        hover_extra_cols = []
    extra_in_df = [c for c in hover_extra_cols if c in df_fr.columns]
    n_extra = len(extra_in_df)

    def _make_customdata(labels, lengths, widths, uuids, extras_df=None):
        if extras_df is None or n_extra == 0:
            return np.column_stack([labels, lengths, widths, uuids])
        extra_arrays = [extras_df[c].values for c in extra_in_df]
        return np.column_stack([labels, lengths, widths, uuids] + extra_arrays)

    def _hovertemplate():
        base = "X: %{x}<br>Y: %{y}<br>Label: %{customdata[0]}<br>size: %{customdata[1]:.2f} x %{customdata[2]:.2f}<br>UUID: %{customdata[3]}"
        for i, c in enumerate(extra_in_df):
            base += f"<br>{c}: %{{customdata[{4 + i}]}}"
        return base + "<extra></extra>"

    fig = go.Figure()
    shown = set()
    hovertemplate = _hovertemplate()
    mask_both_invalid = (df_fr["length"] <= 0) & (df_fr["width"] <= 0)
    mask_one_invalid = ((df_fr["length"] <= 0) | (df_fr["width"] <= 0)) & ~mask_both_invalid
    mask_valid = (df_fr["length"] > 0) & (df_fr["width"] > 0)

    if show_inv and not df_fr[mask_both_invalid].empty:
        d = df_fr[mask_both_invalid]
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["y"], mode="markers",
            marker=dict(symbol="x", size=8, color=d.apply(lambda row: get_color(row.source, row.status), axis=1)),
            opacity=0.9, showlegend=False, hovertemplate=hovertemplate,
            customdata=_make_customdata(d["label"].values, d["length"].values, d["width"].values, d["uuid"].values, d),
            name="invalid"
        ))
    if not df_fr[mask_one_invalid].empty:
        d = df_fr[mask_one_invalid].copy()
        d["name"] = d["source"] + "/" + d["status"]
        for name, group in d.groupby("name"):
            fig.add_trace(go.Scatter(
                x=group["x"], y=group["y"], mode="markers",
                marker=dict(symbol="circle", size=group[["length", "width"]].max(axis=1),
                           color=get_color(group.iloc[0].source, group.iloc[0].status)),
                opacity=0.6, name=name, legendgroup=name, showlegend=name not in shown,
                hovertemplate=hovertemplate,
                customdata=_make_customdata(group["label"].values, group["length"].values, group["width"].values, group["uuid"].values, group)
            ))
            shown.add(name)
    if not df_fr[mask_valid].empty:
        d = df_fr[mask_valid].copy()
        d["name"] = d["source"] + "/" + d["status"]
        for name, group in d.groupby("name"):
            show = name not in shown
            for _, row in group.iterrows():
                x_poly, y_poly = rotated_rect(row.x, row.y, row.length, row.width, row.yaw)
                row_custom = [row.label, row.length, row.width, row.uuid]
                if n_extra:
                    row_custom.extend([row[c] if c in row.index else "" for c in extra_in_df])
                fig.add_trace(go.Scatter(
                    x=x_poly, y=y_poly, mode="lines", fill="toself", opacity=0.6,
                    line=dict(color=get_color(row.source, row.status)),
                    name=name, legendgroup=name, showlegend=show, hovertemplate=hovertemplate,
                    customdata=[row_custom]
                ))
                show = False
            shown.add(name)
    # Velocity arrows (scale: 2 s)
    if show_velocity_arrows and "vx" in df_fr.columns and "vy" in df_fr.columns:
        v_scale = 2.0
        v_mask = df_fr["x"].notna() & df_fr["y"].notna() & df_fr["vx"].notna() & df_fr["vy"].notna()
        v_df = df_fr[v_mask]
        if not v_df.empty:
            xs, ys = [], []
            for _, r in v_df.iterrows():
                x0, y0 = float(r["x"]), float(r["y"])
                vx, vy = float(r["vx"]), float(r["vy"])
                xs.extend([x0, x0 + v_scale * vx, np.nan])
                ys.extend([y0, y0 + v_scale * vy, np.nan])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=pick(_LEGACY_VELOCITY_LINE, tokens()["muted"]), width=2, dash="dot"),
                name="Velocity (2 s)",
                showlegend=True,
            ))
    fig.add_trace(go.Scatter(
        x=[0, -1.5, -1.5, 0], y=[0, -1, 1, 0],
        mode="lines", fill="toself",
        line=dict(color=pick(_LEGACY_EGO_LINE, tokens()["text"]), width=2),
        fillcolor=pick(_LEGACY_EGO_FILL, tokens()["neutral"]), name="Ego Vehicle", showlegend=True
    ))
    layout_kw: dict = {
        "title": plot_title,
        "xaxis": dict(scaleanchor="y", scaleratio=1, title="X [m]"),
        "yaxis": dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
        "legend": dict(groupclick="togglegroup", title="Source / Status"),
        "height": 900,
    }
    if x_range is not None:
        layout_kw["xaxis"]["range"] = list(x_range)
    if y_range is not None:
        layout_kw["yaxis"]["range"] = list(y_range)
    fig.update_layout(**layout_kw)
    _theme_chart(fig)
    return fig


def _bev_axis_range_from_df(df_fr: pd.DataFrame, padding: float = 5.0) -> Tuple[Tuple[float, float], Tuple[float, float]] | None:
    """Compute (x_range, y_range) from frame data for consistent BEV scale. Returns None if no valid points."""
    if df_fr.empty:
        return None
    xs = df_fr["x"].dropna()
    ys = df_fr["y"].dropna()
    if xs.empty and ys.empty:
        return None
    x_min, x_max = float(xs.min()) - padding, float(xs.max()) + padding
    y_min, y_max = float(ys.min()) - padding, float(ys.max()) + padding
    x_min = min(x_min, -padding)
    x_max = max(x_max, padding)
    y_min = min(y_min, -padding)
    y_max = max(y_max, padding)
    return (x_min, x_max), (y_min, y_max)


def _build_overlay_bev_figure(
    df_frame: pd.DataFrame,
    run_order: List[str],
    plot_title: str,
    show_inv: bool,
    hover_extra_cols: Optional[List[str]] = None,
    show_velocity_arrows: bool = False,
) -> go.Figure:
    """Build one BEV with all runs overlaid. Legend = run only (toggle by run). Line style = run."""
    if hover_extra_cols is None:
        hover_extra_cols = []
    extra_in_df = [c for c in hover_extra_cols if c in df_frame.columns]
    n_extra = len(extra_in_df)

    def _overlay_hovertemplate():
        base = (
            "Run: %{customdata[0]}<br>X: %{x}<br>Y: %{y}<br>Label: %{customdata[1]}<br>"
            "Status: %{customdata[4]}<br>size: %{customdata[2]:.2f} x %{customdata[3]:.2f}<br>"
            "UUID: %{customdata[5]}"
        )
        for i, c in enumerate(extra_in_df):
            base += f"<br>{c}: %{{customdata[{6 + i}]}}"
        return base + "<extra></extra>"

    fig = go.Figure()
    dash_styles = ["solid", "dash", "dot", "dashdot"]
    hovertemplate = _overlay_hovertemplate()
    for run_idx, run_lbl in enumerate(run_order):
        dash = dash_styles[run_idx % len(dash_styles)]
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=pick(_LEGACY_RUN_PROXY_LINE, tokens()["muted"]), width=4, dash=dash),
            name=f"Run {run_lbl}",
            legendgroup=run_lbl,
            showlegend=True,
        ))
    for run_idx, run_lbl in enumerate(run_order):
        df_fr = df_frame[df_frame["run"] == run_lbl]
        if df_fr.empty:
            continue
        dash = dash_styles[run_idx % len(dash_styles)]
        mask_both_invalid = (df_fr["length"] <= 0) & (df_fr["width"] <= 0)
        mask_one_invalid = ((df_fr["length"] <= 0) | (df_fr["width"] <= 0)) & ~mask_both_invalid
        mask_valid = (df_fr["length"] > 0) & (df_fr["width"] > 0)

        if show_inv and not df_fr[mask_both_invalid].empty:
            d = df_fr[mask_both_invalid]
            base_cd = np.column_stack([
                np.full(len(d), run_lbl), d["label"].values, d["length"].values, d["width"].values,
                (d["source"] + "/" + d["status"]).values, d["uuid"].values,
            ])
            if n_extra:
                base_cd = np.column_stack([base_cd] + [d[c].values for c in extra_in_df])
            fig.add_trace(go.Scatter(
                x=d["x"], y=d["y"], mode="markers",
                marker=dict(
                    symbol="x", size=8,
                    color=d.apply(lambda row: get_color(row.source, row.status), axis=1),
                    line=dict(width=2, color=pick(_LEGACY_MARKER_OUTLINE_WHITE, tokens()["bg"])),
                ),
                opacity=0.9, legendgroup=run_lbl, showlegend=False,
                hovertemplate=hovertemplate,
                customdata=base_cd,
            ))
        if not df_fr[mask_one_invalid].empty:
            d = df_fr[mask_one_invalid].copy()
            d["status_str"] = d["source"] + "/" + d["status"]
            for _, group in d.groupby("status_str"):
                status_str = group["status_str"].iloc[0]
                base_cd = np.column_stack([
                    np.full(len(group), run_lbl), group["label"].values,
                    group["length"].values, group["width"].values,
                    np.full(len(group), status_str), group["uuid"].values,
                ])
                if n_extra:
                    base_cd = np.column_stack([base_cd] + [group[c].values for c in extra_in_df])
                fig.add_trace(go.Scatter(
                    x=group["x"], y=group["y"], mode="markers",
                    marker=dict(
                        symbol="circle", size=group[["length", "width"]].max(axis=1),
                        color=group.apply(lambda row: get_color(row.source, row.status), axis=1),
                        line=dict(width=2, color=pick(_LEGACY_MARKER_OUTLINE_WHITE, tokens()["bg"])),
                    ),
                    opacity=0.7, legendgroup=run_lbl, showlegend=False,
                    hovertemplate=hovertemplate,
                    customdata=base_cd,
                ))
        if not df_fr[mask_valid].empty:
            d = df_fr[mask_valid].copy()
            d["status_str"] = d["source"] + "/" + d["status"]
            for _, row in d.iterrows():
                x_poly, y_poly = rotated_rect(row.x, row.y, row.length, row.width, row.yaw)
                status_str = row["source"] + "/" + row["status"]
                row_custom = [run_lbl, row.label, row.length, row.width, status_str, row.uuid]
                if n_extra:
                    row_custom.extend([row[c] if c in row.index else "" for c in extra_in_df])
                fig.add_trace(go.Scatter(
                    x=x_poly, y=y_poly, mode="lines", fill="toself", opacity=0.5,
                    line=dict(color=get_color(row.source, row.status), width=2, dash=dash),
                    legendgroup=run_lbl, showlegend=False,
                    hovertemplate=hovertemplate,
                    customdata=[row_custom],
                ))
    if show_velocity_arrows and "vx" in df_frame.columns and "vy" in df_frame.columns:
        v_scale = 2.0
        v_mask = df_frame["x"].notna() & df_frame["y"].notna() & df_frame["vx"].notna() & df_frame["vy"].notna()
        v_df = df_frame[v_mask]
        if not v_df.empty:
            xs, ys = [], []
            for _, r in v_df.iterrows():
                x0, y0 = float(r["x"]), float(r["y"])
                vx, vy = float(r["vx"]), float(r["vy"])
                xs.extend([x0, x0 + v_scale * vx, np.nan])
                ys.extend([y0, y0 + v_scale * vy, np.nan])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=pick(_LEGACY_VELOCITY_LINE, tokens()["muted"]), width=2, dash="dot"),
                name="Velocity (2 s)",
                showlegend=True,
            ))
    fig.add_trace(go.Scatter(
        x=[0, -1.5, -1.5, 0], y=[0, -1, 1, 0],
        mode="lines", fill="toself",
        line=dict(color=pick(_LEGACY_EGO_LINE, tokens()["text"]), width=2),
        fillcolor=pick(_LEGACY_EGO_FILL, tokens()["neutral"]), name="Ego Vehicle", showlegend=True
    ))
    fig.update_layout(
        title=plot_title,
        xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
        yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
        legend=dict(groupclick="togglegroup", title="Run (click to show/hide)"),
        height=900,
    )
    _theme_chart(fig)
    return fig


# ----------------------------
# Status color legend (all BEV views)
# ----------------------------
st.markdown(bev_status_legend_markup(), unsafe_allow_html=True)

# ----------------------------
# Plot (single or side-by-side for multiple runs)
# ----------------------------
if solo_run is not None:
    # Quick view: single run only (full-width BEV)
    df_solo = df_frame[df_frame["run"] == solo_run]
    total_n = len(df_solo)
    valid_n = int(((df_solo["length"] > 0) & (df_solo["width"] > 0)).sum()) if not df_solo.empty else 0
    title = f"Run {solo_run} only — {selected_scenario or 'Scene'}<br>Frame {frame} | Total {total_n:,}, Valid {valid_n:,}"
    st.plotly_chart(
        _build_one_bev_figure(df_solo, title, show_invalid, hover_extra_cols=hover_extra_cols, show_velocity_arrows=show_velocity_arrows),
        width='stretch',
    )
elif len(files_to_load) > 1 and compare_view_mode == "overlay":
    run_lbls = [f[1] for f in files_to_load]
    line_names = ["——— solid", "- - - dashed", "· · · dot", "-·-· dashdot"]
    line_parts = [f"<strong>{run_lbls[i]}</strong> {line_names[i]}" for i in range(min(4, len(run_lbls)))]
    line_hint = " &nbsp;|&nbsp; ".join(line_parts)
    st.markdown(bev_overlay_line_and_status_legend_markup(line_hint), unsafe_allow_html=True)
    title = f"Overlay: {selected_scenario or 'Scene'} — Frame {frame}"
    st.plotly_chart(
        _build_overlay_bev_figure(df_frame, [f[1] for f in files_to_load], title, show_invalid, hover_extra_cols=hover_extra_cols, show_velocity_arrows=show_velocity_arrows),
        width='stretch',
    )
elif len(files_to_load) > 1:
    shared_range = _bev_axis_range_from_df(df_frame)
    x_range, y_range = (shared_range[0], shared_range[1]) if shared_range else (None, None)
    cols_bev = st.columns(len(files_to_load))
    for col, (_, run_lbl) in zip(cols_bev, files_to_load):
        df_fr = df_frame[df_frame["run"] == run_lbl]
        total_n = len(df_fr)
        valid_n = int(((df_fr["length"] > 0) & (df_fr["width"] > 0)).sum()) if not df_fr.empty else 0
        title = f"Run {run_lbl} — {selected_scenario or 'Scene'}<br>Frame {frame} | Total {total_n:,}, Valid {valid_n:,}"
        with col:
            st.plotly_chart(
                _build_one_bev_figure(df_fr, title, show_invalid, x_range=x_range, y_range=y_range, hover_extra_cols=hover_extra_cols, show_velocity_arrows=show_velocity_arrows),
                width='stretch',
            )
else:
    fig = _build_one_bev_figure(
        df_frame,
        f"{selected_scenario or 'Scene'} <br>Frame {frame} | Total {total_records:,}, Valid {valid_records:,}",
        show_invalid,
        hover_extra_cols=hover_extra_cols,
        show_velocity_arrows=show_velocity_arrows,
    )
    st.plotly_chart(fig, width="stretch")

# === Frame別 TP/FN カウントと比率 ===
st.markdown("## 📈 Detection Stability over Frames")

# TP/FN per frame (per run when both A and B are loaded)
groupby_cols = ["frame_index", "run", "status"] if len(files_to_load) > 1 else ["frame_index", "status"]
frame_stats = (
    df_stats.query("source == 'GT' and status in ['TP', 'FN']")
      .groupby(groupby_cols)
      .size()
      .unstack(fill_value=0)
      .reset_index()
)



# 比率 (TP率 = TP / (TP+FN))
# Ensure TP or FN column exists, else fill with 0
for col in ["TP", "FN"]:
    if col not in frame_stats.columns:
        frame_stats[col] = 0
frame_stats["TPR"] = np.where(
    (frame_stats["TP"] + frame_stats["FN"]) > 0,
    frame_stats["TP"] / (frame_stats["TP"] + frame_stats["FN"]),
    np.nan
)

# --- 時系列グラフ (melt so we can color by run when both A and B) ---
id_vars = ["frame_index", "run"] if "run" in frame_stats.columns else ["frame_index"]
value_vars = [c for c in ["TP", "FN"] if c in frame_stats.columns]
frame_stats_melt = frame_stats.melt(id_vars=id_vars, value_vars=value_vars, var_name="Status", value_name="Count")
if "run" in frame_stats_melt.columns:
    fig_tpr = px.line(
        frame_stats_melt,
        x="frame_index",
        y="Count",
        color="run",
        line_dash="Status",
        title="TP / FN Counts per Frame (by run)",
        labels={"Count": "Count", "frame_index": "Frame Index"},
    )
else:
    fig_tpr = px.line(
        frame_stats_melt,
        x="frame_index",
        y="Count",
        color="Status",
        title="TP / FN Counts per Frame",
        labels={"Count": "Count", "frame_index": "Frame Index", "variable": "Status"},
    )
fig_tpr.update_layout(height=400, legend_title="Run / Status" if "run" in frame_stats_melt.columns else "Status")

# --- 現在Frameに縦破線を追加 ---
fig_tpr.add_vline(
    x=frame,
    line=dict(color=pick(_LEGACY_VLINE, tokens()["text"]), dash="dash", width=2),
    annotation_text=f"Frame {frame}",
    annotation_position="top left"
)

_theme_chart(fig_tpr)
st.plotly_chart(fig_tpr, width="stretch")

# TPR比率の推移を別グラフで (side by side when both runs)
if "run" in frame_stats.columns:
    fig_ratio = px.line(
        frame_stats,
        x="frame_index",
        y="TPR",
        color="run",
        title="True Positive Rate (TPR) per Frame (by run)",
        labels={"TPR": "True Positive Rate", "frame_index": "Frame Index"},
    )
else:
    fig_ratio = px.line(
        frame_stats,
        x="frame_index",
        y="TPR",
        title="True Positive Rate (TPR) per Frame",
        labels={"TPR": "True Positive Rate", "frame_index": "Frame Index"},
    )
fig_ratio.update_yaxes(range=[0, 1])
fig_ratio.add_vline(
    x=frame,
    line=dict(color=pick(_LEGACY_VLINE, tokens()["text"]), dash="dash", width=2),
    annotation_text=f"Frame {frame}",
    annotation_position="top left"
)
_theme_chart(fig_ratio)
st.plotly_chart(fig_ratio, width="stretch")

# === Worst-performing objects by FN rate ===
st.markdown("## 🚨 Objects with High FN Rate (GT-based)")

# uuid + label (and run when both) ごとにTP/FNをカウント
groupby_uuid = ["uuid", "label", "run"] if len(files_to_load) > 1 else ["uuid", "label"]
uuid_perf = (
    df_stats.query("source == 'GT' and status in ['TP','FN']")
      .groupby(groupby_uuid)["status"]
      .value_counts()
      .unstack(fill_value=0)
      .reset_index()
)

# Make sure 'TP' and 'FN' columns are present, else fill with 0
for col in ['TP', 'FN']:
    if col not in uuid_perf.columns:
        uuid_perf[col] = 0

uuid_perf["total"] = uuid_perf["TP"] + uuid_perf["FN"]
uuid_perf["FN_rate"] = uuid_perf["FN"] / uuid_perf["total"].replace(0, np.nan)

# total > 0 だけ残す
uuid_perf = uuid_perf[uuid_perf["total"] > 0]

# FN率でソート
uuid_perf_sorted = uuid_perf.sort_values("FN_rate", ascending=False)

# All GT UUIDs in current data (for "inspect any object" use case)
all_gt_uuids = df_stats[df_stats["source"] == "GT"]["uuid"].dropna().unique().tolist()

# 表示 (per run when multiple runs)
if not uuid_perf_sorted.empty:
    total_objects = len(uuid_perf_sorted)
    st.caption(f"Showing top 30 by FN rate ({total_objects} GT objects with TP/FN in this scene).")
    display_cols = ["uuid", "label", "TP", "FN", "total", "FN_rate"]
    if "run" not in uuid_perf_sorted.columns:
        display_cols = [c for c in display_cols if c != "run"]
    if len(files_to_load) > 1 and "run" in uuid_perf_sorted.columns:
        n_cols = min(len(files_to_load), 4)
        cols_disp = st.columns(n_cols)
        for idx, (col, (_, run_lbl)) in enumerate(zip(cols_disp, files_to_load)):
            if idx >= n_cols:
                break
            with col:
                st.markdown(f"**Run {run_lbl}**")
                df_r = uuid_perf_sorted[uuid_perf_sorted["run"] == run_lbl].head(30)
                if not df_r.empty:
                    st.dataframe(df_r[display_cols].style.format({"FN_rate": "{:.2%}"}))
                else:
                    st.info(f"No data for run {run_lbl}.")
        if len(files_to_load) > n_cols:
            for run_lbl in [f[1] for f in files_to_load[n_cols:]]:
                with st.expander(f"Run {run_lbl}"):
                    df_r = uuid_perf_sorted[uuid_perf_sorted["run"] == run_lbl].head(30)
                    if not df_r.empty:
                        st.dataframe(df_r[display_cols].style.format({"FN_rate": "{:.2%}"}))
                    else:
                        st.info(f"No data for run {run_lbl}.")
    else:
        st.dataframe(
            uuid_perf_sorted[display_cols]
                .head(30)
                .style.format({"FN_rate": "{:.2%}"})
        )
else:
    st.info("No GT objects with TP or FN were found.")

st.markdown("### 🔍 Inspect a Specific GT Object")

st.caption("Pick an object from the high-FN list below to view its trajectory. To inspect a different object, open **Inspect another object**.")

# Default path: dropdown from high-FN list (only when no custom UUID is used)
bad_uuid: str | None = None
custom_uuid_input = ""

with st.expander("Inspect another object (browse all or enter UUID)"):
    st.caption("Use this when the object you want is not in the high-FN table. Hover on the BEV chart to see a UUID, or browse the table and copy one.")
    custom_uuid_input = st.text_input(
        "Enter UUID",
        value="",
        key="bbox_inspect_custom_uuid",
        placeholder="Paste or type a UUID from this scene",
        help="Must exist in the current scene/topic/labels.",
    )
    gt_for_browse = df_stats[df_stats["source"] == "GT"][["uuid", "label"] + (["run"] if "run" in df_stats.columns else [])].drop_duplicates()
    gt_for_browse = gt_for_browse.sort_values(["label", "uuid"]).reset_index(drop=True)
    st.caption("All GT UUIDs in this scene (select a cell to copy):")
    st.dataframe(gt_for_browse, width='stretch', hide_index=True)

if custom_uuid_input and str(custom_uuid_input).strip():
    candidate = str(custom_uuid_input).strip()
    if candidate in all_gt_uuids:
        bad_uuid = candidate
    else:
        st.warning(
            f"UUID `{candidate}` was not found in the current data. "
            "Check that it belongs to the selected scene, topic, and labels."
        )
if bad_uuid is None and not uuid_perf_sorted.empty:
    uuid_options = uuid_perf_sorted["uuid"].drop_duplicates().head(100).tolist()
    bad_uuid = st.selectbox(
        "Select UUID to visualize",
        options=uuid_options,
        key="bbox_inspect_uuid_select",
        help="Top 100 by FN rate. To inspect another object, open the expander above.",
    )

if bad_uuid is not None:
    uuid_traj = (
        df_stats[(df_stats["uuid"] == bad_uuid) & (df_stats["source"] == "GT")]
        .sort_values("frame_index")
    )
else:
    uuid_traj = pd.DataFrame()

def _draw_trajectory_figure(traj: pd.DataFrame, title: str) -> go.Figure:
    """Build trajectory figure for one run's data."""
    fig_traj = go.Figure()
    symbol_map = {"TP": "circle", "FN": "x", "FP": "triangle-up"}
    traj = traj.copy()
    traj["marker_symbol"] = traj["status"].map(symbol_map).fillna("circle")
    fig_traj.add_trace(go.Scatter(
        x=traj["x"], y=traj["y"],
        mode="lines",
        line=dict(color=pick(_LEGACY_TRAJ_LINE, tokens()["muted"]), width=1),
        name=f"Trajectory ({traj['label'].iloc[0]})"
    ))
    for status, group in traj.groupby("status"):
        fig_traj.add_trace(go.Scatter(
            x=group["x"], y=group["y"],
            mode="markers",
            marker=dict(
                symbol=symbol_map.get(status, "circle"),
                size=[10 if f == frame else 6 for f in group["frame_index"]],
                color=[
                    pick(_LEGACY_TRAJ_CURRENT, tokens()["bad"])
                    if f == frame
                    else (pick(_LEGACY_TRAJ_FN, tokens()["warn"]) if status == "FN" else pick(_LEGACY_TRAJ_TP, tokens()["ok"]))
                    for f in group["frame_index"]
                ],
                line=dict(width=1, color=pick(_LEGACY_TRAJ_POINT_OUTLINE, tokens()["bg"]))
            ),
            name=f"{status} points"
        ))
    fig_traj.update_layout(
        title=title,
        xaxis=dict(title="X [m]", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y [m]", scaleanchor="x", scaleratio=1),
        height=600,
        legend=dict(title="Status")
    )
    _theme_chart(fig_traj)
    return fig_traj

if not uuid_traj.empty:
    show_multi_runs = len(files_to_load) > 1 and "run" in uuid_traj.columns
    if show_multi_runs:
        label_str = uuid_traj["label"].iloc[0]
        n_traj_cols = min(len(files_to_load), 4)
        traj_cols = st.columns(n_traj_cols)
        for idx, (col, (_, run_lbl)) in enumerate(zip(traj_cols, files_to_load)):
            if idx >= n_traj_cols:
                break
            with col:
                traj_r = uuid_traj[uuid_traj["run"] == run_lbl]
                if not traj_r.empty:
                    st.plotly_chart(_draw_trajectory_figure(traj_r, f"Run {run_lbl}: UUID {bad_uuid} ({label_str})"), width='stretch')
                else:
                    st.info(f"No trajectory for this UUID in run {run_lbl}.")
        for run_lbl in [f[1] for f in files_to_load[n_traj_cols:]]:
            traj_r = uuid_traj[uuid_traj["run"] == run_lbl]
            if not traj_r.empty:
                with st.expander(f"Run {run_lbl}: UUID {bad_uuid}"):
                    st.plotly_chart(_draw_trajectory_figure(traj_r, f"Run {run_lbl}: UUID {bad_uuid} ({label_str})"), width='stretch')
            else:
                with st.expander(f"Run {run_lbl}: UUID {bad_uuid}"):
                    st.info(f"No trajectory for this UUID in run {run_lbl}.")
    else:
        st.plotly_chart(_draw_trajectory_figure(uuid_traj, f"Trajectory of UUID {bad_uuid} ({uuid_traj['label'].iloc[0]})"), width="stretch")
else:
    if bad_uuid is None:
        st.info("Select a UUID from the list above to view its trajectory, or open **Inspect another object** to browse all or enter a UUID.")
    else:
        st.info("No GT trajectory data for the selected UUID.")
