"""T4 dataset Three.js viewer: GT / prediction / matched 3D boxes via postMessage to `/viewer/three`."""

import duckdb
import requests
import streamlit as st
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Any, List

from lib.path_utils import path_display
from lib.overview_url_hydrate import try_hydrate_session_from_overview_query_params
from lib.page_chrome import inject_app_page_styles, render_loaded_data_section, render_page_hero
from lib.t4_dataset_embed import t4_share_query_params
from lib.t4_three_layers import (
    EXTERNAL_BBOX_ALIGNMENT_VERSION,
    build_three_layer_payload_all_frames,
    infer_external_bbox_alignment_query_params,
    render_t4_three_js_embed,
    resolve_t4_dataset_id,
    resolve_t4_scenario,
)
from lib.t4_visualizer_client import (
    DEFAULT_BASE_URL,
    ENV_BASE_URL,
    T4VisualizerClient,
    T4VisualizerError,
    browser_base_url,
)

st.set_page_config(
    layout="wide",
    page_title="T4 3D Viewer",
    page_icon="🧊",
    initial_sidebar_state="expanded",
)
inject_app_page_styles()


def _query_param_text(*names: str) -> str:
    for name in names:
        value = st.query_params.get(name)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _prime_viewer_state_from_query_params() -> None:
    """Map share-link query params onto existing sidebar deep-link session keys."""
    mapping = {
        "bbox_viewer_link_suite": ("viewer_suite", "suite_name"),
        "bbox_viewer_link_scenario": ("viewer_scenario", "scenario_name"),
        "bbox_viewer_link_t4dataset": ("viewer_t4dataset", "t4dataset_name", "t4dataset_id"),
        "bbox_viewer_link_topic": ("viewer_topic", "topic_name"),
        "bbox_viewer_link_frame": ("viewer_frame", "frame_index"),
    }
    for state_key, param_names in mapping.items():
        value = _query_param_text(*param_names)
        if value:
            st.session_state[state_key] = value


try_hydrate_session_from_overview_query_params()
_prime_viewer_state_from_query_params()

# =============================
# Session state from Overview (run path)
# =============================
if "runA" not in st.session_state:
    st.warning("Please load data from the **Overview** page first (select mode and run(s)).")
    st.stop()

runA = st.session_state["runA"]
mode = st.session_state.get("mode", "Single Mode")
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
    p = Path(run_path)
    if not p.is_dir():
        return []
    return sorted([str(f.resolve()) for f in p.glob("*.parquet")])


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
    kicker="T4 visualizer",
    title="T4 3D bounding box viewer",
    description=(
        "Embedded **Three.js** view with GT, prediction (EST), and UUID-matched pairs from parquet (**postMessage**). "
        "Scrub **time inside the viewer** (bottom slider); eval boxes follow that frame. Same filters as the BEV page."
    ),
    mode=mode,
)

# ----------------------------
# Sidebar (Filters) — shared keys with Bounding Box Viewer
# ----------------------------
with st.sidebar:
    st.markdown("##### Filters")
    st.caption("Same scene / topic / labels as the BEV viewer. Frame / playback: use the **3D viewer** controls.")

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

    first_shown = runs_to_show[0] if runs_to_show else run_labels_list[0]
    filter_file = selected_files.get(first_shown) or parquet_lists[run_labels_list.index(first_shown)][0]

con = duckdb.connect()

cols = con.execute("DESCRIBE SELECT * FROM parquet_scan(?)", [filter_file]).df()["column_name"].tolist()
has_visibility = "visibility" in cols
has_suite_name = "suite_name" in cols
has_scenario_name = "scenario_name" in cols
has_t4dataset_name = "t4dataset_name" in cols
hover_extra_cols = [c for c in ["z", "height", "vx", "vy", "confidence", "pointcloud_num"] if c in cols]

scene_where = "1=1"
scene_params: List[str] = [filter_file]

if has_suite_name:
    suite_list = con.execute(
        "SELECT DISTINCT suite_name AS v FROM parquet_scan(?) WHERE suite_name IS NOT NULL ORDER BY v",
        [filter_file],
    ).df()["v"].dropna().astype(str).tolist()
else:
    suite_list = []

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
                [filter_file, selected_suite],
            ).df()["v"].dropna().astype(str).tolist()
        else:
            scenario_list = con.execute(
                "SELECT DISTINCT scenario_name AS v FROM parquet_scan(?) WHERE scenario_name IS NOT NULL ORDER BY v",
                [filter_file],
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

topic_names = con.execute(
    f"SELECT DISTINCT topic_name AS v FROM parquet_scan(?) WHERE {scene_where} ORDER BY v",
    scene_params,
).df()["v"].dropna().tolist()
if not topic_names:
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

if "bbox_viewer_link_topic" in st.session_state:
    _ltopic = st.session_state.pop("bbox_viewer_link_topic", None)
    if _ltopic is not None and str(_ltopic) in [str(t) for t in topic_names]:
        st.session_state["bbox_viewer_topic"] = str(_ltopic)

with st.sidebar:
    selected_topic = st.selectbox("topic_name (single)", topic_names, key="bbox_viewer_topic")

labels = con.execute(
    f"SELECT DISTINCT label AS v FROM parquet_scan(?) WHERE {scene_where} AND topic_name=? ORDER BY v",
    scene_params + [selected_topic],
).df()["v"].dropna().tolist()
if not labels:
    st.warning("No label for selected topic.")
    st.stop()

with st.sidebar:
    selected_labels = st.multiselect("label(s)", labels, default=labels)

selected_visibility = None
if has_visibility:
    vis_list = con.execute(
        f"SELECT DISTINCT COALESCE(visibility,'UNKNOWN') AS v FROM parquet_scan(?) WHERE {scene_where} AND topic_name=? ORDER BY v",
        scene_params + [selected_topic],
    ).df()["v"].tolist()
    with st.sidebar:
        if vis_list:
            selected_visibility = st.multiselect("visibility", vis_list, default=vis_list)
        else:
            st.info("No visibility values found — skipping.")
else:
    with st.sidebar:
        st.info("No 'visibility' column found — skipping visibility filter.")

if not selected_labels:
    st.warning("No label selected.")
    st.stop()

with st.sidebar:
    st.markdown("##### T4 server")
    st.caption("**GET /datasets/{id}/availability** must succeed before the iframe loads.")
    if "bbox_t4_base_url" not in st.session_state:
        st.session_state["bbox_t4_base_url"] = (
            (os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        )
    st.text_input(
        "T4 server base URL",
        key="bbox_t4_base_url",
        help=f"Server-side API URL. Default from env `{ENV_BASE_URL}`. Browser iframes use `T4_VISUALIZER_BROWSER_BASE_URL` when set.",
    )

# ----------------------------
# Load data (same SQL as Bounding Box Viewer)
# ----------------------------
where = [scene_where, "topic_name = ?"]
params = scene_params + [selected_topic]
where.append(f"label IN ({','.join(['?']*len(selected_labels))})")
params.extend(selected_labels)

if has_visibility and selected_visibility:
    where.append(f"COALESCE(visibility,'UNKNOWN') IN ({','.join(['?']*len(selected_visibility))})")
    params.extend(selected_visibility)

_renderer_optional_cols = [
    "unix_time",
    "frame_id",
    "z",
    "height",
    "shape_type",
    "vx",
    "vy",
    "confidence",
    "pointcloud_num",
    "visibility",
    "x_error",
    "y_error",
    "z_error",
    "yaw_error",
    "vx_error",
    "vy_error",
    "speed_error",
    "center_distance",
    "plane_distance",
    "pair_dt_sec",
    "pair_uuid",
    "dx_min",
    "dy_min",
    "t4dataset_id",
    "suite_name",
    "t4dataset_name",
    "scenario_name",
]
_select_cols = [
    "frame_index",
    "x",
    "y",
    "length",
    "width",
    "yaw",
    "label",
    "topic_name",
    "source",
    "status",
    "uuid",
]
_select_cols.extend(c for c in _renderer_optional_cols if c in cols and c not in _select_cols)
sql = f"""
SELECT {", ".join(_select_cols)}
FROM parquet_scan(?)
WHERE {" AND ".join(where)}
ORDER BY frame_index
"""

files_to_load: List[tuple] = [(selected_files[lbl], lbl) for lbl in runs_to_show if lbl in selected_files]
base_params = scene_params[1:] + [selected_topic] + list(selected_labels)
if has_visibility and selected_visibility:
    base_params = base_params + list(selected_visibility)

dfs = []
for file_path, run_label in files_to_load:
    qparams = [file_path] + base_params
    df_part = con.execute(sql, qparams).df()
    if not df_part.empty:
        df_part = df_part.copy()
        df_part["run"] = run_label
        dfs.append(df_part)

if not dfs:
    st.warning("No data matches the selected filters.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)
if len(files_to_load) == 1:
    df["run"] = df["run"].iloc[0]

if "frame_index" in df.columns and not np.issubdtype(df["frame_index"].dtype, np.integer):
    df["frame_index"] = (
        pd.to_numeric(df["frame_index"], errors="coerce").fillna(0).astype(int)
    )

if len(files_to_load) == 1:
    st.info(f"**Currently showing:** Run {files_to_load[0][1]} only")
else:
    run_names = [f[1] for f in files_to_load]
    st.info(f"**Currently showing:** Runs {', '.join(run_names)} — 3D layers include boxes from all selected runs.")

f_min, f_max = int(df.frame_index.min()), int(df.frame_index.max())

_iframe_entry_frame = f_min
if "bbox_viewer_link_frame" in st.session_state:
    _link_frame_raw = st.session_state.pop("bbox_viewer_link_frame", None)
    try:
        _link_frame = int(float(str(_link_frame_raw)))
    except (TypeError, ValueError):
        _link_frame = None
    if _link_frame is not None:
        frame_values = sorted({int(v) for v in df["frame_index"].dropna().tolist()})
        if _link_frame in frame_values:
            _iframe_entry_frame = _link_frame
        elif frame_values:
            _iframe_entry_frame = min(frame_values, key=lambda v: abs(v - _link_frame))

# One reference slice for resolving t4dataset_id / scenario_name (same as iframe entry frame).
_ref_frame = _iframe_entry_frame
df_frame = df[df.frame_index == _ref_frame]
if df_frame.empty and not df.empty:
    df_frame = df.iloc[:1].copy()

# ----------------------------
# T4 Three.js embed
# ----------------------------
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
    st.warning(
        "Cannot resolve a T4 dataset id for this frame. Needs parquet **t4dataset_id** or **t4dataset_name**, "
        f"or **t4dataset_name** in the sidebar when multiple datasets exist. Set **T4 server base URL** or `{ENV_BASE_URL}`."
    )
else:
    _t4_avail_cache_key = f"{base_url_t4.rstrip('/')}|{_ds_t4}"
    _cached_av = st.session_state.get("bbox_t4_availability")
    _need_avail_fetch = _cached_av is None or _cached_av.get("cache_key") != _t4_avail_cache_key
    if _need_avail_fetch:
        try:
            with st.spinner("Checking T4 dataset on the server…"):
                _av_client = T4VisualizerClient(base_url=base_url_t4, timeout=2.0)
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
                "error": f"T4 server error ({ex.status_code}): {ex}",
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
        st.error("Could not verify the dataset on the T4 visualizer server.")
        with st.expander("Details", expanded=False):
            st.markdown(_av.get("error") or "Unknown error.")
    elif not _av.get("available"):
        st.warning("This dataset is not available on the visualizer server host.")
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
        # Fixed entry frame so Streamlit slider does not reload the iframe; eval layers use bbox_layers_by_frame.
        _q_three = t4_share_query_params(_ds_t4, _sc_t4, _iframe_entry_frame)
        _q_three = f"{_q_three}&{infer_external_bbox_alignment_query_params(df)}"
        _viewer_three_url = f"{browser_url_t4.rstrip('/')}/viewer/three?{_q_three}"
        _layer_payload = build_three_layer_payload_all_frames(df)

        _viewer_three_h = 1400
        render_t4_three_js_embed(_viewer_three_url, _layer_payload, height=_viewer_three_h)
        with st.expander("T4 bbox alignment debug", expanded=False):
            first_frame_key = next(iter(sorted((_layer_payload.get("frames") or {}).keys(), key=lambda v: int(v))), "")
            first_frame_payload = (_layer_payload.get("frames") or {}).get(first_frame_key, {})
            first_gt = (first_frame_payload.get("gt") or [{}])[0]
            st.code(_viewer_three_url, language="text")
            st.json(
                {
                    "alignment_version": EXTERNAL_BBOX_ALIGNMENT_VERSION,
                    "frame": first_frame_key,
                    "first_gt": {
                        key: first_gt.get(key)
                        for key in ("uuid", "label", "status", "length", "width", "height", "yaw", "force_wireframe")
                    },
                    "has_corners": "corners" in first_gt,
                }
            )

st.page_link("pages/4_Bounding_Box_Viewer.py", label="Back to Bounding Box & BEV viewer", icon="🖼️")
