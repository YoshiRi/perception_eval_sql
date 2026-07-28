"""T4 dataset Three.js viewer: GT / prediction / matched 3D boxes via postMessage to `/viewer/three`."""

import duckdb
import requests
import streamlit as st
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from typing import Any, List

DEFAULT_OBJECTS_TOPIC = "perception.object_recognition.objects"
DEFAULT_TRACKING_OBJECTS_TOPIC = "perception.object_recognition.tracking.objects"
VIEWER_DEEP_LINK_KEYS = (
    "mode",
    "run_a",
    "run_b",
    "run_c",
    "run_d",
    "run_e",
    "viewer_suite",
    "viewer_scenario",
    "viewer_t4dataset",
    "viewer_topic",
    "viewer_frame",
    "viewer_compare",
)
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

from lib.auth import (
    _read_streamlit_headers,
    get_access_context,
)
from lib.path_utils import path_display
from lib.overview_url_hydrate import try_hydrate_session_from_overview_query_params
from lib.page_chrome import inject_app_page_styles, render_loaded_data_section, render_page_hero
from lib.t4_dataset_embed import t4_share_query_params
from lib.t4_three_layers import (
    EXTERNAL_BBOX_ALIGNMENT_VERSION,
    build_three_layer_payload_all_frames,
    infer_external_bbox_alignment_query_params,
    infer_legacy_width_length_swapped,
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
    format_t4_visualizer_error,
)

st.set_page_config(
    layout="wide",
    page_title="T4 3D Viewer",
    page_icon="🧊",
    initial_sidebar_state="expanded",
)
inject_app_page_styles()


with st.expander("🔎 Access / request header debug", expanded=False):
    _hdrs = _read_streamlit_headers()
    if not _hdrs:
        st.info(
            "No request headers available (`st.context.headers` returned empty). "
            "This can happen in some run contexts; try a hard refresh."
        )
    else:
        _access = get_access_context(_hdrs)
        if _access["is_cloudflare"]:
            _who = _access.get("user_email") or "(no email header)"
            st.success(
                f"Accessed via **Cloudflare** as **{_who}** — Host: `{_access['host']}`, "
                f"Cf-Ray: `{_access['cf_ray']}`"
            )
        else:
            st.warning(
                f"Accessed **directly / non-Cloudflare** — Host: `{_access['host']}` "
                "(no Cf-* headers present; identity cannot be trusted)."
            )
        st.markdown("**Access context**")
        st.json(_access)
        st.markdown("**All request headers**")
        st.json(_hdrs)


def _query_param_text(*names: str) -> str:
    for name in names:
        value = st.query_params.get(name)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "<na>"} else text


def _candidate_annotation_dataset_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("T4DATASET_ROOT", "T4_DATASET_ROOT", "T4_VISUALIZER_DATA_DIR", "T4_VISUALIZER_DATA_ROOT"):
        text = _clean_text(os.environ.get(env_name))
        if text:
            roots.append(Path(text))
    roots.extend(
        [
            Path.home() / ".webauto/data/data/annotation_dataset",
            Path("/home/leigu/.webauto/data/data/annotation_dataset"),
            Path("/mnt/qnapdata/internal/t4datasets"),
            Path("/home/leigu/evaluator_result_parser/t4datasets"),
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _resolve_annotation_dataset_id_from_name(t4dataset_name: str) -> str:
    """Resolve a DB/bag-style dataset name to the local annotation dataset UUID when cached."""
    target = _clean_text(t4dataset_name)
    if not target:
        return ""
    if UUID_RE.match(target):
        return target
    for root in _candidate_annotation_dataset_roots():
        if not root.exists() or not root.is_dir():
            continue
        try:
            dataset_dirs = sorted(p for p in root.iterdir() if p.is_dir() and UUID_RE.match(p.name))
        except OSError:
            continue
        for dataset_dir in dataset_dirs:
            try:
                version_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
            except OSError:
                continue
            for version_dir in version_dirs:
                input_bag = version_dir / "input_bag"
                if not input_bag.is_dir():
                    continue
                try:
                    for bag_path in input_bag.iterdir():
                        if bag_path.is_file() and target in bag_path.name:
                            return dataset_dir.name
                except OSError:
                    continue
    return ""


def _legacy_prefixed_match(options: list[str], base_name: str, suffix_prefix: str = "") -> str:
    base = _clean_text(base_name)
    if not base:
        return ""
    if base in options:
        return base
    if suffix_prefix:
        wanted = f"{base}_{suffix_prefix}"
        if wanted in options:
            return wanted
    matches = [opt for opt in options if opt.startswith(f"{base}_")]
    return matches[0] if len(matches) == 1 else ""


def _prime_viewer_state_from_query_params() -> None:
    """Map share-link query params onto existing sidebar deep-link session keys."""
    sig = _viewer_deep_link_signature()
    if sig is None or st.session_state.get("_t4_viewer_deep_link_primed_sig") == sig:
        return
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
    st.session_state["_t4_viewer_deep_link_primed_sig"] = sig


def _viewer_deep_link_signature() -> tuple[str, ...] | None:
    values = tuple(_query_param_text(k) for k in VIEWER_DEEP_LINK_KEYS)
    return values if any(values) else None


def _viewer_deep_link_signature_from_values(values: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(values.get(k, "") or "").strip() for k in VIEWER_DEEP_LINK_KEYS)


def _sync_viewer_query_params(updates: dict[str, Any]) -> None:
    """Keep the browser URL aligned with the current viewer scene selection."""
    clean_updates = {
        key: str(value).strip()
        for key, value in updates.items()
        if value is not None and str(value).strip()
    }
    if not clean_updates:
        return
    current = {key: _query_param_text(key) for key in VIEWER_DEEP_LINK_KEYS}
    if all(current.get(key, "") == value for key, value in clean_updates.items()):
        return
    merged = {**current, **clean_updates}
    sig = _viewer_deep_link_signature_from_values(merged)
    st.session_state["_t4_viewer_deep_link_sig"] = sig
    st.session_state["_t4_viewer_deep_link_primed_sig"] = sig
    st.query_params.update(clean_updates)


def _reset_viewer_widget_state_for_new_deep_link() -> None:
    sig = _viewer_deep_link_signature()
    if sig is None or st.session_state.get("_t4_viewer_deep_link_sig") == sig:
        return
    for key in list(st.session_state.keys()):
        if str(key).startswith("bbox_viewer_"):
            st.session_state.pop(key, None)
    st.session_state["_t4_viewer_deep_link_sig"] = sig


try_hydrate_session_from_overview_query_params()
_reset_viewer_widget_state_for_new_deep_link()
_prime_viewer_state_from_query_params()

_viewer_compare_mode = _query_param_text("viewer_compare", "compare_view", "compare_mode")
if _viewer_compare_mode not in {"side_by_side", "side-by-side", "sidebyside", "curtain", "overlay"}:
    _viewer_compare_mode = ""
_viewer_link_suite = _query_param_text("viewer_suite", "suite_name")
_viewer_link_scenario = _query_param_text("viewer_scenario", "scenario_name")
_viewer_link_t4dataset = _query_param_text("viewer_t4dataset", "t4dataset_name", "t4dataset_id")
_viewer_link_dataset_id = _resolve_annotation_dataset_id_from_name(_viewer_link_t4dataset)
_viewer_link_dataset_prefix = _viewer_link_dataset_id.split("-", 1)[0] if _viewer_link_dataset_id else ""

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
if not _viewer_compare_mode:
    _viewer_compare_mode = "side_by_side" if multi_run else "overlay"

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
            preferred_file_index = next(
                (
                    idx
                    for idx, path in enumerate(pl)
                    if Path(path).stem == DEFAULT_OBJECTS_TOPIC
                    or os.path.basename(path).startswith(f"{DEFAULT_OBJECTS_TOPIC}.")
                ),
                0,
            )
            selected_files[lbl] = st.selectbox(
                f"File (Run {lbl})",
                pl,
                index=preferred_file_index,
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
    _suite_hit = _legacy_prefixed_match(suite_list, str(_lsu), "") if suite_list and _lsu is not None else ""
    if _suite_hit:
        st.session_state["bbox_viewer_suite"] = _suite_hit

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
                _scenario_hit = (
                    _legacy_prefixed_match(scenario_list, str(_lsc), _viewer_link_dataset_prefix)
                    if _lsc is not None
                    else ""
                )
                if _scenario_hit:
                    st.session_state["bbox_viewer_scenario"] = _scenario_hit
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
    _linked_t4dataset = None
    if "bbox_viewer_link_t4dataset" in st.session_state:
        _lt4 = st.session_state.pop("bbox_viewer_link_t4dataset", None)
        if _lt4 is not None and str(_lt4) in t4dataset_list:
            _linked_t4dataset = str(_lt4)
            st.session_state["bbox_viewer_t4dataset"] = _linked_t4dataset
    if has_multiple_t4dataset and t4dataset_list:
        selected_t4dataset = st.selectbox(
            "t4dataset_name",
            t4dataset_list,
            key="bbox_viewer_t4dataset",
        )
    elif _linked_t4dataset is not None:
        selected_t4dataset = _linked_t4dataset
    elif len(t4dataset_list) == 1:
        selected_t4dataset = t4dataset_list[0]

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
elif st.session_state.get("bbox_viewer_topic") not in topic_names and DEFAULT_OBJECTS_TOPIC in topic_names:
    st.session_state["bbox_viewer_topic"] = DEFAULT_OBJECTS_TOPIC

with st.sidebar:
    selected_topic = st.selectbox("topic_name (single)", topic_names, key="bbox_viewer_topic")

_sync_viewer_query_params(
    {
        "viewer_suite": selected_suite,
        "viewer_scenario": selected_scenario,
        "viewer_t4dataset": selected_t4dataset,
        "viewer_topic": selected_topic,
        "viewer_compare": _viewer_compare_mode,
    }
)

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

files_to_load: List[tuple] = [(selected_files[lbl], lbl) for lbl in runs_to_show if lbl in selected_files]
_load_entry_frame_raw = st.session_state.get("bbox_viewer_link_frame", None)
try:
    _load_entry_frame_hint = int(float(str(_load_entry_frame_raw)))
except (TypeError, ValueError):
    _load_entry_frame_hint = None


def _duckdb_like_prefix(text: str) -> str:
    """Escape a user/data string for DuckDB LIKE and append a trailing wildcard."""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        + "%"
    )


def _label_visibility_filters() -> tuple[list[str], list[Any]]:
    filters = [f"label IN ({','.join(['?'] * len(selected_labels))})"]
    params_out: List[Any] = list(selected_labels)
    if has_visibility and selected_visibility:
        filters.append(f"COALESCE(visibility,'UNKNOWN') IN ({','.join(['?'] * len(selected_visibility))})")
        params_out.extend(selected_visibility)
    return filters, params_out


def _first_available_reference_frame() -> int:
    """Frame used to map modern dataset names to legacy suffixed B-side scenarios."""
    if selected_suite is None or selected_scenario is None:
        return 0
    where_parts = [
        "suite_name = ?",
        "scenario_name = ?",
        "topic_name = ?",
        "source = 'GT'",
    ]
    params_tail: List[Any] = [selected_suite, selected_scenario, selected_topic]
    if selected_t4dataset is not None:
        where_parts.append("t4dataset_name = ?")
        params_tail.append(selected_t4dataset)
    label_filters, label_params = _label_visibility_filters()
    try:
        value = con.execute(
            f"""
            SELECT MIN(TRY_CAST(frame_index AS INTEGER)) AS frame_index
            FROM parquet_scan(?)
            WHERE {" AND ".join(where_parts + label_filters)}
            """,
            [filter_file] + params_tail + label_params,
        ).fetchone()[0]
    except Exception:
        value = None
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


if _load_entry_frame_hint is None:
    _load_entry_frame_hint = _first_available_reference_frame()


def _reference_geometry_df() -> pd.DataFrame:
    if selected_suite is None or selected_scenario is None or selected_t4dataset is None:
        return pd.DataFrame()
    label_filters, label_params = _label_visibility_filters()
    where_parts = [
        "suite_name = ?",
        "scenario_name = ?",
        "t4dataset_name = ?",
        "topic_name = ?",
        "source = 'GT'",
        "TRY_CAST(frame_index AS INTEGER) = ?",
    ]
    params_tail: List[Any] = [
        selected_suite,
        selected_scenario,
        selected_t4dataset,
        selected_topic,
        _load_entry_frame_hint,
    ]
    try:
        return con.execute(
            f"""
            SELECT label, x, y
            FROM parquet_scan(?)
            WHERE {" AND ".join(where_parts + label_filters)}
            """,
            [filter_file] + params_tail + label_params,
        ).df()
    except Exception:
        return pd.DataFrame()


_geometry_reference_df = _reference_geometry_df()


def _candidate_topics_for_file(file_path: str) -> list[str]:
    topics = [str(selected_topic)]
    topic_aliases = {
        DEFAULT_OBJECTS_TOPIC: DEFAULT_TRACKING_OBJECTS_TOPIC,
        DEFAULT_TRACKING_OBJECTS_TOPIC: DEFAULT_OBJECTS_TOPIC,
    }
    fallback_topic = topic_aliases.get(str(selected_topic))
    if fallback_topic:
        try:
            has_fallback_topic = bool(
                con.execute(
                    "SELECT COUNT(*) > 0 FROM parquet_scan(?) WHERE topic_name = ?",
                    [file_path, fallback_topic],
                ).fetchone()[0]
            )
        except Exception:
            has_fallback_topic = False
        if has_fallback_topic and fallback_topic not in topics:
            topics.append(fallback_topic)
    return topics


def _nearest_gt_geometry_match(
    file_path: str,
    topic: str,
    prefix_parts: list[str],
    prefix_params: list[Any],
    label_filters: list[str],
    label_params: list[Any],
) -> tuple | None:
    """Pick a legacy candidate whose GT layout best matches the selected reference dataset."""
    ref_df = _geometry_reference_df
    if ref_df.empty or not {"label", "x", "y"}.issubset(ref_df.columns):
        return None
    try:
        cand_df = con.execute(
            f"""
            SELECT suite_name, scenario_name, t4dataset_name, label, x, y
            FROM parquet_scan(?)
            WHERE {" AND ".join(prefix_parts + label_filters)}
              AND source = 'GT'
              AND TRY_CAST(frame_index AS INTEGER) = ?
            """,
            [file_path] + prefix_params + label_params + [_load_entry_frame_hint],
        ).df()
    except Exception:
        return None
    if cand_df.empty:
        return None

    ref_groups = {
        str(label): group[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
        for label, group in ref_df.groupby("label")
    }
    ref_count = int(sum(len(v) for v in ref_groups.values()))
    if ref_count <= 0:
        return None

    best: tuple[float, int, str, str, str, int] | None = None
    group_cols = ["suite_name", "scenario_name", "t4dataset_name"]
    for (suite_hit, scenario_hit, dataset_hit), group in cand_df.groupby(group_cols, dropna=False):
        total_distance = 0.0
        matched_count = 0
        for label, ref_points in ref_groups.items():
            if len(ref_points) == 0:
                continue
            cand_points = (
                group[group["label"].astype(str) == label][["x", "y"]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            if len(cand_points) == 0:
                total_distance += 1000.0 * len(ref_points)
                matched_count += len(ref_points)
                continue
            distances = np.sqrt(((ref_points[:, None, :] - cand_points[None, :, :]) ** 2).sum(axis=2))
            total_distance += float(distances.min(axis=1).sum())
            matched_count += len(ref_points)
        if matched_count <= 0:
            continue
        cand_count = int(len(group))
        count_delta = abs(cand_count - ref_count)
        score = (total_distance / matched_count) + (0.05 * count_delta)
        dataset_text = "" if pd.isna(dataset_hit) else str(dataset_hit)
        candidate = (
            float(score),
            int(count_delta),
            str(scenario_hit),
            str(suite_hit),
            dataset_text,
            cand_count,
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None

    score, count_delta, scenario_hit, suite_hit, dataset_hit, cand_count = best
    return suite_hit, scenario_hit, dataset_hit, int(cand_count), {
        "geometry_score": score,
        "geometry_count_delta": count_delta,
        "geometry_ref_count": ref_count,
        "geometry_candidate_count": cand_count,
        "topic": topic,
    }


def _resolve_load_filter_for_file(file_path: str) -> tuple[str, list[Any], dict]:
    """Resolve one concrete scene/topic for this run's parquet.

    Modern release parquet can be filtered exactly by t4dataset_name. Older pilot
    exports use suffixed suite/scenario names plus a tracking topic, so comparing
    with a modern release needs a per-run fallback that picks one concrete
    suffixed scenario instead of aggregating every matching dataset.
    """
    label_filters, label_params = _label_visibility_filters()
    topics = _candidate_topics_for_file(file_path)

    def _count(where_parts: list[str], params_tail: list[Any]) -> int:
        q = f"SELECT COUNT(*) FROM parquet_scan(?) WHERE {' AND '.join(where_parts + label_filters)}"
        try:
            return int(con.execute(q, [file_path] + params_tail + label_params).fetchone()[0])
        except Exception:
            return 0

    exact_parts: list[str] = []
    exact_params: list[Any] = []
    if selected_suite is not None:
        exact_parts.append("suite_name = ?")
        exact_params.append(selected_suite)
    if selected_scenario is not None:
        exact_parts.append("scenario_name = ?")
        exact_params.append(selected_scenario)

    for topic in topics:
        parts = list(exact_parts)
        params_tail = list(exact_params)
        if selected_t4dataset is not None:
            parts.append("t4dataset_name = ?")
            params_tail.append(selected_t4dataset)
        parts.append("topic_name = ?")
        params_tail.append(topic)
        if _count(parts, params_tail) > 0:
            return " AND ".join(parts + label_filters), params_tail + label_params, {
                "topic": topic,
                "match": "exact_dataset" if selected_t4dataset is not None else "exact_scene",
            }

    for topic in topics:
        parts = list(exact_parts) + ["topic_name = ?"]
        params_tail = list(exact_params) + [topic]
        if _count(parts, params_tail) > 0:
            return " AND ".join(parts + label_filters), params_tail + label_params, {
                "topic": topic,
                "match": "exact_scene",
            }

    if selected_suite is not None and selected_scenario is not None:
        for topic in topics:
            prefix_parts = [
                "(suite_name = ? OR suite_name LIKE ? ESCAPE '\\')",
                "(scenario_name = ? OR scenario_name LIKE ? ESCAPE '\\')",
                "topic_name = ?",
            ]
            prefix_params = [
                selected_suite,
                _duckdb_like_prefix(f"{selected_suite}_"),
                selected_scenario,
                _duckdb_like_prefix(f"{selected_scenario}_"),
                topic,
            ]
            geometry_hit = _nearest_gt_geometry_match(
                file_path,
                topic,
                prefix_parts,
                prefix_params,
                label_filters,
                label_params,
            )
            if geometry_hit is not None:
                suite_hit, scenario_hit, dataset_hit, row_count, geometry_debug = geometry_hit
                concrete_parts = ["suite_name = ?", "scenario_name = ?", "topic_name = ?"]
                concrete_params: List[Any] = [suite_hit, scenario_hit, topic]
                if dataset_hit:
                    concrete_parts.append("t4dataset_name = ?")
                    concrete_params.append(dataset_hit)
                return " AND ".join(concrete_parts + label_filters), concrete_params + label_params, {
                    "topic": topic,
                    "match": "legacy_geometry_scene",
                    "suite": str(suite_hit),
                    "scenario": str(scenario_hit),
                    "t4dataset": str(dataset_hit),
                    "row_count": int(row_count or 0),
                    **geometry_debug,
                }

            group_sql = f"""
                SELECT
                    suite_name,
                    scenario_name,
                    t4dataset_name,
                    SUM(CASE WHEN TRY_CAST(frame_index AS INTEGER) = ? THEN 1 ELSE 0 END) AS entry_frame_rows,
                    COUNT(*) AS row_count
                FROM parquet_scan(?)
                WHERE {" AND ".join(prefix_parts + label_filters)}
                GROUP BY suite_name, scenario_name, t4dataset_name
                ORDER BY entry_frame_rows DESC, scenario_name ASC, row_count DESC
                LIMIT 1
            """
            try:
                hit = con.execute(
                    group_sql,
                    [_load_entry_frame_hint, file_path] + prefix_params + label_params,
                ).fetchone()
            except Exception:
                hit = None
            if hit is None:
                continue
            suite_hit, scenario_hit, dataset_hit, entry_rows, row_count = hit
            concrete_parts = ["suite_name = ?", "scenario_name = ?", "topic_name = ?"]
            concrete_params: List[Any] = [suite_hit, scenario_hit, topic]
            if dataset_hit is not None:
                concrete_parts.append("t4dataset_name = ?")
                concrete_params.append(dataset_hit)
            return " AND ".join(concrete_parts + label_filters), concrete_params + label_params, {
                "topic": topic,
                "match": "legacy_prefixed_scene",
                "suite": str(suite_hit),
                "scenario": str(scenario_hit),
                "t4dataset": "" if dataset_hit is None else str(dataset_hit),
                "entry_frame_rows": int(entry_rows or 0),
                "row_count": int(row_count or 0),
            }

    fallback_parts = ["topic_name = ?"]
    fallback_params: List[Any] = [topics[0]]
    return " AND ".join(fallback_parts + label_filters), fallback_params + label_params, {
        "topic": topics[0],
        "match": "topic_only_fallback",
    }

dfs = []
for file_path, run_label in files_to_load:
    run_where, run_params, _ = _resolve_load_filter_for_file(file_path)
    sql = f"""
SELECT {", ".join(_select_cols)}
FROM parquet_scan(?)
WHERE {run_where}
ORDER BY frame_index
"""
    qparams = [file_path] + run_params
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
        if _viewer_compare_mode:
            _q_three = f"{_q_three}&compare_view={_viewer_compare_mode}"
        _q_three = f"{_q_three}&{infer_external_bbox_alignment_query_params(df)}"
        _viewer_three_url = f"{browser_url_t4.rstrip('/')}/viewer/three?{_q_three}"
        _layer_payload = build_three_layer_payload_all_frames(df)

        _viewer_three_h = 1400
        _transport_stats = render_t4_three_js_embed(_viewer_three_url, _layer_payload, height=_viewer_three_h)
        with st.expander("T4 overlay transport debug", expanded=True):
            # Aggregate stats across all frames in the payload
            _all_frames = _layer_payload.get("frames") or {}
            _total_gt = sum(len(f.get("gt") or []) for f in _all_frames.values())
            _total_pred = sum(len(f.get("pred") or []) for f in _all_frames.values())
            _total_pairs = sum(len(f.get("matched_pairs") or []) for f in _all_frames.values())
            _frames_with_pred = sum(1 for f in _all_frames.values() if len(f.get("pred") or []) > 0)
            _frames_with_gt = sum(1 for f in _all_frames.values() if len(f.get("gt") or []) > 0)
            st.write(f"Payload summary: {len(_all_frames)} frames, total GT={_total_gt}, total EST={_total_pred}, total pairs={_total_pairs}")
            st.write(f"Frames with GT={_frames_with_gt}, frames with EST={_frames_with_pred}")
            st.write(f"compare_runs in payload: {_layer_payload.get('compare_runs', [])}")
            first_frame_key = next(iter(sorted(_all_frames.keys(), key=lambda v: int(v))), "")
            first_frame_payload = _all_frames.get(first_frame_key, {})
            first_gt = (first_frame_payload.get("gt") or [{}])[0]
            first_pred = (first_frame_payload.get("pred") or [{}])[0]
            st.code(_viewer_three_url, language="text")
            # Per-run breakdown of first frame
            _df_first = df[df["frame_index"] == int(first_frame_key)] if first_frame_key and "frame_index" in df.columns else df
            if "run" in _df_first.columns:
                for _rn in sorted(_df_first["run"].dropna().unique()):
                    _rdf_f = _df_first[_df_first["run"] == _rn]
                    _rgt = int((_rdf_f["source"] == "GT").sum())
                    _rest = int((_rdf_f["source"] == "EST").sum())
                    st.write(f"  Frame {first_frame_key} Run {_rn}: GT={_rgt}, EST={_rest}")
                    if _rest > 0:
                        _est_sample = _rdf_f[_rdf_f["source"] == "EST"].head(3)
                        _show_cols = [c for c in ["x", "y", "z", "length", "width", "height", "yaw", "label", "uuid"] if c in _est_sample.columns]
                        st.write(f"    Sample EST: {_est_sample[_show_cols].to_dict('records')}")
            st.json(
                {
                    "alignment_version": EXTERNAL_BBOX_ALIGNMENT_VERSION,
                    "transport": _transport_stats,
                    "selected_context": {
                        "runs": [lbl for _, lbl in files_to_load],
                        "topic": selected_topic,
                        "labels": list(selected_labels),
                        "visibility": list(selected_visibility or []),
                        "suite": selected_suite,
                        "scenario": selected_scenario,
                        "t4dataset": selected_t4dataset or _ds_t4,
                    },
                    "first_payload_frame": {
                        "frame_index": first_frame_key,
                        "gt_count": len(first_frame_payload.get("gt") or []),
                        "pred_count": len(first_frame_payload.get("pred") or []),
                        "matched_pair_count": len(first_frame_payload.get("matched_pairs") or []),
                    },
                    "first_gt": {
                        key: first_gt.get(key)
                        for key in ("uuid", "label", "status", "x", "y", "z", "length", "width", "height", "yaw", "force_wireframe", "run")
                    },
                    "first_pred": {
                        key: first_pred.get(key)
                        for key in ("uuid", "label", "status", "x", "y", "z", "length", "width", "height", "yaw", "confidence", "run")
                    },
                    "has_gt_corners": "corners" in first_gt,
                    "swap_length_width_detected": infer_legacy_width_length_swapped(df) if "df" in dir() else "unknown",
                    "notes": [
                        "Overlay boxes are sent browser-side as T4BBOX1 binary, not JSON/hex.",
                        "The ArrayBuffer is transferred to the iframe once on iframe load.",
                        "The T4 dataset server receives only normal viewer query params; overlay boxes are not uploaded there.",
                    ],
                }
            )

st.page_link("pages/4_Bounding_Box_Viewer.py", label="Back to Bounding Box & BEV viewer", icon="🖼️")
