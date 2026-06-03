"""
Exercise the T4 visualizer HTTP API (``t4-server``): ``GET /health``, ``GET /datasets``,
``GET /datasets/{t4dataset_id}/scenarios``, and ``POST /render``.
Build embeddable JSON / query strings for T4 dataset context and render payloads.
"""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import pandas as pd
import streamlit as st

from lib.page_chrome import inject_app_page_styles, render_page_hero, section_header
from lib.t4_dataset_embed import (
    build_render_request_embed,
    t4_dataset_context,
    t4_share_query_params,
    target_objects_from_rows,
)
from lib.t4_visualizer_client import (
    DEFAULT_BASE_URL,
    ENV_BASE_URL,
    RenderRequest,
    T4VisualizerClient,
    T4VisualizerError,
    TargetObjectIn,
    browser_base_url,
    render_request_to_json_body,
    render_response_json_for_debug,
    target_object_from_gt_row,
)

st.set_page_config(
    page_title="T4 dataset server",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_app_page_styles()

render_page_hero(
    kicker="Integration",
    title="T4 dataset server & embed helpers",
    description=(
        "Call the Tier4 visualizer HTTP service (same client as Bounding Box Viewer): health, dataset list, "
        "scenarios per dataset (names and frame counts), camera render. Fetch lists, pick ids from the server "
        "or type your own, then render or copy embed JSON."
    ),
    mode="Single Run",
)

if "t4_test_base_url" not in st.session_state:
    st.session_state["t4_test_base_url"] = os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL).rstrip("/")

# Cached API results for pickers
if "t4_dataset_ids" not in st.session_state:
    st.session_state["t4_dataset_ids"] = []
if "t4_last_datasets_payload" not in st.session_state:
    st.session_state["t4_last_datasets_payload"] = None
if "t4_scenario_rows" not in st.session_state:
    st.session_state["t4_scenario_rows"] = []
if "t4_last_scenarios_payload" not in st.session_state:
    st.session_state["t4_last_scenarios_payload"] = None


def _hydrate_t4_from_url() -> None:
    """Fill context + render/embed widgets from ``?render_json=…`` (same JSON as curl ``-d``)."""
    qp = st.query_params
    raw = qp.get("render_json")
    if raw is None:
        return
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if not raw:
        return
    sig = f"render_json:{raw}"
    if st.session_state.get("_t4_hydrate_sig") == sig:
        return
    try:
        body = json.loads(str(raw))
    except json.JSONDecodeError:
        return
    if not isinstance(body, dict):
        return
    st.session_state["t4_ctx_ds"] = str(body.get("t4dataset_id", ""))
    st.session_state["t4_ctx_scen"] = str(body.get("scenario_name", ""))
    try:
        st.session_state["t4_ctx_frame"] = int(body.get("frame_index", 0))
    except (TypeError, ValueError):
        st.session_state["t4_ctx_frame"] = 0
    ver = body.get("version")
    st.session_state["t4_ctx_ver"] = "" if ver is None else str(ver)
    to = body.get("target_objects")
    if isinstance(to, list):
        tgt = json.dumps(to, ensure_ascii=False, indent=2)
        st.session_state["t4_emb_rows"] = tgt
        st.session_state["t4_render_targets"] = tgt
        st.session_state["t4_render_use_tgt"] = len(to) > 0
    else:
        st.session_state["t4_emb_rows"] = "[]"
        st.session_state["t4_render_targets"] = "[]"
        st.session_state["t4_render_use_tgt"] = False
    st.session_state["t4_render_crop"] = bool(body.get("crop_cameras", False))
    st.session_state["t4_render_ann"] = bool(body.get("show_annotations", True))
    st.session_state["_t4_hydrate_sig"] = sig


_hydrate_t4_from_url()

base_url = st.sidebar.text_input(
    "Server base URL",
    key="t4_test_base_url",
    help=f"Server-side API URL. Browser links use `T4_VISUALIZER_BROWSER_BASE_URL` when set.",
)
timeout_s = st.sidebar.number_input("HTTP timeout (s)", min_value=5.0, max_value=600.0, value=120.0, step=5.0)


def _client() -> T4VisualizerClient:
    return T4VisualizerClient(base_url=(base_url or "").strip() or DEFAULT_BASE_URL, timeout=float(timeout_s))


def _bash_single_quoted(s: str) -> str:
    """Wrap *s* for safe use as a bash single-quoted string (e.g. ``-d '…'``)."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _on_dataset_pick() -> None:
    sel = st.session_state.get("t4_pick_ds", "—")
    if sel != "—":
        st.session_state["t4_ctx_ds"] = sel


def _on_scenario_pick() -> None:
    sel = st.session_state.get("t4_pick_scen", "—")
    if sel != "—":
        st.session_state["t4_ctx_scen"] = sel


# --- Shared context (dataset, version, scenario, frame) ---------------------------------
section_header(
    "Context",
    "Fetch lists from the server, then choose **t4dataset_id** and **scenario_name** from the dropdowns "
    "or type any value in the text fields.",
)

row_fetch = st.columns([1, 1, 2])
with row_fetch[0]:
    if st.button("GET /datasets", type="primary", key="t4_btn_datasets"):
        try:
            d = _client().list_datasets()
            st.session_state["t4_last_datasets_payload"] = d
            ds = d.get("datasets")
            st.session_state["t4_dataset_ids"] = [str(x) for x in ds] if isinstance(ds, list) else []
            st.session_state["t4_scenario_rows"] = []
            st.session_state["t4_last_scenarios_payload"] = None
            st.success(f"OK — {len(st.session_state['t4_dataset_ids'])} dataset id(s).")
        except T4VisualizerError as ex:
            st.error(f"{ex} (status={ex.status_code})")
            if ex.response_text:
                st.code(ex.response_text[:4000], language="text")
        except OSError as ex:
            st.error(f"Network error: {ex}")

with row_fetch[1]:
    if st.button("GET /datasets/…/scenarios", type="primary", key="t4_btn_scenarios"):
        _tid = (st.session_state.get("t4_ctx_ds") or "").strip()
        if not _tid:
            st.warning("Set **t4dataset_id** first.")
        else:
            try:
                _ver = (st.session_state.get("t4_ctx_ver") or "").strip() or None
                out = _client().list_dataset_scenarios(_tid, version=_ver)
                st.session_state["t4_last_scenarios_payload"] = out
                rows = out.get("scenarios")
                st.session_state["t4_scenario_rows"] = rows if isinstance(rows, list) else []
                st.success(f"OK — {len(st.session_state['t4_scenario_rows'])} scenario(s).")
            except T4VisualizerError as ex:
                st.error(f"{ex} (status={ex.status_code})")
                if ex.response_text:
                    st.code(ex.response_text[:4000], language="text")
            except OSError as ex:
                st.error(f"Network error: {ex}")

with row_fetch[2]:
    if st.session_state.get("t4_last_datasets_payload") is not None:
        with st.expander("Last GET /datasets JSON", expanded=False):
            st.json(st.session_state["t4_last_datasets_payload"])
    if st.session_state.get("t4_last_scenarios_payload") is not None:
        with st.expander("Last GET /datasets/…/scenarios JSON", expanded=False):
            st.json(st.session_state["t4_last_scenarios_payload"])

_ids = st.session_state["t4_dataset_ids"]
_ds_options = ["—"] + sorted(_ids)
_name_rows = st.session_state["t4_scenario_rows"]
_scen_names: List[str] = []
for r in _name_rows:
    if isinstance(r, dict) and r.get("name") is not None:
        _scen_names.append(str(r["name"]))
_scen_options = ["—"] + sorted(set(_scen_names))

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.selectbox(
        "Pick dataset (from last /datasets)",
        options=_ds_options,
        key="t4_pick_ds",
        on_change=_on_dataset_pick,
        help="Choose a server-reported id, or leave as — and type below.",
    )
    st.text_input(
        "t4dataset_id",
        key="t4_ctx_ds",
        placeholder="uuid or folder id",
    )
with c2:
    st.text_input(
        "version (optional)",
        key="t4_ctx_ver",
        help="Annotation dir version; passed to scenarios and render when non-empty.",
    )
with c3:
    st.selectbox(
        "Pick scenario (from last /scenarios)",
        options=_scen_options,
        key="t4_pick_scen",
        on_change=_on_scenario_pick,
        help="Choose **name** from the server, or type any scenario below.",
    )
    st.text_input(
        "scenario_name",
        key="t4_ctx_scen",
        placeholder="scene name for POST /render",
    )
with c4:
    st.number_input("frame_index", min_value=0, value=0, step=1, key="t4_ctx_frame")

if _name_rows:
    st.caption(
        "Valid **frame_index** for each scene is **0 … nbr_samples − 1** (see table). "
        "Use **Render & embed** to request PNGs."
    )
    st.dataframe(pd.DataFrame(_name_rows), width='stretch', hide_index=True)

st.divider()

tab_overview, tab_render = st.tabs(["Overview", "Render & embed JSON"])

with tab_overview:
    section_header("/health", "GET — server liveness.")
    if st.button("GET /health", type="primary", key="t4_btn_health"):
        try:
            h = _client().health()
            st.success("OK")
            st.json(h)
        except T4VisualizerError as ex:
            st.error(f"{ex} (status={ex.status_code})")
            if ex.response_text:
                st.code(ex.response_text[:4000], language="text")
        except OSError as ex:
            st.error(f"Network error: {ex}")

with tab_render:
    section_header("POST /render", "Request camera PNGs; optional ``target_objects`` from JSON below.")
    ds_id = (st.session_state.get("t4_ctx_ds") or "").strip()
    scen = (st.session_state.get("t4_ctx_scen") or "").strip()
    frame = int(st.session_state.get("t4_ctx_frame") or 0)
    ver_raw = (st.session_state.get("t4_ctx_ver") or "").strip()
    version_opt: Optional[str] = ver_raw if ver_raw else None

    st.caption(
        f"Using context: **t4dataset_id**=`{ds_id or '…'}` · **scenario_name**=`{scen or '…'}` · "
        f"**frame_index**={frame}"
        + (f" · **version**=`{version_opt}`" if version_opt else "")
    )

    tgt_json = st.text_area(
        "target_objects (JSON array, optional)",
        value="[]",
        height=140,
        key="t4_render_targets",
        help="List of objects with uuid/x/y/z/label/width/length/height/yaw (matches GT row shape).",
    )
    o1, o2, o3 = st.columns(3)
    with o1:
        crop = st.checkbox("crop_cameras", value=False, key="t4_render_crop")
    with o2:
        show_ann = st.checkbox("show_annotations", value=True, key="t4_render_ann")
    with o3:
        overlay_gt = st.checkbox("Use target_objects in request", value=True, key="t4_render_use_tgt")

    req: Optional[RenderRequest] = None
    parse_err: Optional[str] = None
    if overlay_gt:
        try:
            raw = json.loads(tgt_json or "[]")
            if not isinstance(raw, list):
                parse_err = "target_objects JSON must be an array"
            else:
                objs: List[TargetObjectIn] = []
                for item in raw:
                    if not isinstance(item, dict):
                        parse_err = "each target must be an object"
                        break
                    d = target_object_from_gt_row(item)
                    objs.append(TargetObjectIn(**d))
                if parse_err is None:
                    req = RenderRequest(
                        t4dataset_id=ds_id,
                        scenario_name=scen,
                        frame_index=frame,
                        target_objects=objs,
                        crop_cameras=crop,
                        show_annotations=show_ann,
                        version=version_opt,
                    )
        except json.JSONDecodeError as ex:
            parse_err = f"Invalid JSON: {ex}"
    else:
        req = RenderRequest(
            t4dataset_id=ds_id,
            scenario_name=scen,
            frame_index=frame,
            target_objects=[],
            crop_cameras=crop,
            show_annotations=show_ann,
            version=version_opt,
        )

    if parse_err:
        st.warning(parse_err)

    col_go, col_prev = st.columns([1, 2])
    with col_go:
        do_render = st.button("POST /render", type="primary", key="t4_btn_render", disabled=req is None)
    with col_prev:
        if req is not None:
            with st.expander("Request body preview", expanded=False):
                st.json(render_request_to_json_body(req))

    if do_render and req is not None:
        try:
            with st.spinner("Rendering…"):
                res = _client().render(req)
            imgs = res.decode_all_images()
            cap_parts = [
                f"sample_token={res.sample_token!r}",
                f"timestamp_us={res.timestamp_us}",
            ]
            if res.elapsed_ms is not None:
                cap_parts.append(f"elapsed_ms={res.elapsed_ms}")
            if res.tier4_load_ms is not None:
                cap_parts.append(f"tier4_load_ms={res.tier4_load_ms}")
            if res.render_ms is not None:
                cap_parts.append(f"render_ms={res.render_ms}")
            st.caption(" · ".join(cap_parts))
            if res.raw_json is not None:
                with st.expander("Response JSON (debug)", expanded=False):
                    st.json(render_response_json_for_debug(res.raw_json))
            if not imgs:
                st.info("No images in response.")
            else:
                n = min(len(imgs), 6)
                cols = st.columns(n)
                for i in range(n):
                    label, png = imgs[i]
                    cols[i].image(png, caption=label, width='stretch')
                if len(imgs) > n:
                    st.caption(f"Showing first {n} of {len(imgs)} images.")
        except T4VisualizerError as ex:
            st.error(f"{ex} (status={ex.status_code})")
            if ex.response_text:
                st.code(ex.response_text[:4000], language="text")
        except OSError as ex:
            st.error(f"Network error: {ex}")

    st.divider()
    section_header(
        "Embed helpers",
        "Same **context** fields as above. Copy structured context, query strings, and full ``POST /render`` JSON.",
    )

    emb_ds = (st.session_state.get("t4_ctx_ds") or "").strip()
    emb_scen = (st.session_state.get("t4_ctx_scen") or "").strip()
    emb_frame = int(st.session_state.get("t4_ctx_frame") or 0)

    emb_ta = st.text_area(
        "Optional GT rows as JSON array (for target_objects_from_rows)",
        value="[]",
        height=120,
        key="t4_emb_rows",
    )

    rows_err: Optional[str] = None
    rows_list: List[dict[str, Any]] = []
    try:
        parsed = json.loads(emb_ta or "[]")
        if not isinstance(parsed, list):
            rows_err = "Must be a JSON array"
        else:
            for i, row in enumerate(parsed):
                if not isinstance(row, dict):
                    rows_err = f"Item {i} is not an object"
                    break
            if rows_err is None:
                rows_list = [r for r in parsed if isinstance(r, dict)]
    except json.JSONDecodeError as ex:
        rows_err = str(ex)

    if rows_err:
        st.warning(rows_err)

    ctx = t4_dataset_context(emb_ds, emb_scen, frame_index=emb_frame)
    emb_ver = (st.session_state.get("t4_ctx_ver") or "").strip()
    full = build_render_request_embed(
        emb_ds,
        emb_scen,
        emb_frame,
        target_rows=rows_list if rows_list else None,
        show_annotations=bool(st.session_state.get("t4_render_ann", True)),
        crop_cameras=bool(st.session_state.get("t4_render_crop", False)),
        version=emb_ver if emb_ver else None,
    )
    viz_base = browser_base_url((base_url or "").strip() or DEFAULT_BASE_URL)
    q = t4_share_query_params(emb_ds, emb_scen, frame_index=emb_frame)
    render_get_url = f"{viz_base}/render?{q}"

    st.subheader("Render GET URL")
    st.caption(
        "GET-style URL on the **visualizer server** (same **Server base URL** as API calls). "
        "Requires **GET /render** with ``t4dataset_id``, ``scenario_name``, ``frame_index``; otherwise use **curl** (POST JSON) below."
    )
    st.markdown(f"[{render_get_url}]({render_get_url})")

    if rows_list:
        st.subheader("target_objects_from_rows (preview)")
        st.json(target_objects_from_rows(rows_list))

    curl_base = (base_url or "").strip() or DEFAULT_BASE_URL
    body_pretty = json.dumps(full["post_render_json"], indent=2, ensure_ascii=False)
    curl_lines = (
        f"curl -sS {curl_base}/render \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d {_bash_single_quoted(body_pretty)}"
    )
    st.subheader("curl")
    st.code(curl_lines, language="bash")
