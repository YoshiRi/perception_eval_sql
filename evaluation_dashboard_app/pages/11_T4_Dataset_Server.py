"""
Exercise the T4 visualizer HTTP API (``t4-server``): ``GET /health``, ``GET /datasets``, ``POST /render``.
Build embeddable JSON / query strings for T4 dataset context and render payloads.
"""
from __future__ import annotations

import json
import os
import shlex
from typing import Any, List, Optional

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
    render_request_to_json_body,
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
        "and camera render. Generate JSON and query strings to embed T4dataset id, scenario, and frame "
        "in tooling or documentation."
    ),
    mode="Single Run",
)

if "t4_test_base_url" not in st.session_state:
    st.session_state["t4_test_base_url"] = os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL).rstrip("/")

base_url = st.sidebar.text_input(
    "Server base URL",
    key="t4_test_base_url",
    help=f"Override env {ENV_BASE_URL} for this session.",
)
timeout_s = st.sidebar.number_input("HTTP timeout (s)", min_value=5.0, max_value=600.0, value=120.0, step=5.0)


def _client() -> T4VisualizerClient:
    return T4VisualizerClient(base_url=(base_url or "").strip() or DEFAULT_BASE_URL, timeout=float(timeout_s))


tab_health, tab_ds, tab_render, tab_embed = st.tabs(
    ["Health", "Datasets", "Render", "Embed JSON"]
)

with tab_health:
    section_header("/health", "GET — server liveness and any metadata the service returns.")
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

with tab_ds:
    section_header("/datasets", "GET — ``data_dir`` and registered dataset ids under the server.")
    if st.button("GET /datasets", type="primary", key="t4_btn_datasets"):
        try:
            d = _client().list_datasets()
            st.success("OK")
            st.json(d)
            ds = d.get("datasets")
            if isinstance(ds, list) and ds:
                st.caption(f"{len(ds)} dataset id(s) returned.")
        except T4VisualizerError as ex:
            st.error(f"{ex} (status={ex.status_code})")
            if ex.response_text:
                st.code(ex.response_text[:4000], language="text")
        except OSError as ex:
            st.error(f"Network error: {ex}")

with tab_render:
    section_header("POST /render", "Request camera PNGs; optional ``target_objects`` from JSON below.")
    c1, c2, c3 = st.columns(3)
    with c1:
        ds_id = st.text_input("t4dataset_id", value="", key="t4_render_ds", placeholder="dataset folder id")
    with c2:
        scen = st.text_input("scenario_name", value="", key="t4_render_scen", placeholder="scenario")
    with c3:
        frame = st.number_input("frame_index", min_value=0, value=0, step=1, key="t4_render_frame")

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
                        t4dataset_id=ds_id.strip(),
                        scenario_name=scen.strip(),
                        frame_index=int(frame),
                        target_objects=objs,
                        crop_cameras=crop,
                        show_annotations=show_ann,
                    )
        except json.JSONDecodeError as ex:
            parse_err = f"Invalid JSON: {ex}"
    else:
        req = RenderRequest(
            t4dataset_id=ds_id.strip(),
            scenario_name=scen.strip(),
            frame_index=int(frame),
            target_objects=[],
            crop_cameras=crop,
            show_annotations=show_ann,
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
            st.caption(f"sample_token={res.sample_token!r} · timestamp_us={res.timestamp_us}")
            if not imgs:
                st.info("No images in response.")
            else:
                n = min(len(imgs), 6)
                cols = st.columns(n)
                for i in range(n):
                    label, png = imgs[i]
                    cols[i].image(png, caption=label, use_container_width=True)
                if len(imgs) > n:
                    st.caption(f"Showing first {n} of {len(imgs)} images.")
        except T4VisualizerError as ex:
            st.error(f"{ex} (status={ex.status_code})")
            if ex.response_text:
                st.code(ex.response_text[:4000], language="text")
        except OSError as ex:
            st.error(f"Network error: {ex}")

with tab_embed:
    section_header(
        "Embed helpers",
        "Copy structured context, query strings, and full ``POST /render`` JSON for scripts or docs.",
    )
    e1, e2, e3 = st.columns(3)
    with e1:
        emb_ds = st.text_input("t4dataset_id", value="", key="t4_emb_ds")
    with e2:
        emb_scen = st.text_input("scenario_name", value="", key="t4_emb_scen")
    with e3:
        emb_frame = st.number_input("frame_index", min_value=0, value=0, step=1, key="t4_emb_frame")

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

    ctx = t4_dataset_context(emb_ds.strip(), emb_scen.strip(), frame_index=int(emb_frame))
    q = t4_share_query_params(emb_ds.strip(), emb_scen.strip(), frame_index=int(emb_frame))

    st.subheader("t4_dataset_context")
    st.json(ctx)

    st.subheader("Shareable query fragment")
    st.code(q, language="text")

    full = build_render_request_embed(
        emb_ds.strip(),
        emb_scen.strip(),
        int(emb_frame),
        target_rows=rows_list if rows_list else None,
        show_annotations=True,
        crop_cameras=False,
    )
    st.subheader("context + post_render_json")
    st.json(full)

    if rows_list:
        st.subheader("target_objects_from_rows (preview)")
        st.json(target_objects_from_rows(rows_list))

    curl_base = shlex.quote((base_url or "").strip() or DEFAULT_BASE_URL)
    body_s = json.dumps(full["post_render_json"])
    st.subheader("Example curl")
    st.code(
        f"curl -sS {curl_base}/render -H 'Content-Type: application/json' -d {shlex.quote(body_s)}",
        language="bash",
    )
