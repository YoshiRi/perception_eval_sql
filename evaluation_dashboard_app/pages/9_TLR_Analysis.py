"""
TLR (Traffic Light Recognition) Evaluation Analysis page.
Visualizes criteria matrices, vehicle status vs traffic light type, and critical/priority zones.
Supports Single (one dataset) or Compare (two datasets: Baseline A vs Compare B).
Supports shareable URLs via query params: mode, path_a, path_b.
"""

import json
import html
import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from urllib.parse import quote

from lib.tlr_eval_analyzer import TLREvaluationAnalyzer
from lib.path_utils import get_data_root, path_display, list_tlr_result_directories
from lib.t4_visualizer_client import DEFAULT_BASE_URL, ENV_BASE_URL
from lib.page_chrome import (
    inject_app_page_styles,
    render_loaded_data_section,
    render_page_hero,
    render_share_link_callout,
    section_header,
)

st.set_page_config(
    page_title="TLR Analysis",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded",
)
inject_app_page_styles()

# ====== URL QUERY PARAMS (for shareable links) ======
params = st.query_params
url_mode = params.get("mode")       # "single" / "compare" / None
url_path_a = params.get("path_a")  # relative path under data root
url_path_b = params.get("path_b")  # for compare mode

# ----- Helpers -----
data_root = get_data_root()


def path_to_tlr_key(path: Path) -> str:
    """Stable key for URL: path relative to data root, or '.' for root."""
    try:
        rel = path.resolve().relative_to(data_root.resolve())
        return str(rel).replace("\\", "/") if str(rel) != "." else "."
    except ValueError:
        return path.name or "."


def format_tlr_option(item):
    path, _count = item
    try:
        rel = path.relative_to(data_root)
        label = str(rel).replace("\\", "/") if str(rel) != "." else data_root.name or "."
    except ValueError:
        label = path.name or str(path)
    return label

def get_or_load_analyzer(resolved_path: str):
    """Load analyzer for path; cache in session_state by path."""
    if not resolved_path:
        return None
    cache_key = "tlr_analyzer_cache_v2"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    cache = st.session_state[cache_key]
    if resolved_path not in cache:
        with st.spinner(f"Loading TLR results: {Path(resolved_path).name}..."):
            analyzer = TLREvaluationAnalyzer(resolved_path)
            analyzer.load_all_results()
            if not analyzer.scenario_results:
                return None
            analyzer.extract_criteria_data()
            analyzer.pre_calculate_all_data()
            cache[resolved_path] = analyzer
    return cache[resolved_path]


def _dataframe_to_json_bytes(df: pd.DataFrame, export_kind: str) -> bytes:
    """Serialize a DataFrame to a stable JSON payload for downstream viewers."""
    payload = {
        "format_version": 1,
        "export_kind": export_kind,
        "columns": df.columns.tolist(),
        "records": df.to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _build_tlr_eval_payload_by_frame(df: pd.DataFrame | None) -> dict:
    """Build per-frame TLR evaluation payload for the embedded viewer."""
    if df is None or df.empty or "frame_index" not in df.columns:
        return {"type": "tlr_eval_clear"}

    frames: dict[str, dict] = {}
    ordered = df.sort_values(["frame_index", "scenario"]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        try:
            frame_key = str(int(row.get("frame_index", 0)))
        except (TypeError, ValueError):
            continue
        if frame_key in frames:
            continue

        def _float_or_none(value):
            try:
                return None if pd.isna(value) else float(value)
            except Exception:
                return None

        def _string_or_none(value):
            try:
                if pd.isna(value) or value == "":
                    return None
            except Exception:
                pass
            return str(value)

        frames[frame_key] = {
            "scenario": str(row.get("scenario", "") or ""),
            "t4dataset_id": str(row.get("t4dataset_id", "") or ""),
            "frame_name": str(row.get("frame_name", "") or ""),
            "status": str(row.get("status", "") or ""),
            "speed_kph": _float_or_none(row.get("speed_kph")),
            "yaw_rate_deg_s": _float_or_none(row.get("yaw_rate_deg_s")),
            "current_time": _float_or_none(row.get("current_time")),
            "current_time_us": (
                int(round(float(row.get("current_time")) * 1_000_000))
                if _float_or_none(row.get("current_time")) not in (None, 0.0)
                else None
            ),
            "traffic_light_type": str(row.get("traffic_light_type", "") or ""),
            "evaluation_result": str(row.get("traffic_light_type", "") or ""),
            "criteria": str(row.get("criteria", "") or ""),
            "tp": _string_or_none(row.get("tp")),
            "fp": _string_or_none(row.get("fp")),
            "fn": _string_or_none(row.get("fn")),
            "tn": _string_or_none(row.get("tn")),
        }
    return {"type": "tlr_eval_by_frame", "frames": frames}


def _render_tlr_viewer_embed(viewer_url: str, payload: dict, *, iframe_id: str, height: int = 1400) -> None:
    """Embed `/viewer/tlr` and post a frame-indexed evaluation payload into the iframe."""
    payload_json = json.dumps(payload, ensure_ascii=True)
    payload_hex = payload_json.encode("utf-8").hex()
    iframe_src = html.escape(viewer_url, quote=True)
    components.html(
        (
            f'<iframe id="{iframe_id}" src="{iframe_src}" '
            f'width="100%" height="{height}" style="border:none;border-radius:8px;background:#e2e8f0" '
            f'allowfullscreen allow="fullscreen *" '
            f'loading="lazy" title="Traffic light viewer" referrerpolicy="no-referrer-when-downgrade"></iframe>'
            "<script>"
            "(()=>{"
            f"const iframe=document.getElementById('{iframe_id}');"
            f"const payloadHex='{payload_hex}';"
            "const hexToUtf8=(hex)=>{"
            "if(!hex||hex.length%2!==0)return '';"
            "const bytes=new Uint8Array(hex.length/2);"
            "for(let i=0;i<hex.length;i+=2){bytes[i/2]=parseInt(hex.slice(i,i+2),16)||0;}"
            "return new TextDecoder().decode(bytes);"
            "};"
            "let payload={type:'tlr_eval_clear'};"
            "try{"
            "const payloadJson=hexToUtf8(payloadHex);"
            "payload=JSON.parse(payloadJson);"
            "const fc=payload.frames&&typeof payload.frames==='object'?Object.keys(payload.frames).length:0;"
            "console.info('[tlr-debug] payload prepared', {type:payload.type,frames:fc});"
            "}catch(err){"
            "console.error('[tlr-debug] payload parse failed', err);"
            "}"
            "let postCount=0;"
            "const post=(reason)=>{"
            "if(!iframe||!iframe.contentWindow)return;"
            "let targetOrigin='*';"
            "try{ targetOrigin = new URL(iframe.src, window.location.href).origin || '*'; }catch(_){ targetOrigin='*'; }"
            "postCount+=1;"
            "iframe.contentWindow.postMessage(payload,targetOrigin);"
            "console.info('[tlr-debug] postMessage sent', {reason,postCount,targetOrigin,payloadType:payload.type});"
            "};"
            "iframe.addEventListener('load',()=>{"
            "post('iframe-load');"
            "let n=0;"
            "const t=setInterval(()=>{post('retry');n+=1;if(n>12)clearInterval(t);},250);"
            "});"
            "setTimeout(()=>post('initial-delay-300ms'),300);"
            "setTimeout(()=>post('initial-delay-1200ms'),1200);"
            "})();"
            "</script>"
        ),
        height=height + 8,
        scrolling=False,
    )


def _render_tlr_viewer_tab(detail_sources: dict[str, pd.DataFrame | None], *, key_prefix: str) -> None:
    st.subheader("Embedded traffic light viewer")
    st.caption("Pick a dataset from the current TLR details, then load the external `/viewer/tlr` page inline.")

    if f"{key_prefix}_base_url" not in st.session_state:
        st.session_state[f"{key_prefix}_base_url"] = (
            (os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        )

    base_url = st.text_input(
        "T4 server base URL",
        key=f"{key_prefix}_base_url",
        help=f"Default from env `{ENV_BASE_URL}`. The viewer URL is `/viewer/tlr?t4dataset_id=...&frame_index=...`.",
    )

    available_labels = [label for label, df in detail_sources.items() if df is not None and not df.empty]
    if not available_labels:
        st.info("No TLR detail rows available to drive the viewer.")
        return

    source_label = available_labels[0]
    if len(available_labels) > 1:
        source_label = st.radio(
            "Use rows from",
            available_labels,
            horizontal=True,
            key=f"{key_prefix}_source_label",
        )

    details_df = detail_sources[source_label].copy()
    details_df = details_df[details_df["t4dataset_id"].fillna("").astype(str) != ""].copy()
    if details_df.empty:
        st.info("The selected rows do not contain any `t4dataset_id` values.")
        return

    dataset_options = sorted(details_df["t4dataset_id"].astype(str).unique().tolist())
    selected_dataset = st.selectbox(
        "Candidate t4dataset_id",
        dataset_options,
        key=f"{key_prefix}_dataset_id",
    )
    dataset_rows = details_df[details_df["t4dataset_id"].astype(str) == selected_dataset].copy()

    if dataset_rows.empty:
        st.info("No rows match the current dataset selection.")
        return

    dataset_rows = dataset_rows.sort_values(["scenario", "frame_index"]).reset_index(drop=True)
    selected_row = dataset_rows.iloc[0]
    selected_frame = int(selected_row["frame_index"])
    payload = _build_tlr_eval_payload_by_frame(dataset_rows)

    viewer_url = f"{base_url.rstrip('/')}/viewer/tlr?t4dataset_id={quote(selected_dataset, safe='')}&frame_index={selected_frame}"
    st.markdown(f"[Open `/viewer/tlr` in new tab]({viewer_url})")
    st.caption(
        f"Using the first available frame for this dataset: `frame_index={selected_frame}` from `{selected_row['scenario']}`."
    )

    preview_cols = ["scenario", "frame_index", "status", "traffic_light_type", "criteria"]
    if "frame_name" in dataset_rows.columns:
        preview_cols.insert(2, "frame_name")
    with st.expander("Matching rows", expanded=False):
        st.dataframe(dataset_rows[preview_cols].sort_values(["scenario", "frame_index"]), width="stretch", hide_index=True)

    _render_tlr_viewer_embed(viewer_url, payload, iframe_id=f"{key_prefix}_iframe", height=1600)


def _render_single_tabs(analyzer, tab_criteria, tab_vehicle, tab_critical, tab_details, tab_tlr_viewer):
    with tab_criteria:
        st.subheader("Criteria: TP rate and total frames")
        criteria_df = analyzer.create_criteria_matrix()
        st.dataframe(criteria_df, width='stretch', hide_index=True)
        criteria_df = criteria_df.copy()
        criteria_df["criteria_num"] = criteria_df["Criteria"].str.replace("criteria_", "").astype(int)
        fig1 = px.line(criteria_df, x="criteria_num", y="TP rate", title="TP rate by criteria", markers=True)
        fig1.update_layout(xaxis_title="Criteria number", yaxis_title="TP rate", yaxis_range=[0, 1.1])
        st.plotly_chart(fig1, width='stretch')
        fig2 = px.bar(criteria_df, x="criteria_num", y="Number of total frames", title="Total frames by criteria")
        fig2.update_layout(xaxis_title="Criteria number")
        st.plotly_chart(fig2, width='stretch')

    with tab_vehicle:
        st.subheader("Vehicle status vs traffic light type (TP rate)")
        status_df = analyzer.create_vehicle_status_matrix()
        tlr_cols = [c for c in status_df.columns if c != "Vehicle Status"]
        fig = go.Figure(
            data=go.Heatmap(
                z=status_df[tlr_cols].values,
                x=tlr_cols,
                y=status_df["Vehicle Status"].tolist(),
                colorscale="RdYlGn", zmin=0, zmax=1,
                text=[[f"{v:.3f}" for v in row] for row in status_df[tlr_cols].values],
                texttemplate="%{text}", textfont={"size": 9}, hoverongaps=False,
            )
        )
        fig.update_layout(title="TP rate: Vehicle status vs traffic light type", height=400, xaxis={"tickangle": -45})
        st.plotly_chart(fig, width='stretch')
        st.subheader("Raw counts (TP / Total)")
        st.dataframe(analyzer.create_vehicle_status_counts_matrix(), width='stretch', hide_index=True)

    with tab_critical:
        st.subheader("Critical (criteria 5–6) and priority (criteria 2–4) zones")
        cp_df = analyzer.create_vehicle_status_critical_priority_matrix()
        tlr_cols_cp = [c for c in cp_df.columns if c != "Vehicle Status"]
        fig_cp = go.Figure(
            data=go.Heatmap(
                z=cp_df[tlr_cols_cp].values, x=tlr_cols_cp, y=cp_df["Vehicle Status"].tolist(),
                colorscale="RdYlGn", zmin=0, zmax=1,
                text=[[f"{v:.3f}" for v in row] for row in cp_df[tlr_cols_cp].values],
                texttemplate="%{text}", textfont={"size": 8}, hoverongaps=False,
            )
        )
        fig_cp.update_layout(
            title="TP rate: Vehicle status vs traffic light type (critical & priority zones)",
            height=400, xaxis={"tickangle": -45},
        )
        st.plotly_chart(fig_cp, width='stretch')
        st.subheader("Raw counts (TP / Total)")
        st.dataframe(analyzer.create_vehicle_status_critical_priority_counts_matrix(), width='stretch', hide_index=True)

    with tab_details:
        st.subheader("Per-frame vehicle status and TLR details")
        details_df = analyzer.get_vehicle_status_details_df()
        if details_df is not None and not details_df.empty:
            st.caption("One row per frame. Use filters to narrow down by scenario, status, or traffic light type.")
            st.dataframe(details_df, width='stretch', hide_index=True)
            dl_col_csv, dl_col_json = st.columns(2)
            with dl_col_csv:
                st.download_button(
                    "Download as CSV",
                    data=details_df.to_csv(index=False).encode("utf-8"),
                    file_name="tlr_details.csv",
                    mime="text/csv",
                    key="tlr_dl_single_tab_csv",
                )
            with dl_col_json:
                st.download_button(
                    "Download as JSON",
                    data=_dataframe_to_json_bytes(details_df, export_kind="single_dataset_details"),
                    file_name="tlr_details.json",
                    mime="application/json",
                    key="tlr_dl_single_tab_json",
                )
        else:
            st.info("No vehicle status details available.")

    with tab_tlr_viewer:
        _render_tlr_viewer_tab({"Current run": analyzer.get_vehicle_status_details_df()}, key_prefix="tlr_single_viewer")


def _render_compare_tabs(analyzer_a, analyzer_b, label_a, label_b, tab_criteria, tab_vehicle, tab_critical, tab_details, tab_tlr_viewer):
    with tab_criteria:
        st.subheader("Criteria: A vs B (TP rate and delta)")
        df_a = analyzer_a.create_criteria_matrix()
        df_b = analyzer_b.create_criteria_matrix()
        compare_criteria = df_a[["Criteria"]].copy()
        compare_criteria["TP rate A"] = df_a["TP rate"].values
        compare_criteria["TP rate B"] = df_b["TP rate"].values
        compare_criteria["Δ (B − A)"] = compare_criteria["TP rate B"] - compare_criteria["TP rate A"]
        st.dataframe(compare_criteria, width='stretch', hide_index=True)
        compare_criteria["criteria_num"] = compare_criteria["Criteria"].str.replace("criteria_", "").astype(int)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=compare_criteria["criteria_num"], y=compare_criteria["TP rate A"], name=label_a, mode="lines+markers"))
        fig.add_trace(go.Scatter(x=compare_criteria["criteria_num"], y=compare_criteria["TP rate B"], name=label_b, mode="lines+markers"))
        fig.update_layout(title="TP rate by criteria: A vs B", xaxis_title="Criteria number", yaxis_title="TP rate", yaxis_range=[0, 1.1])
        st.plotly_chart(fig, width='stretch')
        delta_vals = compare_criteria["Δ (B − A)"].values
        bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in delta_vals]
        fig_delta = go.Figure(
            data=go.Bar(
                x=compare_criteria["criteria_num"],
                y=delta_vals,
                marker_color=bar_colors,
                text=[f"{v:+.3f}" for v in delta_vals],
                textposition="outside",
            )
        )
        fig_delta.update_layout(
            title="TP rate delta (B − A) by criteria",
            xaxis_title="Criteria number",
            yaxis_title="Δ (B − A)",
            showlegend=False,
        )
        fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_delta, width='stretch')

    with tab_vehicle:
        st.subheader("Vehicle status vs TLR type: A vs B (TP rate delta)")
        status_a = analyzer_a.create_vehicle_status_matrix()
        status_b = analyzer_b.create_vehicle_status_matrix()
        tlr_cols = [c for c in status_a.columns if c != "Vehicle Status"]
        delta_df = status_a[["Vehicle Status"]].copy()
        for c in tlr_cols:
            delta_df[c] = status_b[c].values - status_a[c].values
        fig = go.Figure(
            data=go.Heatmap(
                z=delta_df[tlr_cols].values,
                x=tlr_cols,
                y=delta_df["Vehicle Status"].tolist(),
                colorscale=[[0, "#c0392b"], [0.25, "#e74c3c"], [0.5, "#f5f5f5"], [0.75, "#27ae60"], [1, "#1e8449"]],
                zmin=-1,
                zmax=1,
                zmid=0,
                text=[[f"{v:+.3f}" for v in row] for row in delta_df[tlr_cols].values],
                texttemplate="%{text}", textfont={"size": 8}, hoverongaps=False,
            )
        )
        fig.update_layout(
            title=f"TP rate delta (B − A): Vehicle status vs traffic light type",
            height=400, xaxis={"tickangle": -45},
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Green = B better, Red = A better.")
        with st.expander("Raw A"):
            st.dataframe(analyzer_a.create_vehicle_status_counts_matrix(), width='stretch', hide_index=True)
        with st.expander("Raw B"):
            st.dataframe(analyzer_b.create_vehicle_status_counts_matrix(), width='stretch', hide_index=True)

    with tab_critical:
        st.subheader("Critical & priority zones: A vs B (TP rate delta)")
        cp_a = analyzer_a.create_vehicle_status_critical_priority_matrix()
        cp_b = analyzer_b.create_vehicle_status_critical_priority_matrix()
        tlr_cols_cp = [c for c in cp_a.columns if c != "Vehicle Status"]
        delta_cp = cp_a[["Vehicle Status"]].copy()
        for c in tlr_cols_cp:
            delta_cp[c] = cp_b[c].values - cp_a[c].values
        fig_cp = go.Figure(
            data=go.Heatmap(
                z=delta_cp[tlr_cols_cp].values, x=tlr_cols_cp, y=delta_cp["Vehicle Status"].tolist(),
                colorscale=[[0, "#c0392b"], [0.25, "#e74c3c"], [0.5, "#f5f5f5"], [0.75, "#27ae60"], [1, "#1e8449"]],
                zmin=-1,
                zmax=1,
                zmid=0,
                text=[[f"{v:+.3f}" for v in row] for row in delta_cp[tlr_cols_cp].values],
                texttemplate="%{text}", textfont={"size": 7}, hoverongaps=False,
            )
        )
        fig_cp.update_layout(
            title="TP rate delta (B − A): Critical & priority zones",
            height=400, xaxis={"tickangle": -45},
        )
        st.plotly_chart(fig_cp, width='stretch')
        with st.expander("Raw A"):
            st.dataframe(analyzer_a.create_vehicle_status_critical_priority_counts_matrix(), width='stretch', hide_index=True)
        with st.expander("Raw B"):
            st.dataframe(analyzer_b.create_vehicle_status_critical_priority_counts_matrix(), width='stretch', hide_index=True)

    with tab_details:
        st.subheader("Vehicle status details")
        details_a = analyzer_a.get_vehicle_status_details_df()
        details_b = analyzer_b.get_vehicle_status_details_df()
        if details_a is not None and not details_a.empty and details_b is not None and not details_b.empty:
            merge_keys = ["scenario", "frame_index"]
            a_sub = details_a[merge_keys + ["t4dataset_id", "frame_name", "status", "traffic_light_type"]].copy()
            a_sub = a_sub.rename(
                columns={
                    "t4dataset_id": f"t4dataset_id ({label_a})",
                    "frame_name": "frame_name_a",
                    "status": "status_a",
                    "traffic_light_type": f"traffic_light_type ({label_a})",
                }
            )
            b_sub = details_b[merge_keys + ["t4dataset_id", "frame_name", "status", "traffic_light_type"]].copy()
            b_sub = b_sub.rename(
                columns={
                    "t4dataset_id": f"t4dataset_id ({label_b})",
                    "frame_name": "frame_name_b",
                    "status": "status_b",
                    "traffic_light_type": f"traffic_light_type ({label_b})",
                }
            )
            merged = a_sub.merge(b_sub, on=merge_keys, how="inner")
            dataset_col_a = f"t4dataset_id ({label_a})"
            dataset_col_b = f"t4dataset_id ({label_b})"
            tlr_col_a = f"traffic_light_type ({label_a})"
            tlr_col_b = f"traffic_light_type ({label_b})"
            merged["_diff"] = merged[tlr_col_a] != merged[tlr_col_b]
            diff_tlr = merged[merged["_diff"]]

            view_mode = st.radio(
                "Compare view",
                [
                    "Only frames where traffic light type differs (A vs B)",
                    "All frames (mark differences)",
                ],
                horizontal=True,
                key="tlr_compare_view_mode",
            )
            show_only_diff = view_mode == "Only frames where traffic light type differs (A vs B)"

            # ---- Filters & sort (apply to both views) ----
            all_scenarios = sorted(merged["scenario"].unique().tolist())
            all_statuses = ["Driving", "Turning", "No Move"]
            all_tlr_types = sorted(
                set(merged[tlr_col_a].dropna().astype(str).unique()) | set(merged[tlr_col_b].dropna().astype(str).unique())
            )

            with st.expander("Filters & sort", expanded=False):
                f1, f2, f3 = st.columns(3)
                with f1:
                    sel_scenarios = st.multiselect(
                        "Scenario(s)",
                        options=all_scenarios,
                        default=[],
                        key="tlr_filter_scenario",
                        help="Leave empty to show all scenarios. Select one or more to focus on specific scenes.",
                    )
                with f2:
                    sel_status = st.multiselect(
                        "Vehicle status (A or B)",
                        options=all_statuses,
                        default=[],
                        key="tlr_filter_status",
                        help="Show only rows where status in A or B is one of these. Empty = all.",
                    )
                with f3:
                    sel_tlr_a = st.multiselect(
                        f"Traffic light type in {label_a}",
                        options=all_tlr_types,
                        default=[],
                        key="tlr_filter_tlr_a",
                        help="Filter by type in baseline. Empty = any.",
                    )
                s1, s2 = st.columns(2)
                with s1:
                    sel_tlr_b = st.multiselect(
                        f"Traffic light type in {label_b}",
                        options=all_tlr_types,
                        default=[],
                        key="tlr_filter_tlr_b",
                        help="Filter by type in compare. Empty = any.",
                    )
                with s2:
                    sort_by = st.selectbox(
                        "Sort by",
                        [
                            "Scenario, then frame index",
                            "Frame index only",
                            "Difference first (then scenario, frame)",
                            f"Traffic light type ({label_a})",
                            f"Traffic light type ({label_b})",
                        ],
                        key="tlr_sort_by",
                    )

            # Apply filters to merged and diff_tlr
            filtered_merged = merged.copy()
            if sel_scenarios:
                filtered_merged = filtered_merged[filtered_merged["scenario"].isin(sel_scenarios)]
            if sel_status:
                filtered_merged = filtered_merged[
                    filtered_merged["status_a"].isin(sel_status) | filtered_merged["status_b"].isin(sel_status)
                ]
            if sel_tlr_a:
                filtered_merged = filtered_merged[filtered_merged[tlr_col_a].astype(str).isin(sel_tlr_a)]
            if sel_tlr_b:
                filtered_merged = filtered_merged[filtered_merged[tlr_col_b].astype(str).isin(sel_tlr_b)]
            filtered_diff = filtered_merged[filtered_merged["_diff"]]

            # Sort
            if sort_by == "Scenario, then frame index":
                filtered_merged = filtered_merged.sort_values(["scenario", "frame_index"]).reset_index(drop=True)
            elif sort_by == "Frame index only":
                filtered_merged = filtered_merged.sort_values("frame_index").reset_index(drop=True)
            elif sort_by == "Difference first (then scenario, frame)":
                filtered_merged = filtered_merged.sort_values(
                    ["_diff", "scenario", "frame_index"], ascending=[False, True, True]
                ).reset_index(drop=True)
            elif sort_by == f"Traffic light type ({label_a})":
                filtered_merged = filtered_merged.sort_values([tlr_col_a, "scenario", "frame_index"]).reset_index(drop=True)
            else:
                filtered_merged = filtered_merged.sort_values([tlr_col_b, "scenario", "frame_index"]).reset_index(drop=True)

            # Use filtered data for display
            to_show_merged = filtered_merged
            to_show_diff = filtered_diff

            if show_only_diff:
                if not to_show_diff.empty:
                    st.markdown("**Frames where traffic light type differs (A vs B)**")
                    display_df = to_show_diff[[
                        "scenario", dataset_col_a, dataset_col_b, "frame_index",
                        tlr_col_a, tlr_col_b,
                        "status_a", "status_b",
                    ]].copy()
                    display_df = display_df.rename(columns={"status_a": f"status ({label_a})", "status_b": f"status ({label_b})"})
                    def _highlight_diff_columns(series):
                        if series.name in (tlr_col_a, tlr_col_b):
                            return ["background-color: #ffe6e6"] * len(series)
                        return [""] * len(series)
                    styled = display_df.style.apply(_highlight_diff_columns, axis=0)
                    st.dataframe(styled, width='stretch', hide_index=True)
                    caption = f"Showing **{len(to_show_diff)}** frame(s) with different traffic light type (of {len(diff_tlr)} total before filters)."
                    if sel_scenarios or sel_status or sel_tlr_a or sel_tlr_b:
                        caption += " Filters applied."
                    st.caption(caption)
                    # Download CSV
                    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
                    json_bytes = _dataframe_to_json_bytes(display_df, export_kind="compare_diff_frames")
                    dl_col_csv, dl_col_json = st.columns(2)
                    with dl_col_csv:
                        st.download_button("Download as CSV", data=csv_bytes, file_name="tlr_diff_frames.csv", mime="text/csv", key="tlr_dl_diff")
                    with dl_col_json:
                        st.download_button("Download as JSON", data=json_bytes, file_name="tlr_diff_frames.json", mime="application/json", key="tlr_dl_diff_json")
                else:
                    st.info(
                        f"No frames with different traffic light type between {label_a} and {label_b}"
                        + (" for the selected filters." if (sel_scenarios or sel_status or sel_tlr_a or sel_tlr_b) else ".")
                    )
            else:
                st.markdown("**All frames (A vs B)** — rows where traffic light type differs are highlighted.")
                display_df = to_show_merged[[
                    "scenario", dataset_col_a, dataset_col_b, "frame_index",
                    tlr_col_a, tlr_col_b,
                    "status_a", "status_b",
                ]].copy()
                display_df = display_df.rename(columns={"status_a": f"status ({label_a})", "status_b": f"status ({label_b})"})
                def _highlight_diff_rows(df):
                    diff_mask = to_show_merged["_diff"].values
                    data = [
                        ["background-color: #ffe6e6" if diff_mask[i] and col in (tlr_col_a, tlr_col_b) else "" for col in df.columns]
                        for i in range(len(df))
                    ]
                    return pd.DataFrame(data, index=df.index, columns=df.columns)
                styled = display_df.style.apply(_highlight_diff_rows, axis=None)
                st.dataframe(styled, width='stretch', hide_index=True)
                num_diff = to_show_merged["_diff"].sum()
                caption = f"Showing **{len(to_show_merged)}** frame(s) ({int(num_diff)} with different type). Total before filters: {len(merged)}."
                if sel_scenarios or sel_status or sel_tlr_a or sel_tlr_b:
                    caption += " Filters applied."
                st.caption(caption)
                csv_bytes = display_df.to_csv(index=False).encode("utf-8")
                json_bytes = _dataframe_to_json_bytes(display_df, export_kind="compare_all_frames")
                dl_col_csv, dl_col_json = st.columns(2)
                with dl_col_csv:
                    st.download_button("Download as CSV", data=csv_bytes, file_name="tlr_compare_all_frames.csv", mime="text/csv", key="tlr_dl_all")
                with dl_col_json:
                    st.download_button("Download as JSON", data=json_bytes, file_name="tlr_compare_all_frames.json", mime="application/json", key="tlr_dl_all_json")
        else:
            st.caption("Need details from both A and B to show traffic light type differences.")
        st.markdown("---")
        st.markdown("**Per-dataset details** (single run)")
        view_which = st.radio("Show details for", [label_a, label_b], horizontal=True, key="tlr_details_which")
        analyzer = analyzer_b if view_which == label_b else analyzer_a
        details_df = analyzer.get_vehicle_status_details_df()
        if details_df is not None and not details_df.empty:
            # Filter by scenario for per-dataset view too
            single_scenarios = sorted(details_df["scenario"].unique().tolist())
            with st.expander("Filter by scenario", expanded=False):
                single_sel = st.multiselect(
                    "Scenario(s)",
                    options=single_scenarios,
                    default=[],
                    key="tlr_single_filter_scenario",
                    help="Leave empty for all scenarios.",
                )
            if single_sel:
                details_df = details_df[details_df["scenario"].isin(single_sel)]
            st.dataframe(details_df, width='stretch', hide_index=True)
            if not details_df.empty:
                csv_name = f"tlr_details_{view_which.replace(' ', '_')}.csv"
                json_name = f"tlr_details_{view_which.replace(' ', '_')}.json"
                dl_col_csv, dl_col_json = st.columns(2)
                with dl_col_csv:
                    st.download_button(
                        "Download as CSV",
                        data=details_df.to_csv(index=False).encode("utf-8"),
                        file_name=csv_name,
                        mime="text/csv",
                        key="tlr_dl_single",
                    )
                with dl_col_json:
                    st.download_button(
                        "Download as JSON",
                        data=_dataframe_to_json_bytes(details_df, export_kind="per_dataset_details"),
                        file_name=json_name,
                        mime="application/json",
                        key="tlr_dl_single_json",
                    )
        else:
            st.info("No vehicle status details available.")

    with tab_tlr_viewer:
        _render_tlr_viewer_tab(
            {
                label_a: analyzer_a.get_vehicle_status_details_df(),
                label_b: analyzer_b.get_vehicle_status_details_df(),
            },
            key_prefix="tlr_compare_viewer",
        )


# ----- Sidebar: mode and TLR directory selection -----
st.sidebar.markdown("##### TLR data")
st.sidebar.caption(f"Root: `{path_display(data_root)}` — directories with **result.json** are listed below.")

tlr_candidates = list_tlr_result_directories()
# Build stable keys for URL (path relative to data root)
tlr_keys = [path_to_tlr_key(p) for p, _ in tlr_candidates] if tlr_candidates else []

# URL override for mode
saved_mode = "Single"
if url_mode == "compare":
    saved_mode = "Compare"
elif url_mode == "single":
    saved_mode = "Single"
mode_index = 0 if saved_mode == "Single" else 1
mode = st.sidebar.radio("Mode", ["Single", "Compare"], index=mode_index, horizontal=True, key="tlr_mode")

resolved_path_a = None
resolved_path_b = None
sel_a = 0
sel_b = 0

if tlr_candidates:
    options = list(range(len(tlr_candidates)))
    labels = [format_tlr_option(tlr_candidates[i]) for i in options]

    # Initial index for A from URL (if valid)
    run_a_index = tlr_keys.index(url_path_a) if url_path_a in tlr_keys else 0

    if mode == "Single":
        sel_a = st.sidebar.selectbox(
            "Choose TLR result directory (with result.json)",
            options=options,
            index=run_a_index,
            format_func=lambda i: labels[i],
            key="tlr_select_a",
        )
        resolved_path_a = str(tlr_candidates[sel_a][0])
        st.sidebar.success(f"Selected: {labels[sel_a]}")
    else:
        sel_a = st.sidebar.selectbox(
            "Baseline (A)",
            options=options,
            index=run_a_index,
            format_func=lambda i: labels[i],
            key="tlr_select_a",
        )
        resolved_path_a = str(tlr_candidates[sel_a][0])
        other_options = [i for i in options if i != sel_a]
        if not other_options:
            st.sidebar.warning("Add another TLR result directory under the data root to compare.")
        else:
            run_b_index_in_other = 0
            if url_path_b in tlr_keys and url_path_b != tlr_keys[sel_a]:
                try:
                    run_b_index_in_other = other_options.index(tlr_keys.index(url_path_b))
                except ValueError:
                    pass
            sel_b = st.sidebar.selectbox(
                "Compare (B)",
                options=other_options,
                index=min(run_b_index_in_other, len(other_options) - 1),
                format_func=lambda i: labels[i],
                key="tlr_select_b",
            )
            resolved_path_b = str(tlr_candidates[sel_b][0])
            st.sidebar.success(f"A: {labels[sel_a]}  →  B: {labels[sel_b]}")

    # Sync URL with current selection (shareable link)
    query = {"mode": "single" if mode == "Single" else "compare", "path_a": tlr_keys[sel_a]}
    if mode == "Compare" and resolved_path_b:
        query["path_b"] = tlr_keys[sel_b]
    st.query_params.update(query)
else:
    st.sidebar.warning(
        "No TLR result directories found. Under the data root we look for directories that contain "
        "**result.json** (direct subfolders or suite folders whose subfolders have result.json)."
    )

# ----- Load analyzer(s) -----
analyzer_a = get_or_load_analyzer(resolved_path_a) if resolved_path_a else None
analyzer_b = get_or_load_analyzer(resolved_path_b) if (mode == "Compare" and resolved_path_b) else None

if analyzer_a is None:
    st.info(
        "No TLR result directory selected. In the sidebar, choose a directory that contains **result.json**. "
        "Candidates are discovered automatically under the data root."
    )
    st.stop()

_hero_mode = "Compare Mode" if mode == "Compare" else "Single Run"
if mode == "Compare" and resolved_path_b:
    render_loaded_data_section(
        [
            ("Baseline · A", path_display(Path(resolved_path_a))),
            ("Candidate · B", path_display(Path(resolved_path_b))),
        ]
    )
else:
    render_loaded_data_section([("TLR result", path_display(Path(resolved_path_a)))])
render_page_hero(
    kicker="Traffic light recognition",
    title="TLR evaluation analysis",
    description=(
        "Criteria matrices, vehicle status vs. traffic-light type, critical and priority zones — "
        "explore one dataset or compare A vs B with shareable URLs."
    ),
    mode=_hero_mode,
)

# ----- Labels for compare mode -----
label_a = Path(resolved_path_a).name if resolved_path_a else "A"
label_b = Path(resolved_path_b).name if resolved_path_b else "B"

# ========== SINGLE MODE ==========
if mode == "Single":
    stats = analyzer_a.get_summary_stats()
    section_header("Results overview", "Aggregate counts and TP rate for the selected TLR directory.")
    share_q = f"mode=single&path_a={quote(tlr_keys[sel_a], safe='/')}"
    render_share_link_callout(share_q, caption="Append to the TLR Analysis page URL on your server.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scenarios", stats["num_scenarios"])
    c2.metric("Total frames", f"{stats['total_frames']:,}")
    c3.metric("Total TP", f"{stats['total_tp']:,}")
    c4.metric("Overall TP rate", f"{stats['overall_tp_rate']:.2%}")
    c5.metric("Scenarios w/ criteria", stats["num_scenarios_with_criteria"])
    if stats.get("best_criteria") is not None:
        st.caption(
            f"Best criteria: **{stats['best_criteria']}** (TP rate {stats['best_tp_rate']:.2%}) — "
            f"Worst: **{stats['worst_criteria']}** (TP rate {stats['worst_tp_rate']:.2%})"
        )

    tab_criteria, tab_vehicle, tab_critical, tab_details, tab_tlr_viewer = st.tabs([
        "Criteria matrix", "Vehicle status vs TLR type", "Critical & priority zones", "Vehicle status details", "TLR viewer",
    ])
    _render_single_tabs(analyzer_a, tab_criteria, tab_vehicle, tab_critical, tab_details, tab_tlr_viewer)
    st.stop()

# ========== COMPARE MODE ==========
if analyzer_b is None:
    st.info("Select a second TLR result directory (Compare B) in the sidebar to compare.")
    st.stop()

stats_a = analyzer_a.get_summary_stats()
stats_b = analyzer_b.get_summary_stats()

section_header("Compare overview", "Side-by-side stats for baseline A and candidate B.")
share_q_compare = f"mode=compare&path_a={quote(tlr_keys[sel_a], safe='/')}&path_b={quote(tlr_keys[sel_b], safe='/')}"
render_share_link_callout(share_q_compare, caption="Append to the TLR Analysis page URL on your server.")
col_a, col_delta, col_b = st.columns(3)
with col_a:
    st.markdown(f"**{label_a} (Baseline)**")
    st.metric("Scenarios", stats_a["num_scenarios"])
    st.metric("Total frames", f"{stats_a['total_frames']:,}")
    st.metric("Total TP", f"{stats_a['total_tp']:,}")
    st.metric("Overall TP rate", f"{stats_a['overall_tp_rate']:.2%}")
with col_delta:
    st.markdown("**Δ (B − A)**")
    st.metric("Scenarios", stats_b["num_scenarios"] - stats_a["num_scenarios"])
    st.metric("Total frames", f"{stats_b['total_frames'] - stats_a['total_frames']:+,}")
    st.metric("Total TP", f"{stats_b['total_tp'] - stats_a['total_tp']:+,}")
    delta_rate = stats_b["overall_tp_rate"] - stats_a["overall_tp_rate"]
    st.metric("Overall TP rate", f"{delta_rate:+.2%}")
with col_b:
    st.markdown(f"**{label_b} (Compare)**")
    st.metric("Scenarios", stats_b["num_scenarios"])
    st.metric("Total frames", f"{stats_b['total_frames']:,}")
    st.metric("Total TP", f"{stats_b['total_tp']:,}")
    st.metric("Overall TP rate", f"{stats_b['overall_tp_rate']:.2%}")

tab_criteria, tab_vehicle, tab_critical, tab_details, tab_tlr_viewer = st.tabs([
    "Criteria matrix", "Vehicle status vs TLR type", "Critical & priority zones", "Vehicle status details", "TLR viewer",
])
_render_compare_tabs(analyzer_a, analyzer_b, label_a, label_b, tab_criteria, tab_vehicle, tab_critical, tab_details, tab_tlr_viewer)
