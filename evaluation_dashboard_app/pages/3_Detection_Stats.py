import html
from contextlib import contextmanager
import hashlib

import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from lib.path_utils import get_run_display_name, path_display
from lib.detection_stats_debug import (
    ds_debug_init_session_state,
    ds_debug_log_exception,
    ds_debug_log_memory,
    ds_debug_render_expander,
    ds_dlog,
    ds_dtimer,
)
from lib.overview_url_hydrate import try_hydrate_session_from_overview_query_params
from lib.parquet_schema import get_parquet_columns, missing_detection_stats_columns, schema_flags
from lib.page_chrome import inject_app_page_styles, render_loaded_data_section, render_page_hero
from lib.t4_dataset_embed import t4_dashboard_url
from lib.ui.detection_stats import (
    detection_stats_page_loading_banner_markup,
    ds_spot_loading,
    ds_spot_loading_markup,
    inject_detection_stats_kpi_styles,
    inject_detection_stats_styles,
    render_kpi_card,
    section_header_html,
)

# Perception diff: unified improved/degraded palette (Hierarchical view + Comparison lens)
IMPROVED_COLOR = "#1a9850"
DEGRADED_COLOR = "#d73027"
IMPROVED_SCALE = [[0.0, "#f7fcf5"], [1.0, IMPROVED_COLOR]]
DEGRADED_SCALE = [[0.0, "#fff5f0"], [1.0, DEGRADED_COLOR]]
# Run-series colors (Panels 2–4, 6–8) — consistent across page
RUN_COLORS = ["#4A90D9", "#E86A33", "#2d8f47", "#9B59B6", "#1ABC9C", "#95a5a6"]
# Status distribution: semantic colors (TP=green, FN=red, FP=orange)
STATUS_COLORS = {
    "TP": "#2d8f47",
    "FN": "#d73027",
    "FP": "#E86A33",
    "TN": "#4A90D9",
}
DETECTION_STATS_SKIP_INITIAL_FRAMES = 3
DETECTION_STATS_INITIAL_FRAME_FILTER = (
    f"(frame_index IS NULL OR TRY_CAST(frame_index AS BIGINT) >= {DETECTION_STATS_SKIP_INITIAL_FRAMES})"
)

# Unified Plotly layout theme for all charts
PLOTLY_LAYOUT_THEME = dict(
    font=dict(family='"Inter", "Segoe UI", sans-serif', size=11),
    title=dict(font=dict(size=14, color="#1f2937")),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,0.6)",
    margin=dict(t=48, b=40, l=52, r=24),
    height=380,
    xaxis=dict(
        tickfont=dict(size=11),
        title_font=dict(size=12),
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.15)",
    ),
    yaxis=dict(
        tickfont=dict(size=11),
        title_font=dict(size=12),
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.15)",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11),
    ),
    showlegend=True,
)


def _banner_html_with_note(note: str) -> str:
    base = detection_stats_page_loading_banner_markup()
    if not note:
        return base
    return base.replace(
        '<span class="ds-plb-sub">Hang tight — large Parquet files can take a moment.</span>',
        f'<span class="ds-plb-sub">Hang tight — large Parquet files can take a moment.<br>{html.escape(note)}</span>',
    )


def apply_chart_theme(fig, **overrides):
    """Apply unified theme to a Plotly figure; overrides (e.g. height, margin) take precedence."""
    layout_update = {**PLOTLY_LAYOUT_THEME, **overrides}
    fig.update_layout(**layout_update)
    return fig


def _tpr_lollipop_single(df: pd.DataFrame, title: str) -> go.Figure:
    """Horizontal lollipop: rank labels by TPR (highest at top)."""
    d = df.sort_values("tpr", ascending=True).copy()
    fig = go.Figure()
    for _, row in d.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[0, row["tpr"]],
                y=[row["label"], row["label"]],
                mode="lines",
                line=dict(color="rgba(74, 144, 217, 0.45)", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=d["tpr"],
            y=d["label"],
            mode="markers",
            name="TP rate",
            marker=dict(size=14, color=RUN_COLORS[0], line=dict(width=1, color="white")),
            hovertemplate="%{y}<br>TP rate: %{x:.2%}<extra></extra>",
        )
    )
    apply_chart_theme(fig, height=max(320, 40 + 28 * len(d)))
    fig.update_layout(
        title=title,
        xaxis_title="TP rate",
        yaxis_title="",
        xaxis_range=[0, 1.15],
        showlegend=False,
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(0,0,0,0.2)")
    fig.add_vline(x=1.0, line_dash="dot", line_color="rgba(0,0,0,0.12)")
    return fig


def _tpr_spider_compare(
    df_all: pd.DataFrame,
    categories: List[str],
    title: str,
    run_order: List[str],
    *,
    height: int = 440,
) -> go.Figure:
    """Closed polar lines: one trace per run (order matches run_order for colors)."""
    fig = go.Figure()
    for i, run_lbl in enumerate(run_order):
        sub = df_all[df_all["run"] == run_lbl].drop_duplicates("label").set_index("label")
        r_vals = [float(sub.loc[c, "tpr"]) if c in sub.index else 0.0 for c in categories]
        r_closed = r_vals + r_vals[:1]
        theta = categories + categories[:1]
        c = RUN_COLORS[i % len(RUN_COLORS)]
        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta,
                name=str(run_lbl),
                line=dict(color=c, width=2),
                fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)",
                fill="toself",
                hovertemplate="%{theta}<br>TP rate: %{r:.2%}<extra></extra>",
            )
        )
    apply_chart_theme(fig, height=height)
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%", gridcolor="rgba(0,0,0,0.08)"),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    )
    return fig


def _count_spider_compare(
    df_all: pd.DataFrame,
    categories: List[str],
    title: str,
    run_order: List[str],
    hover_metric: str,
) -> go.Figure:
    """Polar chart: one closed polygon per run; r = count per label (same info as stacked bars)."""
    fig = go.Figure()
    max_r = 0.0
    traces_r: List[List[float]] = []
    for run_lbl in run_order:
        sub = df_all[df_all["run"] == run_lbl].drop_duplicates("label").set_index("label")
        r_vals = [float(sub.loc[c, "count"]) if c in sub.index else 0.0 for c in categories]
        traces_r.append(r_vals)
        if r_vals:
            max_r = max(max_r, max(r_vals))
    r_max = max(max_r * 1.08, 1.0)

    for i, run_lbl in enumerate(run_order):
        r_vals = traces_r[i]
        r_closed = r_vals + r_vals[:1]
        theta = categories + categories[:1]
        c = RUN_COLORS[i % len(RUN_COLORS)]
        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta,
                name=str(run_lbl),
                line=dict(color=c, width=2),
                fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)",
                fill="toself",
                hovertemplate=f"%{{theta}}<br>{hover_metric}: %{{r:,.0f}}<extra></extra>",
            )
        )
    apply_chart_theme(fig, height=380)
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, r_max],
                tickformat=",.0f",
                gridcolor="rgba(0,0,0,0.08)",
            ),
            angularaxis=dict(tickfont=dict(size=9)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
    )
    return fig


def _scalar_metric_spider_compare(
    df_all: pd.DataFrame,
    categories: List[str],
    title: str,
    run_order: List[str],
    value_col: str,
    hover_metric: str,
    *,
    height: int = 380,
    tickformat: str = ",.3f",
) -> go.Figure:
    """Polar chart: one polygon per run; r = numeric metric per label (e.g. mean |error|)."""
    fig = go.Figure()
    max_r = 0.0
    traces_r: List[List[float]] = []
    for run_lbl in run_order:
        sub = df_all[df_all["run"] == run_lbl].drop_duplicates("label").set_index("label")
        r_vals = []
        for c in categories:
            if c in sub.index:
                v = sub.loc[c, value_col]
                r_vals.append(0.0 if pd.isna(v) else float(v))
            else:
                r_vals.append(0.0)
        traces_r.append(r_vals)
        if r_vals:
            max_r = max(max_r, max(r_vals))
    r_max = max(max_r * 1.08, 1e-6)

    for i, run_lbl in enumerate(run_order):
        r_vals = traces_r[i]
        r_closed = r_vals + r_vals[:1]
        theta = categories + categories[:1]
        c = RUN_COLORS[i % len(RUN_COLORS)]
        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta,
                name=str(run_lbl),
                line=dict(color=c, width=2),
                fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)",
                fill="toself",
                hovertemplate="%{theta}<br>"
                + hover_metric
                + ": %{r:.4f}<extra></extra>",
            )
        )

    apply_chart_theme(fig, height=height)
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, r_max],
                tickformat=tickformat,
                gridcolor="rgba(0,0,0,0.08)",
            ),
            angularaxis=dict(tickfont=dict(size=9)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
    )
    return fig


st.set_page_config(
    layout="wide",
    page_title="Object Detection",
    page_icon="🎯",
    initial_sidebar_state="expanded",
)

try_hydrate_session_from_overview_query_params()
ds_debug_init_session_state(st.session_state)

# =============================
# Session state from Overview (mode, run paths)
# =============================
if "runA" not in st.session_state:
    st.warning(
        "Please load data from the **Overview** page first (select mode and run(s)). "
        "If you already did, open **Overview** once so the URL includes `run_a=...` (share link), then return — "
        "or hard-refresh. With multiple Streamlit replicas, the server-side session may not follow until the URL is synced."
    )
    st.stop()

inject_app_page_styles()

mode = st.session_state.get("mode", "Single Mode")
runA = st.session_state["runA"]
# Multi-run compare: use all_runs and run_labels when available (Overview sets these in Compare Mode)
all_runs = st.session_state.get("all_runs")
run_labels = st.session_state.get("run_labels")
if mode == "Compare Mode" and all_runs and run_labels and len(all_runs) >= 2:
    runs = all_runs
    run_labels_list = run_labels
else:
    runs = [runA]
    run_labels_list = ["A"]
    if mode == "Compare Mode":
        runB = st.session_state.get("runB")
        if runB is not None:
            runs = [runA, runB]
            run_labels_list = ["A", "B"]
single_mode = len(runs) == 1


def _run_share_names_for_links() -> List[str]:
    names: List[str] = []
    for run in runs:
        try:
            names.append(get_run_display_name(Path(run["path"])))
        except Exception:
            names.append(str(run.get("path") or ""))
    return names


def _with_t4_viewer_links(df: pd.DataFrame, run_share_names: List[str]) -> pd.DataFrame:
    """Add a compact dashboard 3D viewer deep-link column for dataset/frame rows."""
    if df is None or df.empty:
        return df
    has_dataset = "t4dataset_name" in df.columns or "t4dataset_id" in df.columns
    if not has_dataset and "scenario_name" not in df.columns:
        return df
    out = df.copy()

    def _row_url(row: pd.Series) -> str:
        return t4_dashboard_url(
            mode=mode,
            run_names=run_share_names,
            suite_name=row.get("suite_name"),
            scenario_name=row.get("scenario_name"),
            t4dataset_name=row.get("t4dataset_name"),
            t4dataset_id=row.get("t4dataset_id"),
            frame_index=row.get("frame_index") if "frame_index" in row.index else None,
            compare_view_mode="side_by_side",
        )

    out.insert(0, "open_3d", out.apply(_row_url, axis=1))
    return out


def _t4_viewer_link_column_config() -> Dict[str, Any]:
    return {
        "open_3d": st.column_config.LinkColumn(
            "3D viewer",
            display_text="Open 3D",
            help="Open this scene/frame in the dashboard T4 3D Viewer.",
            width="small",
        )
    }


def _compare_availability_mask(df: pd.DataFrame) -> pd.Series:
    """Rows where both compared runs have GT objects for the same compare key."""
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    if "base_gt_cnt" not in df.columns or "candidate_gt_cnt" not in df.columns:
        return pd.Series(True, index=df.index)
    return (pd.to_numeric(df["base_gt_cnt"], errors="coerce").fillna(0) > 0) & (
        pd.to_numeric(df["candidate_gt_cnt"], errors="coerce").fillna(0) > 0
    )


def _compare_availability_summary(df: pd.DataFrame, *, unit: str) -> str:
    if df is None or df.empty or "base_gt_cnt" not in df.columns or "candidate_gt_cnt" not in df.columns:
        return ""
    base_cnt = pd.to_numeric(df["base_gt_cnt"], errors="coerce").fillna(0)
    cand_cnt = pd.to_numeric(df["candidate_gt_cnt"], errors="coerce").fillna(0)
    missing_base = int(((base_cnt <= 0) & (cand_cnt > 0)).sum())
    missing_candidate = int(((base_cnt > 0) & (cand_cnt <= 0)).sum())
    both_avail = int(((base_cnt > 0) & (cand_cnt > 0)).sum())
    total = missing_base + missing_candidate + both_avail
    if missing_base == 0 and missing_candidate == 0:
        return ""
    # If >80% of rows are one-sided, likely a filter mismatch (e.g. different topic_name per run)
    if total > 0 and (missing_base + missing_candidate) > 0.8 * total:
        parts = []
        if missing_base:
            parts.append(f"{missing_base} {unit} only in candidate")
        if missing_candidate:
            parts.append(f"{missing_candidate} {unit} only in baseline A")
        parts.append("⚠️ heavy one-sided data — check Topic Name filter (runs may have different topics)")
        return ", ".join(parts)
    parts = []
    if missing_base:
        parts.append(f"{missing_base} {unit} only in candidate")
    if missing_candidate:
        parts.append(f"{missing_candidate} {unit} only in baseline A")
    return ", ".join(parts)


def _compare_availability_reason(df: pd.DataFrame) -> pd.Series:
    """Human-readable reason for one-sided compare rows."""
    if df is None or df.empty:
        return pd.Series(dtype="object")
    if "base_gt_cnt" not in df.columns or "candidate_gt_cnt" not in df.columns:
        return pd.Series("available in both", index=df.index, dtype="object")
    base_cnt = pd.to_numeric(df["base_gt_cnt"], errors="coerce").fillna(0)
    cand_cnt = pd.to_numeric(df["candidate_gt_cnt"], errors="coerce").fillna(0)
    return pd.Series(
        np.select(
            [
                (base_cnt <= 0) & (cand_cnt > 0),
                (base_cnt > 0) & (cand_cnt <= 0),
            ],
            [
                "Only in candidate",
                "Only in baseline A",
            ],
            default="Available in both",
        ),
        index=df.index,
        dtype="object",
    )


# Internal columns kept for availability filtering but hidden from display tables
_DIFF_INTERNAL_COLS = ["missing_in_base_cnt", "missing_in_candidate_cnt"]
_FP_INTERNAL_COLS = ["total_est", "base_est_cnt", "candidate_est_cnt", "missing_in_base_cnt", "missing_in_candidate_cnt"]

# --- FP-side availability helpers (use base_est_cnt / candidate_est_cnt) ---


def _compare_availability_mask_fp(df: pd.DataFrame) -> pd.Series:
    """Rows where both compared runs have EST objects for the same compare key."""
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    if "base_est_cnt" not in df.columns or "candidate_est_cnt" not in df.columns:
        return pd.Series(True, index=df.index)
    return (pd.to_numeric(df["base_est_cnt"], errors="coerce").fillna(0) > 0) & (
        pd.to_numeric(df["candidate_est_cnt"], errors="coerce").fillna(0) > 0
    )


def _compare_availability_summary_fp(df: pd.DataFrame, *, unit: str) -> str:
    if df is None or df.empty or "base_est_cnt" not in df.columns or "candidate_est_cnt" not in df.columns:
        return ""
    base_cnt = pd.to_numeric(df["base_est_cnt"], errors="coerce").fillna(0)
    cand_cnt = pd.to_numeric(df["candidate_est_cnt"], errors="coerce").fillna(0)
    missing_base = int(((base_cnt <= 0) & (cand_cnt > 0)).sum())
    missing_candidate = int(((base_cnt > 0) & (cand_cnt <= 0)).sum())
    if missing_base == 0 and missing_candidate == 0:
        return ""
    parts = []
    if missing_base:
        parts.append(f"{missing_base} {unit} only in candidate")
    if missing_candidate:
        parts.append(f"{missing_candidate} {unit} only in baseline A")
    return ", ".join(parts)


def _compare_availability_reason_fp(df: pd.DataFrame) -> pd.Series:
    """Human-readable reason for one-sided FP compare rows."""
    if df is None or df.empty:
        return pd.Series(dtype="object")
    if "base_est_cnt" not in df.columns or "candidate_est_cnt" not in df.columns:
        return pd.Series("available in both", index=df.index, dtype="object")
    base_cnt = pd.to_numeric(df["base_est_cnt"], errors="coerce").fillna(0)
    cand_cnt = pd.to_numeric(df["candidate_est_cnt"], errors="coerce").fillna(0)
    return pd.Series(
        np.select(
            [
                (base_cnt <= 0) & (cand_cnt > 0),
                (base_cnt > 0) & (cand_cnt <= 0),
            ],
            [
                "Only in candidate",
                "Only in baseline A",
            ],
            default="Available in both",
        ),
        index=df.index,
        dtype="object",
    )


def _dataset_name_debug_summary(con, base_view: str, base_filter: str, cand_view: str, cand_filter: str, source: str = "GT") -> str:
    """Return a markdown summary of unique t4dataset_name/t4dataset_id from both sides for debugging."""
    lines: List[str] = []
    id_col = "uuid" if source == "GT" else "pair_uuid"
    try:
        for label, view, fc in [("Baseline A", base_view, base_filter), ("Candidate", cand_view, cand_filter)]:
            q = f"""
                SELECT DISTINCT t4dataset_id, COALESCE(try_cast(t4dataset_name AS VARCHAR), '') AS t4dataset_name
                FROM {view}
                WHERE source = '{source}' AND {id_col} IS NOT NULL AND frame_index IS NOT NULL
                    AND {fc}
                ORDER BY t4dataset_id
            """
            df_names = con.execute(q).df()
            lines.append(f"**{label}** ({len(df_names)} unique t4dataset_id):")
            for _, row in df_names.iterrows():
                did = str(row["t4dataset_id"])
                dname = str(row["t4dataset_name"]).strip()
                lines.append(f"- `t4dataset_id={did}`  name=`{dname}`")
            lines.append("")
    except Exception as e:
        lines.append(f"⚠️ dataset name debug query failed: {e}")
    return "\n".join(lines)


def _fp_pair_uuid_debug(con, base_view: str, base_filter: str, cand_view: str, cand_filter: str, limit: int = 5) -> str:
    """Return a markdown summary comparing pair_uuid samples from overlapping datasets
    to diagnose why FP diff JOIN fails to match."""
    lines: List[str] = []
    try:
        # Find overlapping t4dataset_ids
        overlap_q = f"""
            SELECT DISTINCT CAST(b.t4dataset_id AS VARCHAR) AS t4dataset_id
            FROM (SELECT DISTINCT t4dataset_id FROM {base_view}
                  WHERE source = 'EST' AND pair_uuid IS NOT NULL AND frame_index IS NOT NULL
                    AND {base_filter}) b
            INNER JOIN (SELECT DISTINCT t4dataset_id FROM {cand_view}
                        WHERE source = 'EST' AND pair_uuid IS NOT NULL AND frame_index IS NOT NULL
                          AND {cand_filter}) c
                ON CAST(b.t4dataset_id AS VARCHAR) = CAST(c.t4dataset_id AS VARCHAR)
            LIMIT {limit}
        """
        overlap_df = con.execute(overlap_q).df()
        if overlap_df.empty:
            lines.append("⚠️ No overlapping t4dataset_id found between baseline and candidate (EST side).")
            return "\n".join(lines)

        lines.append(f"**Sampling {len(overlap_df)} overlapping datasets to compare pair_uuid values:**")
        lines.append("")

        for _, row in overlap_df.iterrows():
            did = str(row["t4dataset_id"])
            lines.append(f"---")
            lines.append(f"**t4dataset_id=`{did}`**")
            for label, view, fc in [("Baseline A", base_view, base_filter), ("Candidate", cand_view, cand_filter)]:
                sample_q = f"""
                    SELECT frame_index, CAST(pair_uuid AS VARCHAR) AS pair_uuid, status,
                           CAST(label AS VARCHAR) AS label
                    FROM {view}
                    WHERE source = 'EST' AND pair_uuid IS NOT NULL AND frame_index IS NOT NULL
                      AND CAST(t4dataset_id AS VARCHAR) = '{did}'
                      AND {fc}
                    ORDER BY frame_index, pair_uuid
                    LIMIT 10
                """
                try:
                    sample_df = con.execute(sample_q).df()
                    lines.append(f"**{label}** ({len(sample_df)} sample rows):")
                    if sample_df.empty:
                        lines.append("  (no rows)")
                    else:
                        for _, sr in sample_df.iterrows():
                            fi = str(sr["frame_index"])
                            pu = str(sr["pair_uuid"])
                            st = str(sr["status"])
                            lb = str(sr["label"])
                            lines.append(f"  - frame=`{fi}` pair_uuid=`{pu}` status=`{st}` label=`{lb}`")
                except Exception as e:
                    lines.append(f"  ⚠️ query failed: {e}")
            lines.append("")

            # Also show a quick JOIN check
            join_check_q = f"""
                SELECT COUNT(*) AS match_cnt
                FROM (
                    SELECT DISTINCT frame_index, CAST(pair_uuid AS VARCHAR) AS pu
                    FROM {base_view}
                    WHERE source = 'EST' AND pair_uuid IS NOT NULL AND frame_index IS NOT NULL
                      AND CAST(t4dataset_id AS VARCHAR) = '{did}'
                      AND {base_filter}
                ) b
                INNER JOIN (
                    SELECT DISTINCT frame_index, CAST(pair_uuid AS VARCHAR) AS pu
                    FROM {cand_view}
                    WHERE source = 'EST' AND pair_uuid IS NOT NULL AND frame_index IS NOT NULL
                      AND CAST(t4dataset_id AS VARCHAR) = '{did}'
                      AND {cand_filter}
                ) c
                    ON b.frame_index = c.frame_index AND b.pu = c.pu
            """
            try:
                match_cnt = con.execute(join_check_q).fetchone()[0]
                lines.append(f"  🔗 Matching (frame_index, pair_uuid) tuples: **{match_cnt}**")
            except Exception as e:
                lines.append(f"  ⚠️ join check failed: {e}")

        lines.append("")
        lines.append("---")
        lines.append("If matching tuple count is 0 for overlapping datasets, pair_uuid values differ between runs.")
        lines.append("This means EST→GT matching is not deterministic across evaluation runs.")
    except Exception as e:
        lines.append(f"⚠️ pair_uuid debug query failed: {e}")
    return "\n".join(lines)


def list_parquets_in_run(run_path) -> List[str]:
    """Return sorted list of absolute paths to .parquet files in the run directory."""
    p = Path(run_path)
    if not p.is_dir():
        return []
    return sorted([str(f.resolve()) for f in p.glob("*.parquet")])


def default_parquet_index(paths: List[str]) -> int:
    """Prefer current.parquet for each run's file picker."""
    for idx, path in enumerate(paths):
        if os.path.basename(path) == "current.parquet":
            return idx
    return 0


def migrate_old_run_index_parquet_default(widget_key: str, paths: List[str], run_index: int) -> None:
    """Move old run-index defaults (Run B -> second file) to current.parquet once per session."""
    preferred_idx = default_parquet_index(paths)
    old_idx = min(run_index, len(paths) - 1)
    if preferred_idx == old_idx:
        return

    migration_key = f"{widget_key}__current_default_migrated"
    if st.session_state.get(migration_key):
        return

    old_default = paths[old_idx]
    if st.session_state.get(widget_key) in (None, old_default):
        st.session_state[widget_key] = paths[preferred_idx]
    st.session_state[migration_key] = True

# =============================
# DuckDB Connection (one in-memory DB per Streamlit browser session)
# =============================
def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection scoped to this Streamlit session."""
    if "_ds_duckdb" not in st.session_state:
        st.session_state["_ds_duckdb"] = duckdb.connect()
    return st.session_state["_ds_duckdb"]


def _parquet_selection_fingerprint(paths: List[str]) -> Tuple[Tuple[str, float], ...]:
    """Path + mtime per file so filter-only reruns skip rebuilding views when data is unchanged."""
    fp: List[Tuple[str, float]] = []
    for p in paths:
        try:
            fp.append((p, os.path.getmtime(p)))
        except OSError:
            fp.append((p, 0.0))
    return tuple(fp)

# =============================
# Helper Functions
# =============================
def validate_parquet_file(con, path: str) -> Tuple[bool, str]:
    """
    Try to read the parquet file. Returns (True, "") if ok,
    (False, error_message) if the file cannot be read (e.g. empty or invalid schema).
    """
    try:
        con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [path])
        return True, ""
    except Exception as e:
        err = str(e).strip()
        if "non-root column" in err or "Need at least one" in err:
            return False, (
                "This Parquet file has no readable columns (DuckDB: 'Need at least one non-root column'). "
                "The file may be empty, corrupt, or written with a schema DuckDB cannot use. "
                "Try re-generating the parquet from the Download page or check the source data."
            )
        return False, err


def validate_detection_stats_parquet(con, path: str) -> Tuple[bool, str]:
    """Validate that a readable parquet has object-level columns used by Detection Stats."""
    ok, msg = validate_parquet_file(con, path)
    if not ok:
        return ok, msg

    missing = missing_detection_stats_columns(con, path)
    if not missing:
        return True, ""

    columns = get_parquet_columns(con, path)
    return False, (
        "This parquet is readable, but it is not object-level detection data for Detection Stats. "
        f"Missing required columns: {', '.join(missing)}. "
        f"Detected columns: {', '.join(columns[:12])}{'...' if len(columns) > 12 else ''}. "
        "For release spec data, load/select the performance parquet such as performance/current.parquet "
        "or performance/future.parquet. The devops/usecase_devops.parquet file is a suite summary "
        "(Catalog Name, Suite Name, Success, Fail, Total, Pass Rate)."
    )

def list_values(con, pq: str, expr: str, where: Optional[str] = None) -> List:
    """Get distinct values from parquet file."""
    q = f"SELECT DISTINCT {expr} FROM parquet_scan('{pq}')"
    if where:
        q += f" WHERE {where}"
    q += " ORDER BY 1"
    df_ = con.execute(q).df()
    if df_.empty:
        return []
    return df_.iloc[:, 0].dropna().tolist()


def _is_detection_stats_eval_flat_cache(path: str) -> bool:
    p = Path(path)
    return p.suffix == ".parquet" and p.name.endswith("_eval_flat.parquet")


def create_view_eval_flat(con, target_file: str, view_name: str = "view_eval_flat"):
    """Create view_eval_flat with distance bins."""
    safe_target = target_file.replace("'", "''")
    if _is_detection_stats_eval_flat_cache(target_file):
        query = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM parquet_scan('{safe_target}')"
    else:
        query = f"CREATE OR REPLACE VIEW {view_name} AS {eval_flat_select_sql(target_file)}"
    con.execute(query)


def eval_flat_select_sql(target_file: str) -> str:
    safe_target = target_file.replace("'", "''")
    return f"""
    WITH src AS (
        SELECT * FROM parquet_scan('{safe_target}')
        UNION BY NAME
        SELECT CAST(NULL AS VARCHAR) AS visibility,
               CAST(NULL AS VARCHAR) AS suite_name,
               CAST(NULL AS VARCHAR) AS scenario_name,
               CAST(NULL AS VARCHAR) AS t4dataset_name
        WHERE FALSE
    ),
    base AS (
        SELECT
            * REPLACE (coalesce(CAST(visibility AS VARCHAR), 'not available') AS visibility),
            sqrt(CAST(x AS DOUBLE)*CAST(x AS DOUBLE) + CAST(y AS DOUBLE)*CAST(y AS DOUBLE)) AS dist_h
        FROM src
        WHERE x IS NOT NULL AND y IS NOT NULL
    ),
    bins AS (
        SELECT * FROM (
            VALUES
                (0.0,   10.0,   '[0,10)',     10),
                (10.0,  20.0,   '[10,20)',    20),
                (20.0,  30.0,   '[20,30)',    30),
                (30.0,  40.0,   '[30,40)',    40),
                (40.0,  50.0,   '[40,50)',    50),
                (50.0,  60.0,   '[50,60)',    60),
                (60.0,  70.0,   '[60,70)',    70),
                (70.0,  80.0,   '[70,80)',    80),
                (80.0,  90.0,   '[80,90)',    90),
                (90.0,  100.0,  '[90,100)',  100),
                (100.0, 110.0,  '[100,110)', 110),
                (110.0, 120.0,  '[110,120)', 120),
                (120.0, 130.0,  '[120,130)', 130),
                (130.0, 140.0,  '[130,140)', 140),
                (140.0, 150.0,  '[140,150)', 150),
                (150.0, 1e12,   '[150,inf)', 160)
        ) AS t(bin_start, bin_end, distance_bin, bin_idx)
    )
    SELECT
        bse.*,
        b.distance_bin,
        b.bin_idx,
        (status = 'TP') AS is_tp,
        (status = 'FP') AS is_fp,
        (status = 'FN') AS is_fn
    FROM base bse
    JOIN bins b
        ON bse.dist_h >= b.bin_start AND bse.dist_h < b.bin_end
    """


def _ds_cache_dir_for_run(run_path: Path) -> Path:
    return run_path / ".dashboard_cache" / "detection_stats_cache"


def _ds_cache_key_for_source(source_path: str) -> str:
    return hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]


def _ds_cache_path_for_source(run_path: Path, source_path: str) -> Path:
    src = Path(source_path)
    return _ds_cache_dir_for_run(run_path) / f"{src.stem}_{_ds_cache_key_for_source(source_path)}_eval_flat.parquet"


def _ensure_detection_stats_eval_flat_cache(
    con: duckdb.DuckDBPyConnection,
    *,
    run_path: Path,
    source_path: str,
) -> tuple[str, bool]:
    """
    Ensure a materialized eval_flat parquet exists for this source parquet.
    Returns (cached_parquet_path, rebuilt_flag).
    """
    cache_dir = _ds_cache_dir_for_run(run_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _ds_cache_path_for_source(run_path, source_path)
    source_stat = Path(source_path).stat()
    needs_rebuild = (
        not cache_path.exists()
        or cache_path.stat().st_mtime < source_stat.st_mtime
    )
    if needs_rebuild:
        safe_out = str(cache_path).replace("'", "''")
        con.execute(f"COPY ({eval_flat_select_sql(source_path)}) TO '{safe_out}' (FORMAT PARQUET)")
    return str(cache_path), needs_rebuild

# Per-(dataset, topic, label, bin, visibility, suite) aggregates — shared by distance-bin rate queries.
_TPR_FPR_STATS_SELECT = """SELECT
            t4dataset_id,
            topic_name,
            label,
            distance_bin,
            bin_idx,
            coalesce(try(CAST(visibility AS VARCHAR)), 'not available') AS visibility,
            coalesce(try(CAST(suite_name AS VARCHAR)), '') AS suite_name,
            COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) AS gt_total,
            COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS tp_gt,
            COUNT(*) FILTER (WHERE source='EST' AND status IN ('TP','FP')) AS est_total,
            COUNT(*) FILTER (WHERE source='EST' AND status='FP') AS fp_est"""

_TPR_FPR_STATS_GROUP_BY = """t4dataset_id, topic_name, label, distance_bin, bin_idx,
            coalesce(try(CAST(visibility AS VARCHAR)), 'not available'),
            coalesce(try(CAST(suite_name AS VARCHAR)), '')"""


def sql_distance_bin_rates_from_eval_flat(
    source_eval_flat: str,
    filter_clause: str,
    *,
    metrics: str = "both",
) -> str:
    """TPR/FPR by ``distance_bin`` from ``view_eval_flat`` rows, with filters pushed into the stats CTE.

    Distance charts used to ``SELECT ... FROM view_tpr_fpr_* WHERE ...`` (nested view over parquet). On some
    DuckDB builds that plan can **SIGSEGV** the process (container exit **139**). This query inlines the same
    stats aggregation and applies ``WHERE`` on the flat view instead.
    """
    order_by = "ORDER BY CAST(REPLACE(SPLIT_PART(distance_bin, ',', 1), '[', ' ') AS INTEGER)"
    inner = f"""
    WITH stats AS (
        {_TPR_FPR_STATS_SELECT}
        FROM {source_eval_flat}
        WHERE ({filter_clause})
        GROUP BY
            {_TPR_FPR_STATS_GROUP_BY}
    )"""
    if metrics == "both":
        return f"""
        {inner}
        SELECT
            distance_bin,
            CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE 0 END AS tpr,
            CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr
        FROM stats
        GROUP BY distance_bin
        {order_by}
        """
    if metrics == "tpr":
        return f"""
        {inner}
        SELECT distance_bin,
            CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE 0 END AS tpr
        FROM stats
        GROUP BY distance_bin
        {order_by}
        """
    if metrics == "fpr":
        return f"""
        {inner}
        SELECT distance_bin,
            CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr
        FROM stats
        GROUP BY distance_bin
        {order_by}
        """
    raise ValueError(f"metrics must be 'both', 'tpr', or 'fpr', got {metrics!r}")


def sql_distance_bin_label_rates_from_eval_flat(
    source_eval_flat: str,
    filter_clause: str,
) -> str:
    """TPR/FPR by label and distance bin from ``view_eval_flat`` rows."""
    return f"""
    WITH stats AS (
        {_TPR_FPR_STATS_SELECT}
        FROM {source_eval_flat}
        WHERE ({filter_clause})
        GROUP BY
            {_TPR_FPR_STATS_GROUP_BY}
    )
    SELECT
        distance_bin,
        label,
        CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE 0 END AS tpr,
        CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr
    FROM stats
    GROUP BY distance_bin, label
    ORDER BY MIN(bin_idx), label
    """


def _report_int(v: Any) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"{int(round(float(v))):,}"


def _report_pct(v: Any) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"{float(v) * 100:.1f}%"


def _report_ratio(v: Any) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"{float(v):.3f}"


def _report_pp_delta(v: Any, *, lower_is_better: bool = False) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    val = float(v) * 100.0
    sign = "+" if val > 0 else ""
    if abs(val) < 0.05:
        return "flat"
    direction_good = (val > 0 and not lower_is_better) or (val < 0 and lower_is_better)
    suffix = "good" if direction_good else "risk"
    return f"{sign}{val:.1f} pp ({suffix})"


def _analyze_kpi_comparison(baseline: Optional[dict], candidate: Optional[dict], candidate_label: str = "Candidate") -> str:
    """Generate a natural-language interpretation of KPI comparison results.

    Provides a verdict and actionable recommendation for all combined scenarios
    (both improve, both degrade, trade-offs, flat).  Does NOT repeat the raw numbers
    already visible in the KPI cards above — only adds insight.
    """
    if baseline is None or candidate is None:
        return '<div class="kpi-analysis kpi-analysis-neutral">ⓘ Insufficient data for comparison analysis — one or both runs have no KPI data.</div>'

    # Extract values needed for classification
    tp_b, tp_c = baseline.get("tp", 0), candidate.get("tp", 0)
    fp_b, fp_c = baseline.get("fp", 0), candidate.get("fp", 0)
    fn_b, fn_c = baseline.get("fn", 0), candidate.get("fn", 0)
    tpr_b, tpr_c = baseline.get("tpr"), candidate.get("tpr")
    fpr_b, fpr_c = baseline.get("fpr"), candidate.get("fpr")
    prec_b, prec_c = baseline.get("precision"), candidate.get("precision")

    dtp = tp_c - tp_b
    dfp = fp_c - fp_b
    dfn = fn_c - fn_b
    dtpr = (tpr_c - tpr_b) if (tpr_b is not None and tpr_c is not None) else None
    dfpr = (fpr_c - fpr_b) if (fpr_b is not None and fpr_c is not None) else None
    dprec = (prec_c - prec_b) if (prec_b is not None and prec_c is not None) else None

    EPS_RATE = 0.001   # 0.1pp threshold for "flat"

    def classify_rate(delta):
        if delta is None:
            return "n/a"
        if delta > EPS_RATE:
            return "up"
        if delta < -EPS_RATE:
            return "down"
        return "flat"

    def fmt_rate_delta(delta):
        if delta is None:
            return "N/A"
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta * 100:.1f}pp"

    def fmt_count_delta(delta):
        sign = "+" if delta > 0 else ""
        return f"{sign}{abs(delta):,}"

    tpr_dir = classify_rate(dtpr)
    prec_dir = classify_rate(dprec)
    fpr_dir = classify_rate(dfpr)

    # --- Verdict line ---
    if tpr_dir == "up" and prec_dir == "up":
        verdict = (
            f'<strong class="kpi-analysis-verdict kpi-analysis-good">{candidate_label} improves on all key metrics</strong> — '
            f"both Recall and Precision increased vs baseline."
        )
        tone_class = "kpi-analysis-good"
    elif tpr_dir == "down" and prec_dir == "down":
        verdict = (
            f'<strong class="kpi-analysis-verdict kpi-analysis-bad">{candidate_label} degrades on all key metrics</strong> — '
            f"both Recall and Precision decreased vs baseline."
        )
        tone_class = "kpi-analysis-bad"
    elif tpr_dir == "up" and prec_dir == "down":
        verdict = (
            f'<strong class="kpi-analysis-verdict kpi-analysis-warn">Recall-precision trade-off detected:</strong> '
            f"Recall improved ({fmt_rate_delta(dtpr)}) but Precision decreased ({fmt_rate_delta(dprec)})."
        )
        tone_class = "kpi-analysis-warn"
    elif tpr_dir == "down" and prec_dir == "up":
        verdict = (
            f'<strong class="kpi-analysis-verdict kpi-analysis-warn">Precision-recall trade-off detected:</strong> '
            f"Precision improved ({fmt_rate_delta(dprec)}) but Recall decreased ({fmt_rate_delta(dtpr)})."
        )
        tone_class = "kpi-analysis-warn"
    elif tpr_dir == "flat" and prec_dir == "flat":
        verdict = (
            f'<strong class="kpi-analysis-verdict kpi-analysis-neutral">{candidate_label} is essentially unchanged</strong> — '
            f"both Recall and Precision are flat vs baseline."
        )
        tone_class = "kpi-analysis-neutral"
    elif tpr_dir == "flat":
        verdict = (
            f'<strong class="kpi-analysis-verdict kpi-analysis-neutral">{candidate_label}:</strong> '
            f"Recall is flat, Precision {prec_dir} ({fmt_rate_delta(dprec)})."
        )
        tone_class = "kpi-analysis-neutral"
    elif prec_dir == "flat":
        verdict = (
            f'<strong class="kpi-analysis-verdict kpi-analysis-neutral">{candidate_label}:</strong> '
            f"Precision is flat, Recall {tpr_dir} ({fmt_rate_delta(dtpr)})."
        )
        tone_class = "kpi-analysis-neutral"
    else:
        verdict = (
            f'<strong class="kpi-analysis-verdict kpi-analysis-warn">{candidate_label}:</strong> '
            f"Recall {tpr_dir} ({fmt_rate_delta(dtpr)}), Precision {prec_dir} ({fmt_rate_delta(dprec)})."
        )
        tone_class = "kpi-analysis-warn"

    # --- Interpretation / recommendation ---
    if tpr_dir == "up" and fpr_dir == "down":
        recommendation = (
            f"<strong>Strong improvement:</strong> model is both more sensitive (higher recall) "
            f"and more specific (lower FPR). This is the ideal outcome — consider this candidate for production."
        )
    elif tpr_dir == "up" and fpr_dir == "up":
        recommendation = (
            f"<strong>Higher recall at cost of more false positives:</strong> model finds more objects "
            f"but also generates more false alarms. Evaluate whether the recall gain ({fmt_rate_delta(dtpr)}) "
            f"justifies the precision cost ({fmt_rate_delta(dprec)})."
        )
    elif tpr_dir == "down" and fpr_dir == "down":
        recommendation = (
            f"<strong>More conservative model:</strong> fewer false positives but also lower recall. "
            f"Model may be too cautious — {fmt_count_delta(-dfn) if dfn else '0'} more GT objects are now missed."
        )
    elif tpr_dir == "down" and fpr_dir == "up":
        recommendation = (
            f"<strong>Degradation on all fronts:</strong> both recall dropped and false positives increased. "
            f"This candidate is strictly worse than baseline — do not adopt without further tuning."
        )
    elif tpr_dir == "flat" and prec_dir == "flat":
        recommendation = (
            f"<strong>No significant change:</strong> the candidate performs similarly to baseline across all metrics. "
            f"Adoption depends on other factors (e.g., latency, robustness to edge cases)."
        )
    elif tpr_dir == "up":
        recommendation = (
            f"<strong>Net positive:</strong> recall improved with manageable precision impact. "
            f"Review the {fmt_count_delta(abs(dfn) if dfn else 0)} FN-to-TP recoveries and "
            f"{fmt_count_delta(abs(dfp) if dfp else 0)} new FPs in the per-class / per-distance breakdowns below."
        )
    elif prec_dir == "up":
        recommendation = (
            f"<strong>Precision gain:</strong> {fmt_rate_delta(dprec)} improvement with recall impact. "
            f"Review the trade-off: {fmt_count_delta(abs(dfn) if dfn else 0)} more misses for "
            f"{fmt_count_delta(abs(dfp) if dfp else 0)} fewer false alarms."
        )
    else:
        recommendation = (
            f"The overall impact is mixed — check per-class and per-distance breakdowns "
            f"below for more granular insight."
        )

    # Edge-case notes
    notes = []
    gt_b = tp_b + fn_b
    gt_c = tp_c + fn_c
    if gt_b != gt_c:
        notes.append(f"GT total differs between runs ({gt_b:,} vs {gt_c:,}) — filter conditions may not be identical.")
    if tpr_b is None or tpr_c is None or prec_b is None or prec_c is None:
        notes.append("Some rate metrics are N/A due to zero denominators (no GT or EST objects in one run).")

    note_suffix = ""
    if notes:
        note_suffix = ' <span class="kpi-analysis-note">' + " ".join(notes) + "</span>"

    return (
        f'<div class="kpi-analysis {tone_class}">'
        f"{verdict}"
        f'<p class="kpi-analysis-recommendation">{recommendation}</p>'
        f"{note_suffix}"
        f"</div>"
    )


def _report_num_delta(v: Any, *, lower_is_better: bool = False) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    val = int(round(float(v)))
    sign = "+" if val > 0 else ""
    if val == 0:
        return "flat"
    direction_good = (val > 0 and not lower_is_better) or (val < 0 and lower_is_better)
    suffix = "good" if direction_good else "risk"
    return f"{sign}{val:,} ({suffix})"


def _report_nonempty_text(v: Any, fallback: str = "(not named)") -> str:
    if v is None or pd.isna(v):
        return fallback
    s = str(v).strip()
    return s if s else fallback


def _report_error_name(col_name: str) -> str:
    clean = col_name.replace("_delta", "").replace("mean_abs_", "").replace("_", " ").strip()
    if clean == "x error":
        return "mean |x error|"
    if clean == "y error":
        return "mean |y error|"
    if clean == "yaw error":
        return "mean |yaw error|"
    return clean or col_name


def _report_join_phrases(items: List[str], limit: int = 3) -> str:
    clean = [i for i in items if i]
    if not clean:
        return "none identified"
    clean = clean[:limit]
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _report_escape(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _report_html_list(items: List[str], *, empty: str = "このsliceでは明確なhotspotは検出されませんでした。") -> str:
    clean = [i for i in items if i]
    if not clean:
        return f"<p class=\"ds-report-muted\">{_report_escape(empty)}</p>"
    return "<ul>" + "".join(f"<li>{_report_escape(item)}</li>" for item in clean[:4]) + "</ul>"


def _report_metric_card(label: str, value: str, note: str = "", tone: str = "neutral") -> str:
    return (
        f"<div class=\"ds-report-metric ds-report-tone-{tone}\">"
        f"<span>{_report_escape(label)}</span>"
        f"<strong>{_report_escape(value)}</strong>"
        f"<em>{_report_escape(note)}</em>"
        "</div>"
    )


def _report_badge(text: str, tone: str) -> str:
    return f"<span class=\"ds-report-badge ds-report-badge-{tone}\">{_report_escape(text)}</span>"


def _report_html_panel(title: str, body: str, inner_html: str) -> str:
    return (
        "<div class=\"ds-report-panel ds-report-panel-wide\">"
        f"<h4>{_report_escape(title)}</h4>"
        f"<p>{_report_escape(body)}</p>"
        f"{inner_html}"
        "</div>"
    )


def _report_delta_tone(v: Any, *, lower_is_better: bool = False) -> str:
    if v is None or pd.isna(v) or abs(float(v)) < 1e-9:
        return "neutral"
    val = float(v)
    good = (val > 0 and not lower_is_better) or (val < 0 and lower_is_better)
    return "good" if good else "risk"


def _report_signpost(title: str, body: str, items: Optional[List[str]] = None) -> str:
    list_html = _report_html_list(items or [], empty="active filter上で特に支配的な項目は検出されませんでした。")
    return (
        "<div class=\"ds-report-panel\">"
        f"<h4>{_report_escape(title)}</h4>"
        f"<p>{_report_escape(body)}</p>"
        f"{list_html}"
        "</div>"
    )


def _report_shell(
    *,
    title: str,
    subtitle: str,
    badge: str,
    badge_tone: str,
    lead: str,
    metric_cards: List[str],
    sections: List[str],
    footnote: str,
) -> str:
    return f"""
<style>
.ds-report-shell {{
  border: 1px solid rgba(15, 23, 42, 0.12);
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
  overflow: hidden;
  margin: 0.6rem 0 1.2rem 0;
}}
.ds-report-head {{
  padding: 1.35rem 1.55rem;
  background: linear-gradient(135deg, #f8fafc 0%, #eef6f3 52%, #f7f2ea 100%);
  border-bottom: 1px solid rgba(15, 23, 42, 0.1);
}}
.ds-report-kicker {{
  display: flex;
  gap: 0.55rem;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 0.5rem;
}}
.ds-report-title {{
  margin: 0;
  color: #0f172a;
  font-size: 1.55rem;
  line-height: 1.2;
  letter-spacing: 0;
}}
.ds-report-subtitle {{
  margin: 0.35rem 0 0 0;
  color: #475569;
  font-size: 0.93rem;
  line-height: 1.45;
}}
.ds-report-lead {{
  margin: 0;
  padding: 1.15rem 1.55rem 0 1.55rem;
  color: #1f2937;
  font-size: 1.02rem;
  line-height: 1.55;
}}
.ds-report-badge {{
  display: inline-block;
  border-radius: 999px;
  padding: 0.24rem 0.62rem;
  font-size: 0.72rem;
  font-weight: 750;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}}
.ds-report-badge-good {{ background: #dcfce7; color: #166534; }}
.ds-report-badge-risk {{ background: #fee2e2; color: #991b1b; }}
.ds-report-badge-mixed {{ background: #fef3c7; color: #92400e; }}
.ds-report-badge-neutral {{ background: #e2e8f0; color: #334155; }}
.ds-report-metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 0.75rem;
  padding: 1.1rem 1.55rem 0.25rem 1.55rem;
}}
.ds-report-metric {{
  border: 1px solid rgba(15, 23, 42, 0.1);
  border-radius: 8px;
  padding: 0.85rem 0.95rem;
  background: #f8fafc;
  min-height: 102px;
}}
.ds-report-metric span {{
  display: block;
  color: #64748b;
  font-size: 0.74rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.ds-report-metric strong {{
  display: block;
  margin-top: 0.25rem;
  color: #0f172a;
  font-size: 1.35rem;
  line-height: 1.15;
}}
.ds-report-metric em {{
  display: block;
  margin-top: 0.35rem;
  color: #475569;
  font-size: 0.82rem;
  font-style: normal;
  line-height: 1.35;
}}
.ds-report-tone-good {{ background: #f0fdf4; border-color: rgba(22, 101, 52, 0.22); }}
.ds-report-tone-risk {{ background: #fff7ed; border-color: rgba(194, 65, 12, 0.24); }}
.ds-report-tone-neutral {{ background: #f8fafc; }}
.ds-report-body {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.9rem;
  padding: 1rem 1.55rem 1.4rem 1.55rem;
}}
.ds-report-panel {{
  border-top: 3px solid #0d9488;
  background: #ffffff;
  border-radius: 8px;
  padding: 0.95rem 1rem;
  box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
}}
.ds-report-panel-wide {{
  grid-column: 1 / -1;
}}
.ds-report-table-wrap {{
  overflow-x: auto;
}}
.ds-report-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
}}
.ds-report-table th {{
  text-align: left;
  color: #475569;
  background: #f8fafc;
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.ds-report-table th, .ds-report-table td {{
  padding: 0.55rem 0.65rem;
  border-bottom: 1px solid rgba(15, 23, 42, 0.08);
  white-space: nowrap;
}}
.ds-report-table td {{
  color: #1f2937;
  font-variant-numeric: tabular-nums;
}}
.ds-report-panel h4 {{
  margin: 0 0 0.45rem 0;
  font-size: 0.98rem;
  line-height: 1.25;
  color: #0f172a;
  letter-spacing: 0;
}}
.ds-report-panel p, .ds-report-footnote {{
  color: #334155;
  font-size: 0.9rem;
  line-height: 1.48;
}}
.ds-report-panel p {{ margin: 0 0 0.55rem 0; }}
.ds-report-panel ul {{
  margin: 0;
  padding-left: 1.05rem;
}}
.ds-report-panel li {{
  margin: 0.28rem 0;
  color: #1f2937;
  font-size: 0.88rem;
  line-height: 1.42;
}}
.ds-report-muted {{ color: #64748b !important; font-style: italic; }}
.ds-report-footnote {{
  margin: 0;
  padding: 0.85rem 1.55rem 1.1rem 1.55rem;
  border-top: 1px solid rgba(15, 23, 42, 0.08);
  background: #fafafa;
}}
</style>
<article class="ds-report-shell">
  <header class="ds-report-head">
    <div class="ds-report-kicker">{_report_badge(badge, badge_tone)}</div>
    <h3 class="ds-report-title">{_report_escape(title)}</h3>
    <p class="ds-report-subtitle">{_report_escape(subtitle)}</p>
  </header>
  <p class="ds-report-lead">{_report_escape(lead)}</p>
  <section class="ds-report-metrics">{''.join(metric_cards)}</section>
  <section class="ds-report-body">{''.join(sections)}</section>
  <p class="ds-report-footnote">{_report_escape(footnote)}</p>
</article>
"""


def _report_label_metrics(con, view: str, filter_clause: str) -> pd.DataFrame:
    q = f"""
    WITH stats AS (
        SELECT
            COALESCE(CAST(label AS VARCHAR), '') AS label,
            COUNT(*) FILTER (WHERE source = 'GT' AND status IN ('TP', 'FN')) AS gt_total,
            COUNT(*) FILTER (WHERE source = 'GT' AND status = 'TP') AS tp,
            COUNT(*) FILTER (WHERE source = 'GT' AND status = 'FN') AS fn,
            COUNT(*) FILTER (WHERE source = 'EST' AND status IN ('TP', 'FP')) AS est_total,
            COUNT(*) FILTER (WHERE source = 'EST' AND status = 'TP') AS tp_est,
            COUNT(*) FILTER (WHERE source = 'EST' AND status = 'FP') AS fp
        FROM {view}
        WHERE {filter_clause}
        GROUP BY 1
    )
    SELECT
        label,
        gt_total,
        tp,
        fn,
        est_total,
        tp_est,
        fp,
        CASE WHEN gt_total > 0 THEN CAST(tp AS DOUBLE) / gt_total ELSE NULL END AS tpr,
        CASE WHEN est_total > 0 THEN CAST(fp AS DOUBLE) / est_total ELSE NULL END AS fpr,
        CASE WHEN est_total > 0 THEN CAST(tp_est AS DOUBLE) / est_total ELSE NULL END AS precision,
        CASE
            WHEN gt_total > 0 AND est_total > 0 AND (CAST(tp AS DOUBLE) / gt_total + CAST(tp_est AS DOUBLE) / est_total) > 0
            THEN 2 * (CAST(tp AS DOUBLE) / gt_total) * (CAST(tp_est AS DOUBLE) / est_total)
                 / ((CAST(tp AS DOUBLE) / gt_total) + (CAST(tp_est AS DOUBLE) / est_total))
            ELSE NULL
        END AS f1
    FROM stats
    ORDER BY label
    """
    return con.execute(q).df()


def _report_scene_metrics(con, view: str, filter_clause: str) -> pd.DataFrame:
    q = f"""
    WITH stats AS (
        SELECT
            COALESCE(CAST(scenario_name AS VARCHAR), '') AS scenario_name,
            COALESCE(CAST(t4dataset_name AS VARCHAR), '') AS t4dataset_name,
            COALESCE(CAST(suite_name AS VARCHAR), '') AS suite_name,
            COALESCE(CAST(t4dataset_id AS VARCHAR), '') AS t4dataset_id,
            COUNT(*) FILTER (WHERE source = 'GT' AND status IN ('TP', 'FN')) AS gt_total,
            COUNT(*) FILTER (WHERE source = 'GT' AND status = 'TP') AS tp,
            COUNT(*) FILTER (WHERE source = 'GT' AND status = 'FN') AS fn,
            COUNT(*) FILTER (WHERE source = 'EST' AND status IN ('TP', 'FP')) AS est_total,
            COUNT(*) FILTER (WHERE source = 'EST' AND status = 'FP') AS fp
        FROM {view}
        WHERE {filter_clause}
        GROUP BY 1, 2, 3, 4
    )
    SELECT
        *,
        CASE WHEN gt_total > 0 THEN CAST(tp AS DOUBLE) / gt_total ELSE NULL END AS tpr,
        CASE WHEN gt_total > 0 THEN CAST(fn AS DOUBLE) / gt_total ELSE NULL END AS fn_rate,
        CASE WHEN est_total > 0 THEN CAST(fp AS DOUBLE) / est_total ELSE NULL END AS fpr
    FROM stats
    ORDER BY fn DESC, fn_rate DESC, fp DESC
    """
    return con.execute(q).df()


def _report_fn_frames(con, view: str, filter_clause: str) -> pd.DataFrame:
    q = f"""
    SELECT
        COALESCE(CAST(t4dataset_id AS VARCHAR), '') AS t4dataset_id,
        CAST(frame_index AS VARCHAR) AS frame_index,
        COALESCE(MAX(CAST(scenario_name AS VARCHAR)), '') AS scenario_name,
        COALESCE(MAX(CAST(t4dataset_name AS VARCHAR)), '') AS t4dataset_name,
        COALESCE(MAX(CAST(suite_name AS VARCHAR)), '') AS suite_name,
        COUNT(*) AS fn
    FROM {view}
    WHERE source = 'GT' AND status = 'FN' AND frame_index IS NOT NULL AND {filter_clause}
    GROUP BY 1, 2
    ORDER BY fn DESC
    LIMIT 20
    """
    return con.execute(q).df()


def _report_error_metrics(con, view: str, filter_clause: str) -> pd.DataFrame:
    try:
        sample_df = con.execute(f"SELECT * FROM {view} LIMIT 1").df()
    except Exception:
        return pd.DataFrame()
    if not all(c in sample_df.columns for c in ["x_error", "y_error", "yaw_error"]):
        return pd.DataFrame()
    q = f"""
    SELECT
        COALESCE(CAST(label AS VARCHAR), '') AS label,
        AVG(ABS(CAST(x_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND x_error IS NOT NULL) AS mean_abs_x_error,
        AVG(ABS(CAST(y_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND y_error IS NOT NULL) AS mean_abs_y_error,
        AVG(ABS(CAST(yaw_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND yaw_error IS NOT NULL) AS mean_abs_yaw_error
    FROM {view}
    WHERE {filter_clause}
    GROUP BY 1
    ORDER BY label
    """
    return con.execute(q).df()


def _report_diff_by_label(
    con,
    base_view: str,
    comp_view: str,
    base_filter: str,
    comp_filter: str,
) -> pd.DataFrame:
    q = f"""
    WITH base_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COALESCE(MAX(CAST(label AS VARCHAR)), '') AS label,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base
        FROM {base_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {base_filter}
        GROUP BY 1, 2, 3
    ),
    comp_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COALESCE(MAX(CAST(label AS VARCHAR)), '') AS label,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp
        FROM {comp_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {comp_filter}
        GROUP BY 1, 2, 3
    ),
    joined AS (
        SELECT
            COALESCE(b.label, c.label, '') AS label,
            COALESCE(b.tp_base, FALSE) AS tp_base,
            COALESCE(c.tp_comp, FALSE) AS tp_comp
        FROM base_gt b
        FULL OUTER JOIN comp_gt c
            ON b.t4dataset_id = c.t4dataset_id
           AND b.frame_index = c.frame_index
           AND b.gt_uuid = c.gt_uuid
    )
    SELECT
        label,
        COUNT(*) AS total_gt,
        COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS improved_cnt,
        COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS degraded_cnt,
        COUNT(*) FILTER (WHERE tp_base AND tp_comp) AS both_tp_cnt,
        COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp) AS both_fn_cnt,
        SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS net_tp_delta
    FROM joined
    GROUP BY 1
    ORDER BY net_tp_delta DESC
    """
    return con.execute(q).df()


def _report_diff_by_scene_or_frame(
    con,
    base_view: str,
    comp_view: str,
    base_filter: str,
    comp_filter: str,
    *,
    by_frame: bool,
) -> pd.DataFrame:
    frame_select = "COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index," if by_frame else ""
    frame_group = ", frame_index" if by_frame else ""
    q = f"""
    WITH base_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
            COALESCE(MAX(CAST(suite_name AS VARCHAR)), '') AS suite_name,
            COALESCE(MAX(CAST(scenario_name AS VARCHAR)), '') AS scenario_name,
            COALESCE(MAX(CAST(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
        FROM {base_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {base_filter}
        GROUP BY 1, 2, 3
    ),
    comp_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
            COALESCE(MAX(CAST(suite_name AS VARCHAR)), '') AS suite_name,
            COALESCE(MAX(CAST(scenario_name AS VARCHAR)), '') AS scenario_name,
            COALESCE(MAX(CAST(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
        FROM {comp_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {comp_filter}
        GROUP BY 1, 2, 3
    ),
    joined AS (
        SELECT
            COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR), '') AS t4dataset_id,
            {frame_select}
            COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
            COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
            COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name,
            COALESCE(b.tp_base, FALSE) AS tp_base,
            COALESCE(c.tp_comp, FALSE) AS tp_comp
        FROM base_gt b
        FULL OUTER JOIN comp_gt c
            ON b.t4dataset_id = c.t4dataset_id
           AND b.frame_index = c.frame_index
           AND b.gt_uuid = c.gt_uuid
    )
    SELECT
        t4dataset_id,
        {('frame_index,' if by_frame else '')}
        suite_name,
        scenario_name,
        t4dataset_name,
        COUNT(*) AS total_gt,
        COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS improved_cnt,
        COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS degraded_cnt,
        COUNT(*) FILTER (WHERE tp_base AND tp_comp) AS both_tp_cnt,
        COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp) AS both_fn_cnt,
        SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS net_tp_delta
    FROM joined
    GROUP BY t4dataset_id, suite_name, scenario_name, t4dataset_name{frame_group}
    ORDER BY degraded_cnt DESC, improved_cnt ASC, net_tp_delta ASC
    LIMIT 50
    """
    return con.execute(q).df()


def _report_frame_ref(row: pd.Series) -> str:
    scen = _report_nonempty_text(row.get("scenario_name"))
    t4 = _report_nonempty_text(row.get("t4dataset_name"), "")
    fid = _report_nonempty_text(row.get("frame_index"), "?")
    if t4 and t4 != scen:
        return f"{scen} / {t4}, frame {fid}"
    return f"{scen}, frame {fid}"


def _report_scene_ref(row: pd.Series) -> str:
    scen = _report_nonempty_text(row.get("scenario_name"))
    t4 = _report_nonempty_text(row.get("t4dataset_name"), "")
    if t4 and t4 != scen:
        return f"{scen} / {t4}"
    return scen


def _report_top_labels(df: pd.DataFrame, metric: str, count_col: str, ascending: bool = False) -> List[str]:
    if df.empty or metric not in df.columns:
        return []
    d = df.copy()
    if count_col in d.columns:
        d = d[d[count_col].fillna(0) > 0]
    if d.empty:
        return []
    d = d.sort_values([metric, count_col], ascending=[ascending, False]).head(3)
    out = []
    for _, r in d.iterrows():
        if metric in ("tpr", "fpr", "precision", "f1"):
            out.append(f"{_report_nonempty_text(r['label'], '(no label)')} ({metric.upper()} {_report_pct(r[metric])}, n={_report_int(r.get(count_col))})")
        else:
            out.append(f"{_report_nonempty_text(r['label'], '(no label)')} ({metric} {_report_int(r[metric])})")
    return out


def _report_distance_phrases(df_dist: pd.DataFrame, *, compare_base: Optional[pd.DataFrame] = None) -> List[str]:
    if df_dist.empty:
        return []
    d = df_dist.copy()
    phrases: List[str] = []
    if compare_base is not None and not compare_base.empty:
        merged = compare_base.merge(df_dist, on="distance_bin", suffixes=("_base", "_candidate"))
        if not merged.empty:
            merged["tpr_delta"] = merged["tpr_candidate"] - merged["tpr_base"]
            merged["fpr_delta"] = merged["fpr_candidate"] - merged["fpr_base"]
            worst_tpr = merged.sort_values("tpr_delta", ascending=True).head(1)
            best_tpr = merged.sort_values("tpr_delta", ascending=False).head(1)
            worst_fpr = merged.sort_values("fpr_delta", ascending=False).head(1)
            if not worst_tpr.empty:
                r = worst_tpr.iloc[0]
                phrases.append(f"largest TP-rate regression at {r['distance_bin']} ({_report_pp_delta(r['tpr_delta'])})")
            if not best_tpr.empty:
                r = best_tpr.iloc[0]
                phrases.append(f"largest TP-rate gain at {r['distance_bin']} ({_report_pp_delta(r['tpr_delta'])})")
            if not worst_fpr.empty:
                r = worst_fpr.iloc[0]
                phrases.append(f"largest FP-rate increase at {r['distance_bin']} ({_report_pp_delta(r['fpr_delta'], lower_is_better=True)})")
        return phrases
    if "tpr" in d.columns:
        worst = d.sort_values("tpr", ascending=True).head(1)
        if not worst.empty:
            r = worst.iloc[0]
            phrases.append(f"weakest TP rate at {r['distance_bin']} ({_report_pct(r['tpr'])})")
    if "fpr" in d.columns:
        worst_fp = d.sort_values("fpr", ascending=False).head(1)
        if not worst_fp.empty:
            r = worst_fp.iloc[0]
            phrases.append(f"highest FP rate at {r['distance_bin']} ({_report_pct(r['fpr'])})")
    return phrases


def _report_kpi_compare_table(base_kpi: Optional[Dict[str, Any]], candidate_kpi: Optional[Dict[str, Any]]) -> str:
    base = base_kpi or {}
    cand = candidate_kpi or {}
    rows = [
        ("GT", _report_int(base.get("gt")), _report_int(cand.get("gt")), _report_num_delta(cand.get("gt", 0) - base.get("gt", 0))),
        ("TP", _report_int(base.get("tp")), _report_int(cand.get("tp")), _report_num_delta(cand.get("tp", 0) - base.get("tp", 0))),
        ("FP", _report_int(base.get("fp")), _report_int(cand.get("fp")), _report_num_delta(cand.get("fp", 0) - base.get("fp", 0), lower_is_better=True)),
        ("FN", _report_int(base.get("fn")), _report_int(cand.get("fn")), _report_num_delta(cand.get("fn", 0) - base.get("fn", 0), lower_is_better=True)),
        (
            "Precision",
            _report_pct(base.get("precision")),
            _report_pct(cand.get("precision")),
            _report_pp_delta(cand.get("precision") - base.get("precision")) if base.get("precision") is not None and cand.get("precision") is not None else "n/a",
        ),
        (
            "Recall",
            _report_pct(base.get("recall", base.get("tpr"))),
            _report_pct(cand.get("recall", cand.get("tpr"))),
            _report_pp_delta(cand.get("tpr") - base.get("tpr")) if base.get("tpr") is not None and cand.get("tpr") is not None else "n/a",
        ),
        (
            "F1",
            _report_ratio(base.get("f1")),
            _report_ratio(cand.get("f1")),
            f"{(cand.get('f1') - base.get('f1')):+.3f}" if base.get("f1") is not None and cand.get("f1") is not None else "n/a",
        ),
    ]
    body = "".join(
        "<tr>"
        f"<td>{_report_escape(metric)}</td>"
        f"<td>{_report_escape(base_v)}</td>"
        f"<td>{_report_escape(cand_v)}</td>"
        f"<td>{_report_escape(diff)}</td>"
        "</tr>"
        for metric, base_v, cand_v, diff in rows
    )
    return (
        "<div class=\"ds-report-table-wrap\"><table class=\"ds-report-table\">"
        "<thead><tr><th>Metric</th><th>Baseline A</th><th>Candidate</th><th>Diff</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _report_meta_table(rows: List[Tuple[str, str]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{_report_escape(k)}</td>"
        f"<td>{_report_escape(v)}</td>"
        "</tr>"
        for k, v in rows
    )
    return (
        "<div class=\"ds-report-table-wrap\"><table class=\"ds-report-table\">"
        "<thead><tr><th>項目</th><th>内容</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _report_label_distance_compare(
    con,
    base_view: str,
    candidate_view: str,
    base_filter: str,
    candidate_filter: str,
) -> pd.DataFrame:
    base = con.execute(sql_distance_bin_label_rates_from_eval_flat(base_view, base_filter)).df()
    cand = con.execute(sql_distance_bin_label_rates_from_eval_flat(candidate_view, candidate_filter)).df()
    if base.empty or cand.empty:
        return pd.DataFrame()
    merged = base.merge(cand, on=["label", "distance_bin"], suffixes=("_base", "_candidate"))
    if merged.empty:
        return merged
    merged["tpr_delta"] = merged["tpr_candidate"] - merged["tpr_base"]
    merged["fpr_delta"] = merged["fpr_candidate"] - merged["fpr_base"]
    return merged


def _report_label_distance_phrases(df_label_dist_delta: pd.DataFrame) -> List[str]:
    if df_label_dist_delta.empty:
        return []
    out: List[str] = []
    tpr_gain = df_label_dist_delta[df_label_dist_delta["tpr_delta"] > 0.0005].sort_values("tpr_delta", ascending=False).head(1)
    tpr_loss = df_label_dist_delta[df_label_dist_delta["tpr_delta"] < -0.0005].sort_values("tpr_delta", ascending=True).head(1)
    fp_rise = df_label_dist_delta[df_label_dist_delta["fpr_delta"] > 0.0005].sort_values("fpr_delta", ascending=False).head(1)
    if not tpr_gain.empty:
        r = tpr_gain.iloc[0]
        out.append(f"{_report_nonempty_text(r['label'], '(no label)')} @ {r['distance_bin']} TP rate improved {_report_pp_delta(r['tpr_delta'])}")
    if not tpr_loss.empty:
        r = tpr_loss.iloc[0]
        out.append(f"{_report_nonempty_text(r['label'], '(no label)')} @ {r['distance_bin']} TP rate regressed {_report_pp_delta(r['tpr_delta'])}")
    if not fp_rise.empty:
        r = fp_rise.iloc[0]
        out.append(f"{_report_nonempty_text(r['label'], '(no label)')} @ {r['distance_bin']} FP rate increased {_report_pp_delta(r['fpr_delta'], lower_is_better=True)}")
    return out


def _report_frame_concentration_phrase(df_diff_frame: pd.DataFrame, total_degraded: Optional[int]) -> str:
    if df_diff_frame.empty or not total_degraded:
        return "フレーム集中度は現在のデータでは判定できません。"
    degraded = df_diff_frame[df_diff_frame["degraded_cnt"] > 0].copy()
    if degraded.empty:
        return "デグレが集中しているフレームhotspotは検出されませんでした。"
    top10 = float(degraded.head(10)["degraded_cnt"].sum())
    share = top10 / max(float(total_degraded), 1.0)
    if share >= 0.35:
        return f"デグレは集中傾向です。上位10フレームだけでデグレobjectの{_report_pct(share)}を説明しており、少数sceneが主因の可能性があります。"
    if share <= 0.10 and len(degraded) >= 30:
        return f"デグレは分散傾向です。上位10フレームの寄与は{_report_pct(share)}に留まり、局所caseではなくsystematicな挙動変化の可能性があります。"
    return f"デグレは中程度に集中しています。上位10フレームがデグレobjectの{_report_pct(share)}を説明しています。"


def _report_safety_perspective(df_diff_label: pd.DataFrame) -> List[str]:
    if df_diff_label.empty:
        return []
    priority_keywords = ("car", "pedestrian", "truck", "bus", "bicycle", "bike", "cyclist", "motorcycle")
    d = df_diff_label.copy()
    d["label_norm"] = d["label"].astype(str).str.lower()
    priority = d[d["label_norm"].apply(lambda s: any(k in s for k in priority_keywords))]
    if priority.empty:
        return ["一般的な安全重要class名は検出されませんでした。安全観点の結論にはclass命名の確認が必要です。"]
    out = []
    gains = priority[priority["net_tp_delta"] > 0].sort_values("net_tp_delta", ascending=False).head(2)
    losses = priority[priority["net_tp_delta"] < 0].sort_values("net_tp_delta", ascending=True).head(2)
    if not gains.empty:
        out.append("安全重要classの改善: " + _report_join_phrases([
            f"{_report_nonempty_text(r['label'], '(no label)')} net {_report_num_delta(r['net_tp_delta'])}"
            for _, r in gains.iterrows()
        ]))
    if not losses.empty:
        out.append("安全重要classのデグレ: " + _report_join_phrases([
            f"{_report_nonempty_text(r['label'], '(no label)')} net {_report_num_delta(r['net_tp_delta'])}"
            for _, r in losses.iterrows()
        ]))
    if not out:
        out.append("主要な交通参加者classは概ね安定しており、残りの変化は優先度の低いclassに寄っている可能性があります。")
    return out


def _report_recommendation(
    *,
    tpr_delta: Optional[float],
    fp_delta: int,
    total_improved: Optional[int],
    total_degraded: Optional[int],
    diff_loss: List[str],
    dist_phrases: List[str],
) -> List[str]:
    improved = total_improved or 0
    degraded = total_degraded or 0
    if tpr_delta is not None and tpr_delta >= -0.0005 and improved > degraded and fp_delta <= 0:
        return ["Candidateをrelease候補として維持できます。", "ただし記載したhotspotでregression確認を行うべきです。"]
    if improved > degraded and fp_delta > 0:
        return ["Candidateは有望ですが、FP増加をrelease gateの確認項目にしてください。", f"優先確認対象: {_report_join_phrases(dist_phrases + diff_loss)}."]
    if degraded >= improved:
        return ["現時点では純粋な改善とは判断しない方が安全です。", f"重点調査対象: {_report_join_phrases(diff_loss + dist_phrases)}."]
    return ["Candidateは概ねstableです。", "scenario/frame hotspotを確認し、release対象ODDで問題になる構造変化かを判断してください。"]


def _report_human_pp(v: Any, *, sign: bool = True) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    val = float(v)
    prefix = "+" if sign and val > 0 else ""
    return f"{prefix}{val:.3f}"


def _report_class_label_jp(label: str) -> str:
    return str(label)


def _report_exec_summary_compare(
    *,
    candidate_label: str,
    tpr_delta: Optional[float],
    precision_delta: Optional[float],
    f1_delta: Optional[float],
    df_label: pd.DataFrame,
    df_candidate_dist: pd.DataFrame,
    df_critical_cases: pd.DataFrame,
    recommendation: List[str],
) -> str:
    if df_label.empty:
        weak_class_phrase = "安全重要class別の悪化は現在のsliceでは特定できませんでした"
    else:
        d = df_label.copy()
        d["label_norm"] = d["label"].astype(str).str.lower()
        priority_order = ["pedestrian", "car", "truck", "bus", "bicycle", "bike", "cyclist", "motorcycle"]
        priority = d[d["label_norm"].apply(lambda s: any(k in s for k in priority_order))].copy()
        if priority.empty:
            priority = d
        loss = priority.sort_values("tpr_delta", ascending=True).head(1)
        if not loss.empty and float(loss.iloc[0]["tpr_delta"]) < -0.0005:
            r = loss.iloc[0]
            weak_class_phrase = f"{_report_class_label_jp(str(r['label']))}Recallが低下（Δ{_report_human_pp(r['tpr_delta'])}）"
        else:
            weak_class_phrase = "安全重要classのRecall低下は大きくありません"

    far_phrase = "遠距離の改善は限定的です"
    if not df_candidate_dist.empty and "distance_bin" in df_candidate_dist.columns:
        far = df_candidate_dist[df_candidate_dist["distance_bin"].astype(str).str.extract(r"\[(\d+)", expand=False).fillna("0").astype(int) >= 50]
        if not far.empty:
            best_far = far.sort_values("tpr", ascending=False).head(1)
            if not best_far.empty:
                far_phrase = f"遠距離（50m+）では{best_far.iloc[0]['distance_bin']}のTP rateが{_report_pct(best_far.iloc[0]['tpr'])}です"

    critical_phrase = (
        "近距離・高可視性の安全criticalデグレが確認されました"
        if df_critical_cases is not None and not df_critical_cases.empty
        else "近距離・高可視性の安全criticalデグレは検出されていません"
    )
    if df_critical_cases is not None and not df_critical_cases.empty:
        decision = "安全上criticalなデグレが解消されるまで本番採用は推奨しません。"
    elif tpr_delta is not None and tpr_delta > 0.002 and (f1_delta is None or f1_delta >= -0.0005):
        decision = "本番採用に向けて前向きですが、記載hotspotの確認をrelease gate条件とします。"
    else:
        decision = _report_join_phrases(recommendation, limit=1)

    overall = (
        f"{candidate_label}は全体Recallをδ{_report_human_pp(tpr_delta)}改善"
        if tpr_delta is not None and tpr_delta > 0
        else f"{candidate_label}の全体Recall差分はδ{_report_human_pp(tpr_delta)}"
    )
    precision_part = f"Precision差分はδ{_report_human_pp(precision_delta)}" if precision_delta is not None else "Precision差分はn/a"
    f1_part = f"F1差分はδ{_report_human_pp(f1_delta)}" if f1_delta is not None else "F1差分はn/a"
    return (
        f"{overall}していますが、{weak_class_phrase}。"
        f"{far_phrase}。一方で、{critical_phrase}。"
        f"{precision_part}、{f1_part}。"
        f"{decision}"
    )


def _report_degraded_object_details(
    con,
    base_view: str,
    comp_view: str,
    base_filter: str,
    comp_filter: str,
) -> pd.DataFrame:
    q = f"""
    WITH base_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COALESCE(MAX(CAST(label AS VARCHAR)), '') AS label,
            MAX(dist_h) AS dist_h,
            COALESCE(MAX(CAST(visibility AS VARCHAR)), '') AS visibility,
            MAX(try_cast(pointcloud_num AS DOUBLE)) AS pointcloud_num,
            COALESCE(MAX(CAST(scenario_name AS VARCHAR)), '') AS scenario_name,
            COALESCE(MAX(CAST(t4dataset_name AS VARCHAR)), '') AS t4dataset_name,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base
        FROM {base_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {base_filter}
        GROUP BY 1, 2, 3
    ),
    comp_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp
        FROM {comp_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {comp_filter}
        GROUP BY 1, 2, 3
    )
    SELECT
        CAST(b.t4dataset_id AS VARCHAR) AS t4dataset_id,
        CAST(b.frame_index AS VARCHAR) AS frame_index,
        b.gt_uuid,
        b.label,
        b.dist_h,
        b.visibility,
        b.pointcloud_num,
        b.scenario_name,
        b.t4dataset_name
    FROM base_gt b
    LEFT JOIN comp_gt c
        ON b.t4dataset_id = c.t4dataset_id
       AND b.frame_index = c.frame_index
       AND b.gt_uuid = c.gt_uuid
    WHERE b.tp_base AND NOT COALESCE(c.tp_comp, FALSE)
    ORDER BY b.dist_h ASC, b.pointcloud_num DESC NULLS LAST
    LIMIT 2000
    """
    try:
        return con.execute(q).df()
    except Exception:
        return pd.DataFrame()


def _report_critical_case_phrases(df_degraded_objects: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    if df_degraded_objects.empty:
        return [], pd.DataFrame()
    d = df_degraded_objects.copy()
    vis = d["visibility"].fillna("").astype(str).str.upper()
    critical = d[
        (d["dist_h"].fillna(1e9) <= 20.0)
        & (vis.isin(["FULL", "MOST"]))
        & (d["pointcloud_num"].fillna(0) >= 20)
    ].copy()
    if critical.empty:
        return ["20m以内・FULL/MOST・点群20点以上に該当する安全クリティカルなデグレは検出されませんでした。"], critical
    phrases = []
    for _, r in critical.head(3).iterrows():
        uuid_s = _report_nonempty_text(r.get("gt_uuid"), "")[:8]
        phrases.append(
            f"{_report_nonempty_text(r.get('label'), '(no label)')} / "
            f"{float(r.get('dist_h', 0.0)):.1f}m / "
            f"点群{_report_int(r.get('pointcloud_num'))} / "
            f"{_report_frame_ref(r)} / uuid={uuid_s}"
        )
    return phrases, critical


def _report_consecutive_failure_phrases(df_degraded_objects: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    if df_degraded_objects.empty:
        return [], pd.DataFrame()
    d = df_degraded_objects.copy()
    grouped = (
        d.groupby(["t4dataset_id", "gt_uuid", "label", "scenario_name"], dropna=False)
        .agg(
            degraded_frames=("frame_index", "nunique"),
            min_dist=("dist_h", "min"),
            max_pointcloud=("pointcloud_num", "max"),
        )
        .reset_index()
        .sort_values(["degraded_frames", "max_pointcloud"], ascending=[False, False])
    )
    if grouped.empty:
        return [], grouped
    phrases = []
    for _, r in grouped.head(3).iterrows():
        uuid_s = _report_nonempty_text(r.get("gt_uuid"), "")[:8]
        phrases.append(
            f"{_report_nonempty_text(r.get('label'), '(no label)')} / "
            f"{_report_nonempty_text(r.get('scenario_name'))} / "
            f"{_report_int(r.get('degraded_frames'))} frames / "
            f"min {float(r.get('min_dist', 0.0)):.1f}m / uuid={uuid_s}"
        )
    return phrases, grouped


def build_single_detection_report(
    con,
    *,
    run_label: str,
    view: str,
    filter_clause: str,
    scope_label: str,
    kpi: Optional[Dict[str, Any]],
) -> Tuple[str, str, Dict[str, pd.DataFrame]]:
    df_label = _report_label_metrics(con, view, filter_clause)
    df_scene = _report_scene_metrics(con, view, filter_clause)
    df_frames = _report_fn_frames(con, view, filter_clause)
    df_dist = con.execute(sql_distance_bin_rates_from_eval_flat(view, filter_clause, metrics="both")).df()
    df_err = _report_error_metrics(con, view, filter_clause)

    strong_labels = _report_top_labels(df_label, "tpr", "gt_total", ascending=False)
    weak_labels = _report_top_labels(df_label, "fn", "gt_total", ascending=False)
    fp_labels = _report_top_labels(df_label, "fp", "est_total", ascending=False)
    dist_phrases = _report_distance_phrases(df_dist)

    top_scenes = []
    if not df_scene.empty:
        for _, r in df_scene.sort_values(["fn", "fn_rate", "fp"], ascending=[False, False, False]).head(3).iterrows():
            top_scenes.append(
                f"{_report_scene_ref(r)} (FN {_report_int(r['fn'])}, FN rate {_report_pct(r['fn_rate'])}, FP {_report_int(r['fp'])})"
            )
    top_frames = []
    if not df_frames.empty:
        for _, r in df_frames.head(3).iterrows():
            top_frames.append(f"{_report_frame_ref(r)} (FN {_report_int(r['fn'])})")

    error_phrase = ""
    if not df_err.empty:
        err_long = df_err.melt(id_vars=["label"], var_name="error_type", value_name="mean_error")
        err_long = err_long.dropna().sort_values("mean_error", ascending=False)
        if not err_long.empty:
            r = err_long.iloc[0]
            error_phrase = (
                f"The largest TP localization error is {_report_error_name(str(r['error_type']))} "
                f"on {_report_nonempty_text(r['label'], '(no label)')} ({float(r['mean_error']):.3f})."
            )

    tpr = (kpi or {}).get("tpr")
    f1 = (kpi or {}).get("f1")
    fp = (kpi or {}).get("fp")
    fn = (kpi or {}).get("fn")
    if tpr is not None and tpr >= 0.8 and fp <= max(10, 0.15 * max((kpi or {}).get("tp", 0), 1)):
        badge, badge_tone = "Strong baseline", "good"
        lead = "The selected run shows a healthy operating point: recall is high and the remaining quality work is concentrated in identifiable classes and scenes."
    elif tpr is not None and tpr < 0.55:
        badge, badge_tone = "Needs attention", "risk"
        lead = "The selected run should be treated as an investigation baseline rather than a release achievement: misses are still prominent under the current slice."
    else:
        badge, badge_tone = "Mixed baseline", "mixed"
        lead = "The selected run is mixed: some classes are stable while a small set of scenes and labels explain most remaining misses."

    metric_cards = [
        _report_metric_card("TP rate", _report_pct(tpr), "Primary recall signal", "good" if (tpr or 0) >= 0.75 else "risk"),
        _report_metric_card("F1", _report_pct(f1), "Balance of precision and recall", "neutral"),
        _report_metric_card("Misses", _report_int(fn), "GT objects still not detected", "risk" if (fn or 0) > 0 else "good"),
        _report_metric_card("False positives", _report_int(fp), "Extra detections to review", "risk" if (fp or 0) > 0 else "good"),
    ]
    sections = [
        _report_signpost(
            "Manager takeaway",
            "This is the current performance baseline for the active data slice. The hotspot list shows where the next improvement effort should go.",
            [f"Stable classes: {_report_join_phrases(strong_labels)}", f"Range behavior: {_report_join_phrases(dist_phrases)}"],
        ),
        _report_signpost(
            "Main quality risk",
            "The important quality signal is where misses and false positives concentrate.",
            [f"Miss drivers: {_report_join_phrases(weak_labels)}", f"FP drivers: {_report_join_phrases(fp_labels)}"],
        ),
        _report_signpost(
            "Scenes to inspect",
            "These frames and scenes concentrate the visible failures and should be opened in the viewer for root-cause inspection.",
            top_scenes + top_frames,
        ),
    ]
    if error_phrase:
        sections.append(_report_signpost("Localization note", error_phrase, []))
    report_html = _report_shell(
        title=f"Perception Performance Report - Run {run_label}",
        subtitle=f"Single-run assessment over {scope_label}.",
        badge=badge,
        badge_tone=badge_tone,
        lead=lead,
        metric_cards=metric_cards,
        sections=sections,
        footnote=f"Scope: {scope_label}. Filters from the sidebar are applied except the max-distance cap.",
    )
    report_md = f"""# Perception Performance Report - Run {run_label}

Status: {badge}

{lead}

Key signals:
- TP rate: {_report_pct(tpr)}
- F1: {_report_pct(f1)}
- Misses: {_report_int(fn)}
- False positives: {_report_int(fp)}

Manager takeaway:
- Stable classes: {_report_join_phrases(strong_labels)}
- Range behavior: {_report_join_phrases(dist_phrases)}

Quality risks:
- Miss drivers: {_report_join_phrases(weak_labels)}
- FP drivers: {_report_join_phrases(fp_labels)}

Scenes to inspect:
- {_report_join_phrases(top_scenes)}
- {_report_join_phrases(top_frames)}
"""
    if error_phrase:
        report_md += f"\nLocalization note:\n- {error_phrase}\n"

    tables = {
        "Class metrics": df_label,
        "Scene hotspots": df_scene.head(20),
        "FN frames": df_frames,
        "Distance rates": df_dist,
    }
    if not df_err.empty:
        tables["Mean error by class"] = df_err
    return report_html, report_md, tables


def build_compare_detection_report(
    con,
    *,
    base_label: str,
    candidate_label: str,
    base_view: str,
    candidate_view: str,
    base_filter: str,
    candidate_filter: str,
    scope_label: str,
    base_kpi: Optional[Dict[str, Any]],
    candidate_kpi: Optional[Dict[str, Any]],
) -> Tuple[str, str, Dict[str, pd.DataFrame]]:
    df_base_label = _report_label_metrics(con, base_view, base_filter)
    df_candidate_label = _report_label_metrics(con, candidate_view, candidate_filter)
    df_label = df_base_label.merge(df_candidate_label, on="label", suffixes=("_base", "_candidate"), how="outer").fillna(0)
    for col in ["tpr", "fpr", "precision", "f1"]:
        df_label[f"{col}_delta"] = df_label[f"{col}_candidate"] - df_label[f"{col}_base"]

    df_diff_label = pd.DataFrame()
    df_diff_scene = pd.DataFrame()
    df_diff_frame = pd.DataFrame()
    try:
        df_diff_label = _report_diff_by_label(con, base_view, candidate_view, base_filter, candidate_filter)
        df_diff_scene = _report_diff_by_scene_or_frame(con, base_view, candidate_view, base_filter, candidate_filter, by_frame=False)
        df_diff_frame = _report_diff_by_scene_or_frame(con, base_view, candidate_view, base_filter, candidate_filter, by_frame=True)
    except Exception:
        pass
    df_degraded_objects = _report_degraded_object_details(con, base_view, candidate_view, base_filter, candidate_filter)

    df_base_dist = con.execute(sql_distance_bin_rates_from_eval_flat(base_view, base_filter, metrics="both")).df()
    df_candidate_dist = con.execute(sql_distance_bin_rates_from_eval_flat(candidate_view, candidate_filter, metrics="both")).df()
    df_label_dist_delta = pd.DataFrame()
    try:
        df_label_dist_delta = _report_label_distance_compare(con, base_view, candidate_view, base_filter, candidate_filter)
    except Exception:
        df_label_dist_delta = pd.DataFrame()

    top_tpr_gain = []
    top_tpr_loss = []
    if not df_label.empty:
        gt_total = df_label.get("gt_total_candidate", 0) + df_label.get("gt_total_base", 0)
        signal = df_label[gt_total > 0].copy()
        if not signal.empty:
            gain_signal = signal[signal["tpr_delta"] > 0.0005]
            loss_signal = signal[signal["tpr_delta"] < -0.0005]
            for _, r in gain_signal.sort_values("tpr_delta", ascending=False).head(3).iterrows():
                top_tpr_gain.append(f"{_report_nonempty_text(r['label'], '(no label)')} ({_report_pp_delta(r['tpr_delta'])})")
            for _, r in loss_signal.sort_values("tpr_delta", ascending=True).head(3).iterrows():
                top_tpr_loss.append(f"{_report_nonempty_text(r['label'], '(no label)')} ({_report_pp_delta(r['tpr_delta'])})")

    diff_gain = []
    diff_loss = []
    if not df_diff_label.empty:
        diff_gain_df = df_diff_label[df_diff_label["net_tp_delta"] > 0]
        diff_loss_df = df_diff_label[df_diff_label["degraded_cnt"] > 0]
        for _, r in diff_gain_df.sort_values(["net_tp_delta", "improved_cnt"], ascending=[False, False]).head(3).iterrows():
            diff_gain.append(
                f"{_report_nonempty_text(r['label'], '(no label)')} (net {_report_num_delta(r['net_tp_delta'])}, improved {_report_int(r['improved_cnt'])})"
            )
        for _, r in diff_loss_df.sort_values(["net_tp_delta", "degraded_cnt"], ascending=[True, False]).head(3).iterrows():
            diff_loss.append(
                f"{_report_nonempty_text(r['label'], '(no label)')} (net {_report_num_delta(r['net_tp_delta'])}, degraded {_report_int(r['degraded_cnt'])})"
            )

    scene_losses = []
    if not df_diff_scene.empty:
        degraded_scenes = df_diff_scene[df_diff_scene["degraded_cnt"] > 0]
        for _, r in degraded_scenes.sort_values(["degraded_cnt", "net_tp_delta"], ascending=[False, True]).head(3).iterrows():
            scene_losses.append(
                f"{_report_scene_ref(r)} (degraded {_report_int(r['degraded_cnt'])}, improved {_report_int(r['improved_cnt'])}, net {_report_num_delta(r['net_tp_delta'])})"
            )
    frame_losses = []
    if not df_diff_frame.empty:
        degraded_frames = df_diff_frame[df_diff_frame["degraded_cnt"] > 0]
        for _, r in degraded_frames.sort_values(["degraded_cnt", "net_tp_delta"], ascending=[False, True]).head(3).iterrows():
            frame_losses.append(
                f"{_report_frame_ref(r)} (degraded {_report_int(r['degraded_cnt'])}, net {_report_num_delta(r['net_tp_delta'])})"
            )

    dist_phrases = _report_distance_phrases(df_candidate_dist, compare_base=df_base_dist)
    label_dist_phrases = _report_label_distance_phrases(df_label_dist_delta)
    tp_delta = (candidate_kpi or {}).get("tp", 0) - (base_kpi or {}).get("tp", 0)
    fn_delta = (candidate_kpi or {}).get("fn", 0) - (base_kpi or {}).get("fn", 0)
    fp_delta = (candidate_kpi or {}).get("fp", 0) - (base_kpi or {}).get("fp", 0)
    tpr_delta = ((candidate_kpi or {}).get("tpr") - (base_kpi or {}).get("tpr")) if base_kpi and candidate_kpi and base_kpi.get("tpr") is not None and candidate_kpi.get("tpr") is not None else None
    precision_delta = ((candidate_kpi or {}).get("precision") - (base_kpi or {}).get("precision")) if base_kpi and candidate_kpi and base_kpi.get("precision") is not None and candidate_kpi.get("precision") is not None else None
    f1_delta = ((candidate_kpi or {}).get("f1") - (base_kpi or {}).get("f1")) if base_kpi and candidate_kpi and base_kpi.get("f1") is not None and candidate_kpi.get("f1") is not None else None

    total_improved = int(df_diff_label["improved_cnt"].sum()) if not df_diff_label.empty else None
    total_degraded = int(df_diff_label["degraded_cnt"].sum()) if not df_diff_label.empty else None
    net_tp = (total_improved - total_degraded) if total_improved is not None and total_degraded is not None else tp_delta
    frame_concentration = _report_frame_concentration_phrase(df_diff_frame, total_degraded)
    safety_perspective = _report_safety_perspective(df_diff_label)
    critical_phrases, df_critical_cases = _report_critical_case_phrases(df_degraded_objects)
    consecutive_phrases, df_consecutive_failures = _report_consecutive_failure_phrases(df_degraded_objects)

    if tpr_delta is not None and tpr_delta > 0.002 and fp_delta <= 0:
        verdict = "Release positive: Recallが改善し、FP増加も抑制されています。"
        badge, badge_tone = "Release positive", "good"
        lead = "Candidateは見落としを回復しつつ、False Positiveの増加も抑えられており、release候補として前向きな結果です。"
    elif tpr_delta is not None and tpr_delta > 0.002:
        verdict = "Mostly positive: Recallは改善していますが、FP増加の確認が必要です。"
        badge, badge_tone = "Positive with caveat", "mixed"
        lead = "CandidateはRecallを改善していますが、False Positiveも増加しているため、距離帯・class・scenarioごとの確認が必要です。"
    elif tpr_delta is not None and tpr_delta < -0.002:
        verdict = "Release risk: Recallが悪化しており、改善とは判断できません。"
        badge, badge_tone = "Release risk", "risk"
        lead = "Candidateはactive filter上でRecall regressionを示しています。主なrelease riskはデグレhotspotです。"
    elif fp_delta > 0:
        verdict = "Mixed: Recallは概ね維持されていますが、FPが増加しています。"
        badge, badge_tone = "Mixed", "mixed"
        lead = "Recallは概ね維持されていますが、CandidateはFalse Positiveを増やしています。純粋な改善ではなくtrade-offとして扱うべき結果です。"
    else:
        verdict = "Stable: headline KPIには大きな変化はありません。"
        badge, badge_tone = "Stable", "neutral"
        lead = "CandidateはBaselineに対して概ねstableです。ただし、詳細hotspotにODD固有の重要な変化がないか確認が必要です。"
    if (
        f1_delta is not None
        and abs(float(f1_delta)) < 0.0005
        and ((total_improved or 0) + (total_degraded or 0)) > 0
    ):
        lead = (
            "Overall F1はほぼ横ばいですが、CandidateはBaselineと同一挙動ではありません。"
            "object単位では改善とデグレの入れ替わりが発生しており、"
            "headline KPIはstableでも内部的には構造的な変化があります。"
        )

    df_err_base = _report_error_metrics(con, base_view, base_filter)
    df_err_candidate = _report_error_metrics(con, candidate_view, candidate_filter)
    localization_note = ""
    if not df_err_base.empty and not df_err_candidate.empty:
        df_err = df_err_base.merge(df_err_candidate, on="label", suffixes=("_base", "_candidate"), how="inner")
        for c in ["mean_abs_x_error", "mean_abs_y_error", "mean_abs_yaw_error"]:
            df_err[f"{c}_delta"] = df_err[f"{c}_candidate"] - df_err[f"{c}_base"]
        err_long_all = df_err.melt(id_vars=["label"], value_vars=[c for c in df_err.columns if c.endswith("_delta")], var_name="error_type", value_name="delta")
        err_long_all = err_long_all.dropna()
        err_regress = err_long_all[err_long_all["delta"] > 0].sort_values("delta", ascending=False)
        err_improve = err_long_all[err_long_all["delta"] < 0].sort_values("delta", ascending=True)
        notes = []
        if not err_improve.empty:
            r = err_improve.iloc[0]
            notes.append(
                f"best improvement is {_report_error_name(str(r['error_type']))} "
                f"on {_report_nonempty_text(r['label'], '(no label)')} ({float(r['delta']):+.3f})"
            )
        if not err_regress.empty:
            r = err_regress.iloc[0]
            notes.append(
                f"largest regression is {_report_error_name(str(r['error_type']))} "
                f"on {_report_nonempty_text(r['label'], '(no label)')} ({float(r['delta']):+.3f})"
            )
        if notes:
            localization_note = (
                "Localization changed even when detection rates are stable: "
                + "; ".join(notes)
                + "."
            )
    else:
        df_err = pd.DataFrame()

    metric_cards = [
        _report_metric_card("Recall差分", _report_pp_delta(tpr_delta), "Candidate vs baseline", _report_delta_tone(tpr_delta)),
        _report_metric_card("F1差分", _report_pp_delta(f1_delta), "全体バランス", _report_delta_tone(f1_delta)),
        _report_metric_card("改善数", _report_int(total_improved), "FN->TP object", "good" if (total_improved or 0) > 0 else "neutral"),
        _report_metric_card("デグレ数", _report_int(total_degraded), "TP->FN object", "risk" if (total_degraded or 0) > 0 else "good"),
    ]
    recommendation = _report_recommendation(
        tpr_delta=tpr_delta,
        fp_delta=fp_delta,
        total_improved=total_improved,
        total_degraded=total_degraded,
        diff_loss=diff_loss,
        dist_phrases=dist_phrases + label_dist_phrases,
    )
    executive_summary = _report_exec_summary_compare(
        candidate_label=str(candidate_label),
        tpr_delta=tpr_delta,
        precision_delta=precision_delta,
        f1_delta=f1_delta,
        df_label=df_label,
        df_candidate_dist=df_candidate_dist,
        df_critical_cases=df_critical_cases,
        recommendation=recommendation,
    )
    lead = executive_summary
    sections = [
        _report_html_panel(
            "0. ヘッダー / メタ情報",
            "評価対象と分析スコープを明記します。",
            _report_meta_table([
                ("作成日", pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d")),
                ("比較対象", f"Baseline {base_label} vs Candidate {candidate_label}"),
                ("距離スコープ", scope_label),
                ("改善 / デグレ件数", f"{_report_int(total_improved)} improved / {_report_int(total_degraded)} degraded"),
                ("分析方法", "Detection Stats parquetをDuckDB集計し、GT object単位でTP/FN変化を比較"),
            ]),
        ),
        _report_html_panel(
            "1. Executive Summary / Overall KPI",
            executive_summary,
            _report_kpi_compare_table(base_kpi, candidate_kpi),
        ),
        _report_signpost(
            "2. Gain / Loss Analysis",
            "F1だけでは見えない変化です。CandidateがどれだけFNを回復し、同時にどれだけ新しいFNを生んだかを確認します。",
            [f"Object-level net change: {_report_num_delta(net_tp)}", f"Precision差分: {_report_pp_delta(precision_delta)}"],
        ),
        _report_signpost(
            "3. Class Analysis",
            "改善・悪化をカテゴリ単位で分解し、どのobject classが全体差分を作っているかを確認します。",
            [f"改善class: {_report_join_phrases(top_tpr_gain)}", f"FN->TP集中class: {_report_join_phrases(diff_gain)}", f"デグレclass: {_report_join_phrases(top_tpr_loss)}"],
        ),
        _report_signpost(
            "4. Distance Analysis",
            "距離帯別の差分は自動運転perceptionでは特に重要です。近距離・中距離・遠距離のどこで改善/悪化したかを確認します。",
            dist_phrases,
        ),
        _report_signpost(
            "5. Label x Distance Analysis",
            "クラス別かつ距離帯別に見ることで、どのobjectがどの距離で構造的に変化したかを特定します。",
            label_dist_phrases,
        ),
        _report_signpost(
            "6. Scenario Analysis",
            "デグレが特定scenarioに集中している場合、モデル全般の弱点ではなく局所条件・ODD条件の問題である可能性が高くなります。",
            scene_losses if scene_losses else diff_loss,
        ),
        _report_signpost(
            "7. Frame Hotspot / Consecutive Failure",
            frame_concentration,
            frame_losses + consecutive_phrases,
        ),
    ]
    if localization_note:
        sections.append(_report_signpost("8. Localization Quality", localization_note, []))
    else:
        sections.append(_report_signpost("8. Localization Quality", "位置誤差列が存在しない、または明確な位置精度変化は検出されませんでした。", []))
    sections.extend([
        _report_signpost(
            "9. Safety Critical Cases",
            "20m以内 / FULL or MOST visibility / 点群20点以上のTP->FNを抽出し、安全上優先して確認すべきcaseを示します。",
            critical_phrases + safety_perspective,
        ),
        _report_signpost(
            "10. Final Recommendation",
            "現時点の評価結果に基づくrelease gate向け判断です。",
            recommendation,
        ),
    ])
    report_html = _report_shell(
        title=f"Perception Release Report - Run {candidate_label} vs {base_label}",
        subtitle=f"Baseline {base_label} と Candidate {candidate_label} の比較 / {scope_label}",
        badge=badge,
        badge_tone=badge_tone,
        lead=lead,
        metric_cards=metric_cards,
        sections=sections,
        footnote=f"Scope: {scope_label}. Sidebar filterは適用し、max-distance capのみreportでは無効化しています。",
    )
    report_md = f"""# Perception Release Report - Run {candidate_label} vs {base_label}

Status: {badge}

{executive_summary}

0. Header / Meta:
- 作成日: {pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d")}
- 比較対象: Baseline {base_label} vs Candidate {candidate_label}
- 距離スコープ: {scope_label}
- 分析方法: Detection Stats parquetをDuckDB集計し、GT object単位でTP/FN変化を比較

Executive Summary:
- {verdict}
- TP rate movement: {_report_pp_delta(tpr_delta)}
- Precision movement: {_report_pp_delta(precision_delta)}
- F1 movement: {_report_pp_delta(f1_delta)}
- Object-level net change: {_report_num_delta(net_tp)}

1. Overall KPI:
- TP: {_report_int((base_kpi or {}).get('tp'))} -> {_report_int((candidate_kpi or {}).get('tp'))} ({_report_num_delta(tp_delta)})
- FP: {_report_int((base_kpi or {}).get('fp'))} -> {_report_int((candidate_kpi or {}).get('fp'))} ({_report_num_delta(fp_delta, lower_is_better=True)})
- FN: {_report_int((base_kpi or {}).get('fn'))} -> {_report_int((candidate_kpi or {}).get('fn'))} ({_report_num_delta(fn_delta, lower_is_better=True)})

2. Gain / Loss Analysis:
- FN->TP improvements: {_report_int(total_improved)}
- TP->FN degradations: {_report_int(total_degraded)}
- Net TP delta: {_report_num_delta(net_tp)}

3. Class Analysis:
- Class-rate gains: {_report_join_phrases(top_tpr_gain)}
- Recovered-object hotspots: {_report_join_phrases(diff_gain)}
- Class-rate regressions: {_report_join_phrases(top_tpr_loss)}
- Object-level regressions: {_report_join_phrases(diff_loss)}

4. Distance Analysis:
- Distance pattern: {_report_join_phrases(dist_phrases)}

5. Per-Class + Distance:
- {_report_join_phrases(label_dist_phrases)}

6. Scenario Analysis:
- Scenes: {_report_join_phrases(scene_losses)}

7. Frame Hotspot Analysis:
- {frame_concentration}
- Frames: {_report_join_phrases(frame_losses)}
- Consecutive failures: {_report_join_phrases(consecutive_phrases)}

8. Localization Analysis:
- {localization_note or 'No meaningful localization movement detected.'}

9. Safety Critical Cases:
- Critical degraded cases: {_report_join_phrases(critical_phrases)}
- {_report_join_phrases(safety_perspective)}

10. Recommendation:
- {_report_join_phrases(recommendation)}
"""

    tables = {
        "Class rate comparison": df_label.sort_values("tpr_delta", ascending=False),
        "Object diff by class": df_diff_label,
        "Object diff by scene": df_diff_scene,
        "Object diff by frame": df_diff_frame,
        "Distance rates - baseline": df_base_dist,
        "Distance rates - candidate": df_candidate_dist,
        "Per-class distance deltas": df_label_dist_delta,
        "Critical degraded cases": df_critical_cases,
        "Consecutive degraded objects": df_consecutive_failures,
    }
    if not df_err.empty:
        tables["Mean error comparison"] = df_err
    return report_html, report_md, tables


def render_detection_report(report_html: str, report_md: str, tables: Dict[str, pd.DataFrame], *, key_prefix: str) -> None:
    st.markdown(report_html, unsafe_allow_html=True)
    st.download_button(
        "Export report Markdown",
        data=report_md.encode("utf-8"),
        file_name=f"{key_prefix}_detection_report.md",
        mime="text/markdown",
        key=f"{key_prefix}_download_report_md",
    )
    with st.expander("Supporting analysis tables"):
        for name, df in tables.items():
            st.markdown(f"**{name}**")
            if df is None or df.empty:
                st.caption("No rows.")
            else:
                st.dataframe(df.head(100), width="stretch", hide_index=True)


# Topic names that are semantically equivalent across different versions.
# e.g. "perception.object_recognition.objects" (older) and
#      "perception.object_recognition.tracking.objects" (newer) refer to the same data.
_TOPIC_EQUIVALENCE_GROUPS: List[List[str]] = [
    [
        "perception.object_recognition.objects",
        "perception.object_recognition.tracking.objects",
        "perception.object_recognition.detection.bevfusion.objects",
    ],
]


def _equivalent_topic_order(topic: str) -> List[str]:
    """Return equivalent topic names in a stable preference order."""
    for group in _TOPIC_EQUIVALENCE_GROUPS:
        if topic in group:
            return [topic] + [candidate for candidate in group if candidate != topic]
    return [topic]


def _map_topic_to_run(selected_topic: str, run_topics: set) -> str:
    """Map a selected topic to the equivalent topic name used by a specific run.

    If the run has an equivalent topic (e.g. selected="perception.object_recognition.objects"
    but run has "perception.object_recognition.tracking.objects"), return the run's version.
    Otherwise return the selected topic as-is (which may not exist in the run).
    """
    if selected_topic == "__all__":
        return "__all__"
    for candidate in _equivalent_topic_order(selected_topic):
        if candidate in run_topics:
            return candidate
    return selected_topic


def build_filter_clause(filters: dict,*, enable_dist_h: bool = True) -> str:
    """Build WHERE clause from filters.

    For label / suites / visibility: ``None`` means this dimension is inactive (e.g. no suite column).
    An empty list ``[]`` means no restriction on that dimension (same as all options selected).
    Using ``if filters.get('label')`` would treat ``[]`` as falsy and accidentally drop the filter,
    causing full scans (very slow on large Parquet).
    """
    conditions = [DETECTION_STATS_INITIAL_FRAME_FILTER]
    
    topic_val = filters.get('topic_name')
    if topic_val and topic_val != '__all__':
        if isinstance(topic_val, list):
            if len(topic_val) == 1:
                conditions.append(f"topic_name = '{topic_val[0]}'")
            elif len(topic_val) > 1:
                topics_escaped = [str(t).replace("'", "''") for t in topic_val]
                topics_str = "', '".join(topics_escaped)
                conditions.append(f"topic_name IN ('{topics_str}')")
        else:
            conditions.append(f"topic_name = '{topic_val}'")
    
    lbl = filters.get('label')
    if lbl is not None:
        if isinstance(lbl, list):
            if len(lbl) > 0:
                labels_escaped = [str(l).replace("'", "''") for l in lbl]
                labels_str = "', '".join(labels_escaped)
                conditions.append(f"label IN ('{labels_str}')")
        elif not isinstance(lbl, list) and lbl != '__all__':
            label_escaped = str(lbl).replace("'", "''")
            conditions.append(f"label = '{label_escaped}'")
    
    su = filters.get('suites')
    if su is not None:
        if isinstance(su, list):
            if len(su) > 0:
                suite_escaped = [str(s).replace("'", "''") for s in su]
                suite_str = "', '".join(suite_escaped)
                conditions.append(f"COALESCE(CAST(suite_name AS VARCHAR), '') IN ('{suite_str}')")
        elif not isinstance(su, list) and su != '__all__':
            s_escaped = str(su).replace("'", "''")
            conditions.append(f"COALESCE(CAST(suite_name AS VARCHAR), '') = '{s_escaped}'")
    
    vis = filters.get('visibility')
    if vis is not None:
        if isinstance(vis, list):
            if len(vis) > 0:
                vis_escaped = [str(v).replace("'", "''") for v in vis]
                vis_str = "', '".join(vis_escaped)
                conditions.append(f"COALESCE(visibility, 'not available') IN ('{vis_str}')")
        elif not isinstance(vis, list):
            vis_escaped = str(vis).replace("'", "''")
            conditions.append(f"COALESCE(visibility, 'not available') = '{vis_escaped}'")
    
    if enable_dist_h and filters.get('max_eval_range'):
        conditions.append(f"dist_h < {filters['max_eval_range']}")
    
    return " AND ".join(conditions) if conditions else "1=1"


# =============================
# Parquet files from run path(s)
# =============================
parquet_lists = [list_parquets_in_run(r["path"]) for r in runs]
for i, (r, pl) in enumerate(zip(runs, parquet_lists)):
    if not pl:
        label = run_labels_list[i] if i < len(run_labels_list) else str(i)
        st.error(f"No parquet files found in run ({label}): {path_display(r['path'])}. Add a .parquet file or generate one from the Download page.")
        st.stop()

inject_detection_stats_styles()

# =============================
# Loaded Runs (from Overview) + hero
# =============================
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
    kicker="Object detection",
    title="Detection evaluation dashboard",
    description=(
        "Parquet-driven analytics: filters, hierarchical views, scenario breakdowns, "
        "and multi-run compare when you load several runs from Overview. "
        f"Frames 0-{DETECTION_STATS_SKIP_INITIAL_FRAMES - 1} are excluded from statistics."
    ),
    mode=mode,
)

# =============================
# Sidebar - Filters
# =============================
# File selection per run
target_files = []
with st.sidebar:
    st.markdown("##### Filters / inputs")
    st.caption("Pick parquet per run, then slice by suite, scenario, labels, and topics.")
    for i, (pl, lbl) in enumerate(zip(parquet_lists, run_labels_list)):
        if len(pl) == 1:
            target_files.append(pl[0])
        else:
            file_key = f"target_file_{lbl}"
            migrate_old_run_index_parquet_default(file_key, pl, i)
            tf = st.selectbox(
                f"Run ({lbl}) File",
                pl,
                format_func=lambda p: os.path.basename(p),
                index=default_parquet_index(pl),
                key=file_key
            )
            target_files.append(tf)

target_file = target_files[0] if target_files else None
con = get_duckdb_connection()
fp = _parquet_selection_fingerprint(target_files)
cache_hit = st.session_state.get("_ds_parquet_fp") == fp and "_ds_filter_opts" in st.session_state
selected_run_paths = [Path(r["path"]) for r in runs]
cached_target_files = list(target_files)
cache_rebuild_notes: List[str] = []

ds_dlog(
    "duckdb setup: fp=%s cache_hit=%s n_runs=%s target_files=%s",
    fp,
    cache_hit,
    len(target_files),
    [os.path.basename(p) for p in target_files],
)
ds_debug_log_memory("before_duckdb_validate_views")

with ds_dtimer("duckdb_validate_views_list_values_or_cache", st.session_state):
    if not cache_hit:
        for i, (path, lbl) in enumerate(zip(target_files, run_labels_list)):
            ok, msg = validate_detection_stats_parquet(con, path)
            if not ok:
                st.sidebar.error(f"**Run ({lbl}) file** cannot be used here: {msg}")
                st.stop()

        # Automatically materialize eval_flat cache parquet(s) under each run.
        for i, (path, run_path, lbl) in enumerate(zip(target_files, selected_run_paths, run_labels_list)):
            cached_path, rebuilt = _ensure_detection_stats_eval_flat_cache(
                con,
                run_path=run_path,
                source_path=path,
            )
            cached_target_files[i] = cached_path
            if rebuilt:
                cache_rebuild_notes.append(f"Run {lbl}: refreshed detection cache from {os.path.basename(path)}")

        # One eval_flat view per run. (TPR/FPR layered views are not created: Distance queries inline the same
        # stats from eval_flat — nested view + aggregate can segfault DuckDB, exit 139.)
        try:
            for i, path in enumerate(cached_target_files):
                v_flat = "view_eval_flat" if i == 0 else f"view_eval_flat_{i}"
                create_view_eval_flat(con, path, v_flat)
        except Exception as e:
            st.error(f"Error creating views: {e}")
            st.stop()

        # Collect filter options from ALL runs (not just the first).
        # Suite names may differ between runs (e.g. one has UUID suffix, the other doesn't),
        # so we merge and deduplicate them to offer a complete dropdown.
        all_topics: set = set()
        all_labels: set = set()
        all_suite_options: set = set()
        all_vis_options: set = set()
        per_run_suite_lookup: List[Dict[str, List[str]]] = []  # maps each run: display_suite_name -> [actual_suite_names]
        per_run_topics: List[set] = []  # topic_names per run, for cross-run topic mismatch detection
        for i, path in enumerate(cached_target_files):
            run_topics = list_values(con, path, "topic_name")
            all_topics.update(run_topics)
            per_run_topics.append(set(run_topics))
            run_labels = list_values(con, path, "label")
            all_labels.update(run_labels)
            try:
                run_suites = list_values(con, path, "COALESCE(CAST(suite_name AS VARCHAR), '')")
            except Exception:
                run_suites = []
            all_suite_options.update(run_suites)
            try:
                run_vis = list_values(con, path, "COALESCE(CAST(visibility AS VARCHAR), 'not available') AS visibility")
            except Exception:
                run_vis = []
            all_vis_options.update(run_vis)
            # Build a lookup: for each suite in this run, map display_name -> [actual suite names]
            # In compare mode, suites from different runs may differ (e.g. UUID suffix).
            # We store the mapping so we can later resolve the filter per run.
            suite_map: Dict[str, List[str]] = {}
            for s in run_suites:
                s_str = str(s)
                suite_map.setdefault(s_str, []).append(s_str)
            per_run_suite_lookup.append(suite_map)

        topics = sorted(all_topics)
        labels = sorted(all_labels)
        suite_options = sorted(all_suite_options)
        vis_options = sorted(all_vis_options)
        schema = schema_flags(con, cached_target_files[0])
        st.session_state["_ds_parquet_fp"] = fp
        st.session_state["_ds_filter_opts"] = {
            "topics": topics,
            "labels": labels,
            "suite_options": suite_options,
            "vis_options": vis_options,
            "schema": schema,
            "cached_target_files": list(cached_target_files),
            "cache_rebuild_notes": list(cache_rebuild_notes),
            "per_run_suite_lookup": per_run_suite_lookup,
            "per_run_topics": [list(t) for t in per_run_topics],
        }
    else:
        opts = st.session_state["_ds_filter_opts"]
        topics = opts["topics"]
        labels = opts["labels"]
        suite_options = opts["suite_options"]
        vis_options = opts["vis_options"]
        schema = opts["schema"]
        cached_target_files = opts.get("cached_target_files", list(target_files))
        cache_rebuild_notes = opts.get("cache_rebuild_notes", [])
        per_run_suite_lookup = opts.get("per_run_suite_lookup", [])
        per_run_topics = [set(t) for t in opts.get("per_run_topics", [])]
        for i, path in enumerate(cached_target_files):
            v_flat = "view_eval_flat" if i == 0 else f"view_eval_flat_{i}"
            create_view_eval_flat(con, path, v_flat)

ds_debug_log_memory("after_duckdb_validate_views")

with st.sidebar:
    # Smart default: prefer the primary perception topic over __all__
    _PRIMARY_TOPICS = [
        "perception.object_recognition.objects",
        "perception.object_recognition.tracking.objects",
        "perception.object_recognition.detection.bevfusion.objects",
    ]
    _default_topic = "__all__"
    if topics:
        for pt in _PRIMARY_TOPICS:
            if pt in topics:
                _default_topic = pt
                break
    # Use session_state key with index so default only applies on first load
    if "ds_topic_name" not in st.session_state:
        st.session_state["ds_topic_name"] = _default_topic
    topic_name = st.selectbox("Topic Name", ["__all__"] + topics, key="ds_topic_name") if topics else "__all__"

    # In compare mode, map topic to each run's equivalent topic name.
    # e.g. if user selects "perception.object_recognition.objects" but run B has
    # "perception.object_recognition.tracking.objects", we map to the latter.
    _topic_per_run: List[str] = []
    if not single_mode and topic_name != "__all__" and per_run_topics:
        missing_runs = []
        for i, run_topics in enumerate(per_run_topics):
            mapped = _map_topic_to_run(topic_name, run_topics)
            _topic_per_run.append(mapped)
            lbl = run_labels_list[i] if i < len(run_labels_list) else f"Run {i}"
            if mapped not in run_topics:
                missing_runs.append((lbl, mapped))
        if missing_runs:
            missing_str = ", ".join(f"{lbl} (mapped to '{m}')" for l, m in missing_runs)
            st.warning(
                f"⚠️ Topic **'{topic_name}'** could not be found in: {missing_str}. "
                f"Those runs will have no data in comparisons. "
                f"Consider selecting **__all__** instead.",
            )
    else:
        _topic_per_run = [topic_name] * len(runs)
    # Widget keys: avoid generic "labels"/"visibility" (session_state collisions, ambiguous with run_labels).
    if "ds_filter_class_labels" not in st.session_state and "labels" in st.session_state:
        st.session_state["ds_filter_class_labels"] = st.session_state["labels"]
    if "ds_filter_visibility" not in st.session_state and "visibility" in st.session_state:
        st.session_state["ds_filter_visibility"] = st.session_state["visibility"]
    if labels:
        if "ds_filter_class_labels" not in st.session_state:
            st.session_state["ds_filter_class_labels"] = list(labels)
        selected_labels = st.multiselect(
            "Label(s)",
            labels,
            key="ds_filter_class_labels",
        )
    else:
        selected_labels = []
    if suite_options:
        if "suites" not in st.session_state:
            st.session_state["suites"] = list(suite_options)
        selected_suites = st.multiselect(
            "Suites",
            suite_options,
            key="suites",
            help="Filter by suite(s). Default: all included.",
        )
    else:
        selected_suites = []
    if vis_options:
        if "ds_filter_visibility" not in st.session_state:
            st.session_state["ds_filter_visibility"] = list(vis_options)
        selected_visibility = st.multiselect(
            "Visibility",
            vis_options,
            key="ds_filter_visibility",
        )
    else:
        selected_visibility = []
    max_eval_range = st.selectbox("Max Evaluation Range [m]", [50, 80, 100, 120, 150], index=4, key="max_eval_range")

# Build filters (same values for all runs). None = dimension unused (no suite/visibility column in UI).
# When comparing runs with different suite name formats (e.g. one has UUID suffix, the other
# doesn't), we map selected suites to each run's actual suite names via prefix matching.
# Topic names are also mapped per-run via _topic_per_run (equivalence mapping).
filters_base = {
    'topic_name': topic_name,
    'label': selected_labels,
    'suites': selected_suites if suite_options else None,
    'visibility': selected_visibility if vis_options else None,
    'max_eval_range': max_eval_range
}
filters_list = [dict(filters_base) for _ in range(len(runs))]

# Apply per-run topic mapping (handles equivalent topic names across different versions).
# Keep the comparison to one output topic per run. Some releases contain multiple object
# recognition topics for the same dataset; combining them would mix different outputs on
# one side and distort TP/FP deltas.
if topic_name != "__all__" and per_run_topics:
    for run_idx in range(len(runs)):
        if run_idx < len(per_run_topics):
            filters_list[run_idx]['topic_name'] = _map_topic_to_run(topic_name, per_run_topics[run_idx])

# Map selected suites to each run's actual suite names (prefix matching for cross-run compare)
if selected_suites and len(runs) > 1 and per_run_suite_lookup:
    for run_idx in range(len(runs)):
        if run_idx < len(per_run_suite_lookup) and per_run_suite_lookup[run_idx]:
            run_suites_mapped = set()
            for sel_suite in selected_suites:
                sel_str = str(sel_suite)
                # Direct match first
                if sel_str in per_run_suite_lookup[run_idx]:
                    run_suites_mapped.add(sel_str)
                else:
                    # Prefix match: selected suite may be a prefix of this run's suite names
                    # e.g. sel="FullPerformance_V1_Fujiyoshida_PDD" matches
                    #      "FullPerformance_V1_Fujiyoshida_PDD_5df54148-..."
                    for actual_suite in per_run_suite_lookup[run_idx]:
                        if actual_suite.startswith(sel_str):
                            run_suites_mapped.add(actual_suite)
            filters_list[run_idx]['suites'] = sorted(run_suites_mapped) if run_suites_mapped else selected_suites

try:
    _fcl_preview = build_filter_clause(filters_base)
except Exception as _e_fcl:
    _fcl_preview = f"<build_filter_clause error: {_e_fcl}>"
ds_dlog("filters_base keys=%s filter_clause_preview=%s", list(filters_base.keys()), _fcl_preview[:800])

# Banner while the rest of the page (queries + charts) streams in — cleared in finally (even on errors).
_ds_loading_banner = st.empty()
_cache_note = " ".join(cache_rebuild_notes)
_ds_loading_banner.markdown(_banner_html_with_note(_cache_note), unsafe_allow_html=True)
try:
    ds_dlog("main_content_try_enter")
    ds_debug_log_memory("main_content_start")

    # =============================
    # Main Content
    # =============================
    
    # -----------------------------
    # KPI strip (GT, TP, FP, FN, Recall / TP rate, FP rate, Precision, F1)
    # -----------------------------
    def _flat_view(i: int) -> str:
        return "view_eval_flat" if i == 0 else f"view_eval_flat_{i}"
    
    def _kpi_row_for_view(con, view: str, filter_clause: str):
        """Return global KPI values within the active filters."""
        q = f"""
        SELECT
            COUNT(*) FILTER (WHERE source = 'GT' AND status = 'TP') AS tp_gt,
            COUNT(*) FILTER (WHERE source = 'GT' AND status = 'FN') AS fn,
            COUNT(*) FILTER (WHERE source = 'EST' AND status = 'TP') AS tp_est,
            COUNT(*) FILTER (WHERE source = 'EST' AND status = 'FP') AS fp
        FROM {view}
        WHERE {filter_clause}
        """
        row = con.execute(q).fetchone()
        if not row:
            return None
        tp_gt, fn, tp_est, fp = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        gt_total = tp_gt + fn
        est_total = tp_est + fp
        tpr = (tp_gt / gt_total) if gt_total > 0 else None
        fpr = (fp / est_total) if est_total > 0 else None
        precision = (tp_est / est_total) if est_total > 0 else None
        recall = tpr
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        return {
            "gt": gt_total, "tp": tp_gt, "fp": fp, "fn": fn,
            "tpr": tpr, "fpr": fpr, "precision": precision, "recall": recall, "f1": f1,
        }
    
    # =============================
    # Panel 1: t4dataset Summary
    # =============================
    ds_dlog("section: Panel1_Summary_start")
    st.markdown(section_header_html("Summary", "Within selected filters and max evaluation range."), unsafe_allow_html=True)
    if single_mode:
        with ds_spot_loading("Summary · KPI metrics"):
            fc = build_filter_clause(filters_base)
            kpi = _kpi_row_for_view(con, "view_eval_flat", fc)
        inject_detection_stats_kpi_styles()
        if kpi:
            html = '<div class="kpi-wrap">' + render_kpi_card("Metrics (within filters & max range)", kpi) + "</div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.caption("No KPI data.")
    else:
        with ds_spot_loading("Summary · KPI metrics"):
            kpis = []
            for i in range(len(runs)):
                fc = build_filter_clause(filters_list[i])
                kpi = _kpi_row_for_view(con, _flat_view(i), fc)
                kpis.append((run_labels_list[i], kpi))
        inject_detection_stats_kpi_styles()
        baseline = kpis[0][1] if kpis else None
        cards_html_parts = []
        for lbl, kpi in kpis:
            deltas = None
            if baseline and kpi and lbl != run_labels_list[0]:
                deltas = {
                    "gt": kpi["gt"] - baseline["gt"],
                    "tp": kpi["tp"] - baseline["tp"],
                    "fp": kpi["fp"] - baseline["fp"],
                    "fn": kpi["fn"] - baseline["fn"],
                    "tpr": (kpi["tpr"] - baseline["tpr"]) if (kpi.get("tpr") is not None and baseline.get("tpr") is not None) else None,
                    "fpr": (kpi["fpr"] - baseline["fpr"]) if (kpi.get("fpr") is not None and baseline.get("fpr") is not None) else None,
                    "precision": (kpi["precision"] - baseline["precision"]) if (kpi.get("precision") is not None and baseline.get("precision") is not None) else None,
                    "recall": (kpi["recall"] - baseline["recall"]) if (kpi.get("recall") is not None and baseline.get("recall") is not None) else None,
                    "f1": (kpi["f1"] - baseline["f1"]) if (kpi.get("f1") is not None and baseline.get("f1") is not None) else None,
                }
            cards_html_parts.append(render_kpi_card(f"Run {lbl}", kpi or {}, f"kpi-run-{lbl}", deltas=deltas))
        st.markdown('<div class="kpi-wrap">' + "".join(cards_html_parts) + "</div>", unsafe_allow_html=True)
        # --- KPI Comparison Analysis ---
        if not single_mode and baseline and len(kpis) >= 2:
            for lbl, kpi in kpis:
                if lbl != run_labels_list[0] and kpi:
                    analysis_html = _analyze_kpi_comparison(baseline, kpi, candidate_label=str(lbl))
                    st.markdown(analysis_html, unsafe_allow_html=True)
                    break  # analyze first non-baseline run

    if st.checkbox("Debug: Inspect Parquet (All Runs)" if not single_mode else "Debug: Inspect Parquet"):
        cols_used = st.columns(len(target_files))
        file_labels = [(f"Run ({run_labels_list[i]}) File", target_files[i]) for i in range(len(target_files))]
        schema_results = []
        for col, (label, file_path) in zip(cols_used, file_labels):
            with col:
                st.markdown(f"### {label}")
                # Schema
                schema_df = con.execute("""
                    DESCRIBE SELECT * FROM read_parquet(?)
                """, [file_path]).df()
                schema_results.append((label, schema_df))
                st.write("**Schema (Column Names, Types)**")
                st.markdown("Shows the schema (column names and their DuckDB/Parquet data types) of the selected Parquet file. Useful to check data structure and types as interpreted by DuckDB.")
                st.dataframe(schema_df, width='stretch', hide_index=True)
    
                # Preview rows
                row_options = [10, 20, 50, 100, 200, "All"]
                preview_key = f"preview_row_limit_{label.replace(' ', '_').lower()}"
                row_choice = st.selectbox(f"Preview rows to show ({label})", row_options, index=1, key=preview_key)
                if row_choice == "All":
                    limit_clause = ""
                else:
                    limit_clause = f"LIMIT {row_choice}"
                preview_df = con.execute(f"""
                    SELECT *
                    FROM read_parquet(?)
                    {limit_clause}
                """, [file_path]).df()
                st.write(f"**Preview (First {row_choice} rows)**")
                st.markdown(f"Shows the first {row_choice} preview rows from the Parquet file. Use this preview to examine example data contents and check that your file is as expected.")
                st.dataframe(preview_df, width='stretch', hide_index=True)
    
                # Stats
                stats_df = con.execute("""
                    SELECT
                        COUNT(*) AS total_rows,
                        COUNT(t4dataset_id) AS non_null_ids,
                        COUNT(DISTINCT t4dataset_id) AS distinct_ids
                    FROM read_parquet(?)
                """, [file_path]).df()
                st.write("**Stats (Row Count, t4dataset_id non-null count, Distinct t4dataset_id count)**")
                st.markdown("""
                - `total_rows`: Total rows in the file  
                - `non_null_ids`: Rows where t4dataset_id is not null  
                - `distinct_ids`: Unique t4dataset_id values
    
                This helps rapidly assess the completeness and distribution of the key ID field.
                """)
                st.dataframe(stats_df, width='stretch', hide_index=True)
    
        # --- Show info about schema differences (compare mode only) ---
        if not single_mode and len(schema_results) >= 2:
            with st.expander("⚖️ Difference between schemas", expanded=(len(schema_results) == 2)):
                if len(schema_results) == 2:
                    label1, df1 = schema_results[0]
                    label2, df2 = schema_results[1]
                    names1 = set(df1["column_name"])
                    names2 = set(df2["column_name"])
                    added, removed = names2 - names1, names1 - names2
                    common = names1 & names2
                    types1 = {row["column_name"]: row["column_type"] for _, row in df1.iterrows()}
                    types2 = {row["column_name"]: row["column_type"] for _, row in df2.iterrows()}
                    dtype_changes = [(c, types1.get(c), types2.get(c)) for c in sorted(common) if types1.get(c) != types2.get(c)]
                    if not (added or removed or dtype_changes):
                        st.success("✅ The schemas are identical (column names and types match exactly).")
                    else:
                        if added:
                            st.error(f"Columns only in `{label2}`: {', '.join(sorted(added))}")
                        if removed:
                            st.error(f"Columns only in `{label1}`: {', '.join(sorted(removed))}")
                        if dtype_changes:
                            st.warning("Columns with different types:")
                            st.dataframe(pd.DataFrame(dtype_changes, columns=["Column", f"Type in {label1}", f"Type in {label2}"]), width='stretch', hide_index=True)
                else:
                    st.info(f"{len(schema_results)} runs loaded. Compare schemas per run in the columns above.")
    
    
    
    ds_dlog("section: Dataset_summary_status_distribution_try")
    try:
        with ds_spot_loading("Dataset summary & status distribution"):
            if single_mode:
                query_base = f"""
                SELECT COUNT(DISTINCT t4dataset_id) AS id_num, '{os.path.basename(target_file)}' AS series
                FROM view_eval_flat
                WHERE {DETECTION_STATS_INITIAL_FRAME_FILTER}
                """
                df_summary = con.execute(query_base).df()
                query_status = f"""
                SELECT label, status, COUNT(*) AS num
                FROM view_eval_flat
                WHERE {DETECTION_STATS_INITIAL_FRAME_FILTER}
                GROUP BY label, status
                ORDER BY label, status
                """
                df_status = con.execute(query_status).df()
            else:
                parts = [
                    (
                        f"SELECT COUNT(DISTINCT t4dataset_id) AS id_num, '{run_labels_list[i]}' AS series "
                        f"FROM {_flat_view(i)} WHERE {DETECTION_STATS_INITIAL_FRAME_FILTER}"
                    )
                    for i in range(len(runs))
                ]
                query_base = " UNION ALL ".join(parts)
                df_summary = con.execute(query_base).df()
                parts_status = [
                    (
                        f"SELECT '{run_labels_list[i]}' AS dataset, label, status, COUNT(*) AS num "
                        f"FROM {_flat_view(i)} WHERE {DETECTION_STATS_INITIAL_FRAME_FILTER} GROUP BY label, status"
                    )
                    for i in range(len(runs))
                ]
                query_status = " UNION ALL ".join(parts_status) + " ORDER BY dataset, label, status"
                df_status = con.execute(query_status).df()
    
        if single_mode:
            if not df_status.empty:
                if st.checkbox("Debug: Inspect Status Count (All Runs)" if not single_mode else "Debug: Inspect Status Count"):
                    df_status_wide = df_status.pivot_table(index='label', columns='status', values='num', fill_value=0).reset_index()
                    st.download_button("Download status count (CSV)", data=df_status_wide.to_csv(index=False).encode("utf-8"), file_name="detection_status_count.csv", mime="text/csv", key="dl_status_count")
                    st.dataframe(df_status_wide, width='stretch', hide_index=True)
                status_viz = st.radio(
                    "Status chart style",
                    options=["Stacked bar (counts)", "Treemap", "100% stacked (proportions)", "Spider chart (TP, FP & FN)"],
                    index=0,
                    horizontal=True,
                    key="status_dist_viz",
                )
                n_labels = df_status["label"].nunique()
                use_horizontal = n_labels > 6
                if status_viz == "Stacked bar (counts)":
                    if use_horizontal:
                        fig2 = px.bar(
                            df_status,
                            y="label",
                            x="num",
                            color="status",
                            barmode="stack",
                            title="Status Distribution per Label",
                            labels={"num": "Count", "label": "Label", "status": "Status"},
                            color_discrete_map=STATUS_COLORS,
                            orientation="h",
                        )
                    else:
                        fig2 = px.bar(
                            df_status,
                            x="label",
                            y="num",
                            color="status",
                            barmode="stack",
                            title="Status Distribution per Label",
                            labels={"num": "Count", "label": "Label", "status": "Status"},
                            color_discrete_map=STATUS_COLORS,
                        )
                    apply_chart_theme(fig2)
                    st.plotly_chart(fig2, width='stretch')
                elif status_viz == "Treemap":
                    fig2 = px.treemap(
                        df_status,
                        path=["label", "status"],
                        values="num",
                        color="status",
                        color_discrete_map=STATUS_COLORS,
                        title="Status Distribution per Label (area = count)",
                    )
                    fig2.update_traces(
                        textinfo="label+value+percent parent",
                        hovertemplate="%{label}<br>Count: %{value}<extra></extra>",
                    )
                    apply_chart_theme(fig2, height=420)
                    st.plotly_chart(fig2, width='stretch')
                elif status_viz == "Spider chart (TP, FP & FN)":
                    wide = df_status.pivot_table(index="label", columns="status", values="num", fill_value=0)
                    cats = sorted(wide.index.astype(str).unique())
                    if len(cats) > 16:
                        st.caption("Spider charts work best with ≤16 labels; many classes may look crowded.")
                    run_single = [os.path.basename(target_file) if target_file else "Run"]
                    rcols = st.columns(3)
                    for col_i, st_name in enumerate(["TP", "FP", "FN"]):
                        vals = wide[st_name] if st_name in wide.columns else pd.Series(0, index=wide.index)
                        df_m = pd.DataFrame({"label": wide.index.astype(str), "count": vals.values})
                        df_m["run"] = run_single[0]
                        fig_r = _count_spider_compare(
                            df_m,
                            cats,
                            f"{st_name} count per label",
                            run_single,
                            f"{st_name} count",
                        )
                        with rcols[col_i]:
                            st.plotly_chart(fig_r, width='stretch')
                else:
                    # 100% stacked: proportion per label
                    wide = df_status.pivot_table(index="label", columns="status", values="num", fill_value=0)
                    wide_pct = wide.div(wide.sum(axis=1), axis=0)
                    df_pct = wide_pct.reset_index().melt(id_vars="label", var_name="status", value_name="pct")
                    df_pct = df_pct[df_pct["pct"] > 0]
                    if not df_pct.empty:
                        if use_horizontal:
                            fig2 = px.bar(
                                df_pct,
                                y="label",
                                x="pct",
                                color="status",
                                barmode="stack",
                                title="Status proportion per Label (100% stacked)",
                                labels={"pct": "Proportion", "label": "Label", "status": "Status"},
                                color_discrete_map=STATUS_COLORS,
                                orientation="h",
                            )
                        else:
                            fig2 = px.bar(
                                df_pct,
                                x="label",
                                y="pct",
                                color="status",
                                barmode="stack",
                                title="Status proportion per Label (100% stacked)",
                                labels={"pct": "Proportion", "label": "Label", "status": "Status"},
                                color_discrete_map=STATUS_COLORS,
                            )
                        apply_chart_theme(fig2)
                        if use_horizontal:
                            fig2.update_layout(xaxis_tickformat=".0%", xaxis_range=[0, 1])
                        else:
                            fig2.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
                        st.plotly_chart(fig2, width='stretch')
                    else:
                        st.info("No data for proportions.")
            else:
                st.info("No status count data available")
        else:
            if not df_status.empty:
                if st.checkbox("Debug: Inspect Status Count (All Runs)" if not single_mode else "Debug: Inspect Status Count"):
                    df_status_wide = df_status.pivot_table(index='label', columns=['dataset', 'status'], values='num', fill_value=0)
                    df_status_wide.columns = [f"{col[0]} {col[1]}" for col in df_status_wide.columns]
                    df_status_wide = df_status_wide.reset_index()
                    st.dataframe(df_status_wide, width='stretch', hide_index=True)
                status_viz = st.radio(
                    "Status chart style",
                    options=["Stacked bar (counts)", "Treemap", "100% stacked (proportions)", "Spider chart (TP, FP & FN)"],
                    index=0,
                    horizontal=True,
                    key="status_dist_viz_compare",
                )
                if status_viz == "Stacked bar (counts)":
                    fig2 = px.bar(
                        df_status,
                        x="label",
                        y="num",
                        color="status",
                        barmode="stack",
                        facet_col="dataset",
                        title="Status Distribution per Label (by Run)",
                        category_orders={"dataset": run_labels_list},
                        labels={"num": "Count", "label": "Label", "status": "Status"},
                        color_discrete_map=STATUS_COLORS,
                    )
                    apply_chart_theme(fig2)
                    st.plotly_chart(fig2, width='stretch')
                elif status_viz == "Spider chart (TP, FP & FN)":
                    # Same counts as stacked bar: one spider per status (TP / FP / FN), axes = labels, r = count
                    status_wide = df_status.pivot_table(
                        index=["dataset", "label"], columns="status", values="num", fill_value=0
                    ).reset_index()
                    cats = sorted(df_status["label"].astype(str).unique())
                    if len(cats) > 16:
                        st.caption("Spider charts work best with ≤16 labels; many classes may look crowded.")
                    rcols = st.columns(3)
                    for col_i, st_name in enumerate(["TP", "FP", "FN"]):
                        col_data = (
                            status_wide[st_name]
                            if st_name in status_wide.columns
                            else pd.Series(0, index=status_wide.index)
                        )
                        df_m = pd.DataFrame(
                            {
                                "run": status_wide["dataset"].astype(str),
                                "label": status_wide["label"].astype(str),
                                "count": col_data.values,
                            }
                        )
                        fig_r = _count_spider_compare(
                            df_m,
                            cats,
                            f"{st_name} count per label (by run)",
                            run_labels_list,
                            f"{st_name} count",
                        )
                        with rcols[col_i]:
                            st.plotly_chart(fig_r, width='stretch')
                elif status_viz == "Treemap":
                    n_runs = len(run_labels_list)
                    cols = st.columns(min(n_runs, 3))
                    for idx, lbl in enumerate(run_labels_list):
                        df_r = df_status[df_status["dataset"] == lbl]
                        if not df_r.empty:
                            fig_t = px.treemap(
                                df_r,
                                path=["label", "status"],
                                values="num",
                                color="status",
                                color_discrete_map=STATUS_COLORS,
                                title=f"{lbl}",
                            )
                            fig_t.update_traces(
                                textinfo="label+value+percent parent",
                                hovertemplate="%{label}<br>Count: %{value}<extra></extra>",
                            )
                            apply_chart_theme(fig_t, height=360)
                            with cols[idx % len(cols)]:
                                st.plotly_chart(fig_t, width='stretch')
                else:
                    # 100% stacked per run (facet)
                    df_pct_list = []
                    for lbl in run_labels_list:
                        df_r = df_status[df_status["dataset"] == lbl]
                        wide = df_r.pivot_table(index="label", columns="status", values="num", fill_value=0)
                        if wide.empty:
                            continue
                        wide_pct = wide.div(wide.sum(axis=1), axis=0)
                        wide_pct["dataset"] = lbl
                        wide_pct = wide_pct.reset_index()
                        df_pct_list.append(wide_pct)
                    if df_pct_list:
                        wide_all = pd.concat(df_pct_list, ignore_index=True)
                        df_pct_melt = wide_all.melt(
                            id_vars=["label", "dataset"],
                            value_vars=[c for c in wide_all.columns if c not in ("label", "dataset")],
                            var_name="status",
                            value_name="pct",
                        )
                        df_pct_melt = df_pct_melt[df_pct_melt["pct"] > 0]
                        if not df_pct_melt.empty:
                            fig2 = px.bar(
                                df_pct_melt,
                                x="label",
                                y="pct",
                                color="status",
                                barmode="stack",
                                facet_col="dataset",
                                category_orders={"dataset": run_labels_list},
                                title="Status proportion per Label (100% stacked, by Run)",
                                labels={"pct": "Proportion", "label": "Label", "status": "Status"},
                                color_discrete_map=STATUS_COLORS,
                            )
                            apply_chart_theme(fig2)
                            fig2.update_layout(
                                yaxis_tickformat=".0%",
                                yaxis_range=[0, 1],
                            )
                            for ann in fig2.layout.annotations:
                                ann.text = ann.text.split("=")[-1]
                            st.plotly_chart(fig2, width='stretch')
                        else:
                            st.info("No data for proportions.")
                    else:
                        st.info("No data for proportions.")
            else:
                st.info("No status count data available")
    
    except Exception as e:
        st.error(f"Error in summary: {e}")
    
    
    
    def _distance_bin_order_and_label(bin_str: str) -> Tuple[int, str]:
        """Parse distance_bin e.g. '[0,10)' -> (0, '0–10 m'). Used for sorting and axis labels."""
        import re
        s = str(bin_str).strip()
        m = re.match(r"\[(\d+)\s*,\s*(\d+)\)", s)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            return (lo, f"{lo}–{hi} m")
        m = re.match(r"\[(\d+)\s*,\s*inf\)", s, re.I)
        if m:
            return (int(m.group(1)), f"{m.group(1)}+ m")
        return (0, s)

    def _distance_summary_jp_line(line: str) -> str:
        """Lightweight Japanese rendering for the generated distance report prose."""
        s = str(line)
        replacements = {
            "near distance": "近距離",
            "middle distance": "中距離",
            "far distance": "遠距離",
            "Detection quality is consistent across distance ranges": "距離レンジ全体で検出品質は安定しています",
            "show similar recall, so there is no obvious range-specific drop": "のRecallは近く、明確な距離依存の低下は見られません",
            "Detection is strongest in": "検出性能が最も良いのは",
            "and weakest in": "で、最も弱いのは",
            "The": "",
            "result looks strong": "の結果は良好です",
            "result looks acceptable but not yet strong": "の結果は許容範囲ですが、まだ強いとは言えません",
            "result looks weak": "の結果は弱めです",
            "so this range should be treated as the main recall limitation": "この距離レンジが主なRecall制約と考えられます",
            "False positives are not concentrated in a particular distance range": "FPは特定の距離レンジに集中していません",
            "FP behavior is broadly even across the selected scope": "FP傾向は概ね均一です",
            "False positives are mainly a": "FPは主に",
            "issue": "の課題です",
            "FP rate there is low": "この範囲のFP rateは低いです",
            "FP rate there is moderate": "この範囲のFP rateは中程度です",
            "FP rate there is high": "この範囲のFP rateは高いです",
            "while": "一方で",
            "is comparatively cleaner": "は比較的クリーンです",
            "The main performance concern is": "主な性能上の懸念は",
            "where missed detections and false positives are both relatively concentrated": "で、未検出とFPの両方が相対的に集中しています",
            "The main recall concern is": "主なRecall上の懸念は",
            "FP behavior is not necessarily concentrated in the same range": "FPは必ずしも同じ距離レンジに集中していません",
            "The main precision concern is": "主なPrecision上の懸念は",
            "recall is not necessarily the limiting factor in that same range": "同じ距離レンジでRecallが制約とは限りません",
            "candidate is a clear improvement over baseline by distance": "Candidateは距離別ではBaselineに対して明確な改善傾向です",
            "improving recall without adding FPs in the dominant pattern": "主な傾向としてFPを増やさずRecallが改善しています",
            "candidate improves recall but pays for it with more false positives": "CandidateはRecallを改善していますが、FP増加とのトレードオフがあります",
            "candidate is more conservative": "Candidateはより保守的です",
            "it reduces false positives but gives up some recall": "FPは減りますが、一部Recallを失っています",
            "candidate shows a risky distance-level regression because recall drops while FPs increase": "CandidateはRecall低下とFP増加が同時に見られるため、距離別ではリスクの高いデグレ傾向です",
            "candidate is largely unchanged from baseline across distance": "Candidateは距離別ではBaselineから大きな変化はありません",
            "Range detail:": "距離別には、",
            "shows both recall gain and FP increase": "ではRecall改善とFP増加が同時に見られます",
            "shows recall weakness and FP increase": "ではRecall低下とFP増加が同時に見られます",
            "shows recall gain": "ではRecall改善が見られます",
            "shows recall weakness": "ではRecall低下が見られます",
            "shows FP increase": "ではFP増加が見られます",
            "shows fewer FPs": "ではFPが改善しています",
            "while": "一方、",
            "look stable or improved across distance": "は距離方向で安定または改善傾向です",
            "Recall regression is mainly associated with": "Recall低下は主に",
            "False-positive increase is mainly associated with": "FP増加は主に",
            "Main points to review are": "主な確認ポイントは",
            "driven by": "要因は",
            "both recall regression and FP increase": "Recall低下とFP増加の両方",
            "recall regression": "Recall低下",
            "FP increase": "FP増加",
            "work well across distance": "は距離方向で良好に動作しています",
            "Recall is weaker for": "Recallが弱めなのは",
            "so these classes need closer review": "これらのclassは追加確認が必要です",
            "False positives are more visible for": "FPが目立つのは",
            "At far distance, the main class-level concerns are": "遠距離でclass別に主な懸念があるのは",
        }
        for src, dst in replacements.items():
            s = s.replace(src, dst)
        s = s.replace(", 一方、", "。一方、")
        s = s.replace("、 ", "、")
        s = s.replace(": ", "：")
        return s

    def _distance_report_note_bilingual(title: str, lines: List[str]) -> None:
        """Render one combined bilingual section-level summary for distance charts."""
        clean_lines = [str(line).strip() for line in lines if str(line).strip()]
        if not clean_lines:
            return
        st.markdown(f"**{title}**")
        col_en, col_ja = st.columns(2)
        with col_en:
            st.markdown("\n".join(f"- {line}" for line in clean_lines))
        with col_ja:
            st.markdown("\n".join(f"- {_distance_summary_jp_line(line)}" for line in clean_lines))

    def _rate_level_text(value: Any) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return _report_pct(value)

    def _distance_zone_name(bin_order: Any) -> str:
        try:
            order = float(bin_order)
        except Exception:
            return "unknown range"
        if order < 50:
            return "near distance"
        if order < 100:
            return "middle distance"
        return "far distance"

    def _report_join_words(items: List[str], limit: int = 4) -> str:
        clean = [str(x).strip() for x in items if str(x).strip()]
        if not clean:
            return ""
        clean = clean[:limit]
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]} and {clean[1]}"
        return ", ".join(clean[:-1]) + f", and {clean[-1]}"

    def _display_label_name(label: Any) -> str:
        s = str(label).strip()
        return s if s else "(no label)"

    def _recall_quality(value: Any) -> str:
        if value is None or pd.isna(value):
            return "unclear"
        v = float(value)
        if v >= 0.8:
            return "strong"
        if v >= 0.6:
            return "acceptable but not yet strong"
        return "weak"

    def _fp_quality(value: Any) -> str:
        if value is None or pd.isna(value):
            return "unclear"
        v = float(value)
        if v <= 0.1:
            return "low"
        if v <= 0.25:
            return "moderate"
        return "high"

    def _zone_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "bin_order" not in df.columns:
            return pd.DataFrame()
        d = df.copy()
        d["zone"] = d["bin_order"].map(_distance_zone_name)
        return (
            d.groupby("zone", as_index=False)
            .agg(
                tpr=("tpr", "mean"),
                fpr=("fpr", "mean"),
                min_bin=("bin_order", "min"),
            )
            .sort_values("min_bin")
        )

    def _range_detail_sentence(
        *,
        recall_gain_zone: Optional[str] = None,
        recall_loss_zone: Optional[str] = None,
        fp_rise_zone: Optional[str] = None,
        fp_drop_zone: Optional[str] = None,
    ) -> str:
        observations: List[str] = []
        if recall_gain_zone and recall_gain_zone == fp_rise_zone:
            observations.append(f"{recall_gain_zone} shows both recall gain and FP increase")
            fp_rise_zone = None
        elif recall_loss_zone and recall_loss_zone == fp_rise_zone:
            observations.append(f"{recall_loss_zone} shows recall weakness and FP increase")
            fp_rise_zone = None
        elif recall_gain_zone:
            observations.append(f"{recall_gain_zone} shows recall gain")
        if recall_loss_zone and recall_loss_zone != recall_gain_zone:
            observations.append(f"{recall_loss_zone} shows recall weakness")
        if fp_rise_zone:
            observations.append(f"{fp_rise_zone} shows FP increase")
        if fp_drop_zone:
            observations.append(f"{fp_drop_zone} shows fewer FPs")
        if not observations:
            return ""
        if len(observations) == 1:
            return f"Range detail: {observations[0]}."
        return f"Range detail: {observations[0]}, while " + "; ".join(observations[1:]) + "."

    def _distance_single_result_lines(df_rates: pd.DataFrame) -> List[str]:
        if df_rates is None or df_rates.empty:
            return []
        d = df_rates.copy()
        d["tpr"] = pd.to_numeric(d["tpr"], errors="coerce")
        d["fpr"] = pd.to_numeric(d["fpr"], errors="coerce")
        d = d.dropna(subset=["bin_label"])
        if d.empty:
            return []

        lines = []
        tpr_valid = d.dropna(subset=["tpr"])
        fpr_valid = d.dropna(subset=["fpr"])
        zone_summary = _zone_metric_summary(d)
        if not zone_summary.empty and not tpr_valid.empty:
            best_zone = zone_summary.sort_values("tpr", ascending=False).iloc[0]
            weak_zone = zone_summary.sort_values("tpr", ascending=True).iloc[0]
            spread = float(zone_summary["tpr"].max() - zone_summary["tpr"].min())
            if spread < 0.03:
                lines.append(
                    f"Detection quality is consistent across distance ranges; {best_zone['zone']} and "
                    f"{weak_zone['zone']} show similar recall, so there is no obvious range-specific drop."
                )
            else:
                lines.append(
                    f"Detection is strongest in {best_zone['zone']} and weakest in {weak_zone['zone']}. "
                    f"The {weak_zone['zone']} result looks {_recall_quality(weak_zone['tpr'])}, so this range should be treated as the main recall limitation."
                )
        if not zone_summary.empty and not fpr_valid.empty:
            high_fp_zone = zone_summary.sort_values("fpr", ascending=False).iloc[0]
            low_fp_zone = zone_summary.sort_values("fpr", ascending=True).iloc[0]
            spread = float(zone_summary["fpr"].max() - zone_summary["fpr"].min())
            if spread < 0.03:
                lines.append(
                    "False positives are not concentrated in a particular distance range; FP behavior is broadly even across the selected scope."
                )
            else:
                lines.append(
                    f"False positives are mainly a {high_fp_zone['zone']} issue. FP rate there is {_fp_quality(high_fp_zone['fpr'])}, "
                    f"while {low_fp_zone['zone']} is comparatively cleaner."
                )
        if not zone_summary.empty and len(zone_summary) >= 2:
            weak_zone = zone_summary.sort_values("tpr", ascending=True).iloc[0]
            high_fp_zone = zone_summary.sort_values("fpr", ascending=False).iloc[0]
            if weak_zone["zone"] == high_fp_zone["zone"] and (
                _recall_quality(weak_zone["tpr"]) == "weak" or _fp_quality(high_fp_zone["fpr"]) == "high"
            ):
                lines.append(
                    f"The main performance concern is {weak_zone['zone']}, where missed detections and false positives are both relatively concentrated."
                )
            elif _recall_quality(weak_zone["tpr"]) == "weak":
                lines.append(
                    f"The main recall concern is {weak_zone['zone']}; FP behavior is not necessarily concentrated in the same range."
                )
            elif _fp_quality(high_fp_zone["fpr"]) == "high":
                lines.append(
                    f"The main precision concern is {high_fp_zone['zone']}; recall is not necessarily the limiting factor in that same range."
                )
        return lines

    def _distance_compare_result_lines(
        df_tpr: pd.DataFrame,
        df_fpr: pd.DataFrame,
        run_order: List[str],
    ) -> List[str]:
        if not run_order or len(run_order) < 2:
            return []
        base_run = run_order[0]
        lines = []
        tpr_pivot = pd.DataFrame()
        fpr_pivot = pd.DataFrame()
        if df_tpr is not None and not df_tpr.empty:
            tpr_pivot = df_tpr.pivot_table(index="bin_label", columns="run", values="tpr", aggfunc="first")
        if df_fpr is not None and not df_fpr.empty:
            fpr_pivot = df_fpr.pivot_table(index="bin_label", columns="run", values="fpr", aggfunc="first")
        bin_meta_parts = []
        for df_src in (df_tpr, df_fpr):
            if df_src is not None and not df_src.empty and {"bin_label", "bin_order"}.issubset(df_src.columns):
                bin_meta_parts.append(df_src[["bin_label", "bin_order"]])
        if bin_meta_parts:
            bin_meta = pd.concat(bin_meta_parts, ignore_index=True).drop_duplicates("bin_label")
        else:
            bin_meta = pd.DataFrame(columns=["bin_label", "bin_order"])

        for compare_run in run_order[1:]:
            if base_run not in tpr_pivot.columns and base_run not in fpr_pivot.columns:
                continue
            tpr_delta = pd.Series(dtype="float64")
            fpr_delta = pd.Series(dtype="float64")
            if base_run in tpr_pivot.columns and compare_run in tpr_pivot.columns:
                tpr_delta = (tpr_pivot[compare_run] - tpr_pivot[base_run]).dropna()
            if base_run in fpr_pivot.columns and compare_run in fpr_pivot.columns:
                fpr_delta = (fpr_pivot[compare_run] - fpr_pivot[base_run]).dropna()
            if tpr_delta.empty and fpr_delta.empty:
                continue

            overlap_bins = sorted(set(tpr_delta.index).intersection(set(fpr_delta.index)))
            tradeoff_counts = {"clear": 0, "recall_tradeoff": 0, "conservative": 0, "regression": 0, "flat": 0}
            if overlap_bins:
                for bin_label in overlap_bins:
                    dt = float(tpr_delta.loc[bin_label])
                    dfp = float(fpr_delta.loc[bin_label])
                    if abs(dt) < 0.001 and abs(dfp) < 0.001:
                        tradeoff_counts["flat"] += 1
                    elif dt >= 0.001 and dfp <= -0.001:
                        tradeoff_counts["clear"] += 1
                    elif dt >= 0.001 and dfp > 0.001:
                        tradeoff_counts["recall_tradeoff"] += 1
                    elif dt < -0.001 and dfp <= -0.001:
                        tradeoff_counts["conservative"] += 1
                    else:
                        tradeoff_counts["regression"] += 1

            dominant_case = max(tradeoff_counts, key=tradeoff_counts.get) if overlap_bins else "flat"
            case_text = {
                "clear": "is a clear improvement over baseline by distance, improving recall without adding FPs in the dominant pattern",
                "recall_tradeoff": "improves recall but pays for it with more false positives",
                "conservative": "is more conservative: it reduces false positives but gives up some recall",
                "regression": "shows a risky distance-level regression because recall drops while FPs increase",
                "flat": "is largely unchanged from baseline across distance",
            }[dominant_case]

            recall_gain_zone = None
            recall_loss_zone = None
            fp_rise_zone = None
            fp_drop_zone = None
            if not tpr_delta.empty and not bin_meta.empty:
                tpr_df = tpr_delta.rename("tpr_delta").reset_index().merge(bin_meta, on="bin_label", how="left")
                tpr_df["zone"] = tpr_df["bin_order"].map(_distance_zone_name)
                tpr_zone = tpr_df.groupby("zone")["tpr_delta"].mean().dropna()
                if not tpr_zone.empty:
                    best_recall_zone = tpr_zone.sort_values(ascending=False).index[0]
                    weakest_recall_zone = tpr_zone.sort_values().index[0]
                    if float(tpr_zone.loc[best_recall_zone]) > 0.01:
                        recall_gain_zone = str(best_recall_zone)
                    if float(tpr_zone.loc[weakest_recall_zone]) < -0.01:
                        recall_loss_zone = str(weakest_recall_zone)
            if not fpr_delta.empty and not bin_meta.empty:
                fpr_df = fpr_delta.rename("fpr_delta").reset_index().merge(bin_meta, on="bin_label", how="left")
                fpr_df["zone"] = fpr_df["bin_order"].map(_distance_zone_name)
                fpr_zone = fpr_df.groupby("zone")["fpr_delta"].mean().dropna()
                if not fpr_zone.empty:
                    highest_fp_zone = fpr_zone.sort_values(ascending=False).index[0]
                    lowest_fp_zone = fpr_zone.sort_values().index[0]
                    if float(fpr_zone.loc[highest_fp_zone]) > 0.01:
                        fp_rise_zone = str(highest_fp_zone)
                    if float(fpr_zone.loc[lowest_fp_zone]) < -0.01:
                        fp_drop_zone = str(lowest_fp_zone)
            range_detail = _range_detail_sentence(
                recall_gain_zone=recall_gain_zone,
                recall_loss_zone=recall_loss_zone,
                fp_rise_zone=fp_rise_zone,
                fp_drop_zone=fp_drop_zone,
            )
            if range_detail:
                lines.append(f"{compare_run} vs {base_run}: candidate {case_text}. {range_detail}")
            else:
                lines.append(f"{compare_run} vs {base_run}: candidate {case_text}.")
        return lines

    def _distance_label_result_lines(
        df_label_dist: pd.DataFrame,
        label_order: List[str],
        bin_order: Optional[List[str]],
        run_order: List[str],
    ) -> List[str]:
        if df_label_dist is None or df_label_dist.empty:
            return []
        df_label_dist = df_label_dist.copy()
        if "label_str" not in df_label_dist.columns and "label" in df_label_dist.columns:
            df_label_dist["label_str"] = df_label_dist["label"].astype(str)
        if {"bin_label", "bin_order"}.issubset(df_label_dist.columns):
            bin_order_by_label = (
                df_label_dist[["bin_label", "bin_order"]]
                .drop_duplicates("bin_label")
                .set_index("bin_label")["bin_order"]
                .to_dict()
            )
        else:
            bin_order_by_label = {}
        lines = []
        if len(run_order) >= 2:
            base_run = run_order[0]
            compare_run = run_order[1]
            pivot = df_label_dist.pivot_table(
                index=["label_str", "bin_label"],
                columns="run",
                values=["tpr", "fpr"],
                aggfunc="first",
            )
            if ("tpr", base_run) in pivot.columns and ("tpr", compare_run) in pivot.columns:
                tpr_delta = (pivot[("tpr", compare_run)] - pivot[("tpr", base_run)]).dropna()
            else:
                tpr_delta = pd.Series(dtype="float64")
            if ("fpr", base_run) in pivot.columns and ("fpr", compare_run) in pivot.columns:
                fpr_delta = (pivot[("fpr", compare_run)] - pivot[("fpr", base_run)]).dropna()
            else:
                fpr_delta = pd.Series(dtype="float64")

            label_rows = []
            labels_for_delta = sorted(
                set([idx[0] for idx in tpr_delta.index]).union(set([idx[0] for idx in fpr_delta.index]))
            )
            for lab in labels_for_delta:
                lab_tpr = tpr_delta.loc[lab] if lab in tpr_delta.index.get_level_values(0) else pd.Series(dtype="float64")
                lab_fpr = fpr_delta.loc[lab] if lab in fpr_delta.index.get_level_values(0) else pd.Series(dtype="float64")
                label_rows.append(
                    {
                        "label": lab,
                        "tpr_delta": float(lab_tpr.mean()) if len(lab_tpr) else np.nan,
                        "fpr_delta": float(lab_fpr.mean()) if len(lab_fpr) else np.nan,
                    }
                )
            label_delta = pd.DataFrame(label_rows)
            if not label_delta.empty:
                stable = label_delta[
                    (label_delta["tpr_delta"].fillna(0) >= -0.01)
                    & (label_delta["fpr_delta"].fillna(0) <= 0.01)
                ].sort_values(["tpr_delta", "fpr_delta"], ascending=[False, True])
                if not stable.empty:
                    stable_labels = [_display_label_name(x) for x in stable["label"].head(4).tolist()]
                    lines.append(
                        f"{_report_join_words(stable_labels)} look stable or improved across distance."
                    )
                recall_risk = label_delta[label_delta["tpr_delta"] < -0.01].sort_values("tpr_delta").head(3)
                if not recall_risk.empty:
                    risk_labels = [_display_label_name(x) for x in recall_risk["label"].tolist()]
                    lines.append(
                        f"Recall regression is mainly associated with {_report_join_words(risk_labels)}."
                    )
                fp_risk = label_delta[label_delta["fpr_delta"] > 0.01].sort_values("fpr_delta", ascending=False).head(3)
                if not fp_risk.empty:
                    fp_labels = [_display_label_name(x) for x in fp_risk["label"].tolist()]
                    lines.append(
                        f"False-positive increase is mainly associated with {_report_join_words(fp_labels)}."
                    )

            paired = []
            for key in sorted(set(tpr_delta.index).intersection(set(fpr_delta.index))):
                dt = float(tpr_delta.loc[key])
                dfp = float(fpr_delta.loc[key])
                zone_text = _distance_zone_name(bin_order_by_label.get(key[1], 0))
                if dt < -0.001 and dfp > 0.001:
                    paired.append((abs(dt) + abs(dfp), key, zone_text, "both recall regression and FP increase"))
                elif dt < -0.001:
                    paired.append((abs(dt), key, zone_text, "recall regression"))
                elif dfp > 0.001:
                    paired.append((abs(dfp), key, zone_text, "FP increase"))
            if paired:
                paired = sorted(paired, reverse=True)[:3]
                focus = [
                    f"{_display_label_name(key[0])} in {zone_text}"
                    for _, key, zone_text, reason in paired
                ]
                reasons = sorted(set(reason for _, _, _, reason in paired))
                lines.append(
                    f"Main points to review are {_report_join_words(focus, limit=3)}, driven by {_report_join_words(reasons, limit=3)}."
                )
        else:
            d = df_label_dist.copy()
            d["tpr"] = pd.to_numeric(d["tpr"], errors="coerce")
            d["fpr"] = pd.to_numeric(d["fpr"], errors="coerce")
            if "bin_order" in d.columns:
                d["zone"] = d["bin_order"].map(_distance_zone_name)
            else:
                d["zone"] = "selected range"
            label_summary = (
                d.groupby("label_str", as_index=False)
                .agg(tpr=("tpr", "mean"), fpr=("fpr", "mean"))
                .dropna(subset=["tpr", "fpr"], how="all")
            )
            if not label_summary.empty:
                strong = label_summary[
                    (label_summary["tpr"].fillna(0) >= 0.75)
                    & (label_summary["fpr"].fillna(1) <= 0.25)
                ].sort_values(["tpr", "fpr"], ascending=[False, True])
                if not strong.empty:
                    strong_labels = [_display_label_name(x) for x in strong["label_str"].head(4).tolist()]
                    lines.append(
                        f"{_report_join_words(strong_labels)} work well across distance."
                    )
                weak = label_summary[label_summary["tpr"].fillna(1) < 0.6].sort_values("tpr").head(3)
                if not weak.empty:
                    weak_labels = [_display_label_name(x) for x in weak["label_str"].tolist()]
                    lines.append(
                        f"Recall is weaker for {_report_join_words(weak_labels)}, so these classes need closer review."
                    )
                noisy = label_summary[label_summary["fpr"].fillna(0) > 0.25].sort_values("fpr", ascending=False).head(3)
                if not noisy.empty:
                    noisy_labels = [_display_label_name(x) for x in noisy["label_str"].tolist()]
                    lines.append(
                        f"False positives are more visible for {_report_join_words(noisy_labels)}."
                    )
            zone_label_summary = (
                d.groupby(["label_str", "zone"], as_index=False)
                .agg(tpr=("tpr", "mean"), fpr=("fpr", "mean"))
                .dropna(subset=["tpr", "fpr"], how="all")
            )
            if not zone_label_summary.empty:
                far_or_weak = zone_label_summary[
                    (zone_label_summary["zone"] == "far distance")
                    & (
                        (zone_label_summary["tpr"].fillna(1) < 0.65)
                        | (zone_label_summary["fpr"].fillna(0) > 0.25)
                    )
                ].sort_values(["tpr", "fpr"], ascending=[True, False]).head(3)
                if not far_or_weak.empty:
                    labels = [_display_label_name(x) for x in far_or_weak["label_str"].tolist()]
                    lines.append(
                        f"At far distance, the main class-level concerns are {_report_join_words(labels)}."
                    )
        return lines
    
    
    # Same 10 m bins as eval_flat / TPR-FPR stats (used for object-count alignment)
    _DIST_BIN_CASE = """CASE
      WHEN dist_h >= 0 AND dist_h < 10 THEN '[0,10)'
      WHEN dist_h >= 10 AND dist_h < 20 THEN '[10,20)'
      WHEN dist_h >= 20 AND dist_h < 30 THEN '[20,30)'
      WHEN dist_h >= 30 AND dist_h < 40 THEN '[30,40)'
      WHEN dist_h >= 40 AND dist_h < 50 THEN '[40,50)'
      WHEN dist_h >= 50 AND dist_h < 60 THEN '[50,60)'
      WHEN dist_h >= 60 AND dist_h < 70 THEN '[60,70)'
      WHEN dist_h >= 70 AND dist_h < 80 THEN '[70,80)'
      WHEN dist_h >= 80 AND dist_h < 90 THEN '[80,90)'
      WHEN dist_h >= 90 AND dist_h < 100 THEN '[90,100)'
      WHEN dist_h >= 100 AND dist_h < 110 THEN '[100,110)'
      WHEN dist_h >= 110 AND dist_h < 120 THEN '[110,120)'
      WHEN dist_h >= 120 AND dist_h < 130 THEN '[120,130)'
      WHEN dist_h >= 130 AND dist_h < 140 THEN '[130,140)'
      WHEN dist_h >= 140 AND dist_h < 150 THEN '[140,150)'
      WHEN dist_h >= 150 THEN '[150,inf)'
      ELSE '[unknown]' END"""
    
    
    # =============================
    # Panel 3–5: Distance — TP/FP rates by bin + object count vs range
    # =============================
    ds_dlog("section: Panel3_5_Distance_start")
    st.divider()
    st.markdown(
        section_header_html(
            "Distance: TP/FP rates & object count",
            "Same distance bins and chart style (line or bar) for rates and object counts; x-axis order matches across charts.",
        ),
        unsafe_allow_html=True,
    )
    rate_by_dist_style = st.radio(
        "Chart style",
        options=["Line chart (trend)", "Bar chart (histogram)"],
        index=1,
        horizontal=True,
        key="tp_fp_rate_by_dist_style",
    )
    
    # In compare mode, use the per-run mapped baseline topic so old/new equivalent topic
    # names can still be compared without combining multiple outputs from the same run.
    filter_clause_base = build_filter_clause(filters_list[0] if filters_list else filters_base, enable_dist_h=False)
    ds_dlog(
        "distance: filter_clause_base (no dist_h) len=%s preview=%s",
        len(filter_clause_base),
        filter_clause_base[:600],
    )
    _dist_slot = st.empty()
    _dist_slot.markdown(ds_spot_loading_markup("Distance · TP/FP rates & object counts"), unsafe_allow_html=True)
    try:
        ds_dlog("distance_inner_try: single_mode=%s", single_mode)
        ds_debug_log_memory("distance_inner_try_start")
        use_line_chart = rate_by_dist_style == "Line chart (trend)"
        rate_bin_labels_order: Optional[List[str]] = None
        distance_summary_lines: List[str] = []
    
        if single_mode:
            # Inline stats from view_eval_flat (avoid nested TPR/FPR view — DuckDB can SIGSEGV on that plan).
            query_both = sql_distance_bin_rates_from_eval_flat(
                "view_eval_flat", filter_clause_base, metrics="both"
            )
            ds_dlog("distance: executing query_both (single_mode TPR/FPR by bin, inlined from eval_flat)")
            df_both = con.execute(query_both).df()
            ds_dlog("distance: query_both done rows=%s cols=%s", len(df_both), list(df_both.columns))
            ds_debug_log_memory("distance_after_query_both")
            if not df_both.empty:
                df_both["bin_order"], df_both["bin_label"] = zip(
                    *df_both["distance_bin"].map(_distance_bin_order_and_label)
                )
                df_both = df_both.sort_values("bin_order")
                x_labels = df_both["bin_label"].tolist()
                rate_bin_labels_order = x_labels
    
                if use_line_chart:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=x_labels,
                            y=df_both["tpr"],
                            name="TP rate",
                            mode="lines",
                            line=dict(color=RUN_COLORS[0], width=2.5, shape="spline"),
                            fill="tozeroy",
                            fillcolor="rgba(74, 144, 217, 0.2)",
                            hovertemplate="%{x}<br>TP rate: %{y:.2%}<extra></extra>",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_labels,
                            y=df_both["fpr"],
                            name="FP rate",
                            mode="lines",
                            line=dict(color=RUN_COLORS[1], width=2.5, shape="spline"),
                            fill="tozeroy",
                            fillcolor="rgba(232, 106, 51, 0.2)",
                            hovertemplate="%{x}<br>FP rate: %{y:.2%}<extra></extra>",
                        )
                    )
                    apply_chart_theme(fig, height=420)
                    fig.update_layout(
                        title="TP & FP rate by distance bin",
                        xaxis_title="Distance bin",
                        yaxis_title="Rate",
                        yaxis_range=[0, 1],
                        xaxis=dict(
                            tickangle=-35,
                            categoryorder="array",
                            categoryarray=x_labels,
                        ),
                        hovermode="x unified",
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.25)")
                    st.plotly_chart(fig, width='stretch')
                else:
                    # Bar chart (histogram): combined TP + FP grouped bars
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=x_labels,
                            y=df_both["tpr"],
                            name="TP rate",
                            marker_color=RUN_COLORS[0],
                            hovertemplate="%{x}<br>TP rate: %{y:.2%}<extra></extra>",
                        )
                    )
                    fig.add_trace(
                        go.Bar(
                            x=x_labels,
                            y=df_both["fpr"],
                            name="FP rate",
                            marker_color=RUN_COLORS[1],
                            hovertemplate="%{x}<br>FP rate: %{y:.2%}<extra></extra>",
                        )
                    )
                    apply_chart_theme(fig, height=420)
                    fig.update_layout(
                        title="TP & FP rate by distance bin",
                        xaxis_title="Distance bin",
                        yaxis_title="Rate",
                        yaxis_range=[0, 1],
                        barmode="group",
                        xaxis=dict(
                            tickangle=-35,
                            categoryorder="array",
                            categoryarray=x_labels,
                        ),
                        hovermode="x unified",
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.25)")
                    st.plotly_chart(fig, width='stretch')

                distance_summary_lines.extend(_distance_single_result_lines(df_both))

                query_label_rates = sql_distance_bin_label_rates_from_eval_flat(
                    "view_eval_flat", filter_clause_base
                )
                ds_dlog("distance: executing query_label_rates (single_mode TPR/FPR by label and bin)")
                df_label_rates = con.execute(query_label_rates).df()
                ds_dlog(
                    "distance: query_label_rates done rows=%s cols=%s",
                    len(df_label_rates),
                    list(df_label_rates.columns),
                )
                if not df_label_rates.empty:
                    df_label_rates["bin_order"], df_label_rates["bin_label"] = zip(
                        *df_label_rates["distance_bin"].map(_distance_bin_order_and_label)
                    )
                    df_label_rates = df_label_rates.sort_values(["bin_order", "label"])
                    label_order = sorted(df_label_rates["label"].dropna().astype(str).unique().tolist())
                    for metric_col, metric_name in [
                        ("tpr", "TP rate"),
                        ("fpr", "FP rate"),
                    ]:
                        fig_label = go.Figure()
                        for j, lab in enumerate(label_order):
                            d = df_label_rates[df_label_rates["label"].astype(str) == lab].sort_values("bin_order")
                            c = RUN_COLORS[j % len(RUN_COLORS)]
                            if use_line_chart:
                                r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
                                fig_label.add_trace(
                                    go.Scatter(
                                        x=d["bin_label"],
                                        y=d[metric_col],
                                        name=lab,
                                        mode="lines",
                                        line=dict(color=c, width=2.2, shape="spline"),
                                        fill="tozeroy",
                                        fillcolor=f"rgba({r},{g},{b},0.12)",
                                        hovertemplate=f"{lab}<br>%{{x}}<br>{metric_name}: %{{y:.2%}}<extra></extra>",
                                    )
                                )
                            else:
                                fig_label.add_trace(
                                    go.Bar(
                                        x=d["bin_label"],
                                        y=d[metric_col],
                                        name=lab,
                                        marker_color=c,
                                        hovertemplate=f"{lab}<br>%{{x}}<br>{metric_name}: %{{y:.2%}}<extra></extra>",
                                    )
                                )
                        apply_chart_theme(fig_label, height=420)
                        fig_label.update_layout(
                            title=f"{metric_name} by label and distance bin",
                            xaxis_title="Distance bin",
                            yaxis_title=metric_name,
                            yaxis_range=[0, 1],
                            xaxis=dict(
                                tickangle=-35,
                                categoryorder="array",
                                categoryarray=x_labels,
                            ),
                            hovermode="x unified",
                            **({"barmode": "group"} if not use_line_chart else {}),
                        )
                        fig_label.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.25)")
                        st.plotly_chart(fig_label, width='stretch')
                    distance_summary_lines.extend(
                        _distance_label_result_lines(
                            df_label_rates,
                            label_order,
                            rate_bin_labels_order,
                            run_labels_list,
                        )
                    )
            else:
                st.info("No distance-bin data available.")
        else:
            # Compare mode: fetch TP and FP by distance per run
            ds_dlog("distance: compare_mode n_runs=%s", len(runs))
            dfs_tpr = []
            for i in range(len(runs)):
                fc = build_filter_clause(filters_list[i], enable_dist_h=False)
                q = sql_distance_bin_rates_from_eval_flat(_flat_view(i), fc, metrics="tpr")
                ds_dlog("distance: compare run %s/%s TPR by bin query", i + 1, len(runs))
                df_i = con.execute(q).df()
                ds_dlog("distance: compare TPR query run %s rows=%s", i, len(df_i))
                df_i["run"] = run_labels_list[i]
                df_i["bin_order"], df_i["bin_label"] = zip(*df_i["distance_bin"].map(_distance_bin_order_and_label))
                df_i = df_i.sort_values("bin_order")
                dfs_tpr.append(df_i)
            df_tpr_dist = pd.concat(dfs_tpr, ignore_index=True)
            ds_dlog("distance: df_tpr_dist total_rows=%s", len(df_tpr_dist))
    
            dfs_fpr = []
            for i in range(len(runs)):
                fc = build_filter_clause(filters_list[i], enable_dist_h=False)
                q = sql_distance_bin_rates_from_eval_flat(_flat_view(i), fc, metrics="fpr")
                ds_dlog("distance: compare run %s/%s FPR by bin query", i + 1, len(runs))
                df_i = con.execute(q).df()
                ds_dlog("distance: compare FPR query run %s rows=%s", i, len(df_i))
                df_i["run"] = run_labels_list[i]
                df_i["bin_order"], df_i["bin_label"] = zip(*df_i["distance_bin"].map(_distance_bin_order_and_label))
                df_i = df_i.sort_values("bin_order")
                dfs_fpr.append(df_i)
            df_fpr_dist = pd.concat(dfs_fpr, ignore_index=True)
    
            if not df_tpr_dist.empty:
                rate_bin_labels_order = (
                    df_tpr_dist[df_tpr_dist["run"] == run_labels_list[0]]
                    .sort_values("bin_order")["bin_label"]
                    .tolist()
                )
            _xaxis_dist_bins = (
                dict(tickangle=-35, categoryorder="array", categoryarray=rate_bin_labels_order)
                if rate_bin_labels_order
                else dict(tickangle=-35)
            )
    
            if use_line_chart:
                if not df_tpr_dist.empty:
                    fig_tpr = go.Figure()
                    for i, lbl in enumerate(run_labels_list):
                        d = df_tpr_dist[df_tpr_dist["run"] == lbl].sort_values("bin_order")
                        c = RUN_COLORS[i % len(RUN_COLORS)]
                        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
                        fig_tpr.add_trace(
                            go.Scatter(
                                x=d["bin_label"],
                                y=d["tpr"],
                                name=lbl,
                                mode="lines",
                                line=dict(color=c, width=2.2, shape="spline"),
                                fill="tozeroy",
                                fillcolor=f"rgba({r},{g},{b},0.15)",
                                hovertemplate=f"{lbl}<br>%{{x}}<br>TP rate: %{{y:.2%}}<extra></extra>",
                            )
                        )
                    apply_chart_theme(fig_tpr, height=420)
                    fig_tpr.update_layout(
                        title="TP rate by distance",
                        xaxis_title="Distance bin",
                        yaxis_title="TP rate",
                        yaxis_range=[0, 1],
                        xaxis=_xaxis_dist_bins,
                        hovermode="x unified",
                    )
                    fig_tpr.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.25)")
                    st.plotly_chart(fig_tpr, width='stretch')
                else:
                    st.info("No TP rate by distance data.")
    
                if not df_fpr_dist.empty:
                    fig_fpr = go.Figure()
                    for i, lbl in enumerate(run_labels_list):
                        d = df_fpr_dist[df_fpr_dist["run"] == lbl].sort_values("bin_order")
                        c = RUN_COLORS[i % len(RUN_COLORS)]
                        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
                        fig_fpr.add_trace(
                            go.Scatter(
                                x=d["bin_label"],
                                y=d["fpr"],
                                name=lbl,
                                mode="lines",
                                line=dict(color=c, width=2.2, shape="spline"),
                                fill="tozeroy",
                                fillcolor=f"rgba({r},{g},{b},0.15)",
                                hovertemplate=f"{lbl}<br>%{{x}}<br>FP rate: %{{y:.2%}}<extra></extra>",
                            )
                        )
                    apply_chart_theme(fig_fpr, height=420)
                    fig_fpr.update_layout(
                        title="FP rate by distance",
                        xaxis_title="Distance bin",
                        yaxis_title="FP rate",
                        yaxis_range=[0, 1],
                        xaxis=_xaxis_dist_bins,
                        hovermode="x unified",
                    )
                    fig_fpr.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.25)")
                    st.plotly_chart(fig_fpr, width='stretch')
                else:
                    st.info("No FP rate by distance data.")
            else:
                # Bar chart (histogram) for compare: TP then FP, grouped by run
                if not df_tpr_dist.empty:
                    fig_tpr = go.Figure()
                    for i, lbl in enumerate(run_labels_list):
                        d = df_tpr_dist[df_tpr_dist["run"] == lbl].sort_values("bin_order")
                        fig_tpr.add_trace(
                            go.Bar(
                                x=d["bin_label"],
                                y=d["tpr"],
                                name=lbl,
                                marker_color=RUN_COLORS[i % len(RUN_COLORS)],
                                hovertemplate=f"{lbl}<br>%{{x}}<br>TP rate: %{{y:.2%}}<extra></extra>",
                            )
                        )
                    apply_chart_theme(fig_tpr, height=420)
                    fig_tpr.update_layout(
                        title="TP rate by distance",
                        xaxis_title="Distance bin",
                        yaxis_title="TP rate",
                        yaxis_range=[0, 1],
                        barmode="group",
                        xaxis=_xaxis_dist_bins,
                        hovermode="x unified",
                    )
                    fig_tpr.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.25)")
                    st.plotly_chart(fig_tpr, width='stretch')
                else:
                    st.info("No TP rate by distance data.")
    
                if not df_fpr_dist.empty:
                    fig_fpr = go.Figure()
                    for i, lbl in enumerate(run_labels_list):
                        d = df_fpr_dist[df_fpr_dist["run"] == lbl].sort_values("bin_order")
                        fig_fpr.add_trace(
                            go.Bar(
                                x=d["bin_label"],
                                y=d["fpr"],
                                name=lbl,
                                marker_color=RUN_COLORS[i % len(RUN_COLORS)],
                                hovertemplate=f"{lbl}<br>%{{x}}<br>FP rate: %{{y:.2%}}<extra></extra>",
                            )
                        )
                    apply_chart_theme(fig_fpr, height=420)
                    fig_fpr.update_layout(
                        title="FP rate by distance",
                        xaxis_title="Distance bin",
                        yaxis_title="FP rate",
                        yaxis_range=[0, 1],
                        barmode="group",
                        xaxis=_xaxis_dist_bins,
                        hovermode="x unified",
                    )
                    fig_fpr.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.25)")
                    st.plotly_chart(fig_fpr, width='stretch')
                else:
                    st.info("No FP rate by distance data.")

            distance_summary_lines.extend(_distance_compare_result_lines(df_tpr_dist, df_fpr_dist, run_labels_list))

            dfs_label_rates = []
            for i in range(len(runs)):
                fc = build_filter_clause(filters_list[i], enable_dist_h=False)
                q = sql_distance_bin_label_rates_from_eval_flat(_flat_view(i), fc)
                ds_dlog("distance: compare run %s/%s TPR/FPR by label and bin query", i + 1, len(runs))
                df_i = con.execute(q).df()
                ds_dlog("distance: compare label rate query run %s rows=%s", i, len(df_i))
                if not df_i.empty:
                    df_i["run"] = run_labels_list[i]
                    df_i["bin_order"], df_i["bin_label"] = zip(
                        *df_i["distance_bin"].map(_distance_bin_order_and_label)
                    )
                    dfs_label_rates.append(df_i)

            if dfs_label_rates:
                df_label_dist = pd.concat(dfs_label_rates, ignore_index=True)
                df_label_dist["label_str"] = df_label_dist["label"].astype(str)
                selected_label_order = [str(l) for l in (selected_labels if selected_labels else labels)]
                present_labels = set(df_label_dist["label_str"].dropna().tolist())
                label_order = [lab for lab in selected_label_order if lab in present_labels]
                label_order.extend(sorted(present_labels.difference(label_order)))

                if label_order:
                    label_compare_views = [
                        "Change matrices",
                        "Trend grid",
                        "Bar grid",
                    ]
                    if st.session_state.get("distance_label_compare_view") not in (None, *label_compare_views):
                        st.session_state["distance_label_compare_view"] = label_compare_views[0]
                    label_compare_view = st.radio(
                        "Label distance compare view",
                        options=label_compare_views,
                        index=0,
                        horizontal=True,
                        key="distance_label_compare_view",
                    )
                    distance_summary_lines.extend(
                        _distance_label_result_lines(
                            df_label_dist,
                            label_order,
                            rate_bin_labels_order,
                            run_labels_list,
                        )
                    )

                    def _render_distance_delta_matrix(
                        metric_col: str,
                        title: str,
                        color_label: str,
                        *,
                        fp_better_lower: bool = False,
                        delta_label: str = "Change",
                    ):
                        if len(run_labels_list) < 2:
                            st.info("Delta matrix requires at least two runs.")
                            return
                        base_run = run_labels_list[0]
                        compare_run = run_labels_list[1]
                        pivot = df_label_dist.pivot_table(
                            index="label_str",
                            columns=["bin_label", "run"],
                            values=metric_col,
                            aggfunc="first",
                        )
                        matrix_rows = []
                        hover_rows = []
                        for lab in label_order:
                            row_vals = []
                            hover_vals = []
                            for bin_label in rate_bin_labels_order or []:
                                base_val = np.nan
                                compare_val = np.nan
                                if (bin_label, base_run) in pivot.columns and lab in pivot.index:
                                    base_val = pivot.loc[lab, (bin_label, base_run)]
                                if (bin_label, compare_run) in pivot.columns and lab in pivot.index:
                                    compare_val = pivot.loc[lab, (bin_label, compare_run)]
                                display_delta = compare_val - base_val
                                row_vals.append(display_delta)
                                if pd.isna(base_val) or pd.isna(compare_val):
                                    hover_vals.append(f"{lab}<br>{bin_label}<br>No paired data")
                                else:
                                    hover_vals.append(
                                        f"{lab}<br>{bin_label}<br>"
                                        f"{base_run}: {base_val:.1%}<br>"
                                        f"{compare_run}: {compare_val:.1%}<br>"
                                        f"{delta_label}: {display_delta:+.1%}"
                                    )
                            matrix_rows.append(row_vals)
                            hover_rows.append(hover_vals)

                        matrix_sorted = matrix_rows
                        hover_sorted = hover_rows
                        max_abs_delta = max(
                            [
                                abs(float(v))
                                for row in matrix_sorted
                                for v in row
                                if pd.notna(v)
                            ]
                            or [0.01]
                        )
                        max_abs_delta = max(max_abs_delta, 0.01)
                        fig_matrix = px.imshow(
                            matrix_sorted,
                            x=rate_bin_labels_order,
                            y=label_order,
                            labels=dict(x="Distance bin", y="Label", color=color_label),
                            color_continuous_scale=[
                                [0.0, IMPROVED_COLOR if fp_better_lower else DEGRADED_COLOR],
                                [0.5, "#f8fafc"],
                                [1.0, DEGRADED_COLOR if fp_better_lower else IMPROVED_COLOR],
                            ],
                            zmin=-max_abs_delta,
                            zmax=max_abs_delta,
                            aspect="auto",
                        )
                        fig_matrix.update_traces(
                            customdata=hover_sorted,
                            hovertemplate="%{customdata}<extra></extra>",
                        )
                        apply_chart_theme(fig_matrix, height=max(360, 92 + 24 * len(label_order)))
                        fig_matrix.update_layout(
                            title=title,
                            xaxis_side="top",
                            coloraxis_colorbar=dict(tickformat="+.0%"),
                        )
                        fig_matrix.update_xaxes(tickangle=-35)
                        st.plotly_chart(fig_matrix, width='stretch')

                    def _render_distance_grid(*, use_bars: bool = False):
                        small_multiple_cols = min(3, max(1, len(label_order)))
                        small_multiple_rows = int(np.ceil(len(label_order) / small_multiple_cols))

                        for metric_col, metric_name in [("tpr", "TP rate"), ("fpr", "FP rate")]:
                            vertical_spacing = 0.105 if small_multiple_rows <= 1 else min(0.105, 0.9 / (small_multiple_rows - 1))
                            fig_sm = make_subplots(
                                rows=small_multiple_rows,
                                cols=small_multiple_cols,
                                subplot_titles=label_order,
                                shared_yaxes=True,
                                horizontal_spacing=0.055,
                                vertical_spacing=vertical_spacing,
                            )
                            legend_shown = set()
                            for lab_idx, lab in enumerate(label_order):
                                row = lab_idx // small_multiple_cols + 1
                                col = lab_idx % small_multiple_cols + 1
                                for run_idx, run_lbl in enumerate(run_labels_list):
                                    d = df_label_dist[
                                        (df_label_dist["label_str"] == lab)
                                        & (df_label_dist["run"] == run_lbl)
                                    ].sort_values("bin_order")
                                    if d.empty:
                                        continue
                                    c = RUN_COLORS[run_idx % len(RUN_COLORS)]
                                    show_legend = run_lbl not in legend_shown
                                    legend_shown.add(run_lbl)
                                    if use_bars:
                                        fig_sm.add_trace(
                                            go.Bar(
                                                x=d["bin_label"],
                                                y=d[metric_col],
                                                name=str(run_lbl),
                                                marker_color=c,
                                                showlegend=show_legend,
                                                hovertemplate=(
                                                    f"{run_lbl}<br>{lab}<br>%{{x}}<br>"
                                                    f"{metric_name}: %{{y:.2%}}<extra></extra>"
                                                ),
                                            ),
                                            row=row,
                                            col=col,
                                        )
                                    else:
                                        fig_sm.add_trace(
                                            go.Scatter(
                                                x=d["bin_label"],
                                                y=d[metric_col],
                                                name=str(run_lbl),
                                                mode="lines+markers",
                                                line=dict(color=c, width=2.2),
                                                marker=dict(size=4.5, color=c, line=dict(width=0.8, color="white")),
                                                showlegend=show_legend,
                                                hovertemplate=(
                                                    f"{run_lbl}<br>{lab}<br>%{{x}}<br>"
                                                    f"{metric_name}: %{{y:.2%}}<extra></extra>"
                                                ),
                                            ),
                                            row=row,
                                            col=col,
                                        )

                            fig_sm_height = max(420, 235 * small_multiple_rows)
                            apply_chart_theme(
                                fig_sm,
                                height=fig_sm_height,
                                margin=dict(t=72, b=46, l=52, r=24),
                            )
                            fig_sm.update_layout(
                                title=f"{metric_name} by label and distance bin",
                                yaxis_range=[0, 1],
                                hovermode="closest",
                                **({"barmode": "group"} if use_bars else {}),
                            )
                            for r_idx in range(1, small_multiple_rows + 1):
                                for c_idx in range(1, small_multiple_cols + 1):
                                    fig_sm.update_yaxes(
                                        range=[0, 1],
                                        tickformat=".0%",
                                        showticklabels=c_idx == 1,
                                        gridcolor="rgba(0,0,0,0.06)",
                                        zeroline=False,
                                        row=r_idx,
                                        col=c_idx,
                                    )
                                    fig_sm.update_xaxes(
                                        tickangle=-35,
                                        categoryorder="array",
                                        categoryarray=rate_bin_labels_order,
                                        showticklabels=r_idx == small_multiple_rows,
                                        gridcolor="rgba(0,0,0,0.04)",
                                        zeroline=False,
                                        row=r_idx,
                                        col=c_idx,
                                    )
                            st.plotly_chart(fig_sm, width='stretch')

                    if label_compare_view == "Change matrices":
                        _render_distance_delta_matrix(
                            "tpr",
                            f"TP diff by label and distance ({run_labels_list[1]} - {run_labels_list[0]})",
                            "TP diff",
                            delta_label="Diff",
                        )
                        _render_distance_delta_matrix(
                            "fpr",
                            f"FP diff by label and distance ({run_labels_list[1]} - {run_labels_list[0]})",
                            "FP diff",
                            fp_better_lower=True,
                            delta_label="Diff",
                        )
                    elif label_compare_view == "Bar grid":
                        _render_distance_grid(use_bars=True)
                    else:
                        _render_distance_grid()
            else:
                st.info("No label-level TP/FP rate data by distance bin.")
    
        # Object count by same distance bins as TP/FP; same line vs bar style; aligned x-axis
    
        try:
            if single_mode:
                q_oc = f"""
                SELECT ({_DIST_BIN_CASE}) AS distance_bin, label, COUNT(*) AS n
                FROM view_eval_flat
                WHERE {filter_clause_base}
                GROUP BY 1, 2
                """
                df_oc = con.execute(q_oc).df()
            else:
                dfs_oc = []
                for i in range(len(runs)):
                    fc_oc = build_filter_clause(filters_list[i], enable_dist_h=False)
                    q_oc_i = f"""
                    SELECT ({_DIST_BIN_CASE}) AS distance_bin, COUNT(*) AS n
                    FROM {_flat_view(i)}
                    WHERE {fc_oc}
                    GROUP BY 1
                    """
                    df_oci = con.execute(q_oc_i).df()
                    df_oci["run"] = run_labels_list[i]
                    dfs_oc.append(df_oci)
                df_oc = pd.concat(dfs_oc, ignore_index=True)
    
            if df_oc.empty:
                st.info("No object count data by distance bin.")
            else:
                df_oc = df_oc.copy()
                df_oc["bin_order"], df_oc["bin_label"] = zip(*df_oc["distance_bin"].map(_distance_bin_order_and_label))
                if rate_bin_labels_order:
                    align_x = list(rate_bin_labels_order)
                else:
                    align_x = (
                        df_oc.drop_duplicates("distance_bin")
                        .sort_values("bin_order")["bin_label"]
                        .tolist()
                    )
    
                xaxis_oc = dict(tickangle=-35, categoryorder="array", categoryarray=align_x)
    
                if single_mode:
                    pivot_oc = df_oc.pivot_table(
                        index="bin_label", columns="label", values="n", aggfunc="sum", fill_value=0
                    )
                    pivot_oc = pivot_oc.reindex(align_x, fill_value=0)
    
                    fig_oc = go.Figure()
                    if use_line_chart:
                        for j, lab in enumerate(pivot_oc.columns):
                            c = RUN_COLORS[j % len(RUN_COLORS)]
                            r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
                            nm = str(lab)
                            fig_oc.add_trace(
                                go.Scatter(
                                    x=align_x,
                                    y=pivot_oc[lab].values,
                                    name=nm,
                                    mode="lines",
                                    line=dict(color=c, width=2.2, shape="spline"),
                                    fill="tozeroy",
                                    fillcolor=f"rgba({r},{g},{b},0.12)",
                                    hovertemplate=f"{nm}<br>%{{x}}<br>Count: %{{y:.0f}}<extra></extra>",
                                )
                            )
                    else:
                        for j, lab in enumerate(pivot_oc.columns):
                            c = RUN_COLORS[j % len(RUN_COLORS)]
                            nm = str(lab)
                            fig_oc.add_trace(
                                go.Bar(
                                    x=align_x,
                                    y=pivot_oc[lab].values,
                                    name=nm,
                                    marker_color=c,
                                    hovertemplate=f"{nm}<br>%{{x}}<br>Count: %{{y:.0f}}<extra></extra>",
                                )
                            )
                    apply_chart_theme(fig_oc, height=420)
                    fig_oc.update_layout(
                        title="Object count by distance bin",
                        xaxis_title="Distance bin",
                        yaxis_title="Count",
                        xaxis=xaxis_oc,
                        hovermode="x unified",
                        **({"barmode": "group"} if not use_line_chart else {}),
                    )
                    st.plotly_chart(fig_oc, width='stretch')
                else:
                    pivot_oc = df_oc.pivot_table(
                        index="bin_label", columns="run", values="n", aggfunc="sum", fill_value=0
                    )
                    pivot_oc = pivot_oc.reindex(align_x, fill_value=0)
                    run_cols = [r for r in run_labels_list if r in pivot_oc.columns]
    
                    fig_oc = go.Figure()
                    if use_line_chart:
                        for j, rl in enumerate(run_cols):
                            c = RUN_COLORS[j % len(RUN_COLORS)]
                            r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
                            fig_oc.add_trace(
                                go.Scatter(
                                    x=align_x,
                                    y=pivot_oc[rl].values,
                                    name=str(rl),
                                    mode="lines",
                                    line=dict(color=c, width=2.2, shape="spline"),
                                    fill="tozeroy",
                                    fillcolor=f"rgba({r},{g},{b},0.15)",
                                    hovertemplate=f"{rl}<br>%{{x}}<br>Count: %{{y:.0f}}<extra></extra>",
                                )
                            )
                    else:
                        for j, rl in enumerate(run_cols):
                            c = RUN_COLORS[j % len(RUN_COLORS)]
                            fig_oc.add_trace(
                                go.Bar(
                                    x=align_x,
                                    y=pivot_oc[rl].values,
                                    name=str(rl),
                                    marker_color=c,
                                    hovertemplate=f"{rl}<br>%{{x}}<br>Count: %{{y:.0f}}<extra></extra>",
                                )
                            )
                    apply_chart_theme(fig_oc, height=420)
                    fig_oc.update_layout(
                        title="Object count by distance bin",
                        xaxis_title="Distance bin",
                        yaxis_title="Count",
                        xaxis=xaxis_oc,
                        hovermode="x unified",
                        **({"barmode": "group"} if not use_line_chart else {}),
                    )
                    st.plotly_chart(fig_oc, width='stretch')
        except Exception as e_oc:
            st.error(f"Error (object count by distance bin): {e_oc}")

        if single_mode:
            _distance_report_note_bilingual(
                "Distance performance summary",
                distance_summary_lines,
            )
    
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        _dist_slot.empty()
    ds_dlog("section: Panel3_5_Distance_end")
    # =============================
    # Panel 2: TP Rate (single) / TP Rate Comparison (compare)
    # =============================
    ds_dlog("section: Panel2_TP_Rate_start")
    st.markdown(
        section_header_html(
            "TP Rate" + (" Comparison" if not single_mode else ""),
            "TP rate per object class (GT TP / (TP+FN)). Pick a chart style below.",
        ),
        unsafe_allow_html=True,
    )
    
    _tpr_query = """
    SELECT
        label,
        CASE
            WHEN COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) > 0
            THEN CAST(COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS DOUBLE)
                 / COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN'))
            ELSE 0
        END AS tpr
    FROM {view}
    WHERE {filter_clause}
    GROUP BY label
    ORDER BY label
    """
    
    # Compare-mode TP rate spider charts: several distance caps + no cap (sidebar range not used for this view)
    TPR_COMPARE_SPIDER_RANGES: List[Tuple[Optional[int], str]] = [
        (50, "≤50 m"),
        (80, "≤80 m"),
        (100, "≤100 m"),
        (120, "≤120 m"),
        (150, "≤150 m"),
        (None, "All distances"),
    ]
    
    if single_mode:
        tpr_viz = st.radio(
            "TP rate chart style",
            options=["Bar chart", "Lollipop (ranked)"],
            index=0,
            horizontal=True,
            key="tpr_viz_single",
        )
        try:
            with ds_spot_loading("TP rate"):
                filter_clause = build_filter_clause(filters_base)
                query = _tpr_query.format(view="view_eval_flat", filter_clause=filter_clause)
                df_tpr_base = con.execute(query).df()
            if not df_tpr_base.empty:
                title = f"Total TP rate within {max_eval_range} [m]"
                if tpr_viz == "Bar chart":
                    fig = px.bar(
                        df_tpr_base,
                        x="label",
                        y="tpr",
                        title=title,
                        labels={"tpr": "TP Rate", "label": "Label"},
                    )
                    apply_chart_theme(fig)
                    fig.update_layout(yaxis_range=[0, 1.2])
                    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.2)")
                    st.plotly_chart(fig, width='stretch')
                else:
                    fig = _tpr_lollipop_single(df_tpr_base, title)
                    st.plotly_chart(fig, width='stretch')
            else:
                st.info("No data available")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        tpr_opts = ["Spider chart", "Grouped bar", "Heatmap (label × run)", "Line profile"]
        tpr_viz = st.radio(
            "TP rate chart style",
            options=tpr_opts,
            index=0,
            horizontal=True,
            key="tpr_viz_compare",
        )
        try:
            with ds_spot_loading("TP rate"):
                dfs_tpr = []
                for i in range(len(runs)):
                    fc = build_filter_clause(filters_list[i])
                    q = _tpr_query.format(view=_flat_view(i), filter_clause=fc)
                    df_i = con.execute(q).df()
                    df_i["run"] = run_labels_list[i]
                    dfs_tpr.append(df_i)
                df_tpr_all = pd.concat(dfs_tpr, ignore_index=True)
            if tpr_viz == "Spider chart":
                st.caption(
                    "Six spider charts use **fixed distance cutoffs** (50–150 m) plus **all distances**. "
                    "Topic / label / suite / visibility filters still apply. "
                    "Other chart types and the rest of the page use the sidebar **Max Evaluation Range**."
                )
                label_union: set = set()
                for i in range(len(runs)):
                    fb_all = {**filters_list[i], "max_eval_range": None}
                    fc_a = build_filter_clause(fb_all)
                    q_a = _tpr_query.format(view=_flat_view(i), filter_clause=fc_a)
                    dfa = con.execute(q_a).df()
                    label_union |= set(dfa["label"].astype(str))
                cats = sorted(label_union)
                if not cats:
                    st.info("No TP rate data for any distance range with current filters.")
                else:
                    if len(cats) > 16:
                        st.caption("Spider charts work best with ≤16 labels; many classes may look crowded.")
                    for row_start in range(0, len(TPR_COMPARE_SPIDER_RANGES), 3):
                        row_ranges = TPR_COMPARE_SPIDER_RANGES[row_start : row_start + 3]
                        cols = st.columns(len(row_ranges))
                        for col, (max_r, cap_lbl) in zip(cols, row_ranges):
                            dfs_slice = []
                            for i in range(len(runs)):
                                fb = {**filters_list[i], "max_eval_range": max_r}
                                fc = build_filter_clause(fb)
                                q = _tpr_query.format(view=_flat_view(i), filter_clause=fc)
                                dfi = con.execute(q).df()
                                dfi["run"] = run_labels_list[i]
                                dfs_slice.append(dfi)
                            df_slice = pd.concat(dfs_slice, ignore_index=True)
                            with col:
                                if df_slice.empty:
                                    st.info(f"No data ({cap_lbl}).")
                                else:
                                    fig = _tpr_spider_compare(
                                        df_slice,
                                        cats,
                                        f"TP rate ({cap_lbl})",
                                        run_labels_list,
                                        height=360,
                                    )
                                    st.plotly_chart(fig, width='stretch')
            elif not df_tpr_all.empty:
                title = f"Total TP rate within {max_eval_range} [m] by run"
                if tpr_viz == "Grouped bar":
                    fig = px.bar(
                        df_tpr_all,
                        x="label",
                        y="tpr",
                        color="run",
                        barmode="group",
                        title=title,
                        labels={"tpr": "TP Rate", "label": "Label", "run": "Run"},
                        color_discrete_sequence=RUN_COLORS,
                    )
                    apply_chart_theme(fig)
                    fig.update_layout(yaxis_range=[0, 1.2])
                    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.2)")
                    st.plotly_chart(fig, width='stretch')
                elif tpr_viz == "Heatmap (label × run)":
                    pivot = df_tpr_all.pivot_table(index="label", columns="run", values="tpr", aggfunc="first")
                    cols_present = [c for c in run_labels_list if c in pivot.columns]
                    if cols_present:
                        pivot = pivot[cols_present]
                    fig = px.imshow(
                        pivot,
                        labels=dict(x="Run", y="Label", color="TP rate"),
                        title=title,
                        color_continuous_scale="RdYlGn",
                        zmin=0,
                        zmax=1,
                        aspect="auto",
                    )
                    apply_chart_theme(fig, height=max(360, 32 + 22 * len(pivot.index)))
                    fig.update_layout(xaxis_side="top")
                    st.plotly_chart(fig, width='stretch')
                elif tpr_viz == "Line profile":
                    fig = px.line(
                        df_tpr_all,
                        x="label",
                        y="tpr",
                        color="run",
                        markers=True,
                        title=title,
                        labels={"tpr": "TP Rate", "label": "Label", "run": "Run"},
                        color_discrete_sequence=RUN_COLORS,
                    )
                    fig.update_traces(line=dict(width=2.5), marker=dict(size=8))
                    apply_chart_theme(fig, height=400)
                    fig.update_layout(yaxis_range=[0, 1.15], xaxis_tickangle=-35, hovermode="x unified")
                    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.2)")
                    st.plotly_chart(fig, width='stretch')
            else:
                st.info("No data available")
        except Exception as e:
            st.error(f"Error: {e}")
    # =============================
    # Panel 5: Perception diff vs baseline A (compare mode only)
    # =============================
    def _baobab_hierarchy_from_objects(
        df_obj: pd.DataFrame,
        change_type: str,
        root_label: str,
        max_scenarios: int,
        max_datasets: int,
        max_frames: int,
    ) -> pd.DataFrame:
        """
        Build a leaf table for Plotly sunburst/treemap: root → scenario → dataset → frame → label.
        Caps scenarios, datasets per scenario, and frames per dataset; merges the rest into Other buckets.
        """
        if df_obj.empty or "change_type" not in df_obj.columns:
            return pd.DataFrame()
        sub = df_obj[df_obj["change_type"] == change_type].copy()
        if sub.empty:
            return pd.DataFrame()
        sub["scenario_name"] = sub["scenario_name"].fillna("").astype(str).replace("", "(no scenario)")
        sub["t4dataset_id"] = sub["t4dataset_id"].fillna("").astype(str).replace("", "(no dataset)")
        dataset_name = sub.get("t4dataset_name", sub["t4dataset_id"])
        sub["dataset_display"] = dataset_name.fillna("").astype(str)
        sub["dataset_display"] = sub["dataset_display"].where(
            sub["dataset_display"].str.strip() != "",
            sub["t4dataset_id"],
        )
        sub["dataset_key"] = sub["t4dataset_id"] + "|" + sub["dataset_display"]
        sub["label"] = sub["label"].fillna("").astype(str).replace("", "(no label)")
        sub["frame_key"] = "f" + sub["frame_index"].astype(str)
        leaf = (
            sub.groupby(["scenario_name", "dataset_key", "frame_key", "label"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        if leaf.empty:
            return pd.DataFrame()
        ms = max(int(max_scenarios), 1)
        md = max(int(max_datasets), 1)
        mf = max(int(max_frames), 1)
        scen_tot = leaf.groupby("scenario_name")["n"].sum().sort_values(ascending=False)
        top_scen = set(scen_tot.head(ms).index)
        leaf["scen_g"] = np.where(
            leaf["scenario_name"].isin(top_scen),
            leaf["scenario_name"],
            "Other scenarios",
        )
        parts = []
        for _, g in leaf.groupby("scen_g"):
            ds_tot = g.groupby("dataset_key")["n"].sum().sort_values(ascending=False)
            top_ds = set(ds_tot.head(md).index)
            g2 = g.copy()
            g2["dataset_g"] = np.where(
                g2["dataset_key"].isin(top_ds),
                g2["dataset_key"],
                "Other datasets",
            )
            for _, dg in g2.groupby("dataset_g"):
                fr_tot = dg.groupby("frame_key")["n"].sum().sort_values(ascending=False)
                top_fr = set(fr_tot.head(mf).index)
                dg2 = dg.copy()
                dg2["fr_g"] = np.where(dg2["frame_key"].isin(top_fr), dg2["frame_key"], "Other frames")
                agg = dg2.groupby(["scen_g", "dataset_g", "fr_g", "label"], as_index=False)["n"].sum()
                parts.append(agg)
        out = pd.concat(parts, ignore_index=True)
        out["root"] = root_label
    
        def _dataset_ring_label(dataset_g: str) -> str:
            if str(dataset_g) == "Other datasets":
                return "Other datasets"
            text = str(dataset_g).split("|", 1)[-1]
            return text if len(text) <= 34 else (text[:31] + "...")

        def _frame_ring_label(fr_g: str) -> str:
            if fr_g == "Other frames" or str(fr_g) == "Other frames":
                return "Other frames"
            return str(fr_g)
    
        out["dataset_display"] = out["dataset_g"].map(_dataset_ring_label)
        out["fr_display"] = out.apply(
            lambda r: _frame_ring_label(r["fr_g"]), axis=1
        )
        return out
    
    
    def _comparison_lens_treemap_df(
        names: pd.Series,
        improved: pd.Series,
        degraded: pd.Series,
        root_title: str,
        side_labels: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """Rows for px.treemap path root → side1|side2 → item (area = n).
        
        side_labels: optional (improved_label, degraded_label) tuple.
                     Defaults to ("Improved", "Degraded").
        """
        if side_labels is None:
            side_labels = ("Improved", "Degraded")
        rows = []
        for i in range(len(names)):
            nm = str(names.iloc[i]).strip() or "—"
            if len(nm) > 72:
                nm = nm[:69] + "…"
            ip = float(improved.iloc[i]) if pd.notna(improved.iloc[i]) else 0.0
            dg = float(degraded.iloc[i]) if pd.notna(degraded.iloc[i]) else 0.0
            if ip > 0:
                rows.append(
                    {"root": root_title, "side": side_labels[0], "item": nm, "n": ip}
                )
            if dg > 0:
                rows.append(
                    {"root": root_title, "side": side_labels[1], "item": nm, "n": dg}
                )
        return pd.DataFrame(rows)
    
    
    def _plot_comparison_lens_treemap(
        tdf: pd.DataFrame,
        st_key: str,
        title: str,
        path: Optional[List[str]] = None,
    ) -> None:
        if tdf is None or tdf.empty:
            st.caption("_No data for this view._")
            return
        fig = px.treemap(
            tdf,
            path=path or ["root", "side", "item"],
            values="n",
            color="side",
            color_discrete_map={"Improved": IMPROVED_COLOR, "Degraded": DEGRADED_COLOR},
        )
        fig.update_traces(
            textfont_size=12,
            textinfo="label+value+percent parent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "GT objects: %{value:.0f}<br>"
                "% of parent: %{percentParent}<extra></extra>"
            ),
            marker_line_width=1.5,
            marker_line_color="rgba(255,255,255,0.45)",
            root_color="rgba(240,240,245,0.95)",
        )
        _title_layout = {**PLOTLY_LAYOUT_THEME["title"], "text": title}
        apply_chart_theme(
            fig,
            height=560,
            margin=dict(t=20, l=2, r=2, b=2),
            paper_bgcolor="rgba(0,0,0,0)",
            title=_title_layout,
        )
        st.plotly_chart(fig, width='stretch', key=st_key)


    def _comparison_lens_nested_treemap_df(
        df: pd.DataFrame,
        levels: List[str],
        root_title: str,
        side_labels: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """Rows for px.treemap where two sides contain nested focus levels.
        
        side_labels: optional (improved_label, degraded_label) tuple.
                     Defaults to ("Improved", "Degraded").
        """
        if df is None or df.empty:
            return pd.DataFrame()
        if side_labels is None:
            side_labels = ("Improved", "Degraded")
        rows = []
        for _, row in df.iterrows():
            path_values: Dict[str, str] = {}
            for level in levels:
                value = str(row.get(level, "")).strip()
                path_values[level] = value or "-"
            for side, col in zip(side_labels, ("improved_cnt", "degraded_cnt")):
                n = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").fillna(0).iloc[0]
                if float(n) <= 0:
                    continue
                rows.append(
                    {
                        "root": root_title,
                        "side": side,
                        **path_values,
                        "n": float(n),
                    }
                )
        return pd.DataFrame(rows)


    def _sunburst_without_frame_layer(hdf: pd.DataFrame) -> pd.DataFrame:
        if hdf is None or hdf.empty:
            return pd.DataFrame()
        return (
            hdf.groupby(["root", "scen_g", "dataset_display", "label"], as_index=False, dropna=False)["n"]
            .sum()
        )
    
    
    if not single_mode:
        _distance_report_note_bilingual(
            "Distance performance summary",
            locals().get("distance_summary_lines", []),
        )
        ds_dlog("section: Perception_TP_FN_diff_start")
        st.divider()
        st.markdown(
            section_header_html(
                "Perception diff: GT TP/FN changes (vs baseline A)",
                "Per-GT-object comparison vs baseline A: recovered = was FN on A and TP on candidate; lost = was TP on A and FN on candidate.",
            ),
            unsafe_allow_html=True,
        )
        for idx in range(1, len(runs)):
            lbl = run_labels_list[idx]
            _pd_slot = st.empty()
            _pd_slot.markdown(ds_spot_loading_markup(f"GT TP/FN diff · run {lbl}"), unsafe_allow_html=True)
            try:
                filter_clause_comp_p5 = build_filter_clause(filters_list[idx], enable_dist_h=False)
                comp_flat = _flat_view(idx)
                query = f"""
                WITH base_gt AS (
                    SELECT
                        t4dataset_id,
                        frame_index,
                        uuid AS gt_uuid,
                        COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
                        COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                        COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                        COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                    FROM view_eval_flat
                    WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                        AND {filter_clause_base}
                    GROUP BY 1,2,3
                ),
                comp_gt AS (
                    SELECT
                        t4dataset_id,
                        frame_index,
                        uuid AS gt_uuid,
                        COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
                        COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                        COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                        COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                    FROM {comp_flat}
                    WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                        AND {filter_clause_comp_p5}
                    GROUP BY 1,2,3
                ),
                joined AS (
                    SELECT
                        COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR)) AS t4dataset_id,
                        COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index,
                        COALESCE(b.gt_uuid, c.gt_uuid) AS gt_uuid,
                        b.gt_uuid IS NOT NULL AS has_base_gt,
                        c.gt_uuid IS NOT NULL AS has_candidate_gt,
                        COALESCE(b.tp_base, FALSE) AS tp_base,
                        COALESCE(c.tp_comp, FALSE) AS tp_comp,
                        COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
                        COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
                        COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name
                    FROM base_gt b
                    FULL OUTER JOIN comp_gt c
                        ON b.t4dataset_id = c.t4dataset_id
                       AND b.frame_index = c.frame_index
                       AND b.gt_uuid = c.gt_uuid
                )
                SELECT
                    t4dataset_id,
                    CAST(COUNT(*) FILTER (WHERE TRUE) AS DOUBLE) AS total_gt,
                    CAST(COUNT(*) FILTER (WHERE has_base_gt) AS DOUBLE) AS base_gt_cnt,
                    CAST(COUNT(*) FILTER (WHERE has_candidate_gt) AS DOUBLE) AS candidate_gt_cnt,
                    CAST(COUNT(*) FILTER (WHERE NOT has_base_gt AND has_candidate_gt) AS DOUBLE) AS missing_in_base_cnt,
                    CAST(COUNT(*) FILTER (WHERE has_base_gt AND NOT has_candidate_gt) AS DOUBLE) AS missing_in_candidate_cnt,
                    CAST(COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS DOUBLE) AS improved_cnt,
                    CAST(COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS DOUBLE) AS degraded_cnt,
                    CAST(COUNT(*) FILTER (WHERE tp_base AND tp_comp) AS DOUBLE) AS both_tp_cnt,
                    CAST(COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp) AS DOUBLE) AS both_fn_cnt,
                    CAST(SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS DOUBLE) AS net_tp_delta,
                    suite_name,
                    scenario_name,
                    t4dataset_name
                FROM joined
                GROUP BY t4dataset_id, suite_name, scenario_name, t4dataset_name
                ORDER BY net_tp_delta DESC
                """
                df_improved = con.execute(query).df()
                if not df_improved.empty:
                    query_output_availability_p5 = f"""
                    WITH base_output AS (
                        SELECT
                            CAST(t4dataset_id AS VARCHAR) AS t4dataset_id,
                            COUNT(*) AS total_est_base,
                            COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                            COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                            COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                        FROM view_eval_flat
                        WHERE source = 'EST' AND frame_index IS NOT NULL
                            AND {filter_clause_base}
                        GROUP BY 1
                    ),
                    comp_output AS (
                        SELECT
                            CAST(t4dataset_id AS VARCHAR) AS t4dataset_id,
                            COUNT(*) AS total_est_comp,
                            COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                            COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                            COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                        FROM {comp_flat}
                        WHERE source = 'EST' AND frame_index IS NOT NULL
                            AND {filter_clause_comp_p5}
                        GROUP BY 1
                    )
                    SELECT
                        COALESCE(b.t4dataset_id, c.t4dataset_id) AS t4dataset_id,
                        CAST(COALESCE(b.total_est_base, 0) + COALESCE(c.total_est_comp, 0) AS DOUBLE) AS total_est,
                        CAST(COALESCE(b.total_est_base, 0) AS DOUBLE) AS base_est_cnt,
                        CAST(COALESCE(c.total_est_comp, 0) AS DOUBLE) AS candidate_est_cnt,
                        CAST(CASE WHEN b.total_est_base IS NULL AND c.total_est_comp IS NOT NULL THEN c.total_est_comp ELSE 0 END AS DOUBLE) AS missing_in_base_cnt,
                        CAST(CASE WHEN b.total_est_base IS NOT NULL AND c.total_est_comp IS NULL THEN b.total_est_base ELSE 0 END AS DOUBLE) AS missing_in_candidate_cnt,
                        COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
                        COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
                        COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name
                    FROM base_output b
                    FULL OUTER JOIN comp_output c
                        ON b.t4dataset_id = c.t4dataset_id
                    ORDER BY t4dataset_id
                    """
                    try:
                        df_output_availability = con.execute(query_output_availability_p5).df()
                    except Exception:
                        df_output_availability = pd.DataFrame()
                    query_frame_p5 = f"""
                            WITH base_gt AS (
                                SELECT
                                    t4dataset_id,
                                    frame_index,
                                    uuid AS gt_uuid,
                                    COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
                                    COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                                    COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                                    COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                                FROM view_eval_flat
                                WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                                    AND {filter_clause_base}
                                GROUP BY 1, 2, 3
                            ),
                            comp_gt AS (
                                SELECT
                                    t4dataset_id,
                                    frame_index,
                                    uuid AS gt_uuid,
                                    COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
                                    COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                                    COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                                    COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                                FROM {comp_flat}
                                WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                                    AND {filter_clause_comp_p5}
                                GROUP BY 1, 2, 3
                            ),
                            joined AS (
                                SELECT
                                    COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR)) AS t4dataset_id,
                                    COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index,
                                    COALESCE(b.gt_uuid, c.gt_uuid) AS gt_uuid,
                                    b.gt_uuid IS NOT NULL AS has_base_gt,
                                    c.gt_uuid IS NOT NULL AS has_candidate_gt,
                                    COALESCE(b.tp_base, FALSE) AS tp_base,
                                    COALESCE(c.tp_comp, FALSE) AS tp_comp,
                                    COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
                                    COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
                                    COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name
                                FROM base_gt b
                                FULL OUTER JOIN comp_gt c
                                    ON b.t4dataset_id = c.t4dataset_id
                                   AND b.frame_index = c.frame_index
                                   AND b.gt_uuid = c.gt_uuid
                            )
                            SELECT
                                t4dataset_id,
                                frame_index,
                                scenario_name,
                                suite_name,
                                t4dataset_name,
                                CAST(COUNT(*) FILTER (WHERE TRUE) AS DOUBLE) AS total_gt,
                                CAST(COUNT(*) FILTER (WHERE has_base_gt) AS DOUBLE) AS base_gt_cnt,
                                CAST(COUNT(*) FILTER (WHERE has_candidate_gt) AS DOUBLE) AS candidate_gt_cnt,
                                CAST(COUNT(*) FILTER (WHERE NOT has_base_gt AND has_candidate_gt) AS DOUBLE) AS missing_in_base_cnt,
                                CAST(COUNT(*) FILTER (WHERE has_base_gt AND NOT has_candidate_gt) AS DOUBLE) AS missing_in_candidate_cnt,
                                CAST(COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS DOUBLE) AS improved_cnt,
                                CAST(COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS DOUBLE) AS degraded_cnt,
                                CAST(COUNT(*) FILTER (WHERE tp_base AND tp_comp) AS DOUBLE) AS both_tp_cnt,
                                CAST(COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp) AS DOUBLE) AS both_fn_cnt,
                                CAST(SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS DOUBLE) AS net_tp_delta
                            FROM joined
                            GROUP BY t4dataset_id, frame_index, suite_name, scenario_name, t4dataset_name
                            ORDER BY net_tp_delta DESC
                            """
                    query_object_p5 = f"""
                            WITH base_gt AS (
                                SELECT
                                    t4dataset_id,
                                    frame_index,
                                    uuid AS gt_uuid,
                                    COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
                                    COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                                    COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                                    COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                                FROM view_eval_flat
                                WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                                    AND {filter_clause_base}
                                GROUP BY 1, 2, 3
                            ),
                            comp_gt AS (
                                SELECT
                                    t4dataset_id,
                                    frame_index,
                                    uuid AS gt_uuid,
                                    COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
                                    COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                                    COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                                    COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                                FROM {comp_flat}
                                WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                                    AND {filter_clause_comp_p5}
                                GROUP BY 1, 2, 3
                            ),
                            joined AS (
                                SELECT
                                    COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR)) AS t4dataset_id,
                                    COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index,
                                    COALESCE(b.gt_uuid, c.gt_uuid) AS gt_uuid,
                                    b.gt_uuid IS NOT NULL AS has_base_gt,
                                    c.gt_uuid IS NOT NULL AS has_candidate_gt,
                                    COALESCE(b.tp_base, FALSE) AS tp_base,
                                    COALESCE(c.tp_comp, FALSE) AS tp_comp,
                                    COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
                                    COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
                                    COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name
                                FROM base_gt b
                                FULL OUTER JOIN comp_gt c
                                    ON b.t4dataset_id = c.t4dataset_id
                                   AND b.frame_index = c.frame_index
                                   AND b.gt_uuid = c.gt_uuid
                            ),
                            obj_attrs AS (
                                SELECT
                                    t4dataset_id,
                                    frame_index,
                                    uuid,
                                    MAX(CAST(label AS VARCHAR)) AS label,
                                    MAX(dist_h) AS dist_h
                                FROM view_eval_flat
                                WHERE source = 'GT'
                                GROUP BY 1, 2, 3
                            )
                            SELECT
                                j.t4dataset_id,
                                j.frame_index,
                                j.gt_uuid,
                                j.has_base_gt,
                                j.has_candidate_gt,
                                CAST(CASE WHEN j.has_base_gt THEN 1 ELSE 0 END AS DOUBLE) AS base_gt_cnt,
                                CAST(CASE WHEN j.has_candidate_gt THEN 1 ELSE 0 END AS DOUBLE) AS candidate_gt_cnt,
                                CAST(CASE WHEN NOT j.has_base_gt AND j.has_candidate_gt THEN 1 ELSE 0 END AS DOUBLE) AS missing_in_base_cnt,
                                CAST(CASE WHEN j.has_base_gt AND NOT j.has_candidate_gt THEN 1 ELSE 0 END AS DOUBLE) AS missing_in_candidate_cnt,
                                COALESCE(e.label, '') AS label,
                                COALESCE(e.dist_h, 0.0) AS dist_h,
                                {_DIST_BIN_CASE.replace("dist_h", "COALESCE(e.dist_h, 0.0)")} AS distance_bin,
                                j.suite_name,
                                j.scenario_name,
                                j.t4dataset_name,
                                CASE
                                    WHEN NOT j.tp_base AND j.tp_comp THEN 'improved'
                                    WHEN j.tp_base AND NOT j.tp_comp THEN 'degraded'
                                    WHEN j.tp_base AND j.tp_comp THEN 'both_tp'
                                    ELSE 'both_fn'
                                END AS change_type,
                                j.tp_base,
                                j.tp_comp
                            FROM joined j
                            LEFT JOIN obj_attrs e
                                ON CAST(j.t4dataset_id AS VARCHAR) = CAST(e.t4dataset_id AS VARCHAR)
                               AND j.frame_index = CAST(e.frame_index AS VARCHAR)
                               AND j.gt_uuid = e.uuid
                            ORDER BY change_type, j.t4dataset_id, j.frame_index
                            """
                    try:
                        df_by_frame = con.execute(query_frame_p5).df()
                    except Exception:
                        df_by_frame = pd.DataFrame()
                    try:
                        df_by_object_full = con.execute(query_object_p5).df()
                    except Exception:
                        df_by_object_full = pd.DataFrame()

                    availability_messages = [
                        msg
                        for msg in [
                            _compare_availability_summary_fp(df_output_availability, unit="datasets"),
                        ]
                        if msg
                    ]
                    skip_incomplete_key = f"p5_skip_incomplete_{lbl}_{idx}"
                    skip_dataset_compare = True
                    if availability_messages:
                        st.warning(
                            "Some datasets have output rows on only one side. "
                            + "; ".join(availability_messages)
                            + ". These whole datasets can create artificial large improvements/degradations.",
                            icon="⚠️",
                        )
                        skip_dataset_compare = st.checkbox(
                            "Skip datasets with output rows on only one side",
                            value=True,
                            key=skip_incomplete_key,
                            help=(
                                "When enabled, Perception diff charts/tables remove whole datasets where either "
                                "baseline A or the candidate has no EST/output rows after the active filters. "
                                "Frame-level one-sided differences inside a valid dataset are still compared."
                            ),
                        )
                    # --- Dataset name debug ---
                    with st.expander("🔍 Debug: unique dataset names (t4dataset_id / t4dataset_name)"):
                        debug_summary = _dataset_name_debug_summary(
                            con, "view_eval_flat", filter_clause_base,
                            comp_flat, filter_clause_comp_p5,
                        )
                        st.markdown(debug_summary)
                    df_improved_skipped = pd.DataFrame()
                    if skip_dataset_compare:
                        df_improved_skipped = df_output_availability[
                            ~_compare_availability_mask_fp(df_output_availability)
                        ].copy()
                        skipped_dataset_ids = set(df_improved_skipped["t4dataset_id"].dropna().astype(str))
                        if skipped_dataset_ids:
                            df_improved = df_improved[
                                ~df_improved["t4dataset_id"].astype(str).isin(skipped_dataset_ids)
                            ].copy()
                            df_by_frame = df_by_frame[
                                ~df_by_frame["t4dataset_id"].astype(str).isin(skipped_dataset_ids)
                            ].copy()
                            df_by_object_full = df_by_object_full[
                                ~df_by_object_full["t4dataset_id"].astype(str).isin(skipped_dataset_ids)
                            ].copy()
                        if df_improved.empty:
                            st.info(
                                "All datasets for this slice have output rows on only one side after the active filters. "
                                "Disable the skip option above to inspect them."
                            )
                            continue
    
                    tot_imp = float(df_improved["improved_cnt"].sum())
                    tot_deg = float(df_improved["degraded_cnt"].sum())
                    tot_net = tot_imp - tot_deg
                    net_s = f"+{int(tot_net)}" if tot_net > 0 else str(int(tot_net))
    
                    with st.expander(f"Run {lbl} vs A", expanded=(len(runs) == 2)):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Recovered GT (FN->TP)", int(tot_imp))
                        c2.metric("Lost GT (TP->FN)", int(tot_deg))
                        c3.metric("Net TP delta", net_s)
                        c4.caption("Start with scenarios and frames with the most lost GT objects.")
                        st.markdown(
                            f"**Summary:** Net **{net_s}** TP vs baseline A — "
                            f"**{int(tot_deg)}** lost vs **{int(tot_imp)}** recovered GT objects."
                        )
                        skipped_total = len(df_improved_skipped)
                        if skipped_total > 0:
                            with st.expander("Skipped one-sided datasets"):
                                st.caption(
                                    "These whole datasets were excluded because EST/output rows exist on only one side "
                                    "after the active filters. Frames and objects are not skipped independently; they are removed "
                                    "only when their parent dataset is skipped."
                                )
                                if not df_improved_skipped.empty:
                                    skipped_dataset_rows = df_improved_skipped.copy()
                                    skipped_dataset_rows["skip_reason"] = _compare_availability_reason_fp(
                                        skipped_dataset_rows
                                    )
                                    skipped_dataset_rows = skipped_dataset_rows.rename(
                                        columns={
                                            "base_est_cnt": "baseline_output_rows",
                                            "candidate_est_cnt": "candidate_output_rows",
                                        }
                                    )
                                    st.markdown("**Skipped datasets**")
                                    st.download_button(
                                        label="Download skipped datasets (CSV)",
                                        data=skipped_dataset_rows.drop(columns=_DIFF_INTERNAL_COLS, errors="ignore").to_csv(index=False).encode("utf-8"),
                                        file_name=f"perception_diff_{lbl}_vs_A_skipped_datasets.csv",
                                        mime="text/csv",
                                        key=f"p5_dl_skip_dataset_{lbl}_{idx}",
                                    )
                                    st.dataframe(
                                        skipped_dataset_rows.head(200).drop(columns=["total_est", "missing_in_base_cnt", "missing_in_candidate_cnt"], errors="ignore"),
                                        width='stretch',
                                        hide_index=True,
                                    )

                        b_key = f"p5_baobab_{lbl}_{idx}"
                        c1b, c2b, c3b, c4b = st.columns([1, 1, 1, 1])
                        with c1b:
                            baobab_viz = st.radio(
                                "Chart type",
                                ["Sunburst", "Treemap"],
                                horizontal=True,
                                key=f"{b_key}_viz",
                            )
                        with c2b:
                            baobab_ns = st.slider(
                                "Max scenarios",
                                min_value=5,
                                max_value=25,
                                value=15,
                                key=f"{b_key}_ns",
                            )
                        with c3b:
                            baobab_nd = st.slider(
                                "Max datasets / scenario",
                                min_value=5,
                                max_value=30,
                                value=12,
                                key=f"{b_key}_nd",
                            )
                        with c4b:
                            baobab_nf = st.slider(
                                "Max frames / dataset",
                                min_value=5,
                                max_value=20,
                                value=10,
                                key=f"{b_key}_nf",
                            )
                        if df_by_object_full.empty:
                            st.caption("No object-level rows for hierarchy.")
                        else:
                            treemap_path_cols = ["root", "scen_g", "dataset_display", "fr_display", "label"]
                            sunburst_path_cols = ["root", "scen_g", "dataset_display", "label"]
                            h_imp = _baobab_hierarchy_from_objects(
                                df_by_object_full,
                                "improved",
                                f"Improved ({lbl} vs A)",
                                baobab_ns,
                                baobab_nd,
                                baobab_nf,
                            )
                            h_deg = _baobab_hierarchy_from_objects(
                                df_by_object_full,
                                "degraded",
                                f"Degraded ({lbl} vs A)",
                                baobab_ns,
                                baobab_nd,
                                baobab_nf,
                            )
                            pair_both = (not h_imp.empty) and (not h_deg.empty)
                            plot_entries = []
                            for ct, hdf, cmap in (
                                ("improved", h_imp, IMPROVED_SCALE),
                                ("degraded", h_deg, DEGRADED_SCALE),
                            ):
                                if hdf.empty:
                                    plot_entries.append((ct, None))
                                    continue
                                title = f"{baobab_viz}: {ct} (n = {int(hdf['n'].sum())} GT objects)"
                                if baobab_viz == "Sunburst":
                                    hdf_plot = _sunburst_without_frame_layer(hdf)
                                    fig_b = px.sunburst(
                                        hdf_plot,
                                        path=sunburst_path_cols,
                                        values="n",
                                        color="n",
                                        color_continuous_scale=cmap,
                                        title=title,
                                    )
                                    h_sb = 480 if pair_both else 620
                                    apply_chart_theme(fig_b, height=h_sb, margin=dict(t=36, l=4, r=4, b=4))
                                else:
                                    fig_b = px.treemap(
                                        hdf,
                                        path=treemap_path_cols,
                                        values="n",
                                        color="n",
                                        color_continuous_scale=cmap,
                                        title=title,
                                    )
                                    h_tr = 440 if pair_both else 520
                                    apply_chart_theme(fig_b, height=h_tr, margin=dict(t=40, l=4, r=4, b=4))
                                plot_entries.append((ct, fig_b))
    
                            two_up = (
                                len(plot_entries) == 2
                                and plot_entries[0][1] is not None
                                and plot_entries[1][1] is not None
                            )
                            if two_up:
                                bc1, bc2 = st.columns(2, gap="small")
                                with bc1:
                                    st.plotly_chart(
                                        plot_entries[0][1],
                                        width='stretch',
                                        key=f"{b_key}_fig_{plot_entries[0][0]}",
                                    )
                                with bc2:
                                    st.plotly_chart(
                                        plot_entries[1][1],
                                        width='stretch',
                                        key=f"{b_key}_fig_{plot_entries[1][0]}",
                                    )
                            else:
                                for ct, fig_b in plot_entries:
                                    if fig_b is not None:
                                        st.plotly_chart(
                                            fig_b,
                                            width='stretch',
                                            key=f"{b_key}_fig_{ct}",
                                        )
                                    else:
                                        st.caption(f"No **{ct}** objects to chart.")
    
                        # --- Comparison lens: label / scenario / dataset / frame (Baobab-aligned) ---
                        query_label = f"""
                        WITH base_gt AS (
                            SELECT
                                t4dataset_id,
                                frame_index,
                                uuid AS gt_uuid,
                                COALESCE(MAX(try_cast(label AS VARCHAR)), '') AS label,
                                COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base
                            FROM view_eval_flat
                            WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                                AND {filter_clause_base}
                            GROUP BY 1, 2, 3
                        ),
                        comp_gt AS (
                            SELECT
                                t4dataset_id,
                                frame_index,
                                uuid AS gt_uuid,
                                COALESCE(MAX(try_cast(label AS VARCHAR)), '') AS label,
                                COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp
                            FROM {comp_flat}
                            WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                                AND {filter_clause_comp_p5}
                            GROUP BY 1, 2, 3
                        ),
                        joined AS (
                            SELECT
                                COALESCE(b.label, c.label) AS label,
                                b.gt_uuid IS NOT NULL AS has_base_gt,
                                c.gt_uuid IS NOT NULL AS has_candidate_gt,
                                COALESCE(b.tp_base, FALSE) AS tp_base,
                                COALESCE(c.tp_comp, FALSE) AS tp_comp
                            FROM base_gt b
                            FULL OUTER JOIN comp_gt c
                                ON b.t4dataset_id = c.t4dataset_id
                               AND b.frame_index = c.frame_index
                               AND b.gt_uuid = c.gt_uuid
                        )
                        SELECT
                            label,
                            CAST(COUNT(*) FILTER (WHERE TRUE) AS DOUBLE) AS total_gt,
                            CAST(COUNT(*) FILTER (WHERE has_base_gt) AS DOUBLE) AS base_gt_cnt,
                            CAST(COUNT(*) FILTER (WHERE has_candidate_gt) AS DOUBLE) AS candidate_gt_cnt,
                            CAST(COUNT(*) FILTER (WHERE NOT has_base_gt AND has_candidate_gt) AS DOUBLE) AS missing_in_base_cnt,
                            CAST(COUNT(*) FILTER (WHERE has_base_gt AND NOT has_candidate_gt) AS DOUBLE) AS missing_in_candidate_cnt,
                            CAST(COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS DOUBLE) AS improved_cnt,
                            CAST(COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS DOUBLE) AS degraded_cnt,
                            CAST(COUNT(*) FILTER (WHERE tp_base AND tp_comp) AS DOUBLE) AS both_tp_cnt,
                            CAST(COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp) AS DOUBLE) AS both_fn_cnt,
                            CAST(SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS DOUBLE) AS net_tp_delta
                        FROM joined
                        GROUP BY label
                        ORDER BY net_tp_delta DESC
                        """
                        df_by_label = pd.DataFrame()
                        try:
                            df_by_label = con.execute(query_label).df()
                            if skip_dataset_compare:
                                if df_by_object_full.empty:
                                    df_by_label = pd.DataFrame()
                                else:
                                    df_by_label = (
                                        df_by_object_full.groupby("label", dropna=False)
                                        .agg(
                                            total_gt=("gt_uuid", "count"),
                                            base_gt_cnt=("base_gt_cnt", "sum"),
                                            candidate_gt_cnt=("candidate_gt_cnt", "sum"),
                                            missing_in_base_cnt=("missing_in_base_cnt", "sum"),
                                            missing_in_candidate_cnt=("missing_in_candidate_cnt", "sum"),
                                            improved_cnt=(
                                                "change_type",
                                                lambda s: float((s == "improved").sum()),
                                            ),
                                            degraded_cnt=(
                                                "change_type",
                                                lambda s: float((s == "degraded").sum()),
                                            ),
                                            both_tp_cnt=(
                                                "change_type",
                                                lambda s: float((s == "both_tp").sum()),
                                            ),
                                            both_fn_cnt=(
                                                "change_type",
                                                lambda s: float((s == "both_fn").sum()),
                                            ),
                                        )
                                        .reset_index()
                                    )
                                    df_by_label["net_tp_delta"] = df_by_label["improved_cnt"] - df_by_label["degraded_cnt"]
                                    df_by_label = df_by_label.sort_values("net_tp_delta", ascending=False)
                        except Exception as e_label:
                            st.caption(f"Label query: {e_label}")
    
                        scen_agg = pd.DataFrame()
                        if not df_improved.empty:
                            scen_agg = (
                                df_improved.groupby("scenario_name", dropna=False)
                                .agg(
                                    improved_cnt=("improved_cnt", "sum"),
                                    degraded_cnt=("degraded_cnt", "sum"),
                                )
                                .reset_index()
                            )
                            scen_agg = scen_agg.sort_values(
                                by=["degraded_cnt", "improved_cnt"],
                                ascending=[False, True],
                            )
    
                        frame_sort_mode = st.radio(
                            "Dataset/frame focus",
                            ["Degraded first", "Improved first", "Largest net change"],
                            horizontal=True,
                            key=f"p5_frame_focus_{lbl}_{idx}",
                            help="Choose whether dataset and frame views prioritize regressions, recoveries, or the biggest overall swings.",
                        )
                        df_dataset_sorted = pd.DataFrame()
                        df_frame_sorted = pd.DataFrame()
                        frame_caption_metric = "degraded"
                        frame_sort_desc = "degraded desc"
                        if not df_improved.empty:
                            df_dataset_sorted = df_improved.copy()
                            dataset_name = df_dataset_sorted.get(
                                "t4dataset_name",
                                df_dataset_sorted["t4dataset_id"],
                            )
                            df_dataset_sorted["_scenario_focus"] = (
                                df_dataset_sorted["scenario_name"].fillna("").astype(str).replace("", "(no scenario)")
                            )
                            df_dataset_sorted["_dataset_focus"] = dataset_name.fillna("").astype(str)
                            df_dataset_sorted["_dataset_focus"] = df_dataset_sorted["_dataset_focus"].where(
                                df_dataset_sorted["_dataset_focus"].str.strip() != "",
                                df_dataset_sorted["t4dataset_id"].fillna("").astype(str),
                            )
                            if frame_sort_mode == "Improved first":
                                df_dataset_sorted = df_dataset_sorted.sort_values(
                                    by=["improved_cnt", "degraded_cnt"],
                                    ascending=[False, True],
                                )
                            elif frame_sort_mode == "Largest net change":
                                df_dataset_sorted["net_tp_delta"] = (
                                    pd.to_numeric(df_dataset_sorted["improved_cnt"], errors="coerce").fillna(0)
                                    - pd.to_numeric(df_dataset_sorted["degraded_cnt"], errors="coerce").fillna(0)
                                )
                                df_dataset_sorted["_abs_net_tp_delta"] = (
                                    df_dataset_sorted["net_tp_delta"].abs()
                                )
                                df_dataset_sorted = df_dataset_sorted.sort_values(
                                    by=["_abs_net_tp_delta", "degraded_cnt", "improved_cnt"],
                                    ascending=[False, False, False],
                                )
                            else:
                                df_dataset_sorted = df_dataset_sorted.sort_values(
                                    by=["degraded_cnt", "improved_cnt"],
                                    ascending=[False, True],
                                )
                            df_dataset_sorted = df_dataset_sorted.drop(
                                columns=["_abs_net_tp_delta"],
                                errors="ignore",
                            ).reset_index(drop=True)
                        if not df_by_frame.empty:
                            df_frame_sorted = df_by_frame.copy()
                            frame_dataset_name = df_frame_sorted.get(
                                "t4dataset_name",
                                df_frame_sorted["t4dataset_id"],
                            )
                            df_frame_sorted["_scenario_focus"] = (
                                df_frame_sorted["scenario_name"].fillna("").astype(str).replace("", "(no scenario)")
                            )
                            df_frame_sorted["_dataset_focus"] = frame_dataset_name.fillna("").astype(str)
                            df_frame_sorted["_dataset_focus"] = df_frame_sorted["_dataset_focus"].where(
                                df_frame_sorted["_dataset_focus"].str.strip() != "",
                                df_frame_sorted["t4dataset_id"].fillna("").astype(str),
                            )
                            if frame_sort_mode == "Improved first":
                                frame_caption_metric = "improved"
                                frame_sort_desc = "improved desc"
                                df_frame_sorted = df_frame_sorted.sort_values(
                                    by=["improved_cnt", "degraded_cnt"],
                                    ascending=[False, True],
                                )
                            elif frame_sort_mode == "Largest net change":
                                frame_caption_metric = "absolute net change"
                                frame_sort_desc = "largest |net TP delta|"
                                df_frame_sorted["net_tp_delta"] = (
                                    pd.to_numeric(df_frame_sorted["improved_cnt"], errors="coerce").fillna(0)
                                    - pd.to_numeric(df_frame_sorted["degraded_cnt"], errors="coerce").fillna(0)
                                )
                                df_frame_sorted["_abs_net_tp_delta"] = (
                                    df_frame_sorted["net_tp_delta"].abs()
                                )
                                df_frame_sorted = df_frame_sorted.sort_values(
                                    by=["_abs_net_tp_delta", "degraded_cnt", "improved_cnt"],
                                    ascending=[False, False, False],
                                )
                            else:
                                df_frame_sorted = df_frame_sorted.sort_values(
                                    by=["degraded_cnt", "improved_cnt"],
                                    ascending=[False, True],
                                )
                            df_frame_sorted = df_frame_sorted.drop(
                                columns=["_abs_net_tp_delta"],
                                errors="ignore",
                            ).reset_index(drop=True)
                        _t4_link_run_names = _run_share_names_for_links()
    
                        root_lens = f"{lbl} vs A"
                        if not df_by_label.empty:
                            tdf_l = _comparison_lens_treemap_df(
                                df_by_label["label"],
                                df_by_label["improved_cnt"],
                                df_by_label["degraded_cnt"],
                                root_lens,
                            )
                            _plot_comparison_lens_treemap(
                                tdf_l,
                                f"p5_lens_lab_{lbl}_{idx}",
                                "By class",
                            )
                        else:
                            st.caption("_No label data._")
                        if not df_dataset_sorted.empty:
                            ds_cap = 36
                            ds_top = df_dataset_sorted.head(ds_cap).copy()
                            tdf_d = _comparison_lens_nested_treemap_df(
                                ds_top,
                                ["_scenario_focus", "_dataset_focus"],
                                root_lens,
                            )
                            rest = df_dataset_sorted.iloc[ds_cap:]
                            if not rest.empty:
                                io = float(rest["improved_cnt"].sum())
                                do = float(rest["degraded_cnt"].sum())
                                other_rows = []
                                for side, value in (("Improved", io), ("Degraded", do)):
                                    if value > 0:
                                        other_rows.append(
                                            {
                                                "root": root_lens,
                                                "side": side,
                                                "_scenario_focus": "Other scenarios",
                                                "_dataset_focus": f"Other datasets ({len(rest)})",
                                                "n": value,
                                            }
                                        )
                                if other_rows:
                                    tdf_d = pd.concat([tdf_d, pd.DataFrame(other_rows)], ignore_index=True)
                            _plot_comparison_lens_treemap(
                                tdf_d,
                                f"p5_lens_scen_ds_{lbl}_{idx}",
                                "By scenario",
                                path=["root", "side", "_scenario_focus", "_dataset_focus"],
                            )
                            st.caption(
                                f"Scenario view with top **{ds_cap}** datasets by {frame_caption_metric}, plus **Other datasets**."
                            )
                        else:
                            st.caption("_No scenario/dataset data._")
                        with st.expander("Tables behind the lens (label / scenario / dataset / frame)"):
                            if not df_by_label.empty:
                                st.markdown("**Per label**")
                                st.dataframe(
                                    df_by_label.drop(columns=_DIFF_INTERNAL_COLS, errors="ignore"),
                                    width='stretch',
                                    hide_index=True,
                                )
                            if not scen_agg.empty:
                                st.markdown("**Per scenario**")
                                st.dataframe(scen_agg, width='stretch', hide_index=True)
                            if not df_dataset_sorted.empty:
                                st.markdown(f"**Per dataset** (sorted by {frame_caption_metric})")
                                st.dataframe(
                                    _with_t4_viewer_links(
                                        df_dataset_sorted.head(200).drop(columns=_DIFF_INTERNAL_COLS, errors="ignore"),
                                        _t4_link_run_names,
                                    ),
                                    width='stretch',
                                    hide_index=True,
                                    column_config=_t4_viewer_link_column_config(),
                                )
                            if not df_frame_sorted.empty:
                                st.markdown(f"**Per frame** (sorted by {frame_caption_metric})")
                                st.dataframe(
                                    _with_t4_viewer_links(
                                        df_frame_sorted.head(200).drop(columns=_DIFF_INTERNAL_COLS, errors="ignore"),
                                        _t4_link_run_names,
                                    ),
                                    width='stretch',
                                    hide_index=True,
                                    column_config=_t4_viewer_link_column_config(),
                                )
    
                        # --- Drill-down: filters + objects ---
                        with st.expander("Drill-down: objects"):
                            scen_key = f"p5_scen_{lbl}_{idx}"
                            t4_key = f"p5_t4_{lbl}_{idx}"
                            lab_key = f"p5_lab_{lbl}_{idx}"
                            for k, default in ((scen_key, []), (t4_key, []), (lab_key, [])):
                                if k not in st.session_state:
                                    st.session_state[k] = default
    
                            scenarios_all = sorted(
                                df_improved["scenario_name"].dropna().astype(str).unique().tolist()
                            )
                            t4_all = sorted(
                                df_improved["t4dataset_name"].dropna().astype(str).unique().tolist()
                            )
                            labels_all = (
                                sorted(df_by_object_full["label"].dropna().astype(str).unique().tolist())
                                if not df_by_object_full.empty
                                else []
                            )
                            # Keep prior picks valid so Streamlit does not reset widgets when options refresh
                            scenarios_opts = sorted(
                                set(scenarios_all) | set(st.session_state.get(scen_key, []) or [])
                            )
                            t4_opts = sorted(set(t4_all) | set(st.session_state.get(t4_key, []) or []))
                            labels_opts = sorted(
                                set(labels_all) | set(st.session_state.get(lab_key, []) or [])
                            )
    
                            pr1, pr2 = st.columns(2)
                            with pr1:
                                if st.button(
                                    "Preset: top 5 degraded scenarios",
                                    key=f"p5_pre_scen_{lbl}_{idx}",
                                ):
                                    if not df_improved.empty:
                                        sa = (
                                            df_improved.groupby("scenario_name", dropna=False)[
                                                "degraded_cnt"
                                            ]
                                            .sum()
                                            .sort_values(ascending=False)
                                            .head(5)
                                        )
                                        st.session_state[scen_key] = [
                                            str(x) for x in sa.index.tolist()
                                        ]
                                        st.rerun()
                            fr_multiselect_key = f"p5_frkeys_{lbl}_{idx}"
                            if fr_multiselect_key not in st.session_state:
                                st.session_state[fr_multiselect_key] = []
                            frame_key_labels = {}
                            if not df_frame_sorted.empty:
                                for _, rw in df_frame_sorted.head(40).iterrows():
                                    fk = f"{rw['t4dataset_id']}|{rw['frame_index']}"
                                    # Use scenario_name (not suite_name) for frame option labels
                                    frame_key_labels[fk] = (
                                        f"{str(rw.get('scenario_name', ''))[:36]} | "
                                        f"f{rw['frame_index']} | deg {int(rw['degraded_cnt'])} | imp {int(rw['improved_cnt'])}"
                                    )
                            with pr2:
                                if st.button(
                                    f"Preset: top 10 {frame_caption_metric} frames (object filter)",
                                    key=f"p5_pre_fr_{lbl}_{idx}",
                                ):
                                    if frame_key_labels:
                                        topk = list(frame_key_labels.keys())[:10]
                                        st.session_state[fr_multiselect_key] = topk
                                        st.rerun()
    
                            colf1, colf2, colf3 = st.columns(3)
                            with colf1:
                                if scenarios_opts:
                                    st.multiselect(
                                        "Filter scenario_name",
                                        scenarios_opts,
                                        key=scen_key,
                                    )
                                else:
                                    st.caption("No scenarios.")
                            with colf2:
                                if t4_opts:
                                    st.multiselect(
                                        "Filter t4dataset_name",
                                        t4_opts,
                                        key=t4_key,
                                    )
                                else:
                                    st.caption("No t4dataset_name.")
                            with colf3:
                                if labels_opts:
                                    st.multiselect(
                                        "Filter label",
                                        labels_opts,
                                        key=lab_key,
                                    )
                                else:
                                    st.caption("No labels.")
    
                            prev_fr = st.session_state.get(fr_multiselect_key) or []
                            base_frame_keys = list(frame_key_labels.keys())
                            for k in prev_fr:
                                if k not in frame_key_labels:
                                    frame_key_labels[k] = f"(selected) frame {str(k).split('|')[-1]}"
                            frame_opts_keys = base_frame_keys + [
                                k for k in prev_fr if k not in base_frame_keys
                            ]
                            if frame_opts_keys:
                                st.multiselect(
                                    "Limit objects to frames (optional)",
                                    options=frame_opts_keys,
                                    format_func=lambda k: frame_key_labels.get(k, k),
                                    key=fr_multiselect_key,
                                )
    
                            change_type_filter = st.selectbox(
                                "Change type",
                                ["degraded", "improved", "all", "both_tp", "both_fn"],
                                key=f"change_type_{lbl}_{idx}",
                                help="Filter objects by TP change between runs.",
                            )
                            sort_obj = st.selectbox(
                                "Sort objects by",
                                [
                                    "degraded_priority_then_dist",
                                    "frame_then_uuid",
                                    "label_then_dist",
                                ],
                                key=f"p5_sort_{lbl}_{idx}",
                            )
    
                            df_obj_show = (
                                df_by_object_full.copy()
                                if not df_by_object_full.empty
                                else pd.DataFrame()
                            )
                            if not df_obj_show.empty:
                                ss = st.session_state.get(scen_key) or []
                                if ss:
                                    df_obj_show = df_obj_show[
                                        df_obj_show["scenario_name"].astype(str).isin(ss)
                                    ]
                                tt = st.session_state.get(t4_key) or []
                                if tt:
                                    df_obj_show = df_obj_show[
                                        df_obj_show["t4dataset_name"].astype(str).isin(tt)
                                    ]
                                ll = st.session_state.get(lab_key) or []
                                if ll:
                                    df_obj_show = df_obj_show[
                                        df_obj_show["label"].astype(str).isin(ll)
                                    ]
                                fk_sel = st.session_state.get(fr_multiselect_key) or []
                                if fk_sel:
                                    fk_set = set(fk_sel)
                                    df_obj_show = df_obj_show[
                                        (
                                            df_obj_show["t4dataset_id"].astype(str)
                                            + "|"
                                            + df_obj_show["frame_index"].astype(str)
                                        ).isin(fk_set)
                                    ]
                                if change_type_filter != "all":
                                    df_obj_show = df_obj_show[
                                        df_obj_show["change_type"] == change_type_filter
                                    ]
                                if sort_obj == "degraded_priority_then_dist":
                                    df_obj_show = df_obj_show.copy()
                                    df_obj_show["_prio"] = df_obj_show["change_type"].map(
                                        {
                                            "degraded": 0,
                                            "improved": 1,
                                            "both_tp": 2,
                                            "both_fn": 3,
                                        }
                                    )
                                    df_obj_show = df_obj_show.sort_values(
                                        by=["_prio", "dist_h"],
                                        ascending=[True, True],
                                    ).drop(columns=["_prio"], errors="ignore")
                                elif sort_obj == "frame_then_uuid":
                                    df_obj_show = df_obj_show.sort_values(
                                        by=["t4dataset_id", "frame_index", "gt_uuid"]
                                    )
                                else:
                                    df_obj_show = df_obj_show.sort_values(
                                        by=["label", "dist_h", "t4dataset_id", "frame_index"]
                                    )
    
                            n_show = 200
                            st.caption(
                                f"Showing up to {n_show} rows; use **Download CSV** for the full filtered list."
                            )
                            if not df_obj_show.empty:
                                df_obj_show_linked = _with_t4_viewer_links(
                                    df_obj_show,
                                    _t4_link_run_names,
                                )
                                st.download_button(
                                    label="Download filtered objects (CSV)",
                                    data=df_obj_show_linked.drop(columns=_DIFF_INTERNAL_COLS, errors="ignore").to_csv(index=False).encode("utf-8"),
                                    file_name=f"perception_diff_{lbl}_vs_A_objects.csv",
                                    mime="text/csv",
                                    key=f"p5_dl_{lbl}_{idx}",
                                )
                                st.dataframe(
                                    df_obj_show_linked.head(n_show).drop(columns=_DIFF_INTERNAL_COLS, errors="ignore"),
                                    width='stretch',
                                    hide_index=True,
                                    column_config=_t4_viewer_link_column_config(),
                                )
                            else:
                                st.caption("No objects match filters.")
    
                        with st.expander(f"Full frame table (sort: {frame_sort_desc})"):
                            if not df_frame_sorted.empty:
                                st.dataframe(
                                    _with_t4_viewer_links(
                                        df_frame_sorted.drop(columns=_DIFF_INTERNAL_COLS, errors="ignore"),
                                        _t4_link_run_names,
                                    ),
                                    width='stretch',
                                    hide_index=True,
                                    column_config=_t4_viewer_link_column_config(),
                                )
                            else:
                                st.caption("No frame breakdown.")
                else:
                    st.caption(f"Run {lbl} vs A: No data.")
            except Exception as e:
                st.error(f"Error (Run {lbl} vs A): {e}")
            finally:
                _pd_slot.empty()

    # =============================
    # Compare mode: Perception diff — False Positives (EST-side)
    # =============================
    if not single_mode:
        ds_dlog("section: Perception_FP_diff_start")
        st.divider()
        st.markdown(
            section_header_html(
                "Perception diff: False Positives (vs baseline A)",
                "Compares FP counts between runs (per dataset / frame / label). "
                "FP objects are EST-side detections and are not identity-matched across runs. "
                "Shows Baseline FP, Candidate FP, and Net FP change (candidate − baseline). "
                "Negative delta = FP reduced in candidate. Positive delta = FP increased in candidate.",
            ),
            unsafe_allow_html=True,
        )
        for idx in range(1, len(runs)):
            lbl = run_labels_list[idx]
            _fp_slot = st.empty()
            _fp_slot.markdown(ds_spot_loading_markup(f"FP diff · run {lbl}"), unsafe_allow_html=True)
            try:
                filter_clause_comp_fp = build_filter_clause(filters_list[idx], enable_dist_h=False)
                comp_flat = _flat_view(idx)
                # --- Per-dataset FP query ---
                # Aggregate FP/TP counts per dataset (no pair_uuid matching needed).
                # This avoids the issue where pair_uuid values differ between runs
                # because EST→GT matching is not deterministic across evaluations.
                query_fp = f"""
                WITH base_stats AS (
                    SELECT
                        t4dataset_id,
                        COUNT(*) AS total_est_base,
                        COUNT(*) FILTER (WHERE status = 'FP') AS fp_base,
                        COUNT(*) FILTER (WHERE status = 'TP') AS tp_base,
                        COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                        COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                        COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                    FROM view_eval_flat
                    WHERE source = 'EST' AND frame_index IS NOT NULL
                        AND {filter_clause_base}
                    GROUP BY 1
                ),
                comp_stats AS (
                    SELECT
                        t4dataset_id,
                        COUNT(*) AS total_est_comp,
                        COUNT(*) FILTER (WHERE status = 'FP') AS fp_comp,
                        COUNT(*) FILTER (WHERE status = 'TP') AS tp_comp,
                        COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                        COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                        COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                    FROM {comp_flat}
                    WHERE source = 'EST' AND frame_index IS NOT NULL
                        AND {filter_clause_comp_fp}
                    GROUP BY 1
                )
                SELECT
                    COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR)) AS t4dataset_id,
                    CAST(COALESCE(b.total_est_base, 0) + COALESCE(c.total_est_comp, 0) AS DOUBLE) AS total_est,
                    CAST(COALESCE(b.total_est_base, 0) AS DOUBLE) AS base_est_cnt,
                    CAST(COALESCE(c.total_est_comp, 0) AS DOUBLE) AS candidate_est_cnt,
                    CAST(CASE WHEN b.total_est_base IS NULL AND c.total_est_comp IS NOT NULL THEN c.total_est_comp ELSE 0 END AS DOUBLE) AS missing_in_base_cnt,
                    CAST(CASE WHEN b.total_est_base IS NOT NULL AND c.total_est_comp IS NULL THEN b.total_est_base ELSE 0 END AS DOUBLE) AS missing_in_candidate_cnt,
                    CAST(COALESCE(b.fp_base, 0) AS DOUBLE) AS baseline_fp,
                    CAST(COALESCE(c.fp_comp, 0) AS DOUBLE) AS candidate_fp,
                    CAST(COALESCE(c.fp_comp, 0) - COALESCE(b.fp_base, 0) AS DOUBLE) AS fp_delta,
                    COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
                    COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
                    COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name
                FROM base_stats b
                FULL OUTER JOIN comp_stats c
                    ON CAST(b.t4dataset_id AS VARCHAR) = CAST(c.t4dataset_id AS VARCHAR)
                ORDER BY fp_delta DESC
                """
                df_fp = con.execute(query_fp).df()
                if not df_fp.empty:
                    # --- Per-frame FP query ---
                    # Aggregate FP/TP counts per frame (no pair_uuid matching needed).
                    query_fp_frame = f"""
                    WITH base_stats AS (
                        SELECT
                            t4dataset_id,
                            frame_index,
                            COUNT(*) AS total_est_base,
                            COUNT(*) FILTER (WHERE status = 'FP') AS fp_base,
                            COUNT(*) FILTER (WHERE status = 'TP') AS tp_base,
                            COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                            COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                            COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                        FROM view_eval_flat
                        WHERE source = 'EST' AND frame_index IS NOT NULL
                            AND {filter_clause_base}
                        GROUP BY 1, 2
                    ),
                    comp_stats AS (
                        SELECT
                            t4dataset_id,
                            frame_index,
                            COUNT(*) AS total_est_comp,
                            COUNT(*) FILTER (WHERE status = 'FP') AS fp_comp,
                            COUNT(*) FILTER (WHERE status = 'TP') AS tp_comp,
                            COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                            COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                            COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                        FROM {comp_flat}
                        WHERE source = 'EST' AND frame_index IS NOT NULL
                            AND {filter_clause_comp_fp}
                        GROUP BY 1, 2
                    )
                    SELECT
                        COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR)) AS t4dataset_id,
                        COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index,
                        COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
                        COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
                        COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name,
                        CAST(COALESCE(b.total_est_base, 0) + COALESCE(c.total_est_comp, 0) AS DOUBLE) AS total_est,
                        CAST(COALESCE(b.total_est_base, 0) AS DOUBLE) AS base_est_cnt,
                        CAST(COALESCE(c.total_est_comp, 0) AS DOUBLE) AS candidate_est_cnt,
                        CAST(CASE WHEN b.total_est_base IS NULL AND c.total_est_comp IS NOT NULL THEN c.total_est_comp ELSE 0 END AS DOUBLE) AS missing_in_base_cnt,
                        CAST(CASE WHEN b.total_est_base IS NOT NULL AND c.total_est_comp IS NULL THEN b.total_est_base ELSE 0 END AS DOUBLE) AS missing_in_candidate_cnt,
                        CAST(COALESCE(b.fp_base, 0) AS DOUBLE) AS baseline_fp,
                        CAST(COALESCE(c.fp_comp, 0) AS DOUBLE) AS candidate_fp,
                        CAST(COALESCE(c.fp_comp, 0) - COALESCE(b.fp_base, 0) AS DOUBLE) AS fp_delta
                    FROM base_stats b
                    FULL OUTER JOIN comp_stats c
                        ON CAST(b.t4dataset_id AS VARCHAR) = CAST(c.t4dataset_id AS VARCHAR)
                       AND CAST(b.frame_index AS VARCHAR) = CAST(c.frame_index AS VARCHAR)
                    ORDER BY fp_delta DESC
                    """
                    try:
                        df_fp_frame = con.execute(query_fp_frame).df()
                    except Exception:
                        df_fp_frame = pd.DataFrame()

                    # --- Per-object FP query ---
                    # List individual EST detections from both runs side-by-side
                    # per (t4dataset_id, frame_index).  We cannot join on pair_uuid
                    # because EST→GT matching is not deterministic across runs.
                    # Instead we list all EST objects from each run with their FP/TP
                    # status, grouped by frame so users can compare manually.
                    query_fp_object = f"""
                    WITH base_objs AS (
                        SELECT
                            CAST(t4dataset_id AS VARCHAR) AS t4dataset_id,
                            CAST(frame_index AS VARCHAR) AS frame_index,
                            CAST(uuid AS VARCHAR) AS est_uuid,
                            CAST(pair_uuid AS VARCHAR) AS est_gt_uuid,
                            CAST(status AS VARCHAR) AS status,
                            COALESCE(CAST(label AS VARCHAR), '') AS label,
                            dist_h,
                            COALESCE(CAST(suite_name AS VARCHAR), '') AS suite_name,
                            COALESCE(CAST(scenario_name AS VARCHAR), '') AS scenario_name,
                            COALESCE(CAST(t4dataset_name AS VARCHAR), '') AS t4dataset_name
                        FROM view_eval_flat
                        WHERE source = 'EST' AND frame_index IS NOT NULL
                            AND {filter_clause_base}
                    ),
                    comp_objs AS (
                        SELECT
                            CAST(t4dataset_id AS VARCHAR) AS t4dataset_id,
                            CAST(frame_index AS VARCHAR) AS frame_index,
                            CAST(uuid AS VARCHAR) AS est_uuid,
                            CAST(pair_uuid AS VARCHAR) AS est_gt_uuid,
                            CAST(status AS VARCHAR) AS status,
                            COALESCE(CAST(label AS VARCHAR), '') AS label,
                            dist_h,
                            COALESCE(CAST(suite_name AS VARCHAR), '') AS suite_name,
                            COALESCE(CAST(scenario_name AS VARCHAR), '') AS scenario_name,
                            COALESCE(CAST(t4dataset_name AS VARCHAR), '') AS t4dataset_name
                        FROM {comp_flat}
                        WHERE source = 'EST' AND frame_index IS NOT NULL
                            AND {filter_clause_comp_fp}
                    ),
                    -- Per-frame FP/TP counts for change type classification
                    frame_stats AS (
                        SELECT
                            COALESCE(b.t4dataset_id, c.t4dataset_id) AS t4dataset_id,
                            COALESCE(b.frame_index, c.frame_index) AS frame_index,
                            COALESCE(b.fp_cnt, 0) AS fp_base,
                            COALESCE(c.fp_cnt, 0) AS fp_comp,
                            COALESCE(b.tp_cnt, 0) AS tp_base,
                            COALESCE(c.tp_cnt, 0) AS tp_comp,
                            COALESCE(b.total_cnt, 0) AS total_base,
                            COALESCE(c.total_cnt, 0) AS total_comp
                        FROM (
                            SELECT CAST(t4dataset_id AS VARCHAR) AS t4dataset_id,
                                   CAST(frame_index AS VARCHAR) AS frame_index,
                                   COUNT(*) AS total_cnt,
                                   COUNT(*) FILTER (WHERE status = 'FP') AS fp_cnt,
                                   COUNT(*) FILTER (WHERE status = 'TP') AS tp_cnt
                            FROM view_eval_flat
                            WHERE source = 'EST' AND frame_index IS NOT NULL
                                AND {filter_clause_base}
                            GROUP BY 1, 2
                        ) b
                        FULL OUTER JOIN (
                            SELECT CAST(t4dataset_id AS VARCHAR) AS t4dataset_id,
                                   CAST(frame_index AS VARCHAR) AS frame_index,
                                   COUNT(*) AS total_cnt,
                                   COUNT(*) FILTER (WHERE status = 'FP') AS fp_cnt,
                                   COUNT(*) FILTER (WHERE status = 'TP') AS tp_cnt
                            FROM {comp_flat}
                            WHERE source = 'EST' AND frame_index IS NOT NULL
                                AND {filter_clause_comp_fp}
                            GROUP BY 1, 2
                        ) c
                            ON b.t4dataset_id = c.t4dataset_id
                           AND b.frame_index = c.frame_index
                    ),
                    -- Union all EST objects from both runs, tagged with source
                    all_objs AS (
                        SELECT *, 'base' AS run_source FROM base_objs
                        UNION ALL
                        SELECT *, 'comp' AS run_source FROM comp_objs
                    )
                    SELECT
                        o.t4dataset_id,
                        o.frame_index,
                        o.est_uuid,
                        o.est_gt_uuid,
                        TRUE AS has_base_est,
                        TRUE AS has_candidate_est,
                        CAST(1 AS DOUBLE) AS base_est_cnt,
                        CAST(1 AS DOUBLE) AS candidate_est_cnt,
                        CAST(0 AS DOUBLE) AS missing_in_base_cnt,
                        CAST(0 AS DOUBLE) AS missing_in_candidate_cnt,
                        o.label,
                        COALESCE(o.dist_h, 0.0) AS dist_h,
                        {_DIST_BIN_CASE.replace("dist_h", "COALESCE(o.dist_h, 0.0)")} AS distance_bin,
                        o.suite_name,
                        o.scenario_name,
                        o.t4dataset_name,
                        CASE
                            WHEN fs.fp_base > fs.fp_comp THEN 'fp_improved'
                            WHEN fs.fp_comp > fs.fp_base THEN 'fp_degraded'
                            WHEN fs.fp_base > 0 AND fs.fp_comp > 0 THEN 'both_fp'
                            ELSE 'both_tp'
                        END AS change_type,
                        (fs.fp_base > 0) AS fp_base,
                        (fs.fp_comp > 0) AS fp_comp,
                        o.status,
                        o.run_source
                    FROM all_objs o
                    LEFT JOIN frame_stats fs
                        ON o.t4dataset_id = fs.t4dataset_id
                       AND o.frame_index = fs.frame_index
                    ORDER BY change_type, o.t4dataset_id, o.frame_index, o.run_source
                    """
                    try:
                        df_fp_object = con.execute(query_fp_object).df()
                    except Exception:
                        df_fp_object = pd.DataFrame()

                    # --- Availability check ---
                    availability_messages_fp = [
                        msg
                        for msg in [
                            _compare_availability_summary_fp(df_fp, unit="datasets"),
                        ]
                        if msg
                    ]
                    skip_incomplete_key_fp = f"p5fp_skip_incomplete_{lbl}_{idx}"
                    skip_dataset_compare_fp = True
                    if availability_messages_fp:
                        st.warning(
                            "Some datasets have EST result rows on only one side. "
                            + "; ".join(availability_messages_fp)
                            + ". These whole datasets can create artificial large improvements/degradations.",
                            icon="⚠️",
                        )
                        skip_dataset_compare_fp = st.checkbox(
                            "Skip datasets with EST data on only one side",
                            value=True,
                            key=skip_incomplete_key_fp,
                            help=(
                                "When enabled, FP diff charts/tables remove whole datasets where either baseline A "
                                "or the candidate has no EST rows after the active filters. Frame-level one-sided "
                                "differences inside a valid dataset are still compared."
                            ),
                        )
                    # --- Dataset name debug ---
                    with st.expander("🔍 Debug: unique dataset names (t4dataset_id / t4dataset_name, EST side)"):
                        debug_summary_fp = _dataset_name_debug_summary(
                            con, "view_eval_flat", filter_clause_base,
                            comp_flat, filter_clause_comp_fp,
                            source="EST",
                        )
                        st.markdown(debug_summary_fp)
                    # --- Pair UUID debug: sample pair_uuid values for overlapping datasets ---
                    with st.expander("🔍 Debug: pair_uuid samples for overlapping datasets (EST side)"):
                        pair_uuid_debug = _fp_pair_uuid_debug(
                            con, "view_eval_flat", filter_clause_base,
                            comp_flat, filter_clause_comp_fp,
                        )
                        st.markdown(pair_uuid_debug)
                    df_fp_skipped = pd.DataFrame()
                    if skip_dataset_compare_fp:
                        df_fp_skipped = df_fp[~_compare_availability_mask_fp(df_fp)].copy()
                        skipped_dataset_ids_fp = set(df_fp_skipped["t4dataset_id"].dropna().astype(str))
                        df_fp = df_fp[_compare_availability_mask_fp(df_fp)].copy()
                        if skipped_dataset_ids_fp:
                            df_fp_frame = df_fp_frame[
                                ~df_fp_frame["t4dataset_id"].astype(str).isin(skipped_dataset_ids_fp)
                            ].copy()
                            df_fp_object = df_fp_object[
                                ~df_fp_object["t4dataset_id"].astype(str).isin(skipped_dataset_ids_fp)
                            ].copy()
                        if df_fp.empty:
                            st.info(
                                "All FP datasets for this slice are one-sided after the active filters. "
                                "Disable the skip option above to inspect them."
                            )
                            continue

                    # --- KPI summary ---
                    # baseline_fp = baseline FP count, candidate_fp = candidate FP count
                    # fp_delta = candidate_FP - baseline_FP (negative = FP reduced)
                    tot_fp_base = float(df_fp["baseline_fp"].sum())
                    tot_fp_comp = float(df_fp["candidate_fp"].sum())
                    tot_fp_net = tot_fp_comp - tot_fp_base
                    net_fp_s = f"{int(tot_fp_net):+d}"

                    with st.expander(f"FP diff · Run {lbl} vs A", expanded=(len(runs) == 2)):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Baseline FP", int(tot_fp_base))
                        c2.metric("Candidate FP", int(tot_fp_comp))
                        c3.metric("Net FP change", net_fp_s, delta_color="inverse")
                        c4.caption("Negative = FP reduced in candidate. Positive = FP increased in candidate.")
                        st.markdown(
                            f"**FP Summary:** Baseline A had **{int(tot_fp_base)}** FPs, candidate has **{int(tot_fp_comp)}** FPs — "
                            f"delta **{net_fp_s}**."
                        )
                        st.caption(
                            "This is an aggregate FP count comparison, not a same-object state transition."
                        )
                        skipped_total_fp = len(df_fp_skipped)
                        if skipped_total_fp > 0:
                            with st.expander("Skipped one-sided FP datasets"):
                                st.caption(
                                    "These whole datasets were excluded because EST result rows exist on only one side "
                                    "after the active filters. Frames and objects are not skipped independently; they are removed "
                                    "only when their parent dataset is skipped."
                                )
                                if not df_fp_skipped.empty:
                                    skipped_fp_dataset = df_fp_skipped.copy()
                                    skipped_fp_dataset["skip_reason"] = _compare_availability_reason_fp(
                                        skipped_fp_dataset
                                    )
                                    st.markdown("**Skipped FP datasets**")
                                    st.download_button(
                                        label="Download skipped FP datasets (CSV)",
                                        data=skipped_fp_dataset.drop(columns=_FP_INTERNAL_COLS, errors="ignore").to_csv(index=False).encode("utf-8"),
                                        file_name=f"fp_diff_{lbl}_vs_A_skipped_datasets.csv",
                                        mime="text/csv",
                                        key=f"p5fp_dl_skip_dataset_{lbl}_{idx}",
                                    )
                                    st.dataframe(
                                        skipped_fp_dataset.head(200).drop(columns=_FP_INTERNAL_COLS, errors="ignore"),
                                        width='stretch',
                                        hide_index=True,
                                    )

                        # --- Hierarchy charts (Sunburst/Treemap) ---
                        fp_b_key = f"p5fp_baobab_{lbl}_{idx}"
                        c1b, c2b, c3b, c4b = st.columns([1, 1, 1, 1])
                        with c1b:
                            fp_baobab_viz = st.radio(
                                "Chart type",
                                ["Sunburst", "Treemap"],
                                horizontal=True,
                                key=f"{fp_b_key}_viz",
                            )
                        with c2b:
                            fp_baobab_ns = st.slider(
                                "Max scenarios",
                                min_value=5,
                                max_value=25,
                                value=15,
                                key=f"{fp_b_key}_ns",
                            )
                        with c3b:
                            fp_baobab_nd = st.slider(
                                "Max datasets / scenario",
                                min_value=5,
                                max_value=30,
                                value=12,
                                key=f"{fp_b_key}_nd",
                            )
                        with c4b:
                            fp_baobab_nf = st.slider(
                                "Max frames / dataset",
                                min_value=5,
                                max_value=20,
                                value=10,
                                key=f"{fp_b_key}_nf",
                            )
                        if df_fp_object.empty:
                            st.caption("No object-level rows for FP hierarchy.")
                        else:
                            treemap_path_cols = ["root", "scen_g", "dataset_display", "fr_display", "label"]
                            sunburst_path_cols = ["root", "scen_g", "dataset_display", "label"]
                            h_fp_imp = _baobab_hierarchy_from_objects(
                                df_fp_object,
                                "fp_improved",
                                f"FP reduced ({lbl} vs A)",
                                fp_baobab_ns,
                                fp_baobab_nd,
                                fp_baobab_nf,
                            )
                            h_fp_deg = _baobab_hierarchy_from_objects(
                                df_fp_object,
                                "fp_degraded",
                                f"FP increased ({lbl} vs A)",
                                fp_baobab_ns,
                                fp_baobab_nd,
                                fp_baobab_nf,
                            )
                            fp_pair_both = (not h_fp_imp.empty) and (not h_fp_deg.empty)
                            fp_plot_entries = []
                            for ct, hdf, cmap in (
                                ("fp_improved", h_fp_imp, IMPROVED_SCALE),
                                ("fp_degraded", h_fp_deg, DEGRADED_SCALE),
                            ):
                                if hdf.empty:
                                    fp_plot_entries.append((ct, None))
                                    continue
                                fp_change_label = "reduced" if ct == "fp_improved" else "increased"
                                title = f"{fp_baobab_viz}: FP {fp_change_label} frames (n = {int(hdf['n'].sum())} EST rows)"
                                if fp_baobab_viz == "Sunburst":
                                    hdf_plot = _sunburst_without_frame_layer(hdf)
                                    fig_b = px.sunburst(
                                        hdf_plot,
                                        path=sunburst_path_cols,
                                        values="n",
                                        color="n",
                                        color_continuous_scale=cmap,
                                        title=title,
                                    )
                                    h_sb = 480 if fp_pair_both else 620
                                    apply_chart_theme(fig_b, height=h_sb, margin=dict(t=36, l=4, r=4, b=4))
                                else:
                                    fig_b = px.treemap(
                                        hdf,
                                        path=treemap_path_cols,
                                        values="n",
                                        color="n",
                                        color_continuous_scale=cmap,
                                        title=title,
                                    )
                                    h_tr = 440 if fp_pair_both else 520
                                    apply_chart_theme(fig_b, height=h_tr, margin=dict(t=40, l=4, r=4, b=4))
                                fp_plot_entries.append((ct, fig_b))

                            fp_two_up = (
                                len(fp_plot_entries) == 2
                                and fp_plot_entries[0][1] is not None
                                and fp_plot_entries[1][1] is not None
                            )
                            if fp_two_up:
                                bc1, bc2 = st.columns(2, gap="small")
                                with bc1:
                                    st.plotly_chart(
                                        fp_plot_entries[0][1],
                                        width='stretch',
                                        key=f"{fp_b_key}_fig_{fp_plot_entries[0][0]}",
                                    )
                                with bc2:
                                    st.plotly_chart(
                                        fp_plot_entries[1][1],
                                        width='stretch',
                                        key=f"{fp_b_key}_fig_{fp_plot_entries[1][0]}",
                                    )
                            else:
                                for ct, fig_b in fp_plot_entries:
                                    if fig_b is not None:
                                        st.plotly_chart(
                                            fig_b,
                                            width='stretch',
                                            key=f"{fp_b_key}_fig_{ct}",
                                        )
                                    else:
                                        fp_missing_label = "FP reduced" if ct == "fp_improved" else "FP increased"
                                        st.caption(f"No **{fp_missing_label}** frames to chart.")

                        # --- Comparison lens: label / scenario / dataset / frame ---
                        query_fp_label = f"""
                        WITH base_stats AS (
                            SELECT
                                COALESCE(MAX(try_cast(label AS VARCHAR)), '') AS label,
                                COUNT(*) AS total_est_base,
                                COUNT(*) FILTER (WHERE status = 'FP') AS fp_base,
                                COUNT(*) FILTER (WHERE status = 'TP') AS tp_base
                            FROM view_eval_flat
                            WHERE source = 'EST' AND frame_index IS NOT NULL
                                AND {filter_clause_base}
                            GROUP BY label
                        ),
                        comp_stats AS (
                            SELECT
                                COALESCE(MAX(try_cast(label AS VARCHAR)), '') AS label,
                                COUNT(*) AS total_est_comp,
                                COUNT(*) FILTER (WHERE status = 'FP') AS fp_comp,
                                COUNT(*) FILTER (WHERE status = 'TP') AS tp_comp
                            FROM {comp_flat}
                            WHERE source = 'EST' AND frame_index IS NOT NULL
                                AND {filter_clause_comp_fp}
                            GROUP BY label
                        )
                        SELECT
                            COALESCE(b.label, c.label) AS label,
                            CAST(COALESCE(b.total_est_base, 0) + COALESCE(c.total_est_comp, 0) AS DOUBLE) AS total_est,
                            CAST(COALESCE(b.total_est_base, 0) AS DOUBLE) AS base_est_cnt,
                            CAST(COALESCE(c.total_est_comp, 0) AS DOUBLE) AS candidate_est_cnt,
                            CAST(CASE WHEN b.total_est_base IS NULL AND c.total_est_comp IS NOT NULL THEN c.total_est_comp ELSE 0 END AS DOUBLE) AS missing_in_base_cnt,
                            CAST(CASE WHEN b.total_est_base IS NOT NULL AND c.total_est_comp IS NULL THEN b.total_est_base ELSE 0 END AS DOUBLE) AS missing_in_candidate_cnt,
                            CAST(COALESCE(b.fp_base, 0) AS DOUBLE) AS baseline_fp,
                            CAST(COALESCE(c.fp_comp, 0) AS DOUBLE) AS candidate_fp,
                            CAST(COALESCE(c.fp_comp, 0) - COALESCE(b.fp_base, 0) AS DOUBLE) AS fp_delta
                        FROM base_stats b
                        FULL OUTER JOIN comp_stats c
                            ON b.label = c.label
                        ORDER BY fp_delta DESC
                        """
                        df_fp_label = pd.DataFrame()
                        try:
                            df_fp_label = con.execute(query_fp_label).df()
                            # fp_delta is already computed in the query as candidate_FP - baseline_FP
                            # The query does a FULL OUTER JOIN on label, so it correctly shows
                            # labels from both runs with one-sided indicators.
                        except Exception as e_fp_label:
                            st.caption(f"FP Label query: {e_fp_label}")

                        fp_scen_agg = pd.DataFrame()
                        if not df_fp.empty:
                            fp_scen_agg = (
                                df_fp.groupby("scenario_name", dropna=False)
                                .agg(
                                    baseline_fp=("baseline_fp", "sum"),
                                    candidate_fp=("candidate_fp", "sum"),
                                )
                                .reset_index()
                            )
                            fp_scen_agg["fp_delta"] = fp_scen_agg["candidate_fp"] - fp_scen_agg["baseline_fp"]
                            fp_scen_agg = fp_scen_agg.sort_values(
                                by=["candidate_fp", "baseline_fp"],
                                ascending=[False, True],
                            )

                        fp_frame_sort_mode = st.radio(
                            "Dataset/frame focus",
                            ["FP increased first", "FP reduced first", "Largest net change"],
                            horizontal=True,
                            key=f"p5fp_frame_focus_{lbl}_{idx}",
                            help="Choose whether dataset and frame views prioritize new FPs, resolved FPs, or the biggest overall swings.",
                        )
                        df_fp_dataset_sorted = pd.DataFrame()
                        df_fp_frame_sorted = pd.DataFrame()
                        fp_frame_caption_metric = "FP increased"
                        fp_frame_sort_desc = "candidate_fp desc"
                        if not df_fp.empty:
                            df_fp_dataset_sorted = df_fp.copy()
                            dataset_name = df_fp_dataset_sorted.get(
                                "t4dataset_name",
                                df_fp_dataset_sorted["t4dataset_id"],
                            )
                            df_fp_dataset_sorted["_scenario_focus"] = (
                                df_fp_dataset_sorted["scenario_name"].fillna("").astype(str).replace("", "(no scenario)")
                            )
                            df_fp_dataset_sorted["_dataset_focus"] = dataset_name.fillna("").astype(str)
                            df_fp_dataset_sorted["_dataset_focus"] = df_fp_dataset_sorted["_dataset_focus"].where(
                                df_fp_dataset_sorted["_dataset_focus"].str.strip() != "",
                                df_fp_dataset_sorted["t4dataset_id"].fillna("").astype(str),
                            )
                            if fp_frame_sort_mode == "FP reduced first":
                                # baseline_fp = baseline FP, candidate_fp = candidate FP
                                # Sort by largest baseline FP first (potential FP reduction)
                                df_fp_dataset_sorted = df_fp_dataset_sorted.sort_values(
                                    by=["baseline_fp", "candidate_fp"],
                                    ascending=[False, True],
                                )
                            elif fp_frame_sort_mode == "Largest net change":
                                # fp_delta = candidate_FP - baseline_FP (from query)
                                # For "largest net change", sort by absolute delta
                                df_fp_dataset_sorted["_abs_fp_delta"] = (
                                    df_fp_dataset_sorted["fp_delta"].abs()
                                )
                                df_fp_dataset_sorted = df_fp_dataset_sorted.sort_values(
                                    by=["_abs_fp_delta", "candidate_fp", "baseline_fp"],
                                    ascending=[False, False, False],
                                )
                            else:
                                df_fp_dataset_sorted = df_fp_dataset_sorted.sort_values(
                                    by=["candidate_fp", "baseline_fp"],
                                    ascending=[False, True],
                                )
                            df_fp_dataset_sorted = df_fp_dataset_sorted.drop(
                                columns=["_abs_fp_delta"],
                                errors="ignore",
                            ).reset_index(drop=True)
                        if not df_fp_frame.empty:
                            df_fp_frame_sorted = df_fp_frame.copy()
                            frame_dataset_name = df_fp_frame_sorted.get(
                                "t4dataset_name",
                                df_fp_frame_sorted["t4dataset_id"],
                            )
                            df_fp_frame_sorted["_scenario_focus"] = (
                                df_fp_frame_sorted["scenario_name"].fillna("").astype(str).replace("", "(no scenario)")
                            )
                            df_fp_frame_sorted["_dataset_focus"] = frame_dataset_name.fillna("").astype(str)
                            df_fp_frame_sorted["_dataset_focus"] = df_fp_frame_sorted["_dataset_focus"].where(
                                df_fp_frame_sorted["_dataset_focus"].str.strip() != "",
                                df_fp_frame_sorted["t4dataset_id"].fillna("").astype(str),
                            )
                            if fp_frame_sort_mode == "FP reduced first":
                                fp_frame_caption_metric = "FP reduced"
                                fp_frame_sort_desc = "baseline_fp desc"
                                df_fp_frame_sorted = df_fp_frame_sorted.sort_values(
                                    by=["baseline_fp", "candidate_fp"],
                                    ascending=[False, True],
                                )
                            elif fp_frame_sort_mode == "Largest net change":
                                fp_frame_caption_metric = "absolute net change"
                                fp_frame_sort_desc = "largest |FP delta|"
                                df_fp_frame_sorted["_abs_fp_delta"] = (
                                    df_fp_frame_sorted["fp_delta"].abs()
                                )
                                df_fp_frame_sorted = df_fp_frame_sorted.sort_values(
                                    by=["_abs_fp_delta", "candidate_fp", "baseline_fp"],
                                    ascending=[False, False, False],
                                )
                            else:
                                df_fp_frame_sorted = df_fp_frame_sorted.sort_values(
                                    by=["candidate_fp", "baseline_fp"],
                                    ascending=[False, True],
                                )
                            df_fp_frame_sorted = df_fp_frame_sorted.drop(
                                columns=["_abs_fp_delta"],
                                errors="ignore",
                            ).reset_index(drop=True)
                        _t4_link_run_names = _run_share_names_for_links()

                        root_lens_fp = f"FP {lbl} vs A"
                        if not df_fp_label.empty:
                            tdf_fp_l = _comparison_lens_treemap_df(
                                df_fp_label["label"],
                                df_fp_label["baseline_fp"],
                                df_fp_label["candidate_fp"],
                                root_lens_fp,
                                side_labels=("Baseline FP", "Candidate FP"),
                            )
                            _plot_comparison_lens_treemap(
                                tdf_fp_l,
                                f"p5fp_lens_lab_{lbl}_{idx}",
                                "By class (FP)",
                            )
                        else:
                            st.caption("_No FP label data._")
                        if not df_fp_dataset_sorted.empty:
                            ds_cap = 36
                            ds_top = df_fp_dataset_sorted.head(ds_cap).copy()
                            # _comparison_lens_nested_treemap_df expects improved_cnt / degraded_cnt columns
                            ds_top_renamed = ds_top.rename(
                                columns={"baseline_fp": "improved_cnt", "candidate_fp": "degraded_cnt"}
                            )
                            tdf_fp_d = _comparison_lens_nested_treemap_df(
                                ds_top_renamed,
                                ["_scenario_focus", "_dataset_focus"],
                                root_lens_fp,
                                side_labels=("Baseline FP", "Candidate FP"),
                            )
                            rest = df_fp_dataset_sorted.iloc[ds_cap:]
                            if not rest.empty:
                                io = float(rest["baseline_fp"].sum())
                                do = float(rest["candidate_fp"].sum())
                                other_rows = []
                                for side, value in (("Baseline FP", io), ("Candidate FP", do)):
                                    if value > 0:
                                        other_rows.append(
                                            {
                                                "root": root_lens_fp,
                                                "side": side,
                                                "_scenario_focus": "Other scenarios",
                                                "_dataset_focus": f"Other datasets ({len(rest)})",
                                                "n": value,
                                            }
                                        )
                                if other_rows:
                                    tdf_fp_d = pd.concat([tdf_fp_d, pd.DataFrame(other_rows)], ignore_index=True)
                            _plot_comparison_lens_treemap(
                                tdf_fp_d,
                                f"p5fp_lens_scen_ds_{lbl}_{idx}",
                                "By scenario (FP)",
                                path=["root", "side", "_scenario_focus", "_dataset_focus"],
                            )
                            st.caption(
                                f"Scenario view with top **{ds_cap}** datasets by {fp_frame_caption_metric}, plus **Other datasets**."
                            )
                        else:
                            st.caption("_No FP scenario/dataset data._")
                        with st.expander("Tables behind the FP lens (label / scenario / dataset / frame)"):
                            if not df_fp_label.empty:
                                st.markdown("**Per label (FP)**")
                                st.dataframe(
                                    df_fp_label.drop(columns=_FP_INTERNAL_COLS, errors="ignore"),
                                    width='stretch',
                                    hide_index=True,
                                )
                            if not fp_scen_agg.empty:
                                st.markdown("**Per scenario (FP)**")
                                st.dataframe(fp_scen_agg, width='stretch', hide_index=True)
                            if not df_fp_dataset_sorted.empty:
                                st.markdown(f"**Per dataset (FP)** (sorted by {fp_frame_caption_metric})")
                                st.dataframe(
                                    _with_t4_viewer_links(
                                        df_fp_dataset_sorted.head(200).drop(columns=_FP_INTERNAL_COLS, errors="ignore"),
                                        _t4_link_run_names,
                                    ),
                                    width='stretch',
                                    hide_index=True,
                                    column_config=_t4_viewer_link_column_config(),
                                )
                            if not df_fp_frame_sorted.empty:
                                st.markdown(f"**Per frame (FP)** (sorted by {fp_frame_caption_metric})")
                                st.dataframe(
                                    _with_t4_viewer_links(
                                        df_fp_frame_sorted.head(200).drop(columns=_FP_INTERNAL_COLS, errors="ignore"),
                                        _t4_link_run_names,
                                    ),
                                    width='stretch',
                                    hide_index=True,
                                    column_config=_t4_viewer_link_column_config(),
                                )

                        # --- FP Drill-down: filters + objects ---
                        with st.expander("Drill-down: FP objects"):
                            fp_scen_key = f"p5fp_scen_{lbl}_{idx}"
                            fp_t4_key = f"p5fp_t4_{lbl}_{idx}"
                            fp_lab_key = f"p5fp_lab_{lbl}_{idx}"
                            for k, default in ((fp_scen_key, []), (fp_t4_key, []), (fp_lab_key, [])):
                                if k not in st.session_state:
                                    st.session_state[k] = default

                            fp_scenarios_all = sorted(
                                df_fp["scenario_name"].dropna().astype(str).unique().tolist()
                            )
                            fp_t4_all = sorted(
                                df_fp["t4dataset_name"].dropna().astype(str).unique().tolist()
                            )
                            fp_labels_all = (
                                sorted(df_fp_object["label"].dropna().astype(str).unique().tolist())
                                if not df_fp_object.empty
                                else []
                            )
                            fp_scenarios_opts = sorted(
                                set(fp_scenarios_all) | set(st.session_state.get(fp_scen_key, []) or [])
                            )
                            fp_t4_opts = sorted(set(fp_t4_all) | set(st.session_state.get(fp_t4_key, []) or []))
                            fp_labels_opts = sorted(
                                set(fp_labels_all) | set(st.session_state.get(fp_lab_key, []) or [])
                            )

                            pr1, pr2 = st.columns(2)
                            with pr1:
                                if st.button(
                                    "Preset: top 5 FP increased scenarios",
                                    key=f"p5fp_pre_scen_{lbl}_{idx}",
                                ):
                                    if not df_fp.empty:
                                        sa = (
                                            df_fp.groupby("scenario_name", dropna=False)[
                                                "candidate_fp"
                                            ]
                                            .sum()
                                            .sort_values(ascending=False)
                                            .head(5)
                                        )
                                        st.session_state[fp_scen_key] = [
                                            str(x) for x in sa.index.tolist()
                                        ]
                                        st.rerun()
                            fp_fr_multiselect_key = f"p5fp_frkeys_{lbl}_{idx}"
                            if fp_fr_multiselect_key not in st.session_state:
                                st.session_state[fp_fr_multiselect_key] = []
                            fp_frame_key_labels = {}
                            if not df_fp_frame_sorted.empty:
                                for _, rw in df_fp_frame_sorted.head(40).iterrows():
                                    fk = f"{rw['t4dataset_id']}|{rw['frame_index']}"
                                    fp_frame_key_labels[fk] = (
                                        f"{str(rw.get('scenario_name', ''))[:36]} | "
                                        f"f{rw['frame_index']} | candidate FP {int(rw['candidate_fp'])} | baseline FP {int(rw['baseline_fp'])}"
                                    )
                            with pr2:
                                if st.button(
                                    f"Preset: top 10 {fp_frame_caption_metric} frames (FP object filter)",
                                    key=f"p5fp_pre_fr_{lbl}_{idx}",
                                ):
                                    if fp_frame_key_labels:
                                        topk = list(fp_frame_key_labels.keys())[:10]
                                        st.session_state[fp_fr_multiselect_key] = topk
                                        st.rerun()

                            colf1, colf2, colf3 = st.columns(3)
                            with colf1:
                                if fp_scenarios_opts:
                                    st.multiselect(
                                        "Filter scenario_name",
                                        fp_scenarios_opts,
                                        key=fp_scen_key,
                                    )
                                else:
                                    st.caption("No scenarios.")
                            with colf2:
                                if fp_t4_opts:
                                    st.multiselect(
                                        "Filter t4dataset_name",
                                        fp_t4_opts,
                                        key=fp_t4_key,
                                    )
                                else:
                                    st.caption("No t4dataset_name.")
                            with colf3:
                                if fp_labels_opts:
                                    st.multiselect(
                                        "Filter label",
                                        fp_labels_opts,
                                        key=fp_lab_key,
                                    )
                                else:
                                    st.caption("No labels.")

                            prev_fr_fp = st.session_state.get(fp_fr_multiselect_key) or []
                            base_fp_frame_keys = list(fp_frame_key_labels.keys())
                            for k in prev_fr_fp:
                                if k not in fp_frame_key_labels:
                                    fp_frame_key_labels[k] = f"(selected) frame {str(k).split('|')[-1]}"
                            fp_frame_opts_keys = base_fp_frame_keys + [
                                k for k in prev_fr_fp if k not in base_fp_frame_keys
                            ]
                            if fp_frame_opts_keys:
                                st.multiselect(
                                    "Limit objects to frames (optional)",
                                    options=fp_frame_opts_keys,
                                    format_func=lambda k: fp_frame_key_labels.get(k, k),
                                    key=fp_fr_multiselect_key,
                                )

                            fp_change_type_filter = st.selectbox(
                                "Change type",
                                ["fp_degraded", "fp_improved", "all", "both_fp", "both_tp"],
                                key=f"fp_change_type_{lbl}_{idx}",
                                help="Filter EST objects by FP change between runs.",
                                format_func=lambda v: {
                                    "fp_degraded": "FP increased",
                                    "fp_improved": "FP reduced",
                                    "both_fp": "FP exists in both runs",
                                    "both_tp": "No FP in frame",
                                    "all": "All",
                                }.get(v, v),
                            )
                            fp_sort_obj = st.selectbox(
                                "Sort objects by",
                                [
                                    "fp_degraded_priority_then_dist",
                                    "frame_then_uuid",
                                    "label_then_dist",
                                ],
                                key=f"p5fp_sort_{lbl}_{idx}",
                                format_func=lambda v: {
                                    "fp_degraded_priority_then_dist": "FP increased first, then distance",
                                    "frame_then_uuid": "Frame, then EST uuid",
                                    "label_then_dist": "Label, then distance",
                                }.get(v, v),
                            )

                            df_fp_obj_show = (
                                df_fp_object.copy()
                                if not df_fp_object.empty
                                else pd.DataFrame()
                            )
                            if not df_fp_obj_show.empty:
                                ss = st.session_state.get(fp_scen_key) or []
                                if ss:
                                    df_fp_obj_show = df_fp_obj_show[
                                        df_fp_obj_show["scenario_name"].astype(str).isin(ss)
                                    ]
                                tt = st.session_state.get(fp_t4_key) or []
                                if tt:
                                    df_fp_obj_show = df_fp_obj_show[
                                        df_fp_obj_show["t4dataset_name"].astype(str).isin(tt)
                                    ]
                                ll = st.session_state.get(fp_lab_key) or []
                                if ll:
                                    df_fp_obj_show = df_fp_obj_show[
                                        df_fp_obj_show["label"].astype(str).isin(ll)
                                    ]
                                fk_sel = st.session_state.get(fp_fr_multiselect_key) or []
                                if fk_sel:
                                    fk_set = set(fk_sel)
                                    df_fp_obj_show = df_fp_obj_show[
                                        (
                                            df_fp_obj_show["t4dataset_id"].astype(str)
                                            + "|"
                                            + df_fp_obj_show["frame_index"].astype(str)
                                        ).isin(fk_set)
                                    ]
                                if fp_change_type_filter != "all":
                                    df_fp_obj_show = df_fp_obj_show[
                                        df_fp_obj_show["change_type"] == fp_change_type_filter
                                    ]
                                if fp_sort_obj == "fp_degraded_priority_then_dist":
                                    df_fp_obj_show = df_fp_obj_show.copy()
                                    df_fp_obj_show["_prio"] = df_fp_obj_show["change_type"].map(
                                        {
                                            "fp_degraded": 0,
                                            "fp_improved": 1,
                                            "both_fp": 2,
                                            "both_tp": 3,
                                        }
                                    )
                                    df_fp_obj_show = df_fp_obj_show.sort_values(
                                        by=["_prio", "dist_h"],
                                        ascending=[True, True],
                                    ).drop(columns=["_prio"], errors="ignore")
                                elif fp_sort_obj == "frame_then_uuid":
                                    df_fp_obj_show = df_fp_obj_show.sort_values(
                                        by=["t4dataset_id", "frame_index", "est_uuid"]
                                    )
                                else:
                                    df_fp_obj_show = df_fp_obj_show.sort_values(
                                        by=["label", "dist_h", "t4dataset_id", "frame_index"]
                                    )

                            n_show = 200
                            st.caption(
                                f"Showing up to {n_show} rows; use **Download CSV** for the full filtered list."
                            )
                            if not df_fp_obj_show.empty:
                                df_fp_obj_show_linked = _with_t4_viewer_links(
                                    df_fp_obj_show,
                                    _t4_link_run_names,
                                )
                                st.download_button(
                                    label="Download filtered FP objects (CSV)",
                                    data=df_fp_obj_show_linked.drop(columns=_FP_INTERNAL_COLS, errors="ignore").to_csv(index=False).encode("utf-8"),
                                    file_name=f"fp_diff_{lbl}_vs_A_objects.csv",
                                    mime="text/csv",
                                    key=f"p5fp_dl_{lbl}_{idx}",
                                )
                                st.dataframe(
                                    df_fp_obj_show_linked.head(n_show).drop(columns=_FP_INTERNAL_COLS, errors="ignore"),
                                    width='stretch',
                                    hide_index=True,
                                    column_config=_t4_viewer_link_column_config(),
                                )
                            else:
                                st.caption("No FP objects match filters.")

                        with st.expander(f"Full FP frame table (sort: {fp_frame_sort_desc})"):
                            if not df_fp_frame_sorted.empty:
                                st.dataframe(
                                    _with_t4_viewer_links(
                                        df_fp_frame_sorted.drop(columns=_FP_INTERNAL_COLS, errors="ignore"),
                                        _t4_link_run_names,
                                    ),
                                    width='stretch',
                                    hide_index=True,
                                    column_config=_t4_viewer_link_column_config(),
                                )
                            else:
                                st.caption("No FP frame breakdown.")
                else:
                    st.caption(f"FP diff · Run {lbl} vs A: No data.")
            except Exception as e:
                st.error(f"Error (FP diff · Run {lbl} vs A): {e}")
            finally:
                _fp_slot.empty()
    
    # =============================
    # Single mode: Frame / Object level — Where are the misses?
    # =============================
    if single_mode:
        ds_dlog("section: Frame_FN_misses_start")
        st.markdown(section_header_html("Frame / Object level: Where are the misses?"), unsafe_allow_html=True)
        _fn_slot = st.empty()
        _fn_slot.markdown(ds_spot_loading_markup("FN by frame & object"), unsafe_allow_html=True)
        try:
            with st.expander("FN by frame and by object", expanded=True):
                query_fn_frame = f"""
                SELECT
                    t4dataset_id,
                    frame_index,
                    COALESCE(MAX(CAST(scenario_name AS VARCHAR)), '') AS scenario_name,
                    COALESCE(MAX(CAST(suite_name AS VARCHAR)), '') AS suite_name,
                    COALESCE(MAX(CAST(t4dataset_name AS VARCHAR)), '') AS t4dataset_name,
                    COUNT(*) AS fn_cnt
                FROM view_eval_flat
                WHERE source = 'GT' AND status = 'FN' AND {filter_clause_base}
                GROUP BY t4dataset_id, frame_index
                ORDER BY fn_cnt DESC
                """
                df_fn_frame = con.execute(query_fn_frame).df()
                query_fn_object = f"""
                SELECT
                    t4dataset_id,
                    frame_index,
                    uuid,
                    COALESCE(CAST(label AS VARCHAR), '') AS label,
                    dist_h,
                    COALESCE(CAST(scenario_name AS VARCHAR), '') AS scenario_name,
                    COALESCE(CAST(suite_name AS VARCHAR), '') AS suite_name
                FROM view_eval_flat
                WHERE source = 'GT' AND status = 'FN' AND {filter_clause_base}
                ORDER BY t4dataset_id, frame_index, uuid
                """
                df_fn_object = con.execute(query_fn_object).df()
                if not df_fn_frame.empty:
                    st.markdown("**FN count by frame**")
                    st.download_button("Download FN by frame (CSV)", data=df_fn_frame.to_csv(index=False).encode("utf-8"), file_name="fn_by_frame.csv", mime="text/csv", key="dl_fn_frame")
                    st.dataframe(df_fn_frame, width='stretch', hide_index=True)
                else:
                    st.caption("No FN by frame.")
                if not df_fn_object.empty:
                    st.markdown("**FN objects**")
                    if len(df_fn_object) > 500:
                        st.caption(f"Showing first 500 of {len(df_fn_object)} FN objects.")
                        st.dataframe(df_fn_object.head(500), width='stretch', hide_index=True)
                    else:
                        st.dataframe(df_fn_object, width='stretch', hide_index=True)
                else:
                    st.caption("No FN objects.")
        except Exception as e:
            st.error(f"Error in FN by frame/object: {e}")
        finally:
            _fn_slot.empty()
    
    # =============================
    # Panel 6: Mean Error (single) / Mean Error Comparison (compare)
    # =============================
    ds_dlog("section: Panel6_Mean_Error_start")
    st.divider()
    st.markdown(
        section_header_html(
            "Mean Error" + (" Comparison" if not single_mode else ""),
            "Mean absolute error on TP matches (X/Y in m, Yaw in rad)."
            + (" Compare mode: choose grouped bars or spider charts." if not single_mode else ""),
        ),
        unsafe_allow_html=True,
    )
    
    try:
        sample_query = "SELECT * FROM view_eval_flat LIMIT 1"
        sample_df = con.execute(sample_query).df()
        has_error_cols = all(col in sample_df.columns for col in ['x_error', 'y_error', 'yaw_error'])
    except Exception:
        has_error_cols = False
    
    if not has_error_cols:
        st.info("Error columns (x_error, y_error, yaw_error) not found in data. Skipping error analysis.")
    else:
        if single_mode:
            try:
                with ds_spot_loading("Mean error"):
                    query = f"""
                    SELECT
                        label,
                        AVG(ABS(CAST(x_error AS DOUBLE))) FILTER (
                            WHERE status = 'TP' AND x_error IS NOT NULL
                        ) AS mean_abs_x_error,
                        AVG(ABS(CAST(y_error AS DOUBLE))) FILTER (
                            WHERE status = 'TP' AND y_error IS NOT NULL
                        ) AS mean_abs_y_error,
                        AVG(ABS(CAST(yaw_error AS DOUBLE))) FILTER (
                            WHERE status = 'TP' AND yaw_error IS NOT NULL
                        ) AS mean_abs_yaw_error
                    FROM view_eval_flat
                    WHERE {filter_clause_base}
                    GROUP BY label
                    ORDER BY label
                    """
                    df_error_base = con.execute(query).df()
                if not df_error_base.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_error_base['label'],
                        y=df_error_base['mean_abs_x_error'],
                        name='X Error',
                        marker_color=RUN_COLORS[0],
                    ))
                    fig.add_trace(go.Bar(
                        x=df_error_base['label'],
                        y=df_error_base['mean_abs_y_error'],
                        name='Y Error',
                        marker_color=RUN_COLORS[1],
                    ))
                    fig.add_trace(go.Bar(
                        x=df_error_base['label'],
                        y=df_error_base['mean_abs_yaw_error'],
                        name='Yaw Error',
                        marker_color=RUN_COLORS[2],
                    ))
                    apply_chart_theme(fig)
                    fig.update_layout(
                        title=f"Mean Error within {max_eval_range} [m]",
                        xaxis_title="Label",
                        yaxis_title="Error [m] or [rad]",
                        barmode='group'
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No data available")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            try:
                with ds_spot_loading("Mean error"):
                    dfs_err = []
                    for i in range(len(runs)):
                        fc = build_filter_clause(filters_list[i])
                        q = f"""
                        SELECT
                            label,
                            AVG(ABS(CAST(x_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND x_error IS NOT NULL) AS mean_abs_x_error,
                            AVG(ABS(CAST(y_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND y_error IS NOT NULL) AS mean_abs_y_error,
                            AVG(ABS(CAST(yaw_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND yaw_error IS NOT NULL) AS mean_abs_yaw_error
                        FROM {_flat_view(i)}
                        WHERE {fc}
                        GROUP BY label
                        ORDER BY label
                        """
                        df_i = con.execute(q).df()
                        df_i["run"] = run_labels_list[i]
                        dfs_err.append(df_i)
                    df_err_melt = pd.concat(dfs_err, ignore_index=True)
                if not df_err_melt.empty:
                    mean_err_viz = st.radio(
                        "Mean error chart style",
                        options=["Spider chart (X, Y & Yaw)", "Grouped bar"],
                        index=0,
                        horizontal=True,
                        key="mean_err_compare_viz",
                    )
                    if mean_err_viz == "Grouped bar":
                        for err_type, col in [
                            ("X Error", "mean_abs_x_error"),
                            ("Y Error", "mean_abs_y_error"),
                            ("Yaw Error", "mean_abs_yaw_error"),
                        ]:
                            fig = px.bar(
                                df_err_melt,
                                x="label",
                                y=col,
                                color="run",
                                barmode="group",
                                title=f"Mean {err_type} within {max_eval_range} [m] by run",
                                labels={"label": "Label", col: err_type, "run": "Run"},
                                color_discrete_sequence=RUN_COLORS,
                            )
                            apply_chart_theme(fig)
                            st.plotly_chart(fig, width="stretch")
                    else:
                        st.caption(
                            f"Three spiders: mean |error| per label per run (TP only), within **{max_eval_range} m** "
                            "(same as sidebar max range)."
                        )
                        cats = sorted(df_err_melt["label"].astype(str).unique())
                        if len(cats) > 16:
                            st.caption("Spider charts work best with ≤16 labels; many classes may look crowded.")
                        rcols = st.columns(3)
                        err_specs = [
                            (
                                f"Mean |x error| (within {max_eval_range} m)",
                                "mean_abs_x_error",
                                "Mean |x error| (m)",
                                ".3f",
                            ),
                            (
                                f"Mean |y error| (within {max_eval_range} m)",
                                "mean_abs_y_error",
                                "Mean |y error| (m)",
                                ".3f",
                            ),
                            (
                                f"Mean |yaw error| (within {max_eval_range} m)",
                                "mean_abs_yaw_error",
                                "Mean |yaw error| (rad)",
                                ".4f",
                            ),
                        ]
                        for ci, (chart_title, col, hover_lbl, tfmt) in enumerate(err_specs):
                            fig_r = _scalar_metric_spider_compare(
                                df_err_melt,
                                cats,
                                chart_title,
                                run_labels_list,
                                col,
                                hover_lbl,
                                height=400,
                                tickformat=tfmt,
                            )
                            with rcols[ci]:
                                st.plotly_chart(fig_r, width='stretch')
                else:
                    st.info("No data available")
            except Exception as e:
                st.error(f"Error: {e}")
    
            st.markdown(section_header_html("Difference of mean absolute error (each run − Baseline A)"), unsafe_allow_html=True)
            for idx in range(1, len(runs)):
                lbl = run_labels_list[idx]
                _med_slot = st.empty()
                _med_slot.markdown(ds_spot_loading_markup(f"Mean error diff · run {lbl}"), unsafe_allow_html=True)
                try:
                    fc_c = build_filter_clause(filters_list[idx])
                    query = f"""
                    WITH topic_a AS (
                        SELECT label,
                            AVG(ABS(x_error)) FILTER (WHERE status = 'TP') AS x_a,
                            AVG(ABS(y_error)) FILTER (WHERE status = 'TP') AS y_a,
                            AVG(ABS(yaw_error)) FILTER (WHERE status = 'TP') AS yaw_a
                        FROM view_eval_flat
                        WHERE {filter_clause_base}
                        GROUP BY label
                    ),
                    topic_c AS (
                        SELECT label,
                            AVG(ABS(x_error)) FILTER (WHERE status = 'TP') AS x_c,
                            AVG(ABS(y_error)) FILTER (WHERE status = 'TP') AS y_c,
                            AVG(ABS(yaw_error)) FILTER (WHERE status = 'TP') AS yaw_c
                        FROM {_flat_view(idx)}
                        WHERE {fc_c}
                        GROUP BY label
                    )
                    SELECT a.label,
                        (c.x_c - a.x_a) AS x_diff,
                        (c.y_c - a.y_a) AS y_diff,
                        (c.yaw_c - a.yaw_a) AS yaw_diff
                    FROM topic_a a
                    JOIN topic_c c USING (label)
                    ORDER BY label
                    """
                    df_ed = con.execute(query).df()
                    if not df_ed.empty:
                        with st.expander(f"Run {lbl} − A", expanded=(len(runs) == 2)):
                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=df_ed["label"], y=df_ed["x_diff"], name="X Diff", marker_color=RUN_COLORS[0]))
                            fig.add_trace(go.Bar(x=df_ed["label"], y=df_ed["y_diff"], name="Y Diff", marker_color=RUN_COLORS[1]))
                            fig.add_trace(go.Bar(x=df_ed["label"], y=df_ed["yaw_diff"], name="Yaw Diff", marker_color=RUN_COLORS[2]))
                            apply_chart_theme(fig)
                            fig.update_layout(title=f"Error diff ({lbl} − A) within {max_eval_range} [m]", xaxis_title="Label", yaxis_title="Error Difference [m] or [rad]", barmode="group")
                            st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.error(f"Error (Run {lbl} − A): {e}")
                finally:
                    _med_slot.empty()

    # =============================
    # Final section: perception release report
    # =============================
    ds_dlog("section: Manager_report_start")
    st.divider()
    st.markdown(
        section_header_html(
            "Perception release report",
            "Optional all-distance release assessment. Disabled by default to avoid extra report queries.",
        ),
        unsafe_allow_html=True,
    )
    load_release_report = st.toggle(
        "Load perception release report",
        value=False,
        key="ds_load_release_report",
        help="Runs additional all-distance report queries only when enabled.",
    )
    if load_release_report:
        _report_slot = st.empty()
        _report_slot.markdown(ds_spot_loading_markup("Perception release report"), unsafe_allow_html=True)
        try:
            report_filter_clause = build_filter_clause(filters_base, enable_dist_h=False)
            report_scope_label = "all available distances"
            if single_mode:
                report_kpi = _kpi_row_for_view(con, "view_eval_flat", report_filter_clause)
                report_html, report_md, report_tables = build_single_detection_report(
                    con,
                    run_label=run_labels_list[0],
                    view="view_eval_flat",
                    filter_clause=report_filter_clause,
                    scope_label=report_scope_label,
                    kpi=report_kpi,
                )
                render_detection_report(report_html, report_md, report_tables, key_prefix="single")
            else:
                kpi_by_label = {
                    lbl: _kpi_row_for_view(con, _flat_view(i), report_filter_clause)
                    for i, lbl in enumerate(run_labels_list)
                }
                base_label = run_labels_list[0]
                base_kpi = kpi_by_label.get(base_label)
                if len(runs) == 2:
                    for idx, lbl in enumerate(run_labels_list[1:], start=1):
                        safe_lbl = "".join(ch if ch.isalnum() else "_" for ch in str(lbl))
                        report_html, report_md, report_tables = build_compare_detection_report(
                            con,
                            base_label=base_label,
                            candidate_label=lbl,
                            base_view="view_eval_flat",
                            candidate_view=_flat_view(idx),
                            base_filter=report_filter_clause,
                            candidate_filter=report_filter_clause,
                            scope_label=report_scope_label,
                            base_kpi=base_kpi,
                            candidate_kpi=kpi_by_label.get(lbl),
                        )
                        render_detection_report(
                            report_html,
                            report_md,
                            report_tables,
                            key_prefix=f"compare_{idx}_{safe_lbl}",
                        )
                else:
                    report_tabs = st.tabs([f"{lbl} vs {base_label}" for lbl in run_labels_list[1:]])
                    for tab, idx, lbl in zip(report_tabs, range(1, len(runs)), run_labels_list[1:]):
                        with tab:
                            safe_lbl = "".join(ch if ch.isalnum() else "_" for ch in str(lbl))
                            report_html, report_md, report_tables = build_compare_detection_report(
                                con,
                                base_label=base_label,
                                candidate_label=lbl,
                                base_view="view_eval_flat",
                                candidate_view=_flat_view(idx),
                                base_filter=report_filter_clause,
                                candidate_filter=report_filter_clause,
                                scope_label=report_scope_label,
                                base_kpi=base_kpi,
                                candidate_kpi=kpi_by_label.get(lbl),
                            )
                            render_detection_report(
                                report_html,
                                report_md,
                                report_tables,
                                key_prefix=f"compare_{idx}_{safe_lbl}",
                            )
        except Exception as e:
            st.error(f"Error generating perception release report: {e}")
        finally:
            _report_slot.empty()
    else:
        st.caption("Release report is not loaded.")
    
    ds_dlog("main_content_try_exit_ok")
    ds_debug_log_memory("main_content_end")

except Exception as _e_ds_main:
    ds_debug_log_exception("detection_stats_main_try", _e_ds_main)
    raise

finally:
    try:
        ds_debug_render_expander(st.session_state)
    except Exception as _e_dbg_exp:
        ds_debug_log_exception("ds_debug_render_expander", _e_dbg_exp)
    ds_dlog("main_content_finally_banner_clear")
    _ds_loading_banner.empty()
    ds_dlog("detection_stats_script_run_complete")
