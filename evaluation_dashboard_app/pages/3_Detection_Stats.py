import html
from contextlib import contextmanager
import hashlib

import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Tuple

from lib.path_utils import path_display
from lib.detection_stats_debug import (
    ds_debug_init_session_state,
    ds_debug_log_exception,
    ds_debug_log_memory,
    ds_debug_render_expander,
    ds_dlog,
    ds_dtimer,
)
from lib.overview_url_hydrate import try_hydrate_session_from_overview_query_params
from lib.parquet_schema import schema_flags
from lib.page_chrome import inject_app_page_styles, render_loaded_data_section, render_page_hero
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

def list_parquets_in_run(run_path) -> List[str]:
    """Return sorted list of absolute paths to .parquet files in the run directory."""
    p = Path(run_path)
    if not p.is_dir():
        return []
    return sorted([str(f.resolve()) for f in p.glob("*.parquet")])

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


def build_filter_clause(filters: dict,*, enable_dist_h: bool = True) -> str:
    """Build WHERE clause from filters.

    For label / suites / visibility: ``None`` means this dimension is inactive (e.g. no suite column).
    An empty list ``[]`` means no restriction on that dimension (same as all options selected).
    Using ``if filters.get('label')`` would treat ``[]`` as falsy and accidentally drop the filter,
    causing full scans (very slow on large Parquet).
    """
    conditions = []
    
    if filters.get('topic_name') and filters['topic_name'] != '__all__':
        conditions.append(f"topic_name = '{filters['topic_name']}'")
    
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
        "and multi-run compare when you load several runs from Overview."
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
            tf = st.selectbox(
                f"Run ({lbl}) File",
                pl,
                format_func=lambda p: os.path.basename(p),
                index=min(i, len(pl) - 1),
                key=f"target_file_{lbl}"
            )
            target_files.append(tf)

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
            ok, msg = validate_parquet_file(con, path)
            if not ok:
                st.sidebar.error(f"**Run ({lbl}) file** cannot be read: {msg}")
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

        # Filter options from first file (applied to all runs)
        target_file = cached_target_files[0]
        topics = list_values(con, target_file, "topic_name")
        labels = list_values(con, target_file, "label")
        try:
            suite_options = list_values(con, target_file, "COALESCE(CAST(suite_name AS VARCHAR), '')")
        except Exception:
            suite_options = []
        vis_options = list_values(con, target_file, "COALESCE(CAST(visibility AS VARCHAR), 'not available') AS visibility")
        schema = schema_flags(con, target_file)
        st.session_state["_ds_parquet_fp"] = fp
        st.session_state["_ds_filter_opts"] = {
            "topics": topics,
            "labels": labels,
            "suite_options": suite_options,
            "vis_options": vis_options,
            "schema": schema,
            "cached_target_files": list(cached_target_files),
            "cache_rebuild_notes": list(cache_rebuild_notes),
        }
    else:
        opts = st.session_state["_ds_filter_opts"]
        topics = opts["topics"]
        labels = opts["labels"]
        suite_options = opts["suite_options"]
        vis_options = opts["vis_options"]
        schema = opts["schema"]
        target_file = target_files[0]
        cached_target_files = opts.get("cached_target_files", list(target_files))
        cache_rebuild_notes = opts.get("cache_rebuild_notes", [])
        for i, path in enumerate(cached_target_files):
            v_flat = "view_eval_flat" if i == 0 else f"view_eval_flat_{i}"
            create_view_eval_flat(con, path, v_flat)

ds_debug_log_memory("after_duckdb_validate_views")

with st.sidebar:
    topic_name = st.selectbox("Topic Name", ["__all__"] + topics, key="topic_name") if topics else "__all__"
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
    max_eval_range = st.selectbox("Max Evaluation Range [m]", [50, 80, 100, 120, 150], index=0, key="max_eval_range")

# Build filters (same values for all runs). None = dimension unused (no suite/visibility column in UI).
filters_base = {
    'topic_name': topic_name,
    'label': selected_labels,
    'suites': selected_suites if suite_options else None,
    'visibility': selected_visibility if vis_options else None,
    'max_eval_range': max_eval_range
}
filters_list = [filters_base] * len(runs)

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
    # KPI strip (TP, FP, FN, TPR, FPR, Precision, Recall, F1)
    # -----------------------------
    def _flat_view(i: int) -> str:
        return "view_eval_flat" if i == 0 else f"view_eval_flat_{i}"
    
    def _kpi_row_for_view(con, view: str, filter_clause: str):
        """Return dict with tp_gt, fn, tp_est, fp and derived TPR, FPR, Precision, Recall, F1."""
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
            "tp": tp_gt, "fp": fp, "fn": fn,
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
                """
                df_summary = con.execute(query_base).df()
                query_status = """
                SELECT label, status, COUNT(*) AS num
                FROM view_eval_flat
                GROUP BY label, status
                ORDER BY label, status
                """
                df_status = con.execute(query_status).df()
            else:
                parts = [f"SELECT COUNT(DISTINCT t4dataset_id) AS id_num, '{run_labels_list[i]}' AS series FROM {_flat_view(i)}" for i in range(len(runs))]
                query_base = " UNION ALL ".join(parts)
                df_summary = con.execute(query_base).df()
                parts_status = [f"SELECT '{run_labels_list[i]}' AS dataset, label, status, COUNT(*) AS num FROM {_flat_view(i)} GROUP BY label, status" for i in range(len(runs))]
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
        index=0,
        horizontal=True,
        key="tp_fp_rate_by_dist_style",
    )
    
    filter_clause_base = build_filter_clause(filters_base, enable_dist_h=False)
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
                        title=f"TP & FP rate by distance (within {max_eval_range} m)",
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
                        title=f"TP & FP rate by distance (within {max_eval_range} m)",
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
                        title=f"TP rate by distance",
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
                        title=f"FP rate by distance",
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
                        title=f"TP rate by distance",
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
                        title=f"FP rate by distance",
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
                        title=f"Object count by distance bin (within {max_eval_range} m)",
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
                        title=f"Object count by distance bin",
                        xaxis_title="Distance bin",
                        yaxis_title="Count",
                        xaxis=xaxis_oc,
                        hovermode="x unified",
                        **({"barmode": "group"} if not use_line_chart else {}),
                    )
                    st.plotly_chart(fig_oc, width='stretch')
        except Exception as e_oc:
            st.error(f"Error (object count by distance bin): {e_oc}")
    
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
                fb_all = {**filters_base, "max_eval_range": None}
                label_union: set = set()
                for i in range(len(runs)):
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
                            fb = {**filters_base, "max_eval_range": max_r}
                            dfs_slice = []
                            for i in range(len(runs)):
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
        max_frames: int,
    ) -> pd.DataFrame:
        """
        Build a leaf table for Plotly sunburst/treemap: root → scenario → frame → label.
        Caps scenarios and frames per scenario; merges the rest into Other buckets.
        """
        if df_obj.empty or "change_type" not in df_obj.columns:
            return pd.DataFrame()
        sub = df_obj[df_obj["change_type"] == change_type].copy()
        if sub.empty:
            return pd.DataFrame()
        sub["scenario_name"] = sub["scenario_name"].fillna("").astype(str).replace("", "(no scenario)")
        sub["label"] = sub["label"].fillna("").astype(str).replace("", "(no label)")
        sub["frame_key"] = (
            sub["t4dataset_id"].astype(str) + "|f" + sub["frame_index"].astype(str)
        )
        leaf = (
            sub.groupby(["scenario_name", "frame_key", "label"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        if leaf.empty:
            return pd.DataFrame()
        ms = max(int(max_scenarios), 1)
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
            fr_tot = g.groupby("frame_key")["n"].sum().sort_values(ascending=False)
            top_fr = set(fr_tot.head(mf).index)
            g2 = g.copy()
            g2["fr_g"] = np.where(g2["frame_key"].isin(top_fr), g2["frame_key"], "Other frames")
            agg = g2.groupby(["scen_g", "fr_g", "label"], as_index=False)["n"].sum()
            parts.append(agg)
        out = pd.concat(parts, ignore_index=True)
        out["root"] = root_label
    
        def _frame_ring_label(fr_g: str, scen_g: str) -> str:
            if fr_g == "Other frames" or str(fr_g) == "Other frames":
                return "Other frames"
            sfg = str(fr_g)
            if "|f" not in sfg:
                return sfg
            fid = sfg.split("|f", 1)[-1]
            if scen_g == "Other scenarios":
                t4 = sfg.split("|f", 1)[0]
                t4s = t4 if len(t4) <= 14 else ("…" + t4[-12:])
                return f"{t4s}|f{fid}"
            return f"f{fid}"
    
        out["fr_display"] = out.apply(
            lambda r: _frame_ring_label(r["fr_g"], r["scen_g"]), axis=1
        )
        return out
    
    
    def _comparison_lens_treemap_df(
        names: pd.Series,
        improved: pd.Series,
        degraded: pd.Series,
        root_title: str,
    ) -> pd.DataFrame:
        """Rows for px.treemap path root → Improved|Degraded → item (area = n)."""
        rows = []
        for i in range(len(names)):
            nm = str(names.iloc[i]).strip() or "—"
            if len(nm) > 72:
                nm = nm[:69] + "…"
            ip = float(improved.iloc[i]) if pd.notna(improved.iloc[i]) else 0.0
            dg = float(degraded.iloc[i]) if pd.notna(degraded.iloc[i]) else 0.0
            if ip > 0:
                rows.append(
                    {"root": root_title, "side": "Improved", "item": nm, "n": ip}
                )
            if dg > 0:
                rows.append(
                    {"root": root_title, "side": "Degraded", "item": nm, "n": dg}
                )
        return pd.DataFrame(rows)
    
    
    def _plot_comparison_lens_treemap(
        tdf: pd.DataFrame,
        st_key: str,
        title: str,
    ) -> None:
        if tdf is None or tdf.empty:
            st.caption("_No data for this view._")
            return
        fig = px.treemap(
            tdf,
            path=["root", "side", "item"],
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
            height=430,
            margin=dict(t=20, l=2, r=2, b=2),
            paper_bgcolor="rgba(0,0,0,0)",
            title=_title_layout,
        )
        st.plotly_chart(fig, width='stretch', key=st_key)
    
    
    if not single_mode:
        ds_dlog("section: Perception_diff_start")
        st.divider()
        st.markdown(
            section_header_html(
                "Perception diff (vs baseline A)",
                "Per-GT-object comparison vs baseline A: degraded = was TP on A and FN on candidate; improved = was FN on A and TP on candidate. Hotspots prioritize regressions.",
            ),
            unsafe_allow_html=True,
        )
        for idx in range(1, len(runs)):
            lbl = run_labels_list[idx]
            _pd_slot = st.empty()
            _pd_slot.markdown(ds_spot_loading_markup(f"Perception diff · run {lbl}"), unsafe_allow_html=True)
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
    
                    tot_imp = float(df_improved["improved_cnt"].sum())
                    tot_deg = float(df_improved["degraded_cnt"].sum())
                    tot_net = tot_imp - tot_deg
                    net_s = f"+{int(tot_net)}" if tot_net > 0 else str(int(tot_net))
    
                    with st.expander(f"Run {lbl} vs A", expanded=(len(runs) == 2)):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Improved (FN→TP)", int(tot_imp))
                        c2.metric("Degraded (TP→FN)", int(tot_deg))
                        c3.metric("Net TP delta", net_s)
                        c4.caption("Start with scenarios and frames with the most **degraded** counts.")
                        st.markdown(
                            f"**Summary:** Net **{net_s}** TP vs baseline A — "
                            f"**{int(tot_deg)}** degraded vs **{int(tot_imp)}** improved."
                        )
    
                        b_key = f"p5_baobab_{lbl}_{idx}"
                        c1b, c2b, c3b = st.columns([1, 1, 1])
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
                            baobab_nf = st.slider(
                                "Max frames / scenario",
                                min_value=5,
                                max_value=20,
                                value=10,
                                key=f"{b_key}_nf",
                            )
                        if df_by_object_full.empty:
                            st.caption("No object-level rows for hierarchy.")
                        else:
                            path_cols = ["root", "scen_g", "fr_display", "label"]
                            h_imp = _baobab_hierarchy_from_objects(
                                df_by_object_full,
                                "improved",
                                f"Improved ({lbl} vs A)",
                                baobab_ns,
                                baobab_nf,
                            )
                            h_deg = _baobab_hierarchy_from_objects(
                                df_by_object_full,
                                "degraded",
                                f"Degraded ({lbl} vs A)",
                                baobab_ns,
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
                                    fig_b = px.sunburst(
                                        hdf,
                                        path=path_cols,
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
                                        path=path_cols,
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
    
                        # --- Comparison lens: label / scenario / frame (treemap trio, Baobab-aligned) ---
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
    
                        df_frame_sorted = pd.DataFrame()
                        if not df_by_frame.empty:
                            df_frame_sorted = df_by_frame.sort_values(
                                by=["degraded_cnt", "improved_cnt"],
                                ascending=[False, True],
                            ).reset_index(drop=True)
    
                        root_lens = f"{lbl} vs A"
                        lc1, lc2, lc3 = st.columns(3, gap="small")
                        with lc1:
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
                        with lc2:
                            if not scen_agg.empty:
                                tdf_s = _comparison_lens_treemap_df(
                                    scen_agg["scenario_name"].astype(str),
                                    scen_agg["improved_cnt"],
                                    scen_agg["degraded_cnt"],
                                    root_lens,
                                )
                                _plot_comparison_lens_treemap(
                                    tdf_s,
                                    f"p5_lens_scen_{lbl}_{idx}",
                                    "By scenario",
                                )
                            else:
                                st.caption("_No scenario data._")
                        with lc3:
                            if not df_frame_sorted.empty:
                                fr_cap = 36
                                fr_top = df_frame_sorted.head(fr_cap).copy()
                                nms = (
                                    fr_top["scenario_name"].astype(str).str.slice(0, 26)
                                    + "\n· f"
                                    + fr_top["frame_index"].astype(str)
                                ).tolist()
                                ims = fr_top["improved_cnt"].astype(float).tolist()
                                dgs = fr_top["degraded_cnt"].astype(float).tolist()
                                rest = df_frame_sorted.iloc[fr_cap:]
                                if not rest.empty:
                                    io = float(rest["improved_cnt"].sum())
                                    do = float(rest["degraded_cnt"].sum())
                                    if io > 0 or do > 0:
                                        nms.append(
                                            f"Other frames\n({len(rest)} frames)"
                                        )
                                        ims.append(io)
                                        dgs.append(do)
                                tdf_f = _comparison_lens_treemap_df(
                                    pd.Series(nms),
                                    pd.Series(ims),
                                    pd.Series(dgs),
                                    root_lens,
                                )
                                _plot_comparison_lens_treemap(
                                    tdf_f,
                                    f"p5_lens_fr_{lbl}_{idx}",
                                    "By frame",
                                )
                                st.caption(
                                    f"Top **{fr_cap}** frames by degraded, plus **Other frames** "
                                    f"so totals match **By class** / **By scenario**."
                                )
                            else:
                                st.caption("_No frame data._")
    
                        with st.expander("Tables behind the lens (label / scenario / frame)"):
                            if not df_by_label.empty:
                                st.markdown("**Per label**")
                                st.dataframe(
                                    df_by_label,
                                    width='stretch',
                                    hide_index=True,
                                )
                            if not scen_agg.empty:
                                st.markdown("**Per scenario**")
                                st.dataframe(scen_agg, width='stretch', hide_index=True)
                            if not df_frame_sorted.empty:
                                st.markdown("**Per frame** (sorted by degraded)")
                                st.dataframe(
                                    df_frame_sorted.head(200),
                                    width='stretch',
                                    hide_index=True,
                                )
    
                        with st.expander("Full dataset breakdown (per t4dataset_id row)"):
                            st.dataframe(df_improved, width='stretch', hide_index=True)
    
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
                                        f"f{rw['frame_index']} | deg {int(rw['degraded_cnt'])}"
                                    )
                            with pr2:
                                if st.button(
                                    "Preset: top 10 degraded frames (object filter)",
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
                                st.download_button(
                                    label="Download filtered objects (CSV)",
                                    data=df_obj_show.to_csv(index=False).encode("utf-8"),
                                    file_name=f"perception_diff_{lbl}_vs_A_objects.csv",
                                    mime="text/csv",
                                    key=f"p5_dl_{lbl}_{idx}",
                                )
                                st.dataframe(
                                    df_obj_show.head(n_show),
                                    width='stretch',
                                    hide_index=True,
                                )
                            else:
                                st.caption("No objects match filters.")
    
                        with st.expander("Full frame table (sort: degraded desc)"):
                            if not df_frame_sorted.empty:
                                st.dataframe(df_frame_sorted, width='stretch', hide_index=True)
                            else:
                                st.caption("No frame breakdown.")
                else:
                    st.caption(f"Run {lbl} vs A: No data.")
            except Exception as e:
                st.error(f"Error (Run {lbl} vs A): {e}")
            finally:
                _pd_slot.empty()
    
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
