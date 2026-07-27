import html
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lib.path_utils import path_display
from lib.overview_url_hydrate import try_hydrate_session_from_overview_query_params
from lib.page_chrome import (
    inject_app_page_styles,
    render_loaded_data_section,
    render_page_hero,
    section_header,
)
from lib.ui.criteria_banners import (
    criteria_absolute_gate_section_hr_markup,
    criteria_compare_workspace_intro_markup,
    criteria_failing_scenarios_heading_markup,
    gate_compare_overlap_intro_markup,
    gate_verdict_banner_html,
    hover_html_for_scenarios,
)
from lib.criteria_absolute_gates import (
    MetricGateSpec,
    evaluate_scenario_gates,
    export_gate_result,
    failing_scenarios_table,
    gate_summary,
)
from lib.score_schema import (
    SCORE_BLOCK_SIZE,
    SCORE_NUM_COLS,
    SCORE_VIEW_METRIC_COLS,
    build_score_view,
    infer_score_criteria_count,
    score_base_cols,
    score_identity_cols,
)

st.set_page_config(
    layout="wide",
    page_title="Criteria Score",
    page_icon="📊",
    initial_sidebar_state="expanded",
)
try_hydrate_session_from_overview_query_params()

# Plotly theme (multi-run palette aligned with Overview / run cards)
_COMPARE_RUN_COLORS = ["#312e81", "#0f766e", "#e86a33", "#6b8e23", "#9b59b6", "#1abc9c"]
_PX_COLOR_QUAL = px.colors.qualitative.Bold


def _px_color_map_for_runs(run_labels: list[str]) -> dict[str, str]:
    """Map Run column values (Baseline (A), Candidate (B), …) to colors."""
    names = [f"Baseline ({run_labels[0]})"] + [f"Candidate ({lbl})" for lbl in run_labels[1:]]
    return {n: _COMPARE_RUN_COLORS[i % len(_COMPARE_RUN_COLORS)] for i, n in enumerate(names)}


def _plotly_apply_theme(fig, title: str, height: int = 440) -> None:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=16, color="#0f172a"), x=0, xanchor="left", pad=dict(t=8, b=12)),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color="#334155"),
        paper_bgcolor="rgba(248, 250, 252, 0.92)",
        plot_bgcolor="rgba(255, 255, 255, 0.95)",
        margin=dict(l=56, r=28, t=72, b=52),
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="system-ui"),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)


def _safe_default(default_lst, options_lst):
    """Keep only defaults that still exist in options (avoids Streamlit multiselect errors)."""
    if not isinstance(default_lst, list):
        default_lst = list(default_lst) if default_lst is not None else []
    options_set = set(options_lst)
    return [x for x in default_lst if x in options_set]


def _perception_label_options_from_runs(*runs) -> list:
    pl: set = set()
    for run in runs:
        if not run:
            continue
        summary = run.get("summary")
        if summary is None or "perception_label" not in summary.columns:
            continue
        pl.update(
            x for x in summary["perception_label"].dropna().unique() if str(x).strip() != ""
        )
    return sorted(pl)


def _filter_df_view_by_perception_labels(
    df_view: pd.DataFrame,
    summary: pd.DataFrame | None,
    selected_labels: list,
) -> pd.DataFrame:
    """Restrict Score rows to scenarios whose Summary perception_label is selected (Overview-style)."""
    if not selected_labels or df_view is None or df_view.empty:
        return df_view
    if summary is None or summary.empty or "perception_label" not in summary.columns:
        return df_view
    if "id" not in summary.columns or "Scenario" not in df_view.columns:
        return df_view
    s = summary[["id", "perception_label"]].copy()
    s["id"] = s["id"].astype(str)
    s = s[s["perception_label"].notna() & (s["perception_label"].astype(str).str.strip() != "")]
    s = s[s["perception_label"].isin(selected_labels)]
    allowed = set(s["id"].unique())
    if not allowed:
        return df_view.iloc[0:0].copy()

    scenario_key = df_view["Scenario"].astype(str)
    mask = scenario_key.isin(allowed)
    if "Dataset" in df_view.columns:
        # Older generated Score.csv files stored the final scenario suffix in Dataset,
        # while Summary.csv kept the full id. Keep matching those files too.
        composite_key = scenario_key + "_" + df_view["Dataset"].astype(str)
        mask = mask | composite_key.isin(allowed)
    return df_view.loc[mask].copy()


def _filter_df_view_by_scenarios(df_view: pd.DataFrame, selected_scenarios: list) -> pd.DataFrame:
    """Restrict to selected scenario IDs. Empty selection = no extra filter (all rows kept)."""
    if df_view is None or df_view.empty or "Scenario" not in df_view.columns:
        return df_view
    if not selected_scenarios:
        return df_view
    allow = {str(x) for x in selected_scenarios}
    return df_view.loc[df_view["Scenario"].astype(str).isin(allow)].copy()


def _apply_gate_data_filters(
    df_view: pd.DataFrame,
    summary: pd.DataFrame | None,
    perception_labels: list,
    scenario_selection: list,
    *,
    restrict_scenarios_to: set[str] | None = None,
) -> pd.DataFrame:
    """Perception labels, optional clip to a scenario-id set, then optional scenario multiselect."""
    d = _filter_df_view_by_perception_labels(df_view, summary, perception_labels)
    if restrict_scenarios_to is not None:
        d = d.loc[d["Scenario"].astype(str).isin(restrict_scenarios_to)].copy()
    return _filter_df_view_by_scenarios(d, scenario_selection)


# =========================
# Safety check
# =========================
if "runA" not in st.session_state:
    st.warning(
        "Please load data from the Overview page first. "
        "If you already did, open Overview once so the URL includes `run_a=...`, then return (multiple Streamlit replicas)."
    )
    st.stop()

mode = st.session_state.get("mode", "Single Run")

runA = st.session_state["runA"]
df_raw_A = runA.get("score")
if df_raw_A is None:
    st.warning("This run has no **Score.csv**. Load a run that includes Score.csv for this page. Detection Stats and Bounding Box Viewer work with parquet-only runs.")
    st.stop()

compare_runs: list | None = None
compare_labels: list[str] | None = None
if mode == "Compare Mode":
    all_runs = st.session_state.get("all_runs")
    run_labels_ss = st.session_state.get("run_labels")
    if (
        all_runs
        and run_labels_ss
        and len(all_runs) == len(run_labels_ss)
        and len(all_runs) >= 2
        and all(r is not None and r.get("score") is not None for r in all_runs)
    ):
        compare_runs = all_runs
        compare_labels = list(run_labels_ss)
    else:
        runB = st.session_state.get("runB")
        if not runB or runB.get("score") is None:
            st.warning("Compare Mode requires candidate run(s) with Score.csv from the Overview page.")
            st.stop()
        compare_runs = [runA, runB]
        compare_labels = ["A", "B"]

inject_app_page_styles()

if mode == "Compare Mode" and compare_runs and compare_labels:
    _ld_entries = [(f"Baseline · {compare_labels[0]}", path_display(compare_runs[0]["path"]))]
    for i in range(1, len(compare_runs)):
        _ld_entries.append((f"Candidate · {compare_labels[i]}", path_display(compare_runs[i]["path"])))
    render_loaded_data_section(_ld_entries)
else:
    render_loaded_data_section([("Current run", path_display(runA["path"]))])
# =========================
# Constants
# =========================

BASE_COLS = score_base_cols(df_raw_A)
CRITERIA_COLS = SCORE_VIEW_METRIC_COLS
BLOCK_SIZE = SCORE_BLOCK_SIZE
NUM_COLS = SCORE_NUM_COLS

_criteria_n_a = infer_score_criteria_count(df_raw_A)
if mode == "Compare Mode" and compare_runs:
    CRITERIA_COUNT = min(infer_score_criteria_count(r["score"]) for r in compare_runs)
else:
    CRITERIA_COUNT = _criteria_n_a

st.sidebar.markdown("##### Scope")
criteria_idx = st.sidebar.selectbox(
    "Criteria block",
    list(range(CRITERIA_COUNT)),
    format_func=lambda x: f"criteria{x}",
    help="Each block is one criteria set from Score.csv (distance, NM, pass rate, …).",
)
show_debug = st.sidebar.checkbox("Show debug tables", value=False)


def build_view(df_raw, criteria_idx):
    return build_score_view(df_raw, criteria_idx)


def _add_scenario_display(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Dataset" in d.columns:
        d["ScenarioDisplay"] = d["Scenario"].astype(str) + " [" + d["Dataset"].astype(str) + "]"
    else:
        d["ScenarioDisplay"] = d["Scenario"].astype(str)
    return d


st.sidebar.divider()
st.sidebar.markdown("##### Charts")
metric = st.sidebar.selectbox(
    "Y-axis metric",
    NUM_COLS,
    index=NUM_COLS.index("pass_rate"),
    help="Used for histograms and bar charts (pass rate is 0–100).",
)

group_by = st.sidebar.selectbox(
    "Color / group by",
    ["GT_OBJ", "Option"],
)

st.sidebar.divider()
abs_gates_enabled = st.sidebar.checkbox(
    "Enable absolute pass/fail gates",
    value=False,
    help="Count scenarios that pass/fail fixed thresholds (pass rate 0–100; optional 2nd metric).",
)
with st.sidebar.expander("Absolute pass/fail gates", expanded=abs_gates_enabled):
    _label_runs = compare_runs if mode == "Compare Mode" and compare_runs else [runA]
    perception_labels_gate = _perception_label_options_from_runs(*_label_runs)
    if not perception_labels_gate:
        st.caption("No perception labels in Summary.csv — gate filter unavailable for this run.")
        abs_gate_perception_labels: list = []
    else:
        prev_pg = st.session_state.get("selected_perception_labels", perception_labels_gate)
        safe_pg = _safe_default(prev_pg, perception_labels_gate)
        st.session_state["selected_perception_labels"] = safe_pg
        abs_gate_perception_labels = st.multiselect(
            "Perception Label Filter",
            perception_labels_gate,
            default=safe_pg,
            key="criteria_abs_gates_perception_filter",
            help=(
                "Limit gate evaluation to scenarios whose perception label appears in Summary.csv "
                "(same idea as Overview). Clear all selections to include every scenario."
            ),
        )
        st.session_state["selected_perception_labels"] = abs_gate_perception_labels

    _pools = []
    for run in _label_runs:
        _pools.append(
            _filter_df_view_by_perception_labels(
                build_view(run["score"], criteria_idx),
                run.get("summary"),
                abs_gate_perception_labels,
            )
        )
    _scen_pool_a = (
        sorted(_pools[0]["Scenario"].astype(str).unique().tolist()) if _pools and not _pools[0].empty else []
    )
    if len(_pools) > 1:
        set_lists = [
            set(p["Scenario"].astype(str).unique())
            for p in _pools
            if not p.empty and "Scenario" in p.columns
        ]
        scenario_gate_options = sorted(set.intersection(*set_lists)) if set_lists else []
    else:
        scenario_gate_options = _scen_pool_a

    if not scenario_gate_options:
        st.caption("No scenarios in the gate pool after perception filter — check labels or data.")
        abs_gate_selected_scenarios: list = []
    else:
        prev_sc = st.session_state.get("criteria_abs_gate_selected_scenarios", scenario_gate_options)
        safe_sc = _safe_default(prev_sc, scenario_gate_options)
        st.session_state["criteria_abs_gate_selected_scenarios"] = safe_sc
        abs_gate_selected_scenarios = st.multiselect(
            "Scenario filter (gates)",
            scenario_gate_options,
            default=safe_sc,
            key="criteria_abs_gates_scenario_filter_widget",
            help=(
                "Narrow gate evaluation to these scenario names (from Score.csv), after the perception label "
                "filter. In compare mode the pool is **overlap only** (baseline ∩ candidate, each label-filtered). "
                "**Clear all** to include every scenario still in the pool."
            ),
        )
        st.session_state["criteria_abs_gate_selected_scenarios"] = abs_gate_selected_scenarios

    abs_pass_min = st.number_input(
        "Minimum pass rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=95.0,
        step=0.1,
        help=(
            "Scenario pass rate is from Score.csv (same scale as lsim / eval_summary: 0–100). "
            "If a scenario has multiple table rows, the **mean** pass_rate across those rows is used."
        ),
    )
    abs_use_metric2 = st.checkbox("Second condition (numeric metric)", value=False)
    if abs_use_metric2:
        abs_metric2_col = st.selectbox(
            "Metric column",
            NUM_COLS,
            index=NUM_COLS.index("nm") if "nm" in NUM_COLS else 0,
            disabled=not abs_use_metric2,
        )
        abs_metric2_op = st.selectbox(
            "Operator",
            ["<=", ">="],
            index=0,
            disabled=not abs_use_metric2,
        )
        abs_metric2_threshold = st.number_input(
            "Metric threshold",
            value=0.0,
            format="%.6f",
            disabled=not abs_use_metric2,
        )

render_page_hero(
    kicker="Criteria-based evaluation",
    title="Score & pass-rate",
    description=(
        "Distributions, grouped averages, scenario-level comparisons, and optional absolute gates "
        "for a clear pass/fail sign-off."
    ),
    mode=mode,
    secondary_badge_inner_html=(
        f"Active block · <strong>criteria{criteria_idx}</strong> ({CRITERIA_COUNT} available)"
    ),
)

with st.expander("How to use this page", expanded=False):
    st.markdown(
        """
        1. **Pick a criteria block** in the sidebar — each block matches one criteria index from your evaluation pipeline.
        2. **Charts** show how the chosen metric spreads and how it differs by `GT_OBJ` or `Option`.
        3. In **Compare** mode, use the **Overlay / Delta** tabs to see distributions side-by-side or row-level **B − A** changes.
        4. Enable **Absolute pass/fail gates** when you need a binary, scenario-level verdict (e.g. min pass rate %) for sign-off.
        """
    )


def _df_for_absolute_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Columns needed for gating; drop Streamlit helper columns like Run."""
    use = [c for c in BASE_COLS + NUM_COLS if c in df.columns]
    return df.loc[:, use].copy()


def _gate_verdict_donut_fig(summ: dict) -> go.Figure:
    """Donut chart Pass vs Fail — strong visual share."""
    n = summ["n_scenarios"]
    npass = summ["n_pass"]
    nfail = summ["n_fail"]
    pct = summ["pass_pct"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Pass", "Fail"],
                values=[npass, nfail],
                hole=0.58,
                marker=dict(colors=["#22c55e", "#ef4444"], line=dict(color="#ffffff", width=2)),
                textinfo="value",
                textposition="outside",
                textfont=dict(size=15),
                hovertemplate="<b>%{label}</b><br>Scenarios: %{value}<br>%{percent}<extra></extra>",
            )
        ]
    )
    center = f"<b>{pct:.1f}%</b><br><span style='font-size:0.65em;font-weight:normal'>pass</span>"
    if n == 0:
        center = "—"
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
        margin=dict(t=30, b=40, l=24, r=24),
        height=300,
        annotations=[
            dict(text=center, x=0.5, y=0.5, font_size=22, showarrow=False, font_color="#0f172a")
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _gate_compare_overlap_stats(result_a: pd.DataFrame, result_b: pd.DataFrame) -> dict | None:
    """Classify scenarios on inner join (same Scenario id in both gate tables)."""
    if result_a is None or result_b is None or result_a.empty or result_b.empty:
        return None
    key_cols = [c for c in score_identity_cols(result_a) if c in result_b.columns]
    a = result_a[key_cols + ["scenario_pass"]].copy()
    b = result_b[key_cols + ["scenario_pass"]].copy()
    a["pass_a"] = a["scenario_pass"].map(bool)
    b["pass_b"] = b["scenario_pass"].map(bool)
    outer = a.drop(columns=["scenario_pass"]).merge(
        b.drop(columns=["scenario_pass"]),
        on=key_cols,
        how="outer",
        indicator=True,
    )
    n_only_a = int((outer["_merge"] == "left_only").sum())
    n_only_b = int((outer["_merge"] == "right_only").sum())
    both = outer.loc[outer["_merge"] == "both"].copy()
    n_inner = len(both)
    if n_inner == 0:
        return {
            "both_pass": 0,
            "both_fail": 0,
            "a_fail_b_pass": 0,
            "a_pass_b_fail": 0,
            "n_inner": 0,
            "n_only_a": n_only_a,
            "n_only_b": n_only_b,
            "merged": both,
        }
    pa, pb = both["pass_a"], both["pass_b"]
    return {
        "both_pass": int((pa & pb).sum()),
        "both_fail": int((~pa & ~pb).sum()),
        "a_fail_b_pass": int((~pa & pb).sum()),
        "a_pass_b_fail": int((pa & ~pb).sum()),
        "n_inner": n_inner,
        "n_only_a": n_only_a,
        "n_only_b": n_only_b,
        "merged": both,
    }


def _overlap_scenario_lists(merged: pd.DataFrame) -> dict[str, list[str]]:
    """Sorted scenario names per overlap bucket (inner-joined rows only)."""
    if merged is None or merged.empty:
        return {
            "both_pass": [],
            "both_fail": [],
            "a_fail_b_pass": [],
            "a_pass_b_fail": [],
        }
    if "Dataset" in merged.columns:
        scen = merged["Scenario"].astype(str) + " [" + merged["Dataset"].astype(str) + "]"
    else:
        scen = merged["Scenario"].astype(str)
    pa = merged["pass_a"].map(bool)
    pb = merged["pass_b"].map(bool)
    return {
        "both_pass": sorted(scen[pa & pb].tolist()),
        "both_fail": sorted(scen[~pa & ~pb].tolist()),
        "a_fail_b_pass": sorted(scen[~pa & pb].tolist()),
        "a_pass_b_fail": sorted(scen[pa & ~pb].tolist()),
    }


def _gate_compare_venn_style_fig(
    stats: dict,
    label_a: str,
    label_b: str,
    *,
    scenario_buckets: dict[str, list[str]] | None = None,
) -> go.Figure:
    """
    Euler-style diagram: left disk = failed baseline, right disk = failed candidate.
    Regions: both pass (above), only-A fail, intersection both fail, only-B fail.
    Invisible scatter markers carry hover with per-bucket scenario lists.
    """
    c_bp = stats["both_pass"]
    c_ff = stats["both_fail"]
    c_af = stats["a_fail_b_pass"]
    c_pf = stats["a_pass_b_fail"]
    la = html.escape(str(label_a))
    lb = html.escape(str(label_b))

    sb = scenario_buckets or {k: [] for k in ("both_pass", "both_fail", "a_fail_b_pass", "a_pass_b_fail")}

    r = 0.52
    cx1, cx2 = -0.34, 0.34
    fig = go.Figure()
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=cx1 - r,
        y0=-r,
        x1=cx1 + r,
        y1=r,
        fillcolor="rgba(49, 46, 129, 0.22)",
        line=dict(width=2.5, color="#312e81"),
        layer="below",
    )
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=cx2 - r,
        y0=-r,
        x1=cx2 + r,
        y1=r,
        fillcolor="rgba(15, 118, 110, 0.22)",
        line=dict(width=2.5, color="#0f766e"),
        layer="below",
    )

    def _bubble(x, y, n: int, title: str, subtitle: str, color: str) -> None:
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b style='font-size:22px;color:{color}'>{n:,}</b><br>"
            f"<span style='font-size:12px;font-weight:700;color:#0f172a'>{title}</span><br>"
            f"<span style='font-size:11px;color:#64748b'>{subtitle}</span>",
            showarrow=False,
            align="center",
        )

    bx_af, bx_ff, bx_pf, bx_bp = -0.58, 0.0, 0.58, 0.0
    by_af, by_ff, by_pf, by_bp = 0.0, 0.0, 0.0, 0.78
    _bubble(bx_af, by_af, c_af, "Baseline fail only", "Recovered on candidate", "#312e81")
    _bubble(bx_ff, by_ff, c_ff, "Both fail", "Still failing A & B", "#b45309")
    _bubble(bx_pf, by_pf, c_pf, "Candidate fail only", "Regression vs baseline", "#0f766e")
    _bubble(bx_bp, by_bp, c_bp, "Both pass", "Clean on both runs", "#047857")

    # Hit targets for hover (semi-transparent; shows scenario names on hover).
    hover_titles = [
        "Baseline fail only (recovered on candidate)",
        "Both fail",
        "Candidate fail only (regression)",
        "Both pass",
    ]
    hover_bodies = [
        hover_html_for_scenarios(hover_titles[0], sb["a_fail_b_pass"]),
        hover_html_for_scenarios(hover_titles[1], sb["both_fail"]),
        hover_html_for_scenarios(hover_titles[2], sb["a_pass_b_fail"]),
        hover_html_for_scenarios(hover_titles[3], sb["both_pass"]),
    ]
    fig.add_trace(
        go.Scatter(
            x=[bx_af, bx_ff, bx_pf, bx_bp],
            y=[by_af, by_ff, by_pf, by_bp],
            mode="markers",
            marker=dict(
                size=100,
                color=["rgba(49,46,129,0.12)", "rgba(180,83,9,0.14)", "rgba(15,118,110,0.12)", "rgba(4,120,87,0.12)"],
                line=dict(width=0),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_bodies,
        )
    )

    fig.add_annotation(
        x=cx1,
        y=-0.72,
        text=f"<b>{la}</b> · fail set",
        showarrow=False,
        font=dict(size=11, color="#312e81"),
    )
    fig.add_annotation(
        x=cx2,
        y=-0.72,
        text=f"<b>{lb}</b> · fail set",
        showarrow=False,
        font=dict(size=11, color="#0f766e"),
    )

    fig.update_xaxes(visible=False, range=[-1.28, 1.28])
    fig.update_yaxes(visible=False, range=[-0.95, 1.02], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=440,
        margin=dict(l=8, r=8, t=52, b=8),
        paper_bgcolor="rgba(248, 250, 252, 0.85)",
        plot_bgcolor="rgba(255,255,255,0.4)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        title=dict(
            text="<b>Overlap map</b> · hover a region for scenario names",
            x=0.5,
            xanchor="center",
            font=dict(size=15, color="#0f172a"),
        ),
        hoverlabel=dict(align="left", bgcolor="white", font_size=12, font_family="system-ui, sans-serif"),
    )
    return fig


def _gate_compare_sankey_fig(
    stats: dict,
    label_a: str,
    label_b: str,
    *,
    link_hover_html: list[str] | None = None,
) -> go.Figure:
    """Baseline pass/fail → Candidate pass/fail flows (inner-joined scenarios)."""
    bp, ff, af, pf = (
        stats["both_pass"],
        stats["both_fail"],
        stats["a_fail_b_pass"],
        stats["a_pass_b_fail"],
    )
    n_ap = bp + pf
    n_af = af + ff
    n_bp = bp + af
    n_bf = pf + ff

    la = str(label_a).replace("<", "")
    lb = str(label_b).replace("<", "")

    node_colors = ["#22c55e", "#fca5a5", "#22c55e", "#fca5a5"]
    link_colors = [
        "rgba(34, 197, 94, 0.35)",
        "rgba(239, 68, 68, 0.45)",
        "rgba(52, 211, 153, 0.45)",
        "rgba(185, 28, 28, 0.4)",
    ]
    vals = [bp, pf, af, ff]
    if sum(vals) == 0:
        vals = [0, 0, 0, 0]

    link_kw: dict = dict(
        source=[0, 0, 1, 1],
        target=[2, 3, 2, 3],
        value=vals,
        color=link_colors,
    )
    if link_hover_html and len(link_hover_html) == 4:
        link_kw["customdata"] = link_hover_html
        link_kw["hovertemplate"] = "%{customdata}<extra></extra>"
    else:
        link_kw["hovertemplate"] = "%{value:,} scenarios<extra></extra>"

    _sans = "system-ui, -apple-system, 'Segoe UI', sans-serif"
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=",",
                textfont=dict(family=_sans, size=13, color="#0f172a"),
                node=dict(
                    pad=28,
                    thickness=22,
                    line=dict(color="rgba(15,23,42,0.35)", width=1),
                    label=[
                        f"{la}<br><b>Pass</b><br>{n_ap:,}",
                        f"{la}<br><b>Fail</b><br>{n_af:,}",
                        f"{lb}<br><b>Pass</b><br>{n_bp:,}",
                        f"{lb}<br><b>Fail</b><br>{n_bf:,}",
                    ],
                    color=node_colors,
                ),
                link=link_kw,
            )
        ]
    )
    fig.update_layout(
        height=420,
        margin=dict(l=24, r=24, t=48, b=16),
        font=dict(family=_sans, size=12, color="#0f172a"),
        paper_bgcolor="rgba(248, 250, 252, 0.5)",
        title=dict(
            text="<b>Sankey</b> · hover a flow for scenario names",
            x=0.5,
            xanchor="center",
            font=dict(size=15, color="#0f172a"),
        ),
        hoverlabel=dict(align="left", bgcolor="white", font_size=12, font_family="system-ui, sans-serif"),
    )
    return fig


def _short_overlap_legend_label(full: str) -> str:
    """e.g. 'Candidate (B)' -> 'B' for compact metric titles."""
    s = str(full).strip()
    if "(" in s and s.endswith(")"):
        inner = s[s.rindex("(") + 1 : -1].strip()
        if inner:
            return inner
    return s


def _render_gate_compare_overlap(
    gate_results: list,
    label_a: str,
    label_b: str,
    *,
    key_suffix: str = "",
    show_section_intro: bool = True,
    pair_heading: str | None = None,
) -> None:
    """Venn-style + Sankey for baseline vs one candidate (two gate result rows)."""
    if len(gate_results) != 2:
        return
    _, res_a, _ = gate_results[0]
    _, res_b, _ = gate_results[1]
    stats = _gate_compare_overlap_stats(res_a, res_b)
    if stats is None:
        return

    cand_short = _short_overlap_legend_label(label_b)
    if pair_heading:
        st.markdown(
            f'<p style="font-size:1.02rem;font-weight:700;color:#0f172a;margin:0.85rem 0 0.35rem 0;">'
            f"{html.escape(pair_heading)}</p>",
            unsafe_allow_html=True,
        )
    if show_section_intro:
        st.markdown(gate_compare_overlap_intro_markup(), unsafe_allow_html=True)

    cap_parts = []
    if stats["n_inner"]:
        cap_parts.append(f"**{stats['n_inner']:,}** scenarios in both runs (inner join).")
    if stats["n_only_a"]:
        cap_parts.append(f"**{stats['n_only_a']:,}** only in {label_a} — excluded from overlap chart.")
    if stats["n_only_b"]:
        cap_parts.append(f"**{stats['n_only_b']:,}** only in {label_b} — excluded from overlap chart.")
    if cap_parts:
        st.caption(" ".join(cap_parts))

    if stats["n_inner"] == 0:
        st.info("No overlapping scenario names between the two gate tables — cannot draw compare overlap.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Both pass", f"{stats['both_pass']:,}")
    m2.metric(f"Recovered ({cand_short})", f"{stats['a_fail_b_pass']:,}")
    m3.metric(f"Regression ({cand_short})", f"{stats['a_pass_b_fail']:,}")
    m4.metric("Both fail", f"{stats['both_fail']:,}")
    st.caption(
        f"Recovered = failed baseline gate but passed {cand_short} · "
        f"Regression = passed baseline but failed {cand_short}."
    )

    merged = stats["merged"]
    scenario_buckets = _overlap_scenario_lists(merged)
    sankey_link_hover = [
        hover_html_for_scenarios("Pass baseline → Pass candidate", scenario_buckets["both_pass"]),
        hover_html_for_scenarios("Pass baseline → Fail candidate (regression)", scenario_buckets["a_pass_b_fail"]),
        hover_html_for_scenarios("Fail baseline → Pass candidate (recovered)", scenario_buckets["a_fail_b_pass"]),
        hover_html_for_scenarios("Fail baseline → Fail candidate (both fail)", scenario_buckets["both_fail"]),
    ]

    c_venn, c_sankey = st.columns(2, gap="large")
    with c_venn:
        st.plotly_chart(
            _gate_compare_venn_style_fig(
                stats,
                label_a,
                label_b,
                scenario_buckets=scenario_buckets,
            ),
            width='stretch',
            key=f"gate_compare_venn{key_suffix}",
            config={"displayModeBar": False},
        )
    with c_sankey:
        st.plotly_chart(
            _gate_compare_sankey_fig(
                stats,
                label_a,
                label_b,
                link_hover_html=sankey_link_hover,
            ),
            width='stretch',
            key=f"gate_compare_sankey{key_suffix}",
            config={"displayModeBar": True},
        )
    st.caption("Hover the **overlap disks** or **Sankey** flows to list scenarios in that bucket (truncated if very long; full list stays in the table below).")

    merged = stats.get("merged")
    if merged is not None and not merged.empty:
        bucket_rows = []
        for _, row in merged.iterrows():
            pa, pb = bool(row["pass_a"]), bool(row["pass_b"])
            if pa and pb:
                b = "both_pass"
            elif not pa and not pb:
                b = "both_fail"
            elif not pa and pb:
                b = "recovered_on_candidate"
            else:
                b = "regression_on_candidate"
            bucket_rows.append({"Scenario": row["Scenario"], "overlap_bucket": b})
        buck_df = pd.DataFrame(bucket_rows).sort_values(["overlap_bucket", "Scenario"])
        with st.expander(f"Scenario list by overlap bucket ({cand_short})", expanded=False):
            st.dataframe(
                buck_df,
                width="stretch",
                hide_index=True,
                key=f"gate_ov_bucket_df{key_suffix}",
            )


def _render_absolute_gates_section(
    runs: list,
    *,
    criteria_idx: int,
    criteria_count: int,
):
    """runs: list of (label, df_view) — df may include Run; it is stripped."""
    if not abs_gates_enabled:
        return

    spec = None
    if abs_use_metric2:
        op = "<=" if abs_metric2_op == "<=" else ">="
        spec = MetricGateSpec(abs_metric2_col, op, float(abs_metric2_threshold))

    st.markdown(criteria_absolute_gate_section_hr_markup(), unsafe_allow_html=True)
    section_header(
        "Final gate verdict",
        "Pass rate **0–100** (Score.csv / lsim). Thresholds from the sidebar — **last checkpoint** before sign-off.",
    )

    gate_results = []
    cols = st.columns(len(runs))
    for i, (label, dfv) in enumerate(runs):
        with cols[i]:
            try:
                result = evaluate_scenario_gates(
                    _df_for_absolute_gates(dfv),
                    float(abs_pass_min),
                    spec,
                )
            except Exception as e:
                st.error(f"Gate evaluation failed: {e}")
                continue
            summ = gate_summary(result)
            st.markdown(gate_verdict_banner_html(summ, label), unsafe_allow_html=True)

            if summ["n_scenarios"] > 0:
                st.markdown("**Scenario pass rate (bar)**")
                pct_frac = min(1.0, max(0.0, summ["pass_pct"] / 100.0))
                st.progress(pct_frac)
                st.caption(
                    f"{summ['pass_pct']:.1f}% scenarios pass ({summ['n_pass']:,} / {summ['n_scenarios']:,})"
                )
                st.plotly_chart(
                    _gate_verdict_donut_fig(summ),
                    width='stretch',
                    key=f"gate_donut_{i}",
                    config={"displayModeBar": False},
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Scenarios", f"{summ['n_scenarios']:,}")
            m2.metric("Pass", f"{summ['n_pass']:,}")
            m3.metric("Fail", f"{summ['n_fail']:,}")
            m4.metric("Pass %", f"{summ['pass_pct']:.1f}%")

            if summ["n_scenarios"] > 0:
                if summ["all_pass"]:
                    st.success("**Gate cleared** — every scenario satisfies the configured thresholds.")
                else:
                    st.error(
                        f"**Gate not cleared** — {summ['n_fail']:,} scenario(s) still outside thresholds."
                    )

            fails = failing_scenarios_table(result)
            if not fails.empty:
                st.markdown(criteria_failing_scenarios_heading_markup(len(fails)), unsafe_allow_html=True)
                st.dataframe(fails, width="stretch")
            gate_results.append((label, result, spec))

    if len(gate_results) >= 2:
        if len(gate_results) > 2:
            st.markdown(gate_compare_overlap_intro_markup(), unsafe_allow_html=True)
        base_lbl = gate_results[0][0]
        base_short = _short_overlap_legend_label(base_lbl)
        for i in range(1, len(gate_results)):
            if len(gate_results) > 2 and i > 1:
                st.divider()
            cand_lbl = gate_results[i][0]
            cand_short_h = _short_overlap_legend_label(cand_lbl)
            _render_gate_compare_overlap(
                [gate_results[0], gate_results[i]],
                base_lbl,
                cand_lbl,
                key_suffix=f"_{i}",
                show_section_intro=len(gate_results) == 2,
                pair_heading=(
                    f"{base_short} vs {cand_short_h}" if len(gate_results) > 2 else None
                ),
            )

    if len(gate_results) == 1:
        label, result, sp = gate_results[0]
        exp = export_gate_result(result, sp)
        exp.insert(0, "run", label)
        st.download_button(
            "Download gate results (CSV)",
            exp.to_csv(index=False).encode("utf-8"),
            file_name="criteria_absolute_gates.csv",
            mime="text/csv",
            key="dl_abs_gates_single",
            type="primary",
        )
    elif len(gate_results) > 1:
        parts = []
        for label, result, sp in gate_results:
            exp = export_gate_result(result, sp)
            exp.insert(0, "run", label)
            parts.append(exp)
        combined = pd.concat(parts, ignore_index=True)
        st.download_button(
            "Download gate results (CSV)",
            combined.to_csv(index=False).encode("utf-8"),
            file_name="criteria_absolute_gates_compare.csv",
            mime="text/csv",
            key="dl_abs_gates_compare",
            type="primary",
        )


if mode == "Compare Mode" and compare_runs and compare_labels:
    cl = compare_labels
    _px_map = _px_color_map_for_runs(cl)
    run_names = [f"Baseline ({cl[0]})"] + [f"Candidate ({cl[i]})" for i in range(1, len(cl))]
    df_views = [build_view(r["score"], criteria_idx) for r in compare_runs]
    for dv, rn in zip(df_views, run_names):
        dv["Run"] = rn
    df_compare = pd.concat(df_views, axis=0, ignore_index=True)
    df_view_A = df_views[0]
    cand_only = cl[1:]
    if len(cand_only) > 1:
        focus_cand = st.selectbox(
            "Candidate for Δ vs baseline (Delta tab & per-scenario Δ chart)",
            cand_only,
            index=0,
            key="criteria_delta_focus_candidate",
        )
    else:
        focus_cand = cand_only[0]
    focus_idx = cl.index(focus_cand)
    df_view_focus = df_views[focus_idx]

    if show_debug:
        for i, r in enumerate(compare_runs):
            tag = run_names[i]
            st.subheader(f"Raw Data Check — {tag}")
            st.dataframe(r["score"].head(10), width="stretch")
            st.subheader(f"Criteria {criteria_idx} Data — {tag}")
            st.dataframe(df_views[i], width="stretch")

    section_header("Run summary", "Rows loaded and **mean pass rate** (practical pass %, 0–100) per run.")
    n_r = len(df_views)
    row_cols = st.columns(min(n_r, 6))
    for i, dv in enumerate(df_views):
        with row_cols[i % len(row_cols)]:
            st.metric(f"Rows ({cl[i]})", f"{len(dv):,}")
    mean_base = df_views[0]["pass_rate"].mean() if len(df_views[0]) else 0.0
    pr_cols_m = st.columns(min(n_r, 6))
    for i, dv in enumerate(df_views):
        with pr_cols_m[i % len(pr_cols_m)]:
            m = dv["pass_rate"].mean() if len(dv) else 0.0
            if i == 0:
                st.metric(f"Pass mean ({cl[i]})", f"{m:.3f}")
            else:
                st.metric(f"Pass mean ({cl[i]})", f"{m:.3f}", f"{m - mean_base:+.3f}")

    baseline_scen_restrict = set(_scen_pool_a)
    gate_run_specs = []
    for i, run in enumerate(compare_runs):
        filtered = _apply_gate_data_filters(
            df_views[i],
            run.get("summary"),
            abs_gate_perception_labels,
            abs_gate_selected_scenarios,
            restrict_scenarios_to=baseline_scen_restrict if i > 0 else None,
        )
        gate_run_specs.append((run_names[i], filtered))
    _render_absolute_gates_section(
        gate_run_specs,
        criteria_idx=criteria_idx,
        criteria_count=CRITERIA_COUNT,
    )

    st.markdown(criteria_compare_workspace_intro_markup(), unsafe_allow_html=True)
    tab_ov, tab_dl = st.tabs(["Overlay: all runs", f"Delta: {focus_cand} - A"])

    with tab_ov:
        section_header(
            f"{metric} distribution",
            "Semi-transparent overlay — colors follow the run legend.",
        )
        fig = px.histogram(
            df_compare,
            x=metric,
            color="Run",
            color_discrete_map=_px_map,
            nbins=30,
            barmode="overlay",
            opacity=0.55,
            marginal="box",
        )
        _plotly_apply_theme(fig, f"{metric} · row-level distribution")
        st.plotly_chart(fig, width="stretch")

        section_header(f"Average {metric} by {group_by}", "Grouped means — compare runs side-by-side.")
        df_avg = (
            df_compare.groupby([group_by, "Run"], as_index=False)[metric]
            .mean()
            .sort_values(metric, ascending=False)
        )
        fig = px.bar(
            df_avg,
            x=group_by,
            y=metric,
            color="Run",
            color_discrete_map=_px_map,
            barmode="group",
            text_auto=".2f",
        )
        _plotly_apply_theme(fig, f"Mean {metric} by {group_by}")
        st.plotly_chart(fig, width="stretch")

        section_header("Pass rate by group", "Box + points — useful for spotting spread and outliers (0–100 scale).")
        fig = px.box(
            df_compare,
            x=group_by,
            y="pass_rate",
            color="Run",
            color_discrete_map=_px_map,
            points="all",
        )
        _plotly_apply_theme(fig, "Pass rate overview")
        st.plotly_chart(fig, width="stretch")

        section_header(
            "Per-scenario pass rate",
            "Scenarios present in every run (inner join) — filter to focus on regressions or wins.",
        )
        scenario_key_cols = score_identity_cols(df_views[0])
        merges = []
        for i, lbl in enumerate(cl):
            g = df_views[i].groupby(scenario_key_cols, as_index=False)["pass_rate"].mean()
            g = g.rename(columns={"pass_rate": f"pr_{lbl}"})
            merges.append(g)
        per_scenario = merges[0]
        for g in merges[1:]:
            per_scenario = per_scenario.merge(g, on=scenario_key_cols, how="inner")
        per_scenario = _add_scenario_display(per_scenario)
        pr_base = f"pr_{cl[0]}"
        delta_col = f"delta_{focus_cand}"
        for lbl in cand_only:
            per_scenario[f"delta_{lbl}"] = per_scenario[f"pr_{lbl}"] - per_scenario[pr_base]

        filter_method = st.radio(
            "Scenario Filter/Sort",
            [
                "All",
                "Top N by Delta",
                "Top N by Baseline",
                "Custom contains string",
            ],
            horizontal=True,
        )

        if filter_method == "Top N by Delta":
            N = st.number_input(
                f"Show Top N Scenarios by |Δ pass rate| ({focus_cand} − A):",
                min_value=5,
                max_value=100,
                value=20,
                key="crit_topn_delta",
            )
            per_scenario = per_scenario.reindex(
                per_scenario[delta_col].abs().sort_values(ascending=False).index
            )
            per_scenario_vis = per_scenario.head(int(N))
        elif filter_method == "Top N by Baseline":
            N = st.number_input(
                f"Show Top N Scenarios by Baseline ({cl[0]}) Pass Rate:",
                min_value=5,
                max_value=100,
                value=20,
                key="crit_topn_base",
            )
            per_scenario = per_scenario.sort_values(pr_base, ascending=False)
            per_scenario_vis = per_scenario.head(int(N))
        elif filter_method == "Custom contains string":
            search = st.text_input("Show scenarios with name containing (case-insensitive):", "")
            per_scenario_vis = (
                per_scenario[per_scenario["ScenarioDisplay"].str.contains(search, case=False, na=False)]
                if search
                else per_scenario
            )
        else:
            per_scenario_vis = per_scenario.copy()

        pr_cols_melt = [f"pr_{lbl}" for lbl in cl]
        col_to_run = {f"pr_{lbl}": run_names[i] for i, lbl in enumerate(cl)}
        per_scenario_vis_long = pd.melt(
            per_scenario_vis,
            id_vars=scenario_key_cols + ["ScenarioDisplay"],
            value_vars=pr_cols_melt,
            var_name="_k",
            value_name="pass_rate",
        )
        per_scenario_vis_long["Run"] = per_scenario_vis_long["_k"].map(col_to_run)
        per_scenario_vis_long = per_scenario_vis_long.sort_values(["Scenario", "Run"])

        fig = px.bar(
            per_scenario_vis_long,
            x="ScenarioDisplay",
            y="pass_rate",
            color="Run",
            color_discrete_map=_px_map,
            barmode="group",
            text_auto=".2f",
        )
        _plotly_apply_theme(fig, "Per-scenario pass rate (filtered)")
        st.plotly_chart(fig, width="stretch")

        section_header(
            f"Per-scenario delta ({focus_cand} − A)",
            f"Green = higher pass rate on {focus_cand}; red = regression vs baseline.",
        )
        fig2 = px.bar(
            per_scenario_vis.reindex(per_scenario_vis[delta_col].abs().sort_values(ascending=False).index),
            x="ScenarioDisplay",
            y=delta_col,
            color=delta_col,
            color_continuous_scale="RdYlGn",
            text_auto=".2f",
        )
        _plotly_apply_theme(fig2, "Pass rate delta by scenario")
        st.plotly_chart(fig2, width="stretch")

        table_cols = scenario_key_cols + pr_cols_melt + [f"delta_{lbl}" for lbl in cand_only]
        table_cols = [c for c in table_cols if c in per_scenario_vis.columns]
        with st.expander("Show Table: Per Scenario Pass Rates and Deltas"):
            st.dataframe(per_scenario_vis[table_cols], width="stretch")

        scatter_key = f"crit_scatter_{focus_cand}"
        if st.checkbox(
            f"Scatter: baseline ({cl[0]}) vs candidate ({focus_cand}) pass rate",
            value=False,
            key=scatter_key,
        ):
            scatter_fig = px.scatter(
                per_scenario_vis,
                x=pr_base,
                y=f"pr_{focus_cand}",
                text="ScenarioDisplay",
                labels={
                    pr_base: f"Baseline ({cl[0]}) Pass Rate",
                    f"pr_{focus_cand}": f"Candidate ({focus_cand}) Pass Rate",
                },
                title=f"Per scenario: baseline vs {focus_cand} (filtered)",
            )
            lim = float(
                max(
                    per_scenario_vis[pr_base].max(),
                    per_scenario_vis[f"pr_{focus_cand}"].max(),
                    100.0,
                )
            )
            scatter_fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=lim,
                y1=lim,
                line=dict(dash="dash", color="rgba(100,116,139,0.8)", width=2),
                xref="x",
                yref="y",
            )
            scatter_fig.update_xaxes(range=[0, lim])
            scatter_fig.update_yaxes(range=[0, lim])
            scatter_fig.update_traces(
                textposition="top center",
                marker=dict(size=10, line=dict(width=0.5, color="white")),
            )
            _plotly_apply_theme(scatter_fig, "Baseline vs candidate pass rate (parity line = equal)")
            st.plotly_chart(scatter_fig, width="stretch")

        st.download_button(
            "Download overlay data (CSV)",
            df_compare.to_csv(index=False).encode("utf-8"),
            file_name="criteria_compare_filtered.csv",
            mime="text/csv",
            type="primary",
        )

    with tab_dl:
        merged = df_view_A.merge(
            df_view_focus,
            on=BASE_COLS,
            suffixes=("_A", "_B"),
            how="inner",
        )
        merged[f"{metric}_delta"] = merged[f"{metric}_B"] - merged[f"{metric}_A"]
        merged["pass_rate_delta"] = merged["pass_rate_B"] - merged["pass_rate_A"]

        section_header(
            f"{metric} delta distribution",
            f"How much {focus_cand} differs from baseline on the same row (inner join on Scenario/Option/GT_OBJ).",
        )
        fig = px.histogram(
            merged,
            x=f"{metric}_delta",
            nbins=30,
            marginal="box",
            color_discrete_sequence=["#0d9488"],
        )
        _plotly_apply_theme(fig, f"Δ {metric} ({focus_cand} − A)")
        st.plotly_chart(fig, width="stretch")

        section_header(f"Mean Δ {metric} by {group_by}", "Where the shift concentrates across labels.")
        df_delta = (
            merged.groupby(group_by, as_index=False)[f"{metric}_delta"]
            .mean()
            .sort_values(f"{metric}_delta", ascending=False)
        )
        fig = px.bar(
            df_delta,
            x=group_by,
            y=f"{metric}_delta",
            text_auto=".2f",
            color_discrete_sequence=["#312e81"],
        )
        _plotly_apply_theme(fig, f"Grouped mean · Δ {metric}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

        section_header("Pass rate delta overview", "Distribution of per-row pass rate change.")
        fig = px.box(
            merged,
            x=group_by,
            y="pass_rate_delta",
            points="all",
            color_discrete_sequence=["#0369a1"],
        )
        _plotly_apply_theme(fig, "Δ pass rate by group")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

        section_header("Largest changes", "Top 20 rows by |Δ metric| — inspect regressions or improvements.")
        top_changes = merged.copy()
        top_changes["abs_delta"] = top_changes[f"{metric}_delta"].abs()
        top_changes = top_changes.sort_values("abs_delta", ascending=False).head(20)
        cols_show = BASE_COLS + [f"{metric}_A", f"{metric}_B", f"{metric}_delta"]
        if metric != "pass_rate":
            cols_show.append("pass_rate_delta")
        st.dataframe(top_changes[cols_show], width="stretch")

else:
    df_view = build_view(df_raw_A, criteria_idx)
    if show_debug:
        st.subheader("Raw Data Check — Single Mode")
        st.dataframe(df_raw_A.head(10), width="stretch")
        st.subheader(f"Criteria {criteria_idx} Data")
        st.dataframe(df_view, width="stretch")

    section_header("Run summary", "Dataset size and central pass-rate tendency (0–100 scale).")
    count = len(df_view)
    mean_pass = df_view["pass_rate"].mean() if count else 0.0
    median_pass = df_view["pass_rate"].median() if count else 0.0
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{count:,}")
    col2.metric("Pass rate mean", f"{mean_pass:.3f}")
    col3.metric("Pass rate median", f"{median_pass:.3f}")

    _render_absolute_gates_section(
        [
            (
                "Current run",
                _apply_gate_data_filters(
                    df_view,
                    runA.get("summary"),
                    abs_gate_perception_labels,
                    abs_gate_selected_scenarios,
                ),
            ),
        ],
        criteria_idx=criteria_idx,
        criteria_count=CRITERIA_COUNT,
    )

    section_header(f"{metric} distribution", f"How **{metric}** spreads across all rows; colored by **{group_by}**.")
    fig = px.histogram(
        df_view,
        x=metric,
        color=group_by,
        nbins=30,
        marginal="box",
        color_discrete_sequence=_PX_COLOR_QUAL,
    )
    _plotly_apply_theme(fig, f"{metric} · histogram")
    st.plotly_chart(fig, width="stretch")

    section_header(f"Mean {metric} by {group_by}", "Ranked bars — quick read on which labels drive the metric.")
    df_avg = (
        df_view
        .groupby(group_by, as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
    )
    fig = px.bar(
        df_avg,
        x=group_by,
        y=metric,
        text_auto=".2f",
        color=group_by,
        color_discrete_sequence=_PX_COLOR_QUAL,
    )
    _plotly_apply_theme(fig, f"Mean {metric}")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    section_header("Pass rate overview", "Spread and outliers of pass rate (0–100) within each group.")
    fig = px.box(
        df_view,
        x=group_by,
        y="pass_rate",
        points="all",
        color=group_by,
        color_discrete_sequence=_PX_COLOR_QUAL,
    )
    _plotly_apply_theme(fig, "Pass rate by group")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    section_header("Scenario leaderboard", "Mean pass rate per scenario — tune N and sort direction.")
    scenario_key_cols = score_identity_cols(df_view)
    scenario_metric = df_view.groupby(scenario_key_cols, as_index=False)["pass_rate"].mean()
    top_n = st.number_input("Top N scenarios", min_value=5, max_value=100, value=20, key="single_top_n")
    sort_order = st.radio("Order", ["Highest first", "Lowest first"], horizontal=True, key="single_scen_order")
    scenario_metric = scenario_metric.sort_values(
        "pass_rate",
        ascending=sort_order == "Lowest first",
    ).head(int(top_n))
    st.dataframe(scenario_metric, width="stretch")
