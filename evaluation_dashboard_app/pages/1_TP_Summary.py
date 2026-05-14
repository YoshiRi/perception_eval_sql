import streamlit as st
import plotly.express as px
import pandas as pd
from lib.path_utils import path_display
from lib.overview_url_hydrate import try_hydrate_session_from_overview_query_params
from lib.page_chrome import inject_app_page_styles, render_loaded_data_section, render_page_hero, section_header
from lib.summary_compare import build_summary_delta

st.set_page_config(layout="wide", page_title="TP Summary", page_icon="📈", initial_sidebar_state="expanded")
try_hydrate_session_from_overview_query_params()
inject_app_page_styles()

# ========== Safety Check ==========
if "runA" not in st.session_state:
    st.warning(
        "Please load data from the Overview page first. "
        "If you already did, open Overview once so the URL includes `run_a=...`, then return (multiple Streamlit replicas)."
    )
    st.stop()

mode = st.session_state.get("mode", "Single Run")

runA = st.session_state["runA"]
if runA.get("summary") is None:
    st.warning("This run has no **Summary.csv**. Load a run that includes Summary.csv for this page. Detection Stats and Bounding Box Viewer work with parquet-only runs.")
    st.stop()

df_a = runA["summary"]
all_runs = None
run_labels = None
delta_by_label = {}
df_active = df_a
use_delta = False
delta_candidate_label = "B"  # for plot axis labels when use_delta (columns stay *_B from build_summary_delta)

if mode == "Compare Mode":
    all_runs = st.session_state.get("all_runs")
    run_labels = st.session_state.get("run_labels")
    if (
        all_runs
        and run_labels
        and len(all_runs) == len(run_labels)
        and len(all_runs) >= 2
        and all(r is not None and r.get("summary") is not None for r in all_runs)
    ):
        cand_labels = run_labels[1:]
        for i, lbl in enumerate(cand_labels):
            delta_by_label[lbl] = build_summary_delta(all_runs[0]["summary"], all_runs[i + 1]["summary"])
        _ov_entries = [(f"Baseline · {run_labels[0]}", path_display(all_runs[0]["path"]))]
        for i in range(1, len(all_runs)):
            _ov_entries.append((f"Candidate · {run_labels[i]}", path_display(all_runs[i]["path"])))
        render_loaded_data_section(_ov_entries)
    else:
        runB = st.session_state.get("runB")
        df_b = runB["summary"] if runB and runB.get("summary") is not None else None
        df_cmp = st.session_state.get("df_cmp")
        if df_b is None or df_cmp is None:
            st.warning("Compare mode requires candidate run(s) and delta data. Reload from Overview.")
            st.stop()
        all_runs = [runA, runB]
        run_labels = ["A", "B"]
        delta_by_label["B"] = df_cmp
        render_loaded_data_section(
            [
                ("Baseline · A", path_display(runA["path"])),
                ("Candidate · B", path_display(runB["path"])),
            ]
        )
else:
    render_loaded_data_section([("Current run", path_display(runA["path"]))])

render_page_hero(
    kicker="TP & kinematics",
    title="Position, velocity & TP statistics",
    description="Summary.csv metrics: TP, lateral/longitudinal error stats, and velocity — filter, compare runs, or inspect deltas.",
    mode=mode,
)


def _apply_compact_chart_layout(fig, *, height: int = 300) -> None:
    """Keep TP Summary charts visually lighter and more compact."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(t=48, b=40, l=48, r=18),
        paper_bgcolor="rgba(248,250,252,0.9)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        font=dict(family="system-ui, sans-serif", size=12, color="#334155"),
    )

# ========== View Selector ==========
st.sidebar.markdown("##### Scope")
if mode == "Compare Mode" and all_runs and run_labels and delta_by_label:
    cand_labels = run_labels[1:]
    baseline_opt = f"Baseline ({run_labels[0]})"
    cand_opts = [f"Candidate ({lbl})" for lbl in cand_labels]
    delta_opts = [f"Delta ({lbl} - A)" for lbl in cand_labels]
    view_options = [baseline_opt] + cand_opts + delta_opts
    default_idx = 2 if len(cand_labels) == 1 else 2 + len(cand_labels) - 1
    default_idx = min(default_idx, len(view_options) - 1)
    view = st.sidebar.selectbox(
        "Dataset",
        view_options,
        index=default_idx,
        help="Delta = row-wise candidate − baseline after matching Summary keys from Overview.",
    )
    if view == baseline_opt:
        df_active = all_runs[0]["summary"]
        use_delta = False
    elif view.startswith("Candidate ("):
        inner = view[len("Candidate (") : -1]
        idx = run_labels.index(inner)
        df_active = all_runs[idx]["summary"]
        use_delta = False
    else:
        inner = view[len("Delta (") : -len(" - A)")]
        df_active = delta_by_label[inner]
        use_delta = True
        delta_candidate_label = inner
else:
    df_active = df_a
    view = "Single Run"
    use_delta = False

st.sidebar.divider()

# ========== Debug Controls ==========
show_debug = st.sidebar.checkbox("Show debug tables", value=False)

if show_debug:
    section_header("Raw data preview", "First rows of the active dataframe.")
    st.dataframe(df_active.head(10), width="stretch")

# ========== Sidebar Filters ==========
st.sidebar.markdown("##### Filters")
tp_col = "TP_delta" if use_delta else "TP"
if tp_col not in df_active.columns:
    st.warning(f"Missing required column: {tp_col}")
    st.stop()

if df_active.empty:
    if use_delta:
        _keys = "id and perception_label" if "perception_label" in df_active.columns else "id"
        st.warning(
            f"No delta rows: baseline and candidate share no common Summary keys ({_keys}). "
            "Pick Baseline or Candidate in the sidebar, or load runs with overlapping rows."
        )
    else:
        st.warning("The active Summary has no rows for this view.")
    st.stop()

tp_values = df_active[tp_col]
tp_min_val = float(tp_values.min())
tp_max_val = float(tp_values.max())
if use_delta:
    default_range = (tp_min_val, tp_max_val)
else:
    default_low = max(tp_min_val, 0.0)
    default_high = min(tp_max_val, 100.0)
    default_range = (default_low, default_high) if default_low <= default_high else (tp_min_val, tp_max_val)

# Handle case where min and max are equal (e.g., all TP values are the same)
if tp_min_val == tp_max_val:
    # Add a small epsilon so slider is not degenerate
    epsilon = 0.01 if tp_min_val != 0 else 1.0
    slider_min = tp_min_val - epsilon
    slider_max = tp_max_val + epsilon
    slider_default = (tp_min_val, tp_max_val)
else:
    slider_min = tp_min_val
    slider_max = tp_max_val
    slider_default = default_range

tp_min, tp_max = st.sidebar.slider(
    "TP delta range (%)" if use_delta else "TP range (%)",
    min_value=slider_min,
    max_value=slider_max,
    value=slider_default,
)
clip_vel = st.sidebar.checkbox("Clip velocity outliers", value=True)

# ========== Data Filtering ==========
df_f = df_active[(df_active[tp_col] >= tp_min) & (df_active[tp_col] <= tp_max)].copy()

if clip_vel and not df_f.empty:
    vx_col = "vx_delta" if use_delta else "vx"
    vy_col = "vy_delta" if use_delta else "vy"
    for c in (vx_col, vy_col):
        # Avoid changing the original DataFrame dtype warning
        q1, q99 = df_f[c].quantile([0.01, 0.99]).values
        df_f[c] = df_f[c].clip(q1, q99)

if show_debug:
    st.subheader("Filtered Data")
    st.dataframe(df_f, width="stretch")

# ========== Summary ==========
section_header("At-a-glance", "Row count and TP stats after sidebar filters (and velocity clipping if enabled).")
count = len(df_f)
tp_mean = df_f[tp_col].mean() if count else 0.0
tp_median = df_f[tp_col].median() if count else 0.0
tp_min_val = df_f[tp_col].min() if count else 0.0
tp_max_val = df_f[tp_col].max() if count else 0.0

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Rows", f"{count:,}")
col_b.metric(f"{'Delta ' if use_delta else ''}TP mean", f"{tp_mean:.2f}")
col_c.metric(f"{'Delta ' if use_delta else ''}TP median", f"{tp_median:.2f}")
col_d.metric(f"{'Delta ' if use_delta else ''}TP range", f"{tp_min_val:.2f} to {tp_max_val:.2f}")

if use_delta and count:
    pos_rate = (df_f[tp_col] > 0).mean() * 100.0
    neg_rate = (df_f[tp_col] < 0).mean() * 100.0
    col_e, col_f = st.columns(2)
    col_e.metric("Delta positive (%)", f"{pos_rate:.1f}")
    col_f.metric("Delta negative (%)", f"{neg_rate:.1f}")

# ========== Plots ==========
col1, col2 = st.columns(2)
cand = delta_candidate_label

with col1:
    section_header("Position RMS (X vs Y)", "Lateral vs longitudinal RMS error; color encodes TP or ΔTP.")
    # Always compare the two sources side by side (before and after/delta)
    if use_delta:
        # Show both reference and target RMS comparisons in a tighter 2-up row.
        rms_left, rms_right = st.columns(2)
        fig_rms_x_compare = px.scatter(
            df_f,
            x="xrms_B",
            y="xrms",
            color=tp_col,
            hover_data=["id", "xrms_delta", "yrms_delta"],
            labels={
                "xrms_B": f"X RMS ({cand})",
                "xrms": "X RMS (A)",
                tp_col: "Δ TP",
                "xrms_delta": "Δ X RMS",
                "yrms_delta": "Δ Y RMS",
            },
            title=f"X RMS · {cand} vs A",
            color_continuous_scale="Viridis",
        )
        fig_rms_x_compare.update_traces(marker=dict(size=7, opacity=0.58))
        _apply_compact_chart_layout(fig_rms_x_compare, height=290)
        with rms_left:
            st.plotly_chart(fig_rms_x_compare, width="stretch")

        fig_rms_y_compare = px.scatter(
            df_f,
            x="yrms_B",
            y="yrms",
            color=tp_col,
            hover_data=["id", "xrms_delta", "yrms_delta"],
            labels={
                "yrms_B": f"Y RMS ({cand})",
                "yrms": "Y RMS (A)",
                tp_col: "Δ TP",
                "xrms_delta": "Δ X RMS",
                "yrms_delta": "Δ Y RMS",
            },
            title=f"Y RMS · {cand} vs A",
            color_continuous_scale="Viridis",
        )
        fig_rms_y_compare.update_traces(marker=dict(size=7, opacity=0.58))
        _apply_compact_chart_layout(fig_rms_y_compare, height=290)
        with rms_right:
            st.plotly_chart(fig_rms_y_compare, width="stretch")
    else:
        # Just show the submission's RMS (x/y) for standard analysis
        fig_rms = px.scatter(
            df_f,
            x="xrms",
            y="yrms",
            color=tp_col,
            hover_data=["id"],
            labels={
                "xrms": "X RMS",
                "yrms": "Y RMS",
                tp_col: "TP",
            },
            color_continuous_scale="Viridis",
        )
        fig_rms.update_traces(marker=dict(size=8, opacity=0.68))
        _apply_compact_chart_layout(fig_rms, height=320)
        st.plotly_chart(fig_rms, width="stretch")

with col2:
    section_header("Velocity (vx vs vy)", "Planar velocity colored by TP or ΔTP.")

    def plot_velocity(df, vx, vy, vx_label, vy_label, *, title: str):
        fig = px.scatter(
            df,
            x=vx,
            y=vy,
            color=tp_col,
            hover_data=["id"],
            labels={
                vx: vx_label,
                vy: vy_label,
                tp_col: "TP",
            },
            color_continuous_scale="Plasma",
            title=title,
        )
        fig.update_traces(marker=dict(size=7, opacity=0.58))
        _apply_compact_chart_layout(fig, height=290 if use_delta else 320)
        return fig

    if use_delta:
        vel_left, vel_right = st.columns(2)
        with vel_left:
            st.plotly_chart(
                plot_velocity(df_f, "vx", "vy", "Vx (A)", "Vy (A)", title="Velocity · A"),
                width="stretch",
            )
        with vel_right:
            st.plotly_chart(
                plot_velocity(df_f, "vx_B", "vy_B", f"Vx ({cand})", f"Vy ({cand})", title=f"Velocity · {cand}"),
                width="stretch",
            )
    else:
        st.plotly_chart(
            plot_velocity(df_f, "vx", "vy", "Vx", "Vy", title="Velocity"),
            width="stretch",
        )

# ========== Metric Distribution ==========
section_header("Metric distribution", "Compact secondary views for a selected Summary metric.")
metrics = ["xstd", "ystd", "xrms", "yrms", "vx", "vy", "TP"]
metrics_delta = [f"{m}_delta" for m in metrics]
metric_options = metrics_delta if use_delta else metrics
default_metric = "TP_delta" if use_delta else "TP"
if default_metric in metric_options:
    default_index = metric_options.index(default_metric)
else:
    default_index = 0
metric = st.selectbox("Select metric", metric_options, index=default_index)

dist_left, dist_right = st.columns(2)

fig_hist = px.histogram(
    df_f,
    x=metric,
    nbins=36,
    color_discrete_sequence=["#0d9488"],
    marginal="box",
    opacity=0.88,
    title=f"{metric} distribution",
)
fig_hist.update_layout(
    showlegend=False,
    bargap=0.04,
    xaxis_title=metric,
    yaxis_title="Count",
)
_apply_compact_chart_layout(fig_hist, height=280)
with dist_left:
    st.plotly_chart(fig_hist, width="stretch")

fig_density = px.violin(
    df_f,
    y=metric,
    box=True,
    points="outliers",
    color_discrete_sequence=["#312e81"],
    title=f"{metric} density",
)
fig_density.update_layout(
    yaxis_title=metric,
    showlegend=False,
)
_apply_compact_chart_layout(fig_density, height=280)
with dist_right:
    st.plotly_chart(fig_density, width="stretch")

# ========== Scenario-level Delta Analysis (Compare Mode) ==========
df_cmp = df_active if use_delta else None
if mode == "Compare Mode" and use_delta and df_cmp is not None and "id" in df_cmp.columns:
    section_header("Per-scenario ΔTP", "Mean TP delta per scenario — rank by improvement, regression, or magnitude.")

    # Aggregate per scenario using mean delta TP
    scenario_delta = (
        df_cmp.groupby("id", as_index=False)["TP_delta"]
        .mean()
        .rename(columns={"TP_delta": "mean_TP_delta"})
    )

    # Attach counts (optionally show for context)
    scenario_counts = df_cmp.groupby("id")["TP_delta"].count().reset_index(name="Count")
    scenario_delta = pd.merge(scenario_delta, scenario_counts, on="id", how="left")

    # Sort/filter methods
    scen_sort = st.radio(
        "Scenario Ranking",
        [
            "Largest Absolute Change (|ΔTP|)",
            "Most Improved (Highest ΔTP)",
            "Most Degraded (Lowest ΔTP)",
            "Custom filter by name",
            "Show all",
        ],
        horizontal=True,
        key="tp_scen_sort",
    )
    scenario_delta_disp = scenario_delta.copy()
    max_n_scen = max(5, len(scenario_delta))
    default_n = min(20, max_n_scen)
    n_scen = st.number_input(
        "Top N scenarios to show",
        min_value=5,
        max_value=max_n_scen,
        value=default_n,
        key="delta_topN",
    )
    if scen_sort == "Most Improved (Highest ΔTP)":
        scenario_delta_disp = scenario_delta.sort_values("mean_TP_delta", ascending=False).head(int(n_scen))
    elif scen_sort == "Most Degraded (Lowest ΔTP)":
        scenario_delta_disp = scenario_delta.sort_values("mean_TP_delta", ascending=True).head(int(n_scen))
    elif scen_sort == "Largest Absolute Change (|ΔTP|)":
        scenario_delta_disp = scenario_delta.reindex(scenario_delta["mean_TP_delta"].abs().sort_values(ascending=False).index).head(int(n_scen))
    elif scen_sort == "Custom filter by name":
        name_sub = st.text_input("Show scenarios containing (case-insensitive):", "", key="delta_scen_filter")
        if name_sub:
            scenario_delta_disp = scenario_delta[scenario_delta["id"].astype(str).str.contains(name_sub, case=False, na=False)]
    else:
        scenario_delta_disp = scenario_delta

    st.plotly_chart(
        px.bar(
            scenario_delta_disp,
            x="id",
            y="mean_TP_delta",
            text_auto=".2f",
            title=f"Mean ΔTP per Scenario ({cand} − A)",
            color="mean_TP_delta",
            color_continuous_scale="RdYlGn",
            labels={"mean_TP_delta": "Mean ΔTP"},
        ),
        width="stretch",
    )
    st.dataframe(scenario_delta_disp, width="stretch")
