import streamlit as st
import pandas as pd
import io
import urllib.parse
import zipfile
import yaml
from pathlib import Path
from lib.run_loader import load_run
from lib.run_metadata import read_run_metadata
from lib.path_utils import (
    get_data_root,
    get_data_root_display,
    get_run_display_name,
    list_run_directories,
    path_display,
    resolve_run_subdirectory,
)
import plotly.express as px
import plotly.graph_objects as go
from lib.user_config import UserConfig
from lib.run_selection_cookie import persist_selection_cookie
from lib.run_selection_store import save_run_selection
from lib.summary_compare import build_summary_delta, summary_delta_overlap_stats
from lib.overview_pdf_report import build_overview_pdf_report, make_report_filename
from lib.specsheet_report import (
    DEFAULT_SPECSHEET_LABELS,
    DEFAULT_SPECSHEET_PROJECT_ID,
    DEFAULT_SPECSHEET_TOPIC,
    DEFAULT_TREND_METADATA_TEXT,
    discover_trend_release_groups,
    generate_specsheet_pdf,
    get_release_specsheet_context,
    get_specsheet_artifact_paths,
    is_specsheet_pdf_fresh,
    parse_trend_metadata_text,
    progress_fraction_from_message,
    resolve_specsheet_generation_run_path,
    write_trend_metadata,
)
from lib.page_chrome import (
    inject_app_page_styles,
    render_loaded_data_section,
    render_page_hero,
    render_share_link_callout,
    section_header,
)
from lib.deploy_debug import running_in_docker
from lib.ui.theme import CATEGORICAL, apply_plotly_theme, is_dark, pick, tokens

# ====== CHART COLORS ======
# Pre-dark-theme light palette. Light mode must keep rendering exactly these hues
# (up to 6 runs: A, B, C, D, E, F); dark mode uses the theme's categorical tokens.
_LEGACY_COMPARE_COLORS = ["#31356E", "#008E9B", "#E86A33", "#6B8E23", "#9B59B6", "#1ABC9C"]
# Delta-bar marker outline: plain "gray" before the dark theme.
_LEGACY_DELTA_OUTLINE = "gray"


def _chart_palette() -> list:
    """Run series colors: legacy light palette on light, tokens on dark."""
    return pick(_LEGACY_COMPARE_COLORS, CATEGORICAL())


def _theme_chart(fig):
    """Dark-only figure theming; light keeps the pre-dark-theme Plotly defaults."""
    if is_dark():
        apply_plotly_theme(fig)
    return fig

# ====== URL QUERY PARAMS (OPTIONAL OVERRIDE) ======
params = st.query_params

url_mode = params.get("mode")    # "single" / "compare" / None
url_run_a = params.get("run_a")  # str / None
# Candidates B, C, D, ... from URL (e.g. run_b=...&run_c=...)
url_compare_runs = [
    params.get(k) for k in ["run_b", "run_c", "run_d", "run_e"]
    if params.get(k)
]

# ====== CONFIG AND CONSTANTS ======
st.set_page_config(page_title="Overview", layout="wide", initial_sidebar_state="expanded")
inject_app_page_styles()
# if running_in_docker():
#     st.sidebar.page_link(
#         "pages/99_Deployment_Debug.py",
#         label="Deployment debug",
#         icon="🐳",
#     )
RUN_ROOT = get_data_root()


def _run_owner_label(run_path: Path) -> str:
    metadata = read_run_metadata(run_path)
    owner_meta = metadata.get("owner") if isinstance(metadata.get("owner"), dict) else {}
    task_meta = metadata.get("task") if isinstance(metadata.get("task"), dict) else {}
    requester_meta = task_meta.get("requester") if isinstance(task_meta.get("requester"), dict) else {}
    evaluator_meta = metadata.get("evaluator") if isinstance(metadata.get("evaluator"), dict) else {}
    label = str(
        owner_meta.get("name")
        or owner_meta.get("email")
        or owner_meta.get("id")
        or requester_meta.get("name")
        or requester_meta.get("email")
        or requester_meta.get("id")
        or task_meta.get("requested_by")
        or evaluator_meta.get("scheduled_by")
        or ""
    ).strip()
    return label


def _overview_entry_name(run_path: Path) -> str:
    name = get_run_display_name(run_path)
    owner = _run_owner_label(run_path)
    return f"{name} · owner: {owner}" if owner else name

PRODUCT_LABEL_JA = {
    "Occlusion-Case": "遮蔽ケース",
    "False-Positive-Grass": "草誤検知（草停止）",
    "False-Positive-Ground": "地面誤検知",
    "False-Positive-Splash": "水しぶき 誤検知",
    "False-Positive-Exhaust-Fog": "排ガス・霧 誤検知",
    "Missed-Detection-Animal": "動物ロスト（犬）",
    "Missed-Detection-Falling-Object": "落下物未検知",
    "Missed-Detection-Pedestrian-Child": "歩行者未検知：子供",
    "Missed-Detection-Pedestrian-Umbrella": "歩行者未検知：傘",
    "Missed-Detection-Pedestrian-Crouching": "歩行者未検知：しゃがむ",
    "Missed-Detection-Pedestrian-Near-Structure": "歩行者未検知：構造物に近い",
    "False-Positive-Truck": "トラック誤検知",
    "Pose-Estimation-Yaw-Error": "Yawおかしい",
    "Long-Range-Detection-Failure": "遠方見えない",
    "Ghost-Object": "ミサイル",
    "Sudden-Fast-Vehicle-Ghost": "高速車両の突然出現・急ブレーキ誘発",
    "Misclassification-Structure-Grass-as-Pedestrian": "構造物・草を人に誤検知",
    "Misclassification-Structure-Grass-as-Vehicle": "構造物・草を車両に誤検知",
    "Misclassification-Bike-Motorcycle": "自転車・バイクのミスラベル",
    "Missed-Detection-Unridden-Bike": "人の乗ってないバイク自転車ロスト",
    "Missed-Detection-Traffic-Cone": "カラーコーンが認識できない",
    "Missed-Detection-Other": "その他ロスト",
}

# ====== HELPER FUNCTIONS ======
def _safe_default(default_lst, options_lst):
    # Returns only those elements from default_lst that are also in options_lst
    # both expected to be list-like objects of hashables
    if not isinstance(default_lst, list):
        default_lst = list(default_lst) if default_lst is not None else []
    options_set = set(options_lst)
    safe = [x for x in default_lst if x in options_set]
    return safe

def create_filter_widgets(all_runs_data):
    # Collect and sort unique labels from runs that have Summary.csv
    pl, prodl = set(), set()
    for run in all_runs_data:
        summary = run.get("summary")
        if summary is None:
            continue
        # Guard against completely missing columns or all-na
        if "perception_label" in summary.columns:
            pl.update(
                [x for x in summary["perception_label"].dropna().unique() if str(x).strip() != ""]
            )
        if "product_label" in summary.columns:
            prodl.update(
                [x for x in summary["product_label"].dropna().unique() if str(x).strip() != ""]
            )
    perception_labels, product_labels = sorted(pl), sorted(prodl)
    label2ja = {label: PRODUCT_LABEL_JA.get(label, label) for label in product_labels}
    ja2label = {v: k for k, v in label2ja.items()}
    ja_cand = [label2ja.get(l, l) for l in product_labels]

    # Preserve UI state but only keep defaults that exist in actual options (avoid StreamlitAPIException)
    prev_perc = st.session_state.get("selected_perception_labels", perception_labels)
    prev_prod_ja = st.session_state.get("selected_product_ja_labels", ja_cand)

    safe_default_perc = _safe_default(prev_perc, perception_labels)
    safe_default_prod_ja = _safe_default(prev_prod_ja, ja_cand)

    st.session_state["selected_perception_labels"] = safe_default_perc
    st.session_state["selected_product_ja_labels"] = safe_default_prod_ja

    selected_perception_labels = st.sidebar.multiselect("Perception Label Filter", perception_labels,
        default=safe_default_perc, key="perception_filter_widget")
    selected_ja_labels = st.sidebar.multiselect("Product Label Filter", ja_cand,
        default=safe_default_prod_ja, key="product_filter_widget")

    st.session_state["selected_perception_labels"] = selected_perception_labels
    st.session_state["selected_product_ja_labels"] = selected_ja_labels
    selected_product_labels = [ja2label.get(ja, ja) for ja in selected_ja_labels] if selected_ja_labels else []

    return {
        "perception_labels": selected_perception_labels,
        "product_labels": selected_product_labels,
        "label_mappings": {"label2ja": label2ja, "ja2label": ja2label}
    }

def apply_filters(run_data, filters):
    s = run_data.get("summary")
    if s is None:
        return run_data
    if filters["perception_labels"] and "perception_label" in s.columns:
        s = s[s["perception_label"].notna() & (s["perception_label"].astype(str).str.strip() != "")]
        s = s[s["perception_label"].isin(filters["perception_labels"])]
    if filters["product_labels"] and "product_label" in s.columns:
        s = s[s["product_label"].notna() & (s["product_label"].astype(str).str.strip() != "")]
        s = s[s["product_label"].isin(filters["product_labels"])]
    return {**run_data, "summary": s}

def display_metric_with_stats(metric, a, b):
    st.metric(f"{metric} mean", f"{b.mean():.4f}", delta=f"{b.mean()-a.mean():.4f}")
    st.caption(f"median {b.median():.4f} · P95 {b.quantile(0.95):.4f} · min {b.min():.4f} · max {b.max():.4f}")

def display_metric_with_stats_single(metric, s):
    st.metric(f"{metric} mean", f"{s.mean():.4f}")
    st.caption(f"median {s.median():.4f} · P95 {s.quantile(0.95):.4f} · min {s.min():.4f} · max {s.max():.4f}")

def show_grouped_metrics_plot(df, group_col, label_map=None, mode="single", df_b=None):
    st.markdown(f"#### Metrics by {group_col.replace('_', ' ').title()}")
    metrics = ["TP", "xstd", "ystd", "xrms", "yrms"]
    if (group_col not in df.columns or df.empty or
        df[group_col].dropna().astype(str).str.strip().eq("").all() or
        (mode == "compare" and df_b is not None and (df_b.empty or df_b[group_col].dropna().astype(str).str.strip().eq("").all()))
    ):
        st.info("No data for group breakdown."); return
    df, col_map = df.copy(), (label_map if label_map else {})
    df = df[df[group_col].notna() & (df[group_col].astype(str).str.strip() != "")]
    df["__label_jp"] = df[group_col].map(col_map) if col_map else df[group_col]
    show_mode = "compare" if (mode == "compare" and df_b is not None) else "single"
    _palette = _chart_palette()
    colors = {"A": _palette[0], "B": _palette[1], "Δ(B-A)": _palette[2]}
    for m in metrics:
        st.markdown(f"##### {m.upper()} by {group_col.replace('_', ' ').title()}")
        if show_mode == "single":
            if df.empty:
                st.info("No data for group breakdown."); continue
            plot_df = df.groupby("__label_jp")[m].mean().reset_index().rename(columns={m:"Mean"})
            fig = px.bar(plot_df, x="__label_jp", y="Mean", labels={"__label_jp": group_col, "Mean": f"{m} mean"},
                         text_auto=".2f", color_discrete_sequence=[_palette[0]])
            fig.update_layout(xaxis_title=None, yaxis_title=f"{m} Mean", showlegend=False, height=400, margin=dict(t=40, b=0))
            _theme_chart(fig)
            st.plotly_chart(fig, width="stretch")
        else:
            df_b_c = df_b.copy()
            if group_col not in df_b_c.columns:
                st.info("No data for group breakdown."); continue
            df_b_c = df_b_c[df_b_c[group_col].notna() & (df_b_c[group_col].astype(str).str.strip() != "")]
            df_b_c["__label_jp"] = df_b_c[group_col].map(col_map) if col_map else df_b_c[group_col]
            mean_a, mean_b = df.groupby("__label_jp")[m].mean(), df_b_c.groupby("__label_jp")[m].mean()
            plot_labels = sorted(set(mean_a.index).union(mean_b.index))
            plot_df = pd.DataFrame({"__label_jp": plot_labels})
            plot_df["A"] = plot_df["__label_jp"].map(mean_a).fillna(0)
            plot_df["B"] = plot_df["__label_jp"].map(mean_b).fillna(0)
            plot_df["Δ(B-A)"] = plot_df["B"] - plot_df["A"]
            melted = plot_df.melt(id_vars="__label_jp", value_vars=["A", "B", "Δ(B-A)"], var_name="Run", value_name="Mean")
            bar_order = ["A", "B", "Δ(B-A)"]
            fig = px.bar(melted, x="__label_jp", y="Mean", color="Run", text_auto=".2f",
                         category_orders={"Run": bar_order, "__label_jp": plot_labels},
                         barmode="group", color_discrete_map=colors,
                         labels={"__label_jp": group_col, "Mean": f"{m} mean"})
            fig.update_layout(xaxis_title=None, yaxis_title=f"{m} Mean", legend_title="Run",
                              height=400, margin=dict(t=40, b=0))
            _theme_chart(fig)
            st.plotly_chart(fig, width="stretch")

def show_grouped_metrics_plot_multi(df_list, run_labels, group_col, label_map=None):
    """Grouped metrics by label for N runs. df_list and run_labels same length."""
    if not df_list or not run_labels or group_col not in df_list[0].columns:
        st.info("No data for group breakdown.")
        return
    st.markdown(f"#### Metrics by {group_col.replace('_', ' ').title()}")
    metrics = ["TP", "xstd", "ystd", "xrms", "yrms"]
    col_map = label_map or {}
    for m in metrics:
        if m not in df_list[0].columns:
            continue
        st.markdown(f"##### {m.upper()} by {group_col.replace('_', ' ').title()}")
        all_plot_labels = set()
        run_means = []
        for df in df_list:
            xdf = df[df[group_col].notna() & (df[group_col].astype(str).str.strip() != "")].copy()
            if xdf.empty or group_col not in xdf.columns:
                run_means.append(pd.Series(dtype=float))
                continue
            xdf["__label_jp"] = xdf[group_col].map(col_map) if col_map else xdf[group_col]
            s = xdf.groupby("__label_jp")[m].mean()
            run_means.append(s)
            all_plot_labels.update(s.index)
        if not all_plot_labels:
            st.info("No data for group breakdown.")
            continue
        plot_labels = sorted(all_plot_labels)
        plot_df = pd.DataFrame({"__label_jp": plot_labels})
        for i, lbl in enumerate(run_labels):
            if i < len(run_means):
                plot_df[lbl] = plot_df["__label_jp"].map(run_means[i]).fillna(0)
            else:
                plot_df[lbl] = 0
        var_cols = list(run_labels)
        if run_labels[0] == "A" and len(run_labels) > 1:
            for i in range(1, len(run_labels)):
                plot_df[f"Δ({run_labels[i]}-A)"] = plot_df[run_labels[i]] - plot_df["A"]
            var_cols = run_labels + [f"Δ({run_labels[i]}-A)" for i in range(1, len(run_labels))]
        melted = plot_df.melt(id_vars="__label_jp", value_vars=var_cols, var_name="Run", value_name="Mean")
        color_map = {lbl: COMPARE_COLORS[i % len(COMPARE_COLORS)] for i, lbl in enumerate(run_labels)}
        for i in range(1, len(run_labels)):
            color_map[f"Δ({run_labels[i]}-A)"] = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        fig = px.bar(melted, x="__label_jp", y="Mean", color="Run", text_auto=".2f",
                     category_orders={"Run": var_cols, "__label_jp": plot_labels},
                     barmode="group", color_discrete_map=color_map,
                     labels={"__label_jp": group_col, "Mean": f"{m} mean"})
        fig.update_layout(xaxis_title=None, yaxis_title=f"{m} Mean", legend_title="Run",
                          height=400, margin=dict(t=40, b=0))
        _theme_chart(fig)
        st.plotly_chart(fig, width="stretch")

# ====== SIDEBAR UI ======
user_config = UserConfig(warning_fn=st.warning)
saved_mode = user_config.get("overview_mode", "Single Mode")
# URL override (only if exists)
if url_mode == "compare":
    saved_mode = "Compare Mode"
elif url_mode == "single":
    saved_mode = "Single Mode"

st.sidebar.header("Mode")
mode_options = ["Single Mode", "Compare Mode"]
mode_index = mode_options.index(saved_mode) if saved_mode in mode_options else 0
mode = st.sidebar.radio("Mode", mode_options, index=mode_index)
if st.session_state.get("mode") != mode and "overview_compare_run_names" in st.session_state:
    del st.session_state["overview_compare_run_names"]
st.session_state["mode"] = mode
user_config.set("overview_mode", mode)

# --- Handle RUN_ROOT existence and emptiness ---
if not RUN_ROOT.exists() or not RUN_ROOT.is_dir():
    st.warning(f"Data directory not found: '{get_data_root_display()}'.\n\nPlease create the data directory and place your evaluation results inside it.")
    run_dirs = []
    run_names = []
    run_a_dir = None
    run_b_dir = None
    st.stop()

# List run directories (subdirectories in RUN_ROOT)
run_dirs = list_run_directories()
run_names = [get_run_display_name(p) for p in run_dirs]


def _coerce_run_param_to_display_name(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw in run_names:
        return raw
    resolved, err = resolve_run_subdirectory(raw)
    if err or resolved is None:
        return ""
    return get_run_display_name(resolved)

if not run_dirs:
    st.warning(f"No runs found in '{get_data_root_display()}'.\n\nPlease add at least one sub-directory with evaluation results, e.g. `{get_data_root_display()}/my_eval_run/`.")
    st.stop()

render_page_hero(
    kicker="Evaluation",
    title="Overview",
    description=(
        "Choose baseline and optional compare runs, filter perception/product labels, and inspect summary metrics. "
        "Use the sidebar pages for other views; copy the share link below so teammates open the same view."
    ),
    mode=mode,
)

saved_run_a = user_config.get("overview_run_a", run_names[0] if run_names else "")
# URL override (only if valid)
url_run_a_display = _coerce_run_param_to_display_name(url_run_a)
if url_run_a_display in run_names:
    saved_run_a = url_run_a_display

run_a_index = run_names.index(saved_run_a) if saved_run_a in run_names else 0
run_a_dir = st.sidebar.selectbox("Baseline (A)", run_dirs, index=run_a_index, format_func=get_run_display_name)
run_a_name = get_run_display_name(run_a_dir)
user_config.set("overview_run_a", run_a_name)

compare_run_names = []  # list of run names for candidates B, C, D, ...
if mode == "Compare Mode":
    # Use session_state so "Add run" / "Remove" work without relying on config file read-back
    if "overview_compare_run_names" not in st.session_state:
        saved_compare = user_config.get("overview_compare_runs", None)
        if saved_compare is None:
            saved_run_b = user_config.get("overview_run_b", "")
            saved_compare = [saved_run_b] if saved_run_b in run_names else []
        if not saved_compare and run_names:
            saved_compare = [run_names[1]] if len(run_names) > 1 else [run_names[0]]
        if url_compare_runs:
            valid_url = [
                display
                for display in (_coerce_run_param_to_display_name(r) for r in url_compare_runs)
                if display in run_names
            ]
            if valid_url:
                saved_compare = valid_url
        st.session_state["overview_compare_run_names"] = list(saved_compare)
    compare_run_names = list(st.session_state["overview_compare_run_names"])

    st.sidebar.caption("Compare runs")
    new_compare_run_names = []
    for i, run_name in enumerate(compare_run_names):
        letter = chr(66 + i)  # B, C, D, ...
        idx = run_names.index(run_name) if run_name in run_names else 0
        if len(compare_run_names) > 1:
            col_sel, col_rm = st.sidebar.columns([8, 1])
            with col_sel:
                selected = st.selectbox(
                    f"Candidate ({letter})",
                    run_dirs,
                    index=idx,
                    format_func=get_run_display_name,
                    key=f"compare_run_select_{i}",
                )
            with col_rm:
                if st.button("✕", key=f"compare_remove_{i}", help="Remove this run"):
                    removed_list = compare_run_names[:i] + compare_run_names[i + 1:]
                    st.session_state["overview_compare_run_names"] = removed_list
                    user_config.set("overview_compare_runs", removed_list)
                    st.rerun()
        else:
            selected = st.sidebar.selectbox(
                f"Candidate ({letter})",
                run_dirs,
                index=idx,
                format_func=get_run_display_name,
                key=f"compare_run_select_{i}",
            )
        new_compare_run_names.append(get_run_display_name(selected))
    compare_run_names = new_compare_run_names
    st.session_state["overview_compare_run_names"] = compare_run_names

    if st.sidebar.button("➕ Add run", help="Add another run to compare"):
        used = {run_a_name} | set(compare_run_names)
        next_name = next((n for n in run_names if n not in used), run_names[0])
        new_list = compare_run_names + [next_name]
        st.session_state["overview_compare_run_names"] = new_list
        user_config.set("overview_compare_runs", new_list)
        st.rerun()

    user_config.set("overview_compare_runs", compare_run_names)
    if compare_run_names:
        user_config.set("overview_run_b", compare_run_names[0])

compare_run_dirs = []
if mode == "Compare Mode" and compare_run_names:
    name_to_dir = {get_run_display_name(p): p for p in run_dirs}
    compare_run_dirs = [name_to_dir[n] for n in compare_run_names if n in name_to_dir]

# ====== SYNC URL (NON-DESTRUCTIVE) ======
query = {
    "mode": "compare" if mode == "Compare Mode" else "single",
    "run_a": run_a_name,
}
for j, name in enumerate(compare_run_names):
    query[f"run_{chr(98 + j)}"] = name  # run_b, run_c, ...
st.query_params.update(query)
# Remember this selection so pages opened directly later (no query params, fresh browser session)
# restore it instead of asking the user to come back through Overview: a cookie for per-browser
# memory, plus a per-user server-side copy for when cookies are unavailable.
persist_selection_cookie(mode, run_a_name, compare_run_names)
save_run_selection(mode, run_a_name, compare_run_names)
# ====== LOAD DATA ======
def safe_load_run(path, label='Run'):
    try:
        return load_run(path)
    except Exception as e:
        st.error(f"Failed to load {label}: {e}")
        st.stop()

if mode == "Compare Mode" and compare_run_dirs:
    all_run_dirs = [run_a_dir] + compare_run_dirs
    run_labels = ["A"] + [chr(66 + i) for i in range(len(compare_run_dirs))]
    all_runs = [
        safe_load_run(d, f"Run {run_labels[i]}") for i, d in enumerate(all_run_dirs)
    ]
    filters = create_filter_widgets(all_runs)
    all_runs = [apply_filters(r, filters) for r in all_runs]
    runA = all_runs[0]
    df_cmp = None
    if len(all_runs) >= 2 and all_runs[0].get("summary") is not None and all_runs[1].get("summary") is not None:
        df_cmp = build_summary_delta(all_runs[0]["summary"], all_runs[1]["summary"])
    st.session_state.update({
        "runA": runA,
        "all_runs": all_runs,
        "run_labels": run_labels,
        "df_cmp": df_cmp,
    })
    # For backward compat, runB = second run when present
    if len(all_runs) >= 2:
        st.session_state["runB"] = all_runs[1]
    else:
        st.session_state["runB"] = None
elif mode == "Compare Mode":
    st.warning("Add at least one candidate run to compare.")
    st.stop()
else:
    runA = safe_load_run(run_a_dir, 'Run A')
    filters = create_filter_widgets([runA])
    runA = apply_filters(runA, filters)
    st.session_state["runA"] = runA
    # Clear compare-related state so other pages (e.g. Bounding Box Viewer) show single mode
    for key in ("all_runs", "run_labels", "runB", "df_cmp"):
        st.session_state.pop(key, None)

# ====== MAIN PAGE METRICS & CHARTS ======
_ov_entries = [("Baseline · A", _overview_entry_name(runA["path"]))]
if mode == "Compare Mode" and compare_run_dirs:
    all_runs = st.session_state["all_runs"]
    run_labels = st.session_state["run_labels"]
    for i in range(1, len(all_runs)):
        _ov_entries.append((f"Candidate · {run_labels[i]}", _overview_entry_name(all_runs[i]["path"])))
render_loaded_data_section(_ov_entries)

if mode == "Compare Mode" and compare_run_dirs:
    _all_r = st.session_state.get("all_runs")
    _lbls = st.session_state.get("run_labels")
    if _all_r and _lbls and all(r.get("summary") is not None for r in _all_r):
        _cand_stats: list[tuple[str, dict]] = []
        _overlap_rows: list[dict] = []
        _empty_labels: list[str] = []
        _invalid_msgs: list[str] = []
        for i in range(1, len(_all_r)):
            cand = _lbls[i]
            stt = summary_delta_overlap_stats(_all_r[0]["summary"], _all_r[i]["summary"])
            _cand_stats.append((cand, stt))
            if not stt.get("valid"):
                _invalid_msgs.append(f"**{cand}:** {stt.get('error', 'Unknown error')}")
                continue
            join_s = " + ".join(stt["key_cols"])
            _overlap_rows.append(
                {
                    "Candidate": cand,
                    "Join keys": join_s,
                    "Baseline rows": stt["n_rows_baseline"],
                    "Candidate rows": stt["n_rows_candidate"],
                    "Matched (Δ rows)": stt["n_matched_keys"],
                    "Keys only in A": stt["n_only_baseline"],
                    "Keys only in candidate": stt["n_only_candidate"],
                }
            )
            if stt["matched_empty"]:
                _empty_labels.append(cand)
        if _invalid_msgs:
            st.warning(
                "Cannot compute Summary delta alignment for some runs:\n\n"
                + "\n\n".join(_invalid_msgs)
            )
        if _empty_labels:
            _join_cols = next(
                (" + ".join(f"`{c}`" for c in s["key_cols"]) for cnd, s in _cand_stats if cnd in _empty_labels and s.get("valid")),
                "`id` (or `id` + `perception_label` when both have it)",
            )
            st.warning(
                "**TP Summary delta views will be empty** for candidate(s) "
                f"**{', '.join(_empty_labels)}**: baseline **A** and those runs share **no** overlapping "
                f"Summary join keys ({_join_cols}). "
                "The inner join drops every row; use **Baseline** or **Candidate** in the TP Summary sidebar, "
                "or choose runs whose Summary rows use the same keys. "
                "Open **Summary key overlap (delta alignment)** below for row counts and sample keys "
                "that appear on only one side."
            )
            with st.expander("Summary key overlap (delta alignment) — details", expanded=False):
                st.markdown(
                    "Delta tables on **TP Summary** inner-join baseline **A** to each candidate on the "
                    "same keys as here: **`id`**, or **`id` + `perception_label`** when both summaries "
                    "include `perception_label`. Only **matched** keys produce rows; the rest are ignored."
                )
                st.dataframe(pd.DataFrame(_overlap_rows), width="stretch", hide_index=True)
                for cand, stt in _cand_stats:
                    if not stt.get("valid"):
                        continue
                    sb = stt["sample_only_baseline"]
                    sc = stt["sample_only_candidate"]
                    if not sb and not sc:
                        continue
                    st.markdown(f"**Examples — candidate {cand}**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Up to 5 keys only in baseline A")
                        st.code("\n".join(sb) if sb else "(none)")
                    with c2:
                        st.caption(f"Up to 5 keys only in {cand}")
                        st.code("\n".join(sc) if sc else "(none)")

share_query = {
    "mode": "compare" if mode == "Compare Mode" else "single",
    "run_a": run_a_name,
}
if mode == "Compare Mode" and compare_run_names:
    for j, name in enumerate(compare_run_names):
        share_query[f"run_{chr(98 + j)}"] = name
share_q = urllib.parse.urlencode(share_query)
render_share_link_callout(
    share_q,
    caption="Append to your server URL (e.g. `https://host:8501/?` + query). Build links from Data Management too.",
)

section_header("Summary metrics", "TP and error statistics from Summary.csv (respects Overview label filters).")
if runA.get("summary") is None:
    st.info(
        "**Summary.csv** not found for this run. "
        "Detection Stats and Bounding Box Viewer work with parquet-only runs. "
        "TP Summary and Criteria-based Score pages require Summary.csv / Score.csv."
    )

def show_tp_mean_by_label(df, label_col, label_jp_map=None, run_name=None):
    if label_col not in df.columns or df.empty:
        return
    # drop rows that are NA or blank
    xdf = df[df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")]
    if xdf.empty:
        st.info(f"No data for {label_col.replace('_', ' ')} breakdown.")
        return
    group_tp = xdf.groupby(label_col)["TP"].mean()
    labels = group_tp.index.tolist()
    labels_disp = [label_jp_map.get(l, l) for l in labels] if label_jp_map else labels
    title = f"TP mean by {label_col.replace('_', ' ').title()}"
    title += f" — {run_name}" if run_name else ""
    st.markdown(f"**{title}**")
    fig = go.Figure(go.Bar(
        x=labels_disp, y=group_tp.values, text=[f"{x:.2f}" for x in group_tp.values],
        textposition="auto", marker=dict(color=_chart_palette()[0]),
    ))
    fig.update_layout(xaxis_title=label_col.replace('_', ' ').title(),
                      yaxis_title="TP mean", height=400, margin=dict(t=40, b=0))
    _theme_chart(fig)
    st.plotly_chart(fig, width="stretch")

# Colors for up to 6 runs (A, B, C, D, E, F)
COMPARE_COLORS = _chart_palette()

def show_tp_mean_by_label_compare(df_list, run_labels, label_col, label_jp_map=None):
    """Grouped TP mean by label for N runs. df_list and run_labels same length."""
    if not df_list or not run_labels or label_col not in df_list[0].columns:
        return
    all_labels = set()
    groups = []
    for df in df_list:
        if label_col not in df.columns:
            return
        xdf = df[df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")]
        g = xdf.groupby(label_col)["TP"].mean()
        groups.append(g)
        all_labels.update(g.index)
    if not all_labels:
        st.info(f"No data for {label_col.replace('_', ' ')} breakdown.")
        return
    all_labels = sorted(all_labels)
    labels_disp = [label_jp_map.get(l, l) for l in all_labels] if label_jp_map else all_labels
    st.markdown(f"**TP mean by {label_col.replace('_', ' ').title()} ({' vs '.join(run_labels)})**")
    traces = []
    for i, (g, lbl) in enumerate(zip(groups, run_labels)):
        vals = [g.get(l, float('nan')) for l in all_labels]
        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        traces.append(
            go.Bar(name=lbl, x=labels_disp, y=vals, marker=dict(color=color),
                   text=[f"{x:.2f}" if pd.notna(x) else "N/A" for x in vals], textposition="auto")
        )
    # Deltas vs A (baseline) for non-A runs
    if len(df_list) >= 2 and run_labels[0] == "A":
        base_vals = [groups[0].get(l, float('nan')) for l in all_labels]
        for i in range(1, len(groups)):
            vals = [groups[i].get(l, float('nan')) for l in all_labels]
            deltas = [v - b if pd.notna(v) and pd.notna(b) else float('nan') for v, b in zip(vals, base_vals)]
            traces.append(
                go.Bar(name=f"Δ({run_labels[i]}-A)", x=labels_disp, y=deltas,
                       marker=dict(color=COMPARE_COLORS[i % len(COMPARE_COLORS)], line=dict(width=1, color=pick(_LEGACY_DELTA_OUTLINE, tokens()["muted"]))),
                       text=[f"{x:+.2f}" if pd.notna(x) else "N/A" for x in deltas], textposition="auto")
            )
    fig = go.Figure(traces)
    fig.update_layout(barmode="group", xaxis_title=label_col.replace('_', ' ').title(),
                      yaxis_title="TP mean", height=400, margin=dict(t=40, b=0), legend_title="Run")
    _theme_chart(fig)
    st.plotly_chart(fig, width="stretch")

if mode == "Compare Mode" and compare_run_dirs:
    all_runs = st.session_state["all_runs"]
    run_labels = st.session_state["run_labels"]
    if any(r.get("summary") is None for r in all_runs):
        st.info(
            "One or more runs do not have Summary.csv (parquet-only). "
            "Detection Stats and Bounding Box Viewer work with parquet. "
            "Summary metrics here and TP Summary / Criteria Score pages require Summary.csv."
        )
    else:
        summaries = [r["summary"] for r in all_runs]
        df_a = summaries[0]
        # TP mean: show baseline and each candidate with delta vs A
        n_runs = len(summaries)
        tp_means = [s["TP"].mean() for s in summaries]
        st.metric("TP mean (baseline A)", f"{tp_means[0]:.2f}")
        if n_runs > 1:
            comp_cols = st.columns(min(n_runs - 1, 5))
            for i, c in enumerate(comp_cols):
                if i + 1 < n_runs:
                    with c:
                        st.metric(f"TP mean ({run_labels[i + 1]})", f"{tp_means[i + 1]:.2f}",
                                  delta=f"{tp_means[i + 1] - tp_means[0]:+.2f}")
        cols = st.columns(4)
        metrics = [("XRMS", "xrms"), ("YRMS", "yrms"), ("XSTD", "xstd"), ("YSTD", "ystd")]
        for c, (n, col) in zip(cols, metrics):
            with c:
                if n_runs == 2:
                    display_metric_with_stats(n, df_a[col], summaries[1][col])
                else:
                    st.markdown(f"**{n}**")
                    for i, (s, lbl) in enumerate(zip(summaries, run_labels)):
                        st.caption(f"{lbl}: mean {s[col].mean():.4f}" + (f" (Δ vs A: {s[col].mean() - df_a[col].mean():+.4f})" if i > 0 else ""))
        show_tp_mean_by_label_compare(summaries, run_labels, "perception_label")
        show_tp_mean_by_label_compare(summaries, run_labels, "product_label", PRODUCT_LABEL_JA)
        with st.expander("Show metric breakdowns by label", expanded=False):
            show_grouped_metrics_plot_multi(summaries, run_labels, group_col="perception_label")
            show_grouped_metrics_plot_multi(summaries, run_labels, group_col="product_label", label_map=PRODUCT_LABEL_JA)
elif runA.get("summary") is not None:
    df_summary = runA["summary"]
    st.metric("TP mean", f"{df_summary['TP'].mean():.2f}")
    cols = st.columns(4)
    metrics = [("XRMS", "xrms"), ("YRMS", "yrms"), ("XSTD", "xstd"), ("YSTD", "ystd")]
    for c, (n, col) in zip(cols, metrics):
        with c: display_metric_with_stats_single(n, df_summary[col])
    show_tp_mean_by_label(df_summary, "perception_label")
    show_tp_mean_by_label(df_summary, "product_label", PRODUCT_LABEL_JA)
    with st.expander("Show metric breakdowns by label", expanded=False):
        show_grouped_metrics_plot(df_summary, group_col="perception_label", mode="single")
        show_grouped_metrics_plot(df_summary, group_col="product_label", label_map=PRODUCT_LABEL_JA, mode="single")


st.divider()
section_header("Export Dashboard Report", "Generate a curated PDF from the current Overview selection and filters.")
_report_runs = st.session_state.get("all_runs") if mode == "Compare Mode" and compare_run_dirs else [runA]
_report_labels = st.session_state.get("run_labels") if mode == "Compare Mode" and compare_run_dirs else ["A"]
_report_filters = {
    "perception_labels": filters.get("perception_labels", []),
    "product_labels": filters.get("product_labels", []),
}
_report_key = {
    "mode": mode,
    "paths": [str(r.get("path")) for r in _report_runs],
    "perception_labels": list(_report_filters["perception_labels"]),
    "product_labels": list(_report_filters["product_labels"]),
}
pdf_col1, pdf_col2 = st.columns([1.2, 2.8])
with pdf_col1:
    if st.button("Generate Evaluation Dashboard Report", type="primary", use_container_width=True):
        _pdf_status = st.empty()
        try:
            def _update_pdf_status(message: str) -> None:
                _pdf_status.info(f"Generating report: {message}")

            _update_pdf_status("starting")
            pdf_bytes = build_overview_pdf_report(
                mode=mode,
                run_records=_report_runs,
                run_labels=_report_labels,
                filters=_report_filters,
                product_label_map=PRODUCT_LABEL_JA,
                progress_callback=_update_pdf_status,
            )
            st.session_state["overview_pdf_report_bytes"] = pdf_bytes
            st.session_state["overview_pdf_report_key"] = _report_key
            run_names_for_file = [get_run_display_name(r["path"]) for r in _report_runs if r.get("path") is not None]
            st.session_state["overview_pdf_report_name"] = make_report_filename(run_names_for_file)
            _pdf_status.success("PDF report is ready.")
        except Exception as e:
            st.session_state.pop("overview_pdf_report_bytes", None)
            st.session_state.pop("overview_pdf_report_key", None)
            st.session_state.pop("overview_pdf_report_name", None)
            _pdf_status.error(f"PDF generation failed: {e}")
with pdf_col2:
    _pdf_ready = (
        st.session_state.get("overview_pdf_report_bytes") is not None
        and st.session_state.get("overview_pdf_report_key") == _report_key
    )
    if _pdf_ready:
        st.download_button(
            "Download Evaluation Dashboard Report",
            data=st.session_state["overview_pdf_report_bytes"],
            file_name=st.session_state.get("overview_pdf_report_name", "overview_report.pdf"),
            mime="application/pdf",
            use_container_width=True,
        )

specsheet_title = "Export Specsheet Report"
section_header(
    specsheet_title,
    "Generate the release-oriented spec-sheet PDF.",
)

_specsheet_run_records = _report_runs
_specsheet_run_labels = _report_labels
_specsheet_run_options = {}
for label, record in zip(_specsheet_run_labels, _specsheet_run_records):
    source_path = record["path"]
    target_path = resolve_specsheet_generation_run_path(source_path)
    release_context = get_release_specsheet_context(source_path)
    option_label = f"{label} · {get_run_display_name(source_path)}"
    if release_context is not None and target_path != source_path:
        option_label = (
            f"{label} · {get_run_display_name(source_path)} "
            f"(PDF body: {get_run_display_name(target_path)})"
        )
    _specsheet_run_options[option_label] = {
        "source_path": source_path,
        "target_path": target_path,
        "release_context": release_context,
    }
_specsheet_run_option_keys = list(_specsheet_run_options.keys())
_default_specsheet_run_selection = _specsheet_run_option_keys[:1]
_default_specsheet_labels = list(DEFAULT_SPECSHEET_LABELS)
_default_specsheet_project_id = st.session_state.get("specsheet_project_id", DEFAULT_SPECSHEET_PROJECT_ID)
_default_specsheet_topic = st.session_state.get("specsheet_topic_name", DEFAULT_SPECSHEET_TOPIC)
_single_specsheet_run_path = resolve_specsheet_generation_run_path(_specsheet_run_records[0]["path"])
_default_specsheet_version = get_run_display_name(_single_specsheet_run_path)

if mode == "Compare Mode":
    selected_specsheet_run_keys = st.multiselect(
        "Runs to generate spec-sheet for",
        options=list(_specsheet_run_options.keys()),
        default=_default_specsheet_run_selection,
        key="specsheet_target_runs",
        help="Spec-sheet generation is single-run, so multiple selected runs are processed one by one.",
    )
else:
    selected_specsheet_run_keys = _specsheet_run_option_keys[:1]

_selected_specsheet_entries = [
    _specsheet_run_options[key]
    for key in selected_specsheet_run_keys
    if key in _specsheet_run_options
]
selected_specsheet_run_paths = []
_seen_specsheet_targets = set()
for entry in _selected_specsheet_entries:
    target_path = entry["target_path"]
    target_key = str(target_path.resolve())
    if target_key in _seen_specsheet_targets:
        continue
    selected_specsheet_run_paths.append(target_path)
    _seen_specsheet_targets.add(target_key)
selected_specsheet_release_contexts = []
_seen_specsheet_releases = set()
for entry in _selected_specsheet_entries:
    release_context = entry["release_context"]
    if release_context is None:
        continue
    release_dir = release_context.get("release_dir")
    release_key = str(release_dir.resolve()) if isinstance(release_dir, Path) else str(release_dir)
    if release_key in _seen_specsheet_releases:
        continue
    selected_specsheet_release_contexts.append(release_context)
    _seen_specsheet_releases.add(release_key)
_active_specsheet_paths = [get_specsheet_artifact_paths(path) for path in selected_specsheet_run_paths]
_selected_trend_metadata_text = ""
_selected_trend_metadata_path = None
if len(selected_specsheet_release_contexts) == 1:
    candidate_path = selected_specsheet_release_contexts[0].get("metadata")
    if isinstance(candidate_path, Path) and candidate_path.exists():
        _selected_trend_metadata_path = candidate_path
if _selected_trend_metadata_path is None and len(_active_specsheet_paths) == 1 and _active_specsheet_paths[0]["trend_metadata"].exists():
    _selected_trend_metadata_path = _active_specsheet_paths[0]["trend_metadata"]
if _selected_trend_metadata_path is not None:
    try:
        _selected_trend_metadata_text = _selected_trend_metadata_path.read_text(encoding="utf-8")
    except Exception:
        _selected_trend_metadata_text = ""

_selected_metadata_defaults = {}
if _selected_trend_metadata_text:
    try:
        _selected_metadata_defaults = parse_trend_metadata_text(_selected_trend_metadata_text)
    except Exception:
        _selected_metadata_defaults = {}

def _specsheet_title_version_from_metadata(metadata: dict) -> str:
    explicit = str(
        metadata.get("pilot_auto_version_abbr") or metadata.get("version_abbr") or ""
    ).strip()
    if explicit:
        return explicit
    version = str(metadata.get("pilot_auto_version") or "").strip()
    if version.lower().startswith("pilot.auto "):
        return version[len("Pilot.Auto "):].strip()
    return version

_metadata_default_version = _specsheet_title_version_from_metadata(_selected_metadata_defaults)
if _metadata_default_version:
    _default_specsheet_version = _metadata_default_version
_metadata_trend_topic = str(_selected_metadata_defaults.get("topic_name") or "").strip()
if (
    _metadata_trend_topic
    and _metadata_trend_topic != DEFAULT_SPECSHEET_TOPIC
    and st.session_state.get("specsheet_topic_name") == _metadata_trend_topic
):
    st.session_state["specsheet_topic_name"] = DEFAULT_SPECSHEET_TOPIC
    _default_specsheet_topic = DEFAULT_SPECSHEET_TOPIC

_specsheet_defaults_source = str(_selected_trend_metadata_path or _single_specsheet_run_path)
_previous_auto_version = st.session_state.get("specsheet_version_auto_value")
_current_version = st.session_state.get("specsheet_version")
if (
    st.session_state.get("specsheet_version_auto_source") != _specsheet_defaults_source
    and (
        "specsheet_version" not in st.session_state
        or _current_version == _previous_auto_version
        or str(_current_version or "").endswith(("/performance", "/devops"))
    )
):
    st.session_state["specsheet_version"] = _default_specsheet_version
st.session_state["specsheet_version_auto_source"] = _specsheet_defaults_source
st.session_state["specsheet_version_auto_value"] = _default_specsheet_version

_previous_auto_topic = st.session_state.get("specsheet_topic_auto_value")
_current_topic = st.session_state.get("specsheet_topic_name")
if (
    st.session_state.get("specsheet_topic_auto_source") != _specsheet_defaults_source
    and (
        "specsheet_topic_name" not in st.session_state
        or _current_topic == _previous_auto_topic
    )
):
    st.session_state["specsheet_topic_name"] = _default_specsheet_topic
st.session_state["specsheet_topic_auto_source"] = _specsheet_defaults_source
st.session_state["specsheet_topic_auto_value"] = _default_specsheet_topic

specsheet_cfg_col1, specsheet_cfg_col2, specsheet_cfg_col3 = st.columns([1.4, 1.2, 1.4])
with specsheet_cfg_col1:
    specsheet_project_id = st.text_input(
        "Project ID",
        value=_default_specsheet_project_id,
        key="specsheet_project_id",
    ).strip()
with specsheet_cfg_col2:
    specsheet_version = st.text_input(
        "Version",
        value=_default_specsheet_version,
        key="specsheet_version",
    ).strip()
with specsheet_cfg_col3:
    specsheet_topic_name = st.text_input(
        "Topic name",
        value=_default_specsheet_topic,
        key="specsheet_topic_name",
    ).strip()

specsheet_labels = list(_default_specsheet_labels)
if not selected_specsheet_run_paths:
    st.info("Pick at least one run to build the release spec-sheet.")

if _selected_trend_metadata_text and "specsheet_include_trend" not in st.session_state:
    st.session_state["specsheet_include_trend"] = True

_release_trend_dir_text = ""
_release_trend_status_text = ""
if selected_specsheet_release_contexts:
    for release_context in selected_specsheet_release_contexts[:1]:
        release_dir = release_context.get("release_dir")
        roles = release_context.get("roles", {})
        role_status = []
        if isinstance(roles, dict):
            for role_name in ("performance", "devops"):
                role_info = roles.get(role_name)
                if not isinstance(role_info, dict):
                    continue
                bits = []
                bits.append("summary.json" if role_info.get("has_summary") else "no summary.json")
                bits.append("metadata.yaml" if role_info.get("has_metadata") else "no metadata.yaml")
                if role_info.get("has_summary") and role_info.get("has_metadata"):
                    role_status.append(f"{role_name} ready")
                else:
                    role_status.append(f"{role_name}: {', '.join(bits)}")
        _release_trend_dir_text = path_display(release_dir) if isinstance(release_dir, Path) else "detected"
        if role_status:
            _release_trend_status_text = " · ".join(role_status)

trend_toggle_col, trend_status_col = st.columns([1.1, 2.9])
with trend_toggle_col:
    specsheet_trend_enabled = st.toggle(
        "Include trend data",
        value=bool(st.session_state.get("specsheet_include_trend", bool(_selected_trend_metadata_text))),
        key="specsheet_include_trend",
        help="Save release metadata and include available trend history.",
    )
with trend_status_col:
    if specsheet_trend_enabled:
        trend_status_parts = []
        if _selected_trend_metadata_path is not None and _selected_trend_metadata_text:
            trend_status_parts.append(f"Metadata `{path_display(_selected_trend_metadata_path)}`")
        else:
            trend_status_parts.append("Metadata not saved")
        if _release_trend_dir_text:
            trend_status_parts.append(f"Release `{_release_trend_dir_text}`")
        if _release_trend_status_text:
            trend_status_parts.append(_release_trend_status_text)
        st.caption(" · ".join(trend_status_parts))

trend_metadata_payload = None
trend_metadata_changed = False
trend_metadata_change_confirmed = False
if specsheet_trend_enabled:
    _trend_metadata_source_key = str(_selected_trend_metadata_path) if _selected_trend_metadata_path is not None else "__default__"
    if (
        st.session_state.get("specsheet_trend_metadata_source") != _trend_metadata_source_key
        or "specsheet_trend_metadata_text" not in st.session_state
    ):
        st.session_state["specsheet_trend_metadata_text"] = _selected_trend_metadata_text or DEFAULT_TREND_METADATA_TEXT
        st.session_state["specsheet_trend_metadata_source"] = _trend_metadata_source_key
        st.session_state["specsheet_confirm_metadata_changes"] = False
    trend_metadata_text = st.text_area(
        "Trend metadata YAML",
        key="specsheet_trend_metadata_text",
        height=180,
        help="Required keys: tags, pilot_auto_version, data_count, description, date.",
    )
    trend_metadata_changed = bool(_selected_trend_metadata_text) and (
        trend_metadata_text.strip() != _selected_trend_metadata_text.strip()
    )
    if trend_metadata_changed:
        st.warning("Saved metadata was edited. Confirm before generating.")
        trend_metadata_change_confirmed = st.checkbox(
            "Confirm saved metadata changes",
            key="specsheet_confirm_metadata_changes",
        )
    trend_metadata_status = st.empty()
    try:
        trend_metadata_payload = parse_trend_metadata_text(trend_metadata_text)
        trend_metadata_status.success("Trend metadata looks valid.")
    except Exception as trend_exc:
        trend_metadata_status.error(f"Trend metadata error: {trend_exc}")

selected_trend_metadata_paths: list[Path] | None = None
if specsheet_trend_enabled:
    manual_trend_history = st.checkbox(
        "Choose trend history manually",
        value=bool(st.session_state.get("specsheet_manual_trend_history", False)),
        key="specsheet_manual_trend_history",
        help="Show saved trend releases and include only selected history in the PDF.",
    )
    if manual_trend_history:
        try:
            trend_groups = discover_trend_release_groups()
        except Exception as trend_group_exc:
            trend_groups = []
            st.warning(f"Could not load saved trend history: {trend_group_exc}")

        trend_group_options: dict[str, dict[str, object]] = {}
        for idx, group in enumerate(trend_groups):
            metadata = {}
            for role_name in ("full", "usecase", "devops", "performance_blocks", "unknown"):
                role_job = group.jobs.get(role_name)
                if isinstance(role_job, dict) and isinstance(role_job.get("metadata"), dict):
                    metadata = role_job["metadata"]
                    break
            version = str(metadata.get("pilot_auto_version") or group.display_name or "").strip()
            date = str(metadata.get("date") or "").strip()
            description = str(metadata.get("description") or "").strip()
            roles = ", ".join(sorted(str(role) for role in group.jobs.keys()))
            label_parts = [part for part in (date, version, description) if part]
            option_label = " | ".join(label_parts) or group.display_name or f"Trend history {idx + 1}"
            if roles:
                option_label = f"{option_label} ({roles})"
            option_key = f"{group.group_key}::{idx}"
            metadata_paths = [
                job.get("metadata_path")
                for job in group.jobs.values()
                if isinstance(job, dict) and isinstance(job.get("metadata_path"), Path)
            ]
            if metadata_paths:
                trend_group_options[option_key] = {
                    "label": option_label,
                    "metadata_paths": metadata_paths,
                }

        if trend_group_options:
            trend_option_keys = list(trend_group_options.keys())
            current_trend_selection = st.session_state.get("specsheet_selected_trend_groups")
            safe_trend_selection = (
                _safe_default(current_trend_selection, trend_option_keys)
                if current_trend_selection is not None
                else trend_option_keys
            )
            st.session_state["specsheet_selected_trend_groups"] = safe_trend_selection
            selected_trend_group_keys = st.multiselect(
                "Trend history to include",
                options=trend_option_keys,
                default=safe_trend_selection,
                format_func=lambda key: str(trend_group_options[key]["label"]),
                key="specsheet_selected_trend_groups",
                help="The current release is always included; this controls the saved past trend points.",
            )
            selected_trend_metadata_paths = []
            seen_trend_metadata_paths = set()
            for option_key in selected_trend_group_keys:
                option = trend_group_options.get(option_key)
                if not option:
                    continue
                for metadata_path in option["metadata_paths"]:
                    path_key = str(metadata_path.resolve())
                    if path_key in seen_trend_metadata_paths:
                        continue
                    selected_trend_metadata_paths.append(metadata_path)
                    seen_trend_metadata_paths.add(path_key)
            st.caption(f"Selected {len(selected_trend_group_keys)} trend releases.")
        else:
            selected_trend_metadata_paths = []
            st.info("No saved trend history candidates were found.")

_specsheet_key = {
    "run_paths": [str(path) for path in selected_specsheet_run_paths],
    "project_id": specsheet_project_id,
    "version": specsheet_version,
    "topic_name": specsheet_topic_name,
    "labels": list(specsheet_labels),
    "include_trend": specsheet_trend_enabled,
    "trend_metadata": trend_metadata_payload if specsheet_trend_enabled else None,
    "manual_trend_history": bool(specsheet_trend_enabled and selected_trend_metadata_paths is not None),
    "trend_metadata_paths": (
        [str(path) for path in selected_trend_metadata_paths]
        if specsheet_trend_enabled and selected_trend_metadata_paths is not None
        else None
    ),
    "artifact_kind": "zip" if len(selected_specsheet_run_paths) > 1 else "pdf",
}
_specsheet_ready = (
    st.session_state.get("specsheet_pdf_report_bytes") is not None
    and st.session_state.get("specsheet_pdf_report_key") == _specsheet_key
)

def _release_specsheet_pdf_path(release_context: dict, topic_name: str) -> Path | None:
    release_dir = release_context.get("release_dir")
    if not isinstance(release_dir, Path):
        return None
    specsheet_root = release_dir / "specsheet"
    topic = str(topic_name or "").strip()
    candidates = []
    if topic:
        candidates.append(specsheet_root / topic / "specsheet.pdf")
    candidates.append(specsheet_root / "specsheet.pdf")
    candidates.extend(sorted(specsheet_root.glob("*/*.pdf")))
    for candidate in candidates:
        if candidate.exists() and not candidate.is_dir():
            return candidate
    return None

_release_specsheet_paths = [
    pdf_path
    for pdf_path in (
        _release_specsheet_pdf_path(release_context, specsheet_topic_name)
        for release_context in selected_specsheet_release_contexts
    )
    if pdf_path is not None
]
_generated_specsheet_paths = [
    path_info["specsheet_pdf"]
    for path_info in _active_specsheet_paths
    if path_info["specsheet_pdf"].exists() and is_specsheet_pdf_fresh(path_info["run_dir"])
]
_existing_specsheet_paths = _release_specsheet_paths or _generated_specsheet_paths
_all_selected_specsheet_pdfs_ready = (
    len(selected_specsheet_run_paths) > 0
    and len(_existing_specsheet_paths) == len(selected_specsheet_run_paths)
)
_specsheet_has_existing_pdf = _specsheet_ready or _all_selected_specsheet_pdfs_ready
_specsheet_action_label = (
    "Regenerate Release Spec-sheet PDF"
    if _specsheet_has_existing_pdf
    else "Generate Release Spec-sheet PDF"
)

specsheet_action_col1, specsheet_action_col2 = st.columns([1.2, 2.8])
with specsheet_action_col1:
    if st.button(
        _specsheet_action_label,
        type="secondary" if _specsheet_has_existing_pdf else "primary",
        use_container_width=True,
    ):
        _specsheet_status = st.empty()
        _specsheet_progress = st.progress(0.0)
        try:
            if not specsheet_project_id:
                raise ValueError("Project ID is required.")
            if not specsheet_version:
                raise ValueError("Version is required.")
            if not specsheet_topic_name:
                raise ValueError("Topic name is required.")
            if not selected_specsheet_run_paths:
                raise ValueError("At least one run must be selected.")
            if specsheet_trend_enabled and len(selected_specsheet_run_paths) != 1:
                raise ValueError("Trend-enabled release spec-sheet generation currently supports exactly one run.")
            if specsheet_trend_enabled and trend_metadata_payload is None:
                raise ValueError("Valid trend metadata is required when trend mode is enabled.")
            if specsheet_trend_enabled and trend_metadata_changed and not trend_metadata_change_confirmed:
                raise ValueError("Confirm the metadata.yaml changes before generating.")

            stage_progress = {
                "Using existing up-to-date spec-sheet PDF": 1.0,
                "Loading specsheet data": 0.15,
                "Building abstract and detail sections": 0.2,
                "Validating full trend summary": 0.9,
                "Saving trend metadata": 0.9,
                "Collecting trend history": 0.92,
                "Rendering trend plots": 0.94,
                "Rendering PDF": 0.95,
                "Spec-sheet PDF is ready": 1.0,
            }

            def _update_specsheet_status(message: str) -> None:
                fraction = None
                label_fraction = progress_fraction_from_message(message)
                if "[Full] Generating blocks for labels" in message and label_fraction is not None:
                    fraction = 0.2 + (0.7 - 0.2) * label_fraction
                elif (
                    "[Full] Generating annotation count blocks for labels" in message
                    and label_fraction is not None
                ):
                    fraction = 0.7 + (0.9 - 0.7) * label_fraction
                elif label_fraction is not None and "Processing pkl files" in message:
                    fraction = 0.02 + (0.12 - 0.02) * label_fraction
                else:
                    fraction = stage_progress.get(message, 0.05)
                _specsheet_progress.progress(fraction)
                _specsheet_status.info(f"Generating release spec-sheet: {message}")

            generated_pdfs: list[tuple[Path, bool]] = []
            for idx, run_path in enumerate(selected_specsheet_run_paths, start=1):
                _update_specsheet_status(f"Run {idx}/{len(selected_specsheet_run_paths)}: {get_run_display_name(run_path)}")
                if specsheet_trend_enabled and trend_metadata_payload is not None:
                    if _selected_trend_metadata_path is not None and trend_metadata_changed:
                        _selected_trend_metadata_path.write_text(
                            yaml.safe_dump(trend_metadata_payload, allow_unicode=True, sort_keys=False),
                            encoding="utf-8",
                        )
                    if len(selected_specsheet_release_contexts) == 1:
                        roles = selected_specsheet_release_contexts[0].get("roles", {})
                        if isinstance(roles, dict):
                            for role_info in roles.values():
                                if not isinstance(role_info, dict) or not role_info.get("has_summary"):
                                    continue
                                role_run_dir = role_info.get("run_dir")
                                if isinstance(role_run_dir, Path):
                                    write_trend_metadata(role_run_dir, trend_metadata_payload)
                pdf_path, generated = generate_specsheet_pdf(
                    run_path,
                    project_id=specsheet_project_id,
                    version=specsheet_version,
                    labels=specsheet_labels,
                    topic_name=specsheet_topic_name,
                    include_trend=specsheet_trend_enabled,
                    trend_metadata=trend_metadata_payload,
                    trend_metadata_paths=selected_trend_metadata_paths,
                    force=True,
                    progress_callback=_update_specsheet_status,
                )
                generated_pdfs.append((pdf_path, generated))

            if len(generated_pdfs) == 1:
                download_name = generated_pdfs[0][0].name
                download_bytes = generated_pdfs[0][0].read_bytes()
                download_mime = "application/pdf"
            else:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for pdf_path, _ in generated_pdfs:
                        zf.write(pdf_path, arcname=f"{pdf_path.parent.parent.name}/{pdf_path.name}")
                download_name = "specsheet_reports.zip"
                download_bytes = zip_buffer.getvalue()
                download_mime = "application/zip"

            st.session_state["specsheet_pdf_report_bytes"] = download_bytes
            st.session_state["specsheet_pdf_report_key"] = _specsheet_key
            st.session_state["specsheet_pdf_report_name"] = download_name
            st.session_state["specsheet_pdf_report_mime"] = download_mime
            _specsheet_ready = True
            _specsheet_progress.progress(1.0)
            if any(generated for _, generated in generated_pdfs):
                if len(generated_pdfs) == 1:
                    _specsheet_status.success("Release spec-sheet PDF is ready.")
                else:
                    _specsheet_status.success("Release spec-sheet files are ready.")
            else:
                if len(generated_pdfs) == 1:
                    _specsheet_status.success("Using the existing up-to-date release spec-sheet PDF.")
                else:
                    _specsheet_status.success("Using the existing up-to-date release spec-sheet files.")
        except Exception as e:
            st.session_state.pop("specsheet_pdf_report_bytes", None)
            st.session_state.pop("specsheet_pdf_report_key", None)
            st.session_state.pop("specsheet_pdf_report_name", None)
            st.session_state.pop("specsheet_pdf_report_mime", None)
            _specsheet_status.error(f"Spec-sheet generation failed: {e}")
with specsheet_action_col2:
    if _specsheet_ready:
        ready_col, download_col = st.columns([1.7, 1.0])
        with ready_col:
            st.caption("Ready · generated release spec-sheet")
        with download_col:
            st.download_button(
                "Download PDF",
                data=st.session_state["specsheet_pdf_report_bytes"],
                file_name=st.session_state.get("specsheet_pdf_report_name", "specsheet.pdf"),
                mime=st.session_state.get("specsheet_pdf_report_mime", "application/pdf"),
                use_container_width=True,
            )
    elif _all_selected_specsheet_pdfs_ready:
        if len(_existing_specsheet_paths) == 1:
            _disk_pdf_path = _existing_specsheet_paths[0]
            ready_col, download_col = st.columns([1.7, 1.0])
            with ready_col:
                st.caption("Ready · existing release spec-sheet")
            with download_col:
                st.download_button(
                    "Download PDF",
                    data=_disk_pdf_path.read_bytes(),
                    file_name=_disk_pdf_path.name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            _zip_buffer = io.BytesIO()
            with zipfile.ZipFile(_zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for pdf_path in _existing_specsheet_paths:
                    zf.write(pdf_path, arcname=f"{pdf_path.parent.parent.name}/{pdf_path.name}")
            ready_col, download_col = st.columns([1.7, 1.0])
            with ready_col:
                st.caption("Ready · existing release spec-sheets")
            with download_col:
                st.download_button(
                    "Download ZIP",
                    data=_zip_buffer.getvalue(),
                    file_name="specsheet_reports.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
    else:
        if len(selected_specsheet_run_paths) == 1:
            _single_paths = _active_specsheet_paths[0]
