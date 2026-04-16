from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

from lib.overview_url_hydrate import try_hydrate_session_from_overview_query_params
from lib.page_chrome import inject_app_page_styles, render_loaded_data_section, render_page_hero, section_header
from lib.path_utils import list_run_directories, path_display
from lib.prediction_eval import prepare_future_matched_df, build_future_mode_track_summary_from_matched


st.set_page_config(
    layout="wide",
    page_title="Prediction Evaluation",
    page_icon="🧭",
    initial_sidebar_state="expanded",
)
inject_app_page_styles()
st.markdown(
    """
    <style>
    .pred-chip-row {
        display:flex;
        flex-wrap:wrap;
        gap:0.55rem;
        margin:0.35rem 0 1.0rem 0;
    }
    .pred-chip {
        border:1px solid #d6dee7;
        border-radius:999px;
        padding:0.38rem 0.8rem;
        background:linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%);
        color:#254051;
        font-size:0.82rem;
        font-weight:600;
    }
    .pred-card {
        border:1px solid #dce6ee;
        border-radius:18px;
        background:linear-gradient(145deg, #fcfefe 0%, #f6fafb 48%, #f7fbff 100%);
        padding:1rem 1.1rem;
        box-shadow:0 18px 45px -28px rgba(13, 45, 58, 0.28);
        min-height:128px;
    }
    .pred-card-kicker {
        font-size:0.68rem;
        letter-spacing:0.14em;
        text-transform:uppercase;
        color:#5b7283;
        font-weight:800;
    }
    .pred-card-value {
        font-size:1.8rem;
        line-height:1.05;
        letter-spacing:-0.04em;
        color:#0f172a;
        font-weight:850;
        margin-top:0.5rem;
    }
    .pred-card-note {
        margin-top:0.55rem;
        color:#4a6577;
        font-size:0.88rem;
        line-height:1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_COLORS = {
    "ink": "#12344d",
    "teal": "#0f766e",
    "blue": "#1d4ed8",
    "amber": "#c27803",
    "rose": "#be123c",
    "slate": "#475569",
}
DEFAULT_TOPIC = "perception.object_recognition.objects"
CHECKPOINTS = (1.0, 3.0, 5.0)
METRIC_ORDER = [
    "minADE@1s",
    "minADE@3s",
    "minADE@5s",
    "minFDE@1s",
    "minFDE@3s",
    "minFDE@5s",
]
APP_CACHE_ROOT = ".dashboard_cache"
ARTIFACT_DIRNAME = "prediction_eval_cache"
ARTIFACT_TABLES = ["label_summary", "distance_summary", "polar_summary"]
R_MAX, R_STEP, R_INI = 200, 20, 0
THETA_STEP, THETA_INI = 60, -60
THETA_MAX = THETA_INI + 360
R_LABELS = [f"{i}-{i + R_STEP}" for i in range(R_INI, R_MAX, R_STEP)]
R_EDGES = np.arange(R_INI, R_MAX + R_STEP, R_STEP)
THETA_LABELS = [f"{i}-{i + THETA_STEP}" for i in range(THETA_INI, THETA_MAX, THETA_STEP)]
THETA_EDGES_DEG = np.arange(THETA_INI, THETA_MAX + THETA_STEP, THETA_STEP)
DISTANCE_BIN_ORDER = [
    "0-20 m",
    "20-40 m",
    "40-60 m",
    "60-80 m",
    "80-100 m",
    "100-120 m",
    "120-140 m",
    "140-160 m",
    "160-180 m",
    "180-200 m",
    "200+ m",
]


def render_stat_card(kicker: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="pred-card">
          <div class="pred-card-kicker">{kicker}</div>
          <div class="pred-card-value">{value}</div>
          <div class="pred-card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ordered_distance_bins(values: list[str] | pd.Index) -> list[str]:
    present = {str(v) for v in values if pd.notna(v)}
    ordered = [v for v in DISTANCE_BIN_ORDER if v in present]
    leftovers = sorted(present - set(ordered))
    return ordered + leftovers


def build_distance_ring_figure(metric_df: pd.DataFrame, label_order: list[str], metric_name: str) -> go.Figure:
    ring_order = ordered_distance_bins(metric_df["r"].tolist())
    pivot = (
        metric_df.pivot(index="label", columns="r", values="value")
        .reindex(index=label_order)
        .reindex(columns=ring_order)
    )
    theta_width = 360 / max(len(label_order), 1)
    theta_centers = [i * theta_width for i in range(len(label_order))]
    zmin = float(np.nanmin(pivot.values)) if np.isfinite(np.nanmin(pivot.values)) else 0.0
    zmax = float(np.nanmax(pivot.values)) if np.isfinite(np.nanmax(pivot.values)) else 1.0
    if zmin == zmax:
        zmax = zmin + 1.0

    fig = go.Figure()
    for ring_idx, ring_name in enumerate(ring_order):
        vals = pivot[ring_name].tolist()
        fig.add_trace(
            go.Barpolar(
                r=[1.0] * len(label_order),
                base=[ring_idx] * len(label_order),
                theta=theta_centers,
                width=[theta_width * 0.92] * len(label_order),
                marker=dict(
                    color=vals,
                    colorscale="YlOrRd",
                    cmin=zmin,
                    cmax=zmax,
                    line=dict(color="rgba(255,255,255,0.35)", width=1),
                    colorbar=dict(title="m") if ring_idx == len(ring_order) - 1 else None,
                ),
                customdata=np.array([[label_order[i], ring_name, vals[i]] for i in range(len(label_order))], dtype=object),
                hovertemplate="label=%{customdata[0]}<br>distance=%{customdata[1]}<br>value=%{customdata[2]:.3f} m<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=metric_name,
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
        polar=dict(
            radialaxis=dict(
                tickmode="array",
                tickvals=list(range(len(ring_order))),
                ticktext=ring_order,
                angle=90,
                gridcolor="rgba(148,163,184,0.25)",
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=theta_centers,
                ticktext=label_order,
                rotation=90,
                direction="clockwise",
                gridcolor="rgba(148,163,184,0.20)",
            ),
            bgcolor="rgba(248,250,252,0.75)",
        ),
    )
    return fig


def build_theta_ring_figure(label_polar: pd.DataFrame, metric_name: str, label_name: str, value_col: str, *, delta_mode: bool) -> go.Figure:
    theta_order = THETA_LABELS
    radial_order = [r for r in R_LABELS if r in set(label_polar["r"].astype(str))]
    pivot = (
        label_polar.pivot(index="r", columns="theta", values=value_col)
        .reindex(index=radial_order, columns=theta_order)
    )
    theta_width = 360 / max(len(theta_order), 1)
    theta_centers = [i * theta_width for i in range(len(theta_order))]

    values = pivot.values.astype(float) if pivot.size else np.array([[0.0]])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        zmin, zmax = (-1.0, 1.0) if delta_mode else (0.0, 1.0)
    else:
        if delta_mode:
            bound = float(np.nanmax(np.abs(finite))) or 1.0
            zmin, zmax = -bound, bound
        else:
            zmin, zmax = float(np.nanmin(finite)), float(np.nanmax(finite))
            if zmin == zmax:
                zmax = zmin + 1.0

    fig = go.Figure()
    for ring_idx, ring_name in enumerate(radial_order):
        vals = pivot.loc[ring_name].tolist()
        fig.add_trace(
            go.Barpolar(
                r=[1.0] * len(theta_order),
                base=[ring_idx] * len(theta_order),
                theta=theta_centers,
                width=[theta_width * 0.92] * len(theta_order),
                marker=dict(
                    color=vals,
                    colorscale="RdBu" if delta_mode else "YlOrRd",
                    cmin=zmin,
                    cmax=zmax,
                    line=dict(color="rgba(255,255,255,0.32)", width=1),
                    colorbar=dict(title="m") if ring_idx == len(radial_order) - 1 else None,
                ),
                customdata=np.array([[theta_order[i], ring_name, vals[i]] for i in range(len(theta_order))], dtype=object),
                hovertemplate=("theta=%{customdata[0]}<br>distance=%{customdata[1]}<br>Δ=%{customdata[2]:+.3f} m<extra></extra>" if delta_mode else "theta=%{customdata[0]}<br>distance=%{customdata[1]}<br>value=%{customdata[2]:.3f} m<extra></extra>"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"{label_name}{' (B - A)' if delta_mode else ''}",
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
        polar=dict(
            radialaxis=dict(
                tickmode="array",
                tickvals=list(range(len(radial_order))),
                ticktext=radial_order,
                angle=90,
                gridcolor="rgba(148,163,184,0.22)",
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=theta_centers,
                ticktext=theta_order,
                rotation=90,
                direction="clockwise",
                gridcolor="rgba(148,163,184,0.18)",
            ),
            bgcolor="rgba(248,250,252,0.75)",
        ),
    )
    return fig


def render_compare_stat_card(kicker: str, a_value: float | None, b_value: float | None, note: str) -> None:
    delta = None
    if a_value is not None and b_value is not None and pd.notna(a_value) and pd.notna(b_value):
        delta = float(b_value) - float(a_value)
    delta_text = f"Δ {delta:+.2f} m" if delta is not None else "Δ n/a"
    st.markdown(
        f"""
        <div class="pred-card">
          <div class="pred-card-kicker">{kicker}</div>
          <div class="pred-card-value">A {a_value:.2f} / B {b_value:.2f}</div>
          <div class="pred-card-note">{delta_text}<br>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_has_prediction_source(run_path: Path) -> bool:
    return (run_path / "future.parquet").exists() or (run_path / "future.csv").exists()


def _prediction_source_path(run_path: Path) -> Path | None:
    parquet_path = run_path / "future.parquet"
    if parquet_path.exists():
        return parquet_path
    csv_path = run_path / "future.csv"
    if csv_path.exists():
        return csv_path
    return None


@st.cache_data(show_spinner=False)
def load_prediction_metadata(run_path_str: str) -> dict[str, float | int]:
    future_path = _prediction_source_path(Path(run_path_str))
    if future_path is None:
        return {"row_count": 0, "row_groups": 0, "file_size_mb": 0.0, "source_kind": "missing"}
    if future_path.suffix == ".parquet":
        parquet_file = pq.ParquetFile(future_path)
        return {
            "row_count": int(parquet_file.metadata.num_rows),
            "row_groups": int(parquet_file.metadata.num_row_groups),
            "file_size_mb": future_path.stat().st_size / (1024 * 1024),
            "source_kind": "parquet",
        }
    return {
        "row_count": 0,
        "row_groups": 0,
        "file_size_mb": future_path.stat().st_size / (1024 * 1024),
        "source_kind": "csv",
    }


def get_prediction_cache_dir(run_path: Path) -> Path:
    return run_path / APP_CACHE_ROOT / ARTIFACT_DIRNAME


def get_prediction_manifest_path(run_path: Path) -> Path:
    return get_prediction_cache_dir(run_path) / "manifest.json"


def get_prediction_table_path(run_path: Path, table_name: str) -> Path:
    return get_prediction_cache_dir(run_path) / f"{table_name}.parquet"


def load_prediction_artifact_manifest(run_path: Path) -> dict[str, object] | None:
    manifest_path = get_prediction_manifest_path(run_path)
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def prediction_artifacts_ready(run_path: Path) -> bool:
    manifest = load_prediction_artifact_manifest(run_path)
    future_path = _prediction_source_path(run_path)
    if manifest is None or future_path is None or not future_path.exists():
        return False
    if manifest.get("future_mtime_ns") != future_path.stat().st_mtime_ns:
        return False
    return all(get_prediction_table_path(run_path, name).exists() for name in ARTIFACT_TABLES)


def _noop_progress(_: float, __: str) -> None:
    return None


def save_prediction_artifacts(
    run_path: Path,
    artifacts: dict[str, pd.DataFrame],
    progress_callback: Callable[[float, str], None] | None = None,
) -> None:
    report = progress_callback or _noop_progress
    cache_dir = get_prediction_cache_dir(run_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_tables = max(len(ARTIFACT_TABLES), 1)
    for idx, name in enumerate(ARTIFACT_TABLES, start=1):
        report(0.88 + (0.09 * idx / total_tables), f"Saving `{name}` summary...")
        artifacts[name].to_parquet(get_prediction_table_path(run_path, name), index=False)
    manifest = {
        "future_mtime_ns": _prediction_source_path(run_path).stat().st_mtime_ns,
        "table_names": ARTIFACT_TABLES,
    }
    get_prediction_manifest_path(run_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report(1.0, "Prediction summary cache is ready.")


@st.cache_data(show_spinner=False)
def load_saved_prediction_artifacts(run_path_str: str) -> dict[str, pd.DataFrame]:
    run_path = Path(run_path_str)
    out: dict[str, pd.DataFrame] = {}
    for name in ARTIFACT_TABLES:
        out[name] = pd.read_parquet(get_prediction_table_path(run_path, name))
    return out


def _build_prediction_eval_artifacts_impl(
    run_path_str: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    report = progress_callback or _noop_progress
    run_path = Path(run_path_str)
    future_path = _prediction_source_path(run_path)
    if future_path is None:
        raise FileNotFoundError(f"No future.parquet or future.csv found in {run_path}")
    report(0.05, f"Reading `{future_path.name}`...")
    future_cols = [
        "source",
        "label",
        "x",
        "y",
        "tx",
        "ty",
        "mode",
        "future_index",
        "relative_time",
        "pair_uuid",
        "frame_index",
        "scenario_name",
        "suite_name",
        "uuid",
        "confidence",
    ]
    if future_path.suffix == ".parquet":
        schema = pq.read_schema(future_path).names
        optional_cols = [c for c in ["topic_name"] if c in schema]
        future_df = pd.read_parquet(future_path, columns=future_cols + optional_cols)
    else:
        future_df = pd.read_csv(future_path, usecols=lambda c: c in set(future_cols + ["topic_name"]))
    if "topic_name" in future_df.columns:
        report(0.18, "Filtering the default prediction topic...")
        topic_values = future_df["topic_name"].dropna().astype(str).unique().tolist()
        if DEFAULT_TOPIC in topic_values:
            future_df = future_df[future_df["topic_name"].astype(str) == DEFAULT_TOPIC].copy()

    report(0.3, "Matching prediction tracks against GT...")
    matched_df = prepare_future_matched_df(future_df, time_step=0.1, max_error_m=100.0)
    report(0.45, "Computing per-track ADE/FDE summaries...")
    track_summary = build_future_mode_track_summary_from_matched(matched_df, checkpoints=CHECKPOINTS)
    if track_summary.empty:
        report(0.85, "No matched tracks were found. Creating empty summary tables...")
        empty = pd.DataFrame()
        return {
            "label_summary": empty,
            "distance_summary": empty,
            "polar_summary": empty,
        }

    report(0.55, "Preparing radial and angular bins...")
    gt_start = future_df[future_df["source"].astype(str).str.upper() == "GT"].copy()
    gt_start["frame_index_num"] = pd.to_numeric(gt_start["frame_index"], errors="coerce")
    gt_start["relative_time_num"] = pd.to_numeric(gt_start["relative_time"], errors="coerce")
    gt_start["x"] = pd.to_numeric(gt_start["x"], errors="coerce")
    gt_start["y"] = pd.to_numeric(gt_start["y"], errors="coerce")
    gt_start = (
        gt_start.sort_values("relative_time_num")
        .groupby(["suite_name", "scenario_name", "frame_index_num", "uuid"], dropna=False)
        .first()
        .reset_index()
        .rename(columns={"uuid": "uuid_gt", "label": "label_gt"})
    )
    gt_start["r_val"] = np.hypot(gt_start["x"], gt_start["y"])
    raw_deg = np.degrees(np.arctan2(gt_start["y"], gt_start["x"]))
    gt_start["theta_val"] = ((raw_deg - THETA_INI) % 360) + THETA_INI
    gt_start["r"] = pd.cut(gt_start["r_val"], bins=R_EDGES, right=False, include_lowest=True, labels=R_LABELS)
    gt_start["theta"] = pd.cut(
        gt_start["theta_val"],
        bins=THETA_EDGES_DEG,
        right=False,
        include_lowest=True,
        labels=THETA_LABELS,
    )
    track_summary = track_summary.merge(
        gt_start[["suite_name", "scenario_name", "frame_index_num", "uuid_gt", "r", "theta"]],
        on=["suite_name", "scenario_name", "frame_index_num", "uuid_gt"],
        how="left",
    )

    labels = sorted(str(v) for v in track_summary["label_gt"].dropna().unique() if str(v).strip())

    label_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    polar_rows: list[dict[str, object]] = []

    total_labels = max(len(labels), 1)
    for idx, label_name in enumerate(labels, start=1):
        report(0.62 + (0.2 * idx / total_labels), f"Aggregating metrics for `{label_name}` ({idx}/{total_labels})...")
        scoped = track_summary[track_summary["label_gt"].astype(str) == label_name].copy()
        row: dict[str, object] = {
            "label": label_name,
            "future_rows": int(len(scoped)),
        }
        for metric_name in METRIC_ORDER:
            if metric_name not in scoped.columns:
                row[metric_name] = None
                continue

            near = scoped.loc[scoped["start_distance_m"] <= 60.0, metric_name].dropna()
            row[metric_name] = float(near.mean()) if not near.empty else None

            around_df = (
                scoped.groupby("distance_bin", observed=False)[metric_name]
                .mean()
                .reset_index()
                .rename(columns={"distance_bin": "r"})
            )
            if not around_df.empty:
                for rec in around_df[["r", metric_name]].to_dict("records"):
                    distance_rows.append(
                        {
                            "label": label_name,
                            "metric": metric_name,
                            "r": rec["r"],
                            "value": rec[metric_name],
                        }
                    )

            polar_df = (
                scoped.groupby(["r", "theta"], observed=False)[metric_name]
                .mean()
                .reset_index()
                .dropna(subset=[metric_name])
            )
            if not polar_df.empty:
                polar_df["label"] = label_name
                polar_df["metric"] = metric_name
                polar_df = polar_df.rename(columns={metric_name: "value"})
                polar_rows.extend(polar_df[["label", "metric", "r", "theta", "value"]].to_dict("records"))

        label_rows.append(row)

    label_summary = pd.DataFrame(label_rows)
    if not label_summary.empty:
        report(0.84, "Finalizing overall summary row...")
        total_rows = float(label_summary["future_rows"].sum())
        overall_row: dict[str, object] = {
            "label": "All",
            "future_rows": int(total_rows),
        }
        for metric_name in METRIC_ORDER:
            valid = label_summary[["future_rows", metric_name]].dropna()
            if valid.empty or float(valid["future_rows"].sum()) <= 0:
                overall_row[metric_name] = None
            else:
                overall_row[metric_name] = float(
                    (valid["future_rows"] * valid[metric_name]).sum() / valid["future_rows"].sum()
                )
        label_summary = pd.concat([pd.DataFrame([overall_row]), label_summary], ignore_index=True)
    distance_summary = pd.DataFrame(distance_rows)
    polar_summary = pd.DataFrame(polar_rows)
    return {
        "label_summary": label_summary,
        "distance_summary": distance_summary,
        "polar_summary": polar_summary,
    }


@st.cache_data(show_spinner=False)
def build_prediction_eval_artifacts(run_path_str: str) -> dict[str, pd.DataFrame]:
    return _build_prediction_eval_artifacts_impl(run_path_str)


def build_prediction_artifacts_with_progress(run_path: Path, build_label: str) -> None:
    progress_slot = st.empty()
    status_slot = st.empty()
    progress_bar = progress_slot.progress(0, text=f"Starting {build_label} prediction summary build...")

    def report(fraction: float, message: str) -> None:
        bounded_fraction = max(0.0, min(1.0, float(fraction)))
        progress_bar.progress(int(round(bounded_fraction * 100)), text=message)
        status_slot.caption(f"{build_label}: {message}")

    artifacts = _build_prediction_eval_artifacts_impl(str(run_path), progress_callback=report)
    save_prediction_artifacts(run_path, artifacts, progress_callback=report)
    st.cache_data.clear()
    st.rerun()


def merge_label_compare(label_a: pd.DataFrame, label_b: pd.DataFrame) -> pd.DataFrame:
    merged = label_a.merge(label_b, on="label", how="outer", suffixes=("_A", "_B"))
    for metric in METRIC_ORDER:
        merged[f"{metric}_delta"] = merged[f"{metric}_B"] - merged[f"{metric}_A"]
    return merged


def merge_distance_compare(distance_a: pd.DataFrame, distance_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = distance_a.copy()
    base["run"] = "A"
    cand = distance_b.copy()
    cand["run"] = "B"
    both = pd.concat([base, cand], ignore_index=True)
    delta = distance_a.merge(distance_b, on=["label", "metric", "r"], how="outer", suffixes=("_A", "_B"))
    delta["value_delta"] = delta["value_B"] - delta["value_A"]
    return both, delta


def merge_polar_compare(polar_a: pd.DataFrame, polar_b: pd.DataFrame) -> pd.DataFrame:
    delta = polar_a.merge(polar_b, on=["label", "metric", "r", "theta"], how="outer", suffixes=("_A", "_B"))
    delta["value_delta"] = delta["value_B"] - delta["value_A"]
    return delta


run_dirs = list_run_directories()
run_dirs = [p for p in run_dirs if _run_has_prediction_source(p)]
run_names = [p.name for p in run_dirs]
if not run_names:
    st.warning("No run directories with `future.parquet` or `future.csv` found under `data/`.")
    st.stop()

try_hydrate_session_from_overview_query_params()
mode_default = "Compare Mode" if st.session_state.get("mode") == "Compare Mode" else "Single Run"
mode = st.sidebar.selectbox("Mode", ["Single Run", "Compare Mode"], index=0 if mode_default == "Single Run" else 1)

default_run_name = st.session_state.get("runA", {}).get("path").name if st.session_state.get("runA") else run_names[0]
if default_run_name not in run_names:
    default_run_name = run_names[0]

selected_run_a = st.sidebar.selectbox(
    "Baseline (A)" if mode == "Compare Mode" else "Run",
    run_names,
    index=run_names.index(default_run_name),
    help="Select a run directory containing `future.parquet`.",
)
selected_run_b = None
if mode == "Compare Mode":
    compare_candidates = [n for n in run_names if n != selected_run_a] or run_names
    default_b = st.session_state.get("runB", {}).get("path").name if st.session_state.get("runB") else compare_candidates[0]
    if default_b not in compare_candidates:
        default_b = compare_candidates[0]
    selected_run_b = st.sidebar.selectbox("Candidate (B)", compare_candidates, index=compare_candidates.index(default_b))

run_path_a = next(p for p in run_dirs if p.name == selected_run_a)
run_path_b = next((p for p in run_dirs if p.name == selected_run_b), None)
metadata_a = load_prediction_metadata(str(run_path_a))
cache_ready_a = prediction_artifacts_ready(run_path_a)
metadata_b = load_prediction_metadata(str(run_path_b)) if run_path_b is not None else None
cache_ready_b = prediction_artifacts_ready(run_path_b) if run_path_b is not None else False

if mode == "Compare Mode" and run_path_b is not None:
    render_loaded_data_section(
        [
            ("Baseline · A", path_display(run_path_a)),
            ("Candidate · B", path_display(run_path_b)),
        ]
    )
else:
    render_loaded_data_section([("Prediction run", path_display(run_path_a))])
render_page_hero(
    kicker="Prediction quality",
    title="Prediction evaluation",
    description=(
        "ADE/FDE summaries from `future.parquet`, computed from the cached prediction summary artifacts "
        "and presented as interactive cards, ladders, and polar maps."
    ),
    mode=mode,
    secondary_badge_inner_html="Prediction cache",
)
st.markdown(
    f"""
    <div class="pred-chip-row">
      <div class="pred-chip">A: {int(metadata_a['row_count']):,} future rows</div>
      <div class="pred-chip">A: {metadata_a['file_size_mb']:.1f} MB {metadata_a['source_kind']}</div>
      <div class="pred-chip">A cache: {'ready' if cache_ready_a else 'not built'}</div>
      {f'<div class="pred-chip">B: {int(metadata_b["row_count"]):,} future rows</div>' if metadata_b else ''}
      {f'<div class="pred-chip">B: {metadata_b["file_size_mb"]:.1f} MB {metadata_b["source_kind"]}</div>' if metadata_b else ''}
      {f'<div class="pred-chip">B cache: {"ready" if cache_ready_b else "not built"}</div>' if metadata_b else ''}
    </div>
    """,
    unsafe_allow_html=True,
)

build_col, info_col = st.columns([0.34, 0.66])
with build_col:
    build_clicked_a = st.button("Build A Summary", type="primary", use_container_width=True)
    build_clicked_b = st.button("Build B Summary", use_container_width=True) if mode == "Compare Mode" and run_path_b is not None else False
with info_col:
    if mode == "Compare Mode":
        status_lines = [
            f"A `{selected_run_a}`: {'ready' if cache_ready_a else 'not built'}",
            f"B `{selected_run_b}`: {'ready' if cache_ready_b else 'not built'}" if selected_run_b else "",
        ]
        if cache_ready_a and cache_ready_b:
            st.success("Compare result is ready. Both cached summaries are available.")
        else:
            needed = []
            if not cache_ready_a:
                needed.append("Build A Summary")
            if not cache_ready_b:
                needed.append("Build B Summary")
            st.info("Compare mode status:\n\n" + "\n\n".join([x for x in status_lines if x]) + f"\n\nNext step: press {' and '.join(needed)}.")
    elif cache_ready_a:
        st.success("Compact ADE/FDE summary tables are available for fast loading.")
    else:
        st.info(f"Run `{selected_run_a}` is not cached yet. Press Build A Summary to generate the result.")

if build_clicked_a:
    build_prediction_artifacts_with_progress(run_path_a, "A")

if build_clicked_b and run_path_b is not None:
    build_prediction_artifacts_with_progress(run_path_b, "B")

if (mode == "Single Run" and not cache_ready_a) or (mode == "Compare Mode" and (not cache_ready_a or not cache_ready_b)):
    section_header(
        "Build Once, Open Fast",
        "This page now stays responsive by loading only precomputed ADE/FDE summaries instead of processing the full future parquet on navigation.",
    )
    st.stop()

artifacts_a = load_saved_prediction_artifacts(str(run_path_a))
label_summary = artifacts_a["label_summary"].copy()
distance_summary = artifacts_a["distance_summary"].copy()
polar_summary = artifacts_a["polar_summary"].copy()
artifacts_b = load_saved_prediction_artifacts(str(run_path_b)) if mode == "Compare Mode" and run_path_b is not None else None

if label_summary.empty:
    st.warning("No prediction summary data is available for this run.")
    st.stop()

available_labels = [x for x in label_summary["label"].astype(str).tolist() if x != "All"]

overall_row = label_summary[label_summary["label"].astype(str) == "All"]
if overall_row.empty:
    overall_row = label_summary.head(1)
overall = overall_row.iloc[0]
compare_label = merge_label_compare(label_summary, artifacts_b["label_summary"]) if artifacts_b is not None else None
distance_both = distance_delta = None
polar_delta = None
if artifacts_b is not None:
    distance_both, distance_delta = merge_distance_compare(distance_summary, artifacts_b["distance_summary"])
    polar_delta = merge_polar_compare(polar_summary, artifacts_b["polar_summary"])

section_header(
    "At A Glance",
    "These cards mirror the kind of abstract specsheet readout we need in product review, but in a faster dashboard form.",
)
cards = st.columns(3)
with cards[0]:
    if compare_label is not None:
        overall_cmp = compare_label[compare_label["label"] == "All"].iloc[0]
        render_compare_stat_card("minADE@1s <= 60m", overall_cmp["minADE@1s_A"], overall_cmp["minADE@1s_B"], "Best-of-K average displacement error within the near operating zone.")
    else:
        render_stat_card("minADE@1s <= 60m", f"{overall['minADE@1s']:.2f} m" if pd.notna(overall["minADE@1s"]) else "n/a", "Best-of-K average displacement error within the near operating zone.")
with cards[1]:
    if compare_label is not None:
        overall_cmp = compare_label[compare_label["label"] == "All"].iloc[0]
        render_compare_stat_card("minADE@3s <= 60m", overall_cmp["minADE@3s_A"], overall_cmp["minADE@3s_B"], "Mid-horizon shape fidelity aligned with the specsheet future metric.")
    else:
        render_stat_card("minADE@3s <= 60m", f"{overall['minADE@3s']:.2f} m" if pd.notna(overall["minADE@3s"]) else "n/a", "Mid-horizon shape fidelity aligned with the specsheet future metric.")
with cards[2]:
    if compare_label is not None:
        overall_cmp = compare_label[compare_label["label"] == "All"].iloc[0]
        render_compare_stat_card("minFDE@3s <= 60m", overall_cmp["minFDE@3s_A"], overall_cmp["minFDE@3s_B"], "Where the endpoint lands matters most in review discussions, so this gets prime placement.")
    else:
        render_stat_card("minFDE@3s <= 60m", f"{overall['minFDE@3s']:.2f} m" if pd.notna(overall["minFDE@3s"]) else "n/a", "Where the endpoint lands matters most in review discussions, so this gets prime placement.")

cards2 = st.columns(3)
with cards2[0]:
    if compare_label is not None:
        overall_cmp = compare_label[compare_label["label"] == "All"].iloc[0]
        render_compare_stat_card("minADE@5s <= 60m", overall_cmp["minADE@5s_A"], overall_cmp["minADE@5s_B"], "Longer horizon path quality, still scoped to the near-range summary window.")
    else:
        render_stat_card("minADE@5s <= 60m", f"{overall['minADE@5s']:.2f} m" if pd.notna(overall["minADE@5s"]) else "n/a", "Longer horizon path quality, still scoped to the near-range summary window.")
with cards2[1]:
    if compare_label is not None:
        overall_cmp = compare_label[compare_label["label"] == "All"].iloc[0]
        render_compare_stat_card("minFDE@1s <= 60m", overall_cmp["minFDE@1s_A"], overall_cmp["minFDE@1s_B"], "Short horizon endpoint stability.")
    else:
        render_stat_card("minFDE@1s <= 60m", f"{overall['minFDE@1s']:.2f} m" if pd.notna(overall["minFDE@1s"]) else "n/a", "Short horizon endpoint stability.")
with cards2[2]:
    if compare_label is not None:
        overall_cmp = compare_label[compare_label["label"] == "All"].iloc[0]
        render_compare_stat_card("minFDE@5s <= 60m", overall_cmp["minFDE@5s_A"], overall_cmp["minFDE@5s_B"], "Longest specsheet-style endpoint metric.")
    else:
        render_stat_card("minFDE@5s <= 60m", f"{overall['minFDE@5s']:.2f} m" if pd.notna(overall["minFDE@5s"]) else "n/a", f"Longest specsheet-style endpoint metric. Source rows processed: {int(overall['future_rows']):,}.")

section_header(
    "Label Performance",
    "All labels are shown together so you can compare actor classes without touching filters.",
)
label_view = label_summary[label_summary["label"].isin(available_labels)].copy()
if compare_label is not None:
    cmp_view = compare_label[compare_label["label"].isin(available_labels)].copy()
    delta_long = cmp_view.melt(
        id_vars=["label"],
        value_vars=[f"{m}_delta" for m in METRIC_ORDER],
        var_name="metric",
        value_name="value",
    )
    delta_long["metric"] = delta_long["metric"].str.replace("_delta", "", regex=False)
    heat = delta_long.pivot(index="label", columns="metric", values="value").reindex(columns=METRIC_ORDER)
    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=list(heat.columns),
            y=list(heat.index),
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:+.2f}" if pd.notna(v) else "-" for v in row] for row in heat.values],
            texttemplate="%{text}",
            hovertemplate="label=%{y}<br>metric=%{x}<br>Δ=%{z:+.3f} m<extra></extra>",
        )
    )
    fig.update_layout(
        title="ADE/FDE delta matrix: B - A within <= 60m",
        xaxis_title="Metric",
        yaxis_title="Label",
        height=max(360, 70 * len(heat.index)),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig, width="stretch")
elif not label_view.empty:
    label_long = label_view.melt(
        id_vars=["label"],
        value_vars=METRIC_ORDER,
        var_name="metric",
        value_name="value",
    )
    heat = label_long.pivot(index="label", columns="metric", values="value").reindex(columns=METRIC_ORDER)
    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=list(heat.columns),
            y=list(heat.index),
            colorscale="YlOrRd",
            text=[[f"{v:.2f}" if pd.notna(v) else "-" for v in row] for row in heat.values],
            texttemplate="%{text}",
            hovertemplate="label=%{y}<br>metric=%{x}<br>value=%{z:.3f} m<extra></extra>",
        )
    )
    fig.update_layout(
        title="ADE/FDE matrix within <= 60m",
        xaxis_title="Metric",
        yaxis_title="Label",
        height=max(360, 70 * len(heat.index)),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig, width="stretch")

section_header(
    "Distance Ladder",
    "Compare mode defaults to clearer views than a 14-line overlay: delta heatmaps, label small multiples, and the original raw lines only as a fallback.",
)
distance_view = distance_both if distance_both is not None else distance_summary[distance_summary["label"].isin(available_labels)].copy()
if distance_both is not None and not distance_view.empty:
    compare_tabs = st.tabs(["Delta Heatmap", "Label Small Multiples", "Raw Lines"])
    with compare_tabs[0]:
        for start in range(0, len(METRIC_ORDER), 3):
            metric_chunk = METRIC_ORDER[start : start + 3]
            cols = st.columns(len(metric_chunk))
            for col, metric_name in zip(cols, metric_chunk):
                with col:
                    metric_delta = distance_delta[
                        (distance_delta["metric"] == metric_name)
                        & (distance_delta["label"].isin(available_labels))
                    ].copy()
                    if metric_delta.empty:
                        st.caption(f"{metric_name}: no data")
                        continue
                    col_order = ordered_distance_bins(metric_delta["r"].tolist())
                    pivot = (
                        metric_delta.pivot(index="label", columns="r", values="value_delta")
                        .reindex(index=available_labels)
                        .reindex(columns=col_order)
                    )
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=pivot.values,
                            x=[str(v) for v in pivot.columns],
                            y=[str(v) for v in pivot.index],
                            colorscale="RdBu",
                            zmid=0,
                            text=[[f"{v:+.2f}" if pd.notna(v) else "-" for v in row] for row in pivot.values],
                            texttemplate="%{text}",
                            hovertemplate="label=%{y}<br>r=%{x}<br>Δ=%{z:+.3f} m<extra></extra>",
                        )
                    )
                    fig.update_layout(
                        title=metric_name,
                        xaxis_title="Radius bin",
                        yaxis_title="Label",
                        height=max(320, 54 * len(available_labels)),
                        margin=dict(l=10, r=10, t=45, b=10),
                    )
                    st.plotly_chart(fig, width="stretch", key=f"distance_delta_{metric_name}")
    with compare_tabs[1]:
        metric_tabs = st.tabs(METRIC_ORDER)
        for metric_name, metric_tab in zip(METRIC_ORDER, metric_tabs):
            with metric_tab:
                metric_view = distance_view[
                    (distance_view["metric"] == metric_name)
                    & (distance_view["label"].isin(available_labels))
                ].copy()
                if metric_view.empty:
                    st.info(f"No data for {metric_name}.")
                    continue
                metric_view["r"] = pd.Categorical(metric_view["r"], categories=ordered_distance_bins(metric_view["r"].tolist()), ordered=True)
                for start in range(0, len(available_labels), 3):
                    chunk = available_labels[start : start + 3]
                    cols = st.columns(len(chunk))
                    for col, label_name in zip(cols, chunk):
                        with col:
                            label_df = metric_view[metric_view["label"] == label_name].copy()
                            if label_df.empty:
                                st.caption(f"{label_name}: no data")
                                continue
                            fig = px.line(
                                label_df,
                                x="r",
                                y="value",
                                color="run",
                                markers=True,
                                labels={"r": "Radius bin", "value": "Error (m)", "run": "Run"},
                                title=label_name,
                                color_discrete_map={"A": PLOTLY_COLORS["ink"], "B": PLOTLY_COLORS["amber"]},
                            )
                            fig.update_layout(height=280, margin=dict(l=10, r=10, t=45, b=10), legend_title="Run")
                            st.plotly_chart(fig, width="stretch", key=f"distance_small_{metric_name}_{label_name}")
    with compare_tabs[2]:
        fig = px.line(
            distance_view[distance_view["label"].isin(available_labels)],
            x="r",
            y="value",
            color="label",
            line_dash="run",
            markers=True,
            facet_col="metric",
            facet_col_wrap=3,
            category_orders={"r": ordered_distance_bins(distance_view["r"].tolist())},
            labels={"r": "Radius bin (m)", "value": "Error (m)", "label": "Label", "run": "Run"},
            title="ADE/FDE by distance bin: A vs B",
            color_discrete_sequence=[
                PLOTLY_COLORS["ink"],
                PLOTLY_COLORS["blue"],
                PLOTLY_COLORS["teal"],
                PLOTLY_COLORS["amber"],
                PLOTLY_COLORS["rose"],
                PLOTLY_COLORS["slate"],
                "#8b5cf6",
            ],
        )
        fig.update_layout(height=760, margin=dict(l=10, r=10, t=55, b=10), legend_title="Label / Run")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, width="stretch", key="distance_raw_compare")
elif not distance_view.empty:
    single_tabs = st.tabs(["Lines", "Metric Heatmaps", "Circular Rings", "Label Small Multiples"])
    with single_tabs[0]:
        fig = px.line(
            distance_view,
            x="r",
            y="value",
            color="label",
            markers=True,
            facet_col="metric",
            facet_col_wrap=3,
            category_orders={"r": ordered_distance_bins(distance_view["r"].tolist())},
            labels={"r": "Radius bin (m)", "value": "Error (m)", "label": "Label"},
            title="ADE/FDE by distance bin",
            color_discrete_sequence=[
                PLOTLY_COLORS["ink"],
                PLOTLY_COLORS["blue"],
                PLOTLY_COLORS["teal"],
                PLOTLY_COLORS["amber"],
                PLOTLY_COLORS["rose"],
                PLOTLY_COLORS["slate"],
                "#8b5cf6",
            ],
        )
        fig.update_layout(height=760, margin=dict(l=10, r=10, t=55, b=10), legend_title="Label")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, width="stretch", key="distance_single_lines")
    with single_tabs[1]:
        for start in range(0, len(METRIC_ORDER), 3):
            metric_chunk = METRIC_ORDER[start : start + 3]
            cols = st.columns(len(metric_chunk))
            for col, metric_name in zip(cols, metric_chunk):
                with col:
                    metric_df = distance_view[
                        (distance_view["metric"] == metric_name)
                        & (distance_view["label"].isin(available_labels))
                    ].copy()
                    if metric_df.empty:
                        st.caption(f"{metric_name}: no data")
                        continue
                    col_order = ordered_distance_bins(metric_df["r"].tolist())
                    pivot = (
                        metric_df.pivot(index="label", columns="r", values="value")
                        .reindex(index=available_labels)
                        .reindex(columns=col_order)
                    )
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=pivot.values,
                            x=[str(v) for v in pivot.columns],
                            y=[str(v) for v in pivot.index],
                            colorscale="YlOrRd",
                            text=[[f"{v:.2f}" if pd.notna(v) else "-" for v in row] for row in pivot.values],
                            texttemplate="%{text}",
                            hovertemplate="label=%{y}<br>r=%{x}<br>value=%{z:.3f} m<extra></extra>",
                        )
                    )
                    fig.update_layout(
                        title=metric_name,
                        xaxis_title="Radius bin",
                        yaxis_title="Label",
                        height=max(320, 54 * len(available_labels)),
                        margin=dict(l=10, r=10, t=45, b=10),
                    )
                    st.plotly_chart(fig, width="stretch", key=f"distance_single_heat_{metric_name}")
    with single_tabs[2]:
        for start in range(0, len(METRIC_ORDER), 2):
            metric_chunk = METRIC_ORDER[start : start + 2]
            cols = st.columns(len(metric_chunk))
            for col, metric_name in zip(cols, metric_chunk):
                with col:
                    metric_df = distance_view[
                        (distance_view["metric"] == metric_name)
                        & (distance_view["label"].isin(available_labels))
                    ].copy()
                    if metric_df.empty:
                        st.caption(f"{metric_name}: no data")
                        continue
                    fig = build_distance_ring_figure(metric_df, available_labels, metric_name)
                    st.plotly_chart(fig, width="stretch", key=f"distance_single_ring_{metric_name}")
    with single_tabs[3]:
        metric_tabs = st.tabs(METRIC_ORDER)
        for metric_name, metric_tab in zip(METRIC_ORDER, metric_tabs):
            with metric_tab:
                metric_df = distance_view[
                    (distance_view["metric"] == metric_name)
                    & (distance_view["label"].isin(available_labels))
                ].copy()
                if metric_df.empty:
                    st.info(f"No data for {metric_name}.")
                    continue
                metric_df["r"] = pd.Categorical(metric_df["r"], categories=ordered_distance_bins(metric_df["r"].tolist()), ordered=True)
                for start in range(0, len(available_labels), 3):
                    chunk = available_labels[start : start + 3]
                    cols = st.columns(len(chunk))
                    for col, label_name in zip(cols, chunk):
                        with col:
                            label_df = metric_df[metric_df["label"] == label_name].copy()
                            if label_df.empty:
                                st.caption(f"{label_name}: no data")
                                continue
                            fig = px.line(
                                label_df,
                                x="r",
                                y="value",
                                markers=True,
                                title=label_name,
                                labels={"r": "Radius bin", "value": "Error (m)"},
                                color_discrete_sequence=[PLOTLY_COLORS["blue"]],
                            )
                            fig.update_layout(height=280, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
                            st.plotly_chart(fig, width="stretch", key=f"distance_single_small_{metric_name}_{label_name}")

section_header(
    "Polar Field",
    "Each tab is one metric, and every label gets its own heatmap. That keeps the page filter-free while still easy to scan.",
)
polar_view_tabs = st.tabs(["Heatmap", "Circular"])
for view_name, outer_tab in zip(["heatmap", "circular"], polar_view_tabs):
    with outer_tab:
        metric_tabs = st.tabs(METRIC_ORDER)
        for metric_name, metric_tab in zip(METRIC_ORDER, metric_tabs):
            with metric_tab:
                metric_polar = polar_delta[polar_delta["metric"] == metric_name].copy() if polar_delta is not None else polar_summary[polar_summary["metric"] == metric_name].copy()
                value_col = "value_delta" if polar_delta is not None else "value"
                if metric_polar.empty:
                    st.info(f"No data for {metric_name}.")
                    continue
                for start in range(0, len(available_labels), 3):
                    chunk = available_labels[start : start + 3]
                    cols = st.columns(len(chunk))
                    for col, label_name in zip(cols, chunk):
                        with col:
                            label_polar = metric_polar[metric_polar["label"] == label_name].copy()
                            if label_polar.empty:
                                st.caption(f"{label_name}: no data")
                                continue
                            if view_name == "heatmap":
                                pivot = (
                                    label_polar.pivot(index="r", columns="theta", values=value_col)
                                    .reindex(index=R_LABELS, columns=THETA_LABELS)
                                )
                                fig = go.Figure(
                                    data=go.Heatmap(
                                        z=pivot.values,
                                        x=[str(v) for v in pivot.columns],
                                        y=[str(v) for v in pivot.index],
                                        colorscale="RdBu" if polar_delta is not None else "YlOrRd",
                                        zmid=0 if polar_delta is not None else None,
                                        hovertemplate=("theta=%{x}<br>r=%{y}<br>Δ=%{z:+.3f} m<extra></extra>" if polar_delta is not None else "theta=%{x}<br>r=%{y}<br>value=%{z:.3f} m<extra></extra>"),
                                    )
                                )
                                fig.update_layout(
                                    title=f"{label_name} (B - A)" if polar_delta is not None else label_name,
                                    xaxis_title="Theta",
                                    yaxis_title="Radius",
                                    height=320,
                                    margin=dict(l=10, r=10, t=45, b=10),
                                )
                                st.plotly_chart(fig, width="stretch", key=f"polar_{view_name}_{metric_name}_{label_name}")
                            else:
                                fig = build_theta_ring_figure(
                                    label_polar=label_polar,
                                    metric_name=metric_name,
                                    label_name=label_name,
                                    value_col=value_col,
                                    delta_mode=polar_delta is not None,
                                )
                                st.plotly_chart(fig, width="stretch", key=f"polar_{view_name}_{metric_name}_{label_name}")

section_header(
    "Metric Table",
    "Exact summary values for the labels in view, aligned with the specsheet future metric definitions.",
)
table_cols = ["label", "future_rows"] + METRIC_ORDER
st.dataframe(
    (
        compare_label[["label"] + [f"{m}_A" for m in METRIC_ORDER] + [f"{m}_B" for m in METRIC_ORDER] + [f"{m}_delta" for m in METRIC_ORDER]]
        if compare_label is not None
        else (label_view[table_cols] if not label_view.empty else label_summary[table_cols])
    ),
    width="stretch",
    hide_index=True,
    column_config={
        "future_rows": st.column_config.NumberColumn("Rows", format="%d"),
        "minADE@1s": st.column_config.NumberColumn("minADE@1s", format="%.3f m"),
        "minADE@3s": st.column_config.NumberColumn("minADE@3s", format="%.3f m"),
        "minADE@5s": st.column_config.NumberColumn("minADE@5s", format="%.3f m"),
        "minFDE@1s": st.column_config.NumberColumn("minFDE@1s", format="%.3f m"),
        "minFDE@3s": st.column_config.NumberColumn("minFDE@3s", format="%.3f m"),
        "minFDE@5s": st.column_config.NumberColumn("minFDE@5s", format="%.3f m"),
    },
)
