from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.page_chrome import inject_app_page_styles, render_page_hero, section_header
from lib.path_utils import get_data_root, path_display, resolve_under_data_root
from lib.specsheet_report import (
    DEFAULT_TREND_METADATA_TEXT,
    TREND_METADATA_FILENAME,
    TREND_SUMMARY_FILENAME,
    TrendReleaseGroup,
    classify_trend_summary,
    discover_trend_release_groups,
    extract_devops_case_rows,
    extract_performance_metrics_from_summary,
    load_trend_summary_file,
    parse_trend_metadata_text,
)

st.set_page_config(page_title="Trend Insights", layout="wide", initial_sidebar_state="expanded")
inject_app_page_styles()


def _parse_data_count(value: Any) -> int | None:
    text = str(value or "").strip().replace(",", "").replace("+", "")
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _select_primary_metadata(group: TrendReleaseGroup) -> dict[str, Any]:
    for role in ("full", "usecase", "devops", "performance_blocks", "unknown"):
        if role in group.jobs:
            return group.jobs[role]["metadata"]
    return {}


def _safe_path_part(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^\w.\-]+", "_", text).strip("._")
    return text or fallback


def _resolve_summary_json_input(user_path: str) -> tuple[Path | None, str]:
    resolved, err = resolve_under_data_root(user_path, allow_missing=False)
    if err:
        return None, err
    assert resolved is not None
    if resolved.is_file():
        if resolved.name != TREND_SUMMARY_FILENAME:
            return None, f"Expected a {TREND_SUMMARY_FILENAME} file: {path_display(resolved)}"
        return resolved, ""
    for candidate in (
        resolved / TREND_SUMMARY_FILENAME,
        resolved / "resources" / TREND_SUMMARY_FILENAME,
    ):
        if candidate.exists():
            return candidate, ""
    return None, f"No {TREND_SUMMARY_FILENAME} found in {path_display(resolved)} or its resources/ folder."


def _default_job_id_from_summary(summary_path: Path) -> str:
    if summary_path.parent.name == "resources":
        return summary_path.parent.parent.name
    return summary_path.parent.name


def _assemble_trend_release_group(
    *,
    release_name: str,
    topic_name: str,
    role_sources: dict[str, str],
    role_job_ids: dict[str, str],
    metadata: dict[str, Any],
) -> Path:
    data_root = get_data_root()
    release_dir = data_root / _safe_path_part(release_name, "trend_release")
    topic_dir = release_dir / _safe_path_part(topic_name, "perception.object_recognition.objects")
    expected_roles = {"full", "usecase", "devops"}
    seen_roles: dict[str, Path] = {}

    for expected_role, source_text in role_sources.items():
        summary_path, err = _resolve_summary_json_input(source_text)
        if err:
            raise ValueError(f"{expected_role}: {err}")
        assert summary_path is not None
        summary = load_trend_summary_file(summary_path)
        actual_role = classify_trend_summary(summary)
        if actual_role != expected_role:
            raise ValueError(
                f"{expected_role}: {path_display(summary_path)} classified as `{actual_role}`, "
                f"not `{expected_role}`."
            )
        seen_roles[actual_role] = summary_path

    missing = sorted(expected_roles - set(seen_roles))
    if missing:
        raise ValueError(f"Missing required trend roles: {', '.join(missing)}")

    for role, summary_path in seen_roles.items():
        job_id = _safe_path_part(role_job_ids.get(role) or _default_job_id_from_summary(summary_path), role)
        job_dir = topic_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, job_dir / TREND_SUMMARY_FILENAME)
        with (job_dir / TREND_METADATA_FILENAME).open("w", encoding="utf-8") as fh:
            import yaml

            yaml.safe_dump(metadata, fh, allow_unicode=True, sort_keys=False)
    return release_dir


def _render_release_trend_builder() -> None:
    section_header("Build Release Trend Group")
    with st.expander("Assemble full/usecase/devops summaries into one release", expanded=False):
        st.caption(
            "Use this after the three evaluator jobs have analyzer-compatible summary.json files. "
            "Each source can be a job folder, a run folder containing resources/summary.json, or the summary.json file itself."
        )
        with st.form("release_trend_builder_form"):
            form_col1, form_col2 = st.columns([1.1, 1.2])
            with form_col1:
                release_name = st.text_input(
                    "Release folder name",
                    value="trend_release_<full_job>_<usecase_job>_<devops_job>",
                )
                topic_name = st.text_input(
                    "Topic folder",
                    value="perception.object_recognition.objects",
                )
                full_source = st.text_input("Full summary source")
                usecase_source = st.text_input("Usecase summary source")
                devops_source = st.text_input("DevOps summary source")
            with form_col2:
                full_job_id = st.text_input("Full job id override", value="")
                usecase_job_id = st.text_input("Usecase job id override", value="")
                devops_job_id = st.text_input("DevOps job id override", value="")
                metadata_text = st.text_area(
                    "Release metadata YAML",
                    value=DEFAULT_TREND_METADATA_TEXT,
                    height=180,
                    help="Required keys: tags, pilot_auto_version, data_count, description, date.",
                )
            submitted = st.form_submit_button("Create Release Trend Group", type="primary")

        if submitted:
            try:
                metadata = parse_trend_metadata_text(metadata_text)
                created_dir = _assemble_trend_release_group(
                    release_name=release_name,
                    topic_name=topic_name,
                    role_sources={
                        "full": full_source,
                        "usecase": usecase_source,
                        "devops": devops_source,
                    },
                    role_job_ids={
                        "full": full_job_id,
                        "usecase": usecase_job_id,
                        "devops": devops_job_id,
                    },
                    metadata=metadata,
                )
                st.success(f"Created release trend group at `{path_display(created_dir)}`. Refreshing inventory...")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create release trend group: {exc}")


def _release_display_name(version: Any, date: Any, description: Any = "") -> str:
    version_text = str(version or "").strip() or "Unknown Version"
    date_text = str(date or "").strip()
    description_text = str(description or "").strip()
    suffix = f" | {date_text}" if date_text else ""
    if description_text:
        suffix += f" | {description_text}"
    return f"{version_text}{suffix}"


def _with_pass_rate(frame: pd.DataFrame, *, passed_col: str = "passed", total_col: str = "total") -> pd.DataFrame:
    enriched = frame.copy()
    total = pd.to_numeric(enriched[total_col], errors="coerce")
    passed = pd.to_numeric(enriched[passed_col], errors="coerce")
    enriched["pass_rate"] = (passed / total.replace(0, pd.NA)) * 100.0
    return enriched


def _update_version_axis(fig: go.Figure, versions: list[str]) -> None:
    fig.update_xaxes(categoryorder="array", categoryarray=versions)


def _build_pass_combo_chart(
    frame: pd.DataFrame,
    *,
    title: str,
    versions: list[str],
    line_y_col: str = "pass_rate",
    series_col: str | None = None,
    scenario_count_col: str = "total",
    hover_cols: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    show_legend = series_col is not None
    scenario_totals = (
        frame.groupby("version", dropna=False)[scenario_count_col]
        .sum()
        .reindex(versions)
        .fillna(0)
    )
    fig.add_bar(
        x=versions,
        y=scenario_totals.tolist(),
        name="Scenario Count",
        marker_color="#bfdbfe",
        opacity=0.32,
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Scenario Count: %{y:.0f}<extra></extra>",
    )

    hover_cols = hover_cols or ["date", "release_name", "passed", "total"]
    plot_df = frame.copy()
    if series_col is None:
        fig.add_trace(
            go.Scatter(
                x=plot_df["version"],
                y=plot_df[line_y_col],
                name=title,
                mode="lines+markers",
                line=dict(color="#1d4ed8", width=3),
                marker=dict(size=8, color="#1d4ed8"),
                customdata=plot_df[hover_cols].to_numpy() if hover_cols else None,
                hovertemplate="<b>%{x}</b><br>Pass Rate: %{y:.1f}%<br>Date: %{customdata[0]}<br>Release: %{customdata[1]}<extra></extra>",
            )
        )
    else:
        palette = px.colors.qualitative.Bold + px.colors.qualitative.Safe + px.colors.qualitative.Set2
        for idx, series_name in enumerate(plot_df[series_col].dropna().astype(str).unique().tolist()):
            series_df = plot_df[plot_df[series_col].astype(str) == series_name]
            color = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=series_df["version"],
                    y=series_df[line_y_col],
                    name=series_name,
                    mode="lines+markers",
                    line=dict(color=color, width=3),
                    marker=dict(size=7, color=color),
                    customdata=series_df[hover_cols].to_numpy() if hover_cols else None,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        + f"{series_col.replace('_', ' ').title()}: {series_name}<br>"
                        + "Pass Rate: %{y:.1f}%<br>"
                        + "Date: %{customdata[0]}<br>"
                        + "Release: %{customdata[1]}<br>"
                        + "Passed: %{customdata[2]:.0f}<br>"
                        + "Total: %{customdata[3]:.0f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Pilot.Auto Version",
        yaxis_title="Pass Rate (%)",
        yaxis2=dict(title="Scenario Count", overlaying="y", side="right", showgrid=False),
        height=440,
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, xanchor="left"),
        margin=dict(l=20, r=20, t=80, b=90),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=False, categoryorder="array", categoryarray=versions)
    fig.update_yaxes(range=[0, 100], gridcolor="rgba(148, 163, 184, 0.18)")
    return fig


def _build_defect_hierarchy_bars(
    frame: pd.DataFrame,
    *,
    category_cols: list[str],
    title: str,
    color_col: str = "major_category",
    label_cols: list[str] | None = None,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    bars = frame.copy()
    for category_col in category_cols:
        bars[category_col] = bars[category_col].fillna("Unspecified")
    label_cols = label_cols or category_cols
    bars["full_label"] = bars[label_cols].astype(str).agg(" / ".join, axis=1)
    bars["label"] = bars["full_label"]
    bars = bars.sort_values(category_cols + ["pass_rate", "total"], ascending=[True] * len(category_cols) + [False, False])
    fig = px.bar(
        bars,
        x="label",
        y="pass_rate",
        color=color_col,
        color_discrete_map=color_map,
        hover_data={"label": False, "full_label": True, "passed": True, "total": True},
        text=bars["pass_rate"].map(lambda value: f"{value:.1f}%" if pd.notna(value) else "n/a"),
        title=title,
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=70, b=140),
        xaxis_title=" / ".join(label.replace("_", " ").title() for label in label_cols),
        yaxis_title="Pass Rate (%)",
        legend_title_text=color_col.replace("_", " ").title(),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(tickangle=-35, automargin=True)
    fig.update_yaxes(range=[0, 100], automargin=True)
    return fig


def _build_defect_case_bars(
    frame: pd.DataFrame,
    *,
    ordered_mid_categories: list[str],
    max_cases: int = 20,
) -> go.Figure:
    case_bars = frame.copy()
    case_bars["minor_category"] = case_bars["minor_category"].fillna(case_bars["case_name"])
    case_bars["mid_order"] = case_bars["mid_category"].map(
        {mid_category: idx for idx, mid_category in enumerate(ordered_mid_categories)}
    )
    case_bars = case_bars.sort_values(["mid_order", "pass_rate", "total"], ascending=[True, True, False])
    case_bars = case_bars.head(max_cases)
    fig = px.bar(
        case_bars,
        x="minor_category",
        y="pass_rate",
        color="mid_category",
        hover_data=["major_category", "mid_category", "passed", "total"],
        text=case_bars["pass_rate"].map(lambda value: f"{value:.1f}%" if pd.notna(value) else "n/a"),
        title="Case Pass Rates",
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=70, b=140),
        xaxis_title="Case",
        yaxis_title="Pass Rate (%)",
        legend_title_text="Mid Category",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(tickangle=-35, automargin=True, categoryorder="array", categoryarray=case_bars["minor_category"].tolist())
    fig.update_yaxes(range=[0, 100], automargin=True)
    return fig


def _build_metric_timeline_heatmap(
    frame: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    color_title: str,
) -> go.Figure:
    matrix = frame.pivot_table(
        index="label_name",
        columns="release_axis",
        values=value_col,
        aggfunc="first",
    ).dropna(how="all")
    fig = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale=["#7f1d1d", "#f8fafc", "#14532d"] if "delta" in value_col else ["#f8fafc", "#8dd3c7", "#0f766e"],
        color_continuous_midpoint=0 if "delta" in value_col else None,
        text_auto=".3f",
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=70, b=20),
        coloraxis_colorbar=dict(title=color_title),
    )
    fig.update_xaxes(tickangle=-30, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def _build_metric_label_lines(
    frame: pd.DataFrame,
    *,
    title: str,
    ordered_axes: list[str],
) -> go.Figure:
    fig = px.line(
        frame,
        x="release_axis",
        y="value",
        color="label_name",
        markers=True,
        hover_data=["version", "date", "release_name"],
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=70, b=20), legend_title_text="Label")
    fig.update_xaxes(categoryorder="array", categoryarray=ordered_axes, tickangle=-30, automargin=True)
    return fig


def _horizon_metric_sort_key(metric_name: str) -> tuple[float, str]:
    horizon_text = str(metric_name).rsplit("@", 1)[-1].removesuffix("s")
    try:
        return float(horizon_text), str(metric_name)
    except ValueError:
        return float("inf"), str(metric_name)


def _horizon_metric_label(metric_name: str) -> str:
    return str(metric_name).rsplit("@", 1)[-1] if "@" in str(metric_name) else str(metric_name)


def _available_prediction_metric_groups(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    groups: dict[str, tuple[str, ...]] = {}
    metric_series = frame["metric_name"].dropna().astype(str)
    for metric_family in ("minADE", "minFDE"):
        metric_names = sorted(
            metric_series[metric_series.str.startswith(f"{metric_family}@")].unique().tolist(),
            key=_horizon_metric_sort_key,
        )
        if metric_names:
            groups[metric_family] = tuple(metric_names)
    return groups


def _build_prediction_label_profile(
    frame: pd.DataFrame,
    *,
    selected_label: str,
    metric_family: str,
    metric_names: tuple[str, ...],
    ordered_axes: list[str],
) -> go.Figure:
    profile_df = frame[
        (frame["metric_name"].isin(metric_names))
        & (frame["label_name"] == selected_label)
    ].copy()
    fig = px.line(
        profile_df,
        x="release_axis",
        y="value",
        color="metric_name",
        markers=True,
        hover_data=["version", "date", "release_name"],
        title=f"{selected_label} {metric_family} Horizon Profile",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=70, b=20), legend_title_text="Horizon")
    fig.update_xaxes(categoryorder="array", categoryarray=ordered_axes, tickangle=-30, automargin=True)
    return fig


def _build_prediction_release_label_profile(
    frame: pd.DataFrame,
    *,
    metric_family: str,
    selected_release_axis: str,
    selected_labels: list[str],
    metric_names: tuple[str, ...],
) -> go.Figure | None:
    release_df = frame[
        (frame["release_axis"] == selected_release_axis)
        & (frame["label_name"].isin(selected_labels))
        & (frame["metric_name"].isin(metric_names))
    ].copy()
    if release_df.empty:
        return None

    release_df["horizon"] = release_df["metric_name"].map(_horizon_metric_label)
    release_df["horizon_sort"] = release_df["metric_name"].map(lambda name: _horizon_metric_sort_key(str(name))[0])
    release_df = release_df.sort_values(["label_name", "horizon_sort"])
    fig = px.line(
        release_df,
        x="horizon",
        y="value",
        color="label_name",
        markers=True,
        category_orders={"horizon": [_horizon_metric_label(metric_name) for metric_name in metric_names]},
        hover_data=["version", "date", "release_name"],
        title=f"{metric_family} by Label and Horizon",
    )
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=70, b=30),
        legend_title_text="Label",
        xaxis_title="Prediction Horizon",
        yaxis_title=f"{metric_family} (m)",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.18)")
    return fig


def _build_release_frames(groups: list[TrendReleaseGroup]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    release_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for group in groups:
        primary_metadata = _select_primary_metadata(group)
        version = str(primary_metadata.get("pilot_auto_version") or "")
        date = str(primary_metadata.get("date") or "")
        description = str(primary_metadata.get("description") or "")
        data_count = str(primary_metadata.get("data_count") or "")
        release_row = {
            "group_key": group.group_key,
            "release_name": group.display_name,
            "topic_name": group.topic_name,
            "group_kind": group.group_kind,
            "version": version,
            "date": date,
            "description": description,
            "data_count": data_count,
            "data_count_num": _parse_data_count(data_count),
            "full_job_id": group.jobs.get("full", {}).get("job_id"),
            "usecase_job_id": group.jobs.get("usecase", {}).get("job_id"),
            "devops_job_id": group.jobs.get("devops", {}).get("job_id"),
            "mAP": None,
            "precision": None,
            "recall": None,
            "FNR": None,
            "x_error": None,
            "y_error": None,
            "yaw_error": None,
            "speed_error": None,
            "minADE@1s": None,
            "minADE@3s": None,
            "minADE@5s": None,
            "minFDE@1s": None,
            "minFDE@3s": None,
            "minFDE@5s": None,
            "overall_pass_rate": None,
            "scenario_count": None,
            "role_count": len(group.jobs),
            "roles": ", ".join(sorted(group.jobs.keys())),
        }

        if "full" in group.jobs:
            full_summary = group.jobs["full"]["summary"]
            release_row.update(extract_performance_metrics_from_summary(full_summary))
            for block in full_summary.get("blocks", []):
                block_header = str(block.get("header") or "")
                for table in block.get("tables", []):
                    table_data = table.get("data", {})
                    if not isinstance(table_data, dict):
                        continue
                    for metric_name, labels in table_data.items():
                        if not isinstance(labels, dict):
                            continue
                        for label_name, value in labels.items():
                            metric_rows.append(
                                {
                                    "group_key": group.group_key,
                                    "release_name": group.display_name,
                                    "version": version,
                                    "date": date,
                                    "description": description,
                                    "block_header": block_header,
                                    "metric_name": metric_name,
                                    "label_name": label_name,
                                    "value": pd.to_numeric(value, errors="coerce"),
                                }
                            )

        if "devops" in group.jobs:
            flattened = extract_devops_case_rows(group.jobs["devops"]["summary"])
            if flattened:
                total_passed = sum(int(row["passed"]) for row in flattened)
                total_count = sum(int(row["total"]) for row in flattened)
                release_row["scenario_count"] = total_count
                release_row["overall_pass_rate"] = (total_passed / total_count * 100.0) if total_count > 0 else None
                for row in flattened:
                    case_rows.append(
                        {
                            "group_key": group.group_key,
                            "release_name": group.display_name,
                            "version": version,
                            "date": date,
                            "description": description,
                            **row,
                        }
                    )

        release_rows.append(release_row)

    release_df = pd.DataFrame(release_rows)
    if not release_df.empty:
        release_df["date_sort"] = pd.to_datetime(release_df["date"], format="%Y.%m.%d", errors="coerce")
        release_df["release_display"] = release_df.apply(
            lambda row: _release_display_name(row["version"], row["date"], row["description"]),
            axis=1,
        )
    case_df = pd.DataFrame(case_rows)
    if not case_df.empty:
        case_df["date_sort"] = pd.to_datetime(case_df["date"], format="%Y.%m.%d", errors="coerce")
        case_df["release_display"] = case_df.apply(
            lambda row: _release_display_name(row["version"], row["date"], row["description"]),
            axis=1,
        )
    metric_df = pd.DataFrame(metric_rows)
    if not metric_df.empty:
        metric_df["date_sort"] = pd.to_datetime(metric_df["date"], format="%Y.%m.%d", errors="coerce")
        metric_df["release_display"] = metric_df.apply(
            lambda row: _release_display_name(row["version"], row["date"], row["description"]),
            axis=1,
        )
    return release_df, case_df, metric_df


render_page_hero(
    kicker="Release Analytics",
    title="Trend Insights",
    description="Release-level trends across grouped full, usecase, and devops runs.",
)

_render_release_trend_builder()

section_header("Release Inventory")

groups = discover_trend_release_groups()
if not groups:
    st.info("No saved trend metadata was found yet. Use the release trend builder above after the three job summaries are available.")
    st.stop()

try:
    release_df, case_df, metric_df = _build_release_frames(groups)
except Exception as exc:
    st.error(f"Could not build trend insights: {exc}")
    st.stop()

top1, top2, top3, top4, top5 = st.columns(5)
top1.metric("Release Groups", f"{len(release_df):,}")
top2.metric("Unique Versions", f"{release_df['version'].nunique():,}" if not release_df.empty else "0")
top3.metric("Groups with Full", f"{int(release_df['full_job_id'].notna().sum()):,}" if not release_df.empty else "0")
top4.metric("Groups with DevOps", f"{int(release_df['devops_job_id'].notna().sum()):,}" if not release_df.empty else "0")
top5.metric("Latest Date", release_df.sort_values("date_sort")["date"].iloc[-1] if not release_df.empty else "n/a")

inventory_cols = [
    "version",
    "date",
    "description",
    "data_count",
    "mAP",
    "precision",
    "recall",
    "overall_pass_rate",
    "roles",
    "full_job_id",
    "usecase_job_id",
    "devops_job_id",
    "topic_name",
    "group_kind",
]
st.dataframe(
    release_df.sort_values(["date_sort", "version", "release_name"], ascending=[False, False, False])[inventory_cols],
    use_container_width=True,
    hide_index=True,
)

section_header("Major Metrics Trend")

perf_entries = release_df[release_df["full_job_id"].notna()].sort_values(
    ["date_sort", "version", "release_name"],
    ascending=[True, True, True],
)
major_metric_cols = ["mAP", "precision", "recall"]
prediction_cols = [
    "minADE@1s",
    "minADE@3s",
    "minADE@5s",
    "minFDE@1s",
    "minFDE@3s",
    "minFDE@5s",
]
if not perf_entries.empty and perf_entries[major_metric_cols].notna().any().any():
    latest_major_row = perf_entries.dropna(subset=major_metric_cols, how="all").iloc[-1]
    metric_card_cols = st.columns(4)
    for metric_col, card_col in zip(major_metric_cols, metric_card_cols[:3]):
        metric_series = perf_entries.dropna(subset=[metric_col])
        latest_metric_value = metric_series[metric_col].iloc[-1] if not metric_series.empty else pd.NA
        card_col.metric(
            f"Latest {metric_col}",
            f"{latest_metric_value:.3f}" if pd.notna(latest_metric_value) else "n/a",
        )
    metric_card_cols[3].metric(
        "Latest Data Count",
        f"{int(latest_major_row['data_count_num']):,}" if pd.notna(latest_major_row["data_count_num"]) else "n/a",
    )
    fig = go.Figure()
    fig.add_bar(
        x=perf_entries["version"],
        y=perf_entries["data_count_num"],
        name="Data Count",
        marker_color="#f4a7a7",
        opacity=0.5,
        yaxis="y2",
    )
    metric_styles = {
        "mAP": {"color": "#0f766e", "dash": "solid"},
        "precision": {"color": "#1d4ed8", "dash": "solid"},
        "recall": {"color": "#be123c", "dash": "dot"},
    }
    for metric_col in major_metric_cols:
        fig.add_trace(
            go.Scatter(
                x=perf_entries["version"],
                y=perf_entries[metric_col],
                name=metric_col,
                mode="lines+markers",
                line=dict(
                    color=metric_styles[metric_col]["color"],
                    width=3,
                    dash=metric_styles[metric_col]["dash"],
                ),
                customdata=perf_entries[["release_name", "date", "data_count"]].to_numpy(),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + metric_col
                    + ": %{y:.3f}<br>Release: %{customdata[0]}<br>Date: %{customdata[1]}<br>Data Count: %{customdata[2]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Major Detection Metrics Trend",
        xaxis_title="Pilot.Auto Version",
        yaxis_title="Score",
        yaxis2=dict(title="Data Count", overlaying="y", side="right", showgrid=False),
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=0.94, x=0, xanchor="left"),
        margin=dict(l=20, r=20, t=90, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No grouped major metric trend entries are available yet.")

section_header("Prediction Trend")

if not perf_entries.empty and perf_entries[prediction_cols].notna().any().any():
    pred_card_col1, pred_card_col2, pred_card_col3 = st.columns(3)
    latest_pred_row = perf_entries.dropna(subset=prediction_cols, how="all").iloc[-1]
    latest_minade_mean = pd.to_numeric(latest_pred_row[["minADE@1s", "minADE@3s", "minADE@5s"]], errors="coerce").mean()
    latest_minfde_mean = pd.to_numeric(latest_pred_row[["minFDE@1s", "minFDE@3s", "minFDE@5s"]], errors="coerce").mean()
    pred_card_col1.metric(
        "Mean minADE",
        f"{latest_minade_mean:.2f} m" if pd.notna(latest_minade_mean) else "n/a",
    )
    pred_card_col2.metric(
        "Mean minFDE",
        f"{latest_minfde_mean:.2f} m" if pd.notna(latest_minfde_mean) else "n/a",
    )
    pred_card_col3.metric(
        "Latest Data Count",
        f"{int(latest_pred_row['data_count_num']):,}" if pd.notna(latest_pred_row["data_count_num"]) else "n/a",
    )
    pred_story = perf_entries[["version", "date", "description", "release_name", "data_count", "data_count_num"] + prediction_cols].copy()
    pred_fig = go.Figure()
    pred_fig.add_bar(
        x=pred_story["version"],
        y=pred_story["data_count_num"],
        name="Data Count",
        marker_color="#fbbf24",
        opacity=0.20,
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Data Count: %{y:,}<extra></extra>",
    )
    series_specs = [
        ("minADE@1s", "#0f766e", "solid"),
        ("minADE@3s", "#14b8a6", "solid"),
        ("minADE@5s", "#99f6e4", "solid"),
        ("minFDE@1s", "#1d4ed8", "dot"),
        ("minFDE@3s", "#60a5fa", "dot"),
        ("minFDE@5s", "#bfdbfe", "dot"),
    ]
    for metric_name, color, dash in series_specs:
        pred_fig.add_trace(
            go.Scatter(
                x=pred_story["version"],
                y=pred_story[metric_name],
                name=metric_name,
                mode="lines+markers",
                line=dict(color=color, width=3 if metric_name.endswith("@3s") else 2, dash=dash),
                marker=dict(size=8),
                customdata=pred_story[["date", "release_name", "data_count"]].to_numpy(),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + metric_name
                    + ": %{y:.2f} m<br>Date: %{customdata[0]}<br>Release: %{customdata[1]}<br>Data Count: %{customdata[2]}<extra></extra>"
                ),
            )
        )
    pred_fig.update_layout(
        title="Prediction Error Trend",
        xaxis_title="Pilot.Auto Version",
        yaxis_title="Prediction Error (m)",
        yaxis2=dict(title="Data Count", overlaying="y", side="right", showgrid=False),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=0.94, x=0, xanchor="left"),
        margin=dict(l=20, r=20, t=100, b=20),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    pred_fig.update_xaxes(showgrid=False)
    pred_fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.18)")
    st.plotly_chart(pred_fig, use_container_width=True)
else:
    st.info("No usable grouped prediction trend values are available yet.")

atlas_df = pd.DataFrame()
release_manifest = pd.DataFrame()
ordered_release_axes: list[str] = []

if not metric_df.empty:
    atlas_df = metric_df[metric_df["block_header"] == "全数データセット評価"].copy()
    atlas_df = atlas_df.sort_values(["date_sort", "version", "release_name"], ascending=[True, True, True])
    atlas_df["release_axis"] = atlas_df["version"].astype(str) + " | " + atlas_df["date"].astype(str)
    release_manifest = (
        atlas_df[["group_key", "release_axis", "version", "date", "release_name", "release_display"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    ordered_release_axes = release_manifest["release_axis"].tolist()

section_header("Pass Rate Trend")

pass_entries = release_df[release_df["devops_job_id"].notna()].sort_values(
    ["date_sort", "version", "release_name"],
    ascending=[True, True, True],
)
ordered_versions = pass_entries["version"].drop_duplicates().tolist()
overall_plot_df = pd.DataFrame()
major_summary = pd.DataFrame()
mid_summary = pd.DataFrame()

if not pass_entries.empty and pass_entries["overall_pass_rate"].notna().any():
    overall_plot_df = pass_entries[
        ["version", "date", "release_name", "overall_pass_rate", "scenario_count"]
    ].rename(columns={"overall_pass_rate": "pass_rate", "scenario_count": "total"}).copy()

if not case_df.empty:
    major_summary = (
        case_df.groupby(["version", "date", "release_name", "major_category"], dropna=False)[["passed", "total"]]
        .sum()
        .reset_index()
    )
    major_summary = _with_pass_rate(major_summary)

    mid_summary = (
        case_df.groupby(
            ["version", "date", "release_name", "major_category", "mid_category"],
            dropna=False,
        )[["passed", "total"]]
        .sum()
        .reset_index()
    )
    mid_summary = _with_pass_rate(mid_summary)

if not overall_plot_df.empty:
    st.plotly_chart(
        _build_pass_combo_chart(
            overall_plot_df,
            title="Overall Pass Rate",
            versions=ordered_versions,
            series_col=None,
            hover_cols=["date", "release_name"],
        ),
        use_container_width=True,
    )
else:
    st.info("No grouped pass-rate summaries are available yet.")

if not major_summary.empty:
    st.plotly_chart(
        _build_pass_combo_chart(
            major_summary,
            title="Major Category Pass Rate",
            versions=ordered_versions,
            series_col="major_category",
        ),
        use_container_width=True,
    )

if not mid_summary.empty:
    mid_summary_all = mid_summary.drop(columns=["major_category"], errors="ignore")
    st.plotly_chart(
        _build_pass_combo_chart(
            mid_summary_all,
            title="Mid Category Pass Rate",
            versions=ordered_versions,
            series_col="mid_category",
        ),
        use_container_width=True,
    )

section_header("Defect Evaluation")

if not case_df.empty and not pass_entries.empty:
    defect_release_options = pass_entries["release_display"].tolist()
    selected_defect_release = st.selectbox(
        "Version",
        defect_release_options,
        index=len(defect_release_options) - 1,
        key="defect_evaluation_release",
    )
    selected_defect_row = pass_entries.iloc[defect_release_options.index(selected_defect_release)]
    selected_defect_case_df = case_df[case_df["group_key"] == selected_defect_row["group_key"]].copy()
    defect_category_cols = ["major_category", "mid_category", "minor_category"]
    selected_major_mid = (
        selected_defect_case_df.groupby(defect_category_cols, dropna=False)[["passed", "total"]]
        .sum()
        .reset_index()
    )
    selected_major_mid = _with_pass_rate(selected_major_mid)
    if not selected_major_mid.empty:
        latest_view_mode = st.radio(
            "View",
            ["Bars", "Treemap", "Icicle", "Sunburst"],
            horizontal=True,
        )
        if latest_view_mode == "Bars":
            mid_level = (
                selected_defect_case_df.groupby(["major_category", "mid_category"], dropna=False)[["passed", "total"]]
                .sum()
                .reset_index()
            )
            mid_level = _with_pass_rate(mid_level)
            mid_level = mid_level.sort_values(
                ["major_category", "mid_category", "pass_rate", "total"],
                ascending=[True, True, False, False],
            )
            ordered_mid_categories = mid_level["mid_category"].tolist()
            st.plotly_chart(
                _build_defect_hierarchy_bars(
                    mid_level,
                    category_cols=["major_category", "mid_category"],
                    color_col="major_category",
                    title="Major / Mid",
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                _build_defect_case_bars(
                    selected_defect_case_df,
                    ordered_mid_categories=ordered_mid_categories,
                ),
                use_container_width=True,
            )
        elif latest_view_mode == "Treemap":
            latest_fig = px.treemap(
                selected_major_mid,
                path=defect_category_cols,
                values="total",
                color="pass_rate",
                color_continuous_scale=["#7f1d1d", "#fef3c7", "#166534"],
                range_color=(0, 100),
            )
            latest_fig.update_layout(margin=dict(l=20, r=20, t=70, b=20))
            st.plotly_chart(latest_fig, use_container_width=True)
        elif latest_view_mode == "Icicle":
            latest_fig = px.icicle(
                selected_major_mid,
                path=defect_category_cols,
                values="total",
                color="pass_rate",
                color_continuous_scale=["#7f1d1d", "#fef3c7", "#166534"],
                range_color=(0, 100),
            )
            latest_fig.update_layout(margin=dict(l=20, r=20, t=70, b=20))
            st.plotly_chart(latest_fig, use_container_width=True)
        else:
            latest_fig = px.sunburst(
                selected_major_mid,
                path=defect_category_cols,
                values="total",
                color="pass_rate",
                color_continuous_scale=["#7f1d1d", "#fef3c7", "#166534"],
                range_color=(0, 100),
            )
            latest_fig.update_layout(margin=dict(l=20, r=20, t=70, b=20))
            st.plotly_chart(latest_fig, use_container_width=True)

        case_pass_rate = selected_defect_case_df.copy()
        case_pass_rate["case"] = case_pass_rate["minor_category"].fillna(case_pass_rate["case_name"])
        case_pass_rate = case_pass_rate.sort_values(["pass_rate", "total"], ascending=[True, False])
        with st.expander("Case Pass Rates", expanded=False):
            st.dataframe(
                case_pass_rate[
                    ["major_category", "mid_category", "case", "pass_rate", "passed", "total"]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "pass_rate": st.column_config.NumberColumn("pass_rate", format="%.1f%%"),
                },
            )
    else:
        st.info("No defect evaluation hierarchy is available yet.")
else:
    st.info("No defect evaluation summaries are available yet.")

if not atlas_df.empty:
    release_options = release_manifest["release_axis"].tolist()
    section_header("Release Details")
    selected_detail_release = st.selectbox(
        "Version",
        release_options,
        index=len(release_options) - 1,
        key="deep_dive_release_detail",
    )
    horizon_metric_groups = _available_prediction_metric_groups(atlas_df)
    available_horizon_families = [metric_family for metric_family in ("minADE", "minFDE") if metric_family in horizon_metric_groups]
    horizon_labels = sorted(
        atlas_df[
            atlas_df["metric_name"].isin(
                [metric_name for metric_names in horizon_metric_groups.values() for metric_name in metric_names]
            )
        ]["label_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_atlas_group_key = release_manifest.loc[
        release_manifest["release_axis"] == selected_detail_release,
        "group_key",
    ].iloc[0]
    latest_matrix = atlas_df[atlas_df["group_key"] == selected_atlas_group_key].pivot_table(
        index="metric_name",
        columns="label_name",
        values="value",
        aggfunc="first",
    ).dropna(how="all")
    if not latest_matrix.empty:
        latest_min = latest_matrix.min(axis=1)
        latest_range = (latest_matrix.max(axis=1) - latest_min).replace(0, 1)
        latest_norm = latest_matrix.sub(latest_min, axis=0).div(latest_range, axis=0)
        latest_atlas_fig = px.imshow(
            latest_norm,
            aspect="auto",
            color_continuous_scale=["#f8fafc", "#8dd3c7", "#0f766e"],
            text_auto=".2f",
        )
        latest_atlas_fig.update_traces(
            text=latest_matrix.round(2).astype(str),
            hovertemplate="Metric: %{y}<br>Label: %{x}<br>Value: %{text}<extra></extra>",
        )
        latest_atlas_fig.update_layout(
            title="Metric Atlas",
            margin=dict(l=20, r=20, t=70, b=20),
            coloraxis_colorbar=dict(title="Relative"),
        )
        latest_atlas_fig.update_xaxes(automargin=True)
        latest_atlas_fig.update_yaxes(automargin=True)
        st.plotly_chart(latest_atlas_fig, use_container_width=True)
    else:
        st.info("No metric atlas is available for the selected release yet.")

    if available_horizon_families and horizon_labels:
        release_detail_cols = st.columns(len(available_horizon_families))
        for col, metric_family in zip(release_detail_cols, available_horizon_families):
            metric_names = horizon_metric_groups[metric_family]
            family_df = atlas_df[atlas_df["metric_name"].isin(metric_names)].copy()
            release_fig = _build_prediction_release_label_profile(
                family_df,
                metric_family=metric_family,
                selected_release_axis=selected_detail_release,
                selected_labels=horizon_labels,
                metric_names=metric_names,
            )
            with col:
                if release_fig is not None:
                    st.plotly_chart(release_fig, use_container_width=True)
                else:
                    st.info(f"No {metric_family} horizon values are available for the selected release.")

    section_header("Trend Details")
    if available_horizon_families and horizon_labels:
        selected_horizon_label = st.selectbox(
            "Label Trend Focus",
            horizon_labels,
            key="prediction_horizon_label_focus",
        )
        trend_profile_cols = st.columns(len(available_horizon_families))
        for col, metric_family in zip(trend_profile_cols, available_horizon_families):
            metric_names = horizon_metric_groups[metric_family]
            family_df = atlas_df[atlas_df["metric_name"].isin(metric_names)].copy()
            profile_fig = _build_prediction_label_profile(
                family_df,
                selected_label=selected_horizon_label,
                metric_family=metric_family,
                metric_names=metric_names,
                ordered_axes=ordered_release_axes,
            )
            with col:
                st.plotly_chart(profile_fig, use_container_width=True)
    else:
        st.info("No minADE/minFDE horizon trend data is available yet.")

    trend_mode = st.radio(
        "Trend View",
        ["Timeline Heatmap", "Label Trend Lines"],
        horizontal=True,
        key="detailed_metric_trend_view",
    )

    metric_options = sorted(atlas_df["metric_name"].dropna().unique().tolist())
    selected_metric = st.selectbox("Metric", metric_options, key="detailed_metric_trend_metric")

    metric_trend_df = atlas_df[atlas_df["metric_name"] == selected_metric].copy()
    if not metric_trend_df.empty:
        if trend_mode == "Timeline Heatmap":
            explorer_fig = _build_metric_timeline_heatmap(
                metric_trend_df,
                value_col="value",
                title=f"{selected_metric} Timeline Heatmap by Label",
                color_title=selected_metric,
            )
        else:
            explorer_fig = _build_metric_label_lines(
                metric_trend_df,
                title=f"{selected_metric} Label Trend Lines",
                ordered_axes=ordered_release_axes,
            )
        st.plotly_chart(explorer_fig, use_container_width=True)
    else:
        st.info("No detailed trend data is available for the selected metric yet.")
elif not metric_df.empty:
    st.info("No full-dataset metric atlas data is available yet.")

if not case_df.empty:
    with st.expander("Case Explorer", expanded=False):
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        with filter_col1:
            selected_major = st.selectbox("Major Category", ["All"] + sorted(case_df["major_category"].dropna().unique().tolist()))
        case_filtered = case_df.copy()
        if selected_major != "All":
            case_filtered = case_filtered[case_filtered["major_category"] == selected_major]
        with filter_col2:
            selected_mid = st.selectbox("Mid Category", ["All"] + sorted(case_filtered["mid_category"].dropna().unique().tolist()))
        if selected_mid != "All":
            case_filtered = case_filtered[case_filtered["mid_category"] == selected_mid]
        with filter_col3:
            selected_minor = st.selectbox("Minor Category", ["All"] + sorted(case_filtered["minor_category"].dropna().unique().tolist()))
        if selected_minor != "All":
            case_filtered = case_filtered[case_filtered["minor_category"] == selected_minor]
        with filter_col4:
            selected_case = st.selectbox("Case", ["All"] + sorted(case_filtered["case_name"].dropna().unique().tolist()))
        if selected_case != "All":
            case_filtered = case_filtered[case_filtered["case_name"] == selected_case]

        st.dataframe(
            case_filtered.sort_values(["date_sort", "version", "case_name"]).drop(columns=["date_sort"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )

with st.expander("Grouped Raw Browser", expanded=False):
    selection_df = release_df.sort_values(
        ["date_sort", "version", "release_name"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selection_labels = [
        f"{row.release_display} | roles: {row.roles}"
        for row in selection_df.itertuples()
    ]
    selected_label = st.selectbox("Release Group", selection_labels)
    selected_release = selection_df.iloc[selection_labels.index(selected_label)]
    selected_group = next(group for group in groups if group.group_key == selected_release["group_key"])

    group_manifest = {
        "display_name": selected_group.display_name,
        "topic_name": selected_group.topic_name,
        "group_kind": selected_group.group_kind,
        "base_dir": str(selected_group.base_dir),
        "jobs": {
            role: {
                "job_id": payload["job_id"],
                "metadata_path": str(payload["metadata_path"]),
                "summary_path": str(payload["summary_path"]),
            }
            for role, payload in selected_group.jobs.items()
        },
    }

    detail_col1, detail_col2 = st.columns([0.9, 1.1])
    with detail_col1:
        st.markdown("**Release Group Manifest**")
        st.code(json.dumps(group_manifest, ensure_ascii=False, indent=2), language="json")
        role_choice = st.selectbox("Child Role", sorted(selected_group.jobs.keys()))

    with detail_col2:
        st.markdown("**Selected Child Summary JSON**")
        st.code(
            json.dumps(selected_group.jobs[role_choice]["summary"], ensure_ascii=False, indent=2)[:30000],
            language="json",
        )
