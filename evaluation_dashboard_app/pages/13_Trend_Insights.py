from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.page_chrome import inject_app_page_styles, render_page_hero, section_header
from lib.specsheet_report import (
    TrendReleaseGroup,
    discover_trend_release_groups,
    extract_devops_case_rows,
    extract_performance_metrics_from_summary,
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
    case_df = pd.DataFrame(case_rows)
    if not case_df.empty:
        case_df["date_sort"] = pd.to_datetime(case_df["date"], format="%Y.%m.%d", errors="coerce")
    metric_df = pd.DataFrame(metric_rows)
    if not metric_df.empty:
        metric_df["date_sort"] = pd.to_datetime(metric_df["date"], format="%Y.%m.%d", errors="coerce")
    return release_df, case_df, metric_df


render_page_hero(
    kicker="Release Analytics",
    title="Trend Insights",
    description="Inspect grouped release trend data the same way the catalog analyzer models it: one release group with sibling full, usecase, and devops job folders under the same topic.",
)

section_header(
    "Release Inventory",
    "Each row below is one grouped release entry. When full, usecase, and devops sibling folders exist under the same combined PDF group and topic, they are merged into one release view.",
)

groups = discover_trend_release_groups()
if not groups:
    st.info("No saved trend metadata was found yet. Generate a release spec-sheet with trend mode enabled first.")
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
    "roles",
    "full_job_id",
    "usecase_job_id",
    "devops_job_id",
    "release_name",
    "topic_name",
    "group_kind",
]
st.dataframe(
    release_df.sort_values(["date_sort", "version", "release_name"], ascending=[False, False, False])[inventory_cols],
    use_container_width=True,
    hide_index=True,
)

section_header(
    "Performance Trend",
    "Full-performance summaries are now plotted one release group at a time, even when they arrived with sibling usecase and devops folders.",
)

perf_entries = release_df[release_df["full_job_id"].notna()].sort_values(
    ["date_sort", "version", "release_name"],
    ascending=[True, True, True],
)
prediction_cols = [
    "minADE@1s",
    "minADE@3s",
    "minADE@5s",
    "minFDE@1s",
    "minFDE@3s",
    "minFDE@5s",
]

if not perf_entries.empty and perf_entries[prediction_cols].notna().any().any():
    pred_card_col1, pred_card_col2, pred_card_col3 = st.columns(3)
    latest_pred_row = perf_entries.dropna(subset=["minADE@3s", "minFDE@5s"], how="all").iloc[-1]
    pred_card_col1.metric(
        "Latest minADE@3s",
        f"{latest_pred_row['minADE@3s']:.2f} m" if pd.notna(latest_pred_row["minADE@3s"]) else "n/a",
    )
    pred_card_col2.metric(
        "Latest minFDE@5s",
        f"{latest_pred_row['minFDE@5s']:.2f} m" if pd.notna(latest_pred_row["minFDE@5s"]) else "n/a",
    )
    pred_card_col3.metric(
        "Latest Data Count",
        f"{int(latest_pred_row['data_count_num']):,}" if pd.notna(latest_pred_row["data_count_num"]) else "n/a",
    )

perf_col1, perf_col2 = st.columns([1.1, 1.0])
with perf_col1:
    if not perf_entries.empty and perf_entries["mAP"].notna().any():
        fig = go.Figure()
        fig.add_bar(
            x=perf_entries["version"],
            y=perf_entries["data_count_num"],
            name="Data Count",
            marker_color="#f4a7a7",
            opacity=0.5,
            yaxis="y2",
        )
        fig.add_trace(
            go.Scatter(
                x=perf_entries["version"],
                y=perf_entries["mAP"],
                name="mAP",
                mode="lines+markers",
                line=dict(color="#0f766e", width=3),
                customdata=perf_entries[["release_name", "date", "data_count"]].to_numpy(),
                hovertemplate="<b>%{x}</b><br>mAP: %{y:.3f}<br>Release: %{customdata[0]}<br>Date: %{customdata[1]}<br>Data Count: %{customdata[2]}<extra></extra>",
            )
        )
        fig.update_layout(
            title="mAP vs Data Count",
            xaxis_title="Pilot.Auto Version",
            yaxis_title="mAP",
            yaxis2=dict(title="Data Count", overlaying="y", side="right", showgrid=False),
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=0.94, x=0, xanchor="left"),
            margin=dict(l=20, r=20, t=90, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No grouped full-performance trend entries are available yet.")

with perf_col2:
    if not perf_entries.empty and perf_entries[prediction_cols].notna().any().any():
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
            title="Prediction Quality Story: All Horizons with Data Count",
            xaxis_title="Pilot.Auto Version",
            yaxis_title="Prediction Error (m)",
            yaxis2=dict(title="Data Count", overlaying="y", side="right", showgrid=False),
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=0.94, x=0, xanchor="left"),
            margin=dict(l=20, r=20, t=100, b=20),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        pred_fig.update_xaxes(showgrid=False)
        pred_fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.18)")
        st.plotly_chart(pred_fig, use_container_width=True)
        st.caption(
            "Each point is one grouped release. The chart now keeps sibling full/usecase/devops folders together so the performance story stays release-centric."
        )
    else:
        st.info("No usable grouped prediction trend values are available yet.")

if not metric_df.empty:
    atlas_df = metric_df[metric_df["block_header"] == "全数データセット評価"].copy()
    atlas_df = atlas_df.sort_values(["date_sort", "version", "release_name"], ascending=[True, True, True])
    latest_group_key = perf_entries.iloc[-1]["group_key"] if not perf_entries.empty else None
    previous_group_key = perf_entries.iloc[-2]["group_key"] if len(perf_entries) >= 2 else None
    latest_release_name = perf_entries.iloc[-1]["release_name"] if not perf_entries.empty else ""

    atlas_col1, atlas_col2 = st.columns([1.0, 1.0])
    if latest_group_key is not None:
        latest_matrix = atlas_df[atlas_df["group_key"] == latest_group_key].pivot_table(
            index="metric_name",
            columns="label_name",
            values="value",
            aggfunc="first",
        ).dropna(how="all")
        if not latest_matrix.empty:
            latest_min = latest_matrix.min(axis=1)
            latest_range = (latest_matrix.max(axis=1) - latest_min).replace(0, 1)
            latest_norm = latest_matrix.sub(latest_min, axis=0).div(latest_range, axis=0)
            atlas_fig = px.imshow(
                latest_norm,
                aspect="auto",
                color_continuous_scale=["#f8fafc", "#8dd3c7", "#0f766e"],
                text_auto=".2f",
            )
            atlas_fig.update_traces(
                text=latest_matrix.round(2).astype(str),
                hovertemplate="Metric: %{y}<br>Label: %{x}<br>Value: %{text}<extra></extra>",
            )
            atlas_fig.update_layout(
                title=f"Latest Release Metric Atlas: {latest_release_name}",
                margin=dict(l=20, r=20, t=70, b=20),
                coloraxis_colorbar=dict(title="Relative"),
            )
            atlas_col1.plotly_chart(atlas_fig, use_container_width=True)
        else:
            atlas_col1.info("No latest metric atlas is available yet.")

    if latest_group_key is not None and previous_group_key is not None:
        latest_matrix = atlas_df[atlas_df["group_key"] == latest_group_key].pivot_table(
            index="metric_name",
            columns="label_name",
            values="value",
            aggfunc="first",
        )
        previous_matrix = atlas_df[atlas_df["group_key"] == previous_group_key].pivot_table(
            index="metric_name",
            columns="label_name",
            values="value",
            aggfunc="first",
        )
        delta_matrix = latest_matrix.subtract(previous_matrix, fill_value=pd.NA).dropna(how="all")
        if not delta_matrix.empty:
            delta_fig = px.imshow(
                delta_matrix,
                aspect="auto",
                color_continuous_scale=["#7f1d1d", "#f8fafc", "#14532d"],
                color_continuous_midpoint=0,
                text_auto=".2f",
            )
            delta_fig.update_layout(
                title="Release-over-Release Metric Delta",
                margin=dict(l=20, r=20, t=70, b=20),
                coloraxis_colorbar=dict(title="Delta"),
            )
            atlas_col2.plotly_chart(delta_fig, use_container_width=True)
        else:
            atlas_col2.info("No previous release is available for metric delta yet.")
    else:
        atlas_col2.info("Metric delta becomes available after at least two grouped full releases exist.")

section_header(
    "Pass Rate Trend",
    "DevOps-style nested summaries are also grouped by release, so one pass-rate point represents the same release group as the matching performance metrics.",
)

pass_entries = release_df[release_df["devops_job_id"].notna()].sort_values(
    ["date_sort", "version", "release_name"],
    ascending=[True, True, True],
)
pass_col1, pass_col2 = st.columns([1.1, 1.0])
with pass_col1:
    if not pass_entries.empty and pass_entries["overall_pass_rate"].notna().any():
        pass_fig = go.Figure()
        pass_fig.add_bar(
            x=pass_entries["version"],
            y=pass_entries["scenario_count"],
            name="Scenario Count",
            marker_color="#86efac",
            opacity=0.55,
            yaxis="y2",
        )
        pass_fig.add_trace(
            go.Scatter(
                x=pass_entries["version"],
                y=pass_entries["overall_pass_rate"],
                name="Overall Pass Rate",
                mode="lines+markers",
                line=dict(color="#1d4ed8", width=3),
                customdata=pass_entries[["release_name", "date"]].to_numpy(),
                hovertemplate="<b>%{x}</b><br>Pass Rate: %{y:.1f}%<br>Release: %{customdata[0]}<br>Date: %{customdata[1]}<extra></extra>",
            )
        )
        pass_fig.update_layout(
            title="Overall Pass Rate vs Scenario Count",
            xaxis_title="Pilot.Auto Version",
            yaxis_title="Pass Rate (%)",
            yaxis2=dict(title="Scenario Count", overlaying="y", side="right", showgrid=False),
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=0.94, x=0, xanchor="left"),
            margin=dict(l=20, r=20, t=90, b=20),
        )
        st.plotly_chart(pass_fig, use_container_width=True)
    else:
        st.info("No grouped pass-rate summaries are available yet.")

with pass_col2:
    if not case_df.empty:
        major_summary = (
            case_df.groupby(["version", "date", "release_name", "major_category"], dropna=False)[["passed", "total"]]
            .sum()
            .reset_index()
        )
        major_summary["pass_rate"] = major_summary["passed"] / major_summary["total"] * 100.0
        major_summary["label"] = major_summary["major_category"].astype(str)
        pass_cat_fig = px.line(
            major_summary,
            x="version",
            y="pass_rate",
            color="label",
            markers=True,
            hover_data=["date", "release_name", "passed", "total"],
            title="Major Category Pass Rate",
        )
        pass_cat_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(pass_cat_fig, use_container_width=True)

if not case_df.empty:
    pass_detail_col1, pass_detail_col2 = st.columns([1.0, 1.0])
    with pass_detail_col1:
        mid_summary = (
            case_df.groupby(["mid_category", "version"], dropna=False)[["passed", "total"]]
            .sum()
            .reset_index()
        )
        mid_summary["pass_rate"] = mid_summary["passed"] / mid_summary["total"] * 100.0
        mid_matrix = mid_summary.pivot_table(
            index="mid_category",
            columns="version",
            values="pass_rate",
            aggfunc="first",
        )
        mid_fig = px.imshow(
            mid_matrix,
            aspect="auto",
            color_continuous_scale=["#7f1d1d", "#fef3c7", "#166534"],
            zmin=0,
            zmax=100,
            text_auto=".1f",
        )
        mid_fig.update_layout(
            title="Mid Category Pass Rate Matrix",
            margin=dict(l=20, r=20, t=70, b=20),
            coloraxis_colorbar=dict(title="%"),
        )
        st.plotly_chart(mid_fig, use_container_width=True)

    with pass_detail_col2:
        latest_devops_group = pass_entries.iloc[-1]["group_key"] if not pass_entries.empty else None
        latest_case_df = case_df[case_df["group_key"] == latest_devops_group].copy()
        latest_major_mid = (
            latest_case_df.groupby(["major_category", "mid_category"], dropna=False)[["passed", "total"]]
            .sum()
            .reset_index()
        )
        latest_major_mid["pass_rate"] = latest_major_mid["passed"] / latest_major_mid["total"] * 100.0
        if not latest_major_mid.empty:
            sunburst_fig = px.sunburst(
                latest_major_mid,
                path=["major_category", "mid_category"],
                values="total",
                color="pass_rate",
                color_continuous_scale=["#7f1d1d", "#fef3c7", "#166534"],
                range_color=(0, 100),
                title="Latest Release Pass-Rate Hierarchy",
            )
            sunburst_fig.update_layout(margin=dict(l=20, r=20, t=70, b=20))
            st.plotly_chart(sunburst_fig, use_container_width=True)
        else:
            st.info("No latest release pass-rate hierarchy is available yet.")

    st.markdown("**Case Explorer**")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
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
        selected_case = st.selectbox("Case", ["All"] + sorted(case_filtered["case_name"].dropna().unique().tolist()))
    if selected_case != "All":
        case_filtered = case_filtered[case_filtered["case_name"] == selected_case]

    st.dataframe(
        case_filtered.sort_values(["date_sort", "version", "case_name"]).drop(columns=["date_sort"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )

section_header(
    "Grouped Raw Browser",
    "Inspect one grouped release and its child roles directly. This makes it obvious which full, usecase, and devops job folders are contributing to the combined trend view.",
)

selection_df = release_df.sort_values(["date_sort", "version", "release_name"], ascending=[False, False, False]).reset_index(drop=True)
selection_labels = [
    f"{row.release_name} | {row.version} | {row.date} | roles: {row.roles}"
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
