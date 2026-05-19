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
    discover_trend_metadata_files,
    load_performance_trend_data,
    load_trend_metadata_file,
    load_trend_summary_file,
)

st.set_page_config(page_title="Trend Insights", layout="wide", initial_sidebar_state="expanded")
inject_app_page_styles()


def _run_name_from_metadata_path(metadata_path: Path) -> str:
    if metadata_path.parent.name == "resources":
        return metadata_path.parent.parent.name
    return metadata_path.parent.name


def _parse_data_count(value: Any) -> int | None:
    text = str(value or "").strip().replace(",", "").replace("+", "")
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _build_entry_frame(metadata_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_rows = load_performance_trend_data(metadata_files)
    by_version = {str(row.get("version")): row for row in performance_rows}

    entry_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for metadata_path in metadata_files:
        metadata = load_trend_metadata_file(metadata_path)
        summary_path = metadata_path.parent / "summary.json"
        summary = load_trend_summary_file(summary_path)
        version = str(metadata.get("pilot_auto_version") or "")
        trend_row = by_version.get(version, {})
        data_count_raw = str(metadata.get("data_count") or "")
        entry_row = {
            "run_name": _run_name_from_metadata_path(metadata_path),
            "version": version,
            "date": str(metadata.get("date") or ""),
            "description": str(metadata.get("description") or ""),
            "data_count": data_count_raw,
            "data_count_num": _parse_data_count(data_count_raw),
            "metadata_path": str(metadata_path),
            "summary_path": str(summary_path),
            "summary_kind": "performance_blocks" if isinstance(summary.get("blocks"), list) else "case_pass_rate",
            "summary_blocks": len(summary.get("blocks", [])) if isinstance(summary.get("blocks"), list) else 0,
            "mAP": trend_row.get("mAP"),
            "minADE@1s": trend_row.get("minADE@1s"),
            "minADE@3s": trend_row.get("minADE@3s"),
            "minADE@5s": trend_row.get("minADE@5s"),
            "minFDE@1s": trend_row.get("minFDE@1s"),
            "minFDE@3s": trend_row.get("minFDE@3s"),
            "minFDE@5s": trend_row.get("minFDE@5s"),
            "overall_pass_rate": None,
            "scenario_count": None,
        }

        if entry_row["summary_kind"] == "case_pass_rate":
            total_passed = 0
            total_count = 0
            for major_category, mid_categories in summary.items():
                if not isinstance(mid_categories, dict):
                    continue
                for mid_category, cases in mid_categories.items():
                    if not isinstance(cases, dict):
                        continue
                    for case_name, result in cases.items():
                        if not isinstance(result, dict):
                            continue
                        passed = int(result.get("passed", 0) or 0)
                        total = int(result.get("total", 0) or 0)
                        total_passed += passed
                        total_count += total
                        case_rows.append(
                            {
                                "run_name": entry_row["run_name"],
                                "version": version,
                                "date": entry_row["date"],
                                "description": entry_row["description"],
                                "major_category": major_category,
                                "mid_category": mid_category,
                                "case_name": case_name,
                                "passed": passed,
                                "total": total,
                                "pass_rate": (passed / total * 100.0) if total > 0 else None,
                            }
                        )
            entry_row["scenario_count"] = total_count
            entry_row["overall_pass_rate"] = (total_passed / total_count * 100.0) if total_count > 0 else None
        else:
            blocks = summary.get("blocks", [])
            for block in blocks:
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
                                    "run_name": entry_row["run_name"],
                                    "version": version,
                                    "date": entry_row["date"],
                                    "description": entry_row["description"],
                                    "block_header": block_header,
                                    "metric_name": metric_name,
                                    "label_name": label_name,
                                    "value": value,
                                }
                            )

        entry_rows.append(entry_row)

    entry_df = pd.DataFrame(entry_rows)
    if not entry_df.empty:
        entry_df["date_sort"] = pd.to_datetime(entry_df["date"], format="%Y.%m.%d", errors="coerce")
    case_df = pd.DataFrame(case_rows)
    if not case_df.empty:
        case_df["date_sort"] = pd.to_datetime(case_df["date"], format="%Y.%m.%d", errors="coerce")
    metric_df = pd.DataFrame(metric_rows)
    if not metric_df.empty:
        metric_df["date_sort"] = pd.to_datetime(metric_df["date"], format="%Y.%m.%d", errors="coerce")
        metric_df["value"] = pd.to_numeric(metric_df["value"], errors="coerce")
    return entry_df, case_df, metric_df


render_page_hero(
    kicker="Release Analytics",
    title="Trend Insights",
    description="Inspect every saved release-trend entry, including full performance metrics, case-level pass rates, and the raw analyzer summary payloads.",
)

section_header(
    "Trend Inventory",
    "Trend entries are discovered from saved `metadata.yaml` files that sit beside analyzer-compatible `summary.json` files under the dashboard data root.",
)

metadata_files = discover_trend_metadata_files()
if not metadata_files:
    st.info("No saved trend metadata was found yet. Generate a release spec-sheet with trend mode enabled first.")
    st.stop()

try:
    entry_df, case_df, metric_df = _build_entry_frame(metadata_files)
except Exception as exc:
    st.error(f"Could not build trend insights: {exc}")
    st.stop()

top1, top2, top3, top4, top5 = st.columns(5)
top1.metric("Trend Entries", f"{len(entry_df):,}")
top2.metric("Unique Versions", f"{entry_df['version'].nunique():,}" if not entry_df.empty else "0")
top3.metric(
    "Performance Entries",
    f"{int((entry_df['summary_kind'] == 'performance_blocks').sum()):,}" if not entry_df.empty else "0",
)
top4.metric(
    "Pass-rate Entries",
    f"{int((entry_df['summary_kind'] == 'case_pass_rate').sum()):,}" if not entry_df.empty else "0",
)
top5.metric(
    "Latest Date",
    entry_df.sort_values("date_sort")["date"].iloc[-1] if not entry_df.empty else "n/a",
)

inventory_df = entry_df.sort_values(["date_sort", "version", "run_name"], ascending=[False, False, False]).drop(
    columns=["date_sort"],
    errors="ignore",
)
st.dataframe(inventory_df, use_container_width=True, hide_index=True)

section_header(
    "Performance Trend",
    "Full-performance summaries expose mAP, precision, recall, FNR, localization error, and prediction metrics by label.",
)

perf_entries = entry_df[entry_df["summary_kind"] == "performance_blocks"].sort_values("date_sort")
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
        help="Mid-horizon trajectory accuracy. Lower is better.",
    )
    pred_card_col2.metric(
        "Latest minFDE@5s",
        f"{latest_pred_row['minFDE@5s']:.2f} m" if pd.notna(latest_pred_row["minFDE@5s"]) else "n/a",
        help="Longest-horizon endpoint accuracy. Lower is better.",
    )
    pred_card_col3.metric(
        "Latest Data Count",
        f"{int(latest_pred_row['data_count_num']):,}" if pd.notna(latest_pred_row["data_count_num"]) else "n/a",
        help="Release-scale sample count paired with the prediction metrics below.",
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
            )
        )
        fig.update_layout(
            title="mAP vs Data Count",
            xaxis_title="Pilot.Auto Version",
            yaxis_title="mAP",
            yaxis2=dict(title="Data Count", overlaying="y", side="right", showgrid=False),
            height=520,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.94,
                x=0,
                xanchor="left",
            ),
            margin=dict(l=20, r=20, t=90, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No full-performance trend entries are available yet.")

with perf_col2:
    if not perf_entries.empty and perf_entries[prediction_cols].notna().any().any():
        pred_story = perf_entries[["version", "date", "description", "run_name", "data_count", "data_count_num"] + prediction_cols].copy()
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
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        + metric_name
                        + ": %{y:.2f} m<br>"
                        + "Date: %{customdata[0]}<br>"
                        + "Run: %{customdata[1]}<br>"
                        + "Data Count: %{customdata[2]}<extra></extra>"
                    ),
                    customdata=pred_story[["date", "run_name", "data_count"]].to_numpy(),
                )
            )

        pred_fig.add_vrect(
            x0=-0.5,
            x1=len(pred_story) - 0.5,
            fillcolor="#f8fafc",
            opacity=0.35,
            line_width=0,
            layer="below",
        )
        pred_fig.update_layout(
            title="Prediction Quality Story: All Horizons with Data Count",
            xaxis_title="Pilot.Auto Version",
            yaxis_title="Prediction Error (m)",
            yaxis2=dict(title="Data Count", overlaying="y", side="right", showgrid=False),
            height=520,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.94,
                x=0,
                xanchor="left",
            ),
            margin=dict(l=20, r=20, t=100, b=20),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        pred_fig.update_xaxes(showgrid=False)
        pred_fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.18)")
        st.plotly_chart(pred_fig, use_container_width=True)

        st.caption(
            "Solid teal lines show ADE across 1s, 3s, and 5s. Dotted blue lines show FDE at the same horizons. The amber bars in the background show data count so scale changes are visible without leaving the chart."
        )
    else:
        st.info("No usable prediction trend values are available yet.")

if not metric_df.empty:
    perf_drill_col1, perf_drill_col2 = st.columns([1.0, 1.0])
    with perf_drill_col1:
        available_metrics = sorted(metric_df["metric_name"].dropna().unique().tolist())
        selected_metric = st.selectbox("Metric Drilldown", available_metrics, index=available_metrics.index("mAP") if "mAP" in available_metrics else 0)
    with perf_drill_col2:
        metric_labels = sorted(metric_df.loc[metric_df["metric_name"] == selected_metric, "label_name"].dropna().unique().tolist())
        selected_label = st.selectbox("Label", metric_labels, index=0)

    metric_slice = metric_df[
        (metric_df["metric_name"] == selected_metric)
        & (metric_df["label_name"] == selected_label)
        & (metric_df["block_header"] == "全数データセット評価")
    ].sort_values("date_sort")
    if not metric_slice.empty:
        drill_fig = px.line(
            metric_slice,
            x="version",
            y="value",
            markers=True,
            hover_data=["date", "description", "run_name"],
            title=f"{selected_metric} for {selected_label}",
        )
        drill_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(drill_fig, use_container_width=True)

        metric_pivot = metric_df[
            (metric_df["metric_name"] == selected_metric)
            & (metric_df["block_header"] == "全数データセット評価")
        ].pivot_table(
            index=["version", "date"],
            columns="label_name",
            values="value",
            aggfunc="first",
        ).reset_index()
        st.dataframe(metric_pivot, use_container_width=True, hide_index=True)

section_header(
    "Pass Rate Trend",
    "Nested release summaries expose per-case pass rates. We aggregate them into overall and category-level views here.",
)

pass_entries = entry_df[entry_df["summary_kind"] == "case_pass_rate"].sort_values("date_sort")
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
            )
        )
        pass_fig.update_layout(
            title="Overall Pass Rate vs Scenario Count",
            xaxis_title="Pilot.Auto Version",
            yaxis_title="Pass Rate (%)",
            yaxis2=dict(title="Scenario Count", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h"),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(pass_fig, use_container_width=True)
    else:
        st.info("No case-pass-rate summaries are available yet.")

with pass_col2:
    if not case_df.empty:
        category_level = st.selectbox("Category Level", ["major_category", "mid_category"], index=0)
        category_summary = (
            case_df.groupby(["version", "date", category_level], dropna=False)[["passed", "total"]]
            .sum()
            .reset_index()
        )
        category_summary["pass_rate"] = category_summary["passed"] / category_summary["total"] * 100.0
        category_summary["label"] = category_summary[category_level].astype(str)
        pass_cat_fig = px.line(
            category_summary,
            x="version",
            y="pass_rate",
            color="label",
            markers=True,
            hover_data=["date", "passed", "total"],
            title=f"Pass Rate by {category_level}",
        )
        pass_cat_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(pass_cat_fig, use_container_width=True)

if not case_df.empty:
    st.markdown("**Case Explorer**")
    case_filter_col1, case_filter_col2, case_filter_col3 = st.columns(3)
    with case_filter_col1:
        selected_major = st.selectbox(
            "Major Category",
            ["All"] + sorted(case_df["major_category"].dropna().unique().tolist()),
        )
    case_df_filtered = case_df.copy()
    if selected_major != "All":
        case_df_filtered = case_df_filtered[case_df_filtered["major_category"] == selected_major]

    with case_filter_col2:
        selected_mid = st.selectbox(
            "Mid Category",
            ["All"] + sorted(case_df_filtered["mid_category"].dropna().unique().tolist()),
        )
    if selected_mid != "All":
        case_df_filtered = case_df_filtered[case_df_filtered["mid_category"] == selected_mid]

    with case_filter_col3:
        selected_case = st.selectbox(
            "Case",
            ["All"] + sorted(case_df_filtered["case_name"].dropna().unique().tolist()),
        )
    if selected_case != "All":
        case_df_filtered = case_df_filtered[case_df_filtered["case_name"] == selected_case]

    case_rollup = case_df_filtered.sort_values(["date_sort", "version", "case_name"]).drop(
        columns=["date_sort"],
        errors="ignore",
    )
    st.dataframe(case_rollup, use_container_width=True, hide_index=True)

    if selected_case != "All":
        case_line = case_df_filtered.sort_values("date_sort")
        if not case_line.empty:
            case_fig = px.line(
                case_line,
                x="version",
                y="pass_rate",
                markers=True,
                hover_data=["date", "passed", "total", "run_name"],
                title=f"Case Trend: {selected_case}",
            )
            case_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(case_fig, use_container_width=True)

section_header(
    "Raw Summary Browser",
    "Inspect the raw metadata and summary payload for any entry. This is useful when validating what will flow into release spec-sheets and trend charts.",
)

selection_df = entry_df.sort_values(["date_sort", "version", "run_name"], ascending=[False, False, False]).reset_index(drop=True)
selection_labels = [
    f"{row.run_name} | {row.version} | {row.date} | {row.summary_kind}"
    for row in selection_df.itertuples()
]
selected_label = st.selectbox("Trend entry", selection_labels)
selected_row = selection_df.iloc[selection_labels.index(selected_label)]
selected_metadata = load_trend_metadata_file(selected_row["metadata_path"])
selected_summary = load_trend_summary_file(selected_row["summary_path"])

detail_col1, detail_col2 = st.columns([1.0, 1.25])
with detail_col1:
    st.markdown("**Metadata YAML**")
    st.code(json.dumps(selected_metadata, ensure_ascii=False, indent=2), language="json")

with detail_col2:
    st.markdown("**Summary JSON**")
    st.code(json.dumps(selected_summary, ensure_ascii=False, indent=2)[:30000], language="json")
