from __future__ import annotations

import json
import re
import shutil
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from lib.page_chrome import inject_app_page_styles, render_page_hero, section_header
from lib.path_utils import get_data_root, path_display, resolve_under_data_root
from lib.release_specsheet_library import discover_release_specsheet_inventory
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


def _role_overview_url(release_row: dict[str, Any], role: str) -> str:
    role_info = release_row.get("roles", {}).get(role, {})
    return str(role_info.get("overview_url") or "")


def _role_debug_path(release_row: dict[str, Any], role: str) -> str:
    role_info = release_row.get("roles", {}).get(role, {})
    return str(role_info.get("absolute_path") or "")


def _role_evaluator_url(release_row: dict[str, Any], role: str) -> str:
    role_info = release_row.get("roles", {}).get(role, {})
    return str(role_info.get("evaluator_report_url") or "")


def _topic_family(topic_name: Any) -> str:
    topic = str(topic_name or "")
    if topic == "perception.object_recognition.objects":
        return "Perception Performance"
    if topic.startswith("perception.object_recognition.detection."):
        return "ML Model Performance"
    return "Other"


def _date_sort_value(value: Any) -> float:
    parsed = pd.to_datetime(value, format="%Y.%m.%d", errors="coerce")
    if pd.isna(parsed):
        return -1.0
    return float(parsed.timestamp())


def _html_link(url: str, label: str, variant: str = "action") -> str:
    if not url:
        return '<span class="muted-cell">-</span>'
    return (
        f'<a class="link-chip link-chip-{escape(variant, quote=True)}" '
        f'href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(label)}</a>'
    )


def _pdf_links_for_prefix(release: dict[str, Any], prefix: str) -> str:
    links = []
    for pdf in release.get("pdfs", []):
        topic = str(pdf.get("topic") or "")
        if topic == prefix or topic.startswith(prefix):
            label = "Prediction"
            if topic.startswith("perception.object_recognition.detection."):
                label = topic.replace("perception.object_recognition.detection.", "").replace(".objects", "")
                label = label.replace("bevfusion", "BEVFusion").replace("centerpoint", "CenterPoint")
            links.append(_html_link(str(pdf.get("static_url") or ""), label, "pdf"))
    return '<span class="link-chip-row">' + "".join(links) + "</span>" if links else '<span class="muted-cell">-</span>'


def _has_pdf_for_prefix(release: dict[str, Any], prefix: str) -> bool:
    for pdf in release.get("pdfs", []):
        topic = str(pdf.get("topic") or "")
        if topic == prefix or topic.startswith(prefix):
            return True
    return False


def _render_release_library_table(releases: list[dict[str, Any]]) -> None:
    group_headers = [
        ("Release", 4),
        ("Overview", 3),
        ("Specsheet PDF", 2),
        ("Evaluator Job", 3),
    ]
    col_widths = [360, 96, 240, 92, 96, 96, 96, 128, 168, 96, 96, 96]
    headers = [
        "Version",
        "Date",
        "Description",
        "Data",
        "Performance",
        "Usecase",
        "DevOps",
        "Prediction",
        "Detection",
        "Performance",
        "Usecase",
        "DevOps",
    ]
    sort_types = ["text", "date", "text", "number", "text", "text", "text", "text", "text", "text", "text", "text"]
    sortable_columns = {0, 1, 2, 3}
    rows_html = []
    for release in releases:
        sort_values = [
            str(release.get("version") or ""),
            str(_date_sort_value(release.get("date"))),
            str(release.get("description") or ""),
            str(_parse_data_count(release.get("data_count")) or -1),
            "open" if _role_overview_url(release, "performance") else "",
            "open" if _role_overview_url(release, "usecase") else "",
            "open" if _role_overview_url(release, "devops") else "",
            "prediction" if _has_pdf_for_prefix(release, "perception.object_recognition.objects") else "",
            "detection" if _has_pdf_for_prefix(release, "perception.object_recognition.detection.") else "",
            "report" if _role_evaluator_url(release, "performance") else "",
            "report" if _role_evaluator_url(release, "usecase") else "",
            "report" if _role_evaluator_url(release, "devops") else "",
        ]
        cells = [
            escape(str(release.get("version") or "")),
            escape(str(release.get("date") or "")),
            escape(str(release.get("description") or "")),
            escape(str(release.get("data_count") or "")),
            _html_link(_role_overview_url(release, "performance"), "Open", "overview"),
            _html_link(_role_overview_url(release, "usecase"), "Open", "overview"),
            _html_link(_role_overview_url(release, "devops"), "Open", "overview"),
            _pdf_links_for_prefix(release, "perception.object_recognition.objects"),
            _pdf_links_for_prefix(release, "perception.object_recognition.detection."),
            _html_link(_role_evaluator_url(release, "performance"), "Report", "job"),
            _html_link(_role_evaluator_url(release, "usecase"), "Report", "job"),
            _html_link(_role_evaluator_url(release, "devops"), "Report", "job"),
        ]
        rows_html.append(
            "<tr>"
            + "".join(
                f'<td data-sort-value="{escape(sort_value, quote=True)}">{cell}</td>'
                for cell, sort_value in zip(cells, sort_values)
            )
            + "</tr>"
        )
    table_html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{
  box-sizing: border-box;
}}
body {{
  margin: 0;
  padding: 0 0 26px 0;
  background: transparent;
  color: #0f172a;
  font-family: "Source Sans Pro", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.release-library-shell {{
  background: transparent;
}}
.release-library-table-wrapper {{
  overflow-x: auto;
  overflow-y: visible;
  width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 10px;
  padding-bottom: 2px;
  scrollbar-gutter: stable;
}}
.release-library-table {{
  border-collapse: separate;
  border-spacing: 0;
  table-layout: fixed;
  min-width: 1660px;
  width: 100%;
  font-size: 0.88rem;
}}
.release-library-table th,
.release-library-table td {{
  border-bottom: 1px solid rgba(148, 163, 184, 0.28);
  padding: 0.34rem 0.5rem;
  text-align: left;
  vertical-align: middle;
  line-height: 1.22;
  white-space: nowrap;
}}
.release-library-table th {{
  background: #f8fafc;
  color: #334155;
  font-weight: 700;
  white-space: nowrap;
}}
.release-library-table .group-header th {{
  position: sticky;
  top: 0;
  z-index: 3;
  background: #eef2ff;
  color: #3730a3;
  text-align: center;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-right: 1px solid rgba(129, 140, 248, 0.22);
}}
.release-library-table .column-header th {{
  position: sticky;
  top: 29px;
  z-index: 3;
  background: #f8fafc;
  font-size: 0.82rem;
  text-align: center;
  padding: 0;
}}
.sort-button {{
  appearance: none;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  min-height: 30px;
  padding: 0.26rem 0.38rem;
  border: 0;
  background: transparent;
  color: #334155;
  font: inherit;
  font-weight: 750;
  cursor: pointer;
}}
.sort-button:hover {{
  background: rgba(248, 250, 252, 0.92);
  color: #334155;
}}
.plain-header {{
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 30px;
  padding: 0.26rem 0.38rem;
  font-weight: 750;
}}
.release-library-table tbody tr:hover td {{
  background: rgba(248, 250, 252, 0.82);
}}
.release-library-table td:nth-child(1) {{
  font-weight: 650;
  color: #0f172a;
}}
.release-library-table td:nth-child(3) {{
  color: #475569;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.release-library-table td:nth-child(2),
.release-library-table td:nth-child(4) {{
  color: #475569;
}}
.release-library-table td:nth-child(n+5) {{
  text-align: center;
}}
.release-library-table td:nth-child(5),
.release-library-table td:nth-child(6),
.release-library-table td:nth-child(7),
.release-library-table td:nth-child(10),
.release-library-table td:nth-child(11),
.release-library-table td:nth-child(12) {{
}}
.release-library-table td:nth-child(8),
.release-library-table td:nth-child(9) {{
}}
.link-chip {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 64px;
  min-height: 22px;
  padding: 0.08rem 0.46rem;
  margin: 0.03rem 0;
  border-radius: 999px;
  font-weight: 650;
  font-size: 0.8rem;
  text-decoration: none;
  border: 1px solid transparent;
}}
.link-chip-row {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.22rem;
  flex-wrap: nowrap;
}}
.link-chip-overview {{
  color: #1d4ed8;
  background: #eff6ff;
  border-color: #bfdbfe;
}}
.link-chip-pdf {{
  color: #9f1239;
  background: #fff1f2;
  border-color: #fecdd3;
}}
.link-chip-job {{
  color: #166534;
  background: #f0fdf4;
  border-color: #bbf7d0;
}}
.link-chip:hover {{
  text-decoration: underline;
  filter: brightness(0.98);
}}
.muted-cell {{
  color: #94a3b8;
}}
</style>
</head>
<body>
<div class="release-library-shell">
  <div class="release-library-table-wrapper">
    <table id="releaseLibraryTable" class="release-library-table">
      <colgroup>{''.join(f'<col style="width:{width}px">' for width in col_widths)}</colgroup>
      <thead>
        <tr class="group-header">{''.join(f'<th colspan="{span}">{escape(header)}</th>' for header, span in group_headers)}</tr>
        <tr class="column-header">{''.join(f'<th><button class="sort-button" type="button" data-index="{idx}" data-type="{sort_types[idx]}">{escape(header)}</button></th>' if idx in sortable_columns else f'<th><span class="plain-header">{escape(header)}</span></th>' for idx, header in enumerate(headers))}</tr>
      </thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
  </div>
</div>
<script>
(function () {{
  const table = document.getElementById("releaseLibraryTable");
  const tbody = table.querySelector("tbody");
  const buttons = Array.from(table.querySelectorAll(".sort-button"));
  let activeSort = {{ index: 1, dir: "desc", type: "date" }};

  function allRows() {{
    return Array.from(tbody.querySelectorAll("tr"));
  }}

  function cellValue(row, index, type) {{
    const cell = row.children[index];
    const raw = (cell && (cell.dataset.sortValue || cell.innerText) || "").trim();
    if (type === "number" || type === "date") {{
      const value = Number(raw.replace(/,/g, ""));
      return Number.isFinite(value) ? value : -Infinity;
    }}
    return raw.toLowerCase();
  }}

  function compareRows(a, b, sort) {{
    const av = cellValue(a, sort.index, sort.type);
    const bv = cellValue(b, sort.index, sort.type);
    if (av < bv) return sort.dir === "asc" ? -1 : 1;
    if (av > bv) return sort.dir === "asc" ? 1 : -1;
    return 0;
  }}

  function applySort() {{
    const rows = allRows();
    rows.sort((a, b) => compareRows(a, b, activeSort));
    rows.forEach((row) => tbody.appendChild(row));
    buttons.forEach((button) => {{
      const isActive = Number(button.dataset.index) === activeSort.index;
      button.dataset.dir = isActive ? activeSort.dir : "";
    }});
  }}

  buttons.forEach((button) => {{
    button.addEventListener("click", () => {{
      const nextIndex = Number(button.dataset.index);
      const nextType = button.dataset.type || "text";
      const nextDir = activeSort.index === nextIndex && activeSort.dir === "asc" ? "desc" : "asc";
      activeSort = {{ index: nextIndex, dir: nextDir, type: nextType }};
      applySort();
    }});
  }});

  applySort();
}})();
</script>
</body>
</html>
"""
    component_height = 124 + max(1, len(releases)) * 32
    components.html(table_html, height=component_height, scrolling=False)


def _release_inventory_debug_rows(releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for release in releases:
        rows.append(
            {
                "version": release["version"],
                "date": release["date"],
                "release": release["release"],
                "release_dir": release["release_dir_absolute"],
                "performance_dir": _role_debug_path(release, "performance"),
                "usecase_dir": _role_debug_path(release, "usecase"),
                "devops_dir": _role_debug_path(release, "devops"),
                "performance_job_url": _role_evaluator_url(release, "performance"),
                "usecase_job_url": _role_evaluator_url(release, "usecase"),
                "devops_job_url": _role_evaluator_url(release, "devops"),
                "pdf_paths": "\n".join(pdf["absolute_path"] for pdf in release.get("pdfs", [])),
            }
        )
    return rows


def _release_metric_bar_ranges(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    metric_columns = ("mAP", "precision", "recall", "overall_pass_rate", "FNR", "x_error", "y_error", "yaw_error")
    for column in metric_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.notna().any():
            continue
        min_value = float(values.min(skipna=True))
        max_value = float(values.max(skipna=True))
        if abs(max_value - min_value) < 1e-12:
            if column == "overall_pass_rate":
                min_value, max_value = 0.0, 100.0
            elif column in {"mAP", "precision", "recall"}:
                min_value, max_value = 0.0, 1.0
            else:
                min_value, max_value = 0.0, max(max_value, 1.0)
        ranges[column] = (min_value, max_value)
    return ranges


def _release_performance_cell_html(value: Any, column: str, ranges: dict[str, tuple[float, float]]) -> str:
    metric_columns = {"mAP", "precision", "recall", "overall_pass_rate", "FNR", "x_error", "y_error", "yaw_error"}
    if column not in metric_columns:
        return escape(str(value or ""))

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return '<span class="perf-muted">-</span>'

    min_value, max_value = ranges.get(column, (0.0, 1.0))
    span = max(max_value - min_value, 1e-12)
    normalized = max(0.0, min(1.0, (float(numeric) - min_value) / span))
    pct = 8.0 + normalized * 92.0
    if column == "overall_pass_rate":
        label = f"{float(numeric):.1f}%"
    else:
        label = f"{float(numeric):.3f}"

    # Calm app-aligned palette: soft rose for weak/concerning values, soft teal for strong/healthy values.
    teal = (45, 212, 191)
    rose = (251, 113, 133)
    if column in {"mAP", "precision", "recall", "overall_pass_rate"}:
        color_ratio = normalized
    else:
        color_ratio = 1.0 - normalized
    red = round(rose[0] + (teal[0] - rose[0]) * color_ratio)
    green = round(rose[1] + (teal[1] - rose[1]) * color_ratio)
    blue = round(rose[2] + (teal[2] - rose[2]) * color_ratio)

    return (
        f'<div class="perf-bar-cell" '
        f'style="--bar-width:{pct:.1f}%; --bar-r:{red}; --bar-g:{green}; --bar-b:{blue};">'
        f'<span>{escape(label)}</span>'
        "</div>"
    )


def _release_performance_column_group(column: str) -> str:
    if column in {"version", "date", "description", "data_count"}:
        return "Release"
    if column in {"mAP", "precision", "recall"}:
        return "Score"
    if column in {"FNR", "x_error", "y_error", "yaw_error"}:
        return "Error"
    if column == "overall_pass_rate":
        return "Pass Rate"
    return "Jobs / Metadata"


def _render_release_performance_html_table(frame: pd.DataFrame) -> None:
    ranges = _release_metric_bar_ranges(frame)
    numeric_columns = {"mAP", "precision", "recall", "overall_pass_rate", "FNR", "x_error", "y_error", "yaw_error", "data_count"}
    group_spans: list[tuple[str, int]] = []
    for column in frame.columns:
        group = _release_performance_column_group(str(column))
        if group_spans and group_spans[-1][0] == group:
            group_spans[-1] = (group, group_spans[-1][1] + 1)
        else:
            group_spans.append((group, 1))
    group_header_html = "".join(
        f'<th class="perf-group-header" colspan="{span}">{escape(group)}</th>'
        for group, span in group_spans
    )
    header_html = "".join(
        (
            f'<th><button class="perf-sort-button" type="button" data-index="{idx}" '
            f'data-type="{"number" if column in numeric_columns else "text"}">{escape(str(column))}</button></th>'
        )
        for idx, column in enumerate(frame.columns)
    )
    row_html = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row.get(column)
            if column == "data_count":
                parsed_count = _parse_data_count(value)
                sort_value = "" if parsed_count is None else str(parsed_count)
            elif column in numeric_columns:
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                sort_value = "" if pd.isna(numeric) else f"{float(numeric):.12g}"
            else:
                sort_value = str(value or "")
            cells.append(
                f'<td class="perf-selectable-td {"perf-metric-td" if column in ranges else ""}" '
                f'data-row="{len(row_html)}" data-col="{len(cells)}" '
                f'data-sort-value="{escape(sort_value, quote=True)}">'
                f"{_release_performance_cell_html(value, column, ranges)}</td>"
            )
        row_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{
  box-sizing: border-box;
}}
body {{
  margin: 0;
  padding: 0;
  background: transparent;
  color: #0f172a;
  font-family: "Source Sans Pro", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.release-perf-table-wrap {{
  overflow-x: auto;
  overflow-y: visible;
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 10px;
}}
.release-perf-table {{
  border-collapse: separate;
  border-spacing: 0;
  min-width: 1280px;
  width: 100%;
  font-size: 0.86rem;
  user-select: none;
}}
.release-perf-table th,
.release-perf-table td {{
  border-bottom: 1px solid rgba(148, 163, 184, 0.22);
  padding: 0.34rem 0.48rem;
  text-align: left;
  vertical-align: middle;
  white-space: nowrap;
}}
.release-perf-table th {{
  position: sticky;
  z-index: 2;
  background: #f8fafc;
  color: #334155;
  font-weight: 750;
  padding: 0;
}}
.release-perf-table .perf-group-header {{
  top: 0;
  z-index: 3;
  padding: 0.3rem 0.48rem;
  text-align: center;
  background: #eef2ff;
  color: #3730a3;
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-right: 1px solid rgba(129, 140, 248, 0.20);
}}
.release-perf-table .perf-column-header th {{
  top: 30px;
}}
.perf-sort-button {{
  appearance: none;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  width: 100%;
  min-height: 32px;
  padding: 0.34rem 0.48rem;
  border: 0;
  background: transparent;
  color: #334155;
  font: inherit;
  font-weight: 750;
  cursor: pointer;
}}
.perf-sort-button:hover {{
  background: rgba(219, 234, 254, 0.62);
}}
.perf-sort-button[data-dir="asc"]::after {{
  content: "▲";
  margin-left: 0.35rem;
  color: #2563eb;
  font-size: 0.64rem;
}}
.perf-sort-button[data-dir="desc"]::after {{
  content: "▼";
  margin-left: 0.35rem;
  color: #2563eb;
  font-size: 0.64rem;
}}
.release-perf-table tbody tr:hover td {{
  background: rgba(248, 250, 252, 0.82);
}}
.release-perf-table td.perf-selected-cell {{
  outline: 1.5px solid #2563eb;
  outline-offset: -2px;
  background: rgba(219, 234, 254, 0.58) !important;
}}
.release-perf-table td.perf-selected-cell .perf-bar-cell {{
  box-shadow: inset 0 0 0 999px rgba(219, 234, 254, 0.34);
}}
.release-perf-table td.perf-selection-anchor {{
  outline: 2px solid #1d4ed8;
  outline-offset: -2px;
}}
.release-perf-table .perf-metric-td {{
  padding: 0;
  min-width: 86px;
  text-align: right;
}}
.perf-bar-cell {{
  position: relative;
  min-height: 32px;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding: 0 0.5rem;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  color: #0f172a;
  overflow: hidden;
}}
.perf-bar-cell::before {{
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: var(--bar-width);
  z-index: 0;
  background: linear-gradient(
    90deg,
    rgba(var(--bar-r), var(--bar-g), var(--bar-b), 0.34),
    rgba(var(--bar-r), var(--bar-g), var(--bar-b), 0.15)
  );
}}
.perf-bar-cell span {{
  position: relative;
  z-index: 1;
}}
.perf-muted {{
  color: #94a3b8;
  display: block;
  padding: 0.34rem 0.5rem;
}}
</style>
</head>
<body>
<div class="release-perf-table-wrap">
  <table id="releasePerfTable" class="release-perf-table">
    <thead>
      <tr>{group_header_html}</tr>
      <tr class="perf-column-header">{header_html}</tr>
    </thead>
    <tbody>{''.join(row_html)}</tbody>
  </table>
</div>
<script>
(function () {{
  const table = document.getElementById("releasePerfTable");
  const tbody = table.querySelector("tbody");
  const buttons = Array.from(table.querySelectorAll(".perf-sort-button"));
  const cells = Array.from(table.querySelectorAll("td.perf-selectable-td"));
  let activeSort = null;
  let isSelecting = false;
  let selectionAnchor = null;

  function rows() {{
    return Array.from(tbody.querySelectorAll("tr"));
  }}

  function cellValue(row, index, type) {{
    const cell = row.children[index];
    const raw = (cell && (cell.dataset.sortValue || cell.innerText) || "").trim();
    if (type === "number") {{
      const value = Number(raw.replace(/,/g, ""));
      return Number.isFinite(value) ? value : -Infinity;
    }}
    return raw.toLowerCase();
  }}

  function applySort() {{
    if (!activeSort) return;
    clearSelection();
    const sortedRows = rows().sort((a, b) => {{
      const av = cellValue(a, activeSort.index, activeSort.type);
      const bv = cellValue(b, activeSort.index, activeSort.type);
      if (av < bv) return activeSort.dir === "asc" ? -1 : 1;
      if (av > bv) return activeSort.dir === "asc" ? 1 : -1;
      return 0;
    }});
    sortedRows.forEach((row) => tbody.appendChild(row));
    buttons.forEach((button) => {{
      const isActive = Number(button.dataset.index) === activeSort.index;
      button.dataset.dir = isActive ? activeSort.dir : "";
    }});
  }}

  buttons.forEach((button) => {{
    button.addEventListener("click", () => {{
      const nextIndex = Number(button.dataset.index);
      const nextType = button.dataset.type || "text";
      const nextDir = activeSort && activeSort.index === nextIndex && activeSort.dir === "asc" ? "desc" : "asc";
      activeSort = {{ index: nextIndex, type: nextType, dir: nextDir }};
      applySort();
    }});
  }});

  function clearSelection() {{
    cells.forEach((cell) => {{
      cell.classList.remove("perf-selected-cell");
      cell.classList.remove("perf-selection-anchor");
    }});
  }}

  function cellPosition(cell) {{
    return {{
      row: rows().indexOf(cell.parentElement),
      col: cell.cellIndex,
    }};
  }}

  function selectRange(anchorCell, targetCell, additive) {{
    if (!anchorCell || !targetCell) return;
    if (!additive) clearSelection();
    const anchor = cellPosition(anchorCell);
    const target = cellPosition(targetCell);
    const rowMin = Math.min(anchor.row, target.row);
    const rowMax = Math.max(anchor.row, target.row);
    const colMin = Math.min(anchor.col, target.col);
    const colMax = Math.max(anchor.col, target.col);
    rows().forEach((row, rowIndex) => {{
      if (rowIndex < rowMin || rowIndex > rowMax) return;
      Array.from(row.children).forEach((cell, colIndex) => {{
        if (colIndex >= colMin && colIndex <= colMax) {{
          cell.classList.add("perf-selected-cell");
        }}
      }});
    }});
    anchorCell.classList.add("perf-selection-anchor");
  }}

  cells.forEach((cell) => {{
    cell.addEventListener("mousedown", (event) => {{
      isSelecting = true;
      selectionAnchor = cell;
      selectRange(selectionAnchor, cell, event.ctrlKey || event.metaKey);
      event.preventDefault();
    }});
    cell.addEventListener("mouseenter", () => {{
      if (isSelecting) {{
        selectRange(selectionAnchor, cell, false);
      }}
    }});
  }});

  document.addEventListener("mouseup", () => {{
    isSelecting = false;
    selectionAnchor = null;
  }});
}})();
</script>
</body>
</html>
"""
    component_height = 76 + max(1, len(frame)) * 34
    components.html(table_html, height=component_height, scrolling=False)


def _release_performance_table(
    frame: pd.DataFrame,
    *,
    family: str,
    empty_message: str,
    table_mode: str,
) -> None:
    if frame.empty:
        st.info(empty_message)
        return
    view = frame[frame["topic_family"] == family].copy()
    if view.empty:
        st.info(empty_message)
        return
    columns = [
        "version",
        "date",
        "description",
        "data_count",
        "mAP",
        "precision",
        "recall",
        "FNR",
        "x_error",
        "y_error",
        "yaw_error",
        "roles",
        "full_job_id",
        "usecase_job_id",
        "devops_job_id",
        "topic_name",
    ]
    if family == "Perception Performance":
        columns.insert(columns.index("roles"), "overall_pass_rate")
    visible = [column for column in columns if column in view.columns]
    display_frame = view.sort_values(["date_sort", "version", "release_name"], ascending=[False, False, False])[visible]
    if table_mode == "Colored bars":
        _render_release_performance_html_table(display_frame)
    else:
        dataframe_height = 52 + max(1, len(display_frame)) * 36
        dataframe_column_config = {
            "version": st.column_config.TextColumn("version", width="large"),
            "description": st.column_config.TextColumn("description", width="medium"),
            "full_job_id": st.column_config.TextColumn("full_job_id", width="large"),
            "usecase_job_id": st.column_config.TextColumn("usecase_job_id", width="large"),
            "devops_job_id": st.column_config.TextColumn("devops_job_id", width="large"),
            "topic_name": st.column_config.TextColumn("topic_name", width="large"),
        }
        st.dataframe(
            display_frame,
            width="stretch",
            hide_index=True,
            height=dataframe_height,
            column_config={key: value for key, value in dataframe_column_config.items() if key in display_frame.columns},
        )


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
    version_order = {version: idx for idx, version in enumerate(versions)}
    plot_df["__version_order"] = plot_df["version"].map(version_order).fillna(len(version_order))
    plot_df = plot_df.sort_values(["__version_order", "version", "date", "release_name"])
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
            series_df = plot_df[plot_df[series_col].astype(str) == series_name].sort_values(
                ["__version_order", "version", "date", "release_name"]
            )
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
    plot_df = frame.dropna(subset=["value"]).copy()
    axis_order = {axis: idx for idx, axis in enumerate(ordered_axes)}
    plot_df["__axis_order"] = plot_df["release_axis"].map(axis_order).fillna(len(axis_order))
    plot_df = plot_df.sort_values(["label_name", "__axis_order", "release_axis"])
    fig = px.line(
        plot_df,
        x="release_axis",
        y="value",
        color="label_name",
        markers=True,
        hover_data=["version", "date", "release_name"],
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=70, b=20), legend_title_text="Label")
    fig.update_xaxes(categoryorder="array", categoryarray=ordered_axes, tickangle=-30, automargin=True)
    fig.update_traces(connectgaps=True)
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
    ].dropna(subset=["value"]).copy()
    axis_order = {axis: idx for idx, axis in enumerate(ordered_axes)}
    profile_df["__axis_order"] = profile_df["release_axis"].map(axis_order).fillna(len(axis_order))
    profile_df = profile_df.sort_values(["metric_name", "__axis_order", "release_axis"])
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
    fig.update_traces(connectgaps=True)
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
            devops_job = group.jobs["devops"]
            flattened = extract_devops_case_rows(
                devops_job.get("devops_summary") or devops_job["summary"]
            )
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
    description="Release history and performance trends.",
)

groups = discover_trend_release_groups()
if not groups:
    st.info("No saved trend metadata was found yet. Use the release trend builder below after the three job summaries are available.")
    _render_release_trend_builder()
    st.stop()

try:
    release_df, case_df, metric_df = _build_release_frames(groups)
except Exception as exc:
    st.error(f"Could not build trend insights: {exc}")
    st.stop()

if not release_df.empty:
    release_df["topic_family"] = release_df["topic_name"].map(_topic_family)

section_header("Release History")
release_specsheets = discover_release_specsheet_inventory(get_data_root())
if release_specsheets:
    release_specsheets = sorted(
        release_specsheets,
        key=lambda row: (
            pd.to_datetime(row.get("date"), format="%Y.%m.%d", errors="coerce").timestamp()
            if pd.notna(pd.to_datetime(row.get("date"), format="%Y.%m.%d", errors="coerce"))
            else -1.0,
            str(row.get("version") or ""),
            str(row.get("release") or ""),
        ),
        reverse=True,
    )
    _render_release_library_table(release_specsheets)
else:
    st.info("No imported release library was found. Run `python scripts/import_catalog_analyzer_releases.py --force` to import analyzer output.")

section_header("Release Performance")
top1, top2, top3, top4, top5 = st.columns(5)
top1.metric("Performance Groups", f"{len(release_df):,}")
top2.metric("Unique Versions", f"{release_df['version'].nunique():,}" if not release_df.empty else "0")
top3.metric("Perception Performance", f"{int((release_df['topic_family'] == 'Perception Performance').sum()):,}" if not release_df.empty else "0")
top4.metric("ML Model Performance", f"{int((release_df['topic_family'] == 'ML Model Performance').sum()):,}" if not release_df.empty else "0")
top5.metric("Latest Date", release_df.sort_values("date_sort")["date"].iloc[-1] if not release_df.empty else "n/a")

performance_table_mode = st.segmented_control(
    "Table view",
    options=["Dataframe", "Colored bars"],
    default="Dataframe",
    key="release_performance_table_mode",
)

st.markdown("#### Perception Performance")
_release_performance_table(
    release_df,
    family="Perception Performance",
    empty_message="No Perception Performance release rows are available.",
    table_mode=performance_table_mode,
)

st.markdown("#### ML Model Performance")
_release_performance_table(
    release_df,
    family="ML Model Performance",
    empty_message="No ML Model Performance release rows are available.",
    table_mode=performance_table_mode,
)

section_header("Major Performance Scores")

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
    latest_major_rows = (
        perf_entries.dropna(subset=major_metric_cols, how="all")
        .sort_values(["date_sort", "version", "release_name"])
        .groupby("topic_family", dropna=False)
        .tail(1)
    )
    metric_card_cols = st.columns(4)
    for family, card_col in zip(("Perception Performance", "ML Model Performance"), metric_card_cols[:2]):
        family_row = latest_major_rows[latest_major_rows["topic_family"] == family]
        if family_row.empty:
            card_col.metric(f"{family} mAP", "n/a")
            continue
        card_col.metric(
            f"{family} mAP",
            f"{family_row['mAP'].iloc[-1]:.3f}" if pd.notna(family_row["mAP"].iloc[-1]) else "n/a",
        )
    latest_perception_row = latest_major_rows[latest_major_rows["topic_family"] == "Perception Performance"]
    latest_model_row = latest_major_rows[latest_major_rows["topic_family"] == "ML Model Performance"]
    metric_card_cols[2].metric(
        "Perception Recall",
        f"{latest_perception_row['recall'].iloc[-1]:.3f}"
        if not latest_perception_row.empty and pd.notna(latest_perception_row["recall"].iloc[-1])
        else "n/a",
    )
    metric_card_cols[3].metric(
        "ML Model Recall",
        f"{latest_model_row['recall'].iloc[-1]:.3f}"
        if not latest_model_row.empty and pd.notna(latest_model_row["recall"].iloc[-1])
        else "n/a",
    )
    fig = go.Figure()
    scenario_totals = (
        perf_entries[perf_entries["topic_family"] == "Perception Performance"]
        .groupby("version", dropna=False)["data_count_num"]
        .max()
        .reindex(perf_entries["version"].drop_duplicates().tolist())
    )
    fig.add_bar(
        x=scenario_totals.index.tolist(),
        y=scenario_totals.tolist(),
        name="Data Count",
        marker_color="#f4a7a7",
        opacity=0.28,
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Data Count: %{y:,}<extra></extra>",
    )
    metric_styles = {
        "mAP": "#0f766e",
        "precision": "#1d4ed8",
        "recall": "#be123c",
    }
    family_dashes = {
        "Perception Performance": "solid",
        "ML Model Performance": "dot",
    }
    for family in ("Perception Performance", "ML Model Performance"):
        family_df = perf_entries[perf_entries["topic_family"] == family].copy()
        if family_df.empty:
            continue
        for metric_col in major_metric_cols:
            metric_df_for_line = family_df.dropna(subset=[metric_col])
            if metric_df_for_line.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=metric_df_for_line["version"],
                    y=metric_df_for_line[metric_col],
                    name=metric_col,
                    legendgroup=family,
                    legendgrouptitle_text=family,
                    mode="lines+markers",
                    line=dict(
                        color=metric_styles[metric_col],
                        width=3,
                        dash=family_dashes.get(family, "solid"),
                    ),
                    marker=dict(size=7),
                    customdata=metric_df_for_line[["release_name", "date", "data_count", "topic_name"]].to_numpy(),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        + f"{family} {metric_col}"
                        + ": %{y:.3f}<br>Release: %{customdata[0]}<br>Date: %{customdata[1]}<br>Data Count: %{customdata[2]}<br>Topic: %{customdata[3]}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        title="Major Performance Scores",
        xaxis_title="Pilot.Auto Version",
        yaxis_title="Score",
        yaxis2=dict(title="Data Count", overlaying="y", side="right", showgrid=False),
        height=520,
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, xanchor="left"),
        legend_tracegroupgap=18,
        margin=dict(l=20, r=20, t=80, b=125),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No grouped major metric trend entries are available yet.")

section_header("Prediction Trend")

prediction_entries = perf_entries[perf_entries["topic_family"] == "Perception Performance"].copy()
prediction_entries = prediction_entries.sort_values(["date_sort", "version", "release_name"], ascending=[True, True, True])

if not prediction_entries.empty and prediction_entries[prediction_cols].notna().any().any():
    pred_card_col1, pred_card_col2, pred_card_col3 = st.columns(3)
    latest_pred_row = prediction_entries.dropna(subset=prediction_cols, how="all").iloc[-1]
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
    pred_story = prediction_entries[
        ["version", "date", "description", "release_name", "data_count", "data_count_num"] + prediction_cols
    ].copy()
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
        metric_story = pred_story.dropna(subset=[metric_name])
        if metric_story.empty:
            continue
        pred_fig.add_trace(
            go.Scatter(
                x=metric_story["version"],
                y=metric_story[metric_name],
                name=metric_name,
                mode="lines+markers",
                line=dict(color=color, width=3 if metric_name.endswith("@3s") else 2, dash=dash),
                marker=dict(size=8),
                customdata=metric_story[["date", "release_name", "data_count"]].to_numpy(),
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
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, xanchor="left"),
        margin=dict(l=20, r=20, t=80, b=105),
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
if not pass_entries.empty:
    pass_entries = pass_entries.copy()
    pass_entries["pass_axis"] = pass_entries["version"].astype(str) + " | " + pass_entries["date"].astype(str)
ordered_versions = pass_entries["pass_axis"].drop_duplicates().tolist() if not pass_entries.empty else []
overall_plot_df = pd.DataFrame()
major_summary = pd.DataFrame()
mid_summary = pd.DataFrame()

if not pass_entries.empty and pass_entries["overall_pass_rate"].notna().any():
    overall_plot_df = pass_entries[
        ["pass_axis", "date", "release_name", "overall_pass_rate", "scenario_count"]
    ].rename(columns={"overall_pass_rate": "pass_rate", "scenario_count": "total"}).copy()
    overall_plot_df = overall_plot_df.rename(columns={"pass_axis": "version"})

if not case_df.empty:
    case_for_pass = case_df.copy()
    case_for_pass["pass_axis"] = case_for_pass["version"].astype(str) + " | " + case_for_pass["date"].astype(str)
    major_summary = (
        case_for_pass.groupby(["pass_axis", "date", "release_name", "major_category"], dropna=False)[["passed", "total"]]
        .sum()
        .reset_index()
        .rename(columns={"pass_axis": "version"})
    )
    major_summary = _with_pass_rate(major_summary)

    mid_summary = (
        case_for_pass.groupby(
            ["pass_axis", "date", "release_name", "major_category", "mid_category"],
            dropna=False,
        )[["passed", "total"]]
        .sum()
        .reset_index()
        .rename(columns={"pass_axis": "version"})
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

_render_release_trend_builder()

if release_specsheets:
    with st.expander("Debug release inventory paths", expanded=False):
        st.dataframe(
            pd.DataFrame(_release_inventory_debug_rows(release_specsheets)),
            width="stretch",
            hide_index=True,
        )
