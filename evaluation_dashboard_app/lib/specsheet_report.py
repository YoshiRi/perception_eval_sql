from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
import yaml

from lib.perception_catalog_io import build_scene_dataframe_from_pkl_dir
from lib.path_utils import get_data_root

DEFAULT_SPECSHEET_TOPIC = "perception.object_recognition.tracking.objects"
DEFAULT_SPECSHEET_PROJECT_ID = "x2_dev"
DEFAULT_SPECSHEET_LABELS = ["car", "truck", "bus", "bicycle", "pedestrian", "motorcycle"]
DEFAULT_SPECSHEET_METRICS = [
    "mAP",
    "precision",
    "recall",
    "FNR",
    "max_consecutive_fn_duration",
    "x_error",
    "y_error",
    "yaw_error",
    "speed_error",
]
FUTURE_SPECSHEET_METRICS = [
    "minADE@1s",
    "minADE@3s",
    "minADE@5s",
    "minFDE@1s",
    "minFDE@3s",
    "minFDE@5s",
]
TREND_METADATA_FILENAME = "metadata.yaml"
TREND_SUMMARY_FILENAME = "summary.json"
FULL_DATASET_EVALUATION_HEADER = "全数データセット評価"
DEFAULT_TREND_METADATA_TEXT = """tags: [trend]
pilot_auto_version: "Pilot.Auto v4.3.0 (centerpoint x2/2.3.1)"
data_count: 99,776+
description: データの追加
date: 2025.11.7
"""
_TREND_DATE_PATTERN = re.compile(r"^\d{4}\.\d{1,2}\.\d{1,2}$")
_TREND_DATA_COUNT_PATTERN = re.compile(r"^\d[\d,]*\+?$")


@dataclass
class TrendReleaseGroup:
    group_key: str
    display_name: str
    topic_name: str
    group_kind: str
    base_dir: Path
    jobs: dict[str, dict[str, Any]]


def get_specsheet_artifact_paths(run_dir: str | Path) -> dict[str, Path]:
    run_path = Path(run_dir)
    return {
        "run_dir": run_path,
        "current_csv": run_path / "current.csv",
        "future_csv": run_path / "future.csv",
        "current_parquet": run_path / "current.parquet",
        "future_parquet": run_path / "future.parquet",
        "resource_dir": run_path / "resources",
        "trend_metadata": run_path / "resources" / TREND_METADATA_FILENAME,
        "trend_summary": run_path / "resources" / TREND_SUMMARY_FILENAME,
        "specsheet_dir": run_path / "specsheet",
        "specsheet_pdf": run_path / "specsheet" / "specsheet.pdf",
    }


def list_specsheet_source_parquets(run_dir: str | Path) -> list[Path]:
    paths = get_specsheet_artifact_paths(run_dir)
    run_path = paths["run_dir"]
    ordered: list[Path] = []
    seen: set[Path] = set()
    for key in ("current_parquet", "future_parquet"):
        path = paths[key]
        if path.exists():
            ordered.append(path)
            seen.add(path)
    for path in sorted(run_path.glob("*.parquet"), key=lambda p: p.name.lower()):
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def get_latest_source_mtime(run_dir: str | Path) -> float | None:
    candidates = list_specsheet_source_parquets(run_dir)
    if not candidates:
        return None
    return max(path.stat().st_mtime for path in candidates if path.exists())


def is_specsheet_pdf_fresh(run_dir: str | Path) -> bool:
    paths = get_specsheet_artifact_paths(run_dir)
    pdf_path = paths["specsheet_pdf"]
    if not pdf_path.exists():
        return False
    latest_source_mtime = get_latest_source_mtime(run_dir)
    if latest_source_mtime is None:
        return True
    return pdf_path.stat().st_mtime >= latest_source_mtime


def _notify(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


@contextmanager
def _patch_block_generation_progress(
    progress_callback: Callable[[str], None] | None,
):
    if progress_callback is None:
        yield
        return

    try:
        from perception_catalog_analyzer.specsheet import blocks as specsheet_blocks
    except ImportError:
        yield
        return

    original_tqdm = specsheet_blocks.tqdm

    class ProgressTqdm:
        def __init__(self, iterable, desc: str | None = None, **kwargs):
            self._items = list(iterable)
            self._desc = desc or ""
            self._current_index = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            for idx, item in enumerate(self._items, start=1):
                self._current_index = idx
                yield item

        def set_postfix_str(self, text: str) -> None:
            total = len(self._items)
            if total <= 0:
                return
            _notify(
                progress_callback,
                f"{self._desc} {self._current_index}/{total}: {text}",
            )

    specsheet_blocks.tqdm = ProgressTqdm
    try:
        yield
    finally:
        specsheet_blocks.tqdm = original_tqdm


def _copy_parquet_to_csv(parquet_path: Path, csv_path: Path) -> Path:
    frame = pd.read_parquet(parquet_path)
    frame.to_csv(csv_path, index=False)
    return csv_path


def _prefer_cjk_font_stack(html_lines: Sequence[str]) -> list[str]:
    rendered = list(html_lines)
    generic = "font-family: sans-serif;"
    preferred = (
        'font-family: "Noto Sans CJK JP", "Noto Sans JP", '
        '"IPAGothic", "IPA Gothic", sans-serif;'
    )
    return [line.replace(generic, preferred) for line in rendered]


def parse_trend_metadata_text(text: str) -> dict[str, Any]:
    """Parse and validate manual trend metadata YAML input."""
    raw = yaml.safe_load(text or "")
    if not isinstance(raw, dict):
        raise ValueError("Trend metadata must be a YAML object with key/value pairs.")

    tags = raw.get("tags")
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list) or not any(str(tag).strip() == "trend" for tag in tags):
        raise ValueError("Trend metadata must include `tags: [trend]`.")

    pilot_auto_version = str(raw.get("pilot_auto_version") or "").strip()
    if not pilot_auto_version:
        raise ValueError("Trend metadata requires a non-empty `pilot_auto_version`.")

    data_count = str(raw.get("data_count") or "").strip()
    if not data_count or not _TREND_DATA_COUNT_PATTERN.match(data_count):
        raise ValueError(
            "Trend metadata `data_count` must look like `99,776+` or `12345`."
        )

    description = str(raw.get("description") or "").strip()
    date = str(raw.get("date") or "").strip()
    if not date or not _TREND_DATE_PATTERN.match(date):
        raise ValueError("Trend metadata `date` must look like `2025.11.7`.")

    parsed = {
        "tags": ["trend"],
        "pilot_auto_version": pilot_auto_version,
        "data_count": data_count,
        "description": description,
        "date": date,
    }
    for optional_key in ("release_group", "topic_name"):
        optional_value = str(raw.get(optional_key) or "").strip()
        if optional_value:
            parsed[optional_key] = optional_value
    return parsed


def write_trend_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> Path:
    paths = get_specsheet_artifact_paths(run_dir)
    resource_dir = paths["resource_dir"]
    metadata_path = paths["trend_metadata"]
    resource_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(metadata, fh, allow_unicode=True, sort_keys=False)
    return metadata_path


def discover_trend_metadata_files(root_dir: str | Path | None = None) -> list[Path]:
    base_dir = Path(root_dir) if root_dir is not None else get_data_root()
    if not base_dir.exists():
        return []

    matches: list[Path] = []
    for metadata_path in base_dir.rglob(TREND_METADATA_FILENAME):
        if not metadata_path.is_file():
            continue
        if not (metadata_path.parent / TREND_SUMMARY_FILENAME).exists():
            continue
        matches.append(metadata_path)
    return sorted(dict.fromkeys(path.resolve() for path in matches), key=lambda p: str(p))


def load_trend_metadata_file(metadata_path: str | Path) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid trend metadata file: {metadata_path}")
    return data


def load_trend_summary_file(summary_path: str | Path) -> dict[str, Any]:
    with Path(summary_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid trend summary file: {summary_path}")
    return data


def classify_trend_summary(summary: dict[str, Any]) -> str:
    blocks = summary.get("blocks")
    if isinstance(blocks, list):
        headers = [str(block.get("header") or "") for block in blocks]
        if "全数データセット評価" in headers:
            return "full"
        if "ユースケース評価" in headers:
            return "usecase"
        return "performance_blocks"
    if isinstance(summary, dict) and summary:
        return "devops"
    return "unknown"


def discover_trend_release_groups(root_dir: str | Path | None = None) -> list[TrendReleaseGroup]:
    metadata_files = discover_trend_metadata_files(root_dir)
    grouped: dict[str, TrendReleaseGroup] = {}
    standalone_records: list[dict[str, Any]] = []

    for metadata_path in metadata_files:
        summary_path = metadata_path.parent / TREND_SUMMARY_FILENAME
        summary = load_trend_summary_file(summary_path)
        role = classify_trend_summary(summary)
        metadata = load_trend_metadata_file(metadata_path)

        if metadata_path.parent.name == "resources":
            run_dir = metadata_path.parent.parent
            group_key = f"run::{run_dir.resolve()}"
            display_name = run_dir.name
            topic_name = str(metadata.get("topic_name") or "standalone")
            group_kind = "standalone_run"
            base_dir = run_dir
            standalone_records.append(
                {
                    "group_key": group_key,
                    "display_name": display_name,
                    "topic_name": topic_name,
                    "group_kind": group_kind,
                    "base_dir": base_dir,
                    "role": role,
                    "job_id": run_dir.name,
                    "metadata_path": metadata_path,
                    "summary_path": summary_path,
                    "metadata": metadata,
                    "summary": summary,
                }
            )
            continue
        else:
            job_dir = metadata_path.parent
            topic_dir = job_dir.parent
            combined_dir = topic_dir.parent
            group_key = f"group::{combined_dir.resolve()}::{topic_dir.name}"
            display_name = combined_dir.name
            topic_name = topic_dir.name
            group_kind = "library_pdf_group"
            base_dir = combined_dir

        if group_key not in grouped:
            grouped[group_key] = TrendReleaseGroup(
                group_key=group_key,
                display_name=display_name,
                topic_name=topic_name,
                group_kind=group_kind,
                base_dir=base_dir,
                jobs={},
            )
        grouped[group_key].jobs[role] = {
            "role": role,
            "job_id": metadata_path.parent.name if metadata_path.parent.name != "resources" else run_dir.name,
            "metadata_path": metadata_path.resolve(),
            "summary_path": summary_path.resolve(),
            "metadata": metadata,
            "summary": summary,
        }

    standalone_by_release: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for record in standalone_records:
        metadata = record["metadata"]
        release_key = (
            str(metadata.get("release_group") or ""),
            str(record["topic_name"] or ""),
            str(metadata.get("pilot_auto_version") or ""),
            str(metadata.get("date") or ""),
            str(metadata.get("description") or ""),
            str(metadata.get("data_count") or ""),
        )
        standalone_by_release.setdefault(release_key, []).append(record)

    for release_key, records in standalone_by_release.items():
        role_counts: dict[str, int] = {}
        for record in records:
            role = str(record["role"])
            role_counts[role] = role_counts.get(role, 0) + 1

        can_group = len(records) > 1 and all(count == 1 for count in role_counts.values())
        if can_group:
            sample = records[0]
            metadata = sample["metadata"]
            release_label = (
                str(metadata.get("release_group") or "").strip()
                or str(metadata.get("pilot_auto_version") or "").strip()
                or "standalone_release"
            )
            date_label = str(metadata.get("date") or "").strip()
            display_name = f"{release_label} | {date_label}" if date_label else release_label
            group_key = "standalone_group::" + "::".join(release_key)
            grouped[group_key] = TrendReleaseGroup(
                group_key=group_key,
                display_name=display_name,
                topic_name=str(sample["topic_name"]),
                group_kind="standalone_release_group",
                base_dir=Path(root_dir) if root_dir is not None else get_data_root(),
                jobs={},
            )
            target_group = grouped[group_key]
            for record in records:
                target_group.jobs[str(record["role"])] = {
                    "role": record["role"],
                    "job_id": record["job_id"],
                    "metadata_path": record["metadata_path"].resolve(),
                    "summary_path": record["summary_path"].resolve(),
                    "metadata": record["metadata"],
                    "summary": record["summary"],
                }
            continue

        for record in records:
            group_key = str(record["group_key"])
            grouped[group_key] = TrendReleaseGroup(
                group_key=group_key,
                display_name=str(record["display_name"]),
                topic_name=str(record["topic_name"]),
                group_kind=str(record["group_kind"]),
                base_dir=record["base_dir"],
                jobs={
                    str(record["role"]): {
                        "role": record["role"],
                        "job_id": record["job_id"],
                        "metadata_path": record["metadata_path"].resolve(),
                        "summary_path": record["summary_path"].resolve(),
                        "metadata": record["metadata"],
                        "summary": record["summary"],
                    }
                },
            )

    def _sort_key(group: TrendReleaseGroup) -> tuple[str, str]:
        dates = [
            str(job["metadata"].get("date") or "")
            for job in group.jobs.values()
            if isinstance(job.get("metadata"), dict)
        ]
        newest = max(dates) if dates else ""
        return (newest, group.display_name)

    return sorted(grouped.values(), key=_sort_key)


def _trend_version_sort_key(pilot_auto_version: str) -> tuple[tuple[int, int, int], str, tuple[int, int, int]]:
    pattern = r"v(\d+)\.(\d+)\.(\d+)\s*\(([^ ]+)\s+(.+)\)"
    match = re.search(pattern, str(pilot_auto_version or ""))
    if not match:
        return ((999, 999, 999), str(pilot_auto_version or ""), (999, 999, 999))

    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    ml_model_type = match.group(4)
    ml_model_info = match.group(5)
    try:
        _, ml_model_version = ml_model_info.split("/")
        ml_major, ml_minor, ml_patch = ml_model_version.split(".")
        ml_version = (int(ml_major), int(ml_minor), int(ml_patch))
    except ValueError:
        ml_version = (999, 999, 999)
    return ((major, minor, patch), ml_model_type, ml_version)


def _canonical_summary_table_key(table_data: dict[str, Any]) -> str:
    return json.dumps(table_data, ensure_ascii=False, sort_keys=True, allow_nan=True)


def _deduplicate_summary_tables(data_list: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for table_data in data_list:
        key = _canonical_summary_table_key(table_data)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(table_data)
    return deduplicated


def _extract_full_metric_tables(summary: dict[str, Any]) -> list[dict[str, Any]]:
    data_list: list[dict[str, Any]] = []
    blocks = summary.get("blocks", [])
    if not isinstance(blocks, list):
        return data_list
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("header") != FULL_DATASET_EVALUATION_HEADER:
            continue
        if block.get("mode") not in (None, "metrics"):
            continue
        if block.get("evaluation_type") not in (None, "full"):
            continue
        block_tables = block.get("tables", [])
        if not isinstance(block_tables, list):
            continue
        for tables in block_tables:
            if not isinstance(tables, dict):
                continue
            table_data = tables.get("data", {})
            if isinstance(table_data, dict) and table_data:
                data_list.append(table_data)
    return _deduplicate_summary_tables(data_list)


def _load_only_full_summary(summary_path: Path) -> list[dict[str, Any]]:
    summary = load_trend_summary_file(summary_path)
    return _extract_full_metric_tables(summary)


def ensure_full_trend_summary(summary_path: str | Path) -> Path:
    """Validate that analyzer block generation produced a full trend summary."""
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Full trend summary was not created: {path}. "
            "The analyzer must write resources/summary.json before trend PDF generation."
        )
    summary = load_trend_summary_file(path)
    role = classify_trend_summary(summary)
    if role != "full":
        raise ValueError(f"Expected a full trend summary at {path}, but it classified as `{role}`.")
    extract_performance_metrics_from_summary(summary)
    return path


def extract_performance_metrics_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    """Return averaged full-performance metrics from a full summary payload."""
    data_list = _extract_full_metric_tables(summary)

    if len(data_list) != 1:
        raise ValueError(f"Expected exactly one distinct full summary table, but got {len(data_list)}")
    metrics = data_list[0]

    def _avg(metric_name: str) -> float:
        values = metrics.get(metric_name, {})
        if not isinstance(values, dict) or not values:
            return float("nan")
        numeric = pd.to_numeric(pd.Series(list(values.values())), errors="coerce")
        return float(numeric.mean())

    return {
        "mAP": _avg("mAP"),
        "precision": _avg("precision"),
        "recall": _avg("recall"),
        "FNR": _avg("FNR"),
        "x_error": _avg("x_error"),
        "y_error": _avg("y_error"),
        "yaw_error": _avg("yaw_error"),
        "speed_error": _avg("speed_error"),
        "minADE@1s": _avg("minADE@1s"),
        "minFDE@1s": _avg("minFDE@1s"),
        "minADE@3s": _avg("minADE@3s"),
        "minFDE@3s": _avg("minFDE@3s"),
        "minADE@5s": _avg("minADE@5s"),
        "minFDE@5s": _avg("minFDE@5s"),
    }


def extract_devops_case_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten nested devops/pass-rate summary into case rows."""
    rows: list[dict[str, Any]] = []
    for major_category, mid_categories in summary.items():
        if not isinstance(mid_categories, dict):
            continue
        for mid_category, minor_or_cases in mid_categories.items():
            if not isinstance(minor_or_cases, dict):
                continue
            for minor_or_case_name, result_or_cases in minor_or_cases.items():
                if not isinstance(result_or_cases, dict):
                    continue
                if {"passed", "total"}.intersection(result_or_cases.keys()):
                    case_items = [(minor_or_case_name, result_or_cases)]
                    minor_category = minor_or_case_name
                else:
                    case_items = [
                        (case_name, result)
                        for case_name, result in result_or_cases.items()
                        if isinstance(result, dict)
                    ]
                    minor_category = minor_or_case_name

                for case_name, result in case_items:
                    passed = int(result.get("passed", 0) or 0)
                    total = int(result.get("total", 0) or 0)
                    rows.append(
                        {
                            "major_category": major_category,
                            "mid_category": mid_category,
                            "minor_category": minor_category,
                            "case_name": case_name,
                            "passed": passed,
                            "total": total,
                            "pass_rate": (passed / total * 100.0) if total > 0 else None,
                        }
                    )
    return rows


def load_performance_trend_data(metadata_list: Sequence[Path]) -> list[dict[str, str | int | float]]:
    trend_data_rows: list[dict[str, Any]] = []
    for metadata_path in metadata_list:
        metadata = load_trend_metadata_file(metadata_path)
        if "trend" not in [str(tag).strip() for tag in metadata.get("tags", [])]:
            continue
        summary_path = Path(metadata_path).parent / TREND_SUMMARY_FILENAME
        if not summary_path.exists():
            continue
        summary_list = _load_only_full_summary(summary_path)
        if not summary_list:
            continue
        trend_data_rows.append(
            {
                "version": metadata.get("pilot_auto_version"),
                "data_count": metadata.get("data_count"),
                "description": metadata.get("description"),
                "date": metadata.get("date"),
                "summary": summary_list,
            }
        )

    trend_data_rows.sort(key=lambda row: _trend_version_sort_key(str(row.get("version") or "")))

    output: list[dict[str, str | int | float]] = []
    for row in trend_data_rows:
        summary = row.get("summary") or []
        if len(summary) != 1:
            raise ValueError(
                f"Expected exactly one distinct summary block for version {row.get('version')}, "
                f"but got {len(summary)}"
            )
        metrics = summary[0]

        def _avg(metric_name: str) -> float:
            values = metrics.get(metric_name, {})
            if not isinstance(values, dict) or not values:
                return float("nan")
            numeric = pd.to_numeric(pd.Series(list(values.values())), errors="coerce")
            return float(numeric.mean())

        output.append(
            {
                "version": row.get("version"),
                "data_count": row.get("data_count"),
                "description": row.get("description"),
                "date": row.get("date"),
                "mAP": _avg("mAP"),
                "minADE@1s": _avg("minADE@1s"),
                "minFDE@1s": _avg("minFDE@1s"),
                "minADE@3s": _avg("minADE@3s"),
                "minFDE@3s": _avg("minFDE@3s"),
                "minADE@5s": _avg("minADE@5s"),
                "minFDE@5s": _avg("minFDE@5s"),
            }
        )
    return output


def load_devops_trend_data(metadata_list: Sequence[Path]) -> list[dict[str, Any]]:
    trend_data_rows: list[dict[str, Any]] = []
    for metadata_path in metadata_list:
        metadata = load_trend_metadata_file(metadata_path)
        if "trend" not in [str(tag).strip() for tag in metadata.get("tags", [])]:
            continue
        summary_path = Path(metadata_path).parent / TREND_SUMMARY_FILENAME
        if not summary_path.exists():
            continue
        summary = load_trend_summary_file(summary_path)
        if classify_trend_summary(summary) != "devops":
            continue

        rows = extract_devops_case_rows(summary)
        if not rows:
            continue
        overall_passed = sum(int(row["passed"]) for row in rows)
        overall_total = sum(int(row["total"]) for row in rows)
        trend_data_rows.append(
            {
                "version": metadata.get("pilot_auto_version"),
                "data_count": metadata.get("data_count"),
                "description": metadata.get("description"),
                "date": metadata.get("date"),
                "overall_pass_rate": (overall_passed / overall_total * 100.0)
                if overall_total > 0
                else 0.0,
                "scenario_count": overall_total,
                "devops_data": summary,
            }
        )

    trend_data_rows.sort(key=lambda row: _trend_version_sort_key(str(row.get("version") or "")))
    return trend_data_rows


def _add_devops_detail_trend_rates(devops_trend_data: Sequence[dict[str, Any]]) -> list[str]:
    cases: set[str] = set()
    for row in devops_trend_data:
        devops_data = row.get("devops_data", {})
        if not isinstance(devops_data, dict):
            continue
        for mid_categories in devops_data.values():
            if not isinstance(mid_categories, dict):
                continue
            for sub_category, sub_categories in mid_categories.items():
                if not isinstance(sub_categories, dict):
                    continue
                total_passed = sum(
                    int(result.get("passed", 0) or 0)
                    for result in sub_categories.values()
                    if isinstance(result, dict)
                )
                total = sum(
                    int(result.get("total", 0) or 0)
                    for result in sub_categories.values()
                    if isinstance(result, dict)
                )
                row[sub_category] = total_passed / total * 100.0 if total > 0 else 0.0
                cases.add(str(sub_category))
    return sorted(cases)


def _build_trend_context(
    metadata_list: Sequence[Path],
    output_dir: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if not metadata_list:
        return {
            "performance_trend_data": [],
            "map_trend_plot_path": output_dir / "map_trend.png",
            "prediction_trend_plot_path": output_dir / "prediction_trend.png",
            "devops_trend_data": [],
            "devops_trend_plot_path": output_dir / "devops_trend.png",
            "job_ids": [],
        }

    try:
        from perception_catalog_analyzer.plot.map_trend import generate_map_trend_plot
        from perception_catalog_analyzer.plot.prediction_trend import generate_prediction_trend_plot
        from perception_catalog_analyzer.plot.devops_trend import (
            generate_devops_trend_detail_plot,
            generate_devops_trend_plot,
        )
    except ImportError as exc:
        raise RuntimeError(
            "perception_catalog_analyzer trend support is unavailable. "
            f"Original error: {exc!s}"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    _notify(progress_callback, "Collecting trend history")
    performance_trend_data = load_performance_trend_data(list(metadata_list))
    map_trend_plot_path = output_dir / "map_trend.png"
    prediction_trend_plot_path = output_dir / "prediction_trend.png"
    if performance_trend_data:
        _notify(progress_callback, "Rendering trend plots")
        generate_map_trend_plot(performance_trend_data, map_trend_plot_path)
        generate_prediction_trend_plot(performance_trend_data, prediction_trend_plot_path)

    devops_trend_data = load_devops_trend_data(list(metadata_list))
    devops_trend_plot_path = output_dir / "devops_trend.png"
    if devops_trend_data:
        _notify(progress_callback, "Rendering pass-rate trend plots")
        generate_devops_trend_plot(devops_trend_data, devops_trend_plot_path)
        detail_cases = _add_devops_detail_trend_rates(devops_trend_data)
        if detail_cases:
            generate_devops_trend_detail_plot(
                devops_trend_data,
                detail_cases,
                devops_trend_plot_path,
            )

    return {
        "performance_trend_data": performance_trend_data,
        "map_trend_plot_path": map_trend_plot_path,
        "prediction_trend_plot_path": prediction_trend_plot_path,
        "devops_trend_data": devops_trend_data,
        "devops_trend_plot_path": devops_trend_plot_path,
        "job_ids": [],
    }


def _update_template_compat(
    update_template_func: Callable[..., Sequence[str]],
    project_id: str,
    version: str,
    *,
    template_dir: Path,
    context_dir: Path,
    trend_context: dict[str, object] | None = None,
) -> Sequence[str]:
    """Call update_template across analyzer versions with different signatures."""
    try:
        parameters = inspect.signature(update_template_func).parameters
    except (TypeError, ValueError):
        parameters = {}

    trend_context = trend_context or {}
    semantic_kwargs = {
        "project_id": project_id,
        "pilot_auto_version": version,
        "version": version,
        "devops_data": {},
        "devops_plot_path": None,
        "performance_trend_data": trend_context.get("performance_trend_data", []),
        "map_trend_plot_path": trend_context.get("map_trend_plot_path", context_dir / "map_trend.png"),
        "prediction_trend_plot_path": trend_context.get(
            "prediction_trend_plot_path", context_dir / "prediction_trend.png"
        ),
        "devops_trend_data": trend_context.get("devops_trend_data", []),
        "devops_trend_plot_path": trend_context.get(
            "devops_trend_plot_path", context_dir / "devops_trend.png"
        ),
        "job_ids": trend_context.get("job_ids", []),
        "template_name": "static_body.html",
        "extensions": ["html"],
        "template_dir": str(template_dir),
        "show_other_infos": bool(trend_context.get("performance_trend_data")),
    }

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    if accepts_kwargs or not parameters:
        return update_template_func(**semantic_kwargs)

    args: list[object] = []
    kwargs: dict[str, object] = {}
    for name, param in parameters.items():
        if name not in semantic_kwargs:
            continue
        value = semantic_kwargs[name]
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(value)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
    return update_template_func(*args, **kwargs)

def _scene_dataframe_from_dir_compat(
    scene_dataframe_cls,
    run_path: Path,
    *,
    topic_name: str,
):
    """Call SceneDataFrame.from_dir across analyzer versions with/without topic."""
    from_dir = scene_dataframe_cls.from_dir
    try:
        parameters = inspect.signature(from_dir).parameters
    except (TypeError, ValueError):
        parameters = {}

    required_parameters = [
        param
        for param in parameters.values()
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and param.default is inspect.Parameter.empty
    ]
    accepts_varargs = any(
        param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for param in parameters.values()
    )

    if accepts_varargs or len(required_parameters) >= 2:
        return from_dir(run_path, topic_name)
    return from_dir(run_path)


_CURRENT_NUMERIC_COLUMNS = {
    "unix_time",
    "x",
    "y",
    "confidence",
    "pointcloud_num",
    "visibility",
    "x_error",
    "y_error",
    "yaw_error",
    "speed_error",
    "frame_index",
}
_FUTURE_NUMERIC_COLUMNS = {
    "x",
    "y",
    "tx",
    "ty",
    "confidence",
    "visibility",
    "relative_time",
    "pair_dt_sec",
}


def _coerce_numeric_columns(frame: pd.DataFrame, columns: set[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    coerced = frame.copy()
    for column in sorted(columns.intersection(coerced.columns)):
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def _coerce_specsheet_scene_numeric_columns(df):
    """Normalize analyzer-loaded CSV values before NumPy-heavy specsheet metrics."""
    if hasattr(df, "current"):
        df.current = _coerce_numeric_columns(df.current, _CURRENT_NUMERIC_COLUMNS)
        if getattr(df, "future", None) is not None:
            df.future = _coerce_numeric_columns(df.future, _FUTURE_NUMERIC_COLUMNS)
        return df
    if isinstance(df, pd.DataFrame):
        return _coerce_numeric_columns(
            df,
            _CURRENT_NUMERIC_COLUMNS | _FUTURE_NUMERIC_COLUMNS,
        )
    return df


def _get_blocks_compat(
    get_blocks_func: Callable[..., tuple[Sequence[str], Sequence[str]]],
    *,
    df,
    labels: Sequence[str],
    metrics: Sequence[str],
    topic_name: str,
    outdir: Path,
    evaluation_type: str,
):
    """Call get_blocks across analyzer versions with different keyword support."""
    semantic_kwargs = {
        "df": df,
        "labels": list(labels),
        "metrics": list(metrics),
        "topic_name": topic_name,
        "topic": topic_name,
        "path": outdir,
        "outdir": outdir,
        "evaluation_type": evaluation_type,
    }
    try:
        parameters = inspect.signature(get_blocks_func).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    if accepts_kwargs or not parameters:
        return get_blocks_func(**semantic_kwargs)

    args: list[object] = []
    kwargs: dict[str, object] = {}
    for name, param in parameters.items():
        if name not in semantic_kwargs:
            continue
        value = semantic_kwargs[name]
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(value)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
    return get_blocks_func(*args, **kwargs)


def _specsheet_compat(
    specsheet_func: Callable[..., None],
    *,
    html: Sequence[str],
    abstract_html: Sequence[str],
    detailed_html: Sequence[str],
    outdir: Path,
    report_name: str,
) -> None:
    """Call specsheet across analyzer versions with path/outdir differences."""
    semantic_kwargs = {
        "html": list(html),
        "abstract_html": list(abstract_html),
        "detailed_html": list(detailed_html),
        "path": outdir,
        "outdir": outdir,
        "report_name": report_name,
    }
    try:
        parameters = inspect.signature(specsheet_func).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    if accepts_kwargs or not parameters:
        specsheet_func(**semantic_kwargs)
        return

    args: list[object] = []
    kwargs: dict[str, object] = {}
    for name, param in parameters.items():
        if name not in semantic_kwargs:
            continue
        value = semantic_kwargs[name]
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(value)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
    specsheet_func(*args, **kwargs)


def ensure_specsheet_csvs(
    run_dir: str | Path,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Path | None]:
    paths = get_specsheet_artifact_paths(run_dir)
    current_csv = paths["current_csv"]
    future_csv = paths["future_csv"]
    current_parquet = paths["current_parquet"]
    future_parquet = paths["future_parquet"]

    if not current_csv.exists():
        if current_parquet.exists():
            _notify(progress_callback, f"Converting {current_parquet.name} -> {current_csv.name}")
            _copy_parquet_to_csv(current_parquet, current_csv)
        elif list_specsheet_source_parquets(run_dir):
            fallback = list_specsheet_source_parquets(run_dir)[0]
            _notify(progress_callback, f"Converting {fallback.name} -> {current_csv.name}")
            _copy_parquet_to_csv(fallback, current_csv)
        else:
            _notify(progress_callback, "No CSV found. Building CSV from pkl / pkl.z files")
            skip_counts: dict[str, int] = {}

            def _on_progress(done: int, total: int) -> None:
                _notify(progress_callback, f"Processing pkl files {done}/{total}")

            def _on_skip(path: str | Path, reason: str) -> None:
                skip_counts[reason] = skip_counts.get(reason, 0) + 1

            df = build_scene_dataframe_from_pkl_dir(
                run_dir,
                on_progress=_on_progress,
                on_skip=_on_skip,
            )
            if skip_counts:
                details = ", ".join(
                    f"{count} {reason}" for reason, count in sorted(skip_counts.items())
                )
                _notify(progress_callback, f"Skipped pkl files: {details}")
            df.to_csv(run_dir)
            if not current_csv.exists():
                raise FileNotFoundError(f"Failed to generate {current_csv}")

    if not future_csv.exists() and future_parquet.exists():
        _notify(progress_callback, f"Converting {future_parquet.name} -> {future_csv.name}")
        _copy_parquet_to_csv(future_parquet, future_csv)

    return {
        "current_csv": current_csv if current_csv.exists() else None,
        "future_csv": future_csv if future_csv.exists() else None,
    }


def generate_specsheet_pdf(
    run_dir: str | Path,
    *,
    project_id: str,
    version: str,
    labels: Sequence[str],
    topic_name: str = DEFAULT_SPECSHEET_TOPIC,
    include_trend: bool = False,
    trend_metadata: dict[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Path, bool]:
    paths = get_specsheet_artifact_paths(run_dir)
    specsheet_dir = paths["specsheet_dir"]
    pdf_path = paths["specsheet_pdf"]

    if not force and is_specsheet_pdf_fresh(run_dir):
        _notify(progress_callback, "Using existing up-to-date spec-sheet PDF")
        return pdf_path, False

    ensure_specsheet_csvs(run_dir, progress_callback=progress_callback)

    try:
        from perception_catalog_analyzer.dataframe import SceneDataFrame
        from perception_catalog_analyzer.specsheet import get_blocks, specsheet
        from perception_catalog_analyzer import template as template_module
        from perception_catalog_analyzer.template import update_template
    except ImportError as exc:
        raise RuntimeError(
            "perception_catalog_analyzer spec-sheet generation is unavailable. "
            f"Install the dependency first. Original error: {exc!s}"
        ) from exc

    run_path = paths["run_dir"]
    resource_dir = run_path / "resources"
    resource_dir.mkdir(parents=True, exist_ok=True)
    specsheet_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, "Loading CSV files")
    df = _scene_dataframe_from_dir_compat(
        SceneDataFrame,
        run_path,
        topic_name=topic_name,
    )
    df = _coerce_specsheet_scene_numeric_columns(df)
    metrics = list(DEFAULT_SPECSHEET_METRICS)
    if getattr(df, "future", None) is not None:
        metrics.extend(FUTURE_SPECSHEET_METRICS)

    _notify(progress_callback, "Building abstract and detail sections")
    with _patch_block_generation_progress(progress_callback):
        abstract, detailed = _get_blocks_compat(
            get_blocks,
            df=df,
            labels=list(labels),
            metrics=metrics,
            topic_name=topic_name,
            outdir=resource_dir.resolve(),
            evaluation_type="full",
        )

    trend_context: dict[str, object] | None = None
    if include_trend:
        if trend_metadata is None:
            raise ValueError("Trend metadata is required when trend mode is enabled.")
        _notify(progress_callback, "Validating full trend summary")
        ensure_full_trend_summary(paths["trend_summary"])
        _notify(progress_callback, "Saving trend metadata")
        write_trend_metadata(run_path, trend_metadata)
        metadata_list = discover_trend_metadata_files()
        trend_context = _build_trend_context(
            metadata_list,
            specsheet_dir,
            progress_callback=progress_callback,
        )

    _notify(progress_callback, "Rendering PDF")
    template_dir = Path(template_module.__file__).resolve().parent.parent / "template"
    html = _prefer_cjk_font_stack(
        _update_template_compat(
            update_template,
            project_id,
            version,
            template_dir=template_dir,
            context_dir=specsheet_dir,
            trend_context=trend_context,
        )
    )
    _specsheet_compat(
        specsheet,
        html=html,
        abstract_html=abstract,
        detailed_html=detailed,
        outdir=specsheet_dir,
        report_name="specsheet",
    )
    if not pdf_path.exists():
        raise FileNotFoundError(f"Spec-sheet PDF was not created: {pdf_path}")
    _notify(progress_callback, "Spec-sheet PDF is ready")
    return pdf_path, True


def collect_candidate_specsheet_labels(
    run_dir: str | Path,
    *,
    preferred: Iterable[str] | None = None,
) -> list[str]:
    preferred_labels = [str(v) for v in (preferred or []) if str(v).strip()]
    if preferred_labels:
        return sorted(dict.fromkeys(preferred_labels))

    paths = get_specsheet_artifact_paths(run_dir)
    for source in (
        paths["current_csv"],
        paths["current_parquet"],
    ):
        if not source.exists():
            continue
        try:
            if source.suffix == ".csv":
                frame = pd.read_csv(source)
            else:
                frame = pd.read_parquet(source, columns=["label"])
            if "label" not in frame.columns:
                continue
            labels = [str(v) for v in frame["label"].dropna().unique() if str(v).strip()]
            if labels:
                return sorted(labels)
        except Exception:
            continue
    return []


_PROGRESS_FRACTION_PATTERN = re.compile(r"(?P<done>\d+)\s*/\s*(?P<total>\d+)")


def progress_fraction_from_message(message: str) -> float | None:
    match = _PROGRESS_FRACTION_PATTERN.search(message or "")
    if not match:
        return None
    done = int(match.group("done"))
    total = int(match.group("total"))
    if total <= 0:
        return None
    return max(0.0, min(1.0, done / total))
