from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import json
import os
import re
import shutil
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
from pandas.errors import EmptyDataError
import yaml

from lib.path_utils import get_data_root
from lib.run_metadata import read_run_metadata

DEFAULT_SPECSHEET_TOPIC = "perception.object_recognition.objects"
DEFAULT_TREND_TOPIC = "perception.object_recognition.objects"
DETECTION_TREND_TOPIC_BY_MODEL = {
    "bevfusion": "perception.object_recognition.detection.bevfusion.objects",
    "centerpoint": "perception.object_recognition.detection.centerpoint.objects",
}
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
SPECSHEET_RELEASE_ROLE_DIRS = ("performance", "usecase", "devops")
GENERATED_TREND_HISTORY_DIRNAME = "_app_trend_history"
FULL_DATASET_EVALUATION_HEADER = "全数データセット評価"
USECASE_PLANNING_EVALUATION_HEADERS = {"ユースケース評価", "ユースケース(Planning)評価"}
USECASE_DEVOPS_EVALUATION_HEADER = "ユースケース(過去課題)評価"
DEFAULT_TREND_METADATA_TEXT = """tags: [trend]
pilot_auto_version: "Pilot.Auto v4.3.0 (centerpoint x2/2.3.1)"
pilot_auto_version_abbr: p430-c231
data_count: 99,776+
description: データの追加
date: 2025.11.7
"""
_TREND_DATE_PATTERN = re.compile(r"^\d{4}\.\d{1,2}\.\d{1,2}$")
_TREND_DATA_COUNT_PATTERN = re.compile(r"^\d[\d,]*\+?$")
_PILOT_AUTO_PREFIX_PATTERN = re.compile(r"^Pilot\.Auto\s+", re.IGNORECASE)


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


def _topic_values_from_frame(frame: pd.DataFrame) -> list[str]:
    for column in ("topic_name", "topic"):
        if column not in frame.columns:
            continue
        values = [
            str(value).strip()
            for value in frame[column].dropna().unique().tolist()
            if str(value).strip()
        ]
        if values:
            return sorted(values)
    return []


def detect_specsheet_topic_names(run_dir: str | Path, *, csv_sample_rows: int = 50000) -> list[str]:
    """Detect topic names already present in specsheet CSV/parquet artifacts."""
    paths = get_specsheet_artifact_paths(run_dir)
    detected: set[str] = set()

    for parquet_path in (paths["current_parquet"], paths["future_parquet"]):
        if not parquet_path.exists():
            continue
        try:
            import pyarrow.parquet as pq

            columns = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        except Exception:
            try:
                columns = set(pd.read_parquet(parquet_path, columns=[]).columns)
            except Exception:
                columns = set()
        topic_columns = [column for column in ("topic_name", "topic") if column in columns]
        for column in topic_columns:
            try:
                frame = pd.read_parquet(parquet_path, columns=[column])
            except Exception:
                continue
            detected.update(_topic_values_from_frame(frame))

    for csv_path in (paths["current_csv"], paths["future_csv"]):
        if not csv_path.exists():
            continue
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception:
            continue
        topic_columns = [column for column in ("topic_name", "topic") if column in header.columns]
        for column in topic_columns:
            try:
                frame = pd.read_csv(csv_path, usecols=[column], nrows=csv_sample_rows)
            except Exception:
                continue
            detected.update(_topic_values_from_frame(frame))

    return sorted(detected)


def resolve_specsheet_topic_name(
    run_dir: str | Path,
    requested_topic: str | None,
    *,
    fallback_topic: str = DEFAULT_SPECSHEET_TOPIC,
) -> tuple[str, list[str]]:
    """Resolve the topic that should be used for specsheet generation."""
    requested = str(requested_topic or "").strip()
    detected = detect_specsheet_topic_names(run_dir)
    if requested and requested in detected:
        return requested, detected
    if fallback_topic in detected:
        return fallback_topic, detected
    if len(detected) == 1:
        return detected[0], detected
    return requested or fallback_topic, detected


def _looks_like_specsheet_release_container(path: Path) -> bool:
    return (
        (path / TREND_METADATA_FILENAME).exists()
        and any((path / role).is_dir() for role in SPECSHEET_RELEASE_ROLE_DIRS)
    )


def get_release_specsheet_context(run_dir: str | Path) -> dict[str, Any] | None:
    """Return release-folder context for specsheet workflow output, if present."""
    run_path = Path(run_dir)
    if _looks_like_specsheet_release_container(run_path):
        release_dir = run_path
    elif run_path.name in SPECSHEET_RELEASE_ROLE_DIRS and _looks_like_specsheet_release_container(run_path.parent):
        release_dir = run_path.parent
    else:
        return None

    roles: dict[str, dict[str, Path | bool]] = {}
    for role in SPECSHEET_RELEASE_ROLE_DIRS:
        role_dir = release_dir / role
        if not role_dir.is_dir():
            continue
        role_paths = get_specsheet_artifact_paths(role_dir)
        roles[role] = {
            "run_dir": role_dir,
            "metadata": role_paths["trend_metadata"],
            "summary": role_paths["trend_summary"],
            "has_metadata": role_paths["trend_metadata"].exists(),
            "has_summary": role_paths["trend_summary"].exists(),
        }

    metadata_path = release_dir / TREND_METADATA_FILENAME
    if not metadata_path.exists():
        performance_metadata = roles.get("performance", {}).get("metadata")
        if isinstance(performance_metadata, Path) and performance_metadata.exists():
            metadata_path = performance_metadata

    return {
        "release_dir": release_dir,
        "metadata": metadata_path,
        "roles": roles,
        "performance_dir": roles.get("performance", {}).get("run_dir"),
        "devops_dir": roles.get("devops", {}).get("run_dir"),
    }


def resolve_specsheet_generation_run_path(run_dir: str | Path) -> Path:
    """Use the performance child as the PDF body for release workflow folders."""
    run_path = Path(run_dir)
    context = get_release_specsheet_context(run_path)
    if context is None:
        return run_path
    performance_dir = context.get("performance_dir")
    if isinstance(performance_dir, Path):
        return performance_dir
    return run_path


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


def _usecase_devops_parquet_names() -> tuple[str, ...]:
    names: list[str] = []
    try:
        from perception_catalog_analyzer.constants import USECASE_DEVOPS_RESULT_FILENAME

        names.append(str(USECASE_DEVOPS_RESULT_FILENAME))
    except Exception:
        pass
    names.extend(["usecase_devops.parquet", "devops.parquet"])
    return tuple(dict.fromkeys(name for name in names if name))


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


_CURRENT_REQUIRED_COLUMNS = {"frame_index"}
_FUTURE_REQUIRED_COLUMNS = {"frame_index"}


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        try:
            return set(pd.read_parquet(path, columns=[]).columns)
        except Exception:
            return set()


def _csv_columns(path: Path) -> set[str]:
    try:
        return set(pd.read_csv(path, nrows=0).columns)
    except (EmptyDataError, FileNotFoundError):
        return set()
    except Exception:
        return set()


def _has_required_columns(path: Path, required_columns: set[str]) -> bool:
    if not path.exists():
        return False
    if path.suffix.lower() == ".parquet":
        columns = _parquet_columns(path)
    elif path.suffix.lower() == ".csv":
        columns = _csv_columns(path)
    else:
        columns = set()
    return required_columns.issubset(columns)


def _has_pkl_sources(run_dir: str | Path) -> bool:
    run_path = Path(run_dir)
    return any(run_path.rglob("scene_result.pkl")) or any(run_path.rglob("*.pkl.z"))


def _copy_parquet_to_csv(
    parquet_path: Path,
    csv_path: Path,
    *,
    required_columns: set[str] | None = None,
) -> Path:
    required_columns = required_columns or set()
    if required_columns and not _has_required_columns(parquet_path, required_columns):
        columns = sorted(_parquet_columns(parquet_path))
        columns_text = ", ".join(columns) if columns else "none"
        required_text = ", ".join(f"`{column}`" for column in sorted(required_columns))
        raise ValueError(
            f"Cannot convert {parquet_path.name}: missing required column(s) "
            f"{required_text} (columns: {columns_text})."
        )
    frame = pd.read_parquet(parquet_path)
    frame.to_csv(csv_path, index=False)
    return csv_path


def build_scene_dataframe_from_pkl_dir(*args, **kwargs):
    from lib.perception_catalog_io import build_scene_dataframe_from_pkl_dir as build_func

    return build_func(*args, **kwargs)


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
    # Version abbreviation: accept either key on input, always persist the library key
    # (pilot_auto_version_abbr) so the file works with both the dashboard and the library.
    version_abbr = str(
        raw.get("pilot_auto_version_abbr") or raw.get("version_abbr") or ""
    ).strip()
    if version_abbr:
        parsed["pilot_auto_version_abbr"] = version_abbr
    return parsed


def _trend_version_abbr(metadata: dict[str, Any]) -> str:
    # Prefer the library's key (pilot_auto_version_abbr); fall back to the legacy dashboard
    # key (version_abbr) for metadata files not yet migrated (see §4 migration script).
    explicit = str(
        metadata.get("pilot_auto_version_abbr") or metadata.get("version_abbr") or ""
    ).strip()
    if explicit:
        return explicit
    version = str(metadata.get("pilot_auto_version") or "").strip()
    if not version:
        return ""
    # v0.2.0 removed perception_catalog_analyzer.trend._abbreviate_version; derive a short
    # label locally by stripping the "Pilot.Auto " prefix.
    shortened = _PILOT_AUTO_PREFIX_PATTERN.sub("", version).strip() or version
    return shortened[:16]


def _infer_trend_topic(metadata: dict[str, Any], metadata_path: str | Path) -> str:
    explicit = str(metadata.get("topic_name") or "").strip()
    if explicit and explicit != DEFAULT_SPECSHEET_TOPIC:
        return explicit
    for part in reversed(Path(metadata_path).parts):
        if part.startswith("perception.") and part != DEFAULT_SPECSHEET_TOPIC:
            return part
    return DEFAULT_TREND_TOPIC


def write_trend_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> Path:
    paths = get_specsheet_artifact_paths(run_dir)
    resource_dir = paths["resource_dir"]
    metadata_path = paths["trend_metadata"]
    resource_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(metadata, fh, allow_unicode=True, sort_keys=False)
    return metadata_path


def discover_trend_metadata_files(
    root_dir: str | Path | None = None,
    *,
    include_release_specs: bool = False,
) -> list[Path]:
    base_dir = Path(root_dir) if root_dir is not None else get_data_root()
    if not base_dir.exists():
        return []

    matches: list[Path] = []
    for metadata_path in base_dir.rglob(TREND_METADATA_FILENAME):
        if not metadata_path.is_file():
            continue
        if GENERATED_TREND_HISTORY_DIRNAME in metadata_path.parts:
            continue
        in_release_spec = any(part.startswith("release_spec_") for part in metadata_path.parts)
        if not include_release_specs and in_release_spec:
            continue
        if include_release_specs and in_release_spec and metadata_path.parent.name != "resources":
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
        block_items = [block for block in blocks if isinstance(block, dict)]
        headers = [str(block.get("header") or "") for block in block_items]
        evaluation_types = [str(block.get("evaluation_type") or "") for block in block_items]
        if "full" in evaluation_types or FULL_DATASET_EVALUATION_HEADER in headers:
            return "full"
        if "usecase_devops" in evaluation_types or USECASE_DEVOPS_EVALUATION_HEADER in headers:
            return "devops"
        if (
            "usecase" in evaluation_types
            or "usecase_planning" in evaluation_types
            or any(header in USECASE_PLANNING_EVALUATION_HEADERS for header in headers)
        ):
            return "usecase"
        return "performance_blocks"
    if isinstance(summary, dict) and summary:
        return "devops"
    return "unknown"


def _unwrap_devops_summary(summary: dict[str, Any]) -> dict[str, Any]:
    devops = summary.get("DevOps") if isinstance(summary, dict) else None
    if isinstance(devops, dict):
        suite_pass_rate = devops.get("Suite pass rate")
        if isinstance(suite_pass_rate, dict):
            categorized = _category_summary_from_suite_pass_rate(suite_pass_rate)
            if categorized:
                return categorized
        return devops
    return summary


def _find_usecase_devops_parquet_near(path: str | Path) -> Path | None:
    base = Path(path)
    candidates = [base if base.is_dir() else base.parent]
    if candidates[0].name == "resources" and candidates[0].parent != candidates[0]:
        candidates.append(candidates[0].parent)
    for directory in candidates:
        for file_name in _usecase_devops_parquet_names():
            parquet_path = directory / file_name
            if parquet_path.exists():
                return parquet_path
    return None


_USECASE_DEVOPS_CATEGORY_MAPPING: dict[str, dict[str, dict[str, list[str]]]] = {
    "物体未検出 (FN)": {
        "定義済み物体に対する未検出": {
            "前方車未検知": ["DevOps_V1_FN_Object_Ahead"],
            "遠方物体未検知 (>80m)": ["DevOps_V1_FN_Distant_Object"],
            "大型物体未検知": ["DevOps_V1_Misc_Large_Object"],
            "二輪車未検知": ["DevOps_V1_FN_Bicycle_Motorcycle"],
            "歩行者未検知": ["DevOps_V1_FN_Pedestrian"],
        },
        "定義済み物体の特別シーンに対する未検出": {
            "傘を持った歩行者未検知": ["DevOps_V1_FN_Pedestrian_with_Umbrella"],
            "人のいない自転車やバイク未検知": ["DevOps_V1_FN_Riderless_Bicycle_Motorcycle"],
            "子供(75~90cm)未検知": ["DevOps_V1_FN_Small_Child"],
            "しゃがんだ歩行者未検知": ["DevOps_V1_FN_Crouching_Pedestrian"],
            "構造物に近い歩行者未検知": ["DevOps_V1_FN_Pedestrian_near_Structure"],
            "遮蔽ケース": ["DevOps_V1_Misc_Occlusion"],
        },
        "未定義物体に対する未検出": {
            "動物未検知": ["DevOps_V1_FN_Animal"],
            "落下物未検知": ["DevOps_V1_FN_Road_Debris_Fallen_Object"],
            "カラーコーン未検知": ["DevOps_V1_FN_Traffic_Cone"],
            "その他未検知": ["DevOps_V1_Misc_Other_FNs"],
        },
    },
    "物体過検出 (FP)": {
        "定義済み物体に対する誤検出": {
            "構造物を車と誤検知": [
                "DevOps_V1_Misc_Structure_Misclassified_as_Vehicle",
                "DevOps_V1_Misc_Structure_Misclassified_as_Vehicle_perception_fp",
            ],
            "構造物を歩行者と誤検知": [
                "DevOps_V1_Misc_Structure_Misclassified_as_Pedestrian",
                "DevOps_V1_Misc_Structure_Misclassified_as_Pedestrian_perception_fp",
            ],
            "構造物を自転車やバイクと誤検知": [
                "DevOps_V1_Misc_Structure_Misclassified_as_Bicycles_Motorcycles",
                "DevOps_V1_Misc_Structure_Misclassified_as_Bicycles_Motorcycles_perception_fp",
            ],
        },
        "未定義物体に対する誤検出": {
            "植栽誤検知": ["DevOps_V1_FP_Vegetation", "DevOps_V1_FP_Vegetation_perception_fp"],
            "水しぶき誤検知": [
                "DevOps_V1_FP_Water_Spray_Splash",
                "DevOps_V1_FP_Water_Spray_Splash_perception_fp",
            ],
            "雨誤検知": ["DevOps_V1_FP_Rain", "DevOps_V1_FP_Rain_perception_fp"],
            "排ガスや霧誤検知": ["DevOps_V1_FP_Exhaust_Fog", "DevOps_V1_FP_Exhaust_Fog_perception_fp"],
            "地面誤検知": ["DevOps_V1_FP_Ground", "DevOps_V1_FP_Ground_perception_fp"],
            "その他誤検知": ["DevOps_V1_Other_FPs", "DevOps_V1_Other_FPs_perception_fp"],
        },
        "ラベルミス": {
            "自転車とバイクのミスラベル": ["DevOps_V1_Misc_Mislabeled_bicycles_motorcycles"],
        },
    },
    "推定誤差": {
        "位置・姿勢推定誤差": {
            "xy位置ブレ": ["DevOps_V1_Misc_XY_Position_Jitter"],
            "yawがおかしい": ["DevOps_V1_Misc_Inaccurate_Yaw"],
        },
        "速度推定誤差": {
            "ロケット現象": ["DevOps_V1_FP_Rocket"],
        },
    },
}


def _empty_usecase_devops_result() -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    return {
        major: {
            mid: {minor: {"passed": 0, "total": 0} for minor in minors}
            for mid, minors in mids.items()
        }
        for major, mids in _USECASE_DEVOPS_CATEGORY_MAPPING.items()
    }


def _fallback_category_for_usecase_devops_suite(suite_name: str) -> tuple[str, str, str]:
    if "_FN_" in suite_name:
        major = "物体未検出 (FN)"
    elif "_FP_" in suite_name or suite_name.endswith("_perception_fp"):
        major = "物体過検出 (FP)"
    else:
        major = "推定誤差"
    label = re.sub(r"^DevOps_V\d+_", "", suite_name)
    label = re.sub(r"_perception_fp$", "", label).replace("_", " ")
    return major, "未分類", label


def _category_summary_from_suite_pass_rate(
    suite_results: dict[str, Any],
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    categorized: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    matched_suites: set[str] = set()
    for major, mids in _USECASE_DEVOPS_CATEGORY_MAPPING.items():
        for mid, minors in mids.items():
            for minor, suite_names in minors.items():
                passed = 0
                total = 0
                matched = False
                for suite_name in suite_names:
                    result = suite_results.get(str(suite_name))
                    if not isinstance(result, dict):
                        continue
                    matched = True
                    matched_suites.add(str(suite_name))
                    passed += int(result.get("passed", 0) or 0)
                    total += int(result.get("total", 0) or 0)
                if matched:
                    categorized.setdefault(major, {}).setdefault(mid, {})[minor] = {
                        "passed": passed,
                        "total": total,
                    }

    for suite_name, result in suite_results.items():
        if str(suite_name) in matched_suites or not isinstance(result, dict):
            continue
        major, mid, minor = _fallback_category_for_usecase_devops_suite(str(suite_name))
        categorized.setdefault(major, {}).setdefault(mid, {})[minor] = {
            "passed": int(result.get("passed", 0) or 0),
            "total": int(result.get("total", 0) or 0),
        }
    return categorized


def _aggregate_usecase_devops_frame(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    suite_col = "Suite Name" if "Suite Name" in frame.columns else "suite_name"
    success_col = "Success" if "Success" in frame.columns else "success"
    total_col = "Total" if "Total" in frame.columns else "total"
    if suite_col not in frame.columns or success_col not in frame.columns or total_col not in frame.columns:
        return {}

    by_suite: dict[str, tuple[int, int]] = {}
    for _, row in frame.iterrows():
        suite_name = str(row.get(suite_col) or "").strip()
        if not suite_name:
            continue
        passed, total = by_suite.get(suite_name, (0, 0))
        by_suite[suite_name] = (
            passed + int(row.get(success_col, 0) or 0),
            total + int(row.get(total_col, 0) or 0),
        )

    aggregated = _empty_usecase_devops_result()
    mapped_suites: set[str] = set()
    for major, mids in _USECASE_DEVOPS_CATEGORY_MAPPING.items():
        for mid, minors in mids.items():
            for minor, suite_names in minors.items():
                for suite_name in suite_names:
                    mapped_suites.add(suite_name)
                    passed, total = by_suite.get(suite_name, (0, 0))
                    aggregated[major][mid][minor]["passed"] += passed
                    aggregated[major][mid][minor]["total"] += total

    for suite_name, (passed, total) in by_suite.items():
        if suite_name in mapped_suites:
            continue
        major, mid, minor = _fallback_category_for_usecase_devops_suite(suite_name)
        aggregated.setdefault(major, {}).setdefault(mid, {})[minor] = {
            "passed": passed,
            "total": total,
        }

    return aggregated


def _load_usecase_devops_data_from_parquet(parquet_path: str | Path | None) -> dict[str, Any]:
    if parquet_path is None:
        return {}
    try:
        from perception_catalog_analyzer.file_io import load_usecase_devops

        data = load_usecase_devops(Path(parquet_path))
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    try:
        frame = pd.read_parquet(parquet_path)
    except Exception:
        return {}
    if frame.empty:
        return {}
    return _aggregate_usecase_devops_frame(frame)


def _devops_summary_for_metadata(metadata_path: str | Path, summary: dict[str, Any]) -> dict[str, Any]:
    parquet_path = _find_usecase_devops_parquet_near(metadata_path)
    parquet_summary = _load_usecase_devops_data_from_parquet(parquet_path)
    if parquet_summary:
        return parquet_summary
    return summary


def _release_role_key_for_metadata(role: str) -> str:
    if role in {"full", "performance_blocks"}:
        return "performance"
    return role


def _job_id_from_run_metadata(run_dir: Path, role: str) -> str:
    role_key = _release_role_key_for_metadata(role)
    candidates = [run_dir]
    if run_dir.parent != run_dir:
        candidates.append(run_dir.parent)

    for candidate in candidates:
        metadata = read_run_metadata(candidate)
        release_specsheet = metadata.get("release_specsheet") if isinstance(metadata.get("release_specsheet"), dict) else {}
        evaluator_jobs = release_specsheet.get("evaluator_jobs") if isinstance(release_specsheet.get("evaluator_jobs"), dict) else {}
        role_meta = evaluator_jobs.get(role_key) if isinstance(evaluator_jobs.get(role_key), dict) else {}
        job_id = str(role_meta.get("job_id") or "").strip()
        if job_id:
            return job_id

        evaluator_meta = metadata.get("evaluator") if isinstance(metadata.get("evaluator"), dict) else {}
        job_id = str(evaluator_meta.get("job_id") or "").strip()
        if job_id:
            return job_id

        request_meta = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
        parameter_meta = request_meta.get("parameters") if isinstance(request_meta.get("parameters"), dict) else {}
        for key in (f"{role_key}_job_id", "job_id"):
            job_id = str(parameter_meta.get(key) or request_meta.get(key) or "").strip()
            if job_id:
                return job_id
    return ""


def _release_metadata_match(candidate: dict[str, Any], target: dict[str, Any]) -> bool:
    for key in ("release_group", "pilot_auto_version", "topic_name", "description", "data_count"):
        target_value = str(target.get(key) or "").strip()
        if target_value and str(candidate.get(key) or "").strip() != target_value:
            return False
    return True


def _job_id_from_matching_release_run_metadata(root_dir: str | Path | None, target_metadata: dict[str, Any], role: str) -> str:
    root = Path(root_dir) if root_dir is not None else get_data_root()
    if not root.exists() or not root.is_dir():
        return ""
    role_key = _release_role_key_for_metadata(role)
    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for candidate in candidates:
        metadata = read_run_metadata(candidate)
        request_meta = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
        parameter_meta = request_meta.get("parameters") if isinstance(request_meta.get("parameters"), dict) else {}
        trend_metadata = (
            parameter_meta.get("trend_metadata")
            if isinstance(parameter_meta.get("trend_metadata"), dict)
            else {}
        )
        release_specsheet = metadata.get("release_specsheet") if isinstance(metadata.get("release_specsheet"), dict) else {}
        release_metadata = (
            release_specsheet.get("metadata")
            if isinstance(release_specsheet.get("metadata"), dict)
            else trend_metadata
        )
        if not _release_metadata_match(release_metadata, target_metadata):
            continue

        evaluator_jobs = release_specsheet.get("evaluator_jobs") if isinstance(release_specsheet.get("evaluator_jobs"), dict) else {}
        role_meta = evaluator_jobs.get(role_key) if isinstance(evaluator_jobs.get(role_key), dict) else {}
        job_id = str(role_meta.get("job_id") or "").strip()
        if job_id:
            return job_id

        job_id = str(parameter_meta.get(f"{role_key}_job_id") or request_meta.get(f"{role_key}_job_id") or "").strip()
        if job_id:
            return job_id
    return ""


_RELEASE_DATE_PATTERN = re.compile(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})")


def _release_date_key(text: str) -> tuple[int, int, int]:
    """Sortable (year, month, day) for the free-form dates releases record.

    Accepts ``2026.6.9``, ``2026.07.08`` and ``2026-05-18`` alike. Anything
    unparseable sorts oldest, which keeps it out of the way of a trend's recent end.
    """
    match = _RELEASE_DATE_PATTERN.search(str(text or ""))
    if not match:
        return (0, 0, 0)
    year, month, day = (int(part) for part in match.groups())
    return (year, month, day)


def discover_trend_release_groups(root_dir: str | Path | None = None) -> list[TrendReleaseGroup]:
    metadata_files = discover_trend_metadata_files(root_dir, include_release_specs=True)
    grouped: dict[str, TrendReleaseGroup] = {}
    standalone_records: list[dict[str, Any]] = []

    for metadata_path in metadata_files:
        summary_path = metadata_path.parent / TREND_SUMMARY_FILENAME
        summary = load_trend_summary_file(summary_path)
        role = classify_trend_summary(summary)
        metadata = load_trend_metadata_file(metadata_path)
        devops_summary = _devops_summary_for_metadata(metadata_path, summary) if role == "devops" else {}

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
                    "job_id": str(
                        metadata.get("job_id")
                        or _job_id_from_run_metadata(run_dir, role)
                        or _job_id_from_matching_release_run_metadata(root_dir, metadata, role)
                        or ""
                    ),
                    "metadata_path": metadata_path,
                    "summary_path": summary_path,
                    "metadata": metadata,
                    "summary": summary,
                    "devops_summary": devops_summary,
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
            "job_id": str(
                metadata.get("job_id")
                or _job_id_from_run_metadata(metadata_path.parent, role)
                or _job_id_from_matching_release_run_metadata(root_dir, metadata, role)
                or (metadata_path.parent.name if metadata_path.parent.name != "resources" else run_dir.name)
            ),
            "metadata_path": metadata_path.resolve(),
            "summary_path": summary_path.resolve(),
            "metadata": metadata,
            "summary": summary,
            "devops_summary": devops_summary,
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
                    "devops_summary": record.get("devops_summary", {}),
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
                        "devops_summary": record.get("devops_summary", {}),
                    }
                },
            )

    def _sort_key(group: TrendReleaseGroup) -> tuple[tuple[int, int, int], str, str]:
        dates = [
            str(job["metadata"].get("date") or "")
            for job in group.jobs.values()
            if isinstance(job.get("metadata"), dict)
        ]
        # Compared as parsed dates, not as strings: releases write the month both
        # padded and unpadded ("2026.07.08" and "2026.6.9"), and a string comparison
        # puts July 2026 before March 2026, so every trend chart drawn from this order
        # ran out of chronological sequence.
        newest = max(dates, key=_release_date_key) if dates else ""
        return (_release_date_key(newest), newest, group.display_name)

    return sorted(_deduplicate_trend_release_groups(grouped.values()), key=_sort_key)


def _trend_group_identity(group: TrendReleaseGroup) -> tuple[str, str, str, str, str, str, tuple[str, ...]]:
    metadata = {}
    for role in ("full", "usecase", "devops", "performance_blocks", "unknown"):
        if role in group.jobs:
            metadata = group.jobs[role].get("metadata", {})
            break
    return (
        str(metadata.get("release_group") or ""),
        str(group.topic_name or ""),
        str(metadata.get("pilot_auto_version") or ""),
        str(metadata.get("date") or ""),
        str(metadata.get("description") or ""),
        str(metadata.get("data_count") or ""),
        tuple(sorted(group.jobs.keys())),
    )


def _trend_group_preference(group: TrendReleaseGroup) -> tuple[int, int, str]:
    generated_history = any(
        GENERATED_TREND_HISTORY_DIRNAME in Path(job.get("metadata_path", "")).parts
        for job in group.jobs.values()
    )
    return (
        0 if generated_history else 1,
        len(group.jobs),
        str(group.base_dir),
    )


def _deduplicate_trend_release_groups(groups: Iterable[TrendReleaseGroup]) -> list[TrendReleaseGroup]:
    selected: dict[tuple[str, str, str, str, str, str, tuple[str, ...]], TrendReleaseGroup] = {}
    for group in groups:
        identity = _trend_group_identity(group)
        current = selected.get(identity)
        if current is None or _trend_group_preference(group) > _trend_group_preference(current):
            selected[identity] = group
    return list(selected.values())


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


def _extract_usecase_metric_tables(summary: dict[str, Any]) -> list[dict[str, Any]]:
    data_list: list[dict[str, Any]] = []
    blocks = summary.get("blocks", [])
    if not isinstance(blocks, list):
        return data_list
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("header") not in USECASE_PLANNING_EVALUATION_HEADERS:
            continue
        if block.get("mode") not in (None, "metrics"):
            continue
        if block.get("evaluation_type") not in (None, "usecase", "usecase_planning"):
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
    return _average_metric_tables(data_list)


def _average_metric_tables(data_list: list[dict[str, Any]]) -> dict[str, float]:
    def _avg(metric_name: str) -> float:
        raw_values: list[Any] = []
        for metrics in data_list:
            values = metrics.get(metric_name, {})
            if isinstance(values, dict):
                raw_values.extend(values.values())
        if not raw_values:
            return float("nan")
        numeric = pd.to_numeric(pd.Series(raw_values), errors="coerce")
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


def extract_usecase_metrics_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    """Return averaged UseCase planning metrics from a usecase summary payload."""
    data_list = _extract_usecase_metric_tables(summary)
    if not data_list:
        return {}
    return _average_metric_tables(data_list)


def _with_unique_trend_version_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identities_by_label: dict[str, list[tuple[str, str, str]]] = {}
    for row in rows:
        label = str(row.get("version_abbr") or "")
        identity = (
            str(row.get("release_group") or ""),
            str(row.get("date") or ""),
            str(row.get("description") or ""),
        )
        identities = identities_by_label.setdefault(label, [])
        if identity not in identities:
            identities.append(identity)

    suffix_by_identity: dict[tuple[str, tuple[str, str, str]], int] = {}
    for label, identities in identities_by_label.items():
        if len(identities) <= 1:
            continue
        for idx, identity in enumerate(identities, start=1):
            suffix_by_identity[(label, identity)] = idx

    labeled: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        updated = dict(row)
        label = str(updated.get("version_abbr") or "")
        identity = (
            str(updated.get("release_group") or ""),
            str(updated.get("date") or ""),
            str(updated.get("description") or ""),
        )
        suffix = suffix_by_identity.get((label, identity))
        if suffix is not None:
            updated["version_abbr"] = f"{label} #{suffix}"

        dedupe_key = (
            str(updated.get("version_abbr") or ""),
            str(updated.get("topic") or ""),
            str(updated.get("evaluation_type") or ""),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        labeled.append(updated)
    return labeled


def extract_devops_case_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten nested devops/pass-rate summary into case rows."""
    summary = _unwrap_devops_summary(summary)
    rows: list[dict[str, Any]] = []
    for major_category, mid_categories in summary.items():
        if not isinstance(mid_categories, dict):
            continue
        for mid_category, minor_or_cases in mid_categories.items():
            if not isinstance(minor_or_cases, dict):
                continue
            if {"passed", "total"}.intersection(minor_or_cases.keys()):
                passed = int(minor_or_cases.get("passed", 0) or 0)
                total = int(minor_or_cases.get("total", 0) or 0)
                rows.append(
                    {
                        "major_category": major_category,
                        "mid_category": mid_category,
                        "minor_category": mid_category,
                        "case_name": mid_category,
                        "passed": passed,
                        "total": total,
                        "pass_rate": (passed / total * 100.0) if total > 0 else None,
                    }
                )
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


def _normalize_devops_summary_structure(summary: dict[str, Any]) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    summary = _unwrap_devops_summary(summary)
    normalized: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for major_category, mid_categories in summary.items():
        if not isinstance(mid_categories, dict):
            continue
        normalized_major = normalized.setdefault(str(major_category), {})
        for mid_category, minor_or_cases in mid_categories.items():
            if not isinstance(minor_or_cases, dict):
                continue
            normalized_mid = normalized_major.setdefault(str(mid_category), {})
            if {"passed", "total"}.intersection(minor_or_cases.keys()):
                normalized_mid[str(mid_category)] = {
                    "passed": int(minor_or_cases.get("passed", 0) or 0),
                    "total": int(minor_or_cases.get("total", 0) or 0),
                }
                continue
            for case_name, result in minor_or_cases.items():
                if not isinstance(result, dict):
                    continue
                normalized_mid[str(case_name)] = {
                    "passed": int(result.get("passed", 0) or 0),
                    "total": int(result.get("total", 0) or 0),
                }
    return normalized


def _align_devops_trend_data_structures(trend_data_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    structure: dict[str, dict[str, set[str]]] = {}
    for row in trend_data_rows:
        devops_data = _normalize_devops_summary_structure(row.get("devops_data", {}))
        row["devops_data"] = devops_data
        for major_category, mid_categories in devops_data.items():
            major_structure = structure.setdefault(major_category, {})
            for mid_category, cases in mid_categories.items():
                major_structure.setdefault(mid_category, set()).update(cases.keys())

    for row in trend_data_rows:
        devops_data = row.get("devops_data", {})
        if not isinstance(devops_data, dict):
            devops_data = {}
            row["devops_data"] = devops_data
        for major_category, mid_categories in structure.items():
            row_major = devops_data.setdefault(major_category, {})
            for mid_category, cases in mid_categories.items():
                row_mid = row_major.setdefault(mid_category, {})
                for case_name in cases:
                    row_mid.setdefault(case_name, {"passed": 0, "total": 0})
    return trend_data_rows


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
                "version_abbr": _trend_version_abbr(metadata),
                "release_group": metadata.get("release_group"),
                "data_count": metadata.get("data_count"),
                "description": metadata.get("description"),
                "date": metadata.get("date"),
                "topic": _infer_trend_topic(metadata, metadata_path),
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
                "version_abbr": row.get("version_abbr"),
                "release_group": row.get("release_group"),
                "data_count": row.get("data_count"),
                "description": row.get("description"),
                "date": row.get("date"),
                "topic": row.get("topic"),
                "evaluation_type": "full",
                "mAP": _avg("mAP"),
                "precision": _avg("precision"),
                "recall": _avg("recall"),
                "minADE@1s": _avg("minADE@1s"),
                "minFDE@1s": _avg("minFDE@1s"),
                "minADE@3s": _avg("minADE@3s"),
                "minFDE@3s": _avg("minFDE@3s"),
                "minADE@5s": _avg("minADE@5s"),
                "minFDE@5s": _avg("minFDE@5s"),
            }
        )
    return _with_unique_trend_version_labels(output)


def banded_recall_percent_from_summary(summary: Any) -> dict[str, float]:
    """Return ``{distance_band: recall %}`` from a summary, or ``{}`` when banded recall is
    absent (all pre-0.2.0 releases).

    Uses perception_catalog_analyzer's own helpers so the band definitions stay in lock-step
    with the library. Accepts either a summary dict (with a ``blocks`` list) or a raw blocks
    list. Non-finite values are dropped so callers can treat a non-empty result as "has real
    banded recall" (see §7 graceful fallback).
    """
    try:
        from perception_catalog_analyzer.trend import recall_by_subsection_from_summary
        from perception_catalog_analyzer.file_io import (
            overall_recall_by_band_percent_from_subsections,
        )
    except Exception:
        return {}

    if isinstance(summary, dict):
        blocks = summary.get("blocks", [])
    elif isinstance(summary, list):
        blocks = summary
    else:
        blocks = []
    if not isinstance(blocks, list):
        return {}

    try:
        recall_by_subsection = recall_by_subsection_from_summary(blocks)
        bands = overall_recall_by_band_percent_from_subsections(recall_by_subsection)
    except Exception:
        return {}

    return {
        str(band): float(value)
        for band, value in (bands or {}).items()
        if isinstance(value, (int, float)) and value == value  # finite (drop NaN)
    }


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

        devops_summary = _devops_summary_for_metadata(metadata_path, summary)
        rows = extract_devops_case_rows(devops_summary)
        if not rows:
            continue
        normalized_summary = _normalize_devops_summary_structure(devops_summary)
        overall_passed = sum(int(row["passed"]) for row in rows)
        overall_total = sum(int(row["total"]) for row in rows)
        trend_data_rows.append(
            {
                "version": metadata.get("pilot_auto_version"),
                "version_abbr": _trend_version_abbr(metadata),
                "data_count": metadata.get("data_count"),
                "description": metadata.get("description"),
                "date": metadata.get("date"),
                "topic": _infer_trend_topic(metadata, metadata_path),
                "overall_pass_rate": (overall_passed / overall_total * 100.0)
                if overall_total > 0
                else 0.0,
                "scenario_count": overall_total,
                "devops_data": normalized_summary,
                "usecase_devops_data": normalized_summary,
                # §7: banded recall when the summary carries it (0.2.0+); {} otherwise.
                "recall_by_band": banded_recall_percent_from_summary(summary),
            }
        )

    trend_data_rows.sort(key=lambda row: _trend_version_sort_key(str(row.get("version") or "")))
    return _align_devops_trend_data_structures(trend_data_rows)


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


def _devops_trend_rows_for_template(devops_trend_data: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in devops_trend_data:
        display_row = dict(row)
        version_abbr = str(display_row.get("version_abbr") or "").strip()
        if version_abbr:
            display_row["version"] = version_abbr
        rows.append(display_row)
    return rows


def _recall_ratio_to_percent(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    numeric = float(numeric)
    return numeric * 100.0 if -1.0 <= numeric <= 1.0 else numeric


def _build_trend_context(
    metadata_list: Sequence[Path],
    output_dir: Path,
    current_devops_summary_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if not metadata_list:
        return {
            "performance_trend_data": [],
            "map_trend_plot_path": output_dir / "map_trend.png",
            "prediction_trend_plot_path": output_dir / "prediction_trend.png",
            "devops_data": {},
            "devops_plot_path": None,
            "usecase_devops_data": {},
            "usecase_devops_plot_path": None,
            "devops_trend_data": [],
            "devops_trend_plot_path": output_dir / "usecase_devops_trend.png",
            "usecase_devops_trend_data": [],
            "usecase_devops_trend_plot_path": output_dir / "usecase_devops_trend.png",
            "job_ids": [],
        }

    try:
        from perception_catalog_analyzer.plot.map_trend import generate_map_trend_plot
        from perception_catalog_analyzer.plot.prediction_trend import generate_prediction_trend_plot
        try:
            from perception_catalog_analyzer.plot.usecase_devops_trend import (
                generate_usecase_devops_trend_detail_plot as generate_devops_trend_detail_plot,
                generate_usecase_devops_trend_plot as generate_devops_trend_plot,
            )
            from perception_catalog_analyzer.plot.usecase_devops import (
                generate_usecase_devops_plot as generate_devops_plot,
            )
        except ImportError:
            from perception_catalog_analyzer.plot.devops_trend import (
                generate_devops_trend_detail_plot,
                generate_devops_trend_plot,
            )
            from perception_catalog_analyzer.plot.devops import generate_devops_plot
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
    recall_by_version = {
        str(row.get("version") or ""): _recall_ratio_to_percent(row.get("recall"))
        for row in performance_trend_data
        if str(row.get("version") or "")
    }
    for row in devops_trend_data:
        row.setdefault("recall", recall_by_version.get(str(row.get("version") or ""), float("nan")))

    devops_trend_plot_path = output_dir / "usecase_devops_trend.png"
    devops_data = {}
    devops_plot_path = None
    if current_devops_summary_path is not None and current_devops_summary_path.exists():
        current_devops_summary = load_trend_summary_file(current_devops_summary_path)
        if classify_trend_summary(current_devops_summary) == "devops":
            devops_summary = _devops_summary_for_metadata(current_devops_summary_path, current_devops_summary)
            devops_data = _normalize_devops_summary_structure(devops_summary)
            if devops_data:
                _notify(progress_callback, "Rendering current pass-rate plot")
                devops_plot_path = output_dir / "usecase_devops.png"
                _generate_usecase_devops_plot_compat(
                    generate_devops_plot,
                    devops_data=devops_data,
                    devops_summary_blocks=current_devops_summary,
                    path=devops_plot_path,
                )
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
        "devops_data": devops_data,
        "devops_plot_path": devops_plot_path,
        "usecase_devops_data": devops_data,
        "usecase_devops_plot_path": devops_plot_path,
        "devops_trend_data": _devops_trend_rows_for_template(devops_trend_data),
        "devops_trend_plot_path": devops_trend_plot_path,
        "usecase_devops_trend_data": _devops_trend_rows_for_template(devops_trend_data),
        "usecase_devops_trend_plot_path": devops_trend_plot_path,
        "job_ids": [],
    }


def _generate_usecase_devops_plot_compat(
    generate_func: Callable[..., None],
    *,
    devops_data: dict[str, Any],
    devops_summary_blocks: Any,
    path: Path,
) -> None:
    """Call ``generate_usecase_devops_plot`` across analyzer versions.

    v0.1.0 signature was ``(data, path)`` where ``data`` is the normalized 大/中/小
    pass/total tree. v0.2.0 rewrote it to
    ``(recall_by_subsection, category_mapping, path, *, pass_by_subsection, fp_subcats)``
    so the plot shows banded recall (and pass-rate only for FP subcategories).
    """
    try:
        parameters = inspect.signature(generate_func).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "category_mapping" not in parameters:
        # Legacy 0.1.0 signature.
        generate_func(devops_data, path)
        return

    # New 0.2.0 signature: assemble recall_by_subsection + category_mapping (+ pass_by_subsection)
    # using the library's own helpers so we don't reimplement its category/recall logic.
    try:
        from perception_catalog_analyzer.file_io import load_yaml
        from perception_catalog_analyzer.trend import recall_by_subsection_from_summary

        try:
            from perception_catalog_analyzer.constants import USECASE_DEVOPS_MAPPING_PATH
        except ImportError:  # older layouts exposed it via path
            from perception_catalog_analyzer.path import (
                DEVOPS_MAPPING_PATH as USECASE_DEVOPS_MAPPING_PATH,
            )
    except ImportError:
        # Cannot build new-style args; skip the plot rather than crash PDF generation.
        return

    category_mapping = load_yaml(USECASE_DEVOPS_MAPPING_PATH)
    try:
        recall_by_subsection = recall_by_subsection_from_summary(devops_summary_blocks)
    except Exception:
        recall_by_subsection = {}

    # Flatten the normalized 大/中/小 pass/total tree to {小: {passed, total}} for FP subcats.
    pass_by_subsection: dict[str, dict[str, int]] = {}
    for mid_categories in (devops_data or {}).values():
        if not isinstance(mid_categories, dict):
            continue
        for subcats in mid_categories.values():
            if not isinstance(subcats, dict):
                continue
            for subcat, vals in subcats.items():
                if isinstance(vals, dict) and {"passed", "total"} <= set(vals):
                    pass_by_subsection[str(subcat)] = {
                        "passed": int(vals.get("passed", 0) or 0),
                        "total": int(vals.get("total", 0) or 0),
                    }

    kwargs: dict[str, Any] = {}
    if "pass_by_subsection" in parameters:
        kwargs["pass_by_subsection"] = pass_by_subsection
    generate_func(recall_by_subsection, category_mapping, path, **kwargs)


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
    path_manager = SimpleNamespace(specsheet_path=context_dir)
    semantic_kwargs = {
        "project_id": project_id,
        "pilot_auto_version": version,
        "version": version,
        "devops_data": trend_context.get("devops_data", {}),
        "devops_plot_path": trend_context.get("devops_plot_path"),
        "usecase_devops_data": trend_context.get(
            "usecase_devops_data", trend_context.get("devops_data", {})
        ),
        "usecase_devops_plot_path": trend_context.get(
            "usecase_devops_plot_path", trend_context.get("devops_plot_path")
        ),
        "trend_data": trend_context.get("performance_trend_data", []),
        "performance_trend_data": trend_context.get("performance_trend_data", []),
        "map_trend_plot_path": trend_context.get("map_trend_plot_path", context_dir / "map_trend.png"),
        "prediction_trend_plot_path": trend_context.get(
            "prediction_trend_plot_path", context_dir / "prediction_trend.png"
        ),
        "devops_trend_data": trend_context.get("devops_trend_data", []),
        "devops_trend_plot_path": trend_context.get(
            "devops_trend_plot_path", context_dir / "usecase_devops_trend.png"
        ),
        "usecase_devops_trend_data": trend_context.get(
            "usecase_devops_trend_data", trend_context.get("devops_trend_data", [])
        ),
        "usecase_devops_trend_plot_path": trend_context.get(
            "usecase_devops_trend_plot_path",
            trend_context.get("devops_trend_plot_path", context_dir / "usecase_devops_trend.png"),
        ),
        "job_ids": trend_context.get("job_ids", []),
        "template_name": "static_body.html",
        "extensions": ["html"],
        "template_dir": str(template_dir),
        "path_manager": path_manager,
        "show_other_infos": bool(trend_context.get("performance_trend_data")),
    }

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    if accepts_kwargs or not parameters:
        with _patch_template_dataset_paths(update_template_func, context_dir):
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
    with _patch_template_dataset_paths(update_template_func, context_dir):
        return update_template_func(*args, **kwargs)


@contextmanager
def _patch_template_dataset_paths(
    update_template_func: Callable[..., Sequence[str]],
    context_dir: Path,
):
    """Redirect analyzer dataset-summary outputs away from read-only package config."""
    globals_dict = getattr(update_template_func, "__globals__", {})
    patch_keys = ("DATASET_SUMMARY_PATH", "DATASET_TRAIN_PATH", "DATASET_TEST_PATH")
    originals = {key: globals_dict.get(key) for key in patch_keys if key in globals_dict}
    if not originals:
        yield
        return

    dataset_dir = context_dir / "dataset_assets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        for key, original_path in originals.items():
            if not isinstance(original_path, Path) or not original_path.exists():
                continue
            target_path = dataset_dir / original_path.name
            if not target_path.exists():
                shutil.copy2(original_path, target_path)
            globals_dict[key] = target_path
        yield
    finally:
        for key, original_path in originals.items():
            globals_dict[key] = original_path

# Columns that perception_catalog_analyzer >=0.2.0 hardcodes in SceneDataFrame.from_dir
# (select_columns / topic filter / suite exclude). Older (0.1.0-era) stored parquet may lack
# them, which makes from_dir raise ColumnNotFound. We backfill them with no-op defaults so old
# parquet keeps loading: bounding_box is never dropped by the polygon filter and an empty
# suite_name never matches fp_suite_names(). {col: DuckDB default expression}.
_SCENE_PARQUET_BACKFILL_DEFAULTS: dict[str, str] = {
    "pair_uuid": "CAST(NULL AS VARCHAR)",
    "shape_type": "CAST('bounding_box' AS VARCHAR)",
    "suite_name": "CAST('' AS VARCHAR)",
    "scenario_name": "CAST('' AS VARCHAR)",
}


def _backfill_one_scene_parquet(parquet_path: Path, topic_name: str) -> bool:
    """Add missing analyzer-required columns to a single scene parquet in place.

    Returns True if the file was rewritten. Best-effort: any failure is left to the caller.
    """
    import duckdb

    con = duckdb.connect()
    try:
        existing = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(parquet_path)]
            ).fetchall()
        }
        additions: list[str] = []
        for col, default_expr in _SCENE_PARQUET_BACKFILL_DEFAULTS.items():
            if col not in existing:
                additions.append(f"{default_expr} AS {col}")
        # topic_name is a filter column in from_dir; fill with the resolved topic so all rows match.
        if "topic_name" not in existing:
            safe_topic = topic_name.replace("'", "''")
            additions.append(f"CAST('{safe_topic}' AS VARCHAR) AS topic_name")
        if not additions:
            return False

        safe_src = str(parquet_path).replace("'", "''")
        tmp_path = parquet_path.with_name(parquet_path.name + ".backfill.tmp")
        safe_tmp = str(tmp_path).replace("'", "''")
        select_cols = "*, " + ", ".join(additions)
        con.execute(
            f"COPY (SELECT {select_cols} FROM read_parquet('{safe_src}')) "
            f"TO '{safe_tmp}' (FORMAT PARQUET)"
        )
        os.replace(tmp_path, parquet_path)
        return True
    finally:
        con.close()


def _backfill_scene_parquet_columns(
    run_path: Path,
    topic_name: str,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Backfill analyzer-required columns on stored current/future parquet (see §5a).

    No-op when the columns are already present (freshly-generated 0.2.0 parquet), so this is
    safe to call unconditionally before from_dir.
    """
    paths = get_specsheet_artifact_paths(run_path)
    for key in ("current_parquet", "future_parquet"):
        parquet_path = paths.get(key)
        if not isinstance(parquet_path, Path) or not parquet_path.exists():
            continue
        try:
            if _backfill_one_scene_parquet(parquet_path, topic_name):
                _notify(
                    progress_callback,
                    f"Backfilled analyzer columns into {parquet_path.name} for 0.2.0 compatibility",
                )
        except Exception:
            # Leave it to from_dir; the CSV fallback in _load_scene_dataframe_for_specsheet
            # still covers genuinely incompatible files.
            pass


def _scene_dataframe_from_dir_compat(
    scene_dataframe_cls,
    run_path: Path,
    *,
    topic_name: str,
    is_exclude_polygons: bool = False,
):
    """Call SceneDataFrame.from_dir across analyzer versions with/without topic.

    ``is_exclude_polygons`` is forwarded only when from_dir accepts it (analyzer >=0.2.0);
    older versions ignore it.
    """
    from_dir = scene_dataframe_cls.from_dir
    try:
        parameters = inspect.signature(from_dir).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_varargs = any(
        param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for param in parameters.values()
    )
    extra_kwargs: dict[str, Any] = {}
    if is_exclude_polygons and ("is_exclude_polygons" in parameters or accepts_varargs):
        extra_kwargs["is_exclude_polygons"] = True

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

    if accepts_varargs or len(required_parameters) >= 2:
        return from_dir(run_path, topic_name, **extra_kwargs)
    return from_dir(run_path, **extra_kwargs)


def _load_scene_dataframe_for_specsheet(
    scene_dataframe_cls,
    run_path: Path,
    *,
    topic_name: str,
    is_exclude_polygons: bool = False,
    progress_callback: Callable[[str], None] | None = None,
):
    # Ensure stored parquet has the columns analyzer >=0.2.0 requires (§5a). No-op for new data.
    _backfill_scene_parquet_columns(run_path, topic_name, progress_callback)
    try:
        return _scene_dataframe_from_dir_compat(
            scene_dataframe_cls,
            run_path,
            topic_name=topic_name,
            is_exclude_polygons=is_exclude_polygons,
        )
    except Exception as exc:
        paths = get_specsheet_artifact_paths(run_path)
        if not _has_required_columns(paths["current_parquet"], _CURRENT_REQUIRED_COLUMNS):
            raise
        if _has_required_columns(paths["current_csv"], _CURRENT_REQUIRED_COLUMNS):
            raise
        _notify(
            progress_callback,
            "Parquet load failed. Converting to CSV for analyzer compatibility",
        )
        ensure_specsheet_csvs(run_path, progress_callback=progress_callback)
        try:
            return _scene_dataframe_from_dir_compat(
                scene_dataframe_cls,
                run_path,
                topic_name=topic_name,
                is_exclude_polygons=is_exclude_polygons,
            )
        except Exception:
            raise exc


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
    "frame_index",
}


def _coerce_numeric_columns(frame: Any, columns: set[str]) -> Any:
    if _is_polars_frame(frame):
        return _coerce_polars_numeric_columns(frame, columns)
    if frame.empty:
        return frame
    coerced = frame.copy()
    for column in sorted(columns.intersection(coerced.columns)):
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def _is_polars_frame(frame: Any) -> bool:
    return frame.__class__.__module__.startswith("polars.")


def _polars_column_names(frame: Any) -> set[str]:
    collect_schema = getattr(frame, "collect_schema", None)
    if callable(collect_schema):
        try:
            return set(collect_schema().names())
        except Exception:
            pass
    return set(getattr(frame, "columns", []) or [])


def _coerce_polars_numeric_columns(frame: Any, columns: set[str]) -> Any:
    import polars as pl

    available_columns = columns.intersection(_polars_column_names(frame))
    if not available_columns:
        return frame
    expressions = []
    for column in sorted(available_columns):
        dtype = pl.Int64 if column == "frame_index" else pl.Float64
        expressions.append(pl.col(column).cast(dtype, strict=False).alias(column))
    return frame.with_columns(expressions)


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


def _coerce_analyzer_evaluation_type(evaluation_type: str):
    """Normalize analyzer evaluation type strings for versions that expect enums."""
    if hasattr(evaluation_type, "value"):
        return evaluation_type
    try:
        from perception_catalog_analyzer.specsheet.blocks import EvaluationType

        return EvaluationType(evaluation_type)
    except Exception:
        return evaluation_type


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
    parquet_compression = "snappy"
    try:
        from perception_catalog_analyzer.types import ParquetCompression

        parquet_compression = ParquetCompression.SNAPPY
    except Exception:
        pass

    analyzer_evaluation_type = _coerce_analyzer_evaluation_type(evaluation_type)
    semantic_kwargs = {
        "df": df,
        "scene_data_frame": df,
        "labels": list(labels),
        "metrics": list(metrics),
        "resource_path": outdir,
        "html_path": outdir.parent if outdir.name == "resources" else outdir,
        "parquet_compression": parquet_compression,
        "topic_name": topic_name,
        "topic": topic_name,
        "path": outdir,
        "outdir": outdir,
        "evaluation_type": analyzer_evaluation_type,
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
    path_manager = SimpleNamespace(specsheet_path=outdir)
    semantic_kwargs = {
        "html": list(html),
        "abstract_html": list(abstract_html),
        "detailed_html": list(detailed_html),
        "path_manager": path_manager,
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


def _build_current_csv_from_available_sources(
    run_dir: str | Path,
    paths: dict[str, Path],
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    current_csv = paths["current_csv"]
    fallback = next(
        (
            path
            for path in list_specsheet_source_parquets(run_dir)
            if _has_required_columns(path, _CURRENT_REQUIRED_COLUMNS)
        ),
        None,
    )
    if fallback is not None:
        _notify(progress_callback, f"Converting {fallback.name} -> {current_csv.name}")
        _copy_parquet_to_csv(
            fallback,
            current_csv,
            required_columns=_CURRENT_REQUIRED_COLUMNS,
        )
    elif not _has_pkl_sources(run_dir):
        source_names = ", ".join(path.name for path in list_specsheet_source_parquets(run_dir))
        raise ValueError(
            "No usable current specsheet data found. Existing parquet file(s) "
            f"do not include required column `frame_index`: {source_names or 'none'}. "
            "Re-download or regenerate the run with evaluator result data."
        )

    if not current_csv.exists() or not _has_required_columns(current_csv, _CURRENT_REQUIRED_COLUMNS):
        _notify(progress_callback, "No valid CSV found. Building CSV from pkl / pkl.z files")
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
        if not current_csv.exists() or not _has_required_columns(current_csv, _CURRENT_REQUIRED_COLUMNS):
            raise FileNotFoundError(
                f"Failed to generate usable {current_csv}; required column `frame_index` is missing."
            )


def ensure_specsheet_inputs(
    run_dir: str | Path,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Path | None]:
    """Ensure the analyzer has loadable specsheet inputs, preferring parquet.

    Modern perception_catalog_analyzer versions load current.parquet and
    future.parquet directly. CSV is still accepted when it is the only available
    source, and generated only when there is no named current parquet/csv input.
    """
    paths = get_specsheet_artifact_paths(run_dir)
    current_csv = paths["current_csv"]
    future_csv = paths["future_csv"]
    current_parquet = paths["current_parquet"]
    future_parquet = paths["future_parquet"]

    current_path: Path | None = None
    future_path: Path | None = None

    if _has_required_columns(current_parquet, _CURRENT_REQUIRED_COLUMNS):
        current_path = current_parquet
        _notify(progress_callback, f"Using {current_parquet.name} directly")
    elif _has_required_columns(current_csv, _CURRENT_REQUIRED_COLUMNS):
        current_path = current_csv
        _notify(progress_callback, f"Using {current_csv.name}")
    else:
        _build_current_csv_from_available_sources(
            run_dir,
            paths,
            progress_callback=progress_callback,
        )
        current_path = current_csv if current_csv.exists() else None

    if future_parquet.exists() and _has_required_columns(future_parquet, _FUTURE_REQUIRED_COLUMNS):
        future_path = future_parquet
    elif future_csv.exists() and _has_required_columns(future_csv, _FUTURE_REQUIRED_COLUMNS):
        future_path = future_csv

    return {
        "current": current_path,
        "future": future_path,
        "current_csv": current_csv if current_csv.exists() else None,
        "future_csv": future_csv if future_csv.exists() else None,
        "current_parquet": current_parquet if current_parquet.exists() else None,
        "future_parquet": future_parquet if future_parquet.exists() else None,
    }


def ensure_specsheet_csvs(
    run_dir: str | Path,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Path | None]:
    """Ensure current.csv/future.csv exist for older CSV-only analyzer paths."""
    paths = get_specsheet_artifact_paths(run_dir)
    current_csv = paths["current_csv"]
    future_csv = paths["future_csv"]
    current_parquet = paths["current_parquet"]
    future_parquet = paths["future_parquet"]

    if not _has_required_columns(current_csv, _CURRENT_REQUIRED_COLUMNS):
        if _has_required_columns(current_parquet, _CURRENT_REQUIRED_COLUMNS):
            _notify(progress_callback, f"Converting {current_parquet.name} -> {current_csv.name}")
            _copy_parquet_to_csv(
                current_parquet,
                current_csv,
                required_columns=_CURRENT_REQUIRED_COLUMNS,
            )
        else:
            _build_current_csv_from_available_sources(
                run_dir,
                paths,
                progress_callback=progress_callback,
            )

    if not future_csv.exists() and future_parquet.exists():
        _notify(progress_callback, f"Converting {future_parquet.name} -> {future_csv.name}")
        _copy_parquet_to_csv(
            future_parquet,
            future_csv,
            required_columns=_FUTURE_REQUIRED_COLUMNS,
        )

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
    trend_metadata_paths: Sequence[str | Path] | None = None,
    force: bool = False,
    is_exclude_polygons: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Path, bool]:
    paths = get_specsheet_artifact_paths(run_dir)
    specsheet_dir = paths["specsheet_dir"]
    pdf_path = paths["specsheet_pdf"]

    if not force and is_specsheet_pdf_fresh(run_dir):
        _notify(progress_callback, "Using existing up-to-date spec-sheet PDF")
        return pdf_path, False

    ensure_specsheet_inputs(run_dir, progress_callback=progress_callback)
    resolved_topic, detected_topics = resolve_specsheet_topic_name(run_dir, topic_name)
    if resolved_topic != topic_name:
        detected_text = ", ".join(detected_topics) if detected_topics else "none"
        _notify(
            progress_callback,
            f"Using detected topic {resolved_topic} instead of requested topic {topic_name} (detected: {detected_text})",
        )
        topic_name = resolved_topic

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
    block_resource_dir = specsheet_dir / "resources"
    block_resource_dir.mkdir(parents=True, exist_ok=True)
    trend_asset_dir = specsheet_dir / "trend_assets"
    trend_asset_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, "Loading specsheet data")
    df = _load_scene_dataframe_for_specsheet(
        SceneDataFrame,
        run_path,
        topic_name=topic_name,
        is_exclude_polygons=is_exclude_polygons,
        progress_callback=progress_callback,
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
            outdir=block_resource_dir.resolve(),
            evaluation_type="full",
        )

    trend_context: dict[str, object] | None = None
    if include_trend:
        if trend_metadata is None:
            raise ValueError("Trend metadata is required when trend mode is enabled.")
        _notify(progress_callback, "Validating full trend summary")
        generated_trend_summary = block_resource_dir / TREND_SUMMARY_FILENAME
        trend_summary_path = generated_trend_summary if generated_trend_summary.exists() else paths["trend_summary"]
        ensure_full_trend_summary(trend_summary_path)
        if generated_trend_summary.exists() and not paths["trend_summary"].exists():
            shutil.copy2(generated_trend_summary, paths["trend_summary"])
        _notify(progress_callback, "Saving trend metadata")
        write_trend_metadata(run_path, trend_metadata)
        release_context = get_release_specsheet_context(run_path)
        if trend_metadata_paths is None:
            metadata_list = discover_trend_metadata_files()
        else:
            metadata_list = [
                Path(metadata_path)
                for metadata_path in trend_metadata_paths
                if Path(metadata_path).exists()
            ]
        metadata_list.append(paths["trend_metadata"])
        if release_context is not None:
            roles = release_context.get("roles", {})
            if isinstance(roles, dict):
                for role_info in roles.values():
                    if not isinstance(role_info, dict):
                        continue
                    metadata_path = role_info.get("metadata")
                    if isinstance(metadata_path, Path) and metadata_path.exists():
                        metadata_list.append(metadata_path)
        metadata_list = sorted(
            dict.fromkeys(path.resolve() for path in metadata_list if path.exists()),
            key=lambda path: str(path),
        )
        current_devops_summary_path = None
        if release_context is not None:
            roles = release_context.get("roles", {})
            if isinstance(roles, dict):
                devops_info = roles.get("devops", {})
                if isinstance(devops_info, dict):
                    summary_path = devops_info.get("summary")
                    if isinstance(summary_path, Path):
                        current_devops_summary_path = summary_path
        trend_context = _build_trend_context(
            metadata_list,
            trend_asset_dir,
            current_devops_summary_path=current_devops_summary_path,
            progress_callback=progress_callback,
        )

    _notify(progress_callback, "Rendering PDF")
    for stale_output in (specsheet_dir / "specsheet.html", pdf_path):
        if stale_output.exists() and not os.access(stale_output, os.W_OK):
            stale_output.unlink()
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
        paths["current_parquet"],
        paths["current_csv"],
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
