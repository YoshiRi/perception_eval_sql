from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from lib.perception_catalog_io import build_scene_dataframe_from_pkl_dir

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


def get_specsheet_artifact_paths(run_dir: str | Path) -> dict[str, Path]:
    run_path = Path(run_dir)
    return {
        "run_dir": run_path,
        "current_csv": run_path / "current.csv",
        "future_csv": run_path / "future.csv",
        "current_parquet": run_path / "current.parquet",
        "future_parquet": run_path / "future.parquet",
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


def _copy_parquet_to_csv(parquet_path: Path, csv_path: Path) -> Path:
    frame = pd.read_parquet(parquet_path)
    frame.to_csv(csv_path, index=False)
    return csv_path


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

            def _on_progress(done: int, total: int) -> None:
                _notify(progress_callback, f"Processing pkl files {done}/{total}")

            df = build_scene_dataframe_from_pkl_dir(run_dir, on_progress=_on_progress)
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
    df = SceneDataFrame.from_dir(run_path)
    metrics = list(DEFAULT_SPECSHEET_METRICS)
    if getattr(df, "future", None) is not None:
        metrics.extend(FUTURE_SPECSHEET_METRICS)

    _notify(progress_callback, "Building abstract and detail sections")
    abstract, detailed = get_blocks(
        df=df,
        labels=list(labels),
        metrics=metrics,
        topic_name=topic_name,
        outdir=resource_dir.resolve(),
        evaluation_type="full",
    )

    _notify(progress_callback, "Rendering PDF")
    template_dir = Path(template_module.__file__).resolve().parent.parent / "template"
    html = update_template(project_id, version, template_dir=str(template_dir))
    specsheet(
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
