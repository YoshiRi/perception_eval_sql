from __future__ import annotations

from contextlib import contextmanager
import inspect
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


def _update_template_compat(
    update_template_func: Callable[..., Sequence[str]],
    project_id: str,
    version: str,
    *,
    template_dir: Path,
    context_dir: Path,
) -> Sequence[str]:
    """Call update_template across analyzer versions with different signatures."""
    try:
        parameters = inspect.signature(update_template_func).parameters
    except (TypeError, ValueError):
        parameters = {}

    semantic_kwargs = {
        "project_id": project_id,
        "pilot_auto_version": version,
        "version": version,
        "devops_data": {},
        "devops_plot_path": None,
        "performance_trend_data": [],
        "map_trend_plot_path": context_dir / "map_trend.png",
        "prediction_trend_plot_path": context_dir / "prediction_trend.png",
        "devops_trend_data": [],
        "devops_trend_plot_path": context_dir / "devops_trend.png",
        "job_ids": [],
        "template_name": "static_body.html",
        "extensions": ["html"],
        "template_dir": str(template_dir),
        "show_other_infos": False,
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

    _notify(progress_callback, "Rendering PDF")
    template_dir = Path(template_module.__file__).resolve().parent.parent / "template"
    html = _prefer_cjk_font_stack(
        _update_template_compat(
            update_template,
            project_id,
            version,
            template_dir=template_dir,
            context_dir=specsheet_dir,
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
