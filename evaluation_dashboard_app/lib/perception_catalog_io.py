"""
I/O helpers for perception_catalog_analyzer: generate parquet from pkl (or pkl.z) scene results.

Uses SceneDataFrame / scenarios_to_df for building parquet from existing .pkl or .pkl.z files.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, List, Tuple

# Topic key used in pkl.z (WebAutoEvaluatorScenarioResult.frame_results); use same when wrapping plain .pkl list
PKLZ_FRAME_RESULTS_TOPIC = "perception.object_recognition.tracking.objects"

# Default t4dataset_id when not present in pkl/pkl.z (e.g. plain scene_result.pkl); UUID format
DEFAULT_T4DATASET_ID = "00000000-0000-0000-0000-000000000000"
# Default t4dataset_name when not present; UUID format
DEFAULT_T4DATASET_NAME = "00000000-0000-0000-0000-000000000001"

# Attributes we try to read from scenario result or from first frame when normalizing plain .pkl
_SCENARIO_METADATA_ATTRS = (
    "suite_name", "scenario_name", "t4dataset_id", "t4dataset_name",
    "job_id", "project_id", "suite_report_id", "spec_report_id", "test_case_report_id",
)

try:
    import joblib
    import pandas as pd
    from perception_catalog_analyzer.dataframe import SceneDataFrame, scene2df
    from perception_catalog_analyzer.stream_io import export_scene_results_to_pkl, scenarios_to_df
    _ANALYZER_AVAILABLE = True
except ImportError as e:
    _ANALYZER_AVAILABLE = False
    _IMPORT_ERROR = e


def _scenarios_to_df_local(
    scenarios: Any,
    scenario_parser_function: Callable[[list], SceneDataFrame],
    topic_name_list: List[str] | None = None,
    *,
    debug: bool = True,
) -> SceneDataFrame:
    """
    Local implementation of scenarios_to_df with debug and support for flat list of frames.
    scenario_parser_function should accept list of PerceptionFrameResult and return SceneDataFrame.
    """
    if topic_name_list is None:
        topic_name_list = []

    # Normalize to list of "scenarios"
    # Single scenario object (has frame_results) -> wrap in list
    if not isinstance(scenarios, Iterable) or hasattr(scenarios, "frame_results"):
        scenarios = [scenarios]
    # Flat list of PerceptionFrameResult (e.g. from scene_result.pkl) -> treat whole list as one scenario
    elif (
        isinstance(scenarios, list)
        and len(scenarios) > 0
        and not hasattr(scenarios[0], "frame_results")
        and hasattr(scenarios[0], "pass_fail_result")
    ):
        scenarios = [scenarios]  # one scenario = entire list of frames

    if debug:
        print("[scenarios_to_df] type(scenarios) =", type(scenarios))
        print("[scenarios_to_df] len(scenarios) =", len(scenarios))
        if scenarios:
            first = scenarios[0]
            print("[scenarios_to_df] type(first) =", type(first))
            print("[scenarios_to_df] has frame_results =", getattr(first, "frame_results", None) is not None)
            if hasattr(first, "frame_results"):
                fr = getattr(first, "frame_results", {})
                print("[scenarios_to_df] frame_results type =", type(fr))
                if isinstance(fr, dict):
                    print("[scenarios_to_df] frame_results keys =", list(fr.keys()))
                    for k, v in list(fr.items())[:2]:
                        print(f"[scenarios_to_df]   frame_results[{k!r}] len =", len(v) if v is not None else 0, " type(elem) =", type(v[0]).__name__ if v and len(v) else "n/a")
            # Check if first element looks like a frame (list of frames format)
            if isinstance(first, list):
                print("[scenarios_to_df] first element is list, len =", len(first))
                if first:
                    print("[scenarios_to_df] first[0] type =", type(first[0]).__name__)
            elif not hasattr(first, "frame_results") and hasattr(first, "pass_fail_result"):
                print("[scenarios_to_df] first looks like a single frame (has pass_fail_result, no frame_results)")

    output = SceneDataFrame(current=pd.DataFrame())

    for sc_idx, sc in enumerate(scenarios):
        if debug:
            print(f"[scenarios_to_df] scenario index = {sc_idx}, type(sc) =", type(sc).__name__)

        suite_name = getattr(sc, "suite_name", "")
        scenario_name = getattr(sc, "scenario_name", "")
        frame_results_dict = getattr(sc, "frame_results", None)

        # Handle flat list of PerceptionFrameResult (e.g. from scene_result.pkl)
        if frame_results_dict is None or (isinstance(frame_results_dict, dict) and len(frame_results_dict) == 0):
            if isinstance(sc, list):
                frame_list = sc
            else:
                frame_list = [sc] if (hasattr(sc, "pass_fail_result") or not hasattr(sc, "frame_results")) else []
            if frame_list:
                if debug:
                    print(f"[scenarios_to_df] treating as flat list of frames, len = {len(frame_list)}, first type = {type(frame_list[0]).__name__}")
                frame_results_dict = {PKLZ_FRAME_RESULTS_TOPIC: frame_list}
            else:
                if debug:
                    print("[scenarios_to_df] skip: no frame_results and not a list of frames")
                continue
        elif not isinstance(frame_results_dict, dict):
            if debug:
                print("[scenarios_to_df] skip: frame_results is not a dict, type =", type(frame_results_dict))
            continue

        topics_this_scenario = topic_name_list if topic_name_list else list(frame_results_dict.keys())
        if debug:
            print("[scenarios_to_df] topics_this_scenario =", topics_this_scenario)

        scenario_df = SceneDataFrame(current=pd.DataFrame())

        for topic_name in topics_this_scenario:
            if topic_name not in frame_results_dict:
                if debug and sc_idx == 0:
                    print(f"[scenarios_to_df] skip topic {topic_name!r} (not in frame_results)")
                continue
            frame_results = frame_results_dict[topic_name]
            if debug and sc_idx == 0:
                print(f"[scenarios_to_df] topic {topic_name!r} -> {len(frame_results)} frames")
            df_ = scenario_parser_function(frame_results)
            if debug and sc_idx == 0:
                print(f"[scenarios_to_df] parser returned current shape = {getattr(getattr(df_, 'current', None), 'shape', None)}, future = {getattr(df_, 'future', None) is not None}")
            df_["topic_name"] = topic_name
            scenario_df = scenario_df.concatenate(df_, ignore_index=True)

        if scenario_df.empty():
            if debug:
                print(f"[scenarios_to_df] scenario_df is empty after topics, skip")
            continue
        scenario_df["t4dataset_id"] = getattr(sc, "t4dataset_id", None)
        scenario_df["suite_name"] = suite_name
        scenario_df["scenario_name"] = scenario_name
        scenario_df["t4dataset_name"] = getattr(sc, "t4dataset_name", None)
        output = output.concatenate(scenario_df)

    if debug:
        print("[scenarios_to_df] output.current shape =", getattr(output.current, "shape", None), " output.future =", output.future is not None)
    return output


def _infer_scenario_and_suite_from_pkl_path(pkl_file: str | Path) -> Tuple[str, str]:
    """
    Infer scenario_name and suite_name from pkl file path.
    e.g. .../SuiteName_uuid/ScenarioName_uuid/scene_result.pkl -> (ScenarioName_uuid, SuiteName_uuid)
    e.g. .../ScenarioName_uuid/scene_result.pkl -> (ScenarioName_uuid, "")
    """
    path = Path(pkl_file).resolve()
    parent = path.parent
    scenario_name = parent.name if parent.name else ""
    suite_name = ""
    grandparent = parent.parent
    if grandparent and grandparent.name:
        parts = grandparent.name.rsplit("_", 1)
        if len(parts) == 2 and len(parts[1]) == 36 and parts[1].count("-") == 4:
            suite_name = grandparent.name
    return (scenario_name, suite_name)


def _get_metadata_from_frame(frame: Any) -> dict:
    """Extract scenario-like metadata from a frame if present (e.g. PerceptionFrameResult with t4dataset_id)."""
    out = {}
    for attr in _SCENARIO_METADATA_ATTRS:
        val = getattr(frame, attr, None)
        if val is not None and val != "":
            out[attr] = val
    return out


def _load_t4_metadata_sidecar(pkl_file: str | Path | None) -> dict:
    """
    If a t4_metadata.json sidecar exists next to the pkl (written at download time from API),
    load it and return dict with t4_dataset_id / t4_dataset_version_id (keys normalized to t4dataset_*).
    Otherwise return {}.
    """
    if not pkl_file:
        return {}
    pkl_path = Path(pkl_file)
    if not pkl_path.is_absolute():
        pkl_path = pkl_path.resolve()
    sidecar = pkl_path.parent / "t4_metadata.json"
    if not sidecar.is_file():
        return {}
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = {}
        if data.get("t4_dataset_id"):
            out["t4dataset_id"] = data["t4_dataset_id"]
        if data.get("t4_dataset_version_id"):
            out["t4dataset_version_id"] = data.get("t4_dataset_version_id")
        return out
    except (OSError, json.JSONDecodeError):
        return {}


def _normalize_loaded_pkl(
    data: Any,
    *,
    pkl_file: str | Path | None = None,
    project_id: str | None = None,
    job_id: str | None = None,
) -> Any:
    """
    Normalize loaded pkl data so plain .pkl (list of PerceptionFrameResult) has the same
    shape as .pkl.z (WebAutoEvaluatorScenarioResult with frame_results dict).
    Tries to fill suite_name, scenario_name, t4dataset_id, etc. from:
    - pkl_file path (scenario_name = parent dir, suite_name = grandparent when SuiteName_uuid)
    - first frame attributes (t4dataset_id, t4dataset_name, job_id, project_id, ...)
    - explicit project_id, job_id when provided (e.g. from Download page).
    """
    if hasattr(data, "frame_results"):
        fr = getattr(data, "frame_results", None)
        if isinstance(fr, dict) and len(fr) > 0:
            return data
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if hasattr(first, "pass_fail_result"):
            meta = _get_metadata_from_frame(first)
            # Use t4_metadata.json sidecar (written at download when API provides t4_dataset_id) if pkl has no id
            sidecar = _load_t4_metadata_sidecar(pkl_file)
            if sidecar:
                meta = {**meta, **sidecar}
            scenario_name = meta.get("scenario_name") or ""
            suite_name = meta.get("suite_name") or ""
            if pkl_file:
                path_scenario, path_suite = _infer_scenario_and_suite_from_pkl_path(pkl_file)
                scenario_name = scenario_name or path_scenario
                suite_name = suite_name or path_suite
            return SimpleNamespace(
                suite_name=suite_name,
                scenario_name=scenario_name,
                t4dataset_id=meta.get("t4dataset_id") or DEFAULT_T4DATASET_ID,
                t4dataset_name=meta.get("t4dataset_name") or DEFAULT_T4DATASET_NAME,
                job_id=job_id if job_id is not None else meta.get("job_id"),
                project_id=project_id if project_id is not None else meta.get("project_id"),
                suite_report_id=meta.get("suite_report_id"),
                spec_report_id=meta.get("spec_report_id"),
                test_case_report_id=meta.get("test_case_report_id"),
                frame_results={PKLZ_FRAME_RESULTS_TOPIC: data},
            )
    return data


def _require_analyzer() -> None:
    if not _ANALYZER_AVAILABLE:
        raise ImportError(
            "perception_catalog_analyzer is required for pkl download and parquet generation. "
            f"Install it first. Original error: {_IMPORT_ERROR!s}"
        ) from _IMPORT_ERROR


def download_scene_results_to_pkl(
    project_id: str,
    job_id: str,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[str]:
    """
    Download scene report as pkl.z files using perception_catalog_analyzer.

    Creates out_dir and an "archive" subdir, then calls export_scene_results_to_pkl
    to download and save *.pkl.z files there.

    Args:
        project_id: Project ID of the report.
        job_id: Job ID of the report.
        out_dir: Base output directory; archive subdir will be out_dir/archive.
        overwrite: If True, overwrite existing pkl.z files.

    Returns:
        List of paths to downloaded *.pkl.z files (under out_dir/archive).
    """
    _require_analyzer()
    out_dir = Path(out_dir)
    archive_outdir = out_dir / "archive"
    archive_outdir.mkdir(parents=True, exist_ok=True)
    archive_str = os.fspath(archive_outdir)
    pkl_files = export_scene_results_to_pkl(
        project_id=project_id,
        job_id=job_id,
        out_dir=archive_str,
        overwrite=overwrite,
    )
    return pkl_files


def build_scene_dataframe_from_pkl_dir(
    pkl_dir: str | Path,
    *,
    max_files: int | None = None,
    skip_empty: bool = True,
    skip_bad_dtype: bool = True,
    on_skip: Callable[[str, str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    project_id: str | None = None,
    job_id: str | None = None,
) -> SceneDataFrame:
    """
    Build a SceneDataFrame from all *.pkl and *.pkl.z files in a directory.

    Args:
        pkl_dir: Directory containing *.pkl (and/or *.pkl.z) files.
        max_files: If set, process at most this many files (for testing).
        skip_empty: If True, skip files that yield an empty dataframe.
        skip_bad_dtype: If True, skip files where df.current["x_error"].dtype != "float64".
        on_skip: Optional callback (file_path, reason) when a file is skipped.
        on_progress: Optional callback (done_count, total_count) after each file is processed.
        project_id: Optional project ID (e.g. from Download page) to fill when normalizing plain .pkl.
        job_id: Optional job ID (e.g. from Download page) to fill when normalizing plain .pkl.

    Returns:
        SceneDataFrame concatenated from all valid pkl/pkl.z files.
    """
    _require_analyzer()
    pkl_dir = Path(pkl_dir)
    pkl_z = sorted(pkl_dir.rglob("*.pkl.z"))
    pkl_plain = sorted(pkl_dir.rglob("*.pkl"))
    pkl_files = sorted(set(pkl_z + pkl_plain))
    if max_files is not None:
        pkl_files = pkl_files[:max_files]

    total = len(pkl_files)
    df = SceneDataFrame(current=pd.DataFrame())

    def _report_progress(done: int) -> None:
        if on_progress:
            on_progress(done, total)

    for i, pkl_file in enumerate(pkl_files):
        try:
            if str(pkl_file).lower().endswith(".pkl.z"):
                try:
                    data = joblib.load(pkl_file)
                except NameError:
                    raise ImportError("joblib is required for .pkl.z: pip install joblib")
            else:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
        except Exception as e:
            if on_skip:
                on_skip(pkl_file, f"failed to load: {e}")
                _report_progress(i + 1)
                continue
            raise
        data = _normalize_loaded_pkl(
            data,
            pkl_file=pkl_file,
            project_id=project_id,
            job_id=job_id,
        )
        try:
            df_ = _scenarios_to_df_local(data, scenario_parser_function=scene2df, debug=False)
        except Exception as e:
            if on_skip:
                on_skip(pkl_file, f"failed to convert: {e}")
                _report_progress(i + 1)
                continue
            raise
        del data
        if df_.empty():
            if skip_empty:
                if on_skip:
                    on_skip(pkl_file, "empty")
                del df_
                gc.collect()
                _report_progress(i + 1)
                continue
        if skip_bad_dtype and hasattr(df_, "current") and "x_error" in getattr(df_.current, "columns", []):
            if df_.current["x_error"].dtype != "float64":
                if on_skip:
                    on_skip(pkl_file, f"bad dtype x_error={df_.current['x_error'].dtype}")
                del df_
                gc.collect()
                _report_progress(i + 1)
                continue
        df = df.concatenate(df_)
        del df_
        gc.collect()
        _report_progress(i + 1)
    return df


def pkl_archive_to_parquet(
    pkl_dir: str | Path,
    parquet_path: str | Path | None = None,
    *,
    max_files: int | None = None,
    skip_empty: bool = True,
    skip_bad_dtype: bool = True,
    on_skip: Callable[[str, str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    project_id: str | None = None,
    job_id: str | None = None,
) -> str:
    """
    Generate a single parquet file from all *.pkl and *.pkl.z files in a directory.

    Args:
        pkl_dir: Directory containing *.pkl (and/or *.pkl.z) files.
        parquet_path: Output parquet path. Default: pkl_dir / "scene_result.parquet".
        max_files: If set, use at most this many files.
        skip_empty: Skip files that yield empty dataframe.
        skip_bad_dtype: Skip files where x_error is not float64.
        on_skip: Optional callback (file_path, reason) when a file is skipped.
        on_progress: Optional callback (done_count, total_count) after each file is processed.
        project_id: Optional project ID to fill when normalizing plain .pkl (e.g. from Download page).
        job_id: Optional job ID to fill when normalizing plain .pkl (e.g. from Download page).

    Returns:
        Path to the written parquet file.
    """
    _require_analyzer()
    pkl_dir = Path(pkl_dir)
    df = build_scene_dataframe_from_pkl_dir(
        pkl_dir,
        max_files=max_files,
        skip_empty=skip_empty,
        skip_bad_dtype=skip_bad_dtype,
        on_skip=on_skip,
        on_progress=on_progress,
        project_id=project_id,
        job_id=job_id,
    )
    df.to_parquet(pkl_dir)
    parquet_file = pkl_dir / "current.parquet"
    # Check if the file is generated successfully and return its path
    if parquet_file.exists():
        return os.fspath(parquet_file)
    else:
        raise ValueError(f"Failed to generate parquet file: {parquet_file}")
