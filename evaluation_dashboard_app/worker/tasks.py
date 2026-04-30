"""
RQ job handlers for heavy tasks. Each job receives task_id and parameters dict.
Updates Postgres task status (running -> completed/failed).
"""

import os
import re
import sys
from typing import Any, Dict

# App root on path for lib imports
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from lib.db import update_task_status, update_task_progress, append_task_log, update_task_result_summary

# Optional imports for tasks that need them
def _import_eval_summary():
    from lib import eval_summary
    return eval_summary

def _import_catalog_io():
    try:
        from lib.perception_catalog_io import pkl_archive_to_parquet
        return pkl_archive_to_parquet
    except ImportError:
        return None


def job_generate_summary_csv(task_id: str, parameters: Dict[str, Any]) -> None:
    """Generate Summary.csv and Score.csv under eval_root."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting generate_summary_csv")
    try:
        eval_summary = _import_eval_summary()
        eval_root = parameters.get("eval_root")
        if not eval_root:
            update_task_status(task_id, "failed", error_message="Missing eval_root")
            return
        append_task_log(task_id, f"Generating summary under {eval_root}")
        info = eval_summary.generate_summary_and_score_csv(eval_root)
        result_path = info.get("summary_path", eval_root)
        update_task_result_summary(
            task_id,
            {
                "job": "generate_summary_csv",
                "summary_path": result_path,
                "summary_rows": info.get("summary_rows", 0),
                "score_rows": info.get("score_rows", 0),
            },
        )
        append_task_log(task_id, f"Done. Output: {result_path}")
        update_task_status(task_id, "completed", result_path=result_path)
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_run_eval_dirs(task_id: str, parameters: Dict[str, Any]) -> None:
    """Run eval_result for each dir under eval_root, then generate Summary/Score CSV."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting run_eval_dirs")
    try:
        eval_summary = _import_eval_summary()
        eval_root = parameters.get("eval_root")
        recursive = parameters.get("recursive", True)
        overwrite = parameters.get("overwrite", False)
        if not eval_root:
            update_task_status(task_id, "failed", error_message="Missing eval_root")
            return
        target_dirs = eval_summary.find_eval_result_dirs(eval_root, recursive=recursive)
        if not target_dirs:
            update_task_status(task_id, "failed", error_message="No result directories found")
            return
        total = len(target_dirs)
        append_task_log(task_id, f"Processing {total} directories")
        statuses = []
        for i, result_dir in enumerate(target_dirs):
            pct = 100.0 * (i + 1) / total if total else 0
            update_task_progress(task_id, message=f"Processing {i+1}/{total}: {result_dir}", pct=pct)
            append_task_log(task_id, f"Processing {i+1}/{total}: {result_dir}")
            status = eval_summary.run_eval_result_for_dir(result_dir, overwrite=overwrite)
            statuses.append(status)
            if status.get("status") == "failed":
                append_task_log(task_id, f"Eval failed for {result_dir}: {status.get('detail', '')}")
        append_task_log(task_id, "Generating summary CSV")
        info = eval_summary.generate_summary_and_score_csv(eval_root)
        result_path = info.get("summary_path", eval_root)
        failed = [s for s in statuses if s.get("status") == "failed"]
        skipped = [s for s in statuses if s.get("status") == "skipped"]
        succeeded = [s for s in statuses if s.get("status") == "success"]
        summary = {
            "job": "run_eval_dirs",
            "directories_processed": total,
            "success": len(succeeded),
            "failed": len(failed),
            "skipped": len(skipped),
            "summary_path": result_path,
            "summary_rows": info.get("summary_rows", 0),
            "score_rows": info.get("score_rows", 0),
        }
        update_task_result_summary(task_id, summary)
        append_task_log(task_id, f"Done. Output: {result_path}")
        update_task_status(task_id, "completed", result_path=result_path)
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_build_parquet(task_id: str, parameters: Dict[str, Any]) -> None:
    """Build scene_result parquet from pkl directory."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting build_parquet")
    try:
        pkl_archive_to_parquet = _import_catalog_io()
        if pkl_archive_to_parquet is None:
            update_task_status(task_id, "failed", error_message="perception_catalog_io not available")
            return
        pkl_dir = parameters.get("pkl_dir")
        if not pkl_dir:
            update_task_status(task_id, "failed", error_message="Missing pkl_dir")
            return
        append_task_log(task_id, f"Building parquet from {pkl_dir}")
        project_id = parameters.get("project_id")
        job_id = parameters.get("job_id")
        parquet_path = pkl_archive_to_parquet(
            pkl_dir,
            on_progress=None,
            on_skip=None,
            project_id=project_id,
            job_id=job_id,
        )
        update_task_result_summary(task_id, {"job": "build_parquet", "output_path": parquet_path})
        append_task_log(task_id, f"Done. Output: {parquet_path}")
        update_task_status(task_id, "completed", result_path=parquet_path)
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def _progress_callback(task_id: str, message: str) -> None:
    """Append message to task log and update progress_message; derive pct from 'N/M' if present."""
    append_task_log(task_id, message)
    match = re.search(r"(\d+)\s*/\s*(\d+)", message)
    if match:
        n, m = int(match.group(1)), int(match.group(2))
        pct = 100.0 * n / m if m else 0
        update_task_progress(task_id, message=message, pct=pct)
    else:
        update_task_progress(task_id, message=message)


def job_download_results(task_id: str, parameters: Dict[str, Any]) -> None:
    """Download job results (archives or result JSON) and extract/organize. Requires auth."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting download_results")
    try:
        from lib import download_core  # noqa: F401
        output_path = parameters.get("output_path")
        project_id = parameters.get("project_id")
        job_id = parameters.get("job_id")
        suite_id = parameters.get("suite_id")
        suite_ids = parameters.get("suite_ids")  # optional list
        download_type = parameters.get("download_type", "archives")  # archives | result_json
        phase = parameters.get("phase", "first")
        skip_large_file = parameters.get("skip_large_file", False)
        large_file_mb = float(parameters.get("large_file_mb", 50.0))
        keep_zip_files = parameters.get("keep_zip_files", False)
        if not all([output_path, project_id, job_id]):
            update_task_status(task_id, "failed", error_message="Missing output_path, project_id, or job_id")
            return
        on_progress = lambda msg: _progress_callback(task_id, msg)
        on_warning = lambda msg: append_task_log(task_id, msg)
        failure_count, total_attempted, rows = download_core.run_download_results(
            project_id=project_id,
            job_id=job_id,
            suite_id=suite_id,
            output_path=output_path,
            download_type=download_type,
            phase=phase,
            suite_ids=suite_ids,
            skip_large_file=skip_large_file,
            large_file_mb=large_file_mb,
            keep_zip_files=keep_zip_files,
            on_progress=on_progress,
            on_warning=on_warning,
        )
        success_count = total_attempted - failure_count
        summary = {
            "job": "download_results",
            "total": total_attempted,
            "success": success_count,
            "failed": failure_count,
            "output_path": output_path,
            "rows": rows[:500],
        }
        update_task_result_summary(task_id, summary)
        append_task_log(task_id, "Download and extract completed")
        if success_count == 0 and failure_count > 0:
            err_msg = f"Download completed with {failure_count} failures. See task log for details."
            update_task_status(task_id, "failed", result_path=output_path, error_message=err_msg)
        else:
            update_task_status(task_id, "completed", result_path=output_path)
    except ImportError:
        update_task_status(
            task_id,
            "failed",
            error_message="Download worker not available: lib.download_core not implemented",
        )
    except NotImplementedError as e:
        update_task_status(task_id, "failed", error_message=str(e))
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_download_scenarios(task_id: str, parameters: Dict[str, Any]) -> None:
    """Download scenarios from job to output_dir. Requires auth."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting download_scenarios")
    try:
        from lib import download_core  # noqa: F401
        output_dir = parameters.get("output_dir") or parameters.get("output_path")
        project_id = parameters.get("project_id")
        job_id = parameters.get("job_id")
        suite_id = parameters.get("suite_id")
        suite_ids = parameters.get("suite_ids")
        overwrite = parameters.get("overwrite", False)
        scenario_name_filter = parameters.get("scenario_name_filter")
        selected_ids = parameters.get("selected_ids")
        if not all([output_dir, project_id, job_id]):
            update_task_status(task_id, "failed", error_message="Missing output_dir, project_id, or job_id")
            return
        on_progress = lambda msg: _progress_callback(task_id, msg)
        on_warning = lambda msg: append_task_log(task_id, msg)
        failure_count, total_attempted, rows = download_core.run_download_scenarios(
            project_id=project_id,
            job_id=job_id,
            suite_id=suite_id,
            output_dir=output_dir,
            overwrite=overwrite,
            scenario_name_filter=scenario_name_filter,
            selected_ids=selected_ids,
            suite_ids=suite_ids,
            on_progress=on_progress,
            on_warning=on_warning,
        )
        success_count = total_attempted - failure_count
        summary = {
            "job": "download_scenarios",
            "total": total_attempted,
            "success": success_count,
            "failed": failure_count,
            "output_path": output_dir,
            "rows": rows[:500],
        }
        update_task_result_summary(task_id, summary)
        append_task_log(task_id, "Download scenarios completed")
        if failure_count > 0:
            err_msg = f"Download completed with {failure_count} failures. See task log for details."
            update_task_status(task_id, "failed", result_path=output_dir, error_message=err_msg)
        else:
            update_task_status(task_id, "completed", result_path=output_dir)
    except ImportError:
        update_task_status(
            task_id,
            "failed",
            error_message="Download worker not available: lib.download_core not implemented",
        )
    except NotImplementedError as e:
        update_task_status(task_id, "failed", error_message=str(e))
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_download_and_eval(task_id: str, parameters: Dict[str, Any]) -> None:
    """Download results, then run eval and parquet generation. Stops on download failure."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting download_and_eval combined workflow")
    try:
        from lib import download_core
        output_path = parameters.get("output_path")
        project_id = parameters.get("project_id")
        job_id = parameters.get("job_id")
        suite_id = parameters.get("suite_id")
        suite_ids = parameters.get("suite_ids")
        download_type = parameters.get("download_type", "archives")
        phase = parameters.get("phase", "perception.object_recognition.tracking.objects")
        skip_large_file = parameters.get("skip_large_file", False)
        large_file_mb = float(parameters.get("large_file_mb", 50.0))
        keep_zip_files = parameters.get("keep_zip_files", False)
        run_eval = parameters.get("run_eval", True)
        generate_parquet = parameters.get("generate_parquet", True)
        eval_recursive = parameters.get("eval_recursive", True)
        eval_overwrite = parameters.get("eval_overwrite", False)
        
        if not all([output_path, project_id, job_id]):
            update_task_status(task_id, "failed", error_message="Missing output_path, project_id, or job_id")
            return
        
        on_progress = lambda msg: _progress_callback(task_id, msg)
        on_warning = lambda msg: append_task_log(task_id, msg)
        
        result = download_core.run_download_and_eval(
            project_id=project_id,
            job_id=job_id,
            suite_id=suite_id,
            output_path=output_path,
            download_type=download_type,
            phase=phase,
            skip_large_file=skip_large_file,
            large_file_mb=large_file_mb,
            keep_zip_files=keep_zip_files,
            suite_ids=suite_ids,
            run_eval=run_eval,
            generate_parquet=generate_parquet,
            eval_recursive=eval_recursive,
            eval_overwrite=eval_overwrite,
            on_progress=on_progress,
            on_warning=on_warning,
        )
        
        # Build result summary
        summary = {
            "job": "download_and_eval",
            "download_success": result.get("download_success", False),
            "download_summary": result.get("download_summary", {}),
            "eval_summary": result.get("eval_summary", {}),
            "parquet_path": result.get("parquet_path", ""),
            "errors": result.get("errors", []),
        }
        update_task_result_summary(task_id, summary)
        
        if not result.get("download_success"):
            err_msg = result.get("errors", ["Download failed"])[0]
            append_task_log(task_id, f"Stopped: {err_msg}")
            update_task_status(task_id, "failed", result_path=output_path, error_message=err_msg)
        elif result.get("errors"):
            # Partial success with some errors
            errs = "; ".join(result["errors"][:5])
            append_task_log(task_id, f"Completed with errors: {errs}")
            update_task_status(task_id, "completed", result_path=output_path)
        else:
            append_task_log(task_id, "Download and eval completed successfully")
            update_task_status(task_id, "completed", result_path=output_path)
            
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        update_task_status(task_id, "failed", error_message=str(e))
        raise


# Map task_type (from Postgres) to job function
TASK_JOB_MAP = {
    "generate_summary_csv": job_generate_summary_csv,
    "run_eval_dirs": job_run_eval_dirs,
    "build_parquet": job_build_parquet,
    "download_results": job_download_results,
    "download_scenarios": job_download_scenarios,
    "download_and_eval": job_download_and_eval,
}


def run_job(task_id: str, task_type: str, parameters: Dict[str, Any]) -> None:
    """Dispatch to the right job by task_type. Called by RQ worker."""
    fn = TASK_JOB_MAP.get(task_type)
    if not fn:
        update_task_status(task_id, "failed", error_message=f"Unknown task type: {task_type}")
        return
    # Mark running as soon as the worker claims the job (before heavy job_* setup).
    # Otherwise the UI stays "pending" until the first line of each job_* runs.
    update_task_status(task_id, "running")
    fn(task_id, parameters)
