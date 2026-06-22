"""
RQ job handlers for heavy tasks. Each job receives task_id and parameters dict.
Updates Postgres task status (running -> completed/failed).
"""

import os
import re
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# App root on path for lib imports
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from lib.db import (
    append_task_log,
    get_task,
    update_task_progress,
    update_task_result_summary,
    update_task_status,
)
from lib.run_metadata import (
    read_run_metadata,
    resolve_run_directory_from_task_parameters,
    upsert_run_metadata,
)
from lib.specsheet_report import write_trend_metadata

_RELEASE_PERFORMANCE_CATALOG_ID = "e36d75b9-6c3a-4970-9b9b-5cd13f7a9da3"
_RELEASE_PERFORMANCE_INTEGRATION_ID = "96ad8fba-0228-4c2b-9166-07d4de1a0760"
_RELEASE_DEVOPS_CATALOG_ID = "ab0f8498-cc1b-4726-836f-e18e8bcb3200"
_RELEASE_DEVOPS_INTEGRATION_ID = "295cff78-9bc9-4d60-b7aa-f95be6ff96a4"
_RELEASE_OPTIONAL_CATALOG_ID = "09039022-ec91-41bf-9e93-fdefccdfc9bc"
_RELEASE_SKIP_LARGE_FILE = True
_RELEASE_LARGE_FILE_MB = 50.0


def _normalize_max_wait_seconds(value: Any, default: float = 0.0) -> float:
    """Coerce wait timeout values; numbers <= 0 disable the timeout."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = float(default)
    return seconds if seconds > 0 else 0.0


def _wait_progress_pct(
    *,
    elapsed: float,
    timeout_seconds: float,
    pct_start: float,
    pct_end: float,
) -> float:
    """Progress helper that works for both bounded and unbounded waits."""
    if pct_end <= pct_start:
        return pct_end
    if timeout_seconds > 0:
        ratio = min(max(elapsed / timeout_seconds, 0.0), 1.0)
        return pct_start + ratio * (pct_end - pct_start)
    return min(pct_start + max(1.0, (elapsed / 3600.0) * 2.0), pct_end)


def _make_default_evaluator_description(parameters: Dict[str, Any]) -> str:
    stamp = time.strftime("%m-%d %H:%M")
    source_job_id = str(parameters.get("source_job_id") or "").strip()
    if source_job_id:
        return f"♻️ evaluator artifact retest [{source_job_id}] [{stamp}] 📦"
    target_name = str(parameters.get("target_name") or "").strip() or "default"
    return f"🚀 evaluator workflow [{target_name}] [{stamp}] 📦"


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


def _parquet_progress_callback(
    task_id: str,
    *,
    prefix: str = "Parquet",
    pct_start: float = 0.0,
    pct_end: float = 100.0,
):
    """Return a pkl-file progress callback for pkl_archive_to_parquet."""

    def _on_progress(done: int, total: int) -> None:
        total_safe = max(1, int(total or 0))
        done_safe = min(max(0, int(done or 0)), total_safe)
        pct = pct_start + (done_safe / total_safe) * max(0.0, pct_end - pct_start)
        message = f"{prefix}: processing pkl files {done_safe}/{total_safe}"
        update_task_progress(task_id, message=message, pct=min(pct_end, pct))
        append_task_log(task_id, message)

    return _on_progress


def _eval_worker_count(parameters: Dict[str, Any], total: int) -> int:
    """Resolve bounded eval concurrency. Defaults to 4, capped by total dirs."""
    if total <= 0:
        return 1
    raw = parameters.get("eval_workers", os.environ.get("EVAL_WORKERS_DEFAULT", 4))
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        workers = 4
    try:
        max_workers = int(os.environ.get("EVAL_WORKERS_MAX", 16))
    except ValueError:
        max_workers = 16
    return max(1, min(workers, max_workers, total))


def _compact_eval_path(path: Any, *, parts: int = 2) -> str:
    """Return a readable tail path for task logs without flooding the UI."""
    text = str(path or "").strip()
    if not text:
        return "unknown"
    try:
        p = Path(text)
        tail = p.parts[-parts:]
        return "/".join(tail) if tail else text
    except Exception:
        return text


def _run_eval_result_dirs(
    *,
    task_id: str,
    eval_summary: Any,
    target_dirs: list[str],
    overwrite: bool,
    eval_workers: int,
    pct_start: float,
    pct_end: float,
    label: str = "Eval",
) -> list[Dict[str, Any]]:
    """Run eval_result across result dirs with bounded concurrency and calm progress."""
    total = len(target_dirs)
    if total <= 0:
        update_task_progress(task_id, message=f"{label}: no result directories found", pct=pct_end)
        return []

    workers = max(1, min(int(eval_workers or 1), total))
    span = max(0.0, pct_end - pct_start)
    statuses: list[Dict[str, Any]] = []
    counts = {"success": 0, "skipped": 0, "failed": 0}

    def _record(status: Dict[str, Any]) -> str:
        statuses.append(status)
        state = str(status.get("status") or "failed")
        if state not in counts:
            state = "failed"
        counts[state] += 1
        if state == "failed":
            append_task_log(
                task_id,
                f"{label}: eval failed for {status.get('path', '')}: {status.get('detail', '')}",
            )
        return state

    def _progress(done: int, latest: str | None = None) -> None:
        pct = pct_start + (done / total) * span
        latest_text = f" latest: {latest}" if latest else ""
        update_task_progress(
            task_id,
            message=(
                f"{label}: completed {done}/{total} dirs "
                f"(success {counts['success']}, skipped {counts['skipped']}, failed {counts['failed']})"
                f"{latest_text}"
            ),
            pct=min(pct_end, pct),
        )

    append_task_log(task_id, f"{label}: running eval_result for {total} directories with {workers} worker(s)")
    _progress(0)

    if workers == 1:
        for i, result_dir in enumerate(target_dirs, start=1):
            append_task_log(task_id, f"{label}: starting {i}/{total}: {result_dir}")
            status = eval_summary.run_eval_result_for_dir(result_dir, overwrite=overwrite)
            state = _record(status)
            short_path = _compact_eval_path(status.get("path") or result_dir)
            append_task_log(task_id, f"{label}: {i}/{total} {state}: {short_path}")
            _progress(i, short_path)
        return statuses

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(eval_summary.run_eval_result_for_dir, result_dir, overwrite=overwrite): result_dir
            for result_dir in target_dirs
        }
        for done, future in enumerate(as_completed(future_map), start=1):
            result_dir = future_map[future]
            try:
                status = future.result()
            except Exception as exc:
                status = {"path": result_dir, "status": "failed", "detail": str(exc)}
            state = _record(status)
            short_path = _compact_eval_path(status.get("path") or result_dir)
            append_task_log(task_id, f"{label}: {done}/{total} {state}: {short_path}")
            _progress(done, short_path)
    return statuses


def _copy_task_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}
    for key, value in (parameters or {}).items():
        if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
            copied[key] = value
        else:
            copied[key] = str(value)
    return copied


def _resolve_active_integration_id(api: Any, project_id: str, catalog_id: str) -> str:
    """Resolve latest active integration for a catalog when the UI only provided a catalog id."""
    url = f"{api.api_base_url}/projects/{project_id}/integrations"
    response = api.request(url, {"catalog_id": catalog_id, "size": 100}, method="GET")
    if response is None:
        raise RuntimeError(f"No response returned while loading integrations for catalog {catalog_id}.")
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to load integrations for catalog {catalog_id}: status={response.status_code}"
        )
    payload = json.loads(response.content)
    integrations = payload.get("integrations", []) or []
    active = [
        item for item in integrations
        if isinstance(item, dict)
        and str(item.get("catalog_id") or "").strip() == catalog_id
        and not bool(item.get("deleted"))
    ]
    if not active:
        raise RuntimeError(f"No active integration found for catalog {catalog_id}.")

    def _sort_key(item: Dict[str, object]) -> tuple:
        return (
            str(item.get("updated_at") or ""),
            int(item.get("version_id") or 0),
            str(item.get("id") or ""),
        )

    active.sort(key=_sort_key, reverse=True)
    return str(active[0].get("id") or "").strip()


def _task_row_payload(task_id: str) -> Dict[str, Any]:
    row = get_task(task_id) or {}
    return {
        "id": str(row.get("id") or task_id),
        "type": str(row.get("type") or "").strip(),
        "status": str(row.get("status") or "").strip(),
        "requested_by": str(row.get("session_id") or "").strip(),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "result_path": str(row.get("result_path") or "").strip(),
        "error_message": str(row.get("error_message") or "").strip(),
        "progress_message": str(row.get("progress_message") or "").strip(),
        "progress_pct": row.get("progress_pct"),
    }


def _task_request_payload(parameters: Dict[str, Any]) -> Dict[str, Any]:
    params = _copy_task_parameters(parameters)
    return {
        "environment": str(params.get("environment") or "default").strip() or "default",
        "project_id": str(params.get("project_id") or "").strip(),
        "job_id": str(params.get("job_id") or "").strip(),
        "catalog_id": str(params.get("catalog_id") or "").strip(),
        "integration_id": str(params.get("integration_id") or "").strip(),
        "source_job_id": str(params.get("source_job_id") or "").strip(),
        "target_name": str(params.get("target_name") or "").strip(),
        "description": str(params.get("description") or "").strip(),
        "suite_id": str(params.get("suite_id") or "").strip(),
        "suite_ids": list(params.get("suite_ids") or []),
        "download_type": str(params.get("download_type") or "").strip(),
        "phase": str(params.get("phase") or "").strip(),
        "skip_large_file": bool(params.get("skip_large_file", False)),
        "large_file_mb": params.get("large_file_mb"),
        "keep_zip_files": bool(params.get("keep_zip_files", False)),
        "run_eval": bool(params.get("run_eval", False)),
        "generate_parquet": bool(params.get("generate_parquet", False)),
        "eval_recursive": bool(params.get("eval_recursive", False)),
        "eval_overwrite": bool(params.get("eval_overwrite", False)),
        "max_retries": params.get("max_retries"),
        "clean_build": bool(params.get("clean_build", False)),
        "debug": bool(params.get("debug", False)),
        "is_tag": bool(params.get("is_tag", False)),
        "scenario_name_filter": str(params.get("scenario_name_filter") or "").strip(),
        "selected_ids": list(params.get("selected_ids") or []),
        "output_path": str(
            params.get("output_path")
            or params.get("output_dir")
            or params.get("eval_root")
            or params.get("pkl_dir")
            or ""
        ).strip(),
        "parameters": params,
    }


def _build_run_metadata_patch(task_id: str, parameters: Dict[str, Any], *, task_type: str) -> Dict[str, Any]:
    return {
        "source_mode": task_type,
        "task": _task_row_payload(task_id),
        "request": _task_request_payload(parameters),
    }


def _update_run_metadata(
    task_id: str,
    parameters: Dict[str, Any],
    *,
    task_type: str,
    create_missing: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir = resolve_run_directory_from_task_parameters(parameters, create_missing=create_missing)
    if run_dir is None:
        return
    patch = _build_run_metadata_patch(task_id, parameters, task_type=task_type)
    if extra:
        patch.update(extra)
    try:
        upsert_run_metadata(run_dir, patch, create_missing=create_missing)
    except Exception:
        pass


def _append_run_event(
    task_id: str,
    parameters: Dict[str, Any],
    *,
    task_type: str,
    message: str,
) -> None:
    run_dir = resolve_run_directory_from_task_parameters(parameters, create_missing=False)
    if run_dir is None:
        return
    try:
        metadata = read_run_metadata(run_dir)
        events = list(metadata.get("events") or [])
        events.append({"at": _task_row_payload(task_id).get("updated_at"), "message": message})
        if len(events) > 50:
            events = events[-50:]
        upsert_run_metadata(
            run_dir,
            {
                "events": events,
                "task": _task_row_payload(task_id),
            },
            create_missing=False,
        )
    except Exception:
        pass


def _mark_run_status(
    task_id: str,
    parameters: Dict[str, Any],
    *,
    task_type: str,
    status: str,
    error_message: str = "",
    result_path: str = "",
    extra: Optional[Dict[str, Any]] = None,
    create_missing: bool = False,
) -> None:
    patch: Dict[str, Any] = {
        "task": {
            "status": status,
        }
    }
    if error_message:
        patch["task"]["error_message"] = error_message
    if result_path:
        patch["task"]["result_path"] = result_path
    if extra:
        patch.update(extra)
    _update_run_metadata(
        task_id,
        parameters,
        task_type=task_type,
        create_missing=create_missing,
        extra=patch,
    )


def job_generate_summary_csv(task_id: str, parameters: Dict[str, Any]) -> None:
    """Generate Summary.csv and Score.csv under eval_root."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting generate_summary_csv")
    _mark_run_status(task_id, parameters, task_type="generate_summary_csv", status="running")
    try:
        eval_summary = _import_eval_summary()
        eval_root = parameters.get("eval_root")
        if not eval_root:
            _mark_run_status(
                task_id, parameters, task_type="generate_summary_csv", status="failed", error_message="Missing eval_root"
            )
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
        _update_run_metadata(
            task_id,
            parameters,
            task_type="generate_summary_csv",
            extra={
                "evaluation": {
                    "summary_path": result_path,
                    "summary_rows": info.get("summary_rows", 0),
                    "score_rows": info.get("score_rows", 0),
                }
            },
        )
        append_task_log(task_id, f"Done. Output: {result_path}")
        _mark_run_status(
            task_id, parameters, task_type="generate_summary_csv", status="completed", result_path=str(result_path or "")
        )
        update_task_status(task_id, "completed", result_path=result_path)
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        _mark_run_status(
            task_id, parameters, task_type="generate_summary_csv", status="failed", error_message=str(e)
        )
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_run_eval_dirs(task_id: str, parameters: Dict[str, Any]) -> None:
    """Run eval_result for each dir under eval_root, then generate Summary/Score CSV."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting run_eval_dirs")
    _mark_run_status(task_id, parameters, task_type="run_eval_dirs", status="running")
    try:
        eval_summary = _import_eval_summary()
        eval_root = parameters.get("eval_root")
        recursive = parameters.get("recursive", True)
        overwrite = parameters.get("overwrite", False)
        if not eval_root:
            _mark_run_status(task_id, parameters, task_type="run_eval_dirs", status="failed", error_message="Missing eval_root")
            update_task_status(task_id, "failed", error_message="Missing eval_root")
            return
        target_dirs = eval_summary.find_eval_result_dirs(eval_root, recursive=recursive)
        if not target_dirs:
            _mark_run_status(
                task_id, parameters, task_type="run_eval_dirs", status="failed", error_message="No result directories found"
            )
            update_task_status(task_id, "failed", error_message="No result directories found")
            return
        total = len(target_dirs)
        eval_workers = _eval_worker_count(parameters, total)
        statuses = _run_eval_result_dirs(
            task_id=task_id,
            eval_summary=eval_summary,
            target_dirs=target_dirs,
            overwrite=overwrite,
            eval_workers=eval_workers,
            pct_start=0.0,
            pct_end=90.0,
            label="Eval",
        )
        append_task_log(task_id, "Generating summary CSV")
        update_task_progress(task_id, message="Generating Summary.csv / Score.csv", pct=95)
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
        _update_run_metadata(
            task_id,
            parameters,
            task_type="run_eval_dirs",
            extra={
                "evaluation": {
                    "directories_processed": total,
                    "success": len(succeeded),
                    "failed": len(failed),
                    "skipped": len(skipped),
                    "summary_path": result_path,
                    "summary_rows": info.get("summary_rows", 0),
                    "score_rows": info.get("score_rows", 0),
                }
            },
        )
        append_task_log(task_id, f"Done. Output: {result_path}")
        update_task_progress(task_id, message="Eval complete", pct=100)
        _mark_run_status(task_id, parameters, task_type="run_eval_dirs", status="completed", result_path=result_path)
        update_task_status(task_id, "completed", result_path=result_path)
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        _mark_run_status(task_id, parameters, task_type="run_eval_dirs", status="failed", error_message=str(e))
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_build_parquet(task_id: str, parameters: Dict[str, Any]) -> None:
    """Build scene_result parquet from pkl directory."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting build_parquet")
    _mark_run_status(task_id, parameters, task_type="build_parquet", status="running")
    try:
        pkl_archive_to_parquet = _import_catalog_io()
        if pkl_archive_to_parquet is None:
            _mark_run_status(
                task_id, parameters, task_type="build_parquet", status="failed", error_message="perception_catalog_io not available"
            )
            update_task_status(task_id, "failed", error_message="perception_catalog_io not available")
            return
        pkl_dir = parameters.get("pkl_dir")
        if not pkl_dir:
            _mark_run_status(task_id, parameters, task_type="build_parquet", status="failed", error_message="Missing pkl_dir")
            update_task_status(task_id, "failed", error_message="Missing pkl_dir")
            return
        append_task_log(task_id, f"Building parquet from {pkl_dir}")
        update_task_progress(task_id, message=f"Parquet: scanning pkl files in {pkl_dir}", pct=0)
        project_id = parameters.get("project_id")
        job_id = parameters.get("job_id")
        parquet_path = pkl_archive_to_parquet(
            pkl_dir,
            on_progress=_parquet_progress_callback(task_id, pct_start=5, pct_end=95),
            on_skip=lambda path, reason: append_task_log(task_id, f"Parquet skipped {path}: {reason}"),
            project_id=project_id,
            job_id=job_id,
        )
        update_task_progress(task_id, message="Parquet: writing output complete", pct=100)
        update_task_result_summary(task_id, {"job": "build_parquet", "output_path": parquet_path})
        _update_run_metadata(
            task_id,
            parameters,
            task_type="build_parquet",
            extra={
                "parquet": {
                    "enabled": True,
                    "path": parquet_path,
                }
            },
        )
        append_task_log(task_id, f"Done. Output: {parquet_path}")
        _mark_run_status(task_id, parameters, task_type="build_parquet", status="completed", result_path=parquet_path)
        update_task_status(task_id, "completed", result_path=parquet_path)
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        _mark_run_status(task_id, parameters, task_type="build_parquet", status="failed", error_message=str(e))
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


def _is_failed_case_status(case_report: Dict[str, Any]) -> bool:
    """Best-effort failure check for case report payloads."""
    result = case_report.get("result") or {}
    status = (result.get("status") or case_report.get("status") or "").strip().lower()
    return status in {"failed", "failure", "error", "timed_out", "timeout", "canceled", "cancelled", "aborted"}


def _summarize_suite_reports(suite_rows: Any, *, limit: int = 10) -> list[Dict[str, Any]]:
    """Normalize suite rows into a compact summary suitable for task result_summary."""
    normalized = []
    for row in suite_rows or []:
        normalized.append(
            {
                "suite_name": row.get("name", ""),
                "total": int(row.get("all", 0) or 0),
                "success": int(row.get("success", 0) or 0),
                "failed": int(row.get("fail", 0) or 0),
                "canceled": int(row.get("cancel", 0) or 0),
                "simulation": row.get("simulation", ""),
                "url": row.get("url", ""),
            }
        )
    normalized.sort(key=lambda item: (-item["failed"], item["suite_name"]))
    return normalized[:limit]


def _suite_case_totals(suite_rows: Any) -> Dict[str, int]:
    """Aggregate totals from full suite rows."""
    totals = {"total": 0, "success": 0, "failed": 0, "canceled": 0}
    for row in suite_rows or []:
        totals["total"] += int(row.get("all", 0) or 0)
        totals["success"] += int(row.get("success", 0) or 0)
        totals["failed"] += int(row.get("fail", 0) or 0)
        totals["canceled"] += int(row.get("cancel", 0) or 0)
    return totals


def _extract_failed_case_details(case_reports: Any, *, limit: int = 12) -> list[Dict[str, Any]]:
    """Return a compact list of failed cases for UI/log display."""
    failed = []
    for report in case_reports or []:
        if not _is_failed_case_status(report):
            continue
        failed.append(
            {
                "scenario_name": ((report.get("scenario") or {}).get("display_name", "")),
                "suite_name": ((report.get("suite") or {}).get("display_name", "")),
                "status": report.get("status", ""),
                "fail_message": report.get("fail_message", ""),
                "failure_cause_labels": report.get("failure_cause_labels", []),
                "archive_log_id": (((report.get("logs") or {}).get("simulation_archive") or {}).get("id", "")),
                "result_json_log_id": (((report.get("logs") or {}).get("simulation_result_json") or {}).get("id", "")),
            }
        )
    failed.sort(key=lambda item: (item["suite_name"], item["scenario_name"], item["fail_message"]))
    return failed[:limit]


def _extract_git_target_from_report(report: Dict[str, Any]) -> str:
    """Compact branch/tag label from evaluator report metadata."""
    source = ((report.get("event") or {}).get("source") or {})
    git_ref = str(source.get("git_ref") or "").strip()
    if git_ref.startswith("refs/heads/"):
        return git_ref[len("refs/heads/"):]
    if git_ref.startswith("refs/tags/"):
        return git_ref[len("refs/tags/"):]
    return git_ref or str(source.get("git_sha") or "").strip()[:12] or ""


def _extract_job_title_from_report(report: Dict[str, Any]) -> str:
    """Prefer evaluator description for display title, with a readable fallback."""
    description = str(report.get("description") or "").strip()
    if description:
        return description
    started_like = report.get("started_at") or report.get("scheduled_at") or report.get("finished_at")
    return f"no description ({started_like or 'unknown start'})"


def _extract_catalog_url_from_report(report: Dict[str, Any]) -> str:
    """Best-effort catalog URL matching the recent evaluator jobs list."""
    catalog = report.get("catalog") or {}
    direct_url = str(
        catalog.get("web_url")
        or catalog.get("url")
        or catalog.get("catalog_url")
        or ""
    ).strip()
    if direct_url:
        return direct_url
    project_id = str(report.get("project_id") or "").strip()
    catalog_id = str(catalog.get("catalog_id") or catalog.get("id") or "").strip()
    if project_id and catalog_id:
        return f"https://evaluation.tier4.jp/evaluation/vehicle_catalogs/{catalog_id}?project_id={project_id}"
    return ""


def _extract_source_metadata_from_report(report: Dict[str, Any]) -> Dict[str, str]:
    """Best-effort source metadata for local run rendering without refetching."""
    source = ((report.get("event") or {}).get("source") or {})
    git_url = str(source.get("git_web_url") or source.get("git_url") or "").strip()
    return {
        "title": _extract_job_title_from_report(report),
        "target": _extract_git_target_from_report(report),
        "git_sha": str(source.get("git_sha") or "").strip(),
        "git_ref_url": str(source.get("git_ref_url") or "").strip(),
        "git_commit_url": str(source.get("git_commit_url") or "").strip(),
        "source_url": git_url,
        "source_repo_label": git_url.rstrip("/").split("/")[-1] if git_url else "",
    }


def _build_evaluator_result_summary(
    *,
    job_id: str,
    report_url: str,
    evaluator_status: str,
    final_report: Dict[str, Any],
    suite_rows: Any = None,
    failed_cases: Any = None,
) -> Dict[str, Any]:
    """Build a compact evaluator summary that the task detail UI can render."""
    build = final_report.get("build") or {}
    test = final_report.get("test") or {}
    available = test.get("available_case_results") or test.get("case_results") or {}
    case_totals = _suite_case_totals(suite_rows)
    source_meta = _extract_source_metadata_from_report(final_report)
    if not any(case_totals.values()):
        case_totals = {
            "total": int(available.get("total_count", 0) or 0),
            "success": int(available.get("success_count", 0) or 0),
            "failed": int(available.get("failure_count", 0) or 0),
            "canceled": int(available.get("cancellation_count", 0) or 0),
        }
    return {
        "evaluator_job_id": job_id,
        "evaluator_report_url": report_url,
        "evaluator_status": evaluator_status,
        "evaluator_scheduled_by": final_report.get("scheduled_by", ""),
        "evaluator_catalog_id": ((final_report.get("catalog") or {}).get("id") or ""),
        "evaluator_catalog_name": ((final_report.get("catalog") or {}).get("display_name") or ""),
        "evaluator_catalog_version_id": ((final_report.get("catalog") or {}).get("version_id") or ""),
        "evaluator_catalog_url": _extract_catalog_url_from_report(final_report),
        "evaluator_title": source_meta.get("title", ""),
        "evaluator_target": source_meta.get("target", ""),
        "evaluator_git_sha": source_meta.get("git_sha", ""),
        "evaluator_git_ref_url": source_meta.get("git_ref_url", ""),
        "evaluator_git_commit_url": source_meta.get("git_commit_url", ""),
        "evaluator_source_url": source_meta.get("source_url", ""),
        "evaluator_source_repo_label": source_meta.get("source_repo_label", ""),
        "evaluator_build_status": build.get("status", ""),
        "evaluator_test_status": test.get("status", ""),
        "evaluator_fail_message": final_report.get("fail_message", ""),
        "evaluator_case_totals": case_totals,
        "evaluator_suites": _summarize_suite_reports(suite_rows),
        "evaluator_failed_cases": _extract_failed_case_details(failed_cases),
    }


def _fetch_evaluator_context(
    *,
    project_id: str,
    job_id: str,
    environment: str,
) -> Dict[str, Any]:
    """Best-effort evaluator metadata for tasks that start from an existing evaluator job."""
    if not project_id or not job_id:
        return {}
    try:
        from lib import evaluator_api

        os.environ["AUTH_PROFILE"] = environment or "default"
        api = evaluator_api.EvaluationRunAPI()
        report = api.get_job_status(project_id, job_id)
        status = evaluator_api.extract_job_status(report)
        build = report.get("build") or {}
        test = report.get("test") or {}
        available = test.get("available_case_results") or test.get("case_results") or {}
        source_meta = _extract_source_metadata_from_report(report)
        return {
            "job_id": job_id,
            "report_url": evaluator_api.get_job_report_url(project_id, job_id),
            "status": status,
            "scheduled_by": str(report.get("scheduled_by") or "").strip(),
            "catalog_id": str(((report.get("catalog") or {}).get("id") or "")).strip(),
            "catalog_name": str(((report.get("catalog") or {}).get("display_name") or "")).strip(),
            "catalog_version_id": (report.get("catalog") or {}).get("version_id"),
            "catalog_url": _extract_catalog_url_from_report(report),
            "title": source_meta.get("title", ""),
            "target": source_meta.get("target", ""),
            "git_sha": source_meta.get("git_sha", ""),
            "git_ref_url": source_meta.get("git_ref_url", ""),
            "git_commit_url": source_meta.get("git_commit_url", ""),
            "source_url": source_meta.get("source_url", ""),
            "source_repo_label": source_meta.get("source_repo_label", ""),
            "build_status": str(build.get("status") or "").strip(),
            "test_status": str(test.get("status") or "").strip(),
            "fail_message": str(report.get("fail_message") or "").strip(),
            "case_totals": {
                "total": int(available.get("total_count", 0) or 0),
                "success": int(available.get("success_count", 0) or 0),
                "failed": int(available.get("failure_count", 0) or 0),
                "canceled": int(available.get("cancellation_count", 0) or 0),
            },
        }
    except Exception:
        return {}


def job_download_results(task_id: str, parameters: Dict[str, Any]) -> None:
    """Download job results (archives or result JSON) and extract/organize. Requires auth."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting download_results")
    _mark_run_status(
        task_id,
        parameters,
        task_type="download_results",
        status="running",
        create_missing=True,
    )
    try:
        from lib import download_core  # noqa: F401
        output_path = parameters.get("output_path")
        project_id = parameters.get("project_id")
        job_id = parameters.get("job_id")
        environment = str(parameters.get("environment") or "default").strip() or "default"
        suite_id = parameters.get("suite_id")
        suite_ids = parameters.get("suite_ids")  # optional list
        download_type = parameters.get("download_type", "archives")  # archives | result_json
        phase = parameters.get("phase", "first")
        skip_large_file = parameters.get("skip_large_file", False)
        large_file_mb = float(parameters.get("large_file_mb", 50.0))
        keep_zip_files = parameters.get("keep_zip_files", False)
        if not all([output_path, project_id, job_id]):
            _mark_run_status(
                task_id,
                parameters,
                task_type="download_results",
                status="failed",
                error_message="Missing output_path, project_id, or job_id",
                create_missing=True,
            )
            update_task_status(task_id, "failed", error_message="Missing output_path, project_id, or job_id")
            return
        evaluator_context = _fetch_evaluator_context(project_id=project_id, job_id=job_id, environment=environment)
        if evaluator_context:
            _update_run_metadata(
                task_id,
                parameters,
                task_type="download_results",
                create_missing=True,
                extra={"evaluator": evaluator_context},
            )
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
        _update_run_metadata(
            task_id,
            parameters,
            task_type="download_results",
            create_missing=True,
            extra={
                "download": {
                    "mode": "download_results",
                    "total": total_attempted,
                    "success": success_count,
                    "failed": failure_count,
                    "rows": rows[:100],
                    "download_type": download_type,
                    "phase": phase,
                    "skip_large_file": bool(skip_large_file),
                    "large_file_mb": large_file_mb,
                    "keep_zip_files": bool(keep_zip_files),
                }
            },
        )
        append_task_log(task_id, "Download and extract completed")
        if success_count == 0 and failure_count > 0:
            err_msg = f"Download completed with {failure_count} failures. See task log for details."
            _mark_run_status(
                task_id,
                parameters,
                task_type="download_results",
                status="failed",
                result_path=output_path,
                error_message=err_msg,
            )
            update_task_status(task_id, "failed", result_path=output_path, error_message=err_msg)
        else:
            _mark_run_status(
                task_id,
                parameters,
                task_type="download_results",
                status="completed",
                result_path=output_path,
            )
            update_task_status(task_id, "completed", result_path=output_path)
    except ImportError:
        _mark_run_status(
            task_id,
            parameters,
            task_type="download_results",
            status="failed",
            error_message="Download worker not available: lib.download_core not implemented",
            create_missing=True,
        )
        update_task_status(
            task_id,
            "failed",
            error_message="Download worker not available: lib.download_core not implemented",
        )
    except NotImplementedError as e:
        _mark_run_status(task_id, parameters, task_type="download_results", status="failed", error_message=str(e))
        update_task_status(task_id, "failed", error_message=str(e))
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        _mark_run_status(task_id, parameters, task_type="download_results", status="failed", error_message=str(e))
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_download_scenarios(task_id: str, parameters: Dict[str, Any]) -> None:
    """Download scenarios from job to output_dir. Requires auth."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting download_scenarios")
    _mark_run_status(
        task_id,
        parameters,
        task_type="download_scenarios",
        status="running",
        create_missing=True,
    )
    try:
        from lib import download_core  # noqa: F401
        output_dir = parameters.get("output_dir") or parameters.get("output_path")
        project_id = parameters.get("project_id")
        job_id = parameters.get("job_id")
        environment = str(parameters.get("environment") or "default").strip() or "default"
        suite_id = parameters.get("suite_id")
        suite_ids = parameters.get("suite_ids")
        overwrite = parameters.get("overwrite", False)
        scenario_name_filter = parameters.get("scenario_name_filter")
        selected_ids = parameters.get("selected_ids")
        if not all([output_dir, project_id, job_id]):
            _mark_run_status(
                task_id,
                parameters,
                task_type="download_scenarios",
                status="failed",
                error_message="Missing output_dir, project_id, or job_id",
                create_missing=True,
            )
            update_task_status(task_id, "failed", error_message="Missing output_dir, project_id, or job_id")
            return
        evaluator_context = _fetch_evaluator_context(project_id=project_id, job_id=job_id, environment=environment)
        if evaluator_context:
            _update_run_metadata(
                task_id,
                parameters,
                task_type="download_scenarios",
                create_missing=True,
                extra={"evaluator": evaluator_context},
            )
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
        _update_run_metadata(
            task_id,
            parameters,
            task_type="download_scenarios",
            create_missing=True,
            extra={
                "scenario_download": {
                    "total": total_attempted,
                    "success": success_count,
                    "failed": failure_count,
                    "overwrite": bool(overwrite),
                    "scenario_name_filter": str(scenario_name_filter or "").strip(),
                    "selected_ids": list(selected_ids or []),
                    "rows": rows[:100],
                }
            },
        )
        append_task_log(task_id, "Download scenarios completed")
        if failure_count > 0:
            err_msg = f"Download completed with {failure_count} failures. See task log for details."
            _mark_run_status(
                task_id,
                parameters,
                task_type="download_scenarios",
                status="failed",
                result_path=output_dir,
                error_message=err_msg,
            )
            update_task_status(task_id, "failed", result_path=output_dir, error_message=err_msg)
        else:
            _mark_run_status(
                task_id,
                parameters,
                task_type="download_scenarios",
                status="completed",
                result_path=output_dir,
            )
            update_task_status(task_id, "completed", result_path=output_dir)
    except ImportError:
        _mark_run_status(
            task_id,
            parameters,
            task_type="download_scenarios",
            status="failed",
            error_message="Download worker not available: lib.download_core not implemented",
            create_missing=True,
        )
        update_task_status(
            task_id,
            "failed",
            error_message="Download worker not available: lib.download_core not implemented",
        )
    except NotImplementedError as e:
        _mark_run_status(task_id, parameters, task_type="download_scenarios", status="failed", error_message=str(e))
        update_task_status(task_id, "failed", error_message=str(e))
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        _mark_run_status(task_id, parameters, task_type="download_scenarios", status="failed", error_message=str(e))
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_download_and_eval(task_id: str, parameters: Dict[str, Any]) -> None:
    """Download results, then run eval and parquet generation. Stops on download failure."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting download_and_eval combined workflow")
    _mark_run_status(
        task_id,
        parameters,
        task_type="download_and_eval",
        status="running",
        create_missing=True,
    )
    try:
        from lib import download_core
        output_path = parameters.get("output_path")
        project_id = parameters.get("project_id")
        job_id = parameters.get("job_id")
        environment = str(parameters.get("environment") or "default").strip() or "default"
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
            _mark_run_status(
                task_id,
                parameters,
                task_type="download_and_eval",
                status="failed",
                error_message="Missing output_path, project_id, or job_id",
                create_missing=True,
            )
            update_task_status(task_id, "failed", error_message="Missing output_path, project_id, or job_id")
            return
        evaluator_context = _fetch_evaluator_context(project_id=project_id, job_id=job_id, environment=environment)
        if evaluator_context:
            _update_run_metadata(
                task_id,
                parameters,
                task_type="download_and_eval",
                create_missing=True,
                extra={"evaluator": evaluator_context},
            )
        
        def on_progress(msg: str) -> None:
            append_task_log(task_id, msg)
            match = re.search(r"(\d+)\s*/\s*(\d+)", msg)
            pct = None
            if match:
                n, m = int(match.group(1)), max(1, int(match.group(2)))
                ratio = n / m
                if msg.startswith("Eval:"):
                    pct = 60.0 + ratio * 25.0
                elif msg.startswith("Parquet:"):
                    pct = 85.0 + ratio * 13.0
                elif msg.startswith("Downloading"):
                    pct = ratio * 60.0
            if pct is None:
                if msg.startswith("Download complete"):
                    pct = 60.0
                elif msg.startswith("Generating parquet"):
                    pct = 85.0
            if pct is None:
                update_task_progress(task_id, message=msg)
            else:
                update_task_progress(task_id, message=msg, pct=pct)

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
            eval_workers=_eval_worker_count(parameters, 10_000),
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
        _update_run_metadata(
            task_id,
            parameters,
            task_type="download_and_eval",
            create_missing=True,
            extra={
                "download": {
                    "mode": "download_and_eval",
                    **(result.get("download_summary", {}) or {}),
                    "download_type": download_type,
                    "phase": phase,
                    "skip_large_file": bool(skip_large_file),
                    "large_file_mb": large_file_mb,
                    "keep_zip_files": bool(keep_zip_files),
                },
                "evaluation": {
                    **(result.get("eval_summary", {}) or {}),
                    "enabled": bool(run_eval),
                    "recursive": bool(eval_recursive),
                    "overwrite": bool(eval_overwrite),
                },
                "parquet": {
                    "enabled": bool(generate_parquet),
                    "path": result.get("parquet_path", ""),
                },
                "errors": list(result.get("errors", []) or []),
            },
        )
        
        if not result.get("download_success"):
            err_msg = result.get("errors", ["Download failed"])[0]
            append_task_log(task_id, f"Stopped: {err_msg}")
            _mark_run_status(
                task_id,
                parameters,
                task_type="download_and_eval",
                status="failed",
                result_path=output_path,
                error_message=err_msg,
            )
            update_task_status(task_id, "failed", result_path=output_path, error_message=err_msg)
        elif result.get("errors"):
            # Partial success with some errors
            errs = "; ".join(result["errors"][:5])
            append_task_log(task_id, f"Completed with errors: {errs}")
            _mark_run_status(
                task_id,
                parameters,
                task_type="download_and_eval",
                status="completed",
                result_path=output_path,
                error_message=errs,
            )
            update_task_status(task_id, "completed", result_path=output_path)
        else:
            append_task_log(task_id, "Download and eval completed successfully")
            _mark_run_status(
                task_id,
                parameters,
                task_type="download_and_eval",
                status="completed",
                result_path=output_path,
            )
            update_task_status(task_id, "completed", result_path=output_path)
            
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        _mark_run_status(task_id, parameters, task_type="download_and_eval", status="failed", error_message=str(e))
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def _write_release_metadata_file(path: Path, metadata: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(metadata, fh, allow_unicode=True, sort_keys=False)
    return path


def _build_devops_trend_summary_from_suites(rows: list[dict[str, Any]]) -> Dict[str, Any]:
    suite_results: dict[str, dict[str, int]] = {}
    for row in rows or []:
        suite_name = str(row.get("name") or row.get("suite_name") or row.get("simulation") or "suite").strip()
        total = int(row.get("all", 0) or row.get("total", 0) or 0)
        passed = int(row.get("success", 0) or row.get("passed", 0) or 0)
        if total <= 0:
            failed = int(row.get("fail", 0) or row.get("failed", 0) or 0)
            canceled = int(row.get("cancel", 0) or row.get("canceled", 0) or 0)
            total = passed + failed + canceled
        if total <= 0:
            continue
        suite_results[suite_name] = {"passed": passed, "total": total}
    if not suite_results:
        return {"DevOps": {}}

    try:
        from perception_catalog_analyzer.path import DEVOPS_MAPPING_PATH

        with Path(DEVOPS_MAPPING_PATH).open("r", encoding="utf-8") as fh:
            category_mapping = yaml.safe_load(fh) or {}
    except Exception:
        category_mapping = {}

    if not isinstance(category_mapping, dict) or not category_mapping:
        return {"DevOps": {"Suite pass rate": suite_results}}

    mapped: Dict[str, Any] = {}
    matched_suites: set[str] = set()
    for major_category, mid_categories in category_mapping.items():
        if not isinstance(mid_categories, dict):
            continue
        major_payload: Dict[str, Any] = {}
        for mid_category, sub_categories in mid_categories.items():
            if not isinstance(sub_categories, dict):
                continue
            mid_payload: Dict[str, Any] = {}
            for sub_category, suite_names in sub_categories.items():
                if not isinstance(suite_names, list):
                    continue
                passed = 0
                total = 0
                for suite_name in suite_names:
                    result = suite_results.get(str(suite_name))
                    if not result:
                        continue
                    matched_suites.add(str(suite_name))
                    passed += int(result.get("passed", 0) or 0)
                    total += int(result.get("total", 0) or 0)
                mid_payload[str(sub_category)] = {"passed": passed, "total": total}
            if mid_payload:
                major_payload[str(mid_category)] = mid_payload
        if major_payload:
            mapped[str(major_category)] = major_payload

    unmatched = {
        suite_name: result
        for suite_name, result in suite_results.items()
        if suite_name not in matched_suites
    }
    if unmatched:
        mapped.setdefault("その他", {})["未分類"] = unmatched

    return {"DevOps": mapped}


def _write_devops_trend_summary(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    summary_payload = _build_devops_trend_summary_from_suites(rows)
    if not summary_payload.get("DevOps"):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)
    return path


def _suite_rows_from_existing_devops_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    devops = payload.get("DevOps") if isinstance(payload, dict) else {}
    if not isinstance(devops, dict):
        return []
    suite_pass_rate = devops.get("Suite pass rate")
    if not isinstance(suite_pass_rate, dict):
        return []
    rows: list[dict[str, Any]] = []
    for suite_name, result in suite_pass_rate.items():
        if not isinstance(result, dict):
            continue
        rows.append(
            {
                "suite_name": str(suite_name),
                "success": int(result.get("passed", 0) or 0),
                "total": int(result.get("total", 0) or 0),
            }
        )
    return rows


def _has_release_download_artifacts(path: Path) -> bool:
    return any(path.rglob("scene_result.pkl")) or any(path.rglob("*.pkl.z"))


def _find_release_parquet(path: Path) -> Path | None:
    current = path / "current.parquet"
    if current.exists():
        return current
    for parquet in sorted(path.glob("*.parquet"), key=lambda p: p.name.lower()):
        return parquet
    return None


def _build_release_analysis_artifacts(
    *,
    task_id: str,
    project_id: str,
    job_id: str,
    role: str,
    output_path: Path,
    phase: str,
    run_eval: bool = False,
    skip_large_file: bool = _RELEASE_SKIP_LARGE_FILE,
    large_file_mb: float = _RELEASE_LARGE_FILE_MB,
    progress_start: float = 48.0,
    progress_end: float = 78.0,
) -> Dict[str, Any]:
    """Create the normal app analysis files for a release job."""
    from lib import download_core

    eval_summary = _import_eval_summary()
    pkl_archive_to_parquet = _import_catalog_io()
    output_path.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "path": str(output_path),
        "download": {},
        "eval": {},
        "parquet_path": "",
        "warnings": [],
    }
    effective_skip_large_file = _RELEASE_SKIP_LARGE_FILE or bool(skip_large_file)
    effective_large_file_mb = float(large_file_mb or _RELEASE_LARGE_FILE_MB)

    progress_span = max(0.0, progress_end - progress_start)
    download_end = progress_start + progress_span * 0.55
    eval_end = progress_start + progress_span * 0.90
    existing_parquet = _find_release_parquet(output_path)

    def _on_progress(msg: str) -> None:
        append_task_log(task_id, f"{role}: {msg}")
        progress_msg = f"{role}: {msg}"
        pct = progress_start
        match = re.search(r"Downloading\s+(\d+)\s*/\s*(\d+)", msg)
        if match:
            current = int(match.group(1))
            total = max(1, int(match.group(2)))
            pct = progress_start + ((current - 1) / total) * max(0.0, download_end - progress_start)
        elif "Extracting" in msg or "Organizing" in msg:
            pct = download_end
        update_task_progress(task_id, message=progress_msg, pct=min(download_end, pct))

    def _on_warning(msg: str) -> None:
        result["warnings"].append(msg)
        append_task_log(task_id, f"WARNING: {role}: {msg}")

    if existing_parquet or _has_release_download_artifacts(output_path):
        append_task_log(task_id, f"{role}: using existing downloaded artifacts in {output_path}")
        update_task_progress(task_id, message=f"{role}: using existing downloaded artifacts", pct=download_end)
        failure_count = 0
        total_attempted = 0
        success_count = 0
        rows: list[dict[str, Any]] = []
    else:
        if not job_id:
            raise RuntimeError(f"{role}: no local artifacts found and no evaluator job id is available for download.")
        update_task_progress(task_id, message=f"{role}: finding downloadable case logs", pct=progress_start)
        failure_count, total_attempted, rows = download_core.run_download_results(
            project_id=project_id,
            job_id=job_id,
            suite_id=None,
            output_path=str(output_path),
            download_type="archives",
            phase=phase,
            skip_large_file=effective_skip_large_file,
            large_file_mb=effective_large_file_mb,
            keep_zip_files=False,
            suite_ids=None,
            on_progress=_on_progress,
            on_warning=_on_warning,
        )
        success_count = total_attempted - failure_count
        if success_count <= 0:
            raise RuntimeError(f"{role}: download produced no successful case artifacts.")
    result["download"] = {
        "total": total_attempted,
        "success": success_count,
        "failed": failure_count,
        "skip_large_file": effective_skip_large_file,
        "large_file_mb": effective_large_file_mb,
        "rows": rows[:100],
    }

    if run_eval and eval_summary and not existing_parquet:
        target_dirs = eval_summary.find_eval_result_dirs(str(output_path), recursive=True)
        total = len(target_dirs)
        summary_csv = output_path / "Summary.csv"
        score_csv = output_path / "Score.csv"
        if target_dirs and summary_csv.exists() and score_csv.exists():
            append_task_log(task_id, f"{role}: Summary.csv / Score.csv already exist; skipping eval")
            update_task_progress(task_id, message=f"{role}: existing Summary.csv / Score.csv found", pct=eval_end)
            statuses = []
            result["eval"] = {
                "directories_processed": total,
                "success": 0,
                "failed": 0,
                "skipped": total,
                "summary_path": str(summary_csv),
                "summary_rows": 0,
                "score_rows": 0,
            }
        elif target_dirs:
            statuses = _run_eval_result_dirs(
                task_id=task_id,
                eval_summary=eval_summary,
                target_dirs=target_dirs,
                overwrite=False,
                eval_workers=_eval_worker_count({}, total),
                pct_start=download_end,
                pct_end=eval_end,
                label=f"{role}: eval_result",
            )
        else:
            update_task_progress(task_id, message=f"{role}: no eval_result directories found", pct=eval_end)
            statuses = []
        if target_dirs and not result["eval"]:
            update_task_progress(task_id, message=f"{role}: generating Summary.csv / Score.csv", pct=eval_end)
            csv_info = eval_summary.generate_summary_and_score_csv(str(output_path))
            result["eval"] = {
                "directories_processed": len(target_dirs),
                "success": sum(1 for item in statuses if item.get("status") == "success"),
                "failed": sum(1 for item in statuses if item.get("status") == "failed"),
                "skipped": sum(1 for item in statuses if item.get("status") == "skipped"),
                "summary_path": csv_info.get("summary_path", ""),
                "summary_rows": csv_info.get("summary_rows", 0),
                "score_rows": csv_info.get("score_rows", 0),
            }
        elif not result["eval"]:
            result["eval"] = {
                "directories_processed": 0,
                "success": 0,
                "failed": 0,
                "skipped": 0,
            }
    elif not run_eval:
        append_task_log(task_id, f"{role}: skipping eval; parquet is sufficient for release PDF generation")
        result["eval"] = {"enabled": False, "reason": "release_pdf_uses_parquet"}
    elif existing_parquet:
        append_task_log(task_id, f"{role}: skipping eval because parquet already exists")
        result["eval"] = {"enabled": False, "reason": "existing_parquet"}

    existing_parquet = _find_release_parquet(output_path)
    if existing_parquet:
        append_task_log(task_id, f"{role}: existing parquet found: {existing_parquet}")
        result["parquet_path"] = str(existing_parquet)
        update_task_progress(task_id, message=f"{role}: existing parquet found", pct=progress_end)
    elif pkl_archive_to_parquet:
        try:
            update_task_progress(task_id, message=f"{role}: generating parquet", pct=eval_end)
            result["parquet_path"] = pkl_archive_to_parquet(
                str(output_path),
                on_progress=_parquet_progress_callback(
                    task_id,
                    prefix=f"{role}: parquet",
                    pct_start=eval_end,
                    pct_end=99,
                ),
                on_skip=lambda path, reason: append_task_log(
                    task_id,
                    f"WARNING: {role}: parquet skipped {path}: {reason}",
                ),
                project_id=project_id,
                job_id=job_id,
            ) or ""
            update_task_progress(task_id, message=f"{role}: parquet generated", pct=99)
        except Exception as exc:
            warning = f"Parquet generation failed: {exc}"
            result["warnings"].append(warning)
            append_task_log(task_id, f"WARNING: {role}: {warning}")

    append_task_log(
        task_id,
        (
            f"{role}: analysis artifacts ready at {output_path} "
            f"({success_count}/{total_attempted} downloads)"
        ),
    )
    return result


def job_run_release_specsheet_workflow(task_id: str, parameters: Dict[str, Any]) -> None:
    """Schedule the standard release evaluator jobs, process them as app-native runs, then build a release specsheet."""
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting release specsheet workflow")
    _mark_run_status(
        task_id,
        parameters,
        task_type="run_release_specsheet_workflow",
        status="running",
        create_missing=True,
    )
    try:
        from lib import evaluator_api
        from lib.specsheet_report import (
            DEFAULT_SPECSHEET_LABELS,
            DEFAULT_SPECSHEET_TOPIC,
            generate_specsheet_pdf,
            resolve_specsheet_topic_name,
        )

        project_id = str(parameters.get("project_id") or "").strip()
        target_name = str(parameters.get("target_name") or "").strip()
        output_path = str(parameters.get("output_path") or "").strip()
        environment = str(parameters.get("environment") or "default").strip() or "default"
        is_tag = bool(parameters.get("is_tag", False))
        metadata = parameters.get("trend_metadata") if isinstance(parameters.get("trend_metadata"), dict) else {}
        version = str(parameters.get("version") or metadata.get("pilot_auto_version") or "").strip()
        topic = str(parameters.get("topic") or metadata.get("topic_name") or DEFAULT_SPECSHEET_TOPIC).strip()
        description = str(parameters.get("description") or target_name or "").strip()
        poll_interval = float(parameters.get("poll_interval", 60.0))
        max_wait_seconds = _normalize_max_wait_seconds(parameters.get("max_wait_seconds", 0.0))
        analysis_phase = str(
            parameters.get("analysis_phase")
            or "perception.object_recognition.tracking.objects"
        ).strip()
        skip_large_file = _RELEASE_SKIP_LARGE_FILE
        large_file_mb = float(parameters.get("large_file_mb") or _RELEASE_LARGE_FILE_MB)
        labels = parameters.get("labels") or DEFAULT_SPECSHEET_LABELS
        labels = [str(label).strip() for label in labels if str(label).strip()]
        if not labels:
            labels = list(DEFAULT_SPECSHEET_LABELS)

        if not project_id or not target_name or not output_path or not version:
            raise ValueError("Missing project_id, target_name, output_path, or Pilot.Auto version.")
        if "trend" not in [str(tag).strip() for tag in metadata.get("tags", [])]:
            raise ValueError("Release metadata must include tags: [trend].")

        release_root = Path(output_path)
        release_root.mkdir(parents=True, exist_ok=True)
        _write_release_metadata_file(release_root / "metadata.yaml", metadata)
        performance_path = release_root / "performance"
        devops_path = release_root / "devops"
        role_paths = {
            "performance": performance_path,
            "devops": devops_path,
            "planning_test": release_root / "planning_test",
        }
        os.environ["AUTH_PROFILE"] = environment
        os.environ["EVALUATOR_ENVIRONMENT"] = environment

        api = evaluator_api.EvaluationRunAPI()
        optional_catalog_enabled = bool(parameters.get("optional_catalog_enabled", False))
        optional_catalog_id = str(
            parameters.get("optional_catalog_id") or _RELEASE_OPTIONAL_CATALOG_ID
        ).strip()
        optional_integration_id = str(parameters.get("optional_integration_id") or "").strip()
        if optional_catalog_enabled and optional_catalog_id and not optional_integration_id:
            append_task_log(task_id, f"Resolving Planning Test catalog integration: {optional_catalog_id}")
            optional_integration_id = _resolve_active_integration_id(api, project_id, optional_catalog_id)
        jobs = [
            {
                "role": "performance",
                "label": "Performance Test",
                "catalog_id": str(parameters.get("performance_catalog_id") or _RELEASE_PERFORMANCE_CATALOG_ID),
                "integration_id": str(parameters.get("performance_integration_id") or _RELEASE_PERFORMANCE_INTEGRATION_ID),
                "job_id": str(parameters.get("performance_job_id") or "").strip(),
            },
            {
                "role": "devops",
                "label": "Devops Test",
                "catalog_id": str(parameters.get("devops_catalog_id") or _RELEASE_DEVOPS_CATALOG_ID),
                "integration_id": str(parameters.get("devops_integration_id") or _RELEASE_DEVOPS_INTEGRATION_ID),
                "job_id": str(parameters.get("devops_job_id") or "").strip(),
            },
        ]
        if optional_catalog_enabled:
            jobs.append(
                {
                    "role": "planning_test",
                    "label": "Planning Test",
                    "catalog_id": optional_catalog_id,
                    "integration_id": optional_integration_id,
                    "job_id": str(parameters.get("optional_job_id") or "").strip(),
                }
            )
        summary: Dict[str, Any] = {
            "job": "run_release_specsheet_workflow",
            "release_root": str(release_root),
            "version": version,
            "topic": topic,
            "evaluator_jobs": {},
            "analysis_artifacts": {},
            "specsheet_pdf": "",
        }
        update_task_result_summary(task_id, summary)
        update_task_progress(task_id, message="Preparing release evaluator jobs", pct=2)

        for item in jobs:
            schedule_description = f"{description} | {item['label']}"
            item["description"] = schedule_description
            role = str(item["role"])
            local_path = role_paths[role]
            local_ready = _find_release_parquet(local_path) is not None or _has_release_download_artifacts(local_path)
            item["local_artifacts_ready"] = local_ready
            job_id = str(item.get("job_id") or "").strip()
            if job_id:
                append_task_log(task_id, f"Using existing {item['label']}: {job_id}")
                status = "existing"
            elif local_ready:
                append_task_log(task_id, f"Using existing local artifacts for {item['label']}: {local_path}")
                status = "local_artifacts"
            else:
                append_task_log(task_id, f"Scheduling {item['label']}: catalog={item['catalog_id']}")
                result = api.schedule_job(
                    project_id=project_id,
                    catalog_id=item["catalog_id"],
                    integration_id=item["integration_id"],
                    target_name=target_name,
                    suite_ids=None,
                    max_retries=0,
                    description=schedule_description,
                    clean_build=True,
                    debug=False,
                    release=False,
                    record_caret=False,
                    log_expiration_time_in_days=10.0,
                    is_tag=is_tag,
                )
                job_id = str(result.get("job_id") or "").strip()
                if not job_id:
                    raise RuntimeError(f"No job_id returned for {item['label']}.")
                item["job_id"] = job_id
                status = "scheduled"
            report_url = evaluator_api.get_job_report_url(project_id, job_id) if job_id else ""
            summary["evaluator_jobs"][item["role"]] = {
                "job_id": job_id,
                "report_url": report_url,
                "catalog_id": item["catalog_id"],
                "integration_id": item["integration_id"],
                "status": status,
                "description": schedule_description,
            }
            if status == "scheduled":
                append_task_log(task_id, f"Scheduled {item['label']}: {job_id}")
            update_task_result_summary(task_id, summary)

        wait_span = 40.0 / max(1, len(jobs))
        for idx, item in enumerate(jobs, start=1):
            job_id = str(item["job_id"])
            label = str(item["label"])
            base_pct = 5 + (idx - 1) * wait_span
            if not job_id and item.get("local_artifacts_ready"):
                append_task_log(task_id, f"Skipping evaluator wait for {label}; local artifacts already exist.")
                summary["evaluator_jobs"][item["role"]]["status"] = "local_artifacts"
                update_task_progress(task_id, message=f"{label}: using local artifacts", pct=base_pct + wait_span - 2.0)
                update_task_result_summary(task_id, summary)
                continue

            def _on_check(status: str, elapsed: float, *, role: str = str(item["role"]), pct_base: float = base_pct) -> None:
                pct = _wait_progress_pct(
                    elapsed=elapsed,
                    timeout_seconds=max_wait_seconds,
                    pct_start=pct_base,
                    pct_end=pct_base + wait_span - 2.0,
                )
                summary["evaluator_jobs"][role]["status"] = status
                update_task_progress(
                    task_id,
                    message=f"{label}: {status} ({elapsed / 3600:.1f}h elapsed)",
                    pct=pct,
                )
                update_task_result_summary(task_id, summary)

            append_task_log(task_id, f"Waiting for {label}: {job_id}")
            final_report = api.wait_for_job_completion(
                project_id=project_id,
                job_id=job_id,
                poll_interval=poll_interval,
                max_wait_seconds=max_wait_seconds,
                on_check=_on_check,
            )
            status = evaluator_api.extract_job_status(final_report)
            summary["evaluator_jobs"][item["role"]]["status"] = status
            append_task_log(task_id, f"{label} completed with status: {status}")
            try:
                suite_rows = api.get_suite_summary(project_id, job_id, use_available_case_results=True)
            except Exception as exc:
                append_task_log(task_id, f"WARNING: Could not fetch suite summary for {label}: {exc}")
                suite_rows = []
            item["suite_rows"] = suite_rows
            summary["evaluator_jobs"][item["role"]]["suite_count"] = len(suite_rows)
            update_task_result_summary(task_id, summary)

        update_task_progress(task_id, message="Building normal CSV/parquet analysis artifacts", pct=48)
        artifact_span = 30.0 / max(1, len(jobs))
        for artifact_idx, item in enumerate(jobs):
            role = str(item["role"])
            analysis_path = role_paths[role]
            artifact_summary = _build_release_analysis_artifacts(
                task_id=task_id,
                project_id=project_id,
                job_id=str(item["job_id"]),
                role=role,
                output_path=analysis_path,
                phase=analysis_phase,
                run_eval=bool(parameters.get("run_eval", False)),
                skip_large_file=skip_large_file,
                large_file_mb=large_file_mb,
                progress_start=48 + (artifact_span * artifact_idx),
                progress_end=48 + (artifact_span * (artifact_idx + 1)),
            )
            summary["analysis_artifacts"][role] = artifact_summary
            update_task_result_summary(task_id, summary)

            child_params = {
                **parameters,
                "output_path": str(analysis_path),
                "catalog_id": item["catalog_id"],
                "integration_id": item["integration_id"],
                "job_id": item["job_id"],
                "download_type": "archives",
                "phase": analysis_phase,
                "skip_large_file": skip_large_file,
                "large_file_mb": large_file_mb,
                "run_eval": bool(parameters.get("run_eval", False)),
                "generate_parquet": True,
                "eval_recursive": bool(parameters.get("run_eval", False)),
            }
            _mark_run_status(
                task_id,
                child_params,
                task_type="run_release_specsheet_workflow",
                status="completed",
                result_path=str(analysis_path),
                create_missing=True,
                extra={
                    "release_specsheet": {
                        "root": str(release_root),
                        "role": role,
                        "metadata": metadata,
                    },
                    "evaluator": {
                        "job_id": str(item["job_id"]),
                        "report_url": summary["evaluator_jobs"][role].get("report_url", ""),
                        "status": summary["evaluator_jobs"][role].get("status", ""),
                        "catalog_id": item["catalog_id"],
                        "integration_id": item["integration_id"],
                        "target_name": target_name,
                        "description": str(item.get("description") or ""),
                        "title": str(item.get("description") or ""),
                    },
                    "download": {
                        **artifact_summary.get("download", {}),
                        "mode": "release_specsheet",
                        "download_type": "archives",
                        "phase": analysis_phase,
                    },
                    "evaluation": {
                        **artifact_summary.get("eval", {}),
                        "enabled": bool(parameters.get("run_eval", False)),
                        "recursive": bool(parameters.get("run_eval", False)),
                    },
                    "parquet": {
                        "enabled": True,
                        "path": artifact_summary.get("parquet_path", ""),
                    },
                },
            )

        detected_topic, detected_topics = resolve_specsheet_topic_name(
            performance_path,
            topic,
            fallback_topic=DEFAULT_SPECSHEET_TOPIC,
        )
        if detected_topic != topic:
            append_task_log(
                task_id,
                (
                    f"Using detected specsheet topic {detected_topic} instead of requested topic {topic} "
                    f"(detected: {', '.join(detected_topics) if detected_topics else 'none'})"
                ),
            )
            topic = detected_topic
            summary["topic"] = topic
            summary["detected_topics"] = detected_topics
            update_task_result_summary(task_id, summary)

        update_task_progress(task_id, message="Writing release trend summaries", pct=78)
        write_trend_metadata(devops_path, metadata)
        devops_job = next(item for item in jobs if item["role"] == "devops")
        devops_summary_target = devops_path / "resources" / "summary.json"
        devops_suite_rows = list(devops_job.get("suite_rows") or [])
        if not devops_suite_rows:
            existing_suite_rows = _suite_rows_from_existing_devops_summary(devops_summary_target)
            if existing_suite_rows:
                append_task_log(task_id, "Rebuilding DevOps trend summary from existing suite pass-rate rows.")
                devops_suite_rows = existing_suite_rows
        devops_summary_path = _write_devops_trend_summary(devops_summary_target, devops_suite_rows)
        if devops_summary_path is None and devops_summary_target.exists():
            devops_summary_path = devops_summary_target
            append_task_log(task_id, f"Using existing DevOps trend summary: {devops_summary_path}")
        if devops_summary_path is None:
            append_task_log(task_id, "WARNING: DevOps trend summary had no suite pass-rate rows.")
        else:
            append_task_log(task_id, f"DevOps trend summary written: {devops_summary_path}")

        update_task_progress(task_id, message="Generating app-native release specsheet", pct=82)
        specsheet_pdf, generated = generate_specsheet_pdf(
            performance_path,
            project_id=project_id,
            version=version,
            labels=labels,
            topic_name=topic,
            include_trend=True,
            trend_metadata=metadata,
            force=bool(parameters.get("overwrite", True)),
            progress_callback=lambda msg: append_task_log(task_id, f"specsheet: {msg}"),
        )
        summary["specsheet_pdf"] = str(specsheet_pdf)
        summary["specsheet_generated"] = bool(generated)

        update_task_progress(task_id, message="Release specsheet ready", pct=100)
        update_task_result_summary(task_id, summary)
        _mark_run_status(
            task_id,
            parameters,
            task_type="run_release_specsheet_workflow",
            status="completed",
            result_path=str(specsheet_pdf),
            extra={
                "release_specsheet": {
                    "root": str(release_root),
                    "specsheet_pdf": str(specsheet_pdf),
                    "evaluator_jobs": summary["evaluator_jobs"],
                    "analysis_artifacts": summary["analysis_artifacts"],
                    "metadata": metadata,
                }
            },
        )
        append_task_log(task_id, f"Release specsheet PDF ready: {specsheet_pdf}")
        update_task_status(task_id, "completed", result_path=str(specsheet_pdf))
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        _mark_run_status(
            task_id,
            parameters,
            task_type="run_release_specsheet_workflow",
            status="failed",
            error_message=str(e),
            create_missing=True,
        )
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_run_evaluator_and_process(task_id: str, parameters: Dict[str, Any]) -> None:
    """
    Full combined workflow: Run Evaluator + Download + Eval + Parquet.
    
    Steps:
    1. Schedule evaluator job (get job_id)
    2. Poll until evaluator completes
    3. Download results
    4. Run eval
    5. Generate parquet
    """
    update_task_status(task_id, "running")
    append_task_log(task_id, "Starting run_evaluator_and_process workflow")
    _mark_run_status(
        task_id,
        parameters,
        task_type="run_evaluator_and_process",
        status="running",
        create_missing=True,
    )
    
    try:
        from lib import evaluator_api
        from lib import download_core
        
        # Import eval_summary
        eval_summary = _import_eval_summary()
        pkl_archive_to_parquet = _import_catalog_io()
        
        # Extract parameters
        project_id = parameters.get("project_id")
        catalog_id = parameters.get("catalog_id")
        integration_id = parameters.get("integration_id")
        source_job_id = parameters.get("source_job_id")
        suite_ids = parameters.get("suite_ids")
        target_name = parameters.get("target_name")  # branch name or tag
        description = str(parameters.get("description") or "").strip()
        if not description:
            description = _make_default_evaluator_description(parameters)
            parameters["description"] = description
            append_task_log(task_id, f"Using automatic evaluator description: {description}")
        output_path = parameters.get("output_path")
        trend_metadata = parameters.get("trend_metadata") if isinstance(parameters.get("trend_metadata"), dict) else None
        trend_role = str(parameters.get("trend_role") or "").strip()

        def _write_devops_trend_summary_from_suites(rows: list[dict[str, Any]]) -> None:
            if not output_path:
                return
            summary_payload: Dict[str, Any] = {"DevOps": {"Suite pass rate": {}}}
            for row in rows or []:
                suite_name = str(row.get("name") or row.get("suite_name") or row.get("simulation") or "suite").strip()
                total = int(row.get("all", 0) or row.get("total", 0) or 0)
                passed = int(row.get("success", 0) or row.get("passed", 0) or 0)
                if total <= 0:
                    failed = int(row.get("fail", 0) or row.get("failed", 0) or 0)
                    canceled = int(row.get("cancel", 0) or row.get("canceled", 0) or 0)
                    total = passed + failed + canceled
                if total <= 0:
                    continue
                summary_payload["DevOps"]["Suite pass rate"][suite_name] = {
                    "passed": passed,
                    "total": total,
                }
            if not summary_payload["DevOps"]["Suite pass rate"]:
                return
            resource_dir = Path(output_path) / "resources"
            resource_dir.mkdir(parents=True, exist_ok=True)
            with (resource_dir / "summary.json").open("w", encoding="utf-8") as fh:
                json.dump(summary_payload, fh, ensure_ascii=False, indent=2)
        
        # Eval options
        run_eval = parameters.get("run_eval", True)
        generate_parquet = parameters.get("generate_parquet", True)
        eval_recursive = parameters.get("eval_recursive", True)
        eval_overwrite = parameters.get("eval_overwrite", False)
        
        # Download options
        download_type = parameters.get("download_type", "archives")
        phase = parameters.get("phase", "perception.object_recognition.tracking.objects")
        skip_large_file = parameters.get("skip_large_file", False)
        large_file_mb = float(parameters.get("large_file_mb", 50.0))
        keep_zip_files = parameters.get("keep_zip_files", False)
        
        # Evaluator polling options
        poll_interval = float(parameters.get("poll_interval", 60.0))
        max_wait_seconds = _normalize_max_wait_seconds(parameters.get("max_wait_seconds", 0.0))
        download_ready_timeout = float(parameters.get("download_ready_timeout", 1800.0))
        download_ready_poll_interval = float(
            parameters.get("download_ready_poll_interval", min(max(poll_interval, 10.0), 60.0))
        )
        
        # Scheduling options
        max_retries = parameters.get("max_retries", 1)
        clean_build = parameters.get("clean_build", False)
        debug = parameters.get("debug", False)
        is_tag = parameters.get("is_tag", False)
        release = bool(parameters.get("release", False))
        record_caret = bool(parameters.get("record_caret", False))
        log_expiration_time_in_days = float(parameters.get("log_expiration_time_in_days", 14.0))

        has_source_job = bool(source_job_id)
        has_fresh_source = bool(integration_id and target_name)
        if not project_id or not catalog_id or not output_path or (not has_source_job and not has_fresh_source):
            _mark_run_status(
                task_id,
                parameters,
                task_type="run_evaluator_and_process",
                status="failed",
                error_message="Missing required parameters",
                create_missing=True,
            )
            update_task_status(task_id, "failed", error_message="Missing required parameters")
            return
        
        environment = parameters.get("environment", "default")
        os.environ["AUTH_PROFILE"] = environment
        os.environ["EVALUATOR_ENVIRONMENT"] = environment
        
        def on_progress(msg: str) -> None:
            append_task_log(task_id, msg)
            _append_run_event(task_id, parameters, task_type="run_evaluator_and_process", message=msg)
            update_task_progress(task_id, message=msg)
        
        def on_warning(msg: str) -> None:
            append_task_log(task_id, f"WARNING: {msg}")
            _append_run_event(task_id, parameters, task_type="run_evaluator_and_process", message=f"WARNING: {msg}")
        
        # Step 1: Schedule evaluator job
        on_progress("Step 1/5: Scheduling evaluator job...")
        if source_job_id:
            append_task_log(
                task_id,
                f"Project: {project_id}, Catalog: {catalog_id}, Reuse build from job: {source_job_id}",
            )
        else:
            append_task_log(task_id, f"Project: {project_id}, Catalog: {catalog_id}, Target: {target_name}")
        
        try:
            api = evaluator_api.EvaluationRunAPI()
            
            result = api.schedule_job(
                project_id=project_id,
                catalog_id=catalog_id,
                integration_id=integration_id,
                target_name=target_name,
                source_job_id=source_job_id,
                suite_ids=suite_ids,
                max_retries=max_retries,
                description=description,
                clean_build=clean_build,
                debug=debug,
                release=release,
                record_caret=record_caret,
                log_expiration_time_in_days=log_expiration_time_in_days,
                is_tag=is_tag,
            )
        except Exception as e:
            _mark_run_status(
                task_id,
                parameters,
                task_type="run_evaluator_and_process",
                status="failed",
                error_message=f"Failed to schedule evaluator job: {e}",
                create_missing=True,
            )
            update_task_status(task_id, "failed", error_message=f"Failed to schedule evaluator job: {e}")
            return
        
        job_id = result.get("job_id")
        if not job_id:
            _mark_run_status(
                task_id,
                parameters,
                task_type="run_evaluator_and_process",
                status="failed",
                error_message="No job_id returned from evaluator API",
                create_missing=True,
            )
            update_task_status(task_id, "failed", error_message="No job_id returned from evaluator API")
            return
        
        report_url = evaluator_api.get_job_report_url(project_id, job_id)
        append_task_log(task_id, f"Scheduled evaluator job: {job_id}")
        append_task_log(task_id, f"Report URL: {report_url}")
        update_task_progress(task_id, message=f"Evaluator job scheduled: {job_id}", pct=5)
        summary = {
            "job": "run_evaluator_and_process",
            "evaluator_job_id": job_id,
            "evaluator_report_url": report_url,
            "evaluator_status": "scheduled",
            "source_job_id": source_job_id or "",
            "download_summary": {"total": 0, "success": 0, "failed": 0},
            "eval_summary": {},
            "parquet_path": "",
        }
        update_task_result_summary(task_id, summary)
        _update_run_metadata(
            task_id,
            parameters,
            task_type="run_evaluator_and_process",
            create_missing=True,
            extra={
                "evaluator": {
                    "job_id": job_id,
                    "report_url": report_url,
                    "status": "scheduled",
                    "catalog_id": catalog_id,
                    "integration_id": integration_id or "",
                    "source_job_id": source_job_id or "",
                    "target_name": target_name or "",
                    "description": description or "",
                    "is_tag": bool(is_tag),
                    "title": description or "",
                }
            },
        )

        if trend_metadata:
            try:
                write_trend_metadata(output_path, trend_metadata)
                append_task_log(task_id, "Saved release trend metadata.")
                _update_run_metadata(
                    task_id,
                    parameters,
                    task_type="run_evaluator_and_process",
                    extra={
                        "trend": {
                            "enabled": True,
                            "role": trend_role,
                            "metadata": trend_metadata,
                        }
                    },
                )
            except Exception as e:
                append_task_log(task_id, f"WARNING: Could not save release trend metadata: {e}")
        
        # Step 2: Poll for evaluator completion
        on_progress("Step 2/5: Waiting for evaluator to complete...")
        append_task_log(task_id, "This may take a while depending on evaluator queue and run time...")
        last_suite_snapshot = {"key": None, "time": 0.0}
        
        def on_eval_progress(status: str, elapsed: float) -> None:
            hours = elapsed / 3600
            msg = f"Evaluator status: {status} (elapsed: {hours:.1f}h)"
            append_task_log(task_id, msg)
            # Progress: 5% to 40% during evaluation wait
            pct = _wait_progress_pct(
                elapsed=elapsed,
                timeout_seconds=max_wait_seconds,
                pct_start=5.0,
                pct_end=40.0,
            )
            update_task_progress(task_id, message=f"Evaluator: {status} ({hours:.1f}h elapsed)", pct=pct)
            summary["evaluator_status"] = status

            should_snapshot = elapsed < 60 or (elapsed - last_suite_snapshot["time"]) >= 600
            if not should_snapshot:
                update_task_result_summary(task_id, summary)
                return

            try:
                suite_rows = api.get_suite_summary(project_id, job_id, use_available_case_results=True)
            except Exception:
                update_task_result_summary(task_id, summary)
                return

            suite_summary = _summarize_suite_reports(suite_rows)
            totals = _suite_case_totals(suite_rows)
            snapshot_key = (
                totals["total"],
                totals["success"],
                totals["failed"],
                totals["canceled"],
                tuple((row["suite_name"], row["failed"]) for row in suite_summary if row["failed"] > 0),
            )
            last_suite_snapshot["time"] = elapsed
            if snapshot_key == last_suite_snapshot["key"]:
                summary["evaluator_case_totals"] = totals
                summary["evaluator_suites"] = suite_summary
                _update_run_metadata(
                    task_id,
                    parameters,
                    task_type="run_evaluator_and_process",
                    extra={
                        "evaluator": {
                            "status": status,
                            "case_totals": totals,
                            "suites": suite_summary,
                        }
                    },
                )
                update_task_result_summary(task_id, summary)
                return

            last_suite_snapshot["key"] = snapshot_key
            summary["evaluator_case_totals"] = totals
            summary["evaluator_suites"] = suite_summary
            if totals["total"] > 0:
                failing = [row for row in suite_summary if row["failed"] > 0]
                if failing:
                    top = ", ".join(f"{row['suite_name']}={row['failed']}" for row in failing[:3])
                    append_task_log(
                        task_id,
                        (
                            "Evaluator progress snapshot: "
                            f"{totals['success']}/{totals['total']} success, "
                            f"{totals['failed']} failed, {totals['canceled']} canceled. "
                            f"Failing suites: {top}"
                        ),
                    )
                else:
                    append_task_log(
                        task_id,
                        (
                            "Evaluator progress snapshot: "
                            f"{totals['success']}/{totals['total']} success, "
                            f"{totals['failed']} failed, {totals['canceled']} canceled."
                        ),
                    )
            _update_run_metadata(
                task_id,
                parameters,
                task_type="run_evaluator_and_process",
                extra={
                    "evaluator": {
                        "status": status,
                        "case_totals": totals,
                        "suites": suite_summary,
                    }
                },
            )
            update_task_result_summary(task_id, summary)
        
        try:
            final_report = api.wait_for_job_completion(
                project_id=project_id,
                job_id=job_id,
                poll_interval=poll_interval,
                max_wait_seconds=max_wait_seconds,
                on_check=on_eval_progress,
            )
        except evaluator_api.EvaluationAPIError as e:
            append_task_log(task_id, f"Evaluator wait error: {e}")
            _mark_run_status(
                task_id,
                parameters,
                task_type="run_evaluator_and_process",
                status="failed",
                error_message=f"Evaluator failed or timed out: {e}",
            )
            update_task_status(task_id, "failed", error_message=f"Evaluator failed or timed out: {e}")
            return
        
        test_status = evaluator_api.extract_job_status(final_report)
        try:
            suite_rows = api.get_suite_summary(project_id, job_id, use_available_case_results=True)
        except Exception as e:
            append_task_log(task_id, f"Could not fetch suite summary: {e}")
            suite_rows = []
        try:
            case_reports = api.get_case_reports(project_id, job_id)
        except Exception as e:
            append_task_log(task_id, f"Could not fetch case reports: {e}")
            case_reports = []

        if trend_metadata and trend_role == "devops":
            try:
                _write_devops_trend_summary_from_suites(suite_rows)
                append_task_log(task_id, "Saved DevOps trend summary.")
            except Exception as e:
                append_task_log(task_id, f"WARNING: Could not save DevOps trend summary: {e}")

        evaluator_summary = _build_evaluator_result_summary(
            job_id=job_id,
            report_url=report_url,
            evaluator_status=test_status,
            final_report=final_report,
            suite_rows=suite_rows,
            failed_cases=case_reports,
        )
        summary.update(evaluator_summary)
        update_task_result_summary(task_id, summary)
        _update_run_metadata(
            task_id,
            parameters,
            task_type="run_evaluator_and_process",
            extra={
                "evaluator": {
                    "job_id": job_id,
                    "report_url": report_url,
                    "status": test_status,
                    "title": summary.get("evaluator_title", ""),
                    "scheduled_by": summary.get("evaluator_scheduled_by", ""),
                    "catalog_id": summary.get("evaluator_catalog_id", ""),
                    "catalog_name": summary.get("evaluator_catalog_name", ""),
                    "catalog_version_id": summary.get("evaluator_catalog_version_id", ""),
                    "catalog_url": summary.get("evaluator_catalog_url", ""),
                    "target": summary.get("evaluator_target", ""),
                    "git_sha": summary.get("evaluator_git_sha", ""),
                    "git_ref_url": summary.get("evaluator_git_ref_url", ""),
                    "git_commit_url": summary.get("evaluator_git_commit_url", ""),
                    "source_url": summary.get("evaluator_source_url", ""),
                    "source_repo_label": summary.get("evaluator_source_repo_label", ""),
                    "build_status": summary.get("evaluator_build_status", ""),
                    "test_status": summary.get("evaluator_test_status", ""),
                    "fail_message": summary.get("evaluator_fail_message", ""),
                    "case_totals": summary.get("evaluator_case_totals", {}),
                    "suites": summary.get("evaluator_suites", []),
                    "failed_cases": summary.get("evaluator_failed_cases", []),
                }
            },
        )

        fail_message = summary.get("evaluator_fail_message", "")
        if evaluator_api.is_success_job_status(test_status):
            update_task_progress(task_id, message="Evaluator completed successfully", pct=40)
            append_task_log(task_id, f"Evaluator completed with status: {test_status}")
        else:
            append_task_log(task_id, f"Evaluator completed with non-success status: {test_status}")
            if fail_message:
                append_task_log(task_id, f"Evaluator fail message: {fail_message}")
            case_totals = summary.get("evaluator_case_totals", {})
            append_task_log(
                task_id,
                (
                    "Evaluator result summary: "
                    f"{case_totals.get('success', 0)}/{case_totals.get('total', 0)} success, "
                    f"{case_totals.get('failed', 0)} failed, {case_totals.get('canceled', 0)} canceled"
                ),
            )
            failed_cases = summary.get("evaluator_failed_cases", [])
            for case in failed_cases[:5]:
                detail = case.get("fail_message", "") or case.get("status", "")
                append_task_log(
                    task_id,
                    f"Failed case: {case.get('suite_name', '')} / {case.get('scenario_name', '')} - {detail}",
                )
            update_task_progress(task_id, message=f"Evaluator finished with status {test_status}; trying download", pct=40)
        
        # Step 3: Download results
        on_progress("Step 3/5: Downloading results...")
        update_task_progress(task_id, message="Downloading results...", pct=45)
        
        download_deadline = time.time() + download_ready_timeout
        while True:
            try:
                dl_result = download_core.run_download_results(
                    project_id=project_id,
                    job_id=job_id,
                    suite_id=None,
                    output_path=output_path,
                    download_type=download_type,
                    phase=phase,
                    skip_large_file=skip_large_file,
                    large_file_mb=large_file_mb,
                    keep_zip_files=keep_zip_files,
                    suite_ids=suite_ids,
                    on_progress=on_progress,
                    on_warning=on_warning,
                )
                failure_count, total_attempted, rows = dl_result
                success_count = total_attempted - failure_count
                download_success = success_count > 0
                
                if not download_success:
                    evaluator_msg = ""
                    if not evaluator_api.is_success_job_status(test_status):
                        evaluator_msg = f" Evaluator status was {test_status}."
                    _mark_run_status(
                        task_id,
                        parameters,
                        task_type="run_evaluator_and_process",
                        status="failed",
                        error_message=f"Download failed: {failure_count} of {total_attempted} scenarios failed.{evaluator_msg}",
                        result_path=output_path,
                    )
                    update_task_status(task_id, "failed", 
                        error_message=f"Download failed: {failure_count} of {total_attempted} scenarios failed.{evaluator_msg}")
                    return
                break

            except RuntimeError as e:
                if "No case reports found" not in str(e) or time.time() >= download_deadline:
                    evaluator_msg = ""
                    if not evaluator_api.is_success_job_status(test_status):
                        evaluator_msg = (
                            f" Evaluator status was {test_status}. "
                            "This usually means the job failed before producing downloadable case logs."
                        )
                    _mark_run_status(
                        task_id,
                        parameters,
                        task_type="run_evaluator_and_process",
                        status="failed",
                        error_message=f"Download failed: {e}{evaluator_msg}",
                        result_path=output_path,
                    )
                    update_task_status(task_id, "failed", error_message=f"Download failed: {e}{evaluator_msg}")
                    return

                wait_seconds = min(
                    download_ready_poll_interval,
                    max(1.0, download_deadline - time.time()),
                )
                msg = f"Case reports are not ready yet; retrying download in {wait_seconds:.0f}s"
                append_task_log(task_id, f"{msg}. Detail: {e}")
                update_task_progress(task_id, message=msg, pct=45)
                time.sleep(wait_seconds)
                
            except Exception as e:
                _mark_run_status(
                    task_id,
                    parameters,
                    task_type="run_evaluator_and_process",
                    status="failed",
                    error_message=f"Download failed: {e}",
                    result_path=output_path,
                )
                update_task_status(task_id, "failed", error_message=f"Download failed: {e}")
                return
        
        update_task_progress(task_id, message=f"Download complete: {success_count}/{total_attempted} succeeded", pct=60)
        summary["download_summary"] = {
            "total": total_attempted,
            "success": success_count,
            "failed": failure_count,
        }
        summary["download_rows"] = rows[:500]
        update_task_result_summary(task_id, summary)
        _update_run_metadata(
            task_id,
            parameters,
            task_type="run_evaluator_and_process",
            extra={
                "download": {
                    "mode": "run_evaluator_and_process",
                    "total": total_attempted,
                    "success": success_count,
                    "failed": failure_count,
                    "download_type": download_type,
                    "phase": phase,
                    "skip_large_file": bool(skip_large_file),
                    "large_file_mb": large_file_mb,
                    "keep_zip_files": bool(keep_zip_files),
                    "rows": rows[:100],
                }
            },
        )
        
        # Step 4: Run eval
        if run_eval:
            on_progress("Step 4/5: Running evaluation...")
            update_task_progress(task_id, message="Running evaluation...", pct=65)
            
            target_dirs = eval_summary.find_eval_result_dirs(output_path, recursive=eval_recursive)
            if target_dirs:
                total = len(target_dirs)
                eval_statuses = _run_eval_result_dirs(
                    task_id=task_id,
                    eval_summary=eval_summary,
                    target_dirs=target_dirs,
                    overwrite=eval_overwrite,
                    eval_workers=_eval_worker_count(parameters, total),
                    pct_start=65.0,
                    pct_end=85.0,
                    label="Eval",
                )
                
                # Generate summary CSVs
                update_task_progress(task_id, message="Generating Summary.csv / Score.csv", pct=85)
                csv_info = eval_summary.generate_summary_and_score_csv(output_path)
                failed = [s for s in eval_statuses if s.get("status") == "failed"]
                skipped = [s for s in eval_statuses if s.get("status") == "skipped"]
                succeeded = [s for s in eval_statuses if s.get("status") == "success"]
                
                eval_result_summary = {
                    "directories_processed": total,
                    "success": len(succeeded),
                    "failed": len(failed),
                    "skipped": len(skipped),
                    "summary_path": csv_info.get("summary_path", output_path),
                    "summary_rows": csv_info.get("summary_rows", 0),
                    "score_rows": csv_info.get("score_rows", 0),
                }
                append_task_log(task_id, f"Eval complete: {len(succeeded)}/{total} succeeded")
            else:
                eval_result_summary = {"directories_processed": 0, "success": 0, "failed": 0, "skipped": 0}
                append_task_log(task_id, "No eval result directories found")
        else:
            eval_result_summary = {}
        
        update_task_progress(task_id, message="Evaluation complete", pct=85)
        summary["eval_summary"] = eval_result_summary
        update_task_result_summary(task_id, summary)
        _update_run_metadata(
            task_id,
            parameters,
            task_type="run_evaluator_and_process",
            extra={
                "evaluation": {
                    **eval_result_summary,
                    "enabled": bool(run_eval),
                    "recursive": bool(eval_recursive),
                    "overwrite": bool(eval_overwrite),
                }
            },
        )
        
        # Step 5: Generate parquet
        parquet_path = ""
        if generate_parquet and pkl_archive_to_parquet:
            on_progress("Step 5/5: Generating parquet...")
            update_task_progress(task_id, message="Generating parquet...", pct=90)
            
            try:
                parquet_path = pkl_archive_to_parquet(
                    output_path,
                    on_progress=_parquet_progress_callback(
                        task_id,
                        prefix="Parquet",
                        pct_start=90,
                        pct_end=99,
                    ),
                    on_skip=lambda path, reason: append_task_log(
                        task_id,
                        f"Parquet skipped {path}: {reason}",
                    ),
                    project_id=project_id,
                    job_id=job_id,
                )
                update_task_progress(task_id, message="Parquet generated", pct=99)
                append_task_log(task_id, f"Parquet generated: {parquet_path}")
            except Exception as e:
                append_task_log(task_id, f"Parquet generation failed: {e}")
                parquet_path = ""
        
        update_task_progress(task_id, message="All steps complete", pct=100)
        summary["parquet_path"] = parquet_path
        
        # Build final summary
        update_task_result_summary(task_id, summary)
        _update_run_metadata(
            task_id,
            parameters,
            task_type="run_evaluator_and_process",
            extra={
                "parquet": {
                    "enabled": bool(generate_parquet),
                    "path": parquet_path,
                }
            },
        )
        if evaluator_api.is_success_job_status(test_status):
            append_task_log(task_id, "Workflow complete!")
        else:
            append_task_log(task_id, "Workflow complete. Evaluator job had failed test cases, but downloadable results were processed.")
        _mark_run_status(
            task_id,
            parameters,
            task_type="run_evaluator_and_process",
            status="completed",
            result_path=output_path,
        )
        update_task_status(task_id, "completed", result_path=output_path)
        
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        _mark_run_status(task_id, parameters, task_type="run_evaluator_and_process", status="failed", error_message=str(e))
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_prepare_pr_test_branch(task_id: str, parameters: Dict[str, Any]) -> None:
    append_task_log(task_id, "Starting PR test branch preparation")
    update_task_progress(task_id, message="Starting PR test branch preparation", pct=0)
    try:
        from lib.pr_test_branch_workflow import run_prepare_pr_test_branch

        summary = run_prepare_pr_test_branch(
            task_id=task_id,
            parameters=parameters,
            append_log=append_task_log,
            update_progress=update_task_progress,
            update_summary=update_task_result_summary,
        )
        result_path = str(summary.get("pilot_checkout") or summary.get("work_dir") or "")
        update_task_result_summary(task_id, summary)
        update_task_status(task_id, "completed", result_path=result_path)
    except Exception as e:
        append_task_log(task_id, f"Failed: {e}")
        update_task_status(task_id, "failed", error_message=str(e))
        raise


def job_local_evaluator_debug(task_id: str, parameters: Dict[str, Any]) -> None:
    append_task_log(task_id, "Starting local evaluator debug workflow")
    update_task_progress(task_id, message="Starting local evaluator debug workflow", pct=0)
    try:
        from lib.local_evaluator_debug import run_local_evaluator_debug

        summary = run_local_evaluator_debug(
            task_id=task_id,
            parameters=parameters,
            append_log=append_task_log,
            update_progress=update_task_progress,
            update_summary=update_task_result_summary,
        )
        update_task_result_summary(task_id, summary)
        update_task_status(
            task_id,
            "completed",
            result_path=str(summary.get("run_dir") or summary.get("log_path") or ""),
        )
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
    "run_release_specsheet_workflow": job_run_release_specsheet_workflow,
    "run_evaluator_and_process": job_run_evaluator_and_process,
    "prepare_pr_test_branch": job_prepare_pr_test_branch,
    "local_evaluator_debug": job_local_evaluator_debug,
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
