"""Helpers for durable per-run metadata stored alongside local run folders."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Optional

from lib.path_utils import get_data_root, path_display, to_data_relative

RUN_METADATA_FILENAME = ".run_metadata.json"
RUN_METADATA_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    return value


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_run_path(path_like: str | Path, *, allow_missing: bool = True) -> Optional[Path]:
    raw = str(path_like or "").strip()
    if not raw:
        return None
    try:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = get_data_root() / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(get_data_root())
        except ValueError:
            return None
        if not allow_missing and not resolved.exists():
            return None
        return resolved
    except Exception:
        return None


def find_run_directory(path_like: str | Path, *, create_missing: bool = False) -> Optional[Path]:
    resolved = normalize_run_path(path_like, allow_missing=True)
    if resolved is None:
        return None
    try:
        rel = resolved.relative_to(get_data_root())
    except ValueError:
        return None
    if not rel.parts:
        return None
    run_dir = get_data_root() / rel.parts[0]
    if create_missing:
        run_dir.mkdir(parents=True, exist_ok=True)
    elif not run_dir.exists():
        return None
    return run_dir


def resolve_run_directory_from_task_parameters(
    parameters: Dict[str, Any],
    *,
    create_missing: bool = False,
) -> Optional[Path]:
    for key in ("output_path", "output_dir", "eval_root", "pkl_dir", "result_path"):
        path_value = parameters.get(key)
        if not path_value:
            continue
        run_dir = find_run_directory(path_value, create_missing=create_missing)
        if run_dir is not None:
            return run_dir
    return None


def metadata_path_for_run(run_path: Path) -> Path:
    return run_path / RUN_METADATA_FILENAME


def read_run_metadata(run_path: Path) -> Dict[str, Any]:
    meta_path = metadata_path_for_run(run_path)
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_run_metadata(run_path: Path, metadata: Dict[str, Any], *, create_missing: bool = False) -> Dict[str, Any]:
    if create_missing:
        run_path.mkdir(parents=True, exist_ok=True)
    elif not run_path.exists():
        raise FileNotFoundError(str(run_path))

    payload = dict(metadata)
    payload["schema_version"] = RUN_METADATA_SCHEMA_VERSION
    payload["run_name"] = run_path.name
    payload["run_path"] = to_data_relative(run_path)
    payload["run_path_display"] = path_display(run_path)
    payload["updated_at"] = _utc_now_iso()
    payload.setdefault("created_at", payload["updated_at"])

    meta_path = metadata_path_for_run(run_path)
    with NamedTemporaryFile("w", encoding="utf-8", dir=str(run_path), delete=False) as tmp:
        json.dump(_json_safe(payload), tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    try:
        os.chmod(tmp_path, 0o644)
    except Exception:
        pass
    tmp_path.replace(meta_path)
    try:
        os.chmod(meta_path, 0o644)
    except Exception:
        pass
    return payload


def upsert_run_metadata(run_path: Path, patch: Dict[str, Any], *, create_missing: bool = False) -> Dict[str, Any]:
    existing = read_run_metadata(run_path)
    merged = _deep_merge(existing, _json_safe(patch))
    if "created_at" not in merged:
        merged["created_at"] = _utc_now_iso()
    return write_run_metadata(run_path, merged, create_missing=create_missing)


def flatten_metadata_text(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(str(key))
            parts.extend(flatten_metadata_text(item))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            parts.extend(flatten_metadata_text(item))
        return parts
    text = str(value).strip()
    return [text] if text else []


def build_run_search_blob(run_path: Path, metadata: Dict[str, Any], extra_values: Optional[Iterable[Any]] = None) -> str:
    parts = [run_path.name, to_data_relative(run_path), path_display(run_path)]
    parts.extend(flatten_metadata_text(metadata))
    if extra_values:
        for value in extra_values:
            parts.extend(flatten_metadata_text(value))
    return " ".join(part for part in parts if part).lower()


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def resolve_run_directory_from_task_row(task_row: Dict[str, Any]) -> Optional[Path]:
    params = _as_dict(task_row.get("parameters"))
    run_dir = resolve_run_directory_from_task_parameters(params, create_missing=False)
    if run_dir is not None:
        return run_dir
    result_path = task_row.get("result_path")
    if result_path:
        return find_run_directory(result_path, create_missing=False)
    summary = _as_dict(task_row.get("result_summary"))
    for key in ("output_path", "summary_path", "parquet_path"):
        path_value = summary.get(key)
        if path_value:
            run_dir = find_run_directory(path_value, create_missing=False)
            if run_dir is not None:
                return run_dir
    return None


def build_metadata_patch_from_task_row(task_row: Dict[str, Any]) -> Dict[str, Any]:
    params = _as_dict(task_row.get("parameters"))
    summary = _as_dict(task_row.get("result_summary"))
    task_type = str(task_row.get("type") or "").strip()
    requester = _as_dict(params.get("_requester"))
    requested_by = str(task_row.get("session_id") or requester.get("id") or "").strip()
    request_output = str(
        params.get("output_path")
        or params.get("output_dir")
        or params.get("eval_root")
        or params.get("pkl_dir")
        or task_row.get("result_path")
        or ""
    ).strip()

    patch: Dict[str, Any] = {
        "source_mode": task_type,
        "task": {
            "id": str(task_row.get("id") or "").strip(),
            "type": task_type,
            "status": str(task_row.get("status") or "").strip(),
            "requested_by": requested_by,
            "requester": requester,
            "created_at": task_row.get("created_at"),
            "updated_at": task_row.get("updated_at"),
            "result_path": str(task_row.get("result_path") or "").strip(),
            "error_message": str(task_row.get("error_message") or "").strip(),
            "progress_message": str(task_row.get("progress_message") or "").strip(),
            "progress_pct": task_row.get("progress_pct"),
        },
        "request": {
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
            "output_path": request_output,
            "parameters": params,
        },
        "owner": requester or ({"id": requested_by} if requested_by else {}),
        "backfilled_from_task_history": True,
    }

    if task_type == "download_results":
        patch["download"] = {
            "mode": "download_results",
            "total": summary.get("total", 0),
            "success": summary.get("success", 0),
            "failed": summary.get("failed", 0),
            "rows": list(summary.get("rows") or [])[:100],
            "download_type": str(params.get("download_type") or "").strip(),
            "phase": str(params.get("phase") or "").strip(),
            "skip_large_file": bool(params.get("skip_large_file", False)),
            "large_file_mb": params.get("large_file_mb"),
            "keep_zip_files": bool(params.get("keep_zip_files", False)),
        }
    elif task_type == "download_scenarios":
        patch["scenario_download"] = {
            "total": summary.get("total", 0),
            "success": summary.get("success", 0),
            "failed": summary.get("failed", 0),
            "rows": list(summary.get("rows") or [])[:100],
            "overwrite": bool(params.get("overwrite", False)),
            "scenario_name_filter": str(params.get("scenario_name_filter") or "").strip(),
            "selected_ids": list(params.get("selected_ids") or []),
        }
    elif task_type == "run_eval_dirs":
        patch["evaluation"] = {
            "directories_processed": summary.get("directories_processed", 0),
            "success": summary.get("success", 0),
            "failed": summary.get("failed", 0),
            "skipped": summary.get("skipped", 0),
            "summary_path": str(summary.get("summary_path") or "").strip(),
            "summary_rows": summary.get("summary_rows", 0),
            "score_rows": summary.get("score_rows", 0),
            "enabled": True,
            "recursive": bool(params.get("recursive", True)),
            "overwrite": bool(params.get("overwrite", False)),
        }
    elif task_type == "generate_summary_csv":
        patch["evaluation"] = {
            "summary_path": str(summary.get("summary_path") or "").strip(),
            "summary_rows": summary.get("summary_rows", 0),
            "score_rows": summary.get("score_rows", 0),
            "enabled": True,
        }
    elif task_type == "build_parquet":
        patch["parquet"] = {
            "enabled": True,
            "path": str(summary.get("output_path") or "").strip(),
        }
    elif task_type == "download_and_eval":
        patch["download"] = {
            "mode": "download_and_eval",
            **_as_dict(summary.get("download_summary")),
            "download_type": str(params.get("download_type") or "").strip(),
            "phase": str(params.get("phase") or "").strip(),
            "skip_large_file": bool(params.get("skip_large_file", False)),
            "large_file_mb": params.get("large_file_mb"),
            "keep_zip_files": bool(params.get("keep_zip_files", False)),
        }
        patch["evaluation"] = {
            **_as_dict(summary.get("eval_summary")),
            "enabled": bool(params.get("run_eval", False)),
            "recursive": bool(params.get("eval_recursive", False)),
            "overwrite": bool(params.get("eval_overwrite", False)),
        }
        patch["parquet"] = {
            "enabled": bool(params.get("generate_parquet", False)),
            "path": str(summary.get("parquet_path") or "").strip(),
        }
        errors = list(summary.get("errors") or [])
        if errors:
            patch["errors"] = errors
    elif task_type == "run_evaluator_and_process":
        patch["evaluator"] = {
            "job_id": str(summary.get("evaluator_job_id") or params.get("job_id") or "").strip(),
            "report_url": str(summary.get("evaluator_report_url") or "").strip(),
            "status": str(summary.get("evaluator_status") or "").strip(),
            "title": str(summary.get("evaluator_title") or params.get("description") or "").strip(),
            "scheduled_by": str(summary.get("evaluator_scheduled_by") or "").strip(),
            "build_status": str(summary.get("evaluator_build_status") or "").strip(),
            "test_status": str(summary.get("evaluator_test_status") or "").strip(),
            "fail_message": str(summary.get("evaluator_fail_message") or "").strip(),
            "case_totals": _as_dict(summary.get("evaluator_case_totals")),
            "suites": list(summary.get("evaluator_suites") or []),
            "failed_cases": list(summary.get("evaluator_failed_cases") or []),
            "catalog_id": str(params.get("catalog_id") or "").strip(),
            "catalog_name": str(summary.get("evaluator_catalog_name") or "").strip(),
            "catalog_version_id": str(summary.get("evaluator_catalog_version_id") or "").strip(),
            "catalog_url": str(summary.get("evaluator_catalog_url") or "").strip(),
            "integration_id": str(params.get("integration_id") or "").strip(),
            "source_job_id": str(params.get("source_job_id") or "").strip(),
            "target_name": str(params.get("target_name") or "").strip(),
            "target": str(summary.get("evaluator_target") or params.get("target_name") or "").strip(),
            "git_sha": str(summary.get("evaluator_git_sha") or "").strip(),
            "git_ref_url": str(summary.get("evaluator_git_ref_url") or "").strip(),
            "git_commit_url": str(summary.get("evaluator_git_commit_url") or "").strip(),
            "source_url": str(summary.get("evaluator_source_url") or "").strip(),
            "source_repo_label": str(summary.get("evaluator_source_repo_label") or "").strip(),
            "description": str(params.get("description") or "").strip(),
            "is_tag": bool(params.get("is_tag", False)),
        }
        patch["download"] = {
            "mode": "run_evaluator_and_process",
            **_as_dict(summary.get("download_summary")),
            "rows": list(summary.get("download_rows") or [])[:100],
            "download_type": str(params.get("download_type") or "").strip(),
            "phase": str(params.get("phase") or "").strip(),
            "skip_large_file": bool(params.get("skip_large_file", False)),
            "large_file_mb": params.get("large_file_mb"),
            "keep_zip_files": bool(params.get("keep_zip_files", False)),
        }
        patch["evaluation"] = {
            **_as_dict(summary.get("eval_summary")),
            "enabled": bool(params.get("run_eval", False)),
            "recursive": bool(params.get("eval_recursive", False)),
            "overwrite": bool(params.get("eval_overwrite", False)),
        }
        patch["parquet"] = {
            "enabled": bool(params.get("generate_parquet", False)),
            "path": str(summary.get("parquet_path") or "").strip(),
        }

    return patch
