"""Shared Recent Evaluator Jobs UI."""

from __future__ import annotations

import html
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from lib import evaluator_api
from lib.path_utils import resolve_under_data_root, to_data_relative

_JST = timezone(timedelta(hours=9))
_CONFIG_GETTER: Callable[[str, Any], Any] = lambda key, default=None: default
_CONFIG_SETTER: Callable[[str, Any], None] = lambda key, value: None
_ENQUEUE_TASK: Callable[[str, Dict[str, Any]], Optional[str]] = lambda task_type, params: None
CATALOG_IO_AVAILABLE = False
ENVIRONMENT = "default"


def configure_recent_evaluator_jobs_ui(*, get_config_value: Callable[[str, Any], Any], set_config_value: Callable[[str, Any], None], enqueue_task: Callable[[str, Dict[str, Any]], Optional[str]], catalog_io_available: bool, environment: str = "default") -> None:
    global _CONFIG_GETTER, _CONFIG_SETTER, _ENQUEUE_TASK, CATALOG_IO_AVAILABLE, ENVIRONMENT
    _CONFIG_GETTER = get_config_value
    _CONFIG_SETTER = set_config_value
    _ENQUEUE_TASK = enqueue_task
    CATALOG_IO_AVAILABLE = bool(catalog_io_available)
    ENVIRONMENT = environment or "default"


def get_config_value(key: str, default: Any = None) -> Any:
    return _CONFIG_GETTER(key, default)


def set_config_value(key: str, value: Any) -> None:
    _CONFIG_SETTER(key, value)


def _enqueue_task(task_type: str, params: Dict[str, Any]) -> Optional[str]:
    return _ENQUEUE_TASK(task_type, params)


def _friendly_request_error_message(exc: Exception) -> str:
    text = str(exc or "").strip()
    lowered = text.lower()
    if "temporary failure in name resolution" in lowered or "failed to resolve" in lowered or "name resolution" in lowered:
        return "Could not load evaluator jobs because the network appears to be unavailable."
    if "auth.web.auto" in lowered or "/token" in lowered:
        return "Could not load evaluator jobs because the sign-in service is currently unavailable."
    if "connection refused" in lowered or "max retries exceeded" in lowered or "newconnectionerror" in lowered:
        return "Could not connect to the evaluator service right now. Please try again in a moment."
    if "timed out" in lowered or "timeout" in lowered:
        return "Loading evaluator jobs took too long. Please try again."
    return "Could not load evaluator jobs right now. Please check the network connection and try again."


def _load_catalog_presets() -> List[Dict[str, str]]:
    """Load catalog presets from the app-level catalogs.json file if available."""
    app_root = Path(__file__).resolve().parents[2]
    search_paths = [
        app_root / "catalogs.json",
        Path(os.environ.get("CATALOGS_PATH", "")),
        Path.cwd() / "catalogs.json",
    ]
    for path in search_paths:
        if not path or not str(path):
            continue
        try:
            if not path.exists() or not path.is_file():
                continue
            import json

            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            raw_catalogs = data.get("catalogs", []) if isinstance(data, dict) else data
            presets: List[Dict[str, str]] = []
            for item in raw_catalogs or []:
                if not isinstance(item, dict):
                    continue
                display_name = (
                    str(item.get("display_name") or item.get("name") or item.get("catalog_id") or "")
                    .strip()
                )
                if not display_name:
                    continue
                presets.append({**item, "display_name": display_name})
            return presets
        except Exception:
            continue
    return []


def _retest_catalog_emoji(preset_name: str, *, has_custom_catalog: bool = False) -> str:
    mapping = {
        "Build Test Catalog": "🛠️",
        "Performance Test": "📈",
        "Old performance test": "🕰️",
        "Devops Test": "⚙️",
        "Usecase Performance Catalog": "🧭",
        "L4 regression test": "⚠️",
    }
    normalized = str(preset_name or "").strip()
    if normalized in mapping:
        return mapping[normalized]
    if has_custom_catalog:
        return "🧩"
    return "📦"


def _make_retest_description(target_name: str, preset_name: str = "", *, has_custom_catalog: bool = False) -> str:
    clean_target = " ".join(str(target_name or "").strip().split()) or "artifact"
    stamp = datetime.now().strftime("%m-%d %H:%M")
    return (
        f"♻️ evaluator artifact retest [{clean_target}] [{stamp}] "
        f"{_retest_catalog_emoji(preset_name, has_custom_catalog=has_custom_catalog)}"
    )


def _to_jst(dt: Any) -> Optional[datetime]:
    if dt is None:
        return None
    if not hasattr(dt, "astimezone"):
        return None
    try:
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_JST)
    except Exception:
        return None

def _parse_api_dt(value: Any) -> Optional[datetime]:
    """Parse evaluator API timestamps into timezone-aware datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if getattr(value, "tzinfo", None) is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _format_jst_time(value: Any, *, include_seconds: bool = False) -> str:
    """Format timestamps for display in JST."""
    dt = _to_jst(_parse_api_dt(value))
    if not dt:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S JST" if include_seconds else "%Y-%m-%d %H:%M JST")


def _format_jst_time_compact(value: Any) -> str:
    """Compact timestamp for dense recent-job rows."""
    dt = _to_jst(_parse_api_dt(value))
    if not dt:
        return "—"
    return dt.strftime("%m-%d %H:%M")


def _format_jst_time_title(value: Any) -> str:
    """Readable timestamp for fallback job titles."""
    dt = _to_jst(_parse_api_dt(value))
    if not dt:
        return "unknown time"
    return f"{dt.year}/{dt.month}/{dt.day} {dt.hour}:{dt.minute:02d}:{dt.second:02d}"


def _format_relative_time(value: Any) -> str:
    """Human-friendly age/duration from a timestamp until now."""
    dt = _parse_api_dt(value)
    if not dt:
        return "—"
    now = datetime.now(timezone.utc)
    secs = max(0, int((now - dt.astimezone(timezone.utc)).total_seconds()))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _format_duration(start_value: Any, end_value: Any) -> str:
    """Format elapsed duration between two evaluator timestamps."""
    start = _parse_api_dt(start_value)
    end = _parse_api_dt(end_value)
    if not start or not end:
        return "—"
    secs = max(0, int((end - start).total_seconds()))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    return f"{secs // 3600}h {(secs % 3600) // 60}m"


def _extract_git_target(report: Dict[str, Any]) -> str:
    """Return a compact branch/tag label from evaluator job report metadata."""
    source = ((report.get("event") or {}).get("source") or {})
    git_ref = str(source.get("git_ref") or "").strip()
    if git_ref.startswith("refs/heads/"):
        return git_ref[len("refs/heads/"):]
    if git_ref.startswith("refs/tags/"):
        return git_ref[len("refs/tags/"):]
    return git_ref or str(source.get("git_sha") or "").strip()[:12] or "—"


def _extract_catalog_url(report: Dict[str, Any]) -> str:
    """Return a best-effort catalog URL for linking from recent evaluator jobs."""
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
    catalog_id = str(
        catalog.get("catalog_id")
        or catalog.get("id")
        or ""
    ).strip()
    if project_id and catalog_id:
        return f"https://evaluation.tier4.jp/evaluation/vehicle_catalogs/{catalog_id}?project_id={project_id}"
    return ""


def _extract_job_title(report: Dict[str, Any]) -> str:
    """Prefer evaluator description for display title, with a readable fallback."""
    description = str(report.get("description") or "").strip()
    if description:
        return description
    started_like = report.get("started_at") or report.get("scheduled_at") or report.get("finished_at")
    return f"no description (Started at {_format_jst_time_title(started_like)})"


def _extract_case_totals(report: Dict[str, Any]) -> Dict[str, int]:
    """Return total/success/failed/canceled counts from job report."""
    test = report.get("test") or {}
    result = test.get("available_case_results") or test.get("case_results") or {}
    return {
        "total": int(result.get("total_count", 0) or 0),
        "success": int(result.get("success_count", 0) or 0),
        "failed": int(result.get("failure_count", 0) or 0),
        "canceled": int(result.get("cancellation_count", 0) or 0),
    }


def _extract_failed_case_rows(case_reports: List[Dict[str, Any]], *, limit: int = 50) -> List[Dict[str, Any]]:
    """Normalize failed case rows for display tables."""
    rows: List[Dict[str, Any]] = []
    for report in case_reports:
        status = str(report.get("status") or "").strip().lower()
        result_status = str(((report.get("result") or {}).get("status") or "")).strip().lower()
        if status not in evaluator_api.FAILED_JOB_STATUSES and result_status not in evaluator_api.FAILED_JOB_STATUSES:
            continue
        logs = report.get("logs") or {}
        rows.append(
            {
                "Suite": ((report.get("suite") or {}).get("display_name") or ""),
                "Scenario": ((report.get("scenario") or {}).get("display_name") or ""),
                "Status": report.get("status", ""),
                "Fail message": report.get("fail_message", ""),
                "Cause": ", ".join(report.get("failure_cause_labels", []) or []),
                "Archive log": "yes" if ((logs.get("simulation_archive") or {}).get("id")) else "no",
                "Result JSON": "yes" if ((logs.get("simulation_result_json") or {}).get("id")) else "no",
            }
        )
    rows.sort(key=lambda row: (row["Suite"], row["Scenario"], row["Fail message"]))
    return rows[:limit]


def _extract_suite_rows(suite_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize suite summary rows for display tables."""
    rows = [
        {
            "Suite": row.get("name", ""),
            "Total": int(row.get("all", 0) or 0),
            "Success": int(row.get("success", 0) or 0),
            "Failed": int(row.get("fail", 0) or 0),
            "Canceled": int(row.get("cancel", 0) or 0),
            "Simulation": row.get("simulation", ""),
            "Report": row.get("url", ""),
        }
        for row in suite_rows or []
    ]
    rows.sort(key=lambda row: (-row["Failed"], row["Suite"]))
    return rows


def _extract_suite_selection_options(suite_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build suite picker options from evaluator suite summary rows."""
    options: List[Dict[str, str]] = []
    seen_ids = set()
    for row in suite_rows or []:
        report_url = str(row.get("url") or row.get("Report") or "").strip()
        suite_id = ""
        if "/tests/" in report_url:
            tail = report_url.split("/tests/", 1)[1]
            suite_id = tail.split("?", 1)[0].split("/", 1)[0].strip()
        if not suite_id or suite_id in seen_ids:
            continue
        seen_ids.add(suite_id)
        suite_name = str(row.get("name") or row.get("Suite") or suite_id).strip()
        options.append({"id": suite_id, "label": f"{suite_name} ({suite_id})"})
    return options


def _status_color_variant(status: str) -> str:
    """Map evaluator status to a style token used by the recent-job cards."""
    normalized = evaluator_api.normalize_job_status(status)
    if normalized in evaluator_api.SUCCESS_JOB_STATUSES:
        return "success"
    if normalized in ("canceled", "cancelled", "aborted"):
        return "canceled"
    if normalized in evaluator_api.FAILED_JOB_STATUSES:
        return "failed"
    if normalized in ("started", "running", "pending", "queued", "created"):
        return "running"
    return "unknown"


def _status_display_label(status: str) -> str:
    """Short status label for compact list rows."""
    normalized = evaluator_api.normalize_job_status(status)
    if normalized in ("succeeded", "success"):
        return "success"
    if normalized in ("failed", "failure", "error"):
        return "failed"
    if normalized in ("canceled", "cancelled", "aborted"):
        return "canceled"
    if normalized in ("started", "running"):
        return "running"
    if normalized in ("pending", "queued", "created"):
        return "queued"
    return normalized or "unknown"


def _status_filter_values(selected_statuses: List[str]) -> List[str]:
    """Normalize UI status filters into API status values."""
    values: List[str] = []
    for raw in selected_statuses:
        normalized = evaluator_api.normalize_job_status(raw)
        if normalized == "unknown" or not normalized:
            continue
        if normalized == "running":
            values.extend(["running", "started"])
        elif normalized == "success":
            values.extend(["success", "succeeded"])
        elif normalized == "failed":
            values.extend(["failed", "failure", "error"])
        elif normalized == "canceled":
            values.extend(["canceled", "cancelled", "aborted"])
        else:
            values.append(normalized)
    return sorted(set(values))


def _escape_search_match_value(value: str) -> str:
    """Escape wildcard characters for API Match filters."""
    return (
        value.replace("\\", "\\\\")
        .replace("*", "\\*")
        .replace("?", "\\?")
    )


def _build_recent_job_search_filter(
    search_text: str,
    search_scope: str,
    user_directory: Optional[Dict[str, Dict[str, str]]] = None,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Map quick-search UI to one server-side filter and a client-side needle."""
    needle = search_text.strip()
    if not needle:
        return None, ""

    if search_scope == "Branch/tag":
        return (
            {
                "field": "event.source.git_ref",
                "operator": "Match",
                "values": [f"*{_escape_search_match_value(needle)}*"],
            },
            needle.lower(),
        )
    if search_scope == "Description":
        return (
            {
                "field": "description",
                "operator": "Match",
                "values": [f"*{_escape_search_match_value(needle)}*"],
            },
            needle.lower(),
        )
    if search_scope == "Job ID":
        return (
            {
                "field": "job_id",
                "operator": "In",
                "values": [needle],
            },
            needle.lower(),
        )
    if search_scope == "Git SHA":
        return (
            {
                "field": "event.source.git_sha",
                "operator": "Match",
                "values": [f"*{_escape_search_match_value(needle)}*"],
            },
            needle.lower(),
        )
    if search_scope == "Fail message":
        return (
            {
                "field": "fail_message",
                "operator": "Match",
                "values": [f"*{_escape_search_match_value(needle)}*"],
            },
            needle.lower(),
        )
    return None, needle.lower()


def _recent_job_search_history_key(scope: str) -> str:
    return f"recent_eval_jobs_search_history::{scope}"


def _get_recent_job_search_history(scope: str) -> List[str]:
    stored = get_config_value(_recent_job_search_history_key(scope), []) or []
    if not isinstance(stored, list):
        return []
    return [str(v).strip() for v in stored if str(v).strip()]


def _save_recent_job_search_history(scope: str, value: str, *, max_items: int = 8) -> None:
    text = str(value).strip()
    if not text:
        return
    history = _get_recent_job_search_history(scope)
    updated = [text] + [item for item in history if item != text]
    set_config_value(_recent_job_search_history_key(scope), updated[:max_items])


def _get_recent_eval_user_directory() -> Dict[str, Dict[str, str]]:
    stored = get_config_value("recent_eval_jobs_user_directory", {}) or {}
    if not isinstance(stored, dict):
        return {}
    normalized: Dict[str, Dict[str, str]] = {}
    for subject_id, info in stored.items():
        if not isinstance(info, dict):
            continue
        normalized[str(subject_id)] = {
            "name": str(info.get("name") or "").strip(),
            "email": str(info.get("email") or "").strip(),
            "subject_id": str(info.get("subject_id") or subject_id).strip(),
        }
    return normalized


def _save_recent_eval_user_directory(directory: Dict[str, Dict[str, str]]) -> None:
    set_config_value("recent_eval_jobs_user_directory", directory)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _fetch_auth_member_profile(subject_id: str, environment: str) -> Dict[str, str]:
    subject = str(subject_id or "").strip()
    if not subject:
        return {}
    org_id = os.environ.get(
        "WEBAUTO_ORGANIZATION_ID",
        "5a21621d-6968-4f7d-94f8-99cfb77b6e71",
    ).strip()
    if not org_id:
        return {"subject_id": subject, "name": subject, "email": ""}
    os.environ["AUTH_PROFILE"] = environment or ENVIRONMENT
    from webautoauth.token import HttpService, TokenSource, load_config

    config = load_config()
    token_source = TokenSource(HttpService(config))
    access_token = token_source.get_token().access_token
    quoted_subject = urllib.parse.quote(subject, safe="")
    url = f"https://auth.web.auto/v2/organizations/{org_id}/members/{quoted_subject}"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}", "accept": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "subject_id": str(data.get("subject_id") or subject),
        "name": str(data.get("name") or subject).strip(),
        "email": str(data.get("email") or "").strip(),
    }


def _hydrate_recent_eval_user_directory(
    jobs: List[Dict[str, Any]],
    environment: str,
) -> Dict[str, Dict[str, str]]:
    directory = _get_recent_eval_user_directory()
    unresolved = sorted(
        {
            str(job.get("scheduled_by") or "").strip()
            for job in jobs
            if str(job.get("scheduled_by") or "").strip()
            and str(job.get("scheduled_by") or "").strip() not in directory
        }
    )
    if not unresolved:
        return directory

    updates: Dict[str, Dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=min(6, len(unresolved))) as executor:
        future_map = {
            executor.submit(_fetch_auth_member_profile, subject_id, environment): subject_id
            for subject_id in unresolved
        }
        for future in as_completed(future_map):
            subject_id = future_map[future]
            try:
                profile = future.result()
            except Exception:
                profile = {
                    "subject_id": subject_id,
                    "name": subject_id,
                    "email": "",
                }
            updates[subject_id] = {
                "subject_id": str(profile.get("subject_id") or subject_id).strip(),
                "name": str(profile.get("name") or subject_id).strip(),
                "email": str(profile.get("email") or "").strip(),
            }

    if updates:
        directory = {**directory, **updates}
        _save_recent_eval_user_directory(directory)
    return directory


def _build_recent_job_date_filters(
    date_from: Optional[datetime.date],
    date_to: Optional[datetime.date],
) -> List[Dict[str, Any]]:
    """Build scheduled_at date-range filters for the search API."""
    filters: List[Dict[str, Any]] = []
    if date_from:
        start_dt = datetime(date_from.year, date_from.month, date_from.day, 0, 0, 0, tzinfo=_JST)
        filters.append(
            {
                "field": "scheduled_at",
                "operator": "Gte",
                "values": [start_dt.astimezone(timezone.utc).isoformat()],
            }
        )
    if date_to:
        end_dt = datetime(date_to.year, date_to.month, date_to.day, 23, 59, 59, tzinfo=_JST)
        filters.append(
            {
                "field": "scheduled_at",
                "operator": "Lte",
                "values": [end_dt.astimezone(timezone.utc).isoformat()],
            }
        )
    return filters


def _summarize_recent_job(report: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary for one evaluator job card."""
    status = evaluator_api.extract_job_status(report)
    totals = _extract_case_totals(report)
    source = ((report.get("event") or {}).get("source") or {})
    git_url = str(source.get("git_web_url") or source.get("git_url") or "").strip()
    source_repo_label = git_url.rstrip("/").split("/")[-1] if git_url else "—"
    git_ref_label = _extract_git_target(report)
    return {
        "job_id": report.get("job_id") or report.get("id") or "",
        "title": _extract_job_title(report),
        "status": status,
        "status_variant": _status_color_variant(status),
        "build_status": ((report.get("build") or {}).get("status") or ""),
        "test_status": ((report.get("test") or {}).get("status") or ""),
        "target": git_ref_label,
        "catalog": ((report.get("catalog") or {}).get("display_name") or ""),
        "catalog_url": _extract_catalog_url(report),
        "description": report.get("description", ""),
        "source_label": git_ref_label,
        "source_repo_label": source_repo_label,
        "scheduled_at": report.get("scheduled_at"),
        "started_at": report.get("started_at"),
        "finished_at": report.get("finished_at"),
        "duration": _format_duration(report.get("started_at"), report.get("finished_at")),
        "created_label": _format_relative_time(report.get("scheduled_at") or report.get("started_at")),
        "scheduled_by": str(report.get("scheduled_by") or ""),
        "report_url": evaluator_api.get_job_report_url(report.get("project_id", ""), report.get("job_id") or report.get("id") or ""),
        "fail_message": report.get("fail_message", ""),
        "total": totals["total"],
        "success": totals["success"],
        "failed": totals["failed"],
        "canceled": totals["canceled"],
        "git_sha": str(source.get("git_sha") or "")[:12],
        "git_ref_url": source.get("git_ref_url", ""),
        "git_commit_url": source.get("git_commit_url", ""),
        "source_url": git_url,
    }


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_recent_evaluator_job_pages(
    project_id: str,
    environment: str,
    page_size: int,
    pages_to_fetch: int,
    status_values: tuple[str, ...] = (),
    extra_filters: tuple[tuple[str, str, tuple[Any, ...]], ...] = (),
) -> List[Dict[str, Any]]:
    """Fetch recent evaluator jobs from the search endpoint page-by-page."""
    if not project_id:
        return []
    os.environ["AUTH_PROFILE"] = environment or ENVIRONMENT
    api = evaluator_api.EvaluationRunAPI()
    filters: List[Dict[str, Any]] = []
    if status_values:
        filters.append(
            {
                "field": "status",
                "operator": "In",
                "values": list(status_values),
            }
        )
    for field, operator, values in extra_filters:
        filters.append(
            {
                "field": field,
                "operator": operator,
                "values": list(values),
            }
        )
    next_token = ""
    pages: List[Dict[str, Any]] = []
    for _ in range(max(1, int(pages_to_fetch))):
        data = api.search_report_list(
            project_id,
            filters=filters or None,
            next_token=next_token,
            size=max(1, min(int(page_size), 100)),
        )
        reports = data.get("reports", []) or []
        pages.append(
            {
                "jobs": [_summarize_recent_job(report) for report in reports],
                "next_token": data.get("next_token", "") or "",
            }
        )
        next_token = data.get("next_token", "") or ""
        if not next_token:
            break
    return pages


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_evaluator_job_detail(project_id: str, environment: str, job_id: str) -> Dict[str, Any]:
    """Fetch deep evaluator detail for one job on demand."""
    if not project_id or not job_id:
        return {}
    os.environ["AUTH_PROFILE"] = environment or ENVIRONMENT
    api = evaluator_api.EvaluationRunAPI()
    report = api.get_job_report(project_id, job_id)
    suite_rows = api.get_suite_summary(project_id, job_id, use_available_case_results=True)
    case_reports = api.get_case_reports(project_id, job_id)
    summary = _summarize_recent_job(report)
    return {
        **summary,
        "suite_rows": _extract_suite_rows(suite_rows),
        "failed_case_rows": _extract_failed_case_rows(case_reports),
        "raw_report": report,
    }


def _inject_recent_evaluator_jobs_styles() -> None:
    """Task-adjacent styles for the recent evaluator jobs section."""
    st.markdown(
        """
        <style>
        .evj-card {
            border-radius: 16px;
            padding: 0.7rem 0.85rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }
        .evj-card--running {
            border-color: rgba(245, 158, 11, 0.28);
            background: linear-gradient(180deg, rgba(255, 251, 235, 0.98), rgba(255,255,255,0.98));
        }
        .evj-card--success {
            border-color: rgba(16, 185, 129, 0.24);
            background: linear-gradient(180deg, rgba(236, 253, 245, 0.98), rgba(255,255,255,0.98));
        }
        .evj-card--failed {
            border-color: rgba(239, 68, 68, 0.24);
            background: linear-gradient(180deg, rgba(254, 242, 242, 0.98), rgba(255,255,255,0.98));
        }
        .evj-top, .evj-meta, .evj-stats {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .evj-top { justify-content: space-between; }
        .evj-row {
            display: grid;
            grid-template-columns: minmax(180px, 1.25fr) minmax(86px, 0.48fr) minmax(172px, 0.95fr) minmax(170px, 1.05fr) minmax(120px, 0.8fr) minmax(160px, 1fr);
            gap: 8px;
            align-items: center;
        }
        .evj-title {
            font-size: 0.9rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
            word-break: break-word;
        }
        .evj-title a {
            color: inherit;
            text-decoration: none;
        }
        .evj-title a:hover {
            text-decoration: underline;
        }
        .evj-name {
            min-width: 0;
        }
        .evj-name .evj-title,
        .evj-name .evj-name-sub,
        .evj-ref-cell,
        .evj-ref-cell .evj-name-sub {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .evj-name-sub {
            margin-top: 0.15rem;
            font-size: 0.74rem;
            color: #64748b;
        }
        .evj-status {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 0.24rem 0.5rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: lowercase;
            letter-spacing: 0.01em;
            border: 1px solid transparent;
        }
        .evj-status--running { color: #9a6700; background: #fff7db; border-color: rgba(245, 158, 11, 0.28); }
        .evj-status--success { color: #047857; background: #dcfce7; border-color: rgba(16, 185, 129, 0.28); }
        .evj-status--failed { color: #b91c1c; background: #fee2e2; border-color: rgba(239, 68, 68, 0.28); }
        .evj-status--canceled { color: #7c3aed; background: #f3e8ff; border-color: rgba(124, 58, 237, 0.24); }
        .evj-status--unknown { color: #475569; background: #f1f5f9; border-color: rgba(148, 163, 184, 0.28); }
        .evj-status-mark {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 14px;
            height: 14px;
            border-radius: 999px;
            font-size: 0.62rem;
            font-weight: 900;
            line-height: 1;
            border: 1px solid currentColor;
            flex: 0 0 auto;
        }
        .evj-status-mark--success {
            background: rgba(4, 120, 87, 0.08);
        }
        .evj-status-mark--failed {
            background: rgba(185, 28, 28, 0.08);
        }
        .evj-status-mark--canceled {
            background: rgba(124, 58, 237, 0.08);
        }
        .evj-status-mark--unknown {
            background: rgba(71, 85, 105, 0.08);
        }
        .evj-status-mark--running {
            position: relative;
            border-radius: 999px;
            border: 1.5px solid rgba(154, 103, 0, 0.18);
            border-top-color: currentColor;
            border-right-color: currentColor;
            background: transparent;
            animation: evj-spin 0.9s linear infinite;
        }
        .evj-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            display: inline-block;
            background: currentColor;
            opacity: 0.88;
        }
        .evj-dot--pulse {
            animation: evj-pulse 1.4s ease-in-out infinite;
        }
        @keyframes evj-pulse {
            0% { transform: scale(0.9); opacity: 0.55; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.55; }
        }
        @keyframes evj-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .evj-meta {
            color: #475569;
            font-size: 0.82rem;
        }
        .evj-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 0.7rem;
        }
        .evj-toolbar-note {
            margin: 0.15rem 0 0.35rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: #64748b;
            text-transform: uppercase;
        }
        .evj-pager-note {
            margin-top: 0.28rem;
            font-size: 0.76rem;
            color: #475569;
            white-space: nowrap;
        }
        .evj-cell {
            min-width: 0;
            font-size: 0.78rem;
            color: #334155;
        }
        .evj-cell a {
            color: #0f766e;
            text-decoration: none;
            font-weight: 700;
        }
        .evj-cell a:hover {
            text-decoration: underline;
        }
        .evj-cell strong {
            color: #0f172a;
        }
        .evj-cell--nowrap {
            white-space: nowrap;
        }
        .evj-detail {
            margin-top: 1rem;
            padding: 1rem 1rem 0.8rem;
            border-radius: 18px;
            border: 1px solid rgba(15, 118, 110, 0.14);
            background:
                radial-gradient(circle at top right, rgba(45, 212, 191, 0.10), transparent 24%),
                linear-gradient(180deg, rgba(255,255,255,0.99), rgba(247,250,252,0.99));
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        }
        .evj-stat {
            flex: 1 1 80px;
            min-width: 72px;
            padding: 0.55rem 0.7rem;
            border-radius: 14px;
            background: rgba(248, 250, 252, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.16);
        }
        .evj-inline-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            font-size: 0.76rem;
            color: #334155;
        }
        [class*="st-key-recent_eval_view_"] button,
        [class*="st-key-recent_eval_run_"] button,
        [class*="st-key-recent_eval_retest_"] button,
        [class*="st-key-recent_eval_jobs_prev"] button,
        [class*="st-key-recent_eval_jobs_next"] button,
        [class*="st-key-recent_eval_jobs_pagebtn_"] button,
        [class*="st-key-refresh_recent_eval_jobs"] button {
            min-height: 2rem;
            padding: 0.18rem 0.58rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            box-shadow: none;
        }
        [class*="st-key-recent_eval_view_"] button,
        [class*="st-key-recent_eval_retest_"] button,
        [class*="st-key-recent_eval_jobs_prev"] button,
        [class*="st-key-recent_eval_jobs_next"] button,
        [class*="st-key-recent_eval_jobs_pagebtn_"] button,
        [class*="st-key-refresh_recent_eval_jobs"] button {
            border-color: rgba(148, 163, 184, 0.34);
            color: #334155;
            background: #ffffff;
        }
        [class*="st-key-recent_eval_view_"] button:hover,
        [class*="st-key-recent_eval_retest_"] button:hover,
        [class*="st-key-recent_eval_jobs_prev"] button:hover,
        [class*="st-key-recent_eval_jobs_next"] button:hover,
        [class*="st-key-recent_eval_jobs_pagebtn_"] button:hover,
        [class*="st-key-refresh_recent_eval_jobs"] button:hover {
            border-color: rgba(15, 118, 110, 0.28);
            color: #0f766e;
            background: #f8fffd;
        }
        [class*="st-key-recent_eval_jobs_pagebtn_active_"] button {
            border-color: rgba(13, 148, 136, 0.26);
            background: linear-gradient(180deg, #f0fdfa, #ecfeff);
            color: #0f766e;
        }
        [class*="st-key-recent_eval_run_"] button {
            border-color: rgba(13, 148, 136, 0.22);
            background: linear-gradient(180deg, #f0fdfa, #ecfeff);
            color: #0f766e;
        }
        [class*="st-key-recent_eval_run_"] button:hover {
            border-color: rgba(13, 148, 136, 0.34);
            background: linear-gradient(180deg, #ccfbf1, #ecfeff);
            color: #115e59;
        }
        [class*="st-key-recent_eval_retest_"] button {
            border-color: rgba(251, 191, 36, 0.22);
            background: linear-gradient(180deg, #fffbeb, #fff7ed);
            color: #b45309;
        }
        [class*="st-key-recent_eval_retest_"] button:hover {
            border-color: rgba(245, 158, 11, 0.34);
            background: linear-gradient(180deg, #fef3c7, #fff7ed);
            color: #92400e;
        }
        .evj-stat-label {
            display: block;
            font-size: 0.68rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 800;
            margin-bottom: 0.14rem;
        }
        .evj-stat-value {
            display: block;
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
        }
        .evj-desc {
            margin-top: 0.55rem;
            font-size: 0.86rem;
            color: #334155;
        }
        .evj-empty {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: #f8fafc;
            border: 1px dashed rgba(148, 163, 184, 0.4);
            color: #475569;
        }
        @media (max-width: 1080px) {
            .evj-row {
                grid-template-columns: 1fr;
                gap: 8px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_recent_evaluator_job_card(job: Dict[str, Any], *, user_label: str = "Unknown") -> None:
    """Render one recent evaluator job as a single-row list item."""
    variant = html.escape(job.get("status_variant", "unknown"))
    status = html.escape(_status_display_label(job.get("status", "unknown") or "unknown"))
    title_text = html.escape(job.get("title", "—"))
    description = html.escape(job.get("description", "") or "")
    catalog = html.escape(job.get("catalog", "") or "—")
    catalog_url = html.escape(job.get("catalog_url", "") or "")
    scheduled = html.escape(_format_jst_time_compact(job.get("scheduled_at")))
    duration = html.escape(job.get("duration", "—"))
    job_id = html.escape(str(job.get("job_id", "")))
    build_status = html.escape(job.get("build_status", "") or "—")
    test_status = html.escape(job.get("test_status", "") or "—")
    created_label = html.escape(job.get("created_label", "—"))
    git_sha = html.escape(job.get("git_sha", "") or "—")
    source_label = html.escape(job.get("source_label", "") or "—")
    user_text = html.escape(user_label or "Unknown")
    report_url = html.escape(job.get("report_url", "") or "")
    source_url = html.escape(job.get("git_ref_url", "") or job.get("source_url", "") or "")
    status_variant = job.get("status_variant", "unknown")
    status_mark = {
        "running": '<span class="evj-status-mark evj-status-mark--running" aria-hidden="true"></span>',
        "success": '<span class="evj-status-mark evj-status-mark--success" aria-hidden="true">✓</span>',
        "failed": '<span class="evj-status-mark evj-status-mark--failed" aria-hidden="true">!</span>',
        "canceled": '<span class="evj-status-mark evj-status-mark--canceled" aria-hidden="true">×</span>',
    }.get(status_variant, '<span class="evj-status-mark evj-status-mark--unknown" aria-hidden="true">?</span>')
    meta_line = job_id
    counts = (
        f'S <strong>{int(job.get("success", 0))}</strong> · '
        f'F <strong>{int(job.get("failed", 0))}</strong> · '
        f'C <strong>{int(job.get("canceled", 0))}</strong> / '
        f'<strong>{int(job.get("total", 0))}</strong>'
    )
    title_html = f'<a href="{report_url}" target="_blank" rel="noopener noreferrer">{title_text}</a>' if report_url else title_text
    source_html = (
        f'<a href="{source_url}" target="_blank" rel="noopener noreferrer">{source_label}</a>'
        if source_url else source_label
    )
    catalog_html = (
        f'<a href="{catalog_url}" target="_blank" rel="noopener noreferrer">{catalog}</a>'
        if catalog_url else catalog
    )
    st.markdown(
        f"""
        <div class="evj-card evj-card--{variant}">
          <div class="evj-row">
            <div class="evj-name">
              <div class="evj-title">{title_html}</div>
              <div class="evj-name-sub">{meta_line}</div>
            </div>
            <div class="evj-cell evj-cell--nowrap">
              <span class="evj-status evj-status--{variant}">{status_mark}{status}</span>
            </div>
            <div class="evj-cell">
              <strong>{scheduled} ({created_label})</strong><br><span class="evj-name-sub">{duration}</span>
            </div>
            <div class="evj-cell evj-ref-cell">
              <strong>{catalog_html}</strong><br><span class="evj-name-sub">{source_html}</span>
            </div>
            <div class="evj-cell evj-ref-cell">
              <strong>{user_text}</strong>
            </div>
            <div class="evj-cell">
              <span class="evj-name-sub">build {build_status} · test {test_status} · {git_sha}</span><br>
              <span class="evj-inline-stats">{counts}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_recent_evaluator_job_detail(project_id: str, environment: str, job: Dict[str, Any]) -> None:
    """Render detailed evaluator-job information inside an expander."""
    job_id = str(job.get("job_id", "") or "")
    if not job_id:
        st.warning("Missing job id.")
        return
    try:
        detail = _fetch_evaluator_job_detail(project_id, environment, job_id)
    except Exception as e:
        st.error(f"Could not fetch evaluator details: {e}")
        return

    st.markdown("**Overview**")
    top_cols = st.columns(4)
    top_cols[0].metric("Total", int(detail.get("total", 0)))
    top_cols[1].metric("Success", int(detail.get("success", 0)))
    top_cols[2].metric("Failed", int(detail.get("failed", 0)))
    top_cols[3].metric("Canceled", int(detail.get("canceled", 0)))

    overview_left, overview_right = st.columns([1.3, 1.1])
    with overview_left:
        st.write(f"Status: `{detail.get('status', 'unknown')}`")
        st.write(f"Title: `{detail.get('title', '—')}`")
        st.write(f"Build/Test: `{detail.get('build_status', '—')}` / `{detail.get('test_status', '—')}`")
        st.write(f"Ref: `{detail.get('target', '—')}`")
        st.write(f"Catalog: `{detail.get('catalog', '—')}`")
        st.write(f"Repo: `{detail.get('source_repo_label', '—')}`")
    with overview_right:
        st.write(f"Scheduled: `{_format_jst_time(detail.get('scheduled_at'), include_seconds=True)}`")
        st.write(f"Started: `{_format_jst_time(detail.get('started_at'), include_seconds=True)}`")
        st.write(f"Finished: `{_format_jst_time(detail.get('finished_at'), include_seconds=True)}`")
        st.write(f"Duration: `{detail.get('duration', '—')}`")
        st.write(f"SHA: `{detail.get('git_sha', '—')}`")

    action_cols = st.columns([1.2, 1.2, 4])
    report_url = detail.get("report_url", "")
    catalog_url = detail.get("catalog_url", "")
    source_url = detail.get("source_url", "") or detail.get("git_ref_url", "")
    with action_cols[0]:
        if report_url:
            st.link_button("Open report", report_url, use_container_width=True)
    with action_cols[1]:
        if catalog_url:
            st.link_button("Open catalog", catalog_url, use_container_width=True)
    with action_cols[2]:
        if source_url:
            st.link_button("Open source", source_url, use_container_width=True)

    if detail.get("fail_message"):
        st.warning(detail.get("fail_message"))

    suite_rows = detail.get("suite_rows") or []
    with st.expander(f"Suites ({len(suite_rows)})", expanded=bool(suite_rows)):
        if suite_rows:
            st.dataframe(pd.DataFrame(suite_rows), width="stretch", hide_index=True)
        else:
            st.caption("No suite summary available.")

    failed_case_rows = detail.get("failed_case_rows") or []
    with st.expander(f"Failed Cases ({len(failed_case_rows)})", expanded=bool(failed_case_rows)):
        if failed_case_rows:
            st.dataframe(pd.DataFrame(failed_case_rows), width="stretch", hide_index=True)
        else:
            st.caption("No failed cases in the current report.")

    with st.expander("Raw JSON", expanded=False):
        st.json(detail.get("raw_report", {}))


def _render_recent_evaluator_job_run_dialog(
    project_id: str,
    environment: str,
    job: Dict[str, Any],
    *,
    output_path_default: str,
    download_type_default: str,
    phase_default: str,
    skip_large_file_default: bool,
    large_file_mb_default: float,
    keep_zip_files_default: bool,
) -> None:
    """Render the dialog used to enqueue Download + Eval + Parquet from a recent job row."""
    job_id = str(job.get("job_id", "") or "")
    if not job_id:
        st.error("Missing evaluator job id.")
        return

    detail = _fetch_evaluator_job_detail(project_id, environment, job_id)
    suite_options = _extract_suite_selection_options(detail.get("suite_rows") or [])
    suite_label_to_id = {opt["label"]: opt["id"] for opt in suite_options}
    suite_labels = [opt["label"] for opt in suite_options]

    st.caption("Confirm the workflow options for this evaluator job, then start a background task.")
    summary_cols = st.columns([1.45, 1.15, 1.35, 1.05])
    summary_cols[0].markdown(f"**Title**  \n`{detail.get('title', '—')}`")
    summary_cols[1].markdown(f"**Status**  \n`{detail.get('status', 'unknown')}`")
    summary_cols[2].markdown(f"**Catalog**  \n`{detail.get('catalog', '—')}`")
    summary_cols[3].markdown(f"**Cases**  \n`{int(detail.get('total', 0))}`")

    with st.form(key=f"recent_eval_run_form_{job_id}", border=False):
        run_output_path = st.text_input(
            "Output path",
            value=output_path_default,
            help="Folder under the data directory. This uses the same safe path rules as the main download workflow.",
        )

        if not suite_labels:
            hint_cols = st.columns([1.2, 2.8])
            with hint_cols[0]:
                if st.form_submit_button("Refresh suites", use_container_width=True):
                    _fetch_evaluator_job_detail.clear()
                    st.rerun()
            with hint_cols[1]:
                st.caption("No suite candidates were available yet for this job. Refresh to re-read suite data from the evaluator API.")

        selected_suite_labels = st.multiselect(
            "Suites to download (optional)",
            options=suite_labels,
            default=[],
            help="Leave empty to download all suites from this evaluator job.",
            disabled=not suite_labels,
        )

        run_download_type = st.radio(
            "Download type",
            ["Archives (ZIP)", "Result JSON only"],
            index=0 if download_type_default == "Archives (ZIP)" else 1,
            horizontal=True,
        )

        run_phase = ""
        run_skip_large_file = False
        run_large_file_mb = 50.0
        run_keep_zip_files = False
        if run_download_type == "Archives (ZIP)":
            run_phase = st.text_input(
                "Phase to extract",
                value=phase_default,
                help="Enter the phase name to extract from archives.",
            )
            opt_cols = st.columns([1.2, 1.3, 1.2])
            with opt_cols[0]:
                run_skip_large_file = st.checkbox(
                    "Skip large files",
                    value=skip_large_file_default,
                    help="Skip unusually large archives during download.",
                )
            with opt_cols[1]:
                run_large_file_mb = st.number_input(
                    "Skip threshold (MB)",
                    min_value=1.0,
                    max_value=5000.0,
                    step=1.0,
                    value=float(large_file_mb_default),
                )
            with opt_cols[2]:
                run_keep_zip_files = st.checkbox(
                    "Keep ZIP files",
                    value=keep_zip_files_default,
                    help="Keep downloaded ZIPs after extraction.",
                )

        run_cols = st.columns([1.25, 1.25, 1.1])
        with run_cols[0]:
            run_eval = st.checkbox(
                "Run evaluation",
                value=True,
                help="Run eval_result and generate Summary.csv / Score.csv after download.",
            )
        with run_cols[1]:
            generate_parquet = st.checkbox(
                "Generate parquet",
                value=CATALOG_IO_AVAILABLE,
                disabled=not CATALOG_IO_AVAILABLE,
                help="Build scene_result.parquet from .pkl files." if CATALOG_IO_AVAILABLE else "Install perception_catalog_analyzer to enable this.",
            )
        with run_cols[2]:
            eval_recursive = st.checkbox(
                "Recursive eval",
                value=True,
                help="Search subdirectories for evaluation result folders.",
            )

        action_cols = st.columns([1.15, 1.15, 3.7])
        cancel_clicked = action_cols[0].form_submit_button("Cancel", use_container_width=True)
        start_clicked = action_cols[1].form_submit_button("Start", type="primary", use_container_width=True)

    if cancel_clicked:
        st.session_state.pop("recent_eval_jobs_run_selected", None)
        st.rerun()

    if not start_clicked:
        return

    resolved_output, path_err = resolve_under_data_root(run_output_path, allow_create=True)
    if path_err:
        st.error(f"Output path is invalid: {path_err}")
        return

    selected_suite_ids = [suite_label_to_id[label] for label in selected_suite_labels]
    resolved_path_str = str(resolved_output)
    set_config_value("output_path", to_data_relative(resolved_output))
    set_config_value("environment", environment)
    set_config_value("project_id", project_id)
    set_config_value("job_id", job_id)
    set_config_value("suite_id", "")
    set_config_value("suite_ids", selected_suite_ids)
    set_config_value("download_type", run_download_type)
    if run_download_type == "Archives (ZIP)":
        set_config_value("phase", run_phase)
        set_config_value("skip_large_file", run_skip_large_file)
        set_config_value("large_file_mb", run_large_file_mb)
        set_config_value("keep_zip_files", run_keep_zip_files)

    params = {
        "output_path": resolved_path_str,
        "project_id": project_id,
        "job_id": job_id,
        "suite_id": "",
        "suite_ids": selected_suite_ids or None,
        "download_type": "archives" if run_download_type == "Archives (ZIP)" else "result_json",
        "phase": run_phase if run_download_type == "Archives (ZIP)" else "",
        "skip_large_file": run_skip_large_file if run_download_type == "Archives (ZIP)" else False,
        "large_file_mb": run_large_file_mb if run_download_type == "Archives (ZIP)" else 50.0,
        "keep_zip_files": run_keep_zip_files if run_download_type == "Archives (ZIP)" else False,
        "run_eval": run_eval,
        "generate_parquet": generate_parquet,
        "eval_recursive": eval_recursive,
        "eval_overwrite": False,
    }
    task_id = _enqueue_task("download_and_eval", params)
    if not task_id:
        st.error("Failed to enqueue task. Check REDIS_URL and DATABASE_URL.")
        return

    st.session_state["recent_eval_jobs_flash"] = (
        f"Queued Download + Eval + Parquet for `{detail.get('title', job_id)}`. "
        f"Task id: `{task_id}`."
    )
    st.session_state.pop("recent_eval_jobs_run_selected", None)
    st.rerun()


def _render_recent_evaluator_job_retest_dialog(
    project_id: str,
    environment: str,
    job: Dict[str, Any],
    *,
    output_path_default: str,
    phase_default: str,
) -> None:
    """Render a compact workflow launcher that reuses build artifacts from a prior evaluator job."""
    job_id = str(job.get("job_id", "") or "")
    if not job_id:
        st.error("Missing evaluator job id.")
        return

    detail = _fetch_evaluator_job_detail(project_id, environment, job_id)
    raw_report = detail.get("raw_report") or {}
    raw_catalog = raw_report.get("catalog") or {}
    suite_options = _extract_suite_selection_options(detail.get("suite_rows") or [])
    suite_label_to_id = {opt["label"]: opt["id"] for opt in suite_options}
    suite_labels = [opt["label"] for opt in suite_options]
    preset_entries = _load_catalog_presets()
    preset_names = [str(entry.get("display_name") or "").strip() for entry in preset_entries if str(entry.get("display_name") or "").strip()]
    preset_by_name = {str(entry.get("display_name") or "").strip(): entry for entry in preset_entries}

    original_catalog_name = str(raw_catalog.get("display_name") or detail.get("catalog") or "").strip()
    original_catalog_id = str(raw_catalog.get("id") or "").strip()
    default_preset_name = original_catalog_name if original_catalog_name in preset_by_name else ""

    import re

    default_output_path = output_path_default
    if not default_output_path:
        clean_target = re.sub(r"[^\w]+", "_", str(detail.get("target") or job_id).strip()).strip("_") or "artifact"
        default_output_path = f"retest_{clean_target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    st.caption("Schedule a new evaluator workflow that reuses build artifacts from this job, then download and process the new results.")
    summary_cols = st.columns([1.35, 1.0, 1.25, 1.2])
    summary_cols[0].markdown(f"**Source job**  \n`{job_id}`")
    summary_cols[1].markdown(f"**Ref**  \n`{detail.get('target', '—')}`")
    summary_cols[2].markdown(f"**Original catalog**  \n`{original_catalog_name or '—'}`")
    summary_cols[3].markdown(f"**Suites found**  \n`{len(suite_labels)}`")

    preset_key = f"recent_eval_retest_catalog_preset_{job_id}"
    last_preset_key = f"recent_eval_retest_last_catalog_preset_{job_id}"
    catalog_id_key = f"recent_eval_retest_catalog_id_{job_id}"
    if preset_key not in st.session_state:
        st.session_state[preset_key] = default_preset_name
    if last_preset_key not in st.session_state:
        st.session_state[last_preset_key] = ""
    if catalog_id_key not in st.session_state:
        st.session_state[catalog_id_key] = original_catalog_id

    selected_preset_name = st.selectbox(
        "Catalog preset",
        options=[""] + preset_names,
        index=([""] + preset_names).index(default_preset_name) if default_preset_name in preset_names else 0,
        key=preset_key,
        help="Choose a preset catalog, or leave this empty and enter a catalog id manually.",
        format_func=lambda value: value or "Custom / manual",
    )
    selected_preset = preset_by_name.get(selected_preset_name or "", {})
    if st.session_state[last_preset_key] != selected_preset_name and selected_preset_name:
        st.session_state[catalog_id_key] = str(selected_preset.get("catalog_id") or "")
        st.session_state[last_preset_key] = selected_preset_name
    elif st.session_state[last_preset_key] != selected_preset_name and not selected_preset_name:
        st.session_state[catalog_id_key] = original_catalog_id
        st.session_state[last_preset_key] = selected_preset_name
    catalog_id = st.text_input(
        "Catalog ID",
        value="",
        key=catalog_id_key,
        help="You can switch to a different catalog while still reusing the build artifacts from the source job.",
    ).strip()

    selected_suite_labels = st.multiselect(
        "Suites to run",
        options=suite_labels,
        default=suite_labels,
        help="Defaults to the suite set found on the source job. Clear the list to let the evaluator use its default suite selection.",
        disabled=not suite_labels,
    )
    description = st.text_input(
        "Description",
        value="",
        help="Leave empty to use an automatic evaluator artifact-retest name.",
    ).strip()
    retest_output_path = st.text_input(
        "Output path",
        value=default_output_path,
        help="Folder under the data directory for the downloaded retest results.",
    )
    run_download_type = st.radio(
        "Download type",
        ["Archives (ZIP)", "Result JSON only"],
        index=0,
        horizontal=True,
    )
    run_phase = ""
    if run_download_type == "Archives (ZIP)":
        run_phase = st.text_input(
            "Phase to extract",
            value=phase_default,
            help="Enter the phase name to extract from archives.",
        )

    run_cols = st.columns([1.2, 1.2, 1.0])
    with run_cols[0]:
        run_eval = st.checkbox(
            "Run evaluation",
            value=True,
            help="Run eval_result and generate Summary.csv / Score.csv after download.",
        )
    with run_cols[1]:
        generate_parquet = st.checkbox(
            "Generate parquet",
            value=CATALOG_IO_AVAILABLE,
            disabled=not CATALOG_IO_AVAILABLE,
            help="Build scene_result.parquet from .pkl files." if CATALOG_IO_AVAILABLE else "Install perception_catalog_analyzer to enable this.",
        )
    with run_cols[2]:
        eval_recursive = st.checkbox(
            "Recursive eval",
            value=True,
            help="Search subdirectories for evaluation result folders.",
        )

    action_cols = st.columns([1.15, 1.15, 3.7])
    cancel_clicked = action_cols[0].button("Cancel", key=f"recent_eval_retest_cancel_{job_id}", use_container_width=True)
    start_clicked = action_cols[1].button("Retest", key=f"recent_eval_retest_start_{job_id}", type="primary", use_container_width=True)

    if cancel_clicked:
        st.session_state.pop("recent_eval_jobs_retest_selected", None)
        st.rerun()

    if not start_clicked:
        return

    final_catalog_id = str(selected_preset.get("catalog_id") or catalog_id or "").strip()
    if not final_catalog_id:
        st.error("Catalog ID is required.")
        return

    resolved_output, path_err = resolve_under_data_root(retest_output_path, allow_create=True)
    if path_err:
        st.error(f"Output path is invalid: {path_err}")
        return

    selected_suite_ids = [suite_label_to_id[label] for label in selected_suite_labels]
    resolved_path_str = str(resolved_output)
    has_custom_catalog = bool(final_catalog_id and not selected_preset_name)
    final_description = description or _make_retest_description(
        str(detail.get("target") or job_id),
        selected_preset_name,
        has_custom_catalog=has_custom_catalog,
    )

    task_id = _enqueue_task(
        "run_evaluator_and_process",
        {
            "project_id": project_id,
            "catalog_id": final_catalog_id,
            "integration_id": "",
            "source_job_id": job_id,
            "suite_ids": selected_suite_ids or None,
            "target_name": "",
            "description": final_description,
            "output_path": resolved_path_str,
            "environment": environment,
            "max_retries": 0,
            "clean_build": False,
            "debug": False,
            "is_tag": False,
            "download_type": "archives" if run_download_type == "Archives (ZIP)" else "result_json",
            "phase": run_phase,
            "skip_large_file": False,
            "large_file_mb": 50.0,
            "keep_zip_files": False,
            "poll_interval": 60,
            "max_wait_seconds": 6 * 3600,
            "run_eval": run_eval,
            "generate_parquet": generate_parquet,
            "eval_recursive": eval_recursive,
            "eval_overwrite": False,
        },
    )
    if not task_id:
        st.error("Failed to enqueue task. Check REDIS_URL and DATABASE_URL.")
        return

    set_config_value("output_path", to_data_relative(resolved_output))
    set_config_value("environment", environment)
    set_config_value("project_id", project_id)
    set_config_value("catalog_id", final_catalog_id)
    set_config_value("suite_ids", selected_suite_ids)

    st.session_state["recent_eval_jobs_flash"] = (
        f"Queued artifact retest for `{detail.get('title', job_id)}`. "
        f"Task id: `{task_id}`."
    )
    st.session_state.pop("recent_eval_jobs_retest_selected", None)
    st.rerun()


def _render_recent_evaluator_jobs_section(
    project_id: str,
    environment: str,
    *,
    output_path_default: str,
    download_type_default: str,
    phase_default: str,
    skip_large_file_default: bool,
    large_file_mb_default: float,
    keep_zip_files_default: bool,
    show_toggle: bool = True,
    default_visible: bool = False,
    show_title: bool = True,
) -> None:
    """Render a direct evaluator-jobs browser above the download tabs."""
    _inject_recent_evaluator_jobs_styles()
    if show_toggle:
        show_section = st.toggle(
            "Show recent evaluator jobs",
            value=st.session_state.get("recent_eval_jobs_show", default_visible),
            key="recent_eval_jobs_show",
            help="Load recent evaluator jobs only when you want to browse them.",
        )
    else:
        show_section = True
        st.session_state["recent_eval_jobs_show"] = True
    if not show_section:
        return

    if show_title:
        st.subheader("Recent evaluator jobs")
        st.caption("Compact browser for recent evaluator jobs. Select one job to inspect detailed suite and failed-case information.")
    flash_message = st.session_state.pop("recent_eval_jobs_flash", None)
    if flash_message:
        st.success(flash_message)
    user_directory = _get_recent_eval_user_directory()

    control_cols = st.columns([0.75, 1.0, 1.15, 1.45, 1.25, 1.0, 1.0, 0.75])
    with control_cols[0]:
        st.markdown('<div class="evj-toolbar-note">Rows</div>', unsafe_allow_html=True)
        limit = int(
            st.selectbox(
                "Rows",
                options=[10, 20, 50, 100],
                index=1,
                key="recent_eval_jobs_limit",
                help="How many recent evaluator jobs to fetch for this project.",
                label_visibility="collapsed",
            )
        )
    with control_cols[1]:
        st.markdown('<div class="evj-toolbar-note">Status</div>', unsafe_allow_html=True)
        status_filter = st.multiselect(
            "Status",
            options=["running", "success", "failed", "canceled", "unknown"],
            default=[],
            key="recent_eval_jobs_status_filter",
            help="Leave empty to show all recent jobs.",
            label_visibility="collapsed",
            placeholder="All statuses",
        )
    with control_cols[2]:
        st.markdown('<div class="evj-toolbar-note">Search In</div>', unsafe_allow_html=True)
        search_scope = st.selectbox(
            "Search in",
            options=["Branch/tag", "Description", "Job ID", "Git SHA", "Fail message"],
            index=0,
            key="recent_eval_jobs_search_scope",
            help="Choose which evaluator field the quick search should target.",
            label_visibility="collapsed",
        )
    with control_cols[3]:
        st.markdown('<div class="evj-toolbar-note">Search</div>', unsafe_allow_html=True)
        search_text = st.text_input(
            "Search",
            value=st.session_state.get("recent_eval_jobs_search_text", ""),
            key="recent_eval_jobs_search_text",
            help="Server-side search across the selected field.",
            label_visibility="collapsed",
            placeholder="Type to search evaluator jobs",
        ).strip()
    selected_user_name = ""
    user_candidates = sorted(
        {
            info.get("name", "").strip()
            for info in user_directory.values()
            if info.get("name", "").strip()
        },
        key=str.lower,
    )
    with control_cols[4]:
        st.markdown('<div class="evj-toolbar-note">User</div>', unsafe_allow_html=True)
        selected_user_name = st.selectbox(
            "User",
            options=[""] + user_candidates,
            index=0,
            key="recent_eval_jobs_user_filter",
            help="Filter jobs by resolved scheduled user name.",
            label_visibility="collapsed",
        )
    with control_cols[5]:
        st.markdown('<div class="evj-toolbar-note">From</div>', unsafe_allow_html=True)
        date_from = st.date_input(
            "From",
            value=st.session_state.get("recent_eval_jobs_date_from", None),
            key="recent_eval_jobs_date_from",
            label_visibility="collapsed",
            help="Scheduled-at lower bound in JST.",
        )
    with control_cols[6]:
        st.markdown('<div class="evj-toolbar-note">To</div>', unsafe_allow_html=True)
        date_to = st.date_input(
            "To",
            value=st.session_state.get("recent_eval_jobs_date_to", None),
            key="recent_eval_jobs_date_to",
            label_visibility="collapsed",
            help="Scheduled-at upper bound in JST.",
        )
    with control_cols[7]:
        st.markdown('<div class="evj-toolbar-note">Actions</div>', unsafe_allow_html=True)
        if st.button("Refresh", key="refresh_recent_eval_jobs", use_container_width=True):
            _fetch_recent_evaluator_job_pages.clear()
            _fetch_evaluator_job_detail.clear()
            st.rerun()

    page_key = "recent_eval_jobs_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    if date_from and date_to and date_from > date_to:
        st.warning("`From` date must be earlier than or equal to `To` date.")
        return

    def _render_job_list() -> None:
        nonlocal user_directory
        if not project_id:
            st.info("Enter a project id to browse recent evaluator jobs.")
            return
        current_page = max(1, int(st.session_state.get(page_key, 1)))
        pages_to_fetch = max(3, current_page + 2)
        if search_text or status_filter or date_from or date_to or selected_user_name:
            pages_to_fetch = max(pages_to_fetch, 6)
        server_status_values = tuple(_status_filter_values(status_filter))
        server_search_filter, search_needle = _build_recent_job_search_filter(search_text, search_scope, user_directory)
        selected_user_ids = sorted(
            {
                subject_id
                for subject_id, info in user_directory.items()
                if selected_user_name
                and selected_user_name.lower() == str(info.get("name") or "").strip().lower()
            }
        )
        server_date_filters = _build_recent_job_date_filters(date_from, date_to)
        extra_filters: List[Dict[str, Any]] = []
        if server_search_filter:
            extra_filters.append(server_search_filter)
        if selected_user_ids:
            extra_filters.append(
                {
                    "field": "scheduled_by",
                    "operator": "In",
                    "values": selected_user_ids,
                }
            )
        extra_filters.extend(server_date_filters)
        extra_filter_tuples = tuple(
            (
                str(f["field"]),
                str(f["operator"]),
                tuple(f.get("values", []) or []),
            )
            for f in extra_filters
        )
        fetch_help = "Loading evaluator jobs..."
        if search_text or status_filter or date_from or date_to or selected_user_name:
            fetch_help = "Loading evaluator jobs with filters..."
        try:
            with st.spinner(fetch_help):
                fetched_pages = _fetch_recent_evaluator_job_pages(
                    project_id,
                    environment,
                    limit,
                    pages_to_fetch,
                    status_values=server_status_values,
                    extra_filters=extra_filter_tuples,
                )
        except requests.Timeout:
            st.error("Timed out while loading evaluator jobs. The evaluator server may be slow right now. Try Refresh.")
            return
        except requests.RequestException as e:
            st.error(_friendly_request_error_message(e))
            return
        except Exception as e:
            st.error(_friendly_request_error_message(e))
            return
        if search_text:
            _save_recent_job_search_history(search_scope, search_text)

        jobs = [job for page in fetched_pages for job in page.get("jobs", [])]
        user_directory = _hydrate_recent_eval_user_directory(jobs, environment)
        has_more_from_api = bool(fetched_pages and fetched_pages[-1].get("next_token"))

        if not fetched_pages:
            st.warning("No response was returned from the evaluator server. Try Refresh.")
            return

        if search_needle:
            if search_scope == "Branch/tag":
                jobs = [job for job in jobs if search_needle in str(job.get("target", "")).lower()]
            elif search_scope == "Description":
                jobs = [job for job in jobs if search_needle in str(job.get("description", "")).lower() or search_needle in str(job.get("title", "")).lower()]
            elif search_scope == "Job ID":
                jobs = [job for job in jobs if search_needle in str(job.get("job_id", "")).lower()]
            elif search_scope == "Git SHA":
                jobs = [job for job in jobs if search_needle in str(job.get("git_sha", "")).lower()]
            elif search_scope == "Fail message":
                jobs = [job for job in jobs if search_needle in str(job.get("fail_message", "")).lower()]
        if selected_user_name:
            selected_lower = selected_user_name.lower()
            jobs = [
                job for job in jobs
                if selected_lower == str((user_directory.get(str(job.get("scheduled_by") or "").strip(), {}) or {}).get("name", "")).strip().lower()
            ]
        if status_filter:
            selected = {evaluator_api.normalize_job_status(v) for v in status_filter}
            jobs = [job for job in jobs if job.get("status_variant") in selected or evaluator_api.normalize_job_status(job.get("status", "")) in selected]

        if not jobs:
            st.session_state[page_key] = 1
            empty_message = "No recent evaluator jobs were returned."
            if search_text or status_filter or date_from or date_to or selected_user_name:
                empty_message = "No recent evaluator jobs matched the current filters."
            st.markdown(f'<div class="evj-empty">{html.escape(empty_message)}</div>', unsafe_allow_html=True)
            return

        total_loaded = len(jobs)
        has_next_page = total_loaded > current_page * limit or has_more_from_api
        max_known_page = max(1, (total_loaded + limit - 1) // limit)
        if current_page > max_known_page:
            current_page = max_known_page
            st.session_state[page_key] = current_page
        start_idx = (current_page - 1) * limit
        end_idx = start_idx + limit
        visible_jobs = jobs[start_idx:end_idx]
        if not visible_jobs and current_page > 1:
            current_page = max(1, current_page - 1)
            st.session_state[page_key] = current_page
            start_idx = (current_page - 1) * limit
            end_idx = start_idx + limit
            visible_jobs = jobs[start_idx:end_idx]
            has_next_page = total_loaded > current_page * limit

        if current_page == 1:
            page_numbers = list(range(1, min(3, max_known_page) + 1))
        else:
            page_numbers = list(
                range(
                    max(1, current_page - 1),
                    min(max_known_page, current_page + 1) + 1,
                )
            )
        pager_cols = st.columns([0.8, 0.9, 0.9, 0.9, 0.8, 5.7])
        with pager_cols[0]:
            if st.button("‹", key="recent_eval_jobs_prev", use_container_width=True, disabled=current_page <= 1):
                st.session_state[page_key] = max(1, current_page - 1)
                st.rerun()
        for idx, page_num in enumerate(page_numbers[:3], start=1):
            with pager_cols[idx]:
                btn_key = (
                    f"recent_eval_jobs_pagebtn_active_{page_num}"
                    if page_num == current_page
                    else f"recent_eval_jobs_pagebtn_{page_num}"
                )
                if st.button(
                    str(page_num),
                    key=btn_key,
                    use_container_width=True,
                    disabled=page_num == current_page,
                ):
                    st.session_state[page_key] = page_num
                    st.rerun()
        with pager_cols[4]:
            if st.button("›", key="recent_eval_jobs_next", use_container_width=True, disabled=not has_next_page):
                st.session_state[page_key] = current_page + 1
                st.rerun()

        selected_job_id = st.session_state.get("recent_eval_jobs_selected")
        if selected_job_id and not any(str(job.get("job_id", "")) == str(selected_job_id) for job in jobs):
            st.session_state.pop("recent_eval_jobs_selected", None)
            selected_job_id = None

        selected_run_job_id = st.session_state.get("recent_eval_jobs_run_selected")
        if selected_run_job_id and not any(str(job.get("job_id", "")) == str(selected_run_job_id) for job in jobs):
            st.session_state.pop("recent_eval_jobs_run_selected", None)
            selected_run_job_id = None

        selected_retest_job_id = st.session_state.get("recent_eval_jobs_retest_selected")
        if selected_retest_job_id and not any(str(job.get("job_id", "")) == str(selected_retest_job_id) for job in jobs):
            st.session_state.pop("recent_eval_jobs_retest_selected", None)
            selected_retest_job_id = None

        st.markdown('<div class="evj-list">', unsafe_allow_html=True)
        for job in visible_jobs:
            subject_id = str(job.get("scheduled_by") or "").strip()
            user_info = user_directory.get(subject_id, {})
            user_label = str(user_info.get("name") or subject_id or "Unknown").strip()
            row_cols = st.columns([9.2, 2.6])
            with row_cols[0]:
                _render_recent_evaluator_job_card(job, user_label=user_label)
            with row_cols[1]:
                action_cols = st.columns([1.0, 1.0, 1.0], gap="small")
                with action_cols[0]:
                    if st.button("Details", key=f"recent_eval_view_{job['job_id']}", use_container_width=True):
                        st.session_state["recent_eval_jobs_selected"] = str(job["job_id"])
                        _fetch_evaluator_job_detail.clear()
                        st.rerun()
                with action_cols[1]:
                    if st.button("Start", key=f"recent_eval_run_{job['job_id']}", use_container_width=True):
                        st.session_state["recent_eval_jobs_run_selected"] = str(job["job_id"])
                        _fetch_evaluator_job_detail.clear()
                        st.rerun()
                with action_cols[2]:
                    if st.button("Retest", key=f"recent_eval_retest_{job['job_id']}", use_container_width=True):
                        st.session_state["recent_eval_jobs_retest_selected"] = str(job["job_id"])
                        _fetch_evaluator_job_detail.clear()
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        selected_job_id = st.session_state.get("recent_eval_jobs_selected")
        if selected_job_id:
            selected_job = next((job for job in jobs if str(job.get("job_id", "")) == str(selected_job_id)), None)
            if selected_job:
                if callable(getattr(st, "dialog", None)):
                    try:
                        @st.dialog(f"Job details · {selected_job.get('title', '—')}", width="large")
                        def _recent_eval_job_dialog() -> None:
                            _render_recent_evaluator_job_detail(project_id, environment, selected_job)
                            if st.button("Close", key="recent_eval_jobs_close_detail", use_container_width=True):
                                st.session_state.pop("recent_eval_jobs_selected", None)
                                st.rerun()

                        _recent_eval_job_dialog()
                    finally:
                        st.session_state.pop("recent_eval_jobs_selected", None)
                else:
                    st.markdown('<div class="evj-detail">', unsafe_allow_html=True)
                    hdr_cols = st.columns([4.4, 1.1])
                    with hdr_cols[0]:
                        st.subheader(f"Job details · {selected_job.get('title', '—')}")
                    with hdr_cols[1]:
                        if st.button("Close", key="recent_eval_jobs_close_detail_fallback", use_container_width=True):
                            st.session_state.pop("recent_eval_jobs_selected", None)
                            st.rerun()
                    _render_recent_evaluator_job_detail(project_id, environment, selected_job)
                    st.markdown("</div>", unsafe_allow_html=True)

        selected_run_job_id = st.session_state.get("recent_eval_jobs_run_selected")
        if selected_run_job_id:
            selected_run_job = next((job for job in jobs if str(job.get("job_id", "")) == str(selected_run_job_id)), None)
            if selected_run_job:
                if callable(getattr(st, "dialog", None)):
                    try:
                        @st.dialog(f"Download + Eval + Parquet · {selected_run_job.get('title', '—')}", width="large")
                        def _recent_eval_run_dialog() -> None:
                            _render_recent_evaluator_job_run_dialog(
                                project_id,
                                environment,
                                selected_run_job,
                                output_path_default=output_path_default,
                                download_type_default=download_type_default,
                                phase_default=phase_default,
                                skip_large_file_default=skip_large_file_default,
                                large_file_mb_default=large_file_mb_default,
                                keep_zip_files_default=keep_zip_files_default,
                            )

                        _recent_eval_run_dialog()
                    finally:
                        if st.session_state.get("recent_eval_jobs_run_selected") == str(selected_run_job_id):
                            st.session_state.pop("recent_eval_jobs_run_selected", None)
                else:
                    st.markdown('<div class="evj-detail">', unsafe_allow_html=True)
                    hdr_cols = st.columns([4.4, 1.1])
                    with hdr_cols[0]:
                        st.subheader(f"Download + Eval + Parquet · {selected_run_job.get('title', '—')}")
                    with hdr_cols[1]:
                        if st.button("Close", key="recent_eval_jobs_close_run_fallback", use_container_width=True):
                            st.session_state.pop("recent_eval_jobs_run_selected", None)
                            st.rerun()
                    _render_recent_evaluator_job_run_dialog(
                        project_id,
                        environment,
                        selected_run_job,
                        output_path_default=output_path_default,
                        download_type_default=download_type_default,
                        phase_default=phase_default,
                        skip_large_file_default=skip_large_file_default,
                        large_file_mb_default=large_file_mb_default,
                        keep_zip_files_default=keep_zip_files_default,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

        selected_retest_job_id = st.session_state.get("recent_eval_jobs_retest_selected")
        if selected_retest_job_id:
            selected_retest_job = next((job for job in jobs if str(job.get("job_id", "")) == str(selected_retest_job_id)), None)
            if selected_retest_job:
                if callable(getattr(st, "dialog", None)):
                    try:
                        @st.dialog(f"Artifact retest · {selected_retest_job.get('title', '—')}", width="large")
                        def _recent_eval_retest_dialog() -> None:
                            _render_recent_evaluator_job_retest_dialog(
                                project_id,
                                environment,
                                selected_retest_job,
                                output_path_default=output_path_default,
                                phase_default=phase_default,
                            )

                        _recent_eval_retest_dialog()
                    finally:
                        if st.session_state.get("recent_eval_jobs_retest_selected") == str(selected_retest_job_id):
                            st.session_state.pop("recent_eval_jobs_retest_selected", None)
                else:
                    st.markdown('<div class="evj-detail">', unsafe_allow_html=True)
                    hdr_cols = st.columns([4.4, 1.1])
                    with hdr_cols[0]:
                        st.subheader(f"Artifact retest · {selected_retest_job.get('title', '—')}")
                    with hdr_cols[1]:
                        if st.button("Close", key="recent_eval_jobs_close_retest_fallback", use_container_width=True):
                            st.session_state.pop("recent_eval_jobs_retest_selected", None)
                            st.rerun()
                    _render_recent_evaluator_job_retest_dialog(
                        project_id,
                        environment,
                        selected_retest_job,
                        output_path_default=output_path_default,
                        phase_default=phase_default,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

    _render_job_list()
