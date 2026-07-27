"""
Evaluator Workflow page:
- browse finished local runs and launch compare views
- monitor server-side tasks
- start new evaluator pipelines
- run download/eval from existing evaluator jobs
"""

from __future__ import annotations

import html
import io
import json
import os
import re
import urllib.parse
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import requests
import yaml

from lib.db import (
    count_recent_tasks,
    create_task,
    is_task_queue_enabled,
    list_recent_tasks,
    update_task_rq_job_id,
    update_task_status,
)
from lib.page_chrome import (
    inject_app_page_styles,
    render_page_hero,
    section_header,
)
from lib.path_utils import (
    delete_run,
    format_size,
    get_data_root_display,
    get_run_info,
    get_run_storage_name,
    list_run_directories,
    resolve_run_subdirectory,
    resolve_under_data_root,
)
from lib.pr_test_branch_workflow import (
    DEFAULT_BRANCH_PREFIX,
    DEFAULT_PILOT_CHECKOUT,
    DEFAULT_PILOT_REPO_URL,
    DEFAULT_WORK_DIR,
)
from lib.run_metadata import (
    build_run_search_blob,
    read_run_metadata,
    upsert_run_metadata,
)
from lib.specsheet_report import (
    DEFAULT_TREND_TOPIC,
    DETECTION_TREND_TOPIC_BY_MODEL,
    parse_trend_metadata_text,
)
from lib.ui.recent_evaluator_jobs import (
    _fetch_evaluator_job_detail,
    _format_source_ref_html,
    _format_source_ref_text,
    _render_recent_evaluator_job_retest_dialog,
    _render_recent_evaluator_jobs_section,
    configure_recent_evaluator_jobs_ui,
)
from lib.ui.task_history import get_task_list_current_user, render_task_list
from lib.ui.styles_download import inject_download_page_styles
from lib.auth import get_current_user_identity
from lib.user_config import UserConfig

try:
    from lib.perception_catalog_io import pkl_archive_to_parquet

    CATALOG_IO_AVAILABLE = True
except ImportError:
    CATALOG_IO_AVAILABLE = False

_JST = timezone(timedelta(hours=9))
_TASK_LIST_MAX_ROWS = 200
_TASK_LIST_SINCE_DAYS = 7
_RELEASE_PERFORMANCE_CATALOG_ID = "e36d75b9-6c3a-4970-9b9b-5cd13f7a9da3"
_RELEASE_PERFORMANCE_INTEGRATION_ID = "96ad8fba-0228-4c2b-9166-07d4de1a0760"
_RELEASE_DEVOPS_CATALOG_ID = "ab0f8498-cc1b-4726-836f-e18e8bcb3200"
_RELEASE_DEVOPS_INTEGRATION_ID = "295cff78-9bc9-4d60-b7aa-f95be6ff96a4"
_RELEASE_OPTIONAL_CATALOG_ID = "09039022-ec91-41bf-9e93-fdefccdfc9bc"
_RELEASE_SKIP_LARGE_FILE = True
_RELEASE_LARGE_FILE_MB = 50.0
_DEFAULT_MAX_WAIT_HOURS = 48
_WORKFLOW_KIND_PERCEPTION = "Perception"
_WORKFLOW_KIND_TLR = "TLR"
_DEFAULT_PERCEPTION_PHASE = "perception.object_recognition.objects"
_TLR_DOWNLOAD_TYPE = "Result JSON"
_RELEASE_TREND_TOPIC_OPTIONS = {
    "Prediction / object recognition": DEFAULT_TREND_TOPIC,
    "ML model / CenterPoint": DETECTION_TREND_TOPIC_BY_MODEL["centerpoint"],
    "ML model / BEVFusion": DETECTION_TREND_TOPIC_BY_MODEL["bevfusion"],
    "Custom": "",
}
_TASK_HISTORY_RANGE_OPTIONS = {
    "7 days": 7,
    "30 days": 30,
    "90 days": 90,
    "All": None,
}
_WORKFLOW_START_DIALOG_KEY = "workflow_start_dialog_open"
_WORKFLOW_PR_BRANCH_DIALOG_KEY = "workflow_pr_branch_dialog_open"


st.set_page_config(
    page_title="Evaluator Workflow",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_app_page_styles()
inject_download_page_styles()


_user_config = UserConfig(warning_fn=st.warning)


def _looks_like_tlr_catalog(*values: object) -> bool:
    text = " ".join(str(value or "") for value in values).lower()
    return "tlr" in text or "traffic light" in text or "traffic_light" in text


def _parse_rq_timeout_sec(raw: Optional[str], *, default: int, minimum: int) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(minimum, int(str(raw).strip(), 10))
    except ValueError:
        return default


_RQ_JOB_TIMEOUT_DEFAULT_SEC = 7 * 24 * 3600
_RQ_DEFAULT_JOB_TIMEOUT_SEC = _parse_rq_timeout_sec(
    os.environ.get("RQ_JOB_TIMEOUT_SEC"),
    default=_RQ_JOB_TIMEOUT_DEFAULT_SEC,
    minimum=60,
)


def get_config_value(key: str, default=None):
    return _user_config.get(key, default)


def set_config_value(key: str, value) -> None:
    _user_config.set(key, value)


def _open_exclusive_workflow_dialog(dialog_key: str) -> None:
    for key in (_WORKFLOW_START_DIALOG_KEY, _WORKFLOW_PR_BRANCH_DIALOG_KEY):
        st.session_state[key] = key == dialog_key


def _close_workflow_dialog(dialog_key: str) -> None:
    st.session_state[dialog_key] = False


def _close_workflow_dialogs() -> None:
    _close_workflow_dialog(_WORKFLOW_START_DIALOG_KEY)
    _close_workflow_dialog(_WORKFLOW_PR_BRANCH_DIALOG_KEY)


def _to_jst(dt):
    if dt is None:
        return None
    try:
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_JST)
    except Exception:
        return None


def _load_catalog_presets():
    app_root = Path(__file__).parent.parent
    catalogs_filename = "catalogs.json"
    search_paths = [
        app_root / catalogs_filename,
        Path(os.environ.get("CATALOGS_PATH", "")),
        Path.cwd() / catalogs_filename,
    ]
    catalogs = []
    loaded_path = None
    load_error = None
    for path in search_paths:
        if path.exists() and path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    catalogs = data.get("catalogs", [])
                elif isinstance(data, list):
                    catalogs = data
                else:
                    catalogs = []
                loaded_path = str(path)
                load_error = None
                break
            except Exception as exc:
                load_error = str(exc)
    presets = []
    for item in catalogs:
        if not isinstance(item, dict):
            continue
        display_name = item.get("display_name") or item.get("name") or item.get("catalog_id", "Unknown")
        presets.append({**item, "display_name": display_name})
    return presets, loaded_path, load_error


def _fetch_server_catalogs(project_id: str, environment: str) -> List[Dict[str, str]]:
    """Fetch available catalogs for the project on demand."""
    if not project_id:
        return []
    import os
    from lib.WebAPI import catalogAPI

    os.environ["AUTH_PROFILE"] = environment or "default"
    response = catalogAPI(project_id=project_id).list_catalogs()
    response.raise_for_status()
    data = response.json()
    raw_catalogs = data.get("catalogs", []) if isinstance(data, dict) else data
    options: List[Dict[str, str]] = []
    for item in raw_catalogs or []:
        if not isinstance(item, dict):
            continue
        catalog_id = str(item.get("id") or item.get("catalog_id") or "").strip()
        display_name = str(item.get("display_name") or item.get("name") or catalog_id).strip()
        if not catalog_id or not display_name:
            continue
        options.append(
            {
                "catalog_id": catalog_id,
                "display_name": display_name,
                "description": str(item.get("description") or "").strip(),
            }
        )
    options.sort(key=lambda item: item["display_name"].lower())
    return options


def _resolve_integration_id_for_catalog(project_id: str, environment: str, catalog_id: str) -> str:
    """Resolve the most relevant active integration for a catalog."""
    if not project_id or not catalog_id:
        return ""
    from lib import evaluator_api

    os.environ["AUTH_PROFILE"] = environment or "default"
    api = evaluator_api.EvaluationRunAPI()
    url = f"{api.api_base_url}/projects/{project_id}/integrations"
    response = api.request(url, {"catalog_id": catalog_id, "size": 100}, method="GET")
    if response is None:
        raise RuntimeError("No response returned while loading integrations.")
    if response.status_code != 200:
        raise RuntimeError(f"Failed to load integrations: status={response.status_code}")

    payload = json.loads(response.content)
    integrations = payload.get("integrations", []) or []
    active = [
        item for item in integrations
        if isinstance(item, dict)
        and str(item.get("catalog_id") or "").strip() == catalog_id
        and not bool(item.get("deleted"))
    ]
    if not active:
        raise RuntimeError("No active integration was found for the selected catalog.")

    def _sort_key(item: Dict[str, object]) -> tuple:
        return (
            str(item.get("updated_at") or ""),
            int(item.get("version_id") or 0),
            str(item.get("id") or ""),
        )

    active.sort(key=_sort_key, reverse=True)
    return str(active[0].get("id") or "").strip()


def _enqueue_task(task_type: str, params: dict) -> Optional[str]:
    task_id = None
    try:
        identity = get_current_user_identity()
        session_id = str(identity.get("id") or "").strip() or None
        params = dict(params)
        if session_id:
            params.setdefault("_requester", identity)
        task_id = create_task(task_type, params, session_id=session_id)
        if not task_id:
            st.error("Failed to create task row. Check DATABASE_URL and task parameters.")
            return None

        from redis import Redis
        from rq import Queue
        from worker.tasks import run_job

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        redis_conn = Redis.from_url(redis_url)
        queue = Queue(
            name=os.environ.get("RQ_QUEUE", "default"),
            connection=redis_conn,
            default_timeout=_RQ_DEFAULT_JOB_TIMEOUT_SEC,
        )
        job = queue.enqueue(
            run_job,
            task_id,
            task_type,
            params,
            job_timeout=_RQ_DEFAULT_JOB_TIMEOUT_SEC,
            result_ttl=_RQ_DEFAULT_JOB_TIMEOUT_SEC,
        )
        rq_id = getattr(job, "id", None)
        if rq_id:
            update_task_rq_job_id(task_id, str(rq_id))
        return task_id
    except Exception as exc:
        if task_id:
            update_task_status(task_id, "failed", error_message=f"Failed to enqueue RQ job: {exc}")
        st.error(f"Failed to enqueue task: {exc}")
        return None


def _make_default_output_path(branch_name: str) -> str:
    import re

    clean_branch = re.sub(r"[^\w]", "_", branch_name.strip("/")) if branch_name else "eval"
    clean_branch = re.sub(r"_+", "_", clean_branch).strip("_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{clean_branch}_{ts}"


def _safe_output_part(value: object, fallback: str) -> str:
    text = re.sub(r"[^\w.\-]+", "_", str(value or "").strip()).strip("._")
    return text or fallback


def _catalog_preset_emoji(preset_name: str, *, has_custom_catalog: bool = False) -> str:
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


def _make_auto_workflow_description(
    target_name: str,
    preset_name: str = "",
    *,
    has_custom_catalog: bool = False,
) -> str:
    import re

    clean_target = str(target_name or "").strip() or "default"
    clean_target = re.sub(r"\s+", " ", clean_target)
    stamp = datetime.now().strftime("%m-%d %H:%M")
    return (
        f"🚀 evaluator workflow [{clean_target}] [{stamp}] "
        f"{_catalog_preset_emoji(preset_name, has_custom_catalog=has_custom_catalog)}"
    )


def _make_auto_release_workflow_description(target_name: str) -> str:
    clean_target = str(target_name or "").strip() or "default"
    clean_target = re.sub(r"\s+", " ", clean_target)
    stamp = datetime.now().strftime("%m-%d %H:%M")
    return f"🚀 release workflow [{clean_target}] [{stamp}]"


def _make_default_release_pilot_auto_version(target_name: str) -> str:
    target = str(target_name or "").strip()
    match = re.search(r"v?(\d+\.\d+\.\d+)", target)
    if match:
        return f"Pilot.Auto v{match.group(1)}"
    return f"Pilot.Auto {target}" if target else "Pilot.Auto release"


def _make_default_release_metadata_text(target_name: str) -> str:
    release_group = _safe_output_part(target_name, "release")
    pilot_auto_version = _make_default_release_pilot_auto_version(target_name)
    description = f"{target_name} release data update" if target_name else "Release data update"
    date = datetime.now(_JST).strftime("%Y.%m.%d")
    return (
        "tags: [trend]\n"
        f"release_group: {release_group}\n"
        f'pilot_auto_version: "{pilot_auto_version}"\n'
        f"pilot_auto_version_abbr: {_safe_output_part(pilot_auto_version.replace('Pilot.Auto', '').strip(), 'release')[:16]}\n"
        "data_count: 99,776+\n"
        f"description: {description}\n"
        f"date: {date}\n"
        f"topic_name: {DEFAULT_TREND_TOPIC}\n"
    )


def _looks_like_release_trend_metadata_text(text: str) -> bool:
    try:
        data = yaml.safe_load(text or "")
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    tags = data.get("tags")
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list) or not any(str(tag).strip() == "trend" for tag in tags):
        return False
    return bool(
        str(data.get("pilot_auto_version") or "").strip()
        and str(data.get("data_count") or "").strip()
        and str(data.get("date") or "").strip()
    )


def _release_target_from_metadata(metadata: dict) -> str:
    for key in ("target_name", "target", "git_ref"):
        value = str(metadata.get(key) or "").strip()
        if not value:
            continue
        for prefix in ("refs/heads/", "refs/tags/"):
            if value.startswith(prefix):
                return value[len(prefix):]
        return value
    return ""


def _load_existing_release_context(output_path: str) -> dict[str, object]:
    """Return existing release metadata/job context for an output folder, if present."""
    context: dict[str, object] = {
        "metadata_text": "",
        "metadata_source": "",
        "target_name": "",
        "job_ids": {},
    }
    if not str(output_path or "").strip():
        return context
    resolved_output, path_error = resolve_under_data_root(output_path, allow_missing=True)
    if path_error or resolved_output is None:
        return context

    run_metadata = read_run_metadata(resolved_output)
    request_meta = run_metadata.get("request") if isinstance(run_metadata.get("request"), dict) else {}
    parameter_meta = request_meta.get("parameters") if isinstance(request_meta.get("parameters"), dict) else {}
    release_specsheet = (
        run_metadata.get("release_specsheet")
        if isinstance(run_metadata.get("release_specsheet"), dict)
        else {}
    )
    evaluator_jobs = (
        release_specsheet.get("evaluator_jobs")
        if isinstance(release_specsheet.get("evaluator_jobs"), dict)
        else {}
    )
    target_name = str(
        parameter_meta.get("target_name")
        or request_meta.get("target_name")
        or release_specsheet.get("target_name")
        or ""
    ).strip()
    job_ids: dict[str, str] = {}
    for role in ("performance", "devops", "planning_test"):
        role_meta = evaluator_jobs.get(role) if isinstance(evaluator_jobs.get(role), dict) else {}
        job_id = str(
            role_meta.get("job_id")
            or parameter_meta.get(f"{role}_job_id")
            or request_meta.get(f"{role}_job_id")
            or ""
        ).strip()
        if job_id:
            job_ids[role] = job_id

    candidates = [
        ("", resolved_output / "metadata.yaml"),
        ("performance", resolved_output / "performance" / "resources" / "metadata.yaml"),
        ("devops", resolved_output / "devops" / "resources" / "metadata.yaml"),
        ("performance", resolved_output / "performance" / "metadata.yaml"),
        ("devops", resolved_output / "devops" / "metadata.yaml"),
    ]
    for role, candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
            metadata = yaml.safe_load(text or "") or {}
        except Exception:
            continue
        if _looks_like_release_trend_metadata_text(text):
            if not context["metadata_text"]:
                context["metadata_text"] = text
                context["metadata_source"] = str(candidate)
            if not target_name and isinstance(metadata, dict):
                target_name = _release_target_from_metadata(metadata)
            if role and isinstance(metadata, dict):
                job_id = str(metadata.get("job_id") or "").strip()
                if job_id:
                    job_ids.setdefault(role, job_id)
    context["target_name"] = target_name
    context["job_ids"] = job_ids
    return context


def _load_existing_release_metadata_text(output_path: str) -> tuple[str, str]:
    context = _load_existing_release_context(output_path)
    return str(context.get("metadata_text") or ""), str(context.get("metadata_source") or "")


def _release_role_has_local_artifacts(output_path: str, role: str) -> bool:
    resolved_output, path_error = resolve_under_data_root(output_path, allow_missing=True)
    if path_error or resolved_output is None:
        return False
    role_path = resolved_output / role
    if not role_path.exists():
        return False
    if (role_path / "current.parquet").exists():
        return True
    if any(role_path.glob("*.parquet")):
        return True
    return any(role_path.rglob("scene_result.pkl")) or any(role_path.rglob("*.pkl.z"))


def _extract_release_metadata_topic(text: str) -> str:
    try:
        metadata = parse_trend_metadata_text(text)
        return str(metadata.get("topic_name") or DEFAULT_TREND_TOPIC).strip()
    except Exception:
        match = re.search(r"(?m)^topic_name\s*:\s*['\"]?([^'\"\n#]+)", text or "")
        return match.group(1).strip() if match else DEFAULT_TREND_TOPIC


def _replace_release_metadata_topic(text: str, topic: str) -> str:
    topic = str(topic or "").strip()
    if not topic:
        return text
    line = f"topic_name: {topic}"
    if re.search(r"(?m)^topic_name\s*:", text or ""):
        return re.sub(r"(?m)^topic_name\s*:.*$", line, text)
    return (text.rstrip() + "\n" + line + "\n") if text else line + "\n"


def _format_run_mtime(mtime: float) -> str:
    if not mtime:
        return "—"
    try:
        return datetime.fromtimestamp(mtime, tz=_JST).strftime("%Y-%m-%d %H:%M JST")
    except Exception:
        return "—"


def _run_row_key(run: Dict[str, object]) -> str:
    run_path = run.get("run_path")
    if isinstance(run_path, Path):
        return get_run_storage_name(run_path)
    return str(run.get("name") or "").strip()


def _normalize_compare_run_key(
    value: str,
    runs_by_key: Dict[str, Dict[str, object]],
    runs_by_name: Dict[str, List[Dict[str, object]]],
) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if value in runs_by_key:
        return value
    matches = runs_by_name.get(value, [])
    if matches:
        return _run_row_key(matches[0])
    return value


def _build_overview_url(run_a: str, compare_runs: Optional[List[str]] = None) -> str:
    query = {"mode": "single", "run_a": run_a}
    valid_compare_runs = [str(name).strip() for name in (compare_runs or []) if str(name).strip()]
    if valid_compare_runs:
        query["mode"] = "compare"
        for idx, run_name in enumerate(valid_compare_runs[:4]):
            query[f"run_{chr(98 + idx)}"] = run_name
    return f"/?{urllib.parse.urlencode(query)}"


def _format_metadata_time(value: object) -> str:
    if not value:
        return "—"
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return str(value)
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.astimezone(_JST).strftime("%Y-%m-%d %H:%M JST")
    except Exception:
        return str(value)


def _metadata_text(value: object) -> str:
    text = str(value or "").strip()
    return text or "—"


def _run_user_label(subject_id: str, environment: str) -> str:
    subject = str(subject_id or "").strip()
    if not subject:
        return "(Auto)"
    if not subject.startswith("t4:"):
        return subject
    try:
        profile = _resolve_subject_name(subject, environment or "default")
        name = str(profile.get("name") or subject).strip()
        return name or subject
    except Exception:
        return "(Auto)"


def _catalog_url(project_id: str, catalog_id: str, metadata_url: str = "") -> str:
    direct_url = str(metadata_url or "").strip()
    if direct_url:
        return direct_url
    project = str(project_id or "").strip()
    catalog = str(catalog_id or "").strip()
    if project and catalog:
        return f"https://evaluation.tier4.jp/evaluation/vehicle_catalogs/{catalog}?project_id={project}"
    return ""


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _catalog_preset_name_map() -> Dict[str, str]:
    presets, _, _ = _load_catalog_presets()
    mapping: Dict[str, str] = {}
    for item in presets:
        if not isinstance(item, dict):
            continue
        catalog_id = str(item.get("catalog_id") or "").strip()
        display_name = str(item.get("display_name") or item.get("name") or "").strip()
        if catalog_id and display_name:
            mapping[catalog_id] = display_name
    return mapping


def _catalog_label_for_run(catalog_id: str, catalog_name: str) -> str:
    resolved_name = str(catalog_name or "").strip()
    if resolved_name:
        return resolved_name
    catalog = str(catalog_id or "").strip()
    if not catalog:
        return "—"
    preset_match = _catalog_preset_name_map().get(catalog, "").strip()
    return preset_match or catalog


@st.cache_data(ttl=15, show_spinner=False)
def _load_local_runs() -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for run_path in list_run_directories():
        info = get_run_info(run_path)
        metadata = read_run_metadata(run_path)
        owner_meta = metadata.get("owner") if isinstance(metadata.get("owner"), dict) else {}
        task_meta = metadata.get("task") if isinstance(metadata.get("task"), dict) else {}
        requester_meta = task_meta.get("requester") if isinstance(task_meta.get("requester"), dict) else {}
        request_meta = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
        evaluator_meta = metadata.get("evaluator") if isinstance(metadata.get("evaluator"), dict) else {}
        description = str(
            request_meta.get("description")
            or evaluator_meta.get("description")
            or ""
        ).strip()
        requested_by = str(
            owner_meta.get("id")
            or requester_meta.get("id")
            or task_meta.get("requested_by")
            or evaluator_meta.get("scheduled_by")
            or ""
        ).strip()
        environment = str(request_meta.get("environment") or "default").strip() or "default"
        requested_by_label = str(
            owner_meta.get("name")
            or owner_meta.get("email")
            or requester_meta.get("name")
            or requester_meta.get("email")
            or ""
        ).strip() or _run_user_label(requested_by, environment)
        task_type = str(task_meta.get("type") or metadata.get("source_mode") or "").strip()
        task_status = str(task_meta.get("status") or "").strip()
        evaluator_job_id = str(
            evaluator_meta.get("job_id")
            or request_meta.get("job_id")
            or ""
        ).strip()
        evaluator_report_url = str(evaluator_meta.get("report_url") or "").strip()
        evaluator_title = str(
            evaluator_meta.get("title")
            or description
            or evaluator_job_id
            or ""
        ).strip()
        evaluator_target = str(
            evaluator_meta.get("target")
            or request_meta.get("target_name")
            or ""
        ).strip()
        catalog_id = str(
            evaluator_meta.get("catalog_id")
            or request_meta.get("catalog_id")
            or ""
        ).strip()
        catalog_name = str(evaluator_meta.get("catalog_name") or "").strip()
        catalog_label = _catalog_label_for_run(catalog_id, catalog_name)
        catalog_url = _catalog_url(
            str(request_meta.get("project_id") or "").strip(),
            catalog_id,
            str(evaluator_meta.get("catalog_url") or "").strip(),
        )
        case_totals = evaluator_meta.get("case_totals") if isinstance(evaluator_meta.get("case_totals"), dict) else {}
        passed_count = int(case_totals.get("success", 0) or 0)
        failed_count = int(case_totals.get("failed", 0) or 0)
        canceled_count = int(case_totals.get("canceled", 0) or 0)
        search_blob = build_run_search_blob(
            run_path,
            metadata,
            extra_values=[
                description,
                requested_by,
                requested_by_label,
                task_type,
                task_status,
                evaluator_job_id,
                catalog_id,
                catalog_name,
                evaluator_target,
            ],
        )
        runs.append(
            {
                "name": info["name"],
                "run_path": run_path,
                "path_display": f"{get_data_root_display()}/{info['name']}",
                "size": format_size(info["size_bytes"]),
                "mtime": float(info["mtime"] or 0),
                "mtime_date": _to_jst(datetime.fromtimestamp(float(info["mtime"] or 0), tz=timezone.utc)).date() if info["mtime"] else None,
                "modified": _format_run_mtime(info["mtime"]),
                "has_summary": bool(info["has_summary"]),
                "has_score": bool(info["has_score"]),
                "has_parquet": bool(info["has_parquet"]),
                "metadata": metadata,
                "description": description,
                "requested_by": requested_by,
                "requested_by_label": requested_by_label,
                "environment": environment,
                "project_id": str(request_meta.get("project_id") or "").strip(),
                "task_type": task_type,
                "task_status": task_status,
                "evaluator_job_id": evaluator_job_id,
                "evaluator_report_url": evaluator_report_url,
                "evaluator_title": evaluator_title,
                "evaluator_target": evaluator_target,
                "branch_label": evaluator_target,
                "evaluator_git_sha": str(evaluator_meta.get("git_sha") or "").strip(),
                "evaluator_git_ref_url": str(evaluator_meta.get("git_ref_url") or "").strip(),
                "evaluator_git_commit_url": str(evaluator_meta.get("git_commit_url") or "").strip(),
                "evaluator_source_url": str(evaluator_meta.get("source_url") or "").strip(),
                "evaluator_source_repo_label": str(evaluator_meta.get("source_repo_label") or "").strip(),
                "catalog_id": catalog_id,
                "catalog_name": catalog_name,
                "catalog_label": catalog_label,
                "catalog_url": catalog_url,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "canceled_count": canceled_count,
                "search_blob": search_blob,
            }
        )
    runs.sort(key=lambda row: (-float(row["mtime"]), str(row["name"]).lower()))
    return runs


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _resolve_subject_name(subject_id: str, environment: str) -> Dict[str, str]:
    subject = str(subject_id or "").strip()
    if not subject or not subject.startswith("t4:"):
        return {"subject_id": subject, "name": subject, "email": ""}
    org_id = os.environ.get(
        "WEBAUTO_ORGANIZATION_ID",
        "5a21621d-6968-4f7d-94f8-99cfb77b6e71",
    ).strip()
    if not org_id:
        return {"subject_id": subject, "name": subject, "email": ""}
    os.environ["AUTH_PROFILE"] = environment or "default"
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
        "subject_id": str(data.get("subject_id") or subject).strip(),
        "name": str(data.get("name") or subject).strip(),
        "email": str(data.get("email") or "").strip(),
    }


def _inject_workflow_page_styles() -> None:
    st.markdown(
        """
        <style>
        .wf-toolbar-note {
            margin: 0 0 0.28rem 0;
            font-size: 0.66rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 700;
        }
        .wf-panel {
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.98) 100%);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 18px 50px -28px rgba(15, 23, 42, 0.22);
            margin-bottom: 1rem;
        }
        .wf-panel-title {
            margin: 0;
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
        }
        .wf-panel-copy {
            margin: 0.3rem 0 0 0;
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .wf-filter-strip,
        .wf-pager-strip {
            border: none;
            background: linear-gradient(180deg, rgba(248,250,252,0.72) 0%, rgba(248,250,252,0.28) 100%);
            border-radius: 14px;
            padding: 0.72rem 0.78rem 0.28rem 0.78rem;
            box-shadow: none;
            margin-bottom: 0.32rem;
        }
        .wf-filter-strip {
            margin-top: 0.12rem;
        }
        .wf-pager-strip {
            padding-top: 0.28rem;
            padding-bottom: 0.28rem;
        }
        .wf-pager-summary {
            padding-top: 0.2rem;
            color: #475569;
            font-size: 0.82rem;
            line-height: 1.35;
        }
        .wf-pager-summary strong {
            color: #0f172a;
            font-weight: 700;
        }
        .wf-meta-inline {
            margin-top: 0.2rem;
            color: #64748b;
            font-size: 0.78rem;
        }
        .wf-meta-inline a {
            color: inherit;
            text-decoration: none;
        }
        .wf-meta-inline a:hover {
            text-decoration: underline;
        }
        .wf-linked-ref {
            margin-top: 0.35rem;
            color: #475569;
            font-size: 0.82rem;
            line-height: 1.35;
        }
        .wf-linked-ref a {
            color: #0f766e;
            text-decoration: none;
            font-weight: 600;
        }
        .wf-linked-ref a:hover {
            text-decoration: underline;
        }
        .wf-run-list {
            display: block;
            margin-top: 0.35rem;
        }
        .wf-run-name {
            min-width: 0;
        }
        .wf-run-title {
            font-size: 0.8rem;
            line-height: 1.2;
            font-weight: 700;
            color: #0f172a;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .wf-run-title a {
            color: inherit;
            text-decoration: none;
        }
        .wf-run-title a:hover {
            text-decoration: underline;
        }
        .wf-run-title--muted,
        .wf-run-title--muted a {
            color: #94a3b8 !important;
        }
        .wf-run-cell {
            min-width: 0;
            color: #0f172a;
            font-size: 0.78rem;
            line-height: 1.15;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .wf-run-cell--muted {
            color: #94a3b8;
        }
        .wf-run-text {
            padding-top: 0.26rem;
        }
        .wf-meta-inline--muted {
            color: #94a3b8;
        }
        .wf-run-code {
            padding-top: 0.22rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.74rem;
            line-height: 1.22;
            color: #0f172a;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-all;
        }
        .wf-run-flags {
            display: flex;
            flex-wrap: nowrap;
            gap: 0.24rem;
            padding-top: 0.18rem;
            overflow: hidden;
        }
        .wf-flag {
            display: inline-flex;
            align-items: center;
            padding: 0.12rem 0.38rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            background: #e2e8f0;
            color: #475569;
            white-space: nowrap;
        }
        .wf-flag--ok {
            background: #dcfce7;
            color: #166534;
        }
        .wf-run-flags--muted {
            opacity: 0.58;
        }
        .wf-unavailable-note {
            margin-top: 0.18rem;
            font-size: 0.68rem;
            color: #94a3b8;
            letter-spacing: 0.01em;
        }
        .wf-compare-bar {
            border: none;
            background: linear-gradient(135deg, rgba(248,250,252,0.65) 0%, rgba(236,254,255,0.55) 100%);
            border-radius: 12px;
            padding: 0.62rem 0.78rem;
            margin: 0.18rem 0 0.4rem 0;
        }
        .wf-compare-title {
            margin: 0;
            font-size: 0.8rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: 0.01em;
        }
        [class*="st-key-workflow_compare_pick__"] label[data-testid="stWidgetLabel"] {
            display: none;
        }
        [class*="st-key-workflow_compare_pick__"] div[data-testid="stCheckbox"] {
            display: flex;
            justify-content: center;
            padding-top: 0.1rem;
        }
        [class*="st-key-workflow_compare_pick__"] input[type="checkbox"] {
            transform: scale(1.2);
        }
        [class*="st-key-workflow_runs_page_select"] div[data-baseweb="select"] {
            min-height: 1.72rem;
        }
        [class*="st-key-workflow_runs_page_select"] [data-baseweb="select"] > div {
            min-height: 1.72rem;
            font-size: 0.8rem;
        }
        [class*="st-key-workflow_runs_page_prev"] button,
        [class*="st-key-workflow_runs_page_next"] button {
            min-height: 1.72rem;
            height: 1.72rem;
            padding: 0 0.3rem;
            font-size: 0.8rem;
            line-height: 1;
        }
        [class*="st-key-workflow_run_details__"] button,
        [class*="st-key-workflow_run_download__"] button,
        [class*="st-key-workflow_run_delete__"] button,
        [class*="st-key-workflow_local_run_retest__"] button {
            white-space: nowrap;
            min-height: 2.2rem;
            font-size: 0.72rem;
            padding-left: 0.35rem;
            padding-right: 0.35rem;
            letter-spacing: 0.01em;
        }
        .wf-launcher {
            border: 1px solid rgba(20, 184, 166, 0.22);
            background: linear-gradient(135deg, #f0fdfa 0%, #ffffff 100%);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.8rem;
        }
        .wf-launcher-title {
            margin: 0;
            font-size: 0.95rem;
            font-weight: 800;
            color: #0f172a;
        }
        .wf-launcher-copy {
            margin: 0.25rem 0 0 0;
            font-size: 0.84rem;
            color: #475569;
        }
        .wf-launcher-meta {
            margin-top: 0.55rem;
            font-size: 0.78rem;
            color: #475569;
        }
        .wf-empty {
            border: 1px dashed rgba(148, 163, 184, 0.45);
            border-radius: 12px;
            background: rgba(248, 250, 252, 0.8);
            padding: 0.8rem 0.9rem;
            color: #475569;
            font-size: 0.84rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_local_run_artifact_list(run_name: str) -> tuple[Optional[Path], list[tuple[Path, str]], str]:
    run_path, err = resolve_run_subdirectory(run_name)
    if err:
        return None, [], err
    assert run_path is not None
    to_zip: list[tuple[Path, str]] = []
    summary_file = run_path / "Summary.csv"
    score_file = run_path / "Score.csv"
    if summary_file.is_file():
        to_zip.append((summary_file, "Summary.csv"))
    if score_file.is_file():
        to_zip.append((score_file, "Score.csv"))
    for pq in sorted(run_path.glob("*.parquet"), key=lambda p: p.name.lower()):
        to_zip.append((pq, pq.name))
    return run_path, to_zip, ""


def _render_local_run_download_dialog(run_name: str) -> None:
    run_path, to_zip, err = _build_local_run_artifact_list(run_name)
    if err:
        st.error(err)
        return
    if run_path is None:
        st.error("Run path could not be resolved.")
        return

    prepared_key = f"workflow_zip_prepared::{run_name}"
    st.caption("Download the generated local artifacts for this run as one ZIP.")
    if not to_zip:
        st.info("This run has no Summary.csv, Score.csv, or top-level `.parquet` files.")
        return

    st.caption(f"**{len(to_zip)}** file(s): {', '.join(arc for _, arc in to_zip)}")
    prepared = st.session_state.get(prepared_key)

    if st.button("Prepare ZIP", key=f"workflow_prepare_zip::{run_name}", use_container_width=True):
        buf = io.BytesIO()
        zip_errors: list[str] = []
        included: list[str] = []
        with st.spinner("Building ZIP…"):
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fpath, arcname in to_zip:
                    try:
                        zf.write(fpath, arcname=arcname)
                        included.append(arcname)
                    except OSError as exc:
                        zip_errors.append(f"{arcname}: {exc}")
        for msg in zip_errors:
            st.warning(msg)
        if included:
            safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", run_name).strip() or "run"
            st.session_state[prepared_key] = {
                "data": buf.getvalue(),
                "file_name": f"{safe_stem}_artifacts.zip",
            }
            prepared = st.session_state.get(prepared_key)
        else:
            st.session_state.pop(prepared_key, None)
            prepared = None
            st.error("Could not add any files to the ZIP.")

    if prepared and prepared.get("data"):
        st.download_button(
            label=f"Download {prepared['file_name']}",
            data=prepared["data"],
            file_name=prepared["file_name"],
            mime="application/zip",
            key=f"workflow_dl_zip::{run_name}",
            use_container_width=True,
        )


def _render_local_run_delete_dialog(run_name: str, *, confirm_label: Optional[str] = None) -> None:
    confirm_target = str(confirm_label or run_name).strip()
    st.warning("This deletes the local run directory permanently.")
    confirm = st.text_input(
        "Type the run name to confirm",
        value="",
        placeholder=confirm_target,
        key=f"workflow_delete_confirm::{run_name}",
    ).strip()
    if st.button("Delete run", key=f"workflow_delete_btn::{run_name}", type="primary", use_container_width=True):
        if confirm not in {confirm_target, run_name}:
            st.error("Confirmation text does not match the run name.")
            return
        ok, msg = delete_run(run_name)
        if ok:
            st.session_state.pop("workflow_local_run_detail", None)
            st.session_state.pop("workflow_local_run_download", None)
            st.session_state.pop("workflow_local_run_delete", None)
            st.session_state.pop(f"workflow_zip_prepared::{run_name}", None)
            st.success(msg)
            _load_local_runs.clear()
            st.rerun()
        st.error(msg)


def _render_local_runs_header() -> None:
    header_cols = st.columns([0.45, 2.35, 0.72, 1.45, 1.55, 1.0, 1.0, 1.22, 0.68, 1.55], gap="small")
    header_cols[0].markdown('<div class="wf-toolbar-note">Pick</div>', unsafe_allow_html=True)
    header_cols[1].markdown('<div class="wf-toolbar-note">Name</div>', unsafe_allow_html=True)
    header_cols[2].markdown('<div class="wf-toolbar-note">User</div>', unsafe_allow_html=True)
    header_cols[3].markdown('<div class="wf-toolbar-note">Catalog</div>', unsafe_allow_html=True)
    header_cols[4].markdown('<div class="wf-toolbar-note">Evaluator</div>', unsafe_allow_html=True)
    header_cols[5].markdown('<div class="wf-toolbar-note">Result</div>', unsafe_allow_html=True)
    header_cols[6].markdown('<div class="wf-toolbar-note">Updated</div>', unsafe_allow_html=True)
    header_cols[7].markdown('<div class="wf-toolbar-note">Files</div>', unsafe_allow_html=True)
    header_cols[8].markdown('<div class="wf-toolbar-note">Size</div>', unsafe_allow_html=True)
    header_cols[9].markdown('<div class="wf-toolbar-note">Actions</div>', unsafe_allow_html=True)


def _run_needs_source_backfill(run: Dict[str, object]) -> bool:
    return bool(
        str(run.get("evaluator_job_id") or "").strip()
        and str(run.get("project_id") or "").strip()
        and (
            not str(run.get("evaluator_git_ref_url") or "").strip()
            or not str(run.get("evaluator_git_commit_url") or "").strip()
            or not str(run.get("evaluator_source_url") or "").strip()
            or not str(run.get("evaluator_git_sha") or "").strip()
        )
    )


def _backfill_local_run_source_metadata(runs: List[Dict[str, object]]) -> Dict[str, int]:
    updated = 0
    skipped = 0
    failed = 0
    for run in runs:
        if not _run_needs_source_backfill(run):
            skipped += 1
            continue
        run_path = run.get("run_path")
        if not isinstance(run_path, Path):
            failed += 1
            continue
        project_id = str(run.get("project_id") or "").strip()
        environment = str(run.get("environment") or "default").strip() or "default"
        evaluator_job_id = str(run.get("evaluator_job_id") or "").strip()
        try:
            detail = _fetch_evaluator_job_detail(project_id, environment, evaluator_job_id)
        except Exception:
            failed += 1
            continue

        patch = {
            "evaluator": {
                "target": str(detail.get("source_label") or run.get("evaluator_target") or "").strip(),
                "git_sha": str(detail.get("git_sha") or run.get("evaluator_git_sha") or "").strip(),
                "git_ref_url": str(detail.get("git_ref_url") or run.get("evaluator_git_ref_url") or "").strip(),
                "git_commit_url": str(detail.get("git_commit_url") or run.get("evaluator_git_commit_url") or "").strip(),
                "source_url": str(detail.get("source_url") or run.get("evaluator_source_url") or "").strip(),
                "source_repo_label": str(detail.get("source_repo_label") or run.get("evaluator_source_repo_label") or "").strip(),
                "catalog_name": str(detail.get("catalog") or run.get("catalog_name") or "").strip(),
                "catalog_url": str(detail.get("catalog_url") or run.get("catalog_url") or "").strip(),
            }
        }
        try:
            upsert_run_metadata(run_path, patch, create_missing=False)
            updated += 1
        except Exception:
            failed += 1
    return {"updated": updated, "skipped": skipped, "failed": failed}


def _render_local_run_row(run: Dict[str, object], *, selected: bool) -> bool:
    row_key = _run_row_key(run)
    name_raw = str(run["name"])
    name = html.escape(name_raw)
    modified = html.escape(str(run["modified"]))
    user_label = html.escape(str(run.get("requested_by_label") or "—"))
    catalog_label = html.escape(str(run.get("catalog_label") or run.get("catalog_id") or "—"))
    catalog_url = html.escape(str(run.get("catalog_url") or ""))
    evaluator_job_id = str(run.get("evaluator_job_id") or "").strip()
    evaluator_report_url = str(run.get("evaluator_report_url") or "").strip()
    evaluator_target = str(run.get("evaluator_target") or "").strip()
    description = str(run.get("description") or "").strip()
    evaluator_title = html.escape(str(run.get("evaluator_title") or description or evaluator_job_id or "—"))
    source_label = str(run.get("evaluator_target") or evaluator_target or "—").strip()
    source_url = str(run.get("evaluator_git_ref_url") or run.get("evaluator_source_url") or "").strip()
    source_git_sha = str(run.get("evaluator_git_sha") or "").strip()
    source_commit_url = str(run.get("evaluator_git_commit_url") or "").strip()
    result_label = html.escape(
        f"✅ {int(run.get('passed_count') or 0)}  ❌ {int(run.get('failed_count') or 0)}  ⏹ {int(run.get('canceled_count') or 0)}"
    )
    task_type = str(run.get("task_type") or "").strip()
    task_status = str(run.get("task_status") or "").strip()
    meta_bits = [bit for bit in [task_type, task_status] if bit]
    flags = [
        ("Summary", bool(run["has_summary"])),
        ("Score", bool(run["has_score"])),
        ("Parquet", bool(run["has_parquet"])),
    ]
    compare_available = any(enabled for _, enabled in flags)
    title_class = "wf-run-title wf-run-text" + ("" if compare_available else " wf-run-title--muted")
    cell_class = "wf-run-cell wf-run-text" + ("" if compare_available else " wf-run-cell--muted")
    meta_class = "wf-meta-inline" + ("" if compare_available else " wf-meta-inline--muted")
    flag_wrap_class = "wf-run-flags" + ("" if compare_available else " wf-run-flags--muted")
    flag_html = "".join(
        f'<span class="wf-flag {"wf-flag--ok" if enabled else ""}">{label}</span>'
        for label, enabled in flags
    )
    if not compare_available:
        flag_html += '<div class="wf-unavailable-note">Unavailable for compare</div>'
    size_label = html.escape(str(run["size"]))
    checkbox_key = f"workflow_compare_pick::{row_key}"
    if not compare_available:
        st.session_state[checkbox_key] = False
    elif checkbox_key not in st.session_state:
        st.session_state[checkbox_key] = bool(selected)
    row_cols = st.columns([0.45, 2.35, 0.72, 1.45, 1.55, 1.0, 1.0, 1.22, 0.68, 1.55], gap="small")
    with row_cols[0]:
        checked = st.checkbox(
            "Select run",
            key=checkbox_key,
            label_visibility="collapsed",
            disabled=not compare_available,
        )
    with row_cols[1]:
        title_html = f'<div class="{title_class}"><a href="{_build_overview_url(row_key)}" target="_self">{name}</a></div>'
        if meta_bits:
            meta_html = html.escape(" · ".join(meta_bits[:3]))
            title_html += f'<div class="{meta_class}">{meta_html}</div>'
        st.markdown(title_html, unsafe_allow_html=True)
    with row_cols[2]:
        st.markdown(f'<div class="{cell_class}">{user_label}</div>', unsafe_allow_html=True)
    with row_cols[3]:
        if catalog_url and catalog_label != "—":
            st.markdown(
                f'<div class="{title_class}"><a href="{catalog_url}" target="_blank">{catalog_label}</a></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div class="{cell_class}">{catalog_label}</div>', unsafe_allow_html=True)
    with row_cols[4]:
        if evaluator_report_url and evaluator_job_id:
            evaluator_html = f'<div class="{title_class}"><a href="{html.escape(evaluator_report_url)}" target="_blank">{evaluator_title}</a></div>'
        else:
            evaluator_html = f'<div class="{cell_class}">{evaluator_title}</div>'
        source_ref_html = _format_source_ref_html(source_label, source_url, source_git_sha, source_commit_url)
        if source_ref_html and source_ref_html != "—":
            evaluator_html += f'<div class="{meta_class}">{source_ref_html}</div>'
        st.markdown(evaluator_html, unsafe_allow_html=True)
    with row_cols[5]:
        st.markdown(f'<div class="{cell_class}">{result_label}</div>', unsafe_allow_html=True)
    with row_cols[6]:
        st.markdown(f'<div class="{cell_class}">{modified}</div>', unsafe_allow_html=True)
    with row_cols[7]:
        st.markdown(f'<div class="wf-run-cell"><div class="{flag_wrap_class}">{flag_html}</div></div>', unsafe_allow_html=True)
    with row_cols[8]:
        st.markdown(f'<div class="{cell_class}">{size_label}</div>', unsafe_allow_html=True)
    with row_cols[9]:
        action_cols = st.columns([0.78, 0.82, 0.82], gap="small")
        with action_cols[0]:
            if st.button("ℹ", key=f"workflow_run_details::{row_key}", use_container_width=True, help="Show run details"):
                st.session_state["workflow_local_run_detail"] = row_key
        with action_cols[1]:
            if st.button("⬇", key=f"workflow_run_download::{row_key}", use_container_width=True, help="Prepare ZIP download"):
                st.session_state["workflow_local_run_download"] = row_key
        with action_cols[2]:
            if st.button("🗑", key=f"workflow_run_delete::{row_key}", use_container_width=True, help="Delete this local run"):
                st.session_state["workflow_local_run_delete"] = row_key
    return bool(checked)


def _render_local_run_details(run: Dict[str, object]) -> None:
    row_key = _run_row_key(run)
    metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
    owner_meta = metadata.get("owner") if isinstance(metadata.get("owner"), dict) else {}
    task_meta = metadata.get("task") if isinstance(metadata.get("task"), dict) else {}
    requester_meta = task_meta.get("requester") if isinstance(task_meta.get("requester"), dict) else {}
    request_meta = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
    evaluator_meta = metadata.get("evaluator") if isinstance(metadata.get("evaluator"), dict) else {}
    download_meta = metadata.get("download") if isinstance(metadata.get("download"), dict) else {}
    scenario_download_meta = metadata.get("scenario_download") if isinstance(metadata.get("scenario_download"), dict) else {}
    evaluation_meta = metadata.get("evaluation") if isinstance(metadata.get("evaluation"), dict) else {}
    parquet_meta = metadata.get("parquet") if isinstance(metadata.get("parquet"), dict) else {}
    project_id = str(request_meta.get("project_id") or "").strip()
    request_environment = str(request_meta.get("environment") or "default").strip() or "default"
    evaluator_job_id = str(evaluator_meta.get("job_id") or request_meta.get("job_id") or "").strip()
    evaluator_report_url = str(evaluator_meta.get("report_url") or "").strip()
    evaluator_target = str(evaluator_meta.get("target") or evaluator_meta.get("target_name") or request_meta.get("target_name") or "").strip()
    evaluator_detail = {}
    if project_id and evaluator_job_id:
        try:
            evaluator_detail = _fetch_evaluator_job_detail(project_id, request_environment, evaluator_job_id)
        except Exception:
            evaluator_detail = {}
    source_url = str(
        evaluator_meta.get("git_ref_url")
        or evaluator_meta.get("source_url")
        or evaluator_detail.get("source_url")
        or evaluator_detail.get("git_ref_url")
        or ""
    ).strip()
    source_commit_url = str(
        evaluator_meta.get("git_commit_url")
        or evaluator_detail.get("git_commit_url")
        or ""
    ).strip()
    catalog_url = str(evaluator_detail.get("catalog_url") or "").strip()
    source_label = str(evaluator_meta.get("target") or evaluator_detail.get("source_label") or evaluator_target or "").strip()
    source_git_sha = str(evaluator_meta.get("git_sha") or evaluator_detail.get("git_sha") or "").strip()
    source_ref_text = _format_source_ref_text(source_label or evaluator_target, source_git_sha)
    source_ref_html = _format_source_ref_html(source_label or evaluator_target, source_url, source_git_sha, source_commit_url)

    with st.container(border=True):
        title_cols = st.columns([3.4, 1.0])
        with title_cols[0]:
            st.markdown(f"### Local Run Details: `{run['name']}`")
        with title_cols[1]:
            if st.button("Clear", key=f"workflow_clear_run_details::{row_key}", use_container_width=True):
                st.session_state["workflow_local_run_detail"] = ""
                st.rerun()

        if not metadata:
            st.info("This run was created before metadata capture was added. Showing only filesystem information.")

        top_cols = st.columns(4)
        top_cols[0].metric("Updated", _metadata_text(run.get("modified")))
        top_cols[1].metric("Size", _metadata_text(run.get("size")))
        top_cols[2].metric("Task type", _metadata_text(task_meta.get("type") or metadata.get("source_mode")))
        top_cols[3].metric("Task status", _metadata_text(task_meta.get("status")))

        run_cols = st.columns(2)
        with run_cols[0]:
            st.caption("Run folder")
            st.code(str(run.get("path_display") or run.get("name") or ""), language=None)
        with run_cols[1]:
            st.caption("Available outputs")
            st.write(
                " | ".join(
                    label
                    for label, enabled in [
                        ("Summary.csv", bool(run.get("has_summary"))),
                        ("Score.csv", bool(run.get("has_score"))),
                        ("Parquet", bool(run.get("has_parquet"))),
                    ]
                    if enabled
                )
                or "—"
            )

        requested_by = str(
            owner_meta.get("id")
            or requester_meta.get("id")
            or task_meta.get("requested_by")
            or evaluator_meta.get("scheduled_by")
            or ""
        ).strip()
        requested_by_label = str(
            owner_meta.get("name")
            or owner_meta.get("email")
            or requester_meta.get("name")
            or requester_meta.get("email")
            or ""
        ).strip() or _run_user_label(requested_by, request_environment)

        task_cols = st.columns(4)
        task_cols[0].text_input("Requested by", value=requested_by_label, disabled=True, key=f"run_detail_user::{run['name']}")
        task_cols[1].text_input("Task ID", value=_metadata_text(task_meta.get("id")), disabled=True, key=f"run_detail_tid::{run['name']}")
        task_cols[2].text_input("Created", value=_format_metadata_time(task_meta.get("created_at") or metadata.get("created_at")), disabled=True, key=f"run_detail_created::{run['name']}")
        task_cols[3].text_input("Updated", value=_format_metadata_time(task_meta.get("updated_at") or metadata.get("updated_at")), disabled=True, key=f"run_detail_updated::{run['name']}")
        task_error = str(task_meta.get("error_message") or "").strip()
        if task_error:
            st.error(task_error)

        request_cols = st.columns(4)
        request_cols[0].text_input("Project", value=_metadata_text(request_meta.get("project_id")), disabled=True, key=f"run_detail_project::{run['name']}")
        request_cols[1].text_input("Environment", value=_metadata_text(request_environment), disabled=True, key=f"run_detail_env::{run['name']}")
        request_cols[2].text_input("Catalog ID", value=_metadata_text(evaluator_meta.get("catalog_id") or request_meta.get("catalog_id")), disabled=True, key=f"run_detail_catalog::{run['name']}")
        request_cols[3].text_input("Integration ID", value=_metadata_text(evaluator_meta.get("integration_id") or request_meta.get("integration_id")), disabled=True, key=f"run_detail_integration::{run['name']}")

        detail_cols = st.columns(3)
        detail_cols[0].text_input("Evaluator job ID", value=_metadata_text(evaluator_meta.get("job_id") or request_meta.get("job_id")), disabled=True, key=f"run_detail_job::{run['name']}")
        detail_cols[1].text_input("Source job ID", value=_metadata_text(evaluator_meta.get("source_job_id") or request_meta.get("source_job_id")), disabled=True, key=f"run_detail_source_job::{run['name']}")
        detail_cols[2].text_input("Target", value=_metadata_text(evaluator_meta.get("target") or request_meta.get("target_name")), disabled=True, key=f"run_detail_target::{run['name']}")

        st.text_input("Description", value=_metadata_text(request_meta.get("description") or evaluator_meta.get("description")), disabled=True, key=f"run_detail_desc::{run['name']}")

        if evaluator_job_id:
            action_cols = st.columns([1.15, 1.15, 1.15, 2.55])
            with action_cols[0]:
                if evaluator_report_url:
                    st.link_button("Open report", evaluator_report_url, use_container_width=True)
            with action_cols[1]:
                if source_url:
                    st.link_button("Open source", source_url, use_container_width=True)
            with action_cols[2]:
                if catalog_url:
                    st.link_button("Open catalog", catalog_url, use_container_width=True)
            with action_cols[3]:
                if st.button("Artifact retest", key=f"workflow_local_run_retest::{row_key}", type="primary", use_container_width=True):
                    st.session_state.pop(f"recent_eval_retest_suite_selection_{evaluator_job_id}", None)
                    st.session_state["workflow_local_run_retest"] = row_key
                    st.rerun()

            info_cols = st.columns([1.6, 2.4])
            info_cols[0].text_input(
                "Evaluator job",
                value=evaluator_job_id,
                disabled=True,
                key=f"run_detail_job_full::{run['name']}",
            )
            info_cols[1].text_input(
                "Source ref",
                value=_metadata_text(source_ref_text),
                disabled=True,
                key=f"run_detail_source_ref::{run['name']}",
            )
            if source_ref_html and source_ref_html != "—":
                st.markdown(
                    f'<div class="wf-linked-ref">GitHub: {source_ref_html}</div>',
                    unsafe_allow_html=True,
                )

        if evaluator_meta:
            eval_cols = st.columns(4)
            eval_cols[0].text_input("Evaluator status", value=_metadata_text(evaluator_meta.get("status")), disabled=True, key=f"run_detail_estatus::{run['name']}")
            eval_cols[1].text_input("Build status", value=_metadata_text(evaluator_meta.get("build_status")), disabled=True, key=f"run_detail_build::{run['name']}")
            eval_cols[2].text_input("Test status", value=_metadata_text(evaluator_meta.get("test_status")), disabled=True, key=f"run_detail_test::{run['name']}")
            eval_cols[3].text_input("Report URL", value=_metadata_text(evaluator_meta.get("report_url")), disabled=True, key=f"run_detail_report::{run['name']}")
            fail_message = str(evaluator_meta.get("fail_message") or "").strip()
            if fail_message:
                st.warning(fail_message)
            case_totals = evaluator_meta.get("case_totals") if isinstance(evaluator_meta.get("case_totals"), dict) else {}
            if case_totals:
                case_cols = st.columns(4)
                case_cols[0].metric("Cases total", str(case_totals.get("total", 0)))
                case_cols[1].metric("Cases success", str(case_totals.get("success", 0)))
                case_cols[2].metric("Cases failed", str(case_totals.get("failed", 0)))
                case_cols[3].metric("Cases canceled", str(case_totals.get("canceled", 0)))

        if download_meta or scenario_download_meta:
            active_download_meta = download_meta or scenario_download_meta
            download_cols = st.columns(4)
            download_cols[0].text_input("Download mode", value=_metadata_text(active_download_meta.get("mode") or metadata.get("source_mode")), disabled=True, key=f"run_detail_dl_mode::{run['name']}")
            download_cols[1].text_input("Download type", value=_metadata_text(download_meta.get("download_type") or request_meta.get("download_type")), disabled=True, key=f"run_detail_dl_type::{run['name']}")
            download_cols[2].text_input("Phase", value=_metadata_text(download_meta.get("phase") or request_meta.get("phase")), disabled=True, key=f"run_detail_phase::{run['name']}")
            download_cols[3].text_input("Skip large files", value="Yes" if bool(download_meta.get("skip_large_file") or request_meta.get("skip_large_file")) else "No", disabled=True, key=f"run_detail_skip::{run['name']}")

            count_cols = st.columns(3)
            count_cols[0].metric("Download total", str(active_download_meta.get("total", 0)))
            count_cols[1].metric("Download success", str(active_download_meta.get("success", 0)))
            count_cols[2].metric("Download failed", str(active_download_meta.get("failed", 0)))

        if evaluation_meta:
            eval_run_cols = st.columns(4)
            eval_run_cols[0].text_input("Eval enabled", value="Yes" if bool(evaluation_meta.get("enabled") or request_meta.get("run_eval")) else "No", disabled=True, key=f"run_detail_eval_enabled::{run['name']}")
            eval_run_cols[1].text_input("Recursive", value="Yes" if bool(evaluation_meta.get("recursive") or request_meta.get("eval_recursive")) else "No", disabled=True, key=f"run_detail_eval_recursive::{run['name']}")
            eval_run_cols[2].text_input("Summary rows", value=str(evaluation_meta.get("summary_rows", "—")), disabled=True, key=f"run_detail_summary_rows::{run['name']}")
            eval_run_cols[3].text_input("Score rows", value=str(evaluation_meta.get("score_rows", "—")), disabled=True, key=f"run_detail_score_rows::{run['name']}")

        if parquet_meta:
            st.text_input("Parquet path", value=_metadata_text(parquet_meta.get("path")), disabled=True, key=f"run_detail_parquet::{run['name']}")

        suites = evaluator_meta.get("suites") if isinstance(evaluator_meta.get("suites"), list) else []
        failed_cases = evaluator_meta.get("failed_cases") if isinstance(evaluator_meta.get("failed_cases"), list) else []
        if suites:
            with st.expander("Evaluator suites", expanded=False):
                st.dataframe(suites, width="stretch", hide_index=True)
        if failed_cases:
            with st.expander("Failed cases", expanded=False):
                st.dataframe(failed_cases, width="stretch", hide_index=True)

        with st.expander("Raw run metadata", expanded=False):
            st.json(metadata or {})

        selected_retest_run = str(st.session_state.get("workflow_local_run_retest") or "").strip()
        if selected_retest_run == row_key and evaluator_job_id:
            dialog_job = {
                "job_id": evaluator_job_id,
                "title": str(evaluator_detail.get("title") or run.get("description") or run["name"]),
            }
            if callable(getattr(st, "dialog", None)):
                try:
                    @st.dialog(f"Artifact retest · {dialog_job['title']}", width="large")
                    def _workflow_local_run_retest_dialog() -> None:
                        _render_recent_evaluator_job_retest_dialog(
                            project_id,
                            request_environment,
                            dialog_job,
                            output_path_default="",
                            phase_default=str(request_meta.get("phase") or "perception.object_recognition.objects"),
                        )

                    _workflow_local_run_retest_dialog()
                finally:
                    if st.session_state.get("workflow_local_run_retest") == row_key:
                        st.session_state.pop("workflow_local_run_retest", None)
            else:
                st.markdown("---")
                fallback_cols = st.columns([4.2, 1.0])
                with fallback_cols[0]:
                    st.subheader(f"Artifact retest · {dialog_job['title']}")
                with fallback_cols[1]:
                    if st.button("Close", key=f"workflow_local_run_retest_close::{row_key}", use_container_width=True):
                        st.session_state.pop("workflow_local_run_retest", None)
                        st.rerun()
                _render_recent_evaluator_job_retest_dialog(
                    project_id,
                    request_environment,
                    dialog_job,
                    output_path_default="",
                    phase_default=str(request_meta.get("phase") or "perception.object_recognition.objects"),
                )


def _render_local_runs_section() -> None:
    section_header("Local Runs", "")
    runs = _load_local_runs()
    if not runs:
        st.markdown('<div class="wf-empty">No finished runs were found on this server yet.</div>', unsafe_allow_html=True)
        return
    missing_source_runs = sum(1 for run in runs if _run_needs_source_backfill(run))
    local_runs_toolbar_cols = st.columns([4.2, 1.2])
    with local_runs_toolbar_cols[0]:
        if missing_source_runs:
            st.caption(f"{missing_source_runs} run(s) are missing stored GitHub metadata.")
    with local_runs_toolbar_cols[1]:
        if missing_source_runs and st.button(
            "Backfill GitHub",
            key="workflow_backfill_local_run_source_meta",
            use_container_width=True,
        ):
            with st.spinner("Backfilling missing GitHub metadata for local runs..."):
                result = _backfill_local_run_source_metadata(runs)
            _load_local_runs.clear()
            if result["failed"]:
                st.warning(
                    f"Backfill updated {result['updated']} run(s), skipped {result['skipped']} run(s), failed on {result['failed']} run(s)."
                )
            else:
                st.success(
                    f"Backfill updated {result['updated']} run(s); {result['skipped']} run(s) already had metadata."
                )
            st.rerun()

    current_user_id = str(get_task_list_current_user() or "").strip()
    user_options = ["All users"]
    if current_user_id:
        user_options.append("My runs")
    unique_users = []
    seen_users = set()
    user_option_subject_map = {"All users": "", "My runs": current_user_id, "(Auto)": "__auto__"}
    for row in runs:
        subject_id = str(row.get("requested_by") or "").strip()
        label = str(row.get("requested_by_label") or "").strip()
        option = label or "(Auto)"
        if not subject_id:
            if "(Auto)" not in user_options:
                user_options.append("(Auto)")
            continue
        deduped_option = option
        suffix = 2
        while deduped_option in seen_users and user_option_subject_map.get(deduped_option) != subject_id:
            deduped_option = f"{option} [{suffix}]"
            suffix += 1
        if deduped_option not in seen_users:
            unique_users.append(deduped_option)
            seen_users.add(deduped_option)
            user_option_subject_map[deduped_option] = subject_id
    user_options.extend(unique_users)

    catalog_options = ["All catalogs"]
    catalog_option_id_map = {"All catalogs": ""}
    unique_catalogs = []
    seen_catalogs = set()
    for row in runs:
        catalog_id = str(row.get("catalog_id") or "").strip()
        catalog_label = str(row.get("catalog_label") or row.get("catalog_name") or catalog_id or "—").strip()
        if not catalog_id:
            continue
        option = catalog_label or catalog_id
        deduped_option = option
        suffix = 2
        while deduped_option in seen_catalogs and catalog_option_id_map.get(deduped_option) != catalog_id:
            deduped_option = f"{option} [{suffix}]"
            suffix += 1
        if deduped_option not in seen_catalogs:
            unique_catalogs.append(deduped_option)
            seen_catalogs.add(deduped_option)
            catalog_option_id_map[deduped_option] = catalog_id
    catalog_options.extend(sorted(unique_catalogs, key=str.lower))

    current_user_option = str(st.session_state.get("workflow_runs_user_filter", "All users"))
    if current_user_option not in user_options:
        current_user_option = "All users"
        st.session_state["workflow_runs_user_filter"] = current_user_option
    current_catalog_option = str(st.session_state.get("workflow_runs_catalog_filter", "All catalogs"))
    if current_catalog_option not in catalog_options:
        current_catalog_option = "All catalogs"
        st.session_state["workflow_runs_catalog_filter"] = current_catalog_option
    branch_options = ["All branches"]
    unique_branches = sorted(
        {
            str(row.get("branch_label") or row.get("evaluator_target") or "").strip()
            for row in runs
            if str(row.get("branch_label") or row.get("evaluator_target") or "").strip()
        },
        key=str.lower,
    )
    branch_options.extend(unique_branches)
    current_branch_option = str(st.session_state.get("workflow_runs_branch_filter", "All branches"))
    if current_branch_option not in branch_options:
        current_branch_option = "All branches"
        st.session_state["workflow_runs_branch_filter"] = current_branch_option

    st.markdown('<div class="wf-filter-strip">', unsafe_allow_html=True)
    control_cols = st.columns([1.7, 1.15, 1.1, 0.95, 0.95])
    with control_cols[0]:
        st.markdown('<div class="wf-toolbar-note">Search</div>', unsafe_allow_html=True)
        run_search_input = st.text_input(
            "Search runs",
            value=st.session_state.get("workflow_runs_search", ""),
            key="workflow_runs_search",
            label_visibility="collapsed",
            placeholder="Filter by name, description, job id, catalog, user",
        )
    with control_cols[1]:
        st.markdown('<div class="wf-toolbar-note">Catalog</div>', unsafe_allow_html=True)
        catalog_filter_input = st.selectbox(
            "Catalog",
            options=catalog_options,
            index=catalog_options.index(current_catalog_option),
            key="workflow_runs_catalog_filter",
            label_visibility="collapsed",
        )
    with control_cols[2]:
        st.markdown('<div class="wf-toolbar-note">Branch</div>', unsafe_allow_html=True)
        branch_filter_input = st.selectbox(
            "Branch",
            options=branch_options,
            index=branch_options.index(current_branch_option),
            key="workflow_runs_branch_filter",
            label_visibility="collapsed",
        )
    with control_cols[3]:
        st.markdown('<div class="wf-toolbar-note">User</div>', unsafe_allow_html=True)
        user_filter_input = st.selectbox(
            "User",
            options=user_options,
            index=user_options.index(current_user_option),
            key="workflow_runs_user_filter",
            label_visibility="collapsed",
        )
    with control_cols[4]:
        st.markdown('<div class="wf-toolbar-note">Rows</div>', unsafe_allow_html=True)
        page_size_input = int(
            st.selectbox(
                "Rows",
                options=[10, 20, 50, 100],
                index=[10, 20, 50, 100].index(int(st.session_state.get("workflow_runs_page_size", 10) or 10)),
                key="workflow_runs_page_size",
                label_visibility="collapsed",
            )
        )

    second_control_cols = st.columns([0.92, 0.92, 0.6, 0.6, 2.4])
    with second_control_cols[0]:
        st.markdown('<div class="wf-toolbar-note">From</div>', unsafe_allow_html=True)
        date_from_input = st.date_input(
            "From",
            value=st.session_state.get("workflow_runs_date_from", None),
            key="workflow_runs_date_from",
            label_visibility="collapsed",
            help="Run modified-date lower bound in JST.",
        )
    with second_control_cols[1]:
        st.markdown('<div class="wf-toolbar-note">To</div>', unsafe_allow_html=True)
        date_to_input = st.date_input(
            "To",
            value=st.session_state.get("workflow_runs_date_to", None),
            key="workflow_runs_date_to",
            label_visibility="collapsed",
            help="Run modified-date upper bound in JST.",
        )
    with second_control_cols[2]:
        st.markdown('<div class="wf-toolbar-note">Summary</div>', unsafe_allow_html=True)
        require_summary_input = st.toggle(
            "Summary only",
            value=bool(st.session_state.get("workflow_runs_summary_filter", False)),
            key="workflow_runs_summary_filter",
            label_visibility="collapsed",
        )
    with second_control_cols[3]:
        st.markdown('<div class="wf-toolbar-note">Parquet</div>', unsafe_allow_html=True)
        require_parquet_input = st.toggle(
            "Parquet only",
            value=bool(st.session_state.get("workflow_runs_parquet_filter", False)),
            key="workflow_runs_parquet_filter",
            label_visibility="collapsed",
        )
    with second_control_cols[4]:
        st.markdown(
            '<div class="wf-pager-summary">Pick a catalog, branch, or user directly, or narrow with text and dates.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    current_filter_signature = (
        str(run_search_input or ""),
        str(catalog_filter_input or "All catalogs"),
        str(branch_filter_input or "All branches"),
        str(user_filter_input or "All users"),
        date_from_input,
        date_to_input,
        bool(require_summary_input),
        bool(require_parquet_input),
        int(page_size_input),
    )
    previous_filter_signature = st.session_state.get("workflow_runs_filter_signature")
    if previous_filter_signature is None:
        st.session_state["workflow_runs_filter_signature"] = current_filter_signature
    elif previous_filter_signature != current_filter_signature:
        st.session_state["workflow_runs_filter_signature"] = current_filter_signature
        st.session_state["workflow_runs_page"] = 1

    run_search = str(run_search_input).strip().lower()
    selected_catalog_filter = str(catalog_filter_input).strip()
    selected_branch_filter = str(branch_filter_input).strip()
    selected_user_filter = str(user_filter_input).strip()
    selected_date_from = date_from_input
    selected_date_to = date_to_input
    require_summary = bool(require_summary_input)
    require_parquet = bool(require_parquet_input)
    page_size = int(page_size_input)

    if selected_date_from and selected_date_to and selected_date_from > selected_date_to:
        st.warning("`From` date must be earlier than or equal to `To` date.")
        return

    filtered = runs
    if run_search:
        filtered = [row for row in filtered if run_search in str(row.get("search_blob") or row["name"]).lower()]
    if selected_catalog_filter not in ("", "All catalogs"):
        selected_catalog_id = str(catalog_option_id_map.get(selected_catalog_filter) or "").strip()
        filtered = [row for row in filtered if str(row.get("catalog_id") or "").strip() == selected_catalog_id]
    if selected_branch_filter not in ("", "All branches"):
        filtered = [
            row for row in filtered
            if str(row.get("branch_label") or row.get("evaluator_target") or "").strip() == selected_branch_filter
        ]
    if selected_user_filter == "My runs" and current_user_id:
        filtered = [row for row in filtered if str(row.get("requested_by") or "").strip() == current_user_id]
    elif selected_user_filter == "(Auto)":
        filtered = [row for row in filtered if not str(row.get("requested_by") or "").strip()]
    elif selected_user_filter not in ("", "All users", "My runs"):
        selected_subject_id = str(user_option_subject_map.get(selected_user_filter) or "").strip()
        filtered = [row for row in filtered if str(row.get("requested_by") or "").strip() == selected_subject_id]
    if selected_date_from:
        filtered = [row for row in filtered if row.get("mtime_date") and row["mtime_date"] >= selected_date_from]
    if selected_date_to:
        filtered = [row for row in filtered if row.get("mtime_date") and row["mtime_date"] <= selected_date_to]
    if require_summary:
        filtered = [row for row in filtered if bool(row["has_summary"])]
    if require_parquet:
        filtered = [row for row in filtered if bool(row["has_parquet"])]

    compare_ready = [
        _run_row_key(row)
        for row in filtered
        if bool(row["has_summary"]) or bool(row["has_score"]) or bool(row["has_parquet"])
    ]
    runs_by_key = {_run_row_key(row): row for row in runs}
    runs_by_name: Dict[str, List[Dict[str, object]]] = {}
    for row in runs:
        runs_by_name.setdefault(str(row["name"]), []).append(row)
    display_by_key = {_run_row_key(row): str(row["name"]) for row in runs}
    if "workflow_compare_runs" not in st.session_state:
        st.session_state["workflow_compare_runs"] = compare_ready[:1]

    compare_selected = []
    for key in st.session_state.get("workflow_compare_runs", []):
        normalized = _normalize_compare_run_key(str(key), runs_by_key, runs_by_name)
        if normalized in compare_ready and normalized not in compare_selected:
            compare_selected.append(normalized)
    st.session_state["workflow_compare_runs"] = compare_selected

    if not filtered:
        st.markdown('<div class="wf-empty">No local runs matched the current filters.</div>', unsafe_allow_html=True)
        return

    page_key = "workflow_runs_page"
    current_page = max(1, int(st.session_state.get(page_key, 1)))
    page_count = max(1, (len(filtered) + page_size - 1) // page_size)
    if current_page > page_count:
        current_page = page_count
        st.session_state[page_key] = current_page
    start_idx = (current_page - 1) * page_size
    visible_runs = filtered[start_idx:start_idx + page_size]
    visible_keys = {_run_row_key(run) for run in visible_runs}

    visible_end = min(len(filtered), start_idx + len(visible_runs))
    st.markdown('<div class="wf-pager-strip">', unsafe_allow_html=True)
    pager_cols = st.columns([0.65, 1.0, 0.65, 3.2])
    with pager_cols[0]:
        if st.button("‹", key="workflow_runs_page_prev", use_container_width=True, disabled=current_page <= 1):
            current_page -= 1
            st.session_state[page_key] = current_page
            st.rerun()
    with pager_cols[1]:
        selected_page = st.selectbox(
            "Page",
            options=list(range(1, page_count + 1)),
            index=max(0, current_page - 1),
            label_visibility="collapsed",
        )
        if selected_page != current_page:
            st.session_state[page_key] = int(selected_page)
            current_page = int(selected_page)
            start_idx = (current_page - 1) * page_size
            visible_runs = filtered[start_idx:start_idx + page_size]
            visible_keys = {_run_row_key(run) for run in visible_runs}
    with pager_cols[2]:
        if st.button("›", key="workflow_runs_page_next", use_container_width=True, disabled=current_page >= page_count):
            current_page += 1
            st.session_state[page_key] = current_page
            st.rerun()
    with pager_cols[3]:
        st.markdown(
            f'<div class="wf-pager-summary"><strong>{start_idx + 1}</strong>–<strong>{visible_end}</strong> of <strong>{len(filtered)}</strong> runs · {page_size} per page</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    _render_local_runs_header()
    next_selected = [key for key in st.session_state.get("workflow_compare_runs", []) if key not in visible_keys]
    for run in visible_runs:
        row_key = _run_row_key(run)
        if _render_local_run_row(run, selected=row_key in st.session_state.get("workflow_compare_runs", [])) and row_key in compare_ready:
            next_selected.append(row_key)
    st.session_state["workflow_compare_runs"] = [key for key in compare_ready if key in next_selected]

    st.markdown('<div class="wf-compare-bar">', unsafe_allow_html=True)
    st.markdown('<p class="wf-compare-title">Compare</p>', unsafe_allow_html=True)
    compare_cols = st.columns([3.4, 1.0])
    with compare_cols[0]:
        st.markdown('<div class="wf-toolbar-note">Selected runs</div>', unsafe_allow_html=True)
        selected_runs = list(st.session_state.get("workflow_compare_runs", []))
        if selected_runs:
            st.caption(" | ".join(display_by_key.get(key, key) for key in selected_runs))
    with compare_cols[1]:
        st.markdown('<div class="wf-toolbar-note">Action</div>', unsafe_allow_html=True)
        if len(selected_runs) >= 2:
            st.link_button("Compare", _build_overview_url(selected_runs[0], selected_runs[1:]), use_container_width=True)
        elif len(selected_runs) == 1:
            st.link_button("Open", _build_overview_url(selected_runs[0]), use_container_width=True)
        else:
            st.button("Open", disabled=True, use_container_width=True, key="workflow_compare_run_disabled")
    st.markdown("</div>", unsafe_allow_html=True)

    download_run_key = str(st.session_state.get("workflow_local_run_download") or "").strip()
    if download_run_key:
        download_run_name = display_by_key.get(download_run_key, download_run_key)
        if callable(getattr(st, "dialog", None)):
            @st.dialog(f"Download artifacts · {download_run_name}", width="large")
            def _workflow_local_run_download_dialog() -> None:
                _render_local_run_download_dialog(download_run_key)
                if st.button("Close", key=f"workflow_local_run_download_close::{download_run_key}", use_container_width=True):
                    st.session_state.pop("workflow_local_run_download", None)
                    st.rerun()

            _workflow_local_run_download_dialog()
        else:
            st.markdown("---")
            st.subheader(f"Download artifacts · {download_run_name}")
            _render_local_run_download_dialog(download_run_key)

    delete_run_key = str(st.session_state.get("workflow_local_run_delete") or "").strip()
    if delete_run_key:
        delete_run_name = display_by_key.get(delete_run_key, delete_run_key)
        if callable(getattr(st, "dialog", None)):
            @st.dialog(f"Delete local run · {delete_run_name}", width="large")
            def _workflow_local_run_delete_dialog() -> None:
                _render_local_run_delete_dialog(delete_run_key, confirm_label=delete_run_name)
                if st.button("Cancel", key=f"workflow_local_run_delete_close::{delete_run_key}", use_container_width=True):
                    st.session_state.pop("workflow_local_run_delete", None)
                    st.rerun()

            _workflow_local_run_delete_dialog()
        else:
            st.markdown("---")
            st.subheader(f"Delete local run · {delete_run_name}")
            _render_local_run_delete_dialog(delete_run_key, confirm_label=delete_run_name)

    detail_run_key = str(st.session_state.get("workflow_local_run_detail") or "").strip()
    if detail_run_key:
        detail_run = runs_by_key.get(_normalize_compare_run_key(detail_run_key, runs_by_key, runs_by_name))
        if detail_run is not None:
            _render_local_run_details(detail_run)


def _render_current_tasks_section() -> None:
    section_header("Current Tasks", "")
    if not is_task_queue_enabled():
        st.info("Task queue not enabled. Set `USE_TASK_QUEUE=true` to track background tasks.")
        return

    current_user = get_task_list_current_user()
    if "workflow_task_history_range" not in st.session_state:
        st.session_state["workflow_task_history_range"] = "7 days"
    if "workflow_task_history_page_size" not in st.session_state:
        st.session_state["workflow_task_history_page_size"] = 20
    if "workflow_task_history_page" not in st.session_state:
        st.session_state["workflow_task_history_page"] = 1

    control_cols = st.columns([1.3, 1.0, 1.0, 2.7])
    with control_cols[0]:
        selected_range = st.selectbox(
            "History range",
            options=list(_TASK_HISTORY_RANGE_OPTIONS.keys()),
            key="workflow_task_history_range",
        )
    with control_cols[1]:
        page_size = int(
            st.selectbox(
                "Rows",
                options=[20, 50, 100],
                key="workflow_task_history_page_size",
            )
        )
    since_days = _TASK_HISTORY_RANGE_OPTIONS.get(selected_range, _TASK_LIST_SINCE_DAYS)
    total_tasks = count_recent_tasks(session_id=current_user, since_days=since_days)
    page_count = max(1, (total_tasks + page_size - 1) // page_size) if total_tasks else 1
    current_page = min(max(1, int(st.session_state.get("workflow_task_history_page", 1))), page_count)
    st.session_state["workflow_task_history_page"] = current_page
    with control_cols[2]:
        selected_page = st.selectbox(
            "Page",
            options=list(range(1, page_count + 1)),
            index=current_page - 1,
            key="workflow_task_history_page_select",
        )
        if int(selected_page) != current_page:
            current_page = int(selected_page)
            st.session_state["workflow_task_history_page"] = current_page
    with control_cols[3]:
        label = selected_range if since_days is not None else "all time"
        st.caption(f"Showing **{total_tasks}** tasks across **{page_count}** page(s) for **{label}**.")

    offset = (current_page - 1) * page_size
    use_fragment = getattr(st, "fragment", None) is not None
    if use_fragment:
        try:

            @st.fragment(run_every=timedelta(seconds=3))
            def _task_list_poll():
                current_tasks = list_recent_tasks(
                    limit=page_size,
                    offset=offset,
                    session_id=current_user,
                    since_days=since_days,
                )
                render_task_list(current_tasks, current_user, on_delete=_close_workflow_dialogs)

            _task_list_poll()
            return
        except (TypeError, AttributeError):
            use_fragment = False

    tasks = list_recent_tasks(
        limit=page_size,
        offset=offset,
        session_id=current_user,
        since_days=since_days,
    )
    has_active = render_task_list(tasks, current_user, on_delete=_close_workflow_dialogs)
    if st.button("Refresh tasks", key="workflow_refresh_tasks"):
        st.rerun()
    if has_active:
        st.caption("Active jobs are shown live when possible. Use refresh if this browser does not support fragments.")


def _get_start_workflow_defaults() -> Dict[str, object]:
    default_target = get_config_value("target_name", "beta/v4.3.2")
    return {
        "project_id": get_config_value("eval_project_id", "x2_dev"),
        "environment": get_config_value("environment", ""),
        "output_path_default": _make_default_output_path(default_target),
        "download_type_default": get_config_value("eval_download_type", "Archives (ZIP)"),
        "phase_default": get_config_value(
            "eval_phase",
            _DEFAULT_PERCEPTION_PHASE,
        ),
        "skip_large_file_default": True,
        "large_file_mb_default": 50.0,
        "keep_zip_files_default": False,
    }


def _render_start_workflow_form(
    catalog_presets: List[Dict[str, str]],
    catalogs_path: Optional[str],
    catalog_load_error: Optional[str],
) -> Dict[str, object]:
    if catalog_load_error:
        st.warning(f"Could not read catalog presets: {catalog_load_error}")

    catalog_names = [item["display_name"] for item in catalog_presets]
    default_project = get_config_value("eval_project_id", "x2_dev")
    default_target = get_config_value("target_name", "beta/v4.3.2")
    default_download_type = get_config_value("eval_download_type", "Archives (ZIP)")
    default_phase = get_config_value(
        "eval_phase",
        _DEFAULT_PERCEPTION_PHASE,
    )
    default_poll_interval = int(get_config_value("poll_interval", 60))
    try:
        default_max_wait_hours = max(0, int(get_config_value("max_wait_hours", _DEFAULT_MAX_WAIT_HOURS)))
    except (TypeError, ValueError):
        default_max_wait_hours = _DEFAULT_MAX_WAIT_HOURS
    default_environment = get_config_value("environment", "")
    default_output = _make_default_output_path(default_target)
    default_skip_large_file = True

    if "workflow_server_catalogs" not in st.session_state:
        st.session_state["workflow_server_catalogs"] = []
    if "workflow_server_catalog_error" not in st.session_state:
        st.session_state["workflow_server_catalog_error"] = ""
    if "workflow_selected_server_catalog_id" not in st.session_state:
        st.session_state["workflow_selected_server_catalog_id"] = ""
    if "workflow_catalog_id" not in st.session_state:
        st.session_state["workflow_catalog_id"] = ""
    if "workflow_integration_id" not in st.session_state:
        st.session_state["workflow_integration_id"] = ""
    if "workflow_catalog_resolution_error" not in st.session_state:
        st.session_state["workflow_catalog_resolution_error"] = ""
    if "workflow_last_catalog_selection" not in st.session_state:
        st.session_state["workflow_last_catalog_selection"] = ""

    server_catalogs = st.session_state.get("workflow_server_catalogs", []) or []
    server_catalog_labels = [
        f"{item['display_name']} ({item['catalog_id']})" for item in server_catalogs
    ]
    catalog_options = [""] + catalog_names + [
        label for label in server_catalog_labels if label not in catalog_names
    ]
    preset_by_label = {item["display_name"]: item for item in catalog_presets}
    server_by_label = {
        f"{item['display_name']} ({item['catalog_id']})": item for item in server_catalogs
    }

    release_mode = st.checkbox(
        "Release data workflow: schedule Performance Test + Devops Test",
        value=bool(st.session_state.get("workflow_release_mode", False)),
        key="workflow_release_mode",
        help="Queues the two standard release evaluator jobs, processes both as normal app runs, then generates a release specsheet with trend data.",
    )
    previous_release_mode = bool(st.session_state.get("workflow_previous_release_mode", release_mode))
    if previous_release_mode != release_mode:
        if release_mode:
            st.session_state["workflow_run_eval"] = False
            st.session_state["workflow_max_wait_hours"] = _DEFAULT_MAX_WAIT_HOURS
    st.session_state["workflow_previous_release_mode"] = release_mode

    top_cols = st.columns([1.0, 1.8] if release_mode else [1.0, 1.9, 1.2])
    with top_cols[0]:
        st.markdown('<div class="wf-toolbar-note">Project</div>', unsafe_allow_html=True)
        project_id = st.text_input(
            "Project ID",
            value=default_project,
            key="workflow_project_id",
            label_visibility="collapsed",
        ).strip()
    if release_mode:
        selected_catalog_name = ""
        fetch_catalogs_clicked = False
        with top_cols[1]:
            st.markdown('<div class="wf-toolbar-note">Release output folder</div>', unsafe_allow_html=True)
            output_path = st.text_input(
                "Release output folder",
                value=default_output,
                key="workflow_output_path",
                label_visibility="collapsed",
                placeholder=_make_default_output_path(default_target),
                help="Folder under data/. Existing release folders can hydrate metadata and recorded job IDs.",
            ).strip()
        existing_release_context = _load_existing_release_context(output_path)
        context_output_key = "workflow_release_context_output_path"
        if st.session_state.get(context_output_key) != output_path:
            loaded_target = str(existing_release_context.get("target_name") or "").strip()
            if loaded_target:
                st.session_state["workflow_target_name"] = loaded_target
            st.session_state[context_output_key] = output_path
    else:
        existing_release_context = {}
        with top_cols[1]:
            st.markdown('<div class="wf-toolbar-note">Catalog</div>', unsafe_allow_html=True)
            catalog_picker_cols = st.columns([4.2, 1.1], gap="small")
            with catalog_picker_cols[0]:
                selected_catalog_name = st.selectbox(
                    "Catalog",
                    options=catalog_options if catalog_options else [""],
                    index=catalog_options.index(st.session_state.get("workflow_catalog_name", "")) if st.session_state.get("workflow_catalog_name", "") in catalog_options else 0,
                    key="workflow_catalog_name",
                    label_visibility="collapsed",
                    format_func=lambda value: value or "Choose a catalog",
                )
            with catalog_picker_cols[1]:
                fetch_catalogs_clicked = st.button(
                    "Fetch",
                    key="workflow_fetch_server_catalogs",
                    use_container_width=True,
                )
                if fetch_catalogs_clicked:
                    try:
                        current_environment = str(st.session_state.get("workflow_environment", default_environment) or "")
                        st.session_state["workflow_server_catalogs"] = _fetch_server_catalogs(project_id, current_environment)
                        st.session_state["workflow_server_catalog_error"] = ""
                    except Exception as exc:
                        st.session_state["workflow_server_catalogs"] = []
                        st.session_state["workflow_server_catalog_error"] = str(exc)
    selected_catalog = preset_by_label.get(selected_catalog_name)
    selected_server_catalog = server_by_label.get(selected_catalog_name)
    if "workflow_last_catalog_preset" not in st.session_state:
        st.session_state["workflow_last_catalog_preset"] = ""
    if st.session_state["workflow_last_catalog_preset"] != selected_catalog_name and selected_catalog:
        st.session_state["workflow_catalog_id"] = str(selected_catalog.get("catalog_id") or "")
        st.session_state["workflow_integration_id"] = str(selected_catalog.get("integration_id") or "")
        st.session_state["workflow_selected_server_catalog_id"] = ""
        st.session_state["workflow_catalog_resolution_error"] = ""
        if st.session_state["workflow_catalog_id"] and not st.session_state["workflow_integration_id"]:
            current_environment = str(st.session_state.get("workflow_environment", default_environment) or "")
            try:
                st.session_state["workflow_integration_id"] = _resolve_integration_id_for_catalog(
                    project_id,
                    current_environment,
                    st.session_state["workflow_catalog_id"],
                )
                st.session_state["workflow_catalog_resolution_error"] = ""
            except Exception as exc:
                st.session_state["workflow_catalog_resolution_error"] = str(exc)
        st.session_state["workflow_last_catalog_preset"] = selected_catalog_name
    elif selected_server_catalog:
        st.session_state["workflow_catalog_id"] = str(selected_server_catalog.get("catalog_id") or "")
        st.session_state["workflow_selected_server_catalog_id"] = str(selected_server_catalog.get("catalog_id") or "")
        current_environment = str(st.session_state.get("workflow_environment", default_environment) or "")
        if st.session_state["workflow_last_catalog_selection"] != selected_catalog_name:
            try:
                st.session_state["workflow_integration_id"] = _resolve_integration_id_for_catalog(
                    project_id,
                    current_environment,
                    st.session_state["workflow_catalog_id"],
                )
                st.session_state["workflow_catalog_resolution_error"] = ""
            except Exception as exc:
                st.session_state["workflow_integration_id"] = ""
                st.session_state["workflow_catalog_resolution_error"] = str(exc)
            st.session_state["workflow_last_catalog_selection"] = selected_catalog_name
    elif st.session_state["workflow_last_catalog_selection"] != selected_catalog_name:
        st.session_state["workflow_catalog_resolution_error"] = ""
        st.session_state["workflow_last_catalog_selection"] = selected_catalog_name
    release_detail_cols = st.columns([1.2, 1.75]) if release_mode else []
    target_col = release_detail_cols[0] if release_mode else top_cols[2]
    with target_col:
        st.markdown('<div class="wf-toolbar-note">Branch or tag</div>', unsafe_allow_html=True)
        target_name = st.text_input(
            "Branch or Tag",
            value=default_target,
            key="workflow_target_name",
            label_visibility="collapsed",
            placeholder="beta/v4.3.2",
        ).strip()

    catalog_id = str(st.session_state.get("workflow_catalog_id") or "").strip()
    integration_id = str(st.session_state.get("workflow_integration_id") or "").strip()
    catalog_auto_tlr_mode = (
        not release_mode
        and _looks_like_tlr_catalog(
            selected_catalog_name,
            catalog_id,
            selected_catalog.get("display_name") if selected_catalog else "",
            selected_catalog.get("description") if selected_catalog else "",
            selected_server_catalog.get("display_name") if selected_server_catalog else "",
            selected_server_catalog.get("description") if selected_server_catalog else "",
        )
    )
    workflow_kind = _WORKFLOW_KIND_TLR if (
        catalog_auto_tlr_mode or bool(st.session_state.get("workflow_tlr_mode_manual", False))
    ) else _WORKFLOW_KIND_PERCEPTION

    if not release_mode and st.session_state.get("workflow_server_catalog_error"):
        st.warning(f"Could not fetch catalogs: {st.session_state['workflow_server_catalog_error']}")
    catalog_id = str(st.session_state.get("workflow_catalog_id") or "").strip()

    picker_cols = st.columns([1.2, 1.75] if release_mode or workflow_kind == _WORKFLOW_KIND_TLR else [1.2, 1.2, 1.75])
    if not release_mode:
        with picker_cols[0]:
            st.markdown('<div class="wf-toolbar-note">Output folder</div>', unsafe_allow_html=True)
            output_path = st.text_input(
                "Output folder",
                value=default_output,
                key="workflow_output_path",
                label_visibility="collapsed",
                placeholder=_make_default_output_path(target_name),
                help="Output folder under the data directory.",
            ).strip()
    if release_mode:
        phase = _DEFAULT_PERCEPTION_PHASE
    elif workflow_kind == _WORKFLOW_KIND_TLR:
        phase = ""
    else:
        with picker_cols[1]:
            st.markdown('<div class="wf-toolbar-note">Phase</div>', unsafe_allow_html=True)
            phase = st.text_input(
                "Phase",
                value=default_phase,
                key="workflow_phase",
                label_visibility="collapsed",
            )
    description_col = release_detail_cols[1] if release_mode else picker_cols[1 if workflow_kind == _WORKFLOW_KIND_TLR else 2]
    with description_col:
        st.markdown('<div class="wf-toolbar-note">Description</div>', unsafe_allow_html=True)
        description = st.text_input(
            "Description",
            value=get_config_value("workflow_description", ""),
            key="workflow_description",
            label_visibility="collapsed",
            placeholder="Optional label for the evaluator run",
        ).strip()

    trend_metadata: Dict[str, object] = {}
    if release_mode:
        metadata_default_key = "workflow_release_metadata_default_target"
        metadata_output_key = "workflow_release_metadata_output_path"
        metadata_source_key = "workflow_release_metadata_source_path"
        metadata_text_key = "workflow_release_metadata_text"
        existing_metadata_text = str(existing_release_context.get("metadata_text") or "")
        existing_metadata_source = str(existing_release_context.get("metadata_source") or "")
        if (
            st.session_state.get(metadata_output_key) != output_path
            or metadata_text_key not in st.session_state
        ):
            st.session_state[metadata_text_key] = existing_metadata_text or _make_default_release_metadata_text(target_name)
            st.session_state[metadata_default_key] = target_name
            st.session_state[metadata_output_key] = output_path
            st.session_state[metadata_source_key] = existing_metadata_source
        elif (
            not existing_metadata_text
            and st.session_state.get(metadata_default_key) != target_name
            and st.session_state.get(metadata_source_key) == ""
        ):
            st.session_state[metadata_text_key] = _make_default_release_metadata_text(target_name)
            st.session_state[metadata_default_key] = target_name

        metadata_source_path = str(st.session_state.get(metadata_source_key) or "")
        if metadata_source_path:
            st.caption(f"Loaded release metadata from `{metadata_source_path}`")

        current_metadata_text = str(st.session_state.get(metadata_text_key) or "")
        trend_topic_from_metadata = _extract_release_metadata_topic(current_metadata_text)
        option_values = list(_RELEASE_TREND_TOPIC_OPTIONS.values())
        topic_labels = list(_RELEASE_TREND_TOPIC_OPTIONS.keys())
        if trend_topic_from_metadata in option_values:
            topic_index = option_values.index(trend_topic_from_metadata)
        else:
            topic_index = topic_labels.index("Custom")
            st.session_state.setdefault("workflow_release_custom_trend_topic", trend_topic_from_metadata)

        topic_label_key = "workflow_release_trend_topic_label"
        topic_yaml_key = "workflow_release_trend_topic_yaml_value"
        if st.session_state.get(topic_yaml_key) != trend_topic_from_metadata:
            st.session_state[topic_label_key] = topic_labels[topic_index]
            st.session_state[topic_yaml_key] = trend_topic_from_metadata
            if topic_labels[topic_index] == "Custom":
                st.session_state["workflow_release_custom_trend_topic"] = trend_topic_from_metadata

        trend_topic_label = st.selectbox(
            "Trend topic",
            options=topic_labels,
            key=topic_label_key,
            help="Used only for trend graphs. The specsheet data topic is detected from parquet/csv separately.",
        )
        if trend_topic_label == "Custom":
            trend_topic = st.text_input(
                "Custom trend topic",
                value=st.session_state.get("workflow_release_custom_trend_topic", trend_topic_from_metadata),
                key="workflow_release_custom_trend_topic",
                placeholder="perception.object_recognition.objects",
            ).strip()
        else:
            trend_topic = _RELEASE_TREND_TOPIC_OPTIONS[trend_topic_label]
        if trend_topic and trend_topic != trend_topic_from_metadata:
            st.session_state[metadata_text_key] = _replace_release_metadata_topic(
                current_metadata_text,
                trend_topic,
            )
            st.session_state[topic_yaml_key] = trend_topic

        metadata_text = st.text_area(
            "Release metadata YAML",
            key=metadata_text_key,
            height=150,
            help=(
                "Required: tags: [trend], release_group, pilot_auto_version, data_count, description, date. "
                "date must look like 2026.5.22."
            ),
        )
        metadata_error = ""
        try:
            trend_metadata = parse_trend_metadata_text(metadata_text)
            if not str(trend_metadata.get("release_group") or "").strip():
                raise ValueError("Release metadata requires non-empty `release_group`.")
        except Exception as exc:
            metadata_error = str(exc)
            trend_metadata = {}
            st.error(f"Release metadata error: {metadata_error}")

        trend_topic_from_metadata = str(trend_metadata.get("topic_name") or "").strip()
        if release_mode and trend_metadata and not trend_topic_from_metadata:
            metadata_error = metadata_error or "Trend topic is required."
            st.error("Trend topic is required.")
        elif trend_metadata:
            st.success("Release metadata looks valid.")

        optional_catalog_enabled = st.checkbox(
            "Also run Planning Test catalog",
            value=bool(st.session_state.get("workflow_release_optional_catalog_enabled", False)),
            key="workflow_release_optional_catalog_enabled",
            help="Schedules the Planning Test catalog in addition to Performance and DevOps.",
        )
        existing_job_cols = st.columns(2)
        with existing_job_cols[0]:
            performance_job_id = st.text_input(
                "Existing Performance job ID",
                value=st.session_state.get("workflow_release_performance_job_id", ""),
                key="workflow_release_performance_job_id",
                placeholder="Leave empty to schedule a new Performance job",
                help="Use this when the release Performance evaluator job is already scheduled or finished.",
            ).strip()
        with existing_job_cols[1]:
            devops_job_id = st.text_input(
                "Existing DevOps job ID",
                value=st.session_state.get("workflow_release_devops_job_id", ""),
                key="workflow_release_devops_job_id",
                placeholder="Leave empty to schedule a new DevOps job",
                help="Use this when the release DevOps evaluator job is already scheduled or finished.",
            ).strip()
        if optional_catalog_enabled:
            optional_job_id = st.text_input(
                "Existing Planning Test job ID",
                value=st.session_state.get("workflow_release_optional_job_id", ""),
                key="workflow_release_optional_job_id",
                placeholder="Leave empty to schedule the Planning Test catalog",
                help="Use this when the Planning Test evaluator job is already scheduled or finished.",
            ).strip()
        else:
            optional_job_id = ""

        recorded_job_ids = (
            existing_release_context.get("job_ids")
            if isinstance(existing_release_context.get("job_ids"), dict)
            else {}
        )
        force_redownload_roles: list[str] = []

        def _render_redownload_option(role: str, label: str, entered_job_id: str) -> None:
            recorded_job_id = str(recorded_job_ids.get(role) or "").strip()
            if (
                not entered_job_id
                or not recorded_job_id
                or entered_job_id == recorded_job_id
                or not _release_role_has_local_artifacts(output_path, role)
            ):
                return
            st.warning(
                f"{label} job ID differs from the local folder record. "
                f"Recorded: `{recorded_job_id}` / entered: `{entered_job_id}`."
            )
            if st.checkbox(
                f"Clear existing {label} artifacts and download from entered job ID",
                value=False,
                key=f"workflow_release_force_redownload_{role}",
                help="Only this role subfolder will be cleared. Leave unchecked to keep using local artifacts.",
            ):
                force_redownload_roles.append(role)

        _render_redownload_option("performance", "Performance", performance_job_id)
        _render_redownload_option("devops", "DevOps", devops_job_id)
        if optional_catalog_enabled:
            _render_redownload_option("planning_test", "Planning Test", optional_job_id)

        def _render_release_role_plan(role: str, label: str, entered_job_id: str) -> None:
            has_local_artifacts = _release_role_has_local_artifacts(output_path, role)
            force_redownload = role in force_redownload_roles
            recorded_job_id = str(recorded_job_ids.get(role) or "").strip()
            if entered_job_id and force_redownload:
                st.warning(
                    f"{label}: will clear existing local artifacts and download from entered job `{entered_job_id}`."
                )
            elif entered_job_id:
                if has_local_artifacts:
                    st.info(
                        f"{label}: will wait for entered job `{entered_job_id}`, then keep using local artifacts already in this folder."
                    )
                else:
                    st.info(f"{label}: will use entered job `{entered_job_id}` for download and analysis.")
            elif has_local_artifacts:
                suffix = f" Recorded job: `{recorded_job_id}`." if recorded_job_id else ""
                st.success(f"{label}: will use existing local artifacts; no new evaluator job will be scheduled.{suffix}")
            else:
                suffix = f" Recorded job `{recorded_job_id}` is not auto-filled." if recorded_job_id else ""
                st.warning(f"{label}: no job ID entered and no local artifacts found; a new evaluator job will be scheduled.{suffix}")

        st.markdown("**Release execution plan**")
        _render_release_role_plan("performance", "Performance", performance_job_id)
        _render_release_role_plan("devops", "DevOps", devops_job_id)
        if optional_catalog_enabled:
            _render_release_role_plan("planning_test", "Planning Test", optional_job_id)
    else:
        performance_job_id = ""
        devops_job_id = ""
        optional_catalog_enabled = False
        optional_job_id = ""
        metadata_error = ""
        force_redownload_roles = []

    confirm_cols = st.columns([1.0, 1.0, 1.0] if release_mode and optional_catalog_enabled else [1.0, 1.0])
    with confirm_cols[0]:
        if release_mode:
            st.caption(f"Performance catalog: `{_RELEASE_PERFORMANCE_CATALOG_ID}`")
        elif catalog_id:
            st.caption(f"Catalog ID: `{catalog_id}`")
    with confirm_cols[1]:
        if release_mode:
            st.caption(f"DevOps catalog: `{_RELEASE_DEVOPS_CATALOG_ID}`")
        elif integration_id:
            st.caption(f"Integration ID: `{integration_id}`")
    if release_mode and optional_catalog_enabled:
        with confirm_cols[2]:
            st.caption(f"Planning Test catalog: `{_RELEASE_OPTIONAL_CATALOG_ID}`")
    if not release_mode and st.session_state.get("workflow_catalog_resolution_error"):
        st.warning(f"Could not resolve integration automatically: {st.session_state['workflow_catalog_resolution_error']}")

    if not release_mode and selected_catalog:
        desc = str(selected_catalog.get("description") or "").strip() or "Preset selected for quick scheduling."
        st.caption(f"Preset: {desc}")
    elif not release_mode and selected_server_catalog:
        desc = str(selected_server_catalog.get("description") or "").strip()
        if desc:
            st.caption(f"Fetched catalog: {desc}")

    with st.expander("Advanced options", expanded=False):
        if release_mode:
            download_type = "Archives (ZIP)"
            generate_parquet = False
            skip_large_file = _RELEASE_SKIP_LARGE_FILE
            eval_recursive = False
            st.caption(
                "Release mode always uses archive downloads, skips oversized files, and generates parquet automatically when needed."
            )
            adv_cols = st.columns([1.0, 0.8, 0.8])
        elif workflow_kind == _WORKFLOW_KIND_TLR:
            download_type = _TLR_DOWNLOAD_TYPE
            generate_parquet = False
            skip_large_file = False
            eval_recursive = True
            st.caption(
                "TLR mode downloads simulation result JSON for the Traffic Light Recognition analysis page."
            )
            adv_cols = st.columns([1.0, 0.8, 0.8])
        else:
            adv_cols = st.columns([1.0, 1.0, 0.8, 0.8])
            with adv_cols[0]:
                download_type = st.radio(
                    "Download type",
                    ["Archives (ZIP)", "Result JSON"],
                    horizontal=True,
                    index=0 if default_download_type == "Archives (ZIP)" else 1,
                    key="workflow_download_type",
                )
            env_col = adv_cols[1]
            poll_col = adv_cols[2]
            wait_col = adv_cols[3]

        if not release_mode:
            if catalog_auto_tlr_mode:
                st.checkbox(
                    "TLR mode",
                    value=True,
                    disabled=True,
                    key="workflow_tlr_mode_auto",
                    help="Enabled automatically because the selected catalog looks like a TLR catalog.",
                )
            else:
                manual_tlr_mode = st.checkbox(
                    "TLR mode",
                    value=bool(st.session_state.get("workflow_tlr_mode_manual", False)),
                    key="workflow_tlr_mode_manual",
                    help="Use result JSON downloads for Traffic Light Recognition analysis.",
                )
                if manual_tlr_mode != bool(workflow_kind == _WORKFLOW_KIND_TLR):
                    st.rerun()

        if release_mode or workflow_kind == _WORKFLOW_KIND_TLR:
            env_col = adv_cols[0]
            poll_col = adv_cols[1]
            wait_col = adv_cols[2]

        with env_col:
            environment = st.selectbox(
                "Environment",
                options=["", "dev", "stg", "prd"],
                index=["", "dev", "stg", "prd"].index(default_environment) if default_environment in ("", "dev", "stg", "prd") else 0,
                key="workflow_environment",
                format_func=lambda value: value or "default",
            )
        with poll_col:
            poll_interval = st.slider(
                "Poll interval (s)",
                min_value=10,
                max_value=300,
                value=default_poll_interval,
                step=10,
                key="workflow_poll_interval",
            )
        with wait_col:
            max_wait_hours = st.number_input(
                "Max wait (h, 0 = no timeout)",
                min_value=0,
                max_value=24 * 30,
                value=default_max_wait_hours,
                step=1,
                key="workflow_max_wait_hours",
                help="Set to 0 to keep waiting for evaluator completion without an app-side timeout.",
            )

        option_col_count = 2 if release_mode else (3 if workflow_kind == _WORKFLOW_KIND_TLR else 5)
        option_cols = st.columns(option_col_count)
        with option_cols[0]:
            run_eval = st.checkbox(
                "Run evaluation",
                value=False if release_mode or workflow_kind == _WORKFLOW_KIND_TLR else True,
                disabled=workflow_kind == _WORKFLOW_KIND_TLR,
                key="workflow_run_eval",
                help=(
                    "Optional in release mode. Turn this on to also generate Summary.csv and Score.csv."
                    if release_mode
                    else "TLR mode downloads result JSON directly for analysis."
                    if workflow_kind == _WORKFLOW_KIND_TLR
                    else "Generate Summary.csv and Score.csv after download."
                ),
            )
        if workflow_kind == _WORKFLOW_KIND_TLR:
            with option_cols[1]:
                st.checkbox(
                    "Generate parquet",
                    value=False,
                    disabled=True,
                    key="workflow_generate_parquet_tlr",
                    help="TLR result JSON is analyzed directly.",
                )
            tag_col = option_cols[2]
        elif not release_mode:
            with option_cols[1]:
                generate_parquet = st.checkbox(
                    "Generate parquet",
                    value=CATALOG_IO_AVAILABLE,
                    disabled=not CATALOG_IO_AVAILABLE,
                    key="workflow_generate_parquet",
                )
            with option_cols[2]:
                skip_large_file = st.checkbox(
                    "Skip large files",
                    value=default_skip_large_file,
                    key="workflow_skip_large_file",
                    help="Skip unusually large archives during download.",
                )
            with option_cols[3]:
                eval_recursive = st.checkbox(
                    "Recursive scan",
                    value=True,
                    key="workflow_eval_recursive",
                )
            tag_col = option_cols[4]
        else:
            tag_col = option_cols[1]

        with tag_col:
            is_tag = st.checkbox("Target is tag", value=False, key="workflow_is_tag")
            # §6: drop estimated polygon objects so spec-sheet metrics reflect only
            # planning-relevant objects (analyzer >=0.2.0 SceneDataFrame.from_dir).
            is_exclude_polygons = st.checkbox(
                "Exclude polygons (spec-sheet)",
                value=False,
                key="workflow_is_exclude_polygons",
                help=(
                    "Drop estimated polygon objects when computing spec-sheet metrics. "
                    "Requires perception_catalog_analyzer >= 0.2.0."
                ),
            )

    if workflow_kind == _WORKFLOW_KIND_TLR:
        download_type = _TLR_DOWNLOAD_TYPE
        phase = ""
        run_eval = False
        generate_parquet = False
        skip_large_file = False
        eval_recursive = True

    set_config_value("eval_project_id", project_id)
    set_config_value("target_name", target_name)
    set_config_value("workflow_kind", workflow_kind)
    if not release_mode:
        set_config_value("eval_download_type", download_type)
        if workflow_kind != _WORKFLOW_KIND_TLR:
            set_config_value("eval_phase", phase)
    set_config_value("poll_interval", poll_interval)
    set_config_value("max_wait_hours", max_wait_hours)
    set_config_value("environment", environment)
    set_config_value("workflow_description", description)
    errors = []
    if not project_id:
        errors.append("Project ID")
    if not release_mode and not catalog_id:
        errors.append("Catalog")
    if not release_mode and not integration_id:
        errors.append("Integration ID")
    if not target_name:
        errors.append("Branch or tag")
    if release_mode:
        if not trend_metadata.get("release_group"):
            errors.append("Release group")
        if not trend_metadata.get("pilot_auto_version"):
            errors.append("Pilot.Auto version")
        if not trend_metadata.get("data_count"):
            errors.append("Data count")
        if not trend_metadata.get("date"):
            errors.append("Release date")
        if metadata_error:
            errors.append(metadata_error)

    resolved_output = None
    path_error = ""
    if output_path:
        resolved_output, path_error = resolve_under_data_root(output_path, allow_missing=True)
        if path_error:
            errors.append(path_error)
    else:
        errors.append("Output folder")

    return {
        "project_id": project_id,
        "environment": environment,
        "output_path_default": output_path or _make_default_output_path(target_name),
        "download_type_default": download_type,
        "phase_default": phase,
        "skip_large_file_default": True,
        "large_file_mb_default": 50.0,
        "keep_zip_files_default": False,
        "dialog_payload": {
            "errors": errors,
            "project_id": project_id,
            "catalog_id": catalog_id,
            "integration_id": integration_id,
            "catalog_preset_name": selected_catalog_name,
            "has_custom_catalog": bool(catalog_id and not selected_catalog),
            "target_name": target_name,
            "description": description,
            "resolved_output": str(resolved_output) if resolved_output else "",
            "environment": environment,
            "is_tag": is_tag,
            "is_exclude_polygons": bool(is_exclude_polygons) if release_mode else False,
            "download_type": download_type,
            "phase": phase,
            "poll_interval": int(poll_interval),
            "max_wait_hours": int(max_wait_hours),
            "run_eval": bool(run_eval),
            "generate_parquet": False if release_mode else bool(generate_parquet),
            "skip_large_file": _RELEASE_SKIP_LARGE_FILE if release_mode else bool(skip_large_file),
            "eval_recursive": False if release_mode else bool(eval_recursive),
            "workflow_kind": workflow_kind,
            "release_mode": bool(release_mode),
            "trend_metadata": trend_metadata if release_mode else {},
            "performance_job_id": performance_job_id if release_mode else "",
            "devops_job_id": devops_job_id if release_mode else "",
            "optional_catalog_enabled": bool(optional_catalog_enabled) if release_mode else False,
            "optional_catalog_id": _RELEASE_OPTIONAL_CATALOG_ID if release_mode and optional_catalog_enabled else "",
            "optional_job_id": optional_job_id if release_mode and optional_catalog_enabled else "",
            "force_redownload_roles": force_redownload_roles if release_mode else [],
        },
    }


def _render_workflow_launcher_section(
    catalog_presets: List[Dict[str, str]],
    catalogs_path: Optional[str],
    catalog_load_error: Optional[str],
) -> Dict[str, object]:
    section_header("Run Evaluator Workflow", "")
    start_defaults = _get_start_workflow_defaults()
    if _WORKFLOW_START_DIALOG_KEY not in st.session_state:
        st.session_state[_WORKFLOW_START_DIALOG_KEY] = False
    new_job_clicked = st.button(
        "Start new workflow",
        key="workflow_open_start_dialog",
        type="primary",
        use_container_width=False,
    )

    def _reset_start_workflow_state() -> None:
        fresh_target = str(get_config_value("target_name", "beta/v4.3.2") or "beta/v4.3.2")
        st.session_state["workflow_catalog_name"] = ""
        st.session_state["workflow_last_catalog_preset"] = ""
        st.session_state["workflow_catalog_id"] = ""
        st.session_state["workflow_integration_id"] = ""
        st.session_state["workflow_server_catalogs"] = []
        st.session_state["workflow_server_catalog_error"] = ""
        st.session_state["workflow_selected_server_catalog_id"] = ""
        st.session_state["workflow_selected_server_catalog_label"] = ""
        st.session_state["workflow_catalog_resolution_error"] = ""
        st.session_state["workflow_last_catalog_selection"] = ""
        st.session_state["workflow_release_performance_job_id"] = ""
        st.session_state["workflow_release_devops_job_id"] = ""
        st.session_state["workflow_release_trend_topic_label"] = "Prediction / object recognition"
        st.session_state["workflow_release_custom_trend_topic"] = ""
        st.session_state.pop("workflow_release_metadata_default_target", None)
        st.session_state.pop("workflow_release_metadata_output_path", None)
        st.session_state.pop("workflow_release_metadata_source_path", None)
        st.session_state.pop("workflow_release_metadata_text", None)
        if bool(st.session_state.get("workflow_release_mode", False)):
            st.session_state["workflow_run_eval"] = False
        else:
            st.session_state.pop("workflow_run_eval", None)
        st.session_state["workflow_max_wait_hours"] = _DEFAULT_MAX_WAIT_HOURS
        st.session_state["workflow_output_path"] = _make_default_output_path(fresh_target)

    def _render_start_workflow_controls(*, key_suffix: str = "dialog") -> None:
        payload = _render_start_workflow_form(catalog_presets, catalogs_path, catalog_load_error)
        submit_cols = st.columns([1.15, 1.15, 3.7])
        close_clicked = submit_cols[0].button(
            "Close",
            key=f"workflow_close_start_{key_suffix}",
            use_container_width=True,
        )
        start_clicked = submit_cols[1].button(
            "Start workflow",
            key=f"workflow_start_btn_{key_suffix}",
            type="primary",
            use_container_width=True,
        )
        if close_clicked:
            _close_workflow_dialog(_WORKFLOW_START_DIALOG_KEY)
            st.rerun()
        if start_clicked:
            dialog_payload = dict(payload.get("dialog_payload") or {})
            errors = dialog_payload.get("errors", [])
            if errors:
                for err in errors:
                    st.error(f"Missing or invalid: {err}")
            elif not is_task_queue_enabled():
                st.error("Task queue not enabled. Set `USE_TASK_QUEUE=true` and `REDIS_URL`.")
            else:
                common_params = {
                    "project_id": dialog_payload["project_id"],
                    "suite_ids": None,
                    "target_name": dialog_payload["target_name"],
                    "environment": dialog_payload["environment"],
                    "max_retries": 0,
                    "clean_build": False,
                    "debug": False,
                    "release": False,
                    "record_caret": False,
                    "log_expiration_time_in_days": 14.0,
                    "is_tag": dialog_payload["is_tag"],
                    "workflow_kind": dialog_payload.get("workflow_kind", _WORKFLOW_KIND_PERCEPTION),
                    "download_type": "archives" if dialog_payload["download_type"] == "Archives (ZIP)" else "result_json",
                    "phase": dialog_payload["phase"],
                    "skip_large_file": bool(dialog_payload.get("skip_large_file", True)),
                    "large_file_mb": 50.0,
                    "keep_zip_files": False,
                    "poll_interval": dialog_payload["poll_interval"],
                    "max_wait_seconds": dialog_payload["max_wait_hours"] * 3600,
                    "run_eval": dialog_payload["run_eval"],
                    "generate_parquet": dialog_payload["generate_parquet"],
                    "eval_recursive": dialog_payload["eval_recursive"],
                    "eval_overwrite": False,
                }
                if dialog_payload.get("release_mode"):
                    base_description = dialog_payload["description"] or _make_auto_release_workflow_description(
                        dialog_payload["target_name"]
                    )
                    trend_metadata = dict(dialog_payload.get("trend_metadata") or {})
                    task_id = _enqueue_task(
                        "run_release_specsheet_workflow",
                        {
                            "project_id": dialog_payload["project_id"],
                            "target_name": dialog_payload["target_name"],
                            "description": base_description,
                            "output_path": dialog_payload["resolved_output"],
                            "environment": dialog_payload["environment"],
                            "is_tag": dialog_payload["is_tag"],
                            "is_exclude_polygons": dialog_payload.get("is_exclude_polygons", False),
                            "poll_interval": dialog_payload["poll_interval"],
                            "max_wait_seconds": dialog_payload["max_wait_hours"] * 3600,
                            "trend_metadata": trend_metadata,
                            "version": trend_metadata.get("pilot_auto_version", ""),
                            "topic": trend_metadata.get("topic_name", ""),
                            "performance_catalog_id": _RELEASE_PERFORMANCE_CATALOG_ID,
                            "performance_integration_id": _RELEASE_PERFORMANCE_INTEGRATION_ID,
                            "performance_job_id": dialog_payload.get("performance_job_id", ""),
                            "devops_catalog_id": _RELEASE_DEVOPS_CATALOG_ID,
                            "devops_integration_id": _RELEASE_DEVOPS_INTEGRATION_ID,
                            "devops_job_id": dialog_payload.get("devops_job_id", ""),
                            "optional_catalog_enabled": bool(dialog_payload.get("optional_catalog_enabled", False)),
                            "optional_catalog_id": dialog_payload.get("optional_catalog_id", ""),
                            "optional_job_id": dialog_payload.get("optional_job_id", ""),
                            "force_redownload_roles": list(dialog_payload.get("force_redownload_roles") or []),
                            "analysis_phase": "perception.object_recognition.objects",
                            "skip_large_file": _RELEASE_SKIP_LARGE_FILE,
                            "large_file_mb": _RELEASE_LARGE_FILE_MB,
                            "run_eval": bool(dialog_payload.get("run_eval", False)),
                            "overwrite": True,
                        },
                    )
                    if task_id:
                        _close_workflow_dialog(_WORKFLOW_START_DIALOG_KEY)
                        st.success(f"Release specsheet workflow queued. Task id: `{task_id}`")
                        st.rerun()
                    else:
                        st.error("Failed to enqueue release specsheet workflow. Check worker logs.")
                    return

                task_id = _enqueue_task(
                    "run_evaluator_and_process",
                    {
                        **common_params,
                        "catalog_id": dialog_payload["catalog_id"],
                        "integration_id": dialog_payload["integration_id"],
                        "catalog_preset_name": dialog_payload.get("catalog_preset_name", ""),
                        "description": dialog_payload["description"] or _make_auto_workflow_description(
                            dialog_payload["target_name"],
                            dialog_payload.get("catalog_preset_name", ""),
                            has_custom_catalog=bool(dialog_payload.get("has_custom_catalog", False)),
                        ),
                        "output_path": dialog_payload["resolved_output"],
                    },
                )
                if task_id:
                    _close_workflow_dialog(_WORKFLOW_START_DIALOG_KEY)
                    st.success(f"Workflow queued. Task id: `{task_id}`")
                    st.rerun()
                else:
                    st.error("Failed to enqueue task. Check worker logs.")

    if new_job_clicked:
        _open_exclusive_workflow_dialog(_WORKFLOW_START_DIALOG_KEY)
        _reset_start_workflow_state()

    if st.session_state.get(_WORKFLOW_START_DIALOG_KEY):
        _close_workflow_dialog(_WORKFLOW_PR_BRANCH_DIALOG_KEY)
        if callable(getattr(st, "dialog", None)):
            @st.dialog("Start evaluator workflow", width="large")
            def _workflow_start_dialog() -> None:
                _render_start_workflow_controls(key_suffix="dialog")

            _workflow_start_dialog()
        else:
            st.markdown("---")
            st.subheader("Start evaluator workflow")
            _render_start_workflow_controls(key_suffix="inline")

    return start_defaults


def _render_pr_test_branch_launcher_section() -> None:
    section_header("Prepare Git Test Branch", "")
    if _WORKFLOW_PR_BRANCH_DIALOG_KEY not in st.session_state:
        st.session_state[_WORKFLOW_PR_BRANCH_DIALOG_KEY] = False
    if st.button(
        "Prepare PR test branch",
        key="workflow_open_pr_branch_dialog",
        use_container_width=False,
        help="Create local test branches in a reusable pilot-auto checkout and restore the checkout afterwards.",
    ):
        _open_exclusive_workflow_dialog(_WORKFLOW_PR_BRANCH_DIALOG_KEY)

    def _render_pr_branch_controls(*, key_suffix: str = "dialog") -> None:
        st.caption("Prepare a local pilot branch from a sub-repo PR, then restore the reusable checkout.")
        default_base_branch = str(get_config_value("pr_test_pilot_base_branch", "main"))
        default_sub_repo = str(get_config_value("pr_test_sub_repo", "universe"))
        sub_repo_options = ["universe", "launcher"]
        sub_repo_index = sub_repo_options.index(default_sub_repo) if default_sub_repo in sub_repo_options else 0

        top_cols = st.columns([1.0, 1.0, 1.35])
        with top_cols[0]:
            sub_repo = st.selectbox(
                "Sub repo",
                options=sub_repo_options,
                index=sub_repo_index,
                key=f"workflow_pr_sub_repo_{key_suffix}",
            )
        with top_cols[1]:
            pr_number = st.text_input(
                "PR number",
                value=str(get_config_value("pr_test_pr_number", "")),
                key=f"workflow_pr_number_{key_suffix}",
            ).strip()
        with top_cols[2]:
            pilot_base_branch = st.text_input(
                "Base pilot branch",
                value=default_base_branch,
                placeholder="main or beta/v4.x",
                key=f"workflow_pr_pilot_base_{key_suffix}",
            ).strip()

        work_dir = str(get_config_value("pr_test_work_dir", str(DEFAULT_WORK_DIR)) or str(DEFAULT_WORK_DIR))
        pilot_checkout = str(get_config_value("pr_test_pilot_checkout", DEFAULT_PILOT_CHECKOUT) or "")
        pilot_repo_url = str(get_config_value("pr_test_pilot_repo_url", DEFAULT_PILOT_REPO_URL) or DEFAULT_PILOT_REPO_URL)
        pilot_remote = "origin"
        sub_remote = "origin"
        branch_prefix = DEFAULT_BRANCH_PREFIX
        run_vcs_update = False
        reset_cache = True
        restore_after = True
        prepare_only = False

        errors = []
        if not pilot_base_branch:
            errors.append("Base pilot branch")
        if not pr_number:
            errors.append("PR number")
        if not pilot_checkout and not pilot_repo_url:
            errors.append("Set PR_TEST_BRANCH_PILOT_REPO_URL")
        if pr_number and not pr_number.isdigit():
            errors.append("PR number must be numeric")

        if errors:
            st.warning("Missing or invalid: " + ", ".join(errors))
        else:
            st.caption(
                f"Uses `{work_dir}/pilot-auto` as the app cache, syncs only the selected repo from `autoware.repos`, "
                "pushes the prepared branches, and restores the checkout afterwards."
            )

        action_cols = st.columns([1.05, 1.35, 3.6])
        close_clicked = action_cols[0].button("Close", key=f"workflow_pr_close_{key_suffix}", use_container_width=True)
        start_clicked = action_cols[1].button(
            "Start git task",
            key=f"workflow_pr_start_{key_suffix}",
            type="primary",
            use_container_width=True,
        )
        if close_clicked:
            _close_workflow_dialog(_WORKFLOW_PR_BRANCH_DIALOG_KEY)
            st.rerun()
        if start_clicked:
            if errors:
                for err in errors:
                    st.error(f"Missing or invalid: {err}")
                return
            if not is_task_queue_enabled():
                st.error("Task queue not enabled. Set `USE_TASK_QUEUE=true`, `DATABASE_URL`, and `REDIS_URL`.")
                return
            for key, value in {
                "pr_test_work_dir": work_dir,
                "pr_test_pilot_checkout": pilot_checkout,
                "pr_test_pilot_repo_url": pilot_repo_url,
                "pr_test_pilot_base_branch": pilot_base_branch,
                "pr_test_sub_repo": sub_repo,
                "pr_test_pr_number": pr_number,
            }.items():
                set_config_value(key, value)
            params = {
                "work_dir": work_dir,
                "pilot_checkout": pilot_checkout,
                "pilot_repo_url": pilot_repo_url,
                "pilot_base_branch": pilot_base_branch,
                "pilot_remote": pilot_remote or "origin",
                "sub_repo": sub_repo,
                "sub_remote": sub_remote or "origin",
                "sub_repo_branch": "",
                "pr_number": pr_number,
                "branch_prefix": branch_prefix or DEFAULT_BRANCH_PREFIX,
                "run_vcs_update": run_vcs_update,
                "reset_cache": reset_cache,
                "restore_after": restore_after,
                "prepare_only": prepare_only,
            }
            task_id = _enqueue_task("prepare_pr_test_branch", params)
            if task_id:
                _close_workflow_dialog(_WORKFLOW_PR_BRANCH_DIALOG_KEY)
                st.success(f"Git branch preparation queued. Task id: `{task_id}`")
                st.rerun()
            else:
                st.error("Failed to enqueue git branch preparation. Check worker logs.")

    if st.session_state.get(_WORKFLOW_PR_BRANCH_DIALOG_KEY) and not st.session_state.get(_WORKFLOW_START_DIALOG_KEY):
        if callable(getattr(st, "dialog", None)):
            @st.dialog("Prepare PR test branch", width="large")
            def _workflow_pr_branch_dialog() -> None:
                _render_pr_branch_controls(key_suffix="dialog")

            _workflow_pr_branch_dialog()
        else:
            st.markdown("---")
            st.subheader("Prepare PR test branch")
            _render_pr_branch_controls(key_suffix="inline")


_inject_workflow_page_styles()
render_page_hero(
    kicker="Evaluator tasks",
    title="Evaluator Workflow",
    description="Browse finished runs, watch background tasks, start evaluator runs, and reuse existing evaluator reports from one page.",
)

catalog_presets, catalogs_path, catalog_load_error = _load_catalog_presets()

tab_tasks, tab_local = st.tabs(["Run Tasks", "Local Runs"])

with tab_tasks:
    _render_current_tasks_section()
    start_defaults = _render_workflow_launcher_section(catalog_presets, catalogs_path, catalog_load_error)
    _render_pr_test_branch_launcher_section()

    configure_recent_evaluator_jobs_ui(
        get_config_value=get_config_value,
        set_config_value=set_config_value,
        enqueue_task=_enqueue_task,
        catalog_io_available=CATALOG_IO_AVAILABLE,
        environment=str(start_defaults["environment"] or ""),
    )

    _render_recent_evaluator_jobs_section(
        str(start_defaults["project_id"] or ""),
        str(start_defaults["environment"] or ""),
        output_path_default=str(start_defaults["output_path_default"]),
        download_type_default=str(start_defaults["download_type_default"]),
        phase_default=str(start_defaults["phase_default"]),
        skip_large_file_default=bool(start_defaults["skip_large_file_default"]),
        large_file_mb_default=float(start_defaults["large_file_mb_default"]),
        keep_zip_files_default=bool(start_defaults["keep_zip_files_default"]),
        show_toggle=False,
        default_visible=True,
        show_title=False,
    )

with tab_local:
    use_fragment = getattr(st, "fragment", None) is not None
    if use_fragment:
        try:

            @st.fragment
            def _local_runs_fragment():
                _render_local_runs_section()

            _local_runs_fragment()
        except (TypeError, AttributeError):
            _render_local_runs_section()
    else:
        _render_local_runs_section()
