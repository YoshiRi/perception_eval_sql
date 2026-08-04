"""Server-side workflow API: start and watch evaluator pipelines without a browser.

The dashboard's own launcher lives in ``pages/6_Workflow.py`` and is only reachable by a
human with a Streamlit session. These routes expose the same pipelines over HTTP so the
packaged local client -- and any script -- can queue work: they build the same
``parameters`` dict, create the same Postgres task row, and enqueue the same
``worker.tasks.run_job``, so a run started here is indistinguishable from one started in
the page.

Mounted next to the export routes on the shared handler (see ``local_bbox_api``) and
authorized the same way, so a client that is already trusted to download runs is trusted
to start them. Everything from ``lib/`` and ``worker/`` is imported lazily: the packaged
client ships ``backend/`` but not those trees, and a module-scope import would break it.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

try:
    from backend import export_api
except ImportError:  # pragma: no cover - frozen client puts backend/ on sys.path directly
    import export_api  # type: ignore[no-redef]


_JST = timezone(timedelta(hours=9))

# Mirrors of the page's constants. Duplicated rather than imported because pages/ is a
# Streamlit module: importing it would execute st.set_page_config at request time.
WORKFLOW_KIND_PERCEPTION = "Perception"
WORKFLOW_KIND_TLR = "TLR"
DEFAULT_PERCEPTION_PHASE = "perception.object_recognition.objects"
DEFAULT_MAX_WAIT_HOURS = 48
DEFAULT_POLL_INTERVAL = 60

RELEASE_PERFORMANCE_CATALOG_ID = "e36d75b9-6c3a-4970-9b9b-5cd13f7a9da3"
RELEASE_PERFORMANCE_INTEGRATION_ID = "96ad8fba-0228-4c2b-9166-07d4de1a0760"
RELEASE_DEVOPS_CATALOG_ID = "ab0f8498-cc1b-4726-836f-e18e8bcb3200"
RELEASE_DEVOPS_INTEGRATION_ID = "295cff78-9bc9-4d60-b7aa-f95be6ff96a4"
RELEASE_OPTIONAL_CATALOG_ID = "09039022-ec91-41bf-9e93-fdefccdfc9bc"
RELEASE_SKIP_LARGE_FILE = True
RELEASE_LARGE_FILE_MB = 50.0

# What a caller may ask for, and which worker task each kind runs.
KIND_TASK_TYPES = {
    "perception": "run_evaluator_and_process",
    "tlr": "run_evaluator_and_process",
    "release": "run_release_specsheet_workflow",
}
KIND_LABELS = {
    "perception": "Perception evaluator workflow",
    "tlr": "TLR evaluator workflow (result JSON)",
    "release": "Release spec-sheet workflow",
}
# Types this API is willing to report on, so a task list here cannot leak unrelated
# server activity (an admin's PR-branch prep, say) to a laptop client.
REPORTED_TASK_TYPES = tuple(sorted(set(KIND_TASK_TYPES.values())))

_RQ_JOB_TIMEOUT_DEFAULT_SEC = 7 * 24 * 3600
_MAX_LOG_CHARS = 200_000


class WorkflowError(export_api.ExportError):
    """A workflow request that cannot be honoured as sent."""


# ------------------------------------------------------------------ payload coercion


def _text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text or default


def _flag(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text not in ("0", "false", "no", "off")


def _number(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _rq_timeout_sec() -> int:
    raw = os.environ.get("RQ_JOB_TIMEOUT_SEC", "").strip()
    if not raw:
        return _RQ_JOB_TIMEOUT_DEFAULT_SEC
    try:
        return max(60, int(raw, 10))
    except ValueError:
        return _RQ_JOB_TIMEOUT_DEFAULT_SEC


def _iso(value: Any) -> str:
    """Stringify a DB timestamp. The JSON responder does not serialise datetimes."""
    if isinstance(value, datetime):
        return value.isoformat()
    return _text(value)


# ------------------------------------------------------------------------- defaults


def default_output_path(target_name: str) -> str:
    """The page's naming scheme, so folders look the same whoever launched the run."""
    clean = re.sub(r"[^\w]", "_", target_name.strip("/")) if target_name else "eval"
    clean = re.sub(r"_+", "_", clean).strip("_")
    return f"eval_{clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _safe_part(value: Any, fallback: str) -> str:
    text = re.sub(r"[^\w.\-]+", "_", str(value or "").strip()).strip("._")
    return text or fallback


def _auto_description(target_name: str, *, release: bool) -> str:
    target = re.sub(r"\s+", " ", _text(target_name, "default"))
    stamp = datetime.now().strftime("%m-%d %H:%M")
    kind = "release" if release else "evaluator"
    return f"🚀 {kind} workflow [{target}] [{stamp}] 🖥️"


def default_release_metadata_text(target_name: str) -> str:
    """A starting point for the release form, matching what the page pre-fills."""
    version_match = re.search(r"v?(\d+\.\d+\.\d+)", _text(target_name))
    pilot_auto_version = (
        f"Pilot.Auto v{version_match.group(1)}" if version_match
        else f"Pilot.Auto {_text(target_name, 'release')}"
    )
    abbr = _safe_part(pilot_auto_version.replace("Pilot.Auto", "").strip(), "release")[:16]
    return (
        "tags: [trend]\n"
        f"release_group: {_safe_part(target_name, 'release')}\n"
        f'pilot_auto_version: "{pilot_auto_version}"\n'
        f"pilot_auto_version_abbr: {abbr}\n"
        "data_count: 99,776+\n"
        f"description: {_text(target_name, 'Release')} release data update\n"
        f"date: {datetime.now(_JST).strftime('%Y.%m.%d')}\n"
        f"topic_name: {DEFAULT_PERCEPTION_PHASE}\n"
    )


# --------------------------------------------------------------------- param building


def _resolve_output(output_path: str, target_name: str) -> str:
    """Absolute output folder under the server's data root, as the worker expects."""
    from lib.path_utils import resolve_under_data_root

    requested = output_path or default_output_path(target_name)
    resolved, error = resolve_under_data_root(requested, allow_missing=True)
    if error:
        raise WorkflowError(f"Invalid output folder: {error}")
    if not resolved:
        raise WorkflowError("Could not resolve the output folder under the data root.")
    return str(resolved)


def _common_params(payload: dict[str, Any], *, kind: str) -> dict[str, Any]:
    project_id = _text(payload.get("project_id"))
    target_name = _text(payload.get("target_name"))
    missing = [name for name, value in (("project_id", project_id), ("target_name", target_name)) if not value]
    if missing:
        raise WorkflowError(f"Missing required field(s): {', '.join(missing)}")

    tlr = kind == "tlr"
    release = kind == "release"
    download_type = _text(payload.get("download_type"), "archives").lower()
    if download_type not in ("archives", "result_json"):
        raise WorkflowError("download_type must be 'archives' or 'result_json'.")
    if tlr:
        # TLR analysis reads simulation result JSON; archives would produce nothing it
        # can open, so the choice is not the caller's to make.
        download_type = "result_json"
    elif release:
        download_type = "archives"

    max_wait_hours = int(_number(
        payload.get("max_wait_hours", DEFAULT_MAX_WAIT_HOURS), DEFAULT_MAX_WAIT_HOURS,
        minimum=0, maximum=24 * 30,
    ))
    return {
        "project_id": project_id,
        "suite_ids": None,
        "target_name": target_name,
        "environment": _text(payload.get("environment")),
        "max_retries": 0,
        "clean_build": False,
        "debug": False,
        "release": False,
        "record_caret": False,
        "log_expiration_time_in_days": 14.0,
        "is_tag": _flag(payload.get("is_tag"), False),
        "workflow_kind": WORKFLOW_KIND_TLR if tlr else WORKFLOW_KIND_PERCEPTION,
        "download_type": download_type,
        "phase": "" if tlr else _text(payload.get("phase"), DEFAULT_PERCEPTION_PHASE),
        "skip_large_file": False if tlr else _flag(payload.get("skip_large_file"), True),
        "large_file_mb": _number(payload.get("large_file_mb", 50.0), 50.0, minimum=1.0, maximum=100_000.0),
        "keep_zip_files": False,
        "poll_interval": int(_number(
            payload.get("poll_interval", DEFAULT_POLL_INTERVAL), DEFAULT_POLL_INTERVAL,
            minimum=10, maximum=300,
        )),
        "max_wait_seconds": max_wait_hours * 3600,
        "run_eval": False if tlr else _flag(payload.get("run_eval"), True),
        "generate_parquet": False if tlr else _flag(payload.get("generate_parquet"), True),
        "eval_recursive": True if tlr else _flag(payload.get("eval_recursive"), True),
        "eval_overwrite": False,
    }


def build_evaluator_params(payload: dict[str, Any], *, kind: str) -> dict[str, Any]:
    """Parameters for ``run_evaluator_and_process`` (the Perception and TLR kinds)."""
    common = _common_params(payload, kind=kind)
    catalog_id = _text(payload.get("catalog_id"))
    integration_id = _text(payload.get("integration_id"))
    if not catalog_id:
        raise WorkflowError("Missing required field(s): catalog_id")
    if not integration_id:
        # Resolvable from the catalog, but only by calling the evaluator API, which can
        # fail for reasons the caller should see separately from a start request.
        raise WorkflowError(
            "Missing required field(s): integration_id. Call /api/workflow_catalogs with "
            "resolve_catalog_id set to look it up for a catalog."
        )
    return {
        **common,
        "catalog_id": catalog_id,
        "integration_id": integration_id,
        "catalog_preset_name": _text(payload.get("catalog_preset_name")),
        "description": _text(payload.get("description")) or _auto_description(
            common["target_name"], release=False
        ),
        "output_path": _resolve_output(_text(payload.get("output_path")), common["target_name"]),
    }


def build_release_params(payload: dict[str, Any]) -> dict[str, Any]:
    """Parameters for ``run_release_specsheet_workflow``."""
    from lib.specsheet_report import parse_trend_metadata_text

    common = _common_params(payload, kind="release")
    raw_metadata = payload.get("trend_metadata")
    metadata_text = _text(payload.get("metadata_text"))
    if isinstance(raw_metadata, dict) and raw_metadata:
        trend_metadata = dict(raw_metadata)
    elif metadata_text:
        try:
            trend_metadata = parse_trend_metadata_text(metadata_text)
        except ValueError as exc:
            raise WorkflowError(f"Invalid trend metadata: {exc}") from exc
    else:
        raise WorkflowError(
            "A release workflow needs trend metadata. Send metadata_text (YAML) or a "
            "trend_metadata object; /api/workflow_health returns a filled-in template."
        )
    for field in ("release_group", "pilot_auto_version", "data_count", "date"):
        if not _text(trend_metadata.get(field)):
            raise WorkflowError(f"Trend metadata is missing `{field}`.")

    optional_enabled = _flag(payload.get("optional_catalog_enabled"), False)
    return {
        "project_id": common["project_id"],
        "target_name": common["target_name"],
        "description": _text(payload.get("description")) or _auto_description(
            common["target_name"], release=True
        ),
        "output_path": _resolve_output(_text(payload.get("output_path")), common["target_name"]),
        "environment": common["environment"],
        "is_tag": common["is_tag"],
        "is_exclude_polygons": _flag(payload.get("is_exclude_polygons"), False),
        "poll_interval": common["poll_interval"],
        "max_wait_seconds": common["max_wait_seconds"],
        "trend_metadata": trend_metadata,
        "version": _text(trend_metadata.get("pilot_auto_version")),
        "topic": _text(trend_metadata.get("topic_name")),
        "performance_catalog_id": RELEASE_PERFORMANCE_CATALOG_ID,
        "performance_integration_id": RELEASE_PERFORMANCE_INTEGRATION_ID,
        "performance_job_id": _text(payload.get("performance_job_id")),
        "devops_catalog_id": RELEASE_DEVOPS_CATALOG_ID,
        "devops_integration_id": RELEASE_DEVOPS_INTEGRATION_ID,
        "devops_job_id": _text(payload.get("devops_job_id")),
        "optional_catalog_enabled": optional_enabled,
        "optional_catalog_id": RELEASE_OPTIONAL_CATALOG_ID if optional_enabled else "",
        "optional_job_id": _text(payload.get("optional_job_id")) if optional_enabled else "",
        "force_redownload_roles": [
            _text(role) for role in (payload.get("force_redownload_roles") or []) if _text(role)
        ],
        "analysis_phase": DEFAULT_PERCEPTION_PHASE,
        "skip_large_file": RELEASE_SKIP_LARGE_FILE,
        "large_file_mb": RELEASE_LARGE_FILE_MB,
        "run_eval": _flag(payload.get("run_eval"), False),
        "overwrite": True,
    }


def build_params(payload: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Resolve a request into ``(kind, task_type, parameters)``."""
    kind = _text(payload.get("kind"), "perception").lower()
    task_type = KIND_TASK_TYPES.get(kind)
    if not task_type:
        raise WorkflowError(f"Unknown kind '{kind}'. Expected one of: {', '.join(KIND_TASK_TYPES)}")
    if kind == "release":
        return kind, task_type, build_release_params(payload)
    return kind, task_type, build_evaluator_params(payload, kind=kind)


# ------------------------------------------------------------------------ enqueueing


def _requester(who: dict[str, str]) -> dict[str, str]:
    """Attribution for the task row, in the shape the page's identity dict uses."""
    actor = _text(who.get("actor"))
    email = actor if "@" in actor else ""
    return {
        "id": actor,
        "email": email,
        "name": actor or "workflow api",
        "source": f"workflow_api:{_text(who.get('via'), 'unknown')}",
    }


def enqueue(task_type: str, params: dict[str, Any], who: dict[str, str]) -> str:
    """Create the task row and hand the job to RQ. Returns the task id."""
    from lib.db import create_task, is_task_queue_enabled, update_task_rq_job_id, update_task_status

    if not is_task_queue_enabled():
        raise WorkflowError(
            "This server cannot queue workflows: set USE_TASK_QUEUE=true, DATABASE_URL "
            "and REDIS_URL, and run a worker."
        )
    identity = _requester(who)
    params = {**params, "_requester": identity}
    # session_id is what the dashboard filters "my tasks" by, and it only means anything
    # for a real identity; a token or anonymous caller gets an unowned row instead of a
    # row filed under the literal string "token".
    session_id = identity["email"] or None

    task_id = create_task(task_type, params, session_id=session_id)
    if not task_id:
        raise WorkflowError(
            "Could not create the task row. Check DATABASE_URL and the server's task schema."
        )
    timeout = _rq_timeout_sec()
    try:
        from redis import Redis
        from rq import Queue

        from worker.tasks import run_job

        queue = Queue(
            name=os.environ.get("RQ_QUEUE", "default"),
            connection=Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379")),
            default_timeout=timeout,
        )
        job = queue.enqueue(run_job, task_id, task_type, params, job_timeout=timeout, result_ttl=timeout)
    except Exception as exc:
        # The row already exists, so leaving it pending would show up as a task that
        # never starts. Fail it here with the reason instead.
        update_task_status(task_id, "failed", error_message=f"Failed to enqueue RQ job: {exc}")
        raise WorkflowError(f"Task row created but enqueue failed: {exc}") from exc
    rq_id = getattr(job, "id", None)
    if rq_id:
        update_task_rq_job_id(task_id, str(rq_id))
    return task_id


# ----------------------------------------------------------------------- task reading


def _task_view(
    task: dict[str, Any], *, with_log: bool = False, with_result: bool = False
) -> dict[str, Any]:
    params = task.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    requester = params.get("_requester") or {}
    view = {
        "id": _text(task.get("id")),
        "type": _text(task.get("type")),
        "status": _text(task.get("status")),
        "description": _text(params.get("description")),
        "target_name": _text(params.get("target_name")),
        "project_id": _text(params.get("project_id")),
        "workflow_kind": _text(params.get("workflow_kind")),
        "output_path": _text(params.get("output_path")),
        "run_name": os.path.basename(_text(params.get("output_path")).rstrip("/")),
        "requested_by": _text(requester.get("name")) if isinstance(requester, dict) else "",
        "progress_message": _text(task.get("progress_message")),
        "progress_pct": task.get("progress_pct"),
        "error_message": _text(task.get("error_message")),
        "result_path": _text(task.get("result_path")),
        "created_at": _iso(task.get("created_at")),
        "updated_at": _iso(task.get("updated_at")),
        "active": _text(task.get("status")) in ("pending", "running"),
    }
    if with_result or with_log:
        summary = task.get("result_summary")
        view["result_summary"] = summary if isinstance(summary, dict) else {}
    if with_log:
        log = _text(task.get("log_output"))
        view["log"] = log[-_MAX_LOG_CHARS:]
        view["log_truncated"] = len(log) > _MAX_LOG_CHARS
    return view


def _reconcile(task: dict[str, Any]) -> None:
    """Best-effort sync of a pending/running row against Redis; never fatal."""
    try:
        from lib.task_queue import reconcile_task_row_in_place

        reconcile_task_row_in_place(task)
    except Exception:
        pass


# ---------------------------------------------------------------------------- routes


def workflow_health(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Advertise the API without authorizing, so a client knows what it can offer.

    Deliberately answers a caller that every other route would refuse: the local app
    uses this to decide whether to show its workflow page at all.
    """
    try:
        require_auth = export_api.require_auth
        require_auth(handler)
        authorized, reason = True, ""
    except export_api.ExportAuthError as exc:
        authorized, reason = False, str(exc)
    queue_enabled = False
    queue_reason = ""
    try:
        from lib.db import is_task_queue_enabled

        queue_enabled = bool(is_task_queue_enabled())
        if not queue_enabled:
            queue_reason = "USE_TASK_QUEUE / DATABASE_URL are not both set on the server."
    except Exception as exc:  # no lib/ (packaged client) or no psycopg2
        queue_reason = f"Task queue support is unavailable here: {exc}"
    # A configured queue with no worker accepts tasks that then sit pending forever,
    # which a preflight cannot distinguish from a slow run. None means "could not tell".
    workers_alive: int | None = None
    workers_reason = ""
    if queue_enabled:
        try:
            from redis import Redis
            from rq import Queue, Worker

            connection = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
            queue = Queue(name=os.environ.get("RQ_QUEUE", "default"), connection=connection)
            workers_alive = int(Worker.count(queue=queue))
            if workers_alive == 0:
                workers_reason = (
                    "No RQ worker is listening on this queue; a started workflow would "
                    "stay pending until one comes up."
                )
        except Exception as exc:
            workers_reason = f"Could not count workers: {exc}"
    target_hint = _text(payload.get("target_name"), "beta/v4.3.2")
    return {
        "ok": True,
        "service": "eval_dashboard_workflow",
        "authorized": authorized,
        "auth_reason": reason,
        "queue_enabled": queue_enabled,
        "queue_reason": queue_reason,
        "workers_alive": workers_alive,
        "workers_reason": workers_reason,
        "kinds": [
            {"name": name, "label": KIND_LABELS[name], "task_type": task_type}
            for name, task_type in KIND_TASK_TYPES.items()
        ],
        "defaults": {
            "project_id": "x2_dev",
            "environment": "",
            "environments": ["", "dev", "stg", "prd"],
            "phase": DEFAULT_PERCEPTION_PHASE,
            "poll_interval": DEFAULT_POLL_INTERVAL,
            "max_wait_hours": DEFAULT_MAX_WAIT_HOURS,
            "output_path": default_output_path(target_hint),
            "metadata_text": default_release_metadata_text(target_hint),
        },
    }


def workflow_catalogs(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Catalog presets from ``catalogs.json``, plus optional live lookups.

    ``refresh`` asks the evaluator API for the project's catalogs and
    ``resolve_catalog_id`` resolves that catalog's newest active integration -- both are
    opt-in because both are slow network calls that can fail on their own.
    """
    export_api.require_auth(handler)
    project_id = _text(payload.get("project_id"))
    environment = _text(payload.get("environment"))
    result: dict[str, Any] = {"presets": _catalog_presets(), "server_catalogs": [], "integration_id": ""}

    if _flag(payload.get("refresh"), False):
        if not project_id:
            raise WorkflowError("refresh needs a project_id.")
        try:
            result["server_catalogs"] = _server_catalogs(project_id, environment)
        except Exception as exc:
            result["server_catalog_error"] = str(exc)

    resolve_id = _text(payload.get("resolve_catalog_id"))
    if resolve_id:
        if not project_id:
            raise WorkflowError("resolve_catalog_id needs a project_id.")
        try:
            result["integration_id"] = _resolve_integration_id(project_id, environment, resolve_id)
        except Exception as exc:
            result["integration_error"] = str(exc)
    return result


def _catalog_presets() -> list[dict[str, Any]]:
    """Read ``catalogs.json`` the way the page does, from the same search paths."""
    import json
    from pathlib import Path

    from backend import app_paths

    filename = "catalogs.json"
    candidates = [
        app_paths.app_root() / filename,
        Path(os.environ.get("CATALOGS_PATH", "")),
        Path.cwd() / filename,
    ]
    for path in candidates:
        if not str(path) or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = data.get("catalogs", []) if isinstance(data, dict) else data
        presets = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            display = item.get("display_name") or item.get("name") or item.get("catalog_id") or "Unknown"
            presets.append({
                "display_name": str(display),
                "catalog_id": _text(item.get("catalog_id")),
                "integration_id": _text(item.get("integration_id")),
                "description": _text(item.get("description")),
                "phase": _text(item.get("phase")),
            })
        return presets
    return []


def _server_catalogs(project_id: str, environment: str) -> list[dict[str, str]]:
    from lib.WebAPI import catalogAPI

    os.environ["AUTH_PROFILE"] = environment or "default"
    response = catalogAPI(project_id=project_id).list_catalogs()
    response.raise_for_status()
    data = response.json()
    raw = data.get("catalogs", []) if isinstance(data, dict) else data
    options: list[dict[str, str]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        catalog_id = _text(item.get("id") or item.get("catalog_id"))
        display = _text(item.get("display_name") or item.get("name")) or catalog_id
        if not catalog_id:
            continue
        options.append({
            "catalog_id": catalog_id,
            "display_name": display,
            "description": _text(item.get("description")),
        })
    options.sort(key=lambda item: item["display_name"].lower())
    return options


def _resolve_integration_id(project_id: str, environment: str, catalog_id: str) -> str:
    import json

    from lib import evaluator_api

    os.environ["AUTH_PROFILE"] = environment or "default"
    api = evaluator_api.EvaluationRunAPI()
    url = f"{api.api_base_url}/projects/{project_id}/integrations"
    response = api.request(url, {"catalog_id": catalog_id, "size": 100}, method="GET")
    if response is None:
        raise WorkflowError("No response returned while loading integrations.")
    if response.status_code != 200:
        raise WorkflowError(f"Failed to load integrations: status={response.status_code}")
    integrations = (json.loads(response.content) or {}).get("integrations") or []
    active = [
        item for item in integrations
        if isinstance(item, dict)
        and _text(item.get("catalog_id")) == catalog_id
        and not bool(item.get("deleted"))
    ]
    if not active:
        raise WorkflowError("No active integration was found for that catalog.")
    active.sort(
        key=lambda item: (_text(item.get("updated_at")), int(item.get("version_id") or 0), _text(item.get("id"))),
        reverse=True,
    )
    return _text(active[0].get("id"))


def _default_target_repo_url() -> str:
    try:
        from lib.local_evaluator_debug import DEFAULT_REPO_URL

        return DEFAULT_REPO_URL
    except Exception:  # packaged client ships no lib/; mirror its default
        return os.environ.get("LOCAL_EVALUATOR_REPO_URL", "git@github.com:tier4/pilot-auto.x2.git")


def check_target_ref(target_name: str, *, is_tag: bool) -> dict[str, Any]:
    """Best-effort ``git ls-remote`` existence check for the branch/tag a run would build.

    A typo'd branch is otherwise the most expensive mistake this API allows: the evaluator
    accepts the job and fails it during build, hours later. ``exists`` is only meaningful
    when ``checked`` is true; a server without git credentials stays inconclusive rather
    than blocking anyone.
    """
    import subprocess

    repo = os.environ.get("WORKFLOW_TARGET_REPO_URL") or _default_target_repo_url()
    ref = f"refs/tags/{target_name}" if is_tag else f"refs/heads/{target_name}"
    result: dict[str, Any] = {"checked": False, "exists": None, "ref": ref, "repo": repo}
    try:
        proc = subprocess.run(
            ["git", "ls-remote", "--exit-code", repo, ref],
            capture_output=True, text=True, timeout=20,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except Exception as exc:  # no git binary, timeout
        result["detail"] = f"Could not run git ls-remote: {exc}"
        return result
    if proc.returncode == 0:
        result.update(checked=True, exists=True)
    elif proc.returncode == 2:  # connected and listed refs; this one is not there
        result.update(checked=True, exists=False,
                      detail=f"{ref} does not exist in {repo}.")
    else:  # auth/network failure: inconclusive, never blocking
        result["detail"] = (proc.stderr or proc.stdout).strip()[-500:]
    return result


def workflow_start(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a request, queue it, and return the task id."""
    who = export_api.require_auth(handler)
    kind, task_type, params = build_params(payload)
    target_check: dict[str, Any] = {}
    if _flag(payload.get("check_target"), True):
        target_check = check_target_ref(
            params["target_name"], is_tag=bool(params.get("is_tag"))
        )
    if _flag(payload.get("dry_run"), False):
        # Lets a script (or the UI's "Check" button) see the exact parameters the worker
        # would get without putting anything on the queue.
        return {"ok": True, "dry_run": True, "kind": kind, "task_type": task_type,
                "parameters": params, "target_check": target_check}
    if target_check.get("checked") and target_check.get("exists") is False:
        # Definitive miss: the evaluator would accept this job and fail it at build time.
        # An inconclusive check (no credentials, network) never blocks a start.
        raise WorkflowError(
            f"Target '{params['target_name']}' was not found "
            f"({target_check['ref']} in {target_check['repo']}). "
            "Send check_target: false to start anyway."
        )
    task_id = enqueue(task_type, params, who)
    return {
        "ok": True,
        "task_id": task_id,
        "target_check": target_check,
        "kind": kind,
        "task_type": task_type,
        "output_path": _text(params.get("output_path")),
        "run_name": os.path.basename(_text(params.get("output_path")).rstrip("/")),
        "description": _text(params.get("description")),
        "queued_by": _text(who.get("actor")),
    }


def workflow_tasks(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Recent workflow tasks, newest first, reconciled against the queue."""
    export_api.require_auth(handler)
    from lib.db import list_recent_tasks

    limit = int(_number(payload.get("limit", 25), 25, minimum=1, maximum=200))
    since_days_raw = payload.get("since_days", 7)
    since_days = None if since_days_raw in (None, "", 0, "0") else int(
        _number(since_days_raw, 7, minimum=1, maximum=3650)
    )
    mine = _text(payload.get("mine"))
    # Over-fetch because the filter below drops non-workflow types.
    rows = list_recent_tasks(
        limit=min(200, limit * 4), since_days=since_days,
        session_id=mine or None, include_details=False,
    )
    items = []
    for row in rows:
        if _text(row.get("type")) not in REPORTED_TASK_TYPES:
            continue
        _reconcile(row)
        items.append(_task_view(row))
        if len(items) >= limit:
            break
    return {"items": items, "types": list(REPORTED_TASK_TYPES)}


def workflow_task(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """One task with its result summary, for tailing a run.

    The log (up to 200 KB) is included by default; a status poller that only wants the
    structured result can send ``log: false``.
    """
    export_api.require_auth(handler)
    from lib.db import get_task

    task_id = _text(payload.get("task_id"))
    if not task_id:
        raise WorkflowError("A task_id is required.")
    task = get_task(task_id)
    if not task:
        raise WorkflowError(f"No such task: {task_id}")
    if _text(task.get("type")) not in REPORTED_TASK_TYPES:
        raise WorkflowError(f"Task {task_id} is not a workflow task.")
    _reconcile(task)
    return {"task": _task_view(task, with_log=_flag(payload.get("log"), True), with_result=True)}


def workflow_cancel(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Stop a queued or running workflow. The task row is kept, marked failed."""
    who = export_api.require_auth(handler)
    from lib.db import append_task_log, get_task, update_task_status
    from lib.task_queue import try_cancel_rq_job

    task_id = _text(payload.get("task_id"))
    if not task_id:
        raise WorkflowError("A task_id is required.")
    task = get_task(task_id)
    if not task:
        raise WorkflowError(f"No such task: {task_id}")
    if _text(task.get("type")) not in REPORTED_TASK_TYPES:
        raise WorkflowError(f"Task {task_id} is not a workflow task.")
    if _text(task.get("status")) not in ("pending", "running"):
        return {"ok": False, "message": f"Task is already {_text(task.get('status'))}.",
                "task": _task_view(task)}

    cancelled = try_cancel_rq_job(_text(task.get("rq_job_id")))
    actor = _text(who.get("actor"), "an api client")
    message = f"Cancelled by {actor} through the workflow API."
    update_task_status(task_id, "failed", error_message=message)
    append_task_log(task_id, message)
    refreshed = get_task(task_id) or task
    return {
        "ok": True,
        "queue_cancelled": bool(cancelled),
        "message": message if cancelled else message + " The queue job could not be reached.",
        "task": _task_view(refreshed),
    }


# ------------------------------------------------------------------------ trend data

# The Trend Insights page reads metadata from roles in this order; the API must agree
# so both report the same version/date for a group (pages/13_Trend_Insights.py).
_TREND_PRIMARY_ROLES = ("full", "usecase", "devops", "performance_blocks", "unknown")


def _trend_primary_metadata(jobs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for role in _TREND_PRIMARY_ROLES:
        job = jobs.get(role)
        if job and isinstance(job.get("metadata"), dict):
            return job["metadata"]
    return {}


def _trend_group_metrics(jobs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The per-release numbers the Trend Insights charts plot, from the same extractors.

    A summary an extractor cannot digest (pre-0.2.0 layouts, hand-edited files) fails
    per-role into ``errors`` instead of failing the whole listing: one broken release
    on disk must not make history unqueryable.
    """
    from lib.specsheet_report import (
        banded_recall_percent_from_summary,
        extract_devops_case_rows,
        extract_performance_metrics_from_summary,
        extract_usecase_metrics_from_summary,
    )

    metrics: dict[str, Any] = {}
    errors: dict[str, str] = {}

    if "full" in jobs:
        try:
            metrics.update(extract_performance_metrics_from_summary(jobs["full"]["summary"]))
        except Exception as exc:
            errors["full"] = str(exc)

    if "usecase" in jobs:
        try:
            usecase = extract_usecase_metrics_from_summary(jobs["usecase"]["summary"])
            metrics.update({f"usecase_{key}": value for key, value in usecase.items()})
        except Exception as exc:
            errors["usecase"] = str(exc)

    if "devops" in jobs:
        devops_job = jobs["devops"]
        try:
            case_rows = extract_devops_case_rows(
                devops_job.get("devops_summary") or devops_job["summary"]
            )
            total_passed = sum(int(row["passed"]) for row in case_rows)
            total_count = sum(int(row["total"]) for row in case_rows)
            metrics["scenario_count"] = total_count or None
            metrics["overall_pass_rate"] = (
                total_passed / total_count * 100.0 if total_count > 0 else None
            )
        except Exception as exc:
            errors["devops"] = str(exc)
        try:
            banded = banded_recall_percent_from_summary(devops_job.get("summary"))
            if banded:
                metrics["recall_bands"] = banded
        except Exception as exc:
            errors["devops_recall_bands"] = str(exc)

    if errors:
        metrics["errors"] = errors
    return metrics


def _trend_group_view(
    group: Any,
    *,
    with_metrics: bool,
    with_summary: bool,
    with_cases: bool,
) -> dict[str, Any]:
    metadata = _trend_primary_metadata(group.jobs)
    jobs: dict[str, dict[str, Any]] = {}
    for role, job in group.jobs.items():
        entry: dict[str, Any] = {
            "job_id": _text(job.get("job_id")),
            "metadata": job.get("metadata") if isinstance(job.get("metadata"), dict) else {},
            "metadata_path": str(job.get("metadata_path") or ""),
            "summary_path": str(job.get("summary_path") or ""),
        }
        if with_summary:
            entry["summary"] = job.get("summary")
        jobs[str(role)] = entry

    view: dict[str, Any] = {
        "group_key": group.group_key,
        "release": group.display_name,
        "topic": group.topic_name,
        "group_kind": group.group_kind,
        "release_group": _text(metadata.get("release_group")),
        "version": _text(metadata.get("pilot_auto_version_abbr"))
        or _text(metadata.get("version_abbr"))
        or _text(metadata.get("pilot_auto_version")),
        "date": _text(metadata.get("date")),
        "description": _text(metadata.get("description")),
        "data_count": _text(metadata.get("data_count")),
        "roles": sorted(group.jobs),
        "base_dir": str(group.base_dir),
        "jobs": jobs,
    }
    if with_metrics:
        view["metrics"] = _trend_group_metrics(group.jobs)
    if with_cases and "devops" in group.jobs:
        from lib.specsheet_report import extract_devops_case_rows

        devops_job = group.jobs["devops"]
        try:
            view["cases"] = extract_devops_case_rows(
                devops_job.get("devops_summary") or devops_job["summary"]
            )
        except Exception as exc:
            view["cases"] = []
            view["cases_error"] = str(exc)
    return view


def _trend_group_matches(view: dict[str, Any], query: str) -> bool:
    haystack = " ".join(
        str(view.get(field) or "")
        for field in ("release", "release_group", "version", "date", "description", "topic")
    ).lower()
    return query in haystack


def workflow_trends(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Past release/trend groups from the data root, newest first.

    The same inventory the Trend Insights page charts -- trend metadata plus the
    summary-derived metrics per release -- so a script can pull history with one call.
    Raw summaries and per-case pass rates are opt-in (``include_summary``,
    ``include_cases``): they dominate the payload and most callers only trend the
    headline numbers.
    """
    export_api.require_auth(handler)
    from lib.specsheet_report import discover_trend_release_groups

    topic = _text(payload.get("topic")).lower()
    query = _text(payload.get("q")).lower()
    limit = int(_number(payload.get("limit", 100), 100, minimum=1, maximum=1000))
    with_metrics = _flag(payload.get("metrics"), True)
    with_summary = _flag(payload.get("include_summary"), False)
    with_cases = _flag(payload.get("include_cases"), False)

    groups = discover_trend_release_groups()
    items: list[dict[str, Any]] = []
    # discover() sorts oldest-first for the charts; a caller asking for "recent trend
    # data" wants the newest releases before the limit cuts off.
    for group in reversed(groups):
        if topic and str(group.topic_name).lower() != topic:
            continue
        view = _trend_group_view(
            group, with_metrics=with_metrics, with_summary=with_summary, with_cases=with_cases
        )
        if query and not _trend_group_matches(view, query):
            continue
        items.append(view)
        if len(items) >= limit:
            break
    return {"items": items, "total_groups": len(groups)}


JSON_ROUTES: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "/api/workflow_health": workflow_health,
    "/api/workflow_catalogs": workflow_catalogs,
    "/api/workflow_start": workflow_start,
    "/api/workflow_tasks": workflow_tasks,
    "/api/workflow_task": workflow_task,
    "/api/workflow_cancel": workflow_cancel,
    "/api/workflow_trends": workflow_trends,
}
