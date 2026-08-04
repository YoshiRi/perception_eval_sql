"""Client-only HTTP routes backing the in-app run browser.

These exist so the packaged app is self-sufficient: without them, acquiring data means
dropping to a terminal, which defeats the point of a double-clickable app.

They are registered on :class:`ClientHandler`, never on the server's
``LocalBBoxHandler``, and the handler deliberately drops the export routes -- a process
listening on a laptop has no business serving files to anyone.

No authentication: the server binds loopback only, and the routes expose nothing the
user running the app cannot already reach. The stored token is never echoed back.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from client import config, sync
from client.remote import connect, probe_server


class PullJob:
    """A single in-flight pull. One at a time keeps bandwidth and disk predictable."""

    def __init__(self, run: str, role: str, tier: str, include_future: bool) -> None:
        self.run = run
        self.role = role
        self.tier = tier
        self.include_future = include_future
        self.state = "starting"  # starting | planning | downloading | done | failed | cancelled
        self.message = ""
        self.error = ""
        self.progress: dict[str, Any] = {}
        self.result: dict[str, Any] = {}
        self.started_at = time.time()
        self.finished_at: float | None = None
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "role": self.role,
            "tier": self.tier,
            "state": self.state,
            "message": self.message,
            "error": self.error,
            "progress": self.progress,
            "result": self.result,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "active": self.state in ("starting", "planning", "downloading"),
        }


_JOB_LOCK = threading.Lock()
_JOB: PullJob | None = None


def _current_job() -> PullJob | None:
    with _JOB_LOCK:
        return _JOB


def _run_pull(job: PullJob) -> None:
    try:
        remote = connect(config.Config.load())
        job.state = "planning"
        job.message = "Asking the server what this run contains..."
        manifest = remote.manifest(
            job.run,
            role=job.role,
            tier=job.tier,
            include_future=job.include_future,
            checksums=True,
        )
        plan = sync.build_plan(manifest)
        job.progress = {
            "done_bytes": 0,
            "total_bytes": plan.download_bytes,
            "done_files": 0,
            "total_files": len(plan.download),
            "percent": 0.0,
        }
        if not plan.download:
            job.state = "done"
            job.message = "Already up to date."
            job.result = {"downloaded": 0, "kept": len(plan.keep), "bytes": 0}
            job.finished_at = time.time()
            return

        job.state = "downloading"
        job.message = f"Downloading {len(plan.download)} file(s), {sync.human_bytes(plan.download_bytes)}"
        result = sync.execute(
            remote,
            manifest,
            plan,
            on_progress=lambda snap: setattr(job, "progress", snap),
            should_stop=lambda: job.cancelled,
            on_error=lambda rel, exc: None,
        )
        job.result = result
        job.finished_at = time.time()
        if result.get("cancelled"):
            job.state = "cancelled"
            job.message = "Cancelled. Partial files are kept; pulling again resumes."
        elif result.get("failed"):
            job.state = "failed"
            job.error = f"{len(result['failed'])} file(s) failed: " + ", ".join(result["failed"][:3])
            job.message = "Pull again to retry the failures."
        else:
            job.state = "done"
            job.message = (
                f"Downloaded {result['downloaded']} file(s), {sync.human_bytes(result['bytes'])}."
            )
    except Exception as exc:
        job.state = "failed"
        job.error = str(exc)
        job.message = "Pull failed."
        job.finished_at = time.time()


# ------------------------------------------------------------------------- routes


def _config_view() -> dict[str, Any]:
    cfg = config.Config.load()
    return {
        "server_url": cfg.effective_server(),
        "server_source": cfg.server_source(),
        "can_reset": bool(cfg.server_url),
        # Never return the token itself; the UI only needs to know whether one is set.
        "token_set": bool(cfg.resolved_token()),
        "cf_configured": bool(cfg.cf_client_id and cfg.cf_client_secret),
        "verify_tls": cfg.verify_tls,
        "t4_base_url": cfg.effective_t4_base_url(),
        "workspace": str(config.workspace_dir()),
    }


def _server_status() -> dict[str, Any]:
    """Whether the configured server will actually serve this user, without a token.

    The export API authorizes on the dashboard's own identity model, so in most
    deployments no token is needed and the UI should not ask for one. Probing here lets
    the page say which case it is in rather than making the user guess.
    """
    cfg = config.Config.load()
    blank = {"reachable": False, "authorized": False, "token_required": False,
             "identity": "", "origin": "", "reason": ""}
    if not cfg.effective_server():
        return {**blank, "reason": "no server configured"}
    try:
        health = connect(cfg).export_health()
    except Exception as exc:
        return {**blank, "reason": str(exc)[:200]}
    return {
        "reachable": True,
        "authorized": bool(health.get("authorized")),
        "token_required": bool(health.get("token_required")),
        "identity": health.get("identity") or "",
        "origin": health.get("origin") or "",
        "reason": health.get("auth_reason") or "",
    }


def client_state(payload: dict[str, Any]) -> dict[str, Any]:
    job = _current_job()
    return {
        "config": _config_view(),
        "server": _server_status() if payload.get("probe") is not False else None,
        "local_runs": sync.local_run_summary(),
        "job": job.snapshot() if job else None,
        "tiers": [
            {"name": "minimal", "label": "Minimal",
             "hint": "Parquet + metadata. Explorer, viewer and statistics all work."},
            {"name": "criteria", "label": "Criteria (recommended)",
             "hint": "Adds scenario YAML and pre-baked DevOps gate/frame verdicts."},
            {"name": "full", "label": "Full",
             "hint": "Adds pre-baked true-negative objects for the preview overlay."},
            {"name": "raw", "label": "Raw",
             "hint": "Adds scene_result.pkl. Gigabytes; only for recomputing from source."},
        ],
    }


def client_login(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = config.Config.load()
    server = str(payload.get("server_url") or "").strip() or cfg.effective_server()
    if not server:
        raise ValueError("A server URL is required.")
    cfg.server_url = server.rstrip("/")
    token = payload.get("token")
    if token:  # blank means "keep the stored one"
        cfg.token = str(token).strip()
    if payload.get("cf_client_id") is not None:
        cfg.cf_client_id = str(payload.get("cf_client_id") or "").strip()
    if payload.get("cf_client_secret") is not None:
        cfg.cf_client_secret = str(payload.get("cf_client_secret") or "").strip()
    if payload.get("t4_base_url") is not None:
        cfg.t4_base_url = str(payload.get("t4_base_url") or "").strip()
    if payload.get("verify_tls") is not None:
        cfg.verify_tls = bool(payload.get("verify_tls"))

    resolved, health = probe_server(cfg, cfg.server_url)
    cfg.server_url = resolved
    cfg.save()
    return {"ok": True, "config": _config_view(), "health": health, "server": _server_status()}


def client_reset_server(payload: dict[str, Any]) -> dict[str, Any]:
    """Forget the saved server/token so the build default or $EVALDASH_SERVER applies."""
    cfg = config.Config.load()
    fallback = cfg.reset_server()
    return {
        "ok": True,
        "server_url": fallback,
        "message": f"Reset to {fallback}" if fallback else "Cleared; no server configured.",
        "config": _config_view(),
        "server": _server_status() if fallback else None,
    }


def client_remote_runs(payload: dict[str, Any]) -> dict[str, Any]:
    remote = connect(config.Config.load())
    data = remote.runs(sizes=payload.get("sizes", True) is not False, query=str(payload.get("q") or ""))
    return {"server": remote.base_url, **data}


def client_pull(payload: dict[str, Any]) -> dict[str, Any]:
    global _JOB
    run = str(payload.get("run") or "").strip()
    if not run:
        raise ValueError("A run name is required.")
    with _JOB_LOCK:
        if _JOB is not None and _JOB.snapshot()["active"]:
            raise ValueError(f"A pull is already running ({_JOB.run}). Cancel it first.")
        job = PullJob(
            run=run,
            role=str(payload.get("role") or "all"),
            tier=str(payload.get("tier") or "criteria"),
            include_future=payload.get("include_future") is True,
        )
        _JOB = job
    threading.Thread(target=_run_pull, args=(job,), name=f"pull-{run}", daemon=True).start()
    return {"ok": True, "job": job.snapshot()}


def client_pull_status(payload: dict[str, Any]) -> dict[str, Any]:
    job = _current_job()
    return {"job": job.snapshot() if job else None}


def client_pull_cancel(payload: dict[str, Any]) -> dict[str, Any]:
    job = _current_job()
    if job is None or not job.snapshot()["active"]:
        return {"ok": False, "message": "No pull is running."}
    job.cancel()
    return {"ok": True, "message": "Cancelling after the current file."}


def client_delete_run(payload: dict[str, Any]) -> dict[str, Any]:
    run = str(payload.get("run") or "").strip()
    if not run:
        raise ValueError("A run name is required.")
    job = _current_job()
    if job is not None and job.snapshot()["active"] and job.run == run:
        raise ValueError("That run is being downloaded right now. Cancel the pull first.")
    ok, message = sync.remove_run(run)
    if not ok:
        raise ValueError(message)
    return {"ok": True, "message": message, "local_runs": sync.local_run_summary()}


def client_parquets(payload: dict[str, Any]) -> dict[str, Any]:
    """Bbox-viewable parquets in the workspace, so the UI can deep-link the explorer."""
    from backend import local_bbox_api as api

    return api.list_parquets({"bbox_only": True, "limit": 500})


# --------------------------------------------------------------------- 3D point clouds


class T4FetchJob:
    """One in-flight scene fetch. Separate from PullJob: different host, different disk
    budget, and caching a scene while a run downloads is a legitimate combination."""

    def __init__(self, dataset_id: str, scenario: str, frames_label: str) -> None:
        self.dataset_id = dataset_id
        self.scenario = scenario
        self.frames_label = frames_label
        self.state = "starting"  # starting | fetching | done | failed | cancelled
        self.message = ""
        self.error = ""
        self.progress: dict[str, Any] = {}
        self.result: dict[str, Any] = {}
        self.started_at = time.time()
        self.finished_at: float | None = None
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "scenario": self.scenario,
            "frames": self.frames_label,
            "state": self.state,
            "message": self.message,
            "error": self.error,
            "progress": self.progress,
            "result": self.result,
            "active": self.state in ("starting", "fetching"),
        }


_T4_JOB_LOCK = threading.Lock()
_T4_JOB: T4FetchJob | None = None


def _current_t4_job() -> T4FetchJob | None:
    with _T4_JOB_LOCK:
        return _T4_JOB


def parse_frames_spec(text: str) -> range | None:
    """``""`` -> everything, ``"7"`` -> frame 7, ``"0-49"`` -> that span (inclusive).

    The same grammar the CLI's --frames accepts, so knowledge transfers between the two.
    """
    spec = str(text or "").strip()
    if not spec:
        return None
    low, _, high = spec.partition("-")
    try:
        start = int(low)
        end = int(high) if high else start
    except ValueError:
        raise ValueError(f"Frames must look like 12 or 0-49, got {spec!r}.")
    if start < 0 or end < start:
        raise ValueError(f"Frames must be an ascending range, got {spec!r}.")
    return range(start, end + 1)


def normalize_scenarios(payload: Any) -> list[dict[str, Any]]:
    """Flatten t4-server's scenario listing into ``[{name, frames}]``.

    The service has answered with both a bare list and a ``{"scenarios": [...]}``
    wrapper across versions, and items as strings or dicts; accept all of them rather
    than binding the UI to one deployment's vintage.
    """
    items = payload.get("scenarios") if isinstance(payload, dict) else payload
    out: list[dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, str):
            name = item.strip()
            frames = None
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("scenario_name") or "").strip()
            raw = item.get("nbr_samples") or item.get("frames") or item.get("frame_count")
            try:
                frames = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                frames = None
        else:
            continue
        if name:
            out.append({"name": name, "frames": frames})
    return out


def _t4_config_view() -> dict[str, Any]:
    from client import t4

    cfg = config.Config.load()
    effective = cfg.effective_t4_base_url()
    return {
        "t4_base_url": effective,
        # The env var shadows whatever this page saves, so say so instead of letting a
        # save appear to have no effect.
        "env_override": bool(os.environ.get("EVALDASH_T4_BASE_URL", "").strip()),
        "cache_root": str(t4.t4_root()),
    }


def _t4_probe(base_url: str) -> dict[str, Any]:
    from client import t4

    if not base_url:
        return {"reachable": False, "reason": "no T4 server configured"}
    try:
        t4.T4Client(base_url, config.Config.load(), timeout=6.0).health()
        return {"reachable": True, "reason": ""}
    except Exception as exc:
        return {"reachable": False, "reason": str(exc)[:200]}


def client_t4_state(payload: dict[str, Any]) -> dict[str, Any]:
    from client import t4

    view = _t4_config_view()
    job = _current_t4_job()
    return {
        "config": view,
        "server": _t4_probe(view["t4_base_url"]) if payload.get("probe") is True else None,
        "scenes": t4.cached_scenes(),
        "job": job.snapshot() if job else None,
    }


def client_t4_config(payload: dict[str, Any]) -> dict[str, Any]:
    url = str(payload.get("t4_base_url") or "").strip().rstrip("/")
    cfg = config.Config.load()
    cfg.t4_base_url = url
    cfg.save()
    view = _t4_config_view()
    return {"ok": True, "config": view, "server": _t4_probe(view["t4_base_url"])}


def client_t4_scenarios(payload: dict[str, Any]) -> dict[str, Any]:
    from client import t4

    dataset_id = str(payload.get("dataset_id") or "").strip()
    if not dataset_id:
        raise ValueError("A dataset id is required.")
    cfg = config.Config.load()
    client = t4.T4Client(cfg.effective_t4_base_url(), cfg, timeout=30.0)
    return {"dataset_id": dataset_id, "scenarios": normalize_scenarios(client.scenarios(dataset_id))}


def client_t4_estimate(payload: dict[str, Any]) -> dict[str, Any]:
    from client import t4

    dataset_id = str(payload.get("dataset_id") or "").strip()
    scenario = str(payload.get("scenario") or "").strip()
    if not dataset_id or not scenario:
        raise ValueError("A dataset id and scenario are required.")
    cfg = config.Config.load()
    estimate = t4.estimate_scene_bytes(
        dataset_id, scenario, base_url=cfg.effective_t4_base_url()
    )
    return {"dataset_id": dataset_id, "scenario": scenario, **estimate}


def _run_t4_fetch(job: T4FetchJob, frames: range | None, with_camera: bool, with_lanelet: bool, force: bool) -> None:
    from client import t4

    try:
        cfg = config.Config.load()
        job.state = "fetching"
        job.message = "Downloading frames..."
        stats = t4.fetch_scene(
            job.dataset_id,
            job.scenario,
            base_url=cfg.effective_t4_base_url(),
            frames=frames,
            with_camera=with_camera,
            with_lanelet=with_lanelet,
            force=force,
            progress=lambda snap: setattr(job, "progress", snap),
            should_stop=lambda: job.cancelled,
        )
        job.result = stats
        job.finished_at = time.time()
        fetched = int(stats.get("frames_fetched") or 0)
        skipped = int(stats.get("frames_skipped") or 0)
        errors = stats.get("errors") or []
        if stats.get("stopped_early"):
            job.state = "cancelled"
            job.message = "Cancelled. Cached frames are kept; fetching again resumes."
        elif errors:
            job.state = "failed"
            job.error = f"{len(errors)} request(s) failed; fetching again retries just those."
            job.message = f"Fetched {fetched}, skipped {skipped} already cached."
        else:
            job.state = "done"
            job.message = (
                f"Fetched {fetched} frame(s)"
                + (f", skipped {skipped} already cached" if skipped else "")
                + f" ({sync.human_bytes(stats.get('bytes') or 0)} on disk)."
            )
    except Exception as exc:
        job.state = "failed"
        job.error = str(exc)
        job.message = "Fetch failed."
        job.finished_at = time.time()


def client_t4_fetch(payload: dict[str, Any]) -> dict[str, Any]:
    global _T4_JOB
    dataset_id = str(payload.get("dataset_id") or "").strip()
    scenario = str(payload.get("scenario") or "").strip()
    if not dataset_id or not scenario:
        raise ValueError("A dataset id and scenario are required.")
    frames_label = str(payload.get("frames") or "").strip()
    frames = parse_frames_spec(frames_label)
    with _T4_JOB_LOCK:
        if _T4_JOB is not None and _T4_JOB.snapshot()["active"]:
            raise ValueError(
                f"A scene fetch is already running ({_T4_JOB.scenario}). Cancel it first."
            )
        job = T4FetchJob(dataset_id, scenario, frames_label)
        _T4_JOB = job
    threading.Thread(
        target=_run_t4_fetch,
        args=(
            job,
            frames,
            payload.get("with_camera") is not False,
            payload.get("with_lanelet") is not False,
            payload.get("force") is True,
        ),
        name=f"t4-fetch-{scenario}",
        daemon=True,
    ).start()
    return {"ok": True, "job": job.snapshot()}


def client_t4_fetch_status(payload: dict[str, Any]) -> dict[str, Any]:
    job = _current_t4_job()
    return {"job": job.snapshot() if job else None}


def client_t4_fetch_cancel(payload: dict[str, Any]) -> dict[str, Any]:
    job = _current_t4_job()
    if job is None or not job.snapshot()["active"]:
        return {"ok": False, "message": "No scene fetch is running."}
    job.cancel()
    return {"ok": True, "message": "Cancelling after the current frame."}


def client_t4_delete(payload: dict[str, Any]) -> dict[str, Any]:
    from client import t4

    dataset_id = str(payload.get("dataset_id") or "").strip()
    scenario = str(payload.get("scenario") or "").strip()
    if not dataset_id or not scenario:
        raise ValueError("A dataset id and scenario are required.")
    job = _current_t4_job()
    if job is not None and job.snapshot()["active"] and (job.dataset_id, job.scenario) == (dataset_id, scenario):
        raise ValueError("That scene is being fetched right now. Cancel the fetch first.")
    ok, message = t4.remove_scene(dataset_id, scenario)
    if not ok:
        raise ValueError(message)
    return {"ok": True, "message": message, "scenes": t4.cached_scenes()}


def _forward(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call one of the server's workflow routes with the stored credentials.

    Thin on purpose: the page is a client of the server's API, and duplicating
    validation here would only produce a second set of rules to keep in sync. Errors
    come back as the message the server chose.
    """
    remote = connect(config.Config.load())
    return getattr(remote, method)(**payload)


def client_workflow_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Everything the workflow page needs for a first paint."""
    cfg = config.Config.load()
    if not cfg.effective_server():
        return {"server": "", "health": None, "reason": "No server configured yet."}
    try:
        remote = connect(cfg)
        health = remote.workflow_health(target_name=str(payload.get("target_name") or ""))
    except Exception as exc:
        return {"server": cfg.effective_server(), "health": None, "reason": str(exc)[:300]}
    result: dict[str, Any] = {"server": remote.base_url, "health": health, "reason": ""}
    if health.get("authorized"):
        try:
            result["presets"] = (remote.workflow_catalogs()).get("presets") or []
        except Exception as exc:
            result["presets"] = []
            result["presets_error"] = str(exc)[:300]
    return result


def client_workflow_catalogs(payload: dict[str, Any]) -> dict[str, Any]:
    return _forward("workflow_catalogs", {
        "project_id": str(payload.get("project_id") or ""),
        "environment": str(payload.get("environment") or ""),
        "refresh": payload.get("refresh") is True,
        "resolve_catalog_id": str(payload.get("resolve_catalog_id") or ""),
    })


def client_workflow_start(payload: dict[str, Any]) -> dict[str, Any]:
    params = {k: v for k, v in payload.items() if not str(k).startswith("_")}
    return _forward("workflow_start", {"params": params})


def client_workflow_tasks(payload: dict[str, Any]) -> dict[str, Any]:
    since = payload.get("since_days", 7)
    return _forward("workflow_tasks", {
        "limit": int(payload.get("limit") or 25),
        "since_days": None if since in (None, "", 0, "0") else int(since),
        "mine": str(payload.get("mine") or ""),
    })


def client_workflow_task(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("A task_id is required.")
    return _forward("workflow_task", {"task_id": task_id})


def client_workflow_cancel(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("A task_id is required.")
    return _forward("workflow_cancel", {"task_id": task_id})


CLIENT_ROUTES = {
    "/api/client/state": client_state,
    "/api/client/login": client_login,
    "/api/client/reset_server": client_reset_server,
    "/api/client/remote_runs": client_remote_runs,
    "/api/client/pull": client_pull,
    "/api/client/pull_status": client_pull_status,
    "/api/client/pull_cancel": client_pull_cancel,
    "/api/client/delete_run": client_delete_run,
    "/api/client/parquets": client_parquets,
    "/api/client/workflow_state": client_workflow_state,
    "/api/client/workflow_catalogs": client_workflow_catalogs,
    "/api/client/workflow_start": client_workflow_start,
    "/api/client/workflow_tasks": client_workflow_tasks,
    "/api/client/workflow_task": client_workflow_task,
    "/api/client/workflow_cancel": client_workflow_cancel,
    "/api/client/t4_state": client_t4_state,
    "/api/client/t4_config": client_t4_config,
    "/api/client/t4_scenarios": client_t4_scenarios,
    "/api/client/t4_estimate": client_t4_estimate,
    "/api/client/t4_fetch": client_t4_fetch,
    "/api/client/t4_fetch_status": client_t4_fetch_status,
    "/api/client/t4_fetch_cancel": client_t4_fetch_cancel,
    "/api/client/t4_delete": client_t4_delete,
}


def build_handler() -> type:
    """Create the client's request handler.

    Built lazily so importing this module does not pull in DuckDB, and so the base
    class is imported only after :func:`config.apply_server_env` has run.
    """
    from backend import app_paths
    from backend.local_bbox_api import (
        LocalBBoxHandler,
        _html_response,
        _json_response,
        _render_page_html,
        _send_response,
    )

    def page_html(name: str) -> str:
        if app_paths.find_static_file(name) is None:
            raise FileNotFoundError(f"{name} is missing from this build")
        return _render_page_html(name, "")

    class ClientHandler(LocalBBoxHandler):
        # Base viewer/explorer routes plus the client-only ones.
        routes = {**LocalBBoxHandler.routes, **CLIENT_ROUTES}
        # The client is not an export server. Dropping these means the routes 404
        # instead of relying on the token merely being unset.
        auth_routes: dict = {}
        stream_routes: dict = {}

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            page = {
                "/": "client_home.html", "/home": "client_home.html", "/home/": "client_home.html",
                "/workflow": "client_workflow.html", "/workflow/": "client_workflow.html",
            }.get(parsed.path)
            if page:
                try:
                    _html_response(self, 200, page_html(page))
                except Exception as exc:
                    _json_response(self, 500, {"error": str(exc)})
                return
            # Must precede the base class, which owns "/viewer" for the bbox viewer.
            if parsed.path == "/viewer/three" or parsed.path.startswith("/viewer/three/"):
                self._serve_t4(parsed)
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            if parsed.path.startswith("/viewer/three/"):
                # The page POSTs camera overlays when it has external boxes to project.
                # Offline that has to fall back to the cached GET variant.
                self._serve_t4(parsed)
                return
            super().do_POST()

        def _serve_t4(self, parsed: Any) -> None:
            from client import t4

            try:
                body, content_type, extra = t4.serve_request(parsed.path, parse_qs(parsed.query))
            except t4.CacheMiss as exc:
                _json_response(self, 404, {"error": str(exc), "offline_cache_miss": True})
                return
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
                return
            headers = [
                ("Content-Type", content_type),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-cache"),
            ]
            headers.extend(extra.items())
            _send_response(self, 200, headers, body)

    return ClientHandler
