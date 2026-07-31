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

import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from client import config, sync
from client.remote import Remote, probe_server


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
        remote = Remote(config.Config.load())
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
        "server_url": cfg.server_url,
        # Never return the token itself; the UI only needs to know whether one is set.
        "token_set": bool(cfg.resolved_token()),
        "cf_configured": bool(cfg.cf_client_id and cfg.cf_client_secret),
        "verify_tls": cfg.verify_tls,
        "t4_base_url": cfg.t4_base_url,
        "workspace": str(config.workspace_dir()),
    }


def client_state(payload: dict[str, Any]) -> dict[str, Any]:
    job = _current_job()
    return {
        "config": _config_view(),
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
    server = str(payload.get("server_url") or "").strip()
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
    return {"ok": True, "config": _config_view(), "health": health}


def client_remote_runs(payload: dict[str, Any]) -> dict[str, Any]:
    remote = Remote(config.Config.load())
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


CLIENT_ROUTES = {
    "/api/client/state": client_state,
    "/api/client/login": client_login,
    "/api/client/remote_runs": client_remote_runs,
    "/api/client/pull": client_pull,
    "/api/client/pull_status": client_pull_status,
    "/api/client/pull_cancel": client_pull_cancel,
    "/api/client/delete_run": client_delete_run,
    "/api/client/parquets": client_parquets,
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
    )

    def home_html() -> str:
        page = app_paths.find_static_file("client_home.html")
        if page is None:
            raise FileNotFoundError("client_home.html is missing from this build")
        return _render_page_html("client_home.html", "")

    class ClientHandler(LocalBBoxHandler):
        # Base viewer/explorer routes plus the client-only ones.
        routes = {**LocalBBoxHandler.routes, **CLIENT_ROUTES}
        # The client is not an export server. Dropping these means the routes 404
        # instead of relying on the token merely being unset.
        auth_routes: dict = {}
        stream_routes: dict = {}

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/home", "/home/"):
                try:
                    _html_response(self, 200, home_html())
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
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

    return ClientHandler
