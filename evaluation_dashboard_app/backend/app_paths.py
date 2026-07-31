"""Filesystem anchors shared by the bbox API server and the local desktop client.

``local_bbox_api`` originally derived its static, cache and data directories from
``Path.cwd()``. That is correct for both deployments that launch it -- Docker runs it
from ``/app`` and local development runs it from the repo root -- but it silently
resolves to the wrong place when the process is started from an arbitrary directory,
which is the normal case for a double-clicked desktop app.

Every helper here keeps ``Path.cwd()`` as a fallback candidate, so the server keeps
behaving exactly as before while the client can point the same code at a workspace
by setting the environment variables documented on each function.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_root() -> Path:
    """Directory holding ``static/``, ``backend/`` and friends.

    ``EVAL_APP_ROOT`` overrides. Under PyInstaller the bundle unpacks to ``_MEIPASS``,
    which is where the packaged ``static/`` tree lives; otherwise this resolves to the
    repo root via this file's location, which matches ``Path.cwd()`` in both the
    Docker image (``/app``) and a repo-root dev shell.
    """
    override = os.environ.get("EVAL_APP_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        bundled = getattr(sys, "_MEIPASS", None)
        return Path(bundled).resolve() if bundled else Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _dedupe(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def static_dirs() -> list[Path]:
    """Candidate ``static/`` directories, most specific first."""
    return _dedupe(
        [
            app_root() / "static",
            Path.cwd() / "static",
            Path("/app/static"),
        ]
    )


def data_root() -> Path:
    """Root of the run directories, from ``EVAL_DASHBOARD_DATA_ROOT``.

    A relative value is anchored to :func:`app_root` rather than the process cwd.
    """
    raw = os.environ.get("EVAL_DASHBOARD_DATA_ROOT", "data")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = app_root() / path
    return path.resolve()


def cache_root() -> Path:
    """Writable directory for derived caches, from ``EVAL_BBOX_CACHE_DIR``.

    The frozen client must set this: :func:`app_root` points into a read-only
    PyInstaller extraction directory that is deleted on exit.
    """
    override = os.environ.get("EVAL_BBOX_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (app_root() / ".cache").resolve()


def eval_lib_paths() -> list[Path]:
    """``sys.path`` entries providing the evaluator libraries needed to read pickles.

    ``EVAL_LIB_PATHS`` (os.pathsep-separated) overrides the developer-machine defaults
    so that unpickling works on other hosts. When neither the override nor the
    defaults exist, the ``scenario_devops_*`` routes degrade to ``available: false``
    instead of raising -- and a pre-baked run does not need these at all.
    """
    override = os.environ.get("EVAL_LIB_PATHS", "").strip()
    if override:
        return _dedupe(
            [Path(chunk.strip()).expanduser() for chunk in override.split(os.pathsep) if chunk.strip()]
        )
    return [
        Path("/home/leigu/driving_log_replayer_v2/driving_log_replayer_v2"),
        Path("/home/leigu/autoware_perception_evaluation/perception_eval"),
    ]


def find_static_file(name: str) -> Path | None:
    for directory in static_dirs():
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None
