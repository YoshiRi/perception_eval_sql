"""Workspace layout and stored settings for the local client.

Everything the client owns lives under one directory (``~/.evaldash`` by default, or
``EVALDASH_HOME``) so that uninstalling is a single ``rm -rf`` and so the packaged app
never writes into its own read-only bundle.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

CONFIG_NAME = "config.json"
RUN_STATE_NAME = ".evaldash_manifest.json"


def home() -> Path:
    override = os.environ.get("EVALDASH_HOME", "").strip()
    base = Path(override).expanduser() if override else Path.home() / ".evaldash"
    return base.resolve()


def workspace_dir() -> Path:
    """Data root handed to the bbox API. Run directories sit directly inside."""
    return home() / "workspace"


def cache_dir() -> Path:
    return home() / "cache"


def t4_cache_dir() -> Path:
    return home() / "t4"


def config_path() -> Path:
    return home() / CONFIG_NAME


def ensure_dirs() -> None:
    for path in (home(), workspace_dir(), cache_dir()):
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Connection settings. ``token`` is a secret and is stored 0600."""

    server_url: str = ""
    token: str = ""
    cf_client_id: str = ""
    cf_client_secret: str = ""
    verify_tls: bool = True
    timeout_sec: float = 60.0
    t4_base_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        path = config_path()
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        values = {k: v for k, v in raw.items() if k in known}
        values["extra"] = {k: v for k, v in raw.items() if k not in known}
        return cls(**values)

    def save(self) -> Path:
        ensure_dirs()
        path = config_path()
        payload = asdict(self)
        payload.update(payload.pop("extra", {}) or {})
        # Write then tighten, and tighten before the secret lands where possible.
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        return path

    def effective_server(self) -> str:
        """Server URL in force, stored or from the environment. Empty when unset."""
        return (self.server_url or os.environ.get("EVALDASH_SERVER", "")).strip().rstrip("/")

    def require_server(self) -> str:
        url = self.effective_server()
        if not url:
            raise RuntimeError(
                "No server configured. Run: evaldash-local login --server <url> --token <token>"
            )
        return url

    def resolved_token(self) -> str:
        return (self.token or os.environ.get("EVALDASH_TOKEN", "")).strip()


def run_dir(run_name: str) -> Path:
    return workspace_dir() / run_name


def run_state_path(run_name: str) -> Path:
    return run_dir(run_name) / RUN_STATE_NAME


def read_run_state(run_name: str) -> dict[str, Any]:
    """What a previous pull recorded, used to decide what still needs fetching."""
    path = run_state_path(run_name)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_run_state(run_name: str, state: dict[str, Any]) -> None:
    path = run_state_path(run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def local_runs() -> list[str]:
    root = workspace_dir()
    if not root.is_dir():
        return []
    return sorted(
        child.name for child in root.iterdir() if child.is_dir() and not child.name.startswith(".")
    )


def apply_server_env() -> None:
    """Point the bbox API at this workspace.

    Set before importing the API so its module-level defaults resolve correctly.
    """
    ensure_dirs()
    os.environ["EVAL_DASHBOARD_DATA_ROOT"] = str(workspace_dir())
    os.environ["EVAL_BBOX_CACHE_DIR"] = str(cache_dir())
    os.environ.setdefault("LOCAL_BBOX_ALLOWED_ROOTS", str(workspace_dir()))
    # The client is not an export server; leaving the token unset keeps those routes
    # closed, which is what we want for a process listening on a laptop.
    os.environ.pop("EVAL_EXPORT_TOKEN", None)
