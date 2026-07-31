"""
Remember the last run selection per user so any page can restore it.

Overview already persists its widget defaults through :class:`lib.user_config.UserConfig`, but that
file is global and only Overview reads it back. This store keeps a small per-user record of the last
selection Overview actually loaded (mode + baseline + compare runs), so opening a page such as
Detection Stats directly — new browser session, no `run_a=...` in the URL — restores whatever the
user was last looking at instead of asking them to visit Overview again.

This is the server-side half of the memory and is shared by every browser the user logs in from.
For true per-browser memory see :mod:`lib.run_selection_cookie`; the hydration order is
cookie → this store (see :mod:`lib.overview_url_hydrate`).

With several Streamlit replicas behind a load balancer this file should be on shared storage; when
it is not, the browser cookie and the URL query params written by Overview remain the cross-replica
paths.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.user_config import CONFIG_FILE

DEFAULT_USER_KEY = "default"


def _store_path() -> Path:
    override = os.environ.get("EVAL_DASHBOARD_RUN_SELECTION_STORE")
    if override:
        return Path(override)
    return Path(CONFIG_FILE).expanduser().parent / "run_selection_state.json"


def _user_key() -> str:
    """Identify the caller so selections do not leak between users of a shared deployment."""
    try:
        from lib.auth import get_current_user_id

        user_id = get_current_user_id() or ""
    except Exception:
        user_id = ""
    user_id = str(user_id).strip().lower()
    if not user_id:
        return DEFAULT_USER_KEY
    # Keep the key readable in the JSON file while staying safe for arbitrary header values.
    return re.sub(r"[^a-z0-9._@+-]", "_", user_id)[:200]


def _read_store() -> Dict[str, Any]:
    path = _store_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    users = data.get("users")
    return {"users": users} if isinstance(users, dict) else {}


def _write_store(store: Dict[str, Any]) -> None:
    path = _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        # Persistence is a convenience; never break a page render over it.
        pass


def save_run_selection(
    mode: str,
    run_a_name: str,
    compare_run_names: Optional[List[str]] = None,
) -> None:
    """Record the selection currently loaded. `mode` accepts "single"/"compare" or Overview labels."""
    run_a_name = str(run_a_name or "").strip()
    if not run_a_name:
        return
    normalized_mode = "compare" if "compare" in str(mode or "").strip().lower() else "single"
    compare = [str(n).strip() for n in (compare_run_names or []) if str(n).strip()]
    entry = {
        "mode": normalized_mode,
        "run_a": run_a_name,
        "compare_runs": compare,
    }
    store = _read_store()
    users = store.setdefault("users", {})
    previous = users.get(_user_key())
    if isinstance(previous, dict) and all(previous.get(k) == v for k, v in entry.items()):
        return
    entry["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    users[_user_key()] = entry
    _write_store(store)


def load_run_selection() -> Optional[Dict[str, Any]]:
    """Return the last saved selection for this user, or None when nothing usable is stored."""
    entry = (_read_store().get("users") or {}).get(_user_key())
    if not isinstance(entry, dict):
        return None
    run_a_name = str(entry.get("run_a") or "").strip()
    if not run_a_name:
        return None
    compare = entry.get("compare_runs")
    compare_names = [str(n).strip() for n in compare if str(n).strip()] if isinstance(compare, list) else []
    return {
        "mode": "compare" if str(entry.get("mode") or "").lower() == "compare" else "single",
        "run_a": run_a_name,
        "compare_runs": compare_names,
        "updated_at": entry.get("updated_at"),
    }


def clear_run_selection() -> None:
    """Forget this user's saved selection."""
    store = _read_store()
    users = store.get("users")
    if isinstance(users, dict) and users.pop(_user_key(), None) is not None:
        _write_store(store)
