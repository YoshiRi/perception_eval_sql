"""Pixel-art "office floor" that shows workflow tasks working in realtime.

One desk per task, one worker per desk. Workers walk in through the door when a task
is queued, type while the job runs (with the occasional walk to the CAFE machine),
celebrate under confetti -- or slump at a red monitor -- and finally walk back out
the door. Progress percent and the task's ``progress_message`` are drawn above each
desk, so a glance at the floor answers "what is running and how far along".

The scene engine lives in ``static/pixel_office.js`` and is shared with the packaged
local client's workflow page; this module only prepares the task payload (from the
same DB rows the task list uses), picks the theme via ``lib.ui.theme``, and inlines
the engine into a ``st.components.v1.html`` iframe.

The page re-renders the iframe inside its 3-second live fragment; every animation in
the engine is a pure function of wall-clock time, the task's own timestamps, and a
hash of the task id, so a re-render never visibly resets the scene -- the characters
simply keep walking/typing where the clock says they should be.
"""

from __future__ import annotations

import functools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

# A finished task keeps its desk long enough to celebrate/slump and walk out the door
# (mirrors RECENT_FINISH_MS in static/pixel_office.js).
RECENT_FINISH_SECONDS = 75
# Desks beyond this become the queue outside the door (mirrors MAX_DESKS in the JS).
MAX_DESKS = 10

_FLOOR_HEIGHT_PX = 300
_IDLE_HEIGHT_PX = 204
_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
_ENGINE_PATH = _STATIC_DIR / "pixel_office.js"


@functools.lru_cache(maxsize=1)
def _engine_js() -> str:
    return _ENGINE_PATH.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _office_component():
    """The bidirectional component (static/index.html + the engine).

    Serving through declare_component gives the floor a return channel: desk
    clicks come back as the component value, so the page can open the same full
    task-details dialog the task list uses. It also means fragment reruns stream
    new data into the existing iframe instead of reloading it. Returns None when
    the API is unavailable so callers can fall back to the inline render.
    """
    try:
        if not (_STATIC_DIR / "index.html").is_file():
            return None
        return components.declare_component("pixel_office", path=str(_STATIC_DIR))
    except Exception:
        return None


def _theme() -> str:
    try:
        from lib.ui.theme import active_theme

        return active_theme()
    except Exception:
        return "light"


def _epoch_ms(value: Any) -> Optional[int]:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _params(task: Dict[str, Any]) -> Dict[str, Any]:
    params = task.get("parameters") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    return params if isinstance(params, dict) else {}


def _task_name(task: Dict[str, Any]) -> str:
    params = _params(task)
    for key in ("target_name", "output_path", "job_id", "eval_root", "pkl_dir"):
        value = str(params.get(key) or "").strip()
        if value:
            return value.rstrip("/").rsplit("/", 1)[-1]
    return str(task.get("type") or "task")


def _requested_by(task: Dict[str, Any]) -> str:
    requester = _params(task).get("_requester")
    if isinstance(requester, dict):
        return str(requester.get("name") or requester.get("email") or "").strip()
    return ""


def _payload(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tasks worth a desk: everything active, plus finishes recent enough to celebrate."""
    now_ms = int(time.time() * 1000)
    items: List[Dict[str, Any]] = []
    for task in tasks:
        status = str(task.get("status") or "")
        updated = _epoch_ms(task.get("updated_at"))
        if status in ("completed", "failed"):
            if updated is None or now_ms - updated > RECENT_FINISH_SECONDS * 1000:
                continue
        elif status not in ("pending", "running"):
            continue
        pct = task.get("progress_pct")
        try:
            pct = None if pct is None else max(0.0, min(100.0, float(pct)))
        except (TypeError, ValueError):
            pct = None
        params = _params(task)
        output = str(params.get("output_path") or "").strip()
        items.append({
            "id": str(task.get("id") or ""),
            "status": status,
            "name": _task_name(task),
            "type": str(task.get("type") or ""),
            "pct": pct,
            "message": str(task.get("progress_message") or "").strip(),
            "error": str(task.get("error_message") or "").strip(),
            "created": _epoch_ms(task.get("created_at")),
            "updated": updated,
            "run": output.rstrip("/").rsplit("/", 1)[-1] if output else "",
            "target": str(params.get("target_name") or "").strip(),
            "by": _requested_by(task),
            "result": str(task.get("result_path") or "").strip(),
        })
    # Active first (running before pending), then the recently finished.
    order = {"running": 0, "pending": 1, "completed": 2, "failed": 2}
    items.sort(key=lambda t: (order.get(t["status"], 3), -(t["created"] or 0)))
    return items


def render_pixel_office(tasks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Draw the office floor for the given task rows (same rows the task list uses).

    Returns the last desk click as ``{"id": task_id, "t": nonce}`` (or None). The
    value persists across reruns, so callers must de-duplicate on the nonce before
    opening a dialog for it.
    """
    items = _payload(tasks)
    overflow = max(0, len(items) - MAX_DESKS)
    items = items[:MAX_DESKS]
    component = _office_component()
    if component is not None:
        try:
            clicked = component(
                tasks=items,
                overflow=overflow,
                theme=_theme(),
                key="pixel_office_floor",
                default=None,
            )
            return clicked if isinstance(clicked, dict) else None
        except Exception:
            pass  # fall back to the inline render below
    height = _FLOOR_HEIGHT_PX if items else _IDLE_HEIGHT_PX
    data = json.dumps(
        {"tasks": items, "overflow": overflow, "theme": _theme()}
    ).replace("</", "<\\/")
    html = (
        '<div id="pxoffice"></div>'
        "<script>" + _engine_js() + "</script>"
        "<script>PixelOffice.mount(document.getElementById('pxoffice'), " + data + ");</script>"
    )
    components.html(html, height=height + 8, scrolling=False)
    return None
