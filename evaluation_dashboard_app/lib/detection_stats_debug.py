"""
Optional verbose logging for pages/3_Detection_Stats.py (502 / freeze / OOM debugging).

Enable with environment variable:
  EVAL_DETECTION_STATS_DEBUG=1

Logs go to stderr (visible in `docker compose logs streamlit1`).
"""

from __future__ import annotations

import logging
import os
import resource
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, List, Tuple

_LOG = logging.getLogger("eval_dashboard.detection_stats")
_CONFIGURED = False


def detection_stats_debug_enabled() -> bool:
    v = os.environ.get("EVAL_DETECTION_STATS_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _ensure_logging() -> None:
    global _CONFIGURED
    if not detection_stats_debug_enabled():
        return
    if _CONFIGURED:
        return
    _LOG.setLevel(logging.DEBUG)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] detection_stats: %(message)s")
    )
    _LOG.addHandler(h)
    _LOG.propagate = False
    _CONFIGURED = True


def ds_dlog(fmt: str, *args: Any) -> None:
    """Log one line when debug is enabled."""
    if not detection_stats_debug_enabled():
        return
    _ensure_logging()
    try:
        _LOG.info(fmt, *args)
    except Exception:
        _LOG.info("%s %s", fmt, args)


def ds_debug_init_session_state(session_state: Any) -> None:
    """Call once per script run (after set_page_config). Resets timing buffer."""
    if not detection_stats_debug_enabled():
        return
    session_state["_ds_debug_timings"] = []
    session_state["_ds_debug_run_started"] = time.perf_counter()
    ds_dlog("=== Detection Stats script run started ===")
    ds_dlog("pid=%s argv[0]=%s", os.getpid(), sys.argv[0] if sys.argv else "")
    for key in (
        "EVAL_DETECTION_STATS_DEBUG",
        "STREAMLIT_SERVER_COOKIE_SECRET",
        "EVAL_DASHBOARD_DATA_ROOT",
    ):
        v = os.environ.get(key)
        if key == "STREAMLIT_SERVER_COOKIE_SECRET" and v:
            ds_dlog("env %s=(set len=%s)", key, len(v))
        else:
            ds_dlog("env %s=%r", key, v)


def ds_debug_log_memory(note: str = "") -> None:
    if not detection_stats_debug_enabled():
        return
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: ru_maxrss kilobytes; macOS: bytes (best-effort label)
        ds_dlog(
            "MEM %s ru_maxrss=%s ru_utime=%.3fs ru_stime=%.3fs",
            note,
            ru.ru_maxrss,
            ru.ru_utime,
            ru.ru_stime,
        )
    except Exception as e:
        ds_dlog("MEM %s (unavailable: %s)", note, e)


def _append_timing(session_state: Any, name: str, seconds: float) -> None:
    if not detection_stats_debug_enabled():
        return
    lst = session_state.get("_ds_debug_timings")
    if not isinstance(lst, list):
        lst = []
        session_state["_ds_debug_timings"] = lst
    lst.append((name, seconds))


@contextmanager
def ds_dtimer(name: str, session_state: Any):
    """Time a block; record to session_state for the debug expander."""
    if not detection_stats_debug_enabled():
        yield
        return
    t0 = time.perf_counter()
    ds_dlog("TIMER start %s", name)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        ds_dlog("TIMER end %s (%.3fs)", name, dt)
        _append_timing(session_state, name, dt)


def ds_debug_log_exception(where: str, exc: BaseException) -> None:
    if not detection_stats_debug_enabled():
        return
    _ensure_logging()
    _LOG.exception("EXCEPTION in %s: %s", where, exc)


def ds_debug_render_expander(session_state: Any) -> None:
    """Renders a Streamlit expander with timings + env (only if debug on)."""
    import streamlit as st

    if not detection_stats_debug_enabled():
        return
    t_run = session_state.get("_ds_debug_run_started")
    total_s = None
    if isinstance(t_run, (int, float)):
        total_s = time.perf_counter() - float(t_run)

    timings: List[Tuple[str, float]] = session_state.get("_ds_debug_timings") or []
    lines = [
        f"Total wall time (approx): {total_s:.3f}s" if total_s is not None else "Total wall time: n/a",
        "",
        "Section timings (seconds):",
    ]
    for name, sec in timings:
        lines.append(f"  - {name}: {sec:.3f}s")
    if not timings:
        lines.append("  (no ds_dtimer sections recorded)")

    lines.extend(
        [
            "",
            "Environment (subset):",
            f"  EVAL_DETECTION_STATS_DEBUG={os.environ.get('EVAL_DETECTION_STATS_DEBUG', '')!r}",
            f"  EVAL_DASHBOARD_DATA_ROOT={os.environ.get('EVAL_DASHBOARD_DATA_ROOT', '')!r}",
        ]
    )

    with st.expander("Detection Stats debug (EVAL_DETECTION_STATS_DEBUG=1)", expanded=False):
        st.code("\n".join(lines), language="text")
        st.caption("Check `docker compose logs streamlit1` for the same lines on stderr.")
