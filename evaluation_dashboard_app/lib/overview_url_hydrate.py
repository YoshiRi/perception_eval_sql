"""
Rehydrate session_state from the URL, the browser cookie, or the last server-side selection.

Overview syncs `mode`, `run_a`, `run_b`, ... via `st.query_params`. After a load-balancer hop to a
different Streamlit replica, `st.session_state` may not contain `runA` even though the user already
used Overview — the URL still encodes the selection. This module rebuilds `runA` / compare state
from that URL so multipage analysis works without requiring Overview to run again on the same box.
It also refreshes stale in-browser Streamlit sessions when a direct URL points at a different run
selection than the one already stored in `st.session_state`.

When neither session state nor the URL carries a selection — e.g. Detection Stats is opened directly
in a new tab, or after a server restart — the selection is restored from memory, in this order:

1. `st.session_state` (the live session);
2. URL query params (an explicit/shared link always wins);
3. the browser cookie (:mod:`lib.run_selection_cookie`) — per browser, so two browsers or profiles
   can hold different selections;
4. the server-side per-user store (:mod:`lib.run_selection_store`) — the fallback when cookies are
   blocked or this browser has not been used before.

A restored selection is mirrored back into the URL, so refreshes and replica hops keep it and the
page's share links point at the runs actually on screen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import streamlit as st

from lib.path_utils import get_data_root, get_run_display_name, get_run_storage_name, list_run_directories
from lib.run_loader import load_run
from lib.run_selection_cookie import persist_selection_cookie, read_selection_cookie
from lib.run_selection_store import load_run_selection, save_run_selection


def _url_run_signature(params) -> tuple[str, ...] | None:
    run_a_name = params.get("run_a")
    if not run_a_name:
        return None
    mode_param = (params.get("mode") or "single").lower()
    compare_names = tuple(
        params.get(k)
        for k in ("run_b", "run_c", "run_d", "run_e")
        if params.get(k)
    )
    return (mode_param, run_a_name, *compare_names)


def _name_to_dir() -> Dict[str, Path] | None:
    """Map both display and storage names to run directories, or None when data root is unusable."""
    root = get_data_root()
    if not root.exists() or not root.is_dir():
        return None
    run_dirs = list_run_directories()
    mapping = {get_run_display_name(p): p for p in run_dirs}
    mapping.update({get_run_storage_name(p): p for p in run_dirs})
    return mapping


def _apply_selection(
    mode_param: str,
    run_a_name: str,
    compare_names: Sequence[str],
    name_to_dir: Dict[str, Path],
    signature: tuple[str, ...],
) -> bool:
    """Load the runs and populate session_state the way Overview does. False if not loadable."""
    if run_a_name not in name_to_dir:
        return False
    try:
        if mode_param == "compare":
            valid = [n for n in compare_names if n in name_to_dir]
            if not valid:
                return False
            all_dirs = [name_to_dir[run_a_name]] + [name_to_dir[n] for n in valid]
            run_labels = ["A"] + [chr(66 + i) for i in range(len(valid))]
            all_runs = [load_run(d) for d in all_dirs]
            st.session_state.update(
                {
                    "mode": "Compare Mode",
                    "runA": all_runs[0],
                    "all_runs": all_runs,
                    "run_labels": run_labels,
                    "df_cmp": None,
                    "_overview_url_hydrate_sig": signature,
                }
            )
            st.session_state["runB"] = all_runs[1] if len(all_runs) >= 2 else None
            return True
        st.session_state["runA"] = load_run(name_to_dir[run_a_name])
        st.session_state["mode"] = "Single Mode"
        st.session_state["_overview_url_hydrate_sig"] = signature
        for key in ("all_runs", "run_labels", "runB", "df_cmp"):
            st.session_state.pop(key, None)
        return True
    except Exception:
        return False


def _sync_url(mode_param: str, run_a_name: str, compare_names: Sequence[str]) -> None:
    query = {"mode": mode_param, "run_a": run_a_name}
    for j, name in enumerate(compare_names):
        query[f"run_{chr(98 + j)}"] = name
    try:
        st.query_params.update(query)
    except Exception:
        pass


def _apply_remembered_selection(saved: Dict[str, Any], from_cookie: bool) -> bool:
    """Load a selection recovered from the cookie or the server-side store."""
    name_to_dir = _name_to_dir()
    if not name_to_dir:
        return False
    mode_param = saved["mode"]
    run_a_name = saved["run_a"]
    compare_names: List[str] = [n for n in (saved.get("compare_runs") or []) if n in name_to_dir]
    if mode_param == "compare" and not compare_names:
        # The candidates are gone (deleted runs) — fall back to the baseline alone rather than nothing.
        mode_param = "single"
    signature = (mode_param, run_a_name, *compare_names)
    if not _apply_selection(mode_param, run_a_name, compare_names, name_to_dir, signature):
        return False
    _sync_url(mode_param, run_a_name, compare_names)
    if not from_cookie:
        # Recovered from the server-side store: teach this browser the selection too.
        persist_selection_cookie(mode_param, run_a_name, compare_names)
    return True


def _hydrate_from_memory() -> bool:
    """Restore the last selection from the browser cookie, else from the server-side store."""
    cookie_selection = read_selection_cookie()
    if cookie_selection and _apply_remembered_selection(cookie_selection, from_cookie=True):
        return True
    saved = load_run_selection()
    if saved and _apply_remembered_selection(saved, from_cookie=False):
        return True
    return False


def try_hydrate_session_from_overview_query_params() -> bool:
    """
    If the URL has Overview-style params (`run_a`, optional `mode` / `run_b`…), load runs and populate
    `session_state` when state is missing or stale. With no such params and no `runA` in state, fall
    back to the browser cookie and then this user's saved selection. Returns True if `runA` is present
    afterward.
    """
    params = st.query_params
    url_sig = _url_run_signature(params)
    if url_sig is None:
        if "runA" in st.session_state:
            return True
        return _hydrate_from_memory()
    if (
        "runA" in st.session_state
        and st.session_state.get("_overview_url_hydrate_sig") == url_sig
    ):
        return True
    name_to_dir = _name_to_dir()
    if not name_to_dir:
        return False
    mode_param = (params.get("mode") or "single").lower()
    compare_names = [
        params.get(k)
        for k in ("run_b", "run_c", "run_d", "run_e")
        if params.get(k)
    ]
    run_a_name = params.get("run_a")
    if not _apply_selection(mode_param, run_a_name, compare_names, name_to_dir, url_sig):
        return False
    # Viewing a run via URL (shared link, bookmark) counts as "what I looked at last", so remember it
    # in both layers — the user then gets it back on a bare page URL.
    valid_compare = [n for n in compare_names if n in name_to_dir]
    persist_selection_cookie(mode_param, run_a_name, valid_compare)
    save_run_selection(mode_param, run_a_name, valid_compare)
    return True
