"""
Rehydrate session_state from Overview URL query params when server-side session is empty.

Overview syncs `mode`, `run_a`, `run_b`, ... via `st.query_params`. After a load-balancer hop to a
different Streamlit replica, `st.session_state` may not contain `runA` even though the user already
used Overview — the URL still encodes the selection. This module rebuilds `runA` / compare state
from that URL so multipage analysis works without requiring Overview to run again on the same box.
"""

from __future__ import annotations

import streamlit as st

from lib.path_utils import get_data_root, get_run_display_name, list_run_directories
from lib.run_loader import load_run


def try_hydrate_session_from_overview_query_params() -> bool:
    """
    If `runA` is missing but the URL has Overview-style params (`run_a`, optional `mode` / `run_b`…),
    load runs and populate `session_state`. Returns True if `runA` is present afterward.
    """
    if "runA" in st.session_state:
        return True
    params = st.query_params
    run_a_name = params.get("run_a")
    if not run_a_name:
        return False
    root = get_data_root()
    if not root.exists() or not root.is_dir():
        return False
    run_dirs = list_run_directories()
    name_to_dir = {get_run_display_name(p): p for p in run_dirs}
    if run_a_name not in name_to_dir:
        return False
    mode_param = (params.get("mode") or "single").lower()
    try:
        if mode_param == "compare":
            url_compare = [
                params.get(k)
                for k in ("run_b", "run_c", "run_d", "run_e")
                if params.get(k)
            ]
            valid = [n for n in url_compare if n in name_to_dir]
            if not valid:
                return False
            run_a_dir = name_to_dir[run_a_name]
            compare_dirs = [name_to_dir[n] for n in valid]
            all_dirs = [run_a_dir] + compare_dirs
            run_labels = ["A"] + [chr(66 + i) for i in range(len(compare_dirs))]
            all_runs = [load_run(d) for d in all_dirs]
            st.session_state.update(
                {
                    "mode": "Compare Mode",
                    "runA": all_runs[0],
                    "all_runs": all_runs,
                    "run_labels": run_labels,
                    "df_cmp": None,
                }
            )
            if len(all_runs) >= 2:
                st.session_state["runB"] = all_runs[1]
            else:
                st.session_state["runB"] = None
            return True
        run_a = load_run(name_to_dir[run_a_name])
        st.session_state["runA"] = run_a
        st.session_state["mode"] = "Single Mode"
        for key in ("all_runs", "run_labels", "runB", "df_cmp"):
            st.session_state.pop(key, None)
        return True
    except Exception:
        return False
