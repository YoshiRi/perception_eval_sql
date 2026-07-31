"""
Per-browser memory of the last run selection, stored in a browser cookie.

Streamlit can *read* cookies (``st.context.cookies``) but cannot set them, so the write side is a
tiny inline HTML component: a zero-height iframe whose script assigns ``document.cookie``. Streamlit
renders component HTML with ``srcdoc``, which inherits the app's origin, so the cookie lands on the
dashboard's own domain and is sent back with the next page load.

Why this and not only the server-side store (:mod:`lib.run_selection_store`):

* it is genuinely *per browser* — the same account can keep different selections in two browsers,
  two profiles, or a private window, which is what you want when comparing runs side by side;
* it needs no shared filesystem, so it survives a load-balancer hop between Streamlit replicas.

Timing note: a cookie written during a script run reaches the server with the **next full page
load**, because ``st.context.cookies`` reflects the headers of the request that opened the session.
That is exactly the case this feature targets (opening e.g. Detection Stats directly in a new tab or
after a restart); within one live session ``st.session_state`` already carries the selection.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

COOKIE_NAME = "eval_dashboard_run_selection"
COOKIE_MAX_AGE_SECONDS = 90 * 24 * 3600
# Set to 1/true to stop writing the cookie (e.g. a deployment that must not store client-side state).
DISABLE_ENV = "EVAL_DASHBOARD_DISABLE_SELECTION_COOKIE"
_WRITTEN_STATE_KEY = "_run_selection_cookie_written"


def cookies_disabled() -> bool:
    return str(os.environ.get(DISABLE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def normalize_selection(
    mode: str,
    run_a_name: str,
    compare_run_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Shared shape for both persistence layers: {"mode": "single"|"compare", run_a, compare_runs}."""
    run_a_name = str(run_a_name or "").strip()
    if not run_a_name:
        return None
    normalized_mode = "compare" if "compare" in str(mode or "").strip().lower() else "single"
    compare = [str(n).strip() for n in (compare_run_names or []) if str(n).strip()]
    return {"mode": normalized_mode, "run_a": run_a_name, "compare_runs": compare}


def encode_selection_cookie_value(selection: Dict[str, Any]) -> str:
    """Percent-encoded JSON, so run names with spaces/semicolons stay valid inside a cookie."""
    payload = {
        "v": 1,
        "mode": selection["mode"],
        "run_a": selection["run_a"],
        "compare_runs": selection.get("compare_runs") or [],
    }
    return urllib.parse.quote(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), safe="")


def decode_selection_cookie_value(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Inverse of :func:`encode_selection_cookie_value`; None for anything unusable."""
    if not raw:
        return None
    try:
        payload = json.loads(urllib.parse.unquote(str(raw)))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    run_a_name = str(payload.get("run_a") or "").strip()
    if not run_a_name:
        return None
    compare = payload.get("compare_runs")
    compare_names = (
        [str(n).strip() for n in compare if str(n).strip()] if isinstance(compare, list) else []
    )
    return {
        "mode": "compare" if str(payload.get("mode") or "").lower() == "compare" else "single",
        "run_a": run_a_name,
        "compare_runs": compare_names,
    }


def read_selection_cookie() -> Optional[Dict[str, Any]]:
    """Selection remembered by *this browser*, or None when absent/unreadable."""
    try:
        cookies = st.context.cookies or {}
    except Exception:
        # Older Streamlit, or called outside a script run.
        return None
    return decode_selection_cookie_value(cookies.get(COOKIE_NAME))


def _render_cookie_script(cookie_value: str, max_age: int) -> None:
    """Zero-height component that writes the cookie on the parent document."""
    js_name = json.dumps(COOKIE_NAME)
    js_value = json.dumps(cookie_value)
    components.html(
        f"""<script>
(function () {{
  var name = {js_name};
  var value = {js_value};
  var maxAge = {int(max_age)};
  // srcdoc inherits the app origin, so prefer the parent document and fall back to our own.
  var target = document;
  var isSecure = false;
  try {{
    target = window.parent.document;
    isSecure = window.parent.location.protocol === "https:";
  }} catch (e) {{
    target = document;
  }}
  var cookie = name + "=" + value + "; path=/; max-age=" + maxAge + "; SameSite=Lax";
  if (isSecure) {{
    cookie += "; Secure";
  }}
  try {{
    target.cookie = cookie;
  }} catch (e) {{
    /* Cookies blocked: the server-side store still remembers the selection. */
  }}
}})();
</script>""",
        height=0,
        width=0,
    )


def persist_selection_cookie(
    mode: str,
    run_a_name: str,
    compare_run_names: Optional[List[str]] = None,
) -> None:
    """Remember this selection in the browser. No-op when disabled or already written this session."""
    if cookies_disabled():
        return
    selection = normalize_selection(mode, run_a_name, compare_run_names)
    if selection is None:
        return
    cookie_value = encode_selection_cookie_value(selection)
    # Reruns are frequent; only re-inject the script when the value actually changed.
    try:
        if st.session_state.get(_WRITTEN_STATE_KEY) == cookie_value:
            return
        st.session_state[_WRITTEN_STATE_KEY] = cookie_value
    except Exception:
        pass
    _render_cookie_script(cookie_value, COOKIE_MAX_AGE_SECONDS)


def clear_selection_cookie() -> None:
    """Forget the browser-side selection (expire the cookie)."""
    try:
        st.session_state.pop(_WRITTEN_STATE_KEY, None)
    except Exception:
        pass
    _render_cookie_script("", 0)
