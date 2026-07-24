"""
Optional app-level auth: identify the current user for per-user task visibility.
Designed to work with company auth (e.g. WebAutoAuth, OAuth2 proxy) that sets
a header with the user identity. When enabled, users see only their own tasks.
"""

import base64
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Header name set by auth proxy (e.g. X-Forwarded-User, X-Auth-User). Empty = no auth filtering.
AUTH_USER_HEADER = os.environ.get("AUTH_USER_HEADER", "").strip()

# For local/dev: force a user id when no header is available (e.g. AUTH_DEFAULT_USER=dev@example.com).
AUTH_DEFAULT_USER = os.environ.get("AUTH_DEFAULT_USER", "").strip() or None

# Cloudflare Access / Tunnel headers. These are injected by Cloudflare in front of the
# app and are absent on direct (localhost / raw IP) access. The email header is trusted
# only as far as the deployment guarantees traffic cannot bypass Cloudflare — see
# get_access_user_email() docstring.
CF_ACCESS_EMAIL_HEADER = "Cf-Access-Authenticated-User-Email"
CF_ACCESS_JWT_HEADER = "Cf-Access-Jwt-Assertion"
CF_RAY_HEADER = "Cf-Ray"
CF_CONNECTING_IP_HEADER = "Cf-Connecting-Ip"


def _first_nonempty_string(*values: Any) -> str:
    """Return the first non-empty string-like value, else empty string."""
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _read_streamlit_headers() -> Dict[str, str]:
    """Best-effort request headers from Streamlit context."""
    try:
        import streamlit as st

        ctx = getattr(st, "context", None)
        headers = getattr(ctx, "headers", None) if ctx else None
        if callable(headers):
            headers = headers()
        if isinstance(headers, dict):
            normalized: Dict[str, str] = {}
            for key, value in headers.items():
                if not isinstance(key, str):
                    continue
                normalized[key] = str(value)
            return normalized
    except Exception:
        pass
    return {}


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    """Best-effort JWT payload decode without signature verification, for display only."""
    raw = str(token or "").strip()
    if not raw:
        return {}
    parts = raw.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload + padding)
        data = json.loads(decoded.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_identity_from_bearer_token(headers: Dict[str, str]) -> Dict[str, Any]:
    """Extract subject / email / username / display name from common bearer token claims."""
    authz = str(headers.get("Authorization") or headers.get("authorization") or "").strip()
    if not authz.lower().startswith("bearer "):
        return {}
    token = authz.split(" ", 1)[1].strip()
    payload = _decode_jwt_payload(token)
    if not payload:
        return {}

    session = payload.get("session") or {}
    identity = session.get("identity") or {}
    traits = identity.get("traits") or {}
    name = traits.get("name") or {}
    oauth_username = _first_nonempty_string(
        payload.get("preferred_username"),
        payload.get("username"),
        payload.get("upn"),
        payload.get("unique_name"),
        payload.get("cognito:username"),
        traits.get("username"),
        identity.get("username"),
    )
    full_name = " ".join(
        part for part in [str(name.get("first") or "").strip(), str(name.get("last") or "").strip()] if part
    ).strip()
    display_name = _first_nonempty_string(
        payload.get("name"),
        full_name,
        traits.get("display_name"),
        identity.get("display_name"),
        oauth_username,
        traits.get("email"),
    )
    email = _first_nonempty_string(
        payload.get("email"),
        payload.get("upn"),
        traits.get("email"),
        identity.get("email"),
    )
    subject_id = _first_nonempty_string(
        payload.get("sub"),
        session.get("account", {}).get("subject_id"),
        identity.get("id"),
    )
    return {
        "subject_id": subject_id,
        "email": email,
        "username": oauth_username,
        "name": display_name,
        "claims": payload,
    }


def get_current_user_id() -> Optional[str]:
    """
    Return the current user identifier, or None if auth is not configured.
    Uses (in order):
    1. HTTP header named by AUTH_USER_HEADER (when Streamlit is behind an auth proxy / WebAutoAuth).
    2. AUTH_DEFAULT_USER (for development or when proxy does not set the header).
    Streamlit 1.37+ provides st.context.headers; on older versions we fall back to AUTH_DEFAULT_USER only.
    """
    if not AUTH_USER_HEADER and not AUTH_DEFAULT_USER:
        return None
    headers = _read_streamlit_headers()
    value = headers.get(AUTH_USER_HEADER) or headers.get(AUTH_USER_HEADER.lower())
    if value and isinstance(value, str) and value.strip():
        return value.strip()
    return AUTH_DEFAULT_USER


def is_auth_enabled() -> bool:
    """True if AUTH_USER_HEADER or AUTH_DEFAULT_USER is set (per-user task filtering)."""
    return bool(AUTH_USER_HEADER or AUTH_DEFAULT_USER)


def _header_ci(headers: Dict[str, str], name: str) -> str:
    """Case-insensitive header lookup."""
    if name in headers:
        return str(headers[name] or "").strip()
    lname = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lname:
            return str(value or "").strip()
    return ""


def detect_access_origin(headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Classify whether the request arrived via Cloudflare or a direct (localhost/IP) hit.

    Detection relies on Cloudflare-injected headers (Cf-Ray / Cf-Connecting-Ip /
    Cdn-Loop: cloudflare) which are absent on direct access.
    """
    if headers is None:
        headers = _read_streamlit_headers()
    cf_ray = _header_ci(headers, CF_RAY_HEADER)
    cf_connecting_ip = _header_ci(headers, CF_CONNECTING_IP_HEADER)
    cdn_loop = _header_ci(headers, "Cdn-Loop")
    is_cloudflare = bool(cf_ray or cf_connecting_ip or "cloudflare" in cdn_loop.lower())
    return {
        "origin": "cloudflare" if is_cloudflare else "direct",
        "is_cloudflare": is_cloudflare,
        "host": _header_ci(headers, "Host"),
        "cf_ray": cf_ray,
        "cf_connecting_ip": cf_connecting_ip,
        "cdn_loop": cdn_loop,
    }


def get_access_user_email(headers: Optional[Dict[str, str]] = None) -> str:
    """Return the Cloudflare Access authenticated user email, or "" if unavailable.

    Prefers the ``Cf-Access-Authenticated-User-Email`` header, falling back to the
    ``email`` claim of the ``Cf-Access-Jwt-Assertion`` token (decoded, not verified).

    TRUST NOTE: these headers are only meaningful if the deployment guarantees that
    requests cannot reach the app except through Cloudflare Access. On direct access
    (e.g. the raw container port on the internal network) a client could set the same
    header. This helper therefore returns the email only when the request also looks
    like it came through Cloudflare, so a direct hit that forges the email header
    without the other Cf-* signals is ignored.
    """
    if headers is None:
        headers = _read_streamlit_headers()
    if not detect_access_origin(headers)["is_cloudflare"]:
        return ""
    email = _header_ci(headers, CF_ACCESS_EMAIL_HEADER)
    if email:
        return email
    payload = _decode_jwt_payload(_header_ci(headers, CF_ACCESS_JWT_HEADER))
    return _first_nonempty_string(payload.get("email"), payload.get("upn"))


def get_access_context(headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Combined view: origin classification + authenticated user email (if any)."""
    if headers is None:
        headers = _read_streamlit_headers()
    ctx = detect_access_origin(headers)
    ctx["user_email"] = get_access_user_email(headers)
    return ctx


def render_signed_in_user(*, sidebar: bool = True) -> Dict[str, Any]:
    """Show 'Signed in as …' (or an unauthenticated note) and log the access once per session.

    Returns the access context dict. Safe to call from any page; no-ops gracefully if
    Streamlit / headers are unavailable.
    """
    try:
        import streamlit as st
    except Exception:
        return {}

    ctx = get_access_context()
    email = ctx.get("user_email") or ""

    # Log once per browser session to avoid noise on every rerun.
    log_key = f"_access_logged::{email}::{ctx.get('origin')}"
    if not st.session_state.get(log_key):
        logger.info(
            "access origin=%s host=%s user=%s cf_ip=%s cf_ray=%s",
            ctx.get("origin"),
            ctx.get("host"),
            email or "-",
            ctx.get("cf_connecting_ip") or "-",
            ctx.get("cf_ray") or "-",
        )
        st.session_state[log_key] = True

    target = st.sidebar if sidebar else st
    if email:
        target.caption(f"👤 Signed in as **{email}**")
    elif ctx.get("is_cloudflare"):
        target.caption("👤 Signed in via Cloudflare (no email header)")
    else:
        target.caption("👤 Direct access — not authenticated")
    return ctx
