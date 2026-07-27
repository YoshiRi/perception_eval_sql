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
        # Streamlit 1.37+ returns a StreamlitHeaders mapping (not a dict), so accept
        # anything with .items() rather than requiring an exact dict instance.
        if headers is not None and hasattr(headers, "items"):
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


def get_current_user_identity(headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Return the best available request identity.

    Cloudflare Access is used automatically when present. Direct access keeps the
    legacy behavior: no identity unless AUTH_USER_HEADER/AUTH_DEFAULT_USER is
    configured by the deployment.
    """
    if headers is None:
        headers = _read_streamlit_headers()

    access = get_access_context(headers)
    email = str(access.get("user_email") or "").strip()
    if email:
        return {
            "id": email,
            "email": email,
            "name": email,
            "source": "cloudflare_access",
            "origin": access.get("origin") or "",
            "is_cloudflare": bool(access.get("is_cloudflare")),
        }

    if AUTH_USER_HEADER:
        value = _header_ci(headers, AUTH_USER_HEADER)
        if value:
            return {
                "id": value,
                "email": value if "@" in value else "",
                "name": value,
                "source": f"header:{AUTH_USER_HEADER}",
                "origin": access.get("origin") or "",
                "is_cloudflare": bool(access.get("is_cloudflare")),
            }

    bearer_identity = _extract_identity_from_bearer_token(headers)
    bearer_id = _first_nonempty_string(
        bearer_identity.get("email"),
        bearer_identity.get("username"),
        bearer_identity.get("subject_id"),
    )
    if bearer_id:
        return {
            "id": bearer_id,
            "email": str(bearer_identity.get("email") or "").strip(),
            "name": str(bearer_identity.get("name") or bearer_id).strip(),
            "source": "bearer",
            "origin": access.get("origin") or "",
            "is_cloudflare": bool(access.get("is_cloudflare")),
        }

    if AUTH_DEFAULT_USER:
        return {
            "id": AUTH_DEFAULT_USER,
            "email": AUTH_DEFAULT_USER if "@" in AUTH_DEFAULT_USER else "",
            "name": AUTH_DEFAULT_USER,
            "source": "default",
            "origin": access.get("origin") or "",
            "is_cloudflare": bool(access.get("is_cloudflare")),
        }

    return {
        "id": "",
        "email": "",
        "name": "",
        "source": "anonymous",
        "origin": access.get("origin") or "",
        "is_cloudflare": bool(access.get("is_cloudflare")),
    }


def get_current_user_id() -> Optional[str]:
    """Return the current user identifier, or None when this request is anonymous."""
    identity = get_current_user_identity()
    user_id = str(identity.get("id") or "").strip()
    return user_id or None


def is_auth_enabled() -> bool:
    """True when this request has an identity that can scope task history."""
    if AUTH_USER_HEADER or AUTH_DEFAULT_USER:
        return True
    return bool(get_current_user_id())


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


def _log_access_once(st, ctx: Dict[str, Any]) -> None:
    """Log the resolved access context once per browser session (avoids per-rerun noise)."""
    email = ctx.get("user_email") or ""
    log_key = f"_access_logged::{email}::{ctx.get('origin')}"
    if st.session_state.get(log_key):
        return
    logger.info(
        "access origin=%s host=%s user=%s cf_ip=%s cf_ray=%s",
        ctx.get("origin"),
        ctx.get("host"),
        email or "-",
        ctx.get("cf_connecting_ip") or "-",
        ctx.get("cf_ray") or "-",
    )
    st.session_state[log_key] = True


def _identity_label(ctx: Dict[str, Any]) -> str:
    """Human-facing identity string, or '' when there is nothing worth showing."""
    email = ctx.get("user_email") or ""
    if email:
        return f"Signed in as {email}"
    if ctx.get("is_cloudflare"):
        return "Signed in via Cloudflare"
    return ""


def render_identity_badge() -> Dict[str, Any]:
    """Render a small, right-aligned identity badge at the top of the main area.

    Called once per page from `inject_app_page_styles`, so it appears app-wide.
    Logs the access once per session. No-ops when there is no identity to show
    (e.g. local/direct access) or when Streamlit/headers are unavailable.
    """
    try:
        import streamlit as st
    except Exception:
        return {}

    ctx = get_access_context()
    _log_access_once(st, ctx)

    label = _identity_label(ctx)
    if not label:
        return ctx
    st.markdown(
        f"<div style='text-align:right; margin:-0.5rem 0 0.25rem; "
        f"font-size:0.8rem; color:#64748b;'>👤 {label}</div>",
        unsafe_allow_html=True,
    )
    return ctx


def render_signed_in_user(*, sidebar: bool = True) -> Dict[str, Any]:
    """Show 'Signed in as …' (or an unauthenticated note) and log the access once per session.

    Kept for explicit per-page use (e.g. in a sidebar). App-wide display is handled
    by `render_identity_badge` via `inject_app_page_styles`.
    """
    try:
        import streamlit as st
    except Exception:
        return {}

    ctx = get_access_context()
    _log_access_once(st, ctx)
    email = ctx.get("user_email") or ""

    target = st.sidebar if sidebar else st
    if email:
        target.caption(f"👤 Signed in as **{email}**")
    elif ctx.get("is_cloudflare"):
        target.caption("👤 Signed in via Cloudflare (no email header)")
    else:
        target.caption("👤 Direct access — not authenticated")
    return ctx
