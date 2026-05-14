"""
Optional app-level auth: identify the current user for per-user task visibility.
Designed to work with company auth (e.g. WebAutoAuth, OAuth2 proxy) that sets
a header with the user identity. When enabled, users see only their own tasks.
"""

import base64
import json
import os
from typing import Any, Dict, Optional

# Header name set by auth proxy (e.g. X-Forwarded-User, X-Auth-User). Empty = no auth filtering.
AUTH_USER_HEADER = os.environ.get("AUTH_USER_HEADER", "").strip()

# For local/dev: force a user id when no header is available (e.g. AUTH_DEFAULT_USER=dev@example.com).
AUTH_DEFAULT_USER = os.environ.get("AUTH_DEFAULT_USER", "").strip() or None


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
    """Extract subject / email / display name from an Oathkeeper-style bearer token."""
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
    full_name = " ".join(
        part for part in [str(name.get("first") or "").strip(), str(name.get("last") or "").strip()] if part
    ).strip()
    display_name = (
        full_name
        or str(traits.get("display_name") or "").strip()
        or str(traits.get("email") or "").strip()
    )
    return {
        "subject_id": str(payload.get("sub") or session.get("account", {}).get("subject_id") or "").strip(),
        "email": str(traits.get("email") or "").strip(),
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


def get_current_user_session_info() -> Dict[str, Any]:
    """
    Return best-effort request/session auth info for UI debugging.

    This reflects what the Streamlit app can observe from the incoming request,
    not the evaluator token used by background workers.
    """
    headers = _read_streamlit_headers()
    configured_value = ""
    configured_source = "unavailable"
    if AUTH_USER_HEADER:
        raw_value = headers.get(AUTH_USER_HEADER) or headers.get(AUTH_USER_HEADER.lower()) or ""
        configured_value = str(raw_value).strip()
        if configured_value:
            configured_source = f"header:{AUTH_USER_HEADER}"
        else:
            configured_source = f"header:{AUTH_USER_HEADER} (missing)"
    if not configured_value and AUTH_DEFAULT_USER:
        configured_value = AUTH_DEFAULT_USER
        configured_source = "AUTH_DEFAULT_USER"

    authz = headers.get("Authorization") or headers.get("authorization") or ""
    cookie = headers.get("Cookie") or headers.get("cookie") or ""
    bearer_identity = _extract_identity_from_bearer_token(headers)
    if not configured_value and bearer_identity.get("subject_id"):
        configured_value = str(bearer_identity.get("subject_id") or "").strip()
        configured_source = "authorization:bearer"
    safe_header_keys = sorted(
        key for key in headers.keys() if key.lower() not in {"authorization", "cookie"}
    )
    return {
        "user_id": configured_value or None,
        "source": configured_source,
        "auth_user_header": AUTH_USER_HEADER or "",
        "default_user": AUTH_DEFAULT_USER,
        "has_authorization_header": bool(str(authz).strip()),
        "has_cookie_header": bool(str(cookie).strip()),
        "header_keys": safe_header_keys,
        "bearer_subject_id": str(bearer_identity.get("subject_id") or "").strip(),
        "bearer_email": str(bearer_identity.get("email") or "").strip(),
        "bearer_name": str(bearer_identity.get("name") or "").strip(),
    }
