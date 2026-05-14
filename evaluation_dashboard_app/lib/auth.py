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
