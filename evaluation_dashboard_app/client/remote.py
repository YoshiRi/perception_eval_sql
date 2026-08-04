"""HTTP client for the server's export API.

Deliberately stdlib-only (``urllib``) to match the dependency-light backend and to
keep the frozen bundle small; nothing here needs what ``requests`` adds.
"""

from __future__ import annotations

import json
import shutil
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from client.config import Config

# Candidate suffixes tried when probing a bare hostname. Behind nginx the API is
# mounted at /bbox-api; hit directly it is at the root of port 8765.
PROBE_SUFFIXES = ("", "/bbox-api")

CF_ACCESS_HELP = (
    "The server is behind Cloudflare Access and this app has no valid session. "
    "Sign in with `evaldash-local login --cf-login`, or set a service token in Settings "
    "(--cf-client-id / --cf-client-secret) for an unattended machine."
)
CF_LOGIN_TIMEOUT_SEC = 300


class RemoteError(RuntimeError):
    """A request to the server failed."""


class AuthError(RemoteError):
    """Server rejected the token, or exports are disabled there."""


def _is_access_url(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host.endswith("cloudflareaccess.com") or "/cdn-cgi/access/login" in url


class _AccessAwareRedirect(urllib.request.HTTPRedirectHandler):
    """Stop at the Access sign-in bounce instead of downloading its HTML.

    Following it yields a 200 full of login markup, which surfaces as "Non-JSON reply"
    -- indistinguishable from a broken server. Fail here, where the cause is still known.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if _is_access_url(newurl):
            raise AuthError(CF_ACCESS_HELP)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Access JWTs are minted by `cloudflared access login` and cached by cloudflared itself
# under ~/.cloudflared. Reading that cache is what lets a personal sign-in -- done once,
# in a browser -- carry this app for the life of the session, with no secret to store.
_CF_SESSION: dict[str, str] = {}


def _cloudflared() -> str | None:
    return shutil.which("cloudflared")


def cf_session_token(base_url: str, *, refresh: bool = False) -> str:
    """The cached Access JWT for this host, or "" when there is no usable session."""
    host = urllib.parse.urlsplit(base_url)
    app = f"{host.scheme}://{host.netloc}"
    if not refresh and app in _CF_SESSION:
        return _CF_SESSION[app]
    binary = _cloudflared()
    if not binary:
        return ""
    try:
        done = subprocess.run([binary, "access", "token", f"-app={app}"],
                              capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return ""
    out = (done.stdout or "").strip()
    # A miss is prose on stdout ("Unable to find token..."), so shape-check the result
    # rather than trusting the exit code.
    token = out if done.returncode == 0 and out.count(".") == 2 and " " not in out else ""
    _CF_SESSION[app] = token
    return token


def cf_browser_login(base_url: str) -> str:
    """Run the interactive Cloudflare sign-in, then return the JWT it minted."""
    binary = _cloudflared()
    if not binary:
        raise AuthError(
            "A browser sign-in needs `cloudflared`, which is not installed. Install it from "
            "Cloudflare's downloads page, or use a service token instead."
        )
    host = urllib.parse.urlsplit(base_url)
    app = f"{host.scheme}://{host.netloc}"
    try:
        subprocess.run([binary, "access", "login", app], timeout=CF_LOGIN_TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired:
        raise AuthError(
            f"The sign-in did not finish within {CF_LOGIN_TIMEOUT_SEC}s. "
            f"Run it directly: cloudflared access login {app}"
        ) from None
    return cf_session_token(app, refresh=True)


class Remote:
    def __init__(self, config: Config, base_url: str | None = None) -> None:
        self.config = config
        self.base_url = (base_url or config.require_server()).rstrip("/")
        self._ssl_context = None if config.verify_tls else ssl._create_unverified_context()
        handlers: list[Any] = [_AccessAwareRedirect]
        if self._ssl_context is not None:
            handlers.append(urllib.request.HTTPSHandler(context=self._ssl_context))
        self._opener = urllib.request.build_opener(*handlers)

    # ------------------------------------------------------------------ plumbing

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "evaldash-local/1"}
        token = self.config.resolved_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Cloudflare Access service token, mirroring how the dashboard authenticates
        # its own outbound calls to the T4 visualizer. Failing that, a browser session
        # the user established with `cloudflared access login` -- the service token is
        # preferred because it needs no browser and never expires mid-download.
        if self.config.cf_client_id and self.config.cf_client_secret:
            headers["CF-Access-Client-Id"] = self.config.cf_client_id
            headers["CF-Access-Client-Secret"] = self.config.cf_client_secret
        else:
            session = cf_session_token(self.base_url)
            if session:
                headers["Cookie"] = f"CF_Authorization={session}"
        if extra:
            headers.update(extra)
        return headers

    def _open(self, request: urllib.request.Request, timeout: float | None = None):
        try:
            return self._opener.open(request, timeout=timeout or self.config.timeout_sec)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            detail = body
            try:
                parsed = json.loads(body)
                detail = parsed.get("error") or body
            except Exception:
                pass
            # Access rejects at the edge, so its refusals must be read before the
            # generic 401/403 branch turns them into "the server said no".
            lowered = body.lower()
            edge_refusal = (
                "cloudflare-access" in str(exc.headers.get("WWW-Authenticate", "")).lower()
                or "error code: 1010" in lowered
                or ("<html" in lowered and "cloudflare" in lowered)
            )
            if edge_refusal:
                raise AuthError(CF_ACCESS_HELP) from exc
            if exc.code in (401, 403):
                raise AuthError(f"HTTP {exc.code}: {detail or 'unauthorized'}") from exc
            if exc.code == 503:
                raise AuthError(f"HTTP 503: {detail or 'export API disabled on server'}") from exc
            raise RemoteError(f"HTTP {exc.code} from {request.full_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RemoteError(f"Cannot reach {request.full_url}: {exc.reason}") from exc

    def post_json(self, route: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{route}",
            data=body,
            method="POST",
            headers=self._headers({"Content-Type": "application/json"}),
        )
        with self._open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RemoteError(f"Non-JSON reply from {route}: {raw[:200]}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise RemoteError(str(data["error"]))
        if not isinstance(data, dict):
            raise RemoteError(f"Unexpected reply shape from {route}: {type(data).__name__}")
        return data

    # -------------------------------------------------------------------- routes

    def export_health(self) -> dict[str, Any]:
        return self.post_json("/api/export_health", {})

    def runs(self, *, sizes: bool = True, query: str = "") -> dict[str, Any]:
        return self.post_json("/api/runs", {"sizes": sizes, "q": query}, timeout=180.0)

    def manifest(
        self,
        run: str,
        *,
        role: str = "all",
        tier: str = "criteria",
        include_future: bool = False,
        checksums: bool = False,
    ) -> dict[str, Any]:
        return self.post_json(
            "/api/export_manifest",
            {
                "run": run,
                "role": role,
                "tier": tier,
                "include_future": include_future,
                "checksums": checksums,
            },
            timeout=600.0,
        )

    def prebake(
        self, run: str, parquet: str, *, routes: list[str] | None = None, force: bool = False,
        status_only: bool = False, timeout: float = 7200.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"run": run, "parquet": parquet, "force": force}
        if status_only:
            payload["status_only"] = True
        if routes:
            payload["routes"] = routes
        return self.post_json("/api/export_prebake", payload, timeout=timeout)

    # ---------------------------------------------------------------- workflow routes

    def workflow_health(self, *, target_name: str = "") -> dict[str, Any]:
        return self.post_json("/api/workflow_health", {"target_name": target_name})

    def workflow_catalogs(
        self,
        *,
        project_id: str = "",
        environment: str = "",
        refresh: bool = False,
        resolve_catalog_id: str = "",
    ) -> dict[str, Any]:
        return self.post_json(
            "/api/workflow_catalogs",
            {
                "project_id": project_id,
                "environment": environment,
                "refresh": refresh,
                "resolve_catalog_id": resolve_catalog_id,
            },
            # Both optional lookups call the evaluator API from the server side.
            timeout=180.0,
        )

    def workflow_start(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.post_json("/api/workflow_start", params, timeout=180.0)

    def workflow_tasks(
        self, *, limit: int = 25, since_days: int | None = 7, mine: str = ""
    ) -> dict[str, Any]:
        return self.post_json(
            "/api/workflow_tasks",
            {"limit": limit, "since_days": since_days, "mine": mine},
            timeout=120.0,
        )

    def workflow_task(self, task_id: str) -> dict[str, Any]:
        return self.post_json("/api/workflow_task", {"task_id": task_id}, timeout=120.0)

    def workflow_cancel(self, task_id: str) -> dict[str, Any]:
        return self.post_json("/api/workflow_cancel", {"task_id": task_id}, timeout=120.0)

    def open_file(self, run: str, rel_path: str, *, offset: int = 0):
        """Open a byte stream for one exported file, optionally resuming at ``offset``."""
        query = urllib.parse.urlencode({"run": run, "rel_path": rel_path})
        headers = self._headers({"Accept": "application/octet-stream"})
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(
            f"{self.base_url}/api/export_file?{query}", method="GET", headers=headers
        )
        response = self._open(request, timeout=max(self.config.timeout_sec, 300.0))
        if offset > 0 and response.status != 206:
            # Server ignored the range; the caller must restart from zero rather than
            # append a full body onto a partial file.
            response.close()
            raise RemoteError(
                f"Server did not honour Range for {rel_path} (HTTP {response.status}); restart the download."
            )
        return response


def connect(cfg: Config) -> Remote:
    """Build a Remote against a base URL that is known to serve the export API.

    A URL that was never run through :func:`probe_server` may be missing the ``/bbox-api``
    prefix that nginx mounts the API under, in which case requests land on Streamlit and
    come back as ``405 Method Not Allowed``. That is the normal state for a build-time
    default or ``EVALDASH_SERVER``, where no ``login`` step ever ran, so resolve here and
    persist the answer once rather than re-probing on every call.
    """
    source = cfg.server_source()
    if source == "stored":
        return Remote(cfg)
    resolved, _ = probe_server(cfg, cfg.require_server())
    if source == "default":
        # Cache the build default's resolved form so later runs skip the probe. An env
        # override is deliberately not persisted: it is meant to be transient, and
        # writing it would silently become the new saved setting.
        cfg.server_url = resolved
        try:
            cfg.save()
        except OSError:
            # A read-only home is survivable: this run works, the next one re-probes.
            pass
    return Remote(cfg, base_url=resolved)


def probe_server(config: Config, base_url: str) -> tuple[str, dict[str, Any]]:
    """Find the URL that actually serves the export API, tolerating a bare hostname.

    Returns the working base URL and its health payload.
    """
    root = base_url.rstrip("/")
    attempts: list[str] = []
    errors: list[str] = []
    for suffix in PROBE_SUFFIXES:
        candidate = f"{root}{suffix}" if not root.endswith(suffix) or not suffix else root
        if candidate in attempts:
            continue
        attempts.append(candidate)
        try:
            health = Remote(config, base_url=candidate).export_health()
        except RemoteError as exc:
            errors.append(f"  {candidate}: {exc}")
            continue
        if health.get("service") == "eval_dashboard_export":
            return candidate, health
        errors.append(f"  {candidate}: not an export API ({health})")
    raise RemoteError("Could not find the export API. Tried:\n" + "\n".join(errors))
