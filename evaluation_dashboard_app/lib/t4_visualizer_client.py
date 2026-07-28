"""HTTP client for the T4 Visualizer FastAPI server (render_frame over HTTP).

Default base URL: ``T4_VISUALIZER_BASE_URL`` environment variable, or ``http://127.0.0.1:8000``.

Does not import t4_devkit or t4_visualizer; only uses ``requests`` against the server's
``GET /health``, ``GET /server/structure.json``, ``GET /datasets``, ``GET /datasets/{id}/availability``,
``GET /datasets/{id}/scenarios``, and ``POST /render`` endpoints.
"""

from __future__ import annotations

import base64
import os
from dataclasses import asdict, dataclass, field
from typing import Any, List, Mapping, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
ENV_BASE_URL = "T4_VISUALIZER_BASE_URL"
ENV_BROWSER_BASE_URL = "T4_VISUALIZER_BROWSER_BASE_URL"
# Browser base URL to use when the dashboard itself is reached through Cloudflare.
# The browser then cannot use localhost; it must hit the dataset server's public
# Cloudflare hostname. Configure the value via this env var (kept out of the repo).
ENV_CLOUDFLARE_BASE_URL = "T4_VISUALIZER_CLOUDFLARE_BASE_URL"
ENV_CF_ACCESS_CLIENT_ID = "T4_VISUALIZER_CF_ACCESS_CLIENT_ID"
ENV_CF_ACCESS_CLIENT_SECRET = "T4_VISUALIZER_CF_ACCESS_CLIENT_SECRET"


class T4VisualizerError(Exception):
    """Raised when the T4 visualizer HTTP API returns an error or invalid response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_text: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


def format_t4_visualizer_error(exc: T4VisualizerError) -> str:
    """Return a compact user-facing message for a T4 visualizer API error."""
    status = f" ({exc.status_code})" if exc.status_code is not None else ""
    return f"T4 server error{status}: {exc}"


def _cloudflare_access_headers_from_env() -> dict[str, str]:
    client_id = os.environ.get(ENV_CF_ACCESS_CLIENT_ID, "").strip()
    client_secret = os.environ.get(ENV_CF_ACCESS_CLIENT_SECRET, "").strip()
    if not client_id or not client_secret:
        return {}
    return {
        "CF-Access-Client-Id": client_id,
        "CF-Access-Client-Secret": client_secret,
    }


@dataclass
class TargetObjectIn:
    """One object to draw on the render (matches server ``TargetObjectIn``)."""

    uuid: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    label: str = ""
    width: float = 0.0
    length: float = 0.0
    height: float = 0.0
    yaw: float = 0.0


@dataclass
class RenderRequest:
    """Request body for ``POST /render`` (matches server ``RenderRequest``)."""

    t4dataset_id: str
    scenario_name: str
    frame_index: int
    target_objects: List[TargetObjectIn] = field(default_factory=list)
    cameras: Optional[List[str]] = None
    show_annotations: bool = True
    version: Optional[str] = None
    crop_cameras: bool = False
    crop_padding: int = 40
    crop_min_size: int = 300


@dataclass
class ImageResult:
    """One rendered PNG in the response."""

    label: str
    png_base64: str


@dataclass
class RenderResult:
    """Parsed ``POST /render`` JSON response."""

    sample_token: str
    timestamp_us: int
    images: List[ImageResult]
    raw_json: Optional[dict] = None
    # Optional server-reported timings (newer t4-server JSON body)
    elapsed_ms: Optional[float] = None
    tier4_load_ms: Optional[float] = None
    render_ms: Optional[float] = None

    def decode_png(self, label: str) -> bytes:
        """Decode base64 PNG bytes for the image with the given label."""
        for img in self.images:
            if img.label == label:
                return base64.b64decode(img.png_base64)
        raise KeyError(f"No image with label {label!r}")

    def decode_all_images(self) -> List[Tuple[str, bytes]]:
        """Decode all images to ``(label, png_bytes)``."""
        return [(img.label, base64.b64decode(img.png_base64)) for img in self.images]


def render_response_json_for_debug(
    data: Mapping[str, Any], *, max_b64_preview: int = 120
) -> dict[str, Any]:
    """Copy of a ``POST /render`` JSON object with ``png_base64`` truncated for UI/debug."""
    out: dict[str, Any] = dict(data)
    imgs = out.get("images")
    if not isinstance(imgs, list):
        return out
    trimmed: list[Any] = []
    for item in imgs:
        if not isinstance(item, dict):
            trimmed.append(item)
            continue
        row = dict(item)
        b64 = row.get("png_base64")
        if isinstance(b64, str) and len(b64) > max_b64_preview:
            row["png_base64"] = f"{b64[:max_b64_preview]}…"
            row["png_base64_len"] = len(b64)
        trimmed.append(row)
    out["images"] = trimmed
    return out


def _default_base_url() -> str:
    return os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL).rstrip("/")


def _dashboard_accessed_via_cloudflare() -> bool:
    """True if the current Streamlit request arrived through Cloudflare.

    Lazy import keeps this HTTP client free of a hard Streamlit dependency; returns
    False in any non-Streamlit / header-less context.
    """
    try:
        from lib.auth import detect_access_origin

        return bool(detect_access_origin().get("is_cloudflare"))
    except Exception:
        return False


def browser_base_url(api_base_url: str | None = None) -> str:
    """Return the T4 URL that the user's browser should open.

    `T4_VISUALIZER_BASE_URL` is used by Python running inside Streamlit. In Docker,
    that often needs to be `http://host.docker.internal:8000`, but browsers on the
    host cannot resolve that Docker-only hostname. Use
    `T4_VISUALIZER_BROWSER_BASE_URL` for iframe/link URLs; if unset, translate the
    common Docker hostname back to localhost.

    When the dashboard itself is reached through Cloudflare, the browser must instead
    hit the dataset server's public Cloudflare hostname; if `T4_VISUALIZER_CLOUDFLARE_BASE_URL`
    is set and the request came via Cloudflare, that URL wins over everything else.
    """
    cloudflare = os.environ.get(ENV_CLOUDFLARE_BASE_URL, "").strip()
    if cloudflare and _dashboard_accessed_via_cloudflare():
        return cloudflare.rstrip("/")
    explicit = os.environ.get(ENV_BROWSER_BASE_URL, "").strip()
    if explicit:
        return explicit.rstrip("/")
    raw = (api_base_url or _default_base_url()).strip() or DEFAULT_BASE_URL
    parsed = urlparse(raw)
    if parsed.hostname == "host.docker.internal":
        netloc = "localhost"
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urlunparse(parsed._replace(netloc=netloc)).rstrip("/")
    return raw.rstrip("/")


def _serialize_target_object(o: TargetObjectIn) -> dict:
    d = asdict(o)
    return d


def render_request_to_json_body(req: RenderRequest) -> dict:
    """Build a JSON-serializable dict for ``POST /render``."""
    out: dict = {
        "t4dataset_id": req.t4dataset_id,
        "scenario_name": req.scenario_name,
        "frame_index": req.frame_index,
        "target_objects": [_serialize_target_object(o) for o in req.target_objects],
        "show_annotations": req.show_annotations,
        "crop_cameras": req.crop_cameras,
        "crop_padding": req.crop_padding,
        "crop_min_size": req.crop_min_size,
    }
    if req.cameras is not None:
        out["cameras"] = req.cameras
    if req.version is not None:
        out["version"] = req.version
    return out


def target_object_from_gt_row(row: Mapping[str, Any]) -> dict:
    """Map a GT / eval parquet row to one ``target_objects`` entry for ``RenderRequest``.

    Uses ``uuid`` or ``gt_uuid`` for the instance id; position from ``x``, ``y``, ``z``;
    optional bbox fields default to ``0.0`` when missing.
    """
    raw_id = row.get("uuid")
    if raw_id is None or raw_id == "":
        raw_id = row.get("gt_uuid")
    uuid_str = "" if raw_id is None else str(raw_id)

    def _float(key: str, default: float = 0.0) -> float:
        v = row.get(key)
        if v is None:
            return default
        return float(v)

    return {
        "uuid": uuid_str,
        "x": _float("x"),
        "y": _float("y"),
        "z": _float("z"),
        "label": str(row.get("label") or ""),
        "width": _float("width"),
        "length": _float("length"),
        "height": _float("height"),
        "yaw": _float("yaw"),
    }


class T4VisualizerClient:
    """Thin HTTP client for the T4 Visualizer server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout: float = 120.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        raw = base_url if base_url is not None else _default_base_url()
        self.base_url = raw.rstrip("/")
        self.timeout = timeout
        self._session = session if session is not None else requests.Session()
        self._headers = _cloudflare_access_headers_from_env()

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _raise_for_status(self, resp: requests.Response) -> None:
        if resp.ok:
            return
        text = (resp.text or "")[:2000]
        raise T4VisualizerError(
            f"T4 visualizer HTTP {resp.status_code}: {text[:500]}",
            status_code=resp.status_code,
            response_text=text,
        )

    def _json_or_raise(self, resp: requests.Response, endpoint: str) -> dict:
        try:
            data = resp.json()
        except ValueError as exc:
            text = (resp.text or "")[:2000]
            preview = "empty body"
            if text.strip():
                preview = " ".join(text.strip().split())
                if len(preview) > 500:
                    preview = f"{preview[:500]}..."
            headers = getattr(resp, "headers", {}) or {}
            content_type = ""
            if isinstance(headers, Mapping):
                content_type = str(
                    headers.get("content-type") or headers.get("Content-Type") or ""
                )
            meta: list[str] = []
            if getattr(resp, "status_code", None) is not None:
                meta.append(f"status={resp.status_code}")
            if content_type:
                meta.append(f"content-type={content_type}")
            suffix = f" ({', '.join(meta)})" if meta else ""
            cloudflare_hint = ""
            if "cloudflare access" in text.lower() or "cf-access" in text.lower():
                cloudflare_hint = (
                    "; Cloudflare Access returned a sign-in page, so the Streamlit backend "
                    "could not reach the T4 JSON API. Use a direct T4 API URL or set "
                    f"{ENV_CF_ACCESS_CLIENT_ID}/{ENV_CF_ACCESS_CLIENT_SECRET} service-token env vars"
                )
            raise T4VisualizerError(
                (
                    f"Invalid JSON from {endpoint}{suffix}{cloudflare_hint}; "
                    f"response preview: {preview}"
                ),
                status_code=getattr(resp, "status_code", None),
                response_text=text,
            ) from exc
        if not isinstance(data, dict):
            raise T4VisualizerError(
                f"Unexpected JSON from {endpoint}: expected object, got {type(data).__name__}",
                status_code=getattr(resp, "status_code", None),
                response_text=(resp.text or "")[:2000],
            )
        return data

    def health(self) -> dict:
        """GET /health — status, ``service``, ``version``, ``data_dir_exists``, structure paths (newer servers)."""
        resp = self._session.get(
            self._url("/health"),
            headers=self._headers or None,
            timeout=self.timeout,
        )
        print(resp.text)
        self._raise_for_status(resp)
        return self._json_or_raise(resp, "/health")

    def server_structure_json(self) -> dict:
        """GET /server/structure.json — Mermaid source for the server internals plus cache/runtime meta."""
        to = min(30.0, float(self.timeout))
        resp = self._session.get(
            self._url("/server/structure.json"),
            headers=self._headers or None,
            timeout=to,
        )
        self._raise_for_status(resp)
        return self._json_or_raise(resp, "/server/structure.json")

    def list_datasets(self) -> dict:
        """GET /datasets — returns at least ``data_dir`` and ``datasets``."""
        resp = self._session.get(
            self._url("/datasets"),
            headers=self._headers or None,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return self._json_or_raise(resp, "/datasets")

    def list_dataset_scenarios(
        self, t4dataset_id: str, version: Optional[str] = None
    ) -> dict:
        """GET /datasets/{t4dataset_id}/scenarios — scene names and ``nbr_samples`` (frame counts).

        Response keys typically include ``t4dataset_id``, ``scenarios`` (list of dicts with
        ``name``, ``token``, ``description``, ``nbr_samples``), and optional ``version``.
        """
        from urllib.parse import quote

        tid = quote(str(t4dataset_id), safe="")
        params = {"version": version} if version is not None else None
        resp = self._session.get(
            self._url(f"/datasets/{tid}/scenarios"),
            headers=self._headers or None,
            params=params,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return self._json_or_raise(resp, "/datasets/.../scenarios")

    def dataset_availability(self, t4dataset_id: str) -> dict:
        """GET /datasets/{t4dataset_id}/availability — whether the dataset is on disk for this server.

        Typical JSON: ``t4dataset_id``, ``available`` (bool), ``dataset_path`` (str or null).
        """
        from urllib.parse import quote

        tid = quote(str(t4dataset_id), safe="")
        resp = self._session.get(
            self._url(f"/datasets/{tid}/availability"),
            headers=self._headers or None,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return self._json_or_raise(resp, "/datasets/.../availability")

    def render(self, payload: RenderRequest) -> RenderResult:
        """POST /render with a :class:`RenderRequest`."""
        body = render_request_to_json_body(payload)
        resp = self._session.post(
            self._url("/render"),
            headers=self._headers or None,
            json=body,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        data = self._json_or_raise(resp, "/render")

        try:
            images_raw = data["images"]
            imgs = [
                ImageResult(label=str(x["label"]), png_base64=str(x["png_base64"]))
                for x in images_raw
            ]

            def _opt_float(key: str) -> Optional[float]:
                v = data.get(key)
                if v is None:
                    return None
                return float(v)

            return RenderResult(
                sample_token=str(data["sample_token"]),
                timestamp_us=int(data["timestamp_us"]),
                images=imgs,
                raw_json=dict(data),
                elapsed_ms=_opt_float("elapsed_ms"),
                tier4_load_ms=_opt_float("tier4_load_ms"),
                render_ms=_opt_float("render_ms"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise T4VisualizerError(f"Unexpected /render response shape: {data!r}") from exc
