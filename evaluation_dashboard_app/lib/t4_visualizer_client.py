"""HTTP client for the T4 Visualizer FastAPI server (render_frame over HTTP).

Default base URL: ``T4_VISUALIZER_BASE_URL`` environment variable, or ``http://127.0.0.1:8000``.

Does not import t4_devkit or t4_visualizer; only uses ``requests`` against the server's
``GET /health``, ``GET /datasets``, and ``POST /render`` endpoints.
"""

from __future__ import annotations

import base64
import os
from dataclasses import asdict, dataclass, field
from typing import Any, List, Mapping, Optional, Tuple

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
ENV_BASE_URL = "T4_VISUALIZER_BASE_URL"


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

    def decode_png(self, label: str) -> bytes:
        """Decode base64 PNG bytes for the image with the given label."""
        for img in self.images:
            if img.label == label:
                return base64.b64decode(img.png_base64)
        raise KeyError(f"No image with label {label!r}")

    def decode_all_images(self) -> List[Tuple[str, bytes]]:
        """Decode all images to ``(label, png_bytes)``."""
        return [(img.label, base64.b64decode(img.png_base64)) for img in self.images]


def _default_base_url() -> str:
    return os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL).rstrip("/")


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

    def health(self) -> dict:
        """GET /health."""
        resp = self._session.get(self._url("/health"), timeout=self.timeout)
        self._raise_for_status(resp)
        try:
            return resp.json()
        except ValueError as exc:
            raise T4VisualizerError("Invalid JSON from /health") from exc

    def list_datasets(self) -> dict:
        """GET /datasets — returns at least ``data_dir`` and ``datasets``."""
        resp = self._session.get(self._url("/datasets"), timeout=self.timeout)
        self._raise_for_status(resp)
        try:
            return resp.json()
        except ValueError as exc:
            raise T4VisualizerError("Invalid JSON from /datasets") from exc

    def render(self, payload: RenderRequest) -> RenderResult:
        """POST /render with a :class:`RenderRequest`."""
        body = render_request_to_json_body(payload)
        resp = self._session.post(
            self._url("/render"),
            json=body,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        try:
            data = resp.json()
        except ValueError as exc:
            raise T4VisualizerError("Invalid JSON from /render") from exc

        try:
            images_raw = data["images"]
            imgs = [
                ImageResult(label=str(x["label"]), png_base64=str(x["png_base64"]))
                for x in images_raw
            ]
            return RenderResult(
                sample_token=str(data["sample_token"]),
                timestamp_us=int(data["timestamp_us"]),
                images=imgs,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise T4VisualizerError(f"Unexpected /render response shape: {data!r}") from exc
