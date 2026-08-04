"""A stand-in for t4-server, faithful to the contract the real one exposes.

The real service lives in a different repo and needs a multi-terabyte dataset mount, so
it cannot run in CI or on a laptop. This reproduces the parts the offline cache depends
on -- the exact ``T4V3D002`` wire format, the same query parameters, the same
``X-T4V-*`` headers, and a page whose fetches are root-relative -- so the mirror can be
tested for real rather than mocked.

Kept in ``tests/`` because it is test scaffolding, not shipped code.
"""

from __future__ import annotations

import json
import math
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

# Mirrors t4_visualizer/server.py::_pack_viewer_frame_binary.
FRAME_HEADER_STRUCT = "<8sIIQIIH"
FIXED_HEADER_BYTES = 34

# The page's own assets, which it loads from the server rather than inlining. The theme
# module is fatal when missing -- the real page dies on "TH is not defined".
PAGE_ASSETS = {
    "/static/t4_theme.js": (b"window.TH = {hex: () => 0};\n", "application/javascript"),
    "/static/favicon.svg": (b"<svg xmlns='http://www.w3.org/2000/svg'/>", "image/svg+xml"),
    "/viewer/assets/vehicle-mesh/lexus.dae": (b"<COLLADA/>", "model/vnd.collada+xml"),
}

# A page whose fetches are root-relative, like the real viewer_three.html.
PAGE_TEMPLATE = """<!doctype html>
<meta charset="utf-8"><title>T4 3D — __SCENARIO_NAME__</title>
<link rel="icon" href="/static/favicon.svg">
<script src="/static/t4_theme.js?v=test-1"></script>
<body data-dataset="__DATASET_ID__">
<script>
const EGO_MESH = "/viewer/assets/vehicle-mesh/lexus.dae";
const params = new URLSearchParams("__QS__");
const dataset = params.get("t4dataset_id");
const scenario = params.get("scenario_name");
async function loadFrame(i) {
  const url = `/viewer/three/frame.bin?t4dataset_id=${encodeURIComponent(dataset)}`
    + `&scenario_name=${encodeURIComponent(scenario)}&frame_index=${i}`;
  const res = await fetch(url);
  return res.arrayBuffer();
}
</script>
</body>
"""


def pack_frame(frame_index: int, point_count: int, box_count: int = 2) -> bytes:
    """Build a byte-exact T4V3D002 payload with deterministic contents."""
    token = f"sample-token-{frame_index:04d}"
    token_bytes = token.encode("utf-8")
    header = struct.pack(
        FRAME_HEADER_STRUCT,
        b"T4V3D002",
        FIXED_HEADER_BYTES,
        int(frame_index),
        1_700_000_000_000_000 + frame_index * 100_000,
        int(point_count),
        int(box_count),
        len(token_bytes),
    ) + token_bytes

    points = bytearray()
    for i in range(point_count):
        angle = (i / max(point_count, 1)) * math.tau
        points += struct.pack(
            "<ffff",
            math.cos(angle) * (10.0 + frame_index),
            math.sin(angle) * (10.0 + frame_index),
            float(i % 5) * 0.25,
            float(i % 255) / 255.0,
        )

    boxes = bytearray()
    for b in range(box_count):
        for corner in range(8):
            boxes += struct.pack(
                "<fff",
                float(b * 4 + (corner & 1)),
                float(frame_index + ((corner >> 1) & 1)),
                float((corner >> 2) & 1),
            )
    labels = json.dumps([f"label_{b}" for b in range(box_count)], ensure_ascii=True).encode("utf-8")
    return bytes(header + points + boxes + struct.pack("<I", len(labels)) + labels)


class FakeT4Handler(BaseHTTPRequestHandler):
    frames_total = 6
    points_per_frame = 400
    scenario_name = "scene-alpha"
    dataset_id = "T4DS0001"
    hits: dict[str, int] = {}

    def log_message(self, *_args) -> None:  # keep test output readable
        return

    # ---------------------------------------------------------------- helpers

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _count(self, path: str) -> None:
        FakeT4Handler.hits[path] = FakeT4Handler.hits.get(path, 0) + 1

    def _param(self, query, name, default=None):
        values = query.get(name) or []
        return values[-1] if values else default

    # ---------------------------------------------------------------- routing

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        if length:
            self.rfile.read(length)
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path
        self._count(path)

        if path == "/health":
            self._json({"status": "ok", "visibility_mode": "public"})
            return

        if path.endswith("/scenarios"):
            self._json({"scenarios": [{"name": self.scenario_name, "nbr_samples": self.frames_total}]})
            return

        if path.endswith("/availability"):
            self._json({"t4dataset_id": self.dataset_id, "available": True, "dataset_path": "/fake"})
            return

        if path in PAGE_ASSETS:
            body, content_type = PAGE_ASSETS[path]
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/viewer/three":
            qs = urlencode(
                {
                    "t4dataset_id": self._param(query, "t4dataset_id", ""),
                    "frame_index": self._param(query, "frame_index", "0"),
                    "scenario_name": self._param(query, "scenario_name", ""),
                }
            )
            page = (
                PAGE_TEMPLATE
                .replace("__DATASET_ID__", self._param(query, "t4dataset_id", ""))
                .replace("__SCENARIO_NAME__", self._param(query, "scenario_name", "") or "(auto)")
                .replace("__QS__", qs)
            )
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/viewer/three/meta":
            self._json(
                {
                    "t4dataset_id": self._param(query, "t4dataset_id"),
                    "scenario_name": self.scenario_name,
                    "nbr_samples": self.frames_total,
                    "format_version": "T4V3D002",
                    "binary_endpoint_template": "/viewer/three/frame.bin?frame_index={frame_index}",
                }
            )
            return

        if path == "/viewer/three/frame.bin":
            index = int(float(self._param(query, "frame_index", "0")))
            if index < 0 or index >= self.frames_total:
                self._json({"detail": "frame out of range"}, status=404)
                return
            blob = pack_frame(index, self.points_per_frame)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("X-T4V-Format", "T4V3D002")
            self.send_header("X-T4V-Frame-Index", str(index))
            self.send_header("X-T4V-Sample-Token", f"sample-token-{index:04d}")
            self.send_header("X-T4V-Point-Fields", "x,y,z,intensity")
            self.send_header("X-T4V-Box-Fields", "8corners_xyz")
            self.end_headers()
            self.wfile.write(blob)
            return

        if path == "/viewer/three/frames/window":
            center = int(float(self._param(query, "center", "0")))
            radius = int(float(self._param(query, "radius", "2")))
            low, high = max(0, center - radius), min(self.frames_total - 1, center + radius)
            self._json({"total": self.frames_total,
                        "frames": [{"frame_index": i} for i in range(low, high + 1)]})
            return

        if path == "/viewer/three/lanelet-lines":
            index = int(float(self._param(query, "frame_index", "0")))
            self._json({"frame_index": index,
                        "segments": [[[0.0, 0.0], [float(index), 1.0]]],
                        "max_segments": int(self._param(query, "max_segments", "0"))})
            return

        if path == "/viewer/three/camera-overlay":
            index = int(float(self._param(query, "frame_index", "0")))
            self._json({"frame_index": index, "cameras": [
                {"channel": "CAM_FRONT", "image_base64": "ZmFrZQ==", "image_format": "jpeg", "boxes": []}]})
            return

        if path == "/viewer/three/camera-info":
            index = int(float(self._param(query, "frame_index", "0")))
            self._json({"frame_index": index,
                        "cameras": [{"channel": "CAM_FRONT", "intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]})
            return

        if path.startswith("/viewer/three/debug/"):
            self._json({"ok": True})
            return

        self._json({"detail": f"no route {path}"}, status=404)


class FakeT4Server:
    """Context manager that runs :class:`FakeT4Handler` on a loopback port."""

    def __init__(self, frames: int = 6, points: int = 400) -> None:
        FakeT4Handler.frames_total = frames
        FakeT4Handler.points_per_frame = points
        FakeT4Handler.hits = {}
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeT4Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "FakeT4Server":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def hits(self) -> dict[str, int]:
        return dict(FakeT4Handler.hits)
