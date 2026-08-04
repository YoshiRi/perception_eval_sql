"""Offline cache for the t4-server 3D scene viewer.

Point clouds do not come from the dashboard. They come from ``t4-server``, a separate
FastAPI service that reads T4 datasets off a shared mount, and the browser has always
fetched them from it directly -- the dashboard never proxied that traffic. So making 3D
work offline means caching t4-server's responses and replaying them locally.

Two facts make this tractable:

* every URL the viewer page fetches is **root-relative** (``/viewer/three/...``), so a
  mirror serving those same paths needs no change to the page;
* the page is a template whose placeholders are already substituted by the time it
  reaches the browser, so it can be stored verbatim alongside the data.

Sizing is the real constraint. Frames are ``float32[4]`` per point with no decimation
option, so one frame is roughly 1.6-3.2 MB and a 100-frame scene 200-300 MB. Caching is
therefore explicit and per-scenario, never automatic.

Layout::

    ~/.evaldash/t4/<dataset_id>/<scenario>/
      manifest.json           # frame count, version, sizes, what was fetched
      page.html               # /viewer/three as served, placeholders already filled
      meta.json               # /viewer/three/meta
      frames/<i>.bin          # /viewer/three/frame.bin (verbatim T4V3D002 bytes)
      frames/<i>.hdr.json     # its X-T4V-* response headers
      lanelet/<i>.json        # /viewer/three/lanelet-lines
      overlay/<i>__<hash>.json  # /viewer/three/camera-overlay (GET variant)
      caminfo/<i>__<hash>.json  # /viewer/three/camera-info
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from client import config, remote

FRAME_MAGIC = b"T4V3D002"

# Fixed header of t4_visualizer's _pack_viewer_frame_binary: magic[8], header_len u32,
# frame_index u32, timestamp_us u64, point_count u32, box_count u32, token_len u16.
# 34 bytes, which is also the value the server stores in header_len.
FRAME_HEADER_STRUCT = "<8sIIQIIH"
FRAME_HEADER_BYTES = struct.calcsize(FRAME_HEADER_STRUCT)

# Values viewer_three.html hardcodes when it fetches lanelet lines, so a cached scene
# answers the request the page actually makes.
LANELET_PARAMS = {"max_segments": "90000", "clip_radius_m": "170"}

# Headers worth preserving: the page reads the format and frame identity from them.
FRAME_HEADER_PREFIX = "x-t4v-"


class T4Error(RuntimeError):
    """A request to t4-server failed, or the cache cannot answer one."""


def _qs_hash(params: dict[str, Any]) -> str:
    blob = "&".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _safe_name(value: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(value))
    return keep[:120] or "_"


def t4_root() -> Path:
    return config.t4_cache_dir()


def scene_dir(dataset_id: str, scenario: str) -> Path:
    return t4_root() / _safe_name(dataset_id) / _safe_name(scenario)


def human_bytes(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(count) < 1024.0 or unit == "TB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024.0
    return f"{count:.1f} TB"


# --------------------------------------------------------------------------- client


class T4Client:
    """Minimal HTTP client for t4-server, with Cloudflare service-token support."""

    def __init__(self, base_url: str, cfg: config.Config | None = None, timeout: float = 180.0) -> None:
        base = (base_url or "").strip().rstrip("/")
        if not base:
            raise T4Error(
                "No T4 visualizer URL configured. Set one with:\n"
                "  evaldash-local login --server <dashboard> --t4-base-url http://<host>:8000"
            )
        self.base_url = base
        self.cfg = cfg or config.Config.load()
        self.timeout = timeout
        # t4-server sits behind the same Cloudflare Access edge as the dashboard, so it
        # gets the same opener: one that refuses to follow the sign-in bounce, and that
        # honours `verify_tls`.
        self._opener = remote.access_opener(self.cfg)

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": "evaldash-local/1", **remote.access_headers(self.cfg, self.base_url)}

    def _access_error(self) -> "T4Error":
        return T4Error(
            f"{self.base_url} is behind Cloudflare Access and this app has no valid session. "
            "Sign in from Home > 3D scenes, or run "
            f"`cloudflared access login {self.base_url}`, or set a service token in Settings."
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> tuple[bytes, dict[str, str]]:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, method="GET", headers=self._headers())
        try:
            # The opener carries the TLS context, so no `context=` here.
            with self._opener.open(request, timeout=self.timeout) as response:
                return response.read(), {k.lower(): v for k, v in response.headers.items()}
        except remote.AuthError as exc:
            # Access answers an unauthenticated request with a 302 to its login page;
            # the opener stops there, which is the only place the cause is still legible.
            raise self._access_error() from exc
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            lowered = body.lower()
            # A hard Access refusal (no redirect) still arrives as HTML or error 1010.
            if "error code: 1010" in lowered or ("<html" in lowered and "cloudflare" in lowered):
                raise self._access_error() from exc
            raise T4Error(f"HTTP {exc.code} from {url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise T4Error(f"Cannot reach {url}: {exc.reason}") from exc

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        raw, _ = self.get(path, params)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise T4Error(f"Non-JSON reply from {path}: {raw[:200]!r}") from exc

    def health(self) -> dict[str, Any]:
        return self.get_json("/health")

    def scenarios(self, dataset_id: str) -> Any:
        return self.get_json(f"/datasets/{urllib.parse.quote(dataset_id)}/scenarios")


def parse_frame_header(blob: bytes) -> dict[str, Any]:
    """Read the T4V3D002 fixed header, to validate and report a cached frame.

    Layout (little-endian): magic[8], header_len u32, frame_index u32, timestamp_us u64,
    point_count u32, box_count u32, token_len u16, then the token bytes.
    """
    if len(blob) < FRAME_HEADER_BYTES or not blob.startswith(FRAME_MAGIC):
        raise T4Error("Not a T4V3D002 frame payload")
    (
        _magic,
        header_len,
        frame_index,
        timestamp_us,
        point_count,
        box_count,
        token_len,
    ) = struct.unpack(FRAME_HEADER_STRUCT, blob[:FRAME_HEADER_BYTES])
    return {
        "frame_index": int(frame_index),
        "timestamp_us": int(timestamp_us),
        "point_count": int(point_count),
        "box_count": int(box_count),
        "sample_token": blob[FRAME_HEADER_BYTES : FRAME_HEADER_BYTES + token_len].decode(
            "utf-8", "replace"
        ),
        "header_len": int(header_len),
    }


# ---------------------------------------------------------------------------- fetch


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_bytes(path, json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _scene_params(dataset_id: str, scenario: str, version: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {"t4dataset_id": dataset_id, "scenario_name": scenario}
    if version:
        params["version"] = version
    return params


def frame_count_from_meta(meta: Any) -> int:
    """Total frames in a scene, tolerating the several shapes meta can take."""
    if not isinstance(meta, dict):
        return 0
    for key in ("frame_count", "nbr_samples", "total_frames", "frames"):
        value = meta.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, list):
            return len(value)
    scene = meta.get("scene") if isinstance(meta.get("scene"), dict) else None
    if scene:
        for key in ("nbr_samples", "frame_count"):
            if isinstance(scene.get(key), int):
                return int(scene[key])
    return 0


def fetch_scene(
    dataset_id: str,
    scenario: str,
    *,
    base_url: str | None = None,
    version: str | None = None,
    frames: range | None = None,
    with_lanelet: bool = True,
    with_camera: bool = True,
    cameras_all: bool = True,
    force: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Download one scenario's 3D data into the cache.

    Resumable and incremental: a frame already on disk is skipped unless ``force``, so an
    interrupted fetch continues where it stopped.
    """
    cfg = config.Config.load()
    client = T4Client(base_url or cfg.t4_base_url, cfg)
    target = scene_dir(dataset_id, scenario)
    target.mkdir(parents=True, exist_ok=True)
    params = _scene_params(dataset_id, scenario, version)

    meta = client.get_json("/viewer/three/meta", params)
    _write_json(target / "meta.json", meta)
    total = frame_count_from_meta(meta)
    if total <= 0:
        raise T4Error(
            f"Could not determine the frame count for {scenario}. meta.json was saved for inspection."
        )

    # The page as served: placeholders are already substituted, so it can be replayed.
    page_params = dict(params)
    page_params["frame_index"] = 0
    page_bytes, _ = client.get("/viewer/three", page_params)
    _write_bytes(target / "page.html", page_bytes)
    assets = mirror_page_assets(client, page_bytes, force=force)

    wanted = frames if frames is not None else range(total)
    wanted = [i for i in wanted if 0 <= i < total]

    stats: dict[str, Any] = {
        "dataset_id": dataset_id,
        "scenario": scenario,
        "version": version,
        "frames_total": total,
        "frames_requested": len(wanted),
        "frames_fetched": 0,
        "frames_skipped": 0,
        "bytes": 0,
        "points": 0,
        "errors": [],
        "stopped_early": False,
        "with_lanelet": with_lanelet,
        "with_camera": with_camera,
        "assets": assets,
    }
    started = time.monotonic()

    for position, index in enumerate(wanted):
        if should_stop is not None and should_stop():
            stats["stopped_early"] = True
            break
        frame_path = target / "frames" / f"{index}.bin"
        if frame_path.is_file() and not force:
            stats["frames_skipped"] += 1
            stats["bytes"] += frame_path.stat().st_size
        else:
            frame_params = dict(params)
            frame_params["frame_index"] = index
            try:
                blob, headers = client.get("/viewer/three/frame.bin", frame_params)
                info = parse_frame_header(blob)
                _write_bytes(frame_path, blob)
                _write_json(
                    target / "frames" / f"{index}.hdr.json",
                    {k: v for k, v in headers.items() if k.startswith(FRAME_HEADER_PREFIX)},
                )
                stats["frames_fetched"] += 1
                stats["bytes"] += len(blob)
                stats["points"] += info["point_count"]
            except T4Error as exc:
                stats["errors"].append({"frame": index, "what": "frame.bin", "error": str(exc)})

        if with_lanelet:
            lane_path = target / "lanelet" / f"{index}.json"
            if force or not lane_path.is_file():
                lane_params = {**params, "frame_index": index, **LANELET_PARAMS}
                try:
                    raw, _ = client.get("/viewer/three/lanelet-lines", lane_params)
                    _write_bytes(lane_path, raw)
                    stats["bytes"] += len(raw)
                except T4Error as exc:
                    stats["errors"].append({"frame": index, "what": "lanelet-lines", "error": str(exc)})

        if with_camera:
            for what, path_suffix, extra in (
                ("camera-overlay", "overlay", {"all_cameras": str(cameras_all).lower(),
                                               "show_annotations": "true"}),
                ("camera-info", "caminfo", {"all_cameras": str(cameras_all).lower()}),
            ):
                call_params = {**params, "frame_index": index, **extra}
                digest = _qs_hash({k: v for k, v in call_params.items() if k != "frame_index"})
                out_path = target / path_suffix / f"{index}__{digest}.json"
                if out_path.is_file() and not force:
                    stats["bytes"] += out_path.stat().st_size
                    continue
                try:
                    raw, _ = client.get(f"/viewer/three/{what}", call_params)
                    _write_bytes(out_path, raw)
                    stats["bytes"] += len(raw)
                except T4Error as exc:
                    stats["errors"].append({"frame": index, "what": what, "error": str(exc)})

        if progress is not None:
            done = position + 1
            elapsed = max(time.monotonic() - started, 1e-6)
            progress(
                {
                    "index": done,
                    "total": len(wanted),
                    "frame": index,
                    "bytes": stats["bytes"],
                    "elapsed_sec": elapsed,
                    "eta_sec": (len(wanted) - done) * (elapsed / done) if done else None,
                }
            )

    stats["elapsed_sec"] = round(time.monotonic() - started, 1)
    _write_json(target / "manifest.json", {**stats, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return stats


def estimate_scene_bytes(
    dataset_id: str, scenario: str, *, base_url: str | None = None, version: str | None = None
) -> dict[str, Any]:
    """Fetch a single frame to size the whole scene before committing to it.

    A 100-frame scene is 200-300 MB, so the answer to "how big is this" should not be
    "start downloading and find out".
    """
    cfg = config.Config.load()
    client = T4Client(base_url or cfg.t4_base_url, cfg)
    params = _scene_params(dataset_id, scenario, version)
    meta = client.get_json("/viewer/three/meta", params)
    total = frame_count_from_meta(meta)
    probe = dict(params)
    probe["frame_index"] = 0
    blob, _ = client.get("/viewer/three/frame.bin", probe)
    info = parse_frame_header(blob)
    return {
        "frames": total,
        "frame_bytes": len(blob),
        "points_per_frame": info["point_count"],
        "estimated_bytes": len(blob) * max(total, 0),
        "note": "frame.bin only; camera overlays and lanelet lines add to this.",
    }


# ---------------------------------------------------------------------------- serve


def find_scene(dataset_id: str, scenario: str | None) -> Path | None:
    """Locate a cached scene, tolerating an absent scenario name.

    The viewer page can be opened without a scenario (t4-server resolves it), so when
    only one scenario is cached for a dataset, that is unambiguously the one meant.
    """
    root = t4_root() / _safe_name(dataset_id)
    if not root.is_dir():
        return None
    if scenario:
        candidate = root / _safe_name(scenario)
        if (candidate / "manifest.json").is_file():
            return candidate
        return None
    scenes = [child for child in sorted(root.iterdir()) if (child / "manifest.json").is_file()]
    return scenes[0] if len(scenes) == 1 else None


def cached_scenes() -> list[dict[str, Any]]:
    root = t4_root()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for scene in sorted(dataset_dir.iterdir()):
            manifest_path = scene / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            on_disk = sum(f.stat().st_size for f in scene.rglob("*") if f.is_file())
            frames = len(list((scene / "frames").glob("*.bin"))) if (scene / "frames").is_dir() else 0
            out.append(
                {
                    "dataset_id": manifest.get("dataset_id") or dataset_dir.name,
                    "scenario": manifest.get("scenario") or scene.name,
                    "frames_cached": frames,
                    "frames_total": manifest.get("frames_total"),
                    "bytes": on_disk,
                    "fetched_at": manifest.get("fetched_at", ""),
                    "complete": bool(manifest.get("frames_total")) and frames >= int(manifest.get("frames_total") or 0),
                    "path": str(scene),
                }
            )
    return out


def remove_scene(dataset_id: str, scenario: str) -> tuple[bool, str]:
    import shutil

    target = scene_dir(dataset_id, scenario)
    if not target.is_dir():
        return False, f"No cached scene for {dataset_id} / {scenario}"
    shutil.rmtree(target)
    parent = target.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
    return True, f"Removed {target}"


# --------------------------------------------------------------------------- assets

# The viewer page is not self-contained: it pulls its theme module, its favicon and the
# ego-vehicle mesh from the server's own asset routes. Mirroring only /viewer/three*
# left those 404ing, and a missing theme module is fatal -- the page dies on "TH is not
# defined" and renders nothing at all. Assets are shared between scenes, so they live
# beside the scenes rather than inside one.
_ASSET_PREFIXES = ("/static/", "/viewer/assets/")
_ASSET_URL_RE = re.compile(
    rb"""["'](/(?:static|viewer/assets)/[A-Za-z0-9._/-]+)(?:\?[^"']*)?["']"""
)
_ASSET_CONTENT_TYPES = {
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".dae": "model/vnd.collada+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".glb": "model/gltf-binary",
    ".wasm": "application/wasm",
}


def assets_root() -> Path:
    return t4_root() / "_assets"


def _asset_file(url_path: str) -> Path | None:
    """Where an asset URL is stored, or None if the path is not one we mirror."""
    if not any(url_path.startswith(prefix) for prefix in _ASSET_PREFIXES):
        return None
    parts = [_safe_name(part) for part in url_path.strip("/").split("/") if part not in ("", ".", "..")]
    return assets_root().joinpath(*parts) if parts else None


def page_asset_urls(page_bytes: bytes) -> list[str]:
    seen: list[str] = []
    for match in _ASSET_URL_RE.finditer(page_bytes):
        url = match.group(1).decode("utf-8")
        if url not in seen:
            seen.append(url)
    return seen


def mirror_page_assets(client: T4Client, page_bytes: bytes, *, force: bool = False) -> dict[str, Any]:
    """Download whatever the page loads from the server's asset routes."""
    stats = {"fetched": 0, "skipped": 0, "errors": []}
    for url in page_asset_urls(page_bytes):
        target = _asset_file(url)
        if target is None:
            continue
        if target.is_file() and not force:
            stats["skipped"] += 1
            continue
        try:
            body, _ = client.get(url)
        except T4Error as exc:
            # An asset the deployment does not actually serve must not fail the scene.
            stats["errors"].append(f"{url}: {exc}")
            continue
        _write_bytes(target, body)
        stats["fetched"] += 1
    return stats


def refresh_scene_shell(dataset_id: str, scenario: str | None = None) -> dict[str, Any]:
    """Re-download a cached scene's page and assets, leaving its frames alone.

    The frames are the expensive part and never change; the page is the viewer's own
    code, which does. Without this a scene cached once would keep an old viewer forever
    -- there is no "download" left to offer for it, since the frames are all present.
    """
    scene = find_scene(dataset_id, scenario)
    if scene is None:
        raise CacheMiss(f"No cached 3D scene for dataset {dataset_id}")
    manifest = {}
    try:
        manifest = json.loads((scene / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    cfg = config.Config.load()
    client = T4Client(cfg.effective_t4_base_url(), cfg, timeout=30.0)
    params = _scene_params(
        manifest.get("dataset_id") or dataset_id,
        manifest.get("scenario") or scene.name,
        manifest.get("version"),
    )
    params["frame_index"] = 0
    body, _ = client.get("/viewer/three", params)
    before = scene.joinpath("page.html").read_bytes() if (scene / "page.html").is_file() else b""
    _write_bytes(scene / "page.html", body)
    return {
        "changed": before != body,
        "bytes": len(body),
        "assets": mirror_page_assets(client, body),
    }


def serve_asset(url_path: str, *, allow_fetch: bool = True) -> tuple[bytes, str] | None:
    """A mirrored asset, fetching it once if it is missing and the server is reachable.

    Self-healing matters here: scenes downloaded before assets were mirrored would
    otherwise stay broken until the user re-fetched hundreds of megabytes of frames.
    """
    target = _asset_file(url_path)
    if target is None:
        return None
    if not target.is_file() and allow_fetch:
        cfg = config.Config.load()
        base = cfg.effective_t4_base_url()
        if base:
            try:
                body, _ = T4Client(base, cfg, timeout=15.0).get(url_path)
                _write_bytes(target, body)
            except T4Error:
                return None
    if not target.is_file():
        return None
    suffix = target.suffix.lower()
    return target.read_bytes(), _ASSET_CONTENT_TYPES.get(suffix, "application/octet-stream")


class CacheMiss(T4Error):
    """The cache has no answer for this request."""


# The mirrored page was rendered by t4-server with its query string already substituted
# into the script, which is why it replays offline with no server. The cost is that it
# then ignores its own URL: the frame to open on, and the external_bbox_* alignment the
# eval overlay needs, are both read from that literal. Re-point it at the query actually
# being served, keeping the scene identity the cache is keyed by.
_PAGE_QUERY_RE = re.compile(rb'new URLSearchParams\("([^"\\]*)"\)')
_PAGE_IDENTITY_KEYS = ("t4dataset_id", "scenario_name", "version")


def _page_with_query(scene: Path, query: dict[str, list[str]]) -> bytes:
    body = scene.joinpath("page.html").read_bytes()
    match = _PAGE_QUERY_RE.search(body)
    if not match:
        return body  # an older or newer page shape: serve it verbatim rather than guess
    baked = urllib.parse.parse_qs(match.group(1).decode("utf-8"))
    merged = {k: list(v) for k, v in query.items()}
    for key in _PAGE_IDENTITY_KEYS:
        if key in baked:
            merged[key] = baked[key]
    pairs = [(k, v) for k, values in merged.items() for v in values]
    replacement = f'new URLSearchParams("{urllib.parse.urlencode(pairs)}")'.encode("utf-8")
    return body[: match.start()] + replacement + body[match.end() :]


def serve_request(path: str, query: dict[str, list[str]]) -> tuple[bytes, str, dict[str, str]]:
    """Answer a mirrored ``/viewer/three*`` request from the cache.

    Returns ``(body, content_type, extra_headers)``. Raises :class:`CacheMiss` when the
    scene or that particular sub-resource was never fetched, so the caller can explain
    what to run rather than showing a broken viewer.
    """
    def one(name: str, default: str | None = None) -> str | None:
        values = query.get(name) or []
        return values[-1] if values else default

    dataset_id = one("t4dataset_id") or ""
    scenario = one("scenario_name")
    if not dataset_id:
        raise CacheMiss("t4dataset_id is required")
    scene = find_scene(dataset_id, scenario)
    if scene is None:
        raise CacheMiss(
            f"No cached 3D scene for dataset {dataset_id}"
            + (f", scenario {scenario}" if scenario else "")
            + ". Fetch it with: evaldash-local t4 fetch "
            + f"{dataset_id}" + (f" --scenario {scenario}" if scenario else "")
        )

    tail = path[len("/viewer/three") :] or "/"

    if tail in ("", "/"):
        return _page_with_query(scene, query), "text/html; charset=utf-8", {}

    if tail == "/meta":
        return _read_or_miss(scene / "meta.json"), "application/json", {}

    if tail == "/frame.bin":
        index = int(float(one("frame_index", "0") or 0))
        blob = _read_or_miss(
            scene / "frames" / f"{index}.bin",
            f"Frame {index} is not cached for this scene.",
        )
        headers = {}
        header_path = scene / "frames" / f"{index}.hdr.json"
        if header_path.is_file():
            try:
                headers = {k: str(v) for k, v in json.loads(header_path.read_text()).items()}
            except Exception:
                headers = {}
        return blob, "application/octet-stream", headers

    if tail == "/frames/window":
        # Purely arithmetic on the frame count, so it can be answered without a fetch.
        try:
            meta = json.loads((scene / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            raise CacheMiss("meta.json is missing from the cached scene")
        total = frame_count_from_meta(meta)
        center = int(float(one("center", "0") or 0))
        radius = int(float(one("radius", "2") or 2))
        low = max(0, center - radius)
        high = min(total - 1, center + radius) if total > 0 else -1
        frames = [{"frame_index": i} for i in range(low, high + 1)]
        body = json.dumps({"total": total, "center": center, "radius": radius, "frames": frames})
        return body.encode("utf-8"), "application/json", {}

    if tail == "/lanelet-lines":
        index = int(float(one("frame_index", "0") or 0))
        return (
            _read_or_miss(scene / "lanelet" / f"{index}.json", f"Lanelet lines for frame {index} are not cached."),
            "application/json",
            {},
        )

    if tail in ("/camera-overlay", "/camera-info"):
        index = int(float(one("frame_index", "0") or 0))
        folder = "overlay" if tail == "/camera-overlay" else "caminfo"
        directory = scene / folder
        matches = sorted(directory.glob(f"{index}__*.json")) if directory.is_dir() else []
        if not matches:
            raise CacheMiss(f"{tail.lstrip('/')} for frame {index} is not cached.")
        # Parameters vary (camera choice, ranges); the cached variant for this frame is
        # the best available answer offline.
        return matches[0].read_bytes(), "application/json", {}

    if tail.startswith("/debug/"):
        # Telemetry pings from the page; nothing to record offline.
        return b"{}", "application/json", {}

    raise CacheMiss(f"{path} is not mirrored offline")


def _read_or_miss(path: Path, message: str | None = None) -> bytes:
    if not path.is_file():
        raise CacheMiss(message or f"{path.name} is not cached")
    return path.read_bytes()
