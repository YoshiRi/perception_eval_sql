"""Small DuckDB-backed HTTP API for the local 3D bbox viewer.

The API is intentionally dependency-light so it can run next to Streamlit in
both local development and the Docker deployment.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import pickle
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import duckdb

try:
    from backend import app_paths, export_api, prebake, workflow_api
except ImportError:  # pragma: no cover - running the module as a bare script.
    import app_paths  # type: ignore[no-redef]
    import export_api  # type: ignore[no-redef]
    import prebake  # type: ignore[no-redef]
    import workflow_api  # type: ignore[no-redef]

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is optional for the bbox API.
    yaml = None


DEFAULT_PORT = int(os.environ.get("LOCAL_BBOX_API_PORT", "8765"))
DEFAULT_TOPICS = (
    "perception.object_recognition.objects",
    "perception.object_recognition.tracking.objects",
)
FILTER_COLUMNS = (
    "suite_name",
    "scenario_name",
    "t4dataset_name",
    "t4dataset_id",
    "topic_name",
    "label",
    "status",
    "source",
    "visibility",
)
CORE_COLUMNS = ("frame_index", "source", "x", "y", "z", "length", "width", "height", "yaw")
OPTIONAL_COLUMNS = (
    "unix_time",
    "frame_id",
    "type",
    "shape_type",
    "vx",
    "vy",
    "label",
    "status",
    "uuid",
    "confidence",
    "pointcloud_num",
    "visibility",
    "pair_uuid",
    "topic_name",
    "suite_name",
    "scenario_name",
    "t4dataset_name",
    "t4dataset_id",
    "x_error",
    "y_error",
    "z_error",
    "yaw_error",
    "vx_error",
    "vy_error",
    "speed_error",
    "center_distance",
    "plane_distance",
    "pair_dt_sec",
    "length_error",
    "width_error",
    "height_error",
    "dx_min",
    "dy_min",
    # §8: object-local polygon footprint vertices (analyzer >=0.2.0); NULL for boxes.
    "footprint",
)
DISTANCE_BINS_SQL = """
    SELECT * FROM (
        VALUES
            (0.0,   10.0,   '[0,10)',     10,  '0-10 m'),
            (10.0,  20.0,   '[10,20)',    20,  '10-20 m'),
            (20.0,  30.0,   '[20,30)',    30,  '20-30 m'),
            (30.0,  40.0,   '[30,40)',    40,  '30-40 m'),
            (40.0,  50.0,   '[40,50)',    50,  '40-50 m'),
            (50.0,  60.0,   '[50,60)',    60,  '50-60 m'),
            (60.0,  70.0,   '[60,70)',    70,  '60-70 m'),
            (70.0,  80.0,   '[70,80)',    80,  '70-80 m'),
            (80.0,  90.0,   '[80,90)',    90,  '80-90 m'),
            (90.0,  100.0,  '[90,100)',   100, '90-100 m'),
            (100.0, 110.0,  '[100,110)',  110, '100-110 m'),
            (110.0, 120.0,  '[110,120)',  120, '110-120 m'),
            (120.0, 130.0,  '[120,130)',  130, '120-130 m'),
            (130.0, 140.0,  '[130,140)',  140, '130-140 m'),
            (140.0, 150.0,  '[140,150)',  150, '140-150 m'),
            (150.0, 1e12,   '[150,inf)',  160, '150+ m')
    ) AS t(bin_start, bin_end, distance_bin, bin_idx, bin_label)
"""
_SERVER_LOCK = threading.Lock()
_SERVER: ThreadingHTTPServer | None = None
_SCENARIO_CONTEXT_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_SUITE_PASS_CACHE: dict[str, dict[str, Any]] = {}
_COLUMNS_CACHE: dict[tuple[str, int, int], list[str]] = {}
_SCENE_RESULT_PICKLE_CACHE: dict[tuple[str, int, int], Any] = {}
_KNOWN_DEVOPS_TARGETS = {
    "animal", "bicycle", "bus", "car", "exhaust_fog", "fallen_object", "ghost", "ground",
    "motorbike", "opened_door", "pedestrian", "rain", "structure", "traffic_cone",
    "truck", "unknown", "vegetation",
}


def _data_root() -> Path:
    return app_paths.data_root()


def _allowed_roots() -> list[Path]:
    roots = [_data_root(), (app_paths.app_root() / "data").resolve(), (Path.cwd() / "data").resolve()]
    extra = os.environ.get("LOCAL_BBOX_ALLOWED_ROOTS", "")
    for chunk in extra.split(os.pathsep):
        text = chunk.strip()
        if text:
            roots.append(Path(text).expanduser().resolve())
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _container_path_candidates(raw: Path) -> list[Path]:
    if not raw.is_absolute():
        return []
    out: list[Path] = []
    mappings = [
        (os.environ.get("LOCAL_EVALUATOR_HOST_DATA_ROOT"), os.environ.get("LOCAL_EVALUATOR_CONTAINER_DATA_ROOT") or str(_data_root())),
    ]
    for host_root_raw, container_root_raw in mappings:
        if not host_root_raw or not container_root_raw:
            continue
        host_root = Path(host_root_raw).expanduser().resolve()
        container_root = Path(container_root_raw).expanduser().resolve()
        try:
            rel = raw.relative_to(host_root)
        except ValueError:
            continue
        out.append((container_root / rel).resolve())
    return out


def _resolve_local_path(value: str | None, *, allow_file: bool = True) -> Path:
    text = str(value or "").strip()
    if not text:
        return _data_root()
    raw = Path(text).expanduser()
    candidates = [raw.resolve(), *_container_path_candidates(raw)] if raw.is_absolute() else [(Path.cwd() / raw).resolve(), (_data_root() / raw).resolve()]
    roots = _allowed_roots()
    for candidate in candidates:
        if allow_file and candidate.is_file() or candidate.is_dir():
            try:
                if any(candidate == root or candidate.is_relative_to(root) for root in roots):
                    return candidate
            except AttributeError:
                if any(str(candidate).startswith(str(root)) for root in roots):
                    return candidate
    raise ValueError(f"Path is outside allowed data roots or does not exist: {text}")


def _short_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_data_root())).replace("\\", "/")
    except ValueError:
        return path.name


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(_json_safe(payload), separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "content-type")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, status: int, html_text: str) -> None:
    body = html_text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(body)


_EXPLORER_ASSET_TYPES = {
    "bbox_theme.js": "text/javascript; charset=utf-8",
    "bbox_explorer.css": "text/css; charset=utf-8",
    "bbox_api.js": "text/javascript; charset=utf-8",
    "bbox_state.js": "text/javascript; charset=utf-8",
    "metrics.js": "text/javascript; charset=utf-8",
    "preview_renderer.js": "text/javascript; charset=utf-8",
    "map_renderer.js": "text/javascript; charset=utf-8",
    "stats_renderer.js": "text/javascript; charset=utf-8",
    "events.js": "text/javascript; charset=utf-8",
    "bbox_viewer.css": "text/css; charset=utf-8",
    "bbox_viewer_api.js": "text/javascript; charset=utf-8",
    "bbox_viewer_state.js": "text/javascript; charset=utf-8",
    "bbox_viewer_filters.js": "text/javascript; charset=utf-8",
    "bbox_viewer_analysis.js": "text/javascript; charset=utf-8",
    "bbox_viewer_geometry.js": "text/javascript; charset=utf-8",
    "bbox_viewer_renderer.js": "text/javascript; charset=utf-8",
    "bbox_viewer_timeline.js": "text/javascript; charset=utf-8",
    "bbox_viewer_events.js": "text/javascript; charset=utf-8",
}


_ASSET_SUFFIX_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}
# The explorer/viewer HTML is re-read from disk per request, but this list lives in
# the running process. Adding a new asset therefore used to 404 until the server was
# restarted, which breaks the page rather than just its styling, so any plain .js/.css
# basename under static/ is served even when it predates the process.
_SAFE_ASSET_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _asset_content_type(asset_name: str) -> str:
    explicit = _EXPLORER_ASSET_TYPES.get(asset_name)
    if explicit:
        return explicit
    if ".." in asset_name or not _SAFE_ASSET_NAME.match(asset_name):
        return ""
    return _ASSET_SUFFIX_TYPES.get(Path(asset_name).suffix, "")


def _static_asset_response(handler: BaseHTTPRequestHandler, asset_name: str, *, head_only: bool = False) -> bool:
    content_type = _asset_content_type(asset_name)
    if not content_type:
        return False
    for path in (directory / asset_name for directory in app_paths.static_dirs()):
        if not path.exists():
            continue
        body = b"" if head_only else path.read_bytes()
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(path.stat().st_size))
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        if not head_only:
            handler.wfile.write(body)
        return True
    return False


def _static_file_text(name: str) -> str:
    path = app_paths.find_static_file(name)
    return path.read_text(encoding="utf-8") if path else ""


def _render_page_html(name: str, api_base: str) -> str:
    """Read a page from static/ and fill in its server-side placeholders.

    The theme module is inlined rather than linked: it defines the palette every
    renderer needs, so a page that loads without it throws on first paint. Inlining
    ties it to the same disk read as the HTML, instead of a separate asset request
    that an older server process may not know how to serve.
    """
    path = app_paths.find_static_file(name)
    if path is None:
        raise FileNotFoundError(f"static/{name} not found")
    source = path.read_text(encoding="utf-8")
    source = source.replace("__API_BASE__", api_base.rstrip("/"))
    source = source.replace("/*__BBOX_THEME_JS__*/", _static_file_text("bbox_theme.js"))
    if "/*__PIXEL_OFFICE_JS__*/" in source:
        source = source.replace("/*__PIXEL_OFFICE_JS__*/", _static_file_text("pixel_office.js"))
    return source


def _viewer_html(api_base: str = "") -> str:
    return _render_page_html("local_bbox_viewer.html", api_base)


def _explorer_html(api_base: str = "") -> str:
    return _render_page_html("local_bbox_explorer.html", api_base)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("content-length") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _columns(parquet_path: Path) -> list[str]:
    stat = parquet_path.stat()
    key = (str(parquet_path), stat.st_size, stat.st_mtime_ns)
    if key in _COLUMNS_CACHE:
        return _COLUMNS_CACHE[key]
    con = duckdb.connect()
    try:
        cols = con.execute("DESCRIBE SELECT * FROM parquet_scan(?)", [str(parquet_path)]).df()["column_name"].tolist()
    finally:
        con.close()
    _COLUMNS_CACHE[key] = cols
    return cols


def _require_columns(cols: list[str], required: tuple[str, ...]) -> None:
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing required bbox columns: {', '.join(missing)}")


def _where_from_filters(cols: list[str], filters: dict[str, Any]) -> tuple[list[str], list[Any]]:
    where: list[str] = ["1=1"]
    params: list[Any] = []
    for column in FILTER_COLUMNS:
        if column not in cols or column not in filters:
            continue
        value = filters.get(column)
        if value is None or value == "" or value == "Any":
            continue
        if isinstance(value, list):
            clean = [v for v in value if str(v).strip()]
            if not clean:
                continue
            where.append(f"{column} IN ({','.join(['?'] * len(clean))})")
            params.extend(clean)
        else:
            where.append(f"{column} = ?")
            params.append(value)
    frame_min = filters.get("frame_min")
    frame_max = filters.get("frame_max")
    if frame_min not in (None, ""):
        where.append("TRY_CAST(frame_index AS INTEGER) >= ?")
        params.append(int(frame_min))
    if frame_max not in (None, ""):
        where.append("TRY_CAST(frame_index AS INTEGER) <= ?")
        params.append(int(frame_max))
    confidence_min = filters.get("confidence_min")
    if confidence_min not in (None, "") and "confidence" in cols:
        where.append("(confidence IS NULL OR TRY_CAST(confidence AS DOUBLE) >= ?)")
        params.append(float(confidence_min))
    distance_min = filters.get("distance_min")
    distance_max = filters.get("distance_max")
    if distance_min not in (None, ""):
        where.append("SQRT(POWER(TRY_CAST(x AS DOUBLE), 2) + POWER(TRY_CAST(y AS DOUBLE), 2)) >= ?")
        params.append(float(distance_min))
    if distance_max not in (None, ""):
        where.append("SQRT(POWER(TRY_CAST(x AS DOUBLE), 2) + POWER(TRY_CAST(y AS DOUBLE), 2)) <= ?")
        params.append(float(distance_max))
    return where, params


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "<na>"} else text


def _criteria_level_value(value: Any) -> float:
    text = _as_text(value).lower()
    named = {
        "perfect": 100.0,
        "hard": 75.0,
        "normal": 50.0,
        "easy": 25.0,
    }
    if text in named:
        return named[text]
    return _as_float(value, default=float("nan"))


def _normalize_devops_label(value: str) -> str:
    raw = re.sub(r"(?<!^)(?=[A-Z])", "_", str(value or "")).replace("-", "_").lower()
    aliases = {
        "pedestrian": "pedestrian",
        "pedestrians": "pedestrian",
        "pedestrian_child": "pedestrian",
        "pedestrians_children": "pedestrian",
        "child": "pedestrian",
        "children": "pedestrian",
        "adult": "pedestrian",
        "adults": "pedestrian",
        "pedestrian_group": "pedestrian",
        "bicycle_pedestrians": "pedestrian",
        "crouching_pedestrian": "pedestrian",
        "dog": "animal",
        "animal": "animal",
        "card_board": "fallen_object",
        "cardboard": "fallen_object",
        "fallen_object": "fallen_object",
        "fallen_sign": "fallen_object",
        "road_debris": "fallen_object",
        "sandbag": "fallen_object",
        "sunshade": "fallen_object",
        "fallen_cone": "traffic_cone",
        "plastic_bag": "fallen_object",
        "umbrella": "fallen_object",
        "others": "unknown",
        "other": "unknown",
        "unknown": "unknown",
        "cone": "traffic_cone",
        "cones": "traffic_cone",
        "traffic_cone": "traffic_cone",
        "opened_door": "opened_door",
        "door": "opened_door",
        "truck": "truck",
        "trailer": "truck",
        "track": "truck",
        "bus": "bus",
        "car": "car",
        "bicycle": "bicycle",
        "bicycles": "bicycle",
        "motorbike": "motorbike",
        "motorcycle": "motorbike",
        "motorcycles": "motorbike",
        "surface_cluster": "ground",
        "ground": "ground",
        "shrub": "vegetation",
        "tree": "vegetation",
        "vegetation": "vegetation",
        "plant": "vegetation",
        "rain": "rain",
        "watervapor": "exhaust_fog",
        "water_vapor": "exhaust_fog",
        "exhaust": "exhaust_fog",
        "fog": "exhaust_fog",
        "ghost": "ghost",
        "ghost_from_fence": "ghost",
        "ghost_from_guardrail": "ghost",
        "ghost_or_side_mirror": "ghost",
        "rocket": "ghost",
        "streetlight": "structure",
        "signboard": "structure",
        "pole": "structure",
        "rubberpole": "structure",
        "utility_pole_or_banner": "structure",
        "watersupply": "structure",
        "board": "structure",
        "bird": "animal",
        "dragonfly": "animal",
    }
    return aliases.get(raw, raw)


def _suite_base_name(suite_name: str) -> str:
    return re.sub(r"_[0-9a-f]{8}-[0-9a-f-]{27,}$", "", _as_text(suite_name))


def _suite_pass_summary(parquet_path: Path, suite_name: str) -> dict[str, Any]:
    run_dir = parquet_path.parent
    cache_key = str(run_dir)
    if cache_key not in _SUITE_PASS_CACHE:
        data: dict[str, Any] = {}
        for rel in ("resources/summary.json", "summary.json"):
            path = run_dir / rel
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            devops = raw.get("DevOps") if isinstance(raw, dict) else None
            pass_rate = devops.get("Suite pass rate") if isinstance(devops, dict) else None
            if isinstance(pass_rate, dict):
                data = pass_rate
                break
        _SUITE_PASS_CACHE[cache_key] = data
    hit = _SUITE_PASS_CACHE[cache_key].get(_suite_base_name(suite_name)) or {}
    passed = hit.get("passed")
    total = hit.get("total")
    if passed is None or total in (None, 0):
        return {}
    return {"passed": int(passed), "total": int(total), "pass_rate": float(passed) / float(total)}


def _devops_context_from_name(suite_name: str, scenario_name: str) -> dict[str, Any]:
    suite_text = _as_text(suite_name)
    scenario_text = _as_text(scenario_name)
    blob = f"{suite_text} {scenario_text}"
    tokens = [t for t in re.split(r"[_\s]+", scenario_text) if t]
    lower_tokens = [t.lower() for t in tokens]
    suite_tokens = [t for t in re.split(r"[_\s]+", suite_text) if t]
    issue = ""
    for token in [t.lower() for t in suite_tokens] + lower_tokens:
        if token in {"fn", "fp", "tp"}:
            issue = token.upper()
            break
    city = ""
    for idx, token in enumerate(tokens):
        if token.lower().startswith(("j6gen", "j6", "x2")) and idx + 1 < len(tokens):
            city = tokens[idx + 1]
            break
    behavior = ""
    for token in tokens:
        if token in {
            "ObstacleStop", "RoadUserStop", "RunOut", "Crosswalk", "Intersection",
            "IntersectionLeft", "IntersectionRight", "IntersectionStraight", "Normal",
            "LaneChange", "Distant", "NA",
        }:
            behavior = token
            break
    if behavior == "NA":
        behavior = ""
    pc_mode = ""
    for token in tokens:
        if token.lower() in {"pcon", "pcoff"}:
            pc_mode = "PC on" if token.lower() == "pcon" else "PC off"
            break
    target = ""
    for token in tokens + re.split(r"[_\s]+", suite_text):
        label = _normalize_devops_label(token)
        raw_label = re.sub(r"(?<!^)(?=[A-Z])", "_", str(token or "")).replace("-", "_").lower()
        if label in _KNOWN_DEVOPS_TARGETS and (label != "unknown" or raw_label == "unknown"):
            target = label
            break
    family = suite_text.replace("DevOps_V1_", "")
    intent_type = "investigate"
    focus_metric = "fp"
    family_l = family.lower()
    if "inaccurate_yaw" in family_l or "xy_position_jitter" in family_l or "yaw" in blob.lower():
        intent_type = "localization/yaw accuracy"
        focus_metric = "error"
    elif "mislabeled" in family_l:
        intent_type = "label confusion"
        focus_metric = "fn" if issue == "FN" else "fp"
    elif "misclassified" in family_l:
        intent_type = "structure misclassification"
        focus_metric = "fp"
    elif "large_object" in family_l:
        intent_type = "large-object stability"
        focus_metric = "fn"
    elif issue == "FN":
        intent_type = "target detection"
        focus_metric = "fn"
    elif issue == "FP":
        intent_type = "false detection / false stop"
        focus_metric = "fp"
    purpose = " ".join(x for x in [issue, behavior, target, city] if x).strip()
    return {
        "is_devops": "DevOps" in blob,
        "issue_type": issue,
        "intent_type": intent_type,
        "focus_metric": focus_metric,
        "target_label": target,
        "behavior": behavior,
        "pc_mode": pc_mode,
        "city": city,
        "family": family,
        "purpose": purpose,
        "description": "",
        "criteria": [],
        "target_labels": [],
        "matching_thresholds": [],
        "merge_similar_labels": False,
        "matching_label_policy": "",
        "yaml_path": "",
    }


def _scenario_yaml_path(parquet_path: Path, suite_name: str, scenario_name: str) -> Path | None:
    if not suite_name or not scenario_name:
        return None
    direct = parquet_path.parent / suite_name / scenario_name / "scenario.yaml"
    if direct.exists():
        return direct
    # Some source dumps differ only by case in the scenario directory name.
    suite_dir = parquet_path.parent / suite_name
    if suite_dir.is_dir():
        expected = scenario_name.lower()
        for child in suite_dir.iterdir():
            if child.is_dir() and child.name.lower() == expected and (child / "scenario.yaml").exists():
                return child / "scenario.yaml"
    return None


def _suite_scenario_inventory(parquet_path: Path, suite_name: str) -> list[str]:
    suite_text = _as_text(suite_name)
    if not suite_text:
        return []
    run_dir = parquet_path.parent
    candidates = [run_dir / suite_text]
    base = _suite_base_name(suite_text).lower()
    if not candidates[0].exists():
        try:
            candidates.extend(
                child for child in run_dir.iterdir()
                if child.is_dir() and _suite_base_name(child.name).lower() == base
            )
        except OSError:
            pass
    for suite_dir in candidates:
        if not suite_dir.exists() or not suite_dir.is_dir():
            continue
        try:
            names = sorted(
                child.name for child in suite_dir.iterdir()
                if child.is_dir() and (child / "scenario.yaml").exists()
            )
        except OSError:
            continue
        if names:
            return names
    return []


def _criterion_summary(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        filter_obj = item.get("Filter") if isinstance(item.get("Filter"), dict) else {}
        out.append(
            {
                "method": _as_text(item.get("CriteriaMethod")),
                "level": _as_text(item.get("CriteriaLevel")),
                "pass_rate": item.get("PassRate"),
                "filter": {str(k): _as_text(v) for k, v in filter_obj.items()},
            }
        )
    return out


def _planning_factor_summary(scenario_dir: Path) -> dict[str, Any]:
    path = scenario_dir / "planning_factor.jsonl"
    if not path.exists():
        return {}
    out: dict[str, Any] = {"path": _short_path(path), "passed_frames": 0, "failed_frames": 0, "nodata_frames": 0, "conditions": []}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return out
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if idx == 0 and isinstance(obj.get("Condition"), list):
            for cond in obj.get("Condition", [])[:4]:
                if not isinstance(cond, dict):
                    continue
                dist = cond.get("distance") if isinstance(cond.get("distance"), dict) else {}
                out["conditions"].append(
                    {
                        "topic": _as_text(cond.get("topic")),
                        "behavior": [str(x) for x in cond.get("behavior", [])] if isinstance(cond.get("behavior"), list) else [],
                        "judgement": _as_text(cond.get("judgement")),
                        "distance": {
                            "min": dist.get("min"),
                            "max": dist.get("max"),
                        },
                    }
                )
            continue
        result = obj.get("Result") if isinstance(obj.get("Result"), dict) else {}
        summary = _as_text(result.get("Summary"))
        if "Passed:" in summary:
            out["passed_frames"] += 1
        elif "Failed:" in summary:
            out["failed_frames"] += 1
        elif "NoData" in summary:
            out["nodata_frames"] += 1
    return out


def _parse_distance_filter(value: Any) -> tuple[float | None, float | None, str]:
    text = _as_text(value)
    if not text:
        return None, None, "all distances"
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    low = float(nums[0]) if nums else None
    high = float(nums[1]) if len(nums) > 1 else None
    if "-" in text and text.strip().endswith("-"):
        high = None
    if low is None and high is None:
        return None, None, text
    if high is None:
        return low, None, f">= {low:g} m"
    if low is None:
        return None, high, f"< {high:g} m"
    return low, high, f"{low:g} - {high:g} m"


def _parse_region_axis(value: Any) -> tuple[float, float] | None:
    text = _as_text(value)
    if not text:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) < 2:
        return None
    low = float(nums[0])
    high = float(nums[1])
    return (low, high) if low < high else None


def _parse_region_filter(value: Any) -> tuple[Any | None, str]:
    if not isinstance(value, dict):
        return None, ""
    x_range = _parse_region_axis(value.get("x_position"))
    y_range = _parse_region_axis(value.get("y_position"))
    if x_range is None and y_range is None:
        return None, ""
    label_parts = []
    if x_range is not None:
        label_parts.append(f"x {x_range[0]:g} - {x_range[1]:g} m")
    if y_range is not None:
        label_parts.append(f"y {y_range[0]:g} - {y_range[1]:g} m")
    return SimpleNamespace(x_position=x_range, y_position=y_range), ", ".join(label_parts)


def _criteria_filter_label(filter_obj: dict[str, Any]) -> str:
    _, _, distance_label = _parse_distance_filter(filter_obj.get("Distance"))
    if distance_label != "all distances":
        return distance_label
    _, region_label = _parse_region_filter(filter_obj.get("Region"))
    return region_label or distance_label


def _scenario_pickle_path(parquet_path: Path, suite_name: str, scenario_name: str) -> Path | None:
    yaml_path = _scenario_yaml_path(parquet_path, suite_name, scenario_name)
    if not yaml_path:
        return None
    path = yaml_path.parent / "scene_result.pkl"
    return path if path.exists() else None


def _ensure_eval_lib_paths() -> None:
    for lib_path in app_paths.eval_lib_paths():
        text = str(lib_path)
        if lib_path.exists() and text not in sys.path:
            sys.path.append(text)


def _load_scene_result_pickle(pickle_path: Path) -> Any:
    _ensure_eval_lib_paths()
    stat = pickle_path.stat()
    key = (str(pickle_path), int(stat.st_mtime_ns), int(stat.st_size))
    if key in _SCENE_RESULT_PICKLE_CACHE:
        return _SCENE_RESULT_PICKLE_CACHE[key]
    real_makedirs = os.makedirs

    def quiet_makedirs(name: Any, mode: int = 0o777, exist_ok: bool = False) -> None:
        try:
            real_makedirs(name, mode=mode, exist_ok=exist_ok)
        except OSError:
            return None

    old_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        os.makedirs = quiet_makedirs  # type: ignore[assignment]
        with pickle_path.open("rb") as f:
            frames = pickle.load(f)
    finally:
        os.makedirs = real_makedirs  # type: ignore[assignment]
        logging.disable(old_disable)
    _SCENE_RESULT_PICKLE_CACHE.clear()
    _SCENE_RESULT_PICKLE_CACHE[key] = frames
    return frames


def _criteria_filter_namespace(filter_obj: dict[str, Any]) -> Any:
    low, high, _ = _parse_distance_filter(filter_obj.get("Distance") if isinstance(filter_obj, dict) else None)
    distance = None
    if low is not None or high is not None:
        distance = (0.0 if low is None else low, sys.float_info.max if high is None else high)
    region, _ = _parse_region_filter(filter_obj.get("Region") if isinstance(filter_obj, dict) else None)
    if distance is not None:
        region = None
    return SimpleNamespace(Distance=distance, Region=region)


def _gate_label_for_method(method: str, evaluation_task: str, level: float) -> tuple[str, str]:
    method_l = _as_text(method).lower()
    if method_l == "num_gt_tp":
        if evaluation_task == "fp_validation":
            return (
                "frame FP-validation pass rate",
                "Each FP-validation frame must satisfy evaluator TN/FP pass-fail at CriteriaLevel, then enough frames must satisfy PassRate.",
            )
        return (
            "frame GT recall pass rate",
            "Each non-empty frame must reach the CriteriaLevel GT recall, then enough frames must satisfy PassRate.",
        )
    if method_l == "num_tp":
        return (
            "frame object pass rate",
            "Each non-empty frame must reach the CriteriaLevel object success rate, then enough frames must satisfy PassRate.",
        )
    if method_l == "yaw_error":
        return (
            "frame yaw-error pass rate",
            f"Each frame's average TP yaw error must be <= {level:.3f} rad, then enough frames must satisfy PassRate.",
        )
    return (method_l or "criterion", "Evaluator criterion method.")


def _perception_criteria_for(criterion: dict[str, Any]) -> Any:
    from driving_log_replayer_v2.criteria.perception import PerceptionCriteria  # type: ignore

    filter_obj = criterion.get("filter") if isinstance(criterion.get("filter"), dict) else {}
    return PerceptionCriteria(
        methods=criterion.get("method") or None,
        levels=criterion.get("level") if criterion.get("level") not in ("", None) else None,
        filters=_criteria_filter_namespace(filter_obj),
    )


def _pass_fail_stats(frame: Any, method: str = "") -> dict[str, int]:
    pf = getattr(frame, "pass_fail_result", None)
    if pf is None:
        return {
            "gt_tp": 0,
            "gt_tn": 0,
            "gt_fn": 0,
            "est_fp": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "success": 0,
            "fail": 0,
            "gt": 0,
            "total": 0,
        }
    tp_results = getattr(pf, "tp_object_results", None) or []
    fp_results = getattr(pf, "fp_object_results", None) or []
    fn_objects = getattr(pf, "fn_objects", None) or []
    tn_objects = getattr(pf, "tn_objects", None) or []
    success = int(pf.get_num_success()) if hasattr(pf, "get_num_success") else len(tp_results) + len(tn_objects)
    fail = int(pf.get_num_fail()) if hasattr(pf, "get_num_fail") else len(fp_results) + len(fn_objects)
    gt = int(pf.get_num_gt()) if hasattr(pf, "get_num_gt") else success + fail
    return {
        "gt_tp": len(tp_results),
        "gt_tn": len(tn_objects),
        "gt_fn": len(fn_objects),
        "est_fp": len(fp_results),
        "tp": len(tp_results),
        "fp": len(fp_results),
        "fn": len(fn_objects),
        "tn": len(tn_objects),
        "success": success,
        "fail": fail,
        "gt": gt,
        "total": gt if method == "num_gt_tp" else success + fail,
    }


def _frame_gate_detail(
    frame: Any,
    criterion: dict[str, Any],
    idx: int,
    evaluation_task: str,
    PerceptionCriteria: Any,
) -> dict[str, Any]:
    method = _as_text(criterion.get("method")).lower()
    level_raw = criterion.get("level")
    level = _criteria_level_value(level_raw)
    if not math.isfinite(level):
        level = 0.0 if method in {"yaw_error", "velocity_x_error", "velocity_y_error", "speed_error"} else 100.0
    filter_obj = criterion.get("filter") if isinstance(criterion.get("filter"), dict) else {}
    distance_label = _criteria_filter_label(filter_obj)
    metric_label, meaning = _gate_label_for_method(method, evaluation_task, level)
    base = {
        "index": idx,
        "method": method,
        "metric_label": metric_label,
        "meaning": meaning,
        "distance_label": distance_label,
        "filter": filter_obj,
        "criteria_level": _as_text(level_raw),
        "criteria_level_value": None if not math.isfinite(level) else level,
        "evaluation_task": evaluation_task,
        "level": None if not math.isfinite(level) else level,
        "source": "scene_result.pkl",
    }
    try:
        evaluator_criteria = PerceptionCriteria(
            methods=criterion.get("method") or None,
            levels=criterion.get("level") if criterion.get("level") not in ("", None) else None,
            filters=_criteria_filter_namespace(filter_obj),
        )
        result, ret_frame = evaluator_criteria.get_result(frame)
    except Exception as exc:
        return {
            **base,
            "judged": False,
            "passed": None,
            "score": None,
            "score_unit": "rad" if method.endswith("_error") or method == "yaw_error" else "%",
            "reason": f"Evaluator could not compute this frame gate: {exc}",
            "counts": _pass_fail_stats(frame, method),
        }
    stats = _pass_fail_stats(ret_frame, method)
    if result is None:
        return {
            **base,
            "judged": False,
            "passed": None,
            "score": None,
            "score_unit": "rad" if method.endswith("_error") or method == "yaw_error" else "%",
            "reason": f"Evaluator skipped this frame: no success/fail objects after {distance_label}.",
            "counts": stats,
        }
    score = None
    score_unit = "rad" if method in {"yaw_error", "velocity_x_error", "velocity_y_error", "speed_error"} else "%"
    try:
        score = float(evaluator_criteria.methods[0].calculate_score(ret_frame))
    except Exception:
        score = None
    passed = bool(result.is_success()) if hasattr(result, "is_success") else str(result).lower() == "success"
    score_text = "-" if score is None else (f"{score:.3f} rad" if score_unit == "rad" else f"{score:.1f}%")
    level_text = "-" if not math.isfinite(level) else (f"{level:.3f} rad" if score_unit == "rad" else f"{level:.0f}%")
    return {
        **base,
        "judged": True,
        "passed": passed,
        "score": score,
        "score_unit": score_unit,
        "reason": f"{score_text} / {level_text}; evaluator counts success {stats['success']}, fail {stats['fail']}, gt {stats['gt']}.",
        "counts": stats,
    }


def _pickle_devops_gates(
    parquet_path: Path,
    suite_name: str,
    scenario_name: str,
    criteria: list[dict[str, Any]],
    evaluation_task: str,
) -> list[dict[str, Any]] | None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    pickle_path = _scenario_pickle_path(parquet_path, suite_name, scenario_name)
    if not pickle_path or not criteria:
        return None
    try:
        from driving_log_replayer_v2.criteria.perception import PerceptionCriteria  # type: ignore
    except Exception:
        _ensure_eval_lib_paths()
        try:
            from driving_log_replayer_v2.criteria.perception import PerceptionCriteria  # type: ignore
        except Exception:
            return None
    try:
        frames = _load_scene_result_pickle(pickle_path)
    except Exception:
        return None
    gates: list[dict[str, Any]] = []
    for idx, criterion in enumerate(criteria):
        method = _as_text(criterion.get("method")).lower()
        pass_rate = _as_float(criterion.get("pass_rate"), default=0.0)
        level_raw = criterion.get("level")
        level = _criteria_level_value(level_raw)
        if not math.isfinite(level):
            level = 0.0 if method in {"yaw_error", "velocity_x_error", "velocity_y_error", "speed_error"} else 100.0
        filter_obj = criterion.get("filter") if isinstance(criterion.get("filter"), dict) else {}
        distance_label = _criteria_filter_label(filter_obj)
        try:
            evaluator_criteria = PerceptionCriteria(
                methods=criterion.get("method") or None,
                levels=criterion.get("level") if criterion.get("level") not in ("", None) else None,
                filters=_criteria_filter_namespace(filter_obj),
            )
        except Exception:
            metric_label, _ = _gate_label_for_method(method, evaluation_task, level)
            gates.append(
                {
                    "index": idx,
                    "method": method,
                    "metric_label": metric_label,
                    "meaning": "Unsupported criterion method or level in this explorer.",
                    "distance_label": distance_label,
                    "filter": filter_obj,
                    "required_rate": pass_rate / 100.0,
                    "actual_rate": None,
                    "passed": None,
                    "passed_count": 0,
                    "fail_count": 0,
                    "total_count": 0,
                    "object_success_count": 0,
                    "object_fail_count": 0,
                    "object_total_count": 0,
                    "criteria_level": _as_text(level_raw),
                    "criteria_level_value": None if not math.isfinite(level) else level,
                    "evaluation_task": evaluation_task,
                    "level": None if not math.isfinite(level) else level,
                    "source": "scene_result.pkl",
                }
            )
            continue
        passed_count = 0
        total_count = 0
        object_success_count = 0
        object_fail_count = 0
        object_total_count = 0
        for frame in frames:
            try:
                result, ret_frame = evaluator_criteria.get_result(frame)
            except Exception:
                continue
            if result is None:
                continue
            total_count += 1
            if result.is_success():
                passed_count += 1
            pf = ret_frame.pass_fail_result
            success = int(pf.get_num_success())
            fail = int(pf.get_num_fail())
            object_success_count += success
            object_fail_count += fail
            object_total_count += int(pf.get_num_gt()) if method == "num_gt_tp" else success + fail
        fail_count = max(0, total_count - passed_count)
        actual_rate = passed_count / total_count if total_count else None
        passed = (actual_rate * 100 >= pass_rate) if actual_rate is not None else None
        metric_label, meaning = _gate_label_for_method(method, evaluation_task, level)
        gates.append(
            {
                "index": idx,
                "method": method,
                "metric_label": metric_label,
                "meaning": meaning,
                "distance_label": distance_label,
                "filter": filter_obj,
                "required_rate": pass_rate / 100.0,
                "actual_rate": actual_rate,
                "passed": passed,
                "passed_count": passed_count,
                "fail_count": fail_count,
                "total_count": total_count,
                "object_success_count": object_success_count,
                "object_fail_count": object_fail_count,
                "object_total_count": object_total_count,
                "criteria_level": _as_text(level_raw),
                "criteria_level_value": None if not math.isfinite(level) else level,
                "evaluation_task": evaluation_task,
                "level": None if not math.isfinite(level) else level,
                "source": "scene_result.pkl",
            }
        )
    return gates


def _pickle_fp_validation_gates_fast(
    parquet_path: Path,
    suite_name: str,
    scenario_name: str,
    criteria: list[dict[str, Any]],
    evaluation_task: str,
) -> list[dict[str, Any]] | None:
    if evaluation_task != "fp_validation" or not criteria:
        return None
    supported_methods = {"num_gt_tp", "num_tp"}
    for criterion in criteria:
        method = _as_text(criterion.get("method")).lower()
        filter_obj = criterion.get("filter") if isinstance(criterion.get("filter"), dict) else {}
        low, high, _ = _parse_distance_filter(filter_obj.get("Distance"))
        region, _ = _parse_region_filter(filter_obj.get("Region"))
        if method not in supported_methods or low is not None or high is not None or region is not None:
            return None
    pickle_path = _scenario_pickle_path(parquet_path, suite_name, scenario_name)
    if not pickle_path:
        return None
    try:
        frames = _load_scene_result_pickle(pickle_path)
    except Exception:
        return None
    gates: list[dict[str, Any]] = []
    for idx, criterion in enumerate(criteria):
        method = _as_text(criterion.get("method")).lower()
        pass_rate = _as_float(criterion.get("pass_rate"), default=0.0)
        level_raw = criterion.get("level")
        level = _criteria_level_value(level_raw)
        if not math.isfinite(level):
            level = 100.0
        filter_obj = criterion.get("filter") if isinstance(criterion.get("filter"), dict) else {}
        passed_count = 0
        total_count = 0
        object_success_count = 0
        object_fail_count = 0
        object_total_count = 0
        for frame in frames or []:
            pf = getattr(frame, "pass_fail_result", None)
            if pf is None:
                continue
            success = int(pf.get_num_success()) if hasattr(pf, "get_num_success") else 0
            fail = int(pf.get_num_fail()) if hasattr(pf, "get_num_fail") else 0
            if success + fail <= 0:
                continue
            gt = int(pf.get_num_gt()) if hasattr(pf, "get_num_gt") else success + fail
            if method == "num_gt_tp":
                denom = gt
            else:
                denom = success + fail
            if denom <= 0:
                continue
            frame_score = 100.0 * success / denom
            total_count += 1
            if frame_score >= level:
                passed_count += 1
            object_success_count += success
            object_fail_count += fail
            object_total_count += denom
        fail_count = max(0, total_count - passed_count)
        actual_rate = passed_count / total_count if total_count else None
        passed = (actual_rate * 100 >= pass_rate) if actual_rate is not None else None
        metric_label, meaning = _gate_label_for_method(method, evaluation_task, level)
        gates.append(
            {
                "index": idx,
                "method": method,
                "metric_label": metric_label,
                "meaning": meaning,
                "distance_label": _criteria_filter_label(filter_obj),
                "filter": filter_obj,
                "required_rate": pass_rate / 100.0,
                "actual_rate": actual_rate,
                "passed": passed,
                "passed_count": passed_count,
                "fail_count": fail_count,
                "total_count": total_count,
                "object_success_count": object_success_count,
                "object_fail_count": object_fail_count,
                "object_total_count": object_total_count,
                "criteria_level": _as_text(level_raw),
                "criteria_level_value": level,
                "evaluation_task": evaluation_task,
                "level": level,
                "source": "scene_result.pkl_fast",
            }
        )
    return gates


def _load_scenario_context(parquet_path: Path, suite_name: str, scenario_name: str) -> dict[str, Any]:
    key = (str(parquet_path.parent), suite_name, scenario_name)
    if key in _SCENARIO_CONTEXT_CACHE:
        return _SCENARIO_CONTEXT_CACHE[key]
    ctx = _devops_context_from_name(suite_name, scenario_name)
    suite_pass = _suite_pass_summary(parquet_path, suite_name)
    if suite_pass:
        ctx["suite_pass"] = suite_pass
    yaml_path = _scenario_yaml_path(parquet_path, suite_name, scenario_name)
    if yaml_path and yaml is not None:
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        if isinstance(data, dict):
            eval_cfg = data.get("Evaluation") if isinstance(data.get("Evaluation"), dict) else {}
            cond = eval_cfg.get("Conditions") if isinstance(eval_cfg.get("Conditions"), dict) else {}
            pf_cfg = eval_cfg.get("PerceptionPassFailConfig") if isinstance(eval_cfg.get("PerceptionPassFailConfig"), dict) else {}
            pe_cfg = eval_cfg.get("PerceptionEvaluationConfig") if isinstance(eval_cfg.get("PerceptionEvaluationConfig"), dict) else {}
            eval_dict = pe_cfg.get("evaluation_config_dict") if isinstance(pe_cfg.get("evaluation_config_dict"), dict) else {}
            target_labels = pf_cfg.get("target_labels") or eval_dict.get("target_labels") or []
            thresholds = pf_cfg.get("matching_threshold_list") or []
            evaluation_task = _as_text(eval_dict.get("evaluation_task"))
            merge_similar_labels = bool(eval_dict.get("merge_similar_labels"))
            matching_label_policy = _as_text(eval_dict.get("matching_label_policy")).lower()
            description = _as_text(data.get("ScenarioDescription"))
            ctx.update(
                {
                    "description": description,
                    "criteria": _criterion_summary(cond.get("Criterion")),
                    "target_labels": [str(x) for x in target_labels if _as_text(x)],
                    "matching_thresholds": thresholds if isinstance(thresholds, list) else [],
                    "evaluation_task": evaluation_task,
                    "merge_similar_labels": merge_similar_labels,
                    "matching_label_policy": matching_label_policy,
                    "yaml_path": _short_path(yaml_path),
                    "planning_factor": _planning_factor_summary(yaml_path.parent),
                }
            )
            if not ctx.get("target_label"):
                for label in ctx["target_labels"]:
                    if label not in {"unknown"}:
                        ctx["target_label"] = _normalize_devops_label(label)
                        break
    _SCENARIO_CONTEXT_CACHE[key] = ctx
    return ctx


def _load_scenario_context_light(parquet_path: Path, suite_name: str, scenario_name: str) -> dict[str, Any]:
    ctx = _devops_context_from_name(suite_name, scenario_name)
    suite_pass = _suite_pass_summary(parquet_path, suite_name)
    if suite_pass:
        ctx["suite_pass"] = suite_pass
    return ctx


def _parquet_list_cache_dir() -> Path:
    return app_paths.cache_root() / "local_bbox_api" / "parquets"


def _directory_signature(root: Path) -> dict[str, Any]:
    stat = root.stat()
    return {"path": str(root), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def _parquet_tree_signature(root: Path, limit: int) -> dict[str, Any]:
    root_stat = root.stat()
    count = 0
    total_size = 0
    max_mtime_ns = root_stat.st_mtime_ns
    latest_paths: list[str] = []
    if root.is_file():
        if root.suffix == ".parquet":
            stat = root.stat()
            count = 1
            total_size = stat.st_size
            max_mtime_ns = max(max_mtime_ns, stat.st_mtime_ns)
            latest_paths = [str(root)]
    else:
        latest: list[tuple[int, str]] = []
        for path in root.rglob("*.parquet"):
            if ".dashboard_cache" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            count += 1
            total_size += stat.st_size
            max_mtime_ns = max(max_mtime_ns, stat.st_mtime_ns)
            latest.append((stat.st_mtime_ns, str(path.relative_to(root))))
            if count >= limit:
                break
        latest_paths = [p for _, p in sorted(latest, reverse=True)[:12]]
    return {
        "path": str(root),
        "root_mtime_ns": root_stat.st_mtime_ns,
        "count": count,
        "total_size": total_size,
        "max_mtime_ns": max_mtime_ns,
        "latest_paths": latest_paths,
    }


def _parquet_list_cache_path(root: Path, payload: dict[str, Any]) -> Path:
    limit = int(payload.get("limit") or 2000)
    key_payload = {
        "version": 3,
        "root": _path_signature(root) if root.is_file() else _parquet_tree_signature(root, limit),
        "bbox_only": payload.get("bbox_only", False) is True,
        "limit": limit,
    }
    digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    return _parquet_list_cache_dir() / f"{digest}.json"


def list_parquets(payload: dict[str, Any]) -> dict[str, Any]:
    root = _resolve_local_path(payload.get("root") or "", allow_file=True)
    cache_path = _parquet_list_cache_path(root, payload)
    if payload.get("no_cache") is not True:
        cached = _read_dataset_summary_cache(cache_path)
        if cached is not None:
            cached["cache"] = {"hit": True, "path": _short_path(cache_path)}
            return cached
    bbox_only = payload.get("bbox_only", False) is True
    required = {"frame_index", "x", "y", "length", "width", "yaw", "source"}
    if root.is_file():
        paths = [root] if root.suffix == ".parquet" else []
    else:
        limit = int(payload.get("limit") or 2000)
        paths = []
        for path in root.rglob("*.parquet"):
            if ".dashboard_cache" in path.parts or "__pycache__" in path.parts:
                continue
            paths.append(path.resolve())
            if len(paths) >= limit:
                break
    items = []
    skipped = 0
    for p in sorted(paths):
        if bbox_only:
            try:
                cols = set(_columns(p))
            except Exception:
                skipped += 1
                continue
            if not required.issubset(cols):
                skipped += 1
                continue
        items.append({"path": str(p), "name": p.name, "display": _short_path(p)})
    result = {"items": items, "root": str(root), "skipped": skipped, "cache": {"hit": False, "path": _short_path(cache_path)}}
    if payload.get("no_cache") is not True:
        _write_dataset_summary_cache(cache_path, result)
    return result


def describe(payload: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_local_path(payload.get("path"))
    cols = _columns(path)
    topics = [t for t in DEFAULT_TOPICS if t in cols]
    return {"path": str(path), "display": _short_path(path), "columns": cols, "preferred_topics": topics}


def values(payload: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_local_path(payload.get("path"))
    cols = _columns(path)
    column = str(payload.get("column") or "")
    if column not in cols or column not in FILTER_COLUMNS:
        raise ValueError(f"Unsupported value column: {column}")
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    where, params = _where_from_filters(cols, filters)
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""
            SELECT DISTINCT {column} AS v
            FROM parquet_scan(?)
            WHERE {" AND ".join(where)} AND {column} IS NOT NULL
            ORDER BY v
            LIMIT ?
            """,
            [str(path)] + params + [int(payload.get("limit") or 5000)],
        ).fetchall()
    finally:
        con.close()
    return {"values": [str(row[0]) for row in rows if row and row[0] is not None]}


def scenarios(payload: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_local_path(payload.get("path"))
    cols = _columns(path)
    group_cols = [c for c in ("suite_name", "scenario_name", "t4dataset_name", "topic_name") if c in cols]
    if not group_cols:
        return {"items": []}
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    where, params = _where_from_filters(cols, filters)
    q = _as_text(payload.get("q")).lower()
    search_cols = [c for c in ("suite_name", "scenario_name", "t4dataset_name", "topic_name", "label", "status") if c in cols]
    if q and search_cols:
        where.append("(" + " OR ".join([f"LOWER(CAST({c} AS VARCHAR)) LIKE ?" for c in search_cols]) + ")")
        params.extend([f"%{q}%"] * len(search_cols))
    con = duckdb.connect()
    try:
        df = con.execute(
            f"""
            SELECT
                {", ".join(group_cols)},
                COUNT(*) AS rows,
                COUNT(DISTINCT TRY_CAST(frame_index AS INTEGER)) AS frames,
                MIN(TRY_CAST(frame_index AS INTEGER)) AS first_frame,
                MAX(TRY_CAST(frame_index AS INTEGER)) AS last_frame
            FROM parquet_scan(?)
            WHERE {" AND ".join(where)}
            GROUP BY {", ".join(group_cols)}
            ORDER BY rows DESC
            LIMIT ?
            """,
            [str(path)] + params + [int(payload.get("limit") or 300)],
        ).df()
    finally:
        con.close()
    return {"items": df.to_dict("records")}


def _dataset_summary_cache_dir() -> Path:
    return app_paths.cache_root() / "local_bbox_api" / "dataset_summary"


def _path_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _devops_metadata_signature(parquet_path: Path) -> dict[str, Any]:
    run_dir = parquet_path.parent
    files = []
    for rel in ("resources/summary.json", "summary.json"):
        candidate = run_dir / rel
        if candidate.exists():
            files.append(_path_signature(candidate))
    dirs = []
    try:
        for child in run_dir.iterdir():
            if child.is_dir() and child.name.startswith("DevOps_"):
                stat = child.stat()
                dirs.append({"path": child.name, "mtime_ns": stat.st_mtime_ns})
    except OSError:
        pass
    return {"summary_files": files, "suite_dirs": sorted(dirs, key=lambda x: x["path"])}


def _dataset_summary_cache_path(path: Path, payload: dict[str, Any], cols: list[str]) -> Path:
    key_payload = {
        "version": 11,
        "path": _path_signature(path),
        "devops_metadata": _devops_metadata_signature(path),
        "filters": payload.get("filters") if isinstance(payload.get("filters"), dict) else {},
        "limit": int(payload.get("limit") or 800),
        "include_criteria_results": payload.get("include_criteria_results") is True,
        "columns": cols,
    }
    digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    return _dataset_summary_cache_dir() / f"{digest}.json"


def _read_dataset_summary_cache(cache_path: Path) -> dict[str, Any] | None:
    try:
        if not cache_path.is_file():
            return None
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_dataset_summary_cache(cache_path: Path, result: dict[str, Any]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(_json_safe(result), separators=(",", ":"), ensure_ascii=False, allow_nan=False), encoding="utf-8")
        tmp.replace(cache_path)
    except Exception:
        pass


def dataset_summary(payload: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_local_path(payload.get("path"))
    cols = _columns(path)
    cache_path = _dataset_summary_cache_path(path, payload, cols)
    if payload.get("no_cache") is not True:
        cached = _read_dataset_summary_cache(cache_path)
        if cached is not None:
            cached["cache"] = {"hit": True, "path": _short_path(cache_path)}
            return cached
    result = _dataset_summary_uncached(payload, path=path, cols=cols)
    result["cache"] = {"hit": False, "path": _short_path(cache_path)}
    if payload.get("no_cache") is not True:
        _write_dataset_summary_cache(cache_path, result)
    return result


def _dataset_summary_uncached(payload: dict[str, Any], *, path: Path | None = None, cols: list[str] | None = None) -> dict[str, Any]:
    path = path or _resolve_local_path(payload.get("path"))
    cols = cols or _columns(path)
    _require_columns(cols, ("frame_index", "source", "x", "y", "length", "width", "yaw"))
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    where, params = _where_from_filters(cols, filters)
    group_cols = [c for c in ("suite_name", "scenario_name", "t4dataset_name", "topic_name") if c in cols]
    if "scenario_name" not in group_cols:
        raise ValueError("dataset_summary requires scenario_name column")
    label_expr = "COALESCE(NULLIF(CAST(label AS VARCHAR), ''), 'unknown')" if "label" in cols else "'unknown'"
    status_expr = "UPPER(COALESCE(NULLIF(CAST(status AS VARCHAR), ''), ''))" if "status" in cols else "''"
    source_expr = "UPPER(COALESCE(NULLIF(CAST(source AS VARCHAR), ''), ''))"
    center_error_expr = "TRY_CAST(center_distance AS DOUBLE)" if "center_distance" in cols else "NULL"
    con = duckdb.connect()
    try:
        scenario_df = con.execute(
            f"""
            SELECT
                {", ".join(group_cols)},
                COUNT(*) AS rows,
                COUNT(DISTINCT TRY_CAST(frame_index AS INTEGER)) AS frames,
                MIN(TRY_CAST(frame_index AS INTEGER)) AS first_frame,
                MAX(TRY_CAST(frame_index AS INTEGER)) AS last_frame,
                SUM(CASE WHEN {source_expr} = 'GT' THEN 1 ELSE 0 END) AS gt,
                SUM(CASE WHEN {source_expr} = 'EST' THEN 1 ELSE 0 END) AS est,
                SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'TP' THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'FP' THEN 1 ELSE 0 END) AS fp,
                SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'FN' THEN 1 ELSE 0 END) AS fn,
                AVG(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'TP' THEN {center_error_expr} ELSE NULL END) AS avg_tp_error,
                MAX(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'TP' THEN {center_error_expr} ELSE NULL END) AS max_tp_error
            FROM parquet_scan(?)
            WHERE {" AND ".join(where)}
              AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
            GROUP BY {", ".join(group_cols)}
            ORDER BY fp DESC, fn DESC, rows DESC
            LIMIT ?
            """,
            [str(path)] + params + [int(payload.get("limit") or 800)],
        ).df()
        label_df = con.execute(
            f"""
            SELECT
                {", ".join(group_cols)},
                {label_expr} AS label,
                SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'TP' THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'FP' THEN 1 ELSE 0 END) AS fp,
                SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'FN' THEN 1 ELSE 0 END) AS fn,
                COUNT(*) AS rows
            FROM parquet_scan(?)
            WHERE {" AND ".join(where)}
              AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
            GROUP BY {", ".join(group_cols)}, label
            """,
            [str(path)] + params,
        ).df()
    finally:
        con.close()

    label_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    label_totals: dict[str, dict[str, int]] = {}
    for row in label_df.to_dict("records"):
        key = tuple(str(row.get(c) or "") for c in ("suite_name", "scenario_name", "t4dataset_name", "topic_name"))
        item = {
            "label": _as_text(row.get("label")) or "unknown",
            "tp": int(row.get("tp") or 0),
            "fp": int(row.get("fp") or 0),
            "fn": int(row.get("fn") or 0),
            "rows": int(row.get("rows") or 0),
        }
        label_by_key.setdefault(key, []).append(item)
        total = label_totals.setdefault(item["label"], {"tp": 0, "fp": 0, "fn": 0, "rows": 0})
        for metric in ("tp", "fp", "fn", "rows"):
            total[metric] += item[metric]

    scenario_records = scenario_df.to_dict("records")
    contexts_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    include_criteria_results = payload.get("include_criteria_results") is True
    if include_criteria_results:
        for row in scenario_records:
            key = tuple(str(row.get(c) or "") for c in group_cols)
            context = _load_scenario_context(path, _as_text(row.get("suite_name")), _as_text(row.get("scenario_name")))
            if context.get("is_devops") and context.get("criteria"):
                contexts_by_key[key] = context
    criteria_results = (
        _batch_devops_criteria_results(path, cols, where, params, group_cols, contexts_by_key)
        if include_criteria_results
        else {}
    )

    scenarios_out: list[dict[str, Any]] = []
    loaded_by_suite: dict[str, set[str]] = {}
    suite_display_names: dict[str, str] = {}
    for row in scenario_records:
        key = tuple(str(row.get(c) or "") for c in ("suite_name", "scenario_name", "t4dataset_name", "topic_name"))
        labels = sorted(label_by_key.get(key, []), key=lambda item: (item["fp"], item["fn"], item["rows"]), reverse=True)
        scenario_name_text = _as_text(row.get("scenario_name"))
        suite_name_text = _as_text(row.get("suite_name"))
        suite_key = _suite_base_name(suite_name_text)
        loaded_by_suite.setdefault(suite_key, set()).add(scenario_name_text)
        suite_display_names.setdefault(suite_key, suite_name_text)
        context = (
            _load_scenario_context(path, suite_name_text, scenario_name_text)
            if include_criteria_results
            else _load_scenario_context_light(path, suite_name_text, scenario_name_text)
        )
        criteria_result = criteria_results.get(tuple(str(row.get(c) or "") for c in group_cols))
        if criteria_result:
            context = {
                **context,
                "criteria_result": criteria_result,
            }
        tp = int(row.get("tp") or 0)
        fp = int(row.get("fp") or 0)
        fn = int(row.get("fn") or 0)
        gt = int(row.get("gt") or 0)
        est = int(row.get("est") or 0)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        scenarios_out.append(
            {
                **{c: _as_text(row.get(c)) for c in group_cols},
                "rows": int(row.get("rows") or 0),
                "frames": int(row.get("frames") or 0),
                "first_frame": None if row.get("first_frame") is None else int(row.get("first_frame")),
                "last_frame": None if row.get("last_frame") is None else int(row.get("last_frame")),
                "gt": gt,
                "est": est,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "fpr": fp / est if est else None,
                "fnr": fn / gt if gt else None,
                "avg_tp_error": None if row.get("avg_tp_error") is None else _as_float(row.get("avg_tp_error")),
                "max_tp_error": None if row.get("max_tp_error") is None else _as_float(row.get("max_tp_error")),
                "labels": labels[:12],
                "devops": context,
            }
        )
    for suite_key, loaded_names in loaded_by_suite.items():
        suite_name_text = suite_display_names.get(suite_key) or suite_key
        suite_pass = _suite_pass_summary(path, suite_name_text)
        if not suite_pass or suite_pass.get("total", 0) <= len(loaded_names):
            continue
        for scenario_name_text in _suite_scenario_inventory(path, suite_name_text):
            if scenario_name_text in loaded_names:
                continue
            context = (
                _load_scenario_context(path, suite_name_text, scenario_name_text)
                if include_criteria_results
                else _load_scenario_context_light(path, suite_name_text, scenario_name_text)
            )
            scenarios_out.append(
                {
                    **{c: "" for c in group_cols},
                    "suite_name": suite_name_text,
                    "scenario_name": scenario_name_text,
                    "rows": 0,
                    "frames": 0,
                    "first_frame": None,
                    "last_frame": None,
                    "gt": 0,
                    "est": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "precision": None,
                    "recall": None,
                    "fpr": None,
                    "fnr": None,
                    "avg_tp_error": None,
                    "max_tp_error": None,
                    "labels": [],
                    "devops": {
                        **context,
                        "unavailable": True,
                        "unavailable_reason": "Scenario exists in the DevOps suite folder but has no bbox/evaluation rows in this parquet.",
                    },
                }
            )
    return {
        "items": scenarios_out,
        "labels": [{"label": label, **metrics} for label, metrics in sorted(label_totals.items())],
        "path": str(path),
        "display": _short_path(path),
    }


def scenario_curve(payload: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_local_path(payload.get("path"))
    cols = _columns(path)
    _require_columns(cols, ("frame_index", "source", "x", "y", "length", "width", "yaw"))
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    where, params = _where_from_filters(cols, filters)
    label = _as_text(payload.get("label"))
    if label and "label" in cols:
        where.append("label = ?")
        params.append(label)
    source_expr = "UPPER(COALESCE(NULLIF(CAST(source AS VARCHAR), ''), ''))"
    status_expr = "UPPER(COALESCE(NULLIF(CAST(status AS VARCHAR), ''), ''))" if "status" in cols else "''"
    center_error_expr = "TRY_CAST(center_distance AS DOUBLE)" if "center_distance" in cols else "NULL"
    con = duckdb.connect()
    try:
        df = con.execute(
            f"""
            SELECT
                TRY_CAST(frame_index AS INTEGER) AS frame,
                SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'TP' THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'FP' THEN 1 ELSE 0 END) AS fp,
                SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'FN' THEN 1 ELSE 0 END) AS fn,
                SUM(CASE WHEN {source_expr} = 'GT' THEN 1 ELSE 0 END) AS gt,
                SUM(CASE WHEN {source_expr} = 'EST' THEN 1 ELSE 0 END) AS est,
                MAX(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'TP' THEN {center_error_expr} ELSE NULL END) AS max_tp_error
            FROM parquet_scan(?)
            WHERE {" AND ".join(where)}
              AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
            GROUP BY frame
            ORDER BY frame
            LIMIT ?
            """,
            [str(path)] + params + [int(payload.get("limit") or 5000)],
        ).df()
    finally:
        con.close()
    return {"frames": df.to_dict("records")}


def _batch_devops_criteria_results(
    parquet_path: Path,
    cols: list[str],
    where: list[str],
    params: list[Any],
    group_cols: list[str],
    contexts_by_key: dict[tuple[str, ...], dict[str, Any]],
) -> dict[tuple[str, ...], dict[str, Any]]:
    if not contexts_by_key or "status" not in cols or "x" not in cols or "y" not in cols:
        return {}
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    gates_by_key: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    signatures: dict[tuple[Any, ...], list[tuple[tuple[str, ...], int]]] = {}
    for key, context in contexts_by_key.items():
        criteria = context.get("criteria") or []
        evaluation_task = _as_text(context.get("evaluation_task")).lower()
        for idx, criterion in enumerate(criteria):
            method = _as_text(criterion.get("method")).lower()
            if method not in {"num_gt_tp", "num_tp", "yaw_error"}:
                continue
            low, high, distance_label = _parse_distance_filter((criterion.get("filter") or {}).get("Distance"))
            distance_label = _criteria_filter_label(criterion.get("filter") or {})
            level = _criteria_level_value(criterion.get("level"))
            if not math.isfinite(level):
                level = 0.0 if method == "yaw_error" else 100.0
            pass_rate = _as_float(criterion.get("pass_rate"), default=0.0)
            signature = (method, evaluation_task, low, high, distance_label, level, pass_rate)
            signatures.setdefault(signature, []).append((key, idx))
            gates_by_key.setdefault(key, []).append(
                {
                    "index": idx,
                    "method": method,
                    "distance_label": distance_label,
                    "required_rate": pass_rate / 100.0,
                    "actual_rate": None,
                    "passed": None,
                    "filter": criterion.get("filter") or {},
                    "source": "parquet_fallback",
                }
            )
    if not signatures:
        return {}

    group_select = ", ".join(group_cols)
    source_expr = "UPPER(COALESCE(NULLIF(CAST(source AS VARCHAR), ''), ''))"
    status_expr = "UPPER(COALESCE(NULLIF(CAST(status AS VARCHAR), ''), ''))"
    label_expr = "LOWER(COALESCE(NULLIF(CAST(label AS VARCHAR), ''), ''))" if "label" in cols else "''"
    yaw_expr = "TRY_CAST(yaw_error AS DOUBLE)" if "yaw_error" in cols else "CAST(NULL AS DOUBLE)"
    con = duckdb.connect()
    try:
        for signature, members in signatures.items():
            method, evaluation_task, low, high, _distance_label, level, pass_rate = signature
            gate_where = list(where)
            gate_params = list(params)
            if low is not None:
                gate_where.append("SQRT(TRY_CAST(x AS DOUBLE) * TRY_CAST(x AS DOUBLE) + TRY_CAST(y AS DOUBLE) * TRY_CAST(y AS DOUBLE)) >= ?")
                gate_params.append(low)
            if high is not None:
                gate_where.append("SQRT(TRY_CAST(x AS DOUBLE) * TRY_CAST(x AS DOUBLE) + TRY_CAST(y AS DOUBLE) * TRY_CAST(y AS DOUBLE)) < ?")
                gate_params.append(high)
            if method in {"num_gt_tp", "num_tp"}:
                if method == "num_gt_tp":
                    success_expr = "gt_tp + gt_tn" if evaluation_task == "fp_validation" else "gt_tp"
                    fail_expr = "est_fp" if evaluation_task == "fp_validation" else "gt_fn"
                    denominator_expr = "est_fp + gt_tn" if evaluation_task == "fp_validation" else f"({success_expr}) + ({fail_expr})"
                else:
                    success_expr = "gt_tp + gt_tn"
                    fail_expr = "gt_fn + est_fp"
                    denominator_expr = f"({success_expr}) + ({fail_expr})"
                df = con.execute(
                    f"""
                    WITH frame_counts AS (
                        SELECT
                            {group_select},
                            TRY_CAST(frame_index AS INTEGER) AS frame,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'TP' THEN 1 ELSE 0 END) AS gt_tp,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'FP' AND {label_expr} = 'false_positive' THEN 1 ELSE 0 END) AS gt_fp_validation,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'FN' THEN 1 ELSE 0 END) AS gt_fn,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'TN' THEN 1 ELSE 0 END) AS gt_tn,
                            SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'FP' THEN 1 ELSE 0 END) AS est_fp
                        FROM parquet_scan(?)
                        WHERE {" AND ".join(gate_where)}
                          AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
                        GROUP BY {group_select}, frame
                    ),
                    scored AS (
                        SELECT
                            {group_select},
                            frame,
                            {success_expr} AS object_success,
                            {fail_expr} AS object_fail,
                            CASE WHEN {denominator_expr} = 0
                                THEN NULL
                                ELSE 100.0 * ({success_expr}) / ({denominator_expr})
                            END AS frame_score
                        FROM frame_counts
                        WHERE {denominator_expr} > 0
                    )
                    SELECT
                        {group_select},
                        SUM(CASE WHEN frame_score >= ? THEN 1 ELSE 0 END) AS passed_frames,
                        COUNT(*) AS total_frames,
                        SUM(CASE WHEN frame_score < ? THEN 1 ELSE 0 END) AS failed_frames
                    FROM scored
                    GROUP BY {group_select}
                    """,
                    [str(parquet_path)] + gate_params + [level, level],
                ).df()
            else:
                df = con.execute(
                    f"""
                    WITH frame_scores AS (
                        SELECT
                            {group_select},
                            TRY_CAST(frame_index AS INTEGER) AS frame,
                            AVG(ABS({yaw_expr})) FILTER (WHERE {source_expr} = 'EST' AND {status_expr} = 'TP' AND {yaw_expr} IS NOT NULL) AS frame_score,
                            COUNT(*) FILTER (WHERE {source_expr} = 'EST' AND {status_expr} = 'TP' AND {yaw_expr} IS NOT NULL) AS yaw_objects
                        FROM parquet_scan(?)
                        WHERE {" AND ".join(gate_where)}
                          AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
                        GROUP BY {group_select}, frame
                    )
                    SELECT
                        {group_select},
                        SUM(CASE WHEN frame_score <= ? THEN 1 ELSE 0 END) AS passed_frames,
                        COUNT(*) AS total_frames,
                        SUM(CASE WHEN frame_score > ? THEN 1 ELSE 0 END) AS failed_frames
                    FROM frame_scores
                    WHERE yaw_objects > 0
                    GROUP BY {group_select}
                    """,
                    [str(parquet_path)] + gate_params + [level, level],
                ).df()
            scored = {
                tuple(_as_text(row.get(c)) for c in group_cols): row
                for row in df.to_dict("records")
            }
            for key, idx in members:
                row = scored.get(key)
                passed_count = int(row.get("passed_frames") or 0) if row else 0
                total_count = int(row.get("total_frames") or 0) if row else 0
                actual_rate = passed_count / total_count if total_count else None
                passed = (actual_rate * 100 >= pass_rate) if actual_rate is not None else None
                for gate in gates_by_key.get(key, []):
                    if gate["index"] == idx:
                        gate.update({"actual_rate": actual_rate, "passed": passed})
                        break
    finally:
        con.close()

    for key, gates in gates_by_key.items():
        known = [g for g in gates if g.get("passed") is not None]
        failed = [g for g in known if g.get("passed") is False]
        if not known:
            continue
        explanation = "All supported perception criteria pass for this scenario."
        if failed:
            worst = sorted(failed, key=lambda g: (g["actual_rate"] or 0) - g["required_rate"])[0]
            actual = "-" if worst["actual_rate"] is None else f"{worst['actual_rate'] * 100:.1f}%"
            explanation = (
                f"Criterion {worst['index'] + 1} fails: {worst['method']} is {actual}, "
                f"below required {worst['required_rate'] * 100:.1f}% in {worst['distance_label']}."
            )
        out[key] = {
            "overall_pass": not failed,
            "failed_count": len(failed),
            "gate_count": len(known),
            "explanation": explanation,
            "source": "parquet_fallback",
            "warning": "scene_result.pkl was not available; criteria were reconstructed from flattened parquet rows.",
        }
    return out


def _prebaked(route: str, path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Stored answer for a pickle-backed route, or ``None`` to compute it normally.

    A miss -- including a malformed cache -- must never fail the request, so the
    caller always falls through to the live pickle path.
    """
    if not prebake.enabled():
        return None
    try:
        return prebake.read(route, path, payload)
    except Exception as exc:  # pragma: no cover - defensive; cache is never required.
        logging.getLogger(__name__).warning("prebake lookup failed for %s: %s", route, exc)
        return None


def scenario_devops_result(payload: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_local_path(payload.get("path"))
    cached = _prebaked(prebake.ROUTE_DEVOPS_RESULT, path, payload)
    if cached is not None:
        return cached
    cols = _columns(path)
    _require_columns(cols, ("frame_index", "source", "status", "x", "y"))
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    where, params = _where_from_filters(cols, filters)
    suite_name = _as_text(filters.get("suite_name"))
    scenario_name = _as_text(filters.get("scenario_name"))
    if not scenario_name:
        raise ValueError("scenario_devops_result requires scenario_name filter")
    context = _load_scenario_context(path, suite_name, scenario_name)
    criteria = context.get("criteria") or []
    evaluation_task = _as_text(context.get("evaluation_task")).lower()
    use_exact_pickle = payload.get("exact") is True or payload.get("use_pickle") is True
    if use_exact_pickle:
        pickle_gates = _pickle_devops_gates(path, suite_name, scenario_name, criteria, evaluation_task)
    else:
        pickle_gates = _pickle_fp_validation_gates_fast(path, suite_name, scenario_name, criteria, evaluation_task)
    source_expr = "UPPER(COALESCE(NULLIF(CAST(source AS VARCHAR), ''), ''))"
    status_expr = "UPPER(COALESCE(NULLIF(CAST(status AS VARCHAR), ''), ''))"
    label_expr = "LOWER(COALESCE(NULLIF(CAST(label AS VARCHAR), ''), ''))" if "label" in cols else "''"
    yaw_expr = "TRY_CAST(yaw_error AS DOUBLE)" if "yaw_error" in cols else "CAST(NULL AS DOUBLE)"
    con = duckdb.connect()
    gates: list[dict[str, Any]] = [] if pickle_gates is None else pickle_gates
    try:
        for idx, criterion in ([] if pickle_gates is not None else list(enumerate(criteria))):
            method = _as_text(criterion.get("method"))
            pass_rate = _as_float(criterion.get("pass_rate"), default=0.0)
            level_raw = criterion.get("level")
            level = _criteria_level_value(level_raw)
            filter_obj = criterion.get("filter") if isinstance(criterion.get("filter"), dict) else {}
            low, high, distance_label = _parse_distance_filter(filter_obj.get("Distance"))
            distance_label = _criteria_filter_label(filter_obj)
            gate_where = list(where)
            gate_params = list(params)
            if low is not None:
                gate_where.append("SQRT(POWER(TRY_CAST(x AS DOUBLE), 2) + POWER(TRY_CAST(y AS DOUBLE), 2)) >= ?")
                gate_params.append(low)
            if high is not None:
                gate_where.append("SQRT(POWER(TRY_CAST(x AS DOUBLE), 2) + POWER(TRY_CAST(y AS DOUBLE), 2)) < ?")
                gate_params.append(high)
            if method == "num_gt_tp":
                if not math.isfinite(level):
                    level = 100.0
                row = con.execute(
                    f"""
                    WITH frame_counts AS (
                        SELECT
                            TRY_CAST(frame_index AS INTEGER) AS frame,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'TP' THEN 1 ELSE 0 END) AS gt_tp,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'FP' AND {label_expr} = 'false_positive' THEN 1 ELSE 0 END) AS gt_fp_validation,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'FN' THEN 1 ELSE 0 END) AS gt_fn,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'TN' THEN 1 ELSE 0 END) AS gt_tn,
                            SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'FP' THEN 1 ELSE 0 END) AS est_fp
                        FROM parquet_scan(?)
                        WHERE {" AND ".join(gate_where)}
                          AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
                        GROUP BY frame
                    ),
                    scored AS (
                        SELECT
                            *,
                            {'gt_tp + gt_tn AS object_success' if evaluation_task == 'fp_validation' else 'gt_tp AS object_success'},
                            {'est_fp AS object_fail' if evaluation_task == 'fp_validation' else 'gt_fn AS object_fail'},
                            {'est_fp + gt_tn AS object_gt' if evaluation_task == 'fp_validation' else 'gt_tp + gt_fn AS object_gt'},
                            CASE
                                WHEN {'est_fp + gt_tn' if evaluation_task == 'fp_validation' else 'gt_tp + gt_fn'} = 0 THEN 100.0
                                ELSE 100.0 * {'gt_tp + gt_tn' if evaluation_task == 'fp_validation' else 'gt_tp'} / ({'est_fp + gt_tn' if evaluation_task == 'fp_validation' else 'gt_tp + gt_fn'})
                            END AS frame_score
                        FROM frame_counts
                        WHERE {'est_fp + gt_tn' if evaluation_task == 'fp_validation' else 'gt_tp + gt_fn'} > 0
                    )
                    SELECT
                        SUM(CASE WHEN frame_score >= ? THEN 1 ELSE 0 END) AS passed_frames,
                        COUNT(*) AS total_frames,
                        SUM(CASE WHEN frame_score < ? THEN 1 ELSE 0 END) AS failed_frames,
                        SUM(object_success) AS object_success_count,
                        SUM(object_fail) AS object_fail_count,
                        SUM(object_gt) AS object_total_count
                    FROM scored
                    """,
                    [str(path)] + gate_params + [level, level],
                ).fetchone()
                passed_count = int(row[0] or 0)
                total_count = int(row[1] or 0)
                fail_count = int(row[2] or 0)
                actual_rate = passed_count / total_count if total_count else None
                passed = (actual_rate * 100 >= pass_rate) if actual_rate is not None else None
                object_success_count = int(row[3] or 0)
                object_fail_count = int(row[4] or 0)
                object_total_count = int(row[5] or 0)
                if evaluation_task == "fp_validation":
                    meaning = "Each FP-validation frame must keep validation objects as TN, then enough frames must satisfy PassRate."
                    metric_label = "frame FP-validation pass rate"
                else:
                    meaning = "Each non-empty frame must reach the CriteriaLevel GT recall, then enough frames must satisfy PassRate."
                    metric_label = "frame GT recall pass rate"
            elif method == "num_tp":
                if not math.isfinite(level):
                    level = 100.0
                row = con.execute(
                    f"""
                    WITH frame_counts AS (
                        SELECT
                            TRY_CAST(frame_index AS INTEGER) AS frame,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'TP' THEN 1 ELSE 0 END) AS gt_tp,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'FP' AND {label_expr} = 'false_positive' THEN 1 ELSE 0 END) AS gt_fp_validation,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'TN' THEN 1 ELSE 0 END) AS gt_tn,
                            SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'FN' THEN 1 ELSE 0 END) AS gt_fn,
                            SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'FP' THEN 1 ELSE 0 END) AS est_fp
                        FROM parquet_scan(?)
                        WHERE {" AND ".join(gate_where)}
                          AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
                        GROUP BY frame
                    ),
                    scored AS (
                        SELECT
                            *,
                            gt_tp + gt_tn AS object_success,
                            gt_fn + est_fp AS object_fail,
                            CASE
                                WHEN gt_tp + gt_tn + gt_fn + est_fp = 0 THEN 100.0
                                ELSE 100.0 * (gt_tp + gt_tn) / (gt_tp + gt_tn + gt_fn + est_fp)
                            END AS frame_score
                        FROM frame_counts
                        WHERE gt_tp + gt_tn + gt_fn + est_fp > 0
                    )
                    SELECT
                        SUM(CASE WHEN frame_score >= ? THEN 1 ELSE 0 END) AS passed_frames,
                        COUNT(*) AS total_frames,
                        SUM(CASE WHEN frame_score < ? THEN 1 ELSE 0 END) AS failed_frames,
                        SUM(object_success) AS object_success_count,
                        SUM(object_fail) AS object_fail_count,
                        SUM(object_success + object_fail) AS object_total_count
                    FROM scored
                    """,
                    [str(path)] + gate_params + [level, level],
                ).fetchone()
                passed_count = int(row[0] or 0)
                total_count = int(row[1] or 0)
                fail_count = int(row[2] or 0)
                actual_rate = passed_count / total_count if total_count else None
                passed = (actual_rate * 100 >= pass_rate) if actual_rate is not None else None
                object_success_count = int(row[3] or 0)
                object_fail_count = int(row[4] or 0)
                object_total_count = int(row[5] or 0)
                meaning = "Each non-empty frame must reach the CriteriaLevel TP/TN success rate, then enough frames must satisfy PassRate."
                metric_label = "frame object pass rate"
            elif method == "yaw_error":
                if not math.isfinite(level):
                    level = 0.0
                row = con.execute(
                    f"""
                    WITH frame_scores AS (
                        SELECT
                            TRY_CAST(frame_index AS INTEGER) AS frame,
                            AVG(ABS({yaw_expr})) FILTER (WHERE {source_expr} = 'EST' AND {status_expr} = 'TP' AND {yaw_expr} IS NOT NULL) AS frame_score,
                            COUNT(*) FILTER (WHERE {source_expr} = 'EST' AND {status_expr} = 'TP' AND {yaw_expr} IS NOT NULL) AS yaw_objects
                        FROM parquet_scan(?)
                        WHERE {" AND ".join(gate_where)}
                          AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
                        GROUP BY frame
                    )
                    SELECT
                        SUM(CASE WHEN frame_score <= ? THEN 1 ELSE 0 END) AS passed_frames,
                        COUNT(*) FILTER (WHERE yaw_objects > 0) AS total_frames,
                        SUM(CASE WHEN yaw_objects > 0 AND frame_score > ? THEN 1 ELSE 0 END) AS failed_frames,
                        SUM(CASE WHEN yaw_objects > 0 THEN yaw_objects ELSE 0 END) AS object_total_count,
                        AVG(frame_score) FILTER (WHERE yaw_objects > 0) AS avg_error,
                        MAX(frame_score) FILTER (WHERE yaw_objects > 0) AS max_error
                    FROM frame_scores
                    WHERE yaw_objects > 0
                    """,
                    [str(path)] + gate_params + [level, level],
                ).fetchone()
                passed_count = int(row[0] or 0)
                total_count = int(row[1] or 0)
                fail_count = max(0, total_count - passed_count)
                actual_rate = passed_count / total_count if total_count else None
                passed = (actual_rate * 100 >= pass_rate) if actual_rate is not None else None
                object_success_count = max(0, int((row[3] or 0) - fail_count))
                object_fail_count = fail_count
                object_total_count = int(row[3] or 0)
                meaning = f"Each frame's average TP yaw error must be <= {level:.3f} rad, then enough frames must satisfy PassRate."
                metric_label = "frame yaw-error pass rate"
            else:
                passed_count = 0
                total_count = 0
                fail_count = 0
                actual_rate = None
                passed = None
                object_success_count = 0
                object_fail_count = 0
                object_total_count = 0
                meaning = "Unsupported criterion method in this explorer."
                metric_label = method or "criterion"
            gates.append(
                {
                    "index": idx,
                    "method": method,
                    "metric_label": metric_label,
                    "meaning": meaning,
                    "distance_label": distance_label,
                    "filter": filter_obj,
                    "required_rate": pass_rate / 100.0,
                    "actual_rate": actual_rate,
                    "passed": passed,
                    "passed_count": passed_count,
                    "fail_count": fail_count,
                    "total_count": total_count,
                    "object_success_count": object_success_count,
                    "object_fail_count": object_fail_count,
                    "object_total_count": object_total_count,
                    "criteria_level": _as_text(level_raw),
                    "criteria_level_value": None if not math.isfinite(level) else level,
                    "evaluation_task": evaluation_task,
                    "level": None if not math.isfinite(level) else level,
                    "source": "parquet_fallback",
                }
            )
        frame_df = con.execute(
            f"""
            SELECT
                TRY_CAST(frame_index AS INTEGER) AS frame,
                SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'TP' THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'FP' THEN 1 ELSE 0 END) AS fp,
                SUM(CASE WHEN {source_expr} = 'GT' AND {status_expr} = 'FN' THEN 1 ELSE 0 END) AS fn,
                MAX(CASE WHEN {source_expr} = 'EST' AND {status_expr} = 'TP' THEN ABS({yaw_expr}) ELSE NULL END) AS max_yaw_error
            FROM parquet_scan(?)
            WHERE {" AND ".join(where)}
              AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
            GROUP BY frame
            ORDER BY fn DESC, fp DESC, max_yaw_error DESC NULLS LAST, frame
            LIMIT 12
            """,
            [str(path)] + params,
        ).df()
    finally:
        con.close()
    known_gates = [g for g in gates if g["passed"] is not None]
    failed_gates = [g for g in known_gates if not g["passed"]]
    overall_pass = bool(known_gates) and not failed_gates
    pf = context.get("planning_factor") or {}
    if pf.get("failed_frames"):
        overall_pass = False
    reasons: list[str] = []
    if failed_gates:
        worst = sorted(
            failed_gates,
            key=lambda g: (g["actual_rate"] if g["actual_rate"] is not None else 0) - g["required_rate"],
        )[0]
        actual = "-" if worst["actual_rate"] is None else f"{worst['actual_rate'] * 100:.1f}%"
        reasons.append(
            f"Criterion {worst['index'] + 1} fails: {worst['metric_label']} is {actual}, "
            f"below required {worst['required_rate'] * 100:.1f}% in {worst['distance_label']}."
        )
    elif known_gates:
        reasons.append("All supported perception criteria pass for this scenario.")
    else:
        reasons.append("No supported perception criteria could be evaluated from the flattened bbox rows.")
    if pf.get("path"):
        if pf.get("failed_frames"):
            reasons.append(f"Planning factor check has {pf.get('failed_frames')} failed frames.")
        elif pf.get("passed_frames"):
            reasons.append(f"Planning factor check passes on {pf.get('passed_frames')} frames.")
    return {
        "context": context,
        "overall_pass": overall_pass,
        "failed_count": len(failed_gates),
        "gate_count": len(known_gates),
        "gates": gates,
        "hot_frames": frame_df.to_dict("records"),
        "explanation": reasons,
        "warning": None
        if use_exact_pickle
        else (
            "FP-validation pass rate uses fast TN/FP counts from scene_result.pkl."
            if any(g.get("source") == "scene_result.pkl_fast" for g in gates)
            else "Fast result explanation uses flattened parquet/YAML criteria. Exact pickle-backed frame evidence loads separately in the viewer."
        ),
    }


def scenario_devops_frame_results(payload: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_local_path(payload.get("path"))
    cached = _prebaked(prebake.ROUTE_FRAME_RESULTS, path, payload)
    if cached is not None:
        return cached
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    suite_name = _as_text(filters.get("suite_name"))
    scenario_name = _as_text(filters.get("scenario_name"))
    if not scenario_name:
        raise ValueError("scenario_devops_frame_results requires scenario_name filter")
    context = _load_scenario_context(path, suite_name, scenario_name)
    criteria = context.get("criteria") or []
    if not criteria:
        return {"available": False, "source": "scene_result.pkl", "frames": [], "frame_count": 0, "reason": "No YAML criteria found."}
    pickle_path = _scenario_pickle_path(path, suite_name, scenario_name)
    if not pickle_path:
        return {
            "available": False,
            "source": "scene_result.pkl",
            "frames": [],
            "frame_count": 0,
            "reason": "scene_result.pkl was not found; frame judgement is only approximate from parquet boxes.",
        }
    try:
        from driving_log_replayer_v2.criteria.perception import PerceptionCriteria  # type: ignore
    except Exception:
        _ensure_eval_lib_paths()
        try:
            from driving_log_replayer_v2.criteria.perception import PerceptionCriteria  # type: ignore
        except Exception as exc:
            return {
                "available": False,
                "source": "scene_result.pkl",
                "frames": [],
                "frame_count": 0,
                "reason": f"Evaluator library unavailable: {exc}",
            }
    frame_exact_raw = payload.get("frame_index", filters.get("frame_index"))
    frame_exact = None if frame_exact_raw in (None, "") else int(_as_float(frame_exact_raw))
    frame_min = filters.get("frame_min")
    frame_max = filters.get("frame_max")
    max_frames = min(max(int(payload.get("max_frames") or 600), 1), 5000)
    try:
        frames_in = _load_scene_result_pickle(pickle_path)
    except Exception as exc:
        return {
            "available": False,
            "source": "scene_result.pkl",
            "frames": [],
            "frame_count": 0,
            "reason": f"scene_result.pkl could not be read: {exc}",
        }
    evaluation_task = _as_text(context.get("evaluation_task")).lower()
    out_frames: list[dict[str, Any]] = []
    for frame in frames_in or []:
        frame_index = int(_as_float(getattr(frame, "frame_name", getattr(frame, "frame_index", 0))))
        if frame_exact is not None and frame_index != frame_exact:
            continue
        if frame_min not in (None, "") and frame_index < int(_as_float(frame_min)):
            continue
        if frame_max not in (None, "") and frame_index > int(_as_float(frame_max)):
            continue
        gates = [
            _frame_gate_detail(frame, criterion, idx, evaluation_task, PerceptionCriteria)
            for idx, criterion in enumerate(criteria)
        ]
        judged = [g for g in gates if g.get("judged")]
        failed = [g for g in judged if g.get("passed") is False]
        out_frames.append(
            {
                "frame": frame_index,
                "judged": bool(judged),
                "passed": None if not judged else not failed,
                "gates": gates,
                "source": "scene_result.pkl",
            }
        )
        if len(out_frames) >= max_frames:
            break
    return {
        "available": True,
        "source": "scene_result.pkl",
        "pickle_path": str(pickle_path),
        "frames": out_frames,
        "frame_count": len(out_frames),
        "truncated": len(out_frames) >= max_frames,
    }


def dataset_stats(payload: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_local_path(payload.get("path"))
    cols = _columns(path)
    _require_columns(cols, ("frame_index", "source", "x", "y"))
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    where, params = _where_from_filters(cols, filters)
    label_expr = "COALESCE(NULLIF(CAST(label AS VARCHAR), ''), 'unknown')" if "label" in cols else "'unknown'"
    source_expr = "UPPER(COALESCE(NULLIF(CAST(source AS VARCHAR), ''), ''))"
    status_expr = "UPPER(COALESCE(NULLIF(CAST(status AS VARCHAR), ''), ''))" if "status" in cols else "''"
    x_error_expr = "TRY_CAST(x_error AS DOUBLE)" if "x_error" in cols else "CAST(NULL AS DOUBLE)"
    y_error_expr = "TRY_CAST(y_error AS DOUBLE)" if "y_error" in cols else "CAST(NULL AS DOUBLE)"
    yaw_error_expr = "TRY_CAST(yaw_error AS DOUBLE)" if "yaw_error" in cols else "CAST(NULL AS DOUBLE)"
    suite_expr = "COALESCE(NULLIF(CAST(suite_name AS VARCHAR), ''), '')" if "suite_name" in cols else "''"
    scenario_expr = "COALESCE(NULLIF(CAST(scenario_name AS VARCHAR), ''), '')" if "scenario_name" in cols else "''"
    dataset_expr = "COALESCE(NULLIF(CAST(t4dataset_name AS VARCHAR), ''), '')" if "t4dataset_name" in cols else "''"
    dataset_id_expr = "COALESCE(NULLIF(CAST(t4dataset_id AS VARCHAR), ''), '')" if "t4dataset_id" in cols else "''"
    topic_expr = "COALESCE(NULLIF(CAST(topic_name AS VARCHAR), ''), '')" if "topic_name" in cols else "''"
    base_cte = f"""
        WITH src AS (
            SELECT
                *,
                SQRT(POWER(TRY_CAST(x AS DOUBLE), 2) + POWER(TRY_CAST(y AS DOUBLE), 2)) AS dist_h,
                {label_expr} AS label_norm,
                {source_expr} AS source_norm,
                {status_expr} AS status_norm,
                {suite_expr} AS suite_norm,
                {scenario_expr} AS scenario_norm,
                {dataset_expr} AS dataset_norm,
                {dataset_id_expr} AS dataset_id_norm,
                {topic_expr} AS topic_norm,
                TRY_CAST(frame_index AS INTEGER) AS frame_norm
            FROM parquet_scan(?)
            WHERE {" AND ".join(where)}
              AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
              AND TRY_CAST(x AS DOUBLE) IS NOT NULL
              AND TRY_CAST(y AS DOUBLE) IS NOT NULL
        ),
        bins AS ({DISTANCE_BINS_SQL}),
        binned AS (
            SELECT src.*, bins.distance_bin, bins.bin_idx, bins.bin_label
            FROM src
            JOIN bins ON src.dist_h >= bins.bin_start AND src.dist_h < bins.bin_end
        ),
        stats AS (
            SELECT
                distance_bin,
                bin_idx,
                bin_label,
                label_norm AS label,
                COUNT(*) AS rows,
                SUM(CASE WHEN source_norm = 'GT' THEN 1 ELSE 0 END) AS gt,
                SUM(CASE WHEN source_norm = 'EST' THEN 1 ELSE 0 END) AS est,
                SUM(CASE WHEN source_norm = 'EST' AND status_norm = 'TP' THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN source_norm = 'EST' AND status_norm = 'FP' THEN 1 ELSE 0 END) AS fp,
                SUM(CASE WHEN source_norm = 'GT' AND status_norm = 'FN' THEN 1 ELSE 0 END) AS fn,
                SUM(CASE WHEN source_norm = 'GT' AND status_norm IN ('TP','FN') THEN 1 ELSE 0 END) AS gt_total,
                SUM(CASE WHEN source_norm = 'GT' AND status_norm = 'TP' THEN 1 ELSE 0 END) AS tp_gt,
                SUM(CASE WHEN source_norm = 'EST' AND status_norm IN ('TP','FP') THEN 1 ELSE 0 END) AS est_total,
                SUM(CASE WHEN source_norm = 'EST' AND status_norm = 'FP' THEN 1 ELSE 0 END) AS fp_est
            FROM binned
            GROUP BY distance_bin, bin_idx, bin_label, label_norm
        )
    """
    detail_metrics_sql = """
                COUNT(*) AS rows,
                COUNT(DISTINCT frame_norm) AS frames,
                MIN(frame_norm) AS first_frame,
                MAX(frame_norm) AS last_frame,
                SUM(CASE WHEN source_norm = 'GT' THEN 1 ELSE 0 END) AS gt,
                SUM(CASE WHEN source_norm = 'EST' THEN 1 ELSE 0 END) AS est,
                SUM(CASE WHEN source_norm = 'EST' AND status_norm = 'TP' THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN source_norm = 'EST' AND status_norm = 'FP' THEN 1 ELSE 0 END) AS fp,
                SUM(CASE WHEN source_norm = 'GT' AND status_norm = 'FN' THEN 1 ELSE 0 END) AS fn,
                CASE
                    WHEN SUM(CASE WHEN source_norm = 'EST' AND status_norm IN ('TP','FP') THEN 1 ELSE 0 END) > 0
                    THEN CAST(SUM(CASE WHEN source_norm = 'EST' AND status_norm = 'TP' THEN 1 ELSE 0 END) AS DOUBLE)
                        / SUM(CASE WHEN source_norm = 'EST' AND status_norm IN ('TP','FP') THEN 1 ELSE 0 END)
                    ELSE NULL
                END AS precision,
                CASE
                    WHEN SUM(CASE WHEN source_norm = 'GT' AND status_norm IN ('TP','FN') THEN 1 ELSE 0 END) > 0
                    THEN CAST(SUM(CASE WHEN source_norm = 'GT' AND status_norm = 'TP' THEN 1 ELSE 0 END) AS DOUBLE)
                        / SUM(CASE WHEN source_norm = 'GT' AND status_norm IN ('TP','FN') THEN 1 ELSE 0 END)
                    ELSE NULL
                END AS recall
    """
    con = duckdb.connect()
    try:
        distance_df = con.execute(
            f"""
            {base_cte}
            SELECT
                distance_bin,
                MIN(bin_idx) AS bin_idx,
                MIN(bin_label) AS bin_label,
                SUM(rows) AS rows,
                SUM(gt) AS gt,
                SUM(est) AS est,
                SUM(tp) AS tp,
                SUM(fp) AS fp,
                SUM(fn) AS fn,
                CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE NULL END AS tpr,
                CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE NULL END AS fpr
            FROM stats
            GROUP BY distance_bin
            ORDER BY MIN(bin_idx)
            """,
            [str(path)] + params,
        ).df()
        label_distance_df = con.execute(
            f"""
            {base_cte}
            SELECT
                distance_bin,
                bin_idx,
                bin_label,
                label,
                SUM(rows) AS rows,
                SUM(gt) AS gt,
                SUM(est) AS est,
                SUM(tp) AS tp,
                SUM(fp) AS fp,
                SUM(fn) AS fn,
                CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE NULL END AS tpr,
                CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE NULL END AS fpr
            FROM stats
            GROUP BY distance_bin, bin_idx, bin_label, label
            ORDER BY bin_idx, label
            """,
            [str(path)] + params,
        ).df()
        label_df = con.execute(
            f"""
            {base_cte}
            SELECT
                label,
                SUM(rows) AS rows,
                SUM(gt) AS gt,
                SUM(est) AS est,
                SUM(tp) AS tp,
                SUM(fp) AS fp,
                SUM(fn) AS fn,
                CASE WHEN SUM(tp) + SUM(fp) > 0 THEN CAST(SUM(tp) AS DOUBLE) / (SUM(tp) + SUM(fp)) ELSE NULL END AS precision,
                CASE WHEN SUM(tp) + SUM(fn) > 0 THEN CAST(SUM(tp) AS DOUBLE) / (SUM(tp) + SUM(fn)) ELSE NULL END AS recall
            FROM stats
            GROUP BY label
            ORDER BY label
            """,
            [str(path)] + params,
        ).df()
        error_df = con.execute(
            f"""
            WITH src AS (
                SELECT
                    {label_expr} AS label,
                    {source_expr} AS source_norm,
                    {status_expr} AS status_norm,
                    {x_error_expr} AS x_error_value,
                    {y_error_expr} AS y_error_value,
                    {yaw_error_expr} AS yaw_error_value
                FROM parquet_scan(?)
                WHERE {" AND ".join(where)}
            )
            SELECT
                label,
                AVG(ABS(x_error_value)) FILTER (WHERE source_norm = 'EST' AND status_norm = 'TP' AND x_error_value IS NOT NULL) AS mean_abs_x_error,
                AVG(ABS(y_error_value)) FILTER (WHERE source_norm = 'EST' AND status_norm = 'TP' AND y_error_value IS NOT NULL) AS mean_abs_y_error,
                AVG(ABS(yaw_error_value)) FILTER (WHERE source_norm = 'EST' AND status_norm = 'TP' AND yaw_error_value IS NOT NULL) AS mean_abs_yaw_error
            FROM src
            GROUP BY label
            ORDER BY label
            """,
            [str(path)] + params,
        ).df()
        scenario_df = con.execute(
            f"""
            {base_cte}
            SELECT
                suite_norm AS suite_name,
                scenario_norm AS scenario_name,
                dataset_norm AS t4dataset_name,
                dataset_id_norm AS t4dataset_id,
                topic_norm AS topic_name,
                {detail_metrics_sql}
            FROM binned
            GROUP BY suite_norm, scenario_norm, dataset_norm, dataset_id_norm, topic_norm
            ORDER BY fp DESC, fn DESC, rows DESC
            LIMIT 800
            """,
            [str(path)] + params,
        ).df()
        dataset_df = con.execute(
            f"""
            {base_cte}
            SELECT
                suite_norm AS suite_name,
                scenario_norm AS scenario_name,
                dataset_norm AS t4dataset_name,
                dataset_id_norm AS t4dataset_id,
                topic_norm AS topic_name,
                {detail_metrics_sql}
            FROM binned
            GROUP BY suite_norm, scenario_norm, dataset_norm, dataset_id_norm, topic_norm
            ORDER BY fp DESC, fn DESC, rows DESC
            LIMIT 1200
            """,
            [str(path)] + params,
        ).df()
        frame_df = con.execute(
            f"""
            {base_cte}
            SELECT
                suite_norm AS suite_name,
                scenario_norm AS scenario_name,
                dataset_norm AS t4dataset_name,
                dataset_id_norm AS t4dataset_id,
                topic_norm AS topic_name,
                frame_norm AS frame,
                {detail_metrics_sql}
            FROM binned
            GROUP BY suite_norm, scenario_norm, dataset_norm, dataset_id_norm, topic_norm, frame_norm
            ORDER BY fp DESC, fn DESC, rows DESC
            LIMIT 2000
            """,
            [str(path)] + params,
        ).df()
        label_scenario_df = con.execute(
            f"""
            {base_cte}
            SELECT
                label_norm AS label,
                suite_norm AS suite_name,
                scenario_norm AS scenario_name,
                dataset_norm AS t4dataset_name,
                dataset_id_norm AS t4dataset_id,
                topic_norm AS topic_name,
                {detail_metrics_sql}
            FROM binned
            GROUP BY label_norm, suite_norm, scenario_norm, dataset_norm, dataset_id_norm, topic_norm
            ORDER BY fp DESC, fn DESC, rows DESC
            LIMIT 1600
            """,
            [str(path)] + params,
        ).df()
        label_dataset_df = con.execute(
            f"""
            {base_cte}
            SELECT
                label_norm AS label,
                suite_norm AS suite_name,
                scenario_norm AS scenario_name,
                dataset_norm AS t4dataset_name,
                dataset_id_norm AS t4dataset_id,
                topic_norm AS topic_name,
                {detail_metrics_sql}
            FROM binned
            GROUP BY label_norm, suite_norm, scenario_norm, dataset_norm, dataset_id_norm, topic_norm
            ORDER BY fp DESC, fn DESC, rows DESC
            LIMIT 2200
            """,
            [str(path)] + params,
        ).df()
        label_frame_df = con.execute(
            f"""
            {base_cte}
            SELECT
                label_norm AS label,
                suite_norm AS suite_name,
                scenario_norm AS scenario_name,
                dataset_norm AS t4dataset_name,
                dataset_id_norm AS t4dataset_id,
                topic_norm AS topic_name,
                frame_norm AS frame,
                {detail_metrics_sql}
            FROM binned
            GROUP BY label_norm, suite_norm, scenario_norm, dataset_norm, dataset_id_norm, topic_norm, frame_norm
            ORDER BY fp DESC, fn DESC, rows DESC
            LIMIT 3000
            """,
            [str(path)] + params,
        ).df()
    finally:
        con.close()
    return {
        "distance": distance_df.to_dict("records"),
        "label_distance": label_distance_df.to_dict("records"),
        "labels": label_df.to_dict("records"),
        "errors": error_df.to_dict("records"),
        "scenarios": scenario_df.to_dict("records"),
        "datasets": dataset_df.to_dict("records"),
        "frames": frame_df.to_dict("records"),
        "label_scenarios": label_scenario_df.to_dict("records"),
        "label_datasets": label_dataset_df.to_dict("records"),
        "label_frames": label_frame_df.to_dict("records"),
        "path": str(path),
        "display": _short_path(path),
    }


def _row_footprint_base_link(row: dict[str, Any]) -> list[list[float]] | None:
    """Transform an object-local footprint (analyzer >=0.2.0) into base_link vertices.

    Returns None when the row has no footprint (boxes, older parquet). Reuses the analyzer's
    footprint_to_base_link so the rotation matches the library, with a local fallback.
    """
    fp = row.get("footprint")
    if fp is None:
        return None
    # Missing values may arrive as a float NaN rather than None for object columns.
    if isinstance(fp, float):
        return None
    try:
        pts = [[float(p[0]), float(p[1])] for p in fp]
    except (TypeError, ValueError, IndexError):
        return None
    if not pts:
        return None

    x = _as_float(row.get("x"))
    y = _as_float(row.get("y"))
    yaw = _as_float(row.get("yaw"))
    try:
        from perception_catalog_analyzer.dataframe import footprint_to_base_link

        return footprint_to_base_link(pts, x, y, yaw)
    except Exception:
        import math

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        return [[p[0] * cos_y - p[1] * sin_y + x, p[0] * sin_y + p[1] * cos_y + y] for p in pts]


def _enum_or_text(value: Any) -> str:
    if value is None:
        return ""
    inner = getattr(value, "value", None)
    return _as_text(inner if inner is not None else value)


def _seq_item(value: Any, index: int, default: float = 0.0) -> float:
    try:
        return _as_float(value[index], default)
    except Exception:
        return default


def _orientation_yaw(orientation: Any) -> float:
    if orientation is None:
        return 0.0
    try:
        ypr = getattr(orientation, "yaw_pitch_roll")
        return _as_float(ypr[0])
    except Exception:
        pass
    elements = getattr(orientation, "elements", orientation)
    try:
        vals = [float(v) for v in elements]
    except Exception:
        return 0.0
    if len(vals) != 4:
        return 0.0
    # pyquaternion stores [w, x, y, z]; ROS-style arrays are commonly [x, y, z, w].
    if abs(vals[0]) >= abs(vals[3]):
        w, x, y, z = vals
    else:
        x, y, z, w = vals
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _dynamic_label(dynamic_object: Any) -> str:
    semantic_label = getattr(dynamic_object, "semantic_label", None)
    label = getattr(semantic_label, "label", None)
    return _enum_or_text(label) or _enum_or_text(semantic_label)


def _dynamic_object_box(dynamic_object: Any, transforms: Any = None, run_label: str = "A") -> dict[str, Any] | None:
    try:
        from perception_catalog_analyzer.dataframe.record import ObjectRecord

        box = ObjectRecord.from_dynamic_object(dynamic_object, transforms).as_dict()
    except Exception:
        state = getattr(dynamic_object, "state", None)
        if state is None:
            return None
        position = getattr(state, "position", None)
        shape = getattr(state, "shape", None)
        size = getattr(shape, "size", None) or getattr(state, "size", None) or (0.0, 0.0, 0.0)
        width, length, height = _seq_item(size, 0), _seq_item(size, 1), _seq_item(size, 2, 1.5)
        velocity = getattr(state, "velocity", None)
        shape_type = _enum_or_text(getattr(state, "shape_type", None)) or _enum_or_text(getattr(shape, "type", None))
        footprint = None
        raw_footprint = getattr(shape, "footprint", None)
        try:
            coords = list(raw_footprint.exterior.coords)[:-1] if raw_footprint is not None else []
            if coords:
                footprint = [[float(p[0]), float(p[1])] for p in coords]
        except Exception:
            footprint = None
        box = {
            "unix_time": getattr(dynamic_object, "unix_time", None),
            "frame_id": _enum_or_text(getattr(dynamic_object, "frame_id", None)),
            "x": _seq_item(position, 0),
            "y": _seq_item(position, 1),
            "z": _seq_item(position, 2),
            "length": length,
            "width": width,
            "height": height,
            "yaw": _orientation_yaw(getattr(state, "orientation", None)),
            "shape_type": shape_type,
            "vx": None if velocity is None else _seq_item(velocity, 0),
            "vy": None if velocity is None else _seq_item(velocity, 1),
            "confidence": None if getattr(dynamic_object, "semantic_score", None) is None else _as_float(getattr(dynamic_object, "semantic_score")),
            "label": _dynamic_label(dynamic_object),
            "pointcloud_num": getattr(dynamic_object, "pointcloud_num", None),
            "uuid": getattr(dynamic_object, "uuid", None),
            "visibility": None if getattr(dynamic_object, "visibility", None) is None else str(getattr(dynamic_object, "visibility")),
            "footprint": footprint,
        }
    box.update(
        {
            "source": "GT",
            "status": "TN",
            "run": run_label,
            "evaluator_status": "TN",
            "devops_exact_source": "scene_result.pkl",
        }
    )
    if not box.get("label"):
        box["label"] = "false_positive"
    if box.get("footprint"):
        footprint_base_link = _row_footprint_base_link(box)
        if footprint_base_link:
            box["footprint"] = footprint_base_link
    return box


def scenario_devops_tn_objects(payload: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_local_path(payload.get("path"))
    cached = _prebaked(prebake.ROUTE_TN_OBJECTS, path, payload)
    if cached is not None:
        return cached
    run_label = _as_text(payload.get("run")) or "A"
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    suite_name = _as_text(filters.get("suite_name"))
    scenario_name = _as_text(filters.get("scenario_name"))
    if not scenario_name:
        raise ValueError("scenario_devops_tn_objects requires scenario_name filter")
    pickle_path = _scenario_pickle_path(path, suite_name, scenario_name)
    if not pickle_path:
        return {"available": False, "source": "scene_result.pkl", "frames": [], "row_count": 0, "frame_count": 0}
    frame_exact_raw = payload.get("frame_index", filters.get("frame_index"))
    frame_exact = None if frame_exact_raw in (None, "") else int(_as_float(frame_exact_raw))
    frame_min = filters.get("frame_min")
    frame_max = filters.get("frame_max")
    max_rows = min(max(int(payload.get("max_rows") or 120000), 1), 600000)
    try:
        frames_in = _load_scene_result_pickle(pickle_path)
    except Exception as exc:
        return {
            "available": False,
            "source": "scene_result.pkl",
            "frames": [],
            "row_count": 0,
            "frame_count": 0,
            "error": str(exc),
        }
    out: dict[int, list[dict[str, Any]]] = {}
    row_count = 0
    for frame in frames_in or []:
        frame_index = int(_as_float(getattr(frame, "frame_name", getattr(frame, "frame_index", 0))))
        if frame_exact is not None and frame_index != frame_exact:
            continue
        if frame_min not in (None, "") and frame_index < int(_as_float(frame_min)):
            continue
        if frame_max not in (None, "") and frame_index > int(_as_float(frame_max)):
            continue
        pf = getattr(frame, "pass_fail_result", None)
        tn_objects = getattr(pf, "tn_objects", None) or []
        frame_ground_truth = getattr(frame, "frame_ground_truth", None)
        transforms = getattr(frame, "transforms", None) or getattr(frame_ground_truth, "transforms", None)
        for dynamic_object in tn_objects:
            if row_count >= max_rows:
                break
            box = _dynamic_object_box(dynamic_object, transforms=transforms, run_label=run_label)
            if box is None:
                continue
            out.setdefault(frame_index, []).append(box)
            row_count += 1
        if row_count >= max_rows:
            break
    return {
        "available": True,
        "source": "scene_result.pkl",
        "pickle_path": str(pickle_path),
        "frames": [{"frame": frame, "boxes": boxes} for frame, boxes in sorted(out.items())],
        "row_count": row_count,
        "frame_count": len(out),
        "truncated": row_count >= max_rows,
    }


def frames(payload: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_local_path(payload.get("path"))
    run_label = _as_text(payload.get("run")) or "A"
    cols = _columns(path)
    _require_columns(cols, ("frame_index", "source", "x", "y", "length", "width", "yaw"))
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    where, params = _where_from_filters(cols, filters)
    select_cols = [c for c in CORE_COLUMNS + OPTIONAL_COLUMNS if c in cols]
    output_select_cols = list(select_cols)
    select_cols_with_frame_int = select_cols + ["TRY_CAST(frame_index AS INTEGER) AS _frame_index_int"]
    order_cols = ["TRY_CAST(frame_index AS INTEGER)"]
    order_cols.extend(c for c in ("source", "status", "label") if c in select_cols)
    max_rows = min(max(int(payload.get("max_rows") or 120000), 100), 600000)
    dedupe = payload.get("dedupe", True) is not False
    shape_type_col = "shape_type" if "shape_type" in cols else ("type" if "type" in cols else None)
    polygon_keep_sql = (
        f" OR LOWER(COALESCE(CAST({shape_type_col} AS VARCHAR), '')) IN ('polygon', 'point')"
        if shape_type_col
        else ""
    )
    valid_geometry_sql = f"""
          AND (
            (
              TRY_CAST(length AS DOUBLE) > 0
              AND TRY_CAST(width AS DOUBLE) > 0
            )
            {polygon_keep_sql}
          )
    """
    source_sql = f"""
        SELECT {", ".join(select_cols_with_frame_int)}
        FROM parquet_scan(?)
        WHERE {" AND ".join(where)}
          AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
          {valid_geometry_sql}
    """
    if dedupe:
        dedupe_order = "TRY_CAST(confidence AS DOUBLE) DESC NULLS LAST" if "confidence" in select_cols else "TRY_CAST(x AS DOUBLE)"
        identity_expr = (
            "COALESCE(NULLIF(uuid, ''), CONCAT(CAST(ROUND(TRY_CAST(x AS DOUBLE), 2) AS VARCHAR), ':', CAST(ROUND(TRY_CAST(y AS DOUBLE), 2) AS VARCHAR), ':', CAST(ROUND(TRY_CAST(yaw AS DOUBLE), 2) AS VARCHAR)))"
            if "uuid" in select_cols
            else "CONCAT(CAST(ROUND(TRY_CAST(x AS DOUBLE), 2) AS VARCHAR), ':', CAST(ROUND(TRY_CAST(y AS DOUBLE), 2) AS VARCHAR), ':', CAST(ROUND(TRY_CAST(yaw AS DOUBLE), 2) AS VARCHAR))"
        )
        partition_cols = [
            "TRY_CAST(frame_index AS INTEGER)",
            "source",
            "status" if "status" in select_cols else "''",
            "label" if "label" in select_cols else "''",
            identity_expr,
        ]
        source_sql = f"""
            SELECT {", ".join(output_select_cols)}, _frame_index_int
            FROM (
                SELECT
                    {", ".join(select_cols_with_frame_int)},
                    ROW_NUMBER() OVER (
                        PARTITION BY {", ".join(partition_cols)}
                        ORDER BY {dedupe_order}
                    ) AS _bbox_rn
                FROM parquet_scan(?)
                WHERE {" AND ".join(where)}
                  AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
                  {valid_geometry_sql}
            )
            WHERE _bbox_rn = 1
        """
    con = duckdb.connect()
    try:
        df = con.execute(
            f"""
            {source_sql}
            ORDER BY _frame_index_int, {", ".join(c for c in order_cols if c != "TRY_CAST(frame_index AS INTEGER)")}
            LIMIT ?
            """,
            [str(path)] + params + [max_rows],
        ).df()
    finally:
        con.close()
    out_frames: list[dict[str, Any]] = []
    if not df.empty:
        for frame_index, group in df.groupby("_frame_index_int", sort=True):
            boxes: list[dict[str, Any]] = []
            for row in group.to_dict("records"):
                box = {
                        "x": _as_float(row.get("x")),
                        "y": _as_float(row.get("y")),
                        "z": _as_float(row.get("z")),
                        "length": _as_float(row.get("length")),
                        "width": _as_float(row.get("width")),
                        "height": _as_float(row.get("height"), 1.5),
                        "yaw": _as_float(row.get("yaw")),
                        "shape_type": _as_text(row.get("shape_type")) or _as_text(row.get("type")),
                        "source": _as_text(row.get("source")),
                        "status": _as_text(row.get("status")),
                        "label": _as_text(row.get("label")),
                        "uuid": _as_text(row.get("uuid")),
                        "confidence": None if row.get("confidence") is None else _as_float(row.get("confidence")),
                        "vx": None if row.get("vx") is None else _as_float(row.get("vx")),
                        "vy": None if row.get("vy") is None else _as_float(row.get("vy")),
                        "pair_uuid": _as_text(row.get("pair_uuid")),
                        "visibility": _as_text(row.get("visibility")),
                        "pointcloud_num": None if row.get("pointcloud_num") is None else _as_float(row.get("pointcloud_num")),
                        "x_error": None if row.get("x_error") is None else _as_float(row.get("x_error")),
                        "y_error": None if row.get("y_error") is None else _as_float(row.get("y_error")),
                        "z_error": None if row.get("z_error") is None else _as_float(row.get("z_error")),
                        "yaw_error": None if row.get("yaw_error") is None else _as_float(row.get("yaw_error")),
                        "center_distance": None if row.get("center_distance") is None else _as_float(row.get("center_distance")),
                        "plane_distance": None if row.get("plane_distance") is None else _as_float(row.get("plane_distance")),
                        "pair_dt_sec": None if row.get("pair_dt_sec") is None else _as_float(row.get("pair_dt_sec")),
                        "run": run_label,
                    }
                # §8: attach base_link footprint polygon when present (analyzer >=0.2.0).
                footprint_base_link = _row_footprint_base_link(row)
                if footprint_base_link:
                    box["footprint"] = footprint_base_link
                boxes.append(box)
            out_frames.append({"frame": int(float(frame_index)), "boxes": boxes})
    return {
        "frames": out_frames,
        "row_count": int(len(df)),
        "frame_count": len(out_frames),
        "truncated": int(len(df)) >= max_rows,
    }


def compare_frames(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("compare_frames requires runs: [{label,path}, ...]")
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    max_rows = min(max(int(payload.get("max_rows") or 120000), 100), 600000)
    dedupe = payload.get("dedupe", True) is not False
    merged: dict[int, list[dict[str, Any]]] = {}
    run_summaries: list[dict[str, Any]] = []
    total_rows = 0
    truncated = False
    for idx, run in enumerate(runs[:4]):
        if not isinstance(run, dict):
            continue
        label = _as_text(run.get("label")) or chr(ord("A") + idx)
        path = _as_text(run.get("path"))
        if not path:
            continue
        result = frames(
            {
                "path": path,
                "run": label,
                "filters": filters,
                "max_rows": max_rows,
                "dedupe": dedupe,
            }
        )
        total_rows += int(result.get("row_count") or 0)
        truncated = truncated or bool(result.get("truncated"))
        run_summaries.append(
            {
                "label": label,
                "path": path,
                "row_count": int(result.get("row_count") or 0),
                "frame_count": int(result.get("frame_count") or 0),
            }
        )
        for frame_obj in result.get("frames") or []:
            frame_index = int(frame_obj.get("frame") or 0)
            merged.setdefault(frame_index, []).extend(frame_obj.get("boxes") or [])
    out_frames = [{"frame": frame, "boxes": boxes} for frame, boxes in sorted(merged.items())]
    return {
        "frames": out_frames,
        "row_count": total_rows,
        "frame_count": len(out_frames),
        "truncated": truncated,
        "compare_runs": run_summaries,
    }


def _export_status(exc: Exception) -> int:
    """HTTP status for an export failure, so clients can tell apart the causes."""
    if isinstance(exc, export_api.ExportDisabledError):
        return 503
    if isinstance(exc, export_api.ExportAuthError):
        return 401
    return 400


class LocalBBoxHandler(BaseHTTPRequestHandler):
    routes = {
        "/api/parquets": list_parquets,
        "/api/describe": describe,
        "/api/values": values,
        "/api/scenarios": scenarios,
        "/api/dataset_summary": dataset_summary,
        "/api/dataset_stats": dataset_stats,
        "/api/scenario_curve": scenario_curve,
        "/api/scenario_devops_result": scenario_devops_result,
        "/api/scenario_devops_tn_objects": scenario_devops_tn_objects,
        "/api/scenario_devops_frame_results": scenario_devops_frame_results,
        "/api/frames": frames,
        "/api/compare_frames": compare_frames,
    }
    # Routes that need the request itself (for the bearer token), not just a payload.
    # Workflow routes authorize through export_api, so both sets share one policy.
    auth_routes = {**export_api.JSON_ROUTES, **workflow_api.JSON_ROUTES}
    # Routes that write their own response body instead of returning JSON.
    stream_routes = export_api.STREAM_ROUTES

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("LOCAL_BBOX_API_DEBUG") == "1":
            super().log_message(format, *args)

    def do_OPTIONS(self) -> None:
        _json_response(self, 200, {"ok": True})

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        asset_name = parsed.path.rsplit("/", 1)[-1]
        if _static_asset_response(self, asset_name, head_only=True):
            return
        if parsed.path in self.stream_routes:
            query = parse_qs(parsed.query)
            self._dispatch(parsed.path, {k: v[-1] for k, v in query.items()}, head_only=True)
            return
        if parsed.path in ("/", "/viewer", "/viewer/", "/explorer", "/explorer/", "/health", "/api/health"):
            self.send_response(200)
            is_html = parsed.path == "/" or "viewer" in parsed.path or "explorer" in parsed.path
            self.send_header("Content-Type", "text/html; charset=utf-8" if is_html else "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        asset_name = parsed.path.rsplit("/", 1)[-1]
        if _static_asset_response(self, asset_name):
            return
        if parsed.path in ("/health", "/api/health"):
            _json_response(self, 200, {"ok": True, "service": "local_bbox_api"})
            return
        if parsed.path in ("/", "/viewer", "/viewer/"):
            _html_response(self, 200, _viewer_html(""))
            return
        if parsed.path in ("/explorer", "/explorer/"):
            _html_response(self, 200, _explorer_html(""))
            return
        query = parse_qs(parsed.query)
        payload = {k: v[-1] for k, v in query.items()}
        self._dispatch(parsed.path, payload)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        self._dispatch(parsed.path, _read_json(self))

    def _dispatch(self, path: str, payload: dict[str, Any], *, head_only: bool = False) -> None:
        stream_route = self.stream_routes.get(path)
        if stream_route is not None:
            try:
                stream_route(self, payload, head_only=head_only)
            except Exception as exc:
                # Once the stream's headers are on the wire there is no valid way left
                # to report an error, so only answer if nothing was sent yet.
                if not getattr(self, "_export_stream_started", False):
                    _json_response(self, _export_status(exc), {"error": str(exc)})
            return
        auth_route = self.auth_routes.get(path)
        if auth_route is not None:
            try:
                _json_response(self, 200, auth_route(self, payload))
            except Exception as exc:
                _json_response(self, _export_status(exc), {"error": str(exc)})
            return
        route = self.routes.get(path)
        if route is None:
            _json_response(self, 404, {"error": f"Unknown route: {path}"})
            return
        try:
            _json_response(self, 200, route(payload))
        except Exception as exc:
            _json_response(self, 400, {"error": str(exc)})


def run_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, int(port)), LocalBBoxHandler)
    server.serve_forever()


def ensure_background_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> str:
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is None:
            _SERVER = ThreadingHTTPServer((host, int(port)), LocalBBoxHandler)
            thread = threading.Thread(target=_SERVER.serve_forever, name="local-bbox-api", daemon=True)
            thread.start()
    return f"http://{host}:{int(port)}"


if __name__ == "__main__":
    run_server(os.environ.get("LOCAL_BBOX_API_HOST", "0.0.0.0"), DEFAULT_PORT)
