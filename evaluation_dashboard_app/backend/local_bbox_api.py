"""Small DuckDB-backed HTTP API for the local 3D bbox viewer.

The API is intentionally dependency-light so it can run next to Streamlit in
both local development and the Docker deployment.
"""

from __future__ import annotations

import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import duckdb


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
)
_SERVER_LOCK = threading.Lock()
_SERVER: ThreadingHTTPServer | None = None


def _data_root() -> Path:
    raw = os.environ.get("EVAL_DASHBOARD_DATA_ROOT", "data")
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _allowed_roots() -> list[Path]:
    roots = [_data_root(), (Path.cwd() / "data").resolve()]
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


def _resolve_local_path(value: str | None, *, allow_file: bool = True) -> Path:
    text = str(value or "").strip()
    if not text:
        return _data_root()
    raw = Path(text).expanduser()
    candidates = [raw.resolve()] if raw.is_absolute() else [(Path.cwd() / raw).resolve(), (_data_root() / raw).resolve()]
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


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
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


def _viewer_html(api_base: str = "") -> str:
    candidates = [
        Path.cwd() / "static" / "local_bbox_viewer.html",
        Path("/app/static/local_bbox_viewer.html"),
    ]
    for path in candidates:
        if path.exists():
            source = path.read_text(encoding="utf-8")
            return source.replace("__API_BASE__", api_base.rstrip("/"))
    raise FileNotFoundError("static/local_bbox_viewer.html not found")


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
    con = duckdb.connect()
    try:
        return con.execute("DESCRIBE SELECT * FROM parquet_scan(?)", [str(parquet_path)]).df()["column_name"].tolist()
    finally:
        con.close()


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


def list_parquets(payload: dict[str, Any]) -> dict[str, Any]:
    root = _resolve_local_path(payload.get("root") or "", allow_file=True)
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
    return {"items": items, "root": str(root), "skipped": skipped}


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
    source_sql = f"""
        SELECT {", ".join(select_cols_with_frame_int)}
        FROM parquet_scan(?)
        WHERE {" AND ".join(where)}
          AND TRY_CAST(frame_index AS INTEGER) IS NOT NULL
          AND TRY_CAST(length AS DOUBLE) > 0
          AND TRY_CAST(width AS DOUBLE) > 0
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
                  AND TRY_CAST(length AS DOUBLE) > 0
                  AND TRY_CAST(width AS DOUBLE) > 0
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
                boxes.append(
                    {
                        "x": _as_float(row.get("x")),
                        "y": _as_float(row.get("y")),
                        "z": _as_float(row.get("z")),
                        "length": _as_float(row.get("length")),
                        "width": _as_float(row.get("width")),
                        "height": _as_float(row.get("height"), 1.5),
                        "yaw": _as_float(row.get("yaw")),
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
                )
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


class LocalBBoxHandler(BaseHTTPRequestHandler):
    routes = {
        "/api/parquets": list_parquets,
        "/api/describe": describe,
        "/api/values": values,
        "/api/scenarios": scenarios,
        "/api/frames": frames,
        "/api/compare_frames": compare_frames,
    }

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("LOCAL_BBOX_API_DEBUG") == "1":
            super().log_message(format, *args)

    def do_OPTIONS(self) -> None:
        _json_response(self, 200, {"ok": True})

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/viewer", "/viewer/", "/health", "/api/health"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8" if "viewer" in parsed.path or parsed.path == "/" else "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/health", "/api/health"):
            _json_response(self, 200, {"ok": True, "service": "local_bbox_api"})
            return
        if parsed.path in ("/", "/viewer", "/viewer/"):
            _html_response(self, 200, _viewer_html(""))
            return
        query = parse_qs(parsed.query)
        payload = {k: v[-1] for k, v in query.items()}
        self._dispatch(parsed.path, payload)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        self._dispatch(parsed.path, _read_json(self))

    def _dispatch(self, path: str, payload: dict[str, Any]) -> None:
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
