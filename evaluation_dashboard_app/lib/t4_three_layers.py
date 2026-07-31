"""T4 `/viewer/three` embed: GT / pred / matched 3D box layers via postMessage."""

from __future__ import annotations

import html
import math
import base64
import struct
import ast
from urllib.parse import urlencode
from typing import TYPE_CHECKING

import pandas as pd
import streamlit.components.v1 as components

from lib.ui.theme import token

if TYPE_CHECKING:
    import pandas as pd


_OPTIONAL_NUMERIC_FIELDS = (
    "vx",
    "vy",
    "confidence",
    "pointcloud_num",
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
    "dx_min",
    "dy_min",
    "unix_time",
    "frame_index",
)

_OPTIONAL_TEXT_FIELDS = (
    "frame_id",
    "shape_type",
    "visibility",
    "pair_uuid",
    "topic_name",
    "t4dataset_id",
    "suite_name",
    "t4dataset_name",
    "scenario_name",
    "run",
    "source",
)
_BINARY_MAGIC = b"T4BBOX1\x00"
_BINARY_CORE_FLOAT_FIELDS = ("x", "y", "z", "width", "length", "height", "yaw")
_BINARY_TEXT_FIELDS = ("uuid", "label", "status") + _OPTIONAL_TEXT_FIELDS
_BINARY_BOX_FLOAT_COUNT = len(_BINARY_CORE_FLOAT_FIELDS) + len(_OPTIONAL_NUMERIC_FIELDS) + 24
_BINARY_BOX_TEXT_COUNT = len(_BINARY_TEXT_FIELDS)
_BINARY_BOX_STRUCT = struct.Struct("<" + ("f" * _BINARY_BOX_FLOAT_COUNT) + ("I" * _BINARY_BOX_TEXT_COUNT) + "B")
_BINARY_FRAME_STRUCT = struct.Struct("<iIIIIII")
_BINARY_PAIR_STRUCT = struct.Struct("<III")
_BINARY_HEADER_STRUCT = struct.Struct("<8sIIIII")
_BINARY_FOOTPRINT_COUNT_STRUCT = struct.Struct("<I")
_BINARY_FOOTPRINT_POINT_STRUCT = struct.Struct("<fff")

_VEHICLE_LABELS = {"car", "truck", "bus", "trailer"}
_EXTERNAL_EVAL_TO_T4_YAW_OFFSET = 0.0
EXTERNAL_BBOX_ALIGNMENT_VERSION = "eval-yaw0-explicit-corners-v3"
PLACEHOLDER_T4_DATASET_IDS = {
    "00000000-0000-0000-0000-000000000000",
    "00000000-0000-0000-0000-000000000001",
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(value != value)
    except TypeError:
        return True


def _as_float(value: object, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_text(value: object, default: str = "") -> str:
    if _is_missing(value):
        return default
    return str(value)


def _clean_t4_dataset_value(value: object) -> str:
    text = _as_text(value).strip()
    if text.lower() in {"", "none", "nan", "<na>"}:
        return ""
    if text in PLACEHOLDER_T4_DATASET_IDS:
        return ""
    return text


def _as_footprint_vertices(value: object) -> list[list[float]] | None:
    """Return base_link footprint vertices as [[x, y, z], ...], or None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    raw = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            raw = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        return None
    vertices: list[list[float]] = []
    for pt in raw:
        if hasattr(pt, "tolist"):
            pt = pt.tolist()
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return None
        x = _as_float(pt[0], float("nan"))
        y = _as_float(pt[1], float("nan"))
        z = _as_float(pt[2], 0.0) if len(pt) >= 3 else 0.0
        if not all(math.isfinite(v) for v in (x, y, z)):
            return None
        vertices.append([x, y, z])
    if len(vertices) < 3:
        return None
    return vertices


def resolve_t4_dataset_id(dff: "pd.DataFrame") -> str:
    """Parquet **t4dataset_id** or **t4dataset_name** for the current frame (empty if missing)."""
    if dff is None or dff.empty:
        return ""
    if "t4dataset_id" in dff.columns and dff["t4dataset_id"].notna().any():
        for value in dff["t4dataset_id"].dropna().tolist():
            clean = _clean_t4_dataset_value(value)
            if clean:
                return clean
    if "t4dataset_name" in dff.columns and dff["t4dataset_name"].notna().any():
        for value in dff["t4dataset_name"].dropna().tolist():
            clean = _clean_t4_dataset_value(value)
            if clean:
                return clean
    return ""


def resolve_t4_scenario(dff: "pd.DataFrame", scenario_from_sidebar: str | None) -> str:
    if scenario_from_sidebar is not None and str(scenario_from_sidebar).strip() != "":
        return str(scenario_from_sidebar)
    if dff is not None and not dff.empty and "scenario_name" in dff.columns and dff["scenario_name"].notna().any():
        return str(dff["scenario_name"].dropna().iloc[0])
    return ""


def infer_legacy_width_length_swapped(df: "pd.DataFrame") -> bool:
    """Return True when old parquet exports appear to have swapped length/width.

    Older app-generated parquet files were produced before the catalog analyzer
    width/length fix. For vehicle rows they store the raw evaluator tuple as
    ``length=width`` and ``width=length``. Normalize those rows before posting to
    the T4 viewer so both old app data and converted analyzer release data use
    the same body-x ``length`` / body-y ``width`` convention.
    """
    if df is None or df.empty or not {"length", "width"}.issubset(df.columns):
        return False

    sample = df
    if "label" in sample.columns:
        labels = sample["label"].astype(str).str.lower()
        vehicle_sample = sample[labels.isin(_VEHICLE_LABELS)]
        if not vehicle_sample.empty:
            sample = vehicle_sample
    if "source" in sample.columns:
        gt_sample = sample[sample["source"].astype(str) == "GT"]
        if not gt_sample.empty:
            sample = gt_sample

    dims = sample[["length", "width"]].apply(lambda s: s.astype(float), axis=0)
    dims = dims[(dims["length"] > 0) & (dims["width"] > 0)]
    if dims.empty:
        return False

    length_forward_ratio = float((dims["length"] >= dims["width"]).mean())
    return length_forward_ratio < 0.2


def infer_external_bbox_alignment_query_params(df: "pd.DataFrame") -> str:
    """Return `/viewer/three` query params after app-side bbox dimension normalization.

    T4 viewer annotations are rendered from dataset-provided corners. For the
    evaluator parquet used by this dashboard, the stored yaw already matches the
    annotation corner length axis, so only legacy swapped length/width columns
    are normalized in the payload. The query param is still explicit because the
    T4 viewer/server default is +pi/2 for other external payload conventions.
    """
    return urlencode(
        {
            "external_bbox_yaw_offset": f"{_EXTERNAL_EVAL_TO_T4_YAW_OFFSET:.12g}",
            "external_bbox_swap_lw": "false",
            "external_bbox_alignment_version": EXTERNAL_BBOX_ALIGNMENT_VERSION,
        }
    )


def _box_corners_from_pose(box: dict) -> list[float] | None:
    try:
        cx = _as_float(box.get("x"), 0.0)
        cy = _as_float(box.get("y"), 0.0)
        cz = _as_float(box.get("z"), 0.0)
        length = _as_float(box.get("length"), 0.0)
        width = _as_float(box.get("width"), 0.0)
        height = _as_float(box.get("height"), 0.0)
        yaw = _as_float(box.get("yaw"), 0.0)
    except (TypeError, ValueError):
        return None
    values = (cx, cy, cz, length, width, height, yaw)
    if not all(math.isfinite(v) for v in values) or length <= 0 or width <= 0 or height <= 0:
        return None

    half_l = length / 2.0
    half_w = width / 2.0
    half_h = height / 2.0
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    body = (
        (half_l, half_w, half_h),
        (half_l, -half_w, half_h),
        (half_l, -half_w, -half_h),
        (half_l, half_w, -half_h),
        (-half_l, half_w, half_h),
        (-half_l, -half_w, half_h),
        (-half_l, -half_w, -half_h),
        (-half_l, half_w, -half_h),
    )
    corners: list[float] = []
    for bx, by, bz in body:
        corners.extend(
            [
                cx + bx * cos_yaw - by * sin_yaw,
                cy + bx * sin_yaw + by * cos_yaw,
                cz + bz,
            ]
        )
    return corners


def _single_frame_layer_dict(df_frame: "pd.DataFrame", swap_length_width: bool = False) -> dict:
    """Per-frame gt / pred / matched_pairs (no ``type`` field); used by single- and all-frame payloads."""
    if df_frame is None or df_frame.empty:
        return {"gt": [], "pred": [], "matched_pairs": []}

    def _dedupe_eval_rows(rows: "pd.DataFrame") -> "pd.DataFrame":
        if rows.empty or "uuid" not in rows.columns:
            return rows
        work = rows.copy()
        if "shape_type" in work.columns:
            shape_text = work["shape_type"].map(_as_text).str.lower()
            length_num = pd.to_numeric(work["length"], errors="coerce") if "length" in work.columns else pd.Series(1.0, index=work.index)
            width_num = pd.to_numeric(work["width"], errors="coerce") if "width" in work.columns else pd.Series(1.0, index=work.index)
            height_num = pd.to_numeric(work["height"], errors="coerce") if "height" in work.columns else pd.Series(1.0, index=work.index)
            marker_mask = (
                (shape_text == "invalid_polygon_marker")
                | ((shape_text == "polygon") & ((length_num <= 0) | (width_num <= 0)) & (height_num > 0))
            )
            marker_rows = work[marker_mask].copy()
            work = work[~marker_mask].copy()
            if work.empty:
                return marker_rows
        else:
            marker_rows = pd.DataFrame()
        uuid_text = work["uuid"].map(_as_text)
        if "pair_uuid" in work.columns:
            pair_text = work["pair_uuid"].map(_as_text)
        else:
            pair_text = pd.Series([""] * len(work), index=work.index)
        if "status" in work.columns:
            status_text = work["status"].map(_as_text)
        else:
            status_text = pd.Series([""] * len(work), index=work.index)
        identity = uuid_text.where(uuid_text != "", pair_text)
        identity = identity.where(~((status_text == "TP") & (pair_text != "")), pair_text)
        work["_dedupe_identity"] = identity
        work["_dedupe_has_identity"] = identity != ""

        sort_cols = ["_dedupe_has_identity", "_dedupe_identity"]
        ascending = [False, True]
        if "status" in work.columns:
            sort_cols.append("status")
            ascending.append(True)
        if "label" in work.columns:
            sort_cols.append("label")
            ascending.append(True)
        if "run" in work.columns:
            sort_cols.append("run")
            ascending.append(True)
        if "pair_dt_sec" in work.columns:
            work["_dedupe_abs_pair_dt"] = pd.to_numeric(work["pair_dt_sec"], errors="coerce").abs()
            sort_cols.append("_dedupe_abs_pair_dt")
            ascending.append(True)
        if "center_distance" in work.columns:
            work["_dedupe_center_distance"] = pd.to_numeric(work["center_distance"], errors="coerce")
            sort_cols.append("_dedupe_center_distance")
            ascending.append(True)
        if "confidence" in work.columns:
            work["_dedupe_confidence"] = pd.to_numeric(work["confidence"], errors="coerce")
            sort_cols.append("_dedupe_confidence")
            ascending.append(False)
        if "unix_time" in work.columns:
            work["_dedupe_unix_time"] = pd.to_numeric(work["unix_time"], errors="coerce")
            sort_cols.append("_dedupe_unix_time")
            ascending.append(True)

        with_identity = work[work["_dedupe_has_identity"]].copy()
        without_identity = work[~work["_dedupe_has_identity"]].copy()
        group_cols = ["_dedupe_identity"]
        if "status" in with_identity.columns:
            group_cols.append("status")
        if "label" in with_identity.columns:
            group_cols.append("label")
        if "run" in with_identity.columns:
            group_cols.append("run")
        if not with_identity.empty:
            with_identity = (
                with_identity.sort_values(sort_cols, ascending=ascending, na_position="last")
                .groupby(group_cols, sort=False, dropna=False)
                .head(1)
            )
        deduped = pd.concat([with_identity, without_identity, marker_rows], axis=0).sort_index()
        return deduped.drop(columns=[c for c in deduped.columns if c.startswith("_dedupe_")])

    def _row_to_box(row: "pd.Series") -> dict | None:
        length = _as_float(row.get("length"), 0.0)
        width = _as_float(row.get("width"), 0.0)
        height = _as_float(row.get("height"), 1.5)
        shape_type = _as_text(row.get("shape_type")).lower()
        is_invalid_polygon_marker = shape_type == "polygon" and (length <= 0 or width <= 0) and height > 0
        # Skip boxes with invalid dimensions, except polygon detections that can still be shown as point markers.
        if (length <= 0 or width <= 0) and not is_invalid_polygon_marker:
            return None
        if swap_length_width:
            length, width = width, length
        box = {
            "x": _as_float(row.get("x"), 0.0),
            "y": _as_float(row.get("y"), 0.0),
            "z": _as_float(row.get("z"), 0.0),
            "width": width,
            "length": length,
            "height": height,
            "yaw": _as_float(row.get("yaw"), 0.0),
            "label": _as_text(row.get("label")),
            "uuid": _as_text(row.get("uuid")),
            "status": _as_text(row.get("status")),
        }
        if is_invalid_polygon_marker:
            box["shape_type"] = "invalid_polygon_marker"
            box["width"] = 0.0
            box["length"] = 0.0
        # Skip boxes with invalid height
        if box["height"] <= 0:
            return None
        corners = None if is_invalid_polygon_marker else _box_corners_from_pose(box)
        if corners is not None:
            box["corners"] = corners
        for field in _OPTIONAL_NUMERIC_FIELDS:
            if field in row.index:
                value = row.get(field)
                if not _is_missing(value):
                    box[field] = _as_float(value)
        for field in _OPTIONAL_TEXT_FIELDS:
            if field in row.index:
                value = row.get(field)
                if not _is_missing(value):
                    if is_invalid_polygon_marker and field == "shape_type":
                        continue
                    box[field] = _as_text(value)
        if "footprint" in row.index:
            footprint = _as_footprint_vertices(row.get("footprint"))
            if footprint is not None:
                box["footprint"] = footprint
        return box

    gt_df = _dedupe_eval_rows(df_frame[df_frame["source"] == "GT"].copy())
    pred_df = _dedupe_eval_rows(df_frame[df_frame["source"] == "EST"].copy())
    gt_boxes = [b for _, r in gt_df.iterrows() if (b := _row_to_box(r)) is not None]
    for box in gt_boxes:
        box["force_wireframe"] = True
    pred_boxes = [b for _, r in pred_df.iterrows() if (b := _row_to_box(r)) is not None]

    gt_tp_idx: dict[str, int] = {}
    for i, b in enumerate(gt_boxes):
        if b["status"] != "TP":
            continue
        for match_key in (str(b.get("pair_uuid") or ""), str(b.get("uuid") or "")):
            if match_key:
                gt_tp_idx.setdefault(match_key, i)
    pred_tp_idx: dict[str, int] = {}
    for i, b in enumerate(pred_boxes):
        if b["status"] != "TP":
            continue
        for match_key in (str(b.get("pair_uuid") or ""), str(b.get("uuid") or "")):
            if match_key:
                pred_tp_idx.setdefault(match_key, i)
    matched_pairs = []
    seen_pair_indexes: set[tuple[int, int]] = set()
    for match_key, gi in gt_tp_idx.items():
        pi = pred_tp_idx.get(match_key)
        if pi is not None and (gi, pi) not in seen_pair_indexes:
            seen_pair_indexes.add((gi, pi))
            matched_pairs.append({"gt_idx": int(gi), "pred_idx": int(pi), "pair_uuid": match_key})

    return {
        "gt": gt_boxes,
        "pred": pred_boxes,
        "matched_pairs": matched_pairs,
    }


def build_three_layer_payload(df_frame: "pd.DataFrame") -> dict:
    """Build GT/Pred/Matched overlay payload for `/viewer/three` iframe (single frame)."""
    if df_frame is None or df_frame.empty:
        return {"type": "bbox_layers_clear"}
    inner = _single_frame_layer_dict(df_frame, swap_length_width=infer_legacy_width_length_swapped(df_frame))
    return {
        "type": "bbox_layers",
        "gt": inner["gt"],
        "pred": inner["pred"],
        "matched_pairs": inner["matched_pairs"],
    }


def build_three_layer_payload_all_frames(df: "pd.DataFrame") -> dict:
    """Build payload with eval layers for every ``frame_index`` in *df* (viewer picks by internal frame)."""
    if df is None or df.empty:
        return {"type": "bbox_layers_by_frame", "frames": {}}
    if "frame_index" not in df.columns:
        return {"type": "bbox_layers_by_frame", "frames": {}}
    compare_runs: list[str] = []
    if "run" in df.columns:
        for value in df["run"].dropna().astype(str).tolist():
            if value and value not in compare_runs:
                compare_runs.append(value)
    swap_length_width = infer_legacy_width_length_swapped(df)
    frames: dict[str, dict] = {}
    for fi, group in df.groupby("frame_index", sort=True):
        try:
            key = str(int(fi))
        except (TypeError, ValueError):
            continue
        frames[key] = _single_frame_layer_dict(group, swap_length_width=swap_length_width)
    payload = {"type": "bbox_layers_by_frame", "frames": frames}
    if len(compare_runs) >= 2:
        payload["compare_runs"] = compare_runs
    return payload


def _pack_three_layer_payload_binary(layer_payload: dict) -> tuple[bytes, dict]:
    """Pack bbox layer payload as a compact typed-array friendly binary blob."""
    if not isinstance(layer_payload, dict):
        layer_payload = {"type": "bbox_layers_clear"}

    frames_obj: dict[str, dict]
    if layer_payload.get("type") == "bbox_layers_by_frame" and isinstance(layer_payload.get("frames"), dict):
        frames_obj = layer_payload.get("frames") or {}
    elif layer_payload.get("type") == "bbox_layers":
        frames_obj = {"0": layer_payload}
    else:
        frames_obj = {}

    string_ids: dict[str, int] = {"": 0}
    strings: list[str] = [""]

    def sid(value: object) -> int:
        text = "" if value is None else str(value)
        if not text:
            return 0
        existing = string_ids.get(text)
        if existing is not None:
            return existing
        idx = len(strings)
        string_ids[text] = idx
        strings.append(text)
        return idx

    compare_runs = [
        str(v)
        for v in (layer_payload.get("compare_runs") or [])
        if str(v).strip()
    ]

    frame_rows: list[tuple[int, int, int, int, int, int, int]] = []
    box_rows: list[tuple[list[float], list[int], int, list[list[float]] | None]] = []
    pair_rows: list[tuple[int, int, int]] = []
    gt_total = 0
    pred_total = 0
    max_boxes_per_frame = 0

    def add_box(box: dict) -> None:
        floats: list[float] = []
        for field in _BINARY_CORE_FLOAT_FIELDS:
            floats.append(_as_float(box.get(field), 0.0))
        for field in _OPTIONAL_NUMERIC_FIELDS:
            value = box.get(field)
            floats.append(float("nan") if _is_missing(value) else _as_float(value))
        corners = box.get("corners")
        if isinstance(corners, list) and len(corners) >= 24:
            floats.extend(_as_float(v, 0.0) for v in corners[:24])
        else:
            floats.extend([float("nan")] * 24)
        text_ids = [sid(box.get(field)) for field in _BINARY_TEXT_FIELDS]
        force_wireframe = 1 if box.get("force_wireframe") is True else 0
        box_rows.append((floats, text_ids, force_wireframe, _as_footprint_vertices(box.get("footprint"))))

    def frame_sort_key(item: tuple[str, dict]) -> int:
        try:
            return int(item[0])
        except (TypeError, ValueError):
            return 0

    for frame_key, frame_payload in sorted(frames_obj.items(), key=frame_sort_key):
        if not isinstance(frame_payload, dict):
            continue
        try:
            frame_index = int(frame_key)
        except (TypeError, ValueError):
            continue
        gt_boxes = frame_payload.get("gt") if isinstance(frame_payload.get("gt"), list) else []
        pred_boxes = frame_payload.get("pred") if isinstance(frame_payload.get("pred"), list) else []
        pairs = frame_payload.get("matched_pairs") if isinstance(frame_payload.get("matched_pairs"), list) else []
        gt_start = len(box_rows)
        for box in gt_boxes:
            if isinstance(box, dict):
                add_box(box)
        gt_count = len(box_rows) - gt_start
        gt_total += gt_count
        pred_start = len(box_rows)
        for box in pred_boxes:
            if isinstance(box, dict):
                add_box(box)
        pred_count = len(box_rows) - pred_start
        pred_total += pred_count
        max_boxes_per_frame = max(max_boxes_per_frame, gt_count + pred_count)
        pair_start = len(pair_rows)
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            try:
                pair_rows.append((int(pair.get("gt_idx", 0)), int(pair.get("pred_idx", 0)), sid(pair.get("pair_uuid"))))
            except (TypeError, ValueError):
                continue
        pair_count = len(pair_rows) - pair_start
        frame_rows.append((frame_index, gt_start, gt_count, pred_start, pred_count, pair_start, pair_count))

    compare_run_ids = [sid(v) for v in compare_runs]
    parts = [
        _BINARY_HEADER_STRUCT.pack(
            _BINARY_MAGIC,
            len(frame_rows),
            len(strings),
            len(box_rows),
            len(pair_rows),
            len(compare_run_ids),
        )
    ]
    for text in strings:
        raw = text.encode("utf-8")
        parts.append(struct.pack("<I", len(raw)))
        parts.append(raw)
    for run_id in compare_run_ids:
        parts.append(struct.pack("<I", run_id))
    for row in frame_rows:
        parts.append(_BINARY_FRAME_STRUCT.pack(*row))
    for floats, text_ids, force_wireframe, _footprint in box_rows:
        parts.append(_BINARY_BOX_STRUCT.pack(*floats, *text_ids, force_wireframe))
    for row in pair_rows:
        parts.append(_BINARY_PAIR_STRUCT.pack(*row))
    footprint_point_count = 0
    footprint_box_count = 0
    for _floats, _text_ids, _force_wireframe, footprint in box_rows:
        vertices = footprint or []
        if vertices:
            footprint_box_count += 1
        footprint_point_count += len(vertices)
        parts.append(_BINARY_FOOTPRINT_COUNT_STRUCT.pack(len(vertices)))
        for x, y, z in vertices:
            parts.append(_BINARY_FOOTPRINT_POINT_STRUCT.pack(x, y, z))
    blob = b"".join(parts)
    frame_indexes = [row[0] for row in frame_rows]
    stats = {
        "format": "T4BBOX1",
        "transport": "binary_typed_array_post_message",
        "binary_bytes": len(blob),
        "frame_count": len(frame_rows),
        "first_frame": min(frame_indexes) if frame_indexes else None,
        "last_frame": max(frame_indexes) if frame_indexes else None,
        "gt_box_count": gt_total,
        "pred_box_count": pred_total,
        "box_count": len(box_rows),
        "matched_pair_count": len(pair_rows),
        "max_boxes_per_frame": max_boxes_per_frame,
        "string_table_count": len(strings),
        "compare_run_count": len(compare_run_ids),
        "box_row_bytes": _BINARY_BOX_STRUCT.size,
        "frame_row_bytes": _BINARY_FRAME_STRUCT.size,
        "pair_row_bytes": _BINARY_PAIR_STRUCT.size,
        "footprint_box_count": footprint_box_count,
        "footprint_point_count": footprint_point_count,
        "float_fields_per_box": _BINARY_BOX_FLOAT_COUNT,
        "text_id_fields_per_box": _BINARY_BOX_TEXT_COUNT,
        "numeric_fields": list(_BINARY_CORE_FLOAT_FIELDS + _OPTIONAL_NUMERIC_FIELDS),
        "text_fields": list(_BINARY_TEXT_FIELDS),
    }
    return blob, stats


def render_t4_three_js_embed(viewer_three_url: str, layer_payload: dict, height: int = 700) -> dict:
    """Iframe to T4 three viewer + postMessage with bbox layer payload (GT, pred, matched pairs)."""
    _payload_binary, _payload_stats = _pack_three_layer_payload_binary(layer_payload)
    _payload_binary_b64 = base64.b64encode(_payload_binary).decode("ascii")
    _payload_stats = dict(_payload_stats)
    _payload_stats["base64_chars"] = len(_payload_binary_b64)
    _iframe_src = html.escape(viewer_three_url, quote=True)
    # components.html renders in its own iframe, so page-level --t4-* vars are not
    # visible here: resolve the placeholder surface from the palette in Python.
    _placeholder_bg = html.escape(token("surface_3"), quote=True)
    components.html(
        (
            f'<iframe id="t4-three-viewer" src="{_iframe_src}" '
            f'width="100%" height="{height}" style="border:none;border-radius:8px;background:{_placeholder_bg}" '
            f'allowfullscreen allow="fullscreen *" '
            f'loading="lazy" title="T4 three viewer" referrerpolicy="no-referrer-when-downgrade"></iframe>'
            f'<script id="t4-three-layer-payload" type="application/octet-stream">{_payload_binary_b64}</script>'
            "<script>"
            "(()=>{"
            "const iframe=document.getElementById('t4-three-viewer');"
            "const payloadEl=document.getElementById('t4-three-layer-payload');"
            "const b64ToBuffer=(b64)=>{"
            "const clean=(b64||'').replace(/\\s+/g,'');"
            "const bin=atob(clean);"
            "const bytes=new Uint8Array(bin.length);"
            "for(let i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);"
            "return bytes.buffer;"
            "};"
            "let postCount=0;"
            "const post=(reason)=>{"
            "if(!iframe||!iframe.contentWindow)return;"
            "let targetOrigin='*';"
            "try{ targetOrigin = new URL(iframe.src, window.location.href).origin || '*'; }catch(_){ targetOrigin='*'; }"
            "const buffer=b64ToBuffer(payloadEl?payloadEl.textContent:'');"
            "postCount+=1;"
            "iframe.contentWindow.postMessage({type:'bbox_layers_binary_v1',buffer},targetOrigin,[buffer]);"
            "console.info('[bbox-debug] binary payload sent', {reason,postCount,targetOrigin,bytes:buffer.byteLength});"
            "};"
            "iframe.addEventListener('load',()=>post('iframe-load'),{once:true});"
            "})();"
            "</script>"
        ),
        height=height + 24,
        scrolling=True,
    )
    return _payload_stats
