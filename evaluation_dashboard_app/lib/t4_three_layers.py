"""T4 `/viewer/three` embed: GT / pred / matched 3D box layers via postMessage."""

from __future__ import annotations

import html
import json
import math
from urllib.parse import urlencode
from typing import TYPE_CHECKING

import streamlit.components.v1 as components

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

_VEHICLE_LABELS = {"car", "truck", "bus", "trailer"}
_LEGACY_EXTERNAL_BBOX_YAW_OFFSET = math.pi / 2


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(value != value)


def resolve_t4_dataset_id(dff: "pd.DataFrame") -> str:
    """Parquet **t4dataset_id** or **t4dataset_name** for the current frame (empty if missing)."""
    if dff is None or dff.empty:
        return ""
    if "t4dataset_id" in dff.columns and dff["t4dataset_id"].notna().any():
        return str(dff["t4dataset_id"].dropna().astype(str).iloc[0])
    if "t4dataset_name" in dff.columns and dff["t4dataset_name"].notna().any():
        return str(dff["t4dataset_name"].dropna().iloc[0])
    return ""


def resolve_t4_scenario(dff: "pd.DataFrame", scenario_from_sidebar: str | None) -> str:
    if scenario_from_sidebar is not None and str(scenario_from_sidebar).strip() != "":
        return str(scenario_from_sidebar)
    if dff is not None and not dff.empty and "scenario_name" in dff.columns and dff["scenario_name"].notna().any():
        return str(dff["scenario_name"].dropna().iloc[0])
    return ""


def infer_external_bbox_alignment_query_params(df: "pd.DataFrame") -> str:
    """Return `/viewer/three` query params for eval bbox dimension/yaw convention.

    Older eval parquet exports often store vehicle dimensions as width-forward
    (`length < width`) and rely on the T4 viewer's legacy `+pi/2` external bbox
    yaw offset. Newer app/analyzer exports store body-x as `length` and body-y as
    `width`; those must pass `external_bbox_yaw_offset=0` or the viewer rotates
    them by 90 degrees.
    """
    if df is None or df.empty or not {"length", "width"}.issubset(df.columns):
        yaw_offset = _LEGACY_EXTERNAL_BBOX_YAW_OFFSET
    else:
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
            yaw_offset = _LEGACY_EXTERNAL_BBOX_YAW_OFFSET
        else:
            length_forward_ratio = float((dims["length"] >= dims["width"]).mean())
            yaw_offset = 0.0 if length_forward_ratio >= 0.8 else _LEGACY_EXTERNAL_BBOX_YAW_OFFSET

    return urlencode(
        {
            "external_bbox_yaw_offset": f"{yaw_offset:.12g}",
            "external_bbox_swap_lw": "false",
        }
    )


def _single_frame_layer_dict(df_frame: "pd.DataFrame") -> dict:
    """Per-frame gt / pred / matched_pairs (no ``type`` field); used by single- and all-frame payloads."""
    if df_frame is None or df_frame.empty:
        return {"gt": [], "pred": [], "matched_pairs": []}

    def _row_to_box(row: "pd.Series") -> dict:
        box = {
            "x": float(row.get("x", 0.0) or 0.0),
            "y": float(row.get("y", 0.0) or 0.0),
            "z": float(row.get("z", 0.0) or 0.0),
            "width": float(row.get("width", 0.0) or 0.0),
            "length": float(row.get("length", 0.0) or 0.0),
            "height": float(row.get("height", 1.5) or 1.5),
            "yaw": float(row.get("yaw", 0.0) or 0.0),
            "label": str(row.get("label", "") or ""),
            "uuid": str(row.get("uuid", "") or ""),
            "status": str(row.get("status", "") or ""),
        }
        for field in _OPTIONAL_NUMERIC_FIELDS:
            if field in row.index:
                value = row.get(field)
                if not _is_missing(value):
                    box[field] = float(value)
        for field in _OPTIONAL_TEXT_FIELDS:
            if field in row.index:
                value = row.get(field)
                if not _is_missing(value):
                    box[field] = str(value)
        return box

    gt_df = df_frame[df_frame["source"] == "GT"].copy()
    pred_df = df_frame[df_frame["source"] == "EST"].copy()
    gt_boxes = [_row_to_box(r) for _, r in gt_df.iterrows()]
    pred_boxes = [_row_to_box(r) for _, r in pred_df.iterrows()]

    gt_tp_idx: dict[str, int] = {}
    for i, b in enumerate(gt_boxes):
        match_key = str(b.get("pair_uuid") or b.get("uuid") or "")
        if b["status"] == "TP" and match_key:
            gt_tp_idx.setdefault(match_key, i)
    pred_tp_idx: dict[str, int] = {}
    for i, b in enumerate(pred_boxes):
        match_key = str(b.get("pair_uuid") or b.get("uuid") or "")
        if b["status"] == "TP" and match_key:
            pred_tp_idx.setdefault(match_key, i)
    matched_pairs = []
    for match_key, gi in gt_tp_idx.items():
        pi = pred_tp_idx.get(match_key)
        if pi is not None:
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
    inner = _single_frame_layer_dict(df_frame)
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
    frames: dict[str, dict] = {}
    for fi, group in df.groupby("frame_index", sort=True):
        try:
            key = str(int(fi))
        except (TypeError, ValueError):
            continue
        frames[key] = _single_frame_layer_dict(group)
    return {"type": "bbox_layers_by_frame", "frames": frames}


def render_t4_three_js_embed(viewer_three_url: str, layer_payload: dict, height: int = 700) -> None:
    """Iframe to T4 three viewer + postMessage with bbox layer payload (GT, pred, matched pairs)."""
    _payload_json = json.dumps(layer_payload, ensure_ascii=True)
    _payload_b64 = _payload_json.encode("utf-8").hex()
    _iframe_src = html.escape(viewer_three_url, quote=True)
    components.html(
        (
            f'<iframe id="t4-three-viewer" src="{_iframe_src}" '
            f'width="100%" height="{height}" style="border:none;border-radius:8px;background:#e2e8f0" '
            f'allowfullscreen allow="fullscreen *" '
            f'loading="lazy" title="T4 three viewer" referrerpolicy="no-referrer-when-downgrade"></iframe>'
            "<script>"
            "(()=>{"
            "const iframe=document.getElementById('t4-three-viewer');"
            f"const payloadHex='{_payload_b64}';"
            "const hexToUtf8=(hex)=>{"
            "if(!hex||hex.length%2!==0)return '';"
            "const bytes=new Uint8Array(hex.length/2);"
            "for(let i=0;i<hex.length;i+=2){bytes[i/2]=parseInt(hex.slice(i,i+2),16)||0;}"
            "return new TextDecoder().decode(bytes);"
            "};"
            "let payload={type:'bbox_layers_clear'};"
            "try{"
            "const payloadJson=hexToUtf8(payloadHex);"
            "payload=JSON.parse(payloadJson);"
            "const fc=payload.frames&&typeof payload.frames==='object'?Object.keys(payload.frames).length:0;"
            "console.info('[bbox-debug] payload prepared', {type:payload.type,gt:(payload.gt||[]).length,pred:(payload.pred||[]).length,matched:(payload.matched_pairs||[]).length,frames:fc});"
            "}catch(err){"
            "console.error('[bbox-debug] payload parse failed', err);"
            "}"
            "let postCount=0;"
            "const post=(reason)=>{"
            "if(!iframe||!iframe.contentWindow)return;"
            "let targetOrigin='*';"
            "try{ targetOrigin = new URL(iframe.src, window.location.href).origin || '*'; }catch(_){ targetOrigin='*'; }"
            "postCount+=1;"
            "iframe.contentWindow.postMessage(payload,targetOrigin);"
            "console.info('[bbox-debug] postMessage sent', {reason,postCount,targetOrigin,payloadType:payload.type});"
            "};"
            "iframe.addEventListener('load',()=>{"
            "post('iframe-load');"
            "let n=0;"
            "const t=setInterval(()=>{post('retry');n+=1;if(n>12)clearInterval(t);},250);"
            "});"
            "setTimeout(()=>post('initial-delay-300ms'),300);"
            "setTimeout(()=>post('initial-delay-1200ms'),1200);"
            "})();"
            "</script>"
        ),
        height=height + 24,
        scrolling=True,
    )
