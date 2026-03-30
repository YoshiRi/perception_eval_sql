"""Build embeddable T4 dataset metadata: JSON records, query strings, and ``POST /render`` bodies.

Use with :mod:`lib.t4_visualizer_client` when wiring eval parquet rows or dashboards to ``t4-server``.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence
from urllib.parse import quote

from lib.t4_visualizer_client import (
    RenderRequest,
    TargetObjectIn,
    render_request_to_json_body,
    target_object_from_gt_row,
)


def t4_dataset_context(
    t4dataset_id: str,
    scenario_name: str,
    *,
    frame_index: Optional[int] = None,
    data_dir: Optional[str] = None,
    sample_token: Optional[str] = None,
) -> dict[str, Any]:
    """Structured record for logging, sidecar JSON, or UI state."""
    out: dict[str, Any] = {
        "t4dataset_id": t4dataset_id,
        "scenario_name": scenario_name,
    }
    if frame_index is not None:
        out["frame_index"] = int(frame_index)
    if data_dir:
        out["data_dir"] = data_dir
    if sample_token:
        out["sample_token"] = sample_token
    return out


def t4_share_query_params(
    t4dataset_id: str,
    scenario_name: str,
    frame_index: int = 0,
) -> str:
    """Query string without leading ``?`` (for bookmarks or deep links)."""
    return (
        f"t4dataset_id={quote(str(t4dataset_id), safe='')}"
        f"&scenario_name={quote(str(scenario_name), safe='')}"
        f"&frame_index={int(frame_index)}"
    )


def target_objects_from_rows(rows: Sequence[Mapping[str, Any]]) -> List[dict[str, Any]]:
    """Map each row to a ``target_objects`` dict (see :func:`target_object_from_gt_row`)."""
    return [target_object_from_gt_row(r) for r in rows]


def build_render_request_embed(
    t4dataset_id: str,
    scenario_name: str,
    frame_index: int,
    *,
    target_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    target_objects: Optional[Sequence[TargetObjectIn]] = None,
    show_annotations: bool = True,
    crop_cameras: bool = False,
    crop_padding: int = 40,
    crop_min_size: int = 300,
    cameras: Optional[List[str]] = None,
    version: Optional[str] = None,
) -> dict[str, Any]:
    """Return ``context`` plus a ``post_render_json`` body ready for ``POST /render``."""
    to_list: List[TargetObjectIn] = []
    if target_objects is not None:
        to_list = list(target_objects)
    elif target_rows is not None:
        for r in target_rows:
            d = target_object_from_gt_row(r)
            to_list.append(TargetObjectIn(**d))
    req = RenderRequest(
        t4dataset_id=t4dataset_id,
        scenario_name=scenario_name,
        frame_index=int(frame_index),
        target_objects=to_list,
        show_annotations=show_annotations,
        crop_cameras=crop_cameras,
        crop_padding=crop_padding,
        crop_min_size=crop_min_size,
        cameras=cameras,
        version=version,
    )
    body = render_request_to_json_body(req)
    return {
        "context": t4_dataset_context(t4dataset_id, scenario_name, frame_index=frame_index),
        "post_render_json": body,
    }
