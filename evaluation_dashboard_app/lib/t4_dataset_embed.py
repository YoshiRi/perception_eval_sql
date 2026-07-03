"""Build embeddable T4 dataset metadata: JSON records, query strings, and ``POST /render`` bodies.

Use with :mod:`lib.t4_visualizer_client` when wiring eval parquet rows or dashboards to ``t4-server``.
"""

from __future__ import annotations

import json
from typing import Any, List, Mapping, Optional, Sequence
from urllib.parse import quote, urlencode

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


def t4_dashboard_query_params(
    *,
    mode: str,
    run_names: Sequence[str],
    suite_name: Optional[Any] = None,
    scenario_name: Optional[Any] = None,
    t4dataset_name: Optional[Any] = None,
    t4dataset_id: Optional[Any] = None,
    frame_index: Optional[Any] = None,
    compare_view_mode: Optional[Any] = None,
) -> str:
    """Query string for opening the dashboard T4 3D Viewer with run and scene context."""

    placeholder_dataset_ids = {
        "00000000-0000-0000-0000-000000000000",
        "00000000-0000-0000-0000-000000000001",
    }

    def _clean(value: Optional[Any]) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() in {"none", "nan", "<na>"}:
            return ""
        if text in placeholder_dataset_ids:
            return ""
        return text

    query: dict[str, str] = {
        "mode": "compare" if str(mode).lower().startswith("compare") else "single",
    }
    clean_runs = [_clean(name) for name in run_names]
    clean_runs = [name for name in clean_runs if name]
    if clean_runs:
        query["run_a"] = clean_runs[0]
        for idx, run_name in enumerate(clean_runs[1:5]):
            query[f"run_{chr(98 + idx)}"] = run_name

    t4dataset_value = t4dataset_name if _clean(t4dataset_name) else t4dataset_id
    viewer_params = {
        "viewer_suite": suite_name,
        "viewer_scenario": scenario_name,
        "viewer_t4dataset": t4dataset_value,
        "viewer_frame": frame_index,
        "viewer_compare": compare_view_mode,
    }
    for key, value in viewer_params.items():
        clean = _clean(value)
        if clean:
            query[key] = clean
    return urlencode(query)


def t4_dashboard_url(
    *,
    mode: str,
    run_names: Sequence[str],
    suite_name: Optional[Any] = None,
    scenario_name: Optional[Any] = None,
    t4dataset_name: Optional[Any] = None,
    t4dataset_id: Optional[Any] = None,
    frame_index: Optional[Any] = None,
    compare_view_mode: Optional[Any] = None,
    page_path: str = "/T4_3D_Viewer",
) -> str:
    """Relative URL for the Streamlit dashboard T4 3D Viewer."""
    query = t4_dashboard_query_params(
        mode=mode,
        run_names=run_names,
        suite_name=suite_name,
        scenario_name=scenario_name,
        t4dataset_name=t4dataset_name,
        t4dataset_id=t4dataset_id,
        frame_index=frame_index,
        compare_view_mode=compare_view_mode,
    )
    return f"{page_path}?{query}" if query else page_path


def t4_share_query_params_from_post_render_json(body: Mapping[str, Any]) -> str:
    """Query string (no ``?``) with a single ``render_json`` param: same object as curl ``-d`` / ``post_render_json``."""
    compact = json.dumps(dict(body), separators=(",", ":"), ensure_ascii=False)
    return f"render_json={quote(compact, safe='')}"


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
