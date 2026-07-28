from urllib.parse import parse_qs

import pandas as pd

from lib.t4_three_layers import (
    EXTERNAL_BBOX_ALIGNMENT_VERSION,
    _pack_three_layer_payload_binary,
    build_three_layer_payload_all_frames,
    infer_external_bbox_alignment_query_params,
    infer_legacy_width_length_swapped,
    resolve_t4_dataset_id,
)


def _params(df: pd.DataFrame) -> dict[str, list[str]]:
    return parse_qs(infer_external_bbox_alignment_query_params(df))


def _corner_extents(corners: list[float]) -> tuple[float, float, float]:
    xs = corners[0::3]
    ys = corners[1::3]
    zs = corners[2::3]
    return max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)


def test_resolve_t4_dataset_id_uses_name_when_id_is_placeholder():
    df = pd.DataFrame(
        {
            "t4dataset_id": ["00000000-0000-0000-0000-000000000001"],
            "t4dataset_name": ["DevOps_V1_J6Gen2_Shiojiri_FP_IntersectionRight_Board_PCOff_DT001956"],
        }
    )

    assert resolve_t4_dataset_id(df) == (
        "DevOps_V1_J6Gen2_Shiojiri_FP_IntersectionRight_Board_PCOff_DT001956"
    )


def test_infer_external_bbox_alignment_for_length_forward_exports():
    df = pd.DataFrame(
        {
            "source": ["GT", "GT", "EST"],
            "label": ["car", "truck", "car"],
            "length": [4.0, 8.0, 3.8],
            "width": [1.8, 2.4, 1.7],
        }
    )

    params = _params(df)

    assert params["external_bbox_yaw_offset"] == ["0"]
    assert params["external_bbox_swap_lw"] == ["false"]
    assert params["external_bbox_alignment_version"] == [EXTERNAL_BBOX_ALIGNMENT_VERSION]


def test_infer_external_bbox_alignment_normalizes_legacy_width_forward_exports_in_payload():
    df = pd.DataFrame(
        {
            "source": ["GT", "GT", "EST"],
            "label": ["car", "truck", "car"],
            "length": [1.8, 2.4, 1.7],
            "width": [4.0, 8.0, 3.8],
        }
    )

    params = _params(df)

    assert infer_legacy_width_length_swapped(df) is True
    assert params["external_bbox_yaw_offset"] == ["0"]
    assert params["external_bbox_swap_lw"] == ["false"]
    assert params["external_bbox_alignment_version"] == [EXTERNAL_BBOX_ALIGNMENT_VERSION]


def test_payload_keeps_length_forward_release_boxes_with_explicit_corners():
    df = pd.DataFrame(
        {
            "frame_index": [0],
            "source": ["GT"],
            "status": ["TP"],
            "label": ["car"],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "length": [4.0],
            "width": [2.0],
            "height": [1.5],
            "yaw": [0.0],
            "uuid": ["box-1"],
        }
    )

    payload = build_three_layer_payload_all_frames(df)
    box = payload["frames"]["0"]["gt"][0]

    assert box["length"] == 4.0
    assert box["width"] == 2.0
    assert box["force_wireframe"] is True
    assert _corner_extents(box["corners"]) == (4.0, 2.0, 1.5)


def test_payload_includes_polygon_footprint_vertices_in_binary_stats():
    df = pd.DataFrame(
        {
            "frame_index": [0],
            "source": ["EST"],
            "status": ["FP"],
            "label": ["unknown"],
            "shape_type": ["polygon"],
            "x": [10.0],
            "y": [3.0],
            "z": [0.5],
            "length": [0.0],
            "width": [0.0],
            "height": [1.5],
            "yaw": [0.0],
            "uuid": ["poly-1"],
            "footprint": [[[9.0, 2.0, 0.0], [11.0, 2.0, 0.0], [10.5, 4.0, 0.0], [9.0, 2.0, 0.0]]],
        }
    )

    payload = build_three_layer_payload_all_frames(df)
    box = payload["frames"]["0"]["pred"][0]
    _blob, stats = _pack_three_layer_payload_binary(payload)

    assert box["shape_type"] == "invalid_polygon_marker"
    assert box["footprint"] == [[9.0, 2.0, 0.0], [11.0, 2.0, 0.0], [10.5, 4.0, 0.0], [9.0, 2.0, 0.0]]
    assert stats["footprint_box_count"] == 1
    assert stats["footprint_point_count"] == 4


def test_payload_swaps_legacy_width_forward_boxes_with_explicit_corners():
    df = pd.DataFrame(
        {
            "frame_index": [0],
            "source": ["GT"],
            "status": ["TP"],
            "label": ["car"],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "length": [2.0],
            "width": [4.0],
            "height": [1.5],
            "yaw": [0.0],
            "uuid": ["box-1"],
        }
    )

    payload = build_three_layer_payload_all_frames(df)
    box = payload["frames"]["0"]["gt"][0]

    assert box["length"] == 4.0
    assert box["width"] == 2.0
    assert box["yaw"] == 0.0
    assert box["force_wireframe"] is True
    assert _corner_extents(box["corners"]) == (4.0, 2.0, 1.5)


def test_payload_all_frames_includes_compare_run_order():
    df = pd.DataFrame(
        {
            "frame_index": [0, 0],
            "source": ["GT", "GT"],
            "status": ["TP", "FN"],
            "label": ["car", "car"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 0.0],
            "length": [4.0, 4.0],
            "width": [2.0, 2.0],
            "height": [1.5, 1.5],
            "yaw": [0.0, 0.0],
            "uuid": ["a-box", "b-box"],
            "run": ["A", "B"],
        }
    )

    payload = build_three_layer_payload_all_frames(df)

    assert payload["compare_runs"] == ["A", "B"]
    frame_gt = payload["frames"]["0"]["gt"]
    assert [box["run"] for box in frame_gt] == ["A", "B"]


def test_payload_deduplicates_same_frame_tracking_rows_by_object_identity():
    df = pd.DataFrame(
        {
            "frame_index": [2, 2, 2, 2],
            "source": ["EST", "EST", "GT", "GT"],
            "status": ["TP", "TP", "TP", "TP"],
            "label": ["car", "car", "car", "car"],
            "x": [40.52, 40.79, 40.75, 40.75],
            "y": [35.28, 35.39, 35.80, 35.80],
            "z": [2.34, 2.21, 1.63, 1.63],
            "length": [3.99, 4.08, 4.07, 4.07],
            "width": [1.88, 1.88, 1.78, 1.78],
            "height": [1.63, 1.61, 1.49, 1.49],
            "yaw": [-1.64, -1.63, -1.65, -1.65],
            "uuid": ["est-1", "est-1", "gt-1", "gt-1"],
            "pair_uuid": ["gt-1", "gt-1", "est-1", "est-1"],
            "pair_dt_sec": [0.036, -0.054, 0.036, -0.054],
        }
    )

    payload = build_three_layer_payload_all_frames(df)
    frame = payload["frames"]["2"]

    assert len(frame["pred"]) == 1
    assert len(frame["gt"]) == 1
    assert frame["pred"][0]["x"] == 40.52
    assert frame["gt"][0]["uuid"] == "gt-1"
    assert frame["matched_pairs"] == [{"gt_idx": 0, "pred_idx": 0, "pair_uuid": "est-1"}]


def test_payload_deduplicates_tracking_rows_per_compare_run():
    df = pd.DataFrame(
        {
            "frame_index": [2, 2],
            "source": ["GT", "GT"],
            "status": ["TP", "TP"],
            "label": ["car", "car"],
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
            "z": [0.0, 0.0],
            "length": [4.0, 4.0],
            "width": [2.0, 2.0],
            "height": [1.5, 1.5],
            "yaw": [0.0, 0.0],
            "uuid": ["shared-gt", "shared-gt"],
            "pair_uuid": ["shared-est", "shared-est"],
            "run": ["A", "B"],
        }
    )

    payload = build_three_layer_payload_all_frames(df)
    frame_gt = payload["frames"]["2"]["gt"]

    assert len(frame_gt) == 2
    assert [box["run"] for box in frame_gt] == ["A", "B"]


def test_payload_deduplicates_tp_rows_by_pair_uuid_before_uuid():
    df = pd.DataFrame(
        {
            "frame_index": [2, 2],
            "source": ["EST", "EST"],
            "status": ["TP", "TP"],
            "label": ["car", "car"],
            "x": [-13.22, -12.95],
            "y": [1.09, 1.03],
            "z": [0.85, 0.95],
            "length": [4.11, 4.10],
            "width": [1.99, 1.87],
            "height": [1.57, 1.33],
            "yaw": [-0.21, -0.21],
            "uuid": ["est-old", "est-new"],
            "pair_uuid": ["775a6e77", "775a6e77"],
            "pair_dt_sec": [0.036, -0.054],
            "run": ["B", "B"],
        }
    )

    payload = build_three_layer_payload_all_frames(df)
    frame_pred = payload["frames"]["2"]["pred"]

    assert len(frame_pred) == 1
    assert frame_pred[0]["uuid"] == "est-old"
