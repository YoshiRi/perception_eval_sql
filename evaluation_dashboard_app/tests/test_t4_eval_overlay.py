"""The 3D scene is only half the story without the run's own boxes.

The cached t4-server viewer draws the point cloud and the dataset's annotations; what
the evaluator predicted, and which of it was TP/FP/FN, lives in the parquet. These
tests cover the two joints that carry it across: the packed overlay payload, and the
cached page being re-pointed at the query it is actually served with.
"""

import base64
import json
import urllib.parse
from pathlib import Path

import pandas as pd
import pytest

from backend.local_bbox_api import t4_layers
from client import config, t4
from tests.fake_t4_server import FakeT4Server

SUITE = "FullPerformance_V1_Town_PDD_1111"
SCENARIO = "FullPerformance_V1_Town_PDD001_2222"
TOPIC = "perception.object_recognition.objects"


def _row(frame, source, status, x, label="car"):
    return {
        "frame_index": frame, "source": source, "status": status,
        "x": x, "y": 0.0, "z": 0.0, "yaw": 0.0,
        "length": 4.0, "width": 2.0, "height": 1.5, "label": label,
        "uuid": f"{source}-{frame}-{x}",
        "suite_name": SUITE, "scenario_name": SCENARIO, "topic_name": TOPIC,
    }


@pytest.fixture()
def parquet(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(tmp_path))
    path = tmp_path / "current.parquet"
    pd.DataFrame([
        _row(1, "GT", "TP", 10.0), _row(1, "EST", "TP", 10.2),
        _row(1, "GT", "FN", 25.0, "pedestrian"), _row(1, "EST", "FP", 40.0),
        _row(2, "GT", "TP", 11.0), _row(2, "EST", "TP", 11.1),
    ]).to_parquet(path)
    return path


def _layers(parquet, **filters):
    base = {"suite_name": SUITE, "scenario_name": SCENARIO, "topic_name": TOPIC}
    return t4_layers({"path": str(parquet), "filters": {**base, **filters}})


# ------------------------------------------------------------------- the payload


def test_the_payload_carries_gt_and_predictions_per_frame(parquet):
    stats = _layers(parquet)["stats"]
    assert stats["format"] == "T4BBOX1"
    assert stats["frame_count"] == 2
    assert stats["gt_box_count"] == 3 and stats["pred_box_count"] == 3
    assert (stats["first_frame"], stats["last_frame"]) == (1, 2)


def test_the_payload_is_decodable_bytes(parquet):
    out = _layers(parquet)
    blob = base64.b64decode(out["payload_b64"])
    assert blob.startswith(b"T4BBOX1\x00")
    assert len(blob) == out["stats"]["binary_bytes"]


def test_the_alignment_convention_travels_with_it(parquet):
    """Without it the viewer applies its own default yaw offset and boxes sit askew."""
    query = urllib.parse.parse_qs(_layers(parquet)["viewer_query"])
    assert query["external_bbox_swap_lw"] == ["false"]
    assert "external_bbox_yaw_offset" in query and "external_bbox_alignment_version" in query


def test_the_explorer_filters_reach_the_overlay(parquet):
    """The 2D preview and the 3D scene must not disagree about what is drawn."""
    stats = _layers(parquet, distance_max=30.0)["stats"]
    assert stats["pred_box_count"] == 2  # the 40 m FP is filtered out


def test_a_scenario_with_no_rows_yields_an_empty_payload(parquet):
    stats = _layers(parquet, scenario_name="nothing_matches")["stats"]
    assert stats["frame_count"] == 0 and stats["box_count"] == 0


# ------------------------------------------------------- the cached page's query


@pytest.fixture()
def cached_scene(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALDASH_HOME", str(tmp_path / "home"))
    config.ensure_dirs()
    with FakeT4Server(frames=3, points=50) as server:
        t4.fetch_scene("T4DS0001", "scene-alpha", base_url=server.base_url)
    return "T4DS0001", "scene-alpha"


def _page(cached_scene, **params):
    dataset, scenario = cached_scene
    query = {"t4dataset_id": [dataset], "scenario_name": [scenario]}
    query.update({k: [str(v)] for k, v in params.items()})
    body, _, _ = t4.serve_request("/viewer/three", query)
    return body.decode()


def test_the_cached_page_is_re_pointed_at_the_query_it_is_served_with(cached_scene):
    """It was rendered with its query baked in, so offline it ignored its own URL."""
    text = _page(cached_scene, frame_index=2, external_bbox_yaw_offset="0")
    baked = text.split('new URLSearchParams("')[1].split('")')[0]
    query = urllib.parse.parse_qs(baked)
    assert query["frame_index"] == ["2"]
    assert query["external_bbox_yaw_offset"] == ["0"]


def test_the_scene_identity_is_never_taken_from_the_caller(cached_scene):
    """Those two name the cache directory; a mismatch would fetch the wrong frames."""
    text = _page(cached_scene, t4dataset_id="T4DS0001", scenario_name="scene-alpha")
    baked = text.split('new URLSearchParams("')[1].split('")')[0]
    query = urllib.parse.parse_qs(baked)
    assert query["t4dataset_id"] == ["T4DS0001"]
    assert query["scenario_name"] == ["scene-alpha"]


def test_a_page_without_the_literal_is_served_verbatim(cached_scene):
    """A newer viewer shape must degrade to the old behaviour, not to a broken page."""
    dataset, scenario = cached_scene
    page = t4.scene_dir(dataset, scenario) / "page.html"
    page.write_text("<html><body>no query literal here</body></html>", encoding="utf-8")
    assert "no query literal here" in _page(cached_scene, frame_index=2)
