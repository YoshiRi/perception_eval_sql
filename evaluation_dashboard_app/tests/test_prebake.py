"""Pre-baked answers must be keyed on meaning and narrowed like the live handlers."""

import gzip
import json
from pathlib import Path

import pytest

from backend import prebake

FILTERS = {
    "suite_name": "DevOps_V1_FN_Opened_Door_abc",
    "scenario_name": "DevOps_V1_J6Gen2_Shiojiri_FN_NA_Car_PCOff_DT002778",
    "topic_name": "perception.object_recognition.objects",
}


@pytest.fixture()
def parquet(tmp_path: Path) -> Path:
    path = tmp_path / "devops" / "current.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"PAR1")
    return path


def _payload(**extra):
    return {"path": "/anywhere/current.parquet", "filters": dict(FILTERS), **extra}


# ------------------------------------------------------------------ cache keys


def test_key_requires_a_scenario(parquet):
    assert prebake.cache_key(prebake.ROUTE_TN_OBJECTS, parquet, {"filters": {}}) is None


def test_key_ignores_incidental_payload_noise(parquet):
    """run label, timeouts and row caps must not fragment the cache."""
    base = prebake.cache_key(prebake.ROUTE_TN_OBJECTS, parquet, _payload())
    noisy = prebake.cache_key(
        prebake.ROUTE_TN_OBJECTS,
        parquet,
        _payload(run="B", timeout_ms=8000, max_rows=70000),
    )
    assert base == noisy


def test_key_ignores_frame_window(parquet):
    """The stored answer covers the scenario; frame windows are applied on read."""
    base = prebake.cache_key(prebake.ROUTE_FRAME_RESULTS, parquet, _payload())
    windowed = prebake.cache_key(
        prebake.ROUTE_FRAME_RESULTS,
        parquet,
        {"path": "x", "filters": {**FILTERS, "frame_min": 3, "frame_max": 9}},
    )
    assert base == windowed


def test_key_separates_scenarios_and_routes(parquet):
    other = {"path": "x", "filters": {**FILTERS, "scenario_name": "OTHER"}}
    assert prebake.cache_key(prebake.ROUTE_TN_OBJECTS, parquet, _payload()) != prebake.cache_key(
        prebake.ROUTE_TN_OBJECTS, parquet, other
    )
    assert prebake.cache_key(prebake.ROUTE_TN_OBJECTS, parquet, _payload()) != prebake.cache_key(
        prebake.ROUTE_FRAME_RESULTS, parquet, _payload()
    )


def test_devops_result_key_includes_filters_that_reach_sql(parquet):
    """This route queries the parquet, so a different label is a different answer."""
    base = prebake.cache_key(prebake.ROUTE_DEVOPS_RESULT, parquet, _payload())
    labelled = prebake.cache_key(
        prebake.ROUTE_DEVOPS_RESULT, parquet, {"path": "x", "filters": {**FILTERS, "label": "car"}}
    )
    assert base != labelled


def test_devops_result_key_tracks_exact_flag(parquet):
    assert prebake.cache_key(prebake.ROUTE_DEVOPS_RESULT, parquet, _payload()) != prebake.cache_key(
        prebake.ROUTE_DEVOPS_RESULT, parquet, _payload(exact=True)
    )


def test_devops_result_key_ignores_empty_filter_values(parquet):
    """'Any'/'' are how the UI spells "no filter"; they must not split the cache."""
    base = prebake.cache_key(prebake.ROUTE_DEVOPS_RESULT, parquet, _payload())
    padded = prebake.cache_key(
        prebake.ROUTE_DEVOPS_RESULT,
        parquet,
        {"path": "x", "filters": {**FILTERS, "label": "Any", "status": "", "visibility": None}},
    )
    assert base == padded


def test_unknown_route_has_no_key(parquet):
    assert prebake.cache_key("/api/frames", parquet, _payload()) is None


# --------------------------------------------------------------- write and read


def test_round_trip_marks_the_answer_as_prebaked(parquet):
    stored = {"available": True, "source": "scene_result.pkl", "frames": [], "frame_count": 0}
    prebake.write(prebake.ROUTE_FRAME_RESULTS, parquet, _payload(), stored)
    got = prebake.read(prebake.ROUTE_FRAME_RESULTS, parquet, _payload())
    assert got is not None and got["prebaked"] is True


def test_read_miss_returns_none(parquet):
    assert prebake.read(prebake.ROUTE_FRAME_RESULTS, parquet, _payload()) is None


def test_stored_next_to_the_parquet(parquet):
    prebake.write(prebake.ROUTE_TN_OBJECTS, parquet, _payload(), {"available": True, "frames": []})
    directory = prebake.prebake_dir(parquet)
    assert directory == parquet.parent / prebake.PREBAKE_DIRNAME
    assert list((directory / prebake.ROUTE_TN_OBJECTS).glob("*.json.gz"))


def test_corrupt_cache_is_ignored_not_fatal(parquet):
    prebake.write(prebake.ROUTE_TN_OBJECTS, parquet, _payload(), {"available": True, "frames": []})
    victim = next((prebake.prebake_dir(parquet) / prebake.ROUTE_TN_OBJECTS).glob("*.json.gz"))
    victim.write_bytes(b"this is not gzip")
    assert prebake.read(prebake.ROUTE_TN_OBJECTS, parquet, _payload()) is None


def test_written_bytes_are_deterministic(parquet):
    """Stable bytes keep the export manifest's sha256 stable across regeneration."""
    payload = {"available": True, "frames": [{"frame": 1, "boxes": []}]}
    first = prebake.write(prebake.ROUTE_TN_OBJECTS, parquet, _payload(), payload).read_bytes()
    second = prebake.write(prebake.ROUTE_TN_OBJECTS, parquet, _payload(), payload).read_bytes()
    assert first == second


# ------------------------------------------------------------------- narrowing


def _tn_stored():
    return {
        "available": True,
        "source": "scene_result.pkl",
        "frames": [
            {"frame": 1, "boxes": [{"x": 1.0, "run": "A"}, {"x": 2.0, "run": "A"}]},
            {"frame": 5, "boxes": [{"x": 3.0, "run": "A"}]},
            {"frame": 9, "boxes": [{"x": 4.0, "run": "A"}]},
        ],
        "row_count": 4,
        "frame_count": 3,
    }


def test_tn_frame_window_narrows(parquet):
    prebake.write(prebake.ROUTE_TN_OBJECTS, parquet, _payload(), _tn_stored())
    got = prebake.read(
        prebake.ROUTE_TN_OBJECTS,
        parquet,
        {"path": "x", "filters": {**FILTERS, "frame_min": 5, "frame_max": 9}},
    )
    assert [f["frame"] for f in got["frames"]] == [5, 9]
    assert got["row_count"] == 2


def test_tn_exact_frame_narrows(parquet):
    prebake.write(prebake.ROUTE_TN_OBJECTS, parquet, _payload(), _tn_stored())
    got = prebake.read(prebake.ROUTE_TN_OBJECTS, parquet, _payload(frame_index=5))
    assert [f["frame"] for f in got["frames"]] == [5]


def test_tn_boxes_are_restamped_with_the_requested_run(parquet):
    """The live handler stamps the caller's run label; the cache must too, or the
    viewer would attribute Run B's true negatives to Run A."""
    prebake.write(prebake.ROUTE_TN_OBJECTS, parquet, _payload(), _tn_stored())
    got = prebake.read(prebake.ROUTE_TN_OBJECTS, parquet, _payload(run="B"))
    labels = {box["run"] for frame in got["frames"] for box in frame["boxes"]}
    assert labels == {"B"}


def test_tn_max_rows_truncates(parquet):
    prebake.write(prebake.ROUTE_TN_OBJECTS, parquet, _payload(), _tn_stored())
    got = prebake.read(prebake.ROUTE_TN_OBJECTS, parquet, _payload(max_rows=2))
    assert got["row_count"] == 2
    assert got["truncated"] is True


def test_frame_results_max_frames_truncates(parquet):
    stored = {
        "available": True,
        "source": "scene_result.pkl",
        "frames": [{"frame": i, "judged": True, "passed": True} for i in range(10)],
        "frame_count": 10,
    }
    prebake.write(prebake.ROUTE_FRAME_RESULTS, parquet, _payload(), stored)
    got = prebake.read(prebake.ROUTE_FRAME_RESULTS, parquet, _payload(max_frames=3))
    assert got["frame_count"] == 3
    assert got["truncated"] is True


def test_unavailable_answer_is_passed_through_unchanged(parquet):
    """No frame bookkeeping may be invented for an answer that has no frames, or the
    cached shape would differ from what the live handler returns."""
    stored = {
        "available": False,
        "source": "scene_result.pkl",
        "frames": [],
        "frame_count": 0,
        "reason": "scene_result.pkl was not found",
    }
    prebake.write(prebake.ROUTE_FRAME_RESULTS, parquet, _payload(), stored)
    got = prebake.read(prebake.ROUTE_FRAME_RESULTS, parquet, _payload())
    got.pop("prebaked")
    assert got == stored
    assert "truncated" not in got


def test_devops_result_is_not_narrowed(parquet):
    stored = {"overall_pass": True, "gates": [{"name": "g"}], "hot_frames": [1, 2, 3]}
    prebake.write(prebake.ROUTE_DEVOPS_RESULT, parquet, _payload(), stored)
    got = prebake.read(prebake.ROUTE_DEVOPS_RESULT, parquet, _payload(frame_index=2))
    got.pop("prebaked")
    assert got == stored


# ----------------------------------------------------------------------- index


def test_index_round_trip(parquet):
    prebake._write_index(parquet, {"scenarios": 3, "written": 6, "skipped": 1, "failed": 0, "routes": []})
    index = prebake.read_index(parquet)
    assert index["written"] == 6 and index["version"] == prebake.PREBAKE_VERSION


def test_read_index_absent(parquet):
    assert prebake.read_index(parquet) is None


def test_enabled_respects_the_kill_switch(monkeypatch):
    monkeypatch.delenv("EVAL_PREBAKE_READ", raising=False)
    assert prebake.enabled() is True
    monkeypatch.setenv("EVAL_PREBAKE_READ", "0")
    assert prebake.enabled() is False


def test_version_bump_invalidates_old_entries(parquet, monkeypatch):
    """A shape change must not serve entries written by the previous version."""
    prebake.write(prebake.ROUTE_TN_OBJECTS, parquet, _payload(), {"available": True, "frames": []})
    assert prebake.read(prebake.ROUTE_TN_OBJECTS, parquet, _payload()) is not None
    monkeypatch.setattr(prebake, "PREBAKE_VERSION", prebake.PREBAKE_VERSION + 1)
    assert prebake.read(prebake.ROUTE_TN_OBJECTS, parquet, _payload()) is None
