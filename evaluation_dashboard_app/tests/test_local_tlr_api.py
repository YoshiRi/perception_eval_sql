"""Tests for the TLR analysis JSON routes on the local bbox API.

The routes are plain ``fn(payload) -> dict`` functions, so they are exercised
directly against a synthetic TLR result tree (suite/testcase/result.json JSONL)
under a temporary data root — no HTTP server needed.
"""

import json
import math
from pathlib import Path

import pytest

import lib.path_utils as path_utils
from backend import local_bbox_api


def _frame(
    frame_name: str,
    ros_time: float,
    x: float,
    *,
    tp: str = "1 [green]",
    fn: str = "0 []",
    criteria: str = "criteria0",
    summary: str = "",
) -> dict:
    """One result.json line in the shape _frame_to_tlr_dict/_calculate_vehicle_status expect."""
    frame = {
        "FrameName": frame_name,
        "Ego": {
            "TransformStamped": {
                "transform": {"translation": {"x": x, "y": 0.0}},
                "rotation_euler": {"yaw": 0.0},
            }
        },
        criteria: {
            "PassFail": {
                "Info": {"TP": tp, "FP": "0 []", "FN": fn, "TN": "0 []"}
            }
        },
    }
    result = {"Summary": ""}
    if summary:
        frame["FinalScore"] = {"Score": {"TP": {"green": 1, "ALL": 1}}}
        result["Summary"] = summary
    return {"Frame": frame, "Stamp": {"ROS": ros_time}, "Result": result}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture()
def tlr_root(tmp_path: Path, monkeypatch) -> Path:
    """Data root with one TLR run: a suite-style scenario and a flat scenario."""
    monkeypatch.chdir(tmp_path)  # analyzer's derived cache writes under cwd/.cache
    monkeypatch.setenv("EVAL_DASHBOARD_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(path_utils, "_DATA_ROOT", None)
    local_bbox_api._TLR_ANALYZER_CACHE.clear()

    run = tmp_path / "tlr_run"
    # Suite layout: suite_a/testcase_1/result.json — mostly TPs, ends with a summary.
    _write_jsonl(
        run / "suite_a" / "testcase_1" / "result.json",
        [
            _frame("f0", 0.0, 0.0),
            _frame("f1", 0.1, 1.0),
            _frame("f2", 0.2, 2.0, summary="criteria 0 (Success): 2/3 -> done"),
        ],
    )
    # Flat layout: scenario_b/result.json — red-signal FNs on a critical criteria.
    _write_jsonl(
        run / "scenario_b" / "result.json",
        [
            _frame("g0", 0.0, 0.0, tp="0 []", fn="1 [red]", criteria="criteria5"),
            _frame("g1", 0.1, 0.0, tp="0 []", fn="1 [red]", criteria="criteria5"),
            _frame(
                "g2", 0.2, 0.5, tp="1 [red]", fn="0 []", criteria="criteria5",
                summary="criteria 5 (Fail): 1/3 -> done",
            ),
        ],
    )
    return tmp_path


def _assert_json_safe(payload) -> None:
    """The route result must survive the server's JSON encoder (no NaN, no numpy)."""
    json.dumps(local_bbox_api._json_safe(payload), allow_nan=False)


def test_tlr_dirs_discovers_run(tlr_root: Path) -> None:
    out = local_bbox_api.tlr_dirs({})
    _assert_json_safe(out)
    items = {item["path"]: item["scenarios"] for item in out["items"]}
    assert items == {"tlr_run": 2}


def test_tlr_summary_stats_and_rollups(tlr_root: Path) -> None:
    out = local_bbox_api.tlr_summary({"path": "tlr_run"})
    _assert_json_safe(out)

    stats = out["stats"]
    assert stats["num_scenarios"] == 2
    assert stats["total_frames"] == 6  # 3 + 3 from the two Summary strings
    assert stats["total_tp"] == 3
    assert stats["overall_tp_rate"] == pytest.approx(0.5)
    # The matrix always spans criteria_0..20; unevaluated ones report 0.0, so the
    # "worst" is simply the first zero-rate criteria. Assert semantics, not the index.
    assert stats["best_criteria"] == "criteria_0"
    assert stats["best_tp_rate"] == pytest.approx(2 / 3)
    assert stats["worst_tp_rate"] == 0.0

    criteria = out["criteria_matrix"]
    assert "Criteria" in criteria["columns"]
    by_name = {row["Criteria"]: row for row in criteria["records"]}
    assert by_name["criteria_0"]["Number of TP"] == 2
    assert by_name["criteria_5"]["Number of total frames"] == 3

    scenarios = out["scenario_summary"]
    by_scenario = {row["scenario"]: row for row in scenarios["records"]}
    assert set(by_scenario) == {"suite_a/testcase_1", "scenario_b"}
    assert by_scenario["suite_a/testcase_1"]["suite"] == "suite_a"
    assert by_scenario["suite_a/testcase_1"]["frames"] == 3
    assert by_scenario["scenario_b"]["fn_frames"] == 2
    assert by_scenario["scenario_b"]["tp_rate"] == pytest.approx(1 / 3)


def test_tlr_matrices_shape_and_json_safety(tlr_root: Path) -> None:
    out = local_bbox_api.tlr_matrices({"path": "tlr_run"})
    _assert_json_safe(out)

    vs = out["vehicle_status"]
    assert vs["columns"][0] == "Vehicle Status"
    assert [row["Vehicle Status"] for row in vs["records"]] == [
        "Turning", "Driving", "No Move", "All Status Combined",
    ]
    combined = vs["records"][-1]["all types combined"]
    assert combined is None or (0.0 <= combined <= 1.0 and math.isfinite(combined))

    counts = out["vehicle_status_counts"]
    assert all("/" in str(row["all types combined"]) for row in counts["records"])

    critical = out["critical_priority"]
    assert any("critical zone" in c for c in critical["columns"])
    assert len(out["critical_priority_counts"]["records"]) == 4


def test_tlr_frames_filters_one_scenario(tlr_root: Path) -> None:
    out = local_bbox_api.tlr_frames({"path": "tlr_run", "scenario": "scenario_b"})
    _assert_json_safe(out)
    assert out["scenario"] == "scenario_b"
    assert len(out["records"]) == 3
    assert set(out["columns"]) <= set(local_bbox_api._TLR_FRAME_COLUMNS)
    assert [row["frame_index"] for row in out["records"]] == [0, 1, 2]
    assert {row["criteria"] for row in out["records"]} == {"criteria5"}
    fn_rows = [row for row in out["records"] if row["fn"] == "1 [red]"]
    assert len(fn_rows) == 2

    other = local_bbox_api.tlr_frames({"path": "tlr_run", "scenario": "suite_a/testcase_1"})
    assert len(other["records"]) == 3
    assert all(row["tp"] == "1 [green]" for row in other["records"])


def test_tlr_frames_requires_scenario(tlr_root: Path) -> None:
    with pytest.raises(ValueError):
        local_bbox_api.tlr_frames({"path": "tlr_run"})


def test_tlr_analyzer_cache_reuses_and_invalidates(tlr_root: Path) -> None:
    first = local_bbox_api._tlr_analyzer({"path": "tlr_run"})
    second = local_bbox_api._tlr_analyzer({"path": "tlr_run"})
    assert first is second

    # Appending a frame changes the source signature and forces a re-read.
    result_json = tlr_root / "tlr_run" / "scenario_b" / "result.json"
    with result_json.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_frame("g3", 0.3, 1.0)) + "\n")
    third = local_bbox_api._tlr_analyzer({"path": "tlr_run"})
    assert third is not first
    assert len(local_bbox_api._TLR_ANALYZER_CACHE) <= local_bbox_api._TLR_ANALYZER_CACHE_MAX


def test_tlr_summary_rejects_paths_outside_root(tlr_root: Path) -> None:
    with pytest.raises(ValueError):
        local_bbox_api.tlr_summary({"path": "../outside"})
