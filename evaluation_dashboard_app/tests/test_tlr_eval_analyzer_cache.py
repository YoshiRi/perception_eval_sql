import json
import os
from pathlib import Path

from lib.tlr_eval_analyzer import TLREvaluationAnalyzer


def _frame(frame_name: str, ros_time: float, x: float, y: float, *, final: bool = False) -> dict:
    frame = {
        "FrameName": frame_name,
        "Ego": {
            "TransformStamped": {
                "transform": {"translation": {"x": x, "y": y}},
                "rotation_euler": {"yaw": 0.0},
            }
        },
        "criteria0": {
            "PassFail": {
                "Info": {
                    "TP": "1 [green]",
                    "FP": "0 []",
                    "FN": "0 []",
                    "TN": "0 []",
                }
            }
        },
    }
    result = {"Summary": ""}
    if final:
        frame["FinalScore"] = {"Score": {"TP": {"green": 1, "ALL": 1}}}
        result["Summary"] = "criteria 0 (Success): 1/1 -> done"
    return {"Frame": frame, "Stamp": {"ROS": ros_time}, "Result": result}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_tlr_analyzer_uses_compact_cache_and_invalidates_on_source_change(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    scenario_dir = tmp_path / "tlr_run" / "scenario_a"
    scenario_dir.mkdir(parents=True)
    result_json = scenario_dir / "result.json"
    _write_jsonl(
        result_json,
        [
            _frame("frame_0", 0.0, 0.0, 0.0),
            _frame("frame_1", 0.1, 0.0, 0.0),
            _frame("frame_2", 0.2, 1.0, 0.0, final=True),
        ],
    )

    first = TLREvaluationAnalyzer(str(tmp_path / "tlr_run"))
    first.load_all_results()
    first.extract_criteria_data()
    first.pre_calculate_all_data()

    assert not first.loaded_from_cache
    assert first._cache_path().is_file()
    assert first.get_summary_stats()["num_scenarios"] == 1

    second = TLREvaluationAnalyzer(str(tmp_path / "tlr_run"))
    second.load_all_results()
    second.extract_criteria_data()
    second.pre_calculate_all_data()

    assert second.loaded_from_cache
    assert second.scenario_results == {}
    assert second.get_summary_stats()["num_scenarios"] == 1
    assert len(second.get_vehicle_status_details_df()) == 3

    with result_json.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_frame("frame_3", 0.3, 2.0, 0.0)) + "\n")
    os.utime(result_json, None)

    third = TLREvaluationAnalyzer(str(tmp_path / "tlr_run"))
    third.load_all_results()

    assert not third.loaded_from_cache
    assert len(third.scenario_results["scenario_a"]) == 4
