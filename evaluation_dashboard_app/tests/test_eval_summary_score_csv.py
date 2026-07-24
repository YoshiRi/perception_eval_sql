import csv
import json
from pathlib import Path

from lib.eval_summary import generate_summary_and_score_csv


def _write_case(
    root: Path,
    name: str,
    *,
    t4_dataset_id: str | None = None,
    scenario_dataset_id: str | None = None,
) -> None:
    case_dir = root / "suite_11111111-1111-1111-1111-111111111111" / name
    case_dir.mkdir(parents=True)
    case_dir.joinpath("result.txt").write_text(
        "header\nTP xave xstd xrms yave ystd yrms vx vy\n1 0 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    case_dir.joinpath("score.json").write_text(
        json.dumps(
            {
                "Option": "ALLOW_UNKNOWN",
                "criteria0": {
                    "GT_OBJ": "car",
                    "NM": 1,
                    "TP/TN": 1,
                    "ADD": 0,
                    "AIL": 0,
                    "UIL": 0,
                    "PFN/PFP": 0,
                    "UUID_NUM": 1,
                    "MAX_DIST_THRESH": 10,
                    "OBJ_CNTS": {"car": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    if t4_dataset_id is not None:
        case_dir.joinpath("t4_metadata.json").write_text(
            json.dumps({"t4_dataset_id": t4_dataset_id, "t4_dataset_version_id": "0"}),
            encoding="utf-8",
        )
    if scenario_dataset_id is not None:
        case_dir.joinpath("scenario.yaml").write_text(
            "Evaluation:\n"
            "  Datasets:\n"
            f"  - {scenario_dataset_id}:\n"
            "      VehicleId: j6_gen2_01\n",
            encoding="utf-8",
        )


def test_score_csv_dataset_column_uses_real_dataset_id(tmp_path: Path):
    _write_case(tmp_path, "case_with_metadata_gen2_1", t4_dataset_id="dataset-from-json")
    _write_case(tmp_path, "case_with_yaml_gen2_2", scenario_dataset_id="dataset-from-yaml")
    _write_case(tmp_path, "case_without_dataset_gen2_3")

    generate_summary_and_score_csv(str(tmp_path))

    with tmp_path.joinpath("Score.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]

    by_scenario = {row[0]: row for row in rows}
    assert by_scenario["case_with_metadata_gen2_1"][1] == "dataset-from-json"
    assert by_scenario["case_with_yaml_gen2_2"][1] == "dataset-from-yaml"
    assert by_scenario["case_without_dataset_gen2_3"][1] == ""
