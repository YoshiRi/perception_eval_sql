from pathlib import Path

import pytest

from lib.score_schema import build_score_view, infer_score_criteria_count, read_score_csv


OLD_ROW = (
    "case_a,ALLOW_UNKNOWN,pedestrian,"
    "criteria0,10,7,1,2,3,4,5,120.000,28.7,pedestrian:7,"
    "criteria1,0,0,0,0,0,0,0,100.0,74.7,"
)

NEW_TEXT = (
    "Scenario, Dataset, Option, GT_OBJ,"
    "Distance, NM, TP/TN, ADD, AIL, UIL, PFN/PFP, UUID Num, Practical Pass Rate, MAX_DIST_THRESH,OBJ_CNTS,\n"
    "case_a,DT001,ALLOW_UNKNOWN,pedestrian,"
    "criteria0,10,7,1,2,3,4,5,120.000,28.7,pedestrian:7,\n"
)


def test_read_old_score_csv_without_header(tmp_path: Path):
    path = tmp_path / "Score.csv"
    path.write_text(OLD_ROW + "\n", encoding="utf-8")

    raw = read_score_csv(path)
    assert list(raw.columns[:3]) == ["Scenario", "Option", "GT_OBJ"]
    assert "Dataset" not in raw.columns
    assert infer_score_criteria_count(raw) == 2

    view = build_score_view(raw, 0)
    assert view.loc[0, "Scenario"] == "case_a"
    assert view.loc[0, "pass_rate"] == pytest.approx(120.0)


def test_read_new_score_csv_with_dataset_and_header(tmp_path: Path):
    path = tmp_path / "Score.csv"
    path.write_text(NEW_TEXT, encoding="utf-8")

    raw = read_score_csv(path)
    assert list(raw.columns[:4]) == ["Scenario", "Dataset", "Option", "GT_OBJ"]
    assert infer_score_criteria_count(raw) == 1

    view = build_score_view(raw, 0)
    assert view.loc[0, "Scenario"] == "case_a"
    assert view.loc[0, "Dataset"] == "DT001"
    assert view.loc[0, "pass_rate"] == pytest.approx(120.0)


def test_read_new_score_csv_tolerates_trailing_empty_column(tmp_path: Path):
    path = tmp_path / "Score.csv"
    path.write_text(NEW_TEXT.rstrip("\n").rstrip(",") + ",\n", encoding="utf-8")

    raw = read_score_csv(path)
    view = build_score_view(raw, 0)
    assert list(raw.columns[:4]) == ["Scenario", "Dataset", "Option", "GT_OBJ"]
    assert raw.shape[1] == 15
    assert view.loc[0, "pass_rate"] == pytest.approx(120.0)
