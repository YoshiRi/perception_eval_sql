"""The analysis-package route hands agents the same evidence ZIP the Detection Stats
page builds, so run resolution and package content are what matter here."""

import io
import json
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from backend import analysis_api

_COLUMNS = [
    "t4dataset_id", "topic_name", "label", "distance_bin", "bin_idx", "visibility",
    "suite_name", "scenario_name", "t4dataset_name", "frame_index", "source", "status",
    "x_error", "y_error", "yaw_error", "dist_h", "pointcloud_num", "uuid",
]

_ROWS = [
    ("ds-1", "topic", "car", "[0,10)", 10, "FULL", "suite-1", "scn-1", "t4-1", 3, "GT", "TP", 0.1, 0.2, 0.01, 5.0, 100, "gt-1"),
    ("ds-1", "topic", "car", "[0,10)", 10, "FULL", "suite-1", "scn-1", "t4-1", 4, "GT", "FN", None, None, None, 6.0, 80, "gt-2"),
    ("ds-1", "topic", "car", "[0,10)", 10, "FULL", "suite-1", "scn-1", "t4-1", 3, "EST", "TP", 0.1, 0.2, 0.01, 5.0, 100, "est-1"),
    ("ds-1", "topic", "car", "[0,10)", 10, "FULL", "suite-1", "scn-1", "t4-1", 3, "EST", "FP", None, None, None, 7.0, 30, "est-2"),
]


def _write_run(root: Path, name: str, role: str = "performance") -> None:
    tier = root / name / role
    tier.mkdir(parents=True)
    frame = pd.DataFrame(_ROWS, columns=_COLUMNS)
    con = duckdb.connect()
    con.register("src", frame)
    # The *_eval_flat.parquet name marks it as an already-flattened cache, the same
    # shape the page feeds create_view_eval_flat in production.
    con.execute(
        f"COPY src TO '{tier / 'current_eval_flat.parquet'}' (FORMAT PARQUET)"
    )
    con.close()


@pytest.fixture()
def data_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("EVAL_DASHBOARD_DATA_ROOT", str(root))
    _write_run(root, "eval_base")
    _write_run(root, "eval_candidate")
    return root


def test_single_package_matches_the_pages_zip_layout(data_root):
    data, filename = analysis_api.build_analysis_package_bytes(
        {"mode": "single", "run": "eval_base"}
    )
    assert filename == "analysis_eval_base.zip"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        for required in ("README.md", "llm_instructions.md", "analysis_data_brief.md",
                         "recommended_report_blueprint.md", "manifest.json"):
            assert required in names
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["package_type"] == "detection_stats_llm_analysis"
    assert manifest["metadata"]["mode"] == "single"
    assert manifest["metadata"]["scope"]["run"] == "eval_base"
    kpis = manifest["metadata"]["kpis"]["eval_base"]
    assert kpis["tp"] == 1 and kpis["fn"] == 1 and kpis["fp"] == 1
    assert {entry["name"] for entry in manifest["tables"]} >= {
        "Class metrics", "Scene hotspots", "FN frames", "Distance rates",
    }


def test_compare_package_carries_both_runs(data_root):
    data, filename = analysis_api.build_analysis_package_bytes(
        {"mode": "compare", "base_run": "eval_base", "candidate_run": "eval_candidate"}
    )
    assert filename == "compare_eval_base_vs_eval_candidate.zip"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    comparison = manifest["metadata"]["comparison"]
    assert comparison["base"]["run"] == "eval_base"
    assert comparison["candidate"]["run"] == "eval_candidate"
    assert "eval_candidate" in manifest["metadata"]["kpis"]
    assert any("diff" in entry["name"].lower() or "comparison" in entry["name"].lower()
               for entry in manifest["tables"])


def test_filters_are_validated_and_applied(data_root):
    with pytest.raises(analysis_api.AnalysisError, match="Unknown filter"):
        analysis_api.build_analysis_package_bytes(
            {"mode": "single", "run": "eval_base", "filters": {"nope": 1}}
        )
    data, _ = analysis_api.build_analysis_package_bytes(
        {"mode": "single", "run": "eval_base", "filters": {"label": ["pedestrian"]}}
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    # Every row is 'car', so a pedestrian-only filter leaves no ground truth.
    assert not manifest["metadata"]["kpis"]["eval_base"].get("gt")


def test_requests_that_cannot_be_honoured_are_refused(data_root):
    with pytest.raises(analysis_api.AnalysisError, match="Unknown mode"):
        analysis_api.build_analysis_package_bytes({"mode": "sideways", "run": "eval_base"})
    with pytest.raises(Exception, match="No such run"):
        analysis_api.build_analysis_package_bytes({"mode": "single", "run": "missing"})
    with pytest.raises(analysis_api.AnalysisError, match="no 'devops' tier"):
        analysis_api.build_analysis_package_bytes(
            {"mode": "single", "run": "eval_base", "role": "devops"}
        )
    with pytest.raises(analysis_api.AnalysisError, match="same run"):
        analysis_api.build_analysis_package_bytes(
            {"mode": "compare", "base_run": "eval_base", "candidate_run": "eval_base"}
        )
