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


# ------------------------------------------------------------------------- prediction


def _write_prediction_cache(tier: Path) -> None:
    """A fresh page-built prediction cache next to a future.parquet source."""
    future = tier / "future.parquet"
    con = duckdb.connect()
    con.execute(f"COPY (SELECT 1 AS x) TO '{future}' (FORMAT PARQUET)")
    cache = tier / ".dashboard_cache" / "prediction_eval_cache"
    cache.mkdir(parents=True)
    label_summary = pd.DataFrame(
        [{"label": "All", "future_rows": 10, "minADE@1s": 0.5, "minFDE@5s": 2.0}]
    )
    distance_summary = pd.DataFrame(
        [{"label": "car", "metric": "minADE@1s", "r": "0-20", "value": 0.4}]
    )
    con.register("ls", label_summary)
    con.execute(f"COPY ls TO '{cache / 'label_summary.parquet'}' (FORMAT PARQUET)")
    con.register("ds", distance_summary)
    con.execute(f"COPY ds TO '{cache / 'distance_summary.parquet'}' (FORMAT PARQUET)")
    con.close()
    (cache / "manifest.json").write_text(json.dumps({
        "cache_version": 4,
        "future_mtime_ns": future.stat().st_mtime_ns,
        "table_names": ["label_summary", "distance_summary", "polar_summary"],
    }))


def test_prediction_tables_join_the_package_when_the_cache_is_fresh(data_root):
    _write_prediction_cache(data_root / "eval_base" / "performance")
    data, _ = analysis_api.build_analysis_package_bytes({"mode": "single", "run": "eval_base"})
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    names = {entry["name"] for entry in manifest["tables"]}
    assert "Prediction label summary" in names
    assert "Prediction distance summary" in names
    assert "prediction_errors" not in manifest["metadata"]


def test_a_run_without_future_data_gets_no_prediction_tables(data_root):
    data, _ = analysis_api.build_analysis_package_bytes({"mode": "single", "run": "eval_base"})
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert not any("Prediction" in entry["name"] for entry in manifest["tables"])


# -------------------------------------------------------------------------------- TLR


def _tlr_frame(name: str, tp: str, fn: str, *, final: str = "") -> str:
    frame: dict = {
        "FrameName": name,
        "criteria_0": {"PassFail": {"Info": {"TP": tp, "FP": "0 []", "FN": fn, "TN": "0 []"}}},
    }
    record: dict = {"Frame": frame}
    if final:
        frame["FinalScore"] = {"Score": {"TP": {"green": 0.9, "ALL": 0.9}}}
        record["Result"] = {"Summary": final}
    return json.dumps(record)


def _write_tlr_scenario(root: Path, suite: str, case: str, *, fn_frames: int = 1) -> None:
    scenario_dir = root / suite / case
    scenario_dir.mkdir(parents=True)
    lines = [
        _tlr_frame("0", "1 [green]", "0 []"),
        _tlr_frame("1", "1 [green]", "0 []"),
    ]
    for i in range(fn_frames):
        lines.append(_tlr_frame(str(2 + i), "0 []", "1 [green]"))
    lines.append(_tlr_frame("9", "1 [green]", "0 []",
                            final=f"criteria_0 (Success): {3}/{3 + fn_frames} -> 0.75"))
    (scenario_dir / "result.json").write_text("\n".join(lines))


@pytest.fixture()
def tlr_root(data_root, tmp_path, monkeypatch) -> Path:
    # The analyzer's derived cache writes under Path.cwd()/.cache; keep it out of the repo.
    monkeypatch.chdir(tmp_path)
    _write_tlr_scenario(data_root / "tlr_run", "suiteA", "case1", fn_frames=1)
    _write_tlr_scenario(data_root / "tlr_run", "suiteA", "case2", fn_frames=4)
    _write_tlr_scenario(data_root / "tlr_run_b", "suiteA", "case1", fn_frames=0)
    return data_root


def test_tlr_single_package_carries_criteria_and_scenario_evidence(tlr_root):
    data, filename = analysis_api.build_analysis_package_bytes(
        {"mode": "single", "kind": "tlr", "run": "tlr_run"}
    )
    assert filename == "tlr_analysis_tlr_run.zip"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        instructions = archive.read("llm_instructions.md").decode("utf-8")
    assert manifest["package_type"] == "tlr_llm_analysis"
    assert "Traffic Light Recognition" in instructions
    names = {entry["name"] for entry in manifest["tables"]}
    assert {"Criteria matrix", "Vehicle status TP rates", "Scenario summary",
            "Worst scenarios", "FN frames"} <= names
    stats = manifest["metadata"]["scope"]["stats"]
    assert stats["num_scenarios"] == 2
    assert 0 < stats["overall_tp_rate"] < 1


def test_tlr_compare_package_ranks_degraded_scenarios(tlr_root):
    data, filename = analysis_api.build_analysis_package_bytes(
        {"mode": "compare", "kind": "tlr", "base_run": "tlr_run_b",
         "candidate_run": "tlr_run"}
    )
    assert filename == "tlr_compare_tlr_run_b_vs_tlr_run.zip"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    names = {entry["name"] for entry in manifest["tables"]}
    assert {"Criteria comparison", "Scenario comparison"} <= names
    comparison = manifest["metadata"]["comparison"]
    assert comparison["base"]["path"] == "tlr_run_b"
    assert comparison["candidate"]["path"] == "tlr_run"


def test_tlr_requests_are_sandboxed_and_validated(tlr_root):
    with pytest.raises(analysis_api.AnalysisError, match="outside the data root"):
        analysis_api.build_analysis_package_bytes(
            {"mode": "single", "kind": "tlr", "run": "../../etc"}
        )
    with pytest.raises(analysis_api.AnalysisError, match="No TLR result"):
        analysis_api.build_analysis_package_bytes(
            {"mode": "single", "kind": "tlr", "run": "eval_base"}
        )


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
