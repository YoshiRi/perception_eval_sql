import io
import json
import zipfile

import duckdb
import pandas as pd
import pytest

from lib.detection_llm_package import (
    build_llm_analysis_package,
    build_single_llm_analysis_tables,
    safe_export_filename,
)


@pytest.fixture()
def con():
    """In-memory DuckDB with a small table shaped like the page's view_eval_flat."""
    con = duckdb.connect()
    df = pd.DataFrame(
        [
            # t4dataset_id, topic, label, distance_bin, bin_idx, visibility, suite, scenario, t4name, frame, source, status, x_err, y_err, yaw_err, dist_h, pc_num, uuid
            ("ds-1", "topic", "car", "[0,10)", 10, "FULL", "suite-1", "scn-1", "t4-1", 3, "GT", "TP", 0.1, 0.2, 0.01, 5.0, 100, "gt-1"),
            ("ds-1", "topic", "car", "[0,10)", 10, "FULL", "suite-1", "scn-1", "t4-1", 4, "GT", "FN", None, None, None, 6.0, 80, "gt-2"),
            ("ds-1", "topic", "car", "[0,10)", 10, "FULL", "suite-1", "scn-1", "t4-1", 3, "EST", "TP", 0.1, 0.2, 0.01, 5.0, 100, "est-1"),
            ("ds-1", "topic", "car", "[0,10)", 10, "FULL", "suite-1", "scn-1", "t4-1", 3, "EST", "FP", None, None, None, 7.0, 30, "est-2"),
            ("ds-1", "topic", "pedestrian", "[10,20)", 20, "MOST", "suite-1", "scn-1", "t4-1", 5, "GT", "TP", 0.3, 0.1, 0.02, 15.0, 40, "gt-3"),
            ("ds-1", "topic", "pedestrian", "[10,20)", 20, "MOST", "suite-1", "scn-1", "t4-1", 5, "EST", "TP", 0.3, 0.1, 0.02, 15.0, 40, "est-3"),
        ],
        columns=[
            "t4dataset_id",
            "topic_name",
            "label",
            "distance_bin",
            "bin_idx",
            "visibility",
            "suite_name",
            "scenario_name",
            "t4dataset_name",
            "frame_index",
            "source",
            "status",
            "x_error",
            "y_error",
            "yaw_error",
            "dist_h",
            "pointcloud_num",
            "uuid",
        ],
    )
    con.register("eval_flat_src", df)
    con.execute("CREATE TABLE eval_flat AS SELECT * FROM eval_flat_src")
    return con


def test_build_single_llm_analysis_tables(con):
    tables = build_single_llm_analysis_tables(con, view="eval_flat", filter_clause="1=1")

    assert set(tables) == {
        "Class metrics",
        "Scene hotspots",
        "FN frames",
        "Distance rates",
        "Mean error by class",
    }

    cls = tables["Class metrics"].set_index("label")
    assert cls.loc["car", "gt_total"] == 2
    assert cls.loc["car", "tp"] == 1
    assert cls.loc["car", "fn"] == 1
    assert cls.loc["car", "fp"] == 1
    assert cls.loc["car", "tpr"] == pytest.approx(0.5)
    assert cls.loc["car", "fpr"] == pytest.approx(0.5)
    assert cls.loc["car", "precision"] == pytest.approx(0.5)
    assert cls.loc["pedestrian", "tpr"] == pytest.approx(1.0)

    fn_frames = tables["FN frames"]
    assert len(fn_frames) == 1
    assert fn_frames.iloc[0]["frame_index"] == "4"
    assert fn_frames.iloc[0]["fn"] == 1

    dist = tables["Distance rates"].set_index("distance_bin")
    assert dist.loc["[0,10)", "tpr"] == pytest.approx(0.5)
    assert dist.loc["[0,10)", "fpr"] == pytest.approx(0.5)
    assert dist.loc["[10,20)", "tpr"] == pytest.approx(1.0)

    err = tables["Mean error by class"].set_index("label")
    assert err.loc["car", "mean_abs_x_error"] == pytest.approx(0.1)

    scenes = tables["Scene hotspots"]
    assert scenes.iloc[0]["scenario_name"] == "scn-1"
    assert scenes.iloc[0]["fn"] == 1


def test_build_llm_analysis_package_zip(con):
    tables = build_single_llm_analysis_tables(con, view="eval_flat", filter_clause="1=1")
    metadata = {
        "mode": "Detection Stats",
        "comparison": "Single run A",
        "scope": "all distances",
        "filters": {"label": ["car", "pedestrian"]},
        "kpis": {"A": {"tp": 2, "fp": 1, "fn": 1}},
    }
    data = build_llm_analysis_package(tables=tables, metadata=metadata)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        for required in (
            "README.md",
            "llm_instructions.md",
            "analysis_data_brief.md",
            "recommended_report_blueprint.md",
            "manifest.json",
        ):
            assert required in names

        csv_names = {n for n in names if n.startswith("tables/") and n.endswith(".csv")}
        assert len(csv_names) == len(tables)

        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["package_type"] == "detection_stats_llm_analysis"
        assert manifest["metadata"]["comparison"] == "Single run A"
        assert len(manifest["tables"]) == len(tables)
        for entry in manifest["tables"]:
            df = tables[entry["name"]]
            assert entry["file"] in names
            assert entry["rows"] == len(df)
            assert entry["columns"] == [str(c) for c in df.columns]
            csv_df = pd.read_csv(io.BytesIO(zf.read(entry["file"])))
            assert len(csv_df) == entry["rows"]

        brief = zf.read("analysis_data_brief.md").decode("utf-8")
        assert "Detection Analysis Data Brief" in brief
        assert "Single run A" in brief


def test_safe_export_filename_sanitizes():
    assert safe_export_filename("Class metrics") == "Class_metrics"
    assert safe_export_filename("a/b:c*?.csv") == "a_b_c_.csv"
    assert safe_export_filename("  ") == "table"
    assert safe_export_filename(None) == "table"
    assert safe_export_filename("", fallback="fb") == "fb"
    assert len(safe_export_filename("x" * 200)) == 80
