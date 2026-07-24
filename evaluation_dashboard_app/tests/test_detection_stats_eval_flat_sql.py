import ast
from pathlib import Path

import duckdb
import pandas as pd


def _load_eval_flat_select_sql():
    """Load the SQL helper without executing the Streamlit page body."""
    source = Path("pages/3_Detection_Stats.py").read_text()
    module = ast.parse(source)
    func = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "eval_flat_select_sql"
    )
    namespace = {"SKIP_FIRST_N_FRAMES": 3, "SKIP_LAST_N_FRAMES": 1}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "eval_flat_select_sql", "exec"), namespace)
    return namespace["eval_flat_select_sql"]


def test_eval_flat_exclude_polygons_matches_analyzer_semantics(tmp_path):
    df = pd.DataFrame(
        {
            "frame_index": [3, 3, 3, 3, 3, 4, 4],
            "t4dataset_id": ["dataset-a"] * 7,
            "source": ["EST", "GT", "EST", "GT", "EST", "GT", "EST"],
            "status": ["TP", "TP", "FP", "FN", "TP", "TP", "TP"],
            "shape_type": [
                "polygon",
                "bounding_box",
                "polygon",
                "bounding_box",
                "bounding_box",
                "bounding_box",
                "bounding_box",
            ],
            "uuid": [
                "est-poly-tp",
                "gt-1",
                "est-poly-fp",
                "gt-2",
                "est-box",
                "gt-last",
                "est-last",
            ],
            "pair_uuid": ["gt-1", "est-poly-tp", None, None, "gt-other", "est-last", "gt-last"],
            "x_error": [0.1, 0.1, None, None, 0.2, 0.3, 0.3],
            "y_error": [0.0, 0.0, None, None, 0.0, 0.0, 0.0],
            "yaw_error": [0.0, 0.0, None, None, 0.0, 0.0, 0.0],
            "speed_error": [0.0, 0.0, None, None, 0.0, 0.0, 0.0],
            "plane_distance": [0.0, 0.0, None, None, 0.0, 0.0, 0.0],
            "pair_dt_sec": [0.0, 0.0, None, None, 0.0, 0.0, 0.0],
            "label": ["car"] * 7,
            "topic_name": ["perception.object_recognition.objects"] * 7,
            "x": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0],
            "y": [0.0] * 7,
        }
    )
    parquet_path = tmp_path / "current.parquet"
    df.to_parquet(parquet_path)

    eval_flat_select_sql = _load_eval_flat_select_sql()
    con = duckdb.connect()
    result = con.execute(
        eval_flat_select_sql(
            str(parquet_path),
            has_frame_index=True,
            has_t4dataset_id=True,
            exclude_polygons=True,
        )
    ).df()

    assert set(result["uuid"]) == {"gt-1", "gt-2", "est-box"}
    gt1 = result.loc[result["uuid"] == "gt-1"].iloc[0]
    assert gt1["source"] == "GT"
    assert gt1["status"] == "FN"
    assert pd.isna(gt1["x_error"])
    assert pd.isna(gt1["pair_uuid"])
    assert result.loc[result["uuid"] == "est-box", "status"].iloc[0] == "TP"
