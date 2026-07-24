from urllib.parse import parse_qs, urlparse

import pandas as pd

from lib.t4_dataset_embed import t4_dashboard_query_params, t4_dashboard_url


def test_t4_dashboard_query_params_include_run_and_viewer_context():
    query = t4_dashboard_query_params(
        mode="Compare Mode",
        run_names=["baseline run", "candidate/run"],
        suite_name="suite A",
        scenario_name="scenario:1",
        t4dataset_name="dataset-1",
        frame_index=42,
        compare_view_mode="side_by_side",
    )

    params = parse_qs(query)

    assert params["mode"] == ["compare"]
    assert params["run_a"] == ["baseline run"]
    assert params["run_b"] == ["candidate/run"]
    assert params["viewer_suite"] == ["suite A"]
    assert params["viewer_scenario"] == ["scenario:1"]
    assert params["viewer_t4dataset"] == ["dataset-1"]
    assert params["viewer_frame"] == ["42"]
    assert params["viewer_compare"] == ["side_by_side"]


def test_t4_dashboard_url_defaults_to_3d_viewer_page():
    url = t4_dashboard_url(
        mode="Single Mode",
        run_names=["run-a"],
        scenario_name="scene",
        t4dataset_id="dataset-id",
        frame_index=0,
    )

    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert parsed.path == "/T4_3D_Viewer"
    assert params["mode"] == ["single"]
    assert params["run_a"] == ["run-a"]
    assert params["viewer_scenario"] == ["scene"]
    assert params["viewer_t4dataset"] == ["dataset-id"]
    assert params["viewer_frame"] == ["0"]


def test_t4_dashboard_query_params_fall_back_to_dataset_id_for_na_name():
    query = t4_dashboard_query_params(
        mode="Single Mode",
        run_names=["run-a"],
        t4dataset_name=pd.NA,
        t4dataset_id="dataset-id",
        frame_index=7,
    )

    params = parse_qs(query)

    assert params["viewer_t4dataset"] == ["dataset-id"]
    assert params["viewer_frame"] == ["7"]


def test_t4_dashboard_query_params_ignore_placeholder_dataset_name():
    query = t4_dashboard_query_params(
        mode="Compare Mode",
        run_names=["baseline", "candidate"],
        t4dataset_name="00000000-0000-0000-0000-000000000001",
        t4dataset_id="a44bdc69-2404-4081-99a6-631a781ea186",
    )

    params = parse_qs(query)

    assert params["viewer_t4dataset"] == ["a44bdc69-2404-4081-99a6-631a781ea186"]
