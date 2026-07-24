from pathlib import Path


def test_list_run_directories_excludes_internal_trend_release_dirs(monkeypatch, tmp_path: Path):
    from lib import path_utils

    data_root = tmp_path / "data"
    data_root.mkdir()

    normal = data_root / "normal_run"
    normal.mkdir()
    (normal / "current.parquet").write_text("", encoding="utf-8")

    trend = data_root / "trend_release_job_a_job_b_job_c"
    trend.mkdir()
    (trend / "metadata.yaml").write_text("tags: [trend]\n", encoding="utf-8")
    (trend / "current.parquet").write_text("", encoding="utf-8")

    release = data_root / "release_spec_job_a_job_b_job_c"
    performance = release / "performance"
    performance.mkdir(parents=True)
    (release / "metadata.yaml").write_text("pilot_auto_version: Pilot.Auto v4.4.0\n", encoding="utf-8")
    (performance / "current.parquet").write_text("", encoding="utf-8")

    monkeypatch.setenv("EVAL_DASHBOARD_DATA_ROOT", str(data_root))
    monkeypatch.setattr(path_utils, "_DATA_ROOT", None)

    runs = {path.relative_to(data_root).as_posix() for path in path_utils.list_run_directories()}

    assert "normal_run" in runs
    assert "release_spec_job_a_job_b_job_c/performance" in runs
    assert "trend_release_job_a_job_b_job_c" not in runs


def test_list_tlr_result_directories_lists_top_level_and_counts_nested_suites(monkeypatch, tmp_path: Path):
    from lib import path_utils

    data_root = tmp_path / "data"
    data_root.mkdir()

    parent = data_root / "eval_bundle"
    fujiyoshida = parent / "Gen2_TLR_Fujiyoshida"
    shiojiri = parent / "Gen2_TLR_Shiojiri"
    direct = data_root / "TLR_A"
    for run in (fujiyoshida, shiojiri, direct):
        scenario = run / "scenario_001"
        scenario.mkdir(parents=True)
        (scenario / "result.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("EVAL_DASHBOARD_DATA_ROOT", str(data_root))
    monkeypatch.setattr(path_utils, "_DATA_ROOT", None)

    candidates = {
        path.relative_to(data_root).as_posix(): count
        for path, count in path_utils.list_tlr_result_directories()
    }

    assert candidates == {
        "TLR_A": 1,
        "eval_bundle": 2,
    }
    assert candidates["eval_bundle"] == 2
