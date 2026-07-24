from pathlib import Path

from worker import tasks


def test_release_artifact_download_forces_large_file_skip(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run_download_results(**kwargs):
        calls.append(kwargs)
        return 0, 1, [{"Scenario Name": "case_a", "Status": "success"}]

    monkeypatch.setattr(tasks, "_import_eval_summary", lambda: None)
    monkeypatch.setattr(tasks, "_import_catalog_io", lambda: None)
    monkeypatch.setattr(tasks, "append_task_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks, "update_task_progress", lambda *args, **kwargs: None)

    from lib import download_core

    monkeypatch.setattr(download_core, "run_download_results", fake_run_download_results)

    result = tasks._build_release_analysis_artifacts(
        task_id="task-1",
        project_id="x2_dev",
        job_id="job-1",
        role="devops",
        output_path=tmp_path,
        phase="perception.object_recognition.tracking.objects",
        run_eval=False,
        skip_large_file=False,
        large_file_mb=50.0,
    )

    assert calls
    assert calls[0]["skip_large_file"] is True
    assert calls[0]["large_file_mb"] == 50.0
    assert result["download"]["skip_large_file"] is True
    assert result["download"]["large_file_mb"] == 50.0
