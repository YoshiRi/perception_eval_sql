"""The report route ships the official PDFs, so run resolution and refusals matter;
the PDF content itself belongs to the lib generators and their own behavior."""

import duckdb
import pandas as pd
import pytest

from backend import report_api


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    tier = root / "eval_run" / "performance"
    tier.mkdir(parents=True)
    con = duckdb.connect()
    con.register("src", pd.DataFrame([{"label": "car", "status": "TP"}]))
    con.execute(f"COPY src TO '{tier / 'current.parquet'}' (FORMAT PARQUET)")
    con.close()
    monkeypatch.setenv("EVAL_DASHBOARD_DATA_ROOT", str(root))
    return root


def test_dashboard_pdf_builds_for_a_parquet_only_run(data_root):
    """The generator degrades per-section on missing artifacts; a parquet-only run
    must still yield a valid PDF instead of refusing."""
    data, filename = report_api.build_report_bytes({"kind": "dashboard", "run": "eval_run"})
    assert data[:5] == b"%PDF-"
    assert filename.startswith("overview_report_")


def test_report_requests_are_validated(data_root):
    with pytest.raises(report_api.ReportError, match="Unknown kind"):
        report_api.build_report_bytes({"kind": "sideways", "run": "eval_run"})
    with pytest.raises(Exception, match="No such run"):
        report_api.build_report_bytes({"kind": "dashboard", "run": "missing"})
    with pytest.raises(report_api.ReportError, match="no 'devops' tier"):
        report_api.build_report_bytes({"kind": "dashboard", "run": "eval_run", "role": "devops"})


def test_specsheet_failure_is_a_clean_refusal(data_root):
    """A run without spec-sheet inputs must produce a typed error, not a traceback."""
    with pytest.raises(report_api.ReportError):
        report_api.build_report_bytes({"kind": "specsheet", "run": "eval_run"})
