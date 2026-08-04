"""PDF report export: the Overview page's report generators over HTTP.

Two PDFs only a Streamlit button could produce until now: the 4-section evaluation
dashboard report (Overview / TP Summary / Criteria / Detection Stats) and the release
spec-sheet. Serving them lets an agent attach the official artifacts to a ticket or a
Slack thread instead of paraphrasing them. Mounted and authorized like the export
routes; lib/ imports stay lazy for the packaged client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    from backend import export_api
except ImportError:  # pragma: no cover - frozen client puts backend/ on sys.path
    import export_api  # type: ignore[no-redef]


class ReportError(export_api.ExportError):
    """A report request that cannot be honoured as sent."""


def _tier_dir(run_name: str, role: str) -> Path:
    run_dir = export_api._resolve_run(run_name)
    candidate = run_dir / role
    if candidate.is_dir():
        return candidate
    if (run_dir / "Summary.csv").is_file() or any(run_dir.glob("*.parquet")):
        return run_dir
    raise ReportError(f"Run '{run_dir.name}' has no '{role}' tier and no top-level data.")


def _load_record(run_name: str, role: str) -> tuple[dict[str, Any], Path]:
    from lib.run_loader import load_run

    tier = _tier_dir(run_name, role)
    try:
        return load_run(tier), tier
    except FileNotFoundError as exc:
        raise ReportError(str(exc)) from exc


def build_report_bytes(payload: dict[str, Any]) -> tuple[bytes, str]:
    """Build one PDF. Returns ``(pdf_bytes, filename)``; split from the HTTP handler
    so tests and other callers can use it without a socket."""
    kind = str(payload.get("kind") or "dashboard").lower()
    role = str(payload.get("role") or "performance")
    if kind == "dashboard":
        return _build_dashboard_pdf(payload, role)
    if kind == "specsheet":
        return _build_specsheet_pdf(payload, role)
    raise ReportError(f"Unknown kind '{kind}'. Expected dashboard or specsheet.")


def _build_dashboard_pdf(payload: dict[str, Any], role: str) -> tuple[bytes, str]:
    from lib.overview_pdf_report import build_overview_pdf_report, make_report_filename

    run_name = str(payload.get("run") or "")
    candidate_name = str(payload.get("candidate_run") or "")
    if candidate_name:
        base_record, _ = _load_record(run_name, role)
        candidate_record, _ = _load_record(candidate_name, role)
        mode = "Compare Mode"
        records = [base_record, candidate_record]
        labels = ["A", "B"]
        names = [run_name, candidate_name]
    else:
        record, _ = _load_record(run_name, role)
        mode = "Single Mode"
        records = [record]
        labels = ["A"]
        names = [run_name]
    pdf = build_overview_pdf_report(
        mode=mode, run_records=records, run_labels=labels,
        filters={
            "perception_labels": list(payload.get("perception_labels") or []),
            "product_labels": list(payload.get("product_labels") or []),
        },
    )
    return pdf, make_report_filename(names)


def _build_specsheet_pdf(payload: dict[str, Any], role: str) -> tuple[bytes, str]:
    from lib.specsheet_report import generate_specsheet_pdf

    run_name = str(payload.get("run") or "")
    tier = _tier_dir(run_name, role)
    try:
        pdf_path, _generated = generate_specsheet_pdf(
            tier,
            project_id=str(payload.get("project_id") or "x2_dev"),
            version=str(payload.get("version") or run_name),
            labels=list(payload.get("labels") or []),
            force=payload.get("force") is True,
            is_exclude_polygons=payload.get("exclude_polygons") is True,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise ReportError(f"Spec-sheet generation failed: {exc}") from exc
    return Path(pdf_path).read_bytes(), f"specsheet_{export_api._resolve_run(run_name).name}.pdf"


def report(handler: Any, payload: dict[str, Any], *, head_only: bool = False) -> None:
    """Stream one PDF report (kind: dashboard or specsheet)."""
    export_api.require_auth(handler)
    data, filename = build_report_bytes(payload)
    handler.send_response(200)
    handler.send_header("Content-Type", "application/pdf")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    handler._export_stream_started = True
    handler.end_headers()
    if not head_only:
        handler.wfile.write(data)


STREAM_ROUTES: dict[str, Callable[..., None]] = {
    "/api/report": report,
}
