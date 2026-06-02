from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any

import yaml

from lib.path_utils import get_run_display_name, path_display


RELEASE_ROLE_DIRS = ("performance", "usecase", "devops")
DEFAULT_EVALUATOR_PROJECT_ID = "x2_dev"
EVALUATOR_REPORT_BASE_URL = "https://evaluation.tier4.jp/evaluation/reports"


def _overview_query(run_path: Path) -> str:
    return urllib.parse.urlencode({"mode": "single", "run_a": get_run_display_name(run_path)})


def _safe_url_part(value: str, fallback: str) -> str:
    import re

    text = re.sub(r"[^\w.\-]+", "_", str(value or "")).strip("._")
    return text or fallback


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _role_metadata(role_dir: Path) -> dict[str, Any]:
    metadata = _load_yaml(role_dir / "metadata.yaml")
    if metadata:
        return metadata
    return _load_yaml(role_dir / "resources" / "metadata.yaml")


def _evaluator_report_url(job_id: str, project_id: str = DEFAULT_EVALUATOR_PROJECT_ID) -> str:
    if not job_id:
        return ""
    query = urllib.parse.urlencode({"project_id": project_id})
    return f"{EVALUATOR_REPORT_BASE_URL}/{job_id}?{query}"


def _pdf_static_url(release_name: str, topic_name: str) -> str:
    release_part = _safe_url_part(release_name, "release")
    topic_part = _safe_url_part(topic_name, "topic")
    return f"/app/static/release_specs/{release_part}/{topic_part}.pdf"


def discover_release_specsheet_inventory(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for release_dir in sorted(data_root.glob("release_spec_*")):
        if not release_dir.is_dir():
            continue
        metadata_path = release_dir / "metadata.yaml"
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            except Exception:
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}

        specsheet_root = release_dir / "specsheet"
        topic_pdf_paths = {
            path
            for path in specsheet_root.glob("*/*.pdf")
            if path.is_file() or path.is_symlink()
        }
        pdfs: list[dict[str, Any]] = []
        for pdf_path in sorted(specsheet_root.glob("**/*.pdf")):
            if pdf_path.parent == specsheet_root and topic_pdf_paths:
                continue
            topic = pdf_path.parent.name if pdf_path.parent != specsheet_root else "default"
            static_path = (
                Path.cwd()
                / "static"
                / "release_specs"
                / _safe_url_part(release_dir.name.replace("release_spec_", "", 1), "release")
                / f"{_safe_url_part(topic, 'topic')}.pdf"
            )
            pdfs.append(
                {
                    "topic": topic,
                    "path": pdf_path,
                    "display_path": path_display(pdf_path),
                    "absolute_path": str(pdf_path.resolve()),
                    "static_path": static_path,
                    "static_url": _pdf_static_url(release_dir.name.replace("release_spec_", "", 1), topic),
                    "available": pdf_path.exists() and not pdf_path.is_dir(),
                    "static_available": static_path.exists() and not static_path.is_dir(),
                }
            )

        roles: dict[str, dict[str, Any]] = {}
        for role in RELEASE_ROLE_DIRS:
            role_dir = release_dir / role
            if not role_dir.is_dir():
                continue
            role_metadata = _role_metadata(role_dir)
            job_id = str(role_metadata.get("job_id") or "").strip()
            project_id = str(role_metadata.get("project_id") or DEFAULT_EVALUATOR_PROJECT_ID).strip()
            roles[role] = {
                "path": role_dir,
                "display_path": path_display(role_dir),
                "absolute_path": str(role_dir.resolve()),
                "run_name": get_run_display_name(role_dir),
                "overview_query": _overview_query(role_dir),
                "overview_url": f"/?{_overview_query(role_dir)}",
                "job_id": job_id,
                "project_id": project_id,
                "evaluator_report_url": _evaluator_report_url(job_id, project_id),
                "has_parquet": any(role_dir.glob("*.parquet")),
                "has_summary": (role_dir / "summary.json").exists() or (role_dir / "resources" / "summary.json").exists(),
                "has_metadata": (role_dir / "metadata.yaml").exists() or (role_dir / "resources" / "metadata.yaml").exists(),
            }

        rows.append(
            {
                "release_dir": release_dir,
                "release_dir_display": path_display(release_dir),
                "release_dir_absolute": str(release_dir.resolve()),
                "release": release_dir.name.replace("release_spec_", "", 1),
                "version": metadata.get("pilot_auto_version") or metadata.get("version_abbr") or "",
                "date": metadata.get("date") or "",
                "description": metadata.get("description") or "",
                "data_count": metadata.get("data_count") or "",
                "roles": roles,
                "pdfs": pdfs,
                "pdf_topics": ", ".join(pdf["topic"] for pdf in pdfs),
                "main_pdf_url": next((pdf["static_url"] for pdf in pdfs), ""),
                "main_pdf_path": next((pdf["display_path"] for pdf in pdfs), ""),
            }
        )
    return rows


def discover_ready_release_specsheets(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for release in discover_release_specsheet_inventory(data_root):
        default_run = release["roles"].get("performance") or next(iter(release["roles"].values()), {})
        for pdf in release["pdfs"]:
            rows.append(
                {
                    "release_dir": release["release_dir"],
                    "pdf_path": pdf["path"],
                    "release": release["release"],
                    "version": release["version"],
                    "date": release["date"],
                    "description": release["description"],
                    "topic": pdf["topic"],
                    "view_run": default_run.get("run_name", ""),
                    "overview_query": default_run.get("overview_query", ""),
                }
            )
    return rows
