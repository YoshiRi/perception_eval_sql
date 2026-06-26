from __future__ import annotations

import os
import shutil
import urllib.parse
from pathlib import Path
from typing import Any

import yaml

from lib.path_utils import get_run_display_name, path_display
from lib.run_metadata import read_run_metadata


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


def _static_pdf_path(release_name: str, topic_name: str) -> Path:
    return (
        Path.cwd()
        / "static"
        / "release_specs"
        / _safe_url_part(release_name, "release")
        / f"{_safe_url_part(topic_name, 'topic')}.pdf"
    )


def publish_static_release_pdf(
    pdf_path: Path,
    release_name: str,
    topic_name: str,
    *,
    force: bool = False,
) -> Path | None:
    """Publish a release specsheet PDF under static/release_specs for Streamlit static serving."""
    if not pdf_path.exists() or pdf_path.is_dir():
        return None

    static_pdf_path = _static_pdf_path(release_name, topic_name)
    source = pdf_path.resolve()
    if static_pdf_path.exists() or static_pdf_path.is_symlink():
        if not force:
            try:
                if static_pdf_path.resolve() == source:
                    return static_pdf_path
                if static_pdf_path.stat().st_mtime >= source.stat().st_mtime:
                    return static_pdf_path
            except OSError:
                pass
        if static_pdf_path.is_dir():
            shutil.rmtree(static_pdf_path)
        else:
            static_pdf_path.unlink(missing_ok=True)

    static_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, static_pdf_path)
    except OSError:
        try:
            os.symlink(source, static_pdf_path)
        except OSError:
            shutil.copy2(source, static_pdf_path)
    return static_pdf_path


def _release_version_label(metadata: dict[str, Any]) -> str:
    version_abbr = str(metadata.get("version_abbr") or "").strip()
    pilot_auto_version = str(metadata.get("pilot_auto_version") or "").strip()
    if version_abbr:
        return version_abbr
    return pilot_auto_version


def _looks_like_workflow_release_container(path: Path) -> bool:
    return (
        (path / "metadata.yaml").exists()
        and any((path / role).is_dir() for role in RELEASE_ROLE_DIRS)
    )


def _is_internal_release_dir(name: str) -> bool:
    return name.startswith(("release_spec_", "trend_release_"))


def _discover_workflow_release_dirs(data_root: Path) -> list[Path]:
    releases: list[Path] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or _is_internal_release_dir(child.name):
            continue
        if _looks_like_workflow_release_container(child):
            releases.append(child)
    return releases


def _release_dedup_key(row: dict[str, Any]) -> tuple[str, str, str]:
    release_dir = row["release_dir"]
    metadata = _load_yaml(release_dir / "metadata.yaml")
    group = str(metadata.get("release_group") or row.get("release") or release_dir.name).strip()
    return (
        group,
        str(row.get("version") or ""),
        str(row.get("date") or ""),
    )


def _prefer_release_row(candidate: dict[str, Any], existing: dict[str, Any]) -> bool:
    candidate_legacy = str(candidate["release_dir"].name).startswith("release_spec_")
    existing_legacy = str(existing["release_dir"].name).startswith("release_spec_")
    if candidate_legacy != existing_legacy:
        return not candidate_legacy
    return False


def _job_id_for_role(release_dir: Path, role_dir: Path, role: str) -> str:
    role_metadata = _role_metadata(role_dir)
    job_id = str(role_metadata.get("job_id") or "").strip()
    if job_id:
        return job_id

    release_metadata = read_run_metadata(release_dir)
    release_specsheet = (
        release_metadata.get("release_specsheet")
        if isinstance(release_metadata.get("release_specsheet"), dict)
        else {}
    )
    evaluator_jobs = (
        release_specsheet.get("evaluator_jobs")
        if isinstance(release_specsheet.get("evaluator_jobs"), dict)
        else {}
    )
    role_meta = evaluator_jobs.get(role) if isinstance(evaluator_jobs.get(role), dict) else {}
    return str(role_meta.get("job_id") or "").strip()


def _project_id_for_role(release_dir: Path, role_dir: Path, role: str) -> str:
    role_metadata = _role_metadata(role_dir)
    project_id = str(role_metadata.get("project_id") or "").strip()
    if project_id:
        return project_id

    release_metadata = read_run_metadata(release_dir)
    release_specsheet = (
        release_metadata.get("release_specsheet")
        if isinstance(release_metadata.get("release_specsheet"), dict)
        else {}
    )
    evaluator_jobs = (
        release_specsheet.get("evaluator_jobs")
        if isinstance(release_specsheet.get("evaluator_jobs"), dict)
        else {}
    )
    role_meta = evaluator_jobs.get(role) if isinstance(evaluator_jobs.get(role), dict) else {}
    return str(role_meta.get("project_id") or DEFAULT_EVALUATOR_PROJECT_ID).strip() or DEFAULT_EVALUATOR_PROJECT_ID


def _pdf_entry(
    *,
    pdf_path: Path,
    topic: str,
    release_name: str,
    link_url: str = "",
) -> dict[str, Any]:
    published_path = publish_static_release_pdf(pdf_path, release_name, topic)
    static_path = published_path or _static_pdf_path(release_name, topic)
    static_available = static_path.exists() and not static_path.is_dir()
    static_url = _pdf_static_url(release_name, topic) if static_available else ""
    return {
        "topic": topic,
        "path": pdf_path,
        "display_path": path_display(pdf_path),
        "absolute_path": str(pdf_path.resolve()),
        "static_path": static_path,
        "static_url": static_url,
        "link_url": static_url or link_url,
        "available": pdf_path.exists() and not pdf_path.is_dir(),
        "static_available": static_available,
    }


def _collect_specsheet_pdfs(
    release_dir: Path,
    release_name: str,
    metadata: dict[str, Any],
    roles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pdfs: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    default_topic = str(metadata.get("topic_name") or "perception.object_recognition.objects").strip()
    performance_overview = str(roles.get("performance", {}).get("overview_url") or "")

    def add_pdf(pdf_path: Path, topic: str, *, link_url: str = "") -> None:
        if not pdf_path.exists() or pdf_path.is_dir():
            return
        absolute_path = str(pdf_path.resolve())
        if absolute_path in seen_paths:
            return
        seen_paths.add(absolute_path)
        pdfs.append(
            _pdf_entry(
                pdf_path=pdf_path,
                topic=topic,
                release_name=release_name,
                link_url=link_url or performance_overview,
            )
        )

    specsheet_root = release_dir / "specsheet"
    topic_pdf_paths = {
        path
        for path in specsheet_root.glob("*/*.pdf")
        if path.is_file() or path.is_symlink()
    }
    for pdf_path in sorted(specsheet_root.glob("**/*.pdf")):
        if pdf_path.parent == specsheet_root and topic_pdf_paths:
            continue
        topic = pdf_path.parent.name if pdf_path.parent != specsheet_root else "default"
        add_pdf(pdf_path, topic)

    for role in RELEASE_ROLE_DIRS:
        role_pdf = release_dir / role / "specsheet" / "specsheet.pdf"
        if not role_pdf.exists():
            continue
        topic = default_topic if role == "performance" else f"{default_topic}.{role}"
        add_pdf(role_pdf, topic)

    return pdfs


def _build_role_entries(release_dir: Path) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    for role in RELEASE_ROLE_DIRS:
        role_dir = release_dir / role
        if not role_dir.is_dir():
            continue
        job_id = _job_id_for_role(release_dir, role_dir, role)
        project_id = _project_id_for_role(release_dir, role_dir, role)
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
    return roles


def _build_release_inventory_row(release_dir: Path, *, release_name: str) -> dict[str, Any] | None:
    if not release_dir.is_dir():
        return None

    metadata = _load_yaml(release_dir / "metadata.yaml")
    roles = _build_role_entries(release_dir)
    if not roles and not metadata:
        return None

    pdfs = _collect_specsheet_pdfs(release_dir, release_name, metadata, roles)
    return {
        "release_dir": release_dir,
        "release_dir_display": path_display(release_dir),
        "release_dir_absolute": str(release_dir.resolve()),
        "release": release_name,
        "source_kind": "imported" if release_dir.name.startswith("release_spec_") else "workflow",
        "version": _release_version_label(metadata),
        "pilot_auto_version": str(metadata.get("pilot_auto_version") or "").strip(),
        "version_abbr": str(metadata.get("version_abbr") or "").strip(),
        "date": metadata.get("date") or "",
        "description": metadata.get("description") or "",
        "data_count": metadata.get("data_count") or "",
        "roles": roles,
        "pdfs": pdfs,
        "pdf_topics": ", ".join(pdf["topic"] for pdf in pdfs),
        "main_pdf_url": next((pdf["link_url"] or pdf["static_url"] for pdf in pdfs if pdf.get("link_url") or pdf.get("static_url")), ""),
        "main_pdf_path": next((pdf["display_path"] for pdf in pdfs), ""),
    }


def discover_release_specsheet_inventory(data_root: Path) -> list[dict[str, Any]]:
    candidates: list[tuple[Path, str]] = []
    for release_dir in sorted(data_root.glob("release_spec_*")):
        if release_dir.is_dir():
            candidates.append((release_dir, release_dir.name.replace("release_spec_", "", 1)))
    for release_dir in _discover_workflow_release_dirs(data_root):
        candidates.append((release_dir, release_dir.name))

    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for release_dir, release_name in candidates:
        row = _build_release_inventory_row(release_dir, release_name=release_name)
        if row is None:
            continue
        key = _release_dedup_key(row)
        existing = rows_by_key.get(key)
        if existing is None or _prefer_release_row(row, existing):
            rows_by_key[key] = row

    return list(rows_by_key.values())


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
