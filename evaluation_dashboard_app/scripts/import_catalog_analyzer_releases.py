#!/usr/bin/env python3
"""Import perception_catalog_analyzer release exports into dashboard data.

This script converts release data generated directly by
perception_catalog_analyzer into the dashboard's release/trend structure.

Expected source layout:

    perception_catalog_analyzer_output/
      export/
        <job_id>/
          metadata.yaml
          current.parquet
          future.parquet
          usecase_devops.parquet
          detection.yaml
      pdf/
        <group_name>/
          <topic_name>/
            <job_id>/
              metadata.yaml
              summary.json
            specsheet/
              specsheet.pdf

Here <group_name> is usually a joined list of evaluator job IDs, for example:

    <full_job_id>_<usecase_job_id>_<devops_job_id>

Generated dashboard layout:

    data/
      release_spec_<group_name>/
        metadata.yaml
        performance/
          metadata.yaml
          resources/summary.json
          current.parquet
          future.parquet
          detection.yaml
        usecase/
          metadata.yaml
          resources/summary.json
        devops/
          metadata.yaml
          resources/summary.json
          usecase_devops.parquet
        specsheet/
          specsheet.pdf
          <topic_name>/specsheet.pdf

      trend_release_<group_name>/
        <topic_name>/
          <job_id>/
            metadata.yaml
            summary.json
          specsheet/specsheet.pdf

    static/
      release_specs/
        <group_name>/
          <topic_name>.pdf

By default, large artifacts such as parquet/PDF/HTML/PNG are symlinked to avoid
duplicating very large analyzer output. Use --copy-large-artifacts when the
original analyzer output may be removed or unavailable from the server.

Common usage:

    cd /path/to/evaluation_dashboard_app
    python scripts/import_catalog_analyzer_releases.py \\
      --source /path/to/perception_catalog_analyzer_output \\
      --data-root /path/to/dashboard/data \\
      --force

Production/server usage when source data should not remain mounted:

    python scripts/import_catalog_analyzer_releases.py \\
      --source /mnt/catalog_analyzer_output \\
      --data-root /srv/eval_dashboard/data \\
      --copy-large-artifacts \\
      --force

After import, make sure the app serves static PDFs from static/. In this app's
Docker setup, static/ is mounted into /app/static and Streamlit static serving
is enabled.

If the app directory is read-only on a server, either:

    - pass --static-root /writable/path/release_specs and mount that path as
      /app/static/release_specs, or
    - pass --skip-static-publish to import data only. PDF files are still copied
      into data/release_spec_*/specsheet and data/trend_release_*/specsheet.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ANALYZER_ROOT = Path("/home/leigu/Downloads/perception_catalog_analyzer_output")
MAIN_TOPIC = "perception.object_recognition.objects"
ROLE_DIR_BY_SUMMARY_ROLE = {
    "full": "performance",
    "usecase": "usecase",
    "devops": "devops",
    "performance_blocks": "performance",
    "unknown": "unknown",
}
DEFAULT_PROJECT_ID = "x2_dev"
SUMMARY_FULL_HEADER = "全数データセット評価"
SUMMARY_USECASE_HEADER = "ユースケース評価"
SUMMARY_USECASE_DEVOPS_HEADER = "ユースケース(過去課題)評価"
LARGE_SUFFIXES = {".parquet", ".html", ".png"}


def _usecase_devops_parquet_names() -> tuple[str, ...]:
    names: list[str] = []
    try:
        from perception_catalog_analyzer.constants import USECASE_DEVOPS_RESULT_FILENAME

        names.append(str(USECASE_DEVOPS_RESULT_FILENAME))
    except Exception:
        pass
    names.extend(["usecase_devops.parquet", "devops.parquet"])
    return tuple(dict.fromkeys(name for name in names if name))


@dataclass(frozen=True)
class ImportStats:
    releases: int = 0
    trend_jobs: int = 0
    role_runs: int = 0
    linked: int = 0
    copied: int = 0
    skipped: int = 0

    def add(self, **kwargs: int) -> "ImportStats":
        values = self.__dict__.copy()
        for key, value in kwargs.items():
            values[key] = int(values.get(key, 0)) + value
        return ImportStats(**values)


def _data_root() -> Path:
    raw = os.environ.get("EVAL_DASHBOARD_DATA_ROOT", "data")
    root = Path(raw)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _safe_path_part(value: str, fallback: str) -> str:
    import re

    text = re.sub(r"[^\w.\-]+", "_", str(value or "")).strip("._")
    return text or fallback


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _classify_summary(summary: dict[str, Any]) -> str:
    blocks = summary.get("blocks")
    if isinstance(blocks, list):
        block_items = [block for block in blocks if isinstance(block, dict)]
        headers = [str(block.get("header") or "") for block in block_items]
        evaluation_types = [str(block.get("evaluation_type") or "") for block in block_items]
        if "full" in evaluation_types or SUMMARY_FULL_HEADER in headers:
            return "full"
        if "usecase_devops" in evaluation_types or SUMMARY_USECASE_DEVOPS_HEADER in headers:
            return "devops"
        if (
            "usecase" in evaluation_types
            or "usecase_planning" in evaluation_types
            or SUMMARY_USECASE_HEADER in headers
            or "ユースケース(Planning)評価" in headers
        ):
            return "usecase"
        return "performance_blocks"
    if summary:
        return "devops"
    return "unknown"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)


def _copy_or_link(src: Path, dst: Path, *, copy_large_artifacts: bool, force: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return "skipped"
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    should_link = src.suffix.lower() in LARGE_SUFFIXES and not copy_large_artifacts
    if should_link:
        os.symlink(src.resolve(), dst)
        return "linked"
    shutil.copy2(src, dst)
    return "copied"


def _publish_static_pdf(pdf_path: Path, static_pdf_path: Path, *, force: bool) -> str:
    static_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if static_pdf_path.exists() or static_pdf_path.is_symlink():
        if not force:
            return "skipped"
        static_pdf_path.unlink()
    source = pdf_path.resolve()
    try:
        os.link(source, static_pdf_path)
    except OSError:
        shutil.copy2(source, static_pdf_path)
    return "copied"


def _artifact_stat(stats: ImportStats, action: str) -> ImportStats:
    if action == "linked":
        return stats.add(linked=1)
    if action == "copied":
        return stats.add(copied=1)
    if action == "skipped":
        return stats.add(skipped=1)
    return stats


def _merge_metadata(base: dict[str, Any], *, group_name: str, topic_name: str, job_id: str, role: str) -> dict[str, Any]:
    evaluator_info = base.get("evaluator_info") if isinstance(base.get("evaluator_info"), dict) else {}
    catalog = evaluator_info.get("catalog") if isinstance(evaluator_info.get("catalog"), dict) else {}
    source = evaluator_info.get("event", {}).get("source", {}) if isinstance(evaluator_info.get("event"), dict) else {}
    project_id = str(base.get("project_id") or DEFAULT_PROJECT_ID).strip()
    merged = {
        key: base.get(key)
        for key in (
            "tags",
            "pilot_auto_version",
            "version_abbr",
            "data_count",
            "description",
            "date",
        )
        if base.get(key) not in (None, "")
    }
    if catalog:
        merged["catalog_display_name"] = catalog.get("display_name")
        merged["catalog_id"] = catalog.get("id")
        merged["catalog_version_id"] = catalog.get("version_id")
    if isinstance(source, dict):
        for key in ("git_commit_url", "git_ref", "git_commit_date"):
            if source.get(key):
                merged[key] = source.get(key)
    merged["release_group"] = group_name
    merged["topic_name"] = topic_name
    merged["job_id"] = job_id
    merged["project_id"] = project_id
    merged["role"] = role
    merged["imported_from"] = str(ANALYZER_ROOT)
    return merged


def _copy_export_job(
    export_root: Path,
    job_id: str,
    target_dir: Path,
    *,
    group_name: str,
    topic_name: str,
    role: str,
    copy_large_artifacts: bool,
    force: bool,
    stats: ImportStats,
) -> ImportStats:
    source_dir = export_root / job_id
    if not source_dir.is_dir():
        return stats

    source_metadata = _load_yaml(source_dir / "metadata.yaml")
    metadata = _merge_metadata(
        source_metadata,
        group_name=group_name,
        topic_name=topic_name,
        job_id=job_id,
        role=role,
    )
    _write_yaml(target_dir / "metadata.yaml", metadata)
    stats = stats.add(copied=1)

    for file_name in ("current.parquet", "future.parquet", *_usecase_devops_parquet_names(), "detection.yaml"):
        src = source_dir / file_name
        if not src.exists():
            continue
        action = _copy_or_link(src, target_dir / file_name, copy_large_artifacts=copy_large_artifacts, force=force)
        stats = _artifact_stat(stats, action)
    return stats


def _copy_summary_job(
    job_dir: Path,
    target_dir: Path,
    *,
    group_name: str,
    topic_name: str,
    job_id: str,
    role: str,
    force: bool,
    stats: ImportStats,
) -> ImportStats:
    resources = target_dir / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    metadata = _load_yaml(job_dir / "metadata.yaml")
    metadata = _merge_metadata(metadata, group_name=group_name, topic_name=topic_name, job_id=job_id, role=role)
    _write_yaml(resources / "metadata.yaml", metadata)
    stats = stats.add(copied=1)

    for src, dst in (
        (job_dir / "summary.json", resources / "summary.json"),
        (job_dir / "summary.json", target_dir / "summary.json"),
    ):
        if src.exists():
            action = _copy_or_link(src, dst, copy_large_artifacts=True, force=force)
            stats = _artifact_stat(stats, action)
    return stats


def import_releases(
    analyzer_root: Path,
    data_root: Path,
    *,
    static_root: Path | None,
    copy_large_artifacts: bool,
    force: bool,
) -> ImportStats:
    export_root = analyzer_root / "export"
    pdf_root = analyzer_root / "pdf"
    stats = ImportStats()

    if not export_root.is_dir() or not pdf_root.is_dir():
        raise FileNotFoundError(f"Expected export/ and pdf/ under {analyzer_root}")

    if static_root is not None:
        try:
            static_root.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            print(
                f"Warning: cannot write static PDF directory {static_root}: {exc}. "
                "Continuing without static PDF publishing. Use --static-root with a writable path, "
                "fix directory ownership, or pass --skip-static-publish.",
                file=sys.stderr,
            )
            static_root = None

    for pdf_group_dir in sorted(path for path in pdf_root.iterdir() if path.is_dir()):
        group_name = pdf_group_dir.name
        release_dir = data_root / f"release_spec_{_safe_path_part(group_name, 'release')}"
        trend_dir = data_root / f"trend_release_{_safe_path_part(group_name, 'release')}"
        release_dir.mkdir(parents=True, exist_ok=True)
        stats = stats.add(releases=1)

        release_metadata_written = False
        for topic_dir in sorted(path for path in pdf_group_dir.iterdir() if path.is_dir()):
            topic_name = topic_dir.name
            topic_safe = _safe_path_part(topic_name, "topic")
            trend_topic_dir = trend_dir / topic_name
            trend_topic_dir.mkdir(parents=True, exist_ok=True)

            specsheet_pdf = topic_dir / "specsheet" / "specsheet.pdf"
            if specsheet_pdf.exists():
                action = _copy_or_link(
                    specsheet_pdf,
                    release_dir / "specsheet" / topic_safe / "specsheet.pdf",
                    copy_large_artifacts=copy_large_artifacts,
                    force=force,
                )
                stats = _artifact_stat(stats, action)
                action = _copy_or_link(
                    specsheet_pdf,
                    trend_topic_dir / "specsheet" / "specsheet.pdf",
                    copy_large_artifacts=copy_large_artifacts,
                    force=force,
                )
                stats = _artifact_stat(stats, action)
                if topic_name == MAIN_TOPIC:
                    action = _copy_or_link(
                        specsheet_pdf,
                        release_dir / "specsheet" / "specsheet.pdf",
                        copy_large_artifacts=copy_large_artifacts,
                        force=force,
                    )
                    stats = _artifact_stat(stats, action)
                if static_root is not None:
                    static_pdf_path = (
                        static_root
                        / _safe_path_part(group_name, "release")
                        / f"{_safe_path_part(topic_name, 'topic')}.pdf"
                    )
                    action = _publish_static_pdf(specsheet_pdf, static_pdf_path, force=force)
                    stats = _artifact_stat(stats, action)

            for job_dir in sorted(path for path in topic_dir.iterdir() if path.is_dir()):
                if job_dir.name in {"trend", "specsheet"}:
                    continue
                summary_path = job_dir / "summary.json"
                if not summary_path.exists():
                    continue
                job_id = job_dir.name
                role = _classify_summary(_load_json(summary_path))
                role_dir_name = ROLE_DIR_BY_SUMMARY_ROLE.get(role, role)

                trend_job_dir = trend_topic_dir / job_id
                trend_job_dir.mkdir(parents=True, exist_ok=True)
                for src_name in ("summary.json", "metadata.yaml"):
                    src = job_dir / src_name
                    if not src.exists():
                        continue
                    if src_name == "metadata.yaml":
                        metadata = _merge_metadata(
                            _load_yaml(src),
                            group_name=group_name,
                            topic_name=topic_name,
                            job_id=job_id,
                            role=role,
                        )
                        _write_yaml(trend_job_dir / src_name, metadata)
                        stats = stats.add(copied=1)
                    else:
                        action = _copy_or_link(src, trend_job_dir / src_name, copy_large_artifacts=True, force=force)
                        stats = _artifact_stat(stats, action)
                for parquet_name in _usecase_devops_parquet_names():
                    src = job_dir / parquet_name
                    if not src.exists():
                        continue
                    action = _copy_or_link(
                        src,
                        trend_job_dir / parquet_name,
                        copy_large_artifacts=copy_large_artifacts,
                        force=force,
                    )
                    stats = _artifact_stat(stats, action)
                stats = stats.add(trend_jobs=1)

                if topic_name != MAIN_TOPIC:
                    continue
                role_dir = release_dir / role_dir_name
                stats = _copy_export_job(
                    export_root,
                    job_id,
                    role_dir,
                    group_name=group_name,
                    topic_name=topic_name,
                    role=role,
                    copy_large_artifacts=copy_large_artifacts,
                    force=force,
                    stats=stats,
                )
                stats = _copy_summary_job(
                    job_dir,
                    role_dir,
                    group_name=group_name,
                    topic_name=topic_name,
                    job_id=job_id,
                    role=role,
                    force=force,
                    stats=stats,
                )
                stats = stats.add(role_runs=1)

                if not release_metadata_written and role in {"full", "performance_blocks"}:
                    metadata = _merge_metadata(
                        _load_yaml(job_dir / "metadata.yaml"),
                        group_name=group_name,
                        topic_name=topic_name,
                        job_id=job_id,
                        role=role,
                    )
                    _write_yaml(release_dir / "metadata.yaml", metadata)
                    release_metadata_written = True
                    stats = stats.add(copied=1)

        if not release_metadata_written:
            _write_yaml(release_dir / "metadata.yaml", {"release_group": group_name, "imported_from": str(analyzer_root)})
            stats = stats.add(copied=1)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ANALYZER_ROOT,
        help="Analyzer output root containing export/ and pdf/. Default: %(default)s",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dashboard data root. Defaults to EVAL_DASHBOARD_DATA_ROOT or ./data.",
    )
    parser.add_argument(
        "--copy-large-artifacts",
        action="store_true",
        help=(
            "Copy parquet/PDF/PNG/HTML instead of symlinking them. Use this on servers "
            "when the original analyzer output will not stay mounted."
        ),
    )
    parser.add_argument(
        "--static-root",
        type=Path,
        default=None,
        help=(
            "Directory for static PDF copies. Defaults to ./static/release_specs. "
            "Use a writable path on servers and mount it as /app/static/release_specs."
        ),
    )
    parser.add_argument(
        "--skip-static-publish",
        action="store_true",
        help="Do not write static/release_specs PDF copies. Data/specsheet PDFs are still imported.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing imported files and links.")
    args = parser.parse_args()

    data_root = args.data_root.resolve() if args.data_root is not None else _data_root()
    static_root = None
    if not args.skip_static_publish:
        static_root = (args.static_root if args.static_root is not None else Path.cwd() / "static" / "release_specs").resolve()
    stats = import_releases(
        args.source.resolve(),
        data_root,
        static_root=static_root,
        copy_large_artifacts=args.copy_large_artifacts,
        force=args.force,
    )
    print(json.dumps(stats.__dict__, indent=2, ensure_ascii=False))
    print(f"Imported analyzer releases into {data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
