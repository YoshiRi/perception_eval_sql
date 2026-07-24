"""
Path utilities for multi-user deployment: single data root and path safety.

All user-facing paths (output_path, eval_root, run directories) should be
resolved against EVAL_DASHBOARD_DATA_ROOT so that:
- Path traversal (e.g. ../../../etc) is rejected.
- Absolute paths outside the root are rejected.
- Listing and delete operations stay within the data root.
"""

import os
import re
from pathlib import Path
from typing import Optional, List, Tuple

import yaml

# Root for all evaluation data. Set EVAL_DASHBOARD_DATA_ROOT to override (e.g. /var/eval_dashboard/data).
_DATA_ROOT: Optional[Path] = None


def get_data_root() -> Path:
    """Return the canonical data root directory. Created if it does not exist."""
    global _DATA_ROOT
    if _DATA_ROOT is None:
        raw = os.environ.get("EVAL_DASHBOARD_DATA_ROOT", "data")
        p = Path(raw)
        if not p.is_absolute():
            # Resolve relative to CWD (app root when running streamlit)
            p = Path.cwd() / p
        _DATA_ROOT = p.resolve()
    return _DATA_ROOT


def get_data_root_display() -> str:
    """Return a short display path for the data root (e.g. 'data' or 'data/')."""
    root = get_data_root()
    try:
        rel = root.relative_to(Path.cwd())
        return str(rel).replace("\\", "/") or "."
    except ValueError:
        return root.name or "data"


def path_display(path: Path) -> str:
    """Return a short display path for a path under the data root (e.g. 'data/run_name')."""
    root = get_data_root()
    try:
        path.resolve().relative_to(root)
    except (ValueError, OSError):
        return path.name or str(path)
    prefix = get_data_root_display()
    try:
        rel = path.resolve().relative_to(root)
        suffix = str(rel).replace("\\", "/")
        return f"{prefix}/{suffix}" if suffix else prefix
    except (ValueError, OSError):
        return path.name or str(path)


def to_data_relative(path: str | Path) -> str:
    """
    Return the path relative to the data root for display/config.
    If the path is under the data root, returns e.g. 'download' or 'v4.3.1_test4'.
    Otherwise returns the path as a string (e.g. for invalid or legacy absolute paths).
    """
    if path is None or (isinstance(path, str) and not str(path).strip()):
        return ""
    root = get_data_root()
    try:
        p = Path(path) if not isinstance(path, Path) else path
        if not p.is_absolute():
            p = (root / p).resolve()
        else:
            p = p.resolve()
        rel = p.relative_to(root)
        return str(rel).replace("\\", "/")
    except (ValueError, OSError):
        return str(path).strip()


def resolve_under_data_root(
    user_path: str,
    allow_create: bool = False,
    allow_missing: bool = False,
) -> Tuple[Optional[Path], str]:
    """
    Resolve a user-provided path so it lies under the data root.
    Returns (resolved_path, error_message). If error_message is non-empty, resolved_path is None.

    - allow_create: if True, create the path (and parents) if missing.
    - allow_missing: if True, do not require the path to exist (e.g. for eval_root before first run).
    """
    if not user_path or not str(user_path).strip():
        return None, "Path is empty."
    root = get_data_root()
    try:
        p = Path(user_path.strip())
        if not p.is_absolute():
            # Treat relative paths as under the data root (no absolute path exposed to user)
            p = (root / p).resolve()
        else:
            p = p.resolve()
        # Ensure it is under root (use resolve() for both for consistent comparison)
        try:
            p.relative_to(root)
        except ValueError:
            return None, f"Path must be under the data root: {root}"
        if allow_create:
            p.mkdir(parents=True, exist_ok=True)
        elif not allow_missing and not p.exists():
            return None, f"Path does not exist: {p}"
        return p, ""
    except Exception as e:
        return None, str(e)


def _looks_like_analysis_run(path: Path) -> bool:
    return (
        (path / "Summary.csv").exists()
        or (path / "Score.csv").exists()
        or any(path.glob("*.parquet"))
        or (path / "current.csv").exists()
        or (path / "future.csv").exists()
    )


def _is_internal_trend_release_dir(path: Path) -> bool:
    return path.name.startswith("trend_release_")


RELEASE_ROLE_DIRS = ("performance", "usecase", "devops")
RELEASE_ROLE_LABELS = {
    "performance": "Performance",
    "usecase": "Usecase",
    "devops": "DevOps",
}
_PILOT_AUTO_PREFIX_PATTERN = re.compile(r"^\s*Pilot\.Auto\s*", re.IGNORECASE)


def _looks_like_release_container(path: Path) -> bool:
    return (
        (path / "metadata.yaml").exists()
        and any((path / name).is_dir() for name in RELEASE_ROLE_DIRS)
        and not _looks_like_analysis_run(path)
    )


def _load_yaml_metadata(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _compact_release_version(metadata: dict, fallback: str) -> str:
    version = str(
        metadata.get("pilot_auto_version_abbr")
        or metadata.get("version_abbr")
        or metadata.get("pilot_auto_version")
        or ""
    ).strip()
    if not version:
        return fallback
    version = _PILOT_AUTO_PREFIX_PATTERN.sub("", version).strip() or version
    version = version.replace("/", "-")
    return version


def _release_run_display_name(run_path: Path) -> Optional[str]:
    role_label = ""
    release_dir = run_path
    if run_path.name in RELEASE_ROLE_LABELS and _looks_like_release_container(run_path.parent):
        release_dir = run_path.parent
        role_label = RELEASE_ROLE_LABELS[run_path.name]
    elif _looks_like_release_container(run_path):
        role_label = "Release"
    else:
        return None

    metadata = _load_yaml_metadata(run_path / "metadata.yaml") or _load_yaml_metadata(release_dir / "metadata.yaml")
    version = _compact_release_version(metadata, release_dir.name.replace("release_spec_", ""))
    date = str(metadata.get("date") or "").strip()
    parts = [f"[REL] {version}"]
    if role_label:
        parts.append(role_label)
    if date:
        parts.append(date)
    return " | ".join(parts)


def get_run_display_name(run_path: Path) -> str:
    """Return a stable user-facing run selector name."""
    release_name = _release_run_display_name(run_path)
    if release_name:
        return release_name
    root = get_data_root()
    try:
        return run_path.resolve().relative_to(root).as_posix()
    except Exception:
        return run_path.name


def get_run_storage_name(run_path: Path) -> str:
    """Return the raw path-like run name relative to the data root."""
    root = get_data_root()
    try:
        return run_path.resolve().relative_to(root).as_posix()
    except Exception:
        return run_path.name


def list_run_directories() -> List[Path]:
    """Return sorted run directories, including release analysis children."""
    root = get_data_root()
    if not root.exists():
        return []
    runs: List[Path] = []
    seen = set()
    for child in sorted([p for p in root.iterdir() if p.is_dir()]):
        if _is_internal_trend_release_dir(child):
            continue
        resolved = child.resolve()
        if resolved not in seen and not _looks_like_release_container(child):
            runs.append(child)
            seen.add(resolved)
        for release_child_name in RELEASE_ROLE_DIRS:
            release_child = child / release_child_name
            if release_child.is_dir() and _looks_like_analysis_run(release_child):
                release_resolved = release_child.resolve()
                if release_resolved not in seen:
                    runs.append(release_child)
                    seen.add(release_resolved)
    return sorted(runs, key=get_run_display_name)


def count_tlr_scenarios(path: Path) -> int:
    """Count TLR scenarios in path: direct subdirs with result.json or suite subdirs with testcase/result.json."""
    if not path.exists() or not path.is_dir():
        return 0
    count = 0
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if (child / "result.json").exists():
            count += 1
        else:
            for tc in child.iterdir():
                if tc.is_dir() and (tc / "result.json").exists():
                    count += 1
    return count


def list_tlr_result_directories() -> List[Tuple[Path, int]]:
    """Return sorted list of (path, scenario_count) for direct TLR children of the data root.

    A selected top-level folder can still contain suite-style nested folders; the scenario
    count includes all result.json files in those nested suites.
    """
    root = get_data_root()
    if not root.exists():
        return []
    candidates: List[Tuple[Path, int]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        n = count_tlr_scenarios(child)
        if n > 0:
            candidates.append((child, n))
    return sorted(candidates, key=lambda x: str(x[0]))


def get_run_info(run_path: Path) -> dict:
    """Return dict with name, path, size_bytes, mtime, has_summary, has_score, has_parquet."""
    size_bytes = 0
    try:
        for entry in run_path.rglob("*"):
            if entry.is_file():
                try:
                    size_bytes += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    try:
        mtime = run_path.stat().st_mtime
    except OSError:
        mtime = 0
    has_summary = (run_path / "Summary.csv").exists()
    has_score = (run_path / "Score.csv").exists()
    has_parquet = any(run_path.glob("*.parquet"))
    return {
        "name": get_run_display_name(run_path),
        "path": run_path,
        "size_bytes": size_bytes,
        "mtime": mtime,
        "has_summary": has_summary,
        "has_score": has_score,
        "has_parquet": has_parquet,
    }


def resolve_run_subdirectory(run_name: str) -> Tuple[Optional[Path], str]:
    """
    Resolve a run directory by display name under the data root.
    Returns (path, "") on success, or (None, error_message).
    """
    root = get_data_root()
    if not run_name or run_name.strip() != run_name:
        return None, "Invalid run name."
    if "\x00" in run_name or "\\" in run_name:
        return None, "Invalid run name."
    display_matches = [path for path in list_run_directories() if get_run_display_name(path) == run_name]
    if display_matches:
        return display_matches[0], ""

    run_path = (root / run_name).resolve()
    try:
        run_path.relative_to(root)
    except ValueError:
        return None, "Run is not under data root."
    if run_path == root:
        return None, "Invalid run name."
    if not run_path.exists():
        return None, f"Run does not exist: {run_name}"
    if not run_path.is_dir():
        return None, "Not a directory."
    return run_path, ""


def delete_run(run_name: str) -> Tuple[bool, str]:
    """
    Delete a run directory by name (must be a direct child of data root).
    Returns (success, message).
    """
    run_path, err = resolve_run_subdirectory(run_name)
    if err:
        return False, err
    try:
        import shutil
        shutil.rmtree(run_path)
        return True, f"Deleted run: {run_name}"
    except Exception as e:
        return False, str(e)


def format_size(size_bytes: int) -> str:
    """Human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
