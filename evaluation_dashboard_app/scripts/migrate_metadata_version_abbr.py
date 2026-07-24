#!/usr/bin/env python3
"""One-off migration: standardize the trend version-abbreviation key on the library's name.

perception_catalog_analyzer v0.2.0 reads ``pilot_auto_version_abbr`` from ``metadata.yaml``
(the dashboard historically used its own ``version_abbr``). This script copies
``version_abbr`` -> ``pilot_auto_version_abbr`` in every ``metadata.yaml`` under the data root
that has the old key but not the new one. The legacy ``version_abbr`` key is kept for a
transitional period (dashboard readers still fall back to it); drop it in a later pass.

Usage:
    python3 scripts/migrate_metadata_version_abbr.py [--data-root PATH] [--dry-run]
    python3 scripts/migrate_metadata_version_abbr.py --drop-legacy   # also remove version_abbr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running as a plain script (python3 scripts/...) as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from lib.path_utils import get_data_root
except Exception:  # pragma: no cover - fallback when imports are unavailable
    def get_data_root() -> Path:
        return Path("data")


def _iter_metadata_files(data_root: Path):
    yield from sorted(data_root.rglob("metadata.yaml"))


def migrate_file(path: Path, *, drop_legacy: bool, dry_run: bool) -> str | None:
    """Return a human-readable action string if the file changed (or would), else None."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return f"SKIP (unreadable: {exc})"
    if not isinstance(raw, dict):
        return None

    legacy = str(raw.get("version_abbr") or "").strip()
    current = str(raw.get("pilot_auto_version_abbr") or "").strip()

    changed = False
    if legacy and not current:
        raw["pilot_auto_version_abbr"] = legacy
        current = legacy
        changed = True
    if drop_legacy and "version_abbr" in raw and current:
        del raw["version_abbr"]
        changed = True

    if not changed:
        return None
    if dry_run:
        return f"WOULD UPDATE -> pilot_auto_version_abbr={current!r}"

    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, allow_unicode=True, sort_keys=False)
    return f"UPDATED -> pilot_auto_version_abbr={current!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Data root (default: app data root)")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    parser.add_argument(
        "--drop-legacy",
        action="store_true",
        help="Also remove the legacy version_abbr key (run only after readers no longer need it)",
    )
    args = parser.parse_args()

    data_root = args.data_root or get_data_root()
    if not data_root.exists():
        print(f"Data root does not exist: {data_root}", file=sys.stderr)
        return 1

    changed = 0
    scanned = 0
    for path in _iter_metadata_files(data_root):
        scanned += 1
        action = migrate_file(path, drop_legacy=args.drop_legacy, dry_run=args.dry_run)
        if action:
            changed += 1
            print(f"{action}: {path}")

    print(f"\nScanned {scanned} metadata.yaml file(s); {changed} {'would change' if args.dry_run else 'changed'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
