#!/usr/bin/env python3
"""
Tracking Degradation Analysis — CLI entry point.

Subcommands
-----------

  prepare   Split parquets by topic, then rematch GT/EST for both models.
            Produces: detection_<name>.parquet, rematched_<name>.parquet

  analyze   Build the degradation database (cause classification + truly_lost
            subclassification) from the prepared parquets.
            Produces: tracking_degrade_database.csv

  figures   Generate 10 analysis figures from the database and parquets.
            Produces: fig_*.png (10 files)

  all       Run prepare → analyze → figures in sequence.

Usage examples
--------------

  # Full pipeline (all steps)
  python script/run_tracking_degrade.py all \\
    --parquet-cp  data/current_centerpoint.parquet \\
    --parquet-tr  data/current_tracking.parquet \\
    --output-dir  data/reports/tracking_degrade

  # Same parquet file, different topic filters
  python script/run_tracking_degrade.py all \\
    --parquet-cp  data/current.parquet  --topic-cp centerpoint \\
    --parquet-tr  data/current.parquet  --topic-tr tracking \\
    --output-dir  data/reports/tracking_degrade

  # Re-generate figures only (database already exists)
  python script/run_tracking_degrade.py figures \\
    --output-dir  data/reports/tracking_degrade

  # Only prepare parquets, skip analysis
  python script/run_tracking_degrade.py prepare \\
    --parquet-cp  data/current.parquet  --topic-cp centerpoint \\
    --parquet-tr  data/current.parquet  --topic-tr tracking \\
    --output-dir  data/reports/tracking_degrade

Intermediate files are skipped if they already exist (use --force to re-run).

Default cross-class matching
----------------------------
  car:truck, car:bus, truck:bus,
  pedestrian:bicycle, pedestrian:motorbike, bicycle:motorbike,
  unknown:car, unknown:truck, unknown:bus,
  unknown:pedestrian, unknown:bicycle, unknown:motorbike
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tracking_degrade.split   import split_and_prep
from tracking_degrade.rematch import rematch_parquet
from tracking_degrade.analyze import build_degrade_db, print_summary
from tracking_degrade.figures import generate_figures


DEFAULT_CROSS_CLASS = (
    "car:truck,car:bus,truck:bus,"
    "pedestrian:bicycle,pedestrian:motorbike,bicycle:motorbike,"
    "unknown:car,unknown:truck,unknown:bus,"
    "unknown:pedestrian,unknown:bicycle,unknown:motorbike"
)


def _parse_cross_class(s: str) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    for pair in s.split(","):
        a, b = pair.strip().split(":", 1)
        a, b = a.strip(), b.strip()
        mapping.setdefault(a, {a}).add(b)
        mapping.setdefault(b, {b}).add(a)
    return {k: sorted(v) for k, v in mapping.items()}


def _sep(title: str = "") -> str:
    pad = max(0, (60 - len(title) - 2) // 2)
    return f"\n{'='*pad} {title} {'='*pad}" if title else "\n" + "="*60


# ── Step runners ──────────────────────────────────────────────────────────────

def run_prepare(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cross_class = _parse_cross_class(args.cross_class)

    for parquet, topic, name in [
        (args.parquet_cp, args.topic_cp, args.name_cp),
        (args.parquet_tr, args.topic_tr, args.name_tr),
    ]:
        det_path     = out / f"detection_{name}.parquet"
        rematch_path = out / f"rematched_{name}.parquet"

        print(_sep(f"SPLIT: {name}"))
        if det_path.exists() and not args.force:
            print(f"  [skip] {det_path.name} already exists")
        else:
            split_and_prep(parquet, str(det_path), topic, args.iou_threshold)

        print(_sep(f"REMATCH: {name}"))
        if rematch_path.exists() and not args.force:
            print(f"  [skip] {rematch_path.name} already exists")
        else:
            rematch_parquet(str(det_path), str(rematch_path), args.threshold, cross_class)


def run_analyze(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    db_path  = out / "tracking_degrade_database.csv"
    cp_path  = out / f"rematched_{args.name_cp}.parquet"
    tr_path  = out / f"rematched_{args.name_tr}.parquet"
    det_path = out / f"detection_{args.name_tr}.parquet"

    print(_sep("ANALYZE"))
    if db_path.exists() and not args.force:
        print(f"  [skip] {db_path.name} already exists")
        return

    for p in [cp_path, tr_path, det_path]:
        if not p.exists():
            print(f"[ERROR] Required file not found: {p}")
            print("  Run 'prepare' first, or pass --force to re-run.")
            sys.exit(1)

    print("Building degradation database ...")
    db = build_degrade_db(str(cp_path), str(tr_path), str(det_path))
    print_summary(db)

    db.drop(columns=["gt_key"], errors="ignore").to_csv(db_path, index=False)
    print(f"\nSaved: {db_path}  ({len(db):,} rows)")


def run_figures(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    print(_sep("FIGURES"))
    generate_figures(out, args.name_cp, args.name_tr)


# ── Shared arguments ──────────────────────────────────────────────────────────

def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name-cp", default="centerpoint",
                   help="CP model name used in file names (default: centerpoint)")
    p.add_argument("--name-tr", default="tracking",
                   help="TR model name used in file names (default: tracking)")


def _add_prepare_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--parquet-cp", required=True, help="Parquet file for CP model")
    p.add_argument("--parquet-tr", required=True, help="Parquet file for TR model")
    p.add_argument("--topic-cp",   default=None,
                   help="Topic filter substring for CP (default: --name-cp value)")
    p.add_argument("--topic-tr",   default=None,
                   help="Topic filter substring for TR (default: --name-tr value)")
    p.add_argument("--threshold",     type=float, default=2.0,
                   help="Rematch distance threshold in meters (default: 2.0)")
    p.add_argument("--cross-class",   default=DEFAULT_CROSS_CLASS,
                   help="Cross-class matching pairs (default: standard vehicle groups)")
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="IoU threshold for truck bbox merge (default: 0.3)")


def _add_output_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-dir", required=True, help="Output directory")


def _add_force_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--force", action="store_true",
                   help="Re-run steps even if output files already exist")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tracking degradation analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # ── prepare ──
    p_prep = sub.add_parser("prepare", help="Split parquets and rematch")
    _add_model_args(p_prep)
    _add_prepare_args(p_prep)
    _add_output_arg(p_prep)
    _add_force_arg(p_prep)

    # ── analyze ──
    p_ana = sub.add_parser("analyze", help="Build degradation database from prepared parquets")
    _add_model_args(p_ana)
    _add_output_arg(p_ana)
    _add_force_arg(p_ana)

    # ── figures ──
    p_fig = sub.add_parser("figures", help="Generate 10 report figures")
    _add_model_args(p_fig)
    _add_output_arg(p_fig)

    # ── all ──
    p_all = sub.add_parser("all", help="Run prepare → analyze → figures")
    _add_model_args(p_all)
    _add_prepare_args(p_all)
    _add_output_arg(p_all)
    _add_force_arg(p_all)

    args = parser.parse_args()

    # Default topic filters to model names if not specified
    if hasattr(args, "topic_cp") and args.topic_cp is None:
        args.topic_cp = args.name_cp
    if hasattr(args, "topic_tr") and args.topic_tr is None:
        args.topic_tr = args.name_tr

    if args.subcommand == "prepare":
        run_prepare(args)
    elif args.subcommand == "analyze":
        run_analyze(args)
    elif args.subcommand == "figures":
        run_figures(args)
    elif args.subcommand == "all":
        run_prepare(args)
        run_analyze(args)
        run_figures(args)

    print(_sep())
    print(f"  Output: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
