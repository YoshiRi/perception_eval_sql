#!/usr/bin/env python3
"""
Distance-To-Vehicle Weighted mAP — CLI entry point.

Usage examples
--------------
# Preset config:
  python script/run_weighted_map.py --config configs/weighted_map_default.yaml

# Override CSV:
  python script/run_weighted_map.py --config configs/weighted_map_default.yaml \\
      --csv data/output/rematch/default/rematched.csv --label rematch_weighted

# No config file:
  python script/run_weighted_map.py \\
      --csv data/output/.../current.csv \\
      --output-dir data/output/weighted_map/run_01 \\
      --lambda 0.01

# Compare original vs rematched:
  python script/run_weighted_map.py --compare \\
      data/output/weighted_map/original/summary.json \\
      data/output/weighted_map/rematched/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from weighted_map import WeightedMapConfig, compute_weighted_map


# ── Printing ──────────────────────────────────────────────────────────────────

def _sep(title: str = "") -> str:
    if title:
        pad = (65 - len(title) - 2) // 2
        return f"\n{'=' * pad} {title} {'=' * (65 - pad - len(title) - 2)}"
    return "\n" + "=" * 65


def print_results(results: dict, cfg: WeightedMapConfig) -> None:
    print(_sep(f"WEIGHTED MAP  [{cfg.label}]"))
    print(f"  csv:       {cfg.csv_path}")
    print(f"  classes:   {cfg.classes}")
    print(f"  lambda:    {cfg.distance_weight_lambda}  (FOV: {cfg.forward_fov_deg}°, max: {cfg.max_distance_m}m)")
    print(f"  fp_weight: {cfg.fp_weighting}")
    print(f"  GT active: {results['n_gt']:,}  (total_gt_weight: {results['total_gt_weight']:.1f})")

    print(_sep("Class metrics"))
    cols = ["class", "n_gt", "total_gt_weight", "normal_AP", "weighted_AP", "delta_AP",
            "weighted_precision", "weighted_recall"]
    print(results["class_metrics"][cols].to_string(index=False))
    print(f"\n  normal_mAP   = {results['normal_map']:.4f}")
    print(f"  weighted_mAP = {results['weighted_map']:.4f}  (Δ {results['delta']:+.4f})")

    print(_sep("Distance-bin weighted recall"))
    pivot = results["distance_bin"].pivot_table(
        index="dist_bin", columns="class",
        values="weighted_recall",
        aggfunc="first",
    ).reindex(["near", "mid", "far", "very_far", "out_of_scope"])
    print(pivot.to_string())

    print(_sep("Top-5 high-weight FN"))
    fn = results["high_weight_fn"]
    if len(fn):
        print(fn.head(5)[["unix_time", "label", "r_val", "theta_val", "visibility", "gt_weight"]].to_string(index=False))

    print(_sep("Top-5 high-weight FP"))
    fp = results["high_weight_fp"]
    if len(fp):
        print(fp.head(5)[["unix_time", "label", "r_val", "theta_val", "confidence", "fp_weight"]].to_string(index=False))

    print(_sep())


def save_results(results: dict, cfg: WeightedMapConfig) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results["class_metrics"].to_csv(out / "class_metrics.csv", index=False)
    results["distance_bin"].to_csv(out / "distance_bin_recall.csv", index=False)
    results["high_weight_fn"].to_csv(out / "high_weight_fn.csv", index=False)
    results["high_weight_fp"].to_csv(out / "high_weight_fp.csv", index=False)

    summary = {
        "label": cfg.label,
        "csv_path": cfg.csv_path,
        "note": "drivable_surface_not_applied",
        "config": {
            "classes": cfg.classes,
            "max_distance_m": cfg.max_distance_m,
            "distance_weight_lambda": cfg.distance_weight_lambda,
            "forward_fov_deg": cfg.forward_fov_deg,
            "fp_weighting": cfg.fp_weighting,
            "visibility_exclude": cfg.visibility_exclude,
        },
        "n_gt_active": results["n_gt"],
        "total_gt_weight": results["total_gt_weight"],
        "normal_map": results["normal_map"],
        "weighted_map": results["weighted_map"],
        "delta": results["delta"],
        "class_results": results["class_metrics"].to_dict(orient="records"),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    cfg.to_yaml(out / "config_used.yaml")
    return out


# ── Compare mode ──────────────────────────────────────────────────────────────

def compare_summaries(paths: list[str]) -> None:
    summaries = []
    for p in paths:
        with open(p) as f:
            summaries.append(json.load(f))

    labels = [s["label"] for s in summaries]
    print(_sep("COMPARISON"))
    header = f"{'metric':<30}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    print("-" * len(header))

    for k in ["normal_map", "weighted_map", "delta", "total_gt_weight"]:
        row = f"{k:<30}" + "".join(f"{s.get(k, float('nan')):>14.4f}" for s in summaries)
        print(row)

    print()
    for cls in summaries[0].get("class_results", [{}]):
        c = cls.get("class", "?")
        for metric in ["normal_AP", "weighted_AP", "delta_AP", "weighted_recall"]:
            key = f"{c}_{metric}"
            row = f"{key:<30}" + "".join(
                f"{next((r[metric] for r in s.get('class_results', []) if r['class'] == c), float('nan')):>14.4f}"
                for s in summaries
            )
            print(row)
        print()
    print(_sep())


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distance-To-Vehicle weighted mAP from current.csv / rematched.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    WeightedMapConfig.add_args(parser)
    parser.add_argument(
        "--compare", nargs="+", metavar="SUMMARY_JSON",
        help="Compare two or more summary.json files",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")

    args = parser.parse_args()

    if args.compare:
        compare_summaries(args.compare)
        return

    cfg = WeightedMapConfig.from_args(args)

    print(f"Loading {cfg.csv_path} ...", flush=True)
    df = pd.read_csv(cfg.csv_path)
    print(f"  {len(df):,} rows  |  {df['unix_time'].nunique():,} timestamps")

    print("Computing weighted mAP ...", flush=True)
    results = compute_weighted_map(df, cfg)

    if not args.quiet:
        print_results(results, cfg)

    out_dir = save_results(results, cfg)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
