#!/usr/bin/env python3
"""
GT-EST Re-Matching — CLI entry point.

Usage examples
--------------
# Preset config:
  python script/run_rematch.py --config configs/rematch_default.yaml

# Override threshold:
  python script/run_rematch.py --config configs/rematch_default.yaml --threshold 1.0 --label tight_1m

# No config file:
  python script/run_rematch.py \\
      --csv data/output/.../current.csv \\
      --output-dir data/output/rematch/run_01 \\
      --threshold 2.0

# Chain with safety metrics:
  python script/run_rematch.py --config configs/rematch_default.yaml
  python script/run_safety_metrics.py --config configs/default.yaml \\
      --csv data/output/rematch/default/rematched.csv \\
      --label "rematch_default_safety"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from rematch import RematchConfig, rematch_all, compare_status


def _sep(title: str = "") -> str:
    if title:
        pad = (65 - len(title) - 2) // 2
        return f"\n{'=' * pad} {title} {'=' * (65 - pad - len(title) - 2)}"
    return "\n" + "=" * 65


def print_results(comp: dict, cfg: RematchConfig) -> None:
    print(_sep(f"REMATCH RESULTS  [{cfg.label}]"))
    print(f"  algorithm : {cfg.algorithm}")
    print(f"  label_match: {cfg.label_match}")
    print(f"  thresholds: {cfg.distance_thresholds_m}")
    print(f"  vis_exclude: {cfg.visibility_exclude_from_gt}")
    print(f"  conf_min:   {cfg.confidence_min}")

    print(_sep("GT side"))
    gt = comp["gt"]
    print(f"  {'':20s} {'TP':>8} {'FN':>8}  {'Recall':>8}")
    for side, label in [("original", "original "), ("rematched", "rematched")]:
        d = gt[side]
        total = d["TP"] + d["FN"]
        recall = d["TP"] / total if total else float("nan")
        print(f"  {label:20s} {d['TP']:>8,} {d['FN']:>8,}  {recall:>8.4f}")

    print(_sep("EST side"))
    est = comp["est"]
    print(f"  {'':20s} {'TP':>8} {'FP':>8}  {'Precision':>10}")
    for side, label in [("original", "original "), ("rematched", "rematched")]:
        d = est[side]
        total = d["TP"] + d["FP"]
        prec = d["TP"] / total if total else float("nan")
        print(f"  {label:20s} {d['TP']:>8,} {d['FP']:>8,}  {prec:>10.4f}")

    print(_sep("Recall by class"))
    hdr = f"  {'class':12s} {'gt_total':>9} {'recall_orig':>12} {'recall_new':>11} {'Δ':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in comp["recall_by_class"]:
        orig = row["recall_orig"]
        new  = row["recall_new"]
        delta = new - orig if (orig == orig and new == new) else float("nan")
        sign = "+" if delta > 0 else ""
        print(
            f"  {row['class']:12s} {row['gt_total']:>9,}"
            f" {orig:>12.4f} {new:>11.4f} {sign}{delta:>7.4f}"
        )
    print(_sep())


def save_results(
    rematched: pd.DataFrame,
    comp: dict,
    cfg: RematchConfig,
    original_path: str,
) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_out = out / "rematched.csv"
    rematched.to_csv(csv_out, index=False)
    print(f"  rematched CSV → {csv_out}  ({len(rematched):,} rows)")

    summary = {
        "label": cfg.label,
        "original_csv": original_path,
        "config": {
            "algorithm": cfg.algorithm,
            "label_match": cfg.label_match,
            "distance_thresholds_m": cfg.distance_thresholds_m,
            "visibility_exclude_from_gt": cfg.visibility_exclude_from_gt,
            "confidence_min": cfg.confidence_min,
            "label_groups": cfg.label_groups,
        },
        "comparison": comp,
    }
    with open(out / "rematch_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    cfg.to_yaml(out / "config_used.yaml")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-match GT and EST objects with configurable thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    RematchConfig.add_args(parser)
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")

    args = parser.parse_args()
    cfg = RematchConfig.from_args(args)

    print(f"Loading {cfg.csv_path} ...", flush=True)
    df = pd.read_csv(cfg.csv_path)
    print(f"  {len(df):,} rows  |  {df['unix_time'].nunique():,} timestamps")

    print("Re-matching ...", flush=True)
    rematched = rematch_all(df, cfg)

    # sort to match original row order as closely as possible
    rematched = rematched.sort_values(
        ["unix_time", "source", "label"]
    ).reset_index(drop=True)
    original_sorted = df.sort_values(
        ["unix_time", "source", "label"]
    ).reset_index(drop=True)

    comp = compare_status(original_sorted, rematched)

    if not args.quiet:
        print_results(comp, cfg)

    out_dir = save_results(rematched, comp, cfg, cfg.csv_path)
    print(f"\nSaved to {out_dir}/")
    print(
        "\nTo run safety metrics on the rematched CSV:\n"
        f"  python script/run_safety_metrics.py --config configs/default.yaml \\\n"
        f"      --csv {out_dir}/rematched.csv --label rematch_{cfg.label}"
    )


if __name__ == "__main__":
    main()
