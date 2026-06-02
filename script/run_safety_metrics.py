#!/usr/bin/env python3
"""
Safety-Critical Detection Metrics – CLI entry point.

Usage examples
--------------
# Use a preset config:
  python script/run_safety_metrics.py --config configs/default.yaml

# Override individual parameters:
  python script/run_safety_metrics.py --config configs/default.yaml \\
      --label "v4.4.0_nearonly" --dist-max 60

# No config file; supply everything via flags:
  python script/run_safety_metrics.py \\
      --csv data/output/.../current.csv \\
      --output-dir data/output/safety_metrics/run_01 \\
      --classes car,truck,pedestrian \\
      --dist-max 80 \\
      --vis-exclude NONE \\
      --match-thresholds 0.5,1.0,2.0,4.0

Compare two runs
----------------
  python script/run_safety_metrics.py --config configs/default.yaml \\
      --output-dir data/output/safety_metrics/run_a
  python script/run_safety_metrics.py --config configs/near_strict.yaml \\
      --output-dir data/output/safety_metrics/run_b
  python script/run_safety_metrics.py --compare \\
      data/output/safety_metrics/run_a/summary.json \\
      data/output/safety_metrics/run_b/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root: python script/run_safety_metrics.py
sys.path.insert(0, str(Path(__file__).parent))

from safety_metrics import SafetyConfig, run_all


# ── Printing helpers ──────────────────────────────────────────────────────────

def _sep(title: str = "") -> str:
    bar = "=" * 65
    if title:
        pad = (65 - len(title) - 2) // 2
        return f"\n{'=' * pad} {title} {'=' * (65 - pad - len(title) - 2)}"
    return f"\n{bar}"


def print_results(results: dict, cfg: SafetyConfig) -> None:
    print(_sep(f"SAFETY METRICS  [{cfg.label}]"))
    print(f"  csv:      {cfg.csv_path}")
    print(f"  classes:  {cfg.classes}")
    print(f"  dist:     0 – {cfg.dist_max_m} m")
    print(f"  vis excl: {cfg.visibility_exclude}")
    print(f"  GT safe:  {results['n_gt_safe']:,}    EST safe: {results['n_est_safe']:,}")

    print(_sep("1  Safety Recall (aggregated)"))
    print(results["recall_summary"].to_string(index=False))
    print(f"\n  → Mean Safety Recall: {results['mean_safety_recall']:.4f}")

    print(_sep("2  Recall Heatmap  (class × distance bin)"))
    print(results["recall_heatmap"].to_string())

    print(_sep("3  nuScenes-style AP  (class × match threshold m)"))
    print(results["ap_pivot"].to_string())
    print()
    mAP_by_thr = results["ap_table"].groupby("match_thr_m")["ap"].mean()
    for thr, v in mAP_by_thr.items():
        print(f"  mAP @ {thr}m = {v:.4f}")
    print(f"\n  → Safety mAP (mean over classes & thresholds): {results['nds']['safety_map']:.4f}")

    print(_sep("4  TP Error Metrics"))
    print(results["tp_metrics"].to_string(index=False))
    nd = results["nds"]
    tm = nd["tp_means"]
    ts = nd["tp_scores"]
    print(f"\n  mATE  = {tm['mATE_m']:.4f} m    → score {ts['ate_score']:.4f}")
    print(f"  mAOE  = {tm['mAOE_rad']:.4f} rad → score {ts['aoe_score']:.4f}")
    print(f"  mAVE  = {tm['mAVE_mps']:.4f} m/s → score {ts['ave_score']:.4f}")

    print(_sep("5  NDS_safety"))
    n = cfg.nds_n_terms()
    print(
        f"  1/{n} × ({cfg.map_weight_in_nds}×{nd['safety_map']:.4f}"
        f" + {ts['ate_score']:.4f} + {ts['aoe_score']:.4f} + {ts['ave_score']:.4f})"
        f" = {nd['nds_safety']:.4f}"
    )
    print(_sep())


def save_results(results: dict, cfg: SafetyConfig) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results["recall_detail"].to_csv(out / "recall_by_class_dist.csv", index=False)
    results["recall_summary"].to_csv(out / "recall_summary.csv", index=False)
    results["recall_heatmap"].to_csv(out / "recall_heatmap.csv")
    results["ap_table"].to_csv(out / "ap_by_class_threshold.csv", index=False)
    results["tp_metrics"].to_csv(out / "tp_metrics_by_class.csv", index=False)

    # AP pivot as csv
    results["ap_pivot"].to_csv(out / "ap_pivot.csv")

    # Per-class mAP
    per_class_map = {
        cls: round(results["ap_table"][results["ap_table"]["class"] == cls]["ap"].mean(), 4)
        for cls in cfg.classes
    }

    summary = {
        "label": cfg.label,
        "csv_path": cfg.csv_path,
        "config": {
            "classes": cfg.classes,
            "dist_max_m": cfg.dist_max_m,
            "visibility_exclude": cfg.visibility_exclude,
            "match_thresholds_m": cfg.match_thresholds_m,
            "speed_error_clip_mps": cfg.speed_error_clip_mps,
            "ate_norm_m": cfg.ate_norm_m,
            "aoe_norm_rad": cfg.aoe_norm_rad,
            "ave_norm_mps": cfg.ave_norm_mps,
            "map_weight_in_nds": cfg.map_weight_in_nds,
            "class_weights": cfg.class_weights,
        },
        "n_gt_safe": results["n_gt_safe"],
        "n_est_safe": results["n_est_safe"],
        "mean_safety_recall": results["mean_safety_recall"],
        "recall_by_class": dict(
            zip(
                results["recall_summary"]["class"],
                results["recall_summary"]["recall"],
            )
        ),
        "safety_map": results["nds"]["safety_map"],
        "map_by_threshold": {
            str(k): round(v, 4)
            for k, v in results["ap_table"].groupby("match_thr_m")["ap"].mean().items()
        },
        "map_by_class": per_class_map,
        "tp_means": results["nds"]["tp_means"],
        "tp_scores": results["nds"]["tp_scores"],
        "nds_safety": results["nds"]["nds_safety"],
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save the config used for reproducibility
    cfg.to_yaml(out / "config_used.yaml")

    return out


# ── Compare mode ──────────────────────────────────────────────────────────────

def compare_summaries(paths: list[str]) -> None:
    summaries = []
    for p in paths:
        with open(p) as f:
            summaries.append(json.load(f))

    labels = [s["label"] for s in summaries]
    keys_scalar = ["mean_safety_recall", "safety_map", "nds_safety"]
    keys_tp = ["mATE_m", "mAOE_rad", "mAVE_mps"]

    print(_sep("COMPARISON"))
    header = f"{'metric':<28}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    print("-" * len(header))

    for k in keys_scalar:
        row = f"{k:<28}" + "".join(f"{s.get(k, float('nan')):>12.4f}" for s in summaries)
        print(row)

    print()
    for cls in summaries[0].get("recall_by_class", {}).keys():
        k = f"recall_{cls}"
        row = f"{k:<28}" + "".join(
            f"{s['recall_by_class'].get(cls, float('nan')):>12.4f}" for s in summaries
        )
        print(row)

    print()
    for k in keys_tp:
        row = f"{k:<28}" + "".join(
            f"{s['tp_means'].get(k, float('nan')):>12.4f}" for s in summaries
        )
        print(row)

    print()
    for thr in sorted({k for s in summaries for k in s.get("map_by_threshold", {})}):
        k = f"mAP@{thr}m"
        row = f"{k:<28}" + "".join(
            f"{s['map_by_threshold'].get(thr, float('nan')):>12.4f}" for s in summaries
        )
        print(row)
    print(_sep())


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute safety-critical detection metrics (NuScenes-inspired).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    SafetyConfig.add_args(parser)
    parser.add_argument(
        "--compare", nargs="+", metavar="SUMMARY_JSON",
        help="Compare two or more summary.json files instead of running eval",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")

    args = parser.parse_args()

    if args.compare:
        compare_summaries(args.compare)
        return

    cfg = SafetyConfig.from_args(args)

    print(f"Loading {cfg.csv_path} ...", flush=True)
    df = pd.read_csv(cfg.csv_path)
    print(f"  {len(df):,} rows loaded", flush=True)

    results = run_all(df, cfg)

    if not args.quiet:
        print_results(results, cfg)

    out_dir = save_results(results, cfg)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
