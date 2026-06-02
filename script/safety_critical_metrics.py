"""
Safety-Critical Detection Metrics (NuScenes-inspired)
=====================================================
nuScenes Detection Score (NDS) アプローチを Safety-Critical 評価に適用する。

変更点（標準 NuScenes との差分）:
  - GT フィルタ: visibility=NONE を除外（完全遮蔽物体はFN扱いしない）
  - クラス: safety-critical 4クラス (car, truck, bus, pedestrian)
  - 距離: 0-100m 範囲のみ評価
  - matching threshold: nuScenes 同様 0.5 / 1.0 / 2.0 / 4.0 m (center distance)
  - TP metrics: ATE, AOE, AVE (ASE/AAEは列なし)
  - NDS_safety = 1/8 * (5*mAP + Σ(1 - min(TP_i, 1.0)) for 3 TP metrics)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

CSV_PATH = Path(
    "/Users/yoshiri/Documents/OSS/perception_eval_sql/data/output/pdf/"
    "e2fb9ecc-decf-513f-b54f-2a493b2877ad_8e477a87-8e2f-5b15-9618-4597431a85eb_b2fd4c93-1ba3-5c8f-be8f-984e65827bfa/"
    "perception.object_recognition.objects/"
    "e2fb9ecc-decf-513f-b54f-2a493b2877ad/current.csv"
)

OUTPUT_DIR = Path("/Users/yoshiri/Documents/OSS/perception_eval_sql/data/output/safety_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAFETY_CLASSES = ["car", "truck", "bus", "pedestrian"]
DIST_THRESHOLDS_M = [0.5, 1.0, 2.0, 4.0]  # nuScenes matching thresholds
SAFETY_R_BINS = ["0-20", "20-40", "40-60", "60-80", "80-100"]  # 0-100m

# NuScenes TP metric normalization thresholds
ATE_NORM  = 2.0   # m   (nuScenes uses 2.0 for center dist error)
AOE_NORM  = np.pi  # rad (max yaw error)
AVE_NORM  = 10.0  # m/s (reasonable speed error cap)

print("Loading CSV...", flush=True)
df = pd.read_csv(CSV_PATH)
print(f"  Total rows: {len(df):,}")

# ── 1. Safety-critical filter ──────────────────────────────────────────────

gt_safe = df[
    (df["source"] == "GT") &
    (df["label"].isin(SAFETY_CLASSES)) &
    (~df["visibility"].isin(["NONE"])) &
    (df["visibility"].notna()) &
    (df["r"].isin(SAFETY_R_BINS))
].copy()

est_safe = df[
    (df["source"] == "EST") &
    (df["label"].isin(SAFETY_CLASSES)) &
    (df["r"].isin(SAFETY_R_BINS))
].copy()

# speed_error has a small number of extreme outliers (sensor artifact, e.g. >50 m/s).
# Clip to 50 m/s (180 km/h) before computing AVE to avoid mean being dominated.
SPEED_ERROR_MAX = 50.0

est_safe["ate"] = np.sqrt(
    est_safe["x_error"].fillna(np.inf) ** 2 +
    est_safe["y_error"].fillna(np.inf) ** 2
)
est_safe["aoe"] = est_safe["yaw_error"].abs().fillna(np.inf)
est_safe["ave"] = est_safe["speed_error"].abs().clip(upper=SPEED_ERROR_MAX).fillna(np.inf)
# original TP flag (before distance-threshold refinement)
est_safe["is_orig_tp"] = (est_safe["status"] == "TP")

print(f"  Safety-critical GT: {len(gt_safe):,}")
print(f"  Safety-critical EST: {len(est_safe):,}")

# ── 2. Safety-Critical Recall (no confidence sweep) ───────────────────────

recall_rows = []
for cls in SAFETY_CLASSES:
    gt_cls = gt_safe[gt_safe["label"] == cls]
    for rbin in SAFETY_R_BINS:
        gt_rb = gt_cls[gt_cls["r"] == rbin]
        tp = (gt_rb["status"] == "TP").sum()
        fn = (gt_rb["status"] == "FN").sum()
        total = tp + fn
        recall_rows.append({
            "class": cls,
            "dist_bin": rbin,
            "tp": int(tp),
            "fn": int(fn),
            "gt_total": int(total),
            "recall": round(tp / total, 4) if total > 0 else float("nan"),
        })

recall_df = pd.DataFrame(recall_rows)

# Overall per-class recall
recall_summary = recall_df.groupby("class").apply(
    lambda g: pd.Series({
        "tp": g["tp"].sum(),
        "fn": g["fn"].sum(),
        "gt_total": g["gt_total"].sum(),
        "recall_0_100m": round(g["tp"].sum() / g["gt_total"].sum(), 4)
        if g["gt_total"].sum() > 0 else float("nan"),
    })
).reset_index()

mean_recall = recall_summary["recall_0_100m"].mean()

# ── 3. nuScenes-style mAP (multiple dist matching thresholds) ─────────────

def compute_ap(est_sorted_df, n_gt):
    """All-points interpolation AP."""
    if n_gt == 0 or len(est_sorted_df) == 0:
        return 0.0
    is_tp = est_sorted_df["_is_tp_thresh"].values.astype(float)
    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(1 - is_tp)
    rec = cum_tp / n_gt
    prec = cum_tp / (cum_tp + cum_fp)
    # sentinel
    rec  = np.concatenate([[0.0], rec])
    prec = np.concatenate([[1.0], prec])
    # area
    ap = float(np.sum(np.diff(rec) * prec[1:]))
    return ap

ap_table = []  # rows: class x dist_threshold

for cls in SAFETY_CLASSES:
    gt_cls = gt_safe[gt_safe["label"] == cls]
    n_gt = len(gt_cls)

    est_cls = est_safe[est_safe["label"] == cls].copy()
    est_cls = est_cls.sort_values("confidence", ascending=False)

    for thr in DIST_THRESHOLDS_M:
        # TP at this threshold: original TP with ATE <= thr
        # Original FP stays FP; original TP with ATE > thr becomes FP
        est_cls["_is_tp_thresh"] = (
            est_cls["is_orig_tp"] & (est_cls["ate"] <= thr)
        )
        ap = compute_ap(est_cls, n_gt)
        ap_table.append({
            "class": cls,
            "match_thr_m": thr,
            "n_gt": n_gt,
            "n_est": len(est_cls),
            "ap": round(ap, 4),
        })

ap_df = pd.DataFrame(ap_table)

# mAP: mean over classes and thresholds
map_per_thr = ap_df.groupby("match_thr_m")["ap"].mean().rename("mAP")
safety_map = float(ap_df["ap"].mean())

# ── 4. TP Metrics (ATE, AOE, AVE) for all safety-critical TPs ─────────────

est_tp = est_safe[est_safe["is_orig_tp"]].copy()

tp_metrics_rows = []
for cls in SAFETY_CLASSES:
    tp_cls = est_tp[est_tp["label"] == cls]
    if len(tp_cls) == 0:
        continue
    ate_vals = tp_cls["ate"].replace([np.inf, -np.inf], np.nan).dropna()
    aoe_vals = tp_cls["aoe"].replace([np.inf, -np.inf], np.nan).dropna()
    ave_vals = tp_cls["ave"].replace([np.inf, -np.inf], np.nan).dropna()
    tp_metrics_rows.append({
        "class": cls,
        "n_tp": len(tp_cls),
        "mATE": round(ate_vals.mean(), 4) if len(ate_vals) else float("nan"),
        "mAOE_rad": round(aoe_vals.mean(), 4) if len(aoe_vals) else float("nan"),
        "mAVE": round(ave_vals.mean(), 4) if len(ave_vals) else float("nan"),
        "mATE_p50": round(ate_vals.median(), 4) if len(ate_vals) else float("nan"),
        "mATE_p90": round(np.percentile(ate_vals, 90), 4) if len(ate_vals) else float("nan"),
    })

tp_metrics_df = pd.DataFrame(tp_metrics_rows)

# Mean TP metrics across classes
mean_ate = tp_metrics_df["mATE"].mean()
mean_aoe = tp_metrics_df["mAOE_rad"].mean()
mean_ave = tp_metrics_df["mAVE"].mean()

# ── 5. NDS Safety Score ────────────────────────────────────────────────────
# NDS = 1/(5+3) * [5*mAP + (1-min(mATE/ATE_NORM,1)) + (1-min(mAOE/AOE_NORM,1)) + (1-min(mAVE/AVE_NORM,1))]
tp_ate_score = 1.0 - min(mean_ate / ATE_NORM, 1.0)
tp_aoe_score = 1.0 - min(mean_aoe / AOE_NORM, 1.0)
tp_ave_score = 1.0 - min(mean_ave / AVE_NORM, 1.0)
nds_safety = (5 * safety_map + tp_ate_score + tp_aoe_score + tp_ave_score) / 8.0

# ── 6. Print & Save ────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SAFETY-CRITICAL DETECTION METRICS  (0-100m, vis≠NONE)")
print("=" * 65)

print("\n[1] Safety-Critical Recall by class (aggregated 0-100m)")
print(recall_summary.to_string(index=False))
print(f"\n  → Mean Safety Recall: {mean_recall:.4f}")

print("\n[2] nuScenes-style AP by class × matching threshold (m)")
ap_pivot = ap_df.pivot(index="class", columns="match_thr_m", values="ap")
print(ap_pivot.to_string())
print(f"\n  mAP per matching threshold:")
print(map_per_thr.to_string())
print(f"\n  → Safety-Critical mAP (mean over classes & thresholds): {safety_map:.4f}")

print("\n[3] TP Error Metrics (safety-critical TPs only)")
print(tp_metrics_df.to_string(index=False))
print(f"\n  Mean ATE:  {mean_ate:.4f} m  (norm score: {tp_ate_score:.4f})")
print(f"  Mean AOE:  {mean_aoe:.4f} rad (norm score: {tp_aoe_score:.4f})")
print(f"  Mean AVE:  {mean_ave:.4f} m/s (norm score: {tp_ave_score:.4f})")

print("\n[4] NDS_safety Score")
print(f"  NDS_safety = 1/8 × (5 × {safety_map:.4f} + {tp_ate_score:.4f} + {tp_aoe_score:.4f} + {tp_ave_score:.4f})")
print(f"             = {nds_safety:.4f}")

print("\n" + "=" * 65)

# ── Save outputs ──────────────────────────────────────────────────────────
recall_df.to_csv(OUTPUT_DIR / "recall_by_class_dist.csv", index=False)
ap_df.to_csv(OUTPUT_DIR / "ap_by_class_threshold.csv", index=False)
tp_metrics_df.to_csv(OUTPUT_DIR / "tp_metrics_by_class.csv", index=False)
recall_detail = recall_df.pivot_table(
    index="class", columns="dist_bin", values="recall"
)[SAFETY_R_BINS]
recall_detail.to_csv(OUTPUT_DIR / "recall_heatmap.csv")

summary = {
    "job_id": "e2fb9ecc-decf-513f-b54f-2a493b2877ad",
    "catalog": "PerceptionFullPerformanceTest",
    "pilot_auto_version": "Pilot.Auto v4.4.0 (bevfusion x2/2.5.1)",
    "safety_filter": {
        "classes": SAFETY_CLASSES,
        "dist_range_m": "0-100",
        "visibility_excluded": ["NONE"],
    },
    "mean_safety_recall": round(mean_recall, 4),
    "safety_map": round(safety_map, 4),
    "safety_map_by_threshold": {str(k): round(v, 4) for k, v in map_per_thr.items()},
    "safety_map_by_class": {
        row["class"]: round(ap_df[ap_df["class"] == row["class"]]["ap"].mean(), 4)
        for _, row in recall_summary.iterrows()
    },
    "tp_metrics": {
        "mATE_m": round(mean_ate, 4),
        "mAOE_rad": round(mean_aoe, 4),
        "mAVE_mps": round(mean_ave, 4),
    },
    "nds_safety": round(nds_safety, 4),
}

import json
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("  recall_by_class_dist.csv")
print("  ap_by_class_threshold.csv")
print("  tp_metrics_by_class.csv")
print("  recall_heatmap.csv")
print("  summary.json")
