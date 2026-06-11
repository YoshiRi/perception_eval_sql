"""
Distance-To-Vehicle Weighted mAP computation.

All functions are side-effect free: take DataFrames + WeightedMapConfig,
return DataFrames or dicts.  No file I/O here.

Key deviation from standard mAP
---------------------------------
Each GT contributes gt_weight (instead of 1) to total_GT_weight and to
weighted_recall.  Each TP contributes the weight of its matched GT.
Each FP contributes fp_weight (either est_distance or 1).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import WeightedMapConfig

_DIST_BINS = [
    ("near",        0,   10),
    ("mid",        10,   30),
    ("far",        30,   50),
    ("very_far",   50,  100),
    ("out_of_scope", 100, float("inf")),
]


# ── Weight computation ────────────────────────────────────────────────────────

def _in_fov(theta_val: pd.Series, fov_deg: float) -> pd.Series:
    """Return boolean mask: True if object is within ±(fov_deg/2) of forward."""
    half = fov_deg / 2.0
    # Normalise to [-180, 180]
    theta_signed = ((theta_val + 180) % 360) - 180
    return theta_signed.abs() <= half


def add_weights(df: pd.DataFrame, cfg: WeightedMapConfig) -> pd.DataFrame:
    """
    Add weight columns to a copy of df.

    New columns
    -----------
    distance_weight : exp(-lambda * r_val)
    in_fov          : bool — within forward_fov_deg
    gt_weight       : distance_weight * in_fov * (vis OK) * (dist <= max)
                      (GT rows only; 0 for GT outside constraints)
    fp_weight       : weight for FP contribution  (EST FP rows)
    tp_weight       : gt_weight of the matched GT (EST TP rows, derived from
                      matched GT position via x_error / y_error)
    """
    out = df.copy()

    lam = cfg.distance_weight_lambda

    # ── Distance weight (for every row) ──────────────────────────────────
    out["distance_weight"] = np.exp(-lam * out["r_val"].clip(lower=0))

    # ── FOV mask ──────────────────────────────────────────────────────────
    out["in_fov"] = _in_fov(out["theta_val"], cfg.forward_fov_deg)

    # ── GT weight ─────────────────────────────────────────────────────────
    gt_mask = out["source"] == "GT"
    out["gt_weight"] = 0.0

    active_gt = (
        gt_mask
        & out["in_fov"]
        & (out["r_val"] <= cfg.max_distance_m)
        & out["label"].isin(cfg.classes)
    )
    if cfg.visibility_exclude:
        active_gt &= ~out["visibility"].isin(cfg.visibility_exclude)

    out.loc[active_gt, "gt_weight"] = out.loc[active_gt, "distance_weight"]

    # ── FP weight (for EST FP rows) ───────────────────────────────────────
    # Only in-scope FP (in_fov + r_val <= max_distance_m) are penalised.
    # Out-of-scope FP stay at 0: mirroring the PoC spec region_weight=0 logic
    # and preventing out-of-range detections from inflating the FP count.
    est_mask = out["source"] == "EST"
    out["fp_weight"] = 0.0

    fp_active = (
        est_mask
        & (out["status"] == "FP")
        & out["label"].isin(cfg.classes)
        & out["in_fov"]
        & (out["r_val"] <= cfg.max_distance_m)
    )
    if cfg.fp_weighting == "est_distance":
        out.loc[fp_active, "fp_weight"] = out.loc[fp_active, "distance_weight"]
    else:
        # uniform: in-scope FP each contribute 1.0
        out.loc[fp_active, "fp_weight"] = 1.0

    # ── TP weight: join EST TP → GT TP via (unix_time, label, x_error, y_error)
    # This is the correct approach: tp_weight must inherit gt_weight from the
    # matched GT, including visibility exclusions.  Deriving GT position from
    # x_error/y_error alone cannot capture visibility-based weight=0.
    out["tp_weight"] = 0.0
    tp_est_mask = est_mask & (out["status"] == "TP") & out["label"].isin(cfg.classes)
    gt_tp_mask  = gt_mask  & (out["status"] == "TP") & out["label"].isin(cfg.classes)

    if tp_est_mask.any() and gt_tp_mask.any():
        gt_tp = out.loc[gt_tp_mask, ["unix_time", "label", "x_error", "y_error", "gt_weight"]].copy()
        # Round to 4 dp for safe merge (float repr may vary slightly)
        gt_tp["_xe"] = gt_tp["x_error"].round(4)
        gt_tp["_ye"] = gt_tp["y_error"].round(4)

        est_tp = out.loc[tp_est_mask, ["x_error", "y_error", "unix_time", "label"]].copy()
        est_tp["_xe"] = est_tp["x_error"].round(4)
        est_tp["_ye"] = est_tp["y_error"].round(4)

        merged = est_tp.merge(
            gt_tp[["unix_time", "label", "_xe", "_ye", "gt_weight"]],
            on=["unix_time", "label", "_xe", "_ye"],
            how="left",
        )
        # Any unmatched EST TP (should not happen) gets gt_weight=0
        merged["gt_weight"] = merged["gt_weight"].fillna(0.0)
        out.loc[tp_est_mask, "tp_weight"] = merged["gt_weight"].values

    return out


# ── Per-class AP ──────────────────────────────────────────────────────────────

def _weighted_ap(
    est_cls: pd.DataFrame,
    n_gt_active: int,
    total_gt_weight: float,
    mode: str = "weighted",
) -> float:
    """
    Compute AP for one class.

    Parameters
    ----------
    est_cls         : EST rows for this class, sorted by confidence descending
    n_gt_active     : number of GT with gt_weight > 0  (denominator for normal AP)
    total_gt_weight : sum of gt_weight for all GT (denominator for weighted AP)
    mode            : "weighted" uses tp/fp_weight; "normal" uses in-scope TP flag
    """
    if len(est_cls) == 0:
        return 0.0

    if mode == "normal":
        if n_gt_active == 0:
            return 0.0
        # is_tp: TP whose matched GT is within scope (tp_weight > 0)
        is_tp = (est_cls["tp_weight"] > 0).values.astype(float)
        # is_fp: only in-scope FP (fp_weight > 0); out-of-scope FP are ignored
        is_fp = (est_cls["fp_weight"] > 0).values.astype(float)
        cum_tp = np.cumsum(is_tp)
        cum_fp = np.cumsum(is_fp)
        n_gt = float(n_gt_active)
    else:
        if total_gt_weight == 0:
            return 0.0
        tp_w = est_cls["tp_weight"].values
        fp_w = est_cls["fp_weight"].values
        cum_tp = np.cumsum(tp_w)
        cum_fp = np.cumsum(fp_w)
        n_gt = total_gt_weight

    cum_tp = np.minimum(cum_tp, n_gt)  # cap at n_gt to keep recall ≤ 1

    rec  = cum_tp / n_gt
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    rec  = np.concatenate([[0.0], rec,  [rec[-1]  if len(rec)  else 0.0]])
    prec = np.concatenate([[1.0], prec, [0.0]])

    return float(np.sum(np.diff(rec) * prec[1:]))


# ── Full computation ──────────────────────────────────────────────────────────

def compute_weighted_map(df: pd.DataFrame, cfg: WeightedMapConfig) -> Dict:
    """
    Full pipeline: add weights → per-class weighted AP → summary + distance bin.

    Returns a dict with keys:
      class_metrics     : DataFrame with normal_AP / weighted_AP / counts per class
      distance_bin      : DataFrame with recall breakdowns per (class, dist_bin)
      weighted_map      : float  (macro average weighted AP)
      normal_map        : float  (macro average standard AP)
      delta             : weighted_map - normal_map
      high_weight_fn    : DataFrame — FN GT sorted by gt_weight desc
      high_weight_fp    : DataFrame — FP EST sorted by fp_weight desc
      n_gt              : total GT count (unweighted)
      total_gt_weight   : float
    """
    df_w = add_weights(df, cfg)

    gt_df  = df_w[(df_w["source"] == "GT") & df_w["label"].isin(cfg.classes)]
    est_df = df_w[(df_w["source"] == "EST") & df_w["label"].isin(cfg.classes)]

    class_rows = []

    for cls in cfg.classes:
        gt_cls  = gt_df[gt_df["label"] == cls]
        est_cls = (
            est_df[est_df["label"] == cls]
            .sort_values("confidence", ascending=False)
        )

        total_gt_w  = float(gt_cls["gt_weight"].sum())
        tp_count    = int((gt_cls["status"] == "TP").sum())
        fn_count    = int((gt_cls["status"] == "FN").sum())
        n_gt        = tp_count + fn_count
        n_est       = len(est_cls)
        fp_count    = int((est_cls["status"] == "FP").sum())

        weighted_tp = float(est_cls["tp_weight"].sum())
        weighted_fp = float(est_cls["fp_weight"].sum())
        weighted_fn = float(gt_cls.loc[gt_cls["status"] == "FN", "gt_weight"].sum())

        w_rec  = weighted_tp / total_gt_w  if total_gt_w  > 0 else float("nan")
        w_prec = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else float("nan")

        n_gt_active = int((gt_cls["gt_weight"] > 0).sum())
        w_ap = _weighted_ap(est_cls, n_gt_active, total_gt_w, mode="weighted")
        n_ap = _weighted_ap(est_cls, n_gt_active, total_gt_w, mode="normal")

        class_rows.append({
            "class":             cls,
            "n_gt":              n_gt,
            "n_est":             n_est,
            "total_gt_weight":   round(total_gt_w, 4),
            "weighted_TP":       round(weighted_tp, 4),
            "weighted_FP":       round(weighted_fp, 4),
            "weighted_FN":       round(weighted_fn, 4),
            "weighted_precision": round(w_prec, 4) if w_prec == w_prec else float("nan"),
            "weighted_recall":   round(w_rec,  4) if w_rec  == w_rec  else float("nan"),
            "normal_AP":         round(n_ap, 4),
            "weighted_AP":       round(w_ap, 4),
            "delta_AP":          round(w_ap - n_ap, 4),
        })

    class_df = pd.DataFrame(class_rows)
    weighted_map = float(class_df["weighted_AP"].mean())
    normal_map   = float(class_df["normal_AP"].mean())

    # ── Distance bin recall ───────────────────────────────────────────────
    bin_rows = []
    for cls in cfg.classes:
        gt_cls = gt_df[gt_df["label"] == cls]
        for bin_name, lo, hi in _DIST_BINS:
            in_bin = (gt_cls["r_val"] >= lo) & (gt_cls["r_val"] < hi)
            g = gt_cls[in_bin]
            if len(g) == 0:
                continue
            tp_w = float(g.loc[g["status"] == "TP", "gt_weight"].sum())
            gt_w = float(g["gt_weight"].sum())
            n_tp = int((g["status"] == "TP").sum())
            n_total = len(g)
            bin_rows.append({
                "class": cls,
                "dist_bin": bin_name,
                "n_gt": n_total,
                "n_tp": n_tp,
                "total_gt_weight": round(gt_w, 4),
                "weighted_tp": round(tp_w, 4),
                "normal_recall":   round(n_tp / n_total, 4) if n_total else float("nan"),
                "weighted_recall": round(tp_w / gt_w, 4)    if gt_w    else float("nan"),
            })
    dist_df = pd.DataFrame(bin_rows)

    # ── High-weight error cases ───────────────────────────────────────────
    fn_gt = gt_df[gt_df["status"] == "FN"].sort_values("gt_weight", ascending=False)
    fp_est = est_df[est_df["status"] == "FP"].sort_values("fp_weight", ascending=False)

    keep_cols_gt  = ["unix_time", "label", "x", "y", "r_val", "theta_val",
                     "visibility", "gt_weight"]
    keep_cols_est = ["unix_time", "label", "x", "y", "r_val", "theta_val",
                     "confidence", "fp_weight"]
    fn_gt  = fn_gt[[c for c in keep_cols_gt  if c in fn_gt.columns]].head(500)
    fp_est = fp_est[[c for c in keep_cols_est if c in fp_est.columns]].head(500)

    return {
        "class_metrics":   class_df,
        "distance_bin":    dist_df,
        "weighted_map":    round(weighted_map, 4),
        "normal_map":      round(normal_map, 4),
        "delta":           round(weighted_map - normal_map, 4),
        "high_weight_fn":  fn_gt,
        "high_weight_fp":  fp_est,
        "n_gt":            int(gt_df["gt_weight"].gt(0).sum()),
        "total_gt_weight": round(float(gt_df["gt_weight"].sum()), 2),
    }
