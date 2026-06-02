"""
Pure computation functions for safety-critical detection metrics.
All functions are side-effect free: they take DataFrames and a SafetyConfig
and return DataFrames or plain dicts.  No file I/O is performed here.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import SafetyConfig


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare(df: pd.DataFrame, cfg: SafetyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the raw CSV into safety-filtered GT and EST DataFrames.

    GT filter:
      - source == 'GT'
      - label in cfg.classes
      - visibility NOT in cfg.visibility_exclude  (and not NaN)
      - r bin within dist_max_m

    EST filter:
      - source == 'EST'
      - label in cfg.classes
      - r bin within dist_max_m
    Returns (gt_df, est_df).  est_df gains ate / aoe / ave columns.
    """
    r_bins = cfg.r_bins()

    gt = df[
        (df["source"] == "GT") &
        (df["label"].isin(cfg.classes)) &
        (~df["visibility"].isin(cfg.visibility_exclude)) &
        (df["visibility"].notna()) &
        (df["r"].isin(r_bins))
    ].copy()

    est = df[
        (df["source"] == "EST") &
        (df["label"].isin(cfg.classes)) &
        (df["r"].isin(r_bins))
    ].copy()

    est["ate"] = np.sqrt(
        est["x_error"].fillna(np.inf) ** 2 +
        est["y_error"].fillna(np.inf) ** 2
    )
    est["aoe"] = est["yaw_error"].abs().fillna(np.inf)
    est["ave"] = (
        est["speed_error"]
        .abs()
        .clip(upper=cfg.speed_error_clip_mps)
        .fillna(np.inf)
    )
    est["is_orig_tp"] = (est["status"] == "TP")

    return gt, est


# ── Recall ────────────────────────────────────────────────────────────────────

def compute_recall(
    gt: pd.DataFrame, cfg: SafetyConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute safety-critical recall from GT rows.

    Returns:
      detail_df  - recall per (class, dist_bin)
      summary_df - recall per class (aggregated over dist_max_m)
    """
    r_bins = cfg.r_bins()
    rows = []
    for cls in cfg.classes:
        gt_cls = gt[gt["label"] == cls]
        for rbin in r_bins:
            gt_rb = gt_cls[gt_cls["r"] == rbin]
            tp = int((gt_rb["status"] == "TP").sum())
            fn = int((gt_rb["status"] == "FN").sum())
            total = tp + fn
            rows.append({
                "class": cls,
                "dist_bin": rbin,
                "tp": tp,
                "fn": fn,
                "gt_total": total,
                "recall": round(tp / total, 4) if total > 0 else float("nan"),
            })
    detail = pd.DataFrame(rows)

    summary_rows = []
    for cls in cfg.classes:
        g = detail[detail["class"] == cls]
        tp_sum = g["tp"].sum()
        gt_sum = g["gt_total"].sum()
        summary_rows.append({
            "class": cls,
            "tp": int(tp_sum),
            "fn": int(g["fn"].sum()),
            "gt_total": int(gt_sum),
            "recall": round(tp_sum / gt_sum, 4) if gt_sum > 0 else float("nan"),
        })
    summary = pd.DataFrame(summary_rows)

    return detail, summary


# ── AP / mAP ─────────────────────────────────────────────────────────────────

def _ap_at_threshold(
    est_cls: pd.DataFrame, n_gt: int, dist_thr: float
) -> float:
    """
    All-points interpolation AP for one class at one matching threshold.

    A detection is TP if status=='TP' AND ate <= dist_thr,
    otherwise FP.  est_cls must already be sorted by confidence descending.
    """
    if n_gt == 0 or len(est_cls) == 0:
        return 0.0

    is_tp = (
        est_cls["is_orig_tp"] & (est_cls["ate"] <= dist_thr)
    ).values.astype(float)

    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(1.0 - is_tp)

    # Cap cumulative TP at n_gt: this handles the case where the visibility
    # filter on GT is stricter than on EST (e.g. near_strict excludes PARTIAL
    # from GT denominator, but some EST TPs matched those excluded objects).
    # Without pair_uuid we cannot remove those detections, so we cap recall
    # at 1.0 rather than allowing AP > 1.
    cum_tp = np.minimum(cum_tp, n_gt)

    rec  = cum_tp / n_gt
    prec = cum_tp / (cum_tp + cum_fp)

    rec  = np.concatenate([[0.0], rec])
    prec = np.concatenate([[1.0], prec])
    return float(np.sum(np.diff(rec) * prec[1:]))


def compute_ap_table(
    gt: pd.DataFrame, est: pd.DataFrame, cfg: SafetyConfig
) -> pd.DataFrame:
    """
    Compute AP for every (class × match_threshold) combination.

    Returns a DataFrame with columns:
      class, match_thr_m, n_gt, n_est, ap
    """
    rows = []
    for cls in cfg.classes:
        gt_cls = gt[gt["label"] == cls]
        n_gt = len(gt_cls)

        est_cls = (
            est[est["label"] == cls]
            .sort_values("confidence", ascending=False)
        )

        for thr in cfg.match_thresholds_m:
            ap = _ap_at_threshold(est_cls, n_gt, thr)
            rows.append({
                "class": cls,
                "match_thr_m": thr,
                "n_gt": n_gt,
                "n_est": len(est_cls),
                "ap": round(ap, 4),
            })

    return pd.DataFrame(rows)


def compute_map(ap_df: pd.DataFrame, cfg: SafetyConfig) -> float:
    """
    Overall safety mAP: mean AP over all (class × threshold) combinations.
    If class_weights are set, each class AP (averaged over thresholds) is
    weighted before taking the mean.
    """
    if cfg.class_weights is None:
        return float(ap_df["ap"].mean())

    weighted_sum = 0.0
    total_weight = 0.0
    for cls in cfg.classes:
        w = cfg.class_weights.get(cls, 1.0)
        cls_mean = ap_df[ap_df["class"] == cls]["ap"].mean()
        weighted_sum += w * cls_mean
        total_weight += w
    return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0


# ── TP Error Metrics ──────────────────────────────────────────────────────────

def compute_tp_metrics(est: pd.DataFrame, cfg: SafetyConfig) -> pd.DataFrame:
    """
    Compute per-class TP error metrics:
      mATE (m), mAOE (rad), mAVE (m/s), plus p50/p90 for ATE.
    Only safety-critical EST TPs are used.
    """
    est_tp = est[est["is_orig_tp"]]
    rows = []
    for cls in cfg.classes:
        tp_cls = est_tp[est_tp["label"] == cls]
        if len(tp_cls) == 0:
            rows.append({
                "class": cls, "n_tp": 0,
                "mATE": float("nan"), "mAOE_rad": float("nan"),
                "mAVE": float("nan"), "mATE_p50": float("nan"),
                "mATE_p90": float("nan"),
            })
            continue

        def _mean(col):
            v = tp_cls[col].replace([np.inf, -np.inf], np.nan).dropna()
            return round(float(v.mean()), 4) if len(v) else float("nan")

        def _percentile(col, p):
            v = tp_cls[col].replace([np.inf, -np.inf], np.nan).dropna()
            return round(float(np.percentile(v, p)), 4) if len(v) else float("nan")

        rows.append({
            "class": cls,
            "n_tp": len(tp_cls),
            "mATE": _mean("ate"),
            "mAOE_rad": _mean("aoe"),
            "mAVE": _mean("ave"),
            "mATE_p50": _percentile("ate", 50),
            "mATE_p90": _percentile("ate", 90),
        })

    return pd.DataFrame(rows)


# ── NDS ───────────────────────────────────────────────────────────────────────

def compute_nds(
    safety_map: float,
    tp_metrics: pd.DataFrame,
    cfg: SafetyConfig,
) -> Dict:
    """
    Compute NDS_safety and its component scores.

    NDS_safety = 1/N * (map_weight * mAP + ate_score + aoe_score + ave_score)
    where N = map_weight + 3.
    """
    mean_ate = float(tp_metrics["mATE"].mean())
    mean_aoe = float(tp_metrics["mAOE_rad"].mean())
    mean_ave = float(tp_metrics["mAVE"].mean())

    ate_score = 1.0 - min(mean_ate / cfg.ate_norm_m, 1.0)
    aoe_score = 1.0 - min(mean_aoe / cfg.aoe_norm_rad, 1.0)
    ave_score = 1.0 - min(mean_ave / cfg.ave_norm_mps, 1.0)

    n = cfg.nds_n_terms()
    nds = (cfg.map_weight_in_nds * safety_map + ate_score + aoe_score + ave_score) / n

    return {
        "nds_safety": round(nds, 4),
        "safety_map": round(safety_map, 4),
        "tp_scores": {
            "ate_score": round(ate_score, 4),
            "aoe_score": round(aoe_score, 4),
            "ave_score": round(ave_score, 4),
        },
        "tp_means": {
            "mATE_m": round(mean_ate, 4),
            "mAOE_rad": round(mean_aoe, 4),
            "mAVE_mps": round(mean_ave, 4),
        },
    }


# ── Convenience: run everything ───────────────────────────────────────────────

def run_all(df: pd.DataFrame, cfg: SafetyConfig) -> Dict:
    """
    Full pipeline: prepare → recall → AP → TP metrics → NDS.
    Returns a dict with all intermediate DataFrames and final scores.
    """
    gt, est = prepare(df, cfg)

    recall_detail, recall_summary = compute_recall(gt, cfg)
    ap_df = compute_ap_table(gt, est, cfg)
    safety_map = compute_map(ap_df, cfg)
    tp_metrics = compute_tp_metrics(est, cfg)
    nds_result = compute_nds(safety_map, tp_metrics, cfg)

    mean_recall = float(recall_summary["recall"].mean())

    return {
        "recall_detail": recall_detail,
        "recall_summary": recall_summary,
        "recall_heatmap": recall_detail.pivot_table(
            index="class", columns="dist_bin", values="recall"
        )[cfg.r_bins()],
        "ap_table": ap_df,
        "ap_pivot": ap_df.pivot(
            index="class", columns="match_thr_m", values="ap"
        ),
        "tp_metrics": tp_metrics,
        "nds": nds_result,
        "mean_safety_recall": round(mean_recall, 4),
        "n_gt_safe": len(gt),
        "n_est_safe": len(est),
    }
