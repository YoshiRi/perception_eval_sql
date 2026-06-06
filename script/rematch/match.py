"""
GT-EST re-matching logic.

The matching algorithm operates per (unix_time, label-group):
  1. Build a cost matrix of BEV Euclidean distances between GT and EST positions.
  2. Set cost = inf where distance > threshold (no match allowed).
  3. Run Hungarian (or greedy) assignment.
  4. Assign TP / FP / FN and recompute x_error / y_error for matched pairs.

Limitations
-----------
- yaw_error and speed_error cannot be recomputed from this CSV (no raw yaw/velocity
  columns); they are set to NaN for all rematched rows.
- pair_uuid is not available, so the match is re-derived purely from position.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import RematchConfig

_INF = 1e9   # sentinel replacing np.inf for the scipy solver


# ── Label grouping ────────────────────────────────────────────────────────────

def _build_match_groups(
    gt: pd.DataFrame, est: pd.DataFrame, cfg: RematchConfig
) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Return list of (gt_idx, est_idx) pairs that should be matched together.
    Each pair represents one bipartite matching problem.
    """
    if cfg.label_match == "agnostic":
        return [(gt.index, est.index)]

    if cfg.label_match == "grouped" and cfg.label_groups:
        groups: Dict[int, Tuple[List, List]] = {}
        ungrouped_gt: List = []
        ungrouped_est: List = []
        for idx in gt.index:
            g = cfg.label_group_of(gt.loc[idx, "label"])
            if g is None:
                ungrouped_gt.append(idx)
            else:
                groups.setdefault(g, ([], []))[0].append(idx)
        for idx in est.index:
            g = cfg.label_group_of(est.loc[idx, "label"])
            if g is None:
                ungrouped_est.append(idx)
            else:
                groups.setdefault(g, ([], []))[1].append(idx)
        result = [(pd.Index(g[0]), pd.Index(g[1])) for g in groups.values()]
        # ungrouped labels fall back to strict matching
        all_ungrouped_labels = set(
            gt.loc[ungrouped_gt, "label"].unique().tolist()
            + est.loc[ungrouped_est, "label"].unique().tolist()
        )
        for lbl in all_ungrouped_labels:
            gi = pd.Index([i for i in ungrouped_gt if gt.loc[i, "label"] == lbl])
            ei = pd.Index([i for i in ungrouped_est if est.loc[i, "label"] == lbl])
            result.append((gi, ei))
        return result

    # Default: strict — one group per label
    all_labels = set(gt["label"].unique()) | set(est["label"].unique())
    return [
        (gt.index[gt["label"] == lbl], est.index[est["label"] == lbl])
        for lbl in all_labels
    ]


# ── Single bipartite assignment ───────────────────────────────────────────────

def _assign(
    gt_xy: np.ndarray,      # (n_gt, 2)
    est_xy: np.ndarray,     # (n_est, 2)
    est_conf: np.ndarray,   # (n_est,)  confidence scores
    threshold: float,
    algorithm: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal (or greedy) assignment.
    Returns (row_ind, col_ind) of matched (GT, EST) pairs within threshold.
    """
    if len(gt_xy) == 0 or len(est_xy) == 0:
        return np.array([], int), np.array([], int)

    # Cost matrix: BEV Euclidean distance
    diff = gt_xy[:, None, :] - est_xy[None, :, :]   # (n_gt, n_est, 2)
    cost = np.sqrt((diff ** 2).sum(axis=2))           # (n_gt, n_est)

    if algorithm == "greedy":
        return _greedy_assign(cost, est_conf, threshold)

    # Hungarian
    feasible = cost <= threshold
    if not feasible.any():
        return np.array([], int), np.array([], int)

    solver_cost = np.where(cost > threshold, _INF, cost)
    row_ind, col_ind = linear_sum_assignment(solver_cost)
    valid = cost[row_ind, col_ind] <= threshold
    return row_ind[valid], col_ind[valid]


def _greedy_assign(
    cost: np.ndarray, est_conf: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort EST by confidence desc; greedily assign to nearest available GT."""
    order = np.argsort(-est_conf)
    matched_gt: set = set()
    matched_est: set = set()
    rows, cols = [], []
    for est_j in order:
        candidates = np.where(
            (cost[:, est_j] <= threshold)
            & ~np.isin(np.arange(cost.shape[0]), list(matched_gt))
        )[0]
        if len(candidates) == 0:
            continue
        best_gt = candidates[cost[candidates, est_j].argmin()]
        matched_gt.add(best_gt)
        matched_est.add(est_j)
        rows.append(best_gt)
        cols.append(est_j)
    return np.array(rows, int), np.array(cols, int)


# ── Per-timestamp matching ────────────────────────────────────────────────────

def _rematch_timestamp(
    gt_rows: pd.DataFrame,
    est_rows: pd.DataFrame,
    cfg: RematchConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-match GT and EST objects within one unix_time.
    Returns (new_gt_rows, new_est_rows) with updated status / error columns.
    """
    gt_out = gt_rows.copy()
    est_out = est_rows.copy()

    # Default: everything unmatched
    gt_out["status"] = "FN"
    est_out["status"] = "FP"
    for col in ("x_error", "y_error", "yaw_error", "speed_error"):
        gt_out[col] = np.nan
        est_out[col] = np.nan

    # --- Separate visibility-excluded GTs (always FN, skip matching pool) ---
    if cfg.visibility_exclude_from_gt:
        excluded_mask = gt_out["visibility"].isin(cfg.visibility_exclude_from_gt)
        gt_pool = gt_out[~excluded_mask]
    else:
        excluded_mask = pd.Series(False, index=gt_out.index)
        gt_pool = gt_out

    # --- Confidence filter on EST ---
    if cfg.confidence_min > 0:
        est_pool = est_out[est_out["confidence"] >= cfg.confidence_min]
        est_excluded = est_out[est_out["confidence"] < cfg.confidence_min]
        # Low-confidence detections stay FP
    else:
        est_pool = est_out
        est_excluded = est_out.iloc[:0]  # empty

    if gt_pool.empty or est_pool.empty:
        return gt_out, est_out

    # --- Build label groups and run assignment ---
    groups = _build_match_groups(gt_pool, est_pool, cfg)

    for gt_idx, est_idx in groups:
        if len(gt_idx) == 0 or len(est_idx) == 0:
            continue

        # Representative threshold: use first GT label in group
        first_gt_label = gt_pool.loc[gt_idx[0], "label"] if len(gt_idx) > 0 else "__default__"
        threshold = cfg.threshold_for(first_gt_label)

        gt_xy  = gt_pool.loc[gt_idx, ["x", "y"]].values.astype(float)
        est_xy = est_pool.loc[est_idx, ["x", "y"]].values.astype(float)
        est_conf = est_pool.loc[est_idx, "confidence"].values.astype(float)

        row_ind, col_ind = _assign(gt_xy, est_xy, est_conf, threshold, cfg.algorithm)

        for r, c in zip(row_ind, col_ind):
            gi = gt_idx[r]
            ei = est_idx[c]

            x_err = float(est_pool.loc[ei, "x"]) - float(gt_pool.loc[gi, "x"])
            y_err = float(est_pool.loc[ei, "y"]) - float(gt_pool.loc[gi, "y"])

            gt_out.loc[gi, "status"]  = "TP"
            gt_out.loc[gi, "x_error"] = x_err
            gt_out.loc[gi, "y_error"] = y_err

            est_out.loc[ei, "status"]  = "TP"
            est_out.loc[ei, "x_error"] = x_err
            est_out.loc[ei, "y_error"] = y_err

    return gt_out, est_out


# ── Full dataset ──────────────────────────────────────────────────────────────

def rematch_all(df: pd.DataFrame, cfg: RematchConfig) -> pd.DataFrame:
    """
    Apply re-matching to every unix_time in df.
    Returns a new DataFrame with the same schema, updated status/error columns.
    """
    gt_df  = df[df["source"] == "GT"].copy()
    est_df = df[df["source"] == "EST"].copy()
    other  = df[~df["source"].isin(["GT", "EST"])].copy()  # guard: keep unknown rows

    gt_parts: List[pd.DataFrame]  = []
    est_parts: List[pd.DataFrame] = []

    timestamps = sorted(df["unix_time"].unique())
    n = len(timestamps)

    for i, ts in enumerate(timestamps):
        if i % 1000 == 0:
            print(f"  matching {i}/{n} timestamps ...", end="\r", flush=True)

        gt_ts  = gt_df[gt_df["unix_time"] == ts]
        est_ts = est_df[est_df["unix_time"] == ts]

        if gt_ts.empty and est_ts.empty:
            continue

        if gt_ts.empty:
            # all EST → FP
            e = est_ts.copy()
            e["status"] = "FP"
            for col in ("x_error", "y_error", "yaw_error", "speed_error"):
                e[col] = np.nan
            est_parts.append(e)
            continue

        if est_ts.empty:
            # all GT → FN
            g = gt_ts.copy()
            g["status"] = "FN"
            for col in ("x_error", "y_error", "yaw_error", "speed_error"):
                g[col] = np.nan
            gt_parts.append(g)
            continue

        new_gt, new_est = _rematch_timestamp(gt_ts, est_ts, cfg)
        gt_parts.append(new_gt)
        est_parts.append(new_est)

    print(f"  matched {n}/{n} timestamps.    ")

    parts = gt_parts + est_parts + ([other] if not other.empty else [])
    if not parts:
        return df.copy()

    result = pd.concat(parts, ignore_index=True)
    return result


# ── Comparison summary ────────────────────────────────────────────────────────

def compare_status(
    original: pd.DataFrame, rematched: pd.DataFrame
) -> Dict:
    """
    Compute statistics comparing original vs rematched status assignments.
    Both DataFrames must have the same index order (use reset_index before calling).
    """
    # Align on index (should already match after reset_index in caller)
    orig_gt  = original[original["source"] == "GT"]["status"]
    new_gt   = rematched[rematched["source"] == "GT"]["status"]
    orig_est = original[original["source"] == "EST"]["status"]
    new_est  = rematched[rematched["source"] == "EST"]["status"]

    def _counts(s: pd.Series) -> Dict:
        return {v: int((s == v).sum()) for v in ("TP", "FP", "FN")}

    orig_gt_counts  = _counts(orig_gt)
    new_gt_counts   = _counts(new_gt)
    orig_est_counts = _counts(orig_est)
    new_est_counts  = _counts(new_est)

    # Per-class recall change (GT side)
    recall_rows = []
    for cls in sorted(original["label"].unique()):
        og = original[(original["source"] == "GT") & (original["label"] == cls)]
        ng = rematched[(rematched["source"] == "GT") & (rematched["label"] == cls)]
        def _recall(s):
            tp = (s == "TP").sum()
            total = (s.isin(["TP","FN"])).sum()
            return round(tp / total, 4) if total else float("nan")
        recall_rows.append({
            "class": cls,
            "recall_orig": _recall(og["status"]),
            "recall_new":  _recall(ng["status"]),
            "gt_total": len(og[og["status"].isin(["TP","FN"])]),
        })

    return {
        "gt": {"original": orig_gt_counts, "rematched": new_gt_counts},
        "est": {"original": orig_est_counts, "rematched": new_est_counts},
        "recall_by_class": recall_rows,
    }
