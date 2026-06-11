"""
GT/EST re-matching for parquet format.

Reads a detection parquet (output of split.py), re-runs object-level matching
within each frame using Hungarian assignment, and outputs a rematched parquet
with updated status (TP/FP/FN), pair_uuid, and position error metrics.

Note: this module operates on parquet files. For CSV-based rematch, see
script/run_rematch.py (the sibling CSV rematch tool).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    from shapely.geometry import Polygon
    _SHAPELY = True
except ImportError:
    _SHAPELY = False

FRAME_KEYS = ["t4dataset_name", "frame_index", "topic_name"]
_INF = 1e9


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class MatchConfig:
    metric: Literal["center_bev", "center_3d", "iou_bev"] = "center_bev"
    threshold: float = 2.0
    label_thresholds: dict[str, float] = field(default_factory=dict)
    class_match_map: dict[str, list[str]] | None = None
    allow_any_class: bool = False
    fallback_class_map: dict[str, list[str]] | None = None
    fallback_any_class: bool = False
    fallback_labels: set[str] = field(default_factory=set)
    fallback_threshold: float | None = None

    def get_threshold(self, gt_label: str) -> float:
        return self.label_thresholds.get(gt_label, self.threshold)

    def get_fallback_threshold(self, gt_label: str) -> float:
        thr = self.fallback_threshold if self.fallback_threshold is not None else self.threshold
        return self.label_thresholds.get(gt_label, thr)

    def can_match(self, gt_label: str, est_label: str) -> bool:
        if self.allow_any_class:
            return True
        if self.class_match_map is None:
            return gt_label == est_label
        return est_label in self.class_match_map.get(gt_label, [gt_label])

    def has_fallback(self) -> bool:
        return self.fallback_any_class or self.fallback_class_map is not None

    def can_match_fallback(self, gt_label: str, est_label: str) -> bool:
        if self.fallback_labels and gt_label not in self.fallback_labels:
            return False
        if self.fallback_any_class:
            return True
        if self.fallback_class_map is not None:
            return est_label in self.fallback_class_map.get(gt_label, [gt_label])
        return False

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "label_thresholds": self.label_thresholds,
            "class_match_map": self.class_match_map,
            "allow_any_class": self.allow_any_class,
            "fallback_class_map": self.fallback_class_map,
            "fallback_any_class": self.fallback_any_class,
            "fallback_labels": sorted(self.fallback_labels),
            "fallback_threshold": self.fallback_threshold,
        }


# ── Geometry ──────────────────────────────────────────────────────────────────

def _center_bev(gt_x, gt_y, est_x, est_y) -> np.ndarray:
    dx = gt_x[:, None] - est_x[None, :]
    dy = gt_y[:, None] - est_y[None, :]
    return np.sqrt(dx**2 + dy**2)


def _center_3d(gt_xyz, est_xyz) -> np.ndarray:
    diff = gt_xyz[:, None, :] - est_xyz[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def _iou_bev_matrix(gt_df: pd.DataFrame, est_df: pd.DataFrame) -> np.ndarray:
    if not _SHAPELY:
        raise ImportError("shapely required for iou_bev")

    def _poly(r):
        half_l, half_w = r.length / 2, r.width / 2
        corners = np.array([[ half_l, half_w], [-half_l, half_w],
                             [-half_l,-half_w], [ half_l,-half_w]])
        c, s = np.cos(r.yaw), np.sin(r.yaw)
        rotated = corners @ np.array([[c,-s],[s,c]]).T
        rotated[:, 0] += r.x; rotated[:, 1] += r.y
        return Polygon(rotated)

    gt_polys  = [_poly(r) for _, r in gt_df.iterrows()]
    est_polys = [_poly(r) for _, r in est_df.iterrows()]
    iou = np.zeros((len(gt_polys), len(est_polys)), dtype=np.float32)
    for i, gp in enumerate(gt_polys):
        for j, ep in enumerate(est_polys):
            if gp.is_valid and ep.is_valid:
                union = gp.union(ep).area
                iou[i, j] = gp.intersection(ep).area / union if union > 0 else 0.0
    return iou


# ── Matching ──────────────────────────────────────────────────────────────────

def _run_pass(gt_df, est_df, can_match_fn, get_threshold_fn, metric) -> list[tuple[int, int, float]]:
    if len(gt_df) == 0 or len(est_df) == 0:
        return []

    n_gt, n_est = len(gt_df), len(est_df)
    cost = np.full((n_gt, n_est), _INF, dtype=np.float64)

    if metric == "center_bev":
        raw = _center_bev(gt_df["x"].values, gt_df["y"].values,
                          est_df["x"].values, est_df["y"].values)
        higher_is_better = False
    elif metric == "center_3d":
        raw = _center_3d(gt_df[["x","y","z"]].values, est_df[["x","y","z"]].values)
        higher_is_better = False
    elif metric == "iou_bev":
        raw = _iou_bev_matrix(gt_df, est_df)
        higher_is_better = True
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    gt_labels  = gt_df["label"].values
    est_labels = est_df["label"].values

    for i in range(n_gt):
        thr = get_threshold_fn(str(gt_labels[i]))
        for j in range(n_est):
            if not can_match_fn(str(gt_labels[i]), str(est_labels[j])):
                continue
            score = raw[i, j]
            if higher_is_better:
                if score >= thr:
                    cost[i, j] = 1.0 - score
            else:
                if score <= thr:
                    cost[i, j] = score

    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(r), int(c), float(raw[r, c])) for r, c in zip(row_ind, col_ind) if cost[r, c] < _INF]


def match_frame(gt_df, est_df, config: MatchConfig) -> list[tuple[int, int, float]]:
    matched_1st = _run_pass(gt_df, est_df, config.can_match, config.get_threshold, config.metric)
    if not config.has_fallback():
        return matched_1st

    matched_gt  = {gi for gi, _, _ in matched_1st}
    matched_est = {ei for _, ei, _ in matched_1st}
    unmatched_gt  = [i for i in range(len(gt_df))  if i not in matched_gt]
    unmatched_est = [i for i in range(len(est_df)) if i not in matched_est]

    if not unmatched_gt or not unmatched_est:
        return matched_1st

    gt_sub  = gt_df.iloc[unmatched_gt].reset_index(drop=True)
    est_sub = est_df.iloc[unmatched_est].reset_index(drop=True)
    matched_2nd_local = _run_pass(gt_sub, est_sub, config.can_match_fallback,
                                  config.get_fallback_threshold, config.metric)
    matched_2nd = [(unmatched_gt[gi], unmatched_est[ei], s) for gi, ei, s in matched_2nd_local]
    return matched_1st + matched_2nd


# ── Main rematch function ──────────────────────────────────────────────────────

def rematch(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Re-run GT/EST matching, return DataFrame with updated status and error columns."""
    result = df.copy()
    result["status"]    = pd.array([""] * len(df), dtype="string")
    result["pair_uuid"] = pd.array([pd.NA] * len(df), dtype="string")
    for col in ["x_error","y_error","z_error","yaw_error","center_distance",
                "plane_distance","vx_error","vy_error","speed_error"]:
        if col in result.columns:
            result[col] = np.nan
    result["match_score"]  = np.nan
    result["match_metric"] = config.metric

    total_frames = df.groupby(FRAME_KEYS, observed=True).ngroups
    print(f"  Re-matching {total_frames:,} frames ...")

    for i, (_keys, frame_df) in enumerate(df.groupby(FRAME_KEYS, observed=True, sort=False)):
        if i % 500 == 0:
            print(f"    frame {i:,}/{total_frames:,}", flush=True)

        gt_frame  = frame_df[frame_df["source"] == "GT"]
        est_frame = frame_df[frame_df["source"] == "EST"]

        if len(gt_frame) == 0:
            result.loc[est_frame.index, "status"] = "FP"
            continue
        if len(est_frame) == 0:
            result.loc[gt_frame.index, "status"] = "FN"
            continue

        matched = match_frame(gt_frame, est_frame, config)
        gt_idx_list  = gt_frame.index.tolist()
        est_idx_list = est_frame.index.tolist()
        matched_gt, matched_est = set(), set()

        for gi, ei, score in matched:
            gt_idx  = gt_idx_list[gi]
            est_idx = est_idx_list[ei]
            gt_row  = df.loc[gt_idx]
            est_row = df.loc[est_idx]

            result.at[gt_idx,  "status"]      = "TP"
            result.at[est_idx, "status"]       = "TP"
            result.at[gt_idx,  "pair_uuid"]   = str(est_row["uuid"])
            result.at[est_idx, "pair_uuid"]   = str(gt_row["uuid"])
            result.at[gt_idx,  "match_score"] = score
            result.at[est_idx, "match_score"] = score

            dx = float(est_row["x"]) - float(gt_row["x"])
            dy = float(est_row["y"]) - float(gt_row["y"])
            dz = float(est_row["z"]) - float(gt_row["z"])
            result.at[gt_idx, "x_error"] = dx
            result.at[gt_idx, "y_error"] = dy
            result.at[gt_idx, "z_error"] = dz
            result.at[gt_idx, "center_distance"] = np.sqrt(dx**2 + dy**2)

            dyaw = float(est_row["yaw"]) - float(gt_row["yaw"])
            result.at[gt_idx, "yaw_error"] = (dyaw + np.pi) % (2 * np.pi) - np.pi

            if "vx" in df.columns:
                result.at[gt_idx, "vx_error"] = float(est_row["vx"]) - float(gt_row["vx"])
                result.at[gt_idx, "vy_error"] = float(est_row["vy"]) - float(gt_row["vy"])
                gt_spd  = np.sqrt(float(gt_row["vx"])**2  + float(gt_row["vy"])**2)
                est_spd = np.sqrt(float(est_row["vx"])**2 + float(est_row["vy"])**2)
                result.at[gt_idx, "speed_error"] = est_spd - gt_spd

            matched_gt.add(gi); matched_est.add(ei)

        for gi, gt_idx in enumerate(gt_idx_list):
            if gi not in matched_gt:
                result.at[gt_idx, "status"] = "FN"
        for ei, est_idx in enumerate(est_idx_list):
            if ei not in matched_est:
                result.at[est_idx, "status"] = "FP"

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def rematch_parquet(
    input_path: str,
    output_path: str,
    threshold: float = 2.0,
    cross_class: dict[str, list[str]] | None = None,
) -> None:
    """Load detection parquet, rematch, save to output_path."""
    config = MatchConfig(threshold=threshold, class_match_map=cross_class)
    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"  {len(df):,} rows  config={json.dumps(config.to_dict())}")
    rematched = rematch(df, config)

    for src, statuses in [("GT", ["TP","FN"]), ("EST", ["TP","FP"])]:
        sub = rematched[rematched["source"] == src]
        counts = sub["status"].value_counts()
        parts = [f"{s}={counts.get(s,0):,}" for s in statuses]
        print(f"  {src}: {len(sub):,}  ({', '.join(parts)})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rematched.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_cross_class(s: str) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    for pair in s.split(","):
        a, b = pair.strip().split(":", 1)
        a, b = a.strip(), b.strip()
        mapping.setdefault(a, {a}).add(b)
        mapping.setdefault(b, {b}).add(a)
    return {k: sorted(v) for k, v in mapping.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Rematch GT/EST in parquet (Hungarian)")
    p.add_argument("--input",  required=True, help="Input detection parquet")
    p.add_argument("--output", required=True, help="Output rematched parquet")
    p.add_argument("--threshold",   type=float, default=2.0)
    p.add_argument("--cross-class", type=_parse_cross_class, default=None,
                   metavar="L1:L2[,...]", help='e.g. "car:truck,bus:truck"')
    args = p.parse_args()
    rematch_parquet(args.input, args.output, args.threshold, args.cross_class)


if __name__ == "__main__":
    main()
