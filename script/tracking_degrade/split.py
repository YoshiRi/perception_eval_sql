"""
GT evaluation dedup + truck spatial bbox merge → detection parquet.

[Step 1] GT Evaluation Dedup
  Remove duplicate (t4dataset_name, frame_index, topic_name, uuid) GT rows:
  - If TP rows exist: keep the one with minimum abs(pair_dt_sec)
  - If all FN: keep the first row

[Step 2] Truck Spatial BBox Merge
  Cluster GT truck objects within a frame by IoU and merge overlapping bboxes.
  Frames with car/bus IoU violations are skipped (safety guard).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

FRAME_KEYS = ["t4dataset_name", "frame_index", "topic_name"]
EVAL_DEDUP_KEYS = ["t4dataset_name", "frame_index", "topic_name", "uuid"]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _corners_global(x: float, y: float, length: float, width: float, yaw: float) -> np.ndarray:
    half_l, half_w = length / 2.0, width / 2.0
    local = np.array([
        [ half_l,  half_w],
        [-half_l,  half_w],
        [-half_l, -half_w],
        [ half_l, -half_w],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])


def _iou(row_a, row_b) -> float:
    try:
        pa = Polygon(_corners_global(row_a.x, row_a.y, row_a.length, row_a.width, row_a.yaw))
        pb = Polygon(_corners_global(row_b.x, row_b.y, row_b.length, row_b.width, row_b.yaw))
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _merge_bboxes(rep, others: list) -> dict:
    all_corners = _corners_global(rep.x, rep.y, rep.length, rep.width, rep.yaw)
    for other in others:
        all_corners = np.vstack([
            all_corners,
            _corners_global(other.x, other.y, other.length, other.width, other.yaw),
        ])
    c, s = np.cos(rep.yaw), np.sin(rep.yaw)
    R_inv = np.array([[c, s], [-s, c]])
    local = (all_corners - np.array([rep.x, rep.y])) @ R_inv.T
    min_lx, max_lx = local[:, 0].min(), local[:, 0].max()
    min_ly, max_ly = local[:, 1].min(), local[:, 1].max()
    cx_local = (min_lx + max_lx) / 2.0
    cy_local = (min_ly + max_ly) / 2.0
    R = np.array([[c, -s], [s, c]])
    new_center = np.array([cx_local, cy_local]) @ R.T + np.array([rep.x, rep.y])
    all_rows = [rep] + others
    return {
        "x": float(new_center[0]),
        "y": float(new_center[1]),
        "length": float(max_lx - min_lx),
        "width":  float(max_ly - min_ly),
        "height": float(max(float(r.height) for r in all_rows)),
    }


# ── Union-Find ────────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self._parent[self.find(a)] = self.find(b)

    def clusters(self) -> dict[int, list[int]]:
        from collections import defaultdict
        d: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self._parent)):
            d[self.find(i)].append(i)
        return dict(d)


# ── Step 1: GT Evaluation Dedup ───────────────────────────────────────────────

def eval_dedup_gt(df: pd.DataFrame) -> pd.DataFrame:
    gt_df = df[df["source"] == "GT"].copy()
    est_df = df[df["source"] == "EST"].copy()

    dup_counts = gt_df.groupby(EVAL_DEDUP_KEYS, observed=True).size()
    dup_keys = dup_counts[dup_counts > 1]

    if len(dup_keys) == 0:
        print("  GT eval dedup: no duplicates found")
        return df

    print(f"  GT eval dedup: {len(dup_keys):,} duplicate groups")

    gt_df["_key"] = list(zip(
        gt_df["t4dataset_name"],
        gt_df["frame_index"].astype(str),
        gt_df["topic_name"],
        gt_df["uuid"].astype(str),
    ))
    dup_key_set = set(zip(
        dup_keys.reset_index()["t4dataset_name"],
        dup_keys.reset_index()["frame_index"].astype(str),
        dup_keys.reset_index()["topic_name"],
        dup_keys.reset_index()["uuid"].astype(str),
    ))

    mask_dup = gt_df["_key"].isin(dup_key_set)
    gt_no_dup = gt_df[~mask_dup].drop(columns=["_key"])
    gt_dup    = gt_df[mask_dup].drop(columns=["_key"])

    def pick_row(group: pd.DataFrame) -> pd.Series:
        tp_rows = group[group["status"] == "TP"]
        if len(tp_rows) > 0:
            return tp_rows.loc[tp_rows["pair_dt_sec"].abs().idxmin()]
        return group.iloc[0]

    deduped = pd.DataFrame([
        pick_row(grp) for _, grp in gt_dup.groupby(EVAL_DEDUP_KEYS, observed=True, sort=False)
    ])

    before, after = len(gt_df), len(gt_no_dup) + len(deduped)
    print(f"  GT rows: {before:,} → {after:,} (removed {before - after:,})")
    return pd.concat([gt_no_dup, deduped, est_df], ignore_index=True)


# ── Step 2: Truck Spatial BBox Merge ─────────────────────────────────────────

def _merge_truck_frame(
    frame_df: pd.DataFrame,
    iou_threshold: float,
    frame_key: tuple,
) -> tuple[pd.DataFrame, bool]:
    gt_rows  = frame_df[frame_df["source"] == "GT"]

    car_bus_gt = gt_rows[gt_rows["label"].isin({"car", "bus"})]
    cb_rows = list(car_bus_gt.itertuples(index=True))
    for i in range(len(cb_rows)):
        for j in range(i + 1, len(cb_rows)):
            if _iou(cb_rows[i], cb_rows[j]) > iou_threshold:
                print(f"  [WARN] car/bus IoU>{iou_threshold} in frame {frame_key} — skipping truck merge")
                return frame_df, True

    truck_gt = gt_rows[gt_rows["label"] == "truck"].copy()
    if len(truck_gt) < 2:
        return frame_df, False

    rows = list(truck_gt.itertuples(index=True))
    n = len(rows)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _iou(rows[i], rows[j]) >= iou_threshold:
                uf.union(i, j)

    updates: dict[int, dict] = {}
    drop_indices: list[int] = []

    for members in uf.clusters().values():
        if len(members) == 1:
            continue
        dists = [np.sqrt(rows[m].x ** 2 + rows[m].y ** 2) for m in members]
        rep_pos = members[int(np.argmin(dists))]
        rep_row = rows[rep_pos]
        others = [rows[m] for m in members if m != rep_pos]
        merged = _merge_bboxes(rep_row, others)
        merged["status"] = "TP" if any(rows[m].status == "TP" for m in members) else rep_row.status
        updates[rep_row.Index] = merged
        for other in others:
            drop_indices.append(other.Index)

    result = frame_df.copy()
    for idx, upd in updates.items():
        for col, val in upd.items():
            result.at[idx, col] = val
    return result.drop(index=drop_indices), False


def truck_bbox_merge(df: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    topics = sorted(df["topic_name"].dropna().unique())
    parts: list[pd.DataFrame] = []
    total_removed = 0

    for topic in topics:
        short = topic.split(".")[-2] if "." in topic else topic
        topic_df = df[df["topic_name"] == topic].copy()
        before = len(topic_df)
        frame_parts: list[pd.DataFrame] = []
        skip_count = 0

        for keys, frame_df in topic_df.groupby(FRAME_KEYS, observed=True, sort=False):
            merged_frame, skipped = _merge_truck_frame(frame_df, iou_threshold, keys)
            frame_parts.append(merged_frame)
            if skipped:
                skip_count += 1

        topic_df = pd.concat(frame_parts, ignore_index=False)
        removed = before - len(topic_df)
        total_removed += removed
        print(f"  {short}: removed {removed:,} truck rows (skipped {skip_count} frames)")
        parts.append(topic_df)

    print(f"  Total truck rows removed: {total_removed:,}")
    return pd.concat(parts, ignore_index=False)


# ── Public API ────────────────────────────────────────────────────────────────

def split_and_prep(
    input_path: str,
    output_path: str,
    topic_filter: str | None = None,
    iou_threshold: float = 0.3,
) -> None:
    """Run GT dedup + truck merge, save to output_path."""
    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)

    if topic_filter:
        before = len(df)
        df = df[df["topic_name"].str.contains(topic_filter, na=False)].copy()
        print(f"  Topic filter '{topic_filter}': {before:,} → {len(df):,} rows")

    print(f"  {len(df):,} rows  (GT: {(df['source']=='GT').sum():,}  EST: {(df['source']=='EST').sum():,})")

    print("\n[Step 1] GT Evaluation Dedup")
    df = eval_dedup_gt(df)

    print(f"\n[Step 2] Truck Spatial BBox Merge (IoU threshold={iou_threshold})")
    df = truck_bbox_merge(df, iou_threshold)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}  ({len(df):,} rows)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="GT dedup + truck bbox merge → detection parquet")
    p.add_argument("--input",  required=True, help="Input parquet path")
    p.add_argument("--output", required=True, help="Output parquet path")
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--topic-filter",  default=None, help="Topic name substring filter")
    args = p.parse_args()
    split_and_prep(args.input, args.output, args.topic_filter, args.iou_threshold)


if __name__ == "__main__":
    main()
