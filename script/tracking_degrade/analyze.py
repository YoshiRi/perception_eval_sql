"""
Tracking Degradation Analysis
==============================
CP TP → Tracking FN となる「デグレ」物体を分析し、原因分類と truly_lost サブ分類を付与する。

## 原因分類（reason カラム）

  init_empty_frame     : 最初フレームかつ EST ゼロ
  empty_frame          : 同フレームに EST ゼロ
  class_mismatch_close : 2m 以内に異クラス EST
  yaw_or_score_issue   : 2m 以内・同クラス EST があるが FN
  close_miss_2to5m     : 同フレームの最近傍 EST が 2–5m
  temporal_gap         : 同フレームは 5m+ だが前後フレームには 5m 以内に EST
  far_miss_5to10m      : 同フレームの最近傍 EST が 5–10m
  truly_lost           : 同フレームも前後フレームも EST が 10m+ 以遠

## truly_lost サブ分類（tl_subclass カラム）

  init_delay            : CP 出現初期（rank ≤ CP_INIT_THRESH）かつ TR が後で TP
  brief_appearance      : CP 出現初期かつ TR 一度も TP 無し
  init_delay_past       : CP 出現初期かつ TR TP は既に過去
  track_lost_no_recover : CP 確立済み（rank > CP_INIT_THRESH）かつ TR TP は過去にあり未来なし
  permanent_lost        : CP 確立済みかつ TR 一度も TP 無し
  track_lost_reacquired : CP 確立済みかつ TR が後で再取得

## パラメータ

  MATCH_THRESH    = 2.0m  (truly_lost 判定の distance threshold)
  CLOSE_THRESH    = 5.0m  (close_miss の上限)
  FAR_THRESH      = 10.0m (far_miss の上限)
  CP_INIT_THRESH  = 2     (出現初期の rank 上限、0-indexed)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_THRESH   = 2.0
CLOSE_THRESH   = 5.0
FAR_THRESH     = 10.0
CP_INIT_THRESH = 2


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_and_prep(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["unix_time_ms"] = (df["unix_time"] / 1e6).round().astype("int64")
    df["dist"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    return df


def nearest_in_frame(
    est_frame: pd.DataFrame, gx: float, gy: float, g_yaw: float, g_label: str
) -> tuple[float, str, float, bool]:
    if len(est_frame) == 0:
        return np.nan, "NONE", np.nan, True
    dists = np.sqrt((est_frame["x"] - gx) ** 2 + (est_frame["y"] - gy) ** 2)
    min_idx = dists.idxmin()
    nearest_yaw = est_frame.loc[min_idx, "yaw"]
    yaw_diff = abs(((nearest_yaw - g_yaw + np.pi) % (2 * np.pi)) - np.pi) * 180 / np.pi
    return float(dists[min_idx]), str(est_frame.loc[min_idx, "label"]), float(yaw_diff), False


def classify_reason(
    d0: float, lbl0: str, empty0: bool, g_label: str, is_first: bool, min_neighbor: float
) -> str:
    if is_first and empty0:
        return "init_empty_frame"
    if empty0:
        return "empty_frame"
    if d0 <= MATCH_THRESH and lbl0 != g_label:
        return "class_mismatch_close"
    if d0 <= MATCH_THRESH:
        return "yaw_or_score_issue"
    if d0 <= CLOSE_THRESH:
        return "close_miss_2to5m"
    if d0 <= FAR_THRESH:
        if not np.isnan(min_neighbor) and min_neighbor <= CLOSE_THRESH:
            return "temporal_gap"
        return "far_miss_5to10m"
    if not np.isnan(min_neighbor) and min_neighbor <= FAR_THRESH:
        return "temporal_gap"
    return "truly_lost"


def classify_tl_subclass(cp_rank: int, current_t_ms: int, tr_first_tp_ms: float) -> str:
    is_early = cp_rank <= CP_INIT_THRESH
    has_tr_tp = not (isinstance(tr_first_tp_ms, float) and np.isnan(tr_first_tp_ms))

    if is_early:
        if not has_tr_tp:
            return "brief_appearance"
        return "init_delay" if tr_first_tp_ms > current_t_ms else "init_delay_past"
    else:
        if not has_tr_tp:
            return "permanent_lost"
        return "track_lost_reacquired" if tr_first_tp_ms > current_t_ms else "track_lost_no_recover"


# ── Main analysis ─────────────────────────────────────────────────────────────

def build_degrade_db(
    cp_path: str,
    tr_rematched_path: str,
    tr_det_path: str,
) -> pd.DataFrame:
    """Build degradation database with cause classification and truly_lost subclassification."""
    cp = load_and_prep(cp_path)
    tr = load_and_prep(tr_rematched_path)
    tr_det = load_and_prep(tr_det_path)

    speed_raw = np.sqrt(cp["vx"] ** 2 + cp["vy"] ** 2)
    speed_p99 = speed_raw.quantile(0.99)
    print(f"  GT speed p99 = {speed_p99:.2f} m/s")

    cp_gt_tp = cp[(cp["source"] == "GT") & (cp["status"] == "TP")].copy()
    cp_gt_tp = cp_gt_tp.sort_values(["t4dataset_id", "uuid", "unix_time_ms"])
    cp_gt_tp["gt_key"]         = list(zip(cp_gt_tp["t4dataset_id"], cp_gt_tp["uuid"], cp_gt_tp["unix_time_ms"]))
    cp_gt_tp["gt_speed"]       = speed_raw.clip(upper=speed_p99)[cp_gt_tp.index]
    cp_gt_tp["cp_rank"]        = cp_gt_tp.groupby(["t4dataset_id", "uuid"]).cumcount()
    cp_gt_tp["cp_total_frames"] = cp_gt_tp.groupby(["t4dataset_id", "uuid"])["uuid"].transform("count")

    tr_fn_gt  = tr[(tr["source"] == "GT") & (tr["status"] == "FN")].copy()
    tr_fn_gt["gt_key"] = list(zip(tr_fn_gt["t4dataset_id"], tr_fn_gt["uuid"], tr_fn_gt["unix_time_ms"]))
    tr_fn_keys = set(tr_fn_gt["gt_key"])

    degrade = cp_gt_tp[cp_gt_tp["gt_key"].isin(tr_fn_keys)].copy().reset_index(drop=True)
    print(f"  Degrade rows (CP TP & TR FN): {len(degrade):,}")

    tr_tp_gt = tr[(tr["source"] == "GT") & (tr["status"] == "TP")]
    tr_first_tp = (
        tr_tp_gt.groupby(["t4dataset_id", "uuid"])["unix_time_ms"]
        .min().rename("tr_first_tp_ms").reset_index()
    )
    degrade = degrade.merge(tr_first_tp, on=["t4dataset_id", "uuid"], how="left")
    degrade["delay_sec"] = degrade["tr_first_tp_ms"] - degrade["unix_time_ms"]

    tr_est = tr_det[tr_det["source"] == "EST"][
        ["t4dataset_id", "unix_time_ms", "x", "y", "yaw", "label"]
    ].copy()

    ds_sorted_times: dict[str, list[int]] = (
        tr_est.groupby("t4dataset_id")["unix_time_ms"].apply(sorted).to_dict()
    )
    ds_first_frame: dict[str, int] = (
        tr_est.groupby("t4dataset_id")["unix_time_ms"].min().to_dict()
    )
    est_by_frame: dict[tuple, pd.DataFrame] = {
        (ds_id, t_ms): grp
        for (ds_id, t_ms), grp in tr_est.groupby(["t4dataset_id", "unix_time_ms"])
    }

    results = []
    for (ds_id, t_ms), grp_gt in degrade.groupby(["t4dataset_id", "unix_time_ms"]):
        times = ds_sorted_times.get(ds_id, [])
        t_idx = times.index(t_ms) if t_ms in times else -1
        prev_t = times[t_idx - 1] if t_idx > 0 else None
        next_t = times[t_idx + 1] if (t_idx >= 0 and t_idx + 1 < len(times)) else None
        is_first = t_ms == ds_first_frame.get(ds_id)

        frame_est = est_by_frame.get((ds_id, t_ms), pd.DataFrame())
        prev_est  = est_by_frame.get((ds_id, prev_t), pd.DataFrame()) if prev_t else pd.DataFrame()
        next_est  = est_by_frame.get((ds_id, next_t), pd.DataFrame()) if next_t else pd.DataFrame()

        for _, gt_row in grp_gt.iterrows():
            gx, gy, g_yaw, g_label = gt_row["x"], gt_row["y"], gt_row["yaw"], gt_row["label"]
            d0, lbl0, yd0, empty0 = nearest_in_frame(frame_est, gx, gy, g_yaw, g_label)
            d_prev = nearest_in_frame(prev_est, gx, gy, g_yaw, g_label)[0] if len(prev_est) else np.nan
            d_next = nearest_in_frame(next_est, gx, gy, g_yaw, g_label)[0] if len(next_est) else np.nan
            candidates = [v for v in [d_prev, d_next] if not np.isnan(v)]
            min_nb = min(candidates) if candidates else np.nan
            results.append({
                "gt_key": gt_row["gt_key"],
                "nearest_dist_same": d0, "nearest_label": lbl0, "yaw_diff": yd0,
                "nearest_dist_prev": d_prev, "nearest_dist_next": d_next,
                "is_first_frame": is_first,
                "reason": classify_reason(d0, lbl0, empty0, g_label, is_first, min_nb),
            })

    reason_df = pd.DataFrame(results)

    base_cols = [
        "gt_key", "t4dataset_name", "t4dataset_id", "scenario_name", "frame_index",
        "unix_time", "uuid", "label", "x", "y", "z", "yaw", "dist",
        "visibility", "pointcloud_num", "length", "width", "height",
        "gt_speed", "vx", "vy", "cp_rank", "cp_total_frames", "tr_first_tp_ms", "delay_sec",
    ]
    db = degrade[base_cols].merge(
        reason_df[["gt_key","nearest_dist_same","nearest_label","yaw_diff",
                   "nearest_dist_prev","nearest_dist_next","is_first_frame","reason"]],
        on="gt_key", how="left",
    )

    tl_mask = db["reason"] == "truly_lost"
    db["tl_subclass"] = pd.Series(dtype="object")
    if tl_mask.any():
        t_ms_vals = (db.loc[tl_mask, "unix_time"] / 1e6).round().astype("int64")
        db.loc[tl_mask, "tl_subclass"] = [
            classify_tl_subclass(cp_rank, t_ms, tr_tp)
            for cp_rank, t_ms, tr_tp in zip(
                db.loc[tl_mask, "cp_rank"],
                t_ms_vals,
                db.loc[tl_mask, "tr_first_tp_ms"],
            )
        ]

    return db


def print_summary(db: pd.DataFrame) -> None:
    total = len(db)
    print(f"\n{'='*60}")
    print(f"デグレ総数: {total:,} フレーム / {db['uuid'].nunique():,} ユニーク物体")
    print(f"{'='*60}")

    print("\n--- 原因分類 ---")
    for r, c in db["reason"].value_counts().items():
        print(f"  {r:30s}: {c:5d} ({c/total*100:.1f}%)")

    print("\n--- ラベル別 ---")
    print(db["label"].value_counts().to_string())

    tl = db[db["reason"] == "truly_lost"].copy()
    print(f"\n--- truly_lost ({len(tl):,} frames, {len(tl)/total*100:.1f}%) ---")
    if len(tl) and "tl_subclass" in tl.columns and tl["tl_subclass"].notna().any():
        n_tl = len(tl)
        for s, c in tl["tl_subclass"].value_counts().items():
            print(f"  {s:30s}: {c:5d} ({c/n_tl*100:.1f}%)")
        id_rows = tl[tl["tl_subclass"] == "init_delay"]
        if len(id_rows):
            print(f"  init_delay delay: median={id_rows['delay_sec'].median():.0f}f, "
                  f"P75={id_rows['delay_sec'].quantile(.75):.0f}f")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Build tracking degradation database")
    p.add_argument("--cp",  required=True, help="rematched_<cp>.parquet")
    p.add_argument("--tr",  required=True, help="rematched_<tr>.parquet")
    p.add_argument("--det", required=True, help="detection_<tr>.parquet  (pre-rematch)")
    p.add_argument("--output", default="tracking_degrade_database.csv")
    args = p.parse_args()

    print("Building degradation database ...")
    db = build_degrade_db(args.cp, args.tr, args.det)
    print_summary(db)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    db.drop(columns=["gt_key"]).to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(db):,} rows)")


if __name__ == "__main__":
    main()
