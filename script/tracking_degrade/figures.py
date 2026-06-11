"""
Generate figures for the tracking degradation report.

Reads:
  {out_dir}/tracking_degrade_database.csv
  {out_dir}/rematched_{name_cp}.parquet
  {out_dir}/rematched_{name_tr}.parquet

Outputs 10 PNG figures to {out_dir}/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
})

MAIN_LABELS = ["car", "pedestrian", "truck", "bus", "bicycle"]
REASON_COLORS = {
    "truly_lost":           "#d62728",
    "temporal_gap":         "#ff7f0e",
    "far_miss_5to10m":      "#bcbd22",
    "close_miss_2to5m":     "#2ca02c",
    "yaw_or_score_issue":   "#9467bd",
    "empty_frame":          "#8c564b",
    "class_mismatch_close": "#e377c2",
    "init_empty_frame":     "#7f7f7f",
}
TL_SUBCLASS_COLORS = {
    "init_delay":             "#1f77b4",
    "brief_appearance":       "#aec7e8",
    "track_lost_no_recover":  "#d62728",
    "permanent_lost":         "#ff9896",
    "init_delay_past":        "#98df8a",
    "track_lost_reacquired":  "#2ca02c",
}
SUBCLASS_ORDER = [
    "init_delay", "track_lost_no_recover", "brief_appearance",
    "permanent_lost", "init_delay_past", "track_lost_reacquired",
]


def generate_figures(out_dir: Path, name_cp: str = "centerpoint", name_tr: str = "tracking") -> None:
    db_path = out_dir / "tracking_degrade_database.csv"
    cp_path = out_dir / f"rematched_{name_cp}.parquet"
    tr_path = out_dir / f"rematched_{name_tr}.parquet"

    for p in [db_path, cp_path, tr_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # ── Load ───────────────────────────────────────────────────────────────
    db = pd.read_csv(db_path)
    db["unix_time_ms_key"] = (db["unix_time"] / 1e6).round().astype("int64")
    dist_labels_6 = ["0-10m","10-20m","20-30m","30-50m","50-100m","100m+"]
    speed_bins   = [0, 0.5, 2.0, 8.0, 20.0]
    speed_labels = ["Stationary\n<0.5","Slow\n0.5-2","Medium\n2-8","Fast\n8+"]
    db["dist_bin"]  = pd.cut(db["dist"],     bins=[0,10,20,30,50,100,999], labels=dist_labels_6)
    db["speed_bin"] = pd.cut(db["gt_speed"], bins=speed_bins, labels=speed_labels)

    cp = pd.read_parquet(cp_path)
    tr = pd.read_parquet(tr_path)
    for d in [cp, tr]:
        d["unix_time_ms"] = (d["unix_time"] / 1e6).round().astype("int64")
        d["dist"] = np.sqrt(d["x"]**2 + d["y"]**2)
        d["gt_speed"] = np.sqrt(d["vx"]**2 + d["vy"]**2).clip(upper=d["gt_speed"].quantile(0.99) if "gt_speed" in d.columns else 15.0)

    cp_gt_tp = cp[(cp["source"]=="GT") & (cp["status"]=="TP")].copy()
    cp_gt_tp["gt_key"] = list(zip(cp_gt_tp["t4dataset_id"], cp_gt_tp["uuid"], cp_gt_tp["unix_time_ms"]))
    tr_fn_gt  = tr[(tr["source"]=="GT") & (tr["status"]=="FN")].copy()
    tr_fn_gt["gt_key"]  = list(zip(tr_fn_gt["t4dataset_id"],  tr_fn_gt["uuid"],  tr_fn_gt["unix_time_ms"]))
    tr_tp_gt  = tr[(tr["source"]=="GT") & (tr["status"]=="TP")].copy()
    tr_tp_gt["gt_key"]  = list(zip(tr_tp_gt["t4dataset_id"],  tr_tp_gt["uuid"],  tr_tp_gt["unix_time_ms"]))
    tr_fn_keys = set(tr_fn_gt["gt_key"])
    tr_tp_keys = set(tr_tp_gt["gt_key"])
    stable = cp_gt_tp[cp_gt_tp["gt_key"].isin(tr_tp_keys)].copy()
    degrade_rows = cp_gt_tp[cp_gt_tp["gt_key"].isin(tr_fn_keys)].copy()

    dist_labels5 = ["<20m","20-30m","30-50m","50-100m","100m+"]
    for d in [stable, degrade_rows]:
        d["dist_bin"]  = pd.cut(d["dist"],     bins=[0,20,30,50,100,999], labels=dist_labels5)
        d["speed_bin"] = pd.cut(d["gt_speed"], bins=speed_bins, labels=speed_labels)

    reason_lookup = db.set_index(["t4dataset_id","uuid","unix_time_ms_key"])["reason"].to_dict()
    degrade_rows["unix_time_ms_key"] = (degrade_rows["unix_time"] / 1e6).round().astype("int64")
    degrade_rows["reason"] = [
        reason_lookup.get((r.t4dataset_id, r.uuid, r.unix_time_ms_key), "unknown")
        for _, r in degrade_rows.iterrows()
    ]

    all_labeled = pd.concat([
        degrade_rows.assign(group="degrade"),
        stable.assign(reason="stable_tp", group="stable"),
    ])
    all_labeled["dist_bin2"]  = pd.cut(all_labeled["dist"], bins=[0,20,30,50,100,999], labels=dist_labels5)
    all_labeled["speed_bin2"] = pd.cut(all_labeled["gt_speed"], bins=speed_bins, labels=speed_labels)

    all_cp = pd.concat([degrade_rows.assign(degrade=1), stable.assign(degrade=0, reason="stable_tp")])
    all_cp["pc_bin"] = pd.cut(all_cp["pointcloud_num"],
        bins=[0,5,10,20,50,100,500,100000],
        labels=["1-5","6-10","11-20","21-50","51-100","101-500","500+"])
    all_cp["dist_bin2"]  = pd.cut(all_cp["dist"],     bins=[0,20,30,50,100,999], labels=dist_labels5)
    all_cp["speed_bin2"] = pd.cut(all_cp["gt_speed"], bins=speed_bins, labels=speed_labels)

    def _save(name: str):
        plt.tight_layout()
        p = out_dir / name
        plt.gcf().savefig(p, bbox_inches="tight")
        plt.close()
        print(f"  {name}")

    # ── Fig 1: reason donut ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    reason_cnt = db["reason"].value_counts()
    labels_pie = [f"{r}\n({c:,}, {c/len(db)*100:.1f}%)" for r, c in reason_cnt.items()]
    colors_pie = [REASON_COLORS.get(r, "#aec7e8") for r in reason_cnt.index]
    wedges, _ = ax.pie(reason_cnt.values, labels=None, colors=colors_pie,
                       startangle=140, wedgeprops=dict(width=0.55))
    ax.legend(wedges, labels_pie, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.set_title(f"Degradation Cause Classification\n(CP TP -> TR FN: {len(db):,} frames)")
    _save("fig_degrade_reason_pie.png")

    # ── Fig 2: stacked bar by label × reason ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    reasons_order = list(REASON_COLORS.keys())
    tbl = db[db["label"].isin(MAIN_LABELS)].groupby(["label","reason"]).size().unstack(fill_value=0)
    for r in reasons_order:
        if r not in tbl.columns: tbl[r] = 0
    tbl = tbl[reasons_order]
    bottom = np.zeros(len(tbl))
    x = np.arange(len(tbl))
    for r in reasons_order:
        vals = tbl[r].values
        ax.bar(x, vals, bottom=bottom, color=REASON_COLORS[r], label=r, width=0.6)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(tbl.index)
    ax.set_ylabel("Frames"); ax.set_title("Degradation Breakdown by Label")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{int(v):,}"))
    _save("fig_degrade_by_label_stacked.png")

    # ── Fig 3: degrade rate by distance ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, labs, title in [
        (axes[0], ["car","truck","bus"],        "Large/Medium Vehicles"),
        (axes[1], ["pedestrian","bicycle"],     "Pedestrian / Bicycle"),
    ]:
        for lab in labs:
            sub = all_cp[all_cp["label"]==lab]
            t = sub.groupby("dist_bin2", observed=True)["degrade"].agg(["sum","count"])
            rate = (t["sum"]/t["count"]*100).fillna(0).reindex(dist_labels5)
            ax.plot(dist_labels5, rate, marker="o", label=lab)
        ax.set_ylabel("Degrade Rate (%)"); ax.set_title(f"Degrade Rate by Distance: {title}")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, None)
    _save("fig_degrade_rate_by_distance.png")

    # ── Fig 4: degrade rate by PC count ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pc_labels2 = ["1-5","6-10","11-20","21-50","51-100","101-500","500+"]
    for ax, lab in zip(axes, ["car","pedestrian"]):
        sub = all_cp[all_cp["label"]==lab]
        t = sub.groupby("pc_bin", observed=True)["degrade"].agg(["sum","count"])
        rate = (t["sum"]/t["count"]*100).fillna(0).reindex(pc_labels2)
        bars = ax.bar(pc_labels2, rate, color="#1f77b4", alpha=0.8)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=7.5, padding=2)
        ax.set_xlabel("Point Cloud Count"); ax.set_ylabel("Degrade Rate (%)")
        ax.set_title(f"Degrade Rate by PC Count: {lab}")
        ax.set_ylim(0, 60); ax.grid(axis="y", alpha=0.3)
    _save("fig_degrade_rate_by_pc.png")

    # ── Fig 5: speed distribution degrade vs stable ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, lab, xlim in zip(axes, ["car","pedestrian"], [5.0, 3.0]):
        d = degrade_rows[degrade_rows["label"]==lab]["gt_speed"].clip(upper=xlim)
        s = stable[stable["label"]==lab]["gt_speed"].clip(upper=xlim)
        bins_v = np.linspace(0, xlim, 30)
        ax.hist(d, bins=bins_v, density=True, alpha=0.6, color="#d62728", label=f"degrade (n={len(d):,})")
        ax.hist(s, bins=bins_v, density=True, alpha=0.6, color="#2ca02c", label=f"stable TP (n={len(s):,})")
        ax.axvline(d.median(), color="#d62728", linestyle="--", lw=1.4, label=f"degrade med={d.median():.2f}")
        ax.axvline(s.median(), color="#2ca02c", linestyle="--", lw=1.4, label=f"stable med={s.median():.2f}")
        ax.set_xlabel("GT Speed (m/s)"); ax.set_ylabel("Density")
        ax.set_title(f"GT Speed Distribution: {lab}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save("fig_speed_distribution.png")

    # ── Fig 6: truly_lost rate by speed bin ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, lab in zip(axes, ["car","pedestrian"]):
        sub = all_labeled[all_labeled["label"]==lab]
        total = sub.groupby("speed_bin2", observed=True).size().reindex(speed_labels, fill_value=0)
        tl    = sub[sub["reason"]=="truly_lost"].groupby("speed_bin2", observed=True).size().reindex(speed_labels, fill_value=0)
        rate  = (tl / total.replace(0, np.nan) * 100).fillna(0)
        bars = ax.bar(speed_labels, rate, color="#d62728", alpha=0.85)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
        ax.set_xlabel("GT Speed (m/s)"); ax.set_ylabel("Truly Lost Rate (%)")
        ax.set_title(f"Truly Lost Rate by Speed: {lab}")
        ax.set_ylim(0, 60); ax.grid(axis="y", alpha=0.3)
    _save("fig_truly_lost_by_speed.png")

    # ── Fig 7: truly_lost dist × speed heatmap ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    dist_labels_hm = ["<20m","20-30m","30-50m","50-100m","100m+"]
    for ax, lab in zip(axes, ["car","pedestrian"]):
        sub = all_labeled[all_labeled["label"]==lab].copy()
        sub["db"] = pd.cut(sub["dist"],     bins=[0,20,30,50,100,999], labels=dist_labels_hm)
        sub["sb"] = pd.cut(sub["gt_speed"], bins=speed_bins, labels=speed_labels)
        total = sub.groupby(["db","sb"], observed=True).size().unstack(fill_value=0)
        tl    = sub[sub["reason"]=="truly_lost"].groupby(["db","sb"], observed=True).size().unstack(fill_value=0)
        rate  = (tl.reindex_like(total) / total.replace(0, np.nan) * 100).fillna(0)
        im = ax.imshow(rate.values, aspect="auto", cmap="Reds", vmin=0, vmax=60)
        ax.set_xticks(range(len(rate.columns))); ax.set_xticklabels(rate.columns, fontsize=8, rotation=15)
        ax.set_yticks(range(len(rate.index)));   ax.set_yticklabels(rate.index, fontsize=8)
        ax.set_xlabel("GT Speed (m/s)"); ax.set_ylabel("Distance")
        ax.set_title(f"Truly Lost Rate (%) Heatmap: {lab}")
        for i in range(rate.shape[0]):
            for j in range(rate.shape[1]):
                v = rate.values[i,j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 35 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04)
    _save("fig_truly_lost_heatmap.png")

    # ── Fig 8: reason count by distance ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(dist_labels_6)); w = 0.2
    for offset, reason in zip([-1.5,-0.5,0.5,1.5],
                               ["truly_lost","close_miss_2to5m","temporal_gap","far_miss_5to10m"]):
        vals = db[db["reason"]==reason]["dist_bin"].value_counts().reindex(dist_labels_6, fill_value=0)
        ax.bar(x + offset*w, vals, width=w, label=reason, color=REASON_COLORS[reason])
    ax.set_xticks(x); ax.set_xticklabels(dist_labels_6)
    ax.set_ylabel("Frames"); ax.set_title("Degrade Cause by Distance Band")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{int(v):,}"))
    _save("fig_degrade_reason_by_distance.png")

    # ── Fig 9: truly_lost subclass distribution ────────────────────────────
    if "tl_subclass" not in db.columns or db["tl_subclass"].isna().all():
        print("  [skip] fig_truly_lost_subclass.png — tl_subclass column missing")
    else:
        tl_db = db[db["reason"] == "truly_lost"].copy()
        total_tl = len(tl_db)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors9 = [TL_SUBCLASS_COLORS.get(s, "#888") for s in SUBCLASS_ORDER]

        for ax, (title, get_cnt) in zip(axes, [
            ("Frame Level", lambda: tl_db["tl_subclass"].value_counts().reindex(SUBCLASS_ORDER).fillna(0)),
            ("Object Level", lambda: tl_db.groupby("uuid")["tl_subclass"].agg(lambda x: x.value_counts().index[0]).value_counts().reindex(SUBCLASS_ORDER).fillna(0)),
        ]):
            cnt = get_cnt()
            n = total_tl if "Frame" in title else tl_db["uuid"].nunique()
            bars = ax.barh(SUBCLASS_ORDER[::-1], cnt.values[::-1], color=colors9[::-1])
            for bar, v in zip(bars, cnt.values[::-1]):
                ax.text(bar.get_width() + cnt.max()*0.01,
                        bar.get_y() + bar.get_height()/2,
                        f"{int(v):,} ({v/n*100:.1f}%)", va="center", fontsize=8)
            ax.set_xlabel("Frames" if "Frame" in title else "Unique Objects")
            ax.set_title(f"Truly Lost Subclass: {title}\n(total {n:,})")
            ax.set_xlim(0, cnt.max() * 1.4)
            ax.grid(axis="x", alpha=0.3)
        _save("fig_truly_lost_subclass.png")

    # ── Fig 10: init_delay histogram ──────────────────────────────────────
    if "tl_subclass" not in db.columns or db["tl_subclass"].isna().all():
        print("  [skip] fig_init_delay_histogram.png")
    else:
        id_rows = db[db["tl_subclass"] == "init_delay"]
        if len(id_rows) == 0:
            print("  [skip] fig_init_delay_histogram.png — no init_delay rows")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            delay_clip = id_rows["delay_sec"].clip(upper=15)
            bins_d = np.arange(0.5, 16.5, 1)
            axes[0].hist(delay_clip, bins=bins_d, color="#1f77b4", alpha=0.85, edgecolor="white")
            axes[0].axvline(id_rows["delay_sec"].median(), color="red", linestyle="--", lw=1.5,
                            label=f"median={id_rows['delay_sec'].median():.0f} frame")
            axes[0].axvline(id_rows["delay_sec"].quantile(0.75), color="orange", linestyle="--", lw=1.5,
                            label=f"P75={id_rows['delay_sec'].quantile(0.75):.0f} frames")
            axes[0].set_xlabel("Delay (frames until TR becomes TP)")
            axes[0].set_ylabel("Frames")
            axes[0].set_title(f"init_delay: Delay Distribution\n(N={len(id_rows):,}, clipped at 15)")
            axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)

            for lab, color in [("car","#1f77b4"),("pedestrian","#ff7f0e")]:
                sub = id_rows[id_rows["label"]==lab]["delay_sec"].clip(upper=15)
                axes[1].hist(sub, bins=bins_d, alpha=0.6, color=color,
                             label=f"{lab} (n={len(sub):,})", edgecolor="white")
            axes[1].set_xlabel("Delay (frames until TR becomes TP)")
            axes[1].set_ylabel("Frames")
            axes[1].set_title("init_delay: Delay by Label")
            axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=0.3)
            _save("fig_init_delay_histogram.png")

    print(f"Done — all figures saved to {out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Generate tracking degrade report figures")
    p.add_argument("--output-dir", required=True, help="Directory with DB CSV and parquets")
    p.add_argument("--name-cp", default="centerpoint", help="CP model name (default: centerpoint)")
    p.add_argument("--name-tr", default="tracking",    help="TR model name (default: tracking)")
    args = p.parse_args()
    generate_figures(Path(args.output_dir), args.name_cp, args.name_tr)


if __name__ == "__main__":
    main()
