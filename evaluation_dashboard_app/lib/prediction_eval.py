from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


DISTANCE_BIN_LABELS: list[str] = [
    "0-20 m",
    "20-40 m",
    "40-60 m",
    "60-80 m",
    "80-100 m",
    "100-120 m",
    "120-140 m",
    "140-160 m",
    "160-180 m",
    "180-200 m",
    "200+ m",
]


def actor_bucket(label: str | None) -> str:
    value = str(label or "").strip().lower()
    if value in {"car", "truck", "bus", "trailer"}:
        return "vehicle"
    if value == "pedestrian":
        return "pedestrian"
    if value in {"bicycle", "motorbike", "motorcycle"}:
        return "bicycle"
    return "other"


def infer_scenario_context(name: str | None) -> str:
    text = str(name or "").strip().lower().replace("_", " ").replace("-", " ")
    if any(token in text for token in ("crosswalk", "crossing", "jaywalk")):
        return "crossing"
    if any(token in text for token in ("merge", "ramp")):
        return "merge"
    if any(token in text for token in ("same lane", "follow", "following")):
        return "same-lane"
    if any(token in text for token in ("left turn", "right turn", "uturn", "u turn", "turn")):
        return "turning"
    if any(token in text for token in ("cut in", "cutin", "lane change", "overtake")):
        return "cut-in"
    return "other"


def _metric_label(prefix: str, checkpoint: float | int) -> str:
    if float(checkpoint).is_integer():
        checkpoint = int(checkpoint)
    return f"{prefix}@{checkpoint}s"


def _distance_bin(value: float | int | None) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    edges = list(range(0, 201, 20))
    for start, end, label in zip(edges[:-1], edges[1:], DISTANCE_BIN_LABELS[:-1]):
        if start <= float(value) < end:
            return label
    return DISTANCE_BIN_LABELS[-1]


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _noop_progress(_: float, __: str) -> None:
    return None


def _parse_r_upper_bound(label: object) -> float:
    text = str(label)
    try:
        return float(text.split("-")[-1])
    except ValueError:
        return float("inf")


def _bin_polar_for_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Assign analyzer-compatible polar bins for pandas inputs."""
    if df.empty:
        return df
    if not {"x", "y"}.issubset(df.columns):
        raise KeyError("Columns 'x' and 'y' are required.")
    try:
        from perception_catalog_analyzer.constants import (
            R_EDGES,
            R_LABELS,
            THETA_EDGES_DEG,
            THETA_INI,
            THETA_LABELS,
        )
    except Exception:
        r_edges = np.arange(0, 220, 20)
        r_labels = [f"{i}-{i + 20}" for i in range(0, 200, 20)]
        theta_ini = -60
        theta_edges_deg = np.arange(theta_ini, theta_ini + 360 + 60, 60)
        theta_labels = [f"{i}-{i + 60}" for i in range(theta_ini, theta_ini + 360, 60)]
    else:
        r_edges = R_EDGES
        r_labels = R_LABELS
        theta_ini = THETA_INI
        theta_edges_deg = THETA_EDGES_DEG
        theta_labels = THETA_LABELS

    out = df.copy()
    out["r_val"] = np.sqrt(out["x"].pow(2) + out["y"].pow(2))
    out["theta_val"] = ((np.degrees(np.arctan2(out["y"], out["x"])) - theta_ini) % 360) + theta_ini
    out["r"] = pd.cut(out["r_val"], bins=r_edges, labels=r_labels, right=False)
    out["theta"] = pd.cut(out["theta_val"], bins=theta_edges_deg, labels=theta_labels, right=False)
    return out


def prepare_future_matched_df(
    future_df: pd.DataFrame,
    *,
    time_step: float = 1.0,
    coord_abs_limit: float = 1e6,
    max_error_m: float = 200.0,
) -> pd.DataFrame:
    df = future_df.copy()
    df = _ensure_numeric(df, ("frame_index", "relative_time", "x", "y", "tx", "ty", "mode"))
    df["frame_index_num"] = df["frame_index"]
    df["aligned_horizon_sec"] = (df["relative_time"] / max(time_step, 1e-6)).round() * time_step
    if {"x", "y"}.issubset(df.columns):
        df["start_distance_m"] = np.sqrt(df["x"].pow(2) + df["y"].pow(2))
    else:
        df["start_distance_m"] = np.sqrt(df["tx"].pow(2) + df["ty"].pow(2))

    key_cols = ["suite_name", "scenario_name", "frame_index_num"]

    gt = df[df["source"].astype(str).str.upper() == "GT"].copy()
    est = df[df["source"].astype(str).str.upper() == "EST"].copy()
    if "confidence" not in est.columns:
        est["confidence"] = np.nan

    gt_start = (
        gt.sort_values("relative_time")
        .groupby(key_cols + ["uuid"], dropna=False)
        .agg(start_distance_m=("start_distance_m", "first"))
        .reset_index()
        .rename(columns={"uuid": "uuid_gt"})
    )

    gt_h = (
        gt[key_cols + ["uuid", "aligned_horizon_sec", "tx", "ty", "label"]]
        .dropna(subset=["aligned_horizon_sec"])
        .drop_duplicates(key_cols + ["uuid", "aligned_horizon_sec"])
        .rename(
            columns={
                "uuid": "uuid_gt",
                "tx": "tx_f_gt",
                "ty": "ty_f_gt",
                "label": "label_gt",
            }
        )
    )
    est_h = (
        est[key_cols + ["uuid", "pair_uuid", "mode", "aligned_horizon_sec", "tx", "ty", "confidence"]]
        .dropna(subset=["aligned_horizon_sec"])
        .drop_duplicates(key_cols + ["uuid", "pair_uuid", "mode", "aligned_horizon_sec"])
        .rename(
            columns={
                "uuid": "uuid_est",
                "pair_uuid": "uuid_gt",
                "tx": "tx_f_est",
                "ty": "ty_f_est",
                "confidence": "confidence_est",
            }
        )
    )

    matched = est_h.merge(gt_h, on=key_cols + ["uuid_gt", "aligned_horizon_sec"], how="inner")
    if matched.empty:
        return matched

    matched = matched.merge(gt_start, on=key_cols + ["uuid_gt"], how="left")
    matched["track_key"] = (
        matched["scenario_name"].astype("string").fillna("")
        + "::"
        + matched["frame_index_num"].fillna(-1).astype(int).astype(str)
        + "::"
        + matched["uuid_est"].astype("string").fillna("")
    )
    matched["disp_error_m"] = np.sqrt(
        (matched["tx_f_est"] - matched["tx_f_gt"]).pow(2) + (matched["ty_f_est"] - matched["ty_f_gt"]).pow(2)
    )
    matched["is_coordinate_outlier"] = (
        matched[["tx_f_est", "ty_f_est", "tx_f_gt", "ty_f_gt"]].abs().gt(coord_abs_limit).any(axis=1)
    )
    matched["is_metric_outlier"] = matched["is_coordinate_outlier"] | matched["disp_error_m"].gt(max_error_m)
    matched["actor_bucket"] = matched["label_gt"].map(actor_bucket)
    matched["scenario_context"] = matched["scenario_name"].map(infer_scenario_context)
    return matched


def build_future_mode_track_summary(
    future_df: pd.DataFrame,
    *,
    checkpoints: Sequence[float] = (1.0, 2.0, 3.0),
    time_step: float = 1.0,
    coord_abs_limit: float = 1e6,
    max_error_m: float = 200.0,
) -> pd.DataFrame:
    matched = prepare_future_matched_df(
        future_df,
        time_step=time_step,
        coord_abs_limit=coord_abs_limit,
        max_error_m=max_error_m,
    )
    return build_future_mode_track_summary_from_matched(matched, checkpoints=checkpoints)


def build_future_mode_track_summary_from_matched(
    matched: pd.DataFrame,
    *,
    checkpoints: Sequence[float] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    if matched.empty:
        return pd.DataFrame(
            columns=[
                "track_key",
                "suite_name",
                "scenario_name",
                "frame_index_num",
                "uuid_gt",
                "uuid_est",
                "label_gt",
                "mode_count",
                "start_distance_m",
            ]
        )

    sane = matched[~matched["is_metric_outlier"]].copy()
    if sane.empty:
        sane = matched.copy()

    group_cols = ["track_key", "suite_name", "scenario_name", "frame_index_num", "uuid_gt", "uuid_est", "label_gt"]
    track_summary = (
        sane.groupby(group_cols, dropna=False)
        .agg(
            mode_count=("mode", "nunique"),
            start_distance_m=("start_distance_m", "first"),
            confidence_mean=("confidence_est", "mean"),
            horizon_max_sec=("aligned_horizon_sec", "max"),
        )
        .reset_index()
    )

    for checkpoint in checkpoints:
        upto = sane[sane["aligned_horizon_sec"] <= checkpoint].copy()
        if upto.empty:
            track_summary[_metric_label("minADE", checkpoint)] = np.nan
            track_summary[_metric_label("minFDE", checkpoint)] = np.nan
            continue
        ade_mode = (
            upto.groupby(group_cols + ["mode"], dropna=False)["disp_error_m"]
            .mean()
            .reset_index(name="ade_m")
        )
        fde_mode = (
            upto.sort_values("aligned_horizon_sec")
            .groupby(group_cols + ["mode"], dropna=False)
            .tail(1)[group_cols + ["mode", "disp_error_m"]]
            .rename(columns={"disp_error_m": "fde_m"})
        )
        best_ade = ade_mode.groupby(group_cols, dropna=False)["ade_m"].min().reset_index()
        best_fde = fde_mode.groupby(group_cols, dropna=False)["fde_m"].min().reset_index()
        track_summary = track_summary.merge(
            best_ade.rename(columns={"ade_m": _metric_label("minADE", checkpoint)}),
            on=group_cols,
            how="left",
        ).merge(
            best_fde.rename(columns={"fde_m": _metric_label("minFDE", checkpoint)}),
            on=group_cols,
            how="left",
        )

    track_summary["actor_bucket"] = track_summary["label_gt"].map(actor_bucket)
    track_summary["scenario_context"] = track_summary["scenario_name"].map(infer_scenario_context)
    track_summary["distance_bin"] = pd.Categorical(
        track_summary["start_distance_m"].map(_distance_bin),
        categories=DISTANCE_BIN_LABELS,
        ordered=True,
    )
    return track_summary


def build_best_mode_horizon_summary(matched_df: pd.DataFrame) -> pd.DataFrame:
    if matched_df.empty:
        return pd.DataFrame(
            columns=[
                "track_key",
                "aligned_horizon_sec",
                "disp_error_m",
                "scenario_name",
                "actor_bucket",
                "scenario_context",
            ]
        )
    sane = matched_df[~matched_df["is_metric_outlier"]].copy()
    if sane.empty:
        sane = matched_df.copy()
    idx = sane.groupby(["track_key", "aligned_horizon_sec"], dropna=False)["disp_error_m"].idxmin()
    out = sane.loc[idx, ["track_key", "aligned_horizon_sec", "disp_error_m", "scenario_name", "actor_bucket", "scenario_context"]].copy()
    out = out.sort_values(["track_key", "aligned_horizon_sec"]).reset_index(drop=True)
    return out


def build_future_mode_label_summary(
    track_summary: pd.DataFrame,
    *,
    checkpoints: Sequence[float] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    if track_summary.empty:
        return pd.DataFrame(columns=["Actor", "track_count", "mode_count_mean"])
    agg_map: dict[str, tuple[str, str]] = {
        "track_count": ("track_key", "nunique"),
        "mode_count_mean": ("mode_count", "mean"),
    }
    for checkpoint in checkpoints:
        agg_map[_metric_label("minADE", checkpoint)] = (_metric_label("minADE", checkpoint), "mean")
        agg_map[_metric_label("minFDE", checkpoint)] = (_metric_label("minFDE", checkpoint), "mean")
    out = (
        track_summary.groupby("label_gt", dropna=False)
        .agg(**agg_map)
        .reset_index()
        .rename(columns={"label_gt": "Actor"})
        .sort_values("track_count", ascending=False)
    )
    return out


def build_horizon_breakdown(
    matched_df: pd.DataFrame,
    *,
    checkpoints: Sequence[float] | None = None,
) -> pd.DataFrame:
    if matched_df.empty:
        return pd.DataFrame(columns=["metric", "value_m"])
    data = _ensure_numeric(matched_df, ("aligned_horizon_sec", "disp_error_m"))
    if checkpoints is None:
        checkpoints = tuple(sorted(x for x in data["aligned_horizon_sec"].dropna().unique() if x > 0))
    rows: list[dict[str, float | str]] = []
    for checkpoint in checkpoints:
        upto = data[data["aligned_horizon_sec"] <= checkpoint]
        if upto.empty:
            continue
        per_track = upto.groupby("track_key", dropna=False)["disp_error_m"].mean()
        rows.append({"metric": _metric_label("ADE", checkpoint), "value_m": float(per_track.mean())})
    final = (
        data.sort_values("aligned_horizon_sec")
        .groupby("track_key", dropna=False)
        .tail(1)["disp_error_m"]
    )
    rows.append({"metric": "FDE@final", "value_m": float(final.mean())})
    return pd.DataFrame(rows)


def enrich_track_summary(
    track_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    current_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    enriched = track_df.copy()
    current_lookup = None
    if current_df is not None and not current_df.empty:
        current = current_df.copy()
        current = _ensure_numeric(current, ("frame_index", "frame_index_num", "center_distance", "center_distance_f"))
        if "frame_index_num" not in current.columns:
            current["frame_index_num"] = current["frame_index"]
        if "uuid_gt" not in current.columns and "uuid" in current.columns:
            current["uuid_gt"] = current["uuid"]
        distance_col = "center_distance_f" if "center_distance_f" in current.columns else "center_distance"
        if distance_col in current.columns:
            current_lookup = current.rename(columns={distance_col: "current_distance_m"})[
                ["scenario_name", "frame_index_num", "uuid_gt", "current_distance_m"]
            ].drop_duplicates()

    matched_lookup = None
    if not matched_df.empty:
        matched = matched_df.copy()
        matched = _ensure_numeric(matched, ("frame_index_num", "tx_f_gt", "ty_f_gt", "start_distance_m"))
        if "start_distance_m" not in matched.columns and {"tx_f_gt", "ty_f_gt"}.issubset(matched.columns):
            matched["start_distance_m"] = np.sqrt(matched["tx_f_gt"].pow(2) + matched["ty_f_gt"].pow(2))
        cols = ["track_key", "start_distance_m"]
        if {"scenario_name", "frame_index_num", "uuid_gt"}.issubset(matched.columns):
            cols += ["scenario_name", "frame_index_num", "uuid_gt"]
        matched_lookup = matched[cols].drop_duplicates()

    if current_lookup is not None and {"scenario_name", "frame_index_num", "uuid_gt"}.issubset(enriched.columns):
        enriched = enriched.merge(current_lookup, on=["scenario_name", "frame_index_num", "uuid_gt"], how="left")
    else:
        enriched["current_distance_m"] = np.nan

    if matched_lookup is not None:
        join_cols = ["track_key"] if "track_key" in enriched.columns and "track_key" in matched_lookup.columns else []
        if not join_cols and {"scenario_name", "frame_index_num", "uuid_gt"}.issubset(enriched.columns) and {"scenario_name", "frame_index_num", "uuid_gt"}.issubset(matched_lookup.columns):
            join_cols = ["scenario_name", "frame_index_num", "uuid_gt"]
        if join_cols:
            enriched = enriched.merge(
                matched_lookup[join_cols + ["start_distance_m"]].drop_duplicates(),
                on=join_cols,
                how="left",
            )
        else:
            enriched["start_distance_m"] = np.nan
    elif "start_distance_m" not in enriched.columns:
        enriched["start_distance_m"] = np.nan

    if "start_distance_m_x" in enriched.columns:
        enriched["start_distance_m"] = enriched["current_distance_m"].combine_first(enriched["start_distance_m_x"])
        if "start_distance_m_y" in enriched.columns:
            enriched["start_distance_m"] = enriched["start_distance_m"].combine_first(enriched["start_distance_m_y"])
        enriched = enriched.drop(columns=[c for c in ("start_distance_m_x", "start_distance_m_y") if c in enriched.columns])
    else:
        enriched["start_distance_m"] = enriched["current_distance_m"].combine_first(enriched["start_distance_m"])

    label_col = "label_gt" if "label_gt" in enriched.columns else "label"
    enriched["actor_bucket"] = enriched[label_col].map(actor_bucket)
    enriched["scenario_context"] = enriched["scenario_name"].map(infer_scenario_context)
    enriched["distance_bin"] = pd.Categorical(
        enriched["start_distance_m"].map(_distance_bin),
        categories=DISTANCE_BIN_LABELS,
        ordered=True,
    )
    return enriched


def build_distance_bin_metrics(track_df: pd.DataFrame) -> pd.DataFrame:
    data = track_df.copy()
    if "distance_bin" in data.columns:
        data["distance_bin"] = pd.Categorical(data["distance_bin"], categories=DISTANCE_BIN_LABELS, ordered=True)
    else:
        data["distance_bin"] = pd.Categorical(data["start_distance_m"].map(_distance_bin), categories=DISTANCE_BIN_LABELS, ordered=True)

    grouped = (
        data.groupby("distance_bin", observed=False)
        .agg(
            count=("track_key", "nunique"),
            ade_m=("ade_m", "mean"),
            fde_m=("fde_m", "mean"),
            p90_fde_m=("fde_m", lambda s: s.quantile(0.90) if len(s.dropna()) else np.nan),
            p95_fde_m=("fde_m", lambda s: s.quantile(0.95) if len(s.dropna()) else np.nan),
        )
        .reset_index()
    )
    return grouped


def build_specsheet_aligned_prediction_artifacts(
    future_df: pd.DataFrame,
    *,
    checkpoints: Sequence[float] = (1.0, 3.0, 5.0),
    time_step: float = 0.1,
    max_error_m: float = 100.0,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    from perception_catalog_analyzer.specsheet.metrics import load_metrics
    try:
        from perception_catalog_analyzer.specsheet.metrics.functional import FUTURE_ARRAY_CACHE
    except ImportError:
        FUTURE_ARRAY_CACHE = None

    report = progress_callback or _noop_progress
    if FUTURE_ARRAY_CACHE is not None:
        FUTURE_ARRAY_CACHE.clear()
    metric_order = [_metric_label(prefix, checkpoint) for prefix in ("minADE", "minFDE") for checkpoint in checkpoints]
    metric_map = {metric.name: metric for metric in load_metrics(metric_order)}

    report(0.02, "Binning rows in the same polar grid used by the specsheet...")
    normalized_future = _ensure_numeric(
        future_df,
        ("frame_index", "relative_time", "x", "y", "tx", "ty", "mode", "confidence"),
    )
    required_future_cols = ["source", "label", "uuid", "pair_uuid", "frame_index", "relative_time", "tx", "ty"]
    present_required_cols = [col for col in required_future_cols if col in normalized_future.columns]
    if present_required_cols:
        normalized_future = normalized_future.dropna(subset=present_required_cols)
    normalized_future = normalized_future.sort_values(
        [col for col in ["label", "frame_index", "pair_uuid", "uuid", "mode", "relative_time"] if col in normalized_future.columns],
        kind="stable",
    ).reset_index(drop=True)

    binned_future = _bin_polar_for_pandas(normalized_future)
    if binned_future.empty:
        report(0.9, "No future rows were available after binning.")
        empty = pd.DataFrame()
        return {
            "label_summary": empty,
            "distance_summary": empty,
            "polar_summary": empty,
        }

    labels = sorted(str(v) for v in binned_future["label"].dropna().unique() if str(v).strip())
    total_labels = max(len(labels), 1)
    total_metrics = max(len(metric_order), 1)

    label_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    polar_rows: list[dict[str, object]] = []

    report(0.28, f"Found {len(labels)} labels to aggregate.")
    for label_idx, label_name in enumerate(labels, start=1):
        label_start = 0.3 + (0.52 * (label_idx - 1) / total_labels)
        label_end = 0.3 + (0.52 * label_idx / total_labels)
        report(label_start, f"Aggregating label `{label_name}` ({label_idx}/{total_labels})...")
        scoped = binned_future[binned_future["label"].astype(str) == label_name].copy()
        est_scoped = scoped[scoped["source"].astype(str).str.upper() == "EST"].copy()

        label_groups = list(scoped.groupby(["r", "theta"], observed=True))
        total_groups = max(len(label_groups), 1)
        for group_idx, ((r_name, theta_name), sub_df) in enumerate(label_groups, start=1):
            warmup_progress = label_start + ((label_end - label_start) * 0.35 * group_idx / total_groups)
            report(
                warmup_progress,
                f"Preparing label `{label_name}` ({label_idx}/{total_labels}) future arrays: bin `{r_name}` / `{theta_name}` ({group_idx}/{total_groups})...",
            )
            for metric_name in metric_order:
                metric = metric_map[metric_name]
                metric.apply(sub_df)

        row: dict[str, object] = {
            "label": label_name,
            "future_rows": int(est_scoped[["scenario_name", "frame_index", "uuid"]].drop_duplicates().shape[0])
            if {"scenario_name", "frame_index", "uuid"}.issubset(est_scoped.columns)
            else int(len(est_scoped)),
        }
        for metric_idx, metric_name in enumerate(metric_order, start=1):
            metric_progress = label_start + ((label_end - label_start) * (0.35 + (0.65 * metric_idx / total_metrics)))
            report(
                metric_progress,
                f"Aggregating label `{label_name}` ({label_idx}/{total_labels}), metric `{metric_name}` ({metric_idx}/{total_metrics})...",
            )
            metric = metric_map[metric_name]
            metric_df = metric.apply(scoped)
            each_bin_df = metric.get_each_bin(metric_df)
            around_df = metric.get_all_around(scoped).dropna(subset=[metric_name]).copy()
            near_mask = around_df["r"].map(_parse_r_upper_bound) <= 60.0
            near_values = around_df.loc[near_mask, metric_name].dropna()
            row[metric_name] = float(np.nanmean(near_values.to_numpy(dtype=float))) if not near_values.empty else None

            if not around_df.empty:
                for rec in around_df[["r", metric_name]].to_dict("records"):
                    distance_rows.append(
                        {
                            "label": label_name,
                            "metric": metric_name,
                            "r": rec["r"],
                            "value": rec[metric_name],
                        }
                    )

            polar_df = each_bin_df.dropna(subset=[metric_name]).copy()
            if not polar_df.empty:
                polar_df["label"] = label_name
                polar_df["metric"] = metric_name
                polar_df = polar_df.rename(columns={metric_name: "value"})
                polar_rows.extend(polar_df[["label", "metric", "r", "theta", "value"]].to_dict("records"))

        label_rows.append(row)

    label_summary = pd.DataFrame(label_rows)
    if not label_summary.empty:
        report(0.86, "Finalizing overall summary row...")
        total_rows = float(label_summary["future_rows"].sum())
        overall_row: dict[str, object] = {
            "label": "All",
            "future_rows": int(total_rows),
        }
        for metric_name in metric_order:
            valid = label_summary[["future_rows", metric_name]].dropna()
            if valid.empty or float(valid["future_rows"].sum()) <= 0:
                overall_row[metric_name] = None
            else:
                overall_row[metric_name] = float(
                    (valid["future_rows"] * valid[metric_name]).sum() / valid["future_rows"].sum()
                )
        label_summary = pd.concat([pd.DataFrame([overall_row]), label_summary], ignore_index=True)
    report(0.9, "Prediction summary tables are ready for cache save.")

    return {
        "label_summary": label_summary,
        "distance_summary": pd.DataFrame(distance_rows),
        "polar_summary": pd.DataFrame(polar_rows),
    }
