from __future__ import annotations

from typing import Iterable, Sequence

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
