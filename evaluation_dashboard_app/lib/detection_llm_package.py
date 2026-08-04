"""Evidence tables and LLM analysis package builder for detection stats."""

import io
import json
import re
import zipfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from lib.detection_eval_sql import (
    sql_distance_bin_label_rates_from_eval_flat,
    sql_distance_bin_rates_from_eval_flat,
)


def report_int(v: Any) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"{int(round(float(v))):,}"


def report_nonempty_text(v: Any, fallback: str = "(not named)") -> str:
    if v is None or pd.isna(v):
        return fallback
    s = str(v).strip()
    return s if s else fallback


def report_label_metrics(con, view: str, filter_clause: str) -> pd.DataFrame:
    q = f"""
    WITH stats AS (
        SELECT
            COALESCE(CAST(label AS VARCHAR), '') AS label,
            COUNT(*) FILTER (WHERE source = 'GT' AND status IN ('TP', 'FN')) AS gt_total,
            COUNT(*) FILTER (WHERE source = 'GT' AND status = 'TP') AS tp,
            COUNT(*) FILTER (WHERE source = 'GT' AND status = 'FN') AS fn,
            COUNT(*) FILTER (WHERE source = 'EST' AND status IN ('TP', 'FP')) AS est_total,
            COUNT(*) FILTER (WHERE source = 'EST' AND status = 'TP') AS tp_est,
            COUNT(*) FILTER (WHERE source = 'EST' AND status = 'FP') AS fp
        FROM {view}
        WHERE {filter_clause}
        GROUP BY 1
    )
    SELECT
        label,
        gt_total,
        tp,
        fn,
        est_total,
        tp_est,
        fp,
        CASE WHEN gt_total > 0 THEN CAST(tp AS DOUBLE) / gt_total ELSE NULL END AS tpr,
        CASE WHEN est_total > 0 THEN CAST(fp AS DOUBLE) / est_total ELSE NULL END AS fpr,
        CASE WHEN est_total > 0 THEN CAST(tp_est AS DOUBLE) / est_total ELSE NULL END AS precision,
        CASE
            WHEN gt_total > 0 AND est_total > 0 AND (CAST(tp AS DOUBLE) / gt_total + CAST(tp_est AS DOUBLE) / est_total) > 0
            THEN 2 * (CAST(tp AS DOUBLE) / gt_total) * (CAST(tp_est AS DOUBLE) / est_total)
                 / ((CAST(tp AS DOUBLE) / gt_total) + (CAST(tp_est AS DOUBLE) / est_total))
            ELSE NULL
        END AS f1
    FROM stats
    ORDER BY label
    """
    return con.execute(q).df()


def report_scene_metrics(con, view: str, filter_clause: str) -> pd.DataFrame:
    q = f"""
    WITH stats AS (
        SELECT
            COALESCE(CAST(scenario_name AS VARCHAR), '') AS scenario_name,
            COALESCE(CAST(t4dataset_name AS VARCHAR), '') AS t4dataset_name,
            COALESCE(CAST(suite_name AS VARCHAR), '') AS suite_name,
            COALESCE(CAST(t4dataset_id AS VARCHAR), '') AS t4dataset_id,
            COUNT(*) FILTER (WHERE source = 'GT' AND status IN ('TP', 'FN')) AS gt_total,
            COUNT(*) FILTER (WHERE source = 'GT' AND status = 'TP') AS tp,
            COUNT(*) FILTER (WHERE source = 'GT' AND status = 'FN') AS fn,
            COUNT(*) FILTER (WHERE source = 'EST' AND status IN ('TP', 'FP')) AS est_total,
            COUNT(*) FILTER (WHERE source = 'EST' AND status = 'FP') AS fp
        FROM {view}
        WHERE {filter_clause}
        GROUP BY 1, 2, 3, 4
    )
    SELECT
        *,
        CASE WHEN gt_total > 0 THEN CAST(tp AS DOUBLE) / gt_total ELSE NULL END AS tpr,
        CASE WHEN gt_total > 0 THEN CAST(fn AS DOUBLE) / gt_total ELSE NULL END AS fn_rate,
        CASE WHEN est_total > 0 THEN CAST(fp AS DOUBLE) / est_total ELSE NULL END AS fpr
    FROM stats
    ORDER BY fn DESC, fn_rate DESC, fp DESC
    """
    return con.execute(q).df()


def report_fn_frames(con, view: str, filter_clause: str) -> pd.DataFrame:
    q = f"""
    SELECT
        COALESCE(CAST(t4dataset_id AS VARCHAR), '') AS t4dataset_id,
        CAST(frame_index AS VARCHAR) AS frame_index,
        COALESCE(MAX(CAST(scenario_name AS VARCHAR)), '') AS scenario_name,
        COALESCE(MAX(CAST(t4dataset_name AS VARCHAR)), '') AS t4dataset_name,
        COALESCE(MAX(CAST(suite_name AS VARCHAR)), '') AS suite_name,
        COUNT(*) AS fn
    FROM {view}
    WHERE source = 'GT' AND status = 'FN' AND frame_index IS NOT NULL AND {filter_clause}
    GROUP BY 1, 2
    ORDER BY fn DESC
    LIMIT 20
    """
    return con.execute(q).df()


def report_error_metrics(con, view: str, filter_clause: str) -> pd.DataFrame:
    try:
        sample_df = con.execute(f"SELECT * FROM {view} LIMIT 1").df()
    except Exception:
        return pd.DataFrame()
    if not all(c in sample_df.columns for c in ["x_error", "y_error", "yaw_error"]):
        return pd.DataFrame()
    q = f"""
    SELECT
        COALESCE(CAST(label AS VARCHAR), '') AS label,
        AVG(ABS(CAST(x_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND x_error IS NOT NULL) AS mean_abs_x_error,
        AVG(ABS(CAST(y_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND y_error IS NOT NULL) AS mean_abs_y_error,
        AVG(ABS(CAST(yaw_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND yaw_error IS NOT NULL) AS mean_abs_yaw_error
    FROM {view}
    WHERE {filter_clause}
    GROUP BY 1
    ORDER BY label
    """
    return con.execute(q).df()


def report_diff_by_label(
    con,
    base_view: str,
    comp_view: str,
    base_filter: str,
    comp_filter: str,
) -> pd.DataFrame:
    q = f"""
    WITH base_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COALESCE(MAX(CAST(label AS VARCHAR)), '') AS label,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base
        FROM {base_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {base_filter}
        GROUP BY 1, 2, 3
    ),
    comp_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COALESCE(MAX(CAST(label AS VARCHAR)), '') AS label,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp
        FROM {comp_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {comp_filter}
        GROUP BY 1, 2, 3
    ),
    joined AS (
        SELECT
            COALESCE(b.label, c.label, '') AS label,
            COALESCE(b.tp_base, FALSE) AS tp_base,
            COALESCE(c.tp_comp, FALSE) AS tp_comp
        FROM base_gt b
        FULL OUTER JOIN comp_gt c
            ON b.t4dataset_id = c.t4dataset_id
           AND b.frame_index = c.frame_index
           AND b.gt_uuid = c.gt_uuid
    )
    SELECT
        label,
        COUNT(*) AS total_gt,
        COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS improved_cnt,
        COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS degraded_cnt,
        COUNT(*) FILTER (WHERE tp_base AND tp_comp) AS both_tp_cnt,
        COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp) AS both_fn_cnt,
        SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS net_tp_delta
    FROM joined
    GROUP BY 1
    ORDER BY net_tp_delta DESC
    """
    return con.execute(q).df()


def report_diff_by_scene_or_frame(
    con,
    base_view: str,
    comp_view: str,
    base_filter: str,
    comp_filter: str,
    *,
    by_frame: bool,
) -> pd.DataFrame:
    frame_select = "COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index," if by_frame else ""
    frame_group = ", frame_index" if by_frame else ""
    q = f"""
    WITH base_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
            COALESCE(MAX(CAST(suite_name AS VARCHAR)), '') AS suite_name,
            COALESCE(MAX(CAST(scenario_name AS VARCHAR)), '') AS scenario_name,
            COALESCE(MAX(CAST(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
        FROM {base_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {base_filter}
        GROUP BY 1, 2, 3
    ),
    comp_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
            COALESCE(MAX(CAST(suite_name AS VARCHAR)), '') AS suite_name,
            COALESCE(MAX(CAST(scenario_name AS VARCHAR)), '') AS scenario_name,
            COALESCE(MAX(CAST(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
        FROM {comp_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {comp_filter}
        GROUP BY 1, 2, 3
    ),
    joined AS (
        SELECT
            COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR), '') AS t4dataset_id,
            {frame_select}
            COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
            COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
            COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name,
            COALESCE(b.tp_base, FALSE) AS tp_base,
            COALESCE(c.tp_comp, FALSE) AS tp_comp
        FROM base_gt b
        FULL OUTER JOIN comp_gt c
            ON b.t4dataset_id = c.t4dataset_id
           AND b.frame_index = c.frame_index
           AND b.gt_uuid = c.gt_uuid
    )
    SELECT
        t4dataset_id,
        {('frame_index,' if by_frame else '')}
        suite_name,
        scenario_name,
        t4dataset_name,
        COUNT(*) AS total_gt,
        COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS improved_cnt,
        COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS degraded_cnt,
        COUNT(*) FILTER (WHERE tp_base AND tp_comp) AS both_tp_cnt,
        COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp) AS both_fn_cnt,
        SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS net_tp_delta
    FROM joined
    GROUP BY t4dataset_id, suite_name, scenario_name, t4dataset_name{frame_group}
    ORDER BY degraded_cnt DESC, improved_cnt ASC, net_tp_delta ASC
    LIMIT 50
    """
    return con.execute(q).df()


def report_frame_ref(row: pd.Series) -> str:
    scen = report_nonempty_text(row.get("scenario_name"))
    t4 = report_nonempty_text(row.get("t4dataset_name"), "")
    fid = report_nonempty_text(row.get("frame_index"), "?")
    if t4 and t4 != scen:
        return f"{scen} / {t4}, frame {fid}"
    return f"{scen}, frame {fid}"


def report_label_distance_compare(
    con,
    base_view: str,
    candidate_view: str,
    base_filter: str,
    candidate_filter: str,
) -> pd.DataFrame:
    base = con.execute(sql_distance_bin_label_rates_from_eval_flat(base_view, base_filter)).df()
    cand = con.execute(sql_distance_bin_label_rates_from_eval_flat(candidate_view, candidate_filter)).df()
    if base.empty or cand.empty:
        return pd.DataFrame()
    merged = base.merge(cand, on=["label", "distance_bin"], suffixes=("_base", "_candidate"))
    if merged.empty:
        return merged
    merged["tpr_delta"] = merged["tpr_candidate"] - merged["tpr_base"]
    merged["fpr_delta"] = merged["fpr_candidate"] - merged["fpr_base"]
    return merged


def report_degraded_object_details(
    con,
    base_view: str,
    comp_view: str,
    base_filter: str,
    comp_filter: str,
) -> pd.DataFrame:
    q = f"""
    WITH base_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COALESCE(MAX(CAST(label AS VARCHAR)), '') AS label,
            MAX(dist_h) AS dist_h,
            COALESCE(MAX(CAST(visibility AS VARCHAR)), '') AS visibility,
            MAX(try_cast(pointcloud_num AS DOUBLE)) AS pointcloud_num,
            COALESCE(MAX(CAST(scenario_name AS VARCHAR)), '') AS scenario_name,
            COALESCE(MAX(CAST(t4dataset_name AS VARCHAR)), '') AS t4dataset_name,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base
        FROM {base_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {base_filter}
        GROUP BY 1, 2, 3
    ),
    comp_gt AS (
        SELECT
            t4dataset_id,
            frame_index,
            uuid AS gt_uuid,
            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp
        FROM {comp_view}
        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND {comp_filter}
        GROUP BY 1, 2, 3
    )
    SELECT
        CAST(b.t4dataset_id AS VARCHAR) AS t4dataset_id,
        CAST(b.frame_index AS VARCHAR) AS frame_index,
        b.gt_uuid,
        b.label,
        b.dist_h,
        b.visibility,
        b.pointcloud_num,
        b.scenario_name,
        b.t4dataset_name
    FROM base_gt b
    LEFT JOIN comp_gt c
        ON b.t4dataset_id = c.t4dataset_id
       AND b.frame_index = c.frame_index
       AND b.gt_uuid = c.gt_uuid
    WHERE b.tp_base AND NOT COALESCE(c.tp_comp, FALSE)
    ORDER BY b.dist_h ASC, b.pointcloud_num DESC NULLS LAST
    LIMIT 2000
    """
    try:
        return con.execute(q).df()
    except Exception:
        return pd.DataFrame()


def report_critical_case_phrases(df_degraded_objects: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    if df_degraded_objects.empty:
        return [], pd.DataFrame()
    d = df_degraded_objects.copy()
    vis = d["visibility"].fillna("").astype(str).str.upper()
    critical = d[
        (d["dist_h"].fillna(1e9) <= 20.0)
        & (vis.isin(["FULL", "MOST"]))
        & (d["pointcloud_num"].fillna(0) >= 20)
    ].copy()
    if critical.empty:
        return ["20m以内・FULL/MOST・点群20点以上に該当する安全クリティカルなデグレは検出されませんでした。"], critical
    phrases = []
    for _, r in critical.head(3).iterrows():
        uuid_s = report_nonempty_text(r.get("gt_uuid"), "")[:8]
        phrases.append(
            f"{report_nonempty_text(r.get('label'), '(no label)')} / "
            f"{float(r.get('dist_h', 0.0)):.1f}m / "
            f"点群{report_int(r.get('pointcloud_num'))} / "
            f"{report_frame_ref(r)} / uuid={uuid_s}"
        )
    return phrases, critical


def report_consecutive_failure_phrases(df_degraded_objects: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    if df_degraded_objects.empty:
        return [], pd.DataFrame()
    d = df_degraded_objects.copy()
    grouped = (
        d.groupby(["t4dataset_id", "gt_uuid", "label", "scenario_name"], dropna=False)
        .agg(
            degraded_frames=("frame_index", "nunique"),
            min_dist=("dist_h", "min"),
            max_pointcloud=("pointcloud_num", "max"),
        )
        .reset_index()
        .sort_values(["degraded_frames", "max_pointcloud"], ascending=[False, False])
    )
    if grouped.empty:
        return [], grouped
    phrases = []
    for _, r in grouped.head(3).iterrows():
        uuid_s = report_nonempty_text(r.get("gt_uuid"), "")[:8]
        phrases.append(
            f"{report_nonempty_text(r.get('label'), '(no label)')} / "
            f"{report_nonempty_text(r.get('scenario_name'))} / "
            f"{report_int(r.get('degraded_frames'))} frames / "
            f"min {float(r.get('min_dist', 0.0)):.1f}m / uuid={uuid_s}"
        )
    return phrases, grouped


def _report_fp_diff_by_group(
    con,
    base_view: str,
    candidate_view: str,
    base_filter: str,
    candidate_filter: str,
    *,
    group_cols: List[str],
) -> pd.DataFrame:
    select_cols = ",\n                ".join(group_cols)
    group_by = ", ".join(str(i + 1) for i in range(len(group_cols)))
    join_cols = " AND ".join(
        [f"COALESCE(CAST(b.{c} AS VARCHAR), '') = COALESCE(CAST(c.{c} AS VARCHAR), '')" for c in group_cols]
    )
    output_cols = ",\n            ".join(
        [f"COALESCE(CAST(b.{c} AS VARCHAR), CAST(c.{c} AS VARCHAR), '') AS {c}" for c in group_cols]
    )
    return con.execute(
        f"""
        WITH base_stats AS (
            SELECT
                {select_cols},
                COUNT(*) FILTER (WHERE status = 'FP') AS fp_base,
                COUNT(*) FILTER (WHERE status = 'TP') AS tp_base,
                COUNT(*) AS est_base
            FROM {base_view}
            WHERE source = 'EST' AND {base_filter}
            GROUP BY {group_by}
        ),
        candidate_stats AS (
            SELECT
                {select_cols},
                COUNT(*) FILTER (WHERE status = 'FP') AS fp_candidate,
                COUNT(*) FILTER (WHERE status = 'TP') AS tp_candidate,
                COUNT(*) AS est_candidate
            FROM {candidate_view}
            WHERE source = 'EST' AND {candidate_filter}
            GROUP BY {group_by}
        )
        SELECT
            {output_cols},
            CAST(COALESCE(b.fp_base, 0) AS DOUBLE) AS baseline_fp,
            CAST(COALESCE(c.fp_candidate, 0) AS DOUBLE) AS candidate_fp,
            CAST(COALESCE(c.fp_candidate, 0) - COALESCE(b.fp_base, 0) AS DOUBLE) AS fp_delta,
            CAST(COALESCE(b.tp_base, 0) AS DOUBLE) AS baseline_tp_est,
            CAST(COALESCE(c.tp_candidate, 0) AS DOUBLE) AS candidate_tp_est,
            CAST(COALESCE(c.tp_candidate, 0) - COALESCE(b.tp_base, 0) AS DOUBLE) AS tp_est_delta,
            CAST(COALESCE(b.est_base, 0) AS DOUBLE) AS baseline_est_total,
            CAST(COALESCE(c.est_candidate, 0) AS DOUBLE) AS candidate_est_total
        FROM base_stats b
        FULL OUTER JOIN candidate_stats c
            ON {join_cols}
        ORDER BY fp_delta DESC
        """
    ).df()


def report_fp_diff_tables(
    con,
    base_view: str,
    candidate_view: str,
    base_filter: str,
    candidate_filter: str,
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    specs = [
        ("FP diff by label", ["label"]),
        ("FP diff by scenario", ["suite_name", "scenario_name"]),
        ("FP diff by dataset", ["suite_name", "scenario_name", "t4dataset_id", "t4dataset_name"]),
        ("FP diff by frame", ["suite_name", "scenario_name", "t4dataset_id", "t4dataset_name", "frame_index"]),
    ]
    for name, group_cols in specs:
        try:
            tables[name] = _report_fp_diff_by_group(
                con,
                base_view,
                candidate_view,
                base_filter,
                candidate_filter,
                group_cols=group_cols,
            )
        except Exception:
            tables[name] = pd.DataFrame()
    return tables


def safe_export_filename(value: str, *, fallback: str = "table") -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return name[:80] or fallback


def _brief_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    return str(v)


def _brief_table(df: pd.DataFrame, columns: List[str], *, limit: int = 12) -> str:
    if df is None or df.empty:
        return "(no rows)"
    cols = [c for c in columns if c in df.columns]
    if not cols:
        cols = [str(c) for c in df.columns[:8]]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.head(limit).iterrows():
        lines.append("| " + " | ".join(_brief_value(row.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def _sorted_preview(df: pd.DataFrame, sort_col: str, *, ascending: bool = False, limit: int = 12) -> pd.DataFrame:
    if df is None or df.empty or sort_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[sort_col] = pd.to_numeric(out[sort_col], errors="coerce")
    return out.sort_values(sort_col, ascending=ascending).head(limit)


def llm_data_brief(metadata: Dict[str, Any], tables: Dict[str, pd.DataFrame]) -> str:
    comparison = metadata.get("comparison", "")
    scope = metadata.get("scope", "")
    filters = metadata.get("filters", {})
    kpis = metadata.get("kpis", {})

    sections = [
        "# Detection Analysis Data Brief",
        "",
        "This file is a neutral evidence brief for an LLM. The dashboard has already extracted useful metrics and tables from raw detection data.",
        "The LLM should focus on interpretation, comparison, explanation, and report writing based on these prepared evidence tables.",
        "",
        "## Context",
        f"- Comparison: {comparison}",
        f"- Scope: {scope}",
        f"- Filters: {json.dumps(filters, ensure_ascii=False, default=str)}",
        f"- KPI snapshot: {json.dumps(kpis, ensure_ascii=False, default=str)}",
        "",
        "## Table Inventory",
    ]
    for name, df in tables.items():
        rows = 0 if df is None else len(df)
        cols = [] if df is None else [str(c) for c in df.columns]
        sections.append(f"- {name}: {rows} rows; columns: {', '.join(cols[:30])}")

    class_cmp = tables.get("Class rate comparison", pd.DataFrame())
    if not class_cmp.empty:
        sections.extend(
            [
                "",
                "## Class-Level Signal Preview",
                "",
                "Largest TPR gains:",
                _brief_table(
                    _sorted_preview(class_cmp, "tpr_delta", ascending=False),
                    ["label", "tpr_base", "tpr_candidate", "tpr_delta", "precision_delta", "f1_delta", "gt_total_base", "gt_total_candidate"],
                ),
                "",
                "Largest TPR regressions:",
                _brief_table(
                    _sorted_preview(class_cmp, "tpr_delta", ascending=True),
                    ["label", "tpr_base", "tpr_candidate", "tpr_delta", "precision_delta", "f1_delta", "gt_total_base", "gt_total_candidate"],
                ),
            ]
        )

    obj_diff = tables.get("Object diff by class", pd.DataFrame())
    if not obj_diff.empty:
        sections.extend(
            [
                "",
                "## Object-Level TP/FN Change Preview",
                "",
                "Most recovered labels:",
                _brief_table(
                    _sorted_preview(obj_diff, "improved_cnt", ascending=False),
                    ["label", "improved_cnt", "degraded_cnt", "net_tp_delta"],
                ),
                "",
                "Most degraded labels:",
                _brief_table(
                    _sorted_preview(obj_diff, "degraded_cnt", ascending=False),
                    ["label", "improved_cnt", "degraded_cnt", "net_tp_delta"],
                ),
            ]
        )

    fp_label = tables.get("FP diff by label", pd.DataFrame())
    fp_scenario = tables.get("FP diff by scenario", pd.DataFrame())
    fp_dataset = tables.get("FP diff by dataset", pd.DataFrame())
    if not fp_label.empty or not fp_scenario.empty or not fp_dataset.empty:
        sections.extend(["", "## False Positive Concentration Preview"])
        if not fp_label.empty:
            sections.extend(
                [
                    "",
                    "Labels with largest FP increase:",
                    _brief_table(
                        _sorted_preview(fp_label, "fp_delta", ascending=False),
                        ["label", "baseline_fp", "candidate_fp", "fp_delta", "baseline_tp_est", "candidate_tp_est", "tp_est_delta"],
                    ),
                ]
            )
        if not fp_scenario.empty:
            sections.extend(
                [
                    "",
                    "Scenarios with largest FP increase:",
                    _brief_table(
                        _sorted_preview(fp_scenario, "fp_delta", ascending=False),
                        ["suite_name", "scenario_name", "baseline_fp", "candidate_fp", "fp_delta", "baseline_est_total", "candidate_est_total"],
                    ),
                ]
            )
        if not fp_dataset.empty:
            sections.extend(
                [
                    "",
                    "Datasets with largest FP increase:",
                    _brief_table(
                        _sorted_preview(fp_dataset, "fp_delta", ascending=False),
                        ["suite_name", "scenario_name", "t4dataset_name", "baseline_fp", "candidate_fp", "fp_delta"],
                    ),
                ]
            )

    dist_base = tables.get("Distance rates - baseline", pd.DataFrame())
    dist_candidate = tables.get("Distance rates - candidate", pd.DataFrame())
    if not dist_base.empty or not dist_candidate.empty:
        sections.extend(
            [
                "",
                "## Distance Rate Tables Preview",
                "",
                "Baseline:",
                _brief_table(dist_base, ["distance_bin", "tpr", "fpr"], limit=20),
                "",
                "Candidate:",
                _brief_table(dist_candidate, ["distance_bin", "tpr", "fpr"], limit=20),
            ]
        )

    scene_diff = tables.get("Object diff by scene", pd.DataFrame())
    if not scene_diff.empty:
        sections.extend(
            [
                "",
                "## Scenario Hotspot Preview",
                "",
                "Largest TP/FN degraded scenarios:",
                _brief_table(
                    _sorted_preview(scene_diff, "degraded_cnt", ascending=False),
                    ["suite_name", "scenario_name", "improved_cnt", "degraded_cnt", "net_tp_delta"],
                ),
                "",
                "Largest TP/FN improved scenarios:",
                _brief_table(
                    _sorted_preview(scene_diff, "improved_cnt", ascending=False),
                    ["suite_name", "scenario_name", "improved_cnt", "degraded_cnt", "net_tp_delta"],
                ),
            ]
        )

    critical = tables.get("Critical degraded cases", pd.DataFrame())
    if not critical.empty:
        sections.extend(
            [
                "",
                "## Safety-Critical Degraded Case Preview",
                _brief_table(
                    critical,
                    ["label", "dist_h", "visibility", "pointcloud_num", "scenario_name", "t4dataset_name", "frame_index", "gt_uuid"],
                    limit=20,
                ),
            ]
        )

    sections.extend(
        [
            "",
            "## Analysis Boundary",
            "The app is responsible for rule-based extraction from raw data: KPI tables, class/distance/scenario aggregations, FP/TP/FN counts, and case lists.",
            "The LLM is responsible for higher-level analysis: explaining what the prepared evidence means, connecting signals across tables, judging trade-offs, and writing a clear report.",
            "",
            "## Report Style Reference",
            "Prefer a clear comparison structure:",
            "- comparison target metadata",
            "- overall KPI trend",
            "- distance trend",
            "- class trend",
            "- TP improvement hotspots with scenario/dataset/frame evidence",
            "- FP increase hotspots with concentration percentages",
            "- likely causes and concrete inspection links/cases",
            "- final judgment and recommended next checks",
        ]
    )
    return "\n".join(sections)


def llm_report_instructions(metadata: Dict[str, Any], tables: Dict[str, pd.DataFrame]) -> str:
    table_lines = []
    for name, df in tables.items():
        rows = 0 if df is None else len(df)
        cols = [] if df is None else [str(c) for c in df.columns]
        table_lines.append(f"- {name}: {rows} rows; columns: {', '.join(cols[:24])}")

    mode_label = metadata.get("mode", "Detection Stats")
    comparison = metadata.get("comparison", "")
    scope = metadata.get("scope", "")
    filters = metadata.get("filters", {})

    return f"""# LLM Instructions: Detection Performance Analysis Report

You are an expert autonomous-driving perception evaluation analyst. Use the attached CSV tables and `analysis_data_brief.md` to create a polished, graph-rich performance report.

The dashboard app has already extracted useful structured evidence from raw detection data. Treat these tables as the primary source of truth. Do not spend the report mostly re-computing obvious table results; focus on interpreting the prepared evidence, explaining trade-offs, connecting patterns across KPI/class/distance/scenario tables, and writing a high-quality report.

Do not copy or imitate any rule-based dashboard report. Build your conclusions from the CSV evidence.

## Evaluation Context
- Dashboard page: {mode_label}
- Comparison: {comparison}
- Scope: {scope}
- Active filters: {json.dumps(filters, ensure_ascii=False, default=str)}

## Required Output
Create a structured report in Markdown or HTML. The report must include:

1. Executive summary with a clear release/readiness judgment.
2. KPI comparison table covering TP, FP, FN, recall/TP rate, FP rate, precision, and F1 where available.
3. Performance comparison narrative explaining whether the candidate improved, degraded, or traded recall for precision.
4. Many graphs, not just text. Generate charts directly from the CSV files.
5. Class-level analysis showing top improving and degrading labels.
6. Distance-range analysis showing where recall/FPR changes occur.
7. Scenario/frame hotspot analysis explaining concentrated failures.
8. Safety-critical case analysis, especially near-range TP-to-FN degradations.
9. Localization quality analysis when mean error tables are available.
10. Final recommendation with concrete next debugging actions.

## Required Graphs
Include at least these charts when the corresponding tables exist:

- KPI delta bar chart: recall, precision, F1, FP, FN.
- Class TPR delta bar chart sorted from largest gain to largest loss.
- Class precision/F1 comparison chart.
- Object diff by class chart showing improved vs degraded counts.
- Distance-bin line chart comparing baseline and candidate TPR/FPR.
- Label x distance heatmap for TPR/FPR deltas.
- Scenario or frame hotspot bar chart for degraded counts.
- Critical degraded cases table with distance, visibility, points count, scenario, dataset, and frame.
- Mean localization error delta chart for x, y, and yaw when available.

## Analysis Rules
- Treat the first run as the baseline and each later run as a candidate.
- Use baseline vs candidate deltas; do not judge from candidate values alone.
- A higher recall/TP rate, precision, and F1 is better.
- A lower FP count/rate, FN count, and localization error is better.
- Distinguish headline KPI stability from object-level churn; stable F1 can hide many FN-to-TP and TP-to-FN swaps.
- Prioritize near-distance, high-visibility, high-point-count degradations as safety-critical.
- If a table is empty, say that the signal is unavailable instead of inventing data.
- Cite the table names used for each major conclusion.
- Where FP increases are concentrated, quantify concentration, e.g. "top 20 datasets explain X/Y FP increase", when the table supports it.
- Use cautious language for causes. Separate evidence-backed observations from hypotheses such as annotation gaps, parked-object density, or low point count.
- Include scenario/dataset/frame examples, especially when `open_3d` or dataset/frame columns are available.
- Avoid filling the report with generic calculations that are already obvious from the tables. Use calculations only when they support interpretation, concentration analysis, or clearer comparison.

## Available Tables
{chr(10).join(table_lines)}
"""


def llm_report_blueprint() -> str:
    return """# Recommended Report Blueprint

Use this as a structure, not as fixed wording.

## 1. Comparison Target
- Baseline model/run
- Candidate model/run
- Topic, labels, suites, visibility, distance scope
- Important caveats about filters or missing tables

## 2. Executive Summary
- One-sentence verdict
- Recall/TP rate movement
- Precision/FP movement
- Whether the result is clear improvement, clear regression, or recall/precision trade-off
- Most important risk to inspect before release

## 3. Overall KPI Dashboard
Required visuals:
- KPI table
- Delta bar chart for TP, FP, FN, recall, precision, F1
- Short explanation of what changed and why it matters

## 4. Distance Analysis
Required visuals:
- TPR and FPR by distance-bin line charts
- Optional TPR/FPR delta bar chart by distance
Explain near, mid, and far range separately.

## 5. Class Analysis
Required visuals:
- TPR delta by class
- Precision/F1 delta by class
- Improved vs degraded object counts by class
Call out classes that improved in recall but caused FP growth.

## 6. TP Improvement Hotspots
Required visuals:
- Top scenarios/datasets/frames by improved count
- Table with 3D viewer links when available
Explain whether improvements are broad or concentrated.

## 7. FP Increase Hotspots
Required visuals:
- Top labels/scenarios/datasets/frames by FP delta
- Concentration chart: cumulative FP delta share for top N scenarios/datasets
Quantify concentration, such as top 20 datasets explaining X% of FP increase.

## 8. Safety-Critical Regressions
Required visuals:
- Critical degraded cases table
- Near-distance degraded count chart if data exists
Prioritize close range, high visibility, high point count, and repeated failures.

## 9. Localization Quality
Required visuals:
- Mean x/y/yaw error comparison or delta chart
Separate detection-rate changes from localization changes.

## 10. Root-Cause Hypotheses
Separate evidence from hypotheses. Examples:
- Dense parked-object scenes
- Annotation gaps
- Low point-count objects
- Far-distance sparsity
- Specific class confusion

## 11. Recommendation
- Release/readiness judgment
- Must-check scenarios and viewer links
- Suggested next experiments or data review actions
"""


def llm_prompt_to_paste() -> str:
    return """Please unzip and read this analysis package.

First read:
1. README.md
2. llm_instructions.md
3. recommended_report_blueprint.md
4. analysis_data_brief.md
5. manifest.json

Then use the CSV files under tables/ to create a polished detection performance comparison report.

The dashboard app already extracted the useful metrics/tables from raw detection data. Focus on analysis of the given evidence: explain what changed, why it matters, where the trade-offs are, and what should be checked next. Do not make the report mostly about re-computing obvious table values.

The report should include:
- executive summary
- KPI comparison
- recall / precision / FP trade-off explanation
- distance-bin graphs
- class-level graphs
- TP improvement hotspots
- FP increase concentration analysis
- safety-critical degraded cases
- localization error comparison if available
- final recommendation

Please generate many charts from the CSV data, not only text.
Separate evidence-backed conclusions from hypotheses.

If code execution is available, use Python/pandas with plotly or matplotlib to generate graphs from the CSV files.
Quantify concentration such as top scenarios/datasets explaining FP increase when supported by the tables.
Include scenario/dataset/frame examples and 3D viewer links when available.
Output a clean HTML report if possible; otherwise output Markdown with embedded/generated charts.
"""


def build_llm_analysis_package(
    *,
    tables: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
) -> bytes:
    """Create a portable ZIP with prompt, neutral data brief, manifest, and CSV evidence tables."""
    created_at = pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
    manifest = {
        "created_at": created_at,
        "package_type": "detection_stats_llm_analysis",
        "metadata": metadata,
        "tables": [],
    }
    prompt_md = llm_report_instructions(metadata, tables)
    data_brief_md = llm_data_brief(metadata, tables)
    blueprint_md = llm_report_blueprint()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", prompt_md)
        zf.writestr("llm_instructions.md", prompt_md)
        zf.writestr("analysis_data_brief.md", data_brief_md)
        zf.writestr("recommended_report_blueprint.md", blueprint_md)

        for name, df in tables.items():
            safe_name = safe_export_filename(name)
            file_name = f"tables/{safe_name}.csv"
            if df is None:
                df = pd.DataFrame()
            zf.writestr(file_name, df.to_csv(index=False))
            manifest["tables"].append(
                {
                    "name": name,
                    "file": file_name,
                    "rows": int(len(df)),
                    "columns": [str(c) for c in df.columns],
                }
            )

        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))

    return buf.getvalue()


def build_single_llm_analysis_tables(
    con,
    *,
    view: str,
    filter_clause: str,
) -> Dict[str, pd.DataFrame]:
    tables = {
        "Class metrics": report_label_metrics(con, view, filter_clause),
        "Scene hotspots": report_scene_metrics(con, view, filter_clause),
        "FN frames": report_fn_frames(con, view, filter_clause),
        "Distance rates": con.execute(sql_distance_bin_rates_from_eval_flat(view, filter_clause, metrics="both")).df(),
    }
    df_err = report_error_metrics(con, view, filter_clause)
    if not df_err.empty:
        tables["Mean error by class"] = df_err
    return tables


def build_compare_llm_analysis_tables(
    con,
    *,
    base_view: str,
    candidate_view: str,
    base_filter: str,
    candidate_filter: str,
) -> Dict[str, pd.DataFrame]:
    df_base_label = report_label_metrics(con, base_view, base_filter)
    df_candidate_label = report_label_metrics(con, candidate_view, candidate_filter)
    df_label = df_base_label.merge(
        df_candidate_label,
        on="label",
        suffixes=("_base", "_candidate"),
        how="outer",
    ).fillna(0)
    for col in ["tpr", "fpr", "precision", "f1"]:
        if f"{col}_candidate" in df_label.columns and f"{col}_base" in df_label.columns:
            df_label[f"{col}_delta"] = df_label[f"{col}_candidate"] - df_label[f"{col}_base"]

    try:
        df_diff_label = report_diff_by_label(con, base_view, candidate_view, base_filter, candidate_filter)
        df_diff_scene = report_diff_by_scene_or_frame(con, base_view, candidate_view, base_filter, candidate_filter, by_frame=False)
        df_diff_frame = report_diff_by_scene_or_frame(con, base_view, candidate_view, base_filter, candidate_filter, by_frame=True)
    except Exception:
        df_diff_label = pd.DataFrame()
        df_diff_scene = pd.DataFrame()
        df_diff_frame = pd.DataFrame()

    df_base_dist = con.execute(sql_distance_bin_rates_from_eval_flat(base_view, base_filter, metrics="both")).df()
    df_candidate_dist = con.execute(sql_distance_bin_rates_from_eval_flat(candidate_view, candidate_filter, metrics="both")).df()
    try:
        df_label_dist_delta = report_label_distance_compare(con, base_view, candidate_view, base_filter, candidate_filter)
    except Exception:
        df_label_dist_delta = pd.DataFrame()

    df_degraded_objects = report_degraded_object_details(con, base_view, candidate_view, base_filter, candidate_filter)
    _, df_critical_cases = report_critical_case_phrases(df_degraded_objects)
    _, df_consecutive_failures = report_consecutive_failure_phrases(df_degraded_objects)

    df_err_base = report_error_metrics(con, base_view, base_filter)
    df_err_candidate = report_error_metrics(con, candidate_view, candidate_filter)
    df_err = pd.DataFrame()
    if not df_err_base.empty and not df_err_candidate.empty:
        df_err = df_err_base.merge(df_err_candidate, on="label", suffixes=("_base", "_candidate"), how="inner")
        for col in ["mean_abs_x_error", "mean_abs_y_error", "mean_abs_yaw_error"]:
            if f"{col}_candidate" in df_err.columns and f"{col}_base" in df_err.columns:
                df_err[f"{col}_delta"] = df_err[f"{col}_candidate"] - df_err[f"{col}_base"]

    tables = {
        "Class rate comparison": df_label.sort_values("tpr_delta", ascending=False) if "tpr_delta" in df_label.columns else df_label,
        "Object diff by class": df_diff_label,
        "Object diff by scene": df_diff_scene,
        "Object diff by frame": df_diff_frame,
        "Distance rates - baseline": df_base_dist,
        "Distance rates - candidate": df_candidate_dist,
        "Per-class distance deltas": df_label_dist_delta,
        "Critical degraded cases": df_critical_cases,
        "Consecutive degraded objects": df_consecutive_failures,
    }
    if not df_err.empty:
        tables["Mean error comparison"] = df_err
    tables.update(
        report_fp_diff_tables(
            con,
            base_view,
            candidate_view,
            base_filter,
            candidate_filter,
        )
    )
    return tables
