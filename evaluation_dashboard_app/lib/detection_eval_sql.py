"""SQL/view layer for the detection-stats parquet, shared by the Detection Stats page and backend code."""

import hashlib
import os
from pathlib import Path
from typing import List

import duckdb

from lib.parquet_schema import get_parquet_columns

# Skip the first/last few frames per dataset so Detection Stats matches the spec-sheet metrics
# (perception systems may not be fully initialized at the boundaries). Mirrors
# perception_catalog_analyzer.dataframe.operations.filter_lf. Values come from the library so
# the dashboard stays in lock-step with whichever analyzer version is installed. The pinned
# analyzer commit is 0.2.0 + skip-last-frame (#158), so SKIP_LAST_N_FRAMES is expected to be 1;
# it falls back to 0 only for older/local analyzer checkouts that do not expose the constant.
try:
    from perception_catalog_analyzer.constants import (
        SKIP_FIRST_N_FRAMES as _SKIP_FIRST_N_FRAMES,
    )
except Exception:  # pragma: no cover - library optional / older versions
    _SKIP_FIRST_N_FRAMES = 3
try:
    from perception_catalog_analyzer.constants import (
        SKIP_LAST_N_FRAMES as _SKIP_LAST_N_FRAMES,
    )
except Exception:  # pragma: no cover - older analyzer checkouts
    _SKIP_LAST_N_FRAMES = 0
SKIP_FIRST_N_FRAMES = int(_SKIP_FIRST_N_FRAMES or 0)
SKIP_LAST_N_FRAMES = int(_SKIP_LAST_N_FRAMES or 0)
# Bump when the eval_flat SQL changes so stale caches are rebuilt.
_DS_EVAL_FLAT_CACHE_VERSION = "skipframe1_poly1"

DETECTION_STATS_SKIP_INITIAL_FRAMES = 3
DETECTION_STATS_INITIAL_FRAME_FILTER = (
    f"(frame_index IS NULL OR TRY_CAST(frame_index AS BIGINT) >= {DETECTION_STATS_SKIP_INITIAL_FRAMES})"
)


def list_parquets_in_run(run_path) -> List[str]:
    """Return sorted list of absolute paths to .parquet files in the run directory."""
    p = Path(run_path)
    if not p.is_dir():
        return []
    return sorted([str(f.resolve()) for f in p.glob("*.parquet")])


def default_parquet_index(paths: List[str]) -> int:
    """Prefer current.parquet for each run's file picker."""
    for idx, path in enumerate(paths):
        if os.path.basename(path) == "current.parquet":
            return idx
    return 0


def _is_detection_stats_eval_flat_cache(path: str) -> bool:
    p = Path(path)
    return p.suffix == ".parquet" and p.name.endswith("_eval_flat.parquet")


def create_view_eval_flat(
    con,
    target_file: str,
    view_name: str = "view_eval_flat",
    *,
    exclude_polygons: bool = False,
):
    """Create view_eval_flat with distance bins."""
    safe_target = target_file.replace("'", "''")
    if _is_detection_stats_eval_flat_cache(target_file):
        # Cache already had the skip-frame filter applied when it was materialized.
        query = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM parquet_scan('{safe_target}')"
    else:
        cols = set(get_parquet_columns(con, target_file))
        query = (
            f"CREATE OR REPLACE VIEW {view_name} AS "
            + eval_flat_select_sql(
                target_file,
                has_frame_index="frame_index" in cols,
                has_t4dataset_id="t4dataset_id" in cols,
                exclude_polygons=exclude_polygons,
            )
        )
    con.execute(query)


def eval_flat_select_sql(
    target_file: str,
    *,
    has_frame_index: bool = False,
    has_t4dataset_id: bool = False,
    exclude_polygons: bool = False,
) -> str:
    safe_target = target_file.replace("'", "''")

    # Skip-frame filter mirrors perception_catalog_analyzer.dataframe.operations.filter_lf:
    # drop the first SKIP_FIRST_N_FRAMES frames, and (when enabled) the last SKIP_LAST_N_FRAMES
    # frames per t4dataset_id. Only applied when the source parquet exposes the needed columns.
    skip_first_pred = ""
    if has_frame_index and SKIP_FIRST_N_FRAMES > 0:
        skip_first_pred = (
            f"\n          AND TRY_CAST(frame_index AS BIGINT) >= {SKIP_FIRST_N_FRAMES}"
        )
    use_skip_last = has_frame_index and has_t4dataset_id and SKIP_LAST_N_FRAMES > 0
    max_frame_col = (
        "\n            , MAX(TRY_CAST(frame_index AS BIGINT)) OVER (PARTITION BY t4dataset_id) AS _max_frame"
        if use_skip_last
        else ""
    )
    final_star = "bse.* EXCLUDE (_max_frame)" if use_skip_last else "bse.*"
    skip_last_where = (
        f"\n    WHERE bse._max_frame IS NULL OR TRY_CAST(bse.frame_index AS BIGINT) <= bse._max_frame - {SKIP_LAST_N_FRAMES}"
        if use_skip_last
        else ""
    )

    source_cte = "src"
    polygon_exclusion_ctes = ""
    if exclude_polygons:
        clear_cols = (
            "x_error",
            "y_error",
            "yaw_error",
            "speed_error",
            "plane_distance",
            "pair_dt_sec",
            "pair_uuid",
        )
        clear_exprs = ",\n            ".join(
            f"CASE WHEN _poly_matched AND source = 'GT' THEN NULL ELSE {col} END AS {col}"
            for col in clear_cols
        )
        polygon_exclusion_ctes = f""",
    poly_matched_gt AS (
        SELECT DISTINCT frame_index, CAST(pair_uuid AS VARCHAR) AS uuid
        FROM src
        WHERE source = 'EST'
          AND LOWER(COALESCE(CAST(shape_type AS VARCHAR), '')) = 'polygon'
          AND status = 'TP'
          AND pair_uuid IS NOT NULL
    ),
    polygon_adjusted AS (
        SELECT * EXCLUDE (_poly_matched) REPLACE (
            CASE WHEN _poly_matched AND source = 'GT' AND status = 'TP' THEN 'FN' ELSE status END AS status,
            {clear_exprs}
        )
        FROM (
            SELECT src.*, COALESCE(poly_matched_gt.uuid IS NOT NULL, FALSE) AS _poly_matched
            FROM src
            LEFT JOIN poly_matched_gt
              ON src.frame_index = poly_matched_gt.frame_index
             AND CAST(src.uuid AS VARCHAR) = poly_matched_gt.uuid
        ) joined
        WHERE NOT (
            source = 'EST'
            AND LOWER(COALESCE(CAST(shape_type AS VARCHAR), '')) = 'polygon'
        )
    )"""
        source_cte = "polygon_adjusted"

    return f"""
    WITH src AS (
        SELECT * FROM parquet_scan('{safe_target}')
        UNION BY NAME
        SELECT CAST(NULL AS VARCHAR) AS visibility,
               CAST(NULL AS VARCHAR) AS suite_name,
               CAST(NULL AS VARCHAR) AS scenario_name,
               CAST(NULL AS VARCHAR) AS t4dataset_name,
               CAST(NULL AS VARCHAR) AS shape_type,
               CAST(NULL AS VARCHAR) AS uuid,
               CAST(NULL AS VARCHAR) AS pair_uuid,
               CAST(NULL AS BIGINT) AS frame_index,
               CAST(NULL AS DOUBLE) AS x_error,
               CAST(NULL AS DOUBLE) AS y_error,
               CAST(NULL AS DOUBLE) AS yaw_error,
               CAST(NULL AS DOUBLE) AS speed_error,
               CAST(NULL AS DOUBLE) AS plane_distance,
               CAST(NULL AS DOUBLE) AS pair_dt_sec
        WHERE FALSE
    ){polygon_exclusion_ctes},
    base AS (
        SELECT
            * REPLACE (coalesce(CAST(visibility AS VARCHAR), 'not available') AS visibility),
            sqrt(CAST(x AS DOUBLE)*CAST(x AS DOUBLE) + CAST(y AS DOUBLE)*CAST(y AS DOUBLE)) AS dist_h{max_frame_col}
        FROM {source_cte}
        WHERE x IS NOT NULL AND y IS NOT NULL{skip_first_pred}
    ),
    bins AS (
        SELECT * FROM (
            VALUES
                (0.0,   10.0,   '[0,10)',     10),
                (10.0,  20.0,   '[10,20)',    20),
                (20.0,  30.0,   '[20,30)',    30),
                (30.0,  40.0,   '[30,40)',    40),
                (40.0,  50.0,   '[40,50)',    50),
                (50.0,  60.0,   '[50,60)',    60),
                (60.0,  70.0,   '[60,70)',    70),
                (70.0,  80.0,   '[70,80)',    80),
                (80.0,  90.0,   '[80,90)',    90),
                (90.0,  100.0,  '[90,100)',  100),
                (100.0, 110.0,  '[100,110)', 110),
                (110.0, 120.0,  '[110,120)', 120),
                (120.0, 130.0,  '[120,130)', 130),
                (130.0, 140.0,  '[130,140)', 140),
                (140.0, 150.0,  '[140,150)', 150),
                (150.0, 1e12,   '[150,inf)', 160)
        ) AS t(bin_start, bin_end, distance_bin, bin_idx)
    )
    SELECT
        {final_star},
        b.distance_bin,
        b.bin_idx,
        (status = 'TP') AS is_tp,
        (status = 'FP') AS is_fp,
        (status = 'FN') AS is_fn
    FROM base bse
    JOIN bins b
        ON bse.dist_h >= b.bin_start AND bse.dist_h < b.bin_end{skip_last_where}
    """


def _ds_cache_dir_for_run(run_path: Path) -> Path:
    return run_path / ".dashboard_cache" / "detection_stats_cache"


def _ds_cache_key_for_source(source_path: str) -> str:
    return hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]


def _ds_cache_path_for_source(run_path: Path, source_path: str, *, exclude_polygons: bool = False) -> Path:
    src = Path(source_path)
    mode = "exclude_polygons" if exclude_polygons else "all_objects"
    return (
        _ds_cache_dir_for_run(run_path)
        / f"{src.stem}_{_ds_cache_key_for_source(source_path)}_{mode}_{_DS_EVAL_FLAT_CACHE_VERSION}_eval_flat.parquet"
    )


def ensure_detection_stats_eval_flat_cache(
    con: duckdb.DuckDBPyConnection,
    *,
    run_path: Path,
    source_path: str,
    exclude_polygons: bool = False,
) -> tuple[str, bool]:
    """
    Ensure a materialized eval_flat parquet exists for this source parquet.
    Returns (cached_parquet_path, rebuilt_flag).
    """
    cache_dir = _ds_cache_dir_for_run(run_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _ds_cache_path_for_source(run_path, source_path, exclude_polygons=exclude_polygons)
    source_stat = Path(source_path).stat()
    needs_rebuild = (
        not cache_path.exists()
        or cache_path.stat().st_mtime < source_stat.st_mtime
    )
    if needs_rebuild:
        safe_out = str(cache_path).replace("'", "''")
        cols = set(get_parquet_columns(con, source_path))
        con.execute(
            f"COPY ({eval_flat_select_sql(source_path, has_frame_index='frame_index' in cols, has_t4dataset_id='t4dataset_id' in cols, exclude_polygons=exclude_polygons)}) TO '{safe_out}' (FORMAT PARQUET)"
        )
    return str(cache_path), needs_rebuild

# Per-(dataset, topic, label, bin, visibility, suite) aggregates — shared by distance-bin rate queries.
_TPR_FPR_STATS_SELECT = """SELECT
            t4dataset_id,
            topic_name,
            label,
            distance_bin,
            bin_idx,
            coalesce(try(CAST(visibility AS VARCHAR)), 'not available') AS visibility,
            coalesce(try(CAST(suite_name AS VARCHAR)), '') AS suite_name,
            COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) AS gt_total,
            COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS tp_gt,
            COUNT(*) FILTER (WHERE source='EST' AND status IN ('TP','FP')) AS est_total,
            COUNT(*) FILTER (WHERE source='EST' AND status='FP') AS fp_est"""

_TPR_FPR_STATS_GROUP_BY = """t4dataset_id, topic_name, label, distance_bin, bin_idx,
            coalesce(try(CAST(visibility AS VARCHAR)), 'not available'),
            coalesce(try(CAST(suite_name AS VARCHAR)), '')"""


def sql_distance_bin_rates_from_eval_flat(
    source_eval_flat: str,
    filter_clause: str,
    *,
    metrics: str = "both",
) -> str:
    """TPR/FPR by ``distance_bin`` from ``view_eval_flat`` rows, with filters pushed into the stats CTE.

    Distance charts used to ``SELECT ... FROM view_tpr_fpr_* WHERE ...`` (nested view over parquet). On some
    DuckDB builds that plan can **SIGSEGV** the process (container exit **139**). This query inlines the same
    stats aggregation and applies ``WHERE`` on the flat view instead.
    """
    order_by = "ORDER BY CAST(REPLACE(SPLIT_PART(distance_bin, ',', 1), '[', ' ') AS INTEGER)"
    inner = f"""
    WITH stats AS (
        {_TPR_FPR_STATS_SELECT}
        FROM {source_eval_flat}
        WHERE ({filter_clause})
        GROUP BY
            {_TPR_FPR_STATS_GROUP_BY}
    )"""
    if metrics == "both":
        return f"""
        {inner}
        SELECT
            distance_bin,
            CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE 0 END AS tpr,
            CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr
        FROM stats
        GROUP BY distance_bin
        {order_by}
        """
    if metrics == "tpr":
        return f"""
        {inner}
        SELECT distance_bin,
            CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE 0 END AS tpr
        FROM stats
        GROUP BY distance_bin
        {order_by}
        """
    if metrics == "fpr":
        return f"""
        {inner}
        SELECT distance_bin,
            CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr
        FROM stats
        GROUP BY distance_bin
        {order_by}
        """
    raise ValueError(f"metrics must be 'both', 'tpr', or 'fpr', got {metrics!r}")


def sql_distance_bin_label_rates_from_eval_flat(
    source_eval_flat: str,
    filter_clause: str,
) -> str:
    """TPR/FPR by label and distance bin from ``view_eval_flat`` rows."""
    return f"""
    WITH stats AS (
        {_TPR_FPR_STATS_SELECT}
        FROM {source_eval_flat}
        WHERE ({filter_clause})
        GROUP BY
            {_TPR_FPR_STATS_GROUP_BY}
    )
    SELECT
        distance_bin,
        label,
        CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE 0 END AS tpr,
        CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr
    FROM stats
    GROUP BY distance_bin, label
    ORDER BY MIN(bin_idx), label
    """


def build_filter_clause(filters: dict,*, enable_dist_h: bool = True) -> str:
    """Build WHERE clause from filters.

    For label / suites / visibility: ``None`` means this dimension is inactive (e.g. no suite column).
    An empty list ``[]`` means no restriction on that dimension (same as all options selected).
    Using ``if filters.get('label')`` would treat ``[]`` as falsy and accidentally drop the filter,
    causing full scans (very slow on large Parquet).
    """
    conditions = [DETECTION_STATS_INITIAL_FRAME_FILTER]
    
    topic_val = filters.get('topic_name')
    if topic_val and topic_val != '__all__':
        if isinstance(topic_val, list):
            if len(topic_val) == 1:
                conditions.append(f"topic_name = '{topic_val[0]}'")
            elif len(topic_val) > 1:
                topics_escaped = [str(t).replace("'", "''") for t in topic_val]
                topics_str = "', '".join(topics_escaped)
                conditions.append(f"topic_name IN ('{topics_str}')")
        else:
            conditions.append(f"topic_name = '{topic_val}'")
    
    lbl = filters.get('label')
    if lbl is not None:
        if isinstance(lbl, list):
            if len(lbl) > 0:
                labels_escaped = [str(l).replace("'", "''") for l in lbl]
                labels_str = "', '".join(labels_escaped)
                conditions.append(f"label IN ('{labels_str}')")
        elif not isinstance(lbl, list) and lbl != '__all__':
            label_escaped = str(lbl).replace("'", "''")
            conditions.append(f"label = '{label_escaped}'")
    
    su = filters.get('suites')
    if su is not None:
        if isinstance(su, list):
            if len(su) > 0:
                suite_escaped = [str(s).replace("'", "''") for s in su]
                suite_str = "', '".join(suite_escaped)
                conditions.append(f"COALESCE(CAST(suite_name AS VARCHAR), '') IN ('{suite_str}')")
        elif not isinstance(su, list) and su != '__all__':
            s_escaped = str(su).replace("'", "''")
            conditions.append(f"COALESCE(CAST(suite_name AS VARCHAR), '') = '{s_escaped}'")
    
    vis = filters.get('visibility')
    if vis is not None:
        if isinstance(vis, list):
            if len(vis) > 0:
                vis_escaped = [str(v).replace("'", "''") for v in vis]
                vis_str = "', '".join(vis_escaped)
                conditions.append(f"COALESCE(visibility, 'not available') IN ('{vis_str}')")
        elif not isinstance(vis, list):
            vis_escaped = str(vis).replace("'", "''")
            conditions.append(f"COALESCE(visibility, 'not available') = '{vis_escaped}'")
    
    if enable_dist_h and filters.get('max_eval_range'):
        conditions.append(f"dist_h < {filters['max_eval_range']}")
    
    return " AND ".join(conditions) if conditions else "1=1"


def kpi_row_for_view(con, view: str, filter_clause: str):
    """Return global KPI values within the active filters."""
    q = f"""
    SELECT
        COUNT(*) FILTER (WHERE source = 'GT' AND status = 'TP') AS tp_gt,
        COUNT(*) FILTER (WHERE source = 'GT' AND status = 'FN') AS fn,
        COUNT(*) FILTER (WHERE source = 'EST' AND status = 'TP') AS tp_est,
        COUNT(*) FILTER (WHERE source = 'EST' AND status = 'FP') AS fp
    FROM {view}
    WHERE {filter_clause}
    """
    row = con.execute(q).fetchone()
    if not row:
        return None
    tp_gt, fn, tp_est, fp = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    gt_total = tp_gt + fn
    est_total = tp_est + fp
    tpr = (tp_gt / gt_total) if gt_total > 0 else None
    fpr = (fp / est_total) if est_total > 0 else None
    precision = (tp_est / est_total) if est_total > 0 else None
    recall = tpr
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "gt": gt_total, "tp": tp_gt, "fp": fp, "fn": fn,
        "tpr": tpr, "fpr": fpr, "precision": precision, "recall": recall, "f1": f1,
    }
