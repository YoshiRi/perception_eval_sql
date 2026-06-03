"""
Parquet schema detection for evaluation dashboard.
Returns available columns (and optional dtypes) so pages can show/hide
features based on schema (e.g. has_confidence, has_velocity, has_z).
"""

from typing import List, Set, Optional
import duckdb


DETECTION_STATS_REQUIRED_COLUMNS = {
    "x",
    "y",
    "source",
    "status",
    "label",
    "topic_name",
    "t4dataset_id",
}


def get_parquet_columns(con: duckdb.DuckDBPyConnection, path: str) -> List[str]:
    """
    Return list of column names for the parquet file.
    Uses DuckDB DESCRIBE. Returns [] if the file cannot be read.
    """
    try:
        df = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [path],
        ).df()
        if df.empty or "column_name" not in df.columns:
            return []
        return df["column_name"].astype(str).tolist()
    except Exception:
        return []


def get_parquet_columns_set(con: duckdb.DuckDBPyConnection, path: str) -> Set[str]:
    """Return set of column names for the parquet file. Convenience for membership checks."""
    return set(get_parquet_columns(con, path))


def has_columns(con: duckdb.DuckDBPyConnection, path: str, columns: List[str]) -> bool:
    """
    Return True if the parquet file has all of the given columns.
    """
    if not columns:
        return True
    available = get_parquet_columns_set(con, path)
    return all(c in available for c in columns)


def missing_detection_stats_columns(con: duckdb.DuckDBPyConnection, path: str) -> List[str]:
    """Return object-level columns missing for Detection Stats SQL."""
    available = get_parquet_columns_set(con, path)
    return sorted(DETECTION_STATS_REQUIRED_COLUMNS - available)


def is_detection_stats_parquet(con: duckdb.DuckDBPyConnection, path: str) -> bool:
    """Return True when a parquet has the object-level schema expected by Detection Stats."""
    return not missing_detection_stats_columns(con, path)


def schema_flags(con: duckdb.DuckDBPyConnection, path: str) -> dict:
    """
    Return a dict of boolean flags for optional columns used by Detection Stats and BEV Viewer.
    Use the first parquet path when multiple runs; columns are assumed consistent per run.
    """
    cols = get_parquet_columns_set(con, path)
    return {
        "has_confidence": "confidence" in cols,
        "has_z": "z" in cols,
        "has_height": "height" in cols,
        "has_velocity": ("vx" in cols and "vy" in cols),
        "has_pointcloud_num": "pointcloud_num" in cols,
        "has_z_error": "z_error" in cols,
        "has_vx_error": "vx_error" in cols,
        "has_vy_error": "vy_error" in cols,
        "has_speed_error": "speed_error" in cols,
        "has_center_distance": "center_distance" in cols,
        "has_plane_distance": "plane_distance" in cols,
        "has_x_error": "x_error" in cols,
        "has_y_error": "y_error" in cols,
        "has_yaw_error": "yaw_error" in cols,
    }
