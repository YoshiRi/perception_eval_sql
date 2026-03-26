"""Summary.csv helpers for comparing evaluation runs."""

from __future__ import annotations

import pandas as pd


def build_summary_delta(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Row-aligned delta: metrics from A, same metrics with _B suffix from B, and *_delta = B − A."""
    if "perception_label" in df_a.columns and "perception_label" in df_b.columns:
        key_cols = ["id", "perception_label"]
    else:
        key_cols = ["id"]

    df_a, df_b = df_a.set_index(key_cols), df_b.set_index(key_cols)
    common_idx = df_a.index.intersection(df_b.index)
    result = pd.DataFrame(index=common_idx)
    metrics = ["TP", "xstd", "ystd", "xrms", "yrms", "vx", "vy"]
    for m in metrics:
        result[m] = df_a.loc[common_idx, m]
        result[f"{m}_B"] = df_b.loc[common_idx, m]
        result[f"{m}_delta"] = df_b.loc[common_idx, m] - df_a.loc[common_idx, m]
    return result.reset_index()


def summary_delta_overlap_stats(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Describe index overlap used by :func:`build_summary_delta` (same join-key rules)."""
    if df_a is None or df_b is None:
        return {"valid": False, "error": "Summary dataframe missing.", "key_cols": []}
    if "id" not in df_a.columns or "id" not in df_b.columns:
        return {
            "valid": False,
            "error": "Summary must include an `id` column for delta alignment.",
            "key_cols": ["id"],
        }
    if "perception_label" in df_a.columns and "perception_label" in df_b.columns:
        key_cols = ["id", "perception_label"]
    else:
        key_cols = ["id"]
    for c in key_cols:
        if c not in df_a.columns or c not in df_b.columns:
            return {
                "valid": False,
                "error": f"Join needs column `{c}` in both summaries; one run is missing it.",
                "key_cols": key_cols,
            }

    idx_a = df_a.set_index(key_cols).index
    idx_b = df_b.set_index(key_cols).index
    common = idx_a.intersection(idx_b)
    only_a = idx_a.difference(idx_b)
    only_b = idx_b.difference(idx_a)

    def _sample(idx_diff: pd.Index, k: int = 5) -> list[str]:
        if len(idx_diff) == 0:
            return []
        out: list[str] = []
        for x in list(idx_diff)[:k]:
            if isinstance(x, tuple):
                out.append(", ".join(str(p) for p in x))
            else:
                out.append(str(x))
        return out

    return {
        "valid": True,
        "key_cols": key_cols,
        "n_rows_baseline": int(len(df_a)),
        "n_rows_candidate": int(len(df_b)),
        "n_matched_keys": int(len(common)),
        "n_only_baseline": int(len(only_a)),
        "n_only_candidate": int(len(only_b)),
        "sample_only_baseline": _sample(only_a),
        "sample_only_candidate": _sample(only_b),
        "matched_empty": len(common) == 0,
    }
