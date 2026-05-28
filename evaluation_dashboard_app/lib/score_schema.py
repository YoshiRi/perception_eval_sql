from __future__ import annotations

from pathlib import Path

import pandas as pd

SCORE_BASE_COLS = ["Scenario", "Option", "GT_OBJ"]
SCORE_BASE_COLS_WITH_DATASET = ["Scenario", "Dataset", "Option", "GT_OBJ"]

SCORE_SOURCE_METRIC_COLS = [
    "Distance",
    "NM",
    "TP/TN",
    "ADD",
    "AIL",
    "UIL",
    "PFN/PFP",
    "UUID Num",
    "Practical Pass Rate",
    "MAX_DIST_THRESH",
    "OBJ_CNTS",
]

SCORE_VIEW_METRIC_COLS = [
    "distance",
    "nm",
    "tp_tn",
    "add",
    "ail",
    "uil",
    "pfn_pfp",
    "uuid_num",
    "pass_rate",
    "max_dist_thresh",
    "obj_cnts",
]

SCORE_NUM_COLS = [
    "nm",
    "tp_tn",
    "add",
    "ail",
    "uil",
    "pfn_pfp",
    "uuid_num",
    "pass_rate",
    "max_dist_thresh",
]

SCORE_BLOCK_SIZE = len(SCORE_VIEW_METRIC_COLS)


def _looks_like_header(row: pd.Series) -> bool:
    first = str(row.iloc[0]).strip() if len(row) else ""
    return first == "Scenario"


def _looks_like_criteria_cell(value: object) -> bool:
    text = str(value).strip()
    return text.startswith("criteria")


def _drop_extra_empty_trailing_columns(df: pd.DataFrame, base_count: int) -> pd.DataFrame:
    while (
        df.shape[1]
        and df.iloc[:, -1].isna().all()
        and (df.shape[1] - base_count) % SCORE_BLOCK_SIZE != 0
    ):
        df = df.iloc[:, :-1]
    return df


def _infer_base_count(df: pd.DataFrame, header_row: pd.Series | None) -> int:
    if header_row is not None:
        header_values = [str(x).strip() for x in header_row.tolist()]
        if len(header_values) >= 4 and header_values[1] == "Dataset":
            return 4
        return 3

    if df.empty:
        return 3
    first = df.iloc[0]
    if len(first) > 4 and _looks_like_criteria_cell(first.iloc[4]):
        return 4
    if len(first) > 3 and _looks_like_criteria_cell(first.iloc[3]):
        return 3

    ncols = df.shape[1]
    if ncols >= 4 and (ncols - 4) % SCORE_BLOCK_SIZE == 0:
        return 4
    return 3


def score_raw_columns(has_dataset: bool, criteria_count: int) -> list[str]:
    cols = list(SCORE_BASE_COLS_WITH_DATASET if has_dataset else SCORE_BASE_COLS)
    for i in range(criteria_count):
        cols.extend(f"{name}{i}" for name in SCORE_SOURCE_METRIC_COLS)
    return cols


def read_score_csv(score_path: Path) -> pd.DataFrame | None:
    if not score_path.exists():
        return None

    raw = pd.read_csv(score_path, header=None, engine="python")
    if raw.empty:
        return raw

    header_row = raw.iloc[0] if _looks_like_header(raw.iloc[0]) else None
    if header_row is not None:
        raw = raw.iloc[1:].reset_index(drop=True)

    base_count = _infer_base_count(raw, header_row)
    raw = _drop_extra_empty_trailing_columns(raw, base_count)
    criteria_count = max(1, (raw.shape[1] - base_count) // SCORE_BLOCK_SIZE)
    expected_cols = base_count + criteria_count * SCORE_BLOCK_SIZE
    raw = raw.iloc[:, :expected_cols].copy()
    raw.columns = score_raw_columns(base_count == 4, criteria_count)
    return raw.reset_index(drop=True)


def score_base_cols(df_raw: pd.DataFrame) -> list[str]:
    if df_raw is not None and "Dataset" in df_raw.columns:
        return list(SCORE_BASE_COLS_WITH_DATASET)
    return list(SCORE_BASE_COLS)


def infer_score_criteria_count(
    df_raw: pd.DataFrame,
    max_criteria: int = 32,
) -> int:
    if df_raw is None or df_raw.empty:
        return 1
    base_count = len(score_base_cols(df_raw))
    n = (df_raw.shape[1] - base_count) // SCORE_BLOCK_SIZE
    n = max(1, n)
    return int(min(n, max_criteria))


def build_score_view(df_raw: pd.DataFrame, criteria_idx: int) -> pd.DataFrame:
    base_cols = score_base_cols(df_raw)
    start = len(base_cols) + criteria_idx * SCORE_BLOCK_SIZE
    end = start + SCORE_BLOCK_SIZE

    df_view = df_raw.loc[:, base_cols].copy()
    block = df_raw.iloc[:, start:end].copy()
    block.columns = SCORE_VIEW_METRIC_COLS
    df_view = pd.concat([df_view, block], axis=1)
    for column in SCORE_NUM_COLS:
        df_view[column] = pd.to_numeric(df_view[column], errors="coerce")
    return df_view


def score_identity_cols(df: pd.DataFrame) -> list[str]:
    return ["Scenario", "Dataset"] if df is not None and "Dataset" in df.columns else ["Scenario"]
