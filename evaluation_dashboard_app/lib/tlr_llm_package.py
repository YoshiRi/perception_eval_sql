"""Evidence tables + LLM analysis package for TLR (traffic light recognition) results.

The TLR Analysis page charts criteria matrices, vehicle-status heatmaps and scenario
insights; this module flattens the same analyzer outputs into CSV evidence tables and
zips them with instructions, so a coding agent can write the TLR report the way it
already writes detection reports (see lib/detection_llm_package.py). Streamlit-free.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pandas as pd

from lib.detection_llm_package import build_llm_analysis_package
from lib.tlr_eval_analyzer import TLREvaluationAnalyzer

WORST_SCENARIO_LIMIT = 15
FN_FRAME_LIMIT = 200


def load_tlr_analyzer(result_directory: str) -> TLREvaluationAnalyzer:
    """Load and pre-compute a TLR analyzer the way the page does. Raises ValueError
    when the directory has no readable TLR results."""
    analyzer = TLREvaluationAnalyzer(str(result_directory))
    analyzer.load_all_results()
    if not analyzer.scenario_results and not analyzer.loaded_from_cache:
        raise ValueError(f"No TLR result.json data under {result_directory}")
    analyzer.extract_criteria_data()
    analyzer.pre_calculate_all_data()
    return analyzer


def _signal_mask(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str)
    return (text != "") & (text != "0 []") & (text != "null")


def build_tlr_scenario_summary(details_df: pd.DataFrame | None) -> pd.DataFrame:
    """Per-scenario frame/TP/FN roll-up, mirroring the page's scenario insights."""
    if details_df is None or details_df.empty:
        return pd.DataFrame()
    df = details_df.copy()
    df["scenario"] = df["scenario"].fillna("").astype(str)
    split = df["scenario"].str.split("/", n=1, expand=True)
    df["suite"] = split[0].replace("", "Current run")
    df["_has_tp"] = _signal_mask(df["tp"])
    df["_has_fn"] = _signal_mask(df["fn"])
    df["_evaluable"] = df["_has_tp"] | df["_has_fn"]
    summary = (
        df.groupby(["suite", "scenario"], dropna=False)
        .agg(
            frames=("frame_index", "count"),
            evaluable_frames=("_evaluable", "sum"),
            tp_frames=("_has_tp", "sum"),
            fn_frames=("_has_fn", "sum"),
            criteria_count=("criteria", "nunique"),
            signal_types=("traffic_light_type", "nunique"),
        )
        .reset_index()
    )
    summary["tp_rate"] = np.where(
        summary["evaluable_frames"] > 0,
        summary["tp_frames"] / summary["evaluable_frames"],
        np.nan,
    )
    return summary.sort_values(["suite", "scenario"]).reset_index(drop=True)


def build_tlr_llm_analysis_tables(analyzer: TLREvaluationAnalyzer) -> Dict[str, pd.DataFrame]:
    """The evidence tables an LLM needs to write a single-run TLR report."""
    details = analyzer.get_vehicle_status_details_df()
    scenario_summary = build_tlr_scenario_summary(details)

    tables: Dict[str, pd.DataFrame] = {
        "Criteria matrix": analyzer.create_criteria_matrix(),
        "Vehicle status TP rates": analyzer.create_vehicle_status_matrix(),
        "Vehicle status frame counts": analyzer.create_vehicle_status_counts_matrix(),
        "Critical and priority zones": analyzer.create_vehicle_status_critical_priority_matrix(),
        "Critical and priority counts": analyzer.create_vehicle_status_critical_priority_counts_matrix(),
        "Scenario summary": scenario_summary,
    }
    if not scenario_summary.empty:
        evaluable = scenario_summary[scenario_summary["evaluable_frames"] > 0]
        tables["Worst scenarios"] = (
            evaluable.nsmallest(WORST_SCENARIO_LIMIT, "tp_rate").reset_index(drop=True)
        )
    if details is not None and not details.empty:
        fn_frames = details[_signal_mask(details["fn"])]
        tables["FN frames"] = fn_frames.head(FN_FRAME_LIMIT)[
            [c for c in ("scenario", "frame_index", "frame_name", "status", "speed_kph",
                         "traffic_light_type", "criteria", "fn") if c in fn_frames.columns]
        ].reset_index(drop=True)
    return tables


def build_tlr_compare_tables(
    base: TLREvaluationAnalyzer, candidate: TLREvaluationAnalyzer
) -> Dict[str, pd.DataFrame]:
    """Base-vs-candidate evidence: per-criteria and per-scenario TP-rate deltas."""
    criteria = base.create_criteria_matrix().merge(
        candidate.create_criteria_matrix(),
        on="Criteria", how="outer", suffixes=("_base", "_candidate"),
    )
    if "TP rate_base" in criteria.columns and "TP rate_candidate" in criteria.columns:
        criteria["tp_rate_delta"] = criteria["TP rate_candidate"] - criteria["TP rate_base"]
        criteria = criteria.sort_values("tp_rate_delta")

    scenario = build_tlr_scenario_summary(base.get_vehicle_status_details_df()).merge(
        build_tlr_scenario_summary(candidate.get_vehicle_status_details_df()),
        on=["suite", "scenario"], how="outer", suffixes=("_base", "_candidate"),
    )
    if "tp_rate_base" in scenario.columns and "tp_rate_candidate" in scenario.columns:
        scenario["tp_rate_delta"] = scenario["tp_rate_candidate"] - scenario["tp_rate_base"]
        scenario = scenario.sort_values("tp_rate_delta")

    tables: Dict[str, pd.DataFrame] = {
        "Criteria comparison": criteria,
        "Scenario comparison": scenario,
        "Vehicle status TP rates - base": base.create_vehicle_status_matrix(),
        "Vehicle status TP rates - candidate": candidate.create_vehicle_status_matrix(),
        "Critical and priority zones - base": base.create_vehicle_status_critical_priority_matrix(),
        "Critical and priority zones - candidate": candidate.create_vehicle_status_critical_priority_matrix(),
    }
    if "tp_rate_delta" in scenario.columns and not scenario.empty:
        tables["Most degraded scenarios"] = (
            scenario.dropna(subset=["tp_rate_delta"]).head(WORST_SCENARIO_LIMIT).reset_index(drop=True)
        )
    return tables


def tlr_llm_instructions(metadata: Dict[str, Any], tables: Dict[str, pd.DataFrame]) -> str:
    mode = str(metadata.get("mode") or "single")
    table_lines = "\n".join(
        f"- `tables/{name}.csv` ({len(df)} rows)" for name, df in tables.items()
    )
    focus = (
        "Compare the candidate against the base: which criteria and scenarios regressed, "
        "which improved, and whether critical/priority signal zones (the ranges that gate "
        "vehicle behavior) are affected."
        if mode == "compare"
        else "Assess recognition quality per criteria and vehicle status, and name the "
        "scenarios and signal types that drag the TP rate down."
    )
    return f"""# LLM Instructions: Traffic Light Recognition Analysis Report

This package holds evidence tables extracted from a TLR evaluation
({json.dumps(metadata.get('scope') or metadata.get('comparison') or {}, ensure_ascii=False)}).
Your job is interpretation: {focus}

Evidence tables:
{table_lines}

Domain notes:
- `TP rate` counts frames where the recognized signal matched ground truth among
  evaluable frames (frames with a TP or FN judgement).
- Criteria 0-9 index increasing distance/importance ranges; the "Critical and
  priority zones" tables isolate the ranges that matter most for vehicle behavior
  (critical: criteria 5-6, priority: 2-4). A drop there outweighs an average drop.
- Vehicle status (Driving / Turning / No Move) changes camera geometry; a TP-rate
  drop concentrated in Turning usually points at viewpoint robustness, not the
  classifier.
- FN frames list concrete misses with speed and signal type; use them for examples,
  not statistics.

Report structure: verdict first (one paragraph), then criteria-level findings,
vehicle-status findings, scenario hotspots with concrete examples, and a short
recommendation. Quote real numbers from the tables; never invent values.
"""


def tlr_data_brief(metadata: Dict[str, Any], tables: Dict[str, pd.DataFrame]) -> str:
    lines = [
        "# TLR Analysis Data Brief",
        "",
        "Prepared evidence for a traffic-light-recognition report. The dashboard has",
        "already aggregated raw result.json frames; interpret, don't recompute.",
        "",
        f"Context: {json.dumps(metadata, ensure_ascii=False, default=str)}",
        "",
    ]
    for name, df in tables.items():
        lines.append(f"## {name}")
        lines.append(f"{len(df)} rows; columns: {', '.join(str(c) for c in df.columns)}")
        lines.append("")
    return "\n".join(lines)


def build_tlr_llm_analysis_package(
    *, tables: Dict[str, pd.DataFrame], metadata: Dict[str, Any]
) -> bytes:
    instructions = tlr_llm_instructions(metadata, tables)
    return build_llm_analysis_package(
        tables=tables,
        metadata=metadata,
        package_type="tlr_llm_analysis",
        documents={
            "llm_instructions.md": instructions,
            "analysis_data_brief.md": tlr_data_brief(metadata, tables),
            "recommended_report_blueprint.md": instructions,
        },
    )
