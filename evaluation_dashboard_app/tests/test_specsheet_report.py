"""Tests for specsheet trend extraction helpers."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from lib.specsheet_report import (
    FULL_DATASET_EVALUATION_HEADER,
    USECASE_PLANNING_EVALUATION_HEADERS,
    TREND_METADATA_FILENAME,
    TREND_SUMMARY_FILENAME,
    _coerce_specsheet_scene_numeric_columns,
    _get_blocks_compat,
    _release_date_key,
    _recall_ratio_to_percent,
    _aggregate_usecase_devops_frame,
    discover_trend_metadata_files,
    ensure_specsheet_csvs,
    ensure_specsheet_inputs,
    extract_devops_case_rows,
    extract_performance_metrics_from_summary,
    extract_usecase_metrics_from_summary,
    ensure_full_trend_summary,
    load_devops_trend_data,
    load_performance_trend_data,
    parse_trend_metadata_text,
)


def _metric_payload(map_value: float = 0.5) -> dict:
    return {
        "mAP": {"car": map_value, "truck": None},
        "precision": {"car": 0.8, "truck": None},
        "recall": {"car": 0.7, "truck": None},
        "FNR": {"car": 0.3, "truck": None},
        "x_error": {"car": 1.0, "truck": None},
        "y_error": {"car": 2.0, "truck": None},
        "yaw_error": {"car": 0.1, "truck": None},
        "speed_error": {"car": 3.0, "truck": None},
        "minADE@1s": {"car": 1.1, "truck": None},
        "minFDE@1s": {"car": 1.2, "truck": None},
        "minADE@3s": {"car": 3.1, "truck": None},
        "minFDE@3s": {"car": 3.2, "truck": None},
        "minADE@5s": {"car": 5.1, "truck": None},
        "minFDE@5s": {"car": 5.2, "truck": None},
    }


def _metric_block(payload: dict) -> dict:
    return {
        "header": FULL_DATASET_EVALUATION_HEADER,
        "evaluation_type": "full",
        "mode": "metrics",
        "tables": [{"data": payload}],
    }


def _usecase_metric_block(*payloads: dict) -> dict:
    return {
        "header": next(iter(USECASE_PLANNING_EVALUATION_HEADERS)),
        "evaluation_type": "usecase",
        "mode": "metrics",
        "tables": [{"data": payload} for payload in payloads],
    }


def _write_trend_metadata(resources_dir, version: str, abbr: str | None = None) -> None:
    metadata = {
        "tags": ["trend"],
        "pilot_auto_version": version,
        "data_count": "199,776+",
        "description": "data update",
        "date": "2026.11.7",
    }
    if abbr is not None:
        # Manual abbreviation key introduced with perception_catalog_analyzer 0.2.0.
        metadata["pilot_auto_version_abbr"] = abbr
    (resources_dir / TREND_METADATA_FILENAME).write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026.6.9", (2026, 6, 9)),
        ("2026.07.08", (2026, 7, 8)),   # releases pad the month inconsistently
        ("2026-05-18", (2026, 5, 18)),
        ("2026/12/31", (2026, 12, 31)),
        ("", (0, 0, 0)),
        ("no date here", (0, 0, 0)),
    ],
)
def test_release_date_key_parses_the_formats_releases_actually_write(text, expected):
    assert _release_date_key(text) == expected


def test_release_dates_order_chronologically_not_lexically():
    """Trend charts read left to right in time, and every one of them draws from this
    ordering. Compared as strings, "2026.07.31" (July) lands before "2026.3.4" (March)
    because "0" < "3", which silently scrambled the x-axis."""
    dates = ["2026.07.31", "2026.3.4", "2026.6.9", "2026.5.18", "2026.07.08"]
    assert sorted(dates, key=_release_date_key) == [
        "2026.3.4", "2026.5.18", "2026.6.9", "2026.07.08", "2026.07.31",
    ]
    assert max(dates, key=_release_date_key) == "2026.07.31"
    assert sorted(dates) != sorted(dates, key=_release_date_key)  # the bug this replaces


def test_ensure_specsheet_inputs_prefers_parquet_without_csv_conversion(tmp_path):
    frame = pd.DataFrame(
        {
            "frame_index": [1],
            "topic_name": ["perception.object_recognition.objects"],
            "label": ["car"],
        }
    )
    frame.to_parquet(tmp_path / "current.parquet", index=False)

    paths = ensure_specsheet_inputs(tmp_path)

    assert paths["current"] == tmp_path / "current.parquet"
    assert not (tmp_path / "current.csv").exists()


def test_ensure_specsheet_inputs_accepts_csv_only_runs(tmp_path):
    frame = pd.DataFrame(
        {
            "frame_index": [1],
            "topic_name": ["perception.object_recognition.objects"],
            "label": ["car"],
        }
    )
    frame.to_csv(tmp_path / "current.csv", index=False)

    paths = ensure_specsheet_inputs(tmp_path)

    assert paths["current"] == tmp_path / "current.csv"


def test_ensure_specsheet_csvs_keeps_explicit_csv_compatibility(tmp_path):
    frame = pd.DataFrame(
        {
            "frame_index": [1],
            "topic_name": ["perception.object_recognition.objects"],
            "label": ["car"],
        }
    )
    frame.to_parquet(tmp_path / "current.parquet", index=False)

    paths = ensure_specsheet_csvs(tmp_path)

    assert paths["current_csv"] == tmp_path / "current.csv"
    assert (tmp_path / "current.csv").exists()


def test_performance_metrics_deduplicates_identical_full_metric_blocks():
    summary = {
        "blocks": [
            _metric_block(_metric_payload(0.6)),
            {
                "header": "全数アノテーション分布",
                "evaluation_type": "full",
                "mode": "annotation",
                "tables": [{"data": {"annotation_count": {"car": 10}}}],
            },
            _metric_block(copy.deepcopy(_metric_payload(0.6))),
        ]
    }

    metrics = extract_performance_metrics_from_summary(summary)

    assert metrics["mAP"] == pytest.approx(0.6)
    assert metrics["minADE@5s"] == pytest.approx(5.1)


def test_performance_metrics_rejects_distinct_full_metric_blocks():
    summary = {
        "blocks": [
            _metric_block(_metric_payload(0.6)),
            _metric_block(_metric_payload(0.9)),
        ]
    }

    with pytest.raises(ValueError, match="distinct full summary table"):
        extract_performance_metrics_from_summary(summary)


def test_usecase_metrics_average_multiple_planning_tables():
    summary = {
        "blocks": [
            _usecase_metric_block(
                {
                    "recall": {"car": 0.5},
                    "FNR": {"car": 0.5},
                    "x_error": {"car": 1.0},
                    "minADE@5s": {"car": 4.0},
                },
                {
                    "recall": {"car": 0.9},
                    "FNR": {"car": 0.1},
                    "x_error": {"car": 3.0},
                    "minADE@5s": {"car": 6.0},
                },
            )
        ]
    }

    metrics = extract_usecase_metrics_from_summary(summary)

    assert metrics["recall"] == pytest.approx(0.7)
    assert metrics["FNR"] == pytest.approx(0.3)
    assert metrics["x_error"] == pytest.approx(2.0)
    assert metrics["minADE@5s"] == pytest.approx(5.0)


def test_load_performance_trend_data_deduplicates_summary_file_blocks(tmp_path):
    resources_dir = tmp_path / "resources"
    resources_dir.mkdir()
    metadata_path = resources_dir / TREND_METADATA_FILENAME
    _write_trend_metadata(
        resources_dir,
        "Pilot.Auto v4.5.0 (centerpoint x2/2.3.1)",
        abbr="p450-c231",
    )
    payload = _metric_payload(0.6)
    summary = {"blocks": [_metric_block(payload), _metric_block(copy.deepcopy(payload))]}
    (resources_dir / TREND_SUMMARY_FILENAME).write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    rows = load_performance_trend_data([metadata_path])

    assert len(rows) == 1
    assert rows[0]["version"] == "Pilot.Auto v4.5.0 (centerpoint x2/2.3.1)"
    assert rows[0]["version_abbr"] == "p450-c231"
    assert rows[0]["topic"] == "perception.object_recognition.objects"
    assert rows[0]["mAP"] == pytest.approx(0.6)


def test_parse_trend_metadata_preserves_optional_version_abbr():
    # The dashboard standardizes on the library's key (pilot_auto_version_abbr); the legacy
    # version_abbr input is still accepted and normalized to it.
    metadata = parse_trend_metadata_text(
        """
tags: [trend]
pilot_auto_version: Pilot.Auto beta/v4.3.2
version_abbr: beta/v4.3.2
data_count: 99,776+
description: test release
date: 2026.05.28
"""
    )

    assert metadata["pilot_auto_version_abbr"] == "beta/v4.3.2"

    # The library key is used directly when provided.
    metadata_new = parse_trend_metadata_text(
        """
tags: [trend]
pilot_auto_version: Pilot.Auto beta/v4.3.2
pilot_auto_version_abbr: b432
data_count: 99,776+
description: test release
date: 2026.05.28
"""
    )
    assert metadata_new["pilot_auto_version_abbr"] == "b432"


def test_ensure_full_trend_summary_requires_full_summary(tmp_path):
    summary_path = tmp_path / TREND_SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps({"blocks": [_metric_block(_metric_payload(0.6))]}),
        encoding="utf-8",
    )

    assert ensure_full_trend_summary(summary_path) == summary_path

    devops_path = tmp_path / "devops_summary.json"
    devops_path.write_text(
        json.dumps({"Category": {"Case": {"passed": 1, "total": 2}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="classified as `devops`"):
        ensure_full_trend_summary(devops_path)


def test_extract_devops_case_rows_preserves_two_level_hierarchy():
    rows = extract_devops_case_rows(
        {
            "Major": {
                "Case A": {"passed": 1, "total": 2},
                "Case B": {"passed": 3, "total": 4},
            }
        }
    )

    assert rows == [
        {
            "major_category": "Major",
            "mid_category": "Case A",
            "minor_category": "Case A",
            "case_name": "Case A",
            "passed": 1,
            "total": 2,
            "pass_rate": 50.0,
        },
        {
            "major_category": "Major",
            "mid_category": "Case B",
            "minor_category": "Case B",
            "case_name": "Case B",
            "passed": 3,
            "total": 4,
            "pass_rate": 75.0,
        },
    ]


def test_usecase_devops_parquet_fallback_builds_category_hierarchy():
    frame = pd.DataFrame(
        [
            {"Suite Name": "DevOps_V1_FN_Object_Ahead", "Success": 7, "Total": 12},
            {"Suite Name": "DevOps_V1_FP_Ground_perception_fp", "Success": 8, "Total": 15},
            {"Suite Name": "DevOps_V1_FN_New_Unmapped_Case", "Success": 1, "Total": 3},
        ]
    )

    summary = _aggregate_usecase_devops_frame(frame)

    assert "Suite pass rate" not in summary
    assert summary["物体未検出 (FN)"]["定義済み物体に対する未検出"]["前方車未検知"] == {
        "passed": 7,
        "total": 12,
    }
    assert summary["物体過検出 (FP)"]["未定義物体に対する誤検出"]["地面誤検知"] == {
        "passed": 8,
        "total": 15,
    }
    assert summary["物体未検出 (FN)"]["未分類"]["FN New Unmapped Case"] == {
        "passed": 1,
        "total": 3,
    }


def test_extract_devops_case_rows_maps_suite_pass_rate_summary():
    rows = extract_devops_case_rows(
        {
            "DevOps": {
                "Suite pass rate": {
                    "DevOps_V1_FN_Object_Ahead": {"passed": 7, "total": 12},
                    "DevOps_V1_FP_Ground_perception_fp": {"passed": 8, "total": 15},
                    "DevOps_V1_FN_New_Unmapped_Case": {"passed": 1, "total": 3},
                }
            }
        }
    )

    by_case = {row["minor_category"]: row for row in rows}
    assert by_case["前方車未検知"]["major_category"] == "物体未検出 (FN)"
    assert by_case["前方車未検知"]["mid_category"] == "定義済み物体に対する未検出"
    assert by_case["前方車未検知"]["passed"] == 7
    assert by_case["前方車未検知"]["total"] == 12
    assert by_case["地面誤検知"]["major_category"] == "物体過検出 (FP)"
    assert by_case["地面誤検知"]["mid_category"] == "未定義物体に対する誤検出"
    assert by_case["地面誤検知"]["passed"] == 8
    assert by_case["地面誤検知"]["total"] == 15
    assert by_case["FN New Unmapped Case"]["major_category"] == "物体未検出 (FN)"
    assert by_case["FN New Unmapped Case"]["mid_category"] == "未分類"


def test_recall_ratio_to_percent_preserves_percent_values():
    assert _recall_ratio_to_percent(0.83) == pytest.approx(83.0)
    assert _recall_ratio_to_percent("0.75") == pytest.approx(75.0)
    assert _recall_ratio_to_percent(83.0) == pytest.approx(83.0)
    assert pd.isna(_recall_ratio_to_percent(None))


def test_load_devops_trend_data_reads_pass_rate_summary(tmp_path):
    resources_dir = tmp_path / "devops_job"
    resources_dir.mkdir()
    metadata_path = resources_dir / TREND_METADATA_FILENAME
    _write_trend_metadata(
        resources_dir, "Pilot.Auto v4.5.0 (centerpoint x2/2.3.1)", abbr="p450-c231"
    )
    summary = {
        "物体未検出 (FN)": {
            "定義済み物体に対する未検出": {
                "前方車未検知": {"passed": 5, "total": 12},
                "大型物体未検知": {"passed": 1, "total": 10},
            }
        },
        "誤検知 (FP)": {
            "静止物に対する誤検知": {
                "草木誤検知": {"passed": 6, "total": 21},
            }
        },
    }
    (resources_dir / TREND_SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = load_devops_trend_data([metadata_path])

    assert len(rows) == 1
    assert rows[0]["version"] == "Pilot.Auto v4.5.0 (centerpoint x2/2.3.1)"
    assert rows[0]["version_abbr"] == "p450-c231"
    assert rows[0]["scenario_count"] == 43
    assert rows[0]["overall_pass_rate"] == pytest.approx(12 / 43 * 100.0)
    assert rows[0]["devops_data"] == summary


def test_load_devops_trend_data_skips_performance_block_summary(tmp_path):
    resources_dir = tmp_path / "full_job"
    resources_dir.mkdir()
    metadata_path = resources_dir / TREND_METADATA_FILENAME
    _write_trend_metadata(resources_dir, "Pilot.Auto v4.5.0 (centerpoint x2/2.3.1)")
    (resources_dir / TREND_SUMMARY_FILENAME).write_text(
        json.dumps({"blocks": [_metric_block(_metric_payload(0.6))]}),
        encoding="utf-8",
    )

    rows = load_devops_trend_data([metadata_path])

    assert rows == []


def test_load_devops_trend_data_aligns_missing_detail_categories(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_metadata_path = first_dir / TREND_METADATA_FILENAME
    second_metadata_path = second_dir / TREND_METADATA_FILENAME
    _write_trend_metadata(first_dir, "Pilot.Auto v4.5.0 (centerpoint x2/2.3.1)")
    _write_trend_metadata(second_dir, "Pilot.Auto v4.6.0 (centerpoint x2/2.3.1)")
    (first_dir / TREND_SUMMARY_FILENAME).write_text(
        json.dumps(
            {
                "物体未検出 (FN)": {
                    "定義済み物体に対する未検出": {
                        "前方車未検知": {"passed": 5, "total": 12},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (second_dir / TREND_SUMMARY_FILENAME).write_text(
        json.dumps(
            {
                "誤検知 (FP)": {
                    "静止物に対する誤検知": {
                        "草木誤検知": {"passed": 6, "total": 21},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = load_devops_trend_data([first_metadata_path, second_metadata_path])

    assert rows[0]["devops_data"]["誤検知 (FP)"]["静止物に対する誤検知"]["草木誤検知"] == {
        "passed": 0,
        "total": 0,
    }
    assert rows[1]["devops_data"]["物体未検出 (FN)"]["定義済み物体に対する未検出"]["前方車未検知"] == {
        "passed": 0,
        "total": 0,
    }


def test_coerce_specsheet_scene_numeric_columns_preserves_text_ids():
    scene = SimpleNamespace(
        current=pd.DataFrame(
            {
                "x": ["1.5", "bad"],
                "y": ["2", ""],
                "confidence": ["0.7", None],
                "label": ["car", "truck"],
                "uuid": ["001", "002"],
                "status": ["TP", "FN"],
            }
        ),
        future=pd.DataFrame(
            {
                "tx": ["3.0", "bad"],
                "ty": ["4.5", ""],
                "relative_time": ["1000000", "2000000"],
                "pair_uuid": ["001", "002"],
                "label": ["car", "truck"],
            }
        ),
    )

    coerced = _coerce_specsheet_scene_numeric_columns(scene)

    assert coerced.current["x"].tolist()[0] == pytest.approx(1.5)
    assert pd.isna(coerced.current["x"].tolist()[1])
    assert coerced.future["tx"].tolist()[0] == pytest.approx(3.0)
    assert pd.isna(coerced.future["tx"].tolist()[1])
    assert coerced.current["uuid"].tolist() == ["001", "002"]
    assert coerced.future["pair_uuid"].tolist() == ["001", "002"]


def test_get_blocks_compat_passes_analyzer_evaluation_type_enum(tmp_path):
    blocks_module = pytest.importorskip("perception_catalog_analyzer.specsheet.blocks")
    captured = {}

    def fake_get_blocks(
        df,
        labels,
        metrics,
        resource_path,
        html_path,
        parquet_compression,
        evaluation_type,
    ):
        captured["evaluation_type"] = evaluation_type
        return ["abstract"], ["detail"]

    abstract, detailed = _get_blocks_compat(
        fake_get_blocks,
        df=SimpleNamespace(),
        labels=["car"],
        metrics=["mAP"],
        topic_name="perception.object_recognition.tracking.objects",
        outdir=tmp_path,
        evaluation_type="full",
    )

    assert abstract == ["abstract"]
    assert detailed == ["detail"]
    assert captured["evaluation_type"] is blocks_module.EvaluationType.FULL


def test_discover_trend_metadata_files_can_include_release_spec_resources(tmp_path):
    release_dir = tmp_path / "release_spec_abc"
    for role in ("performance", "devops"):
        role_dir = release_dir / role
        resources_dir = role_dir / "resources"
        resources_dir.mkdir(parents=True)
        _write_trend_metadata(role_dir, f"Pilot.Auto v4.5.0 ({role} x2/1.0.0)")
        (role_dir / TREND_SUMMARY_FILENAME).write_text("{}", encoding="utf-8")
        _write_trend_metadata(resources_dir, f"Pilot.Auto v4.5.0 ({role} x2/1.0.0)")
        (resources_dir / TREND_SUMMARY_FILENAME).write_text("{}", encoding="utf-8")

    default_paths = discover_trend_metadata_files(tmp_path)
    included_paths = discover_trend_metadata_files(tmp_path, include_release_specs=True)

    assert default_paths == []
    assert sorted(path.parent.name for path in included_paths) == ["resources", "resources"]
