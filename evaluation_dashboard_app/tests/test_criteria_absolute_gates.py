"""Tests for lib/criteria_absolute_gates."""

import pandas as pd
import pytest

from lib.criteria_absolute_gates import (
    MetricGateSpec,
    evaluate_scenario_gates,
    export_gate_result,
    failing_scenarios_table,
    gate_summary,
    infer_criteria_count,
)


def test_infer_criteria_count():
    # 3 base + 2 blocks * 11 = 25
    raw = pd.DataFrame([[0] * 25])
    assert infer_criteria_count(raw, 11) == 2
    assert infer_criteria_count(raw, 11, max_criteria=1) == 1
    raw_empty = pd.DataFrame()
    assert infer_criteria_count(raw_empty, 11) == 1


def test_infer_criteria_count_with_dataset_column():
    # 4 base + 2 blocks * 11 = 26
    raw = pd.DataFrame([[0] * 26], columns=["Scenario", "Dataset", "Option", "GT_OBJ"] + [f"c{i}" for i in range(22)])
    assert infer_criteria_count(raw, 11) == 2


def test_mean_mode_pass_rate_only():
    df = pd.DataFrame(
        {
            "Scenario": ["s1", "s1", "s2"],
            "Option": ["o1", "o2", "o1"],
            "GT_OBJ": ["g", "g", "g"],
            "pass_rate": [100.0, 90.0, 96.0],
            "nm": [1.0, 1.0, 1.0],
        }
    )
    r = evaluate_scenario_gates(df, 95.0, None)
    # s1 mean 95 -> pass; s2 mean 96 -> pass
    assert len(r) == 2
    assert r["scenario_pass"].tolist() == [True, True]
    s = gate_summary(r)
    assert s["n_pass"] == 2 and s["all_pass"] is True


def test_mean_mode_groups_by_dataset_when_present():
    df = pd.DataFrame(
        {
            "Scenario": ["s1", "s1"],
            "Dataset": ["d1", "d2"],
            "pass_rate": [100.0, 50.0],
            "nm": [1.0, 1.0],
        }
    )
    r = evaluate_scenario_gates(df, 95.0, None)
    assert len(r) == 2
    assert set(r["Dataset"]) == {"d1", "d2"}
    assert r.loc[r["Dataset"] == "d1", "scenario_pass"].iloc[0] is True
    assert r.loc[r["Dataset"] == "d2", "scenario_pass"].iloc[0] is False


def test_mean_mode_pass_rate_fail():
    df = pd.DataFrame(
        {
            "Scenario": ["a", "b"],
            "pass_rate": [80.0, 99.0],
            "nm": [1.0, 1.0],
        }
    )
    r = evaluate_scenario_gates(df, 95.0, None)
    assert r.loc[r["Scenario"] == "a", "scenario_pass"].iloc[0] is False
    assert r.loc[r["Scenario"] == "b", "scenario_pass"].iloc[0] is True


def test_mean_mode_second_metric_max():
    df = pd.DataFrame(
        {
            "Scenario": ["x", "x", "x"],
            "pass_rate": [100.0, 100.0, 100.0],
            "nm": [1.0, 3.0, 2.0],
        }
    )
    spec = MetricGateSpec("nm", "<=", 2.5)
    r = evaluate_scenario_gates(df, 95.0, spec)
    assert len(r) == 1
    assert r["metric_agg"].iloc[0] == 3.0
    assert r["scenario_pass"].iloc[0] is False


def test_failing_scenarios_table():
    df = pd.DataFrame({"Scenario": ["p", "q"], "pass_rate": [50.0, 99.0], "nm": [1.0, 1.0]})
    r = evaluate_scenario_gates(df, 95.0, None)
    f = failing_scenarios_table(r)
    assert len(f) == 1
    assert f["Scenario"].iloc[0] == "p"


def test_export_gate_result_drops_metric_without_spec():
    df = pd.DataFrame({"Scenario": ["a"], "pass_rate": [99.0], "nm": [1.0]})
    r = evaluate_scenario_gates(df, 95.0, None)
    ex = export_gate_result(r, None)
    assert "metric_agg" not in ex.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
