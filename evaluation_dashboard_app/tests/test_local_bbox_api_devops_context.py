import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from backend.local_bbox_api import _criteria_filter_label, _criteria_filter_namespace
from backend.local_bbox_api import _devops_context_from_name, _load_scenario_context
from backend.local_bbox_api import dataset_summary
from backend.local_bbox_api import list_parquets
from backend.local_bbox_api import scenario_devops_result
from backend.local_bbox_api import scenario_devops_frame_results
from backend.local_bbox_api import scenario_devops_tn_objects


class _FakePassFail:
    def __init__(self, success: int, fail: int, gt: int | None = None):
        self.tp_object_results = [object()] * success
        self.fp_object_results = [object()] * fail
        self.fn_objects = []
        self.tn_objects = []
        self._success = success
        self._fail = fail
        self._gt = success + fail if gt is None else gt

    def get_num_success(self) -> int:
        return self._success

    def get_num_fail(self) -> int:
        return self._fail

    def get_num_gt(self) -> int:
        return self._gt


def test_devops_context_from_name_extracts_intent_target_and_mode():
    ctx = _devops_context_from_name(
        "DevOps_V1_FN_Animal_a22295ab",
        "DevOps_V1_J6Gen2_Shiojiri_FN_ObstacleStop_Dog_PCOff_DT001790",
    )

    assert ctx["is_devops"] is True
    assert ctx["issue_type"] == "FN"
    assert ctx["intent_type"] == "target detection"
    assert ctx["focus_metric"] == "fn"
    assert ctx["target_label"] == "animal"
    assert ctx["behavior"] == "ObstacleStop"
    assert ctx["pc_mode"] == "PC off"
    assert ctx["city"] == "Shiojiri"


def test_load_scenario_context_reads_yaml_criteria(tmp_path: Path):
    run_dir = tmp_path / "devops"
    suite = "DevOps_V1_FN_Crouching_Pedestrian_ed54609d"
    scenario = "DevOps_V1_J6Gen2_Shiojiri_FN_NA_Pedestrian_PCOff_DT001715"
    scenario_dir = run_dir / suite / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(
        """
Evaluation:
  Conditions:
    Criterion:
      - CriteriaLevel: normal
        CriteriaMethod: num_gt_tp
        Filter:
          Distance: 0.0 - 30.0
        PassRate: 99
  PerceptionPassFailConfig:
    matching_threshold_list: [2, 2, 1]
    target_labels: [car, truck, pedestrian, unknown]
ScenarioDescription: FN crouching pedestrian near crossing
""",
        encoding="utf-8",
    )

    ctx = _load_scenario_context(run_dir / "current.parquet", suite, scenario)

    assert ctx["target_label"] == "pedestrian"
    assert ctx["description"] == "FN crouching pedestrian near crossing"
    assert ctx["criteria"] == [
        {
            "method": "num_gt_tp",
            "level": "normal",
            "pass_rate": 99,
            "filter": {"Distance": "0.0 - 30.0"},
        }
    ]
    assert ctx["target_labels"] == ["car", "truck", "pedestrian", "unknown"]
    assert ctx["matching_thresholds"] == [2, 2, 1]


def test_scenario_devops_result_explains_failed_num_gt_tp_gate(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(tmp_path))
    run_dir = tmp_path / "devops"
    suite = "DevOps_V1_FN_Crouching_Pedestrian_ed54609d"
    scenario = "DevOps_V1_J6Gen2_Shiojiri_FN_NA_Pedestrian_PCOff_DT001715"
    scenario_dir = run_dir / suite / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(
        """
Evaluation:
  Conditions:
    Criterion:
      - CriteriaLevel: normal
        CriteriaMethod: num_gt_tp
        Filter:
          Distance: 0.0 - 30.0
        PassRate: 99
      - CriteriaLevel: normal
        CriteriaMethod: num_gt_tp
        Filter:
          Distance: 30.0 -
        PassRate: 0
ScenarioDescription: FN crouching pedestrian near crossing
""",
        encoding="utf-8",
    )
    parquet = run_dir / "current.parquet"
    pd.DataFrame(
        [
            {
                "frame_index": 1,
                "source": "GT",
                "status": "FN",
                "x": 10.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "pedestrian",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
            {
                "frame_index": 1,
                "source": "EST",
                "status": "FP",
                "x": 12.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "car",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
        ]
    ).to_parquet(parquet)

    result = scenario_devops_result(
        {
            "path": str(parquet),
            "filters": {
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
        }
    )

    assert result["overall_pass"] is False
    assert result["gates"][0]["passed"] is False
    assert result["gates"][0]["actual_rate"] == 0.0
    assert result["gates"][0]["required_rate"] == 0.99
    assert result["gates"][1]["passed"] is None
    assert "Criterion 1 fails" in result["explanation"][0]


def test_scenario_devops_result_handles_fp_validation_num_gt_tp_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(tmp_path))
    run_dir = tmp_path / "devops"
    suite = "DevOps_V1_FP_Vegetation"
    scenario = "DevOps_V1_J6Gen2_Shiojiri_FP_IntersectionRight_Board_PCOff_DT001956"
    scenario_dir = run_dir / suite / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(
        """
Evaluation:
  Conditions:
    Criterion:
      - CriteriaLevel: normal
        CriteriaMethod: num_gt_tp
        Filter:
          Distance: null
        PassRate: 99
  PerceptionEvaluationConfig:
    evaluation_config_dict:
      matching_label_policy: allow_unknown
      merge_similar_labels: true
      evaluation_task: fp_validation
  PerceptionPassFailConfig:
    matching_threshold_list: [2, 2, 2]
    target_labels: [car, pedestrian, unknown]
ScenarioDescription: FP validation board case
""",
        encoding="utf-8",
    )
    parquet = run_dir / "current.parquet"
    pd.DataFrame(
        [
            {
                "frame_index": 100,
                "source": "GT",
                "status": "FP",
                "x": 22.0,
                "y": -11.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "false_positive",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
            {
                "frame_index": 100,
                "source": "EST",
                "status": "FP",
                "x": 22.1,
                "y": -11.1,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "pedestrian",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
            {
                "frame_index": 101,
                "source": "EST",
                "status": "FP",
                "x": 24.0,
                "y": -8.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "bicycle",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
        ]
    ).to_parquet(parquet)

    result = scenario_devops_result(
        {
            "path": str(parquet),
            "filters": {
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
        }
    )

    assert result["overall_pass"] is False
    assert result["gates"][0]["metric_label"] == "frame FP-validation pass rate"
    assert result["gates"][0]["passed"] is False
    assert result["gates"][0]["actual_rate"] == 0.0
    assert result["gates"][0]["object_fail_count"] == 2
    assert result["gates"][0]["object_total_count"] == 2
    assert result["gates"][0]["source"] == "parquet_fallback"


def test_scenario_devops_result_uses_fast_pickle_counts_for_fp_validation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(tmp_path))
    run_dir = tmp_path / "devops"
    suite = "DevOps_V1_Other_FPs_f2826285-110b-4e1b-b918-6568c851b36b"
    scenario = "DevOps_V1_J6Gen2_Komatsu_FP_ObstacleStop_Signboard_PCOff_DT001585"
    scenario_dir = run_dir / suite / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(
        """
Evaluation:
  Conditions:
    Criterion:
      - CriteriaLevel: normal
        CriteriaMethod: num_gt_tp
        Filter:
          Distance: null
        PassRate: 99
  PerceptionEvaluationConfig:
    evaluation_config_dict:
      evaluation_task: fp_validation
ScenarioDescription: FP validation signboard case
""",
        encoding="utf-8",
    )
    parquet = run_dir / "current.parquet"
    pd.DataFrame(
        [
            {
                "frame_index": 100,
                "source": "GT",
                "status": "FP",
                "x": 18.0,
                "y": 2.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "false_positive",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
            {
                "frame_index": 100,
                "source": "EST",
                "status": "FP",
                "x": 18.2,
                "y": 2.1,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "car",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
            {
                "frame_index": 101,
                "source": "GT",
                "status": "FP",
                "x": 19.0,
                "y": 2.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "false_positive",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
            {
                "frame_index": 101,
                "source": "EST",
                "status": "FP",
                "x": 19.2,
                "y": 2.1,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "pedestrian",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
        ]
    ).to_parquet(parquet)
    frames = [
        *(SimpleNamespace(frame_name=i, pass_fail_result=_FakePassFail(success=1, fail=0, gt=1)) for i in range(39)),
        SimpleNamespace(frame_name=100, pass_fail_result=_FakePassFail(success=0, fail=1, gt=1)),
        SimpleNamespace(frame_name=101, pass_fail_result=_FakePassFail(success=0, fail=1, gt=1)),
    ]
    with (scenario_dir / "scene_result.pkl").open("wb") as f:
        pickle.dump(frames, f)

    result = scenario_devops_result(
        {
            "path": str(parquet),
            "filters": {
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            },
        }
    )

    assert result["overall_pass"] is False
    assert result["gates"][0]["source"] == "scene_result.pkl_fast"
    assert result["gates"][0]["actual_rate"] == 39 / 41
    assert result["gates"][0]["passed"] is False
    assert result["gates"][0]["passed_count"] == 39
    assert result["gates"][0]["fail_count"] == 2
    assert result["gates"][0]["total_count"] == 41
    assert result["gates"][0]["object_success_count"] == 39
    assert result["gates"][0]["object_fail_count"] == 2
    assert result["gates"][0]["object_total_count"] == 41


def test_scenario_devops_tn_objects_reads_current_frame_from_scene_pickle(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(tmp_path))
    run_dir = tmp_path / "devops"
    suite = "DevOps_V1_FP_Vegetation"
    scenario = "DevOps_V1_J6Gen2_Shiojiri_FP_IntersectionRight_Board_PCOff_DT001956"
    scenario_dir = run_dir / suite / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text("Evaluation: {}\n", encoding="utf-8")
    parquet = run_dir / "current.parquet"
    pd.DataFrame(
        [
            {
                "frame_index": 120,
                "source": "GT",
                "status": "FP",
                "x": 1.0,
                "y": 2.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "false_positive",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            }
        ]
    ).to_parquet(parquet)
    tn_object = SimpleNamespace(
        unix_time=123,
        frame_id=SimpleNamespace(value="base_link"),
        state=SimpleNamespace(
            position=(12.0, -3.0, 0.4),
            orientation=(1.0, 0.0, 0.0, 0.0),
            shape=SimpleNamespace(
                size=(2.0, 4.5, 1.7),
                type=SimpleNamespace(value="bounding_box"),
                footprint=None,
            ),
            velocity=(0.0, 0.0, 0.0),
        ),
        semantic_score=1.0,
        semantic_label=SimpleNamespace(label=SimpleNamespace(value="false_positive")),
        pointcloud_num=7,
        uuid="tn-1",
        visibility=None,
    )
    frames = [
        SimpleNamespace(frame_name=119, pass_fail_result=SimpleNamespace(tn_objects=[]), transforms=None),
        SimpleNamespace(frame_name=120, pass_fail_result=SimpleNamespace(tn_objects=[tn_object]), transforms=None),
    ]
    with (scenario_dir / "scene_result.pkl").open("wb") as f:
        pickle.dump(frames, f)

    result = scenario_devops_tn_objects(
        {
            "path": str(parquet),
            "filters": {
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
                "frame_index": 120,
            },
        }
    )

    assert result["available"] is True
    assert result["row_count"] == 1
    assert result["frames"][0]["frame"] == 120
    box = result["frames"][0]["boxes"][0]
    assert box["source"] == "GT"
    assert box["status"] == "TN"
    assert box["label"] == "false_positive"
    assert box["length"] == 4.5
    assert box["width"] == 2.0
    assert box["devops_exact_source"] == "scene_result.pkl"


def test_scenario_devops_frame_results_uses_pickle_evaluator_for_current_frame(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(tmp_path))
    run_dir = tmp_path / "devops"
    suite = "DevOps_V1_FN_Animal"
    scenario = "DevOps_V1_J6Gen2_Shiojiri_FN_ObstacleStop_Dog_PCOff_DT001790"
    scenario_dir = run_dir / suite / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(
        """
Evaluation:
  Conditions:
    Criterion:
      - CriteriaLevel: normal
        CriteriaMethod: num_gt_tp
        Filter:
          Distance: null
        PassRate: 99
  PerceptionEvaluationConfig:
    evaluation_config_dict:
      evaluation_task: tracking
""",
        encoding="utf-8",
    )
    parquet = run_dir / "current.parquet"
    pd.DataFrame(
        [
            {
                "frame_index": 9,
                "source": "GT",
                "status": "TP",
                "x": 10.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "dog",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            }
        ]
    ).to_parquet(parquet)
    frames = [
        SimpleNamespace(frame_name=9, pass_fail_result=_FakePassFail(success=1, fail=0, gt=1)),
        SimpleNamespace(frame_name=10, pass_fail_result=_FakePassFail(success=0, fail=0, gt=0)),
    ]
    with (scenario_dir / "scene_result.pkl").open("wb") as f:
        pickle.dump(frames, f)

    result = scenario_devops_frame_results(
        {
            "path": str(parquet),
            "filters": {"suite_name": suite, "scenario_name": scenario, "frame_min": 9, "frame_max": 10},
        }
    )

    assert result["available"] is True
    assert result["frames"][0]["passed"] is True
    assert result["frames"][0]["gates"][0]["judged"] is True
    assert result["frames"][0]["gates"][0]["score"] == 100.0
    assert result["frames"][0]["gates"][0]["counts"]["success"] == 1
    assert result["frames"][1]["passed"] is None
    assert result["frames"][1]["gates"][0]["judged"] is False


def test_load_scenario_context_reads_matching_policy_and_merge_flag(tmp_path: Path):
    run_dir = tmp_path / "devops"
    suite = "DevOps_V1_Misc_Mislabeled_bicycles_motorcycles"
    scenario = "DevOps_V1_J6Gen2_Shiojiri_FN_NA_Motorcycle_PCOff_DT001721"
    scenario_dir = run_dir / suite / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(
        """
Evaluation:
  Conditions:
    Criterion:
      - CriteriaLevel: normal
        CriteriaMethod: num_gt_tp
        PassRate: 99
  PerceptionEvaluationConfig:
    evaluation_config_dict:
      matching_label_policy: default
      merge_similar_labels: false
      evaluation_task: tracking
ScenarioDescription: Mislabeled motorcycle case
""",
        encoding="utf-8",
    )

    ctx = _load_scenario_context(run_dir / "current.parquet", suite, scenario)

    assert ctx["matching_label_policy"] == "default"
    assert ctx["merge_similar_labels"] is False


def test_criteria_filter_namespace_preserves_region_when_distance_absent():
    ns = _criteria_filter_namespace(
        {
            "Region": {
                "x_position": "-10,20",
                "y_position": "-5,5",
            }
        }
    )

    assert ns.Distance is None
    assert ns.Region.x_position == (-10.0, 20.0)
    assert ns.Region.y_position == (-5.0, 5.0)
    assert _criteria_filter_label({"Region": {"x_position": "-10,20"}}) == "x -10 - 20 m"


def test_dataset_summary_lists_devops_suite_scenarios_missing_from_parquet(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(tmp_path))
    run_dir = tmp_path / "devops"
    resources = run_dir / "resources"
    resources.mkdir(parents=True)
    (resources / "summary.json").write_text(
        """
{
  "DevOps": {
    "Suite pass rate": {
      "DevOps_V1_FP_Vegetation": {"passed": 1, "total": 2}
    }
  }
}
""",
        encoding="utf-8",
    )
    suite = "DevOps_V1_FP_Vegetation_ba7a1141-9b9a-4db2-a2f8-6da49b524285"
    loaded = "DevOps_V1_J6Gen2_Shiojiri_FP_IntersectionRight_Board_PCOff_DT001956"
    missing = "DevOps_V1_J6Gen2_Komatsu_FP_ObstacleStop_Plant_PCOff_DT001924"
    for scenario in (loaded, missing):
        scenario_dir = run_dir / suite / scenario
        scenario_dir.mkdir(parents=True)
        (scenario_dir / "scenario.yaml").write_text(
            """
Evaluation:
  Conditions:
    Criterion:
      - CriteriaLevel: normal
        CriteriaMethod: num_gt_tp
        PassRate: 99
  PerceptionEvaluationConfig:
    evaluation_config_dict:
      evaluation_task: fp_validation
ScenarioDescription: FP vegetation inventory case
""",
            encoding="utf-8",
        )

    parquet = run_dir / "current.parquet"
    pd.DataFrame(
        [
            {
                "frame_index": 1,
                "source": "EST",
                "status": "FP",
                "x": 10.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "pedestrian",
                "suite_name": suite,
                "scenario_name": loaded,
                "topic_name": "perception.object_recognition.tracking.objects",
            }
        ]
    ).to_parquet(parquet)

    result = dataset_summary(
        {
            "path": str(parquet),
            "filters": {"topic_name": "perception.object_recognition.tracking.objects"},
            "limit": 20,
        }
    )

    by_name = {item["scenario_name"]: item for item in result["items"]}
    assert loaded in by_name
    assert missing in by_name
    assert by_name[missing]["rows"] == 0
    assert by_name[missing]["devops"]["unavailable"] is True
    assert "no bbox/evaluation rows" in by_name[missing]["devops"]["unavailable_reason"]


def test_dataset_summary_uses_criteria_pass_for_devops_badge_context(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(tmp_path))
    run_dir = tmp_path / "devops"
    suite = "DevOps_V1_FN_Pedestrian"
    scenario = "DevOps_V1_J6Gen2_Shiojiri_FN_NA_Pedestrian_PCOff_DT001710"
    scenario_dir = run_dir / suite / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(
        """
Evaluation:
  Conditions:
    Criterion:
      - CriteriaLevel: normal
        CriteriaMethod: num_gt_tp
        Filter:
          Distance: 0.0 - 30.0
        PassRate: 99
      - CriteriaLevel: normal
        CriteriaMethod: num_gt_tp
        Filter:
          Distance: 30.0 - 75.0
        PassRate: 99
ScenarioDescription: FN pedestrian case with ignored empty far bucket
""",
        encoding="utf-8",
    )
    parquet = run_dir / "current.parquet"
    rows = []
    for frame in range(1, 101):
        rows.append(
            {
                "frame_index": frame,
                "source": "GT",
                "status": "TP" if frame > 1 else "FN",
                "x": 10.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "label": "pedestrian",
                "suite_name": suite,
                "scenario_name": scenario,
                "topic_name": "perception.object_recognition.tracking.objects",
            }
        )
    pd.DataFrame(rows).to_parquet(parquet)

    result = dataset_summary(
        {
            "path": str(parquet),
            "filters": {"topic_name": "perception.object_recognition.tracking.objects"},
            "limit": 20,
            "include_criteria_results": True,
        }
    )

    item = next(x for x in result["items"] if x["scenario_name"] == scenario)
    assert item["fn"] == 1
    assert item["devops"]["criteria_result"]["overall_pass"] is True
    assert item["devops"]["criteria_result"]["gate_count"] == 1


def test_dataset_summary_uses_persistent_cache_and_invalidates_on_parquet_change(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(tmp_path))
    run_dir = tmp_path / "devops"
    run_dir.mkdir(parents=True)
    suite = "DevOps_V1_FN_Pedestrian"
    scenario = "DevOps_V1_J6Gen2_Shiojiri_FN_NA_Pedestrian_PCOff_DT001710"
    parquet = run_dir / "current.parquet"

    def write(rows: int) -> None:
        pd.DataFrame(
            [
                {
                    "frame_index": i,
                    "source": "GT",
                    "status": "TP",
                    "x": 10.0,
                    "y": 0.0,
                    "z": 0.0,
                    "yaw": 0.0,
                    "length": 1.0,
                    "width": 1.0,
                    "height": 1.0,
                    "label": "pedestrian",
                    "suite_name": suite,
                    "scenario_name": scenario,
                    "topic_name": "perception.object_recognition.tracking.objects",
                }
                for i in range(rows)
            ]
        ).to_parquet(parquet)

    payload = {
        "path": str(parquet),
        "filters": {"topic_name": "perception.object_recognition.tracking.objects"},
        "limit": 20,
    }
    write(2)
    first = dataset_summary(payload)
    second = dataset_summary(payload)
    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert second["items"][0]["rows"] == 2

    write(3)
    third = dataset_summary(payload)
    assert third["cache"]["hit"] is False
    assert third["items"][0]["rows"] == 3


def test_list_parquets_maps_host_data_root_and_uses_cache(tmp_path: Path, monkeypatch):
    host_root = tmp_path / "host_data"
    container_root = tmp_path / "container_data"
    devops_dir = container_root / "pilot" / "devops"
    devops_dir.mkdir(parents=True)
    parquet = devops_dir / "current.parquet"
    pd.DataFrame(
        [
            {
                "frame_index": 1,
                "source": "GT",
                "status": "TP",
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
                "length": 1.0,
                "width": 1.0,
                "height": 1.0,
                "scenario_name": "case",
            }
        ]
    ).to_parquet(parquet)
    monkeypatch.setenv("EVAL_DASHBOARD_DATA_ROOT", str(container_root))
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(container_root))
    monkeypatch.setenv("LOCAL_EVALUATOR_HOST_DATA_ROOT", str(host_root))
    monkeypatch.setenv("LOCAL_EVALUATOR_CONTAINER_DATA_ROOT", str(container_root))

    payload = {"root": str(host_root / "pilot" / "devops"), "limit": 20, "bbox_only": True}
    first = list_parquets(payload)
    second = list_parquets(payload)

    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert first["items"][0]["path"] == str(parquet)


def test_list_parquets_cache_invalidates_when_nested_parquet_is_added(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_BBOX_ALLOWED_ROOTS", str(tmp_path))
    root = tmp_path / "data"
    first_dir = root / "old_run" / "devops"
    first_dir.mkdir(parents=True)
    first_parquet = first_dir / "current.parquet"

    def write_parquet(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "frame_index": 1,
                    "source": "GT",
                    "status": "TP",
                    "x": 1.0,
                    "y": 0.0,
                    "z": 0.0,
                    "yaw": 0.0,
                    "length": 1.0,
                    "width": 1.0,
                    "height": 1.0,
                    "scenario_name": "case",
                }
            ]
        ).to_parquet(path)

    write_parquet(first_parquet)
    payload = {"root": str(root), "limit": 20, "bbox_only": True}
    first = list_parquets(payload)
    second = list_parquets(payload)
    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert [x["path"] for x in second["items"]] == [str(first_parquet)]

    added_parquet = root / "eval_2.4a_0710_streampetr_ptv3_off" / "devops" / "current.parquet"
    write_parquet(added_parquet)
    third = list_parquets(payload)

    assert third["cache"]["hit"] is False
    assert {x["path"] for x in third["items"]} == {str(first_parquet), str(added_parquet)}
