"""The workflow routes start real, expensive pipelines, so what they hand the worker matters.

Two things are load-bearing and covered here: the ``parameters`` dict must match what the
Streamlit launcher builds (a mismatch means the worker misbehaves in ways only visible
hours later), and a bad request must be refused before anything reaches the queue.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend import workflow_api


class _Handler:
    """Minimal stand-in for BaseHTTPRequestHandler: headers only, as these routes read no body."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


@pytest.fixture()
def data_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    root.mkdir(parents=True)
    monkeypatch.setenv("EVAL_DASHBOARD_DATA_ROOT", str(root))
    monkeypatch.delenv("EVAL_EXPORT_TOKEN", raising=False)
    monkeypatch.delenv("EVAL_EXPORT_REQUIRE_TOKEN", raising=False)
    # lib.path_utils caches the resolved root for the life of the process, which is right
    # for a server and wrong for a test that just moved it.
    from lib import path_utils

    monkeypatch.setattr(path_utils, "_DATA_ROOT", None)
    return root


def _evaluator_payload(**extra):
    return {
        "kind": "perception",
        "project_id": "x2_dev",
        "target_name": "beta/v4.3.2",
        "catalog_id": "cat-1",
        "integration_id": "int-1",
        **extra,
    }


def _release_payload(**extra):
    return {
        "kind": "release",
        "project_id": "x2_dev",
        "target_name": "beta/v4.3.2",
        "metadata_text": workflow_api.default_release_metadata_text("beta/v4.3.2"),
        **extra,
    }


# -------------------------------------------------------------------- param building


def test_perception_params_match_the_page_contract(data_root):
    kind, task_type, params = workflow_api.build_params(_evaluator_payload())
    assert (kind, task_type) == ("perception", "run_evaluator_and_process")
    assert params["workflow_kind"] == workflow_api.WORKFLOW_KIND_PERCEPTION
    assert params["download_type"] == "archives"
    assert params["phase"] == workflow_api.DEFAULT_PERCEPTION_PHASE
    assert params["run_eval"] is True and params["generate_parquet"] is True
    assert params["max_wait_seconds"] == workflow_api.DEFAULT_MAX_WAIT_HOURS * 3600
    # Keys the worker reads positionally-by-name; a missing one surfaces as a KeyError
    # inside the job, long after the request succeeded.
    for key in ("suite_ids", "max_retries", "clean_build", "keep_zip_files", "eval_overwrite",
                "log_expiration_time_in_days", "large_file_mb", "poll_interval"):
        assert key in params


def test_tlr_overrides_settings_the_pipeline_cannot_honour(data_root):
    """TLR reads result JSON; archives, eval and parquet are not part of that path."""
    kind, task_type, params = workflow_api.build_params(_evaluator_payload(
        kind="tlr", download_type="archives", run_eval=True, generate_parquet=True,
    ))
    assert (kind, task_type) == ("tlr", "run_evaluator_and_process")
    assert params["workflow_kind"] == workflow_api.WORKFLOW_KIND_TLR
    assert params["download_type"] == "result_json"
    assert params["phase"] == ""
    assert params["run_eval"] is False
    assert params["generate_parquet"] is False
    assert params["eval_recursive"] is True
    assert params["skip_large_file"] is False


def test_release_params_carry_the_fixed_catalogs_and_parsed_metadata(data_root):
    kind, task_type, params = workflow_api.build_params(_release_payload())
    assert (kind, task_type) == ("release", "run_release_specsheet_workflow")
    assert params["performance_catalog_id"] == workflow_api.RELEASE_PERFORMANCE_CATALOG_ID
    assert params["devops_integration_id"] == workflow_api.RELEASE_DEVOPS_INTEGRATION_ID
    assert params["version"] == "Pilot.Auto v4.3.2"
    assert params["trend_metadata"]["tags"] == ["trend"]
    assert params["overwrite"] is True
    assert params["skip_large_file"] is workflow_api.RELEASE_SKIP_LARGE_FILE
    # The optional catalog is opt-in, and its ids stay empty until it is enabled.
    assert params["optional_catalog_id"] == "" and params["optional_job_id"] == ""


def test_release_optional_catalog_is_filled_in_when_enabled(data_root):
    _, _, params = workflow_api.build_params(_release_payload(
        optional_catalog_enabled=True, optional_job_id="job-9",
    ))
    assert params["optional_catalog_id"] == workflow_api.RELEASE_OPTIONAL_CATALOG_ID
    assert params["optional_job_id"] == "job-9"


def test_output_path_defaults_to_the_pages_naming_scheme(data_root):
    _, _, params = workflow_api.build_params(_evaluator_payload())
    name = Path(params["output_path"]).name
    assert name.startswith("eval_beta_v4_3_2_")
    assert params["output_path"].startswith(str(data_root))


def test_numeric_fields_are_clamped_not_trusted(data_root):
    _, _, params = workflow_api.build_params(_evaluator_payload(
        poll_interval=99999, max_wait_hours=-5,
    ))
    assert params["poll_interval"] == 300
    assert params["max_wait_seconds"] == 0  # 0 means "no app-side timeout"


# ------------------------------------------------------------------------ refusals


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"kind": "nope"}, "Unknown kind"),
        (_evaluator_payload(project_id="", target_name=""), "project_id, target_name"),
        (_evaluator_payload(catalog_id=""), "catalog_id"),
        (_evaluator_payload(integration_id=""), "integration_id"),
        ({"kind": "release", "project_id": "p", "target_name": "t"}, "needs trend metadata"),
        (_evaluator_payload(download_type="tarball"), "download_type must be"),
    ],
)
def test_bad_requests_are_refused(data_root, payload, expected):
    with pytest.raises(workflow_api.WorkflowError, match=expected):
        workflow_api.build_params(payload)


def test_output_path_cannot_escape_the_data_root(data_root):
    """The worker writes gigabytes wherever this points, so the sandbox is the boundary."""
    with pytest.raises(workflow_api.WorkflowError, match="Invalid output folder"):
        workflow_api.build_params(_evaluator_payload(output_path="../../etc/evil"))


def test_release_metadata_is_validated_not_just_parsed(data_root):
    with pytest.raises(workflow_api.WorkflowError, match="Invalid trend metadata"):
        workflow_api.build_params(_release_payload(metadata_text="tags: [trend]\n"))
    # A dict is accepted as-is, but still has to carry the fields the spec sheet needs.
    with pytest.raises(workflow_api.WorkflowError, match="missing `date`"):
        workflow_api.build_params(_release_payload(
            metadata_text="",
            trend_metadata={"release_group": "g", "pilot_auto_version": "v", "data_count": "1"},
        ))


# --------------------------------------------------------------------------- routes


def test_health_answers_a_caller_it_would_otherwise_refuse(data_root, monkeypatch):
    """The local app calls this to decide whether to offer the page at all."""
    monkeypatch.setenv("EVAL_EXPORT_REQUIRE_TOKEN", "1")
    monkeypatch.setenv("EVAL_EXPORT_TOKEN", "secret")
    health = workflow_api.workflow_health(_Handler(), {})
    assert health["service"] == "eval_dashboard_workflow"
    assert health["authorized"] is False
    assert "token" in health["auth_reason"].lower()
    assert [kind["name"] for kind in health["kinds"]] == ["perception", "tlr", "release"]
    assert health["defaults"]["metadata_text"].startswith("tags: [trend]")


def test_start_requires_authorization(data_root, monkeypatch):
    monkeypatch.setenv("EVAL_EXPORT_REQUIRE_TOKEN", "1")
    monkeypatch.setenv("EVAL_EXPORT_TOKEN", "secret")
    from backend import export_api

    with pytest.raises(export_api.ExportAuthError):
        workflow_api.workflow_start(_Handler(), _evaluator_payload())


def test_dry_run_start_queues_nothing(data_root, monkeypatch):
    def _explode(*args, **kwargs):
        raise AssertionError("dry_run must not reach the queue")

    monkeypatch.setattr(workflow_api, "enqueue", _explode)
    result = workflow_api.workflow_start(_Handler(), _evaluator_payload(dry_run=True))
    assert result["dry_run"] is True
    assert result["parameters"]["catalog_id"] == "cat-1"


def test_start_reports_a_server_that_cannot_queue(data_root, monkeypatch):
    """Without USE_TASK_QUEUE the row would sit pending forever, so refuse up front."""
    monkeypatch.setattr(workflow_api, "enqueue", workflow_api.enqueue)
    monkeypatch.delenv("USE_TASK_QUEUE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(workflow_api.WorkflowError, match="cannot queue workflows"):
        workflow_api.workflow_start(_Handler(), _evaluator_payload())


def test_enqueue_attributes_the_run_and_files_it_under_a_real_identity(data_root, monkeypatch):
    created: dict = {}
    enqueued: dict = {}

    fake_db = SimpleNamespace(
        is_task_queue_enabled=lambda: True,
        create_task=lambda task_type, params, session_id=None: created.update(
            type=task_type, params=params, session_id=session_id
        ) or "task-42",
        update_task_rq_job_id=lambda task_id, rq_id: enqueued.update(rq_id=rq_id),
        update_task_status=lambda *a, **k: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "lib.db", fake_db)

    class _Queue:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def enqueue(self, fn, task_id, task_type, params, **kwargs):
            enqueued.update(task_id=task_id, task_type=task_type, timeout=kwargs.get("job_timeout"))
            return SimpleNamespace(id="rq-7")

    monkeypatch.setitem(__import__("sys").modules, "rq", SimpleNamespace(Queue=_Queue))
    monkeypatch.setitem(
        __import__("sys").modules, "redis",
        SimpleNamespace(Redis=SimpleNamespace(from_url=lambda url: object())),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "worker.tasks", SimpleNamespace(run_job=lambda *a: None)
    )

    _, _, params = workflow_api.build_params(_evaluator_payload())
    task_id = workflow_api.enqueue(
        "run_evaluator_and_process", params, {"via": "cloudflare", "actor": "lei.gu@tier4.jp"}
    )
    assert task_id == "task-42"
    assert enqueued["rq_id"] == "rq-7"
    assert created["params"]["_requester"]["email"] == "lei.gu@tier4.jp"
    assert created["session_id"] == "lei.gu@tier4.jp"


def test_a_token_caller_gets_an_unowned_row(data_root, monkeypatch):
    """session_id is the dashboard's "my tasks" filter; "token" is not a user."""
    created: dict = {}
    fake_db = SimpleNamespace(
        is_task_queue_enabled=lambda: True,
        create_task=lambda task_type, params, session_id=None: created.update(
            session_id=session_id, params=params
        ) or "task-1",
        update_task_rq_job_id=lambda *a: None,
        update_task_status=lambda *a, **k: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "lib.db", fake_db)
    monkeypatch.setitem(
        __import__("sys").modules, "rq",
        SimpleNamespace(Queue=lambda *a, **k: SimpleNamespace(
            enqueue=lambda *args, **kwargs: SimpleNamespace(id="rq-1"))),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "redis",
        SimpleNamespace(Redis=SimpleNamespace(from_url=lambda url: object())),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "worker.tasks", SimpleNamespace(run_job=lambda *a: None)
    )

    _, _, params = workflow_api.build_params(_evaluator_payload())
    workflow_api.enqueue("run_evaluator_and_process", params, {"via": "token", "actor": "token"})
    assert created["session_id"] is None
    assert created["params"]["_requester"]["source"] == "workflow_api:token"


def test_task_views_only_report_workflow_types(data_root, monkeypatch):
    """A laptop client asking for workflows must not be shown unrelated server activity."""
    rows = [
        {"id": "a" * 32, "type": "run_evaluator_and_process", "status": "completed",
         "parameters": {"target_name": "beta/v1", "output_path": "/data/eval_beta_v1"},
         "created_at": None, "updated_at": None},
        {"id": "b" * 32, "type": "prepare_pr_test_branch", "status": "completed",
         "parameters": {}, "created_at": None, "updated_at": None},
    ]
    monkeypatch.setitem(
        __import__("sys").modules, "lib.db",
        SimpleNamespace(list_recent_tasks=lambda **kwargs: rows),
    )
    result = workflow_api.workflow_tasks(_Handler(), {})
    assert [item["id"] for item in result["items"]] == ["a" * 32]
    assert result["items"][0]["run_name"] == "eval_beta_v1"


# ---------------------------------------------------------------------- trend data


_RELEASE_METADATA = {
    "release_group": "2025Q1",
    "topic_name": "obstacle",
    "pilot_auto_version": "Pilot.Auto v1.2.3",
    "data_count": "120",
    "date": "2025.03.01",
    "description": "spring release",
}

# One full-performance table: mAP averages to 0.5 across the two labels.
_FULL_SUMMARY = {
    "blocks": [
        {
            "header": "全数データセット評価",
            "evaluation_type": "full",
            "tables": [{"data": {"mAP": {"car": 0.4, "bus": 0.6}, "precision": {"car": 0.8}}}],
        }
    ]
}

# No "blocks" and non-empty classifies as devops; 8/10 cases pass.
_DEVOPS_SUMMARY = {"cut_in": {"urban": {"passed": 8, "total": 10}}}


def _write_trend_run(root: Path, name: str, metadata: dict, summary: dict) -> None:
    resources = root / name / "resources"
    resources.mkdir(parents=True)
    (resources / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, allow_unicode=True), encoding="utf-8"
    )
    (resources / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture()
def trend_root(data_root: Path) -> Path:
    # A release pair (full + devops sharing identical release metadata groups into one
    # entry) plus an older standalone run, so ordering and grouping are both exercised.
    _write_trend_run(data_root, "eval_full_v123", _RELEASE_METADATA, _FULL_SUMMARY)
    _write_trend_run(data_root, "eval_devops_v123", _RELEASE_METADATA, _DEVOPS_SUMMARY)
    _write_trend_run(
        data_root,
        "eval_full_v100",
        {**_RELEASE_METADATA, "release_group": "2024Q4", "pilot_auto_version": "Pilot.Auto v1.0.0",
         "date": "2024.11.01", "description": "autumn release"},
        _FULL_SUMMARY,
    )
    return data_root


def test_trends_report_release_groups_newest_first_with_page_metrics(trend_root):
    result = workflow_api.workflow_trends(_Handler(), {})
    assert result["total_groups"] == 2

    newest, older = result["items"]
    assert newest["date"] == "2025.03.01" and older["date"] == "2024.11.01"
    assert newest["version"] == "Pilot.Auto v1.2.3"
    assert newest["release_group"] == "2025Q1"
    assert newest["topic"] == "obstacle"
    # The full and devops runs share release metadata, so they are one release entry.
    assert newest["roles"] == ["devops", "full"]

    metrics = newest["metrics"]
    assert metrics["mAP"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.8)
    assert metrics["overall_pass_rate"] == pytest.approx(80.0)
    assert metrics["scenario_count"] == 10

    # Raw summaries are opt-in; the default listing stays light.
    assert "summary" not in newest["jobs"]["full"]
    assert "cases" not in newest


def test_trends_honour_filters_and_optional_payloads(trend_root):
    assert workflow_api.workflow_trends(_Handler(), {"topic": "lane_change"})["items"] == []

    by_query = workflow_api.workflow_trends(_Handler(), {"q": "autumn"})["items"]
    assert [item["date"] for item in by_query] == ["2024.11.01"]

    limited = workflow_api.workflow_trends(_Handler(), {"limit": 1})["items"]
    assert [item["date"] for item in limited] == ["2025.03.01"]

    full = workflow_api.workflow_trends(
        _Handler(), {"include_summary": True, "include_cases": True}
    )["items"][0]
    assert full["jobs"]["full"]["summary"] == _FULL_SUMMARY
    assert full["cases"] == [
        {"major_category": "cut_in", "mid_category": "urban", "minor_category": "urban",
         "case_name": "urban", "passed": 8, "total": 10, "pass_rate": pytest.approx(80.0)},
    ]


def test_trends_survive_a_summary_the_extractors_reject(trend_root):
    """One broken release on disk must not make the whole history unqueryable."""
    _write_trend_run(
        trend_root,
        "eval_full_broken",
        {**_RELEASE_METADATA, "release_group": "broken", "date": "2025.04.01"},
        {"blocks": [{"header": "全数データセット評価", "tables": []}]},
    )
    result = workflow_api.workflow_trends(_Handler(), {})
    assert result["total_groups"] == 3
    broken = result["items"][0]
    assert broken["date"] == "2025.04.01"
    assert "full" in broken["metrics"]["errors"]
    # The healthy releases still report numbers.
    assert result["items"][1]["metrics"]["mAP"] == pytest.approx(0.5)


def test_trends_require_authorization(trend_root, monkeypatch):
    monkeypatch.setenv("EVAL_EXPORT_REQUIRE_TOKEN", "1")
    monkeypatch.setenv("EVAL_EXPORT_TOKEN", "secret")
    from backend import export_api

    with pytest.raises(export_api.ExportAuthError):
        workflow_api.workflow_trends(_Handler(), {})


def test_a_datetime_survives_the_json_responder(data_root):
    """_json_safe does not serialise datetimes, so the view has to stringify them."""
    from datetime import datetime

    view = workflow_api._task_view({
        "id": "x", "type": "run_evaluator_and_process", "status": "running",
        "parameters": {}, "created_at": datetime(2026, 7, 31, 9, 30), "updated_at": None,
    })
    assert view["created_at"] == "2026-07-31T09:30:00"
    assert view["active"] is True
