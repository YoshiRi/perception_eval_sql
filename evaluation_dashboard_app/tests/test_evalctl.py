"""evalctl is what agents and scripts trust to drive workflows, so its request
building and refusal logic matter more than its printing."""

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "evalctl", Path(__file__).resolve().parent.parent / "scripts" / "evalctl.py"
)
evalctl = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("evalctl", evalctl)
_SPEC.loader.exec_module(evalctl)

_JST = timezone(timedelta(hours=9))


def _args(**overrides):
    """Parse a real command line so defaults stay honest, then override."""
    argv = overrides.pop("argv")
    args = evalctl.build_parser().parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _fake_api(routes):
    calls = []

    def api(args, path, payload=None):
        calls.append((path, payload or {}))
        result = routes[path]
        return result(payload or {}) if callable(result) else result

    return api, calls


# ------------------------------------------------------------- release metadata


def test_release_metadata_autofills_from_today_and_past_releases(monkeypatch):
    routes = {
        "/api/workflow_trends": {
            "items": [{
                "topic": "obstacle",
                "jobs": {"full": {"metadata": {
                    "release_group": "2025Q3", "data_count": "150", "topic_name": "obstacle",
                }}},
            }]
        }
    }
    api, _ = _fake_api(routes)
    monkeypatch.setattr(evalctl, "api", api)
    args = _args(argv=["release", "beta/v4.5.0", "--dry-run"])
    metadata = evalctl._auto_release_metadata(args)
    assert metadata["pilot_auto_version"] == "Pilot.Auto v4.5.0"
    assert metadata["date"] == datetime.now(_JST).strftime("%Y.%m.%d")
    assert metadata["release_group"] == "2025Q3"
    assert metadata["data_count"] == "150"
    assert metadata["topic_name"] == "obstacle"
    assert metadata["tags"] == ["trend"]


def test_release_metadata_honours_explicit_overrides(monkeypatch):
    api, _ = _fake_api({"/api/workflow_trends": {"items": []}})
    monkeypatch.setattr(evalctl, "api", api)
    args = _args(argv=[
        "release", "v9.9.9", "--version", "Pilot.Auto v10", "--date", "2020.01.01",
        "--release-group", "G", "--data-count", "7", "--topic", "tlr",
        "--set", "extra_field=hello",
    ])
    metadata = evalctl._auto_release_metadata(args)
    assert metadata["pilot_auto_version"] == "Pilot.Auto v10"
    assert metadata["date"] == "2020.01.01"
    assert metadata["release_group"] == "G"
    assert metadata["data_count"] == "7"
    assert metadata["topic_name"] == "tlr"
    assert metadata["extra_field"] == "hello"


def test_release_metadata_survives_a_server_without_trends(monkeypatch):
    def api(args, path, payload=None):
        raise evalctl.ApiError("no trends here")

    monkeypatch.setattr(evalctl, "api", api)
    args = _args(argv=["release", "beta/v4.3.2", "--dry-run"])
    metadata = evalctl._auto_release_metadata(args)
    assert metadata["pilot_auto_version"] == "Pilot.Auto v4.3.2"
    assert metadata["release_group"] == "Pilot.Auto v4.3.2"


# ------------------------------------------------------------------- preflight


def test_start_refuses_when_no_worker_is_alive(monkeypatch, capsys):
    routes = {
        "/api/workflow_health": {
            "authorized": True, "queue_enabled": True,
            "workers_alive": 0, "workers_reason": "No RQ worker is listening.",
        },
    }
    api, calls = _fake_api(routes)
    monkeypatch.setattr(evalctl, "api", api)
    code = evalctl.main(["start", "beta/v1"])
    assert code == 2
    assert "worker" in capsys.readouterr().err.lower()
    assert [path for path, _ in calls] == ["/api/workflow_health"]  # nothing was started


def test_start_builds_the_documented_payload(monkeypatch):
    routes = {
        "/api/workflow_health": {"authorized": True, "queue_enabled": True, "workers_alive": 1},
        "/api/workflow_catalogs": {"presets": [
            {"display_name": "Performance Test", "catalog_id": "cat-1", "integration_id": "int-1"},
        ]},
        "/api/workflow_start": {"ok": True, "task_id": "t-1", "kind": "perception",
                                "run_name": "r", "output_path": "/data/r", "target_check": {}},
    }
    api, calls = _fake_api(routes)
    monkeypatch.setattr(evalctl, "api", api)
    code = evalctl.main(["start", "beta/v4.3.2", "--catalog", "performance",
                         "--suite", "s1", "--suite", "s2"])
    assert code == 0
    payload = dict(calls)["/api/workflow_start"]
    assert payload["kind"] == "perception"
    assert payload["target_name"] == "beta/v4.3.2"
    assert payload["catalog_id"] == "cat-1" and payload["integration_id"] == "int-1"
    assert payload["suite_ids"] == ["s1", "s2"]
    assert payload["check_target"] is True


def test_dry_run_skips_the_preflight(monkeypatch):
    routes = {
        "/api/workflow_catalogs": {"presets": [
            {"display_name": "P", "catalog_id": "c", "integration_id": "i"},
        ]},
        "/api/workflow_start": {"ok": True, "dry_run": True, "parameters": {}},
    }
    api, calls = _fake_api(routes)
    monkeypatch.setattr(evalctl, "api", api)
    assert evalctl.main(["start", "beta/v1", "--dry-run"]) == 0
    assert "/api/workflow_health" not in [path for path, _ in calls]


def test_unknown_catalog_name_is_a_clear_error(monkeypatch, capsys):
    routes = {
        "/api/workflow_health": {"authorized": True, "queue_enabled": True, "workers_alive": 1},
        "/api/workflow_catalogs": {"presets": [
            {"display_name": "Performance Test", "catalog_id": "c", "integration_id": "i"},
        ]},
    }
    api, _ = _fake_api(routes)
    monkeypatch.setattr(evalctl, "api", api)
    assert evalctl.main(["start", "beta/v1", "--catalog", "nope"]) == 2
    assert "Performance Test" in capsys.readouterr().err
