"""The in-app routes are what make the packaged app usable without a terminal."""

import time
from pathlib import Path

import pytest

from client import config, webapi


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("EVALDASH_HOME", str(tmp_path / "home"))
    config.ensure_dirs()
    # A build may have written client/_defaults.py with a baked-in server; these tests
    # describe a plain source checkout, so neutralise it explicitly.
    monkeypatch.setattr(config, "DEFAULT_SERVER", "")
    monkeypatch.setattr(config, "DEFAULT_T4_BASE_URL", "")
    monkeypatch.delenv("EVALDASH_SERVER", raising=False)
    monkeypatch.delenv("EVALDASH_TOKEN", raising=False)
    # Each test starts with no in-flight job; the manager is module-level state.
    webapi._JOB = None
    yield tmp_path / "home"
    webapi._JOB = None


# --------------------------------------------------------------------- state view


def test_state_lists_tiers_and_an_empty_workspace(home):
    state = webapi.client_state({})
    assert [t["name"] for t in state["tiers"]] == ["minimal", "criteria", "full", "raw"]
    assert all(t["hint"] for t in state["tiers"])
    assert state["local_runs"] == []
    assert state["job"] is None
    assert state["config"]["workspace"] == str(config.workspace_dir())


def test_state_never_returns_the_token(home):
    config.Config(server_url="https://example.test", token="super-secret").save()
    state = webapi.client_state({})
    assert state["config"]["token_set"] is True
    # The browser has no need for the secret, so it must not be in the payload at all.
    assert "super-secret" not in str(state)
    assert "token" not in {k for k in state["config"] if k != "token_set"}


def test_state_reports_cloudflare_configuration_without_leaking_it(home):
    config.Config(server_url="https://x.test", cf_client_id="id", cf_client_secret="shh").save()
    state = webapi.client_state({})
    assert state["config"]["cf_configured"] is True
    assert "shh" not in str(state)


# -------------------------------------------------------------------------- login


def test_login_requires_a_server(home):
    with pytest.raises(ValueError, match="server URL is required"):
        webapi.client_login({})


def test_login_reports_an_unreachable_server(home):
    from client.remote import RemoteError

    with pytest.raises(RemoteError):
        webapi.client_login({"server_url": "http://127.0.0.1:1", "token": "t"})


def test_login_does_not_persist_a_failed_attempt(home):
    from client.remote import RemoteError

    with pytest.raises(RemoteError):
        webapi.client_login({"server_url": "http://127.0.0.1:1", "token": "t"})
    assert config.Config.load().server_url == ""


# --------------------------------------------------------------------------- pull


def test_pull_requires_a_run(home):
    with pytest.raises(ValueError, match="run name is required"):
        webapi.client_pull({})


def test_only_one_pull_at_a_time(home):
    job = webapi.PullJob("run_a", "all", "criteria", False)
    job.state = "downloading"
    webapi._JOB = job
    with pytest.raises(ValueError, match="already running"):
        webapi.client_pull({"run": "run_b"})


def test_a_finished_job_does_not_block_the_next_pull(home):
    """Only *active* jobs are exclusive; a completed one must not wedge the UI."""
    finished = webapi.PullJob("run_a", "all", "criteria", False)
    finished.state = "done"
    webapi._JOB = finished
    config.Config(server_url="http://127.0.0.1:1", token="t").save()

    # Starting a pull only spawns the worker, so it returns even for an unreachable
    # server; the failure surfaces on the job, which is what the UI polls.
    result = webapi.client_pull({"run": "run_b"})
    assert result["ok"] is True
    assert result["job"]["run"] == "run_b"

    for _ in range(100):
        if not webapi.client_pull_status({})["job"]["active"]:
            break
        time.sleep(0.05)
    job = webapi.client_pull_status({})["job"]
    assert job["run"] == "run_b"
    assert job["state"] == "failed"
    assert "127.0.0.1:1" in job["error"]


def test_pull_status_is_empty_before_any_pull(home):
    assert webapi.client_pull_status({})["job"] is None


def test_cancel_without_a_job_is_not_an_error(home):
    result = webapi.client_pull_cancel({})
    assert result["ok"] is False
    assert "No pull" in result["message"]


def test_cancel_sets_the_flag_on_an_active_job(home):
    job = webapi.PullJob("run_a", "all", "criteria", False)
    job.state = "downloading"
    webapi._JOB = job
    assert webapi.client_pull_cancel({})["ok"] is True
    assert job.cancelled is True


def test_job_snapshot_active_states(home):
    job = webapi.PullJob("r", "all", "criteria", False)
    for state, active in [
        ("starting", True), ("planning", True), ("downloading", True),
        ("done", False), ("failed", False), ("cancelled", False),
    ]:
        job.state = state
        assert job.snapshot()["active"] is active, state


# ------------------------------------------------------------------------- delete


def test_delete_requires_a_run(home):
    with pytest.raises(ValueError, match="run name is required"):
        webapi.client_delete_run({})


def test_delete_removes_a_local_run(home):
    target = config.run_dir("run_a")
    target.mkdir(parents=True)
    (target / "current.parquet").write_bytes(b"PAR1")
    result = webapi.client_delete_run({"run": "run_a"})
    assert result["ok"] is True
    assert not target.exists()
    assert result["local_runs"] == []


def test_delete_reports_an_unknown_run(home):
    with pytest.raises(ValueError, match="No local run"):
        webapi.client_delete_run({"run": "nope"})


def test_cannot_delete_a_run_being_downloaded(home):
    target = config.run_dir("run_a")
    target.mkdir(parents=True)
    job = webapi.PullJob("run_a", "all", "criteria", False)
    job.state = "downloading"
    webapi._JOB = job
    with pytest.raises(ValueError, match="Cancel the pull first"):
        webapi.client_delete_run({"run": "run_a"})
    assert target.exists()


# ------------------------------------------------------------------------ handler


def test_handler_drops_the_export_routes(home):
    """A process listening on a laptop must not serve files to anyone."""
    config.apply_server_env()
    handler = webapi.build_handler()
    assert handler.auth_routes == {}
    assert handler.stream_routes == {}
    # ...while keeping every viewer route plus the client-only ones.
    assert "/api/frames" in handler.routes
    assert "/api/client/state" in handler.routes
    assert "/api/export_file" not in handler.routes


def test_client_routes_are_all_registered(home):
    config.apply_server_env()
    handler = webapi.build_handler()
    for route in webapi.CLIENT_ROUTES:
        assert route in handler.routes, route


# ----------------------------------------------------------------- 3D point clouds


@pytest.fixture()
def t4_home(home, monkeypatch):
    monkeypatch.delenv("EVALDASH_T4_BASE_URL", raising=False)
    webapi._T4_JOB = None
    yield home
    webapi._T4_JOB = None


def test_frames_spec_matches_the_cli_grammar(t4_home):
    assert webapi.parse_frames_spec("") is None
    assert webapi.parse_frames_spec("7") == range(7, 8)
    assert webapi.parse_frames_spec("0-49") == range(0, 50)
    with pytest.raises(ValueError, match="ascending"):
        webapi.parse_frames_spec("9-3")
    with pytest.raises(ValueError, match="look like"):
        webapi.parse_frames_spec("all of them")


def test_scenario_listing_survives_every_shape_the_service_has_used(t4_home):
    wrapped = {"scenarios": [{"name": "a", "nbr_samples": 42}, "b", {"bogus": 1}]}
    assert webapi.normalize_scenarios(wrapped) == [
        {"name": "a", "frames": 42},
        {"name": "b", "frames": None},
    ]
    assert webapi.normalize_scenarios(["x"]) == [{"name": "x", "frames": None}]
    assert webapi.normalize_scenarios(None) == []


def test_t4_state_reports_config_without_probing_by_default(t4_home):
    state = webapi.client_t4_state({})
    assert state["server"] is None  # no network touched
    assert state["scenes"] == []
    assert state["config"]["cache_root"]


def test_t4_config_saves_and_probes(t4_home):
    from tests.fake_t4_server import FakeT4Server

    with FakeT4Server() as server:
        res = webapi.client_t4_config({"t4_base_url": server.base_url + "/"})
        assert res["config"]["t4_base_url"] == server.base_url  # trailing slash dropped
        assert res["server"]["reachable"] is True
    assert config.Config.load().t4_base_url == server.base_url


def test_t4_scenarios_and_estimate_use_the_stored_url(t4_home):
    from tests.fake_t4_server import FakeT4Server

    with FakeT4Server(frames=5, points=100) as server:
        config.Config(t4_base_url=server.base_url).save()
        listing = webapi.client_t4_scenarios({"dataset_id": "T4DS0001"})
        assert listing["scenarios"] == [{"name": "scene-alpha", "frames": 5}]
        estimate = webapi.client_t4_estimate({"dataset_id": "T4DS0001", "scenario": "scene-alpha"})
        assert estimate["frames"] == 5
        assert estimate["frame_bytes"] > 0


def test_t4_fetch_runs_to_done_and_lists_the_scene(t4_home):
    from tests.fake_t4_server import FakeT4Server

    with FakeT4Server(frames=3, points=50) as server:
        config.Config(t4_base_url=server.base_url).save()
        res = webapi.client_t4_fetch({
            "dataset_id": "T4DS0001", "scenario": "scene-alpha",
            "with_camera": False, "with_lanelet": False,
        })
        assert res["job"]["active"] is True
        for _ in range(200):
            job = webapi.client_t4_fetch_status({})["job"]
            if not job["active"]:
                break
            time.sleep(0.05)
        assert job["state"] == "done", job
        assert "3 frame(s)" in job["message"]

    scenes = webapi.client_t4_state({})["scenes"]
    assert [(s["dataset_id"], s["frames_cached"]) for s in scenes] == [("T4DS0001", 3)]
    assert scenes[0]["complete"] is True


def test_a_second_fetch_is_refused_while_one_runs(t4_home):
    job = webapi.T4FetchJob("d", "s", "")
    job.state = "fetching"
    webapi._T4_JOB = job
    with pytest.raises(ValueError, match="already running"):
        webapi.client_t4_fetch({"dataset_id": "d2", "scenario": "s2"})


def test_deleting_the_scene_being_fetched_is_refused(t4_home):
    job = webapi.T4FetchJob("d", "s", "")
    job.state = "fetching"
    webapi._T4_JOB = job
    with pytest.raises(ValueError, match="Cancel the fetch first"):
        webapi.client_t4_delete({"dataset_id": "d", "scenario": "s"})
