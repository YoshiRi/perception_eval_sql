"""
Tests for remembering the last run selection: the browser cookie (per browser), the server-side
per-user store, and the hydration chain that consumes them.
"""

import json
import urllib.parse
from pathlib import Path

import pytest

from lib import overview_url_hydrate, run_selection_cookie, run_selection_store


# ---------------------------------------------------------------- server-side store


@pytest.fixture
def store_file(tmp_path, monkeypatch):
    path = tmp_path / "run_selection_state.json"
    monkeypatch.setenv("EVAL_DASHBOARD_RUN_SELECTION_STORE", str(path))
    monkeypatch.setattr(run_selection_store, "_user_key", lambda: "tester@example.com")
    return path


def test_save_and_load_roundtrip_compare_mode(store_file):
    assert run_selection_store.load_run_selection() is None

    run_selection_store.save_run_selection("Compare Mode", "run_a_name", ["run_b_name", "run_c_name"])

    saved = run_selection_store.load_run_selection()
    assert saved["mode"] == "compare"
    assert saved["run_a"] == "run_a_name"
    assert saved["compare_runs"] == ["run_b_name", "run_c_name"]
    assert saved["updated_at"]


def test_single_mode_overwrites_compare_selection(store_file):
    run_selection_store.save_run_selection("Compare Mode", "run_a_name", ["run_b_name"])
    run_selection_store.save_run_selection("Single Mode", "run_x_name")

    saved = run_selection_store.load_run_selection()
    assert saved["mode"] == "single"
    assert saved["run_a"] == "run_x_name"
    assert saved["compare_runs"] == []


def test_selections_are_kept_per_user(store_file, monkeypatch):
    run_selection_store.save_run_selection("Single Mode", "alice_run")
    monkeypatch.setattr(run_selection_store, "_user_key", lambda: "bob@example.com")
    assert run_selection_store.load_run_selection() is None

    run_selection_store.save_run_selection("Single Mode", "bob_run")
    assert run_selection_store.load_run_selection()["run_a"] == "bob_run"

    monkeypatch.setattr(run_selection_store, "_user_key", lambda: "tester@example.com")
    assert run_selection_store.load_run_selection()["run_a"] == "alice_run"


def test_empty_run_name_is_not_saved(store_file):
    run_selection_store.save_run_selection("Single Mode", "   ")
    assert run_selection_store.load_run_selection() is None
    assert not store_file.exists()


def test_corrupt_store_is_ignored(store_file):
    store_file.write_text("{not json", encoding="utf-8")
    assert run_selection_store.load_run_selection() is None
    run_selection_store.save_run_selection("Single Mode", "run_x_name")
    assert json.loads(store_file.read_text(encoding="utf-8"))["users"]


def test_clear_run_selection(store_file):
    run_selection_store.save_run_selection("Single Mode", "run_x_name")
    run_selection_store.clear_run_selection()
    assert run_selection_store.load_run_selection() is None


# ---------------------------------------------------------------- browser cookie


class _FakeContext:
    def __init__(self, cookies):
        self.cookies = cookies


class _FakeStreamlit:
    """Minimal stand-in for the streamlit module surface these helpers touch."""

    def __init__(self, query_params=None, cookies=None):
        self.session_state = {}
        self.query_params = dict(query_params or {})
        self.context = _FakeContext(dict(cookies or {}))


def test_cookie_value_roundtrip_survives_awkward_run_names():
    selection = run_selection_cookie.normalize_selection(
        "Compare Mode",
        "run a; with, punctuation",
        ["candidate=one", "日本語ラン"],
    )
    encoded = run_selection_cookie.encode_selection_cookie_value(selection)
    # A cookie value must not contain characters that terminate it.
    assert not set(encoded) & set(';, "\\')

    assert run_selection_cookie.decode_selection_cookie_value(encoded) == {
        "mode": "compare",
        "run_a": "run a; with, punctuation",
        "compare_runs": ["candidate=one", "日本語ラン"],
    }


@pytest.mark.parametrize(
    "raw",
    [None, "", "not-json", urllib.parse.quote("[1,2,3]"), urllib.parse.quote('{"mode":"single"}')],
)
def test_decode_rejects_unusable_cookie_values(raw):
    assert run_selection_cookie.decode_selection_cookie_value(raw) is None


def test_read_selection_cookie(monkeypatch):
    selection = run_selection_cookie.normalize_selection("Single Mode", "run_one")
    value = run_selection_cookie.encode_selection_cookie_value(selection)
    monkeypatch.setattr(
        run_selection_cookie,
        "st",
        _FakeStreamlit(cookies={run_selection_cookie.COOKIE_NAME: value}),
    )

    assert run_selection_cookie.read_selection_cookie() == {
        "mode": "single",
        "run_a": "run_one",
        "compare_runs": [],
    }


def test_read_selection_cookie_without_cookie(monkeypatch):
    monkeypatch.setattr(run_selection_cookie, "st", _FakeStreamlit())
    assert run_selection_cookie.read_selection_cookie() is None


def test_persist_writes_cookie_script_once_per_value(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(run_selection_cookie, "st", fake_st)
    written = []
    monkeypatch.setattr(
        run_selection_cookie,
        "_render_cookie_script",
        lambda value, max_age: written.append((value, max_age)),
    )

    run_selection_cookie.persist_selection_cookie("Single Mode", "run_one")
    run_selection_cookie.persist_selection_cookie("Single Mode", "run_one")
    assert len(written) == 1  # reruns must not re-inject the same script

    run_selection_cookie.persist_selection_cookie("Compare Mode", "run_one", ["run_two"])
    assert len(written) == 2

    value, max_age = written[1]
    assert max_age == run_selection_cookie.COOKIE_MAX_AGE_SECONDS
    assert run_selection_cookie.decode_selection_cookie_value(value)["compare_runs"] == ["run_two"]


def test_persist_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.setenv(run_selection_cookie.DISABLE_ENV, "1")
    monkeypatch.setattr(run_selection_cookie, "st", _FakeStreamlit())
    written = []
    monkeypatch.setattr(
        run_selection_cookie,
        "_render_cookie_script",
        lambda value, max_age: written.append(value),
    )

    run_selection_cookie.persist_selection_cookie("Single Mode", "run_one")
    assert written == []


def test_cookie_script_is_self_contained_and_targets_the_parent_document():
    html = []
    original = run_selection_cookie.components.html
    try:
        run_selection_cookie.components.html = lambda body, **kwargs: html.append((body, kwargs))
        run_selection_cookie._render_cookie_script("abc", 123)
    finally:
        run_selection_cookie.components.html = original

    body, kwargs = html[0]
    assert kwargs["height"] == 0
    assert "window.parent.document" in body
    assert 'path=/; max-age=" + maxAge + "; SameSite=Lax' in body
    assert "var maxAge = 123;" in body
    assert f'var name = "{run_selection_cookie.COOKIE_NAME}";' in body


# ---------------------------------------------------------------- hydration chain


@pytest.fixture
def fake_runs(tmp_path, monkeypatch):
    run_dirs = [tmp_path / "run_one", tmp_path / "run_two"]
    for d in run_dirs:
        d.mkdir()
    monkeypatch.setattr(overview_url_hydrate, "get_data_root", lambda: tmp_path)
    monkeypatch.setattr(overview_url_hydrate, "list_run_directories", lambda: run_dirs)
    monkeypatch.setattr(overview_url_hydrate, "get_run_display_name", lambda p: Path(p).name)
    monkeypatch.setattr(overview_url_hydrate, "get_run_storage_name", lambda p: Path(p).name)
    monkeypatch.setattr(
        overview_url_hydrate,
        "load_run",
        lambda d: {"path": d, "summary": None, "score": None},
    )
    return run_dirs


@pytest.fixture
def hydrate_env(monkeypatch):
    """Wire overview_url_hydrate to fakes; returns (fake_st, recorder) with `recorder` factory."""

    def _setup(query_params=None, cookie_selection=None, stored_selection=None):
        fake_st = _FakeStreamlit(query_params=query_params)
        monkeypatch.setattr(overview_url_hydrate, "st", fake_st)
        monkeypatch.setattr(overview_url_hydrate, "read_selection_cookie", lambda: cookie_selection)
        monkeypatch.setattr(overview_url_hydrate, "load_run_selection", lambda: stored_selection)
        persisted = {"cookie": [], "store": []}
        monkeypatch.setattr(
            overview_url_hydrate,
            "persist_selection_cookie",
            lambda mode, run_a, compare=None: persisted["cookie"].append((mode, run_a, list(compare or []))),
        )
        monkeypatch.setattr(
            overview_url_hydrate,
            "save_run_selection",
            lambda mode, run_a, compare=None: persisted["store"].append((mode, run_a, list(compare or []))),
        )
        return fake_st, persisted

    return _setup


def _selection(mode, run_a, compare=()):
    return {"mode": mode, "run_a": run_a, "compare_runs": list(compare)}


def test_hydrate_from_cookie_single_mode(fake_runs, hydrate_env):
    fake_st, persisted = hydrate_env(cookie_selection=_selection("single", "run_one"))

    assert overview_url_hydrate.try_hydrate_session_from_overview_query_params() is True
    assert fake_st.session_state["runA"]["path"].name == "run_one"
    assert fake_st.session_state["mode"] == "Single Mode"
    # Mirrored into the URL so a refresh or replica hop keeps the same selection.
    assert fake_st.query_params == {"mode": "single", "run_a": "run_one"}
    # The cookie is already correct, so no need to rewrite it.
    assert persisted["cookie"] == []


def test_hydrate_from_cookie_compare_mode(fake_runs, hydrate_env):
    fake_st, _ = hydrate_env(cookie_selection=_selection("compare", "run_one", ["run_two"]))

    assert overview_url_hydrate.try_hydrate_session_from_overview_query_params() is True
    assert fake_st.session_state["mode"] == "Compare Mode"
    assert [r["path"].name for r in fake_st.session_state["all_runs"]] == ["run_one", "run_two"]
    assert fake_st.session_state["run_labels"] == ["A", "B"]
    assert fake_st.session_state["runB"]["path"].name == "run_two"
    assert fake_st.query_params == {"mode": "compare", "run_a": "run_one", "run_b": "run_two"}


def test_cookie_wins_over_server_side_store(fake_runs, hydrate_env):
    fake_st, _ = hydrate_env(
        cookie_selection=_selection("single", "run_two"),
        stored_selection=_selection("single", "run_one"),
    )

    assert overview_url_hydrate.try_hydrate_session_from_overview_query_params() is True
    assert fake_st.session_state["runA"]["path"].name == "run_two"


def test_store_is_used_when_browser_has_no_cookie_and_teaches_the_browser(fake_runs, hydrate_env):
    fake_st, persisted = hydrate_env(stored_selection=_selection("single", "run_one"))

    assert overview_url_hydrate.try_hydrate_session_from_overview_query_params() is True
    assert fake_st.session_state["runA"]["path"].name == "run_one"
    assert persisted["cookie"] == [("single", "run_one", [])]


def test_store_is_used_when_cookie_points_at_a_deleted_run(fake_runs, hydrate_env):
    fake_st, _ = hydrate_env(
        cookie_selection=_selection("single", "gone_run"),
        stored_selection=_selection("single", "run_two"),
    )

    assert overview_url_hydrate.try_hydrate_session_from_overview_query_params() is True
    assert fake_st.session_state["runA"]["path"].name == "run_two"


def test_nothing_remembered_leaves_session_empty(fake_runs, hydrate_env):
    fake_st, _ = hydrate_env()

    assert overview_url_hydrate.try_hydrate_session_from_overview_query_params() is False
    assert "runA" not in fake_st.session_state
    assert fake_st.query_params == {}


def test_remembered_compare_degrades_to_single_when_candidates_gone(fake_runs, hydrate_env):
    fake_st, _ = hydrate_env(cookie_selection=_selection("compare", "run_one", ["deleted_run"]))

    assert overview_url_hydrate.try_hydrate_session_from_overview_query_params() is True
    assert fake_st.session_state["mode"] == "Single Mode"
    assert fake_st.query_params == {"mode": "single", "run_a": "run_one"}


def test_url_params_win_over_remembered_selection_and_are_remembered(fake_runs, hydrate_env):
    fake_st, persisted = hydrate_env(
        query_params={"mode": "single", "run_a": "run_two"},
        cookie_selection=_selection("single", "run_one"),
    )

    assert overview_url_hydrate.try_hydrate_session_from_overview_query_params() is True
    assert fake_st.session_state["runA"]["path"].name == "run_two"
    # Opening a shared link is "what I looked at last" too.
    assert persisted["cookie"] == [("single", "run_two", [])]
    assert persisted["store"] == [("single", "run_two", [])]


def test_existing_session_state_is_not_reloaded_from_memory(fake_runs, monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state["runA"] = {"path": fake_runs[1]}
    monkeypatch.setattr(overview_url_hydrate, "st", fake_st)

    def _fail(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("memory should not be consulted when runA is already loaded")

    monkeypatch.setattr(overview_url_hydrate, "read_selection_cookie", _fail)
    monkeypatch.setattr(overview_url_hydrate, "load_run_selection", _fail)

    assert overview_url_hydrate.try_hydrate_session_from_overview_query_params() is True
    assert fake_st.session_state["runA"]["path"].name == "run_two"
