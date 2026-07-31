"""Anchors must keep the server's behaviour and let the client redirect it."""

import os
import sys
from pathlib import Path

from backend import app_paths

REPO_ROOT = Path(__file__).resolve().parent.parent


def _clear(monkeypatch):
    for name in (
        "EVAL_APP_ROOT",
        "EVAL_DASHBOARD_DATA_ROOT",
        "EVAL_BBOX_CACHE_DIR",
        "EVAL_LIB_PATHS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_app_root_is_repo_root_by_default(monkeypatch):
    _clear(monkeypatch)
    assert app_paths.app_root() == REPO_ROOT


def test_defaults_match_the_old_cwd_behaviour(monkeypatch, tmp_path):
    """The server used Path.cwd(); from the repo root the anchors must agree with it.

    This is the regression guard for the refactor: Docker runs from /app and dev runs
    from the repo root, and in both cases cwd == app_root.
    """
    _clear(monkeypatch)
    monkeypatch.chdir(REPO_ROOT)
    assert app_paths.data_root() == (Path.cwd() / "data").resolve()
    assert app_paths.cache_root() == (Path.cwd() / ".cache").resolve()
    assert app_paths.static_dirs()[0] == (Path.cwd() / "static").resolve()

    # ...and must NOT follow the process into an unrelated directory.
    monkeypatch.chdir(tmp_path)
    assert app_paths.data_root() == (REPO_ROOT / "data").resolve()
    assert app_paths.cache_root() == (REPO_ROOT / ".cache").resolve()


def test_env_overrides_win(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("EVAL_APP_ROOT", str(tmp_path / "app"))
    monkeypatch.setenv("EVAL_DASHBOARD_DATA_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("EVAL_BBOX_CACHE_DIR", str(tmp_path / "cache"))
    assert app_paths.app_root() == (tmp_path / "app").resolve()
    assert app_paths.data_root() == (tmp_path / "runs").resolve()
    assert app_paths.cache_root() == (tmp_path / "cache").resolve()


def test_relative_data_root_anchors_to_app_root(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("EVAL_APP_ROOT", str(tmp_path))
    monkeypatch.setenv("EVAL_DASHBOARD_DATA_ROOT", "data")
    monkeypatch.chdir(Path("/"))
    assert app_paths.data_root() == (tmp_path / "data").resolve()


def test_frozen_app_root_uses_bundle_dir(monkeypatch, tmp_path):
    """PyInstaller extracts to _MEIPASS; static/ ships there, so that is the root."""
    _clear(monkeypatch)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    try:
        assert app_paths.app_root() == (tmp_path / "bundle").resolve()
    finally:
        monkeypatch.delattr(sys, "frozen", raising=False)


def test_static_dirs_are_ordered_and_deduped(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.chdir(REPO_ROOT)
    dirs = app_paths.static_dirs()
    assert dirs[0] == (REPO_ROOT / "static").resolve()
    assert len(dirs) == len(set(map(str, dirs)))
    assert Path("/app/static") in dirs  # container fallback preserved


def test_find_static_file(monkeypatch):
    _clear(monkeypatch)
    assert app_paths.find_static_file("bbox_theme.js") is not None
    assert app_paths.find_static_file("definitely_not_here.js") is None


def test_eval_lib_paths_override(monkeypatch, tmp_path):
    _clear(monkeypatch)
    first = tmp_path / "a"
    second = tmp_path / "b"
    monkeypatch.setenv("EVAL_LIB_PATHS", os.pathsep.join([str(first), str(second)]))
    assert app_paths.eval_lib_paths() == [first, second]


def test_eval_lib_paths_default_is_non_empty(monkeypatch):
    _clear(monkeypatch)
    assert len(app_paths.eval_lib_paths()) == 2
