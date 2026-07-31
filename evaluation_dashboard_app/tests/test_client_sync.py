"""Plan building decides what moves over the network, so its edge cases matter."""

import hashlib
import json
from pathlib import Path

import pytest

from client import config, sync


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
    return tmp_path / "home"


def _manifest(files, run="run_one", tier="criteria"):
    return {
        "run": run,
        "tier": tier,
        "roles": ["devops"],
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(f["size"] for f in files),
    }


def _entry(rel, data: bytes, *, sha=True, mtime_ns=111):
    out = {"rel_path": rel, "size": len(data), "mtime_ns": mtime_ns}
    if sha:
        out["sha256"] = hashlib.sha256(data).hexdigest()
    return out


def _write_local(run: str, rel: str, data: bytes) -> Path:
    path = config.run_dir(run) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# -------------------------------------------------------------------- new files


def test_everything_is_new_on_a_fresh_workspace(home):
    manifest = _manifest([_entry("a.parquet", b"abc"), _entry("b.yaml", b"de")])
    plan = sync.build_plan(manifest)
    assert {i.rel_path for i in plan.download} == {"a.parquet", "b.yaml"}
    assert all(i.reason == "new" for i in plan.download)
    assert plan.download_bytes == 5
    assert plan.keep == []


def test_manifest_without_a_run_is_rejected(home):
    with pytest.raises(ValueError):
        sync.build_plan(_manifest([], run=""))


# ------------------------------------------------------------------- keep/skip


def test_recorded_file_with_matching_size_and_sha_is_kept(home):
    data = b"hello world"
    _write_local("run_one", "a.parquet", data)
    entry = _entry("a.parquet", data)
    config.write_run_state(
        "run_one", {"files": {"a.parquet": {"size": len(data), "sha256": entry["sha256"]}}}
    )
    plan = sync.build_plan(_manifest([entry]))
    assert plan.download == []
    assert plan.keep == ["a.parquet"]


def test_present_but_unrecorded_file_is_kept_without_verify(home):
    """Trusting size keeps re-pulls cheap; --verify is the way to be strict."""
    data = b"hello world"
    _write_local("run_one", "a.parquet", data)
    plan = sync.build_plan(_manifest([_entry("a.parquet", data)]))
    assert plan.download == [] and plan.keep == ["a.parquet"]


def test_verify_keeps_an_unrecorded_file_whose_bytes_are_correct(home):
    """No state record, but --verify hashes the file and finds it sound, so keep it."""
    data = b"hello world"
    _write_local("run_one", "a.parquet", data)
    plan = sync.build_plan(_manifest([_entry("a.parquet", data)]), verify=True)
    assert plan.download == []
    assert plan.keep == ["a.parquet"]


def test_verify_detects_same_size_corruption(home):
    """Byte-flip corruption is invisible to a size check; this is why --verify exists."""
    good = b"hello world"
    corrupt = b"hellO world"
    _write_local("run_one", "a.parquet", corrupt)
    entry = _entry("a.parquet", good)
    config.write_run_state(
        "run_one", {"files": {"a.parquet": {"size": len(good), "sha256": entry["sha256"]}}}
    )
    assert sync.build_plan(_manifest([entry])).download == []  # size matches, record trusted
    plan = sync.build_plan(_manifest([entry]), verify=True)
    assert [i.reason for i in plan.download] == ["changed"]


# --------------------------------------------------------------- change reasons


def test_short_local_file_is_incomplete(home):
    data = b"0123456789"
    _write_local("run_one", "a.parquet", data[:4])
    plan = sync.build_plan(_manifest([_entry("a.parquet", data)]))
    assert [i.reason for i in plan.download] == ["incomplete"]


def test_longer_local_file_is_changed(home):
    data = b"0123"
    _write_local("run_one", "a.parquet", data + b"extra")
    plan = sync.build_plan(_manifest([_entry("a.parquet", data)]))
    assert [i.reason for i in plan.download] == ["changed"]


def test_recorded_sha_mismatch_is_changed(home):
    data = b"new content"
    _write_local("run_one", "a.parquet", b"11111111111")  # same length, different bytes
    config.write_run_state(
        "run_one", {"files": {"a.parquet": {"size": len(data), "sha256": "0" * 64}}}
    )
    plan = sync.build_plan(_manifest([_entry("a.parquet", data)]))
    assert [i.reason for i in plan.download] == ["changed"]


def test_mtime_change_is_detected_when_the_server_sends_no_sha(home):
    data = b"abcd"
    _write_local("run_one", "a.parquet", data)
    config.write_run_state("run_one", {"files": {"a.parquet": {"size": 4, "mtime_ns": 111}}})
    same = _manifest([_entry("a.parquet", data, sha=False, mtime_ns=111)])
    assert sync.build_plan(same).download == []
    moved = _manifest([_entry("a.parquet", data, sha=False, mtime_ns=222)])
    assert [i.reason for i in sync.build_plan(moved).download] == ["changed"]


# -------------------------------------------------------------- resume accounting


def test_partial_file_discounts_already_stored_bytes(home):
    """The reported total must be what actually moves, not the whole file again."""
    data = b"x" * 1000
    part = config.run_dir("run_one") / ("a.parquet" + sync.PART_SUFFIX)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(data[:400])
    plan = sync.build_plan(_manifest([_entry("a.parquet", data)]))
    item = plan.download[0]
    assert item.reason == "new"
    assert item.size == 1000
    assert item.remaining == 600
    assert plan.download_bytes == 600
    assert plan.total_bytes == 1000


def test_oversized_part_is_not_credited(home):
    data = b"x" * 100
    part = config.run_dir("run_one") / ("a.parquet" + sync.PART_SUFFIX)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"y" * 500)
    plan = sync.build_plan(_manifest([_entry("a.parquet", data)]))
    assert plan.download[0].remaining == 100


def test_changed_files_do_not_reuse_a_part(home):
    """A stale part belongs to different content, so it cannot be appended to."""
    data = b"x" * 1000
    _write_local("run_one", "a.parquet", b"z" * 1000)
    part = config.run_dir("run_one") / ("a.parquet" + sync.PART_SUFFIX)
    part.write_bytes(b"q" * 400)
    config.write_run_state("run_one", {"files": {"a.parquet": {"size": 1000, "sha256": "0" * 64}}})
    plan = sync.build_plan(_manifest([_entry("a.parquet", data)]))
    assert plan.download[0].reason == "changed"
    assert plan.download[0].remaining == 1000


# ------------------------------------------------------------------- obsolete


def test_files_dropped_by_the_server_are_reported_obsolete(home):
    config.write_run_state(
        "run_one",
        {"files": {"a.parquet": {"size": 3}, "gone.yaml": {"size": 9}}},
    )
    _write_local("run_one", "a.parquet", b"abc")
    plan = sync.build_plan(_manifest([_entry("a.parquet", b"abc")]))
    assert plan.obsolete == ["gone.yaml"]


# --------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    "count,expected",
    [(0, "0 B"), (512, "512 B"), (1024, "1.0 KB"), (1536, "1.5 KB"), (1048576, "1.0 MB"),
     (int(1.5 * 1024**3), "1.5 GB")],
)
def test_human_bytes(count, expected):
    assert sync.human_bytes(count) == expected


def test_sha256_file_matches_hashlib(home, tmp_path):
    payload = b"a" * (sync.CHUNK_BYTES + 17)  # forces multiple read chunks
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert sync.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_local_run_summary_counts_partials(home):
    _write_local("run_one", "a.parquet", b"abc")
    _write_local("run_one", "b.parquet" + sync.PART_SUFFIX, b"de")
    config.write_run_state("run_one", {"tier": "minimal", "roles": ["devops"], "files": {"a.parquet": {}}})
    (summary,) = sync.local_run_summary()
    assert summary["name"] == "run_one"
    assert summary["tier"] == "minimal"
    assert summary["incomplete"] == 1


def test_remove_run(home):
    _write_local("run_one", "a.parquet", b"abc")
    assert config.run_dir("run_one").is_dir()
    ok, _ = sync.remove_run("run_one")
    assert ok and not config.run_dir("run_one").exists()
    ok, message = sync.remove_run("run_one")
    assert not ok and "No local run" in message


# ---------------------------------------------------------------------- config


def test_config_round_trip_and_permissions(home):
    cfg = config.Config(server_url="https://example.test", token="s3cret")
    path = cfg.save()
    assert oct(path.stat().st_mode)[-3:] == "600"  # token is a secret
    loaded = config.Config.load()
    assert loaded.server_url == "https://example.test"
    assert loaded.resolved_token() == "s3cret"


def test_config_survives_a_corrupt_file(home):
    config.config_path().write_text("{not json")
    assert config.Config.load().server_url == ""


def test_missing_server_is_a_clear_error(home):
    with pytest.raises(RuntimeError, match="login"):
        config.Config().require_server()


def test_env_supplies_server_and_token(home, monkeypatch):
    monkeypatch.setenv("EVALDASH_SERVER", "https://env.test")
    monkeypatch.setenv("EVALDASH_TOKEN", "envtoken")
    cfg = config.Config()
    assert cfg.require_server() == "https://env.test"
    assert cfg.resolved_token() == "envtoken"


def test_apply_server_env_points_the_api_at_the_workspace(home, monkeypatch):
    monkeypatch.setenv("EVAL_EXPORT_TOKEN", "leftover")
    config.apply_server_env()
    import os

    assert os.environ["EVAL_DASHBOARD_DATA_ROOT"] == str(config.workspace_dir())
    assert os.environ["EVAL_BBOX_CACHE_DIR"] == str(config.cache_dir())
    # The client must not expose export routes of its own.
    assert "EVAL_EXPORT_TOKEN" not in os.environ


def test_build_default_server_is_used_when_nothing_is_configured(home, monkeypatch):
    """A build made with `build_app.sh --server URL` must connect with no setup at all.

    This is what makes the packaged app zero-configuration, so it needs a test of its
    own rather than relying on whether a build happened to leave _defaults.py behind.
    """
    monkeypatch.setattr(config, "DEFAULT_SERVER", "https://baked.example")
    cfg = config.Config()
    assert cfg.effective_server() == "https://baked.example"
    assert cfg.require_server() == "https://baked.example"


def test_server_precedence_is_env_then_stored_then_default(home, monkeypatch):
    """An environment override must win, or it is useless: connect() persists the
    resolved default on first use, so almost every install has a stored URL."""
    monkeypatch.setattr(config, "DEFAULT_SERVER", "https://baked.example")

    assert config.Config().server_source() == "default"
    assert config.Config().effective_server() == "https://baked.example"

    stored = config.Config(server_url="https://stored.example")
    assert stored.server_source() == "stored"
    assert stored.effective_server() == "https://stored.example"

    monkeypatch.setenv("EVALDASH_SERVER", "https://env.example")
    assert stored.server_source() == "env"
    assert stored.effective_server() == "https://env.example"


def test_no_server_anywhere_is_reported_as_none(home, monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_SERVER", "")
    cfg = config.Config()
    assert cfg.server_source() == "none"
    assert cfg.effective_server() == ""
    with pytest.raises(RuntimeError, match="login"):
        cfg.require_server()


def test_reset_falls_back_to_the_build_default(home, monkeypatch):
    """A stored URL shadows a newer baked-in one, so rebuilding with a different
    --server would keep talking to the old host without a way to clear it."""
    monkeypatch.setattr(config, "DEFAULT_SERVER", "https://baked.example")
    cfg = config.Config(server_url="https://old.example", token="secret")
    cfg.save()

    assert config.Config.load().effective_server() == "https://old.example"
    fallback = config.Config.load().reset_server()
    assert fallback == "https://baked.example"

    after = config.Config.load()
    assert after.server_url == ""
    assert after.resolved_token() == ""
    assert after.server_source() == "default"


def test_reset_with_no_default_leaves_nothing_configured(home, monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_SERVER", "")
    cfg = config.Config(server_url="https://old.example")
    cfg.save()
    assert config.Config.load().reset_server() == ""
    assert config.Config.load().server_source() == "none"
