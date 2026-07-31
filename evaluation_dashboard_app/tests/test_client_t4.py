"""The offline 3D cache must replay t4-server byte-for-byte with the network gone."""

import json
from pathlib import Path

import pytest

from client import config, t4
from tests.fake_t4_server import FakeT4Server, pack_frame

DATASET = "T4DS0001"
SCENARIO = "scene-alpha"


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("EVALDASH_HOME", str(tmp_path / "home"))
    config.ensure_dirs()
    return tmp_path / "home"


@pytest.fixture()
def server():
    with FakeT4Server(frames=6, points=400) as srv:
        yield srv


@pytest.fixture()
def fetched(home, server):
    """A fully cached scene, plus the server it came from."""
    stats = t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url)
    return stats, server


# ----------------------------------------------------------------- wire format


def test_frame_header_parses_the_real_layout():
    blob = pack_frame(3, point_count=100, box_count=2)
    info = t4.parse_frame_header(blob)
    assert info["frame_index"] == 3
    assert info["point_count"] == 100
    assert info["box_count"] == 2
    assert info["sample_token"] == "sample-token-0003"
    # magic(8) + fixed(26) + token, then 16 B per point and 96 B per box.
    assert len(blob) > 100 * 16 + 2 * 96


def test_frame_header_rejects_foreign_payloads():
    with pytest.raises(t4.T4Error):
        t4.parse_frame_header(b"not a frame at all, really not")


# ------------------------------------------------------------------- fetching


def test_fetch_stores_every_frame(fetched):
    stats, _ = fetched
    assert stats["frames_total"] == 6
    assert stats["frames_fetched"] == 6
    assert stats["errors"] == []
    scene = t4.scene_dir(DATASET, SCENARIO)
    assert sorted(p.name for p in (scene / "frames").glob("*.bin")) == [
        "0.bin", "1.bin", "2.bin", "3.bin", "4.bin", "5.bin"
    ]
    assert (scene / "page.html").is_file()
    assert (scene / "meta.json").is_file()
    assert (scene / "manifest.json").is_file()


def test_fetch_stores_bytes_verbatim(fetched):
    """The mirror is only useful if the cached frame is the server's exact payload."""
    _, server = fetched
    cached = (t4.scene_dir(DATASET, SCENARIO) / "frames" / "2.bin").read_bytes()
    assert cached == pack_frame(2, point_count=400, box_count=2)


def test_fetch_captures_response_headers(fetched):
    headers = json.loads((t4.scene_dir(DATASET, SCENARIO) / "frames" / "1.hdr.json").read_text())
    assert headers["x-t4v-format"] == "T4V3D002"
    assert headers["x-t4v-point-fields"] == "x,y,z,intensity"


def test_fetch_is_incremental(home, server):
    t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url)
    again = t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url)
    assert again["frames_fetched"] == 0
    assert again["frames_skipped"] == 6


def test_force_refetches(home, server):
    t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url)
    forced = t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url, force=True)
    assert forced["frames_fetched"] == 6


def test_frame_subset(home, server):
    stats = t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url, frames=range(1, 4))
    assert stats["frames_requested"] == 3
    assert stats["frames_fetched"] == 3
    assert not (t4.scene_dir(DATASET, SCENARIO) / "frames" / "0.bin").exists()


def test_out_of_range_frames_are_dropped(home, server):
    stats = t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url, frames=range(4, 99))
    assert stats["frames_requested"] == 2  # only 4 and 5 exist


def test_optional_payloads_can_be_skipped(home, server):
    t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url,
                   with_lanelet=False, with_camera=False)
    scene = t4.scene_dir(DATASET, SCENARIO)
    assert not (scene / "lanelet").exists()
    assert not (scene / "overlay").exists()


def test_should_stop_halts_between_frames(home, server):
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    stats = t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url, should_stop=stop)
    assert stats["stopped_early"] is True
    assert stats["frames_fetched"] < 6


def test_progress_reports_eta(home, server):
    seen = []
    t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url, progress=seen.append)
    assert len(seen) == 6
    assert seen[-1]["index"] == 6 and seen[-1]["total"] == 6
    assert seen[0]["bytes"] > 0


def test_estimate_sizes_the_scene_from_one_frame(home, server):
    estimate = t4.estimate_scene_bytes(DATASET, SCENARIO, base_url=server.base_url)
    assert estimate["frames"] == 6
    assert estimate["points_per_frame"] == 400
    assert estimate["estimated_bytes"] == estimate["frame_bytes"] * 6


def test_missing_base_url_is_a_clear_error(home):
    with pytest.raises(t4.T4Error, match="t4-base-url"):
        t4.T4Client("")


def test_unreachable_server_is_reported(home):
    with pytest.raises(t4.T4Error, match="Cannot reach"):
        t4.fetch_scene(DATASET, SCENARIO, base_url="http://127.0.0.1:1")


# --------------------------------------------------------------------- serving


def _serve(path, **params):
    query = {k: [str(v)] for k, v in params.items()}
    return t4.serve_request(path, query)


def test_serves_frames_offline(fetched):
    """The whole point: identical bytes with no server involved."""
    _, server = fetched
    server.__exit__()  # network is now gone
    body, ctype, headers = _serve("/viewer/three/frame.bin",
                                  t4dataset_id=DATASET, scenario_name=SCENARIO, frame_index=2)
    assert body == pack_frame(2, point_count=400, box_count=2)
    assert ctype == "application/octet-stream"
    assert headers["x-t4v-format"] == "T4V3D002"


def test_serves_the_page(fetched):
    body, ctype, _ = _serve("/viewer/three", t4dataset_id=DATASET, scenario_name=SCENARIO)
    assert ctype.startswith("text/html")
    text = body.decode()
    # Placeholders must already be substituted, and fetches must stay root-relative so
    # the page works unchanged from a different host and port.
    assert "__QS__" not in text and "__DATASET_ID__" not in text
    assert "/viewer/three/frame.bin?" in text
    assert "http://127.0.0.1" not in text


def test_serves_meta_lanelet_overlay_and_caminfo(fetched):
    for path in ("/viewer/three/meta", "/viewer/three/lanelet-lines",
                 "/viewer/three/camera-overlay", "/viewer/three/camera-info"):
        body, ctype, _ = _serve(path, t4dataset_id=DATASET, scenario_name=SCENARIO, frame_index=1)
        assert ctype == "application/json"
        assert json.loads(body)


def test_frames_window_is_computed_not_cached(fetched):
    body, _, _ = _serve("/viewer/three/frames/window",
                        t4dataset_id=DATASET, scenario_name=SCENARIO, center=2, radius=1)
    data = json.loads(body)
    assert data["total"] == 6
    assert [f["frame_index"] for f in data["frames"]] == [1, 2, 3]


def test_frames_window_clamps_at_the_edges(fetched):
    body, _, _ = _serve("/viewer/three/frames/window",
                        t4dataset_id=DATASET, scenario_name=SCENARIO, center=0, radius=3)
    assert [f["frame_index"] for f in json.loads(body)["frames"]] == [0, 1, 2, 3]


def test_scenario_is_optional_when_only_one_is_cached(fetched):
    body, _, _ = _serve("/viewer/three/frame.bin", t4dataset_id=DATASET, frame_index=0)
    assert body.startswith(t4.FRAME_MAGIC)


def test_debug_pings_are_absorbed(fetched):
    body, ctype, _ = _serve("/viewer/three/debug/message-received",
                            t4dataset_id=DATASET, scenario_name=SCENARIO)
    assert ctype == "application/json" and json.loads(body) == {}


# ------------------------------------------------------------------ cache miss


def test_uncached_dataset_explains_how_to_fetch(home):
    with pytest.raises(t4.CacheMiss) as excinfo:
        _serve("/viewer/three/frame.bin", t4dataset_id="NOPE", frame_index=0)
    assert "t4 fetch NOPE" in str(excinfo.value)


def test_uncached_frame_is_named(home, server):
    t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url, frames=range(0, 2))
    with pytest.raises(t4.CacheMiss, match="Frame 5 is not cached"):
        _serve("/viewer/three/frame.bin", t4dataset_id=DATASET, scenario_name=SCENARIO, frame_index=5)


def test_missing_dataset_id_is_rejected(home):
    with pytest.raises(t4.CacheMiss, match="t4dataset_id is required"):
        _serve("/viewer/three/meta")


def test_unmirrored_path_is_a_miss(fetched):
    with pytest.raises(t4.CacheMiss, match="not mirrored"):
        _serve("/viewer/three/session/abc", t4dataset_id=DATASET, scenario_name=SCENARIO)


# -------------------------------------------------------------- cache managing


def test_cached_scenes_lists_and_reports_completeness(fetched):
    (scene,) = t4.cached_scenes()
    assert scene["dataset_id"] == DATASET
    assert scene["scenario"] == SCENARIO
    assert scene["frames_cached"] == 6
    assert scene["complete"] is True
    assert scene["bytes"] > 0


def test_partial_scene_is_reported_incomplete(home, server):
    t4.fetch_scene(DATASET, SCENARIO, base_url=server.base_url, frames=range(0, 3))
    (scene,) = t4.cached_scenes()
    assert scene["frames_cached"] == 3
    assert scene["complete"] is False


def test_remove_scene(fetched):
    ok, _ = t4.remove_scene(DATASET, SCENARIO)
    assert ok and t4.cached_scenes() == []
    ok, message = t4.remove_scene(DATASET, SCENARIO)
    assert not ok and "No cached scene" in message


def test_scene_dir_sanitises_awkward_names(home):
    path = t4.scene_dir("../../etc", "a/b c")
    assert ".." not in path.parts
    assert path.is_relative_to(t4.t4_root())
