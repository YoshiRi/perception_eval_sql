"""Tests for lib/t4_visualizer_client.py.

Unit tests use mocks (no network). Optional integration tests call a live server when
``T4_VISUALIZER_BASE_URL`` points at a reachable instance (e.g. ``t4-server``); they
skip if the server is down.
"""

from __future__ import annotations

import base64
import os
from unittest.mock import MagicMock

import pytest

from lib.t4_visualizer_client import (
    ENV_BASE_URL,
    ENV_CF_ACCESS_CLIENT_ID,
    ENV_CF_ACCESS_CLIENT_SECRET,
    RenderRequest,
    T4VisualizerClient,
    T4VisualizerError,
    TargetObjectIn,
    format_t4_visualizer_error,
    target_object_from_gt_row,
)


# Minimal valid 1x1 PNG (transparent pixel)
_TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
)
_TINY_PNG_B64 = base64.b64encode(_TINY_PNG_BYTES).decode("ascii")


def _ok_response(json_data):
    r = MagicMock()
    r.ok = True
    r.status_code = 200
    r.text = ""
    r.headers = {"content-type": "application/json"}
    r.json.return_value = json_data
    return r


def _err_response(status_code: int, text: str = "not found"):
    r = MagicMock()
    r.ok = False
    r.status_code = status_code
    r.text = text
    r.headers = {"content-type": "text/plain"}
    return r


def test_health_success():
    session = MagicMock()
    session.get.return_value = _ok_response({"status": "ok"})
    c = T4VisualizerClient(base_url="http://test:9999", session=session)
    assert c.health() == {"status": "ok"}
    session.get.assert_called_once()
    assert "health" in session.get.call_args[0][0]


def test_list_datasets_success():
    session = MagicMock()
    session.get.return_value = _ok_response(
        {"data_dir": "/data", "datasets": ["ds_a", "ds_b"]}
    )
    c = T4VisualizerClient(base_url="http://test", session=session)
    d = c.list_datasets()
    assert d["datasets"] == ["ds_a", "ds_b"]
    assert d["data_dir"] == "/data"
    assert session.get.call_args.kwargs["headers"] is None


def test_cloudflare_access_service_token_headers_from_env(monkeypatch):
    monkeypatch.setenv(ENV_CF_ACCESS_CLIENT_ID, "client-id")
    monkeypatch.setenv(ENV_CF_ACCESS_CLIENT_SECRET, "client-secret")
    session = MagicMock()
    session.get.return_value = _ok_response({"data_dir": "/data", "datasets": []})
    c = T4VisualizerClient(base_url="http://test", session=session)

    c.list_datasets()

    assert session.get.call_args.kwargs["headers"] == {
        "CF-Access-Client-Id": "client-id",
        "CF-Access-Client-Secret": "client-secret",
    }


def test_list_dataset_scenarios_success():
    session = MagicMock()
    session.get.return_value = _ok_response(
        {
            "t4dataset_id": "ds1",
            "scenarios": [
                {
                    "name": "scene-a",
                    "token": "tok",
                    "description": "",
                    "nbr_samples": 42,
                }
            ],
            "version": None,
        }
    )
    c = T4VisualizerClient(base_url="http://test", session=session)
    out = c.list_dataset_scenarios("ds1")
    assert out["t4dataset_id"] == "ds1"
    assert len(out["scenarios"]) == 1
    assert out["scenarios"][0]["name"] == "scene-a"
    assert out["scenarios"][0]["nbr_samples"] == 42
    session.get.assert_called_once()
    call_url = session.get.call_args[0][0]
    assert "ds1" in call_url and "scenarios" in call_url


def test_render_success_decode():
    session = MagicMock()
    session.post.return_value = _ok_response(
        {
            "sample_token": "tok1",
            "timestamp_us": 1234567890000000,
            "images": [{"label": "CAM_FRONT", "png_base64": _TINY_PNG_B64}],
        }
    )
    c = T4VisualizerClient(base_url="http://test", session=session)
    req = RenderRequest(
        t4dataset_id="ds1",
        scenario_name="scene-1",
        frame_index=0,
        target_objects=[TargetObjectIn(uuid="u1", x=1.0, y=2.0, z=0.5, label="car")],
    )
    out = c.render(req)
    assert out.sample_token == "tok1"
    assert out.timestamp_us == 1234567890000000
    assert len(out.images) == 1
    raw = out.decode_png("CAM_FRONT")
    assert raw == _TINY_PNG_BYTES
    all_pairs = out.decode_all_images()
    assert all_pairs == [("CAM_FRONT", _TINY_PNG_BYTES)]


def test_render_http_error():
    session = MagicMock()
    session.post.return_value = _err_response(404, "Dataset 'x' not found")
    c = T4VisualizerClient(base_url="http://test", session=session)
    req = RenderRequest(t4dataset_id="x", scenario_name="s", frame_index=0)
    with pytest.raises(T4VisualizerError) as ei:
        c.render(req)
    assert ei.value.status_code == 404
    assert "404" in str(ei.value) or "not found" in ei.value.response_text.lower()


def test_render_invalid_json_body():
    session = MagicMock()
    r = MagicMock()
    r.ok = True
    r.status_code = 200
    r.text = "<html>login required</html>"
    r.headers = {"content-type": "text/html; charset=utf-8"}
    r.json.side_effect = ValueError("bad json")
    session.post.return_value = r
    c = T4VisualizerClient(base_url="http://test", session=session)
    with pytest.raises(T4VisualizerError, match="Invalid JSON") as ei:
        c.render(RenderRequest(t4dataset_id="a", scenario_name="b", frame_index=0))
    assert ei.value.status_code == 200
    assert "content-type=text/html" in str(ei.value)
    assert "login required" in str(ei.value)
    assert ei.value.response_text == "<html>login required</html>"


def test_cloudflare_access_login_html_gets_actionable_hint():
    session = MagicMock()
    r = MagicMock()
    r.ok = True
    r.status_code = 200
    r.text = "<!DOCTYPE html><title>Sign in - Cloudflare Access</title>"
    r.headers = {"content-type": "text/html"}
    r.json.side_effect = ValueError("bad json")
    session.get.return_value = r
    c = T4VisualizerClient(base_url="http://test", session=session)

    with pytest.raises(T4VisualizerError) as ei:
        c.dataset_availability("ds1")

    message = str(ei.value)
    assert "Cloudflare Access returned a sign-in page" in message
    assert ENV_CF_ACCESS_CLIENT_ID in message
    assert ENV_CF_ACCESS_CLIENT_SECRET in message


def test_dataset_availability_invalid_json_message_has_status_and_preview():
    session = MagicMock()
    r = MagicMock()
    r.ok = True
    r.status_code = 200
    r.text = ""
    r.headers = {"content-type": "text/plain"}
    r.json.side_effect = ValueError("bad json")
    session.get.return_value = r
    c = T4VisualizerClient(base_url="http://test", session=session)
    with pytest.raises(T4VisualizerError) as ei:
        c.dataset_availability("ds1")
    assert ei.value.status_code == 200
    assert "Invalid JSON from /datasets/.../availability" in str(ei.value)
    assert "status=200" in str(ei.value)
    assert "empty body" in str(ei.value)
    assert format_t4_visualizer_error(ei.value).startswith("T4 server error (200):")


def test_format_t4_visualizer_error_omits_missing_status():
    err = T4VisualizerError("Invalid JSON from /datasets/.../availability")
    assert format_t4_visualizer_error(err) == (
        "T4 server error: Invalid JSON from /datasets/.../availability"
    )


def test_target_object_from_gt_row_full():
    row = {
        "uuid": "abc-123",
        "x": 10.5,
        "y": -2.0,
        "z": 0.1,
        "label": "pedestrian",
        "width": 0.5,
        "length": 0.6,
        "height": 1.7,
        "yaw": 0.25,
    }
    d = target_object_from_gt_row(row)
    assert d["uuid"] == "abc-123"
    assert d["x"] == 10.5
    assert d["y"] == -2.0
    assert d["z"] == 0.1
    assert d["label"] == "pedestrian"
    assert d["width"] == 0.5
    assert d["length"] == 0.6
    assert d["height"] == 1.7
    assert d["yaw"] == 0.25


def test_target_object_from_gt_row_gt_uuid_partial():
    row = {"gt_uuid": "g1", "x": 1.0, "y": 2.0, "label": "car"}
    d = target_object_from_gt_row(row)
    assert d["uuid"] == "g1"
    assert d["z"] == 0.0
    assert d["width"] == 0.0
    assert d["length"] == 0.0
    assert d["height"] == 0.0
    assert d["yaw"] == 0.0


def test_target_object_from_gt_row_uuid_precedence():
    row = {"uuid": "u", "gt_uuid": "g", "x": 0, "y": 0}
    d = target_object_from_gt_row(row)
    assert d["uuid"] == "u"


@pytest.mark.integration
def test_live_health_if_configured():
    """Skips unless T4_VISUALIZER_BASE_URL is set and server responds."""
    base = os.environ.get(ENV_BASE_URL)
    if not base:
        pytest.skip(f"Set {ENV_BASE_URL} to run integration test against a live server")
    client = T4VisualizerClient(base_url=base, timeout=5.0)
    try:
        h = client.health()
    except (T4VisualizerError, OSError) as e:
        pytest.skip(f"Server not reachable: {e}")
    assert h.get("status") == "ok"
