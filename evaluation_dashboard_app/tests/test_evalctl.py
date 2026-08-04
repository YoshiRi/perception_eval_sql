"""evalctl is what agents and scripts trust to drive workflows, so its request
building and refusal logic matter more than its printing."""

import base64
import importlib.util
import io
import json
import sys
import urllib.error
import urllib.request
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


# ------------------------------------------------------------- cloudflare access


@pytest.fixture(autouse=True)
def _clean_cf_state(monkeypatch):
    """Access credentials are process-global; no test may inherit another's."""
    monkeypatch.setattr(evalctl, "_CF",
                        {"url": "", "headers": {}, "mode": "none", "expires_at": None})
    for name in ("CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET",
                 "EVAL_CF_CLIENT_ID", "EVAL_CF_CLIENT_SECRET", "EVAL_CF_AUTO_LOGIN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EVAL_DASHBOARD_URL", "https://dash.example.test")


class _Headers(dict):
    """Mimic the case-insensitive lookup of an HTTP message."""

    def get(self, key, default=None):  # type: ignore[override]
        return super().get(key.lower(), default)


def test_access_login_redirect_is_recognised_not_followed():
    handler = evalctl._AccessAwareRedirect()
    request = urllib.request.Request("https://dash.example.test/api/workflow_health")
    login = ("https://tier4inc.cloudflareaccess.com/cdn-cgi/access/login/dash.example.test"
             "?kid=abc")
    with pytest.raises(evalctl.CloudflareAccessRequired):
        handler.redirect_request(request, None, 302, "Found", _Headers(), login)


def test_ordinary_redirects_still_follow():
    handler = evalctl._AccessAwareRedirect()
    request = urllib.request.Request("https://dash.example.test/api/workflow_health")
    assert handler.redirect_request(
        request, None, 302, "Found", _Headers({"location": "https://dash.example.test/x"}),
        "https://dash.example.test/x",
    ) is not None


@pytest.mark.parametrize("code, headers, body", [
    (403, _Headers(), "error code: 1010"),
    (403, _Headers({"www-authenticate": 'Cloudflare-Access resource_metadata="..."'}), ""),
    (302, _Headers({"location": "https://x.cloudflareaccess.com/cdn-cgi/access/login/y"}), ""),
])
def test_access_refusals_are_told_apart_from_app_refusals(code, headers, body):
    assert evalctl._looks_like_access_refusal(code, headers, body)


@pytest.mark.parametrize("code, headers, body", [
    (403, _Headers(), '{"error": "requires an export token"}'),
    (500, _Headers(), "boom"),
    (404, _Headers(), "no such route"),
])
def test_app_refusals_are_not_mistaken_for_access(code, headers, body):
    assert not evalctl._looks_like_access_refusal(code, headers, body)


def test_service_token_is_sent_without_waiting_to_be_challenged(monkeypatch):
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "id.access")
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", "shh")
    headers = evalctl._cf_headers("https://dash.example.test")
    assert headers == {"CF-Access-Client-Id": "id.access", "CF-Access-Client-Secret": "shh"}
    assert evalctl.cf_status() == "service token"


def test_a_challenge_authenticates_once_and_retries(monkeypatch):
    args = _args(argv=["doctor"])
    attempts = []

    def once(_args, path, payload):
        attempts.append(path)
        if len(attempts) == 1:
            raise evalctl.CloudflareAccessRequired("challenged")
        return {"authorized": True}

    monkeypatch.setattr(evalctl, "_api_once", once)
    monkeypatch.setattr(evalctl, "_cloudflared_token", lambda base: "aa.bb.cc")
    assert evalctl.api(args, "/api/workflow_health") == {"authorized": True}
    assert len(attempts) == 2
    assert evalctl._CF["headers"] == {"cookie": "CF_Authorization=aa.bb.cc"}


def test_an_expired_session_forces_a_fresh_login(monkeypatch):
    args = _args(argv=["doctor"])
    logins = []

    def once(_args, path, payload):
        if not logins:
            raise evalctl.CloudflareAccessRequired("stale cookie")
        return {"authorized": True}

    monkeypatch.setattr(evalctl, "_api_once", once)
    monkeypatch.setattr(evalctl, "_cloudflared_token", lambda base: "stale.jwt.x")
    monkeypatch.setattr(evalctl, "_cloudflared_login",
                        lambda base: logins.append(base) or "fresh.jwt.y")
    assert evalctl.api(args, "/api/workflow_health") == {"authorized": True}
    assert logins == ["https://dash.example.test"]  # cached token tried first, then login


def test_no_cf_login_refuses_instead_of_opening_a_browser(monkeypatch, capsys):
    monkeypatch.setattr(evalctl, "_cloudflared_token", lambda base: "")
    monkeypatch.setattr(evalctl, "_cloudflared_login",
                        lambda base: pytest.fail("must not open a browser"))
    monkeypatch.setattr(evalctl, "_api_once",
                        lambda *a, **k: (_ for _ in ()).throw(
                            evalctl.CloudflareAccessRequired("challenged")))
    assert evalctl.main(["--no-cf-login", "doctor"]) == 2
    assert "cloudflared access login" in capsys.readouterr().out


def test_a_rejected_service_token_says_so_instead_of_looping(monkeypatch):
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "wrong.access")
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", "shh")
    args = _args(argv=["doctor"])
    monkeypatch.setattr(evalctl, "_api_once",
                        lambda *a, **k: (_ for _ in ()).throw(
                            evalctl.CloudflareAccessRequired("challenged")))
    with pytest.raises(evalctl.ApiError, match="rejected the service token"):
        evalctl.api(args, "/api/workflow_health")


def test_the_stock_python_user_agent_is_never_sent(monkeypatch):
    """Cloudflare's browser-integrity check 403s Python-urllib even with a valid session."""
    sent = {}

    class _Response:
        headers = {"Content-Type": "application/json"}

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_open(request, timeout=None):
        sent.update(request.header_items())
        return _Response()

    monkeypatch.setattr(evalctl._OPENER, "open", fake_open)
    evalctl._api_once(_args(argv=["doctor"]), "/api/workflow_health", None)
    agent = {k.lower(): v for k, v in sent.items()}["user-agent"]
    assert "python-urllib" not in agent.lower()
    assert agent == evalctl.USER_AGENT


def test_edge_rejecting_the_client_is_not_reported_as_a_login_problem(monkeypatch):
    """1010 with a live session means the WAF dislikes the client, not the identity."""
    monkeypatch.setitem(evalctl._CF, "headers", {"cookie": "CF_Authorization=a.b.c"})
    monkeypatch.setitem(evalctl._CF, "mode", "browser session (cloudflared)")

    def fake_open(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", _Headers(),
                                     io.BytesIO(b"error code: 1010\n"))

    monkeypatch.setattr(evalctl._OPENER, "open", fake_open)
    with pytest.raises(evalctl.ApiError, match="browser-integrity") as caught:
        evalctl._api_once(_args(argv=["doctor"]), "/api/workflow_health", None)
    assert not isinstance(caught.value, evalctl.CloudflareAccessRequired)  # no login loop


def test_a_404_falls_forward_to_the_nginx_mount_point(monkeypatch):
    """The public hostname fronts Streamlit; the API lives under /bbox-api behind it."""
    monkeypatch.setattr(evalctl, "_PREFIX", {})
    seen = []

    def once(args, path, payload):
        seen.append(evalctl._api_base(args) + path)
        if len(seen) == 1:
            raise evalctl._RouteMissing("405")
        return {"ok": True}

    monkeypatch.setattr(evalctl, "_api_once", once)
    assert evalctl.api(_args(argv=["doctor"]), "/api/workflow_health") == {"ok": True}
    assert seen == [
        "https://dash.example.test/api/workflow_health",
        "https://dash.example.test/bbox-api/api/workflow_health",
    ]


def test_a_url_already_carrying_the_prefix_is_left_alone(monkeypatch):
    monkeypatch.setattr(evalctl, "_PREFIX", {})
    monkeypatch.setenv("EVAL_DASHBOARD_URL", "https://dash.example.test/bbox-api")
    monkeypatch.setattr(evalctl, "_api_once",
                        lambda *a, **k: (_ for _ in ()).throw(evalctl._RouteMissing("405")))
    with pytest.raises(evalctl.ApiError, match="backend API base"):
        evalctl.api(_args(argv=["doctor"]), "/api/workflow_health")


def test_jwt_expiry_is_read_for_the_status_line():
    exp = int((datetime.now(timezone.utc) + timedelta(hours=20)).timestamp())
    body = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    assert evalctl._jwt_expiry(f"aa.{body}.cc") is not None
    assert evalctl._jwt_expiry("not-a-jwt") is None
