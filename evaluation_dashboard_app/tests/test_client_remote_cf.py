"""Cloudflare Access sits in front of the dashboard, so the local client has to get
past the edge before any of its own auth matters. These tests cover that boundary:
which credential is sent, and whether an Access refusal is named as one."""

import io
import urllib.error
import urllib.request

import pytest

from client import remote
from client.config import Config


@pytest.fixture(autouse=True)
def _clean_session(monkeypatch):
    """The JWT cache is module-global; no test may inherit another's."""
    monkeypatch.setattr(remote, "_CF_SESSION", {})


def _cfg(**over):
    cfg = Config(server_url="https://dash.example.test", verify_tls=True)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class _Headers(dict):
    def get(self, key, default=None):  # HTTP messages look up case-insensitively
        return super().get(key.lower(), default)


# ------------------------------------------------------------------ credentials


def test_service_token_wins_over_a_browser_session(monkeypatch):
    """An unattended machine must not depend on a session that expires overnight."""
    monkeypatch.setattr(remote, "cf_session_token", lambda url: "aa.bb.cc")
    headers = remote.Remote(_cfg(cf_client_id="id.access", cf_client_secret="shh"))._headers()
    assert headers["CF-Access-Client-Id"] == "id.access"
    assert "Cookie" not in headers


def test_a_cloudflared_session_is_used_when_no_service_token_is_set(monkeypatch):
    monkeypatch.setattr(remote, "cf_session_token", lambda url: "aa.bb.cc")
    assert remote.Remote(_cfg())._headers()["Cookie"] == "CF_Authorization=aa.bb.cc"


def test_no_credential_is_invented_when_there_is_no_session(monkeypatch):
    monkeypatch.setattr(remote, "cf_session_token", lambda url: "")
    headers = remote.Remote(_cfg())._headers()
    assert "Cookie" not in headers and "CF-Access-Client-Id" not in headers


def test_the_user_agent_is_never_the_stock_python_one():
    """Cloudflare's browser-integrity check 403s Python-urllib with error code 1010."""
    agent = remote.Remote(_cfg())._headers()["User-Agent"]
    assert "python-urllib" not in agent.lower()


# ------------------------------------------------------------------ token reading


def _fake_run(stdout, returncode=0):
    class _Done:
        pass

    done = _Done()
    done.stdout, done.returncode = stdout, returncode
    return lambda *a, **k: done


def test_a_cached_jwt_is_read_per_host_and_reused(monkeypatch):
    calls = []
    monkeypatch.setattr(remote.shutil, "which", lambda _: "/usr/bin/cloudflared")
    monkeypatch.setattr(remote.subprocess, "run",
                        lambda *a, **k: calls.append(a[0]) or _fake_run("aa.bb.cc")())
    assert remote.cf_session_token("https://dash.example.test/bbox-api") == "aa.bb.cc"
    assert remote.cf_session_token("https://dash.example.test/bbox-api") == "aa.bb.cc"
    assert len(calls) == 1  # cached, and asked for the host, not the API path
    assert calls[0][-1] == "-app=https://dash.example.test"


def test_cloudflared_prose_is_not_mistaken_for_a_token(monkeypatch):
    monkeypatch.setattr(remote.shutil, "which", lambda _: "/usr/bin/cloudflared")
    monkeypatch.setattr(remote.subprocess, "run",
                        _fake_run("Unable to find token for provided application."))
    assert remote.cf_session_token("https://dash.example.test") == ""


def test_a_missing_cloudflared_is_survivable(monkeypatch):
    monkeypatch.setattr(remote.shutil, "which", lambda _: None)
    assert remote.cf_session_token("https://dash.example.test") == ""


def test_browser_login_without_cloudflared_explains_itself(monkeypatch):
    monkeypatch.setattr(remote.shutil, "which", lambda _: None)
    with pytest.raises(remote.AuthError, match="not installed"):
        remote.cf_browser_login("https://dash.example.test")


# ------------------------------------------------------------------ refusals


def test_the_access_login_bounce_is_named_not_followed():
    handler = remote._AccessAwareRedirect()
    request = urllib.request.Request("https://dash.example.test/api/export_health")
    with pytest.raises(remote.AuthError, match="Cloudflare Access"):
        handler.redirect_request(request, None, 302, "Found", _Headers(),
                                 "https://x.cloudflareaccess.com/cdn-cgi/access/login/y")


@pytest.mark.parametrize("body, header", [
    ("error code: 1010\n", {}),
    ("<html>cloudflare</html>", {}),
    ("", {"www-authenticate": 'Cloudflare-Access resource_metadata="..."'}),
])
def test_edge_refusals_read_as_access_not_as_a_server_saying_no(monkeypatch, body, header):
    remote_obj = remote.Remote(_cfg())

    def boom(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden",
                                     _Headers(header), io.BytesIO(body.encode()))

    monkeypatch.setattr(remote_obj._opener, "open", boom)
    monkeypatch.setattr(remote, "cf_session_token", lambda url: "")
    with pytest.raises(remote.AuthError, match="Cloudflare Access"):
        remote_obj.export_health()


def test_an_ordinary_403_still_reads_as_the_server_refusing(monkeypatch):
    remote_obj = remote.Remote(_cfg())

    def boom(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", _Headers(),
                                     io.BytesIO(b'{"error": "requires an export token"}'))

    monkeypatch.setattr(remote_obj._opener, "open", boom)
    monkeypatch.setattr(remote, "cf_session_token", lambda url: "")
    with pytest.raises(remote.AuthError, match="requires an export token"):
        remote_obj.export_health()
