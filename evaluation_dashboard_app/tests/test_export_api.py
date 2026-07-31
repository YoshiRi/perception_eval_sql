"""The export routes hand out raw bytes, so auth and the path sandbox are load-bearing."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import export_api


class _Handler:
    """Minimal stand-in for BaseHTTPRequestHandler: headers plus a captured response."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.status: int | None = None
        self.sent: dict[str, str] = {}
        self.body = b""
        self.ended = False

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, key: str, value: str) -> None:
        self.sent[key.lower()] = value

    def end_headers(self) -> None:
        self.ended = True

    @property
    def wfile(self):
        handler = self

        class _Sink:
            closed = False

            def write(self, chunk: bytes) -> None:
                handler.body += chunk

        return _Sink()


def _auth(token: str = "secret") -> _Handler:
    return _Handler({"Authorization": f"Bearer {token}"})


@pytest.fixture()
def data_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    run = root / "run_one" / "devops"
    run.mkdir(parents=True)
    (root / "run_one" / "metadata.yaml").write_text("version: 1\n")
    (root / "run_one" / "Summary.csv").write_text("a,b\n1,2\n")
    (run / "current.parquet").write_bytes(b"PAR1" + b"x" * 500)
    (run / "future.parquet").write_bytes(b"PAR1" + b"y" * 100)
    scenario = run / "DevOps_V1_Suite_abc" / "Scenario_One"
    scenario.mkdir(parents=True)
    (scenario / "scenario.yaml").write_text("Evaluation: {}\n")
    (scenario / "scene_result.pkl").write_bytes(b"\x80\x04" + b"z" * 4000)
    # Regenerable derived state and published reports must never be exported.
    (run / ".dashboard_cache").mkdir()
    (run / ".dashboard_cache" / "big.parquet").write_bytes(b"x" * 9999)
    (root / "run_one" / "specsheet").mkdir()
    (root / "run_one" / "specsheet" / "report.pdf").write_bytes(b"%PDF" + b"x" * 9999)
    monkeypatch.setenv("EVAL_DASHBOARD_DATA_ROOT", str(root))
    monkeypatch.setenv("EVAL_EXPORT_TOKEN", "secret")
    return root


# ------------------------------------------------------------------------- auth
#
# Policy under test (see backend/export_api.py): access reuses the dashboard's identity
# model instead of a second secret. Direct hits are internal and allowed; Cloudflare
# hits must carry an authenticated identity; a token still works and can be demanded.


def _cf(email: str = "", **extra) -> _Handler:
    """A request that looks like it came through Cloudflare."""
    headers = {"Cf-Ray": "8abc123-NRT", "Cdn-Loop": "cloudflare", **extra}
    if email:
        headers["Cf-Access-Authenticated-User-Email"] = email
    return _Handler(headers)


def test_direct_hit_is_allowed_without_a_token(data_root, monkeypatch):
    """Anyone who can reach this port can already list and query every run file
    through the existing routes, so a separate secret would be friction, not a boundary."""
    monkeypatch.delenv("EVAL_EXPORT_TOKEN", raising=False)
    who = export_api.require_auth(_Handler())
    assert who == {"via": "direct", "actor": "anonymous"}
    assert export_api.exports_enabled() is True


def test_cloudflare_with_identity_is_allowed_and_attributed(data_root, monkeypatch):
    monkeypatch.delenv("EVAL_EXPORT_TOKEN", raising=False)
    who = export_api.require_auth(_cf("lei.gu@tier4.jp"))
    assert who == {"via": "cloudflare", "actor": "lei.gu@tier4.jp"}


def test_cloudflare_without_identity_fails_closed(data_root, monkeypatch):
    """Falling back to the permissive direct rule here would make the edge bypassable
    by anyone who can set a Cf-Ray header."""
    monkeypatch.delenv("EVAL_EXPORT_TOKEN", raising=False)
    with pytest.raises(export_api.ExportAuthError, match="without an authenticated identity"):
        export_api.require_auth(_cf())


def test_forged_identity_on_a_direct_hit_is_not_believed(data_root, monkeypatch):
    """The email header is only meaningful alongside the other Cf-* signals; a direct
    caller must not be able to manufacture an identity for the audit log."""
    monkeypatch.delenv("EVAL_EXPORT_TOKEN", raising=False)
    forged = _Handler({"Cf-Access-Authenticated-User-Email": "attacker@evil.com"})
    who = export_api.require_auth(forged)
    assert who["actor"] == "anonymous"
    assert who["via"] == "direct"


def test_correct_token_is_accepted_in_either_header(data_root):
    assert export_api.require_auth(_auth())["via"] == "token"
    export_api.require_auth(_Handler({"Authorization": "bearer secret"}))  # case-insensitive
    export_api.require_auth(_Handler({"X-Export-Token": "secret"}))


def test_wrong_token_is_rejected_even_though_direct_would_pass(data_root):
    """Presenting a bad credential is an error, not something to silently fall back from."""
    with pytest.raises(export_api.ExportAuthError, match="Invalid export token"):
        export_api.require_auth(_auth("wrong"))


def test_no_token_presented_falls_through_to_identity(data_root):
    # EVAL_EXPORT_TOKEN is set by the fixture, but presenting none is still fine on a
    # direct hit: the token is an alternative, not a requirement.
    assert export_api.require_auth(_Handler())["via"] == "direct"


def test_require_token_closes_the_identity_paths(data_root, monkeypatch):
    monkeypatch.setenv("EVAL_EXPORT_REQUIRE_TOKEN", "1")
    with pytest.raises(export_api.ExportAuthError, match="requires an export token"):
        export_api.require_auth(_Handler())
    with pytest.raises(export_api.ExportAuthError):
        export_api.require_auth(_cf("lei.gu@tier4.jp"))
    assert export_api.require_auth(_auth())["via"] == "token"


def test_require_token_without_a_token_is_unusable_and_says_so(monkeypatch):
    monkeypatch.setenv("EVAL_EXPORT_REQUIRE_TOKEN", "1")
    monkeypatch.delenv("EVAL_EXPORT_TOKEN", raising=False)
    assert export_api.exports_enabled() is False
    with pytest.raises(export_api.ExportDisabledError, match="no caller can be authorized"):
        export_api.require_auth(_Handler())


def test_allow_direct_off_refuses_lan_but_keeps_cloudflare(data_root, monkeypatch):
    monkeypatch.delenv("EVAL_EXPORT_TOKEN", raising=False)
    monkeypatch.setenv("EVAL_EXPORT_ALLOW_DIRECT", "0")
    with pytest.raises(export_api.ExportAuthError, match="Direct .* is disabled"):
        export_api.require_auth(_Handler())
    assert export_api.require_auth(_cf("lei.gu@tier4.jp"))["via"] == "cloudflare"


@pytest.mark.parametrize("value,expected", [("0", False), ("false", False), ("no", False),
                                            ("off", False), ("1", True), ("yes", True)])
def test_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("EVAL_EXPORT_ALLOW_DIRECT", value)
    assert export_api.allow_direct() is expected


def test_health_reports_the_policy_without_authorizing(monkeypatch, data_root):
    """Clients read this to decide whether to ask for a token at all, so it must answer
    even to a caller that every other route would refuse."""
    monkeypatch.delenv("EVAL_EXPORT_TOKEN", raising=False)
    health = export_api.export_health(_cf(), {})   # would be refused elsewhere
    assert health["service"] == "eval_dashboard_export"
    assert health["authorized"] is False
    assert "without an authenticated identity" in health["auth_reason"]
    assert health["origin"] == "cloudflare"

    ok = export_api.export_health(_Handler(), {})
    assert ok["authorized"] is True
    assert ok["token_required"] is False


# ---------------------------------------------------------------------- sandbox


@pytest.mark.parametrize(
    "name",
    ["", "..", ".", "../etc", "run_one/devops", "run_one\\devops", "~", "/absolute"],
)
def test_run_names_are_names_not_paths(data_root, name):
    with pytest.raises(export_api.ExportError):
        export_api._resolve_run(name)


def test_unknown_run_is_rejected(data_root):
    with pytest.raises(export_api.ExportError):
        export_api._resolve_run("no_such_run")


@pytest.mark.parametrize(
    "rel",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "../run_one/../../outside.txt",
        "devops/../../../etc/passwd",
        "",
        "does/not/exist",
    ],
)
def test_files_cannot_escape_the_run_directory(data_root, rel):
    run_dir = export_api._resolve_run("run_one")
    with pytest.raises(export_api.ExportError):
        export_api._resolve_in_run(run_dir, rel)


def test_traversal_that_lands_back_inside_the_run_is_allowed(data_root):
    """``a/../b`` is not an escape once normalised, and refusing it would only reject
    legitimate manifest paths -- the check is on the resolved location, not the spelling."""
    run_dir = export_api._resolve_run("run_one")
    resolved = export_api._resolve_in_run(run_dir, "devops/../metadata.yaml")
    assert resolved == run_dir / "metadata.yaml"


def test_resolve_in_run_accepts_a_real_file(data_root):
    run_dir = export_api._resolve_run("run_one")
    assert export_api._resolve_in_run(run_dir, "devops/current.parquet").name == "current.parquet"
    assert export_api._resolve_in_run(run_dir, "/devops/current.parquet").name == "current.parquet"


def test_symlink_out_of_the_run_is_refused(data_root, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    link = data_root / "run_one" / "link.txt"
    link.symlink_to(outside)
    run_dir = export_api._resolve_run("run_one")
    with pytest.raises(export_api.ExportError):
        export_api._resolve_in_run(run_dir, "link.txt")


# --------------------------------------------------------------------- manifest


def _rels(manifest) -> set[str]:
    return {f["rel_path"] for f in manifest["files"]}


def test_minimal_tier_is_parquet_plus_identity(data_root):
    manifest = export_api.export_manifest(_auth(), {"run": "run_one", "tier": "minimal"})
    assert "devops/current.parquet" in _rels(manifest)
    assert "metadata.yaml" in _rels(manifest)
    assert not any("scenario.yaml" in r for r in _rels(manifest))
    assert not any("scene_result.pkl" in r for r in _rels(manifest))


def test_future_parquet_is_opt_in(data_root):
    without = export_api.export_manifest(_auth(), {"run": "run_one", "tier": "minimal"})
    assert "devops/future.parquet" not in _rels(without)
    with_future = export_api.export_manifest(
        _auth(), {"run": "run_one", "tier": "minimal", "include_future": True}
    )
    assert "devops/future.parquet" in _rels(with_future)


def test_criteria_tier_adds_sidecars_but_not_pickles(data_root):
    manifest = export_api.export_manifest(_auth(), {"run": "run_one", "tier": "criteria"})
    assert "devops/DevOps_V1_Suite_abc/Scenario_One/scenario.yaml" in _rels(manifest)
    assert not any("scene_result.pkl" in r for r in _rels(manifest))


def test_raw_tier_adds_pickles(data_root):
    manifest = export_api.export_manifest(_auth(), {"run": "run_one", "tier": "raw"})
    assert "devops/DevOps_V1_Suite_abc/Scenario_One/scene_result.pkl" in _rels(manifest)


def test_tiers_are_monotonic(data_root):
    seen = [
        _rels(export_api.export_manifest(_auth(), {"run": "run_one", "tier": tier}))
        for tier in export_api.TIERS
    ]
    for smaller, larger in zip(seen, seen[1:]):
        assert smaller <= larger


def test_derived_and_published_artifacts_are_never_exported(data_root):
    """.dashboard_cache is regenerable and specsheet/ dwarfs everything else."""
    for tier in export_api.TIERS:
        rels = _rels(export_api.export_manifest(_auth(), {"run": "run_one", "tier": tier}))
        assert not any(".dashboard_cache" in r for r in rels)
        assert not any("specsheet" in r for r in rels)


def test_manifest_enforces_the_access_policy(data_root):
    # Direct hits are internal and allowed...
    assert export_api.export_manifest(_Handler(), {"run": "run_one"})["run"] == "run_one"
    # ...but a request through the edge without an identity is refused.
    with pytest.raises(export_api.ExportAuthError):
        export_api.export_manifest(_cf(), {"run": "run_one"})


def test_unknown_tier_is_rejected(data_root):
    with pytest.raises(export_api.ExportError):
        export_api.export_manifest(_auth(), {"run": "run_one", "tier": "enormous"})


def test_unknown_role_is_rejected_with_a_helpful_message(data_root):
    with pytest.raises(export_api.ExportError) as excinfo:
        export_api.export_manifest(_auth(), {"run": "run_one", "role": "performance"})
    assert "devops" in str(excinfo.value)


def test_checksums_are_opt_in(data_root):
    plain = export_api.export_manifest(_auth(), {"run": "run_one", "tier": "minimal"})
    assert all("sha256" not in f for f in plain["files"])
    hashed = export_api.export_manifest(
        _auth(), {"run": "run_one", "tier": "minimal", "checksums": True}
    )
    assert all(len(f["sha256"]) == 64 for f in hashed["files"])


def test_totals_match_the_file_list(data_root):
    manifest = export_api.export_manifest(_auth(), {"run": "run_one", "tier": "criteria"})
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["total_bytes"] == sum(f["size"] for f in manifest["files"])


# ------------------------------------------------------------------------- runs


def test_runs_lists_with_per_tier_sizes(data_root):
    result = export_api.runs(_auth(), {})
    names = [item["name"] for item in result["items"]]
    assert names == ["run_one"]
    tiers = result["items"][0]["tier_bytes"]
    assert tiers["minimal"] <= tiers["criteria"] <= tiers["raw"]


def test_runs_query_filters(data_root):
    assert export_api.runs(_auth(), {"q": "zzz", "sizes": False})["items"] == []
    assert len(export_api.runs(_auth(), {"q": "run", "sizes": False})["items"]) == 1


def test_runs_enforces_the_access_policy(data_root):
    assert export_api.runs(_Handler(), {"sizes": False})["items"]
    with pytest.raises(export_api.ExportAuthError):
        export_api.runs(_cf(), {"sizes": False})


# ------------------------------------------------------------------ range parse


@pytest.mark.parametrize(
    "header,size,expected",
    [
        ("bytes=0-99", 500, (0, 99)),
        ("bytes=100-", 500, (100, 499)),
        ("bytes=-50", 500, (450, 499)),
        ("bytes=0-9999", 500, (0, 499)),  # clamped to the file
        ("bytes=499-499", 500, (499, 499)),
    ],
)
def test_range_parsing(header, size, expected):
    assert export_api._parse_range(header, size) == expected


@pytest.mark.parametrize(
    "header",
    ["", "items=0-1", "bytes=abc-def", "bytes=500-", "bytes=-0", "bytes=200-100", "bytes=0-1,5-6"],
)
def test_unusable_ranges_fall_back_to_a_full_body(header):
    assert export_api._parse_range(header, 500) is None


# ----------------------------------------------------------------- file stream


def test_file_stream_sends_the_whole_body(data_root):
    handler = _auth()
    export_api.export_file(handler, {"run": "run_one", "rel_path": "metadata.yaml"})
    assert handler.status == 200
    assert handler.body == b"version: 1\n"
    assert handler.sent["accept-ranges"] == "bytes"
    assert handler.sent["content-length"] == str(len(handler.body))


def test_ranged_request_returns_206_and_the_slice(data_root):
    handler = _Handler({"Authorization": "Bearer secret", "Range": "bytes=0-6"})
    export_api.export_file(handler, {"run": "run_one", "rel_path": "metadata.yaml"})
    assert handler.status == 206
    assert handler.body == b"version"
    assert handler.sent["content-range"] == "bytes 0-6/11"


def test_ranges_reassemble_exactly(data_root):
    first = _Handler({"Authorization": "Bearer secret", "Range": "bytes=0-3"})
    export_api.export_file(first, {"run": "run_one", "rel_path": "devops/current.parquet"})
    second = _Handler({"Authorization": "Bearer secret", "Range": "bytes=4-"})
    export_api.export_file(second, {"run": "run_one", "rel_path": "devops/current.parquet"})
    whole = (data_root / "run_one" / "devops" / "current.parquet").read_bytes()
    assert first.body + second.body == whole


def test_matching_etag_yields_304_with_no_body(data_root):
    probe = _auth()
    export_api.export_file(probe, {"run": "run_one", "rel_path": "metadata.yaml"})
    etag = probe.sent["etag"]
    cached = _Handler({"Authorization": "Bearer secret", "If-None-Match": etag})
    export_api.export_file(cached, {"run": "run_one", "rel_path": "metadata.yaml"})
    assert cached.status == 304
    assert cached.body == b""


def test_etag_changes_when_the_file_changes(data_root):
    first = _auth()
    export_api.export_file(first, {"run": "run_one", "rel_path": "metadata.yaml"})
    (data_root / "run_one" / "metadata.yaml").write_text("version: 2\nmore: yes\n")
    second = _auth()
    export_api.export_file(second, {"run": "run_one", "rel_path": "metadata.yaml"})
    assert first.sent["etag"] != second.sent["etag"]


def test_head_sends_headers_without_a_body(data_root):
    handler = _auth()
    export_api.export_file(handler, {"run": "run_one", "rel_path": "metadata.yaml"}, head_only=True)
    assert handler.status == 200
    assert handler.body == b""
    assert handler.sent["x-export-size"] == "11"


def test_file_stream_enforces_the_access_policy(data_root):
    allowed = _Handler()
    export_api.export_file(allowed, {"run": "run_one", "rel_path": "metadata.yaml"})
    assert allowed.status == 200

    refused = _cf()
    with pytest.raises(export_api.ExportAuthError):
        export_api.export_file(refused, {"run": "run_one", "rel_path": "metadata.yaml"})
    # Nothing may be committed to the wire before authorization succeeds, or the
    # dispatcher cannot turn the failure into a clean JSON error.
    assert refused.status is None and refused.body == b""


def test_stream_marks_itself_committed(data_root):
    """The dispatcher relies on this flag to avoid overwriting a started response."""
    handler = _auth()
    export_api.export_file(handler, {"run": "run_one", "rel_path": "metadata.yaml"})
    assert getattr(handler, "_export_stream_started", False) is True
