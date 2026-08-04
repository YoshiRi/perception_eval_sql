"""Export routes that let a local client pull generated results off the server.

The rest of the bbox API only ever returns *derived* JSON, so it needs no notion of
file identity. A local client does: it has to discover runs, learn which files matter,
and fetch their bytes. These four routes add exactly that and nothing else.

They plug into ``LocalBBoxHandler`` so they inherit the existing GET/POST dispatch,
JSON error shape, and the nginx ``/bbox-api/`` mapping -- no new port or process.

Access control reuses the dashboard's own identity model rather than inventing a second
one. The dashboard has no in-app login: it trusts Cloudflare Access at the edge and
treats a direct hit as an internal request (``lib/auth.py``). These routes do the same:

* **Via Cloudflare** -- allow when the edge authenticated the user, and record who. A
  request that arrives through Cloudflare *without* an identity is refused, so the gate
  fails closed rather than open.
* **Direct hit** (LAN, container port) -- allow. Anyone who can reach this port can
  already list run files through ``/api/parquets`` and query every one of them through
  the existing routes, so requiring a separate secret here would be friction without a
  boundary.
* **Bearer token** -- still honoured when ``EVAL_EXPORT_TOKEN`` is set, for automation
  and for deployments that want an explicit gate.

Two env knobs tighten this when the exposure changes: ``EVAL_EXPORT_REQUIRE_TOKEN=1``
demands a token from everyone, and ``EVAL_EXPORT_ALLOW_DIRECT=0`` refuses non-Cloudflare
requests. Every path is confined to the resolved run directory regardless, on top of the
data-root sandbox the rest of the API uses.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable

CHUNK_BYTES = 1024 * 1024

# Files worth carrying at every tier: run identity plus the small CSV summaries the
# dashboard pages read. Kilobytes, so there is no reason to make them optional.
RUN_ROOT_FILES = ("metadata.yaml", ".run_metadata.json", "Summary.csv", "Score.csv")

# Per-scenario sidecars that make the DevOps criteria views work.
SCENARIO_SIDECARS = ("scenario.yaml", "planning_factor.jsonl", "t4_metadata.json")

# Role sub-directories of a release container, matching lib.path_utils.RELEASE_ROLE_DIRS.
ROLE_DIRS = ("devops", "performance", "usecase")

# Many runs have no role split at all: the parquet and its scenario folders sit directly
# in the run directory. Everything the viewer reads is relative to the parquet's own
# directory, so such a run is simply one whose data directory *is* the run directory.
FLAT_RUN_MARKERS = ("Summary.csv", "Score.csv", "current.csv", "future.csv")

# Written by the dashboard's own trend machinery, not runs anyone pulls.
INTERNAL_DIR_PREFIX = "trend_release_"

# TLR runs have no devops/performance split. Their scenarios sit directly under the run
# (``<run>/<scenario>/result.json``) or one suite level down
# (``<run>/<suite>/<testcase>/result.json``), which is why role detection alone used to
# skip them entirely and they never appeared in a client's run list.
TLR_RESULT_NAME = "result.json"

# Never exported: regenerable derived state, and the published report bundles that are
# far larger than everything else combined.
EXCLUDED_PARTS = (".dashboard_cache", "__pycache__", "specsheet")

TIERS = ("minimal", "criteria", "full", "raw")

TIER_DESCRIPTIONS = {
    "minimal": "Role parquet + run metadata. Bbox viewer, explorer and statistics all work.",
    "criteria": "minimal + scenario YAML/sidecars + pre-baked gate and frame verdicts.",
    "full": "criteria + pre-baked true-negative objects (the preview overlay).",
    "raw": "full + scene_result.pkl. Only needed to recompute from source; adds GBs.",
}


class ExportError(Exception):
    """Bad request against an export route."""


class ExportAuthError(Exception):
    """Missing or wrong bearer token, or exports disabled."""


class ExportDisabledError(ExportAuthError):
    """EVAL_EXPORT_TOKEN is not configured, so exports are refused outright."""


def export_token() -> str:
    return os.environ.get("EVAL_EXPORT_TOKEN", "").strip()


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def require_token_always() -> bool:
    return _flag("EVAL_EXPORT_REQUIRE_TOKEN", False)


def allow_direct() -> bool:
    return _flag("EVAL_EXPORT_ALLOW_DIRECT", True)


def exports_enabled() -> bool:
    """Whether any caller can be authorized at all.

    Unlike the token-only design this replaced, exports are normally usable without a
    token; they are only unreachable if a token is demanded but none is configured.
    """
    if require_token_always():
        return bool(export_token())
    return True


def _headers_dict(handler: Any) -> dict[str, str]:
    try:
        return {str(k): str(v) for k, v in handler.headers.items()}
    except Exception:
        return {}


def _identity(headers: dict[str, str]) -> tuple[str, str]:
    """Return ``(origin, email)`` using the dashboard's own trust rules.

    ``lib.auth`` is stdlib-only and is the single source of truth for how far a Cf-*
    header may be believed, so it is reused rather than reimplemented. It is imported
    lazily: the packaged client drops these routes entirely and does not ship ``lib/``.
    """
    try:
        from lib import auth
    except Exception:
        # No lib/ available (packaged client). Treat as a direct, unidentified hit.
        return "direct", ""
    origin = "cloudflare" if auth.detect_access_origin(headers).get("is_cloudflare") else "direct"
    # get_access_user_email() deliberately returns "" when the Cf-* signals are absent,
    # so a forged email header on a direct hit cannot manufacture an identity.
    return origin, auth.get_access_user_email(headers)


def _presented_token(handler: Any) -> str:
    raw = str(handler.headers.get("authorization") or "")
    prefix = "bearer "
    if raw.lower().startswith(prefix):
        return raw[len(prefix) :].strip()
    return str(handler.headers.get("x-export-token") or "").strip()


def require_auth(handler: Any) -> dict[str, str]:
    """Authorize a request, or raise. Returns how the caller was identified."""
    expected = export_token()
    presented = _presented_token(handler)

    if expected and presented:
        if hmac.compare_digest(presented, expected):
            return {"via": "token", "actor": "token"}
        raise ExportAuthError("Invalid export token.")

    if require_token_always():
        if not expected:
            raise ExportDisabledError(
                "EVAL_EXPORT_REQUIRE_TOKEN is set but EVAL_EXPORT_TOKEN is not, so no "
                "caller can be authorized. Set a token or unset the requirement."
            )
        raise ExportAuthError("This server requires an export token.")

    origin, email = _identity(_headers_dict(handler))

    if origin == "cloudflare":
        if email:
            return {"via": "cloudflare", "actor": email}
        # Through the edge but unauthenticated: fail closed rather than fall back to the
        # permissive direct-hit rule, which would make the edge trivially bypassable.
        raise ExportAuthError(
            "Request arrived via Cloudflare without an authenticated identity."
        )

    if not allow_direct():
        raise ExportAuthError(
            "Direct (non-Cloudflare) access to the export API is disabled on this server."
        )
    return {"via": "direct", "actor": "anonymous"}


def _data_root() -> Path:
    from backend import app_paths

    return app_paths.data_root()


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _resolve_run(name: str) -> Path:
    """Resolve a run *name* (never a path) to a directory under the data root."""
    text = _as_text(name)
    if not text:
        raise ExportError("run is required")
    if text in (".", "..") or "/" in text or "\\" in text or text.startswith("~"):
        raise ExportError(f"Invalid run name: {name}")
    root = _data_root()
    candidate = (root / text).resolve()
    if candidate != root and root not in candidate.parents:
        raise ExportError(f"Run is outside the data root: {name}")
    if not candidate.is_dir():
        raise ExportError(f"No such run: {name}")
    return candidate


def _resolve_in_run(run_dir: Path, rel_path: str) -> Path:
    """Resolve a manifest-relative path, confined to ``run_dir``."""
    text = _as_text(rel_path).replace("\\", "/").lstrip("/")
    if not text:
        raise ExportError("rel_path is required")
    candidate = (run_dir / text).resolve()
    if candidate != run_dir and run_dir not in candidate.parents:
        raise ExportError(f"Path escapes the run directory: {rel_path}")
    if not candidate.is_file():
        raise ExportError(f"No such file in run: {rel_path}")
    return candidate


def _is_excluded(path: Path, run_dir: Path) -> bool:
    try:
        parts = path.relative_to(run_dir).parts
    except ValueError:
        return True
    return any(part in EXCLUDED_PARTS for part in parts)


def _roles_present(run_dir: Path) -> list[str]:
    return [name for name in ROLE_DIRS if (run_dir / name).is_dir()]


def _data_dirs(run_dir: Path, roles: Iterable[str]) -> list[Path]:
    """Directories holding a parquet and its sidecars.

    One per role for a release container; the run directory itself for a flat run. The
    tier rules below are all expressed relative to this directory, exactly as the viewer
    resolves scenario YAML and pre-bakes relative to the parquet it opened.
    """
    dirs = [run_dir / role for role in roles]
    return dirs or [run_dir]


def _parquet_rel_paths(run_dir: Path, roles: Iterable[str]) -> list[str]:
    return sorted(
        str(path.relative_to(run_dir)).replace("\\", "/")
        for data_dir in _data_dirs(run_dir, roles)
        for path in data_dir.glob("*.parquet")
    )


def _is_flat_run(run_dir: Path) -> bool:
    """A run whose own directory holds the analysis, with no role split.

    Mirrors ``lib.path_utils._looks_like_analysis_run``, which is how the dashboard
    decides the same thing.
    """
    if any(run_dir.glob("*.parquet")):
        return True
    return any((run_dir / name).is_file() for name in FLAT_RUN_MARKERS)


def _tlr_scenarios(run_dir: Path) -> int:
    """How many TLR scenarios this run holds, flat and suite layouts both counted.

    Mirrors ``lib.path_utils.count_tlr_scenarios``, which the dashboard uses to find the
    same directories. Kept local because the export routes must not depend on ``lib``,
    which the packaged client does not ship.
    """
    count = 0
    try:
        children = sorted(run_dir.iterdir())
    except OSError:
        return 0
    for child in children:
        if not child.is_dir() or child.name.startswith(".") or child.name in EXCLUDED_PARTS:
            continue
        if (child / TLR_RESULT_NAME).is_file():
            count += 1
            continue
        try:
            testcases = sorted(child.iterdir())
        except OSError:
            continue
        for testcase in testcases:
            if testcase.is_dir() and (testcase / TLR_RESULT_NAME).is_file():
                count += 1
    return count


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path, run_dir: Path, *, with_sha: bool) -> dict[str, Any]:
    stat = path.stat()
    entry: dict[str, Any] = {
        "rel_path": str(path.relative_to(run_dir)).replace("\\", "/"),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if with_sha:
        entry["sha256"] = _sha256(path)
    return entry


# --------------------------------------------------------------------------- runs


def runs(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """List runs available on this server, with per-tier byte totals."""
    require_auth(handler)
    root = _data_root()
    query = _as_text(payload.get("q")).lower()
    with_sizes = payload.get("sizes", True) is not False
    items: list[dict[str, Any]] = []
    if not root.is_dir():
        return {"root": str(root), "items": [], "error": f"Data root does not exist: {root}"}

    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name.startswith(".") or child.name.startswith(INTERNAL_DIR_PREFIX):
            continue
        roles = _roles_present(child)
        parquets = _parquet_rel_paths(child, roles)
        flat = not roles and _is_flat_run(child)
        # Only probe for TLR results once the cheaper shapes have said no: a perception
        # run would otherwise pay an extra directory walk per listing.
        tlr_scenarios = 0 if roles or flat else _tlr_scenarios(child)
        if not roles and not flat and not tlr_scenarios:
            continue
        if query and query not in child.name.lower():
            continue
        entry: dict[str, Any] = {
            "name": child.name,
            # "tlr" runs are viewed in the TLR page, not the bbox explorer, so clients
            # need to tell them apart before they open one.
            "kind": "tlr" if tlr_scenarios else "perception",
            "tlr_scenarios": tlr_scenarios,
            "roles": roles,
            "parquets": parquets,
            "prebaked": _prebake_state(child, roles),
        }
        if with_sizes:
            entry["total_bytes"] = _dir_size(child)
            entry["tier_bytes"] = {
                tier: sum(int(f["size"]) for f in _collect(child, roles, tier, include_future=False))
                for tier in TIERS
            }
        items.append(entry)
    return {"root": str(root), "items": items, "tiers": TIER_DESCRIPTIONS}


def _prebake_state(run_dir: Path, roles: Iterable[str]) -> dict[str, Any]:
    """Per-parquet pre-bake state, derived from the entries actually on disk.

    The index file is only a summary of the last generation run and may be missing or
    stale (an interrupted batch, entries copied in by hand). Counting the entries is
    cheap and is the thing a client actually depends on, so that is what is reported.
    """
    from backend import prebake

    out: dict[str, Any] = {}
    for data_dir in _data_dirs(run_dir, roles):
        for parquet in sorted(data_dir.glob("*.parquet")):
            entries = list(prebake.prebake_dir(parquet).rglob("*.json.gz"))
            if not entries:
                continue
            state: dict[str, Any] = {
                "entries": len(entries),
                "bytes": sum(entry.stat().st_size for entry in entries),
                "routes": sorted({entry.parent.name for entry in entries}),
            }
            index = prebake.read_index(parquet)
            if index:
                state["index"] = index
            # Keyed by the parquet's path within the run, which for a flat run is just
            # the file name.
            out[str(parquet.relative_to(run_dir)).replace("\\", "/")] = state
    return out


# ----------------------------------------------------------------------- manifest


def _collect(
    run_dir: Path, roles: Iterable[str], tier: str, *, include_future: bool
) -> list[dict[str, Any]]:
    """Files belonging to ``tier``, as manifest entries without checksums."""
    from backend import prebake

    if tier not in TIERS:
        raise ExportError(f"Unknown tier '{tier}'. Expected one of: {', '.join(TIERS)}")
    rank = TIERS.index(tier)
    picked: dict[str, Path] = {}

    def take(path: Path) -> None:
        if not path.is_file() or _is_excluded(path, run_dir):
            return
        picked[str(path.relative_to(run_dir))] = path

    for name in RUN_ROOT_FILES:
        take(run_dir / name)
    take(run_dir / "summary.json")

    # TLR results, in both the flat and the suite layout. Tier-independent: result.json
    # is the whole dataset the TLR viewer reads, and it is JSONL measured in MB, not the
    # GBs that make tiers worth having for perception runs. The globs cost nothing on a
    # perception run, whose scenarios sit one level deeper.
    for pattern in (f"*/{TLR_RESULT_NAME}", f"*/*/{TLR_RESULT_NAME}"):
        for path in run_dir.glob(pattern):
            take(path)
    if rank >= TIERS.index("raw"):
        # The analyzer's fallback source, only worth the gigabytes at the raw tier.
        for pattern in ("*/scene_result.pkl", "*/*/scene_result.pkl", "*/*.pkl.z", "*/*/*.pkl.z"):
            for path in run_dir.glob(pattern):
                take(path)

    for role_dir in _data_dirs(run_dir, roles):
        if not role_dir.is_dir():
            continue
        take(role_dir / "current.parquet")
        if include_future:
            take(role_dir / "future.parquet")
        take(role_dir / "summary.json")
        take(role_dir / "resources" / "summary.json")

        if rank >= TIERS.index("criteria"):
            for sidecar in SCENARIO_SIDECARS:
                for path in role_dir.glob(f"*/*/{sidecar}"):
                    take(path)
            for suite_dir in role_dir.glob("*/resources/summary.json"):
                take(suite_dir)
            wanted_routes = {prebake.ROUTE_DEVOPS_RESULT, prebake.ROUTE_FRAME_RESULTS}
            if rank >= TIERS.index("full"):
                wanted_routes.add(prebake.ROUTE_TN_OBJECTS)
            prebake_dir = role_dir / prebake.PREBAKE_DIRNAME
            if prebake_dir.is_dir():
                take(prebake_dir / prebake.INDEX_NAME)
                for route in sorted(wanted_routes):
                    for path in (prebake_dir / route).glob("*.json.gz"):
                        take(path)

        if rank >= TIERS.index("raw"):
            for path in role_dir.glob("*/*/scene_result.pkl"):
                take(path)

    return [
        _file_entry(path, run_dir, with_sha=False)
        for _, path in sorted(picked.items(), key=lambda kv: kv[0])
    ]


def export_manifest(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Describe the files a client should fetch for one run at one tier.

    ``checksums`` is opt-in: hashing a 465 MB parquet costs real IO, and the client
    only needs it to verify or to resume, not to plan.
    """
    require_auth(handler)
    run_dir = _resolve_run(payload.get("run"))
    tier = _as_text(payload.get("tier")) or "criteria"
    include_future = payload.get("include_future", False) is True
    with_sha = payload.get("checksums", False) is True

    role_raw = payload.get("role")
    available = _roles_present(run_dir)
    if role_raw in (None, "", "all"):
        roles = available
    else:
        requested = [role_raw] if isinstance(role_raw, str) else list(role_raw)
        roles = [r for r in requested if r in available]
        missing = [r for r in requested if r not in available]
        if missing:
            raise ExportError(
                f"Run '{run_dir.name}' has no role(s): {', '.join(missing)}. Available: {', '.join(available) or 'none'}"
            )

    files = _collect(run_dir, roles, tier, include_future=include_future)
    if with_sha:
        files = [_file_entry(_resolve_in_run(run_dir, f["rel_path"]), run_dir, with_sha=True) for f in files]
    tlr_scenarios = 0 if roles or _is_flat_run(run_dir) else _tlr_scenarios(run_dir)
    return {
        "run": run_dir.name,
        "kind": "tlr" if tlr_scenarios else "perception",
        "tlr_scenarios": tlr_scenarios,
        "roles": roles,
        "tier": tier,
        "include_future": include_future,
        "checksums": with_sha,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(f["size"]) for f in files),
    }


# ------------------------------------------------------------------------ prebake


def export_prebake(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Generate pre-baked DevOps answers for one parquet, or report existing state.

    Synchronous and slow -- it unpickles every scenario's ``scene_result.pkl``, which
    for a large run is tens of minutes. Prefer the server-side CLI
    (``python3 -m backend.prebake_cli``) for a whole run; this route exists so a
    client can top up a single parquet it is about to pull.
    """
    require_auth(handler)
    from backend import prebake

    run_dir = _resolve_run(payload.get("run"))
    rel = _as_text(payload.get("parquet"))
    if not rel:
        raise ExportError("parquet is required (e.g. 'devops/current.parquet')")
    parquet_path = _resolve_in_run(run_dir, rel)
    if parquet_path.suffix != ".parquet":
        raise ExportError(f"Not a parquet file: {rel}")

    if payload.get("status_only") is True:
        return {
            "run": run_dir.name,
            "parquet": rel,
            "index": prebake.read_index(parquet_path),
            "scenarios": len(prebake.list_scenarios(parquet_path)),
        }

    routes_raw = payload.get("routes")
    routes = list(prebake.PREBAKE_ROUTES)
    if isinstance(routes_raw, list) and routes_raw:
        routes = [r for r in routes_raw if r in prebake.PREBAKE_ROUTES]
        if not routes:
            raise ExportError(f"No known routes in {routes_raw}")
    stats = prebake.generate(parquet_path, routes=routes, force=payload.get("force") is True)
    return {"run": run_dir.name, "parquet": rel, **stats}


# --------------------------------------------------------------------- file stream


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Parse a single-range ``bytes=`` header into inclusive offsets."""
    text = _as_text(header)
    if not text.lower().startswith("bytes=") or "," in text:
        return None
    spec = text[len("bytes=") :].strip()
    start_text, _, end_text = spec.partition("-")
    try:
        if not start_text:
            length = int(end_text)
            if length <= 0:
                return None
            return max(size - length, 0), size - 1
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or end < start:
        return None
    return start, min(end, size - 1)


def export_file(handler: Any, payload: dict[str, Any], *, head_only: bool = False) -> None:
    """Stream one file from a run, honouring ``Range`` so large pulls can resume."""
    who = require_auth(handler)
    run_dir = _resolve_run(payload.get("run"))
    path = _resolve_in_run(run_dir, payload.get("rel_path"))
    # Exports leave the machine, so record who took what. Enabled with the same switch
    # as the rest of this server's access logging.
    if os.environ.get("LOCAL_BBOX_API_DEBUG") == "1":
        logging.getLogger(__name__).info(
            "export_file %s/%s by %s (%s)", run_dir.name, payload.get("rel_path"),
            who.get("actor"), who.get("via"),
        )
    stat = path.stat()
    size = int(stat.st_size)
    # Cheap, stable validator: content is immutable once written, and rehashing a
    # 465 MB parquet on every request would dominate the transfer itself.
    etag = '"{}"'.format(
        hashlib.sha256(
            f"{path.relative_to(run_dir)}:{size}:{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:32]
    )

    if _as_text(handler.headers.get("if-none-match")) == etag:
        handler.send_response(304)
        handler.send_header("ETag", etag)
        handler.end_headers()
        return

    rng = _parse_range(handler.headers.get("range") or "", size) if size else None
    if rng is None:
        start, end = 0, max(size - 1, 0)
        status = 200
    else:
        start, end = rng
        status = 206
    length = 0 if size == 0 else end - start + 1

    handler.send_response(status)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(length))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("ETag", etag)
    handler.send_header("X-Export-Size", str(size))
    handler.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
    if status == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    # Past this point the response is committed, so the dispatcher must not try to
    # replace it with a JSON error.
    handler._export_stream_started = True
    handler.end_headers()
    if head_only or length <= 0:
        return

    remaining = length
    with path.open("rb") as source:
        source.seek(start)
        while remaining > 0:
            chunk = source.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def export_health(handler: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Advertise the API and the effective access policy, without authorizing.

    Clients use this to decide whether to ask the user for a token at all, so it must
    answer even to a caller that would be refused everywhere else.
    """
    origin, email = _identity(_headers_dict(handler))
    try:
        require_auth(handler)
        authorized, reason = True, ""
    except ExportAuthError as exc:
        authorized, reason = False, str(exc)
    return {
        "ok": True,
        "service": "eval_dashboard_export",
        "enabled": exports_enabled(),
        "authorized": authorized,
        "auth_reason": reason,
        "token_required": require_token_always(),
        "token_configured": bool(export_token()),
        "direct_allowed": allow_direct(),
        "origin": origin,
        "identity": email,
        "tiers": TIER_DESCRIPTIONS,
        "data_root": str(_data_root()),
    }


JSON_ROUTES: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "/api/export_health": export_health,
    "/api/runs": runs,
    "/api/export_manifest": export_manifest,
    "/api/export_prebake": export_prebake,
}

STREAM_ROUTES: dict[str, Callable[..., None]] = {
    "/api/export_file": export_file,
}
