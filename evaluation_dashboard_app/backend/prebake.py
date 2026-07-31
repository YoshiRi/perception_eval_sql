"""Pre-computed responses for the pickle-backed DevOps routes.

Three routes -- ``scenario_devops_result``, ``scenario_devops_frame_results`` and
``scenario_devops_tn_objects`` -- answer by unpickling ``scene_result.pkl``. That is
fine on the server, where the evaluator libraries are installed and the pickles are
already on disk, but it is hostile to a local client:

* the pickles are 27-133 MB each and dominate a run's on-disk size (a 11 GB run is
  ~95% pickles), so shipping them defeats the point of a selective download;
* reading them needs ``driving_log_replayer_v2`` and ``autoware_perception_evaluation``
  importable, which is a heavy install for a viewer;
* ``pickle.load`` executes arbitrary code, which is a very different risk once the
  file has travelled over a network rather than being produced locally.

So the server runs those same handlers ahead of time and stores their JSON output
next to the parquet. The client serves the stored answers, gets byte-identical
results without the pickles, and needs neither evaluator library. On a cache miss the
handler falls through to its normal pickle path, which already degrades to
``available: false`` with a reason when the pickle or the libraries are absent.

Layout, anchored on the parquet's own directory so a pre-baked run stays
self-contained when only that directory is copied::

    <parquet_dir>/.export_prebake/index.json
    <parquet_dir>/.export_prebake/<route>/<key>.json.gz
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

LOGGER = logging.getLogger(__name__)

PREBAKE_DIRNAME = ".export_prebake"
INDEX_NAME = "index.json"

# Bump when a key derivation or a stored payload shape changes, so stale files are
# ignored rather than silently served in the wrong shape.
PREBAKE_VERSION = 1

ROUTE_DEVOPS_RESULT = "scenario_devops_result"
ROUTE_FRAME_RESULTS = "scenario_devops_frame_results"
ROUTE_TN_OBJECTS = "scenario_devops_tn_objects"

PREBAKE_ROUTES = (ROUTE_DEVOPS_RESULT, ROUTE_FRAME_RESULTS, ROUTE_TN_OBJECTS)

# Generate at the handlers' own ceilings so any narrower client request can be served
# by trimming the stored answer. ``local_bbox_api`` caps max_rows at 600k and
# max_frames at 5000.
TN_OBJECT_MAX_ROWS = 600000
FRAME_RESULT_MAX_FRAMES = 5000

# Frame-window keys are deliberately excluded from the cache key: the stored answer
# always covers the whole scenario and :func:`_trim` narrows it per request.
_FRAME_WINDOW_KEYS = ("frame_index", "frame_min", "frame_max")


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _filters(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("filters")
    return raw if isinstance(raw, dict) else {}


def prebake_dir(parquet_path: Path) -> Path:
    return parquet_path.parent / PREBAKE_DIRNAME


def _digest(parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def cache_key(route: str, parquet_path: Path, payload: dict[str, Any]) -> str | None:
    """Canonical key for a request, or ``None`` when the route is not pre-bakeable.

    Only the fields a handler actually reads take part, so incidental payload noise
    (``timeout_ms``, the ``run`` label, the caller's ``max_rows``) does not fragment
    the cache.
    """
    filters = _filters(payload)
    scenario_name = _as_text(filters.get("scenario_name"))
    if not scenario_name:
        return None
    suite_name = _as_text(filters.get("suite_name"))
    base: dict[str, Any] = {
        "version": PREBAKE_VERSION,
        "route": route,
        "parquet": parquet_path.name,
        "suite_name": suite_name,
        "scenario_name": scenario_name,
    }
    if route in (ROUTE_FRAME_RESULTS, ROUTE_TN_OBJECTS):
        # These handlers read nothing from the parquet and ignore every filter except
        # the suite/scenario pair and the frame window.
        return _digest(base)
    if route == ROUTE_DEVOPS_RESULT:
        # This one *does* query the parquet with the full filter set, so every filter
        # that reaches the SQL has to be part of the key.
        semantic = {
            key: value
            for key, value in sorted(filters.items())
            if key not in _FRAME_WINDOW_KEYS and value not in (None, "", "Any", [])
        }
        base["filters"] = semantic
        base["exact"] = bool(payload.get("exact") is True or payload.get("use_pickle") is True)
        return _digest(base)
    return None


def _entry_path(parquet_path: Path, route: str, key: str) -> Path:
    return prebake_dir(parquet_path) / route / f"{key}.json.gz"


def read(route: str, parquet_path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a stored response narrowed to ``payload``, or ``None`` on a miss."""
    key = cache_key(route, parquet_path, payload)
    if not key:
        return None
    path = _entry_path(parquet_path, route, key)
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            stored = json.load(handle)
    except Exception as exc:  # pragma: no cover - corrupt cache must not break a route.
        LOGGER.warning("prebake read failed for %s: %s", path, exc)
        return None
    if not isinstance(stored, dict):
        return None
    result = _trim(route, stored, payload)
    result["prebaked"] = True
    return result


def write(route: str, parquet_path: Path, payload: dict[str, Any], result: dict[str, Any]) -> Path | None:
    key = cache_key(route, parquet_path, payload)
    if not key:
        return None
    path = _entry_path(parquet_path, route, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # mtime=0 keeps the bytes deterministic so the export manifest's sha256 is stable
    # across regenerations that produce identical content.
    with gzip.GzipFile(filename="", mode="wb", fileobj=tmp.open("wb"), mtime=0) as handle:
        handle.write(json.dumps(result, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8"))
    tmp.replace(path)
    return path


def _frame_window(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    filters = _filters(payload)
    exact_raw = payload.get("frame_index", filters.get("frame_index"))
    if exact_raw not in (None, ""):
        exact = int(float(exact_raw))
        return exact, exact
    low_raw = filters.get("frame_min")
    high_raw = filters.get("frame_max")
    low = None if low_raw in (None, "") else int(float(low_raw))
    high = None if high_raw in (None, "") else int(float(high_raw))
    return low, high


def _trim(route: str, stored: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Narrow a whole-scenario stored answer to what this request asked for."""
    if route == ROUTE_DEVOPS_RESULT:
        return dict(stored)

    if not stored.get("available"):
        # An unavailable answer has no frame list to narrow, and the frame/row
        # bookkeeping below would add keys the live handler never returns.
        return dict(stored)

    result = dict(stored)
    frames_in = result.get("frames") or []
    low, high = _frame_window(payload)
    if low is not None or high is not None:
        frames_in = [
            frame
            for frame in frames_in
            if (low is None or int(float(frame.get("frame") or 0)) >= low)
            and (high is None or int(float(frame.get("frame") or 0)) <= high)
        ]

    if route == ROUTE_TN_OBJECTS:
        # The handler stamps every box with the caller's run label, so re-stamp rather
        # than returning whichever label the generator happened to use.
        run_label = _as_text(payload.get("run")) or "A"
        max_rows = min(max(int(payload.get("max_rows") or 120000), 1), TN_OBJECT_MAX_ROWS)
        kept: list[dict[str, Any]] = []
        row_count = 0
        for frame in frames_in:
            boxes = []
            for box in frame.get("boxes") or []:
                if row_count >= max_rows:
                    break
                boxes.append({**box, "run": run_label})
                row_count += 1
            if boxes:
                kept.append({"frame": frame.get("frame"), "boxes": boxes})
            if row_count >= max_rows:
                break
        result["frames"] = kept
        result["row_count"] = row_count
        result["frame_count"] = len(kept)
        result["truncated"] = bool(stored.get("truncated")) or row_count >= max_rows
        return result

    max_frames = min(max(int(payload.get("max_frames") or 600), 1), FRAME_RESULT_MAX_FRAMES)
    truncated = bool(stored.get("truncated")) or len(frames_in) > max_frames
    frames_out = frames_in[:max_frames]
    result["frames"] = frames_out
    result["frame_count"] = len(frames_out)
    result["truncated"] = truncated
    return result


def _generation_payloads(
    parquet_path: Path, suite_name: str, scenario_name: str, topic_name: str
) -> list[tuple[str, dict[str, Any]]]:
    """The payloads to pre-bake for one scenario, matching what the UI issues."""
    filters: dict[str, Any] = {"suite_name": suite_name, "scenario_name": scenario_name}
    if topic_name:
        filters["topic_name"] = topic_name
    path_text = str(parquet_path)
    out: list[tuple[str, dict[str, Any]]] = [
        # The explorer requests this both with and without a topic filter.
        (ROUTE_DEVOPS_RESULT, {"path": path_text, "filters": dict(filters)}),
        (
            ROUTE_FRAME_RESULTS,
            {"path": path_text, "filters": dict(filters), "max_frames": FRAME_RESULT_MAX_FRAMES},
        ),
        (
            ROUTE_TN_OBJECTS,
            {"path": path_text, "filters": dict(filters), "max_rows": TN_OBJECT_MAX_ROWS, "run": "A"},
        ),
    ]
    if topic_name:
        bare = {"suite_name": suite_name, "scenario_name": scenario_name}
        out.append((ROUTE_DEVOPS_RESULT, {"path": path_text, "filters": bare}))
    return out


def list_scenarios(parquet_path: Path, limit: int = 5000) -> list[dict[str, str]]:
    from backend import local_bbox_api as api

    result = api.scenarios({"path": str(parquet_path), "limit": limit})
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in result.get("items") or []:
        key = (
            _as_text(item.get("suite_name")),
            _as_text(item.get("scenario_name")),
            _as_text(item.get("topic_name")),
        )
        if not key[1] or key in seen:
            continue
        seen.add(key)
        out.append({"suite_name": key[0], "scenario_name": key[1], "topic_name": key[2]})
    return out


def coverage(parquet_path: Path, *, routes: Iterable[str] = PREBAKE_ROUTES) -> dict[str, Any]:
    """How much of a parquet is already pre-baked, without generating anything.

    Cheap: it derives the keys and stats the files, so it can be run against a whole
    data root to decide what still needs work.
    """
    wanted = {route for route in routes if route in PREBAKE_ROUTES}
    scenario_list = list_scenarios(parquet_path)
    covered = 0
    missing: list[str] = []
    for scenario in scenario_list:
        payloads = _generation_payloads(
            parquet_path, scenario["suite_name"], scenario["scenario_name"], scenario["topic_name"]
        )
        # A scenario counts as covered when the gate verdict exists; the frame and
        # true-negative entries are legitimately absent when a scenario has no pickle.
        wants_result = [p for route, p in payloads if route == ROUTE_DEVOPS_RESULT]
        if ROUTE_DEVOPS_RESULT not in wanted or not wants_result:
            continue
        key = cache_key(ROUTE_DEVOPS_RESULT, parquet_path, wants_result[0])
        if key and _entry_path(parquet_path, ROUTE_DEVOPS_RESULT, key).is_file():
            covered += 1
        else:
            missing.append(scenario["scenario_name"])
    entries = list(prebake_dir(parquet_path).rglob("*.json.gz"))
    return {
        "parquet": str(parquet_path),
        "scenarios": len(scenario_list),
        "covered": covered,
        "missing": missing,
        "entries": len(entries),
        "bytes": sum(entry.stat().st_size for entry in entries),
        "complete": covered == len(scenario_list) and len(scenario_list) > 0,
    }


def generate(
    parquet_path: Path,
    *,
    routes: Iterable[str] = PREBAKE_ROUTES,
    force: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    time_budget_sec: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Pre-bake every scenario of one parquet. Server-side; needs the evaluator libs.

    Individual failures are recorded and skipped rather than aborting the run: a
    single unreadable pickle among a hundred scenarios should not lose the other
    ninety-nine.

    ``time_budget_sec`` and ``should_stop`` allow a long batch to be time-boxed or
    interrupted between scenarios. Stopping is always safe: entries are written one at
    a time and a later run skips whatever already exists, so the work resumes rather
    than restarting.
    """
    from backend import local_bbox_api as api

    handlers = {
        ROUTE_DEVOPS_RESULT: api.scenario_devops_result,
        ROUTE_FRAME_RESULTS: api.scenario_devops_frame_results,
        ROUTE_TN_OBJECTS: api.scenario_devops_tn_objects,
    }
    wanted = [route for route in routes if route in handlers]
    scenario_list = list_scenarios(parquet_path)
    started = time.monotonic()
    stats: dict[str, Any] = {
        "parquet": str(parquet_path),
        "scenarios": len(scenario_list),
        "written": 0,
        "skipped": 0,
        "failed": 0,
        "bytes": 0,
        "errors": [],
        "routes": wanted,
        "stopped_early": False,
        "scenarios_done": 0,
    }

    for index, scenario in enumerate(scenario_list):
        if should_stop is not None and should_stop():
            stats["stopped_early"] = True
            stats["stop_reason"] = "cancelled"
            break
        if time_budget_sec is not None and time.monotonic() - started >= time_budget_sec:
            stats["stopped_early"] = True
            stats["stop_reason"] = f"time budget of {time_budget_sec:.0f}s reached"
            break
        scenario_started = time.monotonic()
        for route, payload in _generation_payloads(
            parquet_path, scenario["suite_name"], scenario["scenario_name"], scenario["topic_name"]
        ):
            if route not in wanted:
                continue
            key = cache_key(route, parquet_path, payload)
            if not key:
                continue
            target = _entry_path(parquet_path, route, key)
            if target.is_file() and not force:
                stats["skipped"] += 1
                continue
            try:
                result = handlers[route](payload)
            except Exception as exc:
                stats["failed"] += 1
                stats["errors"].append(
                    {"route": route, "scenario": scenario["scenario_name"], "error": str(exc)}
                )
                LOGGER.warning(
                    "prebake failed route=%s scenario=%s: %s", route, scenario["scenario_name"], exc
                )
                continue
            if route in (ROUTE_FRAME_RESULTS, ROUTE_TN_OBJECTS) and not result.get("available"):
                # Storing "unavailable" would make the client's own fallback unreachable
                # and would freeze in a verdict that a later pickle could change. Leave
                # the miss so the route degrades exactly as it does on the server.
                stats["skipped"] += 1
                continue
            written = write(route, parquet_path, payload, result)
            if written is not None:
                stats["written"] += 1
                stats["bytes"] += written.stat().st_size
        stats["scenarios_done"] = index + 1
        if progress is not None:
            elapsed = time.monotonic() - started
            done = index + 1
            remaining = len(scenario_list) - done
            progress(
                {
                    "index": done,
                    "total": len(scenario_list),
                    "scenario": scenario["scenario_name"],
                    "seconds": round(time.monotonic() - scenario_started, 1),
                    "elapsed": round(elapsed, 1),
                    # Mean rate is the honest estimate here: per-scenario cost swings by
                    # an order of magnitude with pickle size.
                    "eta_sec": round(remaining * (elapsed / done), 0) if done else None,
                    "written": stats["written"],
                    "failed": stats["failed"],
                }
            )

    stats["elapsed_sec"] = round(time.monotonic() - started, 1)
    _write_index(parquet_path, stats)
    return stats


def _write_index(parquet_path: Path, stats: dict[str, Any]) -> None:
    directory = prebake_dir(parquet_path)
    directory.mkdir(parents=True, exist_ok=True)
    index = {
        "version": PREBAKE_VERSION,
        "parquet": parquet_path.name,
        "scenarios": stats.get("scenarios"),
        "written": stats.get("written"),
        "skipped": stats.get("skipped"),
        "failed": stats.get("failed"),
        "routes": stats.get("routes"),
    }
    (directory / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def read_index(parquet_path: Path) -> dict[str, Any] | None:
    path = prebake_dir(parquet_path) / INDEX_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def enabled() -> bool:
    """Whether stored answers should be consulted before the pickle path.

    Defaults on: a present cache is authoritative, and a miss costs one ``is_file``
    check. ``EVAL_PREBAKE_READ=0`` forces the pickle path, which is what you want
    when regenerating or comparing against the source of truth.
    """
    return os.environ.get("EVAL_PREBAKE_READ", "1") != "0"
