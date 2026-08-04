#!/usr/bin/env python3
"""evalctl -- drive the evaluation dashboard's workflow/export API from a terminal.

One command per job: preflight the server, start evaluator or release-specsheet
workflows, watch tasks, pull trend history, and fetch analysis packages. Built for
humans and for coding agents (Claude Code / Codex skills wrap these commands), so
every command also has ``--json`` for machine-readable output.

Configuration comes from the environment:
  EVAL_DASHBOARD_URL   e.g. http://eval-server:8502   (or pass --url)
  EVAL_EXPORT_TOKEN    bearer token, if the server demands one

stdlib only: this file must run anywhere Python 3.10+ exists, with no venv.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))
TERMINAL_STATUSES = {"completed", "failed"}
WATCH_INTERVAL_SEC = 20


class ApiError(RuntimeError):
    """The server refused or the request never got through."""


# ------------------------------------------------------------------------ transport


def _base_url(args: argparse.Namespace) -> str:
    url = (getattr(args, "url", "") or os.environ.get("EVAL_DASHBOARD_URL", "")).strip()
    if not url:
        raise ApiError(
            "No server configured. Set EVAL_DASHBOARD_URL (e.g. http://host:8502) or pass --url."
        )
    return url.rstrip("/")


def api(args: argparse.Namespace, path: str, payload: dict[str, Any] | None = None) -> Any:
    """POST JSON to one route; returns parsed JSON or raw bytes for streams."""
    url = _base_url(args) + path
    body = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    token = os.environ.get("EVAL_EXPORT_TOKEN", "").strip()
    if token:
        request.add_header("x-export-token", token)
    try:
        with urllib.request.urlopen(request, timeout=float(args.timeout)) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except Exception:
            pass
        raise ApiError(f"{path} -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"Could not reach {url}: {exc.reason}") from exc
    if "application/json" not in content_type:
        return raw
    data = json.loads(raw.decode("utf-8"))
    if isinstance(data, dict) and data.get("error"):
        raise ApiError(f"{path}: {data['error']}")
    return data


def _print(data: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


# ------------------------------------------------------------------------- preflight


def preflight(args: argparse.Namespace, *, need_queue: bool = True) -> dict[str, Any]:
    """Refuse early with the server's own reason instead of failing mid-request."""
    health = api(args, "/api/workflow_health")
    if not health.get("authorized"):
        raise ApiError(f"Not authorized: {health.get('auth_reason') or 'unknown reason'}")
    if need_queue:
        if not health.get("queue_enabled"):
            raise ApiError(f"Server cannot queue workflows: {health.get('queue_reason')}")
        if health.get("workers_alive") == 0 and not getattr(args, "force", False):
            raise ApiError(
                f"{health.get('workers_reason')} Pass --force to queue anyway."
            )
    return health


def cmd_doctor(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {}
    try:
        workflow = api(args, "/api/workflow_health")
    except ApiError as exc:
        print(f"UNREACHABLE  {exc}")
        return 2
    report["workflow"] = workflow
    try:
        report["export"] = api(args, "/api/export_health")
    except ApiError as exc:
        report["export"] = {"error": str(exc)}
    if args.json:
        _print(report, as_json=True)
        return 0

    def status(ok: bool, label: str, reason: str = "") -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f" -- {reason}" if reason and not ok else ""))

    status(True, f"server reachable at {_base_url(args)}")
    status(bool(workflow.get("authorized")), "authorized", workflow.get("auth_reason", ""))
    status(bool(workflow.get("queue_enabled")), "task queue", workflow.get("queue_reason", ""))
    workers = workflow.get("workers_alive")
    if workers is None:
        print(f"?    workers -- {workflow.get('workers_reason') or 'could not tell'}")
    else:
        status(workers > 0, f"workers alive: {workers}", workflow.get("workers_reason", ""))
    export = report["export"]
    if isinstance(export, dict) and not export.get("error"):
        status(bool(export.get("authorized")), "export/download access",
               export.get("auth_reason", ""))
        identity = export.get("identity") or export.get("origin") or "anonymous"
        print(f"     identity: {identity}   data root: {export.get('data_root', '?')}")
    kinds = ", ".join(k["name"] for k in workflow.get("kinds", []))
    print(f"     workflow kinds: {kinds}")
    bad = not workflow.get("authorized") or not workflow.get("queue_enabled") or workers == 0
    return 1 if bad else 0


# ------------------------------------------------------------------------- workflows


def _resolve_catalog(args: argparse.Namespace) -> tuple[str, str]:
    """Accept a catalog id or display name; names resolve through the server presets."""
    catalog = (args.catalog or "").strip()
    integration = (getattr(args, "integration_id", "") or "").strip()
    if catalog and integration:
        return catalog, integration
    presets = api(args, "/api/workflow_catalogs").get("presets", [])
    if not catalog:
        if not presets:
            raise ApiError("No catalog given and the server has no presets; pass --catalog.")
        chosen = presets[0]
    else:
        matches = [
            p for p in presets
            if p.get("catalog_id") == catalog
            or catalog.lower() in str(p.get("display_name", "")).lower()
        ]
        if not matches:
            # An id the presets don't know is still usable if an integration came with it.
            if integration:
                return catalog, integration
            names = ", ".join(str(p.get("display_name")) for p in presets)
            raise ApiError(f"Catalog '{catalog}' matches no preset. Known: {names}")
        chosen = matches[0]
    return str(chosen.get("catalog_id", "")), integration or str(chosen.get("integration_id", ""))


def _watch(args: argparse.Namespace, task_id: str) -> int:
    last_line = ""
    while True:
        task = api(args, "/api/workflow_task", {"task_id": task_id, "log": False})["task"]
        status = task.get("status", "?")
        pct = task.get("progress_pct")
        line = f"{status}  {task.get('progress_message') or ''}" + (
            f"  [{pct}%]" if pct is not None else ""
        )
        if line != last_line:
            print(f"{datetime.now(JST).strftime('%H:%M:%S')}  {line}")
            last_line = line
        if status in TERMINAL_STATUSES:
            summary = task.get("result_summary") or {}
            if summary:
                print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
            if status == "failed":
                print(f"error: {task.get('error_message', '')}", file=sys.stderr)
            return 0 if status == "completed" else 1
        time.sleep(WATCH_INTERVAL_SEC)


def _after_start(args: argparse.Namespace, result: dict[str, Any]) -> int:
    if args.json:
        _print(result, as_json=True)
    elif result.get("dry_run"):
        print("dry run only -- nothing queued. Parameters the worker would get:")
        _print(result, as_json=True)
    else:
        check = result.get("target_check") or {}
        if check.get("checked"):
            print(f"target verified: {check.get('ref')}")
        elif check.get("detail"):
            print(f"target not verified ({check['detail']})")
        print(f"queued {result.get('kind')} workflow: task {result.get('task_id')}")
        print(f"run name: {result.get('run_name')}   output: {result.get('output_path')}")
    if result.get("task_id") and args.watch:
        return _watch(args, result["task_id"])
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    if not args.dry_run:
        preflight(args)
    catalog_id, integration_id = _resolve_catalog(args)
    payload: dict[str, Any] = {
        "kind": args.kind,
        "project_id": args.project,
        "target_name": args.target,
        "catalog_id": catalog_id,
        "integration_id": integration_id,
        "is_tag": args.tag,
        "dry_run": args.dry_run,
        "check_target": not args.no_check_target,
    }
    if args.description:
        payload["description"] = args.description
    if args.suite:
        payload["suite_ids"] = args.suite
    if args.output_path:
        payload["output_path"] = args.output_path
    return _after_start(args, api(args, "/api/workflow_start", payload))


_VERSION_IN_TARGET = re.compile(r"v?(\d+\.\d+(?:\.\d+)*)")


def _auto_release_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Fill the trend metadata the way past releases wrote it, unless overridden.

    date is always today (JST) in the dotted format history uses; version derives from
    the target name; everything else falls back to the newest release with the same
    topic so conventions (release_group naming, data_count) carry forward.
    """
    previous: dict[str, Any] = {}
    try:
        items = api(args, "/api/workflow_trends", {"limit": 5}).get("items", [])
        for item in items:
            if not args.topic or item.get("topic") == args.topic:
                jobs = item.get("jobs") or {}
                first = next(iter(jobs.values()), {})
                previous = first.get("metadata") or {}
                break
    except ApiError:
        pass  # a server without trend history can still release

    match = _VERSION_IN_TARGET.search(args.target)
    version = f"Pilot.Auto v{match.group(1)}" if match else args.target
    metadata: dict[str, Any] = {
        "tags": ["trend"],
        "release_group": args.release_group or str(previous.get("release_group") or version),
        "pilot_auto_version": args.version or version,
        "data_count": args.data_count or str(previous.get("data_count") or "0"),
        "date": args.date or datetime.now(JST).strftime("%Y.%m.%d"),
        "description": args.description or f"Release run for {args.target}",
    }
    topic = args.topic or str(previous.get("topic_name") or "")
    if topic:
        metadata["topic_name"] = topic
    for pair in args.set or []:
        key, _, value = pair.partition("=")
        if not _:
            raise ApiError(f"--set expects key=value, got '{pair}'")
        metadata[key.strip()] = value.strip()
    return metadata


def cmd_release(args: argparse.Namespace) -> int:
    if not args.dry_run:
        preflight(args)
    if args.metadata_file:
        metadata_text = Path(args.metadata_file).read_text(encoding="utf-8")
        payload_metadata: dict[str, Any] = {"metadata_text": metadata_text}
        shown = metadata_text
    else:
        metadata = _auto_release_metadata(args)
        payload_metadata = {"trend_metadata": metadata}
        shown = json.dumps(metadata, ensure_ascii=False, indent=2)
    if not args.json:
        print("trend metadata for this release:")
        print(shown)
    if not args.dry_run and not args.yes:
        answer = input("Start the release workflow with this metadata? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return 1
    payload: dict[str, Any] = {
        "kind": "release",
        "project_id": args.project,
        "target_name": args.target,
        "is_tag": args.tag,
        "dry_run": args.dry_run,
        "check_target": not args.no_check_target,
        **payload_metadata,
    }
    if args.performance_job_id:
        payload["performance_job_id"] = args.performance_job_id
    if args.devops_job_id:
        payload["devops_job_id"] = args.devops_job_id
    if args.run_eval:
        payload["run_eval"] = True
    return _after_start(args, api(args, "/api/workflow_start", payload))


def cmd_status(args: argparse.Namespace) -> int:
    if args.task_id:
        if args.watch:
            return _watch(args, args.task_id)
        task = api(args, "/api/workflow_task", {"task_id": args.task_id, "log": args.log})["task"]
        if args.json:
            _print(task, as_json=True)
            return 0
        for key in ("id", "type", "status", "target_name", "run_name", "requested_by",
                    "progress_message", "error_message", "created_at", "updated_at"):
            if task.get(key):
                print(f"{key:18} {task[key]}")
        if task.get("result_summary"):
            print(json.dumps(task["result_summary"], ensure_ascii=False, indent=2, default=str))
        if args.log and task.get("log"):
            print("--- log tail ---")
            print(task["log"][-int(args.log_chars):])
        return 0 if task.get("status") != "failed" else 1
    result = api(args, "/api/workflow_tasks", {"limit": args.limit, "mine": args.mine or ""})
    if args.json:
        _print(result, as_json=True)
        return 0
    for item in result.get("items", []):
        pct = f"{item['progress_pct']}%" if item.get("progress_pct") is not None else ""
        print(f"{item['id'][:12]}  {item['status']:9} {pct:5} {item.get('target_name', ''):24} "
              f"{item.get('run_name', '')}  ({item.get('requested_by', '')})")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    result = api(args, "/api/workflow_cancel", {"task_id": args.task_id})
    _print(result, as_json=args.json)
    if not args.json:
        print(result.get("message", ""))
    return 0 if result.get("ok") else 1


def cmd_trends(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {"limit": args.limit}
    if args.topic:
        payload["topic"] = args.topic
    if args.q:
        payload["q"] = args.q
    if args.summaries:
        payload["include_summary"] = True
    if args.cases:
        payload["include_cases"] = True
    result = api(args, "/api/workflow_trends", payload)
    if args.json or args.summaries or args.cases:
        _print(result, as_json=True)
        return 0
    for item in result.get("items", []):
        metrics = item.get("metrics") or {}
        parts = [f"{item.get('date', '?'):10}", f"{item.get('version', ''):22}",
                 f"{item.get('topic', ''):14}"]
        for key in ("mAP", "overall_pass_rate"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value:.3f}" if key == "mAP" else f"{key}={value:.1f}%")
        parts.append(f"roles={','.join(item.get('roles', []))}")
        print("  ".join(parts))
    return 0


# ------------------------------------------------------------------------- downloads


def cmd_runs(args: argparse.Namespace) -> int:
    result = api(args, "/api/runs", {"q": args.q or "", "sizes": not args.no_sizes})
    if args.json:
        _print(result, as_json=True)
        return 0
    for item in result.get("items", []):
        size = item.get("total_bytes")
        size_text = f"{size / 1e9:.2f} GB" if isinstance(size, (int, float)) else ""
        print(f"{item['name']:48} roles={','.join(item.get('roles', []))}  {size_text}")
    return 0


def _download_file(args: argparse.Namespace, run: str, rel_path: str, dest: Path,
                   expected_bytes: int | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected_bytes is not None and dest.exists() and dest.stat().st_size == expected_bytes:
        print(f"  kept    {rel_path}")
        return
    data = api(args, "/api/export_file", {"run": run, "rel_path": rel_path})
    dest.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
    print(f"  fetched {rel_path} ({dest.stat().st_size / 1e6:.1f} MB)")


def cmd_fetch(args: argparse.Namespace) -> int:
    manifest = api(args, "/api/export_manifest",
                   {"run": args.run, "tier": args.tier, "role": args.role or "all"})
    dest_root = Path(args.dest or ".") / manifest["run"]
    files = manifest.get("files", [])
    print(f"{manifest['run']}: {len(files)} file(s), tier={args.tier}")
    for entry in files:
        rel = entry["rel_path"]
        _download_file(args, manifest["run"], rel, dest_root / rel, entry.get("bytes"))
    print(f"saved under {dest_root}")
    return 0


def _save_analysis_package(args: argparse.Namespace, payload: dict[str, Any], name: str) -> int:
    data = api(args, "/api/analysis_package", payload)
    if not isinstance(data, bytes):
        raise ApiError(f"Expected a ZIP stream, got: {str(data)[:200]}")
    dest = Path(args.dest or ".") / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "analysis_package.zip").write_bytes(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(dest)
    print(f"analysis package extracted to {dest}")
    print(f"start with: {dest / 'llm_instructions.md'} and {dest / 'analysis_data_brief.md'}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    payload = {"mode": "single", "run": args.run, "role": args.role,
               "exclude_polygons": args.exclude_polygons}
    return _save_analysis_package(args, payload, f"analysis_{args.run}")


def cmd_compare(args: argparse.Namespace) -> int:
    payload = {"mode": "compare", "base_run": args.base, "candidate_run": args.candidate,
               "role": args.role, "exclude_polygons": args.exclude_polygons}
    return _save_analysis_package(args, payload, f"compare_{args.base}_vs_{args.candidate}")


# ------------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalctl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", help="server base URL (default: $EVAL_DASHBOARD_URL)")
    parser.add_argument("--timeout", default=120, type=float, help="request timeout seconds")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check connectivity, auth, queue and workers").set_defaults(
        func=cmd_doctor
    )

    start = sub.add_parser("start", help="start an evaluator workflow for a branch/tag")
    start.add_argument("target", help="git branch (or tag with --tag), e.g. beta/v4.3.2")
    start.add_argument("--kind", choices=["perception", "tlr"], default="perception")
    start.add_argument("--project", default="x2_dev")
    start.add_argument("--catalog", help="catalog id or preset display name (default: first preset)")
    start.add_argument("--integration-id", default="")
    start.add_argument("--suite", action="append", help="suite id (repeatable)")
    start.add_argument("--description", default="")
    start.add_argument("--output-path", default="")
    start.add_argument("--tag", action="store_true", help="target is a git tag")
    start.add_argument("--dry-run", action="store_true")
    start.add_argument("--no-check-target", action="store_true")
    start.add_argument("--watch", action="store_true", help="poll until the task finishes")
    start.add_argument("--force", action="store_true", help="queue even with no live worker")
    start.set_defaults(func=cmd_start)

    release = sub.add_parser("release", help="start the release spec-sheet workflow")
    release.add_argument("target", help="git branch (or tag with --tag)")
    release.add_argument("--project", default="x2_dev")
    release.add_argument("--metadata-file", help="YAML file; skips all auto-fill")
    release.add_argument("--version", default="", help="pilot_auto_version override")
    release.add_argument("--release-group", default="")
    release.add_argument("--data-count", default="")
    release.add_argument("--date", default="", help="override; default is today JST as YYYY.MM.DD")
    release.add_argument("--description", default="")
    release.add_argument("--topic", default="", help="topic_name (default: newest past release's)")
    release.add_argument("--set", action="append", metavar="KEY=VALUE",
                         help="extra trend metadata field (repeatable)")
    release.add_argument("--performance-job-id", default="", help="reuse an existing job")
    release.add_argument("--devops-job-id", default="", help="reuse an existing job")
    release.add_argument("--run-eval", action="store_true")
    release.add_argument("--tag", action="store_true")
    release.add_argument("--dry-run", action="store_true")
    release.add_argument("--no-check-target", action="store_true")
    release.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    release.add_argument("--watch", action="store_true")
    release.add_argument("--force", action="store_true")
    release.set_defaults(func=cmd_release)

    status = sub.add_parser("status", help="list recent tasks, or inspect/watch one")
    status.add_argument("task_id", nargs="?", default="")
    status.add_argument("--limit", type=int, default=15)
    status.add_argument("--mine", default="", help="filter by requester email")
    status.add_argument("--log", action="store_true", help="include the log tail")
    status.add_argument("--log-chars", type=int, default=4000)
    status.add_argument("--watch", action="store_true")
    status.set_defaults(func=cmd_status)

    cancel = sub.add_parser("cancel", help="cancel a queued or running task")
    cancel.add_argument("task_id")
    cancel.set_defaults(func=cmd_cancel)

    trends = sub.add_parser("trends", help="past release trend data (metadata + metrics)")
    trends.add_argument("--topic", default="")
    trends.add_argument("--q", default="", help="substring filter")
    trends.add_argument("--limit", type=int, default=20)
    trends.add_argument("--summaries", action="store_true", help="include raw summary JSON")
    trends.add_argument("--cases", action="store_true", help="include devops case rows")
    trends.set_defaults(func=cmd_trends)

    runs = sub.add_parser("runs", help="list runs available on the server")
    runs.add_argument("--q", default="")
    runs.add_argument("--no-sizes", action="store_true")
    runs.set_defaults(func=cmd_runs)

    fetch = sub.add_parser("fetch", help="download a run's files")
    fetch.add_argument("run")
    fetch.add_argument("--tier", default="criteria")
    fetch.add_argument("--role", default="")
    fetch.add_argument("--dest", default="")
    fetch.set_defaults(func=cmd_fetch)

    analyze = sub.add_parser("analyze", help="fetch the LLM analysis package for one run")
    analyze.add_argument("run")
    analyze.add_argument("--role", default="performance")
    analyze.add_argument("--exclude-polygons", action="store_true")
    analyze.add_argument("--dest", default="")
    analyze.set_defaults(func=cmd_analyze)

    compare = sub.add_parser("compare", help="fetch the base-vs-candidate analysis package")
    compare.add_argument("base")
    compare.add_argument("candidate")
    compare.add_argument("--role", default="performance")
    compare.add_argument("--exclude-polygons", action="store_true")
    compare.add_argument("--dest", default="")
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ApiError as exc:
        print(f"evalctl: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
