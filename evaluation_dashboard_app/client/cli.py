"""``evaldash-local`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from client import config, serve, sync
from client.remote import AuthError, Remote, RemoteError, connect, probe_server

EPILOG = """\
typical use:
  evaldash-local login --server https://dash.example.com --token <token>
  evaldash-local runs
  evaldash-local pull eval_2.4a_0710_streampetr_ptv3_off --role devops --tier criteria
  evaldash-local open

The server needs EVAL_EXPORT_TOKEN set for the export routes to answer at all.
"""


def _remote(args: argparse.Namespace) -> Remote:
    cfg = config.Config.load()
    if getattr(args, "server", None):
        cfg.server_url = args.server
    if getattr(args, "token", None):
        cfg.token = args.token
    return connect(cfg)


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    if not rows:
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


# ---------------------------------------------------------------------- commands


def cmd_login(args: argparse.Namespace) -> int:
    cfg = config.Config.load()
    if args.server:
        cfg.server_url = args.server.rstrip("/")
    if args.token is not None:
        cfg.token = args.token
    if args.cf_client_id is not None:
        cfg.cf_client_id = args.cf_client_id
    if args.cf_client_secret is not None:
        cfg.cf_client_secret = args.cf_client_secret
    if args.t4_base_url is not None:
        cfg.t4_base_url = args.t4_base_url
    if args.insecure:
        cfg.verify_tls = False
    if not cfg.effective_server():
        print("error: --server is required the first time", file=sys.stderr)
        return 2
    cfg.server_url = cfg.server_url or cfg.effective_server()

    try:
        resolved, health = probe_server(cfg, cfg.server_url)
    except RemoteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    cfg.server_url = resolved
    path = cfg.save()

    print(f"server   {resolved}")
    print(f"exports  {'enabled' if health.get('enabled') else 'DISABLED (set EVAL_EXPORT_TOKEN there)'}")
    print(f"data     {health.get('data_root')}")
    print(f"config   {path}")
    if cfg.resolved_token():
        try:
            # Use the *resolved* base URL, not what the user typed: probe_server may have
            # appended /bbox-api, and re-applying the bare hostname here would send the
            # check to Streamlit instead of the API.
            count = len(Remote(cfg).runs(sizes=False).get("items") or [])
            print(f"auth     ok, {count} run(s) visible")
        except AuthError as exc:
            print(f"auth     FAILED: {exc}", file=sys.stderr)
            return 1
        except RemoteError as exc:
            # Connectivity wobble on the follow-up call does not invalidate the settings
            # that were just verified and saved.
            print(f"auth     could not be confirmed: {exc}", file=sys.stderr)
    else:
        # No token is the normal case now: the server authorizes on the dashboard's own
        # identity model, so only deployments that opt in will ask for one.
        try:
            count = len(Remote(cfg).runs(sizes=False).get("items") or [])
            print(f"access   ok without a token, {count} run(s) visible")
        except AuthError as exc:
            print(f"access   this server wants a token: {exc}", file=sys.stderr)
            return 1
        except RemoteError as exc:
            print(f"access   could not be confirmed: {exc}", file=sys.stderr)
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    remote = _remote(args)
    data = remote.runs(sizes=not args.fast, query=args.query or "")
    items = data.get("items") or []
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    if not items:
        print(f"No runs found under {data.get('root')}")
        return 0
    rows = []
    for item in items:
        tiers = item.get("tier_bytes") or {}
        rows.append(
            [
                item["name"],
                ",".join(item.get("roles") or []),
                sync.human_bytes(item.get("total_bytes") or 0) if not args.fast else "-",
                sync.human_bytes(tiers.get("minimal") or 0) if tiers else "-",
                sync.human_bytes(tiers.get("criteria") or 0) if tiers else "-",
                str(len(item.get("prebaked") or {})),
            ]
        )
    _print_table(rows, ["RUN", "ROLES", "TOTAL", "MINIMAL", "CRITERIA", "PREBAKED"])
    print(f"\n{len(items)} run(s) on {remote.base_url}")
    if not args.fast:
        print("MINIMAL/CRITERIA are what a pull at that tier transfers.")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    remote = _remote(args)
    try:
        manifest = remote.manifest(
            args.run,
            role=args.role,
            tier=args.tier,
            include_future=args.include_future,
            checksums=not args.no_checksums,
        )
    except RemoteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    plan = sync.build_plan(manifest, verify=args.verify)
    print(f"run       {plan.run}")
    print(f"roles     {', '.join(plan.roles) or '-'}")
    print(f"tier      {plan.tier}")
    print(f"server    {manifest.get('file_count')} file(s), {sync.human_bytes(manifest.get('total_bytes') or 0)}")
    print(f"to fetch  {len(plan.download)} file(s), {sync.human_bytes(plan.download_bytes)}")
    print(f"up to date {len(plan.keep)} file(s)")
    if plan.obsolete:
        note = "will be removed" if args.prune else "use --prune to remove"
        print(f"obsolete  {len(plan.obsolete)} local file(s) no longer on server ({note})")

    if args.dry_run:
        for item in plan.download[:40]:
            print(f"  {item.reason:11s} {sync.human_bytes(item.size):>10s}  {item.rel_path}")
        if len(plan.download) > 40:
            print(f"  ... and {len(plan.download) - 40} more")
        return 0
    if not plan.download and not (args.prune and plan.obsolete):
        print("\nNothing to do.")
        return 0

    print()
    result = sync.execute(remote, manifest, plan, prune=args.prune)
    print(
        f"\ndone: {result['downloaded']} downloaded ({sync.human_bytes(result['bytes'])}), "
        f"{result['kept']} kept, {len(result['pruned'])} pruned"
    )
    print(f"workspace: {config.run_dir(plan.run)}")
    if result["failed"]:
        print(f"\n{len(result['failed'])} file(s) FAILED; re-run pull to resume:", file=sys.stderr)
        for rel in result["failed"][:20]:
            print(f"  {rel}", file=sys.stderr)
        return 1
    print("\nNext: evaldash-local open")
    return 0


def cmd_prebake(args: argparse.Namespace) -> int:
    remote = _remote(args)
    try:
        result = remote.prebake(
            args.run,
            args.parquet,
            routes=args.routes or None,
            force=args.force,
            status_only=args.status,
        )
    except RemoteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    rows = sync.local_run_summary()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print(f"No local runs in {config.workspace_dir()}")
        print("Pull one with: evaldash-local pull <run>")
        return 0
    _print_table(
        [
            [
                r["name"],
                r["tier"],
                ",".join(r["roles"]),
                str(r["files"]),
                sync.human_bytes(r["bytes"]),
                str(r["incomplete"]) if r["incomplete"] else "-",
                (r["updated_at"] or "")[:19],
            ]
            for r in rows
        ],
        ["RUN", "TIER", "ROLES", "FILES", "SIZE", "PARTIAL", "UPDATED"],
    )
    print(f"\nworkspace: {config.workspace_dir()}")
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    for name in args.runs:
        target = config.run_dir(name)
        if not target.is_dir():
            print(f"skip {name}: not present locally", file=sys.stderr)
            continue
        if not args.yes:
            size = sync.human_bytes(sum(p.stat().st_size for p in target.rglob("*") if p.is_file()))
            reply = input(f"Delete {target} ({size})? [y/N] ").strip().lower()
            if reply not in ("y", "yes"):
                print(f"skip {name}")
                continue
        ok, message = sync.remove_run(name)
        print(message if ok else f"error: {message}", file=sys.stderr if not ok else sys.stdout)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    server = serve.LocalServer(port=args.port)
    url = server.start()
    # flush=True: stdout is block buffered when this is piped to a log, and a server
    # whose address never appears is useless.
    print(f"home      {url}/", flush=True)
    print(f"explorer  {url}/explorer", flush=True)
    print(f"viewer    {url}/viewer", flush=True)
    print(f"workspace {config.workspace_dir()}", flush=True)
    print("\nCtrl-C to stop.", flush=True)
    server.serve_forever()
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    from client.app import launch

    return launch(port=args.port, page=args.page, prefer_browser=args.browser)


def cmd_t4(args: argparse.Namespace) -> int:
    """Manage the offline 3D point-cloud cache."""
    from client import t4

    if args.t4_command == "ls":
        scenes = t4.cached_scenes()
        if args.json:
            print(json.dumps(scenes, indent=2))
            return 0
        if not scenes:
            print(f"No cached 3D scenes in {t4.t4_root()}")
            print("Fetch one with: evaldash-local t4 fetch <dataset_id> --scenario <name>")
            return 0
        _print_table(
            [
                [
                    s["dataset_id"],
                    s["scenario"],
                    f"{s['frames_cached']}/{s['frames_total'] or '?'}",
                    t4.human_bytes(s["bytes"]),
                    "yes" if s["complete"] else "partial",
                    (s["fetched_at"] or "")[:16],
                ]
                for s in scenes
            ],
            ["DATASET", "SCENARIO", "FRAMES", "SIZE", "COMPLETE", "FETCHED"],
        )
        print(f"\ncache: {t4.t4_root()}")
        return 0

    if args.t4_command == "rm":
        ok, message = t4.remove_scene(args.dataset_id, args.scenario)
        print(message, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1

    if args.t4_command == "scenarios":
        cfg = config.Config.load()
        client = t4.T4Client(args.t4_base_url or cfg.t4_base_url, cfg)
        print(json.dumps(client.scenarios(args.dataset_id), indent=2))
        return 0

    # fetch
    cfg = config.Config.load()
    base = args.t4_base_url or cfg.t4_base_url
    if not args.scenario:
        print("error: --scenario is required (list them with: t4 scenarios <dataset_id>)", file=sys.stderr)
        return 2

    frames = None
    if args.frames:
        try:
            low, _, high = args.frames.partition("-")
            frames = range(int(low), int(high) + 1) if high else range(int(low), int(low) + 1)
        except ValueError:
            print(f"error: --frames expects N or N-M, got {args.frames!r}", file=sys.stderr)
            return 2

    if not args.yes:
        try:
            estimate = t4.estimate_scene_bytes(
                args.dataset_id, args.scenario, base_url=base, version=args.version
            )
        except t4.T4Error as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        count = len(frames) if frames is not None else estimate["frames"]
        approx = estimate["frame_bytes"] * count
        print(f"dataset   {args.dataset_id}")
        print(f"scenario  {args.scenario}")
        print(f"frames    {count} of {estimate['frames']}")
        print(f"size      ~{t4.human_bytes(approx)} ({estimate['points_per_frame']:,} points/frame)")
        print(f"          {estimate['note']}")
        reply = input("\nDownload? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("cancelled")
            return 0
        print()

    def report(state: dict) -> None:
        line = (
            f"\r  [{state['index']}/{state['total']}] frame {state['frame']}  "
            f"{t4.human_bytes(state['bytes'])}  eta {int(state['eta_sec'] or 0)}s"
        )
        sys.stderr.write(line.ljust(78)[:78])
        sys.stderr.flush()

    try:
        stats = t4.fetch_scene(
            args.dataset_id,
            args.scenario,
            base_url=base,
            version=args.version,
            frames=frames,
            with_lanelet=not args.no_lanelet,
            with_camera=not args.no_camera,
            force=args.force,
            progress=None if args.quiet else report,
        )
    except t4.T4Error as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    sys.stderr.write("\n")

    print(
        f"fetched {stats['frames_fetched']} frame(s), skipped {stats['frames_skipped']}, "
        f"{t4.human_bytes(stats['bytes'])} in {int(stats['elapsed_sec'])}s"
    )
    print(f"cache: {t4.scene_dir(args.dataset_id, args.scenario)}")
    if stats["errors"]:
        print(f"\n{len(stats['errors'])} sub-request(s) failed:", file=sys.stderr)
        for err in stats["errors"][:8]:
            print(f"  frame {err['frame']} {err['what']}: {err['error'][:100]}", file=sys.stderr)
    if stats["stopped_early"]:
        print("Stopped early; re-run to resume.")
    print("\nOpen it from the viewer, or directly:")
    print(f"  evaldash-local serve   then  /viewer/three?t4dataset_id={args.dataset_id}"
          f"&scenario_name={args.scenario}")
    return 1 if stats["errors"] and not stats["frames_fetched"] else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what this build can actually do. First stop when something misbehaves."""
    import sys as _sys

    from backend import app_paths

    checks: list[tuple[str, bool | None, str]] = []

    frozen = bool(getattr(_sys, "frozen", False))
    # Informational, not a verdict: running from source is a normal way to use this.
    checks.append(("packaged build", True if frozen else None,
                   "single-file app" if frozen else "running from source"))

    try:
        import duckdb

        checks.append(("duckdb", True, duckdb.__version__))
    except Exception as exc:
        checks.append(("duckdb", False, f"unavailable: {exc} (query routes will fail)"))

    try:
        import yaml  # noqa: F401

        checks.append(("PyYAML", True, "scenario.yaml criteria readable"))
    except Exception:
        checks.append(("PyYAML", None, "missing: DevOps criteria fall back to name heuristics"))

    required_assets = ["local_bbox_explorer.html", "local_bbox_viewer.html", "bbox_theme.js"]
    missing = [name for name in required_assets if app_paths.find_static_file(name) is None]
    checks.append(
        (
            "viewer assets",
            not missing,
            f"found in {app_paths.static_dirs()[0]}" if not missing else f"MISSING: {', '.join(missing)}",
        )
    )

    from client.app import _has_pywebview

    if _has_pywebview():
        backends = []
        for name in ("gtk", "qt", "cocoa", "winforms", "edgechromium"):
            try:
                __import__(f"webview.platforms.{name}")
                backends.append(name)
            except Exception:
                continue
        checks.append(("native window", True, f"pywebview backends: {', '.join(backends) or 'none usable'}"))
    else:
        checks.append(("native window", None, "pywebview unavailable; 'open' uses the system browser"))

    cfg = config.Config.load()
    # Configuration state is reported as a warning, never a failure: a fresh install has
    # none of it and is not broken, so the exit code must stay 0 for scripted checks.
    server = cfg.effective_server()
    checks.append(("server configured", True if server else None,
                   server or "not set yet - use the app's home page or 'login'"))
    checks.append(("token stored", True if cfg.resolved_token() else None,
                   "yes" if cfg.resolved_token() else "none - usually not needed"))

    runs = config.local_runs()
    checks.append((
        "local runs",
        True if runs else None,
        f"{len(runs)} in {config.workspace_dir()}" if runs else "none pulled yet",
    ))

    from client import t4

    scenes = t4.cached_scenes()
    checks.append((
        "offline 3D scenes",
        True if scenes else None,
        f"{len(scenes)} cached in {t4.t4_root()}" if scenes else "none cached (3D needs the T4 visualizer)",
    ))

    if server and cfg.resolved_token():
        try:
            health = _remote(args).export_health()
            checks.append(("server reachable", bool(health.get("enabled")),
                           "exports enabled" if health.get("enabled") else "EVAL_EXPORT_TOKEN unset on server"))
        except Exception as exc:
            checks.append(("server reachable", False, str(exc)[:90]))

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        mark = "ok  " if ok else ("warn" if ok is None else "FAIL")
        print(f"[{mark}] {name.ljust(width)}  {detail}")
    return 0 if all(ok is not False for _, ok, _ in checks) else 1


def cmd_where(args: argparse.Namespace) -> int:
    cfg = config.Config.load()
    print(json.dumps(
        {
            "home": str(config.home()),
            "workspace": str(config.workspace_dir()),
            "cache": str(config.cache_dir()),
            "config": str(config.config_path()),
            "server_url": cfg.server_url,
            "token_set": bool(cfg.resolved_token()),
            "local_runs": config.local_runs(),
        },
        indent=2,
    ))
    return 0


# ------------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaldash-local",
        description="Local client for the perception evaluation dashboard: pull generated "
        "results from the server and inspect them offline.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("login", help="store and verify server connection settings")
    p.add_argument("--server", help="dashboard base URL, e.g. https://dash.example.com")
    p.add_argument("--token", help="only if the server sets EVAL_EXPORT_REQUIRE_TOKEN")
    p.add_argument("--cf-client-id", help="Cloudflare Access service token id")
    p.add_argument("--cf-client-secret", help="Cloudflare Access service token secret")
    p.add_argument("--t4-base-url", help="T4 visualizer URL used for 3D point clouds")
    p.add_argument("--insecure", action="store_true", help="skip TLS verification")
    p.set_defaults(func=cmd_login)

    p = sub.add_parser("runs", help="list runs on the server with per-tier download sizes")
    p.add_argument("-q", "--query", help="substring filter on run name")
    p.add_argument("--fast", action="store_true", help="skip size computation")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("pull", help="download a run into the local workspace")
    p.add_argument("run")
    p.add_argument("--role", default="all", choices=["all", "devops", "performance"])
    p.add_argument("--tier", default="criteria", choices=["minimal", "criteria", "full", "raw"],
                   help="how much to fetch (default: criteria)")
    p.add_argument("--include-future", action="store_true", help="also fetch future.parquet")
    p.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    p.add_argument("--verify", action="store_true", help="re-hash local files instead of trusting size")
    p.add_argument("--prune", action="store_true", help="delete local files no longer on the server")
    p.add_argument("--no-checksums", action="store_true",
                   help="skip server-side hashing (faster manifest, no verification)")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("prebake", help="ask the server to pre-compute DevOps answers for a parquet")
    p.add_argument("run")
    p.add_argument("parquet", help="run-relative path, e.g. devops/current.parquet")
    p.add_argument("--routes", nargs="*", help="limit to specific routes")
    p.add_argument("--force", action="store_true", help="regenerate entries that already exist")
    p.add_argument("--status", action="store_true", help="report existing state without generating")
    p.set_defaults(func=cmd_prebake)

    p = sub.add_parser("ls", help="list local runs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("rm", help="delete local runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("serve", help="serve the viewer locally without opening a window")
    p.add_argument("--port", type=int, help="port to bind (default: first free from 8765)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("open", help="open the desktop app window")
    p.add_argument("--port", type=int)
    p.add_argument("--page", default="home", choices=["home", "explorer", "viewer"],
                   help="which page to open (default: home, where you download runs)")
    p.add_argument("--browser", action="store_true", help="use the system browser instead of a window")
    p.set_defaults(func=cmd_open)

    p = sub.add_parser("where", help="print workspace paths and current settings")
    p.set_defaults(func=cmd_where)

    p = sub.add_parser("doctor", help="check this build's capabilities and server connectivity")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("t4", help="offline 3D point-cloud cache (from the T4 visualizer)")
    p.add_argument("--t4-base-url", help="override the stored T4 visualizer URL")
    t4sub = p.add_subparsers(dest="t4_command", required=True)

    q = t4sub.add_parser("fetch", help="download one scenario's point clouds for offline use")
    q.add_argument("dataset_id")
    q.add_argument("--scenario", help="scenario name (required)")
    q.add_argument("--version", help="dataset version")
    q.add_argument("--frames", help="limit to a frame or range, e.g. 0-49")
    q.add_argument("--no-lanelet", action="store_true", help="skip lanelet map lines")
    q.add_argument("--no-camera", action="store_true", help="skip camera overlays and calibration")
    q.add_argument("--force", action="store_true", help="refetch frames already cached")
    q.add_argument("-y", "--yes", action="store_true", help="skip the size confirmation")
    q.add_argument("--quiet", action="store_true")

    q = t4sub.add_parser("ls", help="list cached scenes")
    q.add_argument("--json", action="store_true")

    q = t4sub.add_parser("rm", help="delete a cached scene")
    q.add_argument("dataset_id")
    q.add_argument("scenario")

    q = t4sub.add_parser("scenarios", help="list a dataset's scenarios on the T4 visualizer")
    q.add_argument("dataset_id")

    p.set_defaults(func=cmd_t4)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Server/token overrides are accepted on any remote command for one-off use.
    for name in ("server", "token"):
        if not hasattr(args, name):
            setattr(args, name, None)
    try:
        return int(args.func(args) or 0)
    except AuthError as exc:
        print(f"auth error: {exc}", file=sys.stderr)
        return 1
    except RemoteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        # Configuration problems (no server stored yet) are user errors, not crashes.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
