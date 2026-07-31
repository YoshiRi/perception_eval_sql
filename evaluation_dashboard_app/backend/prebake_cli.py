"""Server-side pre-baking: ``python3 -m backend.prebake_cli``.

Pre-baking unpickles every scenario's ``scene_result.pkl``, which is tens of minutes for
a large run. That is the wrong shape for an HTTP request, so the bulk path is this CLI;
``/api/export_prebake`` remains for topping up a single parquet.

Run it where the evaluator libraries and the pickles live -- normally in the container:

    # what still needs work, across every run (fast, reads no pickles)
    docker compose exec streamlit1 python3 -m backend.prebake_cli --report

    # bake one run, logging progress
    docker compose exec streamlit1 python3 -m backend.prebake_cli --run <name>

    # bake everything overnight, detached and resumable
    docker compose exec -d streamlit1 sh -c \\
      'python3 -m backend.prebake_cli --all --role devops >/app/data/prebake.log 2>&1'

Interrupting is always safe. Entries are written one at a time and a later run skips
what exists, so work resumes instead of restarting. ``--max-seconds`` time-boxes a batch
for the same reason.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from backend import app_paths, prebake

# Set by SIGINT/SIGTERM so the current scenario finishes and state stays consistent.
_STOP = False


def _install_signal_handlers() -> None:
    def request_stop(signum: int, _frame: object) -> None:
        global _STOP
        if _STOP:  # second signal: the user means now
            raise KeyboardInterrupt
        _STOP = True
        print(
            f"\n[signal {signum}] finishing the current scenario, then stopping. "
            "Progress is saved; re-run to resume. Press again to abort immediately.",
            file=sys.stderr,
            flush=True,
        )

    for name in ("SIGINT", "SIGTERM"):
        handler = getattr(signal, name, None)
        if handler is not None:
            signal.signal(handler, request_stop)


def _human_secs(seconds: float | None) -> str:
    if not seconds:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _human_bytes(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(count) < 1024.0 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024.0
    return f"{count:.1f} GB"


def _iter_parquets(run_dir: Path, roles: list[str] | None) -> list[Path]:
    out: list[Path] = []
    for role in roles or ["devops", "performance"]:
        role_dir = run_dir / role
        if role_dir.is_dir():
            out.extend(sorted(role_dir.glob("current.parquet")))
    return out


def _run_names(root: Path, requested: list[str]) -> list[str]:
    if requested:
        return requested
    return sorted(child.name for child in root.iterdir() if child.is_dir() and not child.name.startswith("."))


def _report(root: Path, names: list[str], roles: list[str] | None) -> int:
    rows: list[tuple[str, str, str, str, str]] = []
    for name in names:
        for parquet in _iter_parquets(root / name, roles):
            state = prebake.coverage(parquet)
            rows.append(
                (
                    name,
                    str(parquet.relative_to(root / name)),
                    f"{state['covered']}/{state['scenarios']}",
                    _human_bytes(state["bytes"]),
                    "complete" if state["complete"] else f"{len(state['missing'])} missing",
                )
            )
    if not rows:
        print("No current.parquet found. Nothing to pre-bake.")
        return 0
    headers = ("RUN", "PARQUET", "COVERED", "SIZE", "STATUS")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    incomplete = [r for r in rows if r[4] != "complete"]
    print(f"\n{len(rows)} parquet(s), {len(incomplete)} need work.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prebake_cli",
        description="Pre-compute pickle-backed DevOps answers so a local client can serve "
        "them without scene_result.pkl or the evaluator libraries.",
        epilog="Interrupting is safe and resumable; see the module docstring for examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", action="append", default=[], help="run name (repeatable)")
    parser.add_argument("--all", action="store_true", help="every run under the data root")
    parser.add_argument("--role", action="append", default=[], choices=["devops", "performance"])
    parser.add_argument("--routes", nargs="*", default=None, help=f"subset of {list(prebake.PREBAKE_ROUTES)}")
    parser.add_argument("--force", action="store_true", help="regenerate entries that already exist")
    parser.add_argument("--report", action="store_true", help="show coverage and exit (reads no pickles)")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="stop after roughly this long; resume by re-running")
    parser.add_argument("--quiet", action="store_true", help="only print per-parquet summaries")
    parser.add_argument("--json", action="store_true", help="machine-readable summary on stdout")
    args = parser.parse_args(argv)

    root = app_paths.data_root()
    if not root.is_dir():
        print(f"error: data root does not exist: {root}", file=sys.stderr)
        return 1
    if not args.run and not args.all and not args.report:
        print("error: pass --run <name>, --all, or --report", file=sys.stderr)
        return 2

    names = [n for n in _run_names(root, args.run) if (root / n).is_dir()]
    for missing in set(args.run) - set(names):
        print(f"skip {missing}: not found under {root}", file=sys.stderr)
    roles = args.role or None

    if args.report:
        return _report(root, names, roles)

    routes = args.routes or list(prebake.PREBAKE_ROUTES)
    unknown = [r for r in routes if r not in prebake.PREBAKE_ROUTES]
    if unknown:
        print(f"error: unknown route(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    _install_signal_handlers()
    batch_started = time.monotonic()
    summaries: list[dict] = []

    for name in names:
        for parquet in _iter_parquets(root / name, roles):
            if _STOP:
                break
            rel = parquet.relative_to(root)
            budget = None
            if args.max_seconds is not None:
                budget = args.max_seconds - (time.monotonic() - batch_started)
                if budget <= 0:
                    print(f"\ntime budget reached; stopping before {rel}", file=sys.stderr)
                    break
            print(f"\n=== {rel} ===", flush=True)

            def report(state: dict) -> None:
                if args.quiet:
                    return
                print(
                    f"  [{state['index']}/{state['total']}] {state['seconds']:>5.1f}s "
                    f"eta {_human_secs(state['eta_sec']):>6s}  {state['scenario']}",
                    file=sys.stderr,
                    flush=True,
                )

            stats = prebake.generate(
                parquet,
                routes=routes,
                force=args.force,
                progress=report,
                time_budget_sec=budget,
                should_stop=lambda: _STOP,
            )
            stats["parquet"] = str(rel)
            summaries.append(stats)
            print(
                f"  scenarios={stats['scenarios_done']}/{stats['scenarios']} "
                f"written={stats['written']} skipped={stats['skipped']} failed={stats['failed']} "
                f"size={_human_bytes(stats['bytes'])} in {_human_secs(stats['elapsed_sec'])}"
                + (f"  [STOPPED: {stats.get('stop_reason')}]" if stats["stopped_early"] else "")
            )
            for error in stats["errors"][:5]:
                print(f"  ! {error['route']} {error['scenario']}: {error['error']}", file=sys.stderr)
            if len(stats["errors"]) > 5:
                print(f"  ! ... and {len(stats['errors']) - 5} more failures", file=sys.stderr)
        if _STOP:
            break

    total_written = sum(s["written"] for s in summaries)
    total_failed = sum(s["failed"] for s in summaries)
    print(
        f"\n{len(summaries)} parquet(s): {total_written} entries written, {total_failed} failed, "
        f"in {_human_secs(time.monotonic() - batch_started)}"
    )
    if _STOP or any(s["stopped_early"] for s in summaries):
        print("Stopped before finishing. Re-run the same command to resume.")
    if args.json:
        print(json.dumps(summaries, indent=2))
    return 1 if total_failed and not total_written else 0


if __name__ == "__main__":
    raise SystemExit(main())
