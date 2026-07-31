"""Manifest diffing and resumable download of run files.

Two properties matter here. A pull must be **resumable**, because the headline file is
a 300-500 MB parquet over a VPN, and it must be **incremental**, because re-pulling a
run after the server regenerated one scenario should not move half a gigabyte again.

Both come from comparing three things: the server manifest, the state file written by
the previous pull, and what is actually on disk right now (which may disagree with the
state file if a download was interrupted or a file was deleted by hand).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from client import config
from client.remote import Remote

CHUNK_BYTES = 1024 * 1024
PART_SUFFIX = ".evaldash-part"


@dataclass
class PlanItem:
    rel_path: str
    size: int
    sha256: str
    reason: str  # "new" | "changed" | "incomplete"
    # Bytes still to transfer. Smaller than ``size`` when a previous attempt left a
    # resumable ``.part`` file, so the reported total matches what actually moves.
    remaining: int = -1

    def __post_init__(self) -> None:
        if self.remaining < 0:
            self.remaining = self.size


@dataclass
class Plan:
    run: str
    tier: str
    roles: list[str]
    download: list[PlanItem]
    keep: list[str]
    obsolete: list[str]

    @property
    def download_bytes(self) -> int:
        return sum(item.remaining for item in self.download)

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.download)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_bytes(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(count) < 1024.0 or unit == "TB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024.0
    return f"{count:.1f} TB"


def build_plan(manifest: dict[str, Any], *, verify: bool = False) -> Plan:
    """Decide what to fetch. ``verify`` re-hashes local files instead of trusting size."""
    run = str(manifest.get("run") or "")
    if not run:
        raise ValueError("Manifest has no run name")
    previous = config.read_run_state(run).get("files") or {}
    target_dir = config.run_dir(run)

    download: list[PlanItem] = []
    keep: list[str] = []
    server_paths: set[str] = set()

    def queue(rel: str, size: int, sha: str, reason: str) -> None:
        """Queue a fetch, discounting bytes a resumable ``.part`` file already holds."""
        remaining = size
        if reason in ("new", "incomplete"):
            part = target_dir / (rel + PART_SUFFIX)
            if part.is_file():
                have = part.stat().st_size
                if 0 < have <= size:
                    remaining = size - have
        download.append(PlanItem(rel, size, sha, reason, remaining))

    for entry in manifest.get("files") or []:
        rel = str(entry.get("rel_path") or "")
        if not rel:
            continue
        server_paths.add(rel)
        size = int(entry.get("size") or 0)
        sha = str(entry.get("sha256") or "")
        local = target_dir / rel
        record = previous.get(rel) or {}

        if not local.is_file():
            queue(rel, size, sha, "new")
            continue
        local_size = local.stat().st_size
        if local_size != size:
            queue(rel, size, sha, "incomplete" if local_size < size else "changed")
            continue
        if sha:
            # With --verify the local bytes are hashed, so the comparison is authoritative.
            # Without it we fall back to what the previous pull recorded; when there is no
            # record (file copied in by hand, state file lost) the matching size is all we
            # have, and re-downloading gigabytes on that basis would make re-pulls useless.
            local_sha = sha256_file(local) if verify else str(record.get("sha256") or "")
            if local_sha and local_sha != sha:
                queue(rel, size, sha, "changed")
                continue
            keep.append(rel)
            continue
        # No server checksum: size plus the server's mtime is the best signal available.
        if record.get("mtime_ns") and entry.get("mtime_ns") and record["mtime_ns"] != entry["mtime_ns"]:
            queue(rel, size, sha, "changed")
            continue
        keep.append(rel)

    obsolete = sorted(set(previous) - server_paths)
    return Plan(
        run=run,
        tier=str(manifest.get("tier") or ""),
        roles=list(manifest.get("roles") or []),
        download=download,
        keep=keep,
        obsolete=obsolete,
    )


class Progress:
    """Single-line transfer progress on a TTY, plain lines otherwise.

    ``sink`` receives the same numbers as a dict, which is how the in-app UI follows a
    pull without parsing terminal output.
    """

    def __init__(
        self,
        total_bytes: int,
        total_files: int,
        stream=sys.stderr,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.total_bytes = total_bytes
        self.total_files = total_files
        self.done_bytes = 0
        self.done_files = 0
        self.stream = stream
        self.tty = hasattr(stream, "isatty") and stream.isatty()
        self.started = time.monotonic()
        self._last_render = 0.0
        self._last_sink = 0.0
        self.current = ""
        self.sink = sink

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.started, 1e-6)
        rate = self.done_bytes / elapsed
        remaining = max(self.total_bytes - self.done_bytes, 0)
        return {
            "done_bytes": self.done_bytes,
            "total_bytes": self.total_bytes,
            "done_files": self.done_files,
            "total_files": self.total_files,
            "current": self.current,
            "rate_bps": rate,
            "elapsed_sec": elapsed,
            "eta_sec": (remaining / rate) if rate > 0 else None,
            "percent": (self.done_bytes / self.total_bytes * 100.0) if self.total_bytes else 100.0,
        }

    def _emit(self, force: bool = False) -> None:
        if self.sink is None:
            return
        now = time.monotonic()
        # Throttled: a 1 MB chunk loop would otherwise call this thousands of times.
        if force or now - self._last_sink >= 0.2:
            self._last_sink = now
            try:
                self.sink(self.snapshot())
            except Exception:
                pass

    def advance(self, count: int) -> None:
        self.done_bytes += count
        now = time.monotonic()
        if self.tty and now - self._last_render >= 0.1:
            self._last_render = now
            self._render()
        self._emit()

    def start_file(self, rel_path: str, resumed: int = 0) -> None:
        self.current = rel_path
        if resumed:
            self.done_bytes += resumed
        if not self.tty:
            note = f" (resuming at {human_bytes(resumed)})" if resumed else ""
            print(f"  fetching {rel_path}{note}", file=self.stream, flush=True)
        self._emit(force=True)

    def finish_file(self) -> None:
        self.done_files += 1
        if self.tty:
            self._render()
        self._emit(force=True)

    def _render(self) -> None:
        elapsed = max(time.monotonic() - self.started, 1e-6)
        rate = self.done_bytes / elapsed
        pct = (self.done_bytes / self.total_bytes * 100.0) if self.total_bytes else 100.0
        name = self.current[-42:]
        line = (
            f"\r  [{pct:5.1f}%] {human_bytes(self.done_bytes)}/{human_bytes(self.total_bytes)} "
            f"| {human_bytes(rate)}/s | file {self.done_files}/{self.total_files} | {name}"
        )
        self.stream.write(line.ljust(118)[:118])
        self.stream.flush()

    def close(self) -> None:
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()


def _download_one(
    remote: Remote, run: str, item: PlanItem, target_dir: Path, progress: Progress, *, retries: int = 3
) -> str:
    """Fetch one file through a ``.part`` sidecar, resuming a previous attempt.

    Returns the sha256 of the stored bytes when it could be computed cheaply.
    """
    final = target_dir / item.rel_path
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + PART_SUFFIX)

    # A stale part longer than the target means the server's copy changed; start over.
    offset = part.stat().st_size if part.is_file() else 0
    if offset > item.size:
        part.unlink()
        offset = 0
    if item.reason == "changed" and offset:
        part.unlink()
        offset = 0

    attempt = 0
    while True:
        attempt += 1
        progress.start_file(item.rel_path, resumed=offset if attempt == 1 else 0)
        try:
            with remote.open_file(run, item.rel_path, offset=offset) as response:
                mode = "ab" if offset else "wb"
                with part.open(mode) as sink:
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        sink.write(chunk)
                        offset += len(chunk)
                        progress.advance(len(chunk))
            break
        except Exception as exc:
            offset = part.stat().st_size if part.is_file() else 0
            if attempt > retries:
                raise RuntimeError(f"{item.rel_path}: {exc}") from exc
            print(
                f"\n  retry {attempt}/{retries} for {item.rel_path} at {human_bytes(offset)}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(min(2.0 * attempt, 10.0))

    actual = part.stat().st_size
    if item.size and actual != item.size:
        raise RuntimeError(
            f"{item.rel_path}: expected {item.size} bytes, stored {actual}. Re-run pull to retry."
        )
    digest = ""
    if item.sha256:
        digest = sha256_file(part)
        if digest != item.sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{item.rel_path}: checksum mismatch; discarded. Re-run pull to retry.")
    part.replace(final)
    progress.finish_file()
    return digest


def execute(
    remote: Remote,
    manifest: dict[str, Any],
    plan: Plan,
    *,
    prune: bool = False,
    on_error: Callable[[str, Exception], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Download everything in ``plan``, updating the run state as files land.

    State is written after every file so that an interrupted pull still records what
    completed, and a re-run resumes rather than restarting.

    ``should_stop`` is checked between files so the UI can cancel a pull; the partial
    file stays as a ``.part`` sidecar and the next pull resumes from it.
    """
    target_dir = config.run_dir(plan.run)
    target_dir.mkdir(parents=True, exist_ok=True)
    state = config.read_run_state(plan.run)
    files_state: dict[str, Any] = dict(state.get("files") or {})
    by_path = {str(f.get("rel_path")): f for f in manifest.get("files") or []}

    progress = Progress(plan.download_bytes, len(plan.download), sink=on_progress)
    failures: list[str] = []
    transferred = 0
    cancelled = False
    try:
        for item in plan.download:
            if should_stop is not None and should_stop():
                cancelled = True
                break
            try:
                digest = _download_one(remote, plan.run, item, target_dir, progress)
            except Exception as exc:
                failures.append(item.rel_path)
                if on_error is not None:
                    on_error(item.rel_path, exc)
                else:
                    print(f"\n  FAILED {item.rel_path}: {exc}", file=sys.stderr, flush=True)
                continue
            entry = by_path.get(item.rel_path) or {}
            files_state[item.rel_path] = {
                "size": item.size,
                "sha256": digest or item.sha256,
                "mtime_ns": entry.get("mtime_ns"),
            }
            transferred += item.size
            config.write_run_state(
                plan.run, {**state, "run": plan.run, "tier": plan.tier, "files": files_state}
            )
    finally:
        progress.close()

    for rel in plan.keep:
        if rel not in files_state:
            entry = by_path.get(rel) or {}
            files_state[rel] = {
                "size": entry.get("size"),
                "sha256": entry.get("sha256") or "",
                "mtime_ns": entry.get("mtime_ns"),
            }

    pruned: list[str] = []
    if prune and not cancelled:
        for rel in plan.obsolete:
            victim = target_dir / rel
            if victim.is_file():
                victim.unlink()
                pruned.append(rel)
            files_state.pop(rel, None)

    config.write_run_state(
        plan.run,
        {
            "run": plan.run,
            "tier": plan.tier,
            "roles": plan.roles,
            "server": remote.base_url,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "files": files_state,
        },
    )
    return {
        "run": plan.run,
        "downloaded": progress.done_files,
        "failed": failures,
        "kept": len(plan.keep),
        "pruned": pruned,
        "bytes": transferred,
        "cancelled": cancelled,
    }


def remove_run(run_name: str) -> tuple[bool, str]:
    target = config.run_dir(run_name)
    if not target.is_dir():
        return False, f"No local run named {run_name}"
    shutil.rmtree(target)
    return True, f"Removed {target}"


def local_run_summary() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in config.local_runs():
        state = config.read_run_state(name)
        directory = config.run_dir(name)
        total = 0
        partial = 0
        for path in directory.rglob("*"):
            if path.is_file():
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                total += size
                if path.name.endswith(PART_SUFFIX):
                    partial += 1
        out.append(
            {
                "name": name,
                "tier": state.get("tier") or "?",
                "roles": state.get("roles") or [],
                "files": len(state.get("files") or {}),
                "bytes": total,
                "incomplete": partial,
                "updated_at": state.get("updated_at") or "",
                "server": state.get("server") or "",
            }
        )
    return out
