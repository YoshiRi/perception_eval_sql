from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml


DEFAULT_WORK_DIR = Path(os.environ.get("PR_TEST_BRANCH_WORKDIR", "/tmp/evaluator-dashboard-pr-workspaces"))
DEFAULT_PILOT_CHECKOUT = os.environ.get("PR_TEST_BRANCH_PILOT_CHECKOUT", "")
DEFAULT_PILOT_REPO_URL = os.environ.get(
    "PR_TEST_BRANCH_PILOT_REPO_URL",
    "git@github.com:tier4/pilot-auto.x2.git",
)
DEFAULT_BRANCH_PREFIX = "evaluator-dashboard"
DEFAULT_GIT_USER_NAME = os.environ.get("PR_TEST_BRANCH_GIT_USER_NAME", "Evaluator Dashboard")
DEFAULT_GIT_USER_EMAIL = os.environ.get("PR_TEST_BRANCH_GIT_USER_EMAIL", "evaluator-dashboard@localhost")

SUB_REPOS: Dict[str, Dict[str, str]] = {
    "universe": {
        "label": "Universe",
        "path": "src/autoware/universe",
        "repos_key": "autoware/universe",
        "upstream_repo": "autowarefoundation/autoware_universe",
        "upstream_url": "https://github.com/autowarefoundation/autoware_universe.git",
    },
    "launcher": {
        "label": "Launcher",
        "path": "src/autoware/launcher",
        "repos_key": "autoware/launcher",
        "upstream_repo": os.environ.get("PR_TEST_BRANCH_LAUNCHER_UPSTREAM_REPO", ""),
        "upstream_url": os.environ.get("PR_TEST_BRANCH_LAUNCHER_UPSTREAM_URL", ""),
    },
}


class CommandError(RuntimeError):
    pass


class WorkflowLogger:
    def __init__(
        self,
        *,
        task_id: str,
        append_log: Callable[[str, str], Any],
        update_progress: Callable[..., Any],
        update_summary: Callable[[str, Dict[str, Any]], Any],
    ) -> None:
        self.task_id = task_id
        self.append_log = append_log
        self.update_progress = update_progress
        self.update_summary = update_summary
        self.steps: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {"job": "prepare_pr_test_branch", "steps": self.steps}

    def log(self, message: str) -> None:
        self.append_log(self.task_id, message)

    def progress(self, message: str, pct: float) -> None:
        self.update_progress(self.task_id, message=message, pct=pct)
        self.log(message)

    def set_summary(self, **updates: Any) -> None:
        self.summary.update(updates)
        self.update_summary(self.task_id, self.summary)

    def step(self, name: str, status: str, **details: Any) -> None:
        payload = {
            "name": name,
            "status": status,
            "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        payload.update({k: v for k, v in details.items() if v not in (None, "", [], {})})
        self.steps.append(payload)
        self.set_summary()


def _sanitize_branch_part(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._/-]+", "-", str(value or "").strip()).strip("/-")
    text = re.sub(r"/+", "/", text)
    return text or fallback


def _short_sha(sha: str) -> str:
    return str(sha or "")[:12]


def _run(
    cmd: List[str],
    *,
    cwd: Path,
    logger: WorkflowLogger,
    step_name: str,
    capture: bool = True,
    stream: bool = False,
    stdin_path: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    display = "$ " + " ".join(cmd) + f" (cwd: {cwd})"
    logger.log(display)
    if stream:
        stdin = stdin_path.open("rb") if stdin_path else None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_lines: List[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                clean = line.rstrip("\n")
                if clean:
                    output_lines.append(clean)
                    logger.log(f"{step_name}: {clean}")
            returncode = proc.wait()
        finally:
            if stdin:
                stdin.close()
        elapsed = round(time.monotonic() - started, 2)
        stdout = "\n".join(output_lines)
        logger.step(step_name, "completed" if returncode == 0 else "failed", command=cmd, cwd=str(cwd), elapsed_seconds=elapsed)
        if check and returncode != 0:
            raise CommandError(f"Command failed ({returncode}): {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=capture,
        stdin=stdin_path.open("rb") if stdin_path else None,
    )
    elapsed = round(time.monotonic() - started, 2)
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.log(f"{step_name}: {line}")
    if result.stderr:
        stderr_label = "stderr" if result.returncode != 0 else "output"
        for line in result.stderr.splitlines():
            logger.log(f"{step_name} {stderr_label}: {line}")
    logger.step(
        step_name,
        "completed" if result.returncode == 0 else "failed",
        command=cmd,
        cwd=str(cwd),
        elapsed_seconds=elapsed,
    )
    if check and result.returncode != 0:
        raise CommandError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def _git(repo: Path, *args: str, logger: WorkflowLogger, step_name: str, check: bool = True) -> str:
    return _run(["git", *args], cwd=repo, logger=logger, step_name=step_name, check=check).stdout.strip()


def _git_quiet(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo), text=True, capture_output=True, check=True).stdout.strip()


def _git_quiet_optional(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=str(repo), text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def _ensure_git_commit_identity(repo: Path, logger: WorkflowLogger, label: str) -> None:
    name = _git_quiet_optional(repo, "config", "user.name")
    email = _git_quiet_optional(repo, "config", "user.email")
    if not name:
        _git(
            repo,
            "config",
            "--local",
            "user.name",
            DEFAULT_GIT_USER_NAME,
            logger=logger,
            step_name=f"{label}: set git user.name",
        )
        name = DEFAULT_GIT_USER_NAME
    if not email:
        _git(
            repo,
            "config",
            "--local",
            "user.email",
            DEFAULT_GIT_USER_EMAIL,
            logger=logger,
            step_name=f"{label}: set git user.email",
        )
        email = DEFAULT_GIT_USER_EMAIL
    logger.step(f"{label}: git commit identity", "completed", user_name=name, user_email=email)


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def _is_clean(repo: Path) -> bool:
    return _git_quiet(repo, "status", "--porcelain") == ""


def _is_clean_except_untracked(repo: Path, allowed_untracked_prefixes: tuple[str, ...]) -> bool:
    for line in _git_quiet(repo, "status", "--porcelain").splitlines():
        if not line:
            continue
        if line.startswith("?? "):
            path = line[3:]
            if any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in allowed_untracked_prefixes):
                continue
        return False
    return True


def _current_branch(repo: Path) -> str:
    return _git_quiet(repo, "branch", "--show-current")


def _current_sha(repo: Path) -> str:
    return _git_quiet(repo, "rev-parse", "HEAD")


def _remote_url(repo: Path, remote: str) -> str:
    return _git_quiet(repo, "remote", "get-url", remote)


def _branch_exists(repo: Path, branch: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=str(repo),
    )
    return result.returncode == 0


def _remote_branch_exists(repo: Path, remote: str, branch: str) -> bool:
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", remote, branch],
        cwd=str(repo),
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def _ensure_clone(repo_url: str, clone_dir: Path, logger: WorkflowLogger) -> Path:
    if clone_dir.exists() and _is_git_repo(clone_dir):
        logger.step("Pilot checkout", "completed", action="reuse", path=str(clone_dir))
        return clone_dir
    if clone_dir.exists() and any(clone_dir.iterdir()):
        raise RuntimeError(f"Pilot checkout path exists but is not a git repo: {clone_dir}")
    if not repo_url:
        raise RuntimeError("Pilot repository URL is required when no reusable local checkout exists.")
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", repo_url, str(clone_dir)], cwd=clone_dir.parent, logger=logger, step_name="Clone pilot repository", stream=True)
    return clone_dir


def _reset_repo(repo: Path, branch_or_ref: str, logger: WorkflowLogger, label: str, *, clean: bool = True) -> None:
    _git(repo, "reset", "--hard", logger=logger, step_name=f"{label}: reset hard")
    if clean:
        _git(repo, "clean", "-fd", logger=logger, step_name=f"{label}: clean untracked")
    _git(repo, "checkout", branch_or_ref, logger=logger, step_name=f"{label}: checkout {branch_or_ref}")
    _git(repo, "reset", "--hard", branch_or_ref, logger=logger, step_name=f"{label}: reset {branch_or_ref}")


def _checkout_base(repo: Path, remote: str, base_branch: str, logger: WorkflowLogger, label: str) -> None:
    _git(repo, "fetch", remote, "--prune", logger=logger, step_name=f"{label}: fetch {remote}", check=True)
    remote_ref = f"{remote}/{base_branch}"
    try:
        _git(repo, "rev-parse", "--verify", remote_ref, logger=logger, step_name=f"{label}: verify remote base")
        if _branch_exists(repo, base_branch):
            _git(repo, "checkout", base_branch, logger=logger, step_name=f"{label}: checkout base")
            _git(repo, "reset", "--hard", remote_ref, logger=logger, step_name=f"{label}: reset to remote base")
        else:
            _git(repo, "checkout", "-B", base_branch, remote_ref, logger=logger, step_name=f"{label}: create local base")
    except CommandError:
        _git(repo, "checkout", base_branch, logger=logger, step_name=f"{label}: checkout local base")


def _run_vcs_import(repo_root: Path, repos_file: Path, logger: WorkflowLogger) -> None:
    if not repos_file.is_file():
        raise RuntimeError(f"Missing repos file: {repos_file}")
    vcs = shutil.which("vcs")
    if not vcs:
        raise RuntimeError("`vcs` command not found. Install vcstool or disable VCS update for an already prepared checkout.")
    (repo_root / "src").mkdir(exist_ok=True)
    _run([vcs, "import", "--debug", "src"], cwd=repo_root, logger=logger, step_name="vcs import --debug", stream=True, stdin_path=repos_file)


def _sync_selected_sub_repo(
    *,
    sub_repo: Path,
    repos_entry: Dict[str, Any],
    fallback_url: str,
    logger: WorkflowLogger,
) -> Dict[str, str]:
    repo_url = str(repos_entry.get("url") or fallback_url or "").strip()
    repos_branch = str(repos_entry.get("branch") or "").strip()
    repos_version = str(repos_entry.get("version") or "").strip()
    if not repo_url and not _is_git_repo(sub_repo):
        raise RuntimeError(f"Sub repo checkout is missing and no repository URL is configured: {sub_repo}")

    if _is_git_repo(sub_repo):
        logger.step("Sub repo: fast sync", "completed", action="reuse", path=str(sub_repo))
    else:
        sub_repo.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repo_url, str(sub_repo)], cwd=sub_repo.parent, logger=logger, step_name="Sub repo: clone selected repo", stream=True)

    origin_url = _git(sub_repo, "remote", "get-url", "origin", logger=logger, step_name="Sub repo: read origin", check=False)
    if not origin_url and repo_url:
        _git(sub_repo, "remote", "add", "origin", repo_url, logger=logger, step_name="Sub repo: add origin")

    _git(sub_repo, "fetch", "origin", "--prune", logger=logger, step_name="Sub repo: fetch origin")
    if repos_version:
        expected_sha = _git(
            sub_repo,
            "rev-parse",
            "--verify",
            f"{repos_version}^{{commit}}",
            logger=logger,
            step_name="Sub repo: verify repos version",
            check=False,
        )
        if not expected_sha:
            _git(sub_repo, "fetch", "origin", repos_version, logger=logger, step_name="Sub repo: fetch repos version", check=False)
            expected_sha = _git(
                sub_repo,
                "rev-parse",
                "--verify",
                f"{repos_version}^{{commit}}",
                logger=logger,
                step_name="Sub repo: resolve repos version",
            )
        _git(sub_repo, "checkout", "--detach", repos_version, logger=logger, step_name="Sub repo: checkout repos version")
        _git(sub_repo, "reset", "--hard", repos_version, logger=logger, step_name="Sub repo: reset repos version")
        actual_sha = _current_sha(sub_repo)
        if actual_sha != expected_sha:
            raise RuntimeError(f"Selected sub repo base mismatch: expected {expected_sha}, got {actual_sha}")
        return {"kind": "version", "ref": repos_version, "sha": expected_sha}
    elif repos_branch:
        remote_branch = f"origin/{repos_branch}"
        expected_sha = _git(
            sub_repo,
            "rev-parse",
            "--verify",
            f"{remote_branch}^{{commit}}",
            logger=logger,
            step_name="Sub repo: resolve repos branch",
        )
        _git(sub_repo, "checkout", "-B", repos_branch, remote_branch, logger=logger, step_name="Sub repo: checkout repos branch")
        _git(sub_repo, "reset", "--hard", remote_branch, logger=logger, step_name="Sub repo: reset repos branch")
        actual_sha = _current_sha(sub_repo)
        if actual_sha != expected_sha:
            raise RuntimeError(f"Selected sub repo base mismatch: expected {expected_sha}, got {actual_sha}")
        return {"kind": "branch", "ref": remote_branch, "sha": expected_sha}
    else:
        raise RuntimeError("autoware.repos entry has neither version nor branch for the selected sub repo.")


def _repos_entry(repos_file: Path, repos_key: str) -> Dict[str, Any]:
    data = yaml.safe_load(repos_file.read_text()) if repos_file.is_file() else {}
    repositories = data.get("repositories", {}) if isinstance(data, dict) else {}
    entry = repositories.get(repos_key, {}) if isinstance(repositories, dict) else {}
    return entry if isinstance(entry, dict) else {}


def _github_repo_from_url(url: str) -> str:
    text = str(url or "").strip()
    match = re.search(r"github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?/?$", text)
    return match.group(1) if match else ""


def _github_branch_url(remote_url: str, branch: str) -> str:
    repo = _github_repo_from_url(remote_url)
    if not repo or not branch:
        return ""
    return f"https://github.com/{repo}/tree/{urllib.parse.quote(branch, safe='/')}"


def _github_compare_url(remote_url: str, base: str, head: str) -> str:
    repo = _github_repo_from_url(remote_url)
    if not repo or not base or not head:
        return ""
    spec = f"{base}...{head}"
    return f"https://github.com/{repo}/compare/{urllib.parse.quote(spec, safe='/...')}"


def _read_pr_info(pr_number: int, upstream_repo: str, logger: WorkflowLogger) -> Dict[str, Any]:
    gh = shutil.which("gh")
    if not gh or not upstream_repo:
        return {"number": pr_number}
    result = _run(
        [
            gh,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            upstream_repo,
            "--json",
            "title,author,baseRefName,headRefName,headRefOid,url",
        ],
        cwd=Path.cwd(),
        logger=logger,
        step_name="Load PR metadata",
        check=False,
    )
    if result.returncode != 0:
        return {"number": pr_number}
    try:
        data = json.loads(result.stdout)
        data["number"] = pr_number
        return data
    except ValueError:
        return {"number": pr_number}


def _temporary_ref_name(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._/-]+", "-", label).strip("/-")
    return f"refs/tmp/evaluator-dashboard/{safe}"


def _fetch_source_ref(
    sub_repo_path: Path,
    *,
    source_branch: str,
    pr_number: Optional[int],
    base_branch: str,
    upstream_url: str,
    logger: WorkflowLogger,
) -> Dict[str, str]:
    if pr_number:
        head_ref = _temporary_ref_name(f"pr-{pr_number}-head")
        base_ref = _temporary_ref_name(f"pr-{pr_number}-base-{base_branch}")
        fetch_url = upstream_url or "origin"
        _git(sub_repo_path, "fetch", fetch_url, f"+pull/{pr_number}/head:{head_ref}", logger=logger, step_name="Sub repo: fetch PR head")
        _git(sub_repo_path, "fetch", fetch_url, f"+{base_branch}:{base_ref}", logger=logger, step_name="Sub repo: fetch PR base")
        return {"head_ref": head_ref, "base_ref": base_ref, "source_label": f"PR #{pr_number}"}

    if not source_branch:
        raise RuntimeError("Sub repo branch or PR number is required.")
    remote_ref = f"origin/{source_branch}"
    _git(sub_repo_path, "fetch", "origin", source_branch, logger=logger, step_name="Sub repo: fetch source branch")
    _git(sub_repo_path, "rev-parse", "--verify", remote_ref, logger=logger, step_name="Sub repo: verify source branch")
    return {"head_ref": remote_ref, "base_ref": "HEAD", "source_label": source_branch}


def _apply_source_diff(
    sub_repo_path: Path,
    *,
    head_ref: str,
    base_ref: str,
    patch_file: Path,
    logger: WorkflowLogger,
) -> List[str]:
    merge_base = _git(sub_repo_path, "merge-base", base_ref, head_ref, logger=logger, step_name="Sub repo: merge-base")
    logger.set_summary(merge_base=merge_base)
    with patch_file.open("wb") as output:
        result = subprocess.run(
            ["git", "diff", "--binary", merge_base, head_ref],
            cwd=str(sub_repo_path),
            stdout=output,
            stderr=subprocess.PIPE,
        )
    logger.step("Sub repo: write patch", "completed" if result.returncode == 0 else "failed", patch_file=str(patch_file))
    if result.returncode != 0:
        raise CommandError(result.stderr.decode(errors="replace") or "git diff failed")
    _git(sub_repo_path, "apply", "--3way", "--index", str(patch_file), logger=logger, step_name="Sub repo: apply patch")
    changed = _git(sub_repo_path, "diff", "--cached", "--name-only", logger=logger, step_name="Sub repo: staged files")
    return [line for line in changed.splitlines() if line.strip()]


def _update_repos_file(repos_file: Path, repos_key: str, branch: str, sha: str) -> None:
    lines = repos_file.read_text().splitlines(keepends=True)
    start = None
    target = f"{repos_key}:"
    for idx, line in enumerate(lines):
        if line.strip() == target and line.startswith("  "):
            start = idx
            break
    if start is None:
        raise RuntimeError(f"Could not find '{repos_key}' in {repos_file.name}")

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            end = idx
            break

    branch_idx = None
    version_idx = None
    for idx in range(start, end):
        if lines[idx].startswith("    branch:"):
            branch_idx = idx
        if lines[idx].startswith("    version:"):
            version_idx = idx
    if version_idx is None:
        raise RuntimeError(f"Could not find 'version:' under '{repos_key}' in {repos_file.name}")

    lines[version_idx] = f"    version: {sha}\n"
    if branch_idx is None:
        lines.insert(version_idx, f"    branch: {branch}\n")
    else:
        lines[branch_idx] = f"    branch: {branch}\n"
    repos_file.write_text("".join(lines))


def make_branch_name(prefix: str, repo_label: str, source_label: str, date: str, base_sha: str) -> str:
    source = _sanitize_branch_part(source_label, "source").replace("#", "pr")
    return f"{_sanitize_branch_part(prefix, DEFAULT_BRANCH_PREFIX)}/{repo_label}-{source}-{date}-{_short_sha(base_sha)}"


def _unique_branch_name(repo: Path, remote: str, base_name: str) -> str:
    candidate = base_name
    for index in range(2, 100):
        if not _branch_exists(repo, candidate) and not _remote_branch_exists(repo, remote, candidate):
            return candidate
        candidate = f"{base_name}-{index}"
    raise RuntimeError(f"Could not find an available branch name for {base_name}")


def run_prepare_pr_test_branch(
    *,
    task_id: str,
    parameters: Dict[str, Any],
    append_log: Callable[[str, str], Any],
    update_progress: Callable[..., Any],
    update_summary: Callable[[str, Dict[str, Any]], Any],
) -> Dict[str, Any]:
    logger = WorkflowLogger(
        task_id=task_id,
        append_log=append_log,
        update_progress=update_progress,
        update_summary=update_summary,
    )
    work_dir = Path(str(parameters.get("work_dir") or DEFAULT_WORK_DIR)).expanduser().resolve()
    pilot_repo_url = str(parameters.get("pilot_repo_url") or DEFAULT_PILOT_REPO_URL).strip()
    pilot_checkout = Path(str(parameters.get("pilot_checkout") or work_dir / "pilot-auto")).expanduser().resolve()
    pilot_remote = str(parameters.get("pilot_remote") or "origin").strip() or "origin"
    sub_remote = str(parameters.get("sub_remote") or "origin").strip() or "origin"
    pilot_base_branch = str(parameters.get("pilot_base_branch") or "").strip()
    sub_repo_name = str(parameters.get("sub_repo") or "universe").strip().lower()
    sub_defaults = SUB_REPOS.get(sub_repo_name, SUB_REPOS["universe"])
    sub_repo_path_text = str(parameters.get("sub_repo_path") or "").strip()
    repos_key = str(parameters.get("repos_key") or sub_defaults["repos_key"]).strip()
    upstream_repo = str(parameters.get("upstream_repo") or "").strip()
    upstream_url = str(parameters.get("upstream_url") or "").strip()
    source_branch = str(parameters.get("sub_repo_branch") or "").strip()
    pr_number_raw = str(parameters.get("pr_number") or "").strip()
    pr_number = int(pr_number_raw) if pr_number_raw.isdigit() else None
    branch_prefix = str(parameters.get("branch_prefix") or DEFAULT_BRANCH_PREFIX).strip() or DEFAULT_BRANCH_PREFIX
    run_vcs = bool(parameters.get("run_vcs_update", False))
    reset_cache = bool(parameters.get("reset_cache", True))
    restore_after = bool(parameters.get("restore_after", True))
    prepare_only = bool(parameters.get("prepare_only", True))

    if not pilot_base_branch:
        raise RuntimeError("Base pilot branch is required.")
    if not source_branch and not pr_number:
        raise RuntimeError("Sub repo branch or PR number is required.")

    logger.set_summary(
        work_dir=str(work_dir),
        pilot_checkout=str(pilot_checkout),
        pilot_base_branch=pilot_base_branch,
        sub_repo=sub_repo_name,
        sub_repo_path=sub_repo_path_text,
        repos_key=repos_key,
        source_branch=source_branch,
        pr_number=pr_number,
        restore_after=restore_after,
        prepare_only=prepare_only,
    )

    original_pilot_branch = ""
    original_sub_branch = ""
    pilot_branch = ""
    sub_branch = ""
    selected_sub_base: Dict[str, str] = {}
    try:
        logger.progress("Preparing reusable pilot checkout", 3)
        pilot_repo = _ensure_clone(pilot_repo_url, pilot_checkout, logger)
        original_pilot_branch = _current_branch(pilot_repo)
        original_pilot_sha = _current_sha(pilot_repo)
        logger.set_summary(original_pilot_branch=original_pilot_branch, original_pilot_sha=original_pilot_sha)

        if reset_cache:
            logger.progress("Restoring pilot checkout before run", 8)
            _git(pilot_repo, "reset", "--hard", logger=logger, step_name="Pilot: reset hard")
        if not _is_clean_except_untracked(pilot_repo, ("src/",)):
            raise RuntimeError("Pilot checkout has uncommitted changes outside src/. Enable reset cache or clean it first.")

        logger.progress(f"Checking out base pilot branch {pilot_base_branch}", 14)
        _checkout_base(pilot_repo, pilot_remote, pilot_base_branch, logger, "Pilot")
        pilot_base_sha = _current_sha(pilot_repo)
        pilot_remote_url = _remote_url(pilot_repo, pilot_remote)
        repos_file = pilot_repo / "autoware.repos"
        if not repos_file.is_file():
            raise RuntimeError(f"Missing autoware.repos in pilot checkout: {repos_file}")
        repos_entry = _repos_entry(repos_file, repos_key)
        repos_url = str(repos_entry.get("url") or "").strip()
        if not sub_repo_path_text:
            sub_repo_path_text = f"src/{repos_key}"
        if not upstream_url:
            upstream_url = str(sub_defaults.get("upstream_url") or repos_url).strip()
        if not upstream_repo:
            upstream_repo = str(sub_defaults.get("upstream_repo") or _github_repo_from_url(upstream_url)).strip()
        logger.set_summary(
            sub_repo_path=sub_repo_path_text,
            repos_key=repos_key,
            repos_url=repos_url,
            upstream_url=upstream_url,
            upstream_repo=upstream_repo,
            repos_version=str(repos_entry.get("version") or "").strip(),
            repos_branch=str(repos_entry.get("branch") or "").strip(),
        )

        sub_repo = (pilot_repo / sub_repo_path_text).resolve()
        if run_vcs:
            logger.progress("Updating source workspace with vcs import --debug", 25)
            _run_vcs_import(pilot_repo, repos_file, logger)
        logger.progress("Syncing selected sub repo checkout", 25)
        selected_sub_base = _sync_selected_sub_repo(
            sub_repo=sub_repo,
            repos_entry=repos_entry,
            fallback_url=upstream_url,
            logger=logger,
        )
        logger.set_summary(selected_sub_base=selected_sub_base)

        if not _is_git_repo(sub_repo):
            raise RuntimeError(f"Sub repo checkout is missing or not a git repo: {sub_repo}")
        original_sub_branch = _current_branch(sub_repo)
        original_sub_sha = _current_sha(sub_repo)
        logger.set_summary(original_sub_branch=original_sub_branch, original_sub_sha=original_sub_sha)
        if reset_cache:
            logger.progress("Restoring sub repo checkout before run", 38)
            _git(sub_repo, "reset", "--hard", logger=logger, step_name="Sub repo: reset hard")
            _git(sub_repo, "clean", "-fd", logger=logger, step_name="Sub repo: clean untracked")
        selected_sub_base_sha = selected_sub_base.get("sha", "")
        if selected_sub_base_sha and _current_sha(sub_repo) != selected_sub_base_sha:
            raise RuntimeError(
                "Selected sub repo base changed before patching: "
                f"expected {selected_sub_base_sha}, got {_current_sha(sub_repo)}"
            )
        if not _is_clean(sub_repo):
            raise RuntimeError("Sub repo checkout has uncommitted changes. Enable reset cache or clean it first.")
        _ensure_git_commit_identity(pilot_repo, logger, "Pilot")
        _ensure_git_commit_identity(sub_repo, logger, "Sub repo")

        logger.progress("Loading source branch metadata", 45)
        pr_info = _read_pr_info(pr_number, upstream_repo, logger) if pr_number else {}
        source_label = f"pr{pr_number}" if pr_number else source_branch
        date = dt.date.today().strftime("%Y%m%d")
        explicit_sub_branch = str(parameters.get("test_sub_branch") or "").strip()
        explicit_pilot_branch = str(parameters.get("test_pilot_branch") or "").strip()
        sub_branch = explicit_sub_branch or make_branch_name(
            branch_prefix,
            sub_repo_name,
            source_label,
            date,
            _current_sha(sub_repo),
        )
        pilot_branch = explicit_pilot_branch or make_branch_name(
            branch_prefix,
            "pilot-auto",
            source_label,
            date,
            pilot_base_sha,
        )
        if not explicit_sub_branch:
            sub_branch = _unique_branch_name(sub_repo, sub_remote, sub_branch)
        if not explicit_pilot_branch:
            pilot_branch = _unique_branch_name(pilot_repo, pilot_remote, pilot_branch)
        logger.set_summary(
            pr_info=pr_info,
            pilot_remote_url=pilot_remote_url,
            pilot_base_sha=pilot_base_sha,
            sub_base_sha=selected_sub_base_sha or _current_sha(sub_repo),
            sub_branch=sub_branch,
            pilot_branch=pilot_branch,
        )

        if explicit_sub_branch and _branch_exists(sub_repo, sub_branch):
            raise RuntimeError(f"Local sub repo branch already exists: {sub_branch}")
        if explicit_pilot_branch and _branch_exists(pilot_repo, pilot_branch):
            raise RuntimeError(f"Local pilot branch already exists: {pilot_branch}")
        if explicit_sub_branch and _remote_branch_exists(sub_repo, sub_remote, sub_branch):
            raise RuntimeError(f"Remote sub repo branch already exists: {sub_branch}")
        if explicit_pilot_branch and _remote_branch_exists(pilot_repo, pilot_remote, pilot_branch):
            raise RuntimeError(f"Remote pilot branch already exists: {pilot_branch}")

        logger.progress(f"Creating sub repo test branch {sub_branch}", 55)
        _git(sub_repo, "checkout", "-b", sub_branch, logger=logger, step_name="Sub repo: create test branch")
        sub_base_branch = str(pr_info.get("baseRefName") or source_branch or "HEAD").strip()
        source_refs = _fetch_source_ref(
            sub_repo,
            source_branch=source_branch,
            pr_number=pr_number,
            base_branch=sub_base_branch,
            upstream_url=upstream_url,
            logger=logger,
        )
        patch_file = work_dir / f"{sub_repo_name}_{source_label}_{int(time.time())}.patch"
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        changed_files = _apply_source_diff(
            sub_repo,
            head_ref=source_refs["head_ref"],
            base_ref=source_refs["base_ref"],
            patch_file=patch_file,
            logger=logger,
        )
        if not changed_files:
            raise RuntimeError("Source diff produced no staged changes.")
        _git(sub_repo, "commit", "-m", f"test: apply {sub_repo_name} {source_refs['source_label']}", logger=logger, step_name="Sub repo: commit test diff")
        sub_new_sha = _current_sha(sub_repo)
        logger.set_summary(changed_files=changed_files, sub_new_sha=sub_new_sha, patch_file=str(patch_file))

        logger.progress(f"Creating pilot branch {pilot_branch}", 74)
        _git(pilot_repo, "checkout", "-b", pilot_branch, logger=logger, step_name="Pilot: create test branch")
        _update_repos_file(repos_file, repos_key, sub_branch, sub_new_sha)
        _git(pilot_repo, "add", "autoware.repos", logger=logger, step_name="Pilot: stage autoware.repos")
        repos_diff = _git(pilot_repo, "diff", "--cached", "--", "autoware.repos", logger=logger, step_name="Pilot: autoware.repos diff")
        _git(pilot_repo, "commit", "-m", f"test: evaluate {sub_repo_name} {source_refs['source_label']}", logger=logger, step_name="Pilot: commit repos update")
        pilot_new_sha = _current_sha(pilot_repo)
        sub_remote_url = _remote_url(sub_repo, sub_remote)
        branch_urls = {
            "sub_repo": _github_branch_url(sub_remote_url, sub_branch),
            "pilot": _github_branch_url(pilot_remote_url, pilot_branch),
        }
        diff_urls = {
            "sub_repo": _github_compare_url(sub_remote_url, selected_sub_base.get("sha", ""), sub_branch),
            "pilot": _github_compare_url(pilot_remote_url, pilot_base_sha, pilot_branch),
        }
        logger.set_summary(
            pilot_new_sha=pilot_new_sha,
            repos_diff=repos_diff[-12000:],
            branch_urls=branch_urls,
            diff_urls=diff_urls,
            push_commands=[
                f"git -C {sub_repo} push -u {sub_remote} {sub_branch}",
                f"git -C {pilot_repo} push -u {pilot_remote} {pilot_branch}",
            ],
        )

        if not prepare_only:
            logger.progress("Pushing prepared test branches", 86)
            _git(sub_repo, "push", "-u", sub_remote, sub_branch, logger=logger, step_name="Sub repo: push test branch")
            _git(pilot_repo, "push", "-u", pilot_remote, pilot_branch, logger=logger, step_name="Pilot: push test branch")
            logger.set_summary(push_status="completed")

        logger.progress("Restoring reusable checkout", 94)
        if restore_after:
            _reset_repo(sub_repo, original_sub_branch or original_sub_sha, logger, "Sub repo restore")
            _reset_repo(pilot_repo, pilot_base_branch, logger, "Pilot restore", clean=False)
        logger.set_summary(
            restore_status="completed" if restore_after else "skipped",
            final_sub_head=_current_sha(sub_repo),
            final_pilot_head=_current_sha(pilot_repo),
        )
        logger.progress("Git test branch preparation complete", 100)
        return logger.summary
    except Exception as exc:
        logger.log(f"Failed: {exc}")
        logger.set_summary(error=str(exc))
        if restore_after:
            try:
                if sub_branch and pilot_checkout.exists():
                    sub_repo = (pilot_checkout / sub_repo_path_text).resolve()
                    restore_target = original_sub_branch or original_sub_sha
                    if _is_git_repo(sub_repo) and restore_target:
                        _reset_repo(sub_repo, restore_target, logger, "Sub repo restore after failure")
                if _is_git_repo(pilot_checkout):
                    target = pilot_base_branch or original_pilot_branch
                    if target:
                        _reset_repo(pilot_checkout, target, logger, "Pilot restore after failure", clean=False)
                logger.set_summary(restore_status="completed_after_failure")
            except Exception as restore_exc:
                logger.log(f"Restore after failure also failed: {restore_exc}")
                logger.set_summary(restore_status=f"failed: {restore_exc}")
        raise
