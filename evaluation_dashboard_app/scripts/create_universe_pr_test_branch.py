#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path


UPSTREAM_REPO = "autowarefoundation/autoware_universe"
UPSTREAM_URL = "https://github.com/autowarefoundation/autoware_universe.git"
UNIVERSE_PATH = Path("src/autoware/universe")
REPOS_PATH = Path("autoware.repos")
DEFAULT_BRANCH_PREFIX = "evaluator-dashboard"
SELF_RELATIVE_PATH = Path("dev/create_universe_pr_test_branch.py")
DEFAULT_GIT_USER_NAME = "Evaluator Dashboard"
DEFAULT_GIT_USER_EMAIL = "evaluator-dashboard@localhost"


class CommandError(RuntimeError):
    pass


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cwd_text = f" (cwd: {cwd})" if cwd else ""
    print(f"$ {' '.join(cmd)}{cwd_text}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture)

    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
        raise CommandError(f"Command failed: {' '.join(cmd)}")

    return result


def run_stdout_to_file(cmd: list[str], output_file: Path, *, cwd: Path | None = None) -> None:
    cwd_text = f" (cwd: {cwd})" if cwd else ""
    print(f"$ {' '.join(cmd)} > {output_file}{cwd_text}")
    with output_file.open("wb") as output:
        result = subprocess.run(cmd, cwd=cwd, stdout=output, stderr=subprocess.PIPE)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.decode(errors="replace"), file=sys.stderr)
        raise CommandError(f"Command failed: {' '.join(cmd)}")


def git_output(repo: Path, *args: str) -> str:
    return run(["git", *args], cwd=repo).stdout.strip()


def git_output_optional(repo: Path, *args: str) -> str:
    result = run(["git", *args], cwd=repo, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def ensure_git_commit_identity(repo: Path, label: str) -> None:
    name = git_output_optional(repo, "config", "user.name")
    email = git_output_optional(repo, "config", "user.email")
    if not name:
        run(["git", "config", "--local", "user.name", DEFAULT_GIT_USER_NAME], cwd=repo)
        name = DEFAULT_GIT_USER_NAME
    if not email:
        run(["git", "config", "--local", "user.email", DEFAULT_GIT_USER_EMAIL], cwd=repo)
        email = DEFAULT_GIT_USER_EMAIL
    print(f"{label} commit identity: {name} <{email}>")


def is_clean(repo: Path) -> bool:
    return git_output(repo, "status", "--porcelain") == ""


def dirty_paths(repo: Path) -> list[str]:
    paths = []
    for line in git_output(repo, "status", "--porcelain").splitlines():
        if not line:
            continue
        paths.append(line[3:])
    return paths


def has_only_allowed_dirty_paths(repo: Path, allowed_paths: set[Path]) -> bool:
    allowed = {path.as_posix() for path in allowed_paths}
    return all(path in allowed for path in dirty_paths(repo))


def current_branch(repo: Path) -> str:
    return git_output(repo, "branch", "--show-current")


def current_sha(repo: Path) -> str:
    return git_output(repo, "rev-parse", "HEAD")


def short_sha(sha: str) -> str:
    return sha[:12]


def local_branch_exists(repo: Path, branch: str) -> bool:
    result = run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=repo, check=False)
    if result.returncode not in (0, 1):
        raise CommandError(f"Could not check local branch: {branch}")
    return result.returncode == 0


def remote_branch_exists(repo: Path, remote: str, branch: str) -> bool:
    result = run(["git", "ls-remote", "--exit-code", "--heads", remote, branch], cwd=repo, check=False)
    if result.returncode not in (0, 2):
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
        raise CommandError(f"Could not check remote branch: {remote}/{branch}")
    return result.returncode == 0


def remote_url(repo: Path, remote: str) -> str:
    return git_output(repo, "remote", "get-url", remote)


def get_pr_info(pr_number: int) -> dict[str, object]:
    result = run(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            UPSTREAM_REPO,
            "--json",
            "title,author,baseRefName,headRefName,headRefOid,url",
        ],
    )
    return json.loads(result.stdout)


def confirm(prompt: str, *, expected: str = "yes") -> bool:
    print()
    answer = input(f"{prompt}\nType '{expected}' to continue: ").strip()
    return answer == expected


def make_branch_name(prefix: str, repo_label: str, pr_number: int, date: str, base_sha: str) -> str:
    return f"{prefix}/{repo_label}-pr{pr_number}-{date}-{short_sha(base_sha)}"


def temporary_ref_name(pr_number: int, suffix: str) -> str:
    safe_suffix = re.sub(r"[^A-Za-z0-9._/-]+", "-", suffix).strip("/-")
    return f"refs/tmp/evaluator-dashboard/pr-{pr_number}/{safe_suffix}"


def delete_ref(repo: Path, ref_name: str) -> None:
    run(["git", "update-ref", "-d", ref_name], cwd=repo, check=False)


def apply_pr_only_diff(
    *,
    repo: Path,
    pr_number: int,
    base_ref_name: str,
    patch_file: Path,
) -> str:
    pr_ref = temporary_ref_name(pr_number, "head")
    base_ref = temporary_ref_name(pr_number, f"base-{base_ref_name}")

    try:
        run(["git", "fetch", UPSTREAM_URL, f"+pull/{pr_number}/head:{pr_ref}"], cwd=repo)
        run(["git", "fetch", UPSTREAM_URL, f"+{base_ref_name}:{base_ref}"], cwd=repo)
        merge_base = git_output(repo, "merge-base", base_ref, pr_ref)
        run_stdout_to_file(["git", "diff", "--binary", merge_base, pr_ref], patch_file, cwd=repo)
        run(["git", "apply", "--3way", "--index", str(patch_file)], cwd=repo)
        return git_output(repo, "diff", "--cached", "--name-only")
    finally:
        delete_ref(repo, pr_ref)
        delete_ref(repo, base_ref)


def update_autoware_repos(repos_file: Path, universe_branch: str, universe_sha: str) -> None:
    lines = repos_file.read_text().splitlines(keepends=True)

    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "autoware/universe:" and line.startswith("  "):
            start = idx
            break

    if start is None:
        raise RuntimeError("Could not find 'autoware/universe' in autoware.repos")

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            end = idx
            break

    block = lines[start:end]
    branch_idx = None
    version_idx = None

    for offset, line in enumerate(block):
        if line.startswith("    branch:"):
            branch_idx = start + offset
        elif line.startswith("    version:"):
            version_idx = start + offset

    if version_idx is None:
        raise RuntimeError("Could not find 'version:' under 'autoware/universe' in autoware.repos")

    lines[version_idx] = f"    version: {universe_sha}\n"

    if branch_idx is None:
        lines.insert(version_idx, f"    branch: {universe_branch}\n")
    else:
        lines[branch_idx] = f"    branch: {universe_branch}\n"

    repos_file.write_text("".join(lines))


def print_plan(
    *,
    pr_number: int,
    pr: dict[str, object],
    universe_remote: str,
    pilot_remote: str,
    universe_base_branch: str,
    universe_base_sha: str,
    pilot_base_branch: str,
    pilot_base_sha: str,
    universe_branch: str,
    pilot_branch: str,
) -> None:
    author = pr.get("author") or {}
    author_login = author.get("login", "unknown") if isinstance(author, dict) else "unknown"

    print("\n========== PLAN ==========")
    print(f"Upstream PR       : #{pr_number} {pr.get('url', '')}")
    print(f"PR title          : {pr.get('title', '')}")
    print(f"PR author         : {author_login}")
    print(f"PR base/head      : {pr.get('baseRefName', '')} <- {pr.get('headRefName', '')}")
    print(f"PR head SHA       : {pr.get('headRefOid', '')}")
    print()
    print(f"Universe remote   : {universe_remote}")
    print(f"Universe base     : {universe_base_branch} @ {universe_base_sha}")
    print(f"Universe branch   : {universe_branch}")
    print()
    print(f"Pilot remote      : {pilot_remote}")
    print(f"Pilot base        : {pilot_base_branch} @ {pilot_base_sha}")
    print(f"Pilot branch      : {pilot_branch}")
    print("==========================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a test autoware_universe branch with an upstream PR merged on top, "
            "then create a pilot-auto branch that points autoware.repos to it."
        )
    )
    parser.add_argument("--pr", type=int, required=True, help="autowarefoundation/autoware_universe PR number")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit before local changes")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="create local branches and commits, but do not ask to push them",
    )
    parser.add_argument(
        "--branch-prefix",
        default=DEFAULT_BRANCH_PREFIX,
        help=f"branch namespace prefix (default: {DEFAULT_BRANCH_PREFIX})",
    )
    parser.add_argument("--universe-branch", help="override the generated autoware_universe branch name")
    parser.add_argument("--pilot-branch", help="override the generated pilot-auto branch name")
    parser.add_argument("--universe-remote", default="origin", help="remote name for src/autoware/universe pushes")
    parser.add_argument("--pilot-remote", default="origin", help="remote name for pilot-auto pushes")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="pilot-auto repository root (default: current directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    universe_repo = repo_root / UNIVERSE_PATH
    repos_file = repo_root / REPOS_PATH

    if not (repo_root / ".git").exists():
        raise RuntimeError(f"Not a git repository root: {repo_root}")
    if not universe_repo.exists():
        raise RuntimeError(f"Missing sub repo: {universe_repo}")
    if not repos_file.exists():
        raise RuntimeError(f"Missing repos file: {repos_file}")

    allowed_pilot_dirty_paths = {SELF_RELATIVE_PATH}

    if not has_only_allowed_dirty_paths(repo_root, allowed_pilot_dirty_paths):
        raise RuntimeError("pilot-auto has uncommitted changes; commit or stash them first")
    if not is_clean(universe_repo):
        raise RuntimeError("src/autoware/universe has uncommitted changes; commit or stash them first")
    ensure_git_commit_identity(repo_root, "Pilot")
    ensure_git_commit_identity(universe_repo, "Universe")

    pr = get_pr_info(args.pr)
    date = dt.date.today().strftime("%Y%m%d")

    universe_base_branch = current_branch(universe_repo)
    universe_base_sha = current_sha(universe_repo)
    pilot_base_branch = current_branch(repo_root)
    pilot_base_sha = current_sha(repo_root)

    universe_branch = args.universe_branch or make_branch_name(
        args.branch_prefix, "universe", args.pr, date, universe_base_sha
    )
    pilot_branch = args.pilot_branch or make_branch_name(args.branch_prefix, "pilot-auto", args.pr, date, pilot_base_sha)

    universe_remote_url = remote_url(universe_repo, args.universe_remote)
    pilot_remote_url = remote_url(repo_root, args.pilot_remote)

    for repo, branch, label in (
        (universe_repo, universe_branch, "Universe"),
        (repo_root, pilot_branch, "Pilot"),
    ):
        if local_branch_exists(repo, branch):
            raise RuntimeError(f"{label} local branch already exists: {branch}")

    if remote_branch_exists(universe_repo, args.universe_remote, universe_branch):
        raise RuntimeError(f"Universe remote branch already exists: {universe_branch}")
    if remote_branch_exists(repo_root, args.pilot_remote, pilot_branch):
        raise RuntimeError(f"Pilot remote branch already exists: {pilot_branch}")

    print_plan(
        pr_number=args.pr,
        pr=pr,
        universe_remote=universe_remote_url,
        pilot_remote=pilot_remote_url,
        universe_base_branch=universe_base_branch,
        universe_base_sha=universe_base_sha,
        pilot_base_branch=pilot_base_branch,
        pilot_base_sha=pilot_base_sha,
        universe_branch=universe_branch,
        pilot_branch=pilot_branch,
    )

    if args.dry_run:
        print("\nDRY RUN COMPLETE: no local branches created and nothing pushed.")
        return 0

    if not confirm("Create local test branches and apply the upstream PR?", expected="create branches"):
        print("Aborted before local changes.")
        return 1

    base_ref_name = str(pr.get("baseRefName") or "")
    if not base_ref_name:
        raise RuntimeError("Could not determine PR base branch")

    run(["git", "checkout", "-b", universe_branch], cwd=universe_repo)
    patch_file = Path("/tmp") / f"autoware_universe_pr_{args.pr}.patch"
    changed_files = apply_pr_only_diff(
        repo=universe_repo,
        pr_number=args.pr,
        base_ref_name=base_ref_name,
        patch_file=patch_file,
    )
    if not changed_files:
        raise RuntimeError("PR-only patch produced no staged changes")
    print("\nPR-only files staged in Universe repo:")
    print(changed_files)
    run(["git", "commit", "-m", f"test: apply autoware_universe PR #{args.pr}"], cwd=universe_repo)
    universe_new_sha = current_sha(universe_repo)

    run(["git", "checkout", "-b", pilot_branch], cwd=repo_root)
    update_autoware_repos(repos_file, universe_branch, universe_new_sha)
    run(["git", "add", str(REPOS_PATH)], cwd=repo_root)
    run(["git", "commit", "-m", f"test: evaluate autoware_universe PR #{args.pr}"], cwd=repo_root)
    pilot_new_sha = current_sha(repo_root)

    print("\n========== READY TO PUSH ==========")
    print(f"Universe push : git push -u {args.universe_remote} {universe_branch}")
    print(f"Universe SHA  : {universe_new_sha}")
    print(f"Pilot push    : git push -u {args.pilot_remote} {pilot_branch}")
    print(f"Pilot SHA     : {pilot_new_sha}")
    print("autoware.repos: autoware/universe branch/version now point to the Universe test branch above")
    print("===================================")

    if args.prepare_only:
        print("\nPREPARE ONLY COMPLETE: local branches are ready; nothing pushed.")
        return 0

    if not confirm("This will push branches to the configured remotes.", expected="push branches"):
        print("Aborted before push. Local branches and commits were kept.")
        return 1

    run(["git", "push", "-u", args.universe_remote, universe_branch], cwd=universe_repo, capture=False)
    run(["git", "push", "-u", args.pilot_remote, pilot_branch], cwd=repo_root, capture=False)

    print("\n========== DONE ==========")
    print(f"Universe branch : {universe_branch}")
    print(f"Universe SHA    : {universe_new_sha}")
    print(f"Pilot branch    : {pilot_branch}")
    print(f"Pilot SHA       : {pilot_new_sha}")
    print("==========================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CommandError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
