"""Local evaluator debug workflow helpers.

The worker runs these commands on the same machine/container where the RQ worker
is running. For host Docker builds from a dashboard container, mount the host
Docker socket and the pilot checkout path into the worker container.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import posixpath
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


DEFAULT_REPO_URL = os.environ.get(
    "LOCAL_EVALUATOR_REPO_URL",
    "git@github.com:tier4/pilot-auto.x2.git",
)
DEFAULT_WORK_ROOT = Path(os.environ.get("LOCAL_EVALUATOR_WORK_ROOT", "/tmp/webauto-local-evaluator"))
DEFAULT_CHECKOUT_ROOT = Path(
    os.environ.get("LOCAL_EVALUATOR_CHECKOUT_ROOT", "/tmp/webauto-local-evaluator/checkouts")
)
DEFAULT_CONTAINER_RUNTIME_ROOT = os.environ.get(
    "LOCAL_EVALUATOR_CONTAINER_RUNTIME_ROOT",
    "/home/leigu/pilot-auto.x2.v4.4_e2e",
)
DEFAULT_SIM_WORK_DIR = os.environ.get("LOCAL_EVALUATOR_SIM_WORK_DIR", "/tmp/webauto-local-sim")
DEFAULT_SIM_ASSET_DIR = os.environ.get("LOCAL_EVALUATOR_SIM_ASSET_DIR", "/tmp/webauto-local-sim/assets")
DEFAULT_ROS_DISTRO = os.environ.get("LOCAL_EVALUATOR_ROS_DISTRO", "humble")
DEFAULT_SIMULATION_NAME = os.environ.get("LOCAL_EVALUATOR_SIMULATION_NAME", "planning_sim_v2_j6_gen2")
DEFAULT_TIMEOUT = os.environ.get("LOCAL_EVALUATOR_TIMEOUT", "45m")
DEFAULT_WEBAUTO_SCENARIO_COMMAND = os.environ.get(
    "LOCAL_EVALUATOR_WEBAUTO_COMMAND",
    (
        "webauto ci scenario run --project-id x2_dev "
        "--scenario-id 78b9286c-a9d5-4293-a0eb-7ff2746168a0 "
        "--scenario-version-id 2 "
        "--scenario-parameters "
        "'t4_dataset_id=4ec4c905-6521-40de-99dd-8ceae04fa348,t4_dataset_version_id=1' "
        "--simulation-name perception"
    ),
)
ERROR_PATTERN = re.compile(r"(error|exception|failed|fatal|traceback)", re.IGNORECASE)


def sanitize_name(value: object, fallback: str = "local") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip(".-")
    text = re.sub(r"-+", "-", text)
    return (text or fallback)[:80]


def default_image_name(branch: object) -> str:
    return f"pilot-auto:evaluation-{sanitize_name(branch, 'local')}"


def default_checkout_path(branch: object) -> Path:
    return DEFAULT_CHECKOUT_ROOT / sanitize_name(branch, "local")


def tail_text(path: Path, *, max_bytes: int = 64 * 1024) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-max_bytes:].decode("utf-8", errors="replace")


class LocalEvaluatorJob:
    def __init__(
        self,
        *,
        task_id: str,
        parameters: Dict[str, Any],
        append_log: Callable[[str, str], Any],
        update_progress: Callable[..., Any],
        update_summary: Callable[[str, Dict[str, Any]], Any],
    ) -> None:
        self.task_id = task_id
        self.parameters = parameters
        self.append_log = append_log
        self.update_progress = update_progress
        self.update_summary = update_summary
        self.started = time.monotonic()
        branch = parameters.get("branch") or parameters.get("image_name") or "local"
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = DEFAULT_WORK_ROOT / "runs" / f"{stamp}_{sanitize_name(branch)}_{task_id[:8]}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "local_evaluator.log"
        self.summary: Dict[str, Any] = {
            "job": "local_evaluator_debug",
            "mode": parameters.get("mode", "build_and_test"),
            "run_dir": str(self.run_dir),
            "log_path": str(self.log_path),
            "steps": [],
        }
        self._save_summary()

    def log(self, message: str, *, important: bool = True) -> None:
        line = str(message)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")
        if important:
            self.append_log(self.task_id, line)

    def progress(self, message: str, pct: float) -> None:
        self.update_progress(self.task_id, message=message, pct=pct)
        self.log(message)

    def step(self, name: str, status: str, **details: Any) -> None:
        payload = {
            "name": name,
            "status": status,
            "elapsed_seconds": round(time.monotonic() - self.started, 2),
        }
        payload.update({k: v for k, v in details.items() if v not in (None, "", [], {})})
        self.summary.setdefault("steps", []).append(payload)
        self._save_summary()

    def set_summary(self, **updates: Any) -> None:
        self.summary.update(updates)
        self._save_summary()

    def _save_summary(self) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(self.summary, indent=2, default=str),
            encoding="utf-8",
        )
        self.update_summary(self.task_id, self.summary)

    def run_command(
        self,
        cmd: List[str],
        *,
        cwd: Path,
        step_name: str,
        pct_start: float,
        pct_end: float,
        env: Optional[Dict[str, str]] = None,
        stdin_path: Optional[Path] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        started = time.monotonic()
        self.log(f"$ {' '.join(cmd)} (cwd: {cwd})")
        output_lines: List[str] = []
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        stdin_handle = stdin_path.open("rb") if stdin_path else None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdin=stdin_handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=proc_env,
            )
            assert proc.stdout is not None
            last_progress = started
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                output_lines.append(line)
                self.log(line, important=bool(ERROR_PATTERN.search(line)))
                if time.monotonic() - last_progress > 20:
                    pct = pct_start + min(0.9, (time.monotonic() - started) / 3600.0) * (pct_end - pct_start)
                    self.update_progress(self.task_id, message=f"{step_name}: running", pct=min(pct_end, pct))
                    last_progress = time.monotonic()
            returncode = proc.wait()
        finally:
            if stdin_handle:
                stdin_handle.close()
        elapsed = round(time.monotonic() - started, 2)
        status = "completed" if returncode == 0 else "failed"
        self.step(step_name, status, command=cmd, cwd=str(cwd), returncode=returncode, elapsed_seconds=elapsed)
        self.update_progress(self.task_id, message=f"{step_name}: {status}", pct=pct_end)
        if check and returncode != 0:
            raise RuntimeError(f"{step_name} failed with exit code {returncode}. Full log: {self.log_path}")
        return subprocess.CompletedProcess(cmd, returncode, "\n".join(output_lines), "")


def _run_quiet(cmd: List[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)


def _run_docker(cmd: List[str], *, timeout: Optional[int] = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(["docker", *cmd], text=True, capture_output=True, timeout=timeout)
    except OSError as exc:
        return subprocess.CompletedProcess(["docker", *cmd], 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(["docker", *cmd], 124, stdout, stderr or "Command timed out")


def docker_error(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or f"docker exited with code {result.returncode}").strip()


def _git_has_changes(path: Path, *, allowed_untracked_prefixes: tuple[str, ...] = ()) -> bool:
    result = _run_quiet(["git", "status", "--porcelain"], cwd=path)
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        if not line:
            continue
        if line.startswith("?? "):
            rel_path = line[3:]
            if any(rel_path == prefix.rstrip("/") or rel_path.startswith(prefix) for prefix in allowed_untracked_prefixes):
                continue
        return True
    return False


def find_repos_file(checkout: Path, configured: object = "") -> Optional[Path]:
    configured_text = str(configured or "").strip()
    if configured_text:
        configured_path = Path(configured_text).expanduser()
        if not configured_path.is_absolute():
            configured_path = checkout / configured_path
        return configured_path if configured_path.is_file() else None

    candidates = [
        checkout / "autoware.repos",
        checkout / "src" / "autoware.repos",
    ]
    candidates.extend(sorted(checkout.glob("*.repos")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def import_src_repos(job: LocalEvaluatorJob, checkout: Path) -> Path:
    repos_file = find_repos_file(checkout, job.parameters.get("repos_file"))
    if repos_file is None:
        raise RuntimeError(f"Could not find autoware.repos under {checkout}. Cannot populate src.")
    src_dir = checkout / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    job.progress(f"Importing src repositories from {repos_file.name}", 18)
    job.run_command(
        ["vcs", "import", "src"],
        cwd=checkout,
        step_name="vcs import src",
        pct_start=18,
        pct_end=22,
        stdin_path=repos_file,
    )
    job.set_summary(repos_file=str(repos_file), src_path=str(src_dir))
    return src_dir


def prepare_checkout(job: LocalEvaluatorJob) -> Path:
    params = job.parameters
    branch = str(params.get("branch") or "").strip()
    checkout = Path(str(params.get("checkout_path") or default_checkout_path(branch))).expanduser()
    repo_url = str(params.get("repo_url") or DEFAULT_REPO_URL).strip()
    clean_checkout = bool(params.get("clean_checkout", False))
    allow_dirty = bool(params.get("allow_dirty_checkout", False))

    if not branch:
        if checkout.exists():
            job.step("checkout", "skipped", checkout_path=str(checkout), reason="No branch provided")
            return checkout
        raise RuntimeError("Branch is required when checkout_path does not already exist.")

    checkout.parent.mkdir(parents=True, exist_ok=True)
    if checkout.exists() and not (checkout / ".git").exists():
        raise RuntimeError(f"Checkout path exists but is not a git repo: {checkout}")

    if not checkout.exists():
        job.progress(f"Cloning {branch}", 5)
        job.run_command(
            ["git", "clone", "--branch", branch, "--single-branch", repo_url, str(checkout)],
            cwd=checkout.parent,
            step_name="git clone",
            pct_start=5,
            pct_end=18,
        )
    else:
        if clean_checkout and _git_has_changes(checkout, allowed_untracked_prefixes=("src/",)):
            job.run_command(["git", "reset", "--hard"], cwd=checkout, step_name="git reset", pct_start=5, pct_end=7)
            job.run_command(["git", "clean", "-fd", "-e", "src/"], cwd=checkout, step_name="git clean", pct_start=7, pct_end=9)
        elif not allow_dirty and _git_has_changes(checkout, allowed_untracked_prefixes=("src/",)):
            raise RuntimeError(
                f"Checkout has local changes: {checkout}. Enable allow_dirty_checkout or choose a clean path."
            )
        job.progress(f"Fetching {branch}", 5)
        job.run_command(["git", "fetch", "origin", branch], cwd=checkout, step_name="git fetch", pct_start=5, pct_end=12)
        fetched = _run_quiet(["git", "rev-parse", "--verify", f"origin/{branch}"], cwd=checkout)
        if fetched.returncode == 0:
            job.run_command(
                ["git", "checkout", "-B", branch, f"origin/{branch}"],
                cwd=checkout,
                step_name="git checkout",
                pct_start=12,
                pct_end=18,
            )
        else:
            job.run_command(["git", "checkout", branch], cwd=checkout, step_name="git checkout", pct_start=12, pct_end=18)

    sha = _run_quiet(["git", "rev-parse", "HEAD"], cwd=checkout).stdout.strip()
    job.set_summary(branch=branch, checkout_path=str(checkout), git_sha=sha)
    return checkout


def build_image(job: LocalEvaluatorJob, checkout: Path) -> str:
    params = job.parameters
    image_name = str(params.get("image_name") or default_image_name(params.get("branch"))).strip()
    build_dir = checkout / "docker-multi-stage"
    if not build_dir.is_dir():
        raise RuntimeError(f"Build directory not found: {build_dir}")
    ros_distro = str(params.get("ros_distro") or DEFAULT_ROS_DISTRO).strip()
    job.progress("Starting Docker build", 20)
    job.run_command(
        ["./build-main.bash", "evaluation", "--yes", "--use-ghcr", "--ros-distro", ros_distro],
        cwd=build_dir,
        step_name="docker build",
        pct_start=20,
        pct_end=70,
    )
    source_image = str(params.get("source_build_image") or "pilot-auto:evaluation").strip()
    if image_name and image_name != source_image:
        job.run_command(["docker", "tag", source_image, image_name], cwd=checkout, step_name="docker tag", pct_start=70, pct_end=74)
    job.set_summary(image_name=image_name, build_status="passed")
    return image_name


def list_scenarios(job: LocalEvaluatorJob) -> None:
    project_id = str(job.parameters.get("project_id") or "").strip()
    if not project_id or not bool(job.parameters.get("list_scenarios", False)):
        return
    page_size = str(int(job.parameters.get("scenario_page_size") or 50))
    job.run_command(
        ["webauto", "ci", "scenario", "list", "--project-id", project_id, "--page-size", page_size],
        cwd=job.run_dir,
        step_name="scenario list",
        pct_start=75,
        pct_end=78,
        check=False,
    )


def _has_cli_option(cmd: List[str], option: str) -> bool:
    prefix = f"{option}="
    return any(part == option or part.startswith(prefix) for part in cmd)


def build_webauto_command(
    raw_command: object,
    *,
    image_name: str,
    container_runtime_path: str,
    work_dir: str,
    asset_dir: str,
    timeout: str,
) -> List[str]:
    raw = str(raw_command or DEFAULT_WEBAUTO_SCENARIO_COMMAND).strip()
    if not raw:
        raw = DEFAULT_WEBAUTO_SCENARIO_COMMAND
    cmd = shlex.split(raw)
    if cmd[:4] != ["webauto", "ci", "scenario", "run"]:
        raise RuntimeError("Test command must start with: webauto ci scenario run")
    additions = [
        ("--docker-image", image_name),
        ("--container-runtime-path", container_runtime_path),
        ("--work-dir", work_dir),
        ("--asset-dir", asset_dir),
        ("--timeout", timeout),
    ]
    for option, value in additions:
        clean_value = str(value or "").strip()
        if clean_value and not _has_cli_option(cmd, option):
            cmd.extend([option, clean_value])
    return cmd


def run_scenario(job: LocalEvaluatorJob, image_name: str) -> str:
    params = job.parameters
    runtime_path = str(params.get("container_runtime_path") or DEFAULT_CONTAINER_RUNTIME_ROOT).strip()
    cmd = build_webauto_command(
        params.get("webauto_command") or DEFAULT_WEBAUTO_SCENARIO_COMMAND,
        image_name=image_name,
        container_runtime_path=runtime_path,
        work_dir=str(params.get("work_dir") or DEFAULT_SIM_WORK_DIR).strip(),
        asset_dir=str(params.get("asset_dir") or DEFAULT_SIM_ASSET_DIR).strip(),
        timeout=str(params.get("timeout") or DEFAULT_TIMEOUT).strip(),
    )
    list_scenarios(job)
    job.progress("Starting scenario run", 78)
    job.set_summary(webauto_command=" ".join(shlex.quote(part) for part in cmd))
    result = job.run_command(cmd, cwd=job.run_dir, step_name="scenario run", pct_start=78, pct_end=98, check=False)
    test_status = "passed" if result.returncode == 0 else "failed"
    job.set_summary(test_status=test_status, scenario_returncode=result.returncode)
    if result.returncode != 0:
        raise RuntimeError(f"Scenario failed with exit code {result.returncode}. Full log: {job.log_path}")
    return test_status


def commit_container(job: LocalEvaluatorJob) -> str:
    container = str(job.parameters.get("container_id") or "").strip()
    target_image = str(job.parameters.get("commit_image_name") or job.parameters.get("image_name") or "").strip()
    if not container or not target_image:
        raise RuntimeError("container_id and commit_image_name are required.")
    job.progress("Committing modified container", 20)
    job.run_command(
        ["docker", "commit", container, target_image],
        cwd=job.run_dir,
        step_name="docker commit",
        pct_start=20,
        pct_end=95,
    )
    job.set_summary(image_name=target_image, commit_status="completed")
    return target_image


def run_local_evaluator_debug(
    *,
    task_id: str,
    parameters: Dict[str, Any],
    append_log: Callable[[str, str], Any],
    update_progress: Callable[..., Any],
    update_summary: Callable[[str, Dict[str, Any]], Any],
) -> Dict[str, Any]:
    job = LocalEvaluatorJob(
        task_id=task_id,
        parameters=parameters,
        append_log=append_log,
        update_progress=update_progress,
        update_summary=update_summary,
    )
    mode = str(parameters.get("mode") or "build_and_test").strip()
    job.set_summary(mode=mode)
    image_name = str(parameters.get("image_name") or default_image_name(parameters.get("branch"))).strip()

    if mode == "commit_container":
        commit_container(job)
        job.progress("Local evaluator debug job completed", 100)
        return job.summary

    if mode in ("build", "build_and_test"):
        checkout = prepare_checkout(job)
        import_src_repos(job, checkout)
        image_name = build_image(job, checkout)
    elif mode == "test":
        if not image_name:
            raise RuntimeError("image_name is required for test-only mode.")
        job.set_summary(image_name=image_name)
    else:
        raise RuntimeError(f"Unsupported local evaluator mode: {mode}")

    if mode in ("test", "build_and_test"):
        run_scenario(job, image_name)
    else:
        job.set_summary(test_status="skipped")

    job.progress("Local evaluator debug job completed", 100)
    return job.summary


def docker_images(prefix: str = "pilot-auto") -> List[str]:
    result = _run_docker(["images", "--format", "{{.Repository}}:{{.Tag}}"])
    if result.returncode != 0:
        return []
    images = []
    for line in result.stdout.splitlines():
        image = line.strip()
        if image and image != "<none>:<none>" and image.startswith(prefix):
            images.append(image)
    return sorted(set(images))


def docker_containers() -> List[Dict[str, str]]:
    result = _run_docker(["ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"])
    if result.returncode != 0:
        return []
    containers: List[Dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            containers.append({"name": parts[0], "image": parts[1], "status": parts[2]})
    return containers


def default_debug_container_name(image_name: object) -> str:
    return f"eval-debug-{sanitize_name(image_name, 'image')}"


def start_debug_container(image_name: str, container_name: str) -> Dict[str, str]:
    image = str(image_name or "").strip()
    name = sanitize_name(container_name, default_debug_container_name(image))
    if not image:
        return {"ok": "false", "message": "Image name is required.", "container": name}
    result = _run_docker(["run", "-dit", "--name", name, image, "sleep", "infinity"])
    if result.returncode != 0:
        return {"ok": "false", "message": docker_error(result), "container": name}
    return {"ok": "true", "message": result.stdout.strip(), "container": name}


def list_container_files(container: str, root: str, *, max_depth: int = 4, limit: int = 200) -> Dict[str, Any]:
    container_name = str(container or "").strip()
    root_path = str(root or "/").strip() or "/"
    if not container_name:
        return {"ok": False, "message": "Container is required.", "files": []}
    result = _run_docker(
        [
            "exec",
            container_name,
            "find",
            root_path,
            "-maxdepth",
            str(max(1, int(max_depth or 1))),
            "-type",
            "f",
        ],
        timeout=30,
    )
    if result.returncode != 0:
        return {"ok": False, "message": docker_error(result), "files": []}
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"ok": True, "message": f"{len(files[:limit])} file(s)", "files": files[:limit]}


def read_container_file(container: str, file_path: str, *, max_bytes: int = 512 * 1024) -> Dict[str, Any]:
    container_name = str(container or "").strip()
    target = str(file_path or "").strip()
    if not container_name or not target:
        return {"ok": False, "message": "Container and file path are required.", "content": ""}
    result = _run_docker(["exec", container_name, "cat", "--", target], timeout=30)
    if result.returncode != 0:
        return {"ok": False, "message": docker_error(result), "content": ""}
    content = result.stdout
    truncated = False
    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) > max_bytes:
        content = encoded[:max_bytes].decode("utf-8", errors="replace")
        truncated = True
    return {"ok": True, "message": "Loaded file", "content": content, "truncated": truncated}


def write_container_file(container: str, file_path: str, content: str) -> Dict[str, str]:
    container_name = str(container or "").strip()
    target = str(file_path or "").strip()
    if not container_name or not target:
        return {"ok": "false", "message": "Container and file path are required."}
    parent = posixpath.dirname(target)
    if parent and parent != "/":
        mkdir_result = _run_docker(["exec", container_name, "mkdir", "-p", "--", parent], timeout=30)
        if mkdir_result.returncode != 0:
            return {"ok": "false", "message": docker_error(mkdir_result)}

    temp_root = DEFAULT_WORK_ROOT / "editor"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=temp_root, delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    try:
        result = _run_docker(["cp", str(temp_path), f"{container_name}:{target}"], timeout=60)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass
    if result.returncode != 0:
        return {"ok": "false", "message": docker_error(result)}
    return {"ok": "true", "message": f"Saved {target}"}


def exec_in_container(container: str, command: str, *, timeout_seconds: int = 120) -> Dict[str, Any]:
    container_name = str(container or "").strip()
    cmd = str(command or "").strip()
    if not container_name or not cmd:
        return {"ok": False, "message": "Container and command are required.", "output": "", "returncode": None}
    result = _run_docker(
        ["exec", container_name, "bash", "-lc", cmd],
        timeout=max(1, int(timeout_seconds or 1)),
    )
    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    return {
        "ok": result.returncode == 0,
        "message": "Command completed" if result.returncode == 0 else docker_error(result),
        "output": output,
        "returncode": result.returncode,
    }


def commit_debug_container(container: str, image_name: str) -> Dict[str, str]:
    container_name = str(container or "").strip()
    image = str(image_name or "").strip()
    if not container_name or not image:
        return {"ok": "false", "message": "Container and target image are required."}
    result = _run_docker(["commit", container_name, image], timeout=300)
    if result.returncode != 0:
        return {"ok": "false", "message": docker_error(result)}
    return {"ok": "true", "message": result.stdout.strip() or f"Committed {image}"}
