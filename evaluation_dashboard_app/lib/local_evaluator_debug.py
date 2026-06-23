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
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml


DEFAULT_REPO_URL = os.environ.get(
    "LOCAL_EVALUATOR_REPO_URL",
    "git@github.com:tier4/pilot-auto.x2.git",
)
DEFAULT_WORK_ROOT = Path(
    os.environ.get(
        "LOCAL_EVALUATOR_WORK_ROOT",
        str(Path(os.environ.get("EVAL_DASHBOARD_DATA_ROOT", "data")) / "local_evaluator_debug"),
    )
)
DEFAULT_CHECKOUT_ROOT = Path(
    os.environ.get("LOCAL_EVALUATOR_CHECKOUT_ROOT", str(DEFAULT_WORK_ROOT / "checkouts"))
)
LOCAL_EVALUATOR_CONTAINER_DATA_ROOT = os.environ.get("LOCAL_EVALUATOR_CONTAINER_DATA_ROOT", "/app/data")
LOCAL_EVALUATOR_HOST_DATA_ROOT = os.environ.get("LOCAL_EVALUATOR_HOST_DATA_ROOT", "")
LOCAL_EVALUATOR_HOST_WEBAUTO_ROOT = os.environ.get("LOCAL_EVALUATOR_HOST_WEBAUTO_ROOT", "")
DEFAULT_CONTAINER_RUNTIME_ROOT = os.environ.get(
    "LOCAL_EVALUATOR_CONTAINER_RUNTIME_ROOT",
    "/home/leigu/pilot-auto.x2.v4.4_e2e",
)
DEFAULT_SIM_WORK_DIR = os.environ.get(
    "LOCAL_EVALUATOR_SIM_WORK_DIR",
    str(DEFAULT_WORK_ROOT / "webauto_sim" / "work"),
)
DEFAULT_SIM_ASSET_DIR = os.environ.get(
    "LOCAL_EVALUATOR_SIM_ASSET_DIR",
    str(DEFAULT_WORK_ROOT / "webauto_sim" / "work" / "assets"),
)
DEFAULT_ROS_DISTRO = os.environ.get("LOCAL_EVALUATOR_ROS_DISTRO", "humble")
DEFAULT_EVALUATOR_ARTIFACT = os.environ.get("LOCAL_EVALUATOR_ARTIFACT", "main")
DEFAULT_WEBAUTO_CI_GITHUB_TOKEN = os.environ.get(
    "WEBAUTO_CI_GITHUB_TOKEN",
    os.environ.get("GITHUB_TOKEN", ""),
)
DEFAULT_LOCAL_EVALUATOR_SSH_KEY = os.environ.get("LOCAL_EVALUATOR_SSH_KEY", "/run/secrets/ssh")
DEFAULT_AGNOCAST_MODE = os.environ.get("LOCAL_EVALUATOR_AGNOCAST_MODE", "auto")
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
DEFAULT_WEBAUTO_BIN = os.environ.get("LOCAL_EVALUATOR_WEBAUTO_BIN", "webauto")
DEFAULT_EVALUATOR_IMAGE_PREFIXES = tuple(
    item.strip()
    for item in os.environ.get("LOCAL_EVALUATOR_IMAGE_PREFIXES", "firmware.ci.web.auto/,pilot-auto").split(",")
    if item.strip()
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


def default_sandbox_container_name(task_id: object, branch: object = "") -> str:
    suffix = str(task_id or "")[:8] or sanitize_name(branch, "local")
    return f"evaluator-sandbox-{sanitize_name(suffix, 'local')}"


def default_pretask_container_name(task_id: object) -> str:
    suffix = str(task_id or "")[:8] or "local"
    return f"evaluator-pretask-{sanitize_name(suffix, 'local')}"


def default_prepared_image_name(image_name: object, task_id: object) -> str:
    suffix = str(task_id or "")[:8] or "local"
    return f"local-evaluator-prepared:{sanitize_name(image_name, 'image')}-{sanitize_name(suffix, 'local')}"


def agnocast_device_available() -> bool:
    return Path("/dev/agnocast").exists()


def resolve_agnocast_env(mode: object = None) -> Dict[str, str]:
    clean_mode = str(mode or DEFAULT_AGNOCAST_MODE or "auto").strip().lower()
    if clean_mode in ("", "auto"):
        clean_mode = "enable" if agnocast_device_available() else "disable"
    if clean_mode in ("off", "disabled", "false", "0", "no"):
        clean_mode = "disable"
    if clean_mode in ("on", "enabled", "true", "1", "yes"):
        clean_mode = "enable"
    if clean_mode == "require":
        if not agnocast_device_available():
            raise RuntimeError(
                "Agnocast was required, but /dev/agnocast was not found. "
                "Load the host agnocast kernel module or switch Agnocast mode to auto/disable."
            )
        clean_mode = "enable"
    if clean_mode == "enable" and not agnocast_device_available():
        raise RuntimeError(
            "Agnocast mode is enable, but /dev/agnocast was not found. "
            "Load the host agnocast kernel module or use auto/disable."
        )
    if clean_mode == "disable":
        return {"ENABLE_AGNOCAST": "0", "AGNOCAST_BRIDGE_MODE": "off"}
    if clean_mode == "enable":
        return {"ENABLE_AGNOCAST": "1", "AGNOCAST_BRIDGE_MODE": "performance"}
    raise RuntimeError(f"Unsupported Agnocast mode: {mode}")


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
            "task_id": task_id,
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
        self.log(f"$ {shlex.join(cmd)} (cwd: {cwd})")
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


def resolve_webauto_bin(configured: object = "") -> Optional[str]:
    candidate = str(configured or DEFAULT_WEBAUTO_BIN or "webauto").strip()
    if not candidate:
        candidate = "webauto"
    if "/" in candidate:
        return candidate if Path(candidate).is_file() else None
    found = shutil.which(candidate)
    if found:
        return found
    fallback = Path("/usr/local/bin/webauto")
    return str(fallback) if fallback.is_file() else None


def require_webauto_bin(parameters: Dict[str, Any]) -> str:
    executable = resolve_webauto_bin(parameters.get("webauto_bin"))
    if executable:
        return executable
    raise RuntimeError(
        "WebAuto CLI executable was not found in the worker PATH. "
        "Install `webauto` in the dashboard image/worker container, mount the host binary at `/usr/local/bin/webauto`, "
        "or set `LOCAL_EVALUATOR_WEBAUTO_BIN` / job `webauto_bin` to the executable path."
    )


def host_path_for_docker_bind(path: Path) -> str:
    resolved = str(path.expanduser().resolve())
    container_root = str(Path(LOCAL_EVALUATOR_CONTAINER_DATA_ROOT)).rstrip("/")
    host_root = str(LOCAL_EVALUATOR_HOST_DATA_ROOT or "").strip().rstrip("/")
    if host_root and resolved == container_root:
        return host_root
    if host_root and resolved.startswith(container_root + "/"):
        return host_root + resolved[len(container_root):]
    if resolved.startswith("/app/data/"):
        raise RuntimeError(
            "Cannot bind-mount the checkout into a host Docker container because "
            "`LOCAL_EVALUATOR_HOST_DATA_ROOT` is not set. Set it to the host path "
            "that is mounted as `/app/data` in the dashboard containers."
        )
    return resolved


def _mapped_container_path(host_path: str) -> Optional[Path]:
    host_root = str(LOCAL_EVALUATOR_HOST_DATA_ROOT or "").strip().rstrip("/")
    container_root = str(Path(LOCAL_EVALUATOR_CONTAINER_DATA_ROOT)).rstrip("/")
    if host_root and host_path == host_root:
        return Path(container_root)
    if host_root and host_path.startswith(host_root + "/"):
        return Path(container_root + host_path[len(host_root):])
    return None


def ensure_host_visible_directory(host_path: str) -> None:
    path = Path(host_path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    mapped_container = _mapped_container_path(str(path))
    if mapped_container and mapped_container != path:
        mapped_container.mkdir(parents=True, exist_ok=True)


def host_visible_webauto_path(path_text: object, fallback: Path, *, create: bool = True) -> str:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else fallback
    if not path.is_absolute() or str(path).startswith("/tmp/"):
        path = fallback
    if create:
        path.mkdir(parents=True, exist_ok=True)
    host_path = host_path_for_docker_bind(path)
    if create:
        ensure_host_visible_directory(host_path)
    return host_path


def host_webauto_root() -> str:
    configured = str(LOCAL_EVALUATOR_HOST_WEBAUTO_ROOT or "").strip()
    if configured:
        return configured
    root_webauto = Path("/root/.webauto")
    if root_webauto.is_dir():
        return str(root_webauto)
    return str(Path.home() / ".webauto")


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


def _preflight_docker_for_build(job: LocalEvaluatorJob, checkout: Path) -> None:
    docker_version = _run_quiet(["docker", "--version"], cwd=checkout)
    if docker_version.returncode != 0:
        raise RuntimeError("Docker CLI is not available in the worker container. Rebuild the dashboard image with Docker CLI support.")
    buildx_version = _run_quiet(["docker", "buildx", "version"], cwd=checkout)
    if buildx_version.returncode != 0:
        raise RuntimeError(
            "Docker buildx is not available in the worker container. "
            "Rebuild the dashboard image with docker-buildx-plugin installed."
        )
    job.set_summary(
        docker_version=(docker_version.stdout or docker_version.stderr or "").strip(),
        docker_buildx_version=(buildx_version.stdout or buildx_version.stderr or "").strip(),
    )


def build_image_docker_multi_stage(job: LocalEvaluatorJob, checkout: Path) -> str:
    params = job.parameters
    image_name = str(params.get("image_name") or default_image_name(params.get("branch"))).strip()
    build_dir = checkout / "docker-multi-stage"
    if not build_dir.is_dir():
        raise RuntimeError(f"Build directory not found: {build_dir}")
    _preflight_docker_for_build(job, checkout)
    ros_distro = str(params.get("ros_distro") or DEFAULT_ROS_DISTRO).strip()
    job.progress("Starting Docker multi-stage build", 20)
    job.run_command(
        ["./build-main.bash", "evaluation", "--yes", "--use-ghcr", "--ros-distro", ros_distro],
        cwd=build_dir,
        step_name="docker multi-stage build",
        pct_start=20,
        pct_end=70,
    )
    source_image = str(params.get("source_build_image") or "pilot-auto:evaluation").strip()
    if image_name and image_name != source_image:
        job.run_command(["docker", "tag", source_image, image_name], cwd=checkout, step_name="docker tag", pct_start=70, pct_end=74)
    job.set_summary(image_name=image_name, build_status="passed")
    return image_name


def find_webauto_ci_config(checkout: Path) -> Optional[Path]:
    for name in (".webauto-ci.yml", ".webauto-ci.yaml"):
        candidate = checkout / name
        if candidate.is_file():
            return candidate
    return None


def _default_evaluator_phases(checkout: Path) -> List[Dict[str, Any]]:
    phases = [
        {
            "name": "environment-setup",
            "user": "root",
            "workdir": "/home/autoware/pilot-auto",
            "exec": "./.webauto-ci/main/environment-setup/run.sh",
        },
        {
            "name": "autoware-setup",
            "user": "autoware",
            "workdir": "/home/autoware/pilot-auto",
            "exec": "./.webauto-ci/main/autoware-setup/run.sh",
        },
        {
            "name": "autoware-build",
            "user": "autoware",
            "workdir": "/home/autoware/pilot-auto",
            "exec": "./.webauto-ci/main/autoware-build/run.sh",
        },
        {
            "name": "asset-deploy",
            "user": "autoware",
            "workdir": "/home/autoware/pilot-auto",
            "exec": "./.webauto-ci/common/asset-deploy/run.sh",
        },
    ]
    _validate_evaluator_phase_scripts(checkout, phases)
    return phases


def _validate_evaluator_phase_scripts(checkout: Path, phases: List[Dict[str, Any]]) -> None:
    missing = []
    for phase in phases:
        exec_cmd = str(phase.get("exec") or "").strip()
        if not exec_cmd.startswith("./"):
            continue
        script = checkout / exec_cmd[2:]
        if not script.is_file():
            missing.append(str(script))
    if missing:
        raise RuntimeError("Missing evaluator CI script(s): " + ", ".join(missing))


def _phase_scripts_require_github_token(checkout: Path, phases: List[Dict[str, Any]]) -> bool:
    for phase in phases:
        exec_cmd = str(phase.get("exec") or "").strip()
        if not exec_cmd.startswith("./"):
            continue
        script = checkout / exec_cmd[2:]
        try:
            if "WEBAUTO_CI_GITHUB_TOKEN" in script.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def _ssh_key_available_for_sandbox() -> bool:
    path = Path(DEFAULT_LOCAL_EVALUATOR_SSH_KEY).expanduser()
    return path.is_file()


def load_evaluator_ci_phases(
    checkout: Path,
    *,
    artifact_name: str = DEFAULT_EVALUATOR_ARTIFACT,
    stop_phase: str = "asset-deploy",
    source_path: str = "/workspace/source",
) -> List[Dict[str, Any]]:
    config_path = find_webauto_ci_config(checkout)
    if config_path is None:
        return _default_evaluator_phases(checkout)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    artifacts = data.get("artifacts") if isinstance(data, dict) else None
    if not isinstance(artifacts, list):
        raise RuntimeError(f"Could not read artifacts from evaluator CI config: {config_path}")

    artifact = next(
        (item for item in artifacts if isinstance(item, dict) and str(item.get("name") or "") == artifact_name),
        None,
    )
    if artifact is None:
        available = [str(item.get("name")) for item in artifacts if isinstance(item, dict) and item.get("name")]
        raise RuntimeError(
            f"Evaluator CI artifact `{artifact_name}` was not found in {config_path}. "
            f"Available artifacts: {', '.join(available) or '(none)'}"
        )

    build = artifact.get("build") if isinstance(artifact, dict) else None
    build_env: Dict[str, Any] = {
        "WEBAUTO_CI_SOURCE_PATH": source_path,
        "WEBAUTO_CI_DEBUG_BUILD": "false",
    }
    config_build_env = build.get("environment_variables") if isinstance(build, dict) else {}
    if isinstance(config_build_env, dict):
        build_env.update(config_build_env)
    raw_phases = build.get("phases") if isinstance(build, dict) else None
    if not isinstance(raw_phases, list):
        raise RuntimeError(f"Evaluator CI artifact `{artifact_name}` has no build phases in {config_path}")

    phases: List[Dict[str, Any]] = []
    found_stop = False
    for raw_phase in raw_phases:
        if not isinstance(raw_phase, dict):
            continue
        name = str(raw_phase.get("name") or "").strip()
        exec_cmd = str(raw_phase.get("exec") or "").strip()
        if not name or not exec_cmd:
            continue
        env: Dict[str, Any] = {}
        if isinstance(build_env, dict):
            env.update(build_env)
        phase_env = raw_phase.get("environment_variables")
        if isinstance(phase_env, dict):
            env.update(phase_env)
        phases.append(
            {
                "name": name,
                "user": str(raw_phase.get("user") or "root").strip() or "root",
                "workdir": str(raw_phase.get("workdir") or "/home/autoware/pilot-auto").strip()
                or "/home/autoware/pilot-auto",
                "exec": exec_cmd,
                "environment_variables": env,
            }
        )
        if name == stop_phase:
            found_stop = True
            break

    if not phases:
        raise RuntimeError(f"Evaluator CI artifact `{artifact_name}` did not contain runnable phases in {config_path}")
    if stop_phase and not found_stop:
        raise RuntimeError(f"Evaluator CI stop phase `{stop_phase}` was not found in artifact `{artifact_name}`")

    _validate_evaluator_phase_scripts(checkout, phases)
    return phases


def _phase_shell_command(phase: Dict[str, Any]) -> str:
    env = phase.get("environment_variables")
    exports = []
    if isinstance(env, dict):
        for key, value in env.items():
            clean_key = str(key).strip()
            if not clean_key or clean_key == "WEBAUTO_CI_GITHUB_TOKEN":
                continue
            exports.append(f"export {clean_key}={shlex.quote(str(value))}")
    exec_cmd = str(phase.get("exec") or "").strip()
    if exports:
        return " && ".join(exports + [exec_cmd])
    return exec_cmd


def _enable_sandbox_ssh_github_auth(job: LocalEvaluatorJob, checkout: Path, sandbox: str) -> None:
    source_key = Path(DEFAULT_LOCAL_EVALUATOR_SSH_KEY).expanduser()
    if not source_key.is_file():
        raise RuntimeError(f"Local evaluator SSH key was not found: {source_key}")
    sandbox_key = "/tmp/eval_dashboard_github_ssh_key"
    job.run_command(
        ["docker", "cp", str(source_key), f"{sandbox}:{sandbox_key}"],
        cwd=checkout,
        step_name="copy github ssh key into evaluator sandbox",
        pct_start=30,
        pct_end=30.2,
    )
    job.run_command(
        ["docker", "exec", sandbox, "bash", "-lc", f"chmod 600 {sandbox_key}"],
        cwd=checkout,
        step_name="secure evaluator sandbox ssh key",
        pct_start=30.2,
        pct_end=30.4,
    )


def _patch_copied_ci_scripts_for_ssh(job: LocalEvaluatorJob, checkout: Path, sandbox: str, container_checkout: str) -> None:
    patch_command = r"""
set -e
find .webauto-ci -name run.sh -print0 | xargs -0 sed -i \
  -e 's/: "${WEBAUTO_CI_GITHUB_TOKEN:?is not set}"/: "${WEBAUTO_CI_GITHUB_TOKEN:=local-ssh-auth}"/' \
  -e '/export GITHUB_TOKEN="\$WEBAUTO_CI_GITHUB_TOKEN"/d' \
  -e '/url\."https:\/\/github.com\/"\.insteadOf/d' \
  -e '/credential\."https:\/\/github.com"\.helper/d' \
  -e '/git config --global --unset credential\."https:\/\/github.com"\.helper/d' \
  -e '/git config --global --unset url\."https:\/\/github.com\/"\.insteadOf/d'
git config --global url."ssh://git@github.com/".insteadOf "https://github.com/"
"""
    job.run_command(
        ["docker", "exec", "-w", container_checkout, sandbox, "bash", "-lc", patch_command],
        cwd=checkout,
        step_name="patch evaluator ci scripts for ssh github auth",
        pct_start=30.4,
        pct_end=30.8,
    )


def build_image_evaluator_ci(job: LocalEvaluatorJob, checkout: Path) -> str:
    params = job.parameters
    image_name = str(params.get("image_name") or default_image_name(params.get("branch"))).strip()
    _preflight_docker_for_build(job, checkout)
    artifact_name = str(params.get("evaluator_artifact") or DEFAULT_EVALUATOR_ARTIFACT).strip() or "main"

    sandbox = default_sandbox_container_name(job.task_id, params.get("branch"))
    container_source = "/workspace/source"
    container_checkout = "/home/autoware/pilot-auto"
    phases = load_evaluator_ci_phases(checkout, artifact_name=artifact_name, source_path=container_source)
    host_checkout = host_path_for_docker_bind(checkout)
    ubuntu_image = str(params.get("evaluator_base_image") or "ubuntu:22.04").strip()
    use_gpu = bool(params.get("evaluator_build_gpu", False))
    gpu_args = ["--gpus", "all"] if use_gpu else []
    github_token = str(params.get("webauto_ci_github_token") or DEFAULT_WEBAUTO_CI_GITHUB_TOKEN).strip()
    use_ssh_github_auth = False
    if not github_token and _phase_scripts_require_github_token(checkout, phases):
        use_ssh_github_auth = _ssh_key_available_for_sandbox()
    if not github_token and _phase_scripts_require_github_token(checkout, phases) and not use_ssh_github_auth:
        raise RuntimeError(
            "Selected evaluator CI phases require WEBAUTO_CI_GITHUB_TOKEN. "
            "Set WEBAUTO_CI_GITHUB_TOKEN or GITHUB_TOKEN in the worker environment, or mount the dashboard SSH secret "
            f"at {DEFAULT_LOCAL_EVALUATOR_SSH_KEY}."
        )
    effective_github_token = github_token or ("local-ssh-auth" if use_ssh_github_auth else "")
    sandbox_git_ssh_command = (
        "ssh -i /tmp/eval_dashboard_github_ssh_key "
        "-o StrictHostKeyChecking=accept-new "
        "-o UserKnownHostsFile=/tmp/eval_dashboard_known_hosts"
    )
    if use_ssh_github_auth:
        for phase in phases:
            phase_env = phase.setdefault("environment_variables", {})
            if isinstance(phase_env, dict):
                phase_env["GIT_SSH_COMMAND"] = sandbox_git_ssh_command

    job.set_summary(
        build_method="evaluator_ci",
        sandbox_container=sandbox,
        evaluator_base_image=ubuntu_image,
        host_checkout_path=host_checkout,
        evaluator_source_path=container_source,
        evaluator_autoware_path=container_checkout,
        evaluator_artifact=artifact_name,
        evaluator_phases=[phase["name"] for phase in phases],
        evaluator_github_auth="token" if github_token else ("ssh" if use_ssh_github_auth else ""),
    )
    job.progress("Pulling evaluator base image", 20)
    job.run_command(["docker", "pull", ubuntu_image], cwd=checkout, step_name="docker pull evaluator base", pct_start=20, pct_end=25)
    job.run_command(["docker", "rm", "-f", sandbox], cwd=checkout, step_name="remove old evaluator sandbox", pct_start=25, pct_end=26, check=False)
    job.progress("Starting evaluator sandbox", 26)
    job.run_command(
        [
            "docker",
            "run",
            "-d",
            "-it",
            "--name",
            sandbox,
            *gpu_args,
            "-v",
            f"{host_checkout}:{container_source}:ro",
            "-w",
            container_source,
            ubuntu_image,
            "sleep",
            "infinity",
        ],
        cwd=checkout,
        step_name="start evaluator sandbox",
        pct_start=26,
        pct_end=30,
    )
    if use_ssh_github_auth:
        _enable_sandbox_ssh_github_auth(job, checkout, sandbox)

    phase_span = 42 / max(len(phases), 1)
    for index, phase in enumerate(phases):
        pct_start = 30 + index * phase_span
        pct_end = min(72, 30 + (index + 1) * phase_span)
        step_name = f"evaluator {phase['name']}"
        cmd = ["docker", "exec", "-w", str(phase.get("workdir") or container_checkout)]
        user = str(phase.get("user") or "root").strip()
        if user and user != "root":
            cmd.extend(["-u", user])
        command_env: Dict[str, str] = {}
        if effective_github_token:
            cmd.extend(["-e", "WEBAUTO_CI_GITHUB_TOKEN"])
            command_env["WEBAUTO_CI_GITHUB_TOKEN"] = effective_github_token
        cmd.extend([sandbox, "bash", "-lc", _phase_shell_command(phase)])
        job.progress(step_name, pct_start)
        job.run_command(
            cmd,
            cwd=checkout,
            step_name=step_name,
            pct_start=pct_start,
            pct_end=pct_end,
            env=command_env or None,
        )
        if phase["name"] == "environment-setup":
            if use_ssh_github_auth:
                _patch_copied_ci_scripts_for_ssh(job, checkout, sandbox, container_checkout)
            job.run_command(
                ["docker", "exec", sandbox, "bash", "-lc", "chown -R autoware:autoware /home/autoware || true"],
                cwd=checkout,
                step_name="evaluator chown autoware home",
                pct_start=pct_end,
                pct_end=min(72, pct_end + 0.5),
            )

    job.progress("Committing evaluator sandbox image", 72)
    job.run_command(["docker", "commit", sandbox, image_name], cwd=checkout, step_name="commit evaluator image", pct_start=72, pct_end=74)
    job.set_summary(image_name=image_name, build_status="passed")
    return image_name


def build_image(job: LocalEvaluatorJob, checkout: Path) -> str:
    method = str(job.parameters.get("build_method") or "evaluator_ci").strip()
    job.set_summary(build_method=method)
    if method == "docker_multi_stage":
        return build_image_docker_multi_stage(job, checkout)
    if method == "evaluator_ci":
        return build_image_evaluator_ci(job, checkout)
    raise RuntimeError(f"Unsupported build method: {method}")


def list_scenarios(job: LocalEvaluatorJob) -> None:
    project_id = str(job.parameters.get("project_id") or "").strip()
    if not project_id or not bool(job.parameters.get("list_scenarios", False)):
        return
    page_size = str(int(job.parameters.get("scenario_page_size") or 50))
    webauto_bin = require_webauto_bin(job.parameters)
    job.run_command(
        [webauto_bin, "ci", "scenario", "list", "--project-id", project_id, "--page-size", page_size],
        cwd=job.run_dir,
        step_name="scenario list",
        pct_start=75,
        pct_end=78,
        check=False,
    )


def _has_cli_option(cmd: List[str], option: str) -> bool:
    prefix = f"{option}="
    return any(part == option or part.startswith(prefix) for part in cmd)


def _set_cli_option(cmd: List[str], option: str, value: str) -> None:
    prefix = f"{option}="
    for index, part in enumerate(list(cmd)):
        if part == option:
            if index + 1 < len(cmd):
                cmd[index + 1] = value
            else:
                cmd.append(value)
            return
        if part.startswith(prefix):
            cmd[index] = f"{option}={value}"
            return
    cmd.extend([option, value])


def build_webauto_command(
    raw_command: object,
    *,
    webauto_bin: str = "webauto",
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
    cmd[0] = webauto_bin or "webauto"
    additions = [
        ("--docker-image", image_name),
        ("--timeout", timeout),
    ]
    for option, value in additions:
        clean_value = str(value or "").strip()
        if clean_value and not _has_cli_option(cmd, option):
            cmd.extend([option, clean_value])
    for option, value in [
        ("--container-runtime-path", container_runtime_path),
        ("--work-dir", work_dir),
        ("--asset-dir", asset_dir),
    ]:
        clean_value = str(value or "").strip()
        if clean_value:
            _set_cli_option(cmd, option, clean_value)
    return cmd


def _raw_webauto_command_parts(raw_command: object) -> List[str]:
    raw = str(raw_command or DEFAULT_WEBAUTO_SCENARIO_COMMAND).strip() or DEFAULT_WEBAUTO_SCENARIO_COMMAND
    try:
        return shlex.split(raw)
    except ValueError:
        return []


def _webauto_command_simulation_name(raw_command: object) -> str:
    parts = _raw_webauto_command_parts(raw_command)
    for index, part in enumerate(parts):
        if part == "--simulation-name" and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith("--simulation-name="):
            return part.split("=", 1)[1]
    return ""


def should_run_simulation_pretasks(params: Dict[str, Any]) -> bool:
    configured = params.get("run_simulation_pretasks")
    if configured is None:
        configured = params.get("run_perception_pretask")
    if configured is not None:
        return bool(configured)
    env_value = os.environ.get(
        "LOCAL_EVALUATOR_RUN_SIMULATION_PRETASKS",
        os.environ.get("LOCAL_EVALUATOR_RUN_PERCEPTION_PRETASK", "auto"),
    ).strip().lower()
    if env_value in ("1", "true", "yes", "on"):
        return True
    if env_value in ("0", "false", "no", "off"):
        return False
    return bool(_webauto_command_simulation_name(params.get("webauto_command") or DEFAULT_WEBAUTO_SCENARIO_COMMAND))


def should_run_perception_pretask(params: Dict[str, Any]) -> bool:
    return should_run_simulation_pretasks(params)


def simulation_pretasks_from_webauto_ci_text(config_text: str, simulation_name: str) -> List[Dict[str, Any]]:
    data = yaml.safe_load(config_text) or {}
    simulations = data.get("simulations") if isinstance(data, dict) else None
    if not isinstance(simulations, list):
        return []
    selected = None
    for item in simulations:
        if isinstance(item, dict) and str(item.get("name") or "") == simulation_name:
            selected = item
            break
    if selected is None:
        return []
    simulator = selected.get("simulator") if isinstance(selected, dict) else None
    raw_tasks = simulator.get("pre_tasks") if isinstance(simulator, dict) else None
    if not isinstance(raw_tasks, list):
        return []
    tasks = []
    for index, raw_task in enumerate(raw_tasks, start=1):
        if not isinstance(raw_task, dict):
            continue
        command = str(raw_task.get("exec") or "").strip()
        if not command:
            continue
        tasks.append({"name": f"{simulation_name} pre-task {index}", "exec": command})
    return tasks


def _image_workdir(image_name: str) -> str:
    result = _run_docker(["inspect", image_name, "--format", "{{.Config.WorkingDir}}"], timeout=30)
    if result.returncode != 0:
        return "/home/autoware/pilot-auto"
    return (result.stdout or "").strip() or "/home/autoware/pilot-auto"


def _webauto_ci_text_from_image(image_name: str) -> str:
    workdir = _image_workdir(image_name)
    candidates = [
        posixpath.join(workdir, ".webauto-ci.yml"),
        posixpath.join(workdir, ".webauto-ci.yaml"),
        "/home/autoware/pilot-auto/.webauto-ci.yml",
        "/home/autoware/pilot-auto/.webauto-ci.yaml",
    ]
    script = " || ".join(f"cat {shlex.quote(path)}" for path in candidates)
    result = _run_docker(["run", "--rm", image_name, "bash", "-lc", script], timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Could not read .webauto-ci.yml from image `{image_name}`: {docker_error(result)}")
    return result.stdout


def _webauto_ci_text_from_checkout(path_text: object) -> str:
    checkout = Path(str(path_text or "")).expanduser()
    if not checkout:
        return ""
    config = find_webauto_ci_config(checkout)
    if not config:
        return ""
    return config.read_text(encoding="utf-8")


def load_simulation_pretasks(job: LocalEvaluatorJob, image_name: str, simulation_name: str) -> List[Dict[str, Any]]:
    config_text = _webauto_ci_text_from_checkout(job.parameters.get("checkout_path"))
    source = str(job.parameters.get("checkout_path") or "").strip()
    if not config_text:
        config_text = _webauto_ci_text_from_image(image_name)
        source = f"image:{image_name}"
    tasks = simulation_pretasks_from_webauto_ci_text(config_text, simulation_name)
    job.set_summary(simulation_pretask_source=source, simulation_pretask_count=len(tasks))
    return tasks


def prepare_simulation_image(job: LocalEvaluatorJob, image_name: str) -> str:
    simulation_name = _webauto_command_simulation_name(
        job.parameters.get("webauto_command") or DEFAULT_WEBAUTO_SCENARIO_COMMAND
    )
    if not simulation_name:
        job.set_summary(simulation_pretasks="skipped", simulation_pretask_reason="No simulation name")
        return image_name
    tasks = load_simulation_pretasks(job, image_name, simulation_name)
    if not tasks:
        job.set_summary(
            simulation_pretasks="skipped",
            simulation_pretask_reason=f"No pre_tasks found for simulation `{simulation_name}`",
        )
        return image_name
    container = default_pretask_container_name(job.task_id)
    prepared_image = str(
        job.parameters.get("prepared_image_name") or default_prepared_image_name(image_name, job.task_id)
    ).strip()
    use_gpu = bool(job.parameters.get("perception_pretask_gpu", True))
    gpu_args = ["--gpus", "all"] if use_gpu else []
    agnocast_env = resolve_agnocast_env(job.parameters.get("agnocast_mode"))
    docker_env_args = []
    for key, value in agnocast_env.items():
        docker_env_args.extend(["-e", f"{key}={value}"])
    webauto_root_path = Path(str(job.parameters.get("host_webauto_root") or host_webauto_root())).expanduser()
    webauto_root_path.mkdir(parents=True, exist_ok=True)

    job.set_summary(
        perception_pretask="enabled",
        simulation_pretasks="enabled",
        simulation_name=simulation_name,
        simulation_pretask_container=container,
        simulation_pretask_image=prepared_image,
        simulation_pretask_commands=[task["exec"] for task in tasks],
        agnocast_mode=str(job.parameters.get("agnocast_mode") or DEFAULT_AGNOCAST_MODE),
        agnocast_env=agnocast_env,
        agnocast_device_available=agnocast_device_available(),
    )
    job.progress("Preparing simulation image", 74)
    job.run_command(
        ["docker", "rm", "-f", container],
        cwd=job.run_dir,
        step_name="remove old simulation pretask container",
        pct_start=74,
        pct_end=74.5,
        check=False,
    )
    job.run_command(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container,
            *gpu_args,
            *docker_env_args,
            "-v",
            f"{webauto_root_path}:/home/autoware/.webauto",
            image_name,
            "sleep",
            "infinity",
        ],
        cwd=job.run_dir,
        step_name="start simulation pretask container",
        pct_start=74.5,
        pct_end=75,
    )
    span = 2 / max(len(tasks), 1)
    for index, task in enumerate(tasks):
        pct_start = 75 + index * span
        pct_end = min(77, 75 + (index + 1) * span)
        job.run_command(
            ["docker", "exec", container, "bash", "-lc", str(task["exec"])],
            cwd=job.run_dir,
            step_name=str(task["name"]),
            pct_start=pct_start,
            pct_end=pct_end,
        )
    job.run_command(
        ["docker", "commit", container, prepared_image],
        cwd=job.run_dir,
        step_name="commit simulation prepared image",
        pct_start=77,
        pct_end=78,
    )
    job.run_command(
        ["docker", "rm", "-f", container],
        cwd=job.run_dir,
        step_name="remove simulation pretask container",
        pct_start=78,
        pct_end=78,
        check=False,
    )
    job.set_summary(image_name=prepared_image, original_image_name=image_name)
    return prepared_image


def prepare_perception_image(job: LocalEvaluatorJob, image_name: str) -> str:
    return prepare_simulation_image(job, image_name)


def run_scenario(job: LocalEvaluatorJob, image_name: str) -> str:
    params = job.parameters
    agnocast_env = resolve_agnocast_env(params.get("agnocast_mode"))
    if should_run_simulation_pretasks(params):
        image_name = prepare_simulation_image(job, image_name)
    else:
        job.set_summary(perception_pretask="skipped")
    webauto_bin = require_webauto_bin(params)
    webauto_root = job.run_dir / "webauto"
    tmp_dir = webauto_root / "tmp"
    work_dir = host_visible_webauto_path(params.get("work_dir") or DEFAULT_SIM_WORK_DIR, webauto_root / "work")
    asset_dir = host_visible_webauto_path(params.get("asset_dir") or DEFAULT_SIM_ASSET_DIR, webauto_root / "work" / "assets")
    raw_runtime_path = str(params.get("container_runtime_path") or "").strip()
    if not raw_runtime_path and str(params.get("mode") or "") == "build_and_test":
        raw_runtime_path = str(params.get("checkout_path") or "").strip()
    runtime_path = (
        host_visible_webauto_path(raw_runtime_path, webauto_root / "runtime", create=False)
        if raw_runtime_path
        else ""
    )
    tmp_host_dir = host_visible_webauto_path(tmp_dir, tmp_dir)
    webauto_root_path = Path(str(params.get("host_webauto_root") or host_webauto_root())).expanduser()
    webauto_root_path.mkdir(parents=True, exist_ok=True)
    webauto_home = str(webauto_root_path.parent)
    cmd = build_webauto_command(
        params.get("webauto_command") or DEFAULT_WEBAUTO_SCENARIO_COMMAND,
        webauto_bin=webauto_bin,
        image_name=image_name,
        container_runtime_path=runtime_path,
        work_dir=work_dir,
        asset_dir=asset_dir,
        timeout=str(params.get("timeout") or DEFAULT_TIMEOUT).strip(),
    )
    list_scenarios(job)
    job.progress("Starting scenario run", 78)
    job.set_summary(
        webauto_command=" ".join(shlex.quote(part) for part in cmd),
        webauto_tmp_dir=tmp_host_dir,
        webauto_work_dir=work_dir,
        webauto_asset_dir=asset_dir,
        webauto_container_runtime_path=runtime_path,
        webauto_home=webauto_home,
        webauto_root=str(webauto_root_path),
        agnocast_mode=str(params.get("agnocast_mode") or DEFAULT_AGNOCAST_MODE),
        agnocast_env=agnocast_env,
        agnocast_device_available=agnocast_device_available(),
    )
    command_env = {
        "TMPDIR": tmp_host_dir,
        "TMP": tmp_host_dir,
        "TEMP": tmp_host_dir,
        "HOME": webauto_home,
    }
    command_env.update(agnocast_env)
    result = job.run_command(
        cmd,
        cwd=job.run_dir,
        step_name="scenario run",
        pct_start=78,
        pct_end=98,
        env=command_env,
        check=False,
    )
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


def docker_images(prefixes: Optional[tuple[str, ...]] = None) -> List[str]:
    active_prefixes = prefixes if prefixes is not None else DEFAULT_EVALUATOR_IMAGE_PREFIXES
    result = _run_docker(["images", "--format", "{{.Repository}}:{{.Tag}}"])
    if result.returncode != 0:
        return []
    images = []
    for line in result.stdout.splitlines():
        image = line.strip()
        if image and image != "<none>:<none>" and any(image.startswith(prefix) for prefix in active_prefixes):
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
