"""Local evaluator debug page for host-side build/test experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from lib.db import create_task, get_task, is_task_queue_enabled, list_recent_tasks, update_task_rq_job_id
from lib.local_evaluator_debug import (
    DEFAULT_AGNOCAST_MODE,
    DEFAULT_EVALUATOR_ARTIFACT,
    DEFAULT_REPO_URL,
    DEFAULT_SIM_ASSET_DIR,
    DEFAULT_SIM_WORK_DIR,
    DEFAULT_TIMEOUT,
    DEFAULT_WEBAUTO_SCENARIO_COMMAND,
    DEFAULT_WORK_ROOT,
    commit_debug_container,
    default_checkout_path,
    default_debug_container_name,
    default_image_name,
    docker_containers,
    docker_images,
    exec_in_container,
    list_container_files,
    read_container_file,
    start_debug_container,
    tail_text,
    write_container_file,
)
from lib.page_chrome import inject_app_page_styles, render_page_hero, section_header
from lib.auth import get_current_user_identity
from lib.ui.task_history import get_task_list_current_user, render_task_list


st.set_page_config(
    page_title="Local Evaluator Debug",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_app_page_styles()


def _query_value(name: str) -> str:
    try:
        value = st.query_params.get(name)
        if isinstance(value, list):
            return str(value[0] if value else "").strip()
        return str(value or "").strip()
    except Exception:
        return ""


def _log_path_from_task(task_id: str) -> str:
    if not task_id:
        return ""
    row = get_task(task_id)
    if not row:
        return ""
    summary_raw = row.get("result_summary")
    if not summary_raw:
        return ""
    try:
        summary = json.loads(summary_raw) if isinstance(summary_raw, str) else summary_raw
    except (TypeError, ValueError):
        return ""
    if not isinstance(summary, dict):
        return ""
    return str(summary.get("log_path") or "").strip()


def _render_log_file(path_text: str, *, key_prefix: str) -> None:
    path = Path(path_text).expanduser()
    if not path.exists() or not path.is_file():
        st.warning("Log file was not found on this worker filesystem.")
        if str(path_text).startswith("/tmp/webauto-local-evaluator/"):
            st.info(
                "This looks like an older log path under `/tmp`. In Docker deployments `/tmp` is not shared "
                "between the worker and Streamlit containers, so the web page cannot read that file. "
                "New jobs write logs under the shared dashboard data directory."
            )
        st.caption(path_text)
        return
    max_kb = st.slider(
        "Tail size",
        min_value=16,
        max_value=1024,
        value=256,
        step=16,
        key=f"{key_prefix}_tail_kb",
    )
    st.code(tail_text(path, max_bytes=max_kb * 1024) or "(empty log)", language=None)
    st.download_button(
        "Download full log",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/plain",
        key=f"{key_prefix}_download",
    )


def _enqueue(params: Dict[str, object]) -> Optional[str]:
    try:
        identity = get_current_user_identity()
        session_id = str(identity.get("id") or "").strip() or None
        params = dict(params)
        if session_id:
            params.setdefault("_requester", identity)
        task_id = create_task("local_evaluator_debug", params, session_id=session_id)
        if not task_id:
            return None

        from redis import Redis
        from rq import Queue
        from worker.tasks import run_job

        redis_conn = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        queue = Queue(
            name=os.environ.get("RQ_QUEUE", "default"),
            connection=redis_conn,
            default_timeout="7d",
        )
        job = queue.enqueue(
            run_job,
            task_id,
            "local_evaluator_debug",
            params,
            job_timeout="7d",
            result_ttl="7d",
        )
        rq_id = getattr(job, "id", None)
        if rq_id:
            update_task_rq_job_id(task_id, str(rq_id))
        return task_id
    except Exception as exc:
        st.error(f"Failed to enqueue local evaluator job: {exc}")
        return None


def _mode_key(label: str) -> str:
    return {
        "Build only": "build",
        "Build + test": "build_and_test",
        "Test existing image": "test",
        "Commit modified container": "commit_container",
    }[label]


render_page_hero(
    kicker="Evaluator",
    title="Local evaluator debug",
    description=(
        "Build a branch on the local evaluator host, run one WebAuto CI scenario with a selected image, "
        "and keep the full command log on disk for failure analysis."
    ),
    mode="Debug",
)

task_queue_enabled = is_task_queue_enabled()
if not task_queue_enabled:
    st.warning(
        "Task queue is not enabled, so build/test jobs cannot be queued. "
        "The container editor can still work if this app process has Docker access."
    )

st.info(
    "Docker commands run where the RQ worker runs. To use the host Docker daemon from a containerized dashboard, "
    "mount `/var/run/docker.sock` into the worker and mount the same pilot checkout/runtime paths."
)

query_log_path = _query_value("log_path")
query_task_id = _query_value("task_id")
if not query_log_path and query_task_id:
    query_log_path = _log_path_from_task(query_task_id)
if query_log_path:
    section_header("Requested log", "Opened from a task link.")
    _render_log_file(query_log_path, key_prefix="query_log")

tab_submit, tab_edit, tab_logs, tab_tasks = st.tabs(["Submit", "Container edit", "Logs", "Tasks"])

with tab_submit:
    if not task_queue_enabled:
        st.info("Enable `DATABASE_URL`, `REDIS_URL`, and `USE_TASK_QUEUE=true` to queue build/test jobs.")
    section_header("Job", "Choose build, test, build+test, or preserve a modified container as a new image.")
    mode_label = st.radio(
        "Mode",
        ["Build + test", "Build only", "Test existing image", "Commit modified container"],
        horizontal=True,
    )
    mode = _mode_key(mode_label)

    if mode == "commit_container":
        c1, c2 = st.columns(2)
        with c1:
            container_id = st.text_input("Container ID or name", placeholder="my-running-debug-container")
        with c2:
            commit_image_name = st.text_input("New image name", placeholder="pilot-auto:evaluation-debug-fix")
        if st.button("Commit container", type="primary", disabled=not task_queue_enabled):
            task_id = _enqueue(
                {
                    "mode": mode,
                    "container_id": container_id.strip(),
                    "commit_image_name": commit_image_name.strip(),
                }
            )
            if task_id:
                st.success(f"Queued local evaluator job: {task_id}")
                st.rerun()
    else:
        branch = ""
        checkout_path = ""
        image_name = ""
        if mode in ("build", "build_and_test"):
            c1, c2 = st.columns([1.2, 1])
            with c1:
                branch = st.text_input("Branch", placeholder="feature/my-evaluator-fix")
            with c2:
                repo_url = st.text_input("Repo URL", value=DEFAULT_REPO_URL)
            checkout_path = str(default_checkout_path(branch)) if branch else ""
            image_name = default_image_name(branch) if branch else ""
            if branch:
                st.caption(f"Derived checkout: `{checkout_path}`")
                st.caption(f"Derived image: `{image_name}`")
                st.caption("Source workspace: `vcs import src < autoware.repos`")
            build_method_label = st.radio(
                "Build method",
                ["Evaluator CI scripts", "Docker multi-stage"],
                horizontal=True,
                help="Evaluator CI scripts reproduces `.webauto-ci/main/*/run.sh`; Docker multi-stage keeps the older local build path.",
            )
            build_method = "evaluator_ci" if build_method_label == "Evaluator CI scripts" else "docker_multi_stage"
            if build_method == "evaluator_ci":
                st.caption(
                    "Evaluator CI build runs Ubuntu 22.04 sandbox steps: environment setup, Autoware setup, "
                    "Autoware build, asset deploy, then commits the sandbox as the derived image."
                )
                with st.expander("Advanced build settings", expanded=False):
                    evaluator_artifact = st.text_input("Evaluator artifact", value=DEFAULT_EVALUATOR_ARTIFACT)
            else:
                evaluator_artifact = DEFAULT_EVALUATOR_ARTIFACT
            c5, c6, c7 = st.columns(3)
            with c5:
                clean_checkout = st.checkbox("Clean checkout before build", value=False)
            with c6:
                allow_dirty_checkout = st.checkbox("Allow dirty checkout", value=False)
            with c7:
                ros_distro = st.text_input("ROS distro", value="humble")
        else:
            repo_url = DEFAULT_REPO_URL
            build_method = "evaluator_ci"
            evaluator_artifact = DEFAULT_EVALUATOR_ARTIFACT
            clean_checkout = False
            allow_dirty_checkout = False
            ros_distro = "humble"
            images = docker_images()
            if images:
                selected_image = st.selectbox("Existing evaluator image", images)
                image_name = st.text_input("Image name", value=selected_image)
            else:
                image_name = st.text_input("Existing image", value="pilot-auto:evaluation")
                st.caption("No firmware/pilot-auto evaluator images were found in host Docker.")

        if mode in ("test", "build_and_test"):
            section_header("Scenario", "Runs a WebAuto command using the selected or newly built image.")
            st.caption("The worker needs access to the WebAuto data/config folder, usually `~/.webauto`.")
            webauto_command = st.text_area(
                "WebAuto command",
                value=DEFAULT_WEBAUTO_SCENARIO_COMMAND,
                height=120,
                key="local_eval_webauto_command",
            )
            runtime_default = checkout_path if mode == "build_and_test" and checkout_path else ""
            with st.expander("Command defaults added when missing", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    container_runtime_path = st.text_input("Container runtime path", value=runtime_default)
                with c2:
                    timeout = st.text_input("Timeout", value=DEFAULT_TIMEOUT)
                c3, c4 = st.columns(2)
                with c3:
                    work_dir = st.text_input("Work dir", value=DEFAULT_SIM_WORK_DIR)
                with c4:
                    asset_dir = st.text_input("Asset dir", value=DEFAULT_SIM_ASSET_DIR)
                webauto_bin = st.text_input("WebAuto executable", value=os.environ.get("LOCAL_EVALUATOR_WEBAUTO_BIN", "webauto"))
                agnocast_options = ["auto", "disable", "require", "enable"]
                agnocast_default = DEFAULT_AGNOCAST_MODE if DEFAULT_AGNOCAST_MODE in agnocast_options else "auto"
                agnocast_mode = st.selectbox(
                    "Agnocast mode",
                    agnocast_options,
                    index=agnocast_options.index(agnocast_default),
                    help="auto disables Agnocast when /dev/agnocast is unavailable; require fails early if the device is missing.",
                )
                run_simulation_pretasks = st.checkbox("Run .webauto-ci.yml pre_tasks", value=True)
                list_scenarios = st.checkbox("List scenarios before run", value=False)
                project_id = st.text_input("Project ID for scenario list", value="x2_dev")
        else:
            project_id = ""
            webauto_command = ""
            timeout = DEFAULT_TIMEOUT
            container_runtime_path = ""
            work_dir = DEFAULT_SIM_WORK_DIR
            asset_dir = DEFAULT_SIM_ASSET_DIR
            webauto_bin = os.environ.get("LOCAL_EVALUATOR_WEBAUTO_BIN", "webauto")
            agnocast_mode = DEFAULT_AGNOCAST_MODE
            run_simulation_pretasks = False
            list_scenarios = False

        if st.button("Queue local evaluator job", type="primary", disabled=not task_queue_enabled):
            task_id = _enqueue(
                {
                    "mode": mode,
                    "branch": branch.strip(),
                    "repo_url": repo_url.strip(),
                    "checkout_path": checkout_path.strip(),
                    "image_name": image_name.strip(),
                    "build_method": build_method,
                    "evaluator_artifact": evaluator_artifact.strip(),
                    "clean_checkout": clean_checkout,
                    "allow_dirty_checkout": allow_dirty_checkout,
                    "ros_distro": ros_distro.strip(),
                    "project_id": project_id.strip(),
                    "webauto_command": webauto_command.strip(),
                    "webauto_bin": webauto_bin.strip(),
                    "agnocast_mode": agnocast_mode,
                    "container_runtime_path": container_runtime_path.strip(),
                    "work_dir": work_dir.strip(),
                    "asset_dir": asset_dir.strip(),
                    "timeout": timeout.strip(),
                    "run_simulation_pretasks": run_simulation_pretasks,
                    "list_scenarios": list_scenarios,
                }
            )
            if task_id:
                st.success(f"Queued local evaluator job: {task_id}")
                st.rerun()

with tab_edit:
    section_header("Debug container", "Start a container from an image, edit files, run commands, then commit a new image.")
    images = docker_images()
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if images:
            editor_image = st.selectbox("Image", images, key="debug_editor_image")
        else:
            editor_image = st.text_input("Image", value="pilot-auto:evaluation", key="debug_editor_image_manual")
    with c2:
        default_container = default_debug_container_name(editor_image)
        container_name = st.text_input("Container name", value=default_container, key="debug_editor_container_name")

    start_col, refresh_col, _ = st.columns([1.1, 1.1, 3])
    with start_col:
        if st.button("Start container", type="primary"):
            result = start_debug_container(editor_image, container_name)
            if result.get("ok") == "true":
                st.session_state["debug_editor_active_container"] = result.get("container", container_name)
                st.success(f"Started `{result.get('container', container_name)}`")
            else:
                st.error(result.get("message", "Failed to start container"))
    with refresh_col:
        st.button("Refresh containers")

    containers = docker_containers()
    container_labels = [f"{item['name']} ({item['image']})" for item in containers]
    container_by_label = {f"{item['name']} ({item['image']})": item["name"] for item in containers}
    preferred = st.session_state.get("debug_editor_active_container") or container_name
    selected_container = preferred
    if container_labels:
        label_values = list(container_by_label)
        default_index = 0
        for idx, label in enumerate(label_values):
            if container_by_label[label] == preferred:
                default_index = idx
                break
        selected_label = st.selectbox("Running container", label_values, index=default_index)
        selected_container = container_by_label[selected_label]
    else:
        selected_container = st.text_input("Running container", value=preferred, key="debug_editor_manual_container")
    st.session_state["debug_editor_active_container"] = selected_container

    section_header("File editor", "Load a text file from the running container, edit it here, and save it back.")
    b1, b2, b3 = st.columns([1.3, 0.7, 1])
    with b1:
        browse_root = st.text_input("Browse root", value="/workspace", key="debug_editor_browse_root")
    with b2:
        browse_depth = st.number_input("Depth", min_value=1, max_value=10, value=4, key="debug_editor_browse_depth")
    with b3:
        if st.button("List files"):
            result = list_container_files(selected_container, browse_root, max_depth=int(browse_depth), limit=300)
            if result.get("ok"):
                st.session_state["debug_editor_file_options"] = result.get("files", [])
                st.success(result.get("message", "Files loaded"))
            else:
                st.error(result.get("message", "Could not list files"))

    file_options = st.session_state.get("debug_editor_file_options", [])
    if file_options:
        picked_file = st.selectbox("Found files", file_options, key="debug_editor_picked_file")
    else:
        picked_file = ""
    file_path = st.text_input(
        "File path",
        value=st.session_state.get("debug_editor_file_path") or picked_file,
        key="debug_editor_file_path",
        placeholder="/workspace/src/path/to/file.py",
    )

    load_col, save_col, _ = st.columns([1, 1, 3])
    with load_col:
        if st.button("Load file"):
            result = read_container_file(selected_container, file_path)
            if result.get("ok"):
                st.session_state["debug_editor_file_content"] = result.get("content", "")
                st.session_state["debug_editor_content_area"] = result.get("content", "")
                if result.get("truncated"):
                    st.warning("File was truncated for browser editing. Use a smaller file or command-line edits.")
                else:
                    st.success("Loaded file")
            else:
                st.error(result.get("message", "Could not load file"))
    content = st.text_area(
        "Content",
        value=st.session_state.get("debug_editor_file_content", ""),
        height=520,
        key="debug_editor_content_area",
    )
    with save_col:
        if st.button("Save file"):
            result = write_container_file(selected_container, file_path, content)
            if result.get("ok") == "true":
                st.session_state["debug_editor_file_content"] = content
                st.success(result.get("message", "Saved file"))
            else:
                st.error(result.get("message", "Could not save file"))

    section_header("Command", "Run a shell command inside the selected container.")
    command = st.text_area(
        "Command",
        value=st.session_state.get("debug_editor_command", "python --version"),
        height=100,
        key="debug_editor_command",
    )
    timeout_seconds = st.number_input("Command timeout seconds", min_value=1, max_value=3600, value=120)
    if st.button("Run command"):
        result = exec_in_container(selected_container, command, timeout_seconds=int(timeout_seconds))
        st.session_state["debug_editor_command_output"] = result.get("output", "")
        if result.get("ok"):
            st.success(f"Command passed ({result.get('returncode')})")
        else:
            st.error(f"Command failed ({result.get('returncode')})")
    if st.session_state.get("debug_editor_command_output"):
        st.code(st.session_state["debug_editor_command_output"], language=None)

    section_header("Commit image", "Preserve the current container filesystem as a new Docker image.")
    default_commit_name = f"{editor_image}-debug" if editor_image else "pilot-auto:evaluation-debug"
    commit_image = st.text_input("New image name", value=default_commit_name, key="debug_editor_commit_image")
    if st.button("Commit edited container"):
        result = commit_debug_container(selected_container, commit_image)
        if result.get("ok") == "true":
            st.success(result.get("message", f"Committed {commit_image}"))
        else:
            st.error(result.get("message", "Could not commit container"))

with tab_logs:
    section_header("Full log file", "The task row keeps a short useful tail; the full build/test log stays here.")
    log_path = st.text_input(
        "Log path",
        value=query_log_path,
        placeholder=str(DEFAULT_WORK_ROOT / "runs" / "..." / "local_evaluator.log"),
    )
    if log_path:
        _render_log_file(log_path, key_prefix="logs_tab")

with tab_tasks:
    section_header("Local evaluator tasks", "Recent jobs from this dashboard session.")
    current_user = get_task_list_current_user()
    tasks = list_recent_tasks(limit=100, session_id=current_user)
    local_tasks = [t for t in tasks if t.get("type") == "local_evaluator_debug"]
    render_task_list(local_tasks, current_user)
