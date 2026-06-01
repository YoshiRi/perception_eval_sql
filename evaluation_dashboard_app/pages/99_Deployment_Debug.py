"""
Deployment debug: environment (redacted), Postgres/Redis/RQ health, task counts, optional Docker list/logs.

Must live as a top-level pages/*.py file so st.page_link can resolve it. Outside Docker, the default
sidebar entry is hidden via CSS in lib/ui/styles_global.py; Overview shows a page_link only in Docker.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from lib.db import TASK_STATUSES, TASK_TYPES
from lib.deploy_debug import (
    EXEC_TIMEOUT_SEC,
    MAX_LOG_TAIL_LINES,
    compose_project_filter,
    container_exec_command,
    container_logs_tail,
    database_recent_task_rows,
    database_table_overview,
    docker_client_or_none,
    is_docker_debug_enabled,
    is_exec_enabled,
    list_containers_for_debug,
    postgres_check,
    redacted_deployment_env_rows,
    redis_ping_check,
    rq_overview,
    running_in_docker,
    task_counts_by_status,
)
from lib.docker_live_structure import live_containers_mermaid, rowset_has_t4_compose_service
from lib.mermaid_render import render_mermaid
from lib.page_chrome import inject_app_page_styles, render_page_hero, section_header
from lib.t4_visualizer_client import (
    ENV_BASE_URL as T4_ENV_BASE_URL,
    T4VisualizerClient,
    T4VisualizerError,
)

st.set_page_config(
    layout="wide",
    page_title="Deployment debug",
    page_icon="🐳",
    initial_sidebar_state="expanded",
)
if not running_in_docker():
    st.info("**Deployment debug** is only available when the app runs inside a container (e.g. Docker).")
    st.stop()

inject_app_page_styles()
render_page_hero(
    kicker="Operations",
    title="Deployment & Docker debug",
    description=(
        "Check Postgres, Redis, and the RQ queue; inspect redacted environment variables; "
        "optionally list containers and tail logs when Docker socket access is enabled; "
        "optional one-shot shell commands when `EVAL_DEPLOYMENT_DEBUG_EXEC=1`. "
        "The Docker tab’s live diagram includes the T4 dataset server (HTTP 2D/3D rendering) when configured."
    ),
    mode="Single Run",
)

tab_env, tab_dep, tab_tasks, tab_db, tab_docker = st.tabs(
    ["Environment", "Dependencies", "Tasks", "Database", "Docker"]
)

with tab_env:
    section_header("Deployment environment", "Sensitive connection strings are redacted.")
    env_df = pd.DataFrame(redacted_deployment_env_rows(), columns=["Variable", "Value"])
    st.dataframe(env_df, width='stretch', hide_index=True)

with tab_dep:
    section_header("Postgres")
    ok, msg = postgres_check()
    if ok:
        st.success(msg)
    else:
        st.error(msg)

    section_header("Redis")
    ok_r, msg_r = redis_ping_check()
    if ok_r:
        st.success(msg_r)
    else:
        st.error(msg_r)

    section_header("RQ queue")
    ok_q, msg_q, details = rq_overview()
    if ok_q and details:
        st.success(msg_q)
        st.json(details)
    else:
        st.warning(msg_q if not ok_q else "No queue details")

with tab_tasks:
    section_header("Task rows by status", "From Postgres `tasks` when the task queue is enabled.")
    ok_t, msg_t, counts = task_counts_by_status()
    if counts is None:
        st.info(msg_t)
    elif ok_t and counts:
        st.success(msg_t)
        cdf = pd.DataFrame(
            [{"status": k, "count": v} for k, v in sorted(counts.items())]
        )
        st.dataframe(cdf, width='stretch', hide_index=True)
    elif ok_t:
        st.success("No task rows yet (empty table).")
    else:
        st.error(msg_t)


def _debug_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _task_rows_dataframe(rows: list) -> pd.DataFrame:
    display_rows = []
    for row in rows:
        params = row.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        display_rows.append(
            {
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "status": row.get("status"),
                "type": row.get("type"),
                "session_id": row.get("session_id"),
                "id": str(row.get("id") or ""),
                "rq_job_id": row.get("rq_job_id"),
                "job_id": params.get("job_id")
                or params.get("performance_job_id")
                or params.get("devops_job_id")
                or params.get("source_job_id")
                or "",
                "output_path": params.get("output_path") or params.get("output_dir") or "",
                "progress_pct": row.get("progress_pct"),
                "progress_message": row.get("progress_message"),
                "result_path": row.get("result_path"),
                "error_message": row.get("error_message"),
            }
        )
    return pd.DataFrame(display_rows)


def _format_progress_metric(value: Any) -> str:
    try:
        return f"{float(value or 0):g}%"
    except (TypeError, ValueError):
        return "0%"


with tab_db:
    section_header(
        "Database inspector",
        "Read-only view into Postgres tables and recent evaluator/task job history.",
    )

    ok_tables, msg_tables, table_rows = database_table_overview()
    if ok_tables and table_rows is not None:
        overview_df = pd.DataFrame(table_rows)
        if not overview_df.empty:
            overview_df["total_mb"] = (overview_df["total_bytes"] / (1024 * 1024)).round(2)
            st.dataframe(
                overview_df[["table_name", "estimated_rows", "total_mb"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No public tables found.")
    elif not ok_tables:
        st.error(msg_tables)

    section_header("Recent job history", "Raw `tasks` rows, newest first, across all sessions.")
    filters = st.columns([1.2, 1.6, 1.2, 1.2])
    with filters[0]:
        status_filter = st.selectbox(
            "Status",
            ["All", *TASK_STATUSES],
            key="deploy_db_status",
        )
    with filters[1]:
        type_filter = st.selectbox(
            "Task type",
            ["All", *TASK_TYPES],
            key="deploy_db_type",
        )
    with filters[2]:
        row_limit = st.number_input(
            "Rows",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            key="deploy_db_limit",
        )
    with filters[3]:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=1000,
            value=1,
            step=1,
            key="deploy_db_page",
        )
    search = st.text_input(
        "Search",
        key="deploy_db_search",
        placeholder="Task id, job id, session, path, error text, parameters",
    )

    ok_rows, msg_rows, rows, total_rows = database_recent_task_rows(
        limit=int(row_limit),
        offset=(int(page) - 1) * int(row_limit),
        status=None if status_filter == "All" else status_filter,
        task_type=None if type_filter == "All" else type_filter,
        search=search.strip() or None,
    )
    if not ok_rows:
        st.error(msg_rows)
    elif not rows:
        st.info("No task rows matched the current filters.")
    else:
        st.caption(f"Showing **{len(rows)}** of **{total_rows}** matching task rows.")
        task_df = _task_rows_dataframe(rows)
        st.dataframe(task_df, width="stretch", hide_index=True)

        id_options = [str(row.get("id") or "") for row in rows]
        selected_id = st.selectbox("Inspect row", id_options, key="deploy_db_task_inspect")
        selected = next((row for row in rows if str(row.get("id") or "") == selected_id), None)
        if selected:
            meta_cols = st.columns(4)
            meta_cols[0].metric("Status", str(selected.get("status") or "—"))
            meta_cols[1].metric("Type", str(selected.get("type") or "—"))
            meta_cols[2].metric("Progress", _format_progress_metric(selected.get("progress_pct")))
            meta_cols[3].metric("Session", str(selected.get("session_id") or "—")[:32])

            detail_tabs = st.tabs(["Parameters", "Result summary", "Log", "Raw row"])
            with detail_tabs[0]:
                st.code(_debug_json(selected.get("parameters") or {}), language="json")
            with detail_tabs[1]:
                raw_summary = selected.get("result_summary")
                if raw_summary:
                    try:
                        parsed = json.loads(raw_summary) if isinstance(raw_summary, str) else raw_summary
                        st.code(_debug_json(parsed), language="json")
                    except (TypeError, ValueError):
                        st.code(str(raw_summary), language=None)
                else:
                    st.info("No result summary stored for this row.")
            with detail_tabs[2]:
                log_text = (selected.get("log_output") or "").strip()
                st.code(log_text or "(empty)", language=None)
            with detail_tabs[3]:
                st.code(_debug_json(selected), language="json")


def _render_docker_disabled(reason: str) -> None:
    st.warning(reason)
    st.markdown(
        """
**Enable Docker debug (trusted operators only)**

1. From the `deploy/` directory, ensure `docker-compose.yml` mounts `/var/run/docker.sock` into each Streamlit service (`streamlit1`, `streamlit2`) and sets `EVAL_DEPLOYMENT_DEBUG_DOCKER=1`, then run `docker compose up -d` (or recreate those services after editing compose).

2. Set `EVAL_DEPLOYMENT_DEBUG_COMPOSE_PROJECT` in `.env` to your Compose project name
   (same value as in `docker compose ls`) so the UI lists only this stack’s containers.

3. Rebuild or restart the Streamlit service after changing dependencies so `docker` (docker-py) is installed.

Anyone who can open this page with socket access can read container logs for listed containers — use network ACLs or auth in front of the app.

With `EVAL_DEPLOYMENT_DEBUG_EXEC=1`, the Docker tab can also run `sh -c` inside a selected container — treat that like full shell access.
        """
    )


def _render_docker_exec_ui(client, full_id: str) -> None:
    if not is_exec_enabled():
        st.caption(
            "To run commands in the selected container, set `EVAL_DEPLOYMENT_DEBUG_EXEC=1` in `.env` "
            "and recreate Streamlit (high risk — same as `docker exec`)."
        )
        return

    prev = st.session_state.get("deploy_debug_exec_cid")
    if prev is not None and prev != full_id:
        st.session_state.pop("deploy_debug_exec_result", None)
    st.session_state["deploy_debug_exec_cid"] = full_id

    st.markdown("**Run command in container**")
    st.caption(
        "Runs `sh -c \"…\"` in the **currently selected** container. Output is capped; long commands time out "
        f"after ~{int(EXEC_TIMEOUT_SEC)}s."
    )
    st.text_input("Shell command", key="deploy_debug_exec_cmd", placeholder="ls -la /app")
    if st.button("Run", key="deploy_debug_exec_run"):
        cmd = (st.session_state.get("deploy_debug_exec_cmd") or "").strip()
        if not cmd:
            st.warning("Enter a command.")
        else:
            with st.spinner("Executing…"):
                code, out = container_exec_command(client, full_id, cmd)
            st.session_state["deploy_debug_exec_result"] = (code, out)
    res = st.session_state.get("deploy_debug_exec_result")
    if res:
        code, out = res
        st.caption(f"Exit code: {code}")
        st.code(out or "(no output)", language=None)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _display_columns_for_containers(rows: list) -> pd.DataFrame:
    """Column order for the live Docker table (hide internal full_id)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    preferred = [
        "name",
        "state",
        "health",
        "compose_service",
        "compose_project",
        "image",
        "id",
    ]
    cols = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in cols and c != "full_id"]
    return df[cols + rest]


def _render_live_stack_mermaid(rows: list) -> None:
    """Help-style Mermaid (Clients / Edge / App Tier / T4 / …) with live container labels."""
    if not rows:
        return

    t4_env = os.environ.get("T4_VISUALIZER_BASE_URL", "").strip()
    if t4_env:
        st.caption(
            "**T4 dataset server** (2D/3D): HTTP API for `/render`, `/viewer/three`, and dataset availability. "
            f"`T4_VISUALIZER_BASE_URL` = `{t4_env}`. "
            "The diagram shows a matching Compose service if present, otherwise a synthetic node for this URL."
        )
    else:
        st.caption(
            "**T4 dataset server** (optional): used by Bounding Box Viewer and T4 3D Viewer. "
            "Set `T4_VISUALIZER_BASE_URL` in `.env` to include it in the diagram (synthetic node). "
            "Compose services named `t4_server`, `t4_visualizer`, or `t4_*` are grouped under **T4 dataset server**."
        )

    # Taller when URL is set or a t4_* Compose service is present (extra subgraph).
    t4_svc = rowset_has_t4_compose_service(rows)
    extra_h = 120 if (t4_env or t4_svc) else 40
    mh = min(920, 280 + 52 * len(rows) + extra_h)
    render_mermaid(live_containers_mermaid(rows), height=mh)


def _render_t4_remote_probe(base_url: str) -> None:
    """Fetch /health and /server/structure.json from the configured T4 visualizer host."""
    base = base_url.rstrip("/")
    section_header(
        "T4 dataset server (HTTP)",
        f"Live probe of `{T4_ENV_BASE_URL}` — same service as Bounding Box / T4 3D pages. "
        "Open the links on the T4 host for the server’s own HTML diagram and diagnostics.",
    )
    st.markdown(
        f"**On the T4 host:** [Structure (HTML)]({base}/server/structure) · "
        f"[structure.json]({base}/server/structure.json) · "
        f"[Health]({base}/health) · "
        f"[Browser diagnostics]({base}/browser/diagnostics)"
    )
    try:
        client = T4VisualizerClient(base_url=base, timeout=8.0)
        health = client.health()
    except T4VisualizerError as ex:
        st.warning(f"Could not reach T4 server (`GET /health`): {ex}")
        return
    except OSError as ex:
        st.warning(f"Could not reach T4 server: {ex}")
        return

    st.caption("GET /health")
    st.json(health)

    try:
        structure = client.server_structure_json()
    except T4VisualizerError as ex:
        if ex.status_code == 404:
            st.info(
                "This T4 server does not expose `/server/structure.json` yet. "
                "Upgrade **t4-server** (evaluator_result_parser) or use the links above if the server is older."
            )
        else:
            st.warning(f"`GET /server/structure.json` failed: {ex}")
        return

    mmd = structure.get("mermaid") or ""
    if mmd:
        st.caption("Internal architecture (returned by t4-server — same diagram as `/server/structure`)")
        mh = min(520, 160 + mmd.count("\n") * 26)
        render_mermaid(mmd, height=mh)
    meta = structure.get("meta")
    if isinstance(meta, dict) and meta:
        st.caption("Server meta (uptime, caches, diagnostics)")
        st.json(meta)


with tab_docker:
    client = docker_client_or_none()

    if client is None:
        if not _env_flag("EVAL_DEPLOYMENT_DEBUG_DOCKER"):
            _render_docker_disabled(
                "Docker debug is off: set `EVAL_DEPLOYMENT_DEBUG_DOCKER=1` and mount the host Docker socket "
                "into the Streamlit container (see compose comments)."
            )
        elif not is_docker_debug_enabled():
            _render_docker_disabled(
                "`EVAL_DEPLOYMENT_DEBUG_DOCKER` is set, but the Docker Unix socket is not available inside this container "
                "(or `DOCKER_HOST` points to a non-Unix endpoint that is unreachable)."
            )
        else:
            try:
                import docker as _docker_check  # noqa: F401
            except ImportError:
                _render_docker_disabled(
                    "Docker debug is enabled and the socket is present, but the `docker` Python package is not installed."
                )
            else:
                _render_docker_disabled(
                    "`docker.from_env()` failed — check socket permissions (Streamlit user must read/write the socket) "
                    "or daemon availability."
                )
    else:
        proj = compose_project_filter()

        _use_fragment = getattr(st, "fragment", None) is not None

        if _use_fragment:

            @st.fragment(run_every=timedelta(seconds=6))
            def _docker_fragment():
                rows, list_warn = list_containers_for_debug(client)
                st.caption(f"Last refreshed (server clock): **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}** — updates about every 6 s.")
                if list_warn and isinstance(list_warn, str) and list_warn.startswith("Docker list failed"):
                    st.error(list_warn)
                    return
                if list_warn:
                    st.markdown(list_warn)
                t4_probe_url = os.environ.get("T4_VISUALIZER_BASE_URL", "").strip()
                if not rows:
                    st.info("No containers match the current filter.")
                    if t4_probe_url:
                        _render_t4_remote_probe(t4_probe_url)
                    return
                section_header("Live container table", "Sortable columns; `full_id` stays internal for log/exec.")
                display_df = _display_columns_for_containers(rows)
                st.dataframe(display_df, width='stretch', hide_index=True)
                _render_live_stack_mermaid(rows)
                if t4_probe_url:
                    _render_t4_remote_probe(t4_probe_url)

                options = [f"{r['name']} ({r['id']})" for r in rows]
                id_by_label = {f"{r['name']} ({r['id']})": r["full_id"] for r in rows}

                prev_cid = st.session_state.get("deploy_debug_cid")
                default_ix = 0
                if prev_cid:
                    for i, opt in enumerate(options):
                        if id_by_label[opt] == prev_cid:
                            default_ix = i
                            break

                pick = st.selectbox("Container", options=options, index=default_ix, key="deploy_debug_pick")
                full_id = id_by_label[pick]
                st.session_state.deploy_debug_cid = full_id

                section_header("Logs", "Stdout/stderr from the selected container.")
                tail = st.slider(
                    "Log tail (lines)",
                    min_value=50,
                    max_value=MAX_LOG_TAIL_LINES,
                    value=300,
                    step=50,
                    key="deploy_debug_tail",
                )
                logs = container_logs_tail(client, full_id, tail)
                st.code(logs or "(empty)", language=None)
                _render_docker_exec_ui(client, full_id)

            _docker_fragment()
        else:
            rows, list_warn = list_containers_for_debug(client)
            st.caption(f"Loaded at **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}** — use Refresh to re-query.")
            if list_warn and isinstance(list_warn, str) and list_warn.startswith("Docker list failed"):
                st.error(list_warn)
            elif list_warn:
                st.markdown(list_warn)
            t4_probe_url = os.environ.get("T4_VISUALIZER_BASE_URL", "").strip()
            if not rows:
                st.info("No containers match the current filter.")
                if t4_probe_url:
                    _render_t4_remote_probe(t4_probe_url)
            else:
                _render_live_stack_mermaid(rows)
                if t4_probe_url:
                    _render_t4_remote_probe(t4_probe_url)
                section_header("Live container table", "Sortable columns; `full_id` stays internal for log/exec.")
                display_df = _display_columns_for_containers(rows)
                st.dataframe(display_df, width='stretch', hide_index=True)
                options = [f"{r['name']} ({r['id']})" for r in rows]
                id_by_label = {f"{r['name']} ({r['id']})": r["full_id"] for r in rows}
                pick = st.selectbox("Container", options=options, key="deploy_debug_pick_legacy")
                section_header("Logs", "Stdout/stderr from the selected container.")
                tail = st.slider(
                    "Log tail (lines)",
                    min_value=50,
                    max_value=MAX_LOG_TAIL_LINES,
                    value=300,
                    step=50,
                    key="deploy_debug_tail_legacy",
                )
                full_id_legacy = id_by_label[pick]
                logs = container_logs_tail(client, full_id_legacy, tail)
                st.code(logs or "(empty)", language=None)
                _render_docker_exec_ui(client, full_id_legacy)
                if st.button("Refresh container list"):
                    st.rerun()

    st.page_link("pages/10_Help.py", label="Help & guide (full README, including static stack Mermaid)", icon="❔")
