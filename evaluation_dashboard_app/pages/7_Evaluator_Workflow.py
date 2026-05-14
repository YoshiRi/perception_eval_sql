"""
Evaluator Workflow page:
- browse finished local runs and launch compare views
- monitor server-side tasks
- start new evaluator pipelines
- run download/eval from existing evaluator jobs
"""

from __future__ import annotations

import html
import json
import os
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from lib.db import create_task, is_task_queue_enabled, list_recent_tasks, update_task_rq_job_id
from lib.page_chrome import inject_app_page_styles, render_page_hero, section_header
from lib.path_utils import (
    format_size,
    get_data_root_display,
    get_run_info,
    list_run_directories,
    resolve_under_data_root,
)
from lib.ui.recent_evaluator_jobs import (
    _render_recent_evaluator_jobs_section,
    configure_recent_evaluator_jobs_ui,
)
from lib.ui.task_history import get_task_list_current_user, render_task_list
from lib.ui.styles_download import inject_download_page_styles
from lib.user_config import UserConfig

try:
    from lib.perception_catalog_io import pkl_archive_to_parquet

    CATALOG_IO_AVAILABLE = True
except ImportError:
    CATALOG_IO_AVAILABLE = False

_JST = timezone(timedelta(hours=9))
_TASK_LIST_MAX_ROWS = 200
_TASK_LIST_SINCE_DAYS = 7


st.set_page_config(
    page_title="Evaluator Workflow",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_app_page_styles()
inject_download_page_styles()


_user_config = UserConfig(warning_fn=st.warning)


def get_config_value(key: str, default=None):
    return _user_config.get(key, default)


def set_config_value(key: str, value) -> None:
    _user_config.set(key, value)


def _to_jst(dt):
    if dt is None:
        return None
    try:
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_JST)
    except Exception:
        return None


def _load_catalog_presets():
    app_root = Path(__file__).parent.parent
    catalogs_filename = "catalogs.json"
    search_paths = [
        app_root / catalogs_filename,
        Path(os.environ.get("CATALOGS_PATH", "")),
        Path.cwd() / catalogs_filename,
    ]
    catalogs = []
    loaded_path = None
    load_error = None
    for path in search_paths:
        if path.exists() and path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    catalogs = data.get("catalogs", [])
                elif isinstance(data, list):
                    catalogs = data
                else:
                    catalogs = []
                loaded_path = str(path)
                load_error = None
                break
            except Exception as exc:
                load_error = str(exc)
    presets = []
    for item in catalogs:
        if not isinstance(item, dict):
            continue
        display_name = item.get("display_name") or item.get("name") or item.get("catalog_id", "Unknown")
        presets.append({**item, "display_name": display_name})
    return presets, loaded_path, load_error


def _enqueue_task(task_type: str, params: dict) -> Optional[str]:
    try:
        session_id = get_task_list_current_user()
        task_id = create_task(task_type, params, session_id=session_id)
        if not task_id:
            return None

        from redis import Redis
        from rq import Queue
        from worker.tasks import run_job

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        redis_conn = Redis.from_url(redis_url)
        queue = Queue(
            name=os.environ.get("RQ_QUEUE", "default"),
            connection=redis_conn,
            default_timeout="7d",
        )
        job = queue.enqueue(
            run_job,
            task_id,
            task_type,
            params,
            job_timeout="7d",
            result_ttl="7d",
        )
        rq_id = getattr(job, "id", None)
        if rq_id:
            update_task_rq_job_id(task_id, str(rq_id))
        return task_id
    except Exception as exc:
        st.error(f"Failed to enqueue task: {exc}")
        return None


def _make_default_output_path(branch_name: str) -> str:
    import re

    clean_branch = re.sub(r"[^\w]", "_", branch_name.strip("/")) if branch_name else "eval"
    clean_branch = re.sub(r"_+", "_", clean_branch).strip("_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{clean_branch}_{ts}"


def _format_run_mtime(mtime: float) -> str:
    if not mtime:
        return "—"
    try:
        return datetime.fromtimestamp(mtime, tz=_JST).strftime("%Y-%m-%d %H:%M JST")
    except Exception:
        return "—"


def _build_overview_url(run_a: str, run_b: Optional[str] = None) -> str:
    query = {"mode": "compare" if run_b else "single", "run_a": run_a}
    if run_b:
        query["run_b"] = run_b
    return f"/?{urllib.parse.urlencode(query)}"


def _inject_workflow_page_styles() -> None:
    st.markdown(
        """
        <style>
        .wf-toolbar-note {
            margin: 0 0 0.28rem 0;
            font-size: 0.66rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 700;
        }
        .wf-panel {
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.98) 100%);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 18px 50px -28px rgba(15, 23, 42, 0.22);
            margin-bottom: 1rem;
        }
        .wf-panel-title {
            margin: 0;
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
        }
        .wf-panel-copy {
            margin: 0.3rem 0 0 0;
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .wf-meta-inline {
            margin-top: 0.2rem;
            color: #64748b;
            font-size: 0.78rem;
        }
        .wf-run-list {
            display: block;
            margin-top: 0.35rem;
        }
        .wf-run-card {
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 16px;
            padding: 0.95rem 1rem;
        }
        .wf-run-row {
            display: grid;
            grid-template-columns: minmax(0, 2.3fr) minmax(110px, 0.95fr) minmax(92px, 0.85fr) minmax(0, 1.1fr);
            gap: 0.95rem;
            align-items: center;
        }
        .wf-run-name {
            min-width: 0;
        }
        .wf-run-title {
            font-size: 0.96rem;
            line-height: 1.25;
            font-weight: 700;
            color: #0f172a;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .wf-run-sub {
            margin-top: 0.18rem;
            color: #64748b;
            font-size: 0.78rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .wf-run-cell {
            min-width: 0;
            color: #0f172a;
            font-size: 0.85rem;
            line-height: 1.35;
        }
        .wf-run-cell strong {
            display: block;
            font-size: 0.88rem;
        }
        .wf-run-flags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.36rem;
        }
        .wf-flag {
            display: inline-flex;
            align-items: center;
            padding: 0.22rem 0.5rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            background: #e2e8f0;
            color: #475569;
        }
        .wf-flag--ok {
            background: #dcfce7;
            color: #166534;
        }
        .wf-compare-bar {
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: linear-gradient(135deg, #f8fafc 0%, #ecfeff 100%);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin: 0.45rem 0 0.75rem 0;
        }
        .wf-compare-title {
            margin: 0;
            font-size: 0.84rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: 0.01em;
        }
        .wf-start-note {
            border: 1px solid rgba(20, 184, 166, 0.22);
            background: linear-gradient(135deg, #f0fdfa 0%, #ffffff 100%);
            border-radius: 16px;
            padding: 1rem;
            min-height: 100%;
        }
        .wf-start-note strong {
            color: #0f172a;
        }
        .wf-start-note p {
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.55;
            margin: 0.45rem 0 0 0;
        }
        .wf-empty {
            border: 1px dashed rgba(148, 163, 184, 0.45);
            border-radius: 16px;
            background: rgba(248, 250, 252, 0.8);
            padding: 1rem;
            color: #475569;
            font-size: 0.9rem;
        }
        @media (max-width: 1080px) {
            .wf-run-row {
                grid-template-columns: 1fr;
                gap: 0.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_local_run_card(run: Dict[str, object]) -> None:
    name = html.escape(str(run["name"]))
    rel_path = html.escape(str(run["path_display"]))
    modified = html.escape(str(run["modified"]))
    size = html.escape(str(run["size"]))
    flags = [
        ("Summary", bool(run["has_summary"])),
        ("Score", bool(run["has_score"])),
        ("Parquet", bool(run["has_parquet"])),
    ]
    flag_html = "".join(
        f'<span class="wf-flag {"wf-flag--ok" if enabled else ""}">{label}</span>'
        for label, enabled in flags
    )
    st.markdown(
        f"""
        <div class="wf-run-card">
          <div class="wf-run-row">
            <div class="wf-run-name">
              <div class="wf-run-title">{name}</div>
              <div class="wf-run-sub">{rel_path}</div>
            </div>
            <div class="wf-run-cell">
              <strong>{modified}</strong>
              <span class="wf-run-sub">last updated</span>
            </div>
            <div class="wf-run-cell">
              <strong>{size}</strong>
              <span class="wf-run-sub">disk usage</span>
            </div>
            <div class="wf-run-cell">
              <div class="wf-run-flags">{flag_html}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_local_runs_section() -> None:
    section_header(
        "Local Runs",
        f"Finished runs already stored under `{get_data_root_display()}/`. Search, browse, and pick two to compare.",
    )
    run_dirs = list_run_directories()
    if not run_dirs:
        st.markdown('<div class="wf-empty">No finished runs were found on this server yet.</div>', unsafe_allow_html=True)
        return

    runs: List[Dict[str, object]] = []
    for run_path in run_dirs:
        info = get_run_info(run_path)
        runs.append(
            {
                "name": info["name"],
                "path_display": f"{get_data_root_display()}/{info['name']}",
                "size": format_size(info["size_bytes"]),
                "mtime": float(info["mtime"] or 0),
                "modified": _format_run_mtime(info["mtime"]),
                "has_summary": bool(info["has_summary"]),
                "has_score": bool(info["has_score"]),
                "has_parquet": bool(info["has_parquet"]),
            }
        )
    runs.sort(key=lambda row: (-float(row["mtime"]), str(row["name"]).lower()))

    control_cols = st.columns([1.7, 0.8, 0.8, 0.55])
    with control_cols[0]:
        st.markdown('<div class="wf-toolbar-note">Search</div>', unsafe_allow_html=True)
        run_search = st.text_input(
            "Search runs",
            value=st.session_state.get("workflow_runs_search", ""),
            key="workflow_runs_search",
            label_visibility="collapsed",
            placeholder="Filter by run name",
        ).strip().lower()
    with control_cols[1]:
        st.markdown('<div class="wf-toolbar-note">Summary</div>', unsafe_allow_html=True)
        require_summary = st.selectbox(
            "Require summary",
            options=["Any", "Yes", "No"],
            index=0,
            key="workflow_runs_summary_filter",
            label_visibility="collapsed",
        )
    with control_cols[2]:
        st.markdown('<div class="wf-toolbar-note">Parquet</div>', unsafe_allow_html=True)
        require_parquet = st.selectbox(
            "Require parquet",
            options=["Any", "Yes", "No"],
            index=0,
            key="workflow_runs_parquet_filter",
            label_visibility="collapsed",
        )
    with control_cols[3]:
        st.markdown('<div class="wf-toolbar-note">Rows</div>', unsafe_allow_html=True)
        page_size = int(
            st.selectbox(
                "Rows per page",
                options=[6, 10, 14, 20],
                index=1,
                key="workflow_runs_page_size",
                label_visibility="collapsed",
            )
        )

    filtered = runs
    if run_search:
        filtered = [row for row in filtered if run_search in str(row["name"]).lower()]
    if require_summary != "Any":
        want = require_summary == "Yes"
        filtered = [row for row in filtered if bool(row["has_summary"]) == want]
    if require_parquet != "Any":
        want = require_parquet == "Yes"
        filtered = [row for row in filtered if bool(row["has_parquet"]) == want]

    run_names = [str(row["name"]) for row in filtered]
    compare_ready = [
        str(row["name"])
        for row in filtered
        if bool(row["has_summary"]) or bool(row["has_score"]) or bool(row["has_parquet"])
    ]
    if "workflow_compare_run_a" not in st.session_state:
        st.session_state["workflow_compare_run_a"] = compare_ready[0] if compare_ready else ""
    if "workflow_compare_run_b" not in st.session_state:
        st.session_state["workflow_compare_run_b"] = compare_ready[1] if len(compare_ready) > 1 else ""

    st.markdown('<div class="wf-compare-bar">', unsafe_allow_html=True)
    st.markdown('<p class="wf-compare-title">Quick compare tray</p>', unsafe_allow_html=True)
    compare_cols = st.columns([1.35, 1.35, 0.95, 0.95])
    with compare_cols[0]:
        st.markdown('<div class="wf-toolbar-note">Baseline A</div>', unsafe_allow_html=True)
        run_a = st.selectbox(
            "Baseline A",
            options=[""] + compare_ready,
            index=([""] + compare_ready).index(st.session_state.get("workflow_compare_run_a", ""))
            if st.session_state.get("workflow_compare_run_a", "") in compare_ready
            else 0,
            key="workflow_compare_run_a",
            label_visibility="collapsed",
        )
    with compare_cols[1]:
        st.markdown('<div class="wf-toolbar-note">Candidate B</div>', unsafe_allow_html=True)
        run_b_options = [""] + [name for name in compare_ready if name != run_a]
        current_b = st.session_state.get("workflow_compare_run_b", "")
        st.selectbox(
            "Candidate B",
            options=run_b_options,
            index=run_b_options.index(current_b) if current_b in run_b_options else 0,
            key="workflow_compare_run_b",
            label_visibility="collapsed",
        )
    run_b = st.session_state.get("workflow_compare_run_b", "")
    with compare_cols[2]:
        st.markdown('<div class="wf-toolbar-note">Single</div>', unsafe_allow_html=True)
        if run_a:
            st.link_button("Open run", _build_overview_url(run_a), use_container_width=True)
        else:
            st.button("Open run", disabled=True, use_container_width=True, key="workflow_open_run_disabled")
    with compare_cols[3]:
        st.markdown('<div class="wf-toolbar-note">Compare</div>', unsafe_allow_html=True)
        if run_a and run_b:
            st.link_button("Compare", _build_overview_url(run_a, run_b), use_container_width=True)
        else:
            st.button("Compare", disabled=True, use_container_width=True, key="workflow_compare_run_disabled")
    st.markdown("</div>", unsafe_allow_html=True)

    if not filtered:
        st.markdown('<div class="wf-empty">No local runs matched the current filters.</div>', unsafe_allow_html=True)
        return

    page_key = "workflow_runs_page"
    current_page = max(1, int(st.session_state.get(page_key, 1)))
    page_count = max(1, (len(filtered) + page_size - 1) // page_size)
    if current_page > page_count:
        current_page = page_count
        st.session_state[page_key] = current_page
    start_idx = (current_page - 1) * page_size
    visible_runs = filtered[start_idx:start_idx + page_size]

    pager_cols = st.columns([0.7, 0.8, 0.8, 0.8, 5.9])
    with pager_cols[0]:
        if st.button("‹", key="workflow_runs_prev", use_container_width=True, disabled=current_page <= 1):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    page_numbers = (
        list(range(1, min(3, page_count) + 1))
        if current_page == 1
        else list(range(max(1, current_page - 1), min(page_count, current_page + 1) + 1))
    )
    for idx, page_num in enumerate(page_numbers[:3], start=1):
        with pager_cols[idx]:
            if st.button(
                str(page_num),
                key=f"workflow_runs_page_{page_num}",
                use_container_width=True,
                disabled=page_num == current_page,
            ):
                st.session_state[page_key] = page_num
                st.rerun()
    with pager_cols[4]:
        if st.button("›", key="workflow_runs_next", use_container_width=True, disabled=current_page >= page_count):
            st.session_state[page_key] = current_page + 1
            st.rerun()

    st.markdown('<div class="wf-run-list">', unsafe_allow_html=True)
    for run in visible_runs:
        row_cols = st.columns([8.9, 2.6])
        with row_cols[0]:
            _render_local_run_card(run)
        with row_cols[1]:
            action_cols = st.columns([1.0, 1.0, 1.0], gap="small")
            with action_cols[0]:
                st.link_button(
                    "Open",
                    _build_overview_url(str(run["name"])),
                    use_container_width=True,
                )
            with action_cols[1]:
                if st.button(f"A", key=f"workflow_pick_a_{run['name']}", use_container_width=True):
                    st.session_state["workflow_compare_run_a"] = str(run["name"])
                    st.rerun()
            with action_cols[2]:
                if st.button(f"B", key=f"workflow_pick_b_{run['name']}", use_container_width=True):
                    st.session_state["workflow_compare_run_b"] = str(run["name"])
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _render_current_tasks_section() -> None:
    section_header(
        "Current Tasks",
        "Jobs queued or running on this server, with recent history folded underneath.",
    )
    if not is_task_queue_enabled():
        st.info("Task queue not enabled. Set `USE_TASK_QUEUE=true` to track background tasks.")
        return

    current_user = get_task_list_current_user()
    use_fragment = getattr(st, "fragment", None) is not None
    if use_fragment:
        try:

            @st.fragment(run_every=timedelta(seconds=3))
            def _task_list_poll():
                current_tasks = list_recent_tasks(
                    limit=_TASK_LIST_MAX_ROWS,
                    session_id=current_user,
                    since_days=_TASK_LIST_SINCE_DAYS,
                )
                render_task_list(current_tasks, current_user)

            _task_list_poll()
            return
        except (TypeError, AttributeError):
            use_fragment = False

    tasks = list_recent_tasks(
        limit=_TASK_LIST_MAX_ROWS,
        session_id=current_user,
        since_days=_TASK_LIST_SINCE_DAYS,
    )
    has_active = render_task_list(tasks, current_user)
    if st.button("Refresh tasks", key="workflow_refresh_tasks"):
        st.rerun()
    if has_active:
        st.caption("Active jobs are shown live when possible. Use refresh if this browser does not support fragments.")


def _render_start_workflow_section(
    catalog_presets: List[Dict[str, str]],
    catalogs_path: Optional[str],
    catalog_load_error: Optional[str],
) -> Dict[str, object]:
    section_header(
        "Start Workflow",
        "Schedule a fresh evaluator run here, or use the recent evaluator jobs browser below to run Download + Eval from an existing report.",
    )

    if catalog_load_error:
        st.warning(f"Could not read catalog presets: {catalog_load_error}")
    elif catalogs_path:
        st.caption(f"Catalog presets loaded from `{catalogs_path}`.")

    catalog_names = [item["display_name"] for item in catalog_presets]
    default_project = get_config_value("eval_project_id", "x2_dev")
    default_target = get_config_value("target_name", "beta/v4.3.2")
    default_download_type = get_config_value("eval_download_type", "Archives (ZIP)")
    default_phase = get_config_value(
        "eval_phase",
        "perception.object_recognition.tracking.objects",
    )
    default_poll_interval = int(get_config_value("poll_interval", 60))
    default_max_wait_hours = int(get_config_value("max_wait_hours", 24))
    default_environment = get_config_value("environment", "")
    default_output = get_config_value("eval_output_path", _make_default_output_path(default_target))

    top_cols = st.columns([1.0, 1.5, 1.2])
    with top_cols[0]:
        st.markdown('<div class="wf-toolbar-note">Project</div>', unsafe_allow_html=True)
        project_id = st.text_input(
            "Project ID",
            value=default_project,
            key="workflow_project_id",
            label_visibility="collapsed",
        ).strip()
    with top_cols[1]:
        st.markdown('<div class="wf-toolbar-note">Catalog</div>', unsafe_allow_html=True)
        selected_catalog_name = st.selectbox(
            "Catalog",
            options=catalog_names if catalog_names else ["No catalog presets"],
            index=0,
            key="workflow_catalog_name",
            label_visibility="collapsed",
        )
    selected_catalog = next(
        (item for item in catalog_presets if item["display_name"] == selected_catalog_name),
        None,
    )
    with top_cols[2]:
        st.markdown('<div class="wf-toolbar-note">Branch or tag</div>', unsafe_allow_html=True)
        target_name = st.text_input(
            "Branch or Tag",
            value=default_target,
            key="workflow_target_name",
            label_visibility="collapsed",
            placeholder="beta/v4.3.2",
        ).strip()

    detail_cols = st.columns([1.25, 0.8, 0.95])
    with detail_cols[0]:
        st.markdown('<div class="wf-toolbar-note">Output folder</div>', unsafe_allow_html=True)
        output_path = st.text_input(
            "Output folder",
            value=default_output,
            key="workflow_output_path",
            label_visibility="collapsed",
            placeholder=_make_default_output_path(target_name),
        ).strip()
    with detail_cols[1]:
        st.markdown('<div class="wf-toolbar-note">Environment</div>', unsafe_allow_html=True)
        environment = st.selectbox(
            "Environment",
            options=["", "dev", "stg", "prd"],
            index=["", "dev", "stg", "prd"].index(default_environment) if default_environment in ("", "dev", "stg", "prd") else 0,
            key="workflow_environment",
            label_visibility="collapsed",
            format_func=lambda value: value or "default",
        )
    with detail_cols[2]:
        st.markdown('<div class="wf-toolbar-note">Description</div>', unsafe_allow_html=True)
        description = st.text_input(
            "Description",
            value=get_config_value("workflow_description", ""),
            key="workflow_description",
            label_visibility="collapsed",
            placeholder="Optional label for the evaluator run",
        ).strip()

    if selected_catalog:
        info_cols = st.columns([1.2, 1.15, 2.2])
        with info_cols[0]:
            st.markdown(f'<div class="wf-meta-inline"><strong>Catalog ID</strong><br>{html.escape(str(selected_catalog.get("catalog_id", "—")))}</div>', unsafe_allow_html=True)
        with info_cols[1]:
            st.markdown(f'<div class="wf-meta-inline"><strong>Integration</strong><br>{html.escape(str(selected_catalog.get("integration_id", "—")))}</div>', unsafe_allow_html=True)
        with info_cols[2]:
            desc = str(selected_catalog.get("description") or "").strip() or "Preset selected for quick scheduling."
            st.markdown(f'<div class="wf-meta-inline"><strong>Preset</strong><br>{html.escape(desc)}</div>', unsafe_allow_html=True)

    with st.expander("Advanced options", expanded=False):
        adv_cols = st.columns([1.0, 1.2, 0.8, 0.8])
        with adv_cols[0]:
            download_type = st.radio(
                "Download type",
                ["Archives (ZIP)", "Result JSON"],
                horizontal=True,
                index=0 if default_download_type == "Archives (ZIP)" else 1,
                key="workflow_download_type",
            )
        with adv_cols[1]:
            phase = st.text_input(
                "Phase",
                value=default_phase,
                key="workflow_phase",
                disabled=download_type != "Archives (ZIP)",
            )
        with adv_cols[2]:
            poll_interval = st.slider(
                "Poll interval (s)",
                min_value=10,
                max_value=300,
                value=default_poll_interval,
                step=10,
                key="workflow_poll_interval",
            )
        with adv_cols[3]:
            max_wait_hours = st.slider(
                "Max wait (h)",
                min_value=1,
                max_value=168,
                value=default_max_wait_hours,
                key="workflow_max_wait_hours",
            )

        option_cols = st.columns(4)
        with option_cols[0]:
            run_eval = st.checkbox("Run evaluation", value=True, key="workflow_run_eval")
        with option_cols[1]:
            generate_parquet = st.checkbox(
                "Generate parquet",
                value=CATALOG_IO_AVAILABLE,
                disabled=not CATALOG_IO_AVAILABLE,
                key="workflow_generate_parquet",
            )
        with option_cols[2]:
            eval_recursive = st.checkbox("Recursive scan", value=True, key="workflow_eval_recursive")
        with option_cols[3]:
            is_tag = st.checkbox("Target is tag", value=False, key="workflow_is_tag")

    start_cols = st.columns([1.45, 0.95])
    with start_cols[0]:
        st.markdown('<div class="wf-panel">', unsafe_allow_html=True)
        st.markdown('<p class="wf-panel-title">Schedule evaluator + download + eval</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="wf-panel-copy">This starts the same background pipeline as the previous workflow launcher, but keeps the controls on-page. Output path, evaluator polling, and eval/parquet behavior are all preserved.</p>',
            unsafe_allow_html=True,
        )
        start_clicked = st.button(
            "Start evaluator workflow",
            key="workflow_start_btn",
            type="primary",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with start_cols[1]:
        st.markdown(
            """
            <div class="wf-start-note">
              <strong>Already have an evaluator report?</strong>
              <p>Use the recent evaluator jobs section right below. Every row can open details or run Download + Eval + Parquet directly, using the same defaults configured on this page.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    set_config_value("eval_project_id", project_id)
    set_config_value("target_name", target_name)
    set_config_value("eval_output_path", output_path)
    set_config_value("eval_download_type", download_type)
    set_config_value("eval_phase", phase)
    set_config_value("poll_interval", poll_interval)
    set_config_value("max_wait_hours", max_wait_hours)
    set_config_value("environment", environment)
    set_config_value("workflow_description", description)

    catalog_id = str((selected_catalog or {}).get("catalog_id") or "").strip()
    integration_id = str((selected_catalog or {}).get("integration_id") or "").strip()
    errors = []
    if not project_id:
        errors.append("Project ID")
    if not catalog_id:
        errors.append("Catalog")
    if not integration_id:
        errors.append("Integration ID")
    if not target_name:
        errors.append("Branch or tag")

    resolved_output = None
    path_error = ""
    if output_path:
        resolved_output, path_error = resolve_under_data_root(output_path, allow_create=True)
        if path_error:
            errors.append(path_error)
    else:
        errors.append("Output folder")

    if start_clicked:
        if errors:
            for err in errors:
                st.error(f"Missing or invalid: {err}")
        elif not is_task_queue_enabled():
            st.error("Task queue not enabled. Set `USE_TASK_QUEUE=true` and `REDIS_URL`.")
        else:
            task_id = _enqueue_task(
                "run_evaluator_and_process",
                {
                    "project_id": project_id,
                    "catalog_id": catalog_id,
                    "integration_id": integration_id,
                    "suite_ids": None,
                    "target_name": target_name,
                    "description": description or f"Eval {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "output_path": str(resolved_output),
                    "environment": environment,
                    "max_retries": 0,
                    "clean_build": False,
                    "debug": False,
                    "is_tag": is_tag,
                    "download_type": "archives" if download_type == "Archives (ZIP)" else "result_json",
                    "phase": phase,
                    "skip_large_file": False,
                    "large_file_mb": 50.0,
                    "keep_zip_files": False,
                    "poll_interval": int(poll_interval),
                    "max_wait_seconds": int(max_wait_hours) * 3600,
                    "run_eval": bool(run_eval),
                    "generate_parquet": bool(generate_parquet),
                    "eval_recursive": bool(eval_recursive),
                    "eval_overwrite": False,
                },
            )
            if task_id:
                st.success(f"Workflow queued. Task id: `{task_id}`")
            else:
                st.error("Failed to enqueue task. Check worker logs.")

    return {
        "project_id": project_id,
        "environment": environment,
        "output_path_default": output_path or _make_default_output_path(target_name),
        "download_type_default": download_type,
        "phase_default": phase,
        "skip_large_file_default": False,
        "large_file_mb_default": 50.0,
        "keep_zip_files_default": False,
    }


_inject_workflow_page_styles()
render_page_hero(
    kicker="Workflow automation",
    title="Evaluator Workflow",
    description="Browse finished runs, watch background tasks, launch fresh evaluator pipelines, and reuse existing evaluator reports from one aligned workspace.",
)

catalog_presets, catalogs_path, catalog_load_error = _load_catalog_presets()

_render_local_runs_section()
_render_current_tasks_section()
start_defaults = _render_start_workflow_section(catalog_presets, catalogs_path, catalog_load_error)

configure_recent_evaluator_jobs_ui(
    get_config_value=get_config_value,
    set_config_value=set_config_value,
    enqueue_task=_enqueue_task,
    catalog_io_available=CATALOG_IO_AVAILABLE,
    environment=str(start_defaults["environment"] or ""),
)

section_header(
    "Recent Evaluator Jobs",
    "Direct evaluator browser for starting Download + Eval from existing reports. Shown by default here so the existing-job path is one click away.",
)
_render_recent_evaluator_jobs_section(
    str(start_defaults["project_id"] or ""),
    str(start_defaults["environment"] or ""),
    output_path_default=str(start_defaults["output_path_default"]),
    download_type_default=str(start_defaults["download_type_default"]),
    phase_default=str(start_defaults["phase_default"]),
    skip_large_file_default=bool(start_defaults["skip_large_file_default"]),
    large_file_mb_default=float(start_defaults["large_file_mb_default"]),
    keep_zip_files_default=bool(start_defaults["keep_zip_files_default"]),
    show_toggle=False,
    default_visible=True,
    show_title=False,
)
