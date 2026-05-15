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
import requests

from lib.db import create_task, is_task_queue_enabled, list_recent_tasks, update_task_rq_job_id
from lib.page_chrome import (
    inject_app_page_styles,
    render_page_hero,
    section_header,
)
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


def _fetch_server_catalogs(project_id: str, environment: str) -> List[Dict[str, str]]:
    """Fetch available catalogs for the project on demand."""
    if not project_id:
        return []
    import os
    from lib.WebAPI import catalogAPI

    os.environ["AUTH_PROFILE"] = environment or "default"
    response = catalogAPI(project_id=project_id).list_catalogs()
    response.raise_for_status()
    data = response.json()
    raw_catalogs = data.get("catalogs", []) if isinstance(data, dict) else data
    options: List[Dict[str, str]] = []
    for item in raw_catalogs or []:
        if not isinstance(item, dict):
            continue
        catalog_id = str(item.get("id") or item.get("catalog_id") or "").strip()
        display_name = str(item.get("display_name") or item.get("name") or catalog_id).strip()
        if not catalog_id or not display_name:
            continue
        options.append(
            {
                "catalog_id": catalog_id,
                "display_name": display_name,
                "description": str(item.get("description") or "").strip(),
            }
        )
    options.sort(key=lambda item: item["display_name"].lower())
    return options


def _resolve_integration_id_for_catalog(project_id: str, environment: str, catalog_id: str) -> str:
    """Resolve the most relevant active integration for a catalog."""
    if not project_id or not catalog_id:
        return ""
    from lib import evaluator_api

    os.environ["AUTH_PROFILE"] = environment or "default"
    api = evaluator_api.EvaluationRunAPI()
    url = f"{api.api_base_url}/projects/{project_id}/integrations"
    response = api.request(url, {"catalog_id": catalog_id, "size": 100}, method="GET")
    if response is None:
        raise RuntimeError("No response returned while loading integrations.")
    if response.status_code != 200:
        raise RuntimeError(f"Failed to load integrations: status={response.status_code}")

    payload = json.loads(response.content)
    integrations = payload.get("integrations", []) or []
    active = [
        item for item in integrations
        if isinstance(item, dict)
        and str(item.get("catalog_id") or "").strip() == catalog_id
        and not bool(item.get("deleted"))
    ]
    if not active:
        raise RuntimeError("No active integration was found for the selected catalog.")

    def _sort_key(item: Dict[str, object]) -> tuple:
        return (
            str(item.get("updated_at") or ""),
            int(item.get("version_id") or 0),
            str(item.get("id") or ""),
        )

    active.sort(key=_sort_key, reverse=True)
    return str(active[0].get("id") or "").strip()


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


def _catalog_preset_emoji(preset_name: str, *, has_custom_catalog: bool = False) -> str:
    mapping = {
        "Build Test Catalog": "🛠️",
        "Performance Test": "📈",
        "Old performance test": "🕰️",
        "Devops Test": "⚙️",
        "Usecase Performance Catalog": "🧭",
        "L4 regression test": "⚠️",
    }
    normalized = str(preset_name or "").strip()
    if normalized in mapping:
        return mapping[normalized]
    if has_custom_catalog:
        return "🧩"
    return "📦"


def _make_auto_workflow_description(
    target_name: str,
    preset_name: str = "",
    *,
    has_custom_catalog: bool = False,
) -> str:
    import re

    clean_target = str(target_name or "").strip() or "default"
    clean_target = re.sub(r"\s+", " ", clean_target)
    stamp = datetime.now().strftime("%m-%d %H:%M")
    return (
        f"🚀 evaluator workflow [{clean_target}] [{stamp}] "
        f"{_catalog_preset_emoji(preset_name, has_custom_catalog=has_custom_catalog)}"
    )


def _format_run_mtime(mtime: float) -> str:
    if not mtime:
        return "—"
    try:
        return datetime.fromtimestamp(mtime, tz=_JST).strftime("%Y-%m-%d %H:%M JST")
    except Exception:
        return "—"


def _build_overview_url(run_a: str, compare_runs: Optional[List[str]] = None) -> str:
    query = {"mode": "single", "run_a": run_a}
    valid_compare_runs = [str(name).strip() for name in (compare_runs or []) if str(name).strip()]
    if valid_compare_runs:
        query["mode"] = "compare"
        for idx, run_name in enumerate(valid_compare_runs[:4]):
            query[f"run_{chr(98 + idx)}"] = run_name
    return f"/?{urllib.parse.urlencode(query)}"


@st.cache_data(ttl=15, show_spinner=False)
def _load_local_runs() -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for run_path in list_run_directories():
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
    return runs


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _resolve_subject_name(subject_id: str, environment: str) -> Dict[str, str]:
    subject = str(subject_id or "").strip()
    if not subject or not subject.startswith("t4:"):
        return {"subject_id": subject, "name": subject, "email": ""}
    org_id = os.environ.get(
        "WEBAUTO_ORGANIZATION_ID",
        "5a21621d-6968-4f7d-94f8-99cfb77b6e71",
    ).strip()
    if not org_id:
        return {"subject_id": subject, "name": subject, "email": ""}
    os.environ["AUTH_PROFILE"] = environment or "default"
    from webautoauth.token import HttpService, TokenSource, load_config

    config = load_config()
    token_source = TokenSource(HttpService(config))
    access_token = token_source.get_token().access_token
    quoted_subject = urllib.parse.quote(subject, safe="")
    url = f"https://auth.web.auto/v2/organizations/{org_id}/members/{quoted_subject}"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}", "accept": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "subject_id": str(data.get("subject_id") or subject).strip(),
        "name": str(data.get("name") or subject).strip(),
        "email": str(data.get("email") or "").strip(),
    }


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
        .wf-run-name {
            min-width: 0;
        }
        .wf-run-title {
            font-size: 0.8rem;
            line-height: 1.2;
            font-weight: 700;
            color: #0f172a;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .wf-run-title a {
            color: inherit;
            text-decoration: none;
        }
        .wf-run-title a:hover {
            text-decoration: underline;
        }
        .wf-run-cell {
            min-width: 0;
            color: #0f172a;
            font-size: 0.78rem;
            line-height: 1.15;
        }
        .wf-run-text {
            padding-top: 0.26rem;
        }
        .wf-run-flags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.24rem;
            padding-top: 0.18rem;
        }
        .wf-flag {
            display: inline-flex;
            align-items: center;
            padding: 0.12rem 0.38rem;
            border-radius: 999px;
            font-size: 0.68rem;
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
            border-radius: 12px;
            padding: 0.7rem 0.85rem;
            margin: 0.35rem 0 0.55rem 0;
        }
        .wf-compare-title {
            margin: 0;
            font-size: 0.8rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: 0.01em;
        }
        [class*="st-key-workflow_compare_pick__"] label[data-testid="stWidgetLabel"] {
            display: none;
        }
        [class*="st-key-workflow_compare_pick__"] div[data-testid="stCheckbox"] {
            display: flex;
            justify-content: center;
            padding-top: 0.1rem;
        }
        [class*="st-key-workflow_compare_pick__"] input[type="checkbox"] {
            transform: scale(1.2);
        }
        [class*="st-key-workflow_runs_page_select"] div[data-baseweb="select"] {
            min-height: 2rem;
        }
        .wf-launcher {
            border: 1px solid rgba(20, 184, 166, 0.22);
            background: linear-gradient(135deg, #f0fdfa 0%, #ffffff 100%);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.8rem;
        }
        .wf-launcher-title {
            margin: 0;
            font-size: 0.95rem;
            font-weight: 800;
            color: #0f172a;
        }
        .wf-launcher-copy {
            margin: 0.25rem 0 0 0;
            font-size: 0.84rem;
            color: #475569;
        }
        .wf-launcher-meta {
            margin-top: 0.55rem;
            font-size: 0.78rem;
            color: #475569;
        }
        .wf-empty {
            border: 1px dashed rgba(148, 163, 184, 0.45);
            border-radius: 12px;
            background: rgba(248, 250, 252, 0.8);
            padding: 0.8rem 0.9rem;
            color: #475569;
            font-size: 0.84rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_local_runs_header() -> None:
    header_cols = st.columns([0.7, 2.45, 1.35, 0.95, 1.55], gap="small")
    header_cols[0].markdown('<div class="wf-toolbar-note">Pick</div>', unsafe_allow_html=True)
    header_cols[1].markdown('<div class="wf-toolbar-note">Name</div>', unsafe_allow_html=True)
    header_cols[2].markdown('<div class="wf-toolbar-note">Updated</div>', unsafe_allow_html=True)
    header_cols[3].markdown('<div class="wf-toolbar-note">Size</div>', unsafe_allow_html=True)
    header_cols[4].markdown('<div class="wf-toolbar-note">Files</div>', unsafe_allow_html=True)


def _render_local_run_row(run: Dict[str, object], *, selected: bool) -> bool:
    name_raw = str(run["name"])
    name = html.escape(name_raw)
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
    checkbox_key = f"workflow_compare_pick::{name_raw}"
    if checkbox_key not in st.session_state:
        st.session_state[checkbox_key] = bool(selected)
    row_cols = st.columns([0.7, 2.45, 1.35, 0.95, 1.55], gap="small")
    with row_cols[0]:
        checked = st.checkbox("Select run", key=checkbox_key, label_visibility="collapsed")
    with row_cols[1]:
        st.markdown(
            f'<div class="wf-run-title wf-run-text"><a href="{_build_overview_url(name_raw)}" target="_self">{name}</a></div>',
            unsafe_allow_html=True,
        )
    with row_cols[2]:
        st.markdown(f'<div class="wf-run-cell wf-run-text">{modified}</div>', unsafe_allow_html=True)
    with row_cols[3]:
        st.markdown(f'<div class="wf-run-cell wf-run-text">{size}</div>', unsafe_allow_html=True)
    with row_cols[4]:
        st.markdown(f'<div class="wf-run-cell"><div class="wf-run-flags">{flag_html}</div></div>', unsafe_allow_html=True)
    return bool(checked)


def _render_local_runs_section() -> None:
    section_header("Local Runs", "")
    runs = _load_local_runs()
    if not runs:
        st.markdown('<div class="wf-empty">No finished runs were found on this server yet.</div>', unsafe_allow_html=True)
        return

    if "workflow_runs_search_applied" not in st.session_state:
        st.session_state["workflow_runs_search_applied"] = st.session_state.get("workflow_runs_search", "")
    if "workflow_runs_summary_filter_applied" not in st.session_state:
        st.session_state["workflow_runs_summary_filter_applied"] = bool(st.session_state.get("workflow_runs_summary_filter", False))
    if "workflow_runs_parquet_filter_applied" not in st.session_state:
        st.session_state["workflow_runs_parquet_filter_applied"] = bool(st.session_state.get("workflow_runs_parquet_filter", False))
    if "workflow_runs_page_size_applied" not in st.session_state:
        st.session_state["workflow_runs_page_size_applied"] = int(st.session_state.get("workflow_runs_page_size", 10) or 10)

    with st.form("workflow_local_runs_filters", border=False):
        control_cols = st.columns([2.2, 0.8, 0.8, 0.7, 0.8])
        with control_cols[0]:
            st.markdown('<div class="wf-toolbar-note">Search</div>', unsafe_allow_html=True)
            run_search_input = st.text_input(
                "Search runs",
                value=st.session_state.get("workflow_runs_search_applied", ""),
                key="workflow_runs_search",
                label_visibility="collapsed",
                placeholder="Filter by run name",
            )
        with control_cols[1]:
            st.markdown('<div class="wf-toolbar-note">Summary</div>', unsafe_allow_html=True)
            require_summary_input = st.toggle(
                "Summary only",
                value=bool(st.session_state.get("workflow_runs_summary_filter_applied", False)),
                key="workflow_runs_summary_filter",
                label_visibility="collapsed",
            )
        with control_cols[2]:
            st.markdown('<div class="wf-toolbar-note">Parquet</div>', unsafe_allow_html=True)
            require_parquet_input = st.toggle(
                "Parquet only",
                value=bool(st.session_state.get("workflow_runs_parquet_filter_applied", False)),
                key="workflow_runs_parquet_filter",
                label_visibility="collapsed",
            )
        with control_cols[3]:
            st.markdown('<div class="wf-toolbar-note">Rows</div>', unsafe_allow_html=True)
            page_size_input = int(
                st.selectbox(
                    "Rows",
                    options=[10, 20, 50, 100],
                    index=[10, 20, 50, 100].index(int(st.session_state.get("workflow_runs_page_size_applied", 10) or 10)),
                    key="workflow_runs_page_size",
                    label_visibility="collapsed",
                )
            )
        with control_cols[4]:
            st.markdown('<div class="wf-toolbar-note">Apply</div>', unsafe_allow_html=True)
            apply_filters = st.form_submit_button("Apply", use_container_width=True)

    if apply_filters:
        st.session_state["workflow_runs_search_applied"] = run_search_input
        st.session_state["workflow_runs_summary_filter_applied"] = bool(require_summary_input)
        st.session_state["workflow_runs_parquet_filter_applied"] = bool(require_parquet_input)
        st.session_state["workflow_runs_page_size_applied"] = int(page_size_input)
        st.session_state["workflow_runs_page"] = 1

    run_search = str(st.session_state.get("workflow_runs_search_applied", "")).strip().lower()
    require_summary = bool(st.session_state.get("workflow_runs_summary_filter_applied", False))
    require_parquet = bool(st.session_state.get("workflow_runs_parquet_filter_applied", False))
    page_size = int(st.session_state.get("workflow_runs_page_size_applied", 10) or 10)

    filtered = runs
    if run_search:
        filtered = [row for row in filtered if run_search in str(row["name"]).lower()]
    if require_summary:
        filtered = [row for row in filtered if bool(row["has_summary"])]
    if require_parquet:
        filtered = [row for row in filtered if bool(row["has_parquet"])]

    compare_ready = [
        str(row["name"])
        for row in filtered
        if bool(row["has_summary"]) or bool(row["has_score"]) or bool(row["has_parquet"])
    ]
    if "workflow_compare_runs" not in st.session_state:
        st.session_state["workflow_compare_runs"] = compare_ready[:1]

    compare_selected = [
        name for name in st.session_state.get("workflow_compare_runs", [])
        if name in compare_ready
    ]
    st.session_state["workflow_compare_runs"] = compare_selected

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
    visible_names = {str(run["name"]) for run in visible_runs}

    pager_cols = st.columns([0.9, 1.2, 4.1])
    with pager_cols[0]:
        st.markdown('<div class="wf-toolbar-note">Page</div>', unsafe_allow_html=True)
        selected_page = st.selectbox(
            "Page",
            options=list(range(1, page_count + 1)),
            index=max(0, current_page - 1),
            key="workflow_runs_page_select",
            label_visibility="collapsed",
        )
        if selected_page != current_page:
            st.session_state[page_key] = int(selected_page)
            current_page = int(selected_page)
            start_idx = (current_page - 1) * page_size
            visible_runs = filtered[start_idx:start_idx + page_size]
            visible_names = {str(run["name"]) for run in visible_runs}
    with pager_cols[1]:
        st.markdown('<div class="wf-toolbar-note">Rows</div>', unsafe_allow_html=True)
        st.caption(str(len(visible_runs)))
    with pager_cols[2]:
        st.markdown('<div class="wf-toolbar-note">Total</div>', unsafe_allow_html=True)
        st.caption(f"{len(filtered)} runs")

    _render_local_runs_header()
    next_selected = [name for name in st.session_state.get("workflow_compare_runs", []) if name not in visible_names]
    for run in visible_runs:
        run_name = str(run["name"])
        if _render_local_run_row(run, selected=run_name in st.session_state.get("workflow_compare_runs", [])) and run_name in compare_ready:
            next_selected.append(run_name)
    st.session_state["workflow_compare_runs"] = [name for name in compare_ready if name in next_selected]

    st.markdown('<div class="wf-compare-bar">', unsafe_allow_html=True)
    st.markdown('<p class="wf-compare-title">Compare</p>', unsafe_allow_html=True)
    compare_cols = st.columns([3.4, 1.0])
    with compare_cols[0]:
        st.markdown('<div class="wf-toolbar-note">Selected runs</div>', unsafe_allow_html=True)
        selected_runs = list(st.session_state.get("workflow_compare_runs", []))
        if selected_runs:
            st.caption(" | ".join(selected_runs))
    with compare_cols[1]:
        st.markdown('<div class="wf-toolbar-note">Action</div>', unsafe_allow_html=True)
        if len(selected_runs) >= 2:
            st.link_button("Compare", _build_overview_url(selected_runs[0], selected_runs[1:]), use_container_width=True)
        elif len(selected_runs) == 1:
            st.link_button("Open", _build_overview_url(selected_runs[0]), use_container_width=True)
        else:
            st.button("Open", disabled=True, use_container_width=True, key="workflow_compare_run_disabled")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_current_tasks_section() -> None:
    section_header("Current Tasks", "")
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


def _get_start_workflow_defaults() -> Dict[str, object]:
    default_target = get_config_value("target_name", "beta/v4.3.2")
    return {
        "project_id": get_config_value("eval_project_id", "x2_dev"),
        "environment": get_config_value("environment", ""),
        "output_path_default": _make_default_output_path(default_target),
        "download_type_default": get_config_value("eval_download_type", "Archives (ZIP)"),
        "phase_default": get_config_value(
            "eval_phase",
            "perception.object_recognition.tracking.objects",
        ),
        "skip_large_file_default": True,
        "large_file_mb_default": 50.0,
        "keep_zip_files_default": False,
    }


def _render_start_workflow_form(
    catalog_presets: List[Dict[str, str]],
    catalogs_path: Optional[str],
    catalog_load_error: Optional[str],
) -> Dict[str, object]:
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
    default_output = _make_default_output_path(default_target)

    if "workflow_server_catalogs" not in st.session_state:
        st.session_state["workflow_server_catalogs"] = []
    if "workflow_server_catalog_error" not in st.session_state:
        st.session_state["workflow_server_catalog_error"] = ""
    if "workflow_selected_server_catalog_id" not in st.session_state:
        st.session_state["workflow_selected_server_catalog_id"] = ""
    if "workflow_catalog_id" not in st.session_state:
        st.session_state["workflow_catalog_id"] = ""
    if "workflow_integration_id" not in st.session_state:
        st.session_state["workflow_integration_id"] = ""
    if "workflow_catalog_resolution_error" not in st.session_state:
        st.session_state["workflow_catalog_resolution_error"] = ""
    if "workflow_last_catalog_selection" not in st.session_state:
        st.session_state["workflow_last_catalog_selection"] = ""

    server_catalogs = st.session_state.get("workflow_server_catalogs", []) or []
    server_catalog_labels = [
        f"{item['display_name']} ({item['catalog_id']})" for item in server_catalogs
    ]
    catalog_options = [""] + catalog_names + [
        label for label in server_catalog_labels if label not in catalog_names
    ]
    preset_by_label = {item["display_name"]: item for item in catalog_presets}
    server_by_label = {
        f"{item['display_name']} ({item['catalog_id']})": item for item in server_catalogs
    }

    top_cols = st.columns([1.0, 1.9, 1.2])
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
        catalog_picker_cols = st.columns([4.2, 1.1], gap="small")
        with catalog_picker_cols[0]:
            selected_catalog_name = st.selectbox(
                "Catalog",
                options=catalog_options if catalog_options else [""],
                index=catalog_options.index(st.session_state.get("workflow_catalog_name", "")) if st.session_state.get("workflow_catalog_name", "") in catalog_options else 0,
                key="workflow_catalog_name",
                label_visibility="collapsed",
                format_func=lambda value: value or "Choose a catalog",
            )
        with catalog_picker_cols[1]:
            fetch_catalogs_clicked = st.button(
                "Fetch",
                key="workflow_fetch_server_catalogs",
                use_container_width=True,
            )
            if fetch_catalogs_clicked:
                try:
                    current_environment = str(st.session_state.get("workflow_environment", default_environment) or "")
                    st.session_state["workflow_server_catalogs"] = _fetch_server_catalogs(project_id, current_environment)
                    st.session_state["workflow_server_catalog_error"] = ""
                except Exception as exc:
                    st.session_state["workflow_server_catalogs"] = []
                    st.session_state["workflow_server_catalog_error"] = str(exc)
    selected_catalog = preset_by_label.get(selected_catalog_name)
    selected_server_catalog = server_by_label.get(selected_catalog_name)
    if "workflow_last_catalog_preset" not in st.session_state:
        st.session_state["workflow_last_catalog_preset"] = ""
    if st.session_state["workflow_last_catalog_preset"] != selected_catalog_name and selected_catalog:
        st.session_state["workflow_catalog_id"] = str(selected_catalog.get("catalog_id") or "")
        st.session_state["workflow_integration_id"] = str(selected_catalog.get("integration_id") or "")
        st.session_state["workflow_selected_server_catalog_id"] = ""
        st.session_state["workflow_catalog_resolution_error"] = ""
        st.session_state["workflow_last_catalog_preset"] = selected_catalog_name
    elif selected_server_catalog:
        st.session_state["workflow_catalog_id"] = str(selected_server_catalog.get("catalog_id") or "")
        st.session_state["workflow_selected_server_catalog_id"] = str(selected_server_catalog.get("catalog_id") or "")
        current_environment = str(st.session_state.get("workflow_environment", default_environment) or "")
        if st.session_state["workflow_last_catalog_selection"] != selected_catalog_name:
            try:
                st.session_state["workflow_integration_id"] = _resolve_integration_id_for_catalog(
                    project_id,
                    current_environment,
                    st.session_state["workflow_catalog_id"],
                )
                st.session_state["workflow_catalog_resolution_error"] = ""
            except Exception as exc:
                st.session_state["workflow_integration_id"] = ""
                st.session_state["workflow_catalog_resolution_error"] = str(exc)
            st.session_state["workflow_last_catalog_selection"] = selected_catalog_name
    elif st.session_state["workflow_last_catalog_selection"] != selected_catalog_name:
        st.session_state["workflow_catalog_resolution_error"] = ""
        st.session_state["workflow_last_catalog_selection"] = selected_catalog_name
    with top_cols[2]:
        st.markdown('<div class="wf-toolbar-note">Branch or tag</div>', unsafe_allow_html=True)
        target_name = st.text_input(
            "Branch or Tag",
            value=default_target,
            key="workflow_target_name",
            label_visibility="collapsed",
            placeholder="beta/v4.3.2",
        ).strip()

    catalog_id = str(st.session_state.get("workflow_catalog_id") or "").strip()
    integration_id = str(st.session_state.get("workflow_integration_id") or "").strip()

    if st.session_state.get("workflow_server_catalog_error"):
        st.warning(f"Could not fetch catalogs: {st.session_state['workflow_server_catalog_error']}")
    catalog_id = str(st.session_state.get("workflow_catalog_id") or "").strip()

    picker_cols = st.columns([1.2, 1.2, 1.75])
    with picker_cols[0]:
        st.markdown('<div class="wf-toolbar-note">Output folder</div>', unsafe_allow_html=True)
        output_path = st.text_input(
            "Output folder",
            value=default_output,
            key="workflow_output_path",
            label_visibility="collapsed",
            placeholder=_make_default_output_path(target_name),
        ).strip()
    with picker_cols[1]:
        st.markdown('<div class="wf-toolbar-note">Environment</div>', unsafe_allow_html=True)
        environment = st.selectbox(
            "Environment",
            options=["", "dev", "stg", "prd"],
            index=["", "dev", "stg", "prd"].index(default_environment) if default_environment in ("", "dev", "stg", "prd") else 0,
            key="workflow_environment",
            label_visibility="collapsed",
            format_func=lambda value: value or "default",
        )
    with picker_cols[2]:
        st.markdown('<div class="wf-toolbar-note">Description</div>', unsafe_allow_html=True)
        description = st.text_input(
            "Description",
            value=get_config_value("workflow_description", ""),
            key="workflow_description",
            label_visibility="collapsed",
            placeholder="Optional label for the evaluator run",
        ).strip()

    confirm_cols = st.columns([1.0, 1.0])
    with confirm_cols[0]:
        if catalog_id:
            st.caption(f"Catalog ID: `{catalog_id}`")
    with confirm_cols[1]:
        if integration_id:
            st.caption(f"Integration ID: `{integration_id}`")
    if st.session_state.get("workflow_catalog_resolution_error"):
        st.warning(f"Could not resolve integration automatically: {st.session_state['workflow_catalog_resolution_error']}")

    if selected_catalog:
        desc = str(selected_catalog.get("description") or "").strip() or "Preset selected for quick scheduling."
        st.caption(f"Preset: {desc}")
    elif selected_server_catalog:
        desc = str(selected_server_catalog.get("description") or "").strip()
        if desc:
            st.caption(f"Fetched catalog: {desc}")

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

    set_config_value("eval_project_id", project_id)
    set_config_value("target_name", target_name)
    set_config_value("eval_download_type", download_type)
    set_config_value("eval_phase", phase)
    set_config_value("poll_interval", poll_interval)
    set_config_value("max_wait_hours", max_wait_hours)
    set_config_value("environment", environment)
    set_config_value("workflow_description", description)
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
        resolved_output, path_error = resolve_under_data_root(output_path, allow_create=False)
        if path_error:
            errors.append(path_error)
    else:
        errors.append("Output folder")

    return {
        "project_id": project_id,
        "environment": environment,
        "output_path_default": output_path or _make_default_output_path(target_name),
        "download_type_default": download_type,
        "phase_default": phase,
        "skip_large_file_default": True,
        "large_file_mb_default": 50.0,
        "keep_zip_files_default": False,
        "dialog_payload": {
            "errors": errors,
            "project_id": project_id,
            "catalog_id": catalog_id,
            "integration_id": integration_id,
            "catalog_preset_name": selected_catalog_name,
            "has_custom_catalog": bool(catalog_id and not selected_catalog),
            "target_name": target_name,
            "description": description,
            "resolved_output": str(resolved_output) if resolved_output else "",
            "environment": environment,
            "is_tag": is_tag,
            "download_type": download_type,
            "phase": phase,
            "poll_interval": int(poll_interval),
            "max_wait_hours": int(max_wait_hours),
            "run_eval": bool(run_eval),
            "generate_parquet": bool(generate_parquet),
            "eval_recursive": bool(eval_recursive),
        },
    }


def _render_workflow_launcher_section(
    catalog_presets: List[Dict[str, str]],
    catalogs_path: Optional[str],
    catalog_load_error: Optional[str],
) -> Dict[str, object]:
    section_header("Run Evaluator Workflow", "")
    start_defaults = _get_start_workflow_defaults()
    if "workflow_start_dialog_open" not in st.session_state:
        st.session_state["workflow_start_dialog_open"] = False
    new_job_clicked = st.button(
        "Start new workflow",
        key="workflow_open_start_dialog",
        type="primary",
        use_container_width=False,
    )

    if new_job_clicked and callable(getattr(st, "dialog", None)):
        st.session_state["workflow_start_dialog_open"] = True
        fresh_target = str(get_config_value("target_name", "beta/v4.3.2") or "beta/v4.3.2")
        st.session_state["workflow_catalog_name"] = ""
        st.session_state["workflow_last_catalog_preset"] = ""
        st.session_state["workflow_catalog_id"] = ""
        st.session_state["workflow_integration_id"] = ""
        st.session_state["workflow_server_catalogs"] = []
        st.session_state["workflow_server_catalog_error"] = ""
        st.session_state["workflow_selected_server_catalog_id"] = ""
        st.session_state["workflow_selected_server_catalog_label"] = ""
        st.session_state["workflow_catalog_resolution_error"] = ""
        st.session_state["workflow_last_catalog_selection"] = ""
        st.session_state["workflow_output_path"] = _make_default_output_path(fresh_target)

        @st.dialog("Start evaluator workflow", width="large")
        def _workflow_start_dialog() -> None:
            st.caption("This is the full launcher for creating a new evaluator job, downloading results, and optionally running eval/parquet.")
            payload = _render_start_workflow_form(catalog_presets, catalogs_path, catalog_load_error)
            submit_cols = st.columns([1.15, 1.15, 3.7])
            close_clicked = submit_cols[0].button("Close", key="workflow_close_start_dialog", use_container_width=True)
            start_clicked = submit_cols[1].button("Start workflow", key="workflow_start_btn_dialog", type="primary", use_container_width=True)
            if close_clicked:
                st.session_state["workflow_start_dialog_open"] = False
                st.rerun()
            if start_clicked:
                dialog_payload = dict(payload.get("dialog_payload") or {})
                errors = dialog_payload.get("errors", [])
                if errors:
                    for err in errors:
                        st.error(f"Missing or invalid: {err}")
                elif not is_task_queue_enabled():
                    st.error("Task queue not enabled. Set `USE_TASK_QUEUE=true` and `REDIS_URL`.")
                else:
                    task_id = _enqueue_task(
                        "run_evaluator_and_process",
                        {
                            "project_id": dialog_payload["project_id"],
                            "catalog_id": dialog_payload["catalog_id"],
                            "integration_id": dialog_payload["integration_id"],
                            "suite_ids": None,
                            "target_name": dialog_payload["target_name"],
                            "description": dialog_payload["description"] or _make_auto_workflow_description(
                                dialog_payload["target_name"],
                                dialog_payload.get("catalog_preset_name", ""),
                                has_custom_catalog=bool(dialog_payload.get("has_custom_catalog", False)),
                            ),
                            "output_path": dialog_payload["resolved_output"],
                            "environment": dialog_payload["environment"],
                            "max_retries": 0,
                            "clean_build": False,
                            "debug": False,
                            "is_tag": dialog_payload["is_tag"],
                            "download_type": "archives" if dialog_payload["download_type"] == "Archives (ZIP)" else "result_json",
                            "phase": dialog_payload["phase"],
                            "skip_large_file": False,
                            "large_file_mb": 50.0,
                            "keep_zip_files": False,
                            "poll_interval": dialog_payload["poll_interval"],
                            "max_wait_seconds": dialog_payload["max_wait_hours"] * 3600,
                            "run_eval": dialog_payload["run_eval"],
                            "generate_parquet": dialog_payload["generate_parquet"],
                            "eval_recursive": dialog_payload["eval_recursive"],
                            "eval_overwrite": False,
                        },
                    )
                    if task_id:
                        st.session_state["workflow_start_dialog_open"] = False
                        st.success(f"Workflow queued. Task id: `{task_id}`")
                        st.rerun()
                    else:
                        st.error("Failed to enqueue task. Check worker logs.")

        _workflow_start_dialog()

    return start_defaults


_inject_workflow_page_styles()
render_page_hero(
    kicker="Workflow automation",
    title="Evaluator Workflow",
    description="Browse finished runs, watch background tasks, launch fresh evaluator pipelines, and reuse existing evaluator reports from one aligned workspace.",
)

catalog_presets, catalogs_path, catalog_load_error = _load_catalog_presets()

tab_tasks, tab_local = st.tabs(["Run Tasks", "Local Runs"])

with tab_tasks:
    _render_current_tasks_section()
    start_defaults = _render_workflow_launcher_section(catalog_presets, catalogs_path, catalog_load_error)

    configure_recent_evaluator_jobs_ui(
        get_config_value=get_config_value,
        set_config_value=set_config_value,
        enqueue_task=_enqueue_task,
        catalog_io_available=CATALOG_IO_AVAILABLE,
        environment=str(start_defaults["environment"] or ""),
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

with tab_local:
    use_fragment = getattr(st, "fragment", None) is not None
    if use_fragment:
        try:

            @st.fragment
            def _local_runs_fragment():
                _render_local_runs_section()

            _local_runs_fragment()
        except (TypeError, AttributeError):
            _render_local_runs_section()
    else:
        _render_local_runs_section()
