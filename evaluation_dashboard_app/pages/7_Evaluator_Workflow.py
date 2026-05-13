"""
Evaluator Workflow Page
=======================
Complete end-to-end workflow for running evaluator jobs and processing results.
"""

import streamlit as st
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from lib.WebAPI import scenarioAPI
from lib.user_config import UserConfig

# Initialize or load user config
_user_config = UserConfig(warning_fn=st.warning)

def get_config_value(key, default=None):
    return _user_config.get(key, default)

def set_config_value(key, value):
    _user_config.set(key, value)

from lib.path_utils import get_data_root, resolve_under_data_root
from lib.page_chrome import inject_app_page_styles, render_page_hero
from lib.db import (
    create_task,
    is_task_queue_enabled,
    list_recent_tasks,
)

try:
    from lib.perception_catalog_io import pkl_archive_to_parquet
    CATALOG_IO_AVAILABLE = True
except ImportError:
    CATALOG_IO_AVAILABLE = False

# JST timezone for display
_JST = timezone(timedelta(hours=9))
_TASK_LIST_MAX_ROWS = 200


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
    import os
    import json
    _APP_ROOT = Path(__file__).parent.parent
    _CATALOGS_FILENAME = "catalogs.json"
    search_paths = [
        _APP_ROOT / _CATALOGS_FILENAME,
        Path(os.environ.get("CATALOGS_PATH", "")),
        Path.cwd() / _CATALOGS_FILENAME,
    ]
    catalogs = []
    loaded_path = None
    load_error = None
    for p in search_paths:
        if p.exists() and p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    catalogs = data.get("catalogs", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    loaded_path = str(p)
                    load_error = None
                    break
            except Exception as e:
                load_error = str(e)
    presets = []
    for c in catalogs:
        if isinstance(c, dict):
            name = c.get("display_name") or c.get("name") or c.get("catalog_id", "Unknown")
            presets.append({**c, "display_name": name})
    return presets, loaded_path, load_error


def _enqueue_task(queue_name: str, params: dict) -> Optional[str]:
    try:
        from redis import Redis
        from rq import Queue
        import os
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        redis_conn = Redis.from_url(redis_url)
        q = Queue(name=queue_name, connection=redis_conn, default_timeout="7d")
        from worker.tasks import job_run_evaluator_and_process
        job = q.enqueue(job_run_evaluator_and_process, kwargs=params, job_timeout="7d", result_ttl="7d")
        user_id = None
        try:
            from lib.auth import get_current_user_id
            user_id = get_current_user_id()
        except Exception:
            pass
        create_task(
            task_id=job.id,
            task_type="run_evaluator_and_process",
            description=f"Evaluator workflow: {params.get('description', 'no description')}",
            parameters=params,
            created_by=user_id,
        )
        return job.id
    except Exception as e:
        st.error(f"Failed to enqueue task: {e}")
        return None


def render_task_row(task):
    """Render a single task row."""
    status = task.get("status", "unknown")
    task_id = task.get("task_id", "")
    description = task.get("description", "")[:70]
    created = task.get("created_at")
    created_str = _to_jst(created).strftime("%m/%d %H:%M") if created else "N/A"

    status_config = {
        "running": {"color": "#f59e0b", "bg": "#fffbeb"},
        "finished": {"color": "#10b981", "bg": "#ecfdf5"},
        "failed": {"color": "#ef4444", "bg": "#fef2f2"},
        "queued": {"color": "#6b7280", "bg": "#f9fafb"},
    }
    cfg = status_config.get(status, status_config["queued"])

    st.markdown(
        f"""
        <div style="
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 16px;
        ">
            <div style="flex: 1;">
                <code style="font-size: 11px; background: #f1f5f9; padding: 3px 8px; border-radius: 6px; color: #475569;">{task_id[:24]}...</code>
                <div style="margin-top: 6px; font-size: 14px; color: #374151;">{description}</div>
                <div style="margin-top: 4px; font-size: 12px; color: #94a3b8;">🕐 {created_str}</div>
            </div>
            <div style="
                background: {cfg['bg']};
                color: {cfg['color']};
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
            ">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Page config
st.set_page_config(page_title="Evaluator Workflow", layout="wide", initial_sidebar_state="expanded")
inject_app_page_styles()

# Load catalog presets
CATALOG_PRESETS, CATALOGS_PATH, catalog_load_error = _load_catalog_presets()
catalog_names = [c["display_name"] for c in CATALOG_PRESETS]

# ============================================
# HERO
# ============================================
render_page_hero(
    kicker="Workflow automation",
    title="Evaluator Workflow",
    description="Schedule jobs, download results, and generate reports — all in one click",
)

# ============================================
# SIDEBAR
# ============================================
st.sidebar.markdown("### ⚙️ Configuration")

eval_project_id = st.sidebar.text_input("Project ID", value=get_config_value("eval_project_id", "x2_dev"))
set_config_value("eval_project_id", eval_project_id)

if catalog_names:
    selected_catalog_name = st.sidebar.selectbox("Catalog", options=catalog_names, index=0)
    selected_catalog = next((c for c in CATALOG_PRESETS if c["display_name"] == selected_catalog_name), None)
    if selected_catalog:
        catalog_id = selected_catalog["catalog_id"]
        integration_id = selected_catalog["integration_id"]
        
        # Display catalog info
        st.sidebar.markdown("#### 📋 Catalog Info")
        info_cols = st.sidebar.columns(2)
        with info_cols[0]:
            st.markdown(f"**ID:** `{catalog_id}`")
        with info_cols[1]:
            st.markdown(f"**Integration:** `{integration_id}`")
        if selected_catalog.get("description"):
            st.sidebar.markdown(f"📝 {selected_catalog['description']}")
        if selected_catalog.get("tags"):
            st.sidebar.markdown(f"🏷️ Tags: {', '.join(selected_catalog['tags'])}")
else:
    catalog_id = None
    integration_id = None

with st.sidebar.expander("Manual override"):
    manual_catalog_id = st.text_input("Catalog ID", value="")
    manual_integration_id = st.text_input("Integration ID", value="")
    if manual_catalog_id:
        catalog_id = manual_catalog_id
    if manual_integration_id:
        integration_id = manual_integration_id

target_name = st.sidebar.text_input("Branch or Tag", value=get_config_value("target_name", "beta/v4.3.2"))
set_config_value("target_name", target_name)

# Auto-generate output folder based on branch name and timestamp
def _make_default_output_path(branch_name):
    import re
    clean_branch = re.sub(r'[^\w]', '_', branch_name.strip('/')) if branch_name else "eval"
    clean_branch = re.sub(r'_+', '_', clean_branch).strip('_')
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{clean_branch}_{ts}"

# Always auto-generate fresh output path
eval_output_path = st.sidebar.text_input("📁 Output folder", value=_make_default_output_path(target_name), key="eval_output_path")

eval_download_type = get_config_value("eval_download_type", "Archives (ZIP)")
eval_phase = get_config_value("eval_phase", "perception.object_recognition.tracking.objects")
poll_interval = int(get_config_value("poll_interval", 60))
max_wait_hours = int(get_config_value("max_wait_hours", 24))
environment = get_config_value("environment", "")

with st.sidebar.expander("Advanced"):
    eval_download_type = st.radio("Download", ["Archives (ZIP)", "Result JSON"], index=0, horizontal=True)
    set_config_value("eval_download_type", eval_download_type)
    if eval_download_type == "Archives (ZIP)":
        eval_phase = st.text_input("Phase", value=eval_phase)
        set_config_value("eval_phase", eval_phase)
    poll_interval = st.slider("Poll interval (s)", 10, 300, poll_interval, step=10)
    set_config_value("poll_interval", poll_interval)
    max_wait_hours = st.slider("Max wait (h)", 1, 168, max_wait_hours)
    set_config_value("max_wait_hours", max_wait_hours)

# ============================================
# MAIN CONTENT
# ============================================

# Validation
validation_errors = []
if not eval_project_id:
    validation_errors.append("Project ID")
if not catalog_id:
    validation_errors.append("Catalog ID")
if not integration_id:
    validation_errors.append("Integration ID")
if not target_name:
    validation_errors.append("Target")

if validation_errors:
    for err in validation_errors:
        st.error(f"❌ {err}")
    st.stop()

resolved_output, path_err = resolve_under_data_root(eval_output_path, allow_create=True)
if path_err:
    st.error(f"❌ {path_err}")
    st.stop()
resolved_path_str = str(resolved_output)
max_wait_seconds = max_wait_hours * 3600

# Pipeline visualization
st.markdown("""
<style>
.pipeline {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.pipeline-step {
    background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
    border: 1px solid #99f6e4;
    border-radius: 10px;
    padding: 12px 18px;
    text-align: center;
    flex: 1;
    max-width: 150px;
    min-width: 100px;
}
.pipeline-step .num {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #0d9488;
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 6px;
}
.pipeline-step .title {
    font-size: 12px;
    font-weight: 600;
    color: #0f766e;
}
.pipeline-arrow {
    display: flex;
    align-items: center;
    color: #99f6e4;
    font-size: 20px;
}
</style>
<div class="pipeline">
    <div class="pipeline-step"><div class="num">1</div><div class="title">📤 Schedule</div></div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step"><div class="num">2</div><div class="title">⏳ Wait</div></div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step"><div class="num">3</div><div class="title">📥 Download</div></div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step"><div class="num">4</div><div class="title">📊 Evaluate</div></div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step"><div class="num">5</div><div class="title">📦 Parquet</div></div>
</div>
""", unsafe_allow_html=True)

# Options
col1, col2, col3 = st.columns(3)
with col1:
    eval_run_eval = st.checkbox("📊 Run Evaluation", value=True)
with col2:
    eval_generate_parquet = st.checkbox("📦 Generate Parquet", value=CATALOG_IO_AVAILABLE, disabled=not CATALOG_IO_AVAILABLE)
with col3:
    eval_recursive = st.checkbox("🔍 Recursive Scan", value=True)

# START BUTTON
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] button[kind="primary"] {
    height: 60px !important;
    font-size: 18px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)
clicked = st.button("🚀 Start Evaluator Workflow", type="primary", use_container_width=True)

if clicked:
    if not is_task_queue_enabled():
        st.error("❌ Task queue not enabled. Set `USE_TASK_QUEUE=true` and `REDIS_URL`.")
        st.stop()

    task_id = _enqueue_task("run_evaluator_and_process", {
        "project_id": eval_project_id,
        "catalog_id": catalog_id,
        "integration_id": integration_id,
        "suite_ids": None,
        "target_name": target_name,
        "description": f"Eval {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "output_path": resolved_path_str,
        "environment": environment,
        "max_retries": 0,
        "clean_build": False,
        "debug": False,
        "is_tag": False,
        "download_type": "archives" if eval_download_type == "Archives (ZIP)" else "result_json",
        "phase": eval_phase,
        "skip_large_file": False,
        "large_file_mb": 50.0,
        "keep_zip_files": False,
        "poll_interval": poll_interval,
        "max_wait_seconds": max_wait_seconds,
        "run_eval": eval_run_eval,
        "generate_parquet": eval_generate_parquet,
        "eval_recursive": eval_recursive,
        "eval_overwrite": False,
    })

    if task_id:
        st.success(f"✅ Workflow queued! Task: `{task_id[:24]}...`")
        st.info("💡 Running in background — close browser, check Task Status below.")
    else:
        st.error("❌ Failed to enqueue task. Check worker logs.")

# ============================================
# TASK STATUS
# ============================================
st.markdown('<div class="section-title">📋 Recent Tasks</div>', unsafe_allow_html=True)

if not is_task_queue_enabled():
    st.info("Task queue not enabled. Set `USE_TASK_QUEUE=true` to track background tasks.")
else:
    tasks = list_recent_tasks(limit=_TASK_LIST_MAX_ROWS)
    running = len([t for t in tasks if t.get("status") == "running"])
    finished = len([t for t in tasks if t.get("status") == "finished"])
    failed = len([t for t in tasks if t.get("status") == "failed"])

    # Metrics row
    m1, m2, m3, m4 = st.columns([1, 1, 1, 2])
    m1.metric("⏳ Running", running)
    m2.metric("✅ Finished", finished)
    m3.metric("❌ Failed", failed)
    filter_status = m4.selectbox("Filter", ["All", "Running", "Finished", "Failed", "Queued"], index=0, label_visibility="collapsed")

    filtered = tasks if filter_status == "All" else [t for t in tasks if t.get("status") == filter_status.lower()]

    # Task list
    for task in filtered[:10]:
        status = task.get("status", "unknown")
        task_id_str = task.get("task_id", "")[:24]
        desc = task.get("description", "No description")[:60]
        created = task.get("created_at")
        created_str = _to_jst(created).strftime("%m/%d %H:%M") if created else "N/A"

        st.markdown(f"""
        <div class="task-card">
            <div class="info">
                <span class="task-id">{task_id_str}...</span>
                <div class="desc">{desc}</div>
                <div class="time">🕐 {created_str}</div>
            </div>
            <div class="status status-{status}">{status}</div>
        </div>
        """, unsafe_allow_html=True)

        if task.get("error"):
            with st.expander("❌ Error details"):
                st.code(task["error"])

    if len(filtered) > 10:
        st.caption(f"Showing 10 of {len(filtered)} tasks")

st.sidebar.divider()
st.sidebar.caption("💡 Runs async — close browser safely")
