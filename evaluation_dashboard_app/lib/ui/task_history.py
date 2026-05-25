"""Shared task history/list rendering used across pages."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

from lib.auth import get_current_user_id, is_auth_enabled
from lib.db import delete_task, get_task
from lib.ui.download_ui import TaskCardMode, render_task_list_empty_state, task_list_card_markup
from lib.ui.task_result_summary import render_task_result_summary

_JST = timezone(timedelta(hours=9))


def _to_jst(dt: Any) -> Optional[datetime]:
    """Convert datetime to JST for display. Naive datetimes are assumed UTC."""
    if dt is None:
        return None
    if not hasattr(dt, "astimezone"):
        return None
    try:
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_JST)
    except Exception:
        return None


def _task_type_label(task_type: str) -> str:
    labels = {
        "download_results": "Download results",
        "download_scenarios": "Download scenarios",
        "run_eval_dirs": "Run eval dirs",
        "generate_summary_csv": "Generate summary CSV",
        "build_parquet": "Build parquet",
        "download_and_eval": "Download + Eval",
        "run_evaluator_and_process": "Run Evaluator + Process",
        "run_release_specsheet_workflow": "Release Specsheet",
    }
    return labels.get(task_type, task_type or "Task")


def _task_summary(t: Dict[str, Any]) -> str:
    params = t.get("parameters") or {}
    task_type = t.get("type", "")
    if task_type == "download_results":
        out = params.get("output_path") or params.get("job_id") or ""
        return f"job_id={params.get('job_id', '')} → {out}"
    if task_type == "download_scenarios":
        out = params.get("output_dir") or params.get("output_path") or ""
        return f"job_id={params.get('job_id', '')} → {out}"
    if task_type in ("run_eval_dirs", "generate_summary_csv"):
        return params.get("eval_root", "")
    if task_type == "build_parquet":
        return params.get("pkl_dir", "")
    if task_type == "download_and_eval":
        out = params.get("output_path") or params.get("job_id") or ""
        parts = ["download"]
        if params.get("run_eval"):
            parts.append("eval")
        if params.get("generate_parquet"):
            parts.append("parquet")
        return f"job_id={params.get('job_id', '')} [{'+'.join(parts)}] → {out}"
    if task_type == "run_evaluator_and_process":
        target = params.get("target_name", "")
        target_type = "tag" if params.get("is_tag", False) else "branch"
        return f"{target_type}={target} → {params.get('output_path', '')}"
    if task_type == "run_release_specsheet_workflow":
        target = params.get("target_name", "")
        target_type = "tag" if params.get("is_tag", False) else "branch"
        return f"{target_type}={target} → {params.get('output_path', '')}"
    return ""


def _task_time_str(t: Dict[str, Any]) -> str:
    created = t.get("created_at")
    dt = _to_jst(created) if created else None
    if not dt:
        return "—"
    try:
        return dt.strftime("%b %d, %H:%M")
    except Exception:
        return str(created)[:16] if created else "—"


def _task_duration(t: Dict[str, Any]) -> Optional[str]:
    created = t.get("created_at")
    updated = t.get("updated_at")
    if not created or not updated:
        return None
    try:
        start = created.timestamp() if hasattr(created, "timestamp") else None
        end = updated.timestamp() if hasattr(updated, "timestamp") else None
        if start is None or end is None:
            return None
        secs = int(end - start)
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m {secs % 60}s"
        return f"{secs // 3600}h {(secs % 3600) // 60}m"
    except Exception:
        return None


def render_task_detail_content(t: Dict[str, Any]) -> None:
    """Render full task detail content."""
    try:
        _render_task_detail_content_impl(t)
    except Exception as e:
        st.error(f"Could not load task details: {e}")
        import traceback
        st.code(traceback.format_exc(), language=None)


def _render_task_detail_content_impl(t: Dict[str, Any]) -> None:
    status = t.get("status", "")
    created_jst = _to_jst(t.get("created_at"))
    updated_jst = _to_jst(t.get("updated_at"))
    time_parts = []
    if created_jst:
        try:
            time_parts.append(f"Created: {created_jst.strftime('%Y-%m-%d %H:%M:%S')} JST")
        except Exception:
            time_parts.append(f"Created: {t.get('created_at')}")
    if updated_jst and updated_jst != created_jst:
        try:
            time_parts.append(f"Updated: {updated_jst.strftime('%Y-%m-%d %H:%M:%S')} JST")
        except Exception:
            time_parts.append(f"Updated: {t.get('updated_at')}")
    if time_parts:
        st.caption(" · ".join(time_parts))

    result_summary_raw = t.get("result_summary")
    if result_summary_raw:
        try:
            result_summary = json.loads(result_summary_raw) if isinstance(result_summary_raw, str) else result_summary_raw
            render_task_result_summary(result_summary)
            st.markdown("---")
        except (TypeError, ValueError):
            pass
    if t.get("result_path"):
        st.text_input(
            "Result path",
            value=t["result_path"],
            key=f"rp_modal_{str(t.get('id'))}",
            disabled=True,
            label_visibility="collapsed",
        )
    if status == "failed" and t.get("error_message"):
        st.error(t.get("error_message"))
    progress_message = (t.get("progress_message") or "").strip()
    if progress_message:
        st.info(progress_message)
    log_output = (t.get("log_output") or "").strip()
    if log_output:
        st.caption("Log output")
        st.code(log_output, language=None)
    params = t.get("parameters") or {}
    if params:
        st.caption("Parameters")
        st.json(params)


def _open_task_detail(task_id: str) -> None:
    st.session_state["_task_detail_id"] = str(task_id)


def _render_one_task_row(
    t: Dict[str, Any],
    current_user: Optional[str],
    use_dialog: bool,
    *,
    mode: TaskCardMode,
) -> None:
    task_id = t.get("id", "")
    status = t.get("status", "")
    status_labels = {"pending": "Pending", "running": "Running", "completed": "Completed", "failed": "Failed"}
    status_label = status_labels.get(status, status)
    summary = _task_summary(t)
    sid = str(task_id)
    summary_short = (
        (summary[:72] + "…") if mode == "history" and summary and len(summary) > 72 else (summary if mode == "history" else "—")
    ) or "—"
    progress_msg = (t.get("progress_message") or "").strip()
    card = task_list_card_markup(
        task_id=sid,
        type_label=_task_type_label(t.get("type", "")),
        status=status,
        status_label=status_label,
        time_str=_task_time_str(t),
        duration=_task_duration(t) or "—",
        summary_short=summary_short,
        progress_pct=t.get("progress_pct"),
        progress_message=progress_msg,
        mode=mode,
    )
    st.markdown(f'<div class="dl-task-stack">{card}</div>', unsafe_allow_html=True)

    if use_dialog:
        bv, bd, _sp = st.columns([1.15, 1.15, 4])
        with bv:
            st.button("View", key=f"view_{sid}", on_click=_open_task_detail, args=(sid,))
        with bd:
            stop_lbl = "Stop" if status in ("pending", "running") else "Remove"
            stop_help = (
                "Cancels the Redis/RQ job when possible, then removes this row from the list."
                if status in ("pending", "running")
                else "Remove this row from the task list."
            )
            if st.button(stop_lbl, key=f"del_{sid}", type="secondary", help=stop_help):
                delete_task(sid, session_id=current_user)
                st.rerun()
    else:
        bd, _sp = st.columns([1.15, 4])
        with bd:
            stop_lbl = "Stop" if status in ("pending", "running") else "Remove"
            stop_help = (
                "Cancels the Redis/RQ job when possible, then removes this row from the list."
                if status in ("pending", "running")
                else "Remove this row from the task list."
            )
            if st.button(stop_lbl, key=f"del_{sid}", type="secondary", help=stop_help):
                delete_task(sid, session_id=current_user)
                st.rerun()

    if not use_dialog:
        with st.expander("More", expanded=False):
            render_task_detail_content(t)


def render_task_list(tasks: List[Dict[str, Any]], current_user: Optional[str]) -> bool:
    """Render the shared active/history task list. Returns True if any active tasks exist."""
    if current_user:
        st.caption(f"Logged in as **{current_user}** · your recent tasks only")
    if not tasks:
        render_task_list_empty_state()
        return False

    active = [t for t in tasks if t.get("status") in ("pending", "running")]
    history = [t for t in tasks if t.get("status") not in ("pending", "running")]
    use_dialog = callable(getattr(st, "dialog", None))

    for t in active:
        _render_one_task_row(t, current_user, use_dialog, mode="active_compact")

    if history:
        with st.expander(f"Task history ({len(history)})", expanded=False):
            for t in history:
                _render_one_task_row(t, current_user, use_dialog, mode="history")

    if use_dialog and st.session_state.get("_task_detail_id"):
        task_id = st.session_state["_task_detail_id"]
        try:
            detail_task = next((x for x in tasks if str(x.get("id")) == task_id), None)
            if detail_task is None:
                detail_task = get_task(task_id)
            if detail_task:

                @st.dialog("Task details", width="large")
                def _task_detail_modal():
                    render_task_detail_content(detail_task)
                    if st.button("Close"):
                        st.session_state.pop("_task_detail_id", None)
                        st.rerun()

                _task_detail_modal()
        except Exception as e:
            st.error(f"Could not open task details: {e}")
        finally:
            st.session_state.pop("_task_detail_id", None)

    return len(active) > 0


def get_task_list_current_user() -> Optional[str]:
    """Return current user id when auth is enabled, else None."""
    return get_current_user_id() if is_auth_enabled() else None
