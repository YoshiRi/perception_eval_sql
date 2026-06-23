"""
RQ helpers: cancel jobs, reconcile Postgres task rows with Redis job state.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _redis_connection():
    from redis import Redis

    return Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


def try_cancel_rq_job(rq_job_id: Optional[str]) -> bool:
    """Cancel an RQ job (queued or running). Idempotent if already finished. False if no id or Redis error."""
    if not rq_job_id:
        return False
    try:
        from rq.job import Job

        conn = _redis_connection()
        job = Job.fetch(str(rq_job_id), connection=conn)
        if job.is_finished or job.is_failed:
            return True
        try:
            js = job.get_status()
            js_name = getattr(js, "name", str(js))
        except Exception:
            js_name = ""
        if getattr(job, "is_started", False) or js_name == "STARTED":
            try:
                from rq.command import send_stop_job_command

                send_stop_job_command(conn, str(rq_job_id))
                return True
            except Exception:
                pass
        job.cancel()
        return True
    except Exception:
        return False


def _task_age_seconds(created_at: Any) -> float:
    if created_at is None:
        return 0.0
    if isinstance(created_at, datetime):
        dt = created_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    return 0.0


def reconcile_task_row_in_place(task: Dict[str, Any]) -> bool:
    """If Postgres says pending/running but RQ says otherwise, fix the DB row.

    Mutates ``task`` so the UI matches without a second query. Returns True if updated.
    """
    from lib.db import append_task_log, update_task_status

    tid = str(task.get("id"))
    st = task.get("status")
    if st not in ("pending", "running"):
        return False

    rqid = task.get("rq_job_id")

    if not rqid:
        if st == "running":
            msg = "No queue job id on this row; cannot verify worker. Marked failed (reconcile)."
            update_task_status(tid, "failed", error_message=msg)
            append_task_log(tid, msg)
            task["status"] = "failed"
            task["error_message"] = msg
            return True
        if st == "pending" and _task_age_seconds(task.get("created_at")) > 900:
            msg = "Stayed pending with no RQ job id (enqueue may have failed). Reconciled."
            update_task_status(tid, "failed", error_message=msg)
            append_task_log(tid, msg)
            task["status"] = "failed"
            task["error_message"] = msg
            return True
        return False

    try:
        from rq.job import Job

        job = Job.fetch(str(rqid), connection=_redis_connection())
    except Exception as ex:
        if type(ex).__name__ != "NoSuchJobError":
            return False
        msg = "RQ job not found in Redis (TTL expired or Redis cleared). Reconciled as failed."
        update_task_status(tid, "failed", error_message=msg)
        append_task_log(tid, msg)
        task["status"] = "failed"
        task["error_message"] = msg
        return True

    try:
        js = job.get_status()
        js_name = getattr(js, "name", str(js))
    except Exception:
        return False

    if job.is_failed:
        raw = job.exc_info or getattr(job, "exc_string", None) or "RQ job failed"
        err = raw if isinstance(raw, str) else str(raw)
        err = err[:8000]
        update_task_status(tid, "failed", error_message=err)
        task["status"] = "failed"
        task["error_message"] = err
        return True

    # RQ is executing the job but Postgres may still say pending (poll before run_job updates DB).
    if st == "pending" and (
        getattr(job, "is_started", False) or js_name == "STARTED"
    ):
        update_task_status(tid, "running")
        task["status"] = "running"
        return True

    if js_name in ("STOPPED", "CANCELED", "CANCELLED"):
        msg = "Job stopped in the queue (cancelled or worker shutdown). Reconciled."
        update_task_status(tid, "failed", error_message=msg)
        append_task_log(tid, msg)
        task["status"] = "failed"
        task["error_message"] = msg
        return True

    if job.is_finished:
        if st in ("pending", "running"):
            update_task_status(tid, "completed", clear_error_message=True)
            append_task_log(tid, "Reconciled: RQ finished; status set to completed.")
            task["status"] = "completed"
            task["error_message"] = None
            return True

    return False
