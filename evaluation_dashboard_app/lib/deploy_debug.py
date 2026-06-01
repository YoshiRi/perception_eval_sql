"""
Deployment diagnostics: redacted env display, Postgres/Redis/RQ checks, optional Docker API (docker-py).
Docker access is gated by EVAL_DEPLOYMENT_DEBUG_DOCKER and a mounted /var/run/docker.sock.
"""

from __future__ import annotations

import concurrent.futures
import os
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

from lib.db import get_connection, get_database_url, is_task_queue_enabled

DOCKER_SOCKET_DEFAULT = "/var/run/docker.sock"
MAX_LOG_TAIL_LINES = 2000


def running_in_docker() -> bool:
    """True when /.dockerenv exists (typical Linux Docker container)."""
    try:
        return Path("/.dockerenv").is_file()
    except OSError:
        return False
EXEC_OUTPUT_MAX_BYTES = 256 * 1024
EXEC_TIMEOUT_SEC = 120.0


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def redact_database_url(url: Optional[str]) -> str:
    if not url:
        return "(not set)"
    try:
        p = urlparse(url)
        netloc = p.hostname or ""
        if p.port:
            netloc = f"{netloc}:{p.port}"
        if p.username:
            user = p.username
            if p.password:
                user = f"{user}:***"
            netloc = f"{user}@{netloc}" if netloc else user
        path = p.path or ""
        q = f"?{p.query}" if p.query else ""
        return urlunparse((p.scheme, netloc, path, "", q, ""))
    except Exception:
        return "(invalid URL)"


def redact_redis_url(url: Optional[str]) -> str:
    if not url:
        return "(not set)"
    try:
        p = urlparse(url)
        if p.password:
            netloc = p.hostname or ""
            if p.port:
                netloc = f"{netloc}:{p.port}"
            user = p.username or ""
            if user:
                netloc = f"{user}:***@{netloc}"
            else:
                netloc = f"***@{netloc}"
            path = p.path or ""
            return urlunparse((p.scheme, netloc, path, "", "", ""))
        return url
    except Exception:
        return "(invalid URL)"


def postgres_check() -> Tuple[bool, str]:
    """Return (ok, message)."""
    url = get_database_url()
    if not url:
        return False, "DATABASE_URL is not set"
    try:
        import psycopg2
    except ImportError:
        return False, "psycopg2 not installed"
    with get_connection() as conn:
        if conn is None:
            return False, "Could not open Postgres connection"
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True, "SELECT 1 OK"
        except Exception as e:
            return False, str(e)


def redis_ping_check() -> Tuple[bool, str]:
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        from redis import Redis
    except ImportError:
        return False, "redis package not installed"
    try:
        conn = Redis.from_url(url, socket_connect_timeout=3)
        if conn.ping():
            return True, "PING OK"
        return False, "PING returned false"
    except Exception as e:
        return False, str(e)


def rq_overview() -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Return (ok, message, details).
    details keys: queue_name, queued_jobs, started_jobs, failed_jobs (integers).
    """
    try:
        from redis import Redis
        from rq import Queue
    except ImportError:
        return False, "rq/redis not installed", None
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    queue_name = os.environ.get("RQ_QUEUE", "default")
    try:
        conn = Redis.from_url(redis_url, socket_connect_timeout=3)
        conn.ping()
        q = Queue(queue_name, connection=conn)
        failed_reg = q.failed_job_registry
        started_reg = q.started_job_registry
        details = {
            "queue_name": queue_name,
            "queued_jobs": len(q),
            "started_jobs": len(started_reg),
            "failed_jobs": len(failed_reg),
        }
        return True, "RQ queue readable", details
    except Exception as e:
        return False, str(e), None


def task_counts_by_status() -> Tuple[bool, str, Optional[Dict[str, int]]]:
    """When task queue is enabled, return counts per tasks.status."""
    if not is_task_queue_enabled():
        return True, "Task queue disabled (skipped)", None
    with get_connection() as conn:
        if conn is None:
            return False, "No database connection", None
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status"
                )
                rows = cur.fetchall()
            counts = {str(r[0]): int(r[1]) for r in rows}
            return True, "OK", counts
        except Exception as e:
            return False, str(e), None


def database_table_overview() -> Tuple[bool, str, Optional[List[Dict[str, Any]]]]:
    """Return public table names with approximate row counts for DB debugging."""
    if not get_database_url():
        return False, "DATABASE_URL is not set", None
    with get_connection() as conn:
        if conn is None:
            return False, "No database connection", None
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        t.table_name,
                        COALESCE(c.reltuples::bigint, 0) AS estimated_rows,
                        CASE WHEN c.oid IS NULL THEN 0 ELSE pg_total_relation_size(c.oid) END AS total_bytes
                    FROM information_schema.tables t
                    LEFT JOIN pg_namespace n ON n.nspname = t.table_schema
                    LEFT JOIN pg_class c ON c.relname = t.table_name AND c.relnamespace = n.oid
                    WHERE t.table_schema = 'public'
                      AND t.table_type = 'BASE TABLE'
                    ORDER BY t.table_name
                    """
                )
                rows = [
                    {
                        "table_name": str(r[0]),
                        "estimated_rows": int(r[1] or 0),
                        "total_bytes": int(r[2] or 0),
                    }
                    for r in cur.fetchall()
                ]
            return True, "OK", rows
        except Exception as e:
            return False, str(e), None


def database_recent_task_rows(
    *,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    task_type: Optional[str] = None,
    search: Optional[str] = None,
) -> Tuple[bool, str, List[Dict[str, Any]], int]:
    """Read recent rows from the task table for the deployment debug DB tab."""
    if not get_database_url():
        return False, "DATABASE_URL is not set", [], 0
    with get_connection() as conn:
        if conn is None:
            return False, "No database connection", [], 0
        try:
            from psycopg2.extras import RealDictCursor
        except ImportError:
            return False, "psycopg2 not installed", [], 0

        where_parts: List[str] = []
        params: List[Any] = []
        if status:
            where_parts.append("status = %s")
            params.append(status)
        if task_type:
            where_parts.append("type = %s")
            params.append(task_type)
        if search:
            needle = f"%{search.strip()}%"
            where_parts.append(
                """
                (
                    id::text ILIKE %s OR type ILIKE %s OR status ILIKE %s OR
                    COALESCE(session_id, '') ILIKE %s OR COALESCE(rq_job_id, '') ILIKE %s OR
                    COALESCE(result_path, '') ILIKE %s OR COALESCE(error_message, '') ILIKE %s OR
                    COALESCE(parameters::text, '') ILIKE %s OR COALESCE(result_summary, '') ILIKE %s
                )
                """
            )
            params.extend([needle] * 9)

        where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        capped_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT COUNT(*) FROM tasks{where_sql}", params)
                total_row = cur.fetchone()
                total = int(total_row["count"] if total_row else 0)
                cur.execute(
                    f"""
                    SELECT
                        id, type, status, session_id, rq_job_id,
                        created_at, updated_at, progress_pct, progress_message,
                        result_path, error_message, parameters, result_summary, log_output
                    FROM tasks{where_sql}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    [*params, capped_limit, safe_offset],
                )
                rows = [dict(row) for row in cur.fetchall()]
            return True, "OK", rows, total
        except Exception as e:
            return False, str(e), [], 0


def docker_unix_socket_for_check() -> Optional[str]:
    """Path to Unix socket for existence check, or None if DOCKER_HOST is non-Unix (e.g. tcp)."""
    host = os.environ.get("DOCKER_HOST", "").strip()
    if not host:
        return DOCKER_SOCKET_DEFAULT
    if host.startswith("unix://"):
        path = host[len("unix://") :]
        return path or DOCKER_SOCKET_DEFAULT
    return None


def is_docker_debug_enabled() -> bool:
    if not _env_truthy("EVAL_DEPLOYMENT_DEBUG_DOCKER"):
        return False
    sock = docker_unix_socket_for_check()
    if sock is None:
        return True
    if not sock.startswith("/") or ".." in sock:
        return False
    try:
        st_mode = os.stat(sock).st_mode
        return stat.S_ISSOCK(st_mode)
    except OSError:
        return False


def docker_client_or_none():
    """Return docker.DockerClient or None if disabled/unavailable."""
    if not is_docker_debug_enabled():
        return None
    try:
        import docker
    except ImportError:
        return None
    try:
        return docker.from_env()
    except Exception:
        return None


def compose_project_filter() -> Optional[str]:
    v = os.environ.get("EVAL_DEPLOYMENT_DEBUG_COMPOSE_PROJECT", "").strip()
    return v or None


def list_containers_for_debug(client) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    Return (rows for display, warning_message).
    warning_message set when no compose project filter is configured.
    """
    project = compose_project_filter()
    warn = None
    if not project:
        warn = (
            "EVAL_DEPLOYMENT_DEBUG_COMPOSE_PROJECT is not set — listing **all** containers "
            "visible to this Docker daemon. Set it to your compose project name "
            "(see `docker compose ls`) to limit the list."
        )
    try:
        kwargs: Dict[str, Any] = {"all": True}
        if project:
            kwargs["filters"] = {"label": [f"com.docker.compose.project={project}"]}
        containers = client.containers.list(**kwargs)
        rows: List[Dict[str, str]] = []
        for c in containers:
            cid = c.id or ""
            attrs = getattr(c, "attrs", None) or {}
            state = attrs.get("State") or {}
            state_status = (state.get("Status") or getattr(c, "status", "") or "").strip()
            health_obj = state.get("Health") or {}
            health_s = (health_obj.get("Status") or "").strip()
            labels = (attrs.get("Config") or {}).get("Labels") or {}
            if not isinstance(labels, dict):
                labels = {}
            compose_service = (labels.get("com.docker.compose.service") or "").strip()
            compose_project = (labels.get("com.docker.compose.project") or "").strip()
            rows.append(
                {
                    "id": cid[:12] if len(cid) >= 12 else cid,
                    "full_id": cid,
                    "name": (c.name or "").lstrip("/"),
                    "state": state_status or "unknown",
                    "health": health_s if health_s else "—",
                    "compose_service": compose_service or "—",
                    "compose_project": compose_project or "—",
                    "image": c.image.tags[0] if c.image and c.image.tags else (c.image.id[:12] if c.image else ""),
                }
            )
        rows.sort(key=lambda r: r["name"])
        return rows, warn
    except Exception as e:
        return [], f"Docker list failed: {e}"


def is_exec_enabled() -> bool:
    """Allow one-shot shell exec in containers (separate gate from Docker list/logs)."""
    return _env_truthy("EVAL_DEPLOYMENT_DEBUG_EXEC")


def container_exec_command(
    client,
    full_container_id: str,
    command: str,
    *,
    timeout_sec: float = EXEC_TIMEOUT_SEC,
    max_output_bytes: int = EXEC_OUTPUT_MAX_BYTES,
) -> Tuple[int, str]:
    """
    Run `sh -c <command>` inside the container. Returns (exit_code, combined stdout/stderr text).
    exit_code -1 means client/timeout error.
    """
    command = (command or "").strip()
    if not command:
        return 127, "(empty command)"
    try:
        c = client.containers.get(full_container_id)
    except Exception as e:
        return -1, f"(container error: {e})"

    cmd = ["sh", "-c", command]

    def _run():
        return c.exec_run(cmd, demux=True, tty=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run)
        try:
            exit_code, demux_out = fut.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            return -1, f"(timeout after {int(timeout_sec)}s — command still running in container)"

    stdout_b, stderr_b = b"", b""
    if demux_out is not None:
        if isinstance(demux_out, (tuple, list)) and len(demux_out) >= 2:
            stdout_b, stderr_b = demux_out[0] or b"", demux_out[1] or b""
        elif isinstance(demux_out, bytes):
            stdout_b = demux_out
    stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
    stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append("--- stderr ---\n" + stderr)
    combined = "\n".join(parts) if parts else "(no output)"
    raw = combined.encode("utf-8")
    if len(raw) > max_output_bytes:
        combined = raw[:max_output_bytes].decode("utf-8", errors="replace") + "\n... (output truncated)"
    try:
        ec = int(exit_code) if exit_code is not None else -1
    except (TypeError, ValueError):
        ec = -1
    return ec, combined


def container_logs_tail(client, full_container_id: str, tail_lines: int) -> str:
    """Fetch last N lines of logs; cap tail_lines internally."""
    n = max(1, min(int(tail_lines), MAX_LOG_TAIL_LINES))
    try:
        c = client.containers.get(full_container_id)
        raw = c.logs(tail=n, timestamps=True)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    except Exception as e:
        return f"(error fetching logs: {e})"


def redacted_deployment_env_rows() -> List[Tuple[str, str]]:
    """Key-value rows for Streamlit table (values redacted where needed)."""
    keys = [
        "EVAL_DASHBOARD_DATA_ROOT",
        "EVAL_DASHBOARD_CONFIG",
        "USE_TASK_QUEUE",
        "DATABASE_URL",
        "REDIS_URL",
        "RQ_QUEUE",
        "TZ",
        "AUTH_USER_HEADER",
        "AUTH_DEFAULT_USER",
        "EVAL_DEPLOYMENT_DEBUG_DOCKER",
        "EVAL_DEPLOYMENT_DEBUG_COMPOSE_PROJECT",
        "EVAL_DEPLOYMENT_DEBUG_EXEC",
    ]
    rows: List[Tuple[str, str]] = []
    for k in keys:
        raw = os.environ.get(k, "")
        if k == "DATABASE_URL":
            rows.append((k, redact_database_url(raw or None)))
        elif k == "REDIS_URL":
            rows.append((k, redact_redis_url(raw or None)))
        else:
            rows.append((k, raw if raw else "(not set)"))
    return rows
