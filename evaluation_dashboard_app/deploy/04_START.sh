#!/usr/bin/env bash
# 04 — Start or update the full stack with docker compose up -d.
# Default: 2 worker replicas (EVAL_COMPOSE_SCALE_WORKER in .env). Override: ./04_START.sh --scale worker=1 (last --scale wins).
# Safety: confirms before restart while queued/running tasks exist. Override with --force.
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
if [[ ! -f .env ]]; then
  echo "Error: .env not found. Run 01_SETUP_ENV.sh first, then edit .env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a
WORKER_SCALE="${EVAL_COMPOSE_SCALE_WORKER:-2}"
FORCE_START=0
COMPOSE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force|--skip-running-task-check)
      FORCE_START=1
      shift
      ;;
    *)
      COMPOSE_ARGS+=("$1")
      shift
      ;;
  esac
done

dc() { docker compose --env-file .env "$@"; }

check_active_tasks() {
  if [[ "$FORCE_START" == "1" ]]; then
    return 0
  fi

  if ! dc ps --services --status running 2>/dev/null | grep -qx "postgres"; then
    return 0
  fi

  local db_user="${POSTGRES_USER:-eval_user}"
  local db_name="${POSTGRES_DB:-eval_dashboard}"
  local active_count
  local active_preview

  if ! active_count="$(
    dc exec -T postgres psql -U "$db_user" -d "$db_name" -Atqc \
      "SELECT COUNT(*) FROM tasks WHERE status IN ('pending', 'running');" 2>/dev/null
  )"; then
    echo "Warning: could not check active tasks in Postgres; continuing with start." >&2
    return 0
  fi

  active_count="$(echo "$active_count" | tr -d '[:space:]')"
  if [[ ! "$active_count" =~ ^[0-9]+$ || "$active_count" -eq 0 ]]; then
    return 0
  fi

  active_preview="$(
    dc exec -T postgres psql -U "$db_user" -d "$db_name" -Atqc \
      "SELECT id || ' | ' || type || ' | ' || status || ' | updated=' || updated_at FROM tasks WHERE status IN ('pending', 'running') ORDER BY updated_at DESC LIMIT 10;" 2>/dev/null || true
  )"

  echo "Active background tasks found (${active_count}); restarting may interrupt worker containers." >&2
  if [[ -n "$active_preview" ]]; then
    echo "Recent active tasks:" >&2
    echo "$active_preview" >&2
  fi
  if [[ ! -t 0 ]]; then
    echo "Non-interactive shell detected. Wait for the tasks to finish, or rerun with --force to start anyway." >&2
    exit 1
  fi

  local answer
  read -r -p "Continue and restart anyway? [y/N] " answer
  case "$answer" in
    [yY]|[yY][eE][sS])
      echo "Continuing despite active tasks." >&2
      ;;
    *)
      echo "Cancelled. Wait for the tasks to finish, or rerun with --force to start anyway." >&2
      exit 1
      ;;
  esac
}

check_active_tasks

dc up -d --scale "worker=${WORKER_SCALE}" "${COMPOSE_ARGS[@]}"

# Nginx resolves Docker service names at startup. Recreate it after Streamlit is
# up so it remounts the current nginx.conf and cannot keep a stale container IP.
dc up -d --no-deps --force-recreate nginx
