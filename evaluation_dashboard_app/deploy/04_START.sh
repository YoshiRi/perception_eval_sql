#!/usr/bin/env bash
# 04 — Start or update the full stack with docker compose up -d.
# Replica counts come from .env: EVAL_COMPOSE_SCALE_WORKER (default 2) and
# EVAL_COMPOSE_STREAMLIT_REPLICAS (default 1, max 3). Override workers per run:
# ./04_START.sh --scale worker=1 (last --scale wins). Streamlit replicas: edit .env.
# Every service is recreated when its config or .env changed, and replicas above the
# configured count are removed, so no container can linger with a stale environment.
# Safety: confirms before restart while queued/running tasks exist. Override with --force.
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
# shellcheck source=_compose_lib.sh
source "$DEPLOY_DIR/_compose_lib.sh"
require_env_file
load_deploy_env
setup_compose_env

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

echo "Starting: ${STREAMLIT_REPLICAS} Streamlit replica(s), ${WORKER_SCALE} worker(s)." >&2

dc up -d --scale "worker=${WORKER_SCALE}" "${COMPOSE_ARGS[@]}"

# Drop replicas left over from a higher EVAL_COMPOSE_STREAMLIT_REPLICAS: they are outside
# the enabled profiles, so compose would neither update nor stop them.
prune_extra_streamlit

# Nginx renders its config from nginx/nginx.conf.template and resolves the Streamlit
# service names at startup, so recreate it last — once every replica answers.
wait_streamlit_healthy
recreate_nginx
