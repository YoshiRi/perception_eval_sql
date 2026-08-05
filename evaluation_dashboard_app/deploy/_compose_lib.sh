#!/usr/bin/env bash
# Shared compose plumbing for the numbered deploy scripts. Source it; do not execute.
#
# Single source of truth for three things that used to be per-script (and drifted):
#   1. --env-file .env on every docker compose call.
#   2. Which Streamlit replicas exist: EVAL_COMPOSE_STREAMLIT_REPLICAS in .env drives
#      both the enabled compose profiles and the nginx upstream list, so the two can
#      never disagree.
#   3. Recreate-not-restart. `docker compose restart` reuses the container's baked-in
#      environment, so an .env edit silently has no effect. Every script here brings
#      services up with `up -d` (recreating when the config changed) instead.
#
# Background: streamlit2 lived behind a bare `profiles: [ha]`, so plain `docker compose
# up -d` ignored it for five months. It kept serving stale env (T4_VISUALIZER_* unset,
# so viewer links fell back to http://localhost:8000). dc_all() below always passes
# --profile "*" for teardown/inspection so nothing can hide from the tooling again.

# shellcheck shell=bash

if [[ -z "${DEPLOY_DIR:-}" ]]; then
  echo "_compose_lib.sh: DEPLOY_DIR must be set before sourcing." >&2
  exit 1
fi

# Highest streamlitN service defined in docker-compose.yml. Raise both together.
MAX_STREAMLIT_REPLICAS=3

require_env_file() {
  if [[ ! -f "$DEPLOY_DIR/.env" ]]; then
    echo "Error: .env not found in $DEPLOY_DIR. Run 01_SETUP_ENV.sh first, then edit .env" >&2
    exit 1
  fi
}

# Load .env into this shell (for EVAL_COMPOSE_* knobs) without clobbering values that
# are already exported on the command line.
load_deploy_env() {
  [[ -f "$DEPLOY_DIR/.env" ]] || return 0
  set -a
  # shellcheck disable=SC1091
  source "$DEPLOY_DIR/.env"
  set +a
}

streamlit_replica_count() {
  local count="${EVAL_COMPOSE_STREAMLIT_REPLICAS:-1}"
  if [[ ! "$count" =~ ^[0-9]+$ || "$count" -lt 1 || "$count" -gt "$MAX_STREAMLIT_REPLICAS" ]]; then
    echo "Error: EVAL_COMPOSE_STREAMLIT_REPLICAS must be 1..${MAX_STREAMLIT_REPLICAS} (got '${count}')." >&2
    echo "       More replicas: add streamlit$((MAX_STREAMLIT_REPLICAS + 1)) to docker-compose.yml and raise MAX_STREAMLIT_REPLICAS." >&2
    exit 1
  fi
  echo "$count"
}

# streamlit1 .. streamlitN for the configured replica count.
streamlit_services() {
  local count i
  count="$(streamlit_replica_count)"
  for ((i = 1; i <= count; i++)); do
    echo "streamlit${i}"
  done
}

# Every streamlitN service the compose file defines, enabled or not.
all_streamlit_services() {
  local i
  for ((i = 1; i <= MAX_STREAMLIT_REPLICAS; i++)); do
    echo "streamlit${i}"
  done
}

# Export COMPOSE_PROFILES (so replicas 2..N are part of *every* compose command in this
# shell, including plain `docker compose ps`) and the nginx upstream server list.
setup_compose_env() {
  local count i profiles=() servers=""
  count="$(streamlit_replica_count)"
  for ((i = 2; i <= count; i++)); do
    profiles+=("streamlit${i}")
  done
  for ((i = 1; i <= count; i++)); do
    servers+="server streamlit${i}:8501; "
  done
  COMPOSE_PROFILES="$(
    IFS=,
    echo "${profiles[*]-}"
  )"
  STREAMLIT_UPSTREAM_SERVERS="${servers% }"
  export COMPOSE_PROFILES STREAMLIT_UPSTREAM_SERVERS
  STREAMLIT_REPLICAS="$count"
  export STREAMLIT_REPLICAS
}

# docker compose with the project's .env and the profiles for the configured replicas.
dc() {
  local args=(docker compose)
  [[ -f "$DEPLOY_DIR/.env" ]] && args+=(--env-file "$DEPLOY_DIR/.env")
  "${args[@]}" "$@"
}

# docker compose across *all* profiles — use for down/ps/logs so containers from a
# previously higher replica count can still be seen and removed.
dc_all() {
  local args=(docker compose --profile "*")
  [[ -f "$DEPLOY_DIR/.env" ]] && args+=(--env-file "$DEPLOY_DIR/.env")
  "${args[@]}" "$@"
}

# Stop and remove streamlit replicas above the configured count (lowering the number
# must not leave an orphan serving stale config behind nginx's back).
prune_extra_streamlit() {
  local count svc keep extras=()
  count="$(streamlit_replica_count)"
  while read -r svc; do
    keep=0
    [[ "${svc#streamlit}" -le "$count" ]] && keep=1
    if [[ "$keep" == "0" ]] && [[ -n "$(dc_all ps -aq "$svc" 2>/dev/null)" ]]; then
      extras+=("$svc")
    fi
  done < <(all_streamlit_services)

  if [[ "${#extras[@]}" -gt 0 ]]; then
    echo "Removing Streamlit replicas above EVAL_COMPOSE_STREAMLIT_REPLICAS=${count}: ${extras[*]}" >&2
    dc_all rm -sf "${extras[@]}" >/dev/null
  fi
}

# Block until every configured replica reports healthy; nginx resolves upstream names
# at startup, so recreating it against a not-yet-running replica crash-loops it.
wait_streamlit_healthy() {
  local timeout="${1:-180}" svc cid state deadline
  deadline=$((SECONDS + timeout))
  while read -r svc; do
    while :; do
      cid="$(dc ps -q "$svc" 2>/dev/null | head -1)"
      if [[ -n "$cid" ]]; then
        state="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid" 2>/dev/null || echo "")"
        [[ "$state" == "healthy" || "$state" == "running" ]] && break
      fi
      if [[ "$SECONDS" -ge "$deadline" ]]; then
        echo "Warning: ${svc} is not healthy after ${timeout}s (state=${state:-missing}); continuing." >&2
        break
      fi
      sleep 2
    done
  done < <(streamlit_services)
}

# Recreate nginx last: it renders nginx.conf from the template with the current
# STREAMLIT_UPSTREAM_SERVERS and re-resolves the Streamlit container IPs.
recreate_nginx() {
  dc up -d --no-deps --force-recreate nginx
  wait_nginx_listening
}

# Wait until nginx accepts connections again. Beyond hiding the second of downtime after
# a recreate, this catches the one way the templated config can fail hard: an upstream
# name that does not resolve aborts startup, and the container would just crash-loop.
wait_nginx_listening() {
  local timeout="${1:-45}" cid deadline
  deadline=$((SECONDS + timeout))
  while [[ "$SECONDS" -lt "$deadline" ]]; do
    cid="$(dc ps -q nginx 2>/dev/null | head -1)"
    if [[ -n "$cid" ]] && docker exec "$cid" sh -c 'nc -z 127.0.0.1 80' >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Warning: nginx is not accepting connections after ${timeout}s. Recent logs:" >&2
  dc logs --tail 20 nginx >&2 || true
  return 1
}
