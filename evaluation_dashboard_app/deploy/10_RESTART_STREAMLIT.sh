#!/usr/bin/env bash
# 10 — Restart Streamlit only (keeps workers, Redis and Postgres running).
#
# Recreates the containers rather than `docker compose restart`ing them: a restart
# reuses the container's baked-in environment, so an .env edit would appear to be
# applied while the app kept the old values. Nginx is recreated afterwards because it
# resolves the Streamlit container IPs at startup.
#
# Default: every Streamlit replica for the configured EVAL_COMPOSE_STREAMLIT_REPLICAS.
# Optional: ./10_RESTART_STREAMLIT.sh streamlit1
#           ./10_RESTART_STREAMLIT.sh --keep-nginx streamlit2
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
# shellcheck source=_compose_lib.sh
source "$DEPLOY_DIR/_compose_lib.sh"
load_deploy_env
setup_compose_env

KEEP_NGINX=0
SERVICES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-nginx)
      KEEP_NGINX=1
      shift
      ;;
    -h|--help)
      sed -n '2,11p' "${BASH_SOURCE[0]}" >&2
      exit 0
      ;;
    *)
      SERVICES+=("$1")
      shift
      ;;
  esac
done

if [[ ${#SERVICES[@]} -eq 0 ]]; then
  mapfile -t SERVICES < <(streamlit_services)
fi

echo "Recreating: ${SERVICES[*]}" >&2
dc up -d --no-deps --force-recreate "${SERVICES[@]}"

prune_extra_streamlit

if [[ "$KEEP_NGINX" == "1" ]]; then
  echo "Left nginx untouched (--keep-nginx); it may hold stale upstream IPs." >&2
  exit 0
fi
wait_streamlit_healthy
recreate_nginx
