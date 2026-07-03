#!/usr/bin/env bash
# 10 — Restart Streamlit only (keeps workers, Redis, Postgres, and Nginx running).
# By default, restarts every currently running compose service named streamlit*.
# Optional: ./10_RESTART_STREAMLIT.sh streamlit1 streamlit2
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

COMPOSE=(docker compose --profile "*")
if [[ -f .env ]]; then
  COMPOSE+=(--env-file .env)
fi

SERVICES=("$@")
if [[ ${#SERVICES[@]} -eq 0 ]]; then
  mapfile -t SERVICES < <("${COMPOSE[@]}" ps --services --status running | sed -n '/^streamlit/p')
  if [[ ${#SERVICES[@]} -eq 0 ]]; then
    echo "No running Streamlit services found." >&2
    exit 1
  fi
fi

exec "${COMPOSE[@]}" restart "${SERVICES[@]}"
