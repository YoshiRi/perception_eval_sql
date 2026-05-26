#!/usr/bin/env bash
# 04 — Start or update the full stack with docker compose up -d.
# Default: 2 worker replicas (EVAL_COMPOSE_SCALE_WORKER in .env). Override: ./04_START.sh --scale worker=1 (last --scale wins).
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

dc() { docker compose --env-file .env "$@"; }

dc up -d --scale "worker=${WORKER_SCALE}" "$@"

# Nginx resolves Docker service names at startup. Recreate it after Streamlit is
# up so it remounts the current nginx.conf and cannot keep a stale container IP.
dc up -d --no-deps --force-recreate nginx
