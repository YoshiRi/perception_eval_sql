#!/usr/bin/env bash
# 08 — Rebuild images then start the stack (= 02_BUILD + 04_START). Build-only: use 02_BUILD.sh alone.
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
if [[ ! -f .env ]]; then
  echo "Error: .env not found. Run 01_SETUP_ENV.sh then edit .env" >&2
  exit 1
fi
docker compose --env-file .env build "$@"
exec "$DEPLOY_DIR/04_START.sh"
