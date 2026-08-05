#!/usr/bin/env bash
# 03 — One-time: start Postgres if needed, then run init_db (creates task tables).
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
# shellcheck source=_compose_lib.sh
source "$DEPLOY_DIR/_compose_lib.sh"
require_env_file
load_deploy_env
setup_compose_env
dc up -d postgres
dc run --rm init_db
echo "Database init finished."
