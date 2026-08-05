#!/usr/bin/env bash
# 08 — Rebuild images then start the stack (= 02_BUILD + 04_START). Build-only: use 02_BUILD.sh alone.
# 04_START.sh recreates every service, applies EVAL_COMPOSE_STREAMLIT_REPLICAS and
# EVAL_COMPOSE_SCALE_WORKER, and refreshes nginx, so nothing keeps the old image or env.
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
# shellcheck source=_compose_lib.sh
source "$DEPLOY_DIR/_compose_lib.sh"
require_env_file
load_deploy_env
setup_compose_env
dc build "$@"
exec "$DEPLOY_DIR/04_START.sh"
