#!/usr/bin/env bash
# 06 — Show containers for this stack (all profiles, so extra Streamlit replicas and
# leftovers from a previous replica count are visible instead of hiding).
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
# shellcheck source=_compose_lib.sh
source "$DEPLOY_DIR/_compose_lib.sh"
load_deploy_env
echo "Configured: EVAL_COMPOSE_STREAMLIT_REPLICAS=${EVAL_COMPOSE_STREAMLIT_REPLICAS:-1}, EVAL_COMPOSE_SCALE_WORKER=${EVAL_COMPOSE_SCALE_WORKER:-2}" >&2
dc_all ps "$@"
