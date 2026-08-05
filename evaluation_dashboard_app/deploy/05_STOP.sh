#!/usr/bin/env bash
# 05 — Stop containers (keeps volumes e.g. postgres_data). Optional: ./05_STOP.sh -v
# Tears down across all profiles, so optional Streamlit replicas go too — whatever the
# current EVAL_COMPOSE_STREAMLIT_REPLICAS happens to be.
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
# shellcheck source=_compose_lib.sh
source "$DEPLOY_DIR/_compose_lib.sh"
dc_all down "$@"
