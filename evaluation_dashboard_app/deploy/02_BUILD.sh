#!/usr/bin/env bash
# 02 — Build images from docker-compose.yml (from this directory).
# Extra args: e.g. ./02_BUILD.sh --no-cache
# All Streamlit replicas and the workers share one image (evaluation-dashboard), so a
# build covers them regardless of how many replicas are enabled.
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
# shellcheck source=_compose_lib.sh
source "$DEPLOY_DIR/_compose_lib.sh"
load_deploy_env
setup_compose_env
dc build "$@"
