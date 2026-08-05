#!/usr/bin/env bash
# 07 — Follow logs (all services by default, all profiles). Narrow: ./07_LOGS.sh worker
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"
# shellcheck source=_compose_lib.sh
source "$DEPLOY_DIR/_compose_lib.sh"
dc_all logs -f "$@"
