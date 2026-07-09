#!/usr/bin/env bash
# 09 — Restart worker containers (pick up worker/ or lib/ code changes without full rebuild).
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

IDLE_ONLY=0
RESTART_ARGS=()

usage() {
  cat >&2 <<'EOF'
Usage: ./09_RESTART_WORKER.sh [--idle-only|--idle] [docker compose restart options]

By default, restarts all worker containers.
With --idle-only, restarts only worker containers whose RQ worker state is idle.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --idle-only|--idle)
      IDLE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      RESTART_ARGS+=("$1")
      shift
      ;;
  esac
done

COMPOSE=(docker compose)
if [[ -f .env ]]; then
  COMPOSE+=(--env-file .env)
fi

if [[ "$IDLE_ONLY" != "1" ]]; then
  exec "${COMPOSE[@]}" restart worker "${RESTART_ARGS[@]}"
fi

mapfile -t WORKER_CONTAINERS < <("${COMPOSE[@]}" ps -q worker)
if [[ "${#WORKER_CONTAINERS[@]}" -eq 0 ]]; then
  echo "No worker containers found." >&2
  exit 0
fi

IDLE_CONTAINERS=()
for container_id in "${WORKER_CONTAINERS[@]}"; do
  idle_status="$(
    docker exec -i "$container_id" python3 - <<'PY' 2>/dev/null || true
import os
import socket
import sys

from redis import Redis
from rq import Worker

conn = Redis.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"))
hostname = socket.gethostname()

workers = [
    worker
    for worker in Worker.all(connection=conn)
    if worker.name == hostname or worker.name.startswith(hostname + ".")
]

if not workers:
    print("unknown")
    sys.exit(0)

for worker in workers:
    raw_state = worker.get_state()
    state = getattr(raw_state, "value", raw_state)
    state = str(state).lower()
    current_job_id = worker.get_current_job_id()
    if state == "idle" and not current_job_id:
        print("idle")
    else:
        print(f"busy:{state}:{current_job_id or ''}")
PY
  )"
  if echo "$idle_status" | grep -qx "idle"; then
    IDLE_CONTAINERS+=("$container_id")
  else
    container_name="$(docker inspect --format '{{.Name}}' "$container_id" 2>/dev/null | sed 's#^/##' || true)"
    echo "Skipping ${container_name:-$container_id}: worker is not idle (${idle_status:-unknown})." >&2
  fi
done

if [[ "${#IDLE_CONTAINERS[@]}" -eq 0 ]]; then
  echo "No idle worker containers to restart." >&2
  exit 0
fi

echo "Restarting ${#IDLE_CONTAINERS[@]} idle worker container(s)." >&2
exec docker restart "${RESTART_ARGS[@]}" "${IDLE_CONTAINERS[@]}"
