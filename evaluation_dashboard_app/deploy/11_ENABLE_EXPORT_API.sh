#!/usr/bin/env bash
# Enable the export API so local clients (evaldash-local) can download run files.
#
#   ./deploy/11_ENABLE_EXPORT_API.sh            # generate a token and show next steps
#   ./deploy/11_ENABLE_EXPORT_API.sh --apply    # write it into deploy/.env
#   ./deploy/11_ENABLE_EXPORT_API.sh --show     # print the currently configured token
#
# The export routes stream raw file bytes, unlike the rest of the bbox API which only
# returns aggregates. They are therefore CLOSED until EVAL_EXPORT_TOKEN is set, and this
# script never enables them implicitly -- you have to pass --apply.
#
# Note on files: docker compose is invoked as `docker compose --env-file .env` (see
# 04_START.sh), so deploy/.env is the only file that reaches the containers.
# deploy/.env.local is just a naming convention mentioned in .env.example; nothing
# loads it, so a token written there has no effect.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$DEPLOY_DIR/.env"
ENV_LOCAL="$DEPLOY_DIR/.env.local"
MODE="${1:-}"

warn_stray_local() {
  if [[ -f "$ENV_LOCAL" ]] && grep -q '^EVAL_EXPORT_TOKEN=' "$ENV_LOCAL"; then
    echo
    echo "WARNING: $ENV_LOCAL also sets EVAL_EXPORT_TOKEN, but compose does not read" >&2
    echo "         that file. It is ignored; deploy/.env is the one that counts." >&2
  fi
}

show_current() {
  if [[ -f "$ENV_FILE" ]] && grep -q '^EVAL_EXPORT_TOKEN=' "$ENV_FILE"; then
    echo "configured in $ENV_FILE:"
    grep '^EVAL_EXPORT_TOKEN=' "$ENV_FILE"
    echo
    echo "Note: a running container only sees this after a RECREATE, not a restart:"
    echo "  cd deploy && docker compose --env-file .env up -d --no-build streamlit1"
  else
    echo "EVAL_EXPORT_TOKEN is not set in $ENV_FILE — the export API is disabled."
  fi
  warn_stray_local
}

if [[ "$MODE" == "--show" ]]; then
  show_current
  exit 0
fi

TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

if [[ "$MODE" == "--apply" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "error: $ENV_FILE does not exist. Create it first: ./deploy/01_SETUP_ENV.sh" >&2
    exit 1
  fi
  if grep -q '^EVAL_EXPORT_TOKEN=' "$ENV_FILE"; then
    echo "EVAL_EXPORT_TOKEN already present in $ENV_FILE; refusing to overwrite." >&2
    echo "Remove the existing line first if you intend to rotate it." >&2
    show_current
    exit 1
  fi
  {
    echo ""
    echo "# Enables the local-client export API (see client/README.md). Secret."
    echo "EVAL_EXPORT_TOKEN=$TOKEN"
  } >> "$ENV_FILE"
  echo "==> appended EVAL_EXPORT_TOKEN to $ENV_FILE"
  warn_stray_local
  echo
  echo "Now RECREATE the container so it picks up the new variable:"
  echo "    cd deploy && docker compose --env-file .env up -d --no-build streamlit1"
  echo
  echo "  (10_RESTART_STREAMLIT.sh is NOT enough: 'docker compose restart' reuses the"
  echo "   existing container config and does not re-read env_file.)"
  echo
  echo "Then verify:"
  echo "    ./deploy/12_VERIFY_EXPORT_API.sh http://localhost $TOKEN"
  echo
  echo "Give teammates this token plus:"
  echo "    evaldash-local login --server http://<this-host> --token $TOKEN"
else
  echo "Generated token (not written anywhere):"
  echo
  echo "    EVAL_EXPORT_TOKEN=$TOKEN"
  echo
  echo "To apply it automatically:  $0 --apply"
  echo "Or append the line above to: $ENV_FILE"
  echo "Then recreate the container: docker compose --env-file .env up -d --no-build streamlit1"
  echo
  show_current
fi
