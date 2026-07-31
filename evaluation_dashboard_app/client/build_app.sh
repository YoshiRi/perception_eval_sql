#!/usr/bin/env bash
# Build the double-clickable local client.
#
#   ./client/build_app.sh              # single-file executable
#   ./client/build_app.sh --desktop    # also install a Linux .desktop launcher
#
# Run from the repo root. Output: dist/evaldash-local
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! python3 -c "import PyInstaller" 2>/dev/null; then
  echo "error: PyInstaller is missing. Install it with: pip install pyinstaller" >&2
  exit 1
fi

if ! python3 -c "import webview" 2>/dev/null; then
  echo "note: pywebview is not installed, so the packaged app will open the system"
  echo "      browser instead of a native window. To get a real window:"
  echo "        pip install pywebview        # plus WebKitGTK on Linux:"
  echo "        sudo apt install gir1.2-webkit2-4.0 python3-gi"
  echo
fi

echo "==> building"
python3 -m PyInstaller client/evaldash_local.spec --noconfirm --clean

BIN="dist/evaldash-local"
if [[ ! -x "$BIN" ]]; then
  echo "error: build did not produce $BIN" >&2
  exit 1
fi
echo "==> built $BIN ($(du -h "$BIN" | cut -f1))"

# A CLI-only smoke test is not enough: `where` and `--help` import nothing, so a
# missing duckdb/static asset in the bundle still passes. Start the real server from a
# different working directory and fetch a page plus a DuckDB-backed route.
echo "==> smoke test"
SMOKE_HOME="$(mktemp -d)"
SMOKE_PORT=$(( 20000 + RANDOM % 30000 ))
trap 'rm -rf "$SMOKE_HOME"' EXIT

"$BIN" --cli where >/dev/null
"$BIN" --cli --help >/dev/null

( cd / && EVALDASH_HOME="$SMOKE_HOME" "$REPO_ROOT/$BIN" --cli serve --port "$SMOKE_PORT" \
    >"$SMOKE_HOME/serve.log" 2>&1 & echo $! > "$SMOKE_HOME/pid" )
for _ in $(seq 1 40); do
  curl -sf -m 2 "http://127.0.0.1:$SMOKE_PORT/api/health" >/dev/null 2>&1 && break
  sleep 0.5
done

SMOKE_PID="$(cat "$SMOKE_HOME/pid" 2>/dev/null || true)"
fail() { echo "error: smoke test failed: $1" >&2; sed 's/^/    /' "$SMOKE_HOME/serve.log" >&2
         [[ -n "$SMOKE_PID" ]] && kill "$SMOKE_PID" 2>/dev/null; exit 1; }

curl -sf -m 5 "http://127.0.0.1:$SMOKE_PORT/api/health" >/dev/null || fail "server did not start"
# Proves the bundled static/ tree is reachable via app_paths, from any cwd.
# "" is the client home page, which lives in the bundle like the other assets.
for ASSET in "" explorer viewer bbox_theme.js bbox_explorer.css events.js; do
  curl -sf -m 5 "http://127.0.0.1:$SMOKE_PORT/$ASSET" -o /dev/null || fail "asset missing: /$ASSET"
done
# The home page drives the in-app download UI; a 200 with no markup means a broken build.
curl -sf -m 5 "http://127.0.0.1:$SMOKE_PORT/" | grep -q 'api/client/state' \
  || fail "home page did not render the client UI"
curl -sf -m 10 -X POST "http://127.0.0.1:$SMOKE_PORT/api/client/state" -d '{}' \
  | grep -q '"tiers"' || fail "client state route did not answer"
# Proves duckdb loaded and can answer a query route (empty workspace is fine).
curl -sf -m 15 -X POST "http://127.0.0.1:$SMOKE_PORT/api/parquets" -d '{}' \
  | grep -q '"items"' || fail "duckdb-backed /api/parquets did not answer"

[[ -n "$SMOKE_PID" ]] && kill "$SMOKE_PID" 2>/dev/null
echo "    ok (health, 5 assets, duckdb route)"

if [[ "${1:-}" == "--desktop" ]]; then
  DESKTOP_DIR="$HOME/.local/share/applications"
  mkdir -p "$DESKTOP_DIR"
  cat > "$DESKTOP_DIR/evaldash-local.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Evaluation Dashboard (Local)
Comment=Inspect downloaded perception evaluation results offline
Exec=$REPO_ROOT/$BIN
Terminal=false
Categories=Development;Science;
EOF
  chmod +x "$DESKTOP_DIR/evaldash-local.desktop"
  echo "==> installed launcher: $DESKTOP_DIR/evaldash-local.desktop"
fi

cat <<EOF

Next steps:
  $BIN --cli login --server <url> --token <token>
  $BIN --cli pull <run> --tier criteria
  $BIN                      # double-click equivalent: opens the viewer
EOF
