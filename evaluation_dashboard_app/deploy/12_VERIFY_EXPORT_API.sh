#!/usr/bin/env bash
# Verify the export API end to end through the real edge (nginx, TLS, Cloudflare).
#
#   ./deploy/12_VERIFY_EXPORT_API.sh <base-url> <token> [cf-client-id] [cf-client-secret]
#
# Example:
#   ./deploy/12_VERIFY_EXPORT_API.sh https://dash.example.com hunter2
#   ./deploy/12_VERIFY_EXPORT_API.sh https://dash.example.com hunter2 <cf-id> <cf-secret>
#
# Checks the things that only differ in production: whether nginx forwards the bearer
# token, whether it forwards Range (so a 465 MB pull can resume), and whether a
# Cloudflare sign-in page is being returned instead of JSON.
set -uo pipefail

BASE="${1:-}"
TOKEN="${2:-}"
CF_ID="${3:-}"
CF_SECRET="${4:-}"

if [[ -z "$BASE" || -z "$TOKEN" ]]; then
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

BASE="${BASE%/}"
PASS=0
FAIL=0

CURL=(curl -sS --max-time 60)
[[ -n "$CF_ID" ]] && CURL+=(-H "CF-Access-Client-Id: $CF_ID")
[[ -n "$CF_SECRET" ]] && CURL+=(-H "CF-Access-Client-Secret: $CF_SECRET")
AUTH=(-H "Authorization: Bearer $TOKEN")

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
note() { printf '        %s\n' "$1"; }

# The API sits at /bbox-api behind nginx, or at the root when hit directly on :8765.
echo "==> locating the export API under $BASE"
API=""
for SUFFIX in "/bbox-api" ""; do
  BODY="$("${CURL[@]}" "$BASE$SUFFIX/api/export_health" 2>/dev/null || true)"
  if [[ "$BODY" == *eval_dashboard_export* ]]; then
    API="$BASE$SUFFIX"
    ok "found export API at $API"
    break
  fi
done
if [[ -z "$API" ]]; then
  bad "no export API found at $BASE/bbox-api or $BASE"
  note "last response: ${BODY:0:200}"
  if [[ "$BODY" == *"<html"* ]]; then
    note "Looks like HTML. If this is a Cloudflare sign-in page, pass a service token:"
    note "  $0 $BASE $TOKEN <cf-client-id> <cf-client-secret>"
  fi
  echo; echo "0 passed, 1 failed"; exit 1
fi

HEALTH="$("${CURL[@]}" "$API/api/export_health")"
if [[ "$HEALTH" == *'"enabled":true'* ]]; then
  ok "exports enabled on the server"
else
  bad "exports are DISABLED (EVAL_EXPORT_TOKEN not set in the container)"
  note "run ./deploy/11_ENABLE_EXPORT_API.sh --apply then ./deploy/10_RESTART_STREAMLIT.sh"
fi
note "data root: $(printf '%s' "$HEALTH" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("data_root"))' 2>/dev/null || echo '?')"

echo "==> auth"
CODE="$("${CURL[@]}" -o /dev/null -w '%{http_code}' -X POST "$API/api/runs" -d '{}')"
[[ "$CODE" == "401" ]] && ok "unauthenticated request rejected (401)" \
  || bad "expected 401 without a token, got $CODE"

CODE="$("${CURL[@]}" -o /dev/null -w '%{http_code}' -H "Authorization: Bearer wrong-$TOKEN" -X POST "$API/api/runs" -d '{}')"
[[ "$CODE" == "401" ]] && ok "wrong token rejected (401)" \
  || bad "expected 401 for a wrong token, got $CODE"

RUNS="$("${CURL[@]}" "${AUTH[@]}" -X POST "$API/api/runs" -d '{"sizes":false}')"
if [[ "$RUNS" == *'"items"'* ]]; then
  COUNT="$(printf '%s' "$RUNS" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("items") or []))')"
  ok "authenticated: $COUNT run(s) visible — nginx forwards Authorization correctly"
else
  bad "authenticated /api/runs did not return items"
  note "response: ${RUNS:0:200}"
  echo; echo "$PASS passed, $FAIL failed"; exit 1
fi

RUN="$(printf '%s' "$RUNS" | python3 -c 'import json,sys
items=json.load(sys.stdin).get("items") or []
print(items[0]["name"] if items else "")')"
if [[ -z "$RUN" ]]; then
  note "no runs on the server; skipping manifest and download checks"
  echo; echo "$PASS passed, $FAIL failed"; exit $(( FAIL > 0 ? 1 : 0 ))
fi

echo "==> manifest for '$RUN'"
MAN="$("${CURL[@]}" "${AUTH[@]}" -X POST "$API/api/export_manifest" \
        -d "{\"run\":\"$RUN\",\"tier\":\"minimal\"}")"
read -r NFILES NBYTES REL <<<"$(printf '%s' "$MAN" | python3 -c 'import json,sys
d=json.load(sys.stdin); files=d.get("files") or []
small=min(files, key=lambda f: f["size"]) if files else {"rel_path":""}
print(len(files), d.get("total_bytes") or 0, small["rel_path"])')"
if [[ "$NFILES" -gt 0 ]]; then
  ok "manifest: $NFILES file(s), $(python3 -c "print(f'{$NBYTES/1e6:.1f} MB')")"
else
  bad "manifest returned no files"
fi

echo "==> download of '$REL'"
SIZE="$("${CURL[@]}" "${AUTH[@]}" -o /dev/null -w '%{http_code} %{size_download}' \
        "$API/api/export_file?run=$RUN&rel_path=$REL")"
[[ "${SIZE%% *}" == "200" ]] && ok "full download ok (${SIZE##* } bytes)" \
  || bad "full download returned HTTP ${SIZE%% *}"

# The critical production check: nginx must forward Range, or resuming a large pull
# silently degrades into re-downloading from zero every time.
HDRS="$("${CURL[@]}" "${AUTH[@]}" -D - -o /dev/null -r 0-3 \
        "$API/api/export_file?run=$RUN&rel_path=$REL" 2>/dev/null | tr -d '\r')"
if grep -qi '^HTTP/[0-9.]* 206' <<<"$HDRS"; then
  ok "ranged request honoured (206) — resumable pulls work through nginx"
  grep -i '^content-range:' <<<"$HDRS" | sed 's/^/        /'
else
  bad "ranged request NOT honoured; large pulls cannot resume"
  note "status: $(grep -i '^HTTP/' <<<"$HDRS" | head -1)"
  note "check that nginx has 'proxy_buffering off' on the /bbox-api/ location"
fi

grep -qi '^accept-ranges: *bytes' <<<"$HDRS" && ok "Accept-Ranges advertised" \
  || note "Accept-Ranges header not visible (harmless if 206 worked)"

echo
echo "==> pre-bake coverage"
PRE="$(printf '%s' "$RUNS" | python3 -c 'import json,sys
items=json.load(sys.stdin).get("items") or []
n=sum(1 for i in items if i.get("prebaked"))
print(f"{n}/{len(items)} run(s) have pre-baked DevOps answers")')"
note "$PRE"
note "bake more with: python3 -m backend.prebake_cli --report"

echo
echo "$PASS passed, $FAIL failed"
if [[ "$FAIL" -eq 0 ]]; then
  echo
  echo "Ready. Teammates can now run:"
  echo "    evaldash-local login --server $BASE --token <token>"
  echo "    evaldash-local runs"
fi
exit $(( FAIL > 0 ? 1 : 0 ))
