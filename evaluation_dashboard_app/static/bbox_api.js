var API_BASE_PLACEHOLDER = window.BBOX_EXPLORER_API_BASE || "__API_BASE__";
var API_BASE = API_BASE_PLACEHOLDER && !API_BASE_PLACEHOLDER.startsWith("__")
  ? API_BASE_PLACEHOLDER
  : (window.location.pathname.startsWith("/bbox-explorer") ? "/bbox-api" : "");

// Where the viewer this explorer opens in its iframe lives. Behind the deployment's
// nginx the two apps sit at /bbox-explorer/ and /bbox-viewer/; served directly by the
// bbox API -- the local client, or a Streamlit page pointed at an API host -- the
// viewer is /viewer under whatever base the API answers on. Hardcoding the nginx path
// made the local client ask itself for an unknown route and render nothing.
var VIEWER_BASE = window.BBOX_VIEWER_URL_BASE
  || (window.location.pathname.startsWith("/bbox-explorer") ? "/bbox-viewer/" : `${API_BASE}/viewer`);

var api = window.api = async function api(route, body = {}) {
  const controller = new AbortController();
  const timeoutMs = Number(body.timeout_ms || 60000);
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  let text;
  try {
    res = await fetch(`${API_BASE}${route}`, {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(body), signal: controller.signal});
    text = await res.text();
  } catch (err) {
    if (err && err.name === "AbortError") throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s: ${route}`);
    throw err;
  } finally {
    clearTimeout(timer);
  }
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch (_err) { throw new Error(`Non-JSON from ${route}: ${text.slice(0, 100)}`); }
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
