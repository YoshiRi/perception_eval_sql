// JSON API bridge for the TLR analysis viewer. Same idiom as bbox_api.js: the
// server substitutes __API_BASE__ when it renders the page; behind nginx the
// page lives at /tlr-viewer/ and the JSON routes at /bbox-api/.
var TLR_API_BASE_PLACEHOLDER = window.TLR_API_BASE || "__API_BASE__";
var API_BASE = TLR_API_BASE_PLACEHOLDER && !TLR_API_BASE_PLACEHOLDER.startsWith("__")
  ? TLR_API_BASE_PLACEHOLDER
  : (window.location.pathname.startsWith("/tlr-viewer") ? "/bbox-api" : "");

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
