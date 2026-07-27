var API_BASE_PLACEHOLDER = window.BBOX_VIEWER_API_BASE || "__API_BASE__";
var API_BASE = API_BASE_PLACEHOLDER && !API_BASE_PLACEHOLDER.startsWith("__")
  ? API_BASE_PLACEHOLDER
  : (window.location.pathname.startsWith("/bbox-viewer") ? "/bbox-api" : "");

function readDeepLink() {
  const q = new URLSearchParams(window.location.search);
  const link = {
    path: q.get("path") || "",
    pathB: q.get("path_b") || q.get("pathB") || "",
    compare: q.get("compare") === "1" || q.get("compare") === "true",
    suite: q.get("suite") || q.get("suite_name") || "",
    scenario: q.get("scenario") || q.get("scenario_name") || "",
    topic: q.get("topic") || q.get("topic_name") || "",
    frame: q.get("frame") || q.get("frame_index") || "",
    lens: q.get("lens") || q.get("compare_lens") || "",
    layout: q.get("layout") || q.get("compare_layout") || ""
  };
  return Object.values(link).some(Boolean) ? link : null;
}

var api = window.api = async function api(route, body = {}) {
  const controller = new AbortController();
  const timeoutMs = Number(body.timeout_ms || 60000);
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  let text;
  try {
    res = await fetch(`${API_BASE}${route}`, {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify(body),
      signal: controller.signal
    });
    text = await res.text();
  } catch (err) {
    if (err && err.name === "AbortError") throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s: ${route}`);
    throw err;
  } finally {
    clearTimeout(timer);
  }
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch (_err) {
    throw new Error(`API returned non-JSON at ${API_BASE}${route}: HTTP ${res.status} ${text.slice(0, 120)}`);
  }
  if (!res.ok || data.error) throw new Error(data.error || `API ${res.status}`);
  return data;
}
