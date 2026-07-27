var PREVIEW_MAX_COORD_ABS = 10000;
var PREVIEW_MAX_VIEW_EXTENT = 500;

function validPreviewBox(box) {
  const x = Number(box && box.x);
  const y = Number(box && box.y);
  return Number.isFinite(x) && Number.isFinite(y)
    && Math.abs(x) <= PREVIEW_MAX_COORD_ABS
    && Math.abs(y) <= PREVIEW_MAX_COORD_ABS;
}
function normalizePreviewFrames(frames) {
  return [...(frames || [])]
    .map(f => ({...f, frame: Number(f.frame), boxes: (f.boxes || []).filter(validPreviewBox)}))
    .filter(f => Number.isFinite(f.frame))
    .sort((a, b) => a.frame - b.frame);
}
function mergePreviewFrames(aFrames, bFrames) {
  const merged = new Map();
  for (const f of normalizePreviewFrames(aFrames)) {
    merged.set(Number(f.frame), {frame: Number(f.frame), boxes: [...(f.boxes || [])]});
  }
  for (const f of normalizePreviewFrames(bFrames)) {
    const frame = Number(f.frame);
    const hit = merged.get(frame) || {frame, boxes: []};
    hit.boxes.push(...(f.boxes || []));
    merged.set(frame, hit);
  }
  return [...merged.values()].sort((a, b) => a.frame - b.frame);
}
function setPreviewFrames(frames, message = "") {
  state.previewFrames = normalizePreviewFrames(frames);
  state.previewIndex = 0;
  els.previewSlider.max = String(Math.max(0, state.previewFrames.length - 1));
  els.previewSlider.value = "0";
  renderPreview(message);
}
function clampPreviewWindow() {
  const stage = els.canvas.getBoundingClientRect();
  const win = els.previewWindow;
  const minW = Math.min(360, Math.max(280, stage.width - 36));
  const minH = Math.min(390, Math.max(320, stage.height - 36));
  const w = Math.max(minW, Math.min(win.offsetWidth || 520, Math.max(minW, stage.width - 36)));
  const h = Math.max(minH, Math.min(win.offsetHeight || 470, Math.max(minH, stage.height - 36)));
  let left = parseFloat(win.style.left || "22");
  let top = parseFloat(win.style.top || "88");
  left = Math.max(8, Math.min(left, stage.width - w - 8));
  top = Math.max(8, Math.min(top, stage.height - h - 8));
  win.style.left = `${left}px`;
  win.style.top = `${top}px`;
  win.style.width = `${w}px`;
  win.style.height = `${h}px`;
}
function showPreviewWindow() {
  state.previewVisible = true;
  els.previewWindow.classList.add("show");
  if (!els.previewWindow.style.left) {
    els.previewWindow.style.left = "22px";
    els.previewWindow.style.top = "88px";
  }
  clampPreviewWindow();
  renderPreview();
}
async function loadPreview(s, options = {}) {
  const requestId = ++state.previewRequestId;
  state.previewFrames = [];
  state.previewIndex = 0;
  state.previewPanX = 0;
  state.previewPanY = 0;
  state.previewScale = 1;
  els.previewSlider.max = "0";
  els.previewSlider.value = "0";
  renderPreview("Loading scene preview...");
  try {
    const previewFilters = sceneFilters(s);
    if (options.centerFrame != null) {
      const center = Number(options.centerFrame);
      const radius = Math.max(0, Number(options.radius ?? 4) || 0);
      if (Number.isFinite(center)) {
        previewFilters.frame_min = Math.floor(center - radius);
        previewFilters.frame_max = Math.ceil(center + radius);
      }
    }
    const request = {filters: previewFilters, max_rows: 70000, dedupe: true};
    let data;
    if (state.compare) {
      renderPreview("Loading Run A preview...");
      const dataA = await api("/api/frames", { ...request, path: state.path, run: "A", timeout_ms: 15000 });
      if (requestId !== state.previewRequestId) return;
      setPreviewFrames(dataA.frames || [], "Loading Run B preview...");
      let dataB;
      try {
        dataB = await api("/api/frames", { ...request, path: els.parquetB.value, run: "B", timeout_ms: 15000 });
      } catch (err) {
        if (requestId !== state.previewRequestId) return;
        setPreviewFrames(dataA.frames || [], `Run B preview failed: ${err.message}`);
        return;
      }
      data = {frames: mergePreviewFrames(dataA.frames || [], dataB.frames || [])};
    } else {
      data = await api("/api/frames", { ...request, path: state.path, run: "A" });
    }
    if (requestId !== state.previewRequestId) return;
    setPreviewFrames(data.frames || []);
  } catch (err) {
    if (requestId !== state.previewRequestId) return;
    state.previewFrames = [];
    renderPreview(`Preview failed: ${err.message}`);
  }
}
function focusPreviewOnCurvePeak() {
  if (!state.previewFrames.length || !state.curve.length) return;
  const score = f => state.compare ? Math.abs(f.fp || 0) + Math.abs(f.fn || 0) : (f.fp || 0) + (f.fn || 0);
  const peak = [...state.curve].sort((a, b) => score(b) - score(a))[0];
  if (!peak) return;
  let best = 0, dist = Infinity;
  state.previewFrames.forEach((f, i) => {
    const d = Math.abs(Number(f.frame) - Number(peak.frame));
    if (d < dist) { best = i; dist = d; }
  });
  state.previewIndex = best;
  els.previewSlider.value = String(best);
}
function previewBoundsMaxAbs() {
  let maxAbs = 35;
  for (const f of state.previewFrames) {
    for (const b of f.boxes || []) {
      const x = Number(b.x);
      const y = Number(b.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      maxAbs = Math.max(maxAbs, Math.abs(x) + 8, Math.abs(y) + 8);
    }
  }
  return Math.min(PREVIEW_MAX_VIEW_EXTENT, maxAbs);
}
function previewScaleForRect(r) {
  const width = Number(r.width ?? r.w) || 1;
  const height = Number(r.height ?? r.h) || 1;
  return Math.min(width, height) / Math.max(50, previewBoundsMaxAbs() * 2.25) * state.previewScale;
}
function previewInteractionViewport(rect, clientX = null) {
  if (!state.compare) return {x: 0, y: 0, width: rect.width, height: rect.height};
  const gap = 3;
  const half = (rect.width - gap) / 2;
  const localX = clientX == null ? rect.width / 2 : clientX - rect.left;
  return localX <= half
    ? {x: 0, y: 0, width: half, height: rect.height}
    : {x: half + gap, y: 0, width: half, height: rect.height};
}
function previewColor(b) {
  const status = String(b.status || "").toUpperCase();
  const source = String(b.source || "").toUpperCase();
  if (source === "GT" && status === "FN") return "#fbbf24";
  if (source === "GT") return "#34d399";
  if (status === "FP") return "#fb7185";
  if (status === "TP") return "#38bdf8";
  return "#cbd5e1";
}
function previewLayerKey(b) {
  const source = String(b.source || "").toUpperCase();
  const status = String(b.status || "").toUpperCase();
  if (source === "GT" && status === "TP") return "gt_tp";
  if (source === "GT" && status === "FN") return "gt_fn";
  if (source === "EST" && status === "TP") return "est_tp";
  if (source === "EST" && status === "FP") return "est_fp";
  if (source === "GT") return "gt_tp";
  if (source === "EST") return "est_tp";
  return "";
}
function previewLayerVisible(b) {
  const key = previewLayerKey(b);
  if (!key) return true;
  const btn = els.previewLayers.querySelector(`[data-layer="${key}"]`);
  return !!(btn && btn.classList.contains("active"));
}
function isPointLikeBox(b) {
  const shape = String(b.shape_type || b.type || "").toLowerCase();
  const l = Number(b.length) || 0;
  const w = Number(b.width) || 0;
  return shape === "polygon" || shape === "point" || l <= 0 || w <= 0;
}
function previewPoint(b, sx, sy, scale) {
  return [
    sx - ((Number(b.y) || 0) - state.previewPanY) * scale,
    sy - ((Number(b.x) || 0) - state.previewPanX) * scale
  ];
}
function previewBoxScreenCenter(b, sx, sy, scale) {
  return previewPoint(b, sx, sy, scale);
}
function previewHoverText(b) {
  if (!b) return "";
  const parts = [];
  if (state.compare && b.run) parts.push(`Run ${b.run}`);
  parts.push(b.label || "object");
  if (b.source || b.status) parts.push(`${b.source || ""}/${b.status || ""}`);
  if (b.confidence != null) parts.push(`conf ${Number(b.confidence).toFixed(2)}`);
  return parts.filter(Boolean).join(" · ");
}
function drawPreviewHoverLabel() {
  const b = state.previewHoverBox;
  if (!b) return;
  const text = previewHoverText(b);
  if (!text) return;
  previewCtx.font = "800 11px Inter, sans-serif";
  const tw = previewCtx.measureText(text).width;
  const x = Math.max(8, Math.min(els.preview.clientWidth - tw - 18, state.previewMouseX + 12));
  const y = Math.max(24, Math.min(els.preview.clientHeight - 10, state.previewMouseY - 12));
  previewCtx.fillStyle = "rgba(2,6,23,.84)";
  previewCtx.strokeStyle = previewColor(b);
  previewCtx.lineWidth = 1;
  previewCtx.fillRect(x - 5, y - 16, tw + 10, 21);
  previewCtx.strokeRect(x - 5, y - 16, tw + 10, 21);
  previewCtx.fillStyle = "#f8fafc";
  previewCtx.fillText(text, x, y);
}
function drawPreviewRings(sx, sy, scale, maxAbs) {
  if (!state.previewShowRings) return;
  previewCtx.save();
  previewCtx.strokeStyle = "rgba(56,189,248,.2)";
  previewCtx.fillStyle = "rgba(145,164,191,.76)";
  previewCtx.font = "700 10px Inter, sans-serif";
  previewCtx.setLineDash([5, 8]);
  const maxRing = Math.min(PREVIEW_MAX_VIEW_EXTENT, Math.max(20, Math.ceil(maxAbs / 20) * 20));
  for (let m = 20, ringCount = 0; m <= maxRing && ringCount < 25; m += 20, ringCount += 1) {
    previewCtx.beginPath();
    previewCtx.arc(sx, sy, m * scale, 0, Math.PI * 2);
    previewCtx.stroke();
    previewCtx.fillText(`${m}m`, sx + m * scale + 4, sy - 4);
  }
  previewCtx.restore();
}
function drawPreviewPersistentLabels(boxes, sx, sy, scale) {
  if (!state.previewShowLabels) return;
  previewCtx.save();
  previewCtx.font = "800 10px Inter, sans-serif";
  previewCtx.textBaseline = "middle";
  boxes.slice(0, 160).forEach(b => {
    const p = previewBoxScreenCenter(b, sx, sy, scale);
    const text = previewHoverText(b);
    if (!text) return;
    const tw = previewCtx.measureText(text).width;
    const x = p[0] + 7;
    const y = p[1] - 7;
    previewCtx.fillStyle = "rgba(2,6,23,.72)";
    previewCtx.fillRect(x - 3, y - 8, tw + 6, 16);
    previewCtx.strokeStyle = previewColor(b);
    previewCtx.lineWidth = 1;
    previewCtx.strokeRect(x - 3, y - 8, tw + 6, 16);
    previewCtx.fillStyle = "#f8fafc";
    previewCtx.fillText(text, x, y);
  });
  previewCtx.restore();
}
function drawPreviewBox(b, sx, sy, scale) {
  if (isPointLikeBox(b)) {
    const p = previewPoint(b, sx, sy, scale);
    const color = previewColor(b);
    const radius = Math.max(4, Math.min(10, scale * .65));
    previewCtx.strokeStyle = color;
    previewCtx.fillStyle = color;
    previewCtx.lineWidth = 1.8;
    previewCtx.setLineDash([]);
    previewCtx.beginPath();
    previewCtx.arc(p[0], p[1], radius, 0, Math.PI * 2);
    previewCtx.stroke();
    previewCtx.setLineDash([]);
    previewCtx.beginPath();
    previewCtx.arc(p[0], p[1], 2.2, 0, Math.PI * 2);
    previewCtx.fill();
    return;
  }
  const l = Math.max(.01, Number(b.length) || 0) / 2;
  const w = Math.max(.01, Number(b.width) || 0) / 2;
  const yaw = Number(b.yaw) || 0;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const pts = [[l, w], [l, -w], [-l, -w], [-l, w]].map(([x, y]) => [
    sx - ((Number(b.y) || 0) + x * s + y * c - state.previewPanY) * scale,
    sy - ((Number(b.x) || 0) + x * c - y * s - state.previewPanX) * scale
  ]);
  previewCtx.strokeStyle = previewColor(b);
  previewCtx.lineWidth = 1.8;
  previewCtx.setLineDash([]);
  previewCtx.beginPath();
  pts.forEach((p, i) => i ? previewCtx.lineTo(p[0], p[1]) : previewCtx.moveTo(p[0], p[1]));
  previewCtx.closePath();
  previewCtx.stroke();
  previewCtx.setLineDash([]);
  const nose = [
    sx - ((Number(b.y) || 0) + l * s - state.previewPanY) * scale,
    sy - ((Number(b.x) || 0) + l * c - state.previewPanX) * scale
  ];
  previewCtx.fillStyle = previewColor(b);
  previewCtx.beginPath(); previewCtx.arc(nose[0], nose[1], 2, 0, Math.PI * 2); previewCtx.fill();
}
function drawPreviewScene(frame, boxes, viewport, label = "", maxAbs = previewBoundsMaxAbs()) {
  const scale = previewScaleForRect(viewport);
  const sx = viewport.x + viewport.w / 2;
  const sy = viewport.y + viewport.h / 2 + 12;
  previewCtx.save();
  previewCtx.beginPath();
  previewCtx.rect(viewport.x, viewport.y, viewport.w, viewport.h);
  previewCtx.clip();
  previewCtx.strokeStyle = "rgba(148,163,184,.13)";
  previewCtx.lineWidth = 1;
  for (let m = -Math.ceil(maxAbs / 10) * 10; m <= maxAbs; m += 10) {
    previewCtx.beginPath(); previewCtx.moveTo(sx - (m - state.previewPanY) * scale, viewport.y); previewCtx.lineTo(sx - (m - state.previewPanY) * scale, viewport.y + viewport.h); previewCtx.stroke();
    previewCtx.beginPath(); previewCtx.moveTo(viewport.x, sy - (m - state.previewPanX) * scale); previewCtx.lineTo(viewport.x + viewport.w, sy - (m - state.previewPanX) * scale); previewCtx.stroke();
  }
  drawPreviewRings(sx - (0 - state.previewPanY) * scale, sy - (0 - state.previewPanX) * scale, scale, maxAbs);
  const egoX = sx - (0 - state.previewPanY) * scale;
  const egoY = sy - (0 - state.previewPanX) * scale;
  previewCtx.fillStyle = "rgba(234,242,255,.86)";
  previewCtx.beginPath();
  previewCtx.moveTo(egoX, egoY - 9); previewCtx.lineTo(egoX - 6, egoY + 8); previewCtx.lineTo(egoX + 6, egoY + 8); previewCtx.closePath(); previewCtx.fill();
  const sorted = [...boxes].filter(previewLayerVisible).sort((a, b) => (String(a.source) === "GT" ? -1 : 1) - (String(b.source) === "GT" ? -1 : 1));
  sorted.forEach(b => drawPreviewBox(b, sx, sy, scale));
  drawPreviewPersistentLabels(sorted, sx, sy, scale);
  previewCtx.restore();
  if (label) {
    const runName = label === "A" ? shortPathName(state.path) : shortPathName(els.parquetB.value || state.pathB);
    let text = `Run ${label}: ${runName}`;
    previewCtx.fillStyle = "rgba(2,6,23,.72)";
    previewCtx.strokeStyle = "rgba(148,163,184,.28)";
    previewCtx.lineWidth = 1;
    previewCtx.font = "800 11px Inter, sans-serif";
    const maxW = Math.max(60, Math.min(viewport.w - 20, 190));
    if (previewCtx.measureText(text).width > maxW - 20) {
      const suffix = "...";
      while (text.length > 8 && previewCtx.measureText(`${text}${suffix}`).width > maxW - 20) {
        text = text.slice(0, -1);
      }
      text = `${text}${suffix}`;
    }
    const labelW = Math.max(66, Math.min(maxW, previewCtx.measureText(text).width + 20));
    previewCtx.beginPath();
    previewCtx.rect(viewport.x + 10, viewport.y + 9, labelW, 23);
    previewCtx.fill();
    previewCtx.stroke();
    previewCtx.fillStyle = label === "A" ? "#60a5fa" : "#a78bfa";
    previewCtx.fillText(text, viewport.x + 20, viewport.y + 25);
  }
}
function previewViewportsForRect(r) {
  if (!state.compare) return [{label: "", x: 0, y: 0, w: r.width, h: r.height, run: ""}];
  const gap = 3;
  const half = (r.width - gap) / 2;
  return [
    {label: "A", x: 0, y: 0, w: half, h: r.height, run: "A"},
    {label: "B", x: half + gap, y: 0, w: half, h: r.height, run: "B"},
  ];
}
function updatePreviewHover() {
  state.previewHoverBox = null;
  if (!state.previewFrames.length) return;
  const frame = state.previewFrames[Math.max(0, Math.min(state.previewIndex, state.previewFrames.length - 1))];
  const rect = els.preview.getBoundingClientRect();
  let best = null, bestD = 16;
  for (const vp of previewViewportsForRect(rect)) {
    if (state.previewMouseX < vp.x || state.previewMouseX > vp.x + vp.w || state.previewMouseY < vp.y || state.previewMouseY > vp.y + vp.h) continue;
    const scale = previewScaleForRect(vp);
    const sx = vp.x + vp.w / 2;
    const sy = vp.y + vp.h / 2 + 12;
    const boxes = ((frame && frame.boxes) || []).filter(b => (!vp.run || b.run === vp.run) && previewLayerVisible(b));
    for (const b of boxes) {
      const p = previewBoxScreenCenter(b, sx, sy, scale);
      const d = Math.hypot(p[0] - state.previewMouseX, p[1] - state.previewMouseY);
      if (d < bestD) { best = b; bestD = d; }
    }
  }
  state.previewHoverBox = best;
}
function renderPreview(message = "") {
  const r = resizeCanvas(els.preview, previewCtx);
  previewCtx.clearRect(0, 0, r.width, r.height);
  previewCtx.fillStyle = "rgba(2,6,23,.76)";
  previewCtx.fillRect(0, 0, r.width, r.height);
  if (!state.previewFrames.length) {
    const text = message || "No preview frames matched.";
    els.previewStatus.textContent = text;
    previewCtx.fillStyle = "#91a4bf";
    previewCtx.font = "12px Inter, sans-serif";
    previewCtx.fillText(text, 12, Math.max(28, r.height / 2));
    return;
  }
  const frame = state.previewFrames[Math.max(0, Math.min(state.previewIndex, state.previewFrames.length - 1))];
  const maxAbs = previewBoundsMaxAbs();
  const boxes = [...frame.boxes].sort((a, b) => (String(a.source) === "GT" ? -1 : 1) - (String(b.source) === "GT" ? -1 : 1));
  if (state.compare) {
    const vps = previewViewportsForRect(r);
    previewCtx.fillStyle = "rgba(226,232,240,.34)";
    previewCtx.fillRect(vps[0].w, 0, 3, r.height);
    drawPreviewScene(frame, boxes.filter(b => b.run === "A"), vps[0], "A", maxAbs);
    drawPreviewScene(frame, boxes.filter(b => b.run === "B"), vps[1], "B", maxAbs);
  } else {
    drawPreviewScene(frame, boxes, {x: 0, y: 0, w: r.width, h: r.height}, "", maxAbs);
  }
  previewCtx.fillStyle = "#eaf2ff";
  previewCtx.font = "800 11px Inter, sans-serif";
  previewCtx.fillText(`Frame ${frame.frame}`, state.compare ? Math.max(88, r.width / 2 - 36) : 10, 16);
  previewCtx.fillStyle = "#91a4bf";
  previewCtx.font = "700 10px Inter, sans-serif";
  previewCtx.fillText(state.compare ? `A vs B compare · ${compareLensLabel()}` : "GT green · TP cyan · FP red · FN amber", 10, 31);
  const aCount = boxes.filter(b => b.run === "A").length;
  const bCount = boxes.filter(b => b.run === "B").length;
  els.previewStatus.textContent = message || (state.compare
    ? `${state.previewIndex + 1}/${state.previewFrames.length} frames · frame ${frame.frame} · A ${aCount.toLocaleString()} / B ${bCount.toLocaleString()} boxes · drag pan / wheel zoom`
    : `${state.previewIndex + 1}/${state.previewFrames.length} frames · frame ${frame.frame} · ${boxes.length.toLocaleString()} boxes · drag pan / wheel zoom`);
  els.previewSlider.max = String(Math.max(0, state.previewFrames.length - 1));
  els.previewSlider.value = String(state.previewIndex);
  updatePreviewHover();
  drawPreviewHoverLabel();
}
