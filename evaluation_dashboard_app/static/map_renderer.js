function layoutNodes() {
  const arr = state.scenarios;
  const cellW = 46;
  const cellH = 32;
  const gap = 8;
  if (state.layout === "grid") {
    const cols = Math.ceil(Math.sqrt(Math.max(1, arr.length)));
    arr.forEach((s, i) => {
      s._cluster = cityName(s);
      s._w = cellW;
      s._h = cellH;
      s._x = (i % cols - cols / 2) * (cellW + gap);
      s._y = (Math.floor(i / cols) - Math.ceil(arr.length / cols) / 2) * (cellH + gap);
      s._groupX = null;
    });
    return;
  }

  const groups = new Map();
  arr.forEach(s => {
    const key = clusterKey(s);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(s);
  });
  const keys = [...groups.keys()].sort((a, b) => groups.get(b).length - groups.get(a).length || a.localeCompare(b));
  const clusterPads = [];
  let cursorX = 0;
  keys.forEach((key, gi) => {
    const items = groups.get(key);
    items.sort((a, b) => scenarioMetric(b) - scenarioMetric(a));
    const cols = Math.ceil(Math.sqrt(items.length));
    const rows = Math.ceil(items.length / cols);
    const width = cols * (cellW + gap) - gap + 34;
    const height = rows * (cellH + gap) - gap + 44;
    clusterPads.push({key, items, cols, rows, width, height});
  });
  const colsOfClusters = Math.ceil(Math.sqrt(Math.max(1, clusterPads.length)));
  const rowHeights = [];
  for (let i = 0; i < clusterPads.length; i += colsOfClusters) {
    rowHeights.push(Math.max(...clusterPads.slice(i, i + colsOfClusters).map(c => c.height)));
  }
  let startY = -rowHeights.reduce((n, h) => n + h + 46, -46) / 2;
  clusterPads.forEach((c, gi) => {
    const col = gi % colsOfClusters;
    const row = Math.floor(gi / colsOfClusters);
    if (col === 0) cursorX = -clusterPads.slice(gi, gi + colsOfClusters).reduce((n, p) => n + p.width + 46, -46) / 2;
    const cx = cursorX + c.width / 2;
    const cy = startY + rowHeights[row] / 2;
    cursorX += c.width + 46;
    c.items.forEach((s, i) => {
      const colI = i % c.cols;
      const rowI = Math.floor(i / c.cols);
      s._cluster = c.key;
      s._groupX = cx;
      s._groupY = cy;
      s._clusterW = c.width;
      s._clusterH = c.height;
      s._clusterSize = c.items.length;
      s._w = cellW;
      s._h = cellH;
      s._x = cx - c.width / 2 + 17 + colI * (cellW + gap);
      s._y = cy - c.height / 2 + 28 + rowI * (cellH + gap);
    });
    if (col === colsOfClusters - 1) startY += rowHeights[row] + 46;
  });
}

function resizeCanvas(c, context) {
  const dpr = window.devicePixelRatio || 1;
  const r = c.getBoundingClientRect();
  c.width = Math.max(1, Math.floor(r.width * dpr));
  c.height = Math.max(1, Math.floor(r.height * dpr));
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return r;
}
function colorFor(v, max) {
  const raw = Number(v || 0);
  const t = max > 0 ? Math.max(0, Math.min(1, Math.abs(raw) / max)) : 0;
  if (raw < 0) return `rgba(52,211,153,${.34 + t * .58})`;
  if (t < .35) return `rgba(56,189,248,${.45 + t})`;
  if (t < .68) return `rgba(251,191,36,${.52 + t * .55})`;
  return `rgba(251,113,133,${.58 + t * .42})`;
}
function screenRect(s, rect) {
  const scale = state.scale;
  return {
    x: rect.width / 2 + ((s._x || 0) + state.panX) * scale,
    y: rect.height / 2 + ((s._y || 0) + state.panY) * scale,
    w: (s._w || 46) * scale,
    h: (s._h || 32) * scale,
  };
}
function drawBoardGrid(ctx, rect) {
  const step = 48 * state.scale;
  if (step < 10) return;
  ctx.strokeStyle = "rgba(148,163,184,.08)";
  ctx.lineWidth = 1;
  for (let x = (rect.width / 2 + state.panX * state.scale) % step; x < rect.width; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, rect.height); ctx.stroke();
  }
  for (let y = (rect.height / 2 + state.panY * state.scale) % step; y < rect.height; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(rect.width, y); ctx.stroke();
  }
}
function drawClusterBlocks(ctx, rect, arr) {
  const clusters = new Map();
  arr.forEach(s => {
    if (!s._cluster || s._groupX == null) return;
    const hit = clusters.get(s._cluster) || {x: s._groupX || 0, y: s._groupY || 0, w: s._clusterW || 80, h: s._clusterH || 80, n: 0, max: 0};
    hit.n += 1; hit.max = Math.max(hit.max, scenarioMetric(s));
    clusters.set(s._cluster, hit);
  });
  for (const [name, c] of clusters) {
    const x = rect.width / 2 + (c.x - c.w / 2 + state.panX) * state.scale;
    const y = rect.height / 2 + (c.y - c.h / 2 + state.panY) * state.scale;
    const w = c.w * state.scale;
    const h = c.h * state.scale;
    if (x > rect.width || y > rect.height || x + w < 0 || y + h < 0) continue;
    ctx.fillStyle = "rgba(8, 47, 73, .16)";
    ctx.strokeStyle = "rgba(56,189,248,.24)";
    ctx.lineWidth = 1;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "rgba(234,242,255,.72)";
    ctx.font = "800 11px Inter, sans-serif";
    ctx.fillText(`${name} · ${c.n}`, x + 9, y + 16);
  }
}
function drawHotspotGlyph(ctx, s, rect, max) {
  const v = scenarioMetric(s);
  const selected = state.selected && scenarioKey(state.selected) === scenarioKey(s);
  const fp = state.label ? ((s.labels || []).find(x => x.label === state.label)?.fp || 0) : (s.fp || 0);
  const fn = state.label ? ((s.labels || []).find(x => x.label === state.label)?.fn || 0) : (s.fn || 0);
  const tp = state.label ? ((s.labels || []).find(x => x.label === state.label)?.tp || 0) : (s.tp || 0);
  const total = Math.max(1, fp + fn + tp);
  const severity = Math.max(0, Math.min(1, v / Math.max(1, max)));
  const absSeverity = Math.max(0, Math.min(1, Math.abs(v) / Math.max(1, max)));
  const box = screenRect(s, rect);
  if (box.x > rect.width + 20 || box.y > rect.height + 20 || box.x + box.w < -20 || box.y + box.h < -20) return;
  ctx.fillStyle = colorFor(v, max);
  ctx.globalAlpha = .28 + absSeverity * .62;
  ctx.fillRect(box.x, box.y, box.w, box.h);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = selected ? "#ffffff" : "rgba(226,232,240,.32)";
  ctx.lineWidth = selected ? 2 : 1;
  ctx.strokeRect(box.x, box.y, box.w, box.h);

  const stripH = Math.max(4, Math.min(9, box.h * .22));
  const fpW = box.w * (fp / total);
  const fnW = box.w * (fn / total);
  ctx.fillStyle = "rgba(251,113,133,.88)";
  ctx.fillRect(box.x, box.y + box.h - stripH, fpW, stripH);
  ctx.fillStyle = "rgba(251,191,36,.84)";
  ctx.fillRect(box.x + fpW, box.y + box.h - stripH, fnW, stripH);
  const tpW = box.w * (tp / total);
  ctx.fillStyle = "rgba(56,189,248,.85)";
  ctx.fillRect(box.x, box.y, tpW, Math.max(3, stripH * .55));

  if (selected || box.w > 58) {
    ctx.fillStyle = selected ? "#ffffff" : "rgba(234,242,255,.8)";
    ctx.font = selected ? "800 11px Inter, sans-serif" : "700 9px Inter, sans-serif";
    ctx.fillText(scenarioName(s).slice(0, selected ? 30 : 16), box.x + 5, box.y + Math.min(17, box.h - 8));
  }
  s._sx = box.x + box.w / 2; s._sy = box.y + box.h / 2; s._sr = Math.max(box.w, box.h) / 2;
}
function labelMetricValue(labelRow) {
  if (state.compare) {
    if (state.lens === "changed_only") return Math.abs(labelRow.delta_fp || 0) + Math.abs(labelRow.delta_fn || 0) + Math.abs(labelRow.delta_tp || 0);
    if (state.lens === "delta_fn") return labelRow.delta_fn || 0;
    if (state.lens === "regression") return Math.max(0, labelRow.delta_fp || 0) + Math.max(0, labelRow.delta_fn || 0);
    return labelRow.delta_fp || 0;
  }
  return metricFromRow(labelRow, state.lens);
}
function scenarioLabelMetric(s, label = state.label) {
  if (!label) return scenarioMetric(s);
  return metricFromRow(labelRow(s, label), state.lens);
}
function mapPoint(x, y, rect) {
  return {
    x: rect.width / 2 + (x + state.panX) * state.scale,
    y: rect.height / 2 + (y + state.panY) * state.scale
  };
}
function scenarioBounds(arr) {
  const boxes = arr.map(s => ({x: s._x || 0, y: s._y || 0, w: s._w || 46, h: s._h || 32}));
  if (!boxes.length) return {minX: -120, maxX: 120, minY: -80, maxY: 80};
  return boxes.reduce((b, r) => ({
    minX: Math.min(b.minX, r.x),
    maxX: Math.max(b.maxX, r.x + r.w),
    minY: Math.min(b.minY, r.y),
    maxY: Math.max(b.maxY, r.y + r.h),
  }), {minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity});
}
function drawLabelBubbles(rect, arr) {
  const labels = allLabelNames().map(label => ({label, row: state.labels.find(v => v.label === label) || zeroLabel(label)}));
  const max = Math.max(1, ...labels.map(v => Math.abs(labelMetricValue(v.row))));
  const bounds = scenarioBounds(arr);
  const cx = bounds.maxX + 170;
  const cy = (bounds.minY + bounds.maxY) / 2;
  const ring = Math.max(78, Math.min(150, 34 + labels.length * 8));
  const center = mapPoint(cx, cy, rect);
  ctx.strokeStyle = "rgba(56,189,248,.2)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(center.x, center.y, ring * state.scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "rgba(234,242,255,.78)";
  ctx.font = "800 12px Inter, sans-serif";
  ctx.fillText("labels", center.x - 18, center.y + 4);
  state.labelNodes = [];
  labels.forEach((item, i) => {
    const a = -Math.PI / 2 + i * Math.PI * 2 / Math.max(1, labels.length);
    const v = labelMetricValue(item.row);
    const total = Math.max(1, (item.row.tp || 0) + (item.row.fp || 0) + (item.row.fn || 0));
    const p = mapPoint(cx + Math.cos(a) * ring, cy + Math.sin(a) * ring, rect);
    const radius = Math.max(9, Math.min(28, 10 + Math.sqrt(Math.abs(v) / max) * 20)) * state.scale;
    state.labelNodes.push({label: item.label, x: p.x, y: p.y, r: radius, value: v});
    ctx.fillStyle = colorFor(v, max);
    ctx.globalAlpha = .34 + Math.min(1, Math.abs(v) / max) * .58;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = item.label === state.label ? "#ffffff" : "rgba(226,232,240,.28)";
    ctx.lineWidth = item.label === state.label ? 2 : 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.stroke();
    const fpArc = (item.row.fp || 0) / total * Math.PI * 2;
    const fnArc = (item.row.fn || 0) / total * Math.PI * 2;
    ctx.lineWidth = Math.max(2, radius * .16);
    ctx.strokeStyle = "#fb7185";
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius + 3, -Math.PI / 2, -Math.PI / 2 + fpArc);
    ctx.stroke();
    ctx.strokeStyle = "#fbbf24";
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius + 6, -Math.PI / 2 + fpArc, -Math.PI / 2 + fpArc + fnArc);
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = "#eaf2ff";
    ctx.font = `${item.label === state.label ? "800" : "700"} ${Math.max(9, Math.min(12, radius * .42))}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText(item.label.slice(0, 10), p.x, p.y + radius + 13);
    ctx.fillStyle = "#91a4bf";
    ctx.fillText(state.compare ? fmtDelta(Math.round(v)) : fmt(Math.round(v)), p.x, p.y + radius + 25);
    ctx.textAlign = "left";
  });
}
function drawLabelConnections(rect, arr) {
  if (!state.label) return;
  const source = state.labelNodes.find(n => n.label === state.label);
  if (!source) return;
  const scored = arr
    .map(s => ({s, v: scenarioLabelMetric(s, state.label)}))
    .filter(x => Number.isFinite(x.v) && Math.abs(x.v) > 0)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    .slice(0, 36);
  if (!scored.length) return;
  const max = Math.max(1, ...scored.map(x => Math.abs(x.v)));
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  scored.forEach(({s, v}, i) => {
    const box = screenRect(s, rect);
    const tx = box.x + box.w / 2;
    const ty = box.y + box.h / 2;
    if (tx < -80 || ty < -80 || tx > rect.width + 80 || ty > rect.height + 80) return;
    const t = Math.abs(v) / max;
    const midX = (source.x + tx) / 2;
    const midY = (source.y + ty) / 2 - Math.min(80, 18 + t * 54);
    ctx.strokeStyle = v < 0 ? `rgba(52,211,153,${.16 + t * .58})` : `rgba(56,189,248,${.14 + t * .5})`;
    ctx.lineWidth = Math.max(1, 1 + t * 4);
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.quadraticCurveTo(midX, midY, tx, ty);
    ctx.stroke();
    if (i < 10) {
      ctx.fillStyle = v < 0 ? "rgba(52,211,153,.9)" : "rgba(234,242,255,.82)";
      ctx.beginPath();
      ctx.arc(tx, ty, Math.max(2.5, 2 + t * 4), 0, Math.PI * 2);
      ctx.fill();
    }
  });
  ctx.restore();
}
function aggregateLabelsFor(arr) {
  const rows = allLabelNames().map(zeroLabel);
  const map = new Map(rows.map(r => [r.label, r]));
  arr.forEach(s => {
    labelsForScenario(s).forEach(x => {
      const r = map.get(x.label) || zeroLabel(x.label);
      r.tp += Number(x.tp) || 0;
      r.fp += Number(x.fp) || 0;
      r.fn += Number(x.fn) || 0;
      r.rows += Number(x.rows) || 0;
      r.delta_tp += Number(x.delta_tp) || 0;
      r.delta_fp += Number(x.delta_fp) || 0;
      r.delta_fn += Number(x.delta_fn) || 0;
      map.set(x.label, r);
    });
  });
  return [...map.values()];
}

function render() {
  els.stage.classList.toggle("stats-mode", state.stageView === "stats");
  const r = resizeCanvas(els.canvas, ctx);
  ctx.clearRect(0, 0, r.width, r.height);
  const bg = ctx.createLinearGradient(0, 0, r.width, r.height);
  bg.addColorStop(0, "#07111f"); bg.addColorStop(.55, "#050812"); bg.addColorStop(1, "#020617");
  ctx.fillStyle = bg; ctx.fillRect(0, 0, r.width, r.height);
  if (state.stageView === "stats") {
    els.legend.style.display = "none";
    renderStatsDashboard(r);
    return;
  }
  els.legend.style.display = "";
  const arr = filteredScenarios();
  const max = Math.max(1, ...arr.map(s => Math.abs(scenarioMetric(s))));
  drawBoardGrid(ctx, r);
  drawClusterBlocks(ctx, r, arr);
  arr.forEach(s => drawHotspotGlyph(ctx, s, r, max));
  drawLabelBubbles(r, arr);
  drawLabelConnections(r, arr);
  drawLabelBubbles(r, arr);
  updateHover(r);
}
function updateStatsHover(rect) {
  let best = null;
  for (const n of state.statNodes || []) {
    const d = Math.hypot(state.mouseX - n.x, state.mouseY - n.y);
    if (d <= n.r) { best = n.s; break; }
  }
  state.hover = best;
  state.hoverLabel = null;
  if (!best) { els.hoverCard.classList.remove("show"); return; }
  const m = scenarioMetric(best);
  els.hoverCard.innerHTML = `<b>${escapeHtml(scenarioName(best))}</b><span>${escapeHtml(best.suite_name || "")} · ${fmt(best.frames)} frames</span><span>${state.compare ? `ΔFP ${fmtDelta(best.delta_fp)} · ΔFN ${fmtDelta(best.delta_fn)}` : `FP ${fmt(best.fp)} · FN ${fmt(best.fn)} · TP ${fmt(best.tp)}`}</span><span>${compareLensLabel()} ${lensMetricText(m)}</span>`;
  els.hoverCard.style.left = `${Math.max(8, Math.min(Math.max(8, rect.width - 340), state.mouseX))}px`;
  els.hoverCard.style.top = `${Math.max(8, Math.min(Math.max(8, rect.height - 120), state.mouseY))}px`;
  els.hoverCard.classList.add("show");
}
function updateHover(rect) {
  let best = null, bestD = 26;
  let bestLabel = null, bestLabelD = 24;
  for (const node of state.labelNodes || []) {
    const d = Math.hypot(state.mouseX - node.x, state.mouseY - node.y);
    if (d < Math.max(bestLabelD, node.r + 10)) { bestLabel = node; bestLabelD = d; }
  }
  state.hoverLabel = bestLabel;
  if (bestLabel) {
    els.hoverCard.innerHTML = `<b>${escapeHtml(bestLabel.label)}</b><span>Click to show scenario contribution links.</span><span>${state.compare ? compareLensLabel() : state.lens.toUpperCase()} ${lensMetricText(bestLabel.value)}</span>`;
    els.hoverCard.style.left = `${Math.max(8, Math.min(Math.max(8, rect.width - 340), state.mouseX))}px`;
    els.hoverCard.style.top = `${Math.max(8, Math.min(Math.max(8, rect.height - 120), state.mouseY))}px`;
    els.hoverCard.classList.add("show");
    state.hover = null;
    return;
  }
  for (const s of filteredScenarios()) {
    if (!Number.isFinite(s._sx) || !Number.isFinite(s._sy)) continue;
    const d = Math.hypot(state.mouseX - s._sx, state.mouseY - s._sy);
    if (d < Math.max(bestD, s._sr || 16)) { best = s; bestD = d; }
  }
  state.hover = best;
  state.hoverLabel = null;
  if (!best) { els.hoverCard.classList.remove("show"); return; }
  const m = scenarioMetric(best);
  els.hoverCard.innerHTML = `<b>${escapeHtml(scenarioName(best))}</b><span>${escapeHtml(best._cluster || "")} · ${fmt(best.frames)} frames</span><span>${compareLensLabel()} ${lensMetricText(m)} · ${state.compare ? `ΔFP ${fmtDelta(best.delta_fp)} · ΔFN ${fmtDelta(best.delta_fn)}` : `FP ${fmt(best.fp)} · FN ${fmt(best.fn)}`}</span><span>${escapeHtml((best.labels || []).slice(0, 3).map(x => state.compare ? `${x.label}:ΔFP${fmtDelta(x.delta_fp || 0)}` : `${x.label}:${x.fp}FP/${x.fn}FN`).join(" · "))}</span>`;
  els.hoverCard.style.left = `${Math.max(8, Math.min(Math.max(8, rect.width - 340), state.mouseX))}px`;
  els.hoverCard.style.top = `${Math.max(8, Math.min(Math.max(8, rect.height - 120), state.mouseY))}px`;
  els.hoverCard.classList.add("show");
}
function renderCurve(message = "") {
  const r = resizeCanvas(els.curve, curveCtx);
  curveCtx.clearRect(0, 0, r.width, r.height);
  curveCtx.fillStyle = "rgba(2,6,23,.72)";
  curveCtx.fillRect(0, 0, r.width, r.height);
  curveCtx.strokeStyle = "rgba(148,163,184,.18)";
  curveCtx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = 24 + i * (r.height - 42) / 3;
    curveCtx.beginPath(); curveCtx.moveTo(8, y); curveCtx.lineTo(r.width - 8, y); curveCtx.stroke();
  }
  if (!state.curve.length) {
    const text = message || "No frames matched the selected label/range.";
    els.curveStatus.textContent = text;
    curveCtx.fillStyle = "#91a4bf";
    curveCtx.font = "12px Inter, sans-serif";
    curveCtx.fillText(text, 12, Math.max(28, r.height / 2));
    return;
  }
  const max = Math.max(1, ...state.curve.map(f => Math.max(Math.abs(f.fp || 0), Math.abs(f.fn || 0), Math.abs(f.tp || 0))));
  const plot = {x: 10, y: 24, w: Math.max(1, r.width - 20), h: Math.max(1, r.height - 42)};
  const barW = Math.max(1, plot.w / Math.max(1, state.curve.length) * .82);
  if (state.compare) {
    const mid = plot.y + plot.h / 2;
    curveCtx.strokeStyle = "rgba(226,232,240,.32)";
    curveCtx.beginPath(); curveCtx.moveTo(plot.x, mid); curveCtx.lineTo(plot.x + plot.w, mid); curveCtx.stroke();
    const drawDeltaBar = (x, value, offset, colorPos, colorNeg) => {
      const h = (Math.abs(Number(value) || 0) / max) * (plot.h / 2);
      curveCtx.fillStyle = value >= 0 ? colorPos : colorNeg;
      curveCtx.fillRect(x + offset - barW / 4, value >= 0 ? mid - h : mid, barW * .28, h);
    };
    state.curve.forEach((f, i) => {
      const x = plot.x + i * plot.w / Math.max(1, state.curve.length - 1);
      drawDeltaBar(x, f.fp || 0, -barW * .18, "rgba(251,113,133,.72)", "rgba(52,211,153,.64)");
      drawDeltaBar(x, f.fn || 0, barW * .18, "rgba(251,191,36,.72)", "rgba(52,211,153,.5)");
    });
    curveCtx.fillStyle = "#fb7185"; curveCtx.font = "700 11px Inter, sans-serif";
    curveCtx.fillText("ΔFP", 10, 16); curveCtx.fillStyle = "#fbbf24"; curveCtx.fillText("ΔFN", 52, 16); curveCtx.fillStyle = "#34d399"; curveCtx.fillText("below = improved", 96, 16);
    const peak = [...state.curve].sort((a, b) => (Math.abs(b.fp || 0) + Math.abs(b.fn || 0)) - (Math.abs(a.fp || 0) + Math.abs(a.fn || 0)))[0];
    els.curveStatus.textContent = `${state.curve.length} frames · largest change frame ${peak ? peak.frame : "-"} · max |delta| ${Math.round(max)}`;
    drawCurveFrameMarker(plot);
    return;
  }
  state.curve.forEach((f, i) => {
    const x = plot.x + i * plot.w / Math.max(1, state.curve.length - 1);
    const fpH = ((Number(f.fp) || 0) / max) * plot.h;
    const fnH = ((Number(f.fn) || 0) / max) * plot.h;
    curveCtx.fillStyle = "rgba(251,113,133,.58)";
    curveCtx.fillRect(x - barW / 2, plot.y + plot.h - fpH, barW * .48, fpH);
    curveCtx.fillStyle = "rgba(251,191,36,.58)";
    curveCtx.fillRect(x, plot.y + plot.h - fnH, barW * .48, fnH);
  });
  const draw = (key, color) => {
    curveCtx.strokeStyle = color; curveCtx.lineWidth = key === "tp" ? 2.4 : 1.6; curveCtx.beginPath();
    state.curve.forEach((f, i) => {
      const x = plot.x + i * plot.w / Math.max(1, state.curve.length - 1);
      const y = plot.y + plot.h - ((Number(f[key]) || 0) / max) * plot.h;
      if (i === 0) curveCtx.moveTo(x, y); else curveCtx.lineTo(x, y);
    });
    curveCtx.stroke();
  };
  draw("tp", "#38bdf8");
  draw("fp", "#fb7185");
  draw("fn", "#fbbf24");
  curveCtx.fillStyle = "#38bdf8"; curveCtx.font = "700 11px Inter, sans-serif";
  curveCtx.fillText("TP", 10, 16); curveCtx.fillStyle = "#fb7185"; curveCtx.fillText("FP", 44, 16); curveCtx.fillStyle = "#fbbf24"; curveCtx.fillText("FN", 78, 16);
  curveCtx.fillStyle = "#91a4bf"; curveCtx.fillText(`max ${Math.round(max)}`, r.width - 68, 16);
  const peak = [...state.curve].sort((a, b) => ((b.fp || 0) + (b.fn || 0)) - ((a.fp || 0) + (a.fn || 0)))[0];
  els.curveStatus.textContent = `${state.curve.length} frames · peak frame ${peak ? peak.frame : "-"} · max ${Math.round(max)} objects/frame`;
  drawCurveFrameMarker(plot);
}
function currentPreviewFrameNumber() {
  return state.previewFrames[state.previewIndex] ? Number(state.previewFrames[state.previewIndex].frame) : null;
}
function curveXForFrame(frame, plot) {
  if (!state.curve.length || frame == null) return null;
  let best = 0, dist = Infinity;
  state.curve.forEach((f, i) => {
    const d = Math.abs(Number(f.frame) - Number(frame));
    if (d < dist) { best = i; dist = d; }
  });
  return plot.x + best * plot.w / Math.max(1, state.curve.length - 1);
}
function drawCurveFrameMarker(plot) {
  const frame = currentPreviewFrameNumber();
  const x = curveXForFrame(frame, plot);
  if (x == null) return;
  curveCtx.strokeStyle = "rgba(255,255,255,.88)";
  curveCtx.lineWidth = 1.5;
  curveCtx.beginPath();
  curveCtx.moveTo(x, plot.y - 4);
  curveCtx.lineTo(x, plot.y + plot.h + 4);
  curveCtx.stroke();
  curveCtx.fillStyle = "#ffffff";
  curveCtx.beginPath();
  curveCtx.arc(x, plot.y - 5, 3, 0, Math.PI * 2);
  curveCtx.fill();
  curveCtx.font = "800 10px Inter, sans-serif";
  curveCtx.textAlign = x > plot.x + plot.w - 74 ? "right" : "left";
  curveCtx.fillText(`frame ${frame}`, x + (curveCtx.textAlign === "right" ? -6 : 6), plot.y + 10);
  curveCtx.textAlign = "left";
}
