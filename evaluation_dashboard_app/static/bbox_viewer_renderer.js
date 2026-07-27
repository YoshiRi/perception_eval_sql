function drawGrid() {
  const m = Math.ceil(state.bounds.maxAbs / 10) * 10;
  ctx.lineWidth = 1;
  if (els.showRings.checked) drawRangeRings(m);
  const xMin = Math.floor((state.panX - m) / 10) * 10;
  const xMax = Math.ceil((state.panX + m) / 10) * 10;
  const yMin = Math.floor((state.panY - m) / 10) * 10;
  const yMax = Math.ceil((state.panY + m) / 10) * 10;
  for (let i = xMin; i <= xMax; i += 10) {
    ctx.strokeStyle = i === 0 ? "rgba(226,232,240,.34)" : "rgba(148,163,184,.14)";
    const a = project([i, yMin, 0]), b = project([i, yMax, 0]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  for (let i = yMin; i <= yMax; i += 10) {
    ctx.strokeStyle = i === 0 ? "rgba(226,232,240,.34)" : "rgba(148,163,184,.14)";
    const a = project([xMin, i, 0]), b = project([xMax, i, 0]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  drawEgoGlyph();
}
function drawRangeRings(maxRange) {
  ctx.save();
  ctx.strokeStyle = "rgba(56,189,248,.18)";
  ctx.fillStyle = "rgba(148,163,184,.62)";
  ctx.font = "11px Inter, sans-serif";
  for (let r = 20; r <= maxRange; r += 20) {
    if (els.viewMode.value === "bev") {
      const c = project([0, 0, 0]);
      ctx.beginPath();
      ctx.arc(c[0], c[1], r * c[2], 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillText(`${r}m`, c[0] + 4, c[1] - r * c[2] - 4);
    } else {
      let first = true;
      ctx.beginPath();
      for (let i = 0; i <= 96; i++) {
        const a = (i / 96) * Math.PI * 2;
        const p = project([Math.cos(a) * r, Math.sin(a) * r, 0]);
        if (first) { ctx.moveTo(p[0], p[1]); first = false; }
        else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
      const label = project([r, 0, 0]);
      ctx.fillText(`${r}m`, label[0] + 4, label[1] - 4);
    }
  }
  ctx.restore();
}
function drawEgoGlyph() {
  const car = [[2.35, .98, .05], [1.55, 1.08, .05], [-2.25, .92, .05], [-2.45, -.92, .05], [1.55, -1.08, .05], [2.35, -.98, .05]].map(project);
  const cabin = [[.8, .64, .18], [-.85, .58, .18], [-.85, -.58, .18], [.8, -.64, .18]].map(project);
  ctx.fillStyle = "rgba(15,23,42,.88)";
  ctx.strokeStyle = "rgba(226,232,240,.78)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(car[0][0], car[0][1]);
  for (let i = 1; i < car.length; i++) ctx.lineTo(car[i][0], car[i][1]);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "rgba(56,189,248,.22)";
  ctx.strokeStyle = "rgba(56,189,248,.75)";
  ctx.beginPath();
  ctx.moveTo(cabin[0][0], cabin[0][1]);
  for (let i = 1; i < cabin.length; i++) ctx.lineTo(cabin[i][0], cabin[i][1]);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  const p = project([0, 0, .25]);
  const forward = project([7.5, 0, .25]);
  const left = project([0, 4.5, .25]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(56,189,248,.95)";
  ctx.fillStyle = "rgba(56,189,248,.95)";
  ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(forward[0], forward[1]); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(left[0], left[1]); ctx.stroke();
  ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill();
  ctx.font = "700 11px Inter, sans-serif";
  ctx.fillText("+X", forward[0] + 5, forward[1]);
  ctx.fillText("+Y", left[0] + 5, left[1]);
}
function boxColor(b) {
  if (els.colorMode.value === "run") {
    if (b.run === "A") return b.status === "FP" ? "#f59e0b" : "#60a5fa";
    if (b.run === "B") return b.status === "FP" ? "#ef4444" : "#a78bfa";
  }
  if (els.colorMode.value === "source") return b.source === "GT" ? "#60a5fa" : "#fb7185";
  if (els.colorMode.value === "confidence") {
    const c = Math.max(0, Math.min(1, Number(b.confidence ?? (b.source === "GT" ? 1 : 0))));
    const r = Math.round(255 * (1 - c));
    const g = Math.round(120 + 115 * c);
    return `rgb(${r},${g},110)`;
  }
  if (els.colorMode.value === "error") {
    const e = Number(b.center_distance ?? 0);
    if (b.status !== "TP" || !Number.isFinite(e)) return b.source === "GT" ? "#64748b" : "#94a3b8";
    if (e > 2.0) return "#ef4444";
    if (e > 1.0) return "#f97316";
    if (e > 0.5) return "#facc15";
    return "#34d399";
  }
  return evalLayerColor(b);
}
function evalLayerColor(b) {
  const source = String(b.source || "").toUpperCase();
  const status = String(b.status || "").toUpperCase();
  if (source === "GT" && status === "TP") return "#00cc66";
  if (source === "GT" && status === "FN") return "#ff9933";
  if (source === "EST" && status === "TP") return "#66b3ff";
  if (source === "EST" && status === "FP") return "#ff6666";
  if (source === "GT") return "#4bd08d";
  if (source === "EST") return "#66b3ff";
  if (status === "FN") return "#ff9933";
  if (status === "FP") return "#ff6666";
  if (status === "TP") return "#00cc66";
  return "#94a3b8";
}
function hexToRgb(hex) {
  const m = String(hex || "").replace("#", "").match(/^([0-9a-f]{6})$/i);
  if (!m) return [148, 163, 184];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function labelText(b) {
  if (els.labelMode.value === "none") return "";
  if (els.labelMode.value === "uuid") return (b.uuid || b.pair_uuid || "").slice(0, 8);
  if (els.labelMode.value === "label_conf") {
    const c = b.confidence == null ? "" : ` ${(Number(b.confidence) || 0).toFixed(2)}`;
    return `${state.compare ? `${b.run}:` : ""}${b.label || "box"}${c}`;
  }
  return `${state.compare ? `${b.run}:` : ""}${b.label || "box"} · ${b.status || b.source || ""}`;
}
function drawLabel(b, pts, color, rank, total) {
  const text = labelText(b);
  if (!text || total > 450 || rank > 180) return;
  const top = pts.reduce((best, p) => p[1] < best[1] ? p : best, pts[0]);
  ctx.font = "700 11px Inter, sans-serif";
  const tw = ctx.measureText(text).width;
  const x = top[0] + 5, y = top[1] - 8;
  ctx.fillStyle = "rgba(2,6,23,.76)";
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  roundRect(x - 4, y - 13, tw + 8, 17, 5);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#f8fafc";
  ctx.fillText(text, x, y);
}
function drawFootprintPolygon(b, rank, total) {
  // §8: render base_link footprint vertices as an extruded polygon prism.
  const color = boxColor(b);
  const selected = state.selected && sameObject(state.selected, b);
  const opacity = Math.max(0.2, Math.min(1, Number(els.boxOpacity.value || 0.9)));
  const zBase = b.z || 0;
  const zTop = zBase + Math.max(0.4, Number(b.height || 1.2));
  const baseP = b.footprint.map(pt => project([Number(pt[0]) || 0, Number(pt[1]) || 0, zBase]));
  const topP = b.footprint.map(pt => project([Number(pt[0]) || 0, Number(pt[1]) || 0, zTop]));
  const rgb = hexToRgb(color);
  ctx.save();
  ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(b.status === "FN" ? 0.05 : 0.14) * opacity})`;
  for (const face of [baseP, topP]) {
    ctx.beginPath();
    ctx.moveTo(face[0][0], face[0][1]);
    for (let i = 1; i < face.length; i++) ctx.lineTo(face[i][0], face[i][1]);
    ctx.closePath();
    ctx.fill();
  }
  ctx.strokeStyle = selected ? "#ffffff" : color;
  ctx.lineWidth = selected ? 3.2 : (b.source === "GT" ? 1.35 : 2.1);
  ctx.globalAlpha = (b.source === "GT" ? .76 : .92) * opacity;
  for (const face of [baseP, topP]) {
    ctx.beginPath();
    ctx.moveTo(face[0][0], face[0][1]);
    for (let i = 1; i < face.length; i++) ctx.lineTo(face[i][0], face[i][1]);
    ctx.closePath();
    ctx.stroke();
  }
  for (let i = 0; i < baseP.length; i++) {
    ctx.beginPath();
    ctx.moveTo(baseP[i][0], baseP[i][1]);
    ctx.lineTo(topP[i][0], topP[i][1]);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.restore();
  drawVelocity(b);
  drawErrorGlyph(b);
  drawLabel(b, topP, color, rank, total);
}
function drawPointObject(b, rank, total) {
  if (Array.isArray(b.footprint) && b.footprint.length >= 3) {
    drawFootprintPolygon(b, rank, total);
    return;
  }
  const color = boxColor(b);
  const selected = state.selected && sameObject(state.selected, b);
  const opacity = Math.max(0.2, Math.min(1, Number(els.boxOpacity.value || 0.9)));
  const base = project([b.x || 0, b.y || 0, b.z || 0]);
  const top = project([b.x || 0, b.y || 0, (b.z || 0) + Math.max(.4, (b.height || 1.2) * .5)]);
  const radius = Math.max(4, Math.min(12, base[2] * .45));
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = selected ? "#ffffff" : color;
  ctx.fillStyle = color;
  ctx.lineWidth = selected ? 3 : (b.source === "GT" ? 1.4 : 2.1);
  ctx.beginPath();
  ctx.arc(top[0], top[1], radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = Math.min(1, opacity * .9);
  ctx.beginPath();
  ctx.arc(top[0], top[1], Math.max(2, radius * .34), 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = Math.min(1, opacity * .42);
  ctx.beginPath();
  ctx.moveTo(base[0], base[1]);
  ctx.lineTo(top[0], top[1]);
  ctx.stroke();
  ctx.restore();
  drawVelocity(b);
  drawErrorGlyph(b);
  drawLabel(b, [top], color, rank, total);
}
function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
function drawVelocity(b) {
  if (!els.showVelocity.checked || b.vx == null || b.vy == null) return;
  const speed = Math.hypot(Number(b.vx) || 0, Number(b.vy) || 0);
  if (speed < 0.15) return;
  const a = project([b.x || 0, b.y || 0, (b.z || 0) + Math.max(.2, (b.height || 1.5) * .55)]);
  const scale = Math.min(2.2, Math.max(.55, speed * .35));
  const bpt = project([(b.x || 0) + (Number(b.vx) || 0) * scale, (b.y || 0) + (Number(b.vy) || 0) * scale, (b.z || 0) + Math.max(.2, (b.height || 1.5) * .55)]);
  ctx.strokeStyle = "rgba(125,211,252,.86)";
  ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(bpt[0], bpt[1]); ctx.stroke();
  ctx.fillStyle = "rgba(125,211,252,.86)";
  ctx.beginPath(); ctx.arc(bpt[0], bpt[1], 2.3, 0, Math.PI * 2); ctx.fill();
}
function drawErrorGlyph(b) {
  if (!els.showErrors.checked || b.status !== "TP") return;
  const ex = Number(b.x_error), ey = Number(b.y_error);
  if (!Number.isFinite(ex) || !Number.isFinite(ey)) return;
  const a = project([b.x || 0, b.y || 0, (b.z || 0) + .15]);
  const e = project([(b.x || 0) + ex, (b.y || 0) + ey, (b.z || 0) + .15]);
  const mag = Math.hypot(ex, ey);
  ctx.strokeStyle = mag > 1.0 ? "rgba(239,68,68,.9)" : "rgba(250,204,21,.82)";
  ctx.lineWidth = mag > 1.0 ? 2 : 1.2;
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(e[0], e[1]); ctx.stroke();
}
function objectCenterPoint(b) {
  return [b.x || 0, b.y || 0, (b.z || 0) + Math.max(.2, (b.height || 1.5) * .5)];
}
function drawObjectHalo(b, color = "#ffffff") {
  const p = project(objectCenterPoint(b));
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.2;
  ctx.globalAlpha = .95;
  ctx.beginPath();
  ctx.arc(p[0], p[1], Math.max(7, Math.min(18, p[2] * .8)), 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = .22;
  ctx.beginPath();
  ctx.arc(p[0], p[1], Math.max(11, Math.min(28, p[2] * 1.2)), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}
function drawBox(b, rank, total) {
  if (isPointLikeBox(b)) {
    drawPointObject(b, rank, total);
    return;
  }
  const pts = corners(b).map(project);
  const color = boxColor(b);
  const selected = state.selected && sameObject(state.selected, b);
  const opacity = Math.max(0.2, Math.min(1, Number(els.boxOpacity.value || 0.9)));
  if (String(b.source || "").toUpperCase() === "GT") {
    const rgb = hexToRgb(color);
    const topFace = [0, 1, 2, 3].map(i => pts[i]);
    const bottomFace = [4, 5, 6, 7].map(i => pts[i]);
    ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${b.status === "FN" ? 0.05 * opacity : 0.14 * opacity})`;
    for (const face of [bottomFace, topFace]) {
      ctx.beginPath();
      ctx.moveTo(face[0][0], face[0][1]);
      for (let i = 1; i < face.length; i++) ctx.lineTo(face[i][0], face[i][1]);
      ctx.closePath();
      ctx.fill();
    }
  }
  ctx.strokeStyle = selected ? "#ffffff" : color;
  ctx.lineWidth = b.source === "GT" ? 1.35 : 2.2;
  if (selected) ctx.lineWidth = 3.4;
  ctx.globalAlpha = (b.source === "GT" ? .76 : .92) * opacity;
  for (const e of edges) {
    ctx.beginPath(); ctx.moveTo(pts[e[0]][0], pts[e[0]][1]); ctx.lineTo(pts[e[1]][0], pts[e[1]][1]); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  drawVelocity(b);
  drawErrorGlyph(b);
  drawLabel(b, pts, color, rank, total);
}
function drawTrails(filterFn = null) {
  if (!state.trails) return;
  ctx.fillStyle = "rgba(56,189,248,.18)";
  for (const f of state.frames) for (const b of f.boxes) {
    if (filterFn && !filterFn(b)) continue;
    const p = project([b.x || 0, b.y || 0, b.z || 0]);
    ctx.fillRect(p[0] - 1, p[1] - 1, 2, 2);
  }
}

function drawViewportLabel(text, vp, color) {
  const name = text === "Run A" ? state.runNames.A : text === "Run B" ? state.runNames.B : "";
  const label = name ? `${text}: ${name}` : text;
  ctx.font = "800 12px Inter, sans-serif";
  const maxW = Math.max(64, Math.min(vp.w - 24, 230));
  let drawText = label;
  if (ctx.measureText(drawText).width > maxW - 24) {
    const suffix = "...";
    while (drawText.length > 8 && ctx.measureText(`${drawText}${suffix}`).width > maxW - 24) {
      drawText = drawText.slice(0, -1);
    }
    drawText = `${drawText}${suffix}`;
  }
  const boxW = Math.max(72, Math.min(maxW, ctx.measureText(drawText).width + 24));
  ctx.fillStyle = "rgba(2,6,23,.72)";
  ctx.strokeStyle = "rgba(148,163,184,.3)";
  ctx.lineWidth = 1;
  roundRect(vp.x + 12, vp.y + 12, boxW, 26, 13);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = color;
  ctx.fillText(drawText, vp.x + 24, vp.y + 30);
}
function drawScene(frame, boxes, viewport, label = "") {
  activeViewport = viewport;
  ctx.save();
  ctx.beginPath();
  ctx.rect(viewport.x, viewport.y, viewport.w, viewport.h);
  ctx.clip();
  drawGrid();
  drawTrails(label ? (b => b.run === label) : null);
  const sorted = [...boxes].sort((a, b) => (a.y || 0) - (b.y || 0));
  sorted.forEach((b, i) => drawBox({...b, frame: frame.frame}, i, sorted.length));
  if (state.compare && state.selected) {
    const peer = findComparePeer(frame, state.selected);
    if (state.selected.run === label) drawObjectHalo(state.selected, "#ffffff");
    if (peer && peer.run === label) drawObjectHalo(peer, "#facc15");
  }
  ctx.restore();
  if (label) drawViewportLabel(`Run ${label}`, viewport, label === "A" ? "#60a5fa" : "#a78bfa");
  activeViewport = null;
}
function drawCurtainScene(frame, visible, w, h) {
  const x = Math.max(24, Math.min(w - 24, state.curtainX * w));
  const vp = {x: 0, y: 0, w, h};
  activeViewport = vp;
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, x, h);
  ctx.clip();
  drawGrid();
  drawTrails(b => b.run === "A");
  visible.filter(b => b.run === "A").sort((a, b) => (a.y || 0) - (b.y || 0)).forEach((b, i, arr) => drawBox({...b, frame: frame.frame}, i, arr.length));
  if (state.selected) {
    const peer = findComparePeer(frame, state.selected);
    if (state.selected.run === "A") drawObjectHalo(state.selected, "#ffffff");
    if (peer && peer.run === "A") drawObjectHalo(peer, "#facc15");
  }
  ctx.restore();
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, 0, w - x, h);
  ctx.clip();
  drawGrid();
  drawTrails(b => b.run === "B");
  visible.filter(b => b.run === "B").sort((a, b) => (a.y || 0) - (b.y || 0)).forEach((b, i, arr) => drawBox({...b, frame: frame.frame}, i, arr.length));
  if (state.selected) {
    const peer = findComparePeer(frame, state.selected);
    if (state.selected.run === "B") drawObjectHalo(state.selected, "#ffffff");
    if (peer && peer.run === "B") drawObjectHalo(peer, "#facc15");
  }
  ctx.restore();
  activeViewport = null;
  ctx.fillStyle = "rgba(226,232,240,.88)";
  ctx.fillRect(x - 1, 0, 2, h);
  drawViewportLabel("Run A", {x: 0, y: 0, w: x, h}, "#60a5fa");
  drawViewportLabel("Run B", {x, y: 0, w: w - x, h}, "#a78bfa");
}
function render() {
  resize();
  const w = els.canvas.clientWidth, h = els.canvas.clientHeight;
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#0f172a"); bg.addColorStop(.48, "#0b1120"); bg.addColorStop(1, "#020617");
  ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
  const f = state.frames[state.framePos] || {frame: 0, boxes: []};
  const visible = displayedBoxes(f);
  if (compareSideBySideActive()) {
    const gap = 3;
    const half = (w - gap) / 2;
    const left = {x: 0, y: 0, w: half, h};
    const right = {x: half + gap, y: 0, w: half, h};
    ctx.fillStyle = "rgba(226,232,240,.34)";
    ctx.fillRect(half, 0, gap, h);
    drawScene(f, visible.filter(b => b.run === "A"), left, "A");
    drawScene(f, visible.filter(b => b.run === "B"), right, "B");
  } else if (compareCurtainActive()) {
    drawCurtainScene(f, visible, w, h);
  } else {
    drawScene(f, visible, {x: 0, y: 0, w, h});
  }
  els.curtainHandle.classList.toggle("show", compareCurtainActive());
  els.splitLabelA.classList.toggle("show", compareSideBySideActive());
  els.splitLabelB.classList.toggle("show", compareSideBySideActive());
  els.curtainHandle.style.left = `${Math.max(2, Math.min(98, state.curtainX * 100))}%`;
  els.slider.value = String(state.framePos);
  const m = frameMetrics(f);
  updateAnalysis(f, visible);
  renderLabelBreakdown(f);
  renderFrameCurve();
  renderOverviewMap();
  els.readout.textContent = state.frames.length ? `frame ${f.frame} · ${state.framePos + 1}/${state.frames.length} · ${visible.length}/${f.boxes.length} visible · TP ${m.tp} FP ${m.fp} FN ${m.fn}` : "no scene loaded";
}
