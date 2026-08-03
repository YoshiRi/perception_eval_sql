function compareKey(box, statusSensitive = true) {
  const id = box.uuid || box.pair_uuid || "";
  const status = statusSensitive ? `${box.status || ""}:` : "";
  if (id) return `${box.source || ""}:${status}${box.label || ""}:${id}`;
  const x = Math.round((Number(box.x) || 0) * 2) / 2;
  const y = Math.round((Number(box.y) || 0) * 2) / 2;
  const yaw = Math.round((Number(box.yaw) || 0) * 5) / 5;
  return `${box.source || ""}:${status}${box.label || ""}:${x}:${y}:${yaw}`;
}
function chipAllows(root, value, emptyMeansAll) {
  const chips = [...root.querySelectorAll(".chip")];
  const active = chips.filter(el => el.classList.contains("active")).map(el => el.dataset.v);
  if (!chips.length || (!active.length && emptyMeansAll)) return true;
  if (!active.length) return false;
  return active.includes(String(value || ""));
}
function layerActive(layer) {
  const chip = els.layerChips.querySelector(`[data-layer="${layer}"]`);
  return !!(chip && chip.classList.contains("active"));
}
function evalTupleVisible(box) {
  const source = String(box.source || "").toUpperCase();
  const status = String(box.status || "").toUpperCase();
  if (source === "GT" && status === "TP") return layerActive("gt_tp");
  if (source === "GT" && status === "FN") return layerActive("gt_fn");
  if (source === "EST" && status === "TP") return layerActive("est_tp");
  if (source === "EST" && status === "FP") return layerActive("est_fp");
  if (source === "GT") return layerActive("gt_tp") || layerActive("gt_fn");
  if (source === "EST") return layerActive("est_tp") || layerActive("est_fp");
  return true;
}
function visibleByLayerControls(box) {
  if (!evalTupleVisible(box)) return false;
  if (!chipAllows(els.labels, box.label, true)) return false;
  if (els.confMin.value !== "" && box.confidence != null) {
    const minConf = Number(els.confMin.value);
    if (Number.isFinite(minConf) && Number(box.confidence) < minConf) return false;
  }
  return true;
}
function displayedBoxes(frame) {
  const boxes = ((frame && frame.boxes) || []).filter(visibleByLayerControls);
  if (!state.compare) return boxes;
  const lens = els.compareLens.value;
  if (lens === "all") return boxes;
  if (lens === "a_only") return boxes.filter(b => b.run === "A");
  if (lens === "b_only") return boxes.filter(b => b.run === "B");
  const aByStatus = new Map();
  const bByStatus = new Map();
  const aByLoose = new Map();
  const bByLoose = new Map();
  for (const box of boxes) {
    const statusKey = compareKey(box, true);
    const looseKey = compareKey(box, false);
    if (box.run === "A") {
      if (!aByStatus.has(statusKey)) aByStatus.set(statusKey, box);
      if (!aByLoose.has(looseKey)) aByLoose.set(looseKey, box);
    } else if (box.run === "B") {
      if (!bByStatus.has(statusKey)) bByStatus.set(statusKey, box);
      if (!bByLoose.has(looseKey)) bByLoose.set(looseKey, box);
    }
  }
  if (lens === "changed_only") {
    const changed = new Set();
    const mark = box => changed.add(compareKey(box, true));
    for (const box of boxes) {
      const statusKey = compareKey(box, true);
      const looseKey = compareKey(box, false);
      const otherByStatus = box.run === "A" ? bByStatus : aByStatus;
      const otherByLoose = box.run === "A" ? bByLoose : aByLoose;
      if (!otherByStatus.has(statusKey)) mark(box);
      const peer = otherByLoose.get(looseKey);
      if (box.status === "TP" && peer && peer.status === "TP") {
        const boxErr = Number(box.center_distance ?? 0);
        const peerErr = Number(peer.center_distance ?? 0);
        if (Number.isFinite(boxErr) && Number.isFinite(peerErr) && Math.abs(boxErr - peerErr) > 0.25) {
          mark(box);
          mark(peer);
        }
      }
    }
    return boxes.filter(box => changed.has(compareKey(box, true)));
  }
  if (lens === "new_fp_b") return boxes.filter(box => box.run === "B" && box.status === "FP" && !aByStatus.has(compareKey(box, true)));
  if (lens === "resolved_fn_b") return boxes.filter(box => box.run === "A" && box.status === "FN" && !bByStatus.has(compareKey(box, true)));
  if (lens === "worse_tp_b" || lens === "better_tp_b") {
    return boxes.filter(box => {
      if (box.run !== "B" || box.status !== "TP") return false;
      const peer = aByLoose.get(compareKey(box, false));
      if (!peer || peer.status !== "TP") return false;
      const bErr = Number(box.center_distance ?? 0);
      const aErr = Number(peer.center_distance ?? 0);
      if (!Number.isFinite(aErr) || !Number.isFinite(bErr)) return false;
      return lens === "worse_tp_b" ? bErr > aErr + 0.25 : bErr + 0.25 < aErr;
    });
  }
  return boxes;
}
function renderHeatStrip() {
  const c = els.heatCanvas, r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.max(1, Math.floor(r.width * dpr));
  c.height = Math.max(1, Math.floor(r.height * dpr));
  const hctx = c.getContext("2d");
  hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  hctx.clearRect(0,0,r.width,r.height);
  const n = Math.max(1, state.frames.length);
  const maxScore = Math.max(1, ...state.frames.map(f => hotspotScore(f)));
  state.frames.forEach((f, i) => {
    const score = Math.min(1, hotspotScore(f) / maxScore);
    hctx.fillStyle = TH.heat(score);
    hctx.fillRect(i * r.width / n, 0, Math.ceil(r.width / n) + 1, r.height);
  });
}
function resizeAuxCanvas(c, context) {
  const dpr = window.devicePixelRatio || 1;
  const r = c.getBoundingClientRect();
  c.width = Math.max(1, Math.floor(r.width * dpr));
  c.height = Math.max(1, Math.floor(r.height * dpr));
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return r;
}
function renderFrameCurve() {
  const r = resizeAuxCanvas(els.frameCurve, frameCurveCtx);
  frameCurveCtx.clearRect(0, 0, r.width, r.height);
  frameCurveCtx.fillStyle = TH.a("deep", .62);
  frameCurveCtx.fillRect(0, 0, r.width, r.height);
  if (!state.frames.length) {
    frameCurveCtx.fillStyle = TH.c("neutralEst");
    frameCurveCtx.font = "11px Inter, sans-serif";
    frameCurveCtx.fillText("load a scene to inspect FP/FN/error over time", 12, 32);
    return;
  }
  const values = state.frames.map(f => {
    if (state.compare) {
      const a = runMetrics(f, "A"), b = runMetrics(f, "B");
      return {frame: f.frame, fp: b.fp - a.fp, fn: b.fn - a.fn, tp: b.tp - a.tp, err: b.worst};
    }
    const m = frameMetrics(f);
    return {frame: f.frame, fp: m.fp, fn: m.fn, tp: m.tp, err: m.worst};
  });
  const max = Math.max(1, ...values.map(v => Math.max(Math.abs(v.fp), Math.abs(v.fn), Math.abs(v.tp), Math.abs(v.err))));
  const plot = {x: 8, y: 17, w: Math.max(1, r.width - 16), h: Math.max(1, r.height - 24)};
  frameCurveCtx.strokeStyle = TH.a("neutralEst", .16);
  frameCurveCtx.lineWidth = 1;
  for (let i = 0; i < 3; i++) {
    const y = plot.y + i * plot.h / 2;
    frameCurveCtx.beginPath(); frameCurveCtx.moveTo(plot.x, y); frameCurveCtx.lineTo(plot.x + plot.w, y); frameCurveCtx.stroke();
  }
  const xAt = i => plot.x + i * plot.w / Math.max(1, values.length - 1);
  if (state.compare) {
    const mid = plot.y + plot.h / 2;
    frameCurveCtx.strokeStyle = TH.a("lineStrong", .32);
    frameCurveCtx.beginPath(); frameCurveCtx.moveTo(plot.x, mid); frameCurveCtx.lineTo(plot.x + plot.w, mid); frameCurveCtx.stroke();
    values.forEach((v, i) => {
      const x = xAt(i);
      const barW = Math.max(1, plot.w / Math.max(1, values.length) * .35);
      for (const [key, color] of [["fp", TH.c("estFp")], ["fn", TH.c("gtFn")], ["tp", TH.c("estTp")]]) {
        const offset = key === "fp" ? -barW : key === "fn" ? 0 : barW;
        const val = Number(v[key]) || 0;
        const h = Math.abs(val) / max * (plot.h / 2);
        frameCurveCtx.fillStyle = val >= 0 ? color : TH.a("good", .7);
        frameCurveCtx.fillRect(x + offset - barW / 2, val >= 0 ? mid - h : mid, barW, h);
      }
    });
  } else {
    const drawLine = (key, color, width) => {
      frameCurveCtx.strokeStyle = color;
      frameCurveCtx.lineWidth = width;
      frameCurveCtx.beginPath();
      values.forEach((v, i) => {
        const x = xAt(i);
        const y = plot.y + plot.h - ((Number(v[key]) || 0) / max) * plot.h;
        if (i === 0) frameCurveCtx.moveTo(x, y); else frameCurveCtx.lineTo(x, y);
      });
      frameCurveCtx.stroke();
    };
    drawLine("tp", TH.c("estTp"), 1.5);
    drawLine("fp", TH.c("estFp"), 2);
    drawLine("fn", TH.c("gtFn"), 2);
    drawLine("err", TH.c("errLow"), 1.2);
  }
  const currentX = xAt(state.framePos);
  frameCurveCtx.strokeStyle = TH.c("marker");
  frameCurveCtx.lineWidth = 1.4;
  frameCurveCtx.beginPath(); frameCurveCtx.moveTo(currentX, 4); frameCurveCtx.lineTo(currentX, r.height - 4); frameCurveCtx.stroke();
  frameCurveCtx.fillStyle = TH.c("text");
  frameCurveCtx.font = "800 10px Inter, sans-serif";
  frameCurveCtx.fillText(state.compare ? "ΔFP red · ΔFN amber · ΔTP blue · green improves" : "FP red · FN amber · TP blue · error yellow", 10, 12);
  frameCurveCtx.textAlign = currentX > r.width - 70 ? "right" : "left";
  frameCurveCtx.fillText(`f ${values[state.framePos]?.frame ?? "-"}`, currentX + (frameCurveCtx.textAlign === "right" ? -5 : 5), r.height - 7);
  frameCurveCtx.textAlign = "left";
}
function renderOverviewMap() {
  const r = resizeAuxCanvas(els.overview, overviewCtx);
  overviewCtx.clearRect(0, 0, r.width, r.height);
  overviewCtx.fillStyle = TH.a("deep", .72);
  overviewCtx.fillRect(0, 0, r.width, r.height);
  const maxAbs = Math.max(20, state.bounds.maxAbs || 80);
  const pad = 10;
  const scale = Math.min((r.width - pad * 2), (r.height - pad * 2)) / (maxAbs * 2);
  const mapPoint = (x, y) => [r.width / 2 - y * scale, r.height / 2 - x * scale];
  overviewCtx.strokeStyle = TH.a("neutralEst", .16);
  overviewCtx.lineWidth = 1;
  for (let d = 20; d <= maxAbs; d += 20) {
    const c = mapPoint(0, 0);
    overviewCtx.beginPath(); overviewCtx.arc(c[0], c[1], d * scale, 0, Math.PI * 2); overviewCtx.stroke();
  }
  const totalBoxes = state.frames.reduce((n, f) => n + ((f.boxes || []).length), 0);
  const sampleEvery = Math.max(1, Math.ceil(totalBoxes / 25000));
  let seen = 0;
  overviewCtx.fillStyle = TH.a("neutralEst", .24);
  for (const f of state.frames) {
    for (const b of f.boxes || []) {
      seen += 1;
      if (seen % sampleEvery !== 0) continue;
      const p = mapPoint(Number(b.x) || 0, Number(b.y) || 0);
      overviewCtx.fillRect(p[0] - .8, p[1] - .8, 1.6, 1.6);
    }
  }
  const frame = state.frames[state.framePos] || {boxes: []};
  for (const b of displayedBoxes(frame)) {
    const p = mapPoint(Number(b.x) || 0, Number(b.y) || 0);
    overviewCtx.fillStyle = evalLayerColor(b);
    overviewCtx.beginPath(); overviewCtx.arc(p[0], p[1], 2.2, 0, Math.PI * 2); overviewCtx.fill();
  }
  const ego = mapPoint(0, 0);
  overviewCtx.fillStyle = TH.c("text");
  overviewCtx.beginPath(); overviewCtx.arc(ego[0], ego[1], 3, 0, Math.PI * 2); overviewCtx.fill();
  const center = mapPoint(state.panX, state.panY);
  const viewMeters = Math.max(12, state.distance * 1.05);
  overviewCtx.strokeStyle = TH.c("accent");
  overviewCtx.lineWidth = 1.4;
  overviewCtx.strokeRect(center[0] - viewMeters * scale / 2, center[1] - viewMeters * scale / 2, viewMeters * scale, viewMeters * scale);
  overviewCtx.fillStyle = TH.c("neutralEst");
  overviewCtx.font = "800 10px Inter, sans-serif";
  overviewCtx.fillText("overview", 9, 14);
}
function sameObject(a, b) {
  if (!a || !b) return false;
  if (a.run && b.run && a.run !== b.run) return false;
  if (a.uuid && b.uuid && a.uuid === b.uuid && a.source === b.source) return true;
  return a.frame === b.frame && a.source === b.source && a.status === b.status && a.label === b.label && Math.abs(a.x - b.x) < .03 && Math.abs(a.y - b.y) < .03;
}
function findComparePeer(frame, box) {
  if (!state.compare || !box) return null;
  const otherRun = box.run === "A" ? "B" : box.run === "B" ? "A" : "";
  if (!otherRun) return null;
  const boxes = (frame && frame.boxes) || [];
  const id = box.pair_uuid || box.uuid || "";
  if (id) {
    const hit = boxes.find(b => b.run === otherRun && (b.uuid === id || b.pair_uuid === id || (box.uuid && b.uuid === box.uuid)));
    if (hit) return hit;
  }
  let best = null, bestScore = Infinity;
  for (const b of boxes) {
    if (b.run !== otherRun) continue;
    if (String(b.label || "") !== String(box.label || "")) continue;
    if (String(b.source || "") !== String(box.source || "")) continue;
    const d = Math.hypot((Number(b.x) || 0) - (Number(box.x) || 0), (Number(b.y) || 0) - (Number(box.y) || 0));
    const yaw = Math.abs((Number(b.yaw) || 0) - (Number(box.yaw) || 0));
    const score = d + yaw * .3;
    if (score < bestScore) { best = b; bestScore = score; }
  }
  return bestScore < 4 ? best : null;
}
function compareDeltaText(box, peer) {
  if (!box || !peer) return "no peer";
  const dx = (Number(box.x) || 0) - (Number(peer.x) || 0);
  const dy = (Number(box.y) || 0) - (Number(peer.y) || 0);
  const dyaw = (Number(box.yaw) || 0) - (Number(peer.yaw) || 0);
  const dv = Math.hypot((Number(box.vx) || 0) - (Number(peer.vx) || 0), (Number(box.vy) || 0) - (Number(peer.vy) || 0));
  return `${peer.run || "peer"} Δxy ${Math.hypot(dx, dy).toFixed(2)}m · Δyaw ${dyaw.toFixed(2)} · Δv ${dv.toFixed(2)}`;
}
function nearestBox(screenX, screenY) {
  const f = state.frames[state.framePos] || {boxes: []};
  let best = null, bestD = 18;
  const hitView = canvasViewportForPoint(screenX, screenY);
  let vp = hitView.viewport;
  let boxes = displayedBoxes(f);
  if (hitView.run) boxes = boxes.filter(b => b.run === hitView.run);
  activeViewport = vp;
  for (const b of boxes) {
    const p = project([b.x || 0, b.y || 0, (b.z || 0) + (b.height || 1.5) * .5]);
    const d = Math.hypot(p[0] - screenX, p[1] - screenY);
    if (d < bestD) { best = {...b, frame: f.frame}; bestD = d; }
  }
  activeViewport = null;
  return best;
}
function fmt(v, suffix = "", n = 2) {
  const x = Number(v);
  return Number.isFinite(x) ? `${x.toFixed(n)}${suffix}` : "-";
}
function updateInspect() {
  const b = state.selected;
  els.inspect.classList.toggle("show", !!b);
  if (!b) return;
  els.inspectTitle.textContent = `${b.label || "box"} · ${b.status || ""}`;
  els.inspectStatus.textContent = `${b.run ? `${b.run} · ` : ""}${b.status || "-"} / ${b.source || "-"}`;
  els.inspectPos.textContent = `${fmt(b.x,"",2)}, ${fmt(b.y,"",2)}, ${fmt(b.z,"",2)}`;
  els.inspectSize.textContent = isPointLikeBox(b)
    ? `${b.shape_type || b.type || "point"} / ${fmt(b.yaw," rad",2)}`
    : `${fmt(b.length,"",2)} x ${fmt(b.width,"",2)} x ${fmt(b.height,"",2)} / ${fmt(b.yaw," rad",2)}`;
  els.inspectConf.textContent = b.confidence == null ? "-" : fmt(b.confidence,"",3);
  els.inspectDist.textContent = `${fmt(b.center_distance,"m",3)} / ${fmt(b.plane_distance,"m",3)}`;
  els.inspectErr.textContent = `${fmt(b.x_error,"",2)} / ${fmt(b.y_error,"",2)} / ${fmt(b.yaw_error,"",2)}`;
  const frame = state.frames[state.framePos] || {boxes: []};
  els.inspectPeer.textContent = state.compare ? compareDeltaText(b, findComparePeer(frame, b)) : "-";
  els.inspectUuid.textContent = b.uuid || b.pair_uuid || "-";
}
function updateHoverCard(ev) {
  const rect = els.canvas.getBoundingClientRect();
  const b = nearestBox(ev.clientX - rect.left, ev.clientY - rect.top);
  state.hover = b;
  els.hoverCard.classList.toggle("show", !!b);
  if (!b) return;
  els.hoverCard.style.left = `${Math.min(rect.width - 270, Math.max(8, ev.clientX - rect.left + 14))}px`;
  els.hoverCard.style.top = `${Math.min(rect.height - 92, Math.max(8, ev.clientY - rect.top + 14))}px`;
  els.hoverCard.innerHTML = `<b>${escapeHtml(b.label || "box")} · ${escapeHtml(b.status || "")}</b><br>` +
    `${escapeHtml(b.source || "")} · conf ${escapeHtml(b.confidence == null ? "-" : Number(b.confidence).toFixed(3))}<br>` +
    `center ${escapeHtml(fmt(b.center_distance, "m", 3))} · yaw err ${escapeHtml(fmt(b.yaw_error, "", 2))}`;
}
