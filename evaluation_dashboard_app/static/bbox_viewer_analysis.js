function updateStats(data = {}) {
  const boxes = state.frames.reduce((n, f) => n + f.boxes.length, 0);
  let gt = 0, est = 0, a = 0, bRun = 0;
  const run = {A: {tp: 0, fp: 0, fn: 0}, B: {tp: 0, fp: 0, fn: 0}};
  for (const f of state.frames) for (const b of f.boxes) {
    const source = String(b.source || "").toUpperCase();
    const status = String(b.status || "").toUpperCase();
    const r = String(b.run || "");
    if (source === "GT") gt++;
    if (source === "EST") est++;
    if (b.run === "A") a++;
    if (b.run === "B") bRun++;
    if (run[r]) {
      if (source === "EST" && status === "TP") run[r].tp++;
      if (source === "EST" && status === "FP") run[r].fp++;
      if (source === "GT" && status === "FN") run[r].fn++;
    }
  }
  if (state.compare) {
    els.boxCount.textContent = `${a.toLocaleString()}/${bRun.toLocaleString()}`;
    els.frameCount.textContent = state.frames.length.toLocaleString();
    els.gtCount.textContent = fmtDelta(run.B.tp - run.A.tp);
    els.estCount.textContent = fmtDelta(run.B.fp - run.A.fp);
    els.boxCount.nextElementSibling.textContent = "boxes A/B";
    els.gtCount.nextElementSibling.textContent = "ΔTP B-A";
    els.estCount.nextElementSibling.textContent = "ΔFP B-A";
    return;
  }
  els.boxCount.textContent = boxes.toLocaleString();
  els.frameCount.textContent = state.frames.length.toLocaleString();
  els.gtCount.textContent = gt.toLocaleString();
  els.estCount.textContent = est.toLocaleString();
  els.boxCount.nextElementSibling.textContent = "boxes";
  els.gtCount.nextElementSibling.textContent = "GT";
  els.estCount.nextElementSibling.textContent = "EST";
}

function frameMetrics(frame) {
  const out = {tp: 0, fp: 0, fn: 0, gt: 0, est: 0, a: 0, b: 0, gtTp: 0, gtFn: 0, estTp: 0, estFp: 0, worst: 0};
  for (const b of (frame && frame.boxes) || []) {
    const source = String(b.source || "").toUpperCase();
    const status = String(b.status || "").toUpperCase();
    if (source === "GT") out.gt++;
    if (source === "EST") out.est++;
    if (b.run === "A") out.a++;
    if (b.run === "B") out.b++;
    if (source === "GT" && status === "TP") out.gtTp++;
    if (source === "GT" && status === "FN") out.gtFn++;
    if (source === "EST" && status === "TP") out.estTp++;
    if (source === "EST" && status === "FP") out.estFp++;
    out.worst = Math.max(out.worst, Number(b.center_distance || 0), Math.abs(Number(b.yaw_error || 0)));
  }
  out.tp = out.estTp;
  out.fp = out.estFp;
  out.fn = out.gtFn;
  return out;
}
function boxDistance(b) {
  return Math.hypot(Number(b.x) || 0, Number(b.y) || 0);
}
function labelFrameRows(frame) {
  const order = ["car", "truck", "bus", "pedestrian", "bicycle", "motorbike", "animal", "unknown"];
  const map = new Map();
  const ensure = label => {
    const key = label || "unknown";
    if (!map.has(key)) map.set(key, {label: key, tp: 0, fp: 0, fn: 0, aTp: 0, aFp: 0, aFn: 0, bTp: 0, bFp: 0, bFn: 0});
    return map.get(key);
  };
  order.forEach(ensure);
  for (const b of (frame && frame.boxes) || []) {
    const row = ensure(String(b.label || "unknown"));
    const source = String(b.source || "").toUpperCase();
    const status = String(b.status || "").toUpperCase();
    const prefix = b.run === "B" ? "b" : "a";
    if (source === "EST" && status === "TP") { row.tp++; row[`${prefix}Tp`]++; }
    if (source === "EST" && status === "FP") { row.fp++; row[`${prefix}Fp`]++; }
    if (source === "GT" && status === "FN") { row.fn++; row[`${prefix}Fn`]++; }
  }
  return [...map.values()].sort((a, b) => {
    const ai = order.indexOf(a.label), bi = order.indexOf(b.label);
    if (ai >= 0 || bi >= 0) return (ai >= 0 ? ai : 999) - (bi >= 0 ? bi : 999);
    return a.label.localeCompare(b.label);
  });
}
function zoneRows(frame) {
  const zones = [
    {label: "<20m", min: 0, max: 20, fp: 0, fn: 0, tp: 0},
    {label: "20-40m", min: 20, max: 40, fp: 0, fn: 0, tp: 0},
    {label: "40m+", min: 40, max: Infinity, fp: 0, fn: 0, tp: 0},
  ];
  for (const b of (frame && frame.boxes) || []) {
    const d = boxDistance(b);
    const z = zones.find(item => d >= item.min && d < item.max) || zones[zones.length - 1];
    const source = String(b.source || "").toUpperCase();
    const status = String(b.status || "").toUpperCase();
    if (source === "EST" && status === "TP") z.tp++;
    if (source === "EST" && status === "FP") z.fp++;
    if (source === "GT" && status === "FN") z.fn++;
  }
  return zones;
}
function renderLabelBreakdown(frame) {
  const rows = labelFrameRows(frame);
  const max = Math.max(1, ...rows.map(r => state.compare
    ? Math.max(Math.abs(r.bFp - r.aFp), Math.abs(r.bFn - r.aFn), Math.abs(r.bTp - r.aTp))
    : r.tp + r.fp + r.fn));
  els.labelFrameMeta.textContent = state.compare ? "B-A current frame" : "current frame";
  els.labelFrameBreakdown.innerHTML = rows.map(r => {
    if (state.compare) {
      const dFp = r.bFp - r.aFp, dFn = r.bFn - r.aFn, dTp = r.bTp - r.aTp;
      const total = Math.max(1, Math.abs(dFp) + Math.abs(dFn) + Math.abs(dTp));
      return `<div class="label-row">
        <strong>${escapeHtml(r.label)}</strong>
        <div class="label-stack">
          <i style="width:${Math.abs(dTp) / total * 100}%;background:${dTp >= 0 ? TH.c("accent") : TH.c("good")}"></i>
          <i style="width:${Math.abs(dFp) / total * 100}%;background:${dFp >= 0 ? TH.c("estFp") : TH.c("good")}"></i>
          <i style="width:${Math.abs(dFn) / total * 100}%;background:${dFn >= 0 ? TH.c("gtFn") : TH.c("good")}"></i>
        </div>
        <span>FP ${fmtDelta(dFp)} FN ${fmtDelta(dFn)}</span>
      </div>`;
    }
    const total = Math.max(1, r.tp + r.fp + r.fn);
    return `<div class="label-row">
      <strong>${escapeHtml(r.label)}</strong>
      <div class="label-stack">
        <i style="width:${r.tp / total * 100}%;background:${TH.c("estTp")}"></i>
        <i style="width:${r.fp / total * 100}%;background:${TH.c("estFp")}"></i>
        <i style="width:${r.fn / total * 100}%;background:${TH.c("gtFn")}"></i>
      </div>
      <span>TP ${r.tp} FP ${r.fp} FN ${r.fn}</span>
    </div>`;
  }).join("");
  els.zoneBreakdown.innerHTML = zoneRows(frame).map(z => `<div class="zone-chip"><b>${escapeHtml(z.label)}</b>FP ${z.fp} · FN ${z.fn} · TP ${z.tp}</div>`).join("");
}
function runMetrics(frame, run) {
  const out = {tp: 0, fp: 0, fn: 0, gt: 0, est: 0, gtTp: 0, gtFn: 0, estTp: 0, estFp: 0, boxes: 0, worst: 0};
  for (const b of (frame && frame.boxes) || []) {
    if (b.run !== run) continue;
    const source = String(b.source || "").toUpperCase();
    const status = String(b.status || "").toUpperCase();
    out.boxes++;
    if (source === "GT") out.gt++;
    if (source === "EST") out.est++;
    if (source === "GT" && status === "TP") out.gtTp++;
    if (source === "GT" && status === "FN") out.gtFn++;
    if (source === "EST" && status === "TP") out.estTp++;
    if (source === "EST" && status === "FP") out.estFp++;
    out.worst = Math.max(out.worst, Number(b.center_distance || 0), Math.abs(Number(b.yaw_error || 0)));
  }
  out.tp = out.estTp;
  out.fp = out.estFp;
  out.fn = out.gtFn;
  return out;
}
function layerCounts(frame) {
  const out = {gtTp: 0, gtFn: 0, estTp: 0, estFp: 0};
  for (const b of (frame && frame.boxes) || []) {
    const source = String(b.source || "").toUpperCase();
    const status = String(b.status || "").toUpperCase();
    if (source === "GT" && status === "TP") out.gtTp++;
    else if (source === "GT" && status === "FN") out.gtFn++;
    else if (source === "EST" && status === "TP") out.estTp++;
    else if (source === "EST" && status === "FP") out.estFp++;
  }
  return out;
}
function rateText(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "-";
}
function fmtDelta(value) {
  const n = Number(value) || 0;
  return n > 0 ? `+${n}` : String(n);
}
function compareFrameSentence(a, b) {
  const dTp = b.tp - a.tp;
  const dFp = b.fp - a.fp;
  const dFn = b.fn - a.fn;
  if (!dTp && !dFp && !dFn) return "No TP/FP/FN count change in this frame.";
  const parts = [`${fmtDelta(dFp)} FP`, `${fmtDelta(dFn)} FN`, `${fmtDelta(dTp)} TP`];
  return `B has ${parts.join(", ")} in this frame.`;
}
function tpErrorDelta(frame) {
  if (!state.compare) return 0;
  const boxes = (frame && frame.boxes) || [];
  const aByLoose = new Map();
  for (const box of boxes) {
    if (box.run === "A" && box.status === "TP") aByLoose.set(compareKey(box, false), box);
  }
  let worst = 0;
  for (const box of boxes) {
    if (box.run !== "B" || box.status !== "TP") continue;
    const peer = aByLoose.get(compareKey(box, false));
    if (!peer) continue;
    const bErr = Number(box.center_distance ?? 0);
    const aErr = Number(peer.center_distance ?? 0);
    if (Number.isFinite(aErr) && Number.isFinite(bErr)) worst = Math.max(worst, bErr - aErr);
  }
  return worst;
}
function hotspotScore(frame, mode = state.hotspotMode) {
  const boxes = (frame && frame.boxes) || [];
  const m = frameMetrics(frame);
  if (state.compare) {
    const a = runMetrics(frame, "A"), b = runMetrics(frame, "B");
    if (mode === "fp") return Math.max(0, b.fp - a.fp) * 10;
    if (mode === "fn") return Math.max(0, b.fn - a.fn) * 10;
    if (mode === "tp_error") return Math.max(0, tpErrorDelta(frame)) * 12;
    return Math.max(0, b.fp - a.fp) * 6 + Math.max(0, b.fn - a.fn) * 8 + Math.max(0, tpErrorDelta(frame)) * 4 + Math.abs(b.tp - a.tp) * 2;
  }
  if (mode === "fp") return m.fp * 10;
  if (mode === "fn") return m.fn * 10;
  if (mode === "ped_fp") return boxes.filter(b => String(b.label || "").toLowerCase().includes("ped") && String(b.source || "").toUpperCase() === "EST" && String(b.status || "").toUpperCase() === "FP").length * 20;
  if (mode === "near") return boxes.filter(b => boxDistance(b) < 30 && ["FP", "FN"].includes(String(b.status || "").toUpperCase())).length * 12;
  if (mode === "tp_error") return m.worst * 12;
  return m.fp * 6 + m.fn * 8 + m.worst * 4;
}
function updateAnalysis(frame, visible) {
  const m = frameMetrics(frame);
  const recall = (m.gtTp + m.fn) > 0 ? m.gtTp / (m.gtTp + m.fn) : NaN;
  const precision = (m.tp + m.fp) > 0 ? m.tp / (m.tp + m.fp) : NaN;
  els.analysisFrame.textContent = state.frames.length ? `frame ${frame.frame}` : "frame -";
  els.analysisTp.nextElementSibling.textContent = state.compare ? "ΔTP B-A" : "TP";
  els.analysisFp.nextElementSibling.textContent = state.compare ? "ΔFP B-A" : "FP";
  els.analysisFn.nextElementSibling.textContent = state.compare ? "ΔFN B-A" : "FN";
  els.analysisErr.nextElementSibling.textContent = state.compare ? "B max err" : "max err";
  els.analysisRecall.nextElementSibling.textContent = state.compare ? "B recall" : "recall";
  els.analysisPrecision.nextElementSibling.textContent = state.compare ? "B precision" : "precision";
  els.analysisGt.nextElementSibling.textContent = state.compare ? "A boxes" : "GT";
  els.analysisEst.nextElementSibling.textContent = state.compare ? "B boxes" : "EST";
  const modeText = {all: "all", fp: "FP", fn: "FN", ped_fp: "ped FP", near: "nearby", tp_error: "TP error"}[state.hotspotMode] || "all";
  els.hotspotModeMeta.textContent = modeText;
  els.analysisFrame.title = `${visible.length}/${(frame.boxes || []).length} boxes visible; metrics use the full current scene frame`;
  const layers = layerCounts(frame);
  els.cntGtTp.textContent = String(layers.gtTp);
  els.cntGtFn.textContent = String(layers.gtFn);
  els.cntEstTp.textContent = String(layers.estTp);
  els.cntEstFp.textContent = String(layers.estFp);
  els.compareSummary.classList.toggle("show", state.compare);
  els.compareFrameNote.classList.toggle("show", state.compare);
  if (state.compare) {
    const a = runMetrics(frame, "A");
    const b = runMetrics(frame, "B");
    const bRecall = (b.gtTp + b.fn) > 0 ? b.gtTp / (b.gtTp + b.fn) : NaN;
    const bPrecision = (b.tp + b.fp) > 0 ? b.tp / (b.tp + b.fp) : NaN;
    els.analysisTp.textContent = fmtDelta(b.tp - a.tp);
    els.analysisFp.textContent = fmtDelta(b.fp - a.fp);
    els.analysisFn.textContent = fmtDelta(b.fn - a.fn);
    els.analysisErr.textContent = b.worst ? b.worst.toFixed(b.worst >= 10 ? 1 : 2) : "-";
    els.analysisRecall.textContent = rateText(bRecall);
    els.analysisPrecision.textContent = rateText(bPrecision);
    els.analysisGt.textContent = String(a.boxes);
    els.analysisEst.textContent = String(b.boxes);
    els.compareAStats.textContent = `TP ${a.tp} · FP ${a.fp} · FN ${a.fn} · R ${rateText((a.gtTp + a.fn) > 0 ? a.gtTp / (a.gtTp + a.fn) : NaN)} · P ${rateText((a.tp + a.fp) > 0 ? a.tp / (a.tp + a.fp) : NaN)}`;
    els.compareBStats.textContent = `TP ${b.tp} · FP ${b.fp} · FN ${b.fn} · R ${rateText(bRecall)} · P ${rateText(bPrecision)}`;
    els.compareDeltaStats.textContent = `ΔTP ${fmtDelta(b.tp - a.tp)} · ΔFP ${fmtDelta(b.fp - a.fp)} · ΔFN ${fmtDelta(b.fn - a.fn)} · boxes ${fmtDelta(b.boxes - a.boxes)}`;
    els.compareFrameNote.textContent = compareFrameSentence(a, b);
  } else {
    els.analysisTp.textContent = String(m.tp);
    els.analysisFp.textContent = String(m.fp);
    els.analysisFn.textContent = String(m.fn);
    els.analysisErr.textContent = m.worst ? m.worst.toFixed(m.worst >= 10 ? 1 : 2) : "-";
    els.analysisRecall.textContent = rateText(recall);
    els.analysisPrecision.textContent = rateText(precision);
    els.analysisGt.textContent = String(m.gt);
    els.analysisEst.textContent = String(m.est);
  }
}
function stepFrame(delta) {
  if (!state.frames.length) return;
  state.framePos = Math.max(0, Math.min(state.frames.length - 1, state.framePos + delta));
  state.playCarryMs = 0;
  state.selected = null;
  updateInspect();
  render();
}
function jumpHotspot(delta) {
  if (!state.frames.length) return;
  const n = state.frames.length;
  for (let offset = 1; offset <= n; offset++) {
    const idx = (state.framePos + delta * offset + n) % n;
    if (hotspotScore(state.frames[idx]) > 0.01) {
      state.framePos = idx;
      state.selected = null;
      updateInspect();
      render();
      return;
    }
  }
  toast("No issue frame in the loaded frame window.");
}
function seekFrameFromClientX(canvas, clientX) {
  if (!state.frames.length) return;
  const rect = canvas.getBoundingClientRect();
  const t = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
  state.framePos = Math.max(0, Math.min(state.frames.length - 1, Math.round(t * (state.frames.length - 1))));
  state.playCarryMs = 0;
  state.selected = null;
  updateInspect();
  render();
}
