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
function setPreviewFramesPreserveFrame(frames, message = "") {
  const currentFrame = state.previewFrames[state.previewIndex] && Number(state.previewFrames[state.previewIndex].frame);
  setPreviewFrames(frames, message);
  if (Number.isFinite(currentFrame) && state.previewFrames.length) {
    let best = 0, dist = Infinity;
    state.previewFrames.forEach((f, i) => {
      const d = Math.abs(Number(f.frame) - currentFrame);
      if (d < dist) { best = i; dist = d; }
    });
    state.previewIndex = best;
    els.previewSlider.value = String(best);
    renderPreview(message);
  }
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
  state.devopsFrameResults = null;
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
      if (requestId !== state.previewRequestId) return;
      setPreviewFrames(data.frames || []);
      if (devopsContext(s).is_devops) {
        (async () => {
          const resultPath = els.parquetB.value || state.pathB || state.path;
          try {
            const tn = await api("/api/scenario_devops_tn_objects", {
              ...request,
              path: resultPath,
              run: "B",
              max_rows: 70000,
              timeout_ms: 8000,
            });
            if (requestId !== state.previewRequestId) return;
            setPreviewFramesPreserveFrame(mergePreviewFrames(state.previewFrames || [], tn.frames || []));
            if (typeof renderResultPanel === "function") renderResultPanel();
          } catch (err) {
            if (requestId !== state.previewRequestId) return;
            console.warn("DevOps Run B TN objects unavailable", err);
          }
          try {
            const frameResult = await api("/api/scenario_devops_frame_results", {
              ...request,
              path: resultPath,
              max_frames: 5000,
              timeout_ms: 8000,
            });
            if (requestId !== state.previewRequestId) return;
            state.devopsFrameResults = frameResult;
          } catch (err) {
            if (requestId !== state.previewRequestId) return;
            state.devopsFrameResults = {available: false, reason: err.message, frames: []};
            console.warn("DevOps Run B frame judgement unavailable", err);
          }
          if (requestId !== state.previewRequestId) return;
          renderPreview();
          if (typeof renderResultPanel === "function") renderResultPanel();
        })();
        return;
      }
    } else {
      data = await api("/api/frames", { ...request, path: state.path, run: "A" });
      if (requestId !== state.previewRequestId) return;
      setPreviewFrames(data.frames || []);
      if (devopsContext(s).is_devops) {
        (async () => {
          try {
            const tn = await api("/api/scenario_devops_tn_objects", {
              ...request,
              path: state.path,
              run: "A",
              max_rows: 70000,
              timeout_ms: 8000,
            });
            if (requestId !== state.previewRequestId) return;
            setPreviewFramesPreserveFrame(mergePreviewFrames(state.previewFrames || [], tn.frames || []));
            if (typeof renderResultPanel === "function") renderResultPanel();
          } catch (err) {
            if (requestId !== state.previewRequestId) return;
            console.warn("DevOps TN objects unavailable", err);
          }
          try {
            const frameResult = await api("/api/scenario_devops_frame_results", {
              ...request,
              path: state.path,
              max_frames: 5000,
              timeout_ms: 8000,
            });
            if (requestId !== state.previewRequestId) return;
            state.devopsFrameResults = frameResult;
          } catch (err) {
            if (requestId !== state.previewRequestId) return;
            state.devopsFrameResults = {available: false, reason: err.message, frames: []};
            console.warn("DevOps frame judgement unavailable", err);
          }
          if (requestId !== state.previewRequestId) return;
          renderPreview();
          if (typeof renderResultPanel === "function") renderResultPanel();
        })();
        return;
      }
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
  if (!state.previewFrames.length) return;
  if (!state.curve.length) return;
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
  if (source === "GT" && status === "FN") return TH.c("warn");
  if (source === "GT") return TH.c("good");
  if (status === "FP") return TH.c("bad");
  if (status === "TP") return TH.c("accent");
  return TH.c("mutedBright");
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
function previewIsDevops() {
  return !!(state.selected && devopsContext(state.selected).is_devops);
}
function previewTargetAliases() {
  if (!previewIsDevops()) return [];
  const ctx = devopsContext(state.selected);
  return targetLabelAliases(ctx.target_label, ctx).map(x => String(x).toLowerCase());
}
function previewBoxMatchesDevopsTarget(b) {
  const aliases = previewTargetAliases();
  if (!aliases.length) return false;
  return aliases.includes(String(b.label || "").toLowerCase());
}
function previewDevopsEvidenceMode() {
  if (!previewIsDevops()) return "";
  const gates = (state.devopsResult && state.devopsResult.gates) || [];
  const supported = gates.filter(g => g && g.passed !== null);
  const basis = supported.length ? supported : gates;
  if (basis.some(g => String(g.evaluation_task || "").toLowerCase() === "fp_validation")) return "fp";
  if (basis.some(g => ["num_gt_tp", "num_tp"].includes(String(g.method || "").toLowerCase()))) return "recall";
  if (basis.some(g => String(g.method || "").toLowerCase().includes("yaw"))) return "error";
  const ctx = devopsContext(state.selected);
  if (ctx.focus_metric === "fn") return "recall";
  if (ctx.focus_metric === "fp") return "fp";
  if (ctx.focus_metric === "error") return "error";
  return "";
}
function previewBoxMatchesDevopsCriterionTarget(b) {
  const mode = previewDevopsEvidenceMode();
  if (mode !== "fp") return previewBoxMatchesDevopsTarget(b);
  if (previewBoxMatchesDevopsTarget(b)) return true;
  const ctx = previewIsDevops() ? devopsContext(state.selected) : {};
  const label = String(b.label || "").toLowerCase();
  const source = String(b.source || "").toUpperCase();
  const status = String(b.status || "").toUpperCase();
  return ctx.focus_metric === "fp" && (
    (source === "GT" && label === "false_positive") ||
    (source === "EST" && status === "FP")
  );
}
function previewBoxIsEvidence(b) {
  if (!previewIsDevops()) return false;
  const mode = previewDevopsEvidenceMode();
  const status = String(b.status || "").toUpperCase();
  const source = String(b.source || "").toUpperCase();
  if (mode === "recall") return previewBoxMatchesDevopsTarget(b) && source === "GT" && status === "FN";
  if (mode === "fp") return source === "EST" && status === "FP";
  if (mode === "error") return previewBoxMatchesDevopsTarget(b) && status === "TP";
  return false;
}
function previewBoxIsCorrectTarget(b) {
  const status = String(b.status || "").toUpperCase();
  if (status === "TN") return previewBoxMatchesDevopsCriterionTarget(b);
  return status === "TP" && previewBoxMatchesDevopsTarget(b);
}
function previewBoxDevopsAnnotation(b) {
  if (previewBoxIsEvidence(b)) {
    const source = String(b.source || "").toUpperCase();
    const status = String(b.status || "").toUpperCase();
    if (source === "GT" && status === "FN") return {kind: "fail", title: "MISSED TARGET"};
    if (source === "EST" && status === "FP") return {kind: "fail", title: "FALSE POSITIVE"};
    return {kind: "fail", title: "CRITERION FAIL"};
  }
  if (previewBoxIsCorrectTarget(b)) return {kind: "ok", title: "TARGET OK"};
  return null;
}
function previewObjectPairKey(b) {
  const uuid = String(b.uuid || "").trim();
  const pair = String(b.pair_uuid || "").trim();
  if (uuid && pair) return [uuid, pair].sort().join("|");
  if (pair) return `pair:${pair}`;
  return "";
}
function previewMergedDevopsAnnotations(boxes) {
  const raw = boxes
    .map(b => ({box: b, ann: previewBoxDevopsAnnotation(b)}))
    .filter(x => x.ann && previewLayerVisible(x.box));
  const out = raw.filter(x => x.ann.kind === "fail");
  const ok = raw.filter(x => x.ann.kind === "ok");
  const used = new Set();
  const keyed = new Map();
  ok.forEach((x, i) => {
    const key = previewObjectPairKey(x.box);
    if (!key) return;
    if (!keyed.has(key)) keyed.set(key, []);
    keyed.get(key).push({...x, index: i});
  });
  keyed.forEach(group => {
    if (group.length < 2) return;
    group.forEach(x => used.add(x.index));
    const est = group.find(x => String(x.box.source || "").toUpperCase() === "EST");
    const gt = group.find(x => String(x.box.source || "").toUpperCase() === "GT");
    out.push({box: (est || gt || group[0]).box, ann: {kind: "ok", title: "TARGET OK", merged: group.length > 1}});
  });
  ok.forEach((x, i) => {
    if (used.has(i)) return;
    const source = String(x.box.source || "").toUpperCase();
    let pairIndex = -1;
    for (let j = i + 1; j < ok.length; j += 1) {
      if (used.has(j)) continue;
      const other = ok[j];
      if (String(other.box.source || "").toUpperCase() === source) continue;
      if (Math.hypot((Number(x.box.x) || 0) - (Number(other.box.x) || 0), (Number(x.box.y) || 0) - (Number(other.box.y) || 0)) < 2.5) {
        pairIndex = j;
        break;
      }
    }
    if (pairIndex >= 0) {
      used.add(i);
      used.add(pairIndex);
      const est = source === "EST" ? x : ok[pairIndex];
      out.push({box: est.box, ann: {kind: "ok", title: "TARGET OK", merged: true}});
    } else {
      used.add(i);
      out.push(x);
    }
  });
  return out.sort((a, b) => (a.ann.kind === "fail" ? -1 : 1) - (b.ann.kind === "fail" ? -1 : 1));
}
function previewDistanceFromEgo(b) {
  const x = Number(b.x) || 0;
  const y = Number(b.y) || 0;
  return Math.hypot(x, y);
}
function devopsGateDistanceBounds(g) {
  const distance = g && g.filter && g.filter.Distance;
  if (typeof distance === "string") {
    const numsFromFilter = distance.match(/-?\d+(?:\.\d+)?/g)?.map(Number) || [];
    if (numsFromFilter.length >= 2) return {min: numsFromFilter[0], max: numsFromFilter[1]};
    if (numsFromFilter.length === 1 && /-$/.test(distance.trim())) return {min: numsFromFilter[0], max: null};
  }
  if (Array.isArray(distance) && distance.length >= 2) return {min: Number(distance[0]), max: Number(distance[1])};
  const raw = String(g && g.distance_label || "");
  const nums = raw.match(/-?\d+(?:\.\d+)?/g)?.map(Number) || [];
  if (/^>=/.test(raw) && nums.length) return {min: nums[0], max: null};
  if (nums.length >= 2) return {min: nums[0], max: nums[1]};
  if (nums.length === 1 && /</.test(raw)) return {min: 0, max: nums[0]};
  return null;
}
function devopsGateRegionBounds(g) {
  const region = g && g.filter && g.filter.Region;
  if (!region || typeof region !== "object") return null;
  const parseAxis = value => {
    if (Array.isArray(value) && value.length >= 2) return [Number(value[0]), Number(value[1])];
    const nums = String(value || "").match(/-?\d+(?:\.\d+)?/g)?.map(Number) || [];
    return nums.length >= 2 ? [nums[0], nums[1]] : null;
  };
  const xr = parseAxis(region.x_position);
  const yr = parseAxis(region.y_position);
  if (!xr && !yr) return null;
  return {xMin: xr ? xr[0] : -PREVIEW_MAX_VIEW_EXTENT, xMax: xr ? xr[1] : PREVIEW_MAX_VIEW_EXTENT, yMin: yr ? yr[0] : -PREVIEW_MAX_VIEW_EXTENT, yMax: yr ? yr[1] : PREVIEW_MAX_VIEW_EXTENT};
}
function previewBoxInGate(b, g) {
  const region = devopsGateRegionBounds(g);
  if (region) {
    const x = Number(b.x) || 0;
    const y = Number(b.y) || 0;
    return x >= region.xMin && x <= region.xMax && y >= region.yMin && y <= region.yMax;
  }
  const bounds = devopsGateDistanceBounds(g);
  if (!bounds) return true;
  const d = previewDistanceFromEgo(b);
  if (Number.isFinite(bounds.min) && d < bounds.min) return false;
  if (Number.isFinite(bounds.max) && d >= bounds.max) return false;
  return true;
}
function previewCriteriaLevelValue(g) {
  const raw = String(g.criteria_level || g.level || "").toLowerCase();
  const named = {perfect: 100, hard: 75, normal: 50, easy: 25};
  if (raw in named) return named[raw];
  const fromApi = Number(g.criteria_level_value);
  if (Number.isFinite(fromApi)) return fromApi;
  const numeric = Number(g.level);
  return Number.isFinite(numeric) ? numeric : null;
}
function previewFrameGateEvidence(boxes, g) {
  const scoped = boxes.filter(b => previewBoxMatchesDevopsCriterionTarget(b) && previewBoxInGate(b, g));
  const source = b => String(b.source || "").toUpperCase();
  const status = b => String(b.status || "").toUpperCase();
  const label = b => String(b.label || "").toLowerCase();
  const gtTp = scoped.filter(b => source(b) === "GT" && status(b) === "TP").length;
  const gtFpValidation = scoped.filter(b => source(b) === "GT" && status(b) === "FP" && label(b) === "false_positive").length;
  const gtFn = scoped.filter(b => source(b) === "GT" && status(b) === "FN").length;
  const gtTn = scoped.filter(b => source(b) === "GT" && status(b) === "TN").length;
  const estFp = scoped.filter(b => source(b) === "EST" && status(b) === "FP").length;
  const estTp = scoped.filter(b => source(b) === "EST" && status(b) === "TP").length;
  const method = String(g.method || "").toLowerCase();
  const evaluationTask = String(g.evaluation_task || (devopsContext(state.selected).focus_metric === "fp" ? "fp_validation" : "")).toLowerCase();
  let passedCount = 0;
  let totalCount = 0;
  let score = null;
  let failCount = 0;
  let scoreUnit = "%";
  let level = previewCriteriaLevelValue(g);
  if (method === "num_gt_tp") {
    if (evaluationTask === "fp_validation") {
      passedCount = gtTn;
      failCount = Math.max(gtFpValidation, estFp);
      totalCount = passedCount + failCount;
    } else {
      passedCount = gtTp;
      totalCount = gtTp + gtFn;
      failCount = gtFn;
    }
    score = totalCount ? 100 * passedCount / totalCount : null;
    if (level == null) level = 100;
  } else if (method === "num_tp") {
    if (evaluationTask === "fp_validation") {
      passedCount = gtTn;
      failCount = Math.max(gtFpValidation, estFp);
    } else {
      passedCount = gtTp + gtTn;
      failCount = gtFn + estFp;
    }
    totalCount = passedCount + failCount;
    score = totalCount ? 100 * passedCount / totalCount : null;
    if (level == null) level = 100;
  } else if (method === "yaw_error") {
    const yawBoxes = scoped.filter(b => source(b) === "EST" && status(b) === "TP" && Number.isFinite(Number(b.yaw_error)));
    if (level == null) level = 0;
    passedCount = yawBoxes.filter(b => Math.abs(Number(b.yaw_error)) <= level).length;
    totalCount = yawBoxes.length;
    failCount = Math.max(0, totalCount - passedCount);
    score = totalCount ? yawBoxes.reduce((sum, b) => sum + Math.abs(Number(b.yaw_error)), 0) / totalCount : null;
    scoreUnit = "rad";
  } else {
    passedCount = gtTp;
    failCount = gtFn + estFp;
    totalCount = passedCount + failCount;
    score = totalCount ? 100 * passedCount / totalCount : null;
    if (level == null) level = 100;
  }
  const judged = totalCount > 0 && score != null;
  const isError = method === "yaw_error";
  const pass = judged ? (isError ? score <= level : score >= level) : null;
  return {gtTp, gtFpValidation, gtFn, gtTn, estFp, estTp, passedCount, failCount, totalCount, score, scoreUnit, level, pass, judged};
}
function drawDevopsCriteriaRings(sx, sy, scale, maxAbs) {
  if (!previewIsDevops() || !state.devopsResult) return;
  const gates = (state.devopsResult.gates || []).map(g => ({...g, bounds: devopsGateDistanceBounds(g), region: devopsGateRegionBounds(g)})).filter(g => g.bounds || g.region);
  if (!gates.length) return;
  previewCtx.save();
  gates.forEach(g => {
    if (g.region) {
      const left = sx - (g.region.yMax - state.previewPanY) * scale;
      const right = sx - (g.region.yMin - state.previewPanY) * scale;
      const top = sy - (g.region.xMax - state.previewPanX) * scale;
      const bottom = sy - (g.region.xMin - state.previewPanX) * scale;
      const color = g.passed === false ? TH.a("bad", .62) : TH.a("good", .46);
      previewCtx.strokeStyle = color;
      previewCtx.fillStyle = g.passed === false ? TH.a("bad", .08) : TH.a("good", .05);
      previewCtx.lineWidth = g.passed === false ? 2 : 1.2;
      previewCtx.setLineDash(g.passed === false ? [7, 5] : [3, 6]);
      previewCtx.strokeRect(left, top, right - left, bottom - top);
      previewCtx.fillRect(left, top, right - left, bottom - top);
      previewCtx.setLineDash([]);
      previewCtx.fillStyle = g.passed === false ? TH.c("badFg") : TH.c("goodFg");
      previewCtx.font = "900 10px Inter, sans-serif";
      previewCtx.fillText(`${g.passed === false ? "FAIL" : "PASS"} ${g.distance_label || "region"}`, left + 5, top + 13);
      return;
    }
    const max = g.bounds.max;
    if (!Number.isFinite(max) || max <= 0 || max > maxAbs * 1.2) return;
    previewCtx.strokeStyle = g.passed === false ? TH.a("bad", .62) : TH.a("good", .46);
    previewCtx.fillStyle = g.passed === false ? TH.a("bad", .08) : TH.a("good", .05);
    previewCtx.lineWidth = g.passed === false ? 2 : 1.2;
    previewCtx.setLineDash(g.passed === false ? [7, 5] : [3, 6]);
    previewCtx.beginPath();
    previewCtx.arc(sx, sy, max * scale, 0, Math.PI * 2);
    previewCtx.stroke();
    previewCtx.beginPath();
    previewCtx.arc(sx, sy, max * scale, 0, Math.PI * 2);
    previewCtx.fill();
    previewCtx.setLineDash([]);
    previewCtx.fillStyle = g.passed === false ? TH.c("badFg") : TH.c("goodFg");
    previewCtx.font = "900 10px Inter, sans-serif";
    previewCtx.fillText(`${g.passed === false ? "FAIL" : "PASS"} ${g.distance_label}`, sx + max * scale + 5, sy - 5);
  });
  previewCtx.restore();
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
  previewCtx.fillStyle = TH.a("deep", .84);
  previewCtx.strokeStyle = previewColor(b);
  previewCtx.lineWidth = 1;
  previewCtx.fillRect(x - 5, y - 16, tw + 10, 21);
  previewCtx.strokeRect(x - 5, y - 16, tw + 10, 21);
  previewCtx.fillStyle = TH.c("text");
  previewCtx.fillText(text, x, y);
}
function drawPreviewRings(sx, sy, scale, maxAbs) {
  if (!state.previewShowRings) return;
  previewCtx.save();
  previewCtx.strokeStyle = TH.a("accent", .2);
  previewCtx.fillStyle = TH.a("muted", .76);
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
    previewCtx.fillStyle = TH.a("deep", .72);
    previewCtx.fillRect(x - 3, y - 8, tw + 6, 16);
    previewCtx.strokeStyle = previewColor(b);
    previewCtx.lineWidth = 1;
    previewCtx.strokeRect(x - 3, y - 8, tw + 6, 16);
    previewCtx.fillStyle = TH.c("text");
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
function drawPreviewDevopsHighlights(boxes, sx, sy, scale) {
  if (!previewIsDevops()) return;
  const evidence = previewMergedDevopsAnnotations(boxes).slice(0, 120);
  previewCtx.save();
  evidence.forEach(({box: b, ann}) => {
    const p = previewBoxScreenCenter(b, sx, sy, scale);
    const color = ann.kind === "fail" ? previewColor(b) : TH.c("goodFg");
    const distance = previewDistanceFromEgo(b);
    previewCtx.strokeStyle = color;
    previewCtx.fillStyle = color;
    previewCtx.lineWidth = ann.kind === "fail" ? 2.6 : 1.8;
    previewCtx.setLineDash(ann.kind === "fail" ? [5, 4] : []);
    previewCtx.beginPath();
    previewCtx.arc(p[0], p[1], Math.max(10, Math.min(28, scale * (ann.kind === "fail" ? 1.8 : 1.35))), 0, Math.PI * 2);
    previewCtx.stroke();
    previewCtx.setLineDash([]);
    previewCtx.globalAlpha = ann.kind === "fail" ? .14 : .08;
    previewCtx.beginPath();
    previewCtx.arc(p[0], p[1], Math.max(16, Math.min(42, scale * (ann.kind === "fail" ? 3.2 : 2.3))), 0, Math.PI * 2);
    previewCtx.fill();
    previewCtx.globalAlpha = 1;
    const label = `${ann.title}${ann.merged ? " MATCH" : ""} · ${b.source}/${b.status} ${b.label || ""} ${distance.toFixed(1)}m`;
    previewCtx.font = "900 10px Inter, sans-serif";
    const tw = previewCtx.measureText(label).width;
    previewCtx.fillStyle = ann.kind === "fail" ? TH.a("deep", .86) : TH.a("goodBg", .78);
    previewCtx.fillRect(p[0] + 8, p[1] - 18, tw + 10, 18);
    previewCtx.strokeStyle = color;
    previewCtx.strokeRect(p[0] + 8, p[1] - 18, tw + 10, 18);
    previewCtx.fillStyle = TH.c("text");
    previewCtx.fillText(label, p[0] + 13, p[1] - 5);
  });
  previewCtx.restore();
}
function drawPreviewScene(frame, boxes, viewport, label = "", maxAbs = previewBoundsMaxAbs()) {
  const scale = previewScaleForRect(viewport);
  const sx = viewport.x + viewport.w / 2;
  const sy = viewport.y + viewport.h / 2 + 12;
  previewCtx.save();
  previewCtx.beginPath();
  previewCtx.rect(viewport.x, viewport.y, viewport.w, viewport.h);
  previewCtx.clip();
  previewCtx.strokeStyle = TH.a("line", .13);
  previewCtx.lineWidth = 1;
  for (let m = -Math.ceil(maxAbs / 10) * 10; m <= maxAbs; m += 10) {
    previewCtx.beginPath(); previewCtx.moveTo(sx - (m - state.previewPanY) * scale, viewport.y); previewCtx.lineTo(sx - (m - state.previewPanY) * scale, viewport.y + viewport.h); previewCtx.stroke();
    previewCtx.beginPath(); previewCtx.moveTo(viewport.x, sy - (m - state.previewPanX) * scale); previewCtx.lineTo(viewport.x + viewport.w, sy - (m - state.previewPanX) * scale); previewCtx.stroke();
  }
  const egoScreenX = sx - (0 - state.previewPanY) * scale;
  const egoScreenY = sy - (0 - state.previewPanX) * scale;
  drawPreviewRings(egoScreenX, egoScreenY, scale, maxAbs);
  drawDevopsCriteriaRings(egoScreenX, egoScreenY, scale, maxAbs);
  const egoX = sx - (0 - state.previewPanY) * scale;
  const egoY = sy - (0 - state.previewPanX) * scale;
  previewCtx.fillStyle = TH.a("text", .86);
  previewCtx.beginPath();
  previewCtx.moveTo(egoX, egoY - 9); previewCtx.lineTo(egoX - 6, egoY + 8); previewCtx.lineTo(egoX + 6, egoY + 8); previewCtx.closePath(); previewCtx.fill();
  const sorted = [...boxes].filter(previewLayerVisible).sort((a, b) => (String(a.source) === "GT" ? -1 : 1) - (String(b.source) === "GT" ? -1 : 1));
  sorted.forEach(b => drawPreviewBox(b, sx, sy, scale));
  drawPreviewDevopsHighlights(sorted, sx, sy, scale);
  drawPreviewPersistentLabels(sorted, sx, sy, scale);
  previewCtx.restore();
  if (label) {
    const runName = label === "A" ? shortPathName(state.path) : shortPathName(els.parquetB.value || state.pathB);
    let text = `Run ${label}: ${runName}`;
    previewCtx.fillStyle = TH.a("deep", .72);
    previewCtx.strokeStyle = TH.a("line", .28);
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
    previewCtx.fillStyle = label === "A" ? TH.c("runA") : TH.c("runB");
    previewCtx.fillText(text, viewport.x + 20, viewport.y + 25);
  }
}
function currentFrameDevopsCounts(boxes) {
  const targetBoxes = boxes.filter(previewBoxMatchesDevopsCriterionTarget);
  const targetAnnotations = previewMergedDevopsAnnotations(targetBoxes);
  const count = (source, status) => targetBoxes.filter(b => String(b.source || "").toUpperCase() === source && String(b.status || "").toUpperCase() === status).length;
  return {
    target: targetBoxes.length,
    gtFn: count("GT", "FN"),
    estFp: count("EST", "FP"),
    ok: targetAnnotations.filter(x => x.ann.kind === "ok").length,
    evidence: targetAnnotations.filter(x => x.ann.kind === "fail").length,
  };
}
function renderPreviewDevopsOverlay(frame, boxes) {
  if (!els.previewDevopsOverlay) return;
  if (!previewIsDevops()) {
    els.previewDevopsOverlay.classList.remove("show");
    els.previewDevopsOverlay.innerHTML = "";
    return;
  }
  const ctx = devopsContext(state.selected);
  const result = state.devopsResult;
  const verdict = result ? (result.overall_pass ? "PASS" : "FAIL") : scenarioJudgement(state.selected).label;
  const verdictClass = verdict === "PASS" ? "pass" : "fail";
  const counts = currentFrameDevopsCounts(boxes);
  const exactFrame = typeof exactFrameResultFor === "function" ? exactFrameResultFor(frame) : null;
  const exactLabel = exactFrame
    ? (exactFrame.passed == null ? "Frame Not Judged" : (exactFrame.passed ? "Frame PASS" : "Frame FAIL"))
    : (state.devopsFrameResults && state.devopsFrameResults.available === false ? "Frame Approx" : "");
  const exactClass = exactFrame
    ? (exactFrame.passed == null ? "warn" : (exactFrame.passed ? "" : "bad"))
    : "warn";
  const exactDetail = exactFrame
    ? `${(exactFrame.gates || []).filter(g => g.judged).length}/${(exactFrame.gates || []).length} gates judged`
    : (state.devopsFrameResults && state.devopsFrameResults.reason ? state.devopsFrameResults.reason : "");
  els.previewDevopsOverlay.innerHTML = `
    <div class="preview-evidence-card">
      <div class="preview-evidence-head">
        <strong>${escapeHtml(ctx.purpose || devopsPurposeText(state.selected) || "DevOps scenario")}</strong>
        <i class="preview-verdict ${verdictClass}">${escapeHtml(verdict)}</i>
      </div>
      <div class="preview-evidence-tags">
        ${[ctx.intent_type, ctx.target_label, ctx.behavior, ctx.pc_mode, ctx.city].filter(Boolean).map(x => `<i>${escapeHtml(x)}</i>`).join("")}
      </div>
      <div class="preview-frame-counts">
        <i>frame ${escapeHtml(frame.frame)}</i>
        ${exactLabel ? `<i class="${exactClass}">${escapeHtml(exactLabel)}</i>` : ""}
        <i class="${counts.gtFn ? "warn" : ""}">GT FN ${fmt(counts.gtFn)}</i>
        <i class="${counts.estFp ? "bad" : ""}">EST FP ${fmt(counts.estFp)}</i>
        <i>Target OK ${fmt(counts.ok)}</i>
      </div>
      ${exactDetail ? `<p>${escapeHtml(exactDetail)}</p>` : ""}
    </div>
  `;
  els.previewDevopsOverlay.classList.add("show");
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
  previewCtx.fillStyle = TH.a("deep", .76);
  previewCtx.fillRect(0, 0, r.width, r.height);
  if (!state.previewFrames.length) {
    const text = message || "No preview frames matched.";
    els.previewStatus.textContent = text;
    previewCtx.fillStyle = TH.c("muted");
    previewCtx.font = "12px Inter, sans-serif";
    previewCtx.fillText(text, 12, Math.max(28, r.height / 2));
    renderPreviewDevopsOverlay({frame: "-"}, []);
    return;
  }
  const frame = state.previewFrames[Math.max(0, Math.min(state.previewIndex, state.previewFrames.length - 1))];
  const maxAbs = previewBoundsMaxAbs();
  const boxes = [...frame.boxes].sort((a, b) => (String(a.source) === "GT" ? -1 : 1) - (String(b.source) === "GT" ? -1 : 1));
  if (state.compare) {
    const vps = previewViewportsForRect(r);
    previewCtx.fillStyle = TH.a("lineStrong", .34);
    previewCtx.fillRect(vps[0].w, 0, 3, r.height);
    drawPreviewScene(frame, boxes.filter(b => b.run === "A"), vps[0], "A", maxAbs);
    drawPreviewScene(frame, boxes.filter(b => b.run === "B"), vps[1], "B", maxAbs);
  } else {
    drawPreviewScene(frame, boxes, {x: 0, y: 0, w: r.width, h: r.height}, "", maxAbs);
  }
  previewCtx.fillStyle = TH.c("text");
  previewCtx.font = "800 11px Inter, sans-serif";
  previewCtx.fillText(`Frame ${frame.frame}`, state.compare ? Math.max(88, r.width / 2 - 36) : 10, 16);
  previewCtx.fillStyle = TH.c("muted");
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
  renderPreviewDevopsOverlay(frame, boxes);
}
