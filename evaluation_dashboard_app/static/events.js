async function scan() {
  try {
    toast("Scanning bbox parquets...");
    const data = await api("/api/parquets", {root: els.root.value, limit: 2000, bbox_only: true});
    state.parquets = data.items || [];
    const options = state.parquets.map(p => `<option value="${escapeHtml(p.path)}">${escapeHtml(p.display)}</option>`).join("");
    els.parquet.innerHTML = options;
    els.parquetB.innerHTML = options;
    if (!state.parquets.length) { toast("No bbox-compatible parquets found."); return; }
    state.path = els.parquet.value || state.parquets[0].path;
    if (state.parquets[1]) els.parquetB.value = state.parquets[1].path;
    state.pathB = els.parquetB.value || "";
    updateCompareControls();
    await hydrate();
    await loadSummary();
  } catch (err) { toast(err.message); }
}
async function hydrate() {
  state.path = els.parquet.value;
  const [topics] = await Promise.all([api("/api/values", {path: state.path, column: "topic_name", filters: {}}).catch(() => ({values: []}))]);
  els.topic.innerHTML = `<option value="">Any</option>` + (topics.values || []).map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
  if ((topics.values || []).includes("perception.object_recognition.objects")) els.topic.value = "perception.object_recognition.objects";
}
async function loadSummary(options = {}) {
  if (!state.path) return;
  const requestId = ++state.summaryRequestId;
  if (els.compareEnabled.checked && els.parquetB.value === els.parquet.value) {
    const alt = state.parquets.find(p => p.path !== els.parquet.value);
    if (alt) els.parquetB.value = alt.path;
  }
  state.compare = els.compareEnabled.checked && els.parquetB.value && els.parquetB.value !== els.parquet.value;
  if (els.compareEnabled.checked && !state.compare) toast("Choose a different Run B parquet for comparison.");
  if (state.compare && !["changed_only", "delta_fp", "delta_fn", "regression", "fp", "fn", "fpr", "fnr"].includes(state.lens)) setLens("changed_only");
  setBusy(true, state.compare ? "Comparing Run B against Run A..." : "Building dataset bbox summary...");
  try {
    const summaryPayload = {filters: filters(), limit: 1200};
    const [dataA, dataB] = state.compare
      ? await Promise.all([
          api("/api/dataset_summary", {path: state.path, ...summaryPayload}),
          api("/api/dataset_summary", {path: els.parquetB.value, ...summaryPayload})
        ])
      : [await api("/api/dataset_summary", {path: state.path, ...summaryPayload}), null];
    if (requestId !== state.summaryRequestId) return;
    let data = dataA;
    if (state.compare) {
      state.scenarios = mergeSummaries(dataA.items || [], dataB.items || []);
      state.labels = mergeLabelRows(dataA.labels || [], dataB.labels || []);
      data = {items: state.scenarios, labels: state.labels};
    } else {
      state.scenarios = dataA.items || [];
      state.labels = dataA.labels || [];
    }
    buildLabelChips();
    layoutNodes();
    renderList();
    updateKpis();
    updateCompareBanner();
    state.stats = null;
    render();
    if (options.selectTop) {
      state.selected = null;
      state.previewFrames = [];
      state.curve = [];
      state.previewVisible = false;
      els.previewWindow.classList.remove("show");
      els.selTitle.textContent = "No scenario selected";
      els.selMeta.textContent = "Click a scenario to load preview frames.";
      renderCurve("Select a scenario to load the frame curve.");
    } else if (state.selected) {
      state.selected = state.scenarios.find(s => scenarioKey(s) === scenarioKey(state.selected)) || null;
    }
    const ranked = filteredScenarios();
    if (!options.selectTop && (!state.selected || scenarioMetric(state.selected) <= 0) && ranked.length) {
      const best = ranked.find(s => scenarioMetric(s) > 0) || ranked[0];
      selectScenario(best, false);
    } else if (!options.selectTop && state.selected) {
      selectScenario(state.selected, false);
    }
    toast(`${state.compare ? "Compared" : "Loaded"} ${state.scenarios.length} scenarios.`);
    if (state.stageView === "stats") await ensureStatsLoaded(requestId);
  } catch (err) {
    if (requestId === state.summaryRequestId) toast(err.message);
  } finally {
    if (requestId === state.summaryRequestId) setBusy(false);
  }
}
async function loadStats(requestId = state.summaryRequestId) {
  state.stats = null;
  if (!state.path) return;
  try {
    const statsPayload = {filters: filters()};
    if (state.compare) {
      const [dataA, dataB] = await Promise.all([
        api("/api/dataset_stats", {path: state.path, ...statsPayload}),
        api("/api/dataset_stats", {path: els.parquetB.value, ...statsPayload})
      ]);
      if (requestId !== state.summaryRequestId) return;
      state.stats = mergeStats(dataA, dataB);
    } else {
      const dataA = await api("/api/dataset_stats", {path: state.path, ...statsPayload});
      if (requestId !== state.summaryRequestId) return;
      state.stats = dataA;
    }
  } catch (err) {
    state.stats = {error: err.message, distance: [], label_distance: [], labels: [], errors: []};
  }
}
async function ensureStatsLoaded(requestId = state.summaryRequestId) {
  if (state.stats || !state.path) return;
  setBusy(true, "Loading summary charts...");
  try {
    await loadStats(requestId);
    if (requestId === state.summaryRequestId) render();
  } finally {
    if (requestId === state.summaryRequestId) setBusy(false);
  }
}
function updateCompareControls() {
  els.parquetB.disabled = !els.compareEnabled.checked;
  els.lensChips.querySelectorAll(".compare-chip").forEach(chip => chip.classList.toggle("disabled", !els.compareEnabled.checked));
  if (!els.compareEnabled.checked && ["changed_only", "delta_fp", "delta_fn", "regression"].includes(state.lens)) setLens("fp");
  updateCompareBanner();
}
function buildLabelChips() {
  const labels = [...state.labels].sort((a, b) => {
    const bm = state.compare ? Math.abs(b.delta_fp || 0) + Math.abs(b.delta_fn || 0) : (b.fp + b.fn);
    const am = state.compare ? Math.abs(a.delta_fp || 0) + Math.abs(a.delta_fn || 0) : (a.fp + a.fn);
    return bm - am;
  });
  if (state.label && !labels.some(x => x.label === state.label)) {
    const hit = state.labels.find(x => x.label === state.label) || {label: state.label, fp: 0, fn: 0, rows: 0};
    labels.push(hit);
  }
  els.labelChips.innerHTML = `<span class="chip ${state.label === "" ? "active" : ""}" data-label="">All labels</span>` +
    labels.map(x => `<span class="chip ${state.label === x.label ? "active" : ""}" data-label="${escapeHtml(x.label)}">${escapeHtml(x.label)} · ${state.compare ? `ΔFP ${fmtDelta(x.delta_fp || 0)}` : `${fmt(x.fp)} FP`}</span>`).join("");
  els.labelChips.querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", () => {
    applyLabel(chip.dataset.label || "");
  }));
}
function applyLabel(label) {
  state.label = label || "";
  buildLabelChips(); layoutNodes(); renderList(); updateKpis(); render();
  const ranked = filteredScenarios();
  if (ranked.length && (!state.selected || scenarioMetric(state.selected) <= 0)) selectScenario(ranked.find(s => scenarioMetric(s) > 0) || ranked[0]);
  else if (state.selected) {
    renderScenarioLabels(state.selected);
    loadScenarioDetails(state.selected);
  }
}
function filteredScenarios() {
  const q = els.search.value.trim().toLowerCase();
  let arr = state.scenarios;
  if (q) {
    arr = arr.filter(s => [s.suite_name, s.scenario_name, s.t4dataset_name, s.topic_name, ...(s.labels || []).map(x => x.label)].join(" ").toLowerCase().includes(q));
  }
  if (state.compare && state.lens === "changed_only") {
    arr = arr.filter(s => scenarioMetric(s, "changed_only") > 0);
  }
  return [...arr].sort((a, b) => scenarioMetric(b) - scenarioMetric(a));
}

function updateKpis() {
  const arr = filteredScenarios();
  const fp = arr.reduce((n, s) => n + (state.label ? (labelRow(s, state.label)?.[state.compare ? "delta_fp" : "fp"] || 0) : (s[state.compare ? "delta_fp" : "fp"] || 0)), 0);
  const fn = arr.reduce((n, s) => n + (state.label ? (labelRow(s, state.label)?.[state.compare ? "delta_fn" : "fn"] || 0) : (s[state.compare ? "delta_fn" : "fn"] || 0)), 0);
  const max = arr.reduce((m, s) => Math.max(m, Math.abs(scenarioMetric(s))), 0);
  els.kScenario.textContent = fmt(arr.length);
  els.kScenarioLabel.textContent = state.compare && state.lens === "changed_only" ? "changed scenarios" : "scenarios";
  els.kFp.textContent = state.compare ? fmtDelta(fp) : fmt(fp);
  els.kFn.textContent = state.compare ? fmtDelta(fn) : fmt(fn);
  els.kFpLabel.textContent = state.compare ? "ΔFP B-A" : "FP";
  els.kFnLabel.textContent = state.compare ? "ΔFN B-A" : "FN";
  els.hudLens.textContent = compareLensLabel();
  els.hudMax.textContent = lensMetricText(max);
  els.hudLabel.textContent = state.label || "All";
  els.hudRange.textContent = state.rangeMax ? `<${state.rangeMax}m` : "All";
}
function renderList() {
  const arr = filteredScenarios().slice(0, 80);
  els.list.innerHTML = arr.map(s => {
    const active = state.selected && scenarioKey(state.selected) === scenarioKey(s);
    const m = scenarioMetric(s);
    const statLine = state.compare
      ? `${compareLensLabel()} ${lensMetricText(m)} · ΔFP ${fmtDelta(s.delta_fp)} · ΔFN ${fmtDelta(s.delta_fn)} · B FP ${fmt(s.fp)}`
      : `${state.lens.toUpperCase()} ${state.lens.endsWith("r") ? rate(m) : fmt(Math.round(m))} · FP ${fmt(s.fp)} · FN ${fmt(s.fn)} · ${fmt(s.frames)} frames${state.rangeMax ? ` · <${state.rangeMax}m` : ""}`;
    return `<div class="scenario ${active ? "active" : ""}" data-key="${escapeHtml(scenarioKey(s))}">
      <strong>${escapeHtml(scenarioName(s))}</strong>
      <span>${escapeHtml(statLine)}</span>
      <span>${escapeHtml((s.labels || []).slice(0, 4).map(x => state.compare ? `${x.label}:ΔFP${fmtDelta(x.delta_fp || 0)}` : `${x.label}:${x.fp}FP`).join(" · "))}</span>
    </div>`;
  }).join("");
  els.list.querySelectorAll(".scenario").forEach(card => card.addEventListener("click", () => {
    const s = state.scenarios.find(x => scenarioKey(x) === card.dataset.key);
    if (s) selectScenario(s);
  }));
}
async function selectScenario(s, flash = true) {
  state.selected = s;
  const view = activeRunStats(s);
  els.selTitle.textContent = scenarioName(s);
  els.selMeta.textContent = state.compare
    ? `${s.suite_name || ""} · A: ${shortPathName(state.path)} · B: ${shortPathName(els.parquetB.value || state.pathB)} · B ${fmt(view.frames)} frames`
    : `${s.suite_name || ""} · ${s.topic_name || ""} · ${fmt(s.frames)} frames`;
  els.selFp.textContent = state.compare ? fmtDelta(s.delta_fp) : fmt(s.fp);
  els.selFn.textContent = state.compare ? fmtDelta(s.delta_fn) : fmt(s.fn);
  els.selPrecision.textContent = rate(view.precision);
  els.selRecall.textContent = rate(view.recall);
  showPreviewWindow();
  els.previewTitle.textContent = scenarioName(s);
  renderScenarioLabels(s);
  renderList();
  render();
  await loadScenarioDetails(s, flash);
}
async function loadScenarioDetails(s, flash = false) {
  const key = scenarioKey(s);
  await loadPreview(s);
  if (!state.selected || scenarioKey(state.selected) !== key) return;
  await loadCurve(s);
  if (!state.selected || scenarioKey(state.selected) !== key) return;
  focusPreviewOnCurvePeak();
  renderPreview();
  renderCurve();
  if (flash) toast("Scenario selected. Curve loaded.");
}
function renderScenarioLabels(s) {
  const labels = labelsForScenario(s);
  if (!labels.length) {
    els.labelBreakdown.innerHTML = `<div class="scenario"><strong>No labels</strong><span>No label-level rows matched the current filters.</span></div>`;
    return;
  }
  if (state.compare) {
    const maxDelta = Math.max(1, ...labels.map(x => Math.max(Math.abs(x.delta_fp || 0), Math.abs(x.delta_fn || 0))));
    els.labelBreakdown.innerHTML = labels.map(x => {
      const delta = (state.lens === "delta_fn" ? x.delta_fn : x.delta_fp) || 0;
      const width = Math.min(50, Math.abs(delta) / maxDelta * 50);
      const left = delta >= 0 ? 50 : 50 - width;
      const color = delta >= 0 ? "#fb7185" : "#34d399";
      return `<div class="label-viz">
        <strong>${escapeHtml(x.label)}</strong>
        <div class="delta-bar"><i class="delta-fill" style="left:${left}%;width:${width}%;background:${color}"></i></div>
        <div class="nums">ΔFP ${fmtDelta(x.delta_fp || 0)} · ΔFN ${fmtDelta(x.delta_fn || 0)} · ΔTP ${fmtDelta(x.delta_tp || 0)}</div>
        <div class="nums">A FP ${fmt(x.a?.fp)} / FN ${fmt(x.a?.fn)} · B FP ${fmt(x.b?.fp)} / FN ${fmt(x.b?.fn)}</div>
      </div>`;
    }).join("");
    return;
  }
  els.labelBreakdown.innerHTML = labels.map(x => {
    const total = Math.max(1, (x.tp || 0) + (x.fp || 0) + (x.fn || 0));
    return `<div class="label-viz">
      <strong>${escapeHtml(x.label)}</strong>
      <div class="label-bar">
        <i class="bar-tp" style="width:${(x.tp || 0) / total * 100}%"></i>
        <i class="bar-fp" style="width:${(x.fp || 0) / total * 100}%"></i>
        <i class="bar-fn" style="width:${(x.fn || 0) / total * 100}%"></i>
      </div>
      <div class="nums">TP ${fmt(x.tp)} · FP ${fmt(x.fp)} · FN ${fmt(x.fn)} · rows ${fmt(x.rows)}</div>
    </div>`;
  }).join("");
}
async function loadCurve(s) {
  const requestId = ++state.curveRequestId;
  const f = {topic_name: s.topic_name, scenario_name: s.scenario_name};
  if (s.suite_name) f.suite_name = s.suite_name;
  if (state.rangeMax !== "") f.distance_max = Number(state.rangeMax);
  els.curveStatus.textContent = `Loading ${state.compare ? "B-A " : ""}frame curve${state.label ? ` for ${state.label}` : ""}${state.rangeMax ? ` within ${state.rangeMax}m` : ""}...`;
  try {
    const curvePayload = {filters: f, label: state.label, limit: 5000};
    if (state.compare) {
      const dataA = await api("/api/scenario_curve", {path: state.path, ...curvePayload});
      if (requestId !== state.curveRequestId) return;
      const dataB = await api("/api/scenario_curve", {path: els.parquetB.value, ...curvePayload});
      if (requestId !== state.curveRequestId) return;
      const map = new Map();
      for (const a of dataA.frames || []) map.set(Number(a.frame), {a, b: {frame: a.frame, tp: 0, fp: 0, fn: 0}});
      for (const b of dataB.frames || []) {
        const hit = map.get(Number(b.frame)) || {a: {frame: b.frame, tp: 0, fp: 0, fn: 0}, b: null};
        hit.b = b;
        map.set(Number(b.frame), hit);
      }
      state.curve = [...map.values()].sort((x, y) => Number(x.a.frame ?? x.b.frame) - Number(y.a.frame ?? y.b.frame)).map(({a, b}) => ({
        frame: Number(a.frame ?? b.frame),
        tp: (b.tp || 0) - (a.tp || 0),
        fp: (b.fp || 0) - (a.fp || 0),
        fn: (b.fn || 0) - (a.fn || 0),
        gt: b.gt || 0,
        est: b.est || 0,
        max_tp_error: b.max_tp_error || null
      }));
    } else {
      const dataA = await api("/api/scenario_curve", {path: state.path, ...curvePayload});
      if (requestId !== state.curveRequestId) return;
      state.curve = dataA.frames || [];
    }
    renderCurve();
  } catch (err) {
    state.curve = [];
    renderCurve(`Curve failed: ${err.message}`);
  }
}
function sceneFilters(s) {
  const f = {topic_name: s.topic_name, scenario_name: s.scenario_name};
  if (s.suite_name) f.suite_name = s.suite_name;
  if (state.rangeMax !== "") f.distance_max = Number(state.rangeMax);
  if (state.label) f.label = state.label;
  return f;
}

function setPreviewToNearestFrame(frame) {
  if (!state.previewFrames.length || frame == null) return;
  const minFrame = Math.min(...state.previewFrames.map(f => Number(f.frame)));
  const maxFrame = Math.max(...state.previewFrames.map(f => Number(f.frame)));
  if (state.compare && state.selected && (Number(frame) < minFrame || Number(frame) > maxFrame)) {
    loadPreview(state.selected, {centerFrame: frame, radius: 4}).then(() => setPreviewToNearestFrame(frame));
    return;
  }
  let best = 0, dist = Infinity;
  state.previewFrames.forEach((f, i) => {
    const d = Math.abs(Number(f.frame) - Number(frame));
    if (d < dist) { best = i; dist = d; }
  });
  state.previewIndex = best;
  els.previewSlider.value = String(best);
  renderPreview();
  renderCurve();
}
function openViewer(frame = null) {
  if (!state.selected) return;
  const p = new URLSearchParams();
  p.set("path", state.path);
  if (state.compare) {
    p.set("path_b", els.parquetB.value || "");
    p.set("compare", "1");
    const lensMap = {changed_only: "changed_only", delta_fp: "new_fp_b", delta_fn: "resolved_fn_b", regression: "changed_only"};
    p.set("lens", lensMap[state.lens] || "all");
    p.set("layout", "side_by_side");
  }
  p.set("suite", state.selected.suite_name || "");
  p.set("scenario", state.selected.scenario_name || "");
  p.set("topic", state.selected.topic_name || "");
  if (frame != null) p.set("frame", String(frame));
  showViewer(`/bbox-viewer/?${p.toString()}`, frame);
}
function showViewer(url, frame = null) {
  state.viewerUrl = url;
  els.viewerShell.classList.add("show");
  els.viewerShell.setAttribute("aria-hidden", "false");
  els.viewerShellTitle.textContent = state.selected ? scenarioName(state.selected) : "BBox Viewer";
  els.viewerShellMeta.textContent = [
    state.compare ? `A: ${shortPathName(state.path)} vs B: ${shortPathName(els.parquetB.value || state.pathB)}` : shortPathName(state.path),
    frame != null ? `frame ${frame}` : "all loaded frames",
  ].join(" · ");
  if (els.viewerFrame.src !== new URL(url, window.location.href).href) {
    els.viewerFrame.src = url;
  }
}
function closeViewer() {
  els.viewerShell.classList.remove("show");
  els.viewerShell.setAttribute("aria-hidden", "true");
}
function openViewerNewTab() {
  if (!state.viewerUrl) return;
  window.open(state.viewerUrl, "_blank");
}
els.scan.addEventListener("click", scan);
els.parquet.addEventListener("change", async () => { state.path = els.parquet.value; await hydrate(); await loadSummary(); });
els.parquetB.addEventListener("change", async () => { state.pathB = els.parquetB.value; if (els.compareEnabled.checked) await loadSummary({selectTop: true}); });
els.compareEnabled.addEventListener("change", async () => { updateCompareControls(); await loadSummary({selectTop: true}); });
els.topic.addEventListener("change", loadSummary);
els.search.addEventListener("input", () => { renderList(); updateKpis(); render(); });
els.lensChips.querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", () => {
  if (chip.classList.contains("compare-chip") && !els.compareEnabled.checked) {
    toast("Turn on comparison to use this lens.");
    return;
  }
  state.lens = chip.dataset.lens;
  els.lensChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c === chip));
  layoutNodes(); renderList(); updateKpis(); updateCompareBanner(); render(); if (state.selected) loadCurve(state.selected);
}));
els.rangeChips.querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", async () => {
  state.rangeMax = chip.dataset.range || "";
  els.rangeChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c === chip));
  await loadSummary({selectTop: true});
}));
function setLens(value) {
  state.lens = value;
  els.lensChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c.dataset.lens === value));
}
function setLabel(value) {
  state.label = value;
  buildLabelChips();
  layoutNodes();
}
async function setRange(value) {
  state.rangeMax = value;
  els.rangeChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", (c.dataset.range || "") === value));
  await loadSummary({selectTop: true});
}
els.nearPed.addEventListener("click", async () => {
  setLens("count");
  setLabel("pedestrian");
  await setRange("30");
});
els.nearPedFp.addEventListener("click", async () => {
  setLens("fp");
  setLabel("pedestrian");
  await setRange("30");
});
function setLayout(value) {
  state.stageView = "map";
  els.stage.scrollTop = 0;
  state.layout = value;
  els.layoutGalaxy.classList.toggle("active", true);
  els.layoutStats.classList.remove("active");
  layoutNodes(); render();
}
els.layoutGalaxy.addEventListener("click", () => setLayout("galaxy"));
els.layoutStats.addEventListener("click", () => {
  state.stageView = "stats";
  els.stage.scrollTop = 0;
  els.layoutGalaxy.classList.remove("active");
  els.layoutStats.classList.add("active");
  render();
  ensureStatsLoaded();
});
els.statsDistanceStyle.addEventListener("change", () => {
  state.statsDistanceStyle = els.statsDistanceStyle.value;
  render();
});
els.statsFrameFocus.addEventListener("change", () => {
  state.statsFrameFocus = els.statsFrameFocus.value;
  render();
});
els.resetView.addEventListener("click", () => {
  state.panX = 0; state.panY = 0; state.scale = 1; render();
});
els.openViewer.addEventListener("click", () => openViewer());
els.previewSlider.addEventListener("input", () => {
  if (!state.previewFrames.length) return;
  state.previewIndex = Math.max(0, Math.min(state.previewFrames.length - 1, Number(els.previewSlider.value) || 0));
  renderPreview();
  renderCurve();
});
els.previewFit.addEventListener("click", () => {
  state.previewPanX = 0;
  state.previewPanY = 0;
  state.previewScale = 1;
  renderPreview();
});
els.previewRings.addEventListener("click", () => {
  state.previewShowRings = !state.previewShowRings;
  els.previewRings.classList.toggle("active", state.previewShowRings);
  renderPreview();
});
els.previewLabels.addEventListener("click", () => {
  state.previewShowLabels = !state.previewShowLabels;
  els.previewLabels.classList.toggle("active", state.previewShowLabels);
  renderPreview();
});
els.previewOpen.addEventListener("click", () => {
  const frame = state.previewFrames[state.previewIndex];
  openViewer(frame && frame.frame);
});
els.previewClose.addEventListener("click", () => {
  state.previewVisible = false;
  els.previewWindow.classList.remove("show");
});
els.previewLayers.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
  btn.classList.toggle("active");
  state.previewHoverBox = null;
  renderPreview();
}));
els.preview.addEventListener("dblclick", () => {
  const frame = state.previewFrames[state.previewIndex];
  openViewer(frame && frame.frame);
});
els.preview.addEventListener("pointerdown", e => {
  e.preventDefault();
  state.previewPanning = true;
  state.previewLastX = e.clientX;
  state.previewLastY = e.clientY;
  els.preview.setPointerCapture(e.pointerId);
});
els.preview.addEventListener("pointermove", e => {
  const rect = els.preview.getBoundingClientRect();
  state.previewMouseX = e.clientX - rect.left;
  state.previewMouseY = e.clientY - rect.top;
  if (!state.previewPanning) {
    updatePreviewHover();
    renderPreview();
    return;
  }
  const r = rect;
  const scale = Math.max(.001, previewScaleForRect(previewInteractionViewport(r, e.clientX)));
  state.previewPanY += (e.clientX - state.previewLastX) / scale;
  state.previewPanX += (e.clientY - state.previewLastY) / scale;
  state.previewLastX = e.clientX;
  state.previewLastY = e.clientY;
  renderPreview();
});
els.preview.addEventListener("pointerleave", () => {
  state.previewHoverBox = null;
  renderPreview();
});
function stopPreviewPan(e) {
  state.previewPanning = false;
  try { els.preview.releasePointerCapture(e.pointerId); } catch (_err) {}
}
els.preview.addEventListener("pointerup", stopPreviewPan);
els.preview.addEventListener("pointercancel", stopPreviewPan);
els.preview.addEventListener("wheel", e => {
  e.preventDefault();
  const rect = els.preview.getBoundingClientRect();
  const vp = previewInteractionViewport(rect, e.clientX);
  const oldScale = Math.max(.001, previewScaleForRect(vp));
  const sx = vp.x + vp.width / 2, sy = vp.y + vp.height / 2 + 12;
  const localX = e.clientX - rect.left;
  const localY = e.clientY - rect.top;
  const worldY = state.previewPanY - (localX - sx) / oldScale;
  const worldX = state.previewPanX - (localY - sy) / oldScale;
  state.previewScale = Math.max(.35, Math.min(8, state.previewScale * (e.deltaY > 0 ? .9 : 1.1)));
  const nextScale = Math.max(.001, previewScaleForRect(vp));
  state.previewPanY = worldY + (localX - sx) / nextScale;
  state.previewPanX = worldX + (localY - sy) / nextScale;
  renderPreview();
}, {passive: false});
els.curve.addEventListener("click", e => {
  if (!state.curve.length) return;
  const rect = els.curve.getBoundingClientRect();
  const plotX = 10;
  const plotW = Math.max(1, rect.width - 20);
  const t = Math.max(0, Math.min(1, (e.clientX - rect.left - plotX) / plotW));
  const idx = Math.round(t * Math.max(0, state.curve.length - 1));
  setPreviewToNearestFrame(state.curve[idx] && state.curve[idx].frame);
});
els.previewTitlebar.addEventListener("pointerdown", e => {
  if (e.target.closest("button")) return;
  const r = els.previewWindow.getBoundingClientRect();
  const stage = els.canvas.getBoundingClientRect();
  state.previewDrag = {dx: e.clientX - r.left, dy: e.clientY - r.top, stageLeft: stage.left, stageTop: stage.top};
  els.previewTitlebar.setPointerCapture(e.pointerId);
});
els.previewTitlebar.addEventListener("pointermove", e => {
  if (!state.previewDrag) return;
  els.previewWindow.style.left = `${e.clientX - state.previewDrag.stageLeft - state.previewDrag.dx}px`;
  els.previewWindow.style.top = `${e.clientY - state.previewDrag.stageTop - state.previewDrag.dy}px`;
  clampPreviewWindow();
});
els.previewTitlebar.addEventListener("pointerup", e => {
  state.previewDrag = null;
  try { els.previewTitlebar.releasePointerCapture(e.pointerId); } catch (_err) {}
});
els.previewResize.addEventListener("pointerdown", e => {
  e.preventDefault();
  state.previewResize = {x: e.clientX, y: e.clientY, w: els.previewWindow.offsetWidth, h: els.previewWindow.offsetHeight};
  els.previewResize.setPointerCapture(e.pointerId);
});
els.previewResize.addEventListener("pointermove", e => {
  if (!state.previewResize) return;
  els.previewWindow.style.width = `${state.previewResize.w + e.clientX - state.previewResize.x}px`;
  els.previewWindow.style.height = `${state.previewResize.h + e.clientY - state.previewResize.y}px`;
  clampPreviewWindow();
  renderPreview();
});
els.previewResize.addEventListener("pointerup", e => {
  state.previewResize = null;
  try { els.previewResize.releasePointerCapture(e.pointerId); } catch (_err) {}
});
els.canvas.addEventListener("pointerdown", e => {
  const rect = els.canvas.getBoundingClientRect();
  state.mouseX = e.clientX - rect.left;
  state.mouseY = e.clientY - rect.top;
  if (state.stageView === "stats") {
    state.dragging = false;
    render();
    return;
  }
  state.dragging = true;
  state.lastX = e.clientX; state.lastY = e.clientY; state.downX = e.clientX; state.downY = e.clientY;
  els.canvas.setPointerCapture(e.pointerId);
});
els.canvas.addEventListener("pointerup", e => {
  if (state.stageView === "stats") {
    if (state.statsHover) openStatsDetail(state.statsHover);
    return;
  }
  const moved = Math.hypot(e.clientX - state.downX, e.clientY - state.downY);
  state.dragging = false;
  if (moved < 5) {
    if (state.hoverLabel) applyLabel(state.hoverLabel.label === state.label ? "" : state.hoverLabel.label);
    else if (state.hover) selectScenario(state.hover);
  }
});
els.canvas.addEventListener("pointermove", e => {
  const rect = els.canvas.getBoundingClientRect();
  state.mouseX = e.clientX - rect.left;
  state.mouseY = e.clientY - rect.top;
  if (state.stageView === "stats") {
    render();
    return;
  }
  if (state.dragging) {
    const dx = e.clientX - state.lastX;
    const dy = e.clientY - state.lastY;
    state.panX += dx / Math.max(.25, state.scale);
    state.panY += dy / Math.max(.25, state.scale);
    state.lastX = e.clientX; state.lastY = e.clientY;
  }
  render();
});
els.canvas.addEventListener("contextmenu", e => e.preventDefault());
els.canvas.addEventListener("pointerleave", () => { state.hover = null; state.statsHover = null; els.hoverCard.classList.remove("show"); });
els.canvas.addEventListener("wheel", e => {
  if (state.stageView === "stats") return;
  e.preventDefault();
  state.scale = Math.max(.35, Math.min(3.4, state.scale * (e.deltaY > 0 ? .92 : 1.08)));
  render();
}, {passive: false});
els.statsDetailClose.addEventListener("click", () => {
  state.statsDetail = null;
  renderStatsDetail();
});
els.statsDetailDownload.addEventListener("click", downloadStatsDetailCsv);
els.statsDetailTable.addEventListener("click", e => {
  const btn = e.target.closest(".stats-viewer-link");
  if (!btn) return;
  openStatsDetailRowViewer(Number(btn.dataset.row));
});
els.viewerClose.addEventListener("click", closeViewer);
els.viewerNewTab.addEventListener("click", openViewerNewTab);
window.addEventListener("keydown", e => {
  if (e.key === "Escape" && els.viewerShell.classList.contains("show")) closeViewer();
});
window.addEventListener("resize", () => { clampPreviewWindow(); render(); renderCurve(); renderPreview(); });
scan();
