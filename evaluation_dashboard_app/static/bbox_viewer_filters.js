function toast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => els.toast.classList.remove("show"), 2600);
}
function setStatus(message) {
  els.loadStatus.innerHTML = message;
}
function loading(on) { els.spinner.classList.toggle("show", Boolean(on)); }
function setOptions(select, values, keep = "") {
  const current = keep || select.value;
  select.innerHTML = `<option value="">Any</option>` + values.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
  if (values.includes(current)) select.value = current;
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function chipGroup(root, values, defaults = []) {
  const useAll = defaults === true;
  const selected = new Set(useAll ? [] : defaults);
  if (!values.length) {
    root.innerHTML = `<span class="control-caption">No label values</span>`;
    return;
  }
  root.innerHTML = values.slice(0, 36).map(v => `<span class="chip ${useAll || selected.has(v) ? "active" : ""}" data-v="${escapeHtml(v)}">${escapeHtml(v)}</span>`).join("");
  root.querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", () => {
    chip.classList.toggle("active");
    state.selected = null;
    updateInspect();
    render();
  }));
}
function activeLabelDefaults(nextLabels) {
  const chips = [...els.labels.querySelectorAll(".chip")];
  if (!chips.length) return true;
  const active = new Set(chips.filter(chip => chip.classList.contains("active")).map(chip => chip.dataset.v));
  if (!active.size && chips.length === 0) return true;
  return nextLabels.filter(label => active.has(label));
}
function ensureLabelChips(values) {
  const labels = [...new Set((values || []).map(v => String(v || "").trim()).filter(Boolean))].sort((a, b) => a.localeCompare(b));
  chipGroup(els.labels, labels, activeLabelDefaults(labels));
}
function labelsFromFrames(frames) {
  const labels = [];
  for (const f of frames || []) for (const b of f.boxes || []) {
    if (b.label != null && String(b.label).trim()) labels.push(String(b.label).trim());
  }
  return labels;
}
function clearSceneWindow() {
  els.frameMin.value = "";
  els.frameMax.value = "";
  state.selectedScenario = null;
  state.selected = null;
  state.frames = [];
  state.framePos = 0;
  updateInspect();
  render();
}
function updateCompareControls() {
  const on = els.compareEnabled.checked;
  els.parquetB.disabled = !on;
  els.compareLens.disabled = !on;
  els.compareLayout.disabled = !on;
  els.compareSideBtn.disabled = !on;
  els.compareCurtainBtn.disabled = !on;
  els.compareLensBtn.disabled = !on;
  if (on) {
    if (els.compareLayout.value === "split") els.compareLayout.value = "side_by_side";
  }
  if (!on && els.colorMode.value === "run") els.colorMode.value = "status";
  if (on && els.colorMode.value === "status") els.colorMode.value = "run";
  setCompareLayout(els.compareLayout.value === "curtain" ? "curtain" : "side_by_side", {skipRender: true});
  syncCycleButtons();
  updateCompareBanner();
  render();
}
function scheduleSceneReload(reason = "setting") {
  if (!state.path || (!state.frames.length && state.selectedScenario == null)) return;
  clearTimeout(state.autoLoadTimer);
  setStatus(`<b>Updating.</b> Applying ${escapeHtml(reason)}...`);
  state.autoLoadTimer = setTimeout(() => loadScene({preserveFrame: true}), 320);
}
function baseFilters() {
  const f = {};
  if (els.suite.value) f.suite_name = els.suite.value;
  if (els.scenario.value) f.scenario_name = els.scenario.value;
  if (els.topic.value) f.topic_name = els.topic.value;
  return f;
}
async function scan() {
  loading(true);
  try {
    setStatus("<b>Scanning.</b> Looking for bbox-compatible parquet files...");
    const data = await api("/api/parquets", {root: els.root.value, limit: 2000, bbox_only: true});
    state.parquets = data.items || [];
    els.parquet.innerHTML = state.parquets.map(p => `<option value="${escapeHtml(p.path)}">${escapeHtml(p.display)}</option>`).join("");
    els.parquetB.innerHTML = els.parquet.innerHTML;
    if (!state.parquets.length) {
      toast(`No bbox-compatible parquet files found under that root${data.skipped ? ` (${data.skipped} skipped)` : ""}.`);
      setStatus(`<b>No bbox parquets.</b> ${data.skipped || 0} files were skipped because they were not bbox result parquets.`);
      return;
    }
    state.path = els.parquet.value || state.parquets[0].path;
    if (state.deepLink && state.deepLink.path) {
      const hit = state.parquets.find(p => p.path === state.deepLink.path || p.display === state.deepLink.path || p.path.endsWith(state.deepLink.path));
      if (hit) {
        els.parquet.value = hit.path;
        state.path = hit.path;
      }
    }
    if (state.parquets.length > 1) {
      const next = state.parquets.find(p => p.path !== state.path);
      if (next) els.parquetB.value = next.path;
    }
    if (state.deepLink && state.deepLink.pathB) {
      const hitB = state.parquets.find(p => p.path === state.deepLink.pathB || p.display === state.deepLink.pathB || p.path.endsWith(state.deepLink.pathB));
      if (hitB) els.parquetB.value = hitB.path;
    }
    if (state.deepLink && state.deepLink.compare) {
      els.compareEnabled.checked = Boolean(els.parquetB.value && els.parquetB.value !== state.path);
      if (state.deepLink.lens && [...els.compareLens.options].some(o => o.value === state.deepLink.lens)) {
        els.compareLens.value = state.deepLink.lens;
      }
      if (state.deepLink.layout && [...els.compareLayout.options].some(o => o.value === state.deepLink.layout)) {
        els.compareLayout.value = state.deepLink.layout;
      }
      updateCompareControls();
    }
    state.pathB = els.parquetB.value || state.path;
    await hydrateFilters();
    await searchScenarios({autoLoad: true});
  } catch (err) {
    toast(err.message);
    setStatus(`<b>Scan failed.</b> ${escapeHtml(err.message)}`);
  } finally {
    loading(false);
  }
}
async function values(column, filters = {}) {
  const data = await api("/api/values", {path: state.path, column, filters});
  return data.values || [];
}
async function hydrateFilters() {
  state.path = els.parquet.value;
  state.pathB = els.parquetB.value || state.path;
  const empty = {};
  const [suites, topics, statuses, sources, labels] = await Promise.all([
    values("suite_name", empty).catch(() => []),
    values("topic_name", empty).catch(() => []),
    values("status", empty).catch(() => []),
    values("source", empty).catch(() => []),
    values("label", empty).catch(() => [])
  ]);
  setOptions(els.suite, suites);
  setOptions(els.topic, topics);
  for (const preferred of ["perception.object_recognition.objects", "perception.object_recognition.tracking.objects"]) {
    if (topics.includes(preferred)) { els.topic.value = preferred; break; }
  }
  if (state.deepLink) {
    if (state.deepLink.suite && suites.includes(state.deepLink.suite)) els.suite.value = state.deepLink.suite;
    if (state.deepLink.topic && topics.includes(state.deepLink.topic)) els.topic.value = state.deepLink.topic;
  }
  setOptions(els.scenario, await values("scenario_name", baseFilters()).catch(() => []));
  setOptions(els.dataset, await values("t4dataset_name", baseFilters()).catch(() => []));
  if (state.deepLink && state.deepLink.scenario) els.scenario.value = state.deepLink.scenario;
  chipGroup(els.statuses, statuses, statuses.filter(v => ["TP","FP","FN"].includes(v)));
  chipGroup(els.sources, sources, sources);
  ensureLabelChips(labels);
}
async function refreshDependentFilters() {
  const current = baseFilters();
  const [scenarios, datasets, labels] = await Promise.all([
    values("scenario_name", current).catch(() => []),
    values("t4dataset_name", current).catch(() => []),
    values("label", current).catch(() => [])
  ]);
  setOptions(els.scenario, scenarios);
  setOptions(els.dataset, datasets);
  ensureLabelChips(labels);
  await searchScenarios();
}
async function searchScenarios(options = {}) {
  if (!state.path) return;
  try {
    const data = await api("/api/scenarios", {path: state.path, q: els.search.value, filters: baseFilters(), limit: 220});
    state.scenarios = data.items || [];
    renderResults();
    if (options.autoLoad && state.scenarios.length) {
      applyScenario(0, false);
      await loadScene();
    } else if (!state.scenarios.length) {
      setStatus("<b>No scenarios matched.</b> Clear search/filter controls or choose another parquet.");
    }
  } catch (err) { toast(err.message); }
}
function applyScenario(idx, keepFullRange = true) {
  state.selectedScenario = idx;
  const s = state.scenarios[idx];
  if (!s) return;
  if (s.suite_name) els.suite.value = s.suite_name;
  if (s.scenario_name) els.scenario.value = s.scenario_name;
  if (s.topic_name) els.topic.value = s.topic_name;
  els.frameMin.value = "";
  els.frameMax.value = "";
  renderResults();
}
function renderResults() {
  els.results.innerHTML = state.scenarios.map((s, idx) => {
    const name = s.scenario_name || s.t4dataset_name || s.topic_name || `Scenario ${idx + 1}`;
    const meta = `${s.frames || 0} frames · ${(s.rows || 0).toLocaleString()} rows · ${s.topic_name || "topic any"}`;
    return `<div class="scenario ${state.selectedScenario === idx ? "active" : ""}" data-i="${idx}">
      <strong>${escapeHtml(name)}</strong>
      <span>${escapeHtml(meta)}</span>
      <span>${escapeHtml(s.suite_name || "")}</span>
    </div>`;
  }).join("");
  els.results.querySelectorAll(".scenario").forEach(card => {
    card.addEventListener("click", () => {
      const idx = Number(card.dataset.i);
      applyScenario(idx, true);
      loadScene();
    });
  });
}
async function loadScene(options = {}) {
  if (!state.path) return;
  loading(true);
  try {
    const previousFrame = options.preserveFrame && state.frames[state.framePos] ? Number(state.frames[state.framePos].frame) : null;
    let filters = baseFilters();
    if (!filters.scenario_name) {
      state.frames = [];
      state.framePos = 0;
      state.selected = null;
      updateInspect();
      render();
      setStatus("<b>Select a scene.</b> Suite/topic filters only narrow the scene list; click a scenario card to load boxes.");
      return;
    }
    setStatus(`<b>Loading.</b> ${escapeHtml(state.path.split("/").slice(-3).join("/"))} with filters ${escapeHtml(JSON.stringify(filters))}`);
    if (els.compareEnabled.checked && els.parquetB.value === els.parquet.value) {
      const alt = state.parquets.find(p => p.path !== els.parquet.value);
      if (alt) {
        els.parquetB.value = alt.path;
        toast("Run B matched Run A, so I switched to the next parquet for comparison.");
      }
    }
    state.compare = els.compareEnabled.checked && els.parquetB.value && els.parquetB.value !== els.parquet.value;
    if (els.compareEnabled.checked && !state.compare) {
      toast("Choose a different Run B parquet to enable comparison.");
    }
    if (state.compare) {
      if (!els.compareLayout.value || els.compareLayout.value === "split") els.compareLayout.value = "side_by_side";
      if (els.colorMode.value === "status") els.colorMode.value = "run";
    }
    syncCycleButtons();
    syncCompareIssueLabels();
    const request = {
      filters,
      max_rows: state.compare ? 90000 : 180000,
      dedupe: els.dedupeRows.checked
    };
    let data = state.compare
      ? await api("/api/compare_frames", {
          ...request,
          runs: [
            {label: "A", path: state.path},
            {label: "B", path: els.parquetB.value}
          ]
        })
      : await api("/api/frames", { ...request, path: state.path, run: "A" });
    state.frames = normalizeFrames(data.frames || []);
    const sceneLabels = labelsFromFrames(state.frames);
    if (sceneLabels.length) ensureLabelChips(sceneLabels);
    if (previousFrame != null && state.frames.length) {
      let bestIdx = 0;
      let bestDist = Infinity;
      state.frames.forEach((f, idx) => {
        const dist = Math.abs(Number(f.frame) - previousFrame);
        if (dist < bestDist) { bestDist = dist; bestIdx = idx; }
      });
      state.framePos = bestIdx;
    } else {
      state.framePos = 0;
    }
    if (state.deepLink && state.deepLink.frame !== "" && state.frames.length) {
      const requested = Number(state.deepLink.frame);
      if (Number.isFinite(requested)) {
        let bestIdx = 0;
        let bestDist = Infinity;
        state.frames.forEach((f, idx) => {
          const dist = Math.abs(Number(f.frame) - requested);
          if (dist < bestDist) { bestDist = dist; bestIdx = idx; }
        });
        state.framePos = bestIdx;
      }
      state.deepLink.frame = "";
    }
    state.selected = null;
    els.slider.max = Math.max(0, state.frames.length - 1);
    fitBounds();
    updateStats(data);
    updateCompareBanner();
    renderHeatStrip();
    updateInspect();
    render();
    if (!state.frames.length) {
      toast("No boxes matched. Try clearing labels/status/source or choose another bbox parquet.");
      setStatus(`<b>0 boxes loaded.</b> Path: ${escapeHtml(state.path)} · filters: ${escapeHtml(JSON.stringify(filters))}`);
    } else {
      toast(data.truncated ? "Loaded row limit; narrow filters for full scene." : "Scene loaded.");
      setStatus(
        `<b>${Number(data.row_count || 0).toLocaleString()} boxes · ${Number(data.frame_count || 0).toLocaleString()} frames.</b> ` +
        `${data.truncated ? "Row cap reached; narrow filters for full scene. " : ""}` +
        `${state.compare ? `Comparison mode: Run A vs Run B. ${escapeHtml((data.compare_runs || []).map(r => `${r.label}:${r.frame_count}f/${r.row_count}r`).join(" · "))}` : "Single run."}`
      );
    }
  } catch (err) { toast(err.message); }
  finally { loading(false); }
}
function normalizeFrames(frames) {
  return [...frames]
    .map(f => ({...f, frame: Number(f.frame)}))
    .filter(f => Number.isFinite(f.frame))
    .sort((a, b) => a.frame - b.frame);
}
