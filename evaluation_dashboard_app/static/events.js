const EXPLORER_SESSION_KEY = "local_bbox_explorer.session.v1";

function readExplorerSession() {
  try {
    const raw = localStorage.getItem(EXPLORER_SESSION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}
function selectedScenarioKey() {
  return state.selected ? scenarioKey(state.selected) : "";
}
function saveExplorerSessionNow() {
  if (state.savingSession) return;
  try {
    localStorage.setItem(EXPLORER_SESSION_KEY, JSON.stringify({
      root: els.root.value || "",
      path: state.path || els.parquet.value || "",
      pathB: state.pathB || els.parquetB.value || "",
      compare: Boolean(els.compareEnabled.checked),
      topic: els.topic.value || "",
      search: els.search.value || "",
      lens: state.lens || "fp",
      label: state.label || "",
      rangeMax: state.rangeMax || "",
      explorerMode: state.explorerMode || "hotspots",
      stageView: state.stageView || "map",
      layout: state.layout || "galaxy",
      expandedSuites: [...(state.expandedSuites || new Set())],
      selectedKey: selectedScenarioKey(),
    }));
  } catch {
  }
}
function saveExplorerSessionSoon() {
  clearTimeout(saveExplorerSessionSoon._t);
  saveExplorerSessionSoon._t = setTimeout(saveExplorerSessionNow, 120);
}
function applyInitialSession() {
  const saved = readExplorerSession();
  state.restoreSession = saved;
  if (!saved) return;
  state.savingSession = true;
  if (saved.root) els.root.value = saved.root;
  if (saved.search != null) els.search.value = saved.search;
  if (saved.lens) state.lens = saved.lens;
  if (saved.label != null) state.label = saved.label;
  if (saved.rangeMax != null) state.rangeMax = String(saved.rangeMax);
  if (saved.explorerMode) state.explorerMode = saved.explorerMode === "devops" ? "devops" : "hotspots";
  if (saved.stageView) state.stageView = saved.stageView === "stats" ? "stats" : "map";
  if (saved.layout) state.layout = saved.layout;
  els.compareEnabled.checked = Boolean(saved.compare);
  updateCompareControls();
  setLens(state.lens);
  els.rangeChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", (c.dataset.range || "") === state.rangeMax));
  updateExplorerModeClass();
  state.savingSession = false;
}
function restoreSelectValue(selectEl, value) {
  if (!selectEl || !value) return false;
  const option = [...selectEl.options].find(o => o.value === value);
  if (!option) return false;
  selectEl.value = value;
  return true;
}

async function scan() {
  try {
    toast("Scanning bbox parquets...");
    const data = await api("/api/parquets", {root: els.root.value, limit: 2000, bbox_only: true});
    state.parquets = data.items || [];
    const options = state.parquets.map(p => `<option value="${escapeHtml(p.path)}">${escapeHtml(p.display)}</option>`).join("");
    els.parquet.innerHTML = options;
    els.parquetB.innerHTML = options;
    if (!state.parquets.length) { toast("No bbox-compatible parquets found."); return; }
    const saved = state.restoreSession;
    restoreSelectValue(els.parquet, saved?.path);
    state.path = els.parquet.value || state.parquets[0].path;
    if (state.parquets[1]) els.parquetB.value = state.parquets[1].path;
    restoreSelectValue(els.parquetB, saved?.pathB);
    state.pathB = els.parquetB.value || "";
    updateCompareControls();
    await hydrate();
    await loadSummary();
    saveExplorerSessionSoon();
  } catch (err) { toast(err.message); }
}
async function hydrate() {
  state.path = els.parquet.value;
  const [topics] = await Promise.all([api("/api/values", {path: state.path, column: "topic_name", filters: {}}).catch(() => ({values: []}))]);
  els.topic.innerHTML = `<option value="">Any</option>` + (topics.values || []).map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
  if ((topics.values || []).includes("perception.object_recognition.objects")) els.topic.value = "perception.object_recognition.objects";
  if (state.restoreSession?.topic && (topics.values || []).includes(state.restoreSession.topic)) {
    els.topic.value = state.restoreSession.topic;
  }
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
  if (state.compare && !["changed_only", "delta_fp", "delta_fn", "regression", "fp", "fn", "fpr", "fnr", "intent_risk", "target_fn", "target_fp"].includes(state.lens)) setLens("changed_only");
  setBusy(true, state.compare ? "Comparing Run B against Run A..." : "Building dataset bbox summary...");
  try {
    const summaryPayload = {
      filters: filters(),
      limit: 1200,
      include_criteria_results: state.explorerMode === "devops" || /devops/i.test(shortPathName(state.path)),
    };
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
    const saved = state.restoreSessionApplied ? null : state.restoreSession;
    state.expandedSuites.clear();
    if (saved?.expandedSuites && Array.isArray(saved.expandedSuites)) {
      state.expandedSuites = new Set(saved.expandedSuites);
    }
    if (state.scenarios.some(s => devopsContext(s).is_devops) && state.explorerMode === "hotspots" && /devops/i.test(shortPathName(state.path))) {
      setExplorerMode("devops", {skipRender: true});
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
      state.devopsResult = null;
      state.previewVisible = false;
      els.previewWindow.classList.remove("show");
      els.selTitle.textContent = "No scenario selected";
      els.selMeta.textContent = "Click a scenario to load preview frames.";
      if (els.intentPanel) els.intentPanel.innerHTML = `<div class="scenario"><strong>No scenario selected</strong><span>Click a devops scenario to see purpose and criteria.</span></div>`;
      if (els.resultPanel) els.resultPanel.innerHTML = `<div class="scenario"><strong>No result loaded</strong><span>Select a scenario to evaluate criteria and hot frames.</span></div>`;
      renderCurve("Select a scenario to load the frame curve.");
    } else if (state.selected) {
      state.selected = state.scenarios.find(s => scenarioKey(s) === scenarioKey(state.selected)) || null;
    }
    if (saved?.selectedKey) {
      state.selected = state.scenarios.find(s => scenarioKey(s) === saved.selectedKey) || state.selected;
    }
    const ranked = filteredScenarios();
    const shouldAutoSelect = state.explorerMode !== "devops";
    if (shouldAutoSelect && !options.selectTop && (!state.selected || scenarioMetric(state.selected) <= 0) && ranked.length) {
      const best = ranked.find(s => scenarioMetric(s) > 0) || ranked[0];
      selectScenario(best, false);
    } else if (!options.selectTop && state.selected) {
      selectScenario(state.selected, false);
    }
    toast(`${state.compare ? "Compared" : "Loaded"} ${state.scenarios.length} scenarios.`);
    if (state.stageView === "stats") await ensureStatsLoaded(requestId);
    state.restoreSessionApplied = true;
    saveExplorerSessionSoon();
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
  saveExplorerSessionSoon();
}
function filteredScenarios() {
  const q = els.search.value.trim().toLowerCase();
  let arr = state.scenarios;
  if (q) {
    const terms = q.split(/\s+/).filter(Boolean);
    arr = arr.filter(s => {
      const haystack = [
        s.suite_name, s.scenario_name, s.t4dataset_name, s.topic_name,
        devopsPurposeText(s), devopsContext(s).description, devopsContext(s).family,
        ...(s.labels || []).map(x => x.label)
      ].join(" ").toLowerCase();
      return terms.some(term => haystack.includes(term));
    });
  }
  if (state.compare && state.lens === "changed_only") {
    arr = arr.filter(s => scenarioMetric(s, "changed_only") > 0);
  }
  return [...arr].sort((a, b) => scenarioMetric(b) - scenarioMetric(a));
}
function devopsFilteredScenarios() {
  const q = els.search.value.trim().toLowerCase();
  let arr = state.scenarios.filter(s => devopsContext(s).is_devops);
  if (q) {
    const terms = q.split(/\s+/).filter(Boolean);
    arr = arr.filter(s => {
      const ctx = devopsContext(s);
      const haystack = [
        s.suite_name, s.scenario_name, s.t4dataset_name, s.topic_name,
        devopsPurposeText(s), ctx.description, ctx.family, ctx.intent_type, ctx.target_label, ctx.behavior, ctx.city,
        ...(s.labels || []).map(x => x.label)
      ].join(" ").toLowerCase();
      return terms.some(term => haystack.includes(term));
    });
  }
  return [...arr].sort((a, b) => {
    if (state.compare) {
      const ac = scenarioCompareSummary(a), bc = scenarioCompareSummary(b);
      return Number(bc.regressed) - Number(ac.regressed)
        || Number(bc.fixed) - Number(ac.fixed)
        || Number(bc.changed) - Number(ac.changed)
        || bc.magnitude - ac.magnitude
        || (a.suite_name || "").localeCompare(b.suite_name || "")
        || scenarioName(a).localeCompare(scenarioName(b));
    }
    const aj = scenarioJudgement(a), bj = scenarioJudgement(b);
    const order = {fail: 0, review: 1, pass: 2};
    return order[aj.status] - order[bj.status]
      || (a.suite_name || "").localeCompare(b.suite_name || "")
      || scenarioName(a).localeCompare(scenarioName(b));
  });
}

function updateKpis() {
  const arr = state.explorerMode === "devops" ? devopsFilteredScenarios() : filteredScenarios();
  const fp = arr.reduce((n, s) => n + (state.explorerMode === "devops" ? targetMetric(s, "fp") : (state.label ? (labelRow(s, state.label)?.[state.compare ? "delta_fp" : "fp"] || 0) : (s[state.compare ? "delta_fp" : "fp"] || 0))), 0);
  const fn = arr.reduce((n, s) => n + (state.explorerMode === "devops" ? targetMetric(s, "fn") : (state.label ? (labelRow(s, state.label)?.[state.compare ? "delta_fn" : "fn"] || 0) : (s[state.compare ? "delta_fn" : "fn"] || 0))), 0);
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
  if (state.explorerMode === "devops") {
    renderDevopsReviewList(els.list, {updateTitle: true});
    return;
  }
  if (els.scenarioListTitle) els.scenarioListTitle.textContent = "Top Issues";
  const arr = filteredScenarios().slice(0, 80);
  els.list.innerHTML = arr.map(s => {
    const active = state.selected && scenarioKey(state.selected) === scenarioKey(s);
    const m = scenarioMetric(s);
    const statLine = state.compare
      ? `${compareLensLabel()} ${lensMetricText(m)} · ΔFP ${fmtDelta(s.delta_fp)} · ΔFN ${fmtDelta(s.delta_fn)} · B FP ${fmt(s.fp)}`
      : `${state.lens.toUpperCase()} ${state.lens.endsWith("r") ? rate(m) : fmt(Math.round(m))} · FP ${fmt(s.fp)} · FN ${fmt(s.fn)} · ${fmt(s.frames)} frames${state.rangeMax ? ` · <${state.rangeMax}m` : ""}`;
    const purpose = devopsPurposeText(s);
    return `<div class="scenario ${active ? "active" : ""}" data-key="${escapeHtml(scenarioKey(s))}">
      <strong>${escapeHtml(scenarioName(s))}</strong>
      ${purpose ? `<em>${escapeHtml(purpose)}</em>` : ""}
      <span>${escapeHtml(statLine)}</span>
      <span>${escapeHtml((s.labels || []).slice(0, 4).map(x => state.compare ? `${x.label}:ΔFP${fmtDelta(x.delta_fp || 0)}` : `${x.label}:${x.fp}FP`).join(" · "))}</span>
    </div>`;
  }).join("");
  els.list.querySelectorAll(".scenario").forEach(card => card.addEventListener("click", () => {
    const s = state.scenarios.find(x => scenarioKey(x) === card.dataset.key);
    if (s) selectScenario(s);
  }));
}
function renderDevopsReviewList(container = els.list, options = {}) {
  const arr = devopsFilteredScenarios();
  if (options.updateTitle && els.scenarioListTitle) els.scenarioListTitle.textContent = state.compare ? "DevOps Suites Compare" : "DevOps Suites";
  if (!container) return;
  if (!arr.length) {
    container.innerHTML = `<div class="scenario"><strong>No DevOps scenarios</strong><span>Choose a devops parquet or clear filters.</span></div>`;
    return;
  }
  const groups = devopsSuiteGroups(arr);
  container.innerHTML = groups.map(group => {
    const expanded = state.expandedSuites.has(group.key);
    const aTotal = group.aPass + group.aFail + group.aReview + group.aMissing;
    const bTotal = group.bPass + group.bFail + group.bReview + group.bMissing;
    const aRateValue = group.suitePassA ? group.suitePassA.pass_rate : group.aPass / Math.max(1, aTotal);
    const bRateValue = group.suitePassB ? group.suitePassB.pass_rate : group.bPass / Math.max(1, bTotal);
    const passText = state.compare
      ? `A ${fmt(group.suitePassA ? group.suitePassA.passed : group.aPass)}/${fmt(group.suitePassA ? group.suitePassA.total : aTotal)} → B ${fmt(group.suitePassB ? group.suitePassB.passed : group.bPass)}/${fmt(group.suitePassB ? group.suitePassB.total : bTotal)}`
      : (group.suitePass
        ? `${fmt(group.suitePass.passed)}/${fmt(group.suitePass.total)} pass`
        : `${fmt(group.pass)} pass · ${fmt(group.fail)} fail · ${fmt(group.review)} check`);
    const rateValue = state.compare ? bRateValue : (group.suitePass ? group.suitePass.pass_rate : group.pass / Math.max(1, group.items.length));
    const ratePct = Math.max(0, Math.min(100, Number(rateValue || 0) * 100));
    const aRatePct = Math.max(0, Math.min(100, Number(aRateValue || 0) * 100));
    const bRatePct = Math.max(0, Math.min(100, Number(bRateValue || 0) * 100));
    const rateDelta = bRateValue - aRateValue;
    const compareMeta = state.compare
      ? `<div class="suite-compare-meta">
          <i class="${rateDelta >= 0 ? "good" : "bad"}">${escapeHtml(`${rateDelta >= 0 ? "+" : ""}${Math.round(rateDelta * 100)}pp`)}</i>
          <span>${fmt(group.changed)} changed</span>
          <span class="${group.regressed ? "bad" : ""}">${fmt(group.regressed)} regressed</span>
          <span class="${group.fixed ? "good" : ""}">${fmt(group.fixed)} fixed</span>
          ${group.added || group.removed ? `<span>${fmt(group.added)} new · ${fmt(group.removed)} gone</span>` : ""}
        </div>`
      : "";
    const title = group.missing
      ? `${fmt(group.missing)} suite scenarios are unavailable in this parquet.`
      : "";
    const sortedItems = [...group.items].sort((a, b) => {
      const aj = scenarioJudgement(a), bj = scenarioJudgement(b);
      const order = {fail: 0, review: 1, pass: 2, missing: 3};
      return order[aj.status] - order[bj.status]
        || intentRiskMetric(b) - intentRiskMetric(a)
        || scenarioName(a).localeCompare(scenarioName(b));
    });
    const scenarios = expanded ? sortedItems
      .map(s => {
        const active = state.selected && scenarioKey(state.selected) === scenarioKey(s);
        const judgement = scenarioJudgement(s);
        const cmp = scenarioCompareSummary(s);
        const aJudgement = state.compare ? cmp.a : null;
        const bJudgement = state.compare ? cmp.b : null;
        const ctx = devopsContext(s);
        const unavailable = Boolean(ctx.unavailable);
        const compareClass = state.compare
          ? (cmp.regressed ? "regressed" : (cmp.fixed ? "fixed" : (cmp.changed ? "changed" : "stable")))
          : "";
        const compareDetail = state.compare
          ? `${aJudgement.label} → ${bJudgement.label} · ΔFP ${fmtDelta(cmp.deltaFp)} · ΔFN ${fmtDelta(cmp.deltaFn)} · ΔTP ${fmtDelta(cmp.deltaTp)}`
          : "";
        const detail = state.compare
          ? compareDetail
          : unavailable
          ? (ctx.unavailable_reason || "No bbox/evaluation rows were recorded or downloaded for this parquet.")
          : `${judgement.reason} · TP ${fmt(targetMetric(s, "tp"))} · FP ${fmt(targetMetric(s, "fp"))} · FN ${fmt(targetMetric(s, "fn"))}`;
        return `<div class="devops-case ${active ? "active" : ""} ${unavailable ? "unavailable" : ""} ${compareClass}" data-key="${escapeHtml(scenarioKey(s))}" data-unavailable="${unavailable ? "1" : "0"}">
          ${state.compare
            ? `<span class="compare-verdict">
                <i class="${aJudgement.status}">${escapeHtml(aJudgement.label)}</i>
                <b>→</b>
                <i class="${bJudgement.status}">${escapeHtml(bJudgement.label)}</i>
              </span>`
            : `<span class="status-pill ${judgement.status}">${escapeHtml(judgement.label)}</span>`}
          <div>
            <strong>${escapeHtml(scenarioName(s))}</strong>
            <span>${escapeHtml([ctx.intent_type, ctx.target_label, ctx.behavior, ctx.city].filter(Boolean).join(" · "))}</span>
            <small>${escapeHtml(detail)}</small>
          </div>
        </div>`;
      }).join("") : "";
    return `<div class="suite-group">
      <button class="suite-head" data-suite="${escapeHtml(group.key)}" title="${escapeHtml(title)}">
        <span>${expanded ? "▾" : "▸"}</span>
        <strong>${escapeHtml(group.key.replace(/^DevOps_V1_/, ""))}</strong>
        <em>${escapeHtml(passText)}</em>
      </button>
      ${compareMeta}
      ${state.compare
        ? `<div class="suite-rate compare-rate"><i class="run-a" style="width:${aRatePct}%"></i><i class="run-b" style="width:${bRatePct}%"></i></div>`
        : `<div class="suite-rate"><i style="width:${ratePct}%"></i></div>`}
      ${scenarios ? `<div class="suite-cases">${scenarios}</div>` : ""}
    </div>`;
  }).join("");
  container.querySelectorAll(".suite-head").forEach(btn => btn.addEventListener("click", () => {
    const key = btn.dataset.suite || "";
    if (state.expandedSuites.has(key)) state.expandedSuites.delete(key);
    else state.expandedSuites.add(key);
    renderList();
    saveExplorerSessionSoon();
  }));
  container.querySelectorAll(".devops-case").forEach(row => row.addEventListener("click", () => {
    if (row.dataset.unavailable === "1") {
      toast("This scenario exists in the DevOps suite, but no bbox rows are available in the selected parquet.");
      return;
    }
    const s = state.scenarios.find(x => scenarioKey(x) === row.dataset.key);
    if (s) selectScenario(s);
  }));
}
async function selectScenario(s, flash = true) {
  state.selected = s;
  saveExplorerSessionSoon();
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
  state.devopsResult = null;
  renderIntentPanel(s);
  renderResultPanel("Loading result explanation...");
  renderScenarioLabels(s);
  renderList();
  render();
  await loadScenarioDetails(s, flash);
}
function renderIntentPanel(s) {
  const panels = [els.intentPanel].filter(Boolean);
  if (!panels.length) return;
  const ctx = devopsContext(s);
  if (!ctx.is_devops) {
    panels.forEach(panel => { panel.innerHTML = `<div class="scenario"><strong>Generic bbox evaluation</strong><span>No DevOps scenario metadata was inferred.</span></div>`; });
    return;
  }
  const targetLabels = (ctx.target_labels || []).slice(0, 8).join(", ");
  const matchingPolicy = ctx.matching_label_policy ? String(ctx.matching_label_policy).replace(/_/g, " ") : "";
  const suitePass = ctx.suite_pass
    ? `<div class="intent-note">Suite pass: ${fmt(ctx.suite_pass.passed)} / ${fmt(ctx.suite_pass.total)} (${rate(ctx.suite_pass.pass_rate)})</div>`
    : "";
  const html = `
    <div class="intent-summary">
      <strong>${escapeHtml(ctx.purpose || ctx.intent_type || "DevOps scenario")}</strong>
      <span>${escapeHtml(devopsQuickRead(s))}</span>
    </div>
    <div class="intent-tags">
      ${[ctx.issue_type, ctx.target_label, ctx.behavior, ctx.pc_mode, ctx.city].filter(Boolean).map(x => `<i>${escapeHtml(x)}</i>`).join("")}
    </div>
    ${ctx.description ? `<p>${escapeHtml(ctx.description)}</p>` : ""}
    ${suitePass}
    ${targetLabels ? `<div class="intent-note">Evaluator labels: ${escapeHtml(targetLabels)}</div>` : ""}
    ${matchingPolicy || ctx.merge_similar_labels ? `<div class="intent-note">Label matching: ${escapeHtml(matchingPolicy || "default")}${ctx.merge_similar_labels ? " · merged similar labels" : ""}</div>` : ""}
    ${ctx.matching_thresholds && ctx.matching_thresholds.length ? `<div class="intent-note">Matching thresholds: ${escapeHtml(ctx.matching_thresholds.join(", "))}</div>` : ""}
  `;
  panels.forEach(panel => { panel.innerHTML = html; });
}
async function loadScenarioDetails(s, flash = false) {
  const key = scenarioKey(s);
  await loadPreview(s);
  if (!state.selected || scenarioKey(state.selected) !== key) return;
  if (state.explorerMode === "devops") {
    state.curve = [];
    await loadScenarioResult(s);
    if (!state.selected || scenarioKey(state.selected) !== key) return;
    renderPreview();
    renderResultPanel();
    renderCurve("Frame curve is hidden in DevOps review.");
    if (flash) toast("Scenario selected. Result loaded.");
    return;
  }
  await loadCurve(s);
  if (!state.selected || scenarioKey(state.selected) !== key) return;
  await loadScenarioResult(s);
  if (!state.selected || scenarioKey(state.selected) !== key) return;
  focusPreviewOnCurvePeak();
  renderPreview();
  renderResultPanel();
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
async function loadScenarioResult(s) {
  const requestId = ++state.resultRequestId;
  if (!s || !devopsContext(s).is_devops) {
    state.devopsResult = null;
    renderResultPanel();
    return;
  }
  const resultPath = state.compare ? (els.parquetB.value || state.path) : state.path;
  try {
    const data = await api("/api/scenario_devops_result", {path: resultPath, filters: sceneFilters(s), timeout_ms: 12000});
    if (requestId !== state.resultRequestId) return;
    state.devopsResult = normalizeScenarioDevopsResult(data, s);
    applyScenarioCriteriaResult(s, state.devopsResult);
    renderResultPanel();
    if (state.previewVisible) renderPreview();
    render();
  } catch (err) {
    if (requestId !== state.resultRequestId) return;
    state.devopsResult = fallbackScenarioResult(s, err.message);
    renderResultPanel();
    if (state.previewVisible) renderPreview();
    render();
  }
}
function applyScenarioCriteriaResult(s, result) {
  if (!s || !result || result.fallback) return;
  const criteriaResult = {
    overall_pass: result.overall_pass,
    failed_count: Number(result.failed_count || 0),
    gate_count: Number(result.gate_count || 0),
    explanation: (result.explanation || [""])[0],
  };
  const key = scenarioKey(s);
  const fullContext = result.context ? {...(s.devops || {}), ...result.context, criteria_result: criteriaResult} : null;
  const apply = item => {
    if (!item || scenarioKey(item) !== key) return;
    item.devops = fullContext || {...(item.devops || {}), criteria_result: criteriaResult};
  };
  apply(s);
  if (state.selected && scenarioKey(state.selected) === key) apply(state.selected);
  state.scenarios.forEach(apply);
  if (state.selected && scenarioKey(state.selected) === key) renderIntentPanel(state.selected);
  renderList();
}
function normalizeScenarioDevopsResult(result, s) {
  if (!result || result.fallback || !s) return result;
  result = {
    ...result,
    gates: (result.gates || []).map(g => {
      const ctxGate = result.context && Array.isArray(result.context.criteria) ? result.context.criteria[Number(g.index)] : null;
      return {...g, filter: g.filter || (ctxGate && ctxGate.filter) || {}};
    }),
  };
  const ctx = devopsContext(s);
  if (ctx.focus_metric !== "fp") return result;
  const fp = targetMetric(s, "fp") || Number(s.fp || 0);
  const frames = Math.max(1, Number(s.frames || 0));
  const gates = (result.gates || []).map(g => {
    if (String(g.method || "").toLowerCase() !== "num_gt_tp") return g;
    if (String(g.evaluation_task || "").toLowerCase() === "fp_validation") return g;
    const required = Number(g.required_rate);
    const actual = fp > 0 ? 0 : 1;
    return {
      ...g,
      metric_label: "frame FP-validation pass rate",
      meaning: "FP-validation: the validation object should remain TN. EST/FP detections make the frame fail.",
      actual_rate: actual,
      passed: Number.isFinite(required) ? actual >= required : fp === 0,
      passed_count: fp > 0 ? 0 : frames,
      fail_count: fp > 0 ? frames : 0,
      total_count: frames,
      object_success_count: 0,
      object_fail_count: fp,
      object_total_count: fp,
      evaluation_task: "fp_validation",
    };
  });
  const known = gates.filter(g => g.passed != null);
  const failed = known.filter(g => g.passed === false);
  if (!failed.length && gates === result.gates) return result;
  const explanation = failed.length
    ? [`Criterion ${Number(failed[0].index) + 1} fails: ${failed[0].metric_label} is ${pctText(failed[0].actual_rate)}, below required ${pctText(failed[0].required_rate)} in ${failed[0].distance_label || "all distances"}.`]
    : result.explanation;
  return {...result, gates, overall_pass: known.length ? failed.length === 0 : result.overall_pass, failed_count: failed.length, explanation};
}
function pctText(value) {
  return value == null || !Number.isFinite(Number(value)) ? "-" : `${(Number(value) * 100).toFixed(1)}%`;
}
function currentPreviewFrameForResult() {
  if (!state.previewFrames.length) return null;
  return state.previewFrames[Math.max(0, Math.min(state.previewIndex, state.previewFrames.length - 1))] || null;
}
function currentFrameBoxesForResult(frame) {
  return [...((frame && frame.boxes) || [])].sort((a, b) => (String(a.source) === "GT" ? -1 : 1) - (String(b.source) === "GT" ? -1 : 1));
}
function exactFrameResultFor(frame) {
  if (!frame || !state.devopsFrameResults || !Array.isArray(state.devopsFrameResults.frames)) return null;
  const frameNo = Number(frame.frame);
  return state.devopsFrameResults.frames.find(f => Number(f.frame) === frameNo) || null;
}
function exactFrameGateFor(g, frame) {
  const exact = exactFrameResultFor(frame);
  if (!exact || !Array.isArray(exact.gates)) return null;
  return exact.gates.find(x => Number(x.index) === Number(g.index)) || null;
}
function frameJudgementHtml(g, frame, boxes) {
  if (!frame || typeof previewFrameGateEvidence !== "function") return "";
  const exactGate = exactFrameGateFor(g, frame);
  if (exactGate) {
    const counts = exactGate.counts || {};
    const frameClass = exactGate.passed == null ? "skip" : (exactGate.passed ? "pass" : "fail");
    const frameTitle = exactGate.passed == null ? "Frame Not Judged" : (exactGate.passed ? "Frame PASS" : "Frame FAIL");
    const score = exactGate.score == null
      ? "-"
      : (exactGate.score_unit === "rad" ? `${Number(exactGate.score).toFixed(3)} rad` : `${Number(exactGate.score).toFixed(1)}%`);
    const level = exactGate.level == null
      ? "-"
      : (exactGate.score_unit === "rad" ? `${Number(exactGate.level).toFixed(3)} rad` : `${Number(exactGate.level).toFixed(0)}%`);
    const detail = exactGate.passed == null
      ? `no evaluator target after ${exactGate.distance_label || "filter"}`
      : `${score} / ${level} · success ${counts.success || 0} · fail ${counts.fail || 0} · GT TP ${counts.gt_tp || 0} · GT TN ${counts.gt_tn || 0} · GT FN ${counts.gt_fn || 0} · EST FP ${counts.est_fp || 0}`;
    return `<div class="preview-frame-gate result-frame-gate ${frameClass}">
      <b>${escapeHtml(frameTitle)}</b>
      <span>${escapeHtml(detail)}</span>
      <small>${escapeHtml("exact from scene_result.pkl")}</small>
    </div>`;
  }
  const frameEvidence = previewFrameGateEvidence(boxes, g);
  const frameScore = frameEvidence.score == null
    ? "-"
    : (frameEvidence.scoreUnit === "rad" ? `${frameEvidence.score.toFixed(3)} rad` : `${frameEvidence.score.toFixed(1)}%`);
  const frameLevel = frameEvidence.level == null
    ? "-"
    : (frameEvidence.scoreUnit === "rad" ? `${frameEvidence.level.toFixed(3)} rad` : `${frameEvidence.level.toFixed(0)}%`);
  const frameClass = frameEvidence.pass == null ? "skip" : (frameEvidence.pass ? "pass" : "fail");
  const frameTitle = frameEvidence.pass == null ? "Frame Not Judged" : (frameEvidence.pass ? "Frame PASS" : "Frame FAIL");
  const frameDetail = frameEvidence.pass == null
    ? `${g.distance_label || "criterion filter"}`
    : `${frameScore} / ${frameLevel} · ok ${frameEvidence.passedCount}/${frameEvidence.totalCount} · GT TP ${frameEvidence.gtTp} · GT FN ${frameEvidence.gtFn} · EST FP ${frameEvidence.estFp}`;
  return `<div class="preview-frame-gate result-frame-gate ${frameClass}">
    <b>${escapeHtml(`${frameTitle} (approx)`)}</b>
    <span>${escapeHtml(frameDetail)}</span>
  </div>`;
}
function planningFactorResultHtml(ctx) {
  const pf = ctx.planning_factor || {};
  if (!pf.path) return "";
  const pfCondition = (pf.conditions || []).map(c => {
    const dist = c.distance ? `${c.distance.min ?? "-"}..${c.distance.max ?? "-"}m` : "";
    return [c.topic, (c.behavior || []).join("/"), c.judgement, dist].filter(Boolean).join(" · ");
  }).join(" | ");
  return `<div class="criterion"><b>Planning factor</b><span>${escapeHtml(`frames pass ${pf.passed_frames || 0} · fail ${pf.failed_frames || 0} · nodata ${pf.nodata_frames || 0}${pfCondition ? ` · ${pfCondition}` : ""}`)}</span></div>`;
}
function fallbackScenarioResult(s, detail = "") {
  const ctx = devopsContext(s);
  const target = ctx.target_label || "target";
  const tp = targetMetric(s, "tp");
  const fp = targetMetric(s, "fp");
  const fn = targetMetric(s, "fn");
  const totalDetect = tp + fn;
  const recall = totalDetect ? tp / totalDetect : null;
  const totalEst = tp + fp;
  const precision = totalEst ? tp / totalEst : null;
  const focus = ctx.focus_metric || (ctx.issue_type === "FN" ? "fn" : "fp");
  const riskCount = focus === "fn" ? fn : (focus === "error" ? (s.max_tp_error || 0) : fp);
  const overallPass = focus === "error" ? false : riskCount === 0;
  const gates = [];
  if (focus === "fn" || ctx.issue_type === "FN") {
    gates.push({
      index: 0, method: "summary_target_fn", metric_label: `${target} detection`,
      meaning: "Fallback check: target objects should not be missed. Restart the bbox API for exact YAML distance gates.",
      distance_label: state.rangeMax ? `<${state.rangeMax}m filter` : "current filters",
      required_rate: 1, actual_rate: recall, passed: fn === 0 && recall != null,
      passed_count: tp, fail_count: fn, total_count: totalDetect,
    });
  }
  if (focus === "fp" || ctx.issue_type === "FP") {
    gates.push({
      index: gates.length, method: "summary_target_fp", metric_label: `${target} false positives`,
      meaning: "Fallback check: false detections should be zero for this intent. Restart the bbox API for exact YAML gates.",
      distance_label: state.rangeMax ? `<${state.rangeMax}m filter` : "current filters",
      required_rate: 1, actual_rate: precision, passed: fp === 0 && precision != null,
      passed_count: tp, fail_count: fp, total_count: totalEst,
    });
  }
  if (!gates.length) {
    gates.push({
      index: 0, method: "summary_error", metric_label: "TP error",
      meaning: "Fallback check: exact yaw/position criteria need the newer bbox API route.",
      distance_label: "current filters", required_rate: null, actual_rate: null, passed: null,
      passed_count: tp, fail_count: 0, total_count: tp,
    });
  }
  const hotFrames = [...(state.curve || [])]
    .sort((a, b) => ((b.fn || 0) + (b.fp || 0)) - ((a.fn || 0) + (a.fp || 0)))
    .slice(0, 8);
  const explanation = [
    focus === "fn"
      ? `${target}: ${fmt(fn)} FN and ${fmt(tp)} TP in the loaded summary. ${fn ? "Likely fail until those misses are inspected." : "No target misses in the loaded summary."}`
      : (focus === "fp"
        ? `${target}: ${fmt(fp)} FP and ${fmt(tp)} TP in the loaded summary. ${fp ? "Likely fail for a false-detection/false-stop case." : "No target false positives in the loaded summary."}`
        : `${target}: exact error gate needs the newer result API; loaded max TP error is ${Number(s.max_tp_error || 0).toFixed(2)} m.`),
  ];
  if (/Unknown route/i.test(detail)) explanation.push("Detailed YAML gate evaluation is unavailable because the running bbox API is older; restart it to enable exact criteria visualization.");
  else if (detail) explanation.push(`Detailed YAML gate evaluation unavailable: ${detail}`);
  return {fallback: true, overall_pass: overallPass, gates, hot_frames: hotFrames, explanation};
}
function renderResultPanel(message = "") {
  const panels = [els.resultPanel].filter(Boolean);
  if (!panels.length) return;
  const s = state.selected;
  if (message) {
    panels.forEach(panel => { panel.innerHTML = `<div class="scenario"><strong>${escapeHtml(message)}</strong><span>Computing YAML criteria against bbox rows.</span></div>`; });
    return;
  }
  if (!s || !devopsContext(s).is_devops) {
    panels.forEach(panel => { panel.innerHTML = `<div class="scenario"><strong>Generic bbox evaluation</strong><span>No DevOps result gates were inferred for this scenario.</span></div>`; });
    return;
  }
  const result = state.devopsResult;
  if (!result) {
    panels.forEach(panel => { panel.innerHTML = `<div class="scenario"><strong>No result loaded</strong><span>Select a scenario to evaluate criteria and hot frames.</span></div>`; });
    return;
  }
  const status = result.overall_pass ? "PASS" : "FAIL";
  const statusClass = result.overall_pass ? "pass" : "fail";
  const gates = result.gates || [];
  const ctx = devopsContext(s);
  const sources = [...new Set(gates.map(g => g.source).filter(Boolean))];
  const frameResultNote = state.devopsFrameResults && state.devopsFrameResults.available === false
    ? `Frame judgement source: approximate only; ${state.devopsFrameResults.reason || "scene_result.pkl unavailable"}.`
    : "";
  const sourceNote = [result.warning || (sources.length ? `Criteria source: ${sources.join(", ")}.` : ""), frameResultNote].filter(Boolean).join(" ");
  const frame = currentPreviewFrameForResult();
  const frameBoxes = currentFrameBoxesForResult(frame);
  const gateHtml = gates.length ? gates.map(g => {
    const actual = Number(g.actual_rate);
    const required = Number(g.required_rate);
    const actualPct = Number.isFinite(actual) ? Math.max(0, Math.min(100, actual * 100)) : 0;
    const requiredPct = Number.isFinite(required) ? Math.max(0, Math.min(100, required * 100)) : 0;
    const gateClass = g.passed === true ? "pass" : (g.passed === false ? "fail" : "unknown");
    return `<div class="gate ${gateClass}">
      <div class="gate-head">
        <b>${escapeHtml(`#${Number(g.index) + 1} ${g.metric_label || g.method || "criterion"}`)}</b>
        <i>${g.passed === true ? "PASS" : (g.passed === false ? "FAIL" : "UNKNOWN")}</i>
      </div>
      <div class="gate-bar">
        <span class="gate-required" style="left:${requiredPct}%"></span>
        <em style="width:${actualPct}%"></em>
      </div>
      <div class="gate-meta">${escapeHtml(`${pctText(g.actual_rate)} actual / ${pctText(g.required_rate)} required · ${g.passed_count || 0}/${g.total_count || 0} passed · ${g.fail_count || 0} failed · ${g.distance_label || "all distances"}`)}</div>
      <div class="gate-meaning">${escapeHtml(g.meaning || "")}</div>
      ${frameJudgementHtml(g, frame, frameBoxes)}
    </div>`;
  }).join("") : `<div class="criterion"><b>No supported gates</b><span>This scenario has no criterion this explorer can compute yet.</span></div>`;
  const planningFactor = planningFactorResultHtml(ctx);
  const html = `
    <div class="verdict ${statusClass}">
      <strong>${status}${result.fallback ? " (estimated)" : ""}</strong>
      <span>${escapeHtml((result.explanation || []).join(" "))}</span>
      ${sourceNote ? `<small>${escapeHtml(sourceNote)}</small>` : ""}
    </div>
    <div class="gates">${gateHtml}</div>
    ${planningFactor}
  `;
  panels.forEach(panel => {
    panel.innerHTML = html;
  });
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
  if (devopsContext(s).is_devops) return f;
  if (state.rangeMax !== "") f.distance_max = Number(state.rangeMax);
  if (state.label) f.label = state.label;
  return f;
}

function setPreviewToNearestFrame(frame) {
  if (!state.previewFrames.length || frame == null) return;
  const minFrame = Math.min(...state.previewFrames.map(f => Number(f.frame)));
  const maxFrame = Math.max(...state.previewFrames.map(f => Number(f.frame)));
  if (state.selected && (Number(frame) < minFrame || Number(frame) > maxFrame)) {
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
  if (devopsContext(state.selected).is_devops) p.set("devops", "1");
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
els.parquet.addEventListener("change", async () => { state.path = els.parquet.value; await hydrate(); await loadSummary(); saveExplorerSessionSoon(); });
els.parquetB.addEventListener("change", async () => { state.pathB = els.parquetB.value; if (els.compareEnabled.checked) await loadSummary({selectTop: true}); saveExplorerSessionSoon(); });
els.compareEnabled.addEventListener("change", async () => { updateCompareControls(); await loadSummary({selectTop: true}); saveExplorerSessionSoon(); });
els.topic.addEventListener("change", async () => { await loadSummary(); saveExplorerSessionSoon(); });
els.search.addEventListener("input", () => { renderList(); updateKpis(); render(); saveExplorerSessionSoon(); });
els.lensChips.querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", () => {
  if (chip.classList.contains("compare-chip") && !els.compareEnabled.checked) {
    toast("Turn on comparison to use this lens.");
    return;
  }
  state.lens = chip.dataset.lens;
  els.lensChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c === chip));
  layoutNodes(); renderList(); updateKpis(); updateCompareBanner(); render(); if (state.selected) loadCurve(state.selected);
  saveExplorerSessionSoon();
}));
els.rangeChips.querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", async () => {
  state.rangeMax = chip.dataset.range || "";
  els.rangeChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c === chip));
  await loadSummary({selectTop: true});
  saveExplorerSessionSoon();
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
  saveExplorerSessionSoon();
});
els.nearPedFp.addEventListener("click", async () => {
  setLens("fp");
  setLabel("pedestrian");
  await setRange("30");
  saveExplorerSessionSoon();
});
els.devopsIntent.addEventListener("click", async () => {
  els.search.value = "DevOps";
  setLens("intent_risk");
  setLabel("");
  await setRange("");
  saveExplorerSessionSoon();
});
els.animal.addEventListener("click", async () => {
  els.search.value = "animal dog cardboard debris fallen";
  setLens("target_fn");
  setLabel("");
  await setRange("40");
  saveExplorerSessionSoon();
});
els.falseStop.addEventListener("click", async () => {
  els.search.value = "FP ObstacleStop RoadUserStop false stop";
  setLens("target_fp");
  setLabel("");
  await setRange("40");
  saveExplorerSessionSoon();
});
function setLayout(value) {
  state.stageView = "map";
  state.explorerMode = "hotspots";
  updateExplorerModeClass();
  els.stage.scrollTop = 0;
  state.layout = value;
  els.layoutGalaxy.classList.toggle("active", true);
  els.layoutReview.classList.remove("active");
  els.layoutStats.classList.remove("active");
  if (els.modeChips) els.modeChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c.dataset.mode === "hotspots"));
  layoutNodes(); render();
  renderList();
  saveExplorerSessionSoon();
}
els.layoutGalaxy.addEventListener("click", () => setLayout("galaxy"));
els.layoutReview.addEventListener("click", () => setExplorerMode("devops"));
els.layoutStats.addEventListener("click", () => {
  state.stageView = "stats";
  state.explorerMode = "hotspots";
  updateExplorerModeClass();
  els.stage.scrollTop = 0;
  els.layoutGalaxy.classList.remove("active");
  els.layoutReview.classList.remove("active");
  els.layoutStats.classList.add("active");
  if (els.modeChips) els.modeChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c.dataset.mode === "hotspots"));
  renderList();
  render();
  ensureStatsLoaded();
  saveExplorerSessionSoon();
});
function updateExplorerModeClass() {
  document.querySelector(".app")?.classList.toggle("devops-review-mode", state.explorerMode === "devops");
}
function setExplorerMode(value, options = {}) {
  const previousRange = state.rangeMax;
  state.explorerMode = value === "devops" ? "devops" : "hotspots";
  updateExplorerModeClass();
  if (state.explorerMode === "devops") state.stageView = "map";
  if (state.explorerMode === "devops") {
    state.lens = "intent_risk";
    state.label = "";
    state.rangeMax = "";
    setLens("intent_risk");
    buildLabelChips();
    els.rangeChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", (c.dataset.range || "") === ""));
  }
  if (els.modeChips) els.modeChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c.dataset.mode === state.explorerMode));
  if (els.layoutReview) els.layoutReview.classList.toggle("active", state.explorerMode === "devops");
  if (els.layoutGalaxy) els.layoutGalaxy.classList.toggle("active", state.stageView === "map" && state.explorerMode !== "devops");
  if (els.layoutStats) els.layoutStats.classList.remove("active");
  if (state.explorerMode === "devops" && state.selected) {
    showPreviewWindow();
    els.previewTitle.textContent = scenarioName(state.selected);
    renderPreview();
  }
  if (!options.skipRender) {
    renderList();
    updateKpis();
    render();
  }
  if (state.explorerMode === "devops" && previousRange !== "" && !options.skipRender && state.path) {
    loadSummary({selectTop: true});
  }
  saveExplorerSessionSoon();
}
els.modeChips.querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", () => setExplorerMode(chip.dataset.mode)));
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
  renderResultPanel();
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
  state.downX = e.clientX;
  state.downY = e.clientY;
  if (state.stageView === "stats" || state.explorerMode === "devops") {
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
  if (state.explorerMode === "devops") {
    const hit = state.devopsHoverHit;
    if (hit && hit.kind === "scenario") selectScenario(hit.s);
    else if (hit && hit.kind === "suite") {
      if (state.expandedSuites.has(hit.suite)) state.expandedSuites.delete(hit.suite);
      else state.expandedSuites.add(hit.suite);
      renderList();
      render();
    }
    else if (hit && hit.kind === "frame") openViewer(hit.frame);
    else if (hit && hit.kind === "viewer") openViewer();
    return;
  }
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
  if (state.explorerMode === "devops") {
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
els.canvas.addEventListener("pointerleave", () => { state.hover = null; state.statsHover = null; state.devopsHoverHit = null; els.hoverCard.classList.remove("show"); });
els.canvas.addEventListener("wheel", e => {
  if (state.stageView === "stats") return;
  if (state.explorerMode === "devops") {
    e.preventDefault();
    state.devopsCanvasScroll = Math.max(0, Math.min(state.devopsCanvasMaxScroll || 0, (state.devopsCanvasScroll || 0) + e.deltaY));
    render();
    return;
  }
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
applyInitialSession();
scan();
