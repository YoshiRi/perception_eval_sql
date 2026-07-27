function filters() {
  const f = {};
  if (els.topic.value) f.topic_name = els.topic.value;
  if (state.rangeMax !== "") f.distance_max = Number(state.rangeMax);
  return f;
}
function scenarioKey(s) { return `${s.suite_name || ""}|${s.scenario_name || ""}|${s.t4dataset_name || ""}|${s.topic_name || ""}`; }
function emptyStats(seed = {}) {
  return {rows: 0, frames: 0, gt: 0, est: 0, tp: 0, fp: 0, fn: 0, fpr: 0, fnr: 0, precision: null, recall: null, max_tp_error: 0, avg_tp_error: 0, labels: [], ...seed};
}
function labelRow(s, label) {
  return (s.labels || []).find(x => x.label === label) || null;
}
function metricFromRow(row, lens) {
  if (!row) return 0;
  if (lens === "count") return row.rows || 0;
  if (lens === "fp") return row.fp || 0;
  if (lens === "fn") return row.fn || 0;
  if (lens === "fpr") return row.fpr != null ? row.fpr : (row.fp || 0) / Math.max(1, row.rows || 0);
  if (lens === "fnr") return row.fnr != null ? row.fnr : (row.fn || 0) / Math.max(1, row.rows || 0);
  if (lens === "error") return row.max_tp_error || 0;
  if (lens === "changed_only") return Math.abs(row.delta_fp || 0) + Math.abs(row.delta_fn || 0) + Math.abs(row.delta_tp || 0);
  if (lens === "delta_fp") return row.delta_fp || 0;
  if (lens === "delta_fn") return row.delta_fn || 0;
  if (lens === "regression") return Math.max(0, row.delta_fp || 0) + Math.max(0, row.delta_fn || 0);
  return row.fp || 0;
}
function scenarioMetric(s, lens = state.lens, label = state.label) {
  if (label) {
    return metricFromRow(labelRow(s, label), lens);
  }
  return metricFromRow(s, lens);
}
function activeRunStats(s) {
  return state.compare ? emptyStats(s.b || {}) : s;
}
function mergeLabelRows(aRows = [], bRows = []) {
  const map = new Map();
  for (const r of aRows) map.set(r.label || "unknown", {label: r.label || "unknown", a: r, b: emptyStats()});
  for (const r of bRows) {
    const key = r.label || "unknown";
    const hit = map.get(key) || {label: key, a: emptyStats(), b: emptyStats()};
    hit.b = r;
    map.set(key, hit);
  }
  return [...map.values()].map(({label, a, b}) => {
    const out = {...b, label, a, b};
    out.delta_tp = (b.tp || 0) - (a.tp || 0);
    out.delta_fp = (b.fp || 0) - (a.fp || 0);
    out.delta_fn = (b.fn || 0) - (a.fn || 0);
    out.delta_rows = (b.rows || 0) - (a.rows || 0);
    return out;
  }).sort((x, y) => Math.abs(y.delta_fp || 0) + Math.abs(y.delta_fn || 0) - (Math.abs(x.delta_fp || 0) + Math.abs(x.delta_fn || 0)));
}
function mergeSummaries(aItems, bItems) {
  const map = new Map();
  for (const a of aItems) map.set(scenarioKey(a), {a, b: null, seed: a});
  for (const b of bItems) {
    const key = scenarioKey(b);
    const hit = map.get(key) || {a: null, b: null, seed: b};
    hit.b = b;
    map.set(key, hit);
  }
  return [...map.values()].map(({a, b, seed}) => {
    const aa = a ? emptyStats(a) : emptyStats();
    const bb = b ? emptyStats(b) : emptyStats();
    const out = {...seed, ...bb, a: aa, b: bb, compare: true};
    out.delta_tp = (bb.tp || 0) - (aa.tp || 0);
    out.delta_fp = (bb.fp || 0) - (aa.fp || 0);
    out.delta_fn = (bb.fn || 0) - (aa.fn || 0);
    out.delta_fpr = (bb.fpr || 0) - (aa.fpr || 0);
    out.delta_fnr = (bb.fnr || 0) - (aa.fnr || 0);
    out.delta_rows = (bb.rows || 0) - (aa.rows || 0);
    out.labels = mergeLabelRows(aa.labels || [], bb.labels || []);
    return out;
  });
}
function metricKey(row, keys) {
  return keys.map(k => String(row[k] ?? "")).join("|");
}
function mergeRowsByKey(aRows = [], bRows = [], keys = []) {
  const metrics = ["rows", "gt", "est", "tp", "fp", "fn"];
  const rates = ["tpr", "fpr", "precision", "recall", "mean_abs_x_error", "mean_abs_y_error", "mean_abs_yaw_error"];
  const map = new Map();
  for (const a of aRows || []) map.set(metricKey(a, keys), {a, b: {}, seed: a});
  for (const b of bRows || []) {
    const key = metricKey(b, keys);
    const hit = map.get(key) || {a: {}, b: {}, seed: b};
    hit.b = b;
    map.set(key, hit);
  }
  return [...map.values()].map(({a, b, seed}) => {
    const out = {...seed, ...b, a, b};
    metrics.forEach(m => out[`delta_${m}`] = (Number(b[m]) || 0) - (Number(a[m]) || 0));
    rates.forEach(m => {
      const av = a[m] == null ? null : Number(a[m]);
      const bv = b[m] == null ? null : Number(b[m]);
      out[`delta_${m}`] = av == null || bv == null ? null : bv - av;
    });
    return out;
  });
}
function mergeStats(a, b) {
  return {
    distance: mergeRowsByKey(a.distance || [], b.distance || [], ["distance_bin"]),
    label_distance: mergeRowsByKey(a.label_distance || [], b.label_distance || [], ["label", "distance_bin"]),
    labels: mergeRowsByKey(a.labels || [], b.labels || [], ["label"]),
    errors: mergeRowsByKey(a.errors || [], b.errors || [], ["label"]),
    scenarios: mergeRowsByKey(a.scenarios || [], b.scenarios || [], ["suite_name", "scenario_name", "t4dataset_name", "topic_name"]),
    datasets: mergeRowsByKey(a.datasets || [], b.datasets || [], ["suite_name", "scenario_name", "t4dataset_name", "t4dataset_id", "topic_name"]),
    frames: mergeRowsByKey(a.frames || [], b.frames || [], ["suite_name", "scenario_name", "t4dataset_name", "t4dataset_id", "topic_name", "frame"]),
    label_scenarios: mergeRowsByKey(a.label_scenarios || [], b.label_scenarios || [], ["label", "suite_name", "scenario_name", "t4dataset_name", "topic_name"]),
    label_datasets: mergeRowsByKey(a.label_datasets || [], b.label_datasets || [], ["label", "suite_name", "scenario_name", "t4dataset_name", "t4dataset_id", "topic_name"]),
    label_frames: mergeRowsByKey(a.label_frames || [], b.label_frames || [], ["label", "suite_name", "scenario_name", "t4dataset_name", "t4dataset_id", "topic_name", "frame"]),
    a, b, compare: true
  };
}
function scenarioName(s) {
  const raw = s.scenario_name || s.t4dataset_name || "scenario";
  return raw.replace(/^FullPerformance_V1_/, "");
}
function cityName(s) {
  const raw = `${s.suite_name || ""} ${s.scenario_name || ""}`;
  const m = raw.match(/FullPerformance_V1_([^_\s]+)/);
  return m ? m[1] : ((s.suite_name || s.scenario_name || "Unknown").split(/[_\s-]/)[0] || "Unknown");
}
function suiteName(s) {
  return (s.suite_name || "suite").replace(/^FullPerformance_V1_/, "");
}
function hashText(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return (h >>> 0) / 4294967295;
}
function clusterKey(s) {
  return state.layout === "clusters" ? suiteName(s) : cityName(s);
}
var LABEL_ORDER = ["car", "truck", "bus", "pedestrian", "bicycle", "motorbike", "animal", "unknown"];
function allLabelNames() {
  const names = new Set();
  LABEL_ORDER.forEach(x => names.add(x));
  state.labels.forEach(x => { if (x.label) names.add(x.label); });
  state.scenarios.forEach(s => (s.labels || []).forEach(x => { if (x.label) names.add(x.label); }));
  return [...names].sort((a, b) => {
    const ai = LABEL_ORDER.indexOf(a), bi = LABEL_ORDER.indexOf(b);
    if (ai >= 0 || bi >= 0) return (ai >= 0 ? ai : 999) - (bi >= 0 ? bi : 999);
    return a.localeCompare(b);
  });
}
function zeroLabel(label) {
  return {label, tp: 0, fp: 0, fn: 0, rows: 0, delta_tp: 0, delta_fp: 0, delta_fn: 0, a: emptyStats(), b: emptyStats()};
}
function labelsForScenario(s) {
  const byLabel = new Map((s.labels || []).map(x => [x.label || "unknown", x]));
  return allLabelNames().map(label => ({...zeroLabel(label), ...(byLabel.get(label) || {})}));
}
