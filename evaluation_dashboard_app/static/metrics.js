function filters() {
  const f = {};
  if (els.topic.value) f.topic_name = els.topic.value;
  if (state.rangeMax !== "") f.distance_max = Number(state.rangeMax);
  return f;
}
function scenarioKey(s) { return `${s.suite_name || ""}|${s.scenario_name || ""}|${s.t4dataset_name || ""}|${s.topic_name || ""}`; }
function suiteBaseName(suite) {
  return String(suite || "Unknown suite").replace(/_[0-9a-f]{8}-[0-9a-f-]{27,}$/i, "");
}
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
function devopsContext(row) {
  if (row && row.devops && row.devops.is_devops) return row.devops;
  if (row && row.b && row.b.devops && row.b.devops.is_devops) return row.b.devops;
  if (row && row.a && row.a.devops && row.a.devops.is_devops) return row.a.devops;
  const inferred = inferDevopsContext(row || {});
  return inferred.is_devops ? inferred : ((row && row.devops) || {});
}
function normalizeDevopsToken(value) {
  const raw = String(value || "").replace(/([a-z])([A-Z])/g, "$1_$2").replace(/[-\s]+/g, "_").toLowerCase();
  const aliases = {
    pedestrian: "pedestrian", pedestrians: "pedestrian", child: "pedestrian", children: "pedestrian", pedestrianchild: "pedestrian",
    dog: "animal", animal: "animal", bird: "animal", dragonfly: "animal",
    cardboard: "fallen_object", card_board: "fallen_object", fallensign: "fallen_object", fallen_sign: "fallen_object", sandbag: "fallen_object", plasticbag: "fallen_object", umbrella: "fallen_object",
    cone: "traffic_cone", cones: "traffic_cone", traffic_cone: "traffic_cone",
    truck: "truck", trailer: "truck", track: "truck", bus: "bus", car: "car",
    bicycle: "bicycle", bicycles: "bicycle", motorbike: "motorbike", motorcycle: "motorbike", motorcycles: "motorbike",
    shrub: "vegetation", tree: "vegetation", plant: "vegetation", vegetation: "vegetation",
    ghost: "ghost", rain: "rain", watervapor: "exhaust_fog", surfacecluster: "ground", surface_cluster: "ground",
    pole: "structure", rubberpole: "structure", signboard: "structure", streetlight: "structure", board: "structure"
  };
  return aliases[raw] || raw;
}
function inferDevopsContext(row) {
  const suite = String(row.suite_name || "");
  const scenario = String(row.scenario_name || "");
  const blob = `${suite} ${scenario}`;
  if (!/DevOps/i.test(blob)) return {is_devops: false};
  const tokens = scenario.split(/[_\s]+/).filter(Boolean);
  const suiteTokens = suite.split(/[_\s]+/).filter(Boolean);
  const issue = ([...suiteTokens, ...tokens].find(t => /^(FN|FP|TP)$/i.test(t)) || "").toUpperCase();
  const behavior = tokens.find(t => /^(ObstacleStop|RoadUserStop|RunOut|Crosswalk|Intersection|IntersectionLeft|IntersectionRight|IntersectionStraight|Normal)$/i.test(t)) || "";
  const cityIdx = tokens.findIndex(t => /^(j6|j6Gen|J6Gen|J6Gen2|j6Gen2|x2)/.test(t));
  const city = cityIdx >= 0 && tokens[cityIdx + 1] ? tokens[cityIdx + 1] : "";
  const pcToken = tokens.find(t => /^(PCOn|PCOff|PCON|PCOFF)$/i.test(t)) || "";
  const pc_mode = /on$/i.test(pcToken) ? "PC on" : (/off$/i.test(pcToken) ? "PC off" : "");
  const known = new Set(["pedestrian", "animal", "fallen_object", "traffic_cone", "truck", "bus", "car", "bicycle", "motorbike", "vegetation", "ghost", "rain", "exhaust_fog", "ground", "structure"]);
  let target_label = "";
  for (const token of [...tokens, ...suite.split(/[_\s]+/)]) {
    const label = normalizeDevopsToken(token);
    if (known.has(label)) { target_label = label; break; }
  }
  let intent_type = issue === "FN" ? "target detection" : (issue === "FP" ? "false detection / false stop" : "investigate");
  let focus_metric = issue === "FN" ? "fn" : "fp";
  if (/yaw|jitter/i.test(suite)) { intent_type = "localization/yaw accuracy"; focus_metric = "error"; }
  if (/misclassified/i.test(suite)) { intent_type = "structure misclassification"; focus_metric = "fp"; }
  return {
    is_devops: true, issue_type: issue, intent_type, focus_metric, target_label, behavior, pc_mode, city,
    family: suite.replace(/^DevOps_V1_/, ""), purpose: [issue, behavior, target_label, city].filter(Boolean).join(" "),
    description: "", criteria: [], target_labels: [], matching_thresholds: []
  };
}
function devopsPolicyAllowsUnknown(ctx = null) {
  const policy = String(ctx && ctx.matching_label_policy || "").toLowerCase();
  return policy === "allow_unknown" || policy === "allow_same_group" || !!(ctx && ctx.merge_similar_labels);
}
function targetLabelAliases(label, ctx = null) {
  const raw = String(label || "").toLowerCase();
  const aliases = {
    animal: ["animal", "dog", "cat", "rabbit", "bird", "crow", "crows", "pigeon", "kite", "kites", "dragonfly"],
    dog: ["animal", "dog"],
    fallen_object: ["fallen_object", "road_debris", "cardboard", "card_board", "fallen_sign", "plastic_bag", "sandbag", "sunshade", "umbrella"],
    cardboard: ["fallen_object", "cardboard", "card_board"],
    traffic_cone: ["traffic_cone", "cone", "cones", "fallen_cone"],
    cone: ["traffic_cone", "cone", "cones", "fallen_cone"],
    opened_door: ["opened_door"],
    motorbike: ["motorbike", "motorcycle", "motorcycles", "motercycle"],
    motorcycle: ["motorbike", "motorcycle", "motorcycles", "motercycle"],
    truck: ["truck", "trailer", "track"],
    ground: ["ground"],
    vegetation: ["vegetation", "plant", "shrub", "tree"],
    structure: ["structure", "streetlight", "signboard", "pole", "rubberpole", "utility_pole_or_banner", "watersupply", "board"],
    ghost: ["ghost", "ghost_from_fence", "ghost_from_guardrail", "ghost_or_side_mirror", "rocket"],
    rain: ["rain"],
    exhaust_fog: ["exhaust_fog", "watervapor", "water_vapor", "exhaust", "fog"],
  };
  const out = aliases[raw] ? [...aliases[raw]] : (raw ? [raw] : []);
  if (devopsPolicyAllowsUnknown(ctx) && raw !== "unknown") {
    if (["animal", "dog", "fallen_object", "cardboard", "traffic_cone", "ground", "vegetation", "structure", "ghost", "rain", "exhaust_fog", "opened_door"].includes(raw)) {
      out.push("unknown");
    }
  }
  if (ctx && ctx.merge_similar_labels) {
    if (["truck", "trailer", "bus"].includes(raw)) out.push("car");
    if (["motorbike", "motorcycle"].includes(raw)) out.push("bicycle");
    if (["traffic_cone", "fallen_object", "ground", "structure"].includes(raw)) out.push("unknown");
  }
  return [...new Set(out)];
}
function targetRows(s) {
  const ctx = devopsContext(s);
  const labels = targetLabelAliases(ctx.target_label, ctx);
  if (!labels.length) return [];
  return labels.map(label => labelRow(s, label)).filter(Boolean);
}
function sumMetric(rows, key) {
  return rows.reduce((n, row) => n + (Number(row && row[key]) || 0), 0);
}
function targetMetric(s, key) {
  const rows = targetRows(s);
  if (!rows.length) return Number(s && s[key]) || 0;
  return sumMetric(rows, key);
}
function intentRiskMetric(s) {
  const ctx = devopsContext(s);
  if (ctx.focus_metric === "fn") return targetMetric(s, state.compare ? "delta_fn" : "fn");
  if (ctx.focus_metric === "error") return s.max_tp_error || 0;
  return targetMetric(s, state.compare ? "delta_fp" : "fp");
}
function scenarioMetric(s, lens = state.lens, label = state.label) {
  if (label) {
    return metricFromRow(labelRow(s, label), lens);
  }
  if (lens === "intent_risk") return intentRiskMetric(s);
  if (lens === "target_fn") return targetMetric(s, state.compare ? "delta_fn" : "fn");
  if (lens === "target_fp") return targetMetric(s, state.compare ? "delta_fp" : "fp");
  return metricFromRow(s, lens);
}
function activeRunStats(s) {
  return state.compare ? emptyStats(s.b || {}) : s;
}
function compareRunAvailable(row) {
  return Boolean(row && (row.scenario_name || row.t4dataset_name || row.rows || row.frames || (row.devops && row.devops.is_devops)));
}
function scenarioRunView(s, side) {
  const run = side === "a" ? s.a : s.b;
  if (!state.compare) return s;
  if (!compareRunAvailable(run)) {
    return {
      ...emptyStats(),
      suite_name: s.suite_name,
      scenario_name: s.scenario_name,
      t4dataset_name: s.t4dataset_name,
      topic_name: s.topic_name,
      devops: {...devopsContext(s), unavailable: true, unavailable_reason: side === "a" ? "Missing in Run A." : "Missing in Run B."},
    };
  }
  return {...s, ...run, a: s.a, b: s.b, labels: run.labels || [], devops: (run.devops && run.devops.is_devops) ? run.devops : devopsContext(s)};
}
function scenarioJudgementForRun(s, side) {
  return scenarioJudgement(scenarioRunView(s, side));
}
function scenarioCompareSummary(s) {
  if (!state.compare) return {changed: false, fixed: false, regressed: false, added: false, removed: false, magnitude: 0};
  const hasA = compareRunAvailable(s.a);
  const hasB = compareRunAvailable(s.b);
  const aJudgement = scenarioJudgementForRun(s, "a");
  const bJudgement = scenarioJudgementForRun(s, "b");
  const aBad = aJudgement.status === "fail" || aJudgement.status === "review";
  const bBad = bJudgement.status === "fail" || bJudgement.status === "review";
  const deltaFp = Number(s.delta_fp || 0);
  const deltaFn = Number(s.delta_fn || 0);
  const deltaTp = Number(s.delta_tp || 0);
  const magnitude = Math.abs(deltaFp) + Math.abs(deltaFn) + Math.abs(deltaTp);
  const added = !hasA && hasB;
  const removed = hasA && !hasB;
  const fixed = hasA && hasB && aBad && bJudgement.status === "pass";
  const regressed = hasA && hasB && aJudgement.status === "pass" && bBad;
  const changed = added || removed || fixed || regressed || aJudgement.status !== bJudgement.status || magnitude > 0;
  return {a: aJudgement, b: bJudgement, changed, fixed, regressed, added, removed, magnitude, deltaFp, deltaFn, deltaTp};
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
    out.devops = (bb.devops && bb.devops.is_devops) ? bb.devops : ((aa.devops && aa.devops.is_devops) ? aa.devops : inferDevopsContext(out));
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
function devopsPurposeText(s) {
  const ctx = devopsContext(s);
  if (!ctx.is_devops) return "";
  return [ctx.intent_type, ctx.target_label, ctx.behavior, ctx.pc_mode].filter(Boolean).join(" · ");
}
function devopsQuickRead(s) {
  const ctx = devopsContext(s);
  if (!ctx.is_devops) return "No devops intent metadata was inferred for this scenario.";
  const target = ctx.target_label || "target";
  const tp = targetMetric(s, "tp");
  const fp = targetMetric(s, "fp");
  const fn = targetMetric(s, "fn");
  if (ctx.focus_metric === "fn") return `${target}: ${fmt(fn)} FN, ${fmt(tp)} TP. Misses are the first thing to inspect.`;
  if (ctx.focus_metric === "error") return `${target}: max TP error ${Number(s.max_tp_error || 0).toFixed(2)} m. Inspect pose/yaw alignment.`;
  return `${target}: ${fmt(fp)} FP, ${fmt(tp)} TP. Extra detections / false stop evidence are the first thing to inspect.`;
}
function scenarioJudgement(s) {
  const ctx = devopsContext(s);
  if (ctx.unavailable) return {status: "missing", label: "MISSING", reason: ctx.unavailable_reason || "Unavailable in this parquet."};
  if (!ctx.is_devops) return {status: "review", label: "REVIEW", reason: "No DevOps intent metadata."};
  const criteriaResult = ctx.criteria_result || {};
  if (criteriaResult.overall_pass === true) {
    return {status: "pass", label: "PASS", reason: criteriaResult.explanation || "criteria pass"};
  }
  if (criteriaResult.overall_pass === false) {
    return {status: "fail", label: "FAIL", reason: criteriaResult.explanation || `${fmt(criteriaResult.failed_count || 1)} criteria fail`};
  }
  const target = ctx.target_label || "target";
  const tp = targetMetric(s, "tp");
  const fp = targetMetric(s, "fp");
  const fn = targetMetric(s, "fn");
  const needsCriteria = (ctx.focus_metric === "fn" && fn > 0)
    || (ctx.focus_metric === "fp" && fp > 0)
    || (ctx.focus_metric === "error" && (s.max_tp_error || 0) > 0);
  if (needsCriteria) return {status: "review", label: "CHECK", reason: "select to evaluate criteria"};
  if (ctx.focus_metric === "fn") {
    return fn > 0
      ? {status: "fail", label: "FAIL", reason: `${fmt(fn)} ${target} FN`}
      : {status: "pass", label: "PASS", reason: `${fmt(tp)} ${target} TP, no target FN`};
  }
  if (ctx.focus_metric === "fp") {
    return fp > 0
      ? {status: "fail", label: "FAIL", reason: `${fmt(fp)} ${target} FP`}
      : {status: "pass", label: "PASS", reason: `no ${target} FP`};
  }
  if (ctx.focus_metric === "error") {
    return (s.max_tp_error || 0) > 0
      ? {status: "review", label: "CHECK", reason: `max TP error ${Number(s.max_tp_error || 0).toFixed(2)} m`}
      : {status: "review", label: "CHECK", reason: "needs exact error gate"};
  }
  return {status: "review", label: "REVIEW", reason: devopsQuickRead(s)};
}
function devopsSuiteGroups(items) {
  const map = new Map();
  for (const s of items || []) {
    const suite = s.suite_name || "Unknown suite";
    const key = suiteBaseName(suite);
    const ctx = devopsContext(s);
    const group = map.get(key) || {
      key, suite, ctx, items: [], pass: 0, fail: 0, review: 0, missing: 0, suitePass: ctx.suite_pass || null,
      aPass: 0, aFail: 0, aReview: 0, aMissing: 0, bPass: 0, bFail: 0, bReview: 0, bMissing: 0,
      changed: 0, fixed: 0, regressed: 0, added: 0, removed: 0, changeMagnitude: 0,
      suitePassA: state.compare && s.a && s.a.devops ? s.a.devops.suite_pass : null,
      suitePassB: state.compare && s.b && s.b.devops ? s.b.devops.suite_pass : null,
    };
    const judgement = scenarioJudgement(s);
    group.items.push(s);
    if (judgement.status === "pass") group.pass += 1;
    else if (judgement.status === "fail") group.fail += 1;
    else if (judgement.status === "missing") group.missing += 1;
    else group.review += 1;
    if (!group.suitePass && ctx.suite_pass) group.suitePass = ctx.suite_pass;
    if (state.compare) {
      const aJudgement = scenarioJudgementForRun(s, "a");
      const bJudgement = scenarioJudgementForRun(s, "b");
      if (aJudgement.status === "pass") group.aPass += 1;
      else if (aJudgement.status === "fail") group.aFail += 1;
      else if (aJudgement.status === "missing") group.aMissing += 1;
      else group.aReview += 1;
      if (bJudgement.status === "pass") group.bPass += 1;
      else if (bJudgement.status === "fail") group.bFail += 1;
      else if (bJudgement.status === "missing") group.bMissing += 1;
      else group.bReview += 1;
      const cmp = scenarioCompareSummary(s);
      if (cmp.changed) group.changed += 1;
      if (cmp.fixed) group.fixed += 1;
      if (cmp.regressed) group.regressed += 1;
      if (cmp.added) group.added += 1;
      if (cmp.removed) group.removed += 1;
      group.changeMagnitude += cmp.magnitude;
      if (!group.suitePassA && s.a && s.a.devops && s.a.devops.suite_pass) group.suitePassA = s.a.devops.suite_pass;
      if (!group.suitePassB && s.b && s.b.devops && s.b.devops.suite_pass) group.suitePassB = s.b.devops.suite_pass;
    }
    map.set(key, group);
  }
  return [...map.values()].sort((a, b) => {
    if (state.compare) {
      return b.regressed - a.regressed
        || b.fixed - a.fixed
        || b.changed - a.changed
        || b.changeMagnitude - a.changeMagnitude
        || a.key.localeCompare(b.key);
    }
    const ar = a.suitePass ? a.suitePass.pass_rate : (a.pass / Math.max(1, a.items.length));
    const br = b.suitePass ? b.suitePass.pass_rate : (b.pass / Math.max(1, b.items.length));
    return ar - br || b.items.length - a.items.length || a.key.localeCompare(b.key);
  });
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
