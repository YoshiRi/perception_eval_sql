var state = {
  parquets: [],
  path: "",
  pathB: "",
  compare: false,
  filters: {},
  runNames: {A: "Run A", B: "Run B"},
  scenarios: [],
  selectedScenario: null,
  frames: [],
  framePos: 0,
  playing: false,
  lastPlayTs: 0,
  playCarryMs: 0,
  autoLoadTimer: null,
  yaw: 0.588,
  pitch: 0.302,
  distance: 92,
  panX: 0,
  panY: 0,
  dragging: false,
  dragButton: 0,
  // Latched at pointerdown so a modifier pressed mid-drag cannot flip pan <-> orbit.
  dragMode: "orbit",
  // Timestamp of the last drag end: the middle button is the scroll wheel, so it keeps
  // emitting wheel ticks for a moment after the drag, and those must not dolly.
  dragEndedAt: 0,
  dragViewport: null,
  lastX: 0,
  lastY: 0,
  downX: 0,
  downY: 0,
  draggingCurtain: false,
  curtainX: 0.5,
  trails: false,
  hotspotMode: "all",
  bounds: {maxAbs: 80},
  selected: null,
  labelChipsTouched: false,
  labelSignature: "",
  labelSceneKey: "",
  hover: null,
  mouse: {x: -9999, y: -9999},
  devopsResult: null,
  deepLink: null
};
var $ = (id) => document.getElementById(id);
var els = {
  root: $("rootInput"), scan: $("scanBtn"), parquet: $("parquetSelect"), search: $("searchInput"),
  parquetB: $("parquetSelectB"), compareEnabled: $("compareEnabled"),
  suite: $("suiteFilter"), scenario: $("scenarioFilter"), dataset: $("datasetFilter"), topic: $("topicFilter"),
  frameMin: $("frameMin"), frameMax: $("frameMax"), statuses: $("statusChips"), sources: $("sourceChips"), labels: $("labelChips"),
  viewMode: $("viewMode"), colorMode: $("colorMode"), labelMode: $("labelMode"), confMin: $("confMin"), boxOpacity: $("boxOpacity"), compareLens: $("compareLens"), compareLayout: $("compareLayout"),
  layerChips: $("layerChips"),
  colorCycleBtn: $("colorCycleBtn"), labelCycleBtn: $("labelCycleBtn"), compareLensBtn: $("compareLensBtn"), advancedBtn: $("advancedBtn"), advancedPanel: $("advancedPanel"),
  labelsAllBtn: $("labelsAllBtn"), labelsNoneBtn: $("labelsNoneBtn"),
  showVelocity: $("showVelocity"), showRings: $("showRings"), showErrors: $("showErrors"), dedupeRows: $("dedupeRows"),
  results: $("results"), load: $("loadBtn"), canvas: $("canvas"), slider: $("frameSlider"), play: $("playBtn"),
  prevFrame: $("prevFrameBtn"), nextFrame: $("nextFrameBtn"), playSpeed: $("playSpeed"),
  readout: $("frameReadout"), boxCount: $("boxCount"), frameCount: $("frameCount"), gtCount: $("gtCount"), estCount: $("estCount"),
  toast: $("toast"), spinner: $("spinner"), camTop: $("camTop"), cam3d: $("cam3d"), camFollow: $("camFollow"), compareSideBtn: $("compareSideBtn"), compareCurtainBtn: $("compareCurtainBtn"), resetCamera: $("resetCamera"), fitCamera: $("fitCamera"), toggleTrails: $("toggleTrails"),
  toggleSidebar: $("toggleSidebar"), fullscreenBtn: $("fullscreenBtn"), loadStatus: $("loadStatus"), themeToggle: $("themeToggleBtn"),
  inspect: $("inspectPanel"), inspectTitle: $("inspectTitle"), inspectStatus: $("inspectStatus"), inspectPos: $("inspectPos"),
  inspectSize: $("inspectSize"), inspectConf: $("inspectConf"), inspectDist: $("inspectDist"), inspectErr: $("inspectErr"),
  inspectPeer: $("inspectPeer"), inspectUuid: $("inspectUuid"), hoverCard: $("hoverCard"), heatCanvas: $("heatCanvas"), frameCurve: $("frameCurveCanvas"), overview: $("overviewCanvas"),
  analysisFrame: $("analysisFrame"), analysisTp: $("analysisTp"), analysisFp: $("analysisFp"), analysisFn: $("analysisFn"),
  analysisErr: $("analysisErr"), analysisRecall: $("analysisRecall"), analysisPrecision: $("analysisPrecision"),
  analysisGt: $("analysisGt"), analysisEst: $("analysisEst"), prevHotFrame: $("prevHotFrame"), nextHotFrame: $("nextHotFrame"),
  labelFrameBreakdown: $("labelFrameBreakdown"), labelFrameMeta: $("labelFrameMeta"), zoneBreakdown: $("zoneBreakdown"), hotspotModes: $("hotspotModes"), hotspotModeMeta: $("hotspotModeMeta"),
  cntGtTp: $("cntGtTp"), cntGtFn: $("cntGtFn"), cntEstTp: $("cntEstTp"), cntEstFp: $("cntEstFp"),
  compareSummary: $("compareSummary"), compareAStats: $("compareAStats"), compareBStats: $("compareBStats"), compareDeltaStats: $("compareDeltaStats"), compareFrameNote: $("compareFrameNote"),
  devopsViewerPanel: $("devopsViewerPanel"),
  compareBanner: $("compareBanner"), compareBannerText: $("compareBannerText"), splitLabelA: $("splitLabelA"), splitLabelB: $("splitLabelB"), curtainHandle: $("curtainHandle")
};
var ctx = els.canvas.getContext("2d");
var frameCurveCtx = els.frameCurve.getContext("2d");
var overviewCtx = els.overview.getContext("2d");
var edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
var activeViewport = null;
state.deepLink = readDeepLink();
var BBOX_VIEWER_PREFS_KEY = "bbox.viewer.status.v1";
var viewerPrefsSaveTimer = null;

function readViewerPrefs() {
  try {
    const raw = window.localStorage.getItem(BBOX_VIEWER_PREFS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch (_err) {
    return {};
  }
}
state.viewerPrefs = readViewerPrefs();

function selectHasValue(select, value) {
  return Boolean(select && [...select.options].some(option => option.value === value));
}
function finiteNumber(value, fallback = null) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}
function applyViewerPrefs({camera = false, controls = false} = {}) {
  const prefs = state.viewerPrefs || {};
  if (controls) {
    if (selectHasValue(els.viewMode, prefs.viewMode)) els.viewMode.value = prefs.viewMode;
    if (selectHasValue(els.colorMode, prefs.colorMode)) els.colorMode.value = prefs.colorMode;
    else if (prefs.colorMode === "run") els.colorMode.value = "status";
    if (selectHasValue(els.labelMode, prefs.labelMode)) els.labelMode.value = prefs.labelMode;
    if (selectHasValue(els.compareLens, prefs.compareLens)) els.compareLens.value = prefs.compareLens;
    if (selectHasValue(els.compareLayout, prefs.compareLayout)) els.compareLayout.value = prefs.compareLayout;
    for (const [el, key] of [[els.showVelocity, "showVelocity"], [els.showRings, "showRings"], [els.showErrors, "showErrors"], [els.dedupeRows, "dedupeRows"]]) {
      if (el && typeof prefs[key] === "boolean") el.checked = prefs[key];
    }
    if (els.boxOpacity && prefs.boxOpacity != null) els.boxOpacity.value = String(prefs.boxOpacity);
    if (els.confMin && prefs.confMin != null) els.confMin.value = String(prefs.confMin);
    if (els.playSpeed && prefs.playSpeed != null) els.playSpeed.value = String(prefs.playSpeed);
    if (els.advancedPanel && typeof prefs.advancedOpen === "boolean") els.advancedPanel.classList.toggle("show", prefs.advancedOpen);
    if (typeof prefs.trails === "boolean") state.trails = prefs.trails;
    if (typeof prefs.hotspotMode === "string") state.hotspotMode = prefs.hotspotMode || "all";
    if (prefs.curtainX != null) state.curtainX = Math.max(0.08, Math.min(0.92, finiteNumber(prefs.curtainX, state.curtainX)));
    if (Array.isArray(prefs.activeLayers)) {
      const active = new Set(prefs.activeLayers);
      els.layerChips.querySelectorAll(".chip").forEach(chip => chip.classList.toggle("active", active.has(chip.dataset.layer)));
    }
    els.hotspotModes.querySelectorAll("button").forEach(btn => btn.classList.toggle("active", btn.dataset.hotspot === state.hotspotMode));
  }
  if (camera && prefs.camera) {
    const yaw = finiteNumber(prefs.camera.yaw);
    const pitch = finiteNumber(prefs.camera.pitch);
    const distance = finiteNumber(prefs.camera.distance);
    const panX = finiteNumber(prefs.camera.panX);
    const panY = finiteNumber(prefs.camera.panY);
    if (yaw != null) state.yaw = yaw;
    if (pitch != null) state.pitch = pitch;
    if (distance != null) state.distance = Math.max(BBOX_VIEWER_MIN_DISTANCE || 0, distance);
    if (panX != null) state.panX = panX;
    if (panY != null) state.panY = panY;
  }
}
function captureViewerPrefs() {
  return {
    viewMode: els.viewMode.value,
    colorMode: els.colorMode.value === "run" ? "status" : els.colorMode.value,
    labelMode: els.labelMode.value,
    compareLens: els.compareLens.value,
    compareLayout: els.compareLayout.value,
    showVelocity: els.showVelocity.checked,
    showRings: els.showRings.checked,
    showErrors: els.showErrors.checked,
    dedupeRows: els.dedupeRows.checked,
    boxOpacity: els.boxOpacity.value,
    confMin: els.confMin.value,
    playSpeed: els.playSpeed.value,
    advancedOpen: els.advancedPanel.classList.contains("show"),
    trails: state.trails,
    hotspotMode: state.hotspotMode,
    curtainX: state.curtainX,
    activeLayers: [...els.layerChips.querySelectorAll(".chip.active")].map(chip => chip.dataset.layer).filter(Boolean),
    camera: {
      yaw: state.yaw,
      pitch: state.pitch,
      distance: state.distance,
      panX: state.panX,
      panY: state.panY
    }
  };
}
function saveViewerPrefs() {
  try {
    state.viewerPrefs = captureViewerPrefs();
    window.localStorage.setItem(BBOX_VIEWER_PREFS_KEY, JSON.stringify(state.viewerPrefs));
  } catch (_err) {
    /* private mode or storage quota: ignore */
  }
}
function scheduleViewerPrefsSave() {
  clearTimeout(viewerPrefsSaveTimer);
  viewerPrefsSaveTimer = setTimeout(saveViewerPrefs, 120);
}

function shortPathName(path) {
  const raw = String(path || "").split(/[\\/]/).filter(Boolean).slice(-2).join("/");
  return raw || "selected run";
}
function refreshRunNames() {
  state.runNames = {
    A: shortPathName(state.path || els.parquet.value),
    B: shortPathName(els.parquetB.value || state.pathB)
  };
}
function compareLensLabel(value = els.compareLens.value) {
  return {
    all: "All",
    a_only: "A only",
    b_only: "B only",
    changed_only: "Changed only",
    new_fp_b: "New FP in B",
    resolved_fn_b: "Resolved FN in B",
    worse_tp_b: "TP worse in B",
    better_tp_b: "TP better in B"
  }[value] || "All";
}
function updateCompareBanner() {
  if (!els.compareBanner) return;
  refreshRunNames();
  els.compareBanner.classList.toggle("show", state.compare);
  if (!state.compare) return;
  if (els.splitLabelA) els.splitLabelA.textContent = `A: ${state.runNames.A}`;
  if (els.splitLabelB) els.splitLabelB.textContent = `B: ${state.runNames.B}`;
  els.compareBannerText.textContent = `A: ${state.runNames.A}  vs  B: ${state.runNames.B} · ${compareLensLabel()} · ${compareLayoutMode().replace(/_/g, " ")}`;
}
