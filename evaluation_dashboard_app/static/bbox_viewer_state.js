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
