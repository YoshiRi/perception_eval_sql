var $ = id => document.getElementById(id);
var state = {
  parquets: [], path: "", pathB: "", compare: false, scenarios: [], labels: [], selected: null, curve: [], stats: null, devopsResult: null, devopsFrameResults: null,
  // Cached T4 scenes by dataset id, or null where /viewer/three is not served (dashboard).
  t4Scenes: null,
  // Probe of the T4 server, fetched once and only when an uncached scene is selected;
  // and the fetch job in flight, if any.
  // Last sized scene, which the Download button then offers by name and size: the
  // estimate the user agreed to, not a modal they clicked through.
  t4Server: null, t4Job: null, t4Estimate: null,
  // True once /api/client answers: this page is served by the desktop client, whose
  // window cannot open tabs, so links go out through its process instead.
  isLocalClient: false,
  // Origin of the dashboard's Streamlit pages, for its server-side 3D viewer. Empty
  // string means same origin, which is the case when the dashboard serves this page.
  t4Dashboard: "",
  previewFrames: [], previewIndex: 0, previewVisible: false, previewDrag: null, previewResize: null,
  previewPanX: 0, previewPanY: 0, previewScale: 1, previewPanning: false, previewLastX: 0, previewLastY: 0,
  previewHoverBox: null, previewMouseX: 0, previewMouseY: 0, previewShowRings: true, previewShowLabels: false,
  summaryRequestId: 0, curveRequestId: 0, previewRequestId: 0, resultRequestId: 0,
  lens: "fp", label: "", rangeMax: "", explorerMode: "hotspots", expandedSuites: new Set(), layout: "galaxy", stageView: "map", statsDistanceStyle: "line", statsFrameFocus: "degraded", scale: 1, panX: 0, panY: 0,
  dragging: false, dragEndedAt: 0, lastX: 0, lastY: 0, downX: 0, downY: 0, hover: null, hoverLabel: null, labelNodes: [], statNodes: [],
  statsHover: null, statsDetail: null, devopsCanvasHits: [], devopsHoverHit: null, devopsCanvasScroll: 0, devopsCanvasMaxScroll: 0, viewerUrl: "", mouseX: 0, mouseY: 0,
  restoreSession: null, restoreSessionApplied: false, savingSession: false
};
var els = {
  root: $("rootInput"), scan: $("scanBtn"), parquet: $("parquetSelect"), parquetB: $("parquetSelectB"), compareEnabled: $("compareEnabled"), topic: $("topicFilter"),
  search: $("searchInput"), modeChips: $("modeChips"), lensChips: $("lensChips"), labelChips: $("labelChips"), rangeChips: $("rangeChips"),
  list: $("scenarioList"), canvas: $("mapCanvas"), stage: $("stage"), toast: $("toast"), legend: $("mapLegend"),
  scenarioListTitle: $("scenarioListTitle"),
  kScenario: $("kScenario"), kFp: $("kFp"), kFn: $("kFn"), kScenarioLabel: $("kScenarioLabel"), kFpLabel: $("kFpLabel"), kFnLabel: $("kFnLabel"), hudLens: $("hudLens"), hudMax: $("hudMax"), hudLabel: $("hudLabel"), hudRange: $("hudRange"),
  layoutGalaxy: $("layoutGalaxy"), layoutReview: $("layoutReview"), layoutStats: $("layoutStats"), statsDistanceStyle: $("statsDistanceStyle"), statsFrameFocus: $("statsFrameFocus"), resetView: $("resetView"),
  selTitle: $("selTitle"), selMeta: $("selMeta"), selFp: $("selFp"), selFn: $("selFn"), selPrecision: $("selPrecision"), selRecall: $("selRecall"),
  previewWindow: $("previewWindow"), previewTitlebar: $("previewTitlebar"), previewTitle: $("previewTitle"), previewResize: $("previewResize"),
  preview: $("previewCanvas"), previewStatus: $("previewStatus"), previewSlider: $("previewSlider"), previewFit: $("previewFitBtn"), previewRings: $("previewRingsBtn"), previewLabels: $("previewLabelsBtn"), previewOpen: $("previewOpenBtn"), previewClose: $("previewCloseBtn"),
  previewLayers: $("previewLayers"), previewDevopsOverlay: $("previewDevopsOverlay"),
  curve: $("curveCanvas"), curveStatus: $("curveStatus"), openViewer: $("openViewerBtn"), open3d: $("open3dBtn"),
  get3d: $("get3dBtn"), open3dServer: $("open3dServerBtn"), t4FetchStatus: $("t4FetchStatus"), labelBreakdown: $("labelBreakdown"),
  intentPanel: $("intentPanel"), resultPanel: $("resultPanel"),
  devopsIntent: $("devopsIntentBtn"), nearPed: $("nearPedBtn"), nearPedFp: $("nearPedFpBtn"), animal: $("animalBtn"), falseStop: $("falseStopBtn"),
  hoverCard: $("hoverCard"), compareBanner: $("compareBanner"), compareBannerText: $("compareBannerText"),
  statsDetail: $("statsDetail"), statsDetailTitle: $("statsDetailTitle"), statsDetailMeta: $("statsDetailMeta"), statsDetailTable: $("statsDetailTable"),
  statsDetailClose: $("statsDetailClose"), statsDetailDownload: $("statsDetailDownload"),
  viewerShell: $("viewerShell"), viewerFrame: $("viewerFrame"), viewerShellTitle: $("viewerShellTitle"), viewerShellMeta: $("viewerShellMeta"),
  viewerClose: $("viewerCloseBtn"), viewerNewTab: $("viewerNewTabBtn"),
  themeToggle: $("themeToggleBtn")
};
var ctx = els.canvas.getContext("2d");
var curveCtx = els.curve.getContext("2d");
var previewCtx = els.preview.getContext("2d");

function toast(msg) {
  els.toast.textContent = msg;
  els.toast.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => els.toast.classList.remove("show"), 2600);
}
function escapeHtml(v) {
  return String(v ?? "").replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function rate(v) { return Number.isFinite(Number(v)) ? `${Math.round(Number(v) * 100)}%` : "-"; }
function fmt(n) { return Number(n || 0).toLocaleString(); }
function fmtBytes(n) {
  if (n === null || n === undefined) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = Number(n) || 0, i = 0;
  while (Math.abs(v) >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return i === 0 ? `${v.toFixed(0)} B` : `${v.toFixed(1)} ${units[i]}`;
}
function fmtDelta(n) {
  const v = Number(n || 0);
  return `${v > 0 ? "+" : ""}${v.toLocaleString()}`;
}
function shortPathName(path) {
  const raw = String(path || "").split(/[\\/]/).filter(Boolean).slice(-2).join("/");
  return raw || "selected run";
}
function compareLensLabel(value = state.lens) {
  return {
    count: "Count",
    fp: "FP",
    fn: "FN",
    fpr: "FPR",
    fnr: "FNR",
    error: "TP error",
    intent_risk: "Intent risk",
    target_fn: "Target FN",
    target_fp: "Target FP",
    changed_only: "Changed only",
    delta_fp: "FP change",
    delta_fn: "FN change",
    regression: "More FP/FN in B"
  }[value] || String(value || "FP").toUpperCase();
}
function lensMetricText(value, lens = state.lens) {
  if (lens.endsWith("r")) return rate(value);
  if (state.compare && lens !== "changed_only") return fmtDelta(Math.round(value));
  return fmt(Math.round(value));
}
function updateCompareBanner() {
  if (!els.compareBanner) return;
  els.compareBanner.classList.toggle("show", state.compare);
  if (!state.compare) return;
  els.compareBannerText.textContent = `A: ${shortPathName(state.path)}  vs  B: ${shortPathName(els.parquetB.value || state.pathB)} · ${compareLensLabel()}`;
}
function setBusy(on, message = "") {
  document.querySelector(".app").classList.toggle("is-busy", Boolean(on));
  if (message) toast(message);
}
