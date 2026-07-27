var $ = id => document.getElementById(id);
var state = {
  parquets: [], path: "", pathB: "", compare: false, scenarios: [], labels: [], selected: null, curve: [], stats: null,
  previewFrames: [], previewIndex: 0, previewVisible: false, previewDrag: null, previewResize: null,
  previewPanX: 0, previewPanY: 0, previewScale: 1, previewPanning: false, previewLastX: 0, previewLastY: 0,
  previewHoverBox: null, previewMouseX: 0, previewMouseY: 0, previewShowRings: true, previewShowLabels: false,
  summaryRequestId: 0, curveRequestId: 0, previewRequestId: 0,
  lens: "fp", label: "", rangeMax: "", layout: "galaxy", stageView: "map", statsDistanceStyle: "line", statsFrameFocus: "degraded", scale: 1, panX: 0, panY: 0,
  dragging: false, lastX: 0, lastY: 0, downX: 0, downY: 0, hover: null, hoverLabel: null, labelNodes: [], statNodes: [],
  statsHover: null, statsDetail: null, viewerUrl: "", mouseX: 0, mouseY: 0
};
var els = {
  root: $("rootInput"), scan: $("scanBtn"), parquet: $("parquetSelect"), parquetB: $("parquetSelectB"), compareEnabled: $("compareEnabled"), topic: $("topicFilter"),
  search: $("searchInput"), lensChips: $("lensChips"), labelChips: $("labelChips"), rangeChips: $("rangeChips"),
  list: $("scenarioList"), canvas: $("mapCanvas"), stage: $("stage"), toast: $("toast"), legend: $("mapLegend"),
  kScenario: $("kScenario"), kFp: $("kFp"), kFn: $("kFn"), kScenarioLabel: $("kScenarioLabel"), kFpLabel: $("kFpLabel"), kFnLabel: $("kFnLabel"), hudLens: $("hudLens"), hudMax: $("hudMax"), hudLabel: $("hudLabel"), hudRange: $("hudRange"),
  layoutGalaxy: $("layoutGalaxy"), layoutStats: $("layoutStats"), statsDistanceStyle: $("statsDistanceStyle"), statsFrameFocus: $("statsFrameFocus"), resetView: $("resetView"),
  selTitle: $("selTitle"), selMeta: $("selMeta"), selFp: $("selFp"), selFn: $("selFn"), selPrecision: $("selPrecision"), selRecall: $("selRecall"),
  previewWindow: $("previewWindow"), previewTitlebar: $("previewTitlebar"), previewTitle: $("previewTitle"), previewResize: $("previewResize"),
  preview: $("previewCanvas"), previewStatus: $("previewStatus"), previewSlider: $("previewSlider"), previewFit: $("previewFitBtn"), previewRings: $("previewRingsBtn"), previewLabels: $("previewLabelsBtn"), previewOpen: $("previewOpenBtn"), previewClose: $("previewCloseBtn"),
  previewLayers: $("previewLayers"),
  curve: $("curveCanvas"), curveStatus: $("curveStatus"), openViewer: $("openViewerBtn"), labelBreakdown: $("labelBreakdown"),
  nearPed: $("nearPedBtn"), nearPedFp: $("nearPedFpBtn"), hoverCard: $("hoverCard"), compareBanner: $("compareBanner"), compareBannerText: $("compareBannerText"),
  statsDetail: $("statsDetail"), statsDetailTitle: $("statsDetailTitle"), statsDetailMeta: $("statsDetailMeta"), statsDetailTable: $("statsDetailTable"),
  statsDetailClose: $("statsDetailClose"), statsDetailDownload: $("statsDetailDownload"),
  viewerShell: $("viewerShell"), viewerFrame: $("viewerFrame"), viewerShellTitle: $("viewerShellTitle"), viewerShellMeta: $("viewerShellMeta"),
  viewerClose: $("viewerCloseBtn"), viewerNewTab: $("viewerNewTabBtn")
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
