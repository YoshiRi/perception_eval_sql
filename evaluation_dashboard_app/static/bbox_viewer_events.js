function loop() {
  const now = performance.now();
  if (!state.lastPlayTs) state.lastPlayTs = now;
  const dt = now - state.lastPlayTs;
  state.lastPlayTs = now;
  if (state.playing && state.frames.length) {
    const fps = Math.max(0.25, Number(els.playSpeed.value || 2));
    state.playCarryMs += dt;
    const frameMs = 1000 / fps;
    let advanced = false;
    while (state.playCarryMs >= frameMs) {
      state.framePos = (state.framePos + 1) % state.frames.length;
      state.selected = null;
      state.playCarryMs -= frameMs;
      advanced = true;
      if (state.playCarryMs > frameMs * 4) state.playCarryMs = 0;
    }
    if (advanced) { updateInspect(); render(); }
  }
  requestAnimationFrame(loop);
}
var colorModes = [
  ["status", "Color: Eval"],
  ["error", "Color: Error"],
  ["confidence", "Color: Conf"],
  ["source", "Color: Source"],
  ["run", "Color: Run"]
];
var labelModes = [
  ["label_status", "Label: Class"],
  ["label_conf", "Label: Conf"],
  ["uuid", "Label: UUID"],
  ["none", "Label: Off"]
];
var compareLensModes = [
  ["all", "Lens: All"],
  ["changed_only", "Lens: Changed"],
  ["new_fp_b", "Lens: New FP"],
  ["resolved_fn_b", "Lens: Resolved FN"],
  ["worse_tp_b", "Lens: Worse TP"],
  ["better_tp_b", "Lens: Better TP"],
  ["a_only", "Lens: A only"],
  ["b_only", "Lens: B only"]
];
function syncCycleButtons() {
  const color = colorModes.find(([value]) => value === els.colorMode.value) || colorModes[0];
  const label = labelModes.find(([value]) => value === els.labelMode.value) || labelModes[0];
  const lens = compareLensModes.find(([value]) => value === els.compareLens.value) || compareLensModes[0];
  els.colorCycleBtn.textContent = color[1];
  els.labelCycleBtn.textContent = label[1];
  els.compareLensBtn.textContent = lens[1];
  els.compareLensBtn.classList.toggle("active", state.compare && els.compareLens.value !== "all");
  els.advancedBtn.classList.toggle("active", els.advancedPanel.classList.contains("show"));
  updateCompareBanner();
  syncCompareIssueLabels();
}
function cycleSelect(select, modes) {
  const idx = Math.max(0, modes.findIndex(([value]) => value === select.value));
  select.value = modes[(idx + 1) % modes.length][0];
  syncCycleButtons();
  render();
}
function syncCompareIssueLabels() {
  const labels = state.compare
    ? {all: "All changes", fp: "More FP in B", fn: "More FN in B", ped_fp: "Ped FP", near: "Nearby", tp_error: "TP worse in B"}
    : {all: "All issues", fp: "FP", fn: "FN", ped_fp: "Ped FP", near: "Nearby", tp_error: "TP error"};
  els.hotspotModes.querySelectorAll("button").forEach(btn => {
    btn.textContent = labels[btn.dataset.hotspot] || btn.textContent;
  });
  els.prevHotFrame.title = state.compare ? "Previous frame with a B-A change" : "Previous frame with FP/FN or high TP error";
  els.nextHotFrame.title = state.compare ? "Next frame with a B-A change" : "Next frame with FP/FN or high TP error";
}
els.scan.addEventListener("click", scan);
els.parquet.addEventListener("change", () => { clearSceneWindow(); hydrateFilters().then(searchScenarios); });
els.parquetB.addEventListener("change", () => { state.pathB = els.parquetB.value; if (els.compareEnabled.checked) scheduleSceneReload("Run B"); });
els.compareEnabled.addEventListener("change", () => { updateCompareControls(); scheduleSceneReload("comparison mode"); });
els.search.addEventListener("input", () => { clearTimeout(els.search._t); els.search._t = setTimeout(searchScenarios, 220); });
for (const el of [els.suite, els.topic]) el.addEventListener("change", async () => {
  els.scenario.value = "";
  clearSceneWindow();
  await refreshDependentFilters();
  setStatus("<b>Scene list updated.</b> Click one scenario card to load its frames.");
});
els.scenario.addEventListener("change", () => { searchScenarios(); scheduleSceneReload("scene"); });
els.load.addEventListener("click", loadScene);
els.slider.addEventListener("input", e => { state.framePos = Number(e.target.value) || 0; state.selected = null; updateInspect(); render(); });
els.prevFrame.addEventListener("click", () => stepFrame(-1));
els.nextFrame.addEventListener("click", () => stepFrame(1));
els.play.addEventListener("click", () => { state.playing = !state.playing; state.playCarryMs = 0; state.lastPlayTs = performance.now(); els.play.textContent = state.playing ? "Ⅱ" : "▶"; });
els.playSpeed.addEventListener("change", () => { state.playCarryMs = 0; });
els.prevHotFrame.addEventListener("click", () => jumpHotspot(-1));
els.nextHotFrame.addEventListener("click", () => jumpHotspot(1));
els.hotspotModes.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
  state.hotspotMode = btn.dataset.hotspot || "all";
  els.hotspotModes.querySelectorAll("button").forEach(x => x.classList.toggle("active", x === btn));
  renderHeatStrip();
  renderFrameCurve();
  render();
}));
els.camTop.addEventListener("click", () => setCameraPreset("top"));
els.cam3d.addEventListener("click", () => setCameraPreset("3d"));
els.camFollow.addEventListener("click", () => setCameraPreset("follow"));
function setCompareLayout(mode, options = {}) {
  els.compareLayout.value = mode;
  els.compareSideBtn.classList.toggle("active", mode === "side_by_side");
  els.compareCurtainBtn.classList.toggle("active", mode === "curtain");
  updateCompareBanner();
  if (!options.skipRender) render();
}
els.compareSideBtn.addEventListener("click", () => setCompareLayout("side_by_side"));
els.compareCurtainBtn.addEventListener("click", () => setCompareLayout("curtain"));
els.colorCycleBtn.addEventListener("click", () => cycleSelect(els.colorMode, colorModes));
els.labelCycleBtn.addEventListener("click", () => cycleSelect(els.labelMode, labelModes));
els.compareLensBtn.addEventListener("click", () => { if (state.compare) cycleSelect(els.compareLens, compareLensModes); });
els.advancedBtn.addEventListener("click", () => {
  els.advancedPanel.classList.toggle("show");
  syncCycleButtons();
});
els.resetCamera.addEventListener("click", () => { state.selected = null; updateInspect(); setCameraPreset("3d"); });
els.fitCamera.addEventListener("click", () => { fitBounds(); render(); });
els.toggleTrails.addEventListener("click", () => { state.trails = !state.trails; els.toggleTrails.style.background = state.trails ? TH.a("accent", .36) : ""; render(); });
els.toggleSidebar.addEventListener("click", () => {
  document.querySelector(".app").classList.toggle("sidebar-collapsed");
  setTimeout(render, 50);
});
els.fullscreenBtn.addEventListener("click", async () => {
  const root = document.documentElement;
  try {
    if (!document.fullscreenElement) await root.requestFullscreen();
    else await document.exitFullscreen();
  } catch (err) {
    toast(`Fullscreen unavailable: ${err.message}`);
  }
  setTimeout(render, 80);
});
for (const el of [els.viewMode, els.colorMode, els.labelMode, els.compareLens, els.compareLayout, els.showVelocity, els.showRings, els.showErrors]) {
  el.addEventListener("change", () => { syncCycleButtons(); render(); });
}
els.layerChips.querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", () => {
  chip.classList.toggle("active");
  state.selected = null;
  updateInspect();
  render();
}));
function setAllLabelChips(active) {
  els.labels.querySelectorAll(".chip").forEach(chip => chip.classList.toggle("active", active));
  state.selected = null;
  updateInspect();
  render();
}
els.labelsAllBtn.addEventListener("click", () => setAllLabelChips(true));
els.labelsNoneBtn.addEventListener("click", () => setAllLabelChips(false));
els.boxOpacity.addEventListener("input", render);
els.confMin.addEventListener("input", () => { state.selected = null; updateInspect(); render(); });
els.confMin.addEventListener("change", render);
els.dedupeRows.addEventListener("change", () => { toast(els.dedupeRows.checked ? "Dedupe enabled." : "Dedupe disabled."); scheduleSceneReload("dedupe"); });
els.heatCanvas.addEventListener("click", e => seekFrameFromClientX(els.heatCanvas, e.clientX));
els.frameCurve.addEventListener("click", e => seekFrameFromClientX(els.frameCurve, e.clientX));
els.overview.addEventListener("click", e => {
  const rect = els.overview.getBoundingClientRect();
  const maxAbs = Math.max(20, state.bounds.maxAbs || 80);
  const pad = 10;
  const scale = Math.min((rect.width - pad * 2), (rect.height - pad * 2)) / (maxAbs * 2);
  state.panY = -((e.clientX - rect.left) - rect.width / 2) / Math.max(.001, scale);
  state.panX = -((e.clientY - rect.top) - rect.height / 2) / Math.max(.001, scale);
  render();
});
function updateCurtainFromEvent(e) {
  const rect = els.canvas.getBoundingClientRect();
  state.curtainX = Math.max(0.08, Math.min(0.92, (e.clientX - rect.left) / Math.max(1, rect.width)));
  render();
}
function bevScreenToWorld(screenX, screenY, distance = state.distance) {
  const scale = Math.min(els.canvas.clientWidth, els.canvas.clientHeight) / Math.max(20, distance * 2.15);
  return {
    x: state.panX - (screenY - els.canvas.clientHeight / 2) / Math.max(0.001, scale),
    y: state.panY - (screenX - els.canvas.clientWidth / 2) / Math.max(0.001, scale),
    scale
  };
}
els.curtainHandle.addEventListener("pointerdown", e => {
  if (!compareCurtainActive()) return;
  e.preventDefault();
  e.stopPropagation();
  state.draggingCurtain = true;
  els.curtainHandle.setPointerCapture(e.pointerId);
  updateCurtainFromEvent(e);
});
els.curtainHandle.addEventListener("pointermove", e => {
  if (!state.draggingCurtain) return;
  e.preventDefault();
  updateCurtainFromEvent(e);
});
els.curtainHandle.addEventListener("pointerup", e => {
  state.draggingCurtain = false;
  try { els.curtainHandle.releasePointerCapture(e.pointerId); } catch (_err) {}
});
els.canvas.addEventListener("pointerdown", e => {
  e.preventDefault();
  if (compareCurtainActive()) {
    const rect = els.canvas.getBoundingClientRect();
    if (Math.abs((e.clientX - rect.left) - state.curtainX * rect.width) < 18) {
      state.draggingCurtain = true;
      els.canvas.setPointerCapture(e.pointerId);
      updateCurtainFromEvent(e);
      return;
    }
  }
  state.dragging = true; state.dragButton = e.button || 0; state.lastX = e.clientX; state.lastY = e.clientY; state.downX = e.clientX; state.downY = e.clientY; els.canvas.setPointerCapture(e.pointerId);
});
els.canvas.addEventListener("pointerup", e => {
  if (state.draggingCurtain) {
    state.draggingCurtain = false;
    try { els.canvas.releasePointerCapture(e.pointerId); } catch (_err) {}
    return;
  }
  const moved = Math.hypot(e.clientX - state.downX, e.clientY - state.downY);
  state.dragging = false;
  if (moved < 5) {
    const rect = els.canvas.getBoundingClientRect();
    state.selected = nearestBox(e.clientX - rect.left, e.clientY - rect.top);
    updateInspect();
    render();
  }
});
els.canvas.addEventListener("pointermove", e => {
  if (state.draggingCurtain) {
    updateCurtainFromEvent(e);
    return;
  }
  updateHoverCard(e);
  if (!state.dragging) return;
  if (els.viewMode.value === "perspective") {
    if (e.shiftKey || state.dragButton === 1 || state.dragButton === 2) {
      panPerspectiveByScreenDelta(e.clientX - state.lastX, e.clientY - state.lastY);
    } else {
      state.yaw -= (e.clientX - state.lastX) * .008;
      state.pitch = Math.max(.18, Math.min(1.18, state.pitch + (e.clientY - state.lastY) * .006));
    }
  } else if (els.viewMode.value === "bev") {
    const vp = activeViewport || {w: els.canvas.clientWidth, h: els.canvas.clientHeight};
    const scale = Math.min(vp.w, vp.h) / Math.max(20, state.distance * 2.15);
    state.panY += (e.clientX - state.lastX) / Math.max(0.001, scale);
    state.panX += (e.clientY - state.lastY) / Math.max(0.001, scale);
  }
  state.lastX = e.clientX; state.lastY = e.clientY; render();
});
els.canvas.addEventListener("contextmenu", e => e.preventDefault());
els.canvas.addEventListener("dblclick", e => {
  const rect = els.canvas.getBoundingClientRect();
  state.selected = nearestBox(e.clientX - rect.left, e.clientY - rect.top);
  updateInspect();
  if (!focusSelectedObject()) render();
});
els.canvas.addEventListener("pointerleave", () => els.hoverCard.classList.remove("show"));
els.canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const rect = els.canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const before = els.viewMode.value === "bev" ? bevScreenToWorld(sx, sy) : null;
  state.distance = Math.max(18, Math.min(420, state.distance + e.deltaY * .08));
  if (before && els.viewMode.value === "bev") {
    const afterScale = Math.min(els.canvas.clientWidth, els.canvas.clientHeight) / Math.max(20, state.distance * 2.15);
    state.panX = before.x + (sy - els.canvas.clientHeight / 2) / Math.max(0.001, afterScale);
    state.panY = before.y + (sx - els.canvas.clientWidth / 2) / Math.max(0.001, afterScale);
  }
  render();
}, {passive: false});
window.addEventListener("resize", () => { renderHeatStrip(); render(); });
// Guarded: if bbox_theme.js is missing this must not abort the rest of the script,
// which is what scans and loads scenes.
if (window.TH) {
  TH.bindToggle(els.themeToggle);
  // The scene, heat strip and label bars are canvas/inline-styled, so they need an
  // explicit repaint when the palette changes.
  TH.onChange(() => { renderHeatStrip(); render(); });
}
window.addEventListener("keydown", (ev) => {
  if (ev.target && ["INPUT", "SELECT", "TEXTAREA"].includes(ev.target.tagName)) return;
  if (ev.key === " ") {
    ev.preventDefault();
    state.playing = !state.playing;
    state.playCarryMs = 0;
    state.lastPlayTs = performance.now();
    els.play.textContent = state.playing ? "Ⅱ" : "▶";
  } else if (ev.key === "f" || ev.key === "F") {
    els.fullscreenBtn.click();
  } else if (ev.key === "h" || ev.key === "H") {
    els.toggleSidebar.click();
  } else if (ev.key === "ArrowRight") {
    stepFrame(1);
  } else if (ev.key === "ArrowLeft") {
    stepFrame(-1);
  } else if (ev.key === "c" || ev.key === "C") {
    els.compareEnabled.checked = !els.compareEnabled.checked;
    updateCompareControls();
    scheduleSceneReload("comparison mode");
  } else if (ev.key === "v" || ev.key === "V") {
    if (state.compare) setCompareLayout(compareLayoutMode() === "curtain" ? "side_by_side" : "curtain");
  } else if (ev.key === "l" || ev.key === "L") {
    if (state.compare) cycleSelect(els.compareLens, compareLensModes);
  }
});
syncCycleButtons();
updateCompareControls();
scan().then(() => render());
loop();
