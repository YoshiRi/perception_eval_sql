/* TLR analysis viewer — vanilla JS + canvas, sibling of the bbox explorer.
   Charts follow the stats_renderer.js idioms (chartPanel frame, niceMax axis,
   hit registry per canvas) and read every color from bbox_theme.js (TH.*). */
(function () {
  "use strict";

  const SESSION_KEY = "local_tlr_viewer.session.v1";

  const els = {};
  ["dirSelect", "reloadBtn", "themeToggleBtn", "kTpRate", "kTpRateBar", "kScenarios", "kFrames",
   "kBest", "kBestLabel", "kWorst", "kWorstLabel", "criteriaCanvas", "heatmapCanvas", "matrixChips",
   "scenarioMeta", "scenarioHead", "scenarioBody", "frameDrawer", "drawerTitle", "drawerMeta",
   "drawerCloseBtn", "timelineCanvas", "frameHead", "frameBody", "worstList", "toast"]
    .forEach(id => { els[id] = document.getElementById(id); });

  const state = {
    dirs: [],
    path: "",
    summary: null,      // {stats, criteria_matrix, scenario_summary}
    matrices: null,     // {vehicle_status, vehicle_status_counts, critical_priority, critical_priority_counts}
    matrixMode: "all",
    scenario: "",
    frames: [],
    sortKey: "tp_rate",
    sortDir: 1,
    hits: {criteria: [], heatmap: []},
    hover: null
  };

  /* ---------- utils ---------- */
  function toast(msg) {
    els.toast.textContent = msg;
    els.toast.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => els.toast.classList.remove("show"), 3200);
  }
  function fmt(n) { return Number(n || 0).toLocaleString(); }
  function rate(v) { return (v == null || !Number.isFinite(Number(v))) ? "–" : `${(Number(v) * 100).toFixed(1)}%`; }
  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  }
  function hasSignal(v) {
    const t = String(v ?? "").trim();
    return t !== "" && t !== "0 []" && t !== "null";
  }
  function loadSession() {
    try { return JSON.parse(localStorage.getItem(SESSION_KEY)) || {}; } catch (_e) { return {}; }
  }
  function saveSession() {
    try { localStorage.setItem(SESSION_KEY, JSON.stringify({path: state.path, scenario: state.scenario})); } catch (_e) { /* private mode */ }
  }
  function syncUrl() {
    try {
      const q = new URLSearchParams(window.location.search);
      if (state.path) q.set("path", state.path); else q.delete("path");
      if (state.scenario) q.set("scenario", state.scenario); else q.delete("scenario");
      const qs = q.toString();
      history.replaceState(null, "", qs ? `?${qs}` : window.location.pathname);
    } catch (_e) { /* sandboxed iframe */ }
  }

  /* Heat encoding: hot = missing. 0 => calm, 1 => hot, matching the explorer. */
  function missHeat(tpRate) { return TH.heat(1 - Math.max(0, Math.min(1, Number(tpRate) || 0))); }
  function signalColor(sig) {
    if (sig === "green") return TH.c("good");
    if (sig === "yellow") return TH.c("warn");
    if (sig === "red") return TH.c("bad");
    return TH.c("muted");
  }

  /* ---------- canvas plumbing ---------- */
  function setupCanvas(canvas) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {ctx, w: rect.width, h: rect.height};
  }
  function chartPanel(ctx, rect, title) {
    ctx.fillStyle = TH.a("deep", .42);
    ctx.strokeStyle = TH.a("line", .2);
    ctx.lineWidth = 1;
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    if (title) {
      ctx.fillStyle = TH.c("text");
      ctx.font = "800 12px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(title, rect.x + 12, rect.y + 18);
    }
    return {x: rect.x + 12, y: rect.y + (title ? 30 : 12), w: rect.w - 24, h: rect.h - (title ? 42 : 24)};
  }
  function emptyNote(ctx, w, h, msg) {
    ctx.fillStyle = TH.c("muted");
    ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(msg, w / 2, h / 2);
    ctx.textAlign = "left";
  }

  /* ---------- criteria bar chart ---------- */
  function criteriaRows() {
    const records = state.summary?.criteria_matrix?.records || [];
    return records
      .map(r => ({
        name: String(r["Criteria"] ?? ""),
        tp: Number(r["Number of TP"] || 0),
        total: Number(r["Number of total frames"] || 0),
        tpRate: Number(r["TP rate"] || 0)
      }))
      .filter(r => r.total > 0);
  }
  function drawCriteria() {
    const {ctx, w, h} = setupCanvas(els.criteriaCanvas);
    ctx.clearRect(0, 0, w, h);
    state.hits.criteria = [];
    const rows = criteriaRows();
    if (!rows.length) { emptyNote(ctx, w, h, "No criteria with evaluated frames in this directory."); return; }
    const plot = chartPanel(ctx, {x: 0, y: 0, w, h}, "");
    const labelW = 78, valueW = 110;
    const rowH = Math.min(30, Math.max(15, plot.h / rows.length));
    const barX = plot.x + labelW;
    const barW = Math.max(30, plot.w - labelW - valueW);
    // Reference gridlines at 50/75/100% of TP rate.
    ctx.strokeStyle = TH.a("line", .16);
    ctx.lineWidth = 1;
    [.5, .75, 1].forEach(g => {
      const gx = Math.round(barX + barW * g) + .5;
      ctx.beginPath(); ctx.moveTo(gx, plot.y); ctx.lineTo(gx, plot.y + rows.length * rowH); ctx.stroke();
    });
    rows.forEach((r, i) => {
      const y = plot.y + i * rowH;
      const bh = Math.max(7, rowH - 7);
      const isHover = state.hover && state.hover.kind === "criteria" && state.hover.name === r.name;
      ctx.fillStyle = isHover ? TH.c("textBright") : TH.c("mutedBright");
      ctx.font = `${isHover ? "800" : "700"} 11px ui-sans-serif, system-ui, sans-serif`;
      ctx.fillText(r.name.replace("criteria_", "criteria "), plot.x, y + bh / 2 + 4);
      ctx.fillStyle = TH.a("surface", .8);
      ctx.fillRect(barX, y, barW, bh);
      ctx.fillStyle = missHeat(r.tpRate);
      ctx.fillRect(barX, y, Math.max(2, barW * r.tpRate), bh);
      if (isHover) {
        ctx.strokeStyle = TH.c("accent");
        ctx.strokeRect(barX + .5, y + .5, barW - 1, bh - 1);
      }
      ctx.fillStyle = TH.c("muted");
      ctx.font = "700 10.5px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(`${rate(r.tpRate)} · ${fmt(r.tp)}/${fmt(r.total)}`, barX + barW + 8, y + bh / 2 + 4);
      state.hits.criteria.push({kind: "criteria", name: r.name, x: plot.x, y, w: plot.w, h: rowH,
        title: r.name, lines: [`TP rate ${rate(r.tpRate)}`, `${fmt(r.tp)} TP of ${fmt(r.total)} frames`]});
    });
  }

  /* ---------- vehicle-status heatmap ---------- */
  function parseHeatColumn(name) {
    const n = String(name).toLowerCase();
    let sig = "other", zone = "";
    if (n.includes("green")) sig = "green";
    else if (n.includes("yellow")) sig = "yellow";
    else if (n.includes("red")) sig = "red";
    if (n.includes("all types combined")) { sig = "all"; zone = "all"; }
    else if (n.includes("criteria0-9")) zone = "crit 0-9";
    else if (n.includes("critical")) zone = "critical";
    else if (n.includes("priority")) zone = "priority";
    else if (n.includes("other criteria")) zone = "other crit";
    return {sig, zone};
  }
  function heatmapData() {
    const m = state.matrices;
    if (!m) return null;
    const rates = state.matrixMode === "critical" ? m.critical_priority : m.vehicle_status;
    const counts = state.matrixMode === "critical" ? m.critical_priority_counts : m.vehicle_status_counts;
    if (!rates || !rates.records || !rates.records.length) return null;
    const cols = rates.columns.filter(c => c !== "Vehicle Status");
    return {cols, rates: rates.records, counts: (counts && counts.records) || []};
  }
  function drawHeatmap() {
    const {ctx, w, h} = setupCanvas(els.heatmapCanvas);
    ctx.clearRect(0, 0, w, h);
    state.hits.heatmap = [];
    const data = heatmapData();
    if (!data) { emptyNote(ctx, w, h, "No vehicle-status matrix for this directory."); return; }
    const plot = chartPanel(ctx, {x: 0, y: 0, w, h}, "");
    const rowLabelW = 66, headH = 40, legendH = 20;
    const gx = plot.x + rowLabelW, gy = plot.y + headH;
    const gw = plot.w - rowLabelW, gh = plot.h - headH - legendH;
    const nc = data.cols.length, nr = data.rates.length;
    const cw = gw / nc, ch = gh / nr;
    ctx.font = "700 10px ui-sans-serif, system-ui, sans-serif";
    // Column heads: a signal lamp dot over a zone code — the axis explains itself.
    data.cols.forEach((col, j) => {
      const meta = parseHeatColumn(col);
      const cx = gx + j * cw + cw / 2;
      if (meta.sig === "all") {
        ctx.fillStyle = TH.c("mutedBright");
        ctx.textAlign = "center";
        ctx.fillText("ALL", cx, plot.y + 16);
      } else {
        ctx.fillStyle = signalColor(meta.sig);
        ctx.beginPath(); ctx.arc(cx, plot.y + 11, 4, 0, Math.PI * 2); ctx.fill();
      }
      ctx.fillStyle = TH.c("muted");
      ctx.textAlign = "center";
      ctx.fillText(meta.zone === "all" ? "combined" : meta.zone, cx, plot.y + 30);
    });
    ctx.textAlign = "left";
    data.rates.forEach((row, i) => {
      const yc = gy + i * ch + ch / 2;
      ctx.fillStyle = TH.c("mutedBright");
      ctx.font = "700 11px ui-sans-serif, system-ui, sans-serif";
      const statusName = String(row["Vehicle Status"] || "");
      ctx.fillText(statusName === "All Status Combined" ? "All status" : statusName, plot.x, yc + 4);
      data.cols.forEach((col, j) => {
        const x = gx + j * cw, y = gy + i * ch;
        const v = row[col];
        const countText = String((data.counts[i] || {})[col] ?? "");
        const evaluated = countText === "" || !/^0\s*\/\s*0$/.test(countText);
        const cellW = cw - 3, cellH = ch - 3;
        if (v == null || !evaluated) {
          ctx.fillStyle = TH.a("line", .1);
          ctx.fillRect(x + 1.5, y + 1.5, cellW, cellH);
          ctx.fillStyle = TH.c("muted");
          ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("–", x + cw / 2, y + ch / 2 + 3);
          ctx.textAlign = "left";
        } else {
          ctx.fillStyle = missHeat(v);
          ctx.fillRect(x + 1.5, y + 1.5, cellW, cellH);
          ctx.fillStyle = TH.c("onAccent");
          ctx.fillStyle = "rgb(255 255 255 / .94)";
          ctx.font = "800 11px ui-sans-serif, system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(`${Math.round(Number(v) * 100)}%`, x + cw / 2, y + ch / 2 + (cellH > 26 ? -1 : 3));
          if (cellH > 26 && countText) {
            ctx.fillStyle = "rgb(255 255 255 / .72)";
            ctx.font = "600 9px ui-sans-serif, system-ui, sans-serif";
            ctx.fillText(countText.replace(/\s+/g, ""), x + cw / 2, y + ch / 2 + 11);
          }
          ctx.textAlign = "left";
        }
        const isHover = state.hover && state.hover.kind === "heat" && state.hover.i === i && state.hover.j === j;
        if (isHover) {
          ctx.strokeStyle = TH.c("accent");
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x + 1.5, y + 1.5, cellW, cellH);
        }
        state.hits.heatmap.push({kind: "heat", i, j, x, y, w: cw, h: ch,
          title: `${row["Vehicle Status"]} × ${col}`,
          lines: [v == null ? "not evaluated" : `TP rate ${rate(v)}`, countText ? `frames ${countText}` : ""]});
      });
    });
    // Legend: miss-heat scale, calm -> hot.
    const lw = Math.min(140, gw * .4), lx = gx + gw - lw - 56, ly = plot.y + plot.h - 8;
    for (let k = 0; k < lw; k++) {
      ctx.fillStyle = missHeat(1 - k / lw);
      ctx.fillRect(lx + k, ly - 6, 1, 6);
    }
    ctx.fillStyle = TH.c("muted");
    ctx.font = "600 9.5px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("miss →", lx - 6, ly);
    ctx.textAlign = "left";
    ctx.fillText("100% TP", lx + lw + 6, ly);
  }

  /* ---------- hover cards (shared for both canvases) ---------- */
  function bindCanvasHover(canvas, hitsKey, repaint) {
    canvas.addEventListener("mousemove", ev => {
      const rect = canvas.getBoundingClientRect();
      const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
      const hit = state.hits[hitsKey].find(hh => x >= hh.x && x <= hh.x + hh.w && y >= hh.y && y <= hh.y + hh.h) || null;
      const changed = JSON.stringify(hit && [hit.kind, hit.name, hit.i, hit.j]) !== JSON.stringify(state.hover && [state.hover.kind, state.hover.name, state.hover.i, state.hover.j]);
      state.hover = hit;
      canvas.title = hit ? `${hit.title}\n${(hit.lines || []).filter(Boolean).join("\n")}` : "";
      if (changed) repaint();
    });
    canvas.addEventListener("mouseleave", () => {
      if (state.hover) { state.hover = null; canvas.title = ""; repaint(); }
    });
  }

  /* ---------- KPI strip ---------- */
  function renderKpis() {
    const s = state.summary?.stats || {};
    els.kTpRate.textContent = rate(s.overall_tp_rate);
    els.kTpRateBar.style.width = `${Math.round((Number(s.overall_tp_rate) || 0) * 100)}%`;
    els.kTpRateBar.style.background = missHeat(s.overall_tp_rate);
    els.kScenarios.textContent = fmt(s.num_scenarios);
    els.kFrames.textContent = fmt(s.total_frames);
    els.kBest.textContent = s.best_criteria ? String(s.best_criteria).replace("criteria_", "criteria ") : "–";
    els.kBestLabel.textContent = s.best_criteria ? `best · ${rate(s.best_tp_rate)}` : "best criteria";
    els.kWorst.textContent = s.worst_criteria ? String(s.worst_criteria).replace("criteria_", "criteria ") : "–";
    els.kWorstLabel.textContent = s.worst_criteria ? `worst · ${rate(s.worst_tp_rate)}` : "worst criteria";
  }

  /* ---------- scenario table ---------- */
  const SCEN_COLS = [
    {key: "suite", label: "Suite"},
    {key: "scenario", label: "Scenario"},
    {key: "frames", label: "Frames", num: true},
    {key: "evaluable_frames", label: "Evaluable", num: true},
    {key: "tp_frames", label: "TP", num: true},
    {key: "fn_frames", label: "FN", num: true},
    {key: "signal_types", label: "Signals", num: true},
    {key: "tp_rate", label: "TP rate", num: true, rate: true}
  ];
  function scenarioRows() {
    const rows = (state.summary?.scenario_summary?.records || []).slice();
    const k = state.sortKey, d = state.sortDir;
    rows.sort((a, b) => {
      const av = a[k], bv = b[k];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" || typeof bv === "number") return (Number(av) - Number(bv)) * d;
      return String(av).localeCompare(String(bv)) * d;
    });
    return rows;
  }
  function renderScenarioTable() {
    els.scenarioHead.innerHTML = SCEN_COLS.map(c => {
      const sorted = state.sortKey === c.key;
      const arrow = sorted ? `<span class="arrow"> ${state.sortDir > 0 ? "▲" : "▼"}</span>` : "";
      return `<th class="${c.num ? "num" : ""}${sorted ? " sorted" : ""}" data-key="${c.key}">${c.label}${arrow}</th>`;
    }).join("");
    const rows = scenarioRows();
    els.scenarioMeta.textContent = rows.length
      ? `${rows.length} scenarios · click a row to open the frame drill-down`
      : "no scenario roll-up for this directory";
    if (!rows.length) {
      els.scenarioBody.innerHTML = `<tr class="empty-row"><td colspan="${SCEN_COLS.length}">No scenarios with per-frame details.</td></tr>`;
      return;
    }
    els.scenarioBody.innerHTML = rows.map(r => {
      const active = String(r.scenario) === state.scenario ? " class=\"active\"" : "";
      const cells = SCEN_COLS.map(c => {
        if (c.rate) {
          const v = r[c.key];
          const pct = v == null ? 0 : Math.max(0, Math.min(1, Number(v)));
          return `<td class="num"><span class="rate-cell"><span class="rate-bar"><i style="width:${Math.round(pct * 100)}%;background:${missHeat(v == null ? 1 : v)}"></i></span>${rate(v)}</span></td>`;
        }
        const v = r[c.key];
        return `<td class="${c.num ? "num" : ""}">${c.num ? fmt(v) : escapeHtml(v)}</td>`;
      }).join("");
      return `<tr data-scenario="${escapeHtml(r.scenario)}"${active}>${cells}</tr>`;
    }).join("");
  }

  /* ---------- worst scenarios quick list ---------- */
  function renderWorstList() {
    const rows = (state.summary?.scenario_summary?.records || [])
      .filter(r => Number(r.evaluable_frames) > 0 && r.tp_rate != null)
      .sort((a, b) => Number(a.tp_rate) - Number(b.tp_rate))
      .slice(0, 8);
    if (!rows.length) {
      els.worstList.innerHTML = `<div class="sub">No evaluable scenarios.</div>`;
      return;
    }
    els.worstList.innerHTML = rows.map(r =>
      `<div class="item${String(r.scenario) === state.scenario ? " active" : ""}" data-scenario="${escapeHtml(r.scenario)}" role="button" tabindex="0">` +
      `<span class="name" title="${escapeHtml(r.scenario)}">${escapeHtml(r.scenario)}</span>` +
      `<span class="val">${rate(r.tp_rate)}</span></div>`
    ).join("");
  }

  /* ---------- frame drill-down ---------- */
  function frameOutcome(r) {
    if (hasSignal(r.tp)) return "tp";
    if (hasSignal(r.fn)) return "fn";
    return "none";
  }
  function drawTimeline() {
    const {ctx, w, h} = setupCanvas(els.timelineCanvas);
    ctx.clearRect(0, 0, w, h);
    if (!state.frames.length) { emptyNote(ctx, w, h, "No frames for this scenario."); return; }
    const plot = chartPanel(ctx, {x: 0, y: 0, w, h}, "");
    const maxIdx = Math.max(1, ...state.frames.map(r => Number(r.frame_index) || 0));
    const y0 = plot.y + 4, barH = plot.h - 18;
    state.frames.forEach(r => {
      const x = plot.x + (Number(r.frame_index) || 0) / maxIdx * (plot.w - 2);
      const out = frameOutcome(r);
      if (out === "tp") { ctx.fillStyle = TH.c("good"); ctx.fillRect(x, y0, 2, barH); }
      else if (out === "fn") { ctx.fillStyle = TH.c("bad"); ctx.fillRect(x, y0 - 2, 2, barH + 4); }
      else { ctx.fillStyle = TH.a("muted", .35); ctx.fillRect(x, y0 + barH * .32, 2, barH * .36); }
    });
    ctx.fillStyle = TH.c("muted");
    ctx.font = "600 9.5px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText("frame 0", plot.x, plot.y + plot.h);
    ctx.textAlign = "right";
    ctx.fillText(`frame ${fmt(maxIdx)} · green TP / red FN / grey unevaluated`, plot.x + plot.w, plot.y + plot.h);
    ctx.textAlign = "left";
  }
  const FRAME_COLS = [
    {key: "frame_index", label: "#", num: true},
    {key: "frame_name", label: "Frame"},
    {key: "outcome", label: "Result"},
    {key: "status", label: "Vehicle"},
    {key: "speed_kph", label: "km/h", num: true, fixed: 1},
    {key: "traffic_light_type", label: "Signal"},
    {key: "criteria", label: "Criteria"},
    {key: "fn", label: "FN detail"}
  ];
  function renderFrameTable() {
    els.frameHead.innerHTML = FRAME_COLS.map(c => `<th class="${c.num ? "num" : ""}">${c.label}</th>`).join("");
    if (!state.frames.length) {
      els.frameBody.innerHTML = `<tr class="empty-row"><td colspan="${FRAME_COLS.length}">No frames.</td></tr>`;
      return;
    }
    els.frameBody.innerHTML = state.frames.map(r => {
      const out = frameOutcome(r);
      const cells = FRAME_COLS.map(c => {
        if (c.key === "outcome") {
          const label = out === "tp" ? "TP" : out === "fn" ? "FN" : "–";
          return `<td><span class="pill ${out}">${label}</span></td>`;
        }
        if (c.key === "traffic_light_type") {
          const sig = String(r[c.key] ?? "");
          return `<td><span class="sig-dot" style="background:${signalColor(sig)}"></span>${escapeHtml(sig || "unknown")}</td>`;
        }
        let v = r[c.key];
        if (c.fixed != null) v = v == null ? "–" : Number(v).toFixed(c.fixed);
        if (c.key === "fn") v = hasSignal(v) ? v : "";
        return `<td class="${c.num ? "num" : ""}">${escapeHtml(v ?? "")}</td>`;
      }).join("");
      return `<tr>${cells}</tr>`;
    }).join("");
  }
  async function openScenario(scenario) {
    if (!scenario) return;
    state.scenario = scenario;
    saveSession();
    syncUrl();
    renderScenarioTable();
    renderWorstList();
    els.frameDrawer.classList.add("open");
    els.frameDrawer.setAttribute("aria-hidden", "false");
    els.drawerTitle.textContent = scenario;
    els.drawerMeta.textContent = "loading frames…";
    try {
      const data = await api("/api/tlr_frames", {path: state.path, scenario});
      state.frames = data.records || [];
      const tp = state.frames.filter(r => frameOutcome(r) === "tp").length;
      const fn = state.frames.filter(r => frameOutcome(r) === "fn").length;
      els.drawerMeta.textContent = `${fmt(state.frames.length)} frames · ${fmt(tp)} TP · ${fmt(fn)} FN`;
      drawTimeline();
      renderFrameTable();
    } catch (err) {
      state.frames = [];
      els.drawerMeta.textContent = "failed to load frames";
      drawTimeline();
      renderFrameTable();
      toast(String(err.message || err));
    }
  }
  function closeDrawer() {
    state.scenario = "";
    saveSession();
    syncUrl();
    els.frameDrawer.classList.remove("open");
    els.frameDrawer.setAttribute("aria-hidden", "true");
    renderScenarioTable();
    renderWorstList();
  }

  /* ---------- data loading ---------- */
  function repaintCharts() { drawCriteria(); drawHeatmap(); if (els.frameDrawer.classList.contains("open")) drawTimeline(); }
  function renderAll() {
    renderKpis();
    renderScenarioTable();
    renderWorstList();
    repaintCharts();
  }
  async function loadDirectory(path, {openScenarioName = ""} = {}) {
    state.path = path;
    saveSession();
    syncUrl();
    els.scenarioMeta.textContent = "loading…";
    try {
      const [summary, matrices] = await Promise.all([
        api("/api/tlr_summary", {path}),
        api("/api/tlr_matrices", {path})
      ]);
      state.summary = summary;
      state.matrices = matrices;
      renderAll();
      if (openScenarioName) {
        const known = (summary.scenario_summary?.records || []).some(r => String(r.scenario) === openScenarioName);
        if (known) await openScenario(openScenarioName); else closeDrawer();
      } else if (els.frameDrawer.classList.contains("open")) {
        closeDrawer();
      }
    } catch (err) {
      state.summary = null;
      state.matrices = null;
      renderAll();
      els.scenarioMeta.textContent = "failed to load directory";
      toast(String(err.message || err));
    }
  }
  async function loadDirs() {
    const data = await api("/api/tlr_dirs", {});
    state.dirs = data.items || [];
    els.dirSelect.innerHTML = state.dirs.length
      ? state.dirs.map(d => `<option value="${escapeHtml(d.path)}">${escapeHtml(d.path)} · ${fmt(d.scenarios)} scenario${d.scenarios === 1 ? "" : "s"}</option>`).join("")
      : `<option value="">no TLR directories under the data root</option>`;
    return state.dirs;
  }

  /* ---------- events ---------- */
  els.dirSelect.addEventListener("change", () => {
    const path = els.dirSelect.value;
    if (path) loadDirectory(path);
  });
  els.reloadBtn.addEventListener("click", async () => {
    try {
      const dirs = await loadDirs();
      const still = dirs.some(d => d.path === state.path);
      if (still) els.dirSelect.value = state.path;
      else if (dirs.length) { els.dirSelect.value = dirs[0].path; await loadDirectory(dirs[0].path); }
      toast(`Found ${dirs.length} TLR director${dirs.length === 1 ? "y" : "ies"}.`);
    } catch (err) { toast(String(err.message || err)); }
  });
  els.matrixChips.addEventListener("click", ev => {
    const chip = ev.target.closest(".chip");
    if (!chip) return;
    state.matrixMode = chip.dataset.matrix || "all";
    els.matrixChips.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c === chip));
    drawHeatmap();
  });
  els.matrixChips.addEventListener("keydown", ev => {
    if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); ev.target.click(); }
  });
  els.scenarioHead.addEventListener("click", ev => {
    const th = ev.target.closest("th");
    if (!th) return;
    const key = th.dataset.key;
    if (state.sortKey === key) state.sortDir = -state.sortDir;
    else { state.sortKey = key; state.sortDir = key === "suite" || key === "scenario" ? 1 : -1; if (key === "tp_rate") state.sortDir = 1; }
    renderScenarioTable();
  });
  els.scenarioBody.addEventListener("click", ev => {
    const tr = ev.target.closest("tr[data-scenario]");
    if (tr) openScenario(tr.dataset.scenario);
  });
  els.worstList.addEventListener("click", ev => {
    const item = ev.target.closest(".item[data-scenario]");
    if (item) openScenario(item.dataset.scenario);
  });
  els.worstList.addEventListener("keydown", ev => {
    if (ev.key === "Enter" || ev.key === " ") {
      const item = ev.target.closest(".item[data-scenario]");
      if (item) { ev.preventDefault(); openScenario(item.dataset.scenario); }
    }
  });
  els.drawerCloseBtn.addEventListener("click", closeDrawer);
  window.addEventListener("keydown", ev => {
    if (ev.key === "Escape" && els.frameDrawer.classList.contains("open")) closeDrawer();
  });
  bindCanvasHover(els.criteriaCanvas, "criteria", drawCriteria);
  bindCanvasHover(els.heatmapCanvas, "heatmap", drawHeatmap);
  let resizeT = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(repaintCharts, 120);
  });

  TH.bindToggle(els.themeToggleBtn);
  TH.onChange(() => { renderAll(); });

  /* ---------- boot: deep link -> session -> first discovered dir ---------- */
  (async function boot() {
    const q = new URLSearchParams(window.location.search);
    const session = loadSession();
    const wantPath = q.get("path") || session.path || "";
    const wantScenario = q.get("scenario") || (q.get("path") ? "" : session.scenario) || "";
    try {
      const dirs = await loadDirs();
      if (!dirs.length) { els.scenarioMeta.textContent = "no TLR directories under the data root"; renderAll(); return; }
      const chosen = dirs.some(d => d.path === wantPath) ? wantPath : dirs[0].path;
      els.dirSelect.value = chosen;
      await loadDirectory(chosen, {openScenarioName: chosen === wantPath ? wantScenario : ""});
    } catch (err) {
      renderAll();
      toast(String(err.message || err));
    }
  })();
})();
