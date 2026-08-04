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
   "drawerCloseBtn", "timelineCanvas", "frameHead", "frameBody", "worstList", "worstTitle", "toast",
   "compareSelect", "swapBtn", "dirTagA", "dirTagB", "compareBanner", "compareBannerText",
   "dTpRate", "dScenarios", "dFrames", "scenarioChips"]
    .forEach(id => { els[id] = document.getElementById(id); });

  const state = {
    dirs: [],
    path: "",
    summary: null,      // {stats, criteria_matrix, scenario_summary}
    matrices: null,     // {vehicle_status, vehicle_status_counts, critical_priority, critical_priority_counts}
    // Compare is the same two payloads for a second directory. The routes are already
    // per-directory and the analyzer caches each one, so B costs a second call rather
    // than a new server-side comparison.
    pathB: "",
    summaryB: null,
    matricesB: null,
    scenarioFilter: "all",
    matrixMode: "all",
    scenario: "",
    frames: [],
    sortKey: "tp_rate",
    sortDir: 1,
    hits: {criteria: [], heatmap: []},
    hover: null
  };
  function comparing() { return Boolean(state.pathB && state.summaryB); }

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
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify({
        path: state.path, pathB: state.pathB, scenario: state.scenario
      }));
    } catch (_e) { /* private mode */ }
  }
  function syncUrl() {
    try {
      const q = new URLSearchParams(window.location.search);
      if (state.path) q.set("path", state.path); else q.delete("path");
      if (state.pathB) q.set("path_b", state.pathB); else q.delete("path_b");
      if (state.scenario) q.set("scenario", state.scenario); else q.delete("scenario");
      const qs = q.toString();
      history.replaceState(null, "", qs ? `?${qs}` : window.location.pathname);
    } catch (_e) { /* sandboxed iframe */ }
  }

  /* ---------- delta helpers ----------
     One convention everywhere: B − A, green when B improved, red when it regressed.
     Rates carry no inherent direction, so "better" is always "higher TP rate". */
  function delta(a, b) {
    if (a == null || b == null || !Number.isFinite(Number(a)) || !Number.isFinite(Number(b))) return null;
    return Number(b) - Number(a);
  }
  function deltaColor(d, {invert = false} = {}) {
    if (d == null || Math.abs(d) < 1e-9) return TH.c("muted");
    return (invert ? d < 0 : d > 0) ? TH.c("good") : TH.c("bad");
  }
  function deltaText(d, {pct = false, fixed = 1} = {}) {
    if (d == null) return "–";
    if (Math.abs(d) < (pct ? 5e-4 : 0.5)) return pct ? `±${(0).toFixed(fixed)}pp` : "±0";
    const sign = d > 0 ? "+" : "−";
    return pct ? `${sign}${Math.abs(d * 100).toFixed(fixed)}pp` : `${sign}${fmt(Math.abs(Math.round(d)))}`;
  }
  function paintDelta(el, d, opts) {
    if (!el) return;
    el.hidden = !comparing() || d == null;
    if (el.hidden) return;
    el.textContent = deltaText(d, opts);
    el.style.color = deltaColor(d, opts);
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
  function criteriaRowsOf(summary) {
    const records = summary?.criteria_matrix?.records || [];
    return records
      .map(r => ({
        name: String(r["Criteria"] ?? ""),
        tp: Number(r["Number of TP"] || 0),
        total: Number(r["Number of total frames"] || 0),
        tpRate: Number(r["TP rate"] || 0)
      }))
      .filter(r => r.total > 0);
  }
  function criteriaRows() {
    const rows = criteriaRowsOf(state.summary);
    if (!comparing()) return rows;
    // Union of both sides: a criteria evaluated only in B is exactly what a comparison
    // is meant to surface, so it must not be dropped for being absent from A.
    const byName = new Map(rows.map(r => [r.name, {...r, b: null}]));
    criteriaRowsOf(state.summaryB).forEach(r => {
      const entry = byName.get(r.name);
      if (entry) entry.b = r;
      else byName.set(r.name, {name: r.name, tp: 0, total: 0, tpRate: null, b: r});
    });
    return [...byName.values()].sort((x, y) => x.name.localeCompare(y.name, undefined, {numeric: true}));
  }
  function drawCriteria() {
    const {ctx, w, h} = setupCanvas(els.criteriaCanvas);
    ctx.clearRect(0, 0, w, h);
    state.hits.criteria = [];
    const rows = criteriaRows();
    if (!rows.length) { emptyNote(ctx, w, h, "No criteria with evaluated frames in this directory."); return; }
    const cmp = comparing();
    const plot = chartPanel(ctx, {x: 0, y: 0, w, h}, "");
    const labelW = 78, valueW = cmp ? 128 : 110;
    const rowH = Math.min(30, Math.max(cmp ? 20 : 15, plot.h / rows.length));
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
      if (!cmp) {
        ctx.fillStyle = missHeat(r.tpRate);
        ctx.fillRect(barX, y, Math.max(2, barW * r.tpRate), bh);
      } else {
        // Two half-height bars: A in its miss-heat colour, B stacked under it, so the
        // pair reads as one row and the gap between their ends is the change.
        const half = Math.max(3, (bh - 2) / 2);
        if (r.tpRate != null) {
          ctx.fillStyle = TH.a("mutedBright", .55);
          ctx.fillRect(barX, y, Math.max(2, barW * r.tpRate), half);
        }
        if (r.b) {
          ctx.fillStyle = missHeat(r.b.tpRate);
          ctx.fillRect(barX, y + half + 2, Math.max(2, barW * r.b.tpRate), half);
        }
      }
      if (isHover) {
        ctx.strokeStyle = TH.c("accent");
        ctx.strokeRect(barX + .5, y + .5, barW - 1, bh - 1);
      }
      ctx.font = "700 10.5px ui-sans-serif, system-ui, sans-serif";
      if (!cmp) {
        ctx.fillStyle = TH.c("muted");
        ctx.fillText(`${rate(r.tpRate)} · ${fmt(r.tp)}/${fmt(r.total)}`, barX + barW + 8, y + bh / 2 + 4);
        state.hits.criteria.push({kind: "criteria", name: r.name, x: plot.x, y, w: plot.w, h: rowH,
          title: r.name, lines: [`TP rate ${rate(r.tpRate)}`, `${fmt(r.tp)} TP of ${fmt(r.total)} frames`]});
      } else {
        const d = delta(r.tpRate, r.b?.tpRate);
        ctx.fillStyle = deltaColor(d);
        ctx.fillText(deltaText(d, {pct: true}), barX + barW + 8, y + bh / 2 + 4);
        state.hits.criteria.push({kind: "criteria", name: r.name, x: plot.x, y, w: plot.w, h: rowH,
          title: r.name,
          lines: [
            `A ${rate(r.tpRate)}${r.total ? ` · ${fmt(r.tp)}/${fmt(r.total)}` : " (absent)"}`,
            `B ${rate(r.b?.tpRate)}${r.b ? ` · ${fmt(r.b.tp)}/${fmt(r.b.total)}` : " (absent)"}`,
            `Δ ${deltaText(d, {pct: true})}`
          ]});
      }
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
  function heatmapDataOf(matrices) {
    if (!matrices) return null;
    const rates = state.matrixMode === "critical" ? matrices.critical_priority : matrices.vehicle_status;
    const counts = state.matrixMode === "critical" ? matrices.critical_priority_counts : matrices.vehicle_status_counts;
    if (!rates || !rates.records || !rates.records.length) return null;
    const cols = rates.columns.filter(c => c !== "Vehicle Status");
    return {cols, rates: rates.records, counts: (counts && counts.records) || []};
  }
  function heatmapData() {
    const a = heatmapDataOf(state.matrices);
    if (!a || !comparing()) return a;
    // B's grid keyed by status name, so a row order difference between the two runs
    // cannot silently pair the wrong statuses.
    const b = heatmapDataOf(state.matricesB);
    const bRows = new Map((b?.rates || []).map(r => [String(r["Vehicle Status"]), r]));
    return {...a, bRows, bCols: new Set(b?.cols || [])};
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
      const bRow = data.bRows ? data.bRows.get(statusName) : null;
      data.cols.forEach((col, j) => {
        const x = gx + j * cw, y = gy + i * ch;
        const v = row[col];
        const bv = bRow ? bRow[col] : null;
        const d = data.bRows ? delta(v, bv) : null;
        const countText = String((data.counts[i] || {})[col] ?? "");
        const evaluated = countText === "" || !/^0\s*\/\s*0$/.test(countText);
        const cellW = cw - 3, cellH = ch - 3;
        const blank = data.bRows ? (d == null) : (v == null || !evaluated);
        if (blank) {
          ctx.fillStyle = TH.a("line", .1);
          ctx.fillRect(x + 1.5, y + 1.5, cellW, cellH);
          ctx.fillStyle = TH.c("muted");
          ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("–", x + cw / 2, y + ch / 2 + 3);
          ctx.textAlign = "left";
        } else if (data.bRows) {
          // Diverging fill: opacity carries the size of the change, hue its direction.
          // A full-scale change is a 25pp swing; beyond that the cell is simply solid.
          const mag = Math.min(1, Math.abs(d) / .25);
          ctx.fillStyle = Math.abs(d) < 1e-9
            ? TH.a("line", .12)
            : (d > 0 ? TH.a("good", .18 + .72 * mag) : TH.a("bad", .18 + .72 * mag));
          ctx.fillRect(x + 1.5, y + 1.5, cellW, cellH);
          ctx.fillStyle = Math.abs(d) < 1e-9 ? TH.c("muted") : "rgb(255 255 255 / .94)";
          ctx.font = "800 11px ui-sans-serif, system-ui, sans-serif";
          ctx.textAlign = "center";
          // One decimal even here: at these TP rates a whole-point rounding turns every
          // real change into "0pp".
          ctx.fillText(deltaText(d, {pct: true}), x + cw / 2, y + ch / 2 + (cellH > 26 ? -1 : 3));
          if (cellH > 26) {
            ctx.fillStyle = Math.abs(d) < 1e-9 ? TH.c("muted") : "rgb(255 255 255 / .72)";
            ctx.font = "600 9px ui-sans-serif, system-ui, sans-serif";
            ctx.fillText(`${Math.round(Number(v) * 100)}→${Math.round(Number(bv) * 100)}`, x + cw / 2, y + ch / 2 + 11);
          }
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
          lines: data.bRows
            ? [`A ${rate(v)}`, `B ${rate(bv)}`, `Δ ${deltaText(d, {pct: true})}`]
            : [v == null ? "not evaluated" : `TP rate ${rate(v)}`, countText ? `frames ${countText}` : ""]});
      });
    });
    // Legend: the miss-heat scale, or the diverging one when cells carry a change.
    const lw = Math.min(140, gw * .4), lx = gx + gw - lw - 56, ly = plot.y + plot.h - 8;
    for (let k = 0; k < lw; k++) {
      const t = k / lw;
      if (data.bRows) {
        const signed = t * 2 - 1;  // -1 = B worse, +1 = B better
        ctx.fillStyle = signed < 0 ? TH.a("bad", .18 + .72 * -signed) : TH.a("good", .18 + .72 * signed);
      } else {
        ctx.fillStyle = missHeat(1 - t);
      }
      ctx.fillRect(lx + k, ly - 6, 1, 6);
    }
    ctx.fillStyle = TH.c("muted");
    ctx.font = "600 9.5px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(data.bRows ? "−25pp" : "miss →", lx - 6, ly);
    ctx.textAlign = "left";
    ctx.fillText(data.bRows ? "+25pp" : "100% TP", lx + lw + 6, ly);
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
    const b = state.summaryB?.stats || {};
    // In compare the headline number stays B's, with the change against A beside it:
    // B is the thing under test, A is the reference.
    const shown = comparing() ? b : s;
    els.kTpRate.textContent = rate(shown.overall_tp_rate);
    els.kTpRateBar.style.width = `${Math.round((Number(shown.overall_tp_rate) || 0) * 100)}%`;
    els.kTpRateBar.style.background = missHeat(shown.overall_tp_rate);
    els.kScenarios.textContent = fmt(shown.num_scenarios);
    els.kFrames.textContent = fmt(shown.total_frames);
    paintDelta(els.dTpRate, delta(s.overall_tp_rate, b.overall_tp_rate), {pct: true});
    paintDelta(els.dScenarios, delta(s.num_scenarios, b.num_scenarios));
    paintDelta(els.dFrames, delta(s.total_frames, b.total_frames));
    els.kBest.textContent = shown.best_criteria ? String(shown.best_criteria).replace("criteria_", "criteria ") : "–";
    els.kBestLabel.textContent = shown.best_criteria ? `best · ${rate(shown.best_tp_rate)}` : "best criteria";
    els.kWorst.textContent = shown.worst_criteria ? String(shown.worst_criteria).replace("criteria_", "criteria ") : "–";
    els.kWorstLabel.textContent = shown.worst_criteria ? `worst · ${rate(shown.worst_tp_rate)}` : "worst criteria";
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
  // In compare each row gains B's numbers and the change, matched on scenario name.
  const SCEN_COMPARE_COLS = [
    {key: "suite", label: "Suite"},
    {key: "scenario", label: "Scenario"},
    {key: "evaluable_frames", label: "Evaluable A", num: true},
    {key: "evaluable_frames_b", label: "Evaluable B", num: true},
    {key: "tp_rate", label: "TP rate A", num: true, rate: true},
    {key: "tp_rate_b", label: "TP rate B", num: true, rate: true},
    {key: "tp_rate_delta", label: "Δ TP rate", num: true, deltaPct: true},
    {key: "fn_delta", label: "Δ FN", num: true, delta: true, invert: true}
  ];
  function scenarioCols() { return comparing() ? SCEN_COMPARE_COLS : SCEN_COLS; }
  function scenarioRows() {
    let rows = (state.summary?.scenario_summary?.records || []).slice();
    if (comparing()) {
      const byName = new Map((state.summaryB?.scenario_summary?.records || [])
        .map(r => [String(r.scenario), r]));
      const seen = new Set();
      rows = rows.map(a => {
        const b = byName.get(String(a.scenario)) || null;
        if (b) seen.add(String(a.scenario));
        return {
          ...a,
          in_a: true, in_b: Boolean(b),
          evaluable_frames_b: b ? b.evaluable_frames : null,
          tp_rate_b: b ? b.tp_rate : null,
          tp_rate_delta: delta(a.tp_rate, b?.tp_rate),
          fn_delta: delta(a.fn_frames, b?.fn_frames)
        };
      });
      // Scenarios only B has: appearing or disappearing is itself a finding.
      (state.summaryB?.scenario_summary?.records || []).forEach(b => {
        if (seen.has(String(b.scenario))) return;
        rows.push({
          ...b, in_a: false, in_b: true,
          tp_rate: null, evaluable_frames: null, fn_frames: null,
          evaluable_frames_b: b.evaluable_frames, tp_rate_b: b.tp_rate,
          tp_rate_delta: null, fn_delta: null
        });
      });
      if (state.scenarioFilter === "changed") {
        rows = rows.filter(r => !r.in_a || !r.in_b || (r.tp_rate_delta != null && Math.abs(r.tp_rate_delta) > 1e-9));
      } else if (state.scenarioFilter === "regressed") {
        rows = rows.filter(r => r.tp_rate_delta != null && r.tp_rate_delta < -1e-9);
      }
    }
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
    const cols = scenarioCols();
    els.scenarioHead.innerHTML = cols.map(c => {
      const sorted = state.sortKey === c.key;
      const arrow = sorted ? `<span class="arrow"> ${state.sortDir > 0 ? "▲" : "▼"}</span>` : "";
      return `<th class="${c.num ? "num" : ""}${sorted ? " sorted" : ""}" data-key="${c.key}">${c.label}${arrow}</th>`;
    }).join("");
    const rows = scenarioRows();
    const filterNote = comparing() && state.scenarioFilter !== "all" ? ` (${state.scenarioFilter})` : "";
    els.scenarioMeta.textContent = rows.length
      ? `${rows.length} scenarios${filterNote} · click a row to open the frame drill-down`
      : "no scenario roll-up for this directory";
    if (!rows.length) {
      const note = comparing() && state.scenarioFilter !== "all"
        ? "No scenario matches this filter — nothing changed between A and B."
        : "No scenarios with per-frame details.";
      els.scenarioBody.innerHTML = `<tr class="empty-row"><td colspan="${cols.length}">${note}</td></tr>`;
      return;
    }
    els.scenarioBody.innerHTML = rows.map(r => {
      const classes = [];
      if (String(r.scenario) === state.scenario) classes.push("active");
      if (comparing() && (!r.in_a || !r.in_b)) classes.push("one-sided");
      const cells = cols.map(c => {
        const v = r[c.key];
        if (c.rate) {
          const pct = v == null ? 0 : Math.max(0, Math.min(1, Number(v)));
          return `<td class="num"><span class="rate-cell"><span class="rate-bar"><i style="width:${Math.round(pct * 100)}%;background:${missHeat(v == null ? 1 : v)}"></i></span>${rate(v)}</span></td>`;
        }
        if (c.deltaPct || c.delta) {
          const text = v == null
            ? (r.in_a ? "only in A" : "new in B")
            : deltaText(v, {pct: Boolean(c.deltaPct)});
          return `<td class="num" style="color:${v == null ? TH.c("muted") : deltaColor(v, c)}">${text}</td>`;
        }
        return `<td class="${c.num ? "num" : ""}">${c.num ? (v == null ? "–" : fmt(v)) : escapeHtml(v)}</td>`;
      }).join("");
      const cls = classes.length ? ` class="${classes.join(" ")}"` : "";
      return `<tr data-scenario="${escapeHtml(r.scenario)}"${cls}>${cells}</tr>`;
    }).join("");
  }

  /* ---------- worst scenarios quick list ---------- */
  function renderWorstList() {
    // In compare the useful shortlist is not "worst in B" but "dropped most from A":
    // a scenario that was always bad is not what a comparison is being read for.
    if (comparing()) {
      els.worstTitle.textContent = "Biggest Regressions";
      const rows = scenarioRows()
        .filter(r => r.tp_rate_delta != null && r.tp_rate_delta < -1e-9)
        .sort((a, b) => a.tp_rate_delta - b.tp_rate_delta)
        .slice(0, 8);
      els.worstList.innerHTML = rows.length
        ? rows.map(r =>
            `<div class="item${String(r.scenario) === state.scenario ? " active" : ""}" data-scenario="${escapeHtml(r.scenario)}" role="button" tabindex="0">` +
            `<span class="name" title="${escapeHtml(r.scenario)}">${escapeHtml(r.scenario)}</span>` +
            `<span class="val" style="color:${deltaColor(r.tp_rate_delta)}">${deltaText(r.tp_rate_delta, {pct: true})}</span></div>`
          ).join("")
        : `<div class="sub">No scenario regressed in B.</div>`;
      return;
    }
    els.worstTitle.textContent = "Worst Scenarios";
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
  function renderCompareChrome() {
    const on = comparing();
    document.body.classList.toggle("comparing", on);
    els.compareBanner.hidden = !on;
    els.swapBtn.hidden = !on;
    els.dirTagA.hidden = !on;
    els.dirTagB.hidden = !on;
    els.scenarioChips.hidden = !on;
    if (on) els.compareBannerText.textContent = `A ${state.path}  vs  B ${state.pathB}`;
  }
  function renderAll() {
    renderCompareChrome();
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
  async function loadCompare(pathB) {
    state.pathB = pathB || "";
    if (!state.pathB) {
      state.summaryB = null;
      state.matricesB = null;
      state.scenarioFilter = "all";
      syncCompareChips();
      saveSession();
      syncUrl();
      renderAll();
      return;
    }
    saveSession();
    syncUrl();
    els.scenarioMeta.textContent = "loading comparison…";
    try {
      const [summaryB, matricesB] = await Promise.all([
        api("/api/tlr_summary", {path: state.pathB}),
        api("/api/tlr_matrices", {path: state.pathB})
      ]);
      state.summaryB = summaryB;
      state.matricesB = matricesB;
      renderAll();
    } catch (err) {
      state.pathB = "";
      state.summaryB = null;
      state.matricesB = null;
      els.compareSelect.value = "";
      renderAll();
      toast(String(err.message || err));
    }
  }
  function syncCompareChips() {
    els.scenarioChips.querySelectorAll(".chip").forEach(chip =>
      chip.classList.toggle("active", (chip.dataset.filter || "all") === state.scenarioFilter));
  }
  function dirOptions(selected, {blankLabel = ""} = {}) {
    const options = state.dirs.map(d =>
      `<option value="${escapeHtml(d.path)}"${d.path === selected ? " selected" : ""}>` +
      `${escapeHtml(d.path)} · ${fmt(d.scenarios)} scenario${d.scenarios === 1 ? "" : "s"}</option>`).join("");
    return blankLabel ? `<option value="">${escapeHtml(blankLabel)}</option>${options}` : options;
  }
  async function loadDirs() {
    const data = await api("/api/tlr_dirs", {});
    state.dirs = data.items || [];
    els.dirSelect.innerHTML = state.dirs.length
      ? dirOptions(state.path)
      : `<option value="">no TLR directories under the data root</option>`;
    els.compareSelect.innerHTML = state.dirs.length > 1
      ? dirOptions(state.pathB, {blankLabel: "— none (single) —"})
      : `<option value="">need a second directory to compare</option>`;
    els.compareSelect.disabled = state.dirs.length < 2;
    return state.dirs;
  }

  /* ---------- events ---------- */
  els.dirSelect.addEventListener("change", () => {
    const path = els.dirSelect.value;
    if (path) loadDirectory(path);
  });
  els.compareSelect.addEventListener("change", () => loadCompare(els.compareSelect.value));
  els.swapBtn.addEventListener("click", async () => {
    // Swap by moving the payloads, not by refetching: both are already in hand.
    const [pathA, summaryA, matricesA] = [state.path, state.summary, state.matrices];
    state.path = state.pathB; state.summary = state.summaryB; state.matrices = state.matricesB;
    state.pathB = pathA; state.summaryB = summaryA; state.matricesB = matricesA;
    els.dirSelect.value = state.path;
    els.compareSelect.value = state.pathB;
    saveSession();
    syncUrl();
    renderAll();
    // The drawer's frames belong to the old A, so reload them against the new one.
    if (state.scenario) await openScenario(state.scenario);
  });
  els.scenarioChips.addEventListener("click", ev => {
    const chip = ev.target.closest(".chip");
    if (!chip) return;
    state.scenarioFilter = chip.dataset.filter || "all";
    syncCompareChips();
    renderScenarioTable();
  });
  els.scenarioChips.addEventListener("keydown", ev => {
    if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); ev.target.click(); }
  });
  els.reloadBtn.addEventListener("click", async () => {
    try {
      const dirs = await loadDirs();
      const still = dirs.some(d => d.path === state.path);
      if (still) els.dirSelect.value = state.path;
      else if (dirs.length) { els.dirSelect.value = dirs[0].path; await loadDirectory(dirs[0].path); }
      // A B that vanished between rescans must not keep showing stale numbers.
      if (state.pathB && !dirs.some(d => d.path === state.pathB)) await loadCompare("");
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
    const wantPathB = q.get("path_b") || (q.get("path") ? "" : session.pathB) || "";
    const wantScenario = q.get("scenario") || (q.get("path") ? "" : session.scenario) || "";
    try {
      const dirs = await loadDirs();
      if (!dirs.length) { els.scenarioMeta.textContent = "no TLR directories under the data root"; renderAll(); return; }
      const chosen = dirs.some(d => d.path === wantPath) ? wantPath : dirs[0].path;
      els.dirSelect.value = chosen;
      await loadDirectory(chosen, {openScenarioName: chosen === wantPath ? wantScenario : ""});
      // B after A, so a failing comparison still leaves a usable single view.
      if (wantPathB && wantPathB !== chosen && dirs.some(d => d.path === wantPathB)) {
        els.compareSelect.value = wantPathB;
        await loadCompare(wantPathB);
      }
    } catch (err) {
      renderAll();
      toast(String(err.message || err));
    }
  })();
})();
