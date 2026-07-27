function chartPanel(rect, title) {
  ctx.fillStyle = "rgba(2,6,23,.42)";
  ctx.strokeStyle = "rgba(148,163,184,.2)";
  ctx.lineWidth = 1;
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  ctx.fillStyle = "#eaf2ff";
  ctx.font = "800 12px Inter, sans-serif";
  ctx.fillText(title, rect.x + 12, rect.y + 18);
  return {x: rect.x + 12, y: rect.y + 32, w: rect.w - 24, h: rect.h - 44};
}
function statsValueText(value, kind = "count") {
  if (kind === "rate") return rate(value);
  if (kind === "pp") return `${Math.round((Number(value) || 0) * 100)}pp`;
  if (kind === "delta") return fmtDelta(Math.round(Number(value) || 0));
  if (kind === "float") return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "-";
  return fmt(Math.round(Number(value) || 0));
}
function statsRowValue(row, key) {
  const value = row && row[key];
  if (key.includes("rate") || ["tpr", "fpr", "precision", "recall"].includes(key)) return rate(value);
  if (key.startsWith("delta_") && ["delta_tpr", "delta_fpr", "delta_precision", "delta_recall"].includes(key)) return statsValueText(value, "pp");
  if (key.startsWith("delta_")) return fmtDelta(Number(value) || 0);
  if (key.includes("error")) return statsValueText(value, "float");
  return typeof value === "number" ? fmt(value) : String(value ?? "-");
}
function addStatsHit(hit) {
  if (!hit) return;
  state.statNodes.push(hit);
}
function addStatsRectHit(x, y, w, h, hit) {
  addStatsHit({...hit, shape: "rect", x, y, w, h});
}
function addStatsPointHit(x, y, r, hit) {
  addStatsHit({...hit, shape: "point", x, y, r});
}
function statsHitContains(hit, x, y) {
  if (hit.shape === "rect") return x >= hit.x && x <= hit.x + hit.w && y >= hit.y && y <= hit.y + hit.h;
  return Math.hypot(x - hit.x, y - hit.y) <= (hit.r || 8);
}
function statsRowsForLabel(label) {
  const labels = (state.stats?.labels || []).filter(r => !label || r.label === label);
  const labelDistance = (state.stats?.label_distance || []).filter(r => !label || r.label === label);
  const scenarios = (state.stats?.label_scenarios || []).filter(r => !label || r.label === label);
  const datasets = (state.stats?.label_datasets || []).filter(r => !label || r.label === label);
  const frames = (state.stats?.label_frames || []).filter(r => !label || r.label === label);
  return [
    ...labels.map(r => ({section: "label", ...r})),
    ...labelDistance.map(r => ({section: "label_distance", ...r})),
    ...scenarios.slice(0, 120).map(r => ({section: "scenario", ...r})),
    ...datasets.slice(0, 120).map(r => ({section: "dataset", ...r})),
    ...frames.slice(0, 120).map(r => ({section: "frame", ...r}))
  ];
}
function statsRowsForDistance(binLabel) {
  const distance = (state.stats?.distance || []).filter(r => r.bin_label === binLabel);
  const labelDistance = (state.stats?.label_distance || []).filter(r => r.bin_label === binLabel);
  return [
    ...distance.map(r => ({section: "distance", ...r})),
    ...labelDistance.map(r => ({section: "label_distance", ...r}))
  ];
}
function statsRowsForScenario(s) {
  if (!s) return [];
  return [
    {
      section: "scenario",
      scenario: scenarioName(s),
      suite_name: s.suite_name || "",
      topic_name: s.topic_name || "",
      frames: s.frames,
      rows: s.rows,
      tp: s.tp,
      fp: s.fp,
      fn: s.fn,
      delta_tp: s.delta_tp,
      delta_fp: s.delta_fp,
      delta_fn: s.delta_fn,
      precision: s.precision,
      recall: s.recall
    },
    ...(s.labels || []).map(r => ({section: "scenario_label", scenario: scenarioName(s), ...r}))
  ];
}
function statsRowCanOpenViewer(row) {
  return row && (row.scenario_name || row.t4dataset_name) && row.topic_name !== undefined;
}
function statsFrameSortValue(row) {
  if (state.statsFrameFocus === "improved") {
    return state.compare
      ? -Math.max(0, -Number(row.delta_fp || 0)) - Math.max(0, -Number(row.delta_fn || 0)) - Math.max(0, Number(row.delta_tp || 0))
      : -Number(row.tp || 0);
  }
  if (state.statsFrameFocus === "abs") {
    return -(
      Math.abs(Number(row.delta_fp || row.fp || 0))
      + Math.abs(Number(row.delta_fn || row.fn || 0))
      + Math.abs(Number(row.delta_tp || row.tp || 0))
    );
  }
  return state.compare
    ? -(Math.max(0, Number(row.delta_fp || 0)) + Math.max(0, Number(row.delta_fn || 0)) + Math.max(0, -Number(row.delta_tp || 0)))
    : -(Number(row.fp || 0) + Number(row.fn || 0));
}
function focusedFrameRows(limit = 24) {
  return [...(state.stats?.frames || [])].sort((a, b) => statsFrameSortValue(a) - statsFrameSortValue(b)).slice(0, limit);
}
function showStatsHover(hit, rect) {
  const lines = hit.lines || [];
  els.hoverCard.innerHTML = `<b>${escapeHtml(hit.title || "Stats")}</b>` + lines.map(line => `<span>${escapeHtml(line)}</span>`).join("");
  els.hoverCard.style.left = `${Math.max(8, Math.min(Math.max(8, rect.width - 340), state.mouseX))}px`;
  els.hoverCard.style.top = `${Math.max(8, Math.min(Math.max(8, rect.height - 140), state.mouseY))}px`;
  els.hoverCard.classList.add("show");
}
function renderStatsDetail() {
  const detail = state.statsDetail;
  els.statsDetail.classList.toggle("show", !!detail);
  if (!detail) return;
  const rows = detail.rows || [];
  els.statsDetail.style.top = `${Math.max(92, els.stage.scrollTop + 92)}px`;
  els.statsDetailTitle.textContent = detail.title || "Stats detail";
  els.statsDetailMeta.textContent = detail.meta || `${rows.length.toLocaleString()} rows`;
  const hasViewerRows = rows.some(statsRowCanOpenViewer);
  const keys = [...new Set(rows.flatMap(row => Object.keys(row)))].filter(k => !["a", "b"].includes(k)).slice(0, hasViewerRows ? 13 : 14);
  if (!rows.length || !keys.length) {
    els.statsDetailTable.innerHTML = `<div class="stats-detail-meta">No detail rows for this item.</div>`;
    return;
  }
  els.statsDetailTable.innerHTML = `<table><thead><tr>${keys.map(k => `<th>${escapeHtml(k)}</th>`).join("")}${hasViewerRows ? "<th>Viewer</th>" : ""}</tr></thead><tbody>` +
    rows.slice(0, 500).map((row, idx) => `<tr>${keys.map(k => `<td>${escapeHtml(statsRowValue(row, k))}</td>`).join("")}` +
      (hasViewerRows ? `<td>${statsRowCanOpenViewer(row) ? `<button class="stats-viewer-link" data-row="${idx}">View</button>` : ""}</td>` : "") +
      `</tr>`).join("") +
    `</tbody></table>`;
}
function openStatsDetail(hit) {
  if (!hit) return;
  if (hit.kind === "scenario" && hit.s) {
    selectScenario(hit.s);
  }
  state.statsDetail = {
    title: hit.detailTitle || hit.title || "Stats detail",
    meta: hit.detailMeta || `${(hit.rows || []).length.toLocaleString()} rows`,
    rows: hit.rows || []
  };
  renderStatsDetail();
}
function statsRowsToCsv(rows) {
  const keys = [...new Set((rows || []).flatMap(row => Object.keys(row)))].filter(k => !["a", "b"].includes(k));
  const esc = value => `"${String(value ?? "").replace(/"/g, '""')}"`;
  return [keys.join(","), ...(rows || []).map(row => keys.map(k => esc(row[k])).join(","))].join("\n");
}
function downloadStatsDetailCsv() {
  const rows = state.statsDetail?.rows || [];
  if (!rows.length) { toast("No stats detail rows to download."); return; }
  const blob = new Blob([statsRowsToCsv(rows)], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${(state.statsDetail.title || "stats_detail").replace(/[^a-z0-9_-]+/gi, "_").slice(0, 80)}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
function openStatsDetailRowViewer(idx) {
  const row = state.statsDetail?.rows?.[idx];
  if (!statsRowCanOpenViewer(row)) return;
  const s = {
    suite_name: row.suite_name || "",
    scenario_name: row.scenario_name || "",
    t4dataset_name: row.t4dataset_name || "",
    topic_name: row.topic_name || "",
    frames: row.frames || 0,
    fp: row.fp || 0,
    fn: row.fn || 0,
    tp: row.tp || 0,
    delta_fp: row.delta_fp || 0,
    delta_fn: row.delta_fn || 0,
    delta_tp: row.delta_tp || 0
  };
  state.selected = s;
  openViewer(row.frame != null ? row.frame : null);
}
function drawStackedLabelStats(rect, rows) {
  const plot = chartPanel(rect, state.compare ? "Label Delta Distribution" : "Label Distribution");
  const labels = rows.filter(r => r.rows || r.tp || r.fp || r.fn || r.delta_fp || r.delta_fn);
  const max = Math.max(1, ...labels.map(r => state.compare ? Math.max(Math.abs(r.delta_fp || 0), Math.abs(r.delta_fn || 0), Math.abs(r.delta_tp || 0)) : (r.tp || 0) + (r.fp || 0) + (r.fn || 0)));
  const rowH = Math.max(18, Math.min(30, plot.h / Math.max(1, labels.length)));
  labels.forEach((r, i) => {
    const y = plot.y + i * rowH + 3;
    ctx.fillStyle = r.label === state.label ? "#ffffff" : "#cbd5e1";
    ctx.font = `${r.label === state.label ? "800" : "700"} 11px Inter, sans-serif`;
    ctx.fillText(r.label || "unknown", plot.x, y + 10);
    const bx = plot.x + 88, bw = Math.max(20, plot.w - 152), bh = Math.max(8, rowH - 9);
    if (state.compare) {
      const mid = bx + bw / 2;
      ctx.fillStyle = "rgba(148,163,184,.18)";
      ctx.fillRect(bx, y + 2, bw, bh);
      ctx.fillStyle = "rgba(226,232,240,.52)";
      ctx.fillRect(mid, y + 1, 1, bh + 2);
      const drawDelta = (v, dy, colorPos, colorNeg) => {
        const w = Math.min(bw / 2, Math.abs(v || 0) / max * bw / 2);
        ctx.fillStyle = v >= 0 ? colorPos : colorNeg;
        ctx.fillRect(v >= 0 ? mid : mid - w, y + 2 + dy, w, Math.max(3, bh / 2 - 1));
      };
      drawDelta(r.delta_fp || 0, 0, "rgba(251,113,133,.82)", "rgba(52,211,153,.72)");
      drawDelta(r.delta_fn || 0, bh / 2, "rgba(251,191,36,.82)", "rgba(52,211,153,.58)");
      ctx.fillStyle = "#91a4bf";
      ctx.fillText(`FP ${fmtDelta(r.delta_fp || 0)}  FN ${fmtDelta(r.delta_fn || 0)}`, bx + bw + 8, y + 10);
    } else {
      const total = Math.max(1, (r.tp || 0) + (r.fp || 0) + (r.fn || 0));
      const tpW = bw * (r.tp || 0) / total;
      const fpW = bw * (r.fp || 0) / total;
      const fnW = bw * (r.fn || 0) / total;
      ctx.fillStyle = "rgba(15,23,42,.8)";
      ctx.fillRect(bx, y + 2, bw, bh);
      ctx.fillStyle = "#38bdf8"; ctx.fillRect(bx, y + 2, tpW, bh);
      ctx.fillStyle = "#fb7185"; ctx.fillRect(bx + tpW, y + 2, fpW, bh);
      ctx.fillStyle = "#fbbf24"; ctx.fillRect(bx + tpW + fpW, y + 2, fnW, bh);
      ctx.fillStyle = "#91a4bf";
      ctx.fillText(`${fmt(r.tp)} / ${fmt(r.fp)} / ${fmt(r.fn)}`, bx + bw + 8, y + 10);
    }
  });
}
function drawRateRadar(rect, rows) {
  const plot = chartPanel(rect, state.compare ? "Precision / Recall Delta" : "Precision / Recall By Label");
  const labels = rows.filter(r => r.rows || r.tp || r.fp || r.fn).slice(0, 10);
  const cx = plot.x + plot.w / 2, cy = plot.y + plot.h / 2 + 4;
  const radius = Math.max(40, Math.min(plot.w, plot.h) * .38);
  ctx.strokeStyle = "rgba(148,163,184,.18)";
  ctx.fillStyle = "rgba(145,164,191,.66)";
  ctx.font = "700 10px Inter, sans-serif";
  for (let ring = 1; ring <= 4; ring++) {
    ctx.beginPath(); ctx.arc(cx, cy, radius * ring / 4, 0, Math.PI * 2); ctx.stroke();
  }
  labels.forEach((r, i) => {
    const a = -Math.PI / 2 + i * Math.PI * 2 / Math.max(1, labels.length);
    const x = cx + Math.cos(a) * radius;
    const y = cy + Math.sin(a) * radius;
    ctx.strokeStyle = "rgba(148,163,184,.14)";
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
    ctx.fillStyle = r.label === state.label ? "#ffffff" : "#91a4bf";
    ctx.textAlign = x < cx - 6 ? "right" : (x > cx + 6 ? "left" : "center");
    ctx.fillText((r.label || "unknown").slice(0, 12), x, y + (y < cy ? -5 : 12));
  });
  const pointFor = (value, i) => {
    const a = -Math.PI / 2 + i * Math.PI * 2 / Math.max(1, labels.length);
    const rr = Math.max(0, Math.min(1, value || 0)) * radius;
    return [cx + Math.cos(a) * rr, cy + Math.sin(a) * rr];
  };
  const drawPoly = (key, color) => {
    ctx.strokeStyle = color; ctx.fillStyle = color.replace(")", ",.12)").replace("rgb", "rgba");
    ctx.lineWidth = 2;
    ctx.beginPath();
    labels.forEach((r, i) => {
      const tp = Number(r.tp) || 0, fp = Number(r.fp) || 0, fn = Number(r.fn) || 0;
      const value = key === "precision" ? tp / Math.max(1, tp + fp) : tp / Math.max(1, tp + fn);
      const p = pointFor(value, i);
      if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]);
    });
    ctx.closePath(); ctx.stroke(); ctx.fill();
  };
  if (labels.length) {
    drawPoly("precision", "rgb(56,189,248)");
    drawPoly("recall", "rgb(52,211,153)");
  }
  ctx.textAlign = "left";
  ctx.fillStyle = "#38bdf8"; ctx.fillText("precision", plot.x, plot.y + 10);
  ctx.fillStyle = "#34d399"; ctx.fillText("recall", plot.x + 70, plot.y + 10);
}
function drawScenarioScatter(rect, arr) {
  const plot = chartPanel(rect, state.compare ? "Scenario Delta Field" : "Scenario FP / FN Field");
  state.statNodes = [];
  const maxX = Math.max(1, ...arr.map(s => Math.abs(state.compare ? s.delta_fp || 0 : s.fp || 0)));
  const maxY = Math.max(1, ...arr.map(s => Math.abs(state.compare ? s.delta_fn || 0 : s.fn || 0)));
  ctx.strokeStyle = "rgba(148,163,184,.16)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const x = plot.x + i * plot.w / 4;
    const y = plot.y + i * plot.h / 4;
    ctx.beginPath(); ctx.moveTo(x, plot.y); ctx.lineTo(x, plot.y + plot.h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
  }
  if (state.compare) {
    ctx.strokeStyle = "rgba(226,232,240,.34)";
    ctx.beginPath(); ctx.moveTo(plot.x + plot.w / 2, plot.y); ctx.lineTo(plot.x + plot.w / 2, plot.y + plot.h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(plot.x, plot.y + plot.h / 2); ctx.lineTo(plot.x + plot.w, plot.y + plot.h / 2); ctx.stroke();
  }
  arr.forEach(s => {
    const vx = state.compare ? s.delta_fp || 0 : s.fp || 0;
    const vy = state.compare ? s.delta_fn || 0 : s.fn || 0;
    const x = state.compare ? plot.x + plot.w / 2 + (vx / maxX) * plot.w * .46 : plot.x + (vx / maxX) * plot.w;
    const y = state.compare ? plot.y + plot.h / 2 - (vy / maxY) * plot.h * .46 : plot.y + plot.h - (vy / maxY) * plot.h;
    const size = Math.max(3.2, Math.min(11, 3 + Math.sqrt(Math.max(0, s.frames || s.rows || 0)) * .18));
    const selected = state.selected && scenarioKey(state.selected) === scenarioKey(s);
    ctx.fillStyle = selected ? "#ffffff" : colorFor(vx + vy, Math.max(maxX, maxY));
    ctx.globalAlpha = selected ? 1 : .72;
    ctx.beginPath(); ctx.arc(x, y, selected ? size + 3 : size, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;
    addStatsPointHit(x, y, size + 6, {
      kind: "scenario",
      s,
      title: scenarioName(s),
      lines: [
        `${s.suite_name || ""} · ${fmt(s.frames)} frames`,
        state.compare ? `ΔFP ${fmtDelta(s.delta_fp)} · ΔFN ${fmtDelta(s.delta_fn)} · ΔTP ${fmtDelta(s.delta_tp)}` : `FP ${fmt(s.fp)} · FN ${fmt(s.fn)} · TP ${fmt(s.tp)}`,
        `${compareLensLabel()} ${lensMetricText(scenarioMetric(s))}`,
        "Click to load this scenario and show detail rows."
      ],
      detailTitle: `Scenario · ${scenarioName(s)}`,
      rows: statsRowsForScenario(s)
    });
  });
  ctx.fillStyle = "#91a4bf";
  ctx.font = "700 10px Inter, sans-serif";
  ctx.fillText(state.compare ? "left/down improves, right/up worsens" : "x FP, y FN, size frames", plot.x, plot.y + plot.h - 4);
}
function drawScenarioRanking(rect, arr) {
  const plot = chartPanel(rect, state.compare ? "Largest Scenario Changes" : "Scenario Issue Ranking");
  const top = arr.slice(0, 12);
  const max = Math.max(1, ...top.map(s => Math.abs(scenarioMetric(s))));
  const rowH = Math.max(16, Math.min(27, plot.h / Math.max(1, top.length)));
  top.forEach((s, i) => {
    const y = plot.y + i * rowH + 2;
    const v = scenarioMetric(s);
    ctx.fillStyle = state.selected && scenarioKey(state.selected) === scenarioKey(s) ? "#ffffff" : "#cbd5e1";
    ctx.font = "800 10px Inter, sans-serif";
    ctx.fillText(scenarioName(s).slice(0, 28), plot.x, y + 10);
    const bx = plot.x + Math.min(210, plot.w * .46);
    const bw = Math.max(70, plot.w - (bx - plot.x) - 46);
    ctx.fillStyle = "rgba(15,23,42,.78)";
    ctx.fillRect(bx, y + 2, bw, Math.max(8, rowH - 8));
    ctx.fillStyle = colorFor(v, max);
    ctx.fillRect(bx, y + 2, bw * Math.abs(v) / max, Math.max(8, rowH - 8));
    ctx.fillStyle = "#91a4bf";
    ctx.fillText(state.lens.endsWith("r") ? rate(v) : (state.compare ? fmtDelta(Math.round(v)) : fmt(Math.round(v))), bx + bw + 8, y + 10);
  });
}
function drawStatsCurve(rect) {
  const plot = chartPanel(rect, "Selected Scenario Frames");
  if (!state.curve.length) {
    ctx.fillStyle = "#91a4bf";
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText("Select a scenario to load frame data.", plot.x, plot.y + 30);
    return;
  }
  const max = Math.max(1, ...state.curve.map(f => Math.max(Math.abs(f.tp || 0), Math.abs(f.fp || 0), Math.abs(f.fn || 0))));
  ctx.strokeStyle = "rgba(148,163,184,.16)";
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + i * plot.h / 3;
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
  }
  const drawLine = (key, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = key === "tp" ? 2.2 : 1.7; ctx.beginPath();
    state.curve.forEach((f, i) => {
      const x = plot.x + i * plot.w / Math.max(1, state.curve.length - 1);
      const y = state.compare
        ? plot.y + plot.h / 2 - ((Number(f[key]) || 0) / max) * plot.h * .46
        : plot.y + plot.h - ((Number(f[key]) || 0) / max) * plot.h;
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
  };
  drawLine("tp", "#38bdf8"); drawLine("fp", "#fb7185"); drawLine("fn", "#fbbf24");
  const frame = currentPreviewFrameNumber();
  const x = curveXForFrame(frame, {x: plot.x, y: plot.y, w: plot.w, h: plot.h});
  if (x != null) {
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath(); ctx.moveTo(x, plot.y); ctx.lineTo(x, plot.y + plot.h); ctx.stroke();
  }
}
function niceMax(v) {
  const n = Math.max(1, Number(v) || 1);
  const pow = Math.pow(10, Math.floor(Math.log10(n)));
  const x = n / pow;
  return (x <= 2 ? 2 : x <= 5 ? 5 : 10) * pow;
}
function drawPlotFrame(plot, xTitle, yTitle, yMax, yFmt = v => fmt(Math.round(v))) {
  ctx.strokeStyle = "rgba(148,163,184,.28)";
  ctx.fillStyle = "#91a4bf";
  ctx.lineWidth = 1;
  ctx.font = "700 10px Inter, sans-serif";
  ctx.beginPath(); ctx.moveTo(plot.x, plot.y); ctx.lineTo(plot.x, plot.y + plot.h); ctx.lineTo(plot.x + plot.w, plot.y + plot.h); ctx.stroke();
  for (let i = 0; i <= 4; i++) {
    const y = plot.y + plot.h - i * plot.h / 4;
    const v = yMax * i / 4;
    ctx.strokeStyle = "rgba(148,163,184,.12)";
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
    ctx.fillStyle = "#91a4bf";
    ctx.textAlign = "right";
    ctx.fillText(yFmt(v), plot.x - 7, y + 3);
  }
  ctx.textAlign = "center";
  ctx.fillText(xTitle, plot.x + plot.w / 2, plot.y + plot.h + 34);
  ctx.save();
  ctx.translate(plot.x - 44, plot.y + plot.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yTitle, 0, 0);
  ctx.restore();
  ctx.textAlign = "left";
}
function drawDistanceRates(rect, stats) {
  const plot = chartPanel(rect, state.compare ? "Distance Rates: Run B - Run A" : "Distance Rates");
  const rows = (stats?.distance || []).filter(r => r.bin_label);
  if (!rows.length) {
    ctx.fillStyle = "#91a4bf"; ctx.fillText(stats?.error || "No distance-bin data.", plot.x, plot.y + 28); return;
  }
  const inner = {x: plot.x + 48, y: plot.y + 8, w: plot.w - 58, h: plot.h - 48};
  const keys = state.compare ? [
    ["delta_tpr", "ΔTP rate", "#34d399"],
    ["delta_fpr", "ΔFP rate", "#fb7185"],
  ] : [
    ["tpr", "TP rate", "#38bdf8"],
    ["fpr", "FP rate", "#fb7185"],
  ];
  const yMax = state.compare ? Math.max(.05, ...rows.flatMap(r => keys.map(k => Math.abs(Number(r[k[0]]) || 0)))) : 1;
  drawPlotFrame(inner, "Distance bin", state.compare ? "Rate delta" : "Rate", yMax, v => state.compare ? `${Math.round(v * 100)}pp` : `${Math.round(v * 100)}%`);
  if (state.compare) {
    const midY = inner.y + inner.h / 2;
    ctx.strokeStyle = "rgba(255,255,255,.36)";
    ctx.beginPath(); ctx.moveTo(inner.x, midY); ctx.lineTo(inner.x + inner.w, midY); ctx.stroke();
  }
  keys.forEach(([key, name, color], ki) => {
    if (state.statsDistanceStyle !== "bar") {
      ctx.strokeStyle = color; ctx.lineWidth = 2.4; ctx.beginPath();
      rows.forEach((r, i) => {
        const x = inner.x + i * inner.w / Math.max(1, rows.length - 1);
        const raw = Number(r[key]) || 0;
        const y = state.compare
          ? inner.y + inner.h / 2 - raw / yMax * inner.h * .46
          : inner.y + inner.h - raw / yMax * inner.h;
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.stroke();
    }
    rows.forEach((r, i) => {
      const x = state.statsDistanceStyle === "bar"
        ? inner.x + (i + .5) * inner.w / Math.max(1, rows.length) + (ki - .5) * Math.max(4, inner.w / Math.max(1, rows.length) * .24)
        : inner.x + i * inner.w / Math.max(1, rows.length - 1);
      const raw = Number(r[key]) || 0;
      const y = state.compare
        ? inner.y + inner.h / 2 - raw / yMax * inner.h * .46
        : inner.y + inner.h - raw / yMax * inner.h;
      if (state.statsDistanceStyle === "bar") {
        const barW = Math.max(4, inner.w / Math.max(1, rows.length) * .2);
        const baseY = state.compare ? inner.y + inner.h / 2 : inner.y + inner.h;
        ctx.fillStyle = color;
        ctx.fillRect(x - barW / 2, Math.min(baseY, y), barW, Math.max(2, Math.abs(baseY - y)));
        addStatsRectHit(x - barW / 2, Math.min(baseY, y), barW, Math.max(4, Math.abs(baseY - y)), {
          title: `${name} · ${r.bin_label}`,
          lines: [
            `${name}: ${state.compare ? statsValueText(raw, "pp") : rate(raw)}`,
            `TP ${fmt(r.tp)} · FP ${fmt(r.fp)} · FN ${fmt(r.fn)}`,
            "Click to inspect this distance bin."
          ],
          detailTitle: `Distance bin · ${r.bin_label}`,
          detailMeta: `${name} in ${r.bin_label}`,
          rows: statsRowsForDistance(r.bin_label)
        });
      } else {
        ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
        addStatsPointHit(x, y, 9, {
        title: `${name} · ${r.bin_label}`,
        lines: [
          `${name}: ${state.compare ? statsValueText(raw, "pp") : rate(raw)}`,
          `TP ${fmt(r.tp)} · FP ${fmt(r.fp)} · FN ${fmt(r.fn)}`,
          "Click to inspect this distance bin."
        ],
        detailTitle: `Distance bin · ${r.bin_label}`,
        detailMeta: `${name} in ${r.bin_label}`,
        rows: statsRowsForDistance(r.bin_label)
        });
      }
      if (i % 3 === ki) {
        ctx.fillStyle = "#cbd5e1"; ctx.font = "700 9px Inter, sans-serif"; ctx.textAlign = "center";
        ctx.fillText(state.compare ? `${Math.round(raw * 100)}pp` : `${Math.round(raw * 100)}%`, x, y - 7);
      }
    });
    ctx.fillStyle = color; ctx.font = "800 11px Inter, sans-serif"; ctx.textAlign = "left";
    ctx.fillText(name, inner.x + 8 + ki * 82, inner.y + 13);
  });
  ctx.fillStyle = "#91a4bf"; ctx.font = "700 9px Inter, sans-serif"; ctx.textAlign = "center";
  rows.forEach((r, i) => {
    if (i % 2 && rows.length > 9) return;
    const x = state.statsDistanceStyle === "bar" ? inner.x + (i + .5) * inner.w / Math.max(1, rows.length) : inner.x + i * inner.w / Math.max(1, rows.length - 1);
    ctx.save(); ctx.translate(x, inner.y + inner.h + 10); ctx.rotate(-Math.PI / 5); ctx.fillText(r.bin_label, 0, 0); ctx.restore();
  });
  ctx.textAlign = "left";
}
function drawObjectCountByDistance(rect, stats) {
  const plot = chartPanel(rect, "Object Count By Distance / Label");
  const rows = (stats?.label_distance || []).filter(r => r.bin_label && r.label);
  if (!rows.length) { ctx.fillStyle = "#91a4bf"; ctx.fillText("No object count by distance.", plot.x, plot.y + 28); return; }
  const labels = allLabelNames().filter(l => rows.some(r => r.label === l && Number(r.rows)));
  const bins = [...new Map(rows.map(r => [r.bin_label, r])).values()].sort((a, b) => (a.bin_idx || 0) - (b.bin_idx || 0));
  const totals = bins.map(b => rows.filter(r => r.bin_label === b.bin_label).reduce((n, r) => n + (Number(state.compare ? r.delta_rows : r.rows) || 0), 0));
  const yMax = state.compare ? Math.max(1, ...totals.map(Math.abs)) : niceMax(Math.max(1, ...totals));
  const inner = {x: plot.x + 50, y: plot.y + 8, w: plot.w - 60, h: plot.h - 48};
  drawPlotFrame(inner, "Distance bin", state.compare ? "Count delta" : "Count", yMax);
  const barW = Math.max(5, inner.w / Math.max(1, bins.length) * .64);
  bins.forEach((b, i) => {
    const x = inner.x + (i + .5) * inner.w / Math.max(1, bins.length);
    let posY = inner.y + inner.h, negY = inner.y + inner.h;
    labels.forEach((lab, li) => {
      const row = rows.find(r => r.bin_label === b.bin_label && r.label === lab);
      const v = Number(row?.[state.compare ? "delta_rows" : "rows"]) || 0;
      const h = Math.abs(v) / yMax * inner.h;
      ctx.fillStyle = ["#38bdf8", "#fb7185", "#fbbf24", "#34d399", "#a78bfa", "#f97316", "#e879f9", "#94a3b8"][li % 8];
      if (state.compare && v < 0) { ctx.fillRect(x - barW / 2, negY, barW, h); negY += h; }
      else { posY -= h; ctx.fillRect(x - barW / 2, posY, barW, h); }
    });
    addStatsRectHit(x - barW / 2, inner.y, barW, inner.h, {
      title: `Objects · ${b.bin_label}`,
      lines: [
        `${state.compare ? "Count change" : "Objects"}: ${state.compare ? fmtDelta(totals[i]) : fmt(totals[i])}`,
        `${labels.slice(0, 4).join(", ")}${labels.length > 4 ? "..." : ""}`,
        "Click to inspect labels in this distance bin."
      ],
      detailTitle: `Object count · ${b.bin_label}`,
      rows: statsRowsForDistance(b.bin_label)
    });
    if (i % 2 === 0 || bins.length <= 9) {
      ctx.fillStyle = "#91a4bf"; ctx.font = "700 9px Inter, sans-serif"; ctx.textAlign = "center";
      ctx.save(); ctx.translate(x, inner.y + inner.h + 10); ctx.rotate(-Math.PI / 5); ctx.fillText(b.bin_label, 0, 0); ctx.restore();
    }
  });
  ctx.textAlign = "left";
  labels.slice(0, 6).forEach((lab, i) => {
    ctx.fillStyle = ["#38bdf8", "#fb7185", "#fbbf24", "#34d399", "#a78bfa", "#f97316"][i % 6];
    ctx.fillRect(inner.x + 8 + i * 78, inner.y + 8, 8, 8);
    ctx.fillStyle = "#cbd5e1"; ctx.font = "700 10px Inter, sans-serif"; ctx.fillText(lab, inner.x + 20 + i * 78, inner.y + 16);
  });
}
function drawLabelDistanceHeatmap(rect, stats, metric) {
  const title = state.compare ? `${metric.toUpperCase()} Delta By Label / Distance` : `${metric.toUpperCase()} By Label / Distance`;
  const plot = chartPanel(rect, title);
  const rows = (stats?.label_distance || []).filter(r => r.bin_label && r.label);
  if (!rows.length) { ctx.fillStyle = "#91a4bf"; ctx.fillText("No label-distance data.", plot.x, plot.y + 28); return; }
  const bins = [...new Map(rows.map(r => [r.bin_label, r])).values()].sort((a, b) => (a.bin_idx || 0) - (b.bin_idx || 0));
  const labels = allLabelNames().filter(l => rows.some(r => r.label === l && (Number(r.rows) || Number(r.delta_rows)))).slice(0, 12);
  const inner = {x: plot.x + 76, y: plot.y + 8, w: plot.w - 90, h: plot.h - 58};
  const cellW = inner.w / Math.max(1, bins.length);
  const cellH = inner.h / Math.max(1, labels.length);
  const key = state.compare ? `delta_${metric}` : metric;
  const maxAbs = state.compare ? Math.max(.01, ...rows.map(r => Math.abs(Number(r[key]) || 0))) : 1;
  labels.forEach((lab, yi) => {
    ctx.fillStyle = lab === state.label ? "#ffffff" : "#cbd5e1";
    ctx.font = "800 10px Inter, sans-serif"; ctx.textAlign = "right";
    ctx.fillText(lab.slice(0, 12), inner.x - 7, inner.y + yi * cellH + cellH * .62);
    bins.forEach((bin, xi) => {
      const row = rows.find(r => r.label === lab && r.bin_label === bin.bin_label);
      const v = row ? Number(row[key]) : null;
      const x = inner.x + xi * cellW, y = inner.y + yi * cellH;
      if (v == null || !Number.isFinite(v)) ctx.fillStyle = "rgba(15,23,42,.8)";
      else if (state.compare) ctx.fillStyle = v < 0
        ? `rgba(52,211,153,${.18 + Math.abs(v) / maxAbs * .72})`
        : `rgba(251,113,133,${.18 + Math.abs(v) / maxAbs * .72})`;
      else ctx.fillStyle = metric === "tpr"
        ? `rgba(56,189,248,${.12 + Math.max(0, v) * .76})`
        : `rgba(251,113,133,${.12 + Math.max(0, v) * .76})`;
      ctx.fillRect(x + 1, y + 1, Math.max(1, cellW - 2), Math.max(1, cellH - 2));
      if (cellW > 34 && cellH > 16 && v != null && Number.isFinite(v)) {
        ctx.fillStyle = Math.abs(v) > .55 && !state.compare ? "#07111f" : "#eaf2ff";
        ctx.font = "800 9px Inter, sans-serif"; ctx.textAlign = "center";
        ctx.fillText(state.compare ? `${Math.round(v * 100)}pp` : `${Math.round(v * 100)}%`, x + cellW / 2, y + cellH / 2 + 3);
      }
      addStatsRectHit(x + 1, y + 1, Math.max(1, cellW - 2), Math.max(1, cellH - 2), {
        title: `${lab} · ${bin.bin_label}`,
        lines: [
          `${metric.toUpperCase()}: ${state.compare ? statsValueText(v, "pp") : rate(v)}`,
          `Rows ${fmt(row?.rows)}${state.compare ? ` · Δrows ${fmtDelta(row?.delta_rows || 0)}` : ""}`,
          "Click to inspect this label and distance bin."
        ],
        detailTitle: `${metric.toUpperCase()} · ${lab} · ${bin.bin_label}`,
        rows: row ? [{section: "label_distance", ...row}] : []
      });
    });
  });
  ctx.fillStyle = "#91a4bf"; ctx.font = "700 9px Inter, sans-serif"; ctx.textAlign = "center";
  bins.forEach((bin, i) => {
    if (i % 2 && bins.length > 9) return;
    const x = inner.x + i * cellW + cellW / 2;
    ctx.save(); ctx.translate(x, inner.y + inner.h + 11); ctx.rotate(-Math.PI / 5); ctx.fillText(bin.bin_label, 0, 0); ctx.restore();
  });
  ctx.textAlign = "left";
  ctx.fillStyle = "#91a4bf"; ctx.fillText("Label", plot.x, inner.y + inner.h / 2);
  ctx.fillText("Distance bin", inner.x + inner.w / 2 - 26, plot.y + plot.h - 4);
}
function drawErrorBars(rect, stats) {
  const plot = chartPanel(rect, state.compare ? "TP Localization Error Delta" : "TP Localization Error By Label");
  const rows = (stats?.errors || []).filter(r => r.label && (r.mean_abs_x_error != null || r.mean_abs_y_error != null || r.mean_abs_yaw_error != null || r.delta_mean_abs_x_error != null));
  if (!rows.length) { ctx.fillStyle = "#91a4bf"; ctx.fillText("No TP error columns found.", plot.x, plot.y + 28); return; }
  const labels = rows.map(r => r.label).slice(0, 10);
  const keys = state.compare
    ? [["delta_mean_abs_x_error", "Δ|x|", "#38bdf8"], ["delta_mean_abs_y_error", "Δ|y|", "#34d399"], ["delta_mean_abs_yaw_error", "Δ|yaw|", "#fbbf24"]]
    : [["mean_abs_x_error", "|x|", "#38bdf8"], ["mean_abs_y_error", "|y|", "#34d399"], ["mean_abs_yaw_error", "|yaw|", "#fbbf24"]];
  const max = Math.max(.01, ...rows.flatMap(r => keys.map(k => Math.abs(Number(r[k[0]]) || 0))));
  const inner = {x: plot.x + 50, y: plot.y + 8, w: plot.w - 60, h: plot.h - 52};
  drawPlotFrame(inner, "Label", state.compare ? "Error delta" : "Mean absolute error", state.compare ? max : niceMax(max), v => v.toFixed(2));
  const groupW = inner.w / Math.max(1, labels.length);
  const barW = Math.max(4, groupW / (keys.length + 1));
  labels.forEach((lab, i) => {
    const row = rows.find(r => r.label === lab) || {};
    keys.forEach(([key, name, color], ki) => {
      const v = Number(row[key]) || 0;
      const h = Math.abs(v) / (state.compare ? max : niceMax(max)) * inner.h;
      const x = inner.x + i * groupW + groupW / 2 + (ki - 1) * barW;
      ctx.fillStyle = color;
      ctx.fillRect(x - barW / 2, inner.y + inner.h - h, barW * .82, h);
      addStatsRectHit(x - barW / 2, inner.y + inner.h - h, barW * .82, Math.max(4, h), {
        title: `${lab} · ${name}`,
        lines: [
          `${name}: ${state.compare ? statsValueText(v, "float") : statsValueText(v, "float")}`,
          "Click to inspect label error rows."
        ],
        detailTitle: `TP localization error · ${lab}`,
        rows: rows.filter(item => item.label === lab).map(item => ({section: "error", ...item}))
      });
    });
    ctx.fillStyle = "#91a4bf"; ctx.font = "700 9px Inter, sans-serif"; ctx.textAlign = "center";
    ctx.save(); ctx.translate(inner.x + i * groupW + groupW / 2, inner.y + inner.h + 11); ctx.rotate(-Math.PI / 5); ctx.fillText(lab.slice(0, 10), 0, 0); ctx.restore();
  });
  ctx.textAlign = "left";
  keys.forEach(([_, name, color], i) => {
    ctx.fillStyle = color; ctx.fillRect(inner.x + 8 + i * 62, inner.y + 8, 8, 8);
    ctx.fillStyle = "#cbd5e1"; ctx.font = "700 10px Inter, sans-serif"; ctx.fillText(name, inner.x + 20 + i * 62, inner.y + 16);
  });
}
function drawDatasetKpis(rect, stats, arr) {
  const plot = chartPanel(rect, state.compare ? "Dataset Summary Delta" : "Dataset Summary");
  const labels = stats?.labels || [];
  const totals = labels.reduce((o, r) => {
    ["tp", "fp", "fn", "gt", "est", "rows", "delta_tp", "delta_fp", "delta_fn", "delta_gt", "delta_est", "delta_rows"].forEach(k => o[k] = (o[k] || 0) + (Number(r[k]) || 0));
    return o;
  }, {});
  const tp = state.compare ? totals.delta_tp : totals.tp;
  const fp = state.compare ? totals.delta_fp : totals.fp;
  const fn = state.compare ? totals.delta_fn : totals.fn;
  const precision = totals.tp / Math.max(1, totals.tp + totals.fp);
  const recall = totals.tp / Math.max(1, totals.tp + totals.fn);
  const cards = [
    ["Scenarios", fmt(arr.length), "#eaf2ff", state.stats?.scenarios || []],
    [state.compare ? "ΔTP" : "TP", state.compare ? fmtDelta(tp) : fmt(tp), "#38bdf8", state.stats?.label_frames || state.stats?.frames || []],
    [state.compare ? "ΔFP" : "FP", state.compare ? fmtDelta(fp) : fmt(fp), "#fb7185", state.stats?.label_frames || state.stats?.frames || []],
    [state.compare ? "ΔFN" : "FN", state.compare ? fmtDelta(fn) : fmt(fn), "#fbbf24", state.stats?.label_frames || state.stats?.frames || []],
    ["Precision", rate(precision), "#34d399", state.stats?.labels || []],
    ["Recall", rate(recall), "#a78bfa", state.stats?.labels || []],
  ];
  const cols = 3;
  const gap = 10;
  const w = (plot.w - gap * (cols - 1)) / cols;
  const h = (plot.h - gap) / 2;
  cards.forEach(([name, value, color, rows], i) => {
    const x = plot.x + (i % cols) * (w + gap);
    const y = plot.y + Math.floor(i / cols) * (h + gap);
    ctx.fillStyle = "rgba(15,23,42,.7)";
    ctx.strokeStyle = "rgba(148,163,184,.18)";
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.font = "900 22px Inter, sans-serif";
    ctx.fillText(value, x + 12, y + 34);
    ctx.fillStyle = "#91a4bf";
    ctx.font = "800 10px Inter, sans-serif";
    ctx.fillText(name, x + 12, y + h - 12);
    addStatsRectHit(x, y, w, h, {
      title: `Dataset summary · ${name}`,
      lines: [`${name}: ${value}`, "Click to inspect rows behind this metric."],
      detailTitle: `Dataset summary · ${name}`,
      rows: (rows || []).slice(0, 500).map(r => ({section: name, ...r}))
    });
  });
}
function drawLabelRateBars(rect, stats) {
  const plot = chartPanel(rect, state.compare ? "Label Rate Delta" : "Precision / Recall By Label");
  const rows = (stats?.labels || []).filter(r => r.label).slice().sort((a, b) => (b.rows || 0) - (a.rows || 0));
  if (!rows.length) { ctx.fillStyle = "#91a4bf"; ctx.fillText("No label rates.", plot.x, plot.y + 28); return; }
  const inner = {x: plot.x + 72, y: plot.y + 8, w: plot.w - 88, h: plot.h - 42};
  const labels = rows.slice(0, 12);
  const rowH = inner.h / Math.max(1, labels.length);
  const max = state.compare ? Math.max(.01, ...labels.flatMap(r => [Math.abs(r.delta_precision || 0), Math.abs(r.delta_recall || 0)])) : 1;
  labels.forEach((r, i) => {
    const y = inner.y + i * rowH;
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "800 10px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(r.label.slice(0, 12), inner.x - 8, y + rowH * .62);
    ctx.textAlign = "left";
    if (state.compare) {
      const mid = inner.x + inner.w / 2;
      ctx.fillStyle = "rgba(148,163,184,.14)";
      ctx.fillRect(inner.x, y + 3, inner.w, rowH - 7);
      ctx.fillStyle = "rgba(226,232,240,.42)";
      ctx.fillRect(mid, y + 2, 1, rowH - 5);
      [["delta_precision", "#38bdf8", -3], ["delta_recall", "#34d399", 4]].forEach(([key, color, dy]) => {
        const v = Number(r[key]) || 0;
        const w = Math.abs(v) / max * inner.w * .48;
        ctx.fillStyle = v >= 0 ? color : "#fb7185";
        ctx.fillRect(v >= 0 ? mid : mid - w, y + rowH / 2 + dy, w, 4);
      });
    } else {
      const precision = Number(r.precision) || 0;
      const recall = Number(r.recall) || 0;
      ctx.fillStyle = "rgba(15,23,42,.76)";
      ctx.fillRect(inner.x, y + 3, inner.w, rowH - 7);
      ctx.fillStyle = "#38bdf8";
      ctx.fillRect(inner.x, y + 4, inner.w * precision, Math.max(4, rowH * .28));
      ctx.fillStyle = "#34d399";
      ctx.fillRect(inner.x, y + rowH * .52, inner.w * recall, Math.max(4, rowH * .28));
      if (inner.w > 260) {
        ctx.fillStyle = "#eaf2ff"; ctx.font = "800 9px Inter, sans-serif";
        ctx.fillText(`${Math.round(precision * 100)}% / ${Math.round(recall * 100)}%`, inner.x + inner.w + 6, y + rowH * .62);
      }
    }
    addStatsRectHit(inner.x, y, inner.w, rowH, {
      title: `Rates · ${r.label}`,
      lines: state.compare
        ? [`Δprecision ${statsValueText(r.delta_precision, "pp")} · Δrecall ${statsValueText(r.delta_recall, "pp")}`, "Click to inspect label rows."]
        : [`Precision ${rate(r.precision)} · Recall ${rate(r.recall)}`, `TP ${fmt(r.tp)} · FP ${fmt(r.fp)} · FN ${fmt(r.fn)}`, "Click to inspect label rows."],
      detailTitle: `Label rates · ${r.label}`,
      rows: statsRowsForLabel(r.label)
    });
  });
  ctx.fillStyle = "#38bdf8"; ctx.fillRect(inner.x, plot.y + 4, 8, 8);
  ctx.fillStyle = "#cbd5e1"; ctx.font = "700 10px Inter, sans-serif"; ctx.fillText(state.compare ? "precision delta" : "precision", inner.x + 12, plot.y + 12);
  ctx.fillStyle = "#34d399"; ctx.fillRect(inner.x + 112, plot.y + 4, 8, 8);
  ctx.fillStyle = "#cbd5e1"; ctx.fillText(state.compare ? "recall delta" : "recall", inner.x + 124, plot.y + 12);
  ctx.textAlign = "left";
}
function drawLabelIssueBars(rect, stats) {
  const plot = chartPanel(rect, state.compare ? "FP / FN Label Drivers Delta" : "FP / FN Label Drivers");
  const rows = (stats?.labels || []).filter(r => r.label).slice().sort((a, b) => {
    const bm = Math.abs(Number(state.compare ? b.delta_fp : b.fp) || 0) + Math.abs(Number(state.compare ? b.delta_fn : b.fn) || 0);
    const am = Math.abs(Number(state.compare ? a.delta_fp : a.fp) || 0) + Math.abs(Number(state.compare ? a.delta_fn : a.fn) || 0);
    return bm - am;
  }).slice(0, 12);
  if (!rows.length) { ctx.fillStyle = "#91a4bf"; ctx.fillText("No label issue data.", plot.x, plot.y + 28); return; }
  const inner = {x: plot.x + 72, y: plot.y + 8, w: plot.w - 92, h: plot.h - 38};
  const max = Math.max(1, ...rows.flatMap(r => [Math.abs(Number(state.compare ? r.delta_fp : r.fp) || 0), Math.abs(Number(state.compare ? r.delta_fn : r.fn) || 0)]));
  const rowH = inner.h / rows.length;
  rows.forEach((r, i) => {
    const y = inner.y + i * rowH + 2;
    ctx.fillStyle = "#cbd5e1"; ctx.font = "800 10px Inter, sans-serif"; ctx.textAlign = "right";
    ctx.fillText(r.label.slice(0, 12), inner.x - 8, y + rowH * .55);
    ctx.textAlign = "left";
    const fp = Number(state.compare ? r.delta_fp : r.fp) || 0;
    const fn = Number(state.compare ? r.delta_fn : r.fn) || 0;
    const mid = state.compare ? inner.x + inner.w / 2 : inner.x;
    ctx.fillStyle = "rgba(15,23,42,.76)";
    ctx.fillRect(inner.x, y, inner.w, rowH - 5);
    if (state.compare) {
      ctx.fillStyle = "rgba(226,232,240,.42)";
      ctx.fillRect(mid, y, 1, rowH - 5);
    }
    [[fp, "#fb7185", 2], [fn, "#fbbf24", Math.max(7, rowH / 2)]].forEach(([v, color, dy]) => {
      const w = Math.abs(v) / max * (state.compare ? inner.w * .48 : inner.w);
      ctx.fillStyle = state.compare && v < 0 ? "#34d399" : color;
      ctx.fillRect(state.compare ? (v >= 0 ? mid : mid - w) : inner.x, y + dy, w, Math.max(3, rowH * .22));
    });
    ctx.fillStyle = "#91a4bf"; ctx.font = "700 9px Inter, sans-serif";
    ctx.fillText(`FP ${state.compare ? fmtDelta(fp) : fmt(fp)}  FN ${state.compare ? fmtDelta(fn) : fmt(fn)}`, inner.x + inner.w + 6, y + rowH * .55);
    addStatsRectHit(inner.x, y, inner.w, rowH, {
      title: `Issues · ${r.label}`,
      lines: [
        `FP ${state.compare ? fmtDelta(fp) : fmt(fp)} · FN ${state.compare ? fmtDelta(fn) : fmt(fn)}`,
        state.compare ? `ΔTP ${fmtDelta(r.delta_tp || 0)}` : `TP ${fmt(r.tp)}`,
        "Click to inspect label rows."
      ],
      detailTitle: `Label issues · ${r.label}`,
      rows: statsRowsForLabel(r.label)
    });
  });
  ctx.textAlign = "left";
}
function drawScenarioDistribution(rect, arr) {
  const plot = chartPanel(rect, state.compare ? "Scenario Change Distribution" : "Scenario Issue Distribution");
  const values = arr.map(s => Math.max(0, scenarioMetric(s))).filter(Number.isFinite);
  if (!values.length) { ctx.fillStyle = "#91a4bf"; ctx.fillText("No scenario distribution.", plot.x, plot.y + 28); return; }
  const max = niceMax(Math.max(...values));
  const buckets = new Array(12).fill(0);
  values.forEach(v => buckets[Math.min(buckets.length - 1, Math.floor(v / max * buckets.length))] += 1);
  const inner = {x: plot.x + 50, y: plot.y + 8, w: plot.w - 66, h: plot.h - 48};
  const yMax = niceMax(Math.max(...buckets));
  drawPlotFrame(inner, state.lens.toUpperCase(), "Scenario count", yMax);
  const barW = inner.w / buckets.length * .72;
  buckets.forEach((n, i) => {
    const x = inner.x + (i + .5) * inner.w / buckets.length;
    const h = n / yMax * inner.h;
    ctx.fillStyle = colorFor(i / Math.max(1, buckets.length - 1) * max, max);
    ctx.fillRect(x - barW / 2, inner.y + inner.h - h, barW, h);
    addStatsRectHit(x - barW / 2, inner.y + inner.h - h, barW, Math.max(4, h), {
      title: `Scenario distribution · bucket ${i + 1}`,
      lines: [
        `${fmt(n)} scenarios`,
        `Range ${fmt(Math.round(i * max / buckets.length))} to ${fmt(Math.round((i + 1) * max / buckets.length))}`,
        "Click to inspect matching scenarios."
      ],
      detailTitle: `Scenario distribution · bucket ${i + 1}`,
      rows: arr
        .filter(s => {
          const v = Math.max(0, scenarioMetric(s));
          const bi = Math.min(buckets.length - 1, Math.floor(v / max * buckets.length));
          return bi === i;
        })
        .map(s => statsRowsForScenario(s)[0])
    });
    ctx.fillStyle = "#91a4bf"; ctx.font = "700 9px Inter, sans-serif"; ctx.textAlign = "center";
    if (i % 2 === 0) ctx.fillText(fmt(Math.round(i * max / buckets.length)), x, inner.y + inner.h + 12);
  });
  ctx.textAlign = "left";
}
function drawScenarioRankingWide(rect, arr) {
  const plot = chartPanel(rect, state.compare ? "Scenario Drivers: Largest Changes" : "Scenario Drivers: Most Issues");
  const top = arr.slice(0, 18);
  if (!top.length) { ctx.fillStyle = "#91a4bf"; ctx.fillText("No scenarios.", plot.x, plot.y + 28); return; }
  const max = Math.max(1, ...top.map(s => Math.abs(scenarioMetric(s))));
  const rowH = plot.h / top.length;
  top.forEach((s, i) => {
    const y = plot.y + i * rowH + 2;
    const v = scenarioMetric(s);
    ctx.fillStyle = "#cbd5e1"; ctx.font = "800 10px Inter, sans-serif";
    ctx.fillText(scenarioName(s).slice(0, 44), plot.x, y + rowH * .58);
    const bx = plot.x + Math.min(310, plot.w * .46);
    const bw = plot.w - (bx - plot.x) - 74;
    ctx.fillStyle = "rgba(15,23,42,.78)";
    ctx.fillRect(bx, y, bw, Math.max(6, rowH - 5));
    ctx.fillStyle = colorFor(v, max);
    ctx.fillRect(bx, y, bw * Math.abs(v) / max, Math.max(6, rowH - 5));
    ctx.fillStyle = "#91a4bf"; ctx.font = "700 9px Inter, sans-serif";
    ctx.fillText(state.compare ? fmtDelta(Math.round(v)) : fmt(Math.round(v)), bx + bw + 8, y + rowH * .58);
    addStatsRectHit(plot.x, y, plot.w, rowH, {
      kind: "scenario",
      s,
      title: scenarioName(s),
      lines: [
        `${compareLensLabel()} ${lensMetricText(v)}`,
        state.compare ? `ΔFP ${fmtDelta(s.delta_fp)} · ΔFN ${fmtDelta(s.delta_fn)}` : `FP ${fmt(s.fp)} · FN ${fmt(s.fn)} · TP ${fmt(s.tp)}`,
        "Click to load this scenario and show detail rows."
      ],
      detailTitle: `Scenario · ${scenarioName(s)}`,
      rows: statsRowsForScenario(s)
    });
  });
}
function drawFrameFocusRanking(rect) {
  const label = {
    degraded: "Frames: Degraded First",
    improved: "Frames: Improved First",
    abs: "Frames: Largest Net Change"
  }[state.statsFrameFocus] || "Frames";
  const plot = chartPanel(rect, label);
  const rows = focusedFrameRows(22);
  if (!rows.length) {
    ctx.fillStyle = "#91a4bf";
    ctx.fillText("No frame-level stats available.", plot.x, plot.y + 28);
    return;
  }
  const valueFor = row => {
    if (state.compare) {
      if (state.statsFrameFocus === "improved") return Math.max(0, -Number(row.delta_fp || 0)) + Math.max(0, -Number(row.delta_fn || 0)) + Math.max(0, Number(row.delta_tp || 0));
      if (state.statsFrameFocus === "abs") return Math.abs(Number(row.delta_fp || 0)) + Math.abs(Number(row.delta_fn || 0)) + Math.abs(Number(row.delta_tp || 0));
      return Math.max(0, Number(row.delta_fp || 0)) + Math.max(0, Number(row.delta_fn || 0)) + Math.max(0, -Number(row.delta_tp || 0));
    }
    if (state.statsFrameFocus === "improved") return Number(row.tp || 0);
    if (state.statsFrameFocus === "abs") return Number(row.tp || 0) + Number(row.fp || 0) + Number(row.fn || 0);
    return Number(row.fp || 0) + Number(row.fn || 0);
  };
  const max = Math.max(1, ...rows.map(valueFor));
  const rowH = plot.h / rows.length;
  rows.forEach((row, i) => {
    const y = plot.y + i * rowH + 2;
    const v = valueFor(row);
    const name = `${scenarioName(row)} · f${row.frame ?? "-"}`;
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "800 10px Inter, sans-serif";
    ctx.fillText(name.slice(0, 46), plot.x, y + rowH * .58);
    const bx = plot.x + Math.min(330, plot.w * .48);
    const bw = plot.w - (bx - plot.x) - 92;
    ctx.fillStyle = "rgba(15,23,42,.78)";
    ctx.fillRect(bx, y, bw, Math.max(6, rowH - 5));
    ctx.fillStyle = state.compare && state.statsFrameFocus === "improved" ? "#34d399" : colorFor(v, max);
    ctx.fillRect(bx, y, bw * v / max, Math.max(6, rowH - 5));
    ctx.fillStyle = "#91a4bf";
    ctx.font = "700 9px Inter, sans-serif";
    ctx.fillText(state.compare
      ? `ΔFP ${fmtDelta(row.delta_fp || 0)} · ΔFN ${fmtDelta(row.delta_fn || 0)}`
      : `FP ${fmt(row.fp)} · FN ${fmt(row.fn)}`,
      bx + bw + 8,
      y + rowH * .58);
    addStatsRectHit(plot.x, y, plot.w, rowH, {
      title: name,
      lines: [
        state.compare ? `ΔTP ${fmtDelta(row.delta_tp || 0)} · ΔFP ${fmtDelta(row.delta_fp || 0)} · ΔFN ${fmtDelta(row.delta_fn || 0)}` : `TP ${fmt(row.tp)} · FP ${fmt(row.fp)} · FN ${fmt(row.fn)}`,
        `${row.t4dataset_name || row.t4dataset_id || ""}`,
        "Click to inspect this frame row."
      ],
      detailTitle: `Frame · ${name}`,
      rows: [{section: "frame", ...row}]
    });
  });
}
function statsInsightLines(stats) {
  if (!stats || stats.error) return [stats?.error || "Stats are still loading."];
  const dist = stats.distance || [];
  const labels = stats.labels || [];
  if (!dist.length) return ["No distance statistics available for the active filters."];
  const byTpr = [...dist].filter(r => r.tpr != null).sort((a, b) => (Number(a.tpr) || 0) - (Number(b.tpr) || 0));
  const byFpr = [...dist].filter(r => r.fpr != null).sort((a, b) => (Number(b.fpr) || 0) - (Number(a.fpr) || 0));
  const fpLabel = [...labels].sort((a, b) => (Number(state.compare ? b.delta_fp : b.fp) || 0) - (Number(state.compare ? a.delta_fp : a.fp) || 0))[0];
  const fnLabel = [...labels].sort((a, b) => (Number(state.compare ? b.delta_fn : b.fn) || 0) - (Number(state.compare ? a.delta_fn : a.fn) || 0))[0];
  if (state.compare) {
    const tprLoss = [...dist].filter(r => r.delta_tpr != null).sort((a, b) => (Number(a.delta_tpr) || 0) - (Number(b.delta_tpr) || 0))[0];
    const fprRise = [...dist].filter(r => r.delta_fpr != null).sort((a, b) => (Number(b.delta_fpr) || 0) - (Number(a.delta_fpr) || 0))[0];
    return [
      `Largest recall drop: ${tprLoss?.bin_label || "-"} (${Math.round((Number(tprLoss?.delta_tpr) || 0) * 100)}pp)`,
      `Largest FP-rate increase: ${fprRise?.bin_label || "-"} (${Math.round((Number(fprRise?.delta_fpr) || 0) * 100)}pp)`,
      `Label drivers: FP ${fpLabel?.label || "-"} ${fmtDelta(fpLabel?.delta_fp || 0)} · FN ${fnLabel?.label || "-"} ${fmtDelta(fnLabel?.delta_fn || 0)}`,
    ];
  }
  return [
    `Weakest recall range: ${byTpr[0]?.bin_label || "-"} (${rate(byTpr[0]?.tpr)})`,
    `Highest FP-rate range: ${byFpr[0]?.bin_label || "-"} (${rate(byFpr[0]?.fpr)})`,
    `Label drivers: FP ${fpLabel?.label || "-"} ${fmt(fpLabel?.fp)} · FN ${fnLabel?.label || "-"} ${fmt(fnLabel?.fn)}`,
  ];
}
function renderStatsDashboard(rect) {
  const arr = filteredScenarios();
  state.labelNodes = [];
  state.statNodes = [];
  ctx.fillStyle = "#eaf2ff";
  ctx.font = "900 18px Inter, sans-serif";
  ctx.fillText("Dataset Statistics", 18, 86);
  ctx.fillStyle = "#91a4bf";
  ctx.font = "700 11px Inter, sans-serif";
  ctx.fillText(`${fmt(arr.length)} scenarios · ${state.compare ? "Run B - Run A" : compareLensLabel()} · ${state.label || "all labels"} · ${state.rangeMax ? `<${state.rangeMax}m` : "all ranges"}`, 18, 105);
  statsInsightLines(state.stats).forEach((line, i) => {
    ctx.fillStyle = i === 0 ? "#fbbf24" : "#cbd5e1";
    ctx.font = "800 11px Inter, sans-serif";
    ctx.fillText(line, 18 + i * Math.max(260, rect.width / 3.15), 128);
  });
  const gap = 12;
  const top = 158;
  const colW = (rect.width - 44) / 2;
  const rowH = 250;
  const fullW = rect.width - 32;
  const y = i => top + i * (rowH + gap);
  drawDatasetKpis({x: 16, y: y(0), w: colW, h: rowH}, state.stats, arr);
  drawScenarioScatter({x: 28 + colW, y: y(0), w: colW, h: rowH}, arr);
  drawDistanceRates({x: 16, y: y(1), w: colW, h: rowH}, state.stats);
  drawObjectCountByDistance({x: 28 + colW, y: y(1), w: colW, h: rowH}, state.stats);
  drawLabelDistanceHeatmap({x: 16, y: y(2), w: colW, h: rowH}, state.stats, "tpr");
  drawLabelDistanceHeatmap({x: 28 + colW, y: y(2), w: colW, h: rowH}, state.stats, "fpr");
  drawLabelRateBars({x: 16, y: y(3), w: colW, h: rowH}, state.stats);
  drawLabelIssueBars({x: 28 + colW, y: y(3), w: colW, h: rowH}, state.stats);
  drawErrorBars({x: 16, y: y(4), w: colW, h: rowH}, state.stats);
  drawScenarioDistribution({x: 28 + colW, y: y(4), w: colW, h: rowH}, arr);
  drawScenarioRankingWide({x: 16, y: y(5), w: fullW, h: rowH + 70}, arr);
  drawFrameFocusRanking({x: 16, y: y(6) + 70, w: fullW, h: rowH + 70});
  updateStatsHover(rect);
  renderStatsDetail();
}
function updateStatsHover(rect) {
  let best = null;
  for (const hit of state.statNodes || []) {
    if (statsHitContains(hit, state.mouseX, state.mouseY)) {
      best = hit;
      break;
    }
  }
  state.statsHover = best;
  state.hover = best && best.kind === "scenario" ? best.s : null;
  state.hoverLabel = null;
  if (!best) {
    els.hoverCard.classList.remove("show");
    return;
  }
  showStatsHover(best, rect);
}
