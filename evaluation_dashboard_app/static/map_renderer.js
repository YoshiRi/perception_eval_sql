function layoutNodes() {
  const arr = state.scenarios;
  const cellW = 46;
  const cellH = 32;
  const gap = 8;
  if (state.layout === "grid") {
    const cols = Math.ceil(Math.sqrt(Math.max(1, arr.length)));
    arr.forEach((s, i) => {
      s._cluster = cityName(s);
      s._w = cellW;
      s._h = cellH;
      s._x = (i % cols - cols / 2) * (cellW + gap);
      s._y = (Math.floor(i / cols) - Math.ceil(arr.length / cols) / 2) * (cellH + gap);
      s._groupX = null;
    });
    return;
  }

  const groups = new Map();
  arr.forEach(s => {
    const key = clusterKey(s);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(s);
  });
  const keys = [...groups.keys()].sort((a, b) => groups.get(b).length - groups.get(a).length || a.localeCompare(b));
  const clusterPads = [];
  let cursorX = 0;
  keys.forEach((key, gi) => {
    const items = groups.get(key);
    items.sort((a, b) => scenarioMetric(b) - scenarioMetric(a));
    const cols = Math.ceil(Math.sqrt(items.length));
    const rows = Math.ceil(items.length / cols);
    const width = cols * (cellW + gap) - gap + 34;
    const height = rows * (cellH + gap) - gap + 44;
    clusterPads.push({key, items, cols, rows, width, height});
  });
  const colsOfClusters = Math.ceil(Math.sqrt(Math.max(1, clusterPads.length)));
  const rowHeights = [];
  for (let i = 0; i < clusterPads.length; i += colsOfClusters) {
    rowHeights.push(Math.max(...clusterPads.slice(i, i + colsOfClusters).map(c => c.height)));
  }
  let startY = -rowHeights.reduce((n, h) => n + h + 46, -46) / 2;
  clusterPads.forEach((c, gi) => {
    const col = gi % colsOfClusters;
    const row = Math.floor(gi / colsOfClusters);
    if (col === 0) cursorX = -clusterPads.slice(gi, gi + colsOfClusters).reduce((n, p) => n + p.width + 46, -46) / 2;
    const cx = cursorX + c.width / 2;
    const cy = startY + rowHeights[row] / 2;
    cursorX += c.width + 46;
    c.items.forEach((s, i) => {
      const colI = i % c.cols;
      const rowI = Math.floor(i / c.cols);
      s._cluster = c.key;
      s._groupX = cx;
      s._groupY = cy;
      s._clusterW = c.width;
      s._clusterH = c.height;
      s._clusterSize = c.items.length;
      s._w = cellW;
      s._h = cellH;
      s._x = cx - c.width / 2 + 17 + colI * (cellW + gap);
      s._y = cy - c.height / 2 + 28 + rowI * (cellH + gap);
    });
    if (col === colsOfClusters - 1) startY += rowHeights[row] + 46;
  });
}

function resizeCanvas(c, context) {
  const dpr = window.devicePixelRatio || 1;
  const r = c.getBoundingClientRect();
  c.width = Math.max(1, Math.floor(r.width * dpr));
  c.height = Math.max(1, Math.floor(r.height * dpr));
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return r;
}
function colorFor(v, max) {
  const raw = Number(v || 0);
  const t = max > 0 ? Math.max(0, Math.min(1, Math.abs(raw) / max)) : 0;
  if (raw < 0) return TH.a("good", .34 + t * .58);
  if (t < .35) return TH.a("accent", .45 + t);
  if (t < .68) return TH.a("warn", .52 + t * .55);
  return TH.a("bad", .58 + t * .42);
}
function screenRect(s, rect) {
  const scale = state.scale;
  return {
    x: rect.width / 2 + ((s._x || 0) + state.panX) * scale,
    y: rect.height / 2 + ((s._y || 0) + state.panY) * scale,
    w: (s._w || 46) * scale,
    h: (s._h || 32) * scale,
  };
}
function drawBoardGrid(ctx, rect) {
  const step = 48 * state.scale;
  if (step < 10) return;
  // Grid weight is a theme token: it stays nearly invisible on the dark and paper
  // themes, and reads as a drafting sheet on blueprint, where every 4th line
  // becomes a major rule.
  const minor = TH.num("gridAlpha", .08);
  const major = TH.num("gridMajorAlpha", 0);
  ctx.lineWidth = 1;
  const originX = rect.width / 2 + state.panX * state.scale;
  const originY = rect.height / 2 + state.panY * state.scale;
  const isMajor = (v, origin) => major > 0 && Math.round((v - origin) / step) % 4 === 0;
  for (let x = originX % step; x < rect.width; x += step) {
    ctx.strokeStyle = TH.a("line", isMajor(x, originX) ? major : minor);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, rect.height); ctx.stroke();
  }
  for (let y = originY % step; y < rect.height; y += step) {
    ctx.strokeStyle = TH.a("line", isMajor(y, originY) ? major : minor);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(rect.width, y); ctx.stroke();
  }
}
function drawClusterBlocks(ctx, rect, arr) {
  const clusters = new Map();
  arr.forEach(s => {
    if (!s._cluster || s._groupX == null) return;
    const hit = clusters.get(s._cluster) || {x: s._groupX || 0, y: s._groupY || 0, w: s._clusterW || 80, h: s._clusterH || 80, n: 0, max: 0};
    hit.n += 1; hit.max = Math.max(hit.max, scenarioMetric(s));
    clusters.set(s._cluster, hit);
  });
  for (const [name, c] of clusters) {
    const x = rect.width / 2 + (c.x - c.w / 2 + state.panX) * state.scale;
    const y = rect.height / 2 + (c.y - c.h / 2 + state.panY) * state.scale;
    const w = c.w * state.scale;
    const h = c.h * state.scale;
    if (x > rect.width || y > rect.height || x + w < 0 || y + h < 0) continue;
    ctx.fillStyle = TH.a("accentSoft", .16);
    ctx.strokeStyle = TH.a("accent", .24);
    ctx.lineWidth = 1;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = TH.a("text", .72);
    ctx.font = "800 11px Inter, sans-serif";
    ctx.fillText(`${name} · ${c.n}`, x + 9, y + 16);
  }
}
function drawHotspotGlyph(ctx, s, rect, max) {
  const v = scenarioMetric(s);
  const selected = state.selected && scenarioKey(state.selected) === scenarioKey(s);
  const fp = state.label ? ((s.labels || []).find(x => x.label === state.label)?.fp || 0) : (s.fp || 0);
  const fn = state.label ? ((s.labels || []).find(x => x.label === state.label)?.fn || 0) : (s.fn || 0);
  const tp = state.label ? ((s.labels || []).find(x => x.label === state.label)?.tp || 0) : (s.tp || 0);
  const total = Math.max(1, fp + fn + tp);
  const severity = Math.max(0, Math.min(1, v / Math.max(1, max)));
  const absSeverity = Math.max(0, Math.min(1, Math.abs(v) / Math.max(1, max)));
  const box = screenRect(s, rect);
  if (box.x > rect.width + 20 || box.y > rect.height + 20 || box.x + box.w < -20 || box.y + box.h < -20) return;
  ctx.fillStyle = colorFor(v, max);
  ctx.globalAlpha = .28 + absSeverity * .62;
  ctx.fillRect(box.x, box.y, box.w, box.h);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = selected ? TH.c("marker") : TH.a("lineStrong", .32);
  ctx.lineWidth = selected ? 2 : 1;
  ctx.strokeRect(box.x, box.y, box.w, box.h);

  const stripH = Math.max(4, Math.min(9, box.h * .22));
  const fpW = box.w * (fp / total);
  const fnW = box.w * (fn / total);
  ctx.fillStyle = TH.a("bad", .88);
  ctx.fillRect(box.x, box.y + box.h - stripH, fpW, stripH);
  ctx.fillStyle = TH.a("warn", .84);
  ctx.fillRect(box.x + fpW, box.y + box.h - stripH, fnW, stripH);
  const tpW = box.w * (tp / total);
  ctx.fillStyle = TH.a("accent", .85);
  ctx.fillRect(box.x, box.y, tpW, Math.max(3, stripH * .55));

  if (selected || box.w > 58) {
    ctx.fillStyle = selected ? TH.c("marker") : TH.a("text", .8);
    ctx.font = selected ? "800 11px Inter, sans-serif" : "700 9px Inter, sans-serif";
    ctx.fillText(scenarioName(s).slice(0, selected ? 30 : 16), box.x + 5, box.y + Math.min(17, box.h - 8));
  }
  s._sx = box.x + box.w / 2; s._sy = box.y + box.h / 2; s._sr = Math.max(box.w, box.h) / 2;
}
function labelMetricValue(labelRow) {
  if (state.compare) {
    if (state.lens === "changed_only") return Math.abs(labelRow.delta_fp || 0) + Math.abs(labelRow.delta_fn || 0) + Math.abs(labelRow.delta_tp || 0);
    if (state.lens === "delta_fn") return labelRow.delta_fn || 0;
    if (state.lens === "regression") return Math.max(0, labelRow.delta_fp || 0) + Math.max(0, labelRow.delta_fn || 0);
    return labelRow.delta_fp || 0;
  }
  return metricFromRow(labelRow, state.lens);
}
function scenarioLabelMetric(s, label = state.label) {
  if (!label) return scenarioMetric(s);
  return metricFromRow(labelRow(s, label), state.lens);
}
function mapPoint(x, y, rect) {
  return {
    x: rect.width / 2 + (x + state.panX) * state.scale,
    y: rect.height / 2 + (y + state.panY) * state.scale
  };
}
function scenarioBounds(arr) {
  const boxes = arr.map(s => ({x: s._x || 0, y: s._y || 0, w: s._w || 46, h: s._h || 32}));
  if (!boxes.length) return {minX: -120, maxX: 120, minY: -80, maxY: 80};
  return boxes.reduce((b, r) => ({
    minX: Math.min(b.minX, r.x),
    maxX: Math.max(b.maxX, r.x + r.w),
    minY: Math.min(b.minY, r.y),
    maxY: Math.max(b.maxY, r.y + r.h),
  }), {minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity});
}
function drawLabelBubbles(rect, arr) {
  const labels = allLabelNames().map(label => ({label, row: state.labels.find(v => v.label === label) || zeroLabel(label)}));
  const max = Math.max(1, ...labels.map(v => Math.abs(labelMetricValue(v.row))));
  const bounds = scenarioBounds(arr);
  const cx = bounds.maxX + 170;
  const cy = (bounds.minY + bounds.maxY) / 2;
  const ring = Math.max(78, Math.min(150, 34 + labels.length * 8));
  const center = mapPoint(cx, cy, rect);
  ctx.strokeStyle = TH.a("accent", .2);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(center.x, center.y, ring * state.scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = TH.a("text", .78);
  ctx.font = "800 12px Inter, sans-serif";
  ctx.fillText("labels", center.x - 18, center.y + 4);
  state.labelNodes = [];
  labels.forEach((item, i) => {
    const a = -Math.PI / 2 + i * Math.PI * 2 / Math.max(1, labels.length);
    const v = labelMetricValue(item.row);
    const total = Math.max(1, (item.row.tp || 0) + (item.row.fp || 0) + (item.row.fn || 0));
    const p = mapPoint(cx + Math.cos(a) * ring, cy + Math.sin(a) * ring, rect);
    const radius = Math.max(9, Math.min(28, 10 + Math.sqrt(Math.abs(v) / max) * 20)) * state.scale;
    state.labelNodes.push({label: item.label, x: p.x, y: p.y, r: radius, value: v});
    ctx.fillStyle = colorFor(v, max);
    ctx.globalAlpha = .34 + Math.min(1, Math.abs(v) / max) * .58;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = item.label === state.label ? TH.c("marker") : TH.a("lineStrong", .28);
    ctx.lineWidth = item.label === state.label ? 2 : 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.stroke();
    const fpArc = (item.row.fp || 0) / total * Math.PI * 2;
    const fnArc = (item.row.fn || 0) / total * Math.PI * 2;
    ctx.lineWidth = Math.max(2, radius * .16);
    ctx.strokeStyle = TH.c("bad");
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius + 3, -Math.PI / 2, -Math.PI / 2 + fpArc);
    ctx.stroke();
    ctx.strokeStyle = TH.c("warn");
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius + 6, -Math.PI / 2 + fpArc, -Math.PI / 2 + fpArc + fnArc);
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = TH.c("text");
    ctx.font = `${item.label === state.label ? "800" : "700"} ${Math.max(9, Math.min(12, radius * .42))}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText(item.label.slice(0, 10), p.x, p.y + radius + 13);
    ctx.fillStyle = TH.c("muted");
    ctx.fillText(state.compare ? fmtDelta(Math.round(v)) : fmt(Math.round(v)), p.x, p.y + radius + 25);
    ctx.textAlign = "left";
  });
}
function drawLabelConnections(rect, arr) {
  if (!state.label) return;
  const source = state.labelNodes.find(n => n.label === state.label);
  if (!source) return;
  const scored = arr
    .map(s => ({s, v: scenarioLabelMetric(s, state.label)}))
    .filter(x => Number.isFinite(x.v) && Math.abs(x.v) > 0)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    .slice(0, 36);
  if (!scored.length) return;
  const max = Math.max(1, ...scored.map(x => Math.abs(x.v)));
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  scored.forEach(({s, v}, i) => {
    const box = screenRect(s, rect);
    const tx = box.x + box.w / 2;
    const ty = box.y + box.h / 2;
    if (tx < -80 || ty < -80 || tx > rect.width + 80 || ty > rect.height + 80) return;
    const t = Math.abs(v) / max;
    const midX = (source.x + tx) / 2;
    const midY = (source.y + ty) / 2 - Math.min(80, 18 + t * 54);
    ctx.strokeStyle = v < 0 ? TH.a("good", .16 + t * .58) : TH.a("accent", .14 + t * .5);
    ctx.lineWidth = Math.max(1, 1 + t * 4);
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.quadraticCurveTo(midX, midY, tx, ty);
    ctx.stroke();
    if (i < 10) {
      ctx.fillStyle = v < 0 ? TH.a("good", .9) : TH.a("text", .82);
      ctx.beginPath();
      ctx.arc(tx, ty, Math.max(2.5, 2 + t * 4), 0, Math.PI * 2);
      ctx.fill();
    }
  });
  ctx.restore();
}
function aggregateLabelsFor(arr) {
  const rows = allLabelNames().map(zeroLabel);
  const map = new Map(rows.map(r => [r.label, r]));
  arr.forEach(s => {
    labelsForScenario(s).forEach(x => {
      const r = map.get(x.label) || zeroLabel(x.label);
      r.tp += Number(x.tp) || 0;
      r.fp += Number(x.fp) || 0;
      r.fn += Number(x.fn) || 0;
      r.rows += Number(x.rows) || 0;
      r.delta_tp += Number(x.delta_tp) || 0;
      r.delta_fp += Number(x.delta_fp) || 0;
      r.delta_fn += Number(x.delta_fn) || 0;
      map.set(x.label, r);
    });
  });
  return [...map.values()];
}
function canvasRoundRect(x, y, w, h, radius = 8) {
  const r = Math.max(0, Math.min(radius, w / 2, h / 2));
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x, y, w, h, r);
    return;
  }
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
}
function fillCard(x, y, w, h, options = {}) {
  canvasRoundRect(x, y, w, h, options.radius || 8);
  ctx.fillStyle = options.fill || TH.a("panel", .72);
  ctx.fill();
  ctx.strokeStyle = options.stroke || TH.a("line", .18);
  ctx.lineWidth = options.lineWidth || 1;
  ctx.stroke();
}
function drawWrappedText(text, x, y, maxW, lineH, maxLines = 3) {
  const words = String(text || "").split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  words.forEach(word => {
    const next = line ? `${line} ${word}` : word;
    if (ctx.measureText(next).width <= maxW || !line) line = next;
    else { lines.push(line); line = word; }
  });
  if (line) lines.push(line);
  lines.slice(0, maxLines).forEach((l, i) => {
    const out = i === maxLines - 1 && lines.length > maxLines ? `${l.replace(/\s+\S+$/, "")}...` : l;
    ctx.fillText(out, x, y + i * lineH);
  });
  return Math.min(lines.length, maxLines) * lineH;
}
function drawCanvasBadge(text, status, x, y, w = 58, h = 21) {
  const colors = {
    pass: [TH.a("good", .18), TH.a("good", .72), TH.c("goodFg")],
    fail: [TH.a("bad", .18), TH.a("bad", .72), TH.c("badFg")],
    review: [TH.a("warn", .16), TH.a("warn", .66), TH.c("warnFg")],
    unknown: [TH.a("line", .14), TH.a("line", .46), TH.c("mutedBright")]
  }[status] || [TH.a("line", .14), TH.a("line", .46), TH.c("mutedBright")];
  fillCard(x, y, w, h, {fill: colors[0], stroke: colors[1], radius: 7});
  ctx.fillStyle = colors[2];
  ctx.font = "800 10px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, x + w / 2, y + 14);
  ctx.textAlign = "left";
}
function selectedOrFirstDevops(arr) {
  if (state.selected && devopsContext(state.selected).is_devops) {
    const key = scenarioKey(state.selected);
    return arr.find(s => scenarioKey(s) === key) || state.selected;
  }
  return arr[0] || null;
}
function devopsResultForCanvas(s) {
  if (state.selected && s && scenarioKey(state.selected) === scenarioKey(s) && state.devopsResult) return state.devopsResult;
  return fallbackScenarioResult(s, "");
}
function drawDevopsSuiteCards(groups, x, y, w, h) {
  ctx.fillStyle = TH.c("text");
  ctx.font = "800 15px Inter, sans-serif";
  ctx.fillText("Suites", x, y);
  ctx.fillStyle = TH.a("muted", .9);
  ctx.font = "600 11px Inter, sans-serif";
  ctx.fillText("Worst pass rate first. Select cases below or from the left list.", x, y + 20);
  const gap = 9;
  const cardH = 60;
  const usableY = y + 36;
  const cols = w > 620 ? 2 : 1;
  const colW = (w - gap * (cols - 1)) / cols;
  const maxCards = Math.max(4, Math.floor(h / (cardH + gap)) * cols);
  groups.slice(0, maxCards).forEach((group, i) => {
    const cx = x + (i % cols) * (colW + gap);
    const cy = usableY + Math.floor(i / cols) * (cardH + gap);
    const suitePass = group.suitePass;
    const rateValue = suitePass ? suitePass.pass_rate : group.pass / Math.max(1, group.items.length);
    const ratePct = Math.max(0, Math.min(100, Number(rateValue || 0) * 100));
    const isFailing = ratePct < 50 || group.fail > group.pass;
    fillCard(cx, cy, colW, cardH, {
      fill: isFailing ? TH.a("badBgDeep", .28) : TH.a("goodBg", .18),
      stroke: isFailing ? TH.a("bad", .30) : TH.a("good", .26)
    });
    ctx.fillStyle = TH.c("text");
    ctx.font = "800 11px Inter, sans-serif";
    drawWrappedText(group.key.replace(/^DevOps_V1_/, ""), cx + 10, cy + 16, colW - 88, 12, 2);
    ctx.fillStyle = isFailing ? TH.c("badFg") : TH.c("goodFg");
    ctx.font = "800 13px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`${Math.round(ratePct)}%`, cx + colW - 10, cy + 18);
    ctx.textAlign = "left";
    ctx.fillStyle = TH.a("surface", .92);
    canvasRoundRect(cx + 10, cy + cardH - 16, colW - 20, 5, 4);
    ctx.fill();
    ctx.fillStyle = isFailing ? TH.c("bad") : TH.c("good");
    canvasRoundRect(cx + 10, cy + cardH - 16, (colW - 20) * ratePct / 100, 5, 4);
    ctx.fill();
    ctx.fillStyle = TH.a("mutedBright", .82);
    ctx.font = "600 10px Inter, sans-serif";
    const counts = suitePass
      ? `${fmt(suitePass.passed)}/${fmt(suitePass.total)} official pass`
      : `${fmt(group.pass)} pass / ${fmt(group.fail)} fail / ${fmt(group.review)} review`;
    ctx.fillText(counts, cx + 10, cy + cardH - 25);
  });
}
function drawDevopsSuiteTree(groups, x, y, w, h, selected) {
  ctx.fillStyle = TH.c("text");
  ctx.font = "800 15px Inter, sans-serif";
  ctx.fillText("Suite Results", x, y);
  ctx.fillStyle = TH.a("muted", .9);
  ctx.font = "600 11px Inter, sans-serif";
  ctx.fillText("Fold suites here, then select a scenario to inspect judgement gates.", x, y + 20);
  const listX = x;
  const listY = y + 36;
  const listW = w;
  const listH = h - 36;
  fillCard(listX, listY, listW, listH, {fill: TH.a("deep", .22), stroke: TH.a("line", .14)});
  ctx.save();
  ctx.beginPath();
  ctx.rect(listX, listY, listW, listH);
  ctx.clip();
  let cursor = listY + 8 - (state.devopsCanvasScroll || 0);
  const selectedKey = selected ? scenarioKey(selected) : "";
  groups.forEach(group => {
    const expanded = state.expandedSuites.has(group.key);
    const headH = 42;
    const visible = cursor + headH >= listY && cursor <= listY + listH;
    if (visible) {
      const suitePass = group.suitePass;
      const rateValue = suitePass ? suitePass.pass_rate : group.pass / Math.max(1, group.items.length);
      const ratePct = Math.max(0, Math.min(100, Number(rateValue || 0) * 100));
      const failing = ratePct < 50 || group.fail > group.pass;
      fillCard(listX + 8, cursor, listW - 16, headH - 4, {
        fill: failing ? TH.a("badBgDeep", .24) : TH.a("goodBg", .16),
        stroke: failing ? TH.a("bad", .28) : TH.a("good", .22)
      });
      state.devopsCanvasHits.push({kind: "suite", suite: group.key, x: listX + 8, y: cursor, w: listW - 16, h: headH - 4});
      ctx.fillStyle = TH.c("text");
      ctx.font = "900 12px Inter, sans-serif";
      ctx.fillText(expanded ? "v" : ">", listX + 20, cursor + 24);
      drawWrappedText(group.key.replace(/^DevOps_V1_/, ""), listX + 42, cursor + 17, listW - 210, 13, 1);
      const countText = suitePass
        ? `${fmt(suitePass.passed)}/${fmt(suitePass.total)} pass`
        : `${fmt(group.pass)} pass / ${fmt(group.fail)} fail`;
      ctx.fillStyle = failing ? TH.c("badFg") : TH.c("goodFg");
      ctx.font = "900 12px Inter, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(`${Math.round(ratePct)}%`, listX + listW - 20, cursor + 17);
      ctx.fillStyle = TH.a("mutedBright", .82);
      ctx.font = "700 10px Inter, sans-serif";
      ctx.fillText(countText, listX + listW - 20, cursor + 31);
      ctx.textAlign = "left";
      ctx.fillStyle = TH.a("surface", .92);
      canvasRoundRect(listX + 42, cursor + 27, Math.max(80, listW - 250), 5, 4);
      ctx.fill();
      ctx.fillStyle = failing ? TH.c("bad") : TH.c("good");
      canvasRoundRect(listX + 42, cursor + 27, Math.max(80, listW - 250) * ratePct / 100, 5, 4);
      ctx.fill();
    }
    cursor += headH;
    if (!expanded) return;
    group.items
      .sort((a, b) => {
        const aj = scenarioJudgement(a), bj = scenarioJudgement(b);
        const order = {fail: 0, review: 1, pass: 2};
        return order[aj.status] - order[bj.status] || scenarioMetric(b) - scenarioMetric(a);
      })
      .forEach(item => {
        const rowH = 43;
        const rowVisible = cursor + rowH >= listY && cursor <= listY + listH;
        if (rowVisible) {
          const active = scenarioKey(item) === selectedKey;
          const j = scenarioJudgement(item);
          fillCard(listX + 24, cursor, listW - 40, rowH - 5, {
            fill: active ? TH.a("accentSoft", .74) : TH.a("surface", .54),
            stroke: active ? TH.a("accent", .62) : TH.a("line", .14),
            radius: 7
          });
          state.devopsCanvasHits.push({kind: "scenario", s: item, x: listX + 24, y: cursor, w: listW - 40, h: rowH - 5});
          drawCanvasBadge(j.label, j.status, listX + 34, cursor + 8, 50, 20);
          ctx.fillStyle = TH.c("text");
          ctx.font = "800 11px Inter, sans-serif";
          drawWrappedText(scenarioName(item), listX + 94, cursor + 16, listW - 176, 12, 1);
          ctx.fillStyle = TH.a("muted", .9);
          ctx.font = "600 10px Inter, sans-serif";
          drawWrappedText(`${j.reason} / TP ${fmt(targetMetric(item, "tp"))} / FP ${fmt(targetMetric(item, "fp"))} / FN ${fmt(targetMetric(item, "fn"))}`, listX + 94, cursor + 31, listW - 176, 12, 1);
        }
        cursor += rowH;
      });
  });
  ctx.restore();
  const totalH = cursor - (listY + 8 - (state.devopsCanvasScroll || 0)) + 16;
  state.devopsCanvasMaxScroll = Math.max(0, totalH - listH);
  state.devopsCanvasScroll = Math.max(0, Math.min(state.devopsCanvasScroll || 0, state.devopsCanvasMaxScroll));
  if (state.devopsCanvasMaxScroll > 0) {
    const thumbH = Math.max(36, listH * listH / (listH + state.devopsCanvasMaxScroll));
    const thumbY = listY + (listH - thumbH) * (state.devopsCanvasScroll / state.devopsCanvasMaxScroll);
    fillCard(listX + listW - 8, thumbY, 4, thumbH, {fill: TH.a("accent", .5), stroke: TH.a("accent", .1), radius: 3});
  }
}
function devopsGateLabel(g) {
  const method = g.metric_label || g.method || "criterion";
  const dist = g.distance_label || "all distances";
  return `${method} / ${dist}`;
}
function drawDevopsGate(g, x, y, w) {
  const passed = g.passed === true;
  const failed = g.passed === false;
  const status = passed ? "pass" : (failed ? "fail" : "review");
  drawCanvasBadge(passed ? "PASS" : (failed ? "FAIL" : "CHECK"), status, x, y - 3, 58, 21);
  ctx.fillStyle = TH.c("accentFg");
  ctx.font = "800 11px Inter, sans-serif";
  drawWrappedText(devopsGateLabel(g), x + 68, y + 11, w - 74, 12, 1);
  const barY = y + 28;
  ctx.fillStyle = TH.a("surface", .92);
  canvasRoundRect(x, barY, w, 7, 4);
  ctx.fill();
  const actual = Number(g.actual_rate);
  const required = Number(g.required_rate);
  const actualPct = Number.isFinite(actual) ? Math.max(0, Math.min(1, actual)) : 0;
  const requiredPct = Number.isFinite(required) ? Math.max(0, Math.min(1, required)) : null;
  ctx.fillStyle = failed ? TH.c("bad") : TH.c("accent");
  canvasRoundRect(x, barY, w * actualPct, 7, 4);
  ctx.fill();
  if (requiredPct != null) {
    ctx.strokeStyle = TH.c("warn");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x + w * requiredPct, barY - 4);
    ctx.lineTo(x + w * requiredPct, barY + 11);
    ctx.stroke();
  }
  ctx.fillStyle = TH.a("muted", .92);
  ctx.font = "600 10px Inter, sans-serif";
  const meta = `${pctText(g.actual_rate)} actual / ${pctText(g.required_rate)} required / ${fmt(g.fail_count)} failed / ${fmt(g.total_count)} total`;
  drawWrappedText(meta, x, y + 49, w, 12, 1);
}
function drawDevopsEvidenceCurve(s, result, x, y, w, h) {
  fillCard(x, y, w, h, {fill: TH.a("deep", .34), stroke: TH.a("line", .16)});
  ctx.fillStyle = TH.c("text");
  ctx.font = "800 12px Inter, sans-serif";
  ctx.fillText("Frame Evidence", x + 12, y + 18);
  const frames = (state.selected && s && scenarioKey(state.selected) === scenarioKey(s) ? state.curve : []) || [];
  const hot = result && (result.hot_frames || []);
  const plotX = x + 12, plotY = y + 34, plotW = w - 24, plotH = h - 48;
  if (!frames.length) {
    ctx.fillStyle = TH.a("muted", .85);
    ctx.font = "600 11px Inter, sans-serif";
    ctx.fillText("Select a scenario to load its per-frame TP / FP / FN curve.", plotX, plotY + 25);
    return;
  }
  const sampled = frames.length > 120 ? frames.filter((_, i) => i % Math.ceil(frames.length / 120) === 0) : frames;
  const maxV = Math.max(1, ...sampled.map(f => Math.abs(f.fp || 0) + Math.abs(f.fn || 0) + Math.abs(f.tp || 0)));
  sampled.forEach((f, i) => {
    const bx = plotX + i / Math.max(1, sampled.length - 1) * plotW;
    const fpH = Math.abs(f.fp || 0) / maxV * plotH;
    const fnH = Math.abs(f.fn || 0) / maxV * plotH;
    const tpH = Math.abs(f.tp || 0) / maxV * plotH;
    ctx.fillStyle = TH.a("bad", .7);
    ctx.fillRect(bx, plotY + plotH - fpH, Math.max(1, plotW / sampled.length - 1), fpH);
    ctx.fillStyle = TH.a("warn", .72);
    ctx.fillRect(bx, plotY + plotH - fpH - fnH, Math.max(1, plotW / sampled.length - 1), fnH);
    ctx.fillStyle = TH.a("accent", .66);
    ctx.fillRect(bx, plotY + plotH - fpH - fnH - tpH, Math.max(1, plotW / sampled.length - 1), tpH);
  });
  (hot || []).slice(0, 5).forEach((f, i) => {
    const fx = plotX + (i + .5) * Math.min(92, plotW / Math.max(1, Math.min(5, hot.length)));
    const fy = y + h - 25;
    fillCard(fx, fy, 78, 18, {fill: TH.a("surface", .86), stroke: TH.a("accent", .25), radius: 6});
    ctx.fillStyle = TH.c("accentFg");
    ctx.font = "800 10px Inter, sans-serif";
    ctx.fillText(`f${f.frame}`, fx + 8, fy + 12);
    state.devopsCanvasHits.push({kind: "frame", frame: Number(f.frame), x: fx, y: fy, w: 78, h: 18});
  });
}
function renderDevopsCanvas(rect) {
  state.devopsCanvasHits = [];
  state.hover = null;
  state.hoverLabel = null;
  const all = filteredScenarios().filter(s => devopsContext(s).is_devops);
  const groups = devopsSuiteGroups(all);
  const pad = 24;
  const topY = 92;
  const contentW = rect.width - pad * 2;
  const contentH = rect.height - topY - pad;
  ctx.fillStyle = TH.c("text");
  ctx.font = "900 24px Inter, sans-serif";
  ctx.fillText("DevOps Result Review", pad, 42);
  ctx.fillStyle = TH.a("accentFg", .86);
  ctx.font = "600 12px Inter, sans-serif";
  ctx.fillText("Suite pass/fail, scenario intent, criteria gates, and evidence frames in one place.", pad, 64);
  const totalPass = groups.reduce((n, g) => n + (g.suitePass ? g.suitePass.passed : g.pass), 0);
  const totalCases = groups.reduce((n, g) => n + (g.suitePass ? g.suitePass.total : g.items.length), 0);
  const kpis = [
    ["Suites", fmt(groups.length)],
    ["Scenarios", fmt(all.length)],
    ["Official Pass", totalCases ? `${fmt(totalPass)}/${fmt(totalCases)}` : "-"],
    ["Estimated Fails", fmt(groups.reduce((n, g) => n + g.fail, 0))]
  ];
  kpis.forEach((k, i) => {
    const x = rect.width - pad - (4 - i) * 126;
    fillCard(x, 24, 112, 46, {fill: TH.a("panel", .58), stroke: TH.a("accent", .18)});
    ctx.fillStyle = TH.c("text");
    ctx.font = "900 16px Inter, sans-serif";
    ctx.fillText(k[1], x + 10, 44);
    ctx.fillStyle = TH.a("muted", .9);
    ctx.font = "700 10px Inter, sans-serif";
    ctx.fillText(k[0], x + 10, 60);
  });
  if (!all.length) {
    fillCard(pad, topY, contentW, Math.min(280, contentH), {fill: TH.a("panel", .58), stroke: TH.a("warn", .25)});
    ctx.fillStyle = TH.c("warnFg");
    ctx.font = "900 18px Inter, sans-serif";
    ctx.fillText("No DevOps scenario metadata was inferred.", pad + 18, topY + 38);
    ctx.fillStyle = TH.a("mutedBright", .9);
    ctx.font = "600 12px Inter, sans-serif";
    drawWrappedText("Choose a devops parquet, clear the search, or restart the bbox API if the run metadata was loaded before the new parser.", pad + 18, topY + 64, contentW - 36, 16, 3);
    updateDevopsCanvasHover(rect);
    return;
  }
  const leftW = Math.max(360, Math.min(700, contentW * .47));
  const rightX = pad + leftW + 20;
  const rightW = contentW - leftW - 20;
  const s = selectedOrFirstDevops(all);
  const result = s ? devopsResultForCanvas(s) : null;
  const judgement = s ? (result ? {
    status: result.overall_pass ? "pass" : "fail",
    label: result.overall_pass ? "PASS" : "FAIL",
    reason: (result.explanation || [])[0] || scenarioJudgement(s).reason
  } : scenarioJudgement(s)) : null;
  fillCard(rightX, topY, rightW, contentH, {fill: TH.a("panel", .62), stroke: TH.a("accent", .20)});
  if (!s) {
    ctx.fillStyle = TH.c("text");
    ctx.font = "900 16px Inter, sans-serif";
    ctx.fillText("Select a scenario", rightX + 16, topY + 32);
    return;
  }
  drawDevopsSuiteTree(groups, pad, topY, leftW, contentH, s);
  drawCanvasBadge(judgement.label, judgement.status, rightX + 16, topY + 16, 66, 24);
  ctx.fillStyle = TH.c("text");
  ctx.font = "900 17px Inter, sans-serif";
  drawWrappedText(scenarioName(s), rightX + 94, topY + 34, rightW - 112, 18, 2);
  const c = devopsContext(s);
  ctx.fillStyle = TH.a("accentFg", .86);
  ctx.font = "700 11px Inter, sans-serif";
  drawWrappedText([c.intent_type, c.target_label, c.behavior, c.pc_mode, c.city].filter(Boolean).join(" / "), rightX + 16, topY + 78, rightW - 32, 14, 2);
  ctx.fillStyle = TH.a("lineStrong", .92);
  ctx.font = "600 12px Inter, sans-serif";
  drawWrappedText((result && result.explanation || [judgement.reason, devopsQuickRead(s)]).join(" "), rightX + 16, topY + 116, rightW - 32, 16, 4);
  const target = c.target_label || "target";
  const metricY = topY + 190;
  [
    ["TP", targetMetric(s, "tp"), TH.c("accent")],
    ["FP", targetMetric(s, "fp"), TH.c("bad")],
    ["FN", targetMetric(s, "fn"), TH.c("warn")],
    ["Frames", s.frames || 0, TH.c("mutedBright")]
  ].forEach((m, i) => {
    const mx = rightX + 16 + i * Math.max(92, (rightW - 32) / 4);
    ctx.fillStyle = m[2];
    ctx.font = "900 17px Inter, sans-serif";
    ctx.fillText(fmt(m[1]), mx, metricY);
    ctx.fillStyle = TH.a("muted", .9);
    ctx.font = "700 10px Inter, sans-serif";
    ctx.fillText(`${target} ${m[0]}`.replace(`${target} Frames`, "Frames"), mx, metricY + 16);
  });
  ctx.fillStyle = TH.c("text");
  ctx.font = "900 13px Inter, sans-serif";
  ctx.fillText("Criteria Gates", rightX + 16, metricY + 50);
  const gates = (result && result.gates || []).slice(0, 4);
  if (gates.length) gates.forEach((g, i) => drawDevopsGate(g, rightX + 16, metricY + 70 + i * 72, rightW - 32));
  else {
    ctx.fillStyle = TH.a("muted", .9);
    ctx.font = "600 11px Inter, sans-serif";
    drawWrappedText("No supported criteria gates for this scenario yet. The summary still shows target TP / FP / FN for investigation.", rightX + 16, metricY + 72, rightW - 32, 14, 2);
  }
  const evidenceY = Math.max(metricY + 70 + Math.max(1, gates.length) * 72 + 8, topY + contentH - 142);
  drawDevopsEvidenceCurve(s, result, rightX + 16, evidenceY, rightW - 32, Math.max(118, topY + contentH - evidenceY - 16));
  fillCard(rightX + rightW - 120, topY + contentH - 44, 96, 26, {fill: TH.a("accentDeep", .26), stroke: TH.a("accent", .40), radius: 7});
  ctx.fillStyle = TH.c("btnFg");
  ctx.font = "900 11px Inter, sans-serif";
  ctx.fillText("Open Viewer", rightX + rightW - 103, topY + contentH - 27);
  state.devopsCanvasHits.push({kind: "viewer", x: rightX + rightW - 120, y: topY + contentH - 44, w: 96, h: 26});
  updateDevopsCanvasHover(rect);
}

function render() {
  els.stage.classList.toggle("stats-mode", state.stageView === "stats");
  els.stage.classList.toggle("devops-mode", state.explorerMode === "devops" && state.stageView !== "stats");
  const r = resizeCanvas(els.canvas, ctx);
  ctx.clearRect(0, 0, r.width, r.height);
  const bg = ctx.createLinearGradient(0, 0, r.width, r.height);
  bg.addColorStop(0, TH.c("onAccent")); bg.addColorStop(.55, TH.c("bg")); bg.addColorStop(1, TH.c("bgDeep"));
  ctx.fillStyle = bg; ctx.fillRect(0, 0, r.width, r.height);
  if (state.stageView === "stats") {
    els.legend.style.display = "none";
    renderStatsDashboard(r);
    return;
  }
  if (state.explorerMode === "devops") {
    els.legend.style.display = "none";
    state.devopsCanvasHits = [];
    state.devopsHoverHit = null;
    els.hoverCard.classList.remove("show");
    return;
  }
  els.legend.style.display = "";
  const arr = filteredScenarios();
  const max = Math.max(1, ...arr.map(s => Math.abs(scenarioMetric(s))));
  drawBoardGrid(ctx, r);
  drawClusterBlocks(ctx, r, arr);
  arr.forEach(s => drawHotspotGlyph(ctx, s, r, max));
  drawLabelBubbles(r, arr);
  drawLabelConnections(r, arr);
  drawLabelBubbles(r, arr);
  updateHover(r);
}
function updateDevopsCanvasHover(rect) {
  let hit = null;
  for (const n of state.devopsCanvasHits || []) {
    if (state.mouseX >= n.x && state.mouseX <= n.x + n.w && state.mouseY >= n.y && state.mouseY <= n.y + n.h) {
      hit = n;
      break;
    }
  }
  state.devopsHoverHit = hit;
  state.hover = hit && hit.kind === "scenario" ? hit.s : null;
  state.hoverLabel = null;
  if (!hit) { els.hoverCard.classList.remove("show"); return; }
  if (hit.kind === "scenario") {
    const j = scenarioJudgement(hit.s);
    els.hoverCard.innerHTML = `<b>${escapeHtml(scenarioName(hit.s))}</b><span>${escapeHtml(hit.s.suite_name || "")}</span><span>${escapeHtml(j.label)}: ${escapeHtml(j.reason)}</span>`;
  } else if (hit.kind === "frame") {
    els.hoverCard.innerHTML = `<b>Frame ${escapeHtml(hit.frame)}</b><span>Click to open the viewer at this evidence frame.</span>`;
  } else if (hit.kind === "suite") {
    els.hoverCard.innerHTML = `<b>${escapeHtml(hit.suite.replace(/^DevOps_V1_/, ""))}</b><span>Click to expand or collapse this suite.</span>`;
  } else {
    els.hoverCard.innerHTML = `<b>Open Viewer</b><span>Open the selected scenario at the most suspicious frame.</span>`;
  }
  els.hoverCard.style.left = `${Math.max(8, Math.min(Math.max(8, rect.width - 340), state.mouseX))}px`;
  els.hoverCard.style.top = `${Math.max(8, Math.min(Math.max(8, rect.height - 120), state.mouseY))}px`;
  els.hoverCard.classList.add("show");
}
function updateStatsHover(rect) {
  let best = null;
  for (const n of state.statNodes || []) {
    const d = Math.hypot(state.mouseX - n.x, state.mouseY - n.y);
    if (d <= n.r) { best = n.s; break; }
  }
  state.hover = best;
  state.hoverLabel = null;
  if (!best) { els.hoverCard.classList.remove("show"); return; }
  const m = scenarioMetric(best);
  els.hoverCard.innerHTML = `<b>${escapeHtml(scenarioName(best))}</b><span>${escapeHtml(best.suite_name || "")} · ${fmt(best.frames)} frames</span><span>${state.compare ? `ΔFP ${fmtDelta(best.delta_fp)} · ΔFN ${fmtDelta(best.delta_fn)}` : `FP ${fmt(best.fp)} · FN ${fmt(best.fn)} · TP ${fmt(best.tp)}`}</span><span>${compareLensLabel()} ${lensMetricText(m)}</span>`;
  els.hoverCard.style.left = `${Math.max(8, Math.min(Math.max(8, rect.width - 340), state.mouseX))}px`;
  els.hoverCard.style.top = `${Math.max(8, Math.min(Math.max(8, rect.height - 120), state.mouseY))}px`;
  els.hoverCard.classList.add("show");
}
function updateHover(rect) {
  let best = null, bestD = 26;
  let bestLabel = null, bestLabelD = 24;
  for (const node of state.labelNodes || []) {
    const d = Math.hypot(state.mouseX - node.x, state.mouseY - node.y);
    if (d < Math.max(bestLabelD, node.r + 10)) { bestLabel = node; bestLabelD = d; }
  }
  state.hoverLabel = bestLabel;
  if (bestLabel) {
    els.hoverCard.innerHTML = `<b>${escapeHtml(bestLabel.label)}</b><span>Click to show scenario contribution links.</span><span>${state.compare ? compareLensLabel() : state.lens.toUpperCase()} ${lensMetricText(bestLabel.value)}</span>`;
    els.hoverCard.style.left = `${Math.max(8, Math.min(Math.max(8, rect.width - 340), state.mouseX))}px`;
    els.hoverCard.style.top = `${Math.max(8, Math.min(Math.max(8, rect.height - 120), state.mouseY))}px`;
    els.hoverCard.classList.add("show");
    state.hover = null;
    return;
  }
  for (const s of filteredScenarios()) {
    if (!Number.isFinite(s._sx) || !Number.isFinite(s._sy)) continue;
    const d = Math.hypot(state.mouseX - s._sx, state.mouseY - s._sy);
    if (d < Math.max(bestD, s._sr || 16)) { best = s; bestD = d; }
  }
  state.hover = best;
  state.hoverLabel = null;
  if (!best) { els.hoverCard.classList.remove("show"); return; }
  const m = scenarioMetric(best);
  els.hoverCard.innerHTML = `<b>${escapeHtml(scenarioName(best))}</b><span>${escapeHtml(best._cluster || "")} · ${fmt(best.frames)} frames</span><span>${compareLensLabel()} ${lensMetricText(m)} · ${state.compare ? `ΔFP ${fmtDelta(best.delta_fp)} · ΔFN ${fmtDelta(best.delta_fn)}` : `FP ${fmt(best.fp)} · FN ${fmt(best.fn)}`}</span><span>${escapeHtml((best.labels || []).slice(0, 3).map(x => state.compare ? `${x.label}:ΔFP${fmtDelta(x.delta_fp || 0)}` : `${x.label}:${x.fp}FP/${x.fn}FN`).join(" · "))}</span>`;
  els.hoverCard.style.left = `${Math.max(8, Math.min(Math.max(8, rect.width - 340), state.mouseX))}px`;
  els.hoverCard.style.top = `${Math.max(8, Math.min(Math.max(8, rect.height - 120), state.mouseY))}px`;
  els.hoverCard.classList.add("show");
}
function renderCurve(message = "") {
  const r = resizeCanvas(els.curve, curveCtx);
  curveCtx.clearRect(0, 0, r.width, r.height);
  curveCtx.fillStyle = TH.a("deep", .72);
  curveCtx.fillRect(0, 0, r.width, r.height);
  curveCtx.strokeStyle = TH.a("line", .18);
  curveCtx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = 24 + i * (r.height - 42) / 3;
    curveCtx.beginPath(); curveCtx.moveTo(8, y); curveCtx.lineTo(r.width - 8, y); curveCtx.stroke();
  }
  if (!state.curve.length) {
    const text = message || "No frames matched the selected label/range.";
    els.curveStatus.textContent = text;
    curveCtx.fillStyle = TH.c("muted");
    curveCtx.font = "12px Inter, sans-serif";
    curveCtx.fillText(text, 12, Math.max(28, r.height / 2));
    return;
  }
  const max = Math.max(1, ...state.curve.map(f => Math.max(Math.abs(f.fp || 0), Math.abs(f.fn || 0), Math.abs(f.tp || 0))));
  const plot = {x: 10, y: 24, w: Math.max(1, r.width - 20), h: Math.max(1, r.height - 42)};
  const barW = Math.max(1, plot.w / Math.max(1, state.curve.length) * .82);
  if (state.compare) {
    const mid = plot.y + plot.h / 2;
    curveCtx.strokeStyle = TH.a("lineStrong", .32);
    curveCtx.beginPath(); curveCtx.moveTo(plot.x, mid); curveCtx.lineTo(plot.x + plot.w, mid); curveCtx.stroke();
    const drawDeltaBar = (x, value, offset, colorPos, colorNeg) => {
      const h = (Math.abs(Number(value) || 0) / max) * (plot.h / 2);
      curveCtx.fillStyle = value >= 0 ? colorPos : colorNeg;
      curveCtx.fillRect(x + offset - barW / 4, value >= 0 ? mid - h : mid, barW * .28, h);
    };
    state.curve.forEach((f, i) => {
      const x = plot.x + i * plot.w / Math.max(1, state.curve.length - 1);
      drawDeltaBar(x, f.fp || 0, -barW * .18, TH.a("bad", .72), TH.a("good", .64));
      drawDeltaBar(x, f.fn || 0, barW * .18, TH.a("warn", .72), TH.a("good", .5));
    });
    curveCtx.fillStyle = TH.c("bad"); curveCtx.font = "700 11px Inter, sans-serif";
    curveCtx.fillText("ΔFP", 10, 16); curveCtx.fillStyle = TH.c("warn"); curveCtx.fillText("ΔFN", 52, 16); curveCtx.fillStyle = TH.c("good"); curveCtx.fillText("below = improved", 96, 16);
    const peak = [...state.curve].sort((a, b) => (Math.abs(b.fp || 0) + Math.abs(b.fn || 0)) - (Math.abs(a.fp || 0) + Math.abs(a.fn || 0)))[0];
    els.curveStatus.textContent = `${state.curve.length} frames · largest change frame ${peak ? peak.frame : "-"} · max |delta| ${Math.round(max)}`;
    drawCurveFrameMarker(plot);
    return;
  }
  state.curve.forEach((f, i) => {
    const x = plot.x + i * plot.w / Math.max(1, state.curve.length - 1);
    const fpH = ((Number(f.fp) || 0) / max) * plot.h;
    const fnH = ((Number(f.fn) || 0) / max) * plot.h;
    curveCtx.fillStyle = TH.a("bad", .58);
    curveCtx.fillRect(x - barW / 2, plot.y + plot.h - fpH, barW * .48, fpH);
    curveCtx.fillStyle = TH.a("warn", .58);
    curveCtx.fillRect(x, plot.y + plot.h - fnH, barW * .48, fnH);
  });
  const draw = (key, color) => {
    curveCtx.strokeStyle = color; curveCtx.lineWidth = key === "tp" ? 2.4 : 1.6; curveCtx.beginPath();
    state.curve.forEach((f, i) => {
      const x = plot.x + i * plot.w / Math.max(1, state.curve.length - 1);
      const y = plot.y + plot.h - ((Number(f[key]) || 0) / max) * plot.h;
      if (i === 0) curveCtx.moveTo(x, y); else curveCtx.lineTo(x, y);
    });
    curveCtx.stroke();
  };
  draw("tp", TH.c("accent"));
  draw("fp", TH.c("bad"));
  draw("fn", TH.c("warn"));
  curveCtx.fillStyle = TH.c("accent"); curveCtx.font = "700 11px Inter, sans-serif";
  curveCtx.fillText("TP", 10, 16); curveCtx.fillStyle = TH.c("bad"); curveCtx.fillText("FP", 44, 16); curveCtx.fillStyle = TH.c("warn"); curveCtx.fillText("FN", 78, 16);
  curveCtx.fillStyle = TH.c("muted"); curveCtx.fillText(`max ${Math.round(max)}`, r.width - 68, 16);
  const peak = [...state.curve].sort((a, b) => ((b.fp || 0) + (b.fn || 0)) - ((a.fp || 0) + (a.fn || 0)))[0];
  els.curveStatus.textContent = `${state.curve.length} frames · peak frame ${peak ? peak.frame : "-"} · max ${Math.round(max)} objects/frame`;
  drawCurveFrameMarker(plot);
}
function currentPreviewFrameNumber() {
  return state.previewFrames[state.previewIndex] ? Number(state.previewFrames[state.previewIndex].frame) : null;
}
function curveXForFrame(frame, plot) {
  if (!state.curve.length || frame == null) return null;
  let best = 0, dist = Infinity;
  state.curve.forEach((f, i) => {
    const d = Math.abs(Number(f.frame) - Number(frame));
    if (d < dist) { best = i; dist = d; }
  });
  return plot.x + best * plot.w / Math.max(1, state.curve.length - 1);
}
function drawCurveFrameMarker(plot) {
  const frame = currentPreviewFrameNumber();
  const x = curveXForFrame(frame, plot);
  if (x == null) return;
  curveCtx.strokeStyle = TH.a("marker", .88);
  curveCtx.lineWidth = 1.5;
  curveCtx.beginPath();
  curveCtx.moveTo(x, plot.y - 4);
  curveCtx.lineTo(x, plot.y + plot.h + 4);
  curveCtx.stroke();
  curveCtx.fillStyle = TH.c("marker");
  curveCtx.beginPath();
  curveCtx.arc(x, plot.y - 5, 3, 0, Math.PI * 2);
  curveCtx.fill();
  curveCtx.font = "800 10px Inter, sans-serif";
  curveCtx.textAlign = x > plot.x + plot.w - 74 ? "right" : "left";
  curveCtx.fillText(`frame ${frame}`, x + (curveCtx.textAlign === "right" ? -6 : 6), plot.y + 10);
  curveCtx.textAlign = "left";
}
