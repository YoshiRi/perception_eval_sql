var BBOX_VIEWER_MIN_DISTANCE = 3;
var BBOX_VIEWER_MIN_PROJECTION_DISTANCE = 4;

function fitBounds() {
  let maxAbs = 30;
  for (const f of state.frames) for (const b of f.boxes) {
    maxAbs = Math.max(maxAbs, Math.abs(b.x || 0) + 6, Math.abs(b.y || 0) + 6);
    if (Array.isArray(b.footprint)) {
      for (const pt of b.footprint) {
        maxAbs = Math.max(maxAbs, Math.abs(Number(pt[0]) || 0) + 6, Math.abs(Number(pt[1]) || 0) + 6);
      }
    }
  }
  state.bounds.maxAbs = maxAbs;
  state.distance = Math.max(45, maxAbs * 1.45);
  state.panX = 0;
  state.panY = 0;
}
function cameraFromOriginalPosition(x, y, z, distance = null) {
  const horiz = Math.hypot(x, y);
  const r = Math.hypot(horiz, z);
  state.yaw = Math.atan2(-y, -x);
  state.pitch = Math.max(.18, Math.min(1.18, Math.asin(Math.max(.01, z) / Math.max(1, r))));
  state.distance = distance == null ? Math.max(32, state.bounds.maxAbs * 1.18) : distance;
}
function setCameraPreset(kind, keepTarget = false) {
  if (!keepTarget) { state.panX = 0; state.panY = 0; }
  if (kind === "top") {
    els.viewMode.value = "bev";
    state.distance = Math.max(45, state.bounds.maxAbs * 1.45);
  } else if (kind === "follow") {
    els.viewMode.value = "perspective";
    cameraFromOriginalPosition(-10, -2, 3.8);
  } else {
    els.viewMode.value = "perspective";
    cameraFromOriginalPosition(-12, -8, 4.5);
  }
  render();
}
function focusSelectedObject() {
  if (!state.selected) return false;
  state.panX = Number(state.selected.x || 0);
  state.panY = Number(state.selected.y || 0);
  render();
  return true;
}
function perspectiveCameraBasis() {
  const target = [state.panX, state.panY, 0];
  const radius = Math.max(BBOX_VIEWER_MIN_DISTANCE, state.distance);
  const elev = Math.max(0.18, Math.min(1.24, state.pitch));
  const cam = [
    target[0] - Math.cos(elev) * Math.cos(state.yaw) * radius,
    target[1] - Math.cos(elev) * Math.sin(state.yaw) * radius,
    Math.max(0.6, Math.sin(elev) * radius)
  ];
  const forward = normalize3([target[0] - cam[0], target[1] - cam[1], target[2] - cam[2]]);
  const worldUp = [0, 0, 1];
  let right = normalize3(cross3(forward, worldUp));
  if (!Number.isFinite(right[0])) right = [1, 0, 0];
  const up = normalize3(cross3(right, forward));
  return {target, cam, forward, right, up};
}
function panPerspectiveByScreenDelta(dx, dy) {
  const basis = perspectiveCameraBasis();
  const scale = Math.min(els.canvas.clientWidth, els.canvas.clientHeight) / Math.max(18, state.distance);
  const sx = Math.max(0.001, scale);
  const move = [
    (-basis.right[0] * dx + basis.up[0] * dy) / sx,
    (-basis.right[1] * dx + basis.up[1] * dy) / sx
  ];
  state.panX += move[0];
  state.panY += move[1];
}
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const r = els.canvas.getBoundingClientRect();
  els.canvas.width = Math.max(1, Math.floor(r.width * dpr));
  els.canvas.height = Math.max(1, Math.floor(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function corners(b) {
  const l = Math.max(.01, b.length) / 2, w = Math.max(.01, b.width) / 2, h = Math.max(.05, b.height || 1.5) / 2;
  const c = Math.cos(b.yaw || 0), s = Math.sin(b.yaw || 0);
  return [[l,w,h],[l,-w,h],[-l,-w,h],[-l,w,h],[l,w,-h],[l,-w,-h],[-l,-w,-h],[-l,w,-h]].map(p => [
    (b.x || 0) + p[0] * c - p[1] * s,
    (b.y || 0) + p[0] * s + p[1] * c,
    (b.z || 0) + p[2]
  ]);
}
function isPointLikeBox(b) {
  const shape = String(b.shape_type || b.type || "").toLowerCase();
  const l = Number(b.length) || 0;
  const w = Number(b.width) || 0;
  return shape === "polygon" || shape === "point" || l <= 0 || w <= 0;
}
function project(p) {
  const vp = activeViewport || {x: 0, y: 0, w: els.canvas.clientWidth, h: els.canvas.clientHeight};
  const panX = state.panX;
  const panY = state.panY;
  if (els.viewMode.value === "bev") {
    const scale = Math.min(vp.w, vp.h) / Math.max(BBOX_VIEWER_MIN_PROJECTION_DISTANCE, state.distance * 2.15);
    return [
      vp.x + vp.w / 2 - (p[1] - panY) * scale,
      vp.y + vp.h / 2 - (p[0] - panX) * scale,
      scale
    ];
  }
  const {cam, forward, right, up} = perspectiveCameraBasis();
  const rel = [p[0] - cam[0], p[1] - cam[1], p[2] - cam[2]];
  const depth = Math.max(1, dot3(rel, forward));
  const px = dot3(rel, right);
  const py = dot3(rel, up);
  const focal = Math.min(vp.w, vp.h) * 0.92;
  const scale = focal / depth;
  return [vp.x + vp.w / 2 + px * scale, vp.y + vp.h / 2 - py * scale, scale];
}
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize3(v) {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > 1e-6 ? [v[0] / n, v[1] / n, v[2] / n] : [NaN, NaN, NaN];
}

function compareLayoutMode() {
  return els.compareLayout.value === "split" ? "side_by_side" : els.compareLayout.value;
}
function compareSideBySideActive() {
  return state.compare && compareLayoutMode() === "side_by_side";
}
function compareCurtainActive() {
  return state.compare && compareLayoutMode() === "curtain";
}
