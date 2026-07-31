"""Pixel-art "office floor" that shows workflow tasks working in realtime.

One desk per task, one worker per desk. Workers walk in through the door when a task
is queued, wait beside the desk (sipping coffee, stretching), sit and type while the
job runs, then celebrate under confetti -- or slump at a red monitor -- and finally
walk back out the door. Progress percent and the task's ``progress_message`` are drawn
above each desk, so a glance at the floor answers "what is running and how far along".

The scene follows the dashboard theme (``lib.ui.theme``): a sunlit office on the light
theme, a night shift on the dark theme. Characters are proper sprites -- 12x18 pixel
maps with shading and an auto-drawn outline -- rather than stacks of rectangles.

Rendered with ``st.components.v1.html``. The page re-renders this inside its 3-second
live fragment; every animation is a pure function of wall-clock time, the task's own
timestamps, and a hash of the task id, so a re-render (which reloads the iframe) never
visibly resets the scene -- the characters simply keep walking/typing where the clock
says they should be.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

# A finished task keeps its desk long enough to celebrate/slump and walk out the door
# (the walk-out timeline in the JS below adds up to just under this).
RECENT_FINISH_SECONDS = 75
# Desks beyond this become a "+N more" note instead of stretching the floor forever.
MAX_DESKS = 10

_FLOOR_HEIGHT_PX = 300
_IDLE_HEIGHT_PX = 204


def _theme() -> str:
    try:
        from lib.ui.theme import active_theme

        return active_theme()
    except Exception:
        return "light"


def _epoch_ms(value: Any) -> Optional[int]:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _params(task: Dict[str, Any]) -> Dict[str, Any]:
    params = task.get("parameters") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    return params if isinstance(params, dict) else {}


def _task_name(task: Dict[str, Any]) -> str:
    params = _params(task)
    for key in ("target_name", "output_path", "job_id", "eval_root", "pkl_dir"):
        value = str(params.get(key) or "").strip()
        if value:
            return value.rstrip("/").rsplit("/", 1)[-1]
    return str(task.get("type") or "task")


def _requested_by(task: Dict[str, Any]) -> str:
    requester = _params(task).get("_requester")
    if isinstance(requester, dict):
        return str(requester.get("name") or requester.get("email") or "").strip()
    return ""


def _payload(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tasks worth a desk: everything active, plus finishes recent enough to celebrate."""
    now_ms = int(time.time() * 1000)
    items: List[Dict[str, Any]] = []
    for task in tasks:
        status = str(task.get("status") or "")
        updated = _epoch_ms(task.get("updated_at"))
        if status in ("completed", "failed"):
            if updated is None or now_ms - updated > RECENT_FINISH_SECONDS * 1000:
                continue
        elif status not in ("pending", "running"):
            continue
        pct = task.get("progress_pct")
        try:
            pct = None if pct is None else max(0.0, min(100.0, float(pct)))
        except (TypeError, ValueError):
            pct = None
        params = _params(task)
        output = str(params.get("output_path") or "").strip()
        items.append({
            "id": str(task.get("id") or ""),
            "status": status,
            "name": _task_name(task),
            "type": str(task.get("type") or ""),
            "pct": pct,
            "message": str(task.get("progress_message") or "").strip(),
            "error": str(task.get("error_message") or "").strip(),
            "created": _epoch_ms(task.get("created_at")),
            "updated": updated,
            "run": output.rstrip("/").rsplit("/", 1)[-1] if output else "",
            "target": str(params.get("target_name") or "").strip(),
            "by": _requested_by(task),
            "result": str(task.get("result_path") or "").strip(),
        })
    # Active first (running before pending), then the recently finished.
    order = {"running": 0, "pending": 1, "completed": 2, "failed": 2}
    items.sort(key=lambda t: (order.get(t["status"], 3), -(t["created"] or 0)))
    return items


def render_pixel_office(tasks: List[Dict[str, Any]]) -> None:
    """Draw the office floor for the given task rows (same rows the task list uses)."""
    items = _payload(tasks)
    overflow = max(0, len(items) - MAX_DESKS)
    items = items[:MAX_DESKS]
    height = _FLOOR_HEIGHT_PX if items else _IDLE_HEIGHT_PX
    data = json.dumps(
        {"tasks": items, "overflow": overflow, "theme": _theme()}
    ).replace("</", "<\\/")
    components.html(_HTML_TEMPLATE.replace("__DATA__", data), height=height + 8, scrolling=False)


_HTML_TEMPLATE = r"""
<div id="wrap" style="overflow-x:auto;overflow-y:hidden;"></div>
<script>
(() => {
const DATA = __DATA__;
const S = 3;                                    // css px per virtual pixel
const CELL_W = 92, FLOOR_H = 100, IDLE_H = 68;  // virtual units
const LEFT = 64, RIGHT = 46;                    // door/whiteboard wing, coffee corner
const tasks = DATA.tasks || [];
const idle = tasks.length === 0;
const minCells = Math.max(2, Math.ceil(((document.body.clientWidth || 700) / S - LEFT - RIGHT) / CELL_W));
const cells = idle ? minCells : tasks.length;
const W = LEFT + cells * CELL_W + RIGHT, H = idle ? IDLE_H : FLOOR_H;
const FLOOR_Y = H - 30;
const DOOR_X = 16;                              // door center on the back wall

// ------------------------------------------------------------------ theme palettes
const dark = DATA.theme === 'dark';
const T = dark ? {
  wall: '#262b3d', wallPanel: '#1f2434', wallLine: '#1a1e2e', skirt: '#171b28',
  floorA: '#2b3145', floorB: '#272c3f', floorEdge: '#313850',
  woodTop: '#8f6845', wood: '#7a5638', woodDark: '#5e4028', woodFront: '#684a2e',
  chair: '#454d66', chairDark: '#343b52', metal: '#586080',
  bezel: '#12151f', bezelHi: '#2b3145', screen: '#0a0e18',
  text: '#e2e6f2', muted: '#8b93a8', faint: '#4a5470',
  barBg: '#161a26', barEdge: '#0f1220',
  bubbleBg: '#e6e9f2', bubbleEdge: '#aab1c6', bubbleInk: '#232839',
  boardBg: '#dfe3ee', boardEdge: '#9aa0b5', boardInk: '#3a4157',
  rack: '#171b28', rackSlot: '#242a3c', rackHi: '#3a4157',
  outline: '#151823',
  running: '#54c0e8', pending: '#98a1b8', completed: '#5fce7f', failed: '#e06060',
  runningDim: '#2e6f86', screenText: '#e8f4f8',
  glow: 'rgba(255, 214, 120, 0.05)',
} : {
  wall: '#eef1f7', wallPanel: '#e3e7f0', wallLine: '#d5dae6', skirt: '#c3cbda',
  floorA: '#dde2ec', floorB: '#d4dae6', floorEdge: '#e7ebf3',
  woodTop: '#d29d6a', wood: '#c08a58', woodDark: '#996b3e', woodFront: '#ad7a4b',
  chair: '#8d97b0', chairDark: '#6e788f', metal: '#9aa4bc',
  bezel: '#2a3040', bezelHi: '#485069', screen: '#10141f',
  text: '#0f172a', muted: '#64748b', faint: '#94a3b8',
  barBg: '#e2e8f0', barEdge: '#cbd5e1',
  bubbleBg: '#ffffff', bubbleEdge: '#b9c2d0', bubbleInk: '#0f172a',
  boardBg: '#ffffff', boardEdge: '#aab4c4', boardInk: '#334155',
  rack: '#b9c2d2', rackSlot: '#98a3b8', rackHi: '#dde3ec',
  outline: '#3d4557',
  running: '#0e7490', pending: '#64748b', completed: '#15803d', failed: '#dc2626',
  runningDim: '#67b7d1', screenText: '#e8f4f8',
  glow: 'rgba(255, 214, 120, 0.0)',
};
const STATUS_LABEL = { running: 'RUNNING', pending: 'QUEUED', completed: 'DONE', failed: 'FAILED' };
const statusColor = s => T[s] || T.pending;

// Finish-line timeline (ms since updated_at): party/slump, then walk out, then gone.
const CHEER_MS = 45000, WALKOUT_MS = 14000;
const WALKIN_MS = 3800;                         // queued workers walk in after created_at

const wrap = document.getElementById('wrap');
const canvas = document.createElement('canvas');
const dpr = Math.min(2, window.devicePixelRatio || 1);
canvas.width = W * S * dpr; canvas.height = H * S * dpr;
canvas.style.width = (W * S) + 'px'; canvas.style.height = (H * S) + 'px';
canvas.style.imageRendering = 'pixelated';
canvas.style.borderRadius = '8px';
wrap.appendChild(canvas);
const ctx = canvas.getContext('2d');
ctx.imageSmoothingEnabled = false;

// ---------------------------------------------------------------- mouse interaction
// Hovering a desk shows a tooltip; clicking it opens a detail card. The selection is
// kept in sessionStorage so the card survives the fragment's periodic iframe reloads.
const mouse = { x: -1, y: -1, over: false };
let selectedId = null;
try { selectedId = sessionStorage.getItem('pxOfficeSel') || null; } catch (e) {}
if (selectedId && !tasks.some(t => t.id === selectedId)) selectedId = null;
let closeRect = null, panelRect = null;                    // set while the card is drawn

function setSelected(id) {
  selectedId = id;
  try {
    if (id) sessionStorage.setItem('pxOfficeSel', id);
    else sessionStorage.removeItem('pxOfficeSel');
  } catch (e) {}
}
function hoverCell() {
  if (idle || !mouse.over || mouse.y < 2) return -1;
  const i = Math.floor((mouse.x - LEFT) / CELL_W);
  return (mouse.x >= LEFT && i >= 0 && i < tasks.length) ? i : -1;
}
const inRect = r => r && mouse.x >= r[0] && mouse.x <= r[0] + r[2] && mouse.y >= r[1] && mouse.y <= r[1] + r[3];
canvas.addEventListener('mousemove', e => {
  const r = canvas.getBoundingClientRect();
  mouse.x = (e.clientX - r.left) / S; mouse.y = (e.clientY - r.top) / S; mouse.over = true;
});
canvas.addEventListener('mouseleave', () => { mouse.over = false; mouse.x = mouse.y = -1; });
canvas.addEventListener('click', () => {
  if (selectedId) {
    if (!inRect(panelRect) || inRect(closeRect)) setSelected(null);
    return;
  }
  const i = hoverCell();
  if (i >= 0) setSelected(tasks[i].id);
});

function hash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function base() { ctx.setTransform(S * dpr, 0, 0, S * dpr, 0, 0); }
const px = (x, y, w, h, c) => { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), w, h); };
function text(str, x, y, color, size, align) {
  ctx.fillStyle = color;
  ctx.font = 'bold ' + (size || 4) + 'px "Courier New", monospace';
  ctx.textAlign = align || 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(str, x, y);
}
function shade(hex, f) {                        // f < 1 darkens, f > 1 lightens
  const n = parseInt(hex.slice(1), 16);
  const ch = v => Math.max(0, Math.min(255, Math.round(v * f)));
  return 'rgb(' + ch(n >>> 16) + ',' + ch((n >>> 8) & 255) + ',' + ch(n & 255) + ')';
}

// ------------------------------------------------------------- characters (12 x 18)
// Sprite maps: H/h hair+shade, K/k skin+shade, E eye, S/s shirt+shade, P/p trousers,
// B shoes, M mug, '.' empty. A 1px outline is drawn automatically around the
// silhouette, which is what makes them read as drawn characters instead of icons.

const SPRITES = {
frontStand: [
'....HHHH....',
'..HHHHHHHH..',
'..HHHHHHHH..',
'..HHHHHHHH..',
'..hKKKKKKh..',
'...KEKKEK...',
'...KKKKKK...',
'...kKKKKk...',
'....KKKK....',
'..SSSSSSSS..',
'.SSSSSSSSSS.',
'.SsSSSSSSsS.',
'.Ss.SSSS.sS.',
'.KK.SsSs.KK.',
'...PPPPPP...',
'...PP..PP...',
'...PP..PP...',
'..BBB..BBB..',
],
frontSip: [
'....HHHH....',
'..HHHHHHHH..',
'..HHHHHHHH..',
'..HHHHHHHH..',
'..hKKKKKKh..',
'...KEKKEK.MM',
'...KKKKKK.MM',
'...kKKKKk.K.',
'....KKKK..K.',
'..SSSSSSSSS.',
'.SSSSSSSSSS.',
'.SsSSSSSSsS.',
'.Ss.SSSS....',
'.KK.SsSs....',
'...PPPPPP...',
'...PP..PP...',
'...PP..PP...',
'..BBB..BBB..',
],
frontCheerA: [
'.K........K.',
'.K.HHHHHH.K.',
'.S.HHHHHH.S.',
'.S.HHHHHH.S.',
'.S.KKKKKK.S.',
'.SsKEKKEKsS.',
'..SKKKKKKS..',
'...kKKKKk...',
'....KKKK....',
'..SSSSSSSS..',
'.SSSSSSSSSS.',
'..SSSSSSSS..',
'...SSSSSS...',
'...SsSsSs...',
'...PPPPPP...',
'...PP..PP...',
'...PP..PP...',
'..BBB..BBB..',
],
frontCheerB: [
'............',
'...HHHHHH...',
'.KHHHHHHHHK.',
'.KHHHHHHHHK.',
'.S.KKKKKK.S.',
'.S.KEKKEK.S.',
'.SsKKKKKKsS.',
'..SkKKKKkS..',
'....KKKK....',
'..SSSSSSSS..',
'.SSSSSSSSSS.',
'..SSSSSSSS..',
'...SSSSSS...',
'...SsSsSs...',
'...PPPPPP...',
'...PP..PP...',
'...PP..PP...',
'..BBB..BBB..',
],
frontSlump: [
'............',
'............',
'....HHHH....',
'..HHHHHHHH..',
'..HHHHHHHH..',
'..HHHHHHHH..',
'..hHHHHHHh..',
'...kKKKKk...',
'....KKKK....',
'..SSSSSSSS..',
'.SsSSSSSSsS.',
'.Ss.SSSS.sS.',
'.Ss.SSSS.sS.',
'.KK.SsSs.KK.',
'...PPPPPP...',
'...PP..PP...',
'...PP..PP...',
'..BBB..BBB..',
],
sideA: [                                        // stride open, facing right
'...HHHHH....',
'..HHHHHHH...',
'..HHHHHHHH..',
'..HHHKKKKH..',
'..hHKKEKKK..',
'...hKKKKKk..',
'....kKKKk...',
'.....KKK....',
'...SSSSSS...',
'...SSSSSSs..',
'...SsSSSSs..',
'...SsKSSSs..',
'....SKSSs...',
'...PPPPPP...',
'..PPP..PPP..',
'..PP....PP..',
'.BBB.....BB.',
'............',
],
sideB: [                                        // legs passing
'............',
'...HHHHH....',
'..HHHHHHH...',
'..HHHHHHHH..',
'..HHHKKKKH..',
'..hHKKEKKK..',
'...hKKKKKk..',
'....kKKKk...',
'.....KKK....',
'...SSSSSS...',
'...SSSSSSs..',
'...SsSSSSs..',
'...SsKSSSs..',
'...PPPPPP...',
'....PPPP....',
'....PPPP....',
'....BBBB....',
'............',
],
backType: [
'....HHHH....',
'..HHHHHHHH..',
'..HHHHHHHH..',
'..HHHHHHHH..',
'..hHHHHHHh..',
'...HHHHHH...',
'....hhhh....',
'....KKKK....',
'..SSSSSSSS..',
'.SSSSSSSSSS.',
'.SsSSSSSSsS.',
'.Ss.SSSS.sS.',
'KK..SsSs..KK',
'....PPPP....',
'............',
'............',
'............',
'............',
],
backSip: [
'....HHHH....',
'..HHHHHHHH..',
'..HHHHHHHH..',
'..HHHHHHHH.M',
'..hHHHHHHhKM',
'...HHHHHH.K.',
'....hhhh..S.',
'....KKKK..S.',
'..SSSSSSSSS.',
'.SSSSSSSSSS.',
'.SsSSSSSSsS.',
'.Ss.SSSS....',
'KK..SsSs....',
'....PPPP....',
'............',
'............',
'............',
'............',
],
};

const SKIN = ['#efb98d', '#d99c6b', '#a06a42', '#f3ccab', '#8a5433'];
const HAIR = ['#5b3d22', '#2d2622', '#c9973f', '#83848f', '#8a4630', '#3e4a68'];
const SHIRT = ['#3f8f7a', '#c08a3e', '#7a5fae', '#4a7dbd', '#b35c76', '#5c8a4a', '#4aa3a3'];
const PANTS = ['#3a4a6b', '#4a4a52', '#5b4636', '#37536b'];

function look(seed) {
  const skin = SKIN[seed % SKIN.length], hair = HAIR[(seed >>> 3) % HAIR.length];
  const shirt = SHIRT[(seed >>> 6) % SHIRT.length], pants = PANTS[(seed >>> 9) % PANTS.length];
  return {
    H: hair, h: shade(hair, 0.72),
    K: skin, k: shade(skin, 0.8), E: '#20242e',
    S: shirt, s: shade(shirt, 0.74),
    P: pants, p: shade(pants, 0.75), B: shade(pants, 0.45),
    M: '#eef0f6',
    glasses: (seed >>> 12) % 3 === 0,
  };
}

function drawMap(map, legend) {
  const rows = map.length, cols = map[0].length;
  const solid = (r, c) => r >= 0 && c >= 0 && r < rows && c < cols && legend[map[r][c]] !== undefined && map[r][c] !== '.';
  ctx.fillStyle = T.outline;                    // silhouette outline first
  for (let r = -1; r <= rows; r++) {
    for (let c = -1; c <= cols; c++) {
      if (solid(r, c)) continue;
      if (solid(r - 1, c) || solid(r + 1, c) || solid(r, c - 1) || solid(r, c + 1)) {
        ctx.fillRect(c, r, 1, 1);
      }
    }
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const col = legend[map[r][c]];
      if (col !== undefined && map[r][c] !== '.') { ctx.fillStyle = col; ctx.fillRect(c, r, 1, 1); }
    }
  }
}

function sprite(name, seed, now, opts) {
  const L = look(seed);
  const legend = { ...L };
  if (opts && opts.blink) legend.E = L.K;
  drawMap(SPRITES[name], legend);
  if (L.glasses && name.startsWith('front') && name !== 'frontSlump') {
    const r = name === 'frontStand' || name === 'frontSip' ? 5 : (name === 'frontCheerA' ? 5 : 5);
    px(2.6, r, 2, 1.4, 'rgba(32,36,46,0.55)'); px(6.6, r, 2, 1.4, 'rgba(32,36,46,0.55)');
    px(5, r, 2, 0.6, '#20242e');
  }
  if (name === 'frontSip' && Math.floor(now / 420 + seed) % 2) px(10, 3, 1, 1, T.muted);
  if (name === 'backSip' && Math.floor(now / 420 + seed) % 2) px(11, 1, 1, 1, T.muted);
}

function drawAt(x, y, flip, fn) {
  ctx.setTransform((flip ? -1 : 1) * S * dpr, 0, 0, S * dpr,
                   Math.round(x + (flip ? 12 : 0)) * S * dpr, Math.round(y) * S * dpr);
  fn();
  base();
}
function blinkNow(now, seed) { return Math.floor(now / 2600 + seed) % 8 === 0; }

// --------------------------------------------------------------------- set dressing

function room(now) {
  base();
  px(0, 0, W, H, T.wall);
  const wainscotY = FLOOR_Y - 22;
  px(0, wainscotY, W, FLOOR_Y - wainscotY, T.wallPanel);           // lower wall band
  px(0, wainscotY, W, 1, T.wallLine);
  for (let i = 0; i <= cells + 1; i++) {                           // wall panel seams
    const x = LEFT + (i - 1) * CELL_W;
    if (x > 4 && x < W - 4) px(x, 2, 1, FLOOR_Y - 4, T.wallLine);
  }
  px(0, FLOOR_Y - 2, W, 2, T.skirt);                               // baseboard
  for (let x = 0; x < W; x += 10) {                                // carpet tiles
    for (let y = FLOOR_Y; y < H; y += 5) {
      const a = ((x / 10 + (y - FLOOR_Y) / 5) % 2);
      px(x, y, 10, 5, a ? T.floorA : T.floorB);
      px(x, y, 10, 1, T.floorEdge);
      px(x, y, 1, 5, a ? T.floorB : T.floorA);
    }
  }
  // hanging ceiling lamps, one per desk cell
  for (let i = 0; i < cells; i++) {
    const lx = LEFT + i * CELL_W + CELL_W / 2;
    px(lx - 1, 0, 2, 3, T.metal);
    px(lx - 5, 3, 10, 3, dark ? '#3a4157' : '#b9c2d2');
    px(lx - 4, 6, 8, 1, dark ? '#ffd678' : '#fff3c4');
    if (dark) { ctx.fillStyle = T.glow; ctx.fillRect(lx - 12, 7, 24, FLOOR_Y - 7); }
  }
  clock(now);
}

function clock(now) {
  const cx = W - 12, cy = 10;
  px(cx - 5, cy - 5, 10, 10, T.outline);
  px(cx - 4, cy - 4, 8, 8, dark ? '#d8dbe6' : '#ffffff');
  px(cx, cy - 4, 1, 1, T.muted); px(cx, cy + 3, 1, 1, T.muted);
  px(cx - 4, cy, 1, 1, T.muted); px(cx + 3, cy, 1, 1, T.muted);
  const d = new Date(now);
  const ha = (d.getHours() % 12 + d.getMinutes() / 60) / 12 * Math.PI * 2 - Math.PI / 2;
  const ma = d.getMinutes() / 60 * Math.PI * 2 - Math.PI / 2;
  px(cx + Math.cos(ha) * 2, cy + Math.sin(ha) * 2, 1, 1, '#232839');
  px(cx + Math.cos(ma) * 3, cy + Math.sin(ma) * 3, 1, 1, T.failed);
}

function door(now) {
  const x = DOOR_X - 8, top = FLOOR_Y - 28;
  px(x - 1, top - 1, 18, 29, T.outline);                           // frame
  px(x, top, 16, 28, dark ? '#4a3a29' : '#a97e52');                // frame wood
  px(x + 1, top + 1, 14, 27, dark ? '#5d4933' : '#c99a68');        // door
  px(x + 3, top + 3, 10, 8, dark ? '#4a3a29' : '#b0824f');         // upper inset
  px(x + 4, top + 4, 8, 6, dark ? '#39485c' : '#d9ecf8');          // window pane
  px(x + 4, top + 4, 8, 1, dark ? '#4c6076' : '#eef7fd');
  px(x + 3, top + 14, 10, 10, dark ? '#4a3a29' : '#b0824f');       // lower inset
  px(x + 4, top + 15, 8, 8, dark ? '#55432f' : '#c08a58');
  px(x + 12, top + 12, 2, 3, dark ? '#c9a24b' : '#8a6a2f');        // handle
  px(x + 2, top - 7, 12, 5, T.outline);                            // EXIT sign
  px(x + 3, top - 6, 10, 3, dark ? '#173322' : '#dcfce7');
  text('EXIT', x + 4, top - 5.8, Math.floor(now / 900) % 4 ? T.completed : shade(T.completed, 0.6), 3.5);
}

function window_(x, y) {
  const w = 30, h = 18;
  px(x - 2, y - 2, w + 4, h + 4, T.outline);
  px(x - 1, y - 1, w + 2, h + 2, dark ? '#3a4157' : '#cdd5e2');    // frame
  if (dark) {                                                      // night
    px(x, y, w, h, '#0c1020');
    const hs = hash('sky');
    for (let i = 0; i < 10; i++) px(x + 1 + ((hs >>> i) % (w - 2)), y + 1 + ((hs >>> (i + 3)) % 7), 1, 1, '#aab1c6');
    px(x + w - 7, y + 2, 4, 4, '#e8e4d0'); px(x + w - 7, y + 2, 1, 1, '#0c1020');  // moon
    px(x + 2, y + 11, 5, 7, '#1a2438'); px(x + 9, y + 8, 6, 10, '#1e2a42');
    px(x + 17, y + 12, 5, 6, '#1a2438'); px(x + 24, y + 9, 4, 9, '#1e2a42');
    px(x + 10, y + 9, 1, 1, '#e8c44a'); px(x + 13, y + 11, 1, 1, '#e8c44a');
    px(x + 25, y + 10, 1, 1, '#54c0e8'); px(x + 3, y + 13, 1, 1, '#e8c44a');
  } else {                                                         // day
    px(x, y, w, h, '#aed7f2');
    px(x, y, w, 5, '#bfe0f8');
    px(x + 4, y + 3, 7, 2, '#ffffff'); px(x + 6, y + 2, 4, 1, '#ffffff');          // clouds
    px(x + 18, y + 5, 8, 2, '#f2f9ff'); px(x + 20, y + 4, 5, 1, '#f2f9ff');
    px(x + 24, y + 1, 3, 3, '#ffdf7e');                                            // sun
    px(x + 2, y + 12, 5, 6, '#8ba6c0'); px(x + 9, y + 9, 6, 9, '#9db4cc');         // skyline
    px(x + 17, y + 13, 5, 5, '#8ba6c0'); px(x + 24, y + 10, 4, 8, '#9db4cc');
  }
  px(x + w / 2 - 0.5, y - 1, 1, h + 2, dark ? '#3a4157' : '#cdd5e2');  // mullions
  px(x - 1, y + h / 2 - 0.5, w + 2, 1, dark ? '#3a4157' : '#cdd5e2');
}

function whiteboard(counts) {
  const w = 28, h = 24, x = 30, y = Math.max(6, FLOOR_Y - 36);
  px(x - 2, y - 2, w + 4, h + 4, T.outline);
  px(x - 1, y - 1, w + 2, h + 2, T.boardEdge);
  px(x, y, w, h, T.boardBg);
  text('TODAY', x + 3, y + 2, T.boardInk, 3.5);
  px(x + 2, y + 7, w - 4, 1, T.boardEdge);
  text('RUN', x + 3, y + 9.5, T.running, 4); text(String(counts.running), x + w - 4, y + 9.5, T.boardInk, 4, 'right');
  text('QUE', x + 3, y + 15, T.muted, 4);     text(String(counts.pending), x + w - 4, y + 15, T.boardInk, 4, 'right');
  px(x + 2, y + h - 2, 5, 1, '#d24545'); px(x + 8, y + h - 2, 5, 1, '#3a6fc4');   // markers
  px(x - 1, y + h + 1, w + 2, 1, T.boardEdge);                     // tray
  px(x + 2, y + h + 2, 1, FLOOR_Y - y - h + 2, T.metal);           // legs
  px(x + w - 3, y + h + 2, 1, FLOOR_Y - y - h + 2, T.metal);
}

function coffeeMachine(now) {
  const x = W - RIGHT + 10, y = FLOOR_Y - 22;
  px(x - 1, y - 1, 14, 24, T.outline);
  px(x, y, 12, 22, dark ? '#2b3145' : '#525c74');
  px(x, y, 12, 1, dark ? '#3d4560' : '#6b7690');
  px(x + 1, y + 2, 10, 4, '#12151f');
  text('CAFE', x + 2, y + 2.5, '#e8c44a', 3);
  px(x + 2, y + 8, 3, 2, '#d24545'); px(x + 6, y + 8, 3, 2, '#3a6fc4');   // buttons
  px(x + 3, y + 11, 6, 4, '#12151f');                              // dispenser
  px(x + 5, y + 13, 2, 2, '#eef0f6');                              // cup
  if (Math.floor(now / 650) % 3 === 0) px(x + 5, y + 11, 1, 2, '#8a6a4a');  // pour
  px(x + 2, y + 17, 8, 1, T.metal);                                // tray
  px(x + 10, y + 19, 1, 1, Math.floor(now / 800) % 2 ? '#e06060' : '#5c1f1f');
}

function plant(x, seed) {
  const y = FLOOR_Y - 4;                                           // pot base on floor
  const leaf = dark ? '#2e6b3f' : '#3f8f57', leafHi = dark ? '#3f8f57' : '#5cb371';
  px(x + 1, y - 9, 1, 5, dark ? '#1f4a2c' : '#2e6b3f');            // stems
  px(x + 4, y - 10, 1, 6, dark ? '#1f4a2c' : '#2e6b3f');
  px(x - 1, y - 12, 4, 4, leaf); px(x, y - 13, 2, 2, leafHi);
  px(x + 3, y - 14, 4, 4, leaf); px(x + 4, y - 15, 2, 2, leafHi);
  px(x + 1, y - 9, 4, 3, leaf); px(x + 2, y - 10, 2, 2, leafHi);
  px(x - 1, y - 5, 8, 2, dark ? '#7a3b2e' : '#b0644f');            // pot rim
  px(x, y - 3, 6, 3, dark ? '#5c2c22' : '#9c5843');
  px(x + 1, y - 3, 1, 3, dark ? '#7a3b2e' : '#b0644f');
}

function serverRack(x, y, seed, now, color) {
  px(x - 1, y - 1, 12, 28, T.outline);
  px(x, y, 10, 26, T.rack);
  px(x, y, 10, 1, T.rackHi);
  for (let i = 0; i < 5; i++) {
    const sy = y + 2 + i * 5;
    px(x + 1, sy, 8, 3, T.rackSlot);
    px(x + 1, sy, 8, 1, T.rackHi);
    px(x + 2, sy + 1, 3, 1, dark ? '#161a26' : '#7c879c');         // vents
    const on = Math.floor(now / (240 + (seed % 5) * 60) + i * 7 + seed) % 3 !== 0;
    px(x + 7, sy + 1, 1, 1, on ? color : T.faint);
  }
  px(x + 2, y + 26, 2, 1, T.outline); px(x + 6, y + 26, 2, 1, T.outline);  // feet
}

function monitorUnit(x0, deskY, seed, now, task, gone) {
  // The screen faces the viewer, read over the worker's shoulder. Drawn before the
  // desk (and before anyone standing behind it) so depth stacks far-to-near.
  const cx = x0 + CELL_W / 2;
  const mw = 24, mh = 15, mx = cx - mw / 2, my = deskY - mh - 3;
  px(mx - 1, my - 1, mw + 2, mh + 2, T.outline);
  px(mx, my, mw, mh, T.bezel);
  px(mx, my, mw, 1, T.bezelHi);
  px(mx + 1, my + 1, mw - 2, mh - 3, T.screen);
  px(mx + mw - 3, my + mh - 1.6, 1, 1, gone ? T.faint : statusColor(task.status));  // power led
  if (task.status === 'running') {
    for (let r = 0; r < 5; r++) {
      const lw = 5 + (hash(task.id + ':' + (r + Math.floor(now / 350))) % 13);
      px(mx + 2, my + 2 + r * 2, lw, 1, r % 2 ? T.runningDim : '#54c0e8');
    }
    if (Math.floor(now / 500 + seed) % 2) px(mx + 3 + (seed % 14), my + 11, 2, 1, T.screenText);
  } else if (task.status === 'completed' && !gone) {
    px(mx + 8, my + 8, 2, 2, '#5fce7f'); px(mx + 10, my + 6, 2, 2, '#5fce7f');
    px(mx + 12, my + 4, 2, 2, '#5fce7f'); px(mx + 14, my + 6, 1, 1, '#2e8a4e');
    text('OK', mx + 3, my + 2, '#5fce7f', 4);
  } else if (task.status === 'failed' && !gone) {
    if (Math.floor(now / 450) % 2) {
      px(mx + 11, my + 3, 2, 6, '#e06060'); px(mx + 11, my + 10, 2, 2, '#e06060');
    }
    text('ERR', mx + 2, my + 2, '#e06060', 3.5);
  } else if (task.status === 'pending') {
    if (Math.floor(now / 700 + seed) % 2) px(mx + 3, my + 3, 4, 1, '#4a5470');
  }
  px(cx - 1, my + mh + 1, 2, 2, T.metal);                          // stand
  px(cx - 4, deskY - 1, 8, 1, T.metal);
}

function deskTable(x0, deskY) {
  const cx = x0 + CELL_W / 2;
  const dx = x0 + 18, dyTop = deskY + 3;                           // drawer pedestal
  px(dx - 1, dyTop, 12, 13, T.outline);
  px(dx, dyTop, 10, 12, T.woodFront);
  for (let i = 0; i < 3; i++) {
    px(dx + 1, dyTop + 1 + i * 4, 8, 3, T.wood);
    px(dx + 1, dyTop + 1 + i * 4, 8, 1, T.woodTop);
    px(dx + 4, dyTop + 2 + i * 4, 2, 1, T.woodDark);               // handle
  }
  const dw = CELL_W - 26;                                          // desk top + apron
  px(x0 + 12, deskY - 1, dw, 1, T.outline);
  px(x0 + 12, deskY, dw, 2, T.woodTop);
  px(x0 + 12, deskY + 2, dw, 2, T.wood);
  px(x0 + 12, deskY + 4, 2, 11, T.woodDark);                       // legs
  px(x0 + 12 + dw - 2, deskY + 4, 2, 11, T.woodDark);
  // keyboard + mouse + desk mug
  px(cx - 7, deskY + 0.4, 10, 1.6, dark ? '#3a4157' : '#8f99b0');
  px(cx + 5, deskY + 0.6, 2, 1.4, dark ? '#3a4157' : '#8f99b0');
  px(x0 + 15, deskY - 3, 3, 3, '#d24545'); px(x0 + 18, deskY - 2, 1, 1, '#d24545');
}

function chairBehind(cx, deskY, seed) {
  // office chair seen from behind, drawn over the seated worker's hips
  const bx = cx - 6, by = deskY + 2;
  px(bx - 1, by - 1, 14, 8, T.outline);
  px(bx, by, 12, 7, T.chair);
  px(bx + 1, by + 1, 10, 1, shade(T.chair, 1.18));
  px(bx + 1, by + 5, 10, 1, T.chairDark);
  px(cx - 1, by + 7, 2, 4, T.metal);                               // gas lift
  px(cx - 5, by + 11, 10, 1, T.metal);                             // base
  px(cx - 5, by + 12, 2, 1, T.outline); px(cx + 3, by + 12, 2, 1, T.outline);
  px(cx - 1, by + 12, 2, 1, T.outline);
}

function bubble(cx, topY, msg, now, seed) {
  if (!msg) return;
  const maxChars = 26;
  let show = msg;
  if (msg.length > maxChars) {                                     // deterministic marquee
    const loop = msg + '   ';
    const off = Math.floor(now / 260 + seed) % loop.length;
    show = (loop + loop).slice(off, off + maxChars);
  }
  ctx.font = 'bold 4px "Courier New", monospace';
  const w = Math.min(CELL_W - 8, Math.ceil(ctx.measureText(show).width) + 5);
  const x = Math.max(2, Math.min(W - w - 2, cx - w / 2));
  px(x - 1, topY - 1, w + 2, 9, T.bubbleEdge);
  px(x, topY, w, 7, T.bubbleBg);
  px(cx - 1, topY + 8, 2, 2, T.bubbleBg);                          // tail
  text(show, x + 2.5, topY + 2, T.bubbleInk, 4);
}

function confetti(cx, cy, seed, now) {
  const colors = ['#5fce7f', '#54c0e8', '#e8c44a', '#e07daa', '#9a7ae0'];
  for (let i = 0; i < 12; i++) {
    const h = hash(seed + ':' + i);
    const t = ((now / 16) + (h % 90)) % 90;
    const x = cx + ((h >>> 4) % 37) - 18 + Math.sin((now / 300) + i) * 2;
    const y = cy - 8 + t * 0.34;
    if (y < cy + 18) px(x, y, 1, ((h >>> 8) % 2) + 1, colors[i % colors.length]);
  }
}

// ------------------------------------------------------------------- the desk + crew

function lerp(a, b, t) { return a + (b - a) * Math.max(0, Math.min(1, t)); }

const DESK_Y = H - 26;
const cellGeom = i => {
  const x0 = LEFT + i * CELL_W, cx = x0 + CELL_W / 2;
  return { x0, cx, standX: cx + 14, standY: DESK_Y - 15 };
};
const dwellFor = task => task.status === 'failed' ? CHEER_MS * 0.9 : CHEER_MS;
const isGone = (task, now) => (task.status === 'completed' || task.status === 'failed')
  && task.updated && now - task.updated > dwellFor(task) + WALKOUT_MS;

// Walkers use the corridor behind the desks, so they are drawn before any desk.
function crewWalking(i, task, now) {
  const seed = hash(task.id || String(i));
  const { standX, standY } = cellGeom(i);
  const doorY = FLOOR_Y - 16;                    // corridor: feet on the floor line
  if (task.status === 'pending') {
    const age = task.created ? now - task.created : WALKIN_MS;
    if (age < WALKIN_MS) {
      const t = age / WALKIN_MS;
      const x = lerp(DOOR_X - 6, standX, t), y = lerp(doorY, standY, t);
      const frame = Math.floor(now / 150) % 2 ? 'sideA' : 'sideB';
      drawAt(x, y, false, () => sprite(frame, seed, now));
      return true;
    }
    return false;
  }
  if (task.status === 'completed' || task.status === 'failed') {
    const since = task.updated ? now - task.updated : 0;
    const dwell = dwellFor(task);
    if (since >= dwell && since < dwell + WALKOUT_MS) {             // clocking off
      const t = (since - dwell) / WALKOUT_MS;
      const x = lerp(standX, DOOR_X - 6, t), y = lerp(standY, doorY, t);
      const slow = task.status === 'failed';
      const frame = Math.floor(now / (slow ? 240 : 150)) % 2 ? 'sideA' : 'sideB';
      drawAt(x, y, true, () => sprite(frame, seed, now));
      return true;
    }
  }
  return false;
}

// Standing crew live behind the desk: drawn after the monitor, before the desk table.
function crewStanding(i, task, now, walking) {
  const seed = hash(task.id || String(i));
  const { cx, standX, standY } = cellGeom(i);
  if (task.status === 'pending' && !walking) {
    const v = Math.floor(now / 3400 + seed) % 4;
    const name = v === 1 ? 'frontSip' : v === 3 ? 'frontCheerB' : 'frontStand';
    drawAt(standX, standY, false, () => sprite(name, seed, now, { blink: blinkNow(now, seed) }));
    return;
  }
  if (task.status === 'completed' || task.status === 'failed') {
    const since = task.updated ? now - task.updated : 0;
    if (since < dwellFor(task)) {
      if (task.status === 'completed') {
        const frame = Math.floor(now / 260 + seed) % 2 ? 'frontCheerA' : 'frontCheerB';
        drawAt(standX, standY, false, () => sprite(frame, seed, now));
        confetti(cx, DESK_Y - 20, task.id, now);
      } else {
        drawAt(standX, standY, false, () => sprite('frontSlump', seed, now));
      }
    }
  }
}

// The seated worker sits between the desk and the viewer, chair over their hips.
function crewSeated(i, task, now) {
  if (task.status !== 'running') return;
  const seed = hash(task.id || String(i));
  const { cx } = cellGeom(i);
  const chairX = cx - 6, sitY = DESK_Y - 14;
  const onBreak = Math.floor(now / 21000 + seed) % 6 === 5;         // occasional coffee sip
  const frame = onBreak ? 'backSip' : 'backType';
  drawAt(chairX, sitY, false, () => {
    const L = look(seed);
    drawMap(SPRITES[frame], { ...L });
    if (frame === 'backType') {                                     // hammering hands
      const k = Math.floor(now / 130 + seed) % 2;
      px(0, 12, 2, 1, L.K); px(10, 12, 2, 1, L.K);
      px(k ? 0 : 10, 11.4, 2, 1, L.K);
    }
  });
  chairBehind(cx, DESK_Y, seed);
}

function desk(i, task, now, walking) {
  const x0 = LEFT + i * CELL_W;
  const seed = hash(task.id || String(i));
  const color = statusColor(task.status);
  const deskY = DESK_Y, cx = x0 + CELL_W / 2;
  const gone = isGone(task, now);

  serverRack(x0 + 3, deskY - 12, seed, now, gone ? T.faint : color);
  if (i % 2 === 1) plant(x0 + CELL_W - 10, seed);
  monitorUnit(x0, deskY, seed, now, task, gone);
  crewStanding(i, task, now, walking);
  deskTable(x0, deskY);
  crewSeated(i, task, now);
  if (task.status !== 'running') chairBehind(cx, deskY, seed);      // empty chair stays put

  // ---- HUD above the desk ----
  const label = STATUS_LABEL[task.status] || task.status.toUpperCase();
  const blink = task.status === 'running' && Math.floor(now / 600) % 2 === 0;
  px(x0 + 6, 5, 2, 2, blink ? color : (task.status === 'running' ? T.barBg : color));
  text(label, x0 + 10, 4, color, 4);
  const elapsed = task.created ? Math.max(0, Math.floor((now - task.created) / 1000)) : null;
  if (elapsed !== null && (task.status === 'running' || task.status === 'pending')) {
    const e = elapsed >= 3600 ? Math.floor(elapsed / 3600) + 'h' + Math.floor((elapsed % 3600) / 60) + 'm'
            : elapsed >= 60 ? Math.floor(elapsed / 60) + 'm' + (elapsed % 60) + 's' : elapsed + 's';
    text(e, x0 + CELL_W - 6, 4, T.muted, 4, 'right');
  }
  let name = task.name || '';
  if (name.length > 19) name = name.slice(0, 17) + '..';
  text(name, x0 + 6, 11, T.text, 5);

  // progress bar
  const bx = x0 + 6, by = 19, bw = CELL_W - 12;
  px(bx - 1, by - 1, bw + 2, 6, T.barEdge);
  px(bx, by, bw, 4, T.barBg);
  if (task.status === 'running' && task.pct == null) {
    const sw = 12, t = (now / 12) % ((bw - sw) * 2);
    const sx = t < (bw - sw) ? t : (bw - sw) * 2 - t;
    px(bx + sx, by + 1, sw, 2, color);                              // indeterminate scanner
  } else if (task.pct != null) {
    px(bx, by + 1, Math.max(1, bw * task.pct / 100), 2, task.status === 'failed' ? color : T.running);
    text(Math.round(task.pct) + '%', x0 + CELL_W - 6, 26, T.text, 4, 'right');
  }
  const msg = task.status === 'failed' && task.error ? task.error : task.message;
  bubble(cx, 29, msg || (task.status === 'pending' ? 'waiting for a worker...' : ''), now, seed);
}

// ------------------------------------------------------------- tooltip + detail card

const TYPE_LABEL = {
  run_evaluator_and_process: 'Run Evaluator + Process',
  run_release_specsheet_workflow: 'Release Specsheet',
  download_results: 'Download results',
  download_scenarios: 'Download scenarios',
  download_and_eval: 'Download + Eval',
  run_eval_dirs: 'Run eval dirs',
  build_parquet: 'Build parquet',
  generate_summary_csv: 'Generate summary CSV',
  prepare_pr_test_branch: 'Prepare PR Test Branch',
  local_evaluator_debug: 'Local Evaluator Debug',
};
function fmtElapsed(sec) {
  if (sec == null || sec < 0) return '-';
  return sec >= 3600 ? Math.floor(sec / 3600) + 'h' + Math.floor((sec % 3600) / 60) + 'm'
       : sec >= 60 ? Math.floor(sec / 60) + 'm' + (sec % 60) + 's' : sec + 's';
}
function fmtClock(ms) {
  if (!ms) return '-';
  const d = new Date(ms);
  const p = n => (n < 10 ? '0' : '') + n;
  return p(d.getHours()) + ':' + p(d.getMinutes());
}
function wrapLines(str, maxChars, maxLines) {
  const words = String(str).split(/\s+/), lines = [];
  let cur = '';
  for (const w of words) {
    if ((cur + ' ' + w).trim().length > maxChars) {
      if (cur) lines.push(cur);
      cur = w.length > maxChars ? w.slice(0, maxChars - 1) + '-' : w;
      if (lines.length >= maxLines) break;
    } else cur = (cur + ' ' + w).trim();
  }
  if (cur && lines.length < maxLines) lines.push(cur);
  if (lines.length === maxLines && str.length > lines.join(' ').length) {
    lines[maxLines - 1] = lines[maxLines - 1].slice(0, maxChars - 2) + '..';
  }
  return lines;
}

function tooltip(task, now) {
  const status = STATUS_LABEL[task.status] || task.status;
  const pct = task.pct != null ? ' ' + Math.round(task.pct) + '%' : '';
  const elapsed = task.created ? fmtElapsed(Math.floor((now - task.created) / 1000)) : '-';
  const lines = [task.name, status + pct + ' · ' + elapsed];
  const msg = task.status === 'failed' && task.error ? task.error : task.message;
  if (msg) lines.push(...wrapLines(msg, 34, 2));
  lines.push('[ click for details ]');
  ctx.font = 'bold 4px "Courier New", monospace';
  const tw = Math.max(...lines.map(l => Math.ceil(ctx.measureText(l).width))) + 8;
  const th = lines.length * 6 + 5;
  const tx = Math.max(2, Math.min(W - tw - 2, mouse.x + 4));
  const ty = Math.max(2, Math.min(H - th - 2, mouse.y + 6));
  px(tx + 1, ty + 1, tw, th, dark ? 'rgba(0,0,0,0.4)' : 'rgba(15,23,42,0.18)');   // shadow
  px(tx - 1, ty - 1, tw + 2, th + 2, T.outline);
  px(tx, ty, tw, th, T.bubbleBg);
  px(tx, ty, tw, 1, statusColor(task.status));
  lines.forEach((l, i) => {
    const last = i === lines.length - 1;
    text(l, tx + 4, ty + 3 + i * 6, last ? T.muted : (i === 0 ? T.bubbleInk : T.bubbleInk), i === 0 ? 4.5 : 4);
  });
}

function detailCard(task, now) {
  const pw = Math.min(W - 12, 216), lineH = 6.5;
  const rows = [];
  const pct = task.pct != null ? Math.round(task.pct) + '%' : (task.status === 'running' ? 'working...' : '-');
  const elapsed = task.created
    ? fmtElapsed(Math.floor(((task.status === 'completed' || task.status === 'failed') && task.updated
        ? task.updated : now) - task.created) / 1000 | 0) : '-';
  rows.push(['STATUS', (STATUS_LABEL[task.status] || task.status) + '  ' + pct + '  ·  ' + elapsed]);
  rows.push(['TYPE', TYPE_LABEL[task.type] || task.type || '-']);
  if (task.target) rows.push(['TARGET', task.target]);
  if (task.run && task.run !== task.target) rows.push(['RUN', task.run]);
  if (task.by) rows.push(['BY', task.by]);
  rows.push(['START', fmtClock(task.created)]);
  if (task.result) rows.push(['RESULT', task.result.split('/').slice(-2).join('/')]);
  const msg = task.status === 'failed' && task.error ? task.error : task.message;
  const msgLines = msg ? wrapLines(msg, 42, 3) : [];
  const ph = 13 + rows.length * lineH + (msgLines.length ? msgLines.length * 5.5 + 4 : 0) + 8;
  const x = Math.round((W - pw) / 2), y = Math.max(4, Math.round((H - ph) / 2));
  panelRect = [x, y, pw, ph];
  px(x + 2, y + 2, pw, ph, dark ? 'rgba(0,0,0,0.45)' : 'rgba(15,23,42,0.2)');     // shadow
  px(x - 1, y - 1, pw + 2, ph + 2, T.outline);
  px(x, y, pw, ph, T.boardBg);
  px(x, y, pw, 9, statusColor(task.status));                                      // title bar
  let title = task.name || task.id;
  if (title.length > 40) title = title.slice(0, 38) + '..';
  text(title, x + 4, y + 2.4, dark ? '#10141f' : '#ffffff', 5);
  closeRect = [x + pw - 9, y + 1.5, 7, 6];
  px(closeRect[0], closeRect[1], 7, 6, dark ? 'rgba(16,20,31,0.25)' : 'rgba(255,255,255,0.3)');
  text('X', x + pw - 6.5, y + 2.4, dark ? '#10141f' : '#ffffff', 4.5);
  let ty = y + 13;
  for (const [k, v] of rows) {
    text(k, x + 5, ty, T.muted, 4);
    let val = String(v);
    if (val.length > 42) val = val.slice(0, 40) + '..';
    text(val, x + 26, ty, T.boardInk, 4.5);
    ty += lineH;
  }
  if (msgLines.length) {
    px(x + 4, ty + 0.5, pw - 8, 1, T.boardEdge);
    ty += 3.5;
    for (const l of msgLines) { text(l, x + 5, ty, task.status === 'failed' ? T.failed : T.boardInk, 4); ty += 5.5; }
  }
  text('id ' + task.id, x + 5, y + ph - 6, T.faint, 3.5);
}

function overlay(now) {
  closeRect = panelRect = null;
  const selected = selectedId && tasks.find(t => t.id === selectedId);
  const hov = hoverCell();
  if (hov >= 0 && !selected) {                                     // corner brackets
    const x0 = LEFT + hov * CELL_W, c = statusColor(tasks[hov].status);
    const bx = x0 + 2, by = 2, bw = CELL_W - 5, bh = H - 5;
    for (const [cx, cy, dx, dy] of [[bx, by, 1, 1], [bx + bw, by, -1, 1], [bx, by + bh, 1, -1], [bx + bw, by + bh, -1, -1]]) {
      px(cx, cy, dx * 5, 1, c); px(cx, cy, 1, dy * 5, c);
    }
  }
  if (selected) detailCard(selected, now);
  else if (hov >= 0) tooltip(tasks[hov], now);
  canvas.style.cursor = (selected ? inRect(closeRect) || !inRect(panelRect) : hov >= 0) ? 'pointer' : 'default';
}

function wanderer(now) {
  // Off-hours: one worker pacing between the door and the coffee machine.
  const seed = 7;
  const x0 = DOOR_X + 6, x1 = W - RIGHT - 6, span = x1 - x0;
  const t = (now / 45) % (span * 2);
  const x = x0 + (t < span ? t : span * 2 - t);
  const flip = t >= span;
  const atCoffee = !flip && x > x1 - 4;
  const frame = atCoffee ? 'frontSip' : (Math.floor(now / 150) % 2 ? 'sideA' : 'sideB');
  drawAt(x, H - 23, flip && frame !== 'frontSip', () => sprite(frame, seed, now));
}

function draw() {
  const now = Date.now();
  room(now);
  door(now);
  if (!idle) window_(32, 6);                     // the short idle strip has no wall space for it
  whiteboard({
    running: tasks.filter(t => t.status === 'running').length,
    pending: tasks.filter(t => t.status === 'pending').length,
  });
  coffeeMachine(now);
  plant(W - RIGHT + 30, 3);
  if (idle) {
    wanderer(now);
    text('ALL QUIET - NO ACTIVE JOBS', LEFT + (W - LEFT - RIGHT) / 2, 6, T.muted, 5, 'center');
    text('workers are on coffee break', LEFT + (W - LEFT - RIGHT) / 2, 13, T.faint, 4, 'center');
  } else {
    const walking = tasks.map((t, i) => crewWalking(i, t, now));    // corridor pass first
    tasks.forEach((t, i) => desk(i, t, now, walking[i]));
    if (DATA.overflow > 0) text('+' + DATA.overflow + ' more on the task list', W - 4, H - 6, T.muted, 4, 'right');
    overlay(now);
  }
  requestAnimationFrame(draw);
}
base();
draw();
})();
</script>
"""
