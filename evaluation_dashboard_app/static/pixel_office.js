/*
  Pixel-art office floor that shows workflow tasks working in realtime.

  Shared by two surfaces:
    - the Streamlit dashboard (lib/ui/pixel_office.py inlines this file into its
      components.html iframe and calls PixelOffice.mount with a prepared payload);
    - the packaged local client (static/client_workflow.html inlines it via the
      /*__PIXEL_OFFICE_JS__* / placeholder and feeds it /api/workflow_tasks rows
      through PixelOffice.fromWorkflowApi).

  One desk per task, one worker per desk. Workers walk in through the door when a
  task is queued, type while it runs, wander to the CAFE machine for coffee breaks,
  celebrate under confetti -- or slump at a red monitor -- and walk back out. Every
  animation is a pure function of wall-clock time, the task's own timestamps, and a
  hash of the task id, so re-mounting (the dashboard reloads the iframe every few
  seconds) never visibly resets the scene.

  API:
    const office = PixelOffice.mount(el, {theme, tasks, overflow});
    office.setData({tasks, overflow, theme});   // cheap; rebuilds only when needed
    office.destroy();
    PixelOffice.fromWorkflowApi(items) -> {tasks, overflow}   // /api/workflow_tasks
*/
(function (global) {
'use strict';

const MAX_DESKS = 10;
const RECENT_FINISH_MS = 75000;                 // finished desks linger this long
const CHEER_MS = 45000, WALKOUT_MS = 14000, WALKIN_MS = 3800;
const BREAK_EVERY = 6, BREAK_LEN_MS = 21000, BREAK_WALK_MS = 3800;
const FLOOR_H = 100, IDLE_H = 68;               // virtual units
// Full layout vs. the compact one used when the full floor would not fit the
// container: narrower cells, door-only left wing, machine-only right corner.
const LAYOUT_FULL = { CELL_W: 92, LEFT: 64, RIGHT: 46 };
const LAYOUT_COMPACT = { CELL_W: 68, LEFT: 30, RIGHT: 28 };

const THEMES = {
  dark: {
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
    outline: '#151823', shadow: 'rgba(0,0,0,0.28)',
    running: '#54c0e8', pending: '#98a1b8', completed: '#5fce7f', failed: '#e06060',
    runningDim: '#2e6f86', screenText: '#e8f4f8',
    glow: 'rgba(255, 214, 120, 0.05)', screenGlow: 'rgba(84, 192, 232, 0.07)',
  },
  light: {
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
    outline: '#3d4557', shadow: 'rgba(15,23,42,0.14)',
    running: '#0e7490', pending: '#64748b', completed: '#15803d', failed: '#dc2626',
    runningDim: '#67b7d1', screenText: '#e8f4f8',
    glow: 'rgba(255, 214, 120, 0.0)', screenGlow: 'rgba(14, 116, 144, 0.0)',
  },
};
const STATUS_LABEL = { running: 'RUNNING', pending: 'QUEUED', completed: 'DONE', failed: 'FAILED' };
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

// ------------------------------------------------------------- characters (12 x 18)
// Sprite maps: H/h hair+shade, K/k skin+shade, E eye, S/s shirt+shade, P trousers,
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
frontWalkA: [                                   // walking toward the viewer
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
'..PP...PP...',
'.BBB....BB..',
],
frontWalkB: [
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
'...PP...PP..',
'..BB....BBB.',
],
backWalkA: [                                    // walking away, into the door
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
'.KK.SsSs.KK.',
'...PPPPPP...',
'...PP..PP...',
'..PP...PP...',
'.BBB....BB..',
'............',
],
backWalkB: [
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
'.KK.SsSs.KK.',
'...PPPPPP...',
'...PP..PP...',
'...PP...PP..',
'..BB....BBB.',
'............',
],
};

const SKIN = ['#efb98d', '#d99c6b', '#a06a42', '#f3ccab', '#8a5433'];
const HAIR = ['#5b3d22', '#2d2622', '#c9973f', '#83848f', '#8a4630', '#3e4a68'];
const SHIRT = ['#3f8f7a', '#c08a3e', '#7a5fae', '#4a7dbd', '#b35c76', '#5c8a4a', '#4aa3a3'];
const PANTS = ['#3a4a6b', '#4a4a52', '#5b4636', '#37536b'];

function hash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function shade(hex, f) {                        // f < 1 darkens, f > 1 lightens
  const n = parseInt(hex.slice(1), 16);
  const ch = v => Math.max(0, Math.min(255, Math.round(v * f)));
  return 'rgb(' + ch(n >>> 16) + ',' + ch((n >>> 8) & 255) + ',' + ch(n & 255) + ')';
}
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
function lerp(a, b, t) { return a + (b - a) * Math.max(0, Math.min(1, t)); }

// ------------------------------------------------------------------------ the scene

function mount(container, opts) {
  opts = opts || {};
  const state = { tasks: [], overflow: 0, theme: 'light' };
  let T = THEMES.light, dark = false;
  let W = 0, H = 0, cells = 0, idle = true, FLOOR_Y = 0, DESK_Y = 0;
  let CELL_W = LAYOUT_FULL.CELL_W, LEFT = LAYOUT_FULL.LEFT, RIGHT = LAYOUT_FULL.RIGHT;
  let compact = false;
  let S = 3;                                     // css px per virtual pixel (see rebuild)
  const DOOR_X = 16;
  let canvas = null, ctx = null;
  const dpr = Math.min(2, (global.devicePixelRatio || 1));
  let raf = 0, destroyed = false;

  const mouse = { x: -1, y: -1, over: false };
  let selectedId = null;
  try { selectedId = sessionStorage.getItem('pxOfficeSel') || null; } catch (e) {}
  let closeRect = null, panelRect = null;

  container.style.overflowX = 'auto';
  container.style.overflowY = 'hidden';

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
    return (mouse.x >= LEFT && i >= 0 && i < state.tasks.length) ? i : -1;
  }
  const inRect = r => r && mouse.x >= r[0] && mouse.x <= r[0] + r[2] && mouse.y >= r[1] && mouse.y <= r[1] + r[3];

  function rebuild() {
    idle = state.tasks.length === 0;
    const avail = (container.clientWidth || (global.document && document.body.clientWidth) || 700);
    // Fit ladder: every desk and the whole room (door to CAFE corner) should be
    // visible at any width. Try the full layout, then the compact one, then keep
    // the compact layout and shrink the pixels; only truly tiny containers scroll.
    const n = state.tasks.length;
    const fits = (lay, sc) => (lay.LEFT + n * lay.CELL_W + lay.RIGHT) * sc <= avail;
    let lay = LAYOUT_FULL, sc = 3;
    if (!idle && !fits(LAYOUT_FULL, 3)) {
      lay = LAYOUT_COMPACT;
      sc = fits(LAYOUT_COMPACT, 3) ? 3 : fits(LAYOUT_COMPACT, 2) ? 2 : 1.5;
    }
    const wantCompact = lay === LAYOUT_COMPACT;
    CELL_W = lay.CELL_W; LEFT = lay.LEFT; RIGHT = lay.RIGHT;
    const minCells = Math.max(2, Math.ceil((avail / sc - LEFT - RIGHT) / CELL_W));
    cells = idle ? minCells : n;
    const w = LEFT + cells * CELL_W + RIGHT, h = idle ? IDLE_H : FLOOR_H;
    if (canvas && w === W && h === H && wantCompact === compact && sc === S) return;
    compact = wantCompact; S = sc;
    W = w; H = h; FLOOR_Y = H - 30; DESK_Y = H - 26;
    if (canvas) canvas.remove();
    canvas = document.createElement('canvas');
    canvas.width = Math.round(W * S * dpr); canvas.height = Math.round(H * S * dpr);
    canvas.style.width = (W * S) + 'px'; canvas.style.height = (H * S) + 'px';
    canvas.style.imageRendering = 'pixelated';
    canvas.style.borderRadius = '8px';
    canvas.style.display = 'block';
    canvas.style.margin = '0 auto';              // center when narrower than the container
    container.appendChild(canvas);
    ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
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
      if (i >= 0) setSelected(state.tasks[i].id);
    });
  }

  function setData(data) {
    data = data || {};
    if (Array.isArray(data.tasks)) state.tasks = data.tasks;
    if (data.overflow !== undefined) state.overflow = Math.max(0, data.overflow | 0);
    if (data.theme) state.theme = data.theme === 'dark' ? 'dark' : 'light';
    dark = state.theme === 'dark';
    T = THEMES[state.theme];
    if (selectedId && !state.tasks.some(t => t.id === selectedId)) setSelected(null);
    rebuild();
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
  const statusColor = s => T[s] || T.pending;
  function groundShadow(x, w, footY) { px(x, footY, w, 1.4, T.shadow); }

  function drawMap(map, legend) {
    const rows = map.length, cols = map[0].length;
    const solid = (r, c) => r >= 0 && c >= 0 && r < rows && c < cols && legend[map[r][c]] !== undefined && map[r][c] !== '.';
    ctx.fillStyle = T.outline;                  // silhouette outline first
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

  function sprite(name, seed, now, spriteOpts) {
    const L = look(seed);
    const legend = Object.assign({}, L);
    if (spriteOpts && spriteOpts.blink) legend.E = L.K;
    drawMap(SPRITES[name], legend);
    if (L.glasses && name.indexOf('front') === 0 && name !== 'frontSlump') {
      px(2.6, 5, 2, 1.4, 'rgba(32,36,46,0.55)'); px(6.6, 5, 2, 1.4, 'rgba(32,36,46,0.55)');
      px(5, 5, 2, 0.6, '#20242e');
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
  function walker(x, y, seed, now, opts) {
    // Walk cycle with a 1px bob on alternate steps -- the bob is what sells it.
    // facing: 'side' (default, mirror with flip), 'front' (out of the door),
    // 'back' (into the door).
    opts = opts || {};
    const period = opts.slow ? 240 : 150;
    const step = Math.floor(now / period) % 2;
    const name = opts.facing === 'front' ? (step ? 'frontWalkA' : 'frontWalkB')
               : opts.facing === 'back' ? (step ? 'backWalkA' : 'backWalkB')
               : (step ? 'sideA' : 'sideB');
    groundShadow(x + 2, 8, y + 17);
    drawAt(x, y - (step ? 1 : 0), !!opts.flip, () => sprite(name, seed, now));
  }

  // ------------------------------------------------------------------- set dressing

  function room(now) {
    base();
    px(0, 0, W, H, T.wall);
    const wainscotY = FLOOR_Y - 22;
    px(0, wainscotY, W, FLOOR_Y - wainscotY, T.wallPanel);         // lower wall band
    px(0, wainscotY, W, 1, T.wallLine);
    for (let i = 0; i <= cells + 1; i++) {                         // wall panel seams
      const x = LEFT + (i - 1) * CELL_W;
      if (x > 4 && x < W - 4) px(x, 2, 1, FLOOR_Y - 4, T.wallLine);
    }
    px(0, FLOOR_Y - 2, W, 2, T.skirt);                             // baseboard
    for (let x = 0; x < W; x += 10) {                              // carpet tiles
      for (let y = FLOOR_Y; y < H; y += 5) {
        const a = ((x / 10 + (y - FLOOR_Y) / 5) % 2);
        px(x, y, 10, 5, a ? T.floorA : T.floorB);
        px(x, y, 10, 1, T.floorEdge);
        px(x, y, 1, 5, a ? T.floorB : T.floorA);
      }
    }
    for (let i = 0; i < cells; i++) {                              // hanging ceiling lamps
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
    px(x - 1, top - 1, 18, 29, T.outline);                         // frame
    px(x, top, 16, 28, dark ? '#4a3a29' : '#a97e52');              // frame wood
    px(x + 1, top + 1, 14, 27, dark ? '#5d4933' : '#c99a68');      // door
    px(x + 3, top + 3, 10, 8, dark ? '#4a3a29' : '#b0824f');       // upper inset
    px(x + 4, top + 4, 8, 6, dark ? '#39485c' : '#d9ecf8');        // window pane
    px(x + 4, top + 4, 8, 1, dark ? '#4c6076' : '#eef7fd');
    px(x + 3, top + 14, 10, 10, dark ? '#4a3a29' : '#b0824f');     // lower inset
    px(x + 4, top + 15, 8, 8, dark ? '#55432f' : '#c08a58');
    px(x + 12, top + 12, 2, 3, dark ? '#c9a24b' : '#8a6a2f');      // handle
    px(x + 2, top - 7, 12, 5, T.outline);                          // EXIT sign
    px(x + 3, top - 6, 10, 3, dark ? '#173322' : '#dcfce7');
    text('EXIT', x + 4, top - 5.8, Math.floor(now / 900) % 4 ? T.completed : shade(T.completed, 0.6), 3.5);
  }

  function window_(x, y) {
    const w = 30, h = 18;
    px(x - 2, y - 2, w + 4, h + 4, T.outline);
    px(x - 1, y - 1, w + 2, h + 2, dark ? '#3a4157' : '#cdd5e2');  // frame
    if (dark) {                                                    // night
      px(x, y, w, h, '#0c1020');
      const hs = hash('sky');
      for (let i = 0; i < 10; i++) px(x + 1 + ((hs >>> i) % (w - 2)), y + 1 + ((hs >>> (i + 3)) % 7), 1, 1, '#aab1c6');
      px(x + w - 7, y + 2, 4, 4, '#e8e4d0'); px(x + w - 7, y + 2, 1, 1, '#0c1020');  // moon
      px(x + 2, y + 11, 5, 7, '#1a2438'); px(x + 9, y + 8, 6, 10, '#1e2a42');
      px(x + 17, y + 12, 5, 6, '#1a2438'); px(x + 24, y + 9, 4, 9, '#1e2a42');
      px(x + 10, y + 9, 1, 1, '#e8c44a'); px(x + 13, y + 11, 1, 1, '#e8c44a');
      px(x + 25, y + 10, 1, 1, '#54c0e8'); px(x + 3, y + 13, 1, 1, '#e8c44a');
    } else {                                                       // day
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
    px(x - 1, y + h + 1, w + 2, 1, T.boardEdge);                   // tray
    px(x + 2, y + h + 2, 1, FLOOR_Y - y - h + 2, T.metal);         // legs
    px(x + w - 3, y + h + 2, 1, FLOOR_Y - y - h + 2, T.metal);
  }

  function coffeeMachine(now, busy) {
    const x = W - RIGHT + 10, y = FLOOR_Y - 22;
    groundShadow(x - 1, 13, y + 22);
    px(x - 1, y - 1, 14, 24, T.outline);
    px(x, y, 12, 22, dark ? '#2b3145' : '#525c74');
    px(x, y, 12, 1, dark ? '#3d4560' : '#6b7690');
    px(x + 1, y + 2, 10, 4, '#12151f');
    text('CAFE', x + 2, y + 2.5, '#e8c44a', 3);
    px(x + 2, y + 8, 3, 2, '#d24545'); px(x + 6, y + 8, 3, 2, '#3a6fc4');   // buttons
    px(x + 3, y + 11, 6, 4, '#12151f');                            // dispenser
    px(x + 5, y + 13, 2, 2, '#eef0f6');                            // cup
    // The machine pours whenever a worker is standing at it, else an occasional drip.
    if (busy || Math.floor(now / 650) % 3 === 0) px(x + 5, y + 11, 1, 2, '#8a6a4a');
    px(x + 2, y + 17, 8, 1, T.metal);                              // tray
    px(x + 10, y + 19, 1, 1, Math.floor(now / 800) % 2 ? '#e06060' : '#5c1f1f');
  }

  function plant(x, seed) {
    const y = FLOOR_Y - 4;                                         // pot base on floor
    const leaf = dark ? '#2e6b3f' : '#3f8f57', leafHi = dark ? '#3f8f57' : '#5cb371';
    groundShadow(x - 1, 8, y + 4.6);
    px(x + 1, y - 9, 1, 5, dark ? '#1f4a2c' : '#2e6b3f');          // stems
    px(x + 4, y - 10, 1, 6, dark ? '#1f4a2c' : '#2e6b3f');
    px(x - 1, y - 12, 4, 4, leaf); px(x, y - 13, 2, 2, leafHi);
    px(x + 3, y - 14, 4, 4, leaf); px(x + 4, y - 15, 2, 2, leafHi);
    px(x + 1, y - 9, 4, 3, leaf); px(x + 2, y - 10, 2, 2, leafHi);
    px(x - 1, y - 5, 8, 2, dark ? '#7a3b2e' : '#b0644f');          // pot rim
    px(x, y - 3, 6, 3, dark ? '#5c2c22' : '#9c5843');
    px(x + 1, y - 3, 1, 3, dark ? '#7a3b2e' : '#b0644f');
  }

  function serverRack(x, y, seed, now, color) {
    groundShadow(x - 1, 12, y + 27);
    px(x - 1, y - 1, 12, 28, T.outline);
    px(x, y, 10, 26, T.rack);
    px(x, y, 10, 1, T.rackHi);
    for (let i = 0; i < 5; i++) {
      const sy = y + 2 + i * 5;
      px(x + 1, sy, 8, 3, T.rackSlot);
      px(x + 1, sy, 8, 1, T.rackHi);
      px(x + 2, sy + 1, 3, 1, dark ? '#161a26' : '#7c879c');       // vents
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
    if (dark && task.status === 'running' && !gone) {              // soft screen glow
      ctx.fillStyle = T.screenGlow;
      ctx.fillRect(mx - 4, my - 3, mw + 8, mh + 7);
    }
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
    px(cx - 1, my + mh + 1, 2, 2, T.metal);                        // stand
    px(cx - 4, deskY - 1, 8, 1, T.metal);
  }

  function deskTable(x0, deskY) {
    const cx = x0 + CELL_W / 2;
    groundShadow(x0 + 12, CELL_W - 26, deskY + 15);
    const dx = x0 + 18, dyTop = deskY + 3;                         // drawer pedestal
    px(dx - 1, dyTop, 12, 13, T.outline);
    px(dx, dyTop, 10, 12, T.woodFront);
    for (let i = 0; i < 3; i++) {
      px(dx + 1, dyTop + 1 + i * 4, 8, 3, T.wood);
      px(dx + 1, dyTop + 1 + i * 4, 8, 1, T.woodTop);
      px(dx + 4, dyTop + 2 + i * 4, 2, 1, T.woodDark);             // handle
    }
    const dw = CELL_W - 26;                                        // desk top + apron
    px(x0 + 12, deskY - 1, dw, 1, T.outline);
    px(x0 + 12, deskY, dw, 2, T.woodTop);
    px(x0 + 12, deskY + 2, dw, 2, T.wood);
    px(x0 + 12, deskY + 4, 2, 11, T.woodDark);                     // legs
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
    px(cx - 1, by + 7, 2, 4, T.metal);                             // gas lift
    px(cx - 5, by + 11, 10, 1, T.metal);                           // base
    px(cx - 5, by + 12, 2, 1, T.outline); px(cx + 3, by + 12, 2, 1, T.outline);
    px(cx - 1, by + 12, 2, 1, T.outline);
  }

  function bubble(cx, topY, msg, now, seed) {
    if (!msg) return;
    const maxChars = compact ? 16 : 26;
    let show = msg;
    if (msg.length > maxChars) {                                   // deterministic marquee
      const loop = msg + '   ';
      const off = Math.floor(now / 260 + seed) % loop.length;
      show = (loop + loop).slice(off, off + maxChars);
    }
    ctx.font = 'bold 4px "Courier New", monospace';
    const w = Math.min(CELL_W - 8, Math.ceil(ctx.measureText(show).width) + 5);
    const x = Math.max(2, Math.min(W - w - 2, cx - w / 2));
    px(x - 1, topY - 1, w + 2, 9, T.bubbleEdge);
    px(x, topY, w, 7, T.bubbleBg);
    px(cx - 1, topY + 8, 2, 2, T.bubbleBg);                        // tail
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

  // ----------------------------------------------------------------- the desk + crew

  const cellGeom = i => {
    const x0 = LEFT + i * CELL_W, cx = x0 + CELL_W / 2;
    return { x0, cx, standX: cx + 14, standY: DESK_Y - 15 };
  };
  const dwellFor = task => task.status === 'failed' ? CHEER_MS * 0.9 : CHEER_MS;
  const isGone = (task, now) => (task.status === 'completed' || task.status === 'failed')
    && task.updated && now - task.updated > dwellFor(task) + WALKOUT_MS;

  // Coffee run: every worker takes one BREAK_LEN_MS break out of BREAK_EVERY windows,
  // walking the corridor to the CAFE machine, sipping there, and walking back.
  function breakPhase(seed, now) {
    if (Math.floor(now / BREAK_LEN_MS + seed) % BREAK_EVERY !== BREAK_EVERY - 1) return null;
    const tIn = now % BREAK_LEN_MS;
    if (tIn < BREAK_WALK_MS) return { phase: 'go', t: tIn / BREAK_WALK_MS };
    if (tIn >= BREAK_LEN_MS - BREAK_WALK_MS) {
      return { phase: 'back', t: (BREAK_LEN_MS - tIn) / BREAK_WALK_MS };
    }
    return { phase: 'at', t: 0 };
  }
  const machineSpot = seed => W - RIGHT - 4 - (seed % 3) * 10;

  // Walkers use the corridor behind the desks, so they are drawn before any desk.
  function crewWalking(i, task, now) {
    const seed = hash(task.id || String(i));
    const { cx, standX, standY } = cellGeom(i);
    const doorY = FLOOR_Y - 16;                  // corridor: feet on the floor line
    if (task.status === 'pending') {
      const age = task.created ? now - task.created : WALKIN_MS;
      if (age < WALKIN_MS) {
        const t = age / WALKIN_MS;
        if (t < 0.22) {                          // stepping out of the doorway
          walker(DOOR_X - 6, lerp(doorY - 7, doorY, t / 0.22), seed, now, { facing: 'front' });
        } else {
          const tt = (t - 0.22) / 0.78;
          walker(lerp(DOOR_X - 6, standX, tt), lerp(doorY, standY, tt), seed, now, {});
        }
        return true;
      }
      return false;
    }
    if (task.status === 'running') {
      const brk = breakPhase(seed, now);
      if (!brk) return false;
      const from = cx - 6, to = machineSpot(seed);
      if (brk.phase === 'go') walker(lerp(from, to, brk.t), doorY, seed, now, {});
      else if (brk.phase === 'back') walker(lerp(from, to, brk.t), doorY, seed, now, { flip: true });
      else {
        groundShadow(to + 2, 8, doorY + 17.6);
        drawAt(to, doorY, false, () => sprite('frontSip', seed, now, { blink: blinkNow(now, seed) }));
      }
      return true;
    }
    if (task.status === 'completed' || task.status === 'failed') {
      const since = task.updated ? now - task.updated : 0;
      const dwell = dwellFor(task);
      if (since >= dwell && since < dwell + WALKOUT_MS) {          // clocking off
        const t = (since - dwell) / WALKOUT_MS;
        const slow = task.status === 'failed';
        if (t < 0.8) {
          const tt = t / 0.8;
          walker(lerp(standX, DOOR_X - 6, tt), lerp(standY, doorY, tt), seed, now,
                 { flip: true, slow });
        } else {                                 // through the doorway, away from us
          walker(DOOR_X - 6, lerp(doorY, doorY - 7, (t - 0.8) / 0.2), seed, now,
                 { facing: 'back', slow });
        }
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
          const f = Math.floor(now / 260 + seed) % 2;
          // Little victory hop: frame A lifts the whole body a pixel or two.
          drawAt(standX, standY - (f ? 2 : 0), false, () => sprite(f ? 'frontCheerA' : 'frontCheerB', seed, now));
          confetti(cx, DESK_Y - 20, task.id, now);
        } else {
          drawAt(standX, standY, false, () => sprite('frontSlump', seed, now));
          const sigh = (now + seed * 700) % 4200;                  // a sad little puff
          if (sigh < 1100) {
            const sy = standY - 1 - sigh / 300;
            px(standX + 11, sy, 1, 1, T.muted);
            if (sigh > 400) px(standX + 12.4, sy - 1.4, 1, 1, T.faint);
          }
        }
      }
    }
  }

  // The seated worker sits between the desk and the viewer, chair over their hips.
  function crewSeated(i, task, now) {
    if (task.status !== 'running') return;
    const seed = hash(task.id || String(i));
    const { cx } = cellGeom(i);
    if (breakPhase(seed, now)) {                                   // out for coffee
      chairBehind(cx, DESK_Y, seed);
      return;
    }
    const chairX = cx - 6, sitY = DESK_Y - 14;
    drawAt(chairX, sitY, false, () => {
      const L = look(seed);
      drawMap(SPRITES.backType, Object.assign({}, L));
      const k = Math.floor(now / 130 + seed) % 2;                  // hammering hands
      px(0, 12, 2, 1, L.K); px(10, 12, 2, 1, L.K);
      px(k ? 0 : 10, 11.4, 2, 1, L.K);
    });
    chairBehind(cx, DESK_Y, seed);
  }

  function desk(i, task, now, walking) {
    const x0 = LEFT + i * CELL_W;
    const seed = hash(task.id || String(i));
    const color = statusColor(task.status);
    const deskY = DESK_Y, cx = x0 + CELL_W / 2;
    const gone = isGone(task, now);

    if (!compact) {
      serverRack(x0 + 3, deskY - 12, seed, now, gone ? T.faint : color);
      if (i % 2 === 1) plant(x0 + CELL_W - 10, seed);
    }
    monitorUnit(x0, deskY, seed, now, task, gone);
    crewStanding(i, task, now, walking);
    deskTable(x0, deskY);
    crewSeated(i, task, now);
    if (task.status !== 'running') chairBehind(cx, deskY, seed);   // empty chair stays put

    // ---- HUD above the desk ----
    const label = STATUS_LABEL[task.status] || task.status.toUpperCase();
    const blink = task.status === 'running' && Math.floor(now / 600) % 2 === 0;
    px(x0 + 6, 5, 2, 2, blink ? color : (task.status === 'running' ? T.barBg : color));
    text(label, x0 + 10, 4, color, 4);
    const elapsed = task.created ? Math.max(0, Math.floor((now - task.created) / 1000)) : null;
    if (elapsed !== null && (task.status === 'running' || task.status === 'pending')) {
      text(fmtElapsed(elapsed), x0 + CELL_W - 6, 4, T.muted, 4, 'right');
    }
    const maxName = compact ? 13 : 19;
    let name = task.name || '';
    if (name.length > maxName) name = name.slice(0, maxName - 2) + '..';
    text(name, x0 + 6, 11, T.text, 5);

    // progress bar
    const bx = x0 + 6, by = 19, bw = CELL_W - 12;
    px(bx - 1, by - 1, bw + 2, 6, T.barEdge);
    px(bx, by, bw, 4, T.barBg);
    if (task.status === 'running' && task.pct == null) {
      const sw = 12, t = (now / 12) % ((bw - sw) * 2);
      const sx = t < (bw - sw) ? t : (bw - sw) * 2 - t;
      px(bx + sx, by + 1, sw, 2, color);                           // indeterminate scanner
    } else if (task.pct != null) {
      const fill = Math.max(1, bw * task.pct / 100);
      px(bx, by + 1, fill, 2, task.status === 'failed' ? color : T.running);
      if (task.status === 'running') {                             // moving shine
        const gx = (now / 30) % (fill + 8) - 4;
        if (gx > 0 && gx < fill - 2) px(bx + gx, by + 1, 2, 1, shade(T.running, 1.6));
      }
      text(Math.round(task.pct) + '%', x0 + CELL_W - 6, 26, T.text, 4, 'right');
    }
    const msg = task.status === 'failed' && task.error ? task.error : task.message;
    bubble(cx, 29, msg || (task.status === 'pending' ? 'waiting for a worker...' : ''), now, seed);
    if ((task.status === 'completed' || task.status === 'failed') && task.updated) {
      const since = now - task.updated;
      if (since >= 0 && since < 2600) toast(x0, task, since);
    }
  }

  function toast(x0, task, since) {
    // A short "it just finished" banner: zooms in over the desk, holds, fades.
    const ok = task.status === 'completed';
    const color = ok ? T.completed : T.failed;
    const ease = 1 - Math.pow(1 - Math.min(1, since / 260), 2);
    const fullW = Math.min(CELL_W - 16, 42), h = 11;
    const w = Math.max(6, Math.round(fullW * ease));
    const cx = x0 + CELL_W / 2, y = 38;
    ctx.globalAlpha = since > 1900 ? Math.max(0, 1 - (since - 1900) / 700) : 1;
    px(cx - w / 2 + 1, y + 1, w, h, T.shadow);
    px(cx - w / 2 - 1, y - 1, w + 2, h + 2, T.outline);
    px(cx - w / 2, y, w, h, color);
    if (ease > 0.75) {
      text(ok ? 'DONE!' : 'FAILED', cx, y + 2.8, dark ? '#10141f' : '#ffffff', 6, 'center');
      if (ok) {                                  // twinkling sparks beside the banner
        const tw = Math.floor(since / 160) % 2;
        px(cx - w / 2 - 4, y + (tw ? 1 : 6), 2, 2, '#e8c44a');
        px(cx + w / 2 + 2, y + (tw ? 7 : 2), 2, 2, '#e8c44a');
      }
    }
    ctx.globalAlpha = 1;
  }

  function hiresQueue(now) {
    // Overflow tasks queue up outside the door, waiting for a desk to open.
    const n = Math.min(compact ? 2 : 4, state.overflow);
    if (n <= 0) return;
    for (let i = 0; i < n; i++) {
      const seed = hash('hire:' + i);
      const x = 4 + i * 11, y = H - 21;
      const v = Math.floor(now / 2900 + seed) % 3;
      groundShadow(x + 2, 8, y + 17.6);
      drawAt(x, y, false, () => sprite(v === 1 ? 'frontSip' : 'frontStand', seed, now, { blink: blinkNow(now, seed) }));
    }
    const label = '+' + state.overflow + ' waiting';
    px(3, H - 27, label.length * 2.6 + 4, 7, T.bubbleEdge);
    px(3.6, H - 26.4, label.length * 2.6 + 2.8, 5.8, T.bubbleBg);
    text(label, 5.6, H - 25, T.bubbleInk, 4);
  }

  // ------------------------------------------------------------ tooltip + detail card

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
    if (msg) lines.push.apply(lines, wrapLines(msg, 34, 2));
    lines.push('[ click for details ]');
    ctx.font = 'bold 4px "Courier New", monospace';
    const tw = Math.max.apply(null, lines.map(l => Math.ceil(ctx.measureText(l).width))) + 8;
    const th = lines.length * 6 + 5;
    const tx = Math.max(2, Math.min(W - tw - 2, mouse.x + 4));
    const ty = Math.max(2, Math.min(H - th - 2, mouse.y + 6));
    px(tx + 1, ty + 1, tw, th, dark ? 'rgba(0,0,0,0.4)' : 'rgba(15,23,42,0.18)');   // shadow
    px(tx - 1, ty - 1, tw + 2, th + 2, T.outline);
    px(tx, ty, tw, th, T.bubbleBg);
    px(tx, ty, tw, 1, statusColor(task.status));
    lines.forEach((l, i) => {
      const last = i === lines.length - 1;
      text(l, tx + 4, ty + 3 + i * 6, last ? T.muted : T.bubbleInk, i === 0 ? 4.5 : 4);
    });
  }

  function detailCard(task, now) {
    const pw = Math.min(W - 12, 216), lineH = 6.5;
    const rows = [];
    const pct = task.pct != null ? Math.round(task.pct) + '%' : (task.status === 'running' ? 'working...' : '-');
    const endMs = (task.status === 'completed' || task.status === 'failed') && task.updated ? task.updated : now;
    const elapsed = task.created ? fmtElapsed(Math.floor((endMs - task.created) / 1000)) : '-';
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
    for (const kv of rows) {
      text(kv[0], x + 5, ty, T.muted, 4);
      let val = String(kv[1]);
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
    const selected = selectedId && state.tasks.find(t => t.id === selectedId);
    const hov = hoverCell();
    if (hov >= 0 && !selected) {                                   // corner brackets
      const x0 = LEFT + hov * CELL_W, c = statusColor(state.tasks[hov].status);
      const bx = x0 + 2, by = 2, bw = CELL_W - 5, bh = H - 5;
      const corners = [[bx, by, 1, 1], [bx + bw, by, -1, 1], [bx, by + bh, 1, -1], [bx + bw, by + bh, -1, -1]];
      for (const co of corners) { px(co[0], co[1], co[2] * 5, 1, c); px(co[0], co[1], 1, co[3] * 5, c); }
    }
    if (selected) detailCard(selected, now);
    else if (hov >= 0) tooltip(state.tasks[hov], now);
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
    if (atCoffee) {
      groundShadow(x + 2, 8, H - 3.4);
      drawAt(x, H - 21, false, () => sprite('frontSip', seed, now));
    } else {
      walker(x, H - 21, seed, now, { flip });
    }
  }

  function anyoneAtMachine(now) {
    return state.tasks.some((t, i) => {
      if (t.status !== 'running') return false;
      const brk = breakPhase(hash(t.id || String(i)), now);
      return brk && brk.phase === 'at';
    });
  }

  function draw() {
    const now = Date.now();
    room(now);
    door(now);
    if (!idle && !compact) window_(32, 6);       // no wall space when idle or compact
    if (!compact) whiteboard({
      running: state.tasks.filter(t => t.status === 'running').length,
      pending: state.tasks.filter(t => t.status === 'pending').length,
    });
    coffeeMachine(now, anyoneAtMachine(now));
    if (!compact) plant(W - RIGHT + 30, 3);
    if (idle) {
      wanderer(now);
      text('ALL QUIET - NO ACTIVE JOBS', LEFT + (W - LEFT - RIGHT) / 2, 6, T.muted, 5, 'center');
      text('workers are on coffee break', LEFT + (W - LEFT - RIGHT) / 2, 13, T.faint, 4, 'center');
    } else {
      const walking = state.tasks.map((t, i) => crewWalking(i, t, now));  // corridor pass first
      state.tasks.forEach((t, i) => desk(i, t, now, walking[i]));
      hiresQueue(now);
      if (state.overflow > 0) text('+' + state.overflow + ' more on the task list', W - 4, H - 6, T.muted, 4, 'right');
      overlay(now);
    }
  }

  function frame() {
    if (destroyed) return;
    draw();
    raf = requestAnimationFrame(frame);
  }

  setData({ tasks: opts.tasks || [], overflow: opts.overflow || 0, theme: opts.theme || 'light' });
  frame();
  let resizeObserver = null;
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => { if (!destroyed) rebuild(); });
    resizeObserver.observe(container);
  }
  return {
    setData,
    destroy() {
      destroyed = true;
      if (resizeObserver) resizeObserver.disconnect();
      if (raf) cancelAnimationFrame(raf);
      if (canvas) canvas.remove();
      canvas = null;
    },
  };
}

// ------------------------------------------------- adapter for /api/workflow_tasks

function fromWorkflowApi(items, nowMs) {
  const now = nowMs || Date.now();
  const parse = v => {
    let s = String(v || '');
    if (!s) return null;
    // DB timestamps come back naive-UTC; without a zone JS would read them as local.
    if (!/[zZ]|[+-]\d\d:?\d\d$/.test(s)) s += 'Z';
    const t = Date.parse(s);
    return isNaN(t) ? null : t;
  };
  const order = { running: 0, pending: 1, completed: 2, failed: 2 };
  const rows = [];
  (items || []).forEach(it => {
    const status = String(it.status || '');
    const updated = parse(it.updated_at);
    if (status === 'completed' || status === 'failed') {
      if (updated == null || now - updated > RECENT_FINISH_MS) return;
    } else if (status !== 'pending' && status !== 'running') return;
    let pct = it.progress_pct;
    pct = (pct === null || pct === undefined || isNaN(Number(pct)))
      ? null : Math.max(0, Math.min(100, Number(pct)));
    rows.push({
      id: String(it.id || ''), status: status,
      name: String(it.run_name || it.target_name || it.workflow_kind || it.type || 'task'),
      type: String(it.type || ''), pct: pct,
      message: String(it.progress_message || ''), error: String(it.error_message || ''),
      created: parse(it.created_at), updated: updated,
      run: String(it.run_name || ''), target: String(it.target_name || ''),
      by: String(it.requested_by || ''), result: String(it.result_path || ''),
    });
  });
  rows.sort((a, b) => ((order[a.status] !== undefined ? order[a.status] : 3)
                     - (order[b.status] !== undefined ? order[b.status] : 3))
                     || ((b.created || 0) - (a.created || 0)));
  return { tasks: rows.slice(0, MAX_DESKS), overflow: Math.max(0, rows.length - MAX_DESKS) };
}

global.PixelOffice = { mount: mount, fromWorkflowApi: fromWorkflowApi, MAX_DESKS: MAX_DESKS };
})(typeof window !== 'undefined' ? window : this);
