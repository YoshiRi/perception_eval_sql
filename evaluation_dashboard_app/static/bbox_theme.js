/**
 * Shared light/dark theme for the bbox explorer and viewer.
 *
 * CSS reads tokens from :root[data-theme="..."]; canvas renderers cannot, so the
 * same palette is mirrored here. `TH.c(name)` returns a solid color and
 * `TH.a(name, alpha)` an alpha variant, so renderer code stays literal-free.
 */
(function () {
  const STORAGE_KEY = "bboxTheme";
  const THEMES = ["dark", "light", "blueprint"];
  // Which structural rules apply: light-family themes need opaque surfaces, since
  // the dark theme's translucent panels lose all separation over a pale ground.
  const MODES = {dark: "dark", light: "light", blueprint: "light"};
  // Shown on the cycle button; the glyph names the theme you get by clicking.
  const LABELS = {dark: "Dark", light: "Paper", blueprint: "Blueprint"};
  const GLYPHS = {dark: "\u263e", light: "\u2600", blueprint: "\u25a6"};

  // Triplets are "r g b" so alpha variants compose without re-parsing.
  const PALETTE = {
    dark: {
      bg: "5 8 18",
      bgDeep: "2 6 23",
      deep: "2 6 23",
      surface: "15 23 42",
      panel: "8 13 28",
      panel2: "13 20 38",
      line: "148 163 184",
      lineStrong: "226 232 240",
      text: "234 242 255",
      textBright: "234 246 255",
      muted: "145 164 191",
      mutedBright: "203 213 225",
      accent: "56 189 248",
      accentDeep: "8 145 178",
      accentSoft: "8 47 73",
      accentFg: "186 230 253",
      accentBright: "125 211 252",
      btnFg: "221 247 255",
      onAccent: "7 17 31",
      marker: "255 255 255",
      good: "52 211 153",
      bad: "251 113 133",
      warn: "251 191 36",
      goodBg: "6 78 59",
      badBg: "127 29 29",
      badBgDeep: "69 10 10",
      warnBg: "120 53 15",
      neutralBg: "51 65 85",
      goodFg: "187 247 208",
      badFg: "254 205 211",
      warnFg: "253 230 138",
      neutralFg: "203 213 225",
      runA: "96 165 250",
      runB: "167 139 250",
      shadow: "0 0 0",
      shadowAlpha: 1,
      alphaGain: 1,
      // Sequential heat scale (green -> red) as HSL saturation/lightness bounds.
      heatSat: 85,
      heatLight: [42, 60],
      // 3D scene / bbox semantics (viewer).
      sceneBg1: "15 23 42",
      sceneBg2: "11 17 32",
      sceneBg3: "2 6 23",
      gtTp: "0 204 102",
      gtFn: "255 153 51",
      gtOther: "75 208 141",
      estTp: "102 179 255",
      estFp: "255 102 102",
      errHigh: "239 68 68",
      errMid: "249 115 22",
      errLow: "250 204 21",
      halo: "250 204 21",
      neutralGt: "100 116 139",
      neutralEst: "148 163 184",
      cat: ["#38bdf8", "#fb7185", "#fbbf24", "#34d399", "#a78bfa", "#f97316", "#e879f9", "#94a3b8"],
      gridAlpha: 0.08
    },
    // Warm paper / ink palette with a coral accent.
    light: {
      bg: "249 247 242",
      bgDeep: "240 237 229",
      deep: "232 228 217",
      surface: "255 255 255",
      panel: "255 255 255",
      panel2: "248 246 240",
      line: "42 37 30",
      lineStrong: "26 23 18",
      text: "31 29 25",
      textBright: "23 21 18",
      muted: "91 88 81",
      mutedBright: "66 62 55",
      accent: "205 102 71",
      accentDeep: "183 82 51",
      accentSoft: "249 230 220",
      accentFg: "145 70 40",
      accentBright: "173 81 49",
      btnFg: "123 53 30",
      onAccent: "255 255 255",
      marker: "27 25 22",
      good: "58 122 84",
      bad: "185 68 53",
      warn: "163 113 12",
      goodBg: "122 190 145",
      badBg: "226 133 116",
      badBgDeep: "210 104 86",
      warnBg: "232 186 88",
      neutralBg: "188 182 168",
      goodFg: "40 92 61",
      badFg: "141 45 33",
      warnFg: "120 84 5",
      neutralFg: "74 71 64",
      runA: "61 110 143",
      runB: "122 94 168",
      // Black shadows at the dark theme's opacity are far too heavy on paper.
      shadow: "0 0 0",
      shadowAlpha: 0.34,
      // Canvas tints were tuned to sit on near-black; the same alpha over paper
      // washes out, so light mode scales every alpha up a little.
      alphaGain: 1.28,
      // Neon heat colors vanish on paper; darker and less saturated instead.
      heatSat: 58,
      heatLight: [40, 30],
      // Scene ground goes to paper; bbox hues are darkened so they keep contrast
      // against it while staying recognizable as the same GT/EST/FP/FN coding.
      sceneBg1: "249 247 242",
      sceneBg2: "240 237 229",
      sceneBg3: "232 228 217",
      gtTp: "23 122 74",
      gtFn: "194 106 16",
      gtOther: "43 138 92",
      estTp: "40 104 173",
      estFp: "192 57 43",
      errHigh: "184 50 39",
      errMid: "194 96 15",
      errLow: "168 134 11",
      halo: "160 108 8",
      neutralGt: "120 116 108",
      neutralEst: "140 136 127",
      // Desaturated so eight series stay distinguishable on paper.
      cat: ["#c2603a", "#3d6e8f", "#4a7c59", "#a3710c", "#7a5ea8", "#a8443a", "#2f7e7a", "#6b6862"],
      gridAlpha: 0.08
    },
    /**
     * Cyanotype / diazo drafting print.
     *
     * Grounded in the subject rather than a general "light mode": the app draws BEV
     * plan views, range rings and box wireframes, which is what an engineering
     * drawing is. Pale cyanotype ground, Prussian-blue ink, and a saturated
     * drafting teal held back for interaction only -- the sole saturated hue, so
     * selection reads as instrument state instead of decoration.
     */
    blueprint: {
      bg: "231 237 241",
      bgDeep: "221 229 235",
      deep: "207 217 226",
      surface: "247 250 252",
      panel: "247 250 252",
      panel2: "238 243 247",
      line: "27 58 82",
      lineStrong: "15 39 57",
      text: "18 41 61",
      textBright: "10 27 41",
      muted: "78 110 134",
      mutedBright: "59 90 112",
      accent: "11 114 133",
      accentDeep: "11 114 133",
      accentSoft: "218 234 240",
      accentFg: "7 90 107",
      accentBright: "14 140 163",
      btnFg: "6 80 95",
      onAccent: "255 255 255",
      marker: "10 27 41",
      good: "46 125 91",
      bad: "179 38 30",
      warn: "138 106 0",
      goodBg: "127 190 158",
      badBg: "227 169 163",
      badBgDeep: "217 139 132",
      warnBg: "222 193 118",
      neutralBg: "184 198 209",
      goodFg: "31 93 66",
      badFg: "140 29 22",
      warnFg: "107 82 0",
      neutralFg: "62 85 104",
      runA: "28 110 140",
      runB: "91 91 166",
      // Cool and tighter than paper: drafting prints sit flat, not lifted.
      shadow: "12 32 48",
      shadowAlpha: 0.26,
      alphaGain: 1.28,
      heatSat: 52,
      heatLight: [38, 26],
      sceneBg1: "231 237 241",
      sceneBg2: "221 229 235",
      sceneBg3: "207 217 226",
      // Same GT/EST/FP/FN coding, stepped for a cyanotype ground.
      gtTp: "23 110 84",
      gtFn: "166 96 12",
      estTp: "28 94 150",
      estFp: "170 46 36",
      gtOther: "40 128 100",
      errHigh: "164 44 34",
      errMid: "172 88 14",
      errLow: "140 112 10",
      halo: "11 114 133",
      neutralGt: "104 128 148",
      neutralEst: "124 148 168",
      // Ink-and-plate hues: one saturated teal, the rest drafting pigments.
      cat: ["#0b7285", "#b3261e", "#8a6a00", "#2e7d5b", "#5b5ba6", "#a64b8b", "#1c6e8c", "#4e6e86"],
      // Drafting paper shows its grid; the other themes keep it nearly invisible.
      gridAlpha: 0.13,
      gridMajorAlpha: 0.3
    }
  };

  function normalize(name) {
    return THEMES.includes(name) ? name : null;
  }

  function stored() {
    try {
      return normalize(window.localStorage.getItem(STORAGE_KEY));
    } catch (err) {
      return null;
    }
  }

  function fromQuery() {
    try {
      return normalize(new URLSearchParams(window.location.search).get("theme"));
    } catch (err) {
      return null;
    }
  }

  function fromSystem() {
    try {
      return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    } catch (err) {
      return "dark";
    }
  }

  // Query param wins (explorer -> viewer iframe), then the saved choice, then the OS.
  let current = fromQuery() || stored() || fromSystem();
  let explicit = Boolean(fromQuery() || stored());
  const listeners = new Set();

  function palette() {
    return PALETTE[current] || PALETTE.dark;
  }

  function trip(name) {
    const p = palette();
    return p[name] || p.text;
  }

  const TH = {
    THEMES,
    get current() {
      return current;
    },
    /** Solid color for a palette token. */
    c(name) {
      return `rgb(${trip(name)})`;
    },
    /** Alpha variant of a palette token. */
    a(name, alpha) {
      const gain = palette().alphaGain || 1;
      const v = Math.max(0, Math.min(1, Number(alpha) * gain));
      return `rgb(${trip(name)} / ${v})`;
    },
    /** Scalar token (e.g. grid alpha), with a caller-supplied fallback. */
    num(name, fallback) {
      const p = palette();
      return name in p ? Number(p[name]) : Number(fallback);
    },
    /** Shadow black, scaled down in light mode so paper surfaces stay soft. */
    shadow(alpha) {
      const p = palette();
      return `rgb(${p.shadow} / ${Math.max(0, Math.min(1, Number(alpha) * p.shadowAlpha))})`;
    },
    /** Sequential heat color for a 0..1 score (0 = calm green, 1 = hot red). */
    heat(t) {
      const p = palette();
      const v = Math.max(0, Math.min(1, Number(t) || 0));
      const [lo, hi] = p.heatLight;
      return `hsl(${145 - v * 120}, ${p.heatSat}%, ${lo + v * (hi - lo)}%)`;
    },
    /** Categorical series color, wrapping around the palette. */
    cat(i) {
      const list = palette().cat;
      return list[((Number(i) || 0) % list.length + list.length) % list.length];
    },
    /** Full categorical palette (first `n` entries when given). */
    cats(n) {
      const list = palette().cat;
      return n ? list.slice(0, n) : list.slice();
    },
    isLight() {
      return current === "light";
    },
    set(name, opts) {
      const next = normalize(name);
      if (!next) return current;
      const persist = !opts || opts.persist !== false;
      current = next;
      if (persist) {
        explicit = true;
        try {
          window.localStorage.setItem(STORAGE_KEY, next);
        } catch (err) {
          /* private mode: theme stays session-only */
        }
      }
      apply();
      return current;
    },
    /** Advance to the next theme in THEMES order (dark -> paper -> blueprint). */
    toggle() {
      const i = THEMES.indexOf(current);
      return TH.set(THEMES[(i + 1) % THEMES.length]);
    },
    /** Human-readable name, for tooltips and status text. */
    label(name) {
      return LABELS[name || current] || String(name || current);
    },
    /** "light" or "dark" -- which structural rules this theme wants. */
    mode() {
      return MODES[current] || "dark";
    },
    /** Register a repaint callback; canvases must redraw on theme change. */
    onChange(fn) {
      if (typeof fn === "function") listeners.add(fn);
      return () => listeners.delete(fn);
    },
    /** Wire a button as a theme toggle and keep its glyph/label in sync. */
    bindToggle(el) {
      if (!el) return;
      const sync = () => {
        const i = THEMES.indexOf(current);
        const next = THEMES[(i + 1) % THEMES.length];
        el.textContent = GLYPHS[next] || "\u25d0";
        el.title = `${LABELS[current]} theme \u2014 switch to ${LABELS[next]}`;
        el.setAttribute("aria-label", el.title);
      };
      el.addEventListener("click", () => TH.toggle());
      TH.onChange(sync);
      sync();
    }
  };

  function apply() {
    const root = document.documentElement;
    root.setAttribute("data-theme", current);
    // Structural rules key off the mode, so a new light-family theme inherits them.
    root.setAttribute("data-mode", MODES[current] || "dark");
    root.style.colorScheme = MODES[current] || "dark";
    listeners.forEach(fn => {
      try {
        fn(current);
      } catch (err) {
        console.error("theme listener failed", err);
      }
    });
    // Keep an embedded viewer in step with the explorer.
    document.querySelectorAll("iframe").forEach(frame => {
      try {
        frame.contentWindow.postMessage({type: "bbox-theme", theme: current}, "*");
      } catch (err) {
        /* cross-origin or not loaded yet */
      }
    });
  }

  window.addEventListener("message", ev => {
    const data = ev && ev.data;
    if (data && data.type === "bbox-theme" && normalize(data.theme)) {
      TH.set(data.theme, {persist: false});
    }
  });

  try {
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", ev => {
      if (!explicit) TH.set(ev.matches ? "light" : "dark", {persist: false});
    });
  } catch (err) {
    /* older browsers: no live OS updates */
  }

  window.BBoxTheme = TH;
  window.TH = TH;
  apply();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply, {once: true});
  }
})();
