# Theming (light + dark)

The dashboard ships a light theme by default and a hand-tuned dark theme. Users switch
via the ⋮ menu → **Settings** → **Appearance**.

**Light is frozen.** It is Streamlit's stock light theme plus the chrome the app already
had before dark mode existed, and it should stay pixel-identical to that. Dark is the
designed theme. Practically:

- `deploy/.streamlit/config.toml` has **no `[theme.light]` section and no styling key at
  `[theme]` level** — keys there apply to *both* themes, so they all live under
  `[theme.dark]`. The chart palettes (`chartCategoricalColors` and friends) are
  deliberately unset for the same reason.
- When a color must differ per theme *and* light's value predates dark, use
  `pick(light, dark)` from `lib/ui/theme.py` rather than a shared token. Page code keeps
  its original literals in `_LEGACY_*` constants for the light branch.
- Figures that had no explicit theming before get **none** on light:
  `if is_dark(): apply_plotly_theme(fig)`. Applying `plotly_white` plus token colors is
  itself a change to light mode.

Two pieces make this work:

| Piece | Owns |
| --- | --- |
| `deploy/.streamlit/config.toml` | Streamlit's own widgets, sidebar, dataframes, alerts — dark only. Bind-mounted to `/app/.streamlit/` by compose; needs Streamlit ≥ 1.50 for `[theme.dark]`. |
| `lib/ui/theme.py` | Our injected HTML/CSS and Plotly figures — the palette as CSS custom properties and Plotly layout defaults, for both themes. |

Keep the two in sync: the `[theme.dark]` values are mirrored in `_DARK` in
`lib/ui/theme.py`, and `tests/test_theme_tokens.py` fails if they drift.

## Rules for page code

**Never hardcode a hex color for chrome.** Surfaces, borders, text, and shadows come
from tokens. `inject_app_page_styles()` (called at the top of every page) publishes them,
so any `st.markdown(..., unsafe_allow_html=True)` on the page can use `var(--t4-*)`
in a stylesheet or in an inline `style=""`.

```python
st.markdown(
    '<div style="background:var(--t4-surface-2);border:1px solid var(--t4-border);'
    'color:var(--t4-text);border-radius:12px;padding:0.8rem 1rem;">…</div>',
    unsafe_allow_html=True,
)
```

For Plotly:

```python
from lib.ui.theme import apply_plotly_theme, SEQUENTIAL_SCALE

fig = px.bar(df, x="class", y="recall")
apply_plotly_theme(fig)            # instead of plot_bgcolor="#ffffff", template="plotly_white"
```

When a raw value is needed in Python (Plotly marker lines, Pandas `Styler`, an
annotation), read it from the palette instead of typing a hex:

```python
from lib.ui.theme import tokens
t = tokens()
fig.add_hline(y=target, line_color=t["muted"])
```

## Token reference

CSS name is the dict key with `--t4-` prefix and dashes, e.g. `surface_2` →
`var(--t4-surface-2)`.

**Surfaces** (ascending elevation — use higher numbers for things that sit on top)
`bg`, `surface`, `surface_2`, `surface_3`, `surface_sunken`, `overlay`

**Borders** `border`, `border_strong`, `border_subtle`

**Type** `text` (headings/values), `text_2`, `text_3` (body), `muted` (kickers, captions)

**Accents** `accent`, `accent_hover`, `accent_soft` (tinted fill), `accent_border`,
`accent_on` (text on an accent fill), `accent_2` + `accent_2_soft` + `accent_2_border` (teal)

**Semantic** — each has a text/icon color, a fill, and a border:
`ok` / `ok_bg` / `ok_border`, `warn` / …, `bad` / …, `info` / …, `neutral` / …

**Composed backgrounds** `hero_bg`, `card_bg`, `panel_bg`, `chip_bg`, `code_bg`

**Elevation** `shadow_sm`, `shadow_md`, `shadow_lg`

**Charts** `chart_bg`, `chart_grid`, `chart_axis`, `chart_text`, plus the palette
functions `CATEGORICAL()`, `SEQUENTIAL_SCALE()`, `DIVERGING_SCALE()` — these are the
*dark* palette in practice, since light keeps its own `_LEGACY_*` colors via `pick()`

**Light-legacy chrome** `card_bg_accent`, `card_border`, `callout_bg`, `badge_bg`,
`badge_fg` — components whose light appearance is held at its pre-dark-theme value
(flat cards, near-black mode pill) while dark gets a border and the accent hue

## Design notes

- Dark surfaces get **lighter** as they come forward (`surface_sunken` → `surface_3`);
  never use a pure-black fill or a light-mode shadow to fake elevation.
- Semantic fills in dark mode are translucent (`rgba(…, 0.14)`) so they tint the canvas
  instead of punching a bright block into it, and the paired `*` text color is lifted
  (green-400, rose-400, amber-400) to hold ≥ 4.5:1 against them.
- In dark mode Plotly figures use a transparent `paper_bgcolor` so they read as part of
  the page.
- Status meaning must never rest on hue alone — keep the icon/label that goes with it.

## iframes need the tokens injected

`st.components.v1.html(...)` renders in its own iframe, and CSS custom properties do
**not** cross that boundary — a `var(--t4-*)` inside embedded HTML silently resolves to
nothing. Two ways out:

- The embedded document has its own stylesheet: give it its own `:root` block with
  `css_variables()` and then use `var(--t4-*)` inside as usual (see the generated tables
  in `pages/13_Trend_Insights.py`).
- It's a one-off inline color: resolve it in Python with `token("surface_3")` (see the
  viewer placeholder in `lib/t4_three_layers.py`).

## Intentional exceptions

- `lib/overview_pdf_report.py` renders a **printable** report. It stays on the light
  palette regardless of the viewer's theme; do not convert it to tokens.
- `lib/perception_eval_result_summarizer.py` writes matplotlib rasters
  (`object_positions_status.jpg`) from the background worker, where there is no Streamlit
  context at all. Same category: a white-canvas export, left light on purpose.
- Colors that encode *data* rather than chrome — the bbox-viewer legend swatches in
  `lib/ui/bounding_box_viewer_ui.py` must keep matching the hues the JS viewer actually
  draws (`static/bbox_viewer_renderer.js`), so only their surrounding chip is tokenized.
- `static/bbox_viewer.css` / `static/bbox_explorer.css` are standalone viewers with their
  own, dark-first `data-theme` system. They are opened in a new tab, not embedded, so they
  are deliberately independent of `--t4-*`.

## Theme detection

`active_theme()` reads `st.context.theme.type`, which reflects the viewer's effective
Streamlit theme (config default plus their Settings choice). It falls back to `"light"`
outside a script run. Switching theme in Settings triggers a rerun, at which point the
tokens are re-emitted; the injected `prefers-color-scheme` fallback covers the frame
before that.
