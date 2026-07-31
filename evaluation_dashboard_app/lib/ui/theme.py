"""
App-wide design tokens for light and dark Streamlit themes.

Everything that draws chrome (cards, heroes, banners, chips, Plotly figures) should
read colors from here instead of hardcoding hex values, so a single palette swap
keeps the whole dashboard coherent in both themes.

Usage
-----
CSS: call :func:`inject_theme_tokens` once per page (done by
``inject_app_page_styles``), then use ``var(--t4-surface)`` etc. in any injected
CSS or inline ``style=""`` attribute.

Plotly: ``fig.update_layout(**plotly_layout())`` or ``apply_plotly_theme(fig)``.

Python: ``tokens()["text"]`` when a raw value is needed (e.g. Plotly marker lines,
Pandas Styler, PDF export).
"""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

__all__ = [
    "active_theme",
    "is_dark",
    "tokens",
    "token",
    "pick",
    "css_variables",
    "inject_theme_tokens",
    "plotly_layout",
    "apply_plotly_theme",
    "plotly_template",
    "SEQUENTIAL_SCALE",
    "DIVERGING_SCALE",
    "CATEGORICAL",
]

# --------------------------------------------------------------------------------------
# Palettes
# --------------------------------------------------------------------------------------
# Light is the app default (see .streamlit/config.toml). Dark is tuned against
# Streamlit's own dark canvas (#0e1117) so injected surfaces sit *above* the page
# rather than fighting it: surfaces lighten as they come forward, borders stay low
# contrast, and saturated accents are lifted (blue-400/teal-300 instead of -600/-700)
# to keep text contrast >= 4.5:1 on dark fills.

_LIGHT: Dict[str, str] = {
    # canvas + surfaces (ascending elevation)
    "bg": "#ffffff",
    "surface": "#ffffff",
    "surface_2": "#f8fafc",
    "surface_3": "#f1f5f9",
    "surface_sunken": "#f1f5f9",
    "overlay": "rgba(15, 23, 42, 0.04)",
    # borders
    "border": "#e2e8f0",
    "border_strong": "#cbd5e1",
    "border_subtle": "rgba(15, 23, 42, 0.06)",
    # type
    "text": "#0f172a",
    "text_2": "#334155",
    "text_3": "#475569",
    "muted": "#64748b",
    # accents
    "accent": "#1d4ed8",
    "accent_hover": "#1e40af",
    "accent_soft": "#eff6ff",
    "accent_border": "#bfdbfe",
    "accent_on": "#ffffff",
    "accent_2": "#0d9488",
    "accent_2_soft": "#ecfeff",
    "accent_2_border": "#99f6e4",
    # semantics: fg (text/icon), bg (fill), border
    "ok": "#15803d",
    "ok_bg": "#ecfdf5",
    "ok_border": "#a7f3d0",
    "warn": "#b45309",
    "warn_bg": "#fffbeb",
    "warn_border": "#fde68a",
    "bad": "#be123c",
    "bad_bg": "#fff1f2",
    "bad_border": "#fecdd3",
    "info": "#0369a1",
    "info_bg": "#f0f9ff",
    "info_border": "#bae6fd",
    "neutral": "#475569",
    "neutral_bg": "#f1f5f9",
    "neutral_border": "#cbd5e1",
    # composed backgrounds
    "hero_bg": "linear-gradient(135deg, #f8fafc 0%, #ecfeff 45%, #e0f2fe 100%)",
    "card_bg": "linear-gradient(90deg, #f8fafc 0%, #ffffff 100%)",
    "panel_bg": "linear-gradient(180deg, #ffffff 0%, #f8fafc 55%, #f1f5f9 100%)",
    "chip_bg": "#f1f5f9",
    "code_bg": "#f8fafc",
    # Chrome whose light appearance predates the dark theme and is kept verbatim, so
    # light still matches Streamlit's stock light theme (red primary, flat cards).
    "card_bg_accent": "linear-gradient(90deg, #eff6ff 0%, #ffffff 100%)",
    "card_border": "transparent",
    "callout_bg": "linear-gradient(135deg, #f8fafc 0%, #f0f9ff 100%)",
    "badge_bg": "#0f172a",
    "badge_fg": "#ffffff",
    # elevation
    "shadow_sm": "0 1px 2px rgba(15, 23, 42, 0.05)",
    "shadow_md": "0 8px 30px -12px rgba(15, 23, 42, 0.12)",
    "shadow_lg": "0 10px 40px -12px rgba(15, 23, 42, 0.12)",
    # charts
    "chart_bg": "#ffffff",
    "chart_grid": "#e8edf3",
    "chart_axis": "#94a3b8",
    "chart_text": "#334155",
}

_DARK: Dict[str, str] = {
    "bg": "#0e1117",
    "surface": "#161b24",
    "surface_2": "#1b2230",
    "surface_3": "#222b3a",
    "surface_sunken": "#11161f",
    "overlay": "rgba(255, 255, 255, 0.05)",
    "border": "#2a3341",
    "border_strong": "#3b475a",
    "border_subtle": "rgba(255, 255, 255, 0.08)",
    "text": "#e8edf5",
    "text_2": "#cbd5e1",
    "text_3": "#b3c0d1",
    "muted": "#93a1b5",
    "accent": "#60a5fa",
    "accent_hover": "#93c5fd",
    "accent_soft": "rgba(96, 165, 250, 0.14)",
    "accent_border": "rgba(96, 165, 250, 0.38)",
    "accent_on": "#0b1220",
    "accent_2": "#2dd4bf",
    "accent_2_soft": "rgba(45, 212, 191, 0.14)",
    "accent_2_border": "rgba(45, 212, 191, 0.38)",
    "ok": "#4ade80",
    "ok_bg": "rgba(34, 197, 94, 0.14)",
    "ok_border": "rgba(74, 222, 128, 0.36)",
    "warn": "#fbbf24",
    "warn_bg": "rgba(245, 158, 11, 0.15)",
    "warn_border": "rgba(251, 191, 36, 0.36)",
    "bad": "#fb7185",
    "bad_bg": "rgba(244, 63, 94, 0.15)",
    "bad_border": "rgba(251, 113, 133, 0.36)",
    "info": "#7dd3fc",
    "info_bg": "rgba(56, 189, 248, 0.14)",
    "info_border": "rgba(125, 211, 252, 0.34)",
    "neutral": "#cbd5e1",
    "neutral_bg": "rgba(148, 163, 184, 0.14)",
    "neutral_border": "rgba(148, 163, 184, 0.32)",
    "hero_bg": "linear-gradient(135deg, #161b24 0%, #16242c 45%, #14293a 100%)",
    "card_bg": "linear-gradient(90deg, #1b2230 0%, #161b24 100%)",
    "panel_bg": "linear-gradient(180deg, #1b2230 0%, #171d28 55%, #141a23 100%)",
    "chip_bg": "#222b3a",
    "code_bg": "#11161f",
    # Dark counterparts of the light-legacy chrome above: on a dark canvas cards need a
    # real border to separate from the page, and the accent pill carries the theme hue.
    "card_bg_accent": "linear-gradient(90deg, #1b2230 0%, #161b24 100%)",
    "card_border": "#2a3341",
    "callout_bg": "linear-gradient(180deg, #1b2230 0%, #171d28 55%, #141a23 100%)",
    "badge_bg": "#60a5fa",
    "badge_fg": "#0b1220",
    "shadow_sm": "0 1px 2px rgba(0, 0, 0, 0.4)",
    "shadow_md": "0 8px 30px -12px rgba(0, 0, 0, 0.65)",
    "shadow_lg": "0 12px 44px -14px rgba(0, 0, 0, 0.8)",
    "chart_bg": "rgba(0, 0, 0, 0)",
    "chart_grid": "#2a3341",
    "chart_axis": "#64748b",
    "chart_text": "#cbd5e1",
}

# Chart palettes. Categorical hues are shared across themes but lifted in dark so
# they hold up against the deep canvas.
_CATEGORICAL_LIGHT = [
    "#1d4ed8",
    "#0d9488",
    "#b45309",
    "#7c3aed",
    "#be123c",
    "#0369a1",
    "#4d7c0f",
    "#c2410c",
]
_CATEGORICAL_DARK = [
    "#60a5fa",
    "#2dd4bf",
    "#fbbf24",
    "#c4b5fd",
    "#fb7185",
    "#7dd3fc",
    "#a3e635",
    "#fdba74",
]

_SEQUENTIAL_LIGHT = ["#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]
_SEQUENTIAL_DARK = ["#0f1e33", "#1e3a8a", "#2563eb", "#60a5fa", "#bfdbfe"]

_DIVERGING_LIGHT = ["#be123c", "#fda4af", "#f1f5f9", "#7dd3fc", "#0369a1"]
_DIVERGING_DARK = ["#fb7185", "#7f1d3a", "#222b3a", "#155e75", "#7dd3fc"]

_SESSION_KEY = "_t4_theme_override"


# --------------------------------------------------------------------------------------
# Theme detection
# --------------------------------------------------------------------------------------
def active_theme() -> str:
    """
    "light" or "dark" for the current viewer.

    Reads ``st.context.theme.type`` (the browser's effective Streamlit theme, which
    honors both config.toml and the user's Settings toggle). Falls back to "light",
    the app default, when unavailable (older Streamlit, bare script runs, tests).
    """
    try:
        override = st.session_state.get(_SESSION_KEY)
        if override in ("light", "dark"):
            return override
    except Exception:
        pass
    try:
        kind = getattr(getattr(st, "context", None), "theme", None)
        value = getattr(kind, "type", None)
        if value in ("light", "dark"):
            return value
    except Exception:
        pass
    return "light"


def is_dark() -> bool:
    """True when the viewer is on the dark theme."""
    return active_theme() == "dark"


def tokens(theme: Optional[str] = None) -> Dict[str, str]:
    """Raw token values for `theme` (defaults to the active one)."""
    name = theme or active_theme()
    return dict(_DARK if name == "dark" else _LIGHT)


def token(name: str, theme: Optional[str] = None) -> str:
    """Single token value, or "" if the name is unknown."""
    return tokens(theme).get(name, "")


def pick(light, dark):
    """
    Return `dark` on the dark theme, `light` otherwise.

    For values that are deliberately *not* shared between themes — chart palettes and
    colorscales whose light-mode appearance predates the dark theme and is kept as-is.
    Prefer plain tokens for chrome; this is for "light must look exactly like it did".
    """
    return dark if is_dark() else light


def CATEGORICAL(theme: Optional[str] = None) -> list:  # noqa: N802 - palette accessor
    """Categorical series colors, ordered for first-N use."""
    return list(_CATEGORICAL_DARK if (theme or active_theme()) == "dark" else _CATEGORICAL_LIGHT)


def SEQUENTIAL_SCALE(theme: Optional[str] = None) -> list:  # noqa: N802 - palette accessor
    """Low-to-high continuous scale (Plotly ``color_continuous_scale``)."""
    return list(_SEQUENTIAL_DARK if (theme or active_theme()) == "dark" else _SEQUENTIAL_LIGHT)


def DIVERGING_SCALE(theme: Optional[str] = None) -> list:  # noqa: N802 - palette accessor
    """Negative-neutral-positive scale for deltas / regressions."""
    return list(_DIVERGING_DARK if (theme or active_theme()) == "dark" else _DIVERGING_LIGHT)


# --------------------------------------------------------------------------------------
# CSS variables
# --------------------------------------------------------------------------------------
def _css_var_name(key: str) -> str:
    return "--t4-" + key.replace("_", "-")


def css_variables(theme: Optional[str] = None) -> str:
    """The token block body, e.g. ``--t4-bg: #fff; --t4-text: ...;``."""
    return "\n".join(f"  {_css_var_name(k)}: {v};" for k, v in tokens(theme).items())


def inject_theme_tokens() -> None:
    """
    Publish the palette as CSS custom properties plus theme-aware base styling.

    The active palette is written to ``:root`` from Python (authoritative: it follows
    the user's Streamlit theme choice, not just the OS setting). A
    ``prefers-color-scheme`` block repeats the dark values as a safety net for the
    first paint after a theme switch, before Streamlit reruns the script.
    """
    theme = active_theme()
    st.markdown(
        f"""
        <style>
        :root {{
{css_variables(theme)}
          color-scheme: {theme};
        }}
        {'' if theme == "dark" else f'''@media (prefers-color-scheme: dark) {{
          :root:not([data-t4-theme="light"]) {{
{css_variables("dark")}
          }}
        }}'''}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Plotly
# --------------------------------------------------------------------------------------
def plotly_template(theme: Optional[str] = None) -> str:
    """Base Plotly template name for the theme."""
    return "plotly_dark" if (theme or active_theme()) == "dark" else "plotly_white"


def plotly_layout(theme: Optional[str] = None, **overrides) -> Dict[str, object]:
    """
    ``update_layout`` kwargs that make a figure sit correctly on the page.

    Transparent paper in dark mode so figures inherit the Streamlit canvas instead of
    punching a lighter rectangle into it. Pass extra kwargs to override.
    """
    t = tokens(theme)
    layout: Dict[str, object] = {
        "template": plotly_template(theme),
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": t["chart_bg"],
        "font": {"color": t["chart_text"]},
        "xaxis": {"gridcolor": t["chart_grid"], "linecolor": t["chart_axis"], "zerolinecolor": t["chart_grid"]},
        "yaxis": {"gridcolor": t["chart_grid"], "linecolor": t["chart_axis"], "zerolinecolor": t["chart_grid"]},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": t["chart_text"]}},
        "hoverlabel": {
            "bgcolor": t["surface"],
            "bordercolor": t["border_strong"],
            "font": {"color": t["text"]},
        },
    }
    layout.update(overrides)
    return layout


def apply_plotly_theme(fig, theme: Optional[str] = None, **overrides):
    """
    Theme an existing figure in place, preserving axis settings it already has.

    Safe on subplots and on figures with no axes (pie, treemap, indicator).
    """
    t = tokens(theme)
    base = plotly_layout(theme, **overrides)
    axis_style = base.pop("xaxis")
    base.pop("yaxis", None)
    fig.update_layout(**base)
    try:
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
    except Exception:
        pass
    try:
        fig.update_annotations(font_color=t["chart_text"])
    except Exception:
        pass
    return fig
