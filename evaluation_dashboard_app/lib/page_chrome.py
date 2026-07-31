"""
Shared Streamlit chrome: hero header, loaded-data cards, section titles, global styles.
Use across dashboard pages for consistent UX.
"""

from __future__ import annotations

import html
from typing import List, Optional, Sequence, Tuple

import streamlit as st

from lib.ui.styles_global import inject_app_page_styles
from lib.ui.theme import CATEGORICAL, pick

# Left accent colors for multi-run cards (indigo, teal, blue, amber, violet, rose).
# Kept verbatim for light, which predates the dark theme.
_LEGACY_MULTI_ACCENTS: List[str] = [
    "#312e81",
    "#0f766e",
    "#1d4ed8",
    "#b45309",
    "#7c3aed",
    "#be123c",
]


def _multi_accents() -> List[str]:
    """Left accent colors for multi-run cards. On dark they come from the shared
    categorical palette, so a run keeps the same identity color in its card and in the
    charts below it; on light the original hues are preserved."""
    return pick(_LEGACY_MULTI_ACCENTS, CATEGORICAL())


def render_share_link_callout(query_string: str, *, caption: Optional[str] = None) -> None:
    """
    Highlight a URL query fragment for sharing (e.g. Overview / TLR).
    query_string: without leading '?', e.g. 'mode=single&run_a=foo'
    """
    q = html.escape(query_string.strip().lstrip("?"))
    cap = (
        caption
        or "Append this to your app URL so others open the same view (see Data Management for examples)."
    )
    st.markdown(
        f"""
        <div style="
            background: var(--t4-callout-bg);
            border: 1px solid var(--t4-border-strong);
            border-radius: 14px;
            padding: 0.9rem 1.1rem;
            margin: 0.5rem 0 0.25rem 0;
        ">
          <div style="font-size:0.68rem;letter-spacing:0.14em;color:var(--t4-muted);text-transform:uppercase;font-weight:700;">
            Shareable link
          </div>
          <div style="margin-top:0.45rem;font-family:ui-monospace,monospace;font-size:0.84rem;color:var(--t4-text);word-break:break-all;line-height:1.45;">
            ?{q}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(cap)


def render_loaded_data_section(entries: Sequence[Tuple[str, str]]) -> None:
    """
    Display loaded run path(s) as cards.

    entries: (label, path_display_string) in order, e.g. [("Baseline · A", "/data/run1"), ...].
    """
    if not entries:
        return
    st.markdown(
        '<p style="margin:0 0 0.5rem 0;font-size:0.7rem;letter-spacing:0.12em;color:var(--t4-muted);text-transform:uppercase;font-weight:700;">Loaded data</p>',
        unsafe_allow_html=True,
    )
    n = len(entries)
    safe: List[Tuple[str, str]] = [(html.escape(la), html.escape(pa)) for la, pa in entries]

    if n == 1:
        la, pa = safe[0]
        st.markdown(
            f"""
            <div style="border-radius:14px;border:1px solid var(--t4-card-border);border-left:5px solid var(--t4-accent);background:var(--t4-card-bg-accent);padding:0.95rem 1.1rem;">
              <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--t4-muted);font-weight:700;">{la}</div>
              <div style="margin-top:0.35rem;font-size:0.86rem;color:var(--t4-text);word-break:break-word;line-height:1.4;font-weight:650;">{pa}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Two or more runs: one markdown per column. A single st.markdown with multiple
    # sibling divs is unreliable in Streamlit (markdown pass often leaves only the
    # first block as HTML; the rest appears as raw markup).
    cols = st.columns(n)
    accents = _multi_accents()
    for i, col in enumerate(cols):
        la, pa = safe[i]
        acc = accents[i % len(accents)]
        with col:
            st.markdown(
                f"""
                <div style="border-radius:14px;border:1px solid var(--t4-card-border);border-left:5px solid {acc};background:var(--t4-card-bg);padding:0.95rem 1.1rem;min-height:4.5rem;">
                  <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--t4-muted);font-weight:700;">{la}</div>
                  <div style="margin-top:0.35rem;font-size:0.86rem;color:var(--t4-text);word-break:break-word;line-height:1.4;font-weight:650;">{pa}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_page_hero(
    *,
    kicker: str,
    title: str,
    description: str,
    mode: str = "Single Run",
    secondary_badge_inner_html: Optional[str] = None,
) -> None:
    """
    Large gradient hero for the top of the main panel.

    kicker / title / description: plain text (HTML-escaped).
    secondary_badge_inner_html: optional inner HTML for a second pill (trusted caller only).
    """
    badge = "Compare runs" if mode == "Compare Mode" else "Single run"
    k, t, d = html.escape(kicker), html.escape(title), html.escape(description)
    second = ""
    has_secondary = bool(secondary_badge_inner_html)
    if has_secondary:
        second = (
            f'<span style="background:var(--t4-surface);border:1px solid var(--t4-border-strong);color:var(--t4-text-2);padding:0.4rem 0.9rem;'
            f'border-radius:10px;font-size:0.82rem;font-weight:600;">{secondary_badge_inner_html}</span>'
        )
    else:
        second = ("<!-- no secondary -->")
    # Conditionally include the secondary badge <span> and only render HTML if present
    # Also, avoid rendering an empty {second} or an extra </div>
    st.markdown(
        f"""
        <div style="
            background: var(--t4-hero-bg);
            border: 1px solid var(--t4-border-strong);
            border-radius: 18px;
            padding: 1.35rem 1.6rem;
            margin-bottom: 1.1rem;
            box-shadow: var(--t4-shadow-lg);
        ">
          <div style="display:flex;flex-wrap:wrap;align-items:flex-start;justify-content:space-between;gap:1rem;">
            <div style="flex:1;min-width:220px;">
              <div style="font-size:0.72rem;letter-spacing:0.14em;color:var(--t4-muted);text-transform:uppercase;font-weight:700;">
                {k}
              </div>
              <h1 style="margin:0.35rem 0 0 0;font-size:clamp(1.45rem, 2.5vw, 1.95rem);font-weight:800;color:var(--t4-text);letter-spacing:-0.03em;line-height:1.15;">
                {t}
              </h1>
              <p style="margin:0.55rem 0 0 0;color:var(--t4-text-3);font-size:0.96rem;max-width:40rem;line-height:1.5;">
                {d}
              </p>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.45rem;">
              <span style="background:var(--t4-badge-bg);color:var(--t4-badge-fg);padding:0.4rem 1rem;border-radius:999px;font-size:0.78rem;font-weight:700;letter-spacing:0.04em;">
                {html.escape(badge)}
              </span>
              {second }
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, description: Optional[str] = None) -> None:
    """Section title with teal left rule + optional Streamlit caption."""
    st.markdown(
        f"""
        <div style="margin:1.5rem 0 0.5rem 0;padding:0 0 0 0.65rem;border-left:4px solid var(--t4-accent-2);">
          <div style="font-size:1.12rem;font-weight:800;color:var(--t4-text);letter-spacing:-0.02em;">{html.escape(title)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if description:
        st.caption(description)
