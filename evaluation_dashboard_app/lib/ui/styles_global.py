"""App-wide Streamlit CSS injected on most pages."""

from __future__ import annotations

import streamlit as st

from lib.ui.theme import inject_theme_tokens


def inject_app_page_styles() -> None:
    """Global polish: metrics, alerts, expanders, buttons, sidebar rhythm."""
    # Publishes the light/dark design tokens (--t4-*) that the rules below and every
    # page-level stylesheet read from.
    inject_theme_tokens()
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        [data-testid="stMetricContainer"] {
            background: var(--t4-surface-2);
            border: 1px solid var(--t4-border);
            border-radius: 10px;
        }
        .stDownloadButton button { border-radius: 10px !important; }
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            border-left-width: 4px !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--t4-border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 0.35rem;
            background: var(--t4-surface-2);
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stMultiSelect label {
            font-weight: 600;
            color: var(--t4-text-2);
        }
        [data-testid="stSidebar"] hr {
            margin: 1rem 0;
            border-color: var(--t4-border);
        }
        div[data-testid="stVerticalBlock"] > div > div[data-testid="stCode"] pre {
            border-radius: 10px !important;
            border: 1px solid var(--t4-border) !important;
        }
        /* Plotly draws its own white modebar/hover surfaces; keep them on-theme.
           The modebar overlaps top-anchored legends, so it needs an opaque backdrop
           rather than a transparent one — otherwise icons and legend text collide. */
        .js-plotly-plot .modebar,
        .js-plotly-plot .modebar-group {
            background: var(--t4-bg) !important;
        }
        .js-plotly-plot .modebar-btn path {
            fill: var(--t4-muted) !important;
        }
        .js-plotly-plot .modebar-btn:hover path {
            fill: var(--t4-text) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # App-wide identity badge (top of the main area). No-ops for local/direct access.
    try:
        from lib.auth import render_identity_badge

        render_identity_badge()
    except Exception:
        pass
    # try:
    #     from lib.deploy_debug import running_in_docker

    #     if not running_in_docker():
    #         st.markdown(
    #             """
    #             <style>
    #             /* 99_Deployment_Debug.py is registered for st.page_link in Docker; hide default nav outside containers. */
    #             section[data-testid="stSidebar"] a[href*="Deployment_Debug"],
    #             section[data-testid="stSidebar"] a[href*="deployment_debug"] {
    #                 display: none !important;
    #             }
    #             </style>
    #             """,
    #             unsafe_allow_html=True,
    #         )
    # except Exception:
    #     pass
