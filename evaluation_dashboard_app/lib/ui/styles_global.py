"""App-wide Streamlit CSS injected on most pages."""

from __future__ import annotations

import streamlit as st


def inject_app_page_styles() -> None:
    """Global polish: metrics, alerts, expanders, buttons, sidebar rhythm."""
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        [data-testid="stMetricContainer"] {
            background: rgba(248, 250, 252, 0.9);
            border: 1px solid #e8edf3;
            border-radius: 10px;
        }
        .stDownloadButton button { border-radius: 10px !important; }
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            border-left-width: 4px !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 0.35rem;
            background: #fafbfc;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stMultiSelect label {
            font-weight: 600;
            color: #334155;
        }
        [data-testid="stSidebar"] hr {
            margin: 1rem 0;
            border-color: #e2e8f0;
        }
        div[data-testid="stVerticalBlock"] > div > div[data-testid="stCode"] pre {
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
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
