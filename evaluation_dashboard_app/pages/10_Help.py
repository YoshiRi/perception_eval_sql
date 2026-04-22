import re
from pathlib import Path

import streamlit as st

from lib.mermaid_render import render_mermaid
from lib.page_chrome import inject_app_page_styles, render_page_hero

st.set_page_config(
    page_title="Help",
    page_icon="❔",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_app_page_styles()
render_page_hero(
    kicker="Documentation",
    title="Help & guide",
    description="In-app copy of the project README with a simple Japanese / English switch.",
    mode="Single Run",
)

# Streamlit markdown does not run Mermaid; split fenced ```mermaid blocks and render via Mermaid.js.
MERMAID_FENCE = re.compile(r"```mermaid\s*\n([\s\S]*?)```", re.IGNORECASE)
IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")
README_FILES = {
    "Japanese": Path("Readme.md"),
    "English": Path("Readme.en.md"),
}


def _render_markdown_with_images(chunk: str) -> None:
    parts = IMAGE_PATTERN.split(chunk)
    i = 0
    while i < len(parts):
        st.markdown(parts[i])
        if i + 2 < len(parts):
            alt_text = parts[i + 1]
            img_path = parts[i + 2]
            img_file = Path(img_path)
            if img_file.exists():
                st.image(str(img_file), caption=alt_text)
            else:
                st.warning(f"Image not found: {img_path}")
            i += 3
        else:
            break


language = st.radio(
    "README language",
    options=list(README_FILES.keys()),
    horizontal=True,
    label_visibility="collapsed",
)

selected_readme_path = README_FILES[language]
if not selected_readme_path.exists():
    st.error(f"README file not found: {selected_readme_path}")
    st.stop()

content = selected_readme_path.read_text(encoding="utf-8")

for idx, piece in enumerate(MERMAID_FENCE.split(content)):
    if idx % 2 == 0:
        _render_markdown_with_images(piece)
    else:
        render_mermaid(piece)
