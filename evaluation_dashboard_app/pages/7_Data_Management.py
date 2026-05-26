"""
Data Management: list runs, show size, delete runs, copy shareable links.
For multi-user server deployment so users can manage evaluation data.
"""

import io
import re
import urllib.parse
import zipfile
import streamlit as st
from pathlib import Path
from datetime import datetime

from lib.path_utils import (
    get_data_root,
    get_data_root_display,
    list_run_directories,
    get_run_info,
    delete_run,
    format_size,
    resolve_run_subdirectory,
)
from lib.page_chrome import inject_app_page_styles, render_page_hero, section_header

st.set_page_config(
    page_title="Data Management",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_app_page_styles()
render_page_hero(
    kicker="Workspace",
    title="Data management",
    description=(
        "Browse runs under the data root, check Summary/Score/Parquet flags, download those artifacts as one zip when present, "
        "copy shareable Overview links, and delete runs to free disk space."
    ),
    mode="Single Run",
)

data_root = get_data_root()
st.sidebar.info(f"**Data root:** `{get_data_root_display()}/`")

run_dirs = list_run_directories()
if not run_dirs:
    st.warning(f"No runs found under `{get_data_root_display()}/`. Add evaluation runs (e.g. from the Download page) to see them here.")
    st.stop()

# Build run info table
rows = []
for run_path in run_dirs:
    info = get_run_info(run_path)
    rows.append({
        "Run name": info["name"],
        "Size": format_size(info["size_bytes"]),
        "Modified": datetime.fromtimestamp(info["mtime"]).strftime("%Y-%m-%d %H:%M") if info["mtime"] else "—",
        "Summary.csv": "✓" if info["has_summary"] else "—",
        "Score.csv": "✓" if info["has_score"] else "—",
        "Parquet": "✓" if info["has_parquet"] else "—",
    })

section_header("Runs in data root", "Size, modified time, and which artifacts exist per run.")
st.dataframe(rows, width="stretch", hide_index=True)

# Shareable link builder
section_header("Share", "Compose a query string for Overview — append it to your app base URL.")
col_a, col_b = st.columns(2)
with col_a:
    run_names = [r["Run name"] for r in rows]
    share_run_a = st.selectbox("Baseline (A)", run_names, key="share_run_a")
with col_b:
    share_compare = st.checkbox("Compare with another run", key="share_compare")
    share_run_b = None
    if share_compare:
        share_run_b = st.selectbox(
            "Candidate (B)",
            [n for n in run_names if n != share_run_a],
            key="share_run_b",
        )
mode = "compare" if share_compare and share_run_b else "single"
query = {"mode": mode, "run_a": share_run_a}
if mode == "compare":
    query["run_b"] = share_run_b
q = urllib.parse.urlencode(query)
st.code(q, language=None)
st.caption("Example: `https://your-server:8501/?` + the query above.")

# Download generated artifacts (Summary.csv, Score.csv, Parquet) as one zip
section_header(
    "Download",
    "Zip includes Summary.csv, Score.csv, and every `.parquet` in the run folder (only files that exist).",
)
dl_run_name = st.selectbox("Run to download", run_names, key="dl_run_select")
dl_path, dl_err = resolve_run_subdirectory(dl_run_name)
if dl_err:
    st.error(dl_err)
else:
    assert dl_path is not None
    to_zip: list[tuple[Path, str]] = []
    summary_file = dl_path / "Summary.csv"
    score_file = dl_path / "Score.csv"
    if summary_file.is_file():
        to_zip.append((summary_file, "Summary.csv"))
    if score_file.is_file():
        to_zip.append((score_file, "Score.csv"))
    for pq in sorted(dl_path.glob("*.parquet"), key=lambda p: p.name.lower()):
        to_zip.append((pq, pq.name))

    if not to_zip:
        st.info("This run has no Summary.csv, Score.csv, or `.parquet` files at the top level.")
    else:
        st.caption(f"**{len(to_zip)}** file(s) will be included: {', '.join(arc for _, arc in to_zip)}")
        # Invalidate prepared zip when user switches runs (avoid stale download + free memory).
        prepared = st.session_state.get("dm_zip_prepared")
        if prepared and prepared.get("run") != dl_run_name:
            st.session_state.pop("dm_zip_prepared", None)
            prepared = None

        if st.button(
            "Prepare download (build ZIP)",
            key="dm_prepare_zip",
            help="Reads and compresses files on demand so the page stays fast until you need the archive.",
        ):
            buf = io.BytesIO()
            zip_errors: list[str] = []
            included: list[str] = []
            with st.spinner("Building ZIP…"):
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fpath, arcname in to_zip:
                        try:
                            zf.write(fpath, arcname=arcname)
                            included.append(arcname)
                        except OSError as e:
                            zip_errors.append(f"{arcname}: {e}")
            for msg in zip_errors:
                st.warning(msg)
            if included:
                safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", dl_run_name).strip() or "run"
                zip_name = f"{safe_stem}_artifacts.zip"
                st.session_state["dm_zip_prepared"] = {
                    "run": dl_run_name,
                    "data": buf.getvalue(),
                    "file_name": zip_name,
                }
            else:
                st.session_state.pop("dm_zip_prepared", None)
                st.error("Could not add any files to the zip.")

        prepared = st.session_state.get("dm_zip_prepared")
        if prepared and prepared.get("run") == dl_run_name and prepared.get("data"):
            st.download_button(
                label=f"Download {prepared['file_name']}",
                data=prepared["data"],
                file_name=prepared["file_name"],
                mime="application/zip",
                key=f"dm_dl_zip_{dl_run_name}",
            )
            st.caption("ZIP is ready in this session. Change run or refresh the page to discard it.")

# Delete section
section_header("Delete", "Permanent — frees disk space. Use with care on shared servers.")
del_run_name = st.selectbox(
    "Run to delete",
    options=run_names,
    key="del_run_select",
    format_func=lambda x: x,
)
confirm = st.text_input(
    "Type the run name to confirm",
    placeholder=del_run_name,
    key="del_confirm",
)
if st.button("Delete run", type="primary", key="del_btn"):
    if confirm != del_run_name:
        st.error("Confirmation text does not match the run name. Type it exactly to delete.")
    else:
        ok, msg = delete_run(del_run_name)
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)
