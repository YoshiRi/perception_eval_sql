"""
Parquet Debug page: inspect format and contents of parquet, PKL, and result.json files.
"""
import duckdb
import json
import pickle
import re
import streamlit as st
import pandas as pd
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from lib.page_chrome import inject_app_page_styles, render_page_hero

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    layout="wide",
    page_title="Parquet & PKL Debug",
    page_icon="🔬",
    initial_sidebar_state="expanded",
)
inject_app_page_styles()
render_page_hero(
    kicker="Developer tools",
    title="Parquet, PKL & result.json inspector",
    description=(
        "Inspect evaluation artifacts: schema, keys, criteria status, and optional plots — "
        "for debugging pipeline output without leaving the dashboard."
    ),
    mode="Single Run",
)

# =============================
# DuckDB
# =============================
def get_con():
    return duckdb.connect()

# =============================
# PKL helpers
# =============================
def _type_summary(obj: Any) -> str:
    """One-line summary: type and len/shape."""
    t = type(obj).__name__
    if isinstance(obj, pd.DataFrame):
        return f"{t} {obj.shape}"
    if hasattr(obj, "shape"):
        return f"{t} shape={getattr(obj, 'shape', '?')}"
    if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes)):
        try:
            return f"{t} len={len(obj)}"
        except Exception:
            return t
    return t


def _describe_pkl(obj: Any, depth: int = 0, max_depth: int = 5) -> List[str]:
    """Describe type and structure of a PKL object recursively (full text)."""
    lines = []
    indent = "  " * depth
    if depth > max_depth:
        lines.append(f"{indent}... (max depth)")
        return lines
    t = type(obj).__name__
    if isinstance(obj, pd.DataFrame):
        lines.append(f"{indent}{t} shape={obj.shape}, columns={list(obj.columns)}")
        return lines
    if hasattr(obj, "shape"):
        lines.append(f"{indent}{t} shape={getattr(obj, 'shape', '?')}")
        return lines
    if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes)):
        try:
            n = len(obj)
        except Exception:
            n = "?"
        lines.append(f"{indent}{t} len={n}")
    else:
        lines.append(f"{indent}{t}")

    if isinstance(obj, dict):
        for k, v in list(obj.items())[:15]:
            lines.append(f"{indent}  [{repr(k)}]:")
            lines.extend(_describe_pkl(v, depth + 2, max_depth))
        if len(obj) > 15:
            lines.append(f"{indent}  ... and {len(obj) - 15} more keys")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:6]):
            lines.append(f"{indent}  [{i}]:")
            lines.extend(_describe_pkl(v, depth + 2, max_depth))
        if len(obj) > 6:
            lines.append(f"{indent}  ... and {len(obj) - 6} more items")
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for k, v in list(obj.__dict__.items())[:12]:
            lines.append(f"{indent}  .{k}:")
            lines.extend(_describe_pkl(v, depth + 2, max_depth))
        if len(obj.__dict__) > 12:
            lines.append(f"{indent}  ... and {len(obj.__dict__) - 12} more attrs")
    return lines


def _summary_rows(obj: Any, prefix: str = "") -> List[Tuple[str, str, str]]:
    """List of (attribute_name, type_name, brief) for tables. No deep recursion."""
    rows: List[Tuple[str, str, str]] = []
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:30]:
            name = f"{prefix}[{repr(k)}]" if prefix else repr(k)
            rows.append((name, type(v).__name__, _type_summary(v)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:20]):
            name = f"{prefix}[{i}]" if prefix else f"[{i}]"
            rows.append((name, type(v).__name__, _type_summary(v)))
        if len(obj) > 20:
            rows.append(("...", "", f"+ {len(obj) - 20} more items"))
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for k, v in list(obj.__dict__.items()):
            name = f".{k}" if not prefix else f"{prefix}.{k}"
            rows.append((name, type(v).__name__, _type_summary(v)))
    return rows


def _to_preview_value(obj: Any, max_depth: int = 3) -> Any:
    """Convert object to JSON-serializable preview with actual values (no object addresses)."""
    if max_depth <= 0:
        return _type_summary(obj)
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj[:500] + ("…" if len(obj) > 500 else "")
    if isinstance(obj, bytes):
        return repr(obj)[:200]
    # numpy and pandas scalars
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return {"_type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    except ImportError:
        pass
    if isinstance(obj, pd.DataFrame):
        return {"_type": "DataFrame", "shape": list(obj.shape), "columns": list(obj.columns)}
    if isinstance(obj, dict):
        return {repr(k): _to_preview_value(v, max_depth - 1) for k, v in list(obj.items())[:20]}
    if isinstance(obj, (list, tuple)):
        preview = [_to_preview_value(v, max_depth - 1) for v in obj[:10]]
        if len(obj) > 10:
            preview.append(f"... +{len(obj) - 10} more")
        return preview
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {
            k: _to_preview_value(v, max_depth - 1)
            for k, v in list(obj.__dict__.items())
        }
    # Enum: show value for readability
    if hasattr(obj, "name") and hasattr(obj, "value"):
        return str(obj.value)
    s = repr(obj)
    if s.startswith("<") and " at 0x" in s:
        return _type_summary(obj)  # avoid useless object-address repr
    return s[:300] + ("…" if len(s) > 300 else "")


def _render_tree_expander(obj: Any, label: str, depth: int, max_depth: int, key_prefix: str) -> None:
    """Render one level as expander; children inside it."""
    if depth > max_depth:
        st.caption(f"… {_type_summary(obj)}")
        return
    summary = _type_summary(obj)
    with st.expander(f"**{label}** — {summary}", expanded=(depth < 2)):
        if isinstance(obj, pd.DataFrame):
            st.dataframe(obj.head(20), width='stretch')
        elif isinstance(obj, dict):
            for k, v in list(obj.items())[:15]:
                _render_tree_expander(v, f"[{repr(k)}]", depth + 1, max_depth, f"{key_prefix}_{repr(k)}")
            if len(obj) > 15:
                st.caption(f"… and {len(obj) - 15} more keys")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj[:8]):
                _render_tree_expander(v, f"[{i}]", depth + 1, max_depth, f"{key_prefix}_{i}")
            if len(obj) > 8:
                st.caption(f"… and {len(obj) - 8} more items")
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            for k, v in list(obj.__dict__.items()):
                _render_tree_expander(v, f".{k}", depth + 1, max_depth, f"{key_prefix}_{k}")
        else:
            st.text(repr(obj)[:2000] + ("…" if len(repr(obj)) > 2000 else ""))


def load_pkl_file(path: str) -> Any:
    """Load .pkl with pickle, .pkl.z with joblib."""
    path_lower = path.lower()
    if path_lower.endswith(".pkl.z"):
        try:
            import joblib
            return joblib.load(path)
        except ImportError:
            raise ImportError("joblib is required for .pkl.z: pip install joblib")
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================
# result.json (TLR / JSONL) helpers
# =============================
def debug_result_json_lines(file_path: str, first_n: int = 5, last_n: int = 5) -> dict:
    """
    Debug a result.json file (JSONL: one JSON object per line).
    Returns dict with: total_lines, first_lines_detail, last_lines_finalscore, errors.
    """
    out = {
        "total_lines": 0,
        "first_lines_detail": [],
        "last_lines_finalscore": [],
        "errors": [],
    }
    if not os.path.exists(file_path):
        out["errors"].append(f"File not found: {file_path}")
        return out
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line for line in f.readlines() if line.strip()]
    except Exception as e:
        out["errors"].append(str(e))
        return out
    out["total_lines"] = len(lines)
    # First N lines: keys, Frame, criteria*, criteria6/PassFail/Info, Result/Summary
    for i, line in enumerate(lines[:first_n]):
        detail = {"line_index": i + 1, "keys": [], "frame_keys": [], "criteria_keys": [], "criteria6": None, "result_summary_preview": None}
        try:
            data = json.loads(line.strip())
            detail["keys"] = list(data.keys())
            if "Frame" in data:
                frame_keys = list(data["Frame"].keys())
                detail["frame_keys"] = frame_keys
                detail["criteria_keys"] = [k for k in frame_keys if k.startswith("criteria")]
                if "criteria6" in data["Frame"]:
                    criteria6 = data["Frame"]["criteria6"]
                    detail["criteria6"] = criteria6
                    if isinstance(criteria6, dict) and "PassFail" in criteria6:
                        detail["pass_fail"] = criteria6["PassFail"]
                        if "Info" in criteria6["PassFail"]:
                            detail["pass_fail_info"] = criteria6["PassFail"]["Info"]
            if "Result" in data:
                result = data["Result"]
                detail["result_keys"] = list(result.keys())
                if "Summary" in result:
                    s = result["Summary"]
                    detail["result_summary_preview"] = (s[:200] + "...") if isinstance(s, str) and len(s) > 200 else s
        except json.JSONDecodeError as e:
            detail["decode_error"] = str(e)
        out["first_lines_detail"].append(detail)
    # Last N lines: look for FinalScore
    for i, line in enumerate(lines[-last_n:]):
        idx = len(lines) - last_n + i + 1
        try:
            data = json.loads(line.strip())
            if "Frame" in data and "FinalScore" in data["Frame"]:
                final_score = data["Frame"]["FinalScore"]
                out["last_lines_finalscore"].append({
                    "line_index": idx,
                    "FinalScore_keys": list(final_score.keys()) if isinstance(final_score, dict) else [],
                    "FinalScore": final_score,
                })
                if isinstance(final_score, dict) and "Score" in final_score:
                    score = final_score["Score"]
                    out["last_lines_finalscore"][-1]["Score_keys"] = list(score.keys()) if isinstance(score, dict) else []
                    if isinstance(score, dict) and "TP" in score:
                        out["last_lines_finalscore"][-1]["TP_keys"] = list(score["TP"].keys()) if isinstance(score["TP"], dict) else []
        except json.JSONDecodeError:
            pass
    return out


def _criteria_status(val: Any) -> str:
    """Classify Frame.criteria_N value: NoGTNoObj | Success | Fail."""
    if not isinstance(val, dict):
        return "—"
    if "NoGTNoObj" in val:
        return "NoGTNoObj"
    if "PassFail" in val:
        pf = val["PassFail"]
        if isinstance(pf, dict) and "Result" in pf:
            res = pf["Result"]
            if isinstance(res, dict):
                frame_res = res.get("Frame", res.get("Total", ""))
                if frame_res == "Success":
                    return "Success"
        return "Fail"
    return "—"


def parse_result_json_for_viz(
    file_path: str,
    max_frames: Optional[int] = 2000,
) -> Dict[str, Any]:
    """
    Parse full result.json (JSONL) for visualization.
    Returns: condition (list of criterion dicts), frames (list of frame records),
             final_score (dict or None), total_lines, errors.
    """
    out = {
        "condition": [],
        "condition_criterion": [],
        "frames": [],
        "final_score": None,
        "total_lines": 0,
        "errors": [],
    }
    if not os.path.exists(file_path):
        out["errors"].append(f"File not found: {file_path}")
        return out
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line for line in f.readlines() if line.strip()]
    except Exception as e:
        out["errors"].append(str(e))
        return out
    out["total_lines"] = len(lines)
    # First line: Condition
    if lines:
        try:
            first = json.loads(lines[0].strip())
            if "Condition" in first and "Criterion" in first["Condition"]:
                out["condition"] = first["Condition"]
                out["condition_criterion"] = first["Condition"]["Criterion"]
        except json.JSONDecodeError:
            pass
    # Frame lines (Result + Frame with criteria_*)
    for i, line in enumerate(lines[1:], start=2):  # 1-based line index
        if max_frames and len(out["frames"]) >= max_frames:
            break
        try:
            data = json.loads(line.strip())
            rec = {
                "line_index": i,
                "success": None,
                "summary": "",
                "frame_name": "",
                "criteria_status": {},
            }
            if "Result" in data:
                r = data["Result"]
                rec["success"] = r.get("Success")
                rec["summary"] = r.get("Summary") or ""
            if "Frame" in data and data["Frame"]:
                frame = data["Frame"]
                rec["frame_name"] = frame.get("FrameName", "")
                for k, v in frame.items():
                    if k.startswith("criteria_") and isinstance(v, dict):
                        rec["criteria_status"][k] = _criteria_status(v)
            out["frames"].append(rec)
        except json.JSONDecodeError:
            pass
    # Last line: FinalScore
    if len(lines) >= 2:
        try:
            last = json.loads(lines[-1].strip())
            if "Frame" in last and "FinalScore" in last["Frame"]:
                out["final_score"] = last["Frame"]["FinalScore"]
        except json.JSONDecodeError:
            pass
    return out


# =============================
# Sidebar – file type and selection
# =============================
with st.sidebar:
    st.header("File selection")
    file_type = st.radio("File type", ["Parquet", "PKL", "result.json"], key="debug_file_type")
    use_custom = st.checkbox("Use custom file path", value=False, key="pq_debug_custom")

    if file_type == "Parquet":
        parquet_files = sorted(str(p) for p in pathlib.Path("data").rglob("*.parquet"))
        if use_custom:
            target_file = st.text_input(
                "Parquet file path",
                value="data/example.parquet",
                key="pq_debug_path",
                help="Absolute or relative path to a .parquet file",
            )
            if not target_file or not target_file.strip():
                st.warning("Enter a file path")
                st.stop()
            target_file = target_file.strip()
        else:
            if not parquet_files:
                st.error("No parquet files found in data/")
                st.stop()
            target_file = st.selectbox("Parquet file", parquet_files, key="pq_debug_file")
    elif file_type == "PKL":
        pkl_files = sorted(str(p) for p in pathlib.Path("data").rglob("*.pkl"))
        pklz_files = sorted(str(p) for p in pathlib.Path("data").rglob("*.pkl.z"))
        all_pkl = pkl_files + [p for p in pklz_files if p not in pkl_files]
        if use_custom:
            target_file = st.text_input(
                "PKL file path",
                value="data/example/scene_result.pkl",
                key="pkl_debug_path",
                help="Absolute or relative path to a .pkl or .pkl.z file",
            )
            if not target_file or not target_file.strip():
                st.warning("Enter a file path")
                st.stop()
            target_file = target_file.strip()
        else:
            if not all_pkl:
                st.error("No .pkl or .pkl.z files found in data/")
                st.stop()
            target_file = st.selectbox("PKL file", all_pkl, key="pkl_debug_file")
    else:
        # result.json: scan current tree or custom path
        result_json_files = sorted(str(p) for p in pathlib.Path(".").rglob("result.json"))
        if use_custom:
            target_file = st.text_input(
                "result.json file path",
                value="data/result/scenario_01/result.json",
                key="result_json_path",
                help="Path to a result.json file (JSONL format)",
            )
            if not target_file or not target_file.strip():
                st.warning("Enter a file path")
                st.stop()
            target_file = target_file.strip()
        else:
            if not result_json_files:
                st.info("No result.json found. Use custom path or add files under current tree.")
                target_file = st.text_input(
                    "result.json path",
                    value="data/result/scenario_01/result.json",
                    key="result_json_path_fallback",
                )
                if not target_file or not target_file.strip():
                    st.stop()
                target_file = target_file.strip()
            else:
                target_file = st.selectbox("result.json file", result_json_files, key="result_json_file")

    if not os.path.isfile(target_file):
        st.error(f"File not found: {target_file}")
        st.stop()

# =============================
# File metadata (size, etc.)
# =============================
def get_file_info(path: str) -> dict:
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    return {"path": path, "size_bytes": size}

file_info = get_file_info(target_file)

# =============================
# result.json VIEW (TLR / JSONL debug)
# =============================
if file_type == "result.json":
    st.subheader("File info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Path", target_file)
    with col2:
        if file_info["size_bytes"] is not None:
            size_kb = file_info["size_bytes"] / 1024
            st.metric("Size", f"{size_kb:.2f} KB" if size_kb >= 1 else f"{file_info['size_bytes']} B")
        else:
            st.metric("Size", "—")

    first_n = st.slider("First N lines to inspect", 1, 20, 5, key="result_json_first_n")
    last_n = st.slider("Last N lines to scan for FinalScore", 1, 20, 5, key="result_json_last_n")
    debug_info = debug_result_json_lines(target_file, first_n=first_n, last_n=last_n)

    if debug_info["errors"]:
        for err in debug_info["errors"]:
            st.error(err)
        st.stop()

    st.metric("Total lines (non-empty)", debug_info["total_lines"])

    # ---------- Visualization ----------
    st.subheader("Visualization")
    max_frames_viz = st.slider("Max frames to load for charts", 100, 5000, 500, key="result_json_max_frames")
    viz = parse_result_json_for_viz(target_file, max_frames=max_frames_viz)
    if viz["errors"]:
        st.caption("Parse for viz: " + "; ".join(viz["errors"]))
    else:
        # Condition table (criteria definition from first line)
        if viz["condition_criterion"]:
            st.markdown("**Condition — criteria definition**")
            rows = []
            for idx, c in enumerate(viz["condition_criterion"]):
                if not isinstance(c, dict):
                    continue
                dist = c.get("Filter", {}) or {}
                dist_arr = dist.get("Distance")
                if isinstance(dist_arr, (list, tuple)) and len(dist_arr) >= 2:
                    lo, hi = dist_arr[0], dist_arr[1]
                    dist_str = f"{lo:.0f}+m" if hi > 1e100 else f"{lo:.0f}–{hi:.0f}m"
                else:
                    dist_str = str(dist_arr) if dist_arr is not None else "—"
                rows.append({
                    "index": idx,
                    "criteria": f"criteria_{idx}",
                    "Distance (m)": dist_str,
                    "Level": c.get("CriteriaLevel", ""),
                    "PassRate %": c.get("PassRate"),
                    "Method": c.get("CriteriaMethod", ""),
                })
            if rows:
                cond_df = pd.DataFrame(rows)
                st.dataframe(cond_df, width='stretch', hide_index=True)

        # Overview metrics
        frames_list = viz["frames"]
        if frames_list:
            n_frames = len(frames_list)
            success_count = sum(1 for f in frames_list if f.get("success") is True)
            no_data_count = sum(1 for f in frames_list if (f.get("summary") or "").strip() == "NoData")
            st.markdown("**Overview**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Frames loaded", n_frames)
            with c2:
                st.metric("Success", success_count)
            with c3:
                pct = (100.0 * success_count / n_frames) if n_frames else 0
                st.metric("Success %", f"{pct:.1f}%")
            with c4:
                st.metric("NoData lines", no_data_count)

            # Success over time (timeline)
            if HAS_PLOTLY and n_frames > 0:
                line_indices = [f["line_index"] for f in frames_list]
                success_vals = [1 if f.get("success") is True else (0 if f.get("success") is False else 0.5) for f in frames_list]
                fig_timeline = go.Figure()
                fig_timeline.add_trace(
                    go.Scatter(
                        x=line_indices,
                        y=success_vals,
                        mode="markers",
                        marker=dict(size=4, color=success_vals, colorscale=[[0, "#e74c3c"], [0.5, "#95a5a6"], [1, "#27ae60"]], showscale=True, colorbar=dict(tickvals=[0, 0.5, 1], ticktext=["Fail", "—", "Success"])),
                        name="Result",
                    )
                )
                fig_timeline.update_layout(
                    title="Result.Success over line index",
                    xaxis_title="Line index",
                    yaxis_title="Success",
                    yaxis=dict(tickvals=[0, 0.5, 1], ticktext=["Fail", "N/A", "Success"], range=[-0.1, 1.1]),
                    height=220,
                    margin=dict(t=40, b=40, l=50, r=30),
                )
                st.plotly_chart(fig_timeline, width='stretch')

            # Criteria heatmap: rows = frames (optionally downsampled), cols = criteria_0..N
            frame_recs_with_criteria = [f for f in frames_list if f.get("criteria_status")]
            if frame_recs_with_criteria and HAS_PLOTLY:
                # Collect all criteria_* keys and sort
                all_criteria = sorted(set().union(*(r["criteria_status"].keys() for r in frame_recs_with_criteria)), key=lambda x: (int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0))
                if all_criteria:
                    # Downsample rows if too many for readable heatmap
                    max_heatmap_rows = 150
                    if len(frame_recs_with_criteria) > max_heatmap_rows:
                        step = len(frame_recs_with_criteria) // max_heatmap_rows
                        subset = frame_recs_with_criteria[:: max(1, step)][:max_heatmap_rows]
                    else:
                        subset = frame_recs_with_criteria
                    status_to_num = {"NoGTNoObj": 0, "Fail": 0.3, "Success": 1, "—": 0.5}
                    z = []
                    customtext = []
                    for r in subset:
                        row_num = [status_to_num.get(r["criteria_status"].get(c, "—"), 0.5) for c in all_criteria]
                        row_txt = [r["criteria_status"].get(c, "—") for c in all_criteria]
                        z.append(row_num)
                        customtext.append(row_txt)
                    fig_heat = go.Figure(
                        data=go.Heatmap(
                            z=z,
                            x=all_criteria,
                            y=[r.get("frame_name") or str(r["line_index"]) for r in subset],
                            text=customtext,
                            hovertemplate="Frame: %{y}<br>Criteria: %{x}<br>Status: %{text}<extra></extra>",
                            colorscale=[[0, "#bdc3c7"], [0.3, "#e74c3c"], [0.5, "#95a5a6"], [1, "#27ae60"]],
                            colorbar=dict(tickvals=[0, 0.3, 0.5, 1], ticktext=["NoGTNoObj", "Fail", "—", "Success"]),
                        )
                    )
                    fig_heat.update_layout(
                        title="Criteria status by frame (NoGTNoObj / Fail / Success)",
                        xaxis_title="Criteria",
                        yaxis_title="Frame",
                        height=min(500, 80 + len(subset) * 12),
                        margin=dict(t=40, b=40, l=80, r=100),
                    )
                    st.plotly_chart(fig_heat, width='stretch')

        if viz["final_score"]:
            with st.expander("FinalScore (from last line)"):
                st.json(viz["final_score"])

    st.subheader("First lines — structure (keys, Frame, criteria, Result)")
    for detail in debug_info["first_lines_detail"]:
        with st.expander(f"Line {detail['line_index']}", expanded=(detail["line_index"] <= 2)):
            st.write("**Top-level keys:**", detail.get("keys", []))
            if detail.get("frame_keys"):
                st.write("**Frame keys:**", detail["frame_keys"])
            if detail.get("criteria_keys"):
                st.write("**Criteria keys:**", detail["criteria_keys"])
            if detail.get("criteria6") is not None:
                st.write("**criteria6:**")
                st.json(detail["criteria6"])
                if "pass_fail" in detail:
                    st.write("**PassFail:**", detail["pass_fail"])
                if "pass_fail_info" in detail:
                    st.write("**PassFail Info:**", detail["pass_fail_info"])
            if detail.get("result_summary_preview") is not None:
                st.write("**Result.Summary (preview):**", detail["result_summary_preview"])
            if detail.get("result_keys") is not None:
                st.write("**Result keys:**", detail["result_keys"])
            if "decode_error" in detail:
                st.error("JSON decode error: " + detail["decode_error"])

    st.subheader("Last lines — FinalScore")
    if not debug_info["last_lines_finalscore"]:
        st.caption("No line with Frame.FinalScore found in the last N lines.")
    else:
        for item in debug_info["last_lines_finalscore"]:
            with st.expander(f"Line {item['line_index']} — FinalScore"):
                st.write("**FinalScore keys:**", item.get("FinalScore_keys", []))
                if item.get("Score_keys"):
                    st.write("**Score keys:**", item["Score_keys"])
                if item.get("TP_keys"):
                    st.write("**TP keys:**", item["TP_keys"])
                st.json(item["FinalScore"])

# =============================
# PARQUET VIEW
# =============================
elif file_type == "Parquet":
    # Optional: PyArrow metadata (row groups, schema from file)
    def get_pyarrow_metadata(path: str) -> Optional[dict]:
        try:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(path)
            return {
                "num_row_groups": meta.num_row_groups,
                "num_rows": meta.num_rows,
                "num_columns": meta.num_columns,
                "serialized_size": meta.serialized_size,
            }
        except Exception as e:
            return {"error": str(e)}

    pyarrow_meta = get_pyarrow_metadata(target_file)

    con = get_con()
    try:
        schema_df = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [target_file]
        ).df()
    except Exception as e:
        st.error(f"Failed to read parquet: {e}")
        st.stop()

    try:
        row_count_df = con.execute(
            "SELECT COUNT(*) AS row_count FROM read_parquet(?)", [target_file]
        ).df()
        row_count = int(row_count_df.at[0, "row_count"])
    except Exception:
        row_count = None

    st.subheader("File info")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Path", target_file)
    with col2:
        if file_info["size_bytes"] is not None:
            size_mb = file_info["size_bytes"] / (1024 * 1024)
            st.metric("Size", f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_info['size_bytes'] / 1024:.2f} KB")
        else:
            st.metric("Size", "—")
    with col3:
        st.metric("Rows", f"{row_count:,}" if row_count is not None else "—")

    if pyarrow_meta and "error" not in pyarrow_meta:
        with st.expander("PyArrow file metadata (row groups, etc.)"):
            st.json(pyarrow_meta)
    elif pyarrow_meta and "error" in pyarrow_meta:
        with st.expander("PyArrow metadata (optional)"):
            st.caption("PyArrow not available or error: " + pyarrow_meta["error"])

    st.subheader("Schema (column names and types)")
    st.caption("DuckDB interpretation of parquet column names and types.")
    st.dataframe(schema_df, width='stretch')

    st.subheader("Preview rows")
    preview_mode = st.radio(
        "Show",
        ["First N rows", "Last N rows", "First and last N rows", "Search", "Filter by columns"],
        horizontal=True,
        key="pq_preview_mode",
    )
    n_rows = st.slider("Number of rows (N)", 5, 500, 50, key="pq_n_rows")

    columns = schema_df["column_name"].tolist()
    # DuckDB DESCRIBE may use 'column_type' or 'Type'; normalize for filter UI
    schema_col_types = {}
    if "column_type" in schema_df.columns:
        schema_col_types = dict(zip(schema_df["column_name"], schema_df["column_type"]))
    elif "Type" in schema_df.columns:
        schema_col_types = dict(zip(schema_df["column_name"], schema_df["Type"]))

    def _is_numeric_type(ct: str) -> bool:
        if not ct or not isinstance(ct, str):
            return False
        ct = ct.upper()
        return "INT" in ct or "BIGINT" in ct or "DOUBLE" in ct or "FLOAT" in ct or "DECIMAL" in ct or "NUMERIC" in ct

    def run_preview(path: str, limit: int, from_end: bool) -> pd.DataFrame:
        if from_end:
            total = con.execute("SELECT COUNT(*) AS c FROM read_parquet(?)", [path]).df().at[0, "c"]
            offset = max(0, total - limit)
            # Ensure limit and offset are standard Python ints (not NumPy types)
            return con.execute(
                "SELECT * FROM read_parquet(?) LIMIT ? OFFSET ?",
                [path, int(limit), int(offset)],
            ).df()
        return con.execute(
            "SELECT * FROM read_parquet(?) LIMIT ?",
            [path, limit],
        ).df()

    def run_search(path: str, query: str, limit: int, case_sensitive: bool) -> pd.DataFrame:
        """Return rows where any column contains query (substring match)."""
        if not query or not query.strip():
            return run_preview(path, limit, from_end=False)
        like_op = "LIKE" if case_sensitive else "ILIKE"
        # One condition per column: CAST to VARCHAR so numbers/dates are searchable
        conditions = []
        for col in columns:
            safe = f'"{col}"' if not col.isidentifier() else col
            conditions.append(f"CAST({safe} AS VARCHAR) {like_op} CONCAT('%', ?, '%')")
        where_clause = " OR ".join(conditions)
        params = [path] + [query.strip()] * len(columns) + [int(limit)]
        return con.execute(
            f"SELECT * FROM read_parquet(?) WHERE ({where_clause}) LIMIT ?",
            params,
        ).df()

    def run_filter(
        path: str,
        filters: List[Tuple[str, str, Any]],
        limit: int,
    ) -> pd.DataFrame:
        """Return rows matching all given (column, operator, value) conditions. Empty value = skip."""
        where_parts = []
        params: List[Any] = [path]
        for col, op, val in filters:
            if val is None or (isinstance(val, str) and not val.strip()):
                continue
            safe = f'"{col}"' if not col.isidentifier() else col
            ct = schema_col_types.get(col, "")
            is_num = _is_numeric_type(str(ct))
            if is_num:
                try:
                    num_val = int(val) if "." not in str(val).strip() else float(val)
                except (ValueError, TypeError):
                    num_val = val
                if op == "=":
                    where_parts.append(f"({safe} = ?)")
                    params.append(num_val)
                elif op == "!=":
                    where_parts.append(f"({safe} != ?)")
                    params.append(num_val)
                elif op == "<":
                    where_parts.append(f"({safe} < ?)")
                    params.append(num_val)
                elif op == "<=":
                    where_parts.append(f"({safe} <= ?)")
                    params.append(num_val)
                elif op == ">":
                    where_parts.append(f"({safe} > ?)")
                    params.append(num_val)
                elif op == ">=":
                    where_parts.append(f"({safe} >= ?)")
                    params.append(num_val)
                else:
                    where_parts.append(f"({safe} = ?)")
                    params.append(num_val)
            else:
                # string: equals, not equals, contains
                str_val = str(val).strip()
                if op == "equals" or op == "=":
                    where_parts.append(f"(CAST({safe} AS VARCHAR) = ?)")
                    params.append(str_val)
                elif op == "not equals" or op == "!=":
                    where_parts.append(f"(CAST({safe} AS VARCHAR) != ?)")
                    params.append(str_val)
                elif op == "contains":
                    where_parts.append(f"(CAST({safe} AS VARCHAR) ILIKE CONCAT('%', ?, '%'))")
                    params.append(str_val)
                else:
                    where_parts.append(f"(CAST({safe} AS VARCHAR) = ?)")
                    params.append(str_val)
        if not where_parts:
            return run_preview(path, limit, from_end=False)
        where_clause = " AND ".join(where_parts)
        params.append(int(limit))
        return con.execute(
            f"SELECT * FROM read_parquet(?) WHERE {where_clause} LIMIT ?",
            params,
        ).df()

    if preview_mode == "First N rows":
        preview_df = run_preview(target_file, n_rows, from_end=False)
        st.dataframe(preview_df, width='stretch')
    elif preview_mode == "Last N rows":
        preview_df = run_preview(target_file, n_rows, from_end=True)
        st.dataframe(preview_df, width='stretch')
    elif preview_mode == "Search":
        search_query = st.text_input(
            "Search for (substring in any column)",
            placeholder="e.g. car, 0.5, or a label name",
            key="pq_search_query",
        )
        case_sensitive = st.checkbox("Case sensitive", value=False, key="pq_search_case")
        if search_query.strip():
            search_df = run_search(target_file, search_query.strip(), n_rows, case_sensitive)
            st.caption(f"Showing up to {n_rows} rows matching « {search_query.strip()} »")
            st.dataframe(search_df, width='stretch')
            if search_df.empty:
                st.info("No rows matched your search.")
        else:
            st.caption("Enter a search term to filter rows by any column. Showing first N rows until you search.")
            st.dataframe(run_preview(target_file, n_rows, from_end=False), width='stretch')
    elif preview_mode == "Filter by columns":
        st.caption("Filter by specific columns (e.g. scenario name, frame index). All non-empty conditions are ANDed.")
        has_scenario = "scenario_name" in columns
        has_frame = "frame_index" in columns or "frame" in columns
        frame_col = "frame_index" if "frame_index" in columns else ("frame" if "frame" in columns else None)
        quick_filters: List[Tuple[str, str, Any]] = []
        with st.expander("Quick filters (scenario & frame)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                scenario_val = st.text_input(
                    "Scenario name (exact)",
                    placeholder="e.g. scenario_01 or leave empty",
                    key="pq_filter_scenario",
                    help="Filter by scenario_name; leave empty to skip.",
                )
                if has_scenario and scenario_val.strip():
                    quick_filters.append(("scenario_name", "equals", scenario_val.strip()))
            with c2:
                frame_val_str = st.text_input(
                    "Frame index",
                    placeholder="e.g. 10 (leave empty for all)",
                    key="pq_filter_frame",
                    help="Filter by frame index; leave empty to show all frames.",
                )
                if frame_col is not None and frame_val_str.strip():
                    try:
                        frame_val = int(frame_val_str.strip())
                        quick_filters.append((frame_col, "=", frame_val))
                    except ValueError:
                        pass
        st.markdown("**Structured filters** — add more conditions (column, operator, value):")
        num_structured = 4
        for i in range(num_structured):
            col_choice = st.selectbox(
                f"Column {i + 1}",
                options=["(none)"] + columns,
                key=f"pq_filter_col_{i}",
            )
            if col_choice and col_choice != "(none)":
                ct = schema_col_types.get(col_choice, "")
                is_num = _is_numeric_type(str(ct))
                if is_num:
                    op_choice = st.selectbox(
                        f"Operator {i + 1}",
                        ["=", "!=", "<", "<=", ">", ">="],
                        key=f"pq_filter_op_{i}",
                    )
                    val_choice = st.text_input(
                        f"Value {i + 1} (number)",
                        placeholder="e.g. 10",
                        key=f"pq_filter_val_{i}",
                    )
                else:
                    op_choice = st.selectbox(
                        f"Operator {i + 1}",
                        ["equals", "not equals", "contains"],
                        key=f"pq_filter_op_{i}",
                    )
                    val_choice = st.text_input(
                        f"Value {i + 1}",
                        placeholder="e.g. scenario_01",
                        key=f"pq_filter_val_{i}",
                    )
                if val_choice.strip():
                    quick_filters.append((col_choice, op_choice, val_choice.strip() if not is_num else val_choice))
        all_filters = quick_filters
        if all_filters:
            filter_df = run_filter(target_file, all_filters, n_rows)
            desc = " and ".join(
                f"{c}={v!r}" for (c, _op, v) in all_filters
            )
            st.caption(f"Showing up to {n_rows} rows where {desc}")
            st.dataframe(filter_df, width='stretch')
            if filter_df.empty:
                st.info("No rows matched the filters.")
        else:
            st.caption("Set at least one quick filter (scenario name or frame index) or a structured filter above. Showing first N rows until then.")
            st.dataframe(run_preview(target_file, n_rows, from_end=False), width='stretch')
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**First N rows**")
            st.dataframe(run_preview(target_file, n_rows, from_end=False), width='stretch')
        with c2:
            st.write("**Last N rows**")
            st.dataframe(run_preview(target_file, n_rows, from_end=True), width='stretch')

    st.subheader("Column statistics")
    st.caption("Null counts, distinct counts, and min/max for numeric columns.")
    columns = schema_df["column_name"].tolist()
    agg_exprs = ["COUNT(*) AS total"]
    for col in columns:
        safe = f'"{col}"' if not col.isidentifier() else col
        ckey = col.replace(" ", "_").replace("-", "_")
        agg_exprs.append(f"COUNT({safe}) AS non_null_{ckey}")
        agg_exprs.append(f"COUNT(DISTINCT {safe}) AS distinct_{ckey}")

    try:
        agg_df = con.execute(
            f"SELECT {', '.join(agg_exprs)} FROM read_parquet(?)",
            [target_file],
        ).df()
    except Exception as e:
        st.warning(f"Could not compute column stats: {e}")
        agg_df = None

    if agg_df is not None and not agg_df.empty:
        total = int(agg_df["total"].iloc[0])
        stats_rows = []
        for col in columns:
            ckey = col.replace(" ", "_").replace("-", "_")
            nn_key = f"non_null_{ckey}"
            dist_key = f"distinct_{ckey}"
            if nn_key not in agg_df.columns or dist_key not in agg_df.columns:
                continue
            non_null = int(agg_df[nn_key].iloc[0])
            distinct = int(agg_df[dist_key].iloc[0])
            null_count = total - non_null
            stats_rows.append({
                "column": col,
                "null_count": null_count,
                "non_null_count": non_null,
                "distinct_count": distinct,
            })
        stats_df = pd.DataFrame(stats_rows)
        st.dataframe(stats_df, width='stretch')

    st.subheader("Sample values by column")
    col_choice = st.selectbox("Column", columns, key="pq_sample_col")
    n_sample = st.slider("Number of sample values to show", 5, 100, 20, key="pq_n_sample")
    safe_col = f'"{col_choice}"' if not col_choice.isidentifier() else col_choice
    try:
        sample_df = con.execute(
            f"SELECT DISTINCT {safe_col} AS value FROM read_parquet(?) ORDER BY value LIMIT ?",
            [target_file, n_sample],
        ).df()
        st.dataframe(sample_df, width='stretch')
    except Exception as e:
        st.warning(f"Could not sample column: {e}")

    with st.expander("Run custom SQL on this parquet file"):
        st.caption("Use placeholder ? for the file path. Example: SELECT * FROM read_parquet(?) LIMIT 10")
        custom_sql = st.text_area("SQL", value="SELECT * FROM read_parquet(?) LIMIT 10", key="pq_custom_sql")
        if st.button("Run"):
            try:
                custom_df = con.execute(custom_sql, [target_file]).df()
                st.dataframe(custom_df, width='stretch')
            except Exception as e:
                st.error(str(e))

# =============================
# PKL VIEW
# =============================
else:  # PKL
    st.subheader("File info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Path", target_file)
    with col2:
        if file_info["size_bytes"] is not None:
            size_mb = file_info["size_bytes"] / (1024 * 1024)
            st.metric("Size", f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_info['size_bytes'] / 1024:.2f} KB")
        else:
            st.metric("Size", "—")

    try:
        pkl_data = load_pkl_file(target_file)
    except Exception as e:
        st.error(f"Failed to load PKL: {e}")
        st.stop()

    st.subheader("Root object")
    st.caption(f"Type: **{type(pkl_data).__module__}.{type(pkl_data).__name__}**")

    st.subheader("Structure")
    view_mode = st.radio(
        "View",
        ["Summary (table)", "Tree (expandable)", "Full text"],
        horizontal=True,
        key="pkl_structure_view",
        help="Summary: compact attribute table. Tree: expand/collapse nodes. Full text: raw structure dump.",
    )

    if view_mode == "Summary (table)":
        # Root summary
        root_summary = _type_summary(pkl_data)
        st.markdown(f"**Root:** `{root_summary}`")
        rows = _summary_rows(pkl_data)
        if rows:
            summary_df = pd.DataFrame(rows, columns=["Attribute", "Type", "Summary"])
            st.dataframe(summary_df, width='stretch', hide_index=True)
        # If root is a list of objects, show "first item" attribute table so users see one frame's shape
        if isinstance(pkl_data, (list, tuple)) and len(pkl_data) > 0:
            first = pkl_data[0]
            if hasattr(first, "__dict__") and not isinstance(first, type):
                st.markdown("**First item attributes** (shape of one element)")
                first_rows = _summary_rows(first, prefix="[0]")
                first_df = pd.DataFrame(first_rows, columns=["Attribute", "Type", "Summary"])
                st.dataframe(first_df, width='stretch', hide_index=True)
                idx = st.selectbox(
                    "Or summarize item at index",
                    options=list(range(min(10, len(pkl_data)))),
                    key="pkl_summary_index",
                )
                if idx != 0:
                    other = pkl_data[idx]
                    if hasattr(other, "__dict__"):
                        other_rows = _summary_rows(other, prefix=f"[{idx}]")
                        other_df = pd.DataFrame(other_rows, columns=["Attribute", "Type", "Summary"])
                        st.dataframe(other_df, width='stretch', hide_index=True)
    elif view_mode == "Tree (expandable)":
        tree_depth = st.slider("Max expand depth", 1, 5, 2, key="pkl_tree_depth")
        _render_tree_expander(pkl_data, "root", depth=0, max_depth=tree_depth, key_prefix="pkl_tree")
    else:
        structure_lines = _describe_pkl(pkl_data)
        st.text("\n".join(structure_lines))

    # If root is a DataFrame, show preview
    if isinstance(pkl_data, pd.DataFrame):
        st.subheader("DataFrame preview")
        n_pkl = st.slider("Rows to show", 5, 500, 50, key="pkl_n_rows")
        st.dataframe(pkl_data.head(n_pkl), width='stretch')
    else:
        # Collect any DataFrames found in the structure for optional preview
        def find_dataframes(obj: Any, path: str = "root") -> List[Tuple[str, pd.DataFrame]]:
            out = []
            if isinstance(obj, pd.DataFrame):
                out.append((path, obj))
                return out
            if isinstance(obj, dict):
                for k, v in obj.items():
                    out.extend(find_dataframes(v, f"{path}[{repr(k)}]"))
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj[:20]):
                    out.extend(find_dataframes(v, f"{path}[{i}]"))
            elif hasattr(obj, "__dict__") and not isinstance(obj, type):
                for k, v in obj.__dict__.items():
                    out.extend(find_dataframes(v, f"{path}.{k}"))
            return out

        dfs_found = find_dataframes(pkl_data)
        if dfs_found:
            st.subheader("DataFrames inside PKL")
            df_choice = st.selectbox(
                "Select DataFrame",
                options=list(range(len(dfs_found))),
                format_func=lambda i: f"{dfs_found[i][0]} (shape {dfs_found[i][1].shape})",
                key="pkl_df_choice",
            )
            path_label, df_preview = dfs_found[df_choice]
            st.caption(f"Path: `{path_label}`")
            n_pkl = st.slider("Rows to show", 5, 500, 50, key="pkl_n_rows")
            st.dataframe(df_preview.head(n_pkl), width='stretch')

    st.subheader("Sample values")
    st.caption("Actual content of the first few items (no object addresses).")
    n_sample = st.slider("Number of items to preview", 1, 5, 2, key="pkl_n_sample")
    preview_depth = st.slider("Preview depth", 1, 4, 2, key="pkl_preview_depth")
    try:
        if isinstance(pkl_data, (list, tuple)):
            to_show = pkl_data[:n_sample]
            for i, item in enumerate(to_show):
                preview = _to_preview_value(item, max_depth=preview_depth)
                with st.expander(f"Item [{i}]", expanded=True):
                    st.json(preview)
        else:
            preview = _to_preview_value(pkl_data, max_depth=preview_depth)
            st.json(preview)
    except Exception as e:
        st.warning(f"Could not build preview: {e}. Showing type summary.")
        st.text(_type_summary(pkl_data))

    with st.expander("Debug: raw repr (first 10k chars)"):
        raw_repr = repr(pkl_data)
        if len(raw_repr) > 10_000:
            raw_repr = raw_repr[:10_000] + "\n... (truncated)"
        st.text(raw_repr)
