import duckdb
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import glob
import os
from typing import Tuple

st.set_page_config(layout="wide")
st.title("BEV Bounding Box Viewer")

# ----------------------------
# Sidebar (Filters)
# ----------------------------
with st.sidebar:
    st.header("Filters")

    # Parquet files
    parquet_files = sorted(glob.glob("data/*.parquet"))
    if not parquet_files:
        st.error("No Parquet files under data/*.parquet")
        st.stop()
    selected_file = st.selectbox("Parquet file", parquet_files)

# DuckDB connection (no cache = 安定優先)
con = duckdb.connect()

# --- Columns (for visibility existence check)
cols = con.execute("DESCRIBE SELECT * FROM parquet_scan(?)", [selected_file]).df()["column_name"].tolist()
has_visibility = "visibility" in cols

# --- t4dataset_id
t4_ids = con.execute(
    "SELECT DISTINCT t4dataset_id AS v FROM parquet_scan(?) ORDER BY v",
    [selected_file]
).df()["v"].dropna().tolist()
if not t4_ids:
    st.error("No t4dataset_id in file")
    st.stop()

with st.sidebar:
    selected_t4 = st.selectbox("t4dataset_id", t4_ids)

# --- topic_name（単一選択）
topic_names = con.execute(
    "SELECT DISTINCT topic_name AS v FROM parquet_scan(?) WHERE t4dataset_id=? ORDER BY v",
    [selected_file, selected_t4]
).df()["v"].dropna().tolist()
if not topic_names:
    st.warning("No topic_name for selected t4dataset_id")
    st.stop()

with st.sidebar:
    selected_topic = st.selectbox("topic_name (single)", topic_names)

# --- label（複数選択）
labels = con.execute(
    "SELECT DISTINCT label AS v FROM parquet_scan(?) WHERE t4dataset_id=? ORDER BY v",
    [selected_file, selected_t4]
).df()["v"].dropna().tolist()
if not labels:
    st.warning("No label for selected t4dataset_id")
    st.stop()

with st.sidebar:
    selected_labels = st.multiselect("label(s)", labels, default=labels)

# --- visibility（列があるときだけ。NULLは UNKNOWN で扱う）
selected_visibility = None
if has_visibility:
    vis_list = con.execute(
        "SELECT DISTINCT COALESCE(visibility,'UNKNOWN') AS v FROM parquet_scan(?) WHERE t4dataset_id=? ORDER BY v",
        [selected_file, selected_t4]
    ).df()["v"].tolist()
    with st.sidebar:
        if vis_list:
            selected_visibility = st.multiselect("visibility", vis_list, default=vis_list)
        else:
            st.info("No visibility values found — skipping.")
else:
    with st.sidebar:
        st.info("No 'visibility' column found — skipping visibility filter.")

# Guard
if not selected_labels:
    st.warning("No label selected.")
    st.stop()

# ----------------------------
# Build query safely & load data
# ----------------------------
where = ["t4dataset_id = ?", "topic_name = ?"]  # topic_name は単一選択
params = [selected_file, selected_t4, selected_topic]

# label IN (...)
where.append(f"label IN ({','.join(['?']*len(selected_labels))})")
params.extend(selected_labels)

# visibility（ある場合のみ、NULLは UNKNOWN で比較）
select_vis = ", visibility" if has_visibility else ""
if has_visibility and selected_visibility:
    where.append(f"COALESCE(visibility,'UNKNOWN') IN ({','.join(['?']*len(selected_visibility))})")
    params.extend(selected_visibility)

sql = f"""
SELECT frame_index, x, y, length, width, yaw, label, topic_name, source, status, uuid
{select_vis}
FROM parquet_scan(?)
WHERE {" AND ".join(where)}
ORDER BY frame_index
"""

df = con.execute(sql, params).df()
if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# frame_index を int に（比較安定化）
if "frame_index" in df.columns and not np.issubdtype(df["frame_index"].dtype, np.integer):
    df["frame_index"] = df["frame_index"].astype("Int64").fillna(0).astype(int)

# ----------------------------
# Color map
# ----------------------------
color_map = {
    ("GT", "TP"): "#00cc66",   # 緑
    ("GT", "FN"): "#ff9933",   # オレンジ
    ("EST", "TP"): "#66b3ff",  # 青
    ("EST", "FP"): "#ff6666",  # 赤
}
def get_color(source, status): return color_map.get((source, status), "#999999")

# ----------------------------
# Frame slider + stats
# ----------------------------
frame = st.slider("Frame index", int(df.frame_index.min()), int(df.frame_index.max()), step=1)
df_frame = df[df.frame_index == frame]

total_records = len(df_frame)
valid_records = int(((df_frame["length"] > 0) & (df_frame["width"] > 0)).sum())

# ----------------------------
# Geometry (yaw補正: x前方, y左方 → +π/2)
# ----------------------------
def rotated_rect(
    x: float, y: float,
    length: float, width: float,
    yaw: float,
    step_depth_ratio: float = 0.25,
    step_width_ratio: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    前方左側に段差（凹み）を入れて向きを表す矩形Polylineを返す。
    - yaw: ラジアン
    - step_depth_ratio: 凹みの「奥行き」（length比）
    - step_width_ratio: 凹みの「横幅」（width比）
    """
    if length < width:
        # something is wrong, fix size
        length, width = max(length, width), min(length, width)

    dx, dy = length / 2.0, width / 2.0
    step_depth = length * step_depth_ratio
    step_width = width * step_width_ratio

    # 頂点順序（時計回り）
    # 後ろ左 → 前左(手前側) → 凹み奥 → 前中央左 → 前右 → 後右 → 後ろ左
    corners = np.array([
        [-dx, -dy],                      # 後ろ左
        [ dx, -dy],                      # 前左端
        [ dx, 0],         # 段差上部
        [ dx - step_depth, 0],  # 凹み奥左
        [dx, 0],
        [dx,  dy],                      # 前右端
        [-dx,  dy],                      # 後右
        [-dx, -dy]                       # 戻る
    ])

    # 回転 (+π/2 でBEV向き調整)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    rotated = corners @ rot.T

    xs, ys = rotated[:, 0] + x, rotated[:, 1] + y
    return xs, ys


# ----------------------------
# Plot
# ----------------------------
fig = go.Figure()
shown = set()

for _, row in df_frame.iterrows():
    # 描けないBBoxはスキップ
    if not (row.length > 0 and row.width > 0):
        continue

    x_poly, y_poly = rotated_rect(row.x, row.y, row.length, row.width, row.yaw)
    name = f"{row.source}/{row.status}"
    show = name not in shown; shown.add(name)
    fig.add_trace(go.Scatter(
        x=x_poly, y=y_poly, mode="lines",
        name=name, legendgroup=name,
        fill="toself", opacity=0.6,
        line=dict(color=get_color(row.source, row.status)),
        showlegend=show
    ))

# Ego marker（固定三角形）
fig.add_trace(go.Scatter(
    x=[0, -1.5, -1.5, 0], y=[0, -1, 1, 0],
    mode="lines", fill="toself",
    line=dict(color="black", width=2),
    fillcolor="gray", name="Ego Vehicle", showlegend=True
))

fig.update_layout(
    title=f"{os.path.basename(selected_file)} | t4dataset_id={selected_t4} | "
          f"topic={selected_topic} | Frame {frame} "
          f"| This frame: Total {total_records:,}, Valid {valid_records:,}",
    xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
    yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
    legend=dict(groupclick="togglegroup", title="Source / Status"),
    height=900
)
st.plotly_chart(fig, use_container_width=True)

# === Frame別 TP/FN カウントと比率 ===
st.markdown("## 📈 Detection Stability over Frames")

# TP/FN集計
frame_stats = (
    df.query("source == 'GT' and status in ['TP', 'FN']")
      .groupby(["frame_index", "status"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

# 比率 (TP率 = TP / (TP+FN))
frame_stats["TPR"] = frame_stats["TP"] / (frame_stats["TP"] + frame_stats["FN"]).replace(0, np.nan)

# Plotlyで折れ線プロット
import plotly.express as px
# --- 時系列グラフ ---
fig_tpr = px.line(
    frame_stats,
    x="frame_index",
    y=["TP", "FN"],
    title="TP / FN Counts per Frame",
    labels={"value": "Count", "frame_index": "Frame Index", "variable": "Status"},
)
fig_tpr.update_layout(height=400, legend_title="Status")

# --- 現在Frameに縦破線を追加 ---
fig_tpr.add_vline(
    x=frame,
    line=dict(color="black", dash="dash", width=2),
    annotation_text=f"Frame {frame}",
    annotation_position="top left"
)

st.plotly_chart(fig_tpr, use_container_width=True)

# TPR比率の推移を別グラフで
fig_ratio = px.line(
    frame_stats,
    x="frame_index",
    y="TPR",
    title="True Positive Rate (TPR) per Frame",
    labels={"TPR": "True Positive Rate", "frame_index": "Frame Index"},
)
fig_ratio.update_yaxes(range=[0, 1])
fig_ratio.add_vline(
    x=frame,
    line=dict(color="black", dash="dash", width=2),
    annotation_text=f"Frame {frame}",
    annotation_position="top left"
)
st.plotly_chart(fig_ratio, use_container_width=True)

# === Worst-performing objects by FN rate ===
st.markdown("## 🚨 Objects with High FN Rate (GT-based)")

# uuid + label ごとにTP/FNをカウント
uuid_perf = (
    df.query("source == 'GT' and status in ['TP','FN']")
      .groupby(["uuid", "label"])["status"]
      .value_counts()
      .unstack(fill_value=0)
      .reset_index()
)

uuid_perf["total"] = uuid_perf["TP"] + uuid_perf["FN"]
uuid_perf["FN_rate"] = uuid_perf["FN"] / uuid_perf["total"].replace(0, np.nan)

# total > 0 だけ残す
uuid_perf = uuid_perf[uuid_perf["total"] > 0]

# FN率でソート
uuid_perf_sorted = uuid_perf.sort_values("FN_rate", ascending=False)

# 表示
st.dataframe(
    uuid_perf_sorted[["uuid", "label", "TP", "FN", "total", "FN_rate"]]
        .head(30)
        .style.format({"FN_rate": "{:.2%}"})
)

st.markdown("### 🔍 Inspect a Specific GT Object")

bad_uuid = st.selectbox("Select UUID to visualize", uuid_perf_sorted["uuid"].head(50))

uuid_traj = (
    df[(df["uuid"] == bad_uuid) & (df["source"] == "GT")]
    .sort_values("frame_index")
)

if not uuid_traj.empty:
    # --- marker symbol by status ---
    symbol_map = {"TP": "circle", "FN": "x", "FP": "triangle-up"}
    uuid_traj["marker_symbol"] = uuid_traj["status"].map(symbol_map).fillna("circle")

    fig_traj = go.Figure()

    # --- 全点を線で繋ぐ ---
    fig_traj.add_trace(go.Scatter(
        x=uuid_traj["x"], y=uuid_traj["y"],
        mode="lines",
        line=dict(color="gray", width=1),
        name=f"Trajectory ({uuid_traj['label'].iloc[0]})"
    ))

    # --- 各点（TP/FN別symbol） ---
    for status, group in uuid_traj.groupby("status"):
        fig_traj.add_trace(go.Scatter(
            x=group["x"], y=group["y"],
            mode="markers",
            marker=dict(
                symbol=symbol_map.get(status, "circle"),
                size=[10 if f == frame else 6 for f in group["frame_index"]],
                color=["red" if f == frame else ("orange" if status == "FN" else "green") for f in group["frame_index"]],
                line=dict(width=1, color="black")
            ),
            name=f"{status} points"
        ))

    fig_traj.update_layout(
        title=f"Trajectory of UUID {bad_uuid} ({uuid_traj['label'].iloc[0]})",
        xaxis=dict(title="X [m]", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y [m]", scaleanchor="x", scaleratio=1),
        height=600,
        legend=dict(title="Status")
    )
    st.plotly_chart(fig_traj, use_container_width=True)
else:
    st.info("No GT trajectory data for the selected UUID.")
