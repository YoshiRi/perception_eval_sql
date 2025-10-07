import duckdb
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import glob
import os
st.set_page_config(layout="wide")
st.title("BEV Bounding Box Viewer")

# === 1. Parquetファイル選択 ===
parquet_files = glob.glob("data/*.parquet")
selected_file = st.selectbox("Select Parquet file", parquet_files)

# === 2. DuckDB接続 ===
con = duckdb.connect()

# === 3. カラム存在チェック ===
cols = con.execute(f"DESCRIBE SELECT * FROM parquet_scan('{selected_file}')").df()["column_name"].tolist()
has_visibility = "visibility" in cols

# === 4. t4dataset_id 一覧 ===
t4_ids = con.execute(f"""
    SELECT DISTINCT t4dataset_id 
    FROM parquet_scan('{selected_file}') 
    ORDER BY t4dataset_id
""").df()["t4dataset_id"].dropna().tolist()
selected_t4 = st.selectbox("Select t4dataset_id", t4_ids)

# === 5. topic_name 一覧 ===
topic_names = con.execute(f"""
    SELECT DISTINCT topic_name 
    FROM parquet_scan('{selected_file}')
    WHERE t4dataset_id = '{selected_t4}'
    ORDER BY topic_name
""").df()["topic_name"].dropna().tolist()
selected_topic = st.multiselect("Select topic_name(s)", topic_names, default=topic_names)

# === 6. label 一覧 ===
labels = con.execute(f"""
    SELECT DISTINCT label 
    FROM parquet_scan('{selected_file}')
    WHERE t4dataset_id = '{selected_t4}'
    ORDER BY label
""").df()["label"].dropna().tolist()
selected_labels = st.multiselect("Select label(s)", labels, default=labels)

# === 7. visibility（存在する場合のみ） ===
if has_visibility:
    visibilities = con.execute(f"""
        SELECT DISTINCT COALESCE(visibility, 'UNKNOWN') AS visibility
        FROM parquet_scan('{selected_file}')
        WHERE t4dataset_id = '{selected_t4}'
        ORDER BY visibility
    """).df()["visibility"].tolist()

    selected_visibility = st.multiselect(
        "Select visibility", visibilities, default=visibilities
    )
else:
    selected_visibility = []
    st.info("No 'visibility' column found — skipping visibility filter.")

# === 8. データ読み込み ===
topic_filter = "', '".join(selected_topic)
label_filter = "', '".join(selected_labels)
where_clause = f"t4dataset_id = '{selected_t4}' AND topic_name IN ('{topic_filter}') AND label IN ('{label_filter}')"

if has_visibility and selected_visibility:
    vis_filter = "', '".join(selected_visibility)
    where_clause += f" AND COALESCE(visibility, 'UNKNOWN') IN ('{vis_filter}')"

df = con.execute(f"""
    SELECT frame_index, x, y, length, width, yaw, label, topic_name, source, status
    {', visibility' if has_visibility else ''}
    FROM parquet_scan('{selected_file}')
    WHERE {where_clause}
    ORDER BY frame_index
""").df()

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# === 9. 色設定 ===
color_map = {
    ("GT", "TP"): "#00cc66",   # 緑
    ("GT", "FN"): "#ff9933",   # オレンジ
    ("EST", "TP"): "#66b3ff",  # 青
    ("EST", "FP"): "#ff6666",  # 赤
}

default_color = "#999999"

def get_color(source, status):
    return color_map.get((source, status), default_color)

# === 10. フレーム選択 ===
frame = st.slider("Frame index", int(df.frame_index.min()), int(df.frame_index.max()), step=1)
df_frame = df[df.frame_index == frame]

# === 追加: データ概要をパネル表示 ===
def get_valid_datanum(df):
    total_records = len(df)
    valid_records = len(df.query("length > 0 and width > 0"))
    return total_records, valid_records

total_records, valid_records = get_valid_datanum(df_frame)


# === 11. 矩形描画関数 ===
def rotated_rect(x, y, length, width, yaw):
    dx = length / 2
    dy = width / 2
    corners = np.array([
        [-dx, -dy],
        [-dx,  dy],
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy]
    ])
    c, s = np.cos(yaw+np.pi/2), np.sin(yaw+np.pi/2)  # BEVでの向きに合わせる
    # c, s = np.cos(yaw), np.sin(yaw)  # BEVでの向きに合わせる
    rot = np.array([[c, -s], [s, c]])
    rotated = corners @ rot.T
    return rotated[:,0] + x, rotated[:,1] + y

# === 12. BEV描画 ===
fig = go.Figure()
shown_labels = set()

for _, row in df_frame.iterrows():
    x_poly, y_poly = rotated_rect(row.x, row.y, row.length, row.width, row.yaw)
    label_text = f"{row.source}/{row.status}"
    group = f"{row.source}/{row.status}"  # ← 独立制御のため細分化

    show_legend = label_text not in shown_labels
    shown_labels.add(label_text)

    fig.add_trace(go.Scatter(
        x=x_poly, y=y_poly, mode='lines',
        name=label_text,
        legendgroup=group,        # ← source+statusごとに独立グループ化
        fill="toself", opacity=0.6,
        line=dict(color=get_color(row.source, row.status)),
        showlegend=show_legend
    ))

# === 自車マーカー追加 ===
ego_x = [0.0, -1.5, -1.5, 0.0]  # 車体形状を簡略化した三角形
ego_y = [0.0, -1.0,  1.0, 0.0]

fig.add_trace(go.Scatter(
    x=ego_x, y=ego_y,
    mode='lines', fill='toself',
    line=dict(color='black', width=2),
    fillcolor='gray',
    name='Ego Vehicle',
    showlegend=True
))


fig.update_layout(
    title=f"{os.path.basename(selected_file)} | t4dataset_id={selected_t4} | Frame {frame} | BBoxes: {len(df_frame)} (Total: {total_records}, Valid: {valid_records})",
    xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
    yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
    width=950, height=900
)

st.plotly_chart(fig, use_container_width=True)
