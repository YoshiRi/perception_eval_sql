import duckdb
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import glob
import os
from typing import Tuple, List, Dict

st.set_page_config(layout="wide")
st.title("BEV Bounding Box Viewer (A/B Comparison)")

# =============================
# 汎用ユーティリティ
# =============================
def rotated_rect(x: float, y: float, length: float, width: float, yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    """yawはラジアン。中心(x,y)、長さlength(前後)、幅width(左右)。BEVの進行方向合わせのため+pi/2回転。"""
    dx, dy = length / 2.0, width / 2.0
    corners = np.array([[-dx, -dy], [-dx,  dy], [ dx,  dy], [ dx, -dy], [-dx, -dy]])
    c, s = np.cos(yaw + np.pi/2), np.sin(yaw + np.pi/2)
    rot = np.array([[c, -s], [s, c]])
    rotated = corners @ rot.T
    return rotated[:, 0] + x, rotated[:, 1] + y

def get_color_map() -> Dict[Tuple[str, str], str]:
    # 既存色 + A/B強調色（ESTのTP/FPはA/Bで濃淡）
    return {
        ("GT", "TP"): "#00cc66",
        ("GT", "FN"): "#ff9933",
        ("EST", "TP_A"): "#3388ff",
        ("EST", "TP_B"): "#66b3ff",
        ("EST", "FP_A"): "#cc3333",
        ("EST", "FP_B"): "#ff6666",
        # 差分用
        ("DIFF", "IMPROVED"): "#00aa88",  # FN->TP
        ("DIFF", "DEGRADED"): "#cc5500",  # TP->FN
        ("DIFF", "NEW_FP"):   "#cc0000",
        ("DIFF", "FIXED_FP"): "#3366cc",
    }

def add_ego(fig: go.Figure):
    ego_x = [0.0, -1.5, -1.5, 0.0]
    ego_y = [0.0, -1.0,  1.0, 0.0]
    fig.add_trace(go.Scatter(x=ego_x, y=ego_y, mode='lines', fill='toself',
                             line=dict(color='black', width=2),
                             fillcolor='gray', name='Ego', showlegend=True))

def plot_frame(fig: go.Figure, df_frame, palette: Dict[Tuple[str,str], str], tag: str,
               opacity: float = 0.55, dash: str | None = None, showlegend: bool = True):
    """tag は 'A' or 'B'。ESTのTP/FPにA/Bのサフィックスを付けて色分け。dash='dash' などでBの描画を差別化可。"""
    shown = set()
    for _, r in df_frame.iterrows():
        x_poly, y_poly = rotated_rect(r.x, r.y, r.length, r.width, r.yaw)
        if r.source == "EST":
            status_key = f"{r.status}_{tag}" if r.status in ("TP","FP") else r.status
        else:
            status_key = r.status
        key = (r.source, status_key)
        name = f"{r.source}/{status_key}_{tag}"
        lg = (name not in shown) and showlegend
        shown.add(name)
        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly, mode='lines', fill='toself', opacity=opacity,
            line=dict(color=palette.get(key, "#999999"), dash=dash),
            name=name, showlegend=lg, legendgroup=name
        ))

def plot_diff(fig: go.Figure, df_diff, palette, types: List[str] | None = None, width: int = 3, opacity: float = 0.45):
    """types を指定すればそのdiffだけ描画（例: ['IMPROVED','DEGRADED']）。"""
    import pandas as pd
    if df_diff is None or df_diff.empty or "diff_type" not in df_diff.columns:
        return
    ddf = df_diff if types is None else df_diff[df_diff["diff_type"].isin(types)]
    if ddf.empty:
        return
    shown = set()
    for _, r in ddf.iterrows():
        x_poly, y_poly = rotated_rect(r.x, r.y, r.length, r.width, r.yaw)
        key = ("DIFF", r.diff_type)
        name = f"Δ {r.diff_type}"
        lg = name not in shown
        shown.add(name)
        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly, mode='lines', fill='toself', opacity=opacity,
            line=dict(color=palette.get(key, "#777777"), width=width),
            name=name, showlegend=lg, legendgroup=name
        ))

def summarize_diff(df_diff):
    import pandas as pd
    if df_diff is None or df_diff.empty or "diff_type" not in df_diff.columns:
        return 0, 0, 0, 0
    s = df_diff["diff_type"].value_counts()
    return int(s.get("IMPROVED", 0)), int(s.get("DEGRADED", 0)), int(s.get("NEW_FP", 0)), int(s.get("FIXED_FP", 0))

# =============================
# データロード & フィルタ
# =============================
with st.sidebar:
    st.header("Filters / Inputs")

    parquet_files = glob.glob("data/*.parquet")
    if not parquet_files:
        st.error("data/*.parquet が見つかりません。")
        st.stop()

    colA, colB = st.columns(2)
    with colA:
        file_A = st.selectbox("Parquet A", parquet_files, key="fileA")
    with colB:
        file_B = st.selectbox("Parquet B (同じでも可)", parquet_files, index=min(1, len(parquet_files)-1), key="fileB")

    con = duckdb.connect()

    # ---- 共通ユーティリティ（SELECT DISTINCTの1列目だけ返す） ----
    def list_values(pq, expr, where=None):
        q = f"SELECT DISTINCT {expr} FROM parquet_scan('{pq}')"
        if where:
            q += f" WHERE {where}"
        q += " ORDER BY 1"
        df_ = con.execute(q).df()
        if df_.empty:
            return []
        return df_.iloc[:, 0].dropna().tolist()

    # 列存在チェック
    colsA = con.execute(f"DESCRIBE SELECT * FROM parquet_scan('{file_A}')").df()["column_name"].tolist()
    colsB = con.execute(f"DESCRIBE SELECT * FROM parquet_scan('{file_B}')").df()["column_name"].tolist()
    has_vis_A = "visibility" in colsA
    has_vis_B = "visibility" in colsB
    has_pair_A = "pair_uuid" in colsA
    has_pair_B = "pair_uuid" in colsB

    # ---- t4dataset_id を共有化（A∩B）----
    t4_A_all = set(list_values(file_A, "t4dataset_id"))
    t4_B_all = set(list_values(file_B, "t4dataset_id"))
    shared_t4_ids = sorted(t4_A_all & t4_B_all)

    if not shared_t4_ids:
        st.error("AとBで共通の t4dataset_id がありません。")
        st.stop()

    selected_t4 = st.selectbox("t4dataset_id (shared)", shared_t4_ids)

    # ---- topic は単一選択・AとBで別々 ----
    topics_A = list_values(file_A, "topic_name", f"t4dataset_id='{selected_t4}'")
    topics_B = list_values(file_B, "topic_name", f"t4dataset_id='{selected_t4}'")

    # state に初期値を確実に入れる
    if "topic_A" not in st.session_state:
        st.session_state.topic_A = topics_A[0] if topics_A else None
    if "topic_B" not in st.session_state:
        st.session_state.topic_B = topics_B[0] if topics_B else None

    col_tA, col_btn, col_tB = st.columns([4,2,4])
    with col_tA:
        topic_A = st.selectbox("topic (A)", topics_A, index=topics_A.index(st.session_state.topic_A) if st.session_state.topic_A in topics_A else 0, key="topic_A")
    with col_btn:
        st.write("")
        st.write("")
        if st.button("BをAと同じtopicにする"):
            st.session_state.topic_B = st.session_state.topic_A
    with col_tB:
        topic_B = st.selectbox("topic (B)", topics_B, index=topics_B.index(st.session_state.topic_B) if st.session_state.topic_B in topics_B else 0, key="topic_B")

    # ---- label は共有（A∩B）----
    labels_A = set(list_values(file_A, "label", f"t4dataset_id='{selected_t4}'"))
    labels_B = set(list_values(file_B, "label", f"t4dataset_id='{selected_t4}'"))
    shared_labels = sorted(labels_A & labels_B)
    selected_labels = st.multiselect("label(s) (shared)", shared_labels, default=shared_labels)

    # ---- visibility を共有（両方に列がある場合のみ）----
    if has_vis_A and has_vis_B:
        vis_A = set(list_values(file_A, "COALESCE(visibility,'UNKNOWN') AS visibility", f"t4dataset_id='{selected_t4}'"))
        vis_B = set(list_values(file_B, "COALESCE(visibility,'UNKNOWN') AS visibility", f"t4dataset_id='{selected_t4}'"))
        shared_vis = sorted(vis_A & vis_B)
        selected_visibility = st.multiselect("visibility (shared)", shared_vis, default=shared_vis)
    else:
        selected_visibility = []
        if has_vis_A or has_vis_B:
            st.info("どちらか一方にしか 'visibility' 列が無いため、共有visibilityフィルタは無効です。")

    # 描画モード
    view_mode = st.radio("View mode", ["Overlay (通常)", "Overlay (Δフォーカス: Improved/Degraded)", "Side-by-side (横並び)"])
    show_diff = st.checkbox("差分レイヤ (Δ: Improved/Degraded/NewFP/FIxedFP) を重ねる", value=True)

def load_filtered_df(pq: str, t4: str, topic: str, labels, sel_vis, has_vis: bool, has_pair: bool):
    label_filter = "', '".join(labels) if labels else ""
    where = f"t4dataset_id='{t4}'"
    if topic:
        where += f" AND topic_name = '{topic}'"
    if label_filter:
        where += f" AND label IN ('{label_filter}')"
    if has_vis and sel_vis:
        vis_filter = "', '".join(sel_vis)
        where += f" AND COALESCE(visibility, 'UNKNOWN') IN ('{vis_filter}')"

    pair_select = "pair_uuid" if has_pair else "NULL AS pair_uuid"
    vis_select  = "visibility," if has_vis else ""

    q = f"""
        SELECT CAST(frame_index AS INT) AS frame_index,
               x, y, length, width, yaw, label, topic_name, source, status,
               {vis_select} {pair_select}
        FROM parquet_scan('{pq}')
        WHERE {where}
        ORDER BY frame_index
    """
    return duckdb.connect().execute(q).df()

# 共有フィルタ + 個別topic を適用
dfA = load_filtered_df(file_A, selected_t4, topic_A, selected_labels, selected_visibility, has_vis_A, has_pair_A)
dfB = load_filtered_df(file_B, selected_t4, topic_B, selected_labels, selected_visibility, has_vis_B, has_pair_B)

if dfA.empty and dfB.empty:
    st.warning("A/Bともに該当データがありません。条件を見直してください。")
    st.stop()

# 共通フレーム範囲（無ければAを優先）
fmin = int(min([x for x in [dfA.frame_index.min() if not dfA.empty else None,
                            dfB.frame_index.min() if not dfB.empty else None] if x is not None]))
fmax = int(max([x for x in [dfA.frame_index.max() if not dfA.empty else None,
                            dfB.frame_index.max() if not dfB.empty else None] if x is not None]))
frame = st.slider("Frame index", fmin, fmax, step=1)

dfA_f = dfA[dfA.frame_index == frame].copy()
dfB_f = dfB[dfB.frame_index == frame].copy()

# =============================
# 差分判定（frame単位）
# =============================
import pandas as pd

def compute_diff(dfAf, dfBf):
    """
    改善: A:FN -> B:TP
    悪化: A:TP -> B:FN
    NEW_FP: Aに無いFP（BのみFP, pair_uuid NULL）
    FIXED_FP: Bに無いFP（AのみFP, pair_uuid NULL）
    """
    cols = ["diff_type", "x", "y", "length", "width", "yaw"]
    out = []

    # --- EST側のTP/FN推移（pair_uuidで突き合わせ） ---
    estA = dfAf[(dfAf["source"] == "EST") & (dfAf["status"].isin(["TP","FN"]))].copy()
    estB = dfBf[(dfBf["source"] == "EST") & (dfBf["status"].isin(["TP","FN"]))].copy()

    # joinキー：pair_uuid が無ければ比較不能なので除外
    if "pair_uuid" not in estA.columns:
        estA["pair_uuid"] = np.nan
    if "pair_uuid" not in estB.columns:
        estB["pair_uuid"] = np.nan
    estA = estA[pd.notna(estA["pair_uuid"])]
    estB = estB[pd.notna(estB["pair_uuid"])]

    if not estA.empty or not estB.empty:
        join_keys = ["pair_uuid", "label"]
        j = estA.merge(estB, on=join_keys, how="outer", suffixes=("_A","_B"))

        for _, r in j.iterrows():
            pa = r.get("pair_uuid")
            if pd.isna(pa):
                continue
            sa = r.get("status_A")
            sb = r.get("status_B")

            # 改善/悪化の位置は「見える側」を優先（BにあるならB、無ければA）
            if sa == "FN" and sb == "TP":
                out.append({"diff_type":"IMPROVED",
                            "x": r.get("x_B", r.get("x_A")), "y": r.get("y_B", r.get("y_A")),
                            "length": r.get("length_B", r.get("length_A")),
                            "width":  r.get("width_B",  r.get("width_A")),
                            "yaw":    r.get("yaw_B",    r.get("yaw_A"))})
            elif sa == "TP" and sb == "FN":
                out.append({"diff_type":"DEGRADED",
                            "x": r.get("x_A", r.get("x_B")), "y": r.get("y_A", r.get("y_B")),
                            "length": r.get("length_A", r.get("length_B")),
                            "width":  r.get("width_A",  r.get("width_B")),
                            "yaw":    r.get("yaw_A",    r.get("yaw_B"))})

    # --- 新規/解消FP（pair_uuid が NULL の EST/FP） ---
    fpA = dfAf[(dfAf["source"] == "EST") & (dfAf["status"] == "FP")].copy()
    fpB = dfBf[(dfBf["source"] == "EST") & (dfBf["status"] == "FP")].copy()

    def fp_key(df):
        # NaN安全化
        def to_key(v):
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            return int(round(float(v)*2))
        keys = set()
        for _, rr in df.iterrows():
            keys.add((rr.label, to_key(rr.x), to_key(rr.y)))
        return keys

    keyA, keyB = fp_key(fpA), fp_key(fpB)
    new_fp_keys   = keyB - keyA
    fixed_fp_keys = keyA - keyB

    for _, r in fpB.iterrows():
        k = (r.label,
             None if pd.isna(r.x) else int(round(float(r.x)*2)),
             None if pd.isna(r.y) else int(round(float(r.y)*2)))
        if k in new_fp_keys:
            out.append({"diff_type":"NEW_FP", "x":r.x, "y":r.y,
                        "length":r.length, "width":r.width, "yaw":r.yaw})

    for _, r in fpA.iterrows():
        k = (r.label,
             None if pd.isna(r.x) else int(round(float(r.x)*2)),
             None if pd.isna(r.y) else int(round(float(r.y)*2)))
        if k in fixed_fp_keys:
            out.append({"diff_type":"FIXED_FP", "x":r.x, "y":r.y,
                        "length":r.length, "width":r.width, "yaw":r.yaw})

    # ★ 重要：空でも必ず列を揃えて返す
    return pd.DataFrame(out, columns=cols)

def compute_diff_all(dfA_all, dfB_all):
    if (dfA_all is None or dfA_all.empty) and (dfB_all is None or dfB_all.empty):
        return pd.DataFrame(columns=["diff_type","x","y","length","width","yaw","frame_index"])
    frames = sorted(set(dfA_all["frame_index"].unique()).union(set(dfB_all["frame_index"].unique())))
    outs = []
    for fr in frames:
        da = dfA_all[dfA_all["frame_index"] == fr]
        db = dfB_all[dfB_all["frame_index"] == fr]
        d = compute_diff(da, db)            # 既存のフレーム単位関数を使う
        if not d.empty:
            d = d.assign(frame_index=int(fr))
            outs.append(d)
    if not outs:
        return pd.DataFrame(columns=["diff_type","x","y","length","width","yaw","frame_index"])
    return pd.concat(outs, ignore_index=True)

# 差分計算
df_diff = compute_diff(dfA_f, dfB_f)
imp, deg, newfp, fixfp = summarize_diff(df_diff)

df_diff_all = compute_diff_all(dfA, dfB)
imp_all, deg_all, newfp_all, fixfp_all = summarize_diff(df_diff_all)


# =============================
# 描画
# =============================
palette = get_color_map()

if view_mode == "Overlay (通常)":
    fig = go.Figure()
    if not dfA_f.empty:
        plot_frame(fig, dfA_f, palette, tag="A", opacity=0.55, dash=None)
    if not dfB_f.empty:
        # Bは点線で差別化
        plot_frame(fig, dfB_f, palette, tag="B", opacity=0.55, dash="dash")
    if show_diff and not df_diff.empty:
        plot_diff(fig, df_diff, palette)
    add_ego(fig)
    fig.update_layout(
        title=f"A: {os.path.basename(file_A)} / {selected_t4} / {topic_A}  vs  B: {os.path.basename(file_B)} / {selected_t4} / {topic_B} | Frame {frame} "
              f"| Δ(Improved:{imp}, Degraded:{deg}, NewFP:{newfp}, FixedFP:{fixfp})",
        xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
        yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
        width=1100, height=900
    )
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Overlay (Δフォーカス: Improved/Degraded)":
    fig = go.Figure()
    # 背景としてA/Bを淡く（Bは点線）
    if not dfA_f.empty:
        plot_frame(fig, dfA_f, palette, tag="A", opacity=0.15, dash=None, showlegend=False)
    if not dfB_f.empty:
        plot_frame(fig, dfB_f, palette, tag="B", opacity=0.15, dash="dash", showlegend=False)
    # 改善/悪化のみ強調描画
    plot_diff(fig, df_diff, palette, types=["IMPROVED","DEGRADED"], width=4, opacity=0.75)
    add_ego(fig)
    fig.update_layout(
        title=f"Δ Focus (Improved/Degraded) | A: {topic_A} vs B: {topic_B} | Frame {frame} "
              f"| Δ(Imp:{imp}, Deg:{deg})",
        xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
        yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
        width=1100, height=900
    )
    st.plotly_chart(fig, use_container_width=True)

else:  # Side-by-side
    c1, c2 = st.columns(2)
    with c1:
        figA = go.Figure()
        if not dfA_f.empty:
            plot_frame(figA, dfA_f, palette, tag="A")
        add_ego(figA)
        figA.update_layout(
            title=f"A | {os.path.basename(file_A)} / {selected_t4} / {topic_A} | Frame {frame}",
            xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
            yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
            width=700, height=800
        )
        st.plotly_chart(figA, use_container_width=True)
    with c2:
        figB = go.Figure()
        if not dfB_f.empty:
            plot_frame(figB, dfB_f, palette, tag="B", dash="dash")
        add_ego(figB)
        if show_diff and not df_diff.empty:
            plot_diff(figB, df_diff, palette)  # 右側に差分を重ねて見せるのもアリ
        figB.update_layout(
            title=f"B | {os.path.basename(file_B)} / {selected_t4} / {topic_B} | Frame {frame} "
                  f"| Δ(Improved:{imp}, Degraded:{deg}, NewFP:{newfp}, FixedFP:{fixfp})",
            xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
            yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
            width=700, height=800
        )
        st.plotly_chart(figB, use_container_width=True)

# =============================
# 参考: このフレームのサマリ
# =============================
def get_valid_datanum(df):
    if df is None or df.empty:
        return 0, 0
    return len(df), int((df["length"] > 0).astype(int).add((df["width"] > 0).astype(int)).eq(2).sum())

col1, col2, col3 = st.columns(3)
with col1:
    trA, vrA = get_valid_datanum(dfA_f)
    st.metric("A this frame: Total / Valid", f"{trA} / {vrA}")
with col2:
    trB, vrB = get_valid_datanum(dfB_f)
    st.metric("B this frame: Total / Valid", f"{trB} / {vrB}")
with col3:
    st.metric("Δ (Imp / Deg / NewFP / FixFP)", f"{imp} / {deg} / {newfp} / {fixfp}")


# =============================
# 参考: 全フレームのサマリ
# =============================
c1, c2 = st.columns(2)
with c1:
    st.metric("Δ(全フレーム合算) Imp / Deg", f"{imp_all} / {deg_all}")
with c2:
    st.metric("Δ(全フレーム合算) NewFP / FixedFP", f"{newfp_all} / {fixfp_all}")

if not df_diff_all.empty:
    byf = df_diff_all.groupby(["frame_index","diff_type"]).size().reset_index(name="count")
    st.dataframe(byf, use_container_width=True)
else:
    st.info("全フレームでの差分は検出されませんでした。")