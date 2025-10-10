import duckdb
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import glob
import os
from typing import Tuple, List, Dict, Optional
# 既存 import 群の下に追記
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False
    plotly_events = None

_duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Return a shared DuckDB connection for all queries."""
    global _duckdb_connection
    if _duckdb_connection is None:
        _duckdb_connection = duckdb.connect()
    return _duckdb_connection

# セッションキーを明示的に分離
FRAME_VALUE_KEY  = "frame_value"   # ← 正規のフレーム値（アプリのソース・オブ・トゥルース）
FRAME_SLIDER_KEY = "frame_slider"  # ← スライダーウィジェット用のキー（※frameは使わない）


st.set_page_config(layout="wide")
st.title("BEV Bounding Box Viewer (A/B Comparison)")

# =============================
# 汎用ユーティリティ
# =============================
def _base_diff(t: str) -> str:
    """'GT_IMPROVED' などを 'IMPROVED' に正規化。未知はそのまま返す。"""
    if not isinstance(t, str):
        return t
    if t.startswith("GT_"):
        return t[3:]
    if t.startswith("EST_"):
        return t[4:]
    return t

# def rotated_rect(x: float, y: float, length: float, width: float, yaw: float) -> Tuple[np.ndarray, np.ndarray]:
#     """yawはラジアン。中心(x,y)、長さlength(前後)、幅width(左右)。BEVの進行方向合わせのため+pi/2回転。"""
#     dx, dy = length / 2.0, width / 2.0
#     corners = np.array([[-dx, -dy], [-dx,  dy], [ dx,  dy], [ dx, -dy], [-dx, -dy]])
#     c, s = np.cos(yaw + np.pi/2), np.sin(yaw + np.pi/2)
#     rot = np.array([[c, -s], [s, c]])
#     rotated = corners @ rot.T
#     return rotated[:, 0] + x, rotated[:, 1] + y

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
               opacity: float = 0.55, dash: str | None = None, showlegend: bool = True, show_invalid: bool = False):
    """tag は 'A' or 'B'。ESTのTP/FPにA/Bのサフィックスを付けて色分け。dash='dash' などでBの描画を差別化可。"""
    if df_frame.empty:
        return

    df = df_frame.copy()

    # --- 1. ベクタライズ処理で描画に必要な情報を事前に追加 ---
    cond_est_tp_fp = (df['source'] == 'EST') & (df['status'].isin(['TP', 'FP']))
    df['status_key'] = np.where(cond_est_tp_fp, df['status'] + f"_{tag}", df['status'])
    df['color_key'] = list(zip(df['source'], df['status_key']))
    df['name'] = df['source'] + '/' + df['status_key'] + f"_{tag}"
    df['color'] = df['color_key'].map(palette).fillna("#999999")
    
    # --- 2. 共通のホバーテンプレートを定義 ---
    hovertemplate = (
        "Label: %{customdata[0]}<br>"
        "size: %{customdata[1]:.2f} x %{customdata[2]:.2f}<br>"
        "X: %{x}<br>"
        "Y: %{y}<br>"
        "<extra></extra>" # Plotlyの余分な情報を非表示にする
    )

    # --- 3. オブジェクトの形状に応じてDataFrameを分割し、描画 ---
    mask_both_invalid = (df['length'] <= 0) & (df['width'] <= 0)
    mask_one_invalid = ((df['length'] <= 0) | (df['width'] <= 0)) & ~mask_both_invalid
    mask_valid = (df['length'] > 0) & (df['width'] > 0)

    shown = set()

    # --- Case A: 両方が無効なオブジェクト (xマーカー) ---
    if show_invalid and mask_both_invalid.any():
        df_invalid = df[mask_both_invalid]
        fig.add_trace(go.Scatter(
            x=df_invalid['x'], y=df_invalid['y'],
            mode="markers",
            marker=dict(symbol="x", size=8, color=df_invalid['color']),
            opacity=0.9, showlegend=False, name="invalid_marker",
            hovertemplate=hovertemplate,
            customdata=df_invalid[['label', 'length', 'width']].values
        ))

    # --- Case B: 片方が無効なオブジェクト (円マーカー) ---
    if mask_one_invalid.any():
        df_cylinder = df[mask_one_invalid]
        for name, group in df_cylinder.groupby('name'):
            lg = (name not in shown) and showlegend
            fig.add_trace(go.Scatter(
                x=group['x'], y=group['y'],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=group[['length', 'width']].max(axis=1),
                    color=group.iloc[0]['color']
                ),
                opacity=opacity, name=name, legendgroup=name, showlegend=lg,
                hovertemplate=hovertemplate,
                customdata=group[['label', 'length', 'width']].values
            ))
            shown.add(name)

    # --- Case C: 有効なオブジェクト (矩形ポリゴン) ---
    if mask_valid.any():
        df_poly = df[mask_valid]
        for name, group in df_poly.groupby('name'):
            lg = (name not in shown) and showlegend
            for _, r in group.iterrows():
                x_poly, y_poly = rotated_rect(r.x, r.y, r.length, r.width, r.yaw)
                fig.add_trace(go.Scatter(
                    x=x_poly, y=y_poly,
                    mode='lines', fill='toself', opacity=opacity,
                    line=dict(color=r['color'], dash=dash),
                    name=name, legendgroup=name, showlegend=lg,
                    hovertemplate=hovertemplate,
                    # customdataは各ポリゴンに個別に設定
                    customdata=np.array([[r.label, r.length, r.width]] * len(x_poly))
                ))
                lg = False
            shown.add(name)

def plot_diff(fig: go.Figure, df_diff, palette, types: List[str] | None = None, width: int = 3, opacity: float = 0.45):
    """types はベース型で指定（例: ['IMPROVED','DEGRADED']）。GT_/EST_ は内部で正規化してフィルタ。"""
    if df_diff is None or df_diff.empty or "diff_type" not in df_diff.columns:
        return

    # 正規化列を作成
    ddf = df_diff.copy()
    ddf["_base"] = ddf["diff_type"].map(_base_diff)

    if types is not None:
        ddf = ddf[ddf["_base"].isin(types)]
        if ddf.empty:
            return

    shown = set()
    for _, r in ddf.iterrows():
        x_poly, y_poly = rotated_rect(r.x, r.y, r.length, r.width, r.yaw)
        base = r._base                       # IMRPOVED / DEGRADED / NEW_FP / FIXED_FP
        color = palette.get(("DIFF", base), "#777777")
        # レジェンドは GT/EST を見分けたいので diff_type そのまま使う
        name = f"Δ {r.diff_type.replace('_', ' ')}"  # 例: 'Δ GT IMPROVED'
        show = name not in shown
        shown.add(name)
        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly, mode='lines', fill='toself', opacity=opacity,
            line=dict(color=color, width=width),
            name=name, showlegend=show, legendgroup=name
        ))

def summarize_diff(df_diff):
    if df_diff is None or df_diff.empty or "diff_type" not in df_diff.columns:
        return 0, 0, 0, 0
    s = df_diff["diff_type"].map(_base_diff).value_counts()
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

    con = get_duckdb_connection()

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

    # --- invalidオブジェクト表示オプション ---
    show_invalid = st.sidebar.checkbox("Show invalid (zero-size) objects", value=False)

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
               x, y, length, width, yaw, label, topic_name, source, status, uuid,
               {vis_select} {pair_select}
        FROM parquet_scan('{pq}')
        WHERE {where}
        ORDER BY frame_index
    """
    con = get_duckdb_connection()
    return con.execute(q).df()


# =============================
# 差分判定（frame単位）
# =============================
def _norm_status(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def compute_diff(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    gt_id_col: str = "uuid",         # UIで選んで渡す（例: "object_id" など）。NoneならGT軸はスキップ
    est_pair_col: str = "pair_uuid",  # ESTの対応ID列（既定: pair_uuid）
    include_extra: Optional[List[str]] = None,  # 例: ["frame_index","label"] を含めて返したい時
) -> pd.DataFrame:
    """
    入力: 同一 frame_index / t4dataset_id でスライス済みの dfA/dfB
    出力: 必須6列 + （必要なら extra）を持つ DataFrame
      必須6列: diff_type, x, y, length, width, yaw
      diff_type ∈ { GT_IMPROVED, GT_DEGRADED, EST_IMPROVED, EST_DEGRADED, EST_NEW_FP, EST_FIXED_FP }
    """
    cols = ["diff_type","x","y","length","width","yaw"]
    extras = include_extra or []
    rows = []

    # ---------- GT軸: 同一GT-IDでの TP/FN 推移 ----------
    if gt_id_col is not None and gt_id_col in dfA.columns and gt_id_col in dfB.columns:
        gtA = dfA[(dfA["source"]=="GT") & (dfA["status"].isin(["TP","FN"]))].copy()
        gtB = dfB[(dfB["source"]=="GT") & (dfB["status"].isin(["TP","FN"]))].copy()
        if not gtA.empty and not gtB.empty:
            gtA["status"] = _norm_status(gtA["status"])
            gtB["status"] = _norm_status(gtB["status"])
            a = gtA[[gt_id_col,"frame_index","status","x","y","length","width","yaw"] + [c for c in extras if c in gtA.columns]].rename(
                columns={gt_id_col:"gt_id","status":"status_A","x":"x_A","y":"y_A","length":"length_A","width":"width_A","yaw":"yaw_A"}
            )
            b = gtB[[gt_id_col,"frame_index","status","x","y","length","width","yaw"] + [c for c in extras if c in gtB.columns]].rename(
                columns={gt_id_col:"gt_id","status":"status_B","x":"x_B","y":"y_B","length":"length_B","width":"width_B","yaw":"yaw_B"}
            )
            j = a.merge(b, on=["gt_id","frame_index"] + [c for c in extras if c != "frame_index"], how="inner", suffixes=("_A","_B"))

            for _, r in j.iterrows():
                sa, sb = r["status_A"], r["status_B"]
                if sa=="FN" and sb=="TP":
                    rows.append(dict(
                        diff_type="GT_IMPROVED",
                        x=r.get("x_B", r.get("x_A")), y=r.get("y_B", r.get("y_A")),
                        length=r.get("length_B", r.get("length_A")),
                        width=r.get("width_B", r.get("width_A")),
                        yaw=r.get("yaw_B", r.get("yaw_A")),
                        **{k: r[k] for k in extras if k in r}
                    ))
                elif sa=="TP" and sb=="FN":
                    rows.append(dict(
                        diff_type="GT_DEGRADED",
                        x=r.get("x_A", r.get("x_B")), y=r.get("y_A", r.get("y_B")),
                        length=r.get("length_A", r.get("length_B")),
                        width=r.get("width_A", r.get("width_B")),
                        yaw=r.get("yaw_A", r.get("yaw_B")),
                        **{k: r[k] for k in extras if k in r}
                    ))

    # ---------- EST軸: 同一pairでの TP/FP 推移 ----------
    estA = dfA[(dfA["source"]=="EST") & (dfA["status"].isin(["TP","FP"]))].copy()
    estB = dfB[(dfB["source"]=="EST") & (dfB["status"].isin(["TP","FP"]))].copy()
    if est_pair_col not in estA.columns: estA[est_pair_col] = np.nan
    if est_pair_col not in estB.columns: estB[est_pair_col] = np.nan
    if not estA.empty or not estB.empty:
        estA["status"] = _norm_status(estA["status"])
        estB["status"] = _norm_status(estB["status"])

        a = estA[estA[est_pair_col].notna()][[est_pair_col,"frame_index","status","x","y","length","width","yaw"] + [c for c in extras if c in estA.columns]].rename(
            columns={est_pair_col:"pair_uuid","status":"status_A","x":"x_A","y":"y_A","length":"length_A","width":"width_A","yaw":"yaw_A"}
        )
        b = estB[estB[est_pair_col].notna()][[est_pair_col,"frame_index","status","x","y","length","width","yaw"] + [c for c in extras if c in estB.columns]].rename(
            columns={est_pair_col:"pair_uuid","status":"status_B","x":"x_B","y":"y_B","length":"length_B","width":"width_B","yaw":"yaw_B"}
        )
        jb = a.merge(b, on=["pair_uuid","frame_index"] + [c for c in extras if c != "frame_index"], how="inner", suffixes=("_A","_B"))

        for _, r in jb.iterrows():
            sa, sb = r["status_A"], r["status_B"]
            if sa=="FP" and sb=="TP":
                rows.append(dict(
                    diff_type="EST_IMPROVED",
                    x=r.get("x_B", r.get("x_A")), y=r.get("y_B", r.get("y_A")),
                    length=r.get("length_B", r.get("length_A")),
                    width=r.get("width_B", r.get("width_A")),
                    yaw=r.get("yaw_B", r.get("yaw_A")),
                    **{k: r[k] for k in extras if k in r}
                ))
            elif sa=="TP" and sb=="FP":
                rows.append(dict(
                    diff_type="EST_DEGRADED",
                    x=r.get("x_A", r.get("x_B")), y=r.get("y_A", r.get("y_B")),
                    length=r.get("length_A", r.get("length_B")),
                    width=r.get("width_A", r.get("width_B")),
                    yaw=r.get("yaw_A", r.get("yaw_B")),
                    **{k: r[k] for k in extras if k in r}
                ))

        # NEW_FP / FIXED_FP（pair_uuidがNaNで来るFPはここでは扱わない） 
        # This part won't work currently because duckdb does not support isin() with NaN
        a_fp_ids = set(estA[(estA["status"]=="FP") & estA[est_pair_col].notna()][est_pair_col].astype(str))
        b_fp_ids = set(estB[(estB["status"]=="FP") & estB[est_pair_col].notna()][est_pair_col].astype(str))
        new_fp_ids   = b_fp_ids - a_fp_ids
        fixed_fp_ids = a_fp_ids - b_fp_ids

        if new_fp_ids:
            for _, r in estB[(estB["status"]=="FP") & (estB[est_pair_col].astype(str).isin(new_fp_ids))].iterrows():
                rows.append(dict(
                    diff_type="EST_NEW_FP",
                    x=r.x, y=r.y, length=r.length, width=r.width, yaw=r.yaw,
                    **{k: r[k] for k in extras if k in r}
                ))
        if fixed_fp_ids:
            for _, r in estA[(estA["status"]=="FP") & (estA[est_pair_col].astype(str).isin(fixed_fp_ids))].iterrows():
                rows.append(dict(
                    diff_type="EST_FIXED_FP",
                    x=r.x, y=r.y, length=r.length, width=r.width, yaw=r.yaw,
                    **{k: r[k] for k in extras if k in r}
                ))

    # 6列(+必要ならextra)で返す
    df_out = pd.DataFrame(rows)
    want_cols = cols + [c for c in extras if c in df_out.columns]
    return df_out[want_cols] if not df_out.empty else pd.DataFrame(columns=want_cols)

def compute_diff_all(dfA_all, dfB_all):
    if (dfA_all is None or dfA_all.empty) and (dfB_all is None or dfB_all.empty):
        return pd.DataFrame(columns=["diff_type","x","y","length","width","yaw","frame_index"])
    frame_sources = []
    if dfA_all is not None and "frame_index" in dfA_all.columns:
        frame_sources.append(dfA_all["frame_index"])
    if dfB_all is not None and "frame_index" in dfB_all.columns:
        frame_sources.append(dfB_all["frame_index"])
    if not frame_sources:
        return pd.DataFrame(columns=["diff_type","x","y","length","width","yaw","frame_index"])
    frames = pd.concat(frame_sources, ignore_index=True).dropna().unique()
    results = [
        diff.assign(frame_index=int(fr))
        for fr in sorted(frames)
        if not (diff := compute_diff(
            dfA_all[dfA_all["frame_index"] == fr],
            dfB_all[dfB_all["frame_index"] == fr]
        )).empty
    ]
    if not results:
        return pd.DataFrame(columns=["diff_type","x","y","length","width","yaw","frame_index"])
    return pd.concat(results, ignore_index=True)

# =============================
# 共有フィルタ + 個別topic を適用
dfA = load_filtered_df(file_A, selected_t4, topic_A, selected_labels, selected_visibility, has_vis_A, has_pair_A)
dfB = load_filtered_df(file_B, selected_t4, topic_B, selected_labels, selected_visibility, has_vis_B, has_pair_B)
df_diff_all = compute_diff_all(dfA, dfB)
imp_all, deg_all, newfp_all, fixfp_all = summarize_diff(df_diff_all)

if dfA.empty and dfB.empty:
    st.warning("A/Bともに該当データがありません。条件を見直してください。")
    st.stop()

# 共通フレーム範囲（無ければAを優先）
fmin = int(min([x for x in [dfA.frame_index.min() if not dfA.empty else None,
                            dfB.frame_index.min() if not dfB.empty else None] if x is not None]))
fmax = int(max([x for x in [dfA.frame_index.max() if not dfA.empty else None,
                            dfB.frame_index.max() if not dfB.empty else None] if x is not None]))

# --- 初期化（1回だけ） ---
if FRAME_VALUE_KEY not in st.session_state:
    st.session_state[FRAME_VALUE_KEY] = int(fmin)
if FRAME_SLIDER_KEY not in st.session_state:
    st.session_state[FRAME_SLIDER_KEY] = int(fmin)

# --- クリックイベントなどで "次回反映予定" の値が入っている場合 ---
# この時点ではウィジェットまだ作られていないため、直接代入してOK
if FRAME_SLIDER_KEY + "_next" in st.session_state:
    st.session_state[FRAME_SLIDER_KEY] = int(
        max(fmin, min(fmax, st.session_state.pop(FRAME_SLIDER_KEY + "_next")))
    )
else:
    # 通常時は正規値に追従させる（クリック→rerun直後に効く）
    st.session_state[FRAME_SLIDER_KEY] = int(
        max(fmin, min(fmax, st.session_state[FRAME_VALUE_KEY]))
    )

def _on_frame_slider_change():
    # ウィジェット変更 → 正規値へ反映
    st.session_state[FRAME_VALUE_KEY] = int(st.session_state[FRAME_SLIDER_KEY])

# スライダー本体（keyはFRAME_SLIDER_KEY）
st.slider("Frame index", fmin, fmax, step=1,
          key=FRAME_SLIDER_KEY, on_change=_on_frame_slider_change)

# 以降は正規値を参照
frame = int(st.session_state[FRAME_VALUE_KEY])


dfA_f = dfA[dfA.frame_index == frame].copy()
dfB_f = dfB[dfB.frame_index == frame].copy()

# 差分計算
df_diff = compute_diff(dfA_f, dfB_f)
imp, deg, newfp, fixfp = summarize_diff(df_diff)


# =============================
# 描画
# =============================
palette = get_color_map()
if "bev_fig_cache" not in st.session_state:
    st.session_state.bev_fig_cache = {}

def get_bev_figure(view_mode: str):
    cache = st.session_state.bev_fig_cache
    if view_mode not in cache:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
            yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
            width=1100, height=900,
            uirevision="bev_fixed",
        )
        cache[view_mode] = fig
    return cache[view_mode]

if view_mode == "Overlay (通常)":
    fig = get_bev_figure(view_mode)
    fig.data = []  # 既存キャッシュをクリア
    if not dfA_f.empty:
        plot_frame(fig, dfA_f, palette, tag="A", opacity=0.55, dash=None, show_invalid=show_invalid)
    if not dfB_f.empty:
        # Bは点線で差別化
        plot_frame(fig, dfB_f, palette, tag="B", opacity=0.55, dash="dash", show_invalid=show_invalid)
    if show_diff and not df_diff.empty:
        plot_diff(fig, df_diff, palette)
    add_ego(fig)
    fig.update_layout(
        title=f"A: {os.path.basename(file_A)} / {selected_t4} / {topic_A}  vs  B: {os.path.basename(file_B)} / {selected_t4} / {topic_B} | Frame {frame} "
              f"| Δ(Improved:{imp}, Degraded:{deg}, NewFP:{newfp}, FixedFP:{fixfp})",
        xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
        yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
        width=1100, height=900,
        uirevision="bev_view",
    )
    st.plotly_chart(fig, use_container_width=True, key="overlay_normal")

elif view_mode == "Overlay (Δフォーカス: Improved/Degraded)":
    fig = get_bev_figure(view_mode)
    fig.data = []  # 既存キャッシュをクリア
    # 背景としてA/Bを淡く（Bは点線）
    if not dfA_f.empty:
        plot_frame(fig, dfA_f, palette, tag="A", opacity=0.15, dash=None, showlegend=False, show_invalid=show_invalid)
    if not dfB_f.empty:
        plot_frame(fig, dfB_f, palette, tag="B", opacity=0.15, dash="dash", showlegend=False, show_invalid=show_invalid)
    # 改善/悪化のみ強調描画
    plot_diff(fig, df_diff, palette, types=["IMPROVED","DEGRADED"], width=4, opacity=0.75)
    add_ego(fig)
    fig.update_layout(
        title=f"Δ Focus (Improved/Degraded) | A: {topic_A} vs B: {topic_B} | Frame {frame} "
              f"| Δ(Imp:{imp}, Deg:{deg})",
        xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
        yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
        width=1100, height=900,
        uirevision="bev_view",
    )
    st.plotly_chart(fig, use_container_width=True, key="overlay_diff_focus")

else:  # Side-by-side
    c1, c2 = st.columns(2)
    with c1:
        figA = get_bev_figure("side_A")
        figA.data = []  # 既存キャッシュをクリア
        if not dfA_f.empty:
            plot_frame(figA, dfA_f, palette, tag="A", show_invalid=show_invalid)
        add_ego(figA)
        figA.update_layout(
            title=f"A | {os.path.basename(file_A)} / {selected_t4} / {topic_A} | Frame {frame}",
            xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
            yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
            width=700, height=800,
            uirevision="bev_view",
        )
        st.plotly_chart(figA, use_container_width=True, key="side_A")
    with c2:
        figB = get_bev_figure("side_B")
        figB.data = []  # 既存キャッシュをクリア
        if not dfB_f.empty:
            plot_frame(figB, dfB_f, palette, tag="B", dash="dash", show_invalid=show_invalid)
        add_ego(figB)
        if show_diff and not df_diff.empty:
            plot_diff(figB, df_diff, palette)  # 右側に差分を重ねて見せるのもアリ
        figB.update_layout(
            title=f"B | {os.path.basename(file_B)} / {selected_t4} / {topic_B} | Frame {frame} "
                  f"| Δ(Improved:{imp}, Degraded:{deg}, NewFP:{newfp}, FixedFP:{fixfp})",
            xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
            yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
            width=700, height=800,
            uirevision="bev_view",
        )
        st.plotly_chart(figB, use_container_width=True, key="side_B")

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
# フレーム別 diff_type 件数（簡素な折れ線）
# =============================
# =============================
# フレーム別 diff_type 件数（折れ線）＋ クリックでフレーム移動（任意）
# =============================
if not df_diff_all.empty:
    # frame_index × diff_type の件数ピボット
    df_line = (
        df_diff_all
        .groupby(["frame_index", "diff_type"]).size()
        .unstack(fill_value=0)              # 列: diff_type（GT_* / EST_* そのまま）
        .sort_index()
    )

    # 折れ線プロット
    palette_diff = get_color_map()
    fig_counts = go.Figure([
        go.Scatter(
            x=df_line.index,
            y=df_line[col],
            mode="lines+markers",
            name=col.replace("_", " "),
            line=dict(color=palette_diff.get(("DIFF", col.split("_", 1)[-1]), "#777777"))
        )
        for col in df_line.columns
    ])

    # 現在フレームの縦線（UI用）
    cur_frame = int(st.session_state.get(FRAME_VALUE_KEY, fmin))
    fig_counts.add_vline(
        x=cur_frame, line_width=2, line_dash="dash", line_color="black",
        annotation_text=f"frame={cur_frame}", annotation_position="top left"
    )

    if HAS_PLOTLY_EVENTS:
        clicks = plotly_events(
            fig_counts,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=360,
            key="counts_clicks",
        )
        # --- クリックイベントが前回処理済みなら無視する ---
        if clicks and (x := clicks[0].get("x")) is not None:
            new_val = int(x)
            # 「前回クリック値」が存在し、同じならスキップ
            prev_val = st.session_state.get("last_click_frame")
            if new_val != prev_val:
                st.session_state["last_click_frame"] = new_val
                st.session_state[FRAME_VALUE_KEY] = new_val
                st.session_state[FRAME_SLIDER_KEY + "_next"] = new_val
                st.rerun()
    else:
        st.plotly_chart(fig_counts, use_container_width=True)
else:
    st.info("全フレームでの差分は検出されませんでした。")
