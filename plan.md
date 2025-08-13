# Grafana 可視化作業の外観と前提知識まとめ

## 1. 現状の到達点

- 元のCSVをParquet化（単一ファイル形式）
- GrafanaにDuckDB経由でParquetを接続済み
- Dashboardが表示可能な状態まで構築済み

---

## 2. 作業の外観（可視化フェーズ）

### フェーズA: ビュー/クエリ準備

- DuckDB上に解析用ビューを定義
  - **view\_eval\_flat**: 従来DF互換ビュー
  - **view\_tpr\_fpr\_by\_class\_dist\_topic**: TPR/FPR集計用
  - **view\_error\_stats\_by\_class\_dist**: 誤差統計用
  - **view\_regression\_by\_topic\_run**: ラン間比較用
- Grafanaのデータソース（DuckDB）にビューが参照可能な状態にする

### フェーズB: Dashboard設計とパネル追加

- **Dashboard #1: 品質サマリ**
  - ヒートマップ（距離×クラスのTPR）
  - topic別TPR棒グラフ
  - 集計KPIテーブル
- **Dashboard #2: 誤差特性**
  - 距離別誤差統計（p50/p90など）
  - 誤差分布箱ひげ
  - 誤差統計テーブル
- **Dashboard #3: ラン比較**
  - ΔTPRヒートマップ
  - TPR\_A vs TPR\_B散布図
  - ΔTPRランキング表

### フェーズC: 変数設定とフィルタUI

- Grafana Variables
  - `${run_id}`（複数選択可）
  - `${class}`（複数選択可）
  - `${topic_name}`
  - `${distance_bin}`
  - `${eval_config_hash}`
- パネルのクエリで変数を利用し、インタラクティブなフィルタを実装

---

## 3. 前提として必須の知識

### データ構造

- 列： \
   `['unix_time', 'frame_id', 'x', 'y', 'z', 'yaw', 'length', 'width', 'height', 'shape_type', 'vx', 'vy', 'confidence', 'label', 'pointcloud_num', 'uuid', 'visibility', 'x_error', 'y_error', 'z_error', 'yaw_error', 'vx_error', 'vy_error', 'speed_error', 'center_distance', 'plane_distance', 'pair_dt_sec', 'pair_uuid', 'frame_index', 'source', 'status', 'dx_min', 'dy_min', 'topic_name', 't4dataset_id', 'suite_name', 't4dataset_name']`  
- **カラムの意味のうち特に説明が必要そうなもの**:
  - `unix_time`：タイムスタンプ
  - `frame_id`：ROSやシミュレータでのフレーム識別子
  - `frame_index`：解析時のフレーム番号（連番）
  - `pair_dt_sec`：GTと予測の時間差
  - `status` (TP/FP/FN)
  - `label` (クラスラベル)
  - `center_distance`/`plane_distance` (ペアとの距離)
  - 誤差系カラム (`*_error`)
  - `pair_uuid`（対応付けられたGTのUUID）
  - `visibility`（GTの可視性）
  - `status`（TP / FP / FN / その他）
  - `t4dataset_id`, `suite_name`, `t4dataset_name`（データセット情報）
  - `topic_name`（検出元のROSトピック）
  - `source`（objectがGT/EST由来の種別）
  - 基本GTとESTのMatchingがとった後のデータである

### DuckDBクエリ基礎

- `SELECT ... FROM 'file.parquet' WHERE ... GROUP BY ...`
- ビュー作成: `CREATE VIEW view_name AS SELECT ...`
- 変数使用時はGrafanaの`${var_name}`をWHERE句やSELECTに埋め込む

### Grafana可視化基本

- **パネルタイプ**: Table / Bar Chart / Heatmap / Time Series / Scatter
- **変数連動**: Variables設定 → パネルクエリに埋め込む
- **カラースケール**: 0〜1固定（TPRなどの比率）
- **注釈**: GT件数などをパネル内に表示

### 集計ロジックの前提

- **TPR = TP / GT**（分母=TP+FN）
- **FPR = FP / (FP+TN?)**（定義はプロジェクト内で統一）
- 距離ビンの定義と固定（例: [0,10), [10,20), ...）
- 誤差統計はTPのみ対象（FP/FNはNaN）

---

## 4. 複数チャットでの作業分担例

- **チャットA**: Dashboard #1 品質サマリ（ビュー作成＋パネル作成）
- **チャットB**: Dashboard #2 誤差特性（統計ロジック確認＋可視化）
- **チャットC**: Dashboard #3 ラン比較（比較SQL＋パネル設計）
- **チャットD**: 変数設計・フィルタUIの共通化
- **チャットE**: データ更新フロー・再計算手順

---

## 5. 最初のアクション

1. 単一Parquetから`view_tpr_fpr_by_class_dist_topic`をDuckDBで作る
2. Grafanaに変数`${run_id}`, `${class}`, `${topic_name}`を設定
3. ヒートマップ＋棒グラフ＋KPIテーブルをDashboard #1に追加
4. チームでレビューし、距離ビンや閾値設定を固める
