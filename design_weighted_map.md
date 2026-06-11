# Rematch + Distance-Weighted mAP — 設計書

## 1. 概要

本ドキュメントは以下の 2 段階パイプラインの設計を記述する。

```
current.csv (perception_eval 出力)
    │
    ▼ [Stage 1: Re-matching] ← 任意ステップ
rematched.csv (TP/FP/FN が再計算済み)
    │
    ▼ [Stage 2: Weighted mAP]
weighted_map_results/
```

Stage 1 (Rematch) は任意。`current.csv` をそのまま Stage 2 に投入してもよい。

---

## 2. Stage 1 — Re-matching 設計

### 2.1 目的

`perception_eval` が生成した `current.csv` の TP/FP/FN 判定は、オリジナルの評価パイプラインの設定（距離閾値・ラベルマッチング・信頼度フィルター）に依存する。  
Re-matching は同 CSV のポジション情報のみを使い、**任意の閾値設定で再マッチングを行い新しい TP/FP/FN を生成する**ことを目的とする。

### 2.2 入出力

| | |
|---|---|
| 入力 | `current.csv` |
| 出力 | `rematched.csv`（同スキーマ、status / x_error / y_error が上書き） |

### 2.3 アルゴリズム

各 `unix_time` × `label` グループで GT と EST の **BEV 2D 距離コスト行列**を構築し、ハンガリアン法（または greedy）で最適割り当てを求める。

```
cost[i,j] = sqrt((gt_x[i] - est_x[j])^2 + (gt_y[i] - est_y[j])^2)
```

距離が閾値を超えるペアは `cost = ∞`（割り当て不可）として扱う。

### 2.4 主要パラメータ（RematchConfig）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `distance_thresholds_m` | `{__default__: 2.0}` | クラスごとのマッチング距離閾値（m） |
| `label_match` | `"strict"` | `strict` / `agnostic` / `grouped` |
| `label_groups` | null | `grouped` 時のクラスグループ定義 |
| `confidence_min` | 0.0 | マッチングプールに入る EST の最低 confidence |
| `visibility_exclude_from_gt` | `[]` | マッチングプールから除く GT の visibility |
| `algorithm` | `"hungarian"` | `hungarian` / `greedy` |

### 2.5 制約・既知の問題

- `yaw_error` / `speed_error` は再計算不可（生速度情報が CSV に存在しない）。再マッチング後はこれらが NaN になる → NDS_safety の AOE/AVE スコアが計算不能
- `pair_uuid` が CSV に存在しないため、可視性フィルターを厳しくすると AP が若干過大評価になる
- ラベルの表記揺れ（`motorbike` / `bicycle`）は `label_groups` で対処する必要がある

### 2.6 実行例

```bash
python script/run_rematch.py --config configs/rematch_default.yaml
python script/run_rematch.py --config configs/rematch_strict.yaml
```

---

## 3. Stage 2 — Distance-To-Vehicle Weighted mAP 設計

### 3.1 目的

通常の mAP では「近距離で危険な物体」と「遠距離の物体」が同等に扱われる。  
**DTV Weighted mAP** は GT/EST のマッチング単位で距離重みを付与することで、近距離の見落としを強くペナルティ化し、planning 上の重要度に近い精度指標を実現する。

本設計は PoC 指示書（Distance-To-Vehicle weighted mAP 計算）に基づくが、**入力データを parquet ではなく `current.csv` / `rematched.csv`** に適応している（マッチングは既に完了しているため不要）。

### 3.2 入力データ仕様

`current.csv`（または `rematched.csv`）の使用カラム：

| カラム | GT / EST | 説明 |
|--------|----------|------|
| `unix_time` | 両 | タイムスタンプ（μs） |
| `x`, `y` | 両 | BEV 位置（m、ego 座標系） |
| `r_val` | 両 | DTV = sqrt(x²+y²)（m） |
| `theta_val` | 両 | atan2(y,x) 度数（-180 ～ +180 または -60 ～ +300）|
| `label` | 両 | 物体クラス |
| `confidence` | EST | 検出信頼度 |
| `status` | 両 | TP / FP / FN |
| `visibility` | GT | FULL / MOST / PARTIAL / NONE |
| `x_error`, `y_error` | TP のみ | 位置誤差（GT と EST で同値） |

### 3.3 Distance Risk Weight

```
DTV = r_val   (= sqrt(x² + y²))

distance_weight = exp(−λ × DTV)
```

デフォルト λ = 0.01 の重みの例：

| DTV | weight |
|-----|--------|
| 2 m | 0.980 |
| 10 m | 0.905 |
| 20 m | 0.819 |
| 50 m | 0.607 |
| 100 m | 0.368 |

### 3.4 FOV フィルター

ego 進行方向（+x 軸 = 0°）を基準に **±120° 以内** を有効範囲とする（240° FOV）。

```
theta_signed = ((theta_val + 180) % 360) - 180   # -180 ～ +180 に正規化
in_fov = |theta_signed| ≤ (fov_deg / 2)           # 240° の場合 ≤ 120°
```

FOV 外 および `r_val > max_distance_m` の GT/EST は `weight = 0` として扱う（評価から除外）。

### 3.5 GT 重み

```
gt_weight = in_fov * distance_weight * (0 if r_val > max_dist else 1)
```

`visibility_exclude` が設定された場合、該当 GT も `gt_weight = 0`。

### 3.6 TP 重み・FN 重み

```
tp_weight = gt_weight_of_matched_gt
fn_weight = gt_weight
```

EST TP 行の場合、対応 GT の位置は `x_error`, `y_error` から復元する：

```
gt_x = est_x - x_error
gt_y = est_y - y_error
gt_r = sqrt(gt_x² + gt_y²)
gt_theta_signed = atan2(gt_y, gt_x) [degrees]
```

### 3.7 FP 重み（2 案）

| 案 | 式 | 主結果 |
|---|---|---|
| **案A: EST 距離重み** | `fp_weight = in_fov_est * exp(−λ × r_val_est)` | ✓ 主結果 |
| **案B: 重み 1** | `fp_weight = 1.0` | 参考出力 |

FP は GT 対応がないため距離重みの意味が TP/FN と完全に一致しないことに注意する（PoC 指示書 § 6.5 参照）。

### 3.8 Weighted Precision / Recall

confidence 降順に EST を並べ、累積集計する：

```
weighted_TP[k] = Σ tp_weight   (k 番目まで)
weighted_FP[k] = Σ fp_weight   (k 番目まで)
total_GT_weight = Σ gt_weight  (全 GT、分母)

weighted_precision[k] = weighted_TP[k] / (weighted_TP[k] + weighted_FP[k])
weighted_recall[k]    = weighted_TP[k] / total_GT_weight
```

### 3.9 Weighted AP

PR 曲線の面積（step-wise AP）:

```
AP = Σ (recall[k] - recall[k-1]) * precision[k]
```

通常 AP との比較のため、同じ curve に対して `weight = 1` の標準 AP も出力する。

### 3.10 Weighted mAP

```
weighted_mAP = mean(weighted_AP_class)              # macro average（主結果）
weighted_mAP_w = Σ(w_c * AP_c) / Σ w_c             # GT weight で重み付き平均（参考）
  where w_c = total_GT_weight for class c
```

### 3.11 距離帯別分析

| bin | 範囲 |
|-----|------|
| near | 0 – 10 m |
| mid | 10 – 30 m |
| far | 30 – 50 m |
| very_far | 50 – 100 m |
| out_of_scope | 100 m 超 / FOV 外 |

### 3.12 主要パラメータ（WeightedMapConfig）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `classes` | car/truck/bus/pedestrian | 評価対象クラス |
| `max_distance_m` | 100.0 | これより遠い GT/EST は weight=0 |
| `distance_weight_lambda` | 0.01 | exp(−λ × DTV) の λ |
| `forward_fov_deg` | 240.0 | 有効 FOV 幅（度）、前方中心 |
| `fp_weighting` | `"est_distance"` | `"est_distance"` / `"uniform"` |
| `visibility_exclude` | `["NONE"]` | weight=0 とする GT visibility |
| `ap_method` | `"stepwise"` | AP 計算方式 |

---

## 4. パイプライン統合

### 4.1 フロー

```
current.csv
    ↓
[run_rematch.py]  (任意: 閾値を変えて TP/FP/FN を再計算)
    ↓
rematched.csv
    ↓
[run_weighted_map.py]
    ↓
output/weighted_map/{label}/
    ├── topic_metrics.csv           (topicなし: class別結果)
    ├── class_metrics.csv
    ├── distance_bin_recall.csv
    ├── high_weight_fn.csv          (上位FN一覧)
    ├── high_weight_fp.csv          (上位FP一覧)
    ├── normal_vs_weighted_map.csv  (比較サマリ)
    └── summary.json
```

### 4.2 比較実験

```bash
# 1. オリジナルマッチングでの weighted mAP
python script/run_weighted_map.py --config configs/weighted_map_default.yaml \
    --csv data/output/pdf/.../current.csv --label original

# 2. re-match 後の weighted mAP
python script/run_rematch.py --config configs/rematch_default.yaml
python script/run_weighted_map.py --config configs/weighted_map_default.yaml \
    --csv data/output/rematch/default/rematched.csv --label rematched

# 3. 比較
python script/run_weighted_map.py --compare \
    data/output/weighted_map/original/summary.json \
    data/output/weighted_map/rematched/summary.json
```

---

## 5. PoC 指示書からの適応点

| PoC 指示書 | 本実装での対応 |
|-----------|--------------|
| 入力: GT/EST 別 parquet | `current.csv`（マッチング済み）を使用。parquet 対応は将来拡張 |
| マッチング実装（Step 4） | マッチング済み status 列を流用。RE-matching は Stage 1 で対応 |
| frame alignment (±50ms) | `unix_time` 単位で既に整合済み |
| drivable surface polygon | 未対応（TODO B相当）。FOV 240° のみで代替。結果に「drivable surface 未反映」と明記 |
| BEV IoU マッチング | center distance ベースのみ対応（parquet の bbox 情報なし） |
| confidence が無い場合 | 現 CSV には confidence 列あり（ほぼ 0.999 付近に集中） |

---

## 6. 既知の制約

1. **confidence 集中**: EST confidence の大半が 0.999。PR 曲線が短くほぼ垂直になり、mAP ≈ Recall になりやすい。weighted mAP でも同様。
2. **drivable surface 未判定**: FOV 外フィルターのみ。非 drivable な GT/EST は weight が残る可能性がある。
3. **yaw/speed 誤差が NaN のとき**: Re-match 後の CSV では AOE/AVE が使えない（NDS との組み合わせ不可）。
4. **FP の重み付け**: 案 A の `est_distance_weight` は近距離誤検出をペナルティ化できるが、GT 対応のない FP への距離重みの意味論的正当性は TP より低い。

---

## 7. ファイル構成

```
script/
  weighted_map/
    config.py          WeightedMapConfig dataclass (YAML I/O + argparse)
    compute.py         重み計算・weighted AP・距離帯分析
    __init__.py
  run_weighted_map.py  CLI エントリポイント

configs/
  weighted_map_default.yaml   標準設定 (λ=0.01, 240°FOV, 0-100m)
  weighted_map_strict.yaml    近距離重視 (λ=0.03, vis≠NONE)
```
