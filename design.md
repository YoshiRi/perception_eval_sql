# Safety-Critical Detection Metrics — Design Document

## 1. 背景と動機

### 1.1 標準 mAP が「体感より低く出る」理由

現行の mAP 計算にはいくつかの系統的なバイアスがある。

| 要因 | 影響 |
|------|------|
| **visibility=NONE の GT を分母に含む** | 完全遮蔽物体は LiDAR/カメラから物理的に検知不可能だが FN としてカウントされ Recall を下げる |
| **unknown クラスの FP が大量発生** | GT に対応しないラベルの EST 検出が Precision を下げ mAP に反映される |
| **全距離（0-150m+）を平均** | 遠距離（100m超）の低 Recall が近距離の高 Recall と混ざる |
| **mAP = Precision × Recall の AUC** | Recall のみ直感的に評価していた場合に Precision 側の低下が見えにくい |

### 1.2 安全性評価として欲しい指標

自動運転の安全評価においては「衝突に関係しうる物体を近距離で取りこぼしていないか」が本質的な問いである。
そのため以下を軸に設計する。

- **Recall 重視**：見落とし（FN）のコストが FP よりはるかに高い
- **近距離優先**：100m 以内の物体が直接的な危険に関係する
- **可視物体のみ**：検知不可能な遮蔽物体を FN として数えない
- **クラス絞り込み**：安全上 critical な 4 クラス（car / truck / bus / pedestrian）

---

## 2. アーキテクチャの思想

### 2.1 マッチングとメトリクス計算は分離する

```
current.csv（マッチング済みデータ）
  ├── 各行は GT または EST 由来
  ├── status 列: TP / FP / FN（perception_eval が決定済み）
  └── TP 行のみ x_error / y_error / yaw_error / speed_error を持つ

           ↓ このプロジェクトの範囲

safety_metrics パッケージ
  ├── prepare():    Safety フィルタを適用（距離・visibility・クラス）
  ├── compute_recall():  GT rows の status を集計
  ├── compute_ap_table(): EST rows を confidence 降順に並べ AP カーブを計算
  └── compute_nds():     TP error metrics から NDS スコアを算出
```

current.csv が「どの GT と EST をペアにするか」という**マッチング**はすでに終わっており、
このパッケージはその結果を**再解釈・再集計**するだけである。
マッチング自体を変えたい場合（例：IoU 閾値変更や追跡ベースのマッチング）は
perception_eval 側を再実行して新しい CSV を生成する必要がある。

### 2.2 Safety フィルタは GT 側にのみ作用する（制約）

`visibility_exclude` を GT に適用すると GT 分母が絞られるが、
EST の TP/FP 判定には **pair_uuid（対応 GT の UUID）が CSV に存在しない**ため
「除外した GT にマッチした EST TP」を FP に変換できない。

対処として AP 計算内で `cum_tp = min(cum_tp, n_gt)` によって recall ≤ 1.0 を保証している。
これは保守的な近似であり、visibility フィルタが厳しいほど（near_strict 等）
AP が若干過大評価になる可能性がある。
完全な解決には pair_uuid を CSV に含める必要がある。

---

## 3. nuScenes Detection Score (NDS) — 数式と本実装の差分

### 3.1 公式 NDS

nuScenes Object Detection Challenge（Caesar et al. 2020）で定義された総合スコア。

```
NDS = 1/10 × (5 × mAP + mATE_score + mASE_score + mAOE_score + mAVE_score + mAAE_score)

各 TP スコア:  score_i = 1 - min(metric_i_raw, 1.0)
```

| TP metric | 内容 | 公式単位 |
|-----------|------|---------|
| mATE | 平均位置誤差（2D BEV center distance） | m |
| mASE | 平均スケール誤差（1 - IoU_3D） | 無次元 [0,1] |
| mAOE | 平均向き誤差（最小角度差） | rad |
| mAVE | 平均速度誤差（L2ノルム） | m/s |
| mAAE | 平均属性誤差（0/1損失） | 無次元 [0,1] |

公式は **正規化係数なし** で raw 値を直接 1.0 でクリップする。
`mATE = 0.5m → score = 0.50`（0.5m ずれはかなり悪い評価）。

mAP は 10 クラス × 4 距離閾値（0.5/1/2/4m）の平均。

### 3.2 本実装（NDS_safety）との差分

```
NDS_safety = 1/8 × (5 × mAP + ATE_score + AOE_score + AVE_score)

各 TP スコア:  score_i = 1 - min(metric_i / norm_i, 1.0)   ← 正規化あり（公式と異なる）
```

| 変更点 | 公式 | 本実装 | 理由 |
|--------|------|--------|------|
| TP metric 数 | 5 | 3（ATE/AOE/AVE） | current.csv に ASE/AAE 列がない |
| 分母 | 10 | 8（=5+3） | TP metric 数に合わせた |
| クラス | 10 | 4（安全クリティカルのみ） | 評価目的に特化 |
| 距離範囲 | 全距離 | 0-100m（設定可変） | 安全上重要な近距離に限定 |
| GT フィルタ | なし | visibility ≠ NONE | 不可視物体を FN 扱いしない |
| TP スコア計算 | `1 - min(raw, 1)` | `1 - min(raw/norm, 1)` | 正規化により尺度を調整 |

### 3.3 TP スコアの正規化係数の意味

本実装のデフォルト設定（`default.yaml`）は以下を使用している。

| metric | norm 値 | 意味 |
|--------|---------|------|
| ATE_norm = 2.0 m | 2m 以上のずれはスコア 0 | 公式より甘い（公式は 1m で 0 相当） |
| AOE_norm = π rad | 180° は完全な向き失敗 | 合理的 |
| AVE_norm = 10.0 m/s | 10 m/s 以上の速度誤差はスコア 0 | 公式（1.5 m/s）より大幅に甘い |

**重要**：この正規化により本実装の NDS_safety は公式式より **5-10 ポイント高く出る**。
公式式（正規化なし）で同データを評価した場合の対照値：

| config | NDS_safety (本実装) | NDS_safety (公式式) | 差分 |
|--------|-------------------|-------------------|------|
| default (0-100m) | 0.714 | 0.650 | +0.064 |
| near_strict (0-60m) | 0.861 | 0.803 | +0.058 |
| full_range (0-150m) | 0.666 | 0.599 | +0.067 |

絶対値を公式 leaderboard と比較するには `ate_norm=1.0, ave_norm=1.5` に変更するか、
あるいはスコア計算を `1 - min(raw, 1.0)` に修正する必要がある。

---

## 4. NDS スコアの目安

### 4.1 公式 nuScenes leaderboard（参考）

以下は 10 クラス・全距離・全可視性での公式 NDS（test set, 2023-24 年頃）。

| モデル | mAP | NDS | 備考 |
|--------|-----|-----|------|
| BEVFusion (Liu 2022) | 0.703 | 0.730 | LiDAR + Camera |
| UniAD (Hu 2023) | — | 0.730 | E2E planning |
| CenterPoint (Yin 2021) | 0.598 | 0.670 | LiDAR のみ |

**直接比較は不可**（クラス数・距離・可視性フィルタが異なる）。

### 4.2 本実装での目安（safety フィルタあり）

本実装で公式式（正規化なし）を使った場合のおおよその解釈：

| NDS_safety | 解釈 |
|-----------|------|
| 0.80 以上 | 優秀：近距離・可視物体をよく検知し位置精度も高い |
| 0.65 – 0.80 | 良好：実用水準。遠距離・遮蔽物体への対策が課題 |
| 0.50 – 0.65 | 要改善：特定クラス（歩行者等）か距離帯の Recall が低い |
| 0.50 未満 | 問題あり：安全上クリティカルな物体の見落としが多い |

今回の評価値（公式式換算）：
- default (0-100m, vis≠NONE): **NDS_safety = 0.650** → 良好水準
- near_strict (0-60m, FULL/MOST のみ): **NDS_safety = 0.803** → 優秀
- full_range (0-150m): **NDS_safety = 0.599** → 要改善（遠距離の低 Recall が影響）

Recall 単体では：歩行者の近距離 Recall（68-80%）が特に改善余地あり。

---

## 5. 各設定パラメータの設計意図

| パラメータ | デフォルト | 設計意図 |
|-----------|-----------|---------|
| `classes` | car/truck/bus/pedestrian | unknown/motorbike を除外。unknown は GT ラベルの定義が曖昧で FP 起因のノイズが大きい |
| `dist_max_m` | 100 m | 時速 36 km の車両で 10 秒分の視野。highway は 150m 推奨 |
| `visibility_exclude` | ["NONE"] | 完全遮蔽物体。PARTIAL は半分見えており検知可能なので除外しない |
| `match_thresholds_m` | [0.5, 1, 2, 4] | 公式 nuScenes に準拠。0.5m で厳格、4m で緩やか。平均で多様な使用シーンをカバー |
| `speed_error_clip_mps` | 50.0 m/s | 180 km/h 相当。センサーノイズによる物理的にあり得ない外れ値（今回 max 1.27e13）を除去 |
| `ate_norm_m` | 2.0 m | 公式より緩い。このシステムは bbox 単位で対応が取れれば十分という前提。厳格評価は 1.0m 推奨 |
| `ave_norm_mps` | 10.0 m/s | 公式（1.5 m/s）より大幅に緩い。速度推定精度をあまり重視しない場合に使用。重視する場合は 2.0-3.0 推奨 |
| `class_weights` | null（均等） | near_strict では pedestrian=2.0 に設定。mAP の加重平均化に使用 |

---

## 6. 既知の制限事項

1. **pair_uuid がない**: EST TP がどの GT にマッチしたか不明なため、visibility フィルタを厳しくすると AP が若干過大評価になる可能性がある。`recall` は GT 側の集計のみなので正確。

2. **confidence 分布の偏り**: このモデルの confidence は 0.999 付近に大量集中しており、PR カーブが短くほぼ垂直になる。mAP ≈ Recall となりやすく、Precision の情報が失われがち。

3. **NDS の非互換性**: 正規化係数（`ate_norm`, `ave_norm` 等）を変えると NDS 値が大きく変わる。ランの比較は**同じ config でのみ有効**。絶対値の比較はできない。

4. **マッチング未変更**: 現行の TP/FP/FN は perception_eval が決定したものを使用している。マッチング距離閾値や追跡ベースの評価に変えたい場合は perception_eval の再実行が必要。

---

## 7. ファイル構成

```
configs/
  default.yaml         標準設定（0-100m, vis≠NONE）
  near_strict.yaml     近距離厳格（0-60m, FULL/MOST のみ、歩行者2倍重み）
  full_range.yaml      広域（0-150m, vis≠NONE）

script/
  run_safety_metrics.py       CLI エントリポイント
  safety_metrics/
    config.py                 SafetyConfig dataclass（YAML I/O + argparse）
    compute.py                純粋計算関数（I/O なし）
    __init__.py
  safety_critical_metrics.py  初期の一発スクリプト（参考用）
```

### 典型的な使い方

```bash
# プリセットで実行
python script/run_safety_metrics.py --config configs/default.yaml

# 距離だけ変えて比較
python script/run_safety_metrics.py --config configs/default.yaml \
    --dist-max 60 --label v4.4.0_60m --output-dir data/output/safety_metrics/60m

# 結果を横並び比較
python script/run_safety_metrics.py --compare \
    data/output/safety_metrics/default/summary.json \
    data/output/safety_metrics/60m/summary.json
```
