# Hailo-8 HEF コンパイル試行記録

ケーブルセグメンテーションモデルを Hailo-8 (PCIe) 向け HEF にコンパイルするまでの
全試行経緯、失敗原因の分析、および成功条件の記録。

**環境**: Hailo DFC 3.33.0、hailo8 hw-arch、GPU: RTX 3070 Ti (8GB)

---

## 結論（先に読む）

**唯一成功した構成**: `VGGSeg (VGG11_BN バックボーン) + 320×240`

成功の本質的な条件は「**残差接続 (skip connection) を持たないエンコーダーを使う**」こと。
MobileNetV2 系は残差接続の存在によりマルチコンテキスト分割で常にクラッシュした。

---

## 背景知識: Hailo-8 のコンテキストとスキップ接続

### コンテキストとは

Hailo-8 は内部 SRAM ≈ **2.5 MB** を持つ。演算時に必要なテンソルがこの SRAM に
収まる場合は **シングルコンテキスト**、収まらない場合は DFC が自動的にネットワークを
複数の **マルチコンテキスト** に分割する。

マルチコンテキストではコンテキスト境界でテンソルを DDR に書き出し、次の
コンテキストで読み込む。

### AddShortcut クラッシュの仕組み

DFC がコンテキスト境界でテンソルを転送するとき、テンソルのメモリフォーマット
（データ配置順・量子化幅）が変わることがある。

ここで問題になるのが **スキップ接続 (ADD 操作)** だ。
スキップ接続では「分岐元のテンソル A」を「複数層後の Add 演算」に渡す。

```
分岐元 A ──────────────────────────→ Add → 出力
          └→ Conv → Conv → Conv ─↗
```

A がコンテキスト境界をまたぐと、境界前後でフォーマットが変わる。
一方 Add 先のテンソルは同じコンテキスト内で別フォーマットになっており、
DFC 3.33.0 の `NetworkGraphAppender::AddShortcut()` がアサーション失敗でクラッシュする。

```
compiler: ../src/allocator/network_graph_appender.cpp:402:
  Assertion `src_node.output_format() == (*first_succ)->input_format()' failed.
[error] Failed to produce compiled graph
[error] BackendAllocatorException: Compilation failed with unexpected crash
```

---

## 試行一覧

### 試行 1: DeepLabV3+ + MobileNetV2 + 640×480

**ディレクトリ**: `src/bmask_gen/hailo_halfsize/`（後に hailo/ も試行）

**結果**: ❌ AddShortcut アサーション失敗

**失敗ログ**:
```
[info] Single context flow failed: Recoverable single context error
[info] Using Multi-context flow
compiler: AddShortcut Assertion `src_node.output_format() == (*first_succ)->input_format()' failed.
[error] Failed to produce compiled graph
```

**原因分析**:

640×480 入力時、stride=2 後の最初の特徴マップは `320×240×32 = 4.7 MB` となり、
Hailo-8 の SRAM (2.5 MB) を超える。このため DFC は強制的にマルチコンテキスト分割する。

DeepLabV3+ はエンコーダーの stride-4 出力 (`160×120×24`) をデコーダーで再利用する
**長距離スキップ接続** を持つ。この接続がコンテキスト境界をまたぐと
AddShortcut クラッシュになる。

```
エンコーダー stride-4 出力
 │
 ├→ [Context 1: stride-4〜stride-16 の畳み込み群] → DDR 転送
 │                                                         ↓
 │                                                 [Context 2: ASPP]
 │                                                         ↓
 └──────────── スキップ (DDR をまたぐ) ──────────→ Decoder でconcat
               ↑ フォーマット不一致でクラッシュ
```

**試したオプション (すべて失敗)**:
- `compiler_optimization_level=max/1/0`
- `allocator_param(timeout=7200)` で探索時間延長
- GPU 使用 (`CUDA_VISIBLE_DEVICES=0`)
- sigmoid 除去 (raw logit 出力化)

---

### 試行 2: DeepLabV3+ + MobileNetV2 + 320×240

**ディレクトリ**: `src/bmask_gen/hailo_halfsize/` (同ディレクトリで入力サイズ変更)

**仮説**: 320×240 なら最初の特徴マップが `160×120×32 ≈ 1.2 MB` となり
SRAM に収まってシングルコンテキストになるはず。

**結果**: ❌ マルチコンテキスト分割失敗 (シングルコンテキスト不可)

**失敗ログ**:
```
[info] Single context flow failed: Recoverable single context error
[info] Using Multi-context flow
[info] Finding the best partition to contexts...
Mapping Failed (allocation time: ~7m)
Automri finished with too many resources on context_0 with 24/50 failures.
[error] Compiler could not find a valid partition to contexts.
```

**原因分析**:

DeepLabV3+ のデコーダースキップ接続 (stride-4 → Decoder) は、エンコーダー全体が
演算している間ずっとその中間テンソルを保持し続けなければならない。
これがシングルコンテキストへの制約として作用し、320×240 でも収まりきらなかった。

マルチコンテキスト分割では context_0 にエンコーダー前段 + スキップ保持が
集中するため、どう分割してもリソース超過になる。

`compiler_optimization_level=max` では長時間 (> 10 分) タイムアウトせず探索を続けるが、
`level=1` と `level=0` では数分で「too many resources on context_0」エラーが出た。

---

### 試行 3: DeepLabV3 (V3+ではない) + MobileNetV2 + 640×480

**ディレクトリ**: `src/bmask_gen/hailo_halfsize_v3/` (最初の版)

**仮説**: DeepLabV3 はデコーダースキップ接続を持たないため、
マルチコンテキスト分割の context_0 集中問題が解消されるはず。

**結果**: ❌ AddShortcut アサーション失敗 (640×480 は変わらずマルチコンテキスト)

**失敗ログ**:
```
[info] Single context flow failed: Recoverable single context error
[info] Using Multi-context flow
[info] Finding the best partition to contexts...
compiler: AddShortcut Assertion `src_node.output_format() == (*first_succ)->input_format()' failed.
[error] Failed to produce compiled graph
```

**原因分析**:

640×480 では最初の特徴マップが SRAM を超えるため、DeepLabV3 でもマルチコンテキスト
になることは変わらない。

DeepLabV3 はデコーダースキップこそないが、MobileNetV2 エンコーダー内の
**各ボトルネックブロックの残差接続** (短距離スキップ) が残っている。
これらがコンテキスト境界をまたぐと同様の AddShortcut クラッシュが発生した。

```
MobileNetV2 ボトルネックブロック (×17):
  入力 x ──────────────────────────→ Add → 出力
           └→ pw_conv → dw_conv → pw_conv ─↗
              (この ADD がコンテキスト境界をまたぐとクラッシュ)
```

---

### 試行 4: DeepLabV3 + MobileNetV2 + 320×240

**ディレクトリ**: `src/bmask_gen/hailo_halfsize_v3/` (320×240版に更新)  
**学習スクリプト**: `src/bmask_gen/scripts_aoyama_halfsize_v3/train.py`

**仮説**: 320×240 でシングルコンテキストに収まればエンコーダー残差問題も回避できる。
Hailo Model Zoo でも DeepLabV3 + MobileNetV2 + 320×240 の実績あり。

**結果**: ❌ AddShortcut アサーション失敗 (320×240 でもシングルコンテキスト不可)

**失敗ログ**:
```
[info] Running Auto-Merger × 5 (単一コンテキストに収めようと繰り返し試行)
[info] Single context flow failed: Recoverable single context error
[info] Using Multi-context flow
[info] Finding the best partition to contexts...
compiler: AddShortcut Assertion `src_node.output_format() == (*first_succ)->input_format()' failed.
[error] Failed to produce compiled graph
```

**原因分析**:

320×240 でも DFC が MobileNetV2 をシングルコンテキストに収めることができず、
マルチコンテキストに移行した。Auto-Merger が 5 回リトライしていることから、
シングルコンテキストに「惜しくも収まらない」サイズであると推測できる。

マルチコンテキスト移行後は試行 3 と同様にエンコーダー残差接続が
コンテキスト境界をまたいでクラッシュした。

**ここで得た教訓**: MobileNetV2 は残差接続の密度が高く、DFC 3.33.0 では
マルチコンテキスト + 残差接続の組み合わせが根本的にサポートされていない
（または非常に壊れやすい）。解決策は「残差接続のないエンコーダー」に切り替えること。

---

### 試行 5: VGGSeg (VGG11_BN) + 320×240 ← **成功**

**ディレクトリ**: `src/bmask_gen/hailo_halfsize_vgg/`  
**学習スクリプト**: `src/bmask_gen/scripts_aoyama_halfsize_vgg/train.py`

**なぜ VGG を選んだか**:

VGG は 1986年の AlexNet 系統の純シーケンシャルアーキテクチャで、
残差接続が一切存在しない。Conv → BN → ReLU → MaxPool のみで構成されており、
DFC がどの層間でコンテキスト境界を置いても AddShortcut が発生しない。

**なぜ smp.DeepLabV3 を使わないか**:

smp の DeepLabV3 は ASPP のためにエンコーダーで `make_dilated(output_stride)` を
呼び出す。これは MaxPool を dilated convolution に置き換えるが、VGG は MaxPool を
使っているため `ValueError: 'VGG' models do not support dilated mode` になる。

代わりにカスタムモデル `VGGSeg` を定義した:
```python
class VGGSeg(nn.Module):
    backbone: vgg11_bn.features  # 全 Conv+BN+ReLU+MaxPool 29 層 (残差なし)
    head:     Conv2d(512→64, 1x1) → ReLU → Conv2d(64→1, 1x1)
    forward:  backbone → head → bilinear_upsample(to input size)
```

VGG11_BN の出力は stride=32 (入力 320×240 → 10×7)。
bilinear アップサンプルで 320×240 に戻す。Hailo は Resize (bilinear) をサポート済み。

**コンパイル過程**:

```
[info] Trying to compile the network in a single context
[info] Using Single-context flow       ← シングルコンテキスト成功
[info] Running resources allocation (mapping) flow, time per context: 9m 59s
[info] Context:0/0 Iteration 0: Mapping prepost...
  (max_control_utilization を 117.5% → 100% → 97.5% → ... → 75% と段階的に調整)
[info] Successful Mapping (allocation time: 5s)  ← 75% 設定で解を発見
[info] Bandwidth of DDR buffers: 0.0 Mbps        ← DDR 転送ゼロ
[info] Bandwidth of inter context tensors: 0.0 Mbps  ← コンテキスト間転送ゼロ
[info] Successful Compilation (compilation time: 7s)
[info] Saved HEF to: .../hailo_halfsize_vgg/model.hef
```

**クラスター使用率 (最終配置)**:

| クラスター | Control | Compute | Memory |
|---|---|---|---|
| cluster_0 | 93.8% | 100% | 55.5% |
| cluster_1 | 87.5% | 90.6% | 82.0% |
| cluster_2 | 93.8% | 89.1% | 99.2% |
| cluster_3 | 62.5% | 81.3% | 65.6% |
| cluster_4 | 93.8% | 100% | 62.5% |
| cluster_5 | 81.3% | 87.5% | 57.8% |
| cluster_6 | 81.3% | 96.9% | 85.9% |
| cluster_7 | 62.5% | 92.2% | 57.8% |
| **合計**  | **82%** | **92.2%** | **70.8%** |

**生成ファイル**: `model.hef` (13 MB)

---

## 各失敗の根本原因まとめ

| 失敗モード | 発生条件 | ログのキーワード |
|---|---|---|
| AddShortcut クラッシュ | スキップ接続がコンテキスト境界をまたぐ | `AddShortcut Assertion failed` |
| context_0 資源集中 | デコーダースキップが context_0 にテンソル保持を強制 | `Automri finished with too many resources on context_0` |
| Mapping Timeout | 探索空間が広すぎて制限時間内に解が見つからない | `Mapping Failed (Timeout)` |

**AddShortcut クラッシュが起きる条件**:
1. ネットワークがマルチコンテキストになる (SRAM に収まらない)
2. かつスキップ接続 (Add/residual) がコンテキスト境界をまたぐ

この 2 条件を同時に満たさないようにするには:
- SRAM に全体が収まるよう縮小する → **難しい** (MobileNetV2 は 320×240 でも収まらなかった)
- スキップ接続を持たないアーキテクチャにする → **VGG で解決**

---

## DFC 3.33.0 に関する重要な知識

### GPU が使われない問題

DFC は内部で GPU メモリ使用率が **≤5%** の GPU を自動選択する
(`hailo_model_optimization/acceleras/utils/nvidia_smi_gpu_selector.py`)。
デスクトップ + Chrome 環境では常に ~17.5% 超えており、自動選択が失敗して CPU になる。

**対処**: `CUDA_VISIBLE_DEVICES=0` を環境変数にセットすると、使用率チェックを
スキップして GPU を強制使用できる。DFC の `__init__.py` に
「`CUDA_VISIBLE_DEVICES` が既にセットされていれば自動選択をスキップする」分岐がある。

```bash
CUDA_VISIBLE_DEVICES=0 hailo compiler model_opt.har ...
```

GPU (RTX 3070 Ti) 使用時の探索速度: CPU 比約 1.7 倍速 (実測)。

### タイムアウトのデフォルト値

`allocator_param(timeout=...)` を指定しない場合、デフォルトは **約 10 分/コンテキスト**
(ログ: `time per context: 9m 59s`)。無制限ではない。
長い探索が必要な場合は `allocator_param(splitter_timeout=7200, timeout=7200)` など追加する。

### compiler_optimization_level の有効値

`0`, `1` (デフォルト), `max` (=2) の 3 値のみ。
- `max`: 探索空間が広い。シングルコンテキストを長時間探索してから諦める。
- `1`: デフォルト。
- `0`: 探索空間が最小。素早く解または失敗を返す。

### smp + VGG の制約

`segmentation_models_pytorch` の DeepLabV3/DeepLabV3Plus は
エンコーダーに `make_dilated()` を呼ぶ。VGG は MaxPool を dilation 化できないため
`ValueError: 'VGG' models do not support dilated mode` になる。

VGG を使う場合は smp を使わず `torchvision.models.vgg11_bn().features` を
バックボーンとしたカスタムモデルを実装する。

### hailo parser onnx のフラグ (DFC 3.33.0)

```bash
# 正しい (3.33.0)
hailo parser onnx model.onnx --tensor-shapes input=[1,3,H,W]

# 誤り (古いバージョン用)
hailo parser onnx model.onnx --net-input-shapes input=[1,3,H,W]  # 認識されない
```

### hailo optimize のフラグ (DFC 3.33.0)

```bash
# 正しい (3.33.0)
hailo optimize model.har --output-har-path model_opt.har

# 誤り (古いバージョン用)
hailo optimize model.har --har-path model_opt.har  # 認識されない
```

---

## 成功した DFC パイプライン (再現手順)

```bash
cd src/bmask_gen/hailo_halfsize_vgg/

# [1] ONNX エクスポート
uv run python export_onnx.py

# [2] キャリブレーションセット生成
uv run python make_calib_npy.py

# [3] ONNX → HAR
/tmp/hailo_extract/hailo_venv/bin/hailo parser onnx model.onnx \
  --hw-arch hailo8 --tensor-shapes input=[1,3,240,320] -y

# [4] HAR → 最適化 HAR (量子化)
CUDA_VISIBLE_DEVICES=0 \
  /tmp/hailo_extract/hailo_venv/bin/hailo optimize model.har \
  --calib-set-path calib_set.npy \
  --output-har-path model_opt.har \
  --model-script model_script.alls

# [5] 最適化 HAR → HEF
CUDA_VISIBLE_DEVICES=0 \
  /tmp/hailo_extract/hailo_venv/bin/hailo compiler model_opt.har \
  --hw-arch hailo8 \
  --model-script model_script.alls \
  --output-dir .
# → model.hef (13 MB) が生成される
```

---

## 関連ファイル

| ファイル | 内容 |
|---|---|
| `src/bmask_gen/hailo_halfsize_vgg/model.hef` | **成功した HEF** (13 MB) |
| `src/bmask_gen/hailo_halfsize_vgg/model_script.alls` | DFC 設定 (level=max, VGG 用) |
| `src/bmask_gen/scripts_aoyama_halfsize_vgg/train.py` | VGGSeg 学習スクリプト |
| `src/bmask_gen/hailo_halfsize_vgg/infer_live.py` | HEF + RealSense D405 推論スクリプト |
| `docs/hailo_hef_compile_guide.md` | DFC 全体の操作ガイド |
| `docs/problem_solving/hailo8_sram_overflow_640x480.md` | 640×480 初期失敗の詳細 |
