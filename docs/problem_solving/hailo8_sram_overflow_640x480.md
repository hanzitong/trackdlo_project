# Hailo-8 SRAM 容量不足による HEF コンパイル失敗

## 症状

DeepLabV3 (MobileNetV2 encoder) を 640×480 入力で Hailo DFC 3.33.0 に通すと、
コンパイルが以下のアサーション失敗でクラッシュする。

```
compiler: ../src/allocator/network_graph_appender.cpp:402:
  Status<network_graph::NetworkNode*>
  allocator::NetworkGraphAppender::AddShortcut(...)
  Assertion `src_node.output_format() == (*first_succ)->input_format()' failed.

[error] Failed to produce compiled graph
[error] BackendAllocatorException: Compilation failed with unexpected crash
```

ログ中に以下の行がある。

```
[info] Trying to compile the network in a single context
[info] Single context flow failed: Recoverable single context error
[info] Using Multi-context flow
```

すなわち「シングルコンテキストに収まらないのでマルチコンテキストに移行したが、
そこでスキップ接続の処理に失敗してクラッシュ」という流れ。

---

## 背景知識

### Hailo-8 のメモリ構造

Hailo-8 チップは、演算ユニットの近傍に約 **2.5 MB の内部 SRAM** を持つ。
この SRAM は演算速度を稼ぐためのものであり、テンソルをここに乗せたまま
演算ユニットが高速に読み書きできる。

外部の DDR メモリもあるが、DDR へのアクセスは SRAM に比べて
**帯域が狭くレイテンシが高い**ため、パイプライン速度が大幅に落ちる。

### コンテキスト（Context）とは

「コンテキスト」は Hailo-8 がひとまとめに処理できる演算ブロックの単位。
1 コンテキスト = SRAM に同時に乗せられる分量 ≈ **2.5 MB** が目安。

DFC コンパイラは、モデル全体を実行するために必要なテンソルが SRAM に
収まるかどうかを判断し、収まるなら **シングルコンテキスト**、
収まらないなら **マルチコンテキスト** にモデルを分割する。

#### シングルコンテキスト

```
入力 → [Layer A] → [Layer B] → [Layer C] → 出力
         ↑──────────────── SRAM の中 ───────────────┘
```

すべての中間テンソルが SRAM に収まるため、DDR アクセスなしに
端から端まで一気に処理できる。**最速・最安定**。

#### マルチコンテキスト

```
入力 → [Layer A] → [Layer B]   →   [Layer C] → [Layer D] → 出力
         ↑── Context 1 (SRAM) ──↓   ↑── Context 2 (SRAM) ──↑
                                DDR (テンソル転送)
```

コンテキスト境界でテンソルを DDR に書き出し、次のコンテキストで読み込む。
速度が落ちるだけでなく、**コンテキスト間でテンソルのフォーマット（データ配置順・
量子化幅など）が一致していないと、コンパイラが境界接続（AddShortcut）を
処理できずクラッシュする**。

### スキップ接続（Skip Connection）とは

スキップ接続（残差接続・ショートカット）は、ネットワークの途中の出力を
数層とばして後段の層に直接接続する構造。

```
     入力テンソル A
      │
      ├──→ [Conv 1] → [Conv 2] → [Conv 3] ──→ +  → 出力
      │                                          ↑
      └──────────────── スキップ ────────────────┘
         (A を変換せずそのままプラスする)
```

#### なぜスキップ接続がマルチコンテキストで問題になるか

スキップ接続は、遠く離れた 2 点（接続元と接続先）の **テンソルフォーマットが
一致していなければならない**。

シングルコンテキストなら SRAM 内で自由にフォーマットを合わせられる。
しかしマルチコンテキストの場合、スキップ接続がコンテキスト境界をまたぐと
接続元（Context 1）と接続先（Context 2）のフォーマット制約が衝突し、
コンパイラが解決できなくなる。

```
Context 1                    Context 2
[Layer A] ───→ DDR ───→ [Layer B] → [Layer C]
    ↓                                  ↑
    └──── スキップ ── DDR ────→ ──────┘
          ↑ フォーマットが Context 境界をまたいで一致しない
```

このとき DFC 3.33.0 のアロケータが `AddShortcut` アサーションで落ちる。

---

## 640×480 でシングルコンテキストに収まらない理由

ネットワーク前段の畳み込み出力テンソルのサイズが大きすぎる。

| 入力解像度 | 最初の特徴マップ (例: stride=2 後) | SRAM 占有 (近似) |
|----------|----------------------------------|-----------------|
| 640×480 | 320×240×32 (float16) ≈ 4.7 MB   | 2.5 MB 超 → マルチコンテキスト |
| 320×240 | 160×120×32 (float16) ≈ 1.2 MB   | 2.5 MB 以内 → シングルコンテキスト候補 |

DeepLabV3 (MobileNetV2) のエンコーダー内部にも残差ブロックのスキップ接続が
多数存在するため、いったんマルチコンテキストになると上記の境界衝突が発生する。

---

## 試行した回避策と結果

| 試行内容 | 結果 |
|----------|------|
| `resources_param(max_*_utilization=92.5%)` でリソース使用率を上げる | `vector::_M_range_check` で別のクラッシュ |
| `performance_param(compiler_optimization_level=1)` + 長いタイムアウト | `AddShortcut` アサーションは変わらず |
| `compiler_optimization_level=max` | `resources_param` と共存不可 |
| sigmoid 除去（raw logit 出力に変更） | sigmoid は原因でなかった（同じクラッシュ） |
| DeepLabV3+ ResNet34 → DeepLabV3 MobileNetV2 に変更 | V3+ のデコーダースキップはなくなったが、<br>エンコーダー内スキップ問題は残る |

根本的には **入力解像度が大きすぎて SRAM に収まらない**ことが原因であり、
モデル構造やコンパイラオプションで回避するのは困難。

---

## 解決策: 入力解像度を 320×240 に縮小する

RealSense D405 は 320×240 をネイティブ解像度として出力できる
（640×480 のちょうど 2 分の 1）。

320×240 では最初の特徴マップが SRAM 2.5 MB に収まる見込みであり、
シングルコンテキストでのコンパイルが期待できる。

### 対応が必要なファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts_aoyama/prepare_dataset.py` | 320×240 にリサイズ（ポリゴン座標を ÷2） |
| `scripts_aoyama/train_deeplabv3.py` | サイズチェックを `(240, 320)` に変更 |
| `hailo/export_onnx.py` | `INPUT_SHAPE` を `(1, 3, 240, 320)` に変更 |
| `hailo/make_calib_npy.py` | キャリブレーション画像を 320×240 にリサイズ |
| `hailo/infer_live.py` | RealSense ストリームを `320, 240` に変更 |
| `hailo/deeplabv3plus_cable.alls` | `calibset_size` の調整（任意） |
| `scripts_aoyama/infer_live_cpu.py` | 解像度コメント・ストリーム設定を更新 |

### アノテーションマスクは再作成不要

既存の LabelMe JSON（640×480 座標空間）はそのまま利用できる。
`prepare_dataset.py` 内でポリゴン座標を `x÷2, y÷2` にスケールしてから
320×240 のキャンバスに描画すれば、整数比リサイズなので劣化なし。

---

## 参照

- 実際のクラッシュログ:
  `src/bmask_gen/hailo/deeplabv3plus_cable.alls` の `#問題:` コメント
- Hailo モデルズーの DeepLabV3 MobileNetV2 (320×240 で対応実績あり):
  https://github.com/hailo-ai/hailo_model_zoo
