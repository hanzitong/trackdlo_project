# Hailo-8 HEF コンパイルガイド

PyTorchモデル (.pth) を Hailo-8 で動作する HEF ファイルに変換する手順。

---

## 全体像

```
PyTorch重み (.pth)
    │
    ▼ [1] export_onnx.py
ONNX モデル (.onnx)
    │
    ▼ [2] hailo parser onnx
HAR ファイル (.har)          ← ネットワーク構造をHailo内部形式に変換
    │
    ▼ [3] hailo optimize
最適化済み HAR (.har)        ← INT8量子化（キャリブレーションセットを使用）
    │
    ▼ [4] hailo compiler
HEF ファイル (.hef)          ← Hailo-8 実行バイナリ
```

DFC (Dataflow Compiler) は Hailo Developer Zone からダウンロードするツール。  
バージョン 3.33.0 で動作確認済み。

---

## 事前準備

### DFC インストール

```bash
# Hailo Developer Zone からダウンロードした .whl を専用 venv に入れる
python3 -m venv /tmp/hailo_extract/hailo_venv
source /tmp/hailo_extract/hailo_venv/bin/activate
pip install hailo_dataflow_compiler-3.33.0-*.whl
```

インストール後に使えるコマンド:
- `hailo parser onnx`
- `hailo optimize`
- `hailo compiler`

### GPU を使うための注意

DFC は内部で GPU メモリ使用率が **5% 以下** の GPU を自動選択する。  
デスクトップ環境 (X11) や Chrome が GPU メモリを占有している場合、しきい値を超えて  
`No GPU chosen and no suitable GPU found, falling back to CPU.` と表示されCPUになる。

**対処法**: `CUDA_VISIBLE_DEVICES=0` を環境変数にセットして実行すると、  
使用率チェックをバイパスして GPU 0 を強制使用できる。

```bash
CUDA_VISIBLE_DEVICES=0 hailo compiler ...
```

---

## ステップごとの手順

作業ディレクトリ例: `src/bmask_gen/hailo_halfsize/`

### [1] ONNX エクスポート

```bash
uv run python src/bmask_gen/hailo_halfsize/export_onnx.py
# → hailo_halfsize/model.onnx が生成される
```

**注意点:**
- `opset_version=11`, `dynamo=False` を指定する（DFC 3.33.0 の要件）
- sigmoid は ONNX に含めない（logit 出力のまま）。Hailo コンパイラがフォーマット不一致でクラッシュするため
- 入力形状は固定 NCHW: `(1, 3, H, W)`

### [2] パース (ONNX → HAR)

```bash
cd src/bmask_gen/hailo_halfsize/
/tmp/hailo_extract/hailo_venv/bin/hailo parser onnx model.onnx \
  --net-input-shapes input=[1,3,240,320]
# → model.har が生成される
```

### [3] 最適化・量子化 (HAR → 最適化済み HAR)

```bash
/tmp/hailo_extract/hailo_venv/bin/hailo optimize model.har \
  --calib-set-path calib_set.npy \
  --output-har-path model_opt.har \
  --model-script model_script.alls
# → model_opt.har が生成される
```

キャリブレーションセット (`calib_set.npy`) の形式:
- shape: `(N, H, W, C)` — **NHWC**（NCHW ではない）
- dtype: `float32`, 値域: `[0, 1]`
- 前処理は学習時と完全に同じにすること（BGR→RGB, /255.0）

### [4] コンパイル (最適化済み HAR → HEF)

```bash
cd src/bmask_gen/hailo_halfsize/
CUDA_VISIBLE_DEVICES=0 \
  /tmp/hailo_extract/hailo_venv/bin/hailo compiler model_opt.har \
  --hw-arch hailo8 \
  --model-script model_script.alls \
  --output-dir .
# → model.hef が生成される
```

`CUDA_VISIBLE_DEVICES=0` を付けると GPU を使ったマッピング探索になり、  
CPU 比で約 1.7 倍速い（RTX 3070 Ti 環境での実測）。

---

## model_script.alls の設定

コンパイルとキャリブレーションの挙動を制御するファイル。

```
# キャリブレーション設定
model_optimization_config(calibration, batch_size=1, calibset_size=20)

# 量子化前後の最適化を無効化（安定性重視）
pre_quantization_optimization(equalization, policy=disabled)
post_quantization_optimization(finetune, policy=disabled)
post_quantization_optimization(bias_correction, policy=enabled)

# コンパイラ最適化レベル (max=2, 1, 0)
# max: 最も性能が高いが探索時間も長い
performance_param(compiler_optimization_level=max)

# マッピングタイムアウト設定
# デフォルトは約10分/コンテキスト。DeepLabV3+のような複雑なモデルは
# デフォルトでは足りないため延長が必要。
# GPU使用時は探索がCPU比1.7倍速いため、2時間あれば十分な見込み。
allocator_param(splitter_timeout=7200, timeout=7200)
```

---

## トラブルシューティング

### Mapping Failed (Timeout)

**症状**: `[error] Mapping Failed (Timeout, allocation time: Xm Xs)`

**原因**: DFC がネットワークのレイヤーを Hailo-8 の 8 クラスターへ割り当てる  
組み合わせ探索（マッピング）がタイムアウト内に解を見つけられなかった。

**対処**:
1. `model_script.alls` に `allocator_param(splitter_timeout=N, timeout=N)` を追加して延長
2. `CUDA_VISIBLE_DEVICES=0` で GPU を使い探索を高速化
3. `compiler_optimization_level` を `max` から `1` に下げると探索空間が狭まり解を見つけやすくなる（推論速度は低下）

### No GPU chosen and no suitable GPU found

**症状**: DFC 起動時にこのメッセージが出て CPU で動く

**原因**: GPU メモリ使用率が 5% を超えている（デスクトップ環境や Chrome が原因）

**対処**: `CUDA_VISIBLE_DEVICES=0` を環境変数にセットして実行する

```bash
CUDA_VISIBLE_DEVICES=0 hailo compiler ...
```

### AddShortcut アサーションエラー (640×480 での失敗)

**症状**: コンパイラが `AddShortcut` 関連でクラッシュ

**原因**: 入力解像度が大きすぎて単一コンテキストに収まらず、  
マルチコンテキスト分割時にスキップ接続のテンソルフォーマットが不一致になった。

**対処**: 入力解像度を 320×240 に縮小する。  
詳細: `docs/problem_solving/hailo8_sram_overflow_640x480.md`

### `--har-path` is not recognized

**症状**: `hailo optimize` で `--har-path` オプションが使えない

**原因**: DFC 3.33.0 でフラグ名が変更された

**対処**: `--output-har-path` を使う

---

## 参考ファイル

| ファイル | 役割 |
|---|---|
| `src/bmask_gen/hailo_halfsize/export_onnx.py` | PyTorch → ONNX |
| `src/bmask_gen/hailo_halfsize/make_calib_npy.py` | キャリブレーションセット生成 |
| `src/bmask_gen/hailo_halfsize/model_script.alls` | DFC設定ファイル |
| `src/bmask_gen/hailo_halfsize/infer_live.py` | HEF + RealSense リアルタイム推論 |
| `docs/problem_solving/hailo8_sram_overflow_640x480.md` | 640×480 失敗の詳細記録 |
