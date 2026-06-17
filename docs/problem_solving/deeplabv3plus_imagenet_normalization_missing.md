# DeepLabV3+ 性能不足の原因: ImageNet 正規化の欠落

## 症状

`train_deeplabv3plus.py` で学習した DeepLabV3+ モデルの IoU が極端に低く、
ケーブルのセグメンテーションがほとんど機能しない。

## 根本原因

**学習・推論スクリプト全てで ImageNet 正規化が欠落していた。**

修正前のコード:
```python
# train_deeplabv3plus.py CableDataset.__getitem__ (修正前)
image = image.astype(np.float32) / 255.0       # [0, 1] のみ
image = np.transpose(image, (2, 0, 1))         # HWC to CHW
# ← ImageNet 正規化がなかった
```

修正後のコード:
```python
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

image = image.astype(np.float32) / 255.0
image = (image - IMAGENET_MEAN) / IMAGENET_STD  # ImageNet 正規化
image = np.transpose(image, (2, 0, 1))
```

同じ修正を推論側 3 箇所にも適用した:
- `test_infer_deeplabv3plus.py`
- `test_cablekeypoint_pipeline.py` の `infer_mask()`

## なぜ正規化が必要なのか (詳細)

### 事前学習モデルと入力分布の関係

ResNet34 エンコーダは `encoder_weights="imagenet"` で読み込まれる。
ImageNet の学習では、全画像に以下の正規化を施していた:

```
normalized_value = (pixel / 255.0 - mean) / std

mean = [0.485, 0.456, 0.406]   # RGB チャンネル別の平均
std  = [0.229, 0.224, 0.225]   # RGB チャンネル別の標準偏差
```

これにより、各チャンネルの値は **平均 0, 標準偏差 1 付近** に揃えられている。

### 正規化なしで何が起きるか

正規化なし (`/255.0` のみ) の場合、入力値の分布は:
- 平均: 約 0.45 (自然画像の平均輝度)
- 範囲: [0, 1]

事前学習時の期待値 (平均 0, std 1) と大きくずれるため:

1. **エンコーダ 1 層目の畳み込み**: 学習済みフィルタが想定していない
   スケール・オフセットの入力を受け取る
2. **BathNorm 層**: 事前学習で推定した running_mean / running_var が
   現在の入力分布に合わなくなる
3. **以降の全層**: 誤った活性化値が伝播し続ける

結果として、何百万枚もの画像で学習した特徴抽出能力がほぼ無効化され、
ランダム初期化に近い状態から学習することになる。

### 転移学習における鉄則

**事前学習時と全く同じ前処理を使わなければならない。**

| 処理 | ImageNet 学習時 | Fine-tuning 時 (正しい) | Fine-tuning 時 (誤り) |
|---|---|---|---|
| BGR→RGB 変換 | RGB | RGB | BGR のまま |
| スケール | /255.0 | /255.0 | そのまま (0-255) |
| 正規化 | (x - mean) / std | (x - mean) / std | なし |

## 副次的な問題: データ不足とデータ拡張なし

正規化問題に加え、以下の問題も性能を制限していた:

- **学習データ 25 枚**: 過学習が起きやすく、汎化性能が出にくい
- **データ拡張なし**: 同じ 25 枚を繰り返すだけで多様性がない

修正: 水平フリップ・垂直フリップを追加 (`augment=True`)

```python
if self.augment:
    if random.random() > 0.5:
        image = np.fliplr(image).copy()
        mask  = np.fliplr(mask).copy()
    if random.random() > 0.5:
        image = np.flipud(image).copy()
        mask  = np.flipud(mask).copy()
```

## 修正ファイル一覧

| ファイル | 修正内容 |
|---|---|
| `src/mz07_demo/train_deeplabv3plus.py` | IMAGENET_MEAN/STD 定数追加、正規化適用、augment 引数追加 |
| `src/mz07_demo/test_infer_deeplabv3plus.py` | IMAGENET_MEAN/STD 追加、正規化適用 |
| `src/mz07_demo/test_cablekeypoint_pipeline.py` | IMAGENET_MEAN/STD 追加、infer_mask() に正規化適用 |

## 対処後の注意

この修正後は **必ず再学習すること**。
修正前に保存した重み (`best_deeplabv3plus_cable.pth`) は
正規化なしの入力で学習されたものであり、修正後の推論コードとは
前処理が一致しないため使用できない。
