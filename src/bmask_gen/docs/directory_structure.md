# ディレクトリ構成

## 全体像

```
ws_deeplab/
├── scripts/       ← 実行スクリプト
├── data/          ← データ（生データ・学習用データセット）
├── tools/         ← データ前処理スクリプト
├── weights/       ← 学習済みモデルの重み
├── output/        ← 推論結果の出力先
└── docs/          ← このドキュメント群
```

---

## scripts/

実際に実行するPythonスクリプト。どのディレクトリからでも実行できる（内部で `__file__` を使ってパスを解決している）。

| ファイル | 役割 |
|---|---|
| `train.py` | DeepLabV3Plus を学習し、重みを `weights/` に保存 |
| `infer.py` | `data/test.jpg` に対して推論し、マスクを `output/` に保存 |
| `capture.py` | カメラを起動し、`s` キーで `data/raw/` に連番PNG保存 |
| `view_cam.py` | カメラプレビューのみ（保存なし）。映り方の確認用 |

---

## data/

```
data/
├── raw/                 ← カメラで撮影した生データとアノテーション結果
│   ├── img_XXXX.png     ← capture.py で撮影した元画像
│   ├── img_XXXX.json    ← labelme で生成したアノテーションファイル
│   ├── img_XXXX_json/   ← convert_json_to_bmask.sh で展開したフォルダ
│   │   ├── img.png
│   │   ├── label.png        ← ラベル画像（ピクセル値がクラスID）
│   │   ├── label_viz.png    ← 可視化用ラベル画像
│   │   └── label_names.txt  ← クラス名一覧（_background_, cable）
│   └── masks/           ← collect_png.sh で収集したラベル画像
│       └── img_XXXX.png ← label.png のコピー（バイナリ化前）
│
├── dataset/             ← train.py に渡す学習用データセット（手動で配置）
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── masks/
│       ├── train/
│       └── val/
│
└── test.jpg             ← infer.py の入力画像
```

### `data/raw/` の状態遷移

```
capture.py で撮影
  → img_XXXX.png が生成

labelme でアノテーション
  → img_XXXX.json が生成

convert_json_to_bmask.sh を実行
  → img_XXXX_json/ フォルダが生成

collect_png.sh を実行
  → masks/img_XXXX.png が生成（label.png のコピー）

convert_png2binary.py を実行
  → masks/img_XXXX.png がバイナリ（0 or 1）に上書き
```

---

## tools/

`data/raw/` 内で実行する前処理スクリプト群。

| ファイル | 役割 | 実行場所 |
|---|---|---|
| `convert_json_to_bmask.sh` | `.json` を labelme CLI で展開し `*_json/` フォルダを生成 | `data/raw/` |
| `collect_png.sh` | `*_json/label.png` を `masks/` にコピー | `data/raw/` |
| `convert_png2binary.py` | `masks/*.png` をバイナリ（0 or 1）値に変換 | プロジェクトルート |

---

## weights/

`train.py` が学習後に重みファイルを保存する場所。

```
weights/
└── deeplabv3plus_cable.pth
```

---

## output/

`infer.py` が推論結果を保存する場所。

```
output/
├── pred_mask.png      ← バイナリマスク（ピクセル値: 0 or 1）
└── pred_mask_vis.png  ← 可視化用マスク（ピクセル値: 0 or 255）
```
