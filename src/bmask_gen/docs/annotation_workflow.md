# アノテーションワークフロー

カメラで画像を撮影し、バイナリマスクを生成するまでの手順。

---

## 全体の流れ

```
① カメラで撮影     →  data/raw/img_XXXX.png
② labelme でアノテーション  →  data/raw/img_XXXX.json
③ json を展開     →  data/raw/img_XXXX_json/
④ label.png を収集 →  data/raw/masks/img_XXXX.png
⑤ バイナリ化       →  data/raw/masks/img_XXXX.png（0 or 1 に上書き）
```

---

## ① 撮影

```bash
python scripts/capture.py
```

- カメラウィンドウが開く
- **`s` キー** を押すたびに `data/raw/img_XXXX.png` に保存される
- **`ESC`** で終了

> **カウンタについて**  
> `capture.py` 内の `count = 5` が連番の開始番号。  
> 既存ファイルと被らないよう、撮影前に `data/raw/` の最大番号を確認して書き換える。

---

## ② アノテーション（labelme）

```bash
python3 -m labelme data/raw/
```

- labelme が起動し、`data/raw/` 内の画像が左ペインに一覧表示される
- 1枚ずつ画像を開いてポリゴンを描く

### 操作手順

1. 左ペインから画像を選択
2. メニュー「Edit → Create Polygons」または **`Ctrl+R`** でポリゴンモードに入る
3. ケーブルの輪郭をクリックしてポリゴンを描く
4. 閉じると「ラベル名を入力」ダイアログが出るので **`cable`** と入力して OK
5. **`Ctrl+S`** で保存 → `img_XXXX.json` が生成される
6. 次の画像へ（`d` キーで次、`a` キーで前）

> **ラベルは `cable` の1種類のみ**（`_background_` は自動で割り当て）。  
> ケーブルが複数本ある場合は、本数分ポリゴンを描く。

---

## ③ json の展開（labelme CLI）

`data/raw/` に移動してシェルスクリプトを実行する。

```bash
cd data/raw
bash ../../tools/convert_json_to_bmask.sh
```

各 `.json` に対して `img_XXXX_json/` フォルダが生成される。

```
img_XXXX_json/
├── img.png          ← 元画像
├── label.png        ← ラベル画像（ピクセル値 = クラスID）
├── label_viz.png    ← 確認用カラー画像
└── label_names.txt  ← _background_ / cable
```

> **確認ポイント**: `label_viz.png` を開いて、ケーブル部分が正しくハイライトされているか確認する。

---

## ④ label.png の収集

```bash
# ③ のあと、data/raw/ にいる状態で続けて実行
bash ../../tools/collect_png.sh
```

各 `img_XXXX_json/label.png` が `masks/img_XXXX.png` にコピーされる。

```
data/raw/masks/
├── img_0000.png
├── img_0001.png
└── ...
```

---

## ⑤ バイナリ化

プロジェクトルートに戻ってから実行する。

```bash
cd ../..   # ws_deeplab/ に戻る
python tools/convert_png2binary.py
```

`masks/*.png` のピクセル値が `0`（背景）または `1`（ケーブル）に上書きされる。

> **なぜバイナリ化が必要か**  
> labelme が出力する `label.png` のピクセル値はクラスID（背景=0、cable=1）だが、  
> 環境によっては `label_viz.png` のRGB値が混入する場合がある。  
> `convert_png2binary.py` で「0より大きければ1」に統一し、マスクを確実に 0/1 にする。

---

## ⑥ 学習データセットへの配置（次ステップ）

マスクが完成したら、`data/raw/` の画像とマスクを train/val に振り分けて `data/dataset/` に配置する。

```
data/dataset/
├── images/
│   ├── train/  ← data/raw/img_XXXX.png をコピー
│   └── val/
└── masks/
    ├── train/  ← data/raw/masks/img_XXXX.png をコピー
    └── val/
```

配置後、学習は以下で実行する。

```bash
python scripts/train.py
```
