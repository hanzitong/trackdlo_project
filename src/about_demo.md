# mz07_demo_0624 デモ概要

---

## 手順書一覧

| ファイル | 内容 |
|---|---|
| [how_to_collect_data.md](how_to_collect_data.md) | データ収集・アノテーション・学習の手順 |
| [how_to_run_demo.md](how_to_run_demo.md) | パラメータ設定・ロボット注意点・実行コマンド |

---

## デモの概要

1. コンベア上に紫色のケーブルを置く（人が逐次置く、もしくはすでに置いてある）
2. 人が TP から Python プログラムを起動する
3. ロボットハンドの先端が自動でケーブルの重心位置まで動く一連の動きが開始する
   1. ロボットハンドの先端がホームポジション（撮影可能な定位置）まで移動する
   2. RealSense D405 がコンベア上を撮影する
   3. AI モデルがケーブルを認識する
   4. ケーブルの重心位置を 3D 座標で計算する
   5. 算出された位置にロボットハンドの先端が移動する
4. もう一度デモを行う場合は最初に戻る

---

## ディレクトリ構成

```
src/mz07_demo_0624/
├── demo_full_pipeline_0624.py   メインのデモスクリプト（ロボット制御まで行う）
├── nkr_take_pic.py              RealSense でデータ収集するスクリプト
├── train_deeplabv3plus.py       DeepLabV3+ の学習スクリプト
├── trackdlo_cdll.py             TrackDLO C++ ライブラリの Python ラッパー（編集不要）
│
├── lib/                         C++ 共有ライブラリ（.dll / .so）置き場（編集不要）
│
├── tool/                        補助ツール
│   ├── tool_json_to_bmask.py    LabelMe の JSON をバイナリマスク PNG に変換する
│   └── tool_see_mask.py         マスクの目視確認ツール
│
├── test/                        動作確認用スクリプト（開発者向け）
│
├── nkr_data/                    本番用データ置き場（nkr_take_pic.py が保存先）
│   ├── bgr_0000.png  …          RealSense で撮影した BGR 画像
│   ├── depth_0000.png …         対応する深度画像
│   ├── bgr_0000.json …          LabelMe でアノテーションした JSON
│   └── masks/                   tool_json_to_bmask.py が生成するマスク PNG
│
├── weights/                     学習済みモデルの保存先
│   └── best_deeplabv3plus_cable.pth
│
└── opennr_test/                 OpenNR ロボット接続ライブラリ（編集不要）
    └── .venv/                   opennr_py パッケージが入っている仮想環境
```
