# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# 私の好み
- とにかくシンプルな構造で書くこと。過度な構造化は不要です。
- コード内のコメント・docstring は日本語で書いてよい。ただし、CLI 画面に表示される文字列 (`print`、`raise`、例外メッセージ等) は**必ず英語**にすること。
- コメント・標準出力・ドキュメントのいずれにも矢印記号 (`→`, `←`, `↑`, `↓`, `->`, `<-` 等) を**一切使わないこと**。
- アルゴリズムのコードはC++で書くこと。
- テストはGoogle Testで書くこと。
- ビルド用のCMakeLists.txtは書くこと。
- 採用された設計判断は、`docs/architecture/` 以下にファイルを作成して詳細に記述すること。特に、判断理由を詳細に記述すること。ひとまず暫定的な判断である場合はその旨を素直に記述すること。
- バグ修正は、`docs/problem_solving/` 内に、バグごとにファイルを作成して保管してください。
- 実装計画は、`docs/development_plan/` 内に書いておいてください。このdevelopment_plan/はclaudeのメモ場として使って良いです。
- 理解を助けるための説明書ドキュメントは、`docs/my_text/` 以下に書いてください。このディレクトリに書く際は特に詳細に書いてほしいです。


# 事前情報
- trackdloは３次元座標空間内でケーブルの代表点座標を認識可能なアルゴリズム名である。
- アルゴリズム本体は `src/pure_trackdlo/`、前処理は `src/preprocessing/`、評価は `src/evaluation/` に切り出し済み。
- `src/trackdlo/src/trackdlo_node.cpp` はROS1参照実装。ROS2ノード作成の参考にするだけで直接編集しない。


# 最終目標: ROS2への移植

trackdlo (ROS1/catkin実装) を **ROS2 (ament_cmake) パッケージとして完全移植する**。

## 移植戦略

アルゴリズムを先にROS非依存ライブラリとして確立し、ROS2ラッパーを薄く書く。

```
目標 (ROS2):
src/trackdlo/    ← ROS2薄ラッパー (ament_cmake) — ノード実装が残り作業
  src/trackdlo_node.cpp  (現在はROS1参照実装。ROS2に書き換える)
        ├─ preprocessing::color_threshold()
        ├─ preprocessing::images_to_pointcloud()
        ├─ preprocessing::compute_visible_nodes()
        └─ tracking_step() [pure_trackdlo]
```

## 完了済み
1. `src/pure_trackdlo/` — ROSなしのtrackdloアルゴリズム (Eigen3のみ)
   - `TrackdloState` / `TrackdloParams` struct + フリー関数設計
2. `src/preprocessing/` — カメラ画像→点群変換・ノード可視性計算 (OpenCV + PCL)
3. `src/evaluation/` — トラッキング精度評価 (ベンチマーク専用)
4. `src/trackdlo/` — ament_cmake パッケージ骨格 (ノード実装はコメントアウト中)

## 残り作業
- `src/trackdlo/src/trackdlo_node.cpp` をROS1からROS2に書き換え
- ROS2 launch ファイル (`.py`) の作成


---

# Build & Development

## colcon ビルド (推奨)

```bash
# ワークスペースルートから全パッケージを一括ビルド
# pure_trackdlo → preprocessing → evaluation → trackdlo の順に自動解決される
colcon build

# 特定パッケージだけビルド (依存も含む)
colcon build --packages-up-to trackdlo

# テスト (cmake パッケージは ctest を自動実行)
colcon test
colcon test-result --verbose
```

## 個別ビルド (開発時の素早い確認用)

```bash
# pure_trackdlo
cd src/pure_trackdlo && mkdir -p build && cd build
cmake .. && make && ctest --output-on-failure

# preprocessing
cd src/preprocessing && mkdir -p build && cd build
cmake .. && make && ctest --output-on-failure
```

## テスト (Google Test 単体)
```bash
# 単一テストケースの実行 (バイナリ名: test_utils, test_trackdlo, test_preprocessing)
./build/test_trackdlo --gtest_filter=TestSuite.TestName
./build/test_utils --gtest_filter=TestSuite.TestName
./build/test_preprocessing --gtest_filter=TestSuite.TestName
```

---

# コードアーキテクチャ

## ディレクトリ構成
```
trackdlo_project/                ← ROS2ワークスペース兼リポジトリルート
├── src/                         # ← 全パッケージはここに格納
│   ├── pure_trackdlo/           # 非ROS2: trackdloアルゴリズム (Eigen3のみ, plain CMake)
│   │   ├── include/trackdlo.h   # TrackdloState / TrackdloParams struct + フリー関数宣言
│   │   ├── include/utils.h      # 幾何学ユーティリティ
│   │   ├── src/trackdlo.cpp     # cpd_lle(), tracking_step() 実装
│   │   ├── src/utils.cpp
│   │   ├── tests/               # Google Test (17テスト: test_trackdlo x6, test_utils x11)
│   │   └── CMakeLists.txt
│   │
│   ├── preprocessing/           # 非ROS2: カメラ画像→点群変換 (OpenCV + PCL, plain CMake)
│   │   ├── include/preprocessing.h
│   │   ├── src/preprocessing.cpp
│   │   ├── tests/               # Google Test (5テスト)
│   │   └── CMakeLists.txt
│   │
│   ├── evaluation/              # 非ROS2: トラッキング精度評価 (plain CMake)
│   │   ├── include/evaluator.h  # evaluator クラス (class設計。他と異なる点に注意)
│   │   ├── src/evaluator.cpp
│   │   └── CMakeLists.txt
│   │
│   └── trackdlo/                # ROS2パッケージ (ament_cmake)
│       ├── package.xml          # <depend>pure_trackdlo</depend> 等でビルド順制御
│       ├── CMakeLists.txt       # ノード実装はコメントアウト中 (残り作業)
│       ├── src/
│       │   ├── trackdlo_node.cpp    # ROS1参照実装 → ROS2書き換えが残り作業
│       │   └── run_evaluation.cpp   # 同上
│       ├── docs/COLOR_THRESHOLD.md  # HSV閾値チューニングガイド
│       ├── scripts/             # Python スクリプト (initialize.py 等)
│       ├── launch/              # ROS1 launch ファイル (ROS2 .py に書き換えが残り作業)
│       ├── config/              # カメラ設定プリセット
│       └── rviz/                # RViz 設定
│
├── docs/
│   ├── architecture/            # 設計判断の詳細記録
│   │   └── trackdlo_class_vs_functions.md
│   ├── development_plan/        # 開発計画 (Claudeのメモ場)
│   ├── my_text/                 # 理解を助けるための詳細説明書
│   ├── problem_solving/         # バグ修正の記録
│   └── LEARN_MORE.md            # パラメータ調整ガイド
│
└── trackdlo_class_vs_functions.md  # ← docs/architecture/ に移動すべき残留ファイル
```

---

# Python 環境 (uv)

ワークスペースルート (`trackdlo_project/`) が uv ワークスペースのルート。
メンバーパッケージは `src/trackdlo_examples/`。

```bash
# 初回セットアップ (Python 3.10.11 + 全依存を .venv/ に構築)
uv sync

# .venv を有効化して実行
source .venv/bin/activate
python src/bmask_gen/scripts/train.py

# または有効化なしで直接実行
uv run python src/bmask_gen/scripts/train.py
```

`cache-dir = "src/trackdlo_examples/.uv-cache"` により、キャッシュはリポジトリ内に置かれる (git 管理外)。
GPU 版 torch は PyPI ではなく `https://download.pytorch.org/whl/cu130` から取得 (`pyproject.toml` の `[[tool.uv.index]]` 参照)。

---

# src/bmask_gen/ — ケーブルセグメンテーション (DeepLabV3+)

trackdlo のカラー閾値 (`color_threshold()`) の代替として、学習ベースのバイナリセグメンテーションを行うパッケージ。

## ディレクトリ構成

```
src/bmask_gen/
├── scripts/          ← 既存データ (data/) を使う学習・推論スクリプト
│   ├── train_deeplabv3plus_binary.py
│   ├── infer_live_cpu.py / infer_live_gpu.py
│   ├── img_capture.py / view_cam.py
│   └── practice/
├── scripts_aoyama/   ← data_aoyama/ を使う学習・推論スクリプト (新規)
│   ├── prepare_dataset.py   ← JSON→マスク変換 + train/val 分割
│   ├── train.py
│   ├── infer_live_cpu.py
│   └── infer_live_gpu.py
├── data/             ← 小規模テストデータ (6枚, train/val 分割済み)
├── data_aoyama/      ← 本番データ (103枚 JPG + 98枚 LabelMe JSON)
│   └── raw/
│       ├── raw_images/      ← 1.jpg ～ 102.jpg (640×480)
│       └── annotated_json/  ← 1.json ～ 98.json (LabelMe polygon形式)
├── weights/          ← 学習済み重み (.pth)
├── tools/            ← データ前処理ユーティリティ
└── docs/annotation_workflow.md
```

## data_aoyama のアノテーション形式

LabelMe polygon JSON。ラベルは `"cable"` 1種類のみ。

```json
{
  "shapes": [
    { "label": "cable", "points": [[x1,y1], [x2,y2], ...], "shape_type": "polygon" }
  ],
  "imageHeight": 480, "imageWidth": 640
}
```

`{n}.json` と `{n}.jpg` は番号で対応 (98 JSON / 103 JPG → 5枚はアノテーションなし)。

## データ前処理ワークフロー (scripts_aoyama/)

```
raw_images/{n}.jpg + annotated_json/{n}.json
    │
    ▼ prepare_dataset.py   (PIL ImageDraw でポリゴン → バイナリマスク)
data_aoyama/dataset/
    ├── images/train|val/  ← JPG コピー
    └── masks/train|val/   ← PNG バイナリマスク (0=背景, 1=ケーブル)
    │
    ▼ train.py
weights_aoyama/best_deeplabv3plus_cable.pth
```

## 学習スクリプトの共通構造

`CableDataset(Dataset)` → `DataLoader` → `smp.DeepLabV3Plus(encoder="resnet34", classes=1, activation=None)` → `BCEWithLogitsLoss` → `Adam(lr=1e-4)`

- モデル出力は raw logit (sigmoid なし)。推論時は `torch.sigmoid()` を別途適用。
- 重みは `best_*.pth` (val IoU 最良) と `last_*.pth` (最終エポック) の2本保存。

## TrackDLOアルゴリズムの処理フロー

```
BGR画像 + 深度画像
    │
    ▼ preprocessing::color_threshold()
2値マスク
    │
    ▼ preprocessing::images_to_pointcloud()
入力点群 X (N×3, Eigen::MatrixXd)
    │
    ▼ preprocessing::compute_visible_nodes()  ← tracking_step() の前に呼ぶ
visible_nodes, visible_nodes_extended
    │
    ▼ tracking_step(state, X, visible_nodes, visible_nodes_extended, params)
    ├── cpd_lle() で guide_nodes を粗く合わせる (前処理)
    ├── オクルージョン状態判定
    │   ├── 全可視 / 小オクルージョン → traverse_euclidean() x2 + 平均
    │   ├── 中間オクルージョン        → traverse_euclidean() x2 + 結合
    │   ├── 片端オクルージョン        → traverse_euclidean() x1
    │   └── 両端オクルージョン        → traverse_euclidean() (alignment=2)
    └── cpd_lle() で Y を最終更新
    │
    ▼
state.Y (M×3行列, Eigen::MatrixXd) ← ノード座標の最終推定値
```

## 主要な関数・データ構造

| シンボル | ファイル | 役割 |
|---|---|---|
| `TrackdloState` | `pure_trackdlo/include/trackdlo.h` | フレーム間で引き継ぐ可変状態 (Y, sigma2 等) |
| `TrackdloParams` | `pure_trackdlo/include/trackdlo.h` | セッション全体のパラメータ (beta, lambda 等) |
| `cpd_lle()` | `pure_trackdlo/src/trackdlo.cpp` | CPD+LLE登録のE/Mステップ (アルゴリズム中核) |
| `tracking_step()` | `pure_trackdlo/src/trackdlo.cpp` | フレームごとの処理オーケストレーション |
| `traverse_euclidean()` | `pure_trackdlo/src/trackdlo.cpp` | Pure Pursuitによるオクルージョン補間 |
| `reg()` | `pure_trackdlo/src/utils.cpp` | 前処理用の簡易CPD登録 |
| `sort_pts()` | `pure_trackdlo/src/utils.cpp` | 最近傍順によるノード並べ替え |
| `line_sphere_intersection()` | `pure_trackdlo/src/utils.cpp` | パス追跡用レイキャスティング |
| `color_threshold()` | `preprocessing/src/preprocessing.cpp` | HSVによるケーブル領域抽出 |
| `images_to_pointcloud()` | `preprocessing/src/preprocessing.cpp` | ピンホール逆投影 + VoxelGrid |
| `compute_visible_nodes()` | `preprocessing/src/preprocessing.cpp` | ノード可視性・セルフオクルージョン判定 |
| `evaluator` クラス | `evaluation/include/evaluator.h` | 予測ノードとグラウンドトゥルースの誤差計算 (class設計) |

## 主要な依存ライブラリ
- **Eigen3** (3.3+): 行列演算 (`Eigen::MatrixXd` が主要データ型)
- **PCL 1.8**: VoxelGridダウンサンプリング
- **OpenCV**: HSVマスク処理・投影描画
