# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# 私の好み
- とにかくシンプルな構造で書くこと。過度な構造化は不要です。
- アルゴリズムのコードはC++で書くこと。
- テストはGoogle Testで書くこと。
- ビルド用のCMakeLists.txtは書くこと。
- 採用された設計判断は、`docs/architecture/` 以下にファイルを作成して詳細に記述すること。特に、判断理由を詳細に記述すること。ひとまず暫定的な判断である場合はその旨を素直に記述すること。
- バグ修正は、`docs/problem_solving/` 内に、バグごとにファイルを作成して保管してください。
- 実装計画は、`docs/development_plan/` 内に書いておいてください。このdevelopment_plan/はclaudeのメモ場として使って良いです。
- 理解を助けるための説明書ドキュメントは、`docs/my_text/` 以下に書いてください。このディレクトリに書く際は特に詳細に書いてほしいです。


# 事前情報
- trackdloは３次元座標空間内でケーブルの代表点座標を認識可能なアルゴリズム名である。
- `trackdlo/` 以下はROS1 (catkin) 実装の残骸。参照用として一部保持しているが、新規実装には使わない。
- アルゴリズム本体は `pure_trackdlo/`、前処理は `preprocessing/`、評価は `evaluation/` に切り出し済み。


# 最終目標: ROS2への移植

trackdlo (ROS1/catkin実装) を **ROS2 (ament_cmake) パッケージとして完全移植する**。

## 移植戦略

アルゴリズムを先にROS非依存ライブラリとして確立し、ROS2ラッパーを薄く書く。

```
現状:                           目標 (ROS2):
trackdlo/ (ROS1残骸)            trackdlo_node_ros2/    ← ROS2薄ラッパー
  trackdlo_node.cpp               trackdlo_node.cpp
  (参照用のみ)                      (純粋な接続層のみ)
                                      ├─ preprocessing::color_threshold()
                                      ├─ preprocessing::images_to_pointcloud()
                                      ├─ preprocessing::compute_visible_nodes()
                                      └─ tracking_step() [pure_trackdlo]
```

## 完了済み
1. `pure_trackdlo/` — ROSなしのtrackdloアルゴリズム (Eigen3のみ)
   - `TrackdloState` / `TrackdloParams` struct + フリー関数設計
2. `preprocessing/` — カメラ画像→点群変換・ノード可視性計算 (OpenCV + PCL)
3. `evaluation/` — トラッキング精度評価 (ベンチマーク専用)
4. `trackdlo/` の移植済みファイル削除 (ヘッダ3本 + 実装3本 + ROS1ユーティリティ)

## 残り作業
- ROS2パッケージ (`trackdlo_node_ros2/`) の作成


---

# Build & Development

## 各ライブラリのビルド

```bash
# pure_trackdlo
cd pure_trackdlo && mkdir -p build && cd build
cmake .. && make && ctest --output-on-failure

# preprocessing
cd preprocessing && mkdir -p build && cd build
cmake .. && make && ctest --output-on-failure

# evaluation
cd evaluation && mkdir -p build && cd build
cmake .. && make
```

## テスト (Google Test)
```bash
# ビルド後、build/ ディレクトリから実行
./build/<test_binary>
# 単一テストケースの実行
./build/<test_binary> --gtest_filter=TestSuite.TestName
```

---

# コードアーキテクチャ

## ディレクトリ構成
```
trackdlo_ros2/                   ← このリポジトリのルート
├── pure_trackdlo/               # ROS非依存のtrackdloアルゴリズム (Eigen3のみ)
│   ├── include/trackdlo.h       # TrackdloState / TrackdloParams struct + フリー関数宣言
│   ├── include/utils.h          # 幾何学ユーティリティ
│   ├── src/trackdlo.cpp         # cpd_lle(), tracking_step() 実装
│   ├── src/utils.cpp
│   ├── tests/                   # Google Test (17テスト)
│   └── CMakeLists.txt
│
├── preprocessing/               # カメラ画像→点群変換・ノード可視性計算 (OpenCV + PCL)
│   ├── include/preprocessing.h  # color_threshold / images_to_pointcloud / compute_visible_nodes
│   ├── src/preprocessing.cpp
│   ├── tests/                   # Google Test (5テスト)
│   └── CMakeLists.txt
│
├── evaluation/                  # トラッキング精度評価 ベンチマーク専用
│   ├── include/evaluator.h      # evaluator クラス
│   ├── src/evaluator.cpp
│   └── CMakeLists.txt
│
├── trackdlo/                    # ROS1実装の残骸 (参照用のみ, 新規実装に使わない)
│   └── src/
│       ├── trackdlo_node.cpp    # ROS2ノード作成時の参照元 → 完成後削除可
│       ├── run_evaluation.cpp   # ROS2評価ランナー作成時の参照元 → 完成後削除可
│       ├── initialize.py        # DLO初期化ロジック (未移植, C++化が残り作業)
│       └── utils.py             # Python可視化ユーティリティ (未移植)
│
├── docs/
│   ├── architecture/            # 設計判断の詳細記録
│   ├── development_plan/        # 開発計画 (Claudeのメモ場)
│   ├── my_text/                 # 理解を助けるための詳細説明書
│   ├── problem_solving/         # バグ修正の記録
│   ├── COLOR_THRESHOLD.md       # HSV閾値チューニングガイド
│   └── LEARN_MORE.md            # パラメータ調整ガイド
│
├── launch/                      # ROS1 launch ファイル (ROS2 launch 作成時の参照)
├── utils/                       # 残存ユーティリティ (color_picker.py 等)
├── config/                      # PCL設定プリセット
├── rviz/                        # RViz設定 (ROS2でも流用可)
└── CMakeLists.txt               # catkin ビルド設定 (ROS2 CMakeLists 作成時の参照)
```

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

## 主要な依存ライブラリ
- **Eigen3** (3.3+): 行列演算 (`Eigen::MatrixXd` が主要データ型)
- **PCL 1.8**: VoxelGridダウンサンプリング
- **OpenCV**: HSVマスク処理・投影描画
