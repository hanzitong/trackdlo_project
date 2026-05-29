# ワークスペース概要

このドキュメントはワークスペース全体の構造・各パッケージの目的・実装概要をまとめたものです。

---

## 全体目標

ケーブル（DLO: Deformable Linear Object）の3次元ノード座標をリアルタイムで推定する
trackdlo アルゴリズムを **ROS2 環境および Windows (Python) 環境で動作させる**。

---

## パッケージ一覧

```
trackdlo_project/
└── src/
    ├── pure_trackdlo/      アルゴリズム本体 (C++, ROS 非依存)
    ├── preprocessing/      カメラ画像→点群変換 (C++, ROS 非依存)
    ├── evaluation/         精度評価 (C++, ROS 非依存)
    ├── trackdlo_cdll/      Python 向け C API + ctypes ラッパー
    ├── trackdlo/           ROS2 ノード (ament_cmake, 未完成)
    ├── bmask_gen/          DeepLabV3+ バイナリマスク生成 (Python)
    ├── realsense_practice/ RealSense SDK 練習コード
    └── trackdlo_python/    削除済み (pybind11 版。trackdlo_cdll に移行・消去)
```

---

## アーキテクチャ構造

### 依存関係

```
[pure_trackdlo]  ←── アルゴリズム核心。他の全パッケージが依存する
      │
      ├── [preprocessing]   ← pure_trackdlo への入力点群を生成する
      │
      ├── [evaluation]      ← 精度評価。pure_trackdlo に依存しない独立ライブラリ
      │
      ├── [trackdlo]        ← ROS2 ノード。preprocessing + pure_trackdlo を呼ぶ
      │
      └── [trackdlo_cdll]   ← C API 層。pure_trackdlo を Windows DLL として公開する
```

### データフロー (本番パイプライン)

```
RealSense D405
  カラー画像 (BGR)
  深度画像 (16bit, mm単位)
        │
        ▼ bmask_gen (DeepLabV3+) ← ニューラルネットでケーブル領域を検出
  バイナリマスク
        │
        ▼ preprocessing::images_to_pointcloud()
  入力点群 X (N×3, Eigen::MatrixXd)
        │
        ▼ preprocessing::compute_visible_nodes()
  visible_nodes, visible_nodes_extended
        │
        ▼ trackdlo::tracking_step(state, X, ...)
  state.Y (M×3) ← ケーブルのノード座標 (最終出力)
```

---

## 各パッケージの詳細

---

### pure_trackdlo

**目的**: trackdlo アルゴリズム本体。ROS にも Python にも依存しない純粋な C++ ライブラリ。

**依存**: Eigen3 のみ

**ビルド成果物**: `libpure_trackdlo.a` (静的ライブラリ)

**主要ファイル**:

| ファイル | 役割 |
|---|---|
| `include/trackdlo.h` | `TrackdloState`, `TrackdloParams` 構造体 + 関数宣言 |
| `include/utils.h` | 幾何学ユーティリティ関数宣言 |
| `src/trackdlo.cpp` | `cpd_lle()`, `tracking_step()` の実装 |
| `src/utils.cpp` | `sort_pts()`, `reg()`, `line_sphere_intersection()` 等の実装 |
| `examples/hello_trackdlo.cpp` | 基本的な使い方のサンプル |
| `examples/state_lifecycle.cpp` | メモリ管理パターン (stack / optional / unique_ptr) のサンプル |
| `tests/` | Google Test (test_trackdlo x6, test_utils x11) |

**主要なデータ構造**:

```cpp
// セッション全体で変化しないパラメータ (毎フレーム同じ値を渡す)
struct TrackdloParams {
    double beta, lambda, alpha, k_vis, mu;
    int    max_iter;
    double tol, beta_pre_proc, lambda_pre_proc, lle_weight, visibility_threshold;
};

// フレーム間で引き継ぐ可変状態 (毎フレーム更新される)
struct TrackdloState {
    Eigen::MatrixXd Y;                          // ノード座標 (M×3)
    Eigen::MatrixXd guide_nodes;                // 前フレームの可視ノード座標
    double sigma2;                              // ガウシアン分散
    std::vector<double> geodesic_coord;         // 各ノードの累積弧長
    std::vector<Eigen::MatrixXd> correspondence_priors; // オクルージョン補間結果
};
```

**設計方針**: クラスではなく struct + フリー関数。ROS2 ノードクラスが
`TrackdloState` をメンバ変数として持ち、コールバックから `tracking_step()` を
呼ぶだけのシンプルな構造にするため。

---

### preprocessing

**目的**: カメラ画像 (BGR + 深度) から `pure_trackdlo` への入力点群を生成する。
ROS に依存しない前処理ライブラリ。

**依存**: OpenCV, Eigen3, PCL 1.8

**ビルド成果物**: `libpreprocessing.a` (静的ライブラリ)

**主要関数**:

| 関数 | 入力 | 出力 |
|---|---|---|
| `color_threshold()` | BGR画像, HSV閾値 | バイナリマスク (cv::Mat) |
| `images_to_pointcloud()` | BGR画像, 深度画像, カメラ行列, マスク | 点群 (N×3 MatrixXd) |
| `compute_visible_nodes()` | ノード座標Y, 点群X, カメラ行列 | visible_nodes, visible_nodes_extended |

**処理詳細**:
- `color_threshold()`: BGR → HSV 変換後、指定色範囲のピクセルを白にした2値マスクを返す
- `images_to_pointcloud()`: マスク内の有効ピクセルをピンホール逆投影で3D座標に変換。PCL VoxelGrid で密度を均一化
- `compute_visible_nodes()`: 前フレームノードをカメラに投影し、点群との距離と自己オクルージョンで可視判定

---

### evaluation

**目的**: trackdlo のトラッキング精度をグラウンドトゥルースと比較して定量評価する。
ベンチマーク・実験専用ライブラリ。

**依存**: Eigen3, OpenCV, PCL 1.8

**設計**: 他パッケージと異なり **class 設計**。評価セッションのメタデータ (試行番号・オクルージョン率・保存先等) を保持するため。

**主要メソッド**:

| メソッド | 役割 |
|---|---|
| `get_ground_truth_nodes()` | RGB画像+点群からマーカー色でグラウンドトゥルースノードを検出 |
| `compute_error()` | 予測ノードと真値ノードの双方向区分的 Hausdorff 誤差を計算 |
| `compute_and_save_error()` | 誤差計算してファイルに保存 |

---

### trackdlo_cdll

**目的**: `pure_trackdlo` の C++ インターフェースを C インターフェースで隠し、
Python (ctypes) から呼べる共有ライブラリとして公開する。
**Windows 64-bit 向けクロスコンパイルに対応**することが主目的。

**依存**: pure_trackdlo (静的リンク), Eigen3

**ビルド成果物**:
- Linux: `libtrackdlo_c.so`
- Windows (クロスコンパイル): `trackdlo_c.dll`

**構成**:

```
trackdlo_cdll/
├── include/trackdlo_c_api.h     C API ヘッダ (extern "C" で公開)
├── src/trackdlo_c_api.cpp       C API の実装 (C++ ↔ C 型変換)
├── python/trackdlo_cdll.py      Python ctypes ラッパー
├── examples/
│   ├── hello_trackdlo_cdll.py   基本的な使い方
│   ├── memory_management.py     メモリ管理パターン (with / try-finally / del)
│   └── realsense_trackdlo.py    RealSense + DeepLab + trackdlo フルパイプライン
└── cmake/toolchain-mingw64.cmake  Windows クロスコンパイル用ツールチェーン
```

**なぜ C API 層が必要か**:

```
[問題] Python ctypes は C の関数しか呼べない
  ・C++ は関数名をマングリング (例: tracking_step → _ZN8trackdlo13...) する
  ・ctypes は文字列で関数を探すため、マングリングされると見つけられない
  ・Eigen::MatrixXd や std::vector は C 型ではないので ctypes で渡せない

[解決] extern "C" で包んだ C API 関数を作る
  ・TdloState = void* (C++ オブジェクトの不透明ハンドル)
  ・引数はすべて double*, int など C 型のみ
  ・Python ↔ C 型変換を trackdlo_c_api.cpp 内で行う
```

**Windows クロスコンパイル方法**:

```bash
# 1. pure_trackdlo を Windows 向けにビルド・インストール
mkdir -p build_win/pure_trackdlo && cd build_win/pure_trackdlo
cmake ../../src/pure_trackdlo \
  -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
  -DCMAKE_INSTALL_PREFIX=../../install_win
make && make install

# 2. trackdlo_cdll を Windows 向けにビルド → trackdlo_c.dll が生成される
mkdir -p build_win/trackdlo_cdll && cd build_win/trackdlo_cdll
cmake ../../src/trackdlo_cdll \
  -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
  -DCMAKE_PREFIX_PATH=../../install_win
make
```

---

### trackdlo (ROS2 ノード)

**目的**: `pure_trackdlo` と `preprocessing` を組み合わせた ROS2 ノード。
RealSense トピックを購読し、トラッキング結果を publish する。

**依存**: ament_cmake, rclcpp, pure_trackdlo, preprocessing

**状態**: **未完成**。`trackdlo_node.cpp` は ROS1 参照実装のまま。ROS2 への書き換えが残り作業。

---

### bmask_gen

**目的**: DeepLabV3+ を使ってカラー画像からケーブルのバイナリマスクを生成する。
Python スクリプト群。ROS にも C++ にも依存しない。

**依存**: PyTorch, torchvision, pyrealsense2, OpenCV

**主要スクリプト**:

| スクリプト | 役割 |
|---|---|
| `scripts/train_deeplabv3plus_binary.py` | バイナリセグメンテーション用にファインチューニング |
| `scripts/infer_live.py` | RealSense からのライブ映像でリアルタイム推論 |
| `scripts/img_capture.py` | 学習データ収集用の画像キャプチャ |
| `tools/convert_png2binary.py` | アノテーション画像を2値ラベルに変換 |

---

### realsense_practice

**目的**: RealSense D405 SDK の練習コード。本番パイプラインの参照実装ではない。

**内容**:
- `python/01_hello_realsense.py` : カラー・深度フレームの基本取得
- `python/02_aligned_depth.py`   : カラーに揃えたアライン済み深度の取得
- `python/03_pointcloud.py`      : SDK の内蔵 PointCloud クラスの使い方
- `src/` : 同内容の C++ 版

---

## ビルド方法

### Linux (colcon、ROS2 環境)

```bash
cd trackdlo_project
colcon build
source install/setup.bash
```

### Windows DLL クロスコンパイル (Ubuntu 上)

```bash
# 前提: sudo apt install mingw-w64

# Step 1: pure_trackdlo の静的ライブラリを Windows 向けにビルド
mkdir -p build_win/pure_trackdlo && cd build_win/pure_trackdlo
cmake ../../src/pure_trackdlo \
  -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
  -DCMAKE_INSTALL_PREFIX=../../install_win
make && make install && cd ../..

# Step 2: trackdlo_c.dll をビルド
mkdir -p build_win/trackdlo_cdll && cd build_win/trackdlo_cdll
cmake ../../src/trackdlo_cdll \
  -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
  -DCMAKE_PREFIX_PATH=../../install_win
make
```

生成物: `build_win/trackdlo_cdll/trackdlo_c.dll`

Windows 側では `trackdlo_c.dll` を `trackdlo_cdll/python/trackdlo_cdll.py` と同じディレクトリに置いて使う。
