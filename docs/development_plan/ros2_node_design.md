# ROS2ノード設計メモ

作成日: 2026-05-11

---

## 概要

`src/trackdlo/src/trackdlo_node.cpp` を ROS1 実装から ROS2 実装に書き換えた。
あわせて `pure_trackdlo` と `preprocessing` に namespace を追加した。

---

## 判断1: namespace の命名

### 採用: `namespace trackdlo` / `namespace preprocessing`

**理由**:
- ROS2ノードから呼ぶ際に `trackdlo::tracking_step()` / `preprocessing::color_threshold()` と書けることで、アルゴリズムライブラリへの呼び出しが一目瞭然になる。
- パッケージ名は `pure_trackdlo` だが、「pure」はROSに依存しない旨を表す修飾語なので、namespace名としては `trackdlo` が自然。
- 型名 (`TrackdloState`, `TrackdloParams`) はリネームしない。差分を最小化し、テストが読みやすいまま保つため。

**検討した代替案**:
- `namespace pure_trackdlo`: パッケージ名に合わせた案。`pure_trackdlo::tracking_step()` は冗長に見えたため却下。

---

## 判断2: ノード設計 (class vs グローバル変数)

### 採用: `class TrackdloNode : public rclcpp::Node`

**理由**:
- ROS2 チュートリアルの標準パターン。フレーム間状態 (`state_`, `proj_matrix_` 等) をメンバ変数として管理できる。
- ROS1 実装はグローバル変数で状態を持っていたが、ROS2 では `rclcpp::Node` を継承するクラスの中で管理するのが標準。

---

## 判断3: 時刻同期 (ApproximateTime vs ExactTime)

### 採用: `message_filters::ApproximateTimeSynchronizer`

**理由**:
- ROS1 では `TimeSynchronizer` (ExactTime) を使っていた。
- RealSense の aligned depth は色画像と同期されているが、実機では数ms のタイムスタンプずれが生じることがある。
- `ApproximateTimeSynchronizer` はずれが小さい場合でも正しくマッチするため、ExactTime より安全。

---

## 判断4: 初期化フロー

### 2段階初期化: `on_init_nodes` + `on_camera_info` → `try_initialize()`

**理由**:
- 初期ノード (PointCloud2) とカメラ行列 (CameraInfo) は別々のタイミングで届く。
- 両方そろってから `state_` を初期化するため、`try_initialize()` 内で両フラグを確認する。
- 一度受け取れれば十分なので `sub.reset()` でサブスクリプションを解除する（メモリ節約）。

---

## 判断5: MarkerArray のライフタイム

### 採用: `rclcpp::Duration(std::chrono::seconds(1))`

**理由**:
- ROS1 実装では DELETEALL マーカーを後送りしていた。
- ROS2 では 1秒の lifetime を設定することで、次フレームの結果がパブリッシュされるまでの間だけ表示される。ノードが 1Hz 以上で動いている限り常に最新マーカーが表示される。
- ROS1 の DELETEALL 方式と比べてコードがシンプルになる。

---

## 判断6: オクルージョンマスク (optional)

### `occ_mask_` は空の場合は cv::Mat() を渡す

**理由**:
- シミュレーション時のみ使う機能。`/mask_with_occlusion` が来ない場合はマスク無しで動作する。
- `preprocessing::images_to_pointcloud()` は `occ_mask` が空の場合を内部でハンドルするため、ノード側で分岐を書く必要がない。

---

## 未実装 / 残り作業

- `initialize.py` の C++ 化 (ROS2 初期化ノード)。現状は ROS1 の `initialize.py` を参照用として残している。ROS2 環境では Python ノードとして移植するか、C++ で書き直す必要がある。
- ROS2 の `image_transport` 対応。現状は `sensor_msgs::msg::Image` を直接購読している。帯域が問題になる場合は `image_transport` を使う。
- パラメータの動的変更 (`rclcpp::ParameterEventHandler`)。HSV 閾値をランタイムに変えたい場合に必要。

---

## ビルド確認

```bash
# ワークスペースルートから
colcon build --packages-up-to trackdlo

# ノードを起動する前に source する
source install/setup.bash

# 起動 (initialize.py も別ターミナルで起動が必要)
ros2 launch trackdlo trackdlo_node.launch.py
```
