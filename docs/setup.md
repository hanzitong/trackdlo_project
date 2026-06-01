# セットアップ手順

## 前提条件

| ツール | 確認コマンド |
|--------|-------------|
| ROS2 Humble | `ros2 --version` |
| uv | `uv --version` |
| Eigen3 3.3+ | `apt list --installed 2>/dev/null \| grep libeigen3` |
| OpenCV | `pkg-config --modversion opencv4` |
| PCL 1.8+ | `pkg-config --modversion pcl_common` |
| GTest | `apt list --installed 2>/dev/null \| grep libgtest` |

---

## 1. Python 環境 (uv)

```bash
# ワークスペースルートで実行
uv sync
```

Python 3.11 の仮想環境 `.venv/` が作成され、以下がインストールされる。

- `pybind11` — C++ バインディングのビルド用
- `numpy` — Eigen ↔ numpy 変換
- `torch 2.12.0+cu130` / `torchvision` — GPU 推論
- `segmentation-models-pytorch` — DeepLabV3+
- `opencv-python`, `pillow`, `labelme` — 画像処理・アノテーション

> CUDA バージョンを変えたい場合は `pyproject.toml` の `[[tool.uv.index]]` の URL を変更して `uv lock && uv sync` を実行する。

---

## 2. C++ ビルド (colcon)

pybind11 拡張モジュールのビルドに Python 3.11 が必要なため、**先に venv を activate する**。

```bash
source .venv/bin/activate
colcon build
```

ビルド順序は colcon が `package.xml` の依存関係から自動解決する。

```
pure_trackdlo → preprocessing → evaluation → trackdlo
                                           → trackdlo_python
```

特定パッケージのみビルドする場合:

```bash
colcon build --packages-up-to trackdlo        # ROS2 ノードまで
colcon build --packages-select trackdlo_python # Python バインディングのみ
```

---

## 3. テスト

```bash
colcon test
colcon test-result --verbose
```

---

## 4. ROS2 ノード起動

```bash
source install/setup.bash
ros2 launch trackdlo trackdlo_node.launch.py
```

HSV 閾値を変えて起動する場合:

```bash
ros2 launch trackdlo trackdlo_node.launch.py hsv_lower:="90 90 30" hsv_upper:="130 255 255"
```

HSV 閾値の調整方法は `src/trackdlo/docs/COLOR_THRESHOLD.md` を参照。

---

## 5. Python バインディング (trackdlo_python)

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/trackdlo_python
uv run python - <<'EOF'
import trackdlo_python as tdlo
state = tdlo.make_trackdlo_state(10)
print("OK  Y.shape:", state.Y.shape)
EOF
```

---

## 6. DeepLab デモ (Web カメラ → バイナリマスク)

```bash
uv run python src/trackdlo_python/scripts/demo_mask.py
```

ESC キーで終了。重みファイルは `src/bmask_gen/weights/best_deeplabv3plus_cable.pth`。

---

## よくある問題

**`.so` が `cpython-310` のまま**
→ `rm -rf build/trackdlo_python` でキャッシュを削除してから再ビルドする。

**torch が CPU 版になっている**
→ `python -c "import torch; print(torch.version.cuda)"` で確認。`None` なら `uv sync` が CUDA インデックスを参照できていない。`pyproject.toml` の `[[tool.uv.index]]` を確認する。
