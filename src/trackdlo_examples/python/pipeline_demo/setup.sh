#!/bin/bash
# =============================================================================
# setup.sh
#
# pipeline_demo を実行するための前準備スクリプト。
# 初回のみ実行すれば良い。
#
# コピーするファイル:
#   libtrackdlo_c.so               ← colcon build 後の build/ から
#   libpreprocessing_c.so          ← build_standalone/ から (PCL/OpenCV 静的リンク済み)
#   trackdlo_cdll.py               ← src/trackdlo_cdll/python/ から
#   preprocessing_cdll.py          ← src/preprocessing_cdll/python/ から
#   best_deeplabv3plus_cable.pth   ← src/bmask_gen/weights/ から
#
# 使い方:
#   cd src/trackdlo_examples/python/pipeline_demo
#   bash setup.sh
# =============================================================================

set -e  # エラーが起きたら即終了

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# このスクリプトの場所: trackdlo_project/src/trackdlo_examples/python/pipeline_demo/
# ワークスペースルート:   trackdlo_project/ (4 階層上)
WS_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "workspace root : $WS_ROOT"
echo "install to     : $SCRIPT_DIR"
echo ""

# ─── libtrackdlo_c.so ────────────────────────────────────────────────────────
SO_SRC="$WS_ROOT/build/trackdlo_cdll/libtrackdlo_c.so"

if [ ! -f "$SO_SRC" ]; then
    echo "[ERROR] $SO_SRC が見つかりません。"
    echo "  先に以下を実行してください:"
    echo "    colcon build --packages-select trackdlo_cdll"
    exit 1
fi

cp "$SO_SRC" "$SCRIPT_DIR/libtrackdlo_c.so"
echo "[OK] copied: libtrackdlo_c.so"

# ─── libpreprocessing_c.so (スタンドアロンビルド) ────────────────────────────
# build_standalone/ は PCL/OpenCV を静的リンクした持ち運び可能版。
# 通常の build/ 版は PCL/OpenCV の .so に動的依存するため持ち運べない。
PREPROC_SO_SRC="$WS_ROOT/build_standalone/preprocessing_cdll/libpreprocessing_c.so"

if [ ! -f "$PREPROC_SO_SRC" ]; then
    echo "[ERROR] $PREPROC_SO_SRC が見つかりません。"
    echo "  先に以下を実行してください:"
    echo "    mkdir -p build_standalone/preprocessing && cd build_standalone/preprocessing"
    echo "    cmake ../../src/preprocessing -DBUILD_STANDALONE=ON -DCMAKE_INSTALL_PREFIX=../../install_standalone"
    echo "    make && make install && cd ../.."
    echo "    mkdir -p build_standalone/preprocessing_cdll && cd build_standalone/preprocessing_cdll"
    echo "    cmake ../../src/preprocessing_cdll -DBUILD_STANDALONE=ON -DCMAKE_PREFIX_PATH=../../install_standalone"
    echo "    make && cd ../.."
    exit 1
fi

cp "$PREPROC_SO_SRC" "$SCRIPT_DIR/libpreprocessing_c.so"
echo "[OK] copied: libpreprocessing_c.so  (standalone, PCL/OpenCV statically linked)"

# ─── Python ラッパー ──────────────────────────────────────────────────────────
# pipeline_demo.py は同ディレクトリにある .py を優先して読み込む。
# これにより持ち運び先の PC にワークスペースがなくても動作する。

TRACKDLO_PY_SRC="$WS_ROOT/src/trackdlo_cdll/python/trackdlo_cdll.py"
if [ ! -f "$TRACKDLO_PY_SRC" ]; then
    echo "[ERROR] $TRACKDLO_PY_SRC が見つかりません。"
    exit 1
fi
cp "$TRACKDLO_PY_SRC" "$SCRIPT_DIR/trackdlo_cdll.py"
echo "[OK] copied: trackdlo_cdll.py"

PREPROC_PY_SRC="$WS_ROOT/src/preprocessing_cdll/python/preprocessing_cdll.py"
if [ ! -f "$PREPROC_PY_SRC" ]; then
    echo "[ERROR] $PREPROC_PY_SRC が見つかりません。"
    exit 1
fi
cp "$PREPROC_PY_SRC" "$SCRIPT_DIR/preprocessing_cdll.py"
echo "[OK] copied: preprocessing_cdll.py"

# ─── DeepLabV3+ 重みファイル ──────────────────────────────────────────────────
WEIGHTS_SRC="$WS_ROOT/src/bmask_gen/weights/best_deeplabv3plus_cable.pth"

if [ ! -f "$WEIGHTS_SRC" ]; then
    echo "[ERROR] $WEIGHTS_SRC が見つかりません。"
    echo "  先に DeepLabV3+ の学習を完了させてください:"
    echo "    python3 src/bmask_gen/scripts/train_deeplabv3plus_binary.py"
    exit 1
fi

cp "$WEIGHTS_SRC" "$SCRIPT_DIR/best_deeplabv3plus_cable.pth"
echo "[OK] copied: best_deeplabv3plus_cable.pth"

echo ""
echo "setup 完了。以下で実行できます:"
echo "  python3 $SCRIPT_DIR/pipeline_demo.py"
