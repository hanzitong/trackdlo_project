#!/bin/bash
# =============================================================================
# setup.sh
#
# pipeline_demo を実行するための前準備スクリプト。
# 初回のみ実行すれば良い。
#
# コピーするファイル:
#   libtrackdlo_c.so               ← colcon build 後の build/ から
#   libpreprocessing_c.so          ← colcon build 後の build/ から
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
    echo "    colcon build --packages-up-to trackdlo_cdll"
    exit 1
fi

cp "$SO_SRC" "$SCRIPT_DIR/libtrackdlo_c.so"
echo "[OK] copied: libtrackdlo_c.so"

# ─── libpreprocessing_c.so ───────────────────────────────────────────────────
PREPROC_SO_SRC="$WS_ROOT/build/preprocessing_cdll/libpreprocessing_c.so"

if [ ! -f "$PREPROC_SO_SRC" ]; then
    echo "[ERROR] $PREPROC_SO_SRC が見つかりません。"
    echo "  先に以下を実行してください:"
    echo "    colcon build --packages-up-to preprocessing_cdll"
    exit 1
fi

cp "$PREPROC_SO_SRC" "$SCRIPT_DIR/libpreprocessing_c.so"
echo "[OK] copied: libpreprocessing_c.so"

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
