# =============================================================================
# infer_single.py  (scripts_aoyama/)
#
# val セットの単一画像に対して推論を実行し、結果をウィンドウで表示する。
# infer_live_cpu.py と同じ表示方式 (元画像 + 緑マスク overlay の 2 窓)。
#
# 使い方 (ワークスペースルートから):
#   # val 先頭の画像を使う (デフォルト)
#   uv run python src/bmask_gen/scripts_aoyama/infer_single.py
#
#   # 画像ファイル名を指定する
#   uv run python src/bmask_gen/scripts_aoyama/infer_single.py --image 12.jpg
#
# 何かキーを押すと終了。
# =============================================================================

import argparse
from pathlib import Path

import cv2
import numpy as np

import torch
import segmentation_models_pytorch as smp

ROOT: Path = Path(__file__).resolve().parent.parent

WEIGHTS_PATH: Path = ROOT / "weights_aoyama/best_deeplabv3plus_cable.pth"
VAL_IMG_DIR: Path  = ROOT / "data_aoyama/raw/dataset/images/val"

THRESHOLD: float = 0.5


def main() -> None:
    """単一 val 画像の推論と表示を行うエントリポイント。"""
    # ─── 引数パース ───────────────────────────────────────────────────────
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="画像ファイル名 (例: 12.jpg)。省略時は val 先頭の画像を使う。",
    )
    args: argparse.Namespace = parser.parse_args()

    # ─── 画像パスの決定 ───────────────────────────────────────────────────
    if args.image is not None:
        img_path: Path = VAL_IMG_DIR / args.image
        if not img_path.exists():
            raise FileNotFoundError(f"指定した画像が見つかりません: {img_path}")
    else:
        candidates: list[Path] = sorted(
            [p for p in VAL_IMG_DIR.iterdir() if p.suffix in [".jpg", ".jpeg"]]
        )
        if not candidates:
            raise RuntimeError(
                f"val 画像が見つかりません: {VAL_IMG_DIR}\n"
                "prepare_dataset.py を先に実行してください。"
            )
        img_path = candidates[0]

    print(f"画像: {img_path}")

    # ─── モデルロード ─────────────────────────────────────────────────────
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"重みが見つかりません: {WEIGHTS_PATH}\n"
            "train.py を先に実行してください。"
        )

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print("model loaded")

    # ─── 画像読み込み ─────────────────────────────────────────────────────
    frame: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"画像を読めませんでした: {img_path}")
    print(f"画像サイズ: {frame.shape}")  # (H, W, C) を確認用に表示

    # ─── 前処理 (infer_live_cpu.py と完全に同じ手順) ─────────────────────
    # [1] BGR → RGB: cv2.imread は BGR、モデルは RGB で学習済みのため変換
    rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # [2] 正規化: uint8 [0,255] → float32 [0,1]
    #     train.py も /255.0 のみで学習しているため、ここも同じにする
    x: np.ndarray = rgb.astype(np.float32) / 255.0

    # [3] 軸変換 HWC → CHW: (H,W,3) → (3,H,W)
    x = np.transpose(x, (2, 0, 1))

    # [4] バッチ次元追加 → (1,3,H,W)
    x_t: torch.Tensor = torch.tensor(x).unsqueeze(0).to(device)

    # ─── 推論 ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        # 出力: (1,1,H,W) raw logit → sigmoid → 確率 [0,1]
        # [0,0] でバッチ・チャンネル次元を除去して (H,W) を取得
        prob: np.ndarray = torch.sigmoid(model(x_t))[0, 0].cpu().numpy()

    mask: np.ndarray = (prob > THRESHOLD).astype(np.uint8)
    print(f"mask pixels (cable): {mask.sum()}")

    # ─── overlay 合成 (infer_live_cpu.py と同じ) ─────────────────────────
    # frame は BGR のまま表示に使う (OpenCV は BGR を期待する)
    # (0, 200, 0) は BGR 順の緑色
    overlay: np.ndarray = frame.copy()
    overlay[mask == 1] = (0, 200, 0)
    result: np.ndarray = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    # ─── 表示 ─────────────────────────────────────────────────────────────
    cv2.imshow("camera", frame)
    cv2.imshow("mask (green = cable)", result)
    print("任意のキーを押すと終了します")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
