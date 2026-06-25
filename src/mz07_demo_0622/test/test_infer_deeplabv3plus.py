"""
test_infer_deeplabv3plus.py

sample_data/ 内の BGR 画像 1 枚に対して DeepLabV3+ (CPU) で推論し、
バイナリマスクを cv2.imshow で表示する。

display windows:
  bgr    : 元画像
  mask   : 推論結果 (ケーブル = 白 255 / 背景 = 黒 0)
  overlay: 元画像にマスクを半透明で重ねたもの

usage:
  cd src/mz07_demo && python test/test_infer_deeplabv3plus.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


def main() -> None:
    """DeepLabV3+ の推論結果を表示するエントリポイント。"""
    here: Path       = Path(__file__).resolve().parent
    weights: Path    = here.parent / "best_deeplabv3plus_cable.pth"
    image_path: Path = here.parent / "sample_data" / "bgr_0000.png"

    # ImageNet 正規化パラメータ (train_deeplabv3plus.py と必ず同じ値にすること)
    imagenet_mean: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std: np.ndarray  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to("cpu")
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    print(f"model loaded: {weights.name}")

    bgr: np.ndarray = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"image not found: {image_path}")
    print(f"image loaded: {image_path.name}  shape={bgr.shape}")

    # 前処理: BGR to RGB, [0,255] to [0,1], ImageNet 正規化, (H,W,C) to (1,C,H,W)
    rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - imagenet_mean) / imagenet_std
    x_t: torch.Tensor = torch.tensor(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)

    with torch.no_grad():
        prob: np.ndarray = torch.sigmoid(model(x_t))[0, 0].numpy()

    mask: np.ndarray = (prob > 0.5).astype(np.uint8)
    print(f"cable pixels: {int(mask.sum())}")

    mask_vis: np.ndarray    = mask * 255
    overlay: np.ndarray     = bgr.copy()
    overlay[mask == 1]      = (255, 255, 255)
    overlay_vis: np.ndarray = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)

    cv2.imshow("bgr", bgr)
    cv2.imshow("mask", mask_vis)
    cv2.imshow("overlay", overlay_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
