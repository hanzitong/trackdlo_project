"""
infer_deeplabv3plus.py

sample_data/ 内の BGR 画像 1 枚に対して DeepLabV3+ (CPU) で推論し、
バイナリマスクを cv2.imshow で表示する。

─── 表示ウィンドウ ────────────────────────────────────────────────────────────
  bgr    : 元画像
  mask   : 推論結果 (ケーブル部分が白 = 255、背景が黒 = 0)
  overlay: 元画像にマスクを半透明で重ねたもの

─── 実行 ─────────────────────────────────────────────────────────────────────
  cd src/mz07_demo && python infer_deeplabv3plus.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


ROOT: Path = Path(__file__).resolve().parent
WEIGHTS: Path = ROOT / "best_deeplabv3plus_cable.pth"
IMAGE_PATH: Path = ROOT / "sample_data" / "bgr_0000.png"

# ImageNet 正規化パラメータ (train_deeplabv3plus.py と必ず同じ値にすること)
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── モデルのロード ────────────────────────────────────────────────────────────

model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to("cpu")

model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()
print(f"model loaded: {WEIGHTS.name}")


# ─── 推論 ─────────────────────────────────────────────────────────────────────

bgr: np.ndarray = cv2.imread(str(IMAGE_PATH))
if bgr is None:
    raise FileNotFoundError(f"image not found: {IMAGE_PATH}")
print(f"image loaded: {IMAGE_PATH.name}  shape={bgr.shape}")

# 前処理: 学習時 (train_deeplabv3plus.py) と完全に同じ順序で行う
# 1. BGR to RGB
# 2. [0,255] to [0,1]
# 3. ImageNet 正規化 (mean/std) --- 学習側と揃えないとモデルの重みが正しく機能しない
# 4. (H,W,C) to (C,H,W) to (1,C,H,W)
rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
x_t: torch.Tensor = torch.tensor(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)  # (1,3,H,W)

with torch.no_grad():
    prob: np.ndarray = torch.sigmoid(model(x_t))[0, 0].numpy()  # (H,W) [0,1]

mask: np.ndarray = (prob > 0.5).astype(np.uint8)  # 0 or 1
print(f"cable pixels: {int(mask.sum())}")


# ─── 表示 ─────────────────────────────────────────────────────────────────────

# mask は 0/1 なので 255 倍して白/黒画像にする
mask_vis: np.ndarray = mask * 255

overlay: np.ndarray = bgr.copy()
overlay[mask == 1] = (255, 255, 255)
overlay_vis: np.ndarray = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)

cv2.imshow("bgr", bgr)
cv2.imshow("mask", mask_vis)
cv2.imshow("overlay", overlay_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()
