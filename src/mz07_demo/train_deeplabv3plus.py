"""
train_deeplabv3plus.py

sample_data/ を使って DeepLabV3+ (ResNet34) を学習する。
マスクの目視確認は see_mask.py で行うこと。

─── データ構成 ──────────────────────────────────────────────────────────────
  sample_data/
    bgr_000N.png        : BGR 入力画像 (640×480, uint8)  32 枚
    masks/bgr_000N.png  : バイナリマスク (640×480, uint8, 0/1) 32 枚

─── 実行フロー ──────────────────────────────────────────────────────────────
  1. 80/20 で train/val 分割し学習
  2. val IoU 最良モデルを weights/best_deeplabv3plus_cable.pth に保存

─── 実行 ─────────────────────────────────────────────────────────────────────
  cd src/mz07_demo && python train_deeplabv3plus.py
"""

from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


ROOT: Path        = Path(__file__).resolve().parent
SAMPLE_DIR: Path  = ROOT / "sample_data"
MASK_DIR: Path    = SAMPLE_DIR / "masks"
WEIGHTS_DIR: Path = ROOT / "weights"

BEST_PATH: Path = WEIGHTS_DIR / "best_deeplabv3plus_cable.pth"
LAST_PATH: Path = WEIGHTS_DIR / "last_deeplabv3plus_cable.pth"

NUM_EPOCHS: int  = 50    # データが少ないので短めに設定
BATCH_SIZE: int  = 4
LR: float        = 1e-4
VAL_RATIO: float = 0.2
RANDOM_SEED: int = 42


# =============================================================================
# CableDataset
# =============================================================================

class CableDataset(Dataset):
    """(BGR 画像, バイナリマスク) ペアを管理する Dataset。"""

    def __init__(self, pairs: list[tuple[Path, Path]]) -> None:
        """Dataset を初期化する。

        Args:
            pairs: (image_path, mask_path) のリスト
        """
        self.pairs: list[tuple[Path, Path]] = pairs

    def __len__(self) -> int:
        """サンプル数を返す。"""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """idx 番目の (image_tensor, mask_tensor) を返す。

        image_tensor: (3, 480, 640), float32, [0, 1]
        mask_tensor:  (1, 480, 640), float32, {0.0, 1.0}
        """
        img_path: Path
        mask_path: Path
        img_path, mask_path = self.pairs[idx]

        image: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {img_path}")
        # BGR to RGB: ResNet34 エンコーダは ImageNet (RGB) で事前学習済み
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask: np.ndarray = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)    # 念のため再 binary 化

        # uint8 [0,255] to float32 [0,1]
        image = image.astype(np.float32) / 255.0

        # HWC to CHW: (480,640,3) to (3,480,640)
        image = np.transpose(image, (2, 0, 1))

        # HW to 1HW: モデル出力 (N,1,H,W) と形状を合わせる
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# =============================================================================
# 学習・検証ユーティリティ
# =============================================================================

def calc_iou(logits: torch.Tensor, masks: torch.Tensor, eps: float = 1e-7) -> float:
    """logit と正解マスクから IoU を計算して返す。

    Args:
        logits: (B, 1, H, W) モデル生出力
        masks:  (B, 1, H, W) 正解マスク (0.0 / 1.0)
        eps:    ゼロ除算防止

    Returns:
        バッチ内の平均 IoU (0.0 〜 1.0)
    """
    preds: torch.Tensor = (torch.sigmoid(logits) > 0.5).float()
    intersection: torch.Tensor = (preds * masks).sum(dim=(1, 2, 3))
    union: torch.Tensor = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    iou: torch.Tensor = (intersection + eps) / (union + eps)
    return iou.mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """1 エポック分の学習を行い (平均損失, 平均 IoU) を返す。"""
    model.train()
    total_loss: float = 0.0
    total_iou: float  = 0.0

    images: torch.Tensor
    masks: torch.Tensor
    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        logits: torch.Tensor = model(images)
        loss: torch.Tensor   = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou  += calc_iou(logits.detach(), masks)

    return total_loss / len(loader), total_iou / len(loader)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """1 エポック分の検証を行い (平均損失, 平均 IoU) を返す。重みは更新しない。"""
    model.eval()
    total_loss: float = 0.0
    total_iou: float  = 0.0

    images: torch.Tensor
    masks: torch.Tensor
    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        logits: torch.Tensor = model(images)
        loss: torch.Tensor   = criterion(logits, masks)

        total_loss += loss.item()
        total_iou  += calc_iou(logits, masks)

    return total_loss / len(loader), total_iou / len(loader)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    """学習全体の設定・実行を行うエントリポイント。"""
    # ─── ペアの収集 ──────────────────────────────────────────────────────────
    all_pairs: list[tuple[Path, Path]] = []
    img_path: Path
    for img_path in sorted(SAMPLE_DIR.glob("bgr_*.png")):
        mask_path: Path = MASK_DIR / img_path.name
        if mask_path.exists():
            all_pairs.append((img_path, mask_path))

    if len(all_pairs) == 0:
        print("[ERROR] no sample pairs found")
        return

    print(f"samples: {len(all_pairs)}")

    # ─── train / val 分割 ─────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    shuffled: list[tuple[Path, Path]] = all_pairs.copy()
    random.shuffle(shuffled)

    n_val: int   = max(1, int(len(shuffled) * VAL_RATIO))
    n_train: int = len(shuffled) - n_val
    train_pairs: list[tuple[Path, Path]] = shuffled[:n_train]
    val_pairs: list[tuple[Path, Path]]   = shuffled[n_train:]
    print(f"\ntrain: {len(train_pairs)} samples / val: {len(val_pairs)} samples")

    # ─── DataLoader ───────────────────────────────────────────────────────
    train_loader: DataLoader = DataLoader(
        CableDataset(train_pairs),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    val_loader: DataLoader = DataLoader(
        CableDataset(val_pairs),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    # ─── モデル ───────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}\n")

    model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    criterion: nn.BCEWithLogitsLoss   = nn.BCEWithLogitsLoss()
    optimizer: torch.optim.Adam       = torch.optim.Adam(model.parameters(), lr=LR)

    # ─── 学習ループ ────────────────────────────────────────────────────────
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_iou: float = -1.0

    epoch: int
    for epoch in range(NUM_EPOCHS):
        train_loss: float
        train_iou: float
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss: float
        val_iou: float
        val_loss, val_iou = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:03d}/{NUM_EPOCHS:03d} "
            f"| train loss={train_loss:.4f} iou={train_iou:.4f} "
            f"| val loss={val_loss:.4f} iou={val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  [BEST] val_iou={best_val_iou:.4f}  saved: {BEST_PATH.name}")

    torch.save(model.state_dict(), LAST_PATH)
    print(f"\ntraining done. last model saved: {LAST_PATH}")


if __name__ == "__main__":
    main()
