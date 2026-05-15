


from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent


class CableDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"画像がありません: {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"対応するマスクがありません: {mask_path}")

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"画像を読めませんでした: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        # 念のため 0/1 に正規化
        mask = (mask > 0).astype(np.float32)

        # 念のためサイズ確認
        if image.shape[0] != 480 or image.shape[1] != 640:
            raise ValueError(f"画像サイズが 640x480 ではありません: {img_path} shape={image.shape}")
        if mask.shape[0] != 480 or mask.shape[1] != 640:
            raise ValueError(f"マスクサイズが 640x480 ではありません: {mask_path} shape={mask.shape}")

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))   # HWC to CHW
        mask = np.expand_dims(mask, axis=0)       # HxW to 1xHxW

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def calc_iou_from_logits(logits, masks, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += calc_iou_from_logits(logits.detach(), masks)

    return total_loss / len(loader), total_iou / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_iou += calc_iou_from_logits(logits, masks)

    return total_loss / len(loader), total_iou / len(loader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    train_dataset = CableDataset(
        image_dir=ROOT / "data/dataset/images/train",   # ROOT is defined by Pathlib
        mask_dir=ROOT / "data/dataset/masks/train",
    )
    val_dataset = CableDataset(
        image_dir=ROOT / "data/dataset/images/val",
        mask_dir=ROOT / "data/dataset/masks/val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_iou = -1.0
    num_epochs = 100

    for epoch in range(num_epochs):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1:03d}/{num_epochs:03d} "
            f"| train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
            f"| val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), ROOT / "weights/best_deeplabv3plus_cable.pth")
            print(f"saved best model: val_iou={best_val_iou:.4f}")

    torch.save(model.state_dict(), ROOT / "weights/last_deeplabv3plus_cable.pth")
    print("saved last model")


if __name__ == "__main__":
    main()





