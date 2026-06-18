

from pathlib import Path
import random
# from typing import Any

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class PracticeDataset(Dataset):

    def __init__(self, image_dir: Path | str, mask_dir: Path | str) -> None: 
        self.image_dir: Path = Path(image_dir)
        self.mask_dir: Path = Path(mask_dir)
        self.image_paths: list[Path] = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix in [".png"]]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"no image: {self.image_dir}")


    def __len__(self) -> int:
        return len(self.image_paths)

        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path: Path = self.image_paths[idx]
        mask_path: Path = self.mask_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"no mask matting the image: {mask_path}")
        
        image: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read image from opencv: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask: np.ndarray = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)

        if image.shape[0] != 480 or image.shape[1] != 640:
            raise ValueError(f"image size is not 640x480: {img_path}")
        if mask.shape[0] != 480 or mask.shape[1] != 640:
            raise ValueError(f"mask size is not 640x480: {mask_path}")


        # convert to tensor type
        image = image.astype(np.float32) / 255.0    # [0, 255] => [0.0, 1.0]
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) => (C, H, W)

        mask = np.expand_dims(mask, axis=0) # (H, W) => (1, H, W)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)



def calc_iou_from_logits(
    logit: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """ モデル出力(logit)と正解マスクからIoUを計算して返す。
    Args:
        logits: (B, 1, H, W)
        masks:  (B, 1, H, W)
        threshold:  ケーブルと判定されるしきい値。sigmoid出力にかかる。
        eps:    ゼロ割防止の微小値

    Returns:
        バッチ内の平均 IoU (0.0 ~ 1.0)

    IoU: 
    """

    # sigmoid: the function for ,,logit ==> probability [0, 1]
    probs: torch.Tensor = torch.sigmoid(logit)
    preds = (probs > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)

    return iou.mean().item()



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """ 1エポック分の学習を行い、平均損失と平均IoUを返す。
    Args:
        model:  学習対象のモデル
        loader: 訓練データのDataLoader
        criterion: 損失関数(BCEWithLogitsLoss　など)
        optimizer: オプティマイザ(Adam など)
        device: 実行デバイス("cpu" または "cuda")

    Returns:
        (平均損失、平均IoU) のタプル
    """

    model.train()
    total_loss: float = 0.0
    total_iou: float = 0.0

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


def main() -> None:
    """ DeepLabV3+ の全体を実行するエントリポイント """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    train_dataset: PracticeDataset = PracticeDataset(
        image_dir = REPO_ROOT / "data/dataset/images/train",
        mask_dir = REPO_ROOT / "data/dataset/masks/train",
    )
    val_dataset: PracticeDataset = PracticeDataset(
        image_dir = REPO_ROOT / "data/dataset/images/val",
        mask_dir = REPO_ROOT / "data/dataset/masks/val",
    )

    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
        encoder_name = "resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)


    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=1e-4)


    # learning phase
    best_val_iou: float = -1.0
    num_epochs: int = 100

    for epoch in range(num_epochs):
        train_loss: float
        train_iou: float
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss: float
        val_iou: float
        fal_loss, val_iou = validate_one_epoch(
            model, val_loader, criterion, device
        )




if __name__ == "__main__":
    main()


