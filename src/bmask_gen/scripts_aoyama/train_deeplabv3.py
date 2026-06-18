# =============================================================================
# train_deeplabv3.py  (scripts_aoyama/)
#
# DeepLabV3 + MobileNetV2 エンコーダーで学習する。
# train.py (DeepLabV3+ ResNet34) からの変更点:
#   - smp.DeepLabV3Plus → smp.DeepLabV3
#   - encoder_name="resnet34" → encoder_name="mobilenet_v2"
#   - 重み保存先ファイル名を _deeplabv3_mobilenet_v2_ に変更
#
# Hailo-8 向けの背景:
#   DeepLabV3+ ResNet34 は Hailo DFC 3.33.0 でコンパイルできない
#   (デコーダーのスキップ接続がマルチコンテキスト分割で非互換になる)。
#   DeepLabV3 MobileNetV2 は Hailo モデルズーで対応実績がある。
#
# 使い方 (ワークスペースルートから):
#   uv run python src/bmask_gen/scripts_aoyama/train_deeplabv3.py
# =============================================================================

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

ROOT: Path = Path(__file__).resolve().parent.parent


class CableDataset(Dataset):
    """data_aoyama 用 Dataset。JPEG 画像と binary PNG マスクのペアを管理する。"""

    def __init__(self, image_dir: Path | str, mask_dir: Path | str) -> None:
        """Dataset を初期化する。

        Args:
            image_dir: JPEG 画像が格納されたディレクトリ
            mask_dir:  binary PNG マスクが格納されたディレクトリ
        """
        self.image_dir: Path = Path(image_dir)
        self.mask_dir: Path  = Path(mask_dir)
        self.image_paths: list[Path] = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix in [".jpg", ".jpeg"]]
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"画像が見つかりません: {self.image_dir}")

    def __len__(self) -> int:
        """データセットのサンプル数を返す。"""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """idx 番目の (image_tensor, mask_tensor) ペアを返す。

        image_tensor: (3, 240, 320), float32, [0, 1]
        mask_tensor:  (1, 240, 320), float32, {0.0, 1.0}
        """
        img_path: Path  = self.image_paths[idx]
        mask_path: Path = self.mask_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"対応するマスクがありません: {mask_path}")

        # [1] BGR → RGB: cv2.imread は BGR、ResNet/MobileNet は RGB で事前学習済み
        image: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"画像を読めませんでした: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask: np.ndarray = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)

        if image.shape[:2] != (240, 320):
            raise ValueError(f"画像サイズが 320x240 ではありません: {img_path}")
        if mask.shape[:2] != (240, 320):
            raise ValueError(f"マスクサイズが 320x240 ではありません: {mask_path}")

        # [2] 正規化: /255.0 のみ (ImageNet mean/std は使わない)
        #     infer_live_cpu.py / hailo/infer_live.py も同じ正規化を使うこと
        image = image.astype(np.float32) / 255.0

        # [3] HWC → CHW: (240,320,3) → (3,240,320)
        image = np.transpose(image, (2, 0, 1))

        # [4] マスク: (H,W) → (1,H,W)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def calc_iou_from_logits(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """モデル出力 (logit) と正解マスクから IoU を計算して返す。"""
    probs = torch.sigmoid(logits)
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
    """1 エポック分の学習を行い、(平均損失, 平均 IoU) を返す。"""
    model.train()
    total_loss: float = 0.0
    total_iou: float  = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)
        loss   = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou  += calc_iou_from_logits(logits.detach(), masks)
    return total_loss / len(loader), total_iou / len(loader)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """1 エポック分の検証を行い、(平均損失, 平均 IoU) を返す。重みは更新しない。"""
    model.eval()
    total_loss: float = 0.0
    total_iou: float  = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item()
        total_iou  += calc_iou_from_logits(logits, masks)
    return total_loss / len(loader), total_iou / len(loader)


def main() -> None:
    """学習全体の設定と実行を行うエントリポイント。"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    dataset_dir: Path = ROOT / "data_aoyama/dataset_320x240"
    weights_dir: Path = ROOT / "weights_aoyama"
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_dataset: CableDataset = CableDataset(
        image_dir=dataset_dir / "images/train",
        mask_dir =dataset_dir / "masks/train",
    )
    val_dataset: CableDataset = CableDataset(
        image_dir=dataset_dir / "images/val",
        mask_dir =dataset_dir / "masks/val",
    )
    print(f"train: {len(train_dataset)} サンプル / val: {len(val_dataset)} サンプル")

    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader: DataLoader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True,
    )

    # ─── モデル ───────────────────────────────────────────────────────────
    # DeepLabV3 (V3+ではない): デコーダーにスキップ接続がないシンプルな構造。
    # Hailo DFC 3.33.0 でコンパイル可能なことが確認されている (モデルズー対応)。
    # encoder: mobilenet_v2 — 軽量で Hailo-8 の単一コンテキストに収まりやすい。
    model: smp.DeepLabV3 = smp.DeepLabV3(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_iou: float = -1.0
    num_epochs: int     = 500

    for epoch in range(num_epochs):
        train_loss: float
        train_iou: float
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss: float
        val_iou: float
        val_loss, val_iou = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:03d}/{num_epochs:03d} "
            f"| train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
            f"| val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(
                model.state_dict(),
                weights_dir / "best_deeplabv3_mobilenet_v2_cable.pth",
            )
            print(f"  → best model 更新: val_iou={best_val_iou:.4f}")

    torch.save(
        model.state_dict(),
        weights_dir / "last_deeplabv3_mobilenet_v2_cable.pth",
    )
    print("saved last model")


if __name__ == "__main__":
    main()
