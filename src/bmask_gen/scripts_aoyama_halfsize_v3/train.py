# =============================================================================
# train.py  (scripts_aoyama_halfsize_v3/)
#
# DeepLabV3 (V3+ ではない) + MobileNetV2 エンコーダーで 320×240 画像を学習する。
#
# 設計判断 — V3+ から V3 に変更した理由:
#   V3+ の Decoder スキップ接続 (stride-4 のエンコーダー出力をデコーダーで使う)
#   が原因で、Hailo-8 のマルチコンテキスト分割が "context_0 に資源集中" で失敗した。
#   V3 はこのスキップ接続を持たないため、320×240 でのシングルコンテキスト
#   コンパイルを目指す。
#
# 設計判断 — 解像度 320×240 を使う理由:
#   640×480 では最初の特徴マップ (320×240×32) が Hailo-8 の SRAM (2.5 MB) を
#   超えるため、常にマルチコンテキストになる。マルチコンテキストでは
#   エンコーダー内の残差接続がコンテキスト境界をまたいで AddShortcut
#   クラッシュを引き起こす。
#   320×240 では最初の特徴マップ (160×120×32) ≈ 1.2 MB となり、
#   シングルコンテキストに収まる見込み。
#
# 入力データ: data_aoyama_halfsize/dataset/ (320×240, train 78 / val 20)
# 出力重み:   weights_halfsize_v3/
#   best_deeplabv3_mobilenetv2.pth  ← val IoU が最良のエポック
#   last_deeplabv3_mobilenetv2.pth  ← 最終エポック
#
# 使い方 (ワークスペースルートから):
#   uv run python src/bmask_gen/scripts_aoyama_halfsize_v3/train.py
# =============================================================================

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

ROOT: Path = Path(__file__).resolve().parent.parent  # src/bmask_gen/

DATASET_DIR: Path  = ROOT / "data_aoyama_halfsize/dataset"
WEIGHTS_DIR: Path  = ROOT / "weights_halfsize_v3"
BEST_PATH: Path    = WEIGHTS_DIR / "best_deeplabv3_mobilenetv2.pth"
LAST_PATH: Path    = WEIGHTS_DIR / "last_deeplabv3_mobilenetv2.pth"

IMG_H: int = 240
IMG_W: int = 320

NUM_EPOCHS: int = 100


class CableDataset(Dataset):
    """data_aoyama_halfsize 用 Dataset。320×240 JPEG 画像と binary PNG マスクのペア。"""

    def __init__(self, image_dir: Path, mask_dir: Path) -> None:
        """Dataset を初期化する。"""
        self.image_dir: Path = image_dir
        self.mask_dir: Path  = mask_dir
        self.image_paths: list[Path] = sorted(
            [p for p in image_dir.iterdir() if p.suffix in [".jpg", ".jpeg"]]
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"画像が見つかりません: {image_dir}")

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
            raise FileNotFoundError(f"対応マスクがありません: {mask_path}")

        image: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"画像を読めませんでした: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask: np.ndarray = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)

        if image.shape[:2] != (IMG_H, IMG_W):
            raise ValueError(f"画像サイズ異常: {img_path} → {image.shape}")
        if mask.shape[:2] != (IMG_H, IMG_W):
            raise ValueError(f"マスクサイズ異常: {mask_path} → {mask.shape}")

        # /255.0 のみ (ImageNet mean/std 正規化なし)
        # 推論スクリプトも同じ正規化を使うこと
        image = image.astype(np.float32) / 255.0

        image = np.transpose(image, (2, 0, 1))  # HWC → CHW: (3, 240, 320)
        mask  = np.expand_dims(mask, axis=0)    # (H,W) → (1,H,W)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def calc_iou(logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> float:
    """logit と正解マスクから IoU を計算して返す。"""
    preds: torch.Tensor = (torch.sigmoid(logits) > threshold).float()
    eps: float = 1e-7
    intersection: torch.Tensor = (preds * masks).sum(dim=(1, 2, 3))
    union: torch.Tensor = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


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
        total_iou  += calc_iou(logits.detach(), masks)
    return total_loss / len(loader), total_iou / len(loader)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """1 エポック分の検証を行い、(平均損失, 平均 IoU) を返す。"""
    model.eval()
    total_loss: float = 0.0
    total_iou: float  = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item()
        total_iou  += calc_iou(logits, masks)
    return total_loss / len(loader), total_iou / len(loader)


def main() -> None:
    """学習全体のエントリポイント。"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset: CableDataset = CableDataset(
        image_dir=DATASET_DIR / "images/train",
        mask_dir =DATASET_DIR / "masks/train",
    )
    val_dataset: CableDataset = CableDataset(
        image_dir=DATASET_DIR / "images/val",
        mask_dir =DATASET_DIR / "masks/val",
    )
    print(f"train: {len(train_dataset)} サンプル / val: {len(val_dataset)} サンプル")

    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader: DataLoader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True,
    )

    # ─── モデル ───────────────────────────────────────────────────────────
    # DeepLabV3 (V3+ ではない): ASPP のみ、Decoder スキップ接続なし。
    # Hailo-8 シングルコンテキストを目指すため、スキップ接続のない V3 を採用。
    model: smp.DeepLabV3 = smp.DeepLabV3(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,  # raw logit 出力。推論時は sigmoid を適用する。
    ).to(device)

    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_iou: float = -1.0

    for epoch in range(NUM_EPOCHS):
        train_loss: float
        train_iou: float
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss: float
        val_iou: float
        val_loss, val_iou = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:03d}/{NUM_EPOCHS:03d} "
            f"| train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
            f"| val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  → best 更新: val_iou={best_val_iou:.4f}")

    torch.save(model.state_dict(), LAST_PATH)
    print(f"学習完了。best_val_iou={best_val_iou:.4f}")
    print(f"best: {BEST_PATH}")
    print(f"last: {LAST_PATH}")


if __name__ == "__main__":
    main()
