# =============================================================================
# train.py  (scripts_aoyama/)
#
# data_aoyama/ を使って DeepLabV3+ のバイナリセグメンテーションを学習する。
# scripts/train_deeplabv3plus_binary.py と同じアーキテクチャだが、
# 以下の点が異なる:
#   - 画像形式: .jpg (scripts/ は .png)
#   - データパス: data_aoyama/raw/dataset/
#   - 重み保存先: weights_aoyama/
#
# 事前準備: prepare_dataset.py を先に実行してデータセットを生成すること。
#
# 使い方 (ワークスペースルートから):
#   uv run python src/bmask_gen/scripts_aoyama/train.py
#
# ─── モデル・損失関数の概要 ────────────────────────────────────────────────
#
# DeepLabV3+ (encoder=ResNet34, classes=1, activation=None)
#   → raw logit (H×W×1) を出力
#   → BCEWithLogitsLoss: logit のまま受け取り数値安定な BCE を計算
#   → 推論時のみ sigmoid を通して確率に変換
#
# =============================================================================

from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

ROOT: Path = Path(__file__).resolve().parent.parent


# =============================================================================
# CableDataset
#
# data_aoyama/raw/dataset/ 以下の JPEG 画像と PNG マスクのペアを管理する。
# scripts/ 版と異なり、画像拡張子が .jpg である点だけが異なる。
# =============================================================================
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

        image_tensor: (3, 480, 640), float32, [0, 1]
        mask_tensor:  (1, 480, 640), float32, {0.0, 1.0}
        """
        img_path: Path  = self.image_paths[idx]
        mask_path: Path = self.mask_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"対応するマスクがありません: {mask_path}")

        # [1] BGR → RGB 変換
        # cv2.imread は BGR 順 (Blue-Green-Red) で読み込む。
        # ResNet34 エンコーダは ImageNet (RGB 順) で事前学習されているため
        # RGB に変換する必要がある。
        # ※ infer_live_cpu.py でも同じ変換を行う。train/infer の一貫性を保つこと。
        image: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"画像を読めませんでした: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # マスクは 0/1 の binary PNG (prepare_dataset.py が生成)。
        # PIL で読み込み、念のため > 0 で再度 binary 化する。
        mask: np.ndarray = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)   # 0.0 (背景) / 1.0 (ケーブル)

        if image.shape[:2] != (480, 640):
            raise ValueError(f"画像サイズが 640x480 ではありません: {img_path}")
        if mask.shape[:2] != (480, 640):
            raise ValueError(f"マスクサイズが 640x480 ではありません: {mask_path}")

        # [2] 正規化: uint8 [0, 255] → float32 [0.0, 1.0]
        # ニューラルネットワークへの入力を小さな値に揃えることで学習が安定する。
        #
        # ※ smp.DeepLabV3Plus(encoder_weights="imagenet") を使う場合、
        #    本来は追加で ImageNet 統計による正規化が推奨される:
        #      preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet34", "imagenet")
        #      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #    ここでは /255.0 のみを適用する。100 epoch の fine-tuning により
        #    BatchNorm 統計がこの入力スケールに適応するため実用上は機能する。
        #    重要: infer_live_cpu.py も /255.0 のみにしてあり、両者が一致している。
        #    将来 ImageNet 正規化を追加する場合は train.py と infer 両方に同時に行うこと。
        image = image.astype(np.float32) / 255.0

        # [3] 軸変換: HWC → CHW
        # numpy / OpenCV 形式: (H, W, C) = (480, 640, 3)
        # PyTorch 形式:        (C, H, W) = (3, 480, 640)
        # np.transpose(image, (2, 0, 1)) で軸 (H,W,C) を (C,H,W) に入れ替える:
        #   元: axis0=H=480, axis1=W=640, axis2=C=3
        #   後: axis0=C=3,   axis1=H=480, axis2=W=640
        # DataLoader が後でバッチ次元 N を自動で追加するため、ここでは不要。
        image = np.transpose(image, (2, 0, 1))

        # [4] マスクへのチャンネル次元追加: (H, W) → (1, H, W)
        # モデル出力は (N, 1, H, W) の形状 (classes=1) なので、
        # 損失関数に渡す正解マスクも同じ (N, 1, H, W) に揃える必要がある。
        # DataLoader が N を追加するため、ここでは (1, H, W) にするだけでよい。
        mask = np.expand_dims(mask, axis=0)

        # torch.tensor() は numpy の dtype を引き継ぐ。
        # image は float32 → dtype 明示で安全にテンソル化する。
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# =============================================================================
# 学習・検証ユーティリティ
# =============================================================================

def calc_iou_from_logits(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """モデル出力 (logit) と正解マスクから IoU を計算して返す。

    Args:
        logits:    モデルの生出力テンソル (B, 1, H, W)
        masks:     正解マスクテンソル (B, 1, H, W)、値は 0.0 か 1.0
        threshold: 予測を 1 とみなす確率の閾値
        eps:       ゼロ除算防止の微小値

    Returns:
        バッチ内の平均 IoU (0.0 ～ 1.0)
    """
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


# =============================================================================
# main
# =============================================================================

def main() -> None:
    """学習全体の設定と実行を行うエントリポイント。"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    dataset_dir: Path = ROOT / "data_aoyama/raw/dataset"
    weights_dir: Path = ROOT / "weights_aoyama"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # ─── DataLoader ───────────────────────────────────────────────────────
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

    # ─── モデル ───────────────────────────────────────────────────────────
    # encoder_weights="imagenet": ImageNet 事前学習済み重みで転移学習
    model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ─── 学習ループ ────────────────────────────────────────────────────────
    best_val_iou: float = -1.0
    # num_epochs: int     = 100
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
            torch.save(model.state_dict(), weights_dir / "best_deeplabv3plus_cable.pth")
            print(f"  → best model 更新: val_iou={best_val_iou:.4f}")

    torch.save(model.state_dict(), weights_dir / "last_deeplabv3plus_cable.pth")
    print("saved last model")


if __name__ == "__main__":
    main()
