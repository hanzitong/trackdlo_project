# =============================================================================
# prepare_dataset.py
#
# data_aoyama/raw/ のアノテーションデータを学習用データセットに変換する。
#
# 処理内容:
#   1. annotated_json/{n}.json の polygon → binary PNG マスクを生成
#   2. 対応する raw_images/{n}.jpg と合わせて 80/20 で train/val に分割
#   3. data_aoyama/dataset_320x240/ 以下に 320×240 リサイズ済みで保存
#
# 解像度を 320×240 に変更した理由:
#   640×480 では DeepLabV3 (MobileNetV2) の最初の特徴マップが Hailo-8 の
#   内部 SRAM (2.5 MB) に収まらずマルチコンテキスト分割が発生し、
#   スキップ接続の境界でコンパイラがクラッシュする。
#   320×240 は RealSense D405 のネイティブ解像度であり、
#   640×480 のちょうど 1/2 サイズのため整数比リサイズで劣化なし。
#   → docs/problem_solving/hailo8_sram_overflow_640x480.md 参照
#
# アノテーション再作成は不要:
#   LabelMe JSON の polygon 座標 (640×480 空間) を 0.5 倍にスケールして
#   320×240 のキャンバスに描画することで同じ形状が得られる。
#
# 使い方 (ワークスペースルートから):
#   uv run python src/bmask_gen/scripts_aoyama/prepare_dataset.py
#
# 出力構造:
#   data_aoyama/dataset_320x240/
#     images/train/   ← {n}.jpg  (320×240)
#     images/val/
#     masks/train/    ← {n}.png  (320×240, 0=背景, 1=ケーブル)
#     masks/val/
# =============================================================================

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT: Path = Path(__file__).resolve().parent.parent

JSON_DIR: Path    = ROOT / "data_aoyama/raw/annotated_json"
IMG_DIR: Path     = ROOT / "data_aoyama/raw/raw_images"
DATASET_DIR: Path = ROOT / "data_aoyama/dataset_320x240"

TARGET_W: int = 320
TARGET_H: int = 240

TRAIN_RATIO: float = 0.8
RANDOM_SEED: int   = 42


def json_to_mask(json_path: Path) -> Image.Image:
    """LabelMe polygon JSON を読み込み、320×240 の binary マスク画像を返す。

    polygon 座標を orig_size → TARGET_W×TARGET_H にスケールしてから描画する。
    "cable" ラベルのポリゴンをすべて 1 で塗りつぶす。背景は 0。
    """
    data: dict = json.loads(json_path.read_text())
    orig_w: int = data["imageWidth"]   # 640
    orig_h: int = data["imageHeight"]  # 480

    scale_x: float = TARGET_W / orig_w  # 320/640 = 0.5
    scale_y: float = TARGET_H / orig_h  # 240/480 = 0.5

    mask: Image.Image = Image.new("L", (TARGET_W, TARGET_H), 0)
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(mask)

    for shape in data["shapes"]:
        if shape["label"] == "cable":
            # points は [[x, y], ...] 形式 → 0.5 倍してから PIL に渡す
            pts: list[tuple[float, float]] = [
                (p[0] * scale_x, p[1] * scale_y) for p in shape["points"]
            ]
            draw.polygon(pts, fill=1)

    return mask


def collect_pairs() -> list[tuple[Path, Path]]:
    """JSON と JPG が両方存在するペアをリストアップして返す。

    {n}.json ↔ {n}.jpg で番号を照合する。
    JSON があっても対応 JPG が存在しない場合はスキップして警告を出す。
    """
    pairs: list[tuple[Path, Path]] = []

    json_paths: list[Path] = sorted(
        JSON_DIR.glob("*.json"),
        key=lambda p: int(p.stem),
    )

    for json_path in json_paths:
        img_path: Path = IMG_DIR / f"{json_path.stem}.jpg"
        if not img_path.exists():
            print(f"[WARN] 対応画像が見つかりません: {img_path.name}")
            continue
        pairs.append((img_path, json_path))

    return pairs


def copy_split(
    pairs: list[tuple[Path, Path]],
    split_name: str,
) -> None:
    """ペアリストを指定 split (train/val) のディレクトリに 320×240 で保存する。

    画像は PIL でリサイズして JPEG 保存、マスクは binary PNG として保存する。
    """
    img_out: Path  = DATASET_DIR / "images" / split_name
    mask_out: Path = DATASET_DIR / "masks" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    for img_path, json_path in pairs:
        # 画像を 320×240 にリサイズして保存
        # LANCZOS: 縮小時に品質が高いリサンプリングフィルタ
        img: Image.Image = Image.open(img_path).resize(
            (TARGET_W, TARGET_H), Image.LANCZOS
        )
        img.save(img_out / img_path.name, quality=95)

        mask: Image.Image = json_to_mask(json_path)
        mask.save(mask_out / f"{json_path.stem}.png")

    print(f"  {split_name}: {len(pairs)} サンプル → {img_out}")


def main() -> None:
    """データセット準備のエントリポイント。"""
    pairs: list[tuple[Path, Path]] = collect_pairs()
    print(f"有効なペア数: {len(pairs)}")

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    split_idx: int = int(len(pairs) * TRAIN_RATIO)
    train_pairs: list[tuple[Path, Path]] = pairs[:split_idx]
    val_pairs: list[tuple[Path, Path]]   = pairs[split_idx:]

    print(f"train: {len(train_pairs)} / val: {len(val_pairs)}")
    copy_split(train_pairs, "train")
    copy_split(val_pairs,   "val")
    print("完了。")


if __name__ == "__main__":
    main()
