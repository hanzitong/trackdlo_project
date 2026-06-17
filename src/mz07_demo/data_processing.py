
"""
This script conduct date pre-processing for DNN input

make binary mask 

"""


import cv2
import numpy as np

from pathlib import Path
import json

from PIL import Image, ImageDraw


ROOT: Path = Path(__file__).resolve().parent
JSON_DIR: Path = ROOT / "sample_data"
MASK_DIR: Path = ROOT / "sample_data" / "masks"

# RealSense D405 / LabelMe annotation resolution (must match bgr_*.png, depth_*.png)
TARGET_W: int = 640
TARGET_H: int = 480

TRAIN_RATIO: float = 0.8
RANDOM_SEED: int = 77



def json_to_mask(json_path: Path) -> Image.Image:
    """
    read labelme polygon JSON and output TARGET_W x TARGET_H binary mask

    cable: 1
    background: 0
    """
    data: dict = json.loads(json_path.read_text())
    orig_w: int = data["imageWidth"]
    orig_h: int = data["imageHeight"]
    scale_x: float = TARGET_W / orig_w
    scale_y: float = TARGET_H / orig_h

    # module method new() creates a new Image object. "L" stands for grayscale, "0" is the value.
    mask: Image.Image = Image.new("L", (TARGET_W, TARGET_H), 0)
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(mask)

    for shape in data["shapes"]:
        if shape["label"] == "cable":
            pts: list[tuple[float, float]] = [
                (p[0] * scale_x, p[1] * scale_y) for p in shape["points"]
            ]
            draw.polygon(pts, fill=1)

    return mask


def main() -> None:
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    json_paths: list[Path] = sorted(JSON_DIR.glob("*.json"))
    if not json_paths:
        print(f"[WARN] no JSON files found: {JSON_DIR}")
        return

    for json_path in json_paths:
        mask: Image.Image = json_to_mask(json_path)
        out_path: Path = MASK_DIR / f"{json_path.stem}.png"
        mask.save(out_path)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
