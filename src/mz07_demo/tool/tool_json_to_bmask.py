"""
tool_json_to_bmask.py

sample_data/ 内の LabelMe JSON からバイナリマスク (PNG) を生成する。

usage:
  cd src/mz07_demo && python tool/tool_json_to_bmask.py
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw


def json_to_mask(json_path: Path, target_w: int, target_h: int) -> Image.Image:
    """
    LabelMe polygon JSON を読み込み、target_w × target_h のバイナリマスクを返す。

    cable: 1 / background: 0
    """
    data: dict = json.loads(json_path.read_text())
    orig_w: int   = data["imageWidth"]
    orig_h: int   = data["imageHeight"]
    scale_x: float = target_w / orig_w
    scale_y: float = target_h / orig_h

    mask: Image.Image          = Image.new("L", (target_w, target_h), 0)
    draw: ImageDraw.ImageDraw  = ImageDraw.Draw(mask)

    for shape in data["shapes"]:
        if shape["label"] == "cable":
            pts: list[tuple[float, float]] = [
                (p[0] * scale_x, p[1] * scale_y) for p in shape["points"]
            ]
            draw.polygon(pts, fill=1)

    return mask


def main() -> None:
    """sample_data/ 内の全 JSON をバイナリマスク PNG に変換する。"""
    here: Path     = Path(__file__).resolve().parent
    json_dir: Path = here.parent / "sample_data"
    mask_dir: Path = here.parent / "sample_data" / "masks"
    target_w: int  = 640
    target_h: int  = 480

    mask_dir.mkdir(parents=True, exist_ok=True)

    json_paths: list[Path] = sorted(json_dir.glob("*.json"))
    if not json_paths:
        print(f"[WARN] no JSON files found: {json_dir}")
        return

    json_path: Path
    for json_path in json_paths:
        mask: Image.Image = json_to_mask(json_path, target_w, target_h)
        out_path: Path = mask_dir / f"{json_path.stem}.png"
        mask.save(out_path)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
