
from pathlib import Path
import numpy as np
from PIL import Image

# data/raw/masks/ にあるマスク画像をバイナリ（0/1）に変換する
ROOT = Path(__file__).resolve().parent.parent
mask_dir = ROOT / "data/raw/masks"

for p in mask_dir.glob("*.png"):
    m = np.array(Image.open(p))
    m = (m > 0).astype(np.uint8)
    Image.fromarray(m).save(p)


