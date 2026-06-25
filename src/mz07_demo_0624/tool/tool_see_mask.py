"""
tool_see_mask.py

sample_data/ 内のバイナリマスクとオリジナル画像を並べて表示する。

key controls:
  right arrow  次のサンプルへ
  left arrow   前のサンプルへ
  ESC          終了

usage:
  cd src/mz07_demo && python tool/tool_see_mask.py
"""

from pathlib import Path

import cv2
import numpy as np


def load_pairs(sample_dir: Path, mask_dir: Path) -> "list[tuple[Path, Path]]":
    """マスクが存在する (bgr_path, mask_path) ペアを返す。"""
    pairs: list[tuple[Path, Path]] = []
    img_path: Path
    for img_path in sorted(sample_dir.glob("bgr_*.png")):
        mask_path: Path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def show(idx: int, pairs: "list[tuple[Path, Path]]") -> None:
    """idx 番目の BGR 画像とマスクを cv2.imshow で表示する。"""
    img_path: Path
    mask_path: Path
    img_path, mask_path = pairs[idx]

    bgr: np.ndarray  = cv2.imread(str(img_path))
    mask: np.ndarray = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    mask_vis: np.ndarray = mask * 255

    title: str = f"[{idx + 1}/{len(pairs)}] {img_path.name}"
    cv2.imshow("original", bgr)
    cv2.imshow("mask (white=cable)", mask_vis)
    cv2.setWindowTitle("original", f"original  {title}")
    cv2.setWindowTitle("mask (white=cable)", f"mask  {title}")


def main() -> None:
    """マスクビューアのエントリポイント。"""
    here: Path       = Path(__file__).resolve().parent
    sample_dir: Path = here.parent / "sample_data"
    mask_dir: Path   = sample_dir / "masks"

    # cv2.waitKeyEx が返す矢印キーのコード (Linux / Windows 両対応)
    key_right_linux: int   = 65363
    key_left_linux: int    = 65361
    key_right_windows: int = 2555904
    key_left_windows: int  = 2424832
    key_esc: int           = 27

    pairs: list[tuple[Path, Path]] = load_pairs(sample_dir, mask_dir)
    if not pairs:
        print("[ERROR] no samples found")
        return

    print(f"{len(pairs)} samples loaded")
    print("left/right arrow: navigate  ESC: quit")

    idx: int = 0
    show(idx, pairs)

    while True:
        key: int = cv2.waitKeyEx(0)
        if key == key_esc:
            break
        elif key in (key_right_linux, key_right_windows):
            idx = (idx + 1) % len(pairs)
            show(idx, pairs)
        elif key in (key_left_linux, key_left_windows):
            idx = (idx - 1) % len(pairs)
            show(idx, pairs)

    cv2.destroyAllWindows()
    print("done.")


if __name__ == "__main__":
    main()
