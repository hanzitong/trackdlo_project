"""
see_mask.py

sample_data/ 内のバイナリマスクとオリジナル画像を並べて表示する。

─── キー操作 ─────────────────────────────────────────────────────────────────
  right arrow  次のサンプルへ
  left arrow   前のサンプルへ
  ESC          終了

─── 実行 ─────────────────────────────────────────────────────────────────────
  cd src/mz07_demo && python see_mask.py
"""

from pathlib import Path

import cv2
import numpy as np


ROOT: Path       = Path(__file__).resolve().parent
SAMPLE_DIR: Path = ROOT / "sample_data"
MASK_DIR: Path   = SAMPLE_DIR / "masks"

# cv2.waitKeyEx が返す矢印キーのコード
# Linux と Windows で異なるため両方に対応する
KEY_RIGHT_LINUX:   int = 65363
KEY_LEFT_LINUX:    int = 65361
KEY_RIGHT_WINDOWS: int = 2555904
KEY_LEFT_WINDOWS:  int = 2424832
KEY_ESC:           int = 27


def load_pairs() -> list[tuple[Path, Path]]:
    """マスクが存在する (bgr_path, mask_path) ペアを返す。"""
    pairs: list[tuple[Path, Path]] = []
    img_path: Path
    for img_path in sorted(SAMPLE_DIR.glob("bgr_*.png")):
        mask_path: Path = MASK_DIR / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def show(idx: int, pairs: list[tuple[Path, Path]]) -> None:
    """idx 番目の BGR 画像とマスクを cv2.imshow で表示する。"""
    img_path: Path
    mask_path: Path
    img_path, mask_path = pairs[idx]

    bgr: np.ndarray  = cv2.imread(str(img_path))
    mask: np.ndarray = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    # マスク: 0/1 を 0/255 に変換して白黒で表示
    mask_vis: np.ndarray = mask * 255

    title: str = f"[{idx + 1}/{len(pairs)}] {img_path.name}"
    cv2.imshow("original", bgr)
    cv2.imshow("mask (white=cable)", mask_vis)

    # ウィンドウタイトルにインデックスを反映するため一度作り直す
    cv2.setWindowTitle("original", f"original  {title}")
    cv2.setWindowTitle("mask (white=cable)", f"mask  {title}")


def main() -> None:
    """マスクビューアのエントリポイント。"""
    pairs: list[tuple[Path, Path]] = load_pairs()
    if not pairs:
        print("[ERROR] no samples found")
        return

    print(f"{len(pairs)} samples loaded")
    print("left/right arrow: navigate  ESC: quit")

    idx: int = 0
    show(idx, pairs)

    while True:
        # waitKeyEx は通常の waitKey より大きいコードを返し、矢印キーを識別できる
        key: int = cv2.waitKeyEx(0)

        if key == KEY_ESC:
            break
        elif key in (KEY_RIGHT_LINUX, KEY_RIGHT_WINDOWS):
            idx = (idx + 1) % len(pairs)
            show(idx, pairs)
        elif key in (KEY_LEFT_LINUX, KEY_LEFT_WINDOWS):
            idx = (idx - 1) % len(pairs)
            show(idx, pairs)

    cv2.destroyAllWindows()
    print("done.")


if __name__ == "__main__":
    main()
