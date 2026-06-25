"""
test_infer_trackdlo.py

sample_data/ 内のバイナリマスク + デプス画像から TrackDLO でケーブルキーポイントを抽出する
オフラインデモ。

data layout:
  sample_data/
    bgr_000N.png        : BGR カラー画像 (640x480, uint8)
    depth_000N.png      : 深度画像 (640x480, uint16, 単位 mm)
    masks/bgr_000N.png  : バイナリマスク (640x480, uint8, 0=背景/1=ケーブル)

pipeline:
  1. bgr + depth + mask を読み込む
  2. images_to_pointcloud() で 3D 点群 X (Nx3) を生成
  3. 初回フレーム: cpd_lle() + sort_pts() で TrackdloState を初期化
  4. 以降のフレーム: compute_visible_nodes() + tracking_step() で更新
  5. ノード座標をターミナル出力 & カラー画像にオーバーレイして表示

key controls:
  any key  次フレームへ
  q / ESC  終了

usage:
  cd src/mz07_demo && python test/test_infer_trackdlo.py
"""

import os
import sys
import platform
from dataclasses import dataclass
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent

# .so / .dll のパスを import より先に環境変数へセットする。
# _load_lib() は import 時に実行されるため、この順序が必須。
_lib_ext: str = ".dll" if platform.system() == "Windows" else ".so"
os.environ["TRACKDLO_LIB_PATH"]      = str(HERE.parent / "lib" / f"libtrackdlo_c{_lib_ext}")
os.environ["PREPROCESSING_LIB_PATH"] = str(HERE.parent / "lib" / f"libpreprocessing_c{_lib_ext}")

# trackdlo_cdll.py / preprocessing_cdll.py は mz07_demo/ にある
sys.path.insert(0, str(HERE.parent))

import cv2
import numpy as np

from trackdlo_cdll import (
    TrackdloState,
    TdloParams,
    default_params,
    tracking_step,
    cpd_lle,
    sort_pts,
)
from preprocessing_cdll import images_to_pointcloud, compute_visible_nodes


@dataclass
class CameraIntrinsics:
    """カメラ内部パラメータ (ピンホールモデル)。"""
    fx: float
    fy: float
    cx: float
    cy: float


def load_frames(
    sample_dir: Path,
    mask_dir: Path,
    depth_scale: float,
) -> "list[tuple[np.ndarray, np.ndarray, np.ndarray]]":
    """マスクが存在するフレームの (bgr, depth_mm, mask) リストを返す。

    depth は mm 単位 (uint16) に換算して返す。
    """
    mask_paths: list[Path] = sorted(mask_dir.glob("bgr_*.png"))
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for mask_path in mask_paths:
        stem: str = mask_path.stem
        idx: str  = stem.split("_")[1]

        bgr_path: Path   = sample_dir / f"bgr_{idx}.png"
        depth_path: Path = sample_dir / f"depth_{idx}.png"

        if not bgr_path.exists() or not depth_path.exists():
            print(f"[SKIP] bgr/depth not found: idx={idx}")
            continue

        bgr: np.ndarray       = cv2.imread(str(bgr_path))
        depth_raw: np.ndarray = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        mask: np.ndarray      = cv2.imread(str(mask_path),  cv2.IMREAD_UNCHANGED)

        if bgr is None or depth_raw is None or mask is None:
            print(f"[SKIP] failed to load: idx={idx}")
            continue

        depth_mm: np.ndarray = np.clip(
            depth_raw.astype(np.float32) * depth_scale * 1000.0, 0, 65535
        ).astype(np.uint16)

        frames.append((bgr, depth_mm, mask))
        print(f"[LOAD] idx={idx}  bgr={bgr.shape}  depth={depth_mm.shape}  "
              f"mask_pixels={int(mask.sum())}")

    return frames


def initialize_state(X: np.ndarray, num_nodes: int) -> "TrackdloState | None":
    """点群 X から TrackdloState を初期化して返す。失敗時は None。"""
    state: TrackdloState = TrackdloState(num_nodes)

    Y_init: np.ndarray = np.linspace(X.min(axis=0), X.max(axis=0), num_nodes)

    Y_fit: np.ndarray
    sigma2: float
    converged: bool
    Y_fit, sigma2, converged = cpd_lle(
        X, Y_init, sigma2=0.0,
        beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05,
    )
    print(f"  [cpd_lle] converged={converged}  sigma2={sigma2:.6f}")

    if not np.all(np.isfinite(Y_fit)):
        print("  [cpd_lle] NaN/Inf detected, skipping initialization")
        state.close()
        return None

    Y_sorted: np.ndarray = sort_pts(Y_fit)
    if not np.all(np.isfinite(Y_sorted)):
        print("  [sort_pts] NaN/Inf detected, skipping initialization")
        state.close()
        return None

    state.Y      = Y_sorted
    state.sigma2 = sigma2

    dists: np.ndarray = np.linalg.norm(np.diff(Y_sorted, axis=0), axis=1)
    state.geodesic_coord = np.concatenate([[0.0], np.cumsum(dists)])
    state.guide_nodes    = Y_sorted
    return state


def draw_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    Y: np.ndarray,
    intr: CameraIntrinsics,
) -> np.ndarray:
    """マスク(緑) + ノード(赤丸) + エッジ(黄線) をオーバーレイした画像を返す。"""
    overlay: np.ndarray = bgr.copy()
    overlay[mask == 1] = (0, 180, 0)
    vis: np.ndarray = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)

    prev_px: "tuple[int, int] | None" = None
    x3: float
    y3: float
    z3: float
    for x3, y3, z3 in Y:
        if not (np.isfinite(x3) and np.isfinite(y3) and np.isfinite(z3) and z3 > 0):
            continue
        px: int = int(x3 * intr.fx / z3 + intr.cx)
        py: int = int(y3 * intr.fy / z3 + intr.cy)
        cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
        if prev_px is not None:
            cv2.line(vis, prev_px, (px, py), (0, 220, 255), 2)
        prev_px = (px, py)

    return vis


def main() -> None:
    """オフラインデモのエントリポイント。"""
    num_nodes: int = 10
    # 旧 sample_data は raw uint16 (0.0001 m/unit) で保存されている
    # 新しいデータ (take_depth_rgb_image.py 修正後) は mm 単位のため depth_scale=1.0 に変更すること
    depth_scale: float = 0.0001

    # RealSense D405 640x480 の典型的な内部パラメータ (実機から取得した値があればそちらを優先)
    intr: CameraIntrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0)

    sample_dir: Path = HERE.parent / "sample_data"
    mask_dir: Path   = sample_dir / "masks"

    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = load_frames(
        sample_dir, mask_dir, depth_scale
    )
    if not frames:
        print("[ERROR] no frames found")
        return
    print(f"\n{len(frames)} frames loaded\n")

    params: TdloParams = default_params()
    params.max_iter = 150
    params.tol      = 1e-4

    state: "TrackdloState | None" = None
    frame_no: int = 0

    try:
        bgr: np.ndarray
        depth_mm: np.ndarray
        mask: np.ndarray
        for bgr, depth_mm, mask in frames:
            X: np.ndarray = images_to_pointcloud(
                bgr, depth_mm, mask, intr.fx, intr.fy, intr.cx, intr.cy, leaf_size=0.005
            )
            print(f"\n[frame {frame_no:04d}] point cloud: {len(X)} pts")

            if len(X) < num_nodes * 3:
                print("  [SKIP] too few points (need at least num_nodes*3)")
                frame_no += 1
                continue

            X = np.ascontiguousarray(X, dtype=np.float64)

            if state is None:
                print("  [INIT] initializing TrackdloState...")
                state = initialize_state(X, num_nodes)
                if state is None:
                    print("  [INIT] failed, retry on next frame")
                    frame_no += 1
                    continue
                print("  [INIT] done")
            else:
                vn: list[int]
                vne: list[int]
                if frame_no == 0:
                    vn = vne = list(range(num_nodes))
                else:
                    img_rows: int
                    img_cols: int
                    img_rows, img_cols = bgr.shape[:2]
                    vn, vne = compute_visible_nodes(
                        state.Y, X, state.geodesic_coord,
                        intr.fx, intr.fy, intr.cx, intr.cy,
                        img_rows, img_cols,
                    )
                tracking_step(state, X, vn, vne, params)
                frame_no += 1

                if not np.all(np.isfinite(state.Y)):
                    print("  [WARN] NaN detected, resetting state")
                    state.close()
                    state = None
                    continue

            Y_now: np.ndarray = state.Y
            i: int
            x: float
            y: float
            z: float
            for i, (x, y, z) in enumerate(Y_now):
                print(f"  node {i:02d}:  X={x:+.4f}  Y={y:+.4f}  Z={z:.4f}  [m]")

            vis: np.ndarray = draw_overlay(bgr, mask, Y_now, intr)
            cv2.imshow("test_infer_trackdlo  [any key: next / q,ESC: quit]", vis)
            cv2.imshow("mask", mask * 255)

            key: int = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        if state is not None:
            state.close()
        cv2.destroyAllWindows()
        print("\ndone.")


if __name__ == "__main__":
    main()
