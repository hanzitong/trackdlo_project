"""
infer_trackdlo.py

sample_data/ 内のバイナリマスク + デプス画像からTrackDLOでケーブルキーポイントを抽出する
オフラインデモ。

─── データ構成 ──────────────────────────────────────────────────────────────
  sample_data/
    bgr_000N.png        : BGR カラー画像 (640×480, uint8)
    depth_000N.png      : 深度画像 (640×480, uint16, 単位 mm)
    masks/bgr_000N.png  : バイナリマスク (640×480, uint8, 0=背景/1=ケーブル)
                           N = 0〜3 の 4 フレーム分のみ存在

─── パイプライン ─────────────────────────────────────────────────────────────
  1. bgr + depth + mask を読み込む
  2. images_to_pointcloud() で 3D 点群 X (N×3) を生成
  3. 初回フレーム: cpd_lle() + sort_pts() で TrackdloState を初期化
  4. 以降のフレーム: compute_visible_nodes() + tracking_step() で更新
  5. ノード座標をターミナル出力 & カラー画像にオーバーレイして表示

─── キー操作 ─────────────────────────────────────────────────────────────────
  任意キー  次フレームへ
  q / ESC  終了

─── 実行 ─────────────────────────────────────────────────────────────────────
  cd src/mz07_demo && python infer_trackdlo.py
"""

import os
import sys
import platform
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent

# .so / .dll のパスを import より先に環境変数へセットする。
# _load_lib() は import 時に実行されるため、この順序が必須。
#
# Windows に移植する場合は以下のコメントアウトと入れ替える:
#   _LIB_EXT = ".dll" if platform.system() == "Windows" else ".so"
#   os.environ["TRACKDLO_LIB_PATH"]      = str(HERE / f"libtrackdlo_c{_LIB_EXT}")
#   os.environ["PREPROCESSING_LIB_PATH"] = str(HERE / f"libpreprocessing_c{_LIB_EXT}")
#
# また Windows では MinGW ランタイム DLL も mz07_demo/ に置く必要がある:
#   libgcc_s_seh-1.dll, libstdc++-6.dll, libwinpthread-1.dll
#   (pipeline_demo/ に既にある)
os.environ["TRACKDLO_LIB_PATH"]      = str(HERE / "libtrackdlo_c.so")
os.environ["PREPROCESSING_LIB_PATH"] = str(HERE / "libpreprocessing_c.so")

# このディレクトリにコピーした trackdlo_cdll.py / preprocessing_cdll.py を優先 import
sys.path.insert(0, str(HERE))

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


# ─── 設定 ─────────────────────────────────────────────────────────────────────

NUM_NODES: int = 10

# RealSense D405 640×480 の典型的な内部パラメータ。
# 実機から取得した値がある場合はそちらを使うこと。
FX: float = 385.0
FY: float = 385.0
CX: float = 320.0
CY: float = 240.0

# sample_data の depth PNG は take_depth_rgb_image.py で保存された値。
# 旧バージョンで raw uint16 (0.1mm 単位) のまま保存されている場合は
# load_frames() 内で mm 換算する。新バージョン (take_depth_rgb_image.py 修正後) で
# 保存したデータはすでに mm 単位のため、DEPTH_SCALE = 1.0 に変更すること。
DEPTH_SCALE: float = 0.0001  # m/unit (旧 sample_data 用)

SAMPLE_DIR: Path = HERE / "sample_data"
MASK_DIR: Path   = SAMPLE_DIR / "masks"


# ─── データ読み込み ────────────────────────────────────────────────────────────

def load_frames() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """マスクが存在するフレームの (bgr, depth_mm, mask) リストを返す。

    depth は mm 単位 (uint16) に換算して返す。
    masks/ 以下の bgr_*.png をキーに、対応する bgr と depth を読み込む。
    """
    mask_paths: list[Path] = sorted(MASK_DIR.glob("bgr_*.png"))
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for mask_path in mask_paths:
        stem: str = mask_path.stem          # "bgr_0000"
        idx: str  = stem.split("_")[1]      # "0000"

        bgr_path: Path   = SAMPLE_DIR / f"bgr_{idx}.png"
        depth_path: Path = SAMPLE_DIR / f"depth_{idx}.png"

        if not bgr_path.exists() or not depth_path.exists():
            print(f"[SKIP] bgr/depth not found: idx={idx}")
            continue

        bgr: np.ndarray       = cv2.imread(str(bgr_path))
        depth_raw: np.ndarray = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        mask: np.ndarray      = cv2.imread(str(mask_path),  cv2.IMREAD_UNCHANGED)

        if bgr is None or depth_raw is None or mask is None:
            print(f"[SKIP] failed to load: idx={idx}")
            continue

        # raw → mm 換算 (以降のパイプラインは depth_mm = uint16 mm 単位で統一)
        depth_mm: np.ndarray = np.clip(
            depth_raw.astype(np.float32) * DEPTH_SCALE * 1000.0, 0, 65535
        ).astype(np.uint16)

        frames.append((bgr, depth_mm, mask))
        print(f"[LOAD] idx={idx}  bgr={bgr.shape}  depth={depth_mm.shape}  "
              f"mask_pixels={int(mask.sum())}")

    return frames


# ─── TrackdloState 初期化 ─────────────────────────────────────────────────────

def initialize_state(X: np.ndarray) -> "TrackdloState | None":
    """点群 X から TrackdloState を初期化して返す。失敗時は None。

    処理の流れ:
      1. X の最小/最大点を結ぶ等間隔の初期ノード Y_init を作る
      2. cpd_lle() で Y_init を点群 X にフィッティング
      3. sort_pts() で最近傍順に並べ直す
      4. 測地線座標 (累積弧長) をセット
    """
    state: TrackdloState = TrackdloState(NUM_NODES)

    Y_init: np.ndarray = np.linspace(X.min(axis=0), X.max(axis=0), NUM_NODES)

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

    # 測地線座標: ノード 0 から各ノードまでの累積経路長 [m]
    dists: np.ndarray = np.linalg.norm(np.diff(Y_sorted, axis=0), axis=1)
    state.geodesic_coord = np.concatenate([[0.0], np.cumsum(dists)])
    state.guide_nodes    = Y_sorted
    return state


# ─── 可視化 ───────────────────────────────────────────────────────────────────

def draw_overlay(bgr: np.ndarray, mask: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """マスク(緑) + ノード(赤丸) + エッジ(黄線) をオーバーレイした画像を返す。

    3D 座標 (x3, y3, z3) をピンホールモデルで 2D 投影:
      px = x3 * FX / z3 + CX
      py = y3 * FY / z3 + CY
    """
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
        px: int = int(x3 * FX / z3 + CX)
        py: int = int(y3 * FY / z3 + CY)
        cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
        if prev_px is not None:
            cv2.line(vis, prev_px, (px, py), (0, 220, 255), 2)
        prev_px = (px, py)

    return vis


# ─── メイン ───────────────────────────────────────────────────────────────────

def main() -> None:
    """オフラインデモのエントリポイント。"""
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = load_frames()
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

            # Step 1: 点群を生成 (depth_mm は load_frames() で mm 換算済み)
            X: np.ndarray = images_to_pointcloud(
                bgr, depth_mm, mask, FX, FY, CX, CY, leaf_size=0.005
            )
            print(f"\n[frame {frame_no:04d}] point cloud: {len(X)} pts")

            if len(X) < NUM_NODES * 3:
                print("  [SKIP] too few points (need at least NUM_NODES*3)")
                frame_no += 1
                continue

            X = np.ascontiguousarray(X, dtype=np.float64)

            if state is None:
                # Step 2: 初回フレームは初期化
                print("  [INIT] initializing TrackdloState...")
                state = initialize_state(X)
                if state is None:
                    print("  [INIT] failed, retry on next frame")
                    frame_no += 1
                    continue
                print("  [INIT] done")
            else:
                # Step 3: 2フレーム目以降はトラッキング
                # frame_no==0 の初回 tracking_step は state.Y が安定していないため
                # 全ノードを可視として渡す (compute_visible_nodes をスキップ)
                vn: list[int]
                vne: list[int]
                if frame_no == 0:
                    vn = vne = list(range(NUM_NODES))
                else:
                    img_rows: int
                    img_cols: int
                    img_rows, img_cols = bgr.shape[:2]
                    vn, vne = compute_visible_nodes(
                        state.Y, X, state.geodesic_coord,
                        FX, FY, CX, CY,
                        img_rows, img_cols,
                    )
                tracking_step(state, X, vn, vne, params)
                frame_no += 1

                if not np.all(np.isfinite(state.Y)):
                    print("  [WARN] NaN detected, resetting state")
                    state.close()
                    state = None
                    continue

            # Step 4: ノード座標を出力
            Y_now: np.ndarray = state.Y
            i: int
            x: float
            y: float
            z: float
            for i, (x, y, z) in enumerate(Y_now):
                print(f"  node {i:02d}:  X={x:+.4f}  Y={y:+.4f}  Z={z:.4f}  [m]")

            # Step 5: 可視化
            vis: np.ndarray = draw_overlay(bgr, mask, Y_now)
            cv2.imshow("infer_trackdlo  [any key: next / q,ESC: quit]", vis)
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
