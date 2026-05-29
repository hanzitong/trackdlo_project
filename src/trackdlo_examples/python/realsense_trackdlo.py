"""
realsense_trackdlo.py  (trackdlo_cdll / ctypes 版)

trackdlo_python/examples/realsense_trackdlo.py と同じパイプラインを
pybind11 ではなく ctypes (Python 標準ライブラリ) で実装したバージョン。

ctypes 版と pybind11 版の主な違い:
  - import: sys.path で trackdlo_cdll/python/ を追加して直接 import
  - TrackdloState: Python クラスが C++ の void* ハンドルをラップ
  - TdloParams: ctypes.Structure (フィールドへの代入で設定)
  - state.Y: property 経由でコピーを取得/設定 (pybind11 版は参照)
  - state.geodesic_coord: numpy 配列を setter で渡す

パイプライン:
  Step 1  RealSense D405 から カラー + 深度フレームを取得
  Step 2  DeepLabV3+ でカラー画像からバイナリマスクを生成
  Step 3  バイナリマスク + 深度画像 → 3D 点群 (N×3)
  Step 4  初回フレーム: 点群から trackdlo 状態を初期化
          2フレーム目以降: tracking_step() でノード座標を更新
  Step 5  state.Y (M×3) の全ノード座標をターミナルに出力

キー操作:
  Space   強制再初期化
  ESC     終了

実行 (ワークスペースルートから):
  python3 src/trackdlo_examples/python/realsense_trackdlo.py
"""

import sys
from pathlib import Path

# trackdlo_cdll.py は src/trackdlo_cdll/python/ にある純粋な .py ファイル。
# このファイルの位置: src/trackdlo_examples/python/
# trackdlo_cdll.py:   src/trackdlo_cdll/python/
_SRC = Path(__file__).resolve().parents[2]   # src/ ディレクトリ
sys.path.insert(0, str(_SRC / "trackdlo_cdll" / "python"))

from trackdlo_cdll import (
    TrackdloState,
    TdloParams,
    default_params,
    tracking_step,
    cpd_lle,
    sort_pts,
)

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import segmentation_models_pytorch as smp

# =============================================================================
# 設定
# =============================================================================

WEIGHTS   = _SRC / "bmask_gen/weights/best_deeplabv3plus_cable.pth"
NUM_NODES = 15
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Step A: RealSense D405 パイプラインの初期化
# =============================================================================

pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
profile  = pipeline.start(cfg)

align = rs.align(rs.stream.color)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
FX, FY = intr.fx, intr.fy
CX, CY = intr.ppx, intr.ppy
print(f"[D405] fx={FX:.1f}  fy={FY:.1f}  cx={CX:.1f}  cy={CY:.1f}")

for _ in range(5):
    pipeline.wait_for_frames()
print("[D405] warm-up done")


# =============================================================================
# Step B: DeepLabV3+ モデルのロード
# =============================================================================

model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(DEVICE)

model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
print(f"[DeepLab] loaded  weights={WEIGHTS.name}  device={DEVICE}")


# =============================================================================
# trackdlo パラメータの設定
#
# ctypes 版では default_params() で TdloParams 構造体を作り、
# フィールドに直接代入して設定する。
# pybind11 版の TrackdloParams() と同じ使い勝手になるよう設計されている。
# =============================================================================
params = default_params()
params.max_iter = 150   # デフォルト 50 → 実世界ノイズへの対応
params.tol      = 1e-4  # デフォルト 1e-5 → 緩和して収束しやすくする
# params.beta     = 5.0
# params.lambda_  = 1.0
# params.mu       = 0.05


# =============================================================================
# ヘルパー関数
# =============================================================================

def infer_mask(bgr_frame: np.ndarray) -> np.ndarray:
    """BGR フレーム → バイナリマスク (uint8, 0=背景, 1=ケーブル)"""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x   = torch.tensor(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(x)
    prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def mask_depth_to_pointcloud(mask: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """
    バイナリマスク + 深度画像 → 3D 点群 (N×3, float64, 単位 m)

    ピンホールカメラの逆投影:
      Z = depth_mm[v, u] / 1000.0
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
    """
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        return np.empty((0, 3))

    z_mm  = depth_mm[vs, us].astype(np.float64)
    valid = z_mm > 0
    vs, us, z_mm = vs[valid], us[valid], z_mm[valid]
    if len(us) == 0:
        return np.empty((0, 3))

    Z   = z_mm / 1000.0
    pts = np.column_stack([
        (us - CX) * Z / FX,
        (vs - CY) * Z / FY,
        Z,
    ])

    # 5mm ボクセルグリッドでダウンサンプリング
    voxels = np.floor(pts / 0.005).astype(np.int32)
    _, idx = np.unique(voxels, axis=0, return_index=True)
    return pts[idx]


def initialize_trackdlo(X: np.ndarray) -> TrackdloState:
    """
    初回フレーム: 点群 X から TrackdloState を作る。

    ctypes 版と pybind11 版の違い:
      - TrackdloState(NUM_NODES) でハンドルを作成 (make_trackdlo_state は不要)
      - state.Y = arr  → C++ 側に値をコピーして設定 (setter 呼び出し)
      - state.geodesic_coord = numpy 配列  (list ではなく ndarray を渡す)
    """
    state  = TrackdloState(NUM_NODES)
    Y_init = np.linspace(X.min(axis=0), X.max(axis=0), NUM_NODES)

    X_c = np.ascontiguousarray(X, dtype=np.float64)
    Y_fit, sigma2, converged = cpd_lle(
        X_c, Y_init, 0.0,
        beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05,
    )
    print(f"  [cpd_lle] converged={converged}  sigma2={sigma2:.6f}")

    Y_sorted = sort_pts(Y_fit)

    state.Y      = Y_sorted
    state.sigma2 = sigma2

    dists = np.linalg.norm(np.diff(Y_sorted, axis=0), axis=1)
    # ctypes 版の geodesic_coord setter は numpy 配列を期待する
    state.geodesic_coord = np.concatenate([[0.0], np.cumsum(dists)])

    state.guide_nodes = Y_sorted   # write-only property
    return state


def print_key_points(frame_no: int, Y: np.ndarray) -> None:
    """
    trackdlo が推定したケーブルの全ノード座標 (key_points) をターミナルに出力する。

    ctypes 版では state.Y が呼ぶたびに C++ からコピーして返す。
    pybind11 版と出力フォーマットは同じ。
    """
    M = len(Y)
    print(f"\n[frame {frame_no:04d}]  key_points ({M} nodes)")
    for i, (x, y, z) in enumerate(Y):
        print(f"  node {i:02d}:  X={x:+.4f}  Y={y:+.4f}  Z={z:.4f}  [m]")


def draw_overlay(bgr: np.ndarray, mask: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """カラーフレームにマスクオーバーレイ・ノード・エッジを描画する"""
    overlay = bgr.copy()
    overlay[mask == 1] = (0, 180, 0)
    vis = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)

    prev_px = None
    for x3, y3, z3 in Y:
        if z3 <= 0:
            continue
        px = int(x3 * FX / z3 + CX)
        py = int(y3 * FY / z3 + CY)
        cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
        if prev_px is not None:
            cv2.line(vis, prev_px, (px, py), (0, 220, 255), 2)
        prev_px = (px, py)

    return vis


# =============================================================================
# メインループ
# =============================================================================
all_idx  = list(range(NUM_NODES))
state    = None
frame_no = 0
WIN      = "trackdlo (ctypes)  [Space: init  ESC: quit]"

print("\n[INFO] Space: 初期化  ESC: 終了\n")

try:
    while True:
        # Step 1: RealSense からカラー + アライン済み深度を取得
        frames         = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        bgr      = np.asanyarray(color_frame.get_data())
        depth_mm = np.asanyarray(depth_frame.get_data())

        # Step 2: DeepLabV3+ でバイナリマスクを生成
        mask = infer_mask(bgr)

        # Step 3: バイナリマスク + 深度画像 → 3D 点群
        X = mask_depth_to_pointcloud(mask, depth_mm)

        # Step 4: trackdlo 状態の初期化 / トラッキング
        if len(X) >= NUM_NODES * 3:
            X_c = np.ascontiguousarray(X, dtype=np.float64)

            if state is None:
                print("[INIT] initializing trackdlo...")
                state = initialize_trackdlo(X_c)
                frame_no = 0
                print("[INIT] done")
            else:
                tracking_step(state, X_c, all_idx, all_idx, params)
                frame_no += 1

            # Step 5: 全 key_point を出力
            # ctypes 版は state.Y を呼ぶたびに C++ からコピーするため、
            # ループ内で何度も呼ぶ場合は一度変数に受けておく。
            Y_now = state.Y
            print_key_points(frame_no, Y_now)

        # 可視化 (常に実行)
        if state is not None:
            vis = draw_overlay(bgr, mask, state.Y)
        else:
            vis = bgr.copy()
            cv2.putText(vis, "waiting for cable... (Space to init)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow(WIN, vis)
        cv2.imshow("mask", mask * 255)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord(' '):
            state = None

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("done.")
