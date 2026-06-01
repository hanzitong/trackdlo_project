"""
pipeline_demo.py

カメラ映像からケーブルのノード座標をリアルタイムで推定するデモ。

─── パイプライン ───────────────────────────────────────────────────────────
  Step 1  RealSense D405 からカラー + 深度フレームを取得
  Step 2  DeepLabV3+ でカラー画像からバイナリマスクを生成
  Step 3  バイナリマスク + 深度画像 → 3D 点群 X (N×3)
  Step 4  初回 / Space キー: cpd_lle() + sort_pts() で TrackdloState を初期化
  Step 5  2 フレーム目以降: tracking_step() でノード座標 state.Y を更新
  Step 6  ノード座標をターミナル出力 & カラー画像にオーバーレイして表示

─── 前準備 ─────────────────────────────────────────────────────────────────
  初回のみ: bash setup.sh

─── 実行 ───────────────────────────────────────────────────────────────────
  python3 pipeline_demo.py

─── キー操作 ────────────────────────────────────────────────────────────────
  Space   trackdlo 状態を強制リセット (再初期化)
  ESC     終了

─── このデモのポイント ────────────────────────────────────────────────────
  このディレクトリに setup.sh でコピーした以下のファイルを直接読む:
    ./libtrackdlo_c.so             ← C++ アルゴリズムの共有ライブラリ
    ./best_deeplabv3plus_cable.pth ← DeepLabV3+ 学習済み重み

  環境変数 TRACKDLO_LIB_PATH に .so のパスをセットしてから trackdlo_cdll を
  import することで、trackdlo_cdll.py 自体はコピーせずに使い回せる。
"""

import os
import sys
from pathlib import Path

# ─── パス (import より先に確定させる) ────────────────────────────────────────
# HERE はこのスクリプトのあるディレクトリ = pipeline_demo/
HERE    = Path(__file__).resolve().parent
WEIGHTS = HERE / "best_deeplabv3plus_cable.pth"

if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"{WEIGHTS} が見つかりません。先に bash setup.sh を実行してください。"
    )

# TRACKDLO_LIB_PATH を設定してから trackdlo_cdll を import する。
# trackdlo_cdll.py の _load_lib() は import 時に実行されるため、
# 環境変数は必ず import より前にセットする必要がある。
os.environ["TRACKDLO_LIB_PATH"] = str(HERE / "libtrackdlo_c.so")

# trackdlo_cdll.py は src/trackdlo_cdll/python/ にある
_SRC = HERE.parents[3] / "src"   # pipeline_demo/ から4つ上が trackdlo_project/src/
sys.path.insert(0, str(_SRC / "trackdlo_cdll" / "python"))

import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import pyrealsense2 as rs

from trackdlo_cdll import (
    TrackdloState,
    default_params,
    tracking_step,
    cpd_lle,
    sort_pts,
)

# ─── 設定 ────────────────────────────────────────────────────────────────────
NUM_NODES = 5                                          # トラッキングするノード数
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEVICE}")


# =============================================================================
# Step B: DeepLabV3+ モデルのロード
#
# 学習時と同じアーキテクチャを定義してから重みだけを読み込む。
# encoder_weights=None: ImageNet 重みをダウンロードしない (state_dict で上書きするため)
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
# Step A: RealSense D405 パイプラインの初期化
#
# カラーストリームと深度ストリームを同時に有効化する。
# align を使うと深度画像をカラー画像の座標系に合わせてくれる。
# =============================================================================
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
profile  = pipeline.start(cfg)

# カメラ内部パラメータの取得 (ピンホール逆投影に使う)
intr   = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
FX, FY = intr.fx, intr.fy    # 焦点距離 [px]
CX, CY = intr.ppx, intr.ppy  # 主点 [px]
print(f"[D405] fx={FX:.1f}  fy={FY:.1f}  cx={CX:.1f}  cy={CY:.1f}")

# 深度フレームをカラー座標系に合わせるアライナー
align = rs.align(rs.stream.color)

# カメラが安定するまで数フレーム捨てる (露出が収束するまで待つ)
for _ in range(5):
    pipeline.wait_for_frames()
print("[D405] warm-up done")


# =============================================================================
# trackdlo パラメータ
#
# default_params() は C++ 側の TrackdloParams のデフォルト値を返す。
# フィールドに直接代入して調整する。
# =============================================================================
params = default_params()
params.max_iter = 150   # デフォルト 50 → 実世界ノイズへの対応で増やす
params.tol      = 1e-4  # デフォルト 1e-5 → 緩和して収束しやすくする


# =============================================================================
# ヘルパー関数
# =============================================================================

def infer_mask(bgr_frame: np.ndarray) -> np.ndarray:
    """
    BGR フレーム → バイナリマスク (uint8, 0=背景, 1=ケーブル)

    前処理は学習時 (train_deeplabv3plus_binary.py) と完全に同じにする。
    異なるとモデルの精度が落ちる。
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # (H,W,C) → (C,H,W) → (1,C,H,W) [バッチ次元を追加]
    x = torch.tensor(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def mask_depth_to_pointcloud(mask: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """
    バイナリマスク + 深度画像 → 3D 点群 (N×3, float64, 単位 m)

    ピンホールカメラの逆投影式:
      Z = depth_mm[v, u] / 1000.0        (mm → m 変換)
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy

    np.where(mask > 0) でマスクが 1 のピクセルの (行, 列) = (v, u) を取得し、
    上式で 3D 座標に変換する。
    """
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        return np.empty((0, 3), dtype=np.float64)

    z_mm  = depth_mm[vs, us].astype(np.float64)
    valid = z_mm > 0                           # 深度が無効 (0) なピクセルを除外
    vs, us, z_mm = vs[valid], us[valid], z_mm[valid]
    if len(us) == 0:
        return np.empty((0, 3), dtype=np.float64)

    Z   = z_mm / 1000.0
    pts = np.column_stack([
        (us - CX) * Z / FX,    # X [m]
        (vs - CY) * Z / FY,    # Y [m]
        Z,                      # Z [m] (カメラからの距離)
    ])

    # 5mm ボクセルグリッドでダウンサンプリング
    # 同じボクセルに入る点を 1 点に統合して点群を間引く
    voxels = np.floor(pts / 0.005).astype(np.int32)
    _, idx = np.unique(voxels, axis=0, return_index=True)
    return pts[idx]


def initialize_state(X: np.ndarray) -> TrackdloState:
    """
    初回フレーム: 点群 X から TrackdloState を構築して返す。

    処理の流れ:
      1. X の両端を結ぶ等間隔の初期ノード Y_init を作る
      2. cpd_lle() で Y_init を点群 X にフィッティング
      3. sort_pts() で最近傍順に並べ直す
      4. 測地線座標 (各ノード間の累積弧長) を計算してセット
    """
    state  = TrackdloState(NUM_NODES)

    # np.linspace(a, b, N): a から b まで N 個の等間隔点を生成する
    # axis=0 なので各行が [x,y,z] になる (M×3)
    Y_init = np.linspace(X.min(axis=0), X.max(axis=0), NUM_NODES)

    Y_fit, sigma2, converged = cpd_lle(
        X, Y_init, sigma2=0.0,
        beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05,
    )
    print(f"  [cpd_lle] converged={converged}  sigma2={sigma2:.6f}")

    Y_sorted = sort_pts(Y_fit)

    state.Y      = Y_sorted
    state.sigma2 = sigma2

    # 測地線座標: ノード 0 から各ノードまでの累積経路長 [m]
    # np.diff(Y, axis=0): 隣接ノード間の差分ベクトル (M-1 × 3)
    # np.linalg.norm(..., axis=1): 各差分ベクトルの長さ (M-1,)
    # np.cumsum: 累積和
    dists = np.linalg.norm(np.diff(Y_sorted, axis=0), axis=1)
    state.geodesic_coord = np.concatenate([[0.0], np.cumsum(dists)])

    state.guide_nodes = Y_sorted
    return state


def draw_overlay(bgr: np.ndarray, mask: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    カラーフレームに以下を重ねて描画する:
      - マスク領域を半透明の緑でオーバーレイ
      - 各ノードを赤い円でプロット
      - 隣接ノード間を黄色い線で結ぶ

    3D 座標 (X,Y,Z) をカメラ平面に投影する:
      u = X * fx / Z + cx
      v = Y * fy / Z + cy
    """
    overlay = bgr.copy()
    overlay[mask == 1] = (0, 180, 0)
    vis = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)

    prev_px = None
    for x3, y3, z3 in Y:
        if z3 <= 0:
            continue
        # 3D → 2D 投影 (ピンホールカメラモデル)
        px = int(x3 * FX / z3 + CX)
        py = int(y3 * FY / z3 + CY)
        cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)   # 赤い円
        if prev_px is not None:
            cv2.line(vis, prev_px, (px, py), (0, 220, 255), 2)  # 黄色い線
        prev_px = (px, py)

    return vis


def print_nodes(frame_no: int, Y: np.ndarray) -> None:
    """全ノード座標をターミナルに出力する"""
    print(f"\n[frame {frame_no:04d}]  nodes ({len(Y)})")
    for i, (x, y, z) in enumerate(Y):
        print(f"  node {i:02d}:  X={x:+.4f}  Y={y:+.4f}  Z={z:.4f}  [m]")


# =============================================================================
# メインループ
# =============================================================================
all_idx  = list(range(NUM_NODES))
state    = None    # None の間は初期化待ち
frame_no = 0
WIN      = "pipeline_demo  [Space: reinit  ESC: quit]"

print("\n[INFO] Space: 再初期化  ESC: 終了\n")

try:
    while True:
        # Step 1: カラー + アライン済み深度フレームを取得
        frames         = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        bgr      = np.asanyarray(color_frame.get_data())   # (480,640,3) uint8
        depth_mm = np.asanyarray(depth_frame.get_data())   # (480,640) uint16 [mm]

        # Step 2: DeepLabV3+ でバイナリマスクを生成
        mask = infer_mask(bgr)

        # Step 3: バイナリマスク + 深度画像 → 3D 点群
        X = mask_depth_to_pointcloud(mask, depth_mm)

        # Step 4 & 5: 初期化またはトラッキング
        # 点群が少なすぎる場合は処理をスキップ (ノード数の 3 倍を下限とする)
        if len(X) >= NUM_NODES * 3:
            X = np.ascontiguousarray(X, dtype=np.float64)

            if state is None:
                print("[INIT] initializing trackdlo...")
                state    = initialize_state(X)
                frame_no = 0
                print("[INIT] done")
            else:
                tracking_step(state, X, all_idx, all_idx, params)
                frame_no += 1

            # Step 6: ノード座標を出力
            # state.Y は呼ぶたびに C++ からコピーされるので、変数に受けておく
            Y_now = state.Y
            print_nodes(frame_no, Y_now)

        # ─── 可視化 ──────────────────────────────────────────────────────────
        if state is not None:
            vis = draw_overlay(bgr, mask, state.Y)
        else:
            vis = bgr.copy()
            cv2.putText(vis, "waiting for cable... (Space to init)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow(WIN, vis)
        cv2.imshow("mask", mask * 255)   # 0/1 → 0/255 に変換して表示

        key = cv2.waitKey(1) & 0xFF
        if key == 27:       # ESC
            break
        if key == ord(' '): # Space → 状態リセット
            if state is not None:
                state.close()
            state = None
            print("[RESET] state cleared")

finally:
    # state が残っていれば C++ メモリを解放する
    if state is not None:
        state.close()
    pipeline.stop()
    cv2.destroyAllWindows()
    print("done.")
