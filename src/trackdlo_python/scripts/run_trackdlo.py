"""
trackdlo パイプラインのデモ (RealSense 版)

  RealSense (カラー + 深度)
    → DeepLab でバイナリマスク生成
    → マスク + 深度画像 → 点群 (N×3) に変換
    → trackdlo でケーブルキーポイントを追跡

[実行]
  uv run python src/trackdlo_python/scripts/run_trackdlo.py
  ESC で終了。
"""

from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import segmentation_models_pytorch as smp
import trackdlo_python as tdlo

# =============================================================================
# 設定
# =============================================================================
WEIGHTS   = Path(__file__).resolve().parents[2] / "bmask_gen/weights/best_deeplabv3plus_cable.pth"
NUM_NODES = 15    # trackdlo のノード数
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# RealSense パイプラインの初期化
# =============================================================================
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile  = pipeline.start(config)

# 深度フレームをカラーフレームに位置合わせするオブジェクト
align = rs.align(rs.stream.color)

# カメラ内部パラメータをカメラ自身から取得する
# (FX, FY, CX, CY をハードコードする必要がない)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
FX, FY = intr.fx, intr.fy
CX, CY = intr.ppx, intr.ppy
print(f"intrinsics: fx={FX:.1f}  fy={FY:.1f}  cx={CX:.1f}  cy={CY:.1f}")

# =============================================================================
# DeepLab モデルのロード
# =============================================================================
model = smp.DeepLabV3Plus(
    encoder_name="resnet34", encoder_weights=None,
    in_channels=3, classes=1, activation=None,
).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
print(f"model loaded  device={DEVICE}")

# =============================================================================
# trackdlo パラメータ
# =============================================================================
params = tdlo.TrackdloParams()
# 必要に応じて調整:
# params.beta    = 5.0   # CPD の平滑化強度
# params.lambda_ = 1.0   # LLE 制約の重み
# params.mu      = 0.05  # 外れ値確率

# =============================================================================
# ヘルパー関数
# =============================================================================

def infer_mask(frame: np.ndarray) -> np.ndarray:
    """BGR フレーム → バイナリマスク (uint8, 0 or 1)"""
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.tensor(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return (torch.sigmoid(model(x)) > 0.5)[0, 0].cpu().numpy().astype(np.uint8)


def mask_to_pointcloud(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    バイナリマスク + 深度画像 → 点群 (N×3, float64)

    ピンホールカメラの逆投影:
        Z = depth[v, u] / 1000.0  [m]
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    depth == 0 の画素 (深度未計測) は除外する。
    5mm voxel grid でダウンサンプルして点数を減らす。
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0, 3))

    z_mm  = depth[ys, xs].astype(np.float64)
    valid = z_mm > 0          # 深度未計測画素を除外
    ys, xs, z_mm = ys[valid], xs[valid], z_mm[valid]
    if len(xs) == 0:
        return np.empty((0, 3))

    Z   = z_mm / 1000.0
    pts = np.column_stack([
        (xs - CX) * Z / FX,
        (ys - CY) * Z / FY,
        Z,
    ])

    # 5mm voxel grid ダウンサンプリング
    voxels = np.floor(pts / 0.005).astype(np.int32)
    _, idx = np.unique(voxels, axis=0, return_index=True)
    return pts[idx]


def init_state(X: np.ndarray) -> tdlo.TrackdloState:
    """
    最初のフレームで trackdlo の状態を初期化する。

    1. 点群の端点を結ぶ直線上に初期ノードを配置
    2. cpd_lle() で点群にフィット (reg() は Y_init を無視するため使わない)
    3. sort_pts() でノードをケーブル沿いに順序付け
    4. geodesic_coord (累積弧長) を計算  ← tracking_step の前に必須
    """
    state  = tdlo.make_trackdlo_state(NUM_NODES)
    Y_init = np.linspace(X.min(axis=0), X.max(axis=0), NUM_NODES)

    # reg() は内部で Y_init を無視して原点から初期化するため、全ノードが
    # 同一点に収束しやすく geodesic_coord が全ゼロになる。
    # cpd_lle は Y_init を出発点として LLE 制約でノード間隔を維持する。
    X_cont        = np.ascontiguousarray(X)
    Y_fit, sigma2, _ = tdlo.cpd_lle(X_cont, Y_init, 0.0,
                                     beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05)

    state.Y      = tdlo.sort_pts(Y_fit)
    state.sigma2 = sigma2

    dists = np.linalg.norm(np.diff(state.Y, axis=0), axis=1)
    state.geodesic_coord = [0.0] + list(np.cumsum(dists))

    state.guide_nodes = state.Y.copy()
    return state


def draw_result(frame: np.ndarray, mask: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """マスクオーバーレイとノード・エッジを描画する"""
    overlay = frame.copy()
    overlay[mask == 1] = (0, 180, 0)
    vis = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    prev_px = None
    for x3, y3, z3 in Y:
        if z3 <= 0:
            continue
        px = int(x3 * FX / z3 + CX)
        py = int(y3 * FY / z3 + CY)
        cv2.circle(vis, (px, py), 6, (0, 0, 255), -1)
        if prev_px is not None:
            cv2.line(vis, prev_px, (px, py), (0, 220, 255), 2)
        prev_px = (px, py)

    return vis


# =============================================================================
# メインループ
# =============================================================================
state   = None
all_idx = list(range(NUM_NODES))  # 全ノードを可視として扱う (occlusion なし)

try:
    while True:
        # Step 1: RealSense からカラーと深度を取得
        frames         = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)   # 深度をカラーに位置合わせ

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())   # BGR, uint8
        depth = np.asanyarray(depth_frame.get_data())   # uint16, mm

        # Step 2: DeepLab でバイナリマスク生成
        mask = infer_mask(frame)

        # Step 3: マスク + 深度 → 点群
        X = mask_to_pointcloud(mask, depth)

        # 点数が足りない場合はスキップ
        if len(X) < NUM_NODES:
            cv2.imshow("trackdlo demo", frame)
            cv2.imshow("mask", mask * 255)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        X = np.ascontiguousarray(X)

        # Step 4: 初回のみ状態を初期化、2フレーム目以降はトラッキング
        if state is None:
            state = init_state(X)
        else:
            tdlo.tracking_step(state, X, all_idx, all_idx, params)

        # Step 5: 結果を描画
        vis = draw_result(frame, mask, state.Y)
        cv2.imshow("trackdlo demo", vis)
        cv2.imshow("mask", mask * 255)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
