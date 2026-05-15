"""
trackdlo 動作確認スクリプト (RealSense 版)

1280×960 の単一ウィンドウに 4 画面を表示する:

  ┌──────────────────────┬──────────────────────┐
  │ [A] カラー           │ [B] 深度 (JET)       │
  │     マスク + ノード  │     マスク + ノード  │
  ├──────────────────────┼──────────────────────┤
  │ [C] 上面図 (X-Z)     │ [D] 側面図 (Z-Y)     │
  │     点群 + ノード    │     点群 + ノード    │
  └──────────────────────┴──────────────────────┘

  [C] X→ / Z↑  (カメラの左右 vs 奥行き)
  [D] Z→ / Y↑  (奥行き vs 高さ)

キー操作:
  Space 初期化 (ケーブルが写っているときに押す)
  r     リセット (再初期化が必要になったら Space を再度押す)
  ESC   終了

実行:
  uv run python src/trackdlo_python/scripts/verify_trackdlo.py
"""

from pathlib import Path
from typing import Optional

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
NUM_NODES = 15       # trackdlo のノード数
VOXEL_M   = 0.010    # voxel grid のサイズ [m]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# 1 画面のサイズ (4 枚並べると 1280×960 になる)
VIEW_W, VIEW_H = 640, 480

# =============================================================================
# RealSense セットアップ
# =============================================================================
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
profile  = pipeline.start(cfg)
align    = rs.align(rs.stream.color)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
FX, FY = intr.fx, intr.fy
CX, CY = intr.ppx, intr.ppy
print(f"[RealSense] fx={FX:.2f}  fy={FY:.2f}  cx={CX:.2f}  cy={CY:.2f}")

# =============================================================================
# DeepLab セットアップ
# =============================================================================
model = smp.DeepLabV3Plus(
    encoder_name="resnet34", encoder_weights=None,
    in_channels=3, classes=1, activation=None,
).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
print(f"[DeepLab]   loaded  device={DEVICE}")

# =============================================================================
# trackdlo パラメータ
# =============================================================================
params = tdlo.TrackdloParams()
# 必要に応じて調整:
# params.beta    = 5.0
# params.lambda_ = 1.0
# params.mu      = 0.05

# =============================================================================
# 処理関数
# =============================================================================

def infer_mask(frame: np.ndarray) -> np.ndarray:
    """BGR フレーム → バイナリマスク (uint8, 0 or 1)"""
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.tensor(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return (torch.sigmoid(model(x)) > 0.5)[0, 0].cpu().numpy().astype(np.uint8)


def mask_to_pointcloud(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """バイナリマスク + 深度画像 → 点群 (N×3, float64, メートル)"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0, 3))

    z_mm  = depth[ys, xs].astype(np.float64)
    valid = z_mm > 0
    ys, xs, z_mm = ys[valid], xs[valid], z_mm[valid]
    if len(xs) == 0:
        return np.empty((0, 3))

    Z   = z_mm / 1000.0
    pts = np.column_stack([
        (xs - CX) * Z / FX,
        (ys - CY) * Z / FY,
        Z,
    ])

    voxels = np.floor(pts / VOXEL_M).astype(np.int32)
    _, idx = np.unique(voxels, axis=0, return_index=True)
    return pts[idx]


def init_state(X: np.ndarray) -> tdlo.TrackdloState:
    """trackdlo の状態を初期化する"""
    state  = tdlo.make_trackdlo_state(NUM_NODES)
    Y_init = np.linspace(X.min(axis=0), X.max(axis=0), NUM_NODES)

    # reg() は Y_init を無視して原点から初期化するため全ノードが同一点に収束しやすく、
    # geodesic_coord が全ゼロになって traverse_euclidean が空を返す → out-of-bounds crash。
    # cpd_lle は Y_init を出発点として使い、LLE 制約でノード間隔を保つため安全。
    Y_fit, sig2, _ = tdlo.cpd_lle(np.ascontiguousarray(X), Y_init, 0.0,
                                   beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05)

    state.Y      = tdlo.sort_pts(Y_fit)
    state.sigma2 = sig2

    dists                = np.linalg.norm(np.diff(state.Y, axis=0), axis=1)
    state.geodesic_coord = [0.0] + list(np.cumsum(dists))
    state.guide_nodes    = state.Y.copy()
    return state


def process_frame(
    frame: np.ndarray,
    depth: np.ndarray,
    state: Optional[tdlo.TrackdloState],
    frame_no: int,
    all_idx: list,
) -> tuple:
    """
    1 フレームの処理パイプライン。

    DeepLab → マスク → 点群 → trackdlo (初期化 or トラッキング)

    Returns:
        mask    : バイナリマスク (H×W, uint8)
        X       : 点群 (N×3, float64)
        state   : 更新済み TrackdloState (None のままの場合あり)
        Y_disp  : 表示用ノード座標 (M×3)
        info    : 表示用ステータス辞書
    """
    mask = infer_mask(frame)
    X    = mask_to_pointcloud(mask, depth)

    if len(X) < NUM_NODES:
        Y_disp = state.Y if state is not None else np.zeros((NUM_NODES, 3))
        info   = dict(
            status=f"Waiting ({len(X)} pts)",
            n_pts=len(X),
            sigma2=state.sigma2 if state is not None else 0.0,
            frame=frame_no,
        )
        return mask, X, state, Y_disp, info

    X = np.ascontiguousarray(X)

    if state is None:
        # Space キー待ち。init_state はメインループで呼ぶ。
        info = dict(status="Press SPACE to initialize", n_pts=len(X),
                    sigma2=0.0, frame=frame_no)
        return mask, X, state, np.zeros((NUM_NODES, 3)), info

    tdlo.tracking_step(state, X, all_idx, all_idx, params)
    info = dict(status="Tracking", n_pts=len(X), sigma2=state.sigma2, frame=frame_no)
    return mask, X, state, state.Y, info


# =============================================================================
# 表示関数
# =============================================================================

def _project(Y: np.ndarray):
    """ノード座標 (M×3) → ピクセル座標 (M×2) と有効フラグ"""
    Z     = Y[:, 2]
    valid = Z > 0
    px    = np.zeros((len(Y), 2), dtype=np.int32)
    px[valid, 0] = (Y[valid, 0] * FX / Z[valid] + CX).astype(np.int32)
    px[valid, 1] = (Y[valid, 1] * FY / Z[valid] + CY).astype(np.int32)
    return px, valid


def _draw_nodes(img: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """ノードをエッジつきで画像上に描画する"""
    out  = img.copy()
    px, valid = _project(Y)
    prev = None
    for p, v in zip(px, valid):
        if not v:
            prev = None
            continue
        pt = (int(p[0]), int(p[1]))
        if prev is not None:
            cv2.line(out, prev, pt, (0, 220, 255), 2)
        cv2.circle(out, pt, 7, (0, 0, 220), -1)
        cv2.circle(out, pt, 7, (255, 255, 255), 1)
        prev = pt
    return out


def view_A_color(frame: np.ndarray, mask: np.ndarray,
                 Y: np.ndarray, info: dict) -> np.ndarray:
    """[A] カラー + マスクオーバーレイ + ノード"""
    ov  = frame.copy()
    ov[mask == 1] = (0, 200, 0)
    vis = cv2.addWeighted(frame, 0.55, ov, 0.45, 0)
    vis = _draw_nodes(vis, Y)
    lines = [
        f"[A] Color+Mask+Nodes   {info['status']}",
        f"    pts={info['n_pts']}  sigma2={info['sigma2']:.4e}  frame={info['frame']}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (8, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def view_B_depth(depth: np.ndarray, mask: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """[B] 深度画像 (JET カラーマップ) + ノード"""
    d_masked        = depth.astype(np.float32)
    d_masked[mask == 0] = 0
    d8              = np.clip(d_masked / 3000.0 * 255, 0, 255).astype(np.uint8)
    vis             = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    vis[mask == 0]  = (30, 30, 30)
    vis             = _draw_nodes(vis, Y)
    cv2.putText(vis, "[B] Depth (JET, 0-3000mm)", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def view_scatter(X: np.ndarray, Y: np.ndarray,
                 label: str, h_axis: int, v_axis: int) -> np.ndarray:
    """[C][D] 点群の 2D 散布図 (h_axis/v_axis: 0=X 1=Y 2=Z)"""
    AXIS   = ["X", "Y", "Z"]
    canvas = np.full((VIEW_H, VIEW_W, 3), 20, dtype=np.uint8)

    if len(X) == 0:
        cv2.putText(canvas, "no points", (20, VIEW_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
        return canvas

    span     = 0.30
    c_h      = float(X[:, h_axis].mean())
    c_v      = float(X[:, v_axis].mean())
    h_lo, h_hi = c_h - span, c_h + span
    v_lo, v_hi = c_v - span, c_v + span

    def px_h(v):
        return int((v - h_lo) / (h_hi - h_lo) * (VIEW_W - 1))

    def px_v(v):
        return int((1.0 - (v - v_lo) / (v_hi - v_lo)) * (VIEW_H - 1))

    for tick in np.arange(np.ceil(h_lo * 10) / 10, h_hi, 0.1):
        tx = px_h(tick)
        if 0 <= tx < VIEW_W:
            cv2.line(canvas, (tx, 0), (tx, VIEW_H), (45, 45, 45), 1)
    for tick in np.arange(np.ceil(v_lo * 10) / 10, v_hi, 0.1):
        ty = px_v(tick)
        if 0 <= ty < VIEW_H:
            cv2.line(canvas, (0, ty), (VIEW_W, ty), (45, 45, 45), 1)

    step = max(1, len(X) // 3000)
    for pt in X[::step]:
        ph, pv = px_h(pt[h_axis]), px_v(pt[v_axis])
        if 0 <= ph < VIEW_W and 0 <= pv < VIEW_H:
            canvas[pv, ph] = (100, 100, 100)

    prev = None
    for n in Y:
        ph, pv = px_h(n[h_axis]), px_v(n[v_axis])
        in_bounds = (0 <= ph < VIEW_W) and (0 <= pv < VIEW_H)
        if in_bounds:
            if prev is not None:
                cv2.line(canvas, prev, (ph, pv), (0, 220, 255), 2)
            cv2.circle(canvas, (ph, pv), 6, (0, 0, 220), -1)
            cv2.circle(canvas, (ph, pv), 6, (255, 255, 255), 1)
        prev = (ph, pv) if in_bounds else None

    cv2.putText(canvas,
                f"{label}  {AXIS[h_axis]}→ / {AXIS[v_axis]}↑  (±{span:.2f}m)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas,
                f"center: {AXIS[h_axis]}={c_h:.3f}m  {AXIS[v_axis]}={c_v:.3f}m",
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return canvas


def build_display(frame: np.ndarray, depth: np.ndarray,
                  mask: np.ndarray, X: np.ndarray,
                  Y_disp: np.ndarray, info: dict) -> np.ndarray:
    """
    4 つのビューを 1280×960 の combined 画像に組み立てる。

      上段: [A] カラー+マスク+ノード  |  [B] 深度 (JET)
      下段: [C] 上面図 (X-Z)          |  [D] 側面図 (Z-Y)
    """
    vA = view_A_color(frame, mask, Y_disp, info)
    vB = view_B_depth(depth, mask, Y_disp)
    vC = view_scatter(X, Y_disp, "[C] Top view",  h_axis=0, v_axis=2)
    vD = view_scatter(X, Y_disp, "[D] Side view", h_axis=2, v_axis=1)
    return np.vstack([np.hstack([vA, vB]), np.hstack([vC, vD])])


# =============================================================================
# メインループ
# =============================================================================
state    = None
all_idx  = list(range(NUM_NODES))
frame_no = 0

print("\nキー操作: SPACE=初期化  r=リセット  ESC=終了\n")

try:
    while True:
        # ---- フレーム取得 ----
        frames         = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame    = aligned_frames.get_color_frame()
        depth_frame    = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame    = np.asanyarray(color_frame.get_data())
        depth    = np.asanyarray(depth_frame.get_data())
        frame_no += 1

        # ---- 処理 ----
        mask, X, state, Y_disp, info = process_frame(frame, depth, state, frame_no, all_idx)

        # ---- 表示 ----
        cv2.imshow("trackdlo verification", build_display(frame, depth, mask, X, Y_disp, info))

        # ---- キー入力 ----
        key = cv2.waitKey(1) & 0xFF
        if key == 27:               # ESC: 終了
            break
        elif key == ord(' '):       # Space: 初期化
            if len(X) >= NUM_NODES:
                state = init_state(X)
                print(f"[frame {frame_no}] initialized  N={len(X)}")
            else:
                print(f"[frame {frame_no}] not enough points ({len(X)})")
        elif key == ord('r'):       # r: リセット
            state = None
            print(f"[frame {frame_no}] reset")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
