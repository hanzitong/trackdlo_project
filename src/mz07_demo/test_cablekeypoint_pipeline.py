"""
full_pipeline.py

RealSense D405 から取得した BGR/depth 画像をもとにケーブルのキーポイントを推定し、
中間キーポイントをロボットアームの目標値として出力するパイプライン。

pipeline:
  1. RealSense D405 から BGR + depth 画像を取得
  2. DeepLabV3+ でバイナリマスクを生成
  3. バイナリマスク + depth から 3D 点群を生成 (preprocessing C++)
  4. TrackDLO でケーブルキーポイントを推定 (trackdlo C++)
  5. 中間キーポイントを openNR 経由でロボットアームへ送る (未実装)

debug display (cv2.imshow):
  "BGR image"    : 生画像 + ノードオーバーレイ
  "depth image"  : depth をグレースケール可視化
  "mask overlay" : mask*255 を BGR に半透明オーバーレイ

CLI output:
  毎フレーム、中間キーポイント 1 点の座標を表示

key controls:
  Space  TrackDLO state をリセット
  ESC    終了

usage:
  cd src/mz07_demo && python full_pipeline.py
"""

import os
import sys
import platform
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent

# .so のパスを import より先に環境変数へセットする
os.environ["TRACKDLO_LIB_PATH"]      = str(HERE / "libtrackdlo_c.so")
os.environ["PREPROCESSING_LIB_PATH"] = str(HERE / "libpreprocessing_c.so")
# Windows の場合は以下に差し替える:
#   _ext = ".dll" if platform.system() == "Windows" else ".so"
#   os.environ["TRACKDLO_LIB_PATH"]      = str(HERE / f"libtrackdlo_c{_ext}")
#   os.environ["PREPROCESSING_LIB_PATH"] = str(HERE / f"libpreprocessing_c{_ext}")

sys.path.insert(0, str(HERE))

import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import segmentation_models_pytorch as smp

from trackdlo_cdll import (
    TrackdloState,
    TdloParams,
    default_params,
    tracking_step,
    cpd_lle,
    sort_pts,
)
from preprocessing_cdll import images_to_pointcloud, compute_visible_nodes

# import openNR  # proprietary library -- not implemented


# ─── 設定 ─────────────────────────────────────────────────────────────────────

WEIGHTS: Path  = HERE / "best_deeplabv3plus_cable.pth"
NUM_NODES: int = 15
DEVICE: str    = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet 正規化パラメータ (train_deeplabv3plus.py と必ず同じ値にすること)
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# カメラ内部パラメータはカメラ起動後に実機から取得して上書きする
FX: float = 0.0
FY: float = 0.0
CX: float = 0.0
CY: float = 0.0


# =============================================================================
# Step 2: DeepLabV3+ モデルのロード
# =============================================================================

print(f"device = {DEVICE}")
print(f"loading weights: {WEIGHTS.name} ...")

model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(DEVICE)

model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
print("model loaded")


# =============================================================================
# Step 1: RealSense D405 初期化
# =============================================================================

_pipeline: rs.pipeline = rs.pipeline()
_cfg: rs.config        = rs.config()
_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
_cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
_profile: rs.pipeline_profile = _pipeline.start(_cfg)

# カメラ内部パラメータを実機から取得
_intr = _profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
FX = _intr.fx
FY = _intr.fy
CX = _intr.ppx
CY = _intr.ppy
print(f"intrinsics: fx={FX:.1f} fy={FY:.1f} cx={CX:.1f} cy={CY:.1f}")

# depth をカラー座標系に合わせるアライナー
_align: rs.align = rs.align(rs.stream.color)

# depth scale: raw uint16 を mm 換算するための係数 (実機から取得)
# D405 は 0.0001 m/unit (0.1mm 単位)。raw × DEPTH_SCALE × 1000 = mm 値。
# C++ 側の根本修正については preprocessing.cpp の TODO コメントを参照。
_depth_sensor = _profile.get_device().first_depth_sensor()
DEPTH_SCALE: float = _depth_sensor.get_depth_scale()
print(f"depth scale: {DEPTH_SCALE:.6f} m/unit")

# 露出が安定するまでウォームアップ
for _ in range(5):
    _pipeline.wait_for_frames()
print("camera warm-up done")


# =============================================================================
# Step 4: TrackDLO パラメータ
# =============================================================================

_params: TdloParams = default_params()
_params.max_iter = 150
_params.tol      = 1e-4


# =============================================================================
# ヘルパー関数
# =============================================================================

def infer_mask(bgr: np.ndarray) -> np.ndarray:
    """BGR 画像からバイナリマスク (uint8, 0=background / 1=cable) を返す。

    前処理は train_deeplabv3plus.py の CableDataset.__getitem__ と完全に同一にすること。
    """
    rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD  # ImageNet 正規化
    x: torch.Tensor = torch.tensor(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob: np.ndarray = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def initialize_state(X: np.ndarray) -> "TrackdloState | None":
    """点群 X から TrackdloState を初期化して返す。失敗時は None。"""
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

    dists: np.ndarray = np.linalg.norm(np.diff(Y_sorted, axis=0), axis=1)
    state.geodesic_coord = np.concatenate([[0.0], np.cumsum(dists)])
    state.guide_nodes    = Y_sorted
    return state


def get_mid_node(Y: np.ndarray) -> np.ndarray:
    """ノード列 Y の中間インデックスの座標 (3,) を返す。"""
    return Y[len(Y) // 2]


def _project_pt(x3: float, y3: float, z3: float) -> "tuple[int, int] | None":
    """3D 点をピンホールモデルで 2D 画素座標 (px, py) に変換する。z3<=0 なら None。"""
    if not (np.isfinite(z3) and z3 > 0):
        return None
    return int(x3 * FX / z3 + CX), int(y3 * FY / z3 + CY)


def _draw_coord_axes(img: np.ndarray, origin: np.ndarray) -> None:
    """カメラ座標系の XY 軸を img に in-place で描画する。

    origin : (3,) カメラ座標での原点 [m]

    軸の長さは奥行き (origin[2]) の 10% に設定するため、
    画面上の矢印サイズが奥行きによらず約 FX*0.1 px に近くなる。

    色 (BGR):
      X (右方向) : 赤  (0, 0, 255)
      Y (下方向) : 緑  (0, 255, 0)
    """
    ox: float = float(origin[0])
    oy: float = float(origin[1])
    oz: float = float(origin[2])

    origin_px: "tuple[int, int] | None" = _project_pt(ox, oy, oz)
    if origin_px is None:
        return

    # 軸長 = 奥行きの 10% (画面上の見かけサイズが奥行きによらず一定になる)
    L: float = oz * 0.1

    axes: list[tuple[float, float, float, tuple[int, int, int], str]] = [
        (ox + L, oy, oz, (0,   0, 255), "X"),
        (ox, oy + L, oz, (0, 255,   0), "Y"),
    ]

    tx: float
    ty: float
    tz: float
    color: tuple[int, int, int]
    label: str
    for tx, ty, tz, color, label in axes:
        tip_px: "tuple[int, int] | None" = _project_pt(tx, ty, tz)
        if tip_px is None:
            continue
        cv2.arrowedLine(img, origin_px, tip_px, color, 2, tipLength=0.25)
        # ラベルをチップの少し外側に配置
        lx: int = int(tip_px[0] + (tip_px[0] - origin_px[0]) * 0.2)
        ly: int = int(tip_px[1] + (tip_px[1] - origin_px[1]) * 0.2)
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_debug(
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    Y: "np.ndarray | None",
) -> None:
    """デバッグ用 3 ウィンドウを cv2.imshow で更新する。

    windows:
      "BGR image"    : 生画像 + ノード円・接続線のオーバーレイ
      "depth image"  : depth をグレースケールで可視化
      "mask overlay" : mask*255 を BGR に半透明オーバーレイ + カメラ座標系の XYZ 軸
    """
    # ── BGR image + node overlay ────────────────────────────────────────────
    vis_bgr: np.ndarray = bgr.copy()
    if Y is not None:
        prev_px: "tuple[int, int] | None" = None
        x3: float
        y3: float
        z3: float
        for x3, y3, z3 in Y:
            if not (np.isfinite(x3) and np.isfinite(y3) and np.isfinite(z3) and z3 > 0):
                continue
            px: int = int(x3 * FX / z3 + CX)
            py: int = int(y3 * FY / z3 + CY)
            cv2.circle(vis_bgr, (px, py), 5, (0, 0, 255), -1)
            if prev_px is not None:
                cv2.line(vis_bgr, prev_px, (px, py), (0, 220, 255), 2)
            prev_px = (px, py)
    cv2.imshow("BGR image", vis_bgr)

    # ── depth image (グレースケール可視化) ────────────────────────────────────
    # depth_mm は mm 換算済み値。10000mm (10m) でクランプして 0-255 にスケール。
    depth_vis: np.ndarray = np.clip(depth_mm, 0, 10000).astype(np.float32)
    depth_vis = (depth_vis / 10000.0 * 255.0).astype(np.uint8)
    cv2.imshow("depth image", depth_vis)

    # ── mask overlay + カメラ座標系 XYZ 軸 ────────────────────────────────────
    overlay: np.ndarray      = bgr.copy()
    overlay[mask == 1]       = (255, 255, 255)
    mask_overlay: np.ndarray = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)

    # 軸の原点: カメラ前方 0.3m の固定点 (画像中央付近に投影される)
    # ノード座標を使うと TrackDLO の推定に連動して動いてしまうため固定する
    _draw_coord_axes(mask_overlay, np.array([0.0, 0.0, 0.3]))

    cv2.imshow("mask overlay", mask_overlay)


# =============================================================================
# メインループ
# =============================================================================

def main() -> None:
    """パイプライン全体のエントリポイント。"""
    state: "TrackdloState | None" = None
    frame_no: int = 0

    print("\nrunning  [Space: reset  ESC: quit]\n")

    try:
        while True:
            # ── Step 1: フレーム取得 ──────────────────────────────────────────
            frames: rs.composite_frame         = _pipeline.wait_for_frames()
            aligned: rs.composite_frame        = _align.process(frames)
            color_frame: rs.video_frame        = aligned.get_color_frame()
            depth_frame: rs.depth_frame        = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr: np.ndarray       = np.asanyarray(color_frame.get_data())   # (480,640,3) uint8
            depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())  # (480,640) uint16 raw

            # raw → mm 換算 (以降のパイプラインは depth_mm = uint16 mm 単位で統一)
            depth_mm: np.ndarray = np.clip(
                depth_raw.astype(np.float32) * DEPTH_SCALE * 1000.0, 0, 65535
            ).astype(np.uint16)

            # ── Step 2: バイナリマスク生成 ────────────────────────────────────
            mask: np.ndarray = infer_mask(bgr)

            # ── Step 3: 3D 点群生成 ───────────────────────────────────────────
            X: np.ndarray = images_to_pointcloud(
                bgr, depth_mm, mask, FX, FY, CX, CY, leaf_size=0.005
            )

            if len(X) >= NUM_NODES * 3:
                X = np.ascontiguousarray(X, dtype=np.float64)

                # ── Step 4: TrackDLO 初期化 or トラッキング ───────────────────
                if state is None:
                    print("[INIT] initializing TrackDLO...")
                    state = initialize_state(X)
                    if state is None:
                        print("[INIT] failed, retry on next frame")
                    else:
                        frame_no = 0
                        print("[INIT] done")
                else:
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
                    tracking_step(state, X, vn, vne, _params)
                    frame_no += 1

                    if not np.all(np.isfinite(state.Y)):
                        print("[WARN] NaN detected in state.Y, resetting")
                        state.close()
                        state = None

            # ── Step 5: openNR でロボットアームへ送る (未実装) ──────────────────
            # mid_node = get_mid_node(state.Y)  # target keypoint (3,) [m]
            # openNR.set_target(mid_node)
            # -- openNR API が確定したらここに実装する --

            # ── CLI 出力 ──────────────────────────────────────────────────────
            if state is not None:
                mid: np.ndarray = get_mid_node(state.Y)
                print(
                    f"[frame {frame_no:04d}] mid node:"
                    f"  X={mid[0]:+.4f}  Y={mid[1]:+.4f}  Z={mid[2]:.4f}  [m]",
                    end="\r",
                )

            # ── デバッグ表示 ──────────────────────────────────────────────────
            draw_debug(bgr, depth_mm, mask, state.Y if state is not None else None)

            # ── キー入力 ──────────────────────────────────────────────────────
            key: int = cv2.waitKey(1) & 0xFF
            if key == 27:           # ESC
                break
            if key == ord(" "):     # Space: リセット
                if state is not None:
                    state.close()
                state = None
                frame_no = 0
                print("\n[RESET] state cleared")

    finally:
        if state is not None:
            state.close()
        _pipeline.stop()
        cv2.destroyAllWindows()
        print("\ndone.")


if __name__ == "__main__":
    main()
