"""
test_cablekeypoint_pipeline.py

RealSense D405 からリアルタイムで BGR/depth を取得し、
DeepLabV3+ + TrackDLO でケーブルキーポイントを連続推定するテストスクリプト。

debug display:
  "BGR image"    : 生画像 + ノードオーバーレイ
  "depth image"  : depth をグレースケール可視化
  "mask overlay" : mask*255 を BGR に半透明オーバーレイ

key controls:
  Space  TrackDLO state をリセット
  ESC    終了

usage:
  cd src/mz07_demo && python test/test_cablekeypoint_pipeline.py
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


@dataclass
class CameraIntrinsics:
    """カメラ内部パラメータ (ピンホールモデル)。"""
    fx: float
    fy: float
    cx: float
    cy: float


def load_model(weights: Path, device: str) -> smp.DeepLabV3Plus:
    """DeepLabV3+ の重みをロードして返す。"""
    model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model


def init_realsense() -> "tuple[CameraIntrinsics, rs.pipeline, rs.align, float]":
    """RealSense D405 を初期化し、内部パラメータ・パイプライン・depth scale を返す。"""
    pipeline: rs.pipeline        = rs.pipeline()
    cfg: rs.config               = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    profile: rs.pipeline_profile = pipeline.start(cfg)

    raw_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr: CameraIntrinsics = CameraIntrinsics(
        fx=raw_intr.fx, fy=raw_intr.fy, cx=raw_intr.ppx, cy=raw_intr.ppy,
    )
    print(f"intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")

    align: rs.align    = rs.align(rs.stream.color)
    depth_scale: float = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"depth scale: {depth_scale:.6f} m/unit")

    for _ in range(5):
        pipeline.wait_for_frames()
    print("camera warm-up done")

    return intr, pipeline, align, depth_scale


def infer_mask(bgr: np.ndarray, model: smp.DeepLabV3Plus, device: str) -> np.ndarray:
    """BGR 画像からバイナリマスク (uint8, 0=background / 1=cable) を返す。"""
    imagenet_mean: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std: np.ndarray  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - imagenet_mean) / imagenet_std
    x: torch.Tensor = torch.tensor(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        prob: np.ndarray = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


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


def get_mid_node(Y: np.ndarray) -> np.ndarray:
    """ノード列 Y の中間インデックスの座標 (3,) を返す。"""
    return Y[len(Y) // 2]


def _project_pt(
    x3: float, y3: float, z3: float,
    intr: CameraIntrinsics,
) -> "tuple[int, int] | None":
    """3D 点をピンホールモデルで 2D 画素座標に変換する。z3<=0 なら None。"""
    if not (np.isfinite(z3) and z3 > 0):
        return None
    return int(x3 * intr.fx / z3 + intr.cx), int(y3 * intr.fy / z3 + intr.cy)


def _draw_coord_axes(img: np.ndarray, origin: np.ndarray, intr: CameraIntrinsics) -> None:
    """カメラ座標系の XY 軸を img に in-place で描画する。"""
    ox: float = float(origin[0])
    oy: float = float(origin[1])
    oz: float = float(origin[2])

    origin_px: "tuple[int, int] | None" = _project_pt(ox, oy, oz, intr)
    if origin_px is None:
        return

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
        tip_px: "tuple[int, int] | None" = _project_pt(tx, ty, tz, intr)
        if tip_px is None:
            continue
        cv2.arrowedLine(img, origin_px, tip_px, color, 2, tipLength=0.25)
        lx: int = int(tip_px[0] + (tip_px[0] - origin_px[0]) * 0.2)
        ly: int = int(tip_px[1] + (tip_px[1] - origin_px[1]) * 0.2)
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_debug(
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    Y: "np.ndarray | None",
    intr: CameraIntrinsics,
) -> None:
    """デバッグ用 3 ウィンドウを cv2.imshow で更新する。"""
    vis_bgr: np.ndarray = bgr.copy()
    if Y is not None:
        prev_px: "tuple[int, int] | None" = None
        h: int
        w: int
        h, w = vis_bgr.shape[:2]
        x3: float
        y3: float
        z3: float
        for x3, y3, z3 in Y:
            if not (np.isfinite(x3) and np.isfinite(y3) and np.isfinite(z3) and z3 > 0):
                continue
            px: int = int(x3 * intr.fx / z3 + intr.cx)
            py: int = int(y3 * intr.fy / z3 + intr.cy)
            if not (0 <= px < w and 0 <= py < h):
                prev_px = None
                continue
            cv2.circle(vis_bgr, (px, py), 5, (0, 0, 255), -1)
            if prev_px is not None:
                cv2.line(vis_bgr, prev_px, (px, py), (0, 220, 255), 2)
            prev_px = (px, py)
    cv2.imshow("BGR image", vis_bgr)

    # depth_mm を 10m でクランプして 0-255 にスケール
    depth_vis: np.ndarray = np.clip(depth_mm, 0, 10000).astype(np.float32)
    depth_vis = (depth_vis / 10000.0 * 255.0).astype(np.uint8)
    cv2.imshow("depth image", depth_vis)

    overlay: np.ndarray      = bgr.copy()
    overlay[mask == 1]       = (255, 255, 255)
    mask_overlay: np.ndarray = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
    _draw_coord_axes(mask_overlay, np.array([0.0, 0.0, 0.3]), intr)
    cv2.imshow("mask overlay", mask_overlay)


def main() -> None:
    """パイプライン全体のエントリポイント。"""
    weights: Path  = HERE.parent / "best_deeplabv3plus_cable.pth"
    num_nodes: int = 15
    device: str    = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device = {device}")
    model: smp.DeepLabV3Plus = load_model(weights, device)
    print("model loaded")

    intr: CameraIntrinsics
    pipeline: rs.pipeline
    align: rs.align
    depth_scale: float
    intr, pipeline, align, depth_scale = init_realsense()

    params: TdloParams = default_params()
    params.max_iter = 150
    params.tol      = 1e-4

    state: "TrackdloState | None" = None
    frame_no: int = 0

    print("\nrunning  [Space: reset  ESC: quit]\n")

    try:
        while True:
            frames_rs: rs.composite_frame  = pipeline.wait_for_frames()
            aligned: rs.composite_frame    = align.process(frames_rs)
            color_frame: rs.video_frame    = aligned.get_color_frame()
            depth_frame: rs.depth_frame    = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr: np.ndarray       = np.asanyarray(color_frame.get_data())
            depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())
            depth_mm: np.ndarray  = np.clip(
                depth_raw.astype(np.float32) * depth_scale * 1000.0, 0, 65535
            ).astype(np.uint16)

            mask: np.ndarray = infer_mask(bgr, model, device)

            X: np.ndarray = images_to_pointcloud(
                bgr, depth_mm, mask, intr.fx, intr.fy, intr.cx, intr.cy, leaf_size=0.005
            )

            if len(X) >= num_nodes * 3:
                X = np.ascontiguousarray(X, dtype=np.float64)

                if state is None:
                    print("[INIT] initializing TrackDLO...")
                    state = initialize_state(X, num_nodes)
                    if state is None:
                        print("[INIT] failed, retry on next frame")
                    else:
                        frame_no = 0
                        print("[INIT] done")
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
                        print("[WARN] NaN detected in state.Y, resetting")
                        state.close()
                        state = None

            if state is not None:
                mid: np.ndarray = get_mid_node(state.Y)
                print(
                    f"[frame {frame_no:04d}] mid node:"
                    f"  X={mid[0]:+.4f}  Y={mid[1]:+.4f}  Z={mid[2]:.4f}  [m]",
                    end="\r",
                )

            draw_debug(bgr, depth_mm, mask, state.Y if state is not None else None, intr)

            key: int = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(" "):
                if state is not None:
                    state.close()
                state = None
                frame_no = 0
                print("\n[RESET] state cleared")

    finally:
        if state is not None:
            state.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\ndone.")


if __name__ == "__main__":
    main()
