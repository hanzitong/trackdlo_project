
"""
demo_full_pipeline_0624.py

RealSense D405 から取得した BGR/depth 画像をもとにケーブルのキーポイントを推定し、
中間キーポイントを OpenNR 経由でロボットアームへ送るフルパイプラインデモ。

pipeline:
  1. RealSense D405 から BGR + depth 画像を取得
  2. DeepLabV3+ でバイナリマスクを生成
  3. バイナリマスク + depth から 3D 点群を生成 (Python 実装)
  4. TrackDLO でケーブルキーポイントを推定 (trackdlo C++)
  5. 中間キーポイントを OpenNR 経由でロボットアームへ送る

key controls:
  Space  リトライ
  ESC    終了

usage:
  cd src/mz07_demo && python demo_full_pipeline_0624.py
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import platform


# import より前に .dll のパスを環境変数へセットする必要がある
HERE: Path = Path(__file__).resolve().parent.parent
if platform.system() == "Linux":
    os.environ["TRACKDLO_LIB_PATH"]      = str(HERE / "lib" / "libtrackdlo_c.so")
    os.environ["PREPROCESSING_LIB_PATH"] = str(HERE / "lib" / "libpreprocessing_c.so")
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "op"))
elif platform.system() == "Windows":
    os.environ["TRACKDLO_LIB_PATH"]      = str(HERE / "mz07_demo_0624" / "lib" / "libtrackdlo_c.dll")
    os.environ["PREPROCESSING_LIB_PATH"] = str(HERE / "mz07_demo_0624" / "lib" / "libpreprocessing_c.dll")
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "opennr_test" / ".venv" / "Lib" / "site-packages"))


import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import segmentation_models_pytorch as smp
from opennr_py import NR_POSE, global_nr  # type: ignore[import]

from trackdlo_cdll import (
    TrackdloState,
    cpd_lle,
    sort_pts,
)


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class CameraIntrinsics:
    """カメラ内部パラメータ (ピンホールモデル)。"""
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class RobotConfig:
    """OpenNR ロボット接続設定・ホーム位置・安全動作範囲。"""
    dll_path: str
    ip_address: str
    home_x_mm: float
    home_y_mm: float
    home_z_mm: float
    home_roll_deg: float
    home_pitch_deg: float
    home_yaw_deg: float
    safe_x_min: float
    safe_x_max: float
    safe_y_min: float
    safe_y_max: float
    safe_z_min: float
    safe_z_max: float

# global setting
z_conveyor = 5.0 # mm


# =============================================================================
# RealSense D405 初期化
# =============================================================================

def init_realsense() -> "tuple[CameraIntrinsics, rs.pipeline, rs.align, float]":
    """RealSense D405 を初期化し、内部パラメータ・パイプライン・depth scale を返す。"""
    pipeline: rs.pipeline        = rs.pipeline()
    cfg: rs.config               = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 640x480 px, 30 fps
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)  # 640x480 px, 30 fps
    profile: rs.pipeline_profile = pipeline.start(cfg)

    raw_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr: CameraIntrinsics = CameraIntrinsics(
        fx=raw_intr.fx, fy=raw_intr.fy, cx=raw_intr.ppx, cy=raw_intr.ppy,
    )
    print(f"intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")

    align: rs.align      = rs.align(rs.stream.color)
    depth_scale: float   = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"depth scale: {depth_scale:.6f} m/unit")

    for _ in range(5):  # 5 warmup frames
        pipeline.wait_for_frames()
    print("camera warm-up done")

    return intr, pipeline, align, depth_scale


# =============================================================================
# DeepLabV3+
# =============================================================================

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


# =============================================================================
# OpenNR ロボットアーム
# =============================================================================

def sleepUntilServoOn() -> None:
    """サーボONが確認できるまで待つ。最大 20 秒でタイムアウト。"""
    isServoOn: list[bool] = [False]
    timeout: int = 20  # sec
    counter: int = 0
    while True:
        time.sleep(1)  # sec
        counter += 1
        if counter > timeout:
            print("sleepUntilServoOn(): timed out")
            break
        global_nr.AcsFixedIOServoOn(isServoOn)
        if isServoOn[0]:
            break


def sleepUntilRobotStopped() -> None:
    """ロボットの全軸速度がゼロになるまで待つ。"""
    speed: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    while True:
        time.sleep(0.5)  # sec
        global_nr.AcsAxisOrderSpeed(speed, 1, len(speed))
        if all(abs(x) < 0.001 for x in speed):  # 0.001 deg/sec (停止判定閾値)
            break
    time.sleep(0.1)  # sec (停止後の整定待ち)


def _init_robot(robot_cfg: RobotConfig) -> None:
    """OpenNR に接続してサーボをONにする。"""
    global_nr.LoadDLL(dllpath=robot_cfg.dll_path)
    openId: int = global_nr.Open(robot_cfg.ip_address)
    if openId < 0:
        raise RuntimeError(f"robot connection failed (error={openId})")
    print(f"[ROBOT] connected (id={openId})")
    time.sleep(1)  # sec (接続安定待ち)
    print("[ROBOT] servo ON")
    global_nr.CtrlMotor(1)
    sleepUntilServoOn()
    print("[ROBOT] servo ON done")


def _close_robot() -> None:
    """ロボットのサーボをOFFにして接続を閉じる。"""
    global_nr.CtrlMotor(0)
    global_nr.Close()
    print("[ROBOT] disconnected")


def is_safe_xyz(x: float, y: float, z: float, robot_cfg: RobotConfig) -> bool:
    """指定座標が安全動作範囲内かどうかチェックする。"""
    in_range: bool = (
        robot_cfg.safe_x_min <= x <= robot_cfg.safe_x_max
        and robot_cfg.safe_y_min <= y <= robot_cfg.safe_y_max
        and robot_cfg.safe_z_min <= z <= robot_cfg.safe_z_max
    )
    if not in_range:
        print(f"not safe xyz: {x:.1f}, {y:.1f}, {z:.1f}")
    return in_range


def _move_to_home(robot_cfg: RobotConfig) -> None:
    """ロボットアームを撮影用初期位置へ移動する。"""
    if not is_safe_xyz(robot_cfg.home_x_mm, robot_cfg.home_y_mm, robot_cfg.home_z_mm, robot_cfg):
        print("_move_to_home() aborted")
        return
    pose: NR_POSE = NR_POSE(
        robot_cfg.home_x_mm, robot_cfg.home_y_mm, robot_cfg.home_z_mm,
        robot_cfg.home_roll_deg, robot_cfg.home_pitch_deg, robot_cfg.home_yaw_deg,
    )
    print(f"[ROBOT] moving to home ({robot_cfg.home_x_mm:.1f}, {robot_cfg.home_y_mm:.1f}, {robot_cfg.home_z_mm:.1f}) mm")
    global_nr.CtrlMoveX(pose, nType=1)
    sleepUntilRobotStopped()
    print("[ROBOT] home position reached")



# =============================================================================
# TrackDLO ヘルパー
# =============================================================================

def initialize_state(
    X: np.ndarray,
    num_nodes: int,
    z_max_m: float,
) -> "TrackdloState | None":
    """点群 X から TrackdloState を初期化して返す。失敗時は None。"""
    state: TrackdloState = TrackdloState(num_nodes)

    X = X[X[:, 2] < z_max_m]
    if len(X) < num_nodes * 3:
        print(f"not enough valid points after Z filter: {len(X)}")
        state.close()
        return None

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
    if np.any(np.abs(Y_fit) > z_max_m * 10):
        print(f"  [cpd_lle] abnormally large value: max={np.abs(Y_fit).max():.2f}")
        state.close()
        return None

    Y_sorted: np.ndarray = sort_pts(Y_fit)
    if not np.all(np.isfinite(Y_sorted)):
        print("  [sort_pts] NaN/Inf detected, skipping initialization")
        state.close()
        return None
    if np.any(np.abs(Y_sorted) > z_max_m * 10):
        print(f"  [sort_pts] abnormally large value: max={np.abs(Y_sorted).max():.2f}")
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

    L: float = oz * 0.1  # mm (軸の描画長 = Z 距離の 10%)
    axes: list[tuple[float, float, float, tuple[int, int, int]]] = [
        (ox + L, oy,     oz, (0,   0, 255)),
        (ox,     oy + L, oz, (0, 255,   0)),
    ]
    tx: float
    ty: float
    tz: float
    color: tuple[int, int, int]
    for tx, ty, tz, color in axes:
        tip_px: "tuple[int, int] | None" = _project_pt(tx, ty, tz, intr)
        if tip_px is None:
            continue
        cv2.arrowedLine(img, origin_px, tip_px, color, 2, tipLength=0.25)


def draw_debug(
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    Y: "np.ndarray | None",
    intr: CameraIntrinsics,
    centroid_px: "tuple[int, int] | None" = None,
) -> None:
    """デバッグ用ウィンドウを cv2.imshow で更新する。"""
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
            cv2.circle(vis_bgr, (px, py), 5, (0, 0, 255), -1)  # radius 5 px
            if prev_px is not None:
                cv2.line(vis_bgr, prev_px, (px, py), (0, 220, 255), 2)
            prev_px = (px, py)
    if centroid_px is not None:
        cv2.circle(vis_bgr, centroid_px, 10, (0, 0, 255), -1)  # radius 10 px
    cv2.imshow("BGR image", vis_bgr)

    overlay: np.ndarray      = bgr.copy()
    overlay[mask == 1]       = (0, 255, 0)
    mask_overlay: np.ndarray = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
    # 座標軸の原点を Z=300mm (カメラ前方 30cm) に置く。
    # mask_depth_to_pointcloud が返す点群は mm 単位なので、
    # ここで渡す origin も mm 単位でなければならない。
    # 以前は座標系が m 単位だったため 0.3 (= 0.3m) を使っていたが、
    # mm 統一後に 0.3 のままにすると 0.3mm (= カメラ極近傍) になり
    # 投影後の軸が画面中央の 1 点に集中して見えなくなる。
    _draw_coord_axes(mask_overlay, np.array([0.0, 0.0, 300.0]), intr)
    cv2.imshow("mask overlay", mask_overlay)



# =============================================================================
# 点群生成 (Python 実装)
# preprocessing C++ の代わりに使う
# =============================================================================

def mask_depth_to_pointcloud(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intr: CameraIntrinsics,
) -> np.ndarray:
    """
    バイナリマスク + 深度画像 → 3D 点群 (N×3, float64, 単位 mm)

    ピンホール逆投影:
      Z = depth_mm[v, u]          (mm のまま)
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
    """
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        print("no cable pixel in mask")
        return np.empty((0, 3))

    Z: np.ndarray     = depth_mm[vs, us].astype(np.float64)
    valid: np.ndarray = Z > 0
    vs, us, Z = vs[valid], us[valid], Z[valid]
    if len(us) == 0:
        return np.empty((0, 3))

    pts: np.ndarray = np.column_stack([
        (us - intr.cx) * Z / intr.fx,
        (vs - intr.cy) * Z / intr.fy,
        Z,
    ])

    # 5mm ボクセルグリッドでダウンサンプリング
    voxels: np.ndarray = np.floor(pts / 5.0).astype(np.int32)
    idx: np.ndarray
    _, idx = np.unique(voxels, axis=0, return_index=True)
    return pts[idx]


# =============================================================================
# pixel-based keypoint calcurator
# =============================================================================

def calc_pixelbased_mid_keypoint(mask: np.ndarray) -> "tuple[int, int] | None":
    """マスク内のピクセル重心座標 (pixel_x, pixel_y) を返す。100 px 未満なら None。"""
    vs, us = np.where(mask > 0)
    n_pixels: int = len(vs)
    print(f"cable pixels: {n_pixels} px")
    if n_pixels < 100:  # 100 px
        return None
    return int(np.mean(us)), int(np.mean(vs))

# C++ code
# std::pair<int, int> calc_pixel_based_mid_keypoint(std::vector<std::vector<int>> mask) {
#     const int ch_idx = (int)mask.size() / 2;
#     const int cw_idx = (int)mask[0].size() / 2;
#     std::pair<int, int> cg; // h, w
#     cnt = 0;
#
#     for (int i = 0; i < (int)mask.size(); i++) {
#         for (int j = 0; j < (int)mask[i].size(); j++) {
#             if (mask[i][j] != 1) continue;
#             else cnt++;
#             if (i - ch_idx < 0) {
#                 cg.first -= i;
#             } else if (i - ch_idx > 0) {
#                 cg.first += i;
#             }
#             if (j - cw_idx < 0) {
#                 cg.second -= j;
#             } else if (j - cw_idx > 0) {
#                 cg.second += j;
#             }
#         }
#     }
#     return std::pair<int, int>(cg.first / cnt, cg.second / cnt);
# }



# =============================================================================
# メイン
# =============================================================================

def main() -> None:
    """RealSense から 1 フレーム取得し、TrackDLO でキーポイントを推定してロボットへ送る。"""
    # ── 設定 ──────────────────────────────────────────────────────────────────
    weights: Path    = HERE / "mz07_demo_0624" / "weights" / "best_deeplabv3plus_cable.pth"
    num_nodes: int   = 15
    max_retries: int = 5
    z_max_m: float   = 2000.0   # mm
    device: str      = "cpu"   # "cuda" if torch.cuda.is_available() else "cpu"

    robot_cfg: RobotConfig = RobotConfig(
        dll_path=str(HERE / "OpenNR-IF" / "OpenNR-IF.dll"),
        ip_address="127.0.0.1",
        home_x_mm=480.0,   home_y_mm=-66.0,   home_z_mm=370.0,
        home_roll_deg=0.0, home_pitch_deg=0.0, home_yaw_deg=-180.0,
        safe_x_min=420.0,  safe_x_max=670.0,
        safe_y_min=-300.0, safe_y_max=500.0,
        safe_z_min=110.0,  safe_z_max=500.0,
    )   # currently, safe value is invalid

    # カメラ座標系からロボットベース座標系への同次変換行列 (4×4)
    # 実機キャリブレーション後にここを書き換えること。
    _r: float = np.deg2rad(90)
    R_camera_to_robot: np.ndarray = np.array([
        [np.cos(_r),   -1 * np.sin(_r),  0.0],
        [np.sin(_r),   np.cos(_r),       0.0],
        [0.0,               0.0,         1.0],
    ], dtype=np.float64)

    # ── 初期化 ────────────────────────────────────────────────────────────────
    print(f"device = {device}")
    model: smp.DeepLabV3Plus = load_model(weights, device)
    print("model loaded")

    intr: CameraIntrinsics
    pipeline: rs.pipeline
    align: rs.align
    depth_scale: float
    intr, pipeline, align, depth_scale = init_realsense()
    print("got camera intrinsics")

    print("[main] init robot")
    _init_robot(robot_cfg)
    print("[main] move to home pos")
    _move_to_home(robot_cfg)
    time.sleep(0.5)  # sec (ホーム位置到達後の整定待ち)

    def send_XT_to_robot(mid_node_robot: np.ndarray) -> None:
        """ロボットベース座標系の中間ノード座標をロボットアームへ送る。"""
        if not np.all(np.isfinite(mid_node_robot)):
            return
        shift_x: float = float(mid_node_robot[0])
        shift_y: float = float(mid_node_robot[1])
        # shift_z: float = float(mid_node_robot[2])

        if robot_cfg.home_z_mm <= robot_cfg.safe_z_min:
            print(f"home_z_mm ({robot_cfg.home_z_mm}) must be greater than safe_z_min ({robot_cfg.safe_z_min})")
            return
        # ホーム位置 (home_z_mm) から safe_z_min の 30mm 上まで降りる相対距離。
        # CtrlMoveXT の正Z方向は下向きなので正の値になる。
        # 着地Z = home_z_mm - shift_z = safe_z_min + 30mm
        shift_z: float = robot_cfg.home_z_mm - z_conveyor   # mm

        # if not is_safe_xyz(shift_x, shift_y, shift_z, robot_cfg):
        #     print("target not in safe range, skipping")
        #     return

        ## instant coordinate transform
        # shift_pose: NR_POSE = NR_POSE(      
        #     -1 * shift_y, shift_x, shift_z,
        #     0.0,      0.0,      0.0,
        # )
        shift_pose: NR_POSE = NR_POSE(
            shift_x, shift_y, shift_z,
            0.0,      0.0,      0.0,
        )

        global_nr.CtrlMoveXT(shift_pose, nType=1)
        sleepUntilRobotStopped()


    # ── フレーム取得〜TrackDLO ────────────────────────────────────────────────
    print("start estimation pipeline")
    try:
        pt_3d: np.ndarray    = np.empty((0, 3))
        bgr: np.ndarray      = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_mm: np.ndarray = np.zeros((480, 640), dtype=np.uint16)
        mask: np.ndarray     = np.zeros((480, 640), dtype=np.uint8)

        attempt: int
        for attempt in range(max_retries):
            frames: rs.composite_frame  = pipeline.wait_for_frames()
            aligned: rs.composite_frame = align.process(frames)
            color_frame: rs.video_frame = aligned.get_color_frame()
            depth_frame: rs.depth_frame = aligned.get_depth_frame()

            bgr = np.asanyarray(color_frame.get_data())
            depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())
            depth_mm = np.clip(
                depth_raw.astype(np.float32) * depth_scale * 1000.0,  # m to mm 変換 (depth_scale の単位は m/unit)
                0, 65535
            ).astype(np.uint16)

            mask = infer_mask(bgr, model, device)

            # pixel-based 重心
            centroid_px: "tuple[int, int] | None" = calc_pixelbased_mid_keypoint(mask)
            if centroid_px is None:
                print(f"no cable pixels found, retry {attempt + 1}/{max_retries}")
                draw_debug(bgr, depth_mm, mask, None, intr)
                print("press Space to retry, ESC to exit")
                key: int = cv2.waitKey(0)
                if key == 27:
                    return
                continue

            u, v = centroid_px
            # デバッグ: 重心ピクセル1点の深度を確認する。
            # depth=0 は RealSense が深度計測できていないことを意味する
            # (ケーブル表面の光沢・端部では depth=0 が頻発する)。
            print(f"depth at centroid pixel: depth_mm[v={v}, u={u}] = {depth_mm[v, u]} mm")

            # 重心ピクセル1点の深度が0になりうるため、単点ルックアップをやめる。
            # ケーブルマスク全体の有効深度ピクセルから点群を作り、その3D平均を重心とする。
            pt_3d_all: np.ndarray = mask_depth_to_pointcloud(mask, depth_mm, intr)
            if len(pt_3d_all) == 0:
                print(f"no valid 3D points in cable mask, retry {attempt + 1}/{max_retries}")
                draw_debug(bgr, depth_mm, mask, None, intr)
                print("press Space to retry, ESC to exit")
                key = cv2.waitKey(0)
                if key == 27:
                    return
                continue
            pt_3d = pt_3d_all.mean(axis=0, keepdims=True)  # shape (1, 3), mm

            # TrackDLO (not stable)
            # X: np.ndarray = mask_depth_to_pointcloud(mask, depth_mm, intr)
            # if len(X) < num_nodes * 3:
            #     print(f"not enough points: {len(X)}, retry {attempt + 1}/{max_retries}")
            #     draw_debug(bgr, depth_mm, mask, None, intr)
            #     print("press Space to retry, ESC to exit")
            #     key: int = cv2.waitKey(0)
            #     if key == 27:
            #         return
            #     continue
            # state = initialize_state(np.ascontiguousarray(X, dtype=np.float64), num_nodes, z_max_m)
            # if state is None:
            #     print(f"TrackDLO initialization failed, retry {attempt + 1}/{max_retries}")
            #     draw_debug(bgr, depth_mm, mask, None, intr)
            #     print("press Space to retry, ESC to exit")
            #     key = cv2.waitKey(0)
            #     if key == 27:
            #         return
            #     continue
            # print(f"TrackDLO converged (attempt {attempt + 1})")

            print(f"centroid pixel: u={u}  v={v}")
            print(f"centroid 3D (camera): X={pt_3d[0, 0]:+.1f}  Y={pt_3d[0, 1]:+.1f}  Z={pt_3d[0, 2]:.1f}  [mm]")
            break

        if len(pt_3d) == 0:
            print(f"estimation failed after {max_retries} retries")
            return

        mid_camera: np.ndarray = pt_3d[0]
        mid_robot: np.ndarray = R_camera_to_robot @ mid_camera
        print(f"centroid (robot): X={mid_robot[0]:+.1f}  Y={mid_robot[1]:+.1f}  Z={mid_robot[2]:.1f}  [mm]")

        # TrackDLO (not stable)
        # mid: np.ndarray = get_mid_node(state.Y)   # mm
        # p_h: np.ndarray = np.array([mid[0], mid[1], mid[2]], dtype=np.float64)
        # mid_robot: np.ndarray = (R_camera_to_robot @ p_h)[:3]
        # print(f"mid node (camera): X={mid[0]:+.1f}  Y={mid[1]:+.1f}  Z={mid[2]:.1f}  [mm]")
        # draw_debug(bgr, depth_mm, mask, state.Y, intr)
        # state.close()

        draw_debug(bgr, depth_mm, mask, None, intr, centroid_px=(u, v))
        cv2.waitKey(1)

        time.sleep(3)  # sec (デバッグ表示を目視確認するための待機)
        # send_XT_to_robot(mid_robot)
        # np.array を使う。np.ndarray([...]) はshapeを受け取るコンストラクタなのでバグになる。
        demo_shift: np.ndarray = np.array([40.0, -20.0, 0.0])  # mm, tool coordinate
        send_XT_to_robot(mid_robot + demo_shift)

    finally:
        _close_robot()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("done.")


if __name__ == "__main__":
    main()
