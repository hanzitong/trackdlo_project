"""
demo_full_pipeline.py

RealSense D405 から取得した BGR/depth 画像をもとにケーブルのキーポイントを推定し、
中間キーポイントを OpenNR 経由でロボットアームへ送るフルパイプラインデモ。

pipeline:
  1. RealSense D405 から BGR + depth 画像を取得
  2. DeepLabV3+ でバイナリマスクを生成
  3. バイナリマスク + depth から 3D 点群を生成 (preprocessing C++)
  4. TrackDLO でケーブルキーポイントを推定 (trackdlo C++)
  5. 中間キーポイントを OpenNR 経由でロボットアームへ送る

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
  cd src/mz07_demo && python demo_full_pipeline.py
"""

import os
import sys
import time
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent

os.environ["TRACKDLO_LIB_PATH"]      = str(HERE / "libtrackdlo_c.dll")
os.environ["PREPROCESSING_LIB_PATH"] = str(HERE / "libpreprocessing_c.dll")

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
    TdloParams,
    default_params,
    tracking_step,
    cpd_lle,
    sort_pts,
)
from preprocessing_cdll import images_to_pointcloud, compute_visible_nodes


# ─── 設定 ─────────────────────────────────────────────────────────────────────

WEIGHTS: Path    = HERE / "best_deeplabv3plus_cable.pth"
NUM_NODES: int   = 15
DEVICE: str      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_RETRIES: int = 5   # TrackDLO が収束しない場合の最大リトライ回数
# バグ修正 (cpd_lle 異常大座標値): 深度外れ値が Y_init スケールを狂わせるため上限を設ける
# 詳細: docs/problem_solving/cpd_lle_abnormal_large_value.md
Z_MAX_M: float = 2.0

IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

_intr = _profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
FX = _intr.fx
FY = _intr.fy
CX = _intr.ppx
CY = _intr.ppy
print(f"intrinsics: fx={FX:.1f} fy={FY:.1f} cx={CX:.1f} cy={CY:.1f}")

_align: rs.align = rs.align(rs.stream.color)

_depth_sensor = _profile.get_device().first_depth_sensor()
DEPTH_SCALE: float = _depth_sensor.get_depth_scale()
print(f"depth scale: {DEPTH_SCALE:.6f} m/unit")

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
    """BGR 画像からバイナリマスク (uint8, 0=background / 1=cable) を返す。"""
    rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    x: torch.Tensor = torch.tensor(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob: np.ndarray = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def initialize_state(X: np.ndarray) -> "TrackdloState | None":
    """点群 X から TrackdloState を初期化して返す。失敗時は None。"""
    state: TrackdloState = TrackdloState(NUM_NODES)

    # バグ修正 (cpd_lle 異常大座標値): Z 外れ値を除去しないと Y_init が巨大スケールになり
    # C++ 内の 0.1m フィルタで全点が除外され、cpd_lle が動かないまま収束を返す
    # 詳細: docs/problem_solving/cpd_lle_abnormal_large_value.md
    X = X[X[:, 2] < Z_MAX_M]
    if len(X) < NUM_NODES * 3:
        print(f"not enough valid points after Z filter: {len(X)}")
        state.close()
        return None

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
    # バグ修正 (cpd_lle 異常大座標値): isfinite は有限な異常大値を見逃すため境界チェックを追加
    if np.any(np.abs(Y_fit) > Z_MAX_M * 10):
        print(f"  [cpd_lle] abnormally large value detected: max={np.abs(Y_fit).max():.2f}")
        state.close()
        return None

    Y_sorted: np.ndarray = sort_pts(Y_fit)
    if not np.all(np.isfinite(Y_sorted)):
        print("  [sort_pts] NaN/Inf detected, skipping initialization")
        state.close()
        return None
    # バグ修正 (sort_pts 後の異常大座標値): sort_pts は最近傍順にノードを並べ替えるが、
    # cpd_lle が端ノードを点群から遠い位置に残した場合、並べ替え後に中間インデックスへ
    # 移動することがある。get_mid_node はこのインデックスを返すため、
    # sort_pts 後にも異常値チェックが必要。
    if np.any(np.abs(Y_sorted) > Z_MAX_M * 10):
        print(f"  [sort_pts] abnormally large value detected: max={np.abs(Y_sorted).max():.2f}")
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
    # return Y[0]
    return Y[len(Y) // 2]


def _project_pt(x3: float, y3: float, z3: float) -> "tuple[int, int] | None":
    """3D 点をピンホールモデルで 2D 画素座標に変換する。z3<=0 なら None。"""
    if not (np.isfinite(z3) and z3 > 0):
        return None
    return int(x3 * FX / z3 + CX), int(y3 * FY / z3 + CY)


def _draw_coord_axes(img: np.ndarray, origin: np.ndarray) -> None:
    """カメラ座標系の XY 軸を img に in-place で描画する。"""
    ox: float = float(origin[0])
    oy: float = float(origin[1])
    oz: float = float(origin[2])

    origin_px: "tuple[int, int] | None" = _project_pt(ox, oy, oz)
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
        tip_px: "tuple[int, int] | None" = _project_pt(tx, ty, tz)
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
) -> None:
    """デバッグ用 3 ウィンドウを cv2.imshow で更新する。"""
    vis_bgr: np.ndarray = bgr.copy()
    if Y is not None:
        prev_px: "tuple[int, int] | None" = None
        h: int = 0
        w: int = 0
        h, w = vis_bgr.shape[:2]    # for preventing cv2.circle() error
        x3: float
        y3: float
        z3: float
        for x3, y3, z3 in Y:
            if not (np.isfinite(x3) and np.isfinite(y3) and np.isfinite(z3) and z3 > 0):
                continue
            px: int = int(x3 * FX / z3 + CX)
            py: int = int(y3 * FY / z3 + CY)
            if not (0 <= px < w and 0 <= py < h):   # for preventing cv2.circle() error
                prev_py = None
                continue
            cv2.circle(vis_bgr, (px, py), 5, (0, 0, 255), -1)
            if prev_px is not None:
                cv2.line(vis_bgr, prev_px, (px, py), (0, 220, 255), 2)
            prev_px = (px, py)
    cv2.imshow("BGR image", vis_bgr)

    depth_vis: np.ndarray = np.clip(depth_mm, 0, 10000).astype(np.float32)
    depth_vis = (depth_vis / 10000.0 * 255.0).astype(np.uint8)
    # cv2.imshow("depth image", depth_vis)

    overlay: np.ndarray      = bgr.copy()
    overlay[mask == 1]       = (0, 255, 0)
    mask_overlay: np.ndarray = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
    _draw_coord_axes(mask_overlay, np.array([0.0, 0.0, 0.3]))
    cv2.imshow("mask overlay", mask_overlay)


# =============================================================================
# Step 5: OpenNR ロボットアーム
# =============================================================================

DLL_PATH: str   = "./OpenNR-IF/OpenNR-IF.dll"
IP_ADDRESS: str = "127.0.0.1"


def sleepUntilServoOn() -> None:
    """サーボONが確認できるまで待つ。最大 10 秒でタイムアウト。"""
    isServoOn: list[bool] = [False]
    timeout: int = 20
    counter: int = 0
    while True:
        time.sleep(1)
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
        time.sleep(0.5)
        global_nr.AcsAxisOrderSpeed(speed, 1, len(speed))
        if all(abs(x) < 0.001 for x in speed):
            break
    time.sleep(0.1)


def _init_robot() -> None:
    """OpenNR に接続してサーボをONにする。"""
    global_nr.LoadDLL(dllpath=DLL_PATH)
    openId: int = global_nr.Open(IP_ADDRESS)
    if openId < 0:
        raise RuntimeError(f"robot connection failed (error={openId})")
    print(f"[ROBOT] connected (id={openId})")
    time.sleep(1)

    print("[ROBOT] servo ON")
    global_nr.CtrlMotor(1)
    sleepUntilServoOn()
    print("[ROBOT] servo ON done")


def _close_robot() -> None:
    """ロボットのサーボをOFFにして接続を閉じる。"""
    global_nr.CtrlMotor(0)
    global_nr.Close()
    print("[ROBOT] disconnected")


# 撮影前に移動する初期位置
HOME_X_MM: float     = 480.0
HOME_Y_MM: float     = -66.0
HOME_Z_MM: float     = 370.0
HOME_ROLL_DEG: float  = 0.0
HOME_PITCH_DEG: float = 0.0
HOME_YAW_DEG: float   = -180.0

# Work space (safe zone)
SAFE_X_MAX: float = 670
SAFE_X_MIN: float = 420
SAFE_Y_MAX: float = 500
SAFE_Y_MIN: float = -300
SAFE_Z_MAX: float = 500
# SAFE_Z_MIN: float = 101
SAFE_Z_MIN: float = 110


offsetX = 35.0
offsetY = 70.0

#offsetX = 0.0
#offsetY = 0.0


def is_safe_xyz(x: float, y: float, z: float) -> bool:
    is_safe = True
    if (x < SAFE_X_MIN or SAFE_X_MAX < x): is_safe = False
    if (y < SAFE_Y_MIN or SAFE_Y_MAX < y): is_safe = False
    if (z < SAFE_Z_MIN or SAFE_Z_MAX < z): is_safe = False

    if(not is_safe): 
        print(f"not safe xyz !!!!!!! {x}, {y}, {z}")

    return is_safe


def _move_to_home() -> None:
    """ロボットアームを撮影用初期位置へ移動する。"""
    if(not is_safe_xyz(HOME_X_MM, HOME_Y_MM, HOME_Z_MM)):
        print("_move_to_home() aborted")
        return None

    pose: NR_POSE = NR_POSE(
        HOME_X_MM, HOME_Y_MM, HOME_Z_MM,
        HOME_ROLL_DEG, HOME_PITCH_DEG, HOME_YAW_DEG,
    )
    print(f"[ROBOT] moving to home ({HOME_X_MM:.1f}, {HOME_Y_MM:.1f}, {HOME_Z_MM:.1f}) mm")
    global_nr.CtrlMoveX(pose, nType=1)
    sleepUntilRobotStopped()
    print("[ROBOT] home position reached")


# def send_target_to_robot(mid_node: np.ndarray) -> None:
#     """TrackDLO の中間ノード座標をロボットアームの目標 TCP 位置として送る。

#     Note:
#         カメラ座標 [m] をロボットベース座標 [mm] に変換するには
#         ハンドアイキャリブレーションが必要。現状は m→mm のスケール変換のみ。
#     """
#     if not np.all(np.isfinite(mid_node)):
#         return

#     # TODO: ハンドアイキャリブレーションで求めた変換行列をここに適用する
#     x_mm: float = float(mid_node[0]) * 1000.0
#     y_mm: float = float(mid_node[1]) * 1000.0
#     z_mm: float = float(mid_node[2]) * 1000.0

#     if (not is_safe_xyz(x_mm, y_mm, z_mm)):
#         print("send_target_to_robot() aborted")
#         return None

#     # NR_POSE の引数順は (x, y, z, roll, pitch, yaw) のはずだが、実機で要確認
#     pose: NR_POSE = NR_POSE(x_mm, y_mm, z_mm, 0.0, 0.0, 0.0)
#     global_nr.CtrlMoveX(pose, nType=1)
#     sleepUntilRobotStopped()


def send_target_to_robot_monodukuri(mid_node: np.ndarray) -> None:
    """TrackDLO の中間ノード座標をロボットアームの目標 TCP 位置として送る。

    Note:
        カメラ座標 [m] をロボットベース座標 [mm] に変換するには
        ハンドアイキャリブレーションが必要。現状は m→mm のスケール変換のみ。
    """
    if not np.all(np.isfinite(mid_node)):
        return

    # TODO: ハンドアイキャリブレーションで求めた変換行列をここに適用する
    x_mm: float = float(mid_node[0]) * 1000.0
    y_mm: float = float(mid_node[1]) * 1000.0
    # z_mm: float = float(mid_node[2]) * 1000.0
    z_mm = 130

    target_x = HOME_X_MM - y_mm
    target_y = HOME_Y_MM + x_mm 
    target_z = 130
    if (not is_safe_xyz(target_x, target_y, target_z)):
        print("send_target_to_robot() aborted")
        return None

    # NR_POSE の引数順は (x, y, z, roll, pitch, yaw) のはずだが、実機で要確認
    pose = NR_POSE(target_x + offsetX, target_y + offsetY , target_z, HOME_ROLL_DEG, HOME_PITCH_DEG, HOME_YAW_DEG)    # zahyouhenkan musi
    # pose = NR_POSE(HOME_X_MM - y_mm, HOME_Y_MM + x_mm, 111, 95, -30, -180)    # tukami pose
    global_nr.CtrlMoveX(pose, nType=1)
    sleepUntilRobotStopped()




# def send_xyzrpy_to_robot(X: float, Y: float , Z: float, r: float, p: float, y: float) -> None:
#     # TODO: make pseud-shift move with OpenNR
#     if (not is_safe_xyz(X, Y, Z)):
#         print("send_xyzrpy_to_robot() aborted")
#         return None


# def send_relxyzrpy_to_robot(X: float, Y: float , Z: float, r: float, p: float, y: float) -> None:
#     if (not is_safe_xyz(X, Y, Z)):
#         print("send_xyzrpy_to_robot() aborted")
#         return None
#     # NR_POSE の値は現在位置からの差分 (dx_mm, dy_mm, dz_mm, ...)
#     print("relative move")
#     pose: NR_POSE = NR_POSE(0.0, -10.0, 50.0, 0.0, 0.0, 0.0)
#     global_nr.CtrlMoveXR(pose, nType=1)
#     sleepUntilRobotStopped()
#     print("relative move done")


    return None


# =============================================================================
# メイン
# =============================================================================



def mask_depth_to_pointcloud(mask: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """
    バイナリマスク + 深度画像 (mm 単位) → 3D 点群 (N×3, float64, 単位 m)

    ピンホールカメラの逆投影:
      Z = depth_mm[v, u] / 1000.0
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy

    depth_mm は呼び出し側で raw * DEPTH_SCALE * 1000 に変換済みであること。
    """
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        return np.empty((0, 3))

    z_mm: np.ndarray = depth_mm[vs, us].astype(np.float64)
    valid: np.ndarray = z_mm > 0
    vs, us, z_mm = vs[valid], us[valid], z_mm[valid]
    if len(us) == 0:
        return np.empty((0, 3))

    Z: np.ndarray = z_mm / 1000.0
    pts: np.ndarray = np.column_stack([
        (us - CX) * Z / FX,
        (vs - CY) * Z / FY,
        Z,
    ])

    # 5mm ボクセルグリッドでダウンサンプリング
    voxels: np.ndarray = np.floor(pts / 0.005).astype(np.int32)
    idx: np.ndarray
    _, idx = np.unique(voxels, axis=0, return_index=True)
    return pts[idx]






def main() -> None:
    """RealSense から 1 フレーム取得し、DeepLabV3+ と TrackDLO を適用して
    ロボットアームへ司令を送るエントリポイント。
    """
    print("[main] init robot")
    _init_robot()

    print("[main] move to home pos")
    _move_to_home()

    try:
        # ── Step 1〜4: フレーム取得〜TrackDLO (収束しない場合はリトライ) ────────
        state: "TrackdloState | None" = None
        bgr: np.ndarray      = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_mm: np.ndarray = np.zeros((480, 640), dtype=np.uint16)
        mask: np.ndarray     = np.zeros((480, 640), dtype=np.uint8)

        attempt: int
        for attempt in range(MAX_RETRIES):
            # Step 1: フレーム取得
            frames: rs.composite_frame  = _pipeline.wait_for_frames()
            aligned: rs.composite_frame = _align.process(frames)
            color_frame: rs.video_frame = aligned.get_color_frame()
            depth_frame: rs.depth_frame = aligned.get_depth_frame()

            bgr = np.asanyarray(color_frame.get_data())
            depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())
            depth_mm = np.clip(
                depth_raw.astype(np.float32) * DEPTH_SCALE * 1000.0, 0, 65535
            ).astype(np.uint16)

            # Step 2: バイナリマスク生成
            mask = infer_mask(bgr)

            # Step 3: 3D 点群生成
            # X: np.ndarray = images_to_pointcloud(
            #     bgr, depth_mm, mask, FX, FY, CX, CY, leaf_size=0.005
            # )

            X: np.ndarray = mask_depth_to_pointcloud(mask, depth_mm)


            if len(X) < NUM_NODES * 3:
                print(f"not enough points: {len(X)}, retry {attempt + 1}/{MAX_RETRIES}")
                draw_debug(bgr, depth_mm, mask, None)
                print("press Space to retry, ESC to exit")
                key: int = cv2.waitKey(0)
                if key == 27:   # ESC
                    return
                # Space キーと同等のリセット: state を明示的に None にして次フレームへ
                state = None
                continue

            # Step 4: TrackDLO で初期化
            state = initialize_state(np.ascontiguousarray(X, dtype=np.float64))
            if state is None:
                print(f"TrackDLO initialization failed, retry {attempt + 1}/{MAX_RETRIES}")
                draw_debug(bgr, depth_mm, mask, None)
                print("press Space to retry, ESC to exit")
                key = cv2.waitKey(0)
                if key == 27:   # ESC
                    return
                # Space キーと同等のリセット: state を明示的に None にして次フレームへ
                state = None
                continue

            print(f"TrackDLO converged (attempt {attempt + 1})")
            break

        if state is None:
            print(f"TrackDLO failed after {MAX_RETRIES} retries")
            return

        mid: np.ndarray = get_mid_node(state.Y)
        print(f"mid node:  X={mid[0]:+.4f}  Y={mid[1]:+.4f}  Z={mid[2]:.4f}  [m]")

        draw_debug(bgr, depth_mm, mask, state.Y)
        cv2.waitKey(1)

        # ── Step 5: OpenNR でロボットアームへ送る ─────────────────────────────
        time.sleep(1)
        send_target_to_robot_monodukuri(mid)
        # send_xyzrpy_to_robot()    # TODO function


        state.close()

    finally:
        _close_robot()
        _pipeline.stop()
        cv2.destroyAllWindows()
        print("done.")


if __name__ == "__main__":
    main()
