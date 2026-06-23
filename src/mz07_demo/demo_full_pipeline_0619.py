"""
demo_full_pipeline.py

RealSense D405 から取得した BGR/depth 画像をもとにケーブルのキーポイントを推定し、
中間キーポイントを OpenNR 経由でロボットアームへ送るフルパイプラインデモ。

pipeline:
  1. RealSense D405 から BGR + depth 画像を取得
  2. DeepLabV3+ でバイナリマスクを生成
  3. バイナリマスク + depth から 3D 点群を生成 (preprocessing C++) 
        * 現状のpreprocessing C++だとバグがあるかも。。Python実装を使いましょう。
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
# from preprocessing_cdll import images_to_pointcloud, compute_visible_nodes


# ─── 設定 ─────────────────────────────────────────────────────────────────────

WEIGHTS: Path    = HERE / "best_deeplabv3plus_cable.pth"
# NUM_NODES: int   = 15
DEVICE: str      = "cuda" if torch.cuda.is_available() else "cpu"

MAX_RETRIES: int = 5   # TrackDLO が収束しない場合の最大リトライ回数


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

def infer_mask(bgr: np.ndarray) -> np.ndarray:
    """BGR 画像からバイナリマスク (uint8, 0=background / 1=cable) を返す。"""
    rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    x: torch.Tensor = torch.tensor(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob: np.ndarray = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.uint8)



# =============================================================================
# Step 1: RealSense D405 初期化
# =============================================================================

_pipeline: rs.pipeline = rs.pipeline()
_cfg: rs.config        = rs.config()
_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
_cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
_profile: rs.pipeline_profile = _pipeline.start(_cfg)

_intr = _profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
FX = _intr.fx       # focal length [pixel]
FY = _intr.fy       # focal length [pixel]
CX = _intr.ppx      # main point x coordinate [pixel]
CY = _intr.ppy      # main point y coordinate [pixel]
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


# カメラ座標系からツール先端座標系への同次変換行列 (4×4)
# 左上 3×3 = 回転行列、右上 3×1 = 平行移動 [m]
# 実機キャリブレーション後にここを書き換えること。現在は暫定値（回転なし）。


# from sig_j6_frange to sig_tool (nachi tool caribration)
# sign perspective is from sig_j6_frange
toolx_frange = -75.3   # mm
tooly_frange = 26.1   # mm
toolz_frange = 119.3  # mm

# from sig_j6_frange to sig_camera (cad data)
camx_frange = -55   # mm
camy_frange = 0     # mm (+-10mm for each monocular camera)
camz_frange = 25    # mm (maybe value) frange 5mm + cam 20mm ?

# Bug fix: 平行移動成分を mm → m に変換 (/1000)。
# 入力点 p_cam は m 単位 (TrackDLO 出力)、行列の単位を合わせる。
# T_CAMERA_TO_TOOL: np.ndarray = np.array([
#     [1.0, 0.0, 0.0,  (toolx_frange - camx_frange) / 1000.0],   # [r00 r01 r02 | tx]
#     [0.0, 1.0, 0.0,  (tooly_frange - camy_frange) / 1000.0],   # [r10 r11 r12 | ty]
#     [0.0, 0.0, 1.0,  (toolz_frange - camz_frange) / 1000.0],   # [r20 r21 r22 | tz]
#     [0.0, 0.0, 0.0,  1.00],                                     # [ 0   0   0  |  1]
# ], dtype=np.float64)

T_CAMERA_TO_TOOL: np.ndarray = np.array([
    [0.0, 1.0, 0.0,  (toolx_frange - camx_frange) / 1000.0],   # [r00 r01 r02 | tx]
    [-1.0, 0.0, 0.0,  (tooly_frange - camy_frange) / 1000.0],   # [r10 r11 r12 | ty]
    [0.0, 0.0, 1.0,  (toolz_frange - camz_frange) / 1000.0],   # [r20 r21 r22 | tz]
    [0.0, 0.0, 0.0,  1.00],                                     # [ 0   0   0  |  1]
], dtype=np.float64)


rotation_rad = np.deg2rad(180)
T_ROBOT_TO_TOOL : np.ndarray = np.array([
    [0.0, 0.0, 0.0 ,HOME_X_MM],
    [np.cos(rotation_rad), -1.0 * np.sin(rotation_rad), 0.0, HOME_Y_MM],
    [np.sin(rotation_rad), np.cos(rotation_rad), 1.0, HOME_Z_MM],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def camera_to_robot(p_cam: np.ndarray) -> np.ndarray:
    p_h: np.ndarray = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
    p_tool_h: np.ndarray = T_ROBOT_TO_TOOL @ p_h
    return p_tool_h[:3]



def camera_to_tool(p_cam: np.ndarray) -> np.ndarray:
    """カメラ座標系の3D点をツール先端座標系に変換する。
    同次変換行列 T_CAMERA_TO_TOOL を使う。
    """
    p_h: np.ndarray = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
    p_tool_h: np.ndarray = T_CAMERA_TO_TOOL @ p_h
    return p_tool_h[:3]


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


def send_target_to_robot_sig_tool(mid_node_tool: np.ndarray) -> None:
    """ツール座標系の中間ノード座標をロボットアームの目標 TCP 位置として送る。

    引数はツール先端座標系 [m]。camera_to_tool() を適用した後に呼ぶこと。
    ツール座標系 Z 軸とロボットベース座標系 Z 軸は 90° 回転の関係にある。
    """
    if not np.all(np.isfinite(mid_node_tool)):
        return

    # Bug fix: camera_to_tool() は main() で呼び済み。ここで再度呼ぶと二重変換になる。
    # mid_node_tool はツール座標系 [m] なので m→mm 変換のみ行う。
    x_mm: float = float(mid_node_tool[0]) * 1000.0
    y_mm: float = float(mid_node_tool[1]) * 1000.0
    # z_mm: float = float(mid_node_tool[2]) * 1000.0  # Z は固定値で上書き
    z_mm: float = 130.0

    # TODO: ツール座標系とロボットベース座標系の間に 90° 回転があるか確認すること。
    # 旧実装 (monodukuri) では超絶雑な座標変換をおこなていた。回転関係は考慮していなかった。
    #   target_x = HOME_X_MM - y_mm   (robot.x = HOME_X - tool.y)
    #   target_y = HOME_Y_MM + x_mm   (robot.y = HOME_Y + tool.x)
    # target_z: float = HOME_Z_MM + z_mm

    # tuzitumawase
    x_mm: float = float(mid_node_tool[0])
    y_mm: float = float(mid_node_tool[1])
    target_x: float = x_mm
    target_y: float = y_mm
    # target_z: float = z_mm
    target_z: float = 120.0     # for safety


    if not is_safe_xyz(target_x, target_y, target_z):
        print("send_target_to_robot() aborted")
        return None

    pose = NR_POSE(target_x, target_y, target_z, HOME_ROLL_DEG, HOME_PITCH_DEG, HOME_YAW_DEG)
    global_nr.CtrlMoveX(pose, nType=1)
    # pose = NR_POSE(HOME_X_MM - y_mm, HOME_Y_MM + x_mm, 111, 95, -30, -180)    # tukami pose abs pose sig_robot

    # pose = NR_POSE(target_x/5.0, target_y/5.0, target_z/5.0, 0, 0, 0)
    # global_nr.CtrlMoveXT(pose, nType=1)

    sleepUntilRobotStopped()



# =============================================================================
# メイン
# =============================================================================


# preprocess.dll/.so  はバグを含んでいるっぽい.
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

    time.sleep(0.5)

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
        print(f"mid node (camera): X={mid[0]:+.4f}  Y={mid[1]:+.4f}  Z={mid[2]:.4f}  [m]")

        # カメラ座標 → ツール先端座標に変換
        # mid_tool: np.ndarray = camera_to_tool(mid)

        # convert camera to robot coordinate 
        mid_tool: np.ndarray = camera_to_robot(mid)
        print(f"mid node (robot):   X={mid_tool[0]:+.4f}  Y={mid_tool[1]:+.4f}  Z={mid_tool[2]:.4f}  [m]")

        draw_debug(bgr, depth_mm, mask, state.Y)
        cv2.waitKey(1)

        # ── Step 5: OpenNR でロボットアームへ送る ─────────────────────────────
        time.sleep(1)
        # send_target_to_robot_sig_tool(mid_tool)
        send_target_to_robot_sig_tool(mid_tool)
        # send_xyzrpy_to_robot()    # TODO function


        state.close()

    finally:
        _close_robot()
        _pipeline.stop()
        cv2.destroyAllWindows()
        print("done.")


if __name__ == "__main__":
    main()
