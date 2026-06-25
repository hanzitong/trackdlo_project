
import os
import sys
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
    os.environ["TRACKDLO_LIB_PATH"]      = str(HERE / "lib" / "libtrackdlo_c.dll")
    os.environ["PREPROCESSING_LIB_PATH"] = str(HERE / "lib" / "libpreprocessing_c.dll")
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "opennr_test" / ".venv" / "Lib" / "site-packages"))


from dataclasses import dataclass
import pyrealsense2 as rs
from opennr_py import NR_POSE, global_nr

import time
import numpy as np


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


# =============================================================================
# OpenNR ロボットアーム
# =============================================================================

def sleepUntilServoOn() -> None:
    """サーボONが確認できるまで待つ。最大 20 秒でタイムアウト。"""
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


def _init_robot(robot_cfg: RobotConfig) -> None:
    """OpenNR に接続してサーボをONにする。"""
    global_nr.LoadDLL(dllpath=robot_cfg.dll_path)
    openId: int = global_nr.Open(robot_cfg.ip_address)
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





# ── 設定 ──────────────────────────────────────────────────────────────────
z_max_m: float   = 2000.0   # mm

robot_cfg: RobotConfig = RobotConfig(
    dll_path=str(HERE / "OpenNR-IF" / "OpenNR-IF.dll"),
    ip_address="127.0.0.1",
    home_x_mm=480.0,
    home_y_mm=-66.0,
    home_z_mm=370.0,
    home_roll_deg=0.0,
    home_pitch_deg=0.0,
    home_yaw_deg=-180.0,
    safe_x_min=420.0,
    safe_x_max=670.0,
    safe_y_min=-300.0,
    safe_y_max=500.0,
    safe_z_min=110.0,
    safe_z_max=500.0,
)

_r: float = np.deg2rad(90)
T_camera_to_robot: np.ndarray = np.array([  # mm
    [1.0,        0.0,         0.0, 0],
    [0.0, np.cos(_r), -np.sin(_r), 0],
    [0.0, np.sin(_r),  np.cos(_r), 0],
    [0.0,          0.0,          0.0, 1.0                 ],
], dtype=np.float64)



print("[main] init robot")
_init_robot(robot_cfg)
print("[main] move to home pos")
_move_to_home(robot_cfg)
time.sleep(0.5)


# inline function 
def send_to_robot(mid_node_robot: np.ndarray) -> None:
    """ロボットベース座標系の中間ノード座標をロボットアームへ送る。"""
    if not np.all(np.isfinite(mid_node_robot)):
        return
    target_x: float = float(mid_node_robot[0])
    target_y: float = float(mid_node_robot[1])
    target_z: float = 120.0    # Z は安全のため固定値
    if not is_safe_xyz(target_x, target_y, target_z, robot_cfg):
        print("target not in safe range, skipping")
        return
    pose: NR_POSE = NR_POSE(
        target_x, target_y, target_z,
        robot_cfg.home_roll_deg, robot_cfg.home_pitch_deg, robot_cfg.home_yaw_deg,
    )
    print("sending pose to opennr")
    # global_nr.CtrlMoveX(pose, nType=1)  # X: TCP ref, robot coordinate 
    global_nr.CtrlMoveXT(pose, nType=1)  # XT: TCP ref, tool coordinate
    print("finish sending pose to opennr")
    sleepUntilRobotStopped()


print("start estimation pipeline")

# mid: np.ndarray = get_mid_node(state.Y)   # mm
p_h: np.ndarray = np.array([300, 300, 300, 1.0], dtype=np.float64)  # mm
mid_robot: np.ndarray = (T_camera_to_robot @ p_h)[:3]

# send_to_robot(mid_robot)


_close_robot()

print("done.")


