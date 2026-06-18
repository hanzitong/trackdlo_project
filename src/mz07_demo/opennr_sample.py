"""
opennr_sample.py

OpenNR の CtrlMoveX (絶対移動) と CtrlMoveXR (相対移動) のサンプル。
movex.py をベースに型ヒントと docstring を追加したもの。

usage:
  cd src/mz07_demo/opennr_test
  python ../opennr_sample.py
"""

import sys
import time
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "opennr_test" / ".venv" / "Lib" / "site-packages"))

from opennr_py import NR_POSE, global_nr  # type: ignore[import]

DLL_PATH: str   = "./OpenNR-IF/OpenNR-IF.dll"
IP_ADDRESS: str = "127.0.0.1"


def sleepUntilServoOn() -> None:
    """サーボONが確認できるまで待つ。最大 10 秒でタイムアウト。"""
    isServoOn: list[bool] = [False]
    timeout: int = 10
    counter: int = 0
    while True:
        time.sleep(1)
        counter += 1
        if counter > timeout:
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


def main() -> None:
    """CtrlMoveX と CtrlMoveXR の動作確認サンプル。"""
    global_nr.LoadDLL(dllpath=DLL_PATH)
    openId: int = global_nr.Open(IP_ADDRESS)
    if openId < 0:
        raise RuntimeError(f"robot connection failed (error={openId})")
    print(f"connected (id={openId})")
    time.sleep(1)

    print("servo ON")
    global_nr.CtrlMotor(1)
    sleepUntilServoOn()
    print("servo ON done")

    # ── 絶対移動: CtrlMoveX ─────────────────────────────────────────────────
    # NR_POSE(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg)
    print("absolute move 1")
    pose: NR_POSE = NR_POSE(469.0, -79.0, 373.0, 29.0, 0.0, -175.0)
    global_nr.CtrlMoveX(pose, nType=1)
    sleepUntilRobotStopped()
    print("absolute move 1 done")

    # ── 相対移動: CtrlMoveXR ────────────────────────────────────────────────
    # NR_POSE の値は現在位置からの差分 (dx_mm, dy_mm, dz_mm, ...)
    print("relative move 1  (Y -10mm, Z +50mm)")
    pose = NR_POSE(0.0, -10.0, 50.0, 0.0, 0.0, 0.0)
    global_nr.CtrlMoveXR(pose, nType=1)
    sleepUntilRobotStopped()
    print("relative move 1 done")

    print("relative move 2  (Y +10mm, Z -50mm)  -- 元の位置に戻る")
    pose = NR_POSE(0.0, 10.0, -50.0, 0.0, 0.0, 0.0)
    global_nr.CtrlMoveXR(pose, nType=1)
    sleepUntilRobotStopped()
    print("relative move 2 done")

    print("servo OFF")
    global_nr.CtrlMotor(0)
    global_nr.Close()
    print("disconnected")


if __name__ == "__main__":
    main()
