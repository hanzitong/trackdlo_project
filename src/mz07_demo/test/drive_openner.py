"""
drive_openner.py

OpenNR を使ったロボットアームの動作確認スクリプト。
CtrlMoveJ (関節角度絶対移動) と CtrlMoveX/CtrlMoveXR (TCP 絶対/相対移動) のサンプル。

usage:
  cd src/mz07_demo/test && python drive_openner.py
"""

import sys
import time
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "opennr_test" / ".venv" / "Lib" / "site-packages"))

from opennr_py import NR_POSE, global_nr  # type: ignore[import]


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
    """OpenNR の動作確認サンプル。"""
    dll_path: str   = "./OpenNR-IF/OpenNR-IF.dll"
    ip_address: str = "127.0.0.1"

    global_nr.LoadDLL(dllpath=dll_path)
    openId: int = global_nr.Open(ip_address)
    if openId < 0:
        raise RuntimeError(f"robot connection failed (error={openId})")
    print(f"connected (id={openId})")
    time.sleep(1)

    print("Servo ON")
    global_nr.CtrlMotor(1)
    sleepUntilServoOn()

    print("init pos (joint angle)")
    angle: list[float] = [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]
    global_nr.CtrlMoveJ(angle, len(angle), nType=1)
    sleepUntilRobotStopped()
    print("init pos done")

    print("shift move 1  (Y -10mm, Z +50mm)")
    pose: NR_POSE = NR_POSE(0.0, -10.0, 50.0, 0.0, 0.0, 0.0)
    global_nr.CtrlMoveXR(pose, nType=1)
    sleepUntilRobotStopped()

    print("shift move 2  (Y +10mm, Z -50mm)")
    pose = NR_POSE(0.0, 10.0, -50.0, 0.0, 0.0, 0.0)
    global_nr.CtrlMoveXR(pose, nType=1)
    sleepUntilRobotStopped()

    print("move 1")
    pose = NR_POSE(469.0, -79.0, 373.0, 29.0, 0.0, -175.0)
    global_nr.CtrlMoveX(pose, nType=1)
    sleepUntilRobotStopped()

    print("move 2")
    pose = NR_POSE(469.0, 0.0, 500.0, 29.0, 0.0, -175.0)
    global_nr.CtrlMoveX(pose, nType=1)
    sleepUntilRobotStopped()

    print("move 3")
    pose = NR_POSE(469.0, 79.0, 373.0, 29.0, 0.0, -175.0)
    global_nr.CtrlMoveX(pose, nType=1)
    sleepUntilRobotStopped()

    print("Servo OFF")
    global_nr.CtrlMotor(0)
    global_nr.Close()
    print("disconnected")


if __name__ == "__main__":
    main()
