
"""
This script is an example explaining how to use OpenNR
OpenNR is Nachi's original library.



"""


import time

from opennr_py import NR_POSE, OpenNRIF, global_nr

DLL_PATH = "./OpenNR-IF/OpenNR-IF.dll"
IP_ADDRESS = "127.0.0.1"


def main():
    print("Hello from pyopennr!")
    # Global
    global_nr.LoadDLL(dllpath=DLL_PATH)
    openId = global_nr.Open(IP_ADDRESS)
    if openId < 0:
        print(f"open error: {openId}")
    else:
        print(f"open: {openId}")
        time.sleep(1)  # 秒

        print("Servo ON")
        global_nr.CtrlMotor(1)
        sleepUntilServoOn()

        print("init pos")
        angle = [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]
        global_nr.CtrlMoveJ(angle, len(angle), nType=1)
        sleepUntilRobotStopped()
        print("init pos End")

        print("Shift Move 1")
        pose = NR_POSE(0.0, -10.0, 50.0, 0.0, 0.0, 0.0)
        global_nr.CtrlMoveXR(pose, nType=1)
        sleepUntilRobotStopped()
        print("Shift Move1 End")

        print("Shift Move 2")
        pose = NR_POSE(0.0, 10.0, -50.0, 0.0, 0.0, 0.0)
        global_nr.CtrlMoveXR(pose, nType=1)
        sleepUntilRobotStopped()
        print("Shift Move2 End")

        print("Move 1")
        pose = NR_POSE(469.0, -79.0, 373.0, 29.0, 0.0, -175.0)
        global_nr.CtrlMoveX(pose, nType=1)
        sleepUntilRobotStopped()
        print("Move1 End")

        print("Move 2")
#        pose = NR_POSE(550.0, -240.0, 750.0, 0.0, 0.0, 179.0)
        pose = NR_POSE(469.0, 0.0, 500.0, 29.0, 0.0, -175.0)
        global_nr.CtrlMoveX(pose, nType=1)
        sleepUntilRobotStopped()
        print("Move2 End")

        print("Move 3")
#        pose = NR_POSE(550.0, -240.0, 500.0, 0.0, 0.0, 179.0)
        pose = NR_POSE(469.0, 79.0, 373.0, 29.0, 0.0, -175.0)
        global_nr.CtrlMoveX(pose, nType=1)
        sleepUntilRobotStopped()
        print("Move3 End")

        print("Servo OFF")
        global_nr.CtrlMotor(0)
        print("close")
        global_nr.Close()


def sleepUntilServoOn():
    isServoOn = [False]
    timeout = 10
    counter = 0
    while True:
        time.sleep(1)
        counter = counter + 1
        if counter > timeout:
            break  # タイムアウト
        global_nr.AcsFixedIOServoOn(isServoOn)
        if isServoOn[0]:  # サーボON待ち
            break


def sleepUntilRobotStopped():
    speed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    while True:
        time.sleep(0.5)
        global_nr.AcsAxisOrderSpeed(speed, 1, len(speed))
        if all(abs(x) < 0.001 for x in speed):  # 停止待ち
            break
    time.sleep(0.1)


if __name__ == "__main__":
    main()





