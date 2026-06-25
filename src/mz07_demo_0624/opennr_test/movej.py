import time

from opennr_py import OpenNRIF, global_nr

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

        print("Move 1")
        angle = [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]
        global_nr.CtrlMoveJ(angle, len(angle), nType=1)
        sleepUntilRobotStopped()
        print("Move1 End")

        print("Move 2")
        angle = [45.0, 110.0, 10.0, 10.0, 10.0, 0.0]
        global_nr.CtrlMoveJ(angle, len(angle), nType=1)
        sleepUntilRobotStopped()
        print("Move2 End")

        print("Move 3")
        angle = [-45.0, 80.0, -10.0, -10.0, -10.0, 0.0]
        global_nr.CtrlMoveJ(angle, len(angle), nType=1)
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
