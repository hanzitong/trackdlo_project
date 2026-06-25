import time

from opennr_py import OpenNRIF, global_nr

DLL_PATH = "./OpenNR-IF/OpenNR-IF.dll"
IP_ADDRESS = "127.0.0.1"


def main():
    print("Hello from pyopennr!")
    val = 0

    # OpenNRオープンとクローズ
    global_nr.LoadDLL(dllpath=DLL_PATH)
    openId = global_nr.Open(IP_ADDRESS)
    if openId < 0:
        print(f"open error: {openId}")
    else:
        print(f"open: {openId}")
        global_nr.AcsCPULoad(val)  # CPU使用率読み取り
        print(f"CPU Load: {val}")
        global_nr.Close()


if __name__ == "__main__":
    main()
