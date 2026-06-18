import time

from opennr_py import OpenNRIF, global_nr

DLL_PATH = "./OpenNR-IF/OpenNR-IF.dll"
IP_ADDRESS = "127.0.0.1"


def main():
    print("Hello from pyopennr!")
    bs = [False] * 10

    # OpenNRオープンとクローズ
    global_nr.LoadDLL(dllpath=DLL_PATH)
    openId = global_nr.Open(IP_ADDRESS)
    if openId < 0:
        print(f"open error: {openId}")
    else:
        print(f"open: {openId}")
        global_nr.AcsGeneralOutputSignal(bs, False, 51, 8)  # 信号51から8つ読み取り
        global_nr.AcsGeneralOutputSignal(bs, True, 61, 8)  # 読取信号8つを61から書込み
        global_nr.AcsGeneralOutputSignal([True], True, 50, 1)  # 信号50に書込み
        global_nr.Close()


if __name__ == "__main__":
    main()
