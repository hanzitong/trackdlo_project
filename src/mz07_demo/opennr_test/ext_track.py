# 外部トラッキング
#
# 外部トラッキングの取説を読むこと、OpenNR-IFとは接続方法がPython/FDともに違う。
#

import time

from opennr_py import (
    NR_ACCESS_WAIT,
    NR_DATA_REAL,
    NR_GET_REAL_DATA_ALL,
    NR_MAX_AXIS_STD,
    NR_SET_CTRL_INFO,
    NR_SET_REAL_DATA_ALL,
    OpenNRIF,
    global_nr,
)

DLL_PATH = "./OpenNR-IF/OpenNR-IF.dll"
IP_ADDRESS = "192.168.1.150"
IP_PORT = 10081


# 外部トラッキングテスト
def ext_track_test():
    for n in range(2):  # 2サイクル
        # NR GetAll
        pnr_get_real_data_all = NR_GET_REAL_DATA_ALL()  # インスタンス作成
        nErr1 = global_nr.GetAll(
            [pnr_get_real_data_all], NR_ACCESS_WAIT
        )  # `NR_ACCESS_WAIT`とすることでクライアントからの応答を待ち続ける
        if nErr1 < 0:
            print("NR_GetAll error")
            print(nErr1)
            break
        else:
            print("NR_GetAll")
            for i in range(NR_MAX_AXIS_STD):
                print(float(pnr_get_real_data_all.ustData.stStd.fCurAngle[i]))
        # NR Set All
        nr_set_ctrl_info = NR_SET_CTRL_INFO()
        nr_set_ctrl_info.ushEstopBit = 0
        nr_set_ctrl_info.ushFinishBit = 1 if n == 1 else 0
        nr_set_ctrl_info.ushOrderBit = 1  # 0:TCP 1:各軸
        nr_set_ctrl_info.ushProtcolBit = (
            pnr_get_real_data_all.stCtrl.ushProtcolBit
        )  # ushProtcolBit=1で固定してください
        pnr_set_real_data_all = NR_SET_REAL_DATA_ALL()  # インスタンス作成
        pnr_set_real_data_all.stCtrl = nr_set_ctrl_info
        for i in range(NR_MAX_AXIS_STD):
            pnr_set_real_data_all.ustData.stStd.fComAngle[i] = (
                pnr_get_real_data_all.ustData.stStd.fCurAngle[i]
            )
        # CtrlMoveはコントローラに通過点の指示を行い、コントローラが補間点演算を自動で行います。（通過点の間を補間周期ごとに分割していくイメージ）
        # そのため現在の位置から大きく離れた位置指令を出したとしても問題ありません。
        # 一方でSetAllでは直接補間点を指示するため現在位置と離れた位置指令を出すことは暴走に繋がります。（ゲインは定数で与えるため）
        # できる限り細かく指令位置を変化させてください。
        pnr_set_real_data_all.ustData.stStd.fComAngle[0] += 0.1  # J1軸を0.1degずつ回転
        nErr2 = global_nr.SetAll(pnr_set_real_data_all, NR_ACCESS_WAIT)
        if nErr2 < 0:
            print("NR_SetAll error")
        else:
            print("NR_SetAll")
            for i in range(NR_MAX_AXIS_STD):
                print(float(pnr_set_real_data_all.ustData.stStd.fComAngle[i]))

    # 取説より、SetAllで終了ビットを投げてから100ms以上は待ってからNR_Closeを行う
    time.sleep(0.1)


def main():
    print("Hello from pyopennr!")

    # OpenNRオープン
    global_nr.LoadDLL(dllpath=DLL_PATH)
    openId = global_nr.Open(
        IP_ADDRESS, lPortNo=IP_PORT, lKind=NR_DATA_REAL
    )  # 外部トラッキングのため`NR_DATA_REAL`を指定
    if openId < 0:
        print(f"open error: {openId}")
    else:
        print(f"open: {openId}")

        # 外部トラッキングテスト
        print("ext track test")
        ext_track_test()

        # クローズ
        global_nr.Close()


if __name__ == "__main__":
    main()
