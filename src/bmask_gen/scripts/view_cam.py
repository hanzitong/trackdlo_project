# =============================================================================
# view_cam.py
#
# カメラの映像をウィンドウに表示するだけの確認用スクリプト。
# カメラデバイスが正しく認識されているか、解像度や FPS の設定が
# 意図通り反映されているかを確認するために使う。
# =============================================================================

import cv2
import numpy as np

# ─── カメラを開く ────────────────────────────────────────────────────────
# cv2.VideoCapture(0): デバイス番号 0 = /dev/video0 を開く
# cv2.CAP_V4L2: Linux の Video4Linux2 バックエンドを明示的に指定する
#   指定しないと自動選択になりフォーマット設定が意図通り効かないことがある
cap: cv2.VideoCapture = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("カメラを開けませんでした")

# ─── フォーマット設定 ────────────────────────────────────────────────────
# FOURCC (Four Character Code): 動画/画像のピクセルフォーマットを識別する 4 文字コード。
#   YUYV: YUV 4:2:2 形式。カメラが生で出力する非圧縮フォーマット。
#         Y = 輝度 (luma)、U/V = 色差 (chroma)。
#         USB カメラのデフォルトが YUYV のことが多い。
#   MJPG: Motion JPEG。圧縮されているため転送データが小さく、高フレームレートに向く。
#
# cv2.VideoWriter_fourcc(*'YUYV'):
#   文字列 'YUYV' を 4 文字に分解して fourcc コードを生成する。
#   *'YUYV' は文字列をアンパック (= 'Y','U','Y','V' を個別引数として渡す)。
#   C++ で言えば: cv::VideoWriter::fourcc('Y','U','Y','V')
fourcc: int = cv2.VideoWriter_fourcc(*'YUYV')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# ─── 解像度・FPS 設定 ────────────────────────────────────────────────────
# cap.set() はあくまで「要求」であり、カメラがサポートしていない値は無視される。
# 実際に設定された値は cap.get() で取得して確認する。
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ─── 実際の設定値を取得して表示 ──────────────────────────────────────────
actual_fourcc: int   = int(cap.get(cv2.CAP_PROP_FOURCC))
actual_width:  int   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height: int   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps:    float = cap.get(cv2.CAP_PROP_FPS)

# FOURCC コードは 32 ビット整数として格納されている。
# 各バイトに 1 文字が格納されており、ビットシフトで取り出す。
#
# decode_fourcc の仕組み:
#   fourcc は 4 バイト整数 (例: 0x56595559 = 'V','Y','U','Y' の逆順など)。
#   v >> (8*i) でバイト単位にシフトし、& 0xFF で下位 8 ビット (1 文字) を取り出す。
#   chr() で ASCII コードを文字に変換。
#   C++ で言えば: char c = (v >> (8*i)) & 0xFF;
def decode_fourcc(v: int) -> str:
    """32 ビット FOURCC 整数を 4 文字の文字列に変換して返す。"""
    return "".join([chr((v >> 8*i) & 0xFF) for i in range(4)])

print("FOURCC:", decode_fourcc(actual_fourcc))
print("SIZE  :", actual_width, "x", actual_height)
print("FPS   :", actual_fps)

# ─── カメラ映像の表示ループ ──────────────────────────────────────────────
while True:
    ret: bool
    frame: np.ndarray   # shape (480, 640, 3), dtype uint8, BGR
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV は内部で YUYV → BGR に変換してから返す。
    # imshow もデフォルトで BGR として表示するので、そのまま渡せばよい。
    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC で終了
        break

cap.release()
cv2.destroyAllWindows()
