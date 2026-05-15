
import cv2

# デバイス指定（通常は /dev/video0）
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("カメラを開けませんでした")

# --- フォーマット設定（YUYV） ---
fourcc = cv2.VideoWriter_fourcc(*'YUYV')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# --- 解像度設定 ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- FPS（任意） ---
cap.set(cv2.CAP_PROP_FPS, 30)

# 実際の設定確認
actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps    = cap.get(cv2.CAP_PROP_FPS)

def decode_fourcc(v):
    return "".join([chr((v >> 8*i) & 0xFF) for i in range(4)])

print("FOURCC:", decode_fourcc(actual_fourcc))
print("SIZE  :", actual_width, "x", actual_height)
print("FPS   :", actual_fps)

# --- 取得ループ ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 表示（OpenCVが内部でBGRに変換する場合あり）
    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC終了
        break

cap.release()
cv2.destroyAllWindows()


