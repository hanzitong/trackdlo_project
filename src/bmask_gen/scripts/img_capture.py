
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
save_dir = ROOT / "data/captured"
save_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("cannot open camera !")

# 解像度設定（任意）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # sキーで保存
    if key == ord('s'):
        filename = save_dir / f"img_{count:04d}.png"
        cv2.imwrite(str(filename), frame)
        print("saved:", filename)
        count += 1

    # ESCで終了
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()




