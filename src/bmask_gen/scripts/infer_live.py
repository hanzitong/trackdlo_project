from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent

# ── モデルのロード ──────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", device)

model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(device)

model.load_state_dict(
    torch.load(ROOT / "weights/best_deeplabv3plus_cable.pth", map_location=device)
)
model.eval()
print("model loaded")

# ── カメラのオープン ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("カメラを開けませんでした")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── 推論ループ ──────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # 前処理（学習時と同じ手順：640x480 のまま渡す）
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))                  # HWC → CHW
    x = torch.tensor(x).unsqueeze(0).to(device)     # バッチ次元を追加

    # 推論
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    # マスク生成（0.5 閾値）
    mask = (prob > 0.5).astype(np.uint8)

    print(f"mask pixels (cable): {mask.sum()}", end="\r")

    # マスクを緑色オーバーレイに変換して視覚化
    overlay = frame.copy()
    overlay[mask == 1] = (0, 200, 0)                # ケーブル部分を緑に
    result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    cv2.imshow("camera", frame)
    cv2.imshow("mask (green = cable)", result)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC で終了
        break

cap.release()
cv2.destroyAllWindows()
