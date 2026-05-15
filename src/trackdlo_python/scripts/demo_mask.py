"""
Webカメラ映像に DeepLabV3+ でバイナリマスクを生成するデモ。
ESC キーで終了。

実行方法:
    # ワークスペースルートから
    uv run python src/trackdlo_python/scripts/demo_mask.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

# 重みファイルのパス (src/bmask_gen/weights/ を参照)
WEIGHTS = Path(__file__).resolve().parents[2] / "bmask_gen/weights/best_deeplabv3plus_cable.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# モデルのロード
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
print(f"model loaded  device={DEVICE}")

# カメラのオープン
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 推論ループ
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 前処理: BGR -> RGB, 正規化, CHW, バッチ次元追加
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.tensor(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(DEVICE)

    # 推論 -> バイナリマスク (0 or 1)
    with torch.no_grad():
        mask = (torch.sigmoid(model(x)) > 0.5)[0, 0].cpu().numpy().astype(np.uint8)

    cv2.imshow("camera", frame)
    cv2.imshow("mask", mask * 255)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC で終了
        break

cap.release()
cv2.destroyAllWindows()
