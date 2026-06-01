# =============================================================================
# infer_live_cpu.py
#
# 学習済み DeepLabV3+ を CPU で実行してリアルタイムにケーブルを検出する。
# GPU 版は infer_live_gpu.py を参照。
#
# CPU 版の特徴:
#   - GPU / CUDA ドライバが不要。どの環境でも動く。
#   - GPU 版より推論が遅い。640x480 の場合は数秒/フレームになることがある。
#   - テンソルは最初から CPU にあるため .cpu() の呼び出しが不要。
#
# =============================================================================

from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent

device = "cpu"
print("device =", device)

# ─── モデルのロード ──────────────────────────────────────────────────────
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)
# CPU 版では .to(device) は厳密には不要 (テンソルのデフォルトが CPU) だが、
# GPU 版との対称性のために記述しておく。
model = model.to(device)

# map_location="cpu": GPU で保存したモデルを CPU でロードする際に
# GPU メモリアドレスを CPU に自動で再配置する。
model.load_state_dict(
    torch.load(ROOT / "weights/best_deeplabv3plus_cable.pth", map_location="cpu")
)

model.eval()
print("model loaded")

# ─── カメラのオープン ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("カメラを開けませんでした")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ─── 推論ループ ──────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # ── 前処理 (学習時と完全に同じ手順) ─────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))              # (H,W,C) → (C,H,W)
    x = torch.tensor(x).unsqueeze(0)            # (C,H,W) → (1,C,H,W)
    # GPU 版と異なり .to(device) は省略 (テンソルは最初から CPU にある)

    # ── 推論 ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        # CPU 版ではテンソルが最初から CPU にあるため .cpu() 呼び出しが不要。
        prob = torch.sigmoid(model(x))[0, 0].numpy()

    # ── マスク生成・表示 ──────────────────────────────────────────────────
    mask = (prob > 0.5).astype(np.uint8)
    print(f"mask pixels (cable): {mask.sum()}", end="\r")

    overlay = frame.copy()
    overlay[mask == 1] = (0, 200, 0)
    result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    cv2.imshow("camera", frame)
    cv2.imshow("mask (green = cable)", result)

    if cv2.waitKey(1) & 0xFF == 27:   # ESC で終了
        break

cap.release()
cv2.destroyAllWindows()
