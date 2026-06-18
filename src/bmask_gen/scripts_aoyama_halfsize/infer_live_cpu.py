# =============================================================================
# infer_live_cpu.py  (scripts_aoyama_halfsize/)
#
# RealSense D405 (320×240) + CPU で DeepLabV3+ をリアルタイム推論する。
#
# 事前準備: train.py で weights_halfsize/best_deeplabv3plus_mobilenetv2.pth を生成すること。
#
# 使い方 (ワークスペースルートから):
#   uv run python src/bmask_gen/scripts_aoyama_halfsize/infer_live_cpu.py
# =============================================================================

from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import segmentation_models_pytorch as smp

ROOT: Path = Path(__file__).resolve().parent.parent

device: str = "cpu"
WEIGHTS_PATH: Path = ROOT / "weights_halfsize/best_deeplabv3plus_mobilenetv2.pth"

model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(device)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()
print(f"model loaded: {WEIGHTS_PATH}")

# ─── RealSense D405 パイプライン ──────────────────────────────────────────
pipeline: rs.pipeline = rs.pipeline()
cfg: rs.config = rs.config()
# 320×240 はネイティブ解像度なのでリサイズ不要
cfg.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 30)
pipeline.start(cfg)
print("RealSense pipeline started")

try:
    while True:
        frames: rs.composite_frame = pipeline.wait_for_frames()
        color_frame: rs.video_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # フレーム取得: (240, 320, 3) BGR uint8
        frame: np.ndarray = np.asanyarray(color_frame.get_data())

        # ── 前処理 (train.py の __getitem__ と完全に同じ手順・数値) ─────────
        rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x: np.ndarray   = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))           # HWC → CHW (3,240,320)
        x_t: torch.Tensor = torch.tensor(x).unsqueeze(0)  # → (1,3,240,320)

        # ── 推論 ─────────────────────────────────────────────────────────
        with torch.no_grad():
            # 出力: (1,1,240,320) logit → sigmoid → 確率マップ
            prob: np.ndarray = torch.sigmoid(model(x_t))[0, 0].numpy()

        # ── マスク生成・表示 ──────────────────────────────────────────────
        mask: np.ndarray = (prob > 0.5).astype(np.uint8)
        print(f"cable pixels: {mask.sum()}", end="\r")

        overlay: np.ndarray = frame.copy()
        overlay[mask == 1] = (0, 200, 0)
        result: np.ndarray = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        cv2.imshow("camera", frame)
        cv2.imshow("mask (green = cable)", result)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC で終了
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
