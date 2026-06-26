
"""
BGR と depth 画像を RealSense D405 で取得して保存するスクリプト。

起動すると自動で 1 枚撮影して nkr_data/ に保存し、終了する。
カメラの自動露光が安定するまで数フレームのウォームアップを行う。

depth は mm 単位 (uint16) に変換して保存する。
D405 の raw depth scale は 0.0001 m/unit (0.1mm 単位) であるため、
raw × depth_scale × 1000 で mm 換算してから cv2.imwrite する。

usage:
  cd src/mz07_demo_0624
  python take_depth_rgb_image.py
"""

import cv2
import pyrealsense2 as rs
import numpy as np

from pathlib import Path


SAVE_DIR: Path = Path(__file__).resolve().parent / "nkr_data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_FRAMES: int = 10  # 自動露光が安定するまで捨てるフレーム数

pipeline: rs.pipeline = rs.pipeline()
cfg: rs.config = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 640x480 px, 30 fps
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)   # 640x480 px, 30 fps

profile: rs.pipeline_profile = pipeline.start(cfg)

depth_sensor: rs.depth_sensor = profile.get_device().first_depth_sensor()
DEPTH_SCALE: float = depth_sensor.get_depth_scale()
print(f"depth scale: {DEPTH_SCALE:.6f} m/unit")
print(f"save dir: {SAVE_DIR}")

try:
    # 自動露光安定のためウォームアップ
    for i in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()
    print(f"warmup done ({WARMUP_FRAMES} frames)")

    frames: rs.composite_frame = pipeline.wait_for_frames()
    color_frame: rs.video_frame = frames.get_color_frame()
    depth_frame: rs.depth_frame = frames.get_depth_frame()

    bgr: np.ndarray       = np.asanyarray(color_frame.get_data())   # (480,640,3) uint8
    depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())   # (480,640) uint16 raw

    # raw から mm 単位に変換 (depth_scale の単位は m/unit)
    depth_mm: np.ndarray = np.clip(
        depth_raw.astype(np.float32) * DEPTH_SCALE * 1000.0,  # m to mm 変換
        0, 65535
    ).astype(np.uint16)

    # 既存ファイルと番号が衝突しないよう、空いている番号を探す
    count: int = 0
    while (SAVE_DIR / f"bgr_{count:04d}.png").exists():
        count += 1

    bgr_path: Path   = SAVE_DIR / f"bgr_{count:04d}.png"
    depth_path: Path = SAVE_DIR / f"depth_{count:04d}.png"
    cv2.imwrite(str(bgr_path), bgr)
    cv2.imwrite(str(depth_path), depth_mm)
    print(f"saved: {bgr_path}  {depth_path}")

finally:
    pipeline.stop()
