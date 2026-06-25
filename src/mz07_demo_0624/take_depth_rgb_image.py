
"""
BGR と depth 画像を RealSense D405 で取得して保存するデータ収集スクリプト。

depth は mm 単位 (uint16) に変換して保存する。
D405 の raw depth scale は 0.0001 m/unit (0.1mm 単位) であるため、
raw × depth_scale × 1000 で mm 換算してから cv2.imwrite する。

key controls:
  Space  BGR + depth PNG を保存
  ESC    終了

usage:
  cd src/mz07_demo && python take_depth_rgb_image.py
"""

import cv2
import pyrealsense2 as rs
import numpy as np

from pathlib import Path


SAVE_DIR: Path = Path("./nkr_data")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

pipeline: rs.pipeline = rs.pipeline()
cfg: rs.config = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile: rs.pipeline_profile = pipeline.start(cfg)

# depth scale を実機から取得 (D405 は通常 0.0001 m/unit = 0.1mm/unit)
depth_sensor: rs.depth_sensor = profile.get_device().first_depth_sensor()
DEPTH_SCALE: float = depth_sensor.get_depth_scale()
print(f"depth scale: {DEPTH_SCALE:.6f} m/unit")

count: int = 0

try:
    while True:
        frames: rs.composite_frame = pipeline.wait_for_frames()
        color_frame: rs.video_frame = frames.get_color_frame()
        depth_frame: rs.depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        bgr: np.ndarray       = np.asanyarray(color_frame.get_data())   # (480,640,3) uint8
        depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())   # (480,640) uint16 raw

        # raw → mm 換算 (保存データを mm 単位に統一する)
        depth_mm: np.ndarray = np.clip(
            depth_raw.astype(np.float32) * DEPTH_SCALE * 1000.0, 0, 65535
        ).astype(np.uint16)

        cv2.imshow("camera", bgr)

        key: int = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            bgr_path: Path   = SAVE_DIR / f"bgr_{count:04d}.png"
            depth_path: Path = SAVE_DIR / f"depth_{count:04d}.png"
            cv2.imwrite(str(bgr_path), bgr)
            cv2.imwrite(str(depth_path), depth_mm)
            print(f"saved: {bgr_path}  {depth_path}")
            count += 1

        if key == 27:   # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
