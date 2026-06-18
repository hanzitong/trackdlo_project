
import pyrealsense2 as rs
import numpy as np
import cv2


pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)


try:
    for i in range(30):
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        print("ok")
finally:
    pipeline.stop()

