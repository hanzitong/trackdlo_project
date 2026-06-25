

import cv2
import numpy as np
import pyrealsense2 as rs


pipeline: rs.pipeline = rs.pipeline()
cfg: rs.config = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(cfg)
frames = pipeline.wait_for_frames()
color_frame: rs.frame = frames.get_color_frame()
ts: float = color_frame.get_timestamp()
print(f"colo frame timestamp: {ts}")    # ms

ts_domain = color_frame.frame_timestamp_domain
print(f"colo frame timestamp domain: {ts_domain}")    # ms

pipeline.stop()

if not color_frame:
    raise RuntimeError("couldn't get color frame")

color_image = np.asanyarray(color_frame.get_data())

cv2.imshow("capture", color_image)
cv2.waitKey(0)  # push any key to continue. code will be waiting here
cv2.destroyAllWindows()


