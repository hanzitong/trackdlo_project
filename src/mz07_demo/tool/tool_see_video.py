

import pyrealsense2 as rs
import numpy as np
import cv2



pipeline: rs.pipeline = rs.pipeline()
cfg: rs.config = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile: rs.pipeline_profile = pipeline.start(cfg)

try:
    # while True:
        frames: rs.composite_frame = pipeline.wait_for_frames()
        color_frame: rs.video_frame = frames.get_color_frame()

        # if not color_frame:
        #     continue

        bgr: np.ndarray = np.asanyarray(color_frame.get_data())
        cv2.imshow("camera bgr", bgr)
        # cv2.imshow("camera color frame", color_frame.get_data())

        key: int = cv2.waitKey(1) & 0xFF

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
