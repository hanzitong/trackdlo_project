

import numpy as np
import cv2
import pyrealsense2 as rs
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fps: int
    output_path: str



def main() -> None:
    camera_config = CameraConfig(
        width = 1280,
        height = 720,
        fps = 30,
        output_path = "demo_mean_cablepix.mp4"
    )

    pipeline: rs.pipeline = rs.pipeline()
    config: rs.config = rs.config()
    config.enable_stream(
        rs.stream.color,
        camera_config.width,
        camera_config.height,
        rs.format.bgr8,
        camera_config.fps,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        "demo_mean_cablepix_close.mp4",
        fourcc,
        30.0,
        (camera_config.width, camera_config.height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {camera_config.output_path}")

    pipeline.start(config)

    try: 
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image: np.ndarray = np.asanyarray(color_frame.get_data())

            writer.write(color_image)

            cv2.imshow("RealSense image (q: quit)", color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()


