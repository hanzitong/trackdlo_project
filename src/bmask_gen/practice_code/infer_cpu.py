
from pathlib import Path

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp


# Prepare model
model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
    encoder_name = "resnet34",
    encoder_weights = None,
    in_channels = 3,
    classes = 1,
    activation = None,
).to("cpu")

ROOT: Path = Path(__file__).resolve().parent.parent
model.load_state_dict(
    torch.load(
        ROOT / "weights_aoyama/best_deeplabv3plus_cable.pth",
        map_location = "cpu",
    )
)

model.eval()
print("model loaded")



# Realsense settings
pipeline: rs.pipeline = rs.pipeline()
cfg: rs.config = rs.config()
# cfg.setting(color, 480, 640, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)    # select config object
pipeline.start(cfg)


try:
    while True:
        # get and process RGB image from realsense pipeline
        frames: rs.composite_frame = pipeline.wait_for_frames()
        color_frame: rs.video_frame = frames.get_color_frame()
        bgr_image: np.ndarray = np.asanyarray(color_frame.get_data())
        rgb_image: np.ndarray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        x: np.ndarray = rgb_image.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # reshape: numpy_manner(H, W, C) ==> torch_manner(C, H, W)

        # x_t is the torch-fomat to input the model
        x_t: torch.Tensor = torch.tensor(x).unsqueeze(0)    # torch requires (N, C, H, W), N is batch size

        with torch.no_grad():
            prob: np.ndarray = torch.sigmoid(model(x_t))[0, 0].numpy()  # convert torch.Tensor to np.ndarray

        mask: np.ndarray = (prob > 0.5).astype(np.uint8)
        print(f"mask pixels (cable): {mask.sum()}", end="\r")

        overlay: np.ndarray = bgr_image.copy()
        overlay[mask == 1] = (0, 200, 0)
        result: np.ndarray = cv2.addWeighted(bgr_image, 0.6, overlay, 0.4, 0)

        cv2.imshow("camera", bgr_image)
        cv2.imshow("mask (green = cable)", result)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()






