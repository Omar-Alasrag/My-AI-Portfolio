import cv2
import numpy as np
import torch as pt
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import torchvision.transforms as T
import imageio

reader = imageio.get_reader(r"video_directory_link")
writer = imageio.get_writer("output.mp4", fps=reader.get_meta_data()["fps"])


device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
COCO_INSTANCE_CATEGORY_NAMES = SSD300_VGG16_Weights.DEFAULT.meta["categories"]
transform = T.Compose([T.ToTensor(), T.Resize((300, 300))])
ssd = ssd300_vgg16(weights=SSD300_VGG16_Weights).eval().to(device)


def detect(frame):
    h, w, _ = frame.shape
    frame_t = transform(frame)
    frame_t = pt.FloatTensor(frame_t).unsqueeze(0).to(device)
    with pt.no_grad():
        detections = ssd(frame_t)[0]

    i = 0
    while detections["scores"][i] > 0.6:
        x, y, xmax, ymax = (
            (detections["boxes"][i] / 300 * pt.tensor([w, h, w, h]).to(device))
            .int()
            .tolist()
        )
        cv2.rectangle(frame, (x, y), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(
            frame,
            COCO_INSTANCE_CATEGORY_NAMES[detections["labels"][i]],
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        i += 1
    return frame


for i, frame in enumerate(reader):

    frame = detect(frame)

    writer.append_data(frame)
    print(i)


writer.close()
