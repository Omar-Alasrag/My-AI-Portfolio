import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import \
    VisionTaskRunningMode
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult)

latest_result = None


def return_result(result, img, timestamp):
    global latest_result
    latest_result = result


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=r"tools\face_landmarker.task"),
    running_mode=VisionTaskRunningMode.LIVE_STREAM,
    result_callback=return_result,
)

cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        imgh, imgw, _ = img.shape
        mp_img = mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        detector.detect_async(mp_img, timestamp)

        if latest_result is not None and len(latest_result.face_landmarks) > 0:
            face_lms = latest_result.face_landmarks[0]

            l = face_lms[145]
            r = face_lms[374]

            lx, ly = int(l.x * imgw), int(l.y * imgh)
            rx, ry = int(r.x * imgw), int(r.y * imgh)

            cv2.circle(img, (lx, ly), 2, (0, 0, 255), -1)
            cv2.circle(img, (rx, ry), 2, (0, 0, 255), -1)

            # getting the focal length f which is 580
            # w = math.dist((lx, ly), (rx, ry))
            # f = 50 * (w / 6.3)

            # getting the distance d
            f = 580
            w = math.dist((lx, ly), (rx, ry))
            d = f * (6.3 / w)

            cv2.putText(
                img,
                f"distance is {int(d)} focal is {f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

        cv2.imshow("calibration", img)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
