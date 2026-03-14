import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks, DrawingSpec 
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmark,
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    PoseLandmarksConnections
    
)


def calc_angle(a, b, c):
    radian = np.abs(
        np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    degree = radian * 180 / np.pi
    if degree > 180:
        degree = 360 - degree
    return degree


options = PoseLandmarkerOptions(
    BaseOptions(r"tools\pose_landmarker_lite.task"), VisionTaskRunningMode.VIDEO
)

cap = cv2.VideoCapture(0)

left = [PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST]

flag = "down"
counter = 0
with PoseLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        _, frame = cap.read()
        h, w, c = frame.shape
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        print(ts)
        frame_mp = mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection: PoseLandmarkerResult = detector.detect_for_video(frame_mp, int(ts))
        if not detection.pose_landmarks:
            continue
        lms = detection.pose_landmarks[0]
        angle = calc_angle(*[(int(lms[i].x * w), int(lms[i].y * h)) for i in left])
        if angle < 40 and flag == "up":
            counter += 1
            flag = "down"

        if angle > 110:
            flag = "up"
        draw_landmarks(frame, detection.pose_landmarks[0], PoseLandmarksConnections.POSE_LANDMARKS)
        cv2.putText(
            frame,
            f"angle-> {angle:0.2f}, counter {counter}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("pose lms", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
