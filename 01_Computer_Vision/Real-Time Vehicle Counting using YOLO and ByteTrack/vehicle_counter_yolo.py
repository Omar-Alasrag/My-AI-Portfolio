from ultralytics.models import YOLO
from ultralytics.engine.results import Results
import numpy as np
import cv2

model = YOLO("yolo26n.pt")

y = 308
offset = 15

counted_cars = set()
cap = cv2.VideoCapture("data/traffictrim.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.line(frame, (0, y), (1000, y), (0, 255, 0), 2)
    detections = model.track(
        frame, persist=True, classes=[2], tracker="bytetrack.yaml", verbose=False
    )
    detections: Results = detections[0]
    boxes = detections.boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].xyxy.detach().cpu().numpy().astype(int)[0]
        cls = int(boxes[i].cls)
        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)
        id = int(boxes[i].id.item()) if boxes[i].id else -1

        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), 2)
        cv2.putText(
            frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2
        )

        if y - offset < cy < y + offset:
            if id not in counted_cars:
                counted_cars.add(id)

    cv2.putText(
        frame,
        f"counter is {len(counted_cars)}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        2,
    )

    cv2.imshow("counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

