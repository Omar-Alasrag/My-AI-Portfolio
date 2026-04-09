from datetime import datetime
import cv2
import numpy as np
import pytesseract
from ultralytics.engine.results import Results
from ultralytics.models import YOLO

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


model = YOLO("models/best.pt")

area = [(27, 417), (16, 456), (1015, 451), (992, 417)]


numbers = []
processed_numbers = set()
cap = cv2.VideoCapture("data/mycarplate.mp4")
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = model.predict(frame, verbose=False)
    detections: Results = detections[0]

    for box in detections.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().cpu().numpy()
        cls = int(box.cls[0])
        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)
        result = cv2.pointPolygonTest(np.array(area), (cx, cy), False)
        if result >= 0:
            plate = frame[y1:y2, x1:x2]
            plate_number = pytesseract.image_to_string(plate, config="--psm 7")
            plate_number = (
                plate_number.replace("(", "").replace(")", "").replace(",", "")
            )
            print(plate_number)

            if plate_number not in processed_numbers:
                processed_numbers.add(plate_number)
                numbers.append(plate_number)

                date = datetime.now().strftime(r"%y-%m-%d %H:%M:%S")
                with open("car_plate_data.txt", "a") as f:
                    f.write(f"{plate_number}\t{date}\n")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("plate", plate)

    cv2.polylines(frame, [np.array(area)], True, (0, 255, 0), 2)
    cv2.imshow("ocr", frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


