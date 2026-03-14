from tools.tracker import EuclideanDistTracker
import cv2

cap = cv2.VideoCapture("data/highway.mp4")
tracker = EuclideanDistTracker()
detector = cv2.createBackgroundSubtractorMOG2(100, 50)


counter = 0
line_pos = 150
counted_ids = set() 


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    roi = frame[340:720, 500:800]
    mask = detector.apply(roi)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    close = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, None)
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            detections.append(cv2.boundingRect(contour))

    boxes = tracker.update(detections)
    
    cv2.line(roi, (0, line_pos), (300, line_pos), (0, 0, 255), 2)

    for x, y, w, h, id in boxes:
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(roi, (cx, cy), 5, (0, 255, 255), -1)

        if cy > (line_pos - 6) and cy < (line_pos + 6):
            if id not in counted_ids:
                counter += 1
                counted_ids.add(id)

        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(roi, f"id:{id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, f"total cars: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("counting", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()