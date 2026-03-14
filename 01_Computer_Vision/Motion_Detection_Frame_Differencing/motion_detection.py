import cv2
import numpy as np

cap = cv2.VideoCapture("data/pedestrians.avi")
cw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
ch = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter().fourcc(*"mp4v")
wr = cv2.VideoWriter("out.mp4", fourcc, fps, (int(cw), int(ch)))

_, img1 = cap.read()
while cap.isOpened():
    ret, img2 = cap.read()
    if not ret:
        break
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, np.ones((3, 3)))

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    canvas = img1.copy()
    for c in contours:
        if cv2.contourArea(c) < 800:
            continue

        x, y, w, h = cv2.boundingRect(c)
        canvas = cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
        canvas = cv2.putText(
            canvas, "Movement", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
        )

    cv2.imshow("out", canvas)
    wr.write(canvas)

    if cv2.waitKey(50) == 27:
        break

    img1 = img2


cap.release()
wr.release()
cv2.destroyAllWindows()
