

# cascades link
# https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")

vc = cv2.VideoCapture(0)

while True:
    correct, frame = vc.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = frame[y : y + h, x : x + w]
        gray = gray[y : y + h, x : x + w]
        eyes = eye_detector.detectMultiScale(gray, 1.1, 22)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)



    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xff == ord("q") :
        break

    
vc.release()
cv2.destroyAllWindows()
