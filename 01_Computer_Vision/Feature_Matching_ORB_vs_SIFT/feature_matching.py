import cv2
import numpy as np

img1 = cv2.imread(r"data\matches\box1.png")
img2 = cv2.imread(r"data\matches\box2.png")

matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING)
orb = cv2.ORB.create()

kps1_o, des1_o = orb.detectAndCompute(img1, None)
kps2_o, des2_o = orb.detectAndCompute(img2, None)

matches_orb = matcher_orb.knnMatch(des1_o, des2_o, k=2)

good_orb = []
for m, n in matches_orb:
    if m.distance < 0.75 * n.distance:
        good_orb.append([m])

good_orb = sorted(good_orb, key=lambda x: x[0].distance)

img_orb = cv2.drawMatchesKnn(
    img1,
    kps1_o,
    img2,
    kps2_o,
    good_orb[:10],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv2.imshow("matches", img_orb)
cv2.waitKey(0)

matcher_sift = cv2.BFMatcher(cv2.NORM_L2)
sift = cv2.SIFT.create()

kps1_s, des1_s = sift.detectAndCompute(img1, None)
kps2_s, des2_s = sift.detectAndCompute(img2, None)

matches_sift = matcher_sift.knnMatch(des1_s, des2_s, k=2)

good_sift = []
for m, n in matches_sift:
    if m.distance < 0.7 * n.distance:
        good_sift.append([m])

good_sift = sorted(good_sift, key=lambda x: x[0].distance)

img_sift = cv2.drawMatchesKnn(
    img1,
    kps1_s,
    img2,
    kps2_s,
    good_sift[:10],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv2.imshow("matches", img_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()
