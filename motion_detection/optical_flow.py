import cv2
import numpy as np
import matplotlib.pyplot as plt

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

motion_test_1 = '../cropped1.jpg'
motion_test_2 = '../cropped2.jpg'

img1 = cv2.imread(motion_test_1)
img2 = cv2.imread(motion_test_2)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
mask = np.zeros_like(img1)

color = np.random.randint(0, 255, (100, 3))

p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

good_new = p1[st == 1]
good_old = p0[st == 1]

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
    img2 = cv2.circle(img2, (int(a), int(b)), 5, color[i].tolist(), -1)
img = cv2.add(img2, mask)
img = cv2.resize(img, (1000, 1000))
cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()