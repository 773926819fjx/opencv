import cv2
import numpy as np

img = cv2.imread("1111.png")
img_back = cv2.imread("1.jpg")
rows, cols, channels = img_back.shape
img_back = cv2.resize(img_back, None, fx=0.7, fy=0.7)
rows, cols, channels = img.shape
img = cv2.resize(img, None, fx=0.4, fy=0.4)
rows, cols, channels = img.shape
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([78, 43, 46])
upper_blue = np.array([110, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
erode = cv2.erode(mask, None, iterations=1)
dilate = cv2.dilate(erode, None, iterations=1)
center = [50, 50]
for i in range(rows):
    for j in range(cols):
        if dilate[i, j] == 0:
            img_back[center[0] + i, center[1] + j] = img[i, j]
imgCrop = img_back[150:230, 150:230].copy()
img1 = cv2.resize(imgCrop, (250, 250))
img2 = np.hstack((img1, img1))
img3 = np.hstack((img2, img1))
img4 = np.vstack((img3, img3))
img5 = np.vstack((img4, img3))
cv2.imwrite("new.jpg", img5)
cv2.imshow("ras", img5)
cv2.imshow('res', img_back)
cv2.waitKey(0)
cv2.destroyAllWindows()
