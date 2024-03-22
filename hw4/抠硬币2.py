import cv2
import numpy as np

img = cv2.imread('C:\\Users\\fjx\\Desktop\\opencv\\hw4\\circle_detection2.jpg')
# 将图像转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯滤波降噪
gaussian_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
# 利用Canny进行边缘检测
edges_img = cv2.Canny(gaussian_img, 80, 180, apertureSize=3)
# 自动检测圆
circles1 = cv2.HoughCircles(edges_img, cv2.HOUGH_GRADIENT, 1, 100, param1=300, param2=40, minRadius=60, maxRadius=130)

circles = circles1[0, :, :]
circles = np.uint16(np.around(circles))
for i in circles[:]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)

cv2.imwrite('hw4\\2.png', img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
