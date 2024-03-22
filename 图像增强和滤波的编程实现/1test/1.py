import cv2
import numpy as np
import os

img_path = 'hw2\\1\\24.jpg'
img = cv2.imread(img_path)
(r, g, b) = cv2.split(img)
normImg = lambda x: 255. * (x - x.min()) / (x.max() - x.min() + 1e-6)
imgGamma = np.power(r, 2.5)
r = np.uint8(normImg(imgGamma))

imgGamma = np.power(g, 2.5)
g = np.uint8(normImg(imgGamma))

imgGamma = np.power(b, 2.5)
b = np.uint8(normImg(imgGamma))

result = cv2.merge((r, g, b))

cv2.imshow('input', img)
cv2.imshow('result', result)
cv2.imwrite('hw2\\1test\\24_1.png', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
