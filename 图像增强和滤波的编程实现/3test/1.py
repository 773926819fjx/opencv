import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = 'hw2\\3\\30.png'

img = cv2.imread(img_path, flags=1)

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

(h, s, v) = cv2.split(imgHSV)

ksize = (5, 5)
imgGaussBlurv = cv2.GaussianBlur(v, (101, 101), sigmaX=30)
img_ret = cv2.divide(v, imgGaussBlurv)
v_ret = cv2.normalize(img_ret, None, 0, 255, cv2.NORM_MINMAX)
v_ret = cv2.pow(v_ret, 2)
v_ret = cv2.normalize(v_ret, None, 0, 255, cv2.NORM_MINMAX)
v_ret = v_ret.astype(np.uint8)

result = cv2.merge((h, s, v_ret))
result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
cv2.imwrite('hw2\\3test\\30_1.png', result)
plt.figure(figsize=(9, 6))
plt.subplot(131), plt.axis('off'), plt.title('Original')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(132), plt.axis('off'), plt.title('shadow')
plt.imshow(cv2.cvtColor(imgGaussBlurv, cv2.COLOR_BGR2RGB))
plt.subplot(133), plt.axis('off'), plt.title('result')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()
