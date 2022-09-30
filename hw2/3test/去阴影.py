import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

img = cv2.imread('hw2\\3\\25.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fimg = np.log(np.abs(fshift))
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.subplot(121), plt.imshow(img, 'gray'), plt.title(u'(a)原始图像')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title(u'(b)傅里叶变换处理')
plt.axis('off')
plt.show()
