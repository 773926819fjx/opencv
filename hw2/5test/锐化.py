import numpy as np
import cv2
from matplotlib import pyplot as plt

# 1.78：图像锐化：拉普拉斯算子 (Laplacian)
img = cv2.imread("hw2\\5\\1.png", flags=0)
# 使用函数 filter2D 实现 Laplace 卷积算子
kernLaplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Laplacian kernel
imgLaplace1 = cv2.filter2D(img, -1, kernLaplace, borderType=cv2.BORDER_REFLECT)
# 使用 cv2.Laplacian 实现 Laplace 卷积算子
imgLaplace2 = cv2.Laplacian(img, -1, ksize=3)
imgRecovery = cv2.add(img, imgLaplace2)  # 恢复原图像
# 二值化边缘图再卷积
ret, binary = cv2.threshold(img, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
imgLaplace3 = cv2.Laplacian(binary, cv2.CV_64F)
imgLaplace3 = cv2.convertScaleAbs(imgLaplace3)
plt.figure(figsize=(9, 6))
plt.subplot(131), plt.axis('off'), plt.title("Original")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(132), plt.axis('off'), plt.title("cv.Laplacian")
plt.imshow(imgLaplace2, cmap='gray', vmin=0, vmax=255)
plt.subplot(133), plt.axis('off'), plt.title("thresh-Laplacian")
plt.imshow(imgLaplace3, cmap='gray', vmin=0, vmax=255)
plt.tight_layout()
plt.show()
