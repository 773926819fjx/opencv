import cv2
import numpy as np
import skimage
import matplotlib
from matplotlib import pyplot as plt

img_original = cv2.imread('hw2\\3\\28.jpg')
img_original_standard = img_original / 255
#添加高斯噪声，函数返回的归一化的多维数组
img_noise = skimage.util.random_noise(img_original_standard, mode='gaussian')
#高斯滤波
img_blur = cv2.GaussianBlur(img_noise, (127, 127), 15)
imgs = img_blur
img_original=img_original.astype(np.float32)
img_blur=img_blur.astype(np.float32)
img = cv2.divide(img_blur,img_original )
cv2.imshow('Images', img)
cv2.waitKey()
cv2.destroyAllWindows()
