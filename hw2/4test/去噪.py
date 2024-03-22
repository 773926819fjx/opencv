import cv2
from matplotlib import pyplot as plt

# Reading image from folder where it is stored
img = cv2.imread('hw2\\4\\8.png', flags=1)

# denoising of image saving it into dst image
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

# Plotting of source and destination image
cv2.imwrite('hw2\\4test\\8_1.png', dst)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(dst)

plt.show()
