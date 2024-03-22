import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image = mpimg.imread(
    'C:\\Users\\fjx\\Desktop\\opencv\\hw3\\line_detection2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

low_threshold = 50
high_threshold = 100
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

mask = np.zeros_like(edges)
ignore_mask_color = 255
imshape = image.shape
vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290),
                      (imshape[1], imshape[0])]],
                    dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

rho = 1.5
theta = np.pi / 180
threshold = 10
min_line_length = 100
max_line_gap = 12
line_image = np.copy(image) * 0

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

color_edges = np.dstack((edges, edges, edges))

lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
cv2.imwrite('hw3\\2.png', lines_edges)
plt.imshow(lines_edges)
plt.show()
