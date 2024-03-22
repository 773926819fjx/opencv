from skimage import feature, exposure
from matplotlib import pyplot as plt
import cv2

image = cv2.imread('hw10\\pedestrain.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 4), visualize=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
print(hog_image_rescaled.shape)

plt.imshow(hog_image_rescaled, cmap=plt.get_cmap('gray'))
plt.savefig("hw10\\2.jpg")
plt.show()
