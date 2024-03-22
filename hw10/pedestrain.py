import cv2
import matplotlib.pyplot as plt

image = cv2.imread('hw10\\pedestrain.jpg')

hog = cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

rects, weights = hog.detectMultiScale(image)

for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

plt.imshow(image[:, :, ::-1])
cv2.imwrite("hw10\\1.jpg", image)
plt.show()
