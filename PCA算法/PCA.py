import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("hw7\\Fig1138a.tif", flags=0)
height, width = img.shape[:2]
nBands = 6
snBands = ['a', 'b', 'c', 'd', 'e', 'f']
imgMulti = np.zeros((height, width, nBands))
Xmat = np.zeros((img.size, nBands))
print(imgMulti.shape, Xmat.shape)

for i in range(nBands):
    path = "hw7\\Fig1138{}.tif".format(snBands[i])
    imgMulti[:, :, i] = cv2.imread(path, flags=0)

m, p = Xmat.shape
Xmat = np.reshape(imgMulti, (-1, nBands))
mean, eigenvectors, eigenvalues = cv2.PCACompute2(Xmat,
                                                  np.empty(0),
                                                  retainedVariance=0.9)

print(mean.shape, eigenvectors.shape, eigenvalues.shape)
eigenvalues = np.squeeze(eigenvalues)

K = eigenvectors.shape[0]
print("number of samples:m=", m)
print("number of features:p=", p)
print("number of PCA features:k=", K)
print("mean:", mean.round(4))
print("topK eigenvalues:\n", eigenvalues.round(4))
print("topK eigenvectors:\n", eigenvectors.round(4))
mbMatPCA = cv2.PCAProject(Xmat, mean, eigenvectors)

fig2 = plt.figure(figsize=(9, 6))
fig2.suptitle("Principal component images")

for i in range(K):
    pca = mbMatPCA[:, i].reshape(-1, img.shape[1])
    imgPCA = cv2.normalize(pca, (height, width), 0, 255, cv2.NORM_MINMAX)
    ax2 = fig2.add_subplot(2, 3, i + 1)
    ax2.set_xticks([]), ax2.set_yticks([])
    ax2.imshow(imgPCA, 'gray')
    a = "hw7\\output\\"
    b = i + 1
    c = ".jpg"
    cv2.imwrite(a + str(b) + c, imgPCA)

plt.tight_layout()
plt.show()
