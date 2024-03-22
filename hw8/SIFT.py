import cv2

img1 = cv2.imread('hw8\\a1.jpg')
img2 = cv2.imread('hw8\\a2.jpg')

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

goodMatchs = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        goodMatchs.append(m)

pic3 = cv2.drawMatches(img1=img1,
                       keypoints1=kp1,
                       img2=img2,
                       keypoints2=kp2,
                       matches1to2=goodMatchs,
                       outImg=None)
cv2.imwrite('hw8\\compare.jpg', pic3)
