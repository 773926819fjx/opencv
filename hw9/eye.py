import cv2

face_xml = cv2.CascadeClassifier('hw9\\haarcascade_frontalface_default.xml')
eye_aml = cv2.CascadeClassifier('hw9\\haarcascade_eye.xml')
img = cv2.imread('hw9\\face_eye_detection2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_xml.detectMultiScale(gray, 1.3, 5)
print("faces=", len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_face = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_aml.detectMultiScale(roi_face)
    print('eye=', len(eyes))
    for (e_x, e_y, e_w, e_h) in eyes:
        cv2.rectangle(roi_color, (e_x, e_y), (e_x + e_w, e_y + e_h),
                      (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.imwrite("hw9\\2.jpg", img)
cv2.waitKey(0)
