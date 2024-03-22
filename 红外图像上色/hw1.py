import cv2

videoCapture = cv2.VideoCapture(
    'C:\\Users\\fjx\\Desktop\\opencv\\hw1\\编程作业1-红外图像上色.mp4')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


def Color(img):
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def writevideo(imgs):
    videoname = 'C:\\Users\\fjx\\Desktop\\opencv\\hw1\\test.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(videoname, fourcc, fps, size, True)
    for img in imgs:
        writer.write(img)
    writer.release()


success, frame = videoCapture.read()
imgs = []
while success:
    frame = Color(frame)
    imgs.append(frame)
    success, frame = videoCapture.read()
writevideo(imgs)

videoCapture.release()
