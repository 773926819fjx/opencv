import numpy as np
import cv2
from matplotlib import pyplot as plt

# 9.17: 陷波带阻滤波器消除周期噪声干扰
def butterworthNRFilter(img, radius=10, uk=10, vk=10, n=2):  # 巴特沃斯陷波带阻滤波器
    M, N = img.shape[1], img.shape[0]
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    Dm = np.sqrt((u - M//2 - uk)**2 + (v - N//2 - vk)**2)
    Dp = np.sqrt((u - M//2 + uk)**2 + (v - N//2 + vk)**2)
    D0 = radius
    n2 = 2 * n
    kernel = (1 / (1 + (D0 / (Dm + 1e-6))**n2)) * (1 / (1 + (D0 / (Dp + 1e-6))**n2))
    return kernel
# (1) 读取原始图像
img = cv2.imread("hw2\\4\\21.png", flags=0)  # flags=0 读取为灰度图像
imgFloat32 = np.float32(img)  # 将图像转换成 float32
rows, cols = img.shape[:2]  # 图片的高度和宽度
fig = plt.figure(figsize=(9, 6))
plt.subplot(231), plt.title("Original image"), plt.axis('off'), plt.imshow(img, cmap='gray')
# (2) 中心化, centralized 2d array f(x,y) * (-1)^(x+y)
mask = np.ones(img.shape)
mask[1::2, ::2] = -1
mask[::2, 1::2] = -1
fImage = imgFloat32 * mask  # f(x,y) * (-1)^(x+y)
# (3) 快速傅里叶变换
rPadded = cv2.getOptimalDFTSize(rows)  # 最优 DFT 扩充尺寸
cPadded = cv2.getOptimalDFTSize(cols)  # 用于快速傅里叶变换
dftImage = np.zeros((rPadded, cPadded, 2), np.float32)  # 对原始图像进行边缘扩充
dftImage[:rows, :cols, 0] = fImage  # 边缘扩充，下侧和右侧补0
cv2.dft(dftImage, dftImage, cv2.DFT_COMPLEX_OUTPUT)  # 快速傅里叶变换
dftAmp = cv2.magnitude(dftImage[:,:,0], dftImage[:,:,1])  # 傅里叶变换的幅度谱 (rPad, cPad)
dftAmpLog = np.log(1.0 + dftAmp)  # 幅度谱对数变换，以便于显示
dftAmpNorm = np.uint8(cv2.normalize(dftAmpLog, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
plt.subplot(232), plt.axis('off'), plt.title("DFT spectrum")
plt.imshow(dftAmpNorm, cmap='gray')
plt.arrow(445, 370, 25, 30, width=5, length_includes_head=True, shape='full')  # 在图像上加上箭头
plt.arrow(550, 490, -25, -30, width=5, length_includes_head=True, shape='full')  # 在图像上加上箭头
# (4) 构建陷波带阻滤波器 传递函数
BRFilter = butterworthNRFilter(dftImage, radius=15, uk=25, vk=16, n=3)  # 巴特沃斯陷波带阻滤波器, 处理周期噪声
plt.subplot(233), plt.axis('off'), plt.title("Butterworth notch resist filter")
plt.imshow(BRFilter, cmap='gray')
# (5) 在频率域修改傅里叶变换: 傅里叶变换 点乘 陷波带阻滤波器
dftFilter = np.zeros(dftImage.shape, dftImage.dtype)  # 快速傅里叶变换的尺寸(优化尺寸)
for i in range(2):
    dftFilter[:rPadded, :cPadded, i] = dftImage[:rPadded, :cPadded, i] * BRFilter
# 频域滤波傅里叶变换的傅里叶谱
nrfDftAmp = cv2.magnitude(dftFilter[:, :, 0], dftFilter[:, :, 1])  # 傅里叶变换的幅度谱
nrfDftAmpLog = np.log(1.0 + nrfDftAmp)  # 幅度谱对数变换，以便于显示
nrfDftAmpNorm = np.uint8(cv2.normalize(nrfDftAmpLog, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
plt.subplot(234), plt.axis('off'), plt.title("BNRF DFT Spectrum")
plt.imshow(nrfDftAmpNorm, cmap='gray')
# (6) 对频域滤波傅里叶变换 执行傅里叶逆变换，并只取实部
idft = np.zeros(dftAmp.shape, np.float32)  # 快速傅里叶变换的尺寸(优化尺寸)
cv2.dft(dftFilter, idft, cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE + cv2.DFT_SCALE)
# (7) 中心化, centralized 2d array g(x,y) * (-1)^(x+y)
mask2 = np.ones(dftAmp.shape)
mask2[1::2, ::2] = -1
mask2[::2, 1::2] = -1
idftCen = idft * mask2  # g(x,y) * (-1)^(x+y)
plt.subplot(235), plt.axis('off'), plt.title("g(x,y)*(-1)^(x+y)")
plt.imshow(idftCen, cmap='gray')
# (8) 截取左上角，大小和输入图像相等
idftCenClip = np.clip(idftCen, 0, 255)  # 截断函数，将数值限制在 [0,255]
imgFiltered = idftCenClip.astype(np.uint8)
imgFiltered = imgFiltered[:rows, :cols]
plt.subplot(236), plt.axis('off'), plt.title("BNRF filtered image")
plt.imshow(imgFiltered, cmap='gray')
plt.tight_layout()
plt.show()
print("image.shape:{}".format(img.shape))
print("imgFloat32.shape:{}".format(imgFloat32.shape))
print("dftImage.shape:{}".format(dftImage.shape))
print("dftAmp.shape:{}".format(dftAmp.shape))
print("idft.shape:{}".format(idft.shape))
print("dftFilter.shape:{}".format(dftFilter.shape))
print("imgFiltered.shape:{}".format(imgFiltered.shape))
