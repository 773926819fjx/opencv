import cv2
import numpy as np
import math


def stretchImage(data, s=0.005, bins=2000):
    ht = np.histogram(data, bins)
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)


g_para = {}


def getPara(radius=5):
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h**2 + w**2)
    m /= m.sum()
    g_para[radius] = m
    return m


def zmIce(I, ratio=4, radius=300):
    para = getPara(radius)
    height, width = I.shape
    zh, zw = [0] * radius + [x for x in range(height)
                             ] + [height - 1] * radius, [0] * radius + [
                                 x for x in range(width)
                             ] + [width - 1] * radius
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip(
                (I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res


def zmIceFast(I, ratio, radius):
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, ((width + 1) // 2, (height + 1) // 2))
    Rf = zmIceFast(Rs, ratio, radius)
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)


def zmIceColor(I, ratio=4, radius=3):
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res


if __name__ == '__main__':
    m = zmIceColor(cv2.imread('hw2\\1\\6.png') / 255.0) * 255
    cv2.imwrite('hw2\\2test\\6test.png', m)
