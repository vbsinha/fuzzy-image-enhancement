import numpy as np
import imageio


def rmse(a, b):
    a = a.astype(float)
    b = b.astype(float)
    return np.sqrt((np.square(a-b)).mean())


def saveimg(path, img):
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    imageio.imwrite(path, img)
    return img
