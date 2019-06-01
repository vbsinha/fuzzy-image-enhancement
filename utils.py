import os

import numpy as np
import imageio

from fuzzysets import small, medium, large


def rmse(a, b):
    a = a.astype(float)
    b = b.astype(float)
    return np.sqrt((np.square(a-b)).mean())


def saveimg(path, img):
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    imageio.imwrite(path, img)
    return img


def normalize_and_save(path, img):
    """ Normalize the image to [0, 1] and then
    scale to [0 255] before saving """
    img_min, img_max = np.amin(img), np.amax(img)
    img = (img - img_min) / (img_max - img_min)
    img *= 255
    saveimg(path, img)


def plot(tc, imagepath):
    l = len(tc)
    sm_img = np.zeros((l, l))
    med_img = np.zeros((l, l))
    large_img = np.zeros((l, l))

    for i in range(l):
        for j in range(l):
            sm_img[i][j] = small(tc[i][j])
            med_img[i][j] = medium(tc[i][j])
            large_img[i][j] = large(tc[i][j])

    op_imagepath = os.path.join('images', 'tc', os.path.basename(imagepath)[
        :-4]+'_Small.png')
    normalize_and_save(op_imagepath, sm_img)

    op_imagepath = os.path.join('images', 'tc', os.path.basename(imagepath)[
        :-4]+'_Medium.png')
    normalize_and_save(op_imagepath, med_img)

    op_imagepath = os.path.join('images', 'tc', os.path.basename(imagepath)[
        :-4]+'_Large.png')
    normalize_and_save(op_imagepath, large_img)
