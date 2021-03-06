import imageio
import numpy as np
from scipy.signal import medfilt2d

from fuzzysets import small, medium, large
from utils import rmse, saveimg

import os

alpha = 100


def _filterA(img, i, j, k, mu, betaij):
    l = len(img)
    m = np.arange(k) - k//2
    n = np.arange(k) - k//2
    d2 = []
    _mu = []
    w = []
    I = []

    for p in m:
        for q in n:
            if 0 <= i+p < l and 0 <= j+q < l:
                d2.append((img[i+p][j+q] - img[i][j])**2)
                _mu.append(mu[i][j][p][q])
                I.append(img[i+p][j+q])
    minusd2bybetaij = d2 / (-1 * betaij)

    weights = _mu * (1+minusd2bybetaij)
    num, den = 0, 0
    for i in range(len(weights)):
        num += weights[i] * I[i]
        den += weights[i]
    return num / den


def filterA(img, k, mu, beta):
    l = len(img)
    result = np.zeros((l, l))
    for i in range(len(img)):
        for j in range(len(img)):
            result[i][j] = _filterA(img, i, j, k, mu, beta[i][j])
    return result


def filterB(img, k, beta):
    """ Apply filter B on img given pre-computed mu and beta """
    l = len(img)
    m = np.arange(k) - k//2
    n = np.arange(k) - k//2
    result = np.zeros((l, l))

    beta = 1 / beta

    # Apply filter
    for i in range(l):
        for j in range(l):
            num = 0
            den = 0
            for p in m:
                for q in n:
                    if p == 0 and q == 0:
                        continue
                    if 0 <= i+p < l and 0 <= j+q < l:
                        num += beta[p+i][q+j] * img[p+i][q+j]
                        den += beta[p+i][q+j]
            result[i][j] = num / den

    return result


def filterC(img, k, mu, beta):
    """ Apply filter C on img given pre-computed mu and beta """
    l = len(img)
    m = np.arange(k) - k//2
    n = np.arange(k) - k//2
    result = np.zeros((l, l))

    for i in range(l):
        for j in range(l):
            num = 0
            den = 0
            for p in m:
                for q in n:
                    if 0 <= i+p < l and 0 <= j+q < l:
                        _mu = mu[i+p][j+q][-p][-q]
                        weight = _mu / beta[i+p][j+q]
                        num += weight * img[i+p][j+q]
                        den += weight
            result[i][j] = num / den

    return result


def compute_mu_beta_w(img, k):
    """ Computes mu beta and w. This is a useful pre-computation  """
    l = len(img)
    mu = np.zeros((l, l, k, k))
    w = np.zeros((l, l, k, k))
    beta = np.zeros((l, l))

    m = np.arange(k) - k//2
    n = np.arange(k) - k//2

    for i in range(l):
        for j in range(l):
            d2 = np.zeros((k, k))
            num = 0
            for p in m:
                for q in n:
                    if 0 <= i+p < l and 0 <= j+q < l:
                        num += 1
                        d2[p][q] = (img[i+p][j+q] - img[i][j])**2
                        w[i][j][p][q] = np.exp(-1*(p**2 + q**2)/alpha)
            beta[i][j] = np.sum(d2) / (num - 1)
            mu[i][j] = w[i][j] * np.exp(d2 / (-1 * beta[i][j]))

    return mu, beta, w


def TC(l, k, mu, w):
    """ Computes total compatibility """

    m = np.arange(k) - k//2
    n = np.arange(k) - k//2

    tc = np.zeros((l, l))

    for i in range(l):
        for j in range(l):
            num, den = 0, 0
            for p in m:
                for q in n:
                    if p == 0 and q == 0:
                        continue
                    if 0 <= i+p < l and 0 <= j+q < l:
                        num += mu[i+p][j+q][-p][-q]
                        den += w[i+p][j+q][-p][-q]
            tc[i][j] = num / den

    return tc


def filterR1(img, k, tc, beta, img_A=None, img_B=None, img_C=None):
    """ Apply filter R1 on img given pre-computed tc and beta """
    l = len(img)
    result = np.zeros((l, l))

    if img_B is None:
        img_B = filterB(img, k, beta)

    for i in range(l):
        for j in range(l):
            c1 = small(tc[i][j])
            c2 = 1 - c1
            result[i][j] = (c1 * img_B[i][j] + c2 * img[i][j]) / (c1 + c2)

    return result


def filterR2(img, k, tc, mu, beta, img_A=None, img_B=None, img_C=None):
    """ Apply filter R2 on img given pre-computed tc, mu and beta """
    l = len(img)
    result = np.zeros((l, l))

    if img_B is None:
        img_B = filterB(img, k, beta)

    if img_C is None:
        img_C = filterC(img, k, mu, beta)

    for i in range(l):
        for j in range(l):
            c1 = small(tc[i][j])
            c2 = 1 - c1
            result[i][j] = (c1 * img_B[i][j] + c2 * img_C[i][j]) / (c1 + c2)

    return result


def filterR3(img, k, tc, mu, beta, img_A=None, img_B=None, img_C=None):
    """ Apply filter R3 on img given pre-computed tc, mu and beta """
    l = len(img)
    result = np.zeros((l, l))

    if img_A is None:
        img_A = filterA(img, k, mu, beta)

    if img_B is None:
        img_B = filterB(img, k, beta)

    if img_C is None:
        img_C = filterC(img, k, mu, beta)

    for i in range(l):
        for j in range(l):
            c1 = small(tc[i][j])
            c2 = medium(tc[i][j])
            c3 = large(tc[i][j])
            result[i][j] = (c1 * img_A[i][j] + c2 * img_B[i]
                            [j] + c3 * img_C[i][j]) / (c1 + c2 + c3)

    return result


def filterR3Crisp(img, k, tc, mu, beta, img_A=None, img_B=None, img_C=None):
    """ Apply filter R3-Crsip on img given pre-computed tc, mu and beta """
    l = len(img)
    result = np.zeros((l, l))

    if img_A is None:
        img_A = filterA(img, k, mu, beta)

    if img_B is None:
        img_B = filterB(img, k, beta)

    if img_C is None:
        img_C = filterC(img, k, mu, beta)

    for i in range(l):
        for j in range(l):
            c1 = small(tc[i][j])
            c2 = medium(tc[i][j])
            c3 = large(tc[i][j])
            if c1 == max(c1, c2, c3):
                result[i][j] = img_A[i][j]
            elif c2 == max(c1, c2, c3):
                result[i][j] = img_B[i][j]
            else:
                result[i][j] = img_C[i][j]

    return result


def filterR4(img, k, tc, mu, beta, img_A=None, img_B=None, img_C=None):
    """ Apply filter R4 on img given pre-computed tc, mu and beta """
    l = len(img)
    result = np.zeros((l, l))

    if img_A is None:
        img_A = filterA(img, k, mu, beta)

    if img_B is None:
        img_B = filterB(img, k, beta)

    if img_C is None:
        img_C = filterC(img, k, mu, beta)

    for i in range(l):
        for j in range(l):
            c1 = small(tc[i][j])
            c2 = medium(tc[i][j])
            c3 = large(tc[i][j])
            result[i][j] = (c1 * img_B[i][j] + c2 * img_A[i]
                            [j] + c3 * img_C[i][j]) / (c1 + c2 + c3)

    return result


def filterR4Crisp(img, k, tc, mu, beta, img_A=None, img_B=None, img_C=None):
    """ Apply filter R4-Crsip on img given pre-computed tc, mu and beta """
    l = len(img)
    result = np.zeros((l, l))

    if img_A is None:
        img_A = filterA(img, k, mu, beta)

    if img_B is None:
        img_B = filterB(img, k, beta)

    if img_C is None:
        img_C = filterC(img, k, mu, beta)

    for i in range(l):
        for j in range(l):
            c1 = small(tc[i][j])
            c2 = medium(tc[i][j])
            c3 = large(tc[i][j])
            if c1 == max(c1, c2, c3):
                result[i][j] = img_B[i][j]
            elif c2 == max(c1, c2, c3):
                result[i][j] = img_A[i][j]
            else:
                result[i][j] = img_C[i][j]

    return result


def allfilters(img, k, tc, mu, beta, imagename, original):
    """ This function applies all the filters and saves the output images and logs the RMSE.
    Filters applied: A, B, C, R1, R2, R3, R4, R3-Crisp, R4-Crisp and Median """
    inp_img = img.copy()

    img_A = filterA(img, k, mu, beta)
    op_imagepath = os.path.join('images', 'enhanced', imagename+'_A.png')
    img_A = saveimg(op_imagepath, img_A)

    img_B = filterB(img, k, beta)
    op_imagepath = os.path.join('images', 'enhanced', imagename+'_B.png')
    img_B = saveimg(op_imagepath, img_B)

    img_C = filterC(img, k, mu, beta)
    op_imagepath = os.path.join('images', 'enhanced', imagename+'_C.png')
    img_C = saveimg(op_imagepath, img_C)

    img_Med = medfilt2d(img, k)
    op_imagepath = os.path.join('images', 'enhanced', imagename+'_Med.png')
    img_Med = saveimg(op_imagepath, img_Med)

    img_R1 = filterR1(img, k, tc, beta, img_A=img_A, img_B=img_B, img_C=img_C)
    op_imagepath = os.path.join('images', 'enhanced', imagename+'_R1.png')
    img_R1 = saveimg(op_imagepath, img_R1)

    img_R2 = filterR2(img, k, tc, mu, beta, img_A=img_A,
                      img_B=img_B, img_C=img_C)
    op_imagepath = os.path.join('images', 'enhanced', imagename+'_R2.png')
    img_R2 = saveimg(op_imagepath, img_R2)

    img_R3 = filterR3(img, k, tc, mu, beta, img_A=img_A,
                      img_B=img_B, img_C=img_C)
    op_imagepath = os.path.join('images', 'enhanced', imagename+'_R3.png')
    img_R3 = saveimg(op_imagepath, img_R3)

    img_R3Crisp = filterR3Crisp(
        img, k, tc, mu, beta, img_A=img_A, img_B=img_B, img_C=img_C)
    op_imagepath = os.path.join('images', 'enhanced', imagename+'_R3Crisp.png')
    img_R3Crisp = saveimg(op_imagepath, img_R3Crisp)

    img_R4 = filterR4(img, k, tc, mu, beta, img_A=img_A,
                      img_B=img_B, img_C=img_C)
    op_imagepath = os.path.join('images', 'enhanced', imagename+'_R4.png')
    img_R4 = saveimg(op_imagepath, img_R4)

    img_R4Crisp = filterR4(img, k, tc, mu, beta,
                           img_A=img_A, img_B=img_B, img_C=img_C)
    op_imagepath = os.path.join('images', 'enhanced', imagename+'_R4Crisp.png')
    img_R4Crisp = saveimg(op_imagepath, img_R4Crisp)

    if original is not None:
        orig_imagepath = os.path.join('images', original)
        orig_img = imageio.imread(orig_imagepath)

        err = rmse(img_A, orig_img)
        print('RMSE of filterA (against original image):{}'.format(err))
        err = rmse(img_B, orig_img)
        print('RMSE of filterB (against original image):{}'.format(err))
        err = rmse(img_C, orig_img)
        print('RMSE of filterC (against original image):{}'.format(err))
        err = rmse(img_Med, orig_img)
        print('RMSE of filterMed (against original image):{}'.format(err))
        err = rmse(img_R1, orig_img)
        print('RMSE of filterR1 (against original image):{}'.format(err))
        err = rmse(img_R2, orig_img)
        print('RMSE of filterR2 (against original image):{}'.format(err))
        err = rmse(img_R3, orig_img)
        print('RMSE of filterR3 (against original image):{}'.format(err))
        err = rmse(img_R3Crisp, orig_img)
        print('RMSE of filterR3Crisp (against original image):{}'.format(err))
        err = rmse(img_R4, orig_img)
        print('RMSE of filterR4 (against original image):{}'.format(err))
        err = rmse(img_R4Crisp, orig_img)
        print('RMSE of filterR3Crisp (against original image):{}'.format(err))

        noisy_err = rmse(orig_img, inp_img)
        print('RMSE of input image (against original image):{}'.format(noisy_err))
