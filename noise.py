import argparse
import os
import random

import imageio
import numpy as np

from utils import rmse


def addnoise(name: str, zero_mean_gaussian_noise_sd: int = 5,
             percent_gaussian_impulse_noise: int = 5, impulse_noise_sd: int = 100):
    """ Create a noisy image
    percent_gaussian_impulse_noise percentage pixels in the image will have gaussian impulse noise
    having mean 128 and standard deviation as impulse_noise_sd
    The other 100 - percent_gaussian_impulse_noise percentage pixels in the image will be added with
    0 mena gaussian noise having standard deviation as zero_mean_gaussian_noise_sd """
    imagepath = os.path.join('images', 'original', name)
    img = imageio.imread(imagepath).astype(float)
    inp_img = np.array(img, dtype=float, copy=True)
    assert img.shape == (256, 256)

    percent_noise = percent_gaussian_impulse_noise / 100
    for i in range(len(img)):
        for j in range(len(img)):
            r = random.uniform(0, 1)
            if r < percent_noise:
                img[i][j] = random.gauss(128, impulse_noise_sd)
            else:
                img[i][j] += random.gauss(0, zero_mean_gaussian_noise_sd)

    oppath = os.path.join('images', 'noisy', '{}_{}_{}_{}.png'.format(name[:-4],
                                                                      zero_mean_gaussian_noise_sd,
                                                                      percent_gaussian_impulse_noise,
                                                                      impulse_noise_sd))
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    imageio.imwrite(oppath, img)

    print('RMSE between generated noisy image and original image: {}'.format(
        rmse(inp_img, img)))


# random.seed(18)
parser = argparse.ArgumentParser(description='Noise addition to images')
parser.add_argument('--image', type=str, required=True,
                    help='Image file path. Must be within images/original/ folder.'
                         'Specify only the image name')
args = parser.parse_args()
addnoise(args.image)
