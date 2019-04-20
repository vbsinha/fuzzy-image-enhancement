import os
import random

import imageio
import numpy as np

from utils import rmse

def addnoise(name: str, zero_mean_gaussian_noise_sd: int = 15, 
             percent_gaussian_impulse_noise: int = 5, impulse_noise_sd: int = 100):
    imagepath = os.path.join('images', 'original', name)
    img = imageio.imread(imagepath)
    inp_img = img.astype(float)
    # print(img)
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
                          zero_mean_gaussian_noise_sd, percent_gaussian_impulse_noise,
                          impulse_noise_sd))
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    # print(img)
    # print(inp_img)
    imageio.imwrite(oppath, img)
    print('RMSE between generated noisy image and original image: {}'.format(rmse(inp_img, img)))

random.seed(18)
addnoise('Cameraman.png')