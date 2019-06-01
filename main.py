import argparse
import os

from scipy.signal import medfilt2d
import imageio
import numpy as np

from utils import rmse, plot
from filters import (
    allfilters,
    compute_mu_beta_w,
    filterA,
    filterB,
    filterC,
    filterR1,
    filterR2,
    filterR3,
    filterR3Crisp,
    filterR4,
    filterR4Crisp,
    TC
)

parser = argparse.ArgumentParser(description='Fuzzy Image Enhancement')
parser.add_argument('--image', type=str, required=True,
                    help='Image file path. Must be within images folder.'
                         'Specify path only after image folder')
parser.add_argument('--method', type=str,
                    choices=['A', 'B', 'C', 'Med', 'R1', 'R2', 'R3',
                             'R3Crisp', 'R4', 'R4Crisp', 'All', 'Plot'],
                    required=True, help='Choose the method to enhance')
parser.add_argument('--original', type=str,
                    help='Path of original image to compute the RMS error within images folder')
parser.add_argument('-k', type=int, default=5, help='Filter size, default: 5')
args = parser.parse_args()

imagepath = os.path.join('images', args.image)
inp_img = imageio.imread(imagepath)
img = inp_img.astype(float)  # Save a copy as inp_img will be modified
assert img.shape == (256, 256), 'Image is not of size (255,255).'

mu, beta, w = compute_mu_beta_w(img, args.k)

if args.method == 'A':
    img = filterA(img, args.k, mu, beta)
elif args.method == 'B':
    img = filterB(img, args.k, beta)
elif args.method == 'C':
    img = filterC(img, args.k, mu, beta)
elif args.method == 'Med':
    img = medfilt2d(img, args.k)
elif args.method == 'R1':
    tc = TC(len(img), args.k, mu, w)
    img = filterR1(img, args.k, tc, beta)
elif args.method == 'R2':
    tc = TC(len(img), args.k, mu, w)
    img = filterR2(img, args.k, tc, mu, beta)
elif args.method == 'R3':
    tc = TC(len(img), args.k, mu, w)
    img = filterR3(img, args.k, tc, mu, beta)
elif args.method == 'R3Crisp':
    tc = TC(len(img), args.k, mu, w)
    img = filterR3Crisp(img, args.k, tc, mu, beta)
elif args.method == 'R4':
    tc = TC(len(img), args.k, mu, w)
    img = filterR4(img, args.k, tc, mu, beta)
elif args.method == 'R4Crisp':
    tc = TC(len(img), args.k, mu, w)
    img = filterR4Crisp(img, args.k, tc, mu, beta)
elif args.method == 'Plot':
    tc = TC(len(img), args.k, mu, w)
    plot(tc, imagepath)
    exit()
elif args.method == 'All':
    tc = TC(len(img), args.k, mu, w)
    allfilters(img, args.k, tc, mu, beta, os.path.basename(
        imagepath)[:-4], args.original)
    exit()


# Save image
op_imagepath = os.path.join('images', 'enhanced', os.path.basename(imagepath)[
                            :-4]+'_'+args.method+'.png')
img = np.clip(img, 0, 255)
img = img.astype(np.uint8)
imageio.imwrite(op_imagepath, img)

if args.original is not None:
    orig_imagepath = os.path.join('images', args.original)
    orig_img = imageio.imread(orig_imagepath)

    noisy_err = rmse(orig_img, inp_img)
    enhanced_err = rmse(img, orig_img)

    print('RMSE of input image (against original image):{}'.format(noisy_err))
    print('RMSE of enhanced image (against original image):{}'.format(enhanced_err))
