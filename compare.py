import argparse
import os

from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_tv_bregman
import imageio

from utils import rmse, saveimg

parser = argparse.ArgumentParser(description='Image Enhancement using scipy and skimage')
parser.add_argument('--image', type=str, required=True,
                    help='Image file path. Must be within images folder.'
                         'Specify path only after image folder')
parser.add_argument('--method', type=str,
                    choices=['Sharpen', 'Gauss', 'TVC', 'TVB', 'Bil'],
                    required=True, help='Choose the method to enhance')
parser.add_argument('--original', type=str,
                    help='Path of original image to compute the RMS error within images folder')
args = parser.parse_args()

imagepath = os.path.join('images', args.image)
inp_img = imageio.imread(imagepath)
img = inp_img.astype(float)  # Save a copy as inp_img will be modified
assert img.shape == (256, 256), 'Image is not of size (255,255).'

orig_imagepath = os.path.join('images', args.original)
orig_img = imageio.imread(orig_imagepath)

if args.method == 'Sharpen':
    blurred_img = ndimage.gaussian_filter(img, 3)
    filter_blurred_img = ndimage.gaussian_filter(blurred_img, 1)
    alpha = 30
    sharpened = blurred_img + alpha * (blurred_img - filter_blurred_img)
    op_imagepath = os.path.join('images', 'enhanced', os.path.basename(imagepath)[
        :-4]+'_sharpen.png')
    sharpened = saveimg(op_imagepath, sharpened)
    err = rmse(sharpened, orig_img)
    print('RMSE of filterSharpen (against original image):{}'.format(err))

if args.method == 'Gauss':
    gauss_denoised = ndimage.gaussian_filter(img, 2)
    op_imagepath = os.path.join('images', 'enhanced', os.path.basename(imagepath)[
        :-4]+'_gauss.png')
    gauss_denoised = saveimg(op_imagepath, gauss_denoised)
    err = rmse(gauss_denoised, orig_img)
    print('RMSE of filterGauss (against original image):{}'.format(err))

if args.method == 'TVC':
    tvc = denoise_tv_chambolle(img, weight=30, multichannel=False)
    op_imagepath = os.path.join('images', 'enhanced', os.path.basename(imagepath)[
        :-4]+'_tvc.png')
    tvc = saveimg(op_imagepath, tvc)
    err = rmse(tvc, orig_img)
    print('RMSE of filterTVC (against original image):{}'.format(err))

if args.method == 'TVB':
    tvb = denoise_tv_bregman(img, weight=0.01)
    op_imagepath = os.path.join('images', 'enhanced', os.path.basename(imagepath)[
        :-4]+'_tvb.png')
    tvb = saveimg(op_imagepath, tvb)
    err = rmse(tvb, orig_img)
    print('RMSE of filterTVB (against original image):{}'.format(err))

if args.method == 'Bil':
    bil = denoise_bilateral(img, multichannel=False, sigma_spatial=5)
    op_imagepath = os.path.join('images', 'enhanced', os.path.basename(imagepath)[
                                :-4]+'_bil.png')
    tvb = saveimg(op_imagepath, bil)
    err = rmse(bil, orig_img)
    print('RMSE of filterBil (against original image):{}'.format(err))
