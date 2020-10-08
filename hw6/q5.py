import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from skimage import io

BLUR_KERNEL = np.ones((11, 11)) / (11 * 11)

def wiener_filter(img, Ax, noise_var, blur=True):
    F = np.fft.rfft2(img)
    if blur:
        G = np.fft.rfft2(BLUR_KERNEL, img.shape)
    else:
        G = np.ones_like(F)

    noise_var += (1 / 256)**2 / 12 # 8-bit quantization noise
    H = G.conj() * Ax / (np.abs(G)**2 * Ax + noise_var)

    return np.fft.irfft2(H * F)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='path to the image to be denoised and deblured')
    parser.add_argument('--original', type=str, required=True,
                        help='path to the ground truth original image')
    return parser.parse_args()

def parse_image_filename(fname):
    blur = True
    if 'noblur' in fname:
        blur = False
    noise_str = '0.'
    if '_' in fname:
        noise_str += fname.split('_')[-1].split('.')[0]
    noise = float(noise_str)

    return noise, blur

def imread_grayscale(fname):
    img = io.imread(fname, as_gray=True)
    if img.dtype == np.uint8:
        img = img / 255.
    return img

if __name__ == '__main__':
    args = parse_args()
    # parse known info from filename
    noise_sigma, blur = parse_image_filename(args.input)
    noise_var = noise_sigma**2
    # read in image and normalize
    img = imread_grayscale(args.input)
    img_true = imread_grayscale(args.original)
    # ground truth image PSD
    X = np.fft.rfft2(img_true)
    Ax = np.abs(X)**2 / np.prod(img.shape)
    # reconstruction
    img_out = wiener_filter(img, Ax, noise_var, blur)
    img_out = np.clip(img_out, 0, 1)
    # MSE
    print(np.mean(np.square(img - img_true)))
    print(np.mean(np.square(img_out - img_true)))
    # plot input image and reconstructed image
    plt.figure(figsize=(25,8))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title(f'Observation\nMSE: {np.mean(np.square(img - img_true)):.8f}')
    plt.subplot(122)
    plt.imshow(img_out, cmap='gray')
    plt.title(f'Reconstruction\nMSE: {np.mean(np.square(img_out - img_true)):.8f}')
    plt.suptitle(f'Noise $\\sigma$: {noise_sigma} Blurred: {blur}')
    plt.show()
