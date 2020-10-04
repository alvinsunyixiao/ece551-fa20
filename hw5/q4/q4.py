import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform

def radon_then_back_proj(img, N, ramp):
    angles = np.linspace(0, 180, N, endpoint=False)
    img_radon = transform.radon(img, theta=angles, circle=False)
    img_proj = transform.iradon(img_radon, theta=angles, circle=False,
                                filter_name='ramp' if ramp else None)
    return img_proj

def plot_pair(img, N):
    img_proj_nofilt = radon_then_back_proj(img, N, ramp=False)
    img_proj_ramp = radon_then_back_proj(img, N, ramp=True)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(img_proj_nofilt, cmap='gray')
    plt.title('No filter')
    plt.subplot(122)
    plt.imshow(img_proj_ramp, cmap='gray')
    plt.title('Ramp filter')
    plt.suptitle(f'Radon Transformed Then Back Projected Image with N = {N} Equally Spaced Angles')

if __name__ == '__main__':
    img = data.shepp_logan_phantom()
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plot_pair(img, 4)
    plot_pair(img, 16)
    plot_pair(img, 64)
    plot_pair(img, 256)
    plt.show()
