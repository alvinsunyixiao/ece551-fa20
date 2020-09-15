import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep

def gen_wave(UIN, n0=0):
    T = np.arange(100)
    Ts = 1 + 3*np.arange(len(UIN))
    sp = splrep(Ts, UIN, k=3)
    return splev(T-n0,sp,ext=1)

def estimate_delay_and_gain(x1, x2):
    # Your algorithm Here
    cross_corr = np.correlate(x1, x2, mode='full')
    delta = np.argmax(np.abs(cross_corr)) - (x1.shape[0] - 1)
    non_zeros_idx = np.argwhere(x1 != 0).flatten()[0]
    rho = x1[non_zeros_idx] / x2[non_zeros_idx - delta]

    return delta, rho, cross_corr

if __name__ == '__main__':

    UIN = np.array([ 6, 6, 4, 5, 0, 3, 7, 3, 4 ])
    n1, n2 = np.random.randint(1,40,size = 2)
    alpha1, alpha2 = np.random.randn(2)
    x1, x2 = gen_wave(alpha1*UIN, n1), gen_wave(alpha2*UIN, n2)

    delta, rho, cross_corr = estimate_delay_and_gain(x1,x2)

    # comparison plot
    x_true = gen_wave(UIN)
    plt.figure()
    # signal comparison
    plt.subplot(231)
    plt.stem(x_true)
    plt.title('Original Signal')
    plt.subplot(232)
    plt.stem(x1)
    plt.title('x1')
    plt.subplot(233)
    plt.stem(x2)
    plt.title('x2')
    # cross correlation
    plt.subplot(212)
    plt.stem(np.arange(cross_corr.shape[0]) - (x1.shape[0] - 1), np.abs(cross_corr))
    plt.xlabel('$\\Delta$')
    plt.ylabel('Absolute Crosscorrelation')
    plt.title('Absolute Crosscorrelation Between x1 and x2')
    # super title
    hat_rho = '$\\hat{\\rho}$'
    hat_Delta = '$\\hat{\\Delta}$'
    plt.suptitle(f'Ground Truth: $\\rho = {alpha1 / alpha2:.4f}$ $\\Delta$ = {n1 - n2} \n'
                 f'Estimated: {hat_rho} = {rho:.4f} {hat_Delta} = {delta}')

    plt.show()

