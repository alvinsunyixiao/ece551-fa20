import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

def H(x, N, spread, noise_power):
    """
    Takes an input waveform `x`, a number of targets `N`,
    a standard `deviation` for the random placement of
    the targets. Returns a length 4096 vector with replicas
    of `x` placed at each chose target point.
    White noise of power `noise_power` is added to the signal.
    """
    recording_length = 4096
    pulse_width = len(x)

    out = np.sqrt(noise_power) * np.random.randn(recording_length)
    targets = np.clip(np.floor(spread * np.random.randn(N) + recording_length/2).astype(int),0,recording_length - pulse_width)

    for target in targets:
        out[target:target+pulse_width] += x

    return out, targets

def rectangular_pulse(L,freq):
    return np.sin(freq*np.arange(L))

def sinc_pulse(L, omega_b):
    delay = 0
    n = (np.linspace(-L//2, L//2, L, endpoint=False) - delay) * omega_b / np.pi
    return np.sinc(n)

# settings
N = 4
spread = 300
noise_power = 1e-3
L = 128
freq = np.pi / 20

def peak_corr(pulse, receiver):
    corr = np.correlate(receiver, pulse, mode='valid')
    corr_abs = np.abs(corr)
    corr_max = ndimage.maximum_filter1d(corr_abs, L//2, mode='constant')
    corr_min = ndimage.minimum_filter1d(corr_abs, L//2, mode='constant')
    indics_mask = (corr_abs == corr_max) & (corr_max - corr_min >= 10)
    indics = np.argwhere(indics_mask).flatten()
    plt.figure()
    plt.plot(corr_abs)
    plt.title(f'Estimated Delays (via peak finding): {indics}')

if __name__ == '__main__':
    pulse = rectangular_pulse(L,freq)
    pulse = sinc_pulse(L, np.pi/20)
    print(pulse.max())
    receiver, delays = H(pulse,N,spread,noise_power)
    plt.figure()
    plt.plot(receiver)
    print(f'Ground Truth Delays: {delays}')
    peak_corr(pulse, receiver)

    plt.show()
