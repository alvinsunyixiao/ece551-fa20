import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

def H(x, N, spread, noise_power, targets=None):
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
    if targets is None:
        targets = np.clip(np.floor(spread * np.random.randn(N) + recording_length/2).astype(int),0,recording_length - pulse_width)
        targets = sorted(targets)

    for target in targets:
        out[target:target+pulse_width] += x

    return out, targets


# settings
N = 4
spread = 200
noise_power = 5e-3
signal_power = 5e-1
L = 128
freq = np.pi / 4

def expected_power(signal):
    return np.sum(signal**2) / signal.shape[0]

def power_normalize(signal):
    return signal / np.sqrt(expected_power(signal) / signal_power)

def rectangular_pulse(L, freq):
    pulse = np.sin(freq * np.arange(L))
    return power_normalize(pulse)

def triangular_pulse(L, freq):
    pulse = np.sin(freq * np.arange(L))
    window = np.bartlett(L)
    return power_normalize(pulse * window)

def hamming_pulse(L, freq):
    pulse = np.sin(freq * np.arange(L))
    window = np.hamming(L)
    return power_normalize(pulse * window)

def sinc_pulse(L, omega_b):
    n = np.linspace(-L//2, L//2, L, endpoint=False)
    pulse = np.sinc(n * omega_b / np.pi)
    return power_normalize(pulse)

def warped_sinc_pulse(L, omega_b, num_bands):
    n = np.arange(L, dtype=float)
    pulse = np.zeros_like(n)
    T = L / num_bands # spread out in time as much as possible
    for m in range(num_bands):
        n_m = n - m * T
        w_h = (m + 1) / num_bands * omega_b / np.pi
        w_l = m / num_bands * omega_b / np.pi
        pulse += w_h * np.sinc(w_h * n_m) - w_l * np.sinc(w_l * n_m)
    return power_normalize(pulse)

def peak_corr(pulse, receiver, name):
    corr = np.correlate(receiver, pulse, mode='valid')
    corr_abs = np.abs(corr)
    corr_max = ndimage.maximum_filter1d(corr_abs, L//8, mode='constant')
    corr_min = ndimage.minimum_filter1d(corr_abs, L//8, mode='constant')
    indics_mask = (corr_abs == corr_max) & (corr_max - corr_min >= .3*signal_power*L)
    indics = np.argwhere(indics_mask).flatten()
    plt.figure(figsize=(15, 5))
    # plot received signal
    plt.subplot(121)
    plt.plot(receiver)
    plt.title(f'{name} pulse receiver')
    # plot absolute correlation
    plt.subplot(122)
    plt.plot(corr_abs)
    plt.title(f'absolute correlation of {name} pulse\nestimated delays: {indics}')

if __name__ == '__main__':
    # rectangle windowed sinusoid
    pulse = rectangular_pulse(L,freq)
    receiver, delays = H(pulse,N,spread,noise_power)
    peak_corr(pulse, receiver, 'sine (rectangle window)')
    # triangle windowed sinusoid
    pulse = triangular_pulse(L,freq)
    receiver, _ = H(pulse,N,spread,noise_power, targets=delays)
    peak_corr(pulse, receiver, 'sine (triangle window)')
    # hamming windowed sinusoid
    pulse = hamming_pulse(L,freq)
    receiver, _ = H(pulse,N,spread,noise_power, targets=delays)
    peak_corr(pulse, receiver, 'sine (hamming window)')
    # rectangle windowed sinc
    pulse = sinc_pulse(L, freq)
    receiver, _ = H(pulse,N,spread,noise_power, targets=delays)
    peak_corr(pulse, receiver, 'sinc')
    # rectangle windowed flattened sinc
    pulse = warped_sinc_pulse(L, freq, 4)
    receiver, _ = H(pulse,N,spread,noise_power, targets=delays)
    peak_corr(pulse, receiver, 'flattened sinc')
    print(f'Ground Truth Delays: {delays}')
    # comparison between original sinc and flattened sinc
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(sinc_pulse(L, freq))
    plt.title('sinc pulse')
    plt.subplot(122)
    plt.plot(warped_sinc_pulse(L, freq, 4))
    plt.title('flattened sinc')
    plt.show()
