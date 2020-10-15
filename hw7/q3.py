import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.io import wavfile
from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='path to the input wav file')
    parser.add_argument('-o', '--output', type=str, default='out.png',
                        help='path to store the output image')
    parser.add_argument('--window', type=float, default=40e-3,
                        help='duration of the window in [s]')
    parser.add_argument('--skip', type=float, default=20e-3,
                        help='duration of skip in [s]')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    fs, data = wavfile.read(args.input)

    frame_size = int(fs * args.window)
    skip = int(fs * args.skip)
    num_frames = (data.shape[0] - frame_size) // skip

    freqs = np.fft.rfftfreq(frame_size, 1 / fs)
    occupation = np.zeros((freqs.shape[0], num_frames))
    win_data_prior_true = np.ones_like(freqs) * .2
    win_data_prior_false = np.ones_like(freqs) * .8
    for i in trange(num_frames):
        win_data = data[i * skip: i * skip + frame_size]
        win_data_fft = np.fft.rfft(win_data)
        win_data_fft_abs = np.abs(win_data_fft)
        win_data_fft_abs_med = np.median(win_data_fft_abs)
        win_data_fft_abs_std = np.std(win_data_fft_abs)
        win_data_fft_dev = (win_data_fft_abs - win_data_fft_abs_med) / win_data_fft_abs_std
        win_data_likely_false = stats.norm.cdf(win_data_fft_dev + 2) - \
                                stats.norm.cdf(win_data_fft_dev - 2)
        win_data_likely_true = 1 - win_data_likely_false
        # bayesian update
        win_data_true = win_data_likely_true * win_data_prior_true + 1e-10
        win_data_false = win_data_likely_false * win_data_prior_false + 1e-10
        win_data_norm = win_data_true + win_data_false
        win_data_prior_true = win_data_true / win_data_norm
        win_data_prior_false = win_data_false / win_data_norm
        # put decision into result buffer
        occupation[:, i] = win_data_prior_true

    plt.imshow(occupation, 'gray', origin='lower', aspect='auto',
               extent=[0, num_frames * args.skip, 0, fs / 2])
    plt.title('Time Frequency Occupation')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.show()
