import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import wavfile
from scipy.linalg import solve_circulant, circulant
from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='path to the input wav file')
    parser.add_argument('-p', '--order', type=int, default=6,
                        help='order of the synthesis filter')
    parser.add_argument('--window', type=float, default=10e-3,
                        help='window duration in [s]')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    fs, data = wavfile.read(args.input)
    # pad 0s in front of the data for performing linear prediction
    frame_size = int(fs * args.window)
    num_frames = data.shape[0] // frame_size
    # construct Yule-Walker matrix index
    i, j = np.meshgrid(np.arange(args.order), np.arange(args.order), indexing='ij')
    YW_idx = np.abs(i - j)
    # error / excitation signal
    data_err = np.zeros_like(data)
    # solve for each frame
    #for i in trange(num_frames):
    i = 3
    win_data = data[i*frame_size: (i+1)*frame_size]
    win_data_pad = np.pad(win_data, [0, args.order])
    win_corr = np.correlate(win_data_pad, win_data, 'valid')
    # solve for filter coefficients
    alpha = np.linalg.solve(win_corr[YW_idx], win_corr[1:])
    # linear prediction
    win_data_left = np.pad(win_data, [args.order, 0])
    win_data_pred = np.convolve(win_data_left[:-1], alpha, 'valid')
    # calculate error signal
    data_err[i*frame_size: (i+1)*frame_size] = win_data - win_data_pred

    plt.figure()
    plt.subplot(411)
    plt.plot(win_data)
    plt.subplot(412)
    plt.plot(win_data_pred)
    plt.subplot(413)
    plt.stem(alpha)
    plt.subplot(414)
    plt.plot(win_data - win_data_pred)
    plt.show()

    #plt.plot(np.arange(data.shape[0]) / fs, data_err)
    #plt.show()
