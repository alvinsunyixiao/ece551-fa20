import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from scipy.io import wavfile
from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='path to the input wav file')
    parser.add_argument('-p', '--order', type=int, default=6,
                        help='order of the synthesis filter')
    parser.add_argument('--window', type=float, default=10e-3,
                        help='window duration in [s]')
    parser.add_argument('-o', '--output', type=str, default='out.wav',
                        help='path to store the output reconstructed audio (Default to out.wav)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    fs, data = wavfile.read(args.input)
    data = data.astype(np.float32) / (1 << 15)
    data_synth = np.zeros_like(data)
    data_err = np.zeros_like(data)
    frame_size = int(fs * args.window)
    num_frames = data.shape[0] // frame_size
    # construct Yule-Walker matrix index
    i, j = np.meshgrid(np.arange(args.order), np.arange(args.order), indexing='ij')
    YW_idx = np.abs(i - j)
    # pitch phase tracking
    phase_last = 0
    # minimum pitch index @ 250Hz
    p_min = fs // 250
    # solve for each frame
    for i in trange(0, num_frames):
        win_data = data[i*frame_size: (i+1)*frame_size]
        win_data_pad = np.pad(win_data, [0, args.order])
        win_corr = np.correlate(win_data_pad, win_data, 'valid')
        # solve for filter coefficients
        alpha = np.linalg.solve(win_corr[YW_idx], win_corr[1:])
        # linear prediction
        win_data_left = np.pad(win_data, [args.order, 0])
        win_data_pred = np.convolve(win_data_left[:-1], alpha, 'valid')
        # estimate for error / excitation signal
        e = win_data - win_data_pred
        data_err[i*frame_size: (i+1)*frame_size] = e
        # voice / unvoiced detection
        win_corr_full = np.correlate(np.pad(win_data, [0, win_data.shape[0]]), win_data, 'valid')
        corr_0 = win_corr_full[0]
        win_corr_full[:p_min] = 0
        pitch_idx = np.argmax(win_corr_full)
        # synthesis
        if win_corr_full[pitch_idx] / corr_0 > .25: # voiced frame
            excitation = np.zeros_like(e)
            pulse_idx = (phase_last + pitch_idx) % frame_size
            while True:
                excitation[pulse_idx] = 1
                phase_last = pulse_idx
                pulse_idx += pitch_idx
                if pulse_idx >= frame_size:
                    break
        else: # unvoiced frame
            excitation = np.random.randn(e.shape[0])
        # normalize with energy
        excitation = excitation * np.sqrt(np.sum(np.square(e)) / np.sum(np.square(excitation)))
        # sythesis
        y_hat = signal.lfilter([1.], np.hstack([[1.], -alpha]), excitation)
        data_synth[i*frame_size: (i+1)*frame_size] = y_hat

    wavfile.write(args.output, fs, data_synth)
