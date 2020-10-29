import argparse
import numpy as np
import matplotlib.pyplot as plt

C = 1
PHI = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-samples', type=int, default=512,
                        help='number of samples to simulate (default to 1000)')
    parser.add_argument('-f', '--frequency', type=float, default=.2,
                        help='normalized frequency to cancel out in the range of [0, 1] (default to .2)')
    parser.add_argument('--mu', type=float, default=.4,
                        help='learning rate of the LMS algorithm (defulat to 0.2)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    omega_0 = args.frequency * np.pi

    y = np.zeros(args.num_samples)
    error = np.zeros_like(y)
    d = np.random.randn(y.shape[0])

    w1 = 0
    w2 = 0
    for i in range(args.num_samples):
        # compute sinusoids
        x1 = C * np.cos(i * omega_0 + PHI)
        x2 = C * np.sin(i * omega_0 + PHI)

        # compute output
        F = x1 * w1
        J = x2 * w2
        y[i] = F + J

        # compute feedback
        error[i] = d[i] - y[i]

        # compute top branch
        D = x1 * error[i]
        w1 += D * 2 * args.mu

        # compute bottom branch
        H = x2 * error[i]
        w2 += H * 2 * args.mu

    plt.plot(np.fft.rfftfreq(d.shape[0], d=1/2), np.abs(np.fft.rfft(error)))
    plt.show()
