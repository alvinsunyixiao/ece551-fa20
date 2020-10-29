import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-samples', type=int, default=512,
                        help='number of samples to simulate (default to 1000)')
    parser.add_argument('-f', '--frequency', type=float, default=.2,
                        help='normalized frequency to cancel out in the range of [0, 1] (default to .2)')
    parser.add_argument('--mu', type=float, default=.2,
                        help='learning rate of the LMS algorithm (default to 0.2)')
    parser.add_argument('--C', type=float, default=1,
                        help='scaling constant multiplied to the sinusoids (default to 1.0)')
    parser.add_argument('--phi', type=float, default=0,
                        help='phase applied to the sinusoids')
    return parser.parse_args()


def frequency_response(w0, C, mu):
    w = np.linspace(0, np.pi, 512)
    H_CG = 2 * mu * C**2 * \
            ((1 - np.cos(w0)*np.exp(-1j*w)) / (1 - 2*np.cos(w0)*np.exp(-1j*w) + np.exp(-2j*w)) - 1)
    H_AC = 1 / (1 + H_CG)

    return H_AC

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
        x1 = args.C * np.cos(i * omega_0 + args.phi)
        x2 = args.C * np.sin(i * omega_0 + args.phi)

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

    # plot theoretical filter response
    H_AC = frequency_response(omega_0, args.C, args.mu)
    plt.figure(figsize=(15, 10))
    plt.subplot(211)
    plt.plot(np.linspace(0, 1, H_AC.shape[0]), np.abs(H_AC))
    plt.title('Theoretical Filter Response')
    plt.ylabel('Filter Magnitude Response')

    plt.subplot(212)
    plt.plot(np.fft.rfftfreq(d.shape[0], d=1/2), np.abs(np.fft.rfft(error)))
    plt.title('Simulated Filter Response')
    plt.xlabel('Normalized Frequency ($\\pi$ rads)')
    plt.ylabel('Filtered Noise Magnitude')

    plt.suptitle(f'LMS Notch Filter Response with C = {args.C}, $\\mu$ = {args.mu}')
    plt.show()
