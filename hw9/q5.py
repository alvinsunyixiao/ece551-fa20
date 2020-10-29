import argparse
import numpy as np
import matplotlib.pyplot as plt


# arbitrary random FIR filter
FIR_FILT = np.random.randn(10)


def lms_filter(data, mu=0.01):
    output = np.zeros(data.shape[0] - FIR_FILT.shape[0], dtype=np.float)
    estimate = np.zeros_like(output)
    mse = np.zeros_like(output)

    filt_hat = np.zeros((output.shape[0]+1, FIR_FILT.shape[0]))
    for i in range(output.shape[0]):
        # compute MSE
        mse[i] = np.dot(FIR_FILT, FIR_FILT) - \
                 2 * np.dot(FIR_FILT, filt_hat[i]) + \
                 np.dot(filt_hat[i], filt_hat[i])

        # compute error
        xi = data[i:i + FIR_FILT.shape[0]]
        output[i] = np.sum(FIR_FILT * xi)
        estimate[i] = np.sum(filt_hat[i] * xi)
        error = output[i] - estimate[i]

        # update filter
        filt_hat[i+1] = filt_hat[i] + 2 * mu * error * xi

    return output, estimate, mse, filt_hat

def rls_filter(data, alpha=0.5, delta=1e-2):
    output = np.zeros(data.shape[0] - FIR_FILT.shape[0], dtype=np.float)
    estimate = np.zeros_like(output)
    mse = np.zeros_like(output)

    R_hat = np.eye(FIR_FILT.shape[0]) * delta
    filt_hat = np.zeros((output.shape[0]+1, FIR_FILT.shape[0]))
    for i in range(output.shape[0]):
        # compute MSE
        mse[i] = np.dot(FIR_FILT, FIR_FILT) - \
                 2 * np.dot(FIR_FILT, filt_hat[i]) + \
                 np.dot(filt_hat[i], filt_hat[i])

        # compute error
        xi = data[i:i + FIR_FILT.shape[0]]
        output[i] = np.sum(FIR_FILT * xi)
        estimate[i] = np.sum(filt_hat[i] * xi)
        error = output[i] - estimate[i]

        # update filter
        R_hat = alpha * R_hat + np.outer(xi, xi)
        filt_hat[i+1] = filt_hat[i] + np.linalg.inv(R_hat) @ xi * error

    return output, estimate, mse, filt_hat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-samples', type=int, default=1000,
                        help='number of data samples to simulate (default to 1000)')
    parser.add_argument('-m', '--method', type=str, choices=['lms', 'rls'], default='lms',
                        help='which adaptive algorithm to use for optimization (default to lms)')
    parser.add_argument('--mu', type=float, default=1e-2,
                        help='gradient descent rate for LMS algorithm (default to 1e-2)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='exponential weighting of RLS algorithm (default to 0.5)')
    parser.add_argument('--delta', type=float, default=1e-3,
                        help='scalar multiplier for initializing correlation matrix for LMS (default to 1e-3)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data = np.random.randn(args.num_samples)

    if args.method == 'lms':
        output, esimate, mse, filt_hat = lms_filter(data, args.mu)
    else:
        output, esimate, mse, filt_hat = rls_filter(data, args.alpha, args.delta)

    plt.figure(figsize=(15,10))
    # plot MSE convergence
    plt.subplot(211)
    plt.plot(mse)
    plt.ylabel('MSE')
    plt.title(f'MSE Covergence')

    # plot weight trajectory
    plt.subplot(212)
    plt.plot(filt_hat[:, 2], label=f'w3 (ground truth {FIR_FILT[2]:.4f})')
    plt.plot(filt_hat[:, 7], label=f'w8 (ground truth {FIR_FILT[7]:.4f})')
    plt.title('Weight trajectory of w3 and w8')
    plt.xlabel('sample')
    plt.ylabel('Weight')
    plt.legend()

    if args.method == 'lms':
        plt.suptitle('LMS Alorithm with $\\mu$ = {}'.format(args.mu))
    else:
        plt.suptitle('RLS Alorithm with $\\alpha$ = {}, $\\delta$ = {}'.format(args.mu, args.delta))

    plt.show()
