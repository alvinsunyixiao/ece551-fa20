import time
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

def get_phi(K, M):
    m, k = np.meshgrid(np.arange(M), np.arange(K), indexing='ij')
    return np.cos(2 * np.pi * k * m / M)

def get_tphi(M, N):
    n, m = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')
    return ((m == n) | (m == n + 1)).astype(np.int32)

def compute_tphi_matrix(tphi, x):
    return tphi @ x

def compute_tphi_fast(N, x):
    return x[:N] + x[1:N+1]

if __name__ == '__main__':
    # part a
    K = 4
    M = 10
    N = 8
    phi = get_phi(K, M)
    tphi = get_tphi(M, N)
    print(f'phi:\n{phi}\n')
    print(f'tphi:\n{tphi}\n')
    # part c
    psi = phi @ np.linalg.pinv(tphi @ phi)
    y = np.random.randn(K)
    x = phi @ y
    x_hat = psi @ tphi @ x
    mse = np.mean(np.square(x - x_hat))
    print(f'Recovery MSE: {mse}')
    # part b
    Ns = np.arange(10, 1000)
    matrix_times = np.zeros(Ns.shape[0])
    fast_times = np.zeros(Ns.shape[0])
    for i, N in enumerate(tqdm(Ns)):
        M = N + 2
        tphi = get_tphi(M, N)
        x = np.random.randn(M)
        t1 = time.time()
        for _ in range(10):
            a = compute_tphi_matrix(tphi, x)
        t2 = time.time()
        for _ in range(10):
            b = compute_tphi_fast(N, x)
        t3 = time.time()
        assert np.allclose(a, b)
        matrix_times[i] = (t2 - t1) / 10
        fast_times[i] = (t3 - t2) / 10
    plt.plot(Ns, matrix_times, label='matrix multiplication')
    plt.plot(Ns, fast_times, label='vector addition')
    plt.legend()
    plt.title('Run Time Analysis')
    plt.xlabel('N')
    plt.ylabel('time [s]')
    plt.show()
