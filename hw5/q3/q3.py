import numpy as np
import matplotlib.pyplot as plt

def G0(w):
    return 1

def G1(w, alpha):
    return np.exp(-1j * w * (1 + alpha))

def G2(w, beta):
    return np.exp(-1j * w * (1 + beta))

def G(w, alpha, beta):
    return np.array([
        [G0(w), G0(w - 2/3*np.pi), G0(w - 4/3*np.pi)],
        [G1(w, alpha), G1(w - 2/3*np.pi, alpha), G1(w - 4/3*np.pi, alpha)],
        [G2(w, beta), G2(w - 2/3*np.pi, beta), G2(w - 4/3*np.pi, beta)],
    ])

if __name__ == '__main__':
    omega = 1
    alpha = 1
    betas = np.linspace(-0.5, 0.5, 100, endpoint=False)
    conds = np.zeros_like(betas)

    for i, beta in enumerate(betas):
        conds[i] = np.linalg.cond(G(omega, alpha, beta))

    plt.plot(betas, conds)
    plt.title('Condition Number of $G(\omega)$ vs $\\beta$')
    plt.xlabel('$\\beta$')
    plt.ylabel('$\kappa(G(\omega))$')
    plt.show()
