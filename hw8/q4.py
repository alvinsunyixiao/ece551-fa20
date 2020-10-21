import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

Y = np.matrix([
    2.3294301243940634 - 0.2296259966276833j,
    -0.4134361204474038 + 1.0082105392432457j,
    -0.3972955429876873 - 0.3077140046653479j,
    -0.328835156792498 + 0.3412442337892796j,
    0.5276539575334861 - 2.6968518609509466j,
    1.4068435099399519 + 1.609400966603145j,
    -1.4358532259748622 + 0.4052265092075893j,
    -0.6869999254086611 + 0.22150298448055664j,
]).T

NUM_SIGNALS = 2

if __name__ == '__main__':
    S = Y @ Y.H
    U, _, _ = np.linalg.svd(S, hermitian=True)

    # noise space
    E = U[:, NUM_SIGNALS:]

    # sweep angles
    thetas = np.linspace(0, np.pi, 1000)
    Pmu = np.zeros_like(thetas, dtype=np.complex128)
    n = np.arange(Y.shape[0], dtype=float)
    for i in range(thetas.shape[0]):
        steer = np.matrix(np.exp(1j * thetas[i] * n)).T
        Pmu[i] = 1. / (steer.H @ E @ E.H @ steer)

    # peak detection
    peaks, _ = signal.find_peaks(np.abs(Pmu), prominence=0.01)
    freqs = peaks / Pmu.shape[0]
    freqs_str = ', '.join([f'{freq} $\\pi$' for freq in freqs])

    # plot pseudo spectrum
    plt.plot(thetas, np.abs(Pmu))
    # plot peaks
    plt.scatter(thetas[peaks], np.abs(Pmu[peaks]), c='red')

    plt.title('Pseudo Spectrum Plot\nDetected Frequencies @ $\\theta =$ {}'.format(freqs_str))
    plt.xlabel('Frequency [rad]')
    plt.ylabel('$P_{MU}(\\theta)$')
    plt.show()
