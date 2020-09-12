from numba import njit
import numpy as np

############################################################################
############################################################################
########################## Window Implementations ##########################
############################################################################
############################################################################


def window_rectangular(width):
    """Returns a normalized rectangular window of length `width`.

    Constructs and returns a numpy array representing a normalized
    rectangular window of length `width`. The window is normalized
    such that the sum of the values is 1.
    In practice, this is just a constant array taking value `1/width`.

    Args:
        width: The length of the rectangular window

    Returns:
        A Numpy array of length width representing a rectangular window.
    """
    return np.ones(width) / width


def window_bartlett(width):
    """Returns a normalized rectangular window of length `width`.

    Constructs and returns a numpy array representing a normalized
    Bartlett window of length `width`. The window is normalized
    such that the sum of the values is 1.
    The unnormalized window follows the equation
    :math:`1 - \\left|\\frac{2n - (\\text{width} - 1)}{\\text{width}-1}\\right|`

    Args:
        width: The length of the Bartlett window

    Returns:
        A Numpy array of length width representing a Bartlett window.
    """
    window = 1 - np.abs((np.arange(width, dtype=np.float) * 2 - (width - 1)) / (width - 1))
    return window / window.sum()


def window_hann(width):
    """Returns a normalized rectangular window of length `width`.

    Constructs and returns a numpy array representing a normalized
    Hann window of length `width`. The window is normalized
    such that the sum of the values is 1.
    The unnormalized window follows the equation
    :math:`\\sin^2\\left(\\frac{n\\pi}{\\text{width-1}}\\right)`

    Args:
        width: The length of the Hann window

    Returns:
        A Numpy array of length width representing a Hann window.
    """
    window = np.square(np.sin(np.arange(width, dtype=np.float) * np.pi / (width - 1)))
    return window / window.sum()


############################################################################
############################################################################
####################### Convolution Implementations ########################
############################################################################
############################################################################

# If you aren't using numba, remove the @njit line
# Also removes IEEE Floating point compliance
@njit(fastmath=True)
def convolve_direct(x,y):
    """Completes a circular convolution of two numpy arrays.

    Completes a circular convolution of two numpy arrays based
    on the definition of circular convolution,
    :math:`out[n] = \\sum_{i=0}^{N-1} x[i]y[n-i \\mod N]`

    Args:
        x: numpy array
        y: numpy array

    Returns:
        The circular convolution of x and y
    """
    ret = np.zeros_like(x)
    for i in range(ret.shape[0]):
        y_idx = (ret.shape[0] + i - np.arange(ret.shape[0])) % ret.shape[0]
        ret[i] = np.sum(x * y[y_idx])

    return ret


def convolve_fft(x, y):
    """Completes a circular convolution of two numpy arrays.

    Completes a circular convolution of two numpy arrays through
    the use of the Discrete Fourier Transform (DFT)

    Args:
        x: numpy array
        y: numpy array

    Returns:
        The circular convolution of x and y
    """
    x_fft = np.fft.fft(x)
    y_fft = np.fft.fft(y)
    ret_fft = x_fft * y_fft

    return np.real(np.fft.ifft(ret_fft))


############################################################################
############################################################################
########################### Helper Functions ###############################
############################################################################
############################################################################

# No need to touch this function
def padwindow(window,width):
    """ Zero pads the provided window to the requested width

    Given a numpy array representing a window function, pads
    the array on either side with zeros such that, after an
    fftshift, the window is properly centered for use as
    a moving average.

    Args:
        window: Numpy array representing a window function
        width: Final width

    Returns:
        A Numpy array of length width with the window placed
        in the center.
    """
    out = np.zeros(width)
    offset = int(np.ceil((width-len(window))/2))
    assignment = np.arange(offset, offset+len(window))
    out[assignment] = window
    return out
