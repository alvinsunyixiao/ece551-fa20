import numpy as np
from HW1 import convolve_fft, convolve_direct, window_rectangular, window_hann, window_bartlett
import unittest

# Enables testing window functions and convolution
# Overrides the later fine-tuned controls
TEST_WINDOWS = True
TEST_CONVOLUTION = True

# Disable Individual Window Tests
TEST_RECTANGULAR = True
TEST_BARTLETT = True
TEST_HANN = True

# Disable Individual Convolution Tests
TEST_FFT = True
TEST_DIRECT = True

class ConvolveTest_fft(unittest.TestCase):
    """ Sport Checks for convolution implementations """

    def test_fft_ID(self):
        a = np.array([1, 0, 0, 0, 0, 0])
        b = np.array([1, 0, 0, 0, 0, 0])
        predict_ab = np.array([1, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(convolve_fft(a,b), predict_ab, atol=1e-8)

    def test_fft_ID2(self):
        a = np.array([1, 0, 0, 0, 0, 0])
        c = np.array([1, 0, 1, 0, 0, 0])
        predict_ac = np.array([1, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(convolve_fft(a,c), predict_ac, atol=1e-8)

    def test_fft_forward(self):
        c = np.array([1, 0, 1, 0, 0, 0])
        predict_cc = np.array([1, 0, 2, 0, 1, 0])
        np.testing.assert_allclose(convolve_fft(c,c), predict_cc, atol=1e-8)
    
    def test_fft_backward(self):
        d = np.array([1, 0, 0, 0, 1, 0])
        predict_dd = np.array([1, 0, 1, 0, 2, 0])
        np.testing.assert_allclose(convolve_fft(d,d), predict_dd, atol=1e-8)
        
    def test_fft_center(self):
        c = np.array([1, 0, 1, 0, 0, 0])
        d = np.array([1, 0, 0, 0, 1, 0])
        predict_cd = np.array([2, 0, 1, 0, 1, 0])
        np.testing.assert_allclose(convolve_fft(c,d), predict_cd, atol=1e-8)

class ConvolveTest_direct(unittest.TestCase):
    """ Sport Checks for convolution implementations """
    def test_direct_ID(self):
        a = np.array([1, 0, 0, 0, 0, 0])
        b = np.array([1, 0, 0, 0, 0, 0])
        predict_ab = np.array([1, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(convolve_direct(a,b), predict_ab)

    def test_direct_ID2(self):
        a = np.array([1, 0, 0, 0, 0, 0])
        c = np.array([1, 0, 1, 0, 0, 0])
        predict_ac = np.array([1, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(convolve_direct(a,c), predict_ac)

    def test_direct_forward(self):
        c = np.array([1, 0, 1, 0, 0, 0])
        predict_cc = np.array([1, 0, 2, 0, 1, 0])
        np.testing.assert_allclose(convolve_direct(c,c), predict_cc)
    
    def test_direct_backward(self):
        d = np.array([1, 0, 0, 0, 1, 0])
        predict_dd = np.array([1, 0, 1, 0, 2, 0])
        np.testing.assert_allclose(convolve_direct(d,d), predict_dd)
        
    def test_direct_center(self):
        c = np.array([1, 0, 1, 0, 0, 0])
        d = np.array([1, 0, 0, 0, 1, 0])
        predict_cd = np.array([2, 0, 1, 0, 1, 0])
        np.testing.assert_allclose(convolve_direct(c,d), predict_cd)


class WindowTest_rectangular(unittest.TestCase):
    """ Spot Checks for Window Functions """

    def test_rectangular(self):
        expect = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_allclose(window_rectangular(4), expect, atol=1e-8)

class WindowTest_hann(unittest.TestCase):
    """ Spot Checks for Window Functions """ 
    def test_hann_odd(self):
        expect = np.array([0, 0.25, 0.5, 0.25, 0])
        np.testing.assert_allclose(window_hann(5), expect, atol=1e-8)
    
    def test_hann_even(self):
        expect = np.array([0, 0.5, 0.5, 0])
        np.testing.assert_allclose(window_hann(4), expect, atol=1e-8)

class WindowTest_bartlett(unittest.TestCase):
    """ Spot Checks for Window Functions """  
    def test_bartlett_odd(self):
        expect = np.array([0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.16, 0.12, 0.08, 0.04, 0])
        np.testing.assert_allclose(window_bartlett(11), expect, atol=1e-8)
    
    def test_bartlett_even(self):
        expect = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.])
        np.testing.assert_allclose(window_bartlett(10), expect, atol=1e-8)

   
# Window Suite
window_tests = unittest.TestSuite()

if TEST_RECTANGULAR:
    window_tests.addTest(unittest.makeSuite(WindowTest_rectangular))
if TEST_BARTLETT:
    window_tests.addTest(unittest.makeSuite(WindowTest_bartlett))
if TEST_HANN:
    window_tests.addTest(unittest.makeSuite(WindowTest_hann))

# Convolution Tests
conv_tests = unittest.TestSuite()

if TEST_DIRECT:
    conv_tests.addTest(unittest.makeSuite(ConvolveTest_direct))
if TEST_FFT:
    conv_tests.addTest(unittest.makeSuite(ConvolveTest_fft))

# Combined Tests
tests = unittest.TestSuite()
if TEST_WINDOWS:
    tests.addTests(window_tests)
if TEST_CONVOLUTION:
    tests.addTests(conv_tests)

runner=unittest.TextTestRunner(verbosity=2)
runner.run(tests)