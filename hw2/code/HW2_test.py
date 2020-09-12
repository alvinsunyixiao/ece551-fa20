import numpy as np
from HW2 import CGS, MGS
import unittest

# Enables testing window functions and convolution
# Overrides the later fine-tuned controls

# Test Classical Gram-Schmidt
TEST_CGS = True

# Test Modified Gram-Schmidt
TEST_MGS= True

class OrthonormalTest_CGS(unittest.TestCase):
    """ Spot Checks for CGS implementations
        Tests are randomized """

    def test_orthogonal(self):
        a = 10*np.random.randn(10,10)
        ortho = CGS(a)
        test_mat = np.matmul(ortho,np.transpose(ortho))
        for i in range(10):
            test_mat[i,i] = 0
        np.testing.assert_allclose(test_mat, np.zeros((10,10)), atol=1e-6)

    def test_normal(self):
        a = 10*np.random.randn(10,10)
        ortho = CGS(a)
        length_list = [np.linalg.norm(ortho[i,:]) for i in range(10)]
        np.testing.assert_allclose(length_list, np.ones(10), atol=1e-6)

class OrthonormalTest_MGS(unittest.TestCase):
    """ Spot Checks for MGS implementations """

    def test_orthogonal(self):
        a = 10*np.random.randn(10,10)
        ortho = MGS(a)
        test_mat = np.matmul(ortho,np.transpose(ortho))
        for i in range(10):
            test_mat[i,i] = 0
        np.testing.assert_allclose(test_mat, np.zeros((10,10)), atol=1e-6)
    
    def test_normal(self):
        a = 10*np.random.randn(10,10)
        ortho = MGS(a)
        length_list = [np.linalg.norm(ortho[i,:]) for i in range(10)]
        np.testing.assert_allclose(length_list, np.ones(10), atol=1e-6)



# Convolution Tests
tests = unittest.TestSuite()

if TEST_CGS:
    tests.addTest(unittest.makeSuite(OrthonormalTest_CGS))
if TEST_MGS:
    tests.addTest(unittest.makeSuite(OrthonormalTest_MGS))


runner=unittest.TextTestRunner(verbosity=2)
runner.run(tests)