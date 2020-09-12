from numba import njit
import numpy as np

############################################################################
############################################################################
###################### Gram-Schmidt Implementations ########################
############################################################################
############################################################################

# Note that numpy stores data in a row-major ordering.
# For caching reasons, we will work with the rowspaces of matrices.
# If implemented in something like Matlab, Julia, or R, you would
# want to reorder the code to work with column-major ordering.

@njit
def CGS(A):
    """Returns an Orthonormal basis spanning the rowspace of A.

    Constructs and returns an orthonormal basis spanning the
    rowspace of A. The basis is returned as another numpy
    matrix. This function implements classical Gram-Schmidt.

    Args:
        A: A matrix with rows corresponding to the collection of vectors

    Returns:
        A numpy matrix with rows corresponding to orthonormal basis vectors.
    """
    basis = np.copy(A)
    for i in range(A.shape[0]):
        projection = np.zeros_like(A[i])
        for j in range(0, i):
            projection += np.sum(A[i] * basis[j]) * basis[j]
        r = A[i] - projection
        basis[i] = r / np.linalg.norm(r)

    return basis

@njit
def MGS(A):
    """Returns an Orthonormal basis spanning the rowspace of A.

    Constructs and returns an orthonormal basis spanning the
    rowspace of A. The basis is returned as another numpy
    matrix. This function implements modified Gram-Schmidt.

    Args:
        A: A matrix with rows corresponding to the collection of vectors

    Returns:
        A numpy matrix with rows corresponding to orthonormal basis vectors.
    """
    basis = np.copy(A)
    for i in range(A.shape[0]):
        basis[i] = basis[i] / np.linalg.norm(basis[i])
        for j in range(i+1, A.shape[0]):
            basis[j] = basis[j] - np.sum(basis[i] * basis[j]) * basis[i]
        #proj_norm = np.sum(basis[i+1:] * basis[np.newaxis, i], axis=-1, keepdims=True)
        #basis[i+1:] = basis[i+1:] - proj_norm * basis[np.newaxis, i]

    return basis
