from dnn.sindy.params import PARAMS as DEFAULT_PARAMS
from dnn.sindy.model import IdentityLibrary, MultiplyLibrary, SinusoidLibrary, \
                            CollectionLibrary, ConstantLibrary, PolynomialLibrary

def construct_library():
    eye_lib = IdentityLibrary()
    sin_lib = SinusoidLibrary()
    mul_lib = MultiplyLibrary(eye_lib, sin_lib)

    const_lib = ConstantLibrary()

    return CollectionLibrary([eye_lib, mul_lib, const_lib])

PARAMS = DEFAULT_PARAMS(
    library=construct_library(),
    l2=20,
    threshold=1e-1,
)
