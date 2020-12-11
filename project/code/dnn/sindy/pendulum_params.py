from dnn.sindy.params import PARAMS as DEFAULT_PARAMS
from dnn.sindy.model import SinusoidLibrary, CollectionLibrary, IdentityLibrary

def construct_library():
    eye_lib = IdentityLibrary()
    sin_lib = SinusoidLibrary()
    col_lib = CollectionLibrary([eye_lib, sin_lib])

    return col_lib

PARAMS = DEFAULT_PARAMS(
    library=construct_library(),
    l2=8,
    threshold=1e-1,
)
