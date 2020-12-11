from dnn.utils.params import ParamDict as o
from dnn.sindy.model import IdentityLibrary

PARAMS = o(
    noise_stds=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    library=IdentityLibrary(),
    l2=1,
    threshold=1e-2,
)
