from dnn.sindy.params import PARAMS as DEFAULT_PARAMS
from dnn.sindy.model import PolynomialLibrary

PARAMS = DEFAULT_PARAMS(
    library=PolynomialLibrary(2),
    l2=1,
    threshold=1e-1,
)
