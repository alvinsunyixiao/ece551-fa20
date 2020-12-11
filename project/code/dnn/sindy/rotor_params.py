from dnn.sindy.params import PARAMS as DEFAULT_PARAMS
from dnn.sindy.model import IdentityLibrary, MultiplyLibrary, SinusoidLibrary, \
                            CollectionLibrary, ConstantLibrary

NUM_STATES = 6
NUM_CONTROLS = 2

def default_lr_scheduler(epoch, lr):
    if epoch < 1000:
        return 1e-3
    else:
        return 1e-4

def construct_library():
    eye_lib = IdentityLibrary()
    sin_lib = SinusoidLibrary(2)
    mul_lib = MultiplyLibrary(eye_lib, sin_lib)

    const_lib = ConstantLibrary()

    return CollectionLibrary([eye_lib, mul_lib, const_lib])

PARAMS = DEFAULT_PARAMS(
    trainer=DEFAULT_PARAMS.trainer(
        num_epochs=1600,
        batch_size=2**10,
        optimizer="adam",
        lr_scheduler=default_lr_scheduler,
    ),
    model=DEFAULT_PARAMS.model(
        l1=1e-3,
        library=construct_library(),
        threshold=1e-2,
    ),
)
