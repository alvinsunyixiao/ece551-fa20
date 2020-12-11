from dnn.sindy.params import PARAMS as DEFAULT_PARAMS
from dnn.sindy.model import SinusoidLibrary, CollectionLibrary, IdentityLibrary

NUM_STATES = 2
NUM_CONTROLS = 1

def default_lr_scheduler(epoch, lr):
    if epoch < 200:
        return 1e0
    else:
        return 1e-1

def construct_library():
    eye_lib = IdentityLibrary()
    sin_lib = SinusoidLibrary()
    col_lib = CollectionLibrary([eye_lib, sin_lib])

    return col_lib

PARAMS = DEFAULT_PARAMS(
    trainer=DEFAULT_PARAMS.trainer(
        num_epochs=400,
        batch_size=2**12,
        lr_scheduler=default_lr_scheduler,
    ),
    model=DEFAULT_PARAMS.model(
        library=construct_library(),
        threshold=1e-1,
    ),
)
