from dnn.sindy.params import PARAMS as DEFAULT_PARAMS
from dnn.sindy.model import PolynomialLibrary, SinusoidLibrary, CollectionLibrary, \
                            ComposedLibrary, IdentityLibrary, DivisionLibrary

NUM_STATES = 3
NUM_CONTROLS = 1

def default_lr_scheduler(epoch, lr):
    if epoch < 200:
        return 1e0
    else:
        return 1e-1

PARAMS = DEFAULT_PARAMS(
    trainer=DEFAULT_PARAMS.trainer(
        num_epochs=400,
        batch_size=2**12,
        lr_scheduler=default_lr_scheduler,
    ),
    model=DEFAULT_PARAMS.model(
        library=PolynomialLibrary(2),
        threshold=1e-2,
    ),
)
