import tensorflow.keras as K

from dnn.utils.data import DynamicsDataPipeline
from dnn.utils.model import LCINDyTrain
from dnn.utils.params import ParamDict as o

def default_lr_scheduler(epoch, lr):
    if epoch < 60:
        return 1e-3
    elif epoch < 85:
        return 1e-4
    else:
        return 1e-5

trainer=o(
    num_epochs=100,
    log_freq=50,
    optimizer="adam",
    lr_scheduler=default_lr_scheduler,
)

PARAMS=o(
    trainer=trainer,
    data=DynamicsDataPipeline.DEFAULT_PARAMS,
    model=LCINDyTrain.DEFAULT_PARAMS,
)
