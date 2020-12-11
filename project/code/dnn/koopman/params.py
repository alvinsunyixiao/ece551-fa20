import tensorflow as tf
import tensorflow.keras as K

from dnn.koopman.model import LCINDyTrain
from dnn.koopman.data import DynamicsDataPipeline
from dnn.utils.params import ParamDict as o

def default_lr_scheduler(epoch, lr):
    if epoch < 15000:
        return 1e-3
    elif epoch < 25000:
        return 1e-4
    else:
        return 1e-5

trainer=o(
    num_epochs=30000,
    log_freq=50,
    optimizer=K.optimizers.Adam(amsgrad=True),
    lr_scheduler=default_lr_scheduler,
)

PARAMS=o(
    trainer=trainer,
    data=DynamicsDataPipeline.DEFAULT_PARAMS,
    model=LCINDyTrain.DEFAULT_PARAMS,
)
