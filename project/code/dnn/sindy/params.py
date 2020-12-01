import tensorflow as tf
import tensorflow.keras as K

from dnn.sindy.model import SINDYcTrain
from dnn.sindy.data import DataPipeline
from dnn.utils.params import ParamDict as o

def default_lr_scheduler(epoch, lr):
    if epoch < 3:
        return 1e2
    elif epoch < 50:
        return 1e1
    else:
        return 1e-0

trainer = o(
    num_epochs=100,
    log_freq=50,
    optimizer="adam",
    lr_scheduler=default_lr_scheduler,
)

data = DataPipeline.DEFAULT_PARAMS

model = SINDYcTrain.DEFAULT_PARAMS(
    threshold=.1,
)

PARAMS=o(
    trainer=trainer,
    data=data,
    model=model,
)
