import tensorflow as tf
import tensorflow.keras as K

from dnn.sindy.model import SINDYcTrain
from dnn.utils.params import ParamDict as o

def default_lr_scheduler(epoch, lr):
    return 1e-0

trainer = o(
    num_epochs=200,
    log_freq=50,
    batch_size=16000,
    optimizer="adam",
    lr_scheduler=default_lr_scheduler,
)

model = SINDYcTrain.DEFAULT_PARAMS(
    threshold=.1,
)

PARAMS = o(
    trainer=trainer,
    model=model,
)
