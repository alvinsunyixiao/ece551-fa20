import tensorflow as tf
import tensorflow.keras as K

from dnn.utils.data import DynamicsDataPipeline
from dnn.utils.model import LCINDyTrain
from dnn.utils.params import ParamDict as o

def default_lr_scheduler(epoch, lr):
    if epoch < 250:
        return 1e-3
    elif epoch < 450:
        return 1e-4
    else:
        return 1e-5

trainer=o(
    num_epochs=600,
    log_freq=50,
    optimizer="adam",
    lr_scheduler=default_lr_scheduler,
)

def default_x_loss(x1_a2, x2_a2):
    theta_loss = 1. - tf.cos(x1_a2[..., 0] - x2_a2[..., 0])
    theta_dot_loss = tf.square(x1_a2[..., 1] - x2_a2[..., 1])

    return theta_loss + theta_dot_loss

PARAMS=o(
    trainer=trainer,
    data=DynamicsDataPipeline.DEFAULT_PARAMS,
    model=LCINDyTrain.DEFAULT_PARAMS(
        x_loss=default_x_loss,
    ),
)
