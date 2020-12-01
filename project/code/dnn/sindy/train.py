import argparse
import time
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from dnn.sindy.data import DataPipeline
from dnn.utils.mem import set_memory_growth
from dnn.utils.params import ParamDict
from dnn.sindy.model import SINDYcTrain, SINDYc

class PendulumDataPipeline(DataPipeline):
    def get_numpy_dataset(self, data_numpy, train=True):
        return self.get_dataset_from_dict({
            "x": data_numpy["state"],
            "x_dot": data_numpy["dstate"],
            "u": data_numpy["control"],
        })

class MaskUpdateCallback(K.callbacks.Callback):
    def __init__(self, sindy: SINDYc, threshold: float):
        self.sindy = sindy
        self.threshold = threshold

    def on_epoch_begin(self, epoch, logs=None):
        tf.print()
        tf.print(self.sindy.dynamics.W, summarize=-1)
        tf.print(self.sindy.dynamics.mask, summarize=-1)

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 10 and epoch % 5 == 0:
            self.sindy.dynamics.update_mask(self.threshold)

class PendulumTrainer:
    def __init__(self):
        self.args = self._parse_args()
        self.p = ParamDict.from_file(self.args.params)
        self.log_dir = self._get_logdir()

        data_pipe = PendulumDataPipeline(self.p.data)
        self.train_data = np.load(os.path.join(self.args.data_dir, "train.npy"))
        self.val_data = np.load(os.path.join(self.args.data_dir, "val.npy"))
        self.train_dataset = data_pipe.get_numpy_dataset(self.train_data, train=True)
        self.val_dataset = data_pipe.get_numpy_dataset(self.val_data, train=False)

        self.sindy_train = SINDYcTrain(
            num_states=self.train_dataset.element_spec['x'].shape[1],
            num_controls=self.train_dataset.element_spec['u'].shape[1],
            params=self.p.model
        )

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--data-dir", type=str, required=True,
                            help="directory for the training / validation data")
        parser.add_argument("-p", "--params", type=str, required=True,
                            help="path to a param file")
        parser.add_argument("-l", "--log-dir", type=str, default="./logs",
                            help="directory for storing output logs and checkpoints")
        parser.add_argument("--tag", type=str, default=None, help="suffix for session dir name")
        parser.add_argument("--load", type=str, default=None,
                            help="specify the path to the model checkpoint to continue upon")
        return parser.parse_args()

    def _get_logdir(self) -> str:
        logdir = os.path.join(self.args.log_dir, time.strftime("sess_%Y-%m-%d_%H-%M-%S"))
        if self.args.tag is not None:
            logdir += f'_{self.args.tag}'
        os.makedirs(logdir, exist_ok=True)

        return logdir

    def run(self) -> None:
        model = self.sindy_train.build()
        model.compile(self.p.trainer.optimizer)

        if self.args.load is not None:
            model.load_weights(self.args.load)

        callbacks = [
            K.callbacks.TensorBoard(
                log_dir=self.log_dir,
                update_freq=self.p.trainer.log_freq
            ),
            K.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.log_dir, "{epoch:02d}-{val_loss:.5f}.h5"),
                save_weights_only=True,
            ),
            K.callbacks.LearningRateScheduler(self.p.trainer.lr_scheduler),
            MaskUpdateCallback(self.sindy_train.sindy, self.p.model.threshold),
        ]

        model.fit(
            x=self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.p.trainer.num_epochs,
            callbacks=callbacks,
        )

if __name__ == "__main__":
    set_memory_growth()

    PendulumTrainer().run()
