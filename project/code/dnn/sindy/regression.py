import argparse
import time
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from dnn.utils.mem import set_memory_growth
from dnn.utils.params import ParamDict
from dnn.sindy.model import SINDYcModel

class DynamicsRegressor:
    def __init__(self):
        self.args = self._parse_args()
        self.p = ParamDict.from_file(self.args.params)
        self.log_dir = self._get_logdir()

        self.train_data = np.load(os.path.join(self.args.data_dir, "train.npy"))
        self.val_data = np.load(os.path.join(self.args.data_dir, "val.npy"))

    def _reset_model(self):
        self.model = SINDYcModel(self.train_data["state"].shape[-1],
                                 self.train_data["control"].shape[-1],
                                 self.p.library)
        self.evaluate_mse(self.train_data)  # this build the model weights
        # keep track of what coefficients are dropped out
        self.mask_stats = self.model.mask.sum(axis=0)

        self.train_features = self.model.sindy.library(np.concatenate([
            self.train_data["state"], self.train_data["control"]], axis=-1)).numpy()
        self.val_features = self.model.sindy.library(np.concatenate([
            self.val_data["state"], self.val_data["control"]], axis=-1)).numpy()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--data-dir", type=str, required=True,
                            help="directory for the training / validation data")
        parser.add_argument("-p", "--params", type=str, required=True,
                            help="path to a param file")
        parser.add_argument("-l", "--log-dir", type=str, default="./logs",
                            help="directory for storing output logs and checkpoints")
        parser.add_argument("--tag", type=str, default=None, help="suffix for session dir name")
        parser.add_argument("--seed", type=int, default=42, help="random seed for noise generation")

        return parser.parse_args()

    def _get_logdir(self) -> str:
        logdir = os.path.join(self.args.log_dir, time.strftime("sess_%Y-%m-%d_%H-%M-%S"))
        if self.args.tag is not None:
            logdir += f'_{self.args.tag}'
        os.makedirs(logdir, exist_ok=True)

        return logdir

    def evaluate_mse(self, data):
        x_dot_hat = self.model((data["state"], data["control"]))
        mse = K.losses.mse(data["dstate"], x_dot_hat)
        return np.mean(mse.numpy())

    def ridge_regression(self, dx_noise):
        mask = self.model.mask
        W = self.model.W

        dx = self.train_data["dstate"] + dx_noise
        for i in range(self.train_data["state"].shape[-1]):
            ind = mask[:, i].astype(bool)
            gi = self.train_features[:, ind]
            W[:, i][ind] = np.linalg.solve(gi.T @ gi + self.p.l2 * np.eye(gi.shape[1]),
                    gi.T @ dx[:, i, None])[:, 0]
            mask[:, i][ind] = np.abs(W[:, i][ind]) >= self.p.threshold

        self.model.W = W
        self.model.mask = mask

        new_mask_stats = mask.sum(axis=0)
        terminate = np.all(new_mask_stats == self.mask_stats)
        self.mask_stats = new_mask_stats

        return terminate

    def run(self) -> None:
        # reproducable randomness
        np.random.seed(self.args.seed)

        for noise_std in self.p.noise_stds:
            self._reset_model()
            log_dir = os.path.join(self.log_dir, f"noise_{noise_std:.3g}")
            os.makedirs(log_dir, exist_ok=True)
            dx_noise = np.random.normal(scale=self.train_data["dstate"].std(axis=0) * noise_std,
                                        size=self.train_data["dstate"].shape)

            print(f"----------------- Noise {noise_std:.3g} --------------------")
            iteration = 0
            while True:
                terminate = self.ridge_regression(dx_noise)

                print(f"[Iteration {iteration}] "
                      f"Training MSE: {self.evaluate_mse(self.train_data):.4g} "
                      f"Validation MSE: {self.evaluate_mse(self.val_data):.4g} "
                      f"Coefficient Stats: {self.mask_stats}")
                self.model.save_weights(os.path.join(log_dir, f"{iteration}.h5"))

                if terminate:
                    break

                iteration += 1

if __name__ == "__main__":
    set_memory_growth()

    DynamicsRegressor().run()

