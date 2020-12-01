import numpy as np
import tensorflow as tf

from dnn.utils.params import ParamDict as o

class DynamicsDataPipeline:

    DEFAULT_PARAMS=o(
        # batch size for training
        batch_size=4096*4,
        # window for prediction horizon
        window_size=33,
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params

    def get_dataset(self, fpath, train=True):
        data_numpy = np.load(fpath)
        data_dict = { name: data_numpy[name] for name in data_numpy.dtype.names }
        return data_dict

    def get_windowed_data(self, data: np.ndarray) -> np.ndarray:
        num_sample = data.shape[0]
        buff = []
        for i in range(self.p.window_size):
            buff.append(data[i:i + num_sample - self.p.window_size, ...])

        return np.stack(buff, axis=1)
