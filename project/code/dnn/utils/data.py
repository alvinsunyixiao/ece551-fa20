import tensorflow as tf

from dnn.utils.params import ParamDict as o

class DynamicsDataPipeline:

    DEFAULT_PARAMS=o(
        # batch size for training
        batch_size=4096,

        # window for prediction horizon
        window_size=16,
        # shuffle buffer size
        shuffle_size=100000,
        # number of parallel calls for dataset.map
        num_parallel_calls=16,
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params

    def get_dataset_from_dict(self, data_dict, train=True):
        assert isinstance(data_dict, dict), f'data_dict: {data_dict} is not a dictionary'

        dataset = tf.data.Dataset.from_tensor_slices(data_dict)
        dataset = dataset.window(self.p.window_size, shift=1, drop_remainder=True)
        if train:
            dataset = dataset.shuffle(self.p.shuffle_size)
        dataset = dataset.map(self._map_func, num_parallel_calls=self.p.num_parallel_calls)
        dataset = dataset.batch(self.p.batch_size)

        return dataset

    def get_numpy_dataset(self, data_numpy, train=True):
        data_dict = {name: data_numpy[name] for name in data_numpy.dtype.names}
        return self.get_dataset_from_dict(data_dict, train)

    def _map_func(self, window):
        return {
            key: tf.data.experimental.get_single_element(window[key].batch(self.p.window_size))
            for key in window
        }
