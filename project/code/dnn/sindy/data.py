import tensorflow as tf

from dnn.utils.params import ParamDict as o

class DataPipeline:

    DEFAULT_PARAMS=o(
        # batch size for training
        batch_size=4096*8,
        shuffle_size=1000000,
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params

    def get_dataset_from_dict(self, data_dict, train=True):
        assert isinstance(data_dict, dict), f'data_dict: {data_dict} is not a dictionary'

        dataset = tf.data.Dataset.from_tensor_slices(data_dict)
        if train:
            dataset = dataset.shuffle(self.p.shuffle_size)
        dataset = dataset.batch(self.p.batch_size)

        return dataset

    def get_numpy_dataset(self, data_numpy, train=True):
        data_dict = {name: data_numpy[name] for name in data_numpy.dtype.names}
        return self.get_dataset_from_dict(data_dict, train)

