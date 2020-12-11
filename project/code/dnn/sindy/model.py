import typing as T
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL

from dnn.utils.params import ParamDict as o

class BaseLibrary(KL.Layer):
    def build(self, input_shape: tf.TensorShape) -> None:
        self.num_states = input_shape[-1]

class IdentityLibrary(BaseLibrary):
    def __init__(self, **kwargs: T.Any):
        super(IdentityLibrary, self).__init__(**kwargs)

    def call(self, x_ac: tf.Tensor) -> tf.Tensor:
        return x_ac

class ConstantLibrary(BaseLibrary):
    def __init__(self, **kwargs: T.Any):
        super(ConstantLibrary, self).__init__(**kwargs)

    def call(self, x_ac: tf.Tensor) -> tf.Tensor:
        return tf.ones_like(x_ac[..., 0])[..., tf.newaxis]

class PolynomialLibrary(BaseLibrary):
    def __init__(self, order: int, **kwargs: T.Any):
        super(PolynomialLibrary, self).__init__(**kwargs)
        self.order = order

    def call(self, x_ac: tf.Tensor) -> tf.Tensor:
        x_a_arr = tf.TensorArray(tf.float32, size=self.order,)
        x_a_arr = [[x_a[..., None] for x_a in tf.unstack(x_ac, axis=-1)]]
        for order in range(1, self.order + 1):
            tmp = []
            if order > 1:
                for i in range(self.num_states):
                    tmp.append(x_ac[..., i, None] * x_a_arr[order - 2][i])
                x_a_arr.append(tmp)
            for i in range(self.num_states - 2, -1, -1):
                x_a_arr[order - 1][i] = tf.concat([x_a_arr[order - 1][i],
                                                   x_a_arr[order - 1][i + 1]], axis=-1)

        return tf.concat([x_a[0] for x_a in x_a_arr], axis=-1)

class SinusoidLibrary(BaseLibrary):
    def __init__(self, index: T.Optional[int] = None, **kwargs: T.Any):
        super(SinusoidLibrary, self).__init__(**kwargs)
        self.index = index

    def call(self, x_ac: tf.Tensor) -> tf.Tensor:
        if self.index is None:
            return tf.concat([tf.cos(x_ac), tf.sin(x_ac)], axis=-1)
        else:
            x_a = x_ac[..., self.index]
            return tf.stack([tf.cos(x_a), tf.sin(x_a)], axis=-1)

class ComposedLibrary(BaseLibrary):
    def __init__(self, libraries: T.Sequence[BaseLibrary], **kwargs: T.Any):
        super(ComposedLibrary, self).__init__(**kwargs)
        self.libraries = libraries
        for library in libraries:
            self.tmp = library  # sub layer registration

    def call(self, x_ac: tf.Tensor) -> tf.Tensor:
        for library in self.libraries:
            x_ac = library(x_ac)
        return x_ac

class CollectionLibrary(BaseLibrary):
    def __init__(self, libraries: T.Sequence[BaseLibrary], **kwargs: T.Any):
        super(CollectionLibrary, self).__init__(**kwargs)
        self.libraries = libraries
        for library in libraries:
            self.tmp = library  # sub layer registration

    def call(self, x_ac: tf.Tensor) -> tf.Tensor:
        ret = []
        for library in self.libraries:
            ret.append(library(x_ac))

        return tf.concat(ret, axis=-1)

class DivisionLibrary(BaseLibrary):
    def __init__(self, lib1: BaseLibrary, lib2: BaseLibrary, **kwargs: T.Any):
        super(DivisionLibrary, self).__init__(**kwargs)
        self.lib1 = lib1
        self.lib2 = lib2

    def call(self, x_ac: tf.Tensor) -> tf.Tensor:
        code1 = tf.expand_dims(self.lib1(x_ac), axis=-1)
        code2 = tf.expand_dims(self.lib2(x_ac), axis=-2)

        c1 = tf.shape(code1)[-2]
        c2 = tf.shape(code2)[-1]
        output_shp = tf.concat([tf.shape(x_ac)[:-1], [c1 * c2]], axis=0)

        return tf.reshape(tf.math.divide_no_nan(code1, code2), output_shp)

class MultiplyLibrary(BaseLibrary):
    def __init__(self, lib1: BaseLibrary, lib2: BaseLibrary, **kwargs: T.Any):
        super(MultiplyLibrary, self).__init__(**kwargs)
        self.lib1 = lib1
        self.lib2 = lib2

    def call(self, x_ac: tf.Tensor) -> tf.Tensor:
        code1 = tf.expand_dims(self.lib1(x_ac), axis=-1)
        code2 = tf.expand_dims(self.lib2(x_ac), axis=-2)

        c1 = code1.shape[-2]
        c2 = code2.shape[-1]

        return tf.reshape(code1 * code2, (-1, c1 * c2))

class SINDYc(KL.Layer):
    def __init__(self,
        library: BaseLibrary,
        **kwargs: T.Any
    ) -> None:
        super(SINDYc, self).__init__(**kwargs)
        self.library = library

    def build(self, shapes: T.Tuple[tf.TensorShape, tf.TensorShape]) -> None:
        num_states = shapes[0][-1]
        num_controls = shapes[1][-1]
        num_features = self.library.compute_output_shape((None, num_states + num_controls))[-1]

        self.W = self.add_weight(
            name="W",
            shape=(num_features, num_states),
            trainable=self.trainable,
        )
        self.mask = self.add_weight(
            name="mask",
            shape=(num_features, num_states),
            initializer="ones",
            trainable=False,
        )

    def call(self, inputs: T.Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x_and_u = tf.concat(inputs, axis=-1)

        code_bc = self.library(x_and_u)

        return tf.matmul(code_bc, self.W * self.mask)

class SINDYcModel:
    def __init__(self, num_states: int, num_controls: int, library: BaseLibrary):
        self.sindy = SINDYc(library)
        self.num_states = num_states
        self.num_controls = num_controls
        self.model = self._build_model()

    def save_weights(self, fpath):
        self.model.save_weights(fpath)

    def load_weights(self, fpath):
        self.model.load_weights(fpath)

    @property
    def W(self):
        return self.sindy.W.numpy()

    @W.setter
    def W(self, value):
        self.sindy.W.assign(value)

    @property
    def mask(self):
        return self.sindy.mask.numpy()

    @mask.setter
    def mask(self, value):
        self.sindy.mask.assign(value)

    def __call__(self, inputs):
        return self.sindy(inputs)

    def _build_model(self) -> K.Model:
        x = KL.Input(shape=(self.num_states,))
        u = KL.Input(shape=(self.num_controls,))
        dx = self.sindy((x, u))

        return K.Model(inputs=[x, u], outputs=[dx])

