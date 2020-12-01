import typing as T
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL

from dnn.utils.params import ParamDict as o

class NonlinearLibrary(KL.Layer):
    def __init__(self,
        num_states: int,
        order: int,
        sinusoid: bool,
        exponential: bool,
        absolute: bool,
        **kwargs: T.Any,
    ):
        super(NonlinearLibrary, self).__init__(**kwargs)
        self.num_states = num_states
        self.order = order
        self.sinusoid = sinusoid
        self.exponential = exponential
        self.absolute = absolute

    def get_polynomial(self, degree: int, x_ac: tf.Tensor) -> tf.Tensor:
        if degree == 1:
            return x_ac
        return x_ac

    def call(self, x_ac: tf.Tensor) -> tf.Tensor:
        features = []

        # build polynomials
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
        features.append(tf.concat([x_a[0] for x_a in x_a_arr], axis=-1))

        # build sinusoids
        if self.sinusoid:
            features.extend([tf.cos(x_ac), tf.sin(x_ac)])

        # build exponential
        if self.exponential:
            features.append(tf.exp(x_ac))

        # build abs
        if self.absolute:
            features.append(tf.abs(x_ac))

        return tf.concat(features, axis=-1)

class SparseLinearMap(KL.Layer):
    def __init__(self, num_states: int, l1: float, **kwargs: T.Any) -> None:
        super(SparseLinearMap, self).__init__(**kwargs)
        self.num_states = num_states
        self.l1 = l1

    def build(self, shape: tf.TensorShape) -> None:
        self.W = self.add_weight(
            name="W",
            shape=(shape[-1], self.num_states),
            trainable=self.trainable,
        )
        self.mask = self.add_weight(
            name="mask",
            shape=(shape[-1], self.num_states),
            initializer="ones",
            trainable=False,
        )
        # l1 sparsity penalty
        self.add_loss(lambda: self.l1 * tf.reduce_sum(tf.abs(self.W * self.mask)))

    def update_mask(self, threshold: float) -> None:
        self.mask.assign(tf.cast(tf.abs(self.W) >= threshold, tf.float32))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        self.add_metric(tf.reduce_sum(tf.abs(self.W * self.mask)), "L1 Loss")
        W = self.W * self.mask
        return tf.matmul(x, W)

class SINDYc(KL.Layer):
    def __init__(self,
        num_states: int,
        num_controls: int,
        l1: float,
        **kwargs: T.Any
    ) -> None:
        super(SINDYc, self).__init__(**kwargs)
        self.library = NonlinearLibrary(num_states + num_controls, 1, True, False, False)
        self.dynamics = SparseLinearMap(num_states, l1, trainable=self.trainable)

    def call(self, inputs: T.Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x_and_u = tf.concat(inputs, axis=-1)
        code = self.library(x_and_u)
        return self.dynamics(code)

class DynamicLoss(KL.Layer):
    def __init__(self,
        loss: T.Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = K.losses.mse,
        **kwargs: T.Any,
    ):
        super(DynamicLoss, self).__init__(**kwargs)
        self.loss = loss

    def call(self, inputs: T.Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x1_bx, x2_bx = inputs

        loss = self.loss(x1_bx, x2_bx)
        self.add_loss(tf.reduce_mean(loss))
        self.add_metric(loss, name="Prediction Loss")

        return loss

class SINDYcTrain:

    DEFAULT_PARAMS=o(
        l1=1e-7,
        loss=K.losses.mse,
    )

    def __init__(self, num_states: int, num_controls: int, params: o = DEFAULT_PARAMS) -> None:
        self.num_states = num_states
        self.num_controls = num_controls
        self.p = params
        self.sindy = SINDYc(
            num_states=num_states,
            num_controls=num_controls,
            l1=self.p.l1,
        )
        self.dynamic_loss = DynamicLoss(loss=self.p.loss)

    def build(self) -> K.Model:
        x = KL.Input((self.num_states,), name="x")
        x_dot = KL.Input((self.num_states,), name="x_dot")
        u = KL.Input((self.num_controls,), name="u")

        x_dot_hat = self.sindy((x, u))
        loss = self.dynamic_loss((x_dot_hat, x_dot))

        return K.Model(inputs=[x, x_dot, u],
                       outputs=loss)

