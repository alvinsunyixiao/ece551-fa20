import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL

class MLP(KL.Layer):
    def __init__(self, num_outputs, num_hidden, num_layers,
                 activation="relu", weight_decay=None, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.layers = []
        for i in range(num_layers):
            regularizer = K.regularizers.l2(weight_decay) if weight_decay is not None else None
            self.tmp = KL.Dense(
                units=num_outputs if i == num_layers - 1 else num_hidden,
                activation="linear" if i == num_layers - 1 else activation,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                name=f'fc_{i}',
                trainable=self.trainable,
            )  # register sub layers
            self.layers.append(self.tmp)

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AutoEncoder(KL.Layer):
    def __init__(self, num_outputs, num_code, num_hidden, num_layers,
                 activation="relu", weight_decay=None, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.encoder = MLP(
            num_outputs=num_code,
            num_hidden=num_hidden,
            num_layers=num_layers,
            activation=activation,
            weight_decay=weight_decay,
            trainable=self.trainable,
            name="encoder",
        )
        self.decoder = MLP(
            num_outputs=num_outputs,
            num_hidden=num_hidden,
            num_layers=num_layers,
            activation=activation,
            weight_decay=weight_decay,
            trainable=self.trainable,
            name="decoder",
        )

    def call(self, x):
        code = self.encoder(x)
        output = self.decoder(code)

        return code, output

class SparseLinearMap(KL.Layer):
    def __init__(self, num_outputs, l1, **kwargs):
        super(SparseLinearMap, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.l1 = l1

    def build(self, input_shape):
        self.w = self.add_weight(name="weights", dtype="float32", trainable=self.trainable,
                                 shape=(input_shape[-1], self.num_outputs),
                                 regularizer=K.regularizers.l1(self.l1))
        self.mask = self.add_weight(name="mask", dtype="float32", trainable=False,
                                    shape=(input_shape[-1], self.num_outputs),
                                    initializer=K.initializers.ones())

    def call(self, x):
        return tf.matmul(x, self.w * self.mask)

class MultiLoss(KL.Layer):
    def __init__(
        self,
        num_states,
        num_controls,
        num_state_code=16,
        num_control_code=8,
        num_hidden=64,
        num_layers=4,
        weight_decay=1e-4,
        sparse_l1_decay=1e-4,
        **kwargs,
    ):
        super(MultiLoss, self).__init__(**kwargs)
        self.num_states = num_states
        self.num_controls = num_controls
        self.num_state_code = num_state_code
        self.num_control_code = num_control_code

        self.state_auto_encoder = AutoEncoder(
            num_outputs=num_states,
            num_code=num_state_code,
            num_hidden=num_hidden,
            num_layers=num_layers,
            weight_decay=weight_decay,
            name="state_auto_encoder",
            trainable=self.trainable,
        )

        self.control_encoder = MLP(
            num_outputs=num_control_code,
            num_hidden=num_hidden,
            num_layers=num_layers,
            weight_decay=weight_decay,
            name="control_encoder",
            trainable=self.trainable,
        )

        self.control_decoder = MLP(
            num_outputs=num_controls,
            num_hidden=num_hidden,
            num_layers=num_layers,
            weight_decay=weight_decay,
            name="control_decoder",
            trainable=self.trainable,
        )

        self.A = SparseLinearMap(num_state_code, sparse_l1_decay)
        self.B = SparseLinearMap(num_state_code, sparse_l1_decay)

    def call(self, inputs):
        x0_bx, u_bu, x1_bx = inputs

        # states
        z0_bz, x0_hat_bx = self.state_auto_encoder(x0_bx)
        z1_bz, _ = self.state_auto_encoder(x1_bx)
        state_ae_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x0_bx - x0_hat_bx), axis=-1))
        self.add_metric(state_ae_loss, name="State AE Loss")

        # control
        x0_and_u_bi = tf.concat([x0_bx, u_bu], axis=-1)
        v_bv = self.control_encoder(x0_and_u_bi)
        z0_and_v_bh = tf.concat([z0_bz, v_bv], axis=-1)
        u_hat_bu = self.control_decoder(z0_and_v_bh)
        control_ae_loss = tf.reduce_mean(tf.reduce_sum(tf.square(u_bu - u_hat_bu), axis=-1))
        self.add_metric(control_ae_loss, name="Control AE Loss")

        # dynamics
        z1_hat_bz = self.A(z0_bz) + self.B(v_bv)
        x1_hat_bx = self.state_auto_encoder.decoder(z1_hat_bz)
        z_loss = tf.reduce_mean(tf.reduce_sum(tf.square(z1_bz - z1_hat_bz), axis=-1))
        x_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x1_bx - x1_hat_bx), axis=-1))
        self.add_metric(z_loss, name="Prediction Z Loss")
        self.add_metric(x_loss, name="Prediction X Loss")

        # add losses
        total_loss = state_ae_loss + control_ae_loss + z_loss + x_loss
        self.add_loss(total_loss)
        self.add_metric(total_loss, name="Total Loss")

        return total_loss


class LCINDyTrain:
    def __init__(
        self,
        num_states,
        num_controls,
        num_state_code=16,
        num_control_code=8,
        num_hidden=64,
        num_layers=4,
        weight_decay=1e-4,
        sparse_l1_decay=1e-2,
    ):
        self.num_states = num_states
        self.num_controls = num_controls
        self.multi_loss = MultiLoss(num_states, num_controls, num_state_code, num_control_code,
                                    num_hidden, num_layers, weight_decay, sparse_l1_decay)

    def build(self):
        x0 = KL.Input((self.num_states,), name="x0")
        u = KL.Input((self.num_controls,), name="u")
        x1 = KL.Input((self.num_states,), name="x1")

        loss = self.multi_loss((x0, u, x1))

        return K.Model(inputs=[x0, u, x1],
                       outputs=loss)

