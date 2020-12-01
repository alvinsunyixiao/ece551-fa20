import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL

from dnn.utils.params import ParamDict as o

class MLP(KL.Layer):
    def __init__(self, num_outputs, num_hidden, num_layers,
                 activation="relu", weight_decay=None, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layers = []
        regularizer = K.regularizers.l2(weight_decay) if weight_decay is not None else None
        for i in range(num_layers):
            self.tmp = KL.Dense(
                units=num_outputs if i == num_layers - 1 else num_hidden,
                activation="linear" if i == num_layers - 1 else activation,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                name=f'dense_{i}',
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
        x_loss=K.losses.mse,
        u_loss=K.losses.mse,
        z_loss=K.losses.mse,
        **kwargs,
    ):
        super(MultiLoss, self).__init__(**kwargs)
        self.num_states = num_states
        self.num_controls = num_controls
        self.num_state_code = num_state_code
        self.num_control_code = num_control_code
        self.x_loss = x_loss
        self.u_loss = u_loss
        self.z_loss = z_loss

        self.state_auto_encoder = AutoEncoder(
            num_outputs=num_states,
            num_code=num_state_code,
            num_hidden=num_hidden,
            num_layers=num_layers,
            weight_decay=weight_decay,
            name="state_auto_encoder",
            trainable=self.trainable,
        )

        self.control_auto_encoder = AutoEncoder(
            num_outputs=num_controls,
            num_code=num_control_code,
            num_hidden=num_hidden,
            num_layers=num_layers,
            weight_decay=weight_decay,
            name="control_auto_encoder",
            trainable=self.trainable,
        )

    def build(self, shapes):
        self.A = self.add_weight(
            name="A",
            shape=(self.num_state_code, self.num_state_code),
            trainable=self.trainable,
        )
        self.B = self.add_weight(
            name="B",
            shape=(self.num_control_code, self.num_state_code),
            trainable=self.trainable,
        )

    def call(self, inputs):
        dx_bx, x_bx, u_bu = inputs

        # pre-compute encoding / decoding space
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_bx)
            z_bz, x_hat_bx = self.state_auto_encoder(x_bx)
            v_bv, u_hat_bu = self.control_auto_encoder(u_bu)

        dz_dx_bzx = tape.batch_jacobian(z_bz, x_bx)
        dx_dt_bx1 = dx_bx[..., None]
        dz_dt_bz = tf.matmul(dz_dx_bzx, dx_dt_bx1)[..., 0]
        dz_dt_hat_bz = z_bz @ self.A + v_bv @ self.B

        # losses
        state_ae_loss = self.x_loss(x_bx, x_hat_bx)
        control_ae_loss = self.u_loss(u_bu, u_hat_bu)
        z_dot_loss = self.z_loss(dz_dt_bz, dz_dt_hat_bz)

        self.add_metric(state_ae_loss, name="State AE Loss")
        self.add_metric(control_ae_loss, name="Control AE Loss")
        self.add_metric(z_dot_loss, name="Z Dot Loss")

        # add losses
        total_loss = state_ae_loss + control_ae_loss + z_dot_loss
        self.add_loss(tf.reduce_mean(total_loss))
        self.add_metric(total_loss, name="Total Loss")

        return total_loss

class LCINDyTrain:

    DEFAULT_PARAMS=o(
        num_state_code=16,
        num_control_code=2,
        num_hidden=80,
        num_layers=2,
        weight_decay=None,
        x_loss=K.losses.mse,
        u_loss=K.losses.mse,
        z_loss=K.losses.mse,
    )

    def __init__(self, num_states, num_controls, params=DEFAULT_PARAMS):
        self.num_states = num_states
        self.num_controls = num_controls
        self.p = params
        self.multi_loss = MultiLoss(
            num_states=num_states,
            num_controls=num_controls,
            num_state_code=self.p.num_state_code,
            num_control_code=self.p.num_control_code,
            num_hidden=self.p.num_hidden,
            num_layers=self.p.num_layers,
            weight_decay=self.p.weight_decay,
            x_loss=self.p.x_loss,
            u_loss=self.p.u_loss,
            z_loss=self.p.z_loss
        )

    def build(self):
        x = KL.Input((self.num_states,), name="state")
        dx = KL.Input((self.num_states,), name="dstate")
        u = KL.Input((self.num_controls,), name="control")

        loss = self.multi_loss((dx, x, u))

        return K.Model(inputs=[dx, x, u],
                       outputs=loss)

