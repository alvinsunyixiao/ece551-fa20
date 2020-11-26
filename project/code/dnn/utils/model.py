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
                activation="linear" if i == num_layers - 1 else "relu",
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
        self.A = self.add_weight(name="A", shape=(self.num_state_code, self.num_state_code))
        self.B = self.add_weight(name="B", shape=(self.num_control_code, self.num_state_code))

    def call(self, inputs):
        x_bkx, u_bku = inputs

        # accumulative state
        z0_acc_bz = self.state_auto_encoder.encoder(x_bkx[:, :1])[:, 0]

        # pre-compute encoding / decoding space
        z_bkz, x_hat_bkx = self.state_auto_encoder(x_bkx)
        v_bkv, u_hat_bku = self.control_auto_encoder(u_bku)
        Bv_bkz = v_bkv @ self.B

        # losses
        b = tf.shape(x_bkx)[0]
        state_ae_loss = tf.reduce_mean(self.x_loss(x_bkx, x_hat_bkx), axis=-1)
        control_ae_loss = tf.reduce_mean(self.u_loss(u_bku, u_hat_bku), axis=-1)
        x_pred_loss = tf.zeros(b)
        z_pred_loss = tf.zeros(b)

        # m-step prediction
        m = tf.shape(x_bkx)[1] - 1
        for i in tf.range(m):
            # dynamics prediciton loss
            z1_hat_bz = z0_acc_bz @ self.A + Bv_bkz[:, i]
            x1_hat_bx = self.state_auto_encoder.decoder(z1_hat_bz[:, None, :])[:, 0]
            z_pred_loss += self.z_loss(z_bkz[:, i + 1], z1_hat_bz)
            x_pred_loss += self.x_loss(x_bkx[:, i + 1], x1_hat_bx)

            # update accumulative states
            z0_acc_bz = z1_hat_bz

        # normalize loss
        m = tf.cast(m, tf.float32)
        x_pred_loss /= m
        z_pred_loss /= m

        self.add_metric(state_ae_loss, name="State AE Loss")
        self.add_metric(control_ae_loss, name="Control AE Loss")
        self.add_metric(z_pred_loss, name="Z Prediction Loss")
        self.add_metric(x_pred_loss, name="X Prediction Loss")

        # add losses
        total_loss = state_ae_loss + control_ae_loss + z_pred_loss + x_pred_loss
        self.add_loss(tf.reduce_mean(total_loss))
        self.add_metric(total_loss, name="Total Loss")

        return total_loss

class LCINDyTrain:

    DEFAULT_PARAMS=o(
        num_state_code=32,
        num_control_code=4,
        num_hidden=128,
        num_layers=3,
        weight_decay=1e-14,
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
        x = KL.Input((None, self.num_states), name="x")
        u = KL.Input((None, self.num_controls), name="u")

        loss = self.multi_loss((x, u))

        return K.Model(inputs=[x, u],
                       outputs=loss)

