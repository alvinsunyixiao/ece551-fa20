import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL

from dnn.utils.params import ParamDict as o

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
                kernel_initializer=K.initializers.glorot_normal(),
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

        self.A = KL.Dense(num_state_code, use_bias=False, kernel_initializer=K.initializers.orthogonal())
        self.B = KL.Dense(num_state_code, use_bias=False, kernel_initializer=K.initializers.orthogonal())

    def call(self, inputs):
        x_bkx, u_bku = inputs

        # losses
        b = tf.shape(x_bkx)[0]
        state_ae_loss = tf.zeros(b)
        control_ae_loss = tf.zeros(b)
        x_pred_loss = tf.zeros(b)
        z_pred_loss = tf.zeros(b)

        # accumulative state
        z0_acc_bz = self.state_auto_encoder.encoder(x_bkx[:, 0])

        # m-step prediction
        m = tf.shape(x_bkx)[1] - 1
        for i in tf.range(m):
            x0_bx = x_bkx[:, i]
            u_bu = u_bku[:, i]

            # state auto encoder loss
            z0_bz, x0_hat_bx = self.state_auto_encoder(x0_bx)
            state_ae_loss += self.x_loss(x0_bx, x0_hat_bx)

            # control auto encoder loss
            v_bv, u_hat_bu = self.control_auto_encoder(u_bu)
            control_ae_loss += self.u_loss(u_bu, u_hat_bu)

            # dynamics prediciton loss
            x1_bx = x_bkx[:, i + 1]
            z1_hat_bz = self.A(z0_acc_bz) + self.B(v_bv)
            x1_hat_bx = self.state_auto_encoder.decoder(z1_hat_bz)
            z1_bz = self.state_auto_encoder.encoder(x1_bx)
            z_pred_loss += self.z_loss(z1_bz, z1_hat_bz)
            x_pred_loss += self.x_loss(x1_bx, x1_hat_bx)

            # update accumulative states
            z0_acc_bz = z1_hat_bz

        # normalize loss
        #m = tf.cast(m, tf.float32)
        #state_ae_loss /= m
        #control_ae_loss /= m
        #x_pred_loss /= m
        #z_pred_loss /= m

        self.add_metric(state_ae_loss, name="State AE Loss")
        self.add_metric(control_ae_loss, name="Control AE Loss")
        self.add_metric(z_pred_loss, name="Z Prediction Loss")
        self.add_metric(x_pred_loss, name="X Prediction Loss")

        # add losses
        total_loss = state_ae_loss + control_ae_loss + z_pred_loss + x_pred_loss
        self.add_loss(tf.reduce_mean(total_loss))
        self.add_metric(total_loss, name="Total Loss")

        return total_loss

def default_x_loss(x1_b3, x2_b3):
    theta1_b = tf.atan2(x1_b3[:, 1], x1_b3[:, 0])
    theta2_b = tf.atan2(x2_b3[:, 1], x2_b3[:, 0])

    loss_theta_b = 1. - tf.cos(theta1_b - theta2_b)
    loss_theta_dot_b = tf.square(x1_b3[:, 2] - x2_b3[:, 2])

    return loss_theta_b + loss_theta_dot_b


class LCINDyTrain:

    DEFAULT_PARAMS=o(
        num_state_code=32,
        num_control_code=8,
        num_hidden=256,
        num_layers=4,
        weight_decay=1e-4,
        x_loss=default_x_loss,
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

