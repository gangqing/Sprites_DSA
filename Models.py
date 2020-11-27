from tensorflow.compat import v1 as tf
import Framework as fm
import Config


class MyTensors(fm.Tensors):
    def compute_grads(self, opt):
        vars = tf.trainable_variables()
        vars_mes = [var for var in vars if "simple_z" not in var.name]
        # vars_f = [var for var in vars if "encode_f" in var.name or "encode_frame" in var.name]
        vars_simple_z = [var for var in vars if "simple_z" in var.name]
        vars = [vars_mes, vars_simple_z, vars_mes]

        grads = []
        for gpu_id, ts in enumerate(self.sub_ts):
            with tf.device('/gpu:%d' % gpu_id):
                grads.append([opt.compute_gradients(loss, vs) for vs, loss in zip(vars, ts.losses)])
                # grads.append([opt.compute_gradients(ts.losses[0], vars)])
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]


def conv2d_bn(inputs, filters, kernel_size, strides, padding, activation=None, training=True, name=""):
    conv2d = tf.layers.conv2d(inputs=inputs,
                              filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              name="conv2d" + name)  # [-1, 64, 64, 32]
    bn = tf.layers.batch_normalization(inputs=conv2d,
                                       axis=-1,  # 相当于axis = [1, 2, 3]，将产生64 * 64 * 32 个均值和方差
                                       training=training,
                                       name="c_bn" + name)  # [-1, 64, 64, 32]
    if activation is None:
        return bn
    return activation(features=bn, name="c_active" + name)


def dense_bn(inputs, units, training, activation=None, name=""):
    d = tf.layers.dense(inputs=inputs, units=units, name="dense" + name)  # [-1, 4096]
    b = tf.layers.batch_normalization(inputs=d, training=training, name="d_bn" + name)
    if activation is None:
        return b
    return activation(features=b, name="d_active" + name)


class SubTensor:
    def __init__(self, c: Config):
        self.c = c
        # 输入
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None, c.frame, c.input_size, c.input_size, 3],
                                name="x")  # x : [batch_size, 8, 64, 64, 3]
        self.inputs = [self.x]
        # encoder_frame
        x = tf.reshape(tensor=self.x, shape=[-1, c.input_size, c.input_size, 3])  # [batch_size * 8, 64, 64, 3]
        x = self.encode_frame(x, "encode_frame")  # [-1, 8, 2048]
        # encoder_F
        self.mean_f, self.logvar_f = self.encode_f(x, "encode_f")  # [-1, 256]
        self.f = self.reparameterize(self.mean_f, self.logvar_f, self.c.training)
        f = tf.reshape(self.f, [-1, 1, c.f_size])
        f = tf.tile(f, [1, c.frame, 1])  # [-1, 8, 256]
        # encoder_Z
        self.mean_z, self.logvar_z = self.encode_z(x, f, "encode_z")  # [-1, 8, 32]
        self.z = self.reparameterize(self.mean_z, self.logvar_z, self.c.training)
        # simple_Z
        self.generator_z_mean, self.generator_z_logvar = self.simple_z(self.c.batch_size, name="simple_z")
        # decoder_frame
        zf = tf.concat((self.z, f), axis=2)  # [-1, 8, 256 + 32]
        y = self.decode_frame(zf, "decode_frame")  # [-1, 64, 64, 3]
        # 输出和loss
        self.predict_y = tf.reshape(tensor=y, shape=[-1, c.frame, c.input_size, c.input_size, 3])  # [-1, 8, 64, 64, 3]
        self.loss()

    def loss(self):
        # [-1, 8, 64, 64, 3]
        mes = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - self.predict_y), axis=(1, 2, 3, 4)))
        # f : [-1, 256]
        kld_f = tf.reduce_mean(
            -0.5 * tf.reduce_sum(1 + self.logvar_f - tf.pow(self.mean_f, 2) - tf.exp(self.logvar_f), 1))
        z_post_var = tf.exp(self.logvar_z)
        z_prior_var = tf.exp(self.generator_z_logvar)
        kld_z = tf.reduce_mean(0.5 * tf.reduce_sum(self.generator_z_logvar - self.logvar_z +
                                                   (z_post_var + tf.pow(self.mean_z - self.generator_z_mean,
                                                                        2)) / z_prior_var - 1, axis=(1, 2)))
        loss = mes + kld_f
        self.losses = [loss, kld_z, loss]

    def encode_frame(self, x, name):
        """
        :param x: [batch_size * 8, 64, 64, 3]
        :return:  [-1, 8, 2048]
        """
        with tf.variable_scope(name):
            base_filter = 32
            strides = 1
            for i in range(4):
                x = conv2d_bn(inputs=x,
                              filters=base_filter,
                              kernel_size=3,
                              strides=strides,
                              padding="same",
                              activation=tf.nn.relu,
                              training=self.c.training,
                              name=str(i))
                base_filter *= 2  # 64, 128, 256, 512
                strides = 2

            h = tf.layers.flatten(inputs=x)  # [-1, 8 * 8 * 256]

            base_units = self.c.x_size * 2  # 2048 * 2
            for i in range(2):
                h = dense_bn(inputs=h,
                             units=base_units,
                             training=self.c.training,
                             activation=tf.nn.relu,
                             name=str(i))
                base_units //= 2

        return tf.reshape(h, [-1, self.c.frame, self.c.x_size])  # [batch_size, 8, 2048]

    def encode_f(self, x, name):
        """
        :param x: [-1, 8, 2048]
        :param name:
        :return: [-1, 256]
        """
        with tf.variable_scope(name):
            x = tf.layers.dense(x, self.c.hidden_size, name="dense1")  # [-1, 8, 512]

            cell_l2r = tf.nn.rnn_cell.LSTMCell(self.c.hidden_size, name="cell_l2r", state_is_tuple=False)
            cell_r2l = tf.nn.rnn_cell.LSTMCell(self.c.hidden_size, name="cell_r2l", state_is_tuple=False)
            batch_size = tf.shape(x)[0]
            state_l2r = cell_l2r.zero_state(batch_size, dtype=x.dtype)
            state_r2l = cell_r2l.zero_state(batch_size, dtype=x.dtype)
            for i in range(self.c.frame):
                y_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # y_l2r : [-1, 512]
                y_r2l, state_r2l = cell_r2l(x[:, self.c.frame - i - 1, :], state_r2l)  # y_r2l : [-1, 512]

            y = tf.concat((y_l2r, y_r2l), axis=1)  # [-1, 1024]
            mean_f = tf.layers.dense(inputs=y, units=self.c.f_size, name="dense_mean")  # [-1, 256]
            logvar_f = tf.layers.dense(inputs=y, units=self.c.f_size, name="dense_logvar")  # [-1, 256]
        return mean_f, logvar_f

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(shape=tf.shape(logvar), dtype=logvar.dtype)
            return mean + std * eps
        else:
            return mean + logvar * 0

    def encode_z(self, x, f, name):
        """
        :param x: [-1, 8, 2048]
        :param f: [-1, 8, 256]
        :param name:
        :return: [-1, 8, 32]
        """
        with tf.variable_scope(name):
            if self.c.factorised is True:
                features = tf.layers.dense(x, self.c.hidden_size, activation=tf.nn.relu,
                                           name="dense_1")  # [-1, 8, 512]
            else:
                xf = tf.concat((x, f), axis=2)  # [-1, 8, 2048 + 256]
                xf = tf.layers.dense(xf, self.c.hidden_size, name="dense_2")  # [-1, 8, 512]

                cell_l2r = tf.nn.rnn_cell.LSTMCell(self.c.f_size, name="cell_l2r", state_is_tuple=False)
                cell_r2l = tf.nn.rnn_cell.LSTMCell(self.c.f_size, name="cell_r2l", state_is_tuple=False)
                batch_size = tf.shape(xf)[0]
                state_l2r = cell_l2r.zero_state(batch_size, dtype=xf.dtype)
                state_r2l = cell_r2l.zero_state(batch_size, dtype=xf.dtype)
                y_l2r = []
                y_r2l = []  # [8, -1, 512]
                for i in range(self.c.frame):
                    yi_l2r, state_l2r = cell_l2r(xf[:, i, :], state_l2r)  # y_l2r : [-1, 512]
                    yi_r2l, state_r2l = cell_r2l(xf[:, self.c.frame - i - 1, :], state_r2l)  # y_r2l : [-1, 512]
                    y_l2r.append(yi_l2r)
                    y_r2l.insert(0, yi_r2l)
                y_lstm = [yi_l2r + yi_r2l for yi_l2r, yi_r2l in zip(y_l2r, y_r2l)]  # [8, -1, 512]
                y_lstm = tf.transpose(y_lstm, [1, 0, 2])  # [-1, 8, 512]
                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.c.f_size * 2)
                features, states = tf.nn.dynamic_rnn(rnn_cell, y_lstm, dtype=tf.float32)  # [-1, 8, 512]
            mean_z = tf.layers.dense(inputs=features, units=self.c.z_size, name="dense_mean")  # [-1, 8, 32]
            logvar_z = tf.layers.dense(inputs=features, units=self.c.z_size, name="dense_logvar")  # [-1, 8, 32]
        return mean_z, logvar_z

    def simple_z(self, batch_size, name):
        """
        :return: [-1, 8, 32]
        """
        z_mean = None
        z_logvar = None
        num_units = self.c.z_size
        with tf.variable_scope(name) as scope:
            cell = tf.nn.rnn_cell.LSTMCell(num_units, name="cell", state_is_tuple=False)
            state = cell.zero_state(batch_size, dtype=self.z.dtype)
            z_t = tf.zeros([batch_size, num_units], dtype=self.z.dtype)
            for i in range(self.c.frame):
                h_t, state = cell(z_t, state)  # [-1, 32]
                mean_z = tf.layers.dense(inputs=h_t, units=num_units, name="mean_z")
                logvar_z = tf.layers.dense(inputs=h_t, units=num_units, name="logvar_z")
                z_t = self.reparameterize(mean_z, logvar_z, self.c.training)
                if z_mean is None:
                    z_mean = tf.reshape(mean_z, [-1, 1, num_units])
                    z_logvar = tf.reshape(logvar_z, [-1, 1, num_units])
                else:
                    z_mean = tf.concat((z_mean, tf.reshape(mean_z, [-1, 1, num_units])), axis=1)
                    z_logvar = tf.concat((z_logvar, tf.reshape(logvar_z, [-1, 1, num_units])), axis=1)
                scope.reuse_variables()
        return z_mean, z_logvar

    def decode_frame(self, zf, name):
        """
        :param zf: [-1, 8, 256 + 32]
        :return:  [-1, 64, 64, 3]
        """
        with tf.variable_scope(name):
            final_size = self.c.input_size // 8  # 8
            base_filter = 256
            y = tf.layers.dense(inputs=zf,
                                units=self.c.x_size * 2,
                                activation=tf.nn.relu,
                                name="dense_1")  # [-1, 8, 4096]
            y = tf.layers.dense(inputs=y,
                                units=base_filter * final_size * final_size,
                                activation=tf.nn.relu,
                                name="dense_2")  # [-1, 8, 8 * 8 * 256]

            y = tf.reshape(tensor=y, shape=[-1, base_filter * final_size * final_size])  # [-1, 8 * 8 * 256]
            y = tf.reshape(tensor=y, shape=[-1, final_size, final_size, base_filter])  # [-1, 8, 8, 256]

            for i in range(3):
                base_filter //= 2
                y = tf.layers.conv2d_transpose(inputs=y,
                                               filters=base_filter,
                                               kernel_size=3,
                                               strides=2,
                                               padding="same",
                                               name="conv_1_{i}".format(i=i))  # [-1, 64, 64, 32]
                y = tf.layers.batch_normalization(inputs=y,
                                                  training=self.c.training,
                                                  name="bn_1_{i}".format(i=i))
                if i == 2:
                    y = tf.nn.sigmoid(y, name="sigmoid")
                else:
                    y = tf.nn.relu(y, name="relu_1_{i}".format(i=i))

            y = tf.layers.conv2d_transpose(inputs=y,
                                           filters=3,
                                           kernel_size=3,
                                           strides=1,
                                           padding="same",
                                           name="conv_2")  # [-1, 64, 64, 3]
            y = tf.layers.batch_normalization(inputs=y,
                                              training=self.c.training,
                                              name="bn_2".format(i=i))
        return y
