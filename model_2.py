from tensorflow.compat import v1 as tf
import framework as fm
import numpy as np
import cv2
from DS import DS

""""
重构github上的代码
"""


class MyTensors(fm.Tensors):
    def compute_grads(self, opt):
        vars = tf.trainable_variables()
        vars_disc = [var for var in vars if "simple_z" not in var.name]
        vars_gene = [var for var in vars if "simple_z" in var.name]
        vars = [vars_disc, vars_gene]

        grads = []
        for gpu_id, ts in enumerate(self.sub_ts):
            with tf.device('/gpu:%d' % gpu_id):
                grads.append([opt.compute_gradients(loss, vs) for vs, loss in zip(vars, ts.losses)])
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]


class Config(fm.Config):
    def __init__(self):
        super().__init__()
        self.frame = 8
        self.input_size = 64
        self.x_size = 2048
        self.f_size = 256
        self.z_size = 32
        self.hidden_size = 512
        self.factorised = False  # True : factorised q, False : full q
        self.lr = 0.0001
        self.epoches = 500
        self.batch_size = 10
        self.new_model = False
        self.simple_path = "images/dsa_for_github/test_"
        self.simple_num = 1
        self.ds = None
        self.training = True

    def get_name(self):
        return "dsa_for_github"

    def get_sub_tensors(self, gpu_index):
        return SubTensors(self)

    def get_app(self):
        return App(self)

    def get_ds_train(self):
        self.read_ds()
        return self.ds

    def get_ds_test(self):
        self.read_ds()
        return self.ds.test_ds()

    def read_ds(self):
        if self.ds is None:
            self.ds = DS()

    def get_tensors(self):
        return MyTensors(self)

    def test(self):
        self.batch_size = 1
        super(Config, self).test()


class SubTensors:
    def __init__(self, config: Config):
        self.config = config
        self.x = tf.placeholder(tf.float32, shape=[None, config.frame, config.input_size, config.input_size, 3],
                                name="x")  # x : [-1, 8, 64, 64, 3]
        self.inputs = [self.x]
        x = tf.reshape(self.x, [-1, config.input_size, config.input_size, 3])  # [-1 * 8, 64, 64, 3]
        x = self.encode_frame(x, "encode_frame")  # [-1, 8, 2048]

        self.mean_f, self.logvar_f, self.f = self.encode_f(x, "encode_f")  # [-1, 256]
        f = tf.reshape(self.f, [-1, 1, config.f_size])
        f = tf.tile(f, [1, config.frame, 1])  # [-1, 8, 256]

        self.mean_z, self.logvar_z = self.encode_z(x, f, "encode_z")  # [-1, 8, 32]
        self.z = self.reparameterize(self.mean_z, self.logvar_z, self.config.training)

        zf = tf.concat((self.z, f), axis=2)  # [-1, 8, 256 + 32]
        y = self.decode_frame(zf, "decode_frame")  # [-1, 64, 64, 3]

        self.predict_y = tf.reshape(y, [-1, config.frame, config.input_size, config.input_size, 3])  # [-1, 8, 64, 64, 3]

        self.generator_z_mean, self.generator_z_logvar = self.simple_z(config.batch_size, name="simple_z")

        # [-1, 8, 64, 64, 3]
        mes = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - self.predict_y), axis=(1, 2, 3, 4)))
        # f : [-1, 256]
        kld_f = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.logvar_f - tf.pow(self.mean_f, 2) - tf.exp(self.logvar_f), 1))
        # z : [-1, 8, 32]
        z_post_var = tf.exp(self.logvar_z)
        z_prior_var = tf.exp(self.generator_z_logvar)
        # kld_z = tf.reduce_mean(0.5 * tf.reduce_sum(self.generator_z_logvar - self.logvar_z +
        #                              (z_post_var + tf.pow(self.mean_z - self.generator_z_mean, 2)) / z_prior_var - 1, axis=(1, 2)))
        kld_z = tf.reduce_mean(0.5 * tf.reduce_sum(self.logvar_z - self.generator_z_logvar +
                                                   (z_prior_var + tf.pow(self.mean_z - self.generator_z_mean, 2)) / z_post_var - 1, axis=(1, 2)))
        loss = mes + kld_f
        self.losses = [loss, 0.01 * kld_z]

    def encode_frame(self, x, name):
        """
        :param x: [-1, 64, 64, 3]
        :return:  [-1, 8, 2048]
        """
        with tf.variable_scope(name):
            # base_filter = self.config.x_size // 8  # 256
            base_filter = 32

            x = tf.layers.conv2d(x, base_filter, 3, 1, padding="same", name="conv_1")  # [-1, 64, 64, 32]
            x = tf.layers.batch_normalization(x, training=self.config.training, name="bn_0")
            x = tf.nn.relu(x, name="relu_0")

            for i in range(3):
                base_filter *= 2
                x = tf.layers.conv2d(x, base_filter, 3, 2, padding="same", name="conv_2_{i}".format(i=i))  # [-1, 32, 32, 256]
                x = tf.layers.batch_normalization(x, training=self.config.training, name="bn_1_{i}".format(i=i))
                x = tf.nn.relu(x, name="relu_1_{i}".format(i=i))

            x = tf.layers.flatten(x)  # [-1, 8 * 8 * 256]

            x = tf.layers.dense(x, self.config.x_size * 2, name="dense_1")  # [-1, 4096]
            x = tf.layers.batch_normalization(x, training=self.config.training, name="bn_2")
            x = tf.nn.relu(x, name="relu_2")

            x = tf.layers.dense(x, self.config.x_size, name="dense_2")  # [-1, 2048]
            x = tf.layers.batch_normalization(x, training=self.config.training, name="bn_3")
            x = tf.nn.relu(x, name="relu_3")

        x = tf.reshape(x, [-1, self.config.frame, self.config.x_size])  # [-1, 8, 2048]
        return x

    def encode_f(self, x, name):
        """
        :param x: [-1, 8, 2048]
        :param name:
        :return: [-1, 256]
        """
        with tf.variable_scope(name):
            x = tf.layers.dense(x, self.config.hidden_size, name="dense1")  # [-1, 8, 512]

            cell_l2r = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, name="cell_l2r", state_is_tuple=False)
            cell_r2l = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, name="cell_r2l", state_is_tuple=False)
            batch_size = tf.shape(x)[0]
            state_l2r = cell_l2r.zero_state(batch_size, dtype=x.dtype)
            state_r2l = cell_r2l.zero_state(batch_size, dtype=x.dtype)
            for i in range(self.config.frame):
                y_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # y_l2r : [-1, 512]
                y_r2l, state_r2l = cell_r2l(x[:, self.config.frame - i - 1, :], state_r2l)  # y_r2l : [-1, 512]

            y = tf.concat((y_l2r, y_r2l), axis=1)  # [-1, 1024]
            mean_f = tf.layers.dense(y, self.config.f_size, name="dense_mean")  # [-1, 256]
            logvar_f = tf.layers.dense(y, self.config.f_size, name="dense_logvar")  # [-1, 256]
        return mean_f, logvar_f, self.reparameterize(mean_f, logvar_f, self.config.training)

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
            if self.config.factorised is True:
                features = tf.layers.dense(x, self.config.hidden_size, activation=tf.nn.relu, name="dense_1")  # [-1, 8, 512]
            else:
                x = tf.concat((x, f), axis=2)  # [-1, 8, 2048 + 256]
                x = tf.layers.dense(x, self.config.hidden_size, name="dense_2")  # [-1, 8, 512]

                cell_l2r = tf.nn.rnn_cell.LSTMCell(self.config.f_size, name="cell_l2r", state_is_tuple=False)
                cell_r2l = tf.nn.rnn_cell.LSTMCell(self.config.f_size, name="cell_r2l", state_is_tuple=False)
                cell_rnn = tf.nn.rnn_cell.LSTMCell(self.config.f_size, name="cell_rnn", state_is_tuple=False)
                batch_size = tf.shape(x)[0]
                state_l2r = cell_l2r.zero_state(batch_size, dtype=x.dtype)
                state_r2l = cell_r2l.zero_state(batch_size, dtype=x.dtype)
                state_rnn = cell_rnn.zero_state(batch_size, dtype=x.dtype)
                y_l2r = []
                y_r2l = []  # [8, -1, 512]
                for i in range(self.config.frame):
                    yi_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # y_l2r : [-1, 512]
                    yi_r2l, state_r2l = cell_r2l(x[:, self.config.frame - i - 1, :], state_r2l)  # y_r2l : [-1, 512]
                    y_l2r.append(yi_l2r)
                    y_r2l.insert(0, yi_r2l)
                y_lstm = [yi_l2r + yi_r2l for yi_l2r, yi_r2l in zip(y_l2r, y_r2l)]  # [8, -1, 512]
                y_lstm = tf.transpose(y_lstm, [1, 0, 2])  # [-1, 8, 512]
                y_rnn = []
                for i in range(self.config.frame):
                    yi_rnn, state_rnn = cell_rnn(y_lstm[:, i, :], state_rnn)  # [-1, 512]
                    y_rnn.append(yi_rnn)
                features = tf.transpose(y_rnn, [1, 0, 2])
            mean_z = tf.layers.dense(features, self.config.z_size, name="dense_mean")  # [-1, 8, 32]
            logvar_z = tf.layers.dense(features, self.config.z_size, name="dense_logvar")  # [-1, 8, 32]
        return mean_z, logvar_z

    def simple_z(self, batch_size, name):
        """
        :return: [-1, 8, 32]
        """
        # z_out = None
        z_mean = None
        z_logvar = None
        num_units = self.config.z_size
        with tf.variable_scope(name) as scope:
            cell = tf.nn.rnn_cell.LSTMCell(num_units, name="cell", state_is_tuple=False)
            state = cell.zero_state(batch_size, dtype=self.z.dtype)
            x_t = tf.zeros([batch_size, num_units], dtype=self.z.dtype)
            for i in range(self.config.frame):
                h_t, state = cell(x_t, state)  # [-1, 32]
                mean_z = tf.layers.dense(h_t, num_units, name="mean_z")
                logvar_z = tf.layers.dense(h_t, num_units, name="logvar_z")
                # z_t = self.reparameterize(mean_z, logvar_z)
                if z_mean is None:
                    z_mean = tf.reshape(mean_z, [-1, 1, num_units])
                    z_logvar = tf.reshape(logvar_z, [-1, 1, num_units])
                    # z_out = tf.reshape(z_t, [-1, 1, num_units])
                else:
                    z_mean = tf.concat((z_mean, tf.reshape(mean_z, [-1, 1, num_units])), axis=1)
                    z_logvar = tf.concat((z_logvar, tf.reshape(logvar_z, [-1, 1, num_units])), axis=1)
                    # z_out = tf.concat((z_out, tf.reshape(z_t, [-1, 1, num_units])), axis=1)
                scope.reuse_variables()

        return z_mean, z_logvar

    def decode_frame(self, zf, name):
        """
        :param zf: [-1, 8, 256 + 32]
        :return:  [-1, 64, 64, 3]
        """
        with tf.variable_scope(name):
            final_size = self.config.input_size // 8  # 8
            # base_filter = self.config.x_size // 8  # 256
            base_filter = 256
            y = tf.layers.dense(zf, self.config.x_size * 2, activation=tf.nn.relu, name="dense_1")  # [-1, 8, 4096]
            y = tf.layers.dense(y, base_filter * final_size * final_size, activation=tf.nn.relu, name="dense_2")  # [-1, 8, 8 * 8 * 256]

            y = tf.reshape(y, [-1, base_filter * final_size * final_size])  # [-1, 8 * 8 * 256]
            y = tf.reshape(y, [-1, final_size, final_size, base_filter])  # [-1, 8, 8, 256]

            for i in range(3):
                base_filter //= 2
                y = tf.layers.conv2d_transpose(y, base_filter, 3, 2, padding="same", name="conv_1_{i}".format(i=i))  # [-1, 64, 64, 32]
                y = tf.layers.batch_normalization(y, training=self.config.training, name="bn_1_{i}".format(i=i))
                y = tf.nn.relu(y, name="relu_1_{i}".format(i=i))

            y = tf.layers.conv2d_transpose(y, 3, 3, 1, padding="same", name="conv_2")  # [-1, 64, 64, 3]
            y = tf.layers.batch_normalization(y, training=self.config.training, name="bn_2".format(i=i))
            y = tf.nn.relu(y, name="sigmoid_2")
        return y


class App(fm.App):

    def test(self, ds_test):
        x1 = ds_test[0]
        x2 = ds_test[10]

        self.reconstruction(x1)
        self.reconstruction_with_random_f(x1)
        self.reconstruction_with_random_z(x1)
        self.features_change(x1, x2)
        self.random()

    def reconstruction(self, x):
        """
        重组：生成图片结构 [x, x_re]
        """""
        ts = self.ts.sub_ts[-1]
        images_re = self.session.run(ts.predict_y, {ts.x: [x]})  # [-1, 8, 64, 64, 3]

        image_x = np.array(x)  # [8, 64, 64,3]
        image_x = np.reshape(image_x, [1, 8, 64, 64, 3])

        images = (image_x, images_re)  # [2, -1, 8, 64, 64, 3]
        images = np.reshape(images, [-1, 8, 64, 64, 3])
        images = np.transpose(images, [0, 2, 1, 3, 4])  # [-1, 64, 8, 64, 3]
        images = np.reshape(images, [-1, 64 * 8, 3])

        cv2.imwrite(self.config.simple_path + "re.jpg", images * 255)

    def reconstruction_with_random_f(self, x):
        """
        重组：随机f + z，生成图片结构 [x, x_re]
        """""
        ts = self.ts.sub_ts[-1]

        z = self.session.run(ts.z, {ts.x: [x]})
        # mean = np.random.uniform(0, 1, size=[len(x), self.config.f_size])
        f = np.random.normal(0, 1, size=[1, self.config.f_size])  # [-1, 256]
        images_pre = self.session.run(ts.predict_y, {ts.f: f, ts.z: z})  # [-1, 8, 64, 64, 3]

        image_x = np.array(x)  # [8, 64, 64,3]
        image_x = np.reshape(image_x, [1, 8, 64, 64, 3])

        images = (image_x, images_pre)
        images = np.reshape(images, [-1, 8, 64, 64, 3])
        images = np.transpose(images, [0, 2, 1, 3, 4])  # [-1, 64, 8, 64, 3]
        images = np.reshape(images, [-1, 64 * 8, 3])
        cv2.imwrite(self.config.simple_path + "re_fre.jpg", images * 255)

    def reconstruction_with_random_z(self, x):
        """
        重组：f + 随机z，生成图片结构 [x, x_re]
        """""
        ts = self.ts.sub_ts[-1]
        f, z_mean, z_logvar = self.session.run([ts.f, ts.generator_z_mean, ts.generator_z_logvar], {ts.x: [x]})
        z = self.reparameterize(z_mean, z_logvar)
        images_pre = self.session.run(ts.predict_y, {ts.f: f, ts.z: z})  # [-1, 8, 64, 64, 3]
        image_x = np.array(x)  # [8, 64, 64,3]
        image_x = np.reshape(image_x, [1, 8, 64, 64, 3])

        images = (image_x, images_pre)
        images = np.reshape(images, [-1, 8, 64, 64, 3])
        images = np.transpose(images, [0, 2, 1, 3, 4])  # [-1, 64, 8, 64, 3]
        images = np.reshape(images, [-1, 64 * 8, 3])
        cv2.imwrite(self.config.simple_path + "re_zre.jpg", images * 255)

    def features_change(self, x1, x2):
        """
        特征交换：生成的图片结构 [x1, x2, f1_z2, f2_z1]
        """""
        ts = self.ts.sub_ts[-1]

        f1, z1 = self.session.run([ts.f, ts.z], {ts.x: [x1]})
        f2, z2 = self.session.run([ts.f, ts.z], {ts.x: [x2]})

        image_x1 = np.array(x1)  # [8, 64, 64,3]
        image_x1 = np.reshape(image_x1, [1, 8, 64, 64, 3])
        image_x2 = np.array(x2)
        image_x2 = np.reshape(image_x2, [1, 8, 64, 64, 3])

        images_f1_z2 = self.session.run(ts.predict_y, {ts.f: f1, ts.z: z2})  # [-1, 8, 64, 64, 3]
        images_f2_z1 = self.session.run(ts.predict_y, {ts.f: f2, ts.z: z1})  # [-1, 8, 64, 64, 3]

        images = (image_x1, image_x2, images_f1_z2, images_f2_z1)  # [4, -1, 8, 64, 64, 3]
        images = np.reshape(images, [-1, 8, 64, 64, 3])
        images = np.transpose(images, [0, 2, 1, 3, 4])  # [-1, 64, 8, 64, 3]
        images = np.reshape(images, [-1, 64 * 8, 3])

        cv2.imwrite(self.config.simple_path + "feature_change.jpg", images * 255)

    def random(self):
        """
        随机生成
        """""
        ts = self.ts.sub_ts[-1]

        f = np.random.normal(0, 1, size=[1, self.config.f_size])  # [-1, 256]
        mean_z, logvar_z = self.session.run([ts.generator_z_mean, ts.generator_z_logvar])  # [-1, 8, 32]
        z = self.reparameterize(mean_z, logvar_z)  # [-1, 8, 32]

        images = self.session.run(ts.predict_y, {ts.f: f, ts.z: z})  # [-1, 8, 64, 64, 3]
        images = np.transpose(images, [0, 2, 1, 3, 4])  # [-1, 64, 8, 64, 3]
        images = np.reshape(images, [-1, 64 * 8, 3])

        cv2.imwrite(self.config.simple_path + "random.jpg", images * 255)

    def reparameterize(self, mean, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.normal(size=logvar.shape)
        return mean + std * eps


if __name__ == '__main__':
    tf.disable_eager_execution()
    config = Config()
    config.call("train")
