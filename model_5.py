from tensorflow.compat import v1 as tf
import framework as fm
import numpy as np
import cv2
from DS import DS

""""
重构github上的代码
"""


class Config(fm.Config):
    def __init__(self):
        super().__init__()
        self.frame = 8
        self.input_size = 64
        # self.x_size = 2048
        # self.f_size = 256
        # self.z_size = 32
        # self.hidden_size = 512
        self.num_units = 256
        self.factorised = False  # True : factorised q, False : full q
        self.lr = 0.0002
        self.epoches = 100
        self.batch_size = 27
        self.momentum = 0.98
        self.new_model = False
        self.simple_path = "images/dsa_for_github/test_"
        self.simple_num = 1
        self.ds = None

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


class SubTensors:
    def __init__(self, config: Config):
        self.config = config
        self.x = tf.placeholder(tf.float64, shape=[None, config.frame, config.input_size, config.input_size, 3],
                                name="x")  # x : [-1, 8, 64, 64, 3]
        self.inputs = [self.x]
        x = tf.reshape(self.x, [-1, config.input_size, config.input_size, 3])  # [-1 * 8, 64, 64, 3]
        x = self.encode_frame(x, "encode_frame")  # [-1, 8, 256]

        self.f = self.encode_f(x, "encode_f")  # [-1, 256]
        self.process_normal_f(self.f)
        f = tf.reshape(self.f, [-1, 1, config.num_units])
        f = tf.tile(f, [1, config.frame, 1])  # [-1, 8, 256]

        self.z = self.encode_z(x, f, "encode_z")  # [-1, 8, 256]
        self.process_normal_z(self.z)

        zf = tf.concat((self.z, f), axis=2)  # [-1, 8, 512]
        y = self.decode_frame(zf, "decode_frame")  # [-1, 64, 64, 3]

        self.predict_y = tf.reshape(y, [-1, config.frame, config.input_size, config.input_size, 3])  # [-1, 8, 64, 64, 3]

        mes = tf.reduce_mean(tf.square(self.x - self.predict_y))
        self.losses = [mes]

    def encode_frame(self, x, name):
        """
        :param x: [-1, 64, 64, 3]
        :return:  [-1, 8, 2048]
        """
        with tf.variable_scope(name):
            base_filter = 32
            x = tf.layers.conv2d(x, base_filter, 3, 1, padding="same", activation=tf.nn.leaky_relu,
                                 name="conv_1")  # [-1, 64, 64, 32]
            for i in range(3):
                base_filter *= 2
                x = tf.layers.conv2d(x, base_filter, 3, 2, padding="same", activation=tf.nn.leaky_relu,
                                     name="conv_2_{i}".format(i=i))  # [-1, 8, 8, 256]
            x = tf.layers.flatten(x)  # [-1, 8 * 8 * 256]
            x = tf.layers.dense(x, self.config.num_units, activation=tf.nn.leaky_relu, name="dense_3")  # [-1, 256]
        x = tf.reshape(x, [-1, self.config.frame, self.config.num_units])
        return x

    def encode_f(self, x, name):
        """
        :param x: [-1, 8, 256]
        :param name:
        :return: [-1, 256]
        """
        with tf.variable_scope(name):
            cell_l2r = tf.nn.rnn_cell.LSTMCell(self.config.num_units, name="cell_l2r", state_is_tuple=False)
            cell_r2l = tf.nn.rnn_cell.LSTMCell(self.config.num_units, name="cell_r2l", state_is_tuple=False)
            batch_size = tf.shape(x)[0]
            state_l2r = cell_l2r.zero_state(batch_size, dtype=x.dtype)
            state_r2l = cell_r2l.zero_state(batch_size, dtype=x.dtype)
            for i in range(self.config.frame):
                y_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # y_l2r : [-1, 256]
                y_r2l, state_r2l = cell_r2l(x[:, self.config.frame - i - 1, :], state_r2l)  # y_r2l : [-1, 256]

            y = tf.concat((y_l2r, y_r2l), axis=1)  # [-1, 512]
            f = tf.layers.dense(y, self.config.num_units, name="dense_f")  # [-1, 256]
        return f

    def process_normal_f(self, f):
        """
        :param f: [-1, 256]
        :return:
        """
        mean = tf.reduce_mean(f, axis=0)  # 当前平均值, [256]
        vec_size = f.shape[1]  # [256]
        self.final_mean_f = tf.get_variable(name="f_mean", shape=[vec_size], dtype=tf.float64, trainable=False)  # 目标平均值
        momentum = self.config.momentum
        mean_assign = tf.assign(self.final_mean_f, self.final_mean_f * momentum + mean * (1 - momentum))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_assign)

        msd = tf.reduce_mean(tf.square(f), axis=0)
        self.final_msd_f = tf.get_variable(name="f_msd", shape=[vec_size], dtype=tf.float64, trainable=False)
        msd_assign = tf.assign(self.final_msd_f, self.final_msd_f * momentum + msd * (1 - momentum))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, msd_assign)

    def encode_z(self, x, f, name):
        """
        :param x: [-1, 8, 2048]
        :param f: [-1, 8, 256]
        :param name:
        :return: [-1, 8, 32]
        """
        with tf.variable_scope(name):
            if self.config.factorised is True:
                features = tf.layers.dense(x, self.config.num_units, activation=tf.nn.leaky_relu, name="dense_1")  # [-1, 8, 256]
            else:
                x = tf.concat((x, f), axis=2)  # [-1, 8, 512]
                num_units = self.config.num_units * 2
                cell_l2r = tf.nn.rnn_cell.LSTMCell(num_units, name="cell_l2r", state_is_tuple=False)
                cell_r2l = tf.nn.rnn_cell.LSTMCell(num_units, name="cell_r2l", state_is_tuple=False)
                batch_size = tf.shape(x)[0]
                state_l2r = cell_l2r.zero_state(batch_size, dtype=x.dtype)
                state_r2l = cell_r2l.zero_state(batch_size, dtype=x.dtype)
                y_l2r = []
                y_r2l = []  # [8, -1, 512]
                for i in range(self.config.frame):
                    yi_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # y_l2r : [-1, 512]
                    yi_r2l, state_r2l = cell_r2l(x[:, self.config.frame - i - 1, :], state_r2l)  # y_r2l : [-1, 512]
                    y_l2r.append(yi_l2r)
                    y_r2l.insert(0, yi_r2l)
                y_lstm = [yi_l2r + yi_r2l for yi_l2r, yi_r2l in zip(y_l2r, y_r2l)]  # [8, -1, 512]
                y_lstm = tf.transpose(y_lstm, [1, 0, 2])  # [-1, 8, 512]

                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.num_units * 2)
                features, states = tf.nn.dynamic_rnn(rnn_cell, y_lstm, dtype=tf.float64)  # [-1, 8, 512]

            z = tf.layers.dense(features, self.config.num_units, name="dense_z")  # [-1, 8, 256]
        return z

    def process_normal_z(self, z):
        """
        :param z: [-1, 8, 256]
        :return:
        """
        mean = tf.reduce_mean(z, axis=0)  # 当前平均值, [8, 256]
        self.final_mean_z = tf.get_variable(name="z_mean", shape=[z.shape[1], z.shape[2]], dtype=tf.float64, trainable=False)  # 目标平均值
        momentum = self.config.momentum
        mean_assign = tf.assign(self.final_mean_z, self.final_mean_z * momentum + mean * (1 - momentum))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_assign)

        msd = tf.reduce_mean(tf.square(z), axis=0)
        self.final_msd_z = tf.get_variable(name="z_msd", shape=[z.shape[1], z.shape[2]], dtype=tf.float64, trainable=False)
        msd_assign = tf.assign(self.final_msd_z, self.final_msd_z * momentum + msd * (1 - momentum))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, msd_assign)

    def decode_frame(self, zf, name):
        """
        :param zf: [-1, 8, 512]
        :return:  [-1, 64, 64, 3]
        """
        with tf.variable_scope(name):
            final_size = self.config.input_size // 8  # 64 / 8 = 8
            base_filter = 256
            y = tf.layers.dense(zf, base_filter * final_size * final_size, activation=tf.nn.leaky_relu, name="dense_1")  # [-1, 8, 8 * 8 * 256]
            y = tf.reshape(y, [-1, base_filter * final_size * final_size])  # [-1, 8 * 8 * 256]
            y = tf.reshape(y, [-1, final_size, final_size, base_filter])  # [-1, 8, 8, 256]

            for i in range(3):
                base_filter //= 2
                y = tf.layers.conv2d_transpose(y, base_filter, 3, 2, padding="same", activation=tf.nn.leaky_relu,
                                     name="conv_1_{i}".format(i=i))  # [-1, 64, 64, 32]

            y = tf.layers.conv2d_transpose(y, 3, 3, 1, padding="same", activation=tf.nn.relu,
                                 name="conv_2")  # [-1, 64, 64, 3]
        return y


class App(fm.App):

    def test(self, ds_test):
        x1 = ds_test[0]
        x2 = ds_test[10]

        self.reconstruction(x1)
        # self.reconstruction_with_random_f(x1)
        # self.reconstruction_with_random_z(x1)
        self.features_change(x1, x2)
        # self.random(x1)

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

        # f_mean, f_logvar, z = self.session.run([ts.mean_f, ts.logvar_f, ts.z], {ts.x: [x]})
        # f = self.reparameterize(f_mean, f_logvar)
        mean_f, msd_f, z = self.session.run([ts.final_mean_f, ts.final_msd_f, ts.z], {ts.x: [x]})
        st_f = msd_f - mean_f ** 2
        std_f = np.sqrt(np.maximum(st_f, 1e-5))
        f = self.reparameterize(mean_f, std_f)

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
        # f, z_mean, z_logvar = self.session.run([ts.f, ts.mean_z, ts.logvar_z], {ts.x: [x]})
        # z = self.reparameterize(z_mean, z_logvar)
        mean_z, msd_z, f = self.session.run([ts.final_mean_z, ts.final_msd_z, ts.f], {ts.x: [x]})
        st_z = msd_z - mean_z ** 2
        std_z = np.sqrt(np.maximum(st_z, 1e-5))
        z = self.reparameterize(mean_z, std_z)

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

    def random(self, x):
        """
        随机生成
        """""
        ts = self.ts.sub_ts[-1]

        # f = np.random.normal(size=[1, self.config.f_size])  # [-1, 256]
        # mean_z, logvar_z = self.session.run([ts.generator_z_mean, ts.generator_z_logvar], {ts.x: [x]})  # [-1, 8, 32]
        # z = self.reparameterize(mean_z, logvar_z)  # [-1, 8, 32]
        #
        # images = self.session.run(ts.predict_y, {ts.f: f, ts.z: z})  # [-1, 8, 64, 64, 3]
        # images = np.transpose(images, [0, 2, 1, 3, 4])  # [-1, 64, 8, 64, 3]
        # images = np.reshape(images, [-1, 64 * 8, 3])
        #
        # cv2.imwrite(self.config.simple_path + "random.jpg", images * 255)

        mean_f, msd_f = self.session.run(ts.final_mean_f, ts.final_msd_f)
        st_f = msd_f - mean_f ** 2
        std_f = np.sqrt(np.maximum(st_f, 1e-5))

        mean_z, msd_z = self.session.run(ts.final_mean_z, ts.final_msd_z)
        st_z = msd_z - mean_z ** 2
        std_z = np.sqrt(np.maximum(st_z, 1e-5))

        # f = self.reparameterize(mean_f, std_f)
        # z = self.reparameterize(mean_z, std_z)
        f = np.random.normal(mean_f, std_f, [self.config.simple_num, len(std_f)])
        z = np.random.normal(mean_z, std_z, [self.config.simple_num, 8, 32])

        images = self.session.run(ts.predict_y, {ts.f: f, ts.z: z})  # [-1, 8, 64, 64, 3]
        images = np.transpose(images, [0, 2, 1, 3, 4])
        images = np.reshape(images, [-1, 64 * 8, 3])
        cv2.imwrite(self.config.simple_path, images * 255)

    def reparameterize(self, mean, std):
        eps = np.random.normal(size=mean.shape)
        return mean + std * eps


if __name__ == '__main__':
    tf.disable_eager_execution()
    config = Config()
    config.call("test")