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
        self.x_size = 2048
        self.f_size = 256
        self.z_size = 32
        self.hidden_size = 512
        self.factorised = True  # True : factorised q, False : full q
        self.lr = 0.0002
        self.epoches = 100
        self.batch_size = 10
        self.momentum = 0.98
        self.new_model = False
        self.simple_path = "images/dsa_for_github/test_"
        self.simple_num = 1
        self.ds = None
        self.training = False

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
        x = self.encode_frame(x, "encode_frame")  # [-1, 8, 2048]

        self.mean_f, self.logvar_f, self.f = self.encode_f(x, "encode_f")  # [-1, 256]

        y = self.decode_frame(self.f, "decode_frame")  # [-1, 64, 64, 3]

        self.predict_y = tf.reshape(y, [-1, config.frame, config.input_size, config.input_size, 3])  # [-1, 8, 64, 64, 3]

        mes = tf.reduce_mean(tf.square(self.x - self.predict_y))
        kld_f = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.logvar_f - tf.pow(self.mean_f, 2) - tf.exp(self.logvar_f, 1)))
        self.losses = [mes + kld_f]

    def encode_frame(self, x, name):
        """
        :param x: [-1, 64, 64, 3]
        :return:  [-1, 8, 2048]
        """
        with tf.variable_scope(name):
            base_filter = self.config.x_size // 8  # 256
            x = tf.layers.conv2d(x, base_filter, 3, 1, padding="same", activation=tf.nn.relu,
                                 name="conv_1")  # [-1, 64, 64, 256]
            for i in range(3):
                x = tf.layers.conv2d(x, base_filter, 3, 2, padding="same", activation=tf.nn.relu,
                                     name="conv_2_{i}".format(i=i))  # [-1, 32, 32, 256]
            x = tf.layers.flatten(x)  # [-1, 8 * 8 * 256]
            x = tf.layers.dense(x, self.config.x_size, activation=tf.nn.relu, name="dense_2")  # [-1, 2048]
        x = tf.reshape(x, [-1, self.config.frame, self.config.x_size])
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
            return mean

    def decode_frame(self, f, name):
        """
        :param f: [-1, 256]
        :return:  [-1, 64, 64, 3]
        """
        with tf.variable_scope(name):
            final_size = self.config.input_size // 8  # 8
            base_filter = self.config.x_size // 8  # 256
            y = tf.layers.dense(f, base_filter * final_size * final_size * final_size, activation=tf.nn.relu, name="dense_2")  # [-1, 8 * 8 * 8 * 256]
            y = tf.reshape(y, [-1, final_size, final_size, final_size, base_filter])  # [-1, 8, 8, 8, 256]
            y = tf.reshape(y, [-1, final_size, final_size, base_filter])  # [-1, 8, 8, 256]

            y = tf.layers.conv2d_transpose(y, base_filter, 3, 2, padding="same", activation=tf.nn.relu,
                                 name="conv_1")  # [-1, 16, 16, 256]
            for i in range(2):
                y = tf.layers.conv2d_transpose(y, base_filter, 3, 2, padding="same", activation=tf.nn.relu,
                                     name="conv_2_{i}".format(i=i))  # [-1, 64, 64, 256]
            y = tf.layers.conv2d_transpose(y, 3, 3, 1, padding="same", activation=tf.nn.relu,
                                 name="conv_3")  # [-1, 64, 64, 3]
        return y


class App(fm.App):

    def test(self, ds_test):
        x1 = ds_test[0]
        x2 = ds_test[10]

        self.reconstruction(x1)
        # self.reconstruction_with_random_f(x1)
        # self.reconstruction_with_random_z(x1)
        # self.features_change(x1, x2)
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

        f_mean, f_logvar, z = self.session.run([ts.mean_f, ts.logvar_f, ts.z], {ts.x: [x]})
        f = self.reparameterize(f_mean, f_logvar)
        # mean_f, msd_f, z = self.session.run([ts.final_mean_f, ts.final_msd_f, ts.z], {ts.x: [x]})
        # st_f = msd_f - mean_f ** 2
        # std_f = np.sqrt(np.maximum(st_f, 1e-5))
        # f = self.reparameterize(mean_f, std_f)

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