from tensorflow.compat import v1 as tf
import Disentangled_Sequential_Autoencoder.framework as fm
import numpy as np
import cv2
from Disentangled_Sequential_Autoencoder.DS import DS


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
        self.lr = 0.00001
        self.epoches = 2000
        self.batch_size = 10
        self.momentum = 0.99
        self.new_model = False
        self.simple_path = "images/dsa_1_2000/test_f1_z2.jpg"
        self.simple_num = 1
        self.ds = None

    def get_name(self):
        return "dsa"

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
        self.f = self.encode_f(x, "encode_f")  # [-1, 256]
        # 计算f的均值和方差
        self.process_normal_f(self.f)
        f = tf.reshape(self.f, [-1, 1, config.f_size])
        f = tf.tile(f, [1, config.frame, 1])  # [-1, 8, 256]
        self.z = self.encode_z(x, f, "encode_z")  # [-1, 8, 32]
        # 计算z的均值和方差
        self.process_normal_z(self.z)

        zf = tf.concat((self.z, f), axis=2)  # [-1, 8, 256 + 32]
        y = self.decode_frame(zf, "decode_frame")  # [-1, 64, 64, 3]
        self.predict_y = tf.reshape(y, [-1, config.frame, config.input_size, config.input_size, 3])  # [-1, 8, 64, 64, 3]

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.predict_y)))
        self.losses = [self.loss]
        # op = tf.data.AdamOptimizer(config.lr)
        # # 控制依赖、summary
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     self.train_op = op.minimize(self.loss)

        # tf.summary.scalar('loss', self.loss)
        # self.summary = tf.summary.merge_all()

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

    def process_normal_z(self, z):
        """
        :param z: [-1, 8, 32]
        :return:
        """
        mean = tf.reduce_mean(z, axis=0)  # 当前平均值, [8, 32]
        self.final_mean_z = tf.get_variable(name="z_mean", shape=[z.shape[1], z.shape[2]], dtype=tf.float64, trainable=False)  # 目标平均值
        momentum = self.config.momentum
        mean_assign = tf.assign(self.final_mean_z, self.final_mean_z * momentum + mean * (1 - momentum))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_assign)

        msd = tf.reduce_mean(tf.square(z), axis=0)
        self.final_msd_z = tf.get_variable(name="z_msd", shape=[z.shape[1], z.shape[2]], dtype=tf.float64, trainable=False)
        msd_assign = tf.assign(self.final_msd_z, self.final_msd_z * momentum + msd * (1 - momentum))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, msd_assign)

    def encode_frame(self, x, name):
        """
        :param x: [-1, 64, 64, 3]
        :return:  [-1, 8, 2048]
        """
        with tf.variable_scope(name):
            base_filter = self.config.x_size // 8  # 256
            x = tf.layers.conv2d(x, base_filter, 5, 1, padding="same", activation=tf.nn.relu,
                                 name="conv_1")  # [-1, 64, 64, 256]
            x = tf.layers.conv2d(x, base_filter, 5, 2, padding="same", activation=tf.nn.relu,
                                 name="conv_2")  # [-1, 32, 32, 256]
            x = tf.layers.conv2d(x, base_filter, 5, 2, padding="same", activation=tf.nn.relu,
                                 name="conv_3")  # [-1, 16, 16, 256]
            x = tf.layers.conv2d(x, base_filter, 5, 2, padding="same", activation=tf.nn.relu,
                                 name="conv_4")  # [-1, 8, 8, 256]

            x = tf.layers.flatten(x)  # [-1, 8 * 8 * 256]
            x = tf.layers.dense(x, self.config.x_size * 2, activation=tf.nn.relu, name="dense_1")  # [-1, 4096]
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
                y_r2l, state_r2l = cell_r2l(x[:, i, :], state_r2l)  # y_r2l : [-1, 512]

            y = tf.concat((y_l2r, y_r2l), axis=1)  # [-1, 1024]

            f = tf.layers.dense(y, self.config.f_size, activation=tf.nn.relu, name="dense_1")  # [-1, 256]
        return f

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
                batch_size = tf.shape(x)[0]
                state_l2r = cell_l2r.zero_state(batch_size, dtype=x.dtype)
                state_r2l = cell_r2l.zero_state(batch_size, dtype=x.dtype)

                y_l2r = []
                y_r2l = []  # [8, -1, 512]
                for i in range(self.config.frame):
                    yi_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # y_l2r : [-1, 512]
                    yi_r2l, state_r2l = cell_r2l(x[:, i, :], state_r2l)  # y_r2l : [-1, 512]
                    y_l2r.append(yi_l2r)
                    y_r2l.insert(0, yi_r2l)
                features = [yi_l2r + yi_r2l for yi_l2r, yi_r2l in zip(y_l2r, y_r2l)]  # [8, -1, 512]

                features = tf.transpose(features, [1, 0, 2])
            z = tf.layers.dense(features, self.config.z_size, name="dense_3")  # [-1, 8, 32]
        return z

    def decode_frame(self, zf, name):
        """
        :param zf: [-1, 8, 256 + 32]
        :return:  [-1, 64, 64, 3]
        """
        with tf.variable_scope(name):
            final_size = self.config.input_size // 8  # 8
            base_filter = self.config.x_size // 8  # 256
            y = tf.layers.dense(zf, self.config.x_size * 2, activation=tf.nn.relu, name="dense_1")  # [-1, 8, 4096]
            y = tf.layers.dense(y, base_filter * final_size * final_size, activation=tf.nn.relu, name="dense_2")  # [-1, 8, 8 * 8 * 256]
            y = tf.reshape(y, [-1, base_filter * final_size * final_size])  # [-1, 8 * 8 * 256]
            y = tf.reshape(y, [-1, final_size, final_size, base_filter])  # [-1, 8, 8, 256]

            y = tf.layers.conv2d_transpose(y, base_filter, 5, 2, padding="same", activation=tf.nn.relu,
                                 name="conv_1")  # [-1, 16, 16, 256]
            y = tf.layers.conv2d_transpose(y, base_filter, 5, 2, padding="same", activation=tf.nn.relu,
                                 name="conv_2")  # [-1, 32, 32, 256]
            y = tf.layers.conv2d_transpose(y, base_filter, 5, 2, padding="same", activation=tf.nn.relu,
                                 name="conv_3")  # [-1, 64, 64, 256]
            y = tf.layers.conv2d_transpose(y, 3, 5, 1, padding="same", activation=tf.nn.relu,
                                 name="conv_4")  # [-1, 64, 64, 3]
        return y


class App(fm.App):

    def test_random(self):
        ts = self.ts.sub_ts[-1]
        mean_f = self.session.run(ts.final_mean_f)
        msd_f = self.session.run(ts.final_msd_f)
        st_f = msd_f - mean_f ** 2
        std_f = np.sqrt(np.maximum(st_f, 1e-5))

        mean_z = self.session.run(ts.final_mean_z)
        msd_z = self.session.run(ts.final_msd_z)
        st_z = msd_z - mean_z ** 2
        std_z = np.sqrt(np.maximum(st_z, 1e-5))

        f = np.random.normal(mean_f, std_f, [self.config.simple_num, len(std_f)])
        z = np.random.normal(mean_z, std_z, [self.config.simple_num, 8, 32])

        images = self.session.run(ts.predict_y, {ts.f: f, ts.z: z})  # [-1, 8, 64, 64, 3]
        images = np.transpose(images, [0, 2, 1, 3, 4])
        images = np.reshape(images, [-1, 64 * 8, 3])
        cv2.imwrite(self.config.simple_path, images * 255)
        # cv2.imwrite(self.config.simple_path, images)

    def test(self, ds_test):
        ts = self.ts.sub_ts[-1]
        f1, z1 = self.session.run([ts.f, ts.z], {ts.x: [ds_test[0]]})
        f2, z2 = self.session.run([ts.f, ts.z], {ts.x: [ds_test[3]]})

        images = self.session.run(ts.predict_y, {ts.f: f1, ts.z: z2})  # [-1, 8, 64, 64, 3]
        images = np.transpose(images, [0, 2, 1, 3, 4])
        images = np.reshape(images, [-1, 64 * 8, 3])
        # todo
        cv2.imwrite(self.config.simple_path, images * 255)
        # cv2.imwrite(self.config.simple_path, images)


if __name__ == '__main__':
    tf.disable_eager_execution()
    config = Config()
    # app = config.get_app()
    # app.test_random()
    config.call("test")

    # ds = DS()
    # data = ds.next_batch(1)[0][0][0]
    # print(data)