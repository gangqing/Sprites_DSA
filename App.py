import numpy as np
import cv2
import Framework as fm


class App(fm.App):

    def test(self, ds_test):
        x1 = ds_test[0]
        x2 = ds_test[-1]

        self.reconstruction(x1)
        self.features_change(x1, x2)
        self.reconstruction_with_random_f(x1)
        self.reconstruction_with_random_z(x1)
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
        simple_f = np.random.normal(0, 1, size=[1, self.config.f_size])  # [-1, 256]
        images_pre = self.session.run(ts.predict_y, {ts.f: simple_f, ts.z: z})  # [-1, 8, 64, 64, 3]

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
        simple_z = self.reparameterize(z_mean, z_logvar)
        images_pre = self.session.run(ts.predict_y, {ts.f: f, ts.z: simple_z})  # [-1, 8, 64, 64, 3]
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
        simple_z = self.reparameterize(mean_z, logvar_z)  # [-1, 8, 32]

        images = self.session.run(ts.predict_y, {ts.f: f, ts.z: simple_z})  # [-1, 8, 64, 64, 3]
        images = np.transpose(images, [0, 2, 1, 3, 4])  # [-1, 64, 8, 64, 3]
        images = np.reshape(images, [-1, 64 * 8, 3])

        cv2.imwrite(self.config.simple_path + "random.jpg", images * 255)

    def reparameterize(self, mean, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.normal(size=logvar.shape)
        return mean + std * eps