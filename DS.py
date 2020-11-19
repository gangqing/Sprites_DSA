import cv2
import os
import numpy as np


class DS:
    def __init__(self):
        self.train_path = "simples/data/train"
        self.test_path = "simples/data/vali"
        images = []
        for filename in os.listdir(self.train_path):
            img = cv2.imread(os.path.join(self.train_path, filename))
            imgs = []
            for i in range(8):
                sub_img = img[:, i * 64: i * 64 + 64, :]
                imgs.append(sub_img)
            images.append(imgs)
        self.ds = (np.array(images) / 255).tolist()
        print("simple : {length}".format(length=len(images)))
        self.op = np.random.randint(0, self.num_examples)

    def next_batch(self, batch_size):
        next_op = self.op + batch_size
        if next_op > self.num_examples:
            result = self.ds[self.op:]
            next_op -= self.num_examples
            result.extend(self.ds[:next_op])
        else:
            result = self.ds[self.op: next_op]
        self.op = next_op

        return [result]

    def test_ds(self):
        images = []
        file_name_list = os.listdir(self.test_path)
        file_name_list.sort()
        for filename in file_name_list:
            img = cv2.imread(os.path.join(self.test_path, filename))
            imgs = []
            for i in range(8):
                sub_img = img[:, i * 64: i * 64 + 64, :]
                imgs.append(sub_img)
            images.append(imgs)
        return (np.array(images) / 255).tolist()

    @property
    def num_examples(self):
        return len(self.ds)