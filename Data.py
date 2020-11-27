import cv2
import os
import numpy as np


def read_file(file_path, is_sort=False):
    images = []
    file_name_list = os.listdir(file_path)
    if is_sort:
        file_name_list.sort()
    for filename in file_name_list:
        img = cv2.imread(os.path.join(file_path, filename))
        imgs = []
        if img is None:
            continue
        for i in range(8):
            sub_img = img[:, i * 64: i * 64 + 64, :]
            imgs.append(sub_img)
        images.append(imgs)
    return (np.array(images) / 255).tolist()


class DS:
    def __init__(self):
        self.train_path = "simples/train"
        self.test_path = "simples/vali"

        self.ds = read_file(self.train_path)
        self.test_ds = read_file(self.test_path)

        print("simple : {length}".format(length=self.num_examples))
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

    @property
    def num_examples(self):
        return len(self.ds)