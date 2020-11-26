import Framework as fm
import Models
from Data import DS
from App import App


class Config(fm.Config):
    def __init__(self):
        super().__init__()
        self.frame = 8
        self.input_size = 64
        self.x_size = 2048
        self.f_size = 256
        self.z_size = 6
        self.hidden_size = 512
        self.factorised = False  # True : factorised q, False : full q
        self.lr = 0.0001
        self.epoches = 800
        self.batch_size = 10
        self.new_model = False
        self.simple_path = "images/{name}/test_".format(name=self.get_name())
        self.simple_num = 1
        self.ds = None
        self.training = True

    def get_name(self):
        return "test13"

    def get_sub_tensors(self, gpu_index):
        return Models.SubTensor(self)

    def get_app(self):
        return App(self)

    def get_ds_train(self):
        self.read_ds()
        return self.ds

    def get_ds_test(self):
        self.read_ds()
        return self.ds.test_ds

    def read_ds(self):
        if self.ds is None:
            self.ds = DS()

    def get_tensors(self):
        return Models.MyTensors(self)

    def test(self):
        self.training = False
        self.batch_size = 1
        super(Config, self).test()


if __name__ == '__main__':
    config = Config()
    # config.get_ds_train()
    config.from_cmd()