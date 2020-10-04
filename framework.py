import argparse
from tensorflow.compat import v1 as tf
import os


def get_gpus():
    value = os.getenv('CUDA_VISIBLE_DEVICES', '1,2')  # 获取可使用的gpu，默认值为0
    value = value.split(",")  # 根据,号划分成数组
    return len(value)


def make_dirs(path: str):
    pos = path.rfind(os.sep)
    if pos < 0:
        raise Exception('Can not find the directory from the path', path)
    path = path[:pos]
    os.makedirs(path, exist_ok=True)


class Config:
    def __init__(self):
        tf.disable_eager_execution()
        self.save_path = "models/{name}/{name}".format(name=self.get_name())
        self.simple_path = None
        self.logdir = "logs/{name}".format(name=self.get_name())
        self.lr = 0.001
        self.epoches = 2000
        self.batch_size = 200
        self.new_model = False
        self.gpus = get_gpus()
        self.col = None
        self.image_path = None

    def get_name(self):
        raise Exception("get_name() is not re-defined !")

    def __repr__(self):
        attrs = self.get_attrs()  # 字典
        result = ["{attr} = {value}".format(attr=attr, value=attrs[attr]) for attr in attrs]

        return ", ".join(result)

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith("_"):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in [int, float, bool, str]:
                result[attr] = value
        return result

    def from_cmd(self):
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            if type(value) == bool:
                parser.add_argument("--{attr}".format(attr=attr), default=value, help="default to %s" % value,
                                   action='store_%s' % ('false' if value else 'true'))
            else:
                parser.add_argument("--{attr}".format(attr=attr), type=type(value), default=value, help="default to %s" % value)

        parser.add_argument("--call", type=str, default="data", help="call method, by default call data()")
        arg = parser.parse_args()
        for attr in attrs:
            if hasattr(arg, attr):
                setattr(self, attr, getattr(arg, attr))

        self.call(arg.call)

    def call(self, name):
        if name == "train":
            self.train()
        elif name == "test":
            self.test()
        else:
            print("unknow method name : %s" % name)

    def train(self):
        app = self.get_app()
        with app:
            app.train(self.get_ds_train(), None)

    def get_ds_train(self):
        raise Exception("get_ds_train() is not default")

    def test(self):
        app = self.get_app()
        with app:
            app.test(self.get_ds_test())

    def get_ds_test(self):
        raise Exception("get_ds_test() is not default")

    def get_sub_tensors(self, gpu_index):
        raise Exception('The get_sub_tensors() is not defined.')

    def get_tensors(self):
        return Tensors(self)

    def get_app(self):
        return App(self)

    def get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)


class Tensors:
    def __init__(self, config: Config):
        self.config = config
        self.sub_ts = []
        with tf.variable_scope(config.get_name()) as scpoe:  # 指定变量范围
            for i in range(config.gpus):  # 遍历所有gpu
                with tf.device("/gpu:{i}".format(i=i)):  # 运行指定gpu
                    self.sub_ts.append(config.get_sub_tensors(i))  # 创建sub_tensors
                    scpoe.reuse_variables()

        # 汇总
        with tf.device("/gpu:1"):  # 指定运行0号gpu, 在0号gpu上计算loss
            with tf.variable_scope("{name}_train".format(name=config.get_name())):  # 汇总张量的命名
                # 计算loss均值
                losses = [ts.losses for ts in self.sub_ts]  # 获取所有gpu上的loss张量, [gpus, losses]
                self.losses = tf.reduce_mean(losses, axis=0)  # 计算loss的均值，[losses]
                # 计算loss递度
                self.lr = tf.placeholder(dtype=tf.float32, shape=None, name="lr")
                opt = config.get_optimizer(self.lr)
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):  # 控制依赖
                    grads = self.compute_grads(opt)  # 计算梯度， [loss, -1, (grad, var)]
                    print(grads)
                    self.apply_grads(grads, opt)
            # summary
            for i in range(len(losses[0])):
                tf.summary.scalar('loss_%d' % i, self.get_loss_for_summary(self.losses[i]))
            self.summary = tf.summary.merge_all()

    def get_loss_for_summary(self, loss):
        return loss

    def apply_grads(self, loss_grads, opt):
        """"
        :param loss_grads:  [loss, -1, (grad, var)]
        :param opt: 
        :return: 
        """""
        self.train_ops = []
        for grad_var in loss_grads:  # [-1, (grad, var)]
            train_op = opt.apply_gradients(grad_var)  # 应用计算得到的梯度来更新对应的variable
            self.train_ops.append(train_op)  # 一个loss对应一个train_op

    def compute_grads(self, opt):
        """"
        计算递度
        :param opt: 优化器
        :return: [loss, -1, (grad, var)]
        """""
        grads = []  # 所有gpu的梯度
        for ts in self.sub_ts:
            loss_grads = []  # 用于记录每个gpu下的梯度
            for loss in ts.losses:
                grad = opt.compute_gradients(loss)  # 计算loss的平均梯度, [-1, (grad, var)]
                loss_grads.append(grad)  # [loss, -1, (grad, var)]
            grads.append(loss_grads)  # [gpu, loss, -1, (grad, var)]

        # 计算loss下各维的梯度的平均值
        loss_grads_mean = []  # [loss, -1, (grad, var)]
        for index in range(len(grads[0])):  # 循环loss
            grad_mean = self.get_grads_mean(grads, index)  # [-1, (grad, var)]
            loss_grads_mean.append(grad_mean)
        return loss_grads_mean

    def get_grads_mean(self, grads, loss_index):
        """"
        计算某个loss的平均递度
        :param grads: [gpus, loss, -1, (gradient, variable)]
        :param loss_index:
        :return: [-1, (grad_mean, var)]
        """""
        loss_grad_var = []
        for gpu in grads:  # gpu : [loss, -1, (gradient, variable)]
            grad_var = gpu[loss_index]  # [-1, (gradient, variable)]
            loss_grad_var.append(grad_var)  # [gpu, -1, (gradient, variable)]

        # 获取变量名，所有loss下的变量都是一样的，所以从第一个loss中获取变量名即可
        vars = []
        for pair in loss_grad_var[0]:  # pair : [-1, (gradient, variable)] , 第一个是梯度，第二个是变量
            var = pair[1]  # var
            vars.append(var)  # [num_var]

        result = []
        for var_index, var in enumerate(vars):
            grads = []
            for grad_var in loss_grad_var:  # grad_var : [-1, (gradient, variable)]
                grad = grad_var[var_index][0]  # grad
                grads.append(grad)  # [grad]
            grad_mean = tf.reduce_mean(grads, axis=0)  # 计算某个变量的平均递度
            print("var: {var}, grad_mean : {mean}".format(var=var,mean=grad_mean))
            result.append((grad_mean, var))  # [-1, (grad_mean, var)]

        return result


class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()  # 图
        with graph.as_default():
            self.ts = config.get_tensors()
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True  # 在没有gpu时使用cpu
            self.session = tf.Session(graph=graph, config=cfg)
            self.saver = tf.train.Saver()
            if config.new_model:
                self.session.run(tf.global_variables_initializer())
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                except:
                    self.session.run(tf.global_variables_initializer())

    def train(self, ds_train, ds_validation):
        self.before_train()
        cfg = self.config
        writer = tf.summary.FileWriter(logdir=cfg.logdir, graph=self.session.graph)
        batches = ds_train.num_examples // (cfg.batch_size * cfg.gpus)

        for epoch in range(cfg.epoches):
            self.before_epoch(epoch)
            for batch in range(batches):
                self.before_batch(epoch, batch)
                feed_dict = self.get_feed_dict(ds_train)
                if len(self.ts.train_ops) == 1:
                    _, summary = self.session.run([self.ts.train_ops[0], self.ts.summary], feed_dict)
                else:
                    for train_op in self.ts.train_ops:
                        self.session.run(train_op, feed_dict)
                    summary = self.session.run(self.ts.summary, feed_dict)

                writer.add_summary(summary, global_step=epoch * batches + batch)
                print("epoch = {epoch} , batch = {batch} , loss = {summary}".format(epoch=epoch, batch=batch, summary=summary))
                self.after_batch(epoch, batch)
            # precise = session.run(self.ts.precise_summary, self.get_feed_dict(ds_validation))
            # writer.add_summary(precise, global_step = epoch)
            self.after_epoch(epoch)
        self.after_train()

    def before_train(self):
        pass

    def before_epoch(self, epoch):
        pass

    def before_batch(self, epoch, batch):
        pass

    def after_batch(self, epoch, batch):
        # self.save()
        pass

    def after_epoch(self, epoch):
        pass

    def after_train(self):
        self.save()
        # pass

    def get_feed_dict(self, ds):
        result = {self.ts.lr: self.config.lr}
        for i in range(self.config.gpus):
            values = ds.next_batch(self.config.batch_size)
            for tensor, value in zip(self.ts.sub_ts[i].inputs, values):
                result[tensor] = value
        return result

    def save(self):
        self.saver.save(self.session, save_path=self.config.save_path)

    def test(self, ds_test):
        pass

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
