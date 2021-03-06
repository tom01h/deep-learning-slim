# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.np import *  # import numpy as np
from common.config import GPU
from common.optimizer import *
import numpy

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = numpy.random.choice(self.train_size, self.batch_size)
        x_batch = numpy.array(self.x_train[batch_mask])
        t_batch = np.array(self.t_train[batch_mask])

        if self.max_iter % 2 == 1:
            x_batch = x_batch[:, :, :, ::-1]

        padded = numpy.pad(x_batch, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')#mean
        crops = numpy.random.randint(4, size=(len(x_batch), 2))
        x_batch = np.array([padded[i, :, c[0]:(c[0]+32), c[1]:(c[1]+32)] for i, c in enumerate(crops)])

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print(str(self.current_epoch) + " : " + str(int(self.current_iter % self.iter_per_epoch)) + " : train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            if self.evaluate_sample_num_per_epoch is None:
                x_train_sample = np.array(self.x_train)
                t_train_sample = np.array(self.t_train)
                x_test_sample = np.array(self.x_test)
                t_test_sample = np.array(self.t_test)
            else:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample = np.array(self.x_train[:t])
                t_train_sample = np.array(self.t_train[:t])
                x_test_sample = np.array(self.x_test[:t])
                t_test_sample = np.array(self.t_test[:t])
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(np.array(self.x_test), np.array(self.t_test))

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

