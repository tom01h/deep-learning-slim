# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import cupy as cp
#import numpy as cp
import numpy as np
from collections import OrderedDict
from common.layers import *


class ConvNet:
    """単純なConvNet
    """
    def __init__(self, input_dim=(3, 32, 32), weight_init_std=0.01):

        # 重みの初期化
        self.params = {}
        self.params['W1'] = cp.array( weight_init_std * \
                            cp.random.randn(32, 3, 3, 3), dtype=np.float32)

        self.params['W2'] = cp.array( weight_init_std * \
                            cp.random.randn(64, 32, 3, 3), dtype=np.float32)

        self.params['W3'] = cp.array( weight_init_std * \
                            cp.random.randn(128, 64, 3 ,3), dtype=np.float32)

        self.params['W4'] = cp.array( weight_init_std * \
                            cp.random.randn(128, 128, 3, 3), dtype=np.float32)

        self.params['W5'] = cp.array( weight_init_std * \
                            cp.random.randn(256, 128, 1, 1), dtype=np.float32)

        self.params['W6'] = cp.array( weight_init_std * \
                            cp.random.randn(256, 10), dtype=np.float32)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution8(self.params['W1'], stride=1, pad=1, fill=-128)
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['LightNorm1'] = LightNormalization()
        self.layers['Activ1'] = Relu8()

        self.layers['Conv2'] = Convolution8(self.params['W2'], stride=1, pad=1, fill=0) # BinActiv<=1 , other<=0
        self.layers['LightNorm2'] = LightNormalization()
        self.layers['Activ2'] = Relu8()

        self.layers['Conv3'] = Convolution8(self.params['W3'], stride=1, pad=1, fill=0) # BinActiv<=1 , other<=0
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['LightNorm3'] = LightNormalization()
        self.layers['Activ3'] = Relu8()

        self.layers['Conv4'] = Convolution8(self.params['W4'], stride=1, pad=1, fill=0) # BinActiv<=1 , other<=0
        self.layers['Pool4'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['LightNorm4'] = LightNormalization()
        self.layers['Activ4'] = Relu8()

        self.layers['Conv5'] = Convolution8(self.params['W5'], stride=1, pad=0, fill=0) # BinActiv<=1 , other<=0
        self.layers['LightNorm5'] = LightNormalization()
        self.layers['Activ5'] = Relu8()

        self.layers['Pool'] = AvgPooling(pool_h=4, pool_w=4, stride=4)

        self.layers['Affine6'] = Affine8(self.params['W6'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "LightNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = cp.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = cp.argmax(y, axis=1)
            acc += cp.sum(y == tt).get() #cupy
#            acc += cp.sum(y == tt) #numpy
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝搬法）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
        """
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['W2'] = self.layers['Conv2'].dW
        grads['W3'] = self.layers['Conv3'].dW
        grads['W4'] = self.layers['Conv4'].dW
        grads['W5'] = self.layers['Conv5'].dW
        grads['W6'] = self.layers['Affine6'].dW
        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val.get()
        for i, key in enumerate(['LightNorm1', 'LightNorm2', 'LightNorm3', 'LightNorm4', 'LightNorm5']):
            params['mean' + str(i+1)] = self.layers[key].running_mean.get()
            params['var' + str(i+1)] = self.layers[key].running_var.get()
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            if "W" in key:
                self.params[key] = val
        for i, key in enumerate(['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Affine6']):
            self.layers[key].W = cp.array(self.params['W' + str(i+1)])
        for i, key in enumerate(['LightNorm1', 'LightNorm2', 'LightNorm3', 'LightNorm4', 'LightNorm5']):
            self.layers[key].running_var = cp.array(params['var' + str(i+1)])
            self.layers[key].running_mean= cp.array(params['mean' + str(i+1)])
