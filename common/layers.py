# coding: utf-8
from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import *
from common.util import im2col, col2im

class BinActiv:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (np.absolute(x) > 1)
        out = np.ones_like(x, dtype=np.float32)
        out[x < 0] = -1

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Relu8:
    def __init__(self, fix=32):
        self.mask = None
        self.fix = fix

    def forward(self, x):
        self.mask = (x <= 0) | (x*self.fix >= 127)
        out = np.array(x*self.fix, dtype=np.int8)
        out[(x*self.fix>=127)] = 127
        out[(x <= 0)] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout*self.fix

        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class BinAffine:
    def __init__(self, W):
        self.W = W
        self.bW = None
#        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
#        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        bW = np.ones_like(self.W)
        bW[self.W<0] = -1
        self.bW = (bW).T

        out = np.dot(self.x, bW) #+ self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.bW)
        self.dW = np.dot(self.x.T, dout)
#        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class Affine8:
    def __init__(self, W, fix=4):
        self.W =W
        self.fix = fix

        self.x = None
        self.coef = None
        self.W8 = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        coef = 1/np.amax(np.absolute(self.W))*127
        W8 = np.array(self.W*coef, dtype=np.int8)
        out = np.dot(self.x, W8)/coef/self.fix
        out[(out>= (2**15-1))] =  (2**15-1)
        out[(out<=-(2**15-1))] = -(2**15-1)
        out = np.array(out, dtype=np.int16)

        self.coef=coef
        self.W8 = W8
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W8.T)/self.coef
        self.dW = np.dot(self.x.T, dout)/self.fix

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx/self.fix


class Affine:
    def __init__(self, W):
        self.W = W
#        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
#        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) #+ self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
#        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class LightNormalization:
    """
    """
    def __init__(self, momentum=0.9, running_mean=None, running_var=None):
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim == 2:
            N, D = x.shape
            x = x.reshape(N, D, 1, 1)

        x = x.transpose(0, 2, 3, 1)
        out = self.__forward(x, train_flg)
        out = out.transpose(0, 3, 1, 2)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, H, W, C = x.shape
            self.running_mean = np.zeros(C, dtype=np.float32)
            self.running_var = np.zeros(C, dtype=np.float32)

        if train_flg:
            mu = x.mean(axis=(0, 1, 2))
            xc = x - mu
            var = np.mean(xc**2, axis=(0, 1, 2), dtype=np.float32)
            std = np.sqrt(var + 10e-7, dtype=np.float32)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7, dtype=np.float32)))

        out = xn
        return out

    def backward(self, dout):
        if dout.ndim == 2:
            N, D = dout.shape
            dout = dout.reshape(N, D, 1, 1)

        dout = dout.transpose(0, 2, 3, 1)
        dx = self.__backward(dout)
        dx = dx.transpose(0, 3, 1, 2)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dxn = dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        return dx

class BinConvolution:
    def __init__(self, W, stride=1, pad=0, fill=0):
        self.W = W
#        self.b = b
        self.stride = stride
        self.pad = pad
        self.fill = fill

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
#        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        scale = np.amax(np.absolute(self.W))
        col = im2col(x, FH, FW, self.stride, self.pad, self.fill)
        col_W = self.W.reshape(FN, -1).T
        col_Wb = np.ones_like(col_W)
        col_Wb[col_W<0] = -1

        out = np.dot(col, col_Wb)*scale #+ self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_Wb

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

#        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Convolution8:
    def __init__(self, W, stride=1, pad=0, fill=0, fix=4):
        self.W = W
#        self.b = b
        self.stride = stride
        self.pad = pad
        self.fill = fill
        self.fix = fix

        # 中間データ（backward時に使用）
        self.x = None
        self.coef = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
#        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad, self.fill)
        col_W = self.W.reshape(FN, -1).T
        coef = 1/np.amax(np.absolute(col_W))*127
        col_W8 = np.array(col_W*coef, dtype=np.int8)

        out = np.dot(col, col_W8)/self.fix #+ self.b
        out[(out>= (2**15-1))] =  (2**15-1)
        out[(out<=-(2**15-1))] = -(2**15-1)
        out = np.array(out, dtype=np.int16)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W8
        self.coef = coef

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

#        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)*self.coef
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)/self.fix

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx/self.fix


class Convolution:
    def __init__(self, W, stride=1, pad=0, fill=0):
        self.W = W
#        self.b = b
        self.stride = stride
        self.pad = pad
        self.fill = fill
    
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
#        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad, self.fill)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) #+ self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

#        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.array(np.max(col, axis=1), dtype=np.float32)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size), dtype=np.float32)
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class AvgPooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.dx = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        out = np.array(np.mean(col, axis=1), dtype=np.float32)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x

        return out

    def backward(self, dout):
        dx = dout.reshape(100,256)
        dx = np.tile(dx,(4,4,1,1))
        dx = dx.transpose(2,3,0,1)
        dx = dx/self.pool_h/self.pool_w
        self.dx = dx

        return dx

