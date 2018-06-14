# coding: utf-8
import cupy as cp
#import numpy as cp
import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - cp.max(x, axis=0)
        y = cp.exp(x, dtype=np.float32) / cp.sum(cp.exp(x, dtype=np.float32), axis=0, dtype=np.float32)
        return y.T 

    x = x - cp.max(x) # オーバーフロー対策
    return cp.exp(x) / cp.sum(cp.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -cp.sum(cp.log(y[cp.arange(batch_size), t])) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
