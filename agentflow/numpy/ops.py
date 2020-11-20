import numpy as np

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x,axis=-1):
    return np.exp(x)/np.exp(x).sum(axis=axis,keepdims=True)

def onehot(x,depth=2):
    shape = list(x.shape)+[depth]
    y = np.zeros(shape)
    y[np.arange(len(x)),x] = 1.
    return y.astype('float32')

def binarize(x,b):
    y = []
    for i in range(b):
        y.append(x % 2)
        x = x // 2
    return np.concatenate(y,axis=-1)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
