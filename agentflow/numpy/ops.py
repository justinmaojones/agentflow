import numpy as np

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x,axis=-1):
    return np.exp(x)/np.exp(x).sum(axis=axis,keepdims=True)

def onehot(x,depth=2):
    shape = list(x.shape)+[2]
    y = np.zeros(shape)
    y[np.arange(len(x)),x] = 1.
    return y.astype('float32')

def noisy_action(action_softmax,eps=1.,clip=5e-2):
    action_softmax_clipped = np.minimum(1-clip,np.maximum(clip,action_softmax))
    logit = np.log(action_softmax_clipped)
    u = np.random.rand(*logit.shape)
    g = -np.log(-np.log(u))
    return (eps*g+logit).argmax(axis=-1)


