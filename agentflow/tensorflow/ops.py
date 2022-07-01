import tensorflow as tf
import numpy as np

def binarize(x, b):
    y = []
    for i in range(b):
        y.append(tf.mod(x, 2))
        x = x // 2
    return tf.concat(y, axis=-1)

def entropy_loss(logits, axis=-1):
    n = len(logits.shape)
    if axis < 0:
        axis += n
    p = tf.nn.softmax(logits, axis=axis)
    return tf.nn.softmax_cross_entropy_with_logits(labels=p, logits=logits, dim=axis)

def l2_loss(weights):
    return 0.5 * tf.reduce_sum([tf.nn.l2_loss(x) for x in weights])

def onehot_argmax(x, axis=-1):
    assert isinstance(axis, int)
    n = len(x.shape)
    if axis < 0:
        axis = n + axis
    indices = tf.argmax(x, axis=axis)
    return tf.one_hot(indices, x.shape[axis], axis=axis)

def value_at_argmax(x, y, axis=-1):
    """
    Returns the value of y at argmax index of x along provided axis.
    """
    assert isinstance(axis, int), 'axis must be an integer'
    assert x.shape.as_list() == y.shape.as_list(), "shapes of x and y must match"
    n = len(x.shape)
    if axis < 0:
        axis = n + axis
    assert axis >= 0 and axis < n, 'invalid axis argument'
    indices = tf.argmax(x, axis=axis)
    if axis != n-1:
        transpose = list(range(n))
        transpose.pop(axis)
        transpose.append(axis)
        y = tf.transpose(y, transpose)
    return tf.gather(y, tf.expand_dims(indices, axis=-1), axis=-1, batch_dims=n-1)[..., 0]

def weighted_avg(x, weights, axis=-1):
    return tf.reduce_sum(x*weights, axis=axis) / (1e-8 + tf.reduce_sum(weights, axis=axis))

