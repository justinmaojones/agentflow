import tensorflow as tf

def _batch_reshape(x, batchsize):
    y = {}
    for k in x:
        s = x[k].shape
        if s[0] % batchsize != 0:
            raise ValueError(f"first dimension (shape {s[0]}) must be a multiple of batchsize (shape {batchsize})")
        y[k] = tf.reshape(x[k], (s[0]//batchsize, batchsize, *s[1:]))
    return y

def split(dataset, batchsize):
    """
    Splits a dataset along the first dimension into elements each with `batchsize`
    elements along the first dimension.  Fails if first dimension is not a multiple
    of `batchsize`.
    """
    return dataset.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(_batch_reshape(x, batchsize))
    )

