import tensorflow as tf

def layer_normalization(x, scope=None, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    scope = 'layer_normalization' if scope is None else scope
    with tf.variable_scope(scope):
        n_state = x.shape[axis].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def dense_net(x,units,layers,batchnorm=False,activation=tf.nn.relu,training=False,layernorm=False,**kwargs):

    assert isinstance(layers,int) and layers > 0, 'layers should be a positive integer'
    assert isinstance(units,int) and units > 0, 'units should be a positive integer'
    assert not (batchnorm and layernorm), "cannot have both batchnorm and layernorm be true"

    h = x
    for l in range(layers):
        h = tf.layers.dense(h,units,**kwargs)
        if batchnorm:
            BN = tf.layers.BatchNormalization()
            h = BN(h,training=training)
        if layernorm:
            h = layer_normalization(h,scope='dense_%d/layer_normalization'%l)
        h = activation(h)

    return h

