import tensorflow as tf

def layer_normalization(x,axis=-1):
    return tf.keras.layers.LayerNormalization(axis)(x)

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
            h = layer_normalization(h)
        h = activation(h)

    return h

