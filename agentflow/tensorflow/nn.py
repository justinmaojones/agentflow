import tensorflow as tf

def dense_net(x,units,layers,batchnorm=True,activation=tf.nn.relu,training=False,**kwargs):

    assert isinstance(layers,int) and layers > 0, 'layers should be a positive integer'
    assert isinstance(units,int) and units > 0, 'units should be a positive integer'

    h = x
    for l in range(layers):
        h = tf.layers.dense(h,units,**kwargs)
        if batchnorm:
            BN = tf.layers.BatchNormalization()
            h = BN(h,training=training)
            with tf.control_dependencies(BN.updates):
                h = activation(h)
        else:
            h = activation(h)
    return h
    
