import tensorflow as tf

def exponential_moving_average(scope=None,**kwargs):
    ema = tf.train.ExponentialMovingAverage(**kwargs)
    model_vars = tf.trainable_variables(scope)
    ema_op = ema.apply(model_vars)

    def ema_vars_getter(getter, name, shape, *args, **kwargs):
        # this will return the moving average of variables in scope
        v = getter(name=name,shape=shape,*args,**kwargs)
        ema_output = ema.average(v)
        if ema_output is None:
            return v
        else:
            return ema_output

    return ema, ema_op, ema_vars_getter

def normalize_ema(input_tensor,training=True,auto_update=False,ema_decay=0.999):
    # IMPORTANT: don't use keras BatchNormalization because of how vars are created
    BN = tf.layers.BatchNormalization(
        momentum=ema_decay,
        center=False,
        scale=False,
    )
    if training:
        batchnorm = BN(input_tensor,training=True)
    if auto_update:
        with tf.control_dependencies(BN.updates):
            normalized = BN(input_tensor,training=False)
    else:
        normalized = BN(input_tensor,training=False)
    return normalized, BN.updates
