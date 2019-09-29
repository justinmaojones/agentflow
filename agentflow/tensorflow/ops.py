import tensorflow as tf

def exponential_moving_average(scopes,**kwargs):
    ema = tf.train.ExponentialMovingAverage(**kwargs)
    model_vars = [v for s in scopes for v in tf.trainable_variables(s)]
    ema_op = ema.apply(model_vars)

    def ema_vars_getter(getter, name, shape, *args, **kwargs):
        # this will return the moving average of variables in scope
        v = getter(name=name,shape=shape,*args,**kwargs)
        ema_output = ema.average(v)
        if ema_output is None:
            print('CANNOT FIND EMA for %s'%name)
            return v
        else:
            return ema_output

    return ema_op, ema_vars_getter

def normalize_ema_old(tensor,axis=0,eps=1e-8,decay=0.999,zero_debias=True,**ema_kwargs):
    X = tf.reduce_mean(tensor,axis=axis)
    X2 = tf.reduce_mean(tf.square(tensor),axis=axis)
    print(X.name)
    ema = tf.train.ExponentialMovingAverage(decay=decay,zero_debias=zero_debias,**ema_kwargs)
    ema_op = ema.apply([X,X2])
    with tf.control_dependencies([ema_op]):
        X_avg, X2_avg = ema.average(X), ema.average(X2)

        for v in [X_avg,X2_avg]:
            if v not in tf.moving_average_variables():
                tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, v)

        ema_variance = X2_avg-tf.square(X_avg)
        normalized = (tensor-X_avg)/tf.sqrt(ema_variance+eps)
    return normalized


def normalize_ema(input_tensor,**kwargs):
    # IMPORTANT: don't use keras BatchNormalization because of how vars are created
    BN = tf.layers.BatchNormalization(**kwargs)
    batchnorm = BN(input_tensor,training=True)
    with tf.control_dependencies(BN.updates):
        normalized = BN(input_tensor,training=False)
    return normalized
