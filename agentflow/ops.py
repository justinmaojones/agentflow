import tensorflow as tf

def exponential_moving_average(scopes,**kwargs):
    ema = tf.train.ExponentialMovingAverage(**kwargs)
    model_vars = [v for s in scopes for v in tf.trainable_variables(s)]
    ema_op = ema.apply(model_vars)

    def custom_getter(getter, name, shape, *args, **kwargs):
        # this will return the moving average of variables in scope
        return ema.average(getter(name=name,shape=shape,*args,**kwargs))

    return ema_op, ema_vars_getter

