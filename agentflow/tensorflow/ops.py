import tensorflow as tf
import numpy as np

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

def normalize_ema(input_tensor,training=True,auto_update=False,ema_decay=0.999,axis=-1):
    # IMPORTANT: don't use keras BatchNormalization because of how vars are created
    BN = tf.layers.BatchNormalization(
        axis=axis,
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

def get_gradient_matrix_old(var_list,objective):
    def func(obj_i):
        grad_i = tf.gradients(obj_i,var_list)
        grad_i = tf.concat([tf.reshape(g,[-1]) for g in grad_i],axis=0)
        return grad_i
    grads = tf.vectorized_map(func,objective)
    nvars = sum([np.prod(v.shape) for v in var_list])
    grads = tf.reshape(grads,[-1,nvars])
    return grads

def get_gradient_matrix(var_list,objective,filter_unconnected_vars=True):
    if filter_unconnected_vars:
        grad_conn_check = tf.gradients(objective,var_list)
        var_list = [v for g,v in zip(grad_conn_check,var_list) if g is not None]
    def func(obj_i):
        grad_i = tf.gradients(obj_i,var_list)
        if not filter_unconnected_vars:
            grad_i = [
                tf.zeros(v.shape) if g is None else g 
                for g,v in zip(grad_i,var_list)
            ]
        grad_i = tf.concat([tf.reshape(g,[-1]) for g in grad_i],axis=0)
        return grad_i
    grads = tf.vectorized_map(func,objective)
    nvars = sum([np.prod(v.shape) for v in var_list])
    grads = tf.reshape(grads,[-1,nvars])
    return var_list, grads

def softmax_hessian(X,p):
    Xp = X*p[:,:,None]
    Xpsum = tf.reduce_sum(Xp,axis=1)
    Xpsum_outer = Xpsum[:,:,None]*Xpsum[:,None,:]
    XXp = tf.reduce_sum(Xp[:,:,:,None]*X[:,:,None,:],axis=1)
    H = XXp - Xpsum_outer
    return tf.reduce_sum(H,axis=0)

def softmax_grad(X,p,y):
    y = onehot(y,p.shape[1])
    return tf.reduce_sum(tf.reduce_sum(X*(y-p)[:,:,None],axis=1),axis=0)

def get_modified_gradients_softmax(y_true,logits,var_list,alpha=1,axis=1):
    assert axis==1
    p = tf.nn.softmax(logits,axis=axis)
    N, C = tf.shape(logits)
    X = tf.reshape(get_gradient_matrix(var_list,tf.reshape(logits,[-1])),[N,C,-1])
    H = softmax_hessian(X,p)
    g = softmax_grad(X,p,y_true)
    g += alpha*w.ravel()
    H += alpha*np.eye(len(g))
    H_inv = np.linalg.inv(H)
    modified_grad_flat = tf.linalg.matmul(H_inv,g)

    modified_grad = []
    i = 0
    for v in var_list:
        w = np.prod(v.shape).value
        g = tf.reshape(modified_grad_flat[i:i+w],v.shape)
        modified_grad.append((g,v))
        i += w

    supplementary_output = {
        'modified_grad_flat': modified_grad_flat,
    }
    return modified_grad, supplementary_output


