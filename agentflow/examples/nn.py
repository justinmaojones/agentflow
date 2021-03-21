import tensorflow as tf

def dqn_atari_paper_net(x, scale=1, **kwargs):
    """
    Neural network based on:
    Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." 
    arXiv preprint arXiv:1312.5602 (2013).
    """
    h = tf.layers.conv2d(x, filters=int(scale*16), kernel_size=8, strides=4, **kwargs)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d(h, filters=int(scale*32), kernel_size=4, strides=2, **kwargs)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d(h, filters=int(scale*32), kernel_size=3, strides=1, **kwargs)
    h = tf.nn.relu(h)
    h = tf.layers.flatten(h)
    h = tf.layers.dense(h, units=int(scale*256), **kwargs) 
    return h

def dqn_atari_paper_net_dueling(x, scale=1, **kwargs):
    """
    Neural network based on:
    Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." 
    arXiv preprint arXiv:1312.5602 (2013).
    Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." 
    International conference on machine learning. PMLR, 2016.
    """
    h = tf.layers.conv2d(x, filters=int(scale*16), kernel_size=8, strides=4, **kwargs)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d(h, filters=int(scale*32), kernel_size=4, strides=2, **kwargs)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d(h, filters=int(scale*32), kernel_size=3, strides=1, **kwargs)
    h = tf.nn.relu(h)
    h = tf.layers.flatten(h)
    h1 = tf.layers.dense(h, units=int(scale*256), **kwargs) 
    h2 = tf.layers.dense(h, units=int(scale*256), **kwargs) 

    return h1, h2

