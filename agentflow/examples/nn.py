import tensorflow as tf


def _get_name_fn(name):
    def f(key):
        if name is None:
            return None
        return f"{name}/{key}"

    return f


def dqn_atari_paper_net_base(x, scale=1, name=None, **kwargs):
    """
    Neural network based on:
    Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
    arXiv preprint arXiv:1312.5602 (2013).
    """

    _name = _get_name_fn(name)

    h = tf.keras.layers.Conv2D(
        filters=int(scale * 16),
        kernel_size=8,
        strides=4,
        name=_name("conv_1"),
        **kwargs,
    )(x)
    h = tf.nn.relu(h)

    h = tf.keras.layers.Conv2D(
        filters=int(scale * 32),
        kernel_size=4,
        strides=2,
        name=_name("conv_2"),
        **kwargs,
    )(h)
    h = tf.nn.relu(h)

    h = tf.keras.layers.Conv2D(
        filters=int(scale * 32),
        kernel_size=3,
        strides=1,
        name=_name("conv_3"),
        **kwargs,
    )(h)
    h = tf.nn.relu(h)

    h = tf.keras.layers.Flatten()(h)

    h = tf.keras.layers.Dense(units=int(scale * 256), name=_name("dense_1"), **kwargs)(
        h
    )
    h = tf.nn.relu(h)

    return h


def dqn_atari_paper_net(x, n_actions, n_heads=1, scale=1, name=None, **kwargs):
    """
    Neural network based on:
    Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
    arXiv preprint arXiv:1312.5602 (2013).
    """

    _name = _get_name_fn(name)

    h = dqn_atari_paper_net_base(x, scale, name, **kwargs)

    # shape = (batch, n_actions * n_heads)
    h = tf.keras.layers.Dense(n_actions * n_heads, name=_name("dense_out"))(h)

    if n_heads > 1:
        # shape = (batch, n_actions, n_heads)
        h = tf.reshape(h, [-1, n_actions, n_heads])

    return h


def dqn_atari_paper_net_dueling(x, n_actions, n_heads=1, scale=1, name=None, **kwargs):
    """
    Neural network based on:
    Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
    arXiv preprint arXiv:1312.5602 (2013).
    Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning."
    International conference on machine learning. PMLR, 2016.
    """

    _name = _get_name_fn(name)

    h = dqn_atari_paper_net_base(x, scale, name, **kwargs)

    h_val = tf.keras.layers.Dense(
        units=int(scale * 256), name=_name("dense_value_1"), **kwargs
    )(h)
    h_adv = tf.keras.layers.Dense(
        units=int(scale * 256), name=_name("dense_advantage_1"), **kwargs
    )(h)

    h_val = tf.nn.relu(h_val)
    h_adv = tf.nn.relu(h_adv)

    # shape = (batch, 1, n_heads)
    val = tf.keras.layers.Dense(n_heads, name=_name("dense_value_out"), **kwargs)(
        h_val
    )[:, None, :]

    # shape = (batch, n_actions * n_heads)
    adv_flat = tf.keras.layers.Dense(
        n_actions * n_heads, name=_name("dense_advantage_out"), **kwargs
    )(h_adv)

    # shape = (batch, n_actions, n_heads)
    adv = tf.reshape(adv_flat, [-1, n_actions, n_heads])

    # shape = (batch, 1, n_heads)
    adv_avg = tf.reduce_mean(adv, axis=-2, keepdims=True)

    # shape = (batch, n_actions, n_heads)
    adv_normed = adv - adv_avg

    # shape = (batch, n_actions, n_heads)
    return val + adv_normed
