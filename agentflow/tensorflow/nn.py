import tensorflow as tf


def dense_net(
    x,
    units,
    layers,
    batchnorm=False,
    dropout=0,
    activation=tf.nn.relu,
    name=None,
    **kwargs,
):

    assert isinstance(layers, int) and layers > 0, "layers should be a positive integer"
    assert isinstance(units, int) and units > 0, "units should be a positive integer"

    h = x
    for l in range(layers):
        layer_name = f"layer_{l}"
        if name is not None:
            layer_name = f"{name}/{layer_name}"

        h = tf.keras.layers.Dense(
            units, activation=activation, name=f"{layer_name}/dense", **kwargs
        )(h)

        if batchnorm:
            h = tf.keras.layers.BatchNormalization(name=f"{layer_name}/batchnorm")(h)

        if dropout > 0:
            h = tf.keras.layers.Dropout(dropout)(h)

    return h


def normalize_ema(input_tensor, ema_decay=0.999, name=None, axis=-1):
    bn = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=ema_decay,
        center=False,
        scale=False,
        name=f"{name}/batch_norm" if name else None,
    )
    return bn(input_tensor)
