import tensorflow as tf


def dense_model(in_shape, num_outputs, name):
    inputs = tf.keras.layers.Input(shape=(in_shape,), name="observations")
    hidden_layer = tf.keras.layers.Dense(256, name="layer1", activation=tf.nn.relu)(
        inputs
    )
    out_layer = tf.keras.layers.Dense(num_outputs, name="out", activation=None)(
        hidden_layer
    )
    value_layer = tf.keras.layers.Dense(1, name="value", activation=None)(hidden_layer)
    return tf.keras.Model(inputs, [out_layer, value_layer], name=name)


def res_net_model(in_shape, num_outputs, name):
    """
    Simple network with residual connection  
    """
    # TODO
    pass
