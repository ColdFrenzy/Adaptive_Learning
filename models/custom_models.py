import tensorflow as tf


def dense_model(in_shape, hidden_shape, num_outputs, name):
    inputs = tf.keras.layers.Input(shape=(in_shape,), name="observations")
    hidden_layer = tf.keras.layers.Dense(
        hidden_shape, name="layer1", activation=tf.nn.relu
    )(inputs)
    out_layer = tf.keras.layers.Dense(num_outputs, name="out", activation=None)(
        hidden_layer
    )
    value_layer = tf.keras.layers.Dense(1, name="value", activation=None)(hidden_layer)
    return tf.keras.Model(inputs, [out_layer, value_layer], name=name)


def res_net_block(in_shape, hidden_shape, num_outputs, name):
    """
    Simple neural network block with 2 dense layers and a residual connection  
    """
    # TODO
    inputs = tf.keras.layers.Input(shape=(in_shape,), name="observations")
    dense_1 = tf.keras.layers.Dense(
        hidden_shape, name="dense_1", activation=tf.nn.relu
    )(inputs)
    dense_2 = tf.keras.layers.Dense(in_shape, name="dense_2", activation=tf.nn.relu)(
        dense_1
    )
    res_layer = tf.keras.layers.Add()([inputs, dense_2])
    res_layer = tf.keras.layers.ReLU()(res_layer)
    res_layer = tf.keras.layers.BatchNormalization()(res_layer)
    out_layer = tf.keras.layers.Dense(num_outputs, name="out", activation=None)(
        res_layer
    )
    value_layer = tf.keras.layers.Dense(1, name="value", activation=None)(res_layer)
    return tf.keras.Model(inputs, [out_layer, value_layer], name=name)


def dense_q_model(in_shape, hidden_shape, num_outputs, name):
    inputs = tf.keras.layers.Input(shape=(in_shape,), name="observations")
    hidden_layer = tf.keras.layers.Dense(
        hidden_shape, name="layer1", activation=tf.nn.relu
    )(inputs)
    out_layer = tf.keras.layers.Dense(num_outputs, name="out", activation=None)(
        hidden_layer
    )
    return tf.keras.Model(inputs, out_layer, name=name)


if __name__ == "__main__":
    model = res_net_block(42, 256, 7, "res_block")
    # model.summary()
    tf.keras.utils.plot_model(model, "res_model.png", True)
