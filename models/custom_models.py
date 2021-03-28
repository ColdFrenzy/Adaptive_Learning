import tensorflow as tf


def dense_model(in_shape, hidden_layer_shapes, num_outputs, name):
    x = None
    inputs = tf.keras.layers.Input(shape=(in_shape,), name="observations")
    for i,layer_shape in enumerate(hidden_layer_shapes):
        x = tf.keras.layers.Dense(
            layer_shape, name="dense_" + str(i), activation=tf.nn.relu
        )(x if x is not None else inputs)
    out_layer = tf.keras.layers.Dense(num_outputs, name="out", activation=None)(
        x
    )
    value_layer = tf.keras.layers.Dense(1, name="value", activation=None)(x)
    return tf.keras.Model(inputs, [out_layer, value_layer], name=name)


def res_net_model(in_shape, hidden_layer_shapes, num_outputs, name):
    """
    hidden_layer_shapes : list
        list with the shape of every hidden layer
    Simple neural network block with n_layers dense layers and a residual connection  
    """
    x = None
    inputs = tf.keras.layers.Input(shape=(in_shape,), name="observations")
    for i,layer_shape in enumerate(hidden_layer_shapes):
        x = tf.keras.layers.Dense(
            layer_shape, name="dense_"+str(i), activation=tf.nn.relu
        )(x if x is not None else inputs)
        x = tf.keras.layers.Dense(in_shape, name="dense_" + str(i) +".2", activation=tf.nn.relu)(
            x
        )
        x = tf.keras.layers.Add()([inputs, x])
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
    out_layer = tf.keras.layers.Dense(num_outputs, name="out", activation=None)(
        x
    )
    value_layer = tf.keras.layers.Dense(1, name="value", activation=None)(x)
    return tf.keras.Model(inputs, [out_layer, value_layer], name=name)


def conv_dense_model(in_shape, num_outputs, name):

    if len(in_shape) == 2:
        in_shape = in_shape + (1,)
    inputs = tf.keras.Input(shape=in_shape , name="observations")

    x = tf.keras.layers.Conv2D(64, 4, name="conv_1")(inputs)
    x = tf.keras.layers.Conv2D(64, 2, name="conv_2")(x)
    x = tf.keras.layers.Conv2D(64, 2, name="conv_3")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, name="dense_1",activation=tf.nn.relu)(x)
    out_layer = tf.keras.layers.Dense(num_outputs, name="out", activation=None)(x)
    value_layer = tf.keras.layers.Dense(1, name="value", activation=None)(x)
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
    # model = res_net_model(42, [256,128,64], 7, "res_model")
    # model = dense_model(42, [256,128,64], 7, "dense_block")
    # model.summary()
    model = conv_dense_model((7,6,1),7,"conv_dense_model")
    tf.keras.utils.plot_model(model, "conv_dense_model.png", True)
