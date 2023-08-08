import tensorflow as tf

def model(num_features, num_actions, history_depth, loss_function):
    model = tf.keras.Sequential([ #Will get more complicated as we learn more, but a good starting point
        tf.keras.layers.Dense(1024, activation='sigmoid', input_shape=(num_features,history_depth)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation = 'sigmoid'),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dense(128, activation = 'sigmoid'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(32, activation = 'sigmoid'),
        tf.keras.layers.Dense(num_actions, activation='tanh')
    ])
    model.summary()
    model.compile(loss=loss_function,optimizer = 'sgd', metrics=['accuracy'])
    return model