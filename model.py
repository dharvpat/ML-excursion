import tensorflow as tf

def model(num_features, num_actions, history_depth, loss_function, optimizer):
    model = tf.keras.Sequential([ #Will get more complicated as we learn more, but a good starting point
        tf.keras.layers.Dense(2048, activation='tanh', input_shape=(history_depth,num_features)),
        #tf.keras.layers.Dense(8192, activation='sigmoid'),
        #tf.keras.layers.Dense(4096, activation='tanh'),
        #tf.keras.layers.Dense(2048, activation='sigmoid'),
        tf.keras.layers.Dense(1024, activation='sigmoid'),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(num_actions, activation='sigmoid')
    ])
    model.compile(loss=loss_function,optimizer = optimizer, metrics=['accuracy'])
    model.summary()
    return model