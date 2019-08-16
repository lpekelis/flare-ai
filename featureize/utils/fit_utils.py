import tensorflow as tf
import tflearn


def define_model_tflearn(num_outcomes: int = 2, input_len: int = 108, model_type: str = 'dnn'):
    # Define neural network using tflearn
    if model_type == 'dnn':
        net = tflearn.input_data(shape=[None, input_len])
        net = tflearn.fully_connected(net, 64, activation='relu')
        net = tflearn.fully_connected(net, 32)
        net = tflearn.fully_connected(net, 32, activation='relu')
        net = tflearn.fully_connected(net, 16, activation='relu')
        net = tflearn.fully_connected(net, num_outcomes, activation='softmax')
    elif model_type == 'cnn':
        net = tflearn.input_data(shape=[None, 84, 84, input_len])
        net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu')
        net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
        net = tflearn.fully_connected(net, 256, activation='relu')
        net = tflearn.fully_connected(net, num_outcomes)

    net = tflearn.regression(net)
    # Define model
    return tflearn.DNN(net, tensorboard_verbose=3)


def define_prediction_model_keras(num_outcomes: int = 1):
    # Define predictive neural network using keras
    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_outcomes),
    ])

    model.compile(
        optimizer=tf.train.AdamOptimizer(0.01),
        loss='mae',
        metrics=['mse']
    )

    return model
