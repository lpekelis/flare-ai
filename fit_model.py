import os
from datetime import datetime

import numpy as np
import pandas as pd
import structlog

import tensorflow as tf
import tflearn

print("TensorFlow version: {}".format(tf.VERSION))

logger = structlog.getLogger(__name__)

GAME_STATE_DATA_DIR = '/Users/lpekelis/flare/flare-ai/data/'

GAME_STATE_FEATURES_FILE = 'game_states_features_20180611_002848.h5'

MODEL_WRITE_DIR = '/Users/lpekelis/flare/flare-ai/models/'

logger.info('game state data files', files=os.listdir(GAME_STATE_DATA_DIR))

# load data
game_state_store = pd.HDFStore(GAME_STATE_DATA_DIR + GAME_STATE_FEATURES_FILE)
print(game_state_store)
X = game_state_store.get('X')
y_entity_damage = game_state_store.get('y_entity_damage')

# old data loading
# f = h5py.File(GAME_STATE_DATA_DIR + GAME_STATE_FEATURES_FILE, 'r')
# X = pd.DataFrame(f['X'].value)
# y = pd.DataFrame(f['y'].value)
# y_entity_damage = pd.DataFrame(f['y_entity_damage'].value)

logger.info(X=X.shape, y_entity_damage=y_entity_damage.shape)

# Build neural network
net = tflearn.input_data(shape=[None, 108])
net = tflearn.fully_connected(net, 64, activation='relu')
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32, activation='relu')
net = tflearn.fully_connected(net, 16, activation='relu')
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(
    np.array(X, dtype=np.float32),
    np.array(y_entity_damage, dtype=np.float32),
    n_epoch=10,
    batch_size=16,
    show_metric=True)

now_time = datetime.now().strftime("%Y%m%d_%H%M%S")

model.save(MODEL_WRITE_DIR + 'median_time_to_damage_%s.tflearn' % now_time)
