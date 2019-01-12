import json
import os
import pickle
import re
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import structlog
import tqdm

# TODO(LP): refactor to using methods in featurize_utils

logger = structlog.getLogger(__name__)

GAME_STATE_DIR = '/Users/lpekelis/flare/flare-ai/log/'
MAP_PATH = '/Users/lpekelis/flare/flare-game/mods/flare_ai/maps/testing_grounds.txt'

FEATURES_WRITE_DIR = '/Users/lpekelis/flare/flare-ai/data/'

# added as map overlay across enemies
FEATURES_TO_MAP = ['stats.hp']

# added for each enemy directly
FEATURES_SELF = ['stats.hp', 'stats.mp', 'stats.pos.x', 'stats.pos.y']

# player character features, distance pc to enemy always added
FEATURES_PC = ['stats.hp', 'stats.mp']

# predict median time to difference for these outcomes
DIFF_Y = ['entity_hp', 'pc_hp']

# entities can featurize all objects within a square region of their position
VISION_DIMENSIONS = [10, 10]

logger.info('game state files', files=os.listdir(GAME_STATE_DIR))

game_states = []
for fname in os.listdir(GAME_STATE_DIR):
    if 'GAME_STATES' in fname and (len(sys.argv) == 1 or fname in sys.argv):
        logger.info('add game state file', file=fname)
        with open(GAME_STATE_DIR + fname) as f:
            for line in f:
                game_states.append(json.loads(line))

logger.info('loaded game states: ', num_states=len(game_states))

logger.info('game state schema:', state=game_states[0])

map_dimensions = [100, 100]
collision_layer = []
is_collision_layer = False

with open(MAP_PATH) as f:
    for line in f:
        if (('[layer]' in line) and ('type=collision' in next(f, ''))):
            line = next(f, '')
            line = next(f, '')
            is_collision_layer = True
        if is_collision_layer:
            if line == '\n':
                is_collision_layer = False
            else:
                collision_layer.append(line.split(','))

collision_layer_flat = [item for sublist in collision_layer for item in sublist]


def num_enemies(state):
    n = 0
    for key in state.keys():
        match = re.search('^e(\d+)', key)
        if match:
            n = max(n, int(match.group(1)))
    return n


def dist_entities(e1, e2, state):
    # Example:
    # dist_entities('e0','pc',state)
    # signed distance assumes origin at upper left corner
    return (
        float(state[e2 + '->stats.pos.x']) - float(state[e1 + '->stats.pos.x']),
        float(state[e2 + '->stats.pos.y']) - float(state[e1 + '->stats.pos.y'])
    )


def flat_pos(x, y, map_dimensions):
    # Return position in flattened vector from 2d map
    return int(max(min(np.floor(y) * map_dimensions[0] + np.floor(x), np.prod(map_dimensions) - 1), 0))


def add_feature_to_relative_overlay(e1, e2, feature_name, state, overlay, overlay_dimensions):
    # overlay is assumed relative to e1
    # e.g. a 10x10 grid centered at e1 location

    dx, dy = dist_entities(e1, e2, state)

    pos = np.array(overlay_dimensions)/2 + np.array([dx, dy])

    if np.logical_and(pos >= 0, pos <= overlay_dimensions).all():
        o_idx = flat_pos(pos[0], pos[1], overlay_dimensions)
        overlay[o_idx] = state['%s->%s' % (e2, feature_name)]

    # for debugging
    return pos


def X_from_state(state, n_e=None):
    if not n_e:
        n_e = num_enemies(state)

    X = {}
    for i in range(0, n_e+1):
        row = []
        for f in FEATURES_SELF:
            row.append(state['e%d->%s' % (i, f)])

        for f in FEATURES_PC:
            row.append(state['pc->%s' % f])

        # distance to pc always added
        dx, dy = dist_entities('e%d' % i, 'pc', state)
        row = row + [dx, dy]

        # TODO(Leo): add overlay features

        # allies hp
        allies_hp_overlay = [0] * (VISION_DIMENSIONS[0] * VISION_DIMENSIONS[1])

        for j in range(0, n_e+1):
            if j is not i:
                add_feature_to_relative_overlay('e%d' % i,
                                                'e%d' % j,
                                                'stats.hp',
                                                state,
                                                allies_hp_overlay,
                                                VISION_DIMENSIONS)
        row = row + allies_hp_overlay

        # add copy of row to data matrix
        X[i] = list(row)

    X = pd.DataFrame.from_dict(X, orient='index').rename_axis('entity').apply(pd.to_numeric)

    X.columns = (
        ['e.' + f for f in FEATURES_SELF]
        + ['pc.' + f for f in FEATURES_PC]
        + ['e.pc.dx', 'e.pc.dy']
        + ['o_hp_%i' % i for i in range(np.prod(VISION_DIMENSIONS))]
    )

    return X


def y_from_state(state, n_e=None):
    if not n_e:
        n_e = num_enemies(state)

    y = {}
    for i in range(0, n_e+1):
        row = [
            state['e%d->stats.alive' % i],
            state['e%d->stats.hp' % i],
            state['pc->stats.hp']
        ]

        y[i] = row

    # TODO(Leo): set index name
    y = pd.DataFrame.from_dict(y, orient='index').rename_axis('entity').apply(pd.to_numeric)

    y.columns = [
        'is_entity_alive',
        'entity_hp',
        'pc_hp'
    ]

    return y


def time_to_diff(df, diff_col):
    return (
        df
        .assign(is_diff=lambda df: df[diff_col] != 0.0)
        .groupby('entity')
        .apply(
            lambda df: (
                df
                .assign(is_diff_cum_sum=lambda df: np.cumsum(df.is_diff))
                # last diff period assumed incomplete
                .assign(
                    time_filtered=lambda df:
                        np.where(df.is_diff_cum_sum < max(df.is_diff_cum_sum), df.time, np.nan)
                )
            )
        )
        .groupby(['entity', 'is_diff_cum_sum'])
        ['time_filtered']
        .transform(lambda x: max(x) - x)
        # pandas does something weird here
        .reset_index()
        ['time_filtered']
    )


def yX_from_data(data):
    # data - list of dicts

    y = {}
    X = {}
    for i, state in tqdm.tqdm(enumerate(data), total=len(data)):
        n_e = num_enemies(state)
        y[i] = y_from_state(state, n_e)
        X[i] = X_from_state(state, n_e)

    y = pd.concat(y, names=['time'])
    X = pd.concat(X, names=['time'])

    y2 = (
        y
        .join(
            y.groupby('entity')[DIFF_Y].diff().fillna(0),
            rsuffix='_diff'
        )
        .reset_index()
    )

    for f in DIFF_Y:
        y2['time_to_' + f + '_diff'] = time_to_diff(y2, f + '_diff')
        y2['time_above_median_' + f + '_diff'] = 1*(
            y2['time_to_' + f + '_diff'] > np.nanmedian(y2['time_to_' + f + '_diff'])
        )

    y2 = y2.set_index(['time', 'entity']).dropna()
    X = X.dropna()

    y2 = y2[y2.index.isin(X.index)]
    X = X[X.index.isin(y2.index)]

    return y2, X


start_time = time.time()
y, X = yX_from_data(game_states)

logger.info(
    'Featurized y and X from game states.',
    min_elapsed=((time.time() - start_time) / 60),
    y_dim=y.shape,
    X_dim=X.shape,
    y0=y.head(1),
    X0=X.head(1)
)

y_entity_damage = y.assign(
    time_above_median_entity_hp_diff_1=lambda df: 1*(df.time_above_median_entity_hp_diff == 1),
    time_above_median_entity_hp_diff_0=lambda df: 1*(df.time_above_median_entity_hp_diff == 0)
)[['time_above_median_entity_hp_diff_1', 'time_above_median_entity_hp_diff_0']]

now_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# h5f = h5py.File(FEATURES_WRITE_DIR + 'game_states_features_%s.h5' % now_time, 'w')
# h5f.create_dataset('X', data=X)
# h5f.create_dataset('y', data=y)
# h5f.create_dataset('y_entity_damage', data=y_entity_damage)
# h5f.close()

store = pd.HDFStore(FEATURES_WRITE_DIR + 'game_states_features_%s.h5' % now_time)
store.append('X', X)
store.append('y', y)
store.append('y_entity_damage', y_entity_damage)
store.close()

pickle.dump(game_states, open(FEATURES_WRITE_DIR + 'game_states_%s.p' % now_time, 'wb'))
