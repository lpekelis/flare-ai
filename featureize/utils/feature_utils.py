import re
import sys
import typing

import numpy as np
import pandas as pd
import structlog
import tqdm

sys.path.append("..")
from ..features.action_features import add_action_features
from ..features.r_features import add_discounted_R

logger = structlog.getLogger(__name__)

FLARE_DIR = '/Users/leopekelis/flare/'
GAME_STATE_DIR = FLARE_DIR + 'flare-ai/log/'
MAP_PATH = FLARE_DIR + 'flare-game/mods/flare_ai/maps/testing_grounds.txt'

FEATURES_WRITE_DIR = FLARE_DIR + 'flare-ai/data/'

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


def num_enemies(state):
    n = 0
    for key in state.keys():
        match = re.search('^e(\d+)', key)
        if match:
            n = max(n, int(match.group(1))+1)
    return n


def dist_entities(e1, e2, state):
    # Example:
    # dist_entities('e0','pc',state)
    # signed distance assumes origin at upper left corner of geography
    return (
        float(state[e2 + '->stats.pos.x']) - float(state[e1 + '->stats.pos.x']),
        float(state[e2 + '->stats.pos.y']) - float(state[e1 + '->stats.pos.y'])
    )


def flat_pos(x, y, map_dimensions):
    # Return position in flattened vector from 2d map
    return int(max(min(np.floor(y) * map_dimensions[0] + np.floor(x), np.prod(map_dimensions)-1), 0))


def flat_to_2d_pos(flat_pos: int, map_dimensions: typing.List[int]):
    pos = np.divmod(flat_pos, map_dimensions[0])
    return (np.clip(pos[1], 0, map_dimensions[0]),
            np.clip(pos[0], 0, map_dimensions[1]))


def is_alive(hp):
    hp = float(hp)
    return hp if np.isnan(hp) else 1.0 * (hp > 0)


def companion_feature(e1, e2, feature_name, vision_dimension, state):
    # compute feature of companion e2, from perspective of e1
    # if e2 is not within vision_dimension of e1, return nan

    dx, dy = dist_entities(e1, e2, state)

    pos = np.array(vision_dimension)/2 + np.array([dx, dy])

    if np.logical_and(pos >= 0, pos <= vision_dimension).all():
        val = state['%s->%s' % (e2, feature_name)]
    else:
        val = np.nan
    return pos, val


def add_feature_to_relative_overlay(e1, e2, feature_name, state, overlay, overlay_dimensions):
    # overlay is assumed relative to e1
    # e.g. a 10x10 grid centered at e1 location

    pos, val = companion_feature(e1, e2, feature_name, overlay_dimensions, state)

    if val is not np.nan:
        o_idx = flat_pos(pos[0], pos[1], overlay_dimensions)
        overlay[o_idx] = val
    # for debugging
    return pos


def X_from_state(state, n_e=None):
    if not n_e:
        n_e = num_enemies(state)

    X = {}
    for i in range(0, n_e):
        row = []
        for f in FEATURES_SELF:
            row.append(state['e%d->%s' % (i, f)])

        for f in FEATURES_PC:
            row.append(state['pc->%s' % f])

        # distance to pc always added
        dx, dy = dist_entities('e%d' % i, 'pc', state)
        row = row + [dx, dy]

        # allies hp overlay
        allies_hp_overlay = [0] * (VISION_DIMENSIONS[0] * VISION_DIMENSIONS[1])

        for j in range(0, n_e):
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
    for i in range(0, n_e):
        row = [
            state['e%d->stats.alive' % i],
            state['e%d->stats.hp' % i],
            is_alive(state['pc->stats.hp']),
            state['pc->stats.hp']
        ]

        for j in range(0, n_e):
            if j is not i:
                pos, val_hp = companion_feature('e%d' % i,
                                                'e%d' % j,
                                                'stats.hp',
                                                VISION_DIMENSIONS,
                                                state
                                                )

                row = row + [is_alive(val_hp), val_hp]

        y[i] = row

    y = pd.DataFrame.from_dict(y, orient='index').rename_axis('entity').apply(pd.to_numeric)

    y_columns = [
        'is_entity_alive',
        'entity_hp',
        'is_pc_alive',
        'pc_hp'
    ]

    for i in range(0, n_e-1):
        y_columns = y_columns + ['is_companion_%d_alive' % i, 'companion_%d_hp' % i]

    y.columns = y_columns

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


def Xy_from_data(data, type='mdp'):
    # data - list of dicts

    y = {}
    X = {}
    for i, state in tqdm.tqdm(enumerate(data), total=len(data)):
        n_e = num_enemies(state)
        y[i] = y_from_state(state, n_e)
        X[i] = X_from_state(state, n_e)

    y = pd.concat(y, names=['time'])
    X = pd.concat(X, names=['time'])

    if type == 'mdp':
        return mdp_features(X, y)
    else:
        return time_to_diff_features(X, y)


def mdp_features(X, y):

    logger.info('Indexing by epoch...')
    X, y = index_by_epoch(X, y)

    logger.info('Filtering stagnant deaths...')
    X, y = filter_stagnant_deaths(X, y)

    logger.info('Adding action features...')
    X, y = add_action_features(X, y)
    X, y = remove_duplicates_and_align(X, y)

    logger.info('Calculating discounted reward score...')
    X, y = add_discounted_R(X, y)
    X, y = remove_duplicates_and_align(X, y)

    return X, y


def time_to_diff_features(X, y):
    y2 = (
        y
        .join(
            y.groupby('entity')[DIFF_Y].diff().fillna(0),
            rsuffix='_diff'
        )
        .reset_index()
    )

    for f in tqdm.tqdm(DIFF_Y, total=len(DIFF_Y)):
        y2[f'time_to_{f}_diff'] = time_to_diff(y2, f + '_diff')
        y2[f'time_above_median_{f}_diff'] = 1*(
            y2[f'time_to_{f}_diff'] > np.nanmedian(y2[f'time_to_{f}_diff'])
        )
        y2[f'time_to_{f}_diff_pctile_10'] = pd.cut(y2[f'time_to_{f}_diff'], bins=10, labels=False)
        y2[f'time_to_{f}_diff_pctile_100'] = pd.cut(y2[f'time_to_{f}_diff'], bins=100, labels=False)

    y2 = y2.set_index(['time', 'entity']).dropna()
    X = X.dropna()

    y2 = y2[y2.index.isin(X.index)]
    X = X[X.index.isin(y2.index)]

    return X, y2


def load_collision_layer():
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

    return collision_layer


def remove_duplicates_and_align(X, y):
    X = X.loc[~X.index.duplicated(keep='last'), :]
    y = y.loc[~y.index.duplicated(keep='last'), :]

    y = y.loc[y.index.isin(X.index), :]
    X = X.loc[X.index.isin(y.index), :]

    return X, y


def index_by_epoch(X, y):
    # Each time a player dies, entities are reset
    # Delineate data by player life for accurate reward calculation
    epoch = (
        y
        .groupby('entity')['is_pc_alive']
        .diff(1)
        .fillna(0)
        .clip(lower=0)
        .groupby('entity')
        .cumsum()
        .rename('epoch')
    )

    y2 = y.join(epoch, how='inner').set_index('epoch', append=True)

    X2 = X.join(epoch, how='inner').set_index('epoch', append=True)

    return X2, y2


def filter_stagnant_deaths(X, y):
    # Remove all event rows for each entity after they have died or the player
    # died within each epoch

    dead_filter = (
        y
        .groupby(['entity', 'epoch'])['is_entity_alive', 'is_pc_alive']
        .transform(lambda x: np.cumsum(1.0 - x))
        .query('is_entity_alive <= 1.0')
        .query('is_pc_alive <= 1.0')
    )

    y2 = y.join(dead_filter, how='inner', rsuffix='_dead_counter')

    return remove_duplicates_and_align(X, y2)


def pctile_assignments(y):
    outcome_vars = [c for c in y.columns if 'pctile' in c or 'median' in c]

    y_pctile_assignments = {}
    for outcome_var in outcome_vars:
        num_outcomes = len(np.unique(y[outcome_var]))

        y_pctile_assignments.update({
            f'{outcome_var}_{i}': lambda df, i=i: 1*(df[outcome_var] == i)
            for i in range(num_outcomes)
        })

    return y.assign(**y_pctile_assignments)
