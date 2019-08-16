from enum import Enum

import numpy as np
import pandas as pd


# entity action set
class Action(Enum):
    NONE = 0
    MOVE_NORTH = 1
    MOVE_EAST = 2
    MOVE_SOUTH = 3
    MOVE_WEST = 4


NUM_ACTIONS = 5


def action(pos_diff):
    if pos_diff[0] == 0.0 and pos_diff[1] > 0.0:
        return Action.MOVE_NORTH
    elif pos_diff[0] == 0.0 and pos_diff[1] < 0.0:
        return Action.MOVE_SOUTH
    elif pos_diff[0] < 0.0 and pos_diff[1] == 0.0:
        return Action.MOVE_EAST
    elif pos_diff[0] > 0.0 and pos_diff[1] == 0.0:
        return Action.MOVE_WEST
    elif pos_diff[0] == 0.0 and pos_diff[1] == 0.0:
        return Action.NONE
    else:
        return np.nan


def featurize_action(action: Action):
    row = [0] * 5
    row[action.value] = 1
    return row


def action_feature_names():
    return [f'is_{action.name}' for action in Action]


def add_action_features(X: pd.DataFrame, y: pd.DataFrame):
    pos_diffs = X.groupby(['entity', 'epoch'])['e.stats.pos.x', 'e.stats.pos.y'].diff(-1).dropna()

    action_features = pos_diffs.apply(lambda row: action(row), axis=1)
    # currently don't support diagonal movement
    action_features = action_features.dropna()

    action_features = action_features.apply(lambda x: pd.Series(featurize_action(x)))
    action_features.columns = action_feature_names()

    X2 = X.join(action_features, how='inner')
    return X2, y


def action_from_feature(row: pd.Series) -> Action:
    idx = row[action_feature_names()].reset_index(drop=True).idxmax()
    return Action(idx)


def pos_from_action(pos, action):
    new_pos = pos.rename(lambda x: x + '_end')

    if action == Action.MOVE_NORTH:
        new_pos = new_pos + [0.0, 1.0]
    elif action == Action.MOVE_SOUTH:
        new_pos = new_pos + [0.0, -1.0]
    elif action == Action.MOVE_EAST:
        new_pos = new_pos + [1.0, 0.0]
    elif action == Action.MOVE_WEST:
        new_pos = new_pos + [-1.0, 0.0]

    return pd.DataFrame(pd.concat([pos, new_pos], axis=0)).T
