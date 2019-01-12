import numpy as np
import pandas as pd
import tqdm

R_coefs = {
    'is_entity_alive_diff': 100,
    'is_pc_alive_diff': -25,
    'entity_hp_diff': 1,
    'pc_hp_diff': -0.25,
}

Y_DIFF_COLS = [
    'entity_hp',
    'pc_hp',
    'is_entity_alive',
    'is_pc_alive',
]

# discount rate for future rewards
GAMMA = 0.8

# aggregate discounted reward over this many periods
MEMORY_LENGTH = 100


def calc_R(y):
    R = 0
    for k, v in R_coefs.items():
        R = R + v * y[k]
    return R


def R_discount(s: float, y: pd.DataFrame, gamma: float, window_length: float = None):
    y = y.query('time > @s')
    if window_length:
        s_max = s + window_length
        y = y.query('time <= @s_max')
    return (
        y
        .assign(discount=lambda df: np.power(gamma, df['time'] - s - 1))
        .assign(R_discount=lambda df: df['discount'] * df['R'])
        .groupby(['entity', 'epoch'])
        [['R_discount']]
        .sum()
        .assign(time=s)
    )


def add_discounted_R(X, y):
    y = y.apply(pd.to_numeric)

    y_diff = y[Y_DIFF_COLS].groupby(['entity', 'epoch']).diff().fillna(0)

    y2 = y.join(y_diff, rsuffix='_diff')
    y2['R'] = calc_R(y2)

    time_bounds = y2.reset_index()['time'].describe()[['min', 'max']]

    y2_reset_index = y2.reset_index()
    R_discount_series = []
    for s in tqdm.tqdm(range(int(time_bounds['min']), int(time_bounds['max']-1))):
        R_discount_series.append(R_discount(s, y2_reset_index, GAMMA, MEMORY_LENGTH))
    R_discount_series = pd.concat(R_discount_series)

    y2 = y2.join(R_discount_series.reset_index().set_index(['time', 'entity', 'epoch']), how='inner')

    return X, y2
