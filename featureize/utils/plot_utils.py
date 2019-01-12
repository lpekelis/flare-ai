import pandas as pd
from plotnine import ggplot, scale_y_reverse, geom_text, theme_bw, aes

from .featurize_utils import X_from_state


def layer_to_df(layer):
    df = []
    for i, row in enumerate(layer):
        for j, cell in enumerate(row):
            if cell not in ['0', '\n']:
                df.append({'pos.x': i, 'pos.y': j, 'value': cell})
    return pd.DataFrame.from_records(df)


def plot_state(state, layers, colors):
    pos_df = pd.concat({
        'enemy':
            X_from_state(state)
            .reset_index()
            [['entity', 'e.stats.pos.x', 'e.stats.pos.y']],
        'player': pd.DataFrame.from_records({
                    'entity': ['pc'],
                    'e.stats.pos.x': [float(state['pc->stats.pos.x'])],
                    'e.stats.pos.y': [float(state['pc->stats.pos.y'])]
                })
        }, names=['who']
    ).reset_index()

    p = (
        ggplot(pos_df)
        + scale_y_reverse()
        + geom_text(aes(x='e.stats.pos.x', y='e.stats.pos.y', label='entity', color='who'))
        + theme_bw()
    )

    for i, layer in enumerate(layers):
        p = p + geom_text(
            aes(x='pos.x', y='pos.y', label='value'),
            data=layer_to_df(layer),
            color=colors[i]
        )

    return p
