import typing

import numpy as np
import pandas as pd
from plotnine import ggplot, scale_y_reverse, geom_text, theme_bw, aes, geom_tile, geom_segment, geom_rect, arrow
import shap
from tensorflow.python.client.session import Session
from tensorflow.python.keras.engine.sequential import Sequential
from toolz import pipe

from .feature_utils import X_from_state, VISION_DIMENSIONS, flat_to_2d_pos
from ..features.action_features import action_from_feature, pos_from_action, Action


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
        }, names=['who'], sort=True
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


def _format_diagnostic_rows(
    x_row: pd.Series,
    y_row: pd.Series,
    shap_values: typing.Optional[np.ndarray] = None,
):
    """
    Format observation for diagnostic display.
    """
    display_idx = [not c.startswith('o_') for c in x_row.index]
    display_row = x_row[display_idx]
    display_shap = pd.Series(shap_values[display_idx], index=display_row.index)

    return (
        pd.concat({'feature_value': display_row, 'shap': display_shap}, axis=1)
        .sort_values('shap', ascending=False),
        y_row.filter(regex='(entity|pc|R)')
    )


def diagnostic_plot(
    x_row: pd.Series,
    y_row: pd.Series,
    state: typing.Dict[str, typing.Any],
    idx: typing.Tuple[str, int, int, int],
    overlays: typing.List[typing.List[typing.List[str]]],
    shap_values: typing.Optional[np.ndarray] = None,
):
    """
    Plot observation for diagnostic display
    """
    plot_idx = [c.startswith('o_') for c in x_row.index]

    # Generate dfs for plotting
    action = action_from_feature(x_row)
    arrow_df = pos_from_action(x_row[['e.stats.pos.x', 'e.stats.pos.y']], action)

    vision_df = {
        'xmin': x_row['e.stats.pos.x'] - VISION_DIMENSIONS[0] / 2,
        'ymin': x_row['e.stats.pos.y'] - VISION_DIMENSIONS[1] / 2,
        'xmax': x_row['e.stats.pos.x'] + VISION_DIMENSIONS[0] / 2,
        'ymax': x_row['e.stats.pos.y'] + VISION_DIMENSIONS[1] / 2,
    }

    def _shap_to_row(idx, value):
        pos = flat_to_2d_pos(idx, VISION_DIMENSIONS)
        return {
            'x': vision_df['xmin'] + pos[0] + 0.5,
            'y': vision_df['ymin'] + pos[1] + 0.5,
            'value': value,
        }

    shap_df = pd.DataFrame(
        [_shap_to_row(i, val) for i, val in enumerate(shap_values[plot_idx])]
    )

    vision_df = pd.DataFrame([vision_df])

    return (
        plot_state(state, overlays, ['black'] * len(overlays))
        + geom_tile(data = shap_df,
                    mapping = aes(x='x', y='y', fill='value'),
                    alpha = 0.5)
        + geom_segment(data = arrow_df, mapping = aes(x = 'e.stats.pos.x',
                                                      y = 'e.stats.pos.y',
                                                      xend = 'e.stats.pos.x_end',
                                                      yend = 'e.stats.pos.y_end'),
                       arrow = arrow() if action is not Action.NONE else None,
                       color = 'purple',
                       size = 1,
                       alpha = 0.5)
        + geom_text(data = pd.DataFrame([x_row]).assign(entity=x_row.name[2]),
                    mapping = aes(x='e.stats.pos.x', y='e.stats.pos.y', label='entity'),
                    color = 'purple')
        + geom_rect(data = vision_df,
                    mapping = aes(xmin='xmin', ymin='ymin', xmax='xmax', ymax='ymax'),
                    fill=None,
                    color='purple',
                    linetype='dotted')
    )


class DiagnosticViz:
    """
        Visualize performance of flare ai model.
    """
    def __init__(
        self, background: pd.DataFrame, session: Session, model: Sequential, overlays: typing.List[typing.List[typing.List[str]]]
    ):
        self.model = model
        self.overlays = overlays
        self.explainer = shap.DeepExplainer(
            model=model,
            data=background,
            session=session,
        )

    def display(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        states: typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]],
        idx: typing.Tuple[str, int, int, int],
    ):
        x_row = X.loc[idx, :]
        test_obs = np.array([x_row], dtype=np.float32)
        y_row = y.loc[idx, :].append(
            pipe(
                test_obs,
                lambda obs: self.model.predict(obs, batch_size=1),
                lambda pred: {'R_pred': pred[0][0]},
                pd.Series
            )
        )
        shap_values = self.explainer.shap_values(test_obs)[0][0]

        return (
            _format_diagnostic_rows(x_row=x_row, y_row=y_row, shap_values=shap_values),
            diagnostic_plot(
                x_row=x_row,
                y_row=y_row,
                state=states[idx[0]][idx[1]],
                idx=idx,
                overlays=self.overlays,
                shap_values=shap_values
            )
        )
