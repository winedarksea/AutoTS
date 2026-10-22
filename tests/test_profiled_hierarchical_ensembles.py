"""Profile ensembles must resolve complete hierarchical series identifiers."""

import datetime
import unittest

import numpy as np
import pandas as pd

from autots.models.ensemble import HorizontalEnsemble, MosaicEnsemble


class TestProfiledHierarchicalEnsembles(unittest.TestCase):
    def test_profiled_handlers_select_forecasts_by_tuple_identifier(self):
        for names in (('group', 'id'), (None, None), ('SERIES', 'id')):
            columns = pd.MultiIndex.from_tuples(
                [('b', 'shared'), ('a', 'shared')], names=names
            )
            df = pd.DataFrame(
                {'flat': np.ones(400), 'binary': np.arange(400) % 2},
                index=pd.date_range('2023-01-01', periods=400),
            )
            df.columns = columns
            forecast_index = pd.date_range(
                df.index[-1] + pd.Timedelta(days=1), periods=3
            )
            forecasts = {
                'flat_model': pd.DataFrame(
                    [[10.0, 11.0]] * 3, index=forecast_index, columns=columns
                ),
                'overall_model': pd.DataFrame(
                    [[20.0, 21.0]] * 3, index=forecast_index, columns=columns
                ),
            }
            for kind, handler in (
                ('Horizontal', HorizontalEnsemble),
                ('Mosaic', MosaicEnsemble),
            ):
                with self.subTest(names=names, kind=kind):
                    mapping = {'flat': 'flat_model', 'overall': 'overall_model'}
                    if kind == 'Mosaic':
                        mapping = {
                            profile: {str(period): model for period in range(3)}
                            for profile, model in mapping.items()
                        }
                    result = handler(
                        ensemble_params={
                            'model_name': kind,
                            'model_metric': kind.lower() + '-profile',
                            'series': mapping,
                            'models': {key: {'Model': key} for key in forecasts},
                        },
                        forecasts_list=list(forecasts),
                        forecasts=forecasts,
                        lower_forecasts={
                            key: frame - 1 for key, frame in forecasts.items()
                        },
                        upper_forecasts={
                            key: frame + 1 for key, frame in forecasts.items()
                        },
                        forecasts_runtime={
                            key: datetime.timedelta(0) for key in forecasts
                        },
                        prediction_interval=0.9,
                        df_train=df,
                    )
                    expected = pd.DataFrame(
                        [[10.0, 21.0]] * 3, index=forecast_index, columns=columns
                    )
                    pd.testing.assert_frame_equal(result.forecast, expected)
                    pd.testing.assert_frame_equal(result.lower_forecast, expected - 1)
                    pd.testing.assert_frame_equal(result.upper_forecast, expected + 1)
                    plot_data = result.plot_df(df_wide=df, series=columns[0])
                    pd.testing.assert_series_equal(
                        plot_data['forecast'].reindex(forecast_index),
                        expected[columns[0]],
                        check_names=False,
                    )
