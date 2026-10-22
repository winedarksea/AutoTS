"""Basic changepoint ramps must stay anchored within short histories."""

import unittest
import warnings

import numpy as np
import pandas as pd

from autots.models.basics import BasicLinearModel
from autots.tools.changepoints import (
    changepoint_fcst_from_last_row,
    create_changepoint_features,
)
from autots.tools.profile import ProfileWarning, summarize_series


class TestBasicChangepointBoundaries(unittest.TestCase):
    def test_short_histories_have_a_start_anchored_ramp(self):
        for rows in (1, 2, 30, 179, 180):
            with self.subTest(rows=rows):
                index = pd.date_range('2024-01-01', periods=rows)
                features = create_changepoint_features(
                    index, changepoint_spacing=60, changepoint_distance_end=180
                )
                self.assertEqual(features.columns.tolist(), ['changepoint_1'])
                np.testing.assert_array_equal(features['changepoint_1'], np.arange(rows))
                pd.testing.assert_index_equal(features.index, index)
                future = changepoint_fcst_from_last_row(features.iloc[-1], 3)
                np.testing.assert_array_equal(
                    future['changepoint_1'], np.arange(rows, rows + 3)
                )

    def test_terminal_changepoint_at_end_stays_in_bounds(self):
        index = pd.date_range('2024-01-01', periods=30)
        features = create_changepoint_features(
            index, changepoint_spacing=10, changepoint_distance_end=0
        )
        expected = np.maximum(0, np.arange(30)[:, None] - [0, 10, 20, 29])
        np.testing.assert_array_equal(features, expected)

    def test_ordinary_changepoint_features_are_unchanged(self):
        index = pd.date_range('2024-01-01', periods=300)
        features = create_changepoint_features(
            index, changepoint_spacing=60, changepoint_distance_end=180
        )
        expected = np.maximum(0, np.arange(300)[:, None] - [0, 60, 120])
        np.testing.assert_array_equal(features, expected)

    def test_short_linear_forecast_continues_the_observed_ramp(self):
        df = pd.DataFrame(
            {'series': np.arange(1.0, 31.0)},
            index=pd.date_range('2024-01-01', periods=30),
        )
        # The default Fourier design has more columns than observations here;
        # this regression checks point forecasts, independently of interval sizing.
        with np.errstate(invalid='ignore'):
            model = BasicLinearModel(holiday_countries_used=False).fit(df)
        forecast = model.predict(3, just_point_forecast=True)
        np.testing.assert_allclose(forecast['series'], [31, 32, 33], rtol=0, atol=0.25)

    def test_short_profiler_decomposition_has_no_negative_hinge(self):
        df = pd.DataFrame(
            {'series': np.arange(1.0, 31.0)},
            index=pd.date_range('2024-01-01', periods=30),
        )
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.simplefilter('error', ProfileWarning)
            summary = summarize_series(df)
            model = BasicLinearModel(
                changepoint_spacing=None, holiday_countries_used=False
            ).fit(df)
        self.assertTrue(np.isfinite(summary.loc['season_trend_percent']).all())
        np.testing.assert_array_equal(
            model.create_x(df, holiday_countries_used=False)['changepoint_1'],
            np.arange(30),
        )
