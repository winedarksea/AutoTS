"""Profiler numerical equivalence, identity, and failure reporting regressions."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from autots import AutoTS
from autots.tools.profile import ProfileWarning, profile_time_series, summarize_series


def legacy_autocorrelation(df):
    with np.errstate(divide='ignore', invalid='ignore'):
        correlations = np.corrcoef(df.bfill().T, df.shift(1).bfill().T)
    return np.diag(correlations[: df.shape[1], df.shape[1] :])


class TestProfile(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(17)
        self.df = pd.DataFrame(
            100 + random.normal(size=(400, 8)).cumsum(axis=0),
            index=pd.date_range('2023-01-01', periods=400),
            columns=list('hgfedcba'),
        )

    def assert_legacy_autocorrelation(self, df):
        original = df.copy(deep=True)
        expected = legacy_autocorrelation(df)
        actual = summarize_series(df).loc['autocorr_1'].to_numpy(dtype=float)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        pd.testing.assert_frame_equal(df, original)

    def test_autocorrelation_numeric_types(self):
        for dtype in ('float64', 'float32', 'int64'):
            with self.subTest(dtype=dtype):
                self.assert_legacy_autocorrelation(self.df.astype(dtype))

    def test_autocorrelation_missing_and_constant_values(self):
        df = self.df.copy()
        df.iloc[:80, 0] = np.nan
        df.iloc[-60:, 1] = np.nan
        df.iloc[100:130, 2] = np.nan
        df.iloc[-1, 3] = np.nan
        df['constant'] = 7.0
        df['zero'] = 0.0
        df['all_nan'] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            self.assert_legacy_autocorrelation(df)

    def test_profiles_match_legacy_autocorrelation(self):
        expected = profile_time_series(self.df)
        with patch(
            'autots.tools.profile._lag_one_autocorrelation',
            side_effect=legacy_autocorrelation,
        ):
            actual = profile_time_series(self.df)
        pd.testing.assert_frame_equal(actual, expected)

    def test_named_columns_preserve_identity_and_input(self):
        expected = profile_time_series(self.df)
        for name in ('series_id', 'PROFILE', 'SERIES'):
            df = self.df.rename_axis(name, axis='columns')
            original = df.copy(deep=True)
            with self.subTest(name=name):
                actual = profile_time_series(df)
                pd.testing.assert_frame_equal(actual, expected)
                self.assertEqual(actual['SERIES'].tolist(), df.columns.tolist())
                pd.testing.assert_frame_equal(df, original)

    def test_fit_data_retains_named_pivot_profiles(self):
        df = self.df.rename_axis('series_id', axis='columns')
        original = df.copy(deep=True)
        model = AutoTS(
            forecast_length=7,
            model_list=['LastValueNaive'],
            ensemble=None,
            max_generations=0,
            num_validations=0,
            verbose=-1,
        )
        model.fit_data(df)
        expected = profile_time_series(model.df_wide_numeric).set_index('SERIES')
        self.assertEqual(model.series_profiles, expected['PROFILE'].to_dict())
        self.assertEqual(set(model.series_profiles), set(df.columns))
        pd.testing.assert_frame_equal(df, original)

    def test_autocorrelation_failure_warns_and_omits_metric(self):
        with patch(
            'autots.tools.profile._lag_one_autocorrelation',
            side_effect=ValueError('autocorrelation unavailable'),
        ), self.assertWarnsRegex(ProfileWarning, 'autocorr_1.*ValueError'):
            summary = summarize_series(self.df)
        self.assertNotIn('autocorr_1', summary.index)
        self.assertIn('season_trend_percent', summary.index)

    def test_decomposition_failure_warns_and_preserves_fallbacks(self):
        with patch(
            'autots.tools.profile.BasicLinearModel.fit',
            side_effect=ValueError('decomposition unavailable'),
        ), self.assertWarnsRegex(ProfileWarning, 'decomposition.*ValueError'):
            summary = summarize_series(self.df)
        np.testing.assert_array_equal(summary.loc['season_trend_percent'], 0)
        pd.testing.assert_series_equal(
            summary.loc['deseasonalized_mean'],
            summary.loc['mean'],
            check_names=False,
        )
        expected_cv = (self.df.std() / self.df.abs().mean().replace(0, 1)) ** 2
        np.testing.assert_allclose(summary.loc['cv_squared'], expected_cv)

    def test_empty_series_labels_remain_unchanged(self):
        df = self.df.iloc[:, :1].copy()
        df['zero'] = 0.0
        df['all_nan'] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            profiles = profile_time_series(df).set_index('SERIES')['PROFILE']
            summary = summarize_series(df)
        self.assertEqual(profiles['zero'], 'flat')
        self.assertEqual(profiles['all_nan'], 'smooth')
        np.testing.assert_array_equal(
            summary.loc['null_percentage', ['zero', 'all_nan']], 0
        )
