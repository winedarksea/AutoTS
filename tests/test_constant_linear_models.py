"""Constant targets must fit without artificial noise or batch-dependent failures."""

import unittest
import warnings

import numpy as np
import pandas as pd

from autots.models.basics import BasicLinearModel, TVVAR
from autots.tools.profile import ProfileWarning, profile_time_series, summarize_series

try:
    from sklearn.decomposition import PCA  # noqa: F401

    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False


class TestConstantLinearModels(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.get_state()
        self.index = pd.date_range('2023-01-01', periods=400)

    def tearDown(self):
        np.random.set_state(self.random_state)

    def constant_frame(self, value, include_peer=False):
        df = pd.DataFrame({'constant': value}, index=self.index)
        if include_peer:
            df['peer'] = 10 + np.sin(np.arange(len(df)) / 7)
        return df

    def test_basic_linear_constant_targets_are_deterministic(self):
        for value in (0.0, 5.0):
            for regularization in (2.0, None):
                expected = None
                for seed in (0, 7):
                    for include_peer in (False, True):
                        with self.subTest(
                            value=value,
                            regularization=regularization,
                            seed=seed,
                            include_peer=include_peer,
                        ):
                            np.random.seed(seed)
                            df = self.constant_frame(value, include_peer)
                            original = df.copy(deep=True)
                            model = BasicLinearModel(
                                lambda_=regularization,
                                holiday_countries_used=False,
                            ).fit(df)
                            forecast = model.predict(7).forecast['constant'].to_numpy()
                            self.assertTrue(np.isfinite(forecast).all())
                            if expected is None:
                                expected = forecast
                            np.testing.assert_allclose(forecast, expected, atol=1e-10)
                            if value == 0:
                                np.testing.assert_array_equal(forecast, 0)
                            pd.testing.assert_frame_equal(df, original)

    def assert_tvvar_constants(self, **parameters):
        for value in (0.0, 5.0):
            expected = None
            for seed in (0, 7):
                np.random.seed(seed)
                df = self.constant_frame(value)
                original = df.copy(deep=True)
                with warnings.catch_warnings():
                    # PCA's explained-variance ratio is undefined for zero-variance
                    # features; its transformed values and forecasts remain finite.
                    warnings.filterwarnings(
                        'ignore', 'invalid value encountered in divide', RuntimeWarning
                    )
                    model = TVVAR(
                        holiday_countries_used=False, **parameters
                    ).fit(df)
                    forecast = model.predict(7).forecast.to_numpy()
                self.assertTrue(np.isfinite(forecast).all())
                if expected is None:
                    expected = forecast
                np.testing.assert_allclose(forecast, expected, atol=1e-10)
                pd.testing.assert_frame_equal(df, original)

    def test_tvvar_solver_paths_support_constant_targets(self):
        for parameters in (
            {},
            {'lambda_': None},
            {'phi': 0.99},
            {'lambda_': None, 'phi': 0.99},
            {'base_scaled': False},
            {'base_scaled': False, 'lambda_': None},
        ):
            with self.subTest(parameters=parameters):
                self.assert_tvvar_constants(**parameters)

    @unittest.skipUnless(PCA_AVAILABLE, 'scikit-learn is optional')
    def test_tvvar_pca_supports_constant_targets(self):
        for phi in (None, 0.99):
            with self.subTest(phi=phi):
                self.assert_tvvar_constants(lags=[1, 2], apply_pca=True, phi=phi)

    def test_constant_profiling_avoids_decomposition_fallback(self):
        for value in (0.0, 5.0):
            for seed in (0, 7):
                for include_peer in (False, True):
                    with self.subTest(value=value, seed=seed, include_peer=include_peer):
                        np.random.seed(seed)
                        df = self.constant_frame(value, include_peer)
                        with warnings.catch_warnings(), np.errstate(
                            divide='ignore', invalid='ignore'
                        ):
                            warnings.simplefilter('error', ProfileWarning)
                            summary = summarize_series(df)
                            profiles = profile_time_series(df).set_index('SERIES')
                        self.assertIn('seasonality_contribution', summary.index)
                        self.assertEqual(profiles.loc['constant', 'PROFILE'], 'flat')
