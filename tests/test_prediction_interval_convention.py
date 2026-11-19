"""prediction_interval is central: 0.8 puts lower_forecast at q0.1 and upper_forecast at q0.9.

One check per interval mechanism (historic_quantile, normal z-scores, neighbour/seed
quantiles) plus SPL, which must score the bounds at the same central quantiles.
"""

import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd

from autots.evaluator.metrics import default_scaler
from autots.models import basics
from autots.models.base import PredictionObject
from autots.models.basics import (
    AverageValueNaive,
    BasicLinearModel,
    KalmanStateSpace,
    NVAR,
    SectionalMotif,
)
from autots.tools.probabilistic import historic_quantile, inferred_normal

try:
    from scipy.stats import norm
except Exception:  # pragma: no cover
    norm = None


def gaussian_noise_frame(n_rows=400, n_series=4, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n_rows, freq="D")
    values = 20.0 + rng.standard_normal((n_rows, n_series))
    return pd.DataFrame(values, index=index, columns=[f"s{i}" for i in range(n_series)])


def half_widths(prediction):
    upper = (prediction.upper_forecast - prediction.forecast).to_numpy()
    lower = (prediction.forecast - prediction.lower_forecast).to_numpy()
    return upper, lower


class TestHistoricQuantile(unittest.TestCase):
    def test_uses_central_quantiles(self):
        # 0..100 has q0.1=10, median=50, q0.9=90, so both distances are exactly 40
        train = pd.DataFrame({"a": np.arange(101, dtype=float)})
        lower, upper = historic_quantile(train, prediction_interval=0.8)
        np.testing.assert_allclose(lower, [40.0])
        np.testing.assert_allclose(upper, [40.0])

    def test_zero_width_falls_back_to_range(self):
        # mostly-constant series: q0.1, median and q0.9 tie, so the width must come
        # from the extremes instead of collapsing to zero
        values = np.full(100, 5.0)
        values[0], values[-1] = 1.0, 13.0
        lower, upper = historic_quantile(
            pd.DataFrame({"a": values}), prediction_interval=0.8
        )
        np.testing.assert_allclose(lower, [1.0])  # (5 - 1) / 4
        np.testing.assert_allclose(upper, [2.0])  # (13 - 5) / 4

    def test_average_value_naive_covers_central_interval(self):
        df = gaussian_noise_frame(n_rows=2000, n_series=8)
        prediction = AverageValueNaive(prediction_interval=0.8).fit(df).predict(5)
        width = (prediction.upper_forecast - prediction.lower_forecast).to_numpy()
        # true sigma is 1, so a central 80% interval is 2 * 1.2816 wide (one-sided: 1.68)
        self.assertAlmostEqual(float(np.median(width)), 2 * 1.2816, delta=0.15)


@unittest.skipIf(norm is None, "scipy not available")
class TestNormalBasedIntervals(unittest.TestCase):
    """For z-score intervals, half-width ratios between two intervals pin the convention."""

    central_ratio = None

    @classmethod
    def setUpClass(cls):
        # central: ppf(0.9) / ppf(0.8) = 1.52; one-sided would be ppf(0.8) / ppf(0.6) = 3.32
        cls.central_ratio = norm.ppf(0.9) / norm.ppf(0.8)

    def assert_central_ratio(self, model_class, **params):
        df = gaussian_noise_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wide = model_class(prediction_interval=0.8, **params).fit(df).predict(7)
            narrow = model_class(prediction_interval=0.6, **params).fit(df).predict(7)
        for wide_half, narrow_half in zip(half_widths(wide), half_widths(narrow)):
            np.testing.assert_allclose(
                wide_half / narrow_half, self.central_ratio, rtol=1e-4
            )

    def test_kalman_state_space(self):
        self.assert_central_ratio(KalmanStateSpace, em_iter=None)

    def test_basic_linear_model(self):
        self.assert_central_ratio(BasicLinearModel)

    def test_inferred_normal(self):
        df = gaussian_noise_frame()
        forecast = pd.DataFrame(
            np.full((3, df.shape[1]), 20.0),
            index=pd.date_range("2021-02-04", periods=3, freq="D"),
            columns=df.columns,
        )
        upper, lower = inferred_normal(df, forecast, prediction_interval=0.8)
        expected = norm.ppf(0.9) * df.std().to_numpy()
        # posterior mean sits at the data/prior blend, so compare total width
        np.testing.assert_allclose((upper - lower).to_numpy()[0], 2 * expected)


class TestEnsembleQuantileIntervals(unittest.TestCase):
    """Neighbour and seed-ensemble models must request q0.1 / q0.9 at 0.8."""

    def requested_quantiles(self, model):
        requested = []
        real_nan_quantile = basics.nan_quantile

        def recording_nan_quantile(arr, q, *args, **kwargs):
            requested.append(np.round(np.atleast_1d(q), 6).tolist())
            return real_nan_quantile(arr, q, *args, **kwargs)

        with mock.patch.object(basics, "nan_quantile", recording_nan_quantile):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(gaussian_noise_frame()).predict(7)
        return {q for qs in requested for q in qs}

    def test_sectional_motif(self):
        quantiles = self.requested_quantiles(SectionalMotif(prediction_interval=0.8))
        self.assertTrue({0.1, 0.9} <= quantiles, quantiles)
        self.assertFalse({0.2, 0.8} & quantiles, quantiles)

    def test_nvar(self):
        quantiles = self.requested_quantiles(NVAR(prediction_interval=0.8))
        self.assertTrue({0.1, 0.9} <= quantiles, quantiles)
        self.assertFalse({0.2, 0.8} & quantiles, quantiles)


class TestScaledPinballLossConvention(unittest.TestCase):
    def test_spl_scores_central_quantiles(self):
        index = pd.date_range("2021-01-01", periods=4, freq="D")
        actual = pd.DataFrame({"a": [0.0, 10.0, 5.0, -5.0]}, index=index)
        prediction = PredictionObject(prediction_interval=0.8)
        prediction.forecast = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0]}, index=index)
        prediction.upper_forecast = pd.DataFrame({"a": [2.0, 2.0, 2.0, 2.0]}, index=index)
        prediction.lower_forecast = pd.DataFrame(
            {"a": [-2.0, -2.0, -2.0, -2.0]}, index=index
        )
        df_train = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0, 0.0]})
        result = prediction.evaluate(actual, df_train=df_train)

        # hand-computed pinball at q0.9 (upper) and q0.1 (lower)
        y = actual["a"].to_numpy()
        upper_loss = np.where(y >= 2, 0.9 * (y - 2), 0.1 * (2 - y)).mean()
        lower_loss = np.where(y >= -2, 0.1 * (y + 2), 0.9 * (-2 - y)).mean()
        scaler = default_scaler(df_train.to_numpy())[0]
        expected = (upper_loss + lower_loss) / scaler
        self.assertAlmostEqual(result.per_series_metrics.loc["spl", "a"], expected)


if __name__ == "__main__":
    unittest.main()
