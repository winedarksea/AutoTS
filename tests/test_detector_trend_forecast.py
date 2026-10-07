# -*- coding: utf-8 -*-
"""Tests for the detector's trend forecast helpers (future-change variance, slope shrinkage)."""

import unittest

import numpy as np
import pandas as pd

from autots.datasets import load_daily
from autots.evaluator.feature_detector import TimeSeriesFeatureDetector
from autots.evaluator.feature_detector.utils.trend_forecast import (
    future_trend_change_variance,
    segment_sxx,
    shrink_last_slope,
    trend_change_rates,
)


def _segments(date_index, boundaries, slopes):
    """slope_info entries over inclusive positional boundaries, as fit builds them."""
    return [
        {
            'start_date': date_index[start],
            'end_date': date_index[end],
            'slope': slope,
        }
        for (start, end), slope in zip(zip(boundaries[:-1], boundaries[1:]), slopes)
    ]


class TestTrendChangeVariance(unittest.TestCase):
    def test_rates_and_closed_form(self):
        date = pd.Timestamp('2020-01-01')
        changepoints = {'a': [(date, 0.0, 1.0), (date, 1.0, -1.0)], 'b': []}
        level_shifts = {'a': [(date, 3.0, 'validated', False)], 'b': []}
        rates = trend_change_rates(changepoints, level_shifts, ['a', 'b'], 100)
        np.testing.assert_allclose(rates['rate_cp'], [0.02, 0.0])
        np.testing.assert_allclose(rates['mean_sq_slope_change'], [2.5, 0.0])
        np.testing.assert_allclose(rates['rate_ls'], [0.01, 0.0])
        np.testing.assert_allclose(rates['mean_sq_level_shift'], [9.0, 0.0])

        variance = future_trend_change_variance(rates, 10)
        self.assertEqual(variance.shape, (10, 2))
        h = np.arange(1, 11, dtype=float)
        expected = 0.02 * 2.5 * (h - 1) * h * (2 * h - 1) / 6 + 0.01 * 9.0 * h
        np.testing.assert_allclose(variance[:, 0], expected)
        # No detected events: no added variance (previous behaviour).
        np.testing.assert_allclose(variance[:, 1], 0.0)
        # Slope changes cannot move the level within one step.
        self.assertAlmostEqual(variance[0, 0], 0.01 * 9.0)

    def test_missing_inputs_give_zero(self):
        rates = trend_change_rates(None, None, ['a'], 50)
        np.testing.assert_allclose(future_trend_change_variance(rates, 5), 0.0)


class TestShrinkLastSlope(unittest.TestCase):
    def setUp(self):
        self.date_index = pd.date_range('2020-01-01', periods=400, freq='D')

    def test_single_segment_unchanged(self):
        info = _segments(self.date_index, [0, 399], [0.7])
        self.assertEqual(shrink_last_slope(info, self.date_index, 1.0, 0.5), 0.7)

    def test_zero_dispersion_unchanged(self):
        info = _segments(self.date_index, [0, 300, 399], [0.1, 0.7])
        self.assertEqual(shrink_last_slope(info, self.date_index, 1.0, 0.0), 0.7)

    def test_shrinks_between_and_stronger_for_short_segment(self):
        long_last = _segments(self.date_index, [0, 200, 399], [0.1, 1.0])
        short_last = _segments(self.date_index, [0, 389, 399], [0.1, 1.0])
        shrunk_long = shrink_last_slope(long_last, self.date_index, 5.0, 0.01)
        shrunk_short = shrink_last_slope(short_last, self.date_index, 5.0, 0.01)
        for shrunk in (shrunk_long, shrunk_short):
            self.assertGreater(shrunk, 0.1)
            self.assertLess(shrunk, 1.0)
        self.assertLess(shrunk_short, shrunk_long)
        # Matches the precision-weighted closed form (11-point last segment).
        tau_sq, slope_var = 0.005, 25.0 / segment_sxx(11)
        expected = (tau_sq * 1.0 + slope_var * 0.1) / (tau_sq + slope_var)
        self.assertAlmostEqual(shrunk_short, expected)

    def test_recent_segments_dominate_prior(self):
        # Old segment slope -1, recent segment slope +1, last segment +5.
        info = _segments(self.date_index, [0, 100, 380, 399], [-1.0, 1.0, 5.0])
        shrunk = shrink_last_slope(info, self.date_index, 50.0, 1.0)
        # Huge noise: result approaches the prior, which leans to the recent +1.
        self.assertGreater(shrunk, 0.0)
        self.assertLess(shrunk, 5.0)


class TestDetectorForecastOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_daily(long=False)
        cls.data = df.loc[:, df.notna().mean() > 0.9].iloc[-500:, :3].ffill().bfill()
        cls.detector = TimeSeriesFeatureDetector()
        cls.detector.fit(cls.data)

    def test_variance_term_only_widens_bounds(self):
        off = self.detector.forecast(30, trend_change_variance=False)
        on = self.detector.forecast(30, trend_change_variance=True)
        pd.testing.assert_frame_equal(off.forecast, on.forecast)
        width_off = (off.upper_forecast - off.lower_forecast).to_numpy()
        width_on = (on.upper_forecast - on.lower_forecast).to_numpy()
        self.assertTrue((width_on >= width_off - 1e-9).all())
        self.assertTrue((np.diff(width_on, axis=0) >= -1e-9).all())

    def test_slope_shrinkage_keeps_anchor_and_finite(self):
        base = self.detector.forecast(30, slope_shrinkage=False)
        shrunk = self.detector.forecast(30, slope_shrinkage=True)
        self.assertTrue(np.isfinite(shrunk.forecast.to_numpy()).all())
        # Same anchor level; only the slope differs, so the step-1 gap is tiny
        # relative to the step-30 gap.
        gap = (shrunk.forecast - base.forecast).abs().to_numpy()
        self.assertTrue((gap[0] <= gap[-1] / 29.0 + 1e-6).all())


if __name__ == '__main__':
    unittest.main()
