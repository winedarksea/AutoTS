# -*- coding: utf-8 -*-
"""Tests for the LRP-review items: origin anchor (R1), forced continuation
(R2), decoupled fit horizon (R3) and the reanchor live-offset fix.

Run with:  python -m pytest tests/test_tva_anchoring.py -v
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from autots.evaluator.tva.anchoring import (
    apply_origin_anchor,
    origin_anchor_adjustment,
)
from autots.evaluator.tva.factor_network import HAS_TORCH
from autots.evaluator.tva.tva import TVA, resolve_fit_horizon


def _weekly(n_time, level=100.0, amplitude=5.0, start=0):
    t = np.arange(start, start + n_time)
    return level + amplitude * np.sin(2 * np.pi * t / 7)


class TestOriginAnchorMath(unittest.TestCase):
    def test_level_gap_is_removed_multiplicatively(self):
        history = _weekly(56)[:, None]
        forecast = (_weekly(28, start=56) * 0.9)[:, None]
        adj = origin_anchor_adjustment(history, forecast, window=7)
        out = apply_origin_anchor(forecast, adj)
        self.assertTrue(adj['multiplicative'][0])
        self.assertAlmostEqual(out[:7].mean(), history[-7:].mean(), places=8)

    def test_weekly_cycle_without_gap_is_untouched(self):
        # full-window means: the phase of the cycle at the origin is not a gap
        history = _weekly(56)[:, None]
        forecast = _weekly(28, start=56)[:, None]
        adj = origin_anchor_adjustment(history, forecast, window=7)
        self.assertAlmostEqual(adj['ratio'][0], 1.0, places=8)

    def test_signed_series_shift_additively(self):
        history = (_weekly(56, level=0.0))[:, None]
        forecast = (_weekly(28, level=-3.0, start=56))[:, None]
        adj = origin_anchor_adjustment(history, forecast, window=7)
        self.assertFalse(adj['multiplicative'][0])
        self.assertAlmostEqual(adj['offset'][0], 3.0, places=8)

    def test_deseasonalized_ignores_a_holiday_dip_at_the_origin(self):
        history = np.full((56, 1), 100.0)
        holiday = np.zeros((56, 1))
        holiday[-5:] = -30.0
        history = history + holiday
        forecast = np.full((28, 1), 100.0)
        raw = origin_anchor_adjustment(history, forecast, window=7)
        deseason = origin_anchor_adjustment(
            history,
            forecast,
            window=7,
            mode='deseasonalized',
            history_periodic=holiday,
            forecast_periodic=np.zeros((28, 1)),
        )
        self.assertLess(raw['ratio'][0], 0.9)
        self.assertAlmostEqual(deseason['ratio'][0], 1.0, places=8)

    def test_all_nan_window_leaves_series_alone(self):
        history = np.column_stack([_weekly(30), np.full(30, np.nan)])
        forecast = np.column_stack([_weekly(14), _weekly(14)])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            adj = origin_anchor_adjustment(history, forecast, window=7)
        np.testing.assert_array_equal(apply_origin_anchor(forecast, adj)[:, 1], forecast[:, 1])

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            origin_anchor_adjustment(np.ones((10, 1)), np.ones((5, 1)), mode='nope')


class TestResolveFitHorizon(unittest.TestCase):
    def test_short_horizons_unchanged(self):
        self.assertEqual(resolve_fit_horizon('factor', 28, None, 900), 28)

    def test_long_horizon_capped_for_factor_and_none(self):
        self.assertEqual(resolve_fit_horizon('factor', 730, None, 900), 225)
        self.assertEqual(resolve_fit_horizon('none', 730, None, 900), 225)

    def test_explicit_and_network_modes(self):
        self.assertEqual(resolve_fit_horizon('factor', 730, 180, 900), 180)
        self.assertEqual(resolve_fit_horizon('v2', 730, None, 900), 730)


def _panel(n_time=420, n_series=5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    common = np.cumsum(rng.normal(0, 0.3, n_time))
    values = (
        100.0
        + common[:, None] * rng.uniform(0.5, 1.5, n_series)
        + 4 * np.sin(2 * np.pi * t / 7)[:, None]
        + rng.normal(0, 0.5, (n_time, n_series))
    )
    index = pd.date_range('2022-01-01', periods=n_time, freq='D')
    return pd.DataFrame(values, index=index, columns=[f's{i}' for i in range(n_series)])


def _fit(df, horizon=28, factor_config=None, trend_network='factor', **kwargs):
    model = TVA(
        trend_network=trend_network,
        forecast_horizon=horizon,
        random_seed=0,
        verbose=0,
        factor_config=factor_config,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(df)
    return model


def _predict(model, horizon):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.predict(horizon)


class TestOriginAnchorNumpyMode(unittest.TestCase):
    def test_none_mode_anchor_matches_recent_level(self):
        df = _panel()
        model = _fit(df, trend_network='none', factor_config={'origin_anchor': 'last_value'})
        fc = _predict(model, 28)
        np.testing.assert_allclose(
            fc.iloc[:7].mean().values, df.iloc[-7:].mean().values, rtol=1e-9
        )

    def test_off_is_bit_for_bit(self):
        df = _panel()
        base = _predict(_fit(df, trend_network='none'), 28)
        off = _predict(
            _fit(df, trend_network='none', factor_config={'origin_anchor': None}), 28
        )
        np.testing.assert_array_equal(base.values, off.values)


@unittest.skipUnless(HAS_TORCH, "torch required for trend_network='factor'")
class TestFactorModeReviewItems(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _panel()

    def test_origin_anchor_pins_the_start(self):
        model = _fit(self.df, factor_config={'origin_anchor': 'last_value'})
        fc = _predict(model, 28)
        np.testing.assert_allclose(
            fc.iloc[:7].mean().values, self.df.iloc[-7:].mean().values, rtol=1e-6
        )
        self.assertIn('origin_anchor', model.get_factor_diagnostics())

    def test_forced_constant_holds_factor_paths(self):
        model = _fit(self.df, factor_config={'continuation_force': 'constant'})
        self.assertEqual(model._factor_info['continuation']['forced'], 'constant')
        deltas = model._selected_continuation(40)
        np.testing.assert_array_equal(deltas, np.zeros_like(deltas))

    def test_forced_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            _fit(self.df, factor_config={'continuation_force': 'not_a_spec'})

    def test_long_horizon_keeps_stage_b(self):
        model = _fit(self.df, horizon=300)
        self.assertEqual(model._fit_horizon, 105)
        self.assertIsNotNone(model._factor_info.get('stage_b_loss'))
        fc = _predict(model, 300)
        self.assertEqual(fc.shape, (300, self.df.shape[1]))
        self.assertTrue(np.isfinite(fc.values).all())

    def test_reanchor_live_offset_uses_last_origin(self):
        from unittest import mock

        from autots.evaluator.tva import safety

        model = _fit(self.df, factor_config={'reanchor': True})
        window = model._reanchor_window()
        anchor_now = np.nanmedian(model._factor_input_panel()[-window:], axis=0)
        live_trend = model._factor_trend_at_origin(len(self.df) - 1, 28)
        n_series = self.df.shape[1]
        # alpha forced to 1: the live step-0 trend must land on the recent level
        with mock.patch.object(
            safety, 'select_reanchor_alpha', return_value=np.ones(n_series)
        ):
            shifted, _info = model._apply_safety_layer(live_trend.copy(), 28)
        np.testing.assert_allclose(shifted[0], anchor_now, rtol=1e-6)

if __name__ == '__main__':
    unittest.main()
