# -*- coding: utf-8 -*-
"""Tests for the TVA factor-mode safety floor (``autots.evaluator.tva.safety``).

The module is pure numpy by design, so nothing here fits a model or needs torch.
Two properties are load-bearing and asserted explicitly rather than
approximately: the identity paths are **bitwise** exact (wiring the floor in
must not perturb an existing run until a selector actually fires), and no
degenerate input raises.

Run with:  python -m pytest tests/test_tva_safety.py -v
"""

import unittest
import json
import numpy as np

from autots.evaluator.tva.safety import (
    DEFAULT_SAFETY_CONFIG,
    apply_error_cap,
    apply_reanchor,
    blend_forecasts,
    conformal_sigma,
    count_capped,
    error_cap_bounds,
    horizon_bucket_scales,
    seasonal_naive_forecast,
    select_blend_weights,
    select_reanchor_alpha,
    summarize,
)


def _panel(n_time=120, n_series=4, seed=0):
    """Weekly-seasonal panel with a mild trend — the ordinary case."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    seasonal = np.sin(t * 2 * np.pi / 7)[:, None]
    trend = np.linspace(0, 2, n_time)[:, None]
    return 10.0 + trend + seasonal + rng.normal(0, 0.1, (n_time, n_series))


def _folds(n_folds=3, horizon=10, n_series=4, seed=0):
    """(tva, sn, actual) fold lists where TVA is exact and SN is offset."""
    rng = np.random.default_rng(seed)
    actual = [rng.normal(0, 1, (horizon, n_series)) for _ in range(n_folds)]
    tva = [a.copy() for a in actual]
    sn = [a + 5.0 for a in actual]
    return tva, sn, actual


class TestSeasonalNaive(unittest.TestCase):
    def test_tiles_the_last_season(self):
        history = np.arange(28.0).reshape(14, 2)
        fc = seasonal_naive_forecast(history, horizon=21, season_m=7)
        self.assertEqual(fc.shape, (21, 2))
        np.testing.assert_array_equal(fc[:7], history[-7:])
        np.testing.assert_array_equal(fc[7:14], history[-7:])
        np.testing.assert_array_equal(fc[14:21], history[-7:])

    def test_recovers_exact_seasonal_pattern(self):
        pattern = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        history = np.tile(pattern, 10)[:, None]
        fc = seasonal_naive_forecast(history, horizon=14, season_m=7)
        np.testing.assert_allclose(fc[:, 0], np.tile(pattern, 2))

    def test_nan_slot_falls_back_to_last_finite(self):
        history = np.arange(20.0).reshape(10, 2)
        history[-3, 0] = np.nan   # a NaN inside the seasonal window
        fc = seasonal_naive_forecast(history, horizon=7, season_m=7)
        self.assertTrue(np.all(np.isfinite(fc)))
        # the NaN slot took the column's last finite value
        self.assertEqual(fc[4, 0], 18.0)

    def test_all_nan_column_falls_back_to_zero(self):
        history = np.full((20, 2), np.nan)
        history[:, 1] = 3.0
        fc = seasonal_naive_forecast(history, horizon=5, season_m=7)
        np.testing.assert_array_equal(fc[:, 0], np.zeros(5))
        np.testing.assert_array_equal(fc[:, 1], np.full(5, 3.0))

    def test_season_longer_than_history(self):
        fc = seasonal_naive_forecast(np.arange(6.0)[:, None], horizon=4, season_m=7)
        self.assertEqual(fc.shape, (4, 1))
        self.assertTrue(np.all(np.isfinite(fc)))


class TestBlendIdentity(unittest.TestCase):
    """Back-compat: weight 1.0 must not perturb a single bit."""

    def test_weight_one_is_bitwise_identical(self):
        rng = np.random.default_rng(3)
        tva = rng.normal(0, 1e3, (30, 5))
        sn = rng.normal(0, 1e3, (30, 5))
        out = blend_forecasts(tva, sn, np.ones(5))
        self.assertTrue(np.array_equal(out, tva))
        self.assertEqual(out.tobytes(), tva.tobytes())

    def test_weight_zero_is_bitwise_identical_to_baseline(self):
        rng = np.random.default_rng(4)
        tva = rng.normal(0, 1e3, (30, 5))
        sn = rng.normal(0, 1e3, (30, 5))
        out = blend_forecasts(tva, sn, np.zeros(5))
        self.assertEqual(out.tobytes(), sn.tobytes())

    def test_partial_weight_is_convex(self):
        tva = np.full((4, 2), 10.0)
        sn = np.zeros((4, 2))
        out = blend_forecasts(tva, sn, np.array([0.25, 0.75]))
        np.testing.assert_allclose(out[:, 0], 2.5)
        np.testing.assert_allclose(out[:, 1], 7.5)

    def test_per_series_weights_broadcast_over_horizon(self):
        tva = np.ones((6, 3))
        sn = np.zeros((6, 3))
        out = blend_forecasts(tva, sn, np.array([0.0, 0.5, 1.0]))
        self.assertEqual(out.shape, (6, 3))
        np.testing.assert_allclose(out[0], [0.0, 0.5, 1.0])


class TestBlendSelection(unittest.TestCase):
    def test_exact_tva_earns_full_weight(self):
        tva, sn, actual = _folds(seed=1)
        w = select_blend_weights(tva, sn, actual, scale=np.ones(4))
        np.testing.assert_array_equal(w, np.ones(4))

    def test_noise_tva_gets_zero_weight(self):
        rng = np.random.default_rng(2)
        actual = [rng.normal(0, 1, (10, 4)) for _ in range(3)]
        sn = [a + rng.normal(0, 0.01, a.shape) for a in actual]  # nearly exact
        tva = [rng.normal(0, 50, (10, 4)) for _ in range(3)]     # pure noise
        w = select_blend_weights(tva, sn, actual, scale=np.ones(4))
        np.testing.assert_array_equal(w, np.zeros(4))

    def test_tie_rule_prefers_the_smaller_tva_weight(self):
        # identical forecasts: every weight on the grid scores exactly the
        # same, so the choice is pure risk preference. The smallest must win.
        actual = [np.zeros((8, 2))]
        tva = [np.full((8, 2), 2.0)]
        sn = [np.full((8, 2), 2.0)]
        w = select_blend_weights(tva, sn, actual, scale=np.ones(2))
        np.testing.assert_array_equal(w, np.zeros(2))

    def test_near_tie_within_tolerance_prefers_smaller_weight(self):
        # TVA is better, but by only 0.5% — inside the 1% band, so the floor
        # takes the lower-risk option rather than the nominally better one.
        actual = [np.zeros((20, 1))]
        sn = [np.full((20, 1), 1.0)]
        tva = [np.full((20, 1), 0.995)]
        w = select_blend_weights(tva, sn, actual, scale=np.ones(1))
        self.assertEqual(float(w[0]), 0.0)
        # widen the gap well past the tolerance and full weight is earned
        tva = [np.full((20, 1), 0.1)]
        w = select_blend_weights(tva, sn, actual, scale=np.ones(1))
        self.assertEqual(float(w[0]), 1.0)

    def test_weights_are_per_series(self):
        actual = [np.zeros((10, 2))]
        tva = [np.column_stack([np.zeros(10), np.full(10, 100.0)])]
        sn = [np.column_stack([np.full(10, 100.0), np.zeros(10)])]
        w = select_blend_weights(tva, sn, actual, scale=np.ones(2))
        np.testing.assert_array_equal(w, np.array([1.0, 0.0]))

    def test_all_nan_series_gets_safe_default(self):
        actual = [np.full((10, 2), np.nan)]
        tva = [np.zeros((10, 2))]
        sn = [np.zeros((10, 2))]
        w = select_blend_weights(tva, sn, actual, scale=np.ones(2))
        np.testing.assert_array_equal(w, np.zeros(2))

    def test_selected_weights_are_on_the_grid(self):
        tva, sn, actual = _folds(seed=7)
        w = select_blend_weights(tva, sn, actual, scale=np.ones(4))
        for value in w:
            self.assertIn(float(value), DEFAULT_SAFETY_CONFIG['blend_grid'])


class TestErrorCap(unittest.TestCase):
    def test_clips_a_blown_up_extrapolation(self):
        sn = np.zeros((10, 2))
        errors = [np.full((10, 2), 1.0)]
        lower, upper = error_cap_bounds(sn, errors, horizon=10)
        fc = np.full((10, 2), 500.0)   # runaway
        capped = apply_error_cap(fc, lower, upper)
        np.testing.assert_allclose(capped, 3.0)  # 3.0 * q99(=1.0)
        self.assertEqual(count_capped(fc, lower, upper), 20)

    def test_leaves_a_reasonable_forecast_alone(self):
        sn = np.zeros((10, 2))
        errors = [np.full((10, 2), 1.0)]
        lower, upper = error_cap_bounds(sn, errors, horizon=10)
        fc = np.full((10, 2), 1.5)
        self.assertEqual(count_capped(fc, lower, upper), 0)
        self.assertTrue(np.array_equal(apply_error_cap(fc, lower, upper), fc))

    def test_accepts_precomputed_per_series_quantiles(self):
        sn = np.zeros((5, 3))
        lower, upper = error_cap_bounds(sn, np.array([1.0, 2.0, 4.0]), horizon=5)
        np.testing.assert_allclose(upper[0], [3.0, 6.0, 12.0])
        np.testing.assert_allclose(lower[0], [-3.0, -6.0, -12.0])

    def test_horizon_scaled_cap_is_wider_late_than_early(self):
        sn = np.zeros((100, 2))
        errors = [np.full((100, 2), 1.0)]
        scales = np.concatenate([np.ones(28), np.full(62, 2.0), np.full(10, 4.0)])
        lower, upper = error_cap_bounds(
            sn, errors, horizon=100, bucket_scales=scales
        )
        self.assertGreater(float(upper[95, 0]), float(upper[50, 0]))
        self.assertGreater(float(upper[50, 0]), float(upper[0, 0]))
        np.testing.assert_allclose(upper[0], 3.0)
        np.testing.assert_allclose(upper[95], 12.0)
        # and the band stays symmetric about the baseline
        np.testing.assert_allclose(lower, -upper)

    def test_nan_bounds_mean_no_cap_and_are_bitwise_identity(self):
        rng = np.random.default_rng(11)
        fc = rng.normal(0, 1e4, (12, 3))
        nan_bounds = np.full((12, 3), np.nan)
        out = apply_error_cap(fc, nan_bounds, nan_bounds)
        self.assertEqual(out.tobytes(), fc.tobytes())
        self.assertEqual(count_capped(fc, nan_bounds, nan_bounds), 0)

    def test_unestimable_series_is_left_uncapped(self):
        sn = np.zeros((5, 2))
        errors = [np.column_stack([np.full(5, 1.0), np.full(5, np.nan)])]
        lower, upper = error_cap_bounds(sn, errors, horizon=5)
        self.assertTrue(np.all(np.isfinite(upper[:, 0])))
        self.assertTrue(np.all(np.isnan(upper[:, 1])))
        fc = np.full((5, 2), 1e6)
        capped = apply_error_cap(fc, lower, upper)
        self.assertEqual(float(capped[0, 1]), 1e6)   # untouched
        self.assertEqual(float(capped[0, 0]), 3.0)


class TestHorizonBucketScales(unittest.TestCase):
    def test_first_bucket_is_exactly_one(self):
        rng = np.random.default_rng(5)
        residuals = rng.normal(0, 1, (6, 180, 4))
        scales = horizon_bucket_scales(residuals)
        self.assertEqual(scales.shape, (180,))
        self.assertAlmostEqual(float(scales[0]), 1.0)
        np.testing.assert_allclose(scales[:28], 1.0)

    def test_growing_residuals_give_growing_scales(self):
        rng = np.random.default_rng(6)
        h = np.arange(1, 181, dtype=float)
        residuals = rng.normal(0, 1, (8, 180, 3)) * h[None, :, None]
        scales = horizon_bucket_scales(residuals)
        self.assertAlmostEqual(float(scales[0]), 1.0)
        self.assertGreater(float(scales[50]), 1.0)
        self.assertGreater(float(scales[150]), float(scales[50]))
        # constant within a bucket
        self.assertEqual(len(set(np.round(scales[29:90], 10))), 1)

    def test_flat_residuals_stay_flat(self):
        residuals = np.ones((4, 180, 2))
        np.testing.assert_allclose(horizon_bucket_scales(residuals), 1.0)

    def test_short_horizon_is_all_first_bucket(self):
        rng = np.random.default_rng(8)
        residuals = rng.normal(0, 1, (3, 10, 2))
        scales = horizon_bucket_scales(residuals)
        self.assertEqual(scales.shape, (10,))
        np.testing.assert_allclose(scales, 1.0)

    def test_steps_past_the_last_bucket_take_the_last_scale(self):
        rng = np.random.default_rng(9)
        h = np.arange(1, 221, dtype=float)
        residuals = rng.normal(0, 1, (5, 220, 2)) * h[None, :, None]
        scales = horizon_bucket_scales(residuals)
        self.assertEqual(scales.shape, (220,))
        np.testing.assert_allclose(scales[180:], scales[179])

    def test_all_nan_residuals_return_flat(self):
        residuals = np.full((3, 60, 2), np.nan)
        np.testing.assert_allclose(horizon_bucket_scales(residuals), 1.0)


class TestConformalSigma(unittest.TestCase):
    def test_none_scales_is_bitwise_the_current_tile(self):
        rng = np.random.default_rng(12)
        base = rng.normal(5, 1, 7)
        out = conformal_sigma(base, 180, None)
        expected = np.tile(base[np.newaxis, :], (180, 1))
        self.assertEqual(out.tobytes(), expected.tobytes())

    def test_ones_scales_match_the_flat_tile(self):
        base = np.array([1.0, 2.5, 4.0])
        flat = conformal_sigma(base, 20, None)
        scaled = conformal_sigma(base, 20, np.ones(20))
        np.testing.assert_allclose(scaled, flat)

    def test_scales_widen_late_horizon(self):
        base = np.array([1.0, 2.0])
        scales = np.concatenate([np.ones(10), np.full(10, 3.0)])
        out = conformal_sigma(base, 20, scales)
        self.assertEqual(out.shape, (20, 2))
        np.testing.assert_allclose(out[0], base)
        np.testing.assert_allclose(out[19], base * 3.0)

    def test_short_scales_extend_with_the_last_value(self):
        out = conformal_sigma(np.array([1.0]), 10, np.array([1.0, 2.0]))
        np.testing.assert_allclose(out[2:, 0], 2.0)


class TestReanchor(unittest.TestCase):
    def test_alpha_zero_is_bitwise_identity(self):
        rng = np.random.default_rng(13)
        fc = rng.normal(0, 1e3, (30, 4))
        out = apply_reanchor(fc, rng.normal(0, 10, 4), 0.0)
        self.assertEqual(out.tobytes(), fc.tobytes())

    def test_apply_shifts_by_alpha_times_offset(self):
        fc = np.zeros((5, 2))
        out = apply_reanchor(fc, np.array([2.0, -4.0]), np.array([0.5, 1.0]))
        np.testing.assert_allclose(out[0], [1.0, -4.0])

    def test_selects_alpha_one_when_offset_is_the_only_error(self):
        n_folds, horizon, n_series = 3, 12, 2
        actual = [np.zeros((horizon, n_series)) for _ in range(n_folds)]
        bias = np.array([5.0, -3.0])
        tva = [np.tile(-bias, (horizon, 1)) for _ in range(n_folds)]
        anchors = np.tile(np.zeros(n_series), (n_folds, 1))
        origins = np.tile(-bias, (n_folds, 1))
        alpha = select_reanchor_alpha(
            tva, actual, anchors, origins, scale=np.ones(n_series)
        )
        np.testing.assert_array_equal(alpha, np.ones(n_series))

    def test_selects_alpha_zero_when_there_is_no_offset(self):
        n_folds, horizon, n_series = 3, 12, 2
        rng = np.random.default_rng(14)
        actual = [rng.normal(0, 1, (horizon, n_series)) for _ in range(n_folds)]
        tva = [a.copy() for a in actual]
        anchors = np.zeros((n_folds, n_series))
        origins = np.zeros((n_folds, n_series))
        alpha = select_reanchor_alpha(
            tva, actual, anchors, origins, scale=np.ones(n_series)
        )
        np.testing.assert_array_equal(alpha, np.zeros(n_series))

    def test_bogus_offset_is_rejected(self):
        # the offset points the wrong way; correcting would hurt, so alpha=0
        n_folds, horizon = 3, 12
        actual = [np.zeros((horizon, 1)) for _ in range(n_folds)]
        tva = [np.zeros((horizon, 1)) for _ in range(n_folds)]
        anchors = np.full((n_folds, 1), 20.0)
        origins = np.zeros((n_folds, 1))
        alpha = select_reanchor_alpha(tva, actual, anchors, origins, np.ones(1))
        np.testing.assert_array_equal(alpha, np.zeros(1))

    def test_selected_alphas_are_on_the_grid(self):
        n_folds, horizon = 4, 10
        rng = np.random.default_rng(15)
        actual = [rng.normal(0, 1, (horizon, 3)) for _ in range(n_folds)]
        tva = [a - 2.0 for a in actual]
        anchors = np.full((n_folds, 3), 1.0)
        origins = np.zeros((n_folds, 3))
        alpha = select_reanchor_alpha(tva, actual, anchors, origins, np.ones(3))
        for value in alpha:
            self.assertIn(float(value), DEFAULT_SAFETY_CONFIG['reanchor_alphas'])


class TestFullIdentityPath(unittest.TestCase):
    """Blend 1.0 + alpha 0.0 + no cap + flat sigma == today's pipeline."""

    def test_composed_identity_is_bitwise_exact(self):
        rng = np.random.default_rng(16)
        tva = rng.normal(100, 30, (180, 6))
        sn = rng.normal(100, 30, (180, 6))
        out = blend_forecasts(tva, sn, np.ones(6))
        out = apply_reanchor(out, rng.normal(0, 5, 6), np.zeros(6))
        out = apply_error_cap(out, np.full((180, 6), np.nan), np.full((180, 6), np.nan))
        self.assertEqual(out.tobytes(), tva.tobytes())

    def test_sigma_identity_is_bitwise_exact(self):
        rng = np.random.default_rng(17)
        base = rng.uniform(0.1, 9.0, 6)
        self.assertEqual(
            conformal_sigma(base, 180, None).tobytes(),
            np.tile(base[np.newaxis, :], (180, 1)).tobytes(),
        )


class TestDegenerateInputs(unittest.TestCase):
    """Nothing here may raise; every path returns a safe default."""

    def test_empty_folds(self):
        np.testing.assert_array_equal(
            select_blend_weights([], [], [], np.ones(3)), np.zeros(0)
        )
        np.testing.assert_array_equal(
            select_reanchor_alpha([], [], np.zeros((0, 3)), np.zeros((0, 3)), np.ones(3)),
            np.zeros(0),
        )

    def test_single_fold(self):
        tva, sn, actual = _folds(n_folds=1, seed=18)
        w = select_blend_weights(tva, sn, actual, np.ones(4))
        self.assertEqual(w.shape, (4,))

    def test_zero_and_nan_scale(self):
        tva, sn, actual = _folds(seed=19)
        scale = np.array([0.0, np.nan, -1.0, 2.0])
        w = select_blend_weights(tva, sn, actual, scale)
        self.assertEqual(w.shape, (4,))
        self.assertTrue(np.all(np.isfinite(w)))

    def test_empty_grids(self):
        tva, sn, actual = _folds(seed=20)
        w = select_blend_weights(tva, sn, actual, np.ones(4), config={'blend_grid': ()})
        np.testing.assert_array_equal(w, np.zeros(4))
        a = select_reanchor_alpha(
            tva, actual, np.zeros((3, 4)), np.zeros((3, 4)), np.ones(4),
            config={'reanchor_alphas': ()},
        )
        np.testing.assert_array_equal(a, np.zeros(4))

    def test_zero_horizon(self):
        self.assertEqual(seasonal_naive_forecast(_panel(), 0).shape, (0, 4))
        lower, upper = error_cap_bounds(np.zeros((0, 2)), np.ones(2), horizon=0)
        self.assertEqual(lower.shape[0], 0)
        self.assertEqual(conformal_sigma(np.ones(3), 0, None).shape, (0, 3))

    def test_empty_history_and_empty_sigma(self):
        self.assertEqual(seasonal_naive_forecast(np.zeros((0, 3)), 5).shape, (5, 3))
        self.assertEqual(conformal_sigma(np.array([]), 5, None).shape, (5, 0))

    def test_all_nan_forecast_survives_the_cap(self):
        fc = np.full((5, 2), np.nan)
        lower, upper = error_cap_bounds(np.zeros((5, 2)), [np.ones((5, 2))], 5)
        out = apply_error_cap(fc, lower, upper)
        self.assertTrue(np.all(np.isnan(out)))
        self.assertEqual(count_capped(fc, lower, upper), 0)

    def test_missing_error_source(self):
        lower, upper = error_cap_bounds(np.zeros((5, 2)), None, horizon=5)
        self.assertTrue(np.all(np.isnan(lower)))
        self.assertTrue(np.all(np.isnan(upper)))

    def test_bad_residual_shapes(self):
        self.assertEqual(horizon_bucket_scales(np.zeros(0)).shape, (0,))
        self.assertEqual(horizon_bucket_scales(np.ones((10, 2))).shape, (10,))

    def test_no_buckets_configured(self):
        scales = horizon_bucket_scales(np.ones((2, 30, 2)), config={'horizon_buckets': ()})
        np.testing.assert_allclose(scales, 1.0)

    def test_nan_weights_and_alphas_are_treated_as_no_op(self):
        tva = np.ones((4, 2))
        sn = np.zeros((4, 2))
        np.testing.assert_allclose(blend_forecasts(tva, sn, np.full(2, np.nan)), 0.0)
        out = apply_reanchor(tva, np.full(2, np.nan), np.ones(2))
        np.testing.assert_allclose(out, 1.0)


class TestSummarize(unittest.TestCase):
    def test_is_json_serializable_and_reports_counts(self):
        diag = summarize(
            blend_weights=np.array([0.0, 1.0, 0.5, 0.0]),
            reanchor_alphas=np.array([0.0, 1.0, 0.0, 0.5]),
            bucket_scales=np.array([1.0, 1.0, 2.0]),
            capped_cells=17,
            scale=np.array([1.0, np.nan, 3.0, 4.0]),
            config=DEFAULT_SAFETY_CONFIG,
            note='factor',
        )
        json.dumps(diag)  # must not raise on NaN/ndarray/tuple
        self.assertEqual(diag['n_series'], 4)
        self.assertEqual(diag['n_series_zero_weight'], 2)
        self.assertEqual(diag['n_series_full_weight'], 1)
        self.assertEqual(diag['n_series_reanchored'], 2)
        self.assertEqual(diag['capped_cells'], 17)
        self.assertAlmostEqual(diag['blend_weight_mean'], 0.375)
        self.assertEqual(diag['blend_weights'], [0.0, 1.0, 0.5, 0.0])
        self.assertIsNone(diag['scale'][1])   # NaN became None
        self.assertEqual(diag['note'], 'factor')

    def test_empty_summary_is_safe(self):
        diag = summarize()
        json.dumps(diag)
        self.assertIsNone(diag['blend_weights'])
        self.assertIsNone(diag['capped_cells'])


class TestEndToEndFloor(unittest.TestCase):
    """The realistic shape of the wiring the caller will do."""

    def test_runaway_series_ends_up_near_the_baseline(self):
        history = _panel(n_time=200, n_series=3, seed=21)
        horizon = 60
        sn = seasonal_naive_forecast(history, horizon, season_m=7)
        # series 0 blows up linearly; the others track the baseline
        tva = sn.copy()
        tva[:, 0] = sn[:, 0] + np.arange(horizon) * 20.0
        inner_errors = [np.full((horizon, 3), 0.5)]
        lower, upper = error_cap_bounds(sn, inner_errors, horizon)
        capped = apply_error_cap(tva, lower, upper)
        self.assertLess(float(np.max(np.abs(capped[:, 0] - sn[:, 0]))), 2.0)
        np.testing.assert_allclose(capped[:, 1:], tva[:, 1:])
        self.assertGreater(count_capped(tva, lower, upper), 0)


if __name__ == '__main__':
    unittest.main()
