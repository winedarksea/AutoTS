# -*- coding: utf-8 -*-
"""Tests for the batched L1 solve fix (NumPy>=2) and the 'noise' kink threshold."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autots.tools.changepoints import (
    ChangepointDetector,
    _approximate_l1_trend_filter_batch,
    _build_difference_matrix,
    _extract_changepoints_from_trend_batch,
    _solve_weighted_trend_system_batch,
)


def _noisy_piecewise_linear(n=300, kink=150, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return 0.05 * t + 0.4 * np.maximum(0, t - kink) + rng.normal(0, noise, n)


def _frame(values):
    index = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"a": values}, index=index)


def _detect(values, method="l1_total_variation", **params):
    det = ChangepointDetector(
        method=method,
        method_params=params,
        aggregate_method="individual",
        min_segment_length=5,
    )
    det.detect(_frame(values))
    return det


class TestBatchSolve(unittest.TestCase):
    def test_solve_shape_and_matches_loop(self):
        rng = np.random.default_rng(1)
        k, n, order = 3, 40, 2
        D = _build_difference_matrix(n, order)
        chunk = rng.normal(size=(k, n))
        weights = rng.uniform(0.5, 2.0, size=(k, D.shape[0]))
        identity = np.eye(n)
        got = _solve_weighted_trend_system_batch(chunk, D, identity, 1.5, weights)
        self.assertEqual(got.shape, (k, n))
        for i in range(k):
            Dw = D * np.sqrt(1.5 * weights[i])[:, None]
            expected = np.linalg.solve(identity + Dw.T @ Dw, chunk[i])
            np.testing.assert_allclose(got[i], expected, rtol=1e-8, atol=1e-10)

    def test_fitted_trend_responds_to_lambda(self):
        values = _noisy_piecewise_linear()
        small = _approximate_l1_trend_filter_batch(values[None, :], 2, 0.1)
        large = _approximate_l1_trend_filter_batch(values[None, :], 2, 500.0)
        self.assertFalse(np.allclose(small, values[None, :]))
        self.assertFalse(np.allclose(small, large))
        det_small = _detect(values, lambda_reg=0.1, difference_order=2)
        det_large = _detect(values, lambda_reg=500.0, difference_order=2)
        a = np.asarray(det_small.fitted_trends_["a"], dtype=float)
        b = np.asarray(det_large.fitted_trends_["a"], dtype=float)
        self.assertFalse(np.allclose(a, values))
        self.assertFalse(np.allclose(a, b))


class TestNoiseThreshold(unittest.TestCase):
    def test_noise_mode_quiet_on_pure_trend(self):
        rng = np.random.default_rng(3)
        values = 0.1 * np.arange(500) + rng.normal(0, 1.0, 500)
        rel = _detect(values, lambda_reg=1.0, difference_order=2)
        noise = _detect(
            values,
            lambda_reg=20.0,
            difference_order=2,
            threshold_mode="noise",
            threshold_multiplier=1.5,
        )
        n_rel = len(rel.changepoints_["a"])
        n_noise = len(noise.changepoints_["a"])
        self.assertGreater(n_rel, 1)
        self.assertLessEqual(n_noise, 1)

    def test_noise_mode_finds_clear_kink(self):
        values = _noisy_piecewise_linear(n=300, kink=150, noise=0.5)
        det = _detect(
            values,
            lambda_reg=20.0,
            difference_order=2,
            threshold_mode="noise",
            threshold_multiplier=1.5,
        )
        cps = np.asarray(det.changepoints_["a"])
        self.assertTrue(np.any(np.abs(cps - 150) <= 30), cps)

    def test_default_mode_unchanged_by_explicit_relative(self):
        values = _noisy_piecewise_linear(seed=5)
        a = _detect(values, lambda_reg=1.0, difference_order=2)
        b = _detect(
            values,
            lambda_reg=1.0,
            difference_order=2,
            threshold_mode="relative",
            threshold_multiplier=7.0,
        )
        np.testing.assert_array_equal(a.changepoints_["a"], b.changepoints_["a"])

    def test_noise_mode_requires_data_and_validates(self):
        fitted = np.arange(50, dtype=float)[None, :]
        with self.assertRaises(ValueError):
            _extract_changepoints_from_trend_batch(
                fitted, 2, 5, threshold_mode="noise"
            )
        with self.assertRaises(ValueError):
            _extract_changepoints_from_trend_batch(fitted, 2, 5, threshold_mode="bad")

    def test_get_new_params_includes_threshold_options(self):
        import random

        random.seed(0)
        modes = set()
        for _ in range(200):
            p = ChangepointDetector.get_new_params(method="l1_fused_lasso")
            mp = p.get("method_params", p)
            modes.add(mp.get("threshold_mode"))
        self.assertEqual(modes, {"relative", "noise"})

    def test_non_l1_method_deterministic(self):
        values = _noisy_piecewise_linear(seed=7)
        a = _detect(values, method="pelt", penalty=10)
        b = _detect(values, method="pelt", penalty=10)
        np.testing.assert_array_equal(a.changepoints_["a"], b.changepoints_["a"])


if __name__ == "__main__":
    unittest.main()
