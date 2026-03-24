# -*- coding: utf-8 -*-
"""
Unit tests for changepoint utilities.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autots.tools.changepoints import (
    ChangepointDetector,
    _build_ed_prefix_sums,
    _calculate_segment_cost,
    _detect_l0_trend_changepoints,
    _detect_pelt_changepoints,
    _detect_wbs2_changepoints,
    create_changepoint_features,
    find_market_changepoints_multivariate,
)

try:  # pragma: no cover - availability check only
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch may be optional
    TORCH_AVAILABLE = False


class TestChangepointFeatures(unittest.TestCase):
    """Tests for individual changepoint feature creation helpers."""

    def test_create_changepoint_features_basic(self):
        dt_index = pd.date_range("2020-01-01", "2021-01-01", freq="D")
        features = create_changepoint_features(dt_index, method="basic")

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all("changepoint" in col for col in features.columns))

    def test_create_changepoint_features_pelt(self):
        dt_index = pd.date_range("2020-01-01", periods=182, freq="D")
        data = np.concatenate([np.ones(100) * 5, np.ones(82) * 9])

        features = create_changepoint_features(
            dt_index,
            method="pelt",
            params={"penalty": 10, "loss_function": "l2"},
            data=data,
        )

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all("pelt_changepoint" in col for col in features.columns))

    def test_create_changepoint_features_l1(self):
        dt_index = pd.date_range("2020-01-01", periods=120, freq="D")
        data = np.concatenate([np.linspace(5, 6, 60), np.linspace(7, 8, 60)])

        features = create_changepoint_features(
            dt_index,
            method="l1_fused_lasso",
            params={"lambda_reg": 1.0},
            data=data,
        )

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(
            all("l1_fused_lasso_changepoint" in col for col in features.columns)
        )

    def test_create_changepoint_features_l1_adaptive_higher_order(self):
        dt_index = pd.date_range("2020-01-01", periods=180, freq="D")
        x = np.arange(180, dtype=float)
        data = np.where(
            x < 95,
            0.0025 * (x**2) + 2.0,
            0.0025 * ((x - 95.0) ** 2) + 14.0,
        )
        data = data + np.sin(x / 8.0) * 0.05

        features = create_changepoint_features(
            dt_index,
            method="l1_total_variation",
            params={
                "lambda_reg": 1.0,
                "difference_order": 3,
                "adaptive": True,
                "adaptive_gamma": 1.0,
                "irls_iterations": 4,
            },
            data=data,
        )

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(
            all("l1_total_variation_changepoint" in col for col in features.columns)
        )

    def test_create_changepoint_features_l0(self):
        dt_index = pd.date_range("2020-01-01", periods=160, freq="D")
        x = np.arange(160, dtype=float)
        data = np.piecewise(
            x,
            [x < 55, (x >= 55) & (x < 105), x >= 105],
            [
                lambda v: 0.08 * v + 5.0,
                lambda v: 0.01 * (v - 55.0) + 10.0,
                lambda v: -0.04 * (v - 105.0) + 10.5,
            ],
        )

        features = create_changepoint_features(
            dt_index,
            method="l0_trend_filter",
            params={
                "lambda_reg": 1.0,
                "difference_order": 2,
                "max_changepoints": 6,
                "min_segment_length": 8,
                "htp_iterations": 3,
            },
            data=data,
        )

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(
            all("l0_trend_filter_changepoint" in col for col in features.columns)
        )

    def test_create_changepoint_features_cusum(self):
        dt_index = pd.date_range("2021-01-01", periods=150, freq="D")
        data = np.concatenate([np.ones(75) * 2, np.ones(75) * 6])

        features = create_changepoint_features(
            dt_index,
            method="cusum",
            params={"threshold": 3.0, "min_distance": 10, "normalize": True},
            data=data,
        )

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all("cusum_changepoint" in col for col in features.columns))

    def test_create_changepoint_features_ewma(self):
        dt_index = pd.date_range("2021-01-01", periods=160, freq="D")
        data = np.concatenate([np.ones(80) * 3, np.ones(80) * 8])

        features = create_changepoint_features(
            dt_index,
            method="ewma",
            params={
                "lambda_param": 0.2,
                "control_limit": 3.0,
                "min_distance": 10,
                "normalize": True,
                "two_sided": True,
                "adaptive": True,
            },
            data=data,
        )

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all("ewma_changepoint" in col for col in features.columns))

    def test_create_changepoint_features_kcpd(self):
        dt_index = pd.date_range("2021-01-01", periods=180, freq="D")
        rng = np.random.default_rng(42)
        data = np.concatenate(
            [np.ones(60) * 2, np.ones(60) * 5, np.ones(60) * 9]
        ) + rng.normal(0, 0.1, 180)

        features = create_changepoint_features(
            dt_index,
            method="kcpd",
            params={
                "window_size": 20,
                "n_features": 16,
                "score_quantile": 0.9,
                "min_distance": 12,
                "max_changepoints": 8,
            },
            data=data,
        )

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all("kcpd_changepoint" in col for col in features.columns))

    def test_create_changepoint_features_bottom_up(self):
        dt_index = pd.date_range("2021-01-01", periods=200, freq="D")
        rng = np.random.default_rng(123)
        data = np.concatenate(
            [np.ones(70) * 3, np.ones(60) * 7, np.ones(70) * 10]
        ) + rng.normal(0, 0.15, 200)

        features = create_changepoint_features(
            dt_index,
            method="bottom_up",
            params={
                "initial_segment_length": 16,
                "penalty": "auto",
                "penalty_scale": 1.0,
                "max_changepoints": 10,
            },
            data=data,
        )

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all("bottom_up_changepoint" in col for col in features.columns))

    def test_create_changepoint_features_wbs2(self):
        dt_index = pd.date_range("2021-01-01", periods=180, freq="D")
        rng = np.random.default_rng(111)
        data = np.concatenate(
            [np.ones(60) * 2, np.ones(60) * 6, np.ones(60) * 10]
        ) + rng.normal(0, 0.2, 180)

        features = create_changepoint_features(
            dt_index,
            method="wbs2",
            params={
                "M": 100,
                "interval_sampling": "systematic",
                "universal": True,
                "lambda_param": 0.9,
                "th_const_min_mult": 0.3,
                "model_selection": "sdll",
                "max_changepoints": 10,
            },
            data=data,
        )

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all("wbs2_changepoint" in col for col in features.columns))

    @unittest.skipUnless(
        TORCH_AVAILABLE, "PyTorch required for autoencoder changepoint detection"
    )
    def test_create_changepoint_features_autoencoder(self):
        dt_index = pd.date_range("2020-01-01", periods=120, freq="D")
        segment_one = np.ones(60) * 3
        segment_two = np.ones(60) * 7
        noise = np.linspace(-0.1, 0.1, 120)
        data = np.concatenate([segment_one, segment_two]) + noise

        params = {
            "window_size": 8,
            "epochs": 2,
            "batch_size": 16,
            "latent_dim": 4,
            "contamination": 0.2,
            "use_anomaly_flags": True,
            "min_distance": 8,
        }
        features = create_changepoint_features(
            dt_index,
            method="autoencoder",
            params=params,
            data=data,
        )

        self.assertEqual(features.shape[0], len(dt_index))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(
            all("autoencoder_changepoint" in col for col in features.columns)
        )


class TestChangepointDetector(unittest.TestCase):
    """Tests for the ChangepointDetector class."""

    def setUp(self):
        np.random.seed(42)

    def test_changepoint_detector_basic(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        values = np.concatenate([np.ones(50) * 10, np.ones(50) * 15])
        df = pd.DataFrame({"series1": values}, index=dates)

        detector = ChangepointDetector(method="pelt", aggregate_method="individual")
        detector.detect(df)

        self.assertIsNotNone(detector.changepoints_)
        self.assertIn("series1", detector.changepoints_)
        self.assertEqual(detector.df.shape, df.shape)

    def test_changepoint_detector_features(self):
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        rng = np.random.default_rng(123)
        # Embed deterministic level shifts so the detector consistently emits features
        base_shift_one = np.concatenate([np.ones(30) * 10, np.ones(30) * 15])
        base_shift_two = np.concatenate([np.ones(30) * 12, np.ones(30) * 18])
        series_one = base_shift_one + rng.normal(0, 0.2, 60)
        series_two = base_shift_two + rng.normal(0, 0.2, 60)
        df = pd.DataFrame(
            {
                "series1": series_one,
                "series2": series_two,
            },
            index=dates,
        )

        detector = ChangepointDetector(
            method="pelt",
            aggregate_method="individual",
            method_params={"penalty": 5, "loss_function": "l2"},
        )
        detector.detect(df)

        features = detector.create_features(forecast_length=12)
        self.assertEqual(features.shape[0], 72)
        self.assertGreater(features.shape[1], 0)

    def test_changepoint_detector_cusum(self):
        dates = pd.date_range("2022-01-01", periods=140, freq="D")
        values = np.concatenate([np.ones(70) * 4, np.ones(70) * 9])
        df = pd.DataFrame({"series1": values}, index=dates)

        detector = ChangepointDetector(
            method="cusum",
            aggregate_method="individual",
            method_params={"threshold": 3.0, "min_distance": 10, "normalize": True},
        )
        detector.detect(df)

        self.assertIn("series1", detector.changepoints_)
        self.assertIsInstance(detector.changepoints_["series1"], np.ndarray)

    def test_changepoint_detector_ewma(self):
        # Use the existing setUp random seed
        dates = pd.date_range("2022-01-01", periods=200, freq="D")
        # Create a clear level shift with realistic noise
        segment1 = np.random.normal(5, 0.5, 100)
        segment2 = np.random.normal(12, 0.5, 100)  # Much larger shift
        values = np.concatenate([segment1, segment2])
        df = pd.DataFrame({"series1": values}, index=dates)

        detector = ChangepointDetector(
            method="ewma",
            aggregate_method="individual",
            method_params={
                "lambda_param": 0.3,  # Higher lambda for quicker response
                "control_limit": 2.5,  # More sensitive
                "min_distance": 10,
                "normalize": True,
                "two_sided": True,
                "adaptive": True,
            },
        )
        detector.detect(df)

        self.assertIn("series1", detector.changepoints_)
        self.assertIsInstance(detector.changepoints_["series1"], np.ndarray)
        # EWMA should detect at least one changepoint for this clear level shift
        self.assertGreater(len(detector.changepoints_["series1"]), 0)

    def test_changepoint_detector_l0_trend_filter(self):
        dates = pd.date_range("2022-01-01", periods=180, freq="D")
        x = np.arange(180, dtype=float)
        series1 = np.piecewise(
            x,
            [x < 70, (x >= 70) & (x < 120), x >= 120],
            [
                lambda v: 0.06 * v + 4.0,
                lambda v: 0.0 * v + 8.5,
                lambda v: -0.05 * (v - 120.0) + 8.5,
            ],
        )
        series2 = series1 * 1.2 + 0.3
        df = pd.DataFrame({"series1": series1, "series2": series2}, index=dates)

        detector = ChangepointDetector(
            method="l0_trend_filter",
            aggregate_method="individual",
            method_params={
                "lambda_reg": 1.0,
                "difference_order": 2,
                "max_changepoints": 6,
                "htp_iterations": 3,
                "hard_weight": 2500.0,
            },
        )
        detector.detect(df)

        self.assertIn("series1", detector.changepoints_)
        self.assertIn("series2", detector.changepoints_)
        self.assertIsInstance(detector.changepoints_["series1"], np.ndarray)
        self.assertGreater(len(detector.changepoints_["series1"]), 0)

    def test_changepoint_detector_kcpd(self):
        dates = pd.date_range("2022-01-01", periods=180, freq="D")
        values = np.concatenate(
            [np.ones(60) * 2, np.ones(60) * 6, np.ones(60) * 9]
        ) + np.random.normal(0, 0.2, 180)
        df = pd.DataFrame({"series1": values}, index=dates)

        detector = ChangepointDetector(
            method="kcpd",
            aggregate_method="individual",
            method_params={
                "window_size": 20,
                "n_features": 16,
                "min_distance": 10,
                "score_quantile": 0.9,
                "max_changepoints": 8,
            },
        )
        detector.detect(df)

        self.assertIn("series1", detector.changepoints_)
        self.assertIsInstance(detector.changepoints_["series1"], np.ndarray)
        self.assertGreater(len(detector.changepoints_["series1"]), 0)

    def test_changepoint_detector_bottom_up(self):
        dates = pd.date_range("2022-01-01", periods=200, freq="D")
        values = np.concatenate(
            [np.ones(70) * 4, np.ones(60) * 8, np.ones(70) * 11]
        ) + np.random.normal(0, 0.2, 200)
        df = pd.DataFrame({"series1": values}, index=dates)

        detector = ChangepointDetector(
            method="bottom_up",
            aggregate_method="individual",
            method_params={
                "initial_segment_length": 16,
                "penalty": "auto",
                "penalty_scale": 1.0,
                "max_changepoints": 10,
            },
        )
        detector.detect(df)

        self.assertIn("series1", detector.changepoints_)
        self.assertIsInstance(detector.changepoints_["series1"], np.ndarray)
        self.assertGreater(len(detector.changepoints_["series1"]), 0)

    def test_changepoint_detector_wbs2(self):
        dates = pd.date_range("2022-01-01", periods=210, freq="D")
        rng = np.random.default_rng(12)
        values = np.concatenate(
            [np.ones(70) * 3, np.ones(70) * 7, np.ones(70) * 12]
        ) + rng.normal(0, 0.2, 210)
        df = pd.DataFrame({"series1": values}, index=dates)

        detector = ChangepointDetector(
            method="wbs2",
            aggregate_method="individual",
            method_params={
                "M": 100,
                "interval_sampling": "systematic",
                "universal": True,
                "model_selection": "sdll",
                "max_changepoints": 10,
            },
        )
        detector.detect(df)

        self.assertIn("series1", detector.changepoints_)
        self.assertIsInstance(detector.changepoints_["series1"], np.ndarray)
        self.assertGreater(len(detector.changepoints_["series1"]), 0)

    def test_changepoint_detector_wbs2_transform_cycle(self):
        dates = pd.date_range("2022-01-01", periods=180, freq="D")
        rng = np.random.default_rng(101)
        base = np.concatenate([np.ones(60) * 1, np.ones(60) * 4, np.ones(60) * 9])
        df = pd.DataFrame(
            {
                "series1": base + rng.normal(0, 0.2, 180),
                "series2": (base * 1.3) + rng.normal(0, 0.2, 180),
            },
            index=dates,
        )

        detector = ChangepointDetector(
            method="wbs2",
            aggregate_method="mean",
            method_params={
                "M": 100,
                "interval_sampling": "systematic",
                "universal": True,
                "model_selection": "sdll",
            },
        )
        detector.fit(df)
        transformed = detector.transform(df)
        reconstructed = detector.inverse_transform(transformed)

        self.assertEqual(transformed.shape, df.shape)
        self.assertEqual(reconstructed.shape, df.shape)
        pd.testing.assert_frame_equal(df, reconstructed, atol=1e-8, check_dtype=False)

    def test_get_new_params_new_methods(self):
        kcpd_params = ChangepointDetector.get_new_params(method="kcpd")
        self.assertEqual(kcpd_params["method"], "kcpd")
        self.assertIn("window_size", kcpd_params["method_params"])
        self.assertIn("n_features", kcpd_params["method_params"])

        bottom_up_params = ChangepointDetector.get_new_params(method="bottom_up")
        self.assertEqual(bottom_up_params["method"], "bottom_up")
        self.assertIn("initial_segment_length", bottom_up_params["method_params"])
        self.assertIn("penalty_scale", bottom_up_params["method_params"])

        wbs2_params = ChangepointDetector.get_new_params(method="wbs2")
        self.assertEqual(wbs2_params["method"], "wbs2")
        self.assertIn("M", wbs2_params["method_params"])
        self.assertIn("model_selection", wbs2_params["method_params"])

        l0_params = ChangepointDetector.get_new_params(method="l0_trend_filter")
        self.assertEqual(l0_params["method"], "l0_trend_filter")
        self.assertIn("difference_order", l0_params["method_params"])
        self.assertIn("max_changepoints", l0_params["method_params"])
        self.assertIn("htp_iterations", l0_params["method_params"])

    @unittest.skipUnless(
        TORCH_AVAILABLE, "PyTorch required for autoencoder changepoint detection"
    )
    def test_changepoint_detector_autoencoder(self):
        dates = pd.date_range("2021-01-01", periods=120, freq="D")
        segment_one = np.ones(60) * 6
        segment_two = np.ones(60) * 9
        noise = np.linspace(0.0, 0.2, 120)
        df = pd.DataFrame(
            {"series1": np.concatenate([segment_one, segment_two]) + noise},
            index=dates,
        )

        params = {
            "window_size": 8,
            "epochs": 2,
            "batch_size": 16,
            "latent_dim": 4,
            "contamination": 0.2,
            "use_anomaly_flags": True,
            "min_distance": 8,
        }
        detector = ChangepointDetector(
            method="autoencoder",
            aggregate_method="individual",
            method_params=params,
        )
        detector.detect(df)

        self.assertIn("series1", detector.changepoints_)
        scores = detector.fitted_trends_["series1"]
        self.assertEqual(len(scores), len(df))

    def test_find_market_changepoints(self):
        dates = pd.date_range("2020-01-01", periods=90, freq="D")
        data1 = np.concatenate([np.ones(45) * 10, np.ones(45) * 15])
        data2 = np.concatenate([np.ones(45) * 12, np.ones(45) * 18])
        df = pd.DataFrame({"series1": data1, "series2": data2}, index=dates)

        results = find_market_changepoints_multivariate(
            df,
            detector_params={"method": "pelt", "aggregate_method": "individual"},
            clustering_method="agreement",
            clustering_params={"tolerance": 3},
            min_series_agreement=0.5,
        )

        self.assertIn("market_changepoints", results)
        self.assertIn("individual_changepoints", results)
        self.assertIn("detector", results)
        self.assertIsInstance(results["market_changepoints"], np.ndarray)
        self.assertIn("series1", results["individual_changepoints"])
        self.assertIn("series2", results["individual_changepoints"])

    def test_changepoint_detector_transformer_edge_cases(self):
        """Test transformer functionality with edge cases that previously caused failures."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        values = np.concatenate([np.ones(50) * 10, np.ones(50) * 15])
        # df = pd.DataFrame({"series1": values}, index=dates)

        # Test 1: Transform with future index
        dates_train = pd.date_range("2020-01-01", periods=50, freq="D")
        dates_test = pd.date_range("2020-02-20", periods=30, freq="D")
        df_train = pd.DataFrame({"series1": values[:50]}, index=dates_train)
        df_test = pd.DataFrame({"series1": np.ones(30) * 12}, index=dates_test)

        detector = ChangepointDetector(method="pelt", aggregate_method="mean")
        detector.fit(df_train)
        transformed = detector.transform(df_test)
        inverse = detector.inverse_transform(transformed)

        self.assertEqual(transformed.shape, df_test.shape)
        self.assertEqual(inverse.shape, df_test.shape)

        # Test 2: Transform with different columns (should use defaults)
        df_fit = pd.DataFrame({"A": np.arange(100), "B": np.arange(100) * 2}, index=dates)
        df_transform = pd.DataFrame(
            {"C": np.arange(100) * 3, "D": np.arange(100) * 4}, index=dates
        )

        detector2 = ChangepointDetector(method="pelt", aggregate_method="mean")
        detector2.fit(df_fit)
        transformed2 = detector2.transform(df_transform)
        inverse2 = detector2.inverse_transform(transformed2)

        self.assertEqual(transformed2.shape, df_transform.shape)
        self.assertEqual(inverse2.shape, df_transform.shape)

        # Test 3: Individual aggregate with new series in transform
        df_fit3 = pd.DataFrame(
            {"A": np.arange(100), "B": np.arange(100) * 2, "C": np.arange(100) * 3},
            index=dates,
        )
        df_transform3 = pd.DataFrame(
            {"A": np.arange(100), "D": np.arange(100) * 5}, index=dates
        )

        detector3 = ChangepointDetector(method="pelt", aggregate_method="individual")
        detector3.fit(df_fit3)
        transformed3 = detector3.transform(df_transform3)
        inverse3 = detector3.inverse_transform(transformed3)

        self.assertEqual(transformed3.shape, df_transform3.shape)
        self.assertEqual(inverse3.shape, df_transform3.shape)

        # Test 4: Very short series
        dates_tiny = pd.date_range("2020-01-01", periods=5, freq="D")
        df_tiny = pd.DataFrame({"series1": [1, 2, 3, 4, 5]}, index=dates_tiny)

        detector4 = ChangepointDetector(
            method="pelt", aggregate_method="mean", min_segment_length=5
        )
        detector4.fit(df_tiny)
        transformed4 = detector4.transform(df_tiny)
        inverse4 = detector4.inverse_transform(transformed4)

        self.assertEqual(transformed4.shape, df_tiny.shape)
        self.assertEqual(inverse4.shape, df_tiny.shape)

        # Test 5: Constant series (no changepoints detected)
        df_const = pd.DataFrame({"series1": np.ones(100) * 5}, index=dates)

        detector5 = ChangepointDetector(method="pelt", aggregate_method="mean")
        detector5.fit(df_const)
        transformed5 = detector5.transform(df_const)
        inverse5 = detector5.inverse_transform(transformed5)

        self.assertEqual(transformed5.shape, df_const.shape)
        self.assertEqual(inverse5.shape, df_const.shape)
        # Should return to approximately original values
        diff = np.abs(df_const - inverse5).max().max()
        self.assertLess(diff, 1e-10)

    def test_changepoint_detector_transformer_load_daily(self):
        """Test ChangepointDetector as a transformer with load_daily data."""
        from autots.datasets import load_daily

        df = load_daily(long=False)

        # Test with a robust method like pelt
        detector = ChangepointDetector(
            method="pelt",
            aggregate_method="individual",
            method_params={"penalty": 10, "loss_function": "l2"},
        )

        # Following transformer patterns: get_new_params, fit, transform, inverse_transform
        params = detector.get_new_params()
        self.assertTrue(params)

        detector.fit(df)
        transformed = detector.transform(df)
        reconstructed = detector.inverse_transform(transformed)

        # Assertions
        self.assertEqual(transformed.shape, df.shape)
        self.assertEqual(reconstructed.shape, df.shape)
        # Check reconstruction accuracy
        pd.testing.assert_frame_equal(df, reconstructed, atol=1e-10, check_dtype=False)

        # At least one changepoint should be found in at least one series
        total_cps = sum(len(cps) for cps in detector.changepoints_.values())
        self.assertGreater(
            total_cps, 0, "No changepoints found in any series of load_daily"
        )


class TestEdPelt(unittest.TestCase):
    """Tests for ED-PELT (energy-distance cost function) changepoint detection."""

    def _brute_ed_cost(self, data, s, t):
        """Reference O(n^2) computation of sum_{i,j in [s,t)} |x_i - x_j|."""
        seg = data[s:t]
        return float(np.sum(np.abs(seg[:, None] - seg[None, :])))

    # ------------------------------------------------------------------
    # _build_ed_prefix_sums correctness
    # ------------------------------------------------------------------

    def test_prefix_sums_small_known(self):
        """Prefix-sum matrix gives the correct cost for a hand-checked example."""
        data = np.array([1.0, 3.0, 6.0])
        P = _build_ed_prefix_sums(data)
        # Full segment: |1-1|+|1-3|+|1-6|+|3-1|+|3-3|+|3-6|+|6-1|+|6-3|+|6-6|
        #             = 0+2+5+2+0+3+5+3+0 = 20
        cost_full = P[3, 3] - P[0, 3] - P[3, 0] + P[0, 0]
        self.assertAlmostEqual(cost_full, 20.0, places=6)
        # Sub-segment [0,2): {1,3}  -> 0+2+2+0 = 4
        cost_01 = P[2, 2] - P[0, 2] - P[2, 0] + P[0, 0]
        self.assertAlmostEqual(cost_01, 4.0, places=6)

    def test_prefix_sums_agrees_with_brute_force(self):
        """Prefix-sum matrix agrees with the brute-force double-loop on random data."""
        rng = np.random.default_rng(7)
        data = rng.normal(0, 1, 50).astype(float)
        P = _build_ed_prefix_sums(data)
        for s, t in [(0, 50), (0, 20), (20, 50), (10, 30), (5, 15)]:
            fast = float(P[t, t] - P[s, t] - P[t, s] + P[s, s])
            slow = self._brute_ed_cost(data, s, t)
            self.assertAlmostEqual(fast, slow, places=5, msg=f"segment [{s},{t})")

    def test_prefix_sums_single_element_zero(self):
        """A one-element segment has zero energy-distance cost."""
        data = np.array([5.0, 9.0, 2.0])
        P = _build_ed_prefix_sums(data)
        for i in range(len(data)):
            cost = float(P[i + 1, i + 1] - P[i, i + 1] - P[i + 1, i] + P[i, i])
            self.assertAlmostEqual(cost, 0.0, places=10)

    # ------------------------------------------------------------------
    # _calculate_segment_cost 'ed' correctness
    # ------------------------------------------------------------------

    def test_segment_cost_ed_matches_brute_force(self):
        """_calculate_segment_cost 'ed' matches the brute-force full-matrix sum."""
        rng = np.random.default_rng(13)
        data = rng.normal(0, 2, 40).astype(float)
        for s, t in [(0, 10), (10, 25), (5, 40)]:
            fast = _calculate_segment_cost(data[s:t], "ed")
            slow = self._brute_ed_cost(data, s, t)
            self.assertAlmostEqual(fast, slow, places=5, msg=f"segment [{s},{t})")

    def test_segment_cost_ed_single_element(self):
        self.assertAlmostEqual(_calculate_segment_cost(np.array([3.0]), "ed"), 0.0, places=10)

    def test_segment_cost_ed_two_elements(self):
        seg = np.array([2.0, 8.0])
        # full matrix: 0+6+6+0 = 12
        self.assertAlmostEqual(_calculate_segment_cost(seg, "ed"), 12.0, places=6)

    def test_prefix_sums_consistent_with_segment_cost(self):
        """_build_ed_prefix_sums and _calculate_segment_cost agree for all sub-segments."""
        rng = np.random.default_rng(99)
        data = rng.normal(0, 1, 30).astype(float)
        P = _build_ed_prefix_sums(data)
        for s in range(0, 25, 5):
            for t in range(s + 2, 30, 5):
                via_prefix = float(P[t, t] - P[s, t] - P[t, s] + P[s, s])
                via_cost = _calculate_segment_cost(data[s:t], "ed")
                self.assertAlmostEqual(
                    via_prefix, via_cost, places=5,
                    msg=f"[{s},{t}) prefix={via_prefix:.4f} cost={via_cost:.4f}",
                )

    # ------------------------------------------------------------------
    # _detect_pelt_changepoints with loss_function='ed'
    # ------------------------------------------------------------------

    def test_ed_pelt_detects_level_shift(self):
        """ED-PELT reliably detects a strong level shift near the midpoint."""
        rng = np.random.default_rng(42)
        signal = np.concatenate([rng.normal(0, 1, 50), rng.normal(8, 1, 50)])
        cps = _detect_pelt_changepoints(signal, penalty=100, loss_function="ed")
        self.assertGreater(len(cps), 0, "No changepoints detected on a clear level shift")
        self.assertTrue(
            any(38 <= cp <= 62 for cp in cps),
            f"None of the detected changepoints are near the true shift at 50: {cps}",
        )

    def test_ed_pelt_no_changepoints_on_constant(self):
        """ED-PELT should not fragment a perfectly constant series."""
        signal = np.ones(80) * 5.0
        cps = _detect_pelt_changepoints(signal, penalty=200, loss_function="ed")
        self.assertEqual(len(cps), 0, f"Spurious changepoints on constant series: {cps}")

    def test_ed_pelt_short_series(self):
        """ED-PELT returns empty array when series is shorter than 2*min_segment_length."""
        cps = _detect_pelt_changepoints(np.array([1.0, 2.0, 3.0]), penalty=10,
                                        loss_function="ed", min_segment_length=5)
        self.assertEqual(len(cps), 0)

    def test_ed_pelt_batch_vs_scalar_paths_agree(self):
        """Vectorised batch path (|R|>10) and scalar path produce same changepoints."""
        rng = np.random.default_rng(55)
        # A longer signal forces ED-PELT to use both paths.
        signal = np.concatenate([rng.normal(i * 4, 0.5, 40) for i in range(4)])
        # Use a very low min_segment_length so R grows quickly and batch path is exercised.
        cps = _detect_pelt_changepoints(signal, penalty=200, loss_function="ed",
                                        min_segment_length=1)
        self.assertIsInstance(cps, np.ndarray)
        self.assertGreater(len(cps), 0)

    def test_ed_pelt_pruning_factor(self):
        """Aggressive pruning (pruning_factor>1) should still find the main changepoint."""
        rng = np.random.default_rng(77)
        signal = np.concatenate([rng.normal(0, 1, 60), rng.normal(7, 1, 60)])
        cps_std = _detect_pelt_changepoints(signal, penalty=200, loss_function="ed",
                                             pruning_factor=1.0)
        cps_agg = _detect_pelt_changepoints(signal, penalty=200, loss_function="ed",
                                             pruning_factor=3.0)
        # Both should find the shift; aggressive pruning may return fewer changepoints
        self.assertGreater(len(cps_std), 0)
        self.assertGreater(len(cps_agg), 0)

    # ------------------------------------------------------------------
    # create_changepoint_features integration
    # ------------------------------------------------------------------

    def test_create_features_pelt_ed(self):
        """create_changepoint_features with method='pelt' and loss_function='ed'."""
        dt_index = pd.date_range("2021-01-01", periods=100, freq="D")
        rng = np.random.default_rng(3)
        data = np.concatenate([rng.normal(0, 1, 50), rng.normal(6, 1, 50)])
        features = create_changepoint_features(
            dt_index,
            method="pelt",
            data=data,
            params={"loss_function": "ed", "penalty": 100, "min_segment_length": 3,
                    "pruning_factor": 1.0},
        )
        self.assertEqual(features.shape[0], 100)
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all("pelt_changepoint" in col for col in features.columns))

    def test_create_features_pelt_ed_pruning_factor_passthrough(self):
        """pruning_factor is correctly forwarded; both values produce valid DataFrames."""
        dt_index = pd.date_range("2021-01-01", periods=80, freq="D")
        rng = np.random.default_rng(5)
        data = np.concatenate([rng.normal(0, 0.5, 40), rng.normal(5, 0.5, 40)])
        for pf in [1.0, 2.5]:
            features = create_changepoint_features(
                dt_index,
                method="pelt",
                data=data,
                params={"loss_function": "ed", "penalty": 80, "pruning_factor": pf},
            )
            self.assertEqual(features.shape[0], 80, msg=f"pruning_factor={pf}")

    # ------------------------------------------------------------------
    # ChangepointDetector integration
    # ------------------------------------------------------------------

    def test_changepoint_detector_pelt_ed(self):
        """ChangepointDetector with method='pelt' and loss_function='ed' end-to-end."""
        rng = np.random.default_rng(21)
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        signal = np.concatenate([rng.normal(0, 1, 50), rng.normal(7, 1, 50)])
        df = pd.DataFrame({"A": signal}, index=dates)
        cd = ChangepointDetector(
            method="pelt",
            aggregate_method="individual",
            method_params={"loss_function": "ed", "penalty": 100},
        )
        cd.detect(df)
        self.assertIn("A", cd.changepoints_)
        self.assertIsInstance(cd.changepoints_["A"], np.ndarray)
        self.assertGreater(len(cd.changepoints_["A"]), 0)

    def test_changepoint_detector_pelt_ed_transform_cycle(self):
        """fit → transform → inverse_transform round-trips cleanly for 'ed' cost."""
        rng = np.random.default_rng(34)
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        signal = np.concatenate([rng.normal(0, 0.3, 50), rng.normal(5, 0.3, 50)])
        df = pd.DataFrame({"A": signal, "B": signal * 0.5 + 1}, index=dates)
        cd = ChangepointDetector(
            method="pelt",
            method_params={"loss_function": "ed", "penalty": 80},
        )
        cd.fit(df)
        transformed = cd.transform(df)
        reconstructed = cd.inverse_transform(transformed)
        self.assertEqual(transformed.shape, df.shape)
        self.assertEqual(reconstructed.shape, df.shape)
        pd.testing.assert_frame_equal(df, reconstructed, atol=1e-8, check_dtype=False)

    # ------------------------------------------------------------------
    # get_new_params includes 'ed' and scales penalty
    # ------------------------------------------------------------------

    def test_get_new_params_pelt_ed_penalty_scaled(self):
        """When 'ed' is drawn, penalty is scaled ×5 relative to the base sample."""
        # get_new_params(method='pelt') uses 'random' selection_mode (full param space).
        # 'ed' weight is 0.08; 100 trials gives P(never hit) = 0.92^100 < 0.01%.
        found_ed = False
        import random
        random.seed(0)
        for _ in range(300):
            params = ChangepointDetector.get_new_params(method="pelt")
            mp = params.get("method_params", {})
            if mp.get("loss_function") == "ed":
                self.assertGreaterEqual(
                    mp.get("penalty", 0), 50,
                    f"ED penalty not scaled up: {mp}",
                )
                found_ed = True
                break
        self.assertTrue(found_ed, "get_new_params never returned loss_function='ed' in 300 tries")


class TestL0TrendFiltering(unittest.TestCase):
    """Tests for L0 trend filtering changepoint detection."""

    def test_l0_constant_series_returns_no_changepoints(self):
        signal = np.ones(160, dtype=float) * 7.5
        cps, fitted = _detect_l0_trend_changepoints(
            signal,
            lambda_reg=1.0,
            difference_order=2,
            max_changepoints='auto',
            min_segment_length=8,
            htp_iterations=3,
        )
        self.assertEqual(len(cps), 0, f"Unexpected changepoints on constant series: {cps}")
        self.assertEqual(len(fitted), len(signal))

    def test_l0_detects_structural_shift(self):
        x = np.arange(180, dtype=float)
        signal = np.piecewise(
            x,
            [x < 75, (x >= 75) & (x < 125), x >= 125],
            [
                lambda v: 0.04 * v + 2.0,
                lambda v: 0.0 * v + 8.0,
                lambda v: -0.06 * (v - 125.0) + 8.0,
            ],
        )
        cps, fitted = _detect_l0_trend_changepoints(
            signal,
            lambda_reg=1.0,
            difference_order=2,
            max_changepoints=6,
            min_segment_length=8,
            htp_iterations=3,
        )
        self.assertIsInstance(cps, np.ndarray)
        self.assertEqual(len(fitted), len(signal))
        self.assertGreater(len(cps), 0)
        self.assertTrue(any(65 <= cp <= 85 for cp in cps), f"cps={cps}")

    def test_l0_detector_vectorized_handles_constant_series(self):
        dates = pd.date_range("2022-01-01", periods=140, freq="D")
        x = np.arange(140, dtype=float)
        changing = np.piecewise(
            x,
            [x < 70, x >= 70],
            [lambda v: 0.05 * v + 1.0, lambda v: -0.03 * (v - 70.0) + 4.5],
        )
        constant = np.ones(140, dtype=float) * 3.0
        df = pd.DataFrame({"changing": changing, "constant": constant}, index=dates)
        cd = ChangepointDetector(
            method="l0_trend_filter",
            aggregate_method="individual",
            method_params={
                "lambda_reg": 1.0,
                "difference_order": 2,
                "max_changepoints": 6,
                "htp_iterations": 3,
            },
        )
        cd.detect(df)
        self.assertEqual(len(cd.changepoints_["constant"]), 0)
        self.assertGreaterEqual(len(cd.changepoints_["changing"]), 1)


class TestWbs2(unittest.TestCase):
    """Tests for WBS2 changepoint detection."""

    def test_wbs2_detects_clear_shift(self):
        rng = np.random.default_rng(42)
        signal = np.concatenate([rng.normal(0, 0.4, 80), rng.normal(7, 0.4, 80)])
        cps, fitted = _detect_wbs2_changepoints(
            signal,
            min_segment_length=5,
            M=100,
            interval_sampling='systematic',
            random_state=42,
            universal=True,
            model_selection='sdll',
            max_changepoints=10,
        )
        self.assertIsInstance(cps, np.ndarray)
        self.assertGreater(len(cps), 0)
        self.assertTrue(
            any(65 <= cp <= 95 for cp in cps),
            f"WBS2 failed to detect main shift near 80: {cps}",
        )
        self.assertEqual(len(fitted), len(signal))

    def test_wbs2_constant_series_returns_no_changepoints(self):
        signal = np.ones(120) * 5.0
        cps, fitted = _detect_wbs2_changepoints(
            signal,
            min_segment_length=5,
            M=100,
            universal=True,
            model_selection='sdll',
        )
        self.assertEqual(len(cps), 0)
        self.assertEqual(len(fitted), len(signal))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
