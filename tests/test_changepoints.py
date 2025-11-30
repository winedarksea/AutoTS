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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
