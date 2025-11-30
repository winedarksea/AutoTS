# -*- coding: utf-8 -*-
"""Test anomalies.
Created on Mon Jul 18 16:27:48 2022

@author: Colin
"""
import unittest
import numpy as np
import pandas as pd
from autots.tools.anomaly_utils import available_methods, fast_methods
from autots.evaluator.anomaly_detector import AnomalyDetector, HolidayDetector
from autots.datasets import load_live_daily
from autots.tools.transform import expanding_transformers


def dict_loop(params):
    if 'transform_dict' in params.keys():
        x = params.get('transform_dict', {})
        if isinstance(x, dict):
            x = x.get('transformations', {})
            return x
    elif 'anomaly_detector_params' in params.keys():
        x = params.get('anomaly_detector_params', {})
        if isinstance(x, dict):
            x = x.get('transform_dict', {})
            if isinstance(x, dict):
                x = x.get('transformations', {})
                return x
    return {}


class TestAnomalies(unittest.TestCase):
    @classmethod
    def setUp(self):
        wiki_pages = [
            "Standard_deviation",  # anti-holiday
            "Christmas",
            "Thanksgiving",  # specific holiday
            "all",
        ]
        self.df = (
            load_live_daily(
                long=False,
                fred_series=None,
                tickers=None,
                trends_list=None,
                earthquake_min_magnitude=None,
                weather_stations=None,
                london_air_stations=None,
                gov_domain_list=None,
                weather_event_types=None,
                wikipedia_pages=wiki_pages,
                caiso_query=None,
                sleep_seconds=10,
            )
            .fillna(0)
            .replace(np.inf, 0)
        )

    def test_anomaly_holiday_detectors(self):
        """Combining these to minimize live data download."""
        print("Starting test_anomaly_holiday_detectors")
        tried = []
        # Check if PyTorch is available for VAEOutlier
        try:
            import torch

            torch_available = True
        except ImportError:
            torch_available = False

        # Filter methods based on torch availability
        test_methods = [
            m for m in available_methods if m != 'VAEOutlier' or torch_available
        ]

        while not all(x in tried for x in test_methods):
            params = AnomalyDetector.get_new_params(method="deep")
            # remove 'Slice' as it messes up assertions
            while any(
                item in dict_loop(params).values() for item in expanding_transformers
            ):
                params = AnomalyDetector.get_new_params(method="deep")
            with self.subTest(i=params['method']):
                print(
                    f"Starting subtest test_anomaly_holiday_detectors method={params['method']}"
                )
                tried.append(params['method'])
                mod = AnomalyDetector(output='multivariate', **params)
                num_cols = 2
                mod.detect(
                    self.df[np.random.choice(self.df.columns, num_cols, replace=False)]
                )
                # detected = mod.anomalies
                # print(params)
                # mod.plot()
                self.assertEqual(
                    mod.anomalies.shape,
                    (self.df.shape[0], num_cols),
                    msg=f"from params {params}",
                )

                mod = AnomalyDetector(output='univariate', **params)
                mod.detect(
                    self.df[np.random.choice(self.df.columns, num_cols, replace=False)]
                )
                self.assertEqual(mod.anomalies.shape, (self.df.shape[0], 1))
        # mod.plot()

        from prophet import Prophet

        tried = []
        forecast_length = 28
        holidays_detected = 0
        full_dates = self.df.index.union(
            pd.date_range(self.df.index.max(), freq="D", periods=forecast_length)
        )

        while not all(x in tried for x in fast_methods):
            params = HolidayDetector.get_new_params(method="fast")
            with self.subTest(i=params["anomaly_detector_params"]['method']):
                print(
                    f"Starting subtest test_anomaly_holiday_detectors method={params['anomaly_detector_params']['method']}"
                )
                while any(
                    item in dict_loop(params).values()
                    for item in expanding_transformers
                ):
                    params = HolidayDetector.get_new_params(method="fast")
                tried.append(params['anomaly_detector_params']['method'])
                mod = HolidayDetector(**params)
                mod.detect(self.df.copy())
                prophet_holidays = mod.dates_to_holidays(full_dates, style="prophet")

                for series in self.df.columns:
                    # series = "wiki_George_Washington"
                    holiday_subset = prophet_holidays[
                        prophet_holidays['series'] == series
                    ]
                    if holiday_subset.shape[0] >= 1:
                        holidays_detected = 1
                    m = Prophet(holidays=holiday_subset)
                    # m = Prophet()
                    m.fit(pd.DataFrame({'ds': self.df.index, 'y': self.df[series]}))
                    future = m.make_future_dataframe(forecast_length)
                    fcst = m.predict(future).set_index('ds')  # noqa
                    # m.plot_components(fcst)
        # mod.plot()
        temp = mod.dates_to_holidays(full_dates, style="flag")
        temp = mod.dates_to_holidays(full_dates, style="series_flag")
        temp = mod.dates_to_holidays(full_dates, style="impact")
        temp = mod.dates_to_holidays(full_dates, style="long")  # noqa
        # this is a weak test, but will capture some functionality
        self.assertEqual(holidays_detected, 1, "no methods detected holidays")


class TestVAEAnomalies(unittest.TestCase):
    """Test VAE anomaly detection functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')

        # Create normal data with patterns
        data = {
            'series_1': np.random.normal(100, 10, 200),
            'series_2': np.random.normal(50, 5, 200)
            + 10 * np.sin(np.arange(200) * 2 * np.pi / 7),
            'series_3': np.random.normal(75, 8, 200)
            + 5 * np.cos(np.arange(200) * 2 * np.pi / 30),
        }

        # Add some anomalies
        data['series_1'][50] = 200  # spike
        data['series_1'][51] = 210  # spike
        data['series_2'][100] = 0  # drop
        data['series_3'][150] = 150  # spike

        cls.df = pd.DataFrame(data, index=dates)

    def test_vae_availability(self):
        """Test that VAEOutlier is available in the methods."""
        print("Starting test_vae_availability")
        from autots.tools.anomaly_utils import available_methods

        self.assertIn(
            "VAEOutlier", available_methods, "VAEOutlier not found in available methods"
        )

    def test_vae_parameter_generation(self):
        """Test parameter generation for VAEOutlier."""
        print("Starting test_vae_parameter_generation")
        params = AnomalyDetector.get_new_params(method="VAEOutlier")
        self.assertEqual(params['method'], 'VAEOutlier', "Method should be VAEOutlier")
        self.assertIn('method_params', params, "method_params should be present")

        # Check that key parameters are present
        method_params = params['method_params']
        expected_params = [
            'depth',
            'batch_size',
            'epochs',
            'learning_rate',
            'loss_function',
            'dropout_rate',
            'beta',
            'contamination',
        ]
        for param in expected_params:
            self.assertIn(param, method_params, f"Parameter {param} should be present")

    def test_vae_anomaly_detection(self):
        """Test VAE anomaly detection functionality."""
        print("Starting test_vae_anomaly_detection")
        try:
            import torch

            torch_available = True
        except ImportError:
            self.skipTest("PyTorch not available, skipping VAE test")

        detector = AnomalyDetector(
            method="VAEOutlier",
            method_params={
                'depth': 1,
                'batch_size': 32,
                'epochs': 10,  # small for testing
                'learning_rate': 1e-3,
                'loss_function': 'elbo',
                'contamination': 0.1,
                'random_state': 42,
            },
        )

        anomalies, scores = detector.detect(self.df)

        # Check output shapes
        self.assertEqual(
            anomalies.shape, self.df.shape, "Anomalies shape should match input"
        )
        self.assertEqual(scores.shape, self.df.shape, "Scores shape should match input")

        # Check that some anomalies were detected
        num_anomalies = np.sum((anomalies == -1).values)
        self.assertGreater(num_anomalies, 0, "Should detect at least some anomalies")
        self.assertLess(
            num_anomalies, len(self.df) * 0.5, "Should not detect too many anomalies"
        )

        # Check that values are in expected range
        self.assertTrue(
            np.all(np.isin(anomalies.values, [-1, 1])), "Anomalies should be -1 or 1"
        )
        self.assertTrue(np.all(scores.values >= 0), "Scores should be non-negative")

    def test_vae_univariate_detection(self):
        """Test VAE anomaly detection in univariate mode."""
        print("Starting test_vae_univariate_detection")
        try:
            import torch

            torch_available = True
        except ImportError:
            self.skipTest("PyTorch not available, skipping VAE test")

        detector = AnomalyDetector(
            output='univariate',
            method="VAEOutlier",
            method_params={
                'depth': 1,
                'batch_size': 32,
                'epochs': 10,
                'learning_rate': 1e-3,
                'contamination': 0.1,
                'random_state': 42,
            },
        )

        anomalies, scores = detector.detect(self.df)

        # Check output shapes for univariate
        self.assertEqual(
            anomalies.shape,
            (len(self.df), 1),
            "Univariate anomalies should have single column",
        )
        self.assertEqual(
            scores.shape,
            (len(self.df), 1),
            "Univariate scores should have single column",
        )

    def test_vae_different_loss_functions(self):
        """Test VAE with different loss functions."""
        print("Starting test_vae_different_loss_functions")
        try:
            import torch

            torch_available = True
        except ImportError:
            self.skipTest("PyTorch not available, skipping VAE test")

        loss_functions = ['elbo', 'mse', 'lmse']

        for loss_func in loss_functions:
            with self.subTest(loss_function=loss_func):
                print(
                    f"Starting subtest test_vae_different_loss_functions loss_function={loss_func}"
                )
                detector = AnomalyDetector(
                    method="VAEOutlier",
                    method_params={
                        'depth': 1,
                        'batch_size': 32,
                        'epochs': 5,  # very small for testing
                        'learning_rate': 1e-3,
                        'loss_function': loss_func,
                        'contamination': 0.1,
                        'random_state': 42,
                    },
                )

                try:
                    anomalies, scores = detector.detect(self.df)
                    self.assertEqual(anomalies.shape, self.df.shape)
                    self.assertTrue(np.all(np.isin(anomalies.values, [-1, 1])))
                except Exception as e:
                    self.fail(f"VAE with {loss_func} loss failed: {e}")

    def test_vae_depth_parameter(self):
        """Test VAE with different depth parameters."""
        print("Starting test_vae_depth_parameter")
        try:
            import torch

            torch_available = True
        except ImportError:
            self.skipTest("PyTorch not available, skipping VAE test")

        for depth in [1, 2]:
            with self.subTest(depth=depth):
                print(f"Starting subtest test_vae_depth_parameter depth={depth}")
                detector = AnomalyDetector(
                    method="VAEOutlier",
                    method_params={
                        'depth': depth,
                        'batch_size': 32,
                        'epochs': 5,
                        'learning_rate': 1e-3,
                        'contamination': 0.1,
                        'random_state': 42,
                    },
                )

                try:
                    anomalies, scores = detector.detect(self.df)
                    self.assertEqual(anomalies.shape, self.df.shape)
                except Exception as e:
                    self.fail(f"VAE with depth {depth} failed: {e}")

    def test_vae_anomaly_removal_transformer(self):
        """Test AnomalyRemoval transformer with VAE method."""
        print("Starting test_vae_anomaly_removal_transformer")
        try:
            import torch

            torch_available = True
        except ImportError:
            self.skipTest("PyTorch not available, skipping VAE test")

        from autots.tools.transform import AnomalyRemoval

        transformer = AnomalyRemoval(
            method="VAEOutlier",
            method_params={
                'depth': 1,
                'batch_size': 32,
                'epochs': 5,
                'learning_rate': 1e-3,
                'contamination': 0.1,
                'random_state': 42,
            },
            fillna="mean",
        )

        try:
            cleaned_df = transformer.fit_transform(self.df)
            # Should have fewer or equal rows (anomalies removed)
            self.assertLessEqual(len(cleaned_df), len(self.df))
            # Should have same number of columns
            self.assertEqual(cleaned_df.shape[1], self.df.shape[1])
        except Exception as e:
            self.fail(f"AnomalyRemoval with VAE failed: {e}")


class TestAnomalyRemoval(unittest.TestCase):
    """Test AnomalyRemoval transformer functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')

        # Create normal data with patterns
        data = {
            'series_1': np.random.normal(100, 10, 200),
            'series_2': np.random.normal(50, 5, 200)
            + 10 * np.sin(np.arange(200) * 2 * np.pi / 7),
            'series_3': np.random.normal(75, 8, 200)
            + 5 * np.cos(np.arange(200) * 2 * np.pi / 30),
        }

        # Add some anomalies
        data['series_1'][50] = 200  # spike
        data['series_1'][51] = 210  # spike
        data['series_2'][100] = 0  # drop
        data['series_3'][150] = 150  # spike

        cls.df = pd.DataFrame(data, index=dates)

    def test_anomaly_removal_basic(self):
        """Test basic AnomalyRemoval functionality."""
        print("Starting test_anomaly_removal_basic")
        from autots.tools.transform import AnomalyRemoval

        transformer = AnomalyRemoval(
            method="zscore", method_params={"distribution": "norm", "alpha": 0.05}
        )

        result = transformer.fit_transform(self.df)

        # Should have fewer or equal rows (anomalies removed)
        self.assertLessEqual(len(result), len(self.df))
        # Should have same number of columns
        self.assertEqual(result.shape[1], self.df.shape[1])
        # Should have anomalies and scores attributes
        self.assertTrue(hasattr(transformer, 'anomalies'))
        self.assertTrue(hasattr(transformer, 'scores'))

    def test_anomaly_removal_with_fillna(self):
        """Test AnomalyRemoval with different fillna methods."""
        print("Starting test_anomaly_removal_with_fillna")
        from autots.tools.transform import AnomalyRemoval

        fillna_methods = ["ffill", "mean", "linear"]

        for fillna_method in fillna_methods:
            with self.subTest(fillna=fillna_method):
                print(
                    f"Starting subtest test_anomaly_removal_with_fillna fillna={fillna_method}"
                )
                transformer = AnomalyRemoval(
                    method="IQR",
                    method_params={"iqr_threshold": 2.0},
                    fillna=fillna_method,
                )

                result = transformer.fit_transform(self.df)

                # Should not have NaN values after fillna
                self.assertFalse(
                    result.isna().any().any(), f"NaN found with fillna={fillna_method}"
                )
                # Should have same shape or fewer rows
                self.assertLessEqual(len(result), len(self.df))
                self.assertEqual(result.shape[1], self.df.shape[1])

    def test_anomaly_removal_isolated_only(self):
        """Test AnomalyRemoval with isolated_only parameter."""
        print("Starting test_anomaly_removal_isolated_only")
        from autots.tools.transform import AnomalyRemoval

        # Without isolated_only
        transformer1 = AnomalyRemoval(
            method="zscore",
            method_params={"distribution": "norm", "alpha": 0.05},
            isolated_only=False,
        )
        result1 = transformer1.fit_transform(self.df)

        # With isolated_only
        transformer2 = AnomalyRemoval(
            method="zscore",
            method_params={"distribution": "norm", "alpha": 0.05},
            isolated_only=True,
        )
        result2 = transformer2.fit_transform(self.df)

        # isolated_only should generally remove fewer anomalies
        self.assertGreaterEqual(len(result2), len(result1))

    def test_anomaly_removal_multivariate_vs_univariate(self):
        """Test AnomalyRemoval with multivariate and univariate output."""
        print("Starting test_anomaly_removal_multivariate_vs_univariate")
        from autots.tools.transform import AnomalyRemoval

        # Multivariate
        transformer_multi = AnomalyRemoval(output='multivariate', method="IQR")
        result_multi = transformer_multi.fit_transform(self.df)

        # Univariate
        transformer_uni = AnomalyRemoval(output='univariate', method="IQR")
        result_uni = transformer_uni.fit_transform(self.df)

        # Both should return valid results
        self.assertLessEqual(len(result_multi), len(self.df))
        self.assertLessEqual(len(result_uni), len(self.df))
        self.assertEqual(result_multi.shape[1], self.df.shape[1])
        self.assertEqual(result_uni.shape[1], self.df.shape[1])

    def test_anomaly_removal_inverse_transform(self):
        """Test AnomalyRemoval inverse_transform."""
        print("Starting test_anomaly_removal_inverse_transform")
        from autots.tools.transform import AnomalyRemoval

        # Test with on_inverse=False (default)
        transformer1 = AnomalyRemoval(method="zscore", on_inverse=False)
        transformer1.fit(self.df)
        result1 = transformer1.inverse_transform(self.df)

        # Should return unchanged data
        pd.testing.assert_frame_equal(result1, self.df)

        # Test with on_inverse=True
        transformer2 = AnomalyRemoval(method="zscore", on_inverse=True)
        transformer2.fit(self.df)
        result2 = transformer2.inverse_transform(self.df)

        # Should apply fit_transform on inverse
        self.assertLessEqual(len(result2), len(self.df))

    def test_anomaly_removal_fit_anomaly_classifier(self):
        """Test fit_anomaly_classifier and score_to_anomaly methods."""
        print("Starting test_anomaly_removal_fit_anomaly_classifier")
        from autots.tools.transform import AnomalyRemoval

        transformer = AnomalyRemoval(method="IQR")
        transformer.fit(self.df)

        # Fit classifier
        transformer.fit_anomaly_classifier()

        # Should have classifier and categories
        self.assertIsNotNone(transformer.anomaly_classifier)
        self.assertIsNotNone(transformer.score_categories)

        # Test score_to_anomaly
        new_scores = transformer.scores.copy()
        anomaly_pred = transformer.score_to_anomaly(new_scores)

        # Should return valid anomaly classifications
        self.assertEqual(anomaly_pred.shape, new_scores.shape)
        self.assertTrue(all(np.isin(anomaly_pred.values.flatten(), [-1, 1])))

    def test_anomaly_removal_get_new_params(self):
        """Test get_new_params static method."""
        print("Starting test_anomaly_removal_get_new_params")
        from autots.tools.transform import AnomalyRemoval

        # Test different methods
        for method in ["random", "fast"]:
            with self.subTest(method=method):
                print(
                    f"Starting subtest test_anomaly_removal_get_new_params method={method}"
                )
                params = AnomalyRemoval.get_new_params(method=method)

                # Should have required keys
                self.assertIn('method', params)
                self.assertIn('method_params', params)
                self.assertIn('fillna', params)
                self.assertIn('isolated_only', params)
                self.assertIn('on_inverse', params)

    def test_anomaly_removal_with_transform_dict(self):
        """Test AnomalyRemoval with transform_dict preprocessing."""
        print("Starting test_anomaly_removal_with_transform_dict")
        from autots.tools.transform import AnomalyRemoval

        transformer = AnomalyRemoval(
            method="zscore",
            transform_dict={
                "transformations": {0: "ClipOutliers"},
                "transformation_params": {0: {"method": "clip", "std_threshold": 3}},
            },
        )

        result = transformer.fit_transform(self.df)

        # Should complete without error
        self.assertLessEqual(len(result), len(self.df))
        self.assertEqual(result.shape[1], self.df.shape[1])

    def test_anomaly_removal_multiple_methods(self):
        """Test AnomalyRemoval with different anomaly detection methods."""
        print("Starting test_anomaly_removal_multiple_methods")
        from autots.tools.transform import AnomalyRemoval
        from autots.tools.anomaly_utils import fast_methods

        for method in fast_methods[:3]:  # Test first 3 fast methods
            with self.subTest(method=method):
                print(
                    f"Starting subtest test_anomaly_removal_multiple_methods method={method}"
                )
                try:
                    transformer = AnomalyRemoval(method=method)
                    result = transformer.fit_transform(self.df)

                    # Should complete without error
                    self.assertLessEqual(len(result), len(self.df))
                    self.assertEqual(result.shape[1], self.df.shape[1])
                except Exception as e:
                    # Some methods might fail on this small dataset, that's okay
                    if "requires" not in str(e).lower():
                        raise


if __name__ == '__main__':
    unittest.main()
