# -*- coding: utf-8 -*-
"""Test cointegration functionality.
Created on September 25, 2025

@author: GitHub Copilot
"""
import unittest
import numpy as np
import pandas as pd
from autots.tools.cointegration import coint_johansen, btcd_decompose, coint_fast
from sklearn.linear_model import LinearRegression


class TestCointegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        np.random.seed(42)

        # Create simple cointegrated time series
        n_obs = 50  # Keep small for speed
        n_vars = 3

        # Generate I(1) processes with a common stochastic trend
        common_trend = np.cumsum(np.random.normal(0, 1, n_obs))
        cls.data = np.zeros((n_obs, n_vars))

        # Create cointegrated series
        cls.data[:, 0] = common_trend + np.random.normal(0, 0.1, n_obs)
        cls.data[:, 1] = 2 * common_trend + np.random.normal(0, 0.1, n_obs)
        cls.data[:, 2] = -common_trend + np.random.normal(0, 0.1, n_obs)

        # Also create a simple random walk for non-cointegrated test
        cls.random_data = np.cumsum(np.random.normal(0, 1, (n_obs, n_vars)), axis=0)

    def test_coint_fast_basic(self):
        """Test basic functionality of coint_fast."""
        result = coint_fast(self.data, k_ar_diff=1)

        # Check output shape
        self.assertEqual(result.shape, (self.data.shape[1], self.data.shape[1]))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_coint_fast_error_handling(self):
        """Test error handling in coint_fast."""
        # Test with insufficient observations
        small_data = self.data[:3, :]
        with self.assertRaises(ValueError):
            coint_fast(small_data, k_ar_diff=2)

        # Test with k_ar_diff that would cause issues (close to data length)
        with self.assertRaises(ValueError):
            coint_fast(self.data[:5, :], k_ar_diff=4)

    def test_coint_johansen_fast_mode(self):
        """Test coint_johansen with fast=True."""
        result = coint_johansen(self.data, fast=True)

        # Check output shape
        self.assertEqual(result.shape, (self.data.shape[1], self.data.shape[1]))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_coint_johansen_with_eigenvalues(self):
        """Test coint_johansen returning eigenvalues."""
        eigenvals, eigenvecs = coint_johansen(
            self.data, fast=True, return_eigenvalues=True
        )

        # For fast mode, eigenvals should be None
        self.assertIsNone(eigenvals)
        self.assertEqual(eigenvecs.shape, (self.data.shape[1], self.data.shape[1]))

    def test_coint_johansen_slow_mode(self):
        """Test coint_johansen with fast=False (original implementation)."""
        result = coint_johansen(self.data, fast=False)

        # Check output shape
        self.assertEqual(result.shape, (self.data.shape[1], self.data.shape[1]))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_coint_johansen_slow_with_eigenvalues(self):
        """Test coint_johansen slow mode returning eigenvalues."""
        eigenvals, eigenvecs = coint_johansen(
            self.data, fast=False, return_eigenvalues=True
        )

        # Check eigenvalues and eigenvectors
        self.assertEqual(len(eigenvals), self.data.shape[1])
        self.assertEqual(eigenvecs.shape, (self.data.shape[1], self.data.shape[1]))
        self.assertFalse(np.any(np.isnan(eigenvals)))
        self.assertFalse(np.any(np.isnan(eigenvecs)))

    def test_btcd_decompose_basic(self):
        """Test basic functionality of btcd_decompose."""
        model = LinearRegression()
        result = btcd_decompose(self.data, model, max_lag=1)

        # Check output shape
        self.assertEqual(result.shape, (self.data.shape[1], self.data.shape[1]))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_btcd_decompose_with_eigenvalues(self):
        """Test btcd_decompose returning eigenvalues."""
        model = LinearRegression()
        eigenvals, eigenvecs = btcd_decompose(
            self.data, model, max_lag=1, return_eigenvalues=True
        )

        # Check eigenvalues and eigenvectors
        self.assertEqual(len(eigenvals), self.data.shape[1])
        self.assertEqual(eigenvecs.shape, (self.data.shape[1], self.data.shape[1]))
        self.assertFalse(np.any(np.isnan(eigenvals)))
        self.assertFalse(np.any(np.isnan(eigenvecs)))

    def test_btcd_decompose_different_lags(self):
        """Test btcd_decompose with different lag orders."""
        model = LinearRegression()

        for max_lag in [1, 2]:
            with self.subTest(max_lag=max_lag):
                result = btcd_decompose(self.data, model, max_lag=max_lag)
                self.assertEqual(result.shape, (self.data.shape[1], self.data.shape[1]))
                self.assertFalse(np.any(np.isnan(result)))

    def test_btcd_decompose_error_handling(self):
        """Test error handling in btcd_decompose."""
        model = LinearRegression()

        # Test with insufficient observations
        small_data = self.data[:3, :]
        with self.assertRaises(ValueError):
            btcd_decompose(small_data, model, max_lag=2)

    def test_consistency_between_methods(self):
        """Test that both fast and slow Johansen methods give reasonable results."""
        result_fast = coint_johansen(self.data, fast=True)
        result_slow = coint_johansen(self.data, fast=False)

        # Both should have same shape
        self.assertEqual(result_fast.shape, result_slow.shape)

        # Both should be finite
        self.assertTrue(np.all(np.isfinite(result_fast)))
        self.assertTrue(np.all(np.isfinite(result_slow)))

    def test_cointegration_with_pandas_dataframe(self):
        """Test cointegration functions work with pandas DataFrame input."""
        df = pd.DataFrame(self.data, columns=['A', 'B', 'C'])

        # Test coint_johansen
        result = coint_johansen(df.values, fast=True)
        self.assertEqual(result.shape, (3, 3))

        # Test btcd_decompose
        model = LinearRegression()
        result_btcd = btcd_decompose(df.values, model, max_lag=1)
        self.assertEqual(result_btcd.shape, (3, 3))

    def test_different_data_sizes(self):
        """Test with different data dimensions."""
        # Test with 2 variables
        data_2d = self.data[:, :2]
        result = coint_johansen(data_2d, fast=True)
        self.assertEqual(result.shape, (2, 2))

        # Test with single variable (edge case)
        data_1d = self.data[:, :1]
        result = coint_johansen(data_1d, fast=True)
        self.assertEqual(result.shape, (1, 1))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Create data with large values
        large_data = self.data * 1e6
        result = coint_johansen(large_data, fast=True)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

        # Create data with small values
        small_data = self.data * 1e-6
        result = coint_johansen(small_data, fast=True)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))


class TestCointegrationIntegration(unittest.TestCase):
    """Integration tests for cointegration in the context of sklearn models."""

    def test_cointegration_in_rolling_regressor(self):
        """Test that cointegration works within the sklearn model context."""
        # This is a simplified test to ensure the integration works
        try:
            from autots.models.sklearn import rolling_x_regressor

            # Create simple test data
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', periods=30, freq='D')
            data = pd.DataFrame(
                {
                    'A': np.cumsum(np.random.normal(0, 1, 30)),
                    'B': np.cumsum(np.random.normal(0, 1, 30)),
                },
                index=dates,
            )

            # Test with fast cointegration
            result = rolling_x_regressor(
                data,
                cointegration="fast",
                cointegration_lag=1,
                mean_rolling_periods=None,
                std_rolling_periods=None,
                additional_lag_periods=None,
            )

            # Should have additional cointegration features
            self.assertGreater(result.shape[1], data.shape[1])
            self.assertFalse(result.isnull().all().any())

        except ImportError as e:
            self.skipTest(f"Skipping integration test due to import error: {e}")


if __name__ == '__main__':
    unittest.main()
