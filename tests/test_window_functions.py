import unittest
import numpy as np
import pandas as pd
from autots.tools.window_functions import window_maker, last_window

# -*- coding: utf-8 -*-
"""Test window functions used in WindowRegression
"""


class TestWindowFunctions(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create sample time series data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Single series for univariate tests
        self.df_univariate = pd.DataFrame({
            'series1': np.random.randn(100).cumsum() + 100
        }, index=dates)
        
        # Multiple series for multivariate tests  
        self.df_multivariate = pd.DataFrame({
            'series1': np.random.randn(100).cumsum() + 100,
            'series2': np.random.randn(100).cumsum() + 200,
            'series3': np.random.randn(100).cumsum() + 50
        }, index=dates)
        
        # Future regressor data
        self.future_regressor = pd.DataFrame({
            'regressor1': np.random.randn(100),
            'regressor2': np.random.randn(100)
        }, index=dates)

    def test_window_maker_basic_univariate(self):
        """Test basic window_maker functionality with univariate data."""
        X, Y = window_maker(
            self.df_univariate,
            window_size=5,
            input_dim='univariate',
            forecast_length=1,
            output_dim='forecast_length'
        )
        
        # Check shapes
        expected_windows = self.df_univariate.shape[0] - 5 - 1 + 1  # 95
        self.assertEqual(X.shape[0], expected_windows)
        self.assertEqual(Y.shape[0], expected_windows)
        self.assertEqual(X.shape[1], 5)  # window_size
        # Y might be 1D for single forecast length
        if Y.ndim == 1:
            self.assertEqual(Y.shape[0], expected_windows)
        else:
            self.assertEqual(Y.shape[1], 1)  # forecast_length

    def test_window_maker_multivariate_and_1step(self):
        """Test window_maker with multivariate and 1step output."""
        # Test multivariate
        X, Y = window_maker(
            self.df_multivariate,
            window_size=5,
            input_dim='multivariate',
            forecast_length=2,
            output_dim='forecast_length'
        )
        
        # For multivariate, X should contain flattened windows
        expected_windows = self.df_multivariate.shape[0] - 5 - 2 + 1  # 94
        self.assertEqual(X.shape[0], expected_windows)
        self.assertEqual(X.shape[1], 5 * 3)  # window_size * n_series
        
        # Test 1step output
        X, Y = window_maker(
            self.df_univariate,
            window_size=5,
            input_dim='univariate',
            forecast_length=3,  # Should be ignored
            output_dim='1step'
        )
        
        expected_windows = self.df_univariate.shape[0] - 5 - 1 + 1  # 95
        self.assertEqual(X.shape[0], expected_windows)
        self.assertEqual(Y.shape[0], expected_windows)

    def test_window_maker_options(self):
        """Test window normalization and max_windows functionality."""
        # Test normalization
        X_norm, Y_norm = window_maker(
            self.df_univariate,
            window_size=5,
            normalize_window=True,
            forecast_length=1,
            max_windows=None
        )
        
        X_no_norm, Y_no_norm = window_maker(
            self.df_univariate,
            window_size=5,
            normalize_window=False,
            forecast_length=1,
            max_windows=None
        )
        
        # Normalized windows should have different values
        self.assertFalse(np.allclose(X_norm, X_no_norm))
        # Y should be the same regardless of window normalization
        self.assertTrue(np.allclose(Y_norm, Y_no_norm))
        
        # Test max_windows limits output size
        max_windows = 50
        X, Y = window_maker(
            self.df_multivariate,
            window_size=5,
            max_windows=max_windows,
            forecast_length=1
        )
        self.assertLessEqual(X.shape[0], max_windows)



    def test_last_window_basic(self):
        """Test basic last_window functionality."""
        # Test univariate
        result = last_window(
            self.df_univariate,
            window_size=5,
            input_dim='univariate'
        )
        
        self.assertEqual(result.shape[0], 1)  # 1 series
        self.assertEqual(result.shape[1], 5)  # window_size
        
        # Values should match the last 5 values of the series
        expected_values = self.df_univariate.iloc[-5:].values.flatten()
        np.testing.assert_array_almost_equal(result.iloc[0].values, expected_values)
        
        # Test multivariate
        result = last_window(
            self.df_multivariate,
            window_size=5,
            input_dim='multivariate'
        )
        
        expected_features = 5 * 3  # window_size * n_series
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], expected_features)

    def test_last_window_options(self):
        """Test last_window with normalization and edge cases."""
        # Test normalization
        result_norm = last_window(
            self.df_univariate,
            window_size=5,
            input_dim='univariate',
            normalize_window=True
        )
        
        result_no_norm = last_window(
            self.df_univariate,
            window_size=5,
            input_dim='univariate',
            normalize_window=False
        )
        
        # Normalized result should be different
        self.assertFalse(np.allclose(result_norm, result_no_norm))
        
        # Normalized values should sum to 1 (approximately) for each row
        row_sums = result_norm.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=10)
        
        # Test small dataset
        small_df = self.df_univariate.head(10)
        X, Y = window_maker(
            small_df,
            window_size=5,
            forecast_length=2
        )
        expected_windows = 10 - 5 - 2 + 1  # 4
        self.assertEqual(X.shape[0], expected_windows)

    def test_data_integrity_and_edge_cases(self):
        """Test data integrity and edge cases."""
        # Test that window_maker preserves data integrity
        # Use max_windows=None to avoid random sampling and preserve order
        X, Y = window_maker(
            self.df_univariate,
            window_size=3,
            forecast_length=1,
            max_windows=None
        )
        
        # Handle both numpy arrays and pandas DataFrames
        if isinstance(X, np.ndarray):
            first_window = X[0]
        else:
            first_window = X.iloc[0].values
        
        expected_first_window = self.df_univariate.iloc[0:3].values.flatten()
        np.testing.assert_array_almost_equal(first_window, expected_first_window, decimal=5)
        
        # Test last_window edge case
        result_large = last_window(
            self.df_univariate.head(3),
            window_size=5,
            input_dim='univariate'
        )
        # Should return at least some data
        self.assertGreater(result_large.shape[1], 0)

    def test_random_seed_consistency(self):
        """Test that random_seed produces consistent results when max_windows is used."""
        # When max_windows limits the data, random sampling should be consistent
        X1, Y1 = window_maker(
            self.df_multivariate,
            window_size=5,
            max_windows=50,
            random_seed=42
        )
        
        X2, Y2 = window_maker(
            self.df_multivariate,
            window_size=5,
            max_windows=50,
            random_seed=42
        )
        
        # Results should be identical with same random seed
        if isinstance(X1, np.ndarray):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(Y1, Y2)
        else:
            pd.testing.assert_frame_equal(X1, X2)
            pd.testing.assert_frame_equal(Y1, Y2)

    def test_output_types(self):
        """Test that functions return correct data types."""
        X, Y = window_maker(
            self.df_univariate,
            window_size=5,
            forecast_length=1
        )
        
        # Should return numpy arrays or pandas DataFrames
        self.assertTrue(isinstance(X, (np.ndarray, pd.DataFrame)))
        self.assertTrue(isinstance(Y, (np.ndarray, pd.DataFrame)))
        
        # last_window should return pandas DataFrame
        result = last_window(
            self.df_univariate,
            window_size=5,
            input_dim='univariate'
        )
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()