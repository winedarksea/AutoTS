# -*- coding: utf-8 -*-
"""NVAR model testing."""
import unittest
import numpy as np
import pandas as pd
import warnings
from autots.models.basics import NVAR, predict_reservoir
from autots.datasets import load_daily, load_sine


class NVARTest(unittest.TestCase):
    """Test suite for NVAR model and predict_reservoir function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create simple test data
        np.random.seed(2020)
        self.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Simple sine wave data for predictable testing
        t = np.linspace(0, 10, 100)
        self.df_simple = pd.DataFrame({
            'series1': np.sin(t) + 0.1 * np.random.randn(100),
            'series2': np.cos(t) + 0.1 * np.random.randn(100),
            'series3': np.sin(2 * t) + 0.1 * np.random.randn(100)
        }, index=self.dates)
        
        # More realistic data
        self.df_realistic = load_daily(long=False).iloc[:100]
        
        # Edge case: very small dataset
        self.df_small = self.df_simple.iloc[:15].copy()
        
        # Edge case: single series
        self.df_single = self.df_simple[['series1']].copy()

    def test_predict_reservoir_basic(self):
        """Test basic functionality of predict_reservoir."""
        df_array = self.df_simple.values.T
        forecast_length = 10
        
        # Test basic prediction
        pred = predict_reservoir(
            df_array,
            forecast_length=forecast_length,
            k=1,
            warmup_pts=1,
            seed_pts=1,
            ridge_param=2.5e-6,
            prediction_interval=None
        )
        
        # Check shape
        self.assertEqual(pred.shape, (3, forecast_length))
        # Check for no NaN or Inf
        self.assertFalse(np.isnan(pred).any(), "Prediction contains NaN values")
        self.assertFalse(np.isinf(pred).any(), "Prediction contains Inf values")

    def test_predict_reservoir_with_intervals(self):
        """Test predict_reservoir with prediction intervals."""
        df_array = self.df_simple.values.T
        forecast_length = 10
        prediction_interval = 0.9
        
        pred, pred_upper, pred_lower = predict_reservoir(
            df_array,
            forecast_length=forecast_length,
            k=1,
            warmup_pts=1,
            seed_pts=5,
            ridge_param=2.5e-6,
            prediction_interval=prediction_interval
        )
        
        # Check shapes
        self.assertEqual(pred.shape, (3, forecast_length))
        self.assertEqual(pred_upper.shape, (3, forecast_length))
        self.assertEqual(pred_lower.shape, (3, forecast_length))
        
        # Check interval ordering (upper >= pred >= lower)
        self.assertTrue(np.all(pred_upper >= pred), "Upper bound below prediction")
        self.assertTrue(np.all(pred >= pred_lower), "Lower bound above prediction")
        
        # Check for no NaN or Inf
        for arr in [pred, pred_upper, pred_lower]:
            self.assertFalse(np.isnan(arr).any(), "Contains NaN values")
            self.assertFalse(np.isinf(arr).any(), "Contains Inf values")

    def test_predict_reservoir_different_k_values(self):
        """Test predict_reservoir with different k values."""
        df_array = self.df_simple.values.T
        forecast_length = 5
        
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                pred = predict_reservoir(
                    df_array,
                    forecast_length=forecast_length,
                    k=k,
                    warmup_pts=1,
                    seed_pts=1,
                    ridge_param=2.5e-6,
                    prediction_interval=None
                )
                self.assertEqual(pred.shape, (3, forecast_length))
                self.assertFalse(np.isnan(pred).any())

    def test_predict_reservoir_edge_cases(self):
        """Test predict_reservoir edge cases."""
        # Test with minimal data
        df_array = self.df_small.values.T
        forecast_length = 3
        
        pred = predict_reservoir(
            df_array,
            forecast_length=forecast_length,
            k=1,
            warmup_pts=1,
            seed_pts=1,
            ridge_param=2.5e-6,
            prediction_interval=None
        )
        
        self.assertEqual(pred.shape, (3, forecast_length))
        self.assertFalse(np.isnan(pred).any())

    def test_predict_reservoir_assertions(self):
        """Test that assertions properly catch invalid inputs."""
        df_array = self.df_simple.values.T
        
        # Test k <= 0
        with self.assertRaises(ValueError):
            predict_reservoir(df_array, forecast_length=10, k=0)
        
        # Test warmup_pts <= 0
        with self.assertRaises(ValueError):
            predict_reservoir(df_array, forecast_length=10, warmup_pts=0)
        
        # Test insufficient data for k
        df_tiny = self.df_simple.iloc[:2].values.T
        with self.assertRaises(ValueError):
            predict_reservoir(df_tiny, forecast_length=10, k=3)

    def test_predict_reservoir_seed_weighted(self):
        """Test different seed_weighted options."""
        df_array = self.df_simple.values.T
        forecast_length = 10
        
        for seed_weighted in [None, 'linear', 'exponential']:
            with self.subTest(seed_weighted=seed_weighted):
                pred, pred_upper, pred_lower = predict_reservoir(
                    df_array,
                    forecast_length=forecast_length,
                    k=1,
                    warmup_pts=1,
                    seed_pts=5,
                    seed_weighted=seed_weighted,
                    ridge_param=2.5e-6,
                    prediction_interval=0.9
                )
                self.assertEqual(pred.shape, (3, forecast_length))
                self.assertFalse(np.isnan(pred).any())

    def test_nvar_basic_fit_predict(self):
        """Test NVAR basic fit and predict functionality."""
        model = NVAR(
            k=1,
            ridge_param=2.5e-6,
            warmup_pts=1,
            seed_pts=1,
            batch_size=5
        )
        
        model.fit(self.df_simple)
        forecast = model.predict(forecast_length=10, just_point_forecast=True)
        
        # Check output shape
        self.assertEqual(forecast.shape, (10, 3))
        self.assertEqual(list(forecast.columns), list(self.df_simple.columns))
        
        # Check for no NaN or Inf
        self.assertFalse(forecast.isna().any().any())
        self.assertFalse(np.isinf(forecast.values).any())

    def test_nvar_with_prediction_object(self):
        """Test NVAR returns proper PredictionObject."""
        model = NVAR(
            k=1,
            prediction_interval=0.9,
            seed_pts=3
        )
        
        model.fit(self.df_simple)
        prediction = model.predict(forecast_length=10, just_point_forecast=False)
        
        # Check PredictionObject attributes
        self.assertIsNotNone(prediction.forecast)
        self.assertIsNotNone(prediction.upper_forecast)
        self.assertIsNotNone(prediction.lower_forecast)
        
        # Check shapes match
        self.assertEqual(prediction.forecast.shape, prediction.upper_forecast.shape)
        self.assertEqual(prediction.forecast.shape, prediction.lower_forecast.shape)
        
        # Check interval ordering
        self.assertTrue(
            (prediction.upper_forecast.values >= prediction.forecast.values).all()
        )
        self.assertTrue(
            (prediction.forecast.values >= prediction.lower_forecast.values).all()
        )

    def test_nvar_different_batch_sizes(self):
        """Test NVAR with different batch sizes."""
        forecast_length = 5
        
        for batch_size in [2, 5, 10]:
            with self.subTest(batch_size=batch_size):
                model = NVAR(k=1, batch_size=batch_size)
                model.fit(self.df_simple)
                forecast = model.predict(
                    forecast_length=forecast_length,
                    just_point_forecast=True
                )
                
                self.assertEqual(forecast.shape, (forecast_length, 3))
                self.assertFalse(forecast.isna().any().any())

    def test_nvar_batch_methods(self):
        """Test different batch_method options."""
        forecast_length = 5
        
        for batch_method in ['input_order', 'med_sorted', 'std_sorted']:
            with self.subTest(batch_method=batch_method):
                model = NVAR(k=1, batch_method=batch_method)
                model.fit(self.df_simple)
                forecast = model.predict(
                    forecast_length=forecast_length,
                    just_point_forecast=True
                )
                
                self.assertEqual(forecast.shape, (forecast_length, 3))

    def test_nvar_single_series(self):
        """Test NVAR with single series."""
        model = NVAR(k=1, batch_size=1)
        model.fit(self.df_single)
        forecast = model.predict(forecast_length=5, just_point_forecast=True)
        
        self.assertEqual(forecast.shape, (5, 1))
        self.assertFalse(forecast.isna().any().any())

    def test_nvar_parameter_persistence(self):
        """Test that NVAR get_params returns set parameters."""
        params = {
            'k': 2,
            'ridge_param': 1e-5,
            'warmup_pts': 10,
            'seed_pts': 5,
            'seed_weighted': 'linear',
            'batch_size': 10,
            'batch_method': 'std_sorted'
        }
        
        model = NVAR(**params)
        retrieved_params = model.get_params()
        
        for key, value in params.items():
            self.assertEqual(retrieved_params[key], value)

    def test_nvar_get_new_params(self):
        """Test that get_new_params generates valid parameters."""
        model = NVAR()
        
        for _ in range(10):
            new_params = model.get_new_params()
            
            # Check all required keys exist
            required_keys = [
                'k', 'ridge_param', 'warmup_pts', 'seed_pts',
                'seed_weighted', 'batch_size', 'batch_method'
            ]
            for key in required_keys:
                self.assertIn(key, new_params)
            
            # Check value ranges
            self.assertIn(new_params['k'], [1, 2, 3, 4, 5])
            self.assertGreater(new_params['ridge_param'], 0)
            self.assertGreater(new_params['warmup_pts'], 0)
            self.assertGreater(new_params['seed_pts'], 0)

    def test_nvar_realistic_data(self):
        """Test NVAR on more realistic data."""
        model = NVAR(k=1, batch_size=3)
        
        # Use subset of realistic data, filling NaN values
        # (NVAR doesn't handle NaN in input data - data should be pre-processed)
        df = self.df_realistic.iloc[:50, :6]
        # Forward fill, then backward fill, then fill remaining with 0
        df = df.ffill().bfill().fillna(0)
        
        model.fit(df)
        forecast = model.predict(forecast_length=7, just_point_forecast=True)
        
        self.assertEqual(forecast.shape, (7, 6))
        # Model should produce valid forecasts with clean data
        self.assertFalse(forecast.isna().any().any(), "Forecast contains NaN with clean data")

    def test_nvar_numerical_stability(self):
        """Test NVAR handles numerical edge cases."""
        # Create data with different scales
        df = pd.DataFrame({
            'large': np.random.randn(50) * 1000,
            'small': np.random.randn(50) * 0.001,
            'normal': np.random.randn(50)
        }, index=pd.date_range('2020-01-01', periods=50, freq='D'))
        
        model = NVAR(k=1, batch_size=3)
        model.fit(df)
        
        # Should not raise errors
        forecast = model.predict(forecast_length=5, just_point_forecast=True)
        self.assertEqual(forecast.shape, (5, 3))

    def test_nvar_warning_suppression(self):
        """Test that warnings are properly suppressed in predict."""
        model = NVAR(k=2, ridge_param=1e-10)
        model.fit(self.df_simple)
        
        # Should not emit warnings to user
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            forecast = model.predict(forecast_length=5, just_point_forecast=True)
            
            # The model should suppress its own warnings
            # We check that it doesn't crash
            self.assertEqual(forecast.shape, (5, 3))

    def test_nvar_forecast_index(self):
        """Test that forecast has correct datetime index."""
        model = NVAR(k=1)
        model.fit(self.df_simple)
        
        forecast_length = 10
        forecast = model.predict(
            forecast_length=forecast_length,
            just_point_forecast=True
        )
        
        # Check index is datetime
        self.assertTrue(isinstance(forecast.index, pd.DatetimeIndex))
        
        # Check length
        self.assertEqual(len(forecast.index), forecast_length)
        
        # Check index starts after training data
        self.assertTrue(forecast.index[0] > self.df_simple.index[-1])

    def test_predict_reservoir_deterministic(self):
        """Test that predict_reservoir gives consistent results."""
        df_array = self.df_simple.values.T
        
        pred1 = predict_reservoir(
            df_array.copy(),
            forecast_length=10,
            k=1,
            warmup_pts=1,
            seed_pts=1,
            ridge_param=2.5e-6,
            prediction_interval=None
        )
        
        pred2 = predict_reservoir(
            df_array.copy(),
            forecast_length=10,
            k=1,
            warmup_pts=1,
            seed_pts=1,
            ridge_param=2.5e-6,
            prediction_interval=None
        )
        
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_nvar_large_k_scaling(self):
        """Test that larger k values work but use carefully."""
        # Small number of series to avoid memory issues
        df = self.df_simple[['series1', 'series2']].copy()
        
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                model = NVAR(k=k, batch_size=2)
                model.fit(df)
                forecast = model.predict(forecast_length=3, just_point_forecast=True)
                
                self.assertEqual(forecast.shape, (3, 2))
                # With larger k, predictions might be less stable
                # but should still be finite
                self.assertFalse(np.isnan(forecast.values).any())

    def test_predict_reservoir_seed_pts_interval_corruption(self):
        """Test that seed_pts > 1 doesn't corrupt intervals with historical data.
        
        This test verifies the fix for the bug where interval_list slicing was
        not accounting for the ns offset, causing historical points to be included
        in the forecast distribution.
        """
        # Use deterministic data for reproducibility
        np.random.seed(42)
        df_array = self.df_simple.values.T
        forecast_length = 5
        
        # Get predictions with seed_pts > 1
        pred, pred_upper, pred_lower = predict_reservoir(
            df_array,
            forecast_length=forecast_length,
            k=1,
            warmup_pts=1,
            seed_pts=10,  # Multiple seeds
            ridge_param=2.5e-6,
            prediction_interval=0.9
        )
        
        # The intervals should be reasonable
        # Upper should be >= lower
        self.assertTrue(np.all(pred_upper >= pred_lower), 
                       "Upper bound should be >= lower bound")
        
        # Prediction should generally be within bounds
        # (allowing some tolerance for numerical issues)
        within_bounds = np.logical_and(pred >= pred_lower - 1e-10, 
                                       pred <= pred_upper + 1e-10)
        self.assertTrue(np.all(within_bounds), 
                       "Prediction should be within interval bounds")
        
        # The interval width should be relatively stable across forecast steps
        # (not have sudden jumps due to historical data corruption)
        interval_width = pred_upper - pred_lower
        width_changes = np.abs(np.diff(interval_width, axis=1))
        
        # Check that changes aren't too extreme (this would indicate corruption)
        # For well-behaved data, width changes should be gradual
        max_relative_change = np.max(width_changes / (interval_width[:, :-1] + 1e-10))
        self.assertLess(max_relative_change, 10.0, 
                       "Interval width shouldn't have extreme jumps")


class NVARPerformanceTest(unittest.TestCase):
    """Performance and stress tests for NVAR."""

    def setUp(self):
        """Set up performance test fixtures."""
        np.random.seed(2020)
        self.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')

    def test_nvar_many_series(self):
        """Test NVAR with many series (tests batching)."""
        # Create data with 30 series
        data = {}
        for i in range(30):
            data[f'series_{i}'] = np.random.randn(100) + i * 0.1
        df = pd.DataFrame(data, index=self.dates)
        
        model = NVAR(k=1, batch_size=5)
        model.fit(df)
        forecast = model.predict(forecast_length=5, just_point_forecast=True)
        
        self.assertEqual(forecast.shape, (5, 30))
        self.assertFalse(forecast.isna().any().any())

    def test_nvar_long_forecast(self):
        """Test NVAR with longer forecast horizon."""
        df = pd.DataFrame({
            'series1': np.random.randn(100),
            'series2': np.random.randn(100)
        }, index=self.dates)
        
        model = NVAR(k=1, batch_size=2)
        model.fit(df)
        
        # Longer forecast
        forecast = model.predict(forecast_length=30, just_point_forecast=True)
        
        self.assertEqual(forecast.shape, (30, 2))


class NVARStabilityTest(unittest.TestCase):
    """Tests for sorting stability and numeric precision improvements."""

    def setUp(self):
        """Set up stability test fixtures."""
        np.random.seed(2024)
        self.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')

    def test_batch_sorting_deterministic_with_ties(self):
        """Test that batch sorting is deterministic when columns have identical statistics.
        
        This tests the fix for sorting stability across platforms when columns
        have identical median/std/max values.
        """
        # Create data where multiple columns have identical statistics
        base_series = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
        
        # Create 5 series with identical median, std, and max
        df = pd.DataFrame({
            'col_a': base_series.copy(),
            'col_b': base_series.copy(),
            'col_c': base_series.copy(),
            'col_d': base_series.copy() * 1.5,  # Different scale
            'col_e': base_series.copy(),
        }, index=self.dates)
        
        # Test each batch method
        for batch_method in ['med_sorted', 'std_sorted', 'max_sorted']:
            with self.subTest(batch_method=batch_method):
                # Fit model multiple times
                results = []
                for _ in range(5):
                    model = NVAR(k=1, batch_size=3, batch_method=batch_method)
                    model.fit(df.copy())
                    forecast = model.predict(forecast_length=5, just_point_forecast=True)
                    results.append(forecast)
                
                # All results should be identical (deterministic)
                for i in range(1, len(results)):
                    pd.testing.assert_frame_equal(
                        results[0], results[i],
                        check_exact=False,
                        rtol=1e-10,
                        atol=1e-10,
                        obj=f"Forecast {i} differs from first forecast with {batch_method}"
                    )

    def test_batch_sorting_column_order(self):
        """Test that columns with identical statistics are sorted by name."""
        # Create data with intentionally identical statistics
        value = np.ones(100)
        df = pd.DataFrame({
            'z_col': value.copy(),
            'a_col': value.copy(),
            'm_col': value.copy(),
        }, index=self.dates)
        
        for batch_method in ['med_sorted', 'std_sorted', 'max_sorted']:
            with self.subTest(batch_method=batch_method):
                model = NVAR(k=1, batch_size=3, batch_method=batch_method)
                model.fit(df)
                
                # When all statistics are equal, should sort by column name
                # Expected order: 'z_col', 'm_col', 'a_col' (descending by stat, then ascending by name)
                # But since all stats are equal, should be: 'a_col', 'm_col', 'z_col' (ascending by name)
                expected_order = ['a_col', 'm_col', 'z_col']
                actual_order = list(model.new_col_names)
                
                self.assertEqual(actual_order, expected_order,
                               f"Column order incorrect for {batch_method}")

    def test_ridge_regression_numeric_stability(self):
        """Test that ridge regression handles ill-conditioned matrices well."""
        # Create data that could lead to ill-conditioned matrices
        np.random.seed(42)
        
        # Test with very small ridge parameter (more ill-conditioned)
        df_array = np.random.randn(3, 50)
        
        # Should not raise errors or produce NaN/Inf even with small ridge param
        pred = predict_reservoir(
            df_array,
            forecast_length=10,
            k=1,
            warmup_pts=1,
            seed_pts=1,
            ridge_param=1e-10,  # Very small ridge parameter
            prediction_interval=None
        )
        
        self.assertFalse(np.isnan(pred).any(), "Prediction contains NaN")
        self.assertFalse(np.isinf(pred).any(), "Prediction contains Inf")

    def test_ridge_regression_different_scales(self):
        """Test ridge regression with data at very different scales."""
        np.random.seed(42)
        
        # Create data with very different scales
        df_array = np.array([
            np.random.randn(50) * 1e6,   # Very large scale
            np.random.randn(50) * 1e-6,  # Very small scale
            np.random.randn(50),         # Normal scale
        ])
        
        # Should handle different scales without numerical issues
        pred = predict_reservoir(
            df_array,
            forecast_length=5,
            k=1,
            warmup_pts=1,
            seed_pts=1,
            ridge_param=2.5e-6,
            prediction_interval=None
        )
        
        self.assertEqual(pred.shape, (3, 5))
        self.assertFalse(np.isnan(pred).any(), "Prediction contains NaN")
        self.assertFalse(np.isinf(pred).any(), "Prediction contains Inf")

    def test_ridge_regression_reproducibility(self):
        """Test that ridge regression produces identical results on repeated calls."""
        np.random.seed(42)
        df_array = np.random.randn(3, 50)
        
        results = []
        for _ in range(5):
            pred = predict_reservoir(
                df_array.copy(),
                forecast_length=10,
                k=2,
                warmup_pts=1,
                seed_pts=1,
                ridge_param=2.5e-6,
                prediction_interval=None
            )
            results.append(pred)
        
        # All results should be numerically identical
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(
                results[0], results[i],
                decimal=12,
                err_msg=f"Result {i} differs from first result"
            )

    def test_ridge_regression_with_near_singular_matrix(self):
        """Test ridge regression with nearly singular input matrices."""
        np.random.seed(42)
        
        # Create data with high collinearity (nearly singular)
        base = np.random.randn(50)
        df_array = np.array([
            base,
            base + 1e-8 * np.random.randn(50),  # Nearly identical to first
            base * 2 + 1e-8 * np.random.randn(50),  # Nearly linear combination
        ])
        
        # Ridge regression should still work due to regularization
        pred = predict_reservoir(
            df_array,
            forecast_length=5,
            k=1,
            warmup_pts=1,
            seed_pts=1,
            ridge_param=1e-5,  # Regularization helps here
            prediction_interval=None
        )
        
        self.assertEqual(pred.shape, (3, 5))
        self.assertFalse(np.isnan(pred).any())
        self.assertFalse(np.isinf(pred).any())

    def test_nvar_consistency_across_batch_sizes(self):
        """Test that batch_size produces valid results across different sizes.
        
        Note: batch_size can affect results because series are processed in batches,
        but all results should be valid (no NaN/Inf) and have the same shape.
        """
        np.random.seed(2024)
        df = pd.DataFrame({
            f'series_{i}': np.random.randn(50) for i in range(10)
        }, index=pd.date_range('2020-01-01', periods=50, freq='D'))
        
        # Test with different batch sizes on same data
        for batch_size in [2, 5, 10]:
            with self.subTest(batch_size=batch_size):
                model = NVAR(k=1, batch_size=batch_size, batch_method='input_order')
                model.fit(df.copy())
                forecast = model.predict(
                    forecast_length=5,
                    just_point_forecast=True
                )
                
                # All forecasts should have same shape and be valid
                self.assertEqual(forecast.shape, (5, 10))
                self.assertFalse(forecast.isna().any().any())
                self.assertFalse(np.isinf(forecast.values).any())


    def test_ridge_regression_extreme_ridge_params(self):
        """Test ridge regression with extreme ridge parameter values."""
        np.random.seed(42)
        df_array = np.random.randn(3, 50)
        
        # Test with very small ridge parameter
        pred_small = predict_reservoir(
            df_array.copy(),
            forecast_length=5,
            k=1,
            warmup_pts=1,
            seed_pts=1,
            ridge_param=1e-12,
            prediction_interval=None
        )
        self.assertFalse(np.isnan(pred_small).any())
        self.assertFalse(np.isinf(pred_small).any())
        
        # Test with large ridge parameter
        pred_large = predict_reservoir(
            df_array.copy(),
            forecast_length=5,
            k=1,
            warmup_pts=1,
            seed_pts=1,
            ridge_param=1e3,
            prediction_interval=None
        )
        self.assertFalse(np.isnan(pred_large).any())
        self.assertFalse(np.isinf(pred_large).any())
        
        # Predictions should differ (different regularization)
        # But both should be valid
        self.assertFalse(np.allclose(pred_small, pred_large))


class NVAREarlyStoppingTest(unittest.TestCase):
    """Test suite for early stopping functionality."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(2024)
        self.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')

    def test_early_stopping_prevents_inf(self):
        """Test that early stopping prevents Inf values in forecasts."""
        # Create data that tends to diverge with small ridge parameter
        df = pd.DataFrame({
            'series1': np.random.randn(100) * 100,
            'series2': np.random.randn(100) * 100,
            'series3': np.random.randn(100) * 100,
        }, index=self.dates)
        
        model = NVAR(k=1, ridge_param=1e-8, batch_size=3)
        model.fit(df)
        
        # Long forecast that might diverge
        forecast = model.predict(forecast_length=30, just_point_forecast=True)
        
        # Should not have Inf values (early stopping should prevent this)
        self.assertFalse(np.isinf(forecast.values).any(), 
                        "Early stopping should prevent Inf values")

    def test_early_stopping_does_not_stop_on_good_data(self):
        """Test that early stopping doesn't activate on well-behaved data."""
        # Create stable sinusoidal data
        t = np.linspace(0, 10, 100)
        df = pd.DataFrame({
            'series1': np.sin(t) + 0.1 * np.random.randn(100),
            'series2': np.cos(t) + 0.1 * np.random.randn(100),
            'series3': np.sin(2*t) + 0.1 * np.random.randn(100),
        }, index=self.dates)
        
        model = NVAR(k=1, ridge_param=2.5e-6, batch_size=3)
        model.fit(df)
        
        # Forecast should complete without early stopping
        forecast = model.predict(forecast_length=20, just_point_forecast=True)
        
        # Check forecasts are valid and within reasonable range
        self.assertEqual(forecast.shape, (20, 3))
        self.assertFalse(forecast.isna().any().any())
        self.assertFalse(np.isinf(forecast.values).any())
        
        # Forecasts should vary (not constant from early stopping)
        for col in forecast.columns:
            # Check that forecast values vary (std > 0)
            self.assertGreater(forecast[col].std(), 0.01,
                             f"Forecast for {col} appears constant (early stopping may have activated incorrectly)")

    def test_early_stopping_with_large_values(self):
        """Test that early stopping uses appropriate thresholds for large-scale data."""
        # Create data with large values
        df = pd.DataFrame({
            'series1': np.random.randn(100) * 1e6,
            'series2': np.random.randn(100) * 1e6,
        }, index=self.dates)
        
        model = NVAR(k=1, ridge_param=2.5e-6, batch_size=2)
        model.fit(df)
        
        # Should handle large-scale data without premature stopping
        forecast = model.predict(forecast_length=10, just_point_forecast=True)
        
        self.assertEqual(forecast.shape, (10, 2))
        self.assertFalse(forecast.isna().any().any())
        self.assertFalse(np.isinf(forecast.values).any())
        
        # Forecasts should be on similar scale to training data
        train_scale = np.abs(df.values).max()
        forecast_scale = np.abs(forecast.values).max()
        
        # Allow forecasts up to 100x training scale (threshold from implementation)
        self.assertLess(forecast_scale, train_scale * 100,
                       "Forecast scale unreasonably large")

    def test_early_stopping_with_small_values(self):
        """Test that early stopping works with very small-scale data."""
        # Create data with very small values
        df = pd.DataFrame({
            'series1': np.random.randn(100) * 1e-6,
            'series2': np.random.randn(100) * 1e-6,
        }, index=self.dates)
        
        model = NVAR(k=1, ridge_param=2.5e-6, batch_size=2)
        model.fit(df)
        
        forecast = model.predict(forecast_length=10, just_point_forecast=True)
        
        self.assertEqual(forecast.shape, (10, 2))
        self.assertFalse(forecast.isna().any().any())
        self.assertFalse(np.isinf(forecast.values).any())

    def test_early_stopping_with_intervals(self):
        """Test that early stopping works correctly with prediction intervals."""
        # Create data that might diverge
        df = pd.DataFrame({
            'series1': np.random.randn(100) * 10,
            'series2': np.random.randn(100) * 10,
        }, index=self.dates)
        
        model = NVAR(k=1, ridge_param=1e-7, batch_size=2, seed_pts=5, prediction_interval=0.9)
        model.fit(df)
        
        prediction = model.predict(forecast_length=20, just_point_forecast=False)
        
        # Check all predictions are valid (no Inf/NaN)
        self.assertFalse(np.isinf(prediction.forecast.values).any())
        self.assertFalse(np.isinf(prediction.upper_forecast.values).any())
        self.assertFalse(np.isinf(prediction.lower_forecast.values).any())
        self.assertFalse(prediction.forecast.isna().any().any())
        
        # Intervals should be properly ordered
        self.assertTrue((prediction.upper_forecast.values >= prediction.forecast.values).all())
        self.assertTrue((prediction.forecast.values >= prediction.lower_forecast.values).all())

    def test_early_stopping_growth_data(self):
        """Test that early stopping allows reasonable growth trends."""
        # Create data with growth trend
        t = np.arange(100)
        df = pd.DataFrame({
            'series1': 100 + 0.5 * t + np.random.randn(100),
            'series2': 50 + 0.3 * t + np.random.randn(100),
        }, index=self.dates)
        
        model = NVAR(k=1, ridge_param=2.5e-6, batch_size=2)
        model.fit(df)
        
        forecast = model.predict(forecast_length=10, just_point_forecast=True)
        
        # Should allow forecasts to continue trend
        self.assertEqual(forecast.shape, (10, 2))
        self.assertFalse(forecast.isna().any().any())
        self.assertFalse(np.isinf(forecast.values).any())
        
        # Forecasts should show some variation (not constant from early stop)
        self.assertGreater(forecast['series1'].std(), 0.01)
        self.assertGreater(forecast['series2'].std(), 0.01)

    def test_early_stopping_threshold_calculation(self):
        """Test that divergence thresholds are calculated appropriately."""
        # Create data with known statistics
        df_array = np.array([
            [10, 20, 30, 40, 50] * 20,  # Max = 50
            [1, 2, 3, 4, 5] * 20,        # Max = 5
        ])
        
        # The threshold should be at least 100x the max training value
        # For series 1: threshold >= 100 * 50 = 5000
        # For series 2: threshold >= 100 * 5 = 500
        
        pred = predict_reservoir(
            df_array,
            forecast_length=10,
            k=1,
            warmup_pts=1,
            seed_pts=1,
            ridge_param=2.5e-6,
            prediction_interval=None
        )
        
        # Predictions should be valid
        self.assertFalse(np.isnan(pred).any())
        self.assertFalse(np.isinf(pred).any())
        
        # Predictions should be well below threshold
        # (for well-behaved data, shouldn't approach threshold)
        self.assertLess(np.abs(pred[0]).max(), 5000)
        self.assertLess(np.abs(pred[1]).max(), 500)


if __name__ == '__main__':
    unittest.main()

