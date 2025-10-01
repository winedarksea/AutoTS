# -*- coding: utf-8 -*-
"""Overall testing."""
import unittest
import numpy as np
import pandas as pd
from autots.datasets import (
    load_daily, load_monthly, load_artificial, load_sine
)
from autots.tools.transform import ThetaTransformer, FIRFilter, HistoricValues, GeneralTransformer, ReconciliationTransformer, UpscaleDownscaleTransformer, MeanPercentSplitter

class TestTransforms(unittest.TestCase):
    
    def test_theta(self):
        # Sample DataFrame with a DatetimeIndex and multiple time series columns
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'series1': np.random.randn(100).cumsum(),
            'series2': np.random.randn(100).cumsum(),
        }, index=dates)

        theta_values = [0, 1, 2]
        theta_transformer = ThetaTransformer(theta_values=theta_values)
        
        params = theta_transformer.get_new_params()
        self.assertTrue(params)

        theta_transformer.fit(data)
        transformed_data = theta_transformer.transform(data)
        reconstructed_data = theta_transformer.inverse_transform(transformed_data)
        self.assertTrue(np.allclose(data.values, reconstructed_data.values, atol=1e-8))

    def test_firfilter(self):
        df = load_daily(long=False)
        transformer = FIRFilter()
        transformed = transformer.fit_transform(df)
        inverse = transformer.inverse_transform(transformed)  # noqa
        
        if False:
            col = df.columns[0]
            pd.concat([df[col], transformed[col].rename("transformed")], axis=1).plot()
        
        self.assertCountEqual(transformed.index.tolist(), df.index.tolist())
        self.assertCountEqual(transformed.columns.tolist(), df.columns.tolist())

    def test_mean_percent_splitter(self):
        """Test MeanPercentSplitter transformer for intermittent demand forecasting."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'series1': np.random.randn(100).cumsum() + 100,
            'series2': np.random.randn(100).cumsum() + 50,
            'series3': np.random.randn(100).cumsum() + 75,
        }, index=dates)
        
        # Test get_new_params
        transformer = MeanPercentSplitter()
        params = transformer.get_new_params()
        self.assertTrue(params)
        self.assertIsInstance(params, dict)
        self.assertIn('window', params)
        
        # Test basic fit/transform/inverse with fixed window
        transformer = MeanPercentSplitter(window=10)
        transformer.fit(df)
        transformed = transformer.transform(df)
        reconstructed = transformer.inverse_transform(transformed)
        
        # Check shapes
        self.assertEqual(transformed.shape[0], df.shape[0])
        self.assertEqual(transformed.shape[1], df.shape[1] * 2)  # mean + percentage for each column
        self.assertEqual(reconstructed.shape, df.shape)
        
        # Check reconstruction accuracy
        self.assertTrue(np.allclose(df.values, reconstructed.values, atol=1e-10))
        
        # Test that original df is not modified
        df_copy = df.copy()
        transformer.fit(df_copy)
        transformed_copy = transformer.transform(df_copy)
        pd.testing.assert_frame_equal(df, df_copy)
        
        # Test forecast_length mode
        forecast_len = 10
        transformer_fl = MeanPercentSplitter(window='forecast_length', forecast_length=forecast_len)
        transformer_fl.fit(df)
        transformed_fl = transformer_fl.transform(df)
        
        # Create forecast data
        forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_len, freq='D')
        forecast_transformed = pd.DataFrame({
            'series1_Xmean': np.ones(forecast_len) * 100,
            'series1_Xpercentage': np.ones(forecast_len) * 1.05,
            'series2_Xmean': np.ones(forecast_len) * 50,
            'series2_Xpercentage': np.ones(forecast_len) * 0.95,
            'series3_Xmean': np.ones(forecast_len) * 75,
            'series3_Xpercentage': np.ones(forecast_len) * 1.0,
        }, index=forecast_dates)
        
        forecast_reconstructed = transformer_fl.inverse_transform(forecast_transformed)
        self.assertEqual(forecast_reconstructed.shape[0], forecast_len)
        self.assertEqual(forecast_reconstructed.shape[1], df.shape[1])
        self.assertTrue(forecast_reconstructed.index.min() > df.index.max())
        
        # Test column naming convention
        expected_mean_cols = [f"{col}_Xmean" for col in df.columns]
        expected_pct_cols = [f"{col}_Xpercentage" for col in df.columns]
        expected_cols = expected_mean_cols + expected_pct_cols
        self.assertCountEqual(transformed.columns.tolist(), expected_cols)

    def test_cointegration_transformer(self):
        from autots.tools.transform import CointegrationTransformer
        
        # Test basic functionality with simple cointegrated data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Create simple cointegrated series
        x1 = np.cumsum(np.random.normal(0, 1, 100))
        x2 = x1 + np.random.normal(0, 0.1, 100)  # Closely related to x1
        x3 = -x1 + np.random.normal(0, 0.1, 100)  # Negatively related to x1
        
        simple_df = pd.DataFrame({
            'series1': x1,
            'series2': x2, 
            'series3': x3
        }, index=dates)
        
        # Test get_new_params method like other transformers
        transformer = CointegrationTransformer()
        params = transformer.get_new_params()
        self.assertTrue(params)
        self.assertIsInstance(params, dict)
        
        # Test fit and transform with simple data
        transformer.fit(simple_df)
        transformed = transformer.transform(simple_df)
        inverse_transformed = transformer.inverse_transform(transformed)
        
        # Basic shape and type checks
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertIsInstance(inverse_transformed, pd.DataFrame)
        self.assertEqual(transformed.shape[0], simple_df.shape[0])
        self.assertEqual(inverse_transformed.shape[0], simple_df.shape[0])
        self.assertCountEqual(transformed.index.tolist(), simple_df.index.tolist())
        self.assertCountEqual(inverse_transformed.index.tolist(), simple_df.index.tolist())
        
        # Use load_daily for additional testing (handles larger dataset)
        df = load_daily(long=False)[:100]
        
        transformer_large = CointegrationTransformer()
        transformer_large.fit(df)
        transformed_large = transformer_large.transform(df)
        inverse_transformed_large = transformer_large.inverse_transform(transformed_large)
        
        # Check that columns are within expected range (depends on fit success/failure)
        self.assertLessEqual(transformed_large.shape[1], transformer_large.max_components)
        
        # Test with insufficient data - should fail gracefully
        small_df = pd.DataFrame(np.random.randn(5, 3), 
                              index=pd.date_range('2020-01-01', periods=5, freq='D'),
                              columns=['a', 'b', 'c'])
        transformer_small = CointegrationTransformer(min_periods=10)
        transformer_small.fit(small_df)
        self.assertTrue(transformer_small.failed_fit)
        
        # Transform should still work even with failed fit (graceful fallback)
        transformed_small = transformer_small.transform(small_df)
        self.assertIsInstance(transformed_small, pd.DataFrame)
        
        # Test with single series - should fail gracefully
        single_df = pd.DataFrame(np.random.randn(100, 1), 
                                index=pd.date_range('2020-01-01', periods=100, freq='D'),
                                columns=['single'])
        transformer_single = CointegrationTransformer()
        transformer_single.fit(single_df)
        self.assertTrue(transformer_single.failed_fit)
        
        # Test with NaN values - should handle gracefully
        nan_df = simple_df.copy()
        nan_df.iloc[10:20, 0] = np.nan
        
        transformer_nan = CointegrationTransformer()
        transformer_nan.fit(nan_df)
        transformed_nan = transformer_nan.transform(nan_df)
        self.assertIsInstance(transformed_nan, pd.DataFrame)
        
        # Test different methods
        for method in ['cca', 'rrr']:
            transformer_method = CointegrationTransformer(method=method)
            transformer_method.fit(simple_df)
            transformed_method = transformer_method.transform(simple_df)
            reconstructed_method = transformer_method.inverse_transform(transformed_method)
            
            self.assertIsInstance(transformed_method, pd.DataFrame)
            self.assertIsInstance(reconstructed_method, pd.DataFrame)
            self.assertEqual(transformed_method.shape[0], simple_df.shape[0])
            self.assertEqual(reconstructed_method.shape[0], simple_df.shape[0])

    def test_historic_values_basic(self):
        """Test HistoricValues transformer basic functionality."""
        df = load_daily(long=False)[:50]  # Small dataset for testing
        
        transformer = HistoricValues()
        params = transformer.get_new_params()
        self.assertTrue(params)
        self.assertIsInstance(params, dict)
        
        # Test basic fit/transform/inverse
        transformer.fit(df)
        transformed = transformer.transform(df)
        inverse_transformed = transformer.inverse_transform(transformed)
        
        # Transform should be identity (no change)
        pd.testing.assert_frame_equal(df, transformed)
        
        # Basic checks
        self.assertIsInstance(inverse_transformed, pd.DataFrame)
        self.assertEqual(inverse_transformed.shape, df.shape)
        self.assertCountEqual(inverse_transformed.index.tolist(), df.index.tolist())
        self.assertCountEqual(inverse_transformed.columns.tolist(), df.columns.tolist())

    def test_historic_values_with_expanding_transformer(self):
        """Test HistoricValues with expanding transformers that change column names/counts."""
        df = load_daily(long=False)[:50]
        
        # Test with CenterSplit (expanding transformer that doubles columns)
        expanding_transform_dict = {
            "transformations": {"0": "CenterSplit", "1": "HistoricValues"},
            "transformation_params": {
                "0": {"center": "zero", "fillna": "linear"},
                "1": {"window": None}
            }
        }
        
        try:
            transformer = GeneralTransformer(**expanding_transform_dict)
            transformer.fit(df)
            transformed = transformer.transform(df)
            inverse_transformed = transformer.inverse_transform(transformed)
            
            # Should handle the expanding transformer gracefully
            self.assertIsInstance(inverse_transformed, pd.DataFrame)
            self.assertEqual(inverse_transformed.shape[0], df.shape[0])
            # Note: columns might be different due to expanding transformer
            
        except Exception as e:
            # If it fails, it should fail gracefully with a clear error
            self.assertIn("HistoricValues", str(e), "Error should mention HistoricValues")

    def test_historic_values_dimension_mismatch(self):
        """Test HistoricValues behavior when forecast data has different dimensions than training data."""
        df = load_daily(long=False)[:50]
        
        transformer = HistoricValues()
        transformer.fit(df)
        
        # Create forecast data with different number of columns
        forecast_data_fewer_cols = df.iloc[:10, :2]  # Fewer columns
        forecast_data_more_cols = pd.concat([df.iloc[:10], df.iloc[:10]], axis=1)  # More columns
        
        # Test with fewer columns - should handle gracefully
        try:
            result_fewer = transformer.inverse_transform(forecast_data_fewer_cols)
            self.assertIsInstance(result_fewer, pd.DataFrame)
        except Exception as e:
            # Should fail gracefully with informative error
            self.assertIsInstance(e, (IndexError, ValueError))
        
        # Test with more columns - should handle gracefully 
        try:
            result_more = transformer.inverse_transform(forecast_data_more_cols)
            self.assertIsInstance(result_more, pd.DataFrame)
        except Exception as e:
            # Should fail gracefully with informative error
            self.assertIsInstance(e, (IndexError, ValueError))

    def test_historic_values_with_different_column_names(self):
        """Test HistoricValues when forecast data has different column names than training data."""
        df = load_daily(long=False)[:50]
        
        transformer = HistoricValues()
        transformer.fit(df)
        
        # Create forecast data with different column names but same shape
        forecast_data = df.iloc[:10].copy()
        forecast_data.columns = [f"new_{col}" for col in forecast_data.columns]
        
        try:
            result = transformer.inverse_transform(forecast_data)
            self.assertIsInstance(result, pd.DataFrame)
            # Result should maintain the forecast data's column names
            self.assertCountEqual(result.columns.tolist(), forecast_data.columns.tolist())
        except Exception as e:
            # Should fail gracefully
            self.assertIsInstance(e, (KeyError, ValueError, IndexError))

    def test_historic_values_window_functionality(self):
        """Test HistoricValues with different window settings."""
        df = load_daily(long=False)[:100]
        
        # Test with window=None (full history)
        transformer_full = HistoricValues(window=None)
        transformer_full.fit(df)
        self.assertEqual(transformer_full.df.shape[0], df.shape[0])
        
        # Test with window=20 (limited history)
        transformer_window = HistoricValues(window=20)
        transformer_window.fit(df)
        self.assertEqual(transformer_window.df.shape[0], 20)
        self.assertEqual(transformer_window.df.shape[1], df.shape[1])
        
        # Test with window larger than data
        transformer_large_window = HistoricValues(window=200)
        transformer_large_window.fit(df)
        self.assertEqual(transformer_large_window.df.shape[0], df.shape[0])  # Should use all available data

    def test_historic_values_with_missing_data(self):
        """Test HistoricValues behavior with NaN values."""
        df = load_daily(long=False)[:50]
        
        # Add some NaN values
        df_with_nan = df.copy()
        df_with_nan.iloc[10:15, 0] = np.nan
        df_with_nan.iloc[20:25, 1] = np.nan
        
        transformer = HistoricValues()
        try:
            transformer.fit(df_with_nan)
            transformed = transformer.transform(df_with_nan)
            inverse_transformed = transformer.inverse_transform(transformed)
            
            self.assertIsInstance(inverse_transformed, pd.DataFrame)
            self.assertEqual(inverse_transformed.shape, df_with_nan.shape)
        except Exception as e:
            # Should handle NaN gracefully or fail with clear error
            self.assertIsInstance(e, (ValueError, TypeError))

    def test_reconciliation_transformer_basic(self):
        """Test ReconciliationTransformer basic functionality."""
        # Create simple test data with multiple series
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            f'series_{i}': np.random.randn(50).cumsum() for i in range(6)
        }, index=dates)
        
        # Test basic MinT method
        transformer = ReconciliationTransformer(group_size=3)
        params = transformer.get_new_params()
        self.assertTrue(params)
        self.assertIsInstance(params, dict)
        
        # Test fit/transform/inverse cycle
        transformer.fit(df)
        transformed = transformer.transform(df)
        inverse_transformed = transformer.inverse_transform(transformed)
        
        # Basic shape checks
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertIsInstance(inverse_transformed, pd.DataFrame)
        self.assertEqual(transformed.shape[0], df.shape[0])
        self.assertEqual(inverse_transformed.shape[0], df.shape[0])
        self.assertEqual(inverse_transformed.shape[1], df.shape[1])  # Should return only bottom-level
        
        # Check that hierarchy columns are added in transform
        self.assertGreater(transformed.shape[1], df.shape[1])
        
        # Test new reconciliation methods
        for method in ["volatility_mint", "iterative_mint"]:
            test_transformer = ReconciliationTransformer(
                group_size=3,
                reconciliation_params={"method": method, "max_iterations": 3}  # Low iterations for speed
            )
            test_transformer.fit(df)
            test_transformed = test_transformer.transform(df)
            test_inverse = test_transformer.inverse_transform(test_transformed)
            
            self.assertIsInstance(test_inverse, pd.DataFrame)
            self.assertEqual(test_inverse.shape, df.shape)
        
        # Test with custom hierarchy map
        hierarchy_map = {"TOP": list(df.columns), "MID1": list(df.columns[:3]), "MID2": list(df.columns[3:])}
        custom_transformer = ReconciliationTransformer(hierarchy_map=hierarchy_map)
        custom_transformer.fit(df)
        custom_transformed = custom_transformer.transform(df)
        custom_inverse = custom_transformer.inverse_transform(custom_transformed)
        
        self.assertIsInstance(custom_inverse, pd.DataFrame)
        self.assertEqual(custom_inverse.shape, df.shape)

    def test_reconciliation_transformer_performance(self):
        """Test ReconciliationTransformer performance with moderately sized data."""
        import time
        
        # Create moderately sized dataset to test performance
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')  # 500 time periods
        df = pd.DataFrame({
            f'series_{i}': np.random.randn(500).cumsum() for i in range(20)  # 20 series
        }, index=dates)
        
        # Test that basic reconciliation completes quickly
        start_time = time.time()
        transformer = ReconciliationTransformer(group_size=5)
        transformer.fit(df)
        transformed = transformer.transform(df)
        inverse_transformed = transformer.inverse_transform(transformed)
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds for this size)
        self.assertLess(elapsed, 5.0, f"ReconciliationTransformer took {elapsed:.2f}s, too slow")
        
        # Test iterative method with limited iterations for performance
        start_time = time.time()
        iter_transformer = ReconciliationTransformer(
            group_size=5,
            reconciliation_params={"method": "iterative_mint", "max_iterations": 5}
        )
        iter_transformer.fit(df)
        iter_transformed = iter_transformer.transform(df)
        iter_inverse = iter_transformer.inverse_transform(iter_transformed)
        iter_elapsed = time.time() - start_time
        
        # Should still be reasonable even with iterations
        self.assertLess(iter_elapsed, 10.0, f"Iterative ReconciliationTransformer took {iter_elapsed:.2f}s, too slow")
        
        # Basic correctness checks
        self.assertEqual(inverse_transformed.shape, df.shape)
        self.assertEqual(iter_inverse.shape, df.shape)

    def test_upscale_downscale_transformer_basic(self):
        """Test UpscaleDownscaleTransformer basic functionality."""
        # Create simple test data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'series1': np.random.randn(20).cumsum(),
            'series2': np.random.randn(20).cumsum(),
        }, index=dates)
        
        # Test get_new_params
        transformer = UpscaleDownscaleTransformer()
        params = transformer.get_new_params()
        self.assertTrue(params)
        self.assertIsInstance(params, dict)
        self.assertIn('mode', params)
        self.assertIn('factor', params)
        
        # Test upscale mode
        upscale_transformer = UpscaleDownscaleTransformer(mode='upscale', factor=2)
        upscale_transformer.fit(df)
        upscaled = upscale_transformer.transform(df)
        upscale_inverse = upscale_transformer.inverse_transform(upscaled)
        
        # Check basic properties
        self.assertIsInstance(upscaled, pd.DataFrame)
        self.assertIsInstance(upscale_inverse, pd.DataFrame)
        self.assertGreater(upscaled.shape[0], df.shape[0])  # Should have more rows
        self.assertEqual(upscaled.shape[1], df.shape[1])    # Same columns
        self.assertEqual(upscale_inverse.shape[1], df.shape[1])  # Same columns after inverse
        
        # Test downscale mode
        downscale_transformer = UpscaleDownscaleTransformer(mode='downscale', factor=2, down_method='mean')
        downscale_transformer.fit(df)
        downscaled = downscale_transformer.transform(df)
        downscale_inverse = downscale_transformer.inverse_transform(downscaled)
        
        # Check basic properties
        self.assertIsInstance(downscaled, pd.DataFrame)
        self.assertIsInstance(downscale_inverse, pd.DataFrame)
        self.assertLess(downscaled.shape[0], df.shape[0])    # Should have fewer rows
        self.assertEqual(downscaled.shape[1], df.shape[1])   # Same columns
        self.assertEqual(downscale_inverse.shape[1], df.shape[1])  # Same columns after inverse
        
        # Test error handling with invalid input
        with self.assertRaises(ValueError):
            UpscaleDownscaleTransformer(mode='invalid_mode')
        
        with self.assertRaises(ValueError):
            UpscaleDownscaleTransformer(factor=0)
