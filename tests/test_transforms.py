# -*- coding: utf-8 -*-
"""Overall testing."""
import unittest
import numpy as np
import pandas as pd
from autots.datasets import (
    load_daily, load_monthly, load_artificial, load_sine
)
from autots.tools.transform import ThetaTransformer, FIRFilter

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
