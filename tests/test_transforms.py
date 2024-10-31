# -*- coding: utf-8 -*-
"""Overall testing."""
import unittest
import numpy as np
import pandas as pd
from autots.datasets import (
    load_daily, load_monthly, load_artificial, load_sine
)
from autots.tools.transform import ThetaTransformer

class TestTransforms(unittest.TestCase):
    
    def test_theta(self):
        # Sample DataFrame with a DatetimeIndex and multiple time series columns
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'series1': np.random.randn(100).cumsum(),
            'series2': np.random.randn(100).cumsum(),
        }, index=dates)

        # Initialize the transformer with custom theta values
        theta_values = [0, 1, 2]  # Example of custom theta values
        theta_transformer = ThetaTransformer(theta_values=theta_values)
        
        params = theta_transformer.get_new_params()
        self.assertTrue(params)

        # Fit the transformer
        theta_transformer.fit(data)

        # Transform the data
        transformed_data = theta_transformer.transform(data)

        # Inverse transform to reconstruct the original data
        reconstructed_data = theta_transformer.inverse_transform(transformed_data)

        # Verify that the reconstructed data matches the original data
        # Note: Due to numerical precision, a small tolerance is acceptable
        self.assertTrue(np.allclose(data.values, reconstructed_data.values, atol=1e-8))
