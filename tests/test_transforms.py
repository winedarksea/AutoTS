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
