"""Tests for statsmodels-based forecasting models."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from autots.models.statsmodels import ARDL  # noqa: E402


class TestARDL(unittest.TestCase):
    """Test cases for ARDL model."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.data_single = pd.DataFrame({
            'series1': np.random.randn(100).cumsum() + 10,
        }, index=dates)
        
        self.data_multi = pd.DataFrame({
            'series1': np.random.randn(100).cumsum() + 10,
            'series2': np.random.randn(100).cumsum() + 20,
        }, index=dates)

    def test_ardl_basic_fit_predict(self):
        """Test basic ARDL fit and predict."""
        model = ARDL(lags=1, trend='c', order=0, verbose=0)
        model.fit(self.data_single)
        prediction = model.predict(forecast_length=14)
        
        self.assertEqual(prediction.forecast.shape, (14, 1))
        self.assertEqual(len(prediction.forecast.columns), 1)
        self.assertIsNotNone(prediction.upper_forecast)
        self.assertIsNotNone(prediction.lower_forecast)

    def test_ardl_list_regression_type(self):
        """Test ARDL with list of regression types."""
        model = ARDL(
            regression_type=['common_fourier_rw', 'holiday'],
            lags=1,
            order=1,
            holiday_country='US',
            verbose=0
        )
        model.fit(self.data_multi)
        
        # Check that regressor_train was created
        self.assertIsNotNone(model.regressor_train)
        
        # Check that it has multiple columns (combined features)
        self.assertGreater(model.regressor_train.shape[1], 1)
        
        # Make prediction
        prediction = model.predict(forecast_length=14)
        self.assertEqual(prediction.forecast.shape, (14, 2))

    def test_ardl_list_regression_feature_combination(self):
        """Test that list regression types combine features correctly."""
        # Test with combined regression types
        model_combined = ARDL(
            regression_type=['common_fourier_rw', 'holiday'],
            lags=1,
            order=1,
            holiday_country='US',
            verbose=0
        )
        model_combined.fit(self.data_single)
        
        # Test individual regression types
        model_fourier = ARDL(
            regression_type='common_fourier_rw',
            lags=1,
            order=1,
            verbose=0
        )
        model_fourier.fit(self.data_single)
        
        model_holiday = ARDL(
            regression_type='holiday',
            lags=1,
            order=0,
            holiday_country='US',
            verbose=0
        )
        model_holiday.fit(self.data_single)
        
        # Combined should have sum of individual feature counts
        expected_cols = (model_fourier.regressor_train.shape[1] + 
                        model_holiday.regressor_train.shape[1])
        self.assertEqual(model_combined.regressor_train.shape[1], expected_cols)

    def test_ardl_list_regression_prediction(self):
        """Test that predictions work with list regression types."""
        model = ARDL(
            regression_type=['common_fourier_rw', 'holiday'],
            lags=2,
            trend='c',
            order=1,
            causal=False,
            holiday_country='US',
            verbose=0
        )
        model.fit(self.data_multi)
        
        forecast_length = 30
        prediction = model.predict(forecast_length=forecast_length)
        
        # Check forecast shape
        self.assertEqual(prediction.forecast.shape[0], forecast_length)
        self.assertEqual(prediction.forecast.shape[1], 2)
        
        # Check that forecast values are not NaN
        self.assertFalse(prediction.forecast.isna().any().any())
        
        # Check prediction intervals exist
        self.assertIsNotNone(prediction.upper_forecast)
        self.assertIsNotNone(prediction.lower_forecast)
        self.assertEqual(prediction.upper_forecast.shape, prediction.forecast.shape)
        self.assertEqual(prediction.lower_forecast.shape, prediction.forecast.shape)

    def test_ardl_get_params_with_list(self):
        """Test that get_params correctly returns list regression type."""
        model = ARDL(
            regression_type=['common_fourier_rw', 'holiday'],
            lags=2,
            trend='c',
            order=1,
            verbose=0
        )
        
        params = model.get_params()
        
        self.assertIsInstance(params['regression_type'], list)
        self.assertIn('common_fourier_rw', params['regression_type'])
        self.assertIn('holiday', params['regression_type'])
        self.assertEqual(params['lags'], 2)
        self.assertEqual(params['trend'], 'c')
        self.assertEqual(params['order'], 1)

    def test_ardl_get_new_params_can_generate_list(self):
        """Test that get_new_params can generate list regression types."""
        model = ARDL(verbose=0)
        
        # Try multiple times since it's random
        found_list = False
        for _ in range(20):
            params = model.get_new_params()
            if isinstance(params['regression_type'], list):
                found_list = True
                self.assertGreater(len(params['regression_type']), 1)
                break
        
        # Should find at least one list in 20 tries (20% probability each)
        self.assertTrue(found_list, "get_new_params should occasionally generate list regression types")

    def test_ardl_single_regression_type(self):
        """Test ARDL with single regression type still works."""
        # Test holiday only
        model_holiday = ARDL(
            regression_type='holiday',
            lags=1,
            order=0,
            holiday_country='US',
            verbose=0
        )
        model_holiday.fit(self.data_single)
        prediction = model_holiday.predict(forecast_length=7)
        self.assertEqual(prediction.forecast.shape, (7, 1))
        
        # Test datepart method
        model_datepart = ARDL(
            regression_type='common_fourier_rw',
            lags=1,
            order=1,
            verbose=0
        )
        model_datepart.fit(self.data_single)
        prediction = model_datepart.predict(forecast_length=7)
        self.assertEqual(prediction.forecast.shape, (7, 1))
        
        # Test None
        model_none = ARDL(
            regression_type=None,
            lags=1,
            order=0,
            verbose=0
        )
        model_none.fit(self.data_single)
        self.assertIsNone(model_none.regressor_train)
        prediction = model_none.predict(forecast_length=7)
        self.assertEqual(prediction.forecast.shape, (7, 1))

    def test_ardl_list_with_none_ignored(self):
        """Test that None in list is properly ignored."""
        model = ARDL(
            regression_type=['common_fourier_rw', None, 'holiday'],
            lags=1,
            order=1,
            holiday_country='US',
            verbose=0
        )
        model.fit(self.data_single)
        
        # Should still create combined regressor
        self.assertIsNotNone(model.regressor_train)
        self.assertGreater(model.regressor_train.shape[1], 1)

    def test_ardl_multivariate_forecast(self):
        """Test ARDL forecasting multiple series."""
        model = ARDL(
            regression_type=['common_fourier_rw', 'holiday'],
            lags=2,
            trend='c',
            order=1,
            holiday_country='US',
            verbose=0,
            n_jobs=1
        )
        
        # Create data with 2 series
        model.fit(self.data_multi)
        prediction = model.predict(forecast_length=14)
        
        # Should forecast both series
        self.assertEqual(len(prediction.forecast.columns), 2)
        self.assertIn('series1', prediction.forecast.columns)
        self.assertIn('series2', prediction.forecast.columns)


if __name__ == '__main__':
    unittest.main()
