# -*- coding: utf-8 -*-
"""MLEnsemble testing."""
import unittest
import numpy as np
import pandas as pd
from autots.models.mlensemble import MLEnsemble


class MLEnsembleTest(unittest.TestCase):
    def test_model_basic(self):
        """Test basic functionality of MLEnsemble."""
        print("Starting MLEnsemble model tests")
        
        # Create simple test data
        n_timesteps = 150
        n_series = 5  # Test with multiple series to catch dimension issues
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')
        
        # Generate simple synthetic time series with trend and seasonality
        np.random.seed(42)
        data = np.random.randn(n_timesteps, n_series).astype(np.float32) * 0.5
        
        # Add trend and weekly seasonality
        trend = np.linspace(10, 15, n_timesteps)[:, None]
        weekly_season = np.sin(np.arange(n_timesteps) * 2 * np.pi / 7)[:, None]
        data += trend + weekly_season * 2
        
        df_train = pd.DataFrame(
            data, 
            index=dates, 
            columns=[f'series_{i}' for i in range(n_series)]
        )
        
        forecast_length = 14
        
        # Initialize model with minimal parameters for speed
        model = MLEnsemble(
            forecast_length=forecast_length,
            num_validations=1,  # Minimal validations for speed
            validation_method="backwards",
            datepart_method="simple",  # Test datepart feature creation
            models=[
                {
                    'Model': 'LastValueNaive',
                    'ModelParameters': '{}',
                    'TransformationParameters': '{}',
                },
                {
                    'Model': 'SeasonalNaive',
                    'ModelParameters': '{"seasonal_periods": 7}',
                    'TransformationParameters': '{}',
                },
            ],
            verbose=1,
            random_seed=42,
        )
        
        # Test fitting
        print("Testing MLEnsemble fit...")
        model.fit(df_train)
        
        # Verify model was trained
        self.assertIsNotNone(model.regr, "Regressor should be initialized after fit")
        self.assertIsNotNone(model.X, "Training features should be created")
        self.assertIsNotNone(model.y, "Training targets should be created")
        
        # Verify training data shapes
        self.assertEqual(model.X.ndim, 2, "X should be 2-dimensional")
        self.assertEqual(model.y.ndim, 1, "y should be 1-dimensional")
        
        # Test prediction
        print("Testing MLEnsemble predict...")
        prediction = model.predict(forecast_length=forecast_length)
        
        # Verify prediction object
        self.assertIsNotNone(prediction, "Prediction should not be None")
        self.assertIsNotNone(prediction.forecast, "Forecast should not be None")
        
        # Verify forecast shape
        expected_shape = (forecast_length, n_series)
        self.assertEqual(
            prediction.forecast.shape, 
            expected_shape,
            f"Forecast shape should be {expected_shape}, got {prediction.forecast.shape}"
        )
        
        # Verify forecast contains no null values
        self.assertFalse(
            prediction.forecast.isnull().any().any(),
            "Forecast should not contain null values"
        )
        
        # Verify forecast is numeric
        self.assertTrue(
            np.isfinite(prediction.forecast.values).all(),
            "Forecast should contain only finite values"
        )
        
        # Verify prediction intervals exist and are properly ordered
        self.assertIsNotNone(prediction.upper_forecast, "Upper forecast should exist")
        self.assertIsNotNone(prediction.lower_forecast, "Lower forecast should exist")
        
        # Verify column names match
        self.assertEqual(
            list(prediction.forecast.columns),
            list(df_train.columns),
            "Forecast columns should match training data columns"
        )
        
        # Verify index is correct
        self.assertEqual(
            len(prediction.forecast.index),
            forecast_length,
            "Forecast index length should match forecast_length"
        )
        
        print("MLEnsemble basic tests passed!")

    def test_model_different_series_counts(self):
        """Test MLEnsemble with different numbers of series to catch hardcoded dimension bugs."""
        print("Testing MLEnsemble with different series counts...")
        
        forecast_length = 10
        
        for n_series in [3, 7, 21, 27]:  # Test various series counts including the problematic 27
            print(f"  Testing with {n_series} series...")
            
            n_timesteps = 100
            dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')
            
            np.random.seed(42)
            data = np.random.randn(n_timesteps, n_series) * 0.5 + 10
            
            df_train = pd.DataFrame(
                data, 
                index=dates, 
                columns=[f'series_{i}' for i in range(n_series)]
            )
            
            model = MLEnsemble(
                forecast_length=forecast_length,
                num_validations=0,  # No validations for speed
                validation_method="backwards",
                datepart_method="simple",
                models=[
                    {
                        'Model': 'LastValueNaive',
                        'ModelParameters': '{}',
                        'TransformationParameters': '{}',
                    },
                ],
                verbose=0,
                random_seed=42,
            )
            
            # This should not raise dimension mismatch errors
            try:
                model.fit(df_train)
                prediction = model.predict(forecast_length=forecast_length)
                
                # Verify correct shape
                self.assertEqual(
                    prediction.forecast.shape, 
                    (forecast_length, n_series),
                    f"Failed for {n_series} series: shape mismatch"
                )
                
            except ValueError as e:
                if "dimension" in str(e).lower():
                    self.fail(f"Dimension mismatch error with {n_series} series: {e}")
                else:
                    raise
        
        print("Different series count tests passed!")

    def test_model_with_datepart_methods(self):
        """Test MLEnsemble with different datepart methods."""
        print("Testing MLEnsemble with different datepart methods...")
        
        n_timesteps = 100
        n_series = 5
        forecast_length = 7
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')
        
        np.random.seed(42)
        data = np.random.randn(n_timesteps, n_series) * 0.5 + 10
        df_train = pd.DataFrame(
            data, 
            index=dates, 
            columns=[f'series_{i}' for i in range(n_series)]
        )
        
        # Test with various datepart methods
        datepart_methods = [None, "simple", "expanded", "simple_binarized"]
        
        for method in datepart_methods:
            print(f"  Testing with datepart_method='{method}'...")
            
            model = MLEnsemble(
                forecast_length=forecast_length,
                num_validations=0,
                validation_method="backwards",
                datepart_method=method,
                models=[
                    {
                        'Model': 'LastValueNaive',
                        'ModelParameters': '{}',
                        'TransformationParameters': '{}',
                    },
                ],
                verbose=0,
                random_seed=42,
            )
            
            model.fit(df_train)
            prediction = model.predict(forecast_length=forecast_length)
            
            # Verify correct shape regardless of datepart method
            self.assertEqual(
                prediction.forecast.shape, 
                (forecast_length, n_series),
                f"Failed with datepart_method='{method}'"
            )
        
        print("Datepart methods tests passed!")


if __name__ == '__main__':
    unittest.main()
