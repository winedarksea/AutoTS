# -*- coding: utf-8 -*-
"""MambaSSM testing."""
import unittest
import pandas as pd
import numpy as np
from autots.models.deepssm import MambaSSM


class MambaSSMTest(unittest.TestCase):
    def test_model(self):
        """Test basic functionality of MambaSSM."""
        print("Starting MambaSSM model tests")
        
        # Create simple test data for speed
        n_timesteps = 200  # Minimal data for faster testing
        n_series = 3
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')
        
        # Generate simple synthetic time series with trend and seasonality
        np.random.seed(42)
        data = np.random.randn(n_timesteps, n_series).astype(np.float32)
        
        # Add trend and weekly seasonality
        trend = np.linspace(0, 2, n_timesteps)[:, None]
        weekly_season = np.sin(np.arange(n_timesteps) * 2 * np.pi / 7)[:, None]
        data += trend + weekly_season * 0.5
        
        df_train = pd.DataFrame(
            data, 
            index=dates, 
            columns=[f'series_{i}' for i in range(n_series)]
        )
        
        forecast_length = 14
        
        # Initialize model with minimal parameters for speed
        model = MambaSSM(
            context_length=30,  # Reduced for speed
            epochs=2,           # Minimal epochs for speed
            batch_size=16,      # Small batch size
            d_model=16,         # Small model size
            n_layers=1,         # Single layer
            d_state=4,          # Small state size
            verbose=1,
            random_seed=42,
            changepoint_method="basic",    # Test changepoint features
            changepoint_params={"changepoint_spacing": 60, "changepoint_distance_end": 30}
        )
        
        # Test fitting
        print("Testing model fit...")
        model.fit(df_train)
        
        # Verify model was trained
        self.assertIsNotNone(model.model, "Model should be initialized after fit")
        self.assertIsNotNone(model.scaler_means, "Scaler means should be set")
        self.assertIsNotNone(model.scaler_stds, "Scaler stds should be set")
        
        # Test prediction
        print("Testing model predict...")
        prediction = model.predict(forecast_length=forecast_length)
        
        # Verify prediction object
        self.assertIsNotNone(prediction, "Prediction should not be None")
        self.assertIsNotNone(prediction.forecast, "Forecast should not be None")
        
        # Verify forecast shape and index
        expected_shape = (forecast_length, n_series)
        self.assertEqual(
            prediction.forecast.shape, 
            expected_shape,
            f"Forecast shape should be {expected_shape}"
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
        
        # Upper should be >= forecast >= lower (approximately, allowing for numerical precision)
        upper_ge_forecast = (prediction.upper_forecast >= prediction.forecast - 1e-6).all().all()
        forecast_ge_lower = (prediction.forecast >= prediction.lower_forecast - 1e-6).all().all()
        
        self.assertTrue(upper_ge_forecast, "Upper forecast should be >= point forecast")
        self.assertTrue(forecast_ge_lower, "Point forecast should be >= lower forecast")
        
        # Verify column names match
        self.assertEqual(
            list(prediction.forecast.columns),
            list(df_train.columns),
            "Forecast columns should match training data columns"
        )
        
        # Verify forecast index is proper datetime
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(prediction.forecast.index),
            "Forecast index should be datetime"
        )
        
        # Test model parameters retrieval
        params = model.get_params()
        self.assertIsInstance(params, dict, "get_params should return a dictionary")
        self.assertIn('epochs', params, "Parameters should include epochs")
        self.assertEqual(params['epochs'], 2, "Epochs parameter should match initialization")
        
        # Test changepoint parameters
        self.assertIn('changepoint_method', params, "Parameters should include changepoint_method")
        self.assertIn('changepoint_params', params, "Parameters should include changepoint_params")
        self.assertEqual(params['changepoint_method'], "basic", "Changepoint method should match initialization")
        self.assertEqual(params['changepoint_params']['changepoint_spacing'], 60, "Changepoint spacing should match initialization")
        self.assertEqual(params['changepoint_params']['changepoint_distance_end'], 30, "Changepoint distance end should match initialization")
        
        print("All MambaSSM tests passed!")

    def test_model_with_regressors(self):
        """Test MambaSSM with future regressors."""
        print("Testing MambaSSM with future regressors")
        
        # Create minimal test data
        n_timesteps = 100
        n_series = 2
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')
        
        np.random.seed(42)
        data = np.random.randn(n_timesteps, n_series).astype(np.float32)
        df_train = pd.DataFrame(data, index=dates, columns=['A', 'B'])
        
        # Create simple future regressor
        regressor_train = pd.DataFrame(
            np.random.randn(n_timesteps, 1),
            index=dates,
            columns=['regressor_1']
        )
        
        forecast_length = 7
        forecast_dates = pd.date_range(
            start=dates[-1] + pd.Timedelta(days=1),
            periods=forecast_length,
            freq='D'
        )
        regressor_forecast = pd.DataFrame(
            np.random.randn(forecast_length, 1),
            index=forecast_dates,
            columns=['regressor_1']
        )
        
        # Test with minimal model for speed
        model = MambaSSM(
            context_length=20,
            epochs=1,
            batch_size=8,
            d_model=8,
            n_layers=1,
            d_state=2,
            verbose=0,  # Reduce verbosity for test speed
            random_seed=42
        )
        
        # Fit with regressor
        model.fit(df_train, future_regressor=regressor_train)
        
        # Predict with regressor
        prediction = model.predict(
            forecast_length=forecast_length,
            future_regressor=regressor_forecast
        )
        
        # Basic validation
        self.assertIsNotNone(prediction.forecast, "Forecast with regressors should not be None")
        self.assertEqual(prediction.forecast.shape, (forecast_length, n_series))
        self.assertFalse(prediction.forecast.isnull().any().any())
        
        print("MambaSSM with regressors test passed!")

    def test_changepoint_features(self):
        """Test MambaSSM with explicit changepoint configuration."""
        print("Testing MambaSSM with changepoint features")
        
        # Create minimal test data
        n_timesteps = 120  # Enough data for changepoints
        n_series = 2
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')
        
        np.random.seed(42)
        # Create data with a clear trend change
        data1 = np.random.randn(n_timesteps // 2, n_series).astype(np.float32)
        data2 = np.random.randn(n_timesteps // 2, n_series).astype(np.float32) + 2  # Level shift
        data = np.vstack([data1, data2])
        
        df_train = pd.DataFrame(data, index=dates, columns=['A', 'B'])
        
        forecast_length = 7
        
        # Test with specific changepoint parameters
        model = MambaSSM(
            context_length=30,
            epochs=1,  # Minimal for speed
            batch_size=8,
            d_model=8,
            n_layers=1,
            d_state=2,
            verbose=0,
            random_seed=42,
            changepoint_method="basic",  # Should detect the change around day 60
            changepoint_params={"changepoint_spacing": 25, "changepoint_distance_end": 10}
        )
        
        # Fit and predict
        model.fit(df_train)
        prediction = model.predict(forecast_length=forecast_length)
        
        # Basic validation
        self.assertIsNotNone(prediction.forecast, "Forecast with changepoints should not be None")
        self.assertEqual(prediction.forecast.shape, (forecast_length, n_series))
        self.assertFalse(prediction.forecast.isnull().any().any())
        
        # Verify changepoint parameters were stored correctly
        params = model.get_params()
        self.assertEqual(params['changepoint_method'], "basic")
        self.assertEqual(params['changepoint_params']['changepoint_spacing'], 25)
        self.assertEqual(params['changepoint_params']['changepoint_distance_end'], 10)
        
        print("MambaSSM changepoint features test passed!")

    def test_get_new_params_includes_changepoints(self):
        """Test that get_new_params includes changepoint parameters."""
        print("Testing get_new_params with changepoint parameters")
        
        # Generate new parameters
        new_params = MambaSSM.get_new_params()
        
        # Verify changepoint parameters are included
        self.assertIn('changepoint_method', new_params, "New params should include changepoint_method")
        self.assertIn('changepoint_params', new_params, "New params should include changepoint_params")
        
        # Verify method is valid
        valid_methods = ['basic', 'pelt', 'l1_fused_lasso', 'l1_total_variation']
        self.assertIn(new_params['changepoint_method'], valid_methods, "Changepoint method should be valid")
        
        # Verify parameters are appropriate for the method
        method = new_params['changepoint_method']
        params = new_params['changepoint_params']
        self.assertIsInstance(params, dict, "Changepoint params should be a dictionary")
        
        if method == 'basic':
            self.assertIn('changepoint_spacing', params)
            self.assertIn('changepoint_distance_end', params)
            self.assertIsInstance(params['changepoint_spacing'], int)
            self.assertIsInstance(params['changepoint_distance_end'], int)
            self.assertGreater(params['changepoint_spacing'], 0)
            self.assertGreater(params['changepoint_distance_end'], 0)
        elif method == 'pelt':
            self.assertIn('penalty', params)
            self.assertIn('loss_function', params)
            self.assertIn('min_segment_length', params)
        elif method in ['l1_fused_lasso', 'l1_total_variation']:
            self.assertIn('lambda_reg', params)
        
        print("get_new_params changepoint test passed!")

    def test_l1_changepoint_methods(self):
        """Test MambaSSM with L1 changepoint methods."""
        print("Testing MambaSSM with L1 changepoint methods")
        
        # Create minimal test data with clear level shifts
        n_timesteps = 40
        n_series = 1
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')
        
        np.random.seed(42)
        # Create data with clear level shifts for better L1 detection
        data1 = np.random.randn(20, n_series).astype(np.float32) + 10
        data2 = np.random.randn(20, n_series).astype(np.float32) + 15  # Clear level shift
        data = np.vstack([data1, data2])
        
        df_train = pd.DataFrame(data, index=dates, columns=['A'])
        
        forecast_length = 5
        
        # Test L1 Fused Lasso method
        model_l1fl = MambaSSM(
            context_length=10,
            epochs=1,
            batch_size=4,
            d_model=8,
            n_layers=1,
            d_state=2,
            verbose=0,
            random_seed=42,
            changepoint_method="l1_fused_lasso",
            changepoint_params={"lambda_reg": 2.0}
        )
        
        model_l1fl.fit(df_train)
        prediction_l1fl = model_l1fl.predict(forecast_length=forecast_length)
        
        self.assertIsNotNone(prediction_l1fl.forecast, "L1 fused lasso forecast should not be None")
        self.assertEqual(prediction_l1fl.forecast.shape, (forecast_length, n_series))
        
        # Test L1 Total Variation method
        model_l1tv = MambaSSM(
            context_length=10,
            epochs=1,
            batch_size=4,
            d_model=8,
            n_layers=1,
            d_state=2,
            verbose=0,
            random_seed=42,
            changepoint_method="l1_total_variation",
            changepoint_params={"lambda_reg": 1.0}
        )
        
        model_l1tv.fit(df_train)
        prediction_l1tv = model_l1tv.predict(forecast_length=forecast_length)
        
        self.assertIsNotNone(prediction_l1tv.forecast, "L1 total variation forecast should not be None")
        self.assertEqual(prediction_l1tv.forecast.shape, (forecast_length, n_series))
        
        # Test parameter retrieval for L1 methods
        params_l1fl = model_l1fl.get_params()
        self.assertEqual(params_l1fl['changepoint_method'], "l1_fused_lasso")
        self.assertEqual(params_l1fl['changepoint_params']['lambda_reg'], 2.0)
        
        params_l1tv = model_l1tv.get_params()
        self.assertEqual(params_l1tv['changepoint_method'], "l1_total_variation")
        self.assertEqual(params_l1tv['changepoint_params']['lambda_reg'], 1.0)
        
        print("L1 changepoint methods test passed!")


if __name__ == '__main__':
    unittest.main()
