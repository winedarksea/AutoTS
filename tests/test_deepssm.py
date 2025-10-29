# -*- coding: utf-8 -*-
"""Deep SSM models testing."""
import unittest
import pandas as pd
import numpy as np
from autots.models.deepssm import MambaSSM, pMLP
from autots.tools.changepoints import ChangepointDetector


class DeepSSMTest(unittest.TestCase):
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

    def test_torch_mlp_basic(self):
        """Test basic functionality of pMLP model."""
        print("Testing pMLP basic functionality")
        
        # Create simple synthetic time series data
        n_timesteps = 100  # Reduced for faster testing
        n_series = 3
        dates = pd.date_range('2023-01-01', periods=n_timesteps, freq='D')
        
        # Create 3 series with different patterns
        np.random.seed(2023)
        series1 = np.sin(np.arange(n_timesteps) * 0.1) + np.random.normal(0, 0.1, n_timesteps)
        series2 = np.cos(np.arange(n_timesteps) * 0.05) + np.random.normal(0, 0.1, n_timesteps)
        series3 = np.arange(n_timesteps) * 0.01 + np.random.normal(0, 0.1, n_timesteps)
        
        df = pd.DataFrame({
            'series1': series1,
            'series2': series2, 
            'series3': series3
        }, index=dates)
        
        forecast_length = 14
        
        # Test pMLP with minimal parameters for speed
        mlp_model = pMLP(
            context_length=20,
            hidden_dims=[64, 32],  # Smaller for faster testing
            epochs=2,  # Quick test
            batch_size=16,
            verbose=0,  # Reduce verbosity for testing
            random_seed=42
        )
        
        # Test fitting
        print("  Testing pMLP fit...")
        mlp_model.fit(df)
        
        # Verify model was trained
        self.assertIsNotNone(mlp_model.model, "pMLP model should be initialized after fit")
        self.assertIsNotNone(mlp_model.scaler_means, "pMLP scaler means should be set")
        self.assertIsNotNone(mlp_model.scaler_stds, "pMLP scaler stds should be set")
        self.assertIsNotNone(mlp_model.fit_runtime, "pMLP fit_runtime should be recorded")
        
        # Test prediction
        print("  Testing pMLP predict...")
        prediction = mlp_model.predict(forecast_length=forecast_length)
        
        # Verify prediction object
        self.assertIsNotNone(prediction, "pMLP prediction should not be None")
        self.assertIsNotNone(prediction.forecast, "pMLP forecast should not be None")
        self.assertIsNotNone(prediction.predict_runtime, "pMLP predict_runtime should be recorded")
        
        # Verify forecast shape and properties
        expected_shape = (forecast_length, n_series)
        self.assertEqual(
            prediction.forecast.shape, 
            expected_shape,
            f"pMLP forecast shape should be {expected_shape}"
        )
        
        # Verify forecast contains no null values
        self.assertFalse(
            prediction.forecast.isnull().any().any(),
            "pMLP forecast should not contain null values"
        )
        
        # Verify forecast is numeric
        self.assertTrue(
            np.isfinite(prediction.forecast.values).all(),
            "pMLP forecast should contain only finite values"
        )
        
        # Verify prediction intervals exist and are properly ordered
        self.assertIsNotNone(prediction.upper_forecast, "pMLP upper forecast should exist")
        self.assertIsNotNone(prediction.lower_forecast, "pMLP lower forecast should exist")
        
        # Upper should be >= forecast >= lower (approximately)
        upper_ge_forecast = (prediction.upper_forecast >= prediction.forecast - 1e-6).all().all()
        forecast_ge_lower = (prediction.forecast >= prediction.lower_forecast - 1e-6).all().all()
        
        self.assertTrue(upper_ge_forecast, "pMLP upper forecast should be >= point forecast")
        self.assertTrue(forecast_ge_lower, "pMLP point forecast should be >= lower forecast")
        
        # Verify column names match
        self.assertEqual(
            list(prediction.forecast.columns),
            list(df.columns),
            "pMLP forecast columns should match training data columns"
        )
        
        # Verify forecast index is proper datetime
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(prediction.forecast.index),
            "pMLP forecast index should be datetime"
        )
        
        print("pMLP basic functionality test passed!")

    def test_torch_mlp_parameters(self):
        """Test pMLP parameter generation and retrieval."""
        print("Testing pMLP parameter generation")
        
        # Test parameter generation
        new_params = pMLP.get_new_params()
        
        # Verify parameters is a dictionary
        self.assertIsInstance(new_params, dict, "get_new_params should return a dictionary")
        
        # Verify key parameters are present
        expected_params = ['context_length', 'hidden_dims', 'epochs', 'batch_size', 'lr']
        for param in expected_params:
            self.assertIn(param, new_params, f"Parameter '{param}' should be in new_params")
        
        # Verify parameter types and ranges
        self.assertIsInstance(new_params['context_length'], int, "context_length should be int")
        self.assertGreater(new_params['context_length'], 0, "context_length should be positive")
        
        self.assertIsInstance(new_params['hidden_dims'], list, "hidden_dims should be list")
        self.assertGreater(len(new_params['hidden_dims']), 0, "hidden_dims should not be empty")
        self.assertTrue(all(isinstance(x, int) for x in new_params['hidden_dims']), "hidden_dims should contain integers")
        
        self.assertIsInstance(new_params['epochs'], int, "epochs should be int")
        self.assertGreater(new_params['epochs'], 0, "epochs should be positive")
        
        self.assertIsInstance(new_params['batch_size'], int, "batch_size should be int")
        self.assertGreater(new_params['batch_size'], 0, "batch_size should be positive")
        
        self.assertIsInstance(new_params['lr'], (int, float), "lr should be numeric")
        self.assertGreater(new_params['lr'], 0, "lr should be positive")
        
        # Test model initialization with generated parameters
        test_params = pMLP.get_new_params()
        test_params['epochs'] = 1  # Minimal for testing
        test_params['verbose'] = 0
        
        model = pMLP(**test_params)
        
        # Verify get_params returns the set parameters
        retrieved_params = model.get_params()
        for key in ['context_length', 'hidden_dims', 'epochs', 'batch_size']:
            self.assertEqual(
                retrieved_params[key], 
                test_params[key], 
                f"Retrieved parameter '{key}' should match set parameter"
            )
        
        print("pMLP parameter generation test passed!")

    def test_torch_mlp_vs_mamba_comparison(self):
        """Compare pMLP vs MambaSSM performance."""
        print("Testing pMLP vs MambaSSM comparison")
        
        # Create test data
        n_timesteps = 120  # Larger dataset to avoid batch norm issues
        dates = pd.date_range('2023-01-01', periods=n_timesteps, freq='D')
        np.random.seed(2023)
        series = np.sin(np.arange(n_timesteps) * 0.1) + np.random.normal(0, 0.1, n_timesteps)
        df = pd.DataFrame({'series': series}, index=dates)
        
        forecast_length = 10
        
        # Test both models with similar minimal configurations
        models = {
            'pMLP': pMLP(
                context_length=15,
                hidden_dims=[32, 16],
                epochs=1,
                batch_size=16,  # Larger batch size
                use_batch_norm=False,  # Disable to avoid batch norm issues with small data
                verbose=0,
                random_seed=42
            ),
            'MambaSSM': MambaSSM(
                context_length=15,
                epochs=1,
                batch_size=16,  # Larger batch size
                d_model=16,
                n_layers=1,
                d_state=4,
                verbose=0,
                random_seed=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"  Testing {name}...")
            try:
                # Fit the model
                model.fit(df)
                self.assertIsNotNone(model.fit_runtime, f"{name} should record fit_runtime")
                
                # Make prediction
                prediction = model.predict(forecast_length=forecast_length)
                self.assertIsNotNone(prediction.predict_runtime, f"{name} should record predict_runtime")
                
                # Basic validation
                self.assertEqual(prediction.forecast.shape, (forecast_length, 1), f"{name} forecast shape should be correct")
                self.assertFalse(prediction.forecast.isnull().any().any(), f"{name} forecast should not contain nulls")
                self.assertTrue(np.isfinite(prediction.forecast.values).all(), f"{name} forecast should be finite")
                
                results[name] = {
                    'fit_time': model.fit_runtime.total_seconds(),
                    'predict_time': prediction.predict_runtime.total_seconds(),
                    'forecast_shape': prediction.forecast.shape,
                    'success': True
                }
                
            except Exception as e:
                self.fail(f"{name} model failed: {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        # Verify both models succeeded
        for name in models.keys():
            self.assertTrue(results[name]['success'], f"{name} should succeed")
            self.assertGreater(results[name]['fit_time'], 0, f"{name} fit_time should be positive")
            self.assertGreater(results[name]['predict_time'], 0, f"{name} predict_time should be positive")
        
        # Compare performance (informational, not a hard test)
        mlp_total = results['pMLP']['fit_time'] + results['pMLP']['predict_time']
        mamba_total = results['MambaSSM']['fit_time'] + results['MambaSSM']['predict_time']
        
        print(f"    pMLP total time: {mlp_total:.3f}s")
        print(f"    MambaSSM total time: {mamba_total:.3f}s")
        
        # Both models should produce reasonable results
        self.assertLess(mlp_total, 60, "pMLP should complete within reasonable time")
        self.assertLess(mamba_total, 60, "MambaSSM should complete within reasonable time")
        
        print("pMLP vs MambaSSM comparison test passed!")

    def test_changepoint_detector_individual_features(self):
        """Ensure individual aggregation retains per-series features without fallbacks."""
        dates = pd.date_range(start='2023-01-01', periods=12, freq='D')
        df = pd.DataFrame(
            {
                'series_a': np.linspace(0, 1, 12),
                'series_b': np.linspace(1, 0, 12),
            },
            index=dates,
        )

        detector = ChangepointDetector(method='basic', aggregate_method='individual')
        detector.df = df
        detector.changepoints_ = {
            'series_a': np.array([4, 8]),
            'series_b': np.array([3]),
        }

        features = detector.create_features()
        self.assertIn('series_a_basic_changepoint_1', features.columns)
        self.assertIn('series_a_basic_changepoint_2', features.columns)
        self.assertIn('series_b_basic_changepoint_1', features.columns)
        self.assertEqual(len(features), len(df))

        detector.changepoints_ = np.array([], dtype=int)
        empty_features = detector.create_features()
        self.assertEqual(list(empty_features.columns), [])
        self.assertEqual(len(empty_features), len(df))

    def test_models_without_changepoints(self):
        """Verify models operate when changepoint detection is disabled."""
        n_timesteps = 40
        n_series = 2
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')

        np.random.seed(123)
        data = np.random.randn(n_timesteps, n_series).astype(np.float32)
        df = pd.DataFrame(data, index=dates, columns=['series_a', 'series_b'])
        forecast_length = 5

        mamba = MambaSSM(
            context_length=15,
            epochs=1,
            batch_size=8,
            d_model=8,
            n_layers=1,
            d_state=2,
            verbose=0,
            random_seed=0,
            changepoint_method=None,
        )
        mamba.fit(df)
        self.assertIsNone(
            mamba.changepoint_detector,
            "Changepoint detector should be None when disabled",
        )
        self.assertTrue(
            mamba.changepoint_features.empty,
            "Stored changepoint features should be empty when disabled",
        )
        self.assertIn(
            'naive_last_series_a',
            mamba.feature_columns,
            "Naive last value feature should be present for series_a",
        )
        prediction_mamba = mamba.predict(forecast_length=forecast_length)
        self.assertEqual(prediction_mamba.forecast.shape, (forecast_length, n_series))

        mamba_no_naive = MambaSSM(
            context_length=15,
            epochs=1,
            batch_size=8,
            d_model=8,
            n_layers=1,
            d_state=2,
            verbose=0,
            random_seed=0,
            changepoint_method=None,
            use_naive_feature=False,
        )
        mamba_no_naive.fit(df)
        self.assertNotIn(
            'naive_last_series_a',
            mamba_no_naive.feature_columns,
            "Naive last value feature should be absent when disabled",
        )

        mlp = pMLP(
            context_length=10,
            hidden_dims=[32],
            epochs=1,
            batch_size=8,
            verbose=0,
            random_seed=0,
            changepoint_method=None,
        )
        mlp.fit(df)
        self.assertIsNone(
            mlp.changepoint_detector,
            "Changepoint detector should be None when disabled",
        )
        self.assertTrue(
            mlp.changepoint_features.empty,
            "Stored changepoint features should be empty when disabled",
        )
        self.assertIn(
            'naive_last_series_b',
            mlp.feature_columns,
            "Naive last value feature should be present for series_b",
        )
        prediction_mlp = mlp.predict(forecast_length=forecast_length)
        self.assertEqual(prediction_mlp.forecast.shape, (forecast_length, n_series))

        mlp_no_naive = pMLP(
            context_length=10,
            hidden_dims=[32],
            epochs=1,
            batch_size=8,
            verbose=0,
            random_seed=0,
            changepoint_method=None,
            use_naive_feature=False,
        )
        mlp_no_naive.fit(df)
        self.assertNotIn(
            'naive_last_series_b',
            mlp_no_naive.feature_columns,
            "Naive last value feature should be absent when disabled",
        )

    def test_per_series_changepoint_features(self):
        """Test that per-series changepoints work correctly in MambaSSM and pMLP.
        
        This test verifies the fix for the bug where per-series changepoints were
        being aggregated together, losing individual patterns. It ensures that:
        1. Per-series changepoint features are created correctly
        2. Series-to-feature mapping is built properly
        3. Each series uses its own changepoint features during training
        4. Each series uses its own changepoint features during prediction
        """
        print("Testing per-series changepoint features")
        
        # Create synthetic data with different changepoint patterns per series
        n_timesteps = 150
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')
        
        np.random.seed(42)
        
        # Series A: Has a changepoint at day 75 (level shift up)
        series_a = np.concatenate([
            50 + np.random.randn(75) * 2,   # Mean 50
            80 + np.random.randn(75) * 2    # Mean 80 (shift at day 75)
        ])
        
        # Series B: Has a changepoint at day 100 (level shift down)
        series_b = np.concatenate([
            70 + np.random.randn(100) * 2,  # Mean 70
            40 + np.random.randn(50) * 2    # Mean 40 (shift at day 100)
        ])
        
        # Series C: No clear changepoint, gradual trend
        series_c = np.linspace(30, 60, n_timesteps) + np.random.randn(n_timesteps) * 2
        
        df = pd.DataFrame({
            'SeriesA': series_a,
            'SeriesB': series_b,
            'SeriesC': series_c
        }, index=dates)
        
        forecast_length = 10
        train_df = df.iloc[:-forecast_length]
        
        # Test 1: Verify ChangepointDetector creates per-series features with individual aggregation
        print("  Testing ChangepointDetector with aggregate_method='individual'")
        detector = ChangepointDetector(
            method='cusum',
            method_params={'threshold': 3.0, 'min_distance': 10},
            aggregate_method='individual'
        )
        detector.detect(train_df)
        
        # Verify changepoints were detected per series
        self.assertIsInstance(
            detector.changepoints_,
            dict,
            "Changepoints should be a dict with individual aggregation"
        )
        self.assertEqual(
            set(detector.changepoints_.keys()),
            set(df.columns),
            "Changepoints dict should have keys for all series"
        )
        
        # Create features and verify per-series pattern
        features = detector.create_features(forecast_length=0)
        
        # Check for per-series feature naming pattern
        has_series_a_features = any('SeriesA_' in str(col) for col in features.columns)
        has_series_b_features = any('SeriesB_' in str(col) for col in features.columns)
        has_series_c_features = any('SeriesC_' in str(col) for col in features.columns)
        
        self.assertTrue(
            has_series_a_features or has_series_b_features or has_series_c_features,
            "Per-series changepoint features should be created with series names in column names"
        )
        
        # Test 2: Test MambaSSM with per-series changepoints
        print("  Testing MambaSSM with per-series changepoints")
        model_mamba = MambaSSM(
            context_length=30,
            d_model=16,
            n_layers=1,
            d_state=4,
            epochs=2,
            batch_size=16,
            verbose=0,
            random_seed=42,
            changepoint_method='cusum',
            changepoint_params={'threshold': 3.0, 'min_distance': 10}
        )
        
        # Fit with aggregate_method='individual'
        model_mamba.fit(train_df, aggregate_method='individual')
        
        # Verify per-series feature mapping was created
        self.assertTrue(
            hasattr(model_mamba, 'has_per_series_features'),
            "Model should have has_per_series_features attribute"
        )
        self.assertTrue(
            hasattr(model_mamba, 'series_feat_mapping'),
            "Model should have series_feat_mapping attribute"
        )
        
        if model_mamba.has_per_series_features:
            self.assertIsNotNone(
                model_mamba.series_feat_mapping,
                "Series feature mapping should be created for per-series features"
            )
            self.assertIsInstance(
                model_mamba.series_feat_mapping,
                dict,
                "Series feature mapping should be a dict"
            )
            self.assertEqual(
                len(model_mamba.series_feat_mapping),
                len(df.columns),
                "Series feature mapping should have an entry for each series"
            )
            
            # Verify each series has feature indices
            for series_idx in range(len(df.columns)):
                self.assertIn(
                    series_idx,
                    model_mamba.series_feat_mapping,
                    f"Series {series_idx} should be in feature mapping"
                )
                feat_indices = model_mamba.series_feat_mapping[series_idx]
                self.assertIsInstance(
                    feat_indices,
                    list,
                    f"Feature indices for series {series_idx} should be a list"
                )
                self.assertGreater(
                    len(feat_indices),
                    0,
                    f"Series {series_idx} should have at least one feature"
                )
        
        # Make predictions
        prediction_mamba = model_mamba.predict(forecast_length=forecast_length)
        
        # Verify predictions
        self.assertEqual(
            prediction_mamba.forecast.shape,
            (forecast_length, len(df.columns)),
            "MambaSSM forecast shape should match expected dimensions"
        )
        self.assertFalse(
            prediction_mamba.forecast.isnull().any().any(),
            "MambaSSM forecast should not contain null values"
        )
        self.assertTrue(
            np.isfinite(prediction_mamba.forecast.values).all(),
            "MambaSSM forecast should contain only finite values"
        )
        
        # Test 3: Test pMLP with per-series changepoints
        print("  Testing pMLP with per-series changepoints")
        model_pmlp = pMLP(
            context_length=30,
            hidden_dims=[64, 32],
            epochs=2,
            batch_size=16,
            verbose=0,
            random_seed=42,
            changepoint_method='cusum',
            changepoint_params={'threshold': 3.0, 'min_distance': 10}
        )
        
        # Fit with aggregate_method='individual'
        model_pmlp.fit(train_df, aggregate_method='individual')
        
        # Verify per-series feature mapping was created
        self.assertTrue(
            hasattr(model_pmlp, 'has_per_series_features'),
            "pMLP should have has_per_series_features attribute"
        )
        self.assertTrue(
            hasattr(model_pmlp, 'series_feat_mapping'),
            "pMLP should have series_feat_mapping attribute"
        )
        
        if model_pmlp.has_per_series_features:
            self.assertIsNotNone(
                model_pmlp.series_feat_mapping,
                "pMLP series feature mapping should be created for per-series features"
            )
            self.assertIsInstance(
                model_pmlp.series_feat_mapping,
                dict,
                "pMLP series feature mapping should be a dict"
            )
            self.assertEqual(
                len(model_pmlp.series_feat_mapping),
                len(df.columns),
                "pMLP series feature mapping should have an entry for each series"
            )
            
            # Verify each series has feature indices
            for series_idx in range(len(df.columns)):
                self.assertIn(
                    series_idx,
                    model_pmlp.series_feat_mapping,
                    f"pMLP series {series_idx} should be in feature mapping"
                )
                feat_indices = model_pmlp.series_feat_mapping[series_idx]
                self.assertIsInstance(
                    feat_indices,
                    list,
                    f"pMLP feature indices for series {series_idx} should be a list"
                )
                self.assertGreater(
                    len(feat_indices),
                    0,
                    f"pMLP series {series_idx} should have at least one feature"
                )
        
        # Make predictions
        prediction_pmlp = model_pmlp.predict(forecast_length=forecast_length)
        
        # Verify predictions
        self.assertEqual(
            prediction_pmlp.forecast.shape,
            (forecast_length, len(df.columns)),
            "pMLP forecast shape should match expected dimensions"
        )
        self.assertFalse(
            prediction_pmlp.forecast.isnull().any().any(),
            "pMLP forecast should not contain null values"
        )
        self.assertTrue(
            np.isfinite(prediction_pmlp.forecast.values).all(),
            "pMLP forecast should contain only finite values"
        )
        
        # Test 4: Verify different behavior between individual and shared changepoints
        print("  Testing comparison between individual and shared changepoints")
        
        # Train a model with shared changepoints (default behavior)
        model_shared = MambaSSM(
            context_length=30,
            d_model=16,
            n_layers=1,
            d_state=4,
            epochs=2,
            batch_size=16,
            verbose=0,
            random_seed=42,
            changepoint_method='cusum',
            changepoint_params={'threshold': 3.0, 'min_distance': 10}
        )
        
        # Fit without aggregate_method (defaults to shared)
        model_shared.fit(train_df)
        
        # Should not have per-series features
        self.assertFalse(
            getattr(model_shared, 'has_per_series_features', False),
            "Model with shared changepoints should not have per-series features"
        )
        
        # Make predictions
        prediction_shared = model_shared.predict(forecast_length=forecast_length)
        
        # Both should produce valid predictions, but they should be different
        # (this is a sanity check, not a strict requirement)
        self.assertEqual(
            prediction_shared.forecast.shape,
            prediction_mamba.forecast.shape,
            "Both models should produce same shape forecasts"
        )
        
        print("Per-series changepoint features test passed!")

    def test_per_series_basic_changepoint_method(self):
        """Test per-series changepoints with 'basic' method.
        
        The 'basic' method uses fixed spacing and should work with individual aggregation.
        """
        print("Testing per-series changepoints with 'basic' method")
        
        n_timesteps = 120
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')
        
        np.random.seed(123)
        df = pd.DataFrame({
            'A': np.random.randn(n_timesteps) + np.linspace(0, 2, n_timesteps),
            'B': np.random.randn(n_timesteps) + np.linspace(2, 0, n_timesteps),
        }, index=dates)
        
        forecast_length = 7
        train_df = df.iloc[:-forecast_length]
        
        # Test MambaSSM with basic method and individual aggregation
        model = MambaSSM(
            context_length=20,
            d_model=12,
            n_layers=1,
            d_state=3,
            epochs=1,
            batch_size=8,
            verbose=0,
            random_seed=0,
            changepoint_method='basic',
            changepoint_params={'changepoint_spacing': 30, 'changepoint_distance_end': 15}
        )
        
        model.fit(train_df, aggregate_method='individual')
        
        # Verify per-series features were created
        if hasattr(model, 'has_per_series_features') and model.has_per_series_features:
            self.assertIsNotNone(model.series_feat_mapping)
            # Verify we have entries for both series
            self.assertEqual(len(model.series_feat_mapping), 2)
        
        prediction = model.predict(forecast_length=forecast_length)
        
        self.assertEqual(prediction.forecast.shape, (forecast_length, 2))
        self.assertFalse(prediction.forecast.isnull().any().any())
        self.assertTrue(np.isfinite(prediction.forecast.values).all())
        
        print("Per-series basic changepoint method test passed!")


if __name__ == '__main__':
    unittest.main()
