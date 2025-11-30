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
            data, index=dates, columns=[f'series_{i}' for i in range(n_series)]
        )

        forecast_length = 14

        # Initialize model with minimal parameters for speed
        model = MambaSSM(
            context_length=30,  # Reduced for speed
            epochs=2,  # Minimal epochs for speed
            batch_size=16,  # Small batch size
            d_model=16,  # Small model size
            n_layers=1,  # Single layer
            d_state=4,  # Small state size
            verbose=1,
            random_seed=42,
            changepoint_method="basic",  # Test changepoint features
            changepoint_params={
                "changepoint_spacing": 60,
                "changepoint_distance_end": 30,
            },
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
            f"Forecast shape should be {expected_shape}",
        )

        # Verify forecast contains no null values
        self.assertFalse(
            prediction.forecast.isnull().any().any(),
            "Forecast should not contain null values",
        )

        # Verify forecast is numeric
        self.assertTrue(
            np.isfinite(prediction.forecast.values).all(),
            "Forecast should contain only finite values",
        )

        # Verify prediction intervals exist and are properly ordered
        self.assertIsNotNone(prediction.upper_forecast, "Upper forecast should exist")
        self.assertIsNotNone(prediction.lower_forecast, "Lower forecast should exist")

        # Upper should be >= forecast >= lower (approximately, allowing for numerical precision)
        upper_ge_forecast = (
            (prediction.upper_forecast >= prediction.forecast - 1e-6).all().all()
        )
        forecast_ge_lower = (
            (prediction.forecast >= prediction.lower_forecast - 1e-6).all().all()
        )

        self.assertTrue(upper_ge_forecast, "Upper forecast should be >= point forecast")
        self.assertTrue(forecast_ge_lower, "Point forecast should be >= lower forecast")

        # Verify column names match
        self.assertEqual(
            list(prediction.forecast.columns),
            list(df_train.columns),
            "Forecast columns should match training data columns",
        )

        # Verify forecast index is proper datetime
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(prediction.forecast.index),
            "Forecast index should be datetime",
        )

        # Test model parameters retrieval
        params = model.get_params()
        self.assertIsInstance(params, dict, "get_params should return a dictionary")
        self.assertIn('epochs', params, "Parameters should include epochs")
        self.assertEqual(
            params['epochs'], 2, "Epochs parameter should match initialization"
        )

        # Test changepoint parameters
        self.assertIn(
            'changepoint_method', params, "Parameters should include changepoint_method"
        )
        self.assertIn(
            'changepoint_params', params, "Parameters should include changepoint_params"
        )
        self.assertEqual(
            params['changepoint_method'],
            "basic",
            "Changepoint method should match initialization",
        )
        self.assertEqual(
            params['changepoint_params']['changepoint_spacing'],
            60,
            "Changepoint spacing should match initialization",
        )
        self.assertEqual(
            params['changepoint_params']['changepoint_distance_end'],
            30,
            "Changepoint distance end should match initialization",
        )

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
            np.random.randn(n_timesteps, 1), index=dates, columns=['regressor_1']
        )

        forecast_length = 7
        forecast_dates = pd.date_range(
            start=dates[-1] + pd.Timedelta(days=1), periods=forecast_length, freq='D'
        )
        regressor_forecast = pd.DataFrame(
            np.random.randn(forecast_length, 1),
            index=forecast_dates,
            columns=['regressor_1'],
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
            random_seed=42,
            prediction_batch_size=20,  # Small enough for 100 timestep dataset
        )

        # Fit with regressor
        model.fit(df_train, future_regressor=regressor_train)

        # Predict with regressor
        prediction = model.predict(
            forecast_length=forecast_length, future_regressor=regressor_forecast
        )

        # Basic validation
        self.assertIsNotNone(
            prediction.forecast, "Forecast with regressors should not be None"
        )
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
        data2 = (
            np.random.randn(n_timesteps // 2, n_series).astype(np.float32) + 2
        )  # Level shift
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
            prediction_batch_size=30,  # Small enough for 120 timestep dataset
            changepoint_method="basic",  # Should detect the change around day 60
            changepoint_params={
                "changepoint_spacing": 25,
                "changepoint_distance_end": 10,
            },
        )

        # Fit and predict
        model.fit(df_train)
        prediction = model.predict(forecast_length=forecast_length)

        # Basic validation
        self.assertIsNotNone(
            prediction.forecast, "Forecast with changepoints should not be None"
        )
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
        data2 = (
            np.random.randn(20, n_series).astype(np.float32) + 15
        )  # Clear level shift
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
            prediction_batch_size=10,  # Small enough for 40 timestep dataset
            changepoint_method="l1_fused_lasso",
            changepoint_params={"lambda_reg": 2.0},
        )

        model_l1fl.fit(df_train)
        prediction_l1fl = model_l1fl.predict(forecast_length=forecast_length)

        self.assertIsNotNone(
            prediction_l1fl.forecast, "L1 fused lasso forecast should not be None"
        )
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
            prediction_batch_size=10,  # Small enough for 40 timestep dataset
            changepoint_method="l1_total_variation",
            changepoint_params={"lambda_reg": 1.0},
        )

        model_l1tv.fit(df_train)
        prediction_l1tv = model_l1tv.predict(forecast_length=forecast_length)

        self.assertIsNotNone(
            prediction_l1tv.forecast, "L1 total variation forecast should not be None"
        )
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
        series1 = np.sin(np.arange(n_timesteps) * 0.1) + np.random.normal(
            0, 0.1, n_timesteps
        )
        series2 = np.cos(np.arange(n_timesteps) * 0.05) + np.random.normal(
            0, 0.1, n_timesteps
        )
        series3 = np.arange(n_timesteps) * 0.01 + np.random.normal(0, 0.1, n_timesteps)

        df = pd.DataFrame(
            {'series1': series1, 'series2': series2, 'series3': series3}, index=dates
        )

        forecast_length = 14

        # Test pMLP with minimal parameters for speed
        mlp_model = pMLP(
            context_length=20,
            hidden_dims=[64, 32],  # Smaller for faster testing
            epochs=2,  # Quick test
            batch_size=16,
            verbose=0,  # Reduce verbosity for testing
            random_seed=42,
            prediction_batch_size=30,  # Small enough for 100 timestep dataset
        )

        # Test fitting
        print("  Testing pMLP fit...")
        mlp_model.fit(df)

        # Verify model was trained
        self.assertIsNotNone(
            mlp_model.model, "pMLP model should be initialized after fit"
        )
        self.assertIsNotNone(mlp_model.scaler_means, "pMLP scaler means should be set")
        self.assertIsNotNone(mlp_model.scaler_stds, "pMLP scaler stds should be set")
        self.assertIsNotNone(
            mlp_model.fit_runtime, "pMLP fit_runtime should be recorded"
        )

        # Test prediction
        print("  Testing pMLP predict...")
        prediction = mlp_model.predict(forecast_length=forecast_length)

        # Verify prediction object
        self.assertIsNotNone(prediction, "pMLP prediction should not be None")
        self.assertIsNotNone(prediction.forecast, "pMLP forecast should not be None")
        self.assertIsNotNone(
            prediction.predict_runtime, "pMLP predict_runtime should be recorded"
        )

        # Verify forecast shape and properties
        expected_shape = (forecast_length, n_series)
        self.assertEqual(
            prediction.forecast.shape,
            expected_shape,
            f"pMLP forecast shape should be {expected_shape}",
        )

        # Verify forecast contains no null values
        self.assertFalse(
            prediction.forecast.isnull().any().any(),
            "pMLP forecast should not contain null values",
        )

        # Verify forecast is numeric
        self.assertTrue(
            np.isfinite(prediction.forecast.values).all(),
            "pMLP forecast should contain only finite values",
        )

        # Verify prediction intervals exist and are properly ordered
        self.assertIsNotNone(
            prediction.upper_forecast, "pMLP upper forecast should exist"
        )
        self.assertIsNotNone(
            prediction.lower_forecast, "pMLP lower forecast should exist"
        )

        # Upper should be >= forecast >= lower (approximately)
        upper_ge_forecast = (
            (prediction.upper_forecast >= prediction.forecast - 1e-6).all().all()
        )
        forecast_ge_lower = (
            (prediction.forecast >= prediction.lower_forecast - 1e-6).all().all()
        )

        self.assertTrue(
            upper_ge_forecast, "pMLP upper forecast should be >= point forecast"
        )
        self.assertTrue(
            forecast_ge_lower, "pMLP point forecast should be >= lower forecast"
        )

        # Verify column names match
        self.assertEqual(
            list(prediction.forecast.columns),
            list(df.columns),
            "pMLP forecast columns should match training data columns",
        )

        # Verify forecast index is proper datetime
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(prediction.forecast.index),
            "pMLP forecast index should be datetime",
        )

        print("pMLP basic functionality test passed!")

    def test_torch_mlp_parameters(self):
        """Test pMLP parameter generation and retrieval."""
        print("Testing pMLP parameter generation")

        # Test parameter generation
        new_params = pMLP.get_new_params()

        # Verify parameters is a dictionary
        self.assertIsInstance(
            new_params, dict, "get_new_params should return a dictionary"
        )

        # Verify key parameters are present
        expected_params = [
            'hidden_dims',
            'epochs',
            'batch_size',
            'lr',
            'prediction_batch_size',
        ]
        for param in expected_params:
            self.assertIn(
                param, new_params, f"Parameter '{param}' should be in new_params"
            )

        # Verify parameter types and ranges
        self.assertIsInstance(
            new_params['hidden_dims'], list, "hidden_dims should be list"
        )
        self.assertGreater(
            len(new_params['hidden_dims']), 0, "hidden_dims should not be empty"
        )
        self.assertTrue(
            all(isinstance(x, int) for x in new_params['hidden_dims']),
            "hidden_dims should contain integers",
        )

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
        for key in ['hidden_dims', 'epochs', 'batch_size']:
            self.assertEqual(
                retrieved_params[key],
                test_params[key],
                f"Retrieved parameter '{key}' should match set parameter",
            )

        print("pMLP parameter generation test passed!")

    def test_torch_mlp_vs_mamba_comparison(self):
        """Compare pMLP vs MambaSSM performance."""
        print("Testing pMLP vs MambaSSM comparison")

        # Create test data
        n_timesteps = 120  # Larger dataset to avoid batch norm issues
        dates = pd.date_range('2023-01-01', periods=n_timesteps, freq='D')
        np.random.seed(2023)
        series = np.sin(np.arange(n_timesteps) * 0.1) + np.random.normal(
            0, 0.1, n_timesteps
        )
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
                random_seed=42,
                prediction_batch_size=30,  # Small enough for 120 timestep dataset
            ),
            'MambaSSM': MambaSSM(
                context_length=15,
                epochs=1,
                batch_size=16,  # Larger batch size
                d_model=16,
                n_layers=1,
                d_state=4,
                verbose=0,
                random_seed=42,
                prediction_batch_size=30,  # Small enough for 120 timestep dataset
            ),
        }

        results = {}

        for name, model in models.items():
            print(f"  Testing {name}...")
            try:
                # Fit the model
                model.fit(df)
                self.assertIsNotNone(
                    model.fit_runtime, f"{name} should record fit_runtime"
                )

                # Make prediction
                prediction = model.predict(forecast_length=forecast_length)
                self.assertIsNotNone(
                    prediction.predict_runtime, f"{name} should record predict_runtime"
                )

                # Basic validation
                self.assertEqual(
                    prediction.forecast.shape,
                    (forecast_length, 1),
                    f"{name} forecast shape should be correct",
                )
                self.assertFalse(
                    prediction.forecast.isnull().any().any(),
                    f"{name} forecast should not contain nulls",
                )
                self.assertTrue(
                    np.isfinite(prediction.forecast.values).all(),
                    f"{name} forecast should be finite",
                )

                results[name] = {
                    'fit_time': model.fit_runtime.total_seconds(),
                    'predict_time': prediction.predict_runtime.total_seconds(),
                    'forecast_shape': prediction.forecast.shape,
                    'success': True,
                }

            except Exception as e:
                self.fail(f"{name} model failed: {e}")
                results[name] = {'success': False, 'error': str(e)}

        # Verify both models succeeded
        for name in models.keys():
            self.assertTrue(results[name]['success'], f"{name} should succeed")
            self.assertGreater(
                results[name]['fit_time'], 0, f"{name} fit_time should be positive"
            )
            self.assertGreater(
                results[name]['predict_time'],
                0,
                f"{name} predict_time should be positive",
            )

        # Compare performance (informational, not a hard test)
        mlp_total = results['pMLP']['fit_time'] + results['pMLP']['predict_time']
        mamba_total = (
            results['MambaSSM']['fit_time'] + results['MambaSSM']['predict_time']
        )

        print(f"    pMLP total time: {mlp_total:.3f}s")
        print(f"    MambaSSM total time: {mamba_total:.3f}s")

        # Both models should produce reasonable results
        self.assertLess(mlp_total, 60, "pMLP should complete within reasonable time")
        self.assertLess(
            mamba_total, 60, "MambaSSM should complete within reasonable time"
        )

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
            prediction_batch_size=10,  # Small enough for 40 timestep dataset
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
            prediction_batch_size=10,  # Small enough for 40 timestep dataset
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
            prediction_batch_size=10,  # Small enough for 40 timestep dataset
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
            prediction_batch_size=10,  # Small enough for 40 timestep dataset
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
        series_a = np.concatenate(
            [
                50 + np.random.randn(75) * 2,  # Mean 50
                80 + np.random.randn(75) * 2,  # Mean 80 (shift at day 75)
            ]
        )

        # Series B: Has a changepoint at day 100 (level shift down)
        series_b = np.concatenate(
            [
                70 + np.random.randn(100) * 2,  # Mean 70
                40 + np.random.randn(50) * 2,  # Mean 40 (shift at day 100)
            ]
        )

        # Series C: No clear changepoint, gradual trend
        series_c = np.linspace(30, 60, n_timesteps) + np.random.randn(n_timesteps) * 2

        df = pd.DataFrame(
            {'SeriesA': series_a, 'SeriesB': series_b, 'SeriesC': series_c}, index=dates
        )

        forecast_length = 10
        train_df = df.iloc[:-forecast_length]

        # Test 1: Verify ChangepointDetector creates per-series features with individual aggregation
        print("  Testing ChangepointDetector with aggregate_method='individual'")
        detector = ChangepointDetector(
            method='cusum',
            method_params={'threshold': 3.0, 'min_distance': 10},
            aggregate_method='individual',
        )
        detector.detect(train_df)

        # Verify changepoints were detected per series
        self.assertIsInstance(
            detector.changepoints_,
            dict,
            "Changepoints should be a dict with individual aggregation",
        )
        self.assertEqual(
            set(detector.changepoints_.keys()),
            set(df.columns),
            "Changepoints dict should have keys for all series",
        )

        # Create features and verify per-series pattern
        features = detector.create_features(forecast_length=0)

        # Check for per-series feature naming pattern
        has_series_a_features = any('SeriesA_' in str(col) for col in features.columns)
        has_series_b_features = any('SeriesB_' in str(col) for col in features.columns)
        has_series_c_features = any('SeriesC_' in str(col) for col in features.columns)

        self.assertTrue(
            has_series_a_features or has_series_b_features or has_series_c_features,
            "Per-series changepoint features should be created with series names in column names",
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
            changepoint_params={'threshold': 3.0, 'min_distance': 10},
        )

        # Fit with aggregate_method='individual'
        model_mamba.fit(train_df, aggregate_method='individual')

        # Verify per-series feature mapping was created
        self.assertTrue(
            hasattr(model_mamba, 'has_per_series_features'),
            "Model should have has_per_series_features attribute",
        )
        self.assertTrue(
            hasattr(model_mamba, 'series_feat_mapping'),
            "Model should have series_feat_mapping attribute",
        )

        if model_mamba.has_per_series_features:
            self.assertIsNotNone(
                model_mamba.series_feat_mapping,
                "Series feature mapping should be created for per-series features",
            )
            self.assertIsInstance(
                model_mamba.series_feat_mapping,
                dict,
                "Series feature mapping should be a dict",
            )
            self.assertEqual(
                len(model_mamba.series_feat_mapping),
                len(df.columns),
                "Series feature mapping should have an entry for each series",
            )

            # Verify each series has feature indices
            for series_idx in range(len(df.columns)):
                self.assertIn(
                    series_idx,
                    model_mamba.series_feat_mapping,
                    f"Series {series_idx} should be in feature mapping",
                )
                feat_indices = model_mamba.series_feat_mapping[series_idx]
                self.assertIsInstance(
                    feat_indices,
                    list,
                    f"Feature indices for series {series_idx} should be a list",
                )
                self.assertGreater(
                    len(feat_indices),
                    0,
                    f"Series {series_idx} should have at least one feature",
                )

        # Make predictions
        prediction_mamba = model_mamba.predict(forecast_length=forecast_length)

        # Verify predictions
        self.assertEqual(
            prediction_mamba.forecast.shape,
            (forecast_length, len(df.columns)),
            "MambaSSM forecast shape should match expected dimensions",
        )
        self.assertFalse(
            prediction_mamba.forecast.isnull().any().any(),
            "MambaSSM forecast should not contain null values",
        )
        self.assertTrue(
            np.isfinite(prediction_mamba.forecast.values).all(),
            "MambaSSM forecast should contain only finite values",
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
            prediction_batch_size=30,  # Small enough for 140 timestep dataset
            changepoint_method='cusum',
            changepoint_params={'threshold': 3.0, 'min_distance': 10},
        )

        # Fit with aggregate_method='individual'
        model_pmlp.fit(train_df, aggregate_method='individual')

        # Verify per-series feature mapping was created
        self.assertTrue(
            hasattr(model_pmlp, 'has_per_series_features'),
            "pMLP should have has_per_series_features attribute",
        )
        self.assertTrue(
            hasattr(model_pmlp, 'series_feat_mapping'),
            "pMLP should have series_feat_mapping attribute",
        )

        if model_pmlp.has_per_series_features:
            self.assertIsNotNone(
                model_pmlp.series_feat_mapping,
                "pMLP series feature mapping should be created for per-series features",
            )
            self.assertIsInstance(
                model_pmlp.series_feat_mapping,
                dict,
                "pMLP series feature mapping should be a dict",
            )
            self.assertEqual(
                len(model_pmlp.series_feat_mapping),
                len(df.columns),
                "pMLP series feature mapping should have an entry for each series",
            )

            # Verify each series has feature indices
            for series_idx in range(len(df.columns)):
                self.assertIn(
                    series_idx,
                    model_pmlp.series_feat_mapping,
                    f"pMLP series {series_idx} should be in feature mapping",
                )
                feat_indices = model_pmlp.series_feat_mapping[series_idx]
                self.assertIsInstance(
                    feat_indices,
                    list,
                    f"pMLP feature indices for series {series_idx} should be a list",
                )
                self.assertGreater(
                    len(feat_indices),
                    0,
                    f"pMLP series {series_idx} should have at least one feature",
                )

        # Make predictions
        prediction_pmlp = model_pmlp.predict(forecast_length=forecast_length)

        # Verify predictions
        self.assertEqual(
            prediction_pmlp.forecast.shape,
            (forecast_length, len(df.columns)),
            "pMLP forecast shape should match expected dimensions",
        )
        self.assertFalse(
            prediction_pmlp.forecast.isnull().any().any(),
            "pMLP forecast should not contain null values",
        )
        self.assertTrue(
            np.isfinite(prediction_pmlp.forecast.values).all(),
            "pMLP forecast should contain only finite values",
        )

        # Test 4: Verify different behavior between individual and shared changepoints
        print("  Testing comparison between individual and shared changepoints")

        # Train a model with shared changepoints (explicitly set aggregate_method='mean')
        # Disable naive features so we can isolate changepoint feature behavior
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
            changepoint_params={'threshold': 3.0, 'min_distance': 10},
            use_naive_feature=False,  # Disable to test only changepoint feature behavior
        )

        # Fit with aggregate_method='mean' for shared changepoints
        model_shared.fit(train_df, aggregate_method='mean')

        # Should not have per-series features with shared changepoints and no naive features
        self.assertFalse(
            getattr(model_shared, 'has_per_series_features', False),
            "Model with shared changepoints and no naive features should not have per-series features",
        )

        # Make predictions
        prediction_shared = model_shared.predict(forecast_length=forecast_length)

        # Both should produce valid predictions, but they should be different
        # (this is a sanity check, not a strict requirement)
        self.assertEqual(
            prediction_shared.forecast.shape,
            prediction_mamba.forecast.shape,
            "Both models should produce same shape forecasts",
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
        df = pd.DataFrame(
            {
                'A': np.random.randn(n_timesteps) + np.linspace(0, 2, n_timesteps),
                'B': np.random.randn(n_timesteps) + np.linspace(2, 0, n_timesteps),
            },
            index=dates,
        )

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
            prediction_batch_size=30,  # Small enough for 113 timestep dataset
            changepoint_method='basic',
            changepoint_params={
                'changepoint_spacing': 30,
                'changepoint_distance_end': 15,
            },
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

    def test_training_data_shape_and_features(self):
        """Test that training data X_data has correct shape and proper feature values.

        This test validates:
        1. X_data shape is (n_samples, prediction_batch_size, n_features)
        2. Seasonality features are populated correctly (date parts)
        3. Changepoint features are populated correctly (not all zeros)
        4. Naive window feature contains proper historical data when use_naive_window=True
        5. All features have reasonable non-zero variance
        """
        print("Testing training data shape and feature content")

        # Create synthetic data with clear patterns
        n_timesteps = 200
        n_series = 3
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')

        np.random.seed(42)

        # Create data with clear weekly seasonality and trend
        time_idx = np.arange(n_timesteps)
        weekly_season = np.sin(time_idx * 2 * np.pi / 7)  # Weekly pattern
        trend = time_idx * 0.05  # Upward trend

        # Series with different patterns but all have seasonality
        series_data = np.zeros((n_timesteps, n_series))
        for i in range(n_series):
            series_data[:, i] = (
                trend + weekly_season * (i + 1) + np.random.randn(n_timesteps) * 0.5
            )
            # Add a level shift at timestep 100 for changepoint detection
            series_data[100:, i] += 5 * (i + 1)

        df = pd.DataFrame(
            series_data, index=dates, columns=['series_a', 'series_b', 'series_c']
        )

        # Test with pMLP (easier to inspect as it stores X_data)
        prediction_batch_size = 30
        model = pMLP(
            context_length=50,
            hidden_dims=[64],
            epochs=1,
            batch_size=16,
            verbose=0,
            random_seed=42,
            prediction_batch_size=prediction_batch_size,
            datepart_method='expanded',  # Rich date features
            holiday_countries_used=True,
            use_naive_feature=True,  # Enable naive window feature
            changepoint_method='ewma',  # Method that should detect our level shifts
            changepoint_params={'window': 10, 'alpha': 0.3},
        )

        # Fit the model with individual aggregate_method
        model.fit(df, aggregate_method='individual')

        # Test 1: Verify X_data shape
        print(f"  X_data shape: {model.X_data.shape}")
        self.assertEqual(len(model.X_data.shape), 3, "X_data should be 3-dimensional")

        n_samples, batch_size, n_features = model.X_data.shape
        self.assertEqual(
            batch_size,
            prediction_batch_size,
            f"Second dimension should be prediction_batch_size ({prediction_batch_size})",
        )

        # Test 2: Verify feature count makes sense
        # Expected features:
        # - Date features (from 'expanded' method - typically 10-20 features)
        # - Changepoint features (per-series with 'individual' aggregation)
        # - Naive features (1 per series = 3)
        print(f"  Total features: {n_features}")
        print(f"  Feature columns: {len(model.feature_columns)}")

        # When per-series features are used, X_data has max feature count (padded)
        # Otherwise, feature count should match exactly
        if model.has_per_series_features:
            # X_data should have max feature count across all series
            max_feat_count = max(
                len(feat_indices) for feat_indices in model.series_feat_mapping.values()
            )
            self.assertEqual(
                n_features,
                max_feat_count,
                "X_data feature count should match max per-series feature count",
            )
            print(f"  Per-series mode: X_data has max feature count {max_feat_count}")
        else:
            self.assertEqual(
                n_features,
                len(model.feature_columns),
                "X_data feature count should match feature_columns length",
            )

        # Test 3: Check seasonality features (date parts) are not all zeros
        # Find date feature indices - need to map to X_data indices correctly
        date_feature_indices = []

        # In per-series mode, we need to find features within any series mapping
        # In shared mode, indices map directly
        if model.has_per_series_features:
            # Collect all unique feature indices used by any series
            all_used_indices = set()
            for feat_indices in model.series_feat_mapping.values():
                all_used_indices.update(feat_indices)

            # Find date features within the used indices
            for feat_idx in sorted(all_used_indices):
                if feat_idx < len(model.feature_columns):
                    col_str = str(model.feature_columns[feat_idx])
                    if any(
                        x in col_str.lower()
                        for x in [
                            'month',
                            'day',
                            'week',
                            'year',
                            'weekday',
                            'sin',
                            'cos',
                        ]
                    ):
                        # Map to X_data index (find position in one of the series mappings)
                        for series_idx, indices in model.series_feat_mapping.items():
                            if feat_idx in indices:
                                xdata_idx = indices.index(feat_idx)
                                if xdata_idx < n_features:
                                    date_feature_indices.append(xdata_idx)
                                break
        else:
            for i, col in enumerate(model.feature_columns):
                col_str = str(col)
                if any(
                    x in col_str.lower()
                    for x in ['month', 'day', 'week', 'year', 'weekday', 'sin', 'cos']
                ):
                    date_feature_indices.append(i)

        print(f"  Date feature count: {len(date_feature_indices)}")
        self.assertGreater(
            len(date_feature_indices), 0, "Should have date/seasonality features"
        )

        # Check that date features have variance (not all zeros)
        for feat_idx in list(set(date_feature_indices))[
            :5
        ]:  # Check first 5 unique date features
            if feat_idx < n_features:
                feat_values = model.X_data[:, :, feat_idx]
                feat_var = np.var(feat_values)
                self.assertGreater(
                    feat_var,
                    0,
                    f"Date feature at index {feat_idx} should have non-zero variance",
                )

        # Test 4: Check changepoint features are not all zeros
        changepoint_feature_indices = []
        changepoint_xdata_mapping = {}  # Maps feature_columns index to X_data index

        if model.has_per_series_features:
            # Build mapping for changepoint features in per-series mode
            for series_idx, feat_indices in model.series_feat_mapping.items():
                for xdata_idx, feat_col_idx in enumerate(feat_indices):
                    if feat_col_idx < len(model.feature_columns):
                        col_str = str(model.feature_columns[feat_col_idx])
                        if (
                            'changepoint' in col_str.lower()
                            or 'ewma' in col_str.lower()
                        ):
                            if feat_col_idx not in changepoint_xdata_mapping:
                                changepoint_xdata_mapping[feat_col_idx] = xdata_idx
                                if xdata_idx < n_features:
                                    changepoint_feature_indices.append(xdata_idx)
        else:
            for i, col in enumerate(model.feature_columns):
                col_str = str(col)
                if 'changepoint' in col_str.lower() or 'ewma' in col_str.lower():
                    changepoint_feature_indices.append(i)
                    changepoint_xdata_mapping[i] = i

        print(f"  Changepoint feature count: {len(changepoint_feature_indices)}")

        if len(changepoint_feature_indices) > 0:
            # Check that at least some changepoint features have non-zero values
            has_nonzero_changepoint = False
            changepoint_stats = []
            for xdata_idx in changepoint_feature_indices:
                if xdata_idx < n_features:
                    feat_values = model.X_data[:, :, xdata_idx]
                    nonzero_count = np.count_nonzero(feat_values)
                    feat_var = np.var(feat_values)

                    # Find the feature column name
                    feat_name = f"X_data_index_{xdata_idx}"
                    for (
                        feat_col_idx,
                        mapped_xdata_idx,
                    ) in changepoint_xdata_mapping.items():
                        if mapped_xdata_idx == xdata_idx and feat_col_idx < len(
                            model.feature_columns
                        ):
                            feat_name = str(model.feature_columns[feat_col_idx])
                            break

                    changepoint_stats.append(
                        {
                            'name': feat_name,
                            'nonzero_count': nonzero_count,
                            'variance': feat_var,
                        }
                    )
                    if nonzero_count > 0:
                        has_nonzero_changepoint = True
                        print(
                            f"    Feature {feat_name}: {nonzero_count} non-zero values, variance={feat_var:.6f}"
                        )

            # This is a critical test - changepoint features should not all be zero
            self.assertTrue(
                has_nonzero_changepoint,
                f"At least some changepoint features should have non-zero values. Stats: {changepoint_stats}",
            )
        else:
            self.fail(
                "No changepoint features found in feature columns despite using changepoint_method='ewma'"
            )  # Test 5: Check naive window features are populated correctly
        naive_feature_indices = []
        naive_xdata_mapping = {}  # Maps feature_columns index to X_data index

        if model.has_per_series_features:
            # Build mapping for naive features in per-series mode
            for series_idx, feat_indices in model.series_feat_mapping.items():
                for xdata_idx, feat_col_idx in enumerate(feat_indices):
                    if feat_col_idx < len(model.feature_columns):
                        col_str = str(model.feature_columns[feat_col_idx])
                        if 'naive_last_' in col_str:
                            if feat_col_idx not in naive_xdata_mapping:
                                naive_xdata_mapping[feat_col_idx] = xdata_idx
                                if xdata_idx < n_features:
                                    naive_feature_indices.append(xdata_idx)
        else:
            for i, col in enumerate(model.feature_columns):
                col_str = str(col)
                if 'naive_last_' in col_str:
                    naive_feature_indices.append(i)
                    naive_xdata_mapping[i] = i

        print(f"  Naive feature count: {len(naive_feature_indices)}")
        self.assertEqual(
            len(naive_feature_indices),
            n_series,
            f"Should have {n_series} naive features (one per series)",
        )

        # Test 6: Verify naive window feature has proper historical data
        # The naive feature should contain window values, not just a single repeated value
        for xdata_idx in naive_feature_indices:
            if xdata_idx < n_features:
                feat_values = model.X_data[:, :, xdata_idx]

                # Find the feature column name
                feat_name = f"X_data_index_{xdata_idx}"
                for feat_col_idx, mapped_xdata_idx in naive_xdata_mapping.items():
                    if mapped_xdata_idx == xdata_idx and feat_col_idx < len(
                        model.feature_columns
                    ):
                        feat_name = str(model.feature_columns[feat_col_idx])
                        break

                # Check variance within individual samples
                # For window-based features, values should vary across the prediction_batch_size dimension
                sample_variances = []
                for sample_idx in range(min(100, n_samples)):  # Check first 100 samples
                    sample_values = feat_values[sample_idx, :]
                    sample_var = np.var(sample_values)
                    sample_variances.append(sample_var)

                avg_sample_variance = np.mean(sample_variances)
                print(
                    f"    Naive feature {feat_name}: avg variance within samples = {avg_sample_variance:.6f}"
                )

                # With window-based naive features, there should be variance within samples
                # (different historical values for different timesteps in the window)
                self.assertGreater(
                    avg_sample_variance,
                    0,
                    f"Naive window feature {feat_name} should have variance within samples",
                )

        # Test 7: Verify no unexpected features are entirely zero
        # Some features can legitimately be all zeros (e.g., months/holidays not in date range)
        all_zero_features = []
        unexpected_zero_features = []
        for feat_idx in range(n_features):
            if np.all(model.X_data[:, :, feat_idx] == 0):
                # Find feature name (need to reverse map from X_data index)
                feat_name = f"unknown_feature_{feat_idx}"
                if model.has_per_series_features:
                    # Find which feature_column this X_data index corresponds to
                    for series_idx, feat_indices in model.series_feat_mapping.items():
                        if feat_idx < len(feat_indices):
                            feat_col_idx = feat_indices[feat_idx]
                            if feat_col_idx < len(model.feature_columns):
                                feat_name = str(model.feature_columns[feat_col_idx])
                                break
                else:
                    if feat_idx < len(model.feature_columns):
                        feat_name = str(model.feature_columns[feat_idx])

                all_zero_features.append(feat_name)

                # Check if this is an expected zero (month/holiday not in range, or padding)
                is_expected_zero = (
                    'month_' in feat_name.lower()
                    or 'holiday' in feat_name.lower()
                    or 'day' in feat_name.lower()
                    and (
                        'christmas' in feat_name.lower()
                        or 'thanksgiving' in feat_name.lower()
                        or 'independence' in feat_name.lower()
                        or 'labor' in feat_name.lower()
                        or 'veterans' in feat_name.lower()
                        or 'columbus' in feat_name.lower()
                        or 'juneteenth' in feat_name.lower()
                        or 'memorial' in feat_name.lower()
                        or 'martin' in feat_name.lower()
                        or 'president' in feat_name.lower()
                        or 'new year' in feat_name.lower()
                    )
                )

                if not is_expected_zero:
                    unexpected_zero_features.append(feat_name)

        if all_zero_features:
            print(f"  All-zero features (may be expected): {all_zero_features}")

        # Only fail if we have unexpected zeros (not months/holidays)
        if unexpected_zero_features:
            self.fail(
                f"These features are unexpectedly all zeros: {unexpected_zero_features}"
            )

        # Test 8: Verify per-series feature mapping if present
        if model.has_per_series_features:
            print(f"  Has per-series features: True")
            self.assertIsNotNone(
                model.series_feat_mapping, "Should have series_feat_mapping"
            )

            # Verify each series has features assigned
            for series_idx in range(n_series):
                self.assertIn(
                    series_idx,
                    model.series_feat_mapping,
                    f"Series {series_idx} should be in mapping",
                )
                feat_indices = model.series_feat_mapping[series_idx]
                self.assertGreater(
                    len(feat_indices),
                    0,
                    f"Series {series_idx} should have features assigned",
                )
                print(f"    Series {series_idx} has {len(feat_indices)} features")
        else:
            print(f"  Has per-series features: False")

        # Test 9: Basic sanity checks on the data
        # Check for NaN or Inf values
        self.assertFalse(
            np.isnan(model.X_data).any(), "X_data should not contain NaN values"
        )
        self.assertFalse(
            np.isinf(model.X_data).any(), "X_data should not contain Inf values"
        )

        # Check that data is properly scaled (should have reasonable range after StandardScaler)
        data_mean = np.mean(model.X_data)
        data_std = np.std(model.X_data)
        print(f"  X_data mean: {data_mean:.6f}, std: {data_std:.6f}")

        # After scaling, mean should be close to 0 and std close to 1 (for most features)
        # Allow some tolerance since some features might be constant or near-constant
        self.assertLess(
            abs(data_mean), 2.0, "Mean should be reasonably close to 0 after scaling"
        )

        print("Training data shape and feature content test passed!")


if __name__ == '__main__':
    unittest.main()
