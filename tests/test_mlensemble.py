# -*- coding: utf-8 -*-
"""MLEnsemble testing."""
import random
import unittest
import numpy as np
import pandas as pd
from autots.models.mlensemble import MLEnsemble
from autots.models.forecast_correction import ForecastCorrector, forecast_context


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
            data, index=dates, columns=[f'series_{i}' for i in range(n_series)]
        )

        forecast_length = 14

        for correction_method in ["pbc", "debias", "persistence", "regressor"]:
            with self.subTest(correction_method=correction_method):
                self._check_basic(df_train, forecast_length, n_series, correction_method)
        print("MLEnsemble basic tests passed!")

    def _check_basic(self, df_train, forecast_length, n_series, correction_method):
        # Initialize model with minimal parameters for speed
        model = MLEnsemble(
            forecast_length=forecast_length,
            num_validations=1,  # Minimal validations for speed
            validation_method="backwards",
            datepart_method="simple",  # Test datepart feature creation
            correction_method=correction_method,
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

        if correction_method == "regressor":
            self.assertIsNotNone(model.regr, "Regressor should be initialized after fit")
            self.assertEqual(model.X.ndim, 2, "X should be 2-dimensional")
            self.assertEqual(model.y.ndim, 1, "y should be 1-dimensional")
        else:
            self.assertIsNotNone(model.corrector)

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
            f"Forecast shape should be {expected_shape}, got {prediction.forecast.shape}",
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

        # Verify column names match
        self.assertEqual(
            list(prediction.forecast.columns),
            list(df_train.columns),
            "Forecast columns should match training data columns",
        )

        # Verify index is correct
        self.assertEqual(
            len(prediction.forecast.index),
            forecast_length,
            "Forecast index length should match forecast_length",
        )
        self.assertTrue(
            (prediction.upper_forecast.values >= prediction.lower_forecast.values).all()
        )

    def test_model_different_series_counts(self):
        """Test MLEnsemble with different numbers of series to catch hardcoded dimension bugs."""
        print("Testing MLEnsemble with different series counts...")

        forecast_length = 10

        for n_series in [
            3,
            7,
            21,
            27,
        ]:  # Test various series counts including the problematic 27
            print(f"  Testing with {n_series} series...")

            n_timesteps = 100
            dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')

            np.random.seed(42)
            data = np.random.randn(n_timesteps, n_series) * 0.5 + 10

            df_train = pd.DataFrame(
                data, index=dates, columns=[f'series_{i}' for i in range(n_series)]
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
                    f"Failed for {n_series} series: shape mismatch",
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
            data, index=dates, columns=[f'series_{i}' for i in range(n_series)]
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
                correction_method="regressor",
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
                f"Failed with datepart_method='{method}'",
            )

        print("Datepart methods tests passed!")

    def test_model_nan_in_holdout(self):
        """NaN actuals in the holdout (fillna=None upstream) must not reach the regressor as labels."""
        n_timesteps = 120
        n_series = 4
        forecast_length = 10
        dates = pd.date_range(start='2023-01-01', periods=n_timesteps, freq='D')

        np.random.seed(42)
        data = np.random.randn(n_timesteps, n_series) * 0.5 + 10
        df_train = pd.DataFrame(
            data, index=dates, columns=[f'series_{i}' for i in range(n_series)]
        )
        df_train.iloc[-forecast_length:, :2] = np.nan

        for correction_method in ["regressor", "pbc"]:
            with self.subTest(correction_method=correction_method):
                self._check_nan_holdout(df_train, forecast_length, correction_method)

    def _check_nan_holdout(self, df_train, forecast_length, correction_method):
        model = MLEnsemble(
            forecast_length=forecast_length,
            num_validations=0,
            validation_method="backwards",
            datepart_method="simple",
            correction_method=correction_method,
            models=[
                {
                    'Model': 'LastValueNaive',
                    'ModelParameters': '{}',
                    'TransformationParameters': '{"fillna": "ffill", "transformations": {}, "transformation_params": {}}',
                },
            ],
            verbose=0,
            random_seed=42,
        )
        model.fit(df_train)
        if correction_method == "regressor":
            self.assertTrue(np.isfinite(model.y).all())
            self.assertTrue(np.isfinite(model.X).all())
        prediction = model.predict(forecast_length=forecast_length)
        self.assertTrue(np.isfinite(prediction.forecast.values).all())


def _naive_models():
    return [
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
    ]


def _hindcast_fixture(n_folds=3, n_models=2, horizon=10, n_series=4, seed=0):
    """Random-walk histories with per-fold actuals, contexts and next-origin data."""
    rng = np.random.default_rng(seed)
    total = 200 + horizon * (n_folds + 1)
    values = np.cumsum(rng.normal(size=(total, n_series)), axis=0) + 50
    index = pd.date_range("2022-01-01", periods=total, freq="D")
    df = pd.DataFrame(values, index=index)
    folds = []
    for fold in range(n_folds + 1):
        origin = 200 + horizon * fold
        history = df.iloc[:origin]
        future = df.iloc[origin : origin + horizon]
        folds.append(
            (forecast_context(history, future.index), future.to_numpy(), n_models)
        )
    return folds


class ForecastCorrectorTest(unittest.TestCase):
    def test_zero_residual_reduces_to_simple_average(self):
        """Base models that already match the actuals leave nothing to correct."""
        folds = _hindcast_fixture()
        contexts = [c for c, _, _ in folds[:-1]]
        actuals = np.stack([a for _, a, _ in folds[:-1]])
        base = np.repeat(actuals[:, None], 2, axis=1)
        live_context, live_actual, _ = folds[-1]
        rng = np.random.default_rng(1)
        live_base = live_actual[None] + rng.normal(size=(2,) + live_actual.shape)
        for method in ["pbc", "debias", "persistence"]:
            corrector = ForecastCorrector(correction_method=method).fit(
                base, actuals, contexts
            )
            np.testing.assert_allclose(
                corrector.predict(live_base, live_context),
                live_base.mean(axis=0),
                atol=1e-8,
            )

    def test_debias_removes_constant_bias(self):
        folds = _hindcast_fixture()
        contexts = [c for c, _, _ in folds[:-1]]
        actuals = np.stack([a for _, a, _ in folds[:-1]])
        # Bias is learned in scale-free units, so inject it in each origin's own scale.
        bias = 5.0
        fold_scale = np.stack([c["scale"] for c in contexts])[:, None, None, :]
        base = np.repeat(actuals[:, None], 2, axis=1) + bias * fold_scale
        live_context, live_actual, _ = folds[-1]
        live_base = (
            np.repeat(live_actual[None], 2, axis=0) + bias * live_context["scale"]
        )
        corrector = ForecastCorrector(
            correction_method="debias", bias_shape="constant", bias_shrinkage=1.0
        ).fit(base, actuals, contexts)
        remaining = corrector.predict(live_base, live_context) - live_actual
        # Shrinkage V / (V + k) = 3 / 4 leaves a quarter of the bias.
        np.testing.assert_allclose(
            remaining, np.broadcast_to(bias / 4 * live_context["scale"], remaining.shape)
        )

    def test_series_failures_are_isolated(self):
        """Missing hindcast labels fall back to the average; a fully missing series is NaN only itself."""
        folds = _hindcast_fixture()
        contexts = [c for c, _, _ in folds[:-1]]
        actuals = np.stack([a for _, a, _ in folds[:-1]])
        base = np.repeat(actuals[:, None], 2, axis=1) + 3.0
        actuals[:, :, 0] = np.nan
        live_context, live_actual, _ = folds[-1]
        live_context = {k: v.copy() for k, v in live_context.items()}
        live_context["last_value"][1] = np.nan
        live_base = np.repeat(live_actual[None], 2, axis=0) + 3.0
        corrector = ForecastCorrector(correction_method="pbc").fit(
            base, actuals, contexts
        )
        result = corrector.predict(live_base, live_context)
        np.testing.assert_allclose(result[:, 0], live_base.mean(axis=0)[:, 0])
        self.assertTrue(np.isnan(result[:, 1]).all())
        self.assertTrue(np.isfinite(result[:, 2:]).all())

    def test_scale_equivariance_end_to_end(self):
        rng = np.random.default_rng(3)
        dates = pd.date_range("2023-01-01", periods=160, freq="D")
        df = pd.DataFrame(
            np.cumsum(rng.normal(size=(160, 3)), axis=0) + 20, index=dates
        )
        scaled = df.copy()
        scaled[1] = scaled[1] * 1e6
        forecasts = []
        for frame in (df, scaled):
            model = MLEnsemble(
                forecast_length=14, num_validations=2, models=_naive_models()
            )
            forecasts.append(model.fit(frame).predict().forecast)
        np.testing.assert_allclose(forecasts[1][1], forecasts[0][1] * 1e6, rtol=1e-6)
        np.testing.assert_allclose(forecasts[1][[0, 2]], forecasts[0][[0, 2]])

    def test_base_model_intervals_respect_prediction_interval(self):
        rng = np.random.default_rng(5)
        dates = pd.date_range("2023-01-01", periods=160, freq="D")
        df = pd.DataFrame(
            np.cumsum(rng.normal(size=(160, 4)), axis=0) + 50, index=dates
        )
        widths = []
        for interval in (0.5, 0.95):
            prediction = MLEnsemble(
                forecast_length=14,
                num_validations=2,
                models=_naive_models(),
                interval_method="base_models",
                prediction_interval=interval,
            ).fit(df).predict()
            widths.append(
                (prediction.upper_forecast - prediction.lower_forecast).to_numpy().mean()
            )
        self.assertLess(widths[0], widths[1])

    def test_new_params_round_trip(self):
        random.seed(0)
        seen_models = set()
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        df = pd.DataFrame(np.random.default_rng(0).normal(size=(120, 3)) + 10, index=dates)
        for _ in range(8):
            params = MLEnsemble().get_new_params()
            seen_models.update(m['Model'] for m in params['models'])
            params['models'] = _naive_models()
            model = MLEnsemble(forecast_length=7, **params).fit(df)
            self.assertEqual(model.predict().forecast.shape, (7, 3))
            self.assertEqual(
                set(model.get_params()) - set(params), set(), "get_params keys drift"
            )
        # RandomTemplate alone cycles the list in order and would only ever give 3 models
        self.assertGreater(len(seen_models), 3)


if __name__ == '__main__':
    unittest.main()
