import os
import tempfile
import random
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from autots.models.sklearn import (  # noqa: E402
    RollingRegression,
    MultivariateRegression,
    generate_regressor_params,
    retrieve_classifier,
    retrieve_regressor,
    rolling_x_regressor,
)


class TestRetrieveRegressor(unittest.TestCase):
    def test_linear_regression_lookup(self):
        model = retrieve_regressor(
            {"model": "LinearRegression", "model_params": {}},
            multioutput=False,
        )
        self.assertIsInstance(model, LinearRegression)

    def test_knn_multioutput_wrapper(self):
        model = retrieve_regressor(
            {"model": "KNN", "model_params": {"n_neighbors": 2}},
            multioutput=True,
            n_jobs=2,
        )
        self.assertIsInstance(model, MultiOutputRegressor)
        self.assertIsInstance(model.estimator, KNeighborsRegressor)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            retrieve_regressor(
                {"model": "UnknownModel", "model_params": {}},
                multioutput=False,
            )


class TestRetrieveClassifier(unittest.TestCase):
    def test_sgd_multioutput_classifier(self):
        model = retrieve_classifier(
            {"model": "SGD", "model_params": {}},
            multioutput=True,
        )
        self.assertIsInstance(model, MultiOutputClassifier)

    def test_knn_classifier_single_output(self):
        model = retrieve_classifier(
            {"model": "KNN", "model_params": {"n_neighbors": 3}},
            multioutput=False,
        )
        self.assertIsInstance(model, KNeighborsClassifier)


class TestRollingXRegressor(unittest.TestCase):
    def test_feature_frame_contains_expected_columns(self):
        idx = pd.date_range("2020-01-01", periods=6, freq="D")
        df = pd.DataFrame({"col_a": range(6), "col_b": range(6, 12)}, index=idx)
        features = rolling_x_regressor(
            df,
            mean_rolling_periods=2,
            std_rolling_periods=None,
            max_rolling_periods=None,
            min_rolling_periods=None,
            quantile90_rolling_periods=None,
            quantile10_rolling_periods=None,
            additional_lag_periods=None,
            window=2,
            holiday=False,
        )
        self.assertEqual(list(features.index), list(df.index))
        self.assertTrue(any("lastvalue" in col for col in features.columns))
        self.assertTrue(any("window" in col for col in features.columns))
        lastvalue_col = next(
            col for col in features.columns if col.endswith("lastvalue_0")
        )
        self.assertEqual(features.loc[idx[0], lastvalue_col], df.iloc[0, 0])


class TestRollingRegression(unittest.TestCase):
    def test_predict_returns_expected_shape(self):
        idx = pd.date_range("2020-01-01", periods=25, freq="D")
        df = pd.DataFrame({"y": np.arange(25)}, index=idx)
        model = RollingRegression(
            regression_model={"model": "LinearRegression", "model_params": {}},
            mean_rolling_periods=2,
            std_rolling_periods=None,
            max_rolling_periods=None,
            min_rolling_periods=None,
            quantile90_rolling_periods=None,
            quantile10_rolling_periods=None,
            additional_lag_periods=None,
            window=2,
        )
        model.fit(df)
        forecast = model.predict(forecast_length=3, just_point_forecast=True)
        self.assertEqual(forecast.shape, (3, 1))
        self.assertEqual(forecast.index[0], df.index[-1] + pd.Timedelta(days=1))

    def test_generate_regressor_params_trees_method(self):
        random.seed(1)
        np.random.seed(1)
        params = generate_regressor_params(method="trees")
        self.assertEqual(params["model"], "DecisionTree")
        self.assertIsInstance(params["model_params"], dict)
        model = retrieve_regressor(params, multioutput=False)
        self.assertIsInstance(model, DecisionTreeRegressor)


class TestMultivariateRegression(unittest.TestCase):
    def test_basic_fit_predict(self):
        """Test basic fit and predict functionality."""
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {
                "series_1": np.arange(50) + np.random.randn(50) * 0.1,
                "series_2": np.arange(50) * 2 + np.random.randn(50) * 0.2,
            },
            index=idx,
        )
        model = MultivariateRegression(
            forecast_length=5,
            regression_model={"model": "LinearRegression", "model_params": {}},
            mean_rolling_periods=3,
            window=3,
            verbose=0,
        )
        model.fit(df)
        forecast = model.predict(forecast_length=5, just_point_forecast=True)
        
        # Check output shape
        self.assertEqual(forecast.shape, (5, 2))
        self.assertEqual(list(forecast.columns), ["series_1", "series_2"])
        self.assertEqual(forecast.index[0], df.index[-1] + pd.Timedelta(days=1))

    def test_predict_with_bounds(self):
        """Test prediction with probabilistic intervals."""
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {"y1": np.arange(50), "y2": np.arange(50) * 1.5},
            index=idx,
        )
        model = MultivariateRegression(
            forecast_length=3,
            regression_model={"model": "LinearRegression", "model_params": {}},
            mean_rolling_periods=5,
            window=5,
            probabilistic=False,
            verbose=0,
        )
        model.fit(df)
        prediction = model.predict(forecast_length=3, just_point_forecast=False)
        
        # Check that we get upper and lower forecasts
        self.assertEqual(prediction.forecast.shape, (3, 2))
        self.assertEqual(prediction.upper_forecast.shape, (3, 2))
        self.assertEqual(prediction.lower_forecast.shape, (3, 2))
        self.assertIsNotNone(prediction.predict_runtime)
        self.assertIsNotNone(prediction.fit_runtime)

    def test_multivariate_with_different_regressors(self):
        """Test with different regression models."""
        idx = pd.date_range("2020-01-01", periods=40, freq="D")
        df = pd.DataFrame(
            {"col_a": np.sin(np.arange(40)) * 10, "col_b": np.cos(np.arange(40)) * 10},
            index=idx,
        )
        
        # Test with DecisionTree
        model_dt = MultivariateRegression(
            forecast_length=3,
            regression_model={"model": "DecisionTree", "model_params": {"max_depth": 5}},
            mean_rolling_periods=3,
            window=3,
            verbose=0,
        )
        model_dt.fit(df)
        forecast_dt = model_dt.predict(forecast_length=3, just_point_forecast=True)
        self.assertEqual(forecast_dt.shape, (3, 2))

    def test_scale_data(self):
        """Test data scaling functionality."""
        idx = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {"s1": np.arange(30) * 10, "s2": np.arange(30) * 5},
            index=idx,
        )
        
        model = MultivariateRegression(
            forecast_length=3,
            regression_model={"model": "LinearRegression", "model_params": {}},
            mean_rolling_periods=3,
            window=3,
            scale_full_X=True,
            verbose=0,
        )
        model.fit(df)
        
        # Check that scaler was initialized
        self.assertIsNotNone(model.scaler_mean)
        self.assertIsNotNone(model.scaler_std)

    def test_min_threshold_calculation(self):
        """Test that min_threshold is calculated correctly."""
        model = MultivariateRegression(
            mean_rolling_periods=10,
            std_rolling_periods=5,
            window=7,
            additional_lag_periods=15,
        )
        
        # min_threshold should be the max of all period parameters
        self.assertGreaterEqual(model.min_threshold, 15)

    def test_fit_data_method(self):
        """Test fit_data method for updating training data."""
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {"series_1": np.arange(100), "series_2": np.arange(100) * 2},
            index=idx,
        )
        
        model = MultivariateRegression(
            forecast_length=5,
            regression_model={"model": "LinearRegression", "model_params": {}},
            mean_rolling_periods=5,
            window=5,
            verbose=0,
        )
        model.fit(df)
        
        # Create new data
        idx_new = pd.date_range("2020-01-01", periods=120, freq="D")
        df_new = pd.DataFrame(
            {"series_1": np.arange(120), "series_2": np.arange(120) * 2},
            index=idx_new,
        )
        
        # Update with fit_data
        model.fit_data(df_new)
        
        # Check that sktraindata was updated
        self.assertEqual(len(model.sktraindata), min(model.min_threshold, len(df_new)))


if __name__ == "__main__":
    unittest.main()
