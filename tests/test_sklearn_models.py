import os
import tempfile
import random
import unittest
import datetime

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
    WindowRegression,
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

    def test_radius_neighbors_radius_is_scale_free(self):
        # an absolute radius of 2 finds no neighbors at this scale
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = X @ np.array([1.0, -2.0, 0.5, 3.0])
        query = rng.normal(size=(20, 4))
        params = {"model": "RadiusNeighbors", "model_params": {"radius": 2.0}}
        small = retrieve_regressor(params, multioutput=False).fit(X, y)
        large = retrieve_regressor(params, multioutput=False).fit(X * 1e5, y)
        np.testing.assert_allclose(
            small.predict(query), large.predict(query * 1e5), rtol=1e-8
        )
        self.assertTrue(np.all(np.isfinite(small.predict(query))))

    def _radius(self, radius=1.5, **kwargs):
        return retrieve_regressor(
            {
                "model": "RadiusNeighbors",
                "model_params": {"radius": radius, **kwargs},
            },
            multioutput=False,
        )

    def test_radius_neighbors_radius_scales_with_query_distance(self):
        X = np.arange(100, dtype=float).reshape(-1, 1)
        y = np.arange(100, dtype=float)
        model = self._radius(1.5).fit(X, y)
        # nearest row is 99 (d1=401); 1.5*d1 holds the 50-row cap, rows 50-99
        np.testing.assert_allclose(model.predict(np.array([[500.0]])), [74.5])
        # in range: d1=0.2, so only rows within 0.3 count
        np.testing.assert_allclose(model.predict(np.array([[50.2]])), [50.0])

    def test_radius_below_one_is_nearest_row(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        y = np.arange(20, dtype=float) * 3
        model = self._radius(0.5).fit(X, y)
        np.testing.assert_allclose(model.predict(np.array([[7.3], [30.0]])), [21, 57])

    def test_radius_neighbors_mixed_scale_stays_local(self):
        # one global radius let small-scale rows average large-scale targets
        rng = np.random.default_rng(0)
        small = rng.normal(1.0, 0.1, (60, 3))
        large = rng.normal(1e4, 1e3, (60, 3))
        X = np.vstack([small, large])
        y = np.r_[np.ones(60), np.full(60, 1e4)]
        for radius in (1.1, 2.0, 5.0):
            pred = self._radius(radius).fit(X, y).predict(small[:10] * 1.05)
            np.testing.assert_allclose(pred, 1.0)

    def test_radius_distance_weights_and_multioutput(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        Y = np.column_stack([X[:, 0], 2 * X[:, 0]])
        model = self._radius(2.0, weights="distance").fit(X, Y)
        pred = model.predict(np.array([[4.0], [6.4]]))
        # exact match takes all the weight; else 1/d over rows 6 and 7
        expected = (6 / 0.4 + 7 / 0.6) / (1 / 0.4 + 1 / 0.6)
        np.testing.assert_allclose(pred, [[4, 8], [expected, 2 * expected]])

    def test_radius_one_row_fit(self):
        model = self._radius(2.0).fit(np.array([[1.0, 2.0]]), np.array([3.0]))
        np.testing.assert_allclose(model.predict(np.array([[5.0, 5.0]])), [3.0])


class _InfOnFirstPredict:
    """Wraps a fitted regressor; the first predict returns inf in every cell."""

    def __init__(self, model):
        self.model = model
        self.calls = 0

    def predict(self, X):
        pred = np.asarray(self.model.predict(X), dtype=float)
        self.calls += 1
        return np.full_like(pred, np.inf) if self.calls == 1 else pred


class TestRecursiveNonFiniteGuards(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2020-01-01", periods=60, freq="D")
        t = np.arange(60, dtype=float)
        self.df = pd.DataFrame({"a": 10 + t, "b": 50 - 0.5 * t}, index=idx)

    def test_rolling_regression_carries_last_value(self):
        model = RollingRegression(
            regression_model={"model": "LinearRegression", "model_params": {}},
            window=3,
            mean_rolling_periods=3,
        ).fit(self.df)
        model.model = _InfOnFirstPredict(model.model)
        forecast = model.predict(forecast_length=4, just_point_forecast=True)
        self.assertTrue(np.isfinite(forecast.to_numpy()).all())
        np.testing.assert_allclose(forecast.iloc[0], self.df.iloc[-1])

    def test_multivariate_regression_carries_last_value(self):
        model = MultivariateRegression(
            forecast_length=4,
            regression_model={"model": "LinearRegression", "model_params": {}},
            mean_rolling_periods=3,
            window=3,
            verbose=0,
        ).fit(self.df)
        model.model = _InfOnFirstPredict(model.model)
        forecast = model.predict(forecast_length=4, just_point_forecast=True)
        self.assertTrue(np.isfinite(forecast.to_numpy()).all())


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

    def test_radius_neighbors_trending_large_scale_is_finite(self):
        # forecasts walk past the training range, where the old absolute radius
        # returned NaN for every step
        idx = pd.date_range("2021-01-01", periods=120, freq="D")
        t = np.arange(120, dtype=float)
        df = pd.DataFrame(
            {"a": 1e5 + 500 * t + 2e3 * np.sin(t / 3), "b": 3e5 - 200 * t},
            index=idx,
        )
        model = RollingRegression(
            regression_model={
                "model": "RadiusNeighbors",
                "model_params": {"radius": 2.0},
            },
            window=7,
            mean_rolling_periods=14,
        )
        model.fit(df)
        forecast = model.predict(forecast_length=10, just_point_forecast=True)
        self.assertEqual(forecast.shape, (10, 2))
        self.assertTrue(np.all(np.isfinite(forecast.to_numpy())))

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
            regression_model={
                "model": "DecisionTree",
                "model_params": {"max_depth": 5},
            },
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

    def test_fit_runtime_captures_fit_work(self):
        """fit_runtime should reflect fit execution, not be reset to near-zero."""
        idx = pd.date_range("2020-01-01", periods=300, freq="D")
        data = {
            f"series_{i}": np.arange(300) * (i + 1) + np.random.randn(300) * 0.1
            for i in range(8)
        }
        df = pd.DataFrame(data, index=idx)

        model = MultivariateRegression(
            forecast_length=5,
            regression_model={"model": "LinearRegression", "model_params": {}},
            mean_rolling_periods=7,
            window=7,
            verbose=0,
        )
        wall_start = datetime.datetime.now()
        model.fit(df)
        elapsed = datetime.datetime.now() - wall_start

        self.assertGreater(model.fit_runtime.total_seconds(), 0)
        self.assertLessEqual(
            model.fit_runtime.total_seconds(), elapsed.total_seconds() + 0.05
        )
        if elapsed.total_seconds() > 0.05:
            self.assertGreater(
                model.fit_runtime.total_seconds(), elapsed.total_seconds() * 0.2
            )


class TestWindowRegression(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2021-01-01", periods=60, freq="D")
        values = np.arange(60, dtype=float)
        self.training_df = pd.DataFrame({"y": values}, index=idx)

    def test_basic_forecast_length_output(self):
        model = WindowRegression(
            window_size=8,
            forecast_length=4,
            regression_model={"model": "LinearRegression", "model_params": {}},
            input_dim="univariate",
            output_dim="forecast_length",
            shuffle=False,
            scale=True,
            random_seed=1,
        )
        model.fit(self.training_df)
        forecast = model.predict(just_point_forecast=True)

        self.assertEqual(forecast.shape, (4, 1))
        self.assertEqual(list(forecast.columns), ["y"])
        self.assertTrue(np.all(np.isfinite(forecast.values)))
        self.assertEqual(
            forecast.index[0], self.training_df.index[-1] + pd.Timedelta(days=1)
        )
        self.assertTrue(hasattr(model, "scaler"))

    def test_requires_future_regressor_for_user_type(self):
        model = WindowRegression(
            window_size=6,
            forecast_length=2,
            regression_model={"model": "LinearRegression", "model_params": {}},
            regression_type="User",
        )

        with self.assertRaises(ValueError):
            model.fit(self.training_df)

    def test_user_regressor_with_future_values(self):
        model = WindowRegression(
            window_size=6,
            forecast_length=3,
            regression_model={"model": "LinearRegression", "model_params": {}},
            regression_type="User",
            random_seed=2,
        )
        future_reg_train = pd.DataFrame(
            {"reg": np.linspace(0.0, 1.0, len(self.training_df))},
            index=self.training_df.index,
        )
        model.fit(self.training_df, future_regressor=future_reg_train)

        future_index = pd.date_range(
            self.training_df.index[-1] + pd.Timedelta(days=1),
            periods=3,
            freq="D",
        )
        future_reg_future = pd.DataFrame({"reg": [1.1, 1.2, 1.3]}, index=future_index)
        forecast = model.predict(
            future_regressor=future_reg_future, just_point_forecast=True
        )

        self.assertEqual(forecast.shape, (3, 1))
        self.assertEqual(list(forecast.columns), ["y"])
        self.assertTrue(np.all(np.isfinite(forecast.values)))
        self.assertEqual(forecast.index[0], future_index[0])

    def test_one_step_prediction_updates_window(self):
        model = WindowRegression(
            window_size=5,
            forecast_length=3,
            regression_model={"model": "LinearRegression", "model_params": {}},
            output_dim="1step",
            random_seed=3,
        )
        model.fit(self.training_df)
        forecast = model.predict(just_point_forecast=True)

        self.assertEqual(forecast.shape, (3, 1))
        self.assertEqual(list(forecast.columns), ["y"])
        self.assertEqual(
            forecast.index[-1], self.training_df.index[-1] + pd.Timedelta(days=3)
        )
        self.assertTrue(np.all(np.isfinite(forecast.values)))
        self.assertEqual(
            model.last_window.shape[0], model.window_size + model.forecast_length
        )


if __name__ == "__main__":
    unittest.main()
