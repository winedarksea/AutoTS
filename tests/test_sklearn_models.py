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


if __name__ == "__main__":
    unittest.main()
