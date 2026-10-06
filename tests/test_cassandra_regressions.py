# -*- coding: utf-8 -*-
"""Regression tests for specific Cassandra bugs (one repro per bug)."""
import unittest
import numpy as np
import pandas as pd
from autots.models.cassandra import Cassandra

FORECAST_LENGTH = 14
ZSCORE_DETECTOR = {
    "method": "zscore",
    "method_params": {"distribution": "norm", "alpha": 0.05},
    "transform_dict": None,
}


def make_panel(n_rows=300, n_series=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_rows, freq="D")
    weekly = 3 * np.sin(2 * np.pi * np.arange(n_rows) / 7)[:, None]
    values = 50 + weekly + rng.normal(size=(n_rows, n_series)).cumsum(axis=0) * 0.3
    return pd.DataFrame(
        values, index=idx, columns=[f"s{i}" for i in range(n_series)]
    )


def base_model(**kwargs):
    params = dict(
        seasonalities=[7],
        scaling="BaseScaler",
        regressors_used=False,
        trend_window=None,
        n_jobs=1,
        forecast_length=FORECAST_LENGTH,
    )
    params.update(kwargs)
    return Cassandra(**params)


class CassandraRegressionTest(unittest.TestCase):
    def test_future_impacts_without_past_impacts(self):
        df = make_panel()
        model = base_model().fit(df)
        baseline = model.predict(FORECAST_LENGTH).forecast
        future_index = baseline.index
        future_impacts = pd.DataFrame(0.0, index=future_index, columns=df.columns)
        future_impacts["s0"] = 0.5
        impacted = model.predict(FORECAST_LENGTH, future_impacts=future_impacts)
        ratio = impacted.forecast / baseline
        np.testing.assert_allclose(ratio["s0"], 1.5, rtol=1e-6)
        np.testing.assert_allclose(ratio["s1"], 1.0, rtol=1e-6)

    def test_max_colinearity_drops_first_of_pair(self):
        df = make_panel()
        rng = np.random.default_rng(1)
        a = rng.normal(size=len(df))
        regr = pd.DataFrame(
            {"a": a, "b": a + rng.normal(scale=1e-3, size=len(df))}, index=df.index
        )
        model = base_model(regressors_used=True, max_multicolinearity=None)
        model.fit(df, future_regressor=regr)
        self.assertIn("regr_a", model.drop_colz)
        self.assertNotIn("regr_b", model.drop_colz)

    def test_max_multicolinearity_drops_one_per_dependency(self):
        df = make_panel()
        rng = np.random.default_rng(2)
        r = rng.normal(size=(len(df), 3))
        regr = pd.DataFrame(
            {
                "r1": r[:, 0],
                "r2": r[:, 1],
                "r3": r[:, 0] + r[:, 1],
                "r4": r[:, 2],
                "r5": r[:, 2] - r[:, 0],
            },
            index=df.index,
        )
        model = base_model(regressors_used=True, max_colinearity=None)
        model.fit(df, future_regressor=regr)
        dropped_regressors = [c for c in model.drop_colz if c.startswith("regr_")]
        self.assertEqual(len(dropped_regressors), 2)
        features = model.x_array.drop(columns="intercept")
        min_eigenvalue = np.linalg.eigvalsh(np.corrcoef(features, rowvar=0)).min()
        self.assertGreaterEqual(min_eigenvalue, model.max_multicolinearity)

    def test_group_average_multivariate_predict(self):
        df = make_panel(n_series=4)
        groups = {"s0": "g1", "s1": "g1", "s2": "g2", "s3": "g2"}
        model = base_model(multivariate_feature="group_average")
        model.fit(df, categorical_groups=groups)
        forecast = model.predict(FORECAST_LENGTH).forecast
        self.assertEqual(forecast.shape, (FORECAST_LENGTH, 4))
        self.assertFalse(forecast.isna().any().any())

    def test_new_df_with_past_impacts_matches_original_fit(self):
        df = make_panel()
        past_impacts = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        past_impacts.iloc[-60:, 0] = 0.3
        model = base_model(past_impacts_intervention="remove")
        model.fit(df, past_impacts=past_impacts)
        original = model.predict(FORECAST_LENGTH).forecast
        refit = model.predict(
            FORECAST_LENGTH, new_df=df, past_impacts=past_impacts
        ).forecast
        np.testing.assert_allclose(refit.to_numpy(), original.to_numpy(), rtol=1e-6)

    def test_components_not_origin_space(self):
        df = make_panel()
        model = base_model().fit(df)
        model.predict(FORECAST_LENGTH)
        components = model.return_components(to_origin_space=False)
        self.assertIn(("s0", "trend"), components.columns)

    def test_anomaly_score_forecast_dict_model_params(self):
        df = make_panel()
        model_parameters = {"datepart_method": "simple", "regression_type": "User"}
        model = base_model(
            anomaly_detector_params=ZSCORE_DETECTOR,
            anomaly_intervention={
                "Model": "DatepartRegression",
                "ModelParameters": model_parameters,
                "TransformationParameters": {},
            },
        ).fit(df)
        in_sample = model.predict(forecast_length=None, include_history=True)
        self.assertFalse(in_sample.forecast.isna().all().all())
        forecast = model.predict(FORECAST_LENGTH).forecast
        self.assertFalse(forecast.isna().any().any())
        # caller's params must not be mutated when regression_type is stripped
        self.assertEqual(model_parameters["regression_type"], "User")

    def test_trend_anomaly_univariate_plots_without_anomaly_detector(self):
        import matplotlib

        matplotlib.use("Agg")
        df = make_panel()
        model = base_model(
            trend_anomaly_detector_params={**ZSCORE_DETECTOR, "output": "univariate"},
            trend_window=30,
        ).fit(df)
        prediction = model.predict(FORECAST_LENGTH, include_history=True)
        model.plot_trend(series="s0")
        model.plot_forecast(prediction, series="s0")

    def test_plot_forecast_accepts_series_actuals(self):
        import matplotlib

        matplotlib.use("Agg")
        df = make_panel()
        model = base_model().fit(df)
        prediction = model.predict(FORECAST_LENGTH, include_history=True)
        model.plot_forecast(prediction, actuals=df["s0"], series="s0")


if __name__ == "__main__":
    unittest.main()
