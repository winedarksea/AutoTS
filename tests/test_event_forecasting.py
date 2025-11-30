# -*- coding: utf-8 -*-
"""Tests for event risk forecasting.
"""
import unittest
import numpy as np
import pandas as pd
from autots import (
    load_weekly,
    load_daily,
    EventRiskForecast,
)


class TestEventRisk(unittest.TestCase):
    def test_event_risk(self):
        print("Starting test_event_risk")
        """This at least assures no changes in behavior go unnoticed, hopefully."""
        forecast_length = 6
        df_full = load_weekly(long=False)
        df = df_full[0 : (df_full.shape[0] - forecast_length)]
        df_test = df[(df.shape[0] - forecast_length) :]

        upper_limit = 0.8
        # if using manual array limits, historic limit must be defined separately (if used)
        lower_limit = pd.DataFrame(
            np.ones((forecast_length, df.shape[1])),
            columns=df.columns,
        )
        historic_lower_limit = pd.DataFrame(
            np.ones(df.shape),
            index=df.index,
            columns=df.columns,
        )

        model = EventRiskForecast(
            df,
            forecast_length=forecast_length,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            model_forecast_kwargs={
                "max_generations": 6,
                "verbose": 1,
                "transformer_list": "no_expanding",
            },
        )
        # .fit() is optional if model_name, model_param_dict, model_transform_dict are already defined (overwrites)
        model.fit()
        risk_df_upper, risk_df_lower = model.predict()
        historic_upper_risk_df, historic_lower_risk_df = model.predict_historic(
            lower_limit=historic_lower_limit
        )
        model.plot(column=df.columns[0])
        self.assertTrue(np.array_equal(model.lower_limit_2d, lower_limit.to_numpy()))

        # also eval summed version
        threshold = 0.1
        eval_upper = EventRiskForecast.generate_historic_risk_array(
            df_test, model.upper_limit_2d, direction="upper"
        )
        pred_upper = np.where(model.upper_risk_array > threshold, 1, 0)

        self.assertTrue(risk_df_lower.shape == (forecast_length, df.shape[1]))
        self.assertFalse(risk_df_upper.isnull().all().all())
        self.assertFalse(risk_df_lower.isnull().all().all())
        self.assertTrue(historic_upper_risk_df.shape == df.shape)
        self.assertTrue(historic_lower_risk_df.shape == df.shape)
        self.assertFalse(historic_lower_risk_df.isnull().all().all())
        self.assertGreaterEqual(np.sum(pred_upper), 1)
        self.assertTrue(eval_upper.shape == pred_upper.shape)

    def test_event_risk_univariate(self):
        print("Starting test_event_risk_univariate")
        """This at least assures no changes in behavior go unnoticed, hopefully."""
        df = load_daily(long=False)
        df = df.iloc[:, 0:1]
        upper_limit = None
        lower_limit = {
            "model_name": "ARIMA",
            "model_param_dict": {'p': 1, "d": 0, "q": 1},
            "model_transform_dict": {},
            "prediction_interval": 0.5,
        }
        forecast_length = 6

        model = EventRiskForecast(
            df,
            forecast_length=forecast_length,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            prediction_interval=[0.9, 0.8, 0.7, 0.6, 0.5],
            model_forecast_kwargs={
                "max_generations": 6,
                "verbose": 2,
                "n_jobs": "auto",
                "random_seed": 321,
                "transformer_list": "no_expanding",
            },
        )
        model.fit(model_list="fast")
        risk_df_upper, risk_df_lower = model.predict()
        historic_upper_risk_df, historic_lower_risk_df = model.predict_historic()
        model.plot(0)

        self.assertTrue(risk_df_upper.isnull().all().all())
        self.assertTrue(risk_df_lower.shape == (forecast_length, df.shape[1]))
        self.assertFalse(risk_df_lower.isnull().all().all())
        self.assertTrue(historic_lower_risk_df.shape == df.shape)

    def test_event_risk_balltree_models(self):
        print("Starting test_event_risk_balltree_models")
        df = load_daily(long=False).iloc[:240, :3]
        forecast_length = 5

        multivariate_params = {
            "window": 10,
            "point_method": "mean",
            "distance_metric": "euclidean",
            "k": 100,
            "sample_fraction": None,
            "comparison_transformation": None,
            "combination_transformation": {
                "fillna": "time",
                "transformations": {"0": "DiffSmoother"},
                "transformation_params": {
                    "0": {
                        "method": "IQR",
                        "method_params": {
                            "iqr_threshold": 2.5,
                            "iqr_quantiles": [0.25, 0.75],
                        },
                        "transform_dict": None,
                        "reverse_alignment": False,
                        "isolated_only": True,
                        "fillna": "rolling_mean_24",
                    }
                },
            },
        }
        multivariate_transform = {
            "fillna": "fake_date",
            "transformations": {
                "0": "Log",
                "1": "ChangepointDetrend",
                "2": "AlignLastDiff",
            },
            "transformation_params": {
                "0": {},
                "1": {
                    "model": "Linear",
                    "changepoint_spacing": 5040,
                    "changepoint_distance_end": 520,
                    "datepart_method": "common_fourier",
                },
                "2": {
                    "rows": 1,
                    "displacement_rows": 1,
                    "quantile": 1.0,
                    "decay_span": 3,
                },
            },
        }

        model_multivariate = EventRiskForecast(
            df,
            forecast_length=forecast_length,
            upper_limit=None,
            lower_limit=0.3,
            model_name="BallTreeMultivariateMotif",
            model_param_dict=multivariate_params,
            model_transform_dict=multivariate_transform,
            model_forecast_kwargs={
                "verbose": 0,
                "n_jobs": 1,
                "random_seed": 321,
            },
        )
        risk_upper, risk_lower = model_multivariate.predict()

        self.assertTrue(risk_lower.shape == (forecast_length, df.shape[1]))
        self.assertTrue(risk_upper.isnull().all().all())
        self.assertIsNotNone(model_multivariate.window_index)
        self.assertEqual(
            model_multivariate.window_index.shape,
            (df.shape[1], multivariate_params["k"]),
        )
        self.assertEqual(
            model_multivariate.result_windows.shape,
            (multivariate_params["k"], forecast_length, df.shape[1]),
        )
        raw_multivariate = model_multivariate.prediction_object.model.result_windows
        self.assertFalse(
            np.allclose(
                raw_multivariate, model_multivariate.result_windows, equal_nan=True
            )
        )
        self.assertTrue(np.isfinite(model_multivariate.result_windows).all())

        regression_params = {
            "window": 3,
            "point_method": "midhinge",
            "distance_metric": "euclidean",
            "k": 100,
            "sample_fraction": 5000000,
            "comparison_transformation": {
                "fillna": "cubic",
                "transformations": {"0": "AlignLastDiff"},
                "transformation_params": {
                    "0": {
                        "rows": 364,
                        "displacement_rows": 1,
                        "quantile": 1.0,
                        "decay_span": None,
                    }
                },
            },
            "combination_transformation": {
                "fillna": "time",
                "transformations": {"0": "AlignLastDiff"},
                "transformation_params": {
                    "0": {
                        "rows": 7,
                        "displacement_rows": 1,
                        "quantile": 1.0,
                        "decay_span": 2,
                    }
                },
            },
            "extend_df": True,
            "mean_rolling_periods": 12,
            "macd_periods": 74,
            "std_rolling_periods": 30,
            "max_rolling_periods": 12,
            "min_rolling_periods": None,
            "quantile90_rolling_periods": 10,
            "quantile10_rolling_periods": 10,
            "ewm_alpha": None,
            "ewm_var_alpha": None,
            "additional_lag_periods": None,
            "abs_energy": False,
            "rolling_autocorr_periods": None,
            "nonzero_last_n": None,
            "datepart_method": None,
            "polynomial_degree": None,
            "regression_type": None,
            "holiday": False,
            "scale_full_X": False,
            "series_hash": True,
            "frac_slice": None,
        }
        regression_transform = {
            "fillna": "akima",
            "transformations": {
                "0": "Log",
                "1": "SinTrend",
                "2": "ChangepointDetrend",
                "3": "RobustScaler",
            },
            "transformation_params": {
                "0": {},
                "1": {},
                "2": {
                    "model": "Linear",
                    "changepoint_spacing": 5040,
                    "changepoint_distance_end": 520,
                    "datepart_method": "common_fourier",
                },
                "3": {},
            },
        }

        model_regression = EventRiskForecast(
            df,
            forecast_length=forecast_length,
            upper_limit=0.8,
            lower_limit=None,
            model_name="BallTreeRegressionMotif",
            model_param_dict=regression_params,
            model_transform_dict=regression_transform,
            model_forecast_kwargs={
                "verbose": 0,
                "n_jobs": 1,
                "random_seed": 321,
            },
        )
        risk_upper_reg, risk_lower_reg = model_regression.predict()

        self.assertTrue(risk_lower_reg.isnull().all().all())
        self.assertFalse(risk_upper_reg.isnull().all().all())
        self.assertIsNotNone(model_regression.window_index)
        self.assertEqual(
            model_regression.window_index.shape,
            (df.shape[1], regression_params["k"]),
        )
        self.assertEqual(
            model_regression.result_windows.shape,
            (regression_params["k"], forecast_length, df.shape[1]),
        )
        raw_regression = model_regression.prediction_object.model.result_windows
        self.assertFalse(
            np.allclose(raw_regression, model_regression.result_windows, equal_nan=True)
        )
        self.assertTrue(np.isfinite(model_regression.result_windows).all())

    def test_event_risk_diff_window_result_windows(self):
        print("Starting test_event_risk_diff_window_result_windows")
        df = load_daily(long=False).iloc[:200, :2]
        forecast_length = 5

        motif_params = {
            "window": 10,
            "point_method": "mean",
            "distance_metric": "euclidean",
            "k": 5,
        }
        motif_transform = {
            "fillna": "fake_date",
            "transformations": {
                "0": "AlignLastValue",
            },
            "transformation_params": {
                "0": {
                    "rows": 1,
                    "lag": 1,
                    "method": "additive",
                    "strength": 1.0,
                    "first_value_only": False,
                    "threshold": None,
                    "threshold_method": "max",
                }
            },
        }

        model = EventRiskForecast(
            df,
            forecast_length=forecast_length,
            upper_limit=0.7,
            lower_limit=None,
            model_name="UnivariateMotif",
            model_param_dict=motif_params,
            model_transform_dict=motif_transform,
            model_forecast_kwargs={
                "verbose": 0,
                "n_jobs": 1,
                "random_seed": 321,
            },
        )
        risk_upper, risk_lower = model.predict()

        self.assertFalse(risk_upper.isnull().all().all())
        self.assertTrue(risk_lower.isnull().all().all())
        self.assertIsNone(model.window_index)
        self.assertEqual(
            model.result_windows.shape,
            (motif_params["k"], forecast_length, df.shape[1]),
        )
        raw_windows = model.prediction_object.model.result_windows
        self.assertIsInstance(raw_windows, dict)
        raw_array = np.moveaxis(np.array(list(raw_windows.values())), 0, -1)
        self.assertFalse(np.allclose(raw_array, model.result_windows, equal_nan=True))
        self.assertTrue(np.isfinite(model.result_windows).all())
