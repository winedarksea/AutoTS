# -*- coding: utf-8 -*-
"""Tests for event risk forecasting.
"""
import unittest
import numpy as np
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
        df = df_full[0: (df_full.shape[0] - forecast_length)]
        df_test = df[(df.shape[0] - forecast_length):]

        upper_limit = 0.8
        # if using manual array limits, historic limit must be defined separately (if used)
        lower_limit = np.ones((forecast_length, df.shape[1]))
        historic_lower_limit = np.ones(df.shape)

        model = EventRiskForecast(
            df,
            forecast_length=forecast_length,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            model_forecast_kwargs={
                "max_generations": 6,
                "verbose": 1,
            }
        )
        # .fit() is optional if model_name, model_param_dict, model_transform_dict are already defined (overwrites)
        model.fit()
        risk_df_upper, risk_df_lower = model.predict()
        historic_upper_risk_df, historic_lower_risk_df = model.predict_historic(lower_limit=historic_lower_limit)
        model.plot(1)

        # also eval summed version
        threshold = 0.1
        eval_upper = EventRiskForecast.generate_historic_risk_array(df_test, model.upper_limit_2d, direction="upper")
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
