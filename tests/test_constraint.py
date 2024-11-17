# -*- coding: utf-8 -*-
"""Test constraint."""
import unittest
import numpy as np
import pandas as pd
from autots import load_daily, ModelPrediction


class TestConstraint(unittest.TestCase):

    def test_constraint(self):
        df = load_daily(long=False)
        if "USW00014771_PRCP" in df.columns:
            # too close to zero, causes one test to fail
            df["USW00014771_PRCP"] = df["USW00014771_PRCP"] + 1
        forecast_length = 30
        constraint_types = {
            "empty": {
                "constraints": None,
            },
            "old_style": {
                "constraint_method": "quantile",
                "constraint_regularization": 0.99,
                "upper_constraint": 0.5,
                "lower_constraint": 0.1,
                "bounds": True,
            },
            "quantile": {
                "constraints": [{
                    "constraint_method": "quantile",
                    "constraint_value": 0.98,
                    "constraint_direction": "upper",
                    "constraint_regularization": 1.0,
                    "bounds": False,
                },]
            },
            "last_value": {
                "constraints": [{
                        "constraint_method": "last_window",
                        "constraint_value": 0.0,
                        "constraint_direction": "upper",
                        "constraint_regularization": 1.0,
                        "bounds": True,
                    },
                    {
                        "constraint_method": "last_window",
                        "constraint_value": 0.0,
                        "constraint_direction": "lower",
                        "constraint_regularization": 1.0,
                        "bounds": True,
                    },
                ]
            },
            "example": {"constraints": [
                {  # don't exceed historic max
                    "constraint_method": "quantile",
                    "constraint_value": 1.0,
                    "constraint_direction": "upper",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
                {  # don't exceed 2% growth by end of forecast horizon
                    "constraint_method": "slope",
                    "constraint_value": {"slope": 0.02, "window": 10, "window_agg": "max", "threshold": 0.01},
                    "constraint_direction": "upper",
                    "constraint_regularization": 0.9,
                    "bounds": False,
                },
                {  # don't go below the last 10 values - 10%
                    "constraint_method": "last_window",
                    "constraint_value": {"window": 10, "threshold": -0.1},
                    "constraint_direction": "lower",
                    "constraint_regularization": 1.0,
                    "bounds": False,
                },
                {  # don't go below zero
                    "constraint_method": "absolute",
                    "constraint_value": 0,  # can also be an array or Series
                    "constraint_direction": "lower",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
                {  # don't go below historic min  - 1 st dev
                    "constraint_method": "stdev_min",
                    "constraint_value": 1.0,
                    "constraint_direction": "lower",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
                {  # don't go above historic mean  + 3 st devs, soft limit
                    "constraint_method": "stdev",
                    "constraint_value": 3.0,
                    "constraint_direction": "upper",
                    "constraint_regularization": 0.5,
                    "bounds": True,
                },
                {  # use a log curve shaped by the historic min/max growth rate to limit
                    "constraint_method": "historic_growth",
                    "constraint_value": 1.0,
                    "constraint_direction": "upper",
                    "constraint_regularization": 1.0,
                    "bounds": True,
                },
            ]},
            "dampening": {
                "constraints": [{
                    "constraint_method": "dampening",
                    "constraint_value": 0.98,
                    "bounds": True,
                },]
            },
        }
        for key, constraint in constraint_types.items():
            with self.subTest(i=key):
                model = ModelPrediction(
                    forecast_length=forecast_length,
                    transformation_dict={
                        "fillna": "median",
                        "transformations": {"0": "SinTrend", "1": "QuantileTransformer", "2": "bkfilter"},
                        "transformation_params": {"0": {}, "1": {"output_distribution": "uniform", "n_quantiles": 1000}, "2": {}}
                    },
                    model_str="SeasonalityMotif",
                    parameter_dict={
                        "window": 7, "point_method": "midhinge",
                        "distance_metric": "canberra", "k": 10,
                        "datepart_method": "common_fourier",
                    },
                    no_negatives=True,
                )
                prediction = model.fit_predict(df, forecast_length=forecast_length)
                # apply an artificially low value
                prediction.forecast.iloc[0, 0] = -10
                prediction.forecast.iloc[0, -1] = df.iloc[:, -1].max() * 1.1
                prediction.plot(df, df.columns[-1])
                prediction.plot(df, df.columns[0])

                prediction.apply_constraints(
                    df_train=df,
                    **constraint
                )
                prediction.plot(df, df.columns[-1])
                prediction.plot(df, df.columns[0])
                # assuming all history was positive as example data currently is
                if key in ["empty", "dampening"]:
                    self.assertTrue(prediction.forecast.min().min() == -10)
                else:
                    self.assertTrue((prediction.forecast.sum() > 0).all())

                if key in ["old_style", "quantile"]:
                    pred_max = prediction.forecast.iloc[:, -1].max()
                    hist_max = df.iloc[:, -1].max()
                    print(pred_max)
                    print(hist_max)
                    self.assertTrue(pred_max <= hist_max)
                if key in ["last_value"]:
                    self.assertTrue(prediction.forecast.iloc[0, :].max() == df.iloc[-1, :].max())