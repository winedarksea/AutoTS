# -*- coding: utf-8 -*-
"""Test metrics."""
import unittest
import numpy as np
import pandas as pd
from autots.models.base import PredictionObject


class TestMetrics(unittest.TestCase):

    def test_metrics(self):
        """This at least assures no changes in behavior go unnoticed, hopefully."""
        def custom_metric(A, F, df_train=None, prediction_interval=None):
            submission = F
            objective = A
            abs_err = np.nansum(np.abs(submission - objective))
            err = np.nansum((submission - objective))
            score = abs_err + abs(err)
            epsilon = 1
            big_sum = (
                np.nan_to_num(objective, nan=0.0, posinf=0.0, neginf=0.0).sum().sum()
                + epsilon
            )
            score /= big_sum
            return score

        predictions = PredictionObject()
        predictions.forecast = pd.DataFrame({
            'a': [-10, 10, 10, -10, 0],  # perfect forecast
            'b': [0, 0, 0, 10, 10],
            'c': [np.nan, np.nan, np.nan, np.nan, np.nan]  # all NaN
        })
        predictions.upper_forecast = pd.DataFrame({
            'a': [10, 20, -10, 10, 20],
            'b': [0, 0, 0, 10, 10],
            'c': [0, np.nan, np.nan, np.nan, np.nan]
        })
        predictions.lower_forecast = pd.DataFrame({
            'a': [-10, 0, 10, 10, 0],
            'b': [-10, -10, -10, -10, -5],
            'c': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        actual = pd.DataFrame({
            'a': [-10, 10, 10, -10, 0],
            'b': [0, 10, 0, 0, 10],
            'c': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })

        output_res = predictions.evaluate(
            actual,
            series_weights={'a': 10, 'b': 1, 'c': 1},
            df_train=actual,  # just used for SPL scaling
            per_timestamp_errors=False,
            custom_metric=custom_metric,
        )

        known_avg_metrics = pd.Series(
            [
                40., 2., 3.162, 0.533, 4.000, 0, 0, -3.333, 1.240, 3.333, 1.240, 
                0.572, 0.467, 0.467, 3.333, 0.533, 1.509, 1.250, 4.934, 1.778, 2.699, 0.267, 0.8, 0.952,
            ],
            index=[
                'smape', 'mae', 'rmse', 'made', 'mage', 'mate', 'matse', 'underestimate', 'mle',
                'overestimate', 'imle', 'spl', 'containment', 'contour', 'maxe',
                'oda', 'dwae', 'mqae', 'ewmae', 'uwmse', 'smoothness', "wasserstein", "dwd", "custom",
            ]
        )
        known_avg_metrics_weighted = pd.Series(
            [
                6.667, 0.333, 0.527, 0.089, 4.000, 0, 0, -0.833, 0.207, 0.833, 0.207,
                0.623, 0.567, 0.717, 0.833, 0.883, 1.127, 0.208, 1.234, 0.444, 2.893, 0.044, 0.133, 0.952,
            ],
            index=[
                'smape', 'mae', 'rmse', 'made', 'mage', 'mate', 'matse', 'underestimate', 'mle',
                'overestimate', 'imle', 'spl', 'containment', 'contour', 'maxe',
                'oda', 'dwae', 'mqae', 'ewmae', 'uwmse', 'smoothness', "wasserstein", "dwd", "custom",
            ]
        )
        b_avg_metrics = pd.Series(
            [
                80., 4., 6.325, 1.067, 4.000, 0, 0, -10.0, 2.480, 10.0, 2.480, 0.44,
                0.8, 0.6, 10.0, 0.60, 2.527, 2.50, 14.803, 5.333, 2.140, 0.533, 1.600, 0.952,
            ],
            index=[
                'smape', 'mae', 'rmse', 'made', 'mage', 'mate', 'matse', 'underestimate',
                'mle', 'overestimate', 'imle', 'spl', 'containment', 'contour',
                'maxe', 'oda', 'dwae', 'mqae', 'ewmae', 'uwmse', 'smoothness', "wasserstein", "dwd", "custom",
            ]
        )

        pred_avg_metrics = output_res.avg_metrics.round(3)
        pred_weighted_avg = output_res.avg_metrics_weighted.round(3)
        b_avg = output_res.per_series_metrics['b'].round(3)
        self.assertTrue((pred_avg_metrics == known_avg_metrics).all())
        self.assertTrue((pred_weighted_avg == known_avg_metrics_weighted).all())
        self.assertTrue((b_avg == b_avg_metrics).all())

        # No custom
        output_res = predictions.evaluate(
            actual,
            series_weights={'a': 10, 'b': 1, 'c': 1},
            df_train=actual,  # just used for SPL scaling
            per_timestamp_errors=False,
            custom_metric=None,
        )
        pred_avg_metrics = output_res.avg_metrics.round(3)
        self.assertEqual(pred_avg_metrics["custom"], 0.0)


class TestConstraint(unittest.TestCase):

    def test_constraints(self):
        """This at least assures no changes in behavior go unnoticed, hopefully."""
        predictions = PredictionObject()
        predictions.forecast = pd.DataFrame({
            'a': [-10, 10, 10, -10, 0],  # perfect forecast
            'b': [0, 0, 0, 10, 10],
        }).astype(float)
        df_train = predictions.forecast.copy() + 1
        predictions.upper_forecast = pd.DataFrame({
            'a': [10, 20, -10, 10, 20],
            'b': [0, 0, 0, 10, 10],
        }).astype(float)
        predictions.lower_forecast = pd.DataFrame({
            'a': [-10, 0, 10, 10, 0],
            'b': [-10, -10, -10, -10, -5],
        }).astype(float)
        predictions = predictions.apply_constraints(
            constraint_method="quantile", constraint_regularization=1,
            upper_constraint=None, lower_constraint=0.0,
            bounds=True, df_train=df_train
        )
        predictions = predictions.apply_constraints(
            constraint_method="absolute", constraint_regularization=1,
            upper_constraint=[5.0, 5.0], lower_constraint=None,
            bounds=False, df_train=df_train
        )
        self.assertTrue(10.0 == predictions.lower_forecast.max().max())
        predictions = predictions.apply_constraints(
            constraint_method="stdev", constraint_regularization=0.5,
            upper_constraint=0.5, lower_constraint=None,
            bounds=True, df_train=df_train
        )
        # test lower constraint
        self.assertTrue((df_train.min() == predictions.lower_forecast.min()).all())
        self.assertTrue((df_train.min() == predictions.forecast.min()).all())
        self.assertTrue((df_train.min() == predictions.upper_forecast.min()).all())
        # test upper constraint
        self.assertTrue(10.0 == predictions.forecast.max().sum())
        self.assertTrue((predictions.upper_forecast.round(2).max() == np.array([13.00, 8.87])).all())
