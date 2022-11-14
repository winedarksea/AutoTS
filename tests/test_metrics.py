# -*- coding: utf-8 -*-
"""Test metrics."""
import unittest
import numpy as np
import pandas as pd
from autots.models.base import PredictionObject


class TestMetrics(unittest.TestCase):

    def test_metrics(self):
        """This at least assures no changes in behavior go unnoticed, hopefully."""
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
        )

        known_avg_metrics = pd.Series(
            [40., 2., 3.162, 0.533, 4.000, 1.240, 1.240, 0.572, 0.467, 0.467, 5.000, 0.533, 2.119, 1.250],
            index=['smape', 'mae', 'rmse', 'made', 'mage', 'mle', 'imle', 'spl', 'containment', 'contour', 'maxe', 'oda', 'dwae', 'mqae']
        )
        known_avg_metrics_weighted = pd.Series(
            [6.667, 0.333, 0.527, 0.089, 4.000, 0.207, 0.207, 0.623, 0.567, 0.717, 0.833, 0.883, 0.530, 0.208],
            index=['smape', 'mae', 'rmse', 'made', 'mage', 'mle', 'imle', 'spl', 'containment', 'contour', 'maxe', 'oda', 'dwae', 'mqae']
        )
        b_avg_metrics = pd.Series(
            [80., 4., 6.325, 1.067, 4.000, 2.480, 2.480, 0.44, 0.8, 0.6, 10.0, 0.60, 6.356, 2.50],
            index=['smape', 'mae', 'rmse', 'made', 'mage', 'mle', 'imle', 'spl', 'containment', 'contour', 'maxe', 'oda', 'dwae', 'mqae']
        )

        self.assertTrue((output_res.avg_metrics.round(3) == known_avg_metrics).all())
        self.assertTrue((output_res.avg_metrics_weighted.round(3) == known_avg_metrics_weighted).all())
        self.assertTrue((output_res.per_series_metrics['b'].round(3) == b_avg_metrics).all())


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
