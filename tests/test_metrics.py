# -*- coding: utf-8 -*-
"""Test metrics."""
import unittest
import numpy as np
import pandas as pd
from autots.evaluator.metrics import PredictionEval
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

        output_res = PredictionEval(
            predictions,
            actual,
            series_weights={'a': 10, 'b': 1, 'c': 1},
            df_train=actual,  # just used for SPL scaling
            per_timestamp_errors=False,
        )

        b_avg_metrics = pd.Series(
            [80., 4., 6.325, 0.8, 0.44, 0.5],
            index=['smape', 'mae', 'rmse', 'containment', 'spl', 'contour']
        )
        known_avg_metrics = pd.Series(
            [40., 2., 3.162, 0.467, 0.572, 0.417],
            index=['smape', 'mae', 'rmse', 'containment', 'spl', 'contour']
        )
        known_avg_metrics_weighted = pd.Series(
            [6.667, 0.333, 0.527, 0.567, 0.623, 0.667],
            index=['smape', 'mae', 'rmse', 'containment', 'spl', 'contour']
        )

        self.assertTrue((output_res.avg_metrics.round(3) == known_avg_metrics).all())
        self.assertTrue((output_res.avg_metrics_weighted.round(3) == known_avg_metrics_weighted).all())
        self.assertTrue((output_res.per_series_metrics['b'].round(3) == b_avg_metrics).all())
