# -*- coding: utf-8 -*-
"""Test anomalies.
Created on Mon Jul 18 16:27:48 2022

@author: Colin
"""
import unittest
import numpy as np
from autots.tools.anomaly_utils import available_methods
from autots.evaluator.anomaly_detector import AnomalyDetector
from autots.datasets import load_live_daily


class TestAnomalies(unittest.TestCase):

    def test_anomalies(self):
        wiki_pages = [
            "Standard_deviation",  # anti-holiday
            "Christmas",
            "Thanksgiving",  # specific holiday
            "all",
        ]
        df = load_live_daily(
            long=False,
            fred_series=None,
            tickers=None,
            trends_list=None,
            earthquake_min_magnitude=None,
            weather_stations=None,
            london_air_stations=None,
            gov_domain_list=None,
            weather_event_types=None,
            wikipedia_pages=wiki_pages,
            sleep_seconds=5,
        )
        tried = []
        while not all(x in tried for x in available_methods):
            params = AnomalyDetector.get_new_params()
            with self.subTest(i=params['method']):
                tried.append(params['method'])
                mod = AnomalyDetector(output='multivariate', **params)
                num_cols = 2
                mod.detect(df[np.random.choice(df.columns, num_cols, replace=False)])
                # detected = mod.anomalies
                # print(params)
                mod.plot()
                self.assertEqual(mod.anomalies.shape, (df.shape, num_cols))

                mod = AnomalyDetector(output='univariate', **params)
                mod.detect(df[np.random.choice(df.columns, num_cols, replace=False)])
                mod.plot()
                self.assertEqual(mod.anomalies.shape, (df.shape[0], 1))
