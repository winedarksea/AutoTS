# -*- coding: utf-8 -*-
"""Test anomalies.
Created on Mon Jul 18 16:27:48 2022

@author: Colin
"""
import unittest
import numpy as np
import pandas as pd
from autots.tools.anomaly_utils import available_methods, fast_methods
from autots.evaluator.anomaly_detector import AnomalyDetector, HolidayDetector
from autots.datasets import load_live_daily


def dict_loop(params):
    if 'transform_dict' in params.keys():
        x = params.get('transform_dict', {})
        if isinstance(x, dict):
            x = x.get('transformations', {})
            return x
    elif 'anomaly_detector_params' in params.keys():
        x = params.get('anomaly_detector_params', {})
        if isinstance(x, dict):
            x = params.get('transform_dict', {})
            if isinstance(x, dict):
                x = x.get('transformations', {})
                return x
    return {}


class TestAnomalies(unittest.TestCase):
    @classmethod
    def setUp(self):
        wiki_pages = [
            "Standard_deviation",  # anti-holiday
            "Christmas",
            "Thanksgiving",  # specific holiday
            "all",
        ]
        self.df = load_live_daily(
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
            caiso_query=None,
            sleep_seconds=10,
        ).fillna(0).replace(np.inf, 0)

    def test_anomaly_holiday_detectors(self):
        print("Starting test_anomaly_holiday_detectors")
        """Combininng these to minimize live data download."""
        tried = []
        while not all(x in tried for x in available_methods):
            params = AnomalyDetector.get_new_params(method="deep")
            # remove 'Slice' as it messes up assertions
            while 'Slice' in dict_loop(params).values():
                params = AnomalyDetector.get_new_params(method="deep")
            with self.subTest(i=params['method']):
                tried.append(params['method'])
                mod = AnomalyDetector(output='multivariate', **params)
                num_cols = 2
                mod.detect(self.df[np.random.choice(self.df.columns, num_cols, replace=False)])
                # detected = mod.anomalies
                # print(params)
                # mod.plot()
                self.assertEqual(mod.anomalies.shape, (self.df.shape[0], num_cols), msg=f"from params {params}")

                mod = AnomalyDetector(output='univariate', **params)
                mod.detect(self.df[np.random.choice(self.df.columns, num_cols, replace=False)])
                self.assertEqual(mod.anomalies.shape, (self.df.shape[0], 1))
        # mod.plot()

        from prophet import Prophet

        tried = []
        forecast_length = 28
        holidays_detected = 0
        full_dates = self.df.index.union(pd.date_range(self.df.index.max(), freq="D", periods=forecast_length))

        while not all(x in tried for x in fast_methods):
            params = HolidayDetector.get_new_params(method="fast")
            with self.subTest(i=params["anomaly_detector_params"]['method']):
                while 'Slice' in dict_loop(params).values():
                    params = HolidayDetector.get_new_params(method="fast")
                tried.append(params['anomaly_detector_params']['method'])
                mod = HolidayDetector(**params)
                mod.detect(self.df.copy())
                prophet_holidays = mod.dates_to_holidays(full_dates, style="prophet")
    
                for series in self.df.columns:
                    # series = "wiki_George_Washington"
                    holiday_subset = prophet_holidays[prophet_holidays['series'] == series]
                    if holiday_subset.shape[0] >= 1:
                        holidays_detected = 1
                    m = Prophet(holidays=holiday_subset)
                    # m = Prophet()
                    m.fit(pd.DataFrame({'ds': self.df.index, 'y': self.df[series]}))
                    future = m.make_future_dataframe(forecast_length)
                    fcst = m.predict(future).set_index('ds')  # noqa
                    # m.plot_components(fcst)
        # mod.plot()
        temp = mod.dates_to_holidays(full_dates, style="flag")
        temp = mod.dates_to_holidays(full_dates, style="series_flag")
        temp = mod.dates_to_holidays(full_dates, style="impact")
        temp = mod.dates_to_holidays(full_dates, style="long")  # noqa
        # this is a weak test, but will capture some functionality
        self.assertEqual(holidays_detected, 1, "no methods detected holidays")
