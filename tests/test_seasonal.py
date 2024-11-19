# -*- coding: utf-8 -*-
"""
Seasonal unittests.

Created on Sat May  4 21:43:02 2024

@author: Colin
"""

import unittest
import numpy as np
import pandas as pd
from autots.tools.seasonal import date_part, base_seasonalities, datepart_components, random_datepart, fourier_df
from autots.tools.holiday import holiday_flag
from autots.tools.wavelet import create_narrowing_wavelets, offset_wavelet


class TestSeasonal(unittest.TestCase):

    def test_date_part(self):
        DTindex = pd.date_range("2020-01-01", "2024-01-01", freq="D")
        for method in base_seasonalities:
            df = date_part(DTindex, method=method, set_index=True)
            self.assertEqual(df.shape[0], DTindex.shape[0])
            self.assertGreater(df.shape[1], 1)

    def test_date_components(self):
        DTindex = pd.date_range("2023-01-01", "2024-01-01", freq="h")
        for method in datepart_components:
            df = date_part(DTindex, method=method, set_index=True)
            self.assertEqual(df.shape[0], DTindex.shape[0])

    def test_random_datepart(self):
        out = random_datepart()
        self.assertTrue(out)

    def test_fourier(self):
        DTindex = pd.date_range("2020-01-02", "2024-01-01", freq="D")
        order = 10
        df = fourier_df(DTindex, seasonality=365.25, order=order)
        self.assertEqual(df.shape[1], order * 2)
        self.assertEqual(df.shape[0], DTindex.shape[0])
        self.assertAlmostEqual(df.mean().sum(), 0.0)

    def test_wavelets_repeat(self):
        DTindex = pd.date_range("2020-01-01", "2024-01-01", freq="D")
        origin_ts = "2030-01-01"
        t = (DTindex - pd.Timestamp(origin_ts)).total_seconds() / 86400

        p = 7
        w_order = 7
        weekly_wavelets = offset_wavelet(
            p=p,  # Weekly period
            t=t,  # A full year (365 days)
            # origin_ts=origin_ts,
            order=w_order,  # One offset for each day of the week
            # frequency=2 * np.pi / p,  # Frequency for weekly pattern
            sigma=0.5,  # Smaller sigma for tighter weekly spread
            wavelet_type="ricker",
        )
        self.assertEqual(weekly_wavelets.shape[1], w_order)

        # Example for yearly seasonality
        p = 365.25
        y_order = 12
        yearly_wavelets = offset_wavelet(
            p=p,  # Yearly period
            t=t,  # Three full years
            # origin_ts=origin_ts,
            order=y_order,  # One offset for each month
            # frequency=2 * np.pi / p,  # Frequency for yearly pattern
            sigma=2.0,  # Larger sigma for broader yearly spread
            wavelet_type="morlet",
        )
        yearly_wavelets2 = offset_wavelet(
            p=p,  # Yearly period
            t=t[-100:],  # Three full years
            # origin_ts=origin_ts,
            order=y_order,  # One offset for each month
            # frequency=2 * np.pi / p,  # Frequency for yearly pattern
            sigma=2.0,  # Larger sigma for broader yearly spread
            wavelet_type="morlet",
        )
        self.assertEqual(yearly_wavelets.shape[1], y_order)
        self.assertTrue(np.allclose(yearly_wavelets[-100:], yearly_wavelets2))

    def test_wavelet_continuous(self):
        DTindex = pd.date_range("2020-01-01", "2024-01-01", freq="D")
        origin_ts = "2020-01-01"
        t_full = (DTindex - pd.Timestamp(origin_ts)).total_seconds() / 86400

        p = 365.25  # Example period
        max_order = 5  # Example maximum order

        # Full set of wavelets
        wavelets = create_narrowing_wavelets(p, max_order, t_full)

        # Wavelets for the last 100 days
        t_subset = t_full[-100:]
        wavelet_short = create_narrowing_wavelets(p, max_order, t_subset)

        # Check if the last 100 days of the full series match the subset
        self.assertTrue(np.allclose(wavelets[-100:], wavelet_short))

    def test_holiday_flag(self):
        # hourly being trickier
        train_index = pd.date_range("2020-01-01", "2023-01-01", freq="h")
        pred_index = pd.date_range("2023-01-02", "2024-01-01", freq="h")
        
        train_holiday = holiday_flag(train_index, country=["US", "CA"], encode_holiday_type=True)
        pred_holiday = holiday_flag(pred_index, country=["US", "CA"], encode_holiday_type=True)

        self.assertCountEqual(train_holiday.columns.tolist(), pred_holiday.columns.tolist())
        self.assertGreaterEqual(train_holiday.sum().sum(), 24)
        self.assertGreaterEqual(pred_holiday.sum().sum(), 24)
