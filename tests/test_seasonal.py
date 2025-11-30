# -*- coding: utf-8 -*-
"""
Seasonal unittests.

Created on Sat May  4 21:43:02 2024

@author: Colin
"""

import unittest
import numpy as np
import pandas as pd
from autots.tools.seasonal import (
    date_part,
    base_seasonalities,
    datepart_components,
    random_datepart,
    fourier_df,
)
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

        train_holiday = holiday_flag(
            train_index, country=["US", "CA"], encode_holiday_type=True
        )
        pred_holiday = holiday_flag(
            pred_index, country=["US", "CA"], encode_holiday_type=True
        )

        self.assertCountEqual(
            train_holiday.columns.tolist(), pred_holiday.columns.tolist()
        )
        self.assertGreaterEqual(train_holiday.sum().sum(), 24)
        self.assertGreaterEqual(pred_holiday.sum().sum(), 24)

    def test_anchored_warped_fourier_alignment(self):
        DTindex = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        df = date_part(
            DTindex,
            method='anchored_warped_fourier:us_school:4',
            set_index=True,
        )
        memorial_2020 = df.loc[pd.Timestamp('2020-05-25')]
        memorial_2021 = df.loc[pd.Timestamp('2021-05-31')]
        thanksgiving_2020 = df.loc[pd.Timestamp('2020-11-26')]
        thanksgiving_2021 = df.loc[pd.Timestamp('2021-11-25')]

        self.assertTrue(np.allclose(memorial_2020.values, memorial_2021.values))
        self.assertTrue(np.allclose(thanksgiving_2020.values, thanksgiving_2021.values))

    def test_anchored_segment_fourier_gating_daily(self):
        DTindex = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        df = date_part(
            DTindex,
            method='anchored_segment_fourier:us_school:3',
            set_index=True,
        )

        self.assertFalse(any('_hour_' in col for col in df.columns))

        segment0_fourier = [col for col in df.columns if 'segment0_fourier' in col]
        segment1_fourier = [col for col in df.columns if 'segment1_fourier' in col]
        segment1_dow = [col for col in df.columns if 'segment1_dow' in col]
        segment0_dow = [col for col in df.columns if 'segment0_dow' in col]

        jan_row = df.loc[pd.Timestamp('2020-01-15')]
        jun_row = df.loc[pd.Timestamp('2020-06-15')]

        self.assertTrue(np.any(np.abs(jan_row[segment0_fourier]) > 1e-6))
        self.assertTrue(
            np.allclose(jan_row[[c for c in df.columns if 'segment1_' in c]], 0.0)
        )
        self.assertEqual(jan_row[segment0_dow].sum(), 1.0)
        self.assertTrue(np.allclose(jan_row[segment1_dow], 0.0))

        self.assertTrue(np.any(np.abs(jun_row[segment1_fourier]) > 1e-6))
        self.assertEqual(jun_row[segment1_dow].sum(), 1.0)
        self.assertTrue(
            np.allclose(
                jun_row[[c for c in df.columns if 'segment0_fourier' in c]],
                0.0,
            )
        )

    def test_anchored_segment_fourier_hourly(self):
        DTindex = pd.date_range('2020-11-20', '2020-11-27 23:00', freq='h')
        df = date_part(
            DTindex,
            method='anchored_segment_fourier:us_school:2',
            set_index=True,
        )

        self.assertTrue(any('_hour_' in col for col in df.columns))
        noon_row = df.loc[pd.Timestamp('2020-11-25 12:00')]
        active_hour_col = 'anchored_segment_us_school_segment2_hour_12'
        self.assertIn(active_hour_col, df.columns)
        self.assertEqual(noon_row[active_hour_col], 1.0)
        self.assertTrue(
            np.allclose(
                [
                    noon_row[col]
                    for col in df.columns
                    if '_hour_' in col and col != active_hour_col
                ],
                0.0,
            )
        )

    def test_anchored_warped_fourier_weekly(self):
        DTindex = pd.date_range('2020-01-01', '2022-12-31', freq='W-WED')
        df = date_part(
            DTindex,
            method='anchored_warped_fourier:us_school:3',
            set_index=True,
        )

        self.assertEqual(df.shape[0], DTindex.shape[0])
        self.assertGreater(df.shape[1], 0)
        self.assertEqual(df.isna().sum().sum(), 0)

    def test_anchored_segment_fourier_weekly(self):
        DTindex = pd.date_range('2020-01-01', '2022-12-31', freq='W-WED')
        df = date_part(
            DTindex,
            method='anchored_segment_fourier:us_school:2',
            set_index=True,
        )

        self.assertEqual(df.isna().sum().sum(), 0)
        self.assertFalse(any('_hour_' in col for col in df.columns))
