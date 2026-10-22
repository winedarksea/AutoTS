"""Automatic spacing depends on elapsed duration, not calendar years touched."""

import unittest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from autots.models.basics import BasicLinearModel
from autots.tools.changepoints import half_yr_spacing
from autots.tools.profile import profile_time_series
from autots.tools.transform import ChangepointDetrend


class TestHalfYearSpacing(unittest.TestCase):
    def test_equal_duration_windows_have_equal_spacing(self):
        for rows in (730, 1095, 1461):
            expected = int(rows / (2 * ((rows - 1) / 365.25)))
            for start in ('2023-01-01', '2022-12-31', '2023-07-01'):
                df = pd.DataFrame(index=pd.date_range(start, periods=rows))
                with self.subTest(rows=rows, start=start):
                    self.assertEqual(half_yr_spacing(df), expected)

    def test_subyear_and_single_row_spacing_stays_positive(self):
        for rows in (1, 2, 30, 180, 365):
            for start in ('2023-01-01', '2023-10-01'):
                df = pd.DataFrame(index=pd.date_range(start, periods=rows))
                with self.subTest(rows=rows, start=start):
                    self.assertEqual(half_yr_spacing(df), max(1, rows // 2))

    def test_fractional_days_are_included(self):
        index = pd.date_range('2020-01-01', '2021-07-01 18:00:00', periods=1000)
        elapsed_days = (index[-1] - index[0]).total_seconds() / 86400
        expected = int(1000 / (2 * elapsed_days / 365.25))
        self.assertEqual(half_yr_spacing(pd.DataFrame(index=index)), expected)

    def test_centuries_of_monthly_history_do_not_overflow(self):
        index = pd.date_range('1800-01-01', '2200-01-01', freq='MS')
        index = pd.DatetimeIndex(index.to_numpy(dtype='datetime64[ns]'))
        self.assertEqual(half_yr_spacing(pd.DataFrame(index=index)), 6)

    def test_timezone_spacing_uses_elapsed_utc_time_across_dst(self):
        index = pd.date_range(
            '2022-01-01', '2023-07-01', periods=287,
            tz=ZoneInfo('America/Chicago'),
        )
        utc_index = index.tz_convert(None)
        self.assertEqual(half_yr_spacing(pd.DataFrame(index=index)), 96)
        self.assertEqual(
            half_yr_spacing(pd.DataFrame(index=index)),
            half_yr_spacing(pd.DataFrame(index=utc_index)),
        )

    def test_automatic_new_product_threshold_is_calendar_independent(self):
        for start in ('2023-01-01', '2022-12-31', '2023-07-01'):
            df = pd.DataFrame(
                {'series': np.r_[np.zeros(640), np.ones(90)]},
                index=pd.date_range(start, periods=730),
            )
            with self.subTest(start=start):
                result = profile_time_series(df, seasonal_threshold=1)
                self.assertEqual(result.loc[0, 'PROFILE'], 'new_product')

    def test_models_and_detrending_share_elapsed_spacing(self):
        df = pd.DataFrame(
            {'series': 5 + np.arange(730) / 100},
            index=pd.date_range('2023-07-01', periods=730),
        )
        model = BasicLinearModel(
            changepoint_spacing=None,
            changepoint_distance_end=None,
            holiday_countries_used=False,
        ).fit(df)
        transformer = ChangepointDetrend(
            changepoint_spacing=None, changepoint_distance_end=None
        ).fit(df)
        for fitted in (model, transformer):
            with self.subTest(fitted=type(fitted).__name__):
                self.assertEqual(fitted.changepoint_spacing, 182)
                self.assertEqual(fitted.changepoint_distance_end, 91)
