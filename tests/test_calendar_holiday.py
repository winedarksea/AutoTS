# -*- coding: utf-8 -*-
"""Test calendars
"""
import unittest
import numpy as np
import pandas as pd
from autots import load_daily
from autots.tools.calendar import gregorian_to_chinese, gregorian_to_islamic, gregorian_to_hebrew, gregorian_to_hindu
from autots.tools.lunar import moon_phase
from autots.tools.holiday import holiday_flag
from autots.tools.seasonal import date_part


class TestCalendar(unittest.TestCase):

    def test_chinese(self):
        print("Starting test_chinese")
        input_dates = [
            "2014-01-01", "2022-06-30", "2026-02-28", "2000-02-05", "2040-04-15"
        ]
        result = gregorian_to_chinese(input_dates)
        result1 = result.iloc[0][['lunar_year', 'lunar_month', 'lunar_day']].tolist()
        result2 = result.iloc[1][['lunar_year', 'lunar_month', 'lunar_day']].tolist()
        result3 = result.iloc[2][['lunar_year', 'lunar_month', 'lunar_day']].tolist()
        result4 = result.iloc[3][['lunar_year', 'lunar_month', 'lunar_day']].tolist()
        result5 = result.iloc[4][['lunar_year', 'lunar_month', 'lunar_day']].tolist()
        self.assertEqual(result1, [2013, 12, 1])
        self.assertEqual(result2, [2022, 6, 2])
        self.assertEqual(result3, [2026, 1, 12])  # 2030 01 26
        self.assertEqual(result4, [2000, 1, 1])
        self.assertEqual(result5, [2040, 3, 5])

    def test_islamic(self):
        print("Starting test_islamic")
        input_dates = [
            "2014-01-01", "2022-06-30", "2030-02-28", "2000-12-31", "2040-04-15"
        ]
        result = gregorian_to_islamic(input_dates)
        result1 = result.iloc[0][['year', 'month', 'day']].tolist()
        result2 = result.iloc[1][['year', 'month', 'day']].tolist()
        result3 = result.iloc[2][['year', 'month', 'day']].tolist()
        result4 = result.iloc[3][['year', 'month', 'day']].tolist()
        result5 = result.iloc[4][['year', 'month', 'day']].tolist()
        self.assertEqual(result1, [1435, 2, 29])
        self.assertEqual(result2, [1443, 12, 1])
        self.assertEqual(result3, [1451, 10, 25])
        self.assertEqual(result4, [1421, 10, 5])
        self.assertEqual(result5, [1462, 4, 3])

    def test_lunar(self):
        print("Starting test_lunar")
        self.assertAlmostEqual(moon_phase(pd.Timestamp("2022-07-18")), 0.686, 3)
        self.assertAlmostEqual(moon_phase(pd.Timestamp("1995-11-07")), 0.998, 3)
        self.assertAlmostEqual(moon_phase(pd.Timestamp("2035-02-08")), 0.002, 3)

    def test_hebrew(self):
        print("Starting test_hebrew")
        input_dates = [
            "2014-01-01", "2022-06-30", "2030-02-28", "2000-12-31", "2040-04-15"
        ]
        result = gregorian_to_hebrew(input_dates)
        result1 = result.iloc[0][['year', 'month', 'day']].tolist()
        result2 = result.iloc[1][['year', 'month', 'day']].tolist()
        result3 = result.iloc[2][['year', 'month', 'day']].tolist()
        result4 = result.iloc[3][['year', 'month', 'day']].tolist()
        result5 = result.iloc[4][['year', 'month', 'day']].tolist()
        self.assertEqual(result1, [5774, 10, 29])
        self.assertEqual(result2, [5782, 4, 1])
        self.assertEqual(result3, [5790, 12, 25])
        self.assertEqual(result4, [5761, 10, 5])
        self.assertEqual(result5, [5800, 2, 2])

    def test_hindu(self):
        # Diwali in 2021 was on November 4, 2021
        date = pd.to_datetime(['2021-11-04'])
        result = gregorian_to_hindu(date)
        # expected_month_name = 'Kartika'
        # expected_lunar_day = 30  # Amavasya is typically the 30th day
        # self.assertEqual(result.iloc[0]['hindu_month_name'], expected_month_name)
        # self.assertEqual(result.iloc[0]['lunar_day'], expected_lunar_day)

        # Diwali in 2024 was on October 31, 2024
        date = pd.to_datetime(['2024-10-31'])
        result = gregorian_to_hindu(date)  # noqa
        # expected_month_name = 'Kartika'
        # expected_lunar_day = 30  # Amavasya is typically the 30th day
        # self.assertEqual(result.iloc[0]['hindu_month_name'], expected_month_name)
        # self.assertEqual(result.iloc[0]['lunar_day'], expected_lunar_day)


class TestHolidayFlag(unittest.TestCase):

    def test_holiday_flag(self):
        print("Starting test_holiday_flag")
        input_dates = pd.date_range("2022-01-01", "2023-01-01", freq='D')
        flag_1 = holiday_flag(input_dates, country="US", encode_holiday_type=False, holidays_subdiv="PR")
        self.assertAlmostEqual(flag_1.loc["2022-07-04"].iloc[0], 1.0)
        self.assertAlmostEqual(flag_1.loc["2022-12-25"].iloc[0], 1.0)
        self.assertAlmostEqual(flag_1.loc["2022-12-13"].iloc[0], 0.0)

        flag_2 = holiday_flag(input_dates, country="US", encode_holiday_type=True, holidays_subdiv=None)
        self.assertAlmostEqual(flag_2.loc["2022-12-25", 'Christmas Day'], 1.0)
        self.assertAlmostEqual(flag_2.loc["2022-12-13", "Christmas Day"], 0.0)

        df = load_daily(long=False)
        hflag = holiday_flag(df.index, country="US")
        test_result = hflag[(hflag.index.month == 7) & (hflag.index.day == 4)].mean()
        self.assertEqual(test_result.iloc[0], 1)


class TestSeasonal(unittest.TestCase):

    def test_date_part(self):
        print("Starting test_holiday_flag")
        input_dates = pd.date_range("2021-01-01", "2023-01-01", freq='D')
        date_part_df = date_part(
            input_dates, method=['simple_binarized', 365.25, 'quarter'],
            set_index=True, holiday_country=["US"], holiday_countries_used=True
        )
        # assert all numeric and not NaN
        self.assertEqual(np.sum(date_part_df.isnull().to_numpy()), 0, msg="date part generating NaN")
        self.assertEqual(date_part_df.select_dtypes("number").shape, date_part_df.shape)
        # assert column names match expected
        expected_cols = [
            'day',
             'weekend',
             'epoch',
             'month_1',
             'month_2',
             'month_3',
             'month_4',
             'month_5',
             'month_6',
             'month_7',
             'month_8',
             'month_9',
             'month_10',
             'month_11',
             'month_12',
             'weekday_0',
             'weekday_1',
             'weekday_2',
             'weekday_3',
             'weekday_4',
             'weekday_5',
             'weekday_6',
             'seasonality365.25_0',
             'seasonality365.25_1',
             'seasonality365.25_2',
             'seasonality365.25_3',
             'seasonality365.25_4',
             'seasonality365.25_5',
             'seasonality365.25_6',
             'seasonality365.25_7',
             'seasonality365.25_8',
             'seasonality365.25_9',
             'seasonality365.25_10',
             'seasonality365.25_11',
             'quarter_1',
             'quarter_2',
             'quarter_3',
             'quarter_4',
             'Christmas Day',
             'Christmas Day (Observed)',
             'Columbus Day',
             'Independence Day',
             'Independence Day (Observed)',
             'Juneteenth National Independence Day',
             'Juneteenth National Independence Day (Observed)',
             'Labor Day',
             'Martin Luther King Jr. Day',
             'Memorial Day',
             "New Year's Day",
             "New Year's Day (Observed)",
             'Thanksgiving',
             'Veterans Day',
             'Veterans Day (Observed)',
             "Washington's Birthday",
         ]
        self.assertCountEqual(date_part_df.columns.tolist(), expected_cols)
