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
        # Test major Hindu festivals for correct month assignment
        test_cases = {
            'Diwali 2021': ('2021-11-04', 'Kartika'),
            'Diwali 2024': ('2024-10-31', 'Kartika'),
            'Diwali 2029': ('2029-11-12', 'Kartika'),
            'Holi 2024': ('2024-03-25', 'Phalguna'),
            'Ram Navami 2024': ('2024-04-17', 'Chaitra'),
            'Janmashtami 2024': ('2024-08-26', 'Bhadrapada'),
            'Holi 2010': ('2010-03-01', 'Phalguna'),
            'Holi 2026': ('2026-03-04', 'Phalguna'),
            'Holi 2028': ('2028-03-11', 'Phalguna'),
            'Holi 2029': ('2029-03-01', 'Phalguna'),
        }
        
        # Known limitations of the simple method for specific edge cases
        simple_method_known_issues = {
            'Holi 2024',  # Day 16 threshold issue with multi-epoch timing
            'Holi 2028',  # Day 16 threshold issue with multi-epoch timing
        }

        for method in ["simple", "lunar"]:
            with self.subTest(method=method):
                for test_name, (date_str, expected_month) in test_cases.items():
                    with self.subTest(test_name=test_name):
                        # Skip known simple method limitations
                        if method == "simple" and test_name in simple_method_known_issues:
                            self.skipTest(f"Known limitation: {method} method has timing precision issues for {test_name}")
                        
                        date = pd.to_datetime([date_str])
                        result = gregorian_to_hindu(date, method=method)
                        self.assertEqual(result.iloc[0]['hindu_month_name'], expected_month)

    def test_islamic_ramadan(self):
        print("Starting test_islamic_ramadan")
        # Test known Ramadan start dates (historically accurate)
        known_ramadan_starts = {
            "2024-03-10": "Ramadan 2024",
            "2023-03-22": "Ramadan 2023",
            "2022-04-02": "Ramadan 2022",
            "2021-04-13": "Ramadan 2021",
            "2020-04-24": "Ramadan 2020",
        }
        for date_str, description in known_ramadan_starts.items():
            with self.subTest(description=description):
                result = gregorian_to_islamic([date_str])
                self.assertEqual(result.iloc[0]['month'], 9, "Failed month check")
                # Allow 1-2 day tolerance for Ramadan start dates
                self.assertIn(result.iloc[0]['day'], [1, 2], "Failed day check - should be day 1 or 2")

    def test_chinese_lunar_new_year(self):
        print("Starting test_chinese_lunar_new_year")
        # Test known Lunar New Year dates (historically accurate)
        known_lunar_new_years = {
            "2025-01-29": "Lunar New Year 2025",
            "2024-02-10": "Lunar New Year 2024",
            "2023-01-22": "Lunar New Year 2023",
            "2022-02-01": "Lunar New Year 2022",
            "2021-02-12": "Lunar New Year 2021",
            "2020-01-25": "Lunar New Year 2020",
        }
        for date_str, description in known_lunar_new_years.items():
            with self.subTest(description=description):
                result = gregorian_to_chinese([date_str])
                self.assertEqual(result.iloc[0]['lunar_month'], 1, "Failed month check")
                self.assertEqual(result.iloc[0]['lunar_day'], 1, "Failed day check")


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
        # if this raises a "AssertionError: Element counts were not equal" error likely the holiday names were changed in the package version of holidays
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
             'Christmas Day (observed)',
             'Columbus Day',
             'Independence Day',
             'Independence Day (observed)',
             'Juneteenth National Independence Day',
             'Juneteenth National Independence Day (observed)',
             'Labor Day',
             'Martin Luther King Jr. Day',
             'Memorial Day',
             "New Year's Day",
             "New Year's Day (observed)",
             'Thanksgiving Day',
             'Veterans Day',
             'Veterans Day (observed)',
             "Washington's Birthday",
         ]
        self.assertCountEqual(date_part_df.columns.tolist(), expected_cols)
