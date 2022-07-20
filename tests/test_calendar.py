# -*- coding: utf-8 -*-
"""Test calendars
"""
import unittest
import pandas as pd
from autots.tools.calendar import gregorian_to_chinese, gregorian_to_islamic, gregorian_to_hebrew
from autots.tools.lunar import moon_phase


class TestCalendar(unittest.TestCase):

    def test_chinese(self):
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
        self.assertAlmostEqual(moon_phase(pd.Timestamp("2022-07-18")), 0.686, 3)
        self.assertAlmostEqual(moon_phase(pd.Timestamp("1995-11-07")), 0.998, 3)
        self.assertAlmostEqual(moon_phase(pd.Timestamp("2035-02-08")), 0.002, 3)

    def test_hebrew(self):
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
