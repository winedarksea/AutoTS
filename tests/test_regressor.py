# -*- coding: utf-8 -*-
"""Tests."""

import unittest
from autots import create_lagged_regressor, load_daily, create_regressor


class test_create_lagged_regressor(unittest.TestCase):
    def test_create_regressor(self):
        print("Starting test_create_regressor")
        df = load_daily(long=False)
        forecast_length = 5
        regr, fcst = create_lagged_regressor(
            df,
            forecast_length=forecast_length,
            summarize=None,
            backfill='bfill',
            fill_na='ffill')

        self.assertEqual(regr.shape, df.shape)
        self.assertEqual(fcst.shape[0], forecast_length)
        self.assertFalse(regr.isna().any().any())
        self.assertFalse(fcst.isna().any().any())
        self.assertTrue((df.index == regr.index).all())

        regr, fcst = create_regressor(
            df,
            forecast_length=forecast_length,
            summarize="auto",
            datepart_method="recurring",
            holiday_countries=["UK", "US"],
            backfill='ffill',
            fill_na='zero')

        self.assertEqual(regr.shape[0], df.shape[0])
        self.assertEqual(fcst.shape[0], forecast_length)
        self.assertFalse(regr.isna().any().any())
        self.assertFalse(fcst.isna().any().any())
        self.assertTrue((df.index == regr.index).all())

        regr, fcst = create_lagged_regressor(
            df,
            forecast_length=forecast_length,
            summarize="mean+std",
            backfill='ETS',
            fill_na='mean')

        self.assertEqual(regr.shape[1], 2)
        self.assertEqual(fcst.shape[0], forecast_length)
        self.assertFalse(regr.isna().any().any())
        self.assertFalse(fcst.isna().any().any())
        self.assertTrue((df.index == regr.index).all())
