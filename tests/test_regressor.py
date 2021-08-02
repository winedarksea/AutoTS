# -*- coding: utf-8 -*-
"""Tests."""

import unittest
from autots import create_lagged_regressor, load_daily


class test_create_lagged_regressor(unittest.TestCase):
    def test_create_regressor(self):
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
        self.assertTrue((fcst.reset_index(drop=True) == df.tail(forecast_length).reset_index(drop=True)).all().all())

        regr, fcst = create_lagged_regressor(
            df,
            forecast_length=forecast_length,
            summarize=None,
            backfill='DatepartRegression',
            fill_na='zero')

        self.assertEqual(regr.shape, df.shape)
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
