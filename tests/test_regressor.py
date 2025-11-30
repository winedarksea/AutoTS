# -*- coding: utf-8 -*-
"""Tests."""

import unittest
from autots import AutoTS, create_lagged_regressor, load_daily, create_regressor
from autots.tools.regressor import create_fft_features


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
            fill_na='ffill',
        )

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
            fill_na='zero',
        )

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
            fill_na='mean',
        )

        self.assertEqual(regr.shape[1], 2)
        self.assertEqual(fcst.shape[0], forecast_length)
        self.assertFalse(regr.isna().any().any())
        self.assertFalse(fcst.isna().any().any())
        self.assertTrue((df.index == regr.index).all())


class FutureRegressorAlignmentTest(unittest.TestCase):
    def test_future_regressor_alignment(self):
        forecast_length = 5
        df = load_daily(long=False).iloc[:50]
        df = df.drop(df.index[5])
        reg_df = df[[df.columns[0]]].copy()
        reg_train, _ = create_lagged_regressor(
            reg_df,
            forecast_length=forecast_length,
            frequency='infer',
            scale=False,
            summarize=None,
            backfill='bfill',
            fill_na='pchip',
        )
        reg_train = reg_train.iloc[forecast_length:]
        df_train = df.iloc[forecast_length:]
        model = AutoTS(
            forecast_length=forecast_length,
            max_generations=1,
            num_validations=1,
            validation_method='backwards',
            model_list=['LastValueNaive'],
            transformer_list=[],
            verbose=0,
        )
        model = model.fit(df_train, future_regressor=reg_train)
        self.assertEqual(
            model.future_regressor_train.shape[0],
            model.df_wide_numeric.shape[0],
        )


class TestFFTFeatures(unittest.TestCase):
    def test_create_fft_features(self):
        """Test basic FFT feature creation"""
        print("Starting test_create_fft_features")
        df = load_daily(long=False).iloc[:100]
        forecast_length = 10

        fft_train, fft_fcst = create_fft_features(
            df, forecast_length=forecast_length, n_harmonics=5, detrend='linear'
        )

        # Check shapes
        self.assertEqual(fft_train.shape[0], df.shape[0])
        self.assertEqual(fft_fcst.shape[0], forecast_length)
        # Each harmonic has 2 components (real and imaginary)
        self.assertEqual(fft_train.shape[1], fft_fcst.shape[1])
        self.assertGreater(fft_train.shape[1], 0)

        # Check no NaNs
        self.assertFalse(fft_train.isna().any().any())
        self.assertFalse(fft_fcst.isna().any().any())

        # Check column names
        self.assertTrue(all('fft_harmonic' in str(col) for col in fft_train.columns))

    def test_create_regressor_with_fft(self):
        """Test create_regressor with FFT features enabled"""
        print("Starting test_create_regressor_with_fft")
        df = load_daily(long=False).iloc[:100]
        forecast_length = 7

        regr_train, regr_fcst = create_regressor(
            df,
            forecast_length=forecast_length,
            summarize="mean",
            datepart_method="simple",
            holiday_countries=None,
            holiday_detector_params=None,
            fft_n_harmonics=8,
            fft_detrend='linear',
            backfill='bfill',
            fill_na='zero',
        )

        # Check basic shape requirements
        self.assertEqual(regr_train.shape[0], df.shape[0])
        self.assertEqual(regr_fcst.shape[0], forecast_length)
        self.assertFalse(regr_train.isna().any().any())
        self.assertFalse(regr_fcst.isna().any().any())

        # Check that FFT features were added
        fft_cols = [col for col in regr_train.columns if 'fft_harmonic' in str(col)]
        self.assertGreater(len(fft_cols), 0, "FFT features should be present")

        # Verify same columns in both train and forecast
        self.assertTrue((regr_train.columns == regr_fcst.columns).all())

    def test_fft_different_detrend_methods(self):
        """Test FFT with different detrending methods"""
        print("Starting test_fft_different_detrend_methods")
        df = load_daily(long=False).iloc[:50]
        forecast_length = 5

        for detrend in [None, 'linear', 'quadratic']:
            with self.subTest(detrend=detrend):
                fft_train, fft_fcst = create_fft_features(
                    df, forecast_length=forecast_length, n_harmonics=3, detrend=detrend
                )

                self.assertEqual(fft_train.shape[0], df.shape[0])
                self.assertEqual(fft_fcst.shape[0], forecast_length)
                self.assertFalse(fft_train.isna().any().any())
