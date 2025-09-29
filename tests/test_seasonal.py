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

    def test_create_changepoint_features_basic(self):
        """Test basic changepoint feature creation (legacy method)."""
        from autots.tools.seasonal import create_changepoint_features
        
        DTindex = pd.date_range("2020-01-01", "2021-01-01", freq="D")
        features = create_changepoint_features(DTindex, method='basic')
        
        self.assertEqual(features.shape[0], len(DTindex))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all(features.columns.str.contains('changepoint')))

    def test_create_changepoint_features_pelt(self):
        """Test PELT changepoint detection."""
        from autots.tools.seasonal import create_changepoint_features
        
        DTindex = pd.date_range("2020-01-01", "2020-07-01", freq="D")
        # Create simple synthetic data with a level shift
        data = np.concatenate([np.ones(100) * 10, np.ones(82) * 15])
        
        features = create_changepoint_features(
            DTindex, 
            method='pelt', 
            params={'penalty': 10},
            data=data
        )
        
        self.assertEqual(features.shape[0], len(DTindex))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all(features.columns.str.contains('pelt_changepoint')))

    def test_create_changepoint_features_l1(self):
        """Test L1 trend filtering changepoint detection."""
        from autots.tools.seasonal import create_changepoint_features
        
        DTindex = pd.date_range("2020-01-01", "2020-04-01", freq="D")
        # Simple synthetic data
        data = np.random.normal(10, 1, len(DTindex))
        
        features = create_changepoint_features(
            DTindex,
            method='l1_fused_lasso',
            params={'lambda_reg': 1.0},
            data=data
        )
        
        self.assertEqual(features.shape[0], len(DTindex))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(all(features.columns.str.contains('l1_fused_lasso_changepoint')))

    def test_changepoint_detector_basic(self):
        """Test ChangePointDetector class basic functionality."""
        from autots.tools.seasonal import ChangePointDetector
        
        # Create simple test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = np.concatenate([np.ones(50) * 10, np.ones(50) * 15])
        df = pd.DataFrame({'series1': data}, index=dates)
        
        detector = ChangePointDetector(method='pelt', aggregate_method='individual')
        detector.detect(df)
        
        self.assertIsNotNone(detector.changepoints_)
        self.assertIn('series1', detector.changepoints_)
        self.assertEqual(detector.df.shape, df.shape)

    def test_changepoint_detector_features(self):
        """Test ChangePointDetector feature creation."""
        from autots.tools.seasonal import ChangePointDetector
        
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'series1': np.random.normal(10, 1, 50),
            'series2': np.random.normal(12, 1, 50)
        }, index=dates)
        
        detector = ChangePointDetector(method='pelt', aggregate_method='individual')
        detector.detect(df)
        
        features = detector.create_features(forecast_length=10)
        self.assertEqual(features.shape[0], 60)  # 50 + 10 forecast
        self.assertGreater(features.shape[1], 0)

    def test_find_market_changepoints(self):
        """Test market changepoint detection."""
        from autots.tools.seasonal import find_market_changepoints_multivariate
        
        dates = pd.date_range('2020-01-01', periods=80, freq='D')
        # Create data with similar changepoints
        data1 = np.concatenate([np.ones(40) * 10, np.ones(40) * 15])
        data2 = np.concatenate([np.ones(38) * 5, np.ones(42) * 12])
        df = pd.DataFrame({'series1': data1, 'series2': data2}, index=dates)
        
        results = find_market_changepoints_multivariate(
            df,
            detector_params={'method': 'pelt', 'aggregate_method': 'individual'},
            clustering_method='agreement',
            min_series_agreement=0.5
        )
        
        self.assertIn('market_changepoints', results)
        self.assertIn('individual_changepoints', results)
        self.assertIn('detector', results)
        self.assertIsInstance(results['market_changepoints'], np.ndarray)
        # Test that individual changepoints were detected for each series
        self.assertIn('series1', results['individual_changepoints'])
        self.assertIn('series2', results['individual_changepoints'])
