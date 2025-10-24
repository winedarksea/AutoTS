# -*- coding: utf-8 -*-
"""
Comprehensive test suite for SyntheticDailyGenerator

Created on Mon Sep 30 2024

@author: Colin
"""
import unittest
import os
import tempfile
import json
import numpy as np
import pandas as pd
from autots.datasets import SyntheticDailyGenerator, generate_synthetic_daily_data


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test suite for synthetic data generation."""
    
    def test_basic_generation(self):
        """Test basic data generation."""
        gen = generate_synthetic_daily_data(n_days=365, n_series=5, random_seed=42)
        data = gen.get_data()
        
        self.assertEqual(data.shape, (365, 5), "Data shape mismatch")
        self.assertIsInstance(data.index, pd.DatetimeIndex, "Index should be DatetimeIndex")
    
    def test_business_day_series(self):
        """Test business day series has NaN on weekends."""
        gen = generate_synthetic_daily_data(n_days=365, n_series=3, random_seed=42)
        data = gen.get_data()
        
        weekend_mask = data.index.dayofweek >= 5
        weekend_values = data['series_0'][weekend_mask]
        
        self.assertTrue(weekend_values.isna().all(), "series_0 should have NaN on all weekends")
    
    def test_scale_differences(self):
        """Test that series have different scales."""
        gen = generate_synthetic_daily_data(n_days=365, n_series=6, random_seed=42)
        scales = gen.get_series_scales()
        
        scale_values = list(scales.values())
        self.assertIn(10.0, scale_values, "Should have series with 10x scale")
        self.assertIn(1.0, scale_values, "Should have series with 1x scale")
    
    def test_trend_changepoints(self):
        """Test trend changepoint generation and labeling."""
        gen = generate_synthetic_daily_data(n_days=1095, n_series=5, random_seed=42)
        
        all_cp = gen.get_trend_changepoints()
        self.assertEqual(len(all_cp), 5, "Should have changepoints for all series")
        
        series_0_cp = gen.get_trend_changepoints('series_0')
        self.assertIsInstance(series_0_cp, list, "Should return list of changepoints")
        
        if len(series_0_cp) > 0:
            date, old_slope, new_slope = series_0_cp[0]
            self.assertIsInstance(date, pd.Timestamp, "Date should be Timestamp")
            self.assertIsInstance(old_slope, (int, float, str), "Slope should be numeric or label")
    
    def test_level_shifts(self):
        """Test level shift generation."""
        gen = generate_synthetic_daily_data(n_days=1095, n_series=5, random_seed=123)
        
        # series_4 should have no level shifts
        series_4_shifts = gen.get_level_shifts('series_4')
        self.assertEqual(len(series_4_shifts), 0, "series_4 should have no level shifts")
    
    def test_anomalies(self):
        """Test anomaly generation and magnitude."""
        gen = generate_synthetic_daily_data(
            n_days=730, 
            n_series=5, 
            random_seed=42,
            anomaly_freq=0.1
        )
        
        all_anomalies = gen.get_anomalies()
        total_anomalies = sum(len(a) for a in all_anomalies.values())
        self.assertGreater(total_anomalies, 0, "Should have some anomalies")
        
        # Check anomaly structure
        series_0_anoms = gen.get_anomalies('series_0')
        if len(series_0_anoms) > 0:
            date, magnitude, anom_type, duration, is_shared = series_0_anoms[0]
            self.assertIsInstance(date, pd.Timestamp, "Date should be Timestamp")
            self.assertIn(anom_type, ['point_outlier', 'persistent_shift', 'impulse_decay', 'ramp_down', 'transient_change'], 
                         "Invalid anomaly type")
            self.assertGreaterEqual(duration, 1, "Duration should be at least 1")
            self.assertIsInstance(is_shared, bool, "is_shared flag should be boolean")
    
    def test_seasonality(self):
        """Test seasonality generation."""
        gen = generate_synthetic_daily_data(n_days=365, n_series=5, random_seed=42)
        
        # series_2 should have time-varying seasonality
        season_cp_2 = gen.get_seasonality_changepoints('series_2')
        self.assertGreaterEqual(len(season_cp_2), 2, 
                               "series_2 should have seasonality markers")
        
        # series_3 should have seasonality changepoints
        season_cp_3 = gen.get_seasonality_changepoints('series_3')
        self.assertGreaterEqual(len(season_cp_3), 1, 
                               "series_3 should have seasonality changepoints")
    
    def test_holidays(self):
        """Test holiday generation."""
        gen = generate_synthetic_daily_data(n_days=1095, n_series=8, random_seed=42)
        
        # Check common holidays (Christmas)
        holidays_0 = gen.get_holiday_impacts('series_0')
        christmas_dates = [d for d in holidays_0.keys() if d.month == 12 and d.day == 25]
        self.assertGreaterEqual(len(christmas_dates), 1, "Should have at least one Christmas")
        
        # Check Lunar holidays (series_5)
        holidays_5 = gen.get_holiday_impacts('series_5')
        self.assertGreater(len(holidays_5), 0, "series_5 should have Lunar holiday impacts")
        chinese_dates_sorted = sorted(holidays_5.keys())
        longest_cny_streak = 0
        current_streak = 0
        last_date = None
        for date in chinese_dates_sorted:
            if last_date is not None and (date - last_date).days == 1:
                current_streak += 1
            else:
                current_streak = 1
            longest_cny_streak = max(longest_cny_streak, current_streak)
            last_date = date
        self.assertGreaterEqual(
            longest_cny_streak,
            5,
            "Chinese New Year impacts should span multiple consecutive days",
        )
        
        # Check Ramadan holidays (series_6)
        holidays_6 = gen.get_holiday_impacts('series_6')
        self.assertGreater(len(holidays_6), 0, "series_6 should have Ramadan holiday impacts")
        ramadan_dates_sorted = sorted(holidays_6.keys())
        longest_ramadan_streak = 0
        current_streak = 0
        last_date = None
        for date in ramadan_dates_sorted:
            if last_date is not None and (date - last_date).days == 1:
                current_streak += 1
            else:
                current_streak = 1
            longest_ramadan_streak = max(longest_ramadan_streak, current_streak)
            last_date = date
        self.assertGreaterEqual(
            longest_ramadan_streak,
            20,
            "Ramadan impacts should cover an extended run of days",
        )

        # Ensure randomly generated holiday patterns exist
        config = gen.get_holiday_config()
        dom_keys = [k for k in config.keys() if k.startswith('dom_')]
        wkdom_keys = [k for k in config.keys() if k.startswith('wkdom_')]
        self.assertGreater(len(dom_keys), 0, "Should generate random day-of-month holidays")
        self.assertGreater(len(wkdom_keys), 0, "Should generate random weekday-of-month holidays")

        all_holiday_dates = list(gen.holiday_dates.values())

        dom_match_found = False
        for dom_key in dom_keys:
            month = int(dom_key.split('_')[1])
            day = int(dom_key.split('_')[2])
            for series_dates in all_holiday_dates:
                if any(date.month == month and date.day == day for date in series_dates):
                    dom_match_found = True
                    break
            if dom_match_found:
                break
        self.assertTrue(dom_match_found, "Random dom_* holiday should appear in generated dates")

        wkdom_match_found = False
        for wkdom_key in wkdom_keys:
            month = int(wkdom_key.split('_')[1])
            week = int(wkdom_key.split('_')[2])
            weekday = int(wkdom_key.split('_')[3])
            for series_dates in all_holiday_dates:
                for date in series_dates:
                    if date.month == month and date.dayofweek == weekday:
                        week_of_month = (date.day - 1) // 7 + 1
                        if week_of_month == week:
                            wkdom_match_found = True
                            break
                if wkdom_match_found:
                    break
            if wkdom_match_found:
                break
        self.assertTrue(wkdom_match_found, "Random wkdom_* holiday should appear in generated dates")
    
    def test_regressors(self):
        """Test regressor generation."""
        gen = generate_synthetic_daily_data(
            n_days=365, 
            n_series=5, 
            random_seed=42,
            include_regressors=True
        )
        
        regressors = gen.get_regressors()
        self.assertIsNotNone(regressors, "Should have regressors")
        self.assertEqual(regressors.shape, (365, 3), "Should have 3 regressor columns")
        self.assertIn('promotion', regressors.columns, "Should have promotion regressor")
        self.assertIn('temperature', regressors.columns, "Should have temperature regressor")
        self.assertIn('precipitation', regressors.columns, "Should have precipitation regressor")
    
    def test_noise_changepoints(self):
        """Test noise distribution changepoints."""
        gen = generate_synthetic_daily_data(n_days=1095, n_series=5, random_seed=42)
        
        noise_cp = gen.get_noise_changepoints('series_0')
        self.assertIsInstance(noise_cp, list, "Should return list of noise changepoints")
    
    def test_variance_regimes(self):
        """Test GARCH-like variance regimes."""
        gen = generate_synthetic_daily_data(n_days=1095, n_series=8, random_seed=42)
        
        # series_7 should have variance regimes
        noise_cp_7 = gen.get_noise_changepoints('series_7')
        self.assertIsInstance(noise_cp_7, list, "series_7 should have variance regime info")
    
    def test_autocorrelated_noise(self):
        """Test AR(1) autocorrelated noise."""
        gen = generate_synthetic_daily_data(n_days=5000, n_series=9, random_seed=42)
        
        # series_8 should have AR noise
        components = gen.get_components('series_8')
        ar_noise = components['noise']
        
        # Get AR parameters
        noise_cp_8 = gen.get_noise_changepoints('series_8')
        self.assertIsInstance(noise_cp_8, list, "series_8 should have AR noise info")
        self.assertGreater(len(noise_cp_8), 0, "Should have at least one noise record")
        
        # Check AR parameters exist
        ar_params = noise_cp_8[0][2]
        self.assertIn('phi', ar_params, "Should have phi parameter")
        self.assertIn('std', ar_params, "Should have std parameter")
        
        phi = ar_params['phi']
        self.assertGreater(phi, 0.7, "phi should be >= 0.7 for smooth noise")
        self.assertLess(phi, 0.98, "phi should be < 0.98 for stationarity")
        
        # Test autocorrelation exists (AR noise should be correlated)
        lag1_corr = np.corrcoef(ar_noise[:-1], ar_noise[1:])[0, 1]
        self.assertGreater(lag1_corr, 0.5, "AR noise should have strong positive autocorrelation")
        
        # Test stationarity: variance should be similar in different segments
        var_first = np.var(ar_noise[:2500])
        var_second = np.var(ar_noise[2500:])
        var_ratio = max(var_first, var_second) / min(var_first, var_second)
        self.assertLess(var_ratio, 1.5, "Variance should be relatively stable (stationary)")
    
    def test_multiplicative_seasonality(self):
        """Test multiplicative seasonality with autocorrelated noise."""
        gen = generate_synthetic_daily_data(n_days=2555, n_series=10, random_seed=42)
        
        # series_9 should have multiplicative seasonality
        series_types = gen.series_types
        self.assertEqual(series_types.get('series_9'), 'multiplicative_seasonality',
                        "series_9 should be multiplicative_seasonality type")
        
        # Get components
        components = gen.get_components('series_9')
        data = gen.get_data()
        series_9_data = data['series_9'].values
        
        trend = components['trend']
        level_shift = components['level_shift']
        seasonality = components['seasonality']
        noise = components['noise']
        
        # Verify autocorrelated noise is used
        noise_cp_9 = gen.get_noise_changepoints('series_9')
        self.assertGreater(len(noise_cp_9), 0, "Should have noise info")
        ar_params = noise_cp_9[0][2]
        self.assertIn('phi', ar_params, "Should use AR noise for multiplicative seasonality")
        
        # Test that seasonality is multiplicative by checking if it scales with trend
        # In multiplicative model: signal = (trend + level_shift) * (1 + seasonality_factor)
        # The seasonal amplitude should scale with the base signal level
        
        # Get a period with positive trend
        mid_point = len(series_9_data) // 2
        window = 365  # One year window
        
        early_period = series_9_data[100:100+window]
        late_period = series_9_data[mid_point:mid_point+window]
        
        # Remove NaN values if any
        early_period = early_period[~np.isnan(early_period)]
        late_period = late_period[~np.isnan(late_period)]
        
        # Calculate seasonal amplitude (using std as proxy)
        # For multiplicative: amplitude increases with level
        # For additive: amplitude stays constant
        early_std = np.std(early_period)
        late_std = np.std(late_period)
        
        # Check if the trend has increased
        early_trend_mean = np.mean(trend[100:100+window])
        late_trend_mean = np.mean(trend[mid_point:mid_point+window])
        
        # If trend increased, seasonal variance should also increase (multiplicative)
        if late_trend_mean > early_trend_mean * 1.1:  # At least 10% increase in trend
            # For multiplicative seasonality, later std should be larger
            self.assertGreater(late_std / early_std, 0.9, 
                             "Multiplicative seasonality should have amplitude that scales with trend")
    
    def test_component_decomposition(self):
        """Test that components sum to total."""
        gen = generate_synthetic_daily_data(n_days=365, n_series=3, random_seed=42)
        
        data = gen.get_data()
        components = gen.get_components('series_1')
        
        component_sum = sum(components.values())
        actual_series = data['series_1'].values
        
        self.assertTrue(np.allclose(component_sum, actual_series, equal_nan=True),
                       "Components should sum to total series")
    
    def test_all_labels_access(self):
        """Test comprehensive label access."""
        gen = generate_synthetic_daily_data(n_days=730, n_series=5, random_seed=42)
        
        all_labels = gen.get_all_labels('series_0')
        
        required_keys = [
            'trend_changepoints',
            'level_shifts',
            'anomalies',
            'holiday_impacts',
            'noise_changepoints',
            'seasonality_changepoints',
            'noise_to_signal_ratio',
            'series_scale'
        ]
        
        for key in required_keys:
            self.assertIn(key, all_labels, f"Missing key: {key}")
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = generate_synthetic_daily_data(n_days=365, n_series=3, random_seed=42)
        gen2 = generate_synthetic_daily_data(n_days=365, n_series=3, random_seed=42)
        
        data1 = gen1.get_data()
        data2 = gen2.get_data()
        
        self.assertTrue(np.allclose(data1.values, data2.values, equal_nan=True),
                       "Same seed should produce same data")
    
    def test_saturating_trend(self):
        """Test saturating trend for series_1."""
        gen = generate_synthetic_daily_data(n_days=1095, n_series=5, random_seed=42)
        
        trend_cp_1 = gen.get_trend_changepoints('series_1')
        
        # Should have saturation-related changepoints
        cp_types = [str(cp[1]) for cp in trend_cp_1 if isinstance(cp[1], str)]
        has_saturation_marker = any('saturation' in t or 'quadratic' in t for t in cp_types)
        
        self.assertTrue(has_saturation_marker, "series_1 should have saturation markers")
    
    def test_noise_to_signal_ratios(self):
        """Test noise-to-signal ratio storage."""
        gen = generate_synthetic_daily_data(n_days=365, n_series=5, random_seed=42)
        
        ratios = gen.get_noise_to_signal_ratios()
        self.assertEqual(len(ratios), 5, "Should have ratios for all series")
        
        for series_name, ratio in ratios.items():
            self.assertIsInstance(ratio, float, "Ratio should be float")
            self.assertGreater(ratio, 0, "Ratio should be positive")
    
    def test_csv_export(self):
        """Test CSV export functionality."""
        gen = generate_synthetic_daily_data(n_days=100, n_series=3, random_seed=42)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            gen.to_csv(temp_file)
            self.assertTrue(os.path.exists(temp_file), "CSV file should be created")
            
            # Read back and verify
            df = pd.read_csv(temp_file, index_col=0, parse_dates=True)
            self.assertEqual(df.shape[0], 100, "Should have 100 rows")
            self.assertEqual(df.shape[1], 3, "Should have 3 columns")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_plot_method(self):
        """Test plotting functionality."""
        gen = generate_synthetic_daily_data(n_days=365, n_series=3, random_seed=42)
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.png') as f:
                temp_file = f.name
            
            try:
                # Test plotting specific series
                fig = gen.plot(series_name='series_0', save_path=temp_file, show=False)
                self.assertTrue(os.path.exists(temp_file), "Plot file should be created")
                self.assertIsNotNone(fig, "Should return figure object")
                
                # Test random selection
                fig2 = gen.plot(show=False)
                self.assertIsNotNone(fig2, "Random plot should return figure")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except ImportError:
            self.skipTest("matplotlib not available")
    
    def test_slope_changes_are_meaningful(self):
        """Test that trend changepoints have meaningful slope changes."""
        gen = generate_synthetic_daily_data(
            n_days=1095,  # 3 years
            n_series=10, 
            random_seed=42,
            trend_changepoint_freq=1.0  # Higher frequency to get more changepoints
        )
        
        all_changepoints = gen.get_trend_changepoints()
        scales = gen.get_series_scales()
        
        for series_name in all_changepoints.keys():
            changepoints = all_changepoints[series_name]
            scale = scales[series_name]
            
            if not changepoints:
                continue
            
            # Filter out non-numeric slope changes (saturating trends have string labels)
            numeric_changepoints = [
                cp for cp in changepoints 
                if isinstance(cp[1], (int, float)) and isinstance(cp[2], (int, float))
            ]
            
            if not numeric_changepoints:
                # Non-linear trend (saturating/quadratic) - skip validation
                continue
            
            # Validate each changepoint has meaningful slope change
            min_threshold = 0.008 * scale * 0.5  # Allow subtle changes at 50% of base threshold
            
            for date, old_slope, new_slope in numeric_changepoints:
                slope_diff = abs(new_slope - old_slope)
                
                self.assertGreaterEqual(
                    slope_diff, 
                    min_threshold,
                    f"{series_name} at {date}: slope change {slope_diff:.6f} is too subtle "
                    f"(min threshold: {min_threshold:.6f}). Old slope: {old_slope:.6f}, "
                    f"new slope: {new_slope:.6f}"
                )
    
    def test_shared_events(self):
        """Test that some events are shared across series."""
        gen = generate_synthetic_daily_data(
            n_days=1095, 
            n_series=10, 
            random_seed=123,
            anomaly_freq=0.1,
            level_shift_freq=0.5,
            shared_anomaly_prob=0.5,
            shared_level_shift_prob=0.5,
        )
        
        # Check for shared anomalies
        all_anomalies = gen.get_anomalies()
        anomaly_dates = {}
        for series, anomalies in all_anomalies.items():
            for anom in anomalies:
                if anom[4]: # is_shared
                    date = anom[0]
                    if date not in anomaly_dates:
                        anomaly_dates[date] = []
                    anomaly_dates[date].append(series)
        
        shared_anomaly_found = any(len(series_list) > 1 for series_list in anomaly_dates.values())
        self.assertTrue(shared_anomaly_found, "Should find at least one shared anomaly")

        # Check for shared level shifts
        all_shifts = gen.get_level_shifts()
        shift_dates = {}
        for series, shifts in all_shifts.items():
            for shift in shifts:
                if shift[3]: # is_shared
                    date = shift[0]
                    if date not in shift_dates:
                        shift_dates[date] = []
                    shift_dates[date].append(series)

        shared_shift_found = any(len(series_list) > 1 for series_list in shift_dates.values())
        self.assertTrue(shared_shift_found, "Should find at least one shared level shift")
    
    def test_anomaly_types_parameter(self):
        """Test that anomaly_types parameter filters anomaly generation."""
        # Test with only point_outlier anomalies
        gen = generate_synthetic_daily_data(
            n_days=730,
            n_series=5,
            random_seed=42,
            anomaly_freq=0.1,
            anomaly_types=['point_outlier']
        )
        
        all_anomalies = gen.get_anomalies()
        anomaly_types_found = set()
        for series_name, anomaly_list in all_anomalies.items():
            for anomaly in anomaly_list:
                anomaly_types_found.add(anomaly[2])  # Type is the 3rd element
        
        # Should only have point_outlier (or no anomalies if none were generated)
        if len(anomaly_types_found) > 0:
            self.assertEqual(anomaly_types_found, {'point_outlier'}, 
                           "Should only generate point_outlier anomalies")
        
        # Test with multiple specific types
        gen2 = generate_synthetic_daily_data(
            n_days=730,
            n_series=5,
            random_seed=123,
            anomaly_freq=0.15,
            anomaly_types=['impulse_decay', 'linear_decay']
        )
        
        all_anomalies2 = gen2.get_anomalies()
        anomaly_types_found2 = set()
        for series_name, anomaly_list in all_anomalies2.items():
            for anomaly in anomaly_list:
                anomaly_types_found2.add(anomaly[2])
        
        if len(anomaly_types_found2) > 0:
            self.assertTrue(anomaly_types_found2.issubset({'impulse_decay', 'linear_decay'}),
                          "Should only generate specified anomaly types")
        
        # Test invalid anomaly type raises error
        with self.assertRaises(ValueError):
            generate_synthetic_daily_data(
                n_days=365,
                n_series=3,
                random_seed=42,
                anomaly_types=['invalid_type']
            )
    
    def test_disable_holiday_splash_parameter(self):
        """Test that disable_holiday_splash parameter controls splash effects."""
        # Test with splash disabled
        gen_no_splash = generate_synthetic_daily_data(
            n_days=730,
            n_series=10,
            random_seed=42,
            disable_holiday_splash=True
        )
        
        holiday_config = gen_no_splash.get_holiday_config()
        
        # All holidays should have splash and bridge disabled
        for holiday_name, config in holiday_config.items():
            self.assertFalse(config['has_splash'], 
                           f"{holiday_name} should have splash disabled")
            self.assertFalse(config['has_bridge'], 
                           f"{holiday_name} should have bridge disabled")
        
        # Check that lunar holidays series has minimal splash impacts
        if 'series_5' in gen_no_splash.series_types:
            splash_impacts = gen_no_splash.get_holiday_splash_impacts('series_5')
            self.assertEqual(len(splash_impacts), 0, 
                           "Lunar holidays should have no splash impacts when disabled")
        
        # Test with splash enabled (default behavior)
        gen_with_splash = generate_synthetic_daily_data(
            n_days=730,
            n_series=10,
            random_seed=42,
            disable_holiday_splash=False
        )
        
        holiday_config2 = gen_with_splash.get_holiday_config()
        
        # At least some holidays should have splash or bridge enabled
        some_enabled = any(config['has_splash'] or config['has_bridge'] 
                          for config in holiday_config2.values())
        self.assertTrue(some_enabled, 
                       "When splash is enabled, some holidays should have splash/bridge effects")


if __name__ == '__main__':
    unittest.main()
