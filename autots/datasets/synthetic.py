# -*- coding: utf-8 -*-
"""
Synthetic Daily Data Generator with Labeled Changepoints, Anomalies, and Holidays

@author: winedarksea with Claude Sonnet v4.5

Matching test file in tests/test_synthetic_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from autots.tools.calendar import gregorian_to_chinese, gregorian_to_islamic

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SyntheticDailyGenerator:
    """
    Generate realistic synthetic daily time series data with labeled components.
    
    Creates multivariate time series with:
    - Piecewise linear trends with changepoints
    - Level shifts (instantaneous and ramped)
    - Seasonality (weekly, yearly) with stochastic variation
    - Holiday effects (common and custom) with splash and bridge effects
    - Anomalies with various post-event patterns
    - Noise with regime changes
    - Optional regressor impacts
    - Business day series with weekend NaN
    - Multiple scales across series
    
    All components are labeled and stored for model evaluation.
    
    **Variability Across Series:**
    - Noise levels vary 0.5x-2.0x the base noise_level per series
    - Weekly seasonality strength varies 0.3x-2.5x per series
    - Yearly seasonality strength varies 0.2x-2.0x per series
    - Level shift frequency varies across series (some have none, some have several)
    - This creates a range from subtle, hard-to-detect patterns to very obvious ones
    
    **Event Scaling with Dataset Length:**
    - Events (anomalies, level shifts, etc.) scale appropriately with n_days
    - Short datasets (< 1 year) use probabilistic event generation
    - Longer datasets use Poisson-based event counts
    - Level shifts are rare events, appropriately distributed
    
    Parameters
    ----------
    start_date : str or pd.Timestamp
        Start date for the time series
    n_days : int
        Number of days to generate
    n_series : int
        Number of time series to generate
    random_seed : int
        Random seed for reproducibility
    trend_changepoint_freq : float
        Probability per year of a trend changepoint (default 0.5)
    level_shift_freq : float
        Probability per year of a level shift (default 0.1)
    anomaly_freq : float
        Probability per week of an anomaly (default 0.05)
    weekly_seasonality_strength : float
        Base strength of weekly seasonality (default 1.0)
        Actual per-series strength will vary 0.3x-2.5x this value
    yearly_seasonality_strength : float
        Base strength of yearly seasonality (default 1.0)
        Actual per-series strength will vary 0.2x-2.0x this value
    noise_level : float
        Base noise level (default 0.1, relative to signal)
        Actual per-series level will vary 0.5x-2.0x this value
    include_regressors : bool
        Whether to include regressor effects (default False)
    """
    
    # Human-readable descriptions for series types
    SERIES_TYPE_DESCRIPTIONS = {
        'business_day': 'Business Day (weekend NaN)',
        'saturating_trend': 'Saturating Trend (logistic)',
        'time_varying_seasonality': 'Time-Varying Seasonality',
        'seasonality_changepoints': 'Seasonality Changepoints',
        'no_level_shifts': 'No Level Shifts',
        'lunar_holidays': 'Lunar Holidays',
        'ramadan_holidays': 'Ramadan Holidays',
        'variance_regimes': 'Variance Regimes (GARCH)',
        'autocorrelated_noise': 'Autocorrelated Noise (AR)',
        'multiplicative_seasonality': 'Multiplicative Seasonality (AR noise)',
        'granger_lagged': 'Granger Lagged (7-day lag from Lunar Holidays)',
        'standard': 'Standard',
    }
    
    def __init__(
        self,
        start_date='2015-01-01',
        n_days=2555,  # ~7 years
        n_series=10,
        random_seed=42,
        trend_changepoint_freq=0.5,
        level_shift_freq=0.1,
        anomaly_freq=0.05,
        shared_anomaly_prob=0.2,
        shared_level_shift_prob=0.2,
        weekly_seasonality_strength=1.0,
        yearly_seasonality_strength=1.0,
        noise_level=0.1,
        include_regressors=False,
    ):
        self.start_date = pd.Timestamp(start_date)
        self.n_days = n_days
        self.n_series = n_series
        self.random_seed = random_seed
        self.trend_changepoint_freq = trend_changepoint_freq
        self.level_shift_freq = level_shift_freq
        self.anomaly_freq = anomaly_freq
        self.shared_anomaly_prob = shared_anomaly_prob
        self.shared_level_shift_prob = shared_level_shift_prob
        self.weekly_seasonality_strength = weekly_seasonality_strength
        self.yearly_seasonality_strength = yearly_seasonality_strength
        self.noise_level = noise_level
        self.include_regressors = include_regressors
        
        # Initialize random state
        self.rng = np.random.RandomState(random_seed)
        
        # Create date index
        self.date_index = pd.date_range(
            start=self.start_date, periods=n_days, freq='D'
        )
        
        # Storage for labels and components
        self.trend_changepoints = {}  # {series_id: [(date, old_slope, new_slope), ...]}
        self.level_shifts = {}  # {series_id: [(date, magnitude, type, shared), ...]}
        self.anomalies = {}  # {series_id: [(date, magnitude, type, duration, shared), ...]}
        self.holiday_impacts = {}  # {series_id: {date: impact}}
        self.holiday_dates = {}  # {series_id: [date, ...]} - distinct holiday dates only (not splash days)
        self.holiday_splash_impacts = {}  # {series_id: {date: impact}} - splash/bridge day effects
        self.holiday_config = {}  # {holiday_name: {'has_splash': bool, 'has_bridge': bool}}
        self.noise_changepoints = {}  # {series_id: [(date, old_params, new_params), ...]}
        self.seasonality_changepoints = {}  # {series_id: [(date, description), ...]}
        self.shared_events = {'anomalies': [], 'level_shifts': []}
        self.noise_to_signal_ratios = {}  # {series_id: ratio}
        self.series_noise_levels = {}  # {series_id: noise_level}
        self.series_seasonality_strengths = {}  # {series_id: {'weekly': float, 'yearly': float}}
        self.series_scales = {}  # {series_id: scale_factor}
        self.series_types = {}  # {series_id: series_type}
        self.effect_sizes = {}  # Detailed storage of all effect magnitudes
        self.regressor_impacts = {}  # {series_id: {date: {regressor_name: impact}}}
        self.lagged_influences = {}  # {series_id: {'source': source_series, 'lag': lag_days, 'coefficient': coef}}
        
        # Component storage for analysis
        self.components = {}  # {series_id: {component_name: array}}

        # Precompute custom holiday templates so all series share the same structure
        self.random_dom_holidays = self._init_random_dom_holidays()
        self.random_wkdom_holidays = self._init_random_wkdom_holidays()
        
        # Generate the data
        self.data = None
        self.regressors = None
        self._generate()
    
    def _generate(self):
        """Main generation pipeline."""
        # Initialize data array
        data_arrays = {}
        
        # Generate shared events first
        self._generate_shared_events()
        
        # Generate optional regressors first
        if self.include_regressors:
            self._generate_regressors()
        
        # Generate each series
        # Map series index to type for cleaner assignment
        series_type_map = {
            0: 'business_day', 1: 'saturating_trend', 2: 'time_varying_seasonality',
            3: 'seasonality_changepoints', 4: 'no_level_shifts', 5: 'lunar_holidays',
            6: 'ramadan_holidays', 7: 'variance_regimes', 8: 'autocorrelated_noise',
            9: 'multiplicative_seasonality', 10: 'granger_lagged'
        }
        
        for i in range(self.n_series):
            series_name = f"series_{i}"
            
            # Determine series characteristics
            series_type = series_type_map.get(i, 'standard')
            self.series_types[series_name] = series_type
            
            # Set scale for this series (every 3rd series is 10x larger)
            scale = 10.0 if i % 3 == 0 else 1.0
            self.series_scales[series_name] = scale
            
            # Generate series components
            series_data = self._generate_series(series_name, series_type, scale)
            data_arrays[series_name] = series_data
        
        # Create DataFrame
        self.data = pd.DataFrame(data_arrays, index=self.date_index)
    
    def _generate_shared_events(self):
        """Generate anomalies and level shifts that are shared across multiple series."""
        # Shared Anomalies
        n_weeks = self.n_days / 7
        n_shared_anomalies = int(self.rng.poisson(
            self.anomaly_freq * n_weeks * self.shared_anomaly_prob * self.n_series
        ))
        
        if n_shared_anomalies > 0:
            anomaly_days = self.rng.choice(
                range(int(self.n_days * 0.1), int(self.n_days * 0.9)),
                size=n_shared_anomalies,
                replace=False
            )
            for day in anomaly_days:
                self.shared_events['anomalies'].append(day)

        # Shared Level Shifts - ensure they scale with dataset length
        n_years = self.n_days / 365.25
        
        # Base calculation
        expected_shared_shifts = self.level_shift_freq * n_years * self.shared_level_shift_prob * self.n_series
        
        # For shorter datasets, ensure at least a possibility
        if n_years < 1.0:
            # Use binomial instead of Poisson for rare events in short datasets
            if self.rng.random() < expected_shared_shifts:
                n_shared_level_shifts = 1
            else:
                n_shared_level_shifts = 0
        else:
            n_shared_level_shifts = int(self.rng.poisson(expected_shared_shifts))
        
        if n_shared_level_shifts > 0:
            shift_days = self.rng.choice(
                range(int(self.n_days * 0.2), int(self.n_days * 0.9)),
                size=n_shared_level_shifts,
                replace=False
            )
            for day in shift_days:
                self.shared_events['level_shifts'].append(day)

    def _generate_regressors(self):
        """Generate synthetic regressors (promotion flag, weather)."""
        # Promotion flag: random binary with some clustering
        promotion = np.zeros(self.n_days)
        promo_dates = self.rng.choice(
            self.n_days, 
            size=int(self.n_days * 0.05),  # 5% of days
            replace=False
        )
        promotion[promo_dates] = 1
        
        # Weather: temperature (seasonal) + random variation
        t = np.arange(self.n_days)
        temp_base = 60 + 20 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
        temp = temp_base + self.rng.normal(0, 5, self.n_days)
        
        # Precipitation: random with occasional larger values
        precip = np.abs(self.rng.gamma(0.5, 0.3, self.n_days))
        
        self.regressors = pd.DataFrame({
            'promotion': promotion,
            'temperature': temp,
            'precipitation': precip
        }, index=self.date_index)
    
    def _generate_series(self, series_name, series_type, scale):
        """Generate a single time series with all components."""
        self.components[series_name] = {}
        
        # Assign per-series noise level (varying around base noise_level)
        # Range: 0.5x to 2.0x the base noise level
        noise_multiplier = self.rng.uniform(0.5, 2.0)
        series_noise_level = self.noise_level * noise_multiplier
        self.series_noise_levels[series_name] = series_noise_level
        
        # Assign per-series seasonality strengths
        # Weekly: range from 0.3x to 2.5x base strength (some very subtle, some very strong)
        # Yearly: range from 0.2x to 2.0x base strength
        weekly_mult = self.rng.uniform(0.3, 2.5)
        yearly_mult = self.rng.uniform(0.2, 2.0)
        series_weekly_strength = self.weekly_seasonality_strength * weekly_mult
        series_yearly_strength = self.yearly_seasonality_strength * yearly_mult
        self.series_seasonality_strengths[series_name] = {
            'weekly': series_weekly_strength,
            'yearly': series_yearly_strength
        }
        
        # 1. Generate trend
        trend = self._generate_trend(series_name, series_type, scale)
        self.components[series_name]['trend'] = trend
        
        # 2. Generate level shifts
        if series_type != 'no_level_shifts':
            level_shift = self._generate_level_shifts(series_name, series_type, scale)
        else:
            level_shift = np.zeros(self.n_days)
            self.level_shifts[series_name] = []
        self.components[series_name]['level_shift'] = level_shift
        
        # 3. Generate seasonality
        seasonality = self._generate_seasonality(series_name, series_type, scale)
        self.components[series_name]['seasonality'] = seasonality
        
        # 4. Generate holiday effects
        holidays = self._generate_holiday_effects(series_name, series_type, scale)
        self.components[series_name]['holidays'] = holidays
        
        # 5. Generate noise
        noise = self._generate_noise(series_name, series_type, scale)
        self.components[series_name]['noise'] = noise
        
        # 6. Generate anomalies
        anomalies = self._generate_anomalies(series_name, series_type, scale)
        self.components[series_name]['anomalies'] = anomalies
        
        # 7. Add regressor effects if available
        if self.include_regressors:
            regressor_effect = self._generate_regressor_effects(series_name, scale)
            self.components[series_name]['regressors'] = regressor_effect
        else:
            regressor_effect = 0
        
        # Combine all components
        if series_type == 'multiplicative_seasonality':
            # For multiplicative seasonality: (trend + level_shift) * (1 + seasonality_factor) + noise + anomalies + regressor
            # Normalize seasonality to be a multiplicative factor (centered around 0)
            base_signal = trend + level_shift
            # Ensure base signal is positive for multiplication
            min_signal = np.min(base_signal)
            if min_signal <= 0:
                base_signal = base_signal - min_signal + scale * 10
            
            # Convert additive seasonality to a multiplicative factor.
            # The seasonality component should be a percentage of the base signal.
            # Avoid division by zero for flat trend sections.
            safe_base_signal = np.where(base_signal == 0, 1, base_signal)
            seasonality_factor = seasonality / safe_base_signal
            
            series_data = (
                base_signal * (1 + seasonality_factor) + 
                holidays + noise + anomalies + regressor_effect
            )
        else:
            # Standard additive combination
            series_data = (
                trend + level_shift + seasonality + holidays + 
                noise + anomalies + regressor_effect
            )
        
        # Apply Granger-style lagged influence if this series type requires it
        if series_type == 'granger_lagged':
            lagged_influence = self._apply_lagged_influence(series_name, scale)
            series_data = series_data + lagged_influence
            self.components[series_name]['lagged_influence'] = lagged_influence
        
        # Apply business day mask if needed
        if series_type == 'business_day':
            is_weekend = (self.date_index.dayofweek >= 5)
            series_data = np.where(is_weekend, np.nan, series_data)
        
        return series_data
    
    def _generate_trend(self, series_name, series_type, scale):
        """Generate piecewise linear or nonlinear trends with changepoints."""
        t = np.arange(self.n_days)
        
        if series_type == 'saturating_trend':
            # Nonlinear saturating trend (logistic + piecewise quadratic)
            trend = np.zeros(self.n_days)
            
            # Logistic growth phase
            L = 100 * scale  # Carrying capacity
            k = 0.01  # Growth rate
            x0 = self.n_days * 0.3  # Midpoint
            logistic = L / (1 + np.exp(-k * (t - x0)))
            
            # Add a quadratic segment in middle
            mid_start = int(self.n_days * 0.4)
            mid_end = int(self.n_days * 0.6)
            if mid_start < mid_end:
                t_mid = np.arange(mid_end - mid_start)
                quadratic = -0.0001 * scale * t_mid**2 + 0.05 * scale * t_mid
                logistic[mid_start:mid_end] += quadratic
            
            trend = logistic
            
            # Label saturation points
            self.trend_changepoints[series_name] = [
                (self.date_index[int(x0)], 'saturation_midpoint', k),
                (self.date_index[mid_start], 'quadratic_start', 0),
                (self.date_index[mid_end], 'quadratic_end', 0),
            ]
        else:
            # Piecewise linear trend
            n_years = self.n_days / 365.25
            n_changepoints = max(1, int(self.rng.poisson(self.trend_changepoint_freq * n_years)))
            
            # Generate changepoint locations
            changepoint_days = sorted(self.rng.choice(
                range(int(self.n_days * 0.1), int(self.n_days * 0.9)),
                size=n_changepoints,
                replace=False
            ))
            changepoint_days = [0] + changepoint_days + [self.n_days]
            
            # Generate slopes with validation for meaningful changes
            # Use a weighted distribution to favor stronger trends
            # 40% chance of stronger slope, 60% chance of moderate slope
            slopes = []
            prev_slope = None
            min_change = 0.008 * scale  # Minimum detectable change threshold
            
            for i in range(len(changepoint_days) - 1):
                if prev_slope is None:
                    # First slope - favor stronger initial trends
                    if self.rng.random() < 0.4:
                        # Stronger trend: -0.03 to -0.015 or 0.02 to 0.05
                        if self.rng.random() < 0.5:
                            new_slope = self.rng.uniform(-0.03, -0.015) * scale
                        else:
                            new_slope = self.rng.uniform(0.02, 0.05) * scale
                    else:
                        # Moderate trend
                        new_slope = self.rng.uniform(-0.01, 0.03) * scale
                else:
                    # Ensure meaningful change: try up to 20 times, then force it
                    for attempt in range(20):
                        # 40% chance of stronger slope
                        if self.rng.random() < 0.4:
                            # Stronger trend
                            if self.rng.random() < 0.5:
                                new_slope = self.rng.uniform(-0.03, -0.015) * scale
                            else:
                                new_slope = self.rng.uniform(0.02, 0.05) * scale
                        else:
                            # Moderate trend
                            new_slope = self.rng.uniform(-0.01, 0.03) * scale
                        # Accept if change is large enough (allow 20% to be subtler)
                        threshold = min_change * (0.5 if self.rng.random() > 0.8 else 1.0)
                        if abs(new_slope - prev_slope) >= threshold:
                            break
                    else:
                        # Force a meaningful change if random sampling failed
                        sign = 1 if self.rng.random() > 0.5 else -1
                        new_slope = np.clip(prev_slope + sign * min_change * 1.5, 
                                          -0.03 * scale, 0.05 * scale)
                
                slopes.append(new_slope)
                prev_slope = new_slope
            
            slopes = np.array(slopes)
            
            # Build piecewise trend
            trend = np.zeros(self.n_days)
            current_level = 50 * scale  # Starting level
            
            changepoint_info = []
            for i in range(len(changepoint_days) - 1):
                start_day = changepoint_days[i]
                end_day = changepoint_days[i + 1]
                slope = slopes[i]
                
                segment_length = end_day - start_day
                segment_trend = current_level + slope * np.arange(segment_length)
                trend[start_day:end_day] = segment_trend
                
                if start_day > 0:  # Don't record the initial point
                    changepoint_info.append((
                        self.date_index[start_day],
                        slopes[i - 1] if i > 0 else 0,
                        slope
                    ))
                
                current_level = segment_trend[-1]
            
            self.trend_changepoints[series_name] = changepoint_info
        
        return trend
    
    def _combine_shared_events(self, non_shared_days, shared_event_days, participation_prob=0.5):
        """Combine non-shared and shared events into a unified dict with shared status."""
        all_days = {day: False for day in non_shared_days}
        for day in shared_event_days:
            if self.rng.rand() < participation_prob:
                all_days[day] = True  # Shared
        return all_days
    
    def _generate_level_shifts(self, series_name, series_type, scale):
        """Generate instantaneous or ramped level shifts."""
        level_shift = np.zeros(self.n_days)
        
        n_years = self.n_days / 365.25
        
        # Vary level shift frequency across series (some rare, some more frequent)
        # Use different multipliers for different series to create variability
        series_index = int(series_name.split('_')[-1]) if '_' in series_name else 0
        
        # Create frequency multiplier based on series index
        # Some series have 0.3x (rare), some 1x, some 2x the base frequency
        freq_options = [0.3, 0.5, 1.0, 1.5, 2.0]
        freq_mult = freq_options[series_index % len(freq_options)]
        
        # Calculate expected number of shifts
        expected_shifts = self.level_shift_freq * n_years * freq_mult
        
        # For shorter datasets, use binomial/probabilistic approach
        if n_years < 1.0:
            # For short datasets, use probability-based sampling
            if self.rng.random() < expected_shifts:
                n_shifts = 1
            elif self.rng.random() < (expected_shifts - 1):
                n_shifts = 2
            else:
                n_shifts = 0
        else:
            # For longer datasets, use Poisson
            n_shifts = int(self.rng.poisson(expected_shifts))
        
        # Generate non-shared shift locations
        shift_days = []
        if n_shifts > 0:
            shift_days = sorted(self.rng.choice(
                range(int(self.n_days * 0.2), int(self.n_days * 0.9)),
                size=n_shifts,
                replace=False
            ))
        
        shift_info = []
        
        # Combine shared and non-shared events
        all_shift_days = self._combine_shared_events(shift_days, self.shared_events['level_shifts'], participation_prob=0.5)
        
        for shift_day, is_shared in sorted(all_shift_days.items()):
            # Determine shift type
            shift_type = self.rng.choice(
                ['instant', 'ramp_2_day', 'ramp_3_day'], 
                p=[0.6, 0.2, 0.2]
            )
            
            # Determine magnitude - should be clearly detectable above noise
            # Use series-specific noise level to ensure detectability
            series_noise_level = self.series_noise_levels[series_name]
            signal_strength = scale * 50
            noise_std = series_noise_level * signal_strength
            
            # Level shifts should be 3-8x the noise std for clear detectability
            shift_multiplier = self.rng.uniform(3, 8)
            magnitude = self.rng.choice([-1, 1]) * shift_multiplier * noise_std
            
            # Apply shift
            if shift_type == 'instant':
                level_shift[shift_day:] += magnitude
            elif shift_type == 'ramp_2_day':
                if shift_day + 2 <= self.n_days:
                    level_shift[shift_day:shift_day+2] += np.linspace(magnitude/2, magnitude, 2)
                    level_shift[shift_day+2:] += magnitude
            elif shift_type == 'ramp_3_day':
                if shift_day + 3 <= self.n_days:
                    level_shift[shift_day:shift_day+3] += np.linspace(magnitude/3, magnitude, 3)
                    level_shift[shift_day+3:] += magnitude
            
            shift_info.append((
                self.date_index[shift_day], magnitude, shift_type, is_shared
            ))
        
        self.level_shifts[series_name] = shift_info
        return level_shift
    
    def _generate_seasonality(self, series_name, series_type, scale):
        """Generate weekly and yearly seasonality with stochastic variation."""
        t = np.arange(self.n_days)
        seasonality = np.zeros(self.n_days)
        
        # Get per-series seasonality strengths
        weekly_strength = self.series_seasonality_strengths[series_name]['weekly']
        yearly_strength = self.series_seasonality_strengths[series_name]['yearly']
        
        # Get series noise level to ensure seasonality is detectable
        series_noise_level = self.series_noise_levels[series_name]
        signal_strength = scale * 50
        noise_std = series_noise_level * signal_strength
        
        # Scale base seasonality amplitude to be a multiple of noise
        # Weekly seasonality should be 1-4x noise (depending on strength setting)
        weekly_amplitude = weekly_strength * noise_std * 2
        yearly_amplitude = yearly_strength * noise_std * 1.5
        
        # Weekly seasonality with noise
        n_weeks = int(np.ceil(self.n_days / 7))
        
        if series_type == 'time_varying_seasonality':
            # Time-varying weekly pattern
            for day_of_week in range(7):
                day_mask = (self.date_index.dayofweek == day_of_week)
                n_occurrences = day_mask.sum()
                
                # Gradually changing mean effect
                base_effect = np.linspace(
                    self.rng.uniform(-1, 1),
                    self.rng.uniform(-1, 1),
                    n_occurrences
                ) * weekly_amplitude
                
                # Add noise (smaller than signal)
                noisy_effect = base_effect + self.rng.normal(0, weekly_amplitude * 0.15, n_occurrences)
                seasonality[day_mask] = noisy_effect
            
            self.seasonality_changepoints[series_name] = [
                (self.date_index[0], 'time_varying_weekly_start'),
                (self.date_index[-1], 'time_varying_weekly_end')
            ]
        
        elif series_type == 'seasonality_changepoints':
            # Weekly pattern with distinct changepoints
            changepoint_day = int(self.n_days * 0.5)
            
            # First half - one pattern
            weekly_pattern_1 = self.rng.uniform(-1, 1, 7) * weekly_amplitude
            
            # Second half - different pattern
            weekly_pattern_2 = self.rng.uniform(-1, 1, 7) * weekly_amplitude
            
            for day_of_week in range(7):
                day_mask = (self.date_index.dayofweek == day_of_week)
                day_indices = np.where(day_mask)[0]
                
                for idx in day_indices:
                    if idx < changepoint_day:
                        base = weekly_pattern_1[day_of_week]
                    else:
                        base = weekly_pattern_2[day_of_week]
                    seasonality[idx] = base + self.rng.normal(0, weekly_amplitude * 0.15)
            
            self.seasonality_changepoints[series_name] = [
                (self.date_index[changepoint_day], 'weekly_pattern_change')
            ]
        
        else:
            # Standard weekly seasonality with stochastic variation
            weekly_pattern = self.rng.uniform(-1, 1, 7) * weekly_amplitude
            
            for day_of_week in range(7):
                day_mask = (self.date_index.dayofweek == day_of_week)
                n_occurrences = day_mask.sum()
                
                # Base pattern with noise
                noisy_effect = (
                    weekly_pattern[day_of_week] + 
                    self.rng.normal(0, weekly_amplitude * 0.15, n_occurrences)
                )
                seasonality[day_mask] = noisy_effect
            
            self.seasonality_changepoints[series_name] = []
        
        # Yearly seasonality (Fourier basis with stochastic amplitude)
        n_fourier = 10
        yearly_seasonality = np.zeros(self.n_days)
        
        for n in range(1, n_fourier + 1):
            # Base Fourier coefficients
            a_n = self.rng.uniform(-1, 1) * yearly_amplitude
            b_n = self.rng.uniform(-1, 1) * yearly_amplitude
            
            # Add small amplitude drift (random walk)
            amplitude_drift = np.cumsum(self.rng.normal(0, yearly_amplitude * 0.01, self.n_days))
            amplitude_drift = amplitude_drift - np.mean(amplitude_drift)  # Center
            
            # Compute Fourier component with drift
            cos_term = (a_n + amplitude_drift * 0.1) * np.cos(2 * np.pi * n * t / 365.25)
            sin_term = (b_n + amplitude_drift * 0.1) * np.sin(2 * np.pi * n * t / 365.25)
            
            yearly_seasonality += (cos_term + sin_term) / n  # Dampen higher frequencies
        
        seasonality += yearly_seasonality
        
        return seasonality
    
    def _generate_holiday_effects(self, series_name, series_type, scale):
        """Generate complex holiday effects with splash and bridge effects."""
        holidays = np.zeros(self.n_days)
        holiday_impacts = {}
        holiday_splash_impacts = {}
        holiday_dates = set()  # Track distinct holiday dates (not splash days)
        
        # Get series noise level for scaling holiday impacts
        series_noise_level = self.series_noise_levels[series_name]
        signal_strength = scale * 50
        noise_std = series_noise_level * signal_strength
        
        # Holiday impacts should be 2-6x noise for detectability
        holiday_scale = self.rng.uniform(2, 6) * noise_std
        
        # Determine splash/bridge configuration for each holiday type (consistent across years)
        if not self.holiday_config:
            # Only ~50% of holidays get splash effects
            self.holiday_config['christmas'] = {
                'has_splash': self.rng.random() < 0.5,
                'has_bridge': self.rng.random() < 0.5
            }
            self.holiday_config['custom_july'] = {
                'has_splash': self.rng.random() < 0.5,
                'has_bridge': self.rng.random() < 0.5
            }
            # Multi-day holidays use deterministic configs to better mimic reality
            self.holiday_config['lunar_new_year'] = {
                'has_splash': True,
                'has_bridge': False
            }
            self.holiday_config['ramadan'] = {
                'has_splash': False,
                'has_bridge': False
            }
            for dom_holiday in self.random_dom_holidays:
                self.holiday_config[dom_holiday['name']] = {
                    'has_splash': dom_holiday['has_splash'],
                    'has_bridge': dom_holiday['has_bridge']
                }
            for wkdom_holiday in self.random_wkdom_holidays:
                self.holiday_config[wkdom_holiday['name']] = {
                    'has_splash': wkdom_holiday['has_splash'],
                    'has_bridge': wkdom_holiday['has_bridge']
                }
        
        # Common holidays: December 25th (Christmas)
        self._add_yearly_holiday(holidays, holiday_impacts, holiday_splash_impacts, 
                                holiday_dates, month=12, day=25, impact=holiday_scale, 
                                splash_days=3, name='christmas')
        
        # Custom holiday: 3rd Tuesday of July
        for year in range(self.date_index.year.min(), self.date_index.year.max() + 1):
            july_dates = pd.date_range(f"{year}-07-01", f"{year}-07-31", freq='D')
            tuesdays = july_dates[july_dates.dayofweek == 1]
            if len(tuesdays) >= 3:
                custom_holiday = tuesdays[2]
                if custom_holiday in self.date_index:
                    self._add_holiday_effect(
                        holidays, holiday_impacts, holiday_splash_impacts, custom_holiday,
                        base_impact=holiday_scale * 0.5,
                        splash_days=2, 
                        holiday_name='custom_july'
                    )
                    holiday_dates.add(custom_holiday)
        
        # Randomly generated day-of-month holidays with consistent naming
        for dom_holiday in self.random_dom_holidays:
            self._add_yearly_holiday(holidays, holiday_impacts, holiday_splash_impacts,
                                    holiday_dates, month=dom_holiday['month'], day=dom_holiday['day'],
                                    impact=holiday_scale * dom_holiday['scale_multiplier'],
                                    splash_days=dom_holiday['splash_days'], name=dom_holiday['name'])
        
        # Randomly generated weekday-of-month holidays
        for wkdom_holiday in self.random_wkdom_holidays:
            for year in range(self.date_index.year.min(), self.date_index.year.max() + 1):
                wkdom_date = self._get_nth_weekday(year, wkdom_holiday['month'], wkdom_holiday['week'], wkdom_holiday['weekday'])
                if wkdom_date is not None and wkdom_date in self.date_index:
                    self._add_holiday_effect(
                        holidays, holiday_impacts, holiday_splash_impacts, wkdom_date,
                        base_impact=holiday_scale * wkdom_holiday['scale_multiplier'],
                        splash_days=wkdom_holiday['splash_days'],
                        holiday_name=wkdom_holiday['name']
                    )
                    holiday_dates.add(wkdom_date)
        
        # Lunar New Year for specific series
        if series_type == 'lunar_holidays':
            try:
                chinese_dates = gregorian_to_chinese(self.date_index)
                # Find Lunar New Year dates (month 1, day 1)
                cny_mask = (chinese_dates['lunar_month'] == 1) & (chinese_dates['lunar_day'] == 1)
                cny_dates = self.date_index[cny_mask]
                
                for cny_date in cny_dates:
                    festival_pattern = [
                        (-3, 0.35),
                        (-2, 0.65),
                        (-1, 0.95),
                        (0, -1.3),
                        (1, -1.15),
                        (2, -1.0),
                        (3, -0.8),
                        (4, -0.6),
                        (5, -0.45),
                        (6, -0.3),
                        (7, 0.4),
                        (8, 0.25),
                    ]
                    applied_dates = self._apply_holiday_pattern(
                        holidays,
                        holiday_impacts,
                        holiday_splash_impacts,
                        cny_date,
                        base_scale=holiday_scale * 1.2,
                        pattern=festival_pattern,
                        core_offsets=set(range(0, 7)),
                        holiday_name='lunar_new_year',
                        include_weekend_boost=True,
                    )
                    for applied_date in applied_dates:
                        holiday_dates.add(applied_date)
            except Exception:
                # If calendar conversion fails, skip
                pass
        
        # Ramadan for specific series
        if series_type == 'ramadan_holidays':
            try:
                islamic_dates = gregorian_to_islamic(self.date_index)
                # Ramadan is month 9, day 1
                ramadan_mask = (islamic_dates['month'] == 9) & (islamic_dates['day'] == 1)
                ramadan_dates = self.date_index[ramadan_mask]
                
                for ramadan_date in ramadan_dates:
                    pre_ramadan = [(-3, 0.3), (-2, 0.55), (-1, 0.85)]
                    core_length = 29
                    early_decline = np.linspace(-0.45, -0.7, 10)
                    deep_decline = np.linspace(-0.75, -0.95, 10)
                    late_decline = np.linspace(-0.9, -0.5, core_length - 20)
                    ramadan_profile = np.concatenate([early_decline, deep_decline, late_decline])
                    post_festival = [(core_length, 0.9), (core_length + 1, 0.6), (core_length + 2, 0.35)]
                    pattern = pre_ramadan + [
                        (offset, weight) for offset, weight in enumerate(ramadan_profile)
                    ] + post_festival
                    applied_dates = self._apply_holiday_pattern(
                        holidays,
                        holiday_impacts,
                        holiday_splash_impacts,
                        ramadan_date,
                        base_scale=holiday_scale * 0.85,
                        pattern=pattern,
                        core_offsets=set(range(0, core_length)),
                        holiday_name='ramadan',
                        include_weekend_boost=False,
                    )
                    for applied_date in applied_dates:
                        holiday_dates.add(applied_date)
            except Exception:
                # If calendar conversion fails, skip
                pass
        
        self.holiday_impacts[series_name] = holiday_impacts
        self.holiday_splash_impacts[series_name] = holiday_splash_impacts
        self.holiday_dates[series_name] = sorted(holiday_dates)
        return holidays
    
    def _add_yearly_holiday(self, holidays_array, holiday_impacts, holiday_splash_impacts,
                           holiday_dates, month, day, impact, splash_days, name):
        """Helper to add a recurring yearly holiday across all years in date range."""
        for year in range(self.date_index.year.min(), self.date_index.year.max() + 1):
            try:
                holiday_date = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
                if holiday_date in self.date_index:
                    self._add_holiday_effect(holidays_array, holiday_impacts, holiday_splash_impacts,
                                           holiday_date, base_impact=impact, splash_days=splash_days, 
                                           holiday_name=name)
                    holiday_dates.add(holiday_date)
            except (ValueError, Exception):
                pass
    
    def _add_holiday_effect(self, holidays_array, holiday_impacts, holiday_splash_impacts, 
                           holiday_date, base_impact, splash_days, holiday_name):
        """Add holiday effect with splash and bridge day effects based on configuration."""
        try:
            holiday_idx = self.date_index.get_loc(holiday_date)
        except KeyError:
            return
        
        # Get configuration for this holiday type
        config = self.holiday_config.get(holiday_name, {'has_splash': False, 'has_bridge': False})
        
        # Check if weekend
        is_weekend = holiday_date.dayofweek >= 5
        
        # Increase impact if weekend
        if is_weekend:
            base_impact *= 1.5
        
        # Main holiday impact (always applied)
        holidays_array[holiday_idx] += base_impact
        holiday_impacts[holiday_date] = holiday_impacts.get(holiday_date, 0) + base_impact
        
        # Splash effects (only if configured)
        if config['has_splash']:
            for offset in range(1, splash_days + 1):
                # Before holiday
                if holiday_idx - offset >= 0:
                    splash_impact = base_impact * (1 - offset / (splash_days + 1)) * 0.4
                    holidays_array[holiday_idx - offset] += splash_impact
                    splash_date = self.date_index[holiday_idx - offset]
                    holiday_splash_impacts[splash_date] = holiday_splash_impacts.get(splash_date, 0) + splash_impact
                
                # After holiday
                if holiday_idx + offset < self.n_days:
                    splash_impact = base_impact * (1 - offset / (splash_days + 1)) * 0.3
                    holidays_array[holiday_idx + offset] += splash_impact
                    splash_date = self.date_index[holiday_idx + offset]
                    holiday_splash_impacts[splash_date] = holiday_splash_impacts.get(splash_date, 0) + splash_impact
        
        # Bridge day effect (only if configured and holiday is Thursday or Tuesday)
        if config['has_bridge']:
            if holiday_date.dayofweek == 3:  # Thursday
                # Friday bridge to weekend
                if holiday_idx + 1 < self.n_days and self.date_index[holiday_idx + 1].dayofweek == 4:
                    bridge_impact = base_impact * 0.6
                    holidays_array[holiday_idx + 1] += bridge_impact
                    bridge_date = self.date_index[holiday_idx + 1]
                    holiday_splash_impacts[bridge_date] = holiday_splash_impacts.get(bridge_date, 0) + bridge_impact
            elif holiday_date.dayofweek == 1:  # Tuesday
                # Monday bridge from weekend
                if holiday_idx - 1 >= 0 and self.date_index[holiday_idx - 1].dayofweek == 0:
                    bridge_impact = base_impact * 0.5
                    holidays_array[holiday_idx - 1] += bridge_impact
                    bridge_date = self.date_index[holiday_idx - 1]
                    holiday_splash_impacts[bridge_date] = holiday_splash_impacts.get(bridge_date, 0) + bridge_impact
    
    def _apply_holiday_pattern(
        self,
        holidays_array,
        holiday_impacts,
        holiday_splash_impacts,
        anchor_date,
        base_scale,
        pattern,
        core_offsets,
        holiday_name,
        include_weekend_boost=True,
    ):
        """Apply a set of relative day offsets and multipliers around a holiday anchor."""
        applied_dates = []
        config = self.holiday_config.get(holiday_name, {'has_splash': False, 'has_bridge': False})
        weekend_multiplier = 1.15 if include_weekend_boost else 1.0
        if config.get('has_bridge') and include_weekend_boost:
            weekend_multiplier = max(weekend_multiplier, 1.2)

        for offset, weight in pattern:
            if weight == 0:
                continue
            current_date = anchor_date + pd.Timedelta(days=int(offset))
            if current_date not in self.date_index:
                continue

            holiday_idx = self.date_index.get_loc(current_date)
            impact = base_scale * weight
            if include_weekend_boost and current_date.dayofweek >= 5:
                impact *= weekend_multiplier

            holidays_array[holiday_idx] += impact
            if offset in core_offsets:
                holiday_impacts[current_date] = holiday_impacts.get(current_date, 0) + impact
            else:
                holiday_splash_impacts[current_date] = holiday_splash_impacts.get(current_date, 0) + impact

            applied_dates.append(current_date)

        return applied_dates

    def _init_random_dom_holidays(self):
        """Create random day-of-month holiday templates shared across series."""
        holidays = []
        n_dom = int(self.rng.randint(2, 4))
        protected = {
            'dom_12_25',
            'dom_12_24',
            'dom_12_31',
            'dom_01_01',
            'dom_07_04',
            'dom_07_01',
        }
        attempts = 0
        while len(holidays) < n_dom and attempts < 40:
            attempts += 1
            month = int(self.rng.randint(1, 13))
            day = int(self.rng.randint(1, 29))  # stay <= 28 to avoid month length issues
            name = f"dom_{month:02d}_{day:02d}"
            if name in protected or any(h['name'] == name for h in holidays):
                continue
            holidays.append({
                'name': name,
                'month': month,
                'day': day,
                'scale_multiplier': self.rng.uniform(0.4, 1.1),
                'splash_days': int(self.rng.randint(1, 4)),
                'has_splash': self.rng.random() < 0.4,
                'has_bridge': self.rng.random() < 0.25,
            })
        return holidays

    def _init_random_wkdom_holidays(self):
        """Create random weekday-of-month holiday templates shared across series."""
        holidays = []
        n_wkdom = int(self.rng.randint(1, 3))
        protected = {
            'wkdom_11_4_4',
            'wkdom_11_4_3',
            'wkdom_05_2_6',
            'wkdom_06_3_6',
            'wkdom_09_1_0',
        }
        attempts = 0
        while len(holidays) < n_wkdom and attempts < 60:
            attempts += 1
            month = int(self.rng.randint(1, 13))
            week = int(self.rng.randint(1, 5))  # 1-4 inclusive
            weekday = int(self.rng.randint(0, 7))
            name = f"wkdom_{month:02d}_{week}_{weekday}"
            if name in protected or any(h['name'] == name for h in holidays):
                continue
            holidays.append({
                'name': name,
                'month': month,
                'week': week,
                'weekday': weekday,
                'scale_multiplier': self.rng.uniform(0.5, 1.2),
                'splash_days': int(self.rng.randint(1, 3)),
                'has_splash': self.rng.random() < 0.35,
                'has_bridge': self.rng.random() < 0.3,
            })
        return holidays

    def _get_nth_weekday(self, year, month, week, weekday):
        """Return the nth weekday of a month; fallback to the last occurrence if needed."""
        month_dates = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq='D')
        month_dates = month_dates[month_dates.month == month]
        weekday_matches = month_dates[month_dates.dayofweek == weekday]
        if len(weekday_matches) == 0:
            return None
        if week <= len(weekday_matches):
            return weekday_matches[week - 1]
        return weekday_matches[-1]

    def _generate_noise(self, series_name, series_type, scale):
        """Generate noise with changepoints and optional GARCH-like regimes."""
        noise = np.zeros(self.n_days)
        
        if series_type == 'variance_regimes':
            # GARCH-like variance regimes
            noise = self._generate_garch_noise(series_name, scale)
        elif series_type == 'autocorrelated_noise' or series_type == 'multiplicative_seasonality':
            # Smoother, autocorrelated noise (also used for multiplicative seasonality)
            noise = self._generate_ar_noise(series_name, scale)
        else:
            # Standard noise with changepoints
            noise = self._generate_standard_noise(series_name, scale)
        
        return noise
    
    def _generate_standard_noise(self, series_name, scale):
        """Standard noise with distribution changepoints and random walk."""
        noise = np.zeros(self.n_days)
        
        # Use per-series noise level
        series_noise_level = self.series_noise_levels[series_name]
        
        # Determine signal strength for noise-to-signal calculation
        signal_strength = scale * 50  # Approximate signal magnitude
        
        # Calculate noise-to-signal ratio
        base_noise_std = series_noise_level * signal_strength
        self.noise_to_signal_ratios[series_name] = series_noise_level
        
        # Generate noise changepoints (rare)
        n_noise_changepoints = self.rng.poisson(0.3 * self.n_days / 365.25)
        
        if n_noise_changepoints == 0:
            changepoint_days = [0, self.n_days]
        else:
            changepoint_days = sorted(self.rng.choice(
                range(int(self.n_days * 0.1), int(self.n_days * 0.9)),
                size=n_noise_changepoints,
                replace=False
            ))
            changepoint_days = [0] + changepoint_days + [self.n_days]
        
        noise_cp_info = []
        
        for i in range(len(changepoint_days) - 1):
            start_day = changepoint_days[i]
            end_day = changepoint_days[i + 1]
            segment_length = end_day - start_day
            
            # Choose distribution for this segment
            dist_type = self.rng.choice(['normal', 'laplace', 't'])
            
            if dist_type == 'normal':
                segment_noise = self.rng.normal(0, base_noise_std, segment_length)
                params = ('normal', base_noise_std)
            elif dist_type == 'laplace':
                segment_noise = self.rng.laplace(0, base_noise_std * 0.7, segment_length)
                params = ('laplace', base_noise_std * 0.7)
            elif dist_type == 't':
                df = self.rng.uniform(3, 10)
                segment_noise = self.rng.standard_t(df, segment_length) * base_noise_std * 0.8
                params = ('t', df, base_noise_std * 0.8)
            
            noise[start_day:end_day] = segment_noise
            
            if start_day > 0:
                old_params = noise_cp_info[-1][1] if noise_cp_info else ('initial', 0)
                noise_cp_info.append((
                    self.date_index[start_day],
                    old_params,
                    params
                ))
        
        self.noise_changepoints[series_name] = noise_cp_info
        
        # Add mean-reverting random walk (small)
        random_walk = np.zeros(self.n_days)
        rw_std = base_noise_std * 0.3  # Much smaller than noise
        mean_reversion_strength = 0.05
        
        for t in range(1, self.n_days):
            # Mean reversion: pull towards zero
            drift = -mean_reversion_strength * random_walk[t - 1]
            innovation = self.rng.normal(drift, rw_std)
            random_walk[t] = random_walk[t - 1] + innovation
        
        # Add random walk to noise
        noise += random_walk
        
        return noise
    
    def _generate_garch_noise(self, series_name, scale):
        """Generate GARCH-like noise with variance regimes."""
        # Use per-series noise level
        series_noise_level = self.series_noise_levels[series_name]
        
        signal_strength = scale * 50
        base_noise_std = series_noise_level * signal_strength
        self.noise_to_signal_ratios[series_name] = series_noise_level
        
        # GARCH(1,1)-like parameters
        omega = 0.1 * base_noise_std**2
        alpha = 0.15  # ARCH term
        beta = 0.75   # GARCH term
        
        # Initialize
        noise = np.zeros(self.n_days)
        variance = np.ones(self.n_days) * base_noise_std**2
        
        # Generate with regime switches
        regime_change_prob = 0.002  # Low probability per day
        current_regime = 'normal'
        regime_multiplier = 1.0
        
        regime_changepoints = []
        
        for t in range(1, self.n_days):
            # Check for regime change
            if self.rng.random() < regime_change_prob:
                old_regime = current_regime
                current_regime = self.rng.choice(['low', 'normal', 'high'], p=[0.2, 0.5, 0.3])
                
                if current_regime == 'low':
                    regime_multiplier = 0.5
                elif current_regime == 'normal':
                    regime_multiplier = 1.0
                elif current_regime == 'high':
                    regime_multiplier = 2.0
                
                regime_changepoints.append((
                    self.date_index[t],
                    old_regime,
                    current_regime
                ))
            
            # GARCH variance update
            variance[t] = (
                omega + 
                alpha * noise[t - 1]**2 + 
                beta * variance[t - 1]
            ) * regime_multiplier**2
            
            # Cap variance to prevent overflow
            variance[t] = np.minimum(variance[t], (base_noise_std * 10)**2)
            
            # Generate noise
            noise[t] = self.rng.normal(0, np.sqrt(variance[t]))
        
        self.noise_changepoints[series_name] = regime_changepoints
        
        return noise
    
    def _generate_ar_noise(self, series_name, scale):
        """Generate smooth, autocorrelated (AR1) noise."""
        # Use per-series noise level
        series_noise_level = self.series_noise_levels[series_name]
        signal_strength = scale * 50
        base_noise_std = series_noise_level * signal_strength
        self.noise_to_signal_ratios[series_name] = series_noise_level
        
        # AR(1) parameters
        # High phi creates smoother noise. Varies per series.
        ar_phi = self.rng.uniform(0.7, 0.98)
        
        # The variance of a stationary AR(1) process is var(epsilon) / (1 - phi^2)
        # We scale the innovation variance to keep the total noise variance consistent.
        innovation_std = base_noise_std * np.sqrt(1 - ar_phi**2)
        
        # Generate noise
        noise = np.zeros(self.n_days)
        for t in range(1, self.n_days):
            noise[t] = ar_phi * noise[t - 1] + self.rng.normal(0, innovation_std)
            
        # Store a record of the noise type
        self.noise_changepoints[series_name] = [
            (self.date_index[0], 'AR(1) noise start', {'phi': ar_phi, 'std': innovation_std})
        ]
        
        return noise
    
    def _generate_anomalies(self, series_name, series_type, scale):
        """Generate anomalies with post-event effects."""
        anomalies = np.zeros(self.n_days)
        
        # Determine anomaly threshold relative to THIS series' noise level
        series_noise_level = self.series_noise_levels[series_name]
        signal_strength = scale * 50
        noise_level = series_noise_level * signal_strength
        # Anomalies should be at least 4x the series-specific noise level
        anomaly_threshold = noise_level * 4
        
        # Generate anomalies
        n_weeks = self.n_days / 7
        n_anomalies = self.rng.poisson(self.anomaly_freq * n_weeks)
        
        if n_anomalies == 0:
            self.anomalies[series_name] = []
            return anomalies
        
        # Get holiday impact dates to avoid overlap
        holiday_impact_dates = set(self.holiday_impacts.get(series_name, {}).keys())
        
        # Convert to day indices for easier checking
        holiday_day_indices = set()
        for holiday_date in holiday_impact_dates:
            try:
                day_idx = (holiday_date - self.date_index[0]).days
                if 0 <= day_idx < self.n_days:
                    holiday_day_indices.add(day_idx)
            except:
                pass
        
        # Helper function to check if anomaly range overlaps with holidays
        def has_holiday_overlap(start_day, max_duration=7):
            """Check if any days in the anomaly range overlap with holidays."""
            for d in range(max_duration):
                if start_day + d >= self.n_days:
                    break
                if start_day + d in holiday_day_indices:
                    return True
            return False
        
        # Generate non-shared anomalies
        anomaly_days = []
        attempts = 0
        while len(anomaly_days) < n_anomalies and attempts < n_anomalies * 10:
            candidate = self.rng.randint(0, self.n_days - 10)
            # Ensure at least 7 days spacing, not on holiday, and no overlap with holiday range
            if (all(abs(candidate - existing) > 7 for existing in anomaly_days) and
                not has_holiday_overlap(candidate)):
                anomaly_days.append(candidate)
            attempts += 1
        
        # Combine shared and non-shared anomalies
        all_anomaly_days = self._combine_shared_events(
            [day for day in anomaly_days],
            [day for day in self.shared_events['anomalies'] 
             if all(abs(day - existing) > 7 for existing in anomaly_days) and not has_holiday_overlap(day)],
            participation_prob=0.5
        )

        anomaly_info = []
        
        for anomaly_day, is_shared in sorted(all_anomaly_days.items()):
            # Decide anomaly characteristics
            duration = self.rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            anomaly_type = self.rng.choice([
                'point_outlier', 'noisy_burst', 'impulse_decay', 'linear_decay', 'transient_change'
            ], p=[0.4, 0.2, 0.15, 0.15, 0.1])
            
            # Generate magnitude (clearly above noise threshold)
            if self.rng.random() < 0.5:
                # Positive anomaly
                magnitude = self.rng.uniform(anomaly_threshold, anomaly_threshold * 3)
            else:
                # Negative anomaly
                magnitude = -self.rng.uniform(anomaly_threshold, anomaly_threshold * 3)
            
            # Apply anomaly
            if anomaly_type == 'point_outlier':
                # Single-day outlier (spike or dip)
                anomalies[anomaly_day] += magnitude
            
            elif anomaly_type == 'noisy_burst':
                # Multi-day anomaly with noisy variation around mean (1-3 days)
                for d in range(duration):
                    if anomaly_day + d < self.n_days:
                        # Draw from same distribution
                        day_magnitude = magnitude + self.rng.normal(0, abs(magnitude) * 0.2)
                        anomalies[anomaly_day + d] += day_magnitude
            
            elif anomaly_type == 'impulse_decay':
                # Impulse with exponential decay
                decay_rate = 0.5
                anomalies[anomaly_day] += magnitude
                for d in range(1, 7):  # 7 day decay
                    if anomaly_day + d < self.n_days:
                        decay_magnitude = magnitude * np.exp(-decay_rate * d)
                        anomalies[anomaly_day + d] += decay_magnitude
            
            elif anomaly_type == 'linear_decay':
                # Linear decay to zero over several days (direction-agnostic)
                ramp_days = 5
                anomalies[anomaly_day] += magnitude
                for d in range(1, ramp_days):
                    if anomaly_day + d < self.n_days:
                        ramp_magnitude = magnitude * (1 - d / ramp_days)
                        anomalies[anomaly_day + d] += ramp_magnitude
            
            elif anomaly_type == 'transient_change':
                # A temporary level shift that reverts
                change_duration = self.rng.randint(3, 8)
                for d in range(change_duration):
                    if anomaly_day + d < self.n_days:
                        anomalies[anomaly_day + d] += magnitude
                duration = change_duration

            anomaly_info.append((
                self.date_index[anomaly_day],
                magnitude,
                anomaly_type,
                duration,
                is_shared
            ))
        
        self.anomalies[series_name] = anomaly_info
        return anomalies
    
    def _generate_regressor_effects(self, series_name, scale):
        """Generate effects from regressors with detailed impact tracking."""
        if self.regressors is None:
            return np.zeros(self.n_days)
        
        regressor_effect = np.zeros(self.n_days)
        regressor_impacts = {}  # Track per-regressor impacts by date
        
        # Promotion effect (if series responds to promotions)
        promotion_coef = 0
        if self.rng.random() < 0.7:  # 70% of series respond to promotions
            promotion_coef = self.rng.uniform(5, 15) * scale
            promo_noise = self.rng.normal(0, promotion_coef * 0.2, self.n_days)
            promotion_effect = self.regressors['promotion'].values * promotion_coef + promo_noise
            regressor_effect += promotion_effect
            
            # Store non-zero impacts
            for i, (date, promo_val, effect) in enumerate(zip(
                self.date_index, 
                self.regressors['promotion'].values, 
                promotion_effect
            )):
                if promo_val > 0 or abs(effect) > 0.01:  # Store if promotion is active or effect is non-trivial
                    if date not in regressor_impacts:
                        regressor_impacts[date] = {}
                    regressor_impacts[date]['promotion'] = effect
        
        # Temperature effect (if series responds to temperature)
        temperature_coef = 0
        if self.rng.random() < 0.5:  # 50% of series respond to temperature
            temperature_coef = self.rng.uniform(-0.5, 0.5) * scale
            temp_noise = self.rng.normal(0, abs(temperature_coef) * 0.1, self.n_days)
            temperature_effect = self.regressors['temperature'].values * temperature_coef + temp_noise
            regressor_effect += temperature_effect
            
            # Store all impacts for temperature (always active)
            for i, (date, effect) in enumerate(zip(self.date_index, temperature_effect)):
                if date not in regressor_impacts:
                    regressor_impacts[date] = {}
                regressor_impacts[date]['temperature'] = effect
        
        # Precipitation effect
        precipitation_coef = 0
        if self.rng.random() < 0.3:  # 30% of series respond to precipitation
            precipitation_coef = self.rng.uniform(-2, 2) * scale
            precip_noise = self.rng.normal(0, abs(precipitation_coef) * 0.15, self.n_days)
            precipitation_effect = self.regressors['precipitation'].values * precipitation_coef + precip_noise
            regressor_effect += precipitation_effect
            
            # Store impacts where precipitation is non-zero
            for i, (date, precip_val, effect) in enumerate(zip(
                self.date_index, 
                self.regressors['precipitation'].values, 
                precipitation_effect
            )):
                if abs(precip_val) > 0.01 or abs(effect) > 0.01:
                    if date not in regressor_impacts:
                        regressor_impacts[date] = {}
                    regressor_impacts[date]['precipitation'] = effect
        
        # Store regressor impacts and coefficients for this series
        self.regressor_impacts[series_name] = {
            'by_date': regressor_impacts,
            'coefficients': {
                'promotion': promotion_coef,
                'temperature': temperature_coef,
                'precipitation': precipitation_coef
            }
        }
        
        return regressor_effect
    
    def _apply_lagged_influence(self, series_name, scale):
        """
        Apply Granger-style lagged influence from lunar_holidays series.
        
        The lagged series will be influenced by the lunar_holidays series (series_5)
        with a 7-day lag. The coefficient is randomly determined but scaled to be
        detectable above noise.
        
        Parameters
        ----------
        series_name : str
            Name of the current series
        scale : float
            Scale factor for this series
            
        Returns
        -------
        np.ndarray
            Lagged influence component
        """
        lagged_influence = np.zeros(self.n_days)
        
        # Identify the lunar_holidays series (series_5)
        source_series_name = 'series_5'
        lag_days = 7
        
        # Check if the source series has been generated
        if source_series_name not in self.components:
            # Source series hasn't been generated yet, return zeros
            # This will be filled later in a second pass if needed
            self.lagged_influences[series_name] = {
                'source': source_series_name,
                'lag': lag_days,
                'coefficient': 0.0,
                'note': 'Source series not yet generated'
            }
            return lagged_influence
        
        # Get the source series data (we'll use the combined signal from components)
        # Use holiday component from lunar series as the primary influence
        source_component = self.components[source_series_name].get('holidays', np.zeros(self.n_days))
        
        # Determine influence coefficient
        # Should be detectable: 0.3-0.7 of the source signal strength
        series_noise_level = self.series_noise_levels[series_name]
        signal_strength = scale * 50
        noise_std = series_noise_level * signal_strength
        
        # Coefficient that makes the lagged effect detectable
        base_coefficient = self.rng.uniform(0.3, 0.7)
        
        # Apply the lag: shift source signal forward by lag_days
        # lagged_influence[t] = coefficient * source[t - lag_days]
        for t in range(lag_days, self.n_days):
            lagged_influence[t] = base_coefficient * source_component[t - lag_days]
        
        # Store the lagged influence information
        self.lagged_influences[series_name] = {
            'source': source_series_name,
            'lag': lag_days,
            'coefficient': base_coefficient
        }
        
        return lagged_influence
    
    # Label access methods - consolidated generic getter
    
    def _get_label_data(self, attr_name, series_name=None, default=None):
        """Generic getter for label data with optional series filtering."""
        data = getattr(self, attr_name)
        if series_name is None:
            return data
        return data.get(series_name, default if default is not None else ([] if isinstance(data.get(next(iter(data), None), None), list) else {}))
    
    def get_trend_changepoints(self, series_name=None):
        """Get trend changepoint labels: {series_name: [(date, old_slope, new_slope), ...]}"""
        return self._get_label_data('trend_changepoints', series_name)
    
    def get_level_shifts(self, series_name=None):
        """Get level shift labels: {series_name: [(date, magnitude, type, shared), ...]}"""
        return self._get_label_data('level_shifts', series_name)
    
    def get_anomalies(self, series_name=None):
        """Get anomaly labels: {series_name: [(date, magnitude, type, duration, shared), ...]}"""
        return self._get_label_data('anomalies', series_name)
    
    def get_holiday_impacts(self, series_name=None):
        """Get holiday impact labels (main holiday dates only): {series_name: {date: impact}}"""
        return self._get_label_data('holiday_impacts', series_name)
    
    def get_holiday_splash_impacts(self, series_name=None):
        """Get holiday splash/bridge day impacts: {series_name: {date: impact}}"""
        return self._get_label_data('holiday_splash_impacts', series_name)
    
    def get_holiday_config(self):
        """Get holiday splash/bridge configuration: {holiday_name: {'has_splash': bool, 'has_bridge': bool}}"""
        return self.holiday_config
    
    def get_regressor_impacts(self, series_name=None):
        """Get regressor impacts: {series_name: {'by_date': {date: {regressor: impact}}, 'coefficients': {...}}}"""
        return self._get_label_data('regressor_impacts', series_name)
    
    def get_noise_changepoints(self, series_name=None):
        """Get noise distribution changepoints: {series_name: [(date, old_params, new_params), ...]}"""
        return self._get_label_data('noise_changepoints', series_name)
    
    def get_seasonality_changepoints(self, series_name=None):
        """Get seasonality changepoints: {series_name: [(date, description), ...]}"""
        return self._get_label_data('seasonality_changepoints', series_name)
    
    def get_noise_to_signal_ratios(self):
        """Get noise-to-signal ratios for all series."""
        return self.noise_to_signal_ratios
    
    def get_series_noise_levels(self):
        """Get per-series noise levels."""
        return self.series_noise_levels
    
    def get_series_seasonality_strengths(self):
        """Get per-series seasonality strengths."""
        return self.series_seasonality_strengths
    
    def get_series_scales(self):
        """Get scale factors for all series."""
        return self.series_scales
    
    def get_lagged_influences(self, series_name=None):
        """
        Get lagged influence information for Granger-style causal relationships.
        
        Parameters
        ----------
        series_name : str, optional
            If provided, return lagged influence info for specific series.
            If None, return all lagged influences.
        
        Returns
        -------
        dict
            Dictionary of {series_name: {'source': source_series, 'lag': lag_days, 'coefficient': coef}}
            or single dict if series_name is specified
        """
        if series_name is None:
            return self.lagged_influences
        return self.lagged_influences.get(series_name, {})
    
    def get_series_type_description(self, series_name):
        """
        Get human-readable description for a series type.
        
        Parameters
        ----------
        series_name : str
            Name of the series
            
        Returns
        -------
        str
            Human-readable description of the series type
        """
        series_type = self.series_types.get(series_name, 'standard')
        return self.SERIES_TYPE_DESCRIPTIONS.get(series_type, series_type)
    
    def get_components(self, series_name=None):
        """
        Get individual components for analysis.
        
        Parameters
        ----------
        series_name : str, optional
            If provided, return components for specific series.
            If None, return all components.
        
        Returns
        -------
        dict
            Dictionary of {series_name: {component_name: array}}
        """
        if series_name is None:
            return self.components
        return self.components.get(series_name, {})
    
    def get_all_labels(self, series_name=None):
        """
        Get all labels in a structured format for easy model evaluation.
        
        Parameters
        ----------
        series_name : str, optional
            If provided, return labels for specific series only.
        
        Returns
        -------
        dict
            Comprehensive dictionary of all labels and metadata.
        """
        if series_name is None:
            return {
                'trend_changepoints': self.trend_changepoints,
                'level_shifts': self.level_shifts,
                'anomalies': self.anomalies,
                'holiday_impacts': self.holiday_impacts,
                'holiday_dates': self.holiday_dates,
                'holiday_splash_impacts': self.holiday_splash_impacts,
                'holiday_config': self.holiday_config,
                'noise_changepoints': self.noise_changepoints,
                'seasonality_changepoints': self.seasonality_changepoints,
                'noise_to_signal_ratios': self.noise_to_signal_ratios,
                'series_noise_levels': self.series_noise_levels,
                'series_seasonality_strengths': self.series_seasonality_strengths,
                'series_scales': self.series_scales,
                'series_types': self.series_types,
                'regressor_impacts': self.regressor_impacts,
            }
        else:
            return {
                'trend_changepoints': self.trend_changepoints.get(series_name, []),
                'level_shifts': self.level_shifts.get(series_name, []),
                'anomalies': self.anomalies.get(series_name, []),
                'holiday_impacts': self.holiday_impacts.get(series_name, {}),
                'holiday_dates': self.holiday_dates.get(series_name, []),
                'holiday_splash_impacts': self.holiday_splash_impacts.get(series_name, {}),
                'holiday_config': self.holiday_config,
                'noise_changepoints': self.noise_changepoints.get(series_name, []),
                'seasonality_changepoints': self.seasonality_changepoints.get(series_name, []),
                'noise_to_signal_ratio': self.noise_to_signal_ratios.get(series_name, None),
                'series_noise_level': self.series_noise_levels.get(series_name, None),
                'series_seasonality_strengths': self.series_seasonality_strengths.get(series_name, {}),
                'series_scale': self.series_scales.get(series_name, None),
                'series_type': self.series_types.get(series_name, 'standard'),
                'regressor_impacts': self.regressor_impacts.get(series_name, {}),
            }
    
    def get_data(self):
        """Get the generated time series data."""
        return self.data
    
    def get_regressors(self):
        """Get the generated regressors (if any)."""
        return self.regressors
    
    def to_csv(self, filepath, include_regressors=False):
        """
        Save generated data to CSV.
        
        Parameters
        ----------
        filepath : str
            Path to save the CSV file
        include_regressors : bool
            Whether to include regressors in the output
        """
        if include_regressors and self.regressors is not None:
            combined = pd.concat([self.data, self.regressors], axis=1)
            combined.to_csv(filepath)
        else:
            self.data.to_csv(filepath)
    
    def summary(self):
        """Print a summary of the generated data."""
        print("=" * 70)
        print("Synthetic Daily Data Generator Summary")
        print("=" * 70)
        print(f"Date range: {self.date_index[0]} to {self.date_index[-1]}")
        print(f"Number of days: {self.n_days}")
        print(f"Number of series: {self.n_series}")
        print(f"Random seed: {self.random_seed}")
        print()
        
        print("Series Characteristics:")
        print("-" * 70)
        for i, series_name in enumerate(self.data.columns):
            series_type = self.series_types.get(series_name, 'standard')
            type_description = self.SERIES_TYPE_DESCRIPTIONS.get(series_type, series_type)
            print(f"\n{series_name} (type: {type_description}):")
            print(f"  Scale factor: {self.series_scales[series_name]:.1f}")
            print(f"  Noise-to-signal ratio: {self.noise_to_signal_ratios[series_name]:.3f}")
            
            # Show seasonality strengths
            if series_name in self.series_seasonality_strengths:
                weekly_str = self.series_seasonality_strengths[series_name]['weekly']
                yearly_str = self.series_seasonality_strengths[series_name]['yearly']
                print(f"  Seasonality strength - Weekly: {weekly_str:.2f}, Yearly: {yearly_str:.2f}")
            
            n_trend_cp = len(self.trend_changepoints.get(series_name, []))
            print(f"  Trend changepoints: {n_trend_cp}")
            
            n_level_shifts = len(self.level_shifts.get(series_name, []))
            print(f"  Level shifts: {n_level_shifts}")
            
            n_anomalies = len(self.anomalies.get(series_name, []))
            print(f"  Anomalies: {n_anomalies}")
            
            n_distinct_holidays = len(self.holiday_dates.get(series_name, []))
            n_main_holiday_days = len(self.holiday_impacts.get(series_name, {}))
            n_splash_days = len(self.holiday_splash_impacts.get(series_name, {}))
            n_total_holiday_days = n_main_holiday_days + n_splash_days
            print(f"  Holidays (distinct): {n_distinct_holidays}")
            print(f"  Holiday impact days: {n_main_holiday_days} main + {n_splash_days} splash/bridge = {n_total_holiday_days} total")
            
            n_season_cp = len(self.seasonality_changepoints.get(series_name, []))
            if n_season_cp > 0:
                print(f"  Seasonality changepoints: {n_season_cp}")
        
        if self.random_dom_holidays or self.random_wkdom_holidays:
            print("\nAdditional synthetic holiday templates:")
            if self.random_dom_holidays:
                dom_names = ", ".join(h['name'] for h in self.random_dom_holidays)
                print(f"  Day-of-month patterns: {dom_names}")
            if self.random_wkdom_holidays:
                wkdom_names = ", ".join(h['name'] for h in self.random_wkdom_holidays)
                print(f"  Weekday-of-month patterns: {wkdom_names}")

        print("\n" + "=" * 70)
    
    def plot(self, series_name=None, figsize=(16, 12), save_path=None, show=True):
        """
        Plot a series with all its labeled components clearly marked.
        
        Parameters
        ----------
        series_name : str, optional
            Name of series to plot. If None, randomly selects one.
        figsize : tuple, optional
            Figure size (width, height) in inches. Default (16, 12).
        save_path : str, optional
            If provided, saves the plot to this path instead of displaying.
        show : bool, optional
            Whether to display the plot. Default True.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object
            
        Raises
        ------
        ImportError
            If matplotlib is not installed
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )
        
        # Select series
        if series_name is None:
            series_name = self.rng.choice(list(self.data.columns))
            print(f"Randomly selected: {series_name}")
        elif series_name not in self.data.columns:
            raise ValueError(f"Series '{series_name}' not found in data")
        
        # Get data and labels
        series_data = self.data[series_name]
        labels = self.get_all_labels(series_name)
        components = self.get_components(series_name)
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Create title with series type using formatted description
        series_type = labels.get('series_type', 'standard')
        type_description = self.SERIES_TYPE_DESCRIPTIONS.get(series_type, series_type)
        
        if series_type != 'standard':
            title = f'Synthetic Data Analysis: {series_name} (type: {type_description})'
        else:
            title = f'Synthetic Data Analysis: {series_name}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        #  Plot 1: Full series with all events marked 
        ax = axes[0]
        ax.plot(self.date_index, series_data, 'b-', alpha=0.7, linewidth=1, label='Full Series')
        
        # Mark anomalies
        for date, magnitude, anom_type, duration, is_shared in labels['anomalies']:
            ax.axvline(date, color='red', alpha=0.4, linestyle='--', linewidth=2)
            # Add small marker at the top
            y_pos = ax.get_ylim()[1] * 0.95
            ax.plot(date, y_pos, 'rv', markersize=8, alpha=0.7)
        
        # Mark trend changepoints
        for cp_data in labels['trend_changepoints']:
            date = cp_data[0]
            ax.axvline(date, color='green', alpha=0.5, linestyle='-', linewidth=2)
            # Add small marker
            y_pos = ax.get_ylim()[1] * 0.90
            ax.plot(date, y_pos, 'g^', markersize=8, alpha=0.7)
        
        # Mark level shifts
        for date, magnitude, shift_type, is_shared in labels['level_shifts']:
            ax.axvline(date, color='purple', alpha=0.6, linestyle=':', linewidth=2.5)
            # Add small marker
            y_pos = ax.get_ylim()[1] * 0.85
            ax.plot(date, y_pos, '*', color='purple', markersize=10, alpha=0.7)
        
        # Mark holidays (main holiday dates, not splash)
        holiday_main_dates = set(labels.get('holiday_dates', []))
        for date in holiday_main_dates:
            ax.axvline(date, color='goldenrod', alpha=0.5, linestyle='-.', linewidth=1.5)
            # Add small marker
            y_pos = ax.get_ylim()[1] * 0.80
            ax.plot(date, y_pos, 'D', color='goldenrod', markersize=6, alpha=0.7)
        
        # Mark seasonality changepoints
        for date, description in labels['seasonality_changepoints']:
            ax.axvline(date, color='darkcyan', alpha=0.4, linestyle='-.', linewidth=1.5)
        
        ax.set_title('Full Time Series with Labeled Events', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='b', linewidth=2, label='Data'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Anomalies'),
            Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='Trend Changes'),
            Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label='Level Shifts'),
            Line2D([0], [0], color='goldenrod', linestyle='-.', linewidth=2, label='Holidays'),
            Line2D([0], [0], color='darkcyan', linestyle='-.', linewidth=2, label='Seasonality Shift'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        #  Plot 2: Trend and Level Shifts 
        ax = axes[1]
        trend_component = components['trend']
        level_shift_component = components['level_shift']
        combined_trend = trend_component + level_shift_component
        
        ax.plot(self.date_index, trend_component, 'g-', linewidth=2, alpha=0.7, label='Trend')
        ax.plot(self.date_index, combined_trend, 'k-', linewidth=2, alpha=0.8, label='Trend + Level Shifts')
        
        # Mark changepoints with vertical lines
        for cp_data in labels['trend_changepoints']:
            date = cp_data[0]
            ax.axvline(date, color='green', alpha=0.3, linestyle='--', linewidth=1)
        
        for date, magnitude, shift_type, is_shared in labels['level_shifts']:
            ax.axvline(date, color='purple', alpha=0.4, linestyle=':', linewidth=2)
            # Annotate shift magnitude
            idx = self.date_index.get_loc(date)
            if idx < len(combined_trend):
                ax.annotate(f'{magnitude:+.1f}', 
                           xy=(date, combined_trend[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, color='purple',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_title('Trend Component with Changepoints and Level Shifts', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        #  Plot 3: Seasonality and Holidays 
        ax = axes[2]
        seasonality_component = components['seasonality']
        holiday_component = components['holidays']
        
        ax.plot(self.date_index, seasonality_component, 'c-', linewidth=1, alpha=0.6, label='Seasonality')
        ax.plot(self.date_index, holiday_component, 'orange', linewidth=1.5, alpha=0.7, label='Holidays')
        ax.plot(self.date_index, seasonality_component + holiday_component, 
                'b-', linewidth=1, alpha=0.8, label='Combined')
        
        # Mark ALL core holiday dates with markers (not just top 5)
        # This ensures multi-day patterns and random holidays are all visible
        holiday_dates = sorted(labels['holiday_impacts'].keys())
        if holiday_dates:
            for date in holiday_dates:
                ax.axvline(date, color='orange', alpha=0.2, linestyle='--', linewidth=1)
                idx = self.date_index.get_loc(date)
                if idx < len(holiday_component):
                    ax.plot(date, holiday_component[idx], 'o', color='orange', markersize=4, alpha=0.6)
            
            # Group holidays by (month, day) or by pattern to find unique holiday types
            # Then annotate ALL occurrences of the top holiday types
            from collections import defaultdict
            holiday_groups = defaultdict(list)
            
            for date in holiday_dates:
                # Create key based on month/day for recurring holidays
                key = (date.month, date.day)
                holiday_groups[key].append((date, labels['holiday_impacts'][date]))
            
            # Find top holiday types by total impact across all occurrences
            holiday_type_impacts = []
            for key, occurrences in holiday_groups.items():
                total_impact = sum(abs(impact) for _, impact in occurrences)
                avg_impact = total_impact / len(occurrences)
                holiday_type_impacts.append((key, occurrences, total_impact, avg_impact))
            
            # Sort by total impact to get most significant holiday types
            holiday_type_impacts.sort(key=lambda x: x[2], reverse=True)
            
            # Annotate ALL occurrences of the top 3-5 holiday types
            max_types = min(5, len(holiday_type_impacts))
            annotated_count = 0
            
            for i, (key, occurrences, total_impact, avg_impact) in enumerate(holiday_type_impacts[:max_types]):
                for date, impact in occurrences:
                    idx = self.date_index.get_loc(date)
                    if idx < len(holiday_component):
                        # Add text annotation showing the impact value
                        ax.annotate(f'{impact:+.0f}', 
                                   xy=(date, holiday_component[idx]),
                                   xytext=(0, 10), textcoords='offset points',
                                   fontsize=7, color='darkorange', ha='center',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='orange'))
                        annotated_count += 1
        
        ax.set_title('Seasonality and Holiday Effects', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        #  Plot 4: Noise and Anomalies 
        ax = axes[3]
        noise_component = components['noise']
        anomaly_component = components['anomalies']
        
        ax.plot(self.date_index, noise_component, 'gray', linewidth=0.5, alpha=0.75, label='Noise')
        ax.plot(self.date_index, anomaly_component, 'r-', linewidth=2, alpha=0.8, label='Anomalies')
        
        # Mark anomalies with details
        for date, magnitude, anom_type, duration, is_shared in labels['anomalies']:
            ax.axvline(date, color='red', alpha=0.2, linestyle='--', linewidth=1)
            idx = self.date_index.get_loc(date)
            if idx < len(anomaly_component):
                # Annotate anomaly type and magnitude
                ax.annotate(f'{anom_type}\n{magnitude:+.1f}', 
                           xy=(date, anomaly_component[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=7, color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
        
        # Mark noise changepoints
        for date, old_params, new_params in labels['noise_changepoints']:
            ax.axvline(date, color='gray', alpha=0.3, linestyle=':', linewidth=1.5)
        
        ax.set_title('Noise and Anomalies', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, self.n_days // 365 // 2)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add text box with summary statistics
        stats_text = (
            f"Scale: {labels['series_scale']:.1f}x | "
            f"Noise/Signal: {labels['noise_to_signal_ratio']:.3f}\n"
            f"Trend CPs: {len(labels['trend_changepoints'])} | "
            f"Level Shifts: {len(labels['level_shifts'])} | "
            f"Anomalies: {len(labels['anomalies'])} | "
            f"Holidays: {len(labels['holiday_impacts'])}"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig


# Convenience function for quick generation
def generate_synthetic_daily_data(
    start_date='2015-01-01',
    n_days=2555,
    n_series=10,
    random_seed=42,
    **kwargs
):
    """
    Quick function to generate synthetic daily data.
    
    Parameters
    ----------
    start_date : str
        Start date for the time series
    n_days : int
        Number of days to generate
    n_series : int
        Number of series to generate
    random_seed : int
        Random seed for reproducibility
    **kwargs
        Additional parameters passed to SyntheticDailyGenerator
    
    Returns
    -------
    generator : SyntheticDailyGenerator
        Generator object with data and labels
    """
    generator = SyntheticDailyGenerator(
        start_date=start_date,
        n_days=n_days,
        n_series=n_series,
        random_seed=random_seed,
        **kwargs
    )
    return generator
