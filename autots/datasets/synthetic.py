# -*- coding: utf-8 -*-
"""
Synthetic Daily Data Generator with Labeled Changepoints, Anomalies, and Holidays

@author: winedarksea with Claude Sonnet v4.5

Matching test file in tests/test_synthetic_data.py
"""

import json
import copy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from autots.tools.calendar import gregorian_to_chinese, gregorian_to_islamic

from autots.tools.plotting import plot_feature_panels, HAS_MATPLOTLIB


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
    
    **Template Compatibility:**
    - Template structure is compatible with TimeSeriesFeatureDetector
    - Both use same JSON-friendly format for components and labels
    - Templates can be saved/loaded and used for model evaluation

    
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
    anomaly_types : list of str or None
        List of anomaly types to generate. Valid types are:
        'point_outlier', 'noisy_burst', 'impulse_decay', 'linear_decay', 'transient_change'
        If None (default), all types will be generated
    disable_holiday_splash : bool
        If True, holidays will only affect a single day with no splash or bridge effects (default False)
    """
    
    # Template version for compatibility tracking
    TEMPLATE_VERSION = "1.0"
    
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
        anomaly_types=None,
        disable_holiday_splash=False,
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
        self.disable_holiday_splash = disable_holiday_splash
        
        # Validate and set anomaly types
        valid_anomaly_types = ['point_outlier', 'noisy_burst', 'impulse_decay', 'linear_decay', 'transient_change']
        if anomaly_types is None:
            self.anomaly_types = valid_anomaly_types
        else:
            # Validate that all provided types are valid
            invalid_types = [t for t in anomaly_types if t not in valid_anomaly_types]
            if invalid_types:
                raise ValueError(f"Invalid anomaly types: {invalid_types}. Valid types are: {valid_anomaly_types}")
            self.anomaly_types = list(anomaly_types)
        
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
        self.template = None  # JSON-friendly template describing generated data

        # Precompute custom holiday templates so all series share the same structure
        self.random_dom_holidays = self._init_random_dom_holidays()
        self.random_wkdom_holidays = self._init_random_wkdom_holidays()
        
        # Generate the data
        self.data = None
        self.regressors = None
        self._generate()
    
    def _generate(self):
        """Main generation pipeline that builds a template first, then renders data from it."""
        self.template = {
            'version': self.TEMPLATE_VERSION,
            'meta': {
                'start_date': self.date_index[0].isoformat(),
                'end_date': self.date_index[-1].isoformat(),
                'n_days': int(self.n_days),
                'n_series': int(self.n_series),
                'frequency': 'D',
                'created_at': pd.Timestamp.now().isoformat(),
                'source': 'SyntheticDailyGenerator',
                'random_seed': int(self.random_seed),
                'series_type_descriptions': copy.deepcopy(self.SERIES_TYPE_DESCRIPTIONS),
                'config': {
                    'trend_changepoint_freq': float(self.trend_changepoint_freq),
                    'level_shift_freq': float(self.level_shift_freq),
                    'anomaly_freq': float(self.anomaly_freq),
                    'shared_anomaly_prob': float(self.shared_anomaly_prob),
                    'shared_level_shift_prob': float(self.shared_level_shift_prob),
                    'weekly_seasonality_strength': float(self.weekly_seasonality_strength),
                    'yearly_seasonality_strength': float(self.yearly_seasonality_strength),
                    'noise_level': float(self.noise_level),
                    'include_regressors': bool(self.include_regressors),
                    'disable_holiday_splash': bool(self.disable_holiday_splash),
                },
                'random_dom_holidays': copy.deepcopy(self.random_dom_holidays),
                'random_wkdom_holidays': copy.deepcopy(self.random_wkdom_holidays),
            },
            'shared_events': {'anomalies': [], 'level_shifts': []},
            'regressors': None,
            'series': {}
        }

        # Generate shared events first so template captures them
        self._generate_shared_events()
        self.template['shared_events'] = copy.deepcopy(self.shared_events)

        # Generate optional regressors
        if self.include_regressors:
            self._generate_regressors()
            self.template['regressors'] = {
                column: self.regressors[column].tolist()
                for column in self.regressors.columns
            }
        else:
            self.template['regressors'] = None

        data_arrays = {}

        # Map series index to type for cleaner assignment
        series_type_map = {
            0: 'business_day', 1: 'saturating_trend', 2: 'time_varying_seasonality',
            3: 'seasonality_changepoints', 4: 'no_level_shifts', 5: 'lunar_holidays',
            6: 'ramadan_holidays', 7: 'variance_regimes', 8: 'autocorrelated_noise',
            9: 'multiplicative_seasonality', 10: 'granger_lagged'
        }

        for i in range(self.n_series):
            series_name = f"series_{i}"
            series_type = series_type_map.get(i, 'standard')
            self.series_types[series_name] = series_type

            # Set scale for this series (every 3rd series is 10x larger)
            scale = 10.0 if i % 3 == 0 else 1.0
            self.series_scales[series_name] = scale

            # Build template for this series and then render it
            series_template = self._build_series_template(series_name, series_type, scale)
            self.template['series'][series_name] = series_template

            component_arrays, series_data = self._render_series_from_template(series_template)
            self.components[series_name] = component_arrays
            data_arrays[series_name] = series_data

        self.template['meta']['holiday_config'] = copy.deepcopy(self.holiday_config)
        self.template['meta']['anomaly_types'] = list(self.anomaly_types)
        self.data = pd.DataFrame(data_arrays, index=self.date_index)

    def _render_series_from_template(self, series_template):
        """Render component arrays and final series from a single series template."""
        component_arrays = {}
        for component_name, component_info in series_template['components'].items():
            values = component_info.get('values')
            if values is None:
                continue
            component_arrays[component_name] = np.array(values, dtype=float)

        n = self.n_days
        trend = component_arrays.get('trend', np.zeros(n))
        level_shift = component_arrays.get('level_shift', np.zeros(n))
        seasonality = component_arrays.get('seasonality', np.zeros(n))
        holidays = component_arrays.get('holidays', np.zeros(n))
        noise = component_arrays.get('noise', np.zeros(n))
        anomalies = component_arrays.get('anomalies', np.zeros(n))
        regressors = component_arrays.get('regressors', np.zeros(n))
        lagged = component_arrays.get('lagged_influence') if 'lagged_influence' in component_arrays else None

        if series_template.get('combination') == 'multiplicative':
            base_signal = trend + level_shift
            with np.errstate(invalid='ignore'):
                min_signal = np.nanmin(base_signal)
            if not np.isfinite(min_signal):
                min_signal = 0.0
            if min_signal <= 0:
                base_signal = base_signal - min_signal + series_template.get('scale_factor', 1.0) * 10.0
            safe_base_signal = np.where(base_signal == 0, 1.0, base_signal)
            seasonality_factor = np.divide(
                seasonality,
                safe_base_signal,
                out=np.zeros_like(seasonality),
                where=safe_base_signal != 0,
            )
            series_data = (
                base_signal * (1.0 + seasonality_factor)
                + holidays
                + noise
                + anomalies
                + regressors
            )
        else:
            series_data = (
                trend
                + level_shift
                + seasonality
                + holidays
                + noise
                + anomalies
                + regressors
            )

        if lagged is not None:
            lagged_array = np.array(lagged, dtype=float)
            component_arrays['lagged_influence'] = lagged_array
            series_data = series_data + lagged_array

        if series_template['series_type'] == 'business_day':
            is_weekend = (self.date_index.dayofweek >= 5)
            series_data = np.where(is_weekend, np.nan, series_data)

        series_template['values'] = series_data.tolist()
        return component_arrays, series_data

    def _serialize_event_list(self, records, field_names):
        """Convert list of tuples with datetimes to JSON-friendly dictionaries."""
        serialized = []
        for record in records:
            item = {}
            for key, value in zip(field_names, record):
                if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
                    value = pd.Timestamp(value).isoformat()
                elif isinstance(value, np.generic):
                    value = value.item()
                item[key] = value
            serialized.append(item)
        return serialized

    def _serialize_datetime_key_dict(self, mapping):
        """Convert dictionary with datetime keys into JSON-friendly form."""
        serialized = {}
        for key, value in mapping.items():
            if isinstance(key, (pd.Timestamp, datetime, np.datetime64)):
                key = pd.Timestamp(key).isoformat()
            if isinstance(value, np.generic):
                value = value.item()
            serialized[key] = value
        return serialized
    
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
    
    def _build_series_template(self, series_name, series_type, scale):
        """Construct a JSON-friendly template describing a single synthetic series."""
        self.components[series_name] = {}

        # Assign per-series noise level (varying around base noise_level)
        noise_multiplier = self.rng.uniform(0.5, 2.0)
        series_noise_level = self.noise_level * noise_multiplier
        self.series_noise_levels[series_name] = series_noise_level

        # Assign per-series seasonality strengths
        weekly_mult = self.rng.uniform(0.3, 2.5)
        yearly_mult = self.rng.uniform(0.2, 2.0)
        series_weekly_strength = self.weekly_seasonality_strength * weekly_mult
        series_yearly_strength = self.yearly_seasonality_strength * yearly_mult
        self.series_seasonality_strengths[series_name] = {
            'weekly': series_weekly_strength,
            'yearly': series_yearly_strength,
        }

        series_template = {
            'series_name': series_name,
            'series_type': series_type,
            'scale_factor': scale,
            'combination': 'multiplicative' if series_type == 'multiplicative_seasonality' else 'additive',
            'components': {},
            'labels': {},
            'metadata': {
                'noise_level': float(series_noise_level),
                'seasonality_strengths': {
                    'weekly': float(series_weekly_strength),
                    'yearly': float(series_yearly_strength),
                },
            },
        }

        # 1. Generate trend
        trend = self._generate_trend(series_name, series_type, scale)
        series_template['components']['trend'] = {
            'values': trend.tolist(),
            'mode': 'saturating' if series_type == 'saturating_trend' else 'piecewise_linear',
        }
        series_template['labels']['trend_changepoints'] = self._serialize_event_list(
            self.trend_changepoints.get(series_name, []),
            ['date', 'prior_slope', 'new_slope'],
        )

        # 2. Generate level shifts
        if series_type != 'no_level_shifts':
            level_shift = self._generate_level_shifts(series_name, series_type, scale)
        else:
            level_shift = np.zeros(self.n_days)
            self.level_shifts[series_name] = []
        series_template['components']['level_shift'] = {
            'values': level_shift.tolist(),
        }
        series_template['labels']['level_shifts'] = self._serialize_event_list(
            self.level_shifts.get(series_name, []),
            ['date', 'magnitude', 'shift_type', 'shared'],
        )

        # 3. Generate seasonality
        seasonality = self._generate_seasonality(series_name, series_type, scale)
        if series_type == 'time_varying_seasonality':
            seasonality_mode = 'time_varying'
        elif series_type == 'seasonality_changepoints':
            seasonality_mode = 'changepoints'
        elif series_type == 'multiplicative_seasonality':
            seasonality_mode = 'multiplicative'
        else:
            seasonality_mode = 'additive'
        series_template['components']['seasonality'] = {
            'values': seasonality.tolist(),
            'mode': seasonality_mode,
        }
        series_template['labels']['seasonality_changepoints'] = self._serialize_event_list(
            self.seasonality_changepoints.get(series_name, []),
            ['date', 'description'],
        )

        # 4. Generate holiday effects
        holidays = self._generate_holiday_effects(series_name, series_type, scale)
        series_template['components']['holidays'] = {
            'values': holidays.tolist(),
        }
        series_template['labels']['holiday_impacts'] = self._serialize_datetime_key_dict(
            self.holiday_impacts.get(series_name, {})
        )
        series_template['labels']['holiday_splash_impacts'] = self._serialize_datetime_key_dict(
            self.holiday_splash_impacts.get(series_name, {})
        )
        series_template['labels']['holiday_dates'] = [
            pd.Timestamp(date).isoformat() for date in self.holiday_dates.get(series_name, [])
        ]

        # 5. Generate noise
        noise = self._generate_noise(series_name, series_type, scale)
        if series_type == 'variance_regimes':
            noise_mode = 'garch'
        elif series_type in ['autocorrelated_noise', 'multiplicative_seasonality']:
            noise_mode = 'ar'
        else:
            noise_mode = 'standard'
        series_template['components']['noise'] = {
            'values': noise.tolist(),
            'mode': noise_mode,
        }
        series_template['labels']['noise_changepoints'] = self._serialize_event_list(
            self.noise_changepoints.get(series_name, []),
            ['date', 'from_params', 'to_params'],
        )

        # 6. Generate anomalies
        anomalies = self._generate_anomalies(series_name, series_type, scale)
        series_template['components']['anomalies'] = {
            'values': anomalies.tolist(),
        }
        series_template['labels']['anomalies'] = self._serialize_event_list(
            self.anomalies.get(series_name, []),
            ['date', 'magnitude', 'pattern', 'duration', 'shared'],
        )

        # 7. Add regressor effects if available
        if self.include_regressors:
            regressor_effect = self._generate_regressor_effects(series_name, scale)
            if np.isscalar(regressor_effect):
                regressor_values = np.zeros(self.n_days).tolist()
            else:
                regressor_values = regressor_effect.tolist()
            series_template['components']['regressors'] = {
                'values': regressor_values,
            }
            regressor_info = self.regressor_impacts.get(series_name, {})
            impacts_by_date = regressor_info.get('by_date', {})
            coefficients = regressor_info.get('coefficients', {})
            series_template['labels']['regressor_impacts'] = {
                'by_date': {
                    pd.Timestamp(event_date).isoformat(): {
                        name: float(val) for name, val in impacts.items()
                    }
                    for event_date, impacts in sorted(impacts_by_date.items())
                },
                'coefficients': {
                    name: float(val) for name, val in coefficients.items()
                },
            }
        else:
            series_template['labels']['regressor_impacts'] = {}

        # 8. Optional Granger-style lagged influence
        if series_type == 'granger_lagged':
            lagged_influence = self._apply_lagged_influence(series_name, scale)
            series_template['components']['lagged_influence'] = {
                'values': lagged_influence.tolist(),
            }
            series_template['labels']['lagged_influence'] = copy.deepcopy(
                self.lagged_influences.get(series_name, {})
            )
        else:
            series_template['labels']['lagged_influence'] = self.lagged_influences.get(series_name, {})

        # Store noise-to-signal ratio (matches previous storage)
        if series_name in self.noise_to_signal_ratios:
            series_template['metadata']['noise_to_signal_ratio'] = float(
                self.noise_to_signal_ratios[series_name]
            )
        else:
            series_template['metadata']['noise_to_signal_ratio'] = float(series_noise_level)

        return series_template
    
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
            # 50% chance of stronger slope, 50% chance of moderate slope
            slopes = []
            prev_slope = None
            min_change = 0.015 * scale  # Minimum detectable change threshold (increased from 0.008)
            
            for i in range(len(changepoint_days) - 1):
                if prev_slope is None:
                    # First slope - favor stronger initial trends
                    if self.rng.random() < 0.5:
                        # Stronger trend: -0.05 to -0.02 or 0.025 to 0.07
                        if self.rng.random() < 0.5:
                            new_slope = self.rng.uniform(-0.05, -0.02) * scale
                        else:
                            new_slope = self.rng.uniform(0.025, 0.07) * scale
                    else:
                        # Moderate trend
                        new_slope = self.rng.uniform(-0.015, 0.04) * scale
                else:
                    # Ensure meaningful change: try up to 20 times, then force it
                    for attempt in range(20):
                        # 50% chance of stronger slope
                        if self.rng.random() < 0.5:
                            # Stronger trend
                            if self.rng.random() < 0.5:
                                new_slope = self.rng.uniform(-0.05, -0.02) * scale
                            else:
                                new_slope = self.rng.uniform(0.025, 0.07) * scale
                        else:
                            # Moderate trend
                            new_slope = self.rng.uniform(-0.015, 0.04) * scale
                        # Accept if change is large enough (allow 10% to be subtler)
                        threshold = min_change * (0.6 if self.rng.random() > 0.9 else 1.0)
                        if abs(new_slope - prev_slope) >= threshold:
                            break
                    else:
                        # Force a meaningful change if random sampling failed
                        sign = 1 if self.rng.random() > 0.5 else -1
                        new_slope = np.clip(prev_slope + sign * min_change * 2.5, 
                                          -0.05 * scale, 0.07 * scale)
                
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
            if self.disable_holiday_splash:
                # Disable all splash and bridge effects
                self.holiday_config['christmas'] = {
                    'has_splash': False,
                    'has_bridge': False
                }
                self.holiday_config['custom_july'] = {
                    'has_splash': False,
                    'has_bridge': False
                }
                self.holiday_config['lunar_new_year'] = {
                    'has_splash': False,
                    'has_bridge': False
                }
                self.holiday_config['ramadan'] = {
                    'has_splash': False,
                    'has_bridge': False
                }
                for dom_holiday in self.random_dom_holidays:
                    self.holiday_config[dom_holiday['name']] = {
                        'has_splash': False,
                        'has_bridge': False
                    }
                for wkdom_holiday in self.random_wkdom_holidays:
                    self.holiday_config[wkdom_holiday['name']] = {
                        'has_splash': False,
                        'has_bridge': False
                    }
            else:
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
                    if self.disable_holiday_splash:
                        # Single day effect only
                        festival_pattern = [(0, -1.3)]
                        core_offsets = {0}
                    else:
                        # Multi-day festival pattern
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
                        core_offsets = set(range(0, 7))
                    
                    applied_dates = self._apply_holiday_pattern(
                        holidays,
                        holiday_impacts,
                        holiday_splash_impacts,
                        cny_date,
                        base_scale=holiday_scale * 1.2,
                        pattern=festival_pattern,
                        core_offsets=core_offsets,
                        holiday_name='lunar_new_year',
                        include_weekend_boost=not self.disable_holiday_splash,
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
                    if self.disable_holiday_splash:
                        # Single day effect only (first day of Ramadan)
                        pattern = [(0, -0.85)]
                        core_offsets = {0}
                    else:
                        # Full month-long Ramadan pattern
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
                        core_offsets = set(range(0, core_length))
                    
                    applied_dates = self._apply_holiday_pattern(
                        holidays,
                        holiday_impacts,
                        holiday_splash_impacts,
                        ramadan_date,
                        base_scale=holiday_scale * 0.85,
                        pattern=pattern,
                        core_offsets=core_offsets,
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
            
            # Select anomaly type from allowed types
            if len(self.anomaly_types) == 1:
                anomaly_type = self.anomaly_types[0]
            else:
                # Use proportional probabilities if multiple types are allowed
                # Default probabilities: point_outlier=0.4, noisy_burst=0.2, impulse_decay=0.15, linear_decay=0.15, transient_change=0.1
                default_probs = {
                    'point_outlier': 0.4,
                    'noisy_burst': 0.2,
                    'impulse_decay': 0.15,
                    'linear_decay': 0.15,
                    'transient_change': 0.1
                }
                # Filter to allowed types and normalize
                allowed_probs = [default_probs[t] for t in self.anomaly_types]
                prob_sum = sum(allowed_probs)
                normalized_probs = [p / prob_sum for p in allowed_probs]
                anomaly_type = self.rng.choice(self.anomaly_types, p=normalized_probs)
            
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

    def get_template(self, series_name=None, deep=True):
        """Get the JSON-friendly template describing the generated data."""
        if self.template is None:
            return None
        template = self.template if not deep else copy.deepcopy(self.template)
        if series_name is None:
            return template
        if series_name not in template['series']:
            raise KeyError(f"Series '{series_name}' not found in template.")
        if deep:
            return template['series'][series_name]
        return self.template['series'][series_name]
    
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

    def machine_summary(
        self,
        series_name=None,
        include_events=True,
        include_regressors=True,
        max_events_per_type=25,
        round_decimals=6,
        as_json=False,
    ):
        """Return a structured summary tailored for LLM or tool consumption."""

        if self.template is None:
            raise RuntimeError("Synthetic template has not been generated yet.")

        if series_name is not None and series_name not in self.template['series']:
            raise KeyError(f"Series '{series_name}' not found in generated data.")

        template_copy = copy.deepcopy(self.template)
        selected_series = [series_name] if series_name else list(template_copy['series'].keys())

        def _normalize_scalar(value):
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                value = float(value)
            if isinstance(value, float) and round_decimals is not None:
                return round(value, round_decimals)
            return value

        def _normalize_datetime(value):
            if isinstance(value, (pd.Timestamp, datetime)):
                return value.isoformat()
            if isinstance(value, np.datetime64):
                return pd.Timestamp(value).isoformat()
            return value

        def _normalize(value):
            if isinstance(value, dict):
                normalized = {}
                for key, val in value.items():
                    if isinstance(key, (pd.Timestamp, datetime, np.datetime64)):
                        key = _normalize_datetime(key)
                    normalized[key] = _normalize(val)
                return normalized
            if isinstance(value, (list, tuple, set)):
                return [_normalize(item) for item in value]
            if isinstance(value, np.ndarray):
                return [_normalize(item) for item in value.tolist()]
            value = _normalize_datetime(value)
            return _normalize_scalar(value)

        def _limit(sequence):
            if max_events_per_type is None or sequence is None:
                return sequence
            return sequence[:max_events_per_type]

        def _sorted_items(mapping):
            if not mapping:
                return []
            return sorted(mapping.items(), key=lambda item: item[0])

        def _holiday_details(main_impacts, splash_impacts):
            main_days = [
                {
                    'date': _normalize(date_str),
                    'impact': _normalize(impact_value),
                }
                for date_str, impact_value in _sorted_items(main_impacts)
            ]
            splash_days = [
                {
                    'date': _normalize(date_str),
                    'impact': _normalize(impact_value),
                }
                for date_str, impact_value in _sorted_items(splash_impacts)
            ]
            return {
                'distinct_dates': _normalize(sorted(main_impacts.keys())),
                'main_days': _limit(main_days),
                'splash_days': _limit(splash_days),
            }

        summary_meta = copy.deepcopy(template_copy['meta'])
        summary_meta['config'] = _normalize(summary_meta.get('config', {}))
        summary_meta['random_seed'] = _normalize(summary_meta.get('random_seed'))
        summary_meta['n_days'] = _normalize(summary_meta.get('n_days'))
        summary_meta['n_series'] = _normalize(summary_meta.get('n_series'))
        holiday_config = summary_meta.get('holiday_config', {})
        summary_meta['synthetic_holiday_templates'] = {
            'day_of_month': _normalize(summary_meta.pop('random_dom_holidays', [])),
            'weekday_of_month': _normalize(summary_meta.pop('random_wkdom_holidays', [])),
            'holiday_config': _normalize(holiday_config),
        }
        summary_meta.pop('holiday_config', None)
        summary_meta['shared_event_days'] = _normalize(template_copy.get('shared_events', {}))

        summary = {
            'meta': summary_meta,
            'series': [],
        }

        if include_regressors and self.include_regressors and template_copy.get('regressors') is not None:
            summary['regressors'] = {
                key: _normalize(values)
                for key, values in template_copy['regressors'].items()
            }

        for name in selected_series:
            template_entry = copy.deepcopy(template_copy['series'][name])
            labels = copy.deepcopy(template_entry.get('labels', {}))
            template_entry.pop('series_name', None)

            series_info = {
                'name': name,
                'type': template_entry.get('series_type', 'standard'),
                'scale_factor': _normalize(template_entry.get('scale_factor')),
                'noise_to_signal_ratio': _normalize(
                    template_entry.get('metadata', {}).get('noise_to_signal_ratio')
                ),
                'seasonality_strengths': _normalize(
                    template_entry.get('metadata', {}).get('seasonality_strengths', {})
                ),
                'template': template_entry,
            }

            series_data = self.data[name]
            non_null = int(series_data.notna().sum())
            value_stats = {
                'mean': _normalize(series_data.mean()),
                'median': _normalize(series_data.median()),
                'std': _normalize(series_data.std()),
                'min': _normalize(series_data.min()),
                'max': _normalize(series_data.max()),
                'non_null_count': non_null,
                'non_null_fraction': _normalize(non_null / len(series_data)),
            }
            series_info['value_stats'] = value_stats

            component_arrays = self.components.get(name, {})
            component_stats = {}
            for component_name, component_values in component_arrays.items():
                component_array = np.asarray(component_values, dtype=float)
                component_stats[component_name] = {
                    'mean': _normalize(np.nanmean(component_array)),
                    'std': _normalize(np.nanstd(component_array)),
                    'min': _normalize(np.nanmin(component_array)),
                    'max': _normalize(np.nanmax(component_array)),
                }
            if component_stats:
                series_info['component_stats'] = component_stats

            holiday_impacts = labels.get('holiday_impacts', {})
            holiday_splash = labels.get('holiday_splash_impacts', {})
            holiday_dates = labels.get('holiday_dates', [])

            event_counts = {
                'trend_changepoints': len(labels.get('trend_changepoints', [])),
                'level_shifts': len(labels.get('level_shifts', [])),
                'anomalies': len(labels.get('anomalies', [])),
                'holidays': len(holiday_dates),
                'holiday_impact_days': len(holiday_impacts),
                'holiday_splash_days': len(holiday_splash),
                'seasonality_changepoints': len(labels.get('seasonality_changepoints', [])),
                'noise_changepoints': len(labels.get('noise_changepoints', [])),
            }
            series_info['event_counts'] = event_counts

            if include_events:
                events = {
                    'trend_changepoints': _limit(
                        _normalize(labels.get('trend_changepoints', []))
                    ),
                    'level_shifts': _limit(
                        _normalize(labels.get('level_shifts', []))
                    ),
                    'anomalies': _limit(
                        _normalize(labels.get('anomalies', []))
                    ),
                    'seasonality_changepoints': _limit(
                        _normalize(labels.get('seasonality_changepoints', []))
                    ),
                    'noise_changepoints': _limit(
                        _normalize(labels.get('noise_changepoints', []))
                    ),
                }

                events['holidays'] = _holiday_details(holiday_impacts, holiday_splash)
                holiday_labels = [
                    {'date': _normalize(date_str)}
                    for date_str in sorted(holiday_dates)
                ]
                events['holiday_labels'] = _limit(holiday_labels)

                series_info['events'] = events

            if include_regressors and self.include_regressors:
                regressor_info = labels.get('regressor_impacts', {})
                regressor_mapping = regressor_info.get('by_date', {})
                regressor_details = []
                for date_str, impacts in sorted(regressor_mapping.items()):
                    regressor_details.append({
                        'date': _normalize(date_str),
                        'impacts': _normalize(impacts),
                    })
                if regressor_details:
                    series_info['regressor_impacts'] = _limit(regressor_details)
                coefficients = regressor_info.get('coefficients', {})
                if coefficients:
                    series_info['regressor_coefficients'] = _normalize(coefficients)

            summary['series'].append(series_info)

        return json.dumps(summary, indent=2) if as_json else summary

    @classmethod
    def render_template(cls, template, return_components=False):
        """Render a template into time series using the generator's renderer."""
        if template is None:
            raise ValueError("Template cannot be None when rendering.")

        template_copy = copy.deepcopy(template)
        meta = template_copy.get('meta', {})
        start_date = pd.Timestamp(meta.get('start_date', datetime.utcnow().date()))
        n_days = int(meta.get('n_days', 0))
        if n_days <= 0:
            first_series = next(iter(template_copy.get('series', {}).values()), None)
            if first_series:
                trend = first_series.get('components', {}).get('trend', {}).get('values', [])
                n_days = len(trend)
        if n_days <= 0:
            raise ValueError("Template must include a positive n_days or component values to infer length.")

        frequency = meta.get('frequency', 'D') or 'D'
        if frequency == 'infer':
            frequency = 'D'
        date_index = pd.date_range(start=start_date, periods=n_days, freq=frequency)

        renderer = cls.__new__(cls)
        renderer.n_days = n_days
        renderer.n_series = meta.get('n_series', len(template_copy.get('series', {})))
        renderer.date_index = date_index
        renderer.template = template_copy
        renderer.components = {}
        renderer.data = None
        renderer.include_regressors = template_copy.get('regressors') is not None
        renderer.regressors = None

        data_arrays = {}
        for series_name, series_template in template_copy.get('series', {}).items():
            series_template_copy = copy.deepcopy(series_template)
            component_arrays, series_data = renderer._render_series_from_template(series_template_copy)
            renderer.components[series_name] = component_arrays
            data_arrays[series_name] = series_data
            template_copy['series'][series_name] = series_template_copy

        renderer.data = pd.DataFrame(data_arrays, index=date_index)

        if return_components:
            return renderer.data, renderer.components
        return renderer.data

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

        if series_name is None:
            series_name = self.rng.choice(list(self.data.columns))
            print(f"Randomly selected: {series_name}")
        elif series_name not in self.data.columns:
            raise ValueError(f"Series '{series_name}' not found in data")

        labels = self.get_all_labels(series_name)
        components = self.get_components(series_name)
        series_type = labels.get('series_type', 'standard')
        description = self.SERIES_TYPE_DESCRIPTIONS.get(series_type, series_type)

        fig = plot_feature_panels(
            series_name=series_name,
            date_index=self.date_index,
            series_data=self.data[series_name],
            components=components,
            labels=labels,
            series_type_description=description if series_type != 'standard' else None,
            scale=labels.get('series_scale'),
            noise_to_signal=labels.get('noise_to_signal_ratio'),
            figsize=figsize,
            title_prefix='Synthetic Data Analysis',
            save_path=save_path,
            show=show,
        )

        if save_path:
            print(f"Plot saved to: {save_path}")

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
