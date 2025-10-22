# -*- coding: utf-8 -*-
"""
Time Series Feature Detection and Optimization

@author: Colin with Claude Sonnet v4.5
"""

import numpy as np
import pandas as pd
import random
import copy
import warnings
from autots.tools.transform import (
    DatepartRegressionTransformer,
    AnomalyRemoval,
    LevelShiftMagic,
)
from autots.evaluator.anomaly_detector import HolidayDetector
from autots.tools.changepoints import ChangePointDetector
from autots.tools.impute import FillNA
from autots.tools.anomaly_utils import anomaly_new_params
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TimeSeriesFeatureDetector:
    """
    Detect time series features from unlabeled data.
    
    Identifies:
    - Trend changepoints
    - Level shifts
    - Seasonality patterns
    - Holiday effects
    - Anomalies
    
    Can be optimized against synthetic labeled data to improve detection accuracy.
    
    Parameters
    ----------
    seasonality_params : dict
        Parameters for DatepartRegressionTransformer
    holiday_params : dict
        Parameters for HolidayDetector
    anomaly_params : dict
        Parameters for AnomalyRemoval
    changepoint_params : dict
        Parameters for ChangePointDetector
    level_shift_params : dict
        Parameters for LevelShiftMagic
    standardize : bool
        Whether to standardize data before detection (default True)
    smoothing_window : int
        Window size for smoothing before changepoint detection (default None)
    refine_seasonality : bool
        Whether to refine seasonality after anomaly removal (default True)
    """
    
    def _sanitize_holiday_params(self, holiday_params):
        """Return holiday detector parameters filtered to supported keys."""
        default_params = {
            'anomaly_detector_params': {},
            'threshold': 0.8,
            'min_occurrences': 2,
            'splash_threshold': 0.65,
            'use_dayofmonth_holidays': True,
            'use_wkdom_holidays': True,
            'use_wkdeom_holidays': True,
            'use_lunar_holidays': True,
            'use_lunar_weekday': False,
            'use_islamic_holidays': False,
            'use_hebrew_holidays': False,
            'use_hindu_holidays': False,
            'output': 'multivariate',
            'n_jobs': 1,
        }

        if holiday_params is None:
            return default_params

        allowed_keys = set(default_params.keys())
        sanitized = default_params.copy()
        unsupported = []

        for key, value in holiday_params.items():
            if key in allowed_keys:
                sanitized[key] = value
            else:
                unsupported.append(key)

        if unsupported:
            warnings.warn(
                f"Ignoring unsupported holiday_params keys: {sorted(set(unsupported))}",
                RuntimeWarning,
            )

        return sanitized

    def __init__(
        self,
        seasonality_params=None,
        holiday_params=None,
        anomaly_params=None,
        changepoint_params=None,
        level_shift_params=None,
        standardize=True,
        smoothing_window=None,
        refine_seasonality=True,
    ):
        # Default parameters
        self.seasonality_params = seasonality_params or {
            'regression_model': {
                'model': 'DecisionTree',
                'model_params': {'max_depth': 5, 'min_samples_split': 2}
            },
            'datepart_method': 'common_fourier',
            'polynomial_degree': None,
            'holiday_countries_used': False,
        }
        
        self.holiday_params = self._sanitize_holiday_params(holiday_params)
        
        self.anomaly_params = anomaly_params or {
            'output': 'multivariate',
            'method': 'zscore',
            'method_params': {'distribution': 'norm', 'alpha': 0.05},
            'transform_dict': None,
            'fillna': 'ffill',
        }
        
        self.changepoint_params = changepoint_params or {
            'method': 'pelt',
            'method_params': {'penalty': 10},
            'min_segment_length': 7,
        }
        
        self.level_shift_params = level_shift_params or {
            'window_size': 90,
            'alpha': 2.5,
            'grouping_forward_limit': 3,
            'max_level_shifts': 20,
        }
        
        self.standardize = standardize
        self.smoothing_window = smoothing_window
        self.refine_seasonality = refine_seasonality
        
        # Storage for detected features
        self.trend_changepoints = {}
        self.level_shifts = {}
        self.anomalies = {}
        self.holiday_impacts = {}
        self.holiday_dates = {}
        self.seasonality_components = {}
        self.seasonality_strength = {}
        
        # Models and transformers
        self.scaler = None
        self.seasonality_model = None
        self.refined_seasonality_model = None
        self.holiday_detector = None
        self.anomaly_detector = None
        self.changepoint_detector = None
        self.level_shift_detector = None
        
    def fit(self, df):
        """
        Detect all features in the time series.
        
        Parameters
        ----------
        df : pd.DataFrame
            Time series data with DatetimeIndex
            
        Returns
        -------
        self
        """
        self.df_original = df.copy()
        self.date_index = df.index
        
        # Step 1: Standardization (optional)
        if self.standardize:
            self.scaler = StandardScaler()
            df_work = pd.DataFrame(
                self.scaler.fit_transform(df),
                index=df.index,
                columns=df.columns
            )
        else:
            df_work = df.copy()
            
        # Step 2: Detect and remove seasonality
        self._detect_seasonality(df_work)
        df_deseasonalized = self.seasonality_model.transform(df_work)
        
        # Step 3: Detect anomalies on deseasonalized data
        self._detect_anomalies(df_deseasonalized)
        df_no_anomalies = self.anomaly_detector.transform(df_deseasonalized)
        
        # Step 4: Optionally refine seasonality without anomalies
        if self.refine_seasonality:
            self._refine_seasonality(df_no_anomalies)
            df_deseasonalized_refined = self.refined_seasonality_model.transform(df_work)
        else:
            df_deseasonalized_refined = df_deseasonalized
            
        # Step 5: Detect holidays (on deseasonalized, anomaly-removed data)
        self._detect_holidays(df_deseasonalized_refined)
        
        # Step 6: Smooth data for changepoint detection (optional)
        if self.smoothing_window is not None and self.smoothing_window > 1:
            df_for_changepoints = df_deseasonalized_refined.rolling(
                window=self.smoothing_window, 
                center=True, 
                min_periods=1
            ).mean()
        else:
            df_for_changepoints = df_deseasonalized_refined
            
        # Step 7: Detect level shifts
        self._detect_level_shifts(df_for_changepoints)
        
        # Step 8: Detect trend changepoints
        self._detect_changepoints(df_for_changepoints)
        
        return self
    
    def _detect_seasonality(self, df):
        """Detect and model seasonality."""
        self.seasonality_model = DatepartRegressionTransformer(**self.seasonality_params)
        self.seasonality_model.fit(df)
        
        # Get the deseasonalized (residual) data
        df_deseasonalized = self.seasonality_model.transform(df)
        
        # Store seasonality strength using multiple metrics
        for col in df.columns:
            # Get predictions (seasonal component)
            X_input = (
                self.seasonality_model.X.fillna(0) if isinstance(self.seasonality_model.X, pd.DataFrame) 
                else np.nan_to_num(self.seasonality_model.X)
            )
            seasonal_pred = self.seasonality_model.model.predict(X_input)
            
            # Handle multioutput vs single output
            if seasonal_pred.ndim > 1 and seasonal_pred.shape[1] > 1:
                col_idx = df.columns.get_loc(col)
                seasonal_component = seasonal_pred[:, col_idx]
            elif seasonal_pred.ndim > 1:
                seasonal_component = seasonal_pred.ravel()
            else:
                seasonal_component = seasonal_pred
            
            # Ensure correct length
            if len(seasonal_component) != len(df):
                # This can happen with partial_nan_rows mode
                # Use the deseasonalized data to calculate residuals
                seasonal_component = (df[col].values - df_deseasonalized[col].values)
            
            # Get actual values and residuals
            y_true = df[col].values
            residuals = df_deseasonalized[col].values
            
            # Remove NaN values for proper calculation
            valid_mask = ~(np.isnan(y_true) | np.isnan(residuals) | np.isnan(seasonal_component))
            if not valid_mask.any():
                self.seasonality_strength[col] = 0.0
                self.seasonality_components[col] = np.zeros(len(df))
                continue
            
            y_true_clean = y_true[valid_mask]
            residuals_clean = residuals[valid_mask]
            seasonal_clean = seasonal_component[valid_mask]
            
            # Metric 1: R-squared (coefficient of determination)
            # R² = 1 - (SS_res / SS_tot)
            # Measures how much variance is explained by the seasonal model
            ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
            ss_res = np.sum(residuals_clean ** 2)
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                # Clip to [0, 1] range (can be negative if model is worse than mean)
                r_squared = max(0.0, min(1.0, r_squared))
            else:
                r_squared = 0.0
            
            # Metric 2: Correlation-based strength
            # Measures how well the seasonal component correlates with original data
            if len(seasonal_clean) > 1:
                correlation = np.corrcoef(y_true_clean, seasonal_clean)[0, 1]
                # Square correlation to get proportion of variance explained
                correlation_strength = max(0.0, correlation ** 2)
            else:
                correlation_strength = 0.0
            
            # Metric 3: Variance ratio
            # Compares variance of seasonal component to total variance
            total_var = np.var(y_true_clean)
            if total_var > 0:
                seasonal_var = np.var(seasonal_clean)
                variance_ratio = min(1.0, seasonal_var / total_var)
            else:
                variance_ratio = 0.0
            
            # Metric 4: Autocorrelation improvement
            # Measures reduction in autocorrelation after deseasonalization
            # Strong seasonality should reduce autocorrelation in residuals
            acf_improvement = self._calculate_acf_improvement(y_true_clean, residuals_clean)
            
            # Metric 5: Periodicity strength
            # Uses autocorrelation to detect if seasonal component has clear periodicity
            periodicity_strength = self._calculate_periodicity_strength(seasonal_clean)
            
            # Combine metrics using a balanced approach
            # Strategy: Use R² as the foundation, confirm/adjust with other metrics
            
            # Core strength from variance explained (most reliable metric)
            core_strength = (
                0.70 * r_squared +           # Primary: variance explained by model
                0.30 * correlation_strength   # Confirmation: correlation with seasonal pattern
            )
            
            # Adjustment factors based on pattern characteristics
            # These should not dominate, but provide evidence for/against seasonality
            
            # Periodicity adjustment (0 to +0.2)
            # Strong periodic patterns boost the score
            periodicity_adj = 0.2 * periodicity_strength
            
            # ACF improvement adjustment (-0.2 to +0.1)
            # Good ACF improvement = small boost
            # Poor ACF improvement = stronger penalty (rejects random walks/trends)
            if acf_improvement > 0.4:
                acf_adj = 0.1 * acf_improvement
            elif acf_improvement < 0.25:
                # Penalize cases with poor ACF improvement more strongly
                # This helps reject random walks which don't improve ACF
                acf_adj = -0.2 * (1 - acf_improvement)
            else:
                # Moderate improvement, small positive adjustment
                acf_adj = 0.05 * (acf_improvement - 0.2)
            
            # Variance ratio bonus (0 to +0.1)
            # If seasonal component has substantial variance, small bonus
            variance_bonus = 0.1 * variance_ratio
            
            # Combine all components
            combined_strength = core_strength + periodicity_adj + acf_adj + variance_bonus
            
            # Final clipping to [0, 1] range
            combined_strength = max(0.0, min(1.0, combined_strength))
            
            self.seasonality_strength[col] = combined_strength
            self.seasonality_components[col] = seasonal_component
    
    def _calculate_acf_improvement(self, original, residuals, max_lag=30):
        """
        Calculate improvement in autocorrelation after deseasonalization.
        
        Returns a value in [0, 1] where higher means better improvement
        (i.e., residuals have less autocorrelation than original).
        """
        if len(original) < max_lag + 1:
            return 0.0
        
        try:
            # Calculate mean autocorrelation (excluding lag 0)
            original_acf = self._calculate_acf(original, max_lag=max_lag)
            residual_acf = self._calculate_acf(residuals, max_lag=max_lag)
            
            # Take mean absolute autocorrelation (excluding lag 0)
            if len(original_acf) > 1 and len(residual_acf) > 1:
                original_mean_acf = np.mean(np.abs(original_acf[1:]))
                residual_mean_acf = np.mean(np.abs(residual_acf[1:]))
                
                # Calculate relative improvement
                if original_mean_acf > 0:
                    improvement = (original_mean_acf - residual_mean_acf) / original_mean_acf
                    # Clip to reasonable range
                    return max(0.0, min(1.0, improvement))
            
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_acf(self, x, max_lag=30):
        """
        Calculate autocorrelation function up to max_lag.
        
        Returns array of autocorrelations for lags 0 to max_lag.
        """
        x = np.asarray(x)
        x = x - np.mean(x)
        
        # Use numpy correlation for efficiency
        c0 = np.dot(x, x) / len(x)
        
        if c0 == 0:
            return np.zeros(max_lag + 1)
        
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        
        for lag in range(1, min(max_lag + 1, len(x))):
            c = np.dot(x[:-lag], x[lag:]) / len(x)
            acf[lag] = c / c0
        
        return acf
    
    def _calculate_periodicity_strength(self, seasonal_component, common_periods=None):
        """
        Calculate how periodic/regular the seasonal component is.
        
        Uses autocorrelation to detect strong periodic patterns.
        Returns value in [0, 1] where higher means more periodic.
        """
        if common_periods is None:
            # Common periods for different time series frequencies
            # For daily data: weekly (7), monthly (~30), quarterly (~90)
            # For hourly: daily (24), weekly (168)
            common_periods = [7, 14, 28, 30, 90, 365]
        
        if len(seasonal_component) < 14:
            return 0.0
        
        try:
            # Calculate autocorrelation
            max_lag = min(365, len(seasonal_component) - 1)
            acf = self._calculate_acf(seasonal_component, max_lag=max_lag)
            
            # Look for peaks at common periods
            period_strength = 0.0
            peaks_found = 0
            
            for period in common_periods:
                if period < len(acf):
                    # Check autocorrelation at this lag
                    # Also check surrounding lags to account for slight misalignment
                    window_start = max(1, period - 2)
                    window_end = min(len(acf), period + 3)
                    window_acf = acf[window_start:window_end]
                    
                    if len(window_acf) > 0:
                        max_acf = np.max(np.abs(window_acf))
                        period_strength += max_acf
                        if max_acf > 0.3:  # Threshold for significant peak
                            peaks_found += 1
            
            # Normalize by number of periods checked
            if len(common_periods) > 0:
                period_strength /= len(common_periods)
            
            # Bonus for finding multiple periods (indicates strong seasonality)
            if peaks_found >= 2:
                period_strength *= 1.2
            
            return min(1.0, period_strength)
            
        except Exception:
            return 0.0
    
    def _calculate_stationarity_improvement(self, original, residuals):
        """
        Calculate improvement in stationarity after deseasonalization.
        
        Uses simple metrics:
        - Reduction in trend strength (mean absolute difference)
        - Reduction in variance of rolling mean
        
        Returns value in [0, 1] where higher means better stationarity improvement.
        """
        if len(original) < 30:
            return 0.0
        
        try:
            # Calculate trend strength using first-order differences
            # More stationary series have differences closer to zero mean
            original_diff = np.diff(original)
            residual_diff = np.diff(residuals)
            
            # Measure 1: Variance of differences (should decrease for stationary)
            original_diff_var = np.var(original_diff)
            residual_diff_var = np.var(residual_diff)
            
            if original_diff_var > 0:
                var_improvement = max(0, (original_diff_var - residual_diff_var) / original_diff_var)
            else:
                var_improvement = 0.0
            
            # Measure 2: Rolling mean stability (more stationary = more stable rolling mean)
            window = min(30, len(original) // 10)
            if window >= 3:
                original_rolling_mean = pd.Series(original).rolling(window=window, center=True, min_periods=1).mean().values
                residual_rolling_mean = pd.Series(residuals).rolling(window=window, center=True, min_periods=1).mean().values
                
                # Variance of rolling mean (should be lower for stationary)
                original_rolling_var = np.var(original_rolling_mean)
                residual_rolling_var = np.var(residual_rolling_mean)
                
                if original_rolling_var > 0:
                    rolling_improvement = max(0, (original_rolling_var - residual_rolling_var) / original_rolling_var)
                else:
                    rolling_improvement = 0.0
            else:
                rolling_improvement = 0.0
            
            # Combine measures
            stationarity_improvement = 0.5 * var_improvement + 0.5 * rolling_improvement
            
            return min(1.0, stationarity_improvement)
            
        except Exception:
            return 0.0
    
    def _refine_seasonality(self, df_no_anomalies):
        """Refine seasonality model after anomaly removal."""
        self.refined_seasonality_model = DatepartRegressionTransformer(**self.seasonality_params)
        self.refined_seasonality_model.fit(df_no_anomalies)
    
    def _detect_anomalies(self, df):
        """Detect anomalies."""
        self.anomaly_detector = AnomalyRemoval(**self.anomaly_params)
        self.anomaly_detector.fit(df)
        
        # Store anomalies per series
        for col in df.columns:
            anomaly_mask = self.anomaly_detector.anomalies[col] == -1
            anomaly_dates = df.index[anomaly_mask].tolist()
            
            # Get magnitudes and types
            anomalies_list = []
            for date in anomaly_dates:
                magnitude = df.loc[date, col]
                score = self.anomaly_detector.scores.loc[date, col]
                
                # Determine type based on pattern
                anomaly_type = self._classify_anomaly_type(df, col, date)
                
                anomalies_list.append({
                    'date': date,
                    'magnitude': magnitude,
                    'score': score,
                    'type': anomaly_type,
                })
            
            self.anomalies[col] = anomalies_list
    
    def _classify_anomaly_type(self, df, col, date):
        """Classify anomaly type based on surrounding pattern."""
        idx = df.index.get_loc(date)
        
        # Check for decay pattern (next few values trending back to normal)
        if idx < len(df) - 3:
            post_values = df[col].iloc[idx:idx+4].values
            if len(post_values) >= 4:
                # Simple decay detection: values monotonically approaching mean
                mean_val = df[col].mean()
                diffs_from_mean = np.abs(post_values - mean_val)
                if np.all(np.diff(diffs_from_mean) < 0):
                    return 'decay'
        
        # Check for ramp pattern
        if idx > 2:
            pre_values = df[col].iloc[idx-3:idx+1].values
            if len(pre_values) >= 4:
                # Check if ramping up to spike
                if np.all(np.diff(pre_values) > 0):
                    return 'ramp_up'
                    
        return 'spike'
    
    def _detect_holidays(self, df):
        """Detect holiday effects."""
        # HolidayDetector expects specific format
        try:
            self.holiday_detector = HolidayDetector(**self.holiday_params)
            self.holiday_detector.detect(df)
            
            # Extract detected holidays
            if hasattr(self.holiday_detector, 'dates_to_holidays'):
                for col in df.columns:
                    if col in self.holiday_detector.dates_to_holidays:
                        self.holiday_dates[col] = list(
                            self.holiday_detector.dates_to_holidays[col].keys()
                        )
                        self.holiday_impacts[col] = self.holiday_detector.dates_to_holidays[col]
                    else:
                        self.holiday_dates[col] = []
                        self.holiday_impacts[col] = {}
        except Exception as e:
            # If holiday detection fails, continue without it
            self.holiday_detector = None
            for col in df.columns:
                self.holiday_dates[col] = []
                self.holiday_impacts[col] = {}
    
    def _detect_level_shifts(self, df):
        """Detect level shifts."""
        self.level_shift_detector = LevelShiftMagic(**self.level_shift_params)
        self.level_shift_detector.fit(df)
        
        # Extract level shift information
        # LevelShiftMagic stores cumulative shifts in self.lvlshft (DataFrame)
        # We need to find where the values change to identify shift dates
        if hasattr(self.level_shift_detector, 'lvlshft'):
            for col in df.columns:
                series_shifts = []
                lvlshft_series = self.level_shift_detector.lvlshft[col]
                
                # Find where the shift values change (indicating a new level shift)
                # Use diff() to find changes, and filter out tiny floating point differences
                shifts = lvlshft_series.diff().abs()
                shift_threshold = 0.01  # Threshold for detecting meaningful changes
                
                shift_dates = shifts[shifts > shift_threshold].index.tolist()
                shift_magnitudes = lvlshft_series.diff()[shifts > shift_threshold].tolist()
                
                for date, mag in zip(shift_dates, shift_magnitudes):
                    series_shifts.append({
                        'date': date,
                        'magnitude': mag
                    })
                
                self.level_shifts[col] = series_shifts
        else:
            for col in df.columns:
                self.level_shifts[col] = []
    
    def _detect_changepoints(self, df):
        """Detect trend changepoints."""
        self.changepoint_detector = ChangePointDetector(**self.changepoint_params)
        self.changepoint_detector.detect(df)

        raw_changepoints = getattr(self.changepoint_detector, 'changepoints_', None)

        if isinstance(raw_changepoints, dict):
            per_series_changepoints = raw_changepoints
        elif raw_changepoints is not None:
            shared_indices = self._sanitize_changepoint_indices(raw_changepoints, len(df))
            per_series_changepoints = {col: shared_indices for col in df.columns}
        else:
            per_series_changepoints = {}

        for col in df.columns:
            series = df[col].astype(float).values
            cp_indices = self._sanitize_changepoint_indices(
                per_series_changepoints.get(col, []), len(series)
            )

            changepoints_list = []
            for cp_idx in cp_indices:
                pre_window_start = max(0, cp_idx - 10)
                post_window_end = min(len(series), cp_idx + 10)
                if post_window_end <= cp_idx or cp_idx <= pre_window_start or cp_idx >= len(series) - 1:
                    continue

                date = df.index[cp_idx]
                pre_slope = self._estimate_slope(series, pre_window_start, cp_idx)
                post_slope = self._estimate_slope(series, cp_idx, post_window_end)

                changepoints_list.append({
                    'date': date,
                    'old_slope': pre_slope,
                    'new_slope': post_slope,
                    'magnitude': abs(post_slope - pre_slope),
                })

            self.trend_changepoints[col] = changepoints_list

    def _sanitize_changepoint_indices(self, indices, series_length):
        """Convert changepoint indices to sorted unique integer positions within bounds."""
        if indices is None:
            return []

        if isinstance(indices, (np.ndarray, pd.Index)):
            iterable = indices.tolist()
        elif isinstance(indices, (list, tuple, set)):
            iterable = list(indices)
        else:
            iterable = [indices]

        cleaned = []
        seen = set()
        for idx in iterable:
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue

            if 0 < idx_int < series_length and idx_int not in seen:
                seen.add(idx_int)
                cleaned.append(idx_int)

        cleaned.sort()
        return cleaned
    
    def _estimate_slope(self, series, start_idx, end_idx):
        """Estimate slope of a segment using linear regression."""
        if end_idx <= start_idx:
            return 0.0
        
        x = np.arange(start_idx, end_idx)
        y = series[start_idx:end_idx]
        
        if len(x) < 2:
            return 0.0
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0.0
        x = x[mask]
        y = y[mask]
        
        # Simple linear regression
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_detected_features(self, series_name=None):
        """
        Get all detected features for a series or all series.
        
        Parameters
        ----------
        series_name : str, optional
            Name of series to get features for. If None, returns all.
            
        Returns
        -------
        dict
            Dictionary of detected features
        """
        if series_name is not None:
            return {
                'trend_changepoints': self.trend_changepoints.get(series_name, []),
                'level_shifts': self.level_shifts.get(series_name, []),
                'anomalies': self.anomalies.get(series_name, []),
                'holiday_dates': self.holiday_dates.get(series_name, []),
                'holiday_impacts': self.holiday_impacts.get(series_name, {}),
                'seasonality_strength': self.seasonality_strength.get(series_name, 0.0),
            }
        else:
            return {
                'trend_changepoints': self.trend_changepoints,
                'level_shifts': self.level_shifts,
                'anomalies': self.anomalies,
                'holiday_dates': self.holiday_dates,
                'holiday_impacts': self.holiday_impacts,
                'seasonality_strength': self.seasonality_strength,
            }
    
    def summary(self):
        """Print summary of detected features."""
        print("=" * 80)
        print("TIME SERIES FEATURE DETECTION SUMMARY")
        print("=" * 80)
        print(f"Date Range: {self.date_index[0]} to {self.date_index[-1]}")
        print(f"Number of Series: {len(self.df_original.columns)}")
        print(f"Number of Observations: {len(self.df_original)}")
        print()
        
        for series_name in self.df_original.columns:
            print(f"\n{'-' * 80}")
            print(f"Series: {series_name}")
            print(f"{'-' * 80}")
            
            # Seasonality
            strength = self.seasonality_strength.get(series_name, 0.0)
            print(f"\nSeasonality Strength: {strength:.3f}")
            
            # Changepoints
            changepoints = self.trend_changepoints.get(series_name, [])
            print(f"\nTrend Changepoints: {len(changepoints)}")
            for i, cp in enumerate(changepoints[:5]):  # Show first 5
                print(f"  {i+1}. {cp['date']}: slope {cp['old_slope']:.3f} → {cp['new_slope']:.3f}")
            if len(changepoints) > 5:
                print(f"  ... and {len(changepoints) - 5} more")
            
            # Level Shifts
            shifts = self.level_shifts.get(series_name, [])
            print(f"\nLevel Shifts: {len(shifts)}")
            for i, shift in enumerate(shifts[:5]):
                print(f"  {i+1}. {shift['date']}: magnitude {shift['magnitude']:.3f}")
            if len(shifts) > 5:
                print(f"  ... and {len(shifts) - 5} more")
            
            # Anomalies
            anomalies = self.anomalies.get(series_name, [])
            print(f"\nAnomalies: {len(anomalies)}")
            for i, anom in enumerate(anomalies[:5]):
                print(f"  {i+1}. {anom['date']}: {anom['type']}, magnitude {anom['magnitude']:.3f}")
            if len(anomalies) > 5:
                print(f"  ... and {len(anomalies) - 5} more")
            
            # Holidays
            holidays = self.holiday_dates.get(series_name, [])
            print(f"\nDetected Holidays: {len(holidays)}")
            for i, hol in enumerate(holidays[:5]):
                print(f"  {i+1}. {hol}")
            if len(holidays) > 5:
                print(f"  ... and {len(holidays) - 5} more")
        
        print("\n" + "=" * 80)
    
    def plot(self, series_name=None, figsize=(16, 12), save_path=None, show=True):
        """
        Plot detected features.
        
        Parameters
        ----------
        series_name : str, optional
            Name of series to plot. If None, plots first series.
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        show : bool
            Whether to show the plot
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return
        
        if series_name is None:
            series_name = self.df_original.columns[0]
        
        series = self.df_original[series_name]
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Original data with anomalies
        ax = axes[0]
        ax.plot(series.index, series.values, label='Original', alpha=0.7)
        
        # Mark anomalies
        anomalies = self.anomalies.get(series_name, [])
        for anom in anomalies:
            color = 'red' if anom['type'] == 'spike' else 'orange'
            ax.axvline(anom['date'], color=color, alpha=0.3, linestyle='--', linewidth=1)
            ax.scatter(anom['date'], anom['magnitude'], color=color, s=50, zorder=5)
        
        ax.set_ylabel('Value')
        ax.set_title(f'Series: {series_name} - Original Data with Anomalies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Seasonality component
        ax = axes[1]
        if series_name in self.seasonality_components:
            seasonal = self.seasonality_components[series_name]
            ax.plot(series.index, seasonal, label='Seasonal Component', color='green')
            strength = self.seasonality_strength.get(series_name, 0.0)
            ax.set_title(f'Detected Seasonality (Strength: {strength:.3f})')
        else:
            ax.set_title('Seasonality (Not Available)')
        ax.set_ylabel('Seasonal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Changepoints and level shifts
        ax = axes[2]
        try:
            deseasonalized = self.seasonality_model.transform(
                pd.DataFrame(series).astype(float)
            )[series_name]
        except Exception:
            # If transform fails, use original data
            deseasonalized = series
        ax.plot(series.index, deseasonalized.values, label='Deseasonalized', alpha=0.7)
        
        # Mark changepoints
        changepoints = self.trend_changepoints.get(series_name, [])
        for cp in changepoints:
            ax.axvline(cp['date'], color='blue', alpha=0.5, linestyle='--', linewidth=2)
        
        # Mark level shifts
        level_shifts = self.level_shifts.get(series_name, [])
        for shift in level_shifts:
            ax.axvline(shift['date'], color='purple', alpha=0.5, linestyle=':', linewidth=2)
        
        ax.set_ylabel('Value')
        ax.set_title('Trend Changepoints (blue) and Level Shifts (purple)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Holidays
        ax = axes[3]
        ax.plot(series.index, series.values, label='Original', alpha=0.5, color='gray')
        
        # Mark holidays
        holidays = self.holiday_dates.get(series_name, [])
        for hol in holidays:
            ax.axvline(hol, color='magenta', alpha=0.4, linestyle='-.', linewidth=1.5)
        
        ax.set_ylabel('Value')
        ax.set_xlabel('Date')
        ax.set_title(f'Detected Holidays ({len(holidays)} found)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


class FeatureDetectionLoss:
    """
    Calculate loss for feature detection optimization.
    
    Compares detected features against ground truth labels from synthetic data.
    Implements sophisticated loss calculation with:
    - Proximity-based scoring for changepoints
    - Asymmetric penalties (level shift as changepoint OK, reverse not OK)
    - Magnitude-weighted changepoint scoring
    - Anomaly type-specific handling
    - Holiday priority over anomaly classification
    """
    
    def __init__(
        self,
        changepoint_tolerance_days=7,
        level_shift_tolerance_days=3,
        anomaly_tolerance_days=1,
        holiday_tolerance_days=0,
        min_changepoint_magnitude_penalty=0.01,
        holiday_over_anomaly_bonus=0.5,
    ):
        self.changepoint_tolerance_days = changepoint_tolerance_days
        self.level_shift_tolerance_days = level_shift_tolerance_days
        self.anomaly_tolerance_days = anomaly_tolerance_days
        self.holiday_tolerance_days = holiday_tolerance_days
        self.min_changepoint_magnitude_penalty = min_changepoint_magnitude_penalty
        self.holiday_over_anomaly_bonus = holiday_over_anomaly_bonus
    
    def calculate_loss(self, detected_features, true_labels, series_name=None):
        """
        Calculate overall loss comparing detected features to true labels.
        
        Parameters
        ----------
        detected_features : dict
            Features from TimeSeriesFeatureDetector
        true_labels : dict
            Ground truth labels from SyntheticDailyGenerator
        series_name : str, optional
            Specific series to evaluate. If None, evaluates all.
            
        Returns
        -------
        dict
            Loss breakdown by component and overall loss
        """
        if series_name is not None:
            return self._calculate_series_loss(
                detected_features, true_labels, series_name
            )
        else:
            # Aggregate across all series
            total_loss = 0.0
            loss_breakdown = {
                'changepoint_loss': 0.0,
                'level_shift_loss': 0.0,
                'anomaly_loss': 0.0,
                'holiday_loss': 0.0,
                'total_loss': 0.0,
            }
            
            series_names = list(detected_features.get('trend_changepoints', {}).keys())
            
            for sname in series_names:
                series_loss = self._calculate_series_loss(
                    detected_features, true_labels, sname
                )
                for key in loss_breakdown:
                    loss_breakdown[key] += series_loss[key]
            
            # Average across series
            n_series = len(series_names) if series_names else 1
            for key in loss_breakdown:
                loss_breakdown[key] /= n_series
            
            return loss_breakdown
    
    def _calculate_series_loss(self, detected_features, true_labels, series_name):
        """Calculate loss for a single series."""
        loss = {
            'changepoint_loss': 0.0,
            'level_shift_loss': 0.0,
            'anomaly_loss': 0.0,
            'holiday_loss': 0.0,
            'total_loss': 0.0,
        }
        
        # Extract features for this series
        detected_cp = detected_features['trend_changepoints'].get(series_name, [])
        detected_ls = detected_features['level_shifts'].get(series_name, [])
        detected_anom = detected_features['anomalies'].get(series_name, [])
        detected_hol = detected_features['holiday_dates'].get(series_name, [])
        
        true_cp = true_labels.get('trend_changepoints', {}).get(series_name, [])
        true_ls = true_labels.get('level_shifts', {}).get(series_name, [])
        true_anom = true_labels.get('anomalies', {}).get(series_name, [])
        true_hol = true_labels.get('holiday_dates', {}).get(series_name, [])
        
        # Changepoint loss
        loss['changepoint_loss'] = self._changepoint_loss(detected_cp, true_cp, detected_ls, true_ls)
        
        # Level shift loss
        loss['level_shift_loss'] = self._level_shift_loss(detected_ls, true_ls, detected_cp)
        
        # Anomaly loss
        loss['anomaly_loss'] = self._anomaly_loss(detected_anom, true_anom)
        
        # Holiday loss
        loss['holiday_loss'] = self._holiday_loss(detected_hol, true_hol, detected_anom)
        
        # Total loss (weighted sum)
        loss['total_loss'] = (
            loss['changepoint_loss'] +
            loss['level_shift_loss'] +
            loss['anomaly_loss'] +
            loss['holiday_loss']
        )
        
        return loss
    
    def _changepoint_loss(self, detected_cp, true_cp, detected_ls, true_ls):
        """
        Calculate changepoint detection loss.
        
        - Rewards finding changepoints near true changepoints
        - Penalizes missed changepoints based on magnitude
        - Doesn't penalize if detected as level shift instead
        """
        if len(true_cp) == 0:
            return 0.0 if len(detected_cp) == 0 else len(detected_cp) * 0.5
        
        loss = 0.0
        true_cp_dates = [cp['date'] if isinstance(cp, dict) else cp[0] for cp in true_cp]
        detected_cp_dates = [cp['date'] if isinstance(cp, dict) else cp for cp in detected_cp]
        detected_ls_dates = [ls['date'] if isinstance(ls, dict) else (ls[0] if isinstance(ls, tuple) else ls) for ls in detected_ls]
        
        # For each true changepoint
        for true_item in true_cp:
            if isinstance(true_item, dict):
                true_date = true_item['date']
                true_mag = true_item.get('magnitude', abs(true_item.get('new_slope', 0) - true_item.get('old_slope', 0)))
            elif isinstance(true_item, (tuple, list)) and len(true_item) >= 3:
                # (date, old_slope, new_slope) format
                true_date = true_item[0]
                try:
                    true_mag = abs(float(true_item[2]) - float(true_item[1]))
                except (ValueError, TypeError):
                    true_mag = 1.0
            else:
                # Unknown format or insufficient data, skip
                continue
            
            # Find closest detected changepoint
            min_dist = float('inf')
            for det_date in detected_cp_dates:
                dist = abs((det_date - true_date).days)
                min_dist = min(min_dist, dist)
            
            # Check if detected as level shift instead
            ls_found = False
            for det_ls_date in detected_ls_dates:
                if abs((det_ls_date - true_date).days) <= self.level_shift_tolerance_days:
                    ls_found = True
                    break
            
            # Calculate penalty
            if ls_found:
                # No penalty if found as level shift
                penalty = 0.0
            elif min_dist <= self.changepoint_tolerance_days:
                # Proximity-based reward: closer is better
                penalty = (min_dist / self.changepoint_tolerance_days) * max(true_mag, self.min_changepoint_magnitude_penalty)
            else:
                # Missed: full penalty based on magnitude
                penalty = max(true_mag, self.min_changepoint_magnitude_penalty)
            
            loss += penalty
        
        # Penalize false positives (detected but not true)
        for det_date in detected_cp_dates:
            found = False
            for true_date in true_cp_dates:
                if abs((det_date - true_date).days) <= self.changepoint_tolerance_days:
                    found = True
                    break
            if not found:
                loss += 0.5  # False positive penalty
        
        return loss
    
    def _level_shift_loss(self, detected_ls, true_ls, detected_cp):
        """
        Calculate level shift detection loss.
        
        - Penalizes detecting changepoint as level shift when not a level shift
        - Rewards correct level shift detection
        """
        if len(true_ls) == 0:
            # If detected level shift but no true level shift, check if it's actually a changepoint
            # This would be penalized in changepoint_loss
            return 0.0
        
        loss = 0.0
        true_ls_dates = [ls['date'] if isinstance(ls, dict) else (ls[0] if isinstance(ls, (tuple, list)) else ls) for ls in true_ls]
        detected_ls_dates = [ls['date'] if isinstance(ls, dict) else (ls if not isinstance(ls, (tuple, list)) else ls[0]) for ls in detected_ls]
        detected_cp_dates = [cp['date'] if isinstance(cp, dict) else (cp if not isinstance(cp, (tuple, list)) else cp[0]) for cp in detected_cp]
        
        # For each true level shift
        for true_item in true_ls:
            if isinstance(true_item, dict):
                true_date = true_item['date']
                true_mag = abs(true_item.get('magnitude', 1.0))
            elif isinstance(true_item, (tuple, list)) and len(true_item) >= 2:
                true_date = true_item[0]
                try:
                    true_mag = abs(float(true_item[1]))
                except (ValueError, TypeError):
                    true_mag = 1.0
            else:
                continue
            
            # Find closest detected level shift
            min_dist_ls = float('inf')
            for det_date in detected_ls_dates:
                dist = abs((det_date - true_date).days)
                min_dist_ls = min(min_dist_ls, dist)
            
            # Also check changepoints (OK if found as changepoint)
            min_dist_cp = float('inf')
            for det_date in detected_cp_dates:
                dist = abs((det_date - true_date).days)
                min_dist_cp = min(min_dist_cp, dist)
            
            # Calculate penalty
            if min_dist_ls <= self.level_shift_tolerance_days:
                # Found as level shift
                penalty = (min_dist_ls / self.level_shift_tolerance_days) * true_mag * 0.5
            elif min_dist_cp <= self.changepoint_tolerance_days:
                # Found as changepoint (acceptable)
                penalty = 0.0
            else:
                # Missed
                penalty = true_mag
            
            loss += penalty
        
        return loss
    
    def _anomaly_loss(self, detected_anom, true_anom):
        """
        Calculate anomaly detection loss.
        
        - Focuses on spike detection
        - More lenient on decay/ramp patterns
        """
        if len(true_anom) == 0:
            return 0.0 if len(detected_anom) == 0 else len(detected_anom) * 0.3
        
        loss = 0.0
        detected_dates = [a['date'] if isinstance(a, dict) else (a[0] if isinstance(a, (tuple, list)) else a) for a in detected_anom]
        
        # For each true anomaly
        for true_item in true_anom:
            if isinstance(true_item, dict):
                true_date = true_item['date']
                true_type = true_item.get('type', 'spike')
                true_mag = abs(true_item.get('magnitude', 1.0))
            elif isinstance(true_item, (tuple, list)):
                true_date = true_item[0]
                # (date, magnitude, type, duration, shared) or (date, magnitude, type, duration)
                true_type = true_item[2] if len(true_item) > 2 else 'spike'
                try:
                    true_mag = abs(float(true_item[1])) if len(true_item) > 1 else 1.0
                except (ValueError, TypeError):
                    true_mag = 1.0
            else:
                continue
            
            # Find closest detected anomaly
            min_dist = float('inf')
            for det_date in detected_dates:
                dist = abs((det_date - true_date).days)
                min_dist = min(min_dist, dist)
            
            # Calculate penalty based on type
            if min_dist <= self.anomaly_tolerance_days:
                # Found
                if true_type in ['decay', 'ramp_up', 'ramp_down']:
                    # More lenient for complex patterns
                    penalty = 0.2
                else:
                    # Spike: should be found exactly
                    penalty = 0.0 if min_dist == 0 else 0.3
            else:
                # Missed
                if true_type in ['decay', 'ramp_up', 'ramp_down']:
                    penalty = 0.5  # Less penalty for missing complex patterns
                else:
                    penalty = 1.0  # Full penalty for missing spike
            
            loss += penalty
        
        return loss
    
    def _holiday_loss(self, detected_hol, true_hol, detected_anom):
        """
        Calculate holiday detection loss.
        
        - Holiday detection has priority over anomaly
        - Bonus if detected as anomaly when not detected as holiday
        """
        if len(true_hol) == 0:
            return 0.0
        
        loss = 0.0
        detected_anom_dates = [a['date'] if isinstance(a, dict) else a[0] for a in detected_anom]
        
        # For each true holiday
        for true_date in true_hol:
            # Check if detected as holiday
            found_as_holiday = any(
                abs((det_date - true_date).days) <= self.holiday_tolerance_days
                for det_date in detected_hol
            )
            
            # Check if detected as anomaly
            found_as_anomaly = any(
                abs((det_date - true_date).days) <= self.anomaly_tolerance_days
                for det_date in detected_anom_dates
            )
            
            if found_as_holiday:
                # Perfect
                penalty = 0.0
            elif found_as_anomaly:
                # Better than nothing
                penalty = self.holiday_over_anomaly_bonus
            else:
                # Missed
                penalty = 1.0
            
            loss += penalty
        
        return loss


class FeatureDetectionOptimizer:
    """
    Optimize TimeSeriesFeatureDetector parameters using synthetic labeled data.
    
    Uses optimization to find parameters that minimize detection loss.
    """
    
    def __init__(
        self,
        synthetic_generator,
        loss_calculator=None,
        optimization_method='grid_search',
        n_iterations=50,
        random_seed=42,
    ):
        """
        Parameters
        ----------
        synthetic_generator : SyntheticDailyGenerator
            Generator with labeled synthetic data
        loss_calculator : FeatureDetectionLoss, optional
            Custom loss calculator
        optimization_method : str
            'grid_search', 'random_search', or 'bayesian'
        n_iterations : int
            Number of optimization iterations
        random_seed : int
            Random seed for reproducibility
        """
        self.synthetic_generator = synthetic_generator
        self.loss_calculator = loss_calculator or FeatureDetectionLoss()
        self.optimization_method = optimization_method
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        self.best_params = None
        self.best_loss = float('inf')
        self.optimization_history = []
        self.baseline_loss = None
    
    def optimize(self):
        """
        Run optimization to find best detector parameters.
        
        Returns
        -------
        dict
            Best parameters found
        """
        self.best_params = None
        self.best_loss = float('inf')
        self.optimization_history = []
        self.baseline_loss = None

        if self.optimization_method == 'random_search':
            return self._random_search()
        elif self.optimization_method == 'grid_search':
            return self._grid_search()
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

    def _default_detector_params(self):
        """Return a deep-copied set of default detector parameters."""
        detector = TimeSeriesFeatureDetector()
        return {
            'seasonality_params': copy.deepcopy(detector.seasonality_params),
            'holiday_params': copy.deepcopy(detector.holiday_params),
            'anomaly_params': copy.deepcopy(detector.anomaly_params),
            'changepoint_params': copy.deepcopy(detector.changepoint_params),
            'level_shift_params': copy.deepcopy(detector.level_shift_params),
            'standardize': detector.standardize,
            'smoothing_window': detector.smoothing_window,
            'refine_seasonality': detector.refine_seasonality,
        }
    
    def _random_search(self):
        """Random search optimization."""
        print(f"Starting random search optimization ({self.n_iterations} iterations)...")

        baseline_params = self._default_detector_params()
        try:
            baseline_loss = self._evaluate_params(baseline_params)
            self.baseline_loss = baseline_loss['total_loss']
            self.best_params = copy.deepcopy(baseline_params)
            self.best_loss = self.baseline_loss
            print(f"Baseline loss = {self.best_loss:.4f}")
        except Exception as e:
            print(f"Warning: Baseline evaluation failed with error: {e}")
            print(f"Baseline params: {baseline_params}")
            self.baseline_loss = float('inf')
            self.best_params = baseline_params
            self.best_loss = float('inf')
        
        for i in range(self.n_iterations):
            # Generate random parameters
            params = self._sample_random_params()
            
            # Evaluate
            try:
                loss = self._evaluate_params(params)
                
                # Track
                self.optimization_history.append({
                    'iteration': i,
                    'params': copy.deepcopy(params),
                    'loss': loss['total_loss'],
                    'loss_breakdown': loss,
                    'status': 'success',
                })
                
                # Update best
                if loss['total_loss'] < self.best_loss:
                    self.best_loss = loss['total_loss']
                    self.best_params = copy.deepcopy(params)
                    print(f"Iteration {i}: New best loss = {self.best_loss:.4f}")
            except Exception as e:
                print(f"Iteration {i} failed with error: {e}")
                print(f"Failed params: {params}")
                # Track failure for transparency
                self.optimization_history.append({
                    'iteration': i,
                    'params': copy.deepcopy(params),
                    'loss': float('inf'),
                    'loss_breakdown': None,
                    'status': 'error',
                    'error': repr(e),
                })
                # Continue to next iteration
                continue
        
        print(f"\nOptimization complete!")
        print(f"Best loss: {self.best_loss:.4f}")
        return self.best_params
    
    def _grid_search(self):
        """Grid search over predefined parameter grid."""
        # Define a coarse grid
        grid = self._create_parameter_grid()
        
        print(f"Starting grid search optimization ({len(grid)} combinations)...")

        baseline_params = self._default_detector_params()
        try:
            baseline_loss = self._evaluate_params(baseline_params)
            self.baseline_loss = baseline_loss['total_loss']
            self.best_params = copy.deepcopy(baseline_params)
            self.best_loss = self.baseline_loss
            print(f"Baseline loss = {self.best_loss:.4f}")
        except Exception as e:
            print(f"Warning: Baseline evaluation failed with error: {e}")
            print(f"Baseline params: {baseline_params}")
            self.baseline_loss = float('inf')
            self.best_params = baseline_params
            self.best_loss = float('inf')
        
        for i, params in enumerate(grid):
            try:
                loss = self._evaluate_params(params)
                
                self.optimization_history.append({
                    'iteration': i,
                    'params': copy.deepcopy(params),
                    'loss': loss['total_loss'],
                    'loss_breakdown': loss,
                    'status': 'success',
                })
                
                if loss['total_loss'] < self.best_loss:
                    self.best_loss = loss['total_loss']
                    self.best_params = copy.deepcopy(params)
                    print(f"Grid {i}: New best loss = {self.best_loss:.4f}")
            except Exception as e:
                print(f"Grid {i} failed with error: {e}")
                print(f"Failed params: {params}")
                self.optimization_history.append({
                    'iteration': i,
                    'params': copy.deepcopy(params),
                    'loss': float('inf'),
                    'loss_breakdown': None,
                    'status': 'error',
                    'error': repr(e),
                })
                # Continue to next grid point
                continue
        
        print(f"\nOptimization complete!")
        print(f"Best loss: {self.best_loss:.4f}")
        return self.best_params
    
    def _evaluate_params(self, params):
        """Evaluate a parameter configuration."""
        # Create detector with these params
        detector = TimeSeriesFeatureDetector(**params)
        
        # Fit on synthetic data
        detector.fit(self.synthetic_generator.get_data())
        
        # Get detected features
        detected_features = detector.get_detected_features()
        
        # Get true labels
        true_labels = self.synthetic_generator.get_all_labels()
        
        # Calculate loss
        loss = self.loss_calculator.calculate_loss(detected_features, true_labels)
        
        return loss
    
    def _sample_random_params(self):
        """Sample random parameters."""
        # Seasonality params
        seasonality_params = DatepartRegressionTransformer.get_new_params(method='random', holiday_countries_used=False)

        # Anomaly params
        method_choice, method_params, _ = anomaly_new_params(method='random')
        anomaly_params = {
            'output': 'multivariate',
            'method': method_choice,
            'method_params': method_params,
            'fillna': 'ffill',
        }
        
        # Changepoint params
        changepoint_params = ChangePointDetector().get_new_params(method='random')
        
        # Level shift params
        level_shift_params = LevelShiftMagic.get_new_params(method='random')
        
        return {
            'seasonality_params': seasonality_params,
            'holiday_params': copy.deepcopy(self._default_detector_params()['holiday_params']),
            'anomaly_params': anomaly_params,
            'changepoint_params': changepoint_params,
            'level_shift_params': level_shift_params,
            'standardize': self.rng.choice([True, False]),
            'smoothing_window': self.rng.choice([None, 3, 5, 7]),
            'refine_seasonality': self.rng.choice([True, False]),
        }
    
    def _create_parameter_grid(self):
        """Create a grid of parameter combinations."""
        grid = []
        
        # Simplified grid for demonstration
        for datepart in ['simple_2', 'common_fourier']:
            for model in ['DecisionTree', 'ElasticNet']:
                for penalty in [10, 20]:
                    for alpha in [2.0, 2.5]:
                        params = {
                            'seasonality_params': {
                                'regression_model': {'model': model, 'model_params': {}},
                                'datepart_method': datepart,
                                'polynomial_degree': None,
                                'holiday_countries_used': False,
                            },
                            'anomaly_params': {
                                'output': 'multivariate',
                                'method': 'zscore',
                                'method_params': {'alpha': 0.05},
                                'fillna': 'ffill',
                            },
                            'changepoint_params': {
                                'method': 'pelt',
                                'method_params': {'penalty': penalty},
                                'min_segment_length': 7,
                            },
                            'level_shift_params': {
                                'window_size': 90,
                                'alpha': alpha,
                                'grouping_forward_limit': 3,
                                'max_level_shifts': 20,
                            },
                            'standardize': True,
                            'smoothing_window': None,
                            'refine_seasonality': True,
                        }
                        grid.append(params)
        
        return grid
    
    def get_optimization_summary(self):
        """Print summary of optimization results."""
        print("=" * 80)
        print("OPTIMIZATION SUMMARY")
        print("=" * 80)
        print(f"Method: {self.optimization_method}")
        print(f"Iterations: {len(self.optimization_history)}")
        print(f"Best Loss: {self.best_loss:.4f}")
        if self.baseline_loss is not None:
            print(f"Baseline Loss: {self.baseline_loss:.4f}")
        print()
        print("Best Parameters:")
        print("-" * 80)
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print()
        
        if self.optimization_history:
            print("Loss History:")
            print("-" * 80)
            losses = [h['loss'] for h in self.optimization_history]
            print(f"  Initial: {losses[0]:.4f}")
            print(f"  Final: {losses[-1]:.4f}")
            print(f"  Best: {min(losses):.4f}")
            print(f"  Mean: {np.mean(losses):.4f}")
            print(f"  Std: {np.std(losses):.4f}")
        else:
            print("No optimization iterations were recorded.")
