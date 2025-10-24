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
from datetime import datetime
from autots.tools.transform import (
    DatepartRegressionTransformer,
    AnomalyRemoval,
    LevelShiftMagic,
    GeneralTransformer,
)
from autots.evaluator.anomaly_detector import HolidayDetector
from autots.tools.changepoints import ChangepointDetector
from autots.tools.anomaly_utils import anomaly_new_params
from autots.tools.plotting import plot_feature_panels, HAS_MATPLOTLIB
from sklearn.preprocessing import StandardScaler



class TimeSeriesFeatureDetector:
    """
    Comprehensive feature detection pipeline for univariate time series.

        Parameters
    ----------
    seasonality_params : dict, optional
        Parameters for DatepartRegressionTransformer used in final seasonality fit
    holiday_params : dict, optional
        Parameters for HolidayDetector
    anomaly_params : dict, optional
        Parameters for AnomalyRemoval
    changepoint_params : dict, optional
        Parameters for ChangepointDetector
    level_shift_params : dict, optional
        Parameters for LevelShiftMagic
    level_shift_validation : dict, optional
        Validation parameters for level shifts
    general_transformer_params : dict, optional
        Parameters for GeneralTransformer applied before trend detection
    smoothing_window : int, optional
        Window size for smoothing before trend detection
    refine_seasonality : bool, default=True
        Whether to refine seasonality after anomaly removal
    standardize : bool, default=True
        Whether to standardize series before processing
    detection_mode : str, default='multivariate'
        Controls whether detections are unique per series ('multivariate') or 
        shared across all series ('univariate'). 
        - 'multivariate': Each series gets unique anomalies, holidays, changepoints, and level shifts
        - 'univariate': All series share common anomalies, holidays, changepoints, and level shifts
          (level shifts are detected on aggregated signal and scaled appropriately per series)
    """
    
    TEMPLATE_VERSION = "1.0"

    def __init__(
        self,
        seasonality_params=None,
        holiday_params=None,
        anomaly_params=None,
        changepoint_params=None,
        level_shift_params=None,
        level_shift_validation=None,
        general_transformer_params=None,
        smoothing_window=None,
        refine_seasonality=True,
        standardize=True,
        detection_mode='multivariate',
    ):
        # Set detection_mode first so it can be used in other initializations
        self.detection_mode = detection_mode
        
        # Validate detection_mode
        if detection_mode not in ['multivariate', 'univariate']:
            raise ValueError(
                f"detection_mode must be 'multivariate' or 'univariate', got '{detection_mode}'"
            )
        
        self.rough_seasonality_params = {
            'regression_model': {
                'model': 'ElasticNet',
                'model_params': {'l1_ratio': 0.2},
            },
            'datepart_method': 'simple_3',
            'polynomial_degree': None,
            'holiday_countries_used': False,
        }
        self.seasonality_params = seasonality_params or {
            'regression_model': {
                'model': 'BayesianMultiOutputRegression',
                'model_params': {},
            },
            'datepart_method': 'common_fourier',
            'polynomial_degree': None,
            'holiday_countries_used': False,
        }
        self.holiday_params = self._sanitize_holiday_params(holiday_params)
        # Ensure anomaly_params uses the correct output mode
        if anomaly_params is None:
            self.anomaly_params = {
                'output': self.detection_mode,
                'method': 'zscore',
                'method_params': {'distribution': 'norm', 'alpha': 0.01},
                'transform_dict': None,
                'fillna': 'ffill',
            }
        else:
            self.anomaly_params = anomaly_params.copy()
            # Override output to match detection_mode
            self.anomaly_params['output'] = self.detection_mode
        # Ensure changepoint_params uses the correct aggregate_method
        if changepoint_params is None:
            # Map detection_mode to aggregate_method: 
            # 'multivariate' -> 'individual' (each series separate)
            # 'univariate' -> 'mean' or 'median' (aggregate across series)
            aggregate_method = 'individual' if self.detection_mode == 'multivariate' else 'mean'
            self.changepoint_params = {
                'method': 'pelt',
                'method_params': {'penalty': 8, 'loss_function': 'l2'},
                'aggregate_method': aggregate_method,
                'min_segment_length': 14,
            }
        else:
            self.changepoint_params = changepoint_params.copy()
            # Override aggregate_method to match detection_mode if not explicitly set
            if 'aggregate_method' not in self.changepoint_params or self.changepoint_params['aggregate_method'] == 'auto':
                aggregate_method = 'individual' if self.detection_mode == 'multivariate' else 'mean'
                self.changepoint_params['aggregate_method'] = aggregate_method
        
        # Ensure level_shift_params uses the correct output mode
        if level_shift_params is None:
            self.level_shift_params = {
                'window_size': 90,
                'alpha': 2.5,
                'grouping_forward_limit': 3,
                'max_level_shifts': 20,
                'alignment': 'average',
                'output': self.detection_mode,
            }
        else:
            self.level_shift_params = level_shift_params.copy()
            # Override output to match detection_mode
            self.level_shift_params['output'] = self.detection_mode
        self.level_shift_validation = level_shift_validation or {
            'window': 14,
            'pad': 2,
            'relative_threshold': 0.1,
            'absolute_threshold': 0.5,
        }
        self.general_transformer_params = general_transformer_params
        self.smoothing_window = smoothing_window
        self.refine_seasonality = refine_seasonality
        self.standardize = standardize

        # Model artifacts
        self.scaler = None
        self.scale_series = None
        self.mean_series = None
        self.rough_seasonality_model = None
        self.seasonality_model = None
        self.holiday_detector = None
        self.anomaly_detector = None
        self.level_shift_detector = None
        self.changepoint_detector = None

        # Stored data and results
        self.df_original = None
        self.date_index = None
        self.template = None
        self.components = {}

        self.trend_changepoints = {}
        self.trend_slopes = {}
        self.level_shifts = {}
        self.anomalies = {}
        self.holiday_impacts = {}
        self.holiday_dates = {}
        self.seasonality_components = {}
        self.seasonality_strength = {}
        self.noise_changepoints = {}
        self.noise_to_signal_ratios = {}

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
            'output': self.detection_mode,  # Use instance's detection_mode
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
        
        # Override output to match detection_mode
        sanitized['output'] = self.detection_mode

        return sanitized

    def fit(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        df_numeric = df.astype(float).copy().sort_index()
        self.df_original = df_numeric
        self.date_index = df_numeric.index

        if self.standardize:
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(df_numeric)
            df_work = pd.DataFrame(scaled, index=self.date_index, columns=df_numeric.columns)
            self.scale_series = pd.Series(self.scaler.scale_, index=df_numeric.columns)
            self.mean_series = pd.Series(self.scaler.mean_, index=df_numeric.columns)
        else:
            self.scaler = None
            df_work = df_numeric.copy()
            self.scale_series = pd.Series(1.0, index=df_numeric.columns)
            self.mean_series = pd.Series(0.0, index=df_numeric.columns)

        # Reset result containers
        self.template = {
            'version': self.TEMPLATE_VERSION,
            'meta': {
                'start_date': self.date_index[0].isoformat(),
                'end_date': self.date_index[-1].isoformat(),
                'n_days': int(len(self.date_index)),
                'n_series': int(df_numeric.shape[1]),
                'frequency': pd.infer_freq(self.date_index) or 'infer',
                'created_at': pd.Timestamp.now().isoformat(),
                'detector_config': {
                    'standardize': self.standardize,
                    'refine_seasonality': self.refine_seasonality,
                    'smoothing_window': self.smoothing_window,
                    'detection_mode': self.detection_mode,
                },
            },
            'series': {},
        }
        self.components = {}
        self.trend_changepoints = {}
        self.trend_slopes = {}
        self.level_shifts = {}
        self.anomalies = {}
        self.holiday_impacts = {}
        self.holiday_dates = {}
        self.seasonality_components = {}
        self.seasonality_strength = {}
        self.noise_changepoints = {}
        self.noise_to_signal_ratios = {}

        rough_residual, rough_seasonality, self.rough_seasonality_model = self._compute_rough_seasonality(df_work)
        holiday_dates = self._detect_holidays(rough_residual)
        residual_without_anomalies, anomaly_records = self._detect_anomalies(rough_residual)
        df_without_anomalies = residual_without_anomalies + rough_seasonality
        final_residual, final_seasonality, seasonality_strength, self.seasonality_model = self._fit_final_seasonality(df_without_anomalies)
        holiday_component_scaled, holiday_impacts = self._estimate_holiday_impacts(final_residual, holiday_dates)
        residual_after_holidays = final_residual - holiday_component_scaled

        residual_for_trend = residual_after_holidays.copy()
        self.general_transformer = None
        if self.general_transformer_params:
            self.general_transformer = GeneralTransformer(**self.general_transformer_params)
            residual_for_trend = self.general_transformer.fit_transform(residual_after_holidays)
        if self.smoothing_window and self.smoothing_window > 1:
            residual_for_trend = residual_for_trend.rolling(
                window=int(self.smoothing_window),
                center=True,
                min_periods=1,
            ).mean()

        level_shift_component_scaled, level_shift_candidates = self._detect_level_shifts(residual_for_trend)
        level_shift_component_valid_scaled, validated_level_shifts = self._validate_level_shifts(
            residual_for_trend, level_shift_component_scaled, level_shift_candidates
        )

        trend_input = residual_for_trend - level_shift_component_valid_scaled
        changepoints, trend_component_scaled = self._detect_trend_changepoints(trend_input)
        slope_info = self._compute_trend_slopes(trend_component_scaled, changepoints)

        noise_component_scaled = trend_input - trend_component_scaled
        anomaly_component_scaled = df_work - df_without_anomalies

        trend_component = self._convert_to_original_scale(trend_component_scaled, include_mean=True)
        level_shift_component = self._convert_to_original_scale(level_shift_component_valid_scaled)
        seasonality_component = self._convert_to_original_scale(final_seasonality)
        holiday_component = self._convert_to_original_scale(holiday_component_scaled)
        noise_component = self._convert_to_original_scale(noise_component_scaled)
        anomaly_component = self._convert_to_original_scale(anomaly_component_scaled)

        holiday_impacts = self._rescale_holiday_impacts(holiday_impacts)
        validated_level_shifts = self._rescale_level_shifts(validated_level_shifts)
        slope_info = self._rescale_slope_info(slope_info)
        anomaly_records = self._rescale_anomalies(anomaly_records)

        for series_name in df_numeric.columns:
            components_dict = {
                'trend': trend_component[series_name].to_numpy(copy=True),
                'level_shift': level_shift_component[series_name].to_numpy(copy=True),
                'seasonality': seasonality_component[series_name].to_numpy(copy=True),
                'holidays': holiday_component[series_name].to_numpy(copy=True),
                'anomalies': anomaly_component[series_name].to_numpy(copy=True),
                'noise': noise_component[series_name].to_numpy(copy=True),
            }
            self.components[series_name] = components_dict

            trend_cp_entries, trend_cp_template = self._build_trend_label_entries(series_name, changepoints, slope_info)
            level_shift_entries, level_shift_template = self._build_level_shift_entries(series_name, validated_level_shifts)
            anomaly_entries, anomaly_template = self._build_anomaly_entries(series_name, anomaly_records)
            holidays_list = holiday_dates.get(series_name, [])
            holiday_template = holiday_impacts.get(series_name, {})

            self.trend_changepoints[series_name] = trend_cp_entries
            self.trend_slopes[series_name] = slope_info.get(series_name, [])
            self.level_shifts[series_name] = level_shift_entries
            self.anomalies[series_name] = anomaly_entries
            self.holiday_dates[series_name] = [pd.Timestamp(x) for x in holidays_list]
            self.holiday_impacts[series_name] = holiday_template
            self.seasonality_components[series_name] = seasonality_component[series_name].to_numpy(copy=True)
            self.seasonality_strength[series_name] = seasonality_strength.get(series_name, 0.0)
            self.noise_changepoints[series_name] = []

            noise_series = noise_component[series_name]
            signal_series = trend_component[series_name] + level_shift_component[series_name]
            numerator = float(np.nanstd(noise_series))
            denominator = float(np.nanstd(signal_series)) or 1e-9
            self.noise_to_signal_ratios[series_name] = numerator / denominator

            metadata = {
                'seasonality_strength': self.seasonality_strength[series_name],
                'noise_to_signal_ratio': self.noise_to_signal_ratios[series_name],
            }
            template_entry = self._build_series_template(
                series_name,
                components_dict,
                {
                    'trend_changepoints': trend_cp_template,
                    'level_shifts': level_shift_template,
                    'anomalies': anomaly_template,
                    'holiday_impacts': holiday_template,
                    'holiday_dates': self.holiday_dates[series_name],
                    'seasonality_changepoints': [],
                    'noise_changepoints': [],
                },
                metadata,
            )
            self.template['series'][series_name] = template_entry

        return self

    def _compute_rough_seasonality(self, df):
        model = DatepartRegressionTransformer(**self.rough_seasonality_params)
        residual = model.fit_transform(df)
        seasonal = df - residual
        return residual, seasonal, model

    def _detect_holidays(self, residual_df):
        """Detect holidays using HolidayDetector."""
        self.holiday_detector = HolidayDetector(**self.holiday_params)
        try:
            self.holiday_detector.detect(residual_df)
            holiday_flags = self.holiday_detector.dates_to_holidays(residual_df.index, style='series_flag')
        except Exception:
            holiday_flags = pd.DataFrame(0, index=residual_df.index, columns=residual_df.columns)
        
        holiday_dates = {}
        
        # Handle both multivariate and univariate outputs
        if self.detection_mode == 'univariate':
            # Univariate mode: single column of holiday flags for all series
            if holiday_flags.shape[1] > 0:
                holiday_col = holiday_flags.iloc[:, 0]
                flagged = holiday_col[holiday_col > 0].index
                holiday_list = [pd.Timestamp(ix) for ix in flagged]
                # In univariate mode, all series share the same holidays
                for col in residual_df.columns:
                    holiday_dates[col] = holiday_list
            else:
                for col in residual_df.columns:
                    holiday_dates[col] = []
        else:
            # Multivariate mode: each series has its own holiday flags
            for col in residual_df.columns:
                series_flags = holiday_flags[col] if col in holiday_flags else pd.Series(0, index=residual_df.index)
                flagged = series_flags[series_flags > 0].index
                holiday_dates[col] = [pd.Timestamp(ix) for ix in flagged]
        
        return holiday_dates

    def _detect_anomalies(self, residual_df):
        """Detect anomalies using AnomalyRemoval."""
        self.anomaly_detector = AnomalyRemoval(**self.anomaly_params)
        cleaned = self.anomaly_detector.fit_transform(residual_df)
        anomalies = {}
        
        # Handle both multivariate and univariate outputs
        if self.detection_mode == 'univariate':
            # Univariate mode: single column of anomaly flags for all series
            # anomalies will be a single column, apply to all series
            anomaly_col = self.anomaly_detector.anomalies.iloc[:, 0]
            mask = anomaly_col == -1
            anomaly_dates = residual_df.index[mask].tolist()
            
            # Create records for each anomalous date
            records = []
            for date in anomaly_dates:
                # For univariate, we could use the max magnitude across series or mean
                magnitudes = residual_df.loc[date, :].values
                magnitude = float(np.nanmean(magnitudes))
                score = None
                if hasattr(self.anomaly_detector, 'scores') and not self.anomaly_detector.scores.empty:
                    try:
                        score = float(self.anomaly_detector.scores.loc[date].iloc[0])
                    except Exception:
                        score = None
                anomaly_type = 'point_outlier'  # Simplified for univariate
                records.append({
                    'date': pd.Timestamp(date),
                    'magnitude': magnitude,
                    'score': score,
                    'type': anomaly_type,
                })
            # In univariate mode, all series share the same anomalies
            for col in residual_df.columns:
                anomalies[col] = records
        else:
            # Multivariate mode: each series has its own anomaly flags
            for col in residual_df.columns:
                if col not in self.anomaly_detector.anomalies.columns:
                    anomalies[col] = []
                    continue
                    
                mask = self.anomaly_detector.anomalies[col] == -1
                if mask.sum() == 0:
                    anomalies[col] = []
                    continue
                    
                anomaly_dates = residual_df.index[mask].tolist()
                records = []
                for date in anomaly_dates:
                    magnitude = residual_df.at[date, col]
                    score = None
                    if hasattr(self.anomaly_detector, 'scores') and not self.anomaly_detector.scores.empty:
                        try:
                            score = float(self.anomaly_detector.scores.loc[date, col])
                        except Exception:
                            score = None
                    anomaly_type = self._classify_anomaly_type(residual_df[col], date)
                    records.append({
                        'date': pd.Timestamp(date),
                        'magnitude': float(magnitude),
                        'score': score,
                        'type': anomaly_type,
                    })
                anomalies[col] = records
        return cleaned, anomalies

    @staticmethod
    def _classify_anomaly_type(series, date):
        """
        Classify anomaly type based on pattern.
        
        Currently returns 'point_outlier' for all detected anomalies.
        Future enhancement: implement detection of decay, ramp, etc.
        """
        # TODO: Implement proper decay/ramp detection
        # For now, all anomalies are classified as point_outlier
        # This matches the synthetic data generation patterns
        return 'point_outlier'

    def _fit_final_seasonality(self, df):
        model = DatepartRegressionTransformer(**self.seasonality_params)
        residual = model.fit_transform(df)
        seasonal = df - residual
        strength = self._compute_seasonality_strength(df, residual, seasonal)
        return residual, seasonal, strength, model

    def _compute_seasonality_strength(self, original_df, residual_df, seasonal_df):
        strength = {}
        for col in original_df.columns:
            y = original_df[col].to_numpy(dtype=float)
            resid = residual_df[col].to_numpy(dtype=float)
            seasonal = seasonal_df[col].to_numpy(dtype=float)
            mask = ~(np.isnan(y) | np.isnan(resid) | np.isnan(seasonal))
            if mask.sum() < 2:
                strength[col] = 0.0
                continue
            y_clean = y[mask]
            resid_clean = resid[mask]
            seasonal_clean = seasonal[mask]
            total_var = np.var(y_clean)
            resid_var = np.var(resid_clean)
            r_squared = 0.0 if total_var == 0 else max(0.0, min(1.0, 1 - resid_var / total_var))
            if len(seasonal_clean) > 1:
                corr = np.corrcoef(y_clean, seasonal_clean)[0, 1]
                corr_strength = max(0.0, corr ** 2) if np.isfinite(corr) else 0.0
            else:
                corr_strength = 0.0
            variance_ratio = 0.0 if total_var == 0 else min(1.0, np.var(seasonal_clean) / total_var)
            combined = 0.6 * r_squared + 0.3 * corr_strength + 0.1 * variance_ratio
            strength[col] = max(0.0, min(1.0, combined))
        return strength

    def _estimate_holiday_impacts(self, residual_df, holiday_dates):
        holiday_component = pd.DataFrame(0.0, index=residual_df.index, columns=residual_df.columns)
        holiday_impacts = {}
        for col in residual_df.columns:
            impacts = {}
            for date in holiday_dates.get(col, []):
                if date in residual_df.index:
                    value = residual_df.at[date, col]
                    holiday_component.at[date, col] = value
                    impacts[pd.Timestamp(date)] = float(value)
            holiday_impacts[col] = impacts
        return holiday_component, holiday_impacts

    def _detect_level_shifts(self, residual_df):
        self.level_shift_detector = LevelShiftMagic(**self.level_shift_params)
        self.level_shift_detector.fit(residual_df)
        lvlshft = self.level_shift_detector.lvlshft.reindex(residual_df.index).fillna(0.0)
        diff = lvlshft.diff().fillna(0.0)
        candidates = {}
        for col in residual_df.columns:
            col_diff = diff[col]
            entries = []
            for date, magnitude in col_diff[col_diff != 0].items():
                entries.append({'date': pd.Timestamp(date), 'magnitude': float(magnitude)})
            candidates[col] = entries
        return lvlshft, candidates

    def _validate_level_shifts(self, residual_df, lvlshft, candidates):
        params = self.level_shift_validation
        window = int(params.get('window', 14))
        pad = int(params.get('pad', 2))
        rel_thresh = float(params.get('relative_threshold', 0.1))
        abs_thresh = float(params.get('absolute_threshold', 0.5))

        validated_component = lvlshft.copy()
        validated = {}

        for col in residual_df.columns:
            series = residual_df[col]
            entries = []
            for candidate in candidates.get(col, []):
                date = candidate['date']
                magnitude = candidate['magnitude']
                try:
                    idx = series.index.get_loc(date)
                except KeyError:
                    continue
                left_end = max(0, idx - pad)
                left_start = max(0, left_end - window)
                right_start = min(len(series), idx + pad + 1)
                right_end = min(len(series), right_start + window)

                left_window = series.iloc[left_start:left_end]
                right_window = series.iloc[right_start:right_end]

                if left_window.empty or right_window.empty:
                    validated_component.loc[date:, col] -= magnitude
                    continue

                before = float(np.nanmedian(left_window))
                after = float(np.nanmedian(right_window))
                change = after - before
                abs_change = abs(change)
                rel_change = abs_change / max(abs(before), 1e-9)

                if abs_change >= abs_thresh or rel_change >= rel_thresh:
                    entries.append({
                        'date': date,
                        'magnitude': magnitude,
                        'validated_change': change,
                        'relative_change': rel_change,
                    })
                else:
                    validated_component.loc[date:, col] -= magnitude
            validated[col] = entries
        return validated_component, validated

    def _detect_trend_changepoints(self, trend_input):
        detector_params = self.changepoint_params.copy()
        aggregate_method = detector_params.pop('aggregate_method', 'individual')
        method = detector_params.pop('method', 'pelt')
        method_params = detector_params.pop('method_params', {})
        min_segment_length = detector_params.pop('min_segment_length', 14)
        self.changepoint_detector = ChangepointDetector(
            method=method,
            method_params=method_params,
            aggregate_method=aggregate_method,
            min_segment_length=min_segment_length,
        )
        safe_df = trend_input.ffill().bfill()
        self.changepoint_detector.fit(safe_df)

        n_samples = len(self.date_index)
        series_names = list(trend_input.columns)
        n_series = len(series_names)

        changepoint_indices = {}
        changepoints = {}

        raw_cps = self.changepoint_detector.changepoints_
        if isinstance(raw_cps, dict):
            for col in series_names:
                indices = np.asarray(raw_cps.get(col, []), dtype=int)
                if indices.size:
                    indices = np.unique(indices[(indices > 0) & (indices < n_samples)])
                changepoint_indices[col] = indices
                changepoints[col] = [self.date_index[idx] for idx in indices]
        else:
            indices = np.asarray(raw_cps if raw_cps is not None else [], dtype=int)
            if indices.size:
                indices = np.unique(indices[(indices > 0) & (indices < n_samples)])
            for col in series_names:
                changepoint_indices[col] = indices
                changepoints[col] = [self.date_index[idx] for idx in indices]

        if not changepoint_indices:
            changepoint_indices = {col: np.array([], dtype=int) for col in series_names}
            changepoints = {col: [] for col in series_names}

        max_segments = 1
        if changepoint_indices:
            max_segments = max((len(idx) + 1) for idx in changepoint_indices.values()) or 1

        segment_starts = np.zeros((max_segments, n_series), dtype=int)
        segment_ends = np.zeros((max_segments, n_series), dtype=int)
        valid_mask = np.zeros((max_segments, n_series), dtype=bool)

        for j, col in enumerate(series_names):
            indices = changepoint_indices.get(col, np.array([], dtype=int))
            if indices.size:
                indices = indices[(indices > 0) & (indices < n_samples)]
                if indices.size:
                    indices = np.unique(indices)
            breaks = np.concatenate(([0], indices, [n_samples]))
            seg_len = len(breaks) - 1
            segment_starts[:seg_len, j] = breaks[:-1]
            segment_ends[:seg_len, j] = breaks[1:]
            valid_mask[:seg_len, j] = True

        values = safe_df.to_numpy(dtype=float, copy=False)
        time_index = np.arange(n_samples, dtype=float)

        prefix_y = np.vstack([np.zeros((1, n_series)), np.cumsum(values, axis=0)])
        prefix_ty = np.vstack([np.zeros((1, n_series)), np.cumsum(values * time_index[:, None], axis=0)])
        prefix_t = np.concatenate(([0.0], np.cumsum(time_index)))
        prefix_t2 = np.concatenate(([0.0], np.cumsum(time_index ** 2)))

        prefix_y_T = prefix_y.T
        sum_y = np.take_along_axis(prefix_y_T, segment_ends.T, axis=1) - np.take_along_axis(prefix_y_T, segment_starts.T, axis=1)
        sum_y = sum_y.T

        prefix_ty_T = prefix_ty.T
        sum_ty = np.take_along_axis(prefix_ty_T, segment_ends.T, axis=1) - np.take_along_axis(prefix_ty_T, segment_starts.T, axis=1)
        sum_ty = sum_ty.T

        sum_t = prefix_t[segment_ends] - prefix_t[segment_starts]
        sum_t2 = prefix_t2[segment_ends] - prefix_t2[segment_starts]
        lengths = (segment_ends - segment_starts).astype(float)

        sum_y = np.where(valid_mask, sum_y, 0.0)
        sum_ty = np.where(valid_mask, sum_ty, 0.0)
        sum_t = np.where(valid_mask, sum_t, 0.0)
        sum_t2 = np.where(valid_mask, sum_t2, 0.0)
        lengths = np.where(valid_mask, lengths, 0.0)

        numerator = lengths * sum_ty - sum_t * sum_y
        denominator = lengths * sum_t2 - sum_t ** 2
        slope = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=(denominator != 0) & valid_mask,
        )

        base_slope = slope[0, :]
        base_length = lengths[0, :]
        base_intercept = np.divide(
            sum_y[0, :] - base_slope * sum_t[0, :],
            base_length,
            out=np.zeros_like(base_slope),
            where=base_length != 0,
        )
        zero_length_mask = base_length == 0
        if np.any(zero_length_mask):
            base_intercept[zero_length_mask] = values[0, zero_length_mask]

        trend_matrix = base_intercept + base_slope * time_index[:, None]

        if max_segments > 1:
            slope_changes = slope[1:, :] - slope[:-1, :]
            slope_changes = np.where(valid_mask[1:, :], slope_changes, 0.0)
            hinge_positions = segment_starts[1:, :].astype(float)
            hinge_contrib = np.maximum(0.0, time_index[:, None, None] - hinge_positions[None, :, :])
            trend_matrix += np.sum(hinge_contrib * slope_changes[None, :, :], axis=1)

        trend_component = pd.DataFrame(trend_matrix, index=self.date_index, columns=series_names)
        return changepoints, trend_component

    @staticmethod
    def _segment_slope(values, start_idx, end_idx):
        if end_idx <= start_idx:
            return 0.0
        segment = values[start_idx:end_idx + 1]
        x = np.arange(len(segment))
        mask = ~np.isnan(segment)
        if mask.sum() < 2:
            return 0.0
        x = x[mask]
        y = segment[mask]
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return 0.0
        return np.sum((x - x_mean) * (y - y_mean)) / denom

    def _compute_trend_slopes(self, trend_component, changepoints):
        slopes = {}
        for col in trend_component.columns:
            cp_dates = sorted(set(changepoints.get(col, [])))
            if not cp_dates:
                slope = self._segment_slope(trend_component[col].to_numpy(), 0, len(trend_component) - 1)
                slopes[col] = [{
                    'start_date': self.date_index[0],
                    'end_date': self.date_index[-1],
                    'slope': float(slope),
                }]
                continue
            indices = [0] + [self.date_index.get_loc(date) for date in cp_dates if date in self.date_index]
            indices = sorted(set(indices))
            if indices[-1] != len(trend_component) - 1:
                indices.append(len(trend_component) - 1)
            segment_info = []
            for start_idx, end_idx in zip(indices[:-1], indices[1:]):
                if end_idx <= start_idx:
                    continue
                slope = self._segment_slope(trend_component[col].to_numpy(), start_idx, end_idx)
                segment_info.append({
                    'start_date': self.date_index[start_idx],
                    'end_date': self.date_index[end_idx],
                    'slope': float(slope),
                })
            slopes[col] = segment_info
        return slopes

    def _convert_to_original_scale(self, component_df, include_mean=False):
        if component_df is None:
            return None
        if not self.standardize or self.scaler is None:
            return component_df.copy()
        scaled = component_df.multiply(self.scale_series, axis=1)
        if include_mean:
            scaled = scaled.add(self.mean_series, axis=1)
        return scaled

    def _to_original_value(self, value, series_name, include_mean=False):
        scale = float(self.scale_series.get(series_name, 1.0))
        mean = float(self.mean_series.get(series_name, 0.0))
        result = float(value) * scale
        if include_mean:
            result += mean
        return result

    def _rescale_holiday_impacts(self, impacts):
        rescaled = {}
        for series_name, mapping in impacts.items():
            converted = {}
            for date, value in mapping.items():
                converted[pd.Timestamp(date)] = self._to_original_value(value, series_name)
            rescaled[series_name] = converted
        return rescaled

    def _rescale_level_shifts(self, level_shifts):
        rescaled = {}
        for series_name, entries in level_shifts.items():
            converted = []
            for entry in entries:
                converted.append({
                    'date': pd.Timestamp(entry['date']),
                    'magnitude': self._to_original_value(entry['magnitude'], series_name),
                    'validated_change': self._to_original_value(entry.get('validated_change', entry['magnitude']), series_name),
                    'relative_change': float(entry.get('relative_change', 0.0)),
                })
            rescaled[series_name] = converted
        return rescaled

    def _rescale_slope_info(self, slope_info):
        rescaled = {}
        for series_name, entries in slope_info.items():
            converted = []
            for entry in entries:
                converted.append({
                    'start_date': pd.Timestamp(entry['start_date']),
                    'end_date': pd.Timestamp(entry['end_date']),
                    'slope': self._to_original_value(entry['slope'], series_name),
                })
            rescaled[series_name] = converted
        return rescaled

    def _rescale_anomalies(self, anomaly_records):
        rescaled = {}
        for series_name, entries in anomaly_records.items():
            converted = []
            for entry in entries:
                converted.append({
                    'date': pd.Timestamp(entry['date']),
                    'magnitude': self._to_original_value(entry['magnitude'], series_name),
                    'score': entry.get('score'),
                    'type': entry.get('type', 'spike'),
                })
            rescaled[series_name] = converted
        return rescaled

    def _build_trend_label_entries(self, series_name, changepoints, slope_info):
        slopes = slope_info.get(series_name, [])
        if not slopes or len(slopes) < 2:
            return [], []
        entries = []
        template_entries = []
        for idx in range(1, len(slopes)):
            cp_date = slopes[idx]['start_date']
            prior_slope = slopes[idx - 1]['slope']
            new_slope = slopes[idx]['slope']
            entries.append((pd.Timestamp(cp_date), prior_slope, new_slope))
            template_entries.append({
                'date': pd.Timestamp(cp_date).isoformat(),
                'prior_slope': prior_slope,
                'new_slope': new_slope,
            })
        return entries, template_entries

    def _build_level_shift_entries(self, series_name, validated_level_shifts):
        entries = []
        template_entries = []
        for item in validated_level_shifts.get(series_name, []):
            date = pd.Timestamp(item['date'])
            magnitude = item['magnitude']
            entries.append((date, magnitude, 'validated', False))
            template_entries.append({
                'date': date.isoformat(),
                'magnitude': magnitude,
                'shift_type': 'validated',
                'shared': False,
            })
        return entries, template_entries

    def _build_anomaly_entries(self, series_name, anomaly_records):
        entries = []
        template_entries = []
        for item in anomaly_records.get(series_name, []):
            date = pd.Timestamp(item['date'])
            magnitude = item['magnitude']
            anomaly_type = item.get('type', 'point_outlier')
            entries.append((date, magnitude, anomaly_type, 1, False))
            template_entries.append({
                'date': date.isoformat(),
                'magnitude': magnitude,
                'pattern': anomaly_type,
                'duration': 1,
                'shared': False,
            })
        return entries, template_entries

    @staticmethod
    def _serialize_datetime_mapping(mapping):
        serialized = {}
        for key, value in mapping.items():
            serialized[pd.Timestamp(key).isoformat()] = value
        return serialized

    def _build_series_template(self, series_name, components, labels, metadata):
        component_dict = {
            name: {'values': values.tolist()}
            for name, values in components.items()
        }
        label_dict = {
            'trend_changepoints': labels.get('trend_changepoints', []),
            'level_shifts': labels.get('level_shifts', []),
            'anomalies': labels.get('anomalies', []),
            'holiday_impacts': self._serialize_datetime_mapping(labels.get('holiday_impacts', {})),
            'holiday_dates': [pd.Timestamp(x).isoformat() for x in labels.get('holiday_dates', [])],
            'seasonality_changepoints': labels.get('seasonality_changepoints', []),
            'noise_changepoints': labels.get('noise_changepoints', []),
        }
        return {
            'series_name': series_name,
            'series_type': 'detected',
            'scale_factor': 1.0,
            'combination': 'additive',
            'components': component_dict,
            'labels': label_dict,
            'metadata': {
                'seasonality_strengths': {'combined': metadata.get('seasonality_strength', 0.0)},
                'noise_to_signal_ratio': metadata.get('noise_to_signal_ratio'),
            },
        }

    def get_detected_features(self, series_name=None):
        if series_name is not None:
            return {
                'trend_changepoints': self.trend_changepoints.get(series_name, []),
                'level_shifts': self.level_shifts.get(series_name, []),
                'anomalies': self.anomalies.get(series_name, []),
                'holiday_dates': self.holiday_dates.get(series_name, []),
                'holiday_impacts': self.holiday_impacts.get(series_name, {}),
                'seasonality_strength': self.seasonality_strength.get(series_name, 0.0),
            }
        return {
            'trend_changepoints': self.trend_changepoints,
            'level_shifts': self.level_shifts,
            'anomalies': self.anomalies,
            'holiday_dates': self.holiday_dates,
            'holiday_impacts': self.holiday_impacts,
            'seasonality_strength': self.seasonality_strength,
        }

    def get_template(self, deep=True):
        if self.template is None:
            return None
        return copy.deepcopy(self.template) if deep else self.template

    def summary(self):
        if self.df_original is None:
            print("TimeSeriesFeatureDetector has not been fit.")
            return
        print("=" * 80)
        print("TIME SERIES FEATURE DETECTION SUMMARY")
        print("=" * 80)
        print(f"Date Range: {self.date_index[0]} to {self.date_index[-1]}")
        print(f"Number of Series: {self.df_original.shape[1]}")
        print(f"Number of Observations: {self.df_original.shape[0]}")
        for series_name in self.df_original.columns:
            print("\n" + "-" * 80)
            print(f"Series: {series_name}")
            strength = self.seasonality_strength.get(series_name, 0.0)
            print(f"Seasonality Strength: {strength:.3f}")
            cps = self.trend_changepoints.get(series_name, [])
            print(f"Trend Changepoints: {len(cps)}")
            for idx, cp in enumerate(cps[:5]):
                date, prior_slope, new_slope = cp
                print(f"  {idx + 1}. {date}: slope {prior_slope:.4f} → {new_slope:.4f}")
            level_shifts = self.level_shifts.get(series_name, [])
            print(f"Level Shifts: {len(level_shifts)}")
            for idx, shift in enumerate(level_shifts[:5]):
                date, magnitude, shift_type, _ = shift
                print(f"  {idx + 1}. {date}: magnitude {magnitude:.4f}")
            anomalies = self.anomalies.get(series_name, [])
            print(f"Anomalies: {len(anomalies)}")
            for idx, anom in enumerate(anomalies[:5]):
                date, magnitude, anomaly_type, _, _ = anom
                print(f"  {idx + 1}. {date}: {anomaly_type}, magnitude {magnitude:.4f}")
            holidays = self.holiday_dates.get(series_name, [])
            print(f"Holidays: {len(holidays)}")
            for idx, hol in enumerate(holidays[:5]):
                print(f"  {idx + 1}. {hol}")
        print("\n" + "=" * 80)

    def plot(self, series_name=None, figsize=(16, 12), save_path=None, show=True):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting.")
        if self.df_original is None:
            raise RuntimeError("Call fit() before plot().")
        if series_name is None:
            series_name = self.df_original.columns[0]
        if series_name not in self.df_original.columns:
            raise ValueError(f"Series '{series_name}' not found.")
        components = self.components.get(series_name)
        if components is None:
            raise RuntimeError(f"No components stored for series '{series_name}'.")
        labels = {
            'trend_changepoints': self.trend_changepoints.get(series_name, []),
            'level_shifts': self.level_shifts.get(series_name, []),
            'anomalies': self.anomalies.get(series_name, []),
            'holiday_impacts': self.holiday_impacts.get(series_name, {}),
            'holiday_dates': self.holiday_dates.get(series_name, []),
            'seasonality_changepoints': [],
            'noise_changepoints': self.noise_changepoints.get(series_name, []),
            'series_scale': 1.0,
            'noise_to_signal_ratio': self.noise_to_signal_ratios.get(series_name, None),
            'series_type': 'detected',
        }
        fig = plot_feature_panels(
            series_name=series_name,
            date_index=self.date_index,
            series_data=self.df_original[series_name],
            components=components,
            labels=labels,
            series_type_description='Detected Features',
            scale=labels.get('series_scale'),
            noise_to_signal=labels.get('noise_to_signal_ratio'),
            figsize=figsize,
            save_path=save_path,
            show=show,
            title_prefix='Feature Detection',
        )
        return fig


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
        detected_cp_dates = [
            cp['date'] if isinstance(cp, dict)
            else (cp[0] if isinstance(cp, (tuple, list)) else cp)
            for cp in detected_cp
        ]
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
        Recommend running synthetic with disable_holiday_splash=True, anomaly_types=["point_outlier"] for simplicity
        
        - Focuses on point_outlier detection
        - More lenient on complex patterns (decay, ramp, impulse_decay, etc.)
        """
        if len(true_anom) == 0:
            return 0.0 if len(detected_anom) == 0 else len(detected_anom) * 0.3
        
        loss = 0.0
        detected_dates = [a['date'] if isinstance(a, dict) else (a[0] if isinstance(a, (tuple, list)) else a) for a in detected_anom]
        
        # Define simple vs complex anomaly types
        simple_types = {'point_outlier', 'spike'}  # 'spike' for backwards compatibility
        complex_types = {'decay', 'ramp_up', 'ramp_down', 'impulse_decay', 'linear_decay', 'noisy_burst', 'transient_change'}
        
        # For each true anomaly
        for true_item in true_anom:
            if isinstance(true_item, dict):
                true_date = true_item['date']
                true_type = true_item.get('type', 'point_outlier')
                true_mag = abs(true_item.get('magnitude', 1.0))
            elif isinstance(true_item, (tuple, list)):
                true_date = true_item[0]
                # (date, magnitude, type, duration, shared) or (date, magnitude, type, duration)
                true_type = true_item[2] if len(true_item) > 2 else 'point_outlier'
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
            is_simple = true_type in simple_types
            
            if min_dist <= self.anomaly_tolerance_days:
                # Found
                if is_simple:
                    # Point outlier: should be found exactly
                    penalty = 0.0 if min_dist == 0 else 0.3
                else:
                    # Complex pattern: more lenient
                    penalty = 0.2
            else:
                # Missed
                if is_simple:
                    penalty = 1.0  # Full penalty for missing point outlier
                else:
                    penalty = 0.5  # Less penalty for missing complex patterns
            
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
            self.baseline_loss = None
            self.best_params = baseline_params
            self.best_loss = float('inf')
        
        successful_iterations = 0
        failed_iterations = 0
        
        for i in range(self.n_iterations):
            # Generate random parameters
            params = self._sample_random_params()
            
            # Evaluate
            try:
                loss = self._evaluate_params(params)
                
                # Only track successful evaluations in history
                self.optimization_history.append({
                    'iteration': successful_iterations,
                    'params': copy.deepcopy(params),
                    'loss': loss['total_loss'],
                    'loss_breakdown': loss,
                })
                
                successful_iterations += 1
                
                # Update best
                if loss['total_loss'] < self.best_loss:
                    self.best_loss = loss['total_loss']
                    self.best_params = copy.deepcopy(params)
                    print(f"Iteration {i}: New best loss = {self.best_loss:.4f}")
            except Exception as e:
                failed_iterations += 1
                if failed_iterations <= 3:  # Only print first few failures
                    print(f"Iteration {i} failed: {str(e)[:100]}")
                # Don't add to history - just skip this iteration
                continue
        
        if failed_iterations > 3:
            print(f"... and {failed_iterations - 3} more failures (suppressed)")
        
        print(f"\nOptimization complete!")
        print(f"Successful iterations: {successful_iterations}/{self.n_iterations}")
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
            self.baseline_loss = None
            self.best_params = baseline_params
            self.best_loss = float('inf')
        
        successful_iterations = 0
        failed_iterations = 0
        
        for i, params in enumerate(grid):
            try:
                loss = self._evaluate_params(params)
                
                # Only track successful evaluations
                self.optimization_history.append({
                    'iteration': successful_iterations,
                    'params': copy.deepcopy(params),
                    'loss': loss['total_loss'],
                    'loss_breakdown': loss,
                })
                
                successful_iterations += 1
                
                if loss['total_loss'] < self.best_loss:
                    self.best_loss = loss['total_loss']
                    self.best_params = copy.deepcopy(params)
                    print(f"Grid {i}: New best loss = {self.best_loss:.4f}")
            except Exception as e:
                failed_iterations += 1
                if failed_iterations <= 3:  # Only print first few failures
                    print(f"Grid {i} failed: {str(e)[:100]}")
                # Don't add to history - just skip
                continue
        
        if failed_iterations > 3:
            print(f"... and {failed_iterations - 3} more failures (suppressed)")
        
        print(f"\nOptimization complete!")
        print(f"Successful iterations: {successful_iterations}/{len(grid)}")
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
        changepoint_params = ChangepointDetector().get_new_params(method='random')
        
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
