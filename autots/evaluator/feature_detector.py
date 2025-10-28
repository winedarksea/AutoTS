# -*- coding: utf-8 -*-
"""
Time Series Feature Detection and Optimization

@author: Colin with Claude Sonnet v4.5

Matching test file in tests/test_feature_detector.py
"""

import numpy as np
import pandas as pd
import random
import copy
import warnings
import json
import time
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
from autots.tools.seasonal import date_part
from autots.datasets.synthetic import SyntheticDailyGenerator

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    from autots.tools.mocks import StandardScaler



class TimeSeriesFeatureDetector:
    """
    Comprehensive feature detection pipeline for time series.

    TODO: upstream more of this code into the component classes (e.g., HolidayDetector, AnomalyRemoval, ChangepointDetector)
    TODO: Handle multiplicative seasonality
    TODO: Handle time varying seasonality using fast_kalman
    TODO: Improve holiday "splash" effect and weekend interactions
    TODO: Support identifying anomaly types beyond just point_outlier
    TODO: Support identifying regressor impacts and granger lag impacts
    TODO: Support identifying variance regime changes
    TODO: Build upon the JSON template so that it can be converted to a fixed size embedding (probably a 2d embedding). The fixed size may vary by parameters, but for a given parameter set should always be the same size. The embedding does not need to be capable of fully reconstructing the time series, just representing it.
    TODO: Support for modeling the trend with a fast kalman state space approach, ideally aligned with changepoints in some way if possible.
    TODO: Improved scaling and option to skip scaling

        Parameters
    ----------
    seasonality_params : dict, optional
        Parameters for DatepartRegressionTransformer used in final seasonality fit
    rough_seasonality_params : dict, optional
        Parameters for DatepartRegressionTransformer used in initial rough seasonality decomposition
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
        rough_seasonality_params=None,
        holiday_params=None,
        anomaly_params=None,
        changepoint_params=None,
        level_shift_params=None,
        level_shift_validation=None,
        general_transformer_params=None,
        smoothing_window=None,
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
        
        self.rough_seasonality_params = rough_seasonality_params or {
            'regression_model': {
                'model': 'RandomForest',
                'model_params': {
                    'n_estimators': 100,
                    'min_samples_leaf': 4,
                    'bootstrap': True
                },
            },
            'datepart_method': 'simple',
            'polynomial_degree': 2,
            'transform_dict': {
                'fillna': None,
                'transformations': {'0': 'EWMAFilter'},
                'transformation_params': {'0': {'span': 2}}
            },
            'holiday_countries_used': False,
            'lags': None,
            'forward_lags': None,
        }
        self.seasonality_params = seasonality_params or {
            'regression_model': {
                'model': 'SVM',
                'model_params': {
                    'C': 1.0,
                    'tol': 0.0001,
                    'loss': 'squared_epsilon_insensitive',
                    'max_iter': 500
                },
            },
            'datepart_method': 'common_fourier',
            'polynomial_degree': None,
            'transform_dict': None,
            'holiday_countries_used': False,
            'lags': None,
            'forward_lags': None,
        }
        self.holiday_params = self._sanitize_holiday_params(holiday_params)
        # Ensure anomaly_params uses the correct output mode
        if anomaly_params is None:
            self.anomaly_params = {
                'output': self.detection_mode,
                'method': 'rolling_zscore',
                'method_params': {
                    'distribution': 'norm',
                    'alpha': 0.05,
                    'rolling_periods': 200,
                    'center': False
                },
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
                'grouping_forward_limit': 5,
                'max_level_shifts': 5,
                'alignment': 'rolling_diff',
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
        self.general_transformer_params = general_transformer_params or {
            'fillna': 'ffill_mean_biased',
            'transformations': {0: 'ClipOutliers', 1: 'ScipyFilter'},
            'transformation_params': {
                0: {
                    'method': 'clip',
                    'std_threshold': 3.5,
                    'fillna': None
                },
                1: {
                    'method': 'butter',
                    'method_args': {
                        'N': 3,
                        'btype': 'lowpass',
                        'analog': False,
                        'output': 'sos',
                        'Wn': 0.5
                    }
                }
            }
        }
        self.smoothing_window = smoothing_window
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
        self.holiday_splash_impacts = {}
        self.holiday_coefficients = {}
        self.seasonality_components = {}
        self.seasonality_strength = {}
        self.series_seasonality_profiles = {}
        self.noise_changepoints = {}
        self.noise_to_signal_ratios = {}
        self.series_noise_levels = {}
        self.series_scales = {}
        self.shared_events = {'anomalies': [], 'level_shifts': []}
        self.reconstructed = None
        self.reconstructed_components = None
        self.reconstruction_error = None
        self.reconstruction_rmse = None

    def _sanitize_holiday_params(self, holiday_params):
        """Return holiday detector parameters filtered to supported keys."""
        default_params = {
            'anomaly_detector_params': {
                'method': 'mad',
                'transform_dict': None,
                'forecast_params': None,
                'method_params': {'distribution': 'uniform', 'alpha': 0.05}
            },
            'threshold': 0.8,
            'min_occurrences': 2,
            'splash_threshold': None,
            'use_dayofmonth_holidays': True,
            'use_wkdom_holidays': True,
            'use_wkdeom_holidays': False,
            'use_lunar_holidays': False,
            'use_lunar_weekday': False,
            'use_islamic_holidays': True,
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
        """
        Fit the feature detector to time series data.
        
        Decomposition follows this sequential removal strategy:
        
        1. INITIAL DECOMPOSITION (for detection only):
           - Remove rough seasonality → rough_residual
           - Detect holidays on rough_residual
           - Detect anomalies on rough_residual
        
        2. FINAL SEASONALITY FIT:
           - Fit on: original - anomalies
           - Holidays fitted simultaneously as regressors
           - Output: final_residual (has seasonality + holidays removed)
        
        3. LEVEL SHIFT DETECTION:
           - Detect on: original - anomalies - seasonality - holidays
           - (This is final_residual)
        
        4. TREND DETECTION:
           - Detect on: original - anomalies - seasonality - holidays - level_shifts
        
        5. NOISE & ANOMALY COMPONENTS:
           - Noise: original - trend - level_shifts - seasonality - holidays - anomalies
           - Anomalies: difference between original and de-anomalied version
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        # Step 1: Prepare data and standardize if requested
        df_work = self._prepare_data(df)
        
        # Reset all result containers
        self._reset_results()
        
        # Step 2-4: Initial decomposition (seasonality, holidays, anomalies)
        rough_residual, rough_seasonality = self._initial_decomposition(df_work)
        
        # Step 5: Final seasonality fit with holiday effects
        final_residual, final_seasonality, seasonality_strength, holiday_component_scaled, holiday_coefficients, holiday_splash_impacts_scaled = self._final_seasonality_fit(
            df_work, rough_residual, rough_seasonality
        )
        
        # Step 6-7: Trend and level shift detection
        # Pass holiday component so we know holidays are already removed in final_residual
        trend_component_scaled, level_shift_component_scaled, validated_level_shifts, changepoints, slope_info = self._detect_trend_and_shifts(
            final_residual, holiday_component_scaled
        )
        
        # Step 8: Noise analysis
        noise_component_scaled, anomaly_component_scaled = self._analyze_noise(
            df_work, trend_component_scaled, level_shift_component_scaled, final_seasonality, holiday_component_scaled
        )
        
        # Step 9: Convert all components to original scale
        components_original = self._rescale_all_components(
            trend_component_scaled,
            level_shift_component_scaled,
            final_seasonality,
            holiday_component_scaled,
            noise_component_scaled,
            anomaly_component_scaled,
        )
        
        # Step 10: Build template and validate reconstruction
        self._build_template(
            components_original,
            validated_level_shifts,
            slope_info,
            changepoints,
            holiday_coefficients,
            holiday_splash_impacts_scaled,
            seasonality_strength,
        )
        
        return self
    
    def _prepare_data(self, df):
        """Prepare and standardize input data."""
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
        
        return df_work
    
    def _reset_results(self):
        """Reset all result containers to empty state."""
        config_metadata = {
            'standardize': self.standardize,
            'smoothing_window': self.smoothing_window,
            'detection_mode': self.detection_mode,
        }
        self.template = {
            'version': self.TEMPLATE_VERSION,
            'meta': {
                'start_date': self.date_index[0].isoformat(),
                'end_date': self.date_index[-1].isoformat(),
                'n_days': int(len(self.date_index)),
                'n_series': int(self.df_original.shape[1]),
                'frequency': pd.infer_freq(self.date_index) or 'infer',
                'created_at': pd.Timestamp.now().isoformat(),
                'source': 'TimeSeriesFeatureDetector',
                # Use shared config key to align with SyntheticDailyGenerator templates.
                'config': config_metadata,
            },
            'regressors': None,
            'series': {},
            'shared_events': {'anomalies': [], 'level_shifts': []},
        }
        self.components = {}
        self.trend_changepoints = {}
        self.trend_slopes = {}
        self.level_shifts = {}
        self.anomalies = {}
        self.holiday_impacts = {}
        self.holiday_dates = {}
        self.holiday_splash_impacts = {}
        self.holiday_coefficients = {}
        self.seasonality_components = {}
        self.seasonality_strength = {}
        self.series_seasonality_profiles = {}
        self.noise_changepoints = {}
        self.noise_to_signal_ratios = {}
        self.series_noise_levels = {}
        self.series_scales = {}
        self.shared_events = {'anomalies': [], 'level_shifts': []}
        self.reconstructed = None
        self.reconstructed_components = None
        self.reconstruction_error = None
        self.reconstruction_rmse = None
    
    def _initial_decomposition(self, df_work):
        """
        Perform initial decomposition: rough seasonality, holidays, and anomalies.
        
        Returns
        -------
        tuple
            (rough_residual, rough_seasonality)
        """
        # Rough seasonality removal
        rough_residual, rough_seasonality, self.rough_seasonality_model = self._compute_rough_seasonality(df_work)
        
        # Holiday detection
        holiday_dates, holiday_splash_dates, holiday_regressors = self._detect_holidays(rough_residual)
        self._holiday_dates_temp = holiday_dates
        self._holiday_regressors_temp = holiday_regressors
        
        # Anomaly detection
        residual_without_anomalies, anomaly_records = self._detect_anomalies(rough_residual)
        self._anomaly_records_temp = anomaly_records
        
        return rough_residual, rough_seasonality
    
    def _final_seasonality_fit(self, df_work, rough_residual, rough_seasonality):
        """
        Fit final seasonality model including holiday effects.
        
        Fits on original data (df_work) with only anomalies removed.
        This ensures final seasonality captures the full seasonal pattern,
        and holidays are fit simultaneously as regressors.
        
        Returns
        -------
        tuple
            (final_residual, final_seasonality, seasonality_strength, holiday_component, 
             holiday_coefficients, holiday_splash_impacts)
        """
        # Reconstruct original data with anomalies removed
        # df_work = original standardized data
        # We need to remove anomalies from df_work, not from rough_residual
        df_without_anomalies = self.anomaly_detector.transform(df_work)
        
        # Fit final seasonality on original data (with anomalies removed)
        # Holiday effects are captured as regressors during this fit
        final_residual, final_seasonality, seasonality_strength, self.seasonality_model, holiday_component_scaled, holiday_coefficients, holiday_splash_impacts_scaled = self._fit_final_seasonality(
            df_without_anomalies, self._holiday_regressors_temp
        )
        
        return final_residual, final_seasonality, seasonality_strength, holiday_component_scaled, holiday_coefficients, holiday_splash_impacts_scaled
    
    def _detect_trend_and_shifts(self, final_residual, holiday_component_scaled):
        """
        Detect trend changepoints and level shifts.
        
        Level shifts are detected on data with: anomalies, final seasonality, and holidays removed.
        Trend is detected on data with: anomalies, final seasonality, holidays, and level shifts removed.
        
        Parameters
        ----------
        final_residual : pd.DataFrame
            Residual after final seasonality fit (has seasonality + holidays removed)
        holiday_component_scaled : pd.DataFrame
            Holiday effects in standardized scale
        
        Returns
        -------
        tuple
            (trend_component, level_shift_component, validated_shifts, changepoints, slope_info)
        """
        # final_residual already has seasonality + holidays removed by DatepartRegressionTransformer
        # We just need to ensure we're working with clean data
        residual_for_level_shifts = final_residual.copy()
        
        # Optionally apply transformations before trend detection
        self.general_transformer = None
        if self.general_transformer_params:
            self.general_transformer = GeneralTransformer(**self.general_transformer_params)
            residual_for_level_shifts = self.general_transformer.fit_transform(residual_for_level_shifts)
        if self.smoothing_window and self.smoothing_window > 1:
            residual_for_level_shifts = residual_for_level_shifts.rolling(
                window=int(self.smoothing_window),
                center=True,
                min_periods=1,
            ).mean()

        # Level shift detection on: original - anomalies - seasonality - holidays
        level_shift_component_scaled, level_shift_candidates = self._detect_level_shifts(residual_for_level_shifts)
        level_shift_component_valid_scaled, validated_level_shifts = self._validate_level_shifts(
            residual_for_level_shifts, level_shift_component_scaled, level_shift_candidates
        )

        # Trend changepoint detection on: original - anomalies - seasonality - holidays - level_shifts
        trend_input = residual_for_level_shifts - level_shift_component_valid_scaled
        changepoints, trend_component_scaled = self._detect_trend_changepoints(trend_input)
        slope_info = self._compute_trend_slopes(trend_component_scaled, changepoints)
        
        return trend_component_scaled, level_shift_component_valid_scaled, validated_level_shifts, changepoints, slope_info
    
    def _analyze_noise(self, df_work, trend_scaled, level_shift_scaled, seasonality_scaled, holiday_scaled):
        """
        Analyze noise component and anomalies.
        
        Returns
        -------
        tuple
            (noise_component, anomaly_component)
        """
        # Noise is what remains after removing trend, level shifts, seasonality, and holidays
        df_without_anomalies_scaled = df_work.copy()
        for col in df_work.columns:
            if col in self._anomaly_records_temp and self._anomaly_records_temp[col]:
                for anom in self._anomaly_records_temp[col]:
                    date = anom['date']
                    if date in df_work.index:
                        # Remove anomaly effect (simple approach: use median of neighbors)
                        try:
                            idx = df_work.index.get_loc(date)
                            neighbors = []
                            if idx > 0:
                                neighbors.append(df_work[col].iloc[idx-1])
                            if idx < len(df_work) - 1:
                                neighbors.append(df_work[col].iloc[idx+1])
                            if neighbors:
                                df_without_anomalies_scaled.loc[date, col] = np.nanmedian(neighbors)
                        except:
                            pass
        
        # Reconstruct signal without anomalies
        reconstructed_scaled = trend_scaled + level_shift_scaled + seasonality_scaled + holiday_scaled
        noise_component_scaled = df_without_anomalies_scaled - reconstructed_scaled
        anomaly_component_scaled = df_work - df_without_anomalies_scaled
        
        return noise_component_scaled, anomaly_component_scaled
    
    def _rescale_all_components(self, trend_scaled, level_shift_scaled, seasonality_scaled, 
                                 holiday_scaled, noise_scaled, anomaly_scaled):
        """
        Convert all components from standardized to original scale.
        
        Returns
        -------
        dict
            Dictionary of component DataFrames in original scale
        """
        trend_component = self._convert_to_original_scale(trend_scaled, include_mean=True)
        level_shift_component = self._convert_to_original_scale(level_shift_scaled)
        seasonality_component = self._convert_to_original_scale(seasonality_scaled)
        holiday_component = self._convert_to_original_scale(holiday_scaled)
        noise_component = self._convert_to_original_scale(noise_scaled)
        anomaly_component = self._convert_to_original_scale(anomaly_scaled)
        
        return {
            'trend': trend_component,
            'level_shift': level_shift_component,
            'seasonality': seasonality_component,
            'holidays': holiday_component,
            'noise': noise_component,
            'anomalies': anomaly_component,
        }
    
    def _build_template(self, components_original, validated_level_shifts, slope_info, changepoints,
                        holiday_coefficients, holiday_splash_impacts_scaled, seasonality_strength):
        """
        Build final template structure and validate reconstruction.
        """
        # Extract rescaled components
        trend_component = components_original['trend']
        level_shift_component = components_original['level_shift']
        seasonality_component = components_original['seasonality']
        holiday_component = components_original['holidays']
        noise_component = components_original['noise']
        anomaly_component = components_original['anomalies']
        
        # Rescale labels
        holiday_impacts = self._component_df_to_mapping(holiday_component)
        holiday_splash_impacts = self._extract_splash_impacts(holiday_splash_impacts_scaled, self._holiday_dates_temp)
        holiday_coefficients = self._rescale_holiday_coefficients(holiday_coefficients)
        validated_level_shifts = self._rescale_level_shifts(validated_level_shifts)
        slope_info = self._rescale_slope_info(slope_info)
        anomaly_records = self._rescale_anomalies(self._anomaly_records_temp)

        mark_shared = self.detection_mode == 'univariate'

        # Build series templates
        for series_name in self.df_original.columns:
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
            level_shift_entries, level_shift_template = self._build_level_shift_entries(
                series_name, validated_level_shifts, shared=mark_shared
            )
            anomaly_entries, anomaly_template = self._build_anomaly_entries(
                series_name, anomaly_records, shared=mark_shared
            )
            holidays_list = self._holiday_dates_temp.get(series_name, [])
            holiday_template = holiday_impacts.get(series_name, {})
            holiday_coeff_template = holiday_coefficients.get(series_name, {})

            self.trend_changepoints[series_name] = trend_cp_entries
            self.trend_slopes[series_name] = slope_info.get(series_name, [])
            self.level_shifts[series_name] = level_shift_entries
            self.anomalies[series_name] = anomaly_entries
            self.holiday_dates[series_name] = [pd.Timestamp(x) for x in holidays_list]
            self.holiday_impacts[series_name] = holiday_template
            self.holiday_splash_impacts[series_name] = holiday_splash_impacts.get(series_name, {})
            self.holiday_coefficients[series_name] = holiday_coeff_template
            self.seasonality_components[series_name] = seasonality_component[series_name].to_numpy(copy=True)
            self.seasonality_strength[series_name] = seasonality_strength.get(series_name, 0.0)
            self.noise_changepoints[series_name] = []
            
            # Calculate noise metrics
            seasonality_series = seasonality_component[series_name]
            noise_series = noise_component[series_name]
            signal_series = trend_component[series_name] + level_shift_component[series_name]
            numerator = float(np.nanstd(noise_series))
            denominator = float(np.nanstd(signal_series)) or 1e-9
            self.noise_to_signal_ratios[series_name] = numerator / denominator

            original_series = self.df_original[series_name]
            series_scale = float(np.nanstd(original_series)) or 1e-9
            self.series_scales[series_name] = series_scale

            # Normalize noise level against original series magnitude
            normalized_noise = numerator / (series_scale or 1e-9)
            self.series_noise_levels[series_name] = normalized_noise

            self.series_seasonality_profiles[series_name] = self._estimate_seasonality_profile(
                seasonality_series, series_scale
            )

            metadata = {
                'seasonality_strength': self.seasonality_strength[series_name],
                'noise_to_signal_ratio': self.noise_to_signal_ratios[series_name],
                'seasonality_profiles': self.series_seasonality_profiles[series_name],
                'noise_level': self.series_noise_levels[series_name],
                'series_scale': series_scale,
            }
            template_entry = self._build_series_template(
                series_name,
                components_dict,
                {
                    'trend_changepoints': trend_cp_template,
                    'level_shifts': level_shift_template,
                    'anomalies': anomaly_template,
                    'holiday_impacts': holiday_template,
                    'holiday_coefficients': holiday_coeff_template,
                    'holiday_dates': self.holiday_dates[series_name],
                    'holiday_splash_impacts': self.holiday_splash_impacts.get(series_name, {}),
                    'seasonality_changepoints': [],
                    'noise_changepoints': [],
                },
                metadata,
            )
            self.template['series'][series_name] = template_entry

        # Handle shared events
        if mark_shared and len(self.df_original.columns) > 0:
            reference_series = self.df_original.columns[0]
            shared_anomalies = {
                self._date_to_day_offset(entry[0]) for entry in self.anomalies.get(reference_series, [])
            }
            shared_level_shifts = {
                self._date_to_day_offset(entry[0]) for entry in self.level_shifts.get(reference_series, [])
            }
            self.shared_events = {
                'anomalies': sorted(shared_anomalies),
                'level_shifts': sorted(shared_level_shifts),
            }
        else:
            self.shared_events = {'anomalies': [], 'level_shifts': []}
        self.template['shared_events'] = copy.deepcopy(self.shared_events)

        # Validate reconstruction
        self._reconstruct_from_template()

    def _compute_rough_seasonality(self, df):
        model = DatepartRegressionTransformer(**self.rough_seasonality_params)
        residual = model.fit_transform(df)
        seasonal = df - residual
        return residual, seasonal, model

    def _detect_holidays(self, residual_df):
        """
        Detect holidays using HolidayDetector.
        
        Returns both core holiday dates and splash/bridge impacts in separate structures
        to align with synthetic generator output.
        """
        self.holiday_detector = HolidayDetector(**self.holiday_params)
        holiday_regressors = pd.DataFrame(index=residual_df.index)
        try:
            self.holiday_detector.detect(residual_df)
            holiday_flags = self.holiday_detector.dates_to_holidays(residual_df.index, style='series_flag')
            holiday_regressors = self.holiday_detector.dates_to_holidays(residual_df.index, style='flag')
            if holiday_regressors is None:
                holiday_regressors = pd.DataFrame(index=residual_df.index)
            else:
                holiday_regressors = holiday_regressors.reindex(residual_df.index).fillna(0.0).astype(float)
                holiday_regressors = holiday_regressors.loc[:, ~holiday_regressors.columns.duplicated()]
        except Exception:
            holiday_flags = pd.DataFrame(0, index=residual_df.index, columns=residual_df.columns)
            holiday_regressors = pd.DataFrame(index=residual_df.index)
        
        holiday_dates = {}
        holiday_splash_dates = {}  # For storing splash/bridge days separately
        
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
                    holiday_splash_dates[col] = []  # Will be populated during final seasonality fit
            else:
                for col in residual_df.columns:
                    holiday_dates[col] = []
                    holiday_splash_dates[col] = []
        else:
            # Multivariate mode: each series has its own holiday flags
            for col in residual_df.columns:
                series_flags = holiday_flags[col] if col in holiday_flags else pd.Series(0, index=residual_df.index)
                flagged = series_flags[series_flags > 0].index
                holiday_dates[col] = [pd.Timestamp(ix) for ix in flagged]
                holiday_splash_dates[col] = []  # Will be populated during final seasonality fit
        
        return holiday_dates, holiday_splash_dates, holiday_regressors

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
        Classify anomaly type based on pattern around the anomaly date.
        TODO: Move this to AnomalyRemoval/AnomalyDetector and utilize anomaly scores properly.
        
        Detects:
        - point_outlier: Single point spike
        - impulse_decay: Spike followed by exponential decay
        - linear_decay: Spike followed by linear decay
        - noisy_burst: Multiple consecutive outliers
        - transient_change: Temporary level shift
        
        Parameters
        ----------
        series : pd.Series
            The time series containing the anomaly
        date : pd.Timestamp or similar
            The date of the detected anomaly
            
        Returns
        -------
        str
            Anomaly type classification
        """
        try:
            idx = series.index.get_loc(date)
        except KeyError:
            return 'point_outlier'
        
        # Get baseline (median before anomaly)
        lookback = 14
        start_idx = max(0, idx - lookback)
        baseline_window = series.iloc[start_idx:idx]
        if baseline_window.empty:
            return 'point_outlier'
        baseline = float(np.nanmedian(baseline_window))
        
        # Get anomaly magnitude
        anomaly_value = float(series.iloc[idx])
        anomaly_mag = abs(anomaly_value - baseline)
        if anomaly_mag < 1e-9:
            return 'point_outlier'
        
        # Check post-anomaly pattern
        lookahead = 7
        end_idx = min(len(series), idx + lookahead + 1)
        post_window = series.iloc[idx+1:end_idx]
        
        if post_window.empty or len(post_window) < 2:
            return 'point_outlier'
        
        # Analyze post-anomaly behavior
        post_values = post_window.to_numpy(dtype=float)
        post_deviations = np.abs(post_values - baseline)
        
        # Check for noisy burst (multiple consecutive outliers)
        n_outliers = np.sum(post_deviations > anomaly_mag * 0.5)
        if n_outliers >= 2:
            return 'noisy_burst'
        
        # Check for decay patterns
        if len(post_deviations) >= 3:
            # Linear decay: check if deviations decrease linearly
            first_dev = post_deviations[0]
            last_dev = post_deviations[-1]
            
            # If first deviation is significant and it decreases
            if first_dev > anomaly_mag * 0.3:
                # Check for exponential decay (rapid drop)
                mid_dev = post_deviations[len(post_deviations)//2]
                if mid_dev < first_dev * 0.5 and last_dev < mid_dev * 0.5:
                    return 'impulse_decay'
                
                # Check for linear decay (gradual drop)
                if last_dev < first_dev * 0.5:
                    return 'linear_decay'
                
                # Check for transient change (sustained then return)
                if np.mean(post_deviations[:3]) > anomaly_mag * 0.3 and last_dev < anomaly_mag * 0.2:
                    return 'transient_change'
        
        # Default: point outlier
        return 'point_outlier'

    def _fit_final_seasonality(self, df, holiday_regressors=None):
        """
        Fit final seasonality model and decompose holiday effects.
        
        Returns
        -------
        tuple
            (residual, seasonal_component, strength, model, holiday_component, holiday_coefficients, holiday_splash_impacts)
        """
        regressor = None
        if holiday_regressors is not None and not holiday_regressors.empty:
            regressor = holiday_regressors.reindex(df.index).fillna(0.0)

        model = DatepartRegressionTransformer(**self.seasonality_params)
        regressor_full = regressor
        df_fit = df.dropna(how='all')
        if df_fit.empty:
            df_fit = df
        regressor_fit = None
        if regressor_full is not None:
            regressor_fit = regressor_full.loc[df_fit.index]
        model.fit(df_fit, regressor=regressor_fit)
        residual = model.transform(df, regressor=regressor_full)
        seasonal_total = df - residual

        holiday_component = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        seasonal_component = seasonal_total
        holiday_coefficients = {col: {} for col in df.columns}
        holiday_splash_impacts = {col: {} for col in df.columns}

        if regressor is not None:
            zero_regressor = regressor.copy()
            zero_regressor.loc[:, :] = 0.0
            zeros_df = pd.DataFrame(0.0, index=df.index, columns=df.columns)
            try:
                baseline_pred = model.inverse_transform(zeros_df, regressor=zero_regressor)
                seasonal_component = baseline_pred
                holiday_component = seasonal_total - seasonal_component
                
                # Detect splash/bridge days: days with holiday impact but not in core holiday list
                # Splash days are typically adjacent to core holidays with reduced impact
                for col in df.columns:
                    holiday_series = holiday_component[col]
                    significant_impacts = holiday_series[abs(holiday_series) > 1e-9]
                    for date, impact in significant_impacts.items():
                        # This will be refined by checking against core holiday dates
                        # For now, store all non-zero holiday impacts
                        # The core vs. splash distinction will be made during template building
                        holiday_splash_impacts[col][pd.Timestamp(date)] = float(impact)
                        
            except Exception as exc:
                warnings.warn(
                    f"Failed to isolate holiday contribution during seasonality fit: {exc}",
                    RuntimeWarning,
                )
                holiday_component = pd.DataFrame(0.0, index=df.index, columns=df.columns)
                seasonal_component = seasonal_total
            holiday_coefficients = self._solve_holiday_coefficients(regressor, holiday_component)

        strength = self._compute_seasonality_strength(df, residual, seasonal_component)
        return residual, seasonal_component, strength, model, holiday_component, holiday_coefficients, holiday_splash_impacts

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

    def _estimate_seasonality_profile(self, seasonal_series, series_scale):
        """
        Estimate relative strength of weekly and yearly seasonal signatures.

        Parameters
        ----------
        seasonal_series : pd.Series
            Estimated seasonal component for a single series.
        series_scale : float
            Standard deviation of the original series used for normalization.

        Returns
        -------
        dict
            Dictionary containing combined, weekly, and yearly strength estimates.
        """
        if series_scale is None or not np.isfinite(series_scale) or series_scale == 0:
            series_scale = 1.0

        if not isinstance(seasonal_series, pd.Series):
            seasonal_series = pd.Series(seasonal_series, index=self.date_index)
        seasonal_series = seasonal_series.astype(float)

        valid = seasonal_series.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            return {'combined': 0.0, 'weekly': 0.0, 'yearly': 0.0}

        combined_strength = float(np.nanstd(valid)) / series_scale

        weekly_strength = 0.0
        if valid.size >= 7:
            weekly_groups = valid.groupby(valid.index.dayofweek).mean()
            if len(weekly_groups) > 1:
                weekly_strength = float(np.nanstd(weekly_groups)) / series_scale

        yearly_strength = 0.0
        date_range_days = (valid.index[-1] - valid.index[0]).days if len(valid.index) > 1 else 0
        if date_range_days >= 180:
            yearly_groups = valid.groupby(valid.index.dayofyear).mean()
            if len(yearly_groups) > 1:
                yearly_strength = float(np.nanstd(yearly_groups)) / series_scale

        return {
            'combined': combined_strength,
            'weekly': weekly_strength,
            'yearly': yearly_strength,
        }

    def _solve_holiday_coefficients(self, regressor_df, holiday_component_df):
        coefficients = {col: {} for col in holiday_component_df.columns}
        if regressor_df is None or regressor_df.empty:
            return coefficients
        X = regressor_df.to_numpy(dtype=float)
        if X.size == 0:
            return coefficients
        regressor_columns = list(regressor_df.columns)
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.pinv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = None

        for idx, series_name in enumerate(holiday_component_df.columns):
            y = holiday_component_df.iloc[:, idx].to_numpy(dtype=float)
            if np.allclose(y, 0.0):
                continue
            try:
                if XtX_inv is not None:
                    beta = XtX_inv @ X.T @ y
                else:
                    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            series_coeffs = {}
            for j, value in enumerate(beta):
                if np.isfinite(value) and abs(value) > 1e-9:
                    series_coeffs[regressor_columns[j]] = float(value)
            if series_coeffs:
                coefficients[series_name] = series_coeffs
        return coefficients

    def _component_df_to_mapping(self, component_df, threshold=1e-9):
        mapping = {}
        for series_name in component_df.columns:
            series_map = {}
            series_values = component_df[series_name]
            for date, value in series_values.items():
                if not np.isfinite(value):
                    continue
                if abs(value) <= threshold:
                    continue
                series_map[pd.Timestamp(date)] = float(value)
            mapping[series_name] = series_map
        return mapping

    def _rescale_holiday_coefficients(self, coefficients):
        rescaled = {}
        for series_name, mapping in coefficients.items():
            scale = float(self.scale_series.get(series_name, 1.0))
            converted = {}
            for name, value in mapping.items():
                converted[name] = float(value * scale)
            rescaled[series_name] = converted
        return rescaled
    
    def _extract_splash_impacts(self, holiday_splash_impacts_scaled, holiday_dates):
        """
        Extract splash/bridge day impacts by filtering out core holiday dates.
        TODO: move this to HolidayDetector and use full functionality there.
        
        Splash days are days with holiday impacts that are NOT in the core holiday list.
        This aligns with the synthetic generator's distinction between direct holidays
        and their splash/bridge effects.
        """
        splash_impacts = {}
        for series_name, impacts_dict in holiday_splash_impacts_scaled.items():
            core_dates = set(holiday_dates.get(series_name, []))
            splash_dict = {}
            for date, impact in impacts_dict.items():
                date_ts = pd.Timestamp(date)
                # If this date has an impact but is NOT a core holiday, it's a splash day
                if date_ts not in core_dates and abs(impact) > 1e-9:
                    # Rescale to original scale
                    scale = float(self.scale_series.get(series_name, 1.0))
                    splash_dict[date_ts] = float(impact * scale)
            splash_impacts[series_name] = splash_dict
        return splash_impacts

    def _date_to_day_offset(self, date):
        base = self.date_index[0]
        return int((pd.Timestamp(date) - pd.Timestamp(base)).days)

    def _reconstruct_from_template(self):
        if self.template is None:
            return
        try:
            reconstructed, components = SyntheticDailyGenerator.render_template(
                copy.deepcopy(self.template), return_components=True
            )
            self.reconstructed = reconstructed
            self.reconstructed_components = components
            aligned = reconstructed.reindex(self.df_original.index, columns=self.df_original.columns)
            self.reconstruction_error = self.df_original - aligned
            mse = np.nanmean(np.square(self.reconstruction_error.to_numpy(dtype=float)))
            self.reconstruction_rmse = float(np.sqrt(mse)) if np.isfinite(mse) else None
            if isinstance(self.template, dict):
                self.template.setdefault('meta', {})['reconstruction_rmse'] = self.reconstruction_rmse
        except Exception as exc:
            warnings.warn(f"Template reconstruction failed: {exc}", RuntimeWarning)
            self.reconstructed = None
            self.reconstructed_components = None
            self.reconstruction_error = None
            self.reconstruction_rmse = None

    def _detect_level_shifts(self, residual_df):
        self.level_shift_detector = LevelShiftMagic(**self.level_shift_params)
        self.level_shift_detector.fit(residual_df)
        lvlshft = self.level_shift_detector.lvlshft.reindex(residual_df.index).fillna(0.0)
        # Use the new utility method to extract level shift dates and magnitudes
        candidates = self.level_shift_detector.extract_level_shift_dates(residual_df)
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

    def _build_level_shift_entries(self, series_name, validated_level_shifts, shared=False):
        entries = []
        template_entries = []
        for item in validated_level_shifts.get(series_name, []):
            date = pd.Timestamp(item['date'])
            magnitude = item['magnitude']
            entries.append((date, magnitude, 'validated', shared))
            template_entries.append({
                'date': date.isoformat(),
                'magnitude': magnitude,
                'shift_type': 'validated',
                'shared': bool(shared),
            })
        return entries, template_entries

    def _build_anomaly_entries(self, series_name, anomaly_records, shared=False):
        entries = []
        template_entries = []
        for item in anomaly_records.get(series_name, []):
            date = pd.Timestamp(item['date'])
            magnitude = item['magnitude']
            anomaly_type = item.get('type', 'point_outlier')
            entries.append((date, magnitude, anomaly_type, 1, shared))
            template_entries.append({
                'date': date.isoformat(),
                'magnitude': magnitude,
                'pattern': anomaly_type,
                'duration': 1,
                'shared': bool(shared),
            })
        return entries, template_entries

    @staticmethod
    def _serialize_datetime_mapping(mapping):
        serialized = {}
        for key, value in mapping.items():
            if isinstance(value, (np.generic,)):
                value = float(value)
            serialized[pd.Timestamp(key).isoformat()] = value
        return serialized

    def _serialize_components(self, series_name):
        components = self.components.get(series_name, {})
        serialized = {}
        for name, values in components.items():
            arr = np.asarray(values, dtype=float)
            serialized[name] = arr.tolist()
        return serialized

    def _build_series_template(self, series_name, components, labels, metadata):
        """
        Build a series template that matches SyntheticDailyGenerator template structure.
        
        The template structure is designed to be compatible with both:
        - SyntheticDailyGenerator.render_template() for reconstruction
        - FeatureDetectionLoss for evaluation
        """
        component_modes = {
            'trend': 'detected_trend',
            'level_shift': 'detected_level_shift',
            'seasonality': 'detected_additive',
            'holidays': 'detected_holiday',
            'anomalies': 'detected_residual',
            'noise': 'detected_noise',
        }
        component_dict = {}
        for name, values in components.items():
            arr = np.asarray(values, dtype=float)
            entry = {'values': arr.tolist()}
            entry['mode'] = component_modes.get(name, 'detected')
            component_dict[name] = entry
        
        # Extract seasonality profile for better alignment with synthetic generator
        seasonality_profile = metadata.get('seasonality_profiles', {})
        if not seasonality_profile:
            seasonality_profile = {
                'combined': metadata.get('seasonality_strength', 0.0),
                'weekly': 0.0,
                'yearly': 0.0,
            }
        
        label_dict = {
            'trend_changepoints': labels.get('trend_changepoints', []),
            'level_shifts': labels.get('level_shifts', []),
            'anomalies': labels.get('anomalies', []),
            'holiday_impacts': self._serialize_datetime_mapping(labels.get('holiday_impacts', {})),
            'holiday_coefficients': labels.get('holiday_coefficients', {}),
            'holiday_dates': [pd.Timestamp(x).isoformat() for x in labels.get('holiday_dates', [])],
            'holiday_splash_impacts': self._serialize_datetime_mapping(labels.get('holiday_splash_impacts', {})),
            'seasonality_changepoints': labels.get('seasonality_changepoints', []),
            'noise_changepoints': labels.get('noise_changepoints', []),
        }
        
        return {
            'series_name': series_name,
            'series_type': 'detected',
            'scale_factor': metadata.get('series_scale', 1.0),
            'combination': 'additive',
            'components': component_dict,
            'labels': label_dict,
            'metadata': {
                'seasonality_strengths': seasonality_profile,
                'noise_to_signal_ratio': metadata.get('noise_to_signal_ratio'),
                'noise_level': metadata.get('noise_level', 0.0),
                'series_scale': metadata.get('series_scale', 1.0),
            },
        }

    def get_detected_features(self, series_name=None, include_components=False, include_metadata=True):
        if self.df_original is None:
            raise RuntimeError("TimeSeriesFeatureDetector has not been fit.")

        def _default_seasonality_profile(name):
            base_strength = self.seasonality_strength.get(name, 0.0)
            return self.series_seasonality_profiles.get(name, {'combined': base_strength})

        if series_name is not None:
            if series_name not in self.df_original.columns:
                raise ValueError(f"Series '{series_name}' not found in detected features.")
            features = {
                'trend_changepoints': copy.deepcopy(self.trend_changepoints.get(series_name, [])),
                'level_shifts': copy.deepcopy(self.level_shifts.get(series_name, [])),
                'anomalies': copy.deepcopy(self.anomalies.get(series_name, [])),
                'holiday_dates': copy.deepcopy(self.holiday_dates.get(series_name, [])),
                'holiday_impacts': copy.deepcopy(self.holiday_impacts.get(series_name, {})),
                'holiday_coefficients': copy.deepcopy(self.holiday_coefficients.get(series_name, {})),
                'holiday_splash_impacts': copy.deepcopy(self.holiday_splash_impacts.get(series_name, {})),
                'seasonality_changepoints': [],
                'noise_changepoints': copy.deepcopy(self.noise_changepoints.get(series_name, [])),
                'seasonality_strength': self.seasonality_strength.get(series_name, 0.0),
                'series_seasonality_strengths': copy.deepcopy(_default_seasonality_profile(series_name)),
            }
            if include_metadata:
                features.update({
                    'noise_to_signal_ratio': self.noise_to_signal_ratios.get(series_name, 0.0),
                    'series_noise_level': self.series_noise_levels.get(series_name, 0.0),
                    'series_scale': self.series_scales.get(series_name, 0.0),
                    'series_type': 'detected',
                    'regressor_impacts': {},
                })
            if include_components:
                features['components'] = self._serialize_components(series_name)
            return features

        series_names = list(self.df_original.columns)
        trend_cp = {name: copy.deepcopy(self.trend_changepoints.get(name, [])) for name in series_names}
        level_shifts = {name: copy.deepcopy(self.level_shifts.get(name, [])) for name in series_names}
        anomalies = {name: copy.deepcopy(self.anomalies.get(name, [])) for name in series_names}
        holiday_dates = {name: copy.deepcopy(self.holiday_dates.get(name, [])) for name in series_names}
        holiday_impacts = {name: copy.deepcopy(self.holiday_impacts.get(name, {})) for name in series_names}
        holiday_coefficients = {name: copy.deepcopy(self.holiday_coefficients.get(name, {})) for name in series_names}
        holiday_splash = {
            name: copy.deepcopy(self.holiday_splash_impacts.get(name, {}))
            for name in series_names
        }
        seasonality_changepoints = {name: [] for name in series_names}
        noise_changepoints = {name: copy.deepcopy(self.noise_changepoints.get(name, [])) for name in series_names}
        seasonality_strength = {name: self.seasonality_strength.get(name, 0.0) for name in series_names}
        seasonality_profiles = {name: copy.deepcopy(_default_seasonality_profile(name)) for name in series_names}

        features = {
            'trend_changepoints': trend_cp,
            'level_shifts': level_shifts,
            'anomalies': anomalies,
            'holiday_dates': holiday_dates,
            'holiday_impacts': holiday_impacts,
            'holiday_coefficients': holiday_coefficients,
            'holiday_splash_impacts': holiday_splash,
            'seasonality_changepoints': seasonality_changepoints,
            'noise_changepoints': noise_changepoints,
            'seasonality_strength': seasonality_strength,
            'series_seasonality_strengths': seasonality_profiles,
        }

        if include_metadata:
            features.update({
                'noise_to_signal_ratios': {name: self.noise_to_signal_ratios.get(name, 0.0) for name in series_names},
                'series_noise_levels': {name: self.series_noise_levels.get(name, 0.0) for name in series_names},
                'series_scales': {name: self.series_scales.get(name, 0.0) for name in series_names},
                'series_types': {name: 'detected' for name in series_names},
                'regressor_impacts': {name: {} for name in series_names},
            })

        if include_components:
            features['components'] = {
                name: self._serialize_components(name)
                for name in series_names
            }

        features['shared_events'] = copy.deepcopy(self.shared_events)
        return features

    def get_template(self, deep=True):
        if self.template is None:
            return None
        return copy.deepcopy(self.template) if deep else self.template
    
    @classmethod
    def render_template(cls, template, return_components=False):
        """
        Render a feature detection template back into time series data.
        """
        # Delegate to SyntheticDailyGenerator's render_template
        # This ensures consistent rendering logic
        return SyntheticDailyGenerator.render_template(template, return_components=return_components)

    def get_cleaned_data(self, series_name=None):
        """
        Return cleaned time series data with anomalies, noise, and level shifts removed.
        
        The cleaned data consists of:
        - Trend (with mean included)
        - Seasonality
        - Holiday effects
        
        Level shifts are corrected by removing the cumulative shift effect, returning
        the data to its baseline level. Anomalies and noise are excluded entirely.
        
        Parameters
        ----------
        series_name : str, optional
            If provided, return cleaned data for only this series.
            If None, return cleaned data for all series.
        
        Returns
        -------
        pd.DataFrame
            Cleaned time series data with the same index as the original data.
            If series_name is specified, returns a DataFrame with a single column.
        
        Raises
        ------
        RuntimeError
            If fit() has not been called yet.
        ValueError
            If series_name is provided but not found in the original data.
        
        Examples
        --------
        >>> detector = TimeSeriesFeatureDetector()
        >>> detector.fit(df)
        >>> cleaned = detector.get_cleaned_data()
        >>> cleaned_single = detector.get_cleaned_data('series_1')
        """
        if self.df_original is None:
            raise RuntimeError("Call fit() before get_cleaned_data().")
        
        if series_name is not None:
            if series_name not in self.df_original.columns:
                raise ValueError(f"Series '{series_name}' not found in original data.")
            series_names = [series_name]
        else:
            series_names = list(self.df_original.columns)
        
        # Get components in original scale
        cleaned_data = pd.DataFrame(index=self.date_index)
        
        for name in series_names:
            components = self.components.get(name)
            if components is None:
                # If components not available, return the original series
                cleaned_data[name] = self.df_original[name]
                continue
            
            # Start with trend (which includes mean)
            trend = components.get('trend')
            if trend is None or not isinstance(trend, pd.Series):
                trend = pd.Series(0.0, index=self.date_index)
            
            # Add seasonality
            seasonality = components.get('seasonality')
            if seasonality is None or not isinstance(seasonality, pd.Series):
                seasonality = pd.Series(0.0, index=self.date_index)
            
            # Add holidays
            holidays = components.get('holidays')
            if holidays is None or not isinstance(holidays, pd.Series):
                holidays = pd.Series(0.0, index=self.date_index)
            
            # Combine: trend + seasonality + holidays
            # Note: level shifts are NOT included, effectively correcting for them
            cleaned_series = trend + seasonality + holidays
            
            # Ensure alignment with original index
            cleaned_data[name] = cleaned_series.reindex(self.date_index)
        
        return cleaned_data

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

    @staticmethod
    def get_new_params(method='random'):
        """Sample random parameters for detector optimization."""
        # Rough seasonality params (used for initial decomposition)
        rough_seasonality_params = DatepartRegressionTransformer.get_new_params(
            method=method, 
            holiday_countries_used=False
        )
        
        # Final seasonality params
        seasonality_params = DatepartRegressionTransformer.get_new_params(
            method=method, 
            holiday_countries_used=False
        )

        # Holiday params
        holiday_params = HolidayDetector.get_new_params(method=method)
        holiday_params['output'] = 'multivariate'  # Ensure correct output mode
        
        # Anomaly params
        method_choice, method_params, _ = anomaly_new_params(method=method)
        anomaly_params = {
            'output': 'multivariate',
            'method': method_choice,
            'method_params': method_params,
            'fillna': 'ffill',
        }
        
        # Changepoint params
        changepoint_params = ChangepointDetector.get_new_params(method=method)

        # Level shift params
        level_shift_params = LevelShiftMagic.get_new_params(method=method)
        level_shift_params['output'] = 'multivariate'  # Ensure correct output mode
        
        # General transformer params (for pre-trend processing)
        general_transformer_params = GeneralTransformer.get_new_params(
            method="filters",
            allow_none=True,
            transformer_max_depth=2
        )
        
        return {
            'rough_seasonality_params': rough_seasonality_params,
            'seasonality_params': seasonality_params,
            'holiday_params': holiday_params,
            'anomaly_params': anomaly_params,
            'changepoint_params': changepoint_params,
            'level_shift_params': level_shift_params,
            'general_transformer_params': general_transformer_params,
            'standardize': random.choice([True, False]),
            'smoothing_window': random.choice([None, 3, 5, 7]),
        }


class FeatureDetectionLoss:
    """
    Comprehensive loss calculator for feature detection optimization.

    Each synthetic label family contributes to the total loss:
    - Trend changepoints and slopes
    - Level shifts
    - Anomalies (including shared events and post patterns)
    - Holiday timing, direct impacts, and splash/bridge days
    - Seasonality strength, patterns, and changepoints
    - Noise regimes and noise-to-signal characteristics
    - Series-level metadata consistency (scale, type)
    - Regressor impacts when present
    """

    DEFAULT_WEIGHTS = {
        'trend_loss': 1.0,
        'level_shift_loss': 0.9,
        'anomaly_loss': 1.3,  # Increased from 1.1 - prioritize anomaly detection
        'holiday_event_loss': 1.2,  # Increased from 0.8 - penalize false holiday detections more
        'holiday_impact_loss': 0.9,  # Increased from 0.6 - ensure holiday impacts are strong enough
        'holiday_splash_loss': 0.5,
        'seasonality_strength_loss': 0.8,
        'seasonality_pattern_loss': 1.0,
        'seasonality_changepoint_loss': 0.6,
        'noise_level_loss': 0.5,
        'noise_regime_loss': 0.4,
        'metadata_loss': 0.2,
        'regressor_loss': 0.3,
    }

    def __init__(
        self,
        changepoint_tolerance_days=7,
        level_shift_tolerance_days=3,
        anomaly_tolerance_days=1,
        holiday_tolerance_days=1,
        seasonality_window=14,
        weights=None,
        holiday_over_anomaly_bonus=0.4,
        trend_component_penalty='component',
        trend_complexity_window=7,
        trend_complexity_weight=0.0,
        focus_component_weights=False,
    ):
        self.changepoint_tolerance_days = changepoint_tolerance_days
        self.level_shift_tolerance_days = level_shift_tolerance_days
        self.anomaly_tolerance_days = anomaly_tolerance_days
        self.holiday_tolerance_days = holiday_tolerance_days
        self.holiday_over_anomaly_bonus = holiday_over_anomaly_bonus
        self.seasonality_window = max(3, int(seasonality_window))

        raw_penalty_mode = (trend_component_penalty or 'component').lower()
        valid_modes = {'component', 'complexity'}
        if raw_penalty_mode not in valid_modes:
            raise ValueError(
                f"trend_component_penalty must be one of {sorted(valid_modes)}, "
                f"got '{trend_component_penalty}'"
            )
        self.trend_component_penalty = raw_penalty_mode
        if trend_complexity_window is None:
            trend_complexity_window = 7
        self.trend_complexity_window = max(3, int(trend_complexity_window))
        self.trend_complexity_weight = max(0.0, float(trend_complexity_weight))
        self.focus_component_weights = bool(focus_component_weights)

        self.weights = copy.deepcopy(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        if self.focus_component_weights:
            self._apply_component_focus_reweighting()

        self._change_tolerance = pd.Timedelta(self.changepoint_tolerance_days, unit='D')
        self._level_shift_tolerance = pd.Timedelta(self.level_shift_tolerance_days, unit='D')
        self._anomaly_tolerance = pd.Timedelta(self.anomaly_tolerance_days, unit='D')
        self._holiday_tolerance = pd.Timedelta(self.holiday_tolerance_days, unit='D')

    def _apply_component_focus_reweighting(self):
        """
        Down-weight trend penalties and up-weight non-trend features when requested.
        """
        emphasis = {
            'trend_loss': 0.6,
            'level_shift_loss': 1.1,
            'anomaly_loss': 1.15,
            'holiday_event_loss': 1.1,
            'holiday_impact_loss': 1.1,
            'holiday_splash_loss': 1.05,
            'seasonality_strength_loss': 1.05,
            'seasonality_pattern_loss': 1.15,
            'seasonality_changepoint_loss': 1.1,
        }
        for key, factor in emphasis.items():
            if key in self.weights:
                self.weights[key] = float(self.weights[key]) * factor

    def calculate_loss(
        self,
        detected_features,
        true_labels,
        series_name=None,
        true_components=None,
        date_index=None,
    ):
        """
        Calculate overall loss comparing detected features to true labels.

        Parameters
        ----------
        detected_features : dict
            Output from TimeSeriesFeatureDetector.get_detected_features(...)
        true_labels : dict
            Labels from SyntheticDailyGenerator.get_all_labels(...)
        series_name : str, optional
            If provided, only evaluate the named series.
        true_components : dict, optional
            Mapping of series -> component arrays from SyntheticDailyGenerator.get_components()
        date_index : pd.DatetimeIndex, optional
            Index used for the time series. Required for seasonality changepoint evaluation.

        Returns
        -------
        dict
            Loss breakdown with per-component metrics and total weighted loss.
        """
        if detected_features is None or true_labels is None:
            raise ValueError('detected_features and true_labels must be provided.')

        detected_components = self._resolve_components(
            detected_features.get('components') if isinstance(detected_features, dict) else None,
            series_name,
        )
        true_components = self._resolve_components(true_components, series_name)

        series_names = self._resolve_series_names(detected_features, true_labels, series_name)
        if not series_names:
            return {'total_loss': 0.0}

        aggregate_loss = {key: 0.0 for key in self.weights}
        series_breakdown = {}

        for name in series_names:
            series_loss = self._calculate_series_loss(
                name,
                detected_features,
                true_labels,
                detected_components.get(name, {}),
                true_components.get(name, {}),
                date_index,
            )
            series_breakdown[name] = series_loss
            for key in self.weights:
                aggregate_loss[key] += series_loss.get(key, 0.0)

        n_series = len(series_names)
        for key in aggregate_loss:
            aggregate_loss[key] /= n_series

        total_loss = 0.0
        for key, value in aggregate_loss.items():
            total_loss += self.weights.get(key, 1.0) * value

        aggregate_loss['total_loss'] = total_loss
        aggregate_loss['series_breakdown'] = series_breakdown
        return aggregate_loss

    def _calculate_series_loss(
        self,
        series_name,
        detected_features,
        true_labels,
        detected_components,
        true_components,
        date_index,
    ):
        detected = self._extract_detected_series(detected_features, series_name)
        true = self._extract_true_series(true_labels, series_name)

        trend_loss = self._trend_loss(
            detected.get('trend_changepoints', []),
            true.get('trend_changepoints', []),
            detected_components,
            true_components,
        )
        level_shift_loss = self._level_shift_loss(
            detected.get('level_shifts', []),
            true.get('level_shifts', []),
            detected.get('trend_changepoints', []),
        )
        anomaly_loss = self._anomaly_loss(
            detected.get('anomalies', []),
            true.get('anomalies', []),
        )

        holiday_event_loss = self._holiday_event_loss(
            detected.get('holiday_dates', []),
            true.get('holiday_dates', []),
            detected.get('anomalies', []),
        )
        holiday_impact_loss = self._holiday_impact_loss(
            detected.get('holiday_impacts', {}),
            true.get('holiday_impacts', {}),
        )
        holiday_splash_loss = self._holiday_splash_loss(
            detected.get('holiday_impacts', {}),
            detected.get('anomalies', []),
            true.get('holiday_splash_impacts', {}),
        )

        seasonality_strength_loss = self._seasonality_strength_loss(
            detected.get('series_seasonality_strengths'),
            true.get('series_seasonality_strengths'),
        )
        seasonality_pattern_loss = self._seasonality_pattern_loss(
            detected_components,
            true_components,
        )
        seasonality_changepoint_loss = self._seasonality_changepoint_loss(
            detected.get('seasonality_changepoints', []),
            true.get('seasonality_changepoints', []),
            detected_components,
            true_components,
            date_index,
        )

        noise_level_loss = self._noise_level_loss(
            detected.get('series_noise_level'),
            true.get('series_noise_level'),
            detected.get('noise_to_signal_ratio'),
            true.get('noise_to_signal_ratio'),
        )
        noise_regime_loss = self._noise_regime_loss(
            detected.get('noise_changepoints', []),
            true.get('noise_changepoints', []),
        )

        metadata_loss = self._metadata_loss(
            detected.get('series_scale'),
            true.get('series_scale'),
            detected.get('series_type'),
            true.get('series_type'),
        )

        regressor_loss = self._regressor_loss(
            detected.get('regressor_impacts', {}),
            true.get('regressor_impacts', {}),
        )

        return {
            'trend_loss': trend_loss,
            'level_shift_loss': level_shift_loss,
            'anomaly_loss': anomaly_loss,
            'holiday_event_loss': holiday_event_loss,
            'holiday_impact_loss': holiday_impact_loss,
            'holiday_splash_loss': holiday_splash_loss,
            'seasonality_strength_loss': seasonality_strength_loss,
            'seasonality_pattern_loss': seasonality_pattern_loss,
            'seasonality_changepoint_loss': seasonality_changepoint_loss,
            'noise_level_loss': noise_level_loss,
            'noise_regime_loss': noise_regime_loss,
            'metadata_loss': metadata_loss,
            'regressor_loss': regressor_loss,
        }

    def _resolve_series_names(self, detected_features, true_labels, series_name):
        if series_name is not None:
            return [series_name]
        names = set()
        if isinstance(detected_features, dict):
            tc = detected_features.get('trend_changepoints')
            if isinstance(tc, dict):
                names.update(tc.keys())
            profiles = detected_features.get('series_seasonality_strengths')
            if isinstance(profiles, dict):
                names.update(profiles.keys())
        tc_true = true_labels.get('trend_changepoints')
        if isinstance(tc_true, dict):
            names.update(tc_true.keys())
        types_true = true_labels.get('series_types')
        if isinstance(types_true, dict):
            names.update(types_true.keys())
        return sorted(names)

    def _extract_detected_series(self, detected_features, series_name):
        if not isinstance(detected_features, dict):
            return copy.deepcopy(detected_features)
        tc = detected_features.get('trend_changepoints')
        if not isinstance(tc, dict):
            return copy.deepcopy(detected_features)

        def _fetch(singular, plural=None, default=None):
            if plural is None:
                plural = singular
            value = detected_features.get(plural, default)
            if isinstance(value, dict):
                return copy.deepcopy(value.get(series_name, default))
            return copy.deepcopy(value)

        return {
            'trend_changepoints': _fetch('trend_changepoints', 'trend_changepoints', []),
            'level_shifts': _fetch('level_shifts', 'level_shifts', []),
            'anomalies': _fetch('anomalies', 'anomalies', []),
            'holiday_dates': _fetch('holiday_dates', 'holiday_dates', []),
            'holiday_impacts': _fetch('holiday_impacts', 'holiday_impacts', {}),
            'holiday_splash_impacts': _fetch('holiday_splash_impacts', 'holiday_splash_impacts', {}),
            'seasonality_changepoints': _fetch('seasonality_changepoints', 'seasonality_changepoints', []),
            'noise_changepoints': _fetch('noise_changepoints', 'noise_changepoints', []),
            'series_seasonality_strengths': _fetch('series_seasonality_strengths', 'series_seasonality_strengths', {}),
            'seasonality_strength': _fetch('seasonality_strength', 'seasonality_strength', 0.0),
            'noise_to_signal_ratio': _fetch('noise_to_signal_ratio', 'noise_to_signal_ratios', 0.0),
            'series_noise_level': _fetch('series_noise_level', 'series_noise_levels', 0.0),
            'series_scale': _fetch('series_scale', 'series_scales', 0.0),
            'series_type': _fetch('series_type', 'series_types', 'detected'),
            'regressor_impacts': _fetch('regressor_impacts', 'regressor_impacts', {}),
        }

    def _extract_true_series(self, true_labels, series_name):
        tc = true_labels.get('trend_changepoints')
        if not isinstance(tc, dict):
            return copy.deepcopy(true_labels)

        def _fetch(singular, plural=None, default=None):
            if plural is None:
                plural = singular
            value = true_labels.get(plural, default)
            if isinstance(value, dict):
                return copy.deepcopy(value.get(series_name, default))
            return copy.deepcopy(value)

        return {
            'trend_changepoints': _fetch('trend_changepoints', 'trend_changepoints', []),
            'level_shifts': _fetch('level_shifts', 'level_shifts', []),
            'anomalies': _fetch('anomalies', 'anomalies', []),
            'holiday_dates': _fetch('holiday_dates', 'holiday_dates', []),
            'holiday_impacts': _fetch('holiday_impacts', 'holiday_impacts', {}),
            'holiday_splash_impacts': _fetch('holiday_splash_impacts', 'holiday_splash_impacts', {}),
            'seasonality_changepoints': _fetch('seasonality_changepoints', 'seasonality_changepoints', []),
            'noise_changepoints': _fetch('noise_changepoints', 'noise_changepoints', []),
            'noise_to_signal_ratio': _fetch('noise_to_signal_ratio', 'noise_to_signal_ratios', 0.0),
            'series_noise_level': _fetch('series_noise_level', 'series_noise_levels', 0.0),
            'series_seasonality_strengths': _fetch('series_seasonality_strengths', 'series_seasonality_strengths', {}),
            'series_scale': _fetch('series_scale', 'series_scales', 0.0),
            'series_type': _fetch('series_type', 'series_types', 'standard'),
            'regressor_impacts': _fetch('regressor_impacts', 'regressor_impacts', {}),
        }

    def _resolve_components(self, component_container, series_name):
        if component_container is None:
            return {}
        if series_name is not None:
            return {series_name: component_container.get(series_name, {})}
        return {name: comps for name, comps in component_container.items()}

    def _trend_loss(self, detected_cp, true_cp, detected_components, true_components):
        if not true_cp:
            return 0.25 * len(detected_cp)

        detected_entries = [self._parse_trend_event(event) for event in detected_cp]
        true_entries = [self._parse_trend_event(event) for event in true_cp]
        unmatched_detected = set(range(len(detected_entries)))

        sigma_days = max(self.changepoint_tolerance_days, 1) / 1.5
        true_magnitudes = [entry[3] for entry in true_entries if np.isfinite(entry[3])]
        avg_true_magnitude = np.mean(true_magnitudes) if true_magnitudes else 0.0
        magnitude_floor = max(0.05, avg_true_magnitude * 0.25)

        loss = 0.0
        score_threshold = 0.15  # ~9 days of tolerance, higher for tighter tolerance

        for true_date, true_prior, true_post, true_mag in true_entries:
            best_idx = None
            best_score = -np.inf
            best_metrics = None

            for idx in unmatched_detected:
                det_date, det_prior, det_post, det_mag = detected_entries[idx]
                dist_days = abs((det_date - true_date).days)

                distance_score = np.exp(-0.5 * (dist_days / (sigma_days + 1e-9)) ** 2)

                slope_change_true = true_post - true_prior
                slope_change_detected = det_post - det_prior
                mag_denom = max(abs(true_mag), magnitude_floor, 1e-3)
                slope_denom = max(abs(slope_change_true), magnitude_floor, 1e-3)

                magnitude_score = np.exp(-0.5 * (abs(det_mag - true_mag) / mag_denom) ** 2)
                slope_score = np.exp(-0.5 * (abs(slope_change_detected - slope_change_true) / slope_denom) ** 2)

                match_score = 0.5 * distance_score + 0.3 * magnitude_score + 0.2 * slope_score

                if match_score > best_score:
                    best_score = match_score
                    best_idx = idx
                    best_metrics = (distance_score, magnitude_score, slope_score, dist_days)

            if best_idx is not None and best_metrics is not None and best_score >= score_threshold:
                distance_score, magnitude_score, slope_score, dist_days = best_metrics
                unmatched_detected.discard(best_idx)

                combined_penalty = (
                    0.5 * (1.0 - distance_score)
                    + 0.3 * (1.0 - magnitude_score)
                    + 0.2 * (1.0 - slope_score)
                )

                if dist_days > self.changepoint_tolerance_days:
                    overshoot = dist_days - self.changepoint_tolerance_days
                    combined_penalty += min(overshoot / (self.changepoint_tolerance_days + 1e-6), 1.5) * 0.3

                loss += combined_penalty * (1.0 + min(abs(true_mag), 2.0))
            else:
                loss += 1.2 + abs(true_mag)

        if true_entries:
            for idx in unmatched_detected:
                det_date, _, _, det_mag = detected_entries[idx]
                nearest_distance = min(
                    abs((det_date - true_date).days) for true_date, _, _, _ in true_entries
                )
                proximity_score = np.exp(-0.5 * (nearest_distance / (sigma_days + 1e-9)) ** 2)
                loss += 0.15 + 0.25 * (1.0 - proximity_score) + 0.05 * min(det_mag, 2.0)
        else:
            loss += 0.25 * len(unmatched_detected)

        # Apply trend component or complexity penalty based on mode
        trend_detected_series = detected_components.get('trend')
        
        if self.trend_component_penalty == 'component':
            # Component mode: use RMSE when both detected and true components available
            if (trend_detected_series is not None
                and 'trend' in true_components
                and true_components.get('trend') is not None):
                loss += self._component_rmse_penalty(
                    trend_detected_series,
                    true_components['trend'],
                )
        elif self.trend_component_penalty == 'complexity':
            # Complexity mode: penalize wiggly trends when weight > 0
            if (trend_detected_series is not None
                and self.trend_complexity_weight > 0):
                complexity_penalty = self._trend_complexity_penalty(trend_detected_series)
                loss += self.trend_complexity_weight * complexity_penalty

        return loss

    def _trend_complexity_penalty(self, trend_values):
        if trend_values is None:
            return 0.0
        arr = np.atleast_1d(np.asarray(trend_values, dtype=float))
        mask = np.isfinite(arr)
        if mask.sum() < 5:
            return 0.0
        arr = arr[mask]
        if arr.size < 5:
            return 0.0

        series = pd.Series(arr)
        window = min(len(series), max(3, self.trend_complexity_window))
        smooth = series.rolling(window=window, center=True, min_periods=1).median()
        residual = series - smooth

        smooth_values = smooth.to_numpy(dtype=float)
        residual_values = residual.to_numpy(dtype=float)

        smooth_scale = np.nanstd(smooth_values)
        if smooth_scale < 1e-6 or not np.isfinite(smooth_scale):
            smooth_scale = np.nanmean(np.abs(smooth_values)) + 1e-6

        if smooth_scale <= 0 or not np.isfinite(smooth_scale):
            return 0.0

        residual_sq = residual_values ** 2
        if residual_sq.size > 0:
            cutoff = np.nanpercentile(residual_sq, 90)
            if np.isfinite(cutoff):
                residual_sq = np.clip(residual_sq, None, cutoff)
        residual_scale = np.sqrt(np.nanmean(residual_sq)) if residual_sq.size else 0.0

        if not np.isfinite(residual_scale):
            return 0.0

        penalty = residual_scale / (smooth_scale + 1e-6)
        return min(max(penalty, 0.0), 2.5)

    def _level_shift_loss(self, detected_ls, true_ls, detected_cp):
        if not true_ls:
            return 0.2 * len(detected_ls)
        detected_entries = [self._parse_level_shift_event(event) for event in detected_ls]
        true_entries = [self._parse_level_shift_event(event) for event in true_ls]
        changepoint_dates = [self._parse_trend_event(event)[0] for event in detected_cp]
        unmatched_detected = set(range(len(detected_entries)))
        loss = 0.0
        for true_date, true_mag in true_entries:
            best_idx = None
            best_dist = None
            for idx, (det_date, det_mag) in enumerate(detected_entries):
                dist = abs((det_date - true_date).days)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None and best_dist is not None and best_dist <= self.level_shift_tolerance_days:
                det_date, det_mag = detected_entries[best_idx]
                distance_penalty = best_dist / (self.level_shift_tolerance_days + 1e-9)
                magnitude_penalty = abs(det_mag - true_mag) / (abs(true_mag) + 1e-6)
                loss += 0.5 * distance_penalty + 0.5 * magnitude_penalty
                unmatched_detected.discard(best_idx)
            else:
                prox_cp = any(abs((cp_date - true_date).days) <= self.changepoint_tolerance_days for cp_date in changepoint_dates)
                if prox_cp:
                    loss += 0.5
                else:
                    loss += 0.8 + abs(true_mag)
        loss += 0.15 * len(unmatched_detected)
        return loss

    def _anomaly_loss(self, detected_anom, true_anom):
        if not true_anom:
            return 0.15 * len(detected_anom)  # Slightly increased penalty for false positives when no true anomalies
        detected_entries = [self._parse_anomaly_event(event) for event in detected_anom]
        true_entries = [self._parse_anomaly_event(event) for event in true_anom]
        used_detected = set()
        loss = 0.0
        for true_event in true_entries:
            true_date, true_mag, true_type, true_duration = true_event
            best_idx = None
            best_dist = None
            for idx, det_event in enumerate(detected_entries):
                det_date, *_ = det_event
                dist = abs((det_date - true_date).days)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None and best_dist is not None and best_dist <= self.anomaly_tolerance_days:
                det_event = detected_entries[best_idx]
                _, det_mag, det_type, det_duration = det_event
                mag_pen = abs(det_mag - true_mag) / (abs(true_mag) + 1e-6)
                type_pen = 0.0 if det_type == true_type else 0.3
                duration_pen = abs(det_duration - true_duration) / (true_duration + 1e-6)
                loss += 0.5 * mag_pen + 0.3 * type_pen + 0.2 * min(duration_pen, 2.0)
                used_detected.add(best_idx)
            else:
                # Increased penalty for missing anomalies - encourages detection
                loss += 1.5 if true_type in {'point_outlier', 'spike'} else 0.9
        false_positives = len(detected_entries) - len(used_detected)
        if false_positives > 0:
            # Reduced false positive penalty - softer on wrong predictions
            fp_penalty = (0.04 * false_positives) + (0.08 * np.sqrt(false_positives))
            loss += fp_penalty
        return loss

    def _holiday_event_loss(self, detected_holidays, true_holidays, detected_anomalies):
        if not true_holidays:
            # Significantly increased penalty for false positive holidays
            return 0.25 * len(detected_holidays)
        detected_dates = [pd.Timestamp(dt) for dt in detected_holidays]
        true_dates = [pd.Timestamp(dt) for dt in true_holidays]
        anomaly_dates = [self._parse_anomaly_event(event)[0] for event in detected_anomalies]
        loss = 0.0
        for true_date in true_dates:
            matches = [det for det in detected_dates if abs(det - true_date) <= self._holiday_tolerance]
            if matches:
                continue
            anomaly_match = [det for det in anomaly_dates if abs(det - true_date) <= self._anomaly_tolerance]
            if anomaly_match:
                loss += self.holiday_over_anomaly_bonus
            else:
                loss += 1.0
        false_positives = sum(
            1 for det in detected_dates
            if not any(abs(det - true_date) <= self._holiday_tolerance for true_date in true_dates)
        )
        if false_positives > 0:
            ratio = false_positives / max(len(true_dates), 1)
            # Significantly increased false positive penalty for holidays
            # Linear component: 0.35 per FP (was 0.12)
            # Ratio penalty: 1.2x when FP ratio > 0.5 (was 0.4x)
            # This heavily discourages over-detection
            loss += 0.35 * false_positives + 1.2 * max(ratio - 0.5, 0.0)
        return loss

    def _holiday_impact_loss(self, detected_impacts, true_impacts):
        if not true_impacts:
            return 0.1 * len(detected_impacts)  # Slight increase for FP penalty
        detected = self._normalize_holiday_dict(detected_impacts)
        true = self._normalize_holiday_dict(true_impacts)
        loss = 0.0
        for date, true_value in true.items():
            det_value = detected.get(date, None)
            if det_value is None:
                # Missing impact - strong penalty
                loss += 0.8 + abs(true_value) * 0.5
            else:
                penalty = abs(det_value - true_value) / (abs(true_value) + 1e-6)
                if abs(true_value) > 1e-6:
                    relative_mag = abs(det_value) / (abs(true_value) + 1e-6)
                    # Significantly increased penalty when detected impact is too weak
                    # This encourages stronger holiday impact detection
                    if relative_mag < 0.3:
                        penalty *= 2.0  # Very weak detection gets 2x penalty
                    elif relative_mag < 0.5:
                        penalty *= 1.5  # Somewhat weak detection gets 1.5x penalty
                    elif relative_mag < 0.7:
                        penalty *= 1.2  # Slightly weak detection gets 1.2x penalty
                loss += min(penalty, 2.5)
        extras = len([date for date in detected if date not in true])
        # Increased FP penalty for holiday impacts
        loss += 0.15 * extras
        return loss

    def _holiday_splash_loss(self, detected_impacts, detected_anomalies, true_splash):
        if not true_splash:
            return 0.0
        detected = self._normalize_holiday_dict(detected_impacts)
        anomaly_dates = [self._parse_anomaly_event(event)[0] for event in detected_anomalies]
        loss = 0.0
        for date, magnitude in self._normalize_holiday_dict(true_splash).items():
            found = date in detected or any(abs(date - anomaly) <= self._anomaly_tolerance for anomaly in anomaly_dates)
            if not found:
                loss += 0.4 + 0.3 * min(abs(magnitude), 2.0)
        return loss

    def _seasonality_strength_loss(self, detected_strengths, true_strengths):
        if not true_strengths:
            return 0.0
        detected_strengths = detected_strengths or {}
        loss = 0.0
        for key, true_value in true_strengths.items():
            det_value = detected_strengths.get(key)
            if det_value is None:
                det_value = detected_strengths.get('combined', detected_strengths.get('seasonality_strength'))
            if det_value is None:
                loss += 0.5 + abs(true_value)
            else:
                penalty = abs(det_value - true_value) / (abs(true_value) + 1e-6)
                loss += min(penalty, 2.0)
        return loss / max(1, len(true_strengths))

    def _seasonality_pattern_loss(self, detected_components, true_components):
        detected_series = detected_components.get('seasonality')
        true_series = true_components.get('seasonality')
        if detected_series is None or true_series is None:
            return 0.5
        return self._component_rmse_penalty(detected_series, true_series)

    def _seasonality_changepoint_loss(self, detected_cp, true_cp, detected_components, true_components, date_index):
        if not true_cp:
            return 0.1 * len(detected_cp or [])
        if date_index is None:
            return 0.5 * len(true_cp)
        detected_dates = [self._parse_generic_date(event) for event in (detected_cp or [])]
        seasonality_array = np.asarray(detected_components.get('seasonality', []), dtype=float)
        true_array = np.asarray(true_components.get('seasonality', []), dtype=float)
        if seasonality_array.size == 0 or seasonality_array.size != len(date_index):
            return 0.6 * len(true_cp)
        loss = 0.0
        for event in true_cp:
            cp_date = self._parse_generic_date(event)
            if cp_date is None:
                continue
            match = any(abs(cp_date - det_date) <= self._change_tolerance for det_date in detected_dates)
            if match:
                continue
            idx = date_index.get_indexer([cp_date], method='nearest')[0]
            left_slice = slice(max(0, idx - self.seasonality_window), idx)
            right_slice = slice(idx, min(len(seasonality_array), idx + self.seasonality_window))
            left_mean = np.nanmean(seasonality_array[left_slice]) if left_slice.stop > left_slice.start else np.nan
            right_mean = np.nanmean(seasonality_array[right_slice]) if right_slice.stop > right_slice.start else np.nan
            true_left = np.nanmean(true_array[left_slice]) if true_array.size == seasonality_array.size else np.nan
            true_right = np.nanmean(true_array[right_slice]) if true_array.size == seasonality_array.size else np.nan
            if np.isnan(left_mean) or np.isnan(right_mean):
                loss += 0.6
            else:
                detected_change = abs(right_mean - left_mean)
                expected_change = abs(true_right - true_left) if not np.isnan(true_left) and not np.isnan(true_right) else np.nan
                if np.isnan(expected_change) or expected_change == 0:
                    penalty = 0.6 if detected_change < 0.1 else 0.0
                else:
                    penalty = max(0.0, 1.0 - detected_change / (expected_change + 1e-6))
                loss += min(penalty, 1.2)
        return loss / max(1, len(true_cp))

    def _noise_level_loss(self, detected_level, true_level, detected_ratio, true_ratio):
        penalties = []
        if true_level is not None:
            if detected_level is None:
                penalties.append(abs(true_level) + 0.5)
            else:
                penalties.append(abs(detected_level - true_level) / (abs(true_level) + 1e-6))
        if true_ratio is not None:
            if detected_ratio is None:
                penalties.append(abs(true_ratio) + 0.5)
            else:
                penalties.append(abs(detected_ratio - true_ratio) / (abs(true_ratio) + 1e-6))
        if not penalties:
            return 0.0
        return sum(min(p, 2.0) for p in penalties) / len(penalties)

    def _noise_regime_loss(self, detected_cp, true_cp):
        detected_dates = [self._parse_generic_date(event) for event in (detected_cp or [])]
        true_dates = [self._parse_generic_date(event) for event in (true_cp or [])]
        if not true_dates:
            return 0.05 * len(detected_dates)
        loss = 0.0
        for true_date in true_dates:
            match = any(abs(true_date - det_date) <= self._change_tolerance for det_date in detected_dates)
            if not match:
                loss += 0.6
        false_positives = len([
            det_date for det_date in detected_dates
            if not any(abs(det_date - true_date) <= self._change_tolerance for true_date in true_dates)
        ])
        loss += 0.1 * false_positives
        return loss

    def _metadata_loss(self, detected_scale, true_scale, detected_type, true_type):
        penalties = []
        if true_scale is not None:
            if detected_scale is None:
                penalties.append(abs(true_scale) + 0.5)
            else:
                penalties.append(abs(detected_scale - true_scale) / (abs(true_scale) + 1e-6))
        if true_type is not None:
            penalties.append(0.0 if detected_type == true_type else 0.3)
        if not penalties:
            return 0.0
        return sum(penalties) / len(penalties)

    def _regressor_loss(self, detected_regressors, true_regressors):
        true_regressors = true_regressors or {}
        if not true_regressors:
            return 0.0
        detected_regressors = detected_regressors or {}
        total = 0
        matched = 0
        for date, impacts in true_regressors.items():
            date = pd.Timestamp(date)
            detected_on_date = detected_regressors.get(date, detected_regressors.get(date.isoformat(), {}))
            for name in impacts:
                total += 1
                if detected_on_date and name in detected_on_date:
                    matched += 1
        if total == 0:
            return 0.0
        return (total - matched) / total

    def _parse_generic_date(self, event):
        if isinstance(event, dict):
            date = event.get('date')
        elif isinstance(event, (tuple, list)) and event:
            date = event[0]
        else:
            date = event
        if date is None:
            return None
        return pd.Timestamp(date)

    def _parse_trend_event(self, event):
        if isinstance(event, dict):
            date = pd.Timestamp(event.get('date'))
            prior = float(event.get('prior_slope', 0.0))
            post = float(event.get('new_slope', event.get('posterior_slope', prior)))
        elif isinstance(event, (tuple, list)) and len(event) >= 3:
            date = pd.Timestamp(event[0])
            prior = float(event[1]) if self._is_number(event[1]) else 0.0
            post = float(event[2]) if self._is_number(event[2]) else prior
        elif isinstance(event, (tuple, list)) and len(event) >= 2:
            date = pd.Timestamp(event[0])
            prior = 0.0
            post = float(event[1]) if self._is_number(event[1]) else 0.0
        else:
            date = pd.Timestamp(event)
            prior = 0.0
            post = 0.0
        magnitude = abs(post - prior)
        return date, prior, post, magnitude

    def _parse_level_shift_event(self, event):
        if isinstance(event, dict):
            date = pd.Timestamp(event.get('date'))
            magnitude = abs(float(event.get('magnitude', 1.0)))
        elif isinstance(event, (tuple, list)) and event:
            date = pd.Timestamp(event[0])
            magnitude = abs(float(event[1])) if len(event) > 1 and self._is_number(event[1]) else 1.0
        else:
            date = pd.Timestamp(event)
            magnitude = 1.0
        return date, magnitude

    def _parse_anomaly_event(self, event):
        if isinstance(event, dict):
            date = pd.Timestamp(event.get('date'))
            magnitude = abs(float(event.get('magnitude', 1.0)))
            anomaly_type = event.get('type', 'point_outlier')
            duration = int(event.get('duration', 1) or 1)
        elif isinstance(event, (tuple, list)) and event:
            date = pd.Timestamp(event[0])
            magnitude = abs(float(event[1])) if len(event) > 1 and self._is_number(event[1]) else 1.0
            anomaly_type = event[2] if len(event) > 2 else 'point_outlier'
            duration = int(event[3]) if len(event) > 3 and isinstance(event[3], (int, float)) else 1
        else:
            date = pd.Timestamp(event)
            magnitude = 1.0
            anomaly_type = 'point_outlier'
            duration = 1
        return date, magnitude, anomaly_type, duration

    @staticmethod
    def _normalize_holiday_dict(mapping):
        normalized = {}
        for key, value in (mapping or {}).items():
            try:
                normalized[pd.Timestamp(key)] = float(value)
            except (ValueError, TypeError):
                continue
        return normalized

    @staticmethod
    def _component_rmse_penalty(detected, true):
        detected_arr = np.asarray(detected, dtype=float)
        true_arr = np.asarray(true, dtype=float)
        length = min(detected_arr.size, true_arr.size)
        if length == 0:
            return 0.5
        detected_arr = detected_arr[:length]
        true_arr = true_arr[:length]
        mask = np.isfinite(detected_arr) & np.isfinite(true_arr)
        if not mask.any():
            return 0.5
        detected_arr = detected_arr[mask]
        true_arr = true_arr[mask]
        if detected_arr.size < 2:
            rmse = np.sqrt(np.nanmean((detected_arr - true_arr) ** 2))
            scale = np.nanstd(true_arr)
            if scale < 1e-6:
                scale = np.nanmean(np.abs(true_arr)) + 1e-6
            return min(rmse / (scale + 1e-6), 2.0)

        spread = np.nanstd(true_arr)
        if spread < 1e-6:
            spread = np.nanmean(np.abs(true_arr - np.nanmean(true_arr))) + 1e-6

        try:
            design = np.vstack([np.ones_like(true_arr), true_arr]).T
            coeffs, *_ = np.linalg.lstsq(design, detected_arr, rcond=None)
            intercept, slope = coeffs
        except np.linalg.LinAlgError:
            intercept, slope = 0.0, 1.0

        fitted = intercept + slope * true_arr
        residual = detected_arr - fitted
        residual_rmse = np.sqrt(np.nanmean(residual ** 2))

        normalized_residual = residual_rmse / (spread + 1e-6)
        slope_penalty = min(abs(slope - 1.0), 2.0)
        intercept_scale = spread + abs(np.nanmean(true_arr)) + 1e-6
        intercept_penalty = min(abs(intercept) / intercept_scale, 2.0)

        combined_penalty = (
            0.6 * normalized_residual
            + 0.25 * slope_penalty
            + 0.15 * intercept_penalty
        )
        return min(combined_penalty, 2.5)

    @staticmethod
    def _is_number(value):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False


class ReconstructionLoss(FeatureDetectionLoss):
    """
    Loss function tailored for real-world datasets lacking component-level labels.

    Focuses on reconstruction quality while discouraging overly complex trend fits and
    encouraging variance to be attributed to seasonality, holidays, anomalies, and level shifts.
    """

    DEFAULT_METRIC_WEIGHTS = {
        'reconstruction_loss': 1.0,
        'trend_smoothness_loss': 1.2,
        'trend_dominance_loss': 0.9,
        'seasonality_capture_loss': 0.8,
        'anomaly_capture_loss': 0.7,
    }

    def __init__(
        self,
        trend_complexity_window=7,
        trend_complexity_weight=1.0,
        metric_weights=None,
        trend_dominance_target=0.65,
        trend_min_other_variance=1e-4,
        seasonality_lags=(7, 365),
        seasonality_min_autocorr=0.1,
        seasonality_improvement_target=0.35,
        anomaly_improvement_target=0.25,
        anomaly_min_pre_std=1e-3,
    ):
        super().__init__(
            trend_component_penalty='complexity',
            trend_complexity_window=trend_complexity_window,
            trend_complexity_weight=trend_complexity_weight,
            focus_component_weights=True,
        )
        self.metric_weights = copy.deepcopy(self.DEFAULT_METRIC_WEIGHTS)
        if metric_weights:
            self.metric_weights.update(metric_weights)

        self.trend_dominance_target = float(trend_dominance_target)
        self.trend_min_other_variance = float(trend_min_other_variance)

        lag_set = [
            int(lag)
            for lag in (seasonality_lags or [])
            if lag is not None and lag > 0
        ]
        self.seasonality_lags = tuple(sorted(set(lag_set)))
        self.seasonality_min_autocorr = float(seasonality_min_autocorr)
        self.seasonality_improvement_target = float(seasonality_improvement_target)

        self.anomaly_improvement_target = float(anomaly_improvement_target)
        self.anomaly_min_pre_std = float(anomaly_min_pre_std)

    def calculate_loss(
        self,
        observed_df,
        detected_features,
        components=None,
        series_name=None,
    ):
        """
        Calculate reconstruction-oriented loss for unlabeled datasets.

        Parameters
        ----------
        observed_df : pd.DataFrame
            Original time series data used for detection.
        detected_features : dict
            Output from TimeSeriesFeatureDetector.get_detected_features(..., include_components=True).
        components : dict, optional
            Explicit component container matching `get_detected_features()['components']`.
        series_name : str, optional
            Restrict evaluation to a single series.

        Returns
        -------
        dict
            Loss metrics per series and aggregate total weighted loss.
        """
        if not isinstance(observed_df, pd.DataFrame):
            raise ValueError("observed_df must be a pandas DataFrame.")
        if detected_features is None:
            raise ValueError("detected_features must be provided.")

        component_container = components
        if component_container is None and isinstance(detected_features, dict):
            component_container = detected_features.get('components')
        if component_container is None:
            raise ValueError(
                "Component container not found. Pass include_components=True when obtaining detected_features "
                "or supply components explicitly."
            )

        resolved_components = self._resolve_components(component_container, series_name)

        if series_name is not None:
            if series_name not in observed_df.columns:
                raise ValueError(f"Series '{series_name}' not found in observed_df.")
            series_names = [series_name]
        else:
            series_names = [name for name in observed_df.columns if name in resolved_components]
            if not series_names:
                raise ValueError("No overlapping series between observed data and component container.")

        series_breakdown = {}
        aggregate_metrics = {key: 0.0 for key in self.metric_weights}

        for name in series_names:
            metrics = self._calculate_series_metrics(
                observed_series=observed_df[name],
                component_dict=resolved_components.get(name, {}),
            )
            series_breakdown[name] = metrics
            for key in self.metric_weights:
                aggregate_metrics[key] += metrics.get(key, 0.0)

        n_series = len(series_names)
        for key in aggregate_metrics:
            aggregate_metrics[key] /= n_series

        total_loss = 0.0
        for key, weight in self.metric_weights.items():
            total_loss += weight * aggregate_metrics.get(key, 0.0)

        aggregate_metrics['total_loss'] = total_loss
        aggregate_metrics['series_breakdown'] = series_breakdown
        return aggregate_metrics

    def _calculate_series_metrics(self, observed_series, component_dict):
        index = observed_series.index
        trend = self._component_to_series(component_dict.get('trend'), index)
        level_shift = self._component_to_series(component_dict.get('level_shift'), index)
        seasonality = self._component_to_series(component_dict.get('seasonality'), index)
        holidays = self._component_to_series(component_dict.get('holidays'), index)
        anomalies = self._component_to_series(component_dict.get('anomalies'), index)
        noise = self._component_to_series(component_dict.get('noise'), index)

        component_sum = trend + level_shift + seasonality + holidays + anomalies + noise
        residual = observed_series - component_sum

        reconstruction_loss = self._normalized_rmse(observed_series, residual)
        trend_smoothness = self._trend_complexity_penalty(trend.to_numpy(dtype=float))
        trend_dominance = self._trend_dominance_penalty(trend, {
            'level_shift': level_shift,
            'seasonality': seasonality,
            'holidays': holidays,
            'anomalies': anomalies,
        })

        seasonality_capture = self._seasonality_capture_penalty(
            observed_series,
            trend,
            level_shift,
            seasonality + holidays,
        )

        anomaly_capture = self._anomaly_capture_penalty(
            observed_series,
            trend,
            level_shift,
            seasonality + holidays,
            anomalies,
        )

        return {
            'reconstruction_loss': reconstruction_loss,
            'trend_smoothness_loss': trend_smoothness,
            'trend_dominance_loss': trend_dominance,
            'seasonality_capture_loss': seasonality_capture,
            'anomaly_capture_loss': anomaly_capture,
        }

    @staticmethod
    def _component_to_series(values, index):
        if values is None:
            return pd.Series(0.0, index=index, dtype=float)
        arr = np.asarray(values, dtype=float).flatten()
        series = pd.Series(arr, dtype=float)
        if series.size < len(index):
            tail = pd.Series(0.0, index=range(series.size, len(index)))
            series = pd.concat([series, tail])
        series = series.iloc[:len(index)]
        series.index = index
        return series.fillna(0.0)

    @staticmethod
    def _normalized_rmse(original_series, residual_series):
        residual = residual_series.to_numpy(dtype=float)
        orig = original_series.to_numpy(dtype=float)
        mask = np.isfinite(residual) & np.isfinite(orig)
        if not mask.any():
            return 0.0
        residual = residual[mask]
        orig = orig[mask]
        rmse = np.sqrt(np.mean(residual ** 2))
        scale = np.nanstd(orig)
        if scale < 1e-6 or not np.isfinite(scale):
            scale = np.nanmean(np.abs(orig)) + 1e-6
        return min(rmse / (scale + 1e-6), 3.0)

    def _trend_dominance_penalty(self, trend_series, component_map):
        trend_values = trend_series.to_numpy(dtype=float)
        trend_var = float(np.nanvar(trend_values))
        other_vars = 0.0
        for key in ('level_shift', 'seasonality', 'holidays', 'anomalies'):
            comp = component_map.get(key)
            if comp is None:
                continue
            comp_var = float(np.nanvar(comp.to_numpy(dtype=float)))
            other_vars += comp_var

        if other_vars < self.trend_min_other_variance:
            return 0.0

        total_var = trend_var + other_vars
        if total_var <= 0:
            return 0.0
        ratio = trend_var / total_var
        if ratio <= self.trend_dominance_target:
            return 0.0
        penalty = (ratio - self.trend_dominance_target) / (1.0 - self.trend_dominance_target + 1e-6)
        return min(max(penalty, 0.0), 2.0)

    def _seasonality_capture_penalty(self, observed, trend, level_shift, seasonal_bundle):
        if not self.seasonality_lags:
            return 0.0

        seasonal_std = float(np.nanstd(seasonal_bundle.to_numpy(dtype=float)))
        if seasonal_std < 1e-6:
            return 0.0

        detrended = observed - trend - level_shift
        residual_pre = detrended.to_numpy(dtype=float)
        residual_post = (detrended - seasonal_bundle).to_numpy(dtype=float)

        improvements = []
        for lag in self.seasonality_lags:
            if lag <= 0 or lag >= len(residual_pre):
                continue
            ac_pre = self._autocorrelation(residual_pre, lag)
            if abs(ac_pre) < self.seasonality_min_autocorr:
                continue
            ac_post = self._autocorrelation(residual_post, lag)
            improvement = max(0.0, (abs(ac_pre) - abs(ac_post)) / (abs(ac_pre) + 1e-6))
            improvements.append(improvement)

        if not improvements:
            return 0.0

        avg_improvement = float(np.mean(improvements))
        if avg_improvement >= self.seasonality_improvement_target:
            return 0.0
        deficit = self.seasonality_improvement_target - avg_improvement
        return min(max(deficit, 0.0), 1.5)

    def _anomaly_capture_penalty(self, observed, trend, level_shift, seasonal_bundle, anomalies):
        anomaly_std = float(np.nanstd(anomalies.to_numpy(dtype=float)))
        if anomaly_std < 1e-6:
            return 0.0

        residual = observed - trend - level_shift - seasonal_bundle
        pre_std = float(np.nanstd(residual.to_numpy(dtype=float)))
        if pre_std < self.anomaly_min_pre_std:
            return 0.0

        post_series = residual - anomalies
        post_std = float(np.nanstd(post_series.to_numpy(dtype=float)))
        if not np.isfinite(post_std):
            return 0.0

        improvement = max(0.0, (pre_std - post_std) / (pre_std + 1e-6))
        if improvement >= self.anomaly_improvement_target:
            return 0.0
        deficit = self.anomaly_improvement_target - improvement
        return min(max(deficit, 0.0), 1.5)

    @staticmethod
    def _autocorrelation(values, lag):
        values = np.asarray(values, dtype=float)
        if lag < 1 or lag >= values.size:
            return 0.0
        x = values[:-lag]
        y = values[lag:]
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            return 0.0
        x = x[mask]
        y = y[mask]
        if x.size < 2 or y.size < 2:
            return 0.0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.mean((x - x_mean) * (y - y_mean))
        denominator = np.std(x) * np.std(y) + 1e-9
        if denominator <= 0 or not np.isfinite(numerator):
            return 0.0
        return float(numerator / denominator)

class FeatureDetectionOptimizer:
    """
    Optimize TimeSeriesFeatureDetector parameters using synthetic labeled data.
    
    Uses a genetic-style search with balanced scoring to minimize detection loss.
    """
    
    def __init__(
        self,
        synthetic_generator,
        loss_calculator=None,
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
        n_iterations : int
            Number of random search iterations
        random_seed : int
            Random seed for reproducibility
        """
        self.synthetic_generator = synthetic_generator
        self.loss_calculator = loss_calculator or FeatureDetectionLoss()
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        
        self.best_params = None
        self.best_loss = float('inf')
        self.best_total_loss = float('inf')
        self.optimization_history = []
        self.baseline_loss = None
        self.history_df = None
    
    def optimize(self):
        """
        Run genetic-style optimization to find best detector parameters.
        
        Returns
        -------
        dict
            Best parameters found
        """
        self.best_params = None
        self.best_loss = float('inf')
        self.best_total_loss = float('inf')
        self.optimization_history = []
        self.baseline_loss = None

        return self._random_search()

    def _default_detector_params(self):
        """Return a deep-copied set of default detector parameters."""
        detector = TimeSeriesFeatureDetector()
        return {
            'rough_seasonality_params': copy.deepcopy(detector.rough_seasonality_params),
            'seasonality_params': copy.deepcopy(detector.seasonality_params),
            'holiday_params': copy.deepcopy(detector.holiday_params),
            'anomaly_params': copy.deepcopy(detector.anomaly_params),
            'changepoint_params': copy.deepcopy(detector.changepoint_params),
            'level_shift_params': copy.deepcopy(detector.level_shift_params),
            'general_transformer_params': copy.deepcopy(detector.general_transformer_params),
            'standardize': detector.standardize,
            'smoothing_window': detector.smoothing_window,
        }
    
    def _random_search(self):
        """Genetic-style optimization with balanced scoring."""
        rng = random.Random(self.random_seed)

        detector_for_sampling = TimeSeriesFeatureDetector()

        baseline_params = self._default_detector_params()
        evaluated_signatures = set()
        try:
            start_time = time.time()
            baseline_loss = self._evaluate_params(baseline_params)
            baseline_runtime = time.time() - start_time
            
            self.baseline_loss = baseline_loss['total_loss']
            baseline_history_entry = {
                'iteration': 'baseline',
                'params': copy.deepcopy(baseline_params),
                'loss': self.baseline_loss,
                'loss_breakdown': baseline_loss,
                'runtime': baseline_runtime,
            }
            self.optimization_history.append(baseline_history_entry)
            evaluated_signatures.add(self._param_signature(baseline_params))
            print(
                f"Baseline loss = {self.baseline_loss:.4f}, runtime = {baseline_runtime:.2f}s"
            )
        except Exception as e:
            print(f"Warning: Baseline evaluation failed with error: {e}")
            self.baseline_loss = None
        
        successful_iterations = 0
        failed_iterations = 0
        
        for i in range(self.n_iterations):
            params = None
            attempts = 0
            parent_pool = sorted(
                self.optimization_history,
                key=lambda x: x.get('balanced_loss', x.get('loss', float('inf'))),
            )
            parent_pool = parent_pool[: max(2, min(6, len(parent_pool)))] if parent_pool else []
            
            # Generate new parameters, avoiding duplicates
            while params is None or self._param_signature(params) in evaluated_signatures:
                attempts += 1
                if parent_pool and rng.random() < 0.7:
                    if len(parent_pool) >= 2:
                        chosen = rng.sample(parent_pool, 2)
                        params = self._crossover_params(chosen[0]['params'], chosen[1]['params'], rng)
                    else:
                        params = copy.deepcopy(parent_pool[0]['params'])
                    if rng.random() < 0.6:
                        params = self._mutate_params(params, detector_for_sampling, rng)
                else:
                    params = detector_for_sampling.get_new_params(method='random')
                if attempts > 8:
                    # Bug fix: ensure we have valid params even if duplicates persist
                    # Force a fresh random sample as last resort
                    params = detector_for_sampling.get_new_params(method='random')
                    break
            
            if params is None:
                continue
            
            # Double-check signature (may still be duplicate if max attempts reached)
            signature = self._param_signature(params)
            if signature in evaluated_signatures:
                continue

            try:
                start_time = time.time()
                loss = self._evaluate_params(params)
                runtime = time.time() - start_time
                
                record = {
                    'iteration': successful_iterations,
                    'params': copy.deepcopy(params),
                    'loss': loss['total_loss'],
                    'loss_breakdown': loss,
                    'runtime': runtime,
                }
                self.optimization_history.append(record)
                evaluated_signatures.add(signature)
                successful_iterations += 1
                
                # Print progress for every iteration
                if i % 20 == 0 or successful_iterations == 1:
                    print(
                        f"Iteration {i} ({successful_iterations} successful): "
                        f"raw loss = {loss['total_loss']:.4f}, runtime = {runtime:.2f}s"
                    )
            except Exception as e:
                failed_iterations += 1
                if failed_iterations <= 3:
                    print(f"Iteration {i} failed: {str(e)[:100]}")
                continue
        
        if failed_iterations > 3:
            print(f"... and {failed_iterations - 3} more failures (suppressed)")
        
        # Calculate runtime statistics
        runtimes = [entry.get('runtime') for entry in self.optimization_history if entry.get('runtime') is not None]
        if runtimes:
            avg_runtime = np.mean(runtimes)
            min_runtime = np.min(runtimes)
            max_runtime = np.max(runtimes)
            total_runtime = np.sum(runtimes)
        
        print(f"\nOptimization iterations complete!")
        print(f"Successful iterations: {successful_iterations}/{self.n_iterations}")
        
        # Print runtime statistics
        if runtimes:
            print(f"\nRuntime statistics:")
            print(f"  Total runtime: {total_runtime:.2f}s")
            print(f"  Average runtime per iteration: {avg_runtime:.2f}s")
            print(f"  Min runtime: {min_runtime:.2f}s")
            print(f"  Max runtime: {max_runtime:.2f}s")
        
        # Now select best model based on properly calculated balanced scores
        print(f"\nCalculating balanced scores and selecting best model...")
        best_params = self._select_best_from_history()
        
        return best_params
    
    def _evaluate_params(self, params):
        """Evaluate a parameter configuration."""
        # Create detector with these params
        detector = TimeSeriesFeatureDetector(**params)
        
        # Fit on synthetic data
        detector.fit(self.synthetic_generator.get_data())
        
        # Get detected features
        detected_features = detector.get_detected_features(include_components=True)
        
        # Get true labels
        true_labels = self.synthetic_generator.get_all_labels()
        true_components = self.synthetic_generator.get_components()
        
        # Calculate loss
        loss = self.loss_calculator.calculate_loss(
            detected_features,
            true_labels,
            true_components=true_components,
            date_index=self.synthetic_generator.date_index,
        )
        
        return loss

    @staticmethod
    def _param_signature(params):
        """Create a hashable signature for parameter configurations."""
        try:
            return json.dumps(params, sort_keys=True)
        except (TypeError, ValueError):
            # Fallback for non-JSON-serializable objects
            try:
                if isinstance(params, dict):
                    return repr(sorted(params.items()))
                else:
                    return repr(params)
            except Exception:
                # Last resort: use id (not ideal but prevents crashes)
                return str(id(params))

    def _crossover_params(self, parent_a, parent_b, rng):
        child = copy.deepcopy(parent_a)
        for key in child.keys():
            if key in parent_b and rng.random() < 0.5:
                child[key] = copy.deepcopy(parent_b[key])
        return child

    def _mutate_params(self, params, sampler, rng):
        mutated = copy.deepcopy(params)
        fresh = sampler.get_new_params(method='random')
        keys = list(mutated.keys())
        if not keys:
            return mutated
        count = max(1, min(len(keys), 2))
        for key in rng.sample(keys, count):
            mutated[key] = copy.deepcopy(fresh[key])
        return mutated

    def _select_best_from_history(self):
        """
        Post-process optimization history to select best model based on balanced scores.
        
        Converts history to DataFrame, calculates balanced scores with fixed scalers,
        and selects the model with the best balanced loss.
        
        Returns
        -------
        dict
            Best parameters based on balanced scoring
        """
        if not self.optimization_history:
            return None
        
        # Build DataFrame from history
        rows = []
        for entry in self.optimization_history:
            row = {
                'iteration': entry.get('iteration'),
                'loss': entry.get('loss'),
                'runtime': entry.get('runtime'),
            }
            # Add all loss breakdown components
            breakdown = entry.get('loss_breakdown', {})
            for key in self.loss_calculator.weights.keys():
                row[key] = breakdown.get(key, np.nan)
            rows.append(row)
        
        self.history_df = pd.DataFrame(rows)
        
        # Calculate scalers based on entire history (minimum positive value per metric)
        scalers = {}
        for key in self.loss_calculator.weights.keys():
            col = self.history_df[key].replace([np.inf, -np.inf], np.nan)
            positive = col[col > 0].dropna()
            if positive.empty:
                scalers[key] = 1.0
            else:
                scale = float(positive.min())
                if np.isfinite(scale) and scale > 1e-6:
                    scalers[key] = scale
                else:
                    scalers[key] = 1.0
        
        # Calculate balanced loss for each entry
        balanced_losses = []
        for idx, entry in enumerate(self.optimization_history):
            balanced = 0.0
            breakdown = entry.get('loss_breakdown', {})
            for key, weight in self.loss_calculator.weights.items():
                value = breakdown.get(key)
                if value is None or not np.isfinite(value):
                    continue
                balanced += weight * (value / scalers.get(key, 1.0))
            balanced_losses.append(balanced)
            # Store balanced loss back in history entry
            entry['balanced_loss'] = balanced
        
        self.history_df['balanced_loss'] = balanced_losses
        
        # Find best model based on balanced loss
        best_idx = np.argmin(balanced_losses)
        best_entry = self.optimization_history[best_idx]
        
        self.best_params = copy.deepcopy(best_entry['params'])
        self.best_loss = best_entry['balanced_loss']
        self.best_total_loss = best_entry['loss']
        
        # Find baseline entry for comparison
        baseline_entry = None
        for entry in self.optimization_history:
            if entry.get('iteration') == 'baseline':
                baseline_entry = entry
                break
        
        if baseline_entry:
            baseline_balanced = baseline_entry.get('balanced_loss', baseline_entry['loss'])
            improvement = baseline_balanced - self.best_loss
            improvement_pct = (improvement / baseline_balanced * 100) if baseline_balanced != 0 else 0
            
            print(f"\n{'='*80}")
            print(f"OPTIMIZATION RESULTS")
            print(f"{'='*80}")
            print(f"Baseline balanced loss: {baseline_balanced:.4f} (raw: {baseline_entry['loss']:.4f})")
            print(f"Best balanced loss:     {self.best_loss:.4f} (raw: {self.best_total_loss:.4f})")
            print(f"Improvement:            {improvement:.4f} ({improvement_pct:.2f}%)")
            print(f"Best found at iteration: {best_entry.get('iteration')}")
        
        return self.best_params

    def get_optimization_summary(self):
        """Return summary of optimization results."""
        summary = {
            'method': 'genetic_search',
            'n_iterations': len(self.optimization_history),
            'best_loss': self.best_loss,
            'baseline_loss': self.baseline_loss,
            'best_total_loss': self.best_total_loss,
            'best_params': copy.deepcopy(self.best_params) if self.best_params else None,
        }
        
        if self.optimization_history:
            losses = [
                h.get('balanced_loss', h.get('loss', float('inf')))
                for h in self.optimization_history
            ]
            summary['initial_loss'] = losses[0]
            summary['final_loss'] = losses[-1]
            summary['worst_loss'] = max(losses)
            summary['mean_loss'] = np.mean(losses)
            summary['std_loss'] = np.std(losses)
        
        return summary
