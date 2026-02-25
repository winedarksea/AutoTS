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
import datetime
from autots.tools.transform import (
    DatepartRegressionTransformer,
    AnomalyRemoval,
    LevelShiftMagic,
    GeneralTransformer,
)
from pandas.tseries.frequencies import to_offset
from autots.models.base import PredictionObject, stack_component_frames
from autots.evaluator.anomaly_detector import HolidayDetector
from autots.tools.changepoints import ChangepointDetector
from autots.tools.anomaly_utils import anomaly_new_params
from autots.tools.plotting import plot_feature_panels, HAS_MATPLOTLIB
from autots.tools.seasonal import date_part, build_adaptive_fourier_features
from autots.tools.fft import FFT
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
    TODO: Support identifying regressor impacts and granger lag impacts
    TODO: Build upon the JSON template so that it can be converted to a fixed size embedding (probably a 2d embedding). The fixed size may vary by parameters, but for a given parameter set should always be the same size. The embedding does not need to be capable of fully reconstructing the time series, just representing it.
    TODO: Support for modeling the trend with a fast kalman state space approach, ideally aligned with changepoints in some way if possible.
    TODO: consider also having "deviation from group" type anomaly detection for multivariate series
    TODO: Improve anomaly typing in univariate mode (currently defaults to point_outlier) and incorporate detector scores into type confidence.
    TODO: Detect and expose non-holiday regressor impacts (not just holiday coefficients), and persist them in template/features output.

        Parameters
    ----------
    seasonality_params : dict, optional
        Parameters for DatepartRegressionTransformer used in final seasonality fit
    rough_seasonality_params : dict, optional
        Parameters for DatepartRegressionTransformer used in initial rough seasonality decomposition (to improve holiday and anomaly detection).
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
    global_holiday_anomaly_suppression : bool, default=True
        If True, anomaly detection suppresses holiday-proximate flags using a merged holiday
        date set from all series. Set False to disable this suppression.
    """

    TEMPLATE_VERSION = "1.1"

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
        global_holiday_anomaly_suppression=True,
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
                    'bootstrap': True,
                },
            },
            'datepart_method': 'simple',
            'polynomial_degree': 2,
            'transform_dict': {
                'fillna': None,
                'transformations': {'0': 'EWMAFilter'},
                'transformation_params': {'0': {'span': 2}},
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
                    'max_iter': 500,
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
                    'alpha': 0.001,
                    'rolling_periods': 200,
                    'center': False,
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
            aggregate_method = (
                'individual' if self.detection_mode == 'multivariate' else 'mean'
            )
            self.changepoint_params = {
                'method': 'pelt',
                'method_params': {'penalty': 8, 'loss_function': 'l2'},
                'aggregate_method': aggregate_method,
                'min_segment_length': 14,
            }
        else:
            self.changepoint_params = changepoint_params.copy()
            # Override aggregate_method to match detection_mode if not explicitly set
            if (
                'aggregate_method' not in self.changepoint_params
                or self.changepoint_params['aggregate_method'] == 'auto'
            ):
                aggregate_method = (
                    'individual' if self.detection_mode == 'multivariate' else 'mean'
                )
                self.changepoint_params['aggregate_method'] = aggregate_method

        # Ensure level_shift_params uses the correct output mode
        if level_shift_params is None:
            self.level_shift_params = {
                'window_size': 364,
                'alpha': 1.8,
                'grouping_forward_limit': 3,
                'max_level_shifts': 10,
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
                0: {'method': 'clip', 'std_threshold': 3.5, 'fillna': None},
                1: {
                    'method': 'butter',
                    'method_args': {
                        'N': 3,
                        'btype': 'lowpass',
                        'analog': False,
                        'output': 'sos',
                        'Wn': 0.5,
                    },
                },
            },
        }
        self.smoothing_window = smoothing_window
        self.standardize = standardize
        self.global_holiday_anomaly_suppression = bool(
            global_holiday_anomaly_suppression
        )

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
        self.series_season_types = {}
        self.series_types = {}
        self.detected_seasonal_periods = None
        self._seasonality_changepoints = {}
        self.noise_changepoints = {}
        self.noise_to_signal_ratios = {}
        self.series_noise_levels = {}
        self.series_scales = {}
        self.shared_events = {'anomalies': [], 'level_shifts': []}
        self.reconstructed = None
        self.reconstructed_components = None
        self.reconstruction_error = None
        self.reconstruction_rmse = None
        self.optimized_detector_params = None
        self.synthetic_tuning_results = None
        self.synthetic_scale_multiplier = None
        self.tuned_synthetic_generator = None
        self.detector_optimization_summary = None
        self.detector_optimizer = None

    def _sanitize_holiday_params(self, holiday_params):
        """Return holiday detector parameters filtered to supported keys."""
        default_params = {
            'anomaly_detector_params': {
                'method': 'mad',
                'transform_dict': None,
                'forecast_params': None,
                'method_params': {'distribution': 'uniform', 'alpha': 0.05},
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
            'auto_relax': False,
            'relax_threshold_floor': 0.55,
            'relax_splash_threshold': 0.55,
            'relax_rounds': 2,
            'min_holidays_per_series': 1,
            'max_holidays_per_series': None,
            'holiday_selection_strategy': 'score',
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
        (
            final_residual,
            final_seasonality,
            seasonality_strength,
            holiday_component_scaled,
            holiday_coefficients,
            holiday_splash_impacts_scaled,
        ) = self._final_seasonality_fit(df_work, rough_residual, rough_seasonality)

        # Step 6-7: Trend and level shift detection
        # Pass holiday component so we know holidays are already removed in final_residual
        (
            trend_component_scaled,
            level_shift_component_scaled,
            validated_level_shifts,
            changepoints,
            slope_info,
        ) = self._detect_trend_and_shifts(final_residual, holiday_component_scaled)

        # Step 8: Noise analysis
        noise_component_scaled, anomaly_component_scaled = self._analyze_noise(
            df_work,
            trend_component_scaled,
            level_shift_component_scaled,
            final_seasonality,
            holiday_component_scaled,
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

    def forecast(self, forecast_length, frequency=None):
        """Generate a simple forward projection similar to BasicLinearModel.
        This detector is not optimized for forecasting; dedicated forecasting models may provide better results.
        """
        if self.df_original is None or self.date_index is None:
            raise ValueError(
                "TimeSeriesFeatureDetector must be fit before forecasting."
            )
        forecast_length = int(forecast_length)
        if forecast_length <= 0:
            raise ValueError("forecast_length must be a positive integer.")
        inferred = (
            frequency
            or getattr(self.date_index, 'freq', None)
            or pd.infer_freq(self.date_index)
        )
        if inferred is None:
            if len(self.date_index) < 2:
                raise ValueError("Unable to infer frequency from the training index.")
            inferred = self.date_index[-1] - self.date_index[-2]
        freq = to_offset(inferred)
        future_index = pd.date_range(
            start=self.date_index[-1], periods=forecast_length + 1, freq=freq
        )[1:]
        columns = self.df_original.columns
        zeros = pd.DataFrame(0.0, index=future_index, columns=columns)
        seasonal = zeros.copy()
        holidays = zeros.copy()
        train_reg = getattr(self, "_holiday_regressors_temp", None)
        if self.seasonality_model is not None:
            future_reg = None
            if (
                self.holiday_detector is not None
                and train_reg is not None
                and not getattr(train_reg, "empty", True)
            ):
                future_reg = self.holiday_detector.dates_to_holidays(
                    future_index, style='flag'
                )
                future_reg = future_reg.reindex(
                    columns=train_reg.columns, fill_value=0.0
                )
                if future_reg.empty:
                    future_reg = None
            if future_reg is not None:
                zero_reg = future_reg.copy()
                zero_reg.loc[:, :] = 0.0
                seasonal = self.seasonality_model.inverse_transform(
                    zeros, regressor=zero_reg
                )
                holidays = (
                    self.seasonality_model.inverse_transform(
                        zeros, regressor=future_reg
                    )
                    - seasonal
                )
            else:
                seasonal = self.seasonality_model.inverse_transform(zeros)
        trend = pd.DataFrame(0.0, index=future_index, columns=columns)
        # level shift by the logic of this detector might always be 0 for forecast, but for now it is here.
        level_shifts = pd.DataFrame(0.0, index=future_index, columns=columns)
        steps = np.arange(1, forecast_length + 1, dtype=float)
        for col in columns:
            comp = self.components.get(col, {})
            trend_hist = np.asarray(comp.get('trend', []), dtype=float)
            if trend_hist.size:
                mask = ~np.isnan(trend_hist)
                last_val = (
                    trend_hist[mask][-1]
                    if mask.any()
                    else float(self.df_original[col].iloc[-1])
                )
            else:
                last_val = float(self.df_original[col].iloc[-1])
            slope_info = self.trend_slopes.get(col, [])
            slope = float(slope_info[-1]['slope']) if slope_info else 0.0
            trend[col] = last_val + slope * steps
            level_shift_hist = np.asarray(comp.get('level_shift', []), dtype=float)
            if level_shift_hist.size:
                mask = ~np.isnan(level_shift_hist)
                last_level_shift = level_shift_hist[mask][-1] if mask.any() else 0.0
            else:
                last_level_shift = 0.0
            level_shifts[col] = last_level_shift
        seasonal = self._convert_to_original_scale(seasonal)
        holidays = self._convert_to_original_scale(holidays)

        component_frames = {
            'trend': trend,
            'level_shift': level_shifts,
            'seasonality': seasonal,
            'holidays': holidays,
        }
        components_df = stack_component_frames(component_frames)
        forecast = trend + level_shifts + seasonal + holidays
        return PredictionObject(
            model_name="TimeSeriesFeatureDetectorForecast",
            forecast_length=forecast_length,
            forecast_index=future_index,
            forecast_columns=columns,
            forecast=forecast,
            lower_forecast=forecast,
            upper_forecast=forecast,
            prediction_interval=0.0,
            predict_runtime=datetime.timedelta(0),
            fit_runtime=datetime.timedelta(0),
            model_parameters={'detection_mode': self.detection_mode},
            components=components_df,
        )

    def _prepare_data(self, df):
        """Prepare and standardize input data."""
        df_numeric = df.astype(float).copy().sort_index()
        self.df_original = df_numeric
        self.date_index = df_numeric.index

        if self.standardize:
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(df_numeric)
            df_work = pd.DataFrame(
                scaled, index=self.date_index, columns=df_numeric.columns
            )
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
            'global_holiday_anomaly_suppression': self.global_holiday_anomaly_suppression,
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
        self.series_season_types = {}
        self.series_types = {}
        self._seasonality_changepoints = {}
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
        (
            rough_residual,
            rough_seasonality,
            self.rough_seasonality_model,
        ) = self._compute_rough_seasonality(df_work)

        # Holiday detection
        holiday_dates, holiday_splash_dates, holiday_regressors = self._detect_holidays(
            rough_residual
        )
        self._holiday_dates_temp = holiday_dates
        self._holiday_regressors_temp = holiday_regressors

        # Optional: suppress holiday-proximate anomalies using a merged holiday
        # date set across all series to reduce holiday/anomaly double counting.
        if self.global_holiday_anomaly_suppression:
            combined_holiday_dates = self._flatten_holiday_dates(holiday_dates)
            self.anomaly_params.pop('holiday_dates', None)
            if combined_holiday_dates:
                self.anomaly_params['holiday_dates'] = combined_holiday_dates
                self.anomaly_params.setdefault('holiday_proximity_days', 2)

        # Anomaly detection
        residual_without_anomalies, anomaly_records = self._detect_anomalies(
            rough_residual
        )
        self._anomaly_records_temp = anomaly_records

        return rough_residual, rough_seasonality

    @staticmethod
    def _flatten_holiday_dates(holiday_dates_dict):
        """Flatten per-series holiday date lists into a single Timestamp set."""
        all_dates = set()
        if not holiday_dates_dict:
            return all_dates
        for dates in holiday_dates_dict.values():
            for date in dates:
                all_dates.add(pd.Timestamp(date))
        return all_dates

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
        (
            final_residual,
            final_seasonality,
            seasonality_strength,
            self.seasonality_model,
            holiday_component_scaled,
            holiday_coefficients,
            holiday_splash_impacts_scaled,
        ) = self._fit_final_seasonality(
            df_without_anomalies, self._holiday_regressors_temp
        )

        return (
            final_residual,
            final_seasonality,
            seasonality_strength,
            holiday_component_scaled,
            holiday_coefficients,
            holiday_splash_impacts_scaled,
        )

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
            self.general_transformer = GeneralTransformer(
                **self.general_transformer_params
            )
            residual_for_level_shifts = self.general_transformer.fit_transform(
                residual_for_level_shifts
            )
        if self.smoothing_window and self.smoothing_window > 1:
            residual_for_level_shifts = residual_for_level_shifts.rolling(
                window=int(self.smoothing_window),
                center=True,
                min_periods=1,
            ).mean()

        # Level shift detection on: original - anomalies - seasonality - holidays
        (
            level_shift_component_scaled,
            level_shift_candidates,
        ) = self._detect_level_shifts(residual_for_level_shifts)
        (
            level_shift_component_valid_scaled,
            validated_level_shifts,
        ) = self._validate_level_shifts(
            residual_for_level_shifts,
            level_shift_component_scaled,
            level_shift_candidates,
        )

        # Trend changepoint detection on: original - anomalies - seasonality - holidays - level_shifts
        trend_input = residual_for_level_shifts - level_shift_component_valid_scaled
        changepoints, trend_component_scaled = self._detect_trend_changepoints(
            trend_input
        )
        slope_info = self._compute_trend_slopes(trend_component_scaled, changepoints)

        return (
            trend_component_scaled,
            level_shift_component_valid_scaled,
            validated_level_shifts,
            changepoints,
            slope_info,
        )

    def _analyze_noise(
        self,
        df_work,
        trend_scaled,
        level_shift_scaled,
        seasonality_scaled,
        holiday_scaled,
    ):
        """
        Analyze noise component and anomalies.

        Returns
        -------
        tuple
            (noise_component, anomaly_component)
        """
        # Use the same anomaly transform path used during seasonality fitting so
        # anomaly removal is internally consistent across pipeline stages.
        df_without_anomalies_scaled = df_work.copy()
        if self.anomaly_detector is not None:
            try:
                transformed = self.anomaly_detector.transform(df_work)
                if transformed is not None:
                    df_without_anomalies_scaled = (
                        transformed.reindex(index=df_work.index, columns=df_work.columns)
                        .astype(float)
                    )
            except Exception as exc:
                warnings.warn(
                    f"Anomaly transform failed during noise analysis; using fallback interpolation. {exc}",
                    RuntimeWarning,
                )

        # Fallback: if transform did not materially change anything while anomaly
        # records exist, apply a conservative neighbor interpolation.
        if (
            df_without_anomalies_scaled.equals(df_work)
            and getattr(self, '_anomaly_records_temp', None)
        ):
            for col in df_work.columns:
                for anom in self._anomaly_records_temp.get(col, []):
                    date = anom.get('date')
                    if date not in df_work.index:
                        continue
                    try:
                        idx = df_work.index.get_loc(date)
                        neighbors = []
                        if idx > 0:
                            neighbors.append(df_work[col].iloc[idx - 1])
                        if idx < len(df_work) - 1:
                            neighbors.append(df_work[col].iloc[idx + 1])
                        if neighbors:
                            df_without_anomalies_scaled.loc[date, col] = np.nanmedian(
                                neighbors
                            )
                    except Exception:
                        continue

        # Reconstruct signal without anomalies
        reconstructed_scaled = (
            trend_scaled + level_shift_scaled + seasonality_scaled + holiday_scaled
        )
        noise_component_scaled = df_without_anomalies_scaled - reconstructed_scaled
        anomaly_component_scaled = df_work - df_without_anomalies_scaled

        # Keep downstream losses/stats stable even if upstream removal produced NaN/Inf.
        noise_component_scaled = noise_component_scaled.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        anomaly_component_scaled = anomaly_component_scaled.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

        return noise_component_scaled, anomaly_component_scaled

    def _rescale_all_components(
        self,
        trend_scaled,
        level_shift_scaled,
        seasonality_scaled,
        holiday_scaled,
        noise_scaled,
        anomaly_scaled,
    ):
        """
        Convert all components from standardized to original scale.

        Returns
        -------
        dict
            Dictionary of component DataFrames in original scale
        """
        trend_component = self._convert_to_original_scale(
            trend_scaled, include_mean=True
        )
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

    def _build_template(
        self,
        components_original,
        validated_level_shifts,
        slope_info,
        changepoints,
        holiday_coefficients,
        holiday_splash_impacts_scaled,
        seasonality_strength,
    ):
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
        holiday_splash_impacts = self._extract_splash_impacts(
            holiday_splash_impacts_scaled, self._holiday_dates_temp
        )
        holiday_coefficients = self._rescale_holiday_coefficients(holiday_coefficients)
        validated_level_shifts = self._rescale_level_shifts(validated_level_shifts)
        slope_info = self._rescale_slope_info(slope_info)
        anomaly_records = self._rescale_anomalies(self._anomaly_records_temp)
        seasonality_changepoints = self._detect_seasonality_changepoints(
            self.df_original, seasonality_component
        )
        self._seasonality_changepoints = seasonality_changepoints

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

            trend_cp_entries, trend_cp_template = self._build_trend_label_entries(
                series_name, changepoints, slope_info
            )
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
            self.holiday_splash_impacts[series_name] = holiday_splash_impacts.get(
                series_name, {}
            )
            self.holiday_coefficients[series_name] = holiday_coeff_template
            self.seasonality_components[series_name] = seasonality_component[
                series_name
            ].to_numpy(copy=True)
            self.seasonality_strength[series_name] = seasonality_strength.get(
                series_name, 0.0
            )

            # Calculate noise metrics
            seasonality_series = seasonality_component[series_name]
            noise_series = noise_component[series_name]
            # Include trend, level shift, seasonality, and holidays in signal
            signal_series = (
                trend_component[series_name]
                + level_shift_component[series_name]
                + seasonality_component[series_name]
                + holiday_component[series_name]
            )
            numerator = float(np.nanstd(noise_series))
            denominator = float(np.nanstd(signal_series)) or 1e-9
            self.noise_to_signal_ratios[series_name] = numerator / denominator

            original_series = self.df_original[series_name]
            series_scale = float(np.nanstd(original_series)) or 1e-9
            self.series_scales[series_name] = series_scale

            # Normalize noise level against original series magnitude
            normalized_noise = numerator / (series_scale or 1e-9)
            self.series_noise_levels[series_name] = normalized_noise

            self.series_seasonality_profiles[
                series_name
            ] = self._estimate_seasonality_profile(seasonality_series, series_scale)
            noise_cp_entries = self._detect_noise_regime_changepoints(noise_series)
            self.noise_changepoints[series_name] = noise_cp_entries
            seasonality_cp_entries = seasonality_changepoints.get(series_name, [])
            season_type = self._infer_season_type(
                self.series_seasonality_profiles[series_name], seasonality_cp_entries
            )
            self.series_season_types[series_name] = season_type
            noise_acf1 = self._lag1_autocorrelation(noise_series.to_numpy(dtype=float))
            series_type = self._infer_series_type(
                season_type=season_type,
                noise_changepoints=noise_cp_entries,
                noise_ratio=self.noise_to_signal_ratios[series_name],
                noise_acf1=noise_acf1,
            )
            self.series_types[series_name] = series_type

            metadata = {
                'seasonality_strength': self.seasonality_strength[series_name],
                'noise_to_signal_ratio': self.noise_to_signal_ratios[series_name],
                'seasonality_profiles': self.series_seasonality_profiles[series_name],
                'noise_level': self.series_noise_levels[series_name],
                'series_scale': series_scale,
                'series_type': series_type,
                'season_type': season_type,
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
                    'holiday_splash_impacts': self.holiday_splash_impacts.get(
                        series_name, {}
                    ),
                    'seasonality_changepoints': seasonality_changepoints.get(
                        series_name, []
                    ),
                    'noise_changepoints': noise_cp_entries,
                },
                metadata,
            )
            self.template['series'][series_name] = template_entry

        # Handle shared events
        if mark_shared and len(self.df_original.columns) > 0:
            reference_series = self.df_original.columns[0]
            shared_anomalies = {
                self._date_to_day_offset(entry[0])
                for entry in self.anomalies.get(reference_series, [])
            }
            shared_level_shifts = {
                self._date_to_day_offset(entry[0])
                for entry in self.level_shifts.get(reference_series, [])
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
            holiday_flags = self.holiday_detector.dates_to_holidays(
                residual_df.index, style='series_flag'
            )
            holiday_regressors = self.holiday_detector.dates_to_holidays(
                residual_df.index, style='flag'
            )
            if holiday_regressors is None:
                holiday_regressors = pd.DataFrame(index=residual_df.index)
            else:
                holiday_regressors = (
                    holiday_regressors.reindex(residual_df.index)
                    .fillna(0.0)
                    .astype(float)
                )
                holiday_regressors = holiday_regressors.loc[
                    :, ~holiday_regressors.columns.duplicated()
                ]
        except Exception:
            holiday_flags = pd.DataFrame(
                0, index=residual_df.index, columns=residual_df.columns
            )
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
                    holiday_splash_dates[
                        col
                    ] = []  # Will be populated during final seasonality fit
            else:
                for col in residual_df.columns:
                    holiday_dates[col] = []
                    holiday_splash_dates[col] = []
        else:
            # Multivariate mode: each series has its own holiday flags
            for col in residual_df.columns:
                series_flags = (
                    holiday_flags[col]
                    if col in holiday_flags
                    else pd.Series(0, index=residual_df.index)
                )
                flagged = series_flags[series_flags > 0].index
                holiday_dates[col] = [pd.Timestamp(ix) for ix in flagged]
                holiday_splash_dates[
                    col
                ] = []  # Will be populated during final seasonality fit

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
                if (
                    hasattr(self.anomaly_detector, 'scores')
                    and not self.anomaly_detector.scores.empty
                ):
                    try:
                        score = float(self.anomaly_detector.scores.loc[date].iloc[0])
                    except Exception:
                        score = None
                anomaly_type = 'point_outlier'  # Simplified for univariate
                records.append(
                    {
                        'date': pd.Timestamp(date),
                        'magnitude': magnitude,
                        'score': score,
                        'type': anomaly_type,
                    }
                )
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
                    if (
                        hasattr(self.anomaly_detector, 'scores')
                        and not self.anomaly_detector.scores.empty
                    ):
                        try:
                            score = float(self.anomaly_detector.scores.loc[date, col])
                        except Exception:
                            score = None
                    anomaly_type = self._classify_anomaly_type(residual_df[col], date)
                    records.append(
                        {
                            'date': pd.Timestamp(date),
                            'magnitude': float(magnitude),
                            'score': score,
                            'type': anomaly_type,
                        }
                    )
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

        # Get baseline from a robust pre-anomaly window.
        lookback = 21
        start_idx = max(0, idx - lookback)
        gap = 3
        end_baseline = max(0, idx - gap)
        baseline_window = series.iloc[start_idx:end_baseline]
        if baseline_window.empty or len(baseline_window) < 3:
            baseline_window = series.iloc[max(0, idx - 7) : idx]
            if baseline_window.empty:
                return 'point_outlier'
        baseline = float(np.nanmedian(baseline_window))
        baseline_std = float(np.nanstd(baseline_window)) or 1e-9

        # Get anomaly magnitude
        anomaly_value = float(series.iloc[idx])
        anomaly_mag = abs(anomaly_value - baseline)
        if anomaly_mag < 1e-9:
            return 'point_outlier'

        # Check post-anomaly pattern
        lookahead = 10
        end_idx = min(len(series), idx + lookahead + 1)
        post_window = series.iloc[idx + 1 : end_idx]

        if post_window.empty or len(post_window) < 2:
            return 'point_outlier'

        # Analyze post-anomaly behavior relative to baseline.
        post_values = post_window.to_numpy(dtype=float)
        post_deviations = np.abs(post_values - baseline)

        # Check for noisy burst.
        n_outliers = np.sum(post_deviations > max(anomaly_mag * 0.4, baseline_std * 2.5))
        if n_outliers >= 3:
            return 'noisy_burst'

        # Check for decay patterns
        if len(post_deviations) >= 3:
            first_dev = post_deviations[0]
            mid_idx = len(post_deviations) // 2
            mid_dev = post_deviations[mid_idx]
            last_dev = post_deviations[-1]

            if first_dev > anomaly_mag * 0.25:
                # Exponential/impulse decay.
                if mid_dev < first_dev * 0.4 and last_dev < first_dev * 0.15:
                    return 'impulse_decay'

                # Gradual linear-ish decay.
                if last_dev < first_dev * 0.4:
                    diffs = np.diff(post_deviations[: min(6, len(post_deviations))])
                    if np.sum(diffs < 0) >= len(diffs) * 0.6:
                        return 'linear_decay'

                # Sustained transient shift that returns.
                sustained_count = np.sum(post_deviations[:4] > anomaly_mag * 0.25)
                if sustained_count >= 3 and last_dev < anomaly_mag * 0.15:
                    return 'transient_change'

        # Slope reversion: sharp onset then slower reversion over a longer horizon.
        if len(post_deviations) >= 5:
            extended_end = min(len(series), idx + 31)
            extended_window = series.iloc[idx + 1 : extended_end]
            if len(extended_window) >= 10:
                extended_devs = np.abs(extended_window.to_numpy(dtype=float) - baseline)
                if (
                    extended_devs[0] > anomaly_mag * 0.3
                    and extended_devs[-1] < extended_devs[0] * 0.5
                ):
                    decay_slope = np.polyfit(
                        np.arange(len(extended_devs)), extended_devs, 1
                    )[0]
                    if decay_slope < 0 and abs(decay_slope) < anomaly_mag * 0.05:
                        return 'slope_reversion'

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

        seasonality_params = copy.deepcopy(self.seasonality_params)

        # Adaptive Fourier mode: detect dominant periods with FFT, then augment
        # regressors with period-specific Fourier features.
        self.detected_seasonal_periods = None
        if seasonality_params.get('datepart_method') == 'adaptive_fourier':
            try:
                fft_model = FFT(n_harm=None, detrend='linear')
                fft_input = df.dropna(how='all').to_numpy(dtype=float)
                nan_mask = np.isfinite(fft_input).all(axis=1)
                if nan_mask.sum() >= 14:
                    fft_model.fit(fft_input[nan_mask])
                    detected = fft_model.detect_dominant_periods(
                        min_period=3, max_periods=5, power_threshold=0.1
                    )
                else:
                    detected = []

                if len(detected) >= 2:
                    self.detected_seasonal_periods = detected
                    adaptive_features = build_adaptive_fourier_features(
                        df.index, detected, max_order=12
                    )
                    if regressor is not None:
                        regressor = pd.concat([regressor, adaptive_features], axis=1)
                    else:
                        regressor = adaptive_features
                    seasonality_params['datepart_method'] = 'simple_3'
                else:
                    seasonality_params['datepart_method'] = 'common_fourier'
            except Exception:
                seasonality_params['datepart_method'] = 'common_fourier'

        model = DatepartRegressionTransformer(**seasonality_params)
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
                baseline_pred = model.inverse_transform(
                    zeros_df, regressor=zero_regressor
                )
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
                holiday_component = pd.DataFrame(
                    0.0, index=df.index, columns=df.columns
                )
                seasonal_component = seasonal_total
            holiday_coefficients = self._solve_holiday_coefficients(
                regressor, holiday_component
            )

        strength = self._compute_seasonality_strength(df, residual, seasonal_component)
        return (
            residual,
            seasonal_component,
            strength,
            model,
            holiday_component,
            holiday_coefficients,
            holiday_splash_impacts,
        )

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
            r_squared = (
                0.0 if total_var == 0 else max(0.0, min(1.0, 1 - resid_var / total_var))
            )
            if len(seasonal_clean) > 1:
                corr = self._safe_correlation(y_clean, seasonal_clean)
                corr_strength = max(0.0, corr**2) if np.isfinite(corr) else 0.0
            else:
                corr_strength = 0.0
            variance_ratio = (
                0.0 if total_var == 0 else min(1.0, np.var(seasonal_clean) / total_var)
            )
            combined = 0.6 * r_squared + 0.3 * corr_strength + 0.1 * variance_ratio
            strength[col] = max(0.0, min(1.0, combined))
        return strength

    @staticmethod
    def _safe_correlation(x, y):
        """Compute correlation robustly, returning 0 when variance is degenerate."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.sum() < 2:
            return 0.0
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]
        x_std = float(np.nanstd(x_arr))
        y_std = float(np.nanstd(y_arr))
        if x_std < 1e-12 or y_std < 1e-12:
            return 0.0
        x_center = x_arr - np.nanmean(x_arr)
        y_center = y_arr - np.nanmean(y_arr)
        denom = np.sqrt(np.sum(x_center**2) * np.sum(y_center**2)) + 1e-12
        if denom <= 0:
            return 0.0
        corr = float(np.sum(x_center * y_center) / denom)
        if not np.isfinite(corr):
            return 0.0
        return max(-1.0, min(1.0, corr))

    @staticmethod
    def _lag1_autocorrelation(values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        mask = np.isfinite(arr)
        arr = arr[mask]
        if arr.size < 3:
            return 0.0
        return TimeSeriesFeatureDetector._safe_correlation(arr[:-1], arr[1:])

    def _detect_noise_regime_changepoints(self, noise_series):
        """Detect changepoints in noise volatility regimes using rolling variance shifts."""
        if not isinstance(noise_series, pd.Series):
            noise_series = pd.Series(noise_series, index=self.date_index)
        values = noise_series.to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if finite_mask.sum() < 30:
            return []
        series = pd.Series(values, index=noise_series.index).interpolate(
            limit_direction='both'
        )

        n = len(series)
        window = min(90, max(21, n // 12))
        min_periods = max(7, window // 3)
        rolling_std = series.rolling(window=window, center=True, min_periods=min_periods).std()
        log_std = np.log(np.clip(rolling_std.to_numpy(dtype=float), 1e-9, None))
        diff = np.abs(np.diff(log_std, prepend=np.nan))

        valid = diff[np.isfinite(diff)]
        if valid.size == 0:
            return []
        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))
        threshold = median + max(3.5 * mad, 0.20)

        candidate_idx = np.where(diff > threshold)[0]
        if candidate_idx.size == 0:
            return []

        min_spacing = max(7, window // 3)
        filtered = []
        for idx in candidate_idx:
            if idx <= 0 or idx >= n:
                continue
            date = noise_series.index[idx]
            if not filtered or (date - filtered[-1]).days >= min_spacing:
                filtered.append(date)

        if not filtered:
            return []
        max_allowed = max(1, min(10, n // 60))
        return filtered[:max_allowed]

    @staticmethod
    def _infer_season_type(seasonality_profile, seasonality_changepoints):
        weekly = float((seasonality_profile or {}).get('weekly', 0.0) or 0.0)
        yearly = float((seasonality_profile or {}).get('yearly', 0.0) or 0.0)
        combined = float((seasonality_profile or {}).get('combined', 0.0) or 0.0)
        cp_count = len(seasonality_changepoints or [])

        if cp_count >= 2 and combined >= 0.005:
            return 'seasonality_changepoints'
        if cp_count == 1 and combined >= 0.005:
            return 'time_varying_seasonality'
        if weekly >= 0.01 and yearly >= 0.02:
            return 'weekly_yearly'
        if weekly >= 0.01:
            return 'weekly'
        if yearly >= 0.02:
            return 'yearly'
        return 'none'

    @staticmethod
    def _infer_series_type(season_type, noise_changepoints, noise_ratio, noise_acf1):
        if season_type == 'seasonality_changepoints':
            return 'seasonality_changepoints'
        if season_type == 'time_varying_seasonality':
            return 'time_varying_seasonality'

        noise_ratio = float(noise_ratio) if noise_ratio is not None else 0.0
        noise_acf1 = float(noise_acf1) if noise_acf1 is not None else 0.0
        regime_count = len(noise_changepoints or [])

        if regime_count >= 2 and noise_ratio >= 0.05:
            return 'variance_regimes'
        if abs(noise_acf1) >= 0.30 and noise_ratio >= 0.02:
            return 'autocorrelated_noise'
        return 'standard'

    def _estimate_seasonality_profile(self, seasonal_series, series_scale):
        """
        Estimate relative strength of weekly, yearly, and detected periodic signatures.

        Parameters
        ----------
        seasonal_series : pd.Series
            Estimated seasonal component for a single series.
        series_scale : float
            Standard deviation of the original series used for normalization.

        Returns
        -------
        dict
            Dictionary containing combined, weekly, yearly, and optional
            `period_{n}` strength estimates when adaptive Fourier periods are available.
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
        date_range_days = (
            (valid.index[-1] - valid.index[0]).days if len(valid.index) > 1 else 0
        )
        if date_range_days >= 180:
            yearly_groups = valid.groupby(valid.index.dayofyear).mean()
            if len(yearly_groups) > 1:
                yearly_strength = float(np.nanstd(yearly_groups)) / series_scale

        profile = {
            'combined': combined_strength,
            'weekly': weekly_strength,
            'yearly': yearly_strength,
        }

        if self.detected_seasonal_periods:
            total_var = float(np.nanvar(valid))
            for period, _ in self.detected_seasonal_periods:
                period_int = int(round(period))
                if period_int < 2 or period_int >= len(valid):
                    continue
                groups = valid.groupby(np.arange(len(valid)) % period_int).mean()
                if len(groups) > 1:
                    period_var = float(np.nanvar(groups))
                    period_strength = period_var / (total_var + 1e-12)
                    profile[f'period_{period_int}'] = min(1.0, period_strength)

        return profile

    def _detect_seasonality_changepoints(self, df, seasonal_component):
        """Detect dates where seasonality strength changes materially over time."""
        n = len(df)
        window_size = min(365, max(60, n // 3))
        stride = max(1, window_size // 2)

        changepoints = {}
        for col in df.columns:
            strengths = []
            dates = []
            for start in range(0, n - window_size, stride):
                end = start + window_size
                window_df = df.iloc[start:end]
                window_seasonal = seasonal_component.iloc[start:end]
                window_residual = window_df - window_seasonal
                strength = self._compute_seasonality_strength(
                    window_df[[col]], window_residual[[col]], window_seasonal[[col]]
                )
                strengths.append(strength.get(col, 0.0))
                dates.append(df.index[start + window_size // 2])

            col_cps = []
            for i in range(1, len(strengths)):
                prev_strength = strengths[i - 1]
                curr_strength = strengths[i]
                abs_change = abs(curr_strength - prev_strength)
                rel_denom = max(prev_strength, curr_strength, 0.02)
                rel_change = abs_change / rel_denom
                # Require meaningful absolute and relative deltas to avoid
                # near-zero baseline windows over-triggering changepoints.
                if abs_change >= 0.03 and rel_change > 0.3:
                    col_cps.append(
                        (
                            dates[i],
                            f"strength_change_{prev_strength:.2f}_to_{curr_strength:.2f}",
                        )
                    )
            changepoints[col] = col_cps
        return changepoints

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
            aligned = reconstructed.reindex(
                self.df_original.index, columns=self.df_original.columns
            )
            self.reconstruction_error = self.df_original - aligned
            mse = np.nanmean(np.square(self.reconstruction_error.to_numpy(dtype=float)))
            self.reconstruction_rmse = float(np.sqrt(mse)) if np.isfinite(mse) else None
            if isinstance(self.template, dict):
                self.template.setdefault('meta', {})[
                    'reconstruction_rmse'
                ] = self.reconstruction_rmse
        except Exception as exc:
            warnings.warn(f"Template reconstruction failed: {exc}", RuntimeWarning)
            self.reconstructed = None
            self.reconstructed_components = None
            self.reconstruction_error = None
            self.reconstruction_rmse = None

    def _detect_level_shifts(self, residual_df):
        self.level_shift_detector = LevelShiftMagic(**self.level_shift_params)
        self.level_shift_detector.fit(residual_df)
        lvlshft = self.level_shift_detector.lvlshft.reindex(residual_df.index).fillna(
            0.0
        )
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
            series_std = float(series.std()) or 1e-9
            series_iqr = (
                float(series.quantile(0.75) - series.quantile(0.25)) or 1e-9
            )
            series_median = float(np.nanmedian(series.to_numpy(dtype=float)))
            adaptive_abs_thresh = min(abs_thresh, series_std * 0.3)
            # Guard against near-zero medians which can explode relative thresholds.
            robust_scale = max(abs(series_median), series_iqr, series_std, 1e-6)
            dynamic_rel = 0.05 + 0.5 * (series_iqr / robust_scale)
            dynamic_rel = float(np.clip(dynamic_rel, 0.05, 0.75))
            adaptive_rel_thresh = min(rel_thresh, dynamic_rel)
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

                if (
                    abs_change >= adaptive_abs_thresh
                    or rel_change >= adaptive_rel_thresh
                ):
                    entries.append(
                        {
                            'date': date,
                            'magnitude': magnitude,
                            'validated_change': change,
                            'relative_change': rel_change,
                        }
                    )
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
            max_segments = (
                max((len(idx) + 1) for idx in changepoint_indices.values()) or 1
            )

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
        prefix_ty = np.vstack(
            [np.zeros((1, n_series)), np.cumsum(values * time_index[:, None], axis=0)]
        )
        prefix_t = np.concatenate(([0.0], np.cumsum(time_index)))
        prefix_t2 = np.concatenate(([0.0], np.cumsum(time_index**2)))

        prefix_y_T = prefix_y.T
        sum_y = np.take_along_axis(
            prefix_y_T, segment_ends.T, axis=1
        ) - np.take_along_axis(prefix_y_T, segment_starts.T, axis=1)
        sum_y = sum_y.T

        prefix_ty_T = prefix_ty.T
        sum_ty = np.take_along_axis(
            prefix_ty_T, segment_ends.T, axis=1
        ) - np.take_along_axis(prefix_ty_T, segment_starts.T, axis=1)
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
        denominator = lengths * sum_t2 - sum_t**2
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
            hinge_contrib = np.maximum(
                0.0, time_index[:, None, None] - hinge_positions[None, :, :]
            )
            trend_matrix += np.sum(hinge_contrib * slope_changes[None, :, :], axis=1)

        trend_component = pd.DataFrame(
            trend_matrix, index=self.date_index, columns=series_names
        )
        return changepoints, trend_component

    @staticmethod
    def _segment_slope(values, start_idx, end_idx):
        if end_idx <= start_idx:
            return 0.0
        segment = values[start_idx : end_idx + 1]
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
                slope = self._segment_slope(
                    trend_component[col].to_numpy(), 0, len(trend_component) - 1
                )
                slopes[col] = [
                    {
                        'start_date': self.date_index[0],
                        'end_date': self.date_index[-1],
                        'slope': float(slope),
                    }
                ]
                continue
            indices = [0] + [
                self.date_index.get_loc(date)
                for date in cp_dates
                if date in self.date_index
            ]
            indices = sorted(set(indices))
            if indices[-1] != len(trend_component) - 1:
                indices.append(len(trend_component) - 1)
            segment_info = []
            for start_idx, end_idx in zip(indices[:-1], indices[1:]):
                if end_idx <= start_idx:
                    continue
                slope = self._segment_slope(
                    trend_component[col].to_numpy(), start_idx, end_idx
                )
                segment_info.append(
                    {
                        'start_date': self.date_index[start_idx],
                        'end_date': self.date_index[end_idx],
                        'slope': float(slope),
                    }
                )
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
                converted.append(
                    {
                        'date': pd.Timestamp(entry['date']),
                        'magnitude': self._to_original_value(
                            entry['magnitude'], series_name
                        ),
                        'validated_change': self._to_original_value(
                            entry.get('validated_change', entry['magnitude']),
                            series_name,
                        ),
                        'relative_change': float(entry.get('relative_change', 0.0)),
                    }
                )
            rescaled[series_name] = converted
        return rescaled

    def _rescale_slope_info(self, slope_info):
        rescaled = {}
        for series_name, entries in slope_info.items():
            converted = []
            for entry in entries:
                converted.append(
                    {
                        'start_date': pd.Timestamp(entry['start_date']),
                        'end_date': pd.Timestamp(entry['end_date']),
                        'slope': self._to_original_value(entry['slope'], series_name),
                    }
                )
            rescaled[series_name] = converted
        return rescaled

    def _rescale_anomalies(self, anomaly_records):
        rescaled = {}
        for series_name, entries in anomaly_records.items():
            converted = []
            for entry in entries:
                converted.append(
                    {
                        'date': pd.Timestamp(entry['date']),
                        'magnitude': self._to_original_value(
                            entry['magnitude'], series_name
                        ),
                        'score': entry.get('score'),
                        'type': entry.get('type', 'spike'),
                    }
                )
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
            template_entries.append(
                {
                    'date': pd.Timestamp(cp_date).isoformat(),
                    'prior_slope': prior_slope,
                    'new_slope': new_slope,
                }
            )
        return entries, template_entries

    def _build_level_shift_entries(
        self, series_name, validated_level_shifts, shared=False
    ):
        entries = []
        template_entries = []
        for item in validated_level_shifts.get(series_name, []):
            date = pd.Timestamp(item['date'])
            magnitude = item['magnitude']
            entries.append((date, magnitude, 'validated', shared))
            template_entries.append(
                {
                    'date': date.isoformat(),
                    'magnitude': magnitude,
                    'shift_type': 'validated',
                    'shared': bool(shared),
                }
            )
        return entries, template_entries

    def _build_anomaly_entries(self, series_name, anomaly_records, shared=False):
        entries = []
        template_entries = []
        for item in anomaly_records.get(series_name, []):
            date = pd.Timestamp(item['date'])
            magnitude = item['magnitude']
            anomaly_type = item.get('type', 'point_outlier')
            entries.append((date, magnitude, anomaly_type, 1, shared))
            template_entries.append(
                {
                    'date': date.isoformat(),
                    'magnitude': magnitude,
                    'pattern': anomaly_type,
                    'duration': 1,
                    'shared': bool(shared),
                }
            )
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
            'holiday_impacts': self._serialize_datetime_mapping(
                labels.get('holiday_impacts', {})
            ),
            'holiday_coefficients': labels.get('holiday_coefficients', {}),
            'holiday_dates': [
                pd.Timestamp(x).isoformat() for x in labels.get('holiday_dates', [])
            ],
            'holiday_splash_impacts': self._serialize_datetime_mapping(
                labels.get('holiday_splash_impacts', {})
            ),
            'seasonality_changepoints': labels.get('seasonality_changepoints', []),
            'noise_changepoints': labels.get('noise_changepoints', []),
        }

        return {
            'series_name': series_name,
            'series_type': metadata.get('series_type', 'detected'),
            'scale_factor': metadata.get('series_scale', 1.0),
            'combination': 'additive',
            'components': component_dict,
            'labels': label_dict,
            'metadata': {
                'seasonality_strengths': seasonality_profile,
                'noise_to_signal_ratio': metadata.get('noise_to_signal_ratio'),
                'noise_level': metadata.get('noise_level', 0.0),
                'series_scale': metadata.get('series_scale', 1.0),
                'season_type': metadata.get('season_type', 'unknown'),
            },
        }

    def get_detected_features(
        self, series_name=None, include_components=False, include_metadata=True
    ):
        if self.df_original is None:
            raise RuntimeError("TimeSeriesFeatureDetector has not been fit.")

        def _default_seasonality_profile(name):
            base_strength = self.seasonality_strength.get(name, 0.0)
            return self.series_seasonality_profiles.get(
                name, {'combined': base_strength}
            )

        if series_name is not None:
            if series_name not in self.df_original.columns:
                raise ValueError(
                    f"Series '{series_name}' not found in detected features."
                )
            features = {
                'trend_changepoints': copy.deepcopy(
                    self.trend_changepoints.get(series_name, [])
                ),
                'level_shifts': copy.deepcopy(self.level_shifts.get(series_name, [])),
                'anomalies': copy.deepcopy(self.anomalies.get(series_name, [])),
                'holiday_dates': copy.deepcopy(self.holiday_dates.get(series_name, [])),
                'holiday_impacts': copy.deepcopy(
                    self.holiday_impacts.get(series_name, {})
                ),
                'holiday_coefficients': copy.deepcopy(
                    self.holiday_coefficients.get(series_name, {})
                ),
                'holiday_splash_impacts': copy.deepcopy(
                    self.holiday_splash_impacts.get(series_name, {})
                ),
                'seasonality_changepoints': copy.deepcopy(
                    getattr(self, '_seasonality_changepoints', {}).get(series_name, [])
                ),
                'noise_changepoints': copy.deepcopy(
                    self.noise_changepoints.get(series_name, [])
                ),
                'seasonality_strength': self.seasonality_strength.get(series_name, 0.0),
                'series_seasonality_strengths': copy.deepcopy(
                    _default_seasonality_profile(series_name)
                ),
            }
            if include_metadata:
                features.update(
                    {
                        'noise_to_signal_ratio': self.noise_to_signal_ratios.get(
                            series_name, 0.0
                        ),
                        'series_noise_level': self.series_noise_levels.get(
                            series_name, 0.0
                        ),
                        'series_scale': self.series_scales.get(series_name, 0.0),
                        'series_type': self.series_types.get(series_name, 'detected'),
                        'season_type': self.series_season_types.get(series_name, 'none'),
                        'regressor_impacts': {},
                    }
                )
            if include_components:
                features['components'] = self._serialize_components(series_name)
            return features

        series_names = list(self.df_original.columns)
        trend_cp = {
            name: copy.deepcopy(self.trend_changepoints.get(name, []))
            for name in series_names
        }
        level_shifts = {
            name: copy.deepcopy(self.level_shifts.get(name, []))
            for name in series_names
        }
        anomalies = {
            name: copy.deepcopy(self.anomalies.get(name, [])) for name in series_names
        }
        holiday_dates = {
            name: copy.deepcopy(self.holiday_dates.get(name, []))
            for name in series_names
        }
        holiday_impacts = {
            name: copy.deepcopy(self.holiday_impacts.get(name, {}))
            for name in series_names
        }
        holiday_coefficients = {
            name: copy.deepcopy(self.holiday_coefficients.get(name, {}))
            for name in series_names
        }
        holiday_splash = {
            name: copy.deepcopy(self.holiday_splash_impacts.get(name, {}))
            for name in series_names
        }
        seasonality_changepoints = {
            name: copy.deepcopy(
                getattr(self, '_seasonality_changepoints', {}).get(name, [])
            )
            for name in series_names
        }
        noise_changepoints = {
            name: copy.deepcopy(self.noise_changepoints.get(name, []))
            for name in series_names
        }
        seasonality_strength = {
            name: self.seasonality_strength.get(name, 0.0) for name in series_names
        }
        seasonality_profiles = {
            name: copy.deepcopy(_default_seasonality_profile(name))
            for name in series_names
        }

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
            features.update(
                {
                    'noise_to_signal_ratios': {
                        name: self.noise_to_signal_ratios.get(name, 0.0)
                        for name in series_names
                    },
                    'series_noise_levels': {
                        name: self.series_noise_levels.get(name, 0.0)
                        for name in series_names
                    },
                    'series_scales': {
                        name: self.series_scales.get(name, 0.0) for name in series_names
                    },
                    'series_types': {
                        name: self.series_types.get(name, 'detected')
                        for name in series_names
                    },
                    'season_types': {
                        name: self.series_season_types.get(name, 'none')
                        for name in series_names
                    },
                    'regressor_impacts': {name: {} for name in series_names},
                }
            )

        if include_components:
            features['components'] = {
                name: self._serialize_components(name) for name in series_names
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
        return SyntheticDailyGenerator.render_template(
            template, return_components=return_components
        )

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
            inferred_type = self.series_types.get(series_name, 'detected')
            season_type = self.series_season_types.get(series_name, 'none')
            print(f"Series Type (inferred): {inferred_type}")
            print(f"Season Type (inferred): {season_type}")
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

    def query_features(
        self,
        dates=None,
        series=None,
        include_components=False,
        include_metadata=False,
        return_json=False,
    ):
        """Query a specific slice of detected features with minimal token usage.

        Designed for LLM-friendly output with compact representation.

        Args:
            dates (str, datetime, list, slice): Date(s) to query for features.
                - Single date: "2024-01-15" or datetime object
                - Date range: slice("2024-01-01", "2024-01-31")
                - List of dates: ["2024-01-15", "2024-01-20"]
                - None: return all features (not filtered by date)
            series (str, list): Series name(s) to query.
                - Single series: "sales"
                - Multiple series: ["sales", "revenue"]
                - None: all series
            include_components (bool): Include component time series values for the date range
            include_metadata (bool): Include metadata like noise levels, scales, etc.
            return_json (bool): Return JSON string instead of dict

        Returns:
            dict or str: Compact feature data including anomalies, changepoints,
                        level shifts, holidays, and optionally components

        Examples:
            >>> # Get all features for one series
            >>> detector.query_features(series="sales")

            >>> # Get features occurring in a date range
            >>> detector.query_features(
            ...     dates=slice("2024-01-01", "2024-01-31"),
            ...     series=["sales", "revenue"]
            ... )

            >>> # Get components for specific dates
            >>> detector.query_features(
            ...     dates=["2024-01-15", "2024-01-16"],
            ...     series="sales",
            ...     include_components=True
            ... )
        """
        if self.df_original is None:
            raise RuntimeError("TimeSeriesFeatureDetector has not been fit.")

        # Handle series selection
        if series is None:
            selected_series = self.df_original.columns.tolist()
        elif isinstance(series, str):
            selected_series = [series]
        else:
            selected_series = list(series)

        # Validate series exist
        missing = set(selected_series) - set(self.df_original.columns)
        if missing:
            raise ValueError(f"Series not found: {missing}")

        # Handle date filtering
        date_filter = None
        if dates is not None:
            if isinstance(dates, slice):
                start = pd.to_datetime(dates.start) if dates.start else None
                stop = pd.to_datetime(dates.stop) if dates.stop else None
                date_filter = (start, stop)
            elif isinstance(dates, (list, pd.Index)):
                date_filter = set(pd.to_datetime(d) for d in dates)
            else:
                # Single date
                date_filter = {pd.to_datetime(dates)}

        def _filter_by_date(items, date_key_idx=0):
            """Filter list of tuples/lists by date (first element by default)."""
            if date_filter is None:
                return items

            if isinstance(date_filter, set):
                # Exact date matching
                return [
                    item
                    for item in items
                    if pd.to_datetime(item[date_key_idx]) in date_filter
                ]
            else:
                # Range matching
                start, stop = date_filter
                filtered = []
                for item in items:
                    dt = pd.to_datetime(item[date_key_idx])
                    if start is not None and dt < start:
                        continue
                    if stop is not None and dt > stop:
                        continue
                    filtered.append(item)
                return filtered

        def _filter_dict_by_date(date_dict):
            """Filter dict with datetime keys."""
            if date_filter is None:
                return date_dict

            if isinstance(date_filter, set):
                return {
                    k: v
                    for k, v in date_dict.items()
                    if pd.to_datetime(k) in date_filter
                }
            else:
                start, stop = date_filter
                filtered = {}
                for k, v in date_dict.items():
                    dt = pd.to_datetime(k)
                    if start is not None and dt < start:
                        continue
                    if stop is not None and dt > stop:
                        continue
                    filtered[k] = v
                return filtered

        # Build result
        result = {'detection_mode': self.detection_mode, 'series': {}}

        # Extract features for each series
        for col in selected_series:
            series_features = {}

            # Trend changepoints (list of [date, prior_slope, new_slope])
            changepoints = self.trend_changepoints.get(col, [])
            if changepoints:
                filtered_cp = _filter_by_date(changepoints)
                if filtered_cp:
                    series_features['trend_changepoints'] = [
                        {
                            'date': pd.to_datetime(cp[0]).isoformat(),
                            'prior_slope': float(cp[1]),
                            'new_slope': float(cp[2]),
                        }
                        for cp in filtered_cp
                    ]

            # Level shifts (list of [date, magnitude, shift_type, metadata])
            level_shifts = self.level_shifts.get(col, [])
            if level_shifts:
                filtered_ls = _filter_by_date(level_shifts)
                if filtered_ls:
                    series_features['level_shifts'] = [
                        {
                            'date': pd.to_datetime(ls[0]).isoformat(),
                            'magnitude': float(ls[1]),
                            'type': ls[2],
                        }
                        for ls in filtered_ls
                    ]

            # Anomalies (typically [date, magnitude, anomaly_type, duration, ...]).
            anomalies = self.anomalies.get(col, [])
            if anomalies:
                filtered_an = _filter_by_date(anomalies)
                if filtered_an:
                    anomaly_entries = []
                    for anomaly in filtered_an:
                        if isinstance(anomaly, dict):
                            entry = {
                                'date': pd.to_datetime(anomaly.get('date')).isoformat(),
                                'magnitude': float(anomaly.get('magnitude', 0.0)),
                                'type': anomaly.get('type', 'point_outlier'),
                                'duration': anomaly.get('duration'),
                            }
                            if 'baseline' in anomaly:
                                entry['baseline'] = anomaly.get('baseline')
                        else:
                            entry = {
                                'date': pd.to_datetime(anomaly[0]).isoformat(),
                                'magnitude': float(anomaly[1]),
                                'type': anomaly[2] if len(anomaly) > 2 else 'point_outlier',
                                'duration': int(anomaly[3]) if len(anomaly) > 3 else None,
                            }
                        anomaly_entries.append(entry)
                    series_features['anomalies'] = anomaly_entries

            # Holiday dates
            holiday_dates = self.holiday_dates.get(col, [])
            if holiday_dates:
                if date_filter is not None:
                    if isinstance(date_filter, set):
                        filtered_hd = [
                            d for d in holiday_dates if pd.to_datetime(d) in date_filter
                        ]
                    else:
                        start, stop = date_filter
                        filtered_hd = []
                        for d in holiday_dates:
                            dt = pd.to_datetime(d)
                            if start is not None and dt < start:
                                continue
                            if stop is not None and dt > stop:
                                continue
                            filtered_hd.append(d)
                else:
                    filtered_hd = holiday_dates

                if filtered_hd:
                    series_features['holiday_dates'] = [
                        pd.to_datetime(d).isoformat() for d in filtered_hd
                    ]

            # Holiday impacts (dict of date: impact)
            holiday_impacts = self.holiday_impacts.get(col, {})
            if holiday_impacts:
                filtered_hi = _filter_dict_by_date(holiday_impacts)
                if filtered_hi:
                    series_features['holiday_impacts'] = {
                        pd.to_datetime(k).isoformat(): float(v)
                        for k, v in filtered_hi.items()
                    }

            # Seasonality changepoints.
            seasonality_cps = getattr(self, '_seasonality_changepoints', {}).get(col, [])
            if seasonality_cps:
                filtered_scp = _filter_by_date(seasonality_cps)
                if filtered_scp:
                    cp_entries = []
                    for cp in filtered_scp:
                        if isinstance(cp, dict):
                            cp_entries.append(
                                {
                                    'date': pd.to_datetime(cp.get('date')).isoformat(),
                                    'description': cp.get('description', 'seasonality_change'),
                                }
                            )
                        else:
                            cp_entries.append(
                                {
                                    'date': pd.to_datetime(cp[0]).isoformat(),
                                    'description': cp[1] if len(cp) > 1 else 'seasonality_change',
                                }
                            )
                    series_features['seasonality_changepoints'] = cp_entries

            # Seasonality strength
            if col in self.seasonality_strength:
                series_features['seasonality_strength'] = float(
                    self.seasonality_strength[col]
                )

            # Add metadata if requested
            if include_metadata:
                metadata = {}
                if col in self.noise_to_signal_ratios:
                    metadata['noise_to_signal_ratio'] = float(
                        self.noise_to_signal_ratios[col]
                    )
                if col in self.series_noise_levels:
                    metadata['noise_level'] = float(self.series_noise_levels[col])
                if col in self.series_scales:
                    metadata['scale'] = float(self.series_scales[col])
                metadata['series_type'] = self.series_types.get(col, 'detected')
                metadata['season_type'] = self.series_season_types.get(col, 'none')

                # Noise changepoints
                noise_cp = self.noise_changepoints.get(col, [])
                if noise_cp:
                    filtered_ncp = _filter_by_date(noise_cp)
                    if filtered_ncp:
                        metadata['noise_changepoints'] = [
                            pd.to_datetime(ncp).isoformat() for ncp in filtered_ncp
                        ]

                if metadata:
                    series_features['metadata'] = metadata

            # Add components if requested
            if include_components and col in self.components:
                comp_dict = self.components[col]

                # Determine date range for components
                if dates is None:
                    comp_index = self.date_index
                elif isinstance(dates, slice):
                    start_dt = (
                        pd.to_datetime(dates.start)
                        if dates.start
                        else self.date_index[0]
                    )
                    stop_dt = (
                        pd.to_datetime(dates.stop)
                        if dates.stop
                        else self.date_index[-1]
                    )
                    comp_index = self.date_index[
                        (self.date_index >= start_dt) & (self.date_index <= stop_dt)
                    ]
                elif isinstance(dates, (list, pd.Index)):
                    comp_index = pd.DatetimeIndex(
                        [d for d in pd.to_datetime(dates) if d in self.date_index]
                    )
                else:
                    single_dt = pd.to_datetime(dates)
                    comp_index = (
                        pd.DatetimeIndex([single_dt])
                        if single_dt in self.date_index
                        else pd.DatetimeIndex([])
                    )

                if len(comp_index) > 0:
                    components_data = {}
                    for comp_name, comp_values in comp_dict.items():
                        comp_array = np.asarray(comp_values)
                        # Map index positions
                        comp_data = {}
                        for dt in comp_index:
                            if dt in self.date_index:
                                idx = self.date_index.get_loc(dt)
                                if idx < len(comp_array):
                                    val = comp_array[idx]
                                    if not np.isnan(val):
                                        comp_data[dt.isoformat()] = float(val)
                        if comp_data:
                            components_data[comp_name] = comp_data

                    if components_data:
                        series_features['components'] = components_data

            result['series'][col] = series_features

        # Add shared events if in univariate mode
        if self.detection_mode == 'univariate':
            shared = {}
            if self.shared_events.get('anomalies'):
                filtered_shared_an = _filter_by_date(self.shared_events['anomalies'])
                if filtered_shared_an:
                    shared['anomalies'] = [
                        {
                            'date': pd.to_datetime(an[0]).isoformat(),
                            'magnitude': float(an[1]),
                            'type': an[2],
                        }
                        for an in filtered_shared_an
                    ]

            if self.shared_events.get('level_shifts'):
                filtered_shared_ls = _filter_by_date(self.shared_events['level_shifts'])
                if filtered_shared_ls:
                    shared['level_shifts'] = [
                        {
                            'date': pd.to_datetime(ls[0]).isoformat(),
                            'magnitude': float(ls[1]),
                            'type': ls[2],
                        }
                        for ls in filtered_shared_ls
                    ]

            if shared:
                result['shared_events'] = shared

        if return_json:
            import json

            return json.dumps(result, indent=2)

        return result

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
            'seasonality_changepoints': getattr(
                self, '_seasonality_changepoints', {}
            ).get(series_name, []),
            'noise_changepoints': self.noise_changepoints.get(series_name, []),
            'series_scale': 1.0,
            'noise_to_signal_ratio': self.noise_to_signal_ratios.get(series_name, None),
            'series_type': self.series_types.get(series_name, 'detected'),
        }
        inferred_type = self.series_types.get(series_name, 'detected')
        season_type = self.series_season_types.get(series_name, 'none')
        fig = plot_feature_panels(
            series_name=series_name,
            date_index=self.date_index,
            series_data=self.df_original[series_name],
            components=components,
            labels=labels,
            series_type_description=f"Detected Features ({inferred_type}, {season_type})",
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
            method=method, holiday_countries_used=False
        )

        # Final seasonality params
        seasonality_params = DatepartRegressionTransformer.get_new_params(
            method=method, holiday_countries_used=False
        )
        if random.random() < 0.15:
            seasonality_params['datepart_method'] = 'adaptive_fourier'

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
        if random.random() < 0.2:
            anomaly_params['two_pass'] = True
            anomaly_params['liberal_alpha_multiplier'] = random.choice(
                [5.0, 10.0, 20.0]
            )
        if random.random() < 0.3:
            anomaly_params['holiday_proximity_days'] = random.choice([1, 2, 3, 5])

        # Changepoint params
        changepoint_params = ChangepointDetector.get_new_params(method=method)

        # Level shift params
        level_shift_params = LevelShiftMagic.get_new_params(method=method)
        level_shift_params['output'] = 'multivariate'  # Ensure correct output mode

        # General transformer params (for pre-trend processing)
        general_transformer_params = GeneralTransformer.get_new_params(
            method="filters", allow_none=True, transformer_max_depth=2
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
            'global_holiday_anomaly_suppression': random.choices(
                [True, False], [0.8, 0.2]
            )[0],
        }

    def _apply_detector_params(self, params):
        """Apply detector hyperparameters to this instance."""
        if not isinstance(params, dict):
            raise ValueError("params must be a dict of detector parameters.")

        if params.get('rough_seasonality_params') is not None:
            self.rough_seasonality_params = copy.deepcopy(
                params['rough_seasonality_params']
            )
        if params.get('seasonality_params') is not None:
            self.seasonality_params = copy.deepcopy(params['seasonality_params'])

        if params.get('holiday_params') is not None:
            self.holiday_params = self._sanitize_holiday_params(
                copy.deepcopy(params['holiday_params'])
            )

        if params.get('anomaly_params') is not None:
            self.anomaly_params = copy.deepcopy(params['anomaly_params'])
            self.anomaly_params['output'] = self.detection_mode

        if params.get('changepoint_params') is not None:
            self.changepoint_params = copy.deepcopy(params['changepoint_params'])
            if (
                'aggregate_method' not in self.changepoint_params
                or self.changepoint_params['aggregate_method'] == 'auto'
            ):
                self.changepoint_params['aggregate_method'] = (
                    'individual' if self.detection_mode == 'multivariate' else 'mean'
                )

        if params.get('level_shift_params') is not None:
            self.level_shift_params = copy.deepcopy(params['level_shift_params'])
            self.level_shift_params['output'] = self.detection_mode

        if params.get('general_transformer_params') is not None:
            self.general_transformer_params = copy.deepcopy(
                params['general_transformer_params']
            )

        if 'standardize' in params:
            self.standardize = bool(params['standardize'])
        if 'smoothing_window' in params:
            self.smoothing_window = params['smoothing_window']
        if 'global_holiday_anomaly_suppression' in params:
            self.global_holiday_anomaly_suppression = bool(
                params['global_holiday_anomaly_suppression']
            )

    def tune_with_synthetic(
        self,
        real_df,
        n_synthetic_series=16,
        n_tune_iterations=25,
        n_detector_iterations=30,
        tune_seed=42,
        loss_params=None,
        loss_weights=None,
        synthetic_starting_params=None,
        starting_params=None,
        verbose=True,
    ):
        """
        Tune synthetic data to a real dataset, optimize detector params, and fit self.

        After completion, this instance is fitted on ``real_df`` with the optimized
        detector parameters and stores optimization artifacts on the instance.
        """
        if not isinstance(real_df, pd.DataFrame):
            raise ValueError("real_df must be a pandas DataFrame.")
        if not isinstance(real_df.index, pd.DatetimeIndex):
            raise ValueError("real_df must use a DatetimeIndex.")
        if real_df.empty:
            raise ValueError("real_df must not be empty.")

        if verbose:
            print("=" * 80)
            print("TUNE WITH SYNTHETIC: real data -> synthetic labels -> detector tuning")
            print("=" * 80)

        if verbose:
            print("\n[Step 1/4] Tuning synthetic generator to match real data...")
        generator = SyntheticDailyGenerator(
            start_date=real_df.index[0],
            n_days=len(real_df),
            n_series=n_synthetic_series,
            random_seed=tune_seed,
        )
        tuning_results = generator.tune_to_data(
            real_df,
            n_iterations=n_tune_iterations,
            verbose=verbose,
            starting_params=synthetic_starting_params,
        )

        if verbose:
            print("\n[Step 2/4] Generating labeled synthetic data with tuned params...")
        best_synth_params = tuning_results['best_params']
        scale_multiplier = tuning_results.get('scale_multiplier')
        target_weekly = tuning_results.get('target_stats', {}).get('weekly_profile')
        target_yearly = tuning_results.get('target_stats', {}).get('yearly_fourier')
        tuned_generator = SyntheticDailyGenerator(
            start_date=real_df.index[0],
            n_days=len(real_df),
            n_series=n_synthetic_series,
            random_seed=tune_seed,
            series_type_override='standard',
            weekly_profile_target=target_weekly,
            yearly_fourier_target=target_yearly,
            **best_synth_params,
        )

        if verbose:
            total_labels = sum(
                len(tuned_generator.get_all_labels(col).get('trend_changepoints', []))
                + len(tuned_generator.get_all_labels(col).get('level_shifts', []))
                + len(tuned_generator.get_all_labels(col).get('anomalies', []))
                for col in tuned_generator.data.columns
            )
            print(
                f"  Generated {n_synthetic_series} series with {total_labels} labeled events"
            )

        if verbose:
            print("\n[Step 3/4] Optimizing feature detector on labeled synthetic data...")
        loss_kwargs = copy.deepcopy(loss_params) if loss_params else {}
        if loss_weights is not None:
            loss_kwargs['weights'] = copy.deepcopy(loss_weights)
        loss_calc = FeatureDetectionLoss(**loss_kwargs)
        optimizer = FeatureDetectionOptimizer(
            synthetic_generator=tuned_generator,
            loss_calculator=loss_calc,
            n_iterations=n_detector_iterations,
            random_seed=tune_seed,
            starting_params=starting_params,
        )
        best_detector_params = optimizer.optimize()
        if best_detector_params is None:
            raise RuntimeError("Feature detector optimization did not produce parameters.")

        if verbose:
            print("\n[Step 4/4] Applying optimized parameters and fitting on real data...")
        self._apply_detector_params(best_detector_params)
        self.fit(real_df)

        optimization_summary = optimizer.get_optimization_summary()
        self.optimized_detector_params = copy.deepcopy(best_detector_params)
        self.synthetic_tuning_results = {
            'best_detector_params': copy.deepcopy(best_detector_params),
            'baseline_loss': optimizer.baseline_loss,
            'tuned_generator': tuned_generator,
            'scale_multiplier': scale_multiplier,
            'tuning_results': tuning_results,
            'optimization_summary': optimization_summary,
        }
        self.synthetic_scale_multiplier = scale_multiplier
        self.tuned_synthetic_generator = tuned_generator
        self.detector_optimization_summary = optimization_summary
        self.detector_optimizer = optimizer
        return self


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
        'level_shift_loss': 1.3,
        'anomaly_loss': 1.3,  # Increased from 1.1 - prioritize anomaly detection
        'holiday_event_loss': 1.2,  # Increased from 0.8 - penalize false holiday detections more
        'holiday_impact_loss': 0.9,  # Increased from 0.6 - ensure holiday impacts are strong enough
        'holiday_splash_loss': 0.5,
        'holiday_recall_loss': 0.9,  # Separate recall metric to penalize zero-detection
        'seasonality_strength_loss': 0.8,
        'seasonality_pattern_loss': 1.0,
        'seasonality_changepoint_loss': 0.6,
        'noise_level_loss': 0.5,
        'noise_regime_loss': 0.4,
        'metadata_loss': 0.2,
        'regressor_loss': 0.3,
    }
    INVALID_LOSS_PENALTY = 1e6

    def __init__(
        self,
        changepoint_tolerance_days=7,
        level_shift_tolerance_days=7,
        anomaly_tolerance_days=1,
        holiday_tolerance_days=1,
        seasonality_window=14,
        weights=None,
        holiday_over_anomaly_bonus=0.4,
        trend_component_penalty='component',
        trend_complexity_window=7,
        trend_complexity_weight=0.0,
        focus_component_weights=False,
        validation_strictness=1.0,
        invalid_loss_mode='penalty',
        invalid_loss_penalty=INVALID_LOSS_PENALTY,
    ):
        self.changepoint_tolerance_days = changepoint_tolerance_days
        self.level_shift_tolerance_days = level_shift_tolerance_days
        self.anomaly_tolerance_days = anomaly_tolerance_days
        self.holiday_tolerance_days = holiday_tolerance_days
        self.holiday_over_anomaly_bonus = holiday_over_anomaly_bonus
        self.seasonality_window = max(3, int(seasonality_window))
        self.validation_strictness = float(validation_strictness)

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
        invalid_loss_mode = str(invalid_loss_mode).lower().strip()
        if invalid_loss_mode not in {'penalty', 'raise'}:
            raise ValueError(
                "invalid_loss_mode must be either 'penalty' or 'raise'."
            )
        self.invalid_loss_mode = invalid_loss_mode
        self.invalid_loss_penalty = max(1.0, float(invalid_loss_penalty))
        self._invalid_loss_warnings = set()
        self.last_effective_weights = None
        self.last_disabled_components = []

        self.weights = copy.deepcopy(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        if self.focus_component_weights:
            self._apply_component_focus_reweighting()

        self._change_tolerance = pd.Timedelta(self.changepoint_tolerance_days, unit='D')
        self._level_shift_tolerance = pd.Timedelta(
            self.level_shift_tolerance_days, unit='D'
        )
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
            'holiday_recall_loss': 1.15,
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
            detected_features.get('components')
            if isinstance(detected_features, dict)
            else None,
            series_name,
        )
        true_components = self._resolve_components(true_components, series_name)

        series_names = self._resolve_series_names(
            detected_features, true_labels, series_name
        )
        if not series_names:
            return {'total_loss': 0.0}

        true_series_by_name = {
            name: self._extract_true_series(true_labels, name) for name in series_names
        }
        effective_weights, disabled_components = self._build_effective_weights(
            true_series_by_name, true_components
        )
        self.last_effective_weights = copy.deepcopy(effective_weights)
        self.last_disabled_components = list(disabled_components)

        aggregate_loss = {key: 0.0 for key in self.weights}
        series_breakdown = {}

        for name in series_names:
            series_loss = self._calculate_series_loss(
                name,
                detected_features,
                true_series_by_name[name],
                detected_components.get(name, {}),
                true_components.get(name, {}),
                date_index,
                effective_weights=effective_weights,
            )
            series_breakdown[name] = series_loss
            for key in self.weights:
                aggregate_loss[key] += series_loss.get(key, 0.0)

        n_series = len(series_names)
        for key in aggregate_loss:
            aggregate_loss[key] /= n_series

        total_loss = 0.0
        for key, value in aggregate_loss.items():
            total_loss += effective_weights.get(key, 0.0) * value
        total_loss = self._guard_loss_value(total_loss, 'total_loss')

        aggregate_loss['total_loss'] = total_loss
        aggregate_loss['series_breakdown'] = series_breakdown
        aggregate_loss['effective_weights'] = copy.deepcopy(effective_weights)
        aggregate_loss['disabled_components'] = list(disabled_components)
        return aggregate_loss

    def _calculate_series_loss(
        self,
        series_name,
        detected_features,
        true_series,
        detected_components,
        true_components,
        date_index,
        effective_weights=None,
    ):
        detected = self._extract_detected_series(detected_features, series_name)
        true = copy.deepcopy(true_series) if isinstance(true_series, dict) else {}

        if effective_weights is None:
            effective_weights = self.weights

        trend_loss = self._evaluate_component_loss(
            key='trend_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._trend_loss,
            detected_cp=detected.get('trend_changepoints', []),
            true_cp=true.get('trend_changepoints', []),
            detected_components=detected_components,
            true_components=true_components,
        )
        level_shift_loss = self._evaluate_component_loss(
            key='level_shift_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._level_shift_loss,
            detected_ls=detected.get('level_shifts', []),
            true_ls=true.get('level_shifts', []),
            detected_cp=detected.get('trend_changepoints', []),
        )
        anomaly_loss = self._evaluate_component_loss(
            key='anomaly_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._anomaly_loss,
            detected_anom=detected.get('anomalies', []),
            true_anom=true.get('anomalies', []),
        )

        holiday_event_loss = self._evaluate_component_loss(
            key='holiday_event_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._holiday_event_loss,
            detected_holidays=detected.get('holiday_dates', []),
            true_holidays=true.get('holiday_dates', []),
            detected_anomalies=detected.get('anomalies', []),
        )
        holiday_impact_loss = self._evaluate_component_loss(
            key='holiday_impact_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._holiday_impact_loss,
            detected_impacts=detected.get('holiday_impacts', {}),
            true_impacts=true.get('holiday_impacts', {}),
        )
        holiday_splash_loss = self._evaluate_component_loss(
            key='holiday_splash_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._holiday_splash_loss,
            detected_impacts=detected.get('holiday_splash_impacts', {}),
            detected_anomalies=detected.get('anomalies', []),
            true_splash=true.get('holiday_splash_impacts', {}),
        )
        holiday_recall_loss = self._evaluate_component_loss(
            key='holiday_recall_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._holiday_recall_loss,
            detected_holidays=detected.get('holiday_dates', []),
            true_holidays=true.get('holiday_dates', []),
        )

        seasonality_strength_loss = self._evaluate_component_loss(
            key='seasonality_strength_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._seasonality_strength_loss,
            detected_strengths=detected.get('series_seasonality_strengths'),
            true_strengths=true.get('series_seasonality_strengths'),
        )
        seasonality_pattern_loss = self._evaluate_component_loss(
            key='seasonality_pattern_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._seasonality_pattern_loss,
            detected_components=detected_components,
            true_components=true_components,
        )
        seasonality_changepoint_loss = self._evaluate_component_loss(
            key='seasonality_changepoint_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._seasonality_changepoint_loss,
            detected_cp=detected.get('seasonality_changepoints', []),
            true_cp=true.get('seasonality_changepoints', []),
            detected_components=detected_components,
            true_components=true_components,
            date_index=date_index,
        )

        noise_level_loss = self._evaluate_component_loss(
            key='noise_level_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._noise_level_loss,
            detected_level=detected.get('series_noise_level'),
            true_level=true.get('series_noise_level'),
            detected_ratio=detected.get('noise_to_signal_ratio'),
            true_ratio=true.get('noise_to_signal_ratio'),
        )
        noise_regime_loss = self._evaluate_component_loss(
            key='noise_regime_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._noise_regime_loss,
            detected_cp=detected.get('noise_changepoints', []),
            true_cp=true.get('noise_changepoints', []),
        )

        metadata_loss = self._evaluate_component_loss(
            key='metadata_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._metadata_loss,
            detected_scale=detected.get('series_scale'),
            true_scale=true.get('series_scale'),
            detected_type=detected.get('series_type'),
            true_type=true.get('series_type'),
        )

        regressor_loss = self._evaluate_component_loss(
            key='regressor_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._regressor_loss,
            detected_regressors=detected.get('regressor_impacts', {}),
            true_regressors=true.get('regressor_impacts', {}),
        )

        return {
            'trend_loss': trend_loss,
            'level_shift_loss': level_shift_loss,
            'anomaly_loss': anomaly_loss,
            'holiday_event_loss': holiday_event_loss,
            'holiday_impact_loss': holiday_impact_loss,
            'holiday_splash_loss': holiday_splash_loss,
            'holiday_recall_loss': holiday_recall_loss,
            'seasonality_strength_loss': seasonality_strength_loss,
            'seasonality_pattern_loss': seasonality_pattern_loss,
            'seasonality_changepoint_loss': seasonality_changepoint_loss,
            'noise_level_loss': noise_level_loss,
            'noise_regime_loss': noise_regime_loss,
            'metadata_loss': metadata_loss,
            'regressor_loss': regressor_loss,
        }

    def _evaluate_component_loss(
        self, key, series_name, effective_weights, fn, *args, **kwargs
    ):
        if effective_weights.get(key, 0.0) == 0.0:
            return 0.0
        value = fn(*args, **kwargs)
        return self._guard_loss_value(value, key, series_name=series_name)

    def _chamfer_penalty(self, detected_dates, true_dates, cap=30.0, recall_weight=0.6):
        """
        Compute asymmetric Chamfer distance between two sets of dates.

        Uses Gaussian-weighted proximity scoring instead of linear min/cap.
        Recall (true->detected) is weighted higher than precision (detected->true)
        by default, since missing a true event is worse than a false positive.

        Parameters
        ----------
        detected_dates : list
            Detected event dates.
        true_dates : list
            True event dates.
        cap : float
            Distance cap in days. Points beyond this contribute maximum penalty.
        recall_weight : float
            Weight for recall direction (true->detected). Precision gets 1 - recall_weight.

        Returns
        -------
        float
            Value between 0 (perfect) and 1 (worst), blending recall and precision.
        """
        if not true_dates and not detected_dates:
            return 0.0
        if not true_dates:
            # Only false positives: penalty scales with count but caps at 1
            return min(0.3 * len(detected_dates), 1.0)
        if not detected_dates:
            return 1.0

        t_dates = [pd.Timestamp(d) for d in true_dates]
        d_dates = [pd.Timestamp(d) for d in detected_dates]

        epoch = min(min(t_dates), min(d_dates))
        t_vals = np.array([(d - epoch).total_seconds() / 86400.0 for d in t_dates])
        d_vals = np.array([(d - epoch).total_seconds() / 86400.0 for d in d_dates])

        dists = np.abs(t_vals[:, None] - d_vals[None, :])

        # Gaussian proximity: exp(-0.5 * (d/sigma)^2), sigma = cap/3
        sigma = max(cap / 3.0, 1.0)

        # Recall: for each true event, how close is the nearest detected?
        t2d_dists = np.min(dists, axis=1)
        t2d_scores = np.exp(-0.5 * (t2d_dists / sigma) ** 2)
        recall_score = 1.0 - np.mean(t2d_scores)

        # Precision: for each detected event, how close is the nearest true?
        d2t_dists = np.min(dists, axis=0)
        d2t_scores = np.exp(-0.5 * (d2t_dists / sigma) ** 2)
        precision_score = 1.0 - np.mean(d2t_scores)

        # Count mismatch penalty (soft)
        count_ratio = abs(len(t_dates) - len(d_dates)) / (len(t_dates) + len(d_dates))
        count_penalty = 0.15 * count_ratio

        precision_weight = 1.0 - recall_weight
        combined = recall_weight * recall_score + precision_weight * precision_score + count_penalty
        return min(combined, 1.0)

    def _build_effective_weights(self, true_series_by_name, true_components):
        effective = copy.deepcopy(self.weights)
        disabled = []
        for key in self.weights:
            is_active = any(
                self._has_supervision_for_component(
                    key,
                    true_series_by_name.get(name, {}),
                    true_components.get(name, {}),
                )
                for name in true_series_by_name.keys()
            )
            if not is_active:
                effective[key] = 0.0
                disabled.append(key)
        return effective, sorted(disabled)

    def _has_supervision_for_component(self, key, true_series, true_component_map):
        true_series = true_series if isinstance(true_series, dict) else {}
        true_component_map = (
            true_component_map if isinstance(true_component_map, dict) else {}
        )
        if key == 'trend_loss':
            return not self._is_empty_label(true_series.get('trend_changepoints'))
        if key == 'level_shift_loss':
            return not self._is_empty_label(true_series.get('level_shifts'))
        if key == 'anomaly_loss':
            return not self._is_empty_label(true_series.get('anomalies'))
        if key in {'holiday_event_loss', 'holiday_recall_loss'}:
            return not self._is_empty_label(true_series.get('holiday_dates'))
        if key == 'holiday_impact_loss':
            return not self._is_empty_label(true_series.get('holiday_impacts'))
        if key == 'holiday_splash_loss':
            return not self._is_empty_label(true_series.get('holiday_splash_impacts'))
        if key == 'seasonality_strength_loss':
            return not self._is_empty_label(
                true_series.get('series_seasonality_strengths')
            )
        if key == 'seasonality_pattern_loss':
            return not self._is_empty_label(true_component_map.get('seasonality'))
        if key == 'seasonality_changepoint_loss':
            return not self._is_empty_label(true_series.get('seasonality_changepoints'))
        if key == 'noise_level_loss':
            has_level = not self._is_empty_label(true_series.get('series_noise_level'))
            has_ratio = not self._is_empty_label(true_series.get('noise_to_signal_ratio'))
            return bool(has_level or has_ratio)
        if key == 'noise_regime_loss':
            return not self._is_empty_label(true_series.get('noise_changepoints'))
        if key == 'metadata_loss':
            has_scale = not self._is_empty_label(true_series.get('series_scale'))
            has_type = not self._is_empty_label(true_series.get('series_type'))
            return bool(has_scale or has_type)
        if key == 'regressor_loss':
            by_date, coefficients = self._normalize_regressor_schema(
                true_series.get('regressor_impacts', {})
            )
            return bool(by_date or coefficients)
        return True

    @classmethod
    def _is_empty_label(cls, value):
        if value is None:
            return True
        if isinstance(value, (np.floating, float)):
            return not np.isfinite(float(value))
        if isinstance(value, (np.integer, int, bool)):
            return False
        if isinstance(value, str):
            return value.strip() == ''
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return True
            if np.issubdtype(value.dtype, np.number):
                return not np.isfinite(value).any()
            return False
        if isinstance(value, dict):
            if not value:
                return True
            return all(cls._is_empty_label(v) for v in value.values())
        if isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                return True
            return all(cls._is_empty_label(v) for v in value)
        if isinstance(value, pd.Timestamp):
            return False
        return False

    def _guard_loss_value(self, value, component_name, series_name=None):
        invalid = False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = self.invalid_loss_penalty
            invalid = True
        if not np.isfinite(numeric) or numeric < 0:
            invalid = True
        if not invalid:
            return numeric

        context = f"{component_name}" if series_name is None else f"{component_name}:{series_name}"
        message = (
            f"Invalid loss value encountered for {context}. "
            f"Using mode '{self.invalid_loss_mode}'."
        )
        if self.invalid_loss_mode == 'raise':
            raise ValueError(message)
        if context not in self._invalid_loss_warnings:
            warnings.warn(message, RuntimeWarning)
            self._invalid_loss_warnings.add(context)
        return self.invalid_loss_penalty

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
            'trend_changepoints': _fetch(
                'trend_changepoints', 'trend_changepoints', []
            ),
            'level_shifts': _fetch('level_shifts', 'level_shifts', []),
            'anomalies': _fetch('anomalies', 'anomalies', []),
            'holiday_dates': _fetch('holiday_dates', 'holiday_dates', []),
            'holiday_impacts': _fetch('holiday_impacts', 'holiday_impacts', {}),
            'holiday_splash_impacts': _fetch(
                'holiday_splash_impacts', 'holiday_splash_impacts', {}
            ),
            'seasonality_changepoints': _fetch(
                'seasonality_changepoints', 'seasonality_changepoints', []
            ),
            'noise_changepoints': _fetch(
                'noise_changepoints', 'noise_changepoints', []
            ),
            'series_seasonality_strengths': _fetch(
                'series_seasonality_strengths', 'series_seasonality_strengths', {}
            ),
            'seasonality_strength': _fetch(
                'seasonality_strength', 'seasonality_strength', 0.0
            ),
            'noise_to_signal_ratio': _fetch(
                'noise_to_signal_ratio', 'noise_to_signal_ratios', 0.0
            ),
            'series_noise_level': _fetch(
                'series_noise_level', 'series_noise_levels', 0.0
            ),
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
            'trend_changepoints': _fetch(
                'trend_changepoints', 'trend_changepoints', []
            ),
            'level_shifts': _fetch('level_shifts', 'level_shifts', []),
            'anomalies': _fetch('anomalies', 'anomalies', []),
            'holiday_dates': _fetch('holiday_dates', 'holiday_dates', []),
            'holiday_impacts': _fetch('holiday_impacts', 'holiday_impacts', {}),
            'holiday_splash_impacts': _fetch(
                'holiday_splash_impacts', 'holiday_splash_impacts', {}
            ),
            'seasonality_changepoints': _fetch(
                'seasonality_changepoints', 'seasonality_changepoints', []
            ),
            'noise_changepoints': _fetch(
                'noise_changepoints', 'noise_changepoints', []
            ),
            'noise_to_signal_ratio': _fetch(
                'noise_to_signal_ratio', 'noise_to_signal_ratios', 0.0
            ),
            'series_noise_level': _fetch(
                'series_noise_level', 'series_noise_levels', 0.0
            ),
            'series_seasonality_strengths': _fetch(
                'series_seasonality_strengths', 'series_seasonality_strengths', {}
            ),
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
        if not true_cp and not detected_cp:
            return 0.0
        if not true_cp:
            return 0.25 * len(detected_cp)

        detected_entries = [self._parse_trend_event(event) for event in detected_cp]
        true_entries = [self._parse_trend_event(event) for event in true_cp]
        n_true = len(true_entries)
        
        # Add Chamfer penalty to guide optimizer when points are far apart
        chamfer_loss = self._chamfer_penalty([x[0] for x in detected_entries], [x[0] for x in true_entries], cap=self.changepoint_tolerance_days * 5)
        
        n_detected = len(detected_entries)
        unmatched_detected = set(range(n_detected))

        positive_true_magnitudes = [
            entry[3] for entry in true_entries if np.isfinite(entry[3]) and entry[3] > 0
        ]
        default_magnitude_scale = (
            float(np.median(positive_true_magnitudes))
            if positive_true_magnitudes
            else 0.0
        )

        def _subtlety_weight(prior_slope, post_slope, magnitude):
            sign_change = (prior_slope * post_slope) < 0
            if sign_change:
                return 1.0
            if magnitude > 0:
                scale = default_magnitude_scale if default_magnitude_scale > 0 else magnitude
                relative = magnitude / (scale + 1e-9)
                return 0.2 + 0.7 * np.tanh(relative)
            return 0.2

        sigma_days = max(self.changepoint_tolerance_days, 1) / 1.5
        matched_true = 0
        loss = 0.0

        for true_date, true_prior, true_post, true_mag in true_entries:
            importance = _subtlety_weight(true_prior, true_post, true_mag)
            best_idx = None
            best_dist = None
            for idx in unmatched_detected:
                det_date = detected_entries[idx][0]
                dist = abs((det_date - true_date).days)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if (
                best_idx is not None
                and best_dist is not None
                and best_dist <= self.changepoint_tolerance_days
            ):
                _, det_prior, det_post, _ = detected_entries[best_idx]
                unmatched_detected.discard(best_idx)
                matched_true += 1

                distance_score = np.exp(-0.5 * (best_dist / (sigma_days + 1e-9)) ** 2)
                distance_penalty = (1.0 - distance_score) * 0.5

                slope_change_true = true_post - true_prior
                slope_change_detected = det_post - det_prior
                slope_denom = max(abs(slope_change_true), 0.05, 1e-3)
                slope_error = abs(slope_change_detected - slope_change_true) / slope_denom
                slope_penalty = min(slope_error, 2.0) * 0.3

                sign_penalty = (
                    0.2 if (slope_change_true * slope_change_detected) < 0 else 0.0
                )
                loss += (distance_penalty + slope_penalty + sign_penalty) * importance
            else:
                # Smoother miss penalty based on distance to nearest match
                dist_penalty = min(best_dist, 60.0) / 60.0 if best_dist is not None else 1.0
                loss += (1.2 + 0.5 * dist_penalty) * importance

        false_positives = len(unmatched_detected)
        loss += 0.2 * false_positives

        count_diff = abs(n_detected - n_true)
        loss += min(count_diff, n_true + 1) * 0.4

        recall = matched_true / (n_true + 1e-9)
        precision = (
            (n_detected - false_positives) / (n_detected + 1e-9)
            if n_detected
            else 1.0
        )
        f_beta = (1.0 + 1.5**2) * (precision * recall) / (
            1.5**2 * precision + recall + 1e-9
        )
        loss += (1.0 - f_beta) * 2.0

        # Apply trend component or complexity penalty based on mode
        trend_detected_series = detected_components.get('trend')

        if self.trend_component_penalty == 'component':
            if (
                trend_detected_series is not None
                and 'trend' in true_components
                and true_components.get('trend') is not None
            ):
                loss += self._component_rmse_penalty(
                    trend_detected_series,
                    true_components['trend'],
                )
        elif self.trend_component_penalty == 'complexity':
            if trend_detected_series is not None and self.trend_complexity_weight > 0:
                complexity_penalty = self._trend_complexity_penalty(
                    trend_detected_series
                )
                loss += self.trend_complexity_weight * complexity_penalty

        return loss + 0.6 * chamfer_loss

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

        residual_sq = residual_values**2
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
        if not true_ls and not detected_ls:
            return 0.0
        if not true_ls:
            return 0.2 * len(detected_ls)
        detected_entries = [
            self._parse_level_shift_event(event) for event in detected_ls
        ]
        true_entries = [self._parse_level_shift_event(event) for event in true_ls]
        
        chamfer_loss = self._chamfer_penalty([x[0] for x in detected_entries], [x[0] for x in true_entries], cap=self.level_shift_tolerance_days * 5)
        
        changepoint_dates = [self._parse_trend_event(event)[0] for event in detected_cp]
        n_true = len(true_entries)
        n_detected = len(detected_entries)
        unmatched_detected = set(range(n_detected))
        matched_true = 0
        loss = 0.0

        magnitude_scale = max([abs(mag) for _, mag in true_entries] or [1.0])

        for true_date, true_mag in true_entries:
            relative_mag = abs(true_mag) / (magnitude_scale + 1e-9)
            importance = 0.3 + 0.7 * min(relative_mag, 1.0)

            best_idx = None
            best_dist = None
            for idx in unmatched_detected:
                det_date, _ = detected_entries[idx]
                dist = abs((det_date - true_date).days)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if (
                best_idx is not None
                and best_dist is not None
                and best_dist <= self.level_shift_tolerance_days
            ):
                _, det_mag = detected_entries[best_idx]
                distance_penalty = best_dist / (self.level_shift_tolerance_days + 1e-9)
                magnitude_penalty = abs(det_mag - true_mag) / (abs(true_mag) + 1e-6)
                loss += (
                    0.5 * distance_penalty + 0.5 * min(magnitude_penalty, 2.0)
                ) * importance
                unmatched_detected.discard(best_idx)
                matched_true += 1
            else:
                prox_cp = any(
                    abs((cp_date - true_date).days) <= self.changepoint_tolerance_days
                    for cp_date in changepoint_dates
                )
                if prox_cp:
                    loss += 0.5 * importance
                else:
                    loss += 1.2 * importance + (0.3 * min(best_dist, 60)/60 if best_dist else 0)

        false_positives = len(unmatched_detected)
        loss += 0.15 * false_positives

        count_diff = abs(n_detected - n_true)
        loss += min(count_diff, n_true + 1) * 0.3

        recall = matched_true / (n_true + 1e-9)
        precision = (
            (n_detected - false_positives) / (n_detected + 1e-9)
            if n_detected
            else 1.0
        )
        f_beta = (1.0 + 1.5**2) * (precision * recall) / (
            1.5**2 * precision + recall + 1e-9
        )
        loss += (1.0 - f_beta) * 1.5
        return loss + 0.5 * chamfer_loss

    def _soft_f1_anomaly(self, detected_entries, true_entries, sigma_days=None):
        """
        Compute soft F1 score for anomaly detection using Gaussian proximity weighting.

        Instead of binary match/no-match within a tolerance window, each
        detected-true pair gets a continuous match score based on Gaussian
        proximity: score = exp(-0.5 * (dist/sigma)^2). This provides smooth
        gradients for the optimizer even when detections are slightly outside
        the hard tolerance boundary.

        Parameters
        ----------
        detected_entries : list of tuples
            Parsed anomaly events (date, magnitude, type, duration).
        true_entries : list of tuples
            Parsed true anomaly events.
        sigma_days : float, optional
            Standard deviation for Gaussian weighting (in days).
            Defaults to anomaly_tolerance_days.

        Returns
        -------
        dict
            Contains 'soft_precision', 'soft_recall', 'soft_f1', and
            'match_scores' (per-true-event best match quality).
        """
        if sigma_days is None:
            sigma_days = max(self.anomaly_tolerance_days, 0.5)

        if not true_entries and not detected_entries:
            return {'soft_precision': 1.0, 'soft_recall': 1.0, 'soft_f1': 1.0, 'match_scores': []}
        if not true_entries:
            return {'soft_precision': 0.0, 'soft_recall': 1.0, 'soft_f1': 0.0, 'match_scores': []}
        if not detected_entries:
            return {'soft_precision': 1.0, 'soft_recall': 0.0, 'soft_f1': 0.0, 'match_scores': []}

        t_dates = np.array([
            (e[0] - pd.Timestamp('1970-01-01')).total_seconds() / 86400.0
            for e in true_entries
        ])
        d_dates = np.array([
            (e[0] - pd.Timestamp('1970-01-01')).total_seconds() / 86400.0
            for e in detected_entries
        ])

        # Pairwise distance matrix
        dists = np.abs(t_dates[:, None] - d_dates[None, :])  # (n_true, n_det)
        # Gaussian proximity scores
        proximity = np.exp(-0.5 * (dists / sigma_days) ** 2)

        # Soft recall: for each true event, best match quality
        match_scores = np.max(proximity, axis=1)  # best detected match per true
        soft_recall = float(np.mean(match_scores))

        # Soft precision: for each detected event, best match quality
        precision_scores = np.max(proximity, axis=0)
        soft_precision = float(np.mean(precision_scores))

        # Soft F1 (beta=1.2, slightly recall-favoring)
        beta = 1.2
        beta_sq = beta ** 2
        denom = beta_sq * soft_precision + soft_recall + 1e-9
        soft_f1 = (1.0 + beta_sq) * (soft_precision * soft_recall) / denom

        return {
            'soft_precision': soft_precision,
            'soft_recall': soft_recall,
            'soft_f1': soft_f1,
            'match_scores': match_scores.tolist(),
        }

    def _anomaly_loss(self, detected_anom, true_anom):
        if not true_anom:
            return 0.3 * len(detected_anom)
        detected_entries = [self._parse_anomaly_event(event) for event in detected_anom]
        true_entries = [self._parse_anomaly_event(event) for event in true_anom]

        # Soft F1 provides smooth gradient for optimizer even when detections
        # are slightly outside hard tolerance boundaries
        soft_f1_result = self._soft_f1_anomaly(detected_entries, true_entries)
        soft_f1_loss = 1.0 - soft_f1_result['soft_f1']

        used_detected = set()
        loss = 0.0
        finite_true_mags = [
            abs(mag)
            for _, mag, _, _ in true_entries
            if np.isfinite(mag) and abs(mag) > 0
        ]
        magnitude_scale = max(finite_true_mags or [1.0])
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
            if (
                best_idx is not None
                and best_dist is not None
                and best_dist <= self.anomaly_tolerance_days
            ):
                det_event = detected_entries[best_idx]
                _, det_mag, det_type, det_duration = det_event
                true_mag = true_mag if np.isfinite(true_mag) else 0.0
                det_mag = det_mag if np.isfinite(det_mag) else 0.0
                mag_pen = abs(det_mag - true_mag) / (abs(true_mag) + 1e-6)
                type_pen = 0.0 if det_type == true_type else 0.3
                duration_pen = abs(det_duration - true_duration) / (
                    true_duration + 1e-6
                )
                loss += 0.5 * mag_pen + 0.3 * type_pen + 0.2 * min(duration_pen, 2.0)
                used_detected.add(best_idx)
            else:
                relative_mag = abs(true_mag) / (magnitude_scale + 1e-9)
                if relative_mag > 0.5:
                    miss_penalty = (
                        1.5 if true_type in {'point_outlier', 'spike'} else 1.2
                    )
                elif relative_mag > 0.2:
                    miss_penalty = (
                        0.6 if true_type in {'point_outlier', 'spike'} else 0.4
                    )
                else:
                    miss_penalty = 0.15
                loss += miss_penalty
        false_positives = len(detected_entries) - len(used_detected)
        if false_positives > 0:
            fp_penalty = (0.15 * false_positives) + (0.1 * np.sqrt(false_positives))
            loss += fp_penalty
        # Blend hard-match detail loss with soft F1 for smoother optimization
        return 0.6 * loss + 0.4 * soft_f1_loss * max(len(true_entries), 1)

    def _holiday_event_loss(self, detected_holidays, true_holidays, detected_anomalies):
        if not true_holidays:
            # Significantly increased penalty for false positive holidays
            return 0.25 * len(detected_holidays)
        detected_dates = [pd.Timestamp(dt) for dt in detected_holidays]
        true_dates = [pd.Timestamp(dt) for dt in true_holidays]
        anomaly_dates = [
            self._parse_anomaly_event(event)[0] for event in detected_anomalies
        ]
        loss = 0.0
        for true_date in true_dates:
            matches = [
                det
                for det in detected_dates
                if abs(det - true_date) <= self._holiday_tolerance
            ]
            if matches:
                continue
            anomaly_match = [
                det
                for det in anomaly_dates
                if abs(det - true_date) <= self._anomaly_tolerance
            ]
            if anomaly_match:
                loss += self.holiday_over_anomaly_bonus
            else:
                loss += 1.0
        false_positives = sum(
            1
            for det in detected_dates
            if not any(
                abs(det - true_date) <= self._holiday_tolerance
                for true_date in true_dates
            )
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
        anomaly_dates = [
            self._parse_anomaly_event(event)[0] for event in detected_anomalies
        ]
        loss = 0.0
        for date, magnitude in self._normalize_holiday_dict(true_splash).items():
            found = date in detected or any(
                abs(date - anomaly) <= self._anomaly_tolerance
                for anomaly in anomaly_dates
            )
            if not found:
                loss += 0.4 + 0.3 * min(abs(magnitude), 2.0)
        return loss

    def _holiday_recall_loss(self, detected_holidays, true_holidays):
        """
        Separate recall-focused loss for holiday detection.

        Heavily penalizes configurations that detect zero holidays when the truth
        has many, which encourages the optimizer to explore holiday-friendly params.
        """
        if not true_holidays:
            return 0.0

        n_true = len(true_holidays)
        if not detected_holidays:
            # Zero holidays detected when truth has holidays - scale with count and strictness
            return min(
                (0.5 + 0.5 * self.validation_strictness) + 0.1 * n_true, 
                2.5 * self.validation_strictness + 1.0
            )

        detected_dates = {pd.Timestamp(dt) for dt in detected_holidays}
        true_dates = [pd.Timestamp(dt) for dt in true_holidays]

        matches = sum(
            1
            for td in true_dates
            if any(
                abs(td - dd) <= self._holiday_tolerance for dd in detected_dates
            )
        )

        recall = matches / n_true

        # Progressive penalty: stronger for very low recall, influenced by strictness
        recall_penalty_scale = self.validation_strictness
        if recall < 0.2:
            return 2.5 * (1.0 - recall) * recall_penalty_scale
        elif recall < 0.4:
            return 1.5 * (1.0 - recall) * recall_penalty_scale
        elif recall < 0.6:
            return 1.0 * (1.0 - recall) * recall_penalty_scale
        else:
            return 0.4 * (1.0 - recall) * recall_penalty_scale

    def _seasonality_strength_loss(self, detected_strengths, true_strengths):
        if not true_strengths:
            return 0.0
        detected_strengths = detected_strengths or {}
        loss = 0.0
        n_items = 0
        for key, true_value in true_strengths.items():
            det_value = detected_strengths.get(key)
            if det_value is None and isinstance(key, str) and key.startswith('period_'):
                try:
                    true_period = int(key.split('_')[1])
                except (IndexError, ValueError):
                    true_period = None
                if true_period is not None:
                    best_match_penalty = None
                    for det_key, det_val in detected_strengths.items():
                        if not (isinstance(det_key, str) and det_key.startswith('period_')):
                            continue
                        try:
                            det_period = int(det_key.split('_')[1])
                        except (IndexError, ValueError):
                            continue
                        period_diff = abs(true_period - det_period)
                        if period_diff <= max(1, true_period * 0.05):
                            val_penalty = abs(det_val - true_value) / (
                                abs(true_value) + 1e-6
                            )
                            proximity_penalty = (
                                period_diff / (true_period + 1e-6)
                            ) * 0.1
                            total_penalty = min(val_penalty + proximity_penalty, 2.0)
                            if (
                                best_match_penalty is None
                                or total_penalty < best_match_penalty
                            ):
                                best_match_penalty = total_penalty
                    if best_match_penalty is not None:
                        loss += best_match_penalty
                        n_items += 1
                        continue
            if det_value is None:
                det_value = detected_strengths.get(
                    'combined', detected_strengths.get('seasonality_strength')
                )
            if det_value is None:
                loss += 0.5 + abs(true_value)
            else:
                penalty = abs(det_value - true_value) / (abs(true_value) + 1e-6)
                loss += min(penalty, 2.0)
            n_items += 1
        return loss / max(1, n_items)

    def _seasonality_pattern_loss(self, detected_components, true_components):
        detected_series = detected_components.get('seasonality')
        true_series = true_components.get('seasonality')
        if detected_series is None or true_series is None:
            return 0.5
        rmse_penalty = self._component_rmse_penalty(detected_series, true_series)
        wasserstein_penalty = self._component_wasserstein_penalty(detected_series, true_series)
        # Blend RMSE (point accuracy) with Wasserstein (shape/energy fitting)
        # Wasserstein captures overall shape and energy distribution better
        # than RMSE alone, which can over-penalize phase shifts
        return 0.5 * rmse_penalty + 0.5 * wasserstein_penalty

    def _seasonality_changepoint_loss(
        self, detected_cp, true_cp, detected_components, true_components, date_index
    ):
        if not true_cp:
            return 0.1 * len(detected_cp or [])
        if date_index is None:
            return 0.5 * len(true_cp)
        detected_dates = [
            self._parse_generic_date(event) for event in (detected_cp or [])
        ]
        seasonality_array = np.asarray(
            detected_components.get('seasonality', []), dtype=float
        )
        true_array = np.asarray(true_components.get('seasonality', []), dtype=float)
        if seasonality_array.size == 0 or seasonality_array.size != len(date_index):
            return 0.6 * len(true_cp)
        loss = 0.0
        for event in true_cp:
            cp_date = self._parse_generic_date(event)
            if cp_date is None:
                continue
            match = any(
                abs(cp_date - det_date) <= self._change_tolerance
                for det_date in detected_dates
            )
            if match:
                continue
            idx = date_index.get_indexer([cp_date], method='nearest')[0]
            left_slice = slice(max(0, idx - self.seasonality_window), idx)
            right_slice = slice(
                idx, min(len(seasonality_array), idx + self.seasonality_window)
            )
            left_mean = (
                np.nanmean(seasonality_array[left_slice])
                if left_slice.stop > left_slice.start
                else np.nan
            )
            right_mean = (
                np.nanmean(seasonality_array[right_slice])
                if right_slice.stop > right_slice.start
                else np.nan
            )
            true_left = (
                np.nanmean(true_array[left_slice])
                if true_array.size == seasonality_array.size
                else np.nan
            )
            true_right = (
                np.nanmean(true_array[right_slice])
                if true_array.size == seasonality_array.size
                else np.nan
            )
            if np.isnan(left_mean) or np.isnan(right_mean):
                loss += 0.6
            else:
                detected_change = abs(right_mean - left_mean)
                expected_change = (
                    abs(true_right - true_left)
                    if not np.isnan(true_left) and not np.isnan(true_right)
                    else np.nan
                )
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
                penalties.append(
                    abs(detected_level - true_level) / (abs(true_level) + 1e-6)
                )
        if true_ratio is not None:
            if detected_ratio is None:
                penalties.append(abs(true_ratio) + 0.5)
            else:
                penalties.append(
                    abs(detected_ratio - true_ratio) / (abs(true_ratio) + 1e-6)
                )
        if not penalties:
            return 0.0
        return sum(min(p, 2.0) for p in penalties) / len(penalties)

    def _noise_regime_loss(self, detected_cp, true_cp):
        detected_dates = [
            self._parse_generic_date(event) for event in (detected_cp or [])
        ]
        true_dates = [self._parse_generic_date(event) for event in (true_cp or [])]
        if not true_dates:
            return 0.05 * len(detected_dates)
        loss = 0.0
        for true_date in true_dates:
            match = any(
                abs(true_date - det_date) <= self._change_tolerance
                for det_date in detected_dates
            )
            if not match:
                loss += 0.6
        false_positives = len(
            [
                det_date
                for det_date in detected_dates
                if not any(
                    abs(det_date - true_date) <= self._change_tolerance
                    for true_date in true_dates
                )
            ]
        )
        loss += 0.1 * false_positives
        return loss

    def _metadata_loss(self, detected_scale, true_scale, detected_type, true_type):
        penalties = []
        if true_scale is not None:
            if detected_scale is None:
                penalties.append(abs(true_scale) + 0.5)
            else:
                penalties.append(
                    abs(detected_scale - true_scale) / (abs(true_scale) + 1e-6)
                )
        if true_type is not None:
            penalties.append(0.0 if detected_type == true_type else 0.3)
        if not penalties:
            return 0.0
        return sum(penalties) / len(penalties)

    def _regressor_loss(self, detected_regressors, true_regressors):
        true_by_date, true_coeffs = self._normalize_regressor_schema(true_regressors)
        if not true_by_date and not true_coeffs:
            return 0.0

        detected_by_date, detected_coeffs = self._normalize_regressor_schema(
            detected_regressors
        )

        penalties = []
        if true_by_date:
            event_penalties = []
            for date, impacts in true_by_date.items():
                detected_on_date = detected_by_date.get(date, {})
                for reg_name, true_value in impacts.items():
                    if reg_name not in detected_on_date:
                        event_penalties.append(1.0)
                        continue
                    det_value = detected_on_date.get(reg_name, 0.0)
                    rel_error = abs(det_value - true_value) / (abs(true_value) + 1e-6)
                    event_penalties.append(min(rel_error, 2.0))
            if event_penalties:
                penalties.append(float(np.mean(event_penalties)))

            # Penalize extra detected event-date pairs not present in truth.
            true_pairs = {
                (pd.Timestamp(date), str(reg_name))
                for date, impacts in true_by_date.items()
                for reg_name in impacts.keys()
            }
            detected_pairs = {
                (pd.Timestamp(date), str(reg_name))
                for date, impacts in detected_by_date.items()
                for reg_name in impacts.keys()
            }
            if detected_pairs:
                extras = len(detected_pairs - true_pairs)
                penalties.append(extras / max(1, len(true_pairs)))

        if true_coeffs:
            coeff_penalties = []
            for reg_name, true_value in true_coeffs.items():
                if reg_name not in detected_coeffs:
                    coeff_penalties.append(1.0)
                    continue
                det_value = detected_coeffs.get(reg_name, 0.0)
                rel_error = abs(det_value - true_value) / (abs(true_value) + 1e-6)
                coeff_penalties.append(min(rel_error, 2.0))
            if coeff_penalties:
                penalties.append(float(np.mean(coeff_penalties)))

            coeff_extras = len(set(detected_coeffs.keys()) - set(true_coeffs.keys()))
            penalties.append(coeff_extras / max(1, len(true_coeffs)))

        if not penalties:
            return 0.0
        return float(np.mean(penalties))

    def _normalize_regressor_schema(self, regressor_payload):
        regressor_payload = regressor_payload or {}
        if not isinstance(regressor_payload, dict):
            return {}, {}

        if 'by_date' in regressor_payload or 'coefficients' in regressor_payload:
            by_date_raw = regressor_payload.get('by_date', {})
            coeffs_raw = regressor_payload.get('coefficients', {})
        else:
            # Backward compatibility: treat direct mapping as by_date schema.
            by_date_raw = regressor_payload
            coeffs_raw = {}

        by_date = {}
        if isinstance(by_date_raw, dict):
            for date, impacts in by_date_raw.items():
                if not isinstance(impacts, dict):
                    continue
                normalized_impacts = {}
                for reg_name, value in impacts.items():
                    if self._is_number(value):
                        normalized_impacts[str(reg_name)] = float(value)
                if normalized_impacts:
                    by_date[pd.Timestamp(date)] = normalized_impacts

        coefficients = {}
        if isinstance(coeffs_raw, dict):
            for reg_name, value in coeffs_raw.items():
                if self._is_number(value):
                    coefficients[str(reg_name)] = float(value)

        return by_date, coefficients

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
            magnitude = (
                abs(float(event[1]))
                if len(event) > 1 and self._is_number(event[1])
                else 1.0
            )
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
            magnitude = (
                abs(float(event[1]))
                if len(event) > 1 and self._is_number(event[1])
                else 1.0
            )
            anomaly_type = event[2] if len(event) > 2 else 'point_outlier'
            duration = (
                int(event[3])
                if len(event) > 3 and isinstance(event[3], (int, float))
                else 1
            )
        else:
            date = pd.Timestamp(event)
            magnitude = 1.0
            anomaly_type = 'point_outlier'
            duration = 1
        if not np.isfinite(magnitude):
            magnitude = 0.0
        if duration < 1:
            duration = 1
        if anomaly_type is None:
            anomaly_type = 'point_outlier'
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
        rmse = np.sqrt(np.nanmean((detected_arr - true_arr) ** 2))

        true_std = float(np.nanstd(true_arr))
        if true_std < 1e-6 or not np.isfinite(true_std):
            true_std = float(np.nanmean(np.abs(true_arr))) + 1e-6
        det_std = float(np.nanstd(detected_arr))

        normalized_rmse = rmse / (true_std + 1e-6)
        amplitude_penalty = abs(det_std - true_std) / (true_std + 1e-6)

        mean_scale = true_std + abs(float(np.nanmean(true_arr))) + 1e-6
        mean_penalty = abs(
            float(np.nanmean(detected_arr)) - float(np.nanmean(true_arr))
        ) / mean_scale

        if detected_arr.size < 3:
            corr_penalty = 1.0
        else:
            det_var = float(np.nanstd(detected_arr))
            true_var = float(np.nanstd(true_arr))
            if det_var < 1e-12 or true_var < 1e-12:
                corr_penalty = 1.0 if abs(det_var - true_var) > 1e-9 else 0.0
            else:
                corr = float(np.corrcoef(detected_arr, true_arr)[0, 1])
                if not np.isfinite(corr):
                    corr_penalty = 1.0
                else:
                    corr_penalty = 1.0 - max(0.0, corr) ** 2

        combined_penalty = (
            0.55 * min(normalized_rmse, 3.0)
            + 0.25 * min(amplitude_penalty, 3.0)
            + 0.15 * min(corr_penalty, 1.5)
            + 0.05 * min(mean_penalty, 2.0)
        )
        return min(combined_penalty, 3.0)

    @staticmethod
    def _component_wasserstein_penalty(detected, true):
        """
        Compute a Wasserstein-inspired shape fitting penalty between two component arrays.

        This metric captures overall energy distribution and shape similarity by
        comparing the sorted cumulative distributions (1D Wasserstein / earth mover's
        distance) of the differentials, plus a direct differential Wasserstein distance.

        Advantages over RMSE for seasonality:
        - Tolerant of small phase shifts (common in seasonality estimation)
        - Captures overall energy/amplitude matching
        - Rewards correct shape even when slightly misaligned in time

        Returns
        -------
        float
            Penalty between 0 (perfect match) and 3.0 (poor match).
        """
        detected_arr = np.asarray(detected, dtype=float).ravel()
        true_arr = np.asarray(true, dtype=float).ravel()
        length = min(detected_arr.size, true_arr.size)
        if length < 2:
            return 0.5
        detected_arr = detected_arr[:length]
        true_arr = true_arr[:length]
        mask = np.isfinite(detected_arr) & np.isfinite(true_arr)
        if mask.sum() < 2:
            return 0.5
        detected_arr = detected_arr[mask]
        true_arr = true_arr[mask]

        true_std = float(np.nanstd(true_arr))
        if true_std < 1e-6 or not np.isfinite(true_std):
            true_std = float(np.nanmean(np.abs(true_arr))) + 1e-6

        # 1. Value-level Wasserstein: compare sorted distributions
        det_sorted = np.sort(detected_arr)
        true_sorted = np.sort(true_arr)
        value_wasserstein = np.mean(np.abs(det_sorted - true_sorted)) / (true_std + 1e-6)

        # 2. Differential Wasserstein: compare step-to-step changes
        # This captures shape/energy better than point-wise comparison
        det_diff = np.diff(detected_arr)
        true_diff = np.diff(true_arr)
        diff_std = float(np.nanstd(true_diff))
        if diff_std < 1e-6 or not np.isfinite(diff_std):
            diff_std = float(np.nanmean(np.abs(true_diff))) + 1e-6

        det_diff_sorted = np.sort(det_diff)
        true_diff_sorted = np.sort(true_diff)
        diff_wasserstein = np.mean(np.abs(det_diff_sorted - true_diff_sorted)) / (diff_std + 1e-6)

        # 3. Energy ratio: total absolute energy comparison
        det_energy = float(np.sum(np.abs(detected_arr)))
        true_energy = float(np.sum(np.abs(true_arr)))
        energy_ratio = abs(det_energy - true_energy) / (true_energy + 1e-6)

        combined = (
            0.40 * min(value_wasserstein, 3.0)
            + 0.40 * min(diff_wasserstein, 3.0)
            + 0.20 * min(energy_ratio, 3.0)
        )
        return min(combined, 3.0)

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
        'seasonality_shape_loss': 0.6,
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
            int(lag) for lag in (seasonality_lags or []) if lag is not None and lag > 0
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
            series_names = [
                name for name in observed_df.columns if name in resolved_components
            ]
            if not series_names:
                raise ValueError(
                    "No overlapping series between observed data and component container."
                )

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
        level_shift = self._component_to_series(
            component_dict.get('level_shift'), index
        )
        seasonality = self._component_to_series(
            component_dict.get('seasonality'), index
        )
        holidays = self._component_to_series(component_dict.get('holidays'), index)
        anomalies = self._component_to_series(component_dict.get('anomalies'), index)
        noise = self._component_to_series(component_dict.get('noise'), index)

        component_sum = trend + level_shift + seasonality + holidays + anomalies + noise
        residual = observed_series - component_sum

        reconstruction_loss = self._normalized_rmse(observed_series, residual)
        trend_smoothness = self._trend_complexity_penalty(trend.to_numpy(dtype=float))
        trend_dominance = self._trend_dominance_penalty(
            trend,
            {
                'level_shift': level_shift,
                'seasonality': seasonality,
                'holidays': holidays,
                'anomalies': anomalies,
            },
        )

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

        seasonality_shape = self._seasonality_shape_penalty(
            observed_series,
            trend,
            level_shift,
            seasonality + holidays,
        )

        return {
            'reconstruction_loss': reconstruction_loss,
            'trend_smoothness_loss': trend_smoothness,
            'trend_dominance_loss': trend_dominance,
            'seasonality_capture_loss': seasonality_capture,
            'seasonality_shape_loss': seasonality_shape,
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
        series = series.iloc[: len(index)]
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
        rmse = np.sqrt(np.mean(residual**2))
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
        penalty = (ratio - self.trend_dominance_target) / (
            1.0 - self.trend_dominance_target + 1e-6
        )
        return min(max(penalty, 0.0), 2.0)

    def _seasonality_capture_penalty(
        self, observed, trend, level_shift, seasonal_bundle
    ):
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

    def _anomaly_capture_penalty(
        self, observed, trend, level_shift, seasonal_bundle, anomalies
    ):
        anomaly_std = float(np.nanstd(anomalies.to_numpy(dtype=float)))
        if anomaly_std < 1e-6:
            return 0.0

        residual = observed - trend - level_shift - seasonal_bundle
        pre_values = residual.to_numpy(dtype=float)
        pre_std = float(np.nanstd(pre_values))
        if pre_std < self.anomaly_min_pre_std:
            return 0.0

        post_series = residual - anomalies
        post_values = post_series.to_numpy(dtype=float)
        post_std = float(np.nanstd(post_values))
        if not np.isfinite(post_std):
            return 0.0

        # Std-based improvement
        std_improvement = max(0.0, (pre_std - post_std) / (pre_std + 1e-6))

        # Kurtosis-based improvement: anomaly removal should reduce heavy tails
        pre_finite = pre_values[np.isfinite(pre_values)]
        post_finite = post_values[np.isfinite(post_values)]
        kurtosis_improvement = 0.0
        if pre_finite.size > 4 and post_finite.size > 4:
            pre_kurtosis = self._excess_kurtosis(pre_finite)
            post_kurtosis = self._excess_kurtosis(post_finite)
            if pre_kurtosis > 0.5:  # Only penalize if there are heavy tails to capture
                kurtosis_improvement = max(0.0, (pre_kurtosis - post_kurtosis) / (pre_kurtosis + 1e-6))

        # Blend std and kurtosis improvement
        combined_improvement = 0.7 * std_improvement + 0.3 * kurtosis_improvement

        if combined_improvement >= self.anomaly_improvement_target:
            return 0.0
        deficit = self.anomaly_improvement_target - combined_improvement
        return min(max(deficit, 0.0), 1.5)

    def _seasonality_shape_penalty(
        self, observed, trend, level_shift, seasonal_bundle
    ):
        """
        Wasserstein-based penalty assessing whether the seasonal component captures
        the right shape/energy of the detrended signal's periodic structure.

        Compares the differential (step-to-step change) distributions of the
        detrended observed signal and the seasonal component. Good seasonality
        extraction should produce a seasonal component whose differential
        Wasserstein distribution is close to the periodic portion of the original.
        """
        seasonal_arr = seasonal_bundle.to_numpy(dtype=float)
        seasonal_std = float(np.nanstd(seasonal_arr))
        if seasonal_std < 1e-6:
            return 0.0  # No seasonality to evaluate

        detrended = (observed - trend - level_shift).to_numpy(dtype=float)
        mask = np.isfinite(detrended) & np.isfinite(seasonal_arr)
        if mask.sum() < 3:
            return 0.0
        detrended = detrended[mask]
        seasonal_arr = seasonal_arr[mask]

        # Compare differential distributions (shape/energy matching)
        det_diff = np.diff(detrended)
        sea_diff = np.diff(seasonal_arr)

        if det_diff.size < 2:
            return 0.0

        # Sort and compute 1D Wasserstein distance on differentials
        det_diff_sorted = np.sort(det_diff)
        sea_diff_sorted = np.sort(sea_diff)
        diff_scale = float(np.std(det_diff))
        if diff_scale < 1e-6 or not np.isfinite(diff_scale):
            diff_scale = float(np.mean(np.abs(det_diff))) + 1e-6

        diff_wasserstein = np.mean(np.abs(det_diff_sorted - sea_diff_sorted)) / (diff_scale + 1e-6)

        # Energy ratio: seasonal should capture a meaningful portion of detrended energy
        detrended_energy = float(np.sum(detrended ** 2))
        residual_energy = float(np.sum((detrended - seasonal_arr) ** 2))
        if detrended_energy > 1e-6:
            energy_capture = 1.0 - (residual_energy / detrended_energy)
            # Penalize if capturing too little or negative (worse than nothing)
            energy_penalty = max(0.0, 0.3 - energy_capture) if energy_capture >= 0 else abs(energy_capture)
        else:
            energy_penalty = 0.0

        combined = 0.6 * min(diff_wasserstein, 2.0) + 0.4 * min(energy_penalty, 2.0)
        return min(combined, 2.0)

    @staticmethod
    def _excess_kurtosis(values):
        """Compute excess kurtosis (Fisher definition: normal = 0)."""
        n = len(values)
        if n < 4:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-9:
            return 0.0
        m4 = np.mean((values - mean) ** 4)
        return (m4 / (std ** 4)) - 3.0

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
        starting_params=None,
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
        starting_params : dict, optional
            Optional detector parameter seed evaluated before random search.
        """
        self.synthetic_generator = synthetic_generator
        self.loss_calculator = loss_calculator or FeatureDetectionLoss()
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        if starting_params is not None and not isinstance(starting_params, dict):
            raise ValueError("starting_params must be a dict or None.")
        self.starting_params = copy.deepcopy(starting_params)

        self.best_params = None
        self.best_loss = float('inf')
        self.best_total_loss = float('inf')
        self.optimization_history = []
        self.baseline_loss = None
        self.history_df = None

    def optimize(self, starting_params=None):
        """
        Run genetic-style optimization to find best detector parameters.

        Parameters
        ----------
        starting_params : dict, optional
            Optional seed parameter configuration. Overrides constructor value
            when provided.

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

        seed = self.starting_params if starting_params is None else starting_params
        if seed is not None and not isinstance(seed, dict):
            raise ValueError("starting_params must be a dict or None.")
        return self._random_search(starting_params=seed)

    def _default_detector_params(self):
        """Return a deep-copied set of default detector parameters."""
        detector = TimeSeriesFeatureDetector()
        return {
            'rough_seasonality_params': copy.deepcopy(
                detector.rough_seasonality_params
            ),
            'seasonality_params': copy.deepcopy(detector.seasonality_params),
            'holiday_params': copy.deepcopy(detector.holiday_params),
            'anomaly_params': copy.deepcopy(detector.anomaly_params),
            'changepoint_params': copy.deepcopy(detector.changepoint_params),
            'level_shift_params': copy.deepcopy(detector.level_shift_params),
            'general_transformer_params': copy.deepcopy(
                detector.general_transformer_params
            ),
            'standardize': detector.standardize,
            'smoothing_window': detector.smoothing_window,
        }

    def _random_search(self, starting_params=None):
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

        if starting_params is not None:
            starting_signature = self._param_signature(starting_params)
            if starting_signature in evaluated_signatures:
                print("Starting params match baseline; skipping duplicate evaluation.")
            else:
                try:
                    start_time = time.time()
                    starting_loss = self._evaluate_params(starting_params)
                    starting_runtime = time.time() - start_time

                    self.optimization_history.append(
                        {
                            'iteration': 'starting',
                            'params': copy.deepcopy(starting_params),
                            'loss': starting_loss['total_loss'],
                            'loss_breakdown': starting_loss,
                            'runtime': starting_runtime,
                        }
                    )
                    evaluated_signatures.add(starting_signature)
                    print(
                        f"Starting params loss = {starting_loss['total_loss']:.4f}, "
                        f"runtime = {starting_runtime:.2f}s"
                    )
                except Exception as e:
                    print(f"Warning: Starting params evaluation failed with error: {e}")

        successful_iterations = 0
        failed_iterations = 0

        for i in range(self.n_iterations):
            params = None
            attempts = 0
            parent_pool = sorted(
                self.optimization_history,
                key=lambda x: x.get('balanced_loss', x.get('loss', float('inf'))),
            )
            parent_pool = (
                parent_pool[: max(2, min(6, len(parent_pool)))] if parent_pool else []
            )

            # Generate new parameters, avoiding duplicates
            while (
                params is None or self._param_signature(params) in evaluated_signatures
            ):
                attempts += 1
                if parent_pool and rng.random() < 0.7:
                    if len(parent_pool) >= 2:
                        chosen = rng.sample(parent_pool, 2)
                        params = self._crossover_params(
                            chosen[0]['params'], chosen[1]['params'], rng
                        )
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
        runtimes = [
            entry.get('runtime')
            for entry in self.optimization_history
            if entry.get('runtime') is not None
        ]
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
            canonical = FeatureDetectionOptimizer._signature_safe_value(params)
            return json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        except Exception:
            return repr(params)

    @staticmethod
    def _signature_safe_value(value):
        """Convert potentially non-serializable params into deterministic JSON-safe form."""
        if isinstance(value, dict):
            return {
                str(k): FeatureDetectionOptimizer._signature_safe_value(v)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        if isinstance(value, (list, tuple)):
            return [FeatureDetectionOptimizer._signature_safe_value(v) for v in value]
        if isinstance(value, set):
            converted = [
                FeatureDetectionOptimizer._signature_safe_value(v) for v in value
            ]
            return sorted(converted, key=lambda x: json.dumps(x, sort_keys=True))
        if isinstance(value, np.ndarray):
            return FeatureDetectionOptimizer._signature_safe_value(value.tolist())
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            val = float(value)
            if not np.isfinite(val):
                return str(val)
            return val
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, float) and not np.isfinite(value):
                return str(value)
            return value
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        # Deterministic fallback for custom objects.
        return {'__class__': value.__class__.__name__, '__repr__': str(value)}

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

        # Calculate scalers based on entire history using a robust lower quantile,
        # which avoids pathological domination from a single tiny metric value.
        scalers = {}
        for key in self.loss_calculator.weights.keys():
            col = self.history_df[key].replace([np.inf, -np.inf], np.nan)
            positive = col[col > 0].dropna()
            if positive.empty:
                scalers[key] = 1.0
            else:
                scale = float(np.nanpercentile(positive, 25))
                if not np.isfinite(scale) or scale <= 1e-6:
                    scale = float(np.nanmedian(positive))
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

        # Select from the strongest balanced candidates, then require the best raw loss
        # within that pool so we don't prefer configurations with worse total loss.
        balanced_arr = np.asarray(balanced_losses, dtype=float)
        raw_arr = np.asarray(
            [entry.get('loss', np.inf) for entry in self.optimization_history],
            dtype=float,
        )
        valid_idx = np.where(np.isfinite(balanced_arr) & np.isfinite(raw_arr))[0]
        if valid_idx.size == 0:
            best_idx = int(np.nanargmin(balanced_arr))
            candidate_pool_size = 1
        else:
            sorted_balanced = valid_idx[np.argsort(balanced_arr[valid_idx])]
            candidate_pool_size = max(1, min(8, int(np.ceil(sorted_balanced.size * 0.2))))
            top_candidates = sorted_balanced[:candidate_pool_size]
            best_idx = int(top_candidates[np.argmin(raw_arr[top_candidates])])

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
            baseline_balanced = baseline_entry.get(
                'balanced_loss', baseline_entry['loss']
            )
            improvement = baseline_balanced - self.best_loss
            improvement_pct = (
                (improvement / baseline_balanced * 100) if baseline_balanced != 0 else 0
            )

            print(f"\n{'='*80}")
            print(f"OPTIMIZATION RESULTS")
            print(f"{'='*80}")
            print(
                f"Baseline balanced loss: {baseline_balanced:.4f} (raw: {baseline_entry['loss']:.4f})"
            )
            print(
                f"Best balanced loss:     {self.best_loss:.4f} (raw: {self.best_total_loss:.4f})"
            )
            print(f"Selection pool size:    {candidate_pool_size}")
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
            'best_params': copy.deepcopy(self.best_params)
            if self.best_params
            else None,
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

        component_ranges = {}
        frozen_components = []
        disabled_component_counts = {}
        if self.optimization_history:
            for key in self.loss_calculator.weights.keys():
                values = []
                for entry in self.optimization_history:
                    breakdown = entry.get('loss_breakdown') or {}
                    val = breakdown.get(key)
                    if val is not None and np.isfinite(val):
                        values.append(float(val))
                if values:
                    comp_min = float(np.min(values))
                    comp_max = float(np.max(values))
                    comp_range = comp_max - comp_min
                    component_ranges[key] = {
                        'min': comp_min,
                        'max': comp_max,
                        'range': comp_range,
                    }
                    if comp_range <= 1e-9:
                        frozen_components.append(key)

            for entry in self.optimization_history:
                breakdown = entry.get('loss_breakdown') or {}
                disabled = breakdown.get('disabled_components') or []
                for key in disabled:
                    disabled_component_counts[key] = (
                        disabled_component_counts.get(key, 0) + 1
                    )

        if component_ranges:
            summary['component_ranges'] = component_ranges
        if frozen_components:
            summary['frozen_components'] = sorted(frozen_components)
        if disabled_component_counts:
            summary['disabled_components'] = sorted(disabled_component_counts.keys())
            summary['disabled_component_counts'] = disabled_component_counts

        return summary
