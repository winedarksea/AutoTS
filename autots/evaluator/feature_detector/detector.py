# -*- coding: utf-8 -*-
"""
TimeSeriesFeatureDetector - Main orchestrator class.

Composes functionality from component mixins for decomposition,
seasonality, trend, holidays, anomalies, rescaling, and formatting.
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

from .components.decomposition import DecompositionMixin
from .components.seasonality import SeasonalityMixin
from .components.trend import TrendMixin
from .components.holidays import HolidayMixin
from .components.anomalies import AnomalyMixin
from .extended_anomaly import ExtendedAnomalyDetector, ExtendedAnomalyMixin
from .event_dag import (
    build_event_dag_from_detector,
    empty_event_dag,
    resolve_event_dag_params,
)
from .event_dag_view import filter_event_dag, plot_event_dag_timeline
from .utils.rescaling import RescalingMixin
from .utils.formatting import FormattingMixin


class TimeSeriesFeatureDetector(
    DecompositionMixin,
    SeasonalityMixin,
    TrendMixin,
    HolidayMixin,
    AnomalyMixin,
    ExtendedAnomalyMixin,
    RescalingMixin,
    FormattingMixin,
):
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

    TEMPLATE_VERSION = "1.2"

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
        extended_anomaly_params=None,
        event_dag_params=None,
        holiday_country=None,
        holiday_countries=None,
    ):
        # Set detection_mode first so it can be used in other initializations
        self.detection_mode = detection_mode

        # Validate detection_mode
        if detection_mode not in ['multivariate', 'univariate']:
            raise ValueError(
                f"detection_mode must be 'multivariate' or 'univariate', got '{detection_mode}'"
            )

        self.rough_seasonality_params = (
            copy.deepcopy(rough_seasonality_params)
            if rough_seasonality_params is not None
            else {
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
        )
        self.seasonality_params = (
            copy.deepcopy(seasonality_params)
            if seasonality_params is not None
            else {
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
        )
        self.holiday_country = holiday_country
        self.holiday_countries = (
            copy.deepcopy(holiday_countries) if holiday_countries is not None else {}
        )
        self._resolved_holiday_countries = {}
        self.rough_seasonality_params['holiday_countries_used'] = False
        self.seasonality_params['holiday_countries_used'] = False
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
        # Extended anomaly detector params; falsy value disables extended detection.
        # Default enables a conservative extended pass.
        if extended_anomaly_params is None:
            self.extended_anomaly_params = {
                'sustained_window': 7,
                'sustained_baseline': 45,
                'sustained_threshold': 2.2,
                'cusum_k': 0.5,
                'cusum_h': 4.0,
                'slope_reversion_min_hold': 5,
                'slope_reversion_min_reversion': 7,
                'slope_reversion_cumsum_threshold': 3.0,
                'slope_reversion_max_duration': 84,
                'decay_lookahead': 21,
                'decay_fit_min_r2': 0.5,
                'min_segment_run': 2,
                'sustained_hysteresis': 0.7,
                'segment_max_gap': 1,
                'merge_distance_days': 2,
            }
        else:
            # Explicitly setting to False/empty dict disables; any truthy dict enables
            self.extended_anomaly_params = (
                extended_anomaly_params if extended_anomaly_params else {}
            )
        self.event_dag_params = resolve_event_dag_params(event_dag_params)

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
        self._holiday_regressors_temp = None
        self._holiday_regressor_columns = None
        self._holiday_dates_temp = {}
        self.noise_changepoints = {}
        self.noise_to_signal_ratios = {}
        self.series_noise_levels = {}
        self.series_scales = {}
        self.shared_events = {'anomalies': [], 'level_shifts': []}
        self.event_dag = empty_event_dag(
            params=self.event_dag_params,
            detection_mode=self.detection_mode,
            construction_mode=(
                'broadcast' if self.detection_mode == 'univariate' else 'full'
            ),
        )
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

        # Step 8.5: Extended anomaly detection on clean residual
        # clean_residual = noise + anomaly signal (structured components removed)
        # This is the cleanest possible input for extended pattern detection.
        # In univariate mode, the mixin aggregates across series before detection
        # and broadcasts shared events back, mirroring AnomalyRemoval behaviour.
        if self.extended_anomaly_params:
            try:
                clean_residual_scaled = (
                    noise_component_scaled + anomaly_component_scaled
                )
                extended_records = self._detect_extended_anomalies(
                    clean_residual_scaled
                )
                self._anomaly_records_temp = self._merge_anomaly_records(
                    self._anomaly_records_temp, extended_records
                )
            except Exception as _ext_exc:
                warnings.warn(
                    f"Extended anomaly detection step failed: {_ext_exc}",
                    RuntimeWarning,
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
        # Fall back to the persisted column list when train_reg was lost (e.g. after
        # a serialization round-trip that preserved config but not runtime state).
        _saved_columns = getattr(self, "_holiday_regressor_columns", None)
        fit_columns = (
            list(train_reg.columns)
            if train_reg is not None and not getattr(train_reg, "empty", True)
            else _saved_columns if _saved_columns else None
        )
        if self.seasonality_model is not None:
            future_reg = None
            if fit_columns is not None:
                future_reg = self._build_holiday_regressors_for_index(
                    future_index,
                    columns=columns,
                    include_anomaly_rules=train_reg is not None,
                )
                future_reg = future_reg.reindex(columns=fit_columns, fill_value=0.0)
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
            'event_dag_params': copy.deepcopy(self.event_dag_params),
            'holiday_country': copy.deepcopy(self.holiday_country),
            'holiday_countries': copy.deepcopy(self.holiday_countries),
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
            'event_dag': empty_event_dag(
                params=self.event_dag_params,
                detection_mode=self.detection_mode,
                construction_mode=(
                    'broadcast' if self.detection_mode == 'univariate' else 'full'
                ),
                series_names=list(self.df_original.columns),
            ),
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
        self.event_dag = empty_event_dag(
            params=self.event_dag_params,
            detection_mode=self.detection_mode,
            construction_mode=(
                'broadcast' if self.detection_mode == 'univariate' else 'full'
            ),
            series_names=list(self.df_original.columns),
        )
        self.reconstructed = None
        self.reconstructed_components = None
        self.reconstruction_error = None
        self.reconstruction_rmse = None
        self._holiday_regressors_temp = None
        self._holiday_regressor_columns = None
        self._holiday_dates_temp = {}

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
                        'season_type': self.series_season_types.get(
                            series_name, 'none'
                        ),
                        'regressor_impacts': {},
                    }
                )
            if include_components:
                features['components'] = self._serialize_components(series_name)
            features['event_dag'] = filter_event_dag(
                self.event_dag,
                series=[series_name],
                include_members=True,
            )
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
        features['event_dag'] = copy.deepcopy(self.event_dag)
        return features

    def get_template(self, deep=True):
        if self.template is None:
            return None
        return copy.deepcopy(self.template) if deep else self.template

    def get_event_dag(self, deep=True):
        """Return Event DAG metadata derived from detector outputs."""
        if self.df_original is None:
            raise RuntimeError("TimeSeriesFeatureDetector has not been fit.")
        return copy.deepcopy(self.event_dag) if deep else self.event_dag

    def _rebuild_event_dag(self):
        """Rebuild Event DAG metadata from current detector event outputs."""
        self.event_dag = build_event_dag_from_detector(self)
        if isinstance(self.template, dict):
            self.template['event_dag'] = copy.deepcopy(self.event_dag)
        return self.event_dag

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
        include_event_dag=False,
        include_event_members=False,
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
            include_event_dag (bool): Include Event DAG cluster and family metadata
            include_event_members (bool): Include raw Event DAG member events
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
                                'type': (
                                    anomaly[2] if len(anomaly) > 2 else 'point_outlier'
                                ),
                                'duration': (
                                    int(anomaly[3]) if len(anomaly) > 3 else None
                                ),
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
            seasonality_cps = getattr(self, '_seasonality_changepoints', {}).get(
                col, []
            )
            if seasonality_cps:
                filtered_scp = _filter_by_date(seasonality_cps)
                if filtered_scp:
                    cp_entries = []
                    for cp in filtered_scp:
                        if isinstance(cp, dict):
                            cp_entries.append(
                                {
                                    'date': pd.to_datetime(cp.get('date')).isoformat(),
                                    'description': cp.get(
                                        'description', 'seasonality_change'
                                    ),
                                }
                            )
                        else:
                            cp_entries.append(
                                {
                                    'date': pd.to_datetime(cp[0]).isoformat(),
                                    'description': (
                                        cp[1] if len(cp) > 1 else 'seasonality_change'
                                    ),
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

        if include_event_dag:
            selected_series_arg = None if series is None else selected_series
            if date_filter is None:
                event_start = None
                event_end = None
            elif isinstance(date_filter, set):
                event_start = min(date_filter) if date_filter else None
                event_end = max(date_filter) if date_filter else None
            else:
                event_start, event_end = date_filter
            result['event_dag'] = filter_event_dag(
                self.event_dag,
                series=selected_series_arg,
                start_date=event_start,
                end_date=event_end,
                include_members=include_event_members,
            )

        if return_json:
            import json

            return json.dumps(result, indent=2)

        return result

    def plot(
        self,
        series_name=None,
        figsize=(16, 14),
        save_path=None,
        show=True,
        separate_noise_anomaly_panels=True,
        dual_axis_seasonality_holidays=True,
        dual_axis_trend_level_shift=True,
    ):
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
            separate_noise_anomaly_panels=separate_noise_anomaly_panels,
            dual_axis_seasonality_holidays=dual_axis_seasonality_holidays,
            dual_axis_trend_level_shift=dual_axis_trend_level_shift,
            show_reconstructed_on_top=True,
        )
        return fig

    def plot_event_dag(
        self,
        series=None,
        start_date=None,
        end_date=None,
        show_members=False,
        figsize=(14, 6),
        save_path=None,
        show=True,
    ):
        """Plot Event DAG macro-events on a timeline-first layout."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting.")
        if self.df_original is None:
            raise RuntimeError("Call fit() before plot_event_dag().")
        if series is None:
            selected_series = None
        elif isinstance(series, str):
            if series not in self.df_original.columns:
                raise ValueError(f"Series '{series}' not found.")
            selected_series = [series]
        else:
            selected_series = list(series)
            missing = set(selected_series) - set(self.df_original.columns)
            if missing:
                raise ValueError(f"Series not found: {missing}")
        return plot_event_dag_timeline(
            self.event_dag,
            series=selected_series,
            start_date=start_date,
            end_date=end_date,
            show_members=show_members,
            figsize=figsize,
            save_path=save_path,
            show=show,
        )

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
        # Keep sampled params aligned with multivariate detector semantics:
        # each series gets its own changepoints by default.
        changepoint_params['aggregate_method'] = 'individual'

        # Level shift params
        level_shift_params = LevelShiftMagic.get_new_params(method=method)
        level_shift_params['output'] = 'multivariate'  # Ensure correct output mode

        # General transformer params (for pre-trend processing)
        general_transformer_params = GeneralTransformer.get_new_params(
            method="filters", allow_none=True, transformer_max_depth=2
        )

        # Extended anomaly detector params (disabled ~15% of the time)
        extended_anomaly_params = (
            None
            if random.random() < 0.15
            else ExtendedAnomalyDetector.get_new_params(method=method)
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
            'extended_anomaly_params': extended_anomaly_params,
        }

    def _apply_detector_params(self, params):
        """Apply detector hyperparameters to this instance."""
        if not isinstance(params, dict):
            raise ValueError("params must be a dict of detector parameters.")

        if params.get('rough_seasonality_params') is not None:
            self.rough_seasonality_params = copy.deepcopy(
                params['rough_seasonality_params']
            )
            self.rough_seasonality_params['holiday_countries_used'] = False
        if params.get('seasonality_params') is not None:
            self.seasonality_params = copy.deepcopy(params['seasonality_params'])
            self.seasonality_params['holiday_countries_used'] = False

        if 'holiday_country' in params:
            self.holiday_country = params.get('holiday_country')
        if 'holiday_countries' in params:
            holiday_countries = params.get('holiday_countries')
            self.holiday_countries = (
                copy.deepcopy(holiday_countries)
                if holiday_countries is not None
                else {}
            )

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
        if 'extended_anomaly_params' in params:
            self.extended_anomaly_params = (
                copy.deepcopy(params['extended_anomaly_params']) or {}
            )
        if 'event_dag_params' in params:
            self.event_dag_params = resolve_event_dag_params(
                copy.deepcopy(params['event_dag_params'])
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
            print(
                "TUNE WITH SYNTHETIC: real data -> synthetic labels -> detector tuning"
            )
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
            print(
                "\n[Step 3/4] Optimizing feature detector on labeled synthetic data..."
            )
        loss_kwargs = copy.deepcopy(loss_params) if loss_params else {}
        if loss_weights is not None:
            loss_kwargs['weights'] = copy.deepcopy(loss_weights)
        from .loss import FeatureDetectionLoss

        loss_calc = FeatureDetectionLoss(**loss_kwargs)
        from .optimizer import FeatureDetectionOptimizer

        optimizer = FeatureDetectionOptimizer(
            synthetic_generator=tuned_generator,
            loss_calculator=loss_calc,
            n_iterations=n_detector_iterations,
            random_seed=tune_seed,
            starting_params=starting_params,
        )
        best_detector_params = optimizer.optimize()
        if best_detector_params is None:
            raise RuntimeError(
                "Feature detector optimization did not produce parameters."
            )

        if verbose:
            print(
                "\n[Step 4/4] Applying optimized parameters and fitting on real data..."
            )
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
