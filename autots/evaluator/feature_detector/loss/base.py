# -*- coding: utf-8 -*-
"""FeatureDetectionLoss - Core loss calculation for feature detection."""

import numpy as np
import pandas as pd
import copy
import warnings
from .metrics import LossMetricsMixin
from .evaluators import LossEvaluatorsMixin


class FeatureDetectionLoss(LossMetricsMixin, LossEvaluatorsMixin):
    """
    Comprehensive loss calculator for feature detection optimization.

    Each synthetic label family contributes to the total loss:
    - Trend changepoints and slopes
    - Level shifts
    - Anomalies (including shared events and post patterns)
    - Holiday timing, direct impacts, and splash/bridge days
    - Seasonality strength, patterns, and changepoints
    - Noise regimes and noise-to-signal characteristics
    - Low-frequency noise structure consistency (drift/shift leakage)
    - Series-level metadata consistency (scale, type)
    - Regressor impacts when present
    """

    DEFAULT_WEIGHTS = {
        'trend_loss': 1.0,
        'level_shift_loss': 1.3,
        'anomaly_loss': 1.3,
        'holiday_event_loss': 1.2,
        'holiday_impact_loss': 0.9,
        'holiday_splash_loss': 0.03,  # tends to have wayyy too much impact
        'holiday_recall_loss': 0.9,
        'seasonality_strength_loss': 2.0,
        'seasonality_pattern_loss': 2.0,
        'seasonality_changepoint_loss': 0.01,
        'noise_level_loss': 0.5,
        'noise_regime_loss': 0.4,
        'noise_structure_loss': 0.2,
        'metadata_loss': 0.05,  # also tends to have too much impact
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
            raise ValueError("invalid_loss_mode must be either 'penalty' or 'raise'.")
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
            (
                detected_features.get('components')
                if isinstance(detected_features, dict)
                else None
            ),
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
            detected_cp=detected.get('trend_changepoints', []),
            detected_ls=detected.get('level_shifts', []),
            true_ls=true.get('level_shifts', []),
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
            date_index=date_index,
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
        noise_structure_loss = self._evaluate_component_loss(
            key='noise_structure_loss',
            series_name=series_name,
            effective_weights=effective_weights,
            fn=self._noise_structure_loss,
            detected_components=detected_components,
            true_components=true_components,
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
            'noise_structure_loss': noise_structure_loss,
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
            has_ratio = not self._is_empty_label(
                true_series.get('noise_to_signal_ratio')
            )
            return bool(has_level or has_ratio)
        if key == 'noise_regime_loss':
            return not self._is_empty_label(true_series.get('noise_changepoints'))
        if key == 'noise_structure_loss':
            return not self._is_empty_label(true_component_map.get('noise'))
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

        context = (
            f"{component_name}"
            if series_name is None
            else f"{component_name}:{series_name}"
        )
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
