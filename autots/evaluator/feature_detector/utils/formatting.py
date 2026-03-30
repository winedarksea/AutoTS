# -*- coding: utf-8 -*-
"""FormattingMixin for TimeSeriesFeatureDetector."""

import numpy as np
import pandas as pd
import copy
import warnings
from autots.datasets.synthetic import SyntheticDailyGenerator

from ..event_dag import build_event_dag_from_detector


class FormattingMixin:
    """Mixin providing template building, serialization, and reconstruction validation."""

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

            self.series_seasonality_profiles[series_name] = (
                self._estimate_seasonality_profile(seasonality_series, series_scale)
            )
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
        self.event_dag = build_event_dag_from_detector(self)
        self.template['event_dag'] = copy.deepcopy(self.event_dag)

        # Validate reconstruction
        self._reconstruct_from_template()

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
            # Preserve duration from extended detection; previously hardcoded to 1
            duration = int(item.get('duration', 1) or 1)
            if duration < 1:
                duration = 1
            entries.append((date, magnitude, anomaly_type, duration, shared))
            template_entries.append(
                {
                    'date': date.isoformat(),
                    'magnitude': magnitude,
                    'pattern': anomaly_type,
                    'duration': duration,
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
