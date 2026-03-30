# -*- coding: utf-8 -*-
"""RescalingMixin for TimeSeriesFeatureDetector."""

import numpy as np
import pandas as pd


class RescalingMixin:
    """Mixin providing methods to convert components from standardized to original scale."""

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
                        # Preserve duration so multi-day anomalies are not collapsed to 1
                        'duration': int(entry.get('duration', 1) or 1),
                        # Preserve end_date if present (from extended detection)
                        'end_date': (
                            pd.Timestamp(entry['end_date'])
                            if entry.get('end_date') is not None
                            else None
                        ),
                    }
                )
            rescaled[series_name] = converted
        return rescaled

    def _rescale_holiday_coefficients(self, coefficients):
        rescaled = {}
        for series_name, mapping in coefficients.items():
            scale = float(self.scale_series.get(series_name, 1.0))
            converted = {}
            for name, value in mapping.items():
                converted[name] = float(value * scale)
            rescaled[series_name] = converted
        return rescaled
