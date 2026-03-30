# -*- coding: utf-8 -*-
"""TrendMixin for TimeSeriesFeatureDetector."""

import numpy as np
import pandas as pd
import copy
import warnings
from autots.tools.transform import (
    LevelShiftMagic,
    GeneralTransformer,
)
from autots.tools.changepoints import ChangepointDetector


class TrendMixin:
    """Mixin providing trend changepoint detection, level shift detection, and slope computation."""

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
            series_iqr = float(series.quantile(0.75) - series.quantile(0.25)) or 1e-9
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
