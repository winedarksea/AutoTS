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
        (
            changepoints,
            trend_component_scaled,
            level_shift_component_valid_scaled,
        ) = self._detect_trend_changepoints(
            trend_input,
            trend_and_shift_target=residual_for_level_shifts,
            validated_level_shifts=validated_level_shifts,
            level_shift_component=level_shift_component_valid_scaled,
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
        # None = adaptive (tightened to each series' noise); explicit values are honored.
        rel_thresh = params.get('relative_threshold')
        abs_thresh = params.get('absolute_threshold')

        # LevelShiftMagic.lvlshft is a reverse cumsum (non-zero before a shift, zero
        # after), and candidate dates are the first post-shift date. Removing a shift
        # therefore adjusts the points before it.
        validated_component = lvlshft.copy()
        validated = {}

        for col in residual_df.columns:
            series = residual_df[col]
            series_std = float(series.std()) or 1e-9
            series_iqr = float(series.quantile(0.75) - series.quantile(0.25)) or 1e-9
            series_median = float(np.nanmedian(series.to_numpy(dtype=float)))
            # Guard against near-zero medians which can explode relative thresholds.
            robust_scale = max(abs(series_median), series_iqr, series_std, 1e-6)
            if abs_thresh is None:
                adaptive_abs_thresh = min(0.5, series_std * 0.3)
            else:
                adaptive_abs_thresh = float(abs_thresh)
            if rel_thresh is None:
                dynamic_rel = 0.05 + 0.5 * (series_iqr / robust_scale)
                dynamic_rel = float(np.clip(dynamic_rel, 0.05, 0.75))
                adaptive_rel_thresh = min(0.1, dynamic_rel)
            else:
                adaptive_rel_thresh = float(rel_thresh)
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
                    validated_component.loc[
                        validated_component.index < date, col
                    ] += magnitude
                    continue

                before = float(np.nanmedian(left_window))
                after = float(np.nanmedian(right_window))
                change = after - before
                abs_change = abs(change)
                # The residual is roughly zero-centred, so dividing by the local level
                # alone makes nearly every change "relatively" large.
                rel_change = abs_change / max(abs(before), robust_scale)

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
                    validated_component.loc[
                        validated_component.index < date, col
                    ] += magnitude
            validated[col] = entries
        return validated_component, validated

    def _detect_trend_changepoints(
        self,
        trend_input,
        trend_and_shift_target=None,
        validated_level_shifts=None,
        level_shift_component=None,
    ):
        """Detect trend changepoints on ``trend_input`` and fit the trend.

        When validated level shifts are given, step magnitudes are re-estimated
        jointly with the hinge trend on ``trend_and_shift_target`` (the residual
        before level-shift removal). LevelShiftMagic's own magnitudes come from
        long rolling windows, so any slope over that window leaks into them.
        Returns (changepoints, trend_component, level_shift_component); entries in
        ``validated_level_shifts`` get their 'magnitude' updated in place.
        """
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

        validated_level_shifts = validated_level_shifts or {}
        joint_shift_fit = trend_and_shift_target is not None and any(
            validated_level_shifts.get(col) for col in series_names
        )
        fit_df = (
            trend_and_shift_target.reindex(columns=series_names).ffill().bfill()
            if joint_shift_fit
            else safe_df
        )
        values = fit_df.to_numpy(dtype=float, copy=False)
        time_index = np.arange(n_samples, dtype=float)
        trend_matrix = np.full((n_samples, n_series), np.nan)
        if level_shift_component is not None:
            level_shift_matrix = (
                level_shift_component.reindex(columns=series_names)
                .to_numpy(dtype=float)
                .copy()
            )
        else:
            level_shift_matrix = np.zeros((n_samples, n_series))

        # Fit the continuous piecewise-linear (hinge) model jointly. Fitting each
        # segment's slope separately and chaining them from the first intercept lets
        # per-segment mismatches accumulate into a growing level error. Series sharing
        # changepoint and shift positions share a design matrix and are solved together.
        finite_cols = np.isfinite(values).all(axis=0)
        groups = {}
        for j, col in enumerate(series_names):
            if not finite_cols[j]:
                continue
            indices = changepoint_indices.get(col, np.array([], dtype=int))
            shift_positions = ()
            if joint_shift_fit:
                shift_positions = tuple(
                    sorted(
                        {
                            int(p)
                            for p in self.date_index.searchsorted(
                                pd.DatetimeIndex(
                                    [
                                        entry['date']
                                        for entry in validated_level_shifts.get(col, [])
                                    ]
                                )
                            )
                            if 0 < p < n_samples
                        }
                    )
                )
            cp_key = tuple(int(cp) for cp in indices)
            groups.setdefault((cp_key, shift_positions), []).append(j)
        for (cp_key, shift_positions), col_positions in groups.items():
            hinge_design = np.column_stack(
                [np.ones(n_samples), time_index]
                + [np.maximum(0.0, time_index - float(cp)) for cp in cp_key]
            )
            # LevelShiftMagic convention: -magnitude before the shift, 0 after.
            step_design = (
                np.column_stack(
                    [-(time_index < p).astype(float) for p in shift_positions]
                )
                if shift_positions
                else np.zeros((n_samples, 0))
            )
            design = np.column_stack([hinge_design, step_design])
            beta, _, _, _ = np.linalg.lstsq(
                design, values[:, col_positions], rcond=None
            )
            n_hinge = hinge_design.shape[1]
            trend_matrix[:, col_positions] = hinge_design @ beta[:n_hinge]
            if shift_positions:
                step_magnitudes = beta[n_hinge:]
                level_shift_matrix[:, col_positions] = step_design @ step_magnitudes
                for k, j in enumerate(col_positions):
                    magnitude_by_position = dict(
                        zip(shift_positions, step_magnitudes[:, k])
                    )
                    for entry in validated_level_shifts.get(series_names[j], []):
                        position = int(self.date_index.searchsorted(entry['date']))
                        if position in magnitude_by_position:
                            entry['magnitude'] = float(magnitude_by_position[position])

        trend_component = pd.DataFrame(
            trend_matrix, index=self.date_index, columns=series_names
        )
        level_shift_out = pd.DataFrame(
            level_shift_matrix, index=self.date_index, columns=series_names
        )
        return changepoints, trend_component, level_shift_out

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
