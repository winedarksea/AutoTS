# -*- coding: utf-8 -*-
"""ReconstructionLoss for real-world datasets without component-level labels."""

import numpy as np
import pandas as pd
import copy
import warnings
from .base import FeatureDetectionLoss


class ReconstructionLoss(FeatureDetectionLoss):
    """
    Loss function tailored for real-world datasets lacking component-level labels.

    Focuses on reconstruction quality while discouraging overly complex trend fits and
    encouraging variance to be attributed to seasonality, holidays, anomalies, and level shifts.
    """

    DEFAULT_METRIC_WEIGHTS = {
        'reconstruction_loss': 0.5,
        'structural_loss': 1.0,
        'noise_whiteness_loss': 0.5,
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

        # Structural components (predictable signal)
        structural_sum = trend + level_shift + seasonality + holidays

        structural_loss = self._robust_structural_loss(observed_series, structural_sum)
        noise_whiteness = self._noise_whiteness_penalty(noise)

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
            'structural_loss': structural_loss,
            'noise_whiteness_loss': noise_whiteness,
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

    def _robust_structural_loss(self, observed, structural):
        """
        Huber-style loss that is robust to anomalies (outliers).
        Measures how well the structural components fit the 'bulk' of the data.
        """
        resid = (observed - structural).to_numpy(dtype=float)
        mask = np.isfinite(resid)
        if mask.sum() < 2:
            return 0.0
        resid = resid[mask]

        # Robust scale estimation (MAD)
        median = np.median(resid)
        abs_dev = np.abs(resid - median)
        mad = np.median(abs_dev)
        scale = mad * 1.4826  # Consistency with sigma for normal distribution
        if scale < 1e-9:
            scale = np.std(resid) + 1e-9

        # Huber Loss with delta = 1.5 * scale
        delta = 1.5 * scale
        error = np.abs(resid)
        is_small = error <= delta

        squared_loss = 0.5 * error[is_small] ** 2
        linear_loss = delta * (error[~is_small] - 0.5 * delta)

        loss = np.sum(squared_loss) + np.sum(linear_loss)
        return (loss / len(resid)) / (scale**2 + 1e-9)

    def _noise_whiteness_penalty(self, noise_series):
        """Penalize autocorrelation in the noise component."""
        values = noise_series.to_numpy(dtype=float)
        acf1 = self._autocorrelation(values, 1)
        return min(abs(acf1) * 2.0, 2.0)

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
                kurtosis_improvement = max(
                    0.0, (pre_kurtosis - post_kurtosis) / (pre_kurtosis + 1e-6)
                )

        # Blend std and kurtosis improvement
        combined_improvement = 0.7 * std_improvement + 0.3 * kurtosis_improvement

        if combined_improvement >= self.anomaly_improvement_target:
            return 0.0
        deficit = self.anomaly_improvement_target - combined_improvement
        return min(max(deficit, 0.0), 1.5)

    def _seasonality_shape_penalty(self, observed, trend, level_shift, seasonal_bundle):
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

        diff_wasserstein = np.mean(np.abs(det_diff_sorted - sea_diff_sorted)) / (
            diff_scale + 1e-6
        )

        # Energy ratio: seasonal should capture a meaningful portion of detrended energy
        detrended_energy = float(np.sum(detrended**2))
        residual_energy = float(np.sum((detrended - seasonal_arr) ** 2))
        if detrended_energy > 1e-6:
            energy_capture = 1.0 - (residual_energy / detrended_energy)
            # Penalize if capturing too little or negative (worse than nothing)
            energy_penalty = (
                max(0.0, 0.3 - energy_capture)
                if energy_capture >= 0
                else abs(energy_capture)
            )
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
        return (m4 / (std**4)) - 3.0

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
