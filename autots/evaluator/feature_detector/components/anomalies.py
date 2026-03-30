# -*- coding: utf-8 -*-
"""AnomalyMixin for TimeSeriesFeatureDetector."""

import copy
import numpy as np
import pandas as pd
import warnings
from autots.tools.transform import AnomalyRemoval


class AnomalyMixin:
    """Mixin providing anomaly detection, noise analysis, and regime changepoint detection."""

    def _build_anomaly_params(self, holiday_dates=None):
        """Build call-scoped anomaly parameters without mutating detector defaults."""
        params = copy.deepcopy(self.anomaly_params)
        params['output'] = self.detection_mode
        params.pop('holiday_dates', None)
        if holiday_dates:
            params['holiday_dates'] = list(holiday_dates)
            params.setdefault('holiday_proximity_days', 2)
        return params

    def _detect_anomalies(
        self, residual_df, anomaly_params=None, detector_attr='anomaly_detector'
    ):
        """Detect anomalies using AnomalyRemoval."""
        active_params = anomaly_params or self.anomaly_params
        detector = AnomalyRemoval(**active_params)
        cleaned = detector.fit_transform(residual_df)
        setattr(self, detector_attr, detector)
        anomalies = {}

        # Handle both multivariate and univariate outputs
        if self.detection_mode == 'univariate':
            # Univariate mode: single column of anomaly flags for all series
            # anomalies will be a single column, apply to all series
            anomaly_col = detector.anomalies.iloc[:, 0]
            mask = anomaly_col == -1
            anomaly_dates = residual_df.index[mask].tolist()

            # Create records for each anomalous date
            records = []
            for date in anomaly_dates:
                # For univariate, we could use the max magnitude across series or mean
                magnitudes = residual_df.loc[date, :].values
                magnitude = float(np.nanmean(magnitudes))
                score = None
                if hasattr(detector, 'scores') and not detector.scores.empty:
                    try:
                        score = float(detector.scores.loc[date].iloc[0])
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
                if col not in detector.anomalies.columns:
                    anomalies[col] = []
                    continue

                mask = detector.anomalies[col] == -1
                if mask.sum() == 0:
                    anomalies[col] = []
                    continue

                anomaly_dates = residual_df.index[mask].tolist()
                records = []
                for date in anomaly_dates:
                    magnitude = residual_df.at[date, col]
                    score = None
                    if hasattr(detector, 'scores') and not detector.scores.empty:
                        try:
                            score = float(detector.scores.loc[date, col])
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
        n_outliers = np.sum(
            post_deviations > max(anomaly_mag * 0.4, baseline_std * 2.5)
        )
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
                    try:
                        decay_slope = np.polyfit(
                            np.arange(len(extended_devs)), extended_devs, 1
                        )[0]
                    except np.linalg.LinAlgError:
                        decay_slope = 0
                    if decay_slope < 0 and abs(decay_slope) < anomaly_mag * 0.05:
                        return 'slope_reversion'

        # Default: point outlier
        return 'point_outlier'

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
                    df_without_anomalies_scaled = transformed.reindex(
                        index=df_work.index, columns=df_work.columns
                    ).astype(float)
            except Exception as exc:
                warnings.warn(
                    f"Anomaly transform failed during noise analysis; using fallback interpolation. {exc}",
                    RuntimeWarning,
                )

        # Fallback: if transform did not materially change anything while anomaly
        # records exist, apply a conservative neighbor interpolation.
        if df_without_anomalies_scaled.equals(df_work) and getattr(
            self, '_anomaly_records_temp', None
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
        rolling_std = series.rolling(
            window=window, center=True, min_periods=min_periods
        ).std()
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

    @staticmethod
    def _lag1_autocorrelation(values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        mask = np.isfinite(arr)
        arr = arr[mask]
        if arr.size < 3:
            return 0.0
        return AnomalyMixin._safe_correlation(arr[:-1], arr[1:])

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
