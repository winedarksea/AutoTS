# -*- coding: utf-8 -*-
"""Distance and penalty metrics for feature detection loss."""

import numpy as np
import pandas as pd
import warnings


class LossMetricsMixin:
    """Mixin providing distance/penalty metric methods for loss calculation."""

    def _chamfer_penalty(self, detected_dates, true_dates, cap=30.0, recall_weight=0.6):
        """
        Compute asymmetric log-Chamfer distance between two sets of dates.

        Uses log(1 + distance / tolerance) scoring for a continuous, uncapped
        gradient everywhere. Unlike Gaussian proximity, this never plateaus: the
        optimizer always receives a directional signal pulling distant false
        positives or misses toward the nearest true event location.

        Parameters
        ----------
        detected_dates : list
            Detected event dates.
        true_dates : list
            True event dates.
        cap : float
            Reference scale in days; tolerance = cap / 3.  Sets log normalization
            so that score = 1.0 when distance equals tolerance.  The penalty itself
            is uncapped and continues to grow beyond this reference.
        recall_weight : float
            Weight for recall direction (true->detected). Precision gets 1 - recall_weight.

        Returns
        -------
        float
            Non-negative value where 0 is perfect.  Scales beyond 1 for events
            separated by more than one tolerance unit, providing gradient everywhere.
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

        dists = np.abs(t_vals[:, None] - d_vals[None, :])  # (n_true, n_det)

        # Normalizing tolerance: log1p(1) = log(2), so score=1 when dist=tolerance
        tolerance = max(cap / 3.0, 1.0)
        _log2 = np.log(2.0)

        # Recall: for each true event, log-penalty for nearest detected
        t2d_dists = np.min(dists, axis=1)
        recall_score = float(np.mean(np.log1p(t2d_dists / tolerance) / _log2))

        # Precision: for each detected event, log-penalty for nearest true
        d2t_dists = np.min(dists, axis=0)
        precision_score = float(np.mean(np.log1p(d2t_dists / tolerance) / _log2))

        # Soft count-mismatch penalty
        count_ratio = abs(len(t_dates) - len(d_dates)) / (len(t_dates) + len(d_dates))
        count_penalty = 0.15 * count_ratio

        precision_weight = 1.0 - recall_weight
        return recall_weight * recall_score + precision_weight * precision_score + count_penalty

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
    def _component_spectral_penalty(detected, true):
        """
        Phase-invariant spectral loss comparing FFT magnitude spectra.

        Compares the absolute magnitudes of the real FFT of detected vs true
        component arrays, ignoring phase entirely. This rewards correct
        frequency content (the "shape" at the right periods) without
        penalizing small time-shifts that cause RMSE to spike.

        Works automatically for any data frequency: the FFT bins adapt to
        whatever sampling rate the data has (daily, monthly, hourly, etc.)
        so no hard-coded periods (7, 12, 365.25) are needed.

        Returns
        -------
        float
            Penalty between 0 (perfect spectral match) and 3.0 (poor match).
        """
        detected_arr = np.asarray(detected, dtype=float).ravel()
        true_arr = np.asarray(true, dtype=float).ravel()
        length = min(detected_arr.size, true_arr.size)
        if length < 4:
            return 0.5
        detected_arr = detected_arr[:length]
        true_arr = true_arr[:length]
        mask = np.isfinite(detected_arr) & np.isfinite(true_arr)
        if mask.sum() < 4:
            return 0.5
        detected_arr = detected_arr[mask]
        true_arr = true_arr[mask]

        # Compute real FFT magnitudes (discarding phase/imaginary component)
        true_spectrum = np.abs(np.fft.rfft(true_arr))
        detected_spectrum = np.abs(np.fft.rfft(detected_arr))

        # Normalize by sequence length for scale-consistent comparison
        n = true_arr.size
        true_spectrum = true_spectrum / (n + 1e-9)
        detected_spectrum = detected_spectrum / (n + 1e-9)

        # 1. Overall spectral MAE (normalized by true spectral energy)
        true_spectral_energy = float(np.sum(true_spectrum))
        if true_spectral_energy < 1e-9:
            # True seasonality has negligible spectral energy
            det_spectral_energy = float(np.sum(detected_spectrum))
            return min(det_spectral_energy * 10.0, 3.0) if det_spectral_energy > 1e-6 else 0.0

        spectral_mae = float(np.mean(np.abs(true_spectrum - detected_spectrum)))
        normalized_spectral_mae = spectral_mae / (true_spectral_energy / len(true_spectrum) + 1e-9)

        # 2. Peak frequency alignment: check if the dominant frequencies match
        # Identify top-k peaks in true spectrum (excluding DC at index 0)
        n_peaks = min(5, max(1, len(true_spectrum) // 10))
        if len(true_spectrum) > 1:
            spectrum_no_dc = true_spectrum[1:]
            det_spectrum_no_dc = detected_spectrum[1:]
            # Indices of the largest true peaks (offset by 1 for DC removal)
            top_true_indices = np.argsort(spectrum_no_dc)[-n_peaks:]
            # How well does the detected spectrum capture these specific peaks?
            peak_true_values = spectrum_no_dc[top_true_indices]
            peak_det_values = det_spectrum_no_dc[top_true_indices]
            peak_scale = float(np.mean(peak_true_values)) + 1e-9
            peak_mae = float(np.mean(np.abs(peak_true_values - peak_det_values))) / peak_scale
        else:
            peak_mae = 0.0

        # 3. Spectral correlation: overall shape similarity in frequency domain
        if len(true_spectrum) > 2:
            t_std = float(np.std(true_spectrum))
            d_std = float(np.std(detected_spectrum))
            if t_std > 1e-12 and d_std > 1e-12:
                corr = float(np.corrcoef(true_spectrum, detected_spectrum)[0, 1])
                if not np.isfinite(corr):
                    spectral_corr_penalty = 1.0
                else:
                    spectral_corr_penalty = 1.0 - max(0.0, corr)
            else:
                spectral_corr_penalty = 0.0 if (t_std < 1e-12 and d_std < 1e-12) else 1.0
        else:
            spectral_corr_penalty = 0.0

        combined = (
            0.35 * min(normalized_spectral_mae, 3.0)
            + 0.35 * min(peak_mae, 3.0)
            + 0.30 * min(spectral_corr_penalty, 3.0)
        )
        return min(combined, 3.0)

    @staticmethod
    def _component_profile_correlation(detected, true, date_index=None):
        """
        Compute periodic profile correlations between detected and true seasonality.

        Automatically detects the data frequency from the datetime index and
        builds appropriate aggregation profiles:
        - Sub-daily data (hourly, etc.): hour-of-day profile + day-of-week profile
        - Daily data: day-of-week profile + month-of-year profile
        - Weekly data: week-of-year profile
        - Monthly data: month-of-year profile

        Each profile is the mean value at each position in the cycle. The
        correlation between detected and true profiles measures whether the
        shape (peaks, troughs, relative ordering) is preserved.

        Parameters
        ----------
        detected : array-like
            Detected seasonality component values.
        true : array-like
            True seasonality component values.
        date_index : pd.DatetimeIndex, optional
            Datetime index for the time series. When None, falls back to
            a simple positional modular profile using common cycle lengths.

        Returns
        -------
        float
            Penalty between 0 (perfect profile correlation) and 3.0 (poor match).
        """
        detected_arr = np.asarray(detected, dtype=float).ravel()
        true_arr = np.asarray(true, dtype=float).ravel()
        length = min(detected_arr.size, true_arr.size)
        if length < 4:
            return 0.5
        detected_arr = detected_arr[:length]
        true_arr = true_arr[:length]
        mask = np.isfinite(detected_arr) & np.isfinite(true_arr)
        if mask.sum() < 4:
            return 0.5
        detected_arr = detected_arr[mask]
        true_arr = true_arr[mask]

        def _profile_corr(values_det, values_true, group_keys):
            """Compute mean profiles by group key and return 1 - correlation."""
            unique_keys = np.unique(group_keys)
            if len(unique_keys) < 3:
                return None  # Too few groups for meaningful profile
            det_means = np.array([np.mean(values_det[group_keys == k]) for k in unique_keys])
            true_means = np.array([np.mean(values_true[group_keys == k]) for k in unique_keys])
            d_std = float(np.std(det_means))
            t_std = float(np.std(true_means))
            if t_std < 1e-12:
                # True profile is flat; penalty proportional to detected variation
                return min(d_std * 5.0, 1.0) if d_std > 1e-9 else 0.0
            if d_std < 1e-12:
                # Detected profile is flat but true has variation
                return 1.0
            corr = float(np.corrcoef(det_means, true_means)[0, 1])
            if not np.isfinite(corr):
                return 1.0
            return max(0.0, 1.0 - corr)

        penalties = []

        if date_index is not None and isinstance(date_index, pd.DatetimeIndex):
            idx = date_index[:length]
            # Trim to mask
            if mask.sum() < length:
                idx = idx[mask]

            # Infer frequency from median timedelta
            if len(idx) > 1:
                median_delta = np.median(np.diff(idx.values).astype('timedelta64[s]').astype(float))
            else:
                median_delta = 86400.0  # Default to daily

            if median_delta < 3600 * 12:  # Sub-daily (e.g., hourly)
                hour_keys = np.array(idx.hour)
                p = _profile_corr(detected_arr, true_arr, hour_keys)
                if p is not None:
                    penalties.append(p)
                dow_keys = np.array(idx.dayofweek)
                p = _profile_corr(detected_arr, true_arr, dow_keys)
                if p is not None:
                    penalties.append(p)
            elif median_delta < 86400 * 3:  # Daily
                dow_keys = np.array(idx.dayofweek)
                p = _profile_corr(detected_arr, true_arr, dow_keys)
                if p is not None:
                    penalties.append(p)
                month_keys = np.array(idx.month)
                p = _profile_corr(detected_arr, true_arr, month_keys)
                if p is not None:
                    penalties.append(p)
            elif median_delta < 86400 * 10:  # Weekly
                woy_keys = np.array(idx.isocalendar().week.values, dtype=int)
                p = _profile_corr(detected_arr, true_arr, woy_keys)
                if p is not None:
                    penalties.append(p)
            else:  # Monthly or longer
                month_keys = np.array(idx.month)
                p = _profile_corr(detected_arr, true_arr, month_keys)
                if p is not None:
                    penalties.append(p)
        else:
            # Fallback: positional modular profiles for common cycle lengths
            n = detected_arr.size
            for period in [7, 12, 52]:
                if n >= period * 2:  # Need at least 2 full cycles
                    pos_keys = np.arange(n) % period
                    p = _profile_corr(detected_arr, true_arr, pos_keys)
                    if p is not None:
                        penalties.append(p)

        if not penalties:
            return 0.5

        # Average penalty across all applicable profiles
        avg_penalty = float(np.mean(penalties))
        return min(avg_penalty, 3.0)

    @staticmethod
    def _is_number(value):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

