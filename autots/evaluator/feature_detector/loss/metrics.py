# -*- coding: utf-8 -*-
"""Distance and penalty metrics for feature detection loss."""

import numpy as np
import pandas as pd
import warnings


class LossMetricsMixin:
    """Mixin providing distance/penalty metric methods for loss calculation."""

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
    def _is_number(value):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

