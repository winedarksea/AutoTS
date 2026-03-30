# -*- coding: utf-8 -*-
"""Component-specific loss evaluators."""

import numpy as np
import pandas as pd
import warnings


class LossEvaluatorsMixin:
    """Mixin providing component-specific loss evaluation methods."""

    def _trend_loss(self, detected_cp, true_cp, detected_components, true_components):
        if not true_cp and not detected_cp:
            return 0.0
        if not true_cp:
            return 0.25 * len(detected_cp)

        detected_entries = [self._parse_trend_event(event) for event in detected_cp]
        true_entries = [self._parse_trend_event(event) for event in true_cp]
        n_true = len(true_entries)

        # Add Chamfer penalty to guide optimizer when points are far apart
        chamfer_loss = self._chamfer_penalty(
            [x[0] for x in detected_entries],
            [x[0] for x in true_entries],
            cap=self.changepoint_tolerance_days * 5,
        )

        n_detected = len(detected_entries)
        unmatched_detected = set(range(n_detected))
        focal_tversky = self._focal_tversky_changepoint_penalty(
            detected_entries,
            true_entries,
            sigma=max(self.changepoint_tolerance_days, 1),
            alpha=0.28,
            beta=0.72,
            gamma=2.0,
        )

        positive_true_magnitudes = [
            entry[3] for entry in true_entries if np.isfinite(entry[3]) and entry[3] > 0
        ]
        default_magnitude_scale = (
            float(np.median(positive_true_magnitudes))
            if positive_true_magnitudes
            else 0.0
        )

        def _subtlety_weight(prior_slope, post_slope, magnitude):
            sign_change = (prior_slope * post_slope) < 0
            if sign_change:
                return 1.0
            if magnitude > 0:
                scale = (
                    default_magnitude_scale
                    if default_magnitude_scale > 0
                    else magnitude
                )
                relative = magnitude / (scale + 1e-9)
                # Bounded via tanh so importance stays in [0.2, 0.9].  An
                # unbounded linear scale amplifies the miss/no-detection penalty
                # inversion, causing the optimizer to collapse to zero detections.
                return 0.2 + 0.7 * np.tanh(relative)
            return 0.2

        sigma_days = max(self.changepoint_tolerance_days, 1) / 1.5
        matched_true = 0
        loss = 0.0

        for true_date, true_prior, true_post, true_mag in true_entries:
            importance = _subtlety_weight(true_prior, true_post, true_mag)
            best_idx = None
            best_dist = None
            for idx in unmatched_detected:
                det_date = detected_entries[idx][0]
                dist = abs((det_date - true_date).days)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if (
                best_idx is not None
                and best_dist is not None
                and best_dist <= self.changepoint_tolerance_days
            ):
                _, det_prior, det_post, _ = detected_entries[best_idx]
                unmatched_detected.discard(best_idx)
                matched_true += 1

                distance_score = np.exp(-0.5 * (best_dist / (sigma_days + 1e-9)) ** 2)
                distance_penalty = (1.0 - distance_score) * 0.5

                slope_change_true = true_post - true_prior
                slope_change_detected = det_post - det_prior
                slope_denom = max(abs(slope_change_true), 0.05, 1e-3)
                slope_error = (
                    abs(slope_change_detected - slope_change_true) / slope_denom
                )
                slope_penalty = min(slope_error, 2.0) * 0.3

                sign_penalty = (
                    0.2 if (slope_change_true * slope_change_detected) < 0 else 0.0
                )
                loss += (distance_penalty + slope_penalty + sign_penalty) * importance
            else:
                # Absolute magnitude penalty: missing a large slope change is
                # worse than missing a small one regardless of normalization — the
                # old implementation baked raw size directly into the miss loss.
                abs_mag_penalty = (
                    min(abs(true_mag) / (default_magnitude_scale + 1e-9), 2.0) * 0.5
                )
                if best_dist is not None:
                    loss += (
                        1.5
                        * np.tanh(
                            np.log1p(
                                best_dist / (self.changepoint_tolerance_days + 1e-9)
                            )
                        )
                        + abs_mag_penalty
                    ) * importance
                else:
                    loss += (1.5 + abs_mag_penalty) * importance

        false_positives = len(unmatched_detected)
        loss += 0.2 * false_positives

        count_diff = abs(n_detected - n_true)
        count_penalty = min(count_diff / max(n_true, 1), 2.0)
        loss += 0.6 * count_penalty

        recall = matched_true / (n_true + 1e-9)
        precision = (
            (n_detected - false_positives) / (n_detected + 1e-9) if n_detected else 1.0
        )
        # β=2.0: strongly recall-biased — over-detecting is preferred over missing
        f_beta = (
            (1.0 + 2.0**2) * (precision * recall) / (2.0**2 * precision + recall + 1e-9)
        )
        loss += (1.0 - f_beta) * 2.4
        loss += focal_tversky * max(n_true, 1) * 1.25

        # Apply trend component or complexity penalty based on mode
        trend_detected_series = detected_components.get('trend')

        if self.trend_component_penalty == 'component':
            if (
                trend_detected_series is not None
                and 'trend' in true_components
                and true_components.get('trend') is not None
            ):
                loss += self._component_rmse_penalty(
                    trend_detected_series,
                    true_components['trend'],
                )
        elif self.trend_component_penalty == 'complexity':
            if trend_detected_series is not None and self.trend_complexity_weight > 0:
                complexity_penalty = self._trend_complexity_penalty(
                    trend_detected_series
                )
                loss += self.trend_complexity_weight * complexity_penalty

        return loss + 0.75 * chamfer_loss

    def _trend_complexity_penalty(self, trend_values):
        if trend_values is None:
            return 0.0
        arr = np.atleast_1d(np.asarray(trend_values, dtype=float))
        mask = np.isfinite(arr)
        if mask.sum() < 5:
            return 0.0
        arr = arr[mask]
        if arr.size < 5:
            return 0.0

        series = pd.Series(arr)
        window = min(len(series), max(3, self.trend_complexity_window))
        smooth = series.rolling(window=window, center=True, min_periods=1).median()
        residual = series - smooth

        smooth_values = smooth.to_numpy(dtype=float)
        residual_values = residual.to_numpy(dtype=float)

        smooth_scale = np.nanstd(smooth_values)
        if smooth_scale < 1e-6 or not np.isfinite(smooth_scale):
            smooth_scale = np.nanmean(np.abs(smooth_values)) + 1e-6

        if smooth_scale <= 0 or not np.isfinite(smooth_scale):
            return 0.0

        residual_sq = residual_values**2
        if residual_sq.size > 0:
            cutoff = np.nanpercentile(residual_sq, 90)
            if np.isfinite(cutoff):
                residual_sq = np.clip(residual_sq, None, cutoff)
        residual_scale = np.sqrt(np.nanmean(residual_sq)) if residual_sq.size else 0.0

        if not np.isfinite(residual_scale):
            return 0.0

        penalty = residual_scale / (smooth_scale + 1e-6)
        return min(max(penalty, 0.0), 2.5)

    def _level_shift_loss(self, detected_ls, true_ls, detected_cp):
        if not true_ls and not detected_ls:
            return 0.0
        if not true_ls:
            return 0.2 * len(detected_ls)
        detected_entries = [
            self._parse_level_shift_event(event) for event in detected_ls
        ]
        true_entries = [self._parse_level_shift_event(event) for event in true_ls]

        chamfer_loss = self._chamfer_penalty(
            [x[0] for x in detected_entries],
            [x[0] for x in true_entries],
            cap=self.level_shift_tolerance_days * 5,
        )

        changepoint_dates = [self._parse_trend_event(event)[0] for event in detected_cp]
        focal_tversky = self._focal_tversky_changepoint_penalty(
            detected_entries,
            true_entries,
            sigma=max(self.level_shift_tolerance_days, 1),
            alpha=0.32,
            beta=0.68,
            gamma=2.0,
        )
        n_true = len(true_entries)
        n_detected = len(detected_entries)
        unmatched_detected = set(range(n_detected))
        matched_true = 0
        loss = 0.0

        magnitude_scale = max([abs(mag) for _, mag in true_entries] or [1.0])

        for true_date, true_mag in true_entries:
            relative_mag = abs(true_mag) / (magnitude_scale + 1e-9)
            importance = 0.3 + 0.7 * min(relative_mag, 1.0)

            best_idx = None
            best_dist = None
            for idx in unmatched_detected:
                det_date, _ = detected_entries[idx]
                dist = abs((det_date - true_date).days)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if (
                best_idx is not None
                and best_dist is not None
                and best_dist <= self.level_shift_tolerance_days
            ):
                _, det_mag = detected_entries[best_idx]
                distance_penalty = best_dist / (self.level_shift_tolerance_days + 1e-9)
                magnitude_penalty = abs(det_mag - true_mag) / (abs(true_mag) + 1e-6)
                loss += (
                    0.5 * distance_penalty + 0.5 * min(magnitude_penalty, 2.0)
                ) * importance
                unmatched_detected.discard(best_idx)
                matched_true += 1
            else:
                prox_cp = any(
                    abs((cp_date - true_date).days) <= self.changepoint_tolerance_days
                    for cp_date in changepoint_dates
                )
                if prox_cp:
                    loss += 0.5 * importance
                else:
                    # Absolute magnitude penalty: missing a large level shift
                    # is worse than missing a small one — bake the raw size in.
                    abs_mag_penalty = (
                        min(abs(true_mag) / (magnitude_scale + 1e-9), 2.0) * 0.5
                    )
                    if best_dist is not None:
                        loss += (
                            1.2
                            * np.tanh(
                                np.log1p(
                                    best_dist / (self.level_shift_tolerance_days + 1e-9)
                                )
                            )
                            + abs_mag_penalty
                        ) * importance
                    else:
                        loss += (1.2 + abs_mag_penalty) * importance

        false_positives = len(unmatched_detected)
        loss += 0.15 * false_positives

        count_diff = abs(n_detected - n_true)
        loss += 0.45 * min(count_diff / max(n_true, 1), 2.0)

        recall = matched_true / (n_true + 1e-9)
        precision = (
            (n_detected - false_positives) / (n_detected + 1e-9) if n_detected else 1.0
        )
        # β=2.0: strongly recall-biased — over-detecting is preferred over missing
        f_beta = (
            (1.0 + 2.0**2) * (precision * recall) / (2.0**2 * precision + recall + 1e-9)
        )
        loss += (1.0 - f_beta) * 1.7
        loss += focal_tversky * max(n_true, 1) * 0.8
        return loss + 0.6 * chamfer_loss

    # Anomaly type categories for type-aware penalty logic
    _POINT_TYPES = frozenset({'point_outlier', 'spike'})
    _EXTENDED_TYPES = frozenset(
        {
            'slope_reversion',
            'transient_change',
            'impulse_decay',
            'linear_decay',
            'noisy_burst',
        }
    )

    def _anomaly_loss(
        self,
        detected_anom,
        true_anom,
        detected_cp=None,
        detected_ls=None,
        true_ls=None,
    ):
        """
        Compute anomaly detection loss with type-aware penalties.

        Parameters
        ----------
        detected_anom : list
            Detected anomaly events.
        true_anom : list
            True anomaly events from synthetic labels.
        detected_cp : list, optional
            Detected trend changepoints (for partial credit on extended misses).
        detected_ls : list, optional
            Detected level shifts (for partial credit on extended misses).
        true_ls : list, optional
            Ground-truth level shifts.  Detected anomalies that overlap with
            true level shift dates are penalised so the optimizer learns that
            structural breaks belong to the level-shift detector.
        """
        if not true_anom:
            return 0.3 * len(detected_anom)

        detected_entries = [self._parse_anomaly_event(event) for event in detected_anom]
        true_entries = [self._parse_anomaly_event(event) for event in true_anom]

        # Soft F1 provides smooth gradient for optimizer even when detections
        # are slightly outside hard tolerance boundaries
        soft_f1_result = self._soft_f1_anomaly(detected_entries, true_entries)
        soft_f1_loss = 1.0 - soft_f1_result['soft_f1']

        # Pre-parse changepoints and level shifts for partial-credit checks
        cp_dates = []
        for cp_ev in detected_cp or []:
            try:
                if isinstance(cp_ev, dict):
                    cp_dates.append(pd.Timestamp(cp_ev.get('date')))
                elif isinstance(cp_ev, (tuple, list)) and cp_ev:
                    cp_dates.append(pd.Timestamp(cp_ev[0]))
                else:
                    cp_dates.append(pd.Timestamp(cp_ev))
            except Exception:
                pass

        ls_dates = []
        for ls_ev in detected_ls or []:
            try:
                if isinstance(ls_ev, dict):
                    ls_dates.append(pd.Timestamp(ls_ev.get('date')))
                elif isinstance(ls_ev, (tuple, list)) and ls_ev:
                    ls_dates.append(pd.Timestamp(ls_ev[0]))
                else:
                    ls_dates.append(pd.Timestamp(ls_ev))
            except Exception:
                pass

        used_detected = set()
        loss = 0.0
        finite_true_mags = [
            abs(mag)
            for _, mag, _, _ in true_entries
            if np.isfinite(mag) and abs(mag) > 0
        ]
        magnitude_scale = max(finite_true_mags or [1.0])

        for true_event in true_entries:
            true_date, true_mag, true_type, true_duration = true_event
            true_mag_safe = true_mag if np.isfinite(true_mag) else 0.0

            is_extended = true_type in self._EXTENDED_TYPES

            # For extended types use a wider date tolerance proportional to duration
            if is_extended:
                effective_tol = max(
                    self.anomaly_tolerance_days,
                    min(7, true_duration // 3),
                )
            else:
                effective_tol = self.anomaly_tolerance_days

            best_idx = None
            best_dist = None
            for idx, det_event in enumerate(detected_entries):
                if idx in used_detected:
                    continue
                det_date, *_ = det_event
                dist = abs((det_date - true_date).days)
                # For extended anomalies: also match if detected date falls
                # within the true event window [true_date, true_date+duration]
                if is_extended and true_duration > 1:
                    window_end = true_date + pd.Timedelta(days=true_duration - 1)
                    if true_date <= det_date <= window_end:
                        dist = 0
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if (
                best_idx is not None
                and best_dist is not None
                and best_dist <= effective_tol
            ):
                det_event = detected_entries[best_idx]
                _, det_mag, det_type, det_duration = det_event
                det_mag_safe = det_mag if np.isfinite(det_mag) else 0.0

                mag_pen = abs(det_mag_safe - true_mag_safe) / (
                    abs(true_mag_safe) + 1e-6
                )

                if is_extended:
                    # Softer type mismatch for extended types (hard to classify)
                    type_pen = 0.0 if det_type == true_type else 0.2
                    # Duration coverage: penalise if detected duration << true
                    duration_coverage = min(det_duration / max(true_duration, 1), 2.0)
                    duration_penalty = max(0.0, 1.0 - duration_coverage) * 0.4
                else:
                    # Point types: stricter type matching
                    type_pen = 0.0 if det_type == true_type else 0.3
                    duration_penalty = 0.2 * min(
                        abs(det_duration - true_duration) / (true_duration + 1e-6),
                        2.0,
                    )

                loss += 0.5 * min(mag_pen, 2.0) + 0.3 * type_pen + duration_penalty
                used_detected.add(best_idx)

            else:
                # Miss penalty — type-weighted
                relative_mag = abs(true_mag_safe) / (magnitude_scale + 1e-9)

                if true_type in self._POINT_TYPES:
                    # Point types: highest priority — keep original aggressive penalties
                    if relative_mag > 0.5:
                        miss_penalty = 1.5
                    elif relative_mag > 0.2:
                        miss_penalty = 0.6
                    else:
                        miss_penalty = 0.15
                else:
                    # Extended types: lower base penalty (harder to detect),
                    # scaled by duration (longer events penalised more)
                    duration_factor = min(2.0, 1.0 + true_duration / 21.0)
                    if relative_mag > 0.3:
                        miss_penalty = 0.9 * duration_factor
                    else:
                        miss_penalty = 0.3 * duration_factor

                    # Partial credit: if near a detected changepoint or level
                    # shift, the extended anomaly was approximately identified
                    # by another detector — reduce miss penalty by 50 %.
                    near_cp = any(
                        abs((cp_d - true_date).days) <= self.changepoint_tolerance_days
                        for cp_d in cp_dates
                    )
                    near_ls = any(
                        abs((ls_d - true_date).days) <= self.level_shift_tolerance_days
                        for ls_d in ls_dates
                    )
                    if near_cp or near_ls:
                        miss_penalty *= 0.5

                loss += miss_penalty

        false_positives = len(detected_entries) - len(used_detected)
        if false_positives > 0:
            fp_penalty = (0.15 * false_positives) + (0.1 * np.sqrt(false_positives))
            loss += fp_penalty

        # Cross-penalty: anomalies detected near true (or detected) level-shift
        # dates signal that the anomaly detector is absorbing structural breaks
        # that the level-shift detector should own.  Applying a small per-event
        # cross-penalty nudges the optimizer away from that configuration.
        true_ls_dates = []
        for ls_ev in true_ls or []:
            try:
                if isinstance(ls_ev, dict):
                    true_ls_dates.append(pd.Timestamp(ls_ev.get('date')))
                elif isinstance(ls_ev, (tuple, list)) and ls_ev:
                    true_ls_dates.append(pd.Timestamp(ls_ev[0]))
                else:
                    true_ls_dates.append(pd.Timestamp(ls_ev))
            except Exception:
                pass
        all_ls_dates = true_ls_dates + ls_dates
        if all_ls_dates:
            ls_overlap_penalty = 0.0
            for det_entry in detected_entries:
                det_date = det_entry[0]
                if any(
                    abs((det_date - ls_d).days) <= self.level_shift_tolerance_days
                    for ls_d in all_ls_dates
                ):
                    ls_overlap_penalty += 0.015  # keep very small extra penalty here
            loss += ls_overlap_penalty

        matches = len(used_detected)
        precision = matches / max(len(detected_entries), 1)
        recall = matches / max(len(true_entries), 1)
        beta = 2.0
        beta_sq = beta**2
        f2 = (
            (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-9)
        )
        precision_floor_penalty = max(0.0, 0.7 - precision)

        # Blend hard-match detail loss with strong recall-oriented penalties.
        return (
            0.45 * loss
            + 0.55 * soft_f1_loss * max(len(true_entries), 1)
            + (1.0 - f2) * 1.8
            + 0.5 * precision_floor_penalty
        )

    def _holiday_event_loss(self, detected_holidays, true_holidays, detected_anomalies):
        if not true_holidays:
            return 0.30 * len(detected_holidays)
        detected_dates = [pd.Timestamp(dt) for dt in detected_holidays]
        true_dates = [pd.Timestamp(dt) for dt in true_holidays]
        anomaly_dates = [
            self._parse_anomaly_event(event)[0] for event in detected_anomalies
        ]
        matched_true = 0
        unmatched_detected = set(range(len(detected_dates)))
        anomaly_credit = 0.0
        for true_date in true_dates:
            best_idx = None
            best_dist = None
            for idx in unmatched_detected:
                dist = abs((detected_dates[idx] - true_date).days)
                if best_dist is None or dist < best_dist:
                    best_idx = idx
                    best_dist = dist
            if (
                best_idx is not None
                and best_dist is not None
                and best_dist <= self.holiday_tolerance_days
            ):
                matched_true += 1
                unmatched_detected.discard(best_idx)
                continue
            if any(
                abs(anomaly_date - true_date) <= self._anomaly_tolerance
                for anomaly_date in anomaly_dates
            ):
                anomaly_credit += self.holiday_over_anomaly_bonus

        false_positives = len(unmatched_detected)
        precision = matched_true / max(len(detected_dates), 1)
        recall = matched_true / max(len(true_dates), 1)
        # F2 (beta=2) is recall-biased: missing a holiday costs more than a false
        # positive.  holiday_recall_loss provides additional zero-detection gradient
        # so this function only needs a balanced quality signal.
        beta_sq = 4.0
        f2 = (
            (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-9)
        )
        # Absolute FP term deters arbitrary over-detection independently of the
        # count ratio already captured by precision.
        fp_penalty = 0.18 * false_positives
        return max(0.0, (1.0 - f2) * 2.5 + fp_penalty - anomaly_credit)

    def _holiday_impact_loss(self, detected_impacts, true_impacts):
        if not true_impacts:
            return 0.1 * len(detected_impacts)  # Slight increase for FP penalty
        detected = self._normalize_holiday_dict(detected_impacts)
        true = self._normalize_holiday_dict(true_impacts)
        loss = 0.0
        for date, true_value in true.items():
            det_value = detected.get(date, None)
            if det_value is None:
                loss += 0.5 + abs(true_value) * 0.35
            else:
                penalty = abs(det_value - true_value) / (abs(true_value) + 1e-6)
                if abs(true_value) > 1e-6:
                    relative_mag = abs(det_value) / (abs(true_value) + 1e-6)
                    if relative_mag < 0.3:
                        penalty *= 1.6
                    elif relative_mag < 0.5:
                        penalty *= 1.3
                    elif relative_mag < 0.7:
                        penalty *= 1.15
                loss += min(penalty, 2.5)
        extras = len([date for date in detected if date not in true])
        loss += 0.08 * extras
        return loss

    def _holiday_splash_loss(self, detected_impacts, detected_anomalies, true_splash):
        if not true_splash:
            return 0.0
        detected = self._normalize_holiday_dict(detected_impacts)
        anomaly_dates = [
            self._parse_anomaly_event(event)[0] for event in detected_anomalies
        ]
        loss = 0.0
        for date, magnitude in self._normalize_holiday_dict(true_splash).items():
            found = date in detected or any(
                abs(date - anomaly) <= self._anomaly_tolerance
                for anomaly in anomaly_dates
            )
            if not found:
                loss += 0.4 + 0.3 * min(abs(magnitude), 2.0)
        return loss

    def _holiday_recall_loss(self, detected_holidays, true_holidays):
        """
        Separate recall-focused loss for holiday detection.

        Heavily penalizes configurations that detect zero holidays when the truth
        has many, which encourages the optimizer to explore holiday-friendly params.
        """
        if not true_holidays:
            return 0.0

        n_true = len(true_holidays)
        if not detected_holidays:
            return min((0.7 + 0.55 * self.validation_strictness) + 0.12 * n_true, 3.0)

        detected_dates = {pd.Timestamp(dt) for dt in detected_holidays}
        true_dates = [pd.Timestamp(dt) for dt in true_holidays]

        matches = sum(
            1
            for td in true_dates
            if any(abs(td - dd) <= self._holiday_tolerance for dd in detected_dates)
        )

        recall = matches / n_true

        recall_penalty_scale = max(self.validation_strictness, 0.75)
        if recall < 0.2:
            return 3.0 * (1.0 - recall) * recall_penalty_scale
        if recall < 0.5:
            return 2.0 * (1.0 - recall) * recall_penalty_scale
        if recall < 0.8:
            return 1.1 * (1.0 - recall) * recall_penalty_scale
        return 0.5 * (1.0 - recall) * recall_penalty_scale

    def _seasonality_strength_loss(self, detected_strengths, true_strengths):
        if not true_strengths:
            return 0.0
        detected_strengths = detected_strengths or {}
        loss = 0.0
        n_items = 0
        for key, true_value in true_strengths.items():
            det_value = detected_strengths.get(key)
            if det_value is None and isinstance(key, str) and key.startswith('period_'):
                try:
                    true_period = int(key.split('_')[1])
                except (IndexError, ValueError):
                    true_period = None
                if true_period is not None:
                    best_match_penalty = None
                    best_match_det_val = None
                    for det_key, det_val in detected_strengths.items():
                        if not (
                            isinstance(det_key, str) and det_key.startswith('period_')
                        ):
                            continue
                        try:
                            det_period = int(det_key.split('_')[1])
                        except (IndexError, ValueError):
                            continue
                        period_diff = abs(true_period - det_period)
                        if period_diff <= max(1, true_period * 0.05):
                            val_penalty = abs(det_val - true_value) / (
                                abs(true_value) + 1e-6
                            )
                            proximity_penalty = (
                                period_diff / (true_period + 1e-6)
                            ) * 0.1
                            total_penalty = min(val_penalty + proximity_penalty, 2.0)
                            if (
                                best_match_penalty is None
                                or total_penalty < best_match_penalty
                            ):
                                best_match_penalty = total_penalty
                                best_match_det_val = det_val
                    if best_match_penalty is not None:
                        # Apply the same underprediction asymmetry (1.5×) used
                        if best_match_det_val < true_value:
                            best_match_penalty = min(best_match_penalty * 1.25, 2.0)
                        loss += best_match_penalty
                        n_items += 1
                        continue
            if det_value is None:
                det_value = detected_strengths.get(
                    'combined', detected_strengths.get('seasonality_strength')
                )
            if det_value is None:
                loss += 0.5 + abs(true_value)
            else:
                error = det_value - true_value
                penalty = abs(error) / (abs(true_value) + 1e-6)
                if error < 0:
                    penalty *= 1.25
                loss += min(penalty, 2.0)
            n_items += 1
        return loss / max(1, n_items)

    def _seasonality_pattern_loss(
        self, detected_components, true_components, date_index=None
    ):
        detected_series = detected_components.get('seasonality')
        true_series = true_components.get('seasonality')
        if detected_series is None or true_series is None:
            return 0.5
        rmse_penalty = self._component_rmse_penalty(detected_series, true_series)
        wasserstein_penalty = self._component_wasserstein_penalty(
            detected_series, true_series
        )
        spectral_penalty = self._component_spectral_penalty(
            detected_series, true_series
        )
        profile_penalty = self._component_profile_correlation(
            detected_series,
            true_series,
            date_index=date_index,
        )

        # Explicit asymmetric amplitude penalty to prevent underprediction
        detected_amp = np.nanstd(np.asarray(detected_series, dtype=float))
        true_amp = np.nanstd(np.asarray(true_series, dtype=float))
        amplitude_penalty = 0.0
        if np.isfinite(detected_amp) and np.isfinite(true_amp) and true_amp > 1e-6:
            ratio = detected_amp / true_amp
            if ratio < 1.0:
                # Heavy penalty for underpredicting amplitude
                amplitude_penalty = (1.0 - ratio) * 0.4
            else:
                # Lighter penalty for slight overpredicting
                amplitude_penalty = min((ratio - 1.0) * 0.1, 0.4)

        # Blend across four complementary metrics:
        # - RMSE: point-wise accuracy (penalizes phase shifts, keeps magnitude honest)
        # - Wasserstein: shape/energy distribution (phase-tolerant)
        # - Spectral: frequency content match (phase-invariant; peak_mae now uses
        #   per-peak relative error so yearly and weekly are equally important)
        # - Profile correlation: periodic shape fidelity (day-of-week, month, etc.;
        #   auto-adapts to data frequency; explicitly checks each period separately
        #   so this is the primary guard against wrong yearly shape and missing weekly)
        # - Amplitude: asymmetric penalty forcing model to capture full swing
        # Wasserstein weight is reduced because it is shape-tolerant
        # Profile weight is raised because it directly checks the seasonal profile
        # (now also includes the quarter profile for daily data, making it more
        # reliable; RMSE is slightly downweighted because it is phase-sensitive)
        return (
            0.25 * rmse_penalty
            + 0.10 * wasserstein_penalty
            + 0.25 * spectral_penalty
            + 0.40 * profile_penalty
            + amplitude_penalty
        )

    def _seasonality_changepoint_loss(
        self, detected_cp, true_cp, detected_components, true_components, date_index
    ):
        if not true_cp:
            return 0.1 * len(detected_cp or [])
        if date_index is None:
            return 0.5 * len(true_cp)
        detected_dates = [
            self._parse_generic_date(event) for event in (detected_cp or [])
        ]
        seasonality_array = np.asarray(
            detected_components.get('seasonality', []), dtype=float
        )
        true_array = np.asarray(true_components.get('seasonality', []), dtype=float)
        if seasonality_array.size == 0 or seasonality_array.size != len(date_index):
            return 0.6 * len(true_cp)
        loss = 0.0
        for event in true_cp:
            cp_date = self._parse_generic_date(event)
            if cp_date is None:
                continue
            match = any(
                abs(cp_date - det_date) <= self._change_tolerance
                for det_date in detected_dates
            )
            if match:
                continue
            idx = date_index.get_indexer([cp_date], method='nearest')[0]
            left_slice = slice(max(0, idx - self.seasonality_window), idx)
            right_slice = slice(
                idx, min(len(seasonality_array), idx + self.seasonality_window)
            )
            left_mean = (
                np.nanmean(seasonality_array[left_slice])
                if left_slice.stop > left_slice.start
                else np.nan
            )
            right_mean = (
                np.nanmean(seasonality_array[right_slice])
                if right_slice.stop > right_slice.start
                else np.nan
            )
            true_left = (
                np.nanmean(true_array[left_slice])
                if (
                    true_array.size == seasonality_array.size
                    and left_slice.stop > left_slice.start
                )
                else np.nan
            )
            true_right = (
                np.nanmean(true_array[right_slice])
                if (
                    true_array.size == seasonality_array.size
                    and right_slice.stop > right_slice.start
                )
                else np.nan
            )
            if np.isnan(left_mean) or np.isnan(right_mean):
                loss += 0.6
            else:
                detected_change = abs(right_mean - left_mean)
                expected_change = (
                    abs(true_right - true_left)
                    if not np.isnan(true_left) and not np.isnan(true_right)
                    else np.nan
                )
                if np.isnan(expected_change) or expected_change == 0:
                    penalty = 0.6 if detected_change < 0.1 else 0.0
                else:
                    penalty = max(0.0, 1.0 - detected_change / (expected_change + 1e-6))
                loss += min(penalty, 1.2)
        return loss / max(1, len(true_cp))

    def _noise_level_loss(self, detected_level, true_level, detected_ratio, true_ratio):
        penalties = []
        if true_level is not None:
            if detected_level is None:
                penalties.append(abs(true_level) + 0.5)
            else:
                penalties.append(
                    abs(detected_level - true_level) / (abs(true_level) + 1e-6)
                )
        if true_ratio is not None:
            if detected_ratio is None:
                penalties.append(abs(true_ratio) + 0.5)
            else:
                penalties.append(
                    abs(detected_ratio - true_ratio) / (abs(true_ratio) + 1e-6)
                )
        if not penalties:
            return 0.0
        return sum(min(p, 2.0) for p in penalties) / len(penalties)

    def _noise_regime_loss(self, detected_cp, true_cp):
        detected_dates = [
            self._parse_generic_date(event) for event in (detected_cp or [])
        ]
        true_dates = [self._parse_generic_date(event) for event in (true_cp or [])]
        if not true_dates:
            return 0.05 * len(detected_dates)
        loss = 0.0
        for true_date in true_dates:
            match = any(
                abs(true_date - det_date) <= self._change_tolerance
                for det_date in detected_dates
            )
            if not match:
                loss += 0.6
        false_positives = len(
            [
                det_date
                for det_date in detected_dates
                if not any(
                    abs(det_date - true_date) <= self._change_tolerance
                    for true_date in true_dates
                )
            ]
        )
        loss += 0.1 * false_positives
        return loss

    def _noise_structure_loss(self, detected_components, true_components):
        """
        Penalize low-frequency structure mismatch in the noise component.

        This targets residual drift/level-shift leakage by comparing smoothed
        noise behavior rather than point-wise high-frequency noise values.
        """
        detected_noise = detected_components.get('noise')
        true_noise = true_components.get('noise')
        if true_noise is None and detected_noise is None:
            return 0.0
        if true_noise is None:
            return 0.0
        if detected_noise is None:
            return 0.8

        detected_arr, true_arr = self._aligned_finite_arrays(detected_noise, true_noise)
        if detected_arr.size < 8:
            return 0.0

        n_obs = detected_arr.size
        smooth_window = min(max(7, n_obs // 14), 45)
        detected_smooth = (
            pd.Series(detected_arr)
            .rolling(smooth_window, center=True, min_periods=1)
            .mean()
        ).to_numpy(dtype=float)
        true_smooth = (
            pd.Series(true_arr)
            .rolling(smooth_window, center=True, min_periods=1)
            .mean()
        ).to_numpy(dtype=float)

        true_scale = float(np.nanstd(true_arr))
        if true_scale < 1e-6 or not np.isfinite(true_scale):
            true_scale = float(np.nanmean(np.abs(true_arr))) + 1e-6
        smooth_scale = float(np.nanstd(true_smooth))
        if smooth_scale < 1e-6 or not np.isfinite(smooth_scale):
            smooth_scale = true_scale
        norm_scale = max(smooth_scale, true_scale * 0.2, 1e-6)

        # Core penalty: low-frequency path mismatch.
        smooth_rmse = np.sqrt(np.nanmean((detected_smooth - true_smooth) ** 2))
        smooth_rmse_penalty = min(smooth_rmse / (norm_scale + 1e-6), 3.0)

        # Drift leakage: compare smoothed global slopes.
        x = np.arange(n_obs, dtype=float)
        det_slope = self._robust_linear_slope(x, detected_smooth)
        true_slope = self._robust_linear_slope(x, true_smooth)
        slope_denom = max(abs(true_slope), norm_scale / max(n_obs, 1), 1e-6)
        slope_penalty = min(abs(det_slope - true_slope) / slope_denom, 3.0)

        # Level-shift leakage proxy: compare frequency of large smooth jumps.
        det_diff = np.diff(detected_smooth)
        true_diff = np.diff(true_smooth)
        if true_diff.size == 0:
            shift_penalty = 0.0
        else:
            shift_threshold = np.nanpercentile(np.abs(true_diff), 90)
            if not np.isfinite(shift_threshold) or shift_threshold < 1e-9:
                shift_threshold = np.nanstd(true_diff) * 1.5
            if not np.isfinite(shift_threshold) or shift_threshold < 1e-9:
                shift_threshold = 1e-9
            det_shift_rate = float(np.mean(np.abs(det_diff) > shift_threshold))
            true_shift_rate = float(np.mean(np.abs(true_diff) > shift_threshold))
            shift_penalty = min(
                abs(det_shift_rate - true_shift_rate) / (true_shift_rate + 0.05),
                3.0,
            )

        combined = (
            0.55 * smooth_rmse_penalty + 0.25 * slope_penalty + 0.20 * shift_penalty
        )
        return float(min(max(combined, 0.0), 3.0))

    def _aligned_finite_arrays(self, detected_values, true_values):
        detected_arr = np.asarray(detected_values, dtype=float).ravel()
        true_arr = np.asarray(true_values, dtype=float).ravel()
        length = min(detected_arr.size, true_arr.size)
        if length == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        detected_arr = detected_arr[:length]
        true_arr = true_arr[:length]
        mask = np.isfinite(detected_arr) & np.isfinite(true_arr)
        if not np.any(mask):
            return np.array([], dtype=float), np.array([], dtype=float)
        return detected_arr[mask], true_arr[mask]

    def _robust_linear_slope(self, x_values, y_values):
        x_arr = np.asarray(x_values, dtype=float).ravel()
        y_arr = np.asarray(y_values, dtype=float).ravel()
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.sum() < 2:
            return 0.0
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]
        x_centered = x_arr - np.mean(x_arr)
        y_centered = y_arr - np.mean(y_arr)
        denom = float(np.sum(x_centered**2))
        if denom <= 1e-12:
            return 0.0
        return float(np.sum(x_centered * y_centered) / denom)

    def _metadata_loss(self, detected_scale, true_scale, detected_type, true_type):
        penalties = []
        if true_scale is not None:
            if detected_scale is None:
                penalties.append(abs(true_scale) + 0.5)
            else:
                penalties.append(
                    abs(detected_scale - true_scale) / (abs(true_scale) + 1e-6)
                )
        if true_type is not None:
            penalties.append(0.0 if detected_type == true_type else 0.3)
        if not penalties:
            return 0.0
        return sum(penalties) / len(penalties)

    def _regressor_loss(self, detected_regressors, true_regressors):
        true_by_date, true_coeffs = self._normalize_regressor_schema(true_regressors)
        if not true_by_date and not true_coeffs:
            return 0.0

        detected_by_date, detected_coeffs = self._normalize_regressor_schema(
            detected_regressors
        )

        penalties = []
        if true_by_date:
            event_penalties = []
            for date, impacts in true_by_date.items():
                detected_on_date = detected_by_date.get(date, {})
                for reg_name, true_value in impacts.items():
                    if reg_name not in detected_on_date:
                        event_penalties.append(1.0)
                        continue
                    det_value = detected_on_date.get(reg_name, 0.0)
                    rel_error = abs(det_value - true_value) / (abs(true_value) + 1e-6)
                    event_penalties.append(min(rel_error, 2.0))
            if event_penalties:
                penalties.append(float(np.mean(event_penalties)))

            # Penalize extra detected event-date pairs not present in truth.
            true_pairs = {
                (pd.Timestamp(date), str(reg_name))
                for date, impacts in true_by_date.items()
                for reg_name in impacts.keys()
            }
            detected_pairs = {
                (pd.Timestamp(date), str(reg_name))
                for date, impacts in detected_by_date.items()
                for reg_name in impacts.keys()
            }
            if detected_pairs:
                extras = len(detected_pairs - true_pairs)
                penalties.append(extras / max(1, len(true_pairs)))

        if true_coeffs:
            coeff_penalties = []
            for reg_name, true_value in true_coeffs.items():
                if reg_name not in detected_coeffs:
                    coeff_penalties.append(1.0)
                    continue
                det_value = detected_coeffs.get(reg_name, 0.0)
                rel_error = abs(det_value - true_value) / (abs(true_value) + 1e-6)
                coeff_penalties.append(min(rel_error, 2.0))
            if coeff_penalties:
                penalties.append(float(np.mean(coeff_penalties)))

            coeff_extras = len(set(detected_coeffs.keys()) - set(true_coeffs.keys()))
            penalties.append(coeff_extras / max(1, len(true_coeffs)))

        if not penalties:
            return 0.0
        return float(np.mean(penalties))

    def _normalize_regressor_schema(self, regressor_payload):
        regressor_payload = regressor_payload or {}
        if not isinstance(regressor_payload, dict):
            return {}, {}

        if 'by_date' in regressor_payload or 'coefficients' in regressor_payload:
            by_date_raw = regressor_payload.get('by_date', {})
            coeffs_raw = regressor_payload.get('coefficients', {})
        else:
            # Backward compatibility: treat direct mapping as by_date schema.
            by_date_raw = regressor_payload
            coeffs_raw = {}

        by_date = {}
        if isinstance(by_date_raw, dict):
            for date, impacts in by_date_raw.items():
                if not isinstance(impacts, dict):
                    continue
                normalized_impacts = {}
                for reg_name, value in impacts.items():
                    if self._is_number(value):
                        normalized_impacts[str(reg_name)] = float(value)
                if normalized_impacts:
                    by_date[pd.Timestamp(date)] = normalized_impacts

        coefficients = {}
        if isinstance(coeffs_raw, dict):
            for reg_name, value in coeffs_raw.items():
                if self._is_number(value):
                    coefficients[str(reg_name)] = float(value)

        return by_date, coefficients

    def _parse_generic_date(self, event):
        if isinstance(event, dict):
            date = event.get('date')
        elif isinstance(event, (tuple, list)) and event:
            date = event[0]
        else:
            date = event
        if date is None:
            return None
        return pd.Timestamp(date)

    def _parse_trend_event(self, event):
        if isinstance(event, dict):
            date = pd.Timestamp(event.get('date'))
            prior = float(event.get('prior_slope', 0.0))
            post = float(event.get('new_slope', event.get('posterior_slope', prior)))
        elif isinstance(event, (tuple, list)) and len(event) >= 3:
            date = pd.Timestamp(event[0])
            prior = float(event[1]) if self._is_number(event[1]) else 0.0
            post = float(event[2]) if self._is_number(event[2]) else prior
        elif isinstance(event, (tuple, list)) and len(event) >= 2:
            date = pd.Timestamp(event[0])
            prior = 0.0
            post = float(event[1]) if self._is_number(event[1]) else 0.0
        else:
            date = pd.Timestamp(event)
            prior = 0.0
            post = 0.0
        magnitude = abs(post - prior)
        return date, prior, post, magnitude

    def _parse_level_shift_event(self, event):
        if isinstance(event, dict):
            date = pd.Timestamp(event.get('date'))
            magnitude = abs(float(event.get('magnitude', 1.0)))
        elif isinstance(event, (tuple, list)) and event:
            date = pd.Timestamp(event[0])
            magnitude = (
                abs(float(event[1]))
                if len(event) > 1 and self._is_number(event[1])
                else 1.0
            )
        else:
            date = pd.Timestamp(event)
            magnitude = 1.0
        return date, magnitude

    def _parse_anomaly_event(self, event):
        if isinstance(event, dict):
            date = pd.Timestamp(event.get('date'))
            magnitude = abs(float(event.get('magnitude', 1.0)))
            anomaly_type = event.get('type', 'point_outlier')
            duration = int(event.get('duration', 1) or 1)
        elif isinstance(event, (tuple, list)) and event:
            date = pd.Timestamp(event[0])
            magnitude = (
                abs(float(event[1]))
                if len(event) > 1 and self._is_number(event[1])
                else 1.0
            )
            anomaly_type = event[2] if len(event) > 2 else 'point_outlier'
            duration = (
                int(event[3])
                if len(event) > 3 and isinstance(event[3], (int, float))
                else 1
            )
        else:
            date = pd.Timestamp(event)
            magnitude = 1.0
            anomaly_type = 'point_outlier'
            duration = 1
        if not np.isfinite(magnitude):
            magnitude = 0.0
        if duration < 1:
            duration = 1
        if anomaly_type is None:
            anomaly_type = 'point_outlier'
        return date, magnitude, anomaly_type, duration

    @staticmethod
    def _normalize_holiday_dict(mapping):
        normalized = {}
        for key, value in (mapping or {}).items():
            try:
                normalized[pd.Timestamp(key)] = float(value)
            except (ValueError, TypeError):
                continue
        return normalized
