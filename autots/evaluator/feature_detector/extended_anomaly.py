# -*- coding: utf-8 -*-
"""
ExtendedAnomalyDetector - Two-pass anomaly detector for multi-day patterns.

Pass 1: Point anomaly proposals via AnomalyRemoval.
Pass 2: Extended/multi-day pattern detection:
  - CUSUM (sustained mean shift for noisy_burst / transient_change)
  - Cumulative-sum template matching (slope_reversion onset + hold + reversion)
  - Decay extension of pass-1 points (impulse_decay / linear_decay)
  - Segmented sliding-window mean shift (noisy_burst / transient_change)

Can be used standalone or embedded in TimeSeriesFeatureDetector via ExtendedAnomalyMixin.
"""

import copy
import random
import warnings

import numpy as np
import pandas as pd

from autots.tools.transform import AnomalyRemoval

try:
    from scipy.signal import argrelextrema
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Type priority for conflict resolution (lower index = higher specificity)
_TYPE_PRIORITY = [
    'slope_reversion',
    'transient_change',
    'impulse_decay',
    'linear_decay',
    'noisy_burst',
    'point_outlier',
]
_TYPE_RANK = {t: i for i, t in enumerate(_TYPE_PRIORITY)}


def _type_rank(anomaly_type):
    return _TYPE_RANK.get(anomaly_type, len(_TYPE_PRIORITY))


class ExtendedAnomalyDetector:
    """
    Two-pass anomaly detector combining point detection with multi-day pattern detection.

    Pass 1 produces point-level proposals via AnomalyRemoval (optional if
    ``pass1_records`` are provided externally).

    Pass 2 runs four independent detection methods on a clean residual:
    - **CUSUM**: cumulative-sum alarm for sustained mean shifts.
    - **Slope-reversion template**: cumulative-sum peak/trough analysis for
      slow onset → hold → reversion patterns.
    - **Decay extension**: extends pass-1 point detections with exponential or
      linear decay tails.
    - **Segmented mean shift**: sliding-window mean comparison against a rolling
      baseline.

    All per-series events are then merged and de-duplicated into a final list
    that preserves ``start_date``, ``end_date``, ``duration``, ``type``,
    ``magnitude``, and ``score``.

    Parameters
    ----------
    point_anomaly_params : dict, optional
        Keyword arguments forwarded to :class:`~autots.tools.transform.AnomalyRemoval`
        for pass-1 point detection.  If *None* a conservative rolling-zscore
        detector is used.
    sustained_window : int
        Short rolling window (days) used by the segmented-shift and CUSUM
        detectors to compute local means.
    sustained_baseline : int
        Longer window used to estimate the baseline mean and standard deviation.
    sustained_threshold : float
        Standardized deviation threshold (in units of baseline σ) above which a
        window is considered anomalous.
    cusum_k : float
        CUSUM allowance parameter (slack / half-width) in standardized units.
        Smaller values are more sensitive.
    cusum_h : float
        CUSUM decision threshold.  An alarm fires when the accumulated statistic
        exceeds this value.
    slope_reversion_min_hold : int
        Minimum number of days that the cumulative sum must stay elevated before
        a slope-reversion event is flagged.
    slope_reversion_min_reversion : int
        Minimum number of days the reversion phase must last.
    slope_reversion_cumsum_threshold : float
        Minimum peak z-score of the cumulative sum (relative to its rolling σ)
        required to trigger a slope-reversion candidate.
    slope_reversion_max_duration : int
        Maximum allowed duration (days) for slope-reversion events. Longer
        candidates are treated as structural drift and ignored.
    decay_lookahead : int
        Number of days to inspect after a pass-1 peak for a decay tail.
    decay_fit_min_r2 : float
        Minimum R² required for a decay template fit to extend a point event.
    min_segment_run : int
        Minimum number of consecutive elevated windows required by the
        segmented-shift detector to form an event.
    sustained_hysteresis : float
        Fraction of ``sustained_threshold`` used to keep an in-progress
        segmented run active after it starts. Values in (0, 1] reduce
        fragmentation from brief dips.
    segment_max_gap : int
        Maximum number of non-flagged days allowed between two segmented runs
        before stitching them into one event.
    merge_distance_days : int
        Events within this many days of each other (or overlapping) are merged.
    max_anomalies_per_series : int
        Cap on the number of events returned per series.
    """

    def __init__(
        self,
        point_anomaly_params=None,
        sustained_window=7,
        sustained_baseline=60,
        sustained_threshold=2.5,
        cusum_k=0.5,
        cusum_h=5.0,
        slope_reversion_min_hold=5,
        slope_reversion_min_reversion=7,
        slope_reversion_cumsum_threshold=3.0,
        slope_reversion_max_duration=84,
        decay_lookahead=14,
        decay_fit_min_r2=0.5,
        min_segment_run=2,
        sustained_hysteresis=0.7,
        segment_max_gap=1,
        merge_distance_days=3,
        max_anomalies_per_series=25,
    ):
        self.point_anomaly_params = point_anomaly_params
        self.sustained_window = max(2, int(sustained_window))
        self.sustained_baseline = max(
            self.sustained_window * 2, int(sustained_baseline)
        )
        self.sustained_threshold = float(sustained_threshold)
        self.cusum_k = float(cusum_k)
        self.cusum_h = float(cusum_h)
        self.slope_reversion_min_hold = max(2, int(slope_reversion_min_hold))
        self.slope_reversion_min_reversion = max(2, int(slope_reversion_min_reversion))
        self.slope_reversion_cumsum_threshold = float(slope_reversion_cumsum_threshold)
        self.slope_reversion_max_duration = max(
            self.slope_reversion_min_hold + self.slope_reversion_min_reversion + 1,
            int(slope_reversion_max_duration),
        )
        self.decay_lookahead = max(3, int(decay_lookahead))
        self.decay_fit_min_r2 = float(decay_fit_min_r2)
        self.min_segment_run = max(1, int(min_segment_run))
        self.sustained_hysteresis = float(np.clip(sustained_hysteresis, 0.05, 1.0))
        self.segment_max_gap = max(0, int(segment_max_gap))
        self.merge_distance_days = max(0, int(merge_distance_days))
        self.max_anomalies_per_series = max(1, int(max_anomalies_per_series))

        # Filled by fit()
        self._events = {}  # {series_name: [event_dict, ...]}
        self._pass1_detector = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, residual_df, pass1_records=None):
        """
        Detect extended anomalies in *residual_df*.

        Parameters
        ----------
        residual_df : pd.DataFrame
            Clean residual DataFrame (original minus all structured components).
            Each column is treated as an independent series.
        pass1_records : dict, optional
            Pre-computed point anomaly records keyed by series name
            ({series: [{'date': ..., 'magnitude': ..., 'type': ..., 'score': ...}]}).
            When provided the internal pass-1 AnomalyRemoval run is skipped.

        Returns
        -------
        self
        """
        if not isinstance(residual_df, pd.DataFrame) or residual_df.empty:
            self._events = {}
            return self

        all_events = {}
        for col in residual_df.columns:
            series = residual_df[col].astype(float)
            col_pass1 = (pass1_records or {}).get(col, [])

            # If no external pass-1 records, run internal pass-1
            if not col_pass1 and pass1_records is None:
                col_pass1 = self._run_pass1_series(series, col)

            events = self._detect_series(series, col_pass1, residual_df.index)
            all_events[col] = events

        self._events = all_events
        return self

    def get_events(self, series_name=None):
        """
        Return detected events.

        Parameters
        ----------
        series_name : str, optional
            If given, return events for that series only (as a list).
            Otherwise return the full dict.
        """
        if series_name is not None:
            return self._events.get(series_name, [])
        return copy.deepcopy(self._events)

    # ------------------------------------------------------------------
    # Parameter sampling for optimizer
    # ------------------------------------------------------------------

    @staticmethod
    def get_new_params(method='random'):
        """Sample random parameters for optimizer search."""
        return {
            'sustained_window': random.choice([5, 7, 10, 14]),
            'sustained_baseline': random.choice([30, 60, 90]),
            'sustained_threshold': round(random.uniform(2.0, 4.0), 2),
            'cusum_k': random.choice([0.25, 0.5, 0.75]),
            'cusum_h': random.choice([3.0, 5.0, 7.0, 10.0]),
            'slope_reversion_min_hold': random.choice([3, 5, 7]),
            'slope_reversion_min_reversion': random.choice([5, 7, 10]),
            'slope_reversion_cumsum_threshold': round(random.uniform(2.0, 5.0), 1),
            'slope_reversion_max_duration': random.choice([56, 72, 84, 112]),
            'decay_lookahead': random.choice([7, 10, 14]),
            'decay_fit_min_r2': round(random.uniform(0.35, 0.70), 2),
            'merge_distance_days': random.choice([1, 2, 3, 5]),
            'min_segment_run': random.choice([2, 3]),
            'sustained_hysteresis': round(random.uniform(0.55, 0.85), 2),
            'segment_max_gap': random.choice([0, 1, 2]),
        }

    # ------------------------------------------------------------------
    # Per-series orchestration
    # ------------------------------------------------------------------

    def _detect_series(self, series, pass1_events, date_index):
        """Run all pass-2 methods for one series and merge results."""
        n = len(series)
        if n < 10:
            return [self._normalise_event(e, date_index) for e in pass1_events]

        values = series.to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if finite_mask.sum() < 5:
            return [self._normalise_event(e, date_index) for e in pass1_events]

        # Pass-2 detections
        cusum_events = self._detect_cusum(values, date_index)
        slope_events = self._detect_slope_reversion(values, date_index)
        seg_events = self._detect_segmented_shift(values, date_index)

        # Normalise pass-1 events and optionally extend with decay tails
        p1_normalised = [self._normalise_event(e, date_index) for e in pass1_events]
        decayed = self._extend_with_decay(values, p1_normalised, date_index)

        all_events = decayed + cusum_events + slope_events + seg_events
        merged = self._merge_events(all_events, date_index)
        if len(merged) <= self.max_anomalies_per_series:
            return merged

        # When capping event count, keep the strongest events rather than the
        # earliest chronological events, then restore date order for output.
        def _event_strength(event):
            score = event.get('score')
            if score is None or not np.isfinite(score):
                score = 0.0
            magnitude = event.get('magnitude', 0.0)
            if magnitude is None or not np.isfinite(magnitude):
                magnitude = 0.0
            duration = int(event.get('duration', 1) or 1)
            return (float(score), float(abs(magnitude)), duration)

        strongest = sorted(merged, key=_event_strength, reverse=True)[
            : self.max_anomalies_per_series
        ]
        strongest.sort(key=lambda event: event['date'])
        return strongest

    # ------------------------------------------------------------------
    # Pass-1 internal runner (when no external records provided)
    # ------------------------------------------------------------------

    def _run_pass1_series(self, series, col_name):
        """Run AnomalyRemoval on a single-column DataFrame."""
        try:
            params = self.point_anomaly_params or {
                'output': 'multivariate',
                'method': 'rolling_zscore',
                'method_params': {
                    'distribution': 'norm',
                    'alpha': 0.05,
                    'rolling_periods': 90,
                },
                'fillna': 'ffill',
            }
            params = copy.deepcopy(params)
            params['output'] = 'multivariate'
            df_single = series.to_frame()
            detector = AnomalyRemoval(**params)
            detector.fit(df_single)
            anomaly_col = detector.anomalies.iloc[:, 0]
            mask = anomaly_col == -1
            events = []
            for date in series.index[mask]:
                idx = series.index.get_loc(date)
                mag = float(series.iloc[idx]) if np.isfinite(series.iloc[idx]) else 0.0
                score = None
                if hasattr(detector, 'scores') and not detector.scores.empty:
                    try:
                        score = float(detector.scores.iloc[idx, 0])
                    except Exception:
                        pass
                events.append(
                    {
                        'date': pd.Timestamp(date),
                        'magnitude': abs(mag),
                        'type': 'point_outlier',
                        'score': score,
                    }
                )
            return events
        except Exception as exc:
            warnings.warn(
                f"ExtendedAnomalyDetector pass-1 failed for series: {exc}",
                RuntimeWarning,
            )
            return []

    # ------------------------------------------------------------------
    # CUSUM detection
    # ------------------------------------------------------------------

    def _detect_cusum(self, values, date_index):
        """
        CUSUM alarm on standardized residual.

        Catches sustained mean shifts (noisy_burst, transient_change) and
        the hold phase of slope_reversion.
        """
        n = len(values)
        bl = min(self.sustained_baseline, n - 1)
        win = min(self.sustained_window, bl - 1)
        if bl < 4 or win < 2:
            return []

        series = pd.Series(values)
        baseline_series = series.shift(1)
        rolling_mean = baseline_series.rolling(bl, min_periods=max(3, bl // 3)).mean()
        rolling_std = baseline_series.rolling(bl, min_periods=max(3, bl // 3)).std()
        rolling_mean = rolling_mean.bfill().ffill()
        rolling_std = rolling_std.bfill().ffill()

        # Standardize
        z = (series - rolling_mean) / (rolling_std.clip(lower=1e-6))
        z = z.fillna(0.0).to_numpy(dtype=float)

        k = self.cusum_k
        h = self.cusum_h

        S_pos = np.zeros(n)
        S_neg = np.zeros(n)
        for i in range(1, n):
            S_pos[i] = max(0.0, S_pos[i - 1] + z[i] - k)
            S_neg[i] = max(0.0, S_neg[i - 1] - z[i] - k)

        events = []
        for S, direction in [(S_pos, 1.0), (S_neg, -1.0)]:
            alarm = S > h
            # Find contiguous alarm runs
            runs = self._find_runs(alarm)
            for start_idx, end_idx in runs:
                # Walk back to find approximate onset (where S first accumulated > 0)
                onset_idx = start_idx
                for j in range(start_idx, max(0, start_idx - win - 1) - 1, -1):
                    if S[j] <= 0:
                        onset_idx = j + 1
                        break
                onset_idx = max(0, onset_idx)
                # Find end: first index after alarm where CUSUM resets and
                # residual z-score is near baseline.
                end_fall = end_idx
                max_end = min(n - 1, end_idx + self.sustained_baseline)
                for j in range(end_idx + 1, max_end + 1):
                    end_fall = j
                    if S[j] <= 0 and abs(z[j]) <= max(0.5, self.cusum_k + 0.1):
                        end_fall = j
                        break

                duration = end_fall - onset_idx + 1
                if duration < 1:
                    duration = 1
                anomaly_type = 'noisy_burst' if duration <= 5 else 'transient_change'

                # Magnitude = peak absolute value in window
                window_vals = values[onset_idx : end_fall + 1]
                finite_window = window_vals[np.isfinite(window_vals)]
                mag = (
                    float(np.max(np.abs(finite_window))) if finite_window.size else 0.0
                )

                onset_date = date_index[onset_idx]
                end_date = date_index[min(end_fall, n - 1)]
                events.append(
                    {
                        'date': pd.Timestamp(onset_date),
                        'end_date': pd.Timestamp(end_date),
                        'duration': duration,
                        'magnitude': mag,
                        'score': float(np.max(S[onset_idx : end_fall + 1])),
                        'type': anomaly_type,
                        'source': 'cusum',
                    }
                )
        return events

    # ------------------------------------------------------------------
    # Slope-reversion template
    # ------------------------------------------------------------------

    def _detect_slope_reversion(self, values, date_index):
        """
        Cumulative-sum peak/trough template matching for slope_reversion.

        A slope_reversion pattern in the clean residual looks like:
        - cumsum rises (or falls) quickly → holds → then decays back to baseline.
        We detect this by finding large local extrema in the cumsum that subsequently
        return to near-baseline within the expected reversion window.
        """
        n = len(values)
        min_total = (
            self.slope_reversion_min_hold + self.slope_reversion_min_reversion + 2
        )
        if n < min_total + 4:
            return []

        arr = np.where(np.isfinite(values), values, 0.0)
        cumsum = np.cumsum(arr)

        # Estimate rolling std of cumsum to normalise threshold
        cs_series = pd.Series(cumsum)
        cs_std_roll = (
            cs_series.rolling(min(60, n // 3), center=True, min_periods=5)
            .std()
            .bfill()
            .ffill()
        )
        cs_std = cs_std_roll.to_numpy(dtype=float)

        threshold = self.slope_reversion_cumsum_threshold

        events = []
        if SCIPY_AVAILABLE:
            order = max(3, self.slope_reversion_min_hold)
            # Find local maxima and minima in cumsum
            max_idx = argrelextrema(cumsum, np.greater_equal, order=order)[0]
            min_idx = argrelextrema(cumsum, np.less_equal, order=order)[0]
            candidate_idx = sorted(set(max_idx.tolist() + min_idx.tolist()))
        else:
            # Fallback: find points where cumsum sign changes after large deviation
            candidate_idx = []
            for i in range(1, n - 1):
                if abs(cumsum[i]) > abs(cumsum[i - 1]) and abs(cumsum[i]) > abs(
                    cumsum[i + 1]
                ):
                    candidate_idx.append(i)

        for peak_i in candidate_idx:
            peak_val = cumsum[peak_i]
            local_std = cs_std[peak_i] if np.isfinite(cs_std[peak_i]) else 1.0
            if local_std < 1e-6:
                continue
            peak_z = abs(peak_val) / local_std
            if peak_z < threshold:
                continue

            # Find onset: where cumsum first crosses 10% of peak value
            cross_level = peak_val * 0.10
            onset_i = peak_i
            for j in range(peak_i - 1, max(0, peak_i - n) - 1, -1):
                if abs(cumsum[j]) <= abs(cross_level):
                    onset_i = j + 1
                    break

            # Find end of reversion: where cumsum returns within 15% of zero after peak
            revert_level = abs(peak_val) * 0.15
            revert_i = min(n - 1, peak_i + self.slope_reversion_min_reversion)
            max_revert_search = min(n - 1, peak_i + n // 3)
            for j in range(peak_i, max_revert_search + 1):
                if abs(cumsum[j]) <= revert_level:
                    revert_i = j
                    break

            hold_days = peak_i - onset_i
            revert_days = revert_i - peak_i
            duration = revert_i - onset_i + 1

            if hold_days < self.slope_reversion_min_hold:
                continue
            if revert_days < self.slope_reversion_min_reversion:
                continue
            if duration < 1:
                continue
            if duration > self.slope_reversion_max_duration:
                continue

            window_vals = values[onset_i : revert_i + 1]
            finite_window = window_vals[np.isfinite(window_vals)]
            mag = float(np.max(np.abs(finite_window))) if finite_window.size else 0.0
            if mag < 1e-9:
                continue

            onset_date = date_index[onset_i]
            end_date = date_index[min(revert_i, n - 1)]
            events.append(
                {
                    'date': pd.Timestamp(onset_date),
                    'end_date': pd.Timestamp(end_date),
                    'duration': duration,
                    'magnitude': mag,
                    'score': float(peak_z),
                    'type': 'slope_reversion',
                    'source': 'slope_template',
                }
            )
        return events

    # ------------------------------------------------------------------
    # Decay extension
    # ------------------------------------------------------------------

    def _extend_with_decay(self, values, pass1_events, date_index):
        """
        Extend pass-1 point events with exponential or linear decay tails.

        For each point event, inspects the ``decay_lookahead`` days following
        the peak.  If a decay template fits well (R² ≥ ``decay_fit_min_r2``)
        the event is extended and its type updated.
        """
        if not pass1_events:
            return []

        n = len(values)
        extended = []
        for event in pass1_events:
            ev = copy.deepcopy(event)
            try:
                onset_date = pd.Timestamp(ev['date'])
                onset_loc = date_index.get_loc(onset_date)
            except (KeyError, TypeError):
                # Nearest fallback
                try:
                    onset_loc = date_index.get_indexer(
                        [pd.Timestamp(ev['date'])], method='nearest'
                    )[0]
                except Exception:
                    extended.append(ev)
                    continue

            peak_val = (
                values[onset_loc]
                if np.isfinite(values[onset_loc])
                else ev.get('magnitude', 0.0)
            )

            if abs(peak_val) < 1e-9:
                extended.append(ev)
                continue

            look_end = min(n, onset_loc + self.decay_lookahead + 1)
            if look_end <= onset_loc + 3:
                extended.append(ev)
                continue

            post_vals = values[onset_loc + 1 : look_end]
            post_t = np.arange(1, len(post_vals) + 1, dtype=float)

            if len(post_vals) < 3:
                extended.append(ev)
                continue

            finite_mask = np.isfinite(post_vals)
            if finite_mask.sum() < 3:
                extended.append(ev)
                continue

            best_type = None
            best_r2 = self.decay_fit_min_r2
            best_end = onset_loc  # default: no extension

            # --- Exponential decay fit ---
            try:
                decay_vals = post_vals[finite_mask] / (peak_val + 1e-12)
                decay_t = post_t[finite_mask]
                # Use log-linear regression: log|y| = log|A| - λ*t
                valid = np.isfinite(np.log(np.abs(decay_vals) + 1e-9))
                if valid.sum() >= 3:
                    log_y = np.log(np.abs(decay_vals[valid]) + 1e-9)
                    t_valid = decay_t[valid]
                    coeffs = np.polyfit(t_valid, log_y, 1)
                    lam = -coeffs[0]
                    if lam > 0:
                        y_pred = np.exp(np.polyval(coeffs, t_valid))
                        ss_res = np.sum((np.abs(decay_vals[valid]) - y_pred) ** 2)
                        ss_tot = (
                            np.sum(
                                (
                                    np.abs(decay_vals[valid])
                                    - np.mean(np.abs(decay_vals[valid]))
                                )
                                ** 2
                            )
                            + 1e-12
                        )
                        r2 = float(1.0 - ss_res / ss_tot)
                        # Find where the decay crosses 0.1 * |peak| (10% of peak)
                        T_cross = np.log(0.1) / (-lam + 1e-9)
                        T_cross = min(int(np.ceil(T_cross)), self.decay_lookahead)
                        if r2 > best_r2 and T_cross >= 2:
                            best_r2 = r2
                            best_type = 'impulse_decay'
                            best_end = onset_loc + T_cross
            except Exception:
                pass

            # --- Linear decay fit ---
            try:
                decay_vals_lin = post_vals[finite_mask] / (peak_val + 1e-12)
                t_lin = post_t[finite_mask]
                if len(t_lin) >= 3:
                    coeffs_lin = np.polyfit(t_lin, decay_vals_lin, 1)
                    slope_lin = coeffs_lin[0]
                    if slope_lin < 0:
                        y_pred_lin = np.polyval(coeffs_lin, t_lin)
                        ss_res_lin = np.sum((decay_vals_lin - y_pred_lin) ** 2)
                        ss_tot_lin = (
                            np.sum((decay_vals_lin - np.mean(decay_vals_lin)) ** 2)
                            + 1e-12
                        )
                        r2_lin = float(1.0 - ss_res_lin / ss_tot_lin)
                        # Intercept: t at which line crosses 0.1
                        if abs(slope_lin) > 1e-9:
                            T_zero = (0.1 - coeffs_lin[1]) / slope_lin
                            T_zero = min(
                                int(np.ceil(max(T_zero, 1.0))), self.decay_lookahead
                            )
                        else:
                            T_zero = self.decay_lookahead
                        if r2_lin > best_r2 and T_zero >= 2:
                            best_r2 = r2_lin
                            best_type = 'linear_decay'
                            best_end = onset_loc + T_zero
            except Exception:
                pass

            if best_type is not None and best_end > onset_loc:
                best_end = min(best_end, n - 1)
                end_date = date_index[best_end]
                ev['end_date'] = pd.Timestamp(end_date)
                ev['duration'] = best_end - onset_loc + 1
                ev['type'] = best_type
                ev['source'] = 'decay_extension'
            else:
                if 'end_date' not in ev:
                    ev['end_date'] = ev['date']
                if 'duration' not in ev:
                    ev['duration'] = 1
                if 'source' not in ev:
                    ev['source'] = 'pass1'

            extended.append(ev)
        return extended

    # ------------------------------------------------------------------
    # Segmented mean shift
    # ------------------------------------------------------------------

    def _detect_segmented_shift(self, values, date_index):
        """
        Sliding short-window mean deviation compared to a longer baseline.

        Detects noisy_burst and transient_change patterns by checking whether
        the short-window mean is significantly displaced from the rolling
        baseline for at least ``min_segment_run`` consecutive windows.
        """
        n = len(values)
        win = self.sustained_window
        bl = self.sustained_baseline
        if n < win + bl:
            return []

        series = pd.Series(values)
        # Use trailing baseline windows so the candidate anomaly period does not
        # self-contaminate its own reference statistics.
        baseline_series = series.shift(win)
        bl_mean = baseline_series.rolling(bl, min_periods=max(3, bl // 3)).mean()
        bl_std = baseline_series.rolling(bl, min_periods=max(3, bl // 3)).std()
        bl_mean = bl_mean.bfill().ffill()
        bl_std = bl_std.bfill().ffill()

        # Compute z-score of local (short-window) mean vs. baseline
        local_mean = series.rolling(win, min_periods=max(2, win // 2)).mean()
        z = (local_mean - bl_mean) / (bl_std.clip(lower=1e-6))
        z = z.fillna(0.0).to_numpy(dtype=float)

        high_threshold = np.abs(z) > self.sustained_threshold
        low_threshold = np.abs(z) > (
            self.sustained_threshold * self.sustained_hysteresis
        )
        flagged = self._apply_hysteresis(high_threshold, low_threshold)
        flagged = self._bridge_small_gaps(flagged, self.segment_max_gap)

        events = []
        runs = self._find_runs(flagged)
        for start_idx, end_idx in runs:
            run_len = end_idx - start_idx + 1
            if run_len < self.min_segment_run:
                continue

            # Expand to nearby low-threshold support to better capture full
            # 1-3 week transients around the high-threshold core.
            actual_start = start_idx
            actual_end = end_idx
            hold_level = self.sustained_threshold * max(self.sustained_hysteresis, 0.5)
            while actual_start > 0 and abs(z[actual_start - 1]) > hold_level:
                actual_start -= 1
            while actual_end < (n - 1) and abs(z[actual_end + 1]) > hold_level:
                actual_end += 1

            duration = actual_end - actual_start + 1
            anomaly_type = 'noisy_burst' if duration <= 5 else 'transient_change'

            window_vals = values[actual_start : actual_end + 1]
            finite_window = window_vals[np.isfinite(window_vals)]
            mag = float(np.max(np.abs(finite_window))) if finite_window.size else 0.0

            onset_date = date_index[actual_start]
            end_date = date_index[actual_end]
            events.append(
                {
                    'date': pd.Timestamp(onset_date),
                    'end_date': pd.Timestamp(end_date),
                    'duration': duration,
                    'magnitude': mag,
                    'score': float(np.max(np.abs(z[start_idx : end_idx + 1]))),
                    'type': anomaly_type,
                    'source': 'segmented_shift',
                }
            )
        return events

    # ------------------------------------------------------------------
    # Merging and de-duplication
    # ------------------------------------------------------------------

    def _merge_events(self, events, date_index):
        """
        Merge overlapping or adjacent events across all sources.

        Priority rule: more specific types win (slope_reversion > transient_change
        > noisy_burst > point_outlier).  Within the same type, the event with
        the highest magnitude is kept.
        """
        if not events:
            return []

        # Normalise all events to have consistent keys
        normalised = []
        for ev in events:
            n_ev = self._normalise_event(ev, date_index)
            if n_ev is not None:
                normalised.append(n_ev)

        if not normalised:
            return []

        # Sort by onset date
        normalised.sort(key=lambda e: e['date'])

        merged = [copy.deepcopy(normalised[0])]
        gap = pd.Timedelta(days=self.merge_distance_days)

        for ev in normalised[1:]:
            prev = merged[-1]
            prev_end = prev['end_date']
            ev_start = ev['date']

            # Overlapping or within merge_distance_days
            if ev_start <= prev_end + gap:
                prev_duration = int(prev.get('duration', 1) or 1)
                ev_duration = int(ev.get('duration', 1) or 1)
                longer = max(prev_duration, ev_duration)
                shorter = min(prev_duration, ev_duration)
                # Do not collapse short transient events into long-window
                # detections of a different type.
                if shorter <= 21 and (longer / max(shorter, 1)) > 2.0:
                    merged.append(copy.deepcopy(ev))
                    continue
                # Combine: extend end, take most specific type, max magnitude
                new_end = max(prev_end, ev['end_date'])
                new_start = min(prev['date'], ev_start)
                new_duration = max(1, int((new_end - new_start).days) + 1)
                new_type = (
                    ev['type']
                    if _type_rank(ev['type']) < _type_rank(prev['type'])
                    else prev['type']
                )
                new_mag = max(prev['magnitude'], ev['magnitude'])
                score_a = prev.get('score') or 0.0
                score_b = ev.get('score') or 0.0
                merged[-1] = {
                    'date': new_start,
                    'end_date': new_end,
                    'duration': new_duration,
                    'magnitude': new_mag,
                    'score': max(score_a, score_b),
                    'type': new_type,
                    'source': prev.get('source', 'merged'),
                }
            else:
                merged.append(copy.deepcopy(ev))

        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_event(self, event, date_index):
        """Ensure event dict has all required keys."""
        if event is None:
            return None
        ev = copy.deepcopy(event)
        try:
            onset_date = pd.Timestamp(ev.get('date') or ev.get('onset_date'))
        except Exception:
            return None

        duration = int(ev.get('duration', 1) or 1)
        if duration < 1:
            duration = 1

        if 'end_date' not in ev or ev['end_date'] is None:
            end_offset = duration - 1
            if hasattr(date_index, '__len__') and len(date_index) > 0:
                try:
                    loc = date_index.get_indexer([onset_date], method='nearest')[0]
                    end_loc = min(loc + end_offset, len(date_index) - 1)
                    end_date = date_index[end_loc]
                except Exception:
                    end_date = onset_date + pd.Timedelta(days=end_offset)
            else:
                end_date = onset_date + pd.Timedelta(days=end_offset)
        else:
            end_date = pd.Timestamp(ev['end_date'])

        # Recompute duration from actual dates if we have both
        actual_duration = max(1, int((end_date - onset_date).days) + 1)

        return {
            'date': onset_date,
            'end_date': end_date,
            'duration': actual_duration,
            'magnitude': float(
                abs(ev.get('magnitude', 0.0))
                if np.isfinite(ev.get('magnitude', 0.0))
                else 0.0
            ),
            'score': ev.get('score'),
            'type': ev.get('type', 'point_outlier'),
            'source': ev.get('source', 'pass1'),
        }

    @staticmethod
    def _find_runs(bool_array):
        """
        Find contiguous True runs in a boolean array.

        Returns
        -------
        list of (start_idx, end_idx) tuples (inclusive).
        """
        if len(bool_array) == 0:
            return []
        runs = []
        in_run = False
        start = 0
        for i, val in enumerate(bool_array):
            if val and not in_run:
                in_run = True
                start = i
            elif not val and in_run:
                in_run = False
                runs.append((start, i - 1))
        if in_run:
            runs.append((start, len(bool_array) - 1))
        return runs

    def _bridge_small_gaps(self, bool_array, max_gap):
        """Fill short False gaps between True runs to reduce run fragmentation."""
        arr = np.asarray(bool_array, dtype=bool).copy()
        if max_gap <= 0 or arr.size < 3:
            return arr
        runs = self._find_runs(arr)
        if len(runs) < 2:
            return arr
        for idx in range(len(runs) - 1):
            left_end = runs[idx][1]
            right_start = runs[idx + 1][0]
            gap_start = left_end + 1
            gap_end = right_start - 1
            gap_len = gap_end - gap_start + 1
            if 0 < gap_len <= max_gap:
                arr[gap_start : gap_end + 1] = True
        return arr

    @staticmethod
    def _apply_hysteresis(high_flags, low_flags):
        """Start on high threshold, continue while low-threshold support holds."""
        high = np.asarray(high_flags, dtype=bool)
        low = np.asarray(low_flags, dtype=bool)
        if high.size == 0:
            return high
        out = np.zeros(high.size, dtype=bool)
        active = False
        for i in range(high.size):
            if high[i]:
                active = True
            elif active and not low[i]:
                active = False
            out[i] = active
        return out


# ---------------------------------------------------------------------------
# Mixin for TimeSeriesFeatureDetector
# ---------------------------------------------------------------------------


class ExtendedAnomalyMixin:
    """
    Mixin for :class:`TimeSeriesFeatureDetector` that adds a second extended
    anomaly detection pass after the main decomposition is complete.

    The extended pass operates on the cleanest available residual:
    ``noise_component + anomaly_component`` (= original minus all structured
    components), so that structured effects do not contaminate the extended
    anomaly detection.

    Requires the host class to expose:
    - ``self.extended_anomaly_params`` (dict or falsy to disable)
    - ``self._anomaly_records_temp`` (populated by pass-1 in DecompositionMixin)
    """

    def _detect_extended_anomalies(self, clean_residual_df):
        """
        Run the extended anomaly detector on *clean_residual_df*.

        Parameters
        ----------
        clean_residual_df : pd.DataFrame
            Residual after removing trend, level_shifts, seasonality, and
            holiday components (noise + anomaly signal).

        Returns
        -------
        dict
            {series_name: [event_dict, ...]}
        """
        params = getattr(self, 'extended_anomaly_params', None)
        if not params:
            return {}
        try:
            detection_mode = getattr(self, 'detection_mode', 'multivariate')

            if detection_mode == 'univariate':
                # Mirror how AnomalyRemoval works with output='univariate':
                # run detection once on the mean residual across all series,
                # then broadcast the shared events to every column.
                mean_residual = clean_residual_df.mean(axis=1).to_frame(
                    '__univariate__'
                )

                # In univariate mode, pass-1 records are identical for all series;
                # use the first series' records as the shared pass-1 proposals.
                pass1_records = getattr(self, '_anomaly_records_temp', None) or {}
                columns = list(clean_residual_df.columns)
                first_col = columns[0] if columns else None
                shared_pass1 = (
                    {'__univariate__': list(pass1_records.get(first_col, []))}
                    if first_col
                    else {}
                )

                detector = ExtendedAnomalyDetector(**params)
                detector.fit(mean_residual, pass1_records=shared_pass1)
                shared_events = detector.get_events('__univariate__')

                # Broadcast the same shared events to all series
                return {col: list(shared_events) for col in columns}
            else:
                detector = ExtendedAnomalyDetector(**params)
                detector.fit(
                    clean_residual_df,
                    pass1_records=getattr(self, '_anomaly_records_temp', None),
                )
                return detector.get_events()
        except Exception as exc:
            warnings.warn(
                f"ExtendedAnomalyMixin._detect_extended_anomalies failed: {exc}",
                RuntimeWarning,
            )
            return {}

    def _merge_anomaly_records(self, base_records, extended_records):
        """
        Merge pass-1 point records with extended anomaly records.

        Extended records take precedence: any base event whose onset falls
        within an extended event's window is dropped in favour of the extended
        version.  Base events not covered by any extended event are kept as-is.

        Parameters
        ----------
        base_records : dict
            {series_name: [event_dict, ...]} from pass-1.
        extended_records : dict
            {series_name: [event_dict, ...]} from extended detection.

        Returns
        -------
        dict
        """
        if not extended_records:
            return base_records

        merged = {}
        all_series = set(list(base_records.keys()) + list(extended_records.keys()))
        for series_name in all_series:
            base = base_records.get(series_name, [])
            extended = extended_records.get(series_name, [])

            combined = list(extended)  # extended events take precedence

            # Add base events not covered by any extended event window
            for event in base:
                try:
                    onset = pd.Timestamp(event['date'])
                except Exception:
                    combined.append(event)
                    continue

                covered = False
                for ext_ev in extended:
                    try:
                        ext_start = pd.Timestamp(ext_ev['date'])
                        ext_dur = int(ext_ev.get('duration', 1) or 1)
                        ext_end = pd.Timestamp(
                            ext_ev.get(
                                'end_date', ext_start + pd.Timedelta(days=ext_dur - 1)
                            )
                        )
                        if ext_start <= onset <= ext_end:
                            covered = True
                            break
                    except Exception:
                        continue

                if not covered:
                    combined.append(event)

            combined.sort(
                key=lambda e: pd.Timestamp(e.get('date', pd.Timestamp('1900-01-01')))
            )
            merged[series_name] = combined

        return merged
