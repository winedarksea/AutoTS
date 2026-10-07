# -*- coding: utf-8 -*-
"""Trend extrapolation helpers for TimeSeriesFeatureDetector.forecast.

Two additions over plain last-segment OLS extrapolation:
- future_trend_change_variance: variance from changepoints and level shifts that
  have not happened yet, at the rates the detector observed in training.
- shrink_last_slope: empirical-Bayes shrinkage of a noisy last-segment slope
  toward a recency-weighted average of earlier segment slopes.

All rates are per observation step (not per day) because forecast horizons and
stored slopes are both in steps.
"""

import numpy as np
import pandas as pd


def segment_sxx(segment_length):
    """Sum of squared x deviations for x = 0..m-1, i.e. m(m^2 - 1)/12.

    Shared by the OLS leverage term and the slope-precision used in shrinkage.
    """
    m = np.asarray(segment_length, dtype=float)
    return m * (m**2 - 1.0) / 12.0


def trend_change_rates(trend_changepoints, level_shifts, columns, n_obs):
    """Per-series event rates and mean squared sizes from detected events.

    Args:
        trend_changepoints: {series: [(date, prior_slope, new_slope), ...]}
        level_shifts: {series: [(date, magnitude, ...), ...]}
        columns: series order for the output arrays.
        n_obs: number of training observations (rate denominator).

    Returns:
        dict of (N,) float arrays: rate_cp, mean_sq_slope_change, rate_ls,
        mean_sq_level_shift. Series with no events get zeros, so their
        future-change variance vanishes (the previous behaviour).
    """
    n = len(columns)
    out = {
        'rate_cp': np.zeros(n),
        'mean_sq_slope_change': np.zeros(n),
        'rate_ls': np.zeros(n),
        'mean_sq_level_shift': np.zeros(n),
    }
    if not n_obs or n_obs <= 0:
        return out
    trend_changepoints = trend_changepoints or {}
    level_shifts = level_shifts or {}
    for i, col in enumerate(columns):
        deltas = np.array(
            [float(cp[2]) - float(cp[1]) for cp in trend_changepoints.get(col, [])],
            dtype=float,
        )
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size:
            out['rate_cp'][i] = deltas.size / n_obs
            out['mean_sq_slope_change'][i] = float(np.mean(deltas**2))
        shifts = np.array(
            [float(ls[1]) for ls in level_shifts.get(col, [])], dtype=float
        )
        shifts = shifts[np.isfinite(shifts)]
        if shifts.size:
            out['rate_ls'][i] = shifts.size / n_obs
            out['mean_sq_level_shift'][i] = float(np.mean(shifts**2))
    return out


def future_trend_change_variance(rates, horizon):
    """(H, N) variance added by future slope changes and level shifts.

    A slope change delta arriving at step t shifts the level at horizon h by
    delta*(h - t); summing over Poisson arrivals gives rate*E[delta^2]*sum k^2
    for k = 0..h-1, the exact grid form of h^3/3. Level shifts add rate*E[s^2]*h.
    """
    h = np.arange(1, int(horizon) + 1, dtype=float)[:, np.newaxis]
    sum_sq_lags = (h - 1.0) * h * (2.0 * h - 1.0) / 6.0
    slope_term = (rates['rate_cp'] * rates['mean_sq_slope_change'])[np.newaxis, :]
    shift_term = (rates['rate_ls'] * rates['mean_sq_level_shift'])[np.newaxis, :]
    return slope_term * sum_sq_lags + shift_term * h


def shrink_last_slope(
    slope_info,
    date_index,
    residual_sigma,
    mean_sq_slope_change,
    recency_days=180.0,
):
    """Shrink the last segment slope toward earlier segment slopes.

    Prior mean: earlier slopes weighted by length * exp(-age_days / recency_days),
    age measured from segment end to the training end in calendar days so the
    decay is frequency independent. Shrinkage is precision weighted: the last
    slope's OLS variance sigma^2/Sxx against tau^2 = E[delta^2]/2, the slope
    dispersion implied by independent adjacent segments. Short last segments
    are pulled strongly; long well-determined ones barely move.

    Returns the last slope unchanged when there is no earlier segment or no
    usable dispersion/noise estimate.
    """
    if not slope_info:
        return 0.0
    last_slope = float(slope_info[-1]['slope'])
    if len(slope_info) < 2 or date_index is None or len(date_index) == 0:
        return last_slope
    tau_sq = float(mean_sq_slope_change) / 2.0
    sigma = float(residual_sigma) if residual_sigma is not None else 0.0
    if not (np.isfinite(tau_sq) and tau_sq > 0 and np.isfinite(sigma) and sigma > 0):
        return last_slope

    starts = pd.DatetimeIndex([pd.Timestamp(s['start_date']) for s in slope_info])
    ends = pd.DatetimeIndex([pd.Timestamp(s['end_date']) for s in slope_info])
    # Inclusive observation counts per segment (boundary point shared, as in fit).
    lengths = (
        date_index.searchsorted(ends, side='right')
        - date_index.searchsorted(starts, side='left')
    ).astype(float)
    slopes = np.array([float(s['slope']) for s in slope_info], dtype=float)
    age_days = (date_index[-1] - ends[:-1]) / pd.Timedelta(days=1)
    weights = lengths[:-1] * np.exp(-np.asarray(age_days, dtype=float) / recency_days)
    valid = np.isfinite(slopes[:-1]) & (weights > 0)
    if not valid.any() or lengths[-1] < 2:
        return last_slope
    prior_mean = float(
        np.sum(weights[valid] * slopes[:-1][valid]) / np.sum(weights[valid])
    )
    slope_var = sigma**2 / float(segment_sxx(lengths[-1]))
    return float((tau_sq * last_slope + slope_var * prior_mean) / (tau_sq + slope_var))
