# -*- coding: utf-8 -*-
"""Trend extrapolation helpers for TimeSeriesFeatureDetector.forecast.

Two additions over plain last-segment OLS extrapolation:
- future_trend_change_variance: variance from changepoints and level shifts that
  have not happened yet, at the rates the detector observed in training.
- shrink_last_slope: empirical-Bayes shrinkage of a noisy last-segment slope
  toward a recency-weighted average of earlier segment slopes.
- damped_trend_steps: per-period damping phi of the extrapolated slope.

All rates are per observation step (not per day) because forecast horizons and
stored slopes are both in steps.
"""

import numpy as np
import pandas as pd


# Calendar damping used by trend_damping='auto'. Undamped extrapolation of a
# precise last-segment slope overshoots at long horizons: on the LRP review panel
# 0.99/day cut 366-730 d MASE to 0.23-0.82x of undamped, and on load_daily /
# synthetic / weekly it improved every bucket on every origin (on top of
# slope_shrinkage, intervals included). Stronger damping scored better still on
# those flat-ish panels, so 0.99 is kept as the more conservative choice that
# still lets genuine growth extrapolate (half-life ~69 days).
DEFAULT_TREND_DAMPING_PER_DAY = 0.99


def validate_trend_damping(trend_damping):
    """Return phi as a float in (0, 1], 'auto', or None (undamped).

    Raises instead of silently ignoring an out-of-range phi: phi > 1 would be an
    accelerating trend and phi <= 0 is meaningless, both almost surely typos.
    """
    if trend_damping is None:
        return None
    if isinstance(trend_damping, str):
        if trend_damping == 'auto':
            return 'auto'
        raise ValueError(f"trend_damping must be 'auto', a float in (0, 1] or None, got {trend_damping!r}")
    phi = float(trend_damping)
    if not 0.0 < phi <= 1.0:
        raise ValueError(f"trend_damping must be 'auto', a float in (0, 1] or None, got {trend_damping!r}")
    return phi


def resolve_trend_damping(trend_damping, date_index=None):
    """Per-period phi (float) or None from a validated trend_damping.

    'auto' is DEFAULT_TREND_DAMPING_PER_DAY compounded over the median period
    length, so the calendar half-life is the same for hourly, daily or weekly
    data (weekly 0.99**7 ~= 0.932). A non-datetime index counts as daily.
    """
    phi = validate_trend_damping(trend_damping)
    if phi != 'auto':
        return phi
    period_days = 1.0
    if isinstance(date_index, pd.DatetimeIndex) and len(date_index) > 1:
        median_gap = pd.Series(date_index).diff().median()
        if pd.notna(median_gap) and median_gap > pd.Timedelta(0):
            period_days = median_gap / pd.Timedelta(days=1)
    return float(DEFAULT_TREND_DAMPING_PER_DAY**period_days)


def damped_trend_steps(forecast_length, trend_damping=None):
    """Effective step counts sum_{i<=h} phi^i for h = 1..forecast_length.

    The trend at step h is ``last + slope * steps[h-1]``; the total increment is
    bounded by slope * phi / (1 - phi) for phi < 1. phi None or 1 gives 1..H.
    """
    steps = np.arange(1, int(forecast_length) + 1, dtype=float)
    phi = validate_trend_damping(trend_damping)
    if phi == 'auto':
        raise ValueError("resolve 'auto' with resolve_trend_damping first")
    if phi is None or phi == 1.0:
        return steps
    return np.cumsum(phi**steps)


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
