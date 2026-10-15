"""Seasonal profile removal used before ``linear_gap`` level shift detection.

Detection-only preprocessing (numpy only): sizing and correction run on the
same deseasonalised values, but the transformer output keeps the seasonality.
"""

import warnings

import numpy as np

from autots.tools.level_shift import _cumsum0, fast_nanmedian, quiet_empty_slices

SEASON_CANDIDATES = (7, 12, 24, 52)


@quiet_empty_slices
def infer_season_length(values, candidates=SEASON_CANDIDATES, min_correlation=0.2):
    """Season length with the largest median lag-m autocorrelation of first differences.

    Returns 0 when no candidate clears ``min_correlation`` (no seasonality).
    """
    T = values.shape[0]
    differences = np.diff(values, axis=0)
    best_length, best_correlation = 0, min_correlation
    for m in candidates:
        if 3 * m >= T:
            continue
        lead, lag = differences[m:], differences[:-m]
        lead = lead - np.nanmean(lead, axis=0)
        lag = lag - np.nanmean(lag, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = np.nansum(lead * lag, axis=0) / np.sqrt(
                np.nansum(lead**2, axis=0) * np.nansum(lag**2, axis=0)
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median_correlation = np.nanmedian(correlation)
        if np.isfinite(median_correlation) and median_correlation > best_correlation:
            best_length, best_correlation = m, median_correlation
    return best_length


@quiet_empty_slices
def deseasonalize(values, season=True):
    """Subtract a median per-phase seasonal profile; detection-only preprocessing.

    ``season``: True infers the length, an int fixes it. Returns
    ``(deseasonalised, season_length)``; season_length 0 means unchanged input.
    Removing the profile beat rounding window sizes to whole seasons, because
    the detector's slope estimate otherwise picks up the seasonal swing.
    """
    values = np.asarray(values, dtype=float)
    T = values.shape[0]
    m = infer_season_length(values) if season is True else int(season)
    if m < 2 or 2 * m > T:
        return values, 0
    valid = ~np.isnan(values)
    sums = _cumsum0(np.where(valid, values, 0.0))
    counts = _cumsum0(valid.astype(float))
    i = np.arange(T)
    lo = np.clip(i - m // 2, 0, T)
    hi = np.clip(lo + m, 0, T)
    window_count = counts.take(hi, axis=0) - counts.take(lo, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centred_mean = (sums.take(hi, axis=0) - sums.take(lo, axis=0)) / window_count
    full_window = (hi - lo == m)[:, None] & (window_count >= 0.5 * m)
    residual = np.where(full_window, values - centred_mean, np.nan)
    cycles = T // m
    shaped = residual[: cycles * m].reshape((cycles, m) + values.shape[1:])
    profile = fast_nanmedian(shaped, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = profile - np.nanmean(profile, axis=0)
    profile = np.nan_to_num(profile)
    return values - profile[i % m], m
