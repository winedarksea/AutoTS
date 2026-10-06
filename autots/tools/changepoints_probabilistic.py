"""
Probabilistic changepoint helpers: Bayesian online changepoint detection and
residual block bootstrap resampling. Used by ChangepointDetector when
probabilistic_output=True.
"""

import math

import numpy as np


def _robust_location_scale(data):
    """Median and noise sigma (MAD of first differences), robust to level shifts."""
    center = float(np.median(data))
    diffs = np.diff(data)
    scale = 0.0
    if diffs.size:
        scale = float(np.median(np.abs(diffs - np.median(diffs)))) / (
            0.6745 * math.sqrt(2.0)
        )
        if scale <= 1e-8:
            scale = float(np.std(diffs)) / math.sqrt(2.0)
    if scale <= 1e-8:
        scale = float(np.std(data))
    return center, scale


def bayesian_online_changepoint_probabilities(data, hazard_rate=0.01, lag=5):
    """
    Bayesian online changepoint detection (Adams & MacKay 2007).

    Gaussian segments with unknown mean and variance (Normal-Inverse-Gamma
    prior, Student-t predictive), computed in log space so long runs cannot
    underflow the evidence to zero.

    The filtered P(changepoint at t | x_1..t) is identically the hazard rate under
    a constant hazard, so it carries no information. Instead this reports
    P(a segment starts at s | x_1..s+lag-1): the posterior mass on the run that
    began at s, read lag-1 steps later once evidence has accumulated. The last
    positions use the final posterior.

    Parameters:
    data (array-like): 1-D series without NaNs.
    hazard_rate (float): Prior probability of a changepoint at each step.
    lag (int): Observations after s used to judge a change at s.

    Returns:
    np.ndarray: Changepoint probability per position (position 0 is 0).
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    probabilities = np.zeros(n, dtype=float)
    if n < 3:
        return probabilities
    center, scale = _robust_location_scale(data)
    if scale <= 1e-12:
        return probabilities
    # Standardize so a unit-scale prior is weakly informative for any series.
    x = (data - center) / scale
    lag = int(max(1, min(lag, n - 1)))
    hazard_rate = float(np.clip(hazard_rate, 1e-9, 1 - 1e-9))
    log_hazard = math.log(hazard_rate)
    log_growth = math.log1p(-hazard_rate)

    # NIG prior: alpha0 = 1, beta0 = 1 puts the prior noise variance near 1.
    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0
    # alpha only depends on run length, so its gamma terms are tabulated once.
    alphas = alpha0 + 0.5 * np.arange(n + 1)
    lgamma_ratio = np.array(
        [math.lgamma(a + 0.5) - math.lgamma(a) for a in alphas], dtype=float
    )

    def update(mu, kappa, beta, value):
        kappa_new = kappa + 1.0
        mu_new = (kappa * mu + value) / kappa_new
        beta_new = beta + kappa * (value - mu) ** 2 / (2.0 * kappa_new)
        return mu_new, kappa_new, beta_new

    # Run-length hypotheses r = 1..t (observations in the current run incl. x_t).
    mu, kappa, beta = update(np.array([mu0]), np.array([kappa0]), np.array([beta0]), x[0])
    log_run = np.array([0.0])

    for t in range(1, n):
        r = np.arange(1, log_run.size + 1)
        alpha = alphas[r]
        var = beta * (kappa + 1.0) / (alpha * kappa)
        log_pred = (
            lgamma_ratio[r]
            - 0.5 * np.log(2.0 * np.pi * alpha * var)
            - (alpha + 0.5) * np.log1p((x[t] - mu) ** 2 / (2.0 * alpha * var))
        )
        var0 = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0)
        log_pred0 = (
            lgamma_ratio[0]
            - 0.5 * math.log(2.0 * math.pi * alpha0 * var0)
            - (alpha0 + 0.5) * math.log1p((x[t] - mu0) ** 2 / (2.0 * alpha0 * var0))
        )
        log_prev_total = np.logaddexp.reduce(log_run)
        # A new run starting at t has length 1 regardless of the old run length.
        log_new = log_prev_total + log_hazard + log_pred0
        log_grow = log_run + log_growth + log_pred
        log_run = np.concatenate(([log_new], log_grow))
        log_run -= np.logaddexp.reduce(log_run)

        mu_grow, kappa_grow, beta_grow = update(mu, kappa, beta, x[t])
        mu_new, kappa_new, beta_new = update(
            np.array([mu0]), np.array([kappa0]), np.array([beta0]), x[t]
        )
        mu = np.concatenate((mu_new, mu_grow))
        kappa = np.concatenate((kappa_new, kappa_grow))
        beta = np.concatenate((beta_new, beta_grow))

        start = t - lag + 1
        if start >= 1:
            probabilities[start] = math.exp(log_run[lag - 1])

    # Positions too close to the end for a full lag: use the final posterior.
    final = np.exp(log_run)
    tail_starts = np.arange(max(1, n - lag + 1), n)
    probabilities[tail_starts] = final[n - tail_starts - 1]
    return probabilities


def piecewise_mean_fit(data, changepoints):
    """Fitted values holding each segment at its mean."""
    data = np.asarray(data, dtype=float)
    bounds = np.unique(
        np.concatenate(([0], np.asarray(changepoints, dtype=int), [data.size]))
    )
    bounds = bounds[(bounds >= 0) & (bounds <= data.size)]
    segment_ids = np.repeat(np.arange(bounds.size - 1), np.diff(bounds))
    sums = np.bincount(segment_ids, weights=data)
    counts = np.bincount(segment_ids)
    return (sums / np.maximum(counts, 1))[segment_ids]


def residual_block_bootstrap(fitted, residuals, block_length, rng):
    """
    One moving-block bootstrap replicate: fitted + resampled residual blocks.

    Shuffling raw observations destroys the segment structure being estimated;
    resampling contiguous residual blocks around the fitted segmentation keeps
    the structure and the residuals' short-range autocorrelation.
    """
    n = residuals.size
    block_length = int(max(1, min(block_length, n)))
    n_blocks = -(-n // block_length)
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    positions = (starts[:, None] + np.arange(block_length)[None, :]).ravel()[:n]
    return fitted + residuals[positions]
