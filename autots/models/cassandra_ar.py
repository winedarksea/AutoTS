# -*- coding: utf-8 -*-
"""Autoregressive helpers for Cassandra (numpy/pandas only).

Shapes used throughout: T dates, N series, L lags, S seasonal-interaction columns
(column 0 is always a constant 1, so the plain lag main effect is always kept).
Coefficients are stored as (N, L, S) and the time-varying lag weight is
gamma_l(t) = sum_s coef[n, l, s] * season[t, s].
"""

import numpy as np
import pandas as pd

# max over t of sum_l |gamma_l(t)| after stabilization; < 1 guarantees geometric decay
AR_STABILITY_BOUND = 0.98


def _usable_frequency(frequency, index):
    if frequency is None or frequency == "infer":
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
            return None
        frequency = index.freq or pd.infer_freq(index)
    return frequency


def build_lag_array(df, lags, frequency=None, fill=None):
    """Lagged values of wide df by *date*, as a (T, N, L) array.

    Rows dropped from the index (e.g. by remove_excess_anomalies) yield NaN rather than
    silently borrowing a value from a different date. Falls back to positional shifts
    only when no frequency can be determined.

    Args:
        fill (None, 'bfill', or float): how to fill dates with no lagged value
    """
    freq = _usable_frequency(frequency, df.index)
    arrays = []
    for lag in lags:
        if freq is not None:
            shifted = df.shift(int(lag), freq=freq).reindex(df.index)
        else:
            shifted = df.shift(int(lag))
        if fill == "bfill":
            shifted = shifted.bfill()
        elif fill is not None:
            shifted = shifted.fillna(fill)
        arrays.append(shifted.to_numpy(dtype=float))
    if not arrays:
        return np.zeros((df.shape[0], df.shape[1], 0))
    return np.stack(arrays, axis=2)


def lag_history_tail(df, max_lag, frequency=None, fill_value=None):
    """Last max_lag values of df on a regular date grid ending at df.index[-1], (max_lag, N).

    Missing grid dates are forward/back filled, or set to fill_value if given.
    """
    freq = _usable_frequency(frequency, df.index)
    if freq is not None:
        grid = pd.date_range(end=df.index[-1], periods=max_lag, freq=freq)
    else:
        grid = df.index[-max_lag:]
    tail = df.reindex(grid)
    if tail.shape[0] < max_lag:
        # history shorter than the longest lag: pad the oldest end with NaN
        pad = pd.DataFrame(np.nan, index=range(max_lag - tail.shape[0]), columns=df.columns)
        tail = pd.concat([pad, tail.reset_index(drop=True)])
    if fill_value is not None:
        tail = tail.fillna(fill_value)
    else:
        tail = tail.ffill().bfill()
    return np.nan_to_num(tail.to_numpy(dtype=float))


def ar_season_matrix(season_feat, keep_mask=None):
    """Prepend a constant column and drop constant/zero seasonal columns.

    Args:
        season_feat (pd.DataFrame or None): seasonal features, (T, K)
        keep_mask (np.ndarray): bool (K,) chosen at fit; computed from season_feat if None
    Returns:
        (T, 1 + kept) matrix, keep_mask, kept column names
    """
    if season_feat is None:
        return None, None, []
    values = np.asarray(season_feat, dtype=float)
    if keep_mask is None:
        keep_mask = np.nanstd(values, axis=0) > 1e-12
    names = [str(x) for x in np.asarray(season_feat.columns)[keep_mask]]
    ones = np.ones((values.shape[0], 1))
    return np.hstack([ones, values[:, keep_mask]]), keep_mask, names


def build_ar_design(lag_array, season_matrix=None):
    """(T, N, L) lags -> (T, N, L * S) design, lag-major: [lag_l, lag_l * s_1, ...].

    season_matrix must already include the leading constant column (see ar_season_matrix).
    """
    if season_matrix is None:
        return lag_array
    T, N, L = lag_array.shape
    design = lag_array[:, :, :, None] * season_matrix[:, None, None, :]
    return design.reshape(T, N, L * season_matrix.shape[1])


def prune_lag_columns(base_x, lag_x, max_colinearity=None):
    """Bool mask of lag columns to keep.

    Drops zero-variance columns and, if max_colinearity is set, columns whose |corr| with
    any base X column or an earlier kept lag column exceeds it (first of a pair is kept).
    """
    keep = np.nanstd(lag_x, axis=0) > 1e-12
    if max_colinearity is None or lag_x.shape[0] < 3:
        return keep
    base_x = base_x[:, np.nanstd(base_x, axis=0) > 1e-12]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(np.hstack([base_x, lag_x]), rowvar=False))
    n_base = base_x.shape[1]
    for j in range(lag_x.shape[1]):
        if not keep[j]:
            continue
        against = np.concatenate([np.arange(n_base), n_base + np.flatnonzero(keep[:j])])
        if against.size and np.max(np.abs(corr[n_base + j, against])) > max_colinearity:
            keep[j] = False
    return keep


def ar_gamma(coef, season_matrix, n_dates):
    """Time-varying lag weights, (T, N, L)."""
    if season_matrix is None:
        return np.broadcast_to(coef[None, :, :, 0], (n_dates,) + coef.shape[:2])
    return np.einsum("ts,nls->tnl", season_matrix, coef)


def fit_ar_batched(design, target, lam=1.0, recency_weighting=None):
    """Per-series ridge regression solved as one batch.

    Rows with any NaN in design or target are masked out per series.

    Args:
        design (np.ndarray): (T, N, K)
        target (np.ndarray): (T, N)
        lam (float): ridge penalty (all coefficients, there is no intercept)
        recency_weighting (float): same convention as Cassandra's linear model, (t+1)**rec
    Returns:
        coef (N, K), innovation_std (N,)
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    T, N, K = design.shape
    valid = np.isfinite(target) & np.all(np.isfinite(design), axis=2)
    x = np.where(valid[..., None], design, 0.0)
    y = np.where(valid, target, 0.0)
    if recency_weighting is not None:
        weights = (np.arange(T) + 1.0) ** recency_weighting
        x = x * weights[:, None, None]
        y = y * weights[:, None]
    # series-major (N, T, K) so the Gram products run as batched BLAS matmuls; einsum
    # with the shared n index does not dispatch to BLAS and was ~30x slower on wide panels
    x = np.ascontiguousarray(x.transpose(1, 0, 2))
    xtx = np.matmul(x.transpose(0, 2, 1), x)
    xty = np.matmul(x.transpose(0, 2, 1), y.T[..., None])[..., 0]
    # tiny floor keeps series with no valid rows (or lam=0) solvable, giving zero coefficients
    lam = 1e-8 if lam is None or lam <= 0 else float(lam)
    xtx = xtx + lam * np.eye(K)[None, :, :]
    coef = np.linalg.solve(xtx, xty[..., None])[..., 0]
    fitted = np.matmul(
        np.nan_to_num(design).transpose(1, 0, 2), coef[..., None]
    )[..., 0].T
    resid = np.where(valid, target - fitted, np.nan)
    count = valid.sum(axis=0)
    innovation_std = np.sqrt(np.nansum(resid**2, axis=0) / np.maximum(count - K, 1))
    return coef, innovation_std


def fit_ar_from_lags(
    lag_array,
    season_matrix,
    target,
    lam=1.0,
    recency_weighting=None,
    max_chunk_elements=25_000_000,
):
    """fit_ar_batched over series chunks, building each chunk's design from the lags.

    The full (T, N, L * S) design plus fit_ar_batched's same-size temporaries reach several
    GB on wide daily panels with seasonal interactions (2,869 x 500 x 74 is ~850 MB each);
    chunking caps peak memory near max_chunk_elements float64s (~200 MB) per temporary.
    Series are independent ridge problems, so chunking does not change the result.
    """
    T, N, L = lag_array.shape
    S = 1 if season_matrix is None else season_matrix.shape[1]
    chunk = max(1, int(max_chunk_elements // max(T * L * S, 1)))
    coef = np.empty((N, L * S))
    innovation_std = np.empty(N)
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        coef[start:stop], innovation_std[start:stop] = fit_ar_batched(
            build_ar_design(lag_array[:, start:stop], season_matrix),
            target[:, start:stop],
            lam=lam,
            recency_weighting=recency_weighting,
        )
    return coef, innovation_std


def stabilize_ar(coef, season_matrix, n_dates, bound=AR_STABILITY_BOUND):
    """Shrink each series' coefficients so max_t sum_l |gamma_l(t)| <= bound.

    This is a sufficient condition for the recursion to decay toward its base.
    """
    gamma = ar_gamma(coef, season_matrix, n_dates)
    peak = np.abs(gamma).sum(axis=2).max(axis=0)  # (N,)
    scale = np.where(peak > bound, bound / np.maximum(peak, 1e-12), 1.0)
    return coef * scale[:, None, None]


def ar_recursive_forecast(
    history_tail, base, coef, season_matrix_future, lags, dynamic_base_fn=None
):
    """Run z_t = base_t + extra_t + sum_l gamma_l(t) z_{t-l} over the horizon.

    Loops over the horizon only, vectorized across series. When every lag reaches back
    into history (min(lag) >= horizon) and there is no dynamic term, no loop is needed.

    Args:
        history_tail (np.ndarray): (max_lag, N) most recent values, last row newest
        base (np.ndarray): (H, N)
        coef (np.ndarray or None): (N, L, S); None means no AR term
        season_matrix_future (np.ndarray or None): (H, S)
        lags (list): lag per L
        dynamic_base_fn (callable): f(step, path_so_far (step, N)) -> (N,) added to base
    Returns:
        path (H, N), ar_contribution (H, N), psi (H, N) impulse response with psi_0 = 1
    """
    base = np.asarray(base, dtype=float)
    H, N = base.shape
    lags = np.asarray(lags if lags is not None else [], dtype=int)
    if coef is None or lags.size == 0:
        gamma = np.zeros((H, N, 0))
        lags = np.zeros(0, dtype=int)
    else:
        gamma = ar_gamma(coef, season_matrix_future, H)
    max_lag = int(lags.max()) if lags.size else 0
    hist = np.asarray(history_tail, dtype=float)[-max_lag:] if max_lag else np.zeros((0, N))
    ext = np.vstack([hist, np.zeros((H, N))])
    steps = np.arange(H)
    contribution = np.zeros((H, N))
    if dynamic_base_fn is None and (lags.size == 0 or lags.min() >= H):
        if lags.size:
            # all referenced values are history: (H, L, N) gather
            ref = ext[max_lag + steps[:, None] - lags[None, :]]
            contribution = np.einsum("hnl,hln->hn", gamma, ref)
        path = base + contribution
    else:
        for h in range(H):
            extra = 0.0
            if dynamic_base_fn is not None:
                extra = dynamic_base_fn(h, ext[max_lag : max_lag + h])
            if lags.size:
                contribution[h] = np.einsum(
                    "nl,ln->n", gamma[h], ext[max_lag + h - lags]
                )
            ext[max_lag + h] = base[h] + extra + contribution[h]
        path = ext[max_lag:]
    # impulse response for interval variance sigma^2 * cumsum(psi^2)
    psi = np.zeros((H + max_lag, N))
    psi[max_lag] = 1.0
    if lags.size:
        for h in range(1, H):
            psi[max_lag + h] = np.einsum("nl,ln->n", gamma[h], psi[max_lag + h - lags])
    return path, contribution, psi[max_lag:]


def interval_scale_from_psi(psi, cap_random_walk=False):
    """sqrt(cumsum(psi^2)), optionally capped at random-walk growth sqrt(h + 1)."""
    scale = np.sqrt(np.cumsum(psi**2, axis=0))
    if cap_random_walk:
        scale = np.minimum(scale, np.sqrt(np.arange(1, psi.shape[0] + 1))[:, None])
    return scale
