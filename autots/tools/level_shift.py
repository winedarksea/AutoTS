"""Vectorized level shift detection: multiscale detrended MOSUM (``linear_gap``).

Pure numpy functions on ``(T, C)`` wide arrays used by ``LevelShiftMagic``.
Index convention: ``i`` is the first post-shift row, ``f = i - 1`` the last
pre-shift row. Detections are returned as sparse ``(rows_f, cols)`` arrays.

The score is a two-intercept, pooled-slope (Chow-style) gap between the back
window ``[i-N, i)`` and the forward window ``[i, i+N)``. It is ~0 on any locally
linear trend or slope change, which is what the legacy ``overlap`` statistic
(``~ -(N-1) * slope`` on a trend) gets wrong. See Eichinger & Kirch (2018) on
MOSUM changepoint detection.

Python loops run only over scales, tail distances and fixed IRLS iterations;
never over columns or breaks. Columns are processed in blocks to keep T x C
temporaries cache-sized.
"""

import functools
import warnings

import numpy as np

from autots.tools.level_shift_selection import (
    near_taken,
    thresholded_peaks,
    top_k_per_column,
)

MAD_TO_STD = 1.4826


def quiet_empty_slices(func):
    """Silence numpy empty/all-NaN slice warnings: missing or short series are expected input."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return func(*args, **kwargs)

    return wrapper


def min_window_points(window_size):
    """Fewest points per side for a valid window score; shared by every method."""
    return max(3, int(window_size) // 3)


def _cumsum0(x):
    """Cumulative sum along axis 0 with a leading zero row, so window sums are P[hi] - P[lo]."""
    out = np.empty((x.shape[0] + 1,) + x.shape[1:], dtype=float)
    out[0] = 0.0
    np.cumsum(x, axis=0, out=out[1:])
    return out


class PrefixSums:
    """Prefix sums of count, t, t^2, y and t*y, built once per column block.

    Shared across every scale so each window score is a handful of ``take`` ops.
    Time is centred to keep t^2 sums well-conditioned.
    """

    def __init__(self, values):
        T = values.shape[0]
        t = (np.arange(T, dtype=float) - T / 2.0)[:, None]
        self.T = T
        if not np.isnan(values).any():
            y = values
            self.Sn = _cumsum0(np.ones((T, 1)))
            self.St = _cumsum0(t)
            self.Stt = _cumsum0(t * t)
        else:
            valid = ~np.isnan(values)
            y = np.where(valid, values, 0.0)
            tt = np.where(valid, t, 0.0)
            self.Sn = _cumsum0(valid.astype(float))
            self.St = _cumsum0(tt)
            self.Stt = _cumsum0(tt * t)
        self.Sy = _cumsum0(y)
        self.Sty = _cumsum0(y * t)

    def window(self, prefix, lo, hi):
        """Sum over rows ``[i+lo, i+hi)`` for every row i, clipped to the series."""
        i = np.arange(self.T)
        upper = np.clip(i + hi, 0, self.T)
        lower = np.clip(i + lo, 0, self.T)
        return prefix.take(upper, axis=0) - prefix.take(lower, axis=0)


def chow_gap(prefix_sums, window_size, min_points=None):
    """Pooled-slope two-intercept gap at each row (row = first post-shift index).

    Returns ``(gap, norm)``: gap in data units (post minus pre) and gap divided
    by its OLS standard-error factor (sigma omitted; robust_z supplies scale).
    Rows with fewer than ``min_points`` on either side are NaN, which is what
    keeps the series edges from ever being flagged.
    """
    N = int(window_size)
    min_points = min_window_points(N) if min_points is None else min_points
    pf = prefix_sums
    i = np.arange(pf.T)
    lower, middle, upper = np.clip(i - N, 0, pf.T), i, np.clip(i + N, 0, pf.T)

    def back_and_forward(prefix):
        at_middle = prefix.take(middle, axis=0)
        return (
            at_middle - prefix.take(lower, axis=0),
            prefix.take(upper, axis=0) - at_middle,
        )

    # count/time moments depend only on the row unless the data has NaNs, so
    # the gap is assembled as a linear combination of y prefix sums with
    # (mostly row-only) coefficients instead of full-size per-side temporaries
    nb, nf = back_and_forward(pf.Sn)
    stb, stf = back_and_forward(pf.St)
    sttb, sttf = back_and_forward(pf.Stt)
    with np.errstate(invalid="ignore", divide="ignore"):
        tb, tf = stb / nb, stf / nf
        sxx = (sttb - stb * tb) + (sttf - stf * tf)
        distance = tf - tb
        k = distance / sxx
        coef_forward = 1.0 / nf + k * tf
        coef_back = -1.0 / nb + k * tb
        inverse_se = 1.0 / np.sqrt(1.0 / nb + 1.0 / nf + distance * k)
    invalid = (nb < min_points) | (nf < min_points) | ~np.isfinite(inverse_se)
    invalid |= ~np.isfinite(coef_forward) | ~np.isfinite(coef_back) | ~np.isfinite(k)
    # NaN coefficients propagate, so no full-size masking pass is needed
    coef_forward = np.where(invalid, np.nan, coef_forward)
    coef_back = np.where(invalid, np.nan, coef_back)
    y_low, y_mid, y_up = (pf.Sy.take(idx, axis=0) for idx in (lower, middle, upper))
    ty_span = pf.Sty.take(upper, axis=0) - pf.Sty.take(lower, axis=0)
    # gap = yf - yb - pooled_slope * distance, expanded in prefix sums
    gap = coef_forward * y_up
    gap += (coef_back - coef_forward) * y_mid
    gap -= coef_back * y_low
    gap -= k * ty_span
    norm = gap * inverse_se
    return gap, norm


@quiet_empty_slices
def robust_z(norm, scale_floor=None, stride=1):
    """Single-pass per-column robust z: ``(x - median) / (1.4826 * MAD)``.

    Uses the MAD of the score itself (not of first differences, which is
    miscalibrated under autocorrelated noise). ``scale_floor`` stops float noise
    on noiseless data from being blown up into detections.
    """
    valid = ~np.isnan(norm)
    # The window score is autocorrelated over ~N rows, so every stride-th valid
    # row carries nearly all the information; stride N//8 matched unthinned FP
    # rates on the benchmark, N//4 doubled them. Columns with <= 20 valid rows
    # keep every row.
    if (valid == valid[:, :1]).all():
        # common case (no missing data): every column shares the same valid rows
        shared_rows = np.flatnonzero(valid[:, 0] if valid.shape[1] else valid.any(1))
        sample = norm[shared_rows[:: stride if len(shared_rows) > 20 else 1]]
        median = np.median(sample, axis=0)
        mad = np.median(np.abs(sample - median), axis=0)
    else:
        # per-column thinning and medians, so a late-starting column cannot
        # change another column's z
        column_stride = np.where(valid.sum(axis=0) > 20, stride, 1)
        keep = valid
        if stride > 1:
            keep = valid & ((np.cumsum(valid, axis=0) - 1) % column_stride == 0)
        # compact each column's kept values (row order) to the front of a
        # (columns, kept) array; boolean indexing of the transpose walks per column
        kept_counts = keep.sum(axis=0)
        sample_t = np.full((norm.shape[1], int(kept_counts.max())), np.nan)
        front = np.arange(sample_t.shape[1]) < kept_counts[:, None]
        sample_t[front] = norm.T[keep.T]
        sample_ok = ~np.isnan(sample_t)
        median = row_masked_median(sample_t, sample_ok)
        mad = row_masked_median(np.abs(sample_t - median[:, None]), sample_ok)
    scale = MAD_TO_STD * mad
    # MAD collapses to 0 when most windows are identical (integer or noiseless data)
    collapsed = ~(scale > 0)
    if collapsed.any():
        scale = scale.copy()
        scale[collapsed] = 1.2533 * np.nanmean(
            np.abs(norm[:, collapsed] - median[collapsed]), axis=0
        )
    if scale_floor is not None:
        scale = np.maximum(scale, scale_floor)
    scale = np.where((scale > 0) & np.isfinite(scale), scale, 1.0)
    z = norm - median
    z /= scale
    return z


def dilate(mask, distance):
    """True where any True lies within +-distance rows in the same column."""
    cs = _cumsum0(mask.astype(float))
    T = mask.shape[0]
    i = np.arange(T)
    hi = np.clip(i + distance + 1, 0, T)
    lo = np.clip(i - distance, 0, T)
    return (cs.take(hi, axis=0) - cs.take(lo, axis=0)) > 0


def gather_windows(values, rows, cols, offsets):
    """Gather ``values[rows + offsets, cols]`` as ``(n, len(offsets))`` with validity mask."""
    T = values.shape[0]
    idx = rows[:, None] + offsets[None, :]
    inside = (idx >= 0) & (idx < T)
    out = values[np.clip(idx, 0, T - 1), cols[:, None]]
    valid = inside & ~np.isnan(out)
    return np.where(valid, out, np.nan), valid


def fast_nanmedian(x, axis=0):
    """np.median when NaN-free (one vectorized partition); nanmedian loops per slice."""
    if not np.isnan(x).any():
        return np.median(x, axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=axis)


def row_masked_median(x, ok):
    """Median of ``x[ok]`` per row with one sort; NaN for rows with no valid points.

    np.nanmedian on small rows goes through masked arrays and is ~10x slower.
    """
    filled = np.where(ok, x, np.inf)
    n = ok.sum(axis=1)
    out = np.full(x.shape[0], np.nan)
    # invalid entries sort last as +inf, so rows sharing a count share order
    # statistics; partitioning per distinct count beats a full sort
    for count in np.unique(n):
        if count == 0:
            continue
        members = np.flatnonzero(n == count)
        kth = sorted({(count - 1) // 2, count // 2})
        part = np.partition(filled[members], kth, axis=1)
        out[members] = (part[:, (count - 1) // 2] + part[:, count // 2]) / 2.0
    return out


def robust_slope(values, lag, columns=None):
    """Per-column median of ``(y[t+lag] - y[t]) / lag``; robust to spikes and isolated steps.

    ``columns`` limits the work to columns that have candidates; others get 0.
    """
    out = np.zeros(values.shape[1])
    if values.shape[0] <= lag:
        return out
    columns = np.arange(values.shape[1]) if columns is None else columns
    # lagged differences overlap by lag rows, so thinning loses little
    starts = np.arange(0, values.shape[0] - lag, max(1, lag // 4))
    slopes = fast_nanmedian(
        (values[starts + lag][:, columns] - values[starts][:, columns]) / lag, axis=0
    )
    out[columns] = np.where(np.isfinite(slopes), slopes, 0.0)
    return out


def _peak_reduction_passes(residual, valid, longest_bump):
    """True where a persistent level explains residuals better than any short bump.

    Level-shift SS reduction ``(sum r)^2 / n`` vs best bump ``max_L (sum_{j<L} r)^2 / L``,
    L counting valid points from the break outward.
    """
    r = np.where(valid, residual, 0.0)
    n = valid.sum(axis=1)
    level_reduction = r.sum(axis=1) ** 2 / np.maximum(n, 1)
    head = min(longest_bump, r.shape[1])
    partial = np.cumsum(r[:, :head], axis=1)
    counts = np.cumsum(valid[:, :head], axis=1)
    bump_reduction = np.max(partial**2 / np.maximum(counts, 1), axis=1)
    return (level_reduction > bump_reduction) & (n > longest_bump)


def bump_test(values, rows_post, cols, gap_sign, window_size, anomaly_window, slopes):
    """Two-sided persistence test: reject candidates a temporary bump explains.

    ``rows_post`` are first post-shift rows. The test window is at least
    ``2 * anomaly_window`` so a bump of ``anomaly_window`` points can always be
    told apart from a step; candidates too close to an edge are rejected.
    """
    test_window = max(int(window_size), 2 * int(anomaly_window))
    skip = min(7, test_window // 3)
    offsets = np.arange(test_window)
    # pre is ordered nearest-the-break first, so both sides read outward
    pre, pre_ok = gather_windows(values, rows_post - 1, cols, -offsets)
    post, post_ok = gather_windows(values, rows_post, cols, offsets)
    slope = slopes[cols][:, None]
    pre_detrended = pre + slope * (offsets + 1)[None, :]
    post_detrended = post - slope * offsets[None, :]
    # empty slices (candidates at the very edge) are NaN and rejected below anyway
    pre_level = row_masked_median(pre_detrended[:, skip:], pre_ok[:, skip:])
    post_level = row_masked_median(post_detrended[:, skip:], post_ok[:, skip:])
    forward_ok = _peak_reduction_passes(
        post_detrended - pre_level[:, None], post_ok, anomaly_window
    )
    backward_ok = _peak_reduction_passes(
        pre_detrended - post_level[:, None], pre_ok, anomaly_window
    )
    same_sign = np.sign(post_level - pre_level) == gap_sign
    return forward_ok & backward_ok & same_sign


def scale_penalty(T, window_size, reference=30):
    """Extra threshold for short windows, which scan more independent positions."""
    T = max(float(T), float(window_size) + 1.0, reference + 1.0)
    return np.sqrt(2 * np.log(T / window_size)) - np.sqrt(2 * np.log(T / reference))


@quiet_empty_slices
def _block_scale_scores(prefix_sums, values, window_size):
    """Gap, robust z and its scale floor for one block at one window size."""
    gap, norm = chow_gap(prefix_sums, window_size)
    magnitude = np.nanmax(np.abs(values), axis=0) if values.size else 1.0
    floor = 1e-9 * np.maximum(1.0, np.nan_to_num(magnitude)) * np.sqrt(window_size)
    z = robust_z(norm, scale_floor=floor, stride=max(1, int(window_size) // 8))
    return gap, z


def _detect_block(values, scales, alpha, anomaly_window):
    """Multiscale detection on one column block; returns sparse candidate arrays."""
    T, C = values.shape
    prefix_sums = PrefixSums(values)
    taken_rows, taken_cols = np.array([], dtype=int), np.array([], dtype=int)
    rows_all, cols_all, scores_all, gaps_all, scale_all = [], [], [], [], []
    for N in sorted(set(int(s) for s in scales)):
        if T < 2 * min_window_points(N) + 1:
            continue
        gap, z = _block_scale_scores(prefix_sums, values, N)
        abs_z = np.abs(np.nan_to_num(z))
        penalty = scale_penalty(T, N)
        rows, cols = thresholded_peaks(abs_z, alpha + penalty, N)
        fresh = ~near_taken(rows, cols, taken_rows, taken_cols, max(3, N // 2), T)
        rows, cols = rows[fresh], cols[fresh]
        if rows.size == 0:
            continue
        gap_values = gap[rows, cols]
        slopes = robust_slope(values, max(N, 2 * anomaly_window), np.unique(cols))
        keep = bump_test(
            values, rows, cols, np.sign(gap_values), N, anomaly_window, slopes
        )
        rows, cols = rows[keep], cols[keep]
        taken_rows = np.concatenate([taken_rows, rows])
        taken_cols = np.concatenate([taken_cols, cols])
        rows_all.append(rows)
        cols_all.append(cols)
        scores_all.append(abs_z[rows, cols] - penalty)
        gaps_all.append(gap_values[keep])
        scale_all.append(np.full(rows.size, N))
    if not rows_all:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([]), np.array([]), empty
    return tuple(
        np.concatenate(x) for x in (rows_all, cols_all, scores_all, gaps_all, scale_all)
    )


def detect_multiscale(
    values, scales, alpha, max_shifts, anomaly_window=10, block_size=2000
):
    """Detect level shifts per column with the multiscale ``linear_gap`` score.

    Returns ``(rows_f, cols, scores, gaps, scales)``: last pre-shift row, column,
    penalised robust z, gap (post minus pre, data units) and detecting window.
    """
    values = np.asarray(values, dtype=float)
    parts = []
    for start in range(0, values.shape[1], block_size):
        block = values[:, start : start + block_size]
        rows, cols, scores, gaps, used = _detect_block(
            block, scales, alpha, anomaly_window
        )
        parts.append((rows, cols + start, scores, gaps, used))
    rows, cols, scores, gaps, used = (np.concatenate(x) for x in zip(*parts))
    keep = top_k_per_column(cols, scores, max_shifts)
    return rows[keep] - 1, cols[keep], scores[keep], gaps[keep], used[keep]


def detect_multiscale_shared(values, scales, alpha, max_shifts, anomaly_window=10):
    """Univariate mode: shared dates from mean per-series z^2, accepted per series.

    Combining z^2 (not standardised levels) stops opposite-sign shifts in
    different series cancelling. Each series then takes a shared date only if it
    is individually significant there and passes its own bump test.
    """
    values = np.asarray(values, dtype=float)
    T, C = values.shape
    prefix_sums = PrefixSums(values)
    taken_dates = np.zeros((T, 1), dtype=bool)
    date_rows, date_scores, date_scales = [], [], []
    accepted_rows, accepted_cols, accepted_gaps = [], [], []
    for N in sorted(set(int(s) for s in scales)):
        if T < 2 * min_window_points(N) + 1:
            continue
        gap, z = _block_scale_scores(prefix_sums, values, N)
        # average only over series with a valid score at each date, so missing
        # series don't dilute a shared shift
        valid_count = np.isfinite(z).sum(axis=1, keepdims=True)
        abs_z = np.abs(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0))
        combined = np.sqrt(
            np.sum(abs_z**2, axis=1, keepdims=True) / np.maximum(valid_count, 1)
        )
        penalty = scale_penalty(T, N)
        dates, _ = thresholded_peaks(combined, alpha + penalty, N)
        dates = dates[~dilate(taken_dates, max(3, N // 2))[dates, 0]]
        if dates.size == 0:
            continue
        # per-series significance tolerates a one-row disagreement in peak location
        near = np.maximum(
            abs_z, np.maximum(np.roll(abs_z, 1, 0), np.roll(abs_z, -1, 0))
        )
        rows = np.repeat(dates, C)
        cols = np.tile(np.arange(C), dates.size)
        significant = near[rows, cols] - penalty > alpha
        slopes = robust_slope(values, max(N, 2 * anomaly_window), np.unique(cols))
        gap_values = gap[rows, cols]
        passes = significant & bump_test(
            values, rows, cols, np.sign(gap_values), N, anomaly_window, slopes
        )
        used_dates = np.unique(rows[passes])
        taken_dates[used_dates, 0] = True
        date_rows.append(used_dates)
        date_scores.append(combined[used_dates, 0] - penalty)
        date_scales.append(np.full(used_dates.size, N))
        accepted_rows.append(rows[passes])
        accepted_cols.append(cols[passes])
        accepted_gaps.append(gap_values[passes])
    empty = np.array([], dtype=int)
    if not date_rows:
        return empty, empty, np.array([]), np.array([]), empty
    date_rows = np.concatenate(date_rows)
    date_scores = np.concatenate(date_scores)
    date_scales = np.concatenate(date_scales)
    keep_dates = date_rows[
        top_k_per_column(np.zeros_like(date_rows), date_scores, max_shifts)
    ]
    rows = np.concatenate(accepted_rows)
    cols = np.concatenate(accepted_cols)
    gaps = np.concatenate(accepted_gaps)
    keep = np.isin(rows, keep_dates)
    rows, cols, gaps = rows[keep], cols[keep], gaps[keep]
    order = np.argsort(date_rows)
    position = order[np.searchsorted(date_rows[order], rows)]
    return rows - 1, cols, date_scores[position], gaps, date_scales[position]


def tail_scan(
    values, alpha, min_post=10, max_post=20, reference=60, last_break_rows=None
):
    """Find at most one shift per column 'min_post'..'max_post' rows from the end.

    The main detector can't see shifts this close to the end (the bump test
    needs more than ``anomaly_window`` post points). A line fitted to the pre
    window is extrapolated over the last k points; a shift needs a
    long-run-variance z above alpha and >=80% of post points on the shifted side.
    ``last_break_rows`` bounds the pre window so an earlier break isn't mistaken
    for a tail one. Returns ``(rows_f, cols, gaps)``.
    """
    values = np.asarray(values, dtype=float)
    T, C = values.shape
    if last_break_rows is None:
        last_break_rows = np.full(C, -1)
    best_z = np.zeros(C)
    best_k = np.zeros(C, dtype=int)
    best_gap = np.zeros(C)
    min_reference = 10
    for k in range(int(min_post), int(max_post) + 1):
        post_start = T - k
        if post_start - min_reference < 0:
            break
        rows = post_start - reference + np.arange(reference)
        pre = values[np.clip(rows, 0, T - 1)]
        pre_ok = (rows >= 0)[:, None] & (rows[:, None] > last_break_rows[None, :])
        pre_ok &= ~np.isnan(pre)
        n_pre = pre_ok.sum(axis=0)
        t_pre = np.where(pre_ok, (rows - post_start)[:, None].astype(float), 0.0)
        y_pre = np.where(pre_ok, pre, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            t_mean = t_pre.sum(axis=0) / n_pre
            y_mean = y_pre.sum(axis=0) / n_pre
            t_centred = np.where(pre_ok, t_pre - t_mean, 0.0)
            sxx = (t_centred**2).sum(axis=0)
            slope = (t_centred * (y_pre - y_mean)).sum(axis=0) / sxx
            residual = np.where(pre_ok, y_pre - y_mean - slope * t_centred, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                centre = np.nanmedian(residual, axis=0)
                sigma = MAD_TO_STD * np.nanmedian(np.abs(residual - centre), axis=0)
                lag_corr = np.nansum(residual[1:] * residual[:-1], axis=0) / np.nansum(
                    residual**2, axis=0
                )
            lag_corr = np.clip(np.nan_to_num(lag_corr), 0.0, 0.95)
            sigma = sigma * np.sqrt((1 + lag_corr) / (1 - lag_corr))
            post = values[post_start:]
            t_post = np.arange(k, dtype=float)[:, None]
            post_residual = post - (y_mean + slope * (t_post - t_mean))
            n_post = (~np.isnan(post_residual)).sum(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                gap = np.nanmean(post_residual, axis=0)
            se = sigma * np.sqrt(
                1.0 / n_post + 1.0 / n_pre + ((k - 1) / 2.0 - t_mean) ** 2 / sxx
            )
            z = gap / se
            on_side = np.sign(gap) * (post_residual - gap / 2.0) > 0
            fraction = on_side.sum(axis=0) / n_post
        ok = (
            (n_pre >= min_reference)
            & (n_post >= min_post)
            & np.isfinite(z)
            & (sigma > 0)
            & (np.abs(z) > alpha)
            & (fraction >= 0.8)
        )
        better = ok & (np.abs(z) > np.abs(best_z))
        best_z = np.where(better, z, best_z)
        best_k = np.where(better, k, best_k)
        best_gap = np.where(better, gap, best_gap)
    cols = np.nonzero(best_k > 0)[0]
    return T - best_k[cols] - 1, cols, best_gap[cols]
