"""Level shift sizing and correction for ``LevelShiftMagic`` (numpy only).

Breaks are sparse ``(rows_f, cols)`` arrays, ``rows_f`` the last pre-shift row.
Sizes are ``post level - pre level`` in data units. Sizing windows are bounded
by the neighbouring breaks in the same column, so close steps don't
contaminate each other.

Breaks are gathered into padded ``(n_breaks, window)`` arrays in blocks, so
there are no Python loops over breaks or columns.
"""

import warnings

import numpy as np

from autots.tools.level_shift import (
    MAD_TO_STD,
    gather_windows,
    row_masked_median,
)

HUBER_C = 1.345


def _masked_median(x, ok):
    return row_masked_median(x, ok & ~np.isnan(x))


def _robust_scale(residual, ok):
    """1.4826 * MAD of residual about zero, with a mean-abs fallback when MAD is 0."""
    ok = ok & ~np.isnan(residual)
    absolute = np.abs(residual)
    scale = MAD_TO_STD * row_masked_median(absolute, ok)
    with np.errstate(invalid="ignore", divide="ignore"):
        fallback = 1.2533 * np.where(ok, absolute, 0.0).sum(axis=1) / ok.sum(axis=1)
    scale = np.where(scale > 0, scale, fallback)
    return np.where(np.isfinite(scale) & (scale > 0), scale, np.inf)


def _huber_weights(residual, ok, scale):
    absolute = np.where(ok, np.abs(residual), 1.0)
    weights = (HUBER_C * scale)[:, None] / np.maximum(absolute, 1e-300)
    np.minimum(weights, 1.0, out=weights)
    return np.where(ok, weights, 0.0)


def huber_location(x, ok, iterations=2, scale=None, start=None):
    """Huber M-estimate of location per row: robust start, fixed scale, IRLS steps.

    ``start`` defaults to the median and ``scale`` to the MAD about it; callers
    with estimates from a longer-window robust fit pass them in.
    """
    location = _masked_median(x, ok) if start is None else start.copy()
    location = np.where(np.isfinite(location), location, 0.0)
    if scale is None:
        scale = _robust_scale(x - location[:, None], ok)
    for _ in range(iterations):
        residual = x - location[:, None]
        weights = _huber_weights(residual, ok, scale)
        total = weights.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            updated = (weights * np.where(ok, x, 0.0)).sum(axis=1) / total
        location = np.where(total > 0, updated, location)
    return location


def _pooled_ols_slope(*sides):
    """Common slope of an OLS fit with one intercept per side, from raw moments."""
    sxx, sxy = 0.0, 0.0
    for t, y, ok in sides:
        t_ok = np.where(ok, t, 0.0)
        y_ok = np.where(ok, y, 0.0)
        n = np.maximum(ok.sum(axis=1), 1)
        t_sum, y_sum = t_ok.sum(axis=1), y_ok.sum(axis=1)
        sxx = sxx + (t_ok * t_ok).sum(axis=1) - t_sum**2 / n
        sxy = sxy + (t_ok * y_ok).sum(axis=1) - t_sum * y_sum / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return sxy / sxx


def _huber_pooled_slope(pre, post, slope, iterations=2, start=None):
    """Common-slope, two-intercept Huber fit; returns (slope, Sxx, scale, intercepts).

    ``start`` = ``(intercept_pre, intercept_post, scale)`` from an earlier fit
    skips the median-based initialisation. Sides stay separate arrays and the
    weighted LS uses raw moment sums, keeping (breaks x window) temporaries few.
    """
    sides = []
    for t, y, ok in (pre, post):
        sides.append((t, np.where(ok, y, 0.0), ok))
    ok_weight = [ok.astype(float) for _, _, ok in sides]
    if start is None:
        # robust median intercepts only to start; later iterations use the WLS ones
        intercepts = [_masked_median(y - slope[:, None] * t, ok) for t, y, ok in sides]
        residuals = np.concatenate(
            [
                y - slope[:, None] * t - a[:, None]
                for (t, y, ok), a in zip(sides, intercepts)
            ],
            axis=1,
        )
        # scale is fixed from the starting residuals (Huber's proposal), which
        # also keeps the IRLS loop free of per-iteration medians
        scale = _robust_scale(
            residuals, np.concatenate([ok for _, _, ok in sides], axis=1)
        )
    else:
        intercepts, scale = list(start[:2]), start[2]
    cutoff = (HUBER_C * scale)[:, None]
    sxx = np.zeros(slope.shape)
    for _ in range(iterations):
        sxx = np.zeros(slope.shape)
        sxy = np.zeros(slope.shape)
        moments = []
        for (t, y, ok), a, ok_float in zip(sides, intercepts, ok_weight):
            # residuals are finite (invalid y zeroed), so a clamped divide is
            # enough; an infinite cutoff gives unit (OLS) weights
            absolute = np.abs(y - slope[:, None] * t - a[:, None])
            w = cutoff / np.maximum(absolute, 1e-300)
            np.minimum(w, 1.0, out=w)
            w *= ok_float
            wt = w * t
            total = np.maximum(w.sum(axis=1), 1e-300)
            t_sum, y_sum = wt.sum(axis=1), (w * y).sum(axis=1)
            sxx += (wt * t).sum(axis=1) - t_sum**2 / total
            sxy += (wt * y).sum(axis=1) - t_sum * y_sum / total
            moments.append((t_sum / total, y_sum / total))
        with np.errstate(invalid="ignore", divide="ignore"):
            updated = sxy / sxx
        slope = np.where(np.isfinite(updated), updated, slope)
        intercepts = [y_mean - slope * t_mean for t_mean, y_mean in moments]
    return slope, sxx, scale, tuple(intercepts)


def neighbour_bounds(rows_f, cols, T):
    """Previous and next break row in the same column (-1 / T-1 when none)."""
    n = rows_f.size
    order = np.lexsort((rows_f, cols))
    r, c = rows_f[order], cols[order]
    same = c[1:] == c[:-1]
    previous = np.full(n, -1)
    following = np.full(n, T - 1)
    previous[1:][same] = r[:-1][same]
    following[:-1][same] = r[1:][same]
    out_prev, out_next = np.empty(n, dtype=int), np.empty(n, dtype=int)
    out_prev[order], out_next[order] = previous, following
    return out_prev, out_next


def _gather_sides(values, rows_f, cols, previous, following, width):
    """Pre (nearest first, t = 0, -1, ...) and post (t = 1, 2, ...) windows, neighbour-bounded."""
    offsets = np.arange(width)
    pre, pre_ok = gather_windows(values, rows_f, cols, -offsets)
    post, post_ok = gather_windows(values, rows_f + 1, cols, offsets)
    pre_ok &= (rows_f[:, None] - offsets[None, :]) > previous[:, None]
    post_ok &= (rows_f[:, None] + 1 + offsets[None, :]) <= following[:, None]
    t_pre = np.broadcast_to(-offsets.astype(float), pre.shape)
    t_post = np.broadcast_to(1.0 + offsets, post.shape)
    return (t_pre, pre, pre_ok), (t_post, post, post_ok)


def _size_block(values, rows_f, cols, windows, previous, following, method):
    width = 3 * int(windows.max())
    pre, post = _gather_sides(values, rows_f, cols, previous, following, width)
    (t_pre, y_pre, ok_pre), (t_post, y_post, ok_post) = pre, post
    j = np.arange(width)[None, :]
    level_window = j < windows[:, None]
    level_pre, level_post = ok_pre & level_window, ok_post & level_window
    if method == "median":
        median_pre = _masked_median(y_pre, level_pre)
        median_post = _masked_median(y_post, level_post)
        return np.nan_to_num(median_post - median_pre), np.nan_to_num(median_pre)

    long_pre, long_post = ok_pre & (j < 3 * windows[:, None]), ok_post & (
        j < 3 * windows[:, None]
    )
    # closed-form pooled OLS start; the fixed-scale Huber steps that follow
    # remove the pull of spikes
    initial = _pooled_ols_slope((t_pre, y_pre, long_pre), (t_post, y_post, long_post))
    initial = np.nan_to_num(initial)
    slope_long, sxx_long, scale_long, intercepts_long = _huber_pooled_slope(
        (t_pre, y_pre, long_pre), (t_post, y_post, long_post), initial
    )
    # local slope skips 2 transition points, catching curvature the 3N fit smooths
    # over. It reuses the long fit's scale: a few-point local MAD badly
    # underestimates sigma and made the curvature test fire on noise.
    local_window = (j >= 2) & level_window
    slope_local, sxx_local, _, intercepts_local = _huber_pooled_slope(
        (t_pre, y_pre, ok_pre & local_window),
        (t_post, y_post, ok_post & local_window),
        slope_long.copy(),
        start=intercepts_long + (scale_long,),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.sqrt(
            np.clip(scale_long**2 * (1.0 / sxx_local - 1.0 / sxx_long), 0, None)
        )
    curved = (np.abs(slope_long - slope_local) > 4 * spread) & (sxx_local > 0)
    curved &= np.isfinite(spread)
    slope = np.where(curved, slope_local, slope_long)
    slope = np.where(np.isfinite(slope), slope, 0.0)

    # the chosen fit's intercepts are the detrended levels at the break, a
    # robust start for the N-window Huber levels
    start_pre = np.where(curved, intercepts_local[0], intercepts_long[0])
    start_post = np.where(curved, intercepts_local[1], intercepts_long[1])
    level_pre_value = huber_location(
        y_pre - slope[:, None] * t_pre, level_pre, scale=scale_long, start=start_pre
    )
    level_post_value = huber_location(
        y_post - slope[:, None] * t_post,
        level_post,
        scale=scale_long,
        start=start_post,
    )
    size = level_post_value - level_pre_value
    pre_level = level_pre_value + 0.5 * slope
    min_points = np.maximum(3, windows // 3)
    short = (level_pre.sum(axis=1) < min_points) | (level_post.sum(axis=1) < min_points)
    fallback = np.flatnonzero(short | ~np.isfinite(size) | ~np.isfinite(pre_level))
    if fallback.size:
        # plain medians only where a flank is too short for a slope
        median_pre = _masked_median(y_pre[fallback], level_pre[fallback])
        median_post = _masked_median(y_post[fallback], level_post[fallback])
        size[fallback] = np.nan_to_num(median_post - median_pre)
        pre_level[fallback] = median_pre
    return size, np.nan_to_num(pre_level)


def size_breaks(
    values, rows_f, cols, window_sizes, method="linear_robust", block_size=1000
):
    """Size each break as post level minus pre level.

    method:
        - "median": difference of medians of the N-windows either side.
        - "linear_robust": Huber pooled slope over up to 3N points per side
          (local N-window slope when curvature is detected), then difference of
          Huber levels of the detrended N-windows. Falls back to "median" when
          a side has fewer than ``max(3, N // 3)`` points.

    Returns ``(sizes, pre_levels)``; pre_level is the level at the break,
    used to build safe ratio corrections.
    """
    values = np.asarray(values, dtype=float)
    rows_f = np.asarray(rows_f, dtype=int)
    cols = np.asarray(cols, dtype=int)
    windows = np.broadcast_to(np.asarray(window_sizes, dtype=int), rows_f.shape)
    windows = np.maximum(windows, 3)
    sizes = np.zeros(rows_f.size)
    pre_levels = np.zeros(rows_f.size)
    if rows_f.size == 0:
        return sizes, pre_levels
    previous, following = neighbour_bounds(rows_f, cols, values.shape[0])
    # sort by window so each block pads to a similar width
    order = np.argsort(windows, kind="stable")
    for start in range(0, order.size, block_size):
        part = order[start : start + block_size]
        sizes[part], pre_levels[part] = _size_block(
            values,
            rows_f[part],
            cols[part],
            windows[part],
            previous[part],
            following[part],
            method,
        )
    return sizes, pre_levels


def additive_shift(shape, rows_f, cols, sizes):
    """Reverse cumulative sum of ``-size`` at each break (the ``lvlshft`` convention).

    History before a break carries ``-size``, so ``values - shift`` aligns all
    history to the latest regime.
    """
    increments = np.zeros(shape)
    np.add.at(increments, (rows_f, cols), -np.asarray(sizes, dtype=float))
    return np.cumsum(increments[::-1], axis=0)[::-1]


def ratio_shift(values, rows_f, cols, sizes, pre_levels, shift):
    """Safe multiplicative correction for columns additive would push below zero.

    Only non-negative columns whose additively corrected history goes negative
    switch to ratios ``(P + size) / P``. Returns ``(ratio_frame, ratio_columns)``
    where ``corrected = values / ratio_frame``; ratio_frame is 1 elsewhere.
    """
    values = np.asarray(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        non_negative = np.nanmin(values, axis=0) >= 0
        goes_negative = np.nanmin(values - shift, axis=0) < 0
    ratio_columns = non_negative & goes_negative
    ratio_frame = np.ones(values.shape)
    if not ratio_columns.any():
        return ratio_frame, ratio_columns
    use = ratio_columns[cols]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (pre_levels[use] + sizes[use]) / pre_levels[use]
    ratio = np.where(
        np.isfinite(ratio) & (ratio > 0) & (pre_levels[use] > 0), ratio, 1.0
    )
    log_increments = np.zeros(values.shape)
    np.add.at(log_increments, (rows_f[use], cols[use]), -np.log(ratio))
    ratio_frame = np.exp(np.cumsum(log_increments[::-1], axis=0)[::-1])
    return ratio_frame, ratio_columns
