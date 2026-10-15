"""Per-column peak selection for level shift candidates (numpy only).

Replaces the legacy progressive while-loop, whose row-index group ids were
shared across columns (a shift in one series removed candidates in others),
NaN at row 0 (stalling selection) and kept one shift too many.
"""

import numpy as np


def top_k_per_column(cols, scores, k):
    """Indices of the k highest-scoring entries per column (ties to the earlier entry)."""
    if k <= 0 or cols.size == 0:
        return np.array([], dtype=int)
    order = np.lexsort((-scores, cols))
    sorted_cols = cols[order]
    first = np.searchsorted(sorted_cols, sorted_cols, side="left")
    rank = np.arange(sorted_cols.size) - first
    return np.sort(order[rank < k])


def select_run_peaks(score, mask, tiebreak, max_shifts, tie_tolerance=0.0):
    """One row per contiguous run of ``mask`` per column, then top-k runs per column.

    In each run the winner is the argmax of ``score``; rows within
    ``tie_tolerance * peak`` of it are treated as tied and the one with the
    largest ``tiebreak`` wins (earliest on exact ties). The legacy window means
    tie exactly at the two rows either side of a clean step, so without this
    integer data flagged both and counted the step twice.

    Returns ``(rows, cols, run_scores)``.
    """
    T, C = mask.shape
    empty = np.array([], dtype=int)
    flat_mask = mask.T.ravel()
    positions = np.flatnonzero(flat_mask)
    if positions.size == 0 or max_shifts <= 0:
        return empty, empty, np.array([])
    starts = mask & ~np.vstack([np.zeros((1, C), dtype=bool), mask[:-1]])
    # column-major flattening keeps each column's runs contiguous and ordered
    run_starts = starts.T.ravel()[positions]
    run_index = np.cumsum(run_starts) - 1
    bounds = np.flatnonzero(run_starts)
    # gather by (row, col) rather than transposing whole T x C score arrays
    position_rows, position_cols = positions % T, positions // T
    run_score = score[position_rows, position_cols]
    run_score = np.where(np.isfinite(run_score), run_score, -np.inf)
    run_max = np.maximum.reduceat(run_score, bounds)
    peak = run_max[run_index]
    margin = np.where(np.isfinite(peak), tie_tolerance * np.abs(peak), 0.0)
    # every run keeps at least its argmax, so winners align one-to-one with run_max
    near = run_score >= peak - margin
    tie_value = np.where(near, tiebreak[position_rows, position_cols], -np.inf)
    tie_value = np.where(np.isnan(tie_value), -np.inf, tie_value)
    best_tie = np.maximum.reduceat(tie_value, bounds)
    is_best = near & (tie_value == best_tie[run_index])
    _, first = np.unique(run_index[np.flatnonzero(is_best)], return_index=True)
    winners = positions[np.flatnonzero(is_best)[first]]
    rows, cols = winners % T, winners // T
    keep = top_k_per_column(cols, run_max, max_shifts)
    return rows[keep], cols[keep], run_max[keep]


def thresholded_peaks(abs_score, threshold, half_width):
    """Rows above threshold that are their column's maximum within +-half_width.

    Equivalent to a dense +-half_width rolling-max test (rows below threshold
    can't dominate a peak above it) but sparse: one argmax per run above
    threshold, then run peaks closer than half_width suppress the lower one.
    Exact ties keep the earlier row. Returns ``(rows, cols)``.
    """
    above = abs_score > threshold
    rows, cols, scores = select_run_peaks(
        abs_score, above, abs_score, abs_score.shape[0], tie_tolerance=0.0
    )
    order = np.lexsort((rows, cols))
    rows, cols, scores = rows[order], cols[order], scores[order]
    keep = np.ones(rows.size, dtype=bool)
    # a few passes resolve chains of close peaks; each pass only compares
    # surviving neighbours, so it converges in at most a handful of steps
    for _ in range(rows.size):
        index = np.flatnonzero(keep)
        if index.size < 2:
            break
        r, c, v = rows[index], cols[index], scores[index]
        close = (c[1:] == c[:-1]) & (r[1:] - r[:-1] <= half_width)
        if not close.any():
            break
        later_wins = close & (v[1:] > v[:-1])
        earlier_wins = close & ~later_wins
        drop = np.zeros(index.size, dtype=bool)
        drop[:-1] |= later_wins
        drop[1:] |= earlier_wins
        keep[index[drop]] = False
    return rows[keep], cols[keep]


def near_taken(rows, cols, taken_rows, taken_cols, distance, T):
    """True for candidates with an accepted detection within +-distance in their column."""
    if taken_rows.size == 0 or rows.size == 0:
        return np.zeros(rows.size, dtype=bool)
    stride = T + 2 * distance + 1
    taken_keys = np.sort(taken_cols * stride + taken_rows)
    keys = cols * stride + rows
    lo = np.searchsorted(taken_keys, keys - distance, side="left")
    hi = np.searchsorted(taken_keys, keys + distance, side="right")
    return hi > lo
