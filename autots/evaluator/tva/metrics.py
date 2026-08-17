"""Shared scoring metrics for the TVA benchmark / validation harnesses.

Kept dependency-light (numpy + pandas only) so harnesses, notebooks and CI
gate scripts import the same definitions instead of drifting private copies.
All reductions are NaN-tolerant; degenerate inputs return ``np.nan`` rather
than raising, since these run inside sweep harnesses where one bad cell must
not abort the run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "mase_value",
    "correlated_pairs",
    "sign_agreement",
    "dca_error",
    "directional_coherence",
    "net_change_direction",
    "direction_coherence",
    "direction_coherence_error",
    "real_data_coherence",
    "trend_only_coherence",
    "oracle_normalized_coherence",
]


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def mase_value(actual, forecast, train, m: int = 7) -> float:
    """Mean (over series) of MAE / in-sample seasonal-naive MAE.

    Scaling by seasonal-naive error makes the number comparable across
    datasets of different magnitude; the ``scale <= 1e-9`` guard drops
    constant series that would otherwise divide by ~zero and dominate.
    """
    actual = pd.DataFrame(actual)
    forecast = pd.DataFrame(forecast)
    train = pd.DataFrame(train)
    A = actual.values.astype(float)
    F = forecast.reindex(columns=actual.columns).values.astype(float)
    with np.errstate(invalid="ignore"):
        mae = _nanmean(np.abs(A - F), axis=0)
    tr = train.values.astype(float)
    if tr.shape[0] > m:
        scale = _nanmean(np.abs(tr[m:] - tr[:-m]), axis=0)
    elif tr.shape[0] > 1:
        scale = _nanmean(np.abs(np.diff(tr, axis=0)), axis=0)
    else:
        return float("nan")
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = mae / scale
    return float(_nanmean(ratio))


# ---------------------------------------------------------------------------
# Pairwise sign coherence
# ---------------------------------------------------------------------------


def correlated_pairs(train, season_m: int = 7, threshold: float = 0.5, max_pairs: int = 200) -> list:
    """Pairs of columns whose smoothed, differenced training trends correlate.

    Derived from history alone (never the future), so the ground-set pairs
    are usable to score any model. Smoothing before differencing suppresses
    seasonal/noise so correlation reflects shared low-frequency drift. Capped
    at ``max_pairs`` via a seed-0 RNG subsample for reproducibility.
    """
    train = pd.DataFrame(train)
    window = max(int(season_m), 3)
    smooth = train.rolling(window, min_periods=1).mean()
    diffs = smooth.diff().dropna(how="all")
    if diffs.empty:
        return []
    corr = diffs.corr()
    cols = list(train.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            try:
                c = corr.iloc[i, j]
            except (IndexError, KeyError):
                continue
            if np.isfinite(c) and c > threshold:
                pairs.append((cols[i], cols[j]))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[x] for x in idx]
    return pairs


def sign_agreement(df, pairs) -> dict:
    """Per-pair rate of first-difference sign agreement.

    The raw ingredient of :func:`dca_error`.  Reported per pair (rather than
    averaged immediately) so callers can compare the same pair between a
    forecast and the actual future.
    """
    df = pd.DataFrame(df)
    diffs = df.diff().dropna(how="all")
    out = {}
    for a, b in pairs:
        if a not in diffs.columns or b not in diffs.columns:
            continue
        da = np.sign(diffs[a].values.astype(float))
        db = np.sign(diffs[b].values.astype(float))
        valid = np.isfinite(da) & np.isfinite(db)
        if valid.sum() == 0:
            continue
        out[(a, b)] = float(np.mean(da[valid] == db[valid]))
    return out


def dca_error(forecast, actual, pairs) -> float:
    """Mean |sign-agreement(forecast) - sign-agreement(actual)| over pairs.

    Scored against the realised future rather than as a raw rate, since a
    model that forces lockstep is as wrong as one that scatters. Lower is
    better; ``np.nan`` when there are no usable pairs.
    """
    if not len(pairs):
        return float("nan")
    forecast = pd.DataFrame(forecast)
    actual = pd.DataFrame(actual)
    fc = sign_agreement(forecast.reindex(columns=actual.columns), pairs)
    ac = sign_agreement(actual, pairs)
    errs = [abs(fc[p] - ac[p]) for p in fc if p in ac]
    return float(np.mean(errs)) if errs else float("nan")


# ---------------------------------------------------------------------------
# Factor-loading coherence
# ---------------------------------------------------------------------------


def directional_coherence(forecast, loadings) -> float:
    """Fraction of same-dominant-factor, same-sign pairs forecast to agree.

    Scored against the known factor structure since MASE alone won't surface
    a mixed-direction forecast where everything should move together.

    Args:
        forecast: DataFrame, horizon rows x series columns.
        loadings: series (index) x factor (columns) loading matrix.

    Returns ``np.nan`` when loadings are empty or fewer than two series carry
    a loading -- undefined, not zero, for a factorless negative control.
    """
    if loadings is None:
        return float("nan")
    loadings_all = pd.DataFrame(loadings)
    if loadings_all.shape[0] == 0 or loadings_all.shape[1] == 0:
        return float("nan")
    forecast = pd.DataFrame(forecast)
    columns = [c for c in forecast.columns if c in loadings_all.index]
    if not columns:
        return float("nan")
    values = np.nan_to_num(loadings_all.loc[columns].values.astype(float))
    if values.size == 0:
        return float("nan")
    keep = np.abs(values).max(axis=1) > 0
    columns = [c for c, k in zip(columns, keep) if k]
    if len(columns) < 2:
        return float("nan")
    values = np.nan_to_num(loadings_all.loc[columns].values.astype(float))
    dom = np.abs(values).argmax(axis=1)
    signs = np.sign(values[np.arange(len(columns)), dom])

    direction = {}
    for name in columns:
        series = np.asarray(forecast[name].values, dtype=float)
        finite = series[np.isfinite(series)]
        direction[name] = float(np.sign(finite[-1] - finite[0])) if finite.size >= 2 else 0.0

    agree, total = 0, 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if dom[i] != dom[j] or signs[i] != signs[j]:
                continue
            total += 1
            if direction[columns[i]] == direction[columns[j]] != 0:
                agree += 1
    return float(agree / total) if total else float("nan")


def trend_only_coherence(trend_forecast, loadings) -> float:
    """:func:`directional_coherence` applied to a trend-only forecast path.

    Isolates the structural component: seasonality/holidays can flip a
    first->last comparison on a short horizon. Named separately so gate
    tables and result JSONs record which path was scored.
    """
    return directional_coherence(trend_forecast, loadings)


def oracle_normalized_coherence(forecast, actual, loadings) -> float:
    """Coherence relative to the metric's own ceiling on the real future.

    A perfect forecast doesn't score 1.0: the realised future only partly
    honours the factor structure once noise is added. Dividing by the actual
    future's score stops gates from chasing an unreachable number.

    Returns ``np.nan`` when the oracle denominator is NaN or <= 0.
    """
    denom = directional_coherence(actual, loadings)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    num = directional_coherence(forecast, loadings)
    if not np.isfinite(num):
        return float("nan")
    return float(num / denom)


# ---------------------------------------------------------------------------
# Net-change ("180 day") direction coherence
# ---------------------------------------------------------------------------


def net_change_direction(frame, window: int = 28) -> "pd.Series":
    """Sign of (mean of last ``window`` rows - mean of first ``window`` rows).

    Deliberately not last-point minus first-point: a single-point comparison
    is dominated by seasonality/noise and lets a near-flat forecast pick up a
    direction from an arbitrarily small difference.

    Shrinks the window to ``max(1, len(frame) // 2)`` when the frame has
    fewer than ``2 * window`` rows, so the two windows never overlap.

    Returns a ``pd.Series`` of -1.0 / 0.0 / 1.0 indexed by column
    (NaN for all-NaN columns).
    """
    frame = pd.DataFrame(frame)
    n = len(frame)
    if n == 0:
        return pd.Series(dtype=float, index=frame.columns)
    window = int(window)
    if window < 1:
        window = 1
    if n < 2 * window:
        window = max(1, n // 2)
    values = frame.values.astype(float)
    head = _nanmean(values[:window], axis=0)
    tail = _nanmean(values[-window:], axis=0)
    with np.errstate(invalid="ignore"):
        delta = tail - head
    out = np.where(np.isfinite(delta), np.sign(delta), np.nan)
    return pd.Series(out, index=frame.columns, dtype=float)


def direction_coherence(frame, pairs, window: int = 28) -> float:
    """Fraction of ``pairs`` whose net-change directions agree and are nonzero.

    The net-change analogue of :func:`sign_agreement`.  Zero-direction (flat)
    columns count toward the denominator but never toward the numerator, so
    "predict nothing moves" scores 0 rather than winning by abstention.
    ``np.nan`` when no pair is usable.
    """
    frame = pd.DataFrame(frame)
    if not len(pairs):
        return float("nan")
    dirs = net_change_direction(frame, window=window)
    agree, total = 0, 0
    for a, b in pairs:
        if a not in dirs.index or b not in dirs.index:
            continue
        da, db = dirs.get(a), dirs.get(b)
        if not (np.isfinite(da) and np.isfinite(db)):
            continue
        total += 1
        if da == db != 0:
            agree += 1
    return float(agree / total) if total else float("nan")


def direction_coherence_error(forecast, actual, pairs, window: int = 28) -> float:
    """|direction_coherence(forecast) - direction_coherence(actual)|.

    Net-change analogue of :func:`dca_error`: penalises both over- and
    under-coupled forecasts against the realised future. Lower is better.
    """
    fc = direction_coherence(forecast, pairs, window=window)
    ac = direction_coherence(actual, pairs, window=window)
    if not (np.isfinite(fc) and np.isfinite(ac)):
        return float("nan")
    return float(abs(fc - ac))


def real_data_coherence(forecast, actual, pairs, window: int = 28) -> float:
    """Recall of co-movement that actually happened.

    Among pairs whose actual net-change directions agreed, the fraction the
    forecast also agreed on -- conditional on truth, unlike
    :func:`direction_coherence_error`, so it reads as a plain success rate
    on real panels with no factor loadings.

    Returns ``np.nan`` when no actual pair agrees -- an undefined denominator,
    not a score of zero.
    """
    forecast = pd.DataFrame(forecast)
    actual = pd.DataFrame(actual)
    if not len(pairs):
        return float("nan")
    a_dirs = net_change_direction(actual, window=window)
    f_dirs = net_change_direction(forecast, window=window)
    agree, total = 0, 0
    for a, b in pairs:
        if a not in a_dirs.index or b not in a_dirs.index:
            continue
        aa, ab = a_dirs.get(a), a_dirs.get(b)
        if not (np.isfinite(aa) and np.isfinite(ab)):
            continue
        if not (aa == ab != 0):
            continue
        total += 1
        fa, fb = f_dirs.get(a, np.nan), f_dirs.get(b, np.nan)
        if np.isfinite(fa) and np.isfinite(fb) and fa == fb != 0:
            agree += 1
    return float(agree / total) if total else float("nan")


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _nanmean(arr, axis=None):
    """np.nanmean without the all-NaN-slice RuntimeWarning."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        if axis is None:
            return float("nan")
        shape = list(arr.shape)
        del shape[axis]
        return np.full(shape, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(arr, axis=axis)
