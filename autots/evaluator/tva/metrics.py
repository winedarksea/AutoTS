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
    "loading_structure_score",
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


def correlated_pairs(
    train, season_m: int = 7, threshold: float = 0.5, max_pairs: int = 200
) -> list:
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
    """Mean ``|sign-agreement(forecast) - sign-agreement(actual)|`` over pairs.

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
        direction[name] = (
            float(np.sign(finite[-1] - finite[0])) if finite.size >= 2 else 0.0
        )

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
    """``|direction_coherence(forecast) - direction_coherence(actual)|``.

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


# ---------------------------------------------------------------------------
# Loading-structure recovery
# ---------------------------------------------------------------------------


def _as_loading_array(loadings):
    """(N, K) float array from a DataFrame/array, non-finite entries zeroed."""
    if loadings is None:
        return np.zeros((0, 0), dtype=float)
    arr = np.asarray(
        loadings.values if hasattr(loadings, "values") else loadings, dtype=float
    )
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _greedy_column_match(true_lam, est_lam) -> dict:
    """``{true_col: (est_col, sign)}`` by greedy ``|correlation|`` on loadings.

    Fallback used when factor paths aren't available to
    :func:`~autots.evaluator.tva.discovery.match_factors`. Greedy rather than
    Hungarian because the fallback only has to be sane, not optimal, and this
    keeps ``metrics`` free of a scipy dependency.
    """
    n_true, n_est = true_lam.shape[1], est_lam.shape[1]
    if n_true == 0 or n_est == 0:
        return {}
    corr = np.zeros((n_true, n_est), dtype=float)
    for a in range(n_true):
        ta = true_lam[:, a]
        for b in range(n_est):
            eb = est_lam[:, b]
            if np.std(ta) > 1e-12 and np.std(eb) > 1e-12:
                with np.errstate(invalid="ignore"):
                    c = np.corrcoef(ta, eb)[0, 1]
                corr[a, b] = c if np.isfinite(c) else 0.0
    out = {}
    used_true, used_est = set(), set()
    for _ in range(min(n_true, n_est)):
        masked = np.abs(corr).copy()
        for a in used_true:
            masked[a, :] = -1.0
        for b in used_est:
            masked[:, b] = -1.0
        a, b = np.unravel_index(int(np.argmax(masked)), masked.shape)
        if masked[a, b] < 0:
            break
        out[int(a)] = (int(b), 1.0 if corr[a, b] >= 0 else -1.0)
        used_true.add(int(a))
        used_est.add(int(b))
    return out


def _same_group_pairs(lam, active=None) -> set:
    """Unordered ``(i, j)`` pairs sharing a dominant factor *and* its sign.

    The relation the coherence graph is trying to assert: series that load the
    same factor the same way should move together. Series with no exposure
    (all-zero row) are excluded rather than pooled into a spurious group.
    """
    n, k = lam.shape
    if n < 2 or k == 0:
        return set()
    magnitude = np.abs(lam)
    exposed = magnitude.max(axis=1) > 0
    if active is not None:
        exposed = exposed & np.asarray(active, dtype=bool)
    dom = magnitude.argmax(axis=1)
    sign = np.sign(lam[np.arange(n), dom])
    pairs = set()
    for i in range(n):
        if not exposed[i]:
            continue
        for j in range(i + 1, n):
            if not exposed[j]:
                continue
            if dom[i] == dom[j] and sign[i] == sign[j]:
                pairs.add((i, j))
    return pairs


def _normalize_asserted(asserted_pairs) -> set:
    """``{(i, j)}`` with ``i < j`` from ``(i, j)`` or ``(i, j, sign)`` tuples.

    Signed graphs also carry ``-1`` links, which assert *opposite* movement —
    not the same-factor-same-sign relation being scored — so they are dropped
    rather than counted as (necessarily wrong) positive assertions.
    """
    out = set()
    for item in asserted_pairs or ():
        try:
            if len(item) >= 3 and float(item[2]) < 0:
                continue
            i, j = int(item[0]), int(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if i == j:
            continue
        out.add((min(i, j), max(i, j)))
    return out


def loading_structure_score(
    true_loadings,
    est_loadings,
    true_factors=None,
    est_factors=None,
    asserted_pairs=None,
) -> dict:
    """Score how well estimated loadings recover the true loading *structure*.

    Canonical correlation says whether the factor *span* was found; this says
    whether the *basis within the span* is the true one. A rotation of the
    correct span scores near chance here while scoring ~1.0 on span metrics,
    which is exactly the failure mode the coherence graph is built on top of.

    Args:
        true_loadings: (N, K_true) generator loadings (DataFrame or array).
        est_loadings: (N, K_est) fitted loadings, **same series order**.
        true_factors: optional (T, K_true) true factor paths.
        est_factors: optional (T, K_est) estimated factor paths. When both are
            supplied, columns are matched with ``discovery.match_factors``
            (Hungarian on differenced ``|corr|``); otherwise a greedy ``|corr|`` match
            on the loading columns themselves is used.
        asserted_pairs: optional iterable of ``(i, j)`` or ``(i, j, sign)``
            series-index pairs the caller's graph asserts as same-group (e.g.
            ``coherence._graph_pairs(group_graph(est_loadings, cfg), N)``).
            Defaults to the pairs implied by the estimated loadings' own
            (dominant factor, sign) partition.

    Returns:
        dict with ``pair_precision``, ``pair_recall``, ``pair_f1``,
        ``n_pairs_asserted``, ``n_pairs_true``, ``dominant_recovery``,
        ``dominant_recovery_matched``, ``sign_agreement``,
        ``matched_loading_corr``, ``n_true``, ``n_est``.
        Undefined quantities are ``np.nan`` (counts are ``0``); never raises.
    """
    empty = {
        "pair_precision": float("nan"),
        "pair_recall": float("nan"),
        "pair_f1": float("nan"),
        "n_pairs_asserted": 0,
        "n_pairs_true": 0,
        "dominant_recovery": float("nan"),
        "dominant_recovery_matched": float("nan"),
        "sign_agreement": float("nan"),
        "matched_loading_corr": float("nan"),
        "n_true": 0,
        "n_est": 0,
    }
    try:
        return _loading_structure_score(
            true_loadings,
            est_loadings,
            true_factors,
            est_factors,
            asserted_pairs,
            empty,
        )
    except Exception:  # pragma: no cover - harness metric, never fatal
        return dict(empty)


def _loading_structure_score(
    true_loadings, est_loadings, true_factors, est_factors, asserted_pairs, empty
) -> dict:
    true_lam = _as_loading_array(true_loadings)
    est_lam = _as_loading_array(est_loadings)
    out = dict(empty)
    out["n_true"] = int(true_lam.shape[1])
    out["n_est"] = int(est_lam.shape[1])
    if true_lam.size == 0 or est_lam.size == 0:
        return out
    n = min(true_lam.shape[0], est_lam.shape[0])
    if n < 2:
        return out
    true_lam, est_lam = true_lam[:n], est_lam[:n]

    # ---- column matching -------------------------------------------------
    assignment = {}
    if true_factors is not None and est_factors is not None:
        from autots.evaluator.tva.discovery import match_factors

        score = match_factors(true_factors, est_factors)
        for t_idx, e_idx in (score.get("assignment") or {}).items():
            sign = float(score.get("signs", {}).get(t_idx, 1.0)) or 1.0
            assignment[int(t_idx)] = (int(e_idx), 1.0 if sign >= 0 else -1.0)
    if not assignment:
        assignment = _greedy_column_match(true_lam, est_lam)
    if not assignment:
        return out

    # est columns expressed in the true basis (unmatched true columns -> 0)
    est_aligned = np.zeros_like(true_lam)
    inverse = {}
    for t_idx, (e_idx, sign) in assignment.items():
        if t_idx >= true_lam.shape[1] or e_idx >= est_lam.shape[1]:
            continue
        est_aligned[:, t_idx] = sign * est_lam[:, e_idx]
        inverse[e_idx] = t_idx

    # ---- per-column loading correlation ----------------------------------
    corrs = []
    for t_idx in sorted(assignment):
        if t_idx >= true_lam.shape[1]:
            continue
        a, b = true_lam[:, t_idx], est_aligned[:, t_idx]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            with np.errstate(invalid="ignore"):
                c = np.corrcoef(a, b)[0, 1]
            if np.isfinite(c):
                corrs.append(float(c))
    if corrs:
        out["matched_loading_corr"] = float(np.mean(corrs))

    # ---- dominance / sign recovery, over exposed series only -------------
    true_mag = np.abs(true_lam)
    exposed = true_mag.max(axis=1) > 0
    if exposed.any():
        dom_true = true_mag.argmax(axis=1)
        # est dominance is judged in the estimated basis and then mapped back,
        # so a series whose strongest loading sits on a spurious (unmatched)
        # column counts as a miss instead of silently falling through to the
        # largest matched column.
        est_mag = np.abs(est_lam)
        dom_est_own = est_mag.argmax(axis=1)
        mapped = np.array([inverse.get(int(d), -1) for d in dom_est_own], dtype=int)
        mapped = np.where(est_mag.max(axis=1) > 0, mapped, -1)
        out["dominant_recovery"] = float(np.mean(mapped[exposed] == dom_true[exposed]))
        # The charitable variant: dominance judged only among the columns that
        # matched a true factor. Reported alongside because it is the number a
        # K-correct fit would produce anyway, and because it isolates "wrong
        # factor" from "dominated by a spurious extra factor" when K is over-
        # specified -- the two diverge only when n_est > n_true.
        if est_aligned.shape[1]:
            aligned_mag = np.abs(est_aligned)
            dom_aligned = aligned_mag.argmax(axis=1)
            dom_aligned = np.where(aligned_mag.max(axis=1) > 0, dom_aligned, -1)
            out["dominant_recovery_matched"] = float(
                np.mean(dom_aligned[exposed] == dom_true[exposed])
            )

        rows = np.arange(n)
        s_true = np.sign(true_lam[rows, dom_true])
        s_est = np.sign(est_aligned[rows, dom_true])
        usable = exposed & (s_true != 0) & (s_est != 0)
        if usable.any():
            out["sign_agreement"] = float(np.mean(s_true[usable] == s_est[usable]))

    # ---- pair precision / recall ----------------------------------------
    truth_pairs = _same_group_pairs(true_lam)
    if asserted_pairs is None:
        asserted = _same_group_pairs(est_lam)
    else:
        asserted = {(i, j) for i, j in _normalize_asserted(asserted_pairs) if j < n}
    out["n_pairs_true"] = len(truth_pairs)
    out["n_pairs_asserted"] = len(asserted)
    hits = len(asserted & truth_pairs)
    if asserted:
        out["pair_precision"] = float(hits / len(asserted))
    if truth_pairs:
        out["pair_recall"] = float(hits / len(truth_pairs))
    p, r = out["pair_precision"], out["pair_recall"]
    if np.isfinite(p) and np.isfinite(r) and (p + r) > 0:
        out["pair_f1"] = float(2 * p * r / (p + r))
    elif np.isfinite(p) and np.isfinite(r):
        out["pair_f1"] = 0.0
    return out
