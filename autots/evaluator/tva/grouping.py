# -*- coding: utf-8 -*-
"""Data-derived global / group factor structure.

Groups are found by consensus clustering (refit on random time/series
subsets, cluster loadings, keep pairs stable across refits) rather than
metadata, so a block only enters the model if it reproduces out of sample.
"""

import numpy as np

DEFAULT_GROUPING_CONFIG = {
    'refits': 8,
    'stability_threshold': 0.70,
    'block_frac': 0.7,
    'series_frac': 0.7,
    'cluster_similarity': 0.5,
    'min_group_size': 3,
    'residual_rank': 3,
    # without this floor, series with no block structure get swept into
    # whichever cluster is least separated, and consensus reports a spurious
    # group
    'min_explained': 0.25,
}


# ---------------------------------------------------------------------------
# clustering primitives
# ---------------------------------------------------------------------------


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    """Row-normalize to the unit sphere; zero rows stay zero.

    Discards loading magnitude — grouping is about which factor a series
    responds to, not how strongly.
    """
    arr = np.asarray(mat, dtype=float)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norm, out=np.zeros_like(arr), where=norm > 1e-12)


def average_linkage_clusters(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage agglomerative clustering on a similarity matrix.

    Implemented directly rather than via scipy/sklearn so this module runs in
    minimal environments. Merging stops when no two clusters have average
    similarity above ``threshold``.

    Args:
        sim: (n, n) symmetric similarity matrix, larger = more similar.
        threshold: minimum average similarity to merge.

    Returns:
        (n,) integer cluster labels, densely numbered from 0.
    """
    sim = np.asarray(sim, dtype=float).copy()
    n = sim.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    np.fill_diagonal(sim, -np.inf)
    members = {i: [i] for i in range(n)}
    active = np.ones(n, dtype=bool)
    while active.sum() > 1:
        masked = np.where(active[:, None] & active[None, :], sim, -np.inf)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        if not np.isfinite(masked[i, j]) or masked[i, j] < threshold:
            break
        ni, nj = len(members[i]), len(members[j])
        # average linkage update (Lance-Williams for the mean)
        new = (ni * sim[i] + nj * sim[j]) / float(ni + nj)
        sim[i] = new
        sim[:, i] = new
        sim[i, i] = -np.inf
        members[i] = members[i] + members[j]
        active[j] = False
    labels = np.full(n, -1, dtype=int)
    for lab, i in enumerate(np.where(active)[0]):
        labels[members[i]] = lab
    return labels


# ---------------------------------------------------------------------------
# stability-screened group discovery
# ---------------------------------------------------------------------------


def _residual_panel(values: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Panel with the (already fitted) global factor span projected out."""
    arr = np.asarray(values, dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    if factors is None or np.size(factors) == 0:
        return arr
    f = np.asarray(factors, dtype=float)
    f = f - f.mean(axis=0, keepdims=True)
    coef, *_ = np.linalg.lstsq(f, arr, rcond=None)
    return arr - f @ coef


def discover_groups(
    values: np.ndarray,
    global_factors=None,
    config: dict = None,
    seed: int = 42,
    **fit_kwargs,
) -> dict:
    """Find residual groups that survive a bootstrap reproduction test.

    Projects the global factors out, then refits a small residual factor
    model on ``refits`` random time/series draws, clustering signed loading
    vectors each time (signed because two series moving opposite a common
    factor are a hedge pair, not the same group). A group is a set of series
    whose co-membership frequency across refits clears ``threshold``.

    Args:
        values: (T, N) normalized panel.
        global_factors: (T, Kg) already-fitted global factor paths, or None.
        config: overrides for ``DEFAULT_GROUPING_CONFIG``.
        seed: RNG seed.
        **fit_kwargs: forwarded to ``estimate_factors_alternating``.

    Returns:
        dict with 'labels' (N,) group id per series (-1 = ungrouped),
        'groups' {gid: [series indices]}, 'co_membership' (N, N) frequency,
        'n_refits', and 'residual' (T, N) the panel groups were found in.
    """
    from autots.evaluator.tva.factor_network import estimate_factors_alternating

    cfg = dict(DEFAULT_GROUPING_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})

    arr = np.asarray(values, dtype=float)
    T, N = arr.shape
    resid = _residual_panel(arr, global_factors)
    labels = np.full(N, -1, dtype=int)
    if N < 2 * int(cfg['min_group_size']) or T < 30:
        return {
            'labels': labels,
            'groups': {},
            'co_membership': np.zeros((N, N)),
            'n_refits': 0,
            'residual': resid,
        }

    rng = np.random.default_rng(seed)
    together = np.zeros((N, N), dtype=float)
    seen = np.zeros((N, N), dtype=float)
    n_done = 0
    block = max(int(cfg['block_frac'] * T), 30)
    n_sub = max(int(cfg['series_frac'] * N), 2 * int(cfg['min_group_size']))
    rank = max(int(cfg['residual_rank']), 1)
    for _ in range(max(int(cfg['refits']), 1)):
        start = int(rng.integers(0, max(T - block, 1)))
        cols = np.sort(rng.choice(N, size=min(n_sub, N), replace=False))
        chunk = resid[start : start + block][:, cols]
        if chunk.shape[0] < 20:  # pragma: no cover - guarded by block floor
            continue
        try:
            fit = estimate_factors_alternating(
                chunk, n_factors=min(rank, chunk.shape[1] - 1), **fit_kwargs
            )
        except Exception:  # pragma: no cover - degenerate draw
            continue
        lam = _unit_rows(fit['loadings'])
        sim = lam @ lam.T
        sub_labels = average_linkage_clusters(sim, float(cfg['cluster_similarity']))
        # unexplained series are left unassigned, else the consensus matrix
        # invents a group out of the noise remainder
        centered = chunk - chunk.mean(axis=0, keepdims=True)
        recon = fit['factors'] @ fit['loadings'].T
        denom = np.maximum((centered**2).sum(axis=0), 1e-12)
        explained = 1.0 - ((centered - recon) ** 2).sum(axis=0) / denom
        # adaptive floor: r factors fit to m noise series still "explain"
        # ~r/m of each series' variance, so a fixed threshold isn't enough
        # on small subsets
        floor = max(
            float(cfg['min_explained']),
            2.0 * fit['loadings'].shape[1] / max(chunk.shape[1], 1),
        )
        sub_labels = np.where(explained >= floor, sub_labels, -1)
        same = (sub_labels[:, None] == sub_labels[None, :]) & (sub_labels[:, None] >= 0)
        idx = np.ix_(cols, cols)
        together[idx] += same.astype(float)
        seen[idx] += 1.0
        n_done += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        freq = np.divide(together, seen, out=np.zeros_like(together), where=seen > 0)
    np.fill_diagonal(freq, 1.0)
    # average linkage, not connected components: single-link chaining (a-b,
    # b-c stable => a,b,c one group) would merge unrelated blocks
    raw = average_linkage_clusters(freq, float(cfg['stability_threshold']))
    gid = 0
    groups = {}
    threshold = float(cfg['stability_threshold'])
    for lab in np.unique(raw):
        comp = np.where(raw == lab)[0]
        if comp.size < int(cfg['min_group_size']):
            continue
        block = freq[np.ix_(comp, comp)]
        internal = (block.sum() - np.trace(block)) / max(comp.size * (comp.size - 1), 1)
        if internal < threshold:
            continue
        labels[comp] = gid
        groups[gid] = sorted(int(c) for c in comp)
        gid += 1
    return {
        'labels': labels,
        'groups': groups,
        'co_membership': freq,
        'n_refits': n_done,
        'residual': resid,
    }


# ---------------------------------------------------------------------------
# rank selection by inner rolling-origin validation
# ---------------------------------------------------------------------------


def _damped_slope_forecast(
    path: np.ndarray, horizon: int, window: int = 90, phi: float = 0.9
) -> np.ndarray:
    """(H, K) damped local-linear continuation of each column of ``path``."""
    arr = np.atleast_2d(np.asarray(path, dtype=float))
    if arr.shape[0] == 0:
        return np.zeros((horizon, 0))
    seg = arr[-min(window, arr.shape[0]) :]
    t = np.arange(seg.shape[0], dtype=float)
    t = t - t.mean()
    denom = float((t**2).sum())
    slope = (
        (seg * t[:, None]).sum(axis=0) / denom
        if denom > 1e-12
        else np.zeros(arr.shape[1])
    )
    damp = np.cumsum(np.power(float(phi), np.arange(1, horizon + 1, dtype=float)))
    return arr[-1][None, :] + damp[:, None] * slope[None, :]


def rolling_origin_score(
    values: np.ndarray,
    rank: int,
    horizon: int,
    n_origins: int = 3,
    membership=None,
    n_global: int = 0,
    **fit_kwargs,
) -> float:
    """Held-out MASE of a rank-``r`` factor model, torch-free.

    Each origin refits on truncated history and extrapolates via the same
    damped local-linear rule the trend model uses, so this is not an
    in-sample criterion. Rank 0 degenerates to a residual-only continuation.

    Args:
        values: (T, N) normalized panel.
        rank: number of factors (0 allowed).
        horizon: forecast horizon per origin.
        n_origins: non-overlapping origins, most recent first.
        membership: optional (N,) group label array restricting a factor's
            loadings to its own group's series.
        **fit_kwargs: forwarded to ``estimate_factors_alternating``.

    Returns:
        Mean MASE over series and origins (lower is better); NaN if no origin
        has enough history.
    """
    from autots.evaluator.tva.factor_network import estimate_factors_alternating

    arr = np.asarray(values, dtype=float)
    T, N = arr.shape
    H = max(int(horizon), 1)
    season = 7
    if T > season:
        scale = np.nanmean(np.abs(arr[season:] - arr[:-season]), axis=0)
    else:  # pragma: no cover - tiny panels
        scale = np.nanmean(np.abs(np.diff(arr, axis=0)), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.nan)

    scores = []
    for i in range(max(int(n_origins), 1)):
        origin = T - 1 - H * (i + 1)
        if origin < max(180, 3 * H // 2):
            break
        train = arr[: origin + 1]
        actual = arr[origin + 1 : origin + 1 + H]
        if actual.shape[0] < H:  # pragma: no cover
            break
        fitted = np.zeros_like(train)
        future = np.zeros((H, N))
        if int(rank) > 0:
            try:
                fit = estimate_factors_alternating(
                    train, n_factors=int(rank), **fit_kwargs
                )
            except Exception:  # pragma: no cover
                continue
            f, lam = fit['factors'], fit['loadings']  # (t, K), (N, K)
            if membership is not None:
                lam = lam * _group_mask(membership, lam.shape, int(n_global))
            fitted = f @ lam.T
            f_future = _damped_slope_forecast(f, H) - f[-1][None, :]
            future = f_future @ lam.T
        # What the factors do not explain is carried by a GLOBAL [1, t] line,
        # damped forward -- deliberately a different functional from the
        # factors' local-slope continuation. Local slope and extrapolation are
        # both linear, so continuing the residual the same way would give
        # extrap(fitted) + extrap(y - fitted) == extrap(y): every rank scores
        # identically (measured: 1.7158 at ranks 0, 1, 2, 3, 4 and 6) and rank
        # selection silently becomes a coin flip.
        resid = train - fitted
        n_train = train.shape[0]
        t_train = np.arange(n_train, dtype=float) / float(n_train)
        design = np.column_stack([np.ones(n_train), t_train])
        line, *_ = np.linalg.lstsq(design, resid, rcond=None)
        damp = np.cumsum(np.power(0.9, np.arange(1, H + 1, dtype=float)))
        idio_last = line[0] + line[1] * t_train[-1]
        idio_future = (
            idio_last[None, :] + (line[1] / float(n_train))[None, :] * damp[:, None]
        )
        pred = fitted[-1][None, :] - idio_last[None, :] + future + idio_future
        mae = np.nanmean(np.abs(pred - actual), axis=0)
        with np.errstate(invalid='ignore'):
            scores.append(np.nanmean(mae / scale))
    return float(np.nanmean(scores)) if scores else float('nan')


def _group_mask(membership, shape, n_global=0):
    """(N, K) 0/1 mask: leading factors global, later ones tied to one group."""
    membership = np.asarray(membership, dtype=int)
    mask = np.zeros(shape, dtype=float)
    mask[:, : int(n_global)] = 1.0
    for k in range(int(n_global), shape[1]):
        mask[membership == (k - int(n_global)), k] = 1.0
    return mask


def select_rank(
    values: np.ndarray,
    candidates=(0, 1, 2, 3, 4, 6),
    horizon: int = 180,
    n_origins: int = 3,
    stability_threshold: float = 0.5,
    seed: int = 42,
    **fit_kwargs,
) -> dict:
    """Choose a factor rank by held-out error *and* split-half stability.

    In-sample fit is monotone in rank, so it can't choose one, and held-out
    error alone is noisy on short panels. Rank is the minimum held-out MASE
    among ranks whose split-half factor agreement clears
    ``stability_threshold``; rank 0 is always admissible.

    Args:
        values: (T, N) normalized panel.
        candidates: ranks to consider.
        horizon: inner validation horizon.
        n_origins: inner rolling origins.
        stability_threshold: minimum split-half ``match_factors`` score.
        seed: RNG seed for the split-half repetitions.
        **fit_kwargs: forwarded to ``estimate_factors_alternating``.

    Returns:
        dict with 'rank', 'table' [{'rank', 'score', 'stability', 'admissible'}].
    """
    from autots.evaluator.tva.factor_network import split_half_stability

    arr = np.asarray(values, dtype=float)
    table = []
    for r in candidates:
        r = int(r)
        if r >= arr.shape[1]:
            continue
        score = rolling_origin_score(arr, r, horizon, n_origins=n_origins, **fit_kwargs)
        if r == 0:
            stab = 1.0
        else:
            stab = split_half_stability(arr, r, n_reps=3, seed=seed, **fit_kwargs)
        admissible = r == 0 or (np.isfinite(stab) and stab >= stability_threshold)
        table.append(
            {
                'rank': r,
                'score': score,
                'stability': float(stab),
                'admissible': bool(admissible),
            }
        )
    ok = [row for row in table if row['admissible'] and np.isfinite(row['score'])]
    if not ok:
        return {'rank': 0, 'table': table}
    best = min(ok, key=lambda row: row['score'])
    return {'rank': int(best['rank']), 'table': table}


def loading_graph(
    loadings: np.ndarray, labels=None, series_names=None, factor_names=None
) -> dict:
    """The series x factor loading matrix as the panel's structural graph.

    A factor-to-series bipartite graph (N x K) instead of series->series
    (N^2): exposes the structure actually fit rather than a
    separately-estimated adjacency.

    Args:
        loadings: (N, K) signed loadings.
        labels: optional (N,) group assignment (-1 for ungrouped).
        series_names: optional list of series names.
        factor_names: optional list of factor names.

    Returns:
        dict with 'loadings', 'group_assignment', 'series', 'factors',
        'dominant_factor' (argmax |loading| per series).
    """
    lam = np.asarray(loadings, dtype=float)
    n, k = lam.shape if lam.ndim == 2 else (0, 0)
    dominant = (
        np.abs(lam).argmax(axis=1).astype(int) if n and k else np.full(n, -1, dtype=int)
    )
    return {
        'loadings': lam,
        'group_assignment': (
            np.asarray(labels, dtype=int)
            if labels is not None
            else np.full(n, -1, dtype=int)
        ),
        'series': list(series_names) if series_names is not None else None,
        'factors': (
            list(factor_names)
            if factor_names is not None
            else [f'factor_{i + 1}' for i in range(k)]
        ),
        'dominant_factor': dominant,
    }
