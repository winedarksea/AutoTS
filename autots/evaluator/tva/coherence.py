# -*- coding: utf-8 -*-
"""Post-forecast coherence correction for the TVA factor trend mode.

Pulls per-series trend forecasts back toward their shared factor consensus,
as a *selected* correction (never assumed) applied to the trend block only —
seasonality/holidays/anomalies/level-shifts are the caller's to re-add
untouched. Loading signs are factor-model-indeterminate, so ``resolve_signs``
picks an orientation from the whole panel (not one series) and reports a
confidence that gates whether ``coherence_shrink`` may act on that factor.

``strength=0.0`` is always a bitwise identity, and nothing here raises —
degenerate inputs (one series, all-zero/NaN loadings, empty or singular
graphs) just return the input unchanged.
"""

from __future__ import annotations

import numpy as np

DEFAULT_COHERENCE_CONFIG = {
    # candidate graph family: 'group' | 'laplacian' | 'auto' | 'none'
    'graph': 'auto',
    # shrinkage strengths offered to the selector. 0.0 is mandatory and is
    # re-inserted if a caller drops it -- "do nothing" must stay reachable.
    'strengths': (0.0, 0.01, 0.03, 0.1, 0.3, 1.0),
    # neighbour counts for the laplacian candidates
    'neighbors': (3, 5, 8),
    # reserved for a factor-stability veto supplied by split_half_stability
    'stability_threshold': 0.7,
    # accuracy veto: reject any candidate costing more than this fraction of
    # the strength=0 baseline's scaled MAE
    'mase_guardrail': 0.02,
    # minimum per-factor sign confidence required to shrink along that factor
    'min_sign_confidence': 0.0,
    # minimum within-group directional agreement required to shrink a group
    'decisiveness_floor': 0.0,
    # ---- C2: graph abstention (defaults are exact no-ops) ------------------
    # A series joins its dominant factor's group only if that dominance is
    # decisive. Precision matters more than recall here: a wrongly-grouped
    # series is actively pulled the wrong way, whereas an abstaining one is
    # merely left alone. Both tests are on the series' own loading row, so a
    # failing series simply does not join any group.
    # |lam[i, dom]| >= margin * (second largest |lam[i, .]|)
    'dominance_margin': 1.0,
    # lam[i, dom]**2 / ||lam[i]||**2 >= share  (fraction of communality)
    'min_loading_share': 0.0,
    # ---- secondary knobs ---------------------------------------------------
    # series indices to leave completely alone (the factor gates' output)
    'gated': (),
    # per-series residual scale for the precision weights; None -> ones
    'sigma': None,
    # (K,) per-factor stability in [0, 1] (factor_network.split_half_factor_
    # stability); multiplied into the sign confidence, so an unreplicable
    # factor falls below min_sign_confidence. None -> no stability veto.
    'stability': None,
    # cosine-similarity floor for a laplacian link to be kept at all
    'min_similarity': 0.0,
    # endpoint window for net-direction; None -> max(1, H // 4)
    'direction_window': None,
    # ---- user graph prior (inert without a prior) --------------------------
    # blend weight of a supplied (N, N) signed prior into the loading-cosine
    # similarity before the kNN step. The blended graphs are offered to
    # select_coherence as *additional* candidates, so a wrong prior simply
    # loses to the unblended graph on held-out origins.
    'prior_weight': 0.5,
    'prior_candidates': True,
}


def _merged_config(config=None) -> dict:
    """Caller config layered over the defaults (callers may pass partials)."""
    merged = dict(DEFAULT_COHERENCE_CONFIG)
    if config:
        merged.update({k: v for k, v in dict(config).items() if v is not None})
    return merged


def _clean_loadings(loadings) -> np.ndarray:
    """(N, K) float loadings with non-finite entries zeroed.

    Zero, not drop: a NaN column would otherwise poison every cosine
    similarity, and zero matches what the factor gates mean by "no exposure".
    """
    arr = np.asarray(loadings, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def resolve_signs(loadings, factors=None, stability=None):
    """Orient each factor by its loading *mass*, and report the margin.

    Unlike the single-largest-loading convention, this votes with every
    series (``sign(sum_i lambda_ik * |lambda_ik|)``), so one outlier loading
    can't flip the orientation. The margin of that vote becomes the
    confidence, so a near-50/50 factor can be flagged as having no reliable
    orientation.

    Args:
        loadings: (N, K) loading matrix (or (N,) for a single factor).
        factors: optional (T, K) factor paths, used only to break an exact
            zero-mass tie via the factor's own net drift.
        stability: optional (K,) per-factor stability in [0, 1], multiplied
            into the confidence.

    Returns:
        ``(signed_loadings, confidence)`` — the (N, K) loadings with each
        column multiplied by its resolved sign, and a (K,) array of
        ``|sum(w * lambda)| / sum(|w * lambda|)`` in [0, 1].
    """
    lam = _clean_loadings(loadings)
    if lam.size == 0:
        return lam, np.zeros(lam.shape[1] if lam.ndim == 2 else 0, dtype=float)

    mass = lam * np.abs(lam)  # w_i * lambda_i with w_i = |lambda_i|
    num = mass.sum(axis=0)
    den = np.abs(mass).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        confidence = np.where(den > 0, np.abs(num) / np.maximum(den, 1e-300), 0.0)
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    confidence = np.clip(confidence, 0.0, 1.0)

    signs = np.sign(num)
    if factors is not None:
        # exact ties only: break via net drift rather than index order
        paths = np.nan_to_num(np.asarray(factors, dtype=float))
        if paths.ndim == 2 and paths.shape[1] == lam.shape[1] and paths.shape[0] > 1:
            drift = np.sign(paths[-1] - paths[0])
            signs = np.where(signs == 0, drift, signs)
    signs = np.where(signs == 0, 1.0, signs)

    if stability is not None:
        stab = np.asarray(stability, dtype=float).ravel()
        if stab.size == confidence.size:
            confidence = confidence * np.clip(np.nan_to_num(stab, nan=0.0), 0.0, 1.0)
    return lam * signs[np.newaxis, :], confidence


def _config_stability(cfg: dict, n_factors: int):
    """``(K,)`` stability vector from config, or None when unusable.

    Silently ignores a wrong-length vector rather than raising: the fit that
    produced it may have selected a different rank than the graph being built,
    and a stale veto is worse than none.
    """
    stability = cfg.get('stability')
    if stability is None:
        return None
    try:
        arr = np.asarray(stability, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size != int(n_factors):
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def _precision_weights(lam: np.ndarray, cfg: dict) -> np.ndarray:
    """(N,) ``|lambda| / sigma`` precision weights, normalized to mean 1.

    A large, well-measured loading is held near its own forecast; a weak or
    noisy one moves more freely toward consensus. Mean-1 normalization keeps
    ``strength`` comparable across panels.
    """
    n_series = lam.shape[0]
    dom = np.abs(lam).max(axis=1) if lam.shape[1] else np.zeros(n_series)
    sigma = cfg.get('sigma')
    if sigma is None:
        sig = np.ones(n_series, dtype=float)
    else:
        sig = np.asarray(sigma, dtype=float).ravel()
        if sig.size != n_series:
            sig = np.ones(n_series, dtype=float)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0)
    w = dom / sig
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    mean = float(w.mean()) if w.size else 0.0
    if mean <= 0:
        return np.ones(n_series, dtype=float)
    w = w / mean
    return np.maximum(w, 0.05)


def group_graph(loadings, config=None) -> dict:
    """Block-structured graph: one group per (dominant factor, loading sign).

    Grouping by dominant factor *and* sign keeps every in-group link a ``+1``
    link — series loading the same factor with opposite signs are supposed to
    diverge, not be pulled together. Factors below ``min_sign_confidence`` are
    dropped: without a trusted orientation, "same sign" isn't meaningful.

    ``dominance_margin`` and ``min_loading_share`` let a series abstain when
    its dominant factor isn't decisive; both default to exact no-ops.

    Args:
        loadings: (N, K) loading matrix (raw; signs are resolved internally).
        config: overrides for ``DEFAULT_COHERENCE_CONFIG``. Reads ``gated``
            (series indices to exclude), ``sigma``, ``min_sign_confidence``,
            ``dominance_margin`` and ``min_loading_share``.

    Returns:
        dict with

        * ``groups``: ``{group_key: [series indices]}``, keys are strings
          ``'f{k}+'`` / ``'f{k}-'`` so the result is JSON-safe.
        * ``weights``: (N,) precision weights.
        * ``confidence``: (K,) per-factor sign confidence.
        * ``signs``: (K,) resolved orientation per factor.
        * ``kind``: ``'group'``.
    """
    cfg = _merged_config(config)
    lam_raw = _clean_loadings(loadings)
    empty = {
        'groups': {},
        'weights': np.ones(max(lam_raw.shape[0], 0), dtype=float),
        'confidence': np.zeros(lam_raw.shape[1] if lam_raw.ndim == 2 else 0),
        'signs': np.ones(lam_raw.shape[1] if lam_raw.ndim == 2 else 0),
        'kind': 'group',
    }
    if lam_raw.size == 0 or lam_raw.shape[0] < 2 or lam_raw.shape[1] == 0:
        return empty

    lam, confidence = resolve_signs(
        lam_raw, stability=_config_stability(cfg, lam_raw.shape[1])
    )
    signs = np.sign((lam_raw * np.abs(lam_raw)).sum(axis=0))
    signs = np.where(signs == 0, 1.0, signs)
    weights = _precision_weights(lam, cfg)

    gated_idx = np.atleast_1d(np.asarray(cfg.get('gated') or [], dtype=int)).ravel()
    gated = set(int(i) for i in gated_idx)
    min_conf = float(cfg.get('min_sign_confidence') or 0.0)

    magnitude = np.abs(lam)
    dom = magnitude.argmax(axis=1)
    margin = float(cfg.get('dominance_margin') or 1.0)
    min_share = float(cfg.get('min_loading_share') or 0.0)
    # squared row norm for the communality-share test; guarded against the
    # all-zero row, which the exposure check below rejects anyway
    row_energy = np.sum(magnitude**2, axis=1)
    groups: dict = {}
    for i in range(lam.shape[0]):
        if i in gated:
            continue
        k = int(dom[i])
        top = magnitude[i, k]
        if top <= 0:
            continue  # no measured exposure: nothing to be coherent with
        if confidence[k] < min_conf:
            continue  # coin-flip orientation: refuse to partition on it
        if margin > 1.0 and lam.shape[1] > 1:
            runner_up = float(np.partition(magnitude[i], -2)[-2])
            if top < margin * runner_up:
                continue  # ambiguous dominance: abstain rather than guess
        if min_share > 0.0:
            energy = float(row_energy[i])
            if energy <= 0 or (top * top) / energy < min_share:
                continue  # exposure too spread out to call a dominant factor
        key = 'f{}{}'.format(k, '+' if lam[i, k] >= 0 else '-')
        groups.setdefault(key, []).append(i)
    groups = {k: v for k, v in sorted(groups.items()) if len(v) >= 2}
    return {
        'groups': groups,
        'weights': weights,
        'confidence': confidence,
        'signs': signs,
        'kind': 'group',
    }


def _symmetric_prior(prior, n: int):
    """(N, N) signed symmetric hollow view of a user prior, or None."""
    if prior is None:
        return None
    arr = np.asarray(prior, dtype=float)
    if arr.ndim != 2 or arr.shape != (n, n) or n == 0:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 0.0)
    peak = float(np.abs(arr).max()) if arr.size else 0.0
    if peak <= 0:
        return None
    if peak > 1.0:
        arr = arr / peak
    return arr


def laplacian_graph(
    loadings, config=None, n_neighbors: int = 5, prior=None
) -> np.ndarray:
    """Signed k-nearest-neighbour graph over sign-resolved loading vectors.

    General form for panels with no clean dominant-factor partition
    (overlapping exposures, more factors than the group view expresses).
    Links are cosine similarity of loading vectors (+1 same profile, -1
    opposite), keeping each node's ``n_neighbors`` strongest before
    symmetrizing — negative links stay meaningful so anti-correlated pairs get
    pulled apart, not together.

    Series below ``min_sign_confidence`` on their dominant factor, and series
    in ``gated``, are isolated (links zeroed) at build time.

    Args:
        loadings: (N, K) loading matrix.
        config: overrides for ``DEFAULT_COHERENCE_CONFIG``.
        n_neighbors: links retained per node before symmetrization.
        prior: optional (N, N) signed user prior, blended into the cosine
            similarity at ``config['prior_weight']`` before the kNN step.
            Both are signed, so a negative prior link pushes apart natively.

    Returns:
        (N, N) signed, symmetric, hollow weighted adjacency. All-zero when the
        graph is degenerate, which every consumer reads as "do nothing".
    """
    cfg = _merged_config(config)
    lam_raw = _clean_loadings(loadings)
    n = lam_raw.shape[0] if lam_raw.ndim == 2 else 0
    if n < 2 or lam_raw.shape[1] == 0:
        return np.zeros((max(n, 0), max(n, 0)), dtype=float)

    lam, confidence = resolve_signs(
        lam_raw, stability=_config_stability(cfg, lam_raw.shape[1])
    )
    norms = np.linalg.norm(lam, axis=1)
    good = norms > 0
    unit = np.zeros_like(lam)
    unit[good] = lam[good] / norms[good][:, np.newaxis]
    cos = unit @ unit.T
    np.fill_diagonal(cos, 0.0)
    cos = np.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)

    prior_sym = _symmetric_prior(prior, n)
    if prior_sym is not None:
        w_prior = float(cfg.get('prior_weight') or 0.0)
        if w_prior > 0:
            w_prior = min(w_prior, 1.0)
            cos = (1.0 - w_prior) * cos + w_prior * prior_sym
            np.fill_diagonal(cos, 0.0)

    floor = float(cfg.get('min_similarity') or 0.0)
    if floor > 0:
        cos = np.where(np.abs(cos) >= floor, cos, 0.0)

    k = int(max(min(int(n_neighbors), n - 1), 1))
    mask = np.zeros((n, n), dtype=bool)
    order = np.argsort(-np.abs(cos), axis=1, kind='stable')
    rows = np.repeat(np.arange(n), k)
    mask[rows, order[:, :k].ravel()] = True
    mask |= mask.T  # symmetrize by union: a link either node kept is a link
    adj = np.where(mask, cos, 0.0)

    isolate = ~good
    min_conf = float(cfg.get('min_sign_confidence') or 0.0)
    if min_conf > 0 and confidence.size:
        dom = np.abs(lam).argmax(axis=1)
        isolate = isolate | (confidence[dom] < min_conf)
    gated = np.atleast_1d(np.asarray(cfg.get('gated') or [], dtype=int)).ravel()
    if gated.size:
        keep = gated[(gated >= 0) & (gated < n)]
        isolate = isolate.copy()
        isolate[keep] = True
    if isolate.any():
        adj[isolate, :] = 0.0
        adj[:, isolate] = 0.0

    adj = 0.5 * (adj + adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj


def block_adjacency(graph, n_series: int) -> np.ndarray:
    """(N, N) block-constant adjacency for a ``group_graph`` result.

    A group is exactly a clique of ``+1`` links, so this makes the group and
    Laplacian views share one solver instead of two.

    Args:
        graph: a ``group_graph`` result dict, or a plain ``{key: [indices]}``.
        n_series: N.

    Returns:
        (N, N) symmetric hollow adjacency of 0.0 / 1.0.
    """
    groups = graph.get('groups', graph) if isinstance(graph, dict) else {}
    n = int(max(n_series, 0))
    adj = np.zeros((n, n), dtype=float)
    for members in (groups or {}).values():
        idx = np.asarray([i for i in members if 0 <= int(i) < n], dtype=int)
        if idx.size < 2:
            continue
        adj[np.ix_(idx, idx)] = 1.0
    np.fill_diagonal(adj, 0.0)
    return adj


def net_direction(paths: np.ndarray, window: int = None) -> np.ndarray:
    """(N,) sign of the windowed end-to-end change of each column.

    Windowed rather than last-minus-first so a single noisy endpoint can't
    decide the direction. A genuinely flat column returns exactly 0.
    """
    arr = np.asarray(paths, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    h = arr.shape[0]
    if h < 2:
        return np.zeros(arr.shape[1], dtype=float)
    w = int(window) if window else max(1, h // 4)
    w = max(1, min(w, h // 2))
    with np.errstate(invalid='ignore'):
        head = np.nanmean(arr[:w], axis=0)
        tail = np.nanmean(arr[-w:], axis=0)
        delta = tail - head
    return np.where(np.isfinite(delta), np.sign(delta), 0.0)


def _decisive_groups(trend_fc: np.ndarray, groups: dict, floor: float, window=None):
    """Split ``groups`` into those that may be shrunk and those that may not.

    Shrinking a group whose members genuinely disagree doesn't make them
    coherent, it makes them flat — the consensus of an evenly split group is
    ~zero movement. Only shrink a group that already leans one way.
    """
    keep, skipped = {}, []
    if not groups:
        return keep, skipped
    floor = float(floor or 0.0)
    dirs = net_direction(trend_fc, window)
    for key in sorted(groups):
        members = [int(i) for i in groups[key] if 0 <= int(i) < dirs.size]
        if len(members) < 2:
            skipped.append(key)
            continue
        d = dirs[np.asarray(members, dtype=int)]
        nz = d[d != 0]
        if nz.size == 0:
            agreement = 0.0
        else:
            up = float((nz > 0).sum()) / float(nz.size)
            agreement = max(up, 1.0 - up)
        if floor > 0 and agreement < floor:
            skipped.append(key)
            continue
        keep[key] = members
    return keep, skipped


def _resolve_graph(trend_fc: np.ndarray, graph, cfg: dict):
    """Normalize any graph form to ``(adjacency, weights, skipped_groups)``.

    Accepts a ``group_graph`` dict or a plain (N, N) adjacency, so
    ``coherence_shrink`` has exactly one code path below this point.
    """
    n = trend_fc.shape[1]
    skipped: list = []
    weights = None
    if isinstance(graph, dict):
        groups = graph.get('groups') or {}
        conf = np.asarray(graph.get('confidence', []), dtype=float).ravel()
        min_conf = float(cfg.get('min_sign_confidence') or 0.0)
        if min_conf > 0 and conf.size:
            # sign gate enforced here for the grouped view: an
            # unconfidently-oriented factor's group must not be pulled together
            kept = {}
            for key, members in groups.items():
                try:
                    k = int(key[1:-1])
                except (ValueError, TypeError):
                    kept[key] = members
                    continue
                if k < conf.size and conf[k] < min_conf:
                    skipped.append(key)
                    continue
                kept[key] = members
            groups = kept
        groups, dropped = _decisive_groups(
            trend_fc,
            groups,
            cfg.get('decisiveness_floor'),
            cfg.get('direction_window'),
        )
        skipped = sorted(set(skipped) | set(dropped))
        adj = block_adjacency({'groups': groups}, n)
        w = graph.get('weights')
        if w is not None:
            w = np.asarray(w, dtype=float).ravel()
            if w.size == n:
                weights = w
    else:
        adj = np.asarray(graph, dtype=float)
        if adj.ndim != 2 or adj.shape != (n, n):
            adj = np.zeros((n, n), dtype=float)
        adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        adj = 0.5 * (adj + adj.T)
        np.fill_diagonal(adj, 0.0)
    return adj, weights, skipped


def coherence_shrink(trend_fc, graph, strength, weights=None, config=None):
    """Pull standardized trend *changes* toward their graph consensus.

    Minimizes, independently at each horizon step,

    ``||dx - dx0||^2_W + strength * sum_ij |A_ij| (s_ij * dx_i - dx_j)^2``

    where ``dx`` is each series' cumulative trend change from the forecast
    anchor, ``s_ij = sign(A_ij)``, and ``W = diag(weights)``. This reduces to
    the signed graph Laplacian ``L = diag(sum_j |A_ij|) - A`` and normal
    equations ``(W + strength * L) dx = W dx0`` — one symmetric solve per
    forecast, shared across horizon steps since ``W``/``L`` don't depend on h.

    Shrinks changes, not levels, so the forecast anchor (row 0, where
    ``dx0 = 0``) is structurally preserved for the safety layer's re-anchoring
    to own. The group form is this same solve on a block-constant adjacency
    (``block_adjacency``), so both views share one solver.

    Args:
        trend_fc: (H, N) standardized trend forecast only — seasonality,
            holidays, anomalies and level-shift intercepts must not be in
            here; the factor structure has no claim on them.
        graph: a ``group_graph`` dict or an (N, N) signed adjacency.
        strength: shrinkage weight. ``0.0`` returns the input unchanged,
            bitwise.
        weights: optional (N,) precision weights, overriding the graph's.
        config: overrides for ``DEFAULT_COHERENCE_CONFIG``.

    Returns:
        (H, N) adjusted trend forecast. The input itself on any degenerate
        case: fewer than two series, an empty or all-zero graph, a
        non-finite panel, or a solve that fails.
    """
    if strength is None or float(strength) == 0.0:
        return trend_fc  # bitwise identity, by construction
    try:
        arr = np.asarray(trend_fc, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 2:
            return trend_fc
        if not np.isfinite(arr).all():
            return trend_fc
        cfg = _merged_config(config)
        adj, graph_weights, _skipped = _resolve_graph(arr, graph, cfg)
        if adj.size == 0 or not np.any(adj):
            return trend_fc

        n = arr.shape[1]
        if weights is not None:
            w = np.asarray(weights, dtype=float).ravel()
        elif graph_weights is not None:
            w = graph_weights
        else:
            w = np.ones(n, dtype=float)
        if w.size != n:
            w = np.ones(n, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

        degree = np.abs(adj).sum(axis=1)
        lap = np.diag(degree) - adj
        mat = np.diag(w) + float(strength) * lap

        dx0 = arr - arr[0][np.newaxis, :]  # changes from the anchor
        rhs = (w[np.newaxis, :] * dx0).T  # (N, H)
        try:
            sol = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:  # pragma: no cover - singular Laplacian
            sol = np.linalg.lstsq(mat, rhs, rcond=None)[0]
        out = arr[0][np.newaxis, :] + sol.T
        if not np.isfinite(out).all():
            return trend_fc
        return out
    except Exception:  # pragma: no cover - never break a forecast
        return trend_fc


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def _as_folds(folds) -> list:
    """Normalize a fold container to a list of 2-D float arrays."""
    if folds is None:
        return []
    if isinstance(folds, (list, tuple)):
        items = list(folds)
    else:
        arr = np.asarray(folds, dtype=float)
        if arr.ndim == 3:
            items = [arr[i] for i in range(arr.shape[0])]
        elif arr.ndim == 2:
            items = [arr]
        else:
            return []
    out = []
    for item in items:
        cur = np.asarray(item, dtype=float)
        if cur.ndim == 1:
            cur = cur[np.newaxis, :]
        if cur.ndim == 2 and cur.size:
            out.append(cur)
    return out


def _safe_scale(scale, n_series: int) -> np.ndarray:
    """MASE denominators with non-positive / non-finite entries neutralized.

    A zero denominator would make every candidate score ``inf`` and the argmin
    arbitrary; 1.0 keeps the *ordering between candidates* well defined, which
    is all the selector needs.
    """
    arr = np.asarray(scale, dtype=float).ravel() if scale is not None else np.array([])
    if arr.size == 0:
        arr = np.ones(n_series, dtype=float)
    elif arr.size == 1 and n_series != 1:
        arr = np.repeat(arr, n_series)
    if arr.size < n_series:
        arr = np.concatenate([arr, np.ones(n_series - arr.size)])
    arr = arr[:n_series].astype(float, copy=True)
    arr[~np.isfinite(arr) | (arr <= 0)] = 1.0
    return arr


def _scaled_mae(preds: list, actuals: list, scale: np.ndarray) -> float:
    """Mean over series of pooled |error| / scale. NaN-tolerant."""
    num = []
    for pred, actual in zip(preds, actuals):
        h = min(pred.shape[0], actual.shape[0])
        n = min(pred.shape[1], actual.shape[1])
        if h < 1 or n < 1:
            continue
        num.append(np.abs(pred[:h, :n] - actual[:h, :n]))
    if not num:
        return float('nan')
    n = min(a.shape[1] for a in num)
    stacked = np.concatenate([a[:, :n].reshape(-1, n) for a in num], axis=0)
    with np.errstate(invalid='ignore'):
        mae = np.nanmean(stacked, axis=0)
    return float(np.nanmean(mae / _safe_scale(scale, n)[:n]))


def _graph_pairs(graph, n_series: int) -> list:
    """``[(i, j, expected_sign)]`` for every link a graph asserts."""
    if isinstance(graph, dict):
        adj = block_adjacency(graph, n_series)
    else:
        adj = np.asarray(graph, dtype=float)
        if adj.ndim != 2 or adj.shape != (n_series, n_series):
            return []
    pairs = []
    for i in range(n_series):
        for j in range(i + 1, n_series):
            v = adj[i, j]
            if np.isfinite(v) and v != 0:
                pairs.append((i, j, 1.0 if v > 0 else -1.0))
    return pairs


def _evaluation_pairs(graph_candidates: dict, n_series: int) -> list:
    """One fixed pair set, pooled over every candidate graph.

    Scoring each candidate on its own pairs would let a sparser graph win by
    being asked an easier question; the pooled union with majority-vote signs
    keeps the comparison honest.
    """
    votes: dict = {}
    for graph in (graph_candidates or {}).values():
        for i, j, s in _graph_pairs(graph, n_series):
            votes[(i, j)] = votes.get((i, j), 0.0) + s
    return [(i, j, 1.0 if v >= 0 else -1.0) for (i, j), v in sorted(votes.items())]


def _coherence_error(preds: list, actuals: list, pairs: list, window=None) -> float:
    """|coherence(forecast) - coherence(realized)| over the graph's pairs.

    Scored as error against the realized future, not raw agreement rate: a
    model that forces every linked pair into lockstep is exactly as wrong as
    one that scatters them when the actuals aren't coherent either. Only the
    gap between forecast and realized is worth minimizing.
    """
    if not pairs:
        return float('nan')
    fc_rate, ac_rate = [], []
    for pred, actual in zip(preds, actuals):
        n = min(pred.shape[1], actual.shape[1])
        pd_ = net_direction(pred[:, :n], window)
        ad_ = net_direction(actual[:, :n], window)
        f_agree = f_total = a_agree = a_total = 0
        for i, j, s in pairs:
            if i >= n or j >= n:
                continue
            f_total += 1
            if pd_[i] != 0 and pd_[j] != 0 and np.sign(pd_[i] * pd_[j]) == s:
                f_agree += 1
            a_total += 1
            if ad_[i] != 0 and ad_[j] != 0 and np.sign(ad_[i] * ad_[j]) == s:
                a_agree += 1
        if f_total and a_total:
            fc_rate.append(f_agree / float(f_total))
            ac_rate.append(a_agree / float(a_total))
    if not fc_rate:
        return float('nan')
    return float(abs(np.mean(fc_rate) - np.mean(ac_rate)))


def select_coherence(
    trend_folds, actual_folds, graph_candidates, scale, config=None
) -> dict:
    """Choose ``(graph, strength)`` on inner rolling origins. Accuracy vetoes.

    Two stages: (1) reject any candidate whose pooled scaled MAE exceeds the
    ``strength=0`` baseline by more than ``mase_guardrail`` (accuracy is the
    floor, not something coherence should be traded for), then (2) among
    survivors minimize trend-only coherence error, ties going to the smaller
    strength. ``strength=0.0`` is always admissible, so "do nothing" is always
    reachable.

    Args:
        trend_folds: list of (H, N) **standardized trend** forecasts, one per
            inner origin.
        actual_folds: list of (H, N) realized values aligned to those origins,
            in the same standardized space.
        graph_candidates: ``{name: graph}``, each a ``group_graph`` dict or an
            adjacency array.
        scale: (N,) MASE denominators in the same standardized space.
        config: overrides for ``DEFAULT_COHERENCE_CONFIG``.

    Returns:
        dict with ``graph`` (name or None), ``strength``, ``baseline_mae``,
        ``baseline_coherence_error``, ``selected_mae``,
        ``selected_coherence_error``, ``n_admissible``, ``reason`` and a
        JSON-safe ``table`` of every candidate evaluated.
    """
    cfg = _merged_config(config)
    null = {
        'graph': None,
        'strength': 0.0,
        'baseline_mae': None,
        'baseline_coherence_error': None,
        'selected_mae': None,
        'selected_coherence_error': None,
        'n_admissible': 0,
        'reason': 'no usable folds',
        'table': [],
    }
    try:
        preds = _as_folds(trend_folds)
        acts = _as_folds(actual_folds)
        if not preds or not acts:
            return null
        pairs_n = min(min(p.shape[1] for p in preds), min(a.shape[1] for a in acts))
        if pairs_n < 2:
            return dict(null, reason='fewer than two series')

        strengths = tuple(cfg.get('strengths') or ())
        strengths = tuple(sorted({0.0} | {float(s) for s in strengths}))
        graphs = {k: v for k, v in (graph_candidates or {}).items() if v is not None}
        window = cfg.get('direction_window')

        base_mae = _scaled_mae(preds, acts, scale)
        pairs = _evaluation_pairs(graphs, pairs_n)
        base_coh = _coherence_error(preds, acts, pairs, window)
        if not graphs or not pairs or not np.isfinite(base_mae):
            return dict(
                null,
                baseline_mae=None if not np.isfinite(base_mae) else float(base_mae),
                baseline_coherence_error=(
                    float(base_coh) if np.isfinite(base_coh) else None
                ),
                reason='no candidate graph asserts a link',
            )

        guard = float(cfg.get('mase_guardrail') or 0.0)
        limit = base_mae * (1.0 + guard)
        table = [
            {
                'graph': None,
                'strength': 0.0,
                'mae': float(base_mae),
                'coherence_error': float(base_coh) if np.isfinite(base_coh) else None,
                'admissible': True,
            }
        ]
        best = (base_coh if np.isfinite(base_coh) else np.inf, 0.0, None)
        n_admissible = 1
        for name in sorted(graphs):
            for s in strengths:
                if s == 0.0:
                    continue
                adj_preds = [
                    coherence_shrink(p, graphs[name], s, config=cfg) for p in preds
                ]
                mae = _scaled_mae(adj_preds, acts, scale)
                coh = _coherence_error(adj_preds, acts, pairs, window)
                ok = bool(np.isfinite(mae) and mae <= limit)
                table.append(
                    {
                        'graph': str(name),
                        'strength': float(s),
                        'mae': float(mae) if np.isfinite(mae) else None,
                        'coherence_error': float(coh) if np.isfinite(coh) else None,
                        'admissible': ok,
                    }
                )
                if not ok or not np.isfinite(coh):
                    continue
                n_admissible += 1
                # strictly better wins; ties keep the smaller strength
                if coh < best[0] - 1e-12:
                    best = (coh, float(s), str(name))
        if best[2] is None:
            return dict(
                null,
                baseline_mae=float(base_mae),
                baseline_coherence_error=(
                    float(base_coh) if np.isfinite(base_coh) else None
                ),
                selected_mae=float(base_mae),
                selected_coherence_error=(
                    float(base_coh) if np.isfinite(base_coh) else None
                ),
                n_admissible=n_admissible,
                reason='no admissible candidate improved coherence',
                table=table,
            )
        chosen = [
            r for r in table if r['graph'] == best[2] and r['strength'] == best[1]
        ]
        return {
            'graph': best[2],
            'strength': best[1],
            'baseline_mae': float(base_mae),
            'baseline_coherence_error': (
                float(base_coh) if np.isfinite(base_coh) else None
            ),
            'selected_mae': chosen[0]['mae'] if chosen else None,
            'selected_coherence_error': float(best[0]),
            'n_admissible': n_admissible,
            'reason': 'selected',
            'table': table,
        }
    except Exception:  # pragma: no cover - selection must never fail a forecast
        return dict(null, reason='selection failed')


def apply_selection(trend_fc, selection, graphs, config=None):
    """Apply a ``select_coherence`` result and report what it actually did.

    ``adjustment_rms`` matters because a selected graph/strength that moves
    the forecast by ~0 is a no-op wearing a hat; the diagnostics should say so.

    Args:
        trend_fc: (H, N) standardized trend forecast.
        selection: a ``select_coherence`` result (or ``None``).
        graphs: the same ``{name: graph}`` mapping the selection ran over.
        config: overrides for ``DEFAULT_COHERENCE_CONFIG``.

    Returns:
        ``(adjusted, info)``. ``adjusted`` is the input itself when nothing was
        selected. ``info`` is JSON-safe with ``graph``, ``strength``,
        ``applied``, ``sign_confidence``, ``skipped_groups``, ``n_links``,
        ``adjustment_rms``, ``mase_cost`` and ``coherence_gain``.
    """
    cfg = _merged_config(config)
    info = {
        'graph': None,
        'strength': 0.0,
        'applied': False,
        'sign_confidence': [],
        'skipped_groups': [],
        'n_links': 0,
        'adjustment_rms': 0.0,
        'mase_cost': None,
        'coherence_gain': None,
        'reason': (selection or {}).get('reason', 'no selection'),
    }
    try:
        arr = np.asarray(trend_fc, dtype=float)
        sel = selection or {}
        name = sel.get('graph')
        strength = float(sel.get('strength') or 0.0)
        base_mae = sel.get('baseline_mae')
        sel_mae = sel.get('selected_mae')
        base_coh = sel.get('baseline_coherence_error')
        sel_coh = sel.get('selected_coherence_error')
        if base_mae and sel_mae is not None:
            info['mase_cost'] = float(sel_mae / base_mae - 1.0)
        if base_coh is not None and sel_coh is not None:
            info['coherence_gain'] = float(base_coh - sel_coh)

        graph = (graphs or {}).get(name)
        if graph is None or strength == 0.0 or arr.ndim != 2 or arr.shape[1] < 2:
            return trend_fc, info

        adj, _w, skipped = _resolve_graph(arr, graph, cfg)
        info['graph'] = str(name)
        info['strength'] = strength
        info['skipped_groups'] = [str(k) for k in skipped]
        info['n_links'] = int((np.triu(adj, 1) != 0).sum())
        if isinstance(graph, dict):
            conf = np.asarray(graph.get('confidence', []), dtype=float).ravel()
            info['sign_confidence'] = [float(c) for c in conf]

        out = coherence_shrink(arr, graph, strength, config=cfg)
        delta = np.asarray(out, dtype=float) - arr
        info['adjustment_rms'] = (
            float(np.sqrt(np.nanmean(delta**2))) if delta.size else 0.0
        )
        info['applied'] = bool(info['adjustment_rms'] > 0.0)
        return out, info
    except Exception:  # pragma: no cover - never fail a forecast
        return trend_fc, info


def build_candidates(loadings, config=None, prior=None):
    """Convenience: the candidate graph set implied by ``config['graph']``.

    Wiring helper so ``tva.py`` builds the same candidate set for the inner
    selection and for the final application with one call each.

    When a ``prior`` is supplied and ``config['prior_weight'] > 0``, the set
    gains a prior-blended ``laplacian_k{k}_prior`` per neighbour count plus a
    ``prior_only`` adjacency. They are only *candidates*: ``select_coherence``
    grades them on held-out inner origins against the unblended graphs and the
    always-admissible "do nothing" baseline, so a wrong prior costs nothing.

    Args:
        loadings: (N, K) loading matrix.
        config: overrides for ``DEFAULT_COHERENCE_CONFIG``. ``graph`` selects
            the family: ``'group'``, ``'laplacian'``, ``'auto'`` (both) or
            ``'none'`` (empty, i.e. the correction is disabled).
        prior: optional (N, N) signed user prior. ``None`` reproduces the
            candidate dict exactly as before.

    Returns:
        ``(graphs, meta)`` where ``graphs`` is ``{name: graph}`` and ``meta``
        carries the JSON-safe ``sign_confidence`` and ``signs``.
    """
    cfg = _merged_config(config)
    lam = _clean_loadings(loadings)
    _signed, confidence = resolve_signs(
        lam, stability=_config_stability(cfg, lam.shape[1] if lam.ndim == 2 else 0)
    )
    meta = {
        'sign_confidence': [float(c) for c in np.atleast_1d(confidence)],
        'n_series': int(lam.shape[0]) if lam.ndim == 2 else 0,
        'n_factors': int(lam.shape[1]) if lam.ndim == 2 else 0,
    }
    mode = str(cfg.get('graph') or 'auto').lower()
    graphs: dict = {}
    if mode in ('none',):
        return graphs, meta
    if mode in ('group', 'auto'):
        g = group_graph(lam, cfg)
        if g.get('groups'):
            graphs['group'] = g
    if mode in ('laplacian', 'auto'):
        for k in cfg.get('neighbors') or (5,):
            adj = laplacian_graph(lam, cfg, n_neighbors=int(k))
            if np.any(adj):
                graphs['laplacian_k{}'.format(int(k))] = adj

    n_series = int(meta['n_series'])
    prior_sym = _symmetric_prior(prior, n_series)
    if (
        prior_sym is not None
        and cfg.get('prior_candidates', True)
        and float(cfg.get('prior_weight') or 0.0) > 0
    ):
        if mode in ('laplacian', 'auto'):
            for k in cfg.get('neighbors') or (5,):
                adj = laplacian_graph(lam, cfg, n_neighbors=int(k), prior=prior_sym)
                if np.any(adj):
                    graphs['laplacian_k{}_prior'.format(int(k))] = adj
        if np.any(prior_sym):
            graphs['prior_only'] = prior_sym
    return graphs, meta
