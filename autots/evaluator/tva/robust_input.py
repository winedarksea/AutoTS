# -*- coding: utf-8 -*-
"""
Robust trend-isolation input estimator for the latent-factor stage.

Alternative to ``TVA._build_adjusted_panel``: jointly and robustly splits a
(T, N) level panel into shared low-rank trend, series-local trend, sparse
anomalies, and persistent intercept shifts, holding steps/spikes out of the
low-rank part so they can't be absorbed into a spurious shared factor.
"""

from __future__ import annotations

import warnings

import numpy as np

DEFAULT_ROBUST_INPUT_CONFIG = {
    # ---- trend basis -------------------------------------------------------
    'knot_spacing': 30,  # candidate hinge every ~month
    'max_knots': 120,  # cap; spacing is widened past this
    # l1 penalty on standardized hinge coefficients; 'auto' picks from
    # alpha_grid by held-out error -- the right value varies a lot by panel
    'alpha': 'auto',
    'alpha_grid': (0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
    'cv_stride': 5,  # hold out every k-th block when picking alpha
    'cv_block': None,  # holdout block length; defaults to knot_spacing
    # ---- robustness --------------------------------------------------------
    'huber_delta': 1.5,  # IRLS downweight starts at 1.5 robust sigma
    'irls_iters': 3,  # inner reweighting steps per outer pass
    'floor_frac': 1e-3,  # MAD floor as a fraction of the panel median MAD
    # ---- shared / idiosyncratic split -------------------------------------
    'rank': 3,  # target rank of the shared trend
    # ---- contamination terms ----------------------------------------------
    'anomaly_k': 3.0,  # soft-threshold at k robust sigma of the residual
    'max_shifts': 6,  # per series cap on detected level shifts
    'min_shift_size': 1.0,  # minimum jump, in robust sigma of the residual
    'min_shift_frac': 0.4,  # ... and in units of the series' own robust scale
    'min_shift_len': 30,  # minimum segment length, in steps
    'shift_stat_k': 6.0,  # step candidate: |diff| in robust sigma of diff
    # ---- outer loop --------------------------------------------------------
    'max_iters': 5,
    'tol': 1e-3,  # max abs change of the adjusted panel, scaled
    # ---- nuisance shrinkage (per detector component, in [0, 1]) -----------
    'shrink': {
        'seasonality': 1.0,
        'holidays': 1.0,
        'anomalies': 1.0,
        'level_shifts': 1.0,
    },
}

COMPONENT_KEYS = ('seasonality', 'holidays', 'anomalies', 'level_shifts')


# ---------------------------------------------------------------------------
# small local reimplementations (kept local so this module stands alone)
# ---------------------------------------------------------------------------


def _hinge_design(n_time: int, knot_spacing: int, max_knots: int = 120):
    """Trend-filtering basis ``[1, t/T, (t - k)_+/T]``.

    Mirrors ``factor_network.hinge_design`` but with an unpenalized intercept
    column prepended, since here the fit targets a raw column, not an
    already-anchored factor score.
    """
    n_time = max(int(n_time), 1)
    spacing = max(int(knot_spacing), 1)
    if max_knots and n_time // spacing > max_knots:
        spacing = int(np.ceil(n_time / float(max_knots)))
    t = np.arange(n_time, dtype=float)
    knots = np.arange(spacing, max(n_time - spacing, spacing + 1), spacing)
    knots = knots[knots < n_time]
    cols = [np.ones(n_time), t / float(n_time)]
    cols += [np.clip(t - k, 0.0, None) / float(n_time) for k in knots]
    return np.column_stack(cols)


def _robust_scale(values: np.ndarray, mask: np.ndarray, floor_frac: float = 1e-3):
    """Per-series robust center/scale in level space, over observed rows only.

    Median/MAD rather than mean/std, since the panel's steps and spikes would
    otherwise set the units themselves. Floor guards a near-constant column's
    (near-zero) MAD.

    Returns:
        ``(center (N,), scale (N,))``.
    """
    arr = np.where(mask, np.asarray(values, dtype=float), np.nan)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')  # all-NaN columns are a valid input
        center = np.nanmedian(arr, axis=0)
        center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
        mad = np.nanmedian(np.abs(arr - center[None, :]), axis=0)
    scale = 1.4826 * np.asarray(mad, dtype=float)
    bad = ~np.isfinite(scale) | (scale <= 0)
    if np.any(bad):
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            alt = np.nanstd(arr[:, bad], axis=0)
        alt = np.asarray(alt, dtype=float)
        scale[bad] = np.where(np.isfinite(alt) & (alt > 0), alt, 1.0)
    finite = scale[np.isfinite(scale) & (scale > 0)]
    typical = float(np.median(finite)) if finite.size else 1.0
    scale = np.maximum(scale, max(typical * floor_frac, 1e-8))
    return center, scale


def _robust_sigma(residual: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Column-wise robust residual scale (MAD) over weighted-in rows."""
    arr = np.where(weights > 0, residual, np.nan)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')  # fully-masked columns are valid
        med = np.nanmedian(arr, axis=0)
        med = np.nan_to_num(med, nan=0.0)
        sigma = 1.4826 * np.nanmedian(np.abs(arr - med[None, :]), axis=0)
    sigma = np.asarray(sigma, dtype=float)
    sigma[~np.isfinite(sigma) | (sigma <= 0)] = 1e-6
    return sigma


# ---------------------------------------------------------------------------
# weighted l1 trend fit
# ---------------------------------------------------------------------------


def _lasso_paths(
    design: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    extra=None,
    max_sweeps: int = 60,
    tol: float = 1e-5,
):
    """Weighted lasso of every column of ``targets`` on the shared ``design``.

    The intercept and ``extra`` are fit unpenalized and partialled out before
    the penalized solve, so ``alpha`` is measured against the trend signal
    alone rather than a target a level shift has inflated. ``extra`` fits per-
    series step regressors unpenalized so a genuine step is charged to the
    step term instead of bending the trend. Hand-rolled coordinate descent
    (not sklearn) because sklearn exposes no per-coefficient penalty.

    Args:
        extra: optional length-N sequence of (T, m_j) unpenalized regressors
            (or None per series).

    Returns:
        ``(trend (T, N), extra_fit (T, N))`` — the hinge part including the
        intercept, and the part explained by ``extra`` (zero when absent).
    """
    T, N = targets.shape
    fitted = np.zeros((T, N), dtype=float)
    extra_fit = np.zeros((T, N), dtype=float)
    base = design[:, 1:]  # penalized part
    ones = np.ones((T, 1), dtype=float)

    for j in range(N):
        w = weights[:, j]
        tot = float(w.sum())
        if tot <= 0 or not np.isfinite(tot):
            continue
        add = None if extra is None else extra[j]
        if add is not None and getattr(add, 'size', 0) == 0:
            add = None
        # free block: intercept plus any unpenalized step regressors
        A = ones if add is None else np.column_stack([ones, add])
        y = targets[:, j]
        sw = np.sqrt(w)
        Aw = A * sw[:, None]
        try:
            gram = Aw.T @ Aw + 1e-10 * np.eye(A.shape[1])
            gy = np.linalg.solve(gram, Aw.T @ (y * sw))
            GB = np.linalg.solve(gram, Aw.T @ (base * sw[:, None]))
        except np.linalg.LinAlgError:  # pragma: no cover
            gy = np.zeros(A.shape[1])
            GB = np.zeros((A.shape[1], base.shape[1]))
        # Frisch-Waugh: partial the free block out of both sides so the
        # penalty sees the trend signal alone, not a target a level shift
        # has inflated
        yr = y - A @ gy
        Br = base - A @ GB
        sd = float(np.sqrt(max((w * yr * yr).sum() / tot, 0.0)))
        if not np.isfinite(sd) or sd <= 1e-12:
            fitted[:, j] = A[:, 0] * gy[0]
            if add is not None:
                extra_fit[:, j] = add @ gy[1:]
            continue
        penalty = np.full(base.shape[1], float(alpha))
        beta = (
            _cd_lasso(Br * sw[:, None], (yr / sd) * sw, penalty, max_sweeps, tol) * sd
        )
        g = gy - GB @ beta
        fitted[:, j] = base @ beta + g[0]
        if add is not None:
            extra_fit[:, j] = add @ g[1:]
    return fitted, extra_fit


def _cd_lasso(Xw, yw, penalty, max_sweeps, tol):
    """Coordinate descent for ``0.5/n ||y - Xb||^2 + sum_k penalty_k |b_k|``.

    Matches sklearn's objective scaling, with a per-coefficient penalty so
    some columns (the step regressors) can be left unpenalized.
    """
    n, p = Xw.shape
    gram = Xw.T @ Xw
    corr = Xw.T @ yw
    diag = np.diag(gram).copy()
    diag[diag <= 1e-12] = 1e-12
    beta = np.zeros(p, dtype=float)
    thresh = np.asarray(penalty, dtype=float) * n
    for _ in range(max_sweeps):
        delta_max = 0.0
        for k in range(p):
            old = beta[k]
            rho = corr[k] - gram[k] @ beta + diag[k] * old
            new = np.sign(rho) * max(abs(rho) - thresh[k], 0.0) / diag[k]
            if new != old:
                beta[k] = new
                delta_max = max(delta_max, abs(new - old))
        if delta_max < tol:
            break
    return beta


def _select_alpha(design, targets, weights, obs, extra, cfg):
    """Pick the l1 penalty by held-out error on a strided row holdout.

    Holdout blocks must be long (``~T // 6``): with short or scattered
    holdout rows, a trend wiggly enough to trace leftover seasonality still
    predicts across the gap, so CV picks the most overfit model on the grid.
    Only a long gap forces genuine extrapolation, which distinguishes real
    trend from overfit wiggle. Score is the median (not mean) absolute
    held-out residual per series, since a mean would let spikes pick alpha.
    """
    grid = [float(a) for a in cfg.get('alpha_grid') or ()]
    grid = sorted({a for a in grid if a > 0})
    if not grid:
        return 0.02
    stride = max(int(cfg.get('cv_stride', 5)), 2)
    n_time = targets.shape[0]
    block = cfg.get('cv_block')
    if not block:
        # long enough to force genuine extrapolation across the gap
        block = int(
            np.clip(n_time // 6, 2 * int(cfg['knot_spacing']), max(n_time // 4, 2))
        )
    block = max(int(block), 2)
    t_idx = np.arange(targets.shape[0])
    rows = ((t_idx // block) % stride) == 0
    holdout = np.zeros(targets.shape, dtype=bool)
    holdout[rows, :] = True
    holdout &= obs
    if holdout.sum() < 10 or (obs & ~holdout).sum() < 10:
        return grid[len(grid) // 2]
    fit_w = np.where(holdout, 0.0, weights)
    best, best_score = grid[0], np.inf
    for alpha in grid:
        w = fit_w.copy()
        fit = np.zeros_like(targets)
        step = np.zeros_like(targets)
        for _ in range(2):
            fit, step = _lasso_paths(design, targets, w, alpha, extra=extra)
            resid = targets - fit - step
            sig = _robust_sigma(resid, fit_w)
            w = np.where(
                holdout,
                0.0,
                _huber_weights(resid, obs, sig, float(cfg['huber_delta'])),
            )
        err = np.abs(targets - fit - step)
        scores = []
        for j in range(targets.shape[1]):
            col = err[holdout[:, j], j]
            if col.size:
                scores.append(float(np.median(col)))
        score = float(np.mean(scores)) if scores else np.inf
        if score < best_score - 1e-12:
            best, best_score = alpha, score
    return best


def _huber_weights(residual, mask, sigma, delta):
    """IRLS weights: 1 inside ``delta`` sigma, ``delta * sigma / |r|`` outside.

    The mask multiplies in, so an unobserved (forward-filled) row carries no
    weight at all rather than the deceptively small residual of a flat run.
    """
    scaled = np.abs(residual) / np.maximum(sigma[None, :], 1e-12)
    w = np.where(scaled <= delta, 1.0, delta / np.maximum(scaled, 1e-12))
    return np.where(mask, w, 0.0)


# ---------------------------------------------------------------------------
# level-shift detection
# ---------------------------------------------------------------------------


def _step_candidates(z_col, weight, cfg):
    """Candidate level-shift times for one series, from its first difference.

    A level shift is a single large value in the first difference, unlike
    trend curvature which is spread thin; candidates are proposed generously
    here and adjudicated later by the joint fit (a step column competes
    against the hinges on cost). Spikes are rejected here rather than
    downstream: a spike's difference immediately reverses, so a candidate
    only survives if the local mean actually moves and stays moved.

    Returns:
        Sorted list of breakpoint indices (the first step *after* the shift).
    """
    T = z_col.shape[0]
    min_len = max(int(cfg['min_shift_len']), 2)
    max_shifts = max(int(cfg['max_shifts']), 0)
    if T < 2 * min_len + 2 or max_shifts == 0:
        return []
    both = (weight[1:] > 0) & (weight[:-1] > 0)
    diff = np.where(both, z_col[1:] - z_col[:-1], 0.0)
    obs_d = diff[both]
    if obs_d.size < 8:
        return []
    med = float(np.median(obs_d))
    sigma_d = 1.4826 * float(np.median(np.abs(obs_d - med)))
    if not np.isfinite(sigma_d) or sigma_d <= 0:
        return []
    thresh = float(cfg['shift_stat_k']) * sigma_d
    idx = np.where(np.abs(diff - med) > thresh)[0] + 1
    idx = idx[(idx >= min_len) & (idx <= T - min_len)]
    if idx.size == 0:
        return []
    order = idx[np.argsort(-np.abs(diff[idx - 1]))]
    accepted = []
    for cut in order:
        if len(accepted) >= max_shifts:
            break
        if any(abs(cut - a) < min_len for a in accepted):
            continue
        pre = z_col[max(cut - min_len, 0) : cut]
        pre_w = weight[max(cut - min_len, 0) : cut]
        post = z_col[cut : cut + min_len]
        post_w = weight[cut : cut + min_len]
        if pre_w.sum() < 2 or post_w.sum() < 2:
            continue
        move = float(np.median(post[post_w > 0]) - np.median(pre[pre_w > 0]))
        jump = float(diff[cut - 1] - med)
        # a spike reverses; a shift persists, in the same direction
        if move * jump <= 0 or abs(move) < 0.4 * abs(jump):
            continue
        if abs(move) < float(cfg['min_shift_frac']):
            continue
        accepted.append(int(cut))
    return sorted(accepted)


def _step_design(n_time, breaks):
    """(T, m) 0/1 step regressors, one per breakpoint (``1`` from the cut on)."""
    if not breaks:
        return None
    cols = np.zeros((n_time, len(breaks)), dtype=float)
    for i, cut in enumerate(breaks):
        cols[cut:, i] = 1.0
    return cols


def _prune_steps(step_fit_coefs, breaks, sigma, cfg):
    """Drop step candidates whose fitted jump is too small to be a regime.

    Two floors, both required. ``min_shift_size`` is in units of the fit's own
    residual sigma — the statistical floor. ``min_shift_frac`` is in units of
    the series' robust scale — the practical floor, and the one that matters,
    because a series the trend filter already fits almost perfectly has a tiny
    residual sigma and a sigma-only rule would promote a rounding-level wobble
    to a regime change.
    """
    floor = max(float(cfg['min_shift_size']) * sigma, float(cfg['min_shift_frac']))
    keep = [b for b, c in zip(breaks, step_fit_coefs) if abs(c) >= floor]
    return keep


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def _resolve_config(config):
    cfg = dict(DEFAULT_ROBUST_INPUT_CONFIG)
    cfg['shrink'] = dict(DEFAULT_ROBUST_INPUT_CONFIG['shrink'])
    if config:
        for key, val in dict(config).items():
            if key == 'shrink' and isinstance(val, dict):
                cfg['shrink'].update(val)
            else:
                cfg[key] = val
    return cfg


def _as_array(obj, shape):
    """Coerce a DataFrame/array component to a finite (T, N) float array."""
    if obj is None:
        return None
    arr = getattr(obj, 'to_numpy', None)
    arr = obj.to_numpy(dtype=float) if arr is not None else np.asarray(obj, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape != shape:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def robust_adjusted_panel(
    values,
    mask=None,
    components=None,
    shrink=None,
    config=None,
) -> dict:
    """Jointly estimate a trend-only level panel from raw levels + nuisances.

    Drop-in alternative to ``TVA._build_adjusted_panel``: treats detector
    components as nuisance regressors with adjustable trust (``shrink``) and
    alternates, to convergence, between an IRLS-Huber l1 trend fit per series,
    a truncated SVD splitting shared vs. series-local trend, soft-thresholded
    anomalies, and binary-segmented intercept shifts. Anomaly/shift terms are
    subtracted before the next pass's trend fit so a step can't be laundered
    into a shared "regime" via the low-rank factorization.

    Never raises on degenerate input; falls back to the raw panel minus the
    shrunk components (the pre-existing behaviour) on any failure.

    Args:
        mask: (T, N) bool, True where observed; inferred from finiteness of
            ``values`` when None. A filled run must be False here, or it
            carries estimation weight it shouldn't.
        components: dict of detector estimates, any subset of
            ``('seasonality', 'holidays', 'anomalies', 'level_shifts')``.
        shrink: {component_name: float in [0, 1]} -- how much of each
            component to subtract; < 1 hedges against a component estimate
            that absorbed real trend.
        config: overrides of ``DEFAULT_ROBUST_INPUT_CONFIG``.

    Returns:
        dict with ``'adjusted'``, ``'shared'``, ``'idio'``, ``'anomalies'``,
        ``'shifts'`` (all (T, N) level units), ``'scale'``, ``'center'``
        (N,), ``'n_iters'``, ``'converged'``, ``'diagnostics'``.
    """
    cfg = _resolve_config(config)
    if shrink:
        cfg['shrink'] = dict(cfg['shrink'])
        cfg['shrink'].update({k: float(v) for k, v in dict(shrink).items()})

    raw = getattr(values, 'to_numpy', None)
    raw = (
        values.to_numpy(dtype=float)
        if raw is not None
        else np.asarray(values, dtype=float)
    )
    raw = np.atleast_2d(np.asarray(raw, dtype=float))
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    T, N = raw.shape
    shape = (T, N)

    if mask is None:
        obs = np.isfinite(raw)
    else:
        m = getattr(mask, 'to_numpy', None)
        m = mask.to_numpy() if m is not None else np.asarray(mask)
        m = np.asarray(m)
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        obs = np.isfinite(raw) & (
            m.astype(bool) if m.shape == shape else np.isfinite(raw)
        )

    comps = {}
    for key in COMPONENT_KEYS:
        arr = _as_array((components or {}).get(key), shape)
        if arr is not None:
            comps[key] = arr

    # ---- nuisance subtraction with per-component shrinkage -----------------
    nuisance = np.zeros(shape, dtype=float)
    for key, arr in comps.items():
        s = float(np.clip(cfg['shrink'].get(key, 1.0), 0.0, 1.0))
        if s != 0.0:
            nuisance = nuisance + s * arr
    base = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0) - nuisance
    # rows we cannot see get a locally-carried value purely so the arrays are
    # finite; they carry zero weight everywhere it matters
    filled = _fill_gaps(base, obs)

    fallback = {
        'adjusted': filled.copy(),
        'shared': np.zeros(shape),
        'idio': filled.copy(),
        'anomalies': np.zeros(shape),
        'shifts': np.zeros(shape),
        'scale': np.ones(N),
        'center': np.zeros(N),
        'n_iters': 0,
        'converged': False,
        'diagnostics': {'fallback': True, 'reason': 'degenerate input'},
    }
    if T < 3 or N < 1 or not np.any(obs):
        fallback['diagnostics']['reason'] = 'too few observations'
        return fallback

    try:
        return _estimate(filled, obs, cfg, comps)
    except Exception as err:  # pragma: no cover - safety net, never raise
        fallback['diagnostics']['reason'] = f'solver failure: {err!r}'
        return fallback


def _fill_gaps(arr, obs):
    """Forward/backward fill unobserved entries so arrays stay finite.

    Placeholders only -- every downstream step weights by ``obs``.
    """
    out = np.array(arr, dtype=float, copy=True)
    out[~obs] = np.nan
    for j in range(out.shape[1]):
        col = out[:, j]
        good = np.isfinite(col)
        if not np.any(good):
            out[:, j] = 0.0
            continue
        idx = np.where(good, np.arange(col.shape[0]), 0)
        np.maximum.accumulate(idx, out=idx)
        col = col[idx]
        # backfill the head
        first = int(np.argmax(good))
        col[:first] = col[first]
        out[:, j] = col
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _estimate(base, obs, cfg, comps):
    """Inner joint estimator; see ``robust_adjusted_panel`` for semantics."""
    T, N = base.shape
    center, scale = _robust_scale(base, obs, cfg['floor_frac'])
    Z = (base - center[None, :]) / scale[None, :]
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    design = _hinge_design(T, cfg['knot_spacing'], cfg['max_knots'])
    mask_f = obs.astype(float)
    rank = int(np.clip(int(cfg['rank']), 0, min(N, T)))

    # Step candidates come from the *raw* scaled panel, before the trend
    # filter has had a chance to explain a step away as a steep segment.
    breaks = [_step_candidates(Z[:, j], mask_f[:, j], cfg) for j in range(N)]

    alpha = cfg['alpha']
    if isinstance(alpha, str) or alpha is None:
        alpha = _select_alpha(
            design,
            Z,
            mask_f,
            obs,
            [_step_design(T, b) for b in breaks],
            cfg,
        )
    alpha = float(alpha)

    anomalies = np.zeros((T, N), dtype=float)
    shifts = np.zeros((T, N), dtype=float)
    trend = np.zeros((T, N), dtype=float)
    prev = None
    converged = False
    n_iters = 0

    def joint_fit(anomaly_term, break_sets):
        """IRLS-Huber joint fit of hinges (penalized) + steps (free)."""
        target = Z - anomaly_term
        extra = [_step_design(T, b) for b in break_sets]
        weights = mask_f.copy()
        fit = np.zeros((T, N), dtype=float)
        step_fit = np.zeros((T, N), dtype=float)
        for _ in range(max(int(cfg['irls_iters']), 1)):
            fit, step_fit = _lasso_paths(design, target, weights, alpha, extra=extra)
            resid = target - fit - step_fit
            sig = _robust_sigma(resid, mask_f)
            weights = _huber_weights(resid, obs, sig, float(cfg['huber_delta']))
        return fit, step_fit

    for it in range(max(int(cfg['max_iters']), 1)):
        n_iters = it + 1
        trend, shifts = joint_fit(anomalies, breaks)

        # ---- prune steps the fit did not actually want --------------------
        resid = Z - trend - shifts
        sigma = _robust_sigma(resid, mask_f)
        pruned = False
        for j in range(N):
            if not breaks[j]:
                continue
            jumps = [shifts[c, j] - shifts[c - 1, j] for c in breaks[j]]
            keep = _prune_steps(jumps, breaks[j], float(sigma[j]), cfg)
            if len(keep) != len(breaks[j]):
                breaks[j] = keep
                pruned = True
        if pruned:
            trend, shifts = joint_fit(anomalies, breaks)
            resid = Z - trend - shifts
            sigma = _robust_sigma(resid, mask_f)

        # ---- sparse transient contamination -------------------------------
        thr = float(cfg['anomaly_k']) * sigma[None, :]
        anomalies = np.sign(resid) * np.maximum(np.abs(resid) - thr, 0.0)
        anomalies = np.where(obs, anomalies, 0.0)

        if prev is not None:
            delta = float(np.max(np.abs(trend - prev))) if trend.size else 0.0
            if delta < float(cfg['tol']):
                converged = True
                break
        prev = trend

    # the step term carries no net level; the intercept belongs to the trend
    shift_mean = (shifts * mask_f).sum(axis=0) / np.maximum(mask_f.sum(axis=0), 1.0)
    shifts = shifts - shift_mean[None, :]
    trend = trend + shift_mean[None, :]

    # ---- shared / idiosyncratic split of the final smooth fits ------------
    col_mean = (trend * mask_f).sum(axis=0) / np.maximum(mask_f.sum(axis=0), 1.0)
    centered = trend - col_mean[None, :]
    shared = np.zeros_like(centered)
    if rank >= 1:
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            r = min(rank, S.shape[0])
            shared = (U[:, :r] * S[:r]) @ Vt[:r]
        except np.linalg.LinAlgError:  # pragma: no cover
            shared = np.zeros_like(centered)
    idio = trend - shared

    shared_lvl = shared * scale[None, :]
    idio_lvl = idio * scale[None, :] + center[None, :]
    adjusted = shared_lvl + idio_lvl

    resid_final = Z - trend - shifts - anomalies
    diagnostics = {
        'fallback': False,
        'n_shifts': int(sum(len(b) for b in breaks)),
        'shift_times': {j: list(b) for j, b in enumerate(breaks) if b},
        'anomaly_fraction': float(
            (np.abs(anomalies) > 0).sum() / max(anomalies.size, 1)
        ),
        'shared_share': float(np.sum(shared**2) / max(np.sum(trend**2), 1e-12)),
        'residual_sigma': _robust_sigma(resid_final, mask_f),
        'rank': int(rank),
        'observed_fraction': float(obs.mean()),
        'knot_columns': int(design.shape[1]),
        'alpha': float(alpha),
    }
    return {
        'adjusted': adjusted,
        'shared': shared_lvl,
        'idio': idio_lvl,
        'anomalies': anomalies * scale[None, :],
        'shifts': shifts * scale[None, :],
        'scale': scale,
        'center': center,
        'n_iters': int(n_iters),
        'converged': bool(converged),
        'diagnostics': diagnostics,
    }


# ---------------------------------------------------------------------------
# diagnostic: which input is actually closer to the trend?
# ---------------------------------------------------------------------------


def _to_array(obj):
    arr = getattr(obj, 'to_numpy', None)
    arr = obj.to_numpy(dtype=float) if arr is not None else np.asarray(obj, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _score_against(candidate, oracle, raw_ref):
    """Per-input scores against the oracle trend; see ``compare_inputs``."""
    T = min(candidate.shape[0], oracle.shape[0])
    N = min(candidate.shape[1], oracle.shape[1])
    cand = candidate[:T, :N]
    orc = oracle[:T, :N]
    ref = raw_ref[:T, :N]
    corrs, nrmses, retained = [], [], []
    for j in range(N):
        c = cand[:, j]
        o = orc[:, j]
        r = ref[:, j]
        good = np.isfinite(c) & np.isfinite(o) & np.isfinite(r)
        if good.sum() < 3:
            continue
        c, o, r = c[good], o[good], r[good]
        # intercepts are not part of the question: every candidate is free to
        # sit at a different level, so compare shapes about their own means
        cc = c - c.mean()
        oc = o - o.mean()
        rc = r - r.mean()
        so, sc = float(oc.std()), float(cc.std())
        if so > 1e-12 and sc > 1e-12:
            corrs.append(abs(float(np.corrcoef(cc, oc)[0, 1])))
        if so > 1e-12:
            nrmses.append(float(np.sqrt(np.mean((cc - oc) ** 2)) / so))
        denom = float(np.sum((rc - oc) ** 2))
        if denom > 1e-12:
            retained.append(float(np.sum((cc - oc) ** 2) / denom))
    return {
        'mean_abs_corr': float(np.mean(corrs)) if corrs else float('nan'),
        'nrmse': float(np.mean(nrmses)) if nrmses else float('nan'),
        'residual_energy_retained': (
            float(np.mean(retained)) if retained else float('nan')
        ),
        'n_scored': int(len(corrs)),
    }


def compare_inputs(raw, detector_adjusted, robust_adjusted, oracle_trend) -> dict:
    """Score four candidate factor-model inputs against the true trend.

    Separates decomposition failure from factor-model failure: if the robust
    input scores near the oracle and the factor model still recovers poorly,
    the loss is downstream; if the robust input sits near the
    detector-adjusted panel, decomposition is the binding constraint.

    Scores (all column-wise on mean-removed series, then averaged):
    ``mean_abs_corr`` -- mean |correlation| with oracle (1 is best);
    ``nrmse`` -- RMSE / oracle std (0 is best, 1.0 = no better than flat);
    ``residual_energy_retained`` -- non-trend energy left, as a fraction of
    the raw panel's (raw scores 1.0, oracle 0.0, >1.0 means added noise).

    Returns:
        dict keyed by ``'raw'``, ``'detector'``, ``'robust'``, ``'oracle'``,
        each a dict of the three scores plus ``'n_scored'``, and a
        ``'ranking'`` key listing input names best-first by
        ``mean_abs_corr``.
    """
    oracle = _to_array(oracle_trend)
    raw_a = _to_array(raw)
    candidates = {
        'raw': raw_a,
        'detector': (
            _to_array(detector_adjusted) if detector_adjusted is not None else raw_a
        ),
        'robust': _to_array(robust_adjusted) if robust_adjusted is not None else raw_a,
        'oracle': oracle,
    }
    out = {}
    for name, arr in candidates.items():
        try:
            out[name] = _score_against(arr, oracle, raw_a)
        except Exception as err:  # pragma: no cover - never raise on diagnostics
            out[name] = {
                'mean_abs_corr': float('nan'),
                'nrmse': float('nan'),
                'residual_energy_retained': float('nan'),
                'n_scored': 0,
                'error': repr(err),
            }
    ranked = sorted(
        out.items(),
        key=lambda kv: (
            -kv[1]['mean_abs_corr'] if np.isfinite(kv[1]['mean_abs_corr']) else np.inf
        ),
    )
    out['ranking'] = [name for name, _ in ranked]
    return out
