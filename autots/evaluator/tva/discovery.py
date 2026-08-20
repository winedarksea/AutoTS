# -*- coding: utf-8 -*-
"""
TVA structure discovery — factors first, conditional edges second.

Torch-free (numpy + sklearn + statsmodels only). This module replaces the
free-fit latent adjacency approach: composite trends are discovered as
varimax-rotated factors of the differenced trend panel (identifiable,
nameable, stable across seeds), and pairwise lead-lag edges are discovered
only on the factor RESIDUALS (killing the common-driver false positives that
dominate co-trending panels), directed by lag structure through
stability-selected lasso VAR, and pruned by edge-level falsification.

Single entry point: :func:`discover_structure`.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_DISCOVERY_CONFIG = {
    # factor extraction
    'variance_target': 0.6,  # smallest r explaining >= this share of variance
    'max_factors': 4,
    # Optional Hodrick-Prescott smoothing applied to the panel
    'factor_hp_lambda': None,
    'soft_threshold': 0.15,  # fraction of per-factor max |loading| zeroed out
    # deconfounding
    'factor_lags': (0, 1),
    # also regress out detected shared components (panel-mean seasonality /
    # holidays / level shifts / anomalies) -- an unmodeled calendar effect is
    # a confounder like any other
    'deconfound_components': False,
    # circular-shift null reps for edge-count calibration; 0 disables
    'null_calibration_reps': 0,
    'null_quantile': 0.95,
    # conditional screening
    'top_k': 8,
    # lasso VAR direction/lag/weight
    'max_lag': 3,
    'n_subsamples': 60,
    'subsample_frac': 0.5,
    'stability_threshold': 0.7,
    # LIFT lead-lag family
    'leadlag_max_lag': 7,
    'leadlag_min_corr': 0.3,
    'leadlag_top_k': 3,
    'leadlag_window': 180,
    # falsification
    'holdout_frac': 0.2,
    # merge
    'family_weights': {
        'factor': 1.0,
        'lasso': 1.0,
        'leadlag': 0.5,
        'event': 0.5,
        'metadata': 0.25,
        # user-supplied priors: guesses, not measurements. TVA overrides
        # both with prior_confidence unless the caller sets them here.
        'business': 1.0,
        'causal': 0.5,
    },
}


def _resolve_config(config: Optional[dict]) -> dict:
    resolved = dict(DEFAULT_DISCOVERY_CONFIG)
    resolved['family_weights'] = dict(DEFAULT_DISCOVERY_CONFIG['family_weights'])
    if config:
        # family_weights is merged, not replaced: a partial user dict used to
        # wipe out every default weight it did not mention.
        user_weights = config.get('family_weights')
        resolved.update(config)
        if isinstance(user_weights, dict):
            merged = dict(DEFAULT_DISCOVERY_CONFIG['family_weights'])
            merged.update(user_weights)
            resolved['family_weights'] = merged
    return resolved


# ---------------------------------------------------------------------------
# D-1: preprocessing
# ---------------------------------------------------------------------------


def _hp_smooth(values: np.ndarray, lam: float) -> np.ndarray:
    """Hodrick-Prescott trend of each column (NaN-safe, degrades to input)."""
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
    except ImportError:  # pragma: no cover - statsmodels is an AutoTS dep
        return values
    filled = pd.DataFrame(values).ffill().bfill().values
    if not np.all(np.isfinite(filled)):
        filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.empty_like(filled, dtype=float)
    for j in range(filled.shape[1]):
        column = filled[:, j]
        if np.ptp(column) < 1e-12:
            out[:, j] = column
            continue
        try:
            out[:, j] = hpfilter(column, lamb=float(lam))[1]
        except Exception:
            out[:, j] = column
    return out


def _difference_and_standardize(trend_values: np.ndarray) -> tuple:
    """Difference once and scale each series by the std of its differences.

    Trends are near-integrated; levels give spurious everything. Returns
    (X_diff_std (T-1, N), scale (N,)).
    """
    diffs = np.diff(trend_values, axis=0)
    diffs = np.nan_to_num(diffs, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nanstd(diffs, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return diffs / scale, scale


# ---------------------------------------------------------------------------
# D-2: factor extraction (the primary object)
# ---------------------------------------------------------------------------


def _varimax(
    loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-8
) -> np.ndarray:
    """Varimax rotation (numpy only). Returns the rotation matrix R."""
    n, k = loadings.shape
    R = np.eye(k)
    if k < 2:
        return R
    var_prev = 0.0
    for _ in range(max_iter):
        L = loadings @ R
        u, s, vt = np.linalg.svd(
            loadings.T @ (L**3 - (gamma / n) * L @ np.diag(np.sum(L**2, axis=0)))
        )
        R = u @ vt
        var_new = np.sum(s)
        if var_prev != 0 and (var_new - var_prev) < tol * var_prev:
            break
        var_prev = var_new
    return R


def _fix_factor_signs(loadings: np.ndarray, scores: np.ndarray) -> tuple:
    """Deterministic sign convention: each factor's largest |loading| is positive."""
    for j in range(loadings.shape[1]):
        col = loadings[:, j]
        if col.size and col[np.argmax(np.abs(col))] < 0:
            loadings[:, j] = -col
            scores[:, j] = -scores[:, j]
    return loadings, scores


def extract_factors(X: np.ndarray, config: dict) -> tuple:
    """Truncated SVD + varimax + soft-thresholding on the differenced panel.

    Args:
        X: (T-1, N) differenced, standardized panel.
        config: resolved discovery config.

    Returns:
        (factors_level (T, r), loadings (N, r), scores_diff (T-1, r))
    """
    T1, N = X.shape
    r_cap = max(1, min(int(config['max_factors']), max(N // 4, 1)))
    Xc = X - X.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = float(np.sum(s**2))
    if total_var <= 0:
        r = 1
    else:
        # Parallel analysis: keep factors whose singular value beats the mean
        # top singular value of column-permuted (signal-destroyed) panels.
        # More robust than a raw variance target on differenced (noise-heavy)
        # data, and deterministic (internal fixed seed, independent of the
        # subsampling seed so r never varies across discovery seeds).
        pa_rng = np.random.default_rng(0)
        null_tops = []
        for _ in range(3):
            Xp = np.column_stack([pa_rng.permutation(Xc[:, j]) for j in range(N)])
            null_tops.append(np.linalg.svd(Xp, compute_uv=False)[0])
        threshold = float(np.mean(null_tops))
        r = int(np.sum(s > threshold))
        if r == 0:
            # fall back to the variance target when nothing beats the null
            explained = np.cumsum(s**2) / total_var
            r = int(np.searchsorted(explained, float(config['variance_target'])) + 1)
        r = max(1, min(r, r_cap, len(s)))

    # loadings scaled by singular values so magnitude reflects variance share
    loadings = vt[:r].T * (s[:r] / np.sqrt(max(T1, 1)))  # (N, r)
    R = _varimax(loadings)
    loadings = loadings @ R

    # scores consistent with rotated loadings via least squares:
    # X ≈ scores @ loadings.T  ->  scores = X @ L (L.T L)^-1
    gram = loadings.T @ loadings
    gram_inv = np.linalg.pinv(gram)
    scores = Xc @ loadings @ gram_inv  # (T-1, r)

    loadings, scores = _fix_factor_signs(loadings, scores)

    # soft-threshold small loadings so each factor is dominated by a nameable set
    tau = float(config['soft_threshold']) * np.abs(loadings).max(axis=0, keepdims=True)
    loadings = np.sign(loadings) * np.maximum(np.abs(loadings) - tau, 0.0)

    # composite trend tracks in level space
    factors_level = np.concatenate(
        [np.zeros((1, scores.shape[1])), np.cumsum(scores, axis=0)], axis=0
    )
    return factors_level, loadings, scores


def name_factors(
    loadings: np.ndarray, series_names: list, series_metadata: Optional[list]
) -> list:
    """Name each factor by the modal shared metadata attribute of its top loaders."""
    r = loadings.shape[1]
    metadata_lookup = {}
    for meta in series_metadata or []:
        if isinstance(meta, dict):
            name = meta.get('name')
            attrs = {k: v for k, v in meta.items() if k != 'name'}
        else:
            name = getattr(meta, 'name', None)
            attrs = dict(getattr(meta, 'attribute_values', {}) or {})
        if name is not None:
            metadata_lookup[name] = attrs

    names = []
    for j in range(r):
        col = np.abs(loadings[:, j])
        if not np.any(col > 0):
            names.append(f'factor_{j + 1}')
            continue
        top_idx = np.argsort(col)[::-1][: max(3, int(np.sum(col > 0) // 3) or 1)]
        top_series = [series_names[i] for i in top_idx if col[i] > 0]
        best_label = None
        best_share = 0.0
        attr_values = {}
        for series in top_series:
            for attr, value in metadata_lookup.get(series, {}).items():
                if value in (None, ''):
                    continue
                attr_values.setdefault(attr, []).append(str(value))
        for attr, values in attr_values.items():
            counts = pd.Series(values).value_counts()
            share = counts.iloc[0] / len(top_series)
            if share > best_share:
                best_share = share
                best_label = f'{counts.index[0]}'
        if best_label is not None and best_share >= 0.5:
            names.append(f'{best_label} (factor_{j + 1})')
        elif top_series:
            names.append(f'{top_series[0]}-led (factor_{j + 1})')
        else:
            names.append(f'factor_{j + 1}')
    return names


# ---------------------------------------------------------------------------
# D-3: deconfounding
# ---------------------------------------------------------------------------


def deconfound(
    X: np.ndarray,
    scores: np.ndarray,
    factor_lags=(0, 1),
    shared: np.ndarray = None,
) -> np.ndarray:
    """Regress each series on the factors at the given lags; return residuals.

    Downstream edges then mean "B leads A after the shared macro force is
    removed."

    Args:
        shared: optional (T, M) extra nuisance columns (detected seasonality,
            holidays, shifts, anomalies), entered contemporaneously only --
            lagging a calendar effect would let it absorb real lead-lag signal.
    """
    T1, N = X.shape
    max_l = max(factor_lags) if len(factor_lags) else 0
    has_shared = shared is not None and np.size(shared) > 0
    if (scores.shape[1] == 0 and not has_shared) or T1 <= max_l + 2:
        return X.copy()
    design_cols = [np.ones((T1 - max_l, 1))]
    for lag in factor_lags:
        design_cols.append(scores[max_l - lag : T1 - lag])
    if has_shared:
        shared = np.asarray(shared, dtype=float)
        if shared.ndim == 1:
            shared = shared[:, None]
        if shared.shape[0] >= T1:
            block = shared[max_l:T1]
            # drop constant / degenerate nuisance columns: they duplicate the
            # intercept and only make the least-squares solve ill-conditioned
            keep = np.std(block, axis=0) > 1e-12
            if keep.any():
                design_cols.append(block[:, keep])
    design = np.concatenate(design_cols, axis=1)
    Y = X[max_l:]
    beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
    return Y - design @ beta


def shared_component_columns(components: dict, columns=None) -> np.ndarray:
    """(T, M) panel-mean nuisance columns from a detector component dict.

    One column per available component family. Uses the panel mean, not the
    per-series value, so the per-series remainder stays visible to screening.
    """
    if not components:
        return None
    blocks = []
    for key in ('seasonality', 'holidays', 'level_shifts', 'anomalies'):
        comp = components.get(key)
        if comp is None:
            continue
        values = np.asarray(
            (
                comp[columns].values
                if columns is not None and hasattr(comp, 'columns')
                else (comp.values if hasattr(comp, 'values') else comp)
            ),
            dtype=float,
        )
        if values.ndim != 2 or values.size == 0:
            continue
        with np.errstate(invalid='ignore'):
            blocks.append(np.nanmean(values, axis=1))
    if not blocks:
        return None
    out = np.column_stack(blocks)
    return np.where(np.isfinite(out), out, 0.0)


def circular_shift_null(
    residuals: np.ndarray,
    screen_fn,
    n_reps: int = 20,
    seed: int = 42,
    quantile: float = 0.95,
) -> dict:
    """How many edges does this screening procedure invent on coupling-free data?

    Circularly shifting each series destroys cross-series timing while
    preserving its own autocorrelation and marginal distribution, so any edge
    still found is manufactured by the procedure rather than the data.

    Args:
        screen_fn: callable (T, N) -> number of edges found.

    Returns:
        dict with ``counts``, ``mean``, ``quantile`` and ``floor``.
    """
    residuals = np.asarray(residuals, dtype=float)
    T, N = residuals.shape
    if n_reps <= 0 or T < 4 or N < 2:
        return {'counts': [], 'mean': None, 'quantile': quantile, 'floor': None}
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(int(n_reps)):
        shifted = np.column_stack(
            [np.roll(residuals[:, i], int(rng.integers(1, T))) for i in range(N)]
        )
        try:
            counts.append(int(screen_fn(shifted)))
        except Exception:  # pragma: no cover - a failed replicate is not a null
            continue
    if not counts:
        return {'counts': [], 'mean': None, 'quantile': quantile, 'floor': None}
    return {
        'counts': counts,
        'mean': float(np.mean(counts)),
        'quantile': float(quantile),
        'floor': float(np.quantile(counts, quantile)),
    }


# ---------------------------------------------------------------------------
# D-4: conditional screening
# ---------------------------------------------------------------------------


def conditional_candidates(R: np.ndarray, top_k: int = 8, max_lag: int = 3) -> dict:
    """Candidate neighbors per node: conditional dependence + lagged screening.

    Contemporaneous screening uses GraphicalLassoCV when well-posed
    (Ledoit-Wolf + pinv fallback when N > T/2 or it fails to converge) —
    structurally what pairwise Granger cannot do: a candidate survives
    conditioning on all other series. Because a purely LAGGED parent can
    leave little contemporaneous partial correlation, the candidate set is
    unioned with the top-k series by |cross-correlation| over lags
    1..max_lag. Stability-selected lasso downstream rejects the false ones.
    """
    T, N = R.shape
    if N < 2:
        return {}
    Rs = R / np.maximum(R.std(axis=0, keepdims=True), 1e-12)

    precision = None
    if N <= T / 2:
        try:
            from sklearn.covariance import GraphicalLassoCV

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = GraphicalLassoCV(max_iter=200)
                model.fit(Rs)
            precision = model.precision_
        except Exception:
            precision = None
    if precision is None:
        try:
            from autots.tools.hierarchial import ledoit_wolf_covariance

            cov = ledoit_wolf_covariance(Rs)
        except Exception:
            cov = np.cov(Rs, rowvar=False) + 1e-6 * np.eye(N)
        precision = np.linalg.pinv(cov)

    d = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    partial = -precision / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)

    # lagged cross-correlation screening: max |corr(R_i[lag:], R_j[:-lag])|
    lagged_strength = np.zeros((N, N))
    for lag in range(1, max_lag + 1):
        if T - lag < 8:
            break
        a = Rs[lag:]
        b = Rs[: T - lag]
        a_c = a - a.mean(axis=0)
        b_c = b - b.mean(axis=0)
        denom = np.outer(
            np.maximum(np.linalg.norm(b_c, axis=0), 1e-12),
            np.maximum(np.linalg.norm(a_c, axis=0), 1e-12),
        )
        cross = np.abs(b_c.T @ a_c) / denom  # [j, i] = corr(j lagged -> i)
        lagged_strength = np.maximum(lagged_strength, cross)
    np.fill_diagonal(lagged_strength, 0.0)

    candidates = {}
    for i in range(N):
        strengths = np.abs(partial[i])
        # tolerance: pinv fallback never yields exact zeros
        nonzero = np.where(strengths > 1e-4)[0]
        order = nonzero[np.argsort(strengths[nonzero])[::-1]]
        selected = [int(j) for j in order[:top_k]]
        lag_order = np.argsort(lagged_strength[:, i])[::-1]
        for j in lag_order[:top_k]:
            if lagged_strength[j, i] > 0.1 and int(j) not in selected:
                selected.append(int(j))
        candidates[i] = selected[: 2 * top_k]
    return candidates


# ---------------------------------------------------------------------------
# D-5: stability-selected lasso VAR (direction from lag structure, full stop)
# ---------------------------------------------------------------------------


def _lagged_design(R: np.ndarray, target: int, sources: list, max_lag: int) -> tuple:
    """Build y, own-lag design, and candidate-lag design for one target."""
    T = R.shape[0]
    rows = T - max_lag
    y = R[max_lag:, target]
    own = np.column_stack(
        [R[max_lag - lag : T - lag, target] for lag in range(1, max_lag + 1)]
    )
    cand_cols = []
    cand_meta = []  # (source, lag)
    for j in sources:
        for lag in range(1, max_lag + 1):
            cand_cols.append(R[max_lag - lag : T - lag, j])
            cand_meta.append((j, lag))
    cand = np.column_stack(cand_cols) if cand_cols else np.zeros((rows, 0))
    return y, own, cand, cand_meta


def _residualize_on_own_lags(y: np.ndarray, own: np.ndarray) -> np.ndarray:
    """OLS-residualize the target on its own lags so autoregression is never
    misattributed to neighbors (own lags are effectively unpenalized)."""
    design = np.column_stack([np.ones(len(y)), own])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def stability_selected_edges(
    R: np.ndarray,
    candidates: dict,
    config: dict,
    rng: np.random.Generator = None,
) -> list:
    """Per-target stability-selected LassoLarsIC over lagged candidates.

    Subsamples are contiguous windows (block length >> 2 * max_lag) because a
    plain iid bootstrap is wrong under serial dependence. The windows are a
    DETERMINISTIC rolling grid over the sample — seed-variance of the
    discovered topology is zero by construction. An edge (j -> i, lag) is
    kept when its selection frequency >= stability_threshold.
    """
    from sklearn.linear_model import LassoLarsIC

    max_lag = int(config['max_lag'])
    B = int(config['n_subsamples'])
    frac = float(config['subsample_frac'])
    threshold = float(config['stability_threshold'])
    T, N = R.shape
    edges = []
    if T <= max_lag * 4 + 8:
        return edges

    block_len = max(2 * max_lag, 4)

    for i in range(N):
        sources = candidates.get(i, [])
        if not sources:
            continue
        y, own, cand, cand_meta = _lagged_design(R, i, sources, max_lag)
        if cand.shape[1] == 0:
            continue
        y_resid = _residualize_on_own_lags(y, own)
        rows = len(y_resid)
        sub_len = max(int(rows * frac), block_len * 2)
        if sub_len >= rows:
            sub_len = rows

        # deterministic rolling grid of contiguous windows
        max_start = max(rows - sub_len, 0)
        starts = np.unique(np.linspace(0, max_start, num=max(B, 1)).astype(int))

        counts = np.zeros(cand.shape[1])
        coef_samples = [[] for _ in range(cand.shape[1])]
        n_draws = 0
        for s in starts:
            idx = np.arange(s, min(s + sub_len, rows))
            n_draws += 1
            ys = y_resid[idx]
            Xs = cand[idx]
            col_std = Xs.std(axis=0)
            usable = col_std > 1e-12
            if not usable.any() or ys.std() <= 1e-12:
                continue
            Xn = np.zeros_like(Xs)
            Xn[:, usable] = (Xs[:, usable] - Xs[:, usable].mean(axis=0)) / col_std[
                usable
            ]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = LassoLarsIC(criterion='bic')
                    model.fit(Xn, ys - ys.mean())
            except Exception:
                continue
            nonzero = np.abs(model.coef_) > 1e-10
            counts += nonzero
            for c in np.where(nonzero)[0]:
                coef_samples[c].append(model.coef_[c])

        freq = counts / max(n_draws, 1)
        # per (source) keep only the best lag, so one relationship isn't
        # split into three weak duplicate edges
        best_by_source = {}
        for c, (source, lag) in enumerate(cand_meta):
            if freq[c] < threshold or not coef_samples[c]:
                continue
            med = float(np.median(coef_samples[c]))
            weight = float(freq[c] * abs(med))
            prev = best_by_source.get(source)
            if prev is None or weight > prev['weight']:
                best_by_source[source] = {
                    'source': source,
                    'target': i,
                    'lag': int(lag),
                    'sign': int(np.sign(med)) or 1,
                    'weight': weight,
                    'stability': float(freq[c]),
                    'median_coef': med,
                }
        edges.extend(best_by_source.values())
    return edges


# ---------------------------------------------------------------------------
# D-9: edge-level falsification
# ---------------------------------------------------------------------------


def falsify_edges(R: np.ndarray, edges: list, config: dict) -> list:
    """Held-out one-step MSE with vs without each edge (zeroed coefficient).

    Effect-level validation: an edge that does not reduce held-out error is
    pruned regardless of how significant it looked in-sample.
    """
    max_lag = int(config['max_lag'])
    holdout_frac = float(config['holdout_frac'])
    T, N = R.shape
    if not edges or T <= max_lag * 5:
        return edges

    surviving = []
    by_target = {}
    for edge in edges:
        by_target.setdefault(edge['target'], []).append(edge)

    for target, target_edges in by_target.items():
        sources = [e['source'] for e in target_edges]
        y, own, cand, cand_meta = _lagged_design(R, target, sources, max_lag)
        rows = len(y)
        split = max(int(rows * (1.0 - holdout_frac)), max_lag + 2)
        if split >= rows:
            surviving.extend(target_edges)
            continue
        design = np.column_stack([np.ones(rows), own, cand])
        beta, *_ = np.linalg.lstsq(design[:split], y[:split], rcond=None)
        pred = design[split:] @ beta
        base_mse = float(np.mean((y[split:] - pred) ** 2))

        for edge in target_edges:
            # zero every coefficient belonging to this edge's source
            beta_zeroed = beta.copy()
            for c, (source, _lag) in enumerate(cand_meta):
                if source == edge['source']:
                    beta_zeroed[1 + own.shape[1] + c] = 0.0
            pred_wo = design[split:] @ beta_zeroed
            mse_wo = float(np.mean((y[split:] - pred_wo) ** 2))
            delta = mse_wo - base_mse  # positive => edge helps
            edge['delta_mse'] = delta
            if delta > 0:
                surviving.append(edge)
    return surviving


# ---------------------------------------------------------------------------
# D-6: LIFT-style lead-lag family
# ---------------------------------------------------------------------------


def leadlag_edges(R: np.ndarray, config: dict) -> list:
    """Windowed cross-correlation lead-lag candidates on trailing residuals.

    Keep j -> i where the |cross-correlation| over positive lags of j peaks
    at lag > 0 and exceeds leadlag_min_corr; top leadlag_top_k per target.
    Cheap, fitting-free, recomputable at predict time.
    """
    L = int(config['leadlag_max_lag'])
    min_corr = float(config['leadlag_min_corr'])
    top_k = int(config['leadlag_top_k'])
    window = int(config['leadlag_window'])
    T, N = R.shape
    if T <= L * 3 or N < 2:
        return []
    Rw = R[-min(T, window) :]
    Rw = (Rw - Rw.mean(axis=0)) / np.maximum(Rw.std(axis=0), 1e-12)
    rows = Rw.shape[0]
    edges = []
    for i in range(N):
        scored = []
        for j in range(N):
            if j == i:
                continue
            best_corr = 0.0
            best_lag = 0
            for lag in range(0, L + 1):
                a = Rw[lag:, i]
                b = Rw[: rows - lag, j]
                if len(a) < 8:
                    continue
                c = float(np.corrcoef(a, b)[0, 1])
                if np.isfinite(c) and abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag = lag
            if best_lag > 0 and abs(best_corr) >= min_corr:
                scored.append((abs(best_corr), j, best_lag, best_corr))
        scored.sort(reverse=True)
        for strength, j, lag, corr in scored[:top_k]:
            edges.append(
                {
                    'source': j,
                    'target': i,
                    'lag': int(lag),
                    'sign': int(np.sign(corr)) or 1,
                    'weight': float(strength),
                    'stability': float(strength),
                    'median_coef': float(corr),
                }
            )
    return edges


# ---------------------------------------------------------------------------
# D-8: merge + entry point
# ---------------------------------------------------------------------------


def _edges_from_adjacency(adjacency: np.ndarray, family: str) -> list:
    """Convert an (N, N) signed adjacency into edge rows.

    Lag stays 0 -- a co-movement prior makes no lag claim -- but the sign is
    carried through, so a substitution prior ("storm up shovels, down
    everything else") survives instead of being silently dropped.
    """
    edges = []
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        return edges
    n = adjacency.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = float(adjacency[i, j])
            if w != 0 and np.isfinite(w):
                edges.append(
                    {
                        'source': i,
                        'target': j,
                        'lag': 0,
                        'sign': int(np.sign(w)),
                        'weight': abs(w),
                        'stability': abs(w),
                        'median_coef': w,
                        'family': family,
                        'delta_mse': None,
                    }
                )
    return edges


def discover_structure(
    trend_df: pd.DataFrame,
    anchor_mask: Optional[np.ndarray] = None,
    series_metadata: Optional[list] = None,
    config: Optional[dict] = None,
    seed: int = 42,
    extra_adjacencies: Optional[dict] = None,
    external_factors: Optional[np.ndarray] = None,
    components: Optional[dict] = None,
) -> dict:
    """Discover factors, loadings, and a conditional lead-lag edge table.

    Args:
        trend_df: (T, N) wide DataFrame of decomposed trend components.
        anchor_mask: optional (N,) bool; factor extraction uses anchors only
            (responders still get loadings via regression on the factors).
        series_metadata: optional list of SeriesMetadata (or dicts) used for
            factor naming.
        config: overrides for DEFAULT_DISCOVERY_CONFIG.
        seed: RNG seed for block subsampling (topology is deterministic
            given seed).
        extra_adjacencies: optional {family_name: (N, N) adjacency} for the
            'event', 'metadata', 'business', and 'causal' families built
            elsewhere. Signed entries are honoured.
        external_factors: optional (T, K) level-space factor paths to
            deconfound against instead of the internal difference-space SVD.
        components: optional detector component dict (seasonality, holidays,
            level_shifts, anomalies), regressed out alongside the factors when
            ``config['deconfound_components']`` is set.

    Returns:
        dict with 'factors' (T, r) level-space composite tracks, 'loadings'
        (N, r) sparse signed, 'factor_names', 'edges' (list of dicts over
        real series names), 'adjacency' (N, N) merged dense form,
        'family_adjacencies' {family: (N, N)}, 'communities' (N,),
        'config' (resolved).
    """
    cfg = _resolve_config(config)
    rng = np.random.default_rng(seed)
    series_names = list(trend_df.columns)
    N = len(series_names)
    values = trend_df.values.astype(float)
    T = values.shape[0]

    if N == 0 or T < 8:
        return {
            'factors': np.zeros((T, 1)),
            'loadings': np.zeros((N, 1)),
            'factor_names': ['factor_1'],
            'edges': [],
            'adjacency': np.zeros((N, N), dtype=np.float32),
            'family_adjacencies': {},
            'communities': np.zeros(N, dtype=int),
            'config': cfg,
        }

    # D-1: difference + standardize
    X, _scale = _difference_and_standardize(values)

    # D-1b: a separate, HP-smoothed view used for factor extraction.
    hp_lambda = cfg.get('factor_hp_lambda')
    if hp_lambda:
        X_factor, _ = _difference_and_standardize(_hp_smooth(values, hp_lambda))
    else:
        X_factor = X

    # D-2: factors from anchors (short-history series shouldn't shape them)
    if anchor_mask is not None and np.any(anchor_mask) and not np.all(anchor_mask):
        anchor_idx = np.where(np.asarray(anchor_mask, dtype=bool))[0]
        _f_anchor, _l_anchor, scores = extract_factors(X_factor[:, anchor_idx], cfg)
        # loadings for ALL series by regression on the factor scores
        gram = scores.T @ scores
        loadings = (X_factor.T @ scores) @ np.linalg.pinv(gram)
        tau = float(cfg['soft_threshold']) * np.abs(loadings).max(axis=0, keepdims=True)
        loadings = np.sign(loadings) * np.maximum(np.abs(loadings) - tau, 0.0)
        factors = np.concatenate(
            [np.zeros((1, scores.shape[1])), np.cumsum(scores, axis=0)], axis=0
        )
    elif external_factors is not None:
        supplied = np.asarray(
            (
                external_factors.values
                if hasattr(external_factors, 'values')
                else external_factors
            ),
            dtype=float,
        )
        if supplied.ndim == 1:
            supplied = supplied[:, None]
        scores = np.diff(supplied, axis=0)
        if scores.shape[0] != X.shape[0]:  # align to the differenced panel
            scores = (
                scores[-X.shape[0] :]
                if scores.shape[0] > X.shape[0]
                else np.pad(
                    scores, ((X.shape[0] - scores.shape[0], 0), (0, 0)), mode='edge'
                )
            )
        sd = scores.std(axis=0, keepdims=True)
        scores = np.divide(scores, sd, out=np.zeros_like(scores), where=sd > 1e-12)
        gram = scores.T @ scores
        loadings = (X_factor.T @ scores) @ np.linalg.pinv(gram)
        tau = float(cfg['soft_threshold']) * np.abs(loadings).max(axis=0, keepdims=True)
        loadings = np.sign(loadings) * np.maximum(np.abs(loadings) - tau, 0.0)
        factors = np.concatenate(
            [np.zeros((1, scores.shape[1])), np.cumsum(scores, axis=0)], axis=0
        )
    else:
        factors, loadings, scores = extract_factors(X_factor, cfg)

    factor_names = name_factors(loadings, series_names, series_metadata)

    # D-3: deconfound (factors, plus detected shared components when enabled)
    shared_cols = None
    if cfg.get('deconfound_components') and components:
        shared_cols = shared_component_columns(components, series_names)
    R = deconfound(X, scores, cfg['factor_lags'], shared=shared_cols)

    # D-4: conditional screening (contemporaneous + lagged)
    candidates = conditional_candidates(
        R, top_k=int(cfg['top_k']), max_lag=int(cfg['max_lag'])
    )

    # diagnostic only -- never drops edges, just reports the acceptance floor
    null_calibration = None
    if int(cfg.get('null_calibration_reps', 0) or 0) > 0:

        def _screen(shifted):
            cands = conditional_candidates(
                shifted, top_k=int(cfg['top_k']), max_lag=int(cfg['max_lag'])
            )
            return len(stability_selected_edges(shifted, cands, cfg, rng))

        null_calibration = circular_shift_null(
            R,
            _screen,
            n_reps=int(cfg['null_calibration_reps']),
            seed=seed,
            quantile=float(cfg.get('null_quantile', 0.95)),
        )

    # D-5: stability-selected lasso VAR + D-9 falsification
    lasso_edges = stability_selected_edges(R, candidates, cfg, rng)
    lasso_edges = falsify_edges(R, lasso_edges, cfg)
    for edge in lasso_edges:
        edge['family'] = 'lasso'
        edge.setdefault('delta_mse', None)

    # D-6: lead-lag family
    lift_edges = leadlag_edges(R, cfg)
    for edge in lift_edges:
        edge['family'] = 'leadlag'
        edge['delta_mse'] = None  # not regression-fitted; nothing to falsify

    edges = list(lasso_edges) + list(lift_edges)

    # D-8: extra families supplied by the caller (event, metadata,
    # business, causal)
    for family, adjacency in (extra_adjacencies or {}).items():
        if adjacency is None:
            continue
        edges.extend(_edges_from_adjacency(adjacency, family))

    # merge per family into dense adjacency (weights renormalized per family)
    family_adjacencies = {}
    for edge in edges:
        fam = edge['family']
        family_adjacencies.setdefault(fam, np.zeros((N, N), dtype=np.float32))
        family_adjacencies[fam][edge['source'], edge['target']] = max(
            family_adjacencies[fam][edge['source'], edge['target']],
            abs(edge['weight']),
        )
    merged = np.zeros((N, N), dtype=np.float32)
    fam_weights = dict(cfg['family_weights'])
    total_w = 0.0
    for fam, A_f in family_adjacencies.items():
        max_w = float(A_f.max())
        if max_w > 0:
            w = float(fam_weights.get(fam, 0.5))
            merged += w * (A_f / max_w)
            total_w += w
    if total_w > 0:
        merged = np.clip(merged / total_w, 0.0, 1.0)

    # convert edge indices to series names for the public table
    named_edges = []
    for edge in sorted(
        edges, key=lambda e: (e['family'], -abs(e['weight']), e['source'], e['target'])
    ):
        named_edges.append(
            {
                'source': series_names[edge['source']],
                'target': series_names[edge['target']],
                'lag': int(edge['lag']),
                'sign': int(edge['sign']),
                'weight': float(edge['weight']),
                'family': edge['family'],
                'stability': float(edge['stability']),
                'delta_mse': (
                    None if edge.get('delta_mse') is None else float(edge['delta_mse'])
                ),
            }
        )

    # communities: each series belongs to its dominant factor (or -1 if none)
    if loadings.shape[1] > 0 and np.any(np.abs(loadings) > 0):
        communities = np.where(
            np.abs(loadings).max(axis=1) > 0,
            np.abs(loadings).argmax(axis=1),
            -1,
        ).astype(int)
    else:
        communities = np.full(N, -1, dtype=int)

    return {
        'factors': factors,
        'loadings': loadings,
        'factor_names': factor_names,
        'edges': named_edges,
        'adjacency': merged,
        'family_adjacencies': family_adjacencies,
        'communities': communities,
        'null_calibration': null_calibration,
        'config': cfg,
    }


def forecast_factors(
    factors: np.ndarray, horizon: int, phi: float = 0.9, fit_window: int = 90
) -> np.ndarray:
    """Damped local-linear forecast of the factor tracks.

    Factors are few series with the full history each — a well-posed
    estimation problem — so a cheap damped-trend continuation is reliable.

    Args:
        factors: (T, r) level-space factor tracks.
        horizon: steps ahead.
        phi: damping coefficient.
        fit_window: trailing window used for the slope estimate.

    Returns:
        (horizon, r) forecast deltas RELATIVE to the last factor value.
    """
    T, r = factors.shape
    L = max(min(fit_window, T), 2)
    x = factors[-L:]
    t_idx = np.arange(L) - (L - 1) / 2.0
    denom = float(np.sum(t_idx**2)) or 1.0
    slope = (x * t_idx[:, None]).sum(axis=0) / denom  # (r,)
    steps = np.arange(1, horizon + 1)
    damp = np.cumsum(phi**steps)
    return damp[:, None] * slope[None, :]


def match_factors(true_factors, est_factors) -> dict:
    """Align estimated factors to known/true factors and score the recovery.

    Both inputs are level-space tracks (discovery returns cumulative sums of
    the differenced factor scores; generator factors are levels), so they are
    differenced before comparison — the shared *movement* is what identifies a
    factor, not its arbitrary integration constant. Matching is a Hungarian
    assignment on the |correlation| matrix, which handles both permutation and
    sign indeterminacy of the factor basis.

    Args:
        true_factors: (T, K) array-like (DataFrame ok) of reference factors.
        est_factors: (T, r) array-like of estimated factors. May have a
            different number of factors than ``true_factors``.

    Returns:
        dict with:
            'assignment': {true_index: est_index} for the matched pairs
            'correlations': {true_index: signed correlation} per matched pair
            'signs': {true_index: +1/-1} sign of each matched correlation
            'mean_abs_corr': mean |corr| over matched pairs, averaged over
                ``max(K, r)`` so missing/spurious factors are penalized
            'n_true', 'n_est': factor counts
    """
    true_arr = np.asarray(
        true_factors.values if hasattr(true_factors, 'values') else true_factors,
        dtype=float,
    )
    est_arr = np.asarray(
        est_factors.values if hasattr(est_factors, 'values') else est_factors,
        dtype=float,
    )
    if true_arr.ndim == 1:
        true_arr = true_arr[:, None]
    if est_arr.ndim == 1:
        est_arr = est_arr[:, None]

    empty = {
        'assignment': {},
        'correlations': {},
        'signs': {},
        'mean_abs_corr': 0.0,
        'n_true': int(true_arr.shape[1]),
        'n_est': int(est_arr.shape[1]),
    }
    if true_arr.size == 0 or est_arr.size == 0:
        return empty

    # difference to level-invariant movement, then trim to a common length
    d_true = np.diff(true_arr, axis=0)
    d_est = np.diff(est_arr, axis=0)
    T = min(d_true.shape[0], d_est.shape[0])
    if T < 3:
        return empty
    d_true = d_true[-T:]
    d_est = d_est[-T:]

    def _z(arr):
        arr = arr - arr.mean(axis=0, keepdims=True)
        sd = arr.std(axis=0, keepdims=True)
        return np.divide(arr, sd, out=np.zeros_like(arr), where=sd > 1e-12)

    zt = _z(d_true)
    ze = _z(d_est)
    corr = (zt.T @ ze) / float(T)  # (K, r)
    corr = np.nan_to_num(corr, nan=0.0)

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-np.abs(corr))
    except ImportError:  # pragma: no cover - scipy is a hard AutoTS dep
        rows, cols, used = [], [], set()
        for i in np.argsort(-np.abs(corr).max(axis=1)):
            order = np.argsort(-np.abs(corr[i]))
            for j in order:
                if j not in used:
                    rows.append(i)
                    cols.append(j)
                    used.add(j)
                    break
        rows, cols = np.array(rows, dtype=int), np.array(cols, dtype=int)

    assignment, correlations, signs = {}, {}, {}
    for i, j in zip(rows, cols):
        c = float(corr[i, j])
        assignment[int(i)] = int(j)
        correlations[int(i)] = c
        signs[int(i)] = 1 if c >= 0 else -1

    denom = max(true_arr.shape[1], est_arr.shape[1])
    mean_abs = sum(abs(c) for c in correlations.values()) / denom if denom else 0.0
    return {
        'assignment': assignment,
        'correlations': correlations,
        'signs': signs,
        'mean_abs_corr': float(mean_abs),
        'n_true': int(true_arr.shape[1]),
        'n_est': int(est_arr.shape[1]),
    }
