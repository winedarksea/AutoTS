# -*- coding: utf-8 -*-
"""
Learned latent-factor trend mode (``trend_network='factor'``).
"""

import warnings
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:  # pragma: no cover - torch-free environments
    HAS_TORCH = False


DEFAULT_FACTOR_CONFIG = {
    # identification (alternating estimator)
    'knot_spacing': 7,
    'alpha': 1e-3,
    'alt_iters': 6,
    'init_window': 91,
    'gls': True,
    'max_lag': 0,
    # refinement (stage A)
    'stage_a_steps': 600,
    'lr_coef': 0.0,
    'lr_aux': 1e-2,
    'patience': 120,
    'check_every': 20,
    'val_frac': 0.1,
    'w_decorr': 0.0,
    'w_lag_mean': 0.01,
    'w_lag_entropy': 1e-3,
    'w_l1_loadings': 1e-4,
    'w_prox': 1e-4,
    # ---- C1: identification-basis rotation (default off) -------------------
    # None | 'varimax' | 'quartimax' | 'promax'. Resolves the rotational
    # indeterminacy the alternating estimator leaves open; reconstruction (and
    # therefore accuracy) is invariant, only the basis changes.
    'rotate': None,
    'rotate_kaiser': True,
    # ---- C3: sparse loading solve (default off; 0.0 == today's lstsq) ------
    'loading_l1': 0.0,
    'loading_l1_adaptive': True,
    'loading_relax': True,
    # proximal soft-threshold on model.loadings after each stage-A step,
    # without which 600 Adam steps wash the identified zeros back out
    'w_prox_loadings': 0.0,
    # ---- user loading-graph prior (0.0 == exact no-op) ---------------------
    # A prior on which series share factors IS a prior on the loading matrix.
    # `prior_adjacency` is an (N, N) signed graph; at w_prior_loadings > 0 a
    # penalty pulls linked series' normalized loading rows together and pushes
    # negatively-linked ones apart. At weight 0 the term is never constructed.
    'w_prior_loadings': 0.0,
    'prior_adjacency': None,
    # ---- C4: per-factor split-half stability (0 == not computed) ----------
    # Feeds coherence.resolve_signs' long-dead `stability` argument, so a
    # factor nobody can reproduce cannot partition the panel.
    'factor_stability_reps': 0,
    # ---- C9: sparse-code identification (default off) ---------------------
    # 'alternating' == today. 'sparse_alt' is torch-free sparse dictionary
    # learning over series; 'sparse_ae' is the gradient autoencoder and falls
    # back to 'sparse_alt' without torch. Both refine *from* the alternating
    # fit and are rejected back to it if they reconstruct materially worse, so
    # turning one on cannot leave a panel worse identified than the default.
    'identification': 'alternating',
    'sparse_config': None,
    # Project stage A's loadings back onto the identified support after each
    # step. Without it the sparse structure never reaches fitted_loadings(),
    # which is what the coherence graph actually reads -- the same class of
    # miss the C5 measurement trap turned out to be. Inert unless a sparse
    # identification actually ran.
    'sparse_freeze_support': True,
    # ---- I1: cross-series level-shift veto (default off) ------------------
    # The changepoint detector is univariate, so one shared factor move is
    # charged to N series as N independent level shifts and then subtracted
    # out of the panel entirely -- the largest measured loss in the pipeline
    # (true-factor R^2 0.338 -> 0.156). With the veto on, a step that most of
    # the panel takes in the same direction stays in the panel.
    # False | True | 'all'. Thresholds are calibrated against what the
    # detector actually produces: on the primary synthetic cell it emits 38
    # step events across 24 series, and at most 17% of the panel steps within
    # +/-7 days of each other, so the obvious 30%-at-7-days rule never fires.
    'level_shift_veto': False,
    'level_shift_min_share': 0.15,
    'level_shift_min_agreement': 0.60,
    'level_shift_window': 31,
    'level_shift_veto_shrink': 1.0,
    # extrapolation (stage B)
    'stage_b_steps': 200,
    'lr_phi': 1e-2,
    'n_origins': 64,
    'slope_window': 90,
    'min_trend_to_noise': 0.6,
    # exposure
    'prune_share': 0.02,
    'grad_clip': 1.0,
    # ---- Phase 1 safety layer (all default-off until each clears its gate) --
    # 1a: pick the factor continuation rule by held-out reconstruction error
    'continuation_select': False,
    'continuation_origins': 3,
    'continuation_config': None,
    # 1c: zero loadings of series the factor model forecasts worse than a
    # damped local-linear baseline on their own raw target
    'gate_forecast_margin': None,
    # 1d: quarantine series whose recent tail is numerically constant
    'frozen_tail_gate': False,
    'frozen_tail_min_len': 14,
    # 1b / 1b' / 1e: post-forecast safety layer (see tva/safety.py)
    'sn_blend': False,
    'error_cap': False,
    'reanchor': False,
    'conformal_sigma': False,
    'inner_folds': 3,
    # refit the factor stage on truncated history so inner validation origins
    # aren't in-sample (only the factor stage is refit, the extrapolation
    # being graded)
    'inner_refit': False,
    'safety_config': None,
    # ---- Phase 2 -----------------------------------------------------------
    # 2a: per-series seasonal-path arbitration (datepart / empirical / amplitude)
    'seasonal_arbitration': False,
    'seasonal_config': None,
    # 2c: model the trend block in log space
    'space': 'level',
    'log_epsilon': 1e-6,
    # 2b: alternative trend-isolation input ('detector' | 'robust')
    'input_estimator': 'detector',
    'input_config': None,
    # ---- Phase 4 -----------------------------------------------------------
    # post-forecast coherence shrink on standardized trend trajectories;
    # default-off, re-gated on loading recovery
    'coherence': False,
    'coherence_config': None,
    # ---- Phase 3 -----------------------------------------------------------
    # 3b: fit shared factors on long-history *anchors* only, then project
    # responder series onto the frozen paths via their own observed overlap
    # (avoids handing a late-launching series full weight via ffill/bfill)
    'anchor_selection': False,
    # multiplied by the forecast horizon to get the anchor min_observed threshold
    'min_observed_multiple': 3.0,
    # below this a responder can't identify a loading vector; it is zeroed
    # and reported in info['insufficient_overlap']
    'min_responder_overlap': 90,
    # ridge on the responder projection and a cap on assigned shared movement:
    # a short overlap can make the factor columns nearly collinear with
    # [1, t], and an unregularized loading then explodes under extrapolation
    'responder_ridge': 1e-2,
    'responder_loading_cap': 2.0,
    # 3c: data-derived group factors underneath the global ones
    'group_factors': False,
    'group_stability_threshold': 0.70,
    'group_refits': 8,
    'rank_candidates': (0, 1, 2, 3, 4, 6),
}


def hinge_design(
    n_time: int, knot_spacing: int, max_knots: int = 200
) -> np.ndarray:
    """Trend-filtering basis: ``[t/T, (t-k)_+/T]`` for knots every ``spacing``.

    ``B @ c`` is piecewise linear with breakpoints only where ``c`` is nonzero,
    and is exactly zero at ``t = 0`` — which is the model's factor anchor.

    Args:
        n_time: number of time steps T.
        knot_spacing: steps between candidate knots (>= 1).
        max_knots: cap on candidate knots; widens spacing instead, since the
            l1 solve is superlinear in column count.

    Returns:
        (T, P) design matrix, float32.
    """
    spacing = max(int(knot_spacing), 1)
    if max_knots and n_time // spacing > max_knots:
        spacing = int(np.ceil(n_time / float(max_knots)))
    t = np.arange(n_time, dtype=float)
    knots = np.arange(spacing, max(n_time - spacing, spacing + 1), spacing)
    cols = [t / float(n_time)] + [
        np.clip(t - k, 0.0, None) / float(n_time) for k in knots
    ]
    return np.column_stack(cols).astype(np.float32)


def hinge_knot_times(
    n_time: int, knot_spacing: int, max_knots: int = 200
) -> np.ndarray:
    """Time indices of the hinge knots ``hinge_design`` would build.

    Derived rather than stored so callers (e.g. the regime-slope continuation
    candidate) can't drift out of sync with ``hinge_design``'s widening rule.
    """
    spacing = max(int(knot_spacing), 1)
    if max_knots and n_time // spacing > max_knots:
        spacing = int(np.ceil(n_time / float(max_knots)))
    return np.arange(spacing, max(n_time - spacing, spacing + 1), spacing)


def robust_level_scale(values: np.ndarray, floor_frac: float = 1e-3):
    """Per-series robust center/scale in **level** space.

    Args:
        values: (T, N) adjusted level panel.
        floor_frac: floor on the scale as a fraction of the panel's median
            scale, so near-constant series do not explode when normalized.

    Returns:
        (center (N,), scale (N,)) float arrays.
    """
    arr = np.asarray(values, dtype=float)
    center = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - center[None, :]), axis=0)
    scale = 1.4826 * mad
    bad = ~np.isfinite(scale) | (scale <= 0)
    if np.any(bad):
        alt = np.nanstd(arr[:, bad], axis=0)
        scale[bad] = np.where(np.isfinite(alt) & (alt > 0), alt, 1.0)
    finite = scale[np.isfinite(scale) & (scale > 0)]
    typical = float(np.median(finite)) if finite.size else 1.0
    scale = np.maximum(scale, max(typical * floor_frac, 1e-8))
    return np.nan_to_num(center, nan=0.0), scale


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with edge replication (no zero-pad artifacts)."""
    w = max(int(window), 1)
    if w <= 1:
        return arr.copy()
    half = w // 2
    padded = np.pad(arr, ((half, half), (0, 0)), mode='edge')
    kern = np.ones(w) / w
    out = np.vstack(
        [np.convolve(padded[:, i], kern, mode='valid')[: arr.shape[0]]
         for i in range(arr.shape[1])]
    ).T
    return out


def _l1_trend_filter(scores: np.ndarray, design: np.ndarray, alpha: float):
    """Column-wise l1 trend filtering of noisy factor scores.

    Each score column is standardized before the Lasso and rescaled after, so
    ``alpha`` is a pure smoothness knob rather than something retuned per
    panel scale.

    Returns:
        (fitted (T, K) centered, coefficients (P, K)).
    """
    from sklearn.linear_model import Lasso

    T, K = scores.shape
    fitted = np.empty((T, K), dtype=float)
    coefs = np.zeros((design.shape[1], K), dtype=float)
    for k in range(K):
        sd = float(scores[:, k].std()) or 1.0
        model = Lasso(alpha=alpha, max_iter=5000, tol=1e-5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(design, scores[:, k] / sd)
        coefs[:, k] = model.coef_ * sd
        # centered, i.e. equivalent to keeping the Lasso intercept
        col = design @ coefs[:, k]
        fitted[:, k] = col - col.mean()
    return fitted, coefs


def select_n_factors(values: np.ndarray, cap: int = 6, window: int = 181) -> int:
    """Rank hint from the smoothed level panel's singular spectrum.

    Daily noise dominates the raw spectrum, so it is smoothed first; rank is
    the Ahn-Horenstein eigenvalue-ratio argmax (``argmax_k s_k / s_{k+1}``),
    which needs no threshold calibration.

    Returns:
        Integer in [1, cap].
    """
    arr = np.asarray(values, dtype=float)
    cap = max(int(cap), 1)
    sm = _rolling_mean(arr - arr.mean(axis=0, keepdims=True), min(window, max(len(arr) // 4, 1)))
    sm = sm - sm.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(sm, compute_uv=False)
    except np.linalg.LinAlgError:  # pragma: no cover
        return 1
    if s.size < 2 or s[0] <= 0:
        return 1
    ratios = s[:-1] / np.maximum(s[1:], 1e-12)
    return int(np.clip(int(np.argmax(ratios[:cap])) + 1, 1, cap))


def _fit_loadings(factors: np.ndarray, yc: np.ndarray, l1: float = 0.0,
                  adaptive: bool = True, relax: bool = True) -> np.ndarray:
    """Solve the loading matrix given fixed factor paths. Returns (K, N).

    ``l1 == 0`` is the plain least-squares solve the alternating estimator has
    always used, returned bit-identically so the sparse path is opt-in.

    ``l1 > 0`` fits a per-series lasso instead. This is the identifying
    restriction that least squares cannot express: lstsq is rotation-invariant,
    so it is equally happy with the true simple-structure loadings and with any
    rotation of them, whereas an l1 penalty prefers the sparse one. Two
    refinements make the penalty pay for structure rather than for shrinkage:

    * ``adaptive`` re-weights each coefficient's penalty by ``1/|lstsq est|``,
      so an already-large loading is barely penalized and a near-zero one is
      pushed to exactly zero (plain lasso would shrink the big ones most, which
      is where the signal is);
    * ``relax`` refits unpenalized least squares on the selected support, so
      surviving coefficients are unbiased and only *selection* came from the
      penalty.

    The target column is standardized before the fit and rescaled after, so
    ``l1`` is a scale-free sparsity knob rather than something retuned per
    panel.
    """
    base, *_ = np.linalg.lstsq(factors, yc, rcond=None)  # (K, N)
    if not l1 or float(l1) <= 0:
        return base
    from sklearn.linear_model import Lasso

    K, N = base.shape
    eps = 1e-6
    out = np.zeros_like(base)
    for j in range(N):
        target = yc[:, j]
        sd = float(np.std(target))
        if not np.isfinite(sd) or sd <= 1e-12:
            continue  # constant series: no loading is identified
        col = base[:, j]
        if adaptive:
            penalty_w = 1.0 / (np.abs(col) + eps)
        else:
            penalty_w = np.ones(K)
        # penalizing |b_k * w_k| is the same as an unweighted lasso on
        # columns scaled by 1 / w_k
        design = factors / penalty_w[None, :]
        model = Lasso(alpha=float(l1), max_iter=5000, tol=1e-5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(design, target / sd)
        coef = (model.coef_ / penalty_w) * sd
        support = np.abs(coef) > 0
        if not support.any():
            continue  # the lasso's honest answer: this series loads nothing
        if relax:
            refit, *_ = np.linalg.lstsq(factors[:, support], target, rcond=None)
            coef = np.zeros(K)
            coef[support] = refit
        out[:, j] = coef
    return out


def estimate_factors_alternating(
    values: np.ndarray,
    n_factors: int,
    knot_spacing: int = 7,
    alpha: float = 1e-3,
    iters: int = 6,
    init_window: int = 91,
    gls: bool = True,
    loading_l1: float = 0.0,
    loading_l1_adaptive: bool = True,
    loading_relax: bool = True,
):
    """Identify latent factor paths and loadings by alternating GLS/l1-TF.

    Torch-free. This is the stage that actually finds the factors; the torch
    model refines around it.

    ``loading_l1 > 0`` swaps every loading solve for the sparse solve in
    :func:`_fit_loadings` — including the final one, which would otherwise
    overwrite the sparse structure the loop just found.

    Returns:
        dict with 'factors' (T, K), 'loadings' (N, K), 'coefs' (P, K),
        'design' (T, P), 'weights' (N,).
    """
    def solve(F, Y):
        return _fit_loadings(
            F, Y, l1=loading_l1, adaptive=loading_l1_adaptive, relax=loading_relax
        )

    arr = np.asarray(values, dtype=float)
    T, N = arr.shape
    K = int(max(1, min(int(n_factors), N)))
    yc = arr - arr.mean(axis=0, keepdims=True)

    sm = _rolling_mean(yc, min(int(init_window), max(T // 3, 1)))
    sm = sm - sm.mean(axis=0, keepdims=True)
    try:
        U, S, _ = np.linalg.svd(sm, full_matrices=False)
        factors = U[:, :K] * S[:K][None, :]
    except np.linalg.LinAlgError:  # pragma: no cover
        factors = np.zeros((T, K))
    if factors.shape[1] < K:  # pragma: no cover - degenerate panels
        factors = np.pad(factors, ((0, 0), (0, K - factors.shape[1])))

    design = hinge_design(T, knot_spacing).astype(float)
    weights = np.ones(N)
    coefs = np.zeros((design.shape[1], K))
    loadings = np.zeros((K, N))
    for _ in range(max(int(iters), 1)):
        loadings = solve(factors, yc)  # (K, N)
        lw = loadings * weights[None, :]
        scores = (yc * weights[None, :]) @ np.linalg.pinv(lw)  # (T, K)
        factors, coefs = _l1_trend_filter(scores, design, alpha)
        # normalize on the LEVEL scale: alpha penalizes hinge coefficients in
        # level units, rescaling the target here would silently gut the penalty
        sd = factors.std(axis=0)
        sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
        factors = factors / sd[None, :]
        coefs = coefs / sd[None, :]
        if gls:
            loadings = solve(factors, yc)
            resid = yc - factors @ loadings
            hf = np.diff(resid, axis=0).std(axis=0) / np.sqrt(2.0)
            weights = 1.0 / np.maximum(hf, 1e-6)
            weights = weights / max(weights.mean(), 1e-12)
    # unit-std increments, matching the torch model's factor_paths normalization
    sd = np.diff(factors, axis=0).std(axis=0)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    factors = factors / sd[None, :]
    coefs = coefs / sd[None, :]
    loadings = solve(factors, yc)
    return {
        'factors': factors,
        'loadings': loadings.T,
        'coefs': coefs,
        'design': design,
        'weights': weights,
    }


def _kaiser_normalize(lam: np.ndarray):
    """Row-normalize loadings to unit length for rotation; return (L, norms).

    Kaiser weighting stops high-communality series (large loading vectors)
    from dominating the varimax criterion, so a rotation is chosen for the
    structure of the panel rather than for its loudest few members.
    """
    norms = np.sqrt(np.sum(lam ** 2, axis=1))
    norms = np.where(np.isfinite(norms) & (norms > 1e-12), norms, 1.0)
    return lam / norms[:, None], norms


def _promax_rotation(lam_varimax: np.ndarray, power: int = 4) -> np.ndarray:
    """Oblique promax rotation matrix, applied *after* varimax.

    Chases simple structure harder than varimax can by allowing correlated
    factors. That correlation is the price: increments of the rotated factors
    are no longer decorrelated. Offered as the fallback when the orthogonal
    rotation leaves too much cross-loading, not as a default.
    """
    target = np.abs(lam_varimax) ** (power - 1) * lam_varimax
    Q, *_ = np.linalg.lstsq(lam_varimax, target, rcond=None)
    inv = np.linalg.inv(Q.T @ Q)
    scale = np.sqrt(np.abs(np.diag(inv)))
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return Q * scale[None, :]


def identify_factors(values, n_factors, cfg, seed=42, device='cpu'):
    """Identification dispatch: alternating estimator, optional sparse refit, rotation.

    At the default config this is exactly :func:`estimate_factors_alternating`
    followed by the optional C1 rotation, so the seam is a no-op until a knob
    moves. The sparse backends refine *from* the alternating fit and fall back
    *to* it, which is why they are safe to expose as a plain config value.

    Args:
        values: (T, N) normalized panel.
        n_factors: fitted rank K.
        cfg: a merged factor config (see :data:`DEFAULT_FACTOR_CONFIG`).
        seed: passed to the gradient tier.
        device: passed to the gradient tier.

    Returns:
        The identification contract dict.
    """
    ident = estimate_factors_alternating(
        values,
        n_factors=n_factors,
        knot_spacing=cfg['knot_spacing'],
        alpha=cfg['alpha'],
        iters=cfg['alt_iters'],
        init_window=cfg['init_window'],
        gls=cfg['gls'],
        loading_l1=cfg['loading_l1'],
        loading_l1_adaptive=cfg['loading_l1_adaptive'],
        loading_relax=cfg['loading_relax'],
    )
    method = str(cfg.get('identification') or 'alternating').lower()
    if method != 'alternating':
        try:
            from autots.evaluator.tva import sparse_factor

            refined = sparse_factor.identify(
                values,
                n_factors,
                ident,
                method=method,
                config=cfg.get('sparse_config'),
                alpha=cfg['alpha'],
                seed=seed,
                device=device,
            )
            if refined is not None:
                ident = refined
        except Exception as exc:  # pragma: no cover - never fail a fit
            warnings.warn(
                f"TVA sparse identification unavailable; keeping the "
                f"alternating basis. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    if cfg.get('rotate'):
        ident = rotate_identification(
            ident,
            method=cfg['rotate'],
            kaiser=bool(cfg.get('rotate_kaiser', True)),
        )
    return ident


def rotate_identification(ident: dict, method: str = 'varimax',
                          kaiser: bool = True) -> dict:
    """Re-express an identification result in a simple-structure basis.

    Nothing upstream breaks the factor model's rotational indeterminacy: the
    SVD initialization picks an arbitrary orthogonal basis and every loading
    solve is rotation-invariant lstsq, so the fit recovers the factor *span*
    but lands on a rotation of the true basis within it. Since generator (and,
    by assumption, real) loadings are simple-structure -- each series
    predominantly loading one factor -- rotating toward simple structure is
    the identifying restriction that was missing.

    The reconstruction ``factors @ loadings.T`` is preserved exactly (to
    floating point), so this is a re-parameterization, not a refit: canonical
    correlation, subspace recovery and forecast accuracy are all invariant to
    it. Only basis-dependent quantities -- which series loads which factor,
    and with what sign -- change, which is precisely what the coherence graph
    reads.

    Three transforms are composed:

    1. the rotation itself (``R`` from :func:`discovery._varimax`, or its
       promax extension), applied as ``L @ R`` with the counter-transform
       ``F @ inv(R).T`` so the product is unchanged;
    2. re-normalization to unit-std factor increments, matching the
       convention :func:`estimate_factors_alternating` exits on (a rotation
       mixes columns of differing scale, so this is not a no-op);
    3. a mass-vote sign orientation per factor -- the same rule
       ``coherence.resolve_signs`` uses -- baked into the parameters, so the
       model itself is oriented rather than re-oriented at graph-build time.

    Args:
        ident: the dict returned by :func:`estimate_factors_alternating`.
        method: ``'varimax'`` | ``'quartimax'`` | ``'promax'``.
        kaiser: row-normalize loadings before computing the rotation.

    Returns:
        A new dict with rotated ``factors``, ``loadings`` and ``coefs``
        (``design``/``weights`` passed through). Returns ``ident`` unchanged
        on any numerical failure or unknown method -- never raises, since a
        rotation failure must degrade to today's behavior, not kill a fit.
    """
    if not ident or not method:
        return ident
    method = str(method).lower()
    if method not in ('varimax', 'quartimax', 'promax'):
        warnings.warn(f"unknown rotate method {method!r}; leaving basis as-is")
        return ident
    try:
        from autots.evaluator.tva.discovery import _varimax

        factors = np.asarray(ident['factors'], dtype=float)
        loadings = np.asarray(ident['loadings'], dtype=float)  # (N, K)
        coefs = np.asarray(ident['coefs'], dtype=float)        # (P, K)
        K = loadings.shape[1]
        if K < 2 or factors.size == 0 or loadings.size == 0:
            R = np.eye(max(K, 0))
        else:
            basis = loadings
            if kaiser:
                basis, _ = _kaiser_normalize(loadings)
            gamma = 0.0 if method == 'quartimax' else 1.0
            R = _varimax(np.nan_to_num(basis), gamma=gamma)
            if method == 'promax':
                R = R @ _promax_rotation(np.nan_to_num(basis) @ R)

        loadings_r = loadings @ R
        # F @ inv(R).T keeps F @ L.T exact for oblique R too; for orthogonal R
        # this reduces to F @ R.
        counter = np.linalg.inv(R).T if R.size else R
        factors_r = factors @ counter
        coefs_r = coefs @ counter

        # unit-std increments (the convention the torch model is initialized
        # under); guard degenerate columns rather than dividing by ~0
        sd = np.std(np.diff(factors_r, axis=0), axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        factors_r = factors_r / sd[None, :]
        coefs_r = coefs_r / sd[None, :]
        loadings_r = loadings_r * sd[None, :]

        # mass-vote orientation, identical rule to coherence.resolve_signs
        mass = (loadings_r * np.abs(loadings_r)).sum(axis=0)
        signs = np.sign(mass)
        signs = np.where(signs == 0, 1.0, signs)
        loadings_r = loadings_r * signs[None, :]
        factors_r = factors_r * signs[None, :]
        coefs_r = coefs_r * signs[None, :]

        if not (
            np.all(np.isfinite(factors_r))
            and np.all(np.isfinite(loadings_r))
            and np.all(np.isfinite(coefs_r))
        ):
            return ident
    except (np.linalg.LinAlgError, ValueError, KeyError):  # pragma: no cover
        return ident

    out = dict(ident)
    out['factors'] = factors_r
    out['loadings'] = loadings_r
    out['coefs'] = coefs_r
    out['rotation'] = R
    return out


def split_half_stability(
    values: np.ndarray,
    n_factors: int,
    n_reps: int = 3,
    seed: int = 42,
    **kwargs,
) -> float:
    """Truth-free check that the factors are shared structure, not noise.

    Real shared factors are recoverable from any subset of the series, so
    fitting two disjoint halves of the panel should yield matching paths.
    Factors that are merely absorbing idiosyncratic trends do not replicate.

    Returns:
        Mean ``match_factors`` score between the two halves, in [0, 1].
    """
    from autots.evaluator.tva.discovery import match_factors

    arr = np.asarray(values, dtype=float)
    n_series = arr.shape[1]
    if n_series < 4:
        return float('nan')
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(max(int(n_reps), 1)):
        perm = rng.permutation(n_series)
        left, right = perm[: n_series // 2], perm[n_series // 2:]
        try:
            fa = estimate_factors_alternating(arr[:, left], n_factors, **kwargs)
            fb = estimate_factors_alternating(arr[:, right], n_factors, **kwargs)
        except Exception:  # pragma: no cover - degenerate panels
            continue
        scores.append(match_factors(fa['factors'], fb['factors'])['mean_abs_corr'])
    return float(np.mean(scores)) if scores else float('nan')


def split_half_factor_stability(
    values: np.ndarray,
    n_factors: int,
    reference=None,
    n_reps: int = 3,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """Per-factor version of :func:`split_half_stability`. Returns (K,).

    The panel-level score answers "is there shared structure here at all";
    this answers "which of these K columns is shared structure" — the question
    the coherence graph needs, since a single unreplicable column is enough to
    partition the panel wrongly while the panel-level score still looks fine.

    Each replicate fits both disjoint halves, matches each half's columns to
    the full-panel reference with ``match_factors``, and scores a factor by the
    **weaker** of its two matches: a column that only one half can find is not
    shared structure. Scores are averaged over replicates.

    Args:
        values: (T, N) panel, same input the full fit received.
        n_factors: K, the reference rank.
        reference: optional (T, K) full-panel factor paths. Fitted here when
            not supplied.
        n_reps: split replicates.
        seed: RNG seed for the splits.
        **kwargs: forwarded to :func:`estimate_factors_alternating`.

    Returns:
        (K,) array in [0, 1]; NaN entries where nothing could be scored.
        Never raises — a degenerate panel returns all-NaN.
    """
    from autots.evaluator.tva.discovery import match_factors

    arr = np.asarray(values, dtype=float)
    K = int(max(1, n_factors))
    out = np.full(K, np.nan)
    n_series = arr.shape[1]
    if n_series < 4:
        return out
    try:
        if reference is None:
            reference = estimate_factors_alternating(arr, K, **kwargs)['factors']
        reference = np.asarray(reference, dtype=float)
    except Exception:  # pragma: no cover - degenerate panels
        return out

    rng = np.random.default_rng(seed)
    per_rep = []
    for _ in range(max(int(n_reps), 1)):
        perm = rng.permutation(n_series)
        halves = (perm[: n_series // 2], perm[n_series // 2:])
        scores = []
        for cols in halves:
            try:
                fit = estimate_factors_alternating(arr[:, cols], K, **kwargs)
            except Exception:  # pragma: no cover
                scores = []
                break
            matched = match_factors(reference, fit['factors'])
            corr = np.zeros(K)
            for t_idx, value in (matched.get('correlations') or {}).items():
                if 0 <= int(t_idx) < K and np.isfinite(value):
                    corr[int(t_idx)] = abs(float(value))
            scores.append(corr)
        if len(scores) == 2:
            # min over halves: a factor only one half can find is not shared
            per_rep.append(np.minimum(scores[0], scores[1]))
    if per_rep:
        out = np.clip(np.mean(per_rep, axis=0), 0.0, 1.0)
    return out


def observed_mask(values: np.ndarray, mask=None) -> np.ndarray:
    """(T, N) boolean mask of genuinely observed cells.

    When ``mask`` is None, derived from finiteness of ``values`` — correct for
    a raw panel, degenerates to "everything observed" for a filled one (the
    back-compatible default).
    """
    arr = np.asarray(values, dtype=float)
    if mask is None:
        return np.isfinite(arr)
    m = np.asarray(getattr(mask, 'values', mask))
    if m.shape != arr.shape:
        raise ValueError(
            f"observation mask shape {m.shape} does not match panel {arr.shape}"
        )
    return m.astype(bool) & np.isfinite(arr)


def select_anchors(values: np.ndarray, mask=None, min_observed: int = 540):
    """Split a panel into long-history anchors and short-history responders.

    Anchors are chosen purely from **actual observed history** — never from
    metadata, series names or declared launch dates, which are routinely wrong
    in production panels. Shared factor paths are then identified from anchors
    only, so a late-launching series can't inject fabricated pre-launch rows
    into the common-trend estimate.

    Callers should set ``min_observed`` to roughly ``3 x forecast_horizon``: a
    series needs enough history to identify its own loading and cover several
    non-overlapping validation windows before it defines the factor basis.

    Returns:
        (anchor_idx, responder_idx) — both ascending int arrays partitioning
        ``range(N)``. If no column clears the threshold the longest-history
        columns are promoted instead, since a factor model needs anchors.
    """
    obs = observed_mask(values, mask)
    counts = obs.sum(axis=0).astype(int)
    n_series = counts.size
    keep = counts >= int(min_observed)
    if not keep.any() and n_series:
        # degenerate panel: fall back to the top half by observed history
        order = np.argsort(-counts)
        keep = np.zeros(n_series, dtype=bool)
        keep[order[: max(n_series // 2, 1)]] = True
    anchor_idx = np.where(keep)[0]
    responder_idx = np.where(~keep)[0]
    return anchor_idx, responder_idx


def _masked_line_fit(target: np.ndarray, t_norm: np.ndarray, obs: np.ndarray):
    """(level, slope) of ``[1, t]`` fit to the observed rows of one series."""
    rows = np.where(obs)[0]
    if rows.size < 2:
        level = float(np.nanmean(target[rows])) if rows.size else 0.0
        return (0.0 if not np.isfinite(level) else level), 0.0
    design = np.column_stack([np.ones(rows.size), t_norm[rows]])
    coef, *_ = np.linalg.lstsq(design, target[rows], rcond=None)
    return float(coef[0]), float(coef[1])


def _fit_series_on_frozen_factors(
    paths: np.ndarray,
    target: np.ndarray,
    obs: np.ndarray,
    t_norm: np.ndarray,
    max_lag: int,
    ridge: float = 1e-2,
    cap: float = 2.0,
):
    """Masked least squares of one series against frozen factor paths.

    Fits ``y_t = a + b t + sum_k w_k F_k(t - d)`` over rows where ``y`` was
    genuinely observed; lag ``d`` is chosen by exhaustive search over
    ``0..max_lag`` (cheap: one lstsq per candidate, no gradient through rows
    that don't exist).

    Two guards, neither optional: a **ridge** on the factor columns only (over
    a short window a factor path is close to a line, nearly collinear with
    ``[1, t]``, so an unregularized loading can explode under extrapolation),
    and a **variance cap** on the implied shared component so a responder
    can't be assigned more shared movement than it has movement.

    Returns:
        (loadings (K,), lag int, level float, slope float, sse float).
    """
    rows = np.where(obs)[0]
    K = paths.shape[1]
    y = target[rows]
    base = np.column_stack([np.ones(rows.size), t_norm[rows]])
    y_sd = float(np.std(y)) if rows.size > 1 else 0.0
    best = None
    for d in range(int(max_lag) + 1):
        shifted = paths[np.clip(rows - d, 0, None)]
        design = np.column_stack([base, shifted])
        gram = design.T @ design
        energy = float(np.mean(np.diag(gram)[2:])) if K else 0.0
        pen = np.zeros(design.shape[1])
        pen[2:] = float(ridge) * max(energy, 1e-12)
        coef = np.linalg.solve(gram + np.diag(pen), design.T @ y)
        w = np.asarray(coef[2:], dtype=float)
        if cap and y_sd > 0 and K:
            full = paths[np.clip(np.arange(paths.shape[0]) - d, 0, None)] @ w
            shared_sd = float(np.std(full))
            if shared_sd > float(cap) * y_sd:
                w = w * (float(cap) * y_sd / shared_sd)
                # refit line against what the shrunken loading leaves over
                resid_y = y - (paths[np.clip(rows - d, 0, None)] @ w)
                line, *_ = np.linalg.lstsq(base, resid_y, rcond=None)
                coef = np.concatenate([line, w])
        resid = y - np.column_stack(
            [base, paths[np.clip(rows - d, 0, None)]]
        ) @ np.concatenate([coef[:2], w])
        sse = float(resid @ resid)
        if best is None or sse < best[-1]:
            best = (w, d, float(coef[0]), float(coef[1]), sse)
    if best is None:  # pragma: no cover - max_lag is never negative
        return np.zeros(K), 0, 0.0, 0.0, float('inf')
    return best


if HAS_TORCH:

    class LatentFactorTrend(nn.Module):
        """Dynamic factor model over the whole training range, full batch.

        Args:
            n_series: N.
            n_time: T (training length).
            n_factors: K factor paths (pruned at exposure time).
            knot_spacing: candidate-knot spacing of the trend-filter basis.
            max_lag: maximum per-series response lag (soft, learned).
            slope_window: trailing window for the extrapolation slope.
        """

        def __init__(
            self,
            n_series: int,
            n_time: int,
            n_factors: int = 6,
            knot_spacing: int = 7,
            max_lag: int = 14,
            slope_window: int = 90,
        ):
            super().__init__()
            self.N = int(n_series)
            self.T = int(n_time)
            self.K = max(int(n_factors), 1)
            self.max_lag = int(np.clip(max_lag, 0, max(self.T // 4, 0)))
            self.D = self.max_lag + 1
            self.slope_window = max(int(min(slope_window, self.T)), 2)

            design = hinge_design(self.T, knot_spacing)
            self.register_buffer('design', torch.tensor(design))
            self.P = design.shape[1]
            # knot positions of the same basis, for regime-aware extrapolation
            self.knot_times = hinge_knot_times(self.T, knot_spacing)

            self.coef = nn.Parameter(torch.zeros(self.P, self.K))
            self.loadings = nn.Parameter(torch.zeros(self.N, self.K))
            lag_logits = torch.zeros(self.N, self.D)
            lag_logits[:, 0] = 2.0
            self.lag_logits = nn.Parameter(lag_logits)
            self.idio_level = nn.Parameter(torch.zeros(self.N))
            self.idio_slope = nn.Parameter(torch.zeros(self.N))
            # phi = 0.5 + 0.49 * sigmoid(x); x = 1.4922 -> phi ~= 0.90
            self.phi_logit = nn.Parameter(torch.full((self.K,), 1.4922))
            self.idio_phi_logit = nn.Parameter(torch.tensor(1.4922))

            self.register_buffer(
                'time_index',
                torch.arange(self.T, dtype=torch.float32) / float(self.T),
            )

        # ---- core pieces ---------------------------------------------------

        def factor_paths(self) -> 'torch.Tensor':
            """(T, K) identified factor paths: unit-std increments, F(0) = 0.

            The hinge basis is zero at t = 0, so the level anchor is structural;
            only the increment scale needs normalizing (it lives in loadings).
            """
            raw = self.design @ self.coef
            d = raw[1:] - raw[:-1]
            sd = d.std(dim=0, unbiased=False).clamp(min=1e-4)
            return raw / sd

        def lag_weights(self) -> 'torch.Tensor':
            """(N, D) soft response-lag distribution per series."""
            return torch.softmax(self.lag_logits, dim=-1)

        def series_factor_paths(self, paths: 'torch.Tensor') -> 'torch.Tensor':
            """(N, T, K) per-series lag-weighted factor paths."""
            idx = torch.arange(paths.shape[0], device=paths.device)
            shifts = torch.arange(self.D, device=paths.device)
            stack = paths[(idx[None, :] - shifts[:, None]).clamp(min=0)]
            return torch.einsum('nd,dtk->ntk', self.lag_weights(), stack)

        def idio_component(self, t: 'torch.Tensor') -> 'torch.Tensor':
            """(N, len(t)) idiosyncratic line at normalized times ``t``."""
            return self.idio_level[:, None] + self.idio_slope[:, None] * t[None, :]

        def forward(self) -> 'torch.Tensor':
            """(N, T) in-sample reconstruction in normalized level space."""
            paths = self.factor_paths()
            shared = torch.einsum(
                'ntk,nk->nt', self.series_factor_paths(paths), self.loadings
            )
            return self.idio_component(self.time_index) + shared

        # ---- extrapolation --------------------------------------------------

        def phi(self) -> 'torch.Tensor':
            return 0.5 + 0.49 * torch.sigmoid(self.phi_logit)

        def idio_phi(self) -> 'torch.Tensor':
            return 0.5 + 0.49 * torch.sigmoid(self.idio_phi_logit)

        def rolling_forecast(
            self, origins: 'torch.Tensor', horizon: int
        ) -> 'torch.Tensor':
            """(O, H, N) normalized-space forecasts from many origins.

            Times at or before an origin are read from the in-sample path, which
            is what lets a follower with lag ``d`` consume the leader's already
            *observed* movement for its first ``d`` horizon steps; later times
            are damped local-linear continuations of each factor.
            """
            deltas = self.continuation_deltas(origins, horizon)
            return self.rolling_forecast_from(origins, deltas, horizon)

        def continuation_deltas(
            self, origins: 'torch.Tensor', horizon: int
        ) -> 'torch.Tensor':
            """(O, H, K) incumbent factor deltas: damped local-linear.

            Split out of ``rolling_forecast`` so the extrapolation rule is a
            replaceable input rather than a hard-wired step (plan item 1a).
            Returned as deltas relative to ``paths[origin]`` so every candidate
            rule is anchored identically.
            """
            paths = self.factor_paths()
            device = paths.device
            H = max(int(horizon), 1)

            min_origin = int(origins.min().item())
            L = max(min(self.slope_window, min_origin + 1), 2)
            offs = torch.arange(L, device=device)
            widx = (origins[:, None] - (L - 1) + offs[None, :]).clamp(min=0)
            win = paths[widx]  # (O, L, K)
            t_c = offs.to(paths.dtype) - offs.to(paths.dtype).mean()
            slope = (win * t_c[None, :, None]).sum(dim=1) / (t_c**2).sum().clamp(
                min=1e-8
            )  # (O, K)

            steps = torch.arange(1, H + 1, device=device, dtype=paths.dtype)
            damp = torch.cumsum(self.phi()[None, :] ** steps[:, None], dim=0)
            return damp[None, :, :] * slope[:, None, :]

        def rolling_forecast_from(
            self, origins: 'torch.Tensor', deltas: 'torch.Tensor', horizon: int
        ) -> 'torch.Tensor':
            """(O, H, N) forecasts recombined from supplied factor deltas.

            Everything downstream of the extrapolation rule — lag weighting,
            loadings, the idiosyncratic line — is identical regardless of how
            the future factor path was produced, so candidate continuations are
            compared through exactly the same recombination the model uses.
            """
            paths = self.factor_paths()
            device = paths.device
            H = max(int(horizon), 1)
            if not torch.is_tensor(deltas):
                deltas = torch.as_tensor(
                    np.asarray(deltas, dtype=np.float32), device=device
                )
            deltas = deltas.to(device=device, dtype=paths.dtype)
            steps = torch.arange(1, H + 1, device=device, dtype=paths.dtype)
            future = paths[origins][:, None, :] + deltas

            past_idx = (
                origins[:, None]
                - self.D
                + torch.arange(self.D + 1, device=device)[None, :]
            ).clamp(min=0)
            ext = torch.cat([paths[past_idx], future], dim=1)  # (O, D+H+1, K)

            h_idx = torch.arange(1, H + 1, device=device)
            d_idx = torch.arange(self.D, device=device)
            gather = (h_idx[:, None] - d_idx[None, :] + self.D).clamp(min=0)
            shared = torch.einsum(
                'ohdk,nd,nk->ohn', ext[:, gather], self.lag_weights(), self.loadings
            )

            idio_damp = torch.cumsum(self.idio_phi() ** steps, dim=0)
            t_o = origins.to(paths.dtype) / float(self.T)
            idio_at = self.idio_level[None, :] + self.idio_slope[None, :] * t_o[:, None]
            per_step = self.idio_slope / float(self.T)
            idio = (
                idio_at[:, None, :]
                + per_step[None, None, :] * idio_damp[None, :, None]
            )
            return idio + shared

        def forecast(
            self, horizon: int, origin: int = None, deltas=None
        ) -> 'torch.Tensor':
            """(N, H) normalized-space forecast from ``origin`` (default last).

            ``deltas`` optionally supplies (1, H, K) factor deltas from a
            validation-selected continuation instead of the incumbent damped
            local-linear rule.
            """
            if origin is None:
                origin = self.T - 1
            origins = torch.tensor(
                [int(origin)], device=self.coef.device, dtype=torch.long
            )
            if deltas is None:
                return self.rolling_forecast(origins, horizon)[0].T
            return self.rolling_forecast_from(origins, deltas, horizon)[0].T

        # ---- introspection ---------------------------------------------------

        def variance_share(self) -> np.ndarray:
            """(K,) share of shared trend movement attributable to each factor.

            Factor increments are unit-std by construction, so the loading
            energy per column is the movement each factor contributes.
            """
            with torch.no_grad():
                w = (self.loadings.detach() ** 2).sum(dim=0).cpu().numpy()
            total = float(w.sum())
            return w / total if total > 0 else np.zeros_like(w)

        def live_factors(self, prune_share: float = 0.02) -> np.ndarray:
            """Indices of factors above the pruning threshold, strongest first.

            Threshold is deliberately low: pruning a genuine but minor factor
            costs more recovery than a dead column does. Guarding against
            spurious factors is ``select_n_factors``'s job, not this.
            """
            share = self.variance_share()
            keep = np.where(share >= prune_share)[0]
            if keep.size == 0:
                keep = np.array([int(np.argmax(share))])
            return keep[np.argsort(-share[keep])]

        def _signs(self, lam: np.ndarray) -> np.ndarray:
            signs = np.sign(lam[np.abs(lam).argmax(axis=0), np.arange(lam.shape[1])])
            signs[signs == 0] = 1.0
            return signs

        def fitted_factors(self, prune_share: float = 0.02) -> np.ndarray:
            """(T, k) pruned, variance-ordered, sign-fixed factor paths."""
            with torch.no_grad():
                paths = self.factor_paths().detach().cpu().numpy()
                lam = self.loadings.detach().cpu().numpy()
            keep = self.live_factors(prune_share)
            return paths[:, keep] * self._signs(lam)[keep][None, :]

        def fitted_loadings(self, prune_share: float = 0.02) -> np.ndarray:
            """(N, k) loadings matching ``fitted_factors`` order and sign."""
            with torch.no_grad():
                lam = self.loadings.detach().cpu().numpy()
            keep = self.live_factors(prune_share)
            return lam[:, keep] * self._signs(lam)[keep][None, :]

        def fitted_lags(self) -> np.ndarray:
            """(N,) expected response lag per series, rounded to integers."""
            w = self.lag_distribution()
            return np.rint((w * np.arange(w.shape[1])[None, :]).sum(axis=1)).astype(int)

        def lag_distribution(self) -> np.ndarray:
            with torch.no_grad():
                return self.lag_weights().detach().cpu().numpy()

        def diagnostics(self) -> dict:
            """Degeneracy checks: noise-chasing, collapse, factor share."""
            with torch.no_grad():
                paths = self.factor_paths().detach().cpu().numpy()
                lam = self.loadings.detach().cpu().numpy()
                idio_slope = self.idio_slope.detach().cpu().numpy()
            d = np.diff(paths, axis=0)
            sd = d.std(axis=0)
            sd[sd <= 0] = 1.0
            zd = (d - d.mean(axis=0)) / sd
            autocorr = float(np.mean((zd[1:] * zd[:-1]).mean(axis=0)))
            if zd.shape[1] > 1:
                c = np.corrcoef(zd.T)
                np.fill_diagonal(c, 0.0)
                max_corr = float(np.nanmax(np.abs(c)))
            else:
                max_corr = 0.0
            shared = paths @ lam.T  # (T, N)
            shared = shared - shared.mean(axis=0, keepdims=True)
            t_norm = np.arange(self.T, dtype=float) / float(self.T)
            idio = idio_slope[None, :] * (t_norm - t_norm.mean())[:, None]
            shared_move = float(np.sum(shared**2))
            idio_move = float(np.sum(idio**2))
            denom = shared_move + idio_move
            return {
                'delta_autocorr': autocorr,
                'max_pairwise_delta_corr': max_corr,
                # NOTE: reads ~1.00 on every panel
                'factor_variance_share': (
                    float(shared_move / denom) if denom > 0 else 0.0
                ),
                'n_live_factors': int(len(self.live_factors())),
                'phi': [float(v) for v in self.phi().detach().cpu().numpy()],
                'idio_phi': float(self.idio_phi().detach().cpu().item()),
            }

    # ---- training -------------------------------------------------------------

    def _loading_prior_penalty(model, prior_adj) -> 'torch.Tensor':
        """Graph-smoothness penalty on the normalized loading rows.

        ``sum_ij |A_ij| * ||u_i - sign(A_ij) * u_j||^2 / (2 * sum_ij |A_ij|)``
        with ``u`` the row-normalized loadings. Linked series are pulled
        toward the same loading profile, negatively linked ones apart.
        Normalizing by ``sum |A|`` keeps the weight scale-free in the number
        of prior edges, so the same ``w_prior_loadings`` means the same thing
        on a 2-edge and a 200-edge prior.

        Backprops before the ``w_prox_loadings`` proximal step, so the
        sparsity behaviour is unchanged.
        """
        lam = model.loadings
        n = lam.shape[0]
        adjacency = torch.as_tensor(
            np.asarray(prior_adj, dtype=np.float32),
            dtype=lam.dtype,
            device=lam.device,
        )
        if adjacency.ndim != 2 or adjacency.shape != (n, n):
            return lam.sum() * 0.0
        adjacency = 0.5 * (adjacency + adjacency.T)
        adjacency = adjacency - torch.diag(torch.diag(adjacency))
        # guard zero rows: a gated/unloaded series has no profile to align,
        # and dividing by its ~zero norm would explode the gradient. Drop it
        # from both the penalty and its normalizer.
        norms = lam.norm(dim=1, keepdim=True)
        good = (norms > 1e-6).to(lam.dtype)
        u = lam / norms.clamp(min=1e-6) * good
        magnitude = adjacency.abs() * (good @ good.T)
        total = magnitude.sum()
        if float(total.detach().cpu().item()) <= 0:
            return lam.sum() * 0.0
        sign = torch.sign(adjacency)
        # ||u_i - s_ij u_j||^2 = 2 - 2 s_ij <u_i, u_j>  (rows are unit norm)
        gram = u @ u.T
        sq = 2.0 - 2.0 * sign * gram
        return (magnitude * sq).sum() / (2.0 * total)

    def _aux_penalties(model, cfg: dict):
        """Lag and loading penalties (the factor prior lives in the prox step)."""
        w = model.lag_weights()
        lag_idx = torch.arange(model.D, device=w.device, dtype=w.dtype)
        # asymmetric cost pins the lag/shift confound to the lag-0 majority
        lag_mean = (w * lag_idx[None, :]).sum(dim=-1).mean()
        entropy = -(w * torch.log(w.clamp(min=1e-8))).sum(dim=-1).mean()
        penalty = (
            cfg['w_lag_mean'] * lag_mean
            + cfg['w_lag_entropy'] * entropy
            + cfg['w_l1_loadings'] * model.loadings.abs().mean()
        )
        w_prior = float(cfg.get('w_prior_loadings') or 0.0)
        prior_adj = cfg.get('prior_adjacency')
        if w_prior > 0 and prior_adj is not None:
            penalty = penalty + w_prior * _loading_prior_penalty(model, prior_adj)
        if cfg.get('w_decorr'):
            paths = model.factor_paths()
            d = paths[1:] - paths[:-1]
            zd = d - d.mean(dim=0, keepdim=True)
            zd = zd / zd.std(dim=0, unbiased=False).clamp(min=1e-6)
            corr = (zd.T @ zd) / float(zd.shape[0])
            off = corr - torch.diag(torch.diag(corr))
            penalty = penalty + cfg['w_decorr'] * (off**2).sum() / max(
                model.K * (model.K - 1), 1
            )
        return penalty

    def _zero_and_refit_idio(model, target: np.ndarray, drop: np.ndarray) -> list:
        """Zero the loadings of ``drop`` and refit their idiosyncratic line.

        Shared by every gate so they cannot drift apart in how they retire a
        series: a gated series still needs some forecast, and "no shared
        structure" is a plain linear idio line fit to its raw target.

        Returns:
            The retired indices as a plain list.
        """
        drop = np.asarray(drop, dtype=int)
        if drop.size == 0:
            return []
        with torch.no_grad():
            t = model.time_index.detach().cpu().numpy()
            design = np.column_stack([np.ones_like(t), t])
            coef, *_ = np.linalg.lstsq(design, target[:, drop], rcond=None)
            model.loadings[drop] = 0.0
            model.idio_level[drop] = torch.tensor(
                coef[0], dtype=model.idio_level.dtype,
                device=model.idio_level.device,
            )
            model.idio_slope[drop] = torch.tensor(
                coef[1], dtype=model.idio_slope.dtype,
                device=model.idio_slope.device,
            )
        return [int(i) for i in drop]

    def _gate_trendless_series(model, y: 'torch.Tensor', min_ratio: float) -> list:
        """Zero the loadings of series that have no low-frequency structure."""
        if not min_ratio:
            return []
        with torch.no_grad():
            target = y.detach().cpu().numpy().T  # (T, N)
            window = max(min(model.T // 20, 28), 3)
            smoothed = _rolling_mean(target, window)
            ratio = smoothed.std(axis=0) / np.maximum(
                (target - smoothed).std(axis=0), 1e-9
            )
            drop = np.where(ratio < float(min_ratio))[0]
        return _zero_and_refit_idio(model, target, drop)

    def _frozen_tail_series(target: np.ndarray, min_len: int = 14) -> np.ndarray:
        """Indices whose last ``min_len`` observations are numerically constant.

        A dead-flat segment has ~zero high-frequency residual, and the GLS
        weighting in ``estimate_factors_alternating`` is ``1 / hf`` — so a
        frozen series would otherwise get outsized weight while carrying no
        information.
        """
        min_len = max(int(min_len), 2)
        if target.shape[0] < min_len:
            return np.array([], dtype=int)
        tail = target[-min_len:]
        spread = np.nanmax(tail, axis=0) - np.nanmin(tail, axis=0)
        overall = np.nanstd(target, axis=0)
        overall = np.where(np.isfinite(overall) & (overall > 0), overall, 1.0)
        # relative to the series' own variation, so a low-variance but live
        # series doesn't false-positive
        return np.where(np.isfinite(spread) & (spread <= 1e-8 * overall))[0]

    def _gate_frozen_series(model, y: 'torch.Tensor', min_len: int) -> list:
        """Retire frozen-tail series and forecast them as a constant.

        Zeroing the loadings is not enough: the idiosyncratic line refit would
        still put a slope through the whole history. A frozen series' only
        defensible forecast is its held value, so the idio line is pinned flat
        at that value instead.
        """
        with torch.no_grad():
            target = y.detach().cpu().numpy().T  # (T, N)
            drop = _frozen_tail_series(target, min_len)
            if drop.size == 0:
                return []
            held = target[-1, drop]
            model.loadings[drop] = 0.0
            model.idio_level[drop] = torch.tensor(
                held, dtype=model.idio_level.dtype, device=model.idio_level.device
            )
            model.idio_slope[drop] = torch.zeros(
                drop.size, dtype=model.idio_slope.dtype,
                device=model.idio_slope.device,
            )
        return [int(i) for i in drop]

    def _damped_linear_baseline(
        target: np.ndarray, origins: np.ndarray, horizon: int, window: int = 90,
        phi: float = 0.9,
    ) -> np.ndarray:
        """(O, H, N) damped local-linear continuation of the raw target.

        The comparator for the per-series predictive gate: the simplest thing
        that could possibly work on a single series, with no shared factors at
        all. A series the factor model cannot beat this on has nothing to gain
        from the factor layer.
        """
        origins = np.atleast_1d(np.asarray(origins, dtype=int))
        H = max(int(horizon), 1)
        steps = np.arange(1, H + 1, dtype=float)
        damp = np.cumsum(np.power(float(phi), steps))
        out = np.empty((origins.size, H, target.shape[1]), dtype=float)
        for i, origin in enumerate(origins):
            lo = max(int(origin) - int(window) + 1, 0)
            seg = target[lo : int(origin) + 1]
            t = np.arange(seg.shape[0], dtype=float)
            t = t - t.mean()
            denom = float((t**2).sum())
            slope = (
                (seg * t[:, None]).sum(axis=0) / denom if denom > 1e-12
                else np.zeros(target.shape[1])
            )
            out[i] = target[int(origin)][None, :] + damp[:, None] * slope[None, :]
        return out

    def _gate_underperforming_series(
        model, y: 'torch.Tensor', origins: np.ndarray, horizon: int,
        margin: float,
    ) -> list:
        """Retire series the factor model forecasts worse than a naive line.

        Reuses the rolling-origin forecasts stage B already computes for sigma.
        Retired when ``MAE(factor) > margin * MAE(baseline)``; the margin
        means a tie keeps the factor layer, since coherence is built on it.
        """
        if not margin or origins.size == 0:
            return []
        with torch.no_grad():
            target = y.detach().cpu().numpy().T  # (T, N)
            H = max(int(horizon), 1)
            valid = origins[origins + H < target.shape[0]]
            if valid.size == 0:
                return []
            idx = torch.tensor(valid, device=y.device, dtype=torch.long)
            pred = model.rolling_forecast(idx, H).cpu().numpy()  # (O, H, N)
            h_off = np.arange(1, H + 1)
            actual = target[valid[:, None] + h_off[None, :]]  # (O, H, N)
            base = _damped_linear_baseline(
                target, valid, H, window=model.slope_window
            )
            mae_model = np.nanmean(np.abs(pred - actual), axis=(0, 1))
            mae_base = np.nanmean(np.abs(base - actual), axis=(0, 1))
            drop = np.where(mae_model > float(margin) * mae_base)[0]
        return _zero_and_refit_idio(model, target, drop)

    def _inner_origins(n_time: int, horizon: int, n_origins: int, floor: int) -> np.ndarray:
        """Non-overlapping validation origins ending just before ``n_time``.

        Oldest first. Origins needing less history than ``floor`` are dropped
        rather than clamped — clamping would make two "independent" folds
        reuse the same window.
        """
        H = max(int(horizon), 1)
        out = []
        for i in range(max(int(n_origins), 0)):
            origin = n_time - 1 - H * (i + 1)
            if origin < max(int(floor), 2):
                break
            out.append(origin)
        return np.array(sorted(out), dtype=int)

    def _select_continuation(
        model, y: 'torch.Tensor', horizon: int, cfg: dict
    ) -> dict:
        """Plan item 1a: pick a continuation rule per factor by validation.

        Scored on reconstructed-series MASE (scaled by each series' in-sample
        seasonal-naive MAE so a large-scale series cannot dominate the choice),
        pooled over held-out inner origins.
        """
        from autots.evaluator.tva.continuation import (
            build_specs,
            select_continuations,
        )

        H = max(int(horizon), 1)
        target = y.detach().cpu().numpy().T  # (T, N)
        origins = _inner_origins(
            model.T, H, int(cfg['continuation_origins']), model.slope_window
        )
        if origins.size == 0:
            return {'choice': {}, 'origins': [], 'reason': 'insufficient history'}

        h_off = np.arange(1, H + 1)
        actual = target[origins[:, None] + h_off[None, :]]  # (O, H, N)
        season = 7
        if target.shape[0] > season:
            scale = np.nanmean(
                np.abs(target[season:] - target[:-season]), axis=0
            )
        else:
            scale = np.nanmean(np.abs(np.diff(target, axis=0)), axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.nan)

        idx = torch.tensor(origins, device=y.device, dtype=torch.long)
        with torch.no_grad():
            model_deltas = model.continuation_deltas(idx, H).cpu().numpy()

        def score_fn(deltas):
            with torch.no_grad():
                pred = model.rolling_forecast_from(idx, deltas, H).cpu().numpy()
            mae = np.nanmean(np.abs(pred - actual), axis=(0, 1))
            with np.errstate(invalid='ignore'):
                return float(np.nanmean(mae / scale))

        with torch.no_grad():
            paths = model.factor_paths().cpu().numpy()
            coef = model.coef.detach().cpu().numpy()
        result = select_continuations(
            paths,
            origins,
            H,
            score_fn,
            specs=build_specs(cfg.get('continuation_config')),
            knot_times=model.knot_times,
            coef=coef,
            model_deltas=model_deltas,
            config=cfg.get('continuation_config'),
        )
        result['origins'] = [int(o) for o in origins]
        return result

    def selected_continuation_deltas(
        model, horizon: int, continuation: dict, origin: int = None,
        config: dict = None,
    ):
        """(1, H, K) factor deltas for a validation-selected continuation.

        Returns ``None`` when no selection was made, so callers fall through to
        the model's own damped local-linear rule and today's behavior is
        reproduced exactly.
        """
        if not continuation or not continuation.get('choice'):
            return None
        from autots.evaluator.tva.continuation import apply_choice, build_specs

        H = max(int(horizon), 1)
        if origin is None:
            origin = model.T - 1
        with torch.no_grad():
            paths = model.factor_paths().cpu().numpy()
            coef = model.coef.detach().cpu().numpy()
            origins = torch.tensor(
                [int(origin)], device=model.coef.device, dtype=torch.long
            )
            model_deltas = model.continuation_deltas(origins, H).cpu().numpy()
        choice = {int(k): v for k, v in continuation['choice'].items()}
        return apply_choice(
            paths,
            np.array([int(origin)]),
            H,
            choice,
            specs=build_specs(config),
            knot_times=model.knot_times,
            coef=coef,
            model_deltas=model_deltas,
            config=config,
        )

    def fit_latent_factor_model(
        values: np.ndarray,
        n_factors: int = 6,
        knot_spacing: int = 7,
        max_lag: int = 14,
        horizon: int = 28,
        config: dict = None,
        seed: int = 42,
        device: str = 'cpu',
        verbose: int = 0,
    ):
        """Fit the latent-factor trend model on a normalized level panel.

        ``values`` must already be normalized (see ``robust_level_scale``).

        Returns:
            (model, info) — info carries the identification result, losses,
            diagnostics and per-series rolling-origin residual sigma.
        """
        cfg = dict(DEFAULT_FACTOR_CONFIG)
        if config:
            cfg.update({k: v for k, v in config.items() if v is not None})

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        arr = np.asarray(values, dtype=np.float32)
        T, N = arr.shape
        dev = torch.device(device)
        y = torch.tensor(arr.T, dtype=torch.float32, device=dev)  # (N, T)
        K = int(max(1, min(int(n_factors), N)))

        # ---- identification (torch-free, the stage that finds the factors) --
        # With lr_coef == 0 the coefficients below are frozen, so whatever
        # basis leaves this line is the model's final basis. Both the C1
        # rotation and the C9 sparse backends therefore act here -- after
        # identification, before the parameter copy -- fixing the basis the
        # loadings graph reads while leaving the reconstruction (and stage B's
        # per-factor phi, calibrated afterwards) untouched.
        ident = identify_factors(arr, K, cfg, seed=seed, device=device)

        model = LatentFactorTrend(
            n_series=N,
            n_time=T,
            n_factors=K,
            knot_spacing=cfg['knot_spacing'],
            max_lag=max_lag,
            slope_window=cfg['slope_window'],
        ).to(dev)
        with torch.no_grad():
            model.coef.copy_(
                torch.tensor(ident['coefs'], dtype=torch.float32, device=dev)
            )
            model.loadings.copy_(
                torch.tensor(ident['loadings'], dtype=torch.float32, device=dev)
            )
            # A sparse identification fits the per-series line jointly with
            # the factors, so start stage A from that solution rather than
            # from the column mean it would otherwise have to rediscover.
            level = ident.get('idio_level')
            if level is None or np.shape(level) != (N,):
                level = arr.mean(axis=0)
            model.idio_level.copy_(
                torch.tensor(np.asarray(level, dtype=float), dtype=torch.float32, device=dev)
            )
            slope = ident.get('idio_slope')
            if slope is not None and np.shape(slope) == (N,):
                model.idio_slope.copy_(
                    torch.tensor(np.asarray(slope, dtype=float), dtype=torch.float32, device=dev)
                )
        # Support the sparse identification found, if any. Stage A's
        # w_l1_loadings is a subgradient term that never actually reaches zero,
        # so without projecting back onto this mask the identified zeros are
        # gone long before fitted_loadings() -- and fitted_loadings() is what
        # _apply_coherence builds the graph from.
        support_mask = None
        if cfg.get('sparse_freeze_support') and ident.get('identification_method', '').startswith('sparse'):
            support_mask = torch.tensor(
                (np.abs(np.asarray(ident['loadings'], dtype=float)) > 0).astype('float32'),
                device=dev,
            )

        # ---- stage A: refine loadings, lags and the idiosyncratic line ------
        val_mask = torch.zeros(T, dtype=torch.bool, device=dev)
        n_val = int(np.clip(int(cfg['val_frac'] * T), 1, max(T - 2, 1)))
        val_mask[torch.tensor(np.sort(rng.choice(T, n_val, replace=False)), device=dev)] = True
        train_mask = ~val_mask

        param_groups = []
        if cfg['lr_coef'] > 0:
            param_groups.append({'params': [model.coef], 'lr': cfg['lr_coef']})
        else:
            model.coef.requires_grad_(False)
        opt = torch.optim.Adam(
            param_groups
            + [
                {
                    'params': [
                        model.loadings,
                        model.lag_logits,
                        model.idio_level,
                        model.idio_slope,
                    ],
                    'lr': cfg['lr_aux'],
                },
            ]
        )
        best_val, best_state, bad_checks = float('inf'), None, 0
        history = []
        for step in range(int(cfg['stage_a_steps'])):
            opt.zero_grad(set_to_none=True)
            recon = model()
            loss = F.huber_loss(recon[:, train_mask], y[:, train_mask], delta=1.0)
            (loss + _aux_penalties(model, cfg)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()
            if cfg['lr_coef'] > 0:
                # proximal l1: plain Adam on an l1 term never reaches zero,
                # which would blur changepoints
                with torch.no_grad():
                    thresh = cfg['w_prox'] * cfg['lr_coef']
                    model.coef[1:] = torch.sign(model.coef[1:]) * (
                        model.coef[1:].abs() - thresh
                    ).clamp(min=0)
            if cfg['w_prox_loadings'] > 0:
                # C3 companion: the same argument for the loadings. The
                # identification's sparse structure is what the coherence
                # graph reads, and w_l1_loadings alone (a subgradient term)
                # only shrinks toward zero -- it never arrives, so the zeros
                # are gone long before fitted_loadings() is called.
                with torch.no_grad():
                    thresh_l = cfg['w_prox_loadings'] * cfg['lr_aux']
                    model.loadings.copy_(
                        torch.sign(model.loadings)
                        * (model.loadings.abs() - thresh_l).clamp(min=0)
                    )
            if support_mask is not None:
                # C9: a projection, not a threshold -- it costs no
                # hyperparameter, preserves the identified graph exactly, and
                # still lets the surviving magnitudes adapt to the lag weights.
                with torch.no_grad():
                    model.loadings.mul_(support_mask)
            if (step + 1) % int(cfg['check_every']) == 0:
                with torch.no_grad():
                    v = float(
                        F.huber_loss(
                            model()[:, val_mask], y[:, val_mask], delta=1.0
                        ).item()
                    )
                history.append((step + 1, float(loss.item()), v))
                if verbose >= 2:
                    print(
                        f"  [stage A] step {step+1} train {loss.item():.5f} val {v:.5f}"
                    )
                if v < best_val - 1e-6:
                    best_val, bad_checks = v, 0
                    best_state = {
                        k: t.detach().clone() for k, t in model.state_dict().items()
                    }
                else:
                    bad_checks += 1
                    if bad_checks * int(cfg['check_every']) >= int(cfg['patience']):
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.coef.requires_grad_(True)

        # ---- stage B: extrapolation damping calibration ---------------------
        for p in model.parameters():
            p.requires_grad_(False)
        model.phi_logit.requires_grad_(True)
        model.idio_phi_logit.requires_grad_(True)

        H = max(int(horizon), 1)
        span = max(min(300, T // 3), 1)
        lo = max(T - span, model.slope_window)
        hi = T - H - 1
        pool = np.arange(lo, hi + 1) if hi >= lo else np.array([], dtype=int)
        stage_b_loss = None
        h_off = torch.arange(1, H + 1, device=dev)
        if pool.size >= 4:
            optb = torch.optim.Adam(
                [model.phi_logit, model.idio_phi_logit], lr=cfg['lr_phi']
            )
            for _ in range(int(cfg['stage_b_steps'])):
                sel = rng.choice(
                    pool, size=min(int(cfg['n_origins']), pool.size), replace=False
                )
                origins = torch.tensor(np.sort(sel), device=dev, dtype=torch.long)
                optb.zero_grad(set_to_none=True)
                pred = model.rolling_forecast(origins, H)
                target = y.T[origins[:, None] + h_off[None, :]]
                lossb = F.huber_loss(pred, target, delta=1.0)
                lossb.backward()
                optb.step()
                stage_b_loss = float(lossb.item())
            with torch.no_grad():
                origins = torch.tensor(pool, device=dev, dtype=torch.long)
                if origins.shape[0] > 128:
                    keep = np.linspace(0, origins.shape[0] - 1, 128).astype(int)
                    origins = origins[torch.tensor(keep, device=dev)]
                resid = (
                    model.rolling_forecast(origins, H)
                    - y.T[origins[:, None] + h_off[None, :]]
                ).reshape(-1, N)
                sigma = resid.std(dim=0).cpu().numpy()
                # kept, not discarded: this is the only rolling-origin
                # cross-series residual matrix the model ever forms, and it is
                # the raw material for the forecast covariance (covariance.py)
                residual_matrix = resid.cpu().numpy()
        else:
            with torch.no_grad():
                in_sample = (model() - y).T
                sigma = in_sample.std(dim=0).cpu().numpy()
                residual_matrix = in_sample.cpu().numpy()
        for p in model.parameters():
            p.requires_grad_(True)

        # ---- Phase 1 gates -------------------------------------------------
        # order matters: frozen (dead data) -> trendless -> predictive, so the
        # predictive gate never retires a series for another's contamination
        frozen = (
            _gate_frozen_series(model, y, cfg['frozen_tail_min_len'])
            if cfg.get('frozen_tail_gate')
            else []
        )
        gated = _gate_trendless_series(model, y, cfg['min_trend_to_noise'])
        underperforming = _gate_underperforming_series(
            model, y, pool if pool.size else np.array([], dtype=int), H,
            cfg.get('gate_forecast_margin'),
        )

        # ---- 1a: validation-selected factor continuation --------------------
        continuation = None
        if cfg.get('continuation_select'):
            try:
                continuation = _select_continuation(model, y, H, cfg)
            except Exception as exc:  # pragma: no cover - never fail the fit
                warnings.warn(
                    f"TVA factor continuation selection failed, keeping the "
                    f"default damped local-linear rule. {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continuation = None

        factor_stability = None
        if int(cfg.get('factor_stability_reps') or 0) > 0:
            factor_stability = split_half_factor_stability(
                arr,
                K,
                reference=ident['factors'],
                n_reps=int(cfg['factor_stability_reps']),
                seed=seed,
                knot_spacing=cfg['knot_spacing'],
                alpha=cfg['alpha'],
                iters=cfg['alt_iters'],
                init_window=cfg['init_window'],
                gls=cfg['gls'],
            )

        diag = model.diagnostics()
        info = {
            'gated_series': sorted(set(gated) | set(frozen) | set(underperforming)),
            'gated_trendless': gated,
            'gated_frozen': frozen,
            'gated_underperforming': underperforming,
            'continuation': continuation,
            'identification': ident,
            'factor_stability': factor_stability,
            'history': history,
            'stage_a_val': best_val if np.isfinite(best_val) else None,
            'stage_b_loss': stage_b_loss,
            'sigma': np.asarray(sigma, dtype=float),
            'residual_matrix': np.asarray(residual_matrix, dtype=float),
            'diagnostics': diag,
            'n_factors_fit': model.K,
            'n_factors_live': diag['n_live_factors'],
            # C9 diagnostics: the live-atom count is a rank opinion grounded in
            # who actually uses an atom, unlike n_factors_live (an energy
            # threshold) and select_n_factors (an eigenvalue ratio).
            'identification_method': ident.get('identification_method', 'alternating'),
            'atom_usage': ident.get('atom_usage'),
            'n_atoms_live': ident.get('n_atoms_live'),
            'sparse_rejected': bool(ident.get('sparse_rejected', False)),
            'init_rotate_used': ident.get('init_rotate_used'),
            'config': cfg,
        }
        if verbose:
            print(
                f"TVA factor model: K={model.K} live={diag['n_live_factors']} "
                f"val={best_val:.5f} share={diag['factor_variance_share']:.3f} "
                f"dF-autocorr={diag['delta_autocorr']:.3f}"
            )
        if diag['factor_variance_share'] < 0.2:
            warnings.warn(
                "TVA factor model: shared factors explain <20% of trend "
                "movement; the fit may be degenerate.",
                RuntimeWarning,
                stacklevel=2,
            )
        return model, info

    def _lag_logits_from_lags(lags: np.ndarray, n_lags: int) -> np.ndarray:
        """(n, D) near-one-hot logits reproducing integer response lags.

        The responder lags are chosen by exhaustive search rather than learned
        as a soft distribution, so they are written back as logits whose
        softmax is one-hot to within 1e-13 — the model's forecast path only
        ever reads ``lag_weights()``.
        """
        out = np.full((len(lags), int(n_lags)), -30.0, dtype=np.float32)
        for i, d in enumerate(lags):
            out[i, int(np.clip(d, 0, n_lags - 1))] = 0.0
        return out

    def _responder_sigma(
        model, arr: np.ndarray, obs: np.ndarray, cols: np.ndarray,
        horizon: int, device,
    ) -> np.ndarray:
        """Rolling-origin residual sigma for projected series, observed rows only.

        Anchors keep the sigma the anchor-only fit produced (it is the same
        model on the same rows); responders need their own, and it has to be
        computed over the rows they actually observed, otherwise the fabricated
        pre-launch region would set their forecast interval width.
        """
        T = arr.shape[0]
        H = max(int(horizon), 1)
        span = max(min(300, T // 3), 1)
        lo = max(T - span, model.slope_window)
        hi = T - H - 1
        pool = np.arange(lo, hi + 1) if hi >= lo else np.array([], dtype=int)
        if pool.size >= 4:
            if pool.size > 128:
                pool = pool[np.linspace(0, pool.size - 1, 128).astype(int)]
            origins = torch.tensor(pool, device=device, dtype=torch.long)
            with torch.no_grad():
                pred = model.rolling_forecast(origins, H).cpu().numpy()
            h_off = np.arange(1, H + 1)
            idx = pool[:, None] + h_off[None, :]
            actual = arr[idx][:, :, cols]
            seen = obs[idx][:, :, cols]
            resid = np.where(seen, pred[:, :, cols] - actual, np.nan)
        else:
            with torch.no_grad():
                recon = model().detach().cpu().numpy().T  # (T, N)
            resid = np.where(obs[:, cols], recon[:, cols] - arr[:, cols], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sig = np.nanstd(resid.reshape(-1, len(cols)), axis=0)
        return np.nan_to_num(sig, nan=0.0)

    def fit_anchor_factor_model(
        values: np.ndarray,
        n_factors: int = 6,
        knot_spacing: int = 7,
        max_lag: int = 14,
        horizon: int = 28,
        config: dict = None,
        seed: int = 42,
        device: str = 'cpu',
        verbose: int = 0,
        mask=None,
        anchors=None,
    ):
        """Anchor-aware entry point for the latent-factor trend model.

        Superset of ``fit_latent_factor_model``: without ``mask``/``anchors``
        and default config it delegates to it verbatim. With
        ``config['anchor_selection'] = True`` (or explicit ``anchors``) the fit
        becomes two-stage: factors/loadings/lags/idio lines are estimated on
        **anchors** only (long-history series), then frozen and every other
        series is projected onto them by masked least squares over its own
        observed overlap. This avoids the bias of padding a late-launching
        series backwards (``ffill().bfill()``): the fabricated flat segment
        would otherwise skew both the factor path and the GLS weights.

        Args:
            mask: optional (T, N) observation mask (bool array or DataFrame).
            anchors: optional explicit anchor column indices, overriding
                ``select_anchors``. Passing every column reproduces the
                all-series fit exactly (the negative control).

        Returns:
            (model, info). ``info`` carries everything ``fit_latent_factor_model``
            returns, plus ``'anchor_idx'``, ``'responder_idx'``,
            ``'observed_counts'``, ``'responder_lags'`` and
            ``'insufficient_overlap'`` (overlap too short to identify a
            loading — zero loadings, caller should fall back to SeasonalNaive).
        """
        cfg = dict(DEFAULT_FACTOR_CONFIG)
        if config:
            cfg.update({k: v for k, v in config.items() if v is not None})

        use_anchors = bool(cfg.get('anchor_selection')) or anchors is not None
        base_kwargs = dict(
            n_factors=n_factors,
            knot_spacing=knot_spacing,
            max_lag=max_lag,
            horizon=horizon,
            config=config,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        if not use_anchors:
            model, info = fit_latent_factor_model(values, **base_kwargs)
            info.setdefault('anchor_idx', np.arange(model.N))
            info.setdefault('responder_idx', np.array([], dtype=int))
            info.setdefault(
                'observed_counts', observed_mask(values, mask).sum(axis=0)
            )
            info.setdefault('insufficient_overlap', [])
        else:
            model, info = _fit_with_anchors(
                values, mask, anchors, cfg, base_kwargs
            )

        if cfg.get('group_factors'):
            _attach_group_factors(model, values, mask, cfg, info, horizon, seed)
        return model, info

    def _fit_with_anchors(values, mask, anchors, cfg, base_kwargs):
        """Two-stage anchor fit; see ``fit_anchor_factor_model`` for the why."""
        arr = np.asarray(values, dtype=np.float32)
        T, N = arr.shape
        obs = observed_mask(arr, mask)
        horizon = max(int(base_kwargs['horizon']), 1)
        min_observed = int(round(float(cfg['min_observed_multiple']) * horizon))
        if anchors is not None:
            anchor_idx = np.unique(np.asarray(anchors, dtype=int))
            responder_idx = np.setdiff1d(np.arange(N), anchor_idx)
        else:
            anchor_idx, responder_idx = select_anchors(
                arr, obs, min_observed=min_observed
            )
        counts = obs.sum(axis=0).astype(int)

        # ---- stage 1: the shared factors, anchors only ----------------------
        anchor_kwargs = dict(base_kwargs)
        anchor_kwargs['n_factors'] = int(
            max(1, min(int(base_kwargs['n_factors']), len(anchor_idx)))
        )
        # a loading prior is indexed by panel column; stage 1 sees only the
        # anchor columns, so slice it to match. Responders get their loadings
        # by masked least squares in _expand_to_full_panel and are correctly
        # unaffected -- short-history series do not shape the shared trend.
        prior_adj = cfg.get('prior_adjacency')
        if prior_adj is not None:
            sliced = np.asarray(prior_adj, dtype=np.float32)
            sub_cfg = dict(anchor_kwargs.get('config') or {})
            if sliced.ndim == 2 and sliced.shape == (N, N):
                sub_cfg['prior_adjacency'] = sliced[np.ix_(anchor_idx, anchor_idx)]
            else:
                sub_cfg['prior_adjacency'] = None
            anchor_kwargs['config'] = sub_cfg
        sub = arr[:, anchor_idx]
        # anchors may still carry short gaps; fill only within their observed
        # span, never past it
        sub = _interpolate_within_span(sub, obs[:, anchor_idx])
        anchor_model, info = fit_latent_factor_model(sub, **anchor_kwargs)
        dev = torch.device(base_kwargs['device'])

        if responder_idx.size == 0 and len(anchor_idx) == N and np.array_equal(
            anchor_idx, np.arange(N)
        ):
            model = anchor_model  # negative control: identical to the flat fit
        else:
            model = _expand_to_full_panel(
                anchor_model, arr, obs, anchor_idx, responder_idx, cfg,
                base_kwargs, info, dev,
            )
            sigma = np.zeros(N, dtype=float)
            sigma[anchor_idx] = np.asarray(info['sigma'], dtype=float)
            if responder_idx.size:
                sigma[responder_idx] = _responder_sigma(
                    model, arr.astype(float), obs, responder_idx, horizon, dev
                )
            info['sigma'] = sigma
            # gate indices were anchor-local; lift them to panel columns
            for key in (
                'gated_series', 'gated_trendless', 'gated_frozen',
                'gated_underperforming',
            ):
                info[key] = sorted(int(anchor_idx[i]) for i in info.get(key, []))
            info['gated_series'] = sorted(
                set(info['gated_series']) | set(info.get('insufficient_overlap', []))
            )
        if info.get('residual_matrix') is not None and (
            np.asarray(info['residual_matrix']).shape[1] != N
        ):
            # anchor-local columns: meaningless against the full panel, and
            # the responder block has its own lag structure. TVA recomputes it
            # from the expanded model when it needs one (see
            # TVA._factor_residual_matrix).
            info['residual_matrix'] = None
        info['anchor_idx'] = anchor_idx
        info['responder_idx'] = responder_idx
        info['observed_counts'] = counts
        info['min_observed'] = min_observed
        info.setdefault('insufficient_overlap', [])
        info.setdefault('responder_lags', {})
        return model, info

    def _interpolate_within_span(sub: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Linear interpolation of interior gaps only; never before first obs."""
        # ascontiguousarray, not a bare copy: a column-fancy-indexed slice can
        # come back non-C-contiguous, which perturbs torch's reduction order
        out = np.ascontiguousarray(np.asarray(sub, dtype=np.float32))
        n_time = out.shape[0]
        t = np.arange(n_time, dtype=float)
        for j in range(out.shape[1]):
            rows = np.where(obs[:, j])[0]
            if rows.size == 0 or rows.size == n_time:
                continue
            lo, hi = rows[0], rows[-1]
            span = np.arange(lo, hi + 1)
            fill = np.interp(t[span], t[rows], out[rows, j])
            out[span, j] = fill.astype(np.float32)
            # outside the observed span, hold the nearest observed edge
            out[:lo, j] = out[lo, j]
            out[hi + 1:, j] = out[hi, j]
        return out

    def _expand_to_full_panel(
        anchor_model, arr, obs, anchor_idx, responder_idx, cfg, base_kwargs,
        info, dev,
    ):
        """Copy the anchor fit into an N-series model and project responders."""
        T, N = arr.shape
        model = LatentFactorTrend(
            n_series=N,
            n_time=T,
            n_factors=anchor_model.K,
            knot_spacing=cfg['knot_spacing'],
            max_lag=base_kwargs['max_lag'],
            slope_window=cfg['slope_window'],
        ).to(dev)
        with torch.no_grad():
            model.coef.copy_(anchor_model.coef)
            model.phi_logit.copy_(anchor_model.phi_logit)
            model.idio_phi_logit.copy_(anchor_model.idio_phi_logit)
            a_idx = torch.tensor(anchor_idx, device=dev, dtype=torch.long)
            model.loadings[a_idx] = anchor_model.loadings.detach()
            model.lag_logits[a_idx] = anchor_model.lag_logits.detach()
            model.idio_level[a_idx] = anchor_model.idio_level.detach()
            model.idio_slope[a_idx] = anchor_model.idio_slope.detach()

            paths = model.factor_paths().detach().cpu().numpy().astype(float)
            t_norm = model.time_index.detach().cpu().numpy().astype(float)

        target = arr.astype(float)
        min_overlap = max(
            int(cfg['min_responder_overlap']), anchor_model.K + 3
        )
        loadings = np.zeros((responder_idx.size, anchor_model.K))
        lags = np.zeros(responder_idx.size, dtype=int)
        levels = np.zeros(responder_idx.size)
        slopes = np.zeros(responder_idx.size)
        short = []
        for i, col in enumerate(responder_idx):
            column_obs = obs[:, col]
            if int(column_obs.sum()) < min_overlap:
                # not enough overlap to identify a loading vector: the model's
                # own representation of "no shared structure" is a bare line
                levels[i], slopes[i] = _masked_line_fit(
                    target[:, col], t_norm, column_obs
                )
                short.append(int(col))
                continue
            w, d, level, slope, _ = _fit_series_on_frozen_factors(
                paths, target[:, col], column_obs, t_norm, model.max_lag,
                ridge=float(cfg['responder_ridge']),
                cap=float(cfg['responder_loading_cap']),
            )
            loadings[i], lags[i], levels[i], slopes[i] = w, d, level, slope

        with torch.no_grad():
            r_idx = torch.tensor(responder_idx, device=dev, dtype=torch.long)
            model.loadings[r_idx] = torch.tensor(
                loadings, dtype=torch.float32, device=dev
            )
            model.lag_logits[r_idx] = torch.tensor(
                _lag_logits_from_lags(lags, model.D), device=dev
            )
            model.idio_level[r_idx] = torch.tensor(
                levels, dtype=torch.float32, device=dev
            )
            model.idio_slope[r_idx] = torch.tensor(
                slopes, dtype=torch.float32, device=dev
            )
        info['insufficient_overlap'] = sorted(short)
        info['responder_lags'] = {
            int(col): int(lags[i]) for i, col in enumerate(responder_idx)
        }
        return model

    def _attach_group_factors(model, values, mask, cfg, info, horizon, seed):
        """Plan item 3c: add stability-screened group factors under the global.

        Finds block structure (movement shared by a subset of series) in the
        global-factor residual via consensus clustering (``tva/grouping.py``),
        fits a small factor model inside each surviving block, and appends
        those as extra columns with loadings zero outside the block.

        Kept only if it beats the flat rank on inner rolling-origin
        validation — a group factor is strictly more parameters, so in-sample
        improvement alone is not evidence. Mutates ``info`` and the model in
        place; does not return anything.
        """
        from autots.evaluator.tva import grouping

        arr = np.asarray(values, dtype=float)
        T, N = arr.shape
        obs = observed_mask(arr, mask)
        prune = cfg['prune_share']
        with torch.no_grad():
            paths = model.factor_paths().detach().cpu().numpy().astype(float)
        threshold = float(cfg['group_stability_threshold'])
        group_cfg = {
            'refits': int(cfg['group_refits']),
            'stability_threshold': threshold,
        }
        found = grouping.discover_groups(
            arr, global_factors=paths, config=group_cfg, seed=seed,
            knot_spacing=cfg['knot_spacing'], alpha=cfg['alpha'],
            iters=cfg['alt_iters'], init_window=cfg['init_window'],
            gls=cfg['gls'],
        )
        labels = found['labels']
        groups = found['groups']
        info['group_labels'] = labels
        info['groups'] = groups
        info['group_consensus'] = found['co_membership']

        flat_score = grouping.rolling_origin_score(
            arr, model.K, horizon, n_origins=cfg['inner_folds'],
            knot_spacing=cfg['knot_spacing'], alpha=cfg['alpha'],
            iters=cfg['alt_iters'], init_window=cfg['init_window'],
            gls=cfg['gls'],
        )
        info['group_flat_score'] = flat_score
        if not groups:
            info['group_applied'] = False
            info['group_reason'] = 'no cluster reproduced above threshold'
            info['loading_graph'] = grouping.loading_graph(
                model.fitted_loadings(prune), labels
            )
            return

        # ---- fit one small factor model inside each surviving block ---------
        resid = found['residual']
        extra_coefs, extra_loadings, member_of = [], [], []
        rank_table = {}
        for gid, members in groups.items():
            members = np.asarray(members, dtype=int)
            sub = resid[:, members]
            sel = grouping.select_rank(
                sub,
                candidates=tuple(int(r) for r in cfg['rank_candidates'] if r <= 2),
                horizon=horizon,
                n_origins=cfg['inner_folds'],
                seed=seed,
                knot_spacing=cfg['knot_spacing'], alpha=cfg['alpha'],
                iters=cfg['alt_iters'], init_window=cfg['init_window'],
                gls=cfg['gls'],
            )
            rank_table[int(gid)] = sel
            r = int(sel['rank'])
            if r <= 0:
                continue
            fit = estimate_factors_alternating(
                sub, n_factors=r, knot_spacing=cfg['knot_spacing'],
                alpha=cfg['alpha'], iters=cfg['alt_iters'],
                init_window=cfg['init_window'], gls=cfg['gls'],
            )
            for k in range(r):
                col = np.zeros(N)
                col[members] = fit['loadings'][:, k]
                extra_coefs.append(fit['coefs'][:, k])
                extra_loadings.append(col)
                member_of.append(int(gid))
        info['group_rank_selection'] = rank_table

        if not extra_coefs:
            info['group_applied'] = False
            info['group_reason'] = 'every surviving cluster selected rank 0'
            info['loading_graph'] = grouping.loading_graph(
                model.fitted_loadings(prune), labels
            )
            return

        # ---- held-out comparison: grouped layer vs the flat rank ------------
        grouped_score = grouping.rolling_origin_score(
            arr, model.K + len(extra_coefs), horizon, n_origins=cfg['inner_folds'],
            membership=labels, n_global=model.K,
            knot_spacing=cfg['knot_spacing'], alpha=cfg['alpha'],
            iters=cfg['alt_iters'], init_window=cfg['init_window'],
            gls=cfg['gls'],
        )
        info['group_score'] = grouped_score
        wins = np.isfinite(grouped_score) and (
            not np.isfinite(flat_score) or grouped_score < flat_score
        )
        if not wins:
            info['group_applied'] = False
            info['group_reason'] = (
                f'group layer did not beat flat rank on inner validation '
                f'({grouped_score:.4f} vs {flat_score:.4f})'
            )
            info['loading_graph'] = grouping.loading_graph(
                model.fitted_loadings(prune), labels
            )
            return

        _append_factor_columns(
            model, np.column_stack(extra_coefs), np.column_stack(extra_loadings),
            arr, obs,
        )
        info['group_applied'] = True
        info['group_factor_of'] = member_of
        info['n_factors_fit'] = model.K
        with torch.no_grad():
            info['loading_graph'] = grouping.loading_graph(
                model.loadings.detach().cpu().numpy(), labels
            )

    def _append_factor_columns(model, coefs, loadings, arr, obs):
        """Grow a fitted model by extra factor columns, in place.

        Group factors share the global hinge design, so they simply
        concatenate onto ``coef``. Idio lines are refit afterwards since the
        group factor absorbed part of what they were carrying.
        """
        K_new = coefs.shape[1]
        dev = model.coef.device
        with torch.no_grad():
            model.coef = nn.Parameter(
                torch.cat(
                    [model.coef.detach(),
                     torch.tensor(coefs, dtype=torch.float32, device=dev)],
                    dim=1,
                )
            )
            model.loadings = nn.Parameter(
                torch.cat(
                    [model.loadings.detach(),
                     torch.tensor(loadings, dtype=torch.float32, device=dev)],
                    dim=1,
                )
            )
            model.phi_logit = nn.Parameter(
                torch.cat(
                    [model.phi_logit.detach(),
                     torch.full((K_new,), 1.4922, device=dev)]
                )
            )
            model.K = model.K + K_new
            # refit the idiosyncratic line against what the factors now explain
            paths = model.factor_paths().detach().cpu().numpy()
            lam = model.loadings.detach().cpu().numpy()
            t = model.time_index.detach().cpu().numpy()
            shared = paths @ lam.T
            design = np.column_stack([np.ones_like(t), t])
            for j in range(model.N):
                rows = np.where(obs[:, j])[0]
                if rows.size < 2:  # pragma: no cover
                    continue
                coef, *_ = np.linalg.lstsq(
                    design[rows], arr[rows, j] - shared[rows, j], rcond=None
                )
                model.idio_level[j] = float(coef[0])
                model.idio_slope[j] = float(coef[1])


else:  # pragma: no cover - torch-free environments

    class LatentFactorTrend:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "trend_network='factor' requires torch. Use trend_network='none'."
            )

    def _gate_trendless_series(model, y: 'torch.Tensor', min_ratio: float) -> list:
        """Zero the loadings of series that have no low-frequency structure."""
        if not min_ratio:
            return []
        with torch.no_grad():
            target = y.detach().cpu().numpy().T  # (T, N)
            window = max(min(model.T // 20, 28), 3)
            smoothed = _rolling_mean(target, window)
            ratio = smoothed.std(axis=0) / np.maximum(
                (target - smoothed).std(axis=0), 1e-9
            )
            drop = np.where(ratio < float(min_ratio))[0]
            if drop.size:
                t = model.time_index.detach().cpu().numpy()
                design = np.column_stack([np.ones_like(t), t])
                coef, *_ = np.linalg.lstsq(design, target[:, drop], rcond=None)
                model.loadings[drop] = 0.0
                model.idio_level[drop] = torch.tensor(
                    coef[0], dtype=model.idio_level.dtype,
                    device=model.idio_level.device,
                )
                model.idio_slope[drop] = torch.tensor(
                    coef[1], dtype=model.idio_slope.dtype,
                    device=model.idio_slope.device,
                )
        return [int(i) for i in drop]

    def fit_latent_factor_model(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "trend_network='factor' requires torch. Use trend_network='none'."
        )

    def fit_anchor_factor_model(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "trend_network='factor' requires torch. Use trend_network='none'."
        )
