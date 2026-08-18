# -*- coding: utf-8 -*-
"""Sparse-code factor identification for ``trend_network='factor'`` (C9).

The alternating estimator in :mod:`factor_network` finds the factor *span* by
SVD and then solves loadings by regression. Both steps are rotation-invariant,
so the *basis* it lands on is arbitrary -- and the basis is exactly what the
coherence graph consumes. Varimax (C1) addresses that by rotating afterwards,
and demonstrably works on a clean panel (dominant recovery 0.972), but a
post-hoc rotation needs an already-clean span: on the detector-adjusted panel
the C1-C8 ladder measured it moving dominant recovery 0.250 -> 0.139, i.e.
actively backwards.

This module takes the other route. Treat each *series* as a sample whose code
is its loading vector, parameterize the dictionary directly in the hinge trend
basis, and allow each code at most ``code_topk`` nonzeros. Sparsity is then
learned jointly with the span instead of rotated in afterwards, and a hard
support constraint is not rotation-invariant, so the basis is pinned without
any rotation step at all. Atoms nobody selects fall out, which makes the live
atom count an implicit rank estimate.

Two tiers share one objective:

``'sparse_alt'``
    Torch-free coordinate descent. Greedy support selection per series against
    an unpenalized idiosyncratic line, an unpenalized refit on the chosen
    support, then a dictionary update through the existing
    :func:`factor_network._l1_trend_filter`. Support is re-selected from
    scratch every iteration, so nothing locks in.
``'sparse_ae'``
    The same objective trained by Adam as an autoencoder over free codes.
    Signed magnitude-TopK, AuxK revival for dead atoms, dense warmup. Support
    *does* lock in after warmup, which is the substantive difference from the
    tier above and the reason both are worth measuring. Falls back to
    ``'sparse_alt'`` when torch is unavailable.

Both are initialized from -- and fall back to -- the alternating estimator, so
enabling one can never leave a panel worse identified than the default.

Returns the same identification contract the rest of the module speaks:
``factors`` (T, K) centered with unit-std increments, ``loadings`` (N, K),
``coefs`` (P, K) in the hinge basis, ``design`` (T, P), ``weights`` (N,), plus
``idio_level``/``idio_slope`` and the atom diagnostics.
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


DEFAULT_SPARSE_FACTOR_CONFIG = {
    # ---- sparsity ---------------------------------------------------------
    # nonzero loadings per series. An integer, deliberately: this module has
    # twice been burned by a continuous threshold whose useful range was
    # nowhere near the swept grid.
    'code_topk': 1,
    # fraction of the schedule spent dense (k = K) before the support
    # constraint bites, so the span is established before any winner-take-all
    'warmup_frac': 0.2,
    # zero a series' whole code when its shared component is this small a
    # fraction of the series' own scale. group_graph already reads an all-zero
    # row as "no exposure", which is the only abstention channel left once
    # code_topk == 1 makes dominance_margin/min_loading_share unconditional.
    'min_code_share': 0.0,
    # Rotate the initializer toward simple structure before the sparse fit.
    # This is not belt-and-braces: with hard k=1 codes, a dictionary atom that
    # happens to be a MIXTURE of two true factors is a saddle point. A series
    # loading purely on f0 correlates equally with (f0+f1) and (f0-f1), so
    # greedy selection cannot tell a merged atom from a split one and no amount
    # of dead-atom revival escapes it -- the residual direction is symmetric
    # between exactly the two factors that need separating. Starting from an
    # approximately simple basis removes the tie before it forms. Measured:
    # atom usage [0 16 0 8] (two factors merged) -> [8 8 8 0] (correct).
    # A tuple runs one sparse fit per entry and keeps the lowest-reconstruction
    # -loss result. Restarts are not belt-and-braces either: the measured
    # collapses are basin-dependent, and reconstruction loss ranks them
    # correctly. On the cell where varimax init collapsed to [8 16 0 0]
    # (loss 4.01), the quartimax start recovered [8 8 8 0] at loss 2.28 -- the
    # best-loss pick is the right pick, so no new selection criterion is needed.
    'init_rotate': (None, 'varimax', 'quartimax'),
    # ---- idiosyncratic term ----------------------------------------------
    # per-series level + slope, unpenalized. A LINE, never a hinge trend --
    # see the module note in _solve_codes.
    'idio': True,
    # ---- dead atoms -------------------------------------------------------
    'min_atom_users': 2,     # an atom used by one series asserts no pairs
    'aux_k': 2,              # dead atoms given residual gradient per step
    'aux_weight': 0.03125,   # 1/32, the published AuxK default
    'dead_steps': 50,
    # Stop reviving dead atoms after this fraction of the schedule. Revival
    # exists to escape a collapse early; near convergence a dead atom is a
    # legitimate rank signal, and reseeding it every iteration just keeps
    # handing a spurious direction fresh chances to steal series -- which is
    # precisely what over-specified K produces. Measured: with revival running
    # to the end, the correctly-dead 4th atom re-acquired members and usage
    # oscillated; gated, it stays dead.
    'aux_until': 0.6,
    # ---- optimization -----------------------------------------------------
    'alt_iters': 6,          # 'sparse_alt' only
    'steps': 1000,           # 'sparse_ae' only
    'lr': 5e-2,
    'huber_delta': 1.0,
    'use_gls_weights': True,
    'val_frac': 0.1,
    'check_every': 25,
    'patience': 200,
    'grad_clip': 1.0,
    # ---- guards -----------------------------------------------------------
    # Accept the sparse fit when its weighted reconstruction loss is within
    # this multiple of the initializer's. Above 1.0 on purpose: a k-sparse fit
    # is strictly more constrained than the dense lstsq it starts from, so
    # demanding an exact match would reject it unconditionally. This is a
    # catastrophe guard, not a model-selection criterion.
    'accept_margin': 1.10,
}

__all__ = [
    'DEFAULT_SPARSE_FACTOR_CONFIG',
    'signed_topk',
    'identify',
    'HAS_TORCH',
]


def _merge_config(config=None):
    cfg = dict(DEFAULT_SPARSE_FACTOR_CONFIG)
    if config:
        cfg.update({k: v for k, v in dict(config).items() if v is not None})
    return cfg


def signed_topk(z, k):
    """Keep the ``k`` largest-magnitude entries of each row, with their signs.

    Non-negative activations are wrong for this problem: ``coherence`` builds
    ``'f{k}+'``/``'f{k}-'`` group keys and a series can be negatively exposed
    to a factor, so the sign is load-bearing information rather than a nuisance
    to be rectified away.

    Selection is by exact rank, never by comparing against the k-th value: at
    initialization many entries are exactly zero and a ``>=`` threshold would
    silently admit more than ``k`` of them.

    Args:
        z: (N, K) array of unconstrained codes.
        k: number of nonzeros to keep per row.

    Returns:
        (N, K) array, zero everywhere outside the selected support.
    """
    arr = np.asarray(z, dtype=float)
    if arr.ndim != 2:
        return arr
    n_cols = arr.shape[1]
    k = int(k)
    if k >= n_cols:
        return arr.copy()
    if k <= 0:
        return np.zeros_like(arr)
    idx = np.argpartition(-np.abs(arr), k - 1, axis=1)[:, :k]
    out = np.zeros_like(arr)
    np.put_along_axis(out, idx, np.take_along_axis(arr, idx, axis=1), axis=1)
    return out


def _idio_basis(n_time, idio=True):
    """(T, 1) intercept, or (T, 2) intercept + normalized time."""
    ones = np.ones((int(n_time), 1))
    if not idio:
        return ones
    t = (np.arange(int(n_time), dtype=float) / max(int(n_time), 1))[:, None]
    return np.hstack([ones, t])


def _residualize(mat, base_pinv, base):
    """Project ``mat`` onto the orthogonal complement of ``base``."""
    return mat - base @ (base_pinv @ mat)


def _solve_codes(factors, arr, k, idio=True, min_code_share=0.0):
    """Sparse per-series loading solve against an unpenalized idio line.

    Selection is greedy forward (orthogonal matching pursuit) on the
    idio-residualized problem, which for ``k == 1`` is exact best-subset --
    deliberately not the argmax of a dense solve, because on correlated atoms a
    dense coefficient vector routinely puts its largest entry on the wrong one,
    and the graph partitions on precisely that argmax.

    The idiosyncratic term is a LINE and nothing more. That is what
    ``LatentFactorTrend.idio_component`` already carries, so identification and
    refinement agree; and it is deliberately not the per-series hinge trend the
    robust input estimator uses, which was flexible enough to absorb the
    factors themselves (true-factor R^2 0.951 with chance-level loading
    recovery). A line can take the genuinely linear part of idiosyncratic drift
    -- the part that competes with the factors in a variance-ranked
    decomposition -- and cannot absorb a changepoint.

    Args:
        factors: (T, K) factor paths.
        arr: (T, N) panel, in level units (not centered).
        k: nonzeros per series.
        idio: fit a per-series slope in addition to the level.
        min_code_share: zero a series' code when its shared component's std is
            below this fraction of the series' own std.

    Returns:
        (loadings (N, K), level (N,), slope (N,)).
    """
    fac = np.asarray(factors, dtype=float)
    panel = np.asarray(arr, dtype=float)
    n_time, n_fac = fac.shape
    n_series = panel.shape[1]
    k = int(max(1, min(int(k), n_fac)))

    base = _idio_basis(n_time, idio)
    base_pinv = np.linalg.pinv(base)
    # residualize the dictionary once, not per series
    fac_r = _residualize(fac, base_pinv, base)
    energy = np.einsum('tk,tk->k', fac_r, fac_r)
    alive = energy > 1e-12

    loadings = np.zeros((n_series, n_fac))
    level = np.zeros(n_series)
    slope = np.zeros(n_series)

    for j in range(n_series):
        y = panel[:, j]
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=float(np.nanmean(y)) if np.isfinite(y).any() else 0.0)
        y_r = y - base @ (base_pinv @ y)
        chosen = []
        resid = y_r.copy()
        for _ in range(k):
            best, best_gain = -1, 0.0
            for c in range(n_fac):
                if c in chosen or not alive[c]:
                    continue
                dot = float(fac_r[:, c] @ resid)
                gain = dot * dot / float(energy[c])
                if gain > best_gain:
                    best_gain, best = gain, c
            if best < 0 or best_gain <= 1e-12:
                break
            chosen.append(best)
            cols = fac_r[:, chosen]
            coef, *_ = np.linalg.lstsq(cols, y_r, rcond=None)
            resid = y_r - cols @ coef
        if not chosen:
            # no atom explains anything here: honest zero exposure, and
            # the idio line still absorbs the level/slope
            fit, *_ = np.linalg.lstsq(base, y, rcond=None)
            level[j] = float(fit[0])
            if idio and fit.size > 1:
                slope[j] = float(fit[1])
            continue
        design = np.hstack([base, fac[:, chosen]])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        n_base = base.shape[1]
        level[j] = float(coef[0])
        if idio and n_base > 1:
            slope[j] = float(coef[1])
        row = np.zeros(n_fac)
        row[chosen] = coef[n_base:]
        if min_code_share > 0.0:
            shared = fac @ row
            y_sd = float(np.std(y))
            if y_sd > 0 and float(np.std(shared)) < float(min_code_share) * y_sd:
                row[:] = 0.0  # abstain rather than assert a weak exposure
        loadings[j] = row
    return loadings, level, slope


def _atom_stats(loadings, min_users=2):
    """Usage counts, energy, live count and dead indices for the dictionary."""
    lam = np.asarray(loadings, dtype=float)
    usage = (np.abs(lam) > 0).sum(axis=0).astype(int)
    energy = np.einsum('nk,nk->k', lam, lam)
    live = int((usage >= int(min_users)).sum())
    dead = np.where(usage < int(min_users))[0]
    return usage, energy, live, dead


def _reseed_dead(scores, resid_centered, factors, loadings, dead):
    """Point dead atoms at the unexplained residual.

    An atom nobody selects gets an all-zero row in ``loadings``, so the
    pseudo-inverse in the dictionary step returns a zero score column and the
    atom's content is *destroyed* -- collapse is absorbing. Re-seeding from the
    leading directions of the current residual is the coordinate-descent
    analogue of AuxK: it moves the atom to where the unexplained variance is
    and lets the next code solve decide whether to use it.

    Crucially this perturbs the *dictionary*, never the codes. Forcing
    worst-fit series onto dead atoms was measured during planning at pair
    precision 0.855 -> 0.763 on a clean panel, at identical final usage.
    """
    if dead.size == 0:
        return scores
    resid = resid_centered - factors @ loadings.T
    try:
        u, s, _ = np.linalg.svd(resid, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate residual
        return scores
    for i, atom in enumerate(dead):
        if i >= u.shape[1] or s[i] <= 0:
            break
        scores[:, atom] = u[:, i] * s[i]
    return scores


def _finalize(factors, coefs, arr, design, weights, cfg, method):
    """Normalize, resolve the final codes, and build the contract dict.

    Reproduces the alternating estimator's exit convention exactly: unit-std
    factor *increments*, with the scale absorbed into the loadings. Not unit-l2
    columns (the sparse-autoencoder literature's default) -- ``coherence``'s
    precision weights divide ``max|loading|`` by ``info['sigma']``, a residual
    std in the normalized panel, so a loading has to mean "normalized units per
    unit-std factor increment" or every precision weight silently rescales.
    """
    fac = np.asarray(factors, dtype=float)
    coe = np.asarray(coefs, dtype=float)
    sd = np.diff(fac, axis=0).std(axis=0)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    fac = fac / sd[None, :]
    coe = coe / sd[None, :]

    loadings, level, slope = _solve_codes(
        fac, arr, cfg['code_topk'], idio=bool(cfg['idio']),
        min_code_share=float(cfg['min_code_share']),
    )
    # Orient by the same magnitude-weighted mass vote coherence.resolve_signs
    # uses, so the basis ships already oriented instead of being re-oriented at
    # graph-build time.
    mass = (loadings * np.abs(loadings)).sum(axis=0)
    signs = np.sign(mass)
    signs[signs == 0] = 1.0
    fac = fac * signs[None, :]
    coe = coe * signs[None, :]
    loadings = loadings * signs[None, :]

    usage, energy, live, dead = _atom_stats(loadings, cfg['min_atom_users'])
    # The model anchors its factor paths at t=0 rather than centering them, so
    # the column mean of design @ coefs has to come out of the idio level for
    # the reconstruction to survive the parameter copy unchanged.
    paths_mean = (design @ coe).mean(axis=0)
    return {
        'factors': fac,
        'loadings': loadings,
        'coefs': coe,
        'design': design,
        'weights': weights,
        'idio_level': level - loadings @ paths_mean,
        'idio_slope': slope,
        'atom_usage': usage,
        'atom_energy': energy,
        'n_atoms_live': live,
        'dead_atoms': dead.tolist(),
        'identification_method': method,
        'sparse_rejected': False,
    }


def _weighted_huber(resid, weights, delta=1.0):
    """Mean Huber loss with per-series weights. Never raises."""
    r = np.asarray(resid, dtype=float)
    if not np.isfinite(r).any():
        return float('inf')
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.abs(r)
    loss = np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
    return float(np.mean(loss * np.asarray(weights, dtype=float)[None, :]))


def _prepare_init(init, cfg):
    """Rotate the initializer toward simple structure. See ``init_rotate``."""
    method = cfg.get('init_rotate')
    if not method:
        return init
    from autots.evaluator.tva.factor_network import rotate_identification

    return rotate_identification(init, method=str(method), kaiser=True)


def _recon_loss(arr, out, weights, cfg):
    """Weighted Huber of an identification's implied reconstruction."""
    base = _idio_basis(arr.shape[0], bool(cfg['idio']))
    idio = base @ np.vstack([out['idio_level'], out['idio_slope']])[: base.shape[1]]
    paths = out['design'] @ out['coefs']
    return _weighted_huber(
        arr - (paths @ out['loadings'].T + idio), weights, float(cfg['huber_delta'])
    )


def _sparse_alt(arr, init, cfg, alpha):
    """Torch-free sparse dictionary learning over series. See module docstring."""
    from autots.evaluator.tva.factor_network import _l1_trend_filter

    init = _prepare_init(init, cfg)
    design = np.asarray(init['design'], dtype=float)
    weights = np.asarray(init['weights'], dtype=float)
    if not bool(cfg['use_gls_weights']):
        weights = np.ones_like(weights)
    factors = np.asarray(init['factors'], dtype=float)
    coefs = np.asarray(init['coefs'], dtype=float)
    n_fac = factors.shape[1]

    iters = max(int(cfg['alt_iters']), 1)
    n_warm = int(round(float(cfg['warmup_frac']) * iters))
    k_target = int(max(1, min(int(cfg['code_topk']), n_fac)))
    yc = arr - arr.mean(axis=0, keepdims=True)

    loadings = np.asarray(init['loadings'], dtype=float)
    for it in range(iters):
        # dense during warmup so the span settles before the support does
        k_now = n_fac if it < n_warm else k_target
        loadings, level, slope = _solve_codes(
            factors, arr, k_now, idio=bool(cfg['idio']),
            min_code_share=0.0,  # abstention is applied once, at exit
        )
        base = _idio_basis(arr.shape[0], bool(cfg['idio']))
        idio = base @ np.vstack([level, slope])[: base.shape[1]]
        resid = arr - idio
        resid = resid - resid.mean(axis=0, keepdims=True)

        lw = loadings.T * weights[None, :]  # (K, N)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scores = (resid * weights[None, :]) @ np.linalg.pinv(lw)  # (T, K)
        if it < max(int(round(float(cfg['aux_until']) * iters)), 1):
            _, _, _, dead = _atom_stats(loadings, cfg['min_atom_users'])
            scores = _reseed_dead(scores, resid, factors, loadings, dead)
        factors, coefs = _l1_trend_filter(scores, design, alpha)
        sd = factors.std(axis=0)
        sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
        factors = factors / sd[None, :]
        coefs = coefs / sd[None, :]

    del yc
    return _finalize(factors, coefs, arr, design, weights, cfg, 'sparse_alt')


def identify(values, n_factors, init, method='sparse_alt', config=None,
             alpha=1e-3, seed=42, device='cpu'):
    """Refine an alternating identification into a sparse-code one.

    Args:
        values: (T, N) normalized panel, as handed to the alternating estimator.
        n_factors: fitted rank K.
        init: the identification dict from
            :func:`factor_network.estimate_factors_alternating`. Used as both
            the initializer and the fallback.
        method: ``'sparse_alt'`` or ``'sparse_ae'``.
        config: overrides for :data:`DEFAULT_SPARSE_FACTOR_CONFIG`.
        alpha: trend-filter smoothness, from the parent factor config.
        seed: torch seed for the autoencoder tier.
        device: torch device for the autoencoder tier.

    Returns:
        An identification dict, or ``None`` when the sparse fit failed or
        reconstructed materially worse than ``init`` -- in which case the
        caller keeps ``init``. Never raises.
    """
    cfg = _merge_config(config)
    name = str(method or 'sparse_alt').lower()
    if name == 'sparse_ae' and not HAS_TORCH:
        name = 'sparse_alt'
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 1:
        return None
    if not init or 'design' not in init:
        return None

    weights_ref = np.asarray(init['weights'], dtype=float)
    if not bool(cfg['use_gls_weights']):
        weights_ref = np.ones_like(weights_ref)
    starts = cfg.get('init_rotate')
    if starts is None or isinstance(starts, str):
        starts = (starts,)
    out, out_loss = None, float('inf')
    for start in tuple(starts) or (None,):
        run_cfg = dict(cfg)
        run_cfg['init_rotate'] = start
        try:
            if name == 'sparse_ae':
                cand = _sparse_ae(arr, init, run_cfg, alpha, seed=seed, device=device)
            else:
                cand = _sparse_alt(arr, init, run_cfg, alpha)
        except Exception as exc:  # pragma: no cover - never fail a fit
            warnings.warn(
                f"TVA sparse identification ({name}, init_rotate={start}) failed. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if cand is None:
            continue
        loss = _recon_loss(arr, cand, weights_ref, cfg)
        if loss < out_loss:
            cand['init_rotate_used'] = start
            out, out_loss = cand, loss
    if out is None:
        return None
    for key in ('factors', 'loadings', 'coefs'):
        if not np.isfinite(np.asarray(out[key], dtype=float)).all():
            return None

    weights = weights_ref
    delta = float(cfg['huber_delta'])
    base = _idio_basis(arr.shape[0], bool(cfg['idio']))
    loss_new = out_loss
    # Compare against the initializer under the SAME sparsity, not against its
    # dense lstsq solution. A k-sparse fit reconstructs worse than a dense one
    # by construction, so the dense comparison would reject unconditionally;
    # the question worth asking is whether the learned dictionary beats the
    # initializer's dictionary when both are held to k nonzeros per series.
    ref_lam, ref_lev, ref_slo = _solve_codes(
        init['factors'], arr, cfg['code_topk'], idio=bool(cfg['idio']),
        min_code_share=float(cfg['min_code_share']),
    )
    ref_idio = base @ np.vstack([ref_lev, ref_slo])[: base.shape[1]]
    loss_init = _weighted_huber(
        arr - (init['factors'] @ ref_lam.T + ref_idio), weights, delta
    )
    margin = float(cfg['accept_margin'])
    if not np.isfinite(loss_new) or loss_new > margin * max(loss_init, 1e-12):
        rejected = dict(init)
        rejected['sparse_rejected'] = True
        rejected['identification_method'] = 'alternating'
        return rejected
    return out


if HAS_TORCH:

    class SparseFactorAutoencoder(nn.Module):
        """Autoencoder whose codes are loadings and whose atoms are trends.

        The decoder is parameterized directly in the hinge basis, so the atoms
        are piecewise-linear trends by construction and ``coefs`` falls out of
        the fit rather than having to be projected back afterwards. Atoms carry
        their own unit-increment normalization inside :meth:`atoms`, which is
        what makes the code magnitudes comparable across atoms -- and TopK
        compares magnitudes, so without it an atom could win selection simply
        by inflating its own scale.
        """

        def __init__(self, design, n_series, n_factors, code_topk=1, idio=True):
            super().__init__()
            self.register_buffer(
                'design', torch.as_tensor(design, dtype=torch.float32)
            )
            self.T, self.P = int(self.design.shape[0]), int(self.design.shape[1])
            self.N = int(n_series)
            self.K = int(n_factors)
            self.k = int(max(1, min(int(code_topk), self.K)))
            self.idio = bool(idio)
            self.coef = nn.Parameter(torch.zeros(self.P, self.K))
            self.z = nn.Parameter(torch.zeros(self.N, self.K))
            self.level = nn.Parameter(torch.zeros(self.N))
            self.slope = nn.Parameter(torch.zeros(self.N))
            self.register_buffer(
                't_norm', torch.arange(self.T, dtype=torch.float32) / max(self.T, 1)
            )
            self.register_buffer('since_used', torch.zeros(self.K))

        def atoms(self):
            """(T, K) factor paths with unit-std increments, anchored at t=0."""
            raw = self.design @ self.coef
            step = raw[1:] - raw[:-1]
            sd = step.std(dim=0, unbiased=False).clamp(min=1e-4)
            return raw / sd[None, :]

        def codes(self, k=None):
            """(N, K) signed magnitude-TopK codes -- i.e. the loadings."""
            k = self.k if k is None else int(k)
            if k >= self.K:
                return self.z
            idx = self.z.abs().topk(k, dim=1).indices
            mask = torch.zeros_like(self.z).scatter_(1, idx, 1.0)
            return self.z * mask

        def forward(self, k=None):
            out = self.atoms() @ self.codes(k).t() + self.level[None, :]
            if self.idio:
                out = out + self.t_norm[:, None] * self.slope[None, :]
            return out

        @torch.no_grad()
        def renormalize_(self):
            """Rescale ``coef`` to unit-increment atoms. Exactly output-preserving.

            ``atoms`` divides by its own increment std, so scaling ``coef``
            leaves the atoms -- and therefore the reconstruction -- untouched;
            the codes must NOT be counter-scaled. This only keeps the raw
            parameter numerically well-conditioned so the exported ``coefs``
            are directly usable by ``LatentFactorTrend``.
            """
            raw = self.design @ self.coef
            step = raw[1:] - raw[:-1]
            sd = step.std(dim=0, unbiased=False).clamp(min=1e-4)
            self.coef.div_(sd[None, :])

    def _sparse_ae(arr, init, cfg, alpha, seed=42, device='cpu'):
        """Gradient tier. Same objective as ``sparse_alt``, trained by Adam."""
        torch.manual_seed(int(seed))
        rng = np.random.default_rng(int(seed))
        dev = torch.device(device)
        init = _prepare_init(init, cfg)
        n_time, n_series = arr.shape
        design = np.asarray(init['design'], dtype=float)
        weights = np.asarray(init['weights'], dtype=float)
        if not bool(cfg['use_gls_weights']):
            weights = np.ones_like(weights)
        n_fac = int(init['loadings'].shape[1])

        model = SparseFactorAutoencoder(
            design, n_series, n_fac,
            code_topk=int(cfg['code_topk']), idio=bool(cfg['idio']),
        ).to(dev)
        with torch.no_grad():
            model.coef.copy_(torch.as_tensor(init['coefs'], dtype=torch.float32, device=dev))
            model.z.copy_(torch.as_tensor(init['loadings'], dtype=torch.float32, device=dev))
            model.level.copy_(torch.as_tensor(arr.mean(axis=0), dtype=torch.float32, device=dev))
            model.renormalize_()

        y = torch.as_tensor(arr, dtype=torch.float32, device=dev)          # (T, N)
        w = torch.as_tensor(weights, dtype=torch.float32, device=dev)[None, :]
        steps = max(int(cfg['steps']), 1)
        n_warm = int(round(float(cfg['warmup_frac']) * steps))
        delta = float(cfg['huber_delta'])
        aux_k = int(cfg['aux_k'])
        aux_weight = float(cfg['aux_weight'])
        dead_steps = int(cfg['dead_steps'])

        n_val = int(np.clip(int(float(cfg['val_frac']) * n_time), 1, n_time - 2))
        val_idx = rng.choice(n_time, size=n_val, replace=False)
        val_mask = torch.zeros(n_time, dtype=torch.bool, device=dev)
        val_mask[torch.as_tensor(np.sort(val_idx), device=dev)] = True
        train_mask = ~val_mask

        opt = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
        best_val, best_state, bad = float('inf'), None, 0
        check_every = max(int(cfg['check_every']), 1)
        patience = int(cfg['patience'])

        def _loss(pred, target, mask):
            per = F.huber_loss(pred[mask], target[mask], delta=delta, reduction='none')
            return (per * w).mean()

        for step in range(steps):
            # Anneal the support K -> k_target across warmup rather than
            # dropping off a cliff. An unselected entry receives no gradient
            # (the TopK mask is a constant), so a hard cliff freezes the
            # ranking the initializer happened to arrive at; stepping down lets
            # the ranking be revised while every entry is still learning.
            if n_warm > 0 and step < n_warm:
                frac = step / float(n_warm)
                k_now = int(round(n_fac + frac * (model.k - n_fac)))
                k_now = int(np.clip(k_now, model.k, n_fac))
            else:
                k_now = None
            opt.zero_grad(set_to_none=True)
            recon = model(k_now)
            loss = _loss(recon, y, train_mask)
            # AuxK: dead atoms reconstruct the residual, so a collapsed atom
            # can come back. Never by overwriting codes -- forcing assignments
            # measured strictly worse at identical final usage.
            if aux_k > 0 and n_warm <= step < int(float(cfg['aux_until']) * steps):
                dead = (model.since_used > dead_steps).nonzero().flatten()
                if dead.numel():
                    pick = dead[: aux_k]
                    resid = (y - recon).detach()
                    aux = model.atoms()[:, pick] @ model.z[:, pick].t()
                    loss = loss + aux_weight * _loss(aux, resid, train_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip']))
            opt.step()
            with torch.no_grad():
                model.renormalize_()
                used = (model.codes(k_now).abs() > 0).any(dim=0)
                model.since_used.add_(1.0).masked_fill_(used, 0.0)

            # Only score checkpoints once the support constraint is fully on:
            # a denser warmup state always wins on reconstruction, so tracking
            # from step 0 would lock the best state to the initializer.
            if (step + 1) % check_every == 0 and step >= n_warm:
                with torch.no_grad():
                    val = float(_loss(model(), y, val_mask))
                if val < best_val - 1e-9:
                    best_val, bad = val, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad * check_every >= patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)

        with torch.no_grad():
            model.renormalize_()
            atoms = model.atoms().cpu().numpy().astype(float)
            coefs = model.coef.cpu().numpy().astype(float)
        factors = atoms - atoms.mean(axis=0, keepdims=True)
        return _finalize(factors, coefs, arr, design, weights, cfg, 'sparse_ae')

else:  # pragma: no cover - torch-free environments

    class SparseFactorAutoencoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "identification='sparse_ae' requires torch. "
                "Use identification='sparse_alt' for the torch-free tier."
            )

    def _sparse_ae(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("identification='sparse_ae' requires torch.")
