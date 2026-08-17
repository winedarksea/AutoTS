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
    # extrapolation (stage B)
    'stage_b_steps': 200,
    'lr_phi': 1e-2,
    'n_origins': 64,
    'slope_window': 90,
    'min_trend_to_noise': 0.6,
    # exposure
    'prune_share': 0.02,
    'grad_clip': 1.0,
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
        max_knots: cap on candidate knots. The l1 solve is superlinear in the
            number of columns, so an uncapped 7-day grid makes long panels slow
            (131 s at N=200 / T=3000); the spacing is widened instead. The cap
            only binds beyond ~1400 steps at the default spacing, where knots
            that fine are far below the resolution the noise supports anyway.

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
    ``alpha`` is a pure smoothness knob rather than a quantity that has to be
    retuned for every panel's scale (an unstandardized penalty measured 0.05 vs
    0.68 recovery on the same design across two panels).

    Args:
        scores: (T, K) noisy level-space factor scores.
        design: (T, P) hinge design from ``hinge_design``.
        alpha: Lasso penalty on the standardized hinge coefficients.

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

    Daily noise dominates the raw level-space spectrum, so it is smoothed away
    first; the rank is then the Ahn-Horenstein eigenvalue-ratio argmax
    (``argmax_k s_k / s_{k+1}``), which needs no threshold calibration.

    Args:
        values: (T, N) normalized level panel.
        cap: maximum factors to report.
        window: smoothing window for noise suppression.

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


def estimate_factors_alternating(
    values: np.ndarray,
    n_factors: int,
    knot_spacing: int = 7,
    alpha: float = 1e-3,
    iters: int = 6,
    init_window: int = 91,
    gls: bool = True,
):
    """Identify latent factor paths and loadings by alternating GLS/l1-TF.

    Torch-free. This is the stage that actually finds the factors; the torch
    model refines around it.

    Args:
        values: (T, N) normalized adjusted level panel.
        n_factors: K.
        knot_spacing: candidate-knot spacing for the trend-filter basis.
        alpha: l1 penalty on hinge coefficients.
        iters: alternating iterations.
        init_window: smoothing window for the SVD initialization.
        gls: reweight series by inverse high-frequency residual scale.

    Returns:
        dict with 'factors' (T, K), 'loadings' (N, K), 'coefs' (P, K),
        'design' (T, P), 'weights' (N,).
    """
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
        loadings, *_ = np.linalg.lstsq(factors, yc, rcond=None)  # (K, N)
        lw = loadings * weights[None, :]
        scores = (yc * weights[None, :]) @ np.linalg.pinv(lw)  # (T, K)
        factors, coefs = _l1_trend_filter(scores, design, alpha)
        # normalize on the LEVEL scale inside the loop: ``alpha`` penalizes
        # hinge coefficients in level units, so rescaling the target here (e.g.
        # to unit-increment std, ~300x larger) would silently gut the penalty
        # and collapse every factor onto the same noise-chasing path
        sd = factors.std(axis=0)
        sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
        factors = factors / sd[None, :]
        coefs = coefs / sd[None, :]
        if gls:
            loadings, *_ = np.linalg.lstsq(factors, yc, rcond=None)
            resid = yc - factors @ loadings
            hf = np.diff(resid, axis=0).std(axis=0) / np.sqrt(2.0)
            weights = 1.0 / np.maximum(hf, 1e-6)
            weights = weights / max(weights.mean(), 1e-12)
    # final identification: unit-std increments, matching the torch model's
    # ``factor_paths`` normalization so coefficients transfer without rescale
    sd = np.diff(factors, axis=0).std(axis=0)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    factors = factors / sd[None, :]
    coefs = coefs / sd[None, :]
    loadings, *_ = np.linalg.lstsq(factors, yc, rcond=None)
    return {
        'factors': factors,
        'loadings': loadings.T,
        'coefs': coefs,
        'design': design,
        'weights': weights,
    }


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

    Args:
        values: (T, N) normalized adjusted level panel.
        n_factors: K to fit on each half.
        n_reps: random splits to average over.
        seed: RNG seed for the splits.
        **kwargs: forwarded to ``estimate_factors_alternating``.

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
            future = paths[origins][:, None, :] + damp[None, :, :] * slope[:, None, :]

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

        def forecast(self, horizon: int, origin: int = None) -> 'torch.Tensor':
            """(N, H) normalized-space forecast from ``origin`` (default last)."""
            if origin is None:
                origin = self.T - 1
            origins = torch.tensor(
                [int(origin)], device=self.coef.device, dtype=torch.long
            )
            return self.rolling_forecast(origins, horizon)[0].T

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

            The threshold is deliberately low: a genuine but minor factor can
            carry only ~5% of the loading energy, and pruning it costs more
            recovery than a dead column does. Guarding against spurious factors
            is the job of ``select_n_factors`` / an explicit ``n_factors``.
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
            # share of the *reconstructed* trend variation carried by the
            # shared factor term.
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

    def _gate_trendless_series(model, y: 'torch.Tensor', min_ratio: float) -> list:
        """Zero the loadings of series that have no low-frequency structure.

        Args:
            model: fitted LatentFactorTrend (mutated in place).
            y: (N, T) normalized panel the model was fit to.
            min_ratio: minimum smoothed-to-residual spread ratio to keep
                loadings.

        Returns:
            Indices of the series whose loadings were zeroed.
        """
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

        Args:
            values: (T, N) **normalized** adjusted level panel (see
                ``robust_level_scale``).
            n_factors: K to fit before pruning.
            knot_spacing: candidate-knot spacing of the trend-filter basis.
            max_lag: maximum learned response lag.
            horizon: horizon used by the stage-B damping calibration.
            config: overrides for ``DEFAULT_FACTOR_CONFIG``.
            seed: torch/numpy seed.
            device: torch device string.
            verbose: 0 silent, 1 summary, 2 per-check losses.

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
        ident = estimate_factors_alternating(
            arr,
            n_factors=K,
            knot_spacing=cfg['knot_spacing'],
            alpha=cfg['alpha'],
            iters=cfg['alt_iters'],
            init_window=cfg['init_window'],
            gls=cfg['gls'],
        )

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
            model.idio_level.copy_(
                torch.tensor(arr.mean(axis=0), dtype=torch.float32, device=dev)
            )

        # ---- stage A: refine loadings, lags and the idiosyncratic line ------
        val_mask = torch.zeros(T, dtype=torch.bool, device=dev)
        n_val = int(np.clip(int(cfg['val_frac'] * T), 1, max(T - 2, 1)))
        val_mask[torch.tensor(np.sort(rng.choice(T, n_val, replace=False)), device=dev)] = True
        train_mask = ~val_mask

        # the alternating estimator identifies the factor paths
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
                # proximal l1 on the hinge coefficients keeps the factor paths
                # piecewise linear with sparse breakpoints (plain Adam on an l1
                # term never reaches zero, so refinement blurs changepoints)
                with torch.no_grad():
                    thresh = cfg['w_prox'] * cfg['lr_coef']
                    model.coef[1:] = torch.sign(model.coef[1:]) * (
                        model.coef[1:].abs() - thresh
                    ).clamp(min=0)
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
        else:
            with torch.no_grad():
                sigma = (model() - y).T.std(dim=0).cpu().numpy()
        for p in model.parameters():
            p.requires_grad_(True)

        gated = _gate_trendless_series(model, y, cfg['min_trend_to_noise'])

        diag = model.diagnostics()
        info = {
            'gated_series': gated,
            'identification': ident,
            'history': history,
            'stage_a_val': best_val if np.isfinite(best_val) else None,
            'stage_b_loss': stage_b_loss,
            'sigma': np.asarray(sigma, dtype=float),
            'diagnostics': diag,
            'n_factors_fit': model.K,
            'n_factors_live': diag['n_live_factors'],
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

else:  # pragma: no cover - torch-free environments

    class LatentFactorTrend:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "trend_network='factor' requires torch. Use trend_network='none'."
            )

    def _gate_trendless_series(model, y: 'torch.Tensor', min_ratio: float) -> list:
        """Zero the loadings of series that have no low-frequency structure.

        Args:
            model: fitted LatentFactorTrend (mutated in place).
            y: (N, T) normalized panel the model was fit to.
            min_ratio: minimum smoothed-to-residual spread ratio to keep
                loadings.

        Returns:
            Indices of the series whose loadings were zeroed.
        """
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
