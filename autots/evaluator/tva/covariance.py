# -*- coding: utf-8 -*-
"""Forecast covariance (Sigma) assembly for TVA.

One cross-series forecast-error covariance object, built where the residuals
already exist, serving three callers that otherwise each invent their own
weighting:

* MinT reconciliation, which currently falls back to ``W = I`` in the default
  forecasting modes because no residual matrix is available there;
* the closed-form what-if solver in :mod:`autots.evaluator.tva.scenario`,
  whose minimum-disruption update is a Gaussian conditioning step under Sigma;
* interval coherence (later — nothing here consumes it yet).

The estimator is a shrinkage blend of an empirical covariance and a low-rank
structural target::

    Sigma = (1 - alpha) * Sigma_emp + alpha * Sigma_struct
    Sigma_struct = beta * (Lambda Sigma_f Lambda') + diag(psi)

with ``alpha`` the Ledoit-Wolf shrinkage intensity of the residual matrix.
``beta`` is a single least-squares scalar fitted on the off-diagonals, which
is what makes the same assembly usable with loadings that do not live in the
residuals' units (the ``trend_network='none'`` path takes its loadings from
structure discovery, where they are in standardized-difference units). In
``'factor'`` mode Lambda is already in the residual metric and beta lands
near 1, reducing the expression to the plain ``Lambda Sigma_f Lambda' +
diag(psi)`` factor-model target. ``psi`` is then set residually, so
``diag(Sigma_struct) == diag(Sigma_emp)`` exactly and the blend only ever
changes the off-diagonal (correlation) structure.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'ledoit_wolf_shrinkage',
    'damped_accumulated_variance',
    'structural_target',
    'apply_variance_floor',
    'nearest_psd',
    'assemble_covariance',
]


def ledoit_wolf_shrinkage(X: np.ndarray) -> tuple:
    """Ledoit-Wolf shrunk covariance and the shrinkage intensity that made it.

    Reproduces :func:`autots.tools.hierarchial.ledoit_wolf_covariance` (there
    is a test asserting they agree) while also returning ``gamma``, which that
    function computes and discards. The intensity is the piece this module
    actually needs: it is a data-driven statement about how badly conditioned
    the sample covariance is, which is exactly the question "how far should we
    lean on a structural target instead".

    Args:
        X: (n_samples, n_features) residual matrix.

    Returns:
        ``(cov, gamma)`` — the (n_features, n_features) shrunk covariance and
        the shrinkage intensity in [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    if n_samples < 2:
        return np.eye(n_features), 1.0

    Xc = X - X.mean(axis=0, keepdims=True)
    emp_cov = (Xc.T @ Xc) / (n_samples - 1)
    mu = np.trace(emp_cov) / n_features

    # beta = mean over samples of ||x x' - emp_cov||_F^2, expanded so the
    # per-sample outer products never have to be formed (the reference
    # implementation loops; this is the same quantity in closed form).
    sq_norms = np.einsum('ij,ij->i', Xc, Xc)  # ||x_i||^2
    sum_x4 = float(np.sum(sq_norms**2))  # sum_i ||x_i x_i'||_F^2
    cross = float(np.einsum('ij,jk,ik->', Xc, emp_cov, Xc))  # sum_i x_i' C x_i
    frob_c = float((emp_cov * emp_cov).sum())
    beta = (sum_x4 - 2.0 * cross + n_samples * frob_c) / (n_samples - 1.0) ** 2

    diff = emp_cov - mu * np.eye(n_features)
    denom = float((diff * diff).sum())
    gamma = 0.0 if denom <= 0 else beta / denom
    gamma = float(min(max(gamma, 0.0), 1.0))

    cov = (1.0 - gamma) * emp_cov + gamma * mu * np.eye(n_features)
    return cov, gamma


def damped_accumulated_variance(phi: np.ndarray, horizon: int) -> np.ndarray:
    """Per-factor forecast-error variance of a damped random walk, horizon-averaged.

    A factor continued with damping ``phi`` accumulates unit innovations as
    ``e_h = sum_{j=1..h} phi^(j-1) eps_j``, so ``Var(e_h) = (1 - phi^(2h)) /
    (1 - phi^2)``. The residual matrix this is compared against pools every
    horizon step 1..H into one sample set, so the matching summary is the mean
    over ``h``, not the terminal value.

    Args:
        phi: (K,) damping coefficients in [0, 1).
        horizon: forecast horizon H.

    Returns:
        (K,) mean-over-horizon accumulated variance per factor.
    """
    phi = np.clip(np.asarray(phi, dtype=np.float64).ravel(), 0.0, 0.999999)
    H = max(int(horizon), 1)
    h = np.arange(1, H + 1, dtype=np.float64)[:, None]  # (H, 1)
    p2 = (phi**2)[None, :]  # (1, K)
    var_h = (1.0 - p2**h) / np.maximum(1.0 - p2, 1e-12)
    return var_h.mean(axis=0)


def structural_target(
    sigma_emp: np.ndarray, loadings: np.ndarray, factor_var: np.ndarray
) -> tuple:
    """Low-rank structural covariance target matched to ``sigma_emp``'s diagonal.

    Args:
        sigma_emp: (N, N) empirical covariance.
        loadings: (N, K) factor loadings, any consistent scaling.
        factor_var: (K,) factor forecast-error variances.

    Returns:
        ``(sigma_struct, beta, psi)``.
    """
    N = sigma_emp.shape[0]
    Lam = np.asarray(loadings, dtype=np.float64)
    if Lam.ndim != 2 or Lam.shape[0] != N or Lam.shape[1] == 0:
        raise ValueError(
            f"loadings must be (N, K) with N={N}; got {np.shape(loadings)}"
        )
    v = np.asarray(factor_var, dtype=np.float64).ravel()
    if v.shape[0] != Lam.shape[1]:
        raise ValueError(
            f"factor_var has {v.shape[0]} entries for {Lam.shape[1]} factors"
        )
    B = (Lam * v[None, :]) @ Lam.T  # (N, N), PSD by construction

    # beta: least-squares scale of the common block against the empirical
    # off-diagonals. Fitting on the off-diagonals only is what keeps beta a
    # statement about correlation structure rather than about variance level
    # (the diagonal is handled exactly by psi below).
    off = ~np.eye(N, dtype=bool)
    if N > 1:
        num = float(np.sum(B[off] * sigma_emp[off]))
        den = float(np.sum(B[off] * B[off]))
        beta = num / den if den > 1e-300 else 0.0
    else:
        beta = 0.0
    beta = max(beta, 0.0)

    psi = np.maximum(np.diag(sigma_emp) - beta * np.diag(B), 0.0)
    sigma_struct = beta * B + np.diag(psi)
    return sigma_struct, float(beta), psi


def apply_variance_floor(sigma: np.ndarray, floor_sd: np.ndarray) -> tuple:
    """Raise ``diag(sigma)`` to ``floor_sd**2`` without distorting correlations.

    The idiosyncratic term in the factor model is capped at 2 degrees of
    freedom by design, which is why ``factor_variance_share`` reads 1.00 on
    every panel including factor-free ones: the model's own residual spread is
    structurally over-confident. Floor the variances at the same sigma the
    shipped prediction intervals use, rescaling rows and columns so the
    correlation matrix is untouched and PSD-ness is preserved.

    Args:
        sigma: (N, N) covariance.
        floor_sd: (N,) minimum standard deviation per series.

    Returns:
        ``(sigma_floored, binding)`` where ``binding`` is an (N,) bool array
        marking the series whose variance the floor actually raised.
    """
    sd = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    floor_sd = np.maximum(np.nan_to_num(np.asarray(floor_sd, dtype=np.float64), nan=0.0), 0.0)
    binding = floor_sd > sd
    if not np.any(binding):
        return sigma, binding
    target = np.where(binding, floor_sd, sd)
    ratio = np.divide(target, sd, out=np.ones_like(sd), where=sd > 0)
    out = sigma * ratio[:, None] * ratio[None, :]
    # series with zero empirical spread have no correlation to preserve; give
    # them the floor as pure variance rather than a zero row
    dead = sd <= 0
    if np.any(dead):
        out[dead, :] = 0.0
        out[:, dead] = 0.0
        out[dead, dead] = floor_sd[dead] ** 2
    return out, binding


def nearest_psd(sigma: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Symmetrize, clip negative eigenvalues to zero, add a relative ridge."""
    S = np.asarray(sigma, dtype=np.float64)
    S = 0.5 * (S + S.T)
    try:
        w, V = np.linalg.eigh(S)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return S + np.eye(S.shape[0]) * ridge
    if np.any(w < 0):
        S = (V * np.maximum(w, 0.0)) @ V.T
        S = 0.5 * (S + S.T)
    trace = float(np.trace(S))
    if trace > 0:
        S = S + np.eye(S.shape[0]) * (ridge * trace / S.shape[0])
    return S


def assemble_covariance(
    residuals: np.ndarray,
    loadings: np.ndarray = None,
    factor_var: np.ndarray = None,
    scale: np.ndarray = None,
    jacobian: np.ndarray = None,
    floor_sd: np.ndarray = None,
) -> tuple:
    """Assemble the forecast covariance from a residual matrix and a factor model.

    Args:
        residuals: (n_samples, N) rolling-origin forecast residuals, in the
            space the model was fit in (normalized, for ``'factor'`` mode).
        loadings: optional (N, K) factor loadings for the structural target.
            Without them the estimate is the shrunk empirical covariance alone.
        factor_var: optional (K,) factor forecast-error variances; required
            when ``loadings`` is given.
        scale: optional (N,) per-series scale mapping the model space back to
            raw units.
        jacobian: optional (N,) additional per-series derivative, used for the
            signed-log1p modeling space (``d level / d log-space``).
        floor_sd: optional (N,) raw-unit standard-deviation floor.

    Returns:
        ``(sigma, info)`` — the (N, N) raw-unit covariance and a diagnostics
        dict with ``alpha``, ``beta``, ``psi``, ``n_samples``, ``floor_binding``
        and ``has_structure``.
    """
    R = np.asarray(residuals, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] < 2 or R.shape[1] == 0:
        raise ValueError(
            f"residuals must be (n_samples>=2, N>=1); got {R.shape}"
        )
    R = R[np.isfinite(R).all(axis=1)]
    if R.shape[0] < 2:
        raise ValueError("residual matrix has fewer than 2 complete rows")
    N = R.shape[1]

    sigma_emp, alpha = ledoit_wolf_shrinkage(R)

    info = {
        'n_samples': int(R.shape[0]),
        'alpha': float(alpha),
        'beta': None,
        'psi': None,
        'has_structure': False,
    }

    sigma = sigma_emp
    if loadings is not None and np.size(loadings):
        if factor_var is None:
            raise ValueError("factor_var is required when loadings are supplied")
        sigma_struct, beta, psi = structural_target(sigma_emp, loadings, factor_var)
        sigma = (1.0 - alpha) * sigma_emp + alpha * sigma_struct
        info.update({'beta': beta, 'psi': psi, 'has_structure': True})

    # back to raw units: model scale first, then the modeling-space Jacobian
    if scale is not None:
        s = np.asarray(scale, dtype=np.float64).ravel()
        sigma = sigma * s[:, None] * s[None, :]
    if jacobian is not None:
        j = np.asarray(jacobian, dtype=np.float64).ravel()
        sigma = sigma * j[:, None] * j[None, :]

    sigma = nearest_psd(sigma)

    if floor_sd is not None:
        sigma, binding = apply_variance_floor(sigma, floor_sd)
        info['floor_binding'] = binding
    else:
        info['floor_binding'] = np.zeros(N, dtype=bool)

    return nearest_psd(sigma), info
