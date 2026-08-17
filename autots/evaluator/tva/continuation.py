# -*- coding: utf-8 -*-
"""
Validation-selected factor continuation (plan item 1a).

Replaces the single damped local-linear extrapolation rule with a candidate
set (constant, damped, ridge_ar, regime) chosen per factor by held-out
reconstruction error over inner origins, never in-sample fit — an in-sample
criterion always prefers the wiggliest continuation.

Pure numpy: nothing here imports torch, so the candidate maths is testable
without fitting a model.
"""

from __future__ import annotations

import numpy as np


DEFAULT_CONTINUATION_CONFIG = {
    'slope_windows': (28, 56, 90, 180),
    'damping': (0.0, 0.5, 0.9, 0.99, 1.0),
    'include_constant': True,
    'include_model': True,
    'include_ridge_ar': True,
    'include_regime_slope': True,
    'ridge_ar_lags': 5,
    'ridge_alpha': 1.0,
    'ridge_min_history': 120,
    # younger regimes damp harder: phi_eff = phi * (1 - exp(-age / halflife))
    'regime_damp_halflife': 90.0,
    'regime_min_age': 7,
    'selection_passes': 2,
    'selection_tolerance': 0.01,
}


# ---------------------------------------------------------------------------
# Candidate specs
# ---------------------------------------------------------------------------


def build_specs(config: dict = None) -> list:
    """Enumerate candidate continuation specs.

    Order is deliberate: model's own rule first, then the constant null, so a
    tie during selection resolves toward the incumbent and then toward doing
    nothing.

    Returns:
        List of dicts, each with a ``kind`` and a stable ``name``.
    """
    cfg = dict(DEFAULT_CONTINUATION_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})

    specs = []
    if cfg['include_model']:
        specs.append({'kind': 'model', 'name': 'model'})
    if cfg['include_constant']:
        specs.append({'kind': 'constant', 'name': 'constant'})
    for window in cfg['slope_windows']:
        for phi in cfg['damping']:
            specs.append(
                {
                    'kind': 'damped',
                    'window': int(window),
                    'phi': float(phi),
                    'name': f'damped_w{int(window)}_p{float(phi):g}',
                }
            )
    if cfg['include_ridge_ar']:
        specs.append(
            {
                'kind': 'ridge_ar',
                'lags': int(cfg['ridge_ar_lags']),
                'alpha': float(cfg['ridge_alpha']),
                'min_history': int(cfg['ridge_min_history']),
                'name': f"ridge_ar_l{int(cfg['ridge_ar_lags'])}",
            }
        )
    if cfg['include_regime_slope']:
        # raw regime slope and the young-regime-trust variant as separate
        # candidates, so validation decides whether the modifier earns its place
        specs.append(
            {
                'kind': 'regime', 'phi': 1.0, 'trust': False,
                'halflife': float(cfg['regime_damp_halflife']),
                'min_age': int(cfg['regime_min_age']), 'name': 'regime',
            }
        )
        specs.append(
            {
                'kind': 'regime', 'phi': 1.0, 'trust': True,
                'halflife': float(cfg['regime_damp_halflife']),
                'min_age': int(cfg['regime_min_age']), 'name': 'regime_trust',
            }
        )
        specs.append(
            {
                'kind': 'regime', 'phi': 0.9, 'trust': False,
                'halflife': float(cfg['regime_damp_halflife']),
                'min_age': int(cfg['regime_min_age']), 'name': 'regime_p0.9',
            }
        )
    return specs


# ---------------------------------------------------------------------------
# Continuation primitives
# ---------------------------------------------------------------------------


def _damping_profile(phi: float, horizon: int) -> np.ndarray:
    """Cumulative damping weights ``sum_{i<=h} phi^i`` for h = 1..H.

    Matches the torch model's ``cumsum(phi ** steps)`` so a selected
    ``damped`` candidate is directly comparable to the incumbent.
    """
    steps = np.arange(1, max(int(horizon), 1) + 1, dtype=float)
    return np.cumsum(np.power(float(phi), steps))


def _ols_slope(path: np.ndarray, origin: int, window: int) -> float:
    """Least-squares slope per step over the trailing ``window`` at ``origin``."""
    lo = max(int(origin) - int(window) + 1, 0)
    segment = np.asarray(path[lo : int(origin) + 1], dtype=float)
    segment = segment[np.isfinite(segment)]
    if segment.size < 2:
        return 0.0
    t = np.arange(segment.size, dtype=float)
    t = t - t.mean()
    denom = float((t**2).sum())
    if denom <= 1e-12:
        return 0.0
    return float((segment * t).sum() / denom)


def _ridge_ar_path(
    path: np.ndarray, origin: int, horizon: int, lags: int, alpha: float,
    min_history: int,
) -> np.ndarray:
    """Recursive ridge-AR continuation on factor first differences.

    Fits ``d_t ~ d_{t-1..t-lags}`` by ridge, then iterates forward
    accumulating predicted increments. Falls back to a constant continuation
    if the fit is unstable (companion spectral radius >= 1) or history is short.
    """
    horizon = max(int(horizon), 1)
    history = np.asarray(path[: int(origin) + 1], dtype=float)
    history = history[np.isfinite(history)]
    lags = max(int(lags), 1)
    if history.size < max(int(min_history), lags + 5):
        return np.zeros(horizon, dtype=float)
    diffs = np.diff(history)
    if diffs.size < lags + 5:
        return np.zeros(horizon, dtype=float)

    rows = diffs.size - lags
    design = np.column_stack(
        [diffs[lags - j - 1 : lags - j - 1 + rows] for j in range(lags)]
    )
    target = diffs[lags:]
    gram = design.T @ design + float(alpha) * np.eye(lags)
    try:
        coef = np.linalg.solve(gram, design.T @ target)
    except np.linalg.LinAlgError:
        return np.zeros(horizon, dtype=float)
    if not np.all(np.isfinite(coef)):
        return np.zeros(horizon, dtype=float)

    # stability guard against an explosive AR producing a runaway extrapolation
    companion = np.zeros((lags, lags))
    companion[0] = coef
    if lags > 1:
        companion[1:, :-1] = np.eye(lags - 1)
    try:
        radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    except np.linalg.LinAlgError:
        return np.zeros(horizon, dtype=float)
    if not np.isfinite(radius) or radius >= 1.0:
        return np.zeros(horizon, dtype=float)

    window = list(diffs[-lags:][::-1])  # most recent first
    out = np.empty(horizon, dtype=float)
    running = 0.0
    for h in range(horizon):
        step = float(np.dot(coef, window))
        running += step
        out[h] = running
        window = [step] + window[:-1]
    return out


def _last_active_knot(coef_k: np.ndarray, knot_times: np.ndarray, origin: int,
                      tol: float = 1e-8) -> int:
    """Time index of the most recent knot the trend filter actually used.

    ``coef_k[0]`` is the global linear term; entries 1.. correspond to
    ``knot_times``. A nonzero entry is a slope change the l1 solve chose to pay
    for, i.e. a changepoint the model believes in.
    """
    if coef_k is None or knot_times is None or len(knot_times) == 0:
        return 0
    hinge = np.asarray(coef_k, dtype=float)[1 : len(knot_times) + 1]
    if hinge.size == 0:
        return 0
    scale = np.max(np.abs(hinge))
    if not np.isfinite(scale) or scale <= 0:
        return 0
    active = np.asarray(knot_times)[np.abs(hinge) > max(tol, scale * 1e-3)]
    active = active[active <= int(origin)]
    return int(active[-1]) if active.size else 0


def continue_factor(
    path: np.ndarray,
    origins: np.ndarray,
    horizon: int,
    spec: dict,
    knot_times: np.ndarray = None,
    coef_k: np.ndarray = None,
) -> np.ndarray:
    """Future *deltas* of one factor, relative to its value at each origin.

    Deltas (not levels) so every candidate is anchored identically — an
    anchoring difference would otherwise masquerade as a slope difference
    during selection.

    Returns:
        (O, H) array of deltas.
    """
    path = np.asarray(path, dtype=float)
    origins = np.atleast_1d(np.asarray(origins, dtype=int))
    horizon = max(int(horizon), 1)
    out = np.zeros((origins.size, horizon), dtype=float)
    kind = spec['kind']

    if kind == 'constant':
        return out

    if kind == 'damped':
        profile = _damping_profile(spec['phi'], horizon)
        for i, origin in enumerate(origins):
            out[i] = _ols_slope(path, origin, spec['window']) * profile
        return out

    if kind == 'ridge_ar':
        for i, origin in enumerate(origins):
            out[i] = _ridge_ar_path(
                path, origin, horizon, spec['lags'], spec['alpha'],
                spec['min_history'],
            )
        return out

    if kind == 'regime':
        for i, origin in enumerate(origins):
            knot = _last_active_knot(coef_k, knot_times, origin)
            age = int(origin) - int(knot) + 1
            if age < int(spec['min_age']):
                # too young to estimate a slope from: stay put
                continue
            slope = _ols_slope(path, origin, age)
            phi_eff = float(np.clip(spec['phi'], 0.0, 1.0))
            if spec.get('trust'):
                halflife = float(spec['halflife'])
                trust = 1.0 - np.exp(-age / halflife) if halflife > 0 else 1.0
                phi_eff = float(np.clip(phi_eff * trust, 0.0, 1.0))
            out[i] = slope * _damping_profile(phi_eff, horizon)
        return out

    raise ValueError(f"unknown continuation kind: {kind!r}")


def candidate_futures(
    paths: np.ndarray,
    origins: np.ndarray,
    horizon: int,
    specs: list,
    knot_times: np.ndarray = None,
    coef: np.ndarray = None,
    model_deltas: np.ndarray = None,
) -> dict:
    """{(factor_index, spec_name): (O, H) deltas} for every candidate.

    ``model_deltas`` is the incumbent extrapolator's (O, H, K) deltas (lives in
    the torch model, so supplied by the caller); backs the ``model`` spec.
    """
    paths = np.asarray(paths, dtype=float)
    K = paths.shape[1]
    out = {}
    for k in range(K):
        coef_k = None
        if coef is not None:
            coef_arr = np.asarray(coef, dtype=float)
            coef_k = coef_arr[:, k] if coef_arr.ndim == 2 else coef_arr
        for spec in specs:
            if spec['kind'] == 'model':
                if model_deltas is None:
                    continue
                out[(k, spec['name'])] = np.asarray(
                    model_deltas, dtype=float
                )[:, :, k]
            else:
                out[(k, spec['name'])] = continue_factor(
                    paths[:, k], origins, horizon, spec,
                    knot_times=knot_times, coef_k=coef_k,
                )
    return out


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_continuations(
    paths: np.ndarray,
    origins: np.ndarray,
    horizon: int,
    score_fn,
    specs: list = None,
    knot_times: np.ndarray = None,
    coef: np.ndarray = None,
    model_deltas: np.ndarray = None,
    config: dict = None,
) -> dict:
    """Choose a continuation per factor by held-out reconstruction error.

    Coordinate descent: factors are not independent once loadings mix them, so
    each factor's candidate is scored with the others held at their current
    choice, and the sweep repeated (default 2 passes; the objective is
    monotone non-increasing by construction).

    Args:
        score_fn: callable taking (O, H, K) future deltas, returning a scalar
            loss (lower is better) — reconstruction into series space, lag
            weights, loadings and MASE scaling happen here, deliberately
            outside this module so the selection logic stays pure.

    Returns:
        dict with ``choice`` ({factor_index: spec_name}), ``score`` (final
        loss), ``baseline_score`` (the incumbent's loss) and ``scores``
        (per-factor per-candidate losses from the final pass).
    """
    cfg = dict(DEFAULT_CONTINUATION_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    if specs is None:
        specs = build_specs(cfg)

    paths = np.asarray(paths, dtype=float)
    origins = np.atleast_1d(np.asarray(origins, dtype=int))
    K = paths.shape[1]
    horizon = max(int(horizon), 1)

    futures = candidate_futures(
        paths, origins, horizon, specs,
        knot_times=knot_times, coef=coef, model_deltas=model_deltas,
    )
    available = [
        spec['name'] for spec in specs if any((k, spec['name']) in futures for k in range(K))
    ]
    if not available:
        return {'choice': {}, 'score': np.nan, 'baseline_score': np.nan, 'scores': {}}

    default_name = 'model' if 'model' in available else available[0]
    choice = {k: default_name for k in range(K)}

    def assemble(current: dict) -> np.ndarray:
        stacked = np.zeros((origins.size, horizon, K), dtype=float)
        for k in range(K):
            key = (k, current[k])
            if key in futures:
                stacked[:, :, k] = futures[key]
        return stacked

    def evaluate(current: dict) -> float:
        try:
            value = float(score_fn(assemble(current)))
        except Exception:
            return np.inf
        return value if np.isfinite(value) else np.inf

    baseline_score = evaluate(choice)
    best_score = baseline_score
    scores = {}
    tolerance = float(cfg['selection_tolerance'])

    for _ in range(max(int(cfg['selection_passes']), 1)):
        moved = False
        for k in range(K):
            incumbent = choice[k]
            local = {}
            for name in available:
                if (k, name) not in futures:
                    continue
                trial = dict(choice)
                trial[k] = name
                local[name] = evaluate(trial)
            scores[k] = local
            if not local:
                continue
            finite = {n: s for n, s in local.items() if np.isfinite(s)}
            if not finite:
                continue
            best_name = min(finite, key=finite.get)
            best_local = finite[best_name]
            # an unmeasurable improvement over the incumbent is not an improvement
            incumbent_score = finite.get(incumbent, np.inf)
            if (
                np.isfinite(incumbent_score)
                and best_local >= incumbent_score * (1.0 - tolerance)
            ):
                best_name, best_local = incumbent, incumbent_score
            if best_name != incumbent:
                choice[k] = best_name
                moved = True
            best_score = min(best_score, best_local)
        if not moved:
            break

    final_score = evaluate(choice)
    return {
        'choice': {int(k): v for k, v in choice.items()},
        'score': float(final_score) if np.isfinite(final_score) else None,
        'baseline_score': (
            float(baseline_score) if np.isfinite(baseline_score) else None
        ),
        'scores': {
            int(k): {n: (float(s) if np.isfinite(s) else None) for n, s in v.items()}
            for k, v in scores.items()
        },
    }


def apply_choice(
    paths: np.ndarray,
    origins: np.ndarray,
    horizon: int,
    choice: dict,
    specs: list = None,
    knot_times: np.ndarray = None,
    coef: np.ndarray = None,
    model_deltas: np.ndarray = None,
    config: dict = None,
) -> np.ndarray:
    """(O, H, K) future deltas for an already-selected per-factor choice."""
    cfg = dict(DEFAULT_CONTINUATION_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    if specs is None:
        specs = build_specs(cfg)
    by_name = {spec['name']: spec for spec in specs}
    paths = np.asarray(paths, dtype=float)
    origins = np.atleast_1d(np.asarray(origins, dtype=int))
    horizon = max(int(horizon), 1)
    K = paths.shape[1]

    out = np.zeros((origins.size, horizon, K), dtype=float)
    for k in range(K):
        name = choice.get(k, choice.get(str(k), 'model'))
        spec = by_name.get(name)
        if spec is None or spec['kind'] == 'model':
            if model_deltas is not None:
                out[:, :, k] = np.asarray(model_deltas, dtype=float)[:, :, k]
            continue
        coef_k = None
        if coef is not None:
            coef_arr = np.asarray(coef, dtype=float)
            coef_k = coef_arr[:, k] if coef_arr.ndim == 2 else coef_arr
        out[:, :, k] = continue_factor(
            paths[:, k], origins, horizon, spec,
            knot_times=knot_times, coef_k=coef_k,
        )
    return out
