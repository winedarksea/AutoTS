# -*- coding: utf-8 -*-
"""
Per-series seasonal-path arbitration.

Chooses per series, by held-out error, among: datepart (the panel-wide
incumbent, kept on ties since only it expresses holiday/calendar structure),
empirical (tile the series' own recent detrended cycles), and amplitude
(datepart path rescaled by a ridge-estimated per-series scalar).
"""

from __future__ import annotations

import numpy as np


DEFAULT_SEASONAL_CONFIG = {
    'season_m': 7,
    'yearly_m': 365,
    'n_cycles': 2,
    'min_years_for_yearly': 2,
    # margin the incumbent must be beaten by to be replaced; a small edge on
    # one held-out window is within measurement noise
    'selection_margin': 0.03,
    'amplitude_ridge': 0.25,
    'amplitude_clip': (0.25, 2.5),
    'amplitude_window_cycles': 2,
}


def empirical_profile(residual: np.ndarray, season_m: int, n_cycles: int = 2):
    """(season_m, N) average of the last ``n_cycles`` detrended cycles.

    Args:
        residual: (T, N) panel with trend and level shifts already removed.
        season_m: cycle length in steps.
        n_cycles: how many trailing cycles to average.

    Returns:
        (season_m, N) profile aligned so row 0 is the phase immediately
        following the end of ``residual``, or None when history is too short.
        Centered per series: a nonzero-mean cycle would double-count level,
        which the trend term already owns.
    """
    residual = np.asarray(residual, dtype=float)
    m = max(int(season_m), 1)
    cycles = max(int(n_cycles), 1)
    if residual.shape[0] < m:
        return None
    usable = min(cycles, residual.shape[0] // m)
    if usable < 1:
        return None
    tail = residual[-usable * m :]
    stacked = tail.reshape(usable, m, residual.shape[1])
    with np.errstate(invalid='ignore'):
        profile = np.nanmean(stacked, axis=0)
    profile = np.where(np.isfinite(profile), profile, 0.0)
    profile = profile - np.nanmean(profile, axis=0, keepdims=True)
    return np.where(np.isfinite(profile), profile, 0.0)


def tile_profile(profile: np.ndarray, horizon: int) -> np.ndarray:
    """(H, N) forward tiling of a phase-aligned cycle profile."""
    if profile is None:
        return None
    profile = np.asarray(profile, dtype=float)
    m = profile.shape[0]
    if m < 1:
        return None
    idx = np.arange(max(int(horizon), 1)) % m
    return profile[idx]


def amplitude_scale(
    fitted: np.ndarray, residual: np.ndarray, config: dict = None
) -> np.ndarray:
    """(N,) per-series multiplier on the fitted seasonal path.

    Ridge projection of recent detrended residual onto the fitted path,
    shrunk toward 1 (not 0) and clipped — a weak seasonal signal should keep
    the model's estimate rather than collapse to no seasonality.
    """
    cfg = dict(DEFAULT_SEASONAL_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    fitted = np.asarray(fitted, dtype=float)
    residual = np.asarray(residual, dtype=float)
    n = fitted.shape[1]
    alpha = np.ones(n, dtype=float)
    if fitted.shape != residual.shape or fitted.shape[0] < 2:
        return alpha
    ridge = float(cfg['amplitude_ridge'])
    lo, hi = cfg['amplitude_clip']
    for i in range(n):
        x, y = fitted[:, i], residual[:, i]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        xx = float(np.dot(x[ok], x[ok]))
        if not np.isfinite(xx) or xx <= 1e-12:
            continue
        xy = float(np.dot(x[ok], y[ok]))
        # ridge toward 1: (x'y + ridge*xx) / (x'x + ridge*xx)
        alpha[i] = (xy + ridge * xx) / (xx + ridge * xx)
    alpha = np.where(np.isfinite(alpha), alpha, 1.0)
    return np.clip(alpha, float(lo), float(hi))


def select_seasonal_paths(
    candidates: dict,
    actual: np.ndarray,
    scale: np.ndarray = None,
    default: str = 'datepart',
    config: dict = None,
) -> dict:
    """Choose a seasonal path per series from held-out error.

    Args:
        candidates: {name: (H, N) full forecast for the held-out window},
            each already summed with the same trend/holiday/shift terms so the
            only difference between them is the seasonal path.
        actual: (H, N) realized values over that window.
        scale: (N,) MASE denominator; ``None`` scores in raw units.
        default: the incumbent's name -- kept unless beaten by more than the
            selection margin.
        config: overrides for ``DEFAULT_SEASONAL_CONFIG``.

    Returns:
        dict with ``choice`` ({series_index: name}), ``scores``
        ({name: (N,) scaled MAE}) and ``n_changed``.
    """
    cfg = dict(DEFAULT_SEASONAL_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    actual = np.asarray(actual, dtype=float)
    n = actual.shape[1]
    names = [k for k, v in candidates.items() if v is not None]
    if not names:
        return {'choice': {}, 'scores': {}, 'n_changed': 0}
    if scale is None:
        scale = np.ones(n, dtype=float)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, np.nan)

    scores = {}
    for name in names:
        pred = np.asarray(candidates[name], dtype=float)
        if pred.shape != actual.shape:
            continue
        with np.errstate(invalid='ignore'):
            scores[name] = np.nanmean(np.abs(pred - actual), axis=0) / scale
    if default not in scores:
        default = next(iter(scores)) if scores else default

    margin = float(cfg['selection_margin'])
    choice, changed = {}, 0
    for i in range(n):
        base = scores.get(default, np.full(n, np.nan))[i]
        best_name, best = default, base
        for name, values in scores.items():
            value = values[i]
            if not np.isfinite(value):
                continue
            if not np.isfinite(best) or value < best:
                best_name, best = name, value
        # replace the incumbent only on a materially better score
        if (
            best_name != default
            and np.isfinite(base)
            and not (best < base * (1.0 - margin))
        ):
            best_name = default
        choice[i] = best_name
        if best_name != default:
            changed += 1
    return {
        'choice': choice,
        'scores': {k: v.tolist() for k, v in scores.items()},
        'n_changed': int(changed),
        'default': default,
    }


def assemble_choice(candidates: dict, choice: dict, default: str) -> np.ndarray:
    """(H, N) seasonal path assembled column-by-column from a selection."""
    base = candidates.get(default)
    if base is None:
        base = next(v for v in candidates.values() if v is not None)
    out = np.array(base, dtype=float, copy=True)
    for i, name in choice.items():
        source = candidates.get(name)
        if source is not None and name != default:
            out[:, int(i)] = np.asarray(source, dtype=float)[:, int(i)]
    return out
