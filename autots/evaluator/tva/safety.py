# -*- coding: utf-8 -*-
"""Safety floor for the TVA factor trend mode.

``trend_network='factor'`` extrapolates a whole-history latent trend, which is
prone to integrating a small slope error into a large level error at long
horizons. Four independent, optional corrections — blend toward
SeasonalNaive, cap the extrapolation, re-anchor the level, calibrate sigma by
horizon bucket — each selected by inner rolling-origin validation rather than
assumed.

Every identity path (blend weight 1.0, alpha 0.0, no cap, bucket_scales=None)
is bitwise exact, so wiring this module in cannot perturb existing results
until a selector actually fires. Nothing here raises; degenerate input
returns the safe default. Pure numpy/pandas — no torch, no sklearn.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SAFETY_CONFIG = {
    'blend_grid': (0.0, 0.25, 0.5, 0.75, 1.0),  # weight on the TVA forecast
    'blend_tie_tolerance': 0.01,  # ties prefer smaller TVA weight
    'error_cap_mult': 3.0,
    'error_cap_quantile': 0.99,
    'horizon_buckets': ((1, 28), (29, 90), (91, 180)),
    'blend_risk_weight': 0.0,
    'reanchor_alphas': (0.0, 0.5, 1.0),
    'reanchor_window': 14,
}


def _merged_config(config=None) -> dict:
    """Caller config layered over the defaults (callers may pass partials)."""
    merged = dict(DEFAULT_SAFETY_CONFIG)
    if config:
        merged.update({k: v for k, v in dict(config).items() if v is not None})
    return merged


def _as_2d_list(folds) -> list:
    """Normalize a fold container to a list of 2-D float arrays.

    Accepts a list/tuple of (H, N) arrays, a single (H, N) array, or a stacked
    (F, H, N) array.
    """
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
        if cur.ndim != 2 or cur.size == 0:
            continue
        out.append(cur)
    return out


def _safe_scale(scale, n_series: int) -> np.ndarray:
    """MASE denominators with non-positive / non-finite entries neutralized.

    A zero scale would make every candidate score ``inf`` and the argmin
    arbitrary; substituting 1.0 keeps candidates comparable.
    """
    arr = np.asarray(scale, dtype=float).ravel()
    if arr.size == 0:
        arr = np.ones(n_series, dtype=float)
    elif arr.size == 1 and n_series != 1:
        arr = np.repeat(arr, n_series)
    if arr.size < n_series:
        arr = np.concatenate([arr, np.ones(n_series - arr.size)])
    arr = arr[:n_series].astype(float, copy=True)
    bad = ~np.isfinite(arr) | (arr <= 0)
    arr[bad] = 1.0
    return arr


def _pooled_scaled_mae(errors: list, scale: np.ndarray, n_series: int) -> np.ndarray:
    """Per-series MAE pooled over folds and horizon steps, divided by ``scale``.

    Pooling (vs. averaging per-fold means) weights every (origin, step) pair
    equally so a shorter fold can't dominate.

    Returns:
        (N,) array; ``nan`` where a series had no finite error anywhere.
    """
    if not errors:
        return np.full(n_series, np.nan)
    stacked = np.concatenate([e.reshape(-1, e.shape[-1]) for e in errors], axis=0)
    with np.errstate(invalid='ignore'):
        finite = np.isfinite(stacked)
        counts = finite.sum(axis=0)
        totals = np.where(finite, np.abs(stacked), 0.0).sum(axis=0)
    out = np.full(n_series, np.nan)
    ok = counts > 0
    out[ok] = totals[ok] / counts[ok] / scale[ok]
    return out


def _argmin_prefer_small(scores: np.ndarray, grid: np.ndarray, tol: float) -> float:
    """Smallest grid value whose score is within ``tol`` (relative) of the best.

    Ties and near-ties break toward the smaller value: less weight on the
    extrapolating model means less risk on the unseen future.

    Returns:
        The selected grid value, or ``nan`` if no score was finite.
    """
    finite = np.isfinite(scores)
    if not np.any(finite):
        return float('nan')
    best = float(np.min(scores[finite]))
    threshold = best + abs(best) * float(tol)
    # +epsilon handles best == 0, where the relative band collapses to a point
    acceptable = finite & (scores <= threshold + 1e-15)
    return float(np.min(grid[acceptable]))


def seasonal_naive_forecast(history, horizon: int, season_m: int = 7) -> np.ndarray:
    """Tile the last season of observations forward.

    The baseline the safety floor blends toward and caps around: no
    parameters to overfit and no slope to integrate, so error stays bounded
    by the seasonal pattern rather than growing with h. A missing seasonal
    slot falls back to the last finite value in that column (0.0 if none).

    Args:
        history: (T, N) observed panel, most recent observation last.
        horizon: number of steps to produce (H).
        season_m: seasonal period, in steps. Values < 1 are treated as 1
            (last-value persistence).

    Returns:
        (H, N) float array. Empty history returns zeros.
    """
    horizon = max(int(horizon), 0)
    arr = np.asarray(history, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        n_series = arr.shape[1] if arr.ndim == 2 else 1
        return np.zeros((horizon, max(n_series, 0)), dtype=float)
    n_time, n_series = arr.shape
    season = max(int(season_m), 1)
    season = min(season, n_time)

    # per-column last finite value, used wherever the seasonal slot is missing
    last_finite = np.zeros(n_series, dtype=float)
    for j in range(n_series):
        col = arr[:, j]
        finite = np.flatnonzero(np.isfinite(col))
        if finite.size:
            last_finite[j] = col[finite[-1]]

    tail = arr[n_time - season :, :]
    out = np.empty((horizon, n_series), dtype=float)
    for h in range(horizon):
        row = tail[h % season, :]
        out[h, :] = np.where(np.isfinite(row), row, last_finite)
    return out


def blend_forecasts(tva_fc, sn_fc, weights) -> np.ndarray:
    """Convex combination ``w * tva + (1 - w) * sn``.

    Args:
        tva_fc: (H, N) factor-mode forecast.
        sn_fc: (H, N) seasonal-naive forecast.
        weights: scalar or (N,) weight on the TVA forecast, broadcast over the
            horizon. Non-finite weights are treated as 0.0.

    Returns:
        (H, N) blended forecast. ``weights == 1.0``/``0.0`` return ``tva_fc``/
        ``sn_fc`` bitwise identical — callers rely on this.
    """
    tva = np.asarray(tva_fc, dtype=float)
    sn = np.asarray(sn_fc, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        w = np.zeros(tva.shape[-1] if tva.ndim else 1)
    w = np.where(np.isfinite(w), w, 0.0)
    if w.size == 1 and tva.ndim == 2 and tva.shape[1] != 1:
        w = np.repeat(w, tva.shape[1])
    row = w[np.newaxis, :] if tva.ndim == 2 else w
    # exact identity at the endpoints, not just numerically close
    out = np.where(
        row == 1.0, tva, np.where(row == 0.0, sn, row * tva + (1.0 - row) * sn)
    )
    return np.asarray(out, dtype=float)


def select_blend_weights(
    tva_folds, sn_folds, actual_folds, scale, config=None
) -> np.ndarray:
    """Pick, per series, how much weight the factor forecast has earned.

    Scored on inner rolling-origin validation only, never on data the factor
    model was fit to or the outer test set. Ties favor the smaller TVA
    weight: equal measured accuracy is not equal risk, since the higher
    weight carries unmeasured extrapolation risk into the future.

    Args:
        tva_folds: list of (H, N) factor forecasts, one per inner origin.
        sn_folds: list of (H, N) seasonal-naive forecasts, same origins/shapes.
        actual_folds: list of (H, N) realized values, same origins/shapes.
        scale: (N,) MASE denominator (in-sample seasonal-naive MAE per series).
        config: overrides for ``DEFAULT_SAFETY_CONFIG``.

    Returns:
        (N,) weights on the TVA forecast, each drawn from ``blend_grid``. A
        series with no finite score anywhere — all-NaN history, no usable fold —
        gets 0.0, i.e. pure SeasonalNaive, the safe default.
    """
    cfg = _merged_config(config)
    tva = _as_2d_list(tva_folds)
    sn = _as_2d_list(sn_folds)
    act = _as_2d_list(actual_folds)
    n_folds = min(len(tva), len(sn), len(act))
    if n_folds == 0:
        n_series = 0
        for group in (tva, sn, act):
            if group:
                n_series = group[0].shape[1]
                break
        return np.zeros(n_series, dtype=float)
    n_series = min(a.shape[1] for a in (tva[:n_folds] + sn[:n_folds] + act[:n_folds]))
    scale_v = _safe_scale(scale, n_series)

    grid = np.asarray(cfg.get('blend_grid') or (), dtype=float).ravel()
    if grid.size == 0:
        return np.zeros(n_series, dtype=float)
    tol = float(cfg.get('blend_tie_tolerance', 0.0) or 0.0)

    # a pure mean-over-folds objective misses per-fold blowups the worst-series
    # promotion gate cares about; mean + risk * worst_fold penalizes those
    risk = float(cfg.get('blend_risk_weight', 0.0) or 0.0)

    scores = np.full((grid.size, n_series), np.nan)
    for gi, w in enumerate(grid):
        errors = []
        per_fold = []
        for f in range(n_folds):
            n_h = min(tva[f].shape[0], sn[f].shape[0], act[f].shape[0])
            if n_h == 0:
                continue
            t = tva[f][:n_h, :n_series]
            s = sn[f][:n_h, :n_series]
            a = act[f][:n_h, :n_series]
            err = (w * t + (1.0 - w) * s) - a
            errors.append(err)
            if risk > 0:
                per_fold.append(_pooled_scaled_mae([err], scale_v, n_series))
        pooled = _pooled_scaled_mae(errors, scale_v, n_series)
        if risk > 0 and per_fold:
            with np.errstate(invalid='ignore'):
                worst = np.nanmax(np.vstack(per_fold), axis=0)
            pooled = pooled + risk * np.where(np.isfinite(worst), worst, 0.0)
        scores[gi] = pooled

    out = np.zeros(n_series, dtype=float)
    for j in range(n_series):
        pick = _argmin_prefer_small(scores[:, j], grid, tol)
        out[j] = 0.0 if not np.isfinite(pick) else pick
    return out


def error_cap_bounds(
    sn_fc, inner_abs_errors, horizon: int, config=None, bucket_scales=None
) -> tuple:
    """Band around SeasonalNaive that the final forecast may not leave.

    The blend bounds the *average* damage; the cap bounds the *worst* cell.
    Band is ``sn_fc ± error_cap_mult * q``, where ``q`` is the
    ``error_cap_quantile`` of the baseline's own absolute inner-validation
    error. When ``bucket_scales`` is supplied the half-width is scaled per
    horizon step (1b'), since a flat band is too tight late and too loose
    early.

    Args:
        sn_fc: (H, N) seasonal-naive forecast, the band's center.
        inner_abs_errors: either a (N,) array of per-series quantiles already
            computed by the caller, or a list/array of (H, N) per-fold absolute
            error arrays, in which case the quantile is computed here.
        horizon: H, the number of steps the bounds must cover.
        config: overrides for ``DEFAULT_SAFETY_CONFIG``.
        bucket_scales: optional (H,) per-step multiplier from
            :func:`horizon_bucket_scales`. ``None`` gives a flat band.

    Returns:
        ``(lower, upper)``, each (H, N). Cells whose quantile could not be
        estimated are ``nan``, which :func:`apply_error_cap` reads as "no cap".
    """
    cfg = _merged_config(config)
    horizon = max(int(horizon), 0)
    center = np.asarray(sn_fc, dtype=float)
    if center.ndim == 1:
        center = center[:, np.newaxis]
    if center.ndim != 2 or center.size == 0:
        return np.full((horizon, 0), np.nan), np.full((horizon, 0), np.nan)
    n_series = center.shape[1]
    if center.shape[0] < horizon and center.shape[0] > 0:
        pad = np.repeat(center[-1:, :], horizon - center.shape[0], axis=0)
        center = np.concatenate([center, pad], axis=0)
    center = center[:horizon, :]

    quantile = float(cfg.get('error_cap_quantile', 0.99))
    quantile = min(max(quantile, 0.0), 1.0)
    mult = float(cfg.get('error_cap_mult', 3.0))

    # per-series half-width source: either given directly, or pooled from folds
    q = None
    if inner_abs_errors is not None:
        direct = (
            np.asarray(inner_abs_errors, dtype=float)
            if not isinstance(inner_abs_errors, (list, tuple))
            else None
        )
        if direct is not None and direct.ndim == 1 and direct.size == n_series:
            q = np.abs(direct.astype(float, copy=True))
        else:
            folds = _as_2d_list(inner_abs_errors)
            if folds:
                pooled = np.concatenate(
                    [np.abs(f.reshape(-1, f.shape[-1])[:, :n_series]) for f in folds],
                    axis=0,
                )
                q = np.full(n_series, np.nan)
                for j in range(min(n_series, pooled.shape[1])):
                    col = pooled[:, j]
                    col = col[np.isfinite(col)]
                    if col.size:
                        q[j] = float(np.quantile(col, quantile))
    if q is None:
        q = np.full(n_series, np.nan)
    if q.size < n_series:
        q = np.concatenate([q, np.full(n_series - q.size, np.nan)])
    q = q[:n_series]
    q = np.where(np.isfinite(q) & (q >= 0), q, np.nan)

    half = np.repeat((mult * q)[np.newaxis, :], horizon, axis=0)
    if bucket_scales is not None:
        scales = np.asarray(bucket_scales, dtype=float).ravel()
        if scales.size:
            if scales.size < horizon:
                scales = np.concatenate(
                    [scales, np.repeat(scales[-1:], horizon - scales.size)]
                )
            scales = scales[:horizon]
            scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
            half = half * scales[:, np.newaxis]

    lower = center - half
    upper = center + half
    return lower, upper


def apply_error_cap(forecast, lower, upper) -> np.ndarray:
    """Clip a forecast into the band, elementwise.

    NaN in a bound means "no cap on that cell". With both bounds all-NaN the
    output is bitwise identical to the input.

    Args:
        forecast: (H, N) forecast to clip.
        lower: (H, N) lower bounds, NaN where uncapped.
        upper: (H, N) upper bounds, NaN where uncapped.

    Returns:
        (H, N) clipped forecast. Use :func:`count_capped` to learn how many
        cells the band actually bound.
    """
    fc = np.asarray(forecast, dtype=float)
    out = fc.astype(float, copy=True)
    if lower is not None:
        lo = np.broadcast_to(np.asarray(lower, dtype=float), fc.shape)
        mask = np.isfinite(lo) & np.isfinite(fc) & (fc < lo)
        out = np.where(mask, lo, out)
    if upper is not None:
        up = np.broadcast_to(np.asarray(upper, dtype=float), fc.shape)
        mask = np.isfinite(up) & np.isfinite(fc) & (fc > up)
        out = np.where(mask, up, out)
    return np.asarray(out, dtype=float)


def count_capped(forecast, lower, upper) -> int:
    """Number of cells :func:`apply_error_cap` would move.

    Reported through :func:`summarize` — a large fraction of ``H * N`` signals
    structural divergence worth surfacing, not just silently correcting.
    """
    fc = np.asarray(forecast, dtype=float)
    total = 0
    if lower is not None:
        lo = np.broadcast_to(np.asarray(lower, dtype=float), fc.shape)
        total += int(np.sum(np.isfinite(lo) & np.isfinite(fc) & (fc < lo)))
    if upper is not None:
        up = np.broadcast_to(np.asarray(upper, dtype=float), fc.shape)
        total += int(np.sum(np.isfinite(up) & np.isfinite(fc) & (fc > up)))
    return total


def horizon_bucket_scales(residuals_by_h, config=None, quantile=None) -> np.ndarray:
    """How much wider the error distribution gets as the horizon grows.

    Replaces a single sigma tiled flat across H, which under- and
    over-covers in opposite directions and averages out to look fine. Each
    bucket's scale is the empirical absolute-residual quantile pooled over
    origins/steps/series, normalized so the first bucket is 1.0 — a pure
    shape that composes with any per-series sigma the caller has.

    Args:
        residuals_by_h: (O, H, N) rolling-origin residuals (actual - forecast).
        config: overrides for ``DEFAULT_SAFETY_CONFIG``; ``horizon_buckets`` are
            inclusive 1-indexed step ranges.
        quantile: nominal level for the conformal scale. Defaults to the
            config's ``error_cap_quantile``.

    Returns:
        (H,) multiplier per horizon step. Steps past the last bucket take the
        last bucket's scale; a bucket with no usable residuals inherits the
        previous bucket's scale rather than collapsing to zero. Degenerate
        input returns all ones (flat, back-compatible).
    """
    cfg = _merged_config(config)
    arr = np.asarray(residuals_by_h, dtype=float)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3 or arr.size == 0:
        n_h = arr.shape[1] if arr.ndim == 3 else 0
        return np.ones(n_h)
    n_h = arr.shape[1]
    buckets = cfg.get('horizon_buckets') or ()
    if not buckets:
        return np.ones(n_h)
    level = float(cfg.get('error_cap_quantile', 0.99) if quantile is None else quantile)
    level = min(max(level, 0.0), 1.0)

    abs_res = np.abs(arr)
    raw = []
    for start, stop in buckets:
        lo = max(int(start) - 1, 0)
        hi = min(int(stop), n_h)
        if hi <= lo:
            raw.append(np.nan)
            continue
        chunk = abs_res[:, lo:hi, :].ravel()
        chunk = chunk[np.isfinite(chunk)]
        raw.append(float(np.quantile(chunk, level)) if chunk.size else np.nan)
    raw = np.asarray(raw, dtype=float)

    base = raw[0] if raw.size else np.nan
    if not np.isfinite(base) or base <= 0:
        finite = raw[np.isfinite(raw) & (raw > 0)]
        base = float(finite[0]) if finite.size else np.nan
    if not np.isfinite(base) or base <= 0:
        return np.ones(n_h)
    scales = raw / base

    # carry the last defensible scale forward through unusable buckets
    filled = []
    last = 1.0
    for value in scales:
        if np.isfinite(value) and value > 0:
            last = float(value)
        filled.append(last)

    out = np.full(n_h, filled[-1] if filled else 1.0, dtype=float)
    for bi, (start, stop) in enumerate(buckets):
        lo = max(int(start) - 1, 0)
        hi = min(int(stop), n_h)
        if hi > lo:
            out[lo:hi] = filled[bi]
    return out


def conformal_sigma(base_sigma, horizon: int, bucket_scales, config=None) -> np.ndarray:
    """Expand a per-series sigma into a per-(step, series) sigma.

    Args:
        base_sigma: scalar or (N,) per-series sigma from the model.
        horizon: H.
        bucket_scales: (H,) multipliers from :func:`horizon_bucket_scales`, or
            ``None`` for the flat tile.
        config: unused today; accepted so callers can pass their config through
            uniformly with the rest of this module.

    Returns:
        (H, N) sigma. With ``bucket_scales=None`` this is exactly
        ``np.tile(base_sigma, (H, 1))``, bitwise.
    """
    horizon = max(int(horizon), 0)
    sigma = np.asarray(base_sigma, dtype=float).ravel()
    if sigma.size == 0:
        return np.zeros((horizon, 0), dtype=float)
    if bucket_scales is None:
        return np.tile(sigma[np.newaxis, :], (horizon, 1))
    scales = np.asarray(bucket_scales, dtype=float).ravel()
    if scales.size == 0:
        return np.tile(sigma[np.newaxis, :], (horizon, 1))
    if scales.size < horizon:
        scales = np.concatenate([scales, np.repeat(scales[-1:], horizon - scales.size)])
    scales = scales[:horizon]
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    return sigma[np.newaxis, :] * scales[:, np.newaxis]


def select_reanchor_alpha(
    tva_folds, actual_folds, anchor_levels, origin_levels, scale, config=None
) -> np.ndarray:
    """Decide how much of the origin level offset to remove, per series.

    The whole-history factor reconstruction can sit at a level that disagrees
    with recent data — a constant bias worth removing over a long horizon.
    Not removed unconditionally: the recent median is also the noisiest level
    estimate available, so alpha is selected on inner folds with the same
    tie rule preferring alpha=0.0.

    Args:
        tva_folds: list of F (H, N) factor forecasts at each inner origin.
        actual_folds: list of F (H, N) realized values.
        anchor_levels: (F, N) robust recent level — the median of the last
            ``reanchor_window`` observations of the adjusted panel at each
            inner origin.
        origin_levels: (F, N) level the model's own forecast implies at step 0
            of each fold.
        scale: (N,) MASE denominator.
        config: overrides for ``DEFAULT_SAFETY_CONFIG``.

    Returns:
        (N,) alpha per series, drawn from ``reanchor_alphas``. Series with no
        finite score, or with no usable offset, get 0.0 — no correction.
    """
    cfg = _merged_config(config)
    tva = _as_2d_list(tva_folds)
    act = _as_2d_list(actual_folds)
    n_folds = min(len(tva), len(act))
    if n_folds == 0:
        n_series = tva[0].shape[1] if tva else (act[0].shape[1] if act else 0)
        return np.zeros(n_series, dtype=float)
    n_series = min(a.shape[1] for a in (tva[:n_folds] + act[:n_folds]))

    anchor = np.atleast_2d(np.asarray(anchor_levels, dtype=float))
    origin = np.atleast_2d(np.asarray(origin_levels, dtype=float))
    offsets = np.zeros((n_folds, n_series), dtype=float)
    for f in range(n_folds):
        a_row = anchor[f] if f < anchor.shape[0] else np.full(n_series, np.nan)
        o_row = origin[f] if f < origin.shape[0] else np.full(n_series, np.nan)
        row = np.full(n_series, np.nan)
        width = min(n_series, a_row.size, o_row.size)
        row[:width] = a_row[:width] - o_row[:width]
        offsets[f] = np.where(np.isfinite(row), row, 0.0)

    grid = np.asarray(cfg.get('reanchor_alphas') or (), dtype=float).ravel()
    if grid.size == 0:
        return np.zeros(n_series, dtype=float)
    tol = float(cfg.get('blend_tie_tolerance', 0.0) or 0.0)
    scale_v = _safe_scale(scale, n_series)

    scores = np.full((grid.size, n_series), np.nan)
    for gi, alpha in enumerate(grid):
        errors = []
        for f in range(n_folds):
            n_h = min(tva[f].shape[0], act[f].shape[0])
            if n_h == 0:
                continue
            t = tva[f][:n_h, :n_series]
            a = act[f][:n_h, :n_series]
            errors.append(t + alpha * offsets[f][np.newaxis, :] - a)
        scores[gi] = _pooled_scaled_mae(errors, scale_v, n_series)

    out = np.zeros(n_series, dtype=float)
    for j in range(n_series):
        pick = _argmin_prefer_small(scores[:, j], grid, tol)
        out[j] = 0.0 if not np.isfinite(pick) else pick
    return out


def apply_reanchor(forecast, offset, alpha) -> np.ndarray:
    """Shift a forecast by ``alpha * offset``, one constant per series.

    Args:
        forecast: (H, N) forecast.
        offset: scalar or (N,) level offset (``anchor - origin``). Non-finite
            entries are treated as 0.0.
        alpha: scalar or (N,) selected fraction of the offset to apply.

    Returns:
        (H, N) shifted forecast. With ``alpha == 0`` the output is bitwise
        identical to the input.
    """
    fc = np.asarray(forecast, dtype=float)
    n_series = fc.shape[1] if fc.ndim == 2 else fc.size
    off = np.asarray(offset, dtype=float).ravel()
    alp = np.asarray(alpha, dtype=float).ravel()
    if off.size == 0:
        off = np.zeros(n_series)
    if alp.size == 0:
        alp = np.zeros(n_series)
    if off.size == 1 and n_series != 1:
        off = np.repeat(off, n_series)
    if alp.size == 1 and n_series != 1:
        alp = np.repeat(alp, n_series)
    off = np.where(np.isfinite(off[:n_series]), off[:n_series], 0.0)
    alp = np.where(np.isfinite(alp[:n_series]), alp[:n_series], 0.0)
    shift = alp * off
    if not np.any(shift != 0.0):
        return fc.astype(float, copy=True)
    return fc + shift[np.newaxis, :]


def _json_list(values):
    """NaN-free python list, or ``None`` — json.dumps cannot emit NaN."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).ravel()
    return [None if not np.isfinite(v) else float(v) for v in arr]


def _mean_or_none(values):
    arr = (
        np.asarray(values, dtype=float).ravel() if values is not None else np.array([])
    )
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def summarize(
    blend_weights=None,
    reanchor_alphas=None,
    bucket_scales=None,
    capped_cells=None,
    scale=None,
    offsets=None,
    n_series=None,
    horizon=None,
    config=None,
    **extra,
) -> dict:
    """JSON-safe diagnostics for ``get_factor_diagnostics()``.

    Reports both aggregate summary stats and full per-series arrays.
    ``n_series_zero_weight == N`` is a valid outcome (factor mode earned
    nothing on this dataset), not a bug to tune away.

    Args:
        blend_weights: (N,) selected weights on the TVA forecast.
        reanchor_alphas: (N,) selected re-anchor fractions.
        bucket_scales: (H,) horizon-bucket sigma multipliers.
        capped_cells: number of cells the error cap moved.
        scale: (N,) MASE denominators used for selection.
        offsets: (N,) applied level offsets.
        n_series: N, if the caller wants it recorded explicitly.
        horizon: H, if the caller wants it recorded explicitly.
        config: the safety config in force; stored stringified-safe.
        **extra: any additional scalars/arrays to record verbatim.

    Returns:
        dict of json-serializable values; NaN becomes ``None``.
    """
    weights = (
        np.asarray(blend_weights, dtype=float).ravel()
        if blend_weights is not None
        else np.array([])
    )
    alphas = (
        np.asarray(reanchor_alphas, dtype=float).ravel()
        if reanchor_alphas is not None
        else np.array([])
    )
    out = {
        'n_series': (
            int(n_series)
            if n_series is not None
            else (int(weights.size) if weights.size else None)
        ),
        'horizon': int(horizon) if horizon is not None else None,
        'blend_weight_mean': _mean_or_none(weights),
        'blend_weights': _json_list(weights) if weights.size else None,
        'n_series_zero_weight': int(np.sum(weights == 0.0)) if weights.size else None,
        'n_series_full_weight': int(np.sum(weights == 1.0)) if weights.size else None,
        'reanchor_alpha_mean': _mean_or_none(alphas),
        'reanchor_alphas': _json_list(alphas) if alphas.size else None,
        'n_series_reanchored': int(np.sum(alphas != 0.0)) if alphas.size else None,
        'bucket_scales': _json_list(bucket_scales),
        'capped_cells': int(capped_cells) if capped_cells is not None else None,
        'scale': _json_list(scale),
        'offsets': _json_list(offsets),
    }
    if config is not None:
        out['config'] = {str(k): _jsonable(v) for k, v in dict(config).items()}
    for key, value in extra.items():
        out[str(key)] = _jsonable(value)
    return out


def _jsonable(value):
    """Best-effort conversion of a config/diagnostic value to json-safe form."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return str(value)
