# -*- coding: utf-8 -*-
"""
Origin anchoring: pin a forecast's starting level to the recent observed level.

The LRP backtest found TVA's largest error is a level step at the forecast
origin (the factor reconstruction and the projected seasonal path disagree
with where the data actually ended), not trend extrapolation. This is the
native equivalent of AutoTS ``AlignLastValue``.

Both sides are compared as a full-window mean (last ``window`` observations vs
first ``window`` forecast steps) so a weekly cycle doesn't read as a level gap.

Modes:
    'last_value': raw values on both sides (the backtested arm).
    'deseasonalized': periodic components (seasonality + holidays) removed
        from both windows first, so an origin inside a holiday season isn't
        anchored to the holiday-depressed level.

A multiplicative anchor scales the whole path, so a badly placed first step
(observed / forecast start far from 1) multiplies every later step too: the
LRP detector review saw 180x blow-ups. Ratios outside ``max_ratio`` therefore
fall back to an additive shift, which corrects the same start gap without
rescaling the trend.

Pure numpy.
"""

from __future__ import annotations

import warnings

import numpy as np

ORIGIN_ANCHOR_MODES = ('last_value', 'deseasonalized')
DEFAULT_ORIGIN_ANCHOR_MAX_RATIO = 1.5


def origin_anchor_adjustment(
    history: np.ndarray,
    forecast: np.ndarray,
    window: int = 7,
    mode: str = 'last_value',
    history_periodic: np.ndarray = None,
    forecast_periodic: np.ndarray = None,
    max_ratio: float = DEFAULT_ORIGIN_ANCHOR_MAX_RATIO,
) -> dict:
    """Per-series adjustment that moves the forecast's start to the recent level.

    Args:
        history: (T, N) observed values (NaN allowed).
        forecast: (H, N) forecast to be anchored.
        window: number of periods averaged on each side.
        mode: one of ``ORIGIN_ANCHOR_MODES``.
        history_periodic: (T, N) in-sample seasonality + holidays; used by
            'deseasonalized' only.
        forecast_periodic: (H, N) future seasonality + holidays; used by
            'deseasonalized' only.
        max_ratio: multiplicative ratios outside ``[1/max_ratio, max_ratio]``
            switch that series to an additive shift. None = unbounded.

    Returns:
        dict with ``ratio`` (N,) multiplicative factor (1.0 where unused),
        ``offset`` (N,) additive shift (0.0 where unused) and
        ``multiplicative`` (N,) bool. Apply with :func:`apply_origin_anchor`.
        A series with no finite recent observation is left untouched.
    """
    if mode not in ORIGIN_ANCHOR_MODES:
        raise ValueError(
            f"origin_anchor={mode!r} not recognized; use one of {ORIGIN_ANCHOR_MODES}"
        )
    hist = np.asarray(history, dtype=float)
    fc = np.asarray(forecast, dtype=float)
    n_series = fc.shape[1]
    w_hist = int(max(1, min(int(window), hist.shape[0])))
    w_fc = int(max(1, min(int(window), fc.shape[0])))

    recent = hist[-w_hist:]
    start = fc[:w_fc]
    if mode == 'deseasonalized':
        if history_periodic is not None:
            recent = recent - np.asarray(history_periodic, dtype=float)[-w_hist:]
        if forecast_periodic is not None:
            start = start - np.asarray(forecast_periodic, dtype=float)[:w_fc]

    with warnings.catch_warnings():
        # all-NaN windows are expected for short or late-starting series
        warnings.simplefilter('ignore', RuntimeWarning)
        observed_level = np.nanmean(recent, axis=0)
        forecast_level = np.nanmean(start, axis=0)
        raw_min = np.nanmin(hist[-w_hist:], axis=0)

    usable = np.isfinite(observed_level) & np.isfinite(forecast_level)
    # scaling is the AlignLastValue behavior for positive data; signed or
    # zero-touching series can't be scaled meaningfully so they shift instead
    multiplicative = (
        usable & np.isfinite(raw_min) & (raw_min > 0) & (forecast_level > 0)
    )
    ratio = np.ones(n_series, dtype=float)
    offset = np.zeros(n_series, dtype=float)
    ratio[multiplicative] = (
        observed_level[multiplicative] / forecast_level[multiplicative]
    )
    if max_ratio is not None:
        bound = float(max_ratio)
        if not bound >= 1.0:
            raise ValueError(f"max_ratio must be >= 1 or None, got {max_ratio!r}")
        out_of_bounds = multiplicative & (
            (ratio > bound) | (ratio < 1.0 / bound)
        )
        multiplicative = multiplicative & ~out_of_bounds
        ratio[out_of_bounds] = 1.0
    additive = usable & ~multiplicative
    offset[additive] = observed_level[additive] - forecast_level[additive]
    return {'ratio': ratio, 'offset': offset, 'multiplicative': multiplicative}


def apply_origin_anchor(forecast: np.ndarray, adjustment: dict) -> np.ndarray:
    """(H, N) forecast scaled and shifted by an :func:`origin_anchor_adjustment`."""
    fc = np.asarray(forecast, dtype=float)
    return (
        fc * adjustment['ratio'][np.newaxis, :] + adjustment['offset'][np.newaxis, :]
    )
