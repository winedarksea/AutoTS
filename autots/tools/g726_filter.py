"""G.726-inspired adaptive differential pulse-code modulation utilities.

The functions in this module implement a light-weight version of a G.726 style
encoder/decoder pair.  Instead of storing the encoded stream, the quantization
cycle is used as a denoising filter: data are projected onto a low bit-depth
adaptive differential representation and then reconstructed.  High-frequency
noise is attenuated while low-frequency structure is preserved, making the
filter useful for short, daily time series (≈500 observations).

Only NumPy, SciPy, and pandas are required, which keeps the dependency surface
small so the transformer can run inside the AutoTS toolbox without optional
libraries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - SciPy is expected but AutoTS has fallbacks.
    from scipy.signal import lfilter
except Exception:  # pragma: no cover
    lfilter = None

EPSILON = np.finfo(float).eps

# Non-uniform quantizer decision levels and reconstruction values
# Optimized for daily time series data (4-bit variant, 16 levels)
# Based on G.726 32kbps tables but tuned for daily/hourly patterns
QUANTIZER_DECISION_LEVELS = np.array([
    -7.5, -5.5, -4.0, -3.0, -2.0, -1.3, -0.7, 0.0,
    0.7, 1.3, 2.0, 3.0, 4.0, 5.5, 7.5
])

QUANTIZER_RECONSTRUCTION = np.array([
    -8.5, -6.5, -4.75, -3.5, -2.5, -1.65, -1.0, -0.35,
    0.35, 1.0, 1.65, 2.5, 3.5, 4.75, 6.5, 8.5
])

# Scale factor adaptation multipliers (based on G.726)
# Index by quantizer output (0-15 for 4-bit)
SCALE_FACTOR_MULTIPLIERS = np.array([
    0.60, 0.93, 0.93, 0.93, 1.20, 1.20, 1.20, 1.20,
    1.20, 1.20, 1.20, 1.20, 0.93, 0.93, 0.93, 0.60
])

# Adaptation speed control
FAST_SCALE_ALPHA = 0.875  # Fast adaptation for scale factor
SLOW_SCALE_ALPHA = 0.985  # Slow adaptation for scale factor
PREDICTOR_LEAK = 0.9999   # Coefficient leakage factor


def _adaptive_predictor_step(
    a1: np.ndarray,
    a2: np.ndarray,
    b: np.ndarray,
    sez_history: np.ndarray,
    dq: np.ndarray,
    leak: float = PREDICTOR_LEAK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update 2-pole/6-zero predictor coefficients and state.
    
    Parameters
    ----------
    a1, a2 : ndarray
        Adaptive pole coefficients (shape: n_series)
    b : ndarray
        Adaptive zero coefficients (shape: 6, n_series)
    sez_history : ndarray
        Quantized difference signal history (shape: 6, n_series)
    dq : ndarray
        Quantized difference signal (current sample, shape: n_series)
    leak : float
        Coefficient leakage factor to prevent unbounded growth
    
    Returns
    -------
    a1_new, a2_new, b_new, sez_new : tuple of ndarrays
        Updated coefficients and shifted state
    """
    # Shift quantized difference history
    sez_new = np.roll(sez_history, 1, axis=0)
    sez_new[0] = dq
    
    # Gradient adaptation using sign-based LMS (more stable than full LMS)
    # Very small step sizes for stability
    sgn_dq = np.sign(dq)
    
    # Update pole coefficients with strong leakage
    mu_a = 0.001  # Very small step size
    a1_new = leak * a1 + mu_a * sgn_dq * np.sign(sez_history[0])
    a2_new = leak * a2 + mu_a * sgn_dq * np.sign(sez_history[1])
    
    # Strong clamping for stability (poles must be inside unit circle)
    a1_new = np.clip(a1_new, -0.7, 0.7)
    a2_new = np.clip(a2_new, -0.7, 0.7)
    
    # Update zero coefficients
    mu_b = 0.0005  # Even smaller for zeros
    b_new = np.copy(b)
    for i in range(6):
        b_new[i] = leak * b[i] + mu_b * sgn_dq * np.sign(sez_history[i])
        b_new[i] = np.clip(b_new[i], -1.0, 1.0)
    
    return a1_new, a2_new, b_new, sez_new


def _compute_prediction(
    a1: np.ndarray,
    a2: np.ndarray,
    b: np.ndarray,
    sr_history: np.ndarray,
    sez_history: np.ndarray,
) -> np.ndarray:
    """Compute adaptive prediction from pole-zero model.
    
    Parameters
    ----------
    a1, a2 : ndarray
        Pole coefficients (shape: n_series)
    b : ndarray
        Zero coefficients (shape: 6, n_series)
    sr_history : ndarray
        Past reconstructed samples (shape: 2, n_series)
    sez_history : ndarray
        Past quantized differences (shape: 6, n_series)
    
    Returns
    -------
    ndarray
        Predicted value (shape: n_series)
    """
    # Clip inputs to prevent numerical issues
    sr_history = np.clip(sr_history, -1e6, 1e6)
    sez_history = np.clip(sez_history, -1e6, 1e6)
    
    # Pole contribution (2nd order recursive on reconstructed signal)
    pole_pred = a1 * sr_history[0] + a2 * sr_history[1]
    
    # Zero contribution (6-tap FIR on quantized differences)
    zero_pred = (b * sez_history).sum(axis=0)
    
    # Combined prediction
    prediction = pole_pred + zero_pred
    
    # Clip to prevent overflow
    prediction = np.clip(prediction, -1e6, 1e6)
    
    return prediction


def _quantize_uniform(residual: np.ndarray, scale: np.ndarray, bits: int) -> np.ndarray:
    """Uniform quantization (original method)."""
    max_level = (1 << (bits - 1)) - 1
    min_level = -(1 << (bits - 1))
    return np.clip(np.round(residual / (scale + EPSILON)), min_level, max_level)


def _quantize_nonuniform(residual: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-uniform quantization using decision levels.
    
    Returns
    -------
    indices : ndarray
        Quantizer indices (0-15 for 4-bit)
    reconstructed : ndarray
        Reconstructed values
    """
    # Normalize by scale factor
    normalized = residual / (scale + EPSILON)
    
    # Find quantizer bin using decision levels
    indices = np.searchsorted(QUANTIZER_DECISION_LEVELS, normalized)
    
    # Get reconstruction values and scale back
    reconstructed = QUANTIZER_RECONSTRUCTION[indices] * scale
    
    return indices, reconstructed


def _update_scale_factor(
    scale: np.ndarray,
    quant_index: np.ndarray,
    fast_mode: np.ndarray,
) -> np.ndarray:
    """Update scale factor using G.726 adaptation logic.
    
    Parameters
    ----------
    scale : ndarray
        Current scale factor (shape: n_series)
    quant_index : ndarray
        Quantizer output indices (shape: n_series)
    fast_mode : ndarray
        Boolean array indicating fast adaptation mode (shape: n_series)
    
    Returns
    -------
    ndarray
        Updated scale factor
    """
    # Get multiplier from lookup table
    multiplier = SCALE_FACTOR_MULTIPLIERS[quant_index.astype(int)]
    
    # Choose adaptation speed
    alpha = np.where(fast_mode, FAST_SCALE_ALPHA, SLOW_SCALE_ALPHA)
    
    # Update: scale = alpha * scale + (1 - alpha) * (multiplier * scale)
    # Simplified: scale = scale * (alpha + (1 - alpha) * multiplier)
    scale_new = scale * (alpha + (1.0 - alpha) * multiplier)
    
    # Bound scale factor (prevent too small or too large)
    scale_new = np.clip(scale_new, 0.001, 100.0)
    
    return scale_new


def _prep_input(values: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, bool]:
    """Return a float64 array with shape (n_obs, n_series) and squeeze flag."""
    if isinstance(values, pd.DataFrame):
        arr = values.to_numpy(dtype=float, copy=True)
    else:
        arr = np.asarray(values, dtype=float)
    squeezed = False
    if arr.ndim == 1:
        arr = arr[:, None]
        squeezed = True
    return arr.copy(), squeezed


def g726_adpcm_filter(
    values: np.ndarray | pd.DataFrame,
    quant_bits: int = 4,
    adaptation_rate: float = 0.96,
    prediction_alpha: float = 0.92,
    floor_step: float = 0.01,
    dynamic_range: float = 1.5,
    blend: float = 0.15,
    noise_gate: float = 0.0,
    quantizer: str = "uniform",
    use_adaptive_predictor: bool = True,
    predictor_leak: float = PREDICTOR_LEAK,
) -> np.ndarray:
    """Apply the adaptive encode/decode cycle to an array of time series.

    Parameters
    ----------
    values : array-like or DataFrame
        Shape (observations, series). Each column is processed independently,
        but operations stay vectorized across series.
    quant_bits : int, default 4
        Number of bits used for the quantizer (only for uniform mode).
    adaptation_rate : float, default 0.96
        Only used when use_adaptive_predictor=False (legacy EMA mode).
    prediction_alpha : float, default 0.92
        Only used when use_adaptive_predictor=False (legacy EMA mode).
    floor_step : float, default 0.01
        Initial scale factor (minimum allowed value).
    dynamic_range : float, default 1.5
        Initial scale multiplier (legacy mode only).
    blend : float, default 0.15
        Optional blending factor with the predictor baseline. Must be in [0, 1].
    noise_gate : float, default 0.0
        Additional soft threshold applied to residual magnitudes.
    quantizer : str, default "uniform"
        Quantization method: "uniform" for simple rounding, "nonuniform" for
        G.726-style decision levels optimized for time series.
    use_adaptive_predictor : bool, default True
        Use 2-pole/6-zero adaptive predictor. If False, falls back to EMA.
    predictor_leak : float, default 0.9999
        Leakage factor for predictor coefficients to prevent unbounded growth.

    Returns
    -------
    numpy.ndarray
        Filtered data with the same shape as ``values``.
    """

    arr, squeezed = _prep_input(values)
    if arr.size == 0 or arr.shape[0] <= 1:
        return arr if not squeezed else arr[:, 0]

    n_obs, n_series = arr.shape
    blend = float(np.clip(blend, 0.0, 1.0))
    
    if not use_adaptive_predictor:
        # Legacy mode: simple EMA predictor (backward compatibility)
        return _legacy_filter(
            arr, squeezed, quant_bits, adaptation_rate, prediction_alpha,
            floor_step, dynamic_range, blend, noise_gate
        )
    
    # Initialize adaptive predictor state (vectorized across series)
    a1 = np.zeros(n_series)  # 1st pole coefficient
    a2 = np.zeros(n_series)  # 2nd pole coefficient
    b = np.zeros((6, n_series))  # 6 zero coefficients
    
    # History buffers
    sr_history = np.zeros((2, n_series))  # Past reconstructed samples (for poles)
    sez_history = np.zeros((6, n_series))  # Past quantized differences (for zeros)
    
    # Scale factor initialization - estimate from first few samples
    if n_obs > 5:
        initial_scale = np.maximum(np.std(arr[:10], axis=0), floor_step)
    else:
        initial_scale = np.full(n_series, floor_step, dtype=float)
    scale = np.asarray(initial_scale, dtype=float).copy()
    
    # Track fast/slow adaptation mode
    fast_mode = np.ones(n_series, dtype=bool)  # Start in fast mode
    transition_count = np.zeros(n_series, dtype=int)
    
    # Output array
    filtered = np.empty_like(arr)
    filtered[0] = arr[0]  # First sample passes through
    sr_history[0] = 0.0  # Initialize with zero difference
    
    # Synchronous coding: iterate over time, vectorize across series
    for t in range(1, n_obs):
        # Compute prediction from adaptive model
        prediction = _compute_prediction(a1, a2, b, sr_history, sez_history)
        
        # Residual (difference from prediction)
        residual = arr[t] - arr[t-1] - prediction
        
        # Optional noise gate
        if noise_gate > 0.0:
            gate = np.maximum(0.0, 1.0 - (noise_gate / (np.abs(residual) + noise_gate)))
            residual = residual * gate
        
        # Quantization
        if quantizer == "nonuniform":
            quant_idx, reconstructed_diff = _quantize_nonuniform(residual, scale)
        else:
            # Uniform quantization (legacy mode)
            # Note: Scale adaptation LUT is optimized for 4-bit; other bit depths
            # will have approximate adaptation behavior
            bits = int(np.clip(np.round(quant_bits), 2, 6))
            encoded = _quantize_uniform(residual, scale, bits)
            reconstructed_diff = encoded * scale
            # Map to indices for scale adaptation (approximate for non-4-bit)
            quant_idx = (encoded + (1 << (bits - 1))).astype(int)
            quant_idx = np.clip(quant_idx, 0, 15)
        
        # Reconstruct signal (integrate differences)
        reconstructed = arr[t-1] + prediction + reconstructed_diff
        
        # Clip to prevent overflow
        reconstructed = np.clip(reconstructed, -1e8, 1e8)
        filtered[t] = reconstructed
        
        # Update scale factor with adaptive speed
        scale = _update_scale_factor(scale, quant_idx, fast_mode)
        
        # Update fast/slow mode based on stability
        # Switch to slow mode if signal is stable (center quantizer indices)
        stable = (quant_idx >= 6) & (quant_idx <= 9)
        transition_count = np.where(stable, transition_count + 1, 0)
        fast_mode = transition_count < 5  # Require 5 stable samples to switch
        
        # Update predictor coefficients and state (uses quantized differences)
        a1, a2, b, sez_history = _adaptive_predictor_step(
            a1, a2, b, sez_history, reconstructed_diff, predictor_leak
        )
        
        # Update reconstructed signal history for poles
        sr_history = np.roll(sr_history, 1, axis=0)
        sr_history[0] = reconstructed
    
    # Optional blending with slow-moving baseline
    if blend > 0.0:
        # Compute simple baseline for blending using prediction_alpha
        if lfilter is not None:
            alpha = float(np.clip(prediction_alpha, 0.0, 0.9999))
            baseline = lfilter([1.0 - alpha], [1.0, -alpha], arr, axis=0)
        else:
            baseline = arr  # Skip blending if no scipy
        filtered = (1.0 - blend) * filtered + blend * baseline
    
    return filtered if not squeezed else filtered[:, 0]


def _legacy_filter(
    arr: np.ndarray,
    squeezed: bool,
    quant_bits: int,
    adaptation_rate: float,
    prediction_alpha: float,
    floor_step: float,
    dynamic_range: float,
    blend: float,
    noise_gate: float,
) -> np.ndarray:
    """Legacy EMA-based filter (for backward compatibility)."""
    bits = int(np.clip(np.round(quant_bits), 2, 6))
    
    # Simple EMA predictor (non-synchronous, uses original signal)
    if lfilter is not None:
        alpha = float(np.clip(prediction_alpha, 0.0, 0.9999))
        baseline = lfilter([1.0 - alpha], [1.0, -alpha], arr, axis=0)
    else:
        baseline = arr * 0.5  # Fallback
    
    baseline[:1, :] = arr[:1, :]
    residual = arr - baseline
    
    if noise_gate > 0.0:
        gate = np.maximum(0.0, 1.0 - (noise_gate / (np.abs(residual) + noise_gate)))
        residual = residual * gate
    
    # Simple scale adaptation
    if lfilter is not None:
        alpha = float(np.clip(adaptation_rate, 0.0, 0.9999))
        scale = lfilter([1.0 - alpha], [1.0, -alpha], np.abs(residual), axis=0)
    else:
        scale = np.abs(residual)
    
    scale = floor_step + dynamic_range * scale
    scale = np.maximum(scale, floor_step)
    
    # Quantize
    encoded = _quantize_uniform(residual, scale, bits)
    decoded = encoded * scale
    filtered = baseline + decoded
    filtered[:1, :] = arr[:1, :]
    
    if blend > 0.0:
        filtered = (1.0 - blend) * filtered + blend * baseline
    
    return filtered if not squeezed else filtered[:, 0]


__all__ = ["g726_adpcm_filter"]
