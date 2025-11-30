"""G.711 and G.726 audio codec inspired utilities for time series processing.

This module provides:
1. G.711 companding (mu-law and A-law) for non-linear scaling
2. G.726 adaptive differential PCM for denoising/smoothing
3. Transformer classes (G711Scaler, G726Filter) for AutoTS integration

G.711 is used as a non-linear scaler: encode on transform, decode on inverse_transform.
G.726 is used as an adaptive filter to denoise while preserving structure.

All operations are fully vectorized for efficient processing of multiple time series.
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd

try:  # pragma: no cover - SciPy is expected but AutoTS has fallbacks.
    from scipy.signal import lfilter
except Exception:  # pragma: no cover
    lfilter = None

# Use consistent epsilon for numerical stability
EPS = 1e-12
EPSILON = np.finfo(float).eps

# ============================================================================
# G.711 Companding (Mu-law and A-law)
# ============================================================================


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


def _robust_center_scale(
    arr: np.ndarray,
    center: str = "median",
    scale_method: str = "mad",
    scale_factor: float = 3.0,
    min_scale: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-series center and scale using robust statistics.

    Parameters
    ----------
    arr : ndarray of shape (n_obs, n_series)
    center : {'median','mean'}
    scale_method : {'mad','std','maxabs','percentile'}
    scale_factor : float multiplier applied to scale estimate
    min_scale : minimum scale to avoid division by zero
    """
    if center == "mean":
        c = np.nanmean(arr, axis=0, keepdims=True)
    else:
        c = np.nanmedian(arr, axis=0, keepdims=True)

    x = arr - c
    sm = scale_method.lower()
    if sm == "std":
        s = np.nanstd(x, axis=0, ddof=0, keepdims=True)
    elif sm == "maxabs":
        s = np.nanmax(np.abs(x), axis=0, keepdims=True)
    elif sm == "percentile":
        # Use 90th percentile of absolute deviations for better sparse data handling
        s = np.nanpercentile(np.abs(x), 90, axis=0, keepdims=True)
    else:  # 'mad' (scaled to be comparable to std under normality)
        med = np.nanmedian(np.abs(x), axis=0, keepdims=True)
        s = 1.4826 * med

        # Fallback to percentile if MAD is suspiciously small (indicates sparse data)
        # This helps with intermittent series where median deviation is near zero
        percentile_scale = np.nanpercentile(np.abs(x), 90, axis=0, keepdims=True)
        too_small = s < (percentile_scale * 0.1)  # MAD < 10% of p90
        if np.any(too_small):
            s = np.where(too_small, percentile_scale, s)

    s = np.maximum(s * float(scale_factor), float(min_scale))
    return c, s


def _mu_law_compress(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """Mu-law companding for normalized input in [-1, 1]."""
    mu = float(mu)
    x = np.clip(x, -1.0, 1.0)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def _mu_law_expand(y: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """Inverse mu-law for y in [-1, 1]."""
    mu = float(mu)
    y = np.clip(y, -1.0, 1.0)
    return np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu)


def _a_law_compress(x: np.ndarray, A: float = 87.6) -> np.ndarray:
    """A-law companding for normalized input in [-1, 1]."""
    A = float(A)
    x = np.clip(x, -1.0, 1.0)
    ax = np.abs(x)
    k = 1.0 / (1.0 + np.log(A))
    # piecewise without explicit Python loops
    small = ax <= (1.0 / A)
    y = np.empty_like(x)
    y[small] = np.sign(x[small]) * (A * ax[small]) * k
    y[~small] = np.sign(x[~small]) * (1.0 + np.log(A * ax[~small])) * k
    return y


def _a_law_expand(y: np.ndarray, A: float = 87.6) -> np.ndarray:
    """Inverse A-law for y in [-1, 1]."""
    A = float(A)
    y = np.clip(y, -1.0, 1.0)
    ay = np.abs(y)
    k = 1.0 + np.log(A)
    thresh = 1.0 / k
    x = np.empty_like(y)
    small = ay <= thresh
    x[small] = np.sign(y[small]) * (ay[small] * k) / A
    # exp component
    x[~small] = np.sign(y[~small]) * (np.exp(ay[~small] * k - 1.0) / A)
    return x


def g711_encode(
    values: np.ndarray | pd.DataFrame,
    mode: str = "mu",
    mu: float = 255.0,
    A: float = 87.6,
    center: str = "median",
    scale_method: str = "mad",
    scale_factor: float = 3.0,
    min_scale: float = 1e-6,
    clip: bool = True,
    zero_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode values via G.711-style companding after robust normalization.

    Returns encoded array and the center/scale used so that decoding can
    recover the original feature space.

    Parameters
    ----------
    zero_offset : float, optional
        Small constant added before transformation to handle exact zeros.
        Useful for sparse/intermittent data. Default 0.0 (no offset).
    """
    arr, squeezed = _prep_input(values)
    if arr.size == 0:
        # Return empty arrays with appropriate shapes
        return arr, np.array([]), np.array([])

    # Apply zero offset if specified
    if zero_offset != 0.0:
        arr = arr + zero_offset

    c, s = _robust_center_scale(
        arr,
        center=center,
        scale_method=scale_method,
        scale_factor=scale_factor,
        min_scale=min_scale,
    )
    x = (arr - c) / (s + EPS)
    if clip:
        x = np.clip(x, -1.0, 1.0)

    m = mode.lower()
    if m in ("mu", "mulaw", "u", "ulaw"):
        y = _mu_law_compress(x, mu=mu)
    elif m in ("a", "alaw"):
        y = _a_law_compress(x, A=A)
    else:
        raise ValueError(f"Unknown G711 mode: {mode}")

    return y if not squeezed else y[:, 0], c, s


def g711_decode(
    encoded: np.ndarray | pd.DataFrame,
    center: np.ndarray,
    scale: np.ndarray,
    mode: str = "mu",
    mu: float = 255.0,
    A: float = 87.6,
    zero_offset: float = 0.0,
) -> np.ndarray:
    """Decode from G.711 companded space back to original feature space.

    Parameters
    ----------
    zero_offset : float, optional
        Must match the offset used in encoding. Subtracted after decoding.
    """
    arr, squeezed = _prep_input(encoded)
    if arr.size == 0:
        return arr

    m = mode.lower()
    if m in ("mu", "mulaw", "u", "ulaw"):
        x = _mu_law_expand(arr, mu=mu)
    elif m in ("a", "alaw"):
        x = _a_law_expand(arr, A=A)
    else:
        raise ValueError(f"Unknown G711 mode: {mode}")

    out = x * (scale + EPS) + center

    # Remove zero offset if it was applied
    if zero_offset != 0.0:
        out = out - zero_offset

    return out if not squeezed else out[:, 0]


# ============================================================================
# G.726 Adaptive Differential PCM
# ============================================================================

# Non-uniform quantizer decision levels and reconstruction values
# Optimized for daily time series data (4-bit variant, 16 levels)
# Based on G.726 32kbps tables but tuned for daily/hourly patterns
QUANTIZER_DECISION_LEVELS = np.array(
    [-7.5, -5.5, -4.0, -3.0, -2.0, -1.3, -0.7, 0.0, 0.7, 1.3, 2.0, 3.0, 4.0, 5.5, 7.5]
)

QUANTIZER_RECONSTRUCTION = np.array(
    [
        -8.5,
        -6.5,
        -4.75,
        -3.5,
        -2.5,
        -1.65,
        -1.0,
        -0.35,
        0.35,
        1.0,
        1.65,
        2.5,
        3.5,
        4.75,
        6.5,
        8.5,
    ]
)

# Scale factor adaptation multipliers (based on G.726)
# Index by quantizer output (0-15 for 4-bit)
SCALE_FACTOR_MULTIPLIERS = np.array(
    [
        0.60,
        0.93,
        0.93,
        0.93,
        1.20,
        1.20,
        1.20,
        1.20,
        1.20,
        1.20,
        1.20,
        1.20,
        0.93,
        0.93,
        0.93,
        0.60,
    ]
)

# Adaptation speed control
FAST_SCALE_ALPHA = 0.875  # Fast adaptation for scale factor
SLOW_SCALE_ALPHA = 0.985  # Slow adaptation for scale factor
PREDICTOR_LEAK = 0.9999  # Coefficient leakage factor


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


def _quantize_nonuniform(
    residual: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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

    # Handle NaN/Inf (replace with zero) and clamp to sane range
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)
    normalized = np.clip(normalized, -15.0, 15.0)

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
            arr,
            squeezed,
            quant_bits,
            adaptation_rate,
            prediction_alpha,
            floor_step,
            dynamic_range,
            blend,
            noise_gate,
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
    sr_history[0] = arr[0]  # Initialize with first sample level

    # Synchronous coding: iterate over time, vectorize across series
    for t in range(1, n_obs):
        # Compute prediction from adaptive model
        prediction = _compute_prediction(a1, a2, b, sr_history, sez_history)

        # Residual (difference from prediction)
        residual = arr[t] - arr[t - 1] - prediction

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
        reconstructed = arr[t - 1] + prediction + reconstructed_diff

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


# ============================================================================
# Transformer Classes for AutoTS Integration
# ============================================================================


class EmptyTransformer(object):
    """Base transformer returning raw data."""

    def __init__(self, name: str = "EmptyTransformer", **kwargs):
        self.name = name

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self._fit(df)
        return self

    def transform(self, df):
        """Transform data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def inverse_transform(self, df, trans_method: str = "forecast", adjustment=None):
        """Inverse transform data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def fit_transform(self, df):
        """Fit and Transform data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)


class G711Scaler(EmptyTransformer):
    """G.711-style non-linear scaler (mu-law or A-law).

    Encodes (compands) on transform and decodes (expands to original space)
    on inverse_transform. Optimized for daily/hourly data by using robust
    per-series centering and scaling to normalize into [-1, 1] prior to
    companding.

    Args:
        mode: 'mu' for mu-law or 'a' for A-law companding
        mu: mu-law parameter (default 255.0, try 100-512 range)
        A: A-law parameter (default 87.6, try 50-100 range)
        center: 'median' or 'mean' for centering
        scale_method: 'mad', 'std', 'maxabs', or 'percentile' for scale estimation
        scale_factor: Multiplier for scale (default 3.0, lower=more compression, higher=better reconstruction)
        min_scale: Minimum scale to prevent division by zero
        clip: Whether to clip normalized values to [-1, 1]
        zero_offset: Small constant added to handle exact zeros (e.g., 1e-6 for sparse data)
        fill_method: Method to handle NaN values
        on_transform: Apply encoding on transform
        on_inverse: Apply decoding on inverse_transform
        bounds_only: Only apply during bounds adjustment

    Note:
        For sparse/intermittent data with many zeros, consider setting zero_offset
        or using alternative transformers like ReplaceConstant or CenterSplit.
    """

    def __init__(
        self,
        mode: str = "mu",
        mu: float = 100.0,
        A: float = 87.6,
        center: str = "median",
        scale_method: str = "mad",
        scale_factor: float = 3.0,
        min_scale: float = 1e-6,
        clip: bool = True,
        zero_offset: float = 0.0,
        fill_method: str = "interpolate",
        on_transform: bool = True,
        on_inverse: bool = True,
        bounds_only: bool = False,
        **kwargs,
    ):
        super().__init__(name="G711Scaler")
        self.mode = mode
        self.mu = float(mu)
        self.A = float(A)
        self.center = center
        self.scale_method = scale_method
        self.scale_factor = float(scale_factor)
        self.min_scale = float(min_scale)
        self.clip = bool(clip)
        self.zero_offset = float(zero_offset)
        self.fill_method = fill_method
        self.on_transform = on_transform
        self.on_inverse = on_inverse
        self.bounds_only = bounds_only
        self._center = None
        self._scale = None
        self.adjustment = None

    def fit(self, df):
        self.columns = df.columns
        frame = self._prepare_frame(df)
        # use g711_encode to compute and store center/scale without consuming time
        _, c, s = g711_encode(
            frame.to_numpy(),
            mode=self.mode,
            mu=self.mu,
            A=self.A,
            center=self.center,
            scale_method=self.scale_method,
            scale_factor=self.scale_factor,
            min_scale=self.min_scale,
            clip=self.clip,
            zero_offset=self.zero_offset,
        )
        # store with 2D shape for broadcasting
        if c.size == 0:
            self._center = np.zeros((1, 0))
            self._scale = np.ones((1, 0))
        else:
            self._center = c
            self._scale = s
        return self

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.apply(pd.to_numeric, errors="coerce")
        if self.fill_method == "interpolate":
            frame = frame.interpolate(limit_direction="both")
        elif self.fill_method == "ffill":
            frame = frame.ffill().bfill()
        elif self.fill_method == "bfill":
            frame = frame.bfill().ffill()
        elif self.fill_method == "median":
            frame = frame.fillna(frame.median())
        else:
            frame = frame.fillna(0.0)
        return frame.fillna(0.0)

    def transform(self, df):
        if not self.on_transform:
            return df
        if df.empty:
            return df
        frame = self._prepare_frame(df)
        arr = frame.to_numpy(dtype=float)

        # Apply zero offset if specified
        if self.zero_offset != 0.0:
            arr = arr + self.zero_offset

        # Normalize using stored center/scale and encode (consistent EPS)
        x = (arr - self._center) / (self._scale + 1e-12)
        if self.clip:
            x = np.clip(x, -1.0, 1.0)
        m = self.mode.lower()
        if m in ("mu", "mulaw", "u", "ulaw"):
            encoded = _mu_law_compress(x, mu=self.mu)
        elif m in ("a", "alaw"):
            encoded = _a_law_compress(x, A=self.A)
        else:
            raise ValueError(f"Unknown G711 mode: {self.mode}")
        return pd.DataFrame(encoded, index=df.index, columns=df.columns)

    def inverse_transform(self, df, trans_method: str = "forecast", adjustment=None):
        if not self.on_inverse:
            return df
        if self.bounds_only and adjustment is None:
            self.adjustment = True
            return df
        if df.empty:
            return df
        arr = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        decoded = g711_decode(
            arr,
            center=self._center,
            scale=self._scale,
            mode=self.mode,
            mu=self.mu,
            A=self.A,
            zero_offset=self.zero_offset,
        )
        return pd.DataFrame(decoded, index=df.index, columns=self.columns)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        mode = random.choices(["mu", "a"], [0.7, 0.3])[0]
        return {
            "mode": mode,
            # Reduced mu values to lessen non-linear dampening
            # Favor lower values (50-150) for better forecast amplitude preservation
            "mu": random.choices(
                [50.0, 75.0, 100.0, 150.0, 200.0, 255.0],
                [0.25, 0.25, 0.2, 0.15, 0.1, 0.05],
            )[0]
            if mode == "mu"
            else 100.0,
            "A": random.choice([50.0, 75.0, 87.6, 100.0]) if mode == "a" else 87.6,
            "center": random.choices(["median", "mean"], [0.8, 0.2])[0],
            "scale_method": random.choices(
                ["mad", "std", "maxabs", "percentile"], [0.5, 0.2, 0.1, 0.2]
            )[0],
            "scale_factor": random.uniform(1.5, 3.5),
            "min_scale": 1e-6,
            "clip": random.choice([True, True, False]),
            "zero_offset": random.choice([0.0, 0.0, 0.0, 1e-6, 1e-8]),
            "fill_method": random.choices(
                ["interpolate", "ffill", "median"], [0.6, 0.3, 0.1]
            )[0],
            "on_transform": True,
            "on_inverse": True,
            "bounds_only": False,
        }


class G726Filter(EmptyTransformer):
    """Adaptive differential PCM smoothing inspired by the G.726 codec.

    Implements proper G.726-style adaptive filtering with:
    - 2-pole/6-zero adaptive predictor with decision-directed updates (not unlike an ARMA (2,6) model)
    - Non-uniform quantization optimized for time series
    - Fast/slow scale factor adaptation with speed switching
    """

    def __init__(
        self,
        quant_bits: int = 4,
        adaptation_rate: float = 0.96,
        prediction_alpha: float = 0.92,
        floor_step: float = 0.01,
        dynamic_range: float = 1.5,
        blend: float = 0.15,
        noise_gate: float = 0.0,
        fill_method: str = "interpolate",
        on_transform: bool = True,
        on_inverse: bool = False,
        bounds_only: bool = False,
        quantizer: str = "uniform",
        use_adaptive_predictor: bool = True,
        predictor_leak: float = 0.9999,
        **kwargs,
    ):
        super().__init__(name="G726Filter")
        self.quant_bits = quant_bits
        self.adaptation_rate = adaptation_rate
        self.prediction_alpha = prediction_alpha
        self.floor_step = floor_step
        self.dynamic_range = dynamic_range
        self.blend = blend
        self.noise_gate = noise_gate
        self.fill_method = fill_method
        self.on_transform = on_transform
        self.on_inverse = on_inverse
        self.bounds_only = bounds_only
        self.quantizer = quantizer
        self.use_adaptive_predictor = use_adaptive_predictor
        self.predictor_leak = predictor_leak
        self.adjustment = None

    def fit(self, df):
        self.columns = df.columns
        return self

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.apply(pd.to_numeric, errors="coerce")
        if self.fill_method == "interpolate":
            frame = frame.interpolate(limit_direction="both")
        elif self.fill_method == "ffill":
            frame = frame.ffill().bfill()
        elif self.fill_method == "bfill":
            frame = frame.bfill().ffill()
        elif self.fill_method == "median":
            frame = frame.fillna(frame.median())
        else:
            frame = frame.fillna(0.0)
        return frame.fillna(0.0)

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        prepared = self._prepare_frame(df)
        filtered = g726_adpcm_filter(
            prepared.to_numpy(),
            quant_bits=self.quant_bits,
            adaptation_rate=self.adaptation_rate,
            prediction_alpha=self.prediction_alpha,
            floor_step=self.floor_step,
            dynamic_range=self.dynamic_range,
            blend=self.blend,
            noise_gate=self.noise_gate,
            quantizer=self.quantizer,
            use_adaptive_predictor=self.use_adaptive_predictor,
            predictor_leak=self.predictor_leak,
        )
        return pd.DataFrame(filtered, index=df.index, columns=df.columns)

    def transform(self, df):
        if self.on_transform:
            return self._filter(df)
        return df

    def inverse_transform(self, df, trans_method: str = "forecast", adjustment=None):
        if self.on_inverse:
            if not self.bounds_only or (self.bounds_only and adjustment is not None):
                return self._filter(df)
            self.adjustment = True
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        selection = random.choices([True, False], [0.75, 0.25])[0]
        quant_bits = random.choices([3, 4, 5], [0.2, 0.6, 0.2])[0]
        use_adaptive = random.choices([True, False], [0.8, 0.2])[0]
        params = {
            "quant_bits": quant_bits,
            "adaptation_rate": random.uniform(0.9, 0.99),
            "prediction_alpha": random.uniform(0.7, 0.97),
            "floor_step": random.uniform(0.002, 0.05),
            "dynamic_range": random.uniform(0.5, 2.5),
            "blend": random.uniform(0.0, 0.3),
            "noise_gate": random.choice([0.0, random.uniform(0.0, 0.1)]),
            "fill_method": random.choices(
                ["interpolate", "ffill", "median"], [0.6, 0.3, 0.1]
            )[0],
            "on_transform": selection,
            "on_inverse": not selection,
            "quantizer": random.choices(["uniform", "nonuniform"], [0.4, 0.6])[0],
            "use_adaptive_predictor": use_adaptive,
            "predictor_leak": random.uniform(0.9995, 0.99995)
            if use_adaptive
            else 0.9999,
        }
        if not selection:
            params["bounds_only"] = random.choices([True, False], [0.2, 0.8])[0]
        else:
            params["bounds_only"] = False
        return params


__all__ = [
    "g711_encode",
    "g711_decode",
    "g726_adpcm_filter",
    "G711Scaler",
    "G726Filter",
]
