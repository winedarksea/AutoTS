import random
from functools import lru_cache
import numpy as np
import pandas as pd
from autots.tools.shaping import infer_frequency
from autots.tools.window_functions import chunk_reshape


def _compute_segment_statistics(series, changepoints):
    """
    Calculate segment boundaries and representative values for a series given changepoints.

    Parameters:
    series (pd.Series or array-like): Input data aligned to the training index.
    changepoints (array-like): Indices of changepoints relative to the non-NaN values.

    Returns:
    tuple(pd.Index, np.ndarray): Segment boundary index values and segment means.
    """
    if isinstance(series, pd.Series):
        index = series.index
        values = series.to_numpy(dtype=float, copy=False)
    else:
        series = pd.Series(series)
        index = series.index
        values = series.to_numpy(dtype=float, copy=False)

    if len(series) == 0:
        return pd.Index([]), np.array([], dtype=float)

    mask = np.isfinite(values)
    if not mask.any():
        # Default to zeros if no finite values are available
        return pd.Index([index[0]]), np.array([0.0], dtype=float)

    valid_positions = np.flatnonzero(mask)
    filtered_values = values[mask]

    changepoints = np.asarray(changepoints if changepoints is not None else [], dtype=int)
    if changepoints.size > 0:
        changepoints = changepoints[(changepoints > 0) & (changepoints < len(filtered_values))]
        changepoints = np.unique(changepoints)
        # Validate boundaries are monotonically increasing
        if changepoints.size > 1 and not np.all(np.diff(changepoints) > 0):
            changepoints = np.unique(changepoints)
    boundaries = np.concatenate(([0], changepoints, [len(filtered_values)]))

    # Vectorized computation of segment means
    segment_lengths = np.diff(boundaries)
    if np.any(segment_lengths == 0):
        # Handle edge case of zero-length segments
        segment_breaks = []
        segment_means = []
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            segment_slice = filtered_values[start:end]
            if segment_slice.size == 0:
                if segment_means:
                    mean_val = segment_means[-1]
                else:
                    mean_val = np.nanmean(filtered_values)
            else:
                mean_val = np.nanmean(segment_slice)
            if np.isnan(mean_val):
                mean_val = 0.0

            if start < len(valid_positions):
                start_position = valid_positions[start]
            else:
                start_position = valid_positions[-1]
            segment_breaks.append(index[start_position])
            segment_means.append(float(mean_val))
    else:
        # Vectorized path for normal case
        segment_sums = np.add.reduceat(filtered_values, boundaries[:-1])
        segment_means_arr = segment_sums / segment_lengths
        
        # Handle NaN values
        nan_mask = np.isnan(segment_means_arr)
        if np.any(nan_mask):
            fallback_mean = np.nanmean(filtered_values) if np.any(np.isfinite(filtered_values)) else 0.0
            segment_means_arr[nan_mask] = fallback_mean
        
        # Map boundary indices to original index positions
        boundary_positions = boundaries[:-1]
        valid_boundary_positions = np.minimum(boundary_positions, len(valid_positions) - 1)
        segment_break_positions = valid_positions[valid_boundary_positions]
        
        segment_breaks = [index[pos] for pos in segment_break_positions]
        segment_means = segment_means_arr.tolist()


    if segment_breaks:
        segment_breaks[0] = index[0]
    else:
        segment_breaks = [index[0]]
        segment_means = [float(np.nanmean(filtered_values))]

    segment_breaks = pd.Index(segment_breaks)
    segment_means = np.array(segment_means, dtype=float)

    if segment_breaks.has_duplicates:
        keep_mask = ~segment_breaks.duplicated()
        segment_breaks = segment_breaks[keep_mask]
        segment_means = segment_means[keep_mask.to_numpy()]

    return segment_breaks, segment_means


def _evaluate_segment_trend(index, segment_breaks, segment_values):
    """
    Evaluate segment-wise trend values for a given index based on pre-computed statistics.

    Parameters:
    index (pd.Index): Target index (training or inference) to evaluate against.
    segment_breaks (pd.Index): Segment boundary points.
    segment_values (np.ndarray): Representative values per segment.

    Returns:
    np.ndarray: Trend values aligned with the provided index.
    """
    if segment_values is None or len(segment_values) == 0:
        return np.zeros(len(index), dtype=float)

    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    if not isinstance(segment_breaks, pd.Index):
        segment_breaks = pd.Index(segment_breaks)

    if len(segment_breaks) == 0:
        return np.zeros(len(index), dtype=float)

    segment_break_array = segment_breaks.values
    segment_values = np.asarray(segment_values, dtype=float)
    order = np.argsort(segment_break_array)
    segment_break_array = segment_break_array[order]
    segment_values = segment_values[order]

    index_array = index.values
    positions = np.searchsorted(segment_break_array, index_array, side='right') - 1
    positions = np.clip(positions, 0, len(segment_values) - 1)
    return segment_values[positions]


def _create_basic_changepoints(DTindex, changepoint_spacing, changepoint_distance_end):
    """
    Utility function for creating basic evenly spaced changepoint features.
    
    Parameters:
    DTindex (pd.DatetimeIndex): a datetimeindex
    changepoint_spacing (int): Distance between consecutive changepoints.
    changepoint_distance_end (int): Number of rows that belong to the final changepoint.
    
    Returns:
    pd.DataFrame: DataFrame containing changepoint features for linear regression.
    """
    n = len(DTindex)

    # Calculate the number of data points available for changepoints
    changepoint_range_end = n - changepoint_distance_end

    # Calculate the number of changepoints based on changepoint_spacing
    # Only place changepoints within the range [0, changepoint_range_end)
    changepoints = np.arange(0, changepoint_range_end, changepoint_spacing)

    # Ensure the last changepoint is exactly at changepoint_distance_end from the end
    changepoints = np.append(changepoints, changepoint_range_end)

    # Efficient concatenation approach to generate changepoint features
    res = []
    for i, cp in enumerate(changepoints):
        feature_name = f'changepoint_{i+1}'
        res.append(pd.Series(np.maximum(0, np.arange(n) - cp), name=feature_name))

    # Concatenate the changepoint features and set the index
    changepoint_features = pd.concat(res, axis=1)
    changepoint_features.index = DTindex

    return changepoint_features


def _calculate_segment_cost(segment_data, loss_function):
    """Helper to calculate cost of a segment."""
    if loss_function == 'l2':
        segment_mean = np.mean(segment_data)
        return np.sum((segment_data - segment_mean) ** 2)
    elif loss_function == 'l1':
        segment_median = np.median(segment_data)
        return np.sum(np.abs(segment_data - segment_median))
    elif loss_function == 'huber':
        delta = 1.345
        segment_median = np.median(segment_data)
        residuals = segment_data - segment_median
        abs_residuals = np.abs(residuals)
        return np.sum(np.where(abs_residuals <= delta, 
                                0.5 * residuals**2,
                                delta * (abs_residuals - 0.5 * delta)))
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

def _detect_pelt_changepoints(data, penalty=10, loss_function='l2', min_segment_length=1):
    """
    PELT (Pruned Exact Linear Time) changepoint detection algorithm.
    
    Parameters:
    data (array-like): Time series data
    penalty (float): Penalty parameter for model complexity
    loss_function (str): Loss function ('l2', 'l1', 'huber')
    min_segment_length (int): Minimum segment length
    
    Returns:
    np.array: Array of changepoint indices
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 2 * min_segment_length:
        return np.array([])
    
    # Initialize cost and optimal segmentation arrays
    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    cp = np.zeros(n + 1, dtype=int)
    
    # Pre-compute fast segment cost helpers
    if loss_function == 'l2':
        prefix_sum = np.concatenate(([0.0], np.cumsum(data)))
        prefix_sq_sum = np.concatenate(([0.0], np.cumsum(data ** 2)))

        def segment_cost(start, end):
            length = end - start
            if length <= 0:
                return 0.0
            segment_sum = prefix_sum[end] - prefix_sum[start]
            segment_sq_sum = prefix_sq_sum[end] - prefix_sq_sum[start]
            return float(segment_sq_sum - (segment_sum ** 2) / length)

    else:

        @lru_cache(maxsize=10000)
        def segment_cost(start, end):
            if end <= start:
                return 0.0
            return float(_calculate_segment_cost(data[start:end], loss_function))

    # PELT algorithm
    R = [0]  # Set of potential changepoints
    
    for t in range(1, n + 1):
        # Vectorize candidate evaluation when R is large
        if len(R) > 10:
            R_array = np.array(R, dtype=int)
            valid_mask = (t - R_array) >= min_segment_length
            if np.any(valid_mask):
                valid_R = R_array[valid_mask]
                costs = np.array([segment_cost(s, t) for s in valid_R], dtype=float)
                total_costs = F[valid_R] + costs + penalty
                best_idx = np.argmin(total_costs)
                F[t] = total_costs[best_idx]
                cp[t] = valid_R[best_idx]
            else:
                continue
        else:
            # Use list for small R (overhead of numpy not worth it)
            candidates = []
            for s in R:
                if t - s >= min_segment_length:
                    cost = segment_cost(s, t)
                    total_cost = F[s] + cost + penalty
                    candidates.append((total_cost, s))
            
            if candidates:
                F[t], cp[t] = min(candidates, key=lambda x: x[0])
            else:
                continue
        
        # Pruning step - keep only competitive changepoints
        # PELT pruning: keep s if F[s] could be part of optimal solution
        threshold = F[t]
        R_new = [s for s in R if F[s] <= threshold]
        R_new.append(t)
        R = R_new
    
    # Backtrack to find changepoints
    changepoints = []
    t = n
    while t > 0 and cp[t] != 0:
        changepoints.append(cp[t])
        t = cp[t]
    
    return np.array(sorted(changepoints)) if changepoints else np.array([])


def _detect_l1_trend_changepoints(data, lambda_reg=1.0, method='fused_lasso'):
    """
    L1 trend filtering for changepoint detection.
    
    Parameters:
    data (array-like): Time series data
    lambda_reg (float): Regularization parameter
    method (str): Method type ('fused_lasso', 'total_variation')
    
    Returns:
    tuple: (changepoints, fitted_trend)
    """
    try:
        from scipy.optimize import minimize
        from scipy.sparse import diags
    except ImportError:
        raise ImportError("scipy is required for L1 trend filtering")
    
    data = np.asarray(data).flatten()  # Ensure 1D array
    n = len(data)
    if n < 3:
        return np.array([]), data.copy()
    
    # For very small data, fall back to simple thresholding
    if n < 10:
        return _simple_threshold_changepoints(data), data.copy()
    
    # Create difference matrix for trend filtering
    if method == 'fused_lasso':
        # First-order differences (detects level changes)
        try:
            D = diags([1, -1], [0, 1], shape=(n-1, n)).toarray()
        except:
            # Fallback for very small arrays
            D = np.eye(n-1, n) - np.eye(n-1, n, k=1)
    elif method == 'total_variation':
        # Second-order differences (detects trend changes)
        if n < 4:
            return _simple_threshold_changepoints(data), data.copy()
        try:
            D = diags([1, -2, 1], [0, 1, 2], shape=(n-2, n)).toarray()
        except:
            # Fallback for very small arrays
            D = np.eye(n-2, n) - 2*np.eye(n-2, n, k=1) + np.eye(n-2, n, k=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Use a simpler approximation method instead of full L1 optimization
    # This avoids the complex optimization issues while still providing trend filtering
    try:
        fitted_trend = _approximate_l1_trend_filter(data, D, lambda_reg)
    except Exception:
        # Final fallback: return original data as trend
        return _simple_threshold_changepoints(data), data.copy()
    
    # Find changepoints from the fitted trend
    changepoints = _extract_changepoints_from_trend(fitted_trend, method)
    
    return changepoints, fitted_trend

def _simple_threshold_changepoints(data):
    """Simple changepoint detection using thresholding."""
    if len(data) < 3:
        return np.array([])
    
    # Calculate differences and find significant changes
    diffs = np.abs(np.diff(data))
    threshold = np.mean(diffs) + 2 * np.std(diffs)
    changepoints = np.where(diffs > threshold)[0] + 1
    
    return changepoints


def _detect_ewma_changepoints(
    data,
    lambda_param=0.2,
    control_limit=3.0,
    min_distance=5,
    normalize=True,
    two_sided=True,
    adaptive=False,
):
    """
    Detect changepoints using EWMA (Exponentially Weighted Moving Average) control charts.
    
    EWMA is effective at detecting small to moderate shifts in the process mean and
    responds more quickly to process changes than standard Shewhart charts. This 
    implementation includes several optimizations from the statistical process control
    literature including adaptive control limits and fast-initial-response (FIR).
    
    Parameters:
    data (array-like): Time series values.
    lambda_param (float): Smoothing parameter (0 < lambda <= 1). 
        Smaller values = more smoothing, better for detecting small persistent shifts.
        Larger values = less smoothing, better for detecting larger sudden shifts.
        Recommended: 0.2 for small shifts, 0.4-0.6 for moderate shifts.
        Common industry standard: 0.2.
    control_limit (float): Number of standard deviations for control limits.
        Higher values = fewer, more significant changepoints.
        Recommended: 2.5-3.5 for balance of sensitivity and specificity.
        Common industry standard: 3.0.
    min_distance (int): Minimum distance between successive changepoints.
        Prevents clustering of detections. Recommended: 5-10% of series length.
    normalize (bool): Whether to z-score the data before applying EWMA.
        Recommended: True for comparing across different series.
    two_sided (bool): Whether to detect both upward and downward shifts.
        If False, only detects upward shifts.
    adaptive (bool): Use adaptive control limits that tighten over time.
        This implements Lucas & Saccucci (1990) fast initial response (FIR).
        Recommended: True for better performance in initial periods.
    
    Returns:
    np.ndarray: Indices of detected changepoints.
    
    References:
    - Lucas & Saccucci (1990): Exponentially weighted moving average control schemes
    - Roberts (1959): Control Chart Tests Based on Geometric Moving Averages
    - Hunter (1986): The exponentially weighted moving average
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return np.array([])
    
    # Normalize if requested
    series = data.copy()
    data_mean = np.mean(series)
    series -= data_mean
    
    if normalize:
        std = np.std(series)
        if std > 1e-8:
            series /= std
            sigma = 1.0  # Normalized standard deviation
        else:
            # Constant data - no changepoints
            return np.array([])
    else:
        sigma = np.std(series)
        if sigma < 1e-8:
            return np.array([])
    
    # Validate parameters
    if not (0 < lambda_param <= 1):
        raise ValueError(f"lambda_param must be in (0, 1], got {lambda_param}")
    
    # Adaptive min_distance
    effective_min_distance = max(min_distance, int(n * 0.02))
    
    # Initialize EWMA
    z = np.zeros(n)
    z[0] = series[0]  # Fast Initial Response (FIR): start at first observation
    
    # Calculate standard error of EWMA at each point
    # Lucas & Saccucci (1990) formula for time-varying control limits
    if adaptive:
        # Adaptive control limits that asymptotically approach the steady-state
        # This provides better detection in the initial periods
        def ewma_std_error(t):
            """Standard error of EWMA at time t."""
            factor = (1 - (1 - lambda_param) ** (2 * (t + 1)))
            return sigma * np.sqrt(lambda_param / (2 - lambda_param) * factor)
    else:
        # Constant control limits (steady-state approximation)
        steady_state_std = sigma * np.sqrt(lambda_param / (2 - lambda_param))
        def ewma_std_error(t):
            return steady_state_std
    
    changepoints = []
    in_signal_state = False
    signal_start = None
    
    # Calculate EWMA recursively and track absolute deviations
    abs_z = np.zeros(n)
    
    for t in range(1, n):
        # EWMA update: z_t = λ * x_t + (1 - λ) * z_{t-1}
        z[t] = lambda_param * series[t] + (1 - lambda_param) * z[t - 1]
        
        # Track absolute deviation for validation
        abs_z[t] = abs(z[t])
        
        # Calculate control limits for this time point
        ucl = control_limit * ewma_std_error(t)  # Upper control limit
        lcl = -ucl if two_sided else -np.inf  # Lower control limit
        
        # Check for out-of-control signal
        signal = False
        if z[t] > ucl:
            signal = True
            direction = 'up'
        elif two_sided and z[t] < lcl:
            signal = True
            direction = 'down'
        
        if signal:
            if not in_signal_state:
                # New signal detected
                in_signal_state = True
                signal_start = t
            # Continue accumulating signal
        else:
            if in_signal_state:
                # Signal ended - record changepoint at the start of the signal
                # This is more accurate than using the midpoint for EWMA
                changepoint_idx = signal_start
                
                # Enforce minimum distance
                if len(changepoints) == 0 or (changepoint_idx - changepoints[-1]) >= effective_min_distance:
                    changepoints.append(changepoint_idx)
                
                in_signal_state = False
                signal_start = None
    
    # Handle case where signal persists to the end
    if in_signal_state and signal_start is not None:
        changepoint_idx = signal_start
        if len(changepoints) == 0 or (changepoint_idx - changepoints[-1]) >= effective_min_distance:
            changepoints.append(changepoint_idx)
    
    if len(changepoints) == 0:
        return np.array([])
    
    # Post-processing: validate changepoints by checking for sustained changes
    # in the original data (not just EWMA) to reduce false positives
    validated_changepoints = []
    window_size = min(30, effective_min_distance, n // 10)
    
    for cp in changepoints:
        if window_size < 3:
            validated_changepoints.append(cp)
            continue
        
        # Check for sustained shift in the original data
        before_start = max(0, cp - window_size)
        before_window = series[before_start:cp]
        after_end = min(n, cp + window_size)
        after_window = series[cp:after_end]
        
        if len(before_window) > 0 and len(after_window) > 0:
            # Calculate mean values before and after
            mean_before = np.mean(before_window)
            mean_after = np.mean(after_window)
            
            # For normalized data, require meaningful change
            # Lower threshold than CUSUM since EWMA is already conservative
            mean_diff = abs(mean_after - mean_before)
            
            # Use a more lenient threshold for EWMA validation
            # EWMA already does smoothing, so we don't need as strict validation
            std_before = np.std(before_window) if len(before_window) > 1 else 1.0
            std_after = np.std(after_window) if len(after_window) > 1 else 1.0
            avg_std = (std_before + std_after) / 2
            
            # Accept changepoint if:
            # 1. Mean difference is substantial (> 0.15 for normalized data)
            # 2. Or if EWMA signal was very strong (> control_limit * 0.6)
            threshold_val = max(0.15, 0.2 * avg_std)
            ewma_signal_strength = abs(z[cp]) if cp < len(z) else 0
            
            if mean_diff > threshold_val or ewma_signal_strength > control_limit * 0.6:
                validated_changepoints.append(cp)
        else:
            validated_changepoints.append(cp)
    
    if len(validated_changepoints) == 0:
        return np.array([])
    
    return np.array(sorted(set(validated_changepoints)))


def _detect_cusum_changepoints(
    data,
    threshold=5.0,
    drift=0.0,
    min_distance=5,
    normalize=True,
):
    """
    Detect changepoints using a two-sided CUSUM procedure.

    Parameters:
    data (array-like): Time series values.
    threshold (float): Threshold for the cumulative sum to trigger a changepoint.
        Higher values = fewer, more significant changepoints. Recommended: 10-20 for
        normalized data. If normalize=False, scale threshold to ~2-3x the expected
        change magnitude.
    drift (float): Drift parameter to control sensitivity. Higher drift = less sensitive
        to small sustained changes. Recommended: 0.5 for normalized data, or ~10-20% of
        expected change magnitude.
    min_distance (int): Minimum distance between successive changepoints.
        Prevents clustering of detections. Recommended: 5-10% of series length.
    normalize (bool): Whether to z-score the data before applying CUSUM.
        Recommended: True for comparing across different series.

    Returns:
    np.ndarray: Indices of detected changepoints.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return np.array([])

    # Calculate statistics for adaptive parameters
    series = data.copy()
    data_mean = np.mean(series)
    series -= data_mean
    
    if normalize:
        std = np.std(series)
        if std > 1e-8:
            series /= std
        else:
            # Constant data - no changepoints
            return np.array([])
    
    # Adaptive drift: if drift is 0, use small default based on data
    effective_drift = drift
    if abs(drift) < 1e-10 and normalize:
        # For normalized data, use a small drift to reduce false positives
        effective_drift = 0.25  # Reduce sensitivity to noise
    
    # Adaptive min_distance: if very small, use percentage of data length
    effective_min_distance = max(min_distance, int(n * 0.02))  # At least 2% of series length

    g_pos = 0.0
    g_neg = 0.0
    changepoints = []
    candidate_triggers = []  # Store potential changepoints for validation

    for idx, value in enumerate(series):
        g_pos = max(0.0, g_pos + value - effective_drift)
        g_neg = min(0.0, g_neg + value + effective_drift)

        trigger = None
        if g_pos > threshold:
            trigger = idx
            candidate_triggers.append((idx, 'pos'))
            g_pos = 0.0
            g_neg = 0.0
        elif g_neg < -threshold:
            trigger = idx
            candidate_triggers.append((idx, 'neg'))
            g_pos = 0.0
            g_neg = 0.0

        if trigger is not None:
            # Enforce minimum distance between changepoints
            if len(changepoints) == 0 or (trigger - changepoints[-1]) >= effective_min_distance:
                changepoints.append(trigger)

    if len(changepoints) == 0:
        return np.array([])
    
    # Post-processing: validate changepoints by checking for sustained level changes
    # This helps reduce false positives from temporary spikes
    validated_changepoints = []
    for cp in changepoints:
        # Check if there's a sustained change after this point
        # Compare mean before and after (using windows to avoid edge effects)
        window_size = min(30, effective_min_distance, cp, n - cp - 1)
        if window_size < 3:
            # Not enough data to validate, but keep it (edge case)
            validated_changepoints.append(cp)
            continue
        
        # Calculate means in windows before and after
        before_window = series[max(0, cp - window_size):cp]
        after_window = series[cp:min(n, cp + window_size)]
        
        if len(before_window) > 0 and len(after_window) > 0:
            mean_diff = abs(np.mean(after_window) - np.mean(before_window))
            # For normalized data, require at least 0.2 std change (less strict)
            # The mean difference should be larger than noise level
            std_before = np.std(before_window) if len(before_window) > 1 else 1.0
            std_after = np.std(after_window) if len(after_window) > 1 else 1.0
            avg_std = (std_before + std_after) / 2
            
            # The mean difference should be larger than the typical variation within segments
            # Use max of 0.2 (for weak changes) or 0.3*avg_std (for noisy data)
            threshold_val = max(0.2, 0.3 * avg_std)
            if mean_diff > threshold_val:
                validated_changepoints.append(cp)
        else:
            validated_changepoints.append(cp)
    
    if len(validated_changepoints) == 0:
        return np.array([])
    return np.array(sorted(set(validated_changepoints)))


def _build_series_matrix(series_list):
    """Pad a list of 1D arrays into a 2D matrix with lengths."""
    if not series_list:
        return np.empty((0, 0), dtype=float), np.array([], dtype=int)
    lengths = np.array([len(arr) for arr in series_list], dtype=int)
    max_len = lengths.max(initial=0)
    matrix = np.zeros((len(series_list), max_len), dtype=float)
    for idx, arr in enumerate(series_list):
        matrix[idx, : lengths[idx]] = arr
    return matrix, lengths


def _vectorized_cusum_changepoints(
    series_list,
    threshold,
    drift,
    min_distance,
    normalize,
    min_segment_length,
):
    """Vectorized CUSUM detection across multiple series."""
    if not series_list:
        return []

    matrix, lengths = _build_series_matrix(series_list)
    max_len = matrix.shape[1]
    mask = np.arange(max_len)[None, :] < lengths[:, None]
    # Center series
    sums = (matrix * mask).sum(axis=1)
    means = sums / np.maximum(lengths, 1)
    series = matrix - means[:, None]
    series *= mask

    valid_mask = lengths >= 2 * np.maximum(min_segment_length, 1)

    if normalize:
        variances = (series ** 2).sum(axis=1) / np.maximum(lengths, 1)
        stds = np.sqrt(variances)
        safe = stds > 1e-8
        normalized = np.zeros_like(series)
        safe_idx = np.nonzero(safe)[0]
        if safe_idx.size:
            normalized[safe_idx] = series[safe_idx] / stds[safe_idx, None]
        normalized *= mask
        series = normalized
        valid_mask &= safe
    else:
        stds = np.sqrt((series ** 2).sum(axis=1) / np.maximum(lengths, 1))
        valid_mask &= stds > 1e-8

    n_series = len(series_list)
    g_pos = np.zeros(n_series, dtype=float)
    g_neg = np.zeros(n_series, dtype=float)
    effective_drift = drift if (abs(drift) >= 1e-10 or not normalize) else 0.25
    effective_min_distance = np.maximum(
        min_distance, np.maximum(1, (lengths * 0.02).astype(int))
    )

    last_cp = -effective_min_distance.astype(int)
    changepoint_lists = [[] for _ in range(n_series)]

    for idx in range(max_len):
        active = (idx < lengths) & valid_mask
        if not np.any(active):
            continue
        values = series[:, idx]
        g_pos[active] = np.maximum(0.0, g_pos[active] + values[active] - effective_drift)
        g_neg[active] = np.minimum(0.0, g_neg[active] + values[active] + effective_drift)

        triggered_pos = (g_pos > threshold) & active
        triggered_neg = (g_neg < -threshold) & active
        triggered = triggered_pos | triggered_neg

        if np.any(triggered):
            eligible = triggered & ((idx - last_cp) >= effective_min_distance)
            if np.any(eligible):
                eligible_idx = np.nonzero(eligible)[0]
                for series_idx in eligible_idx:
                    changepoint_lists[series_idx].append(idx)
                    last_cp[series_idx] = idx
            g_pos[triggered] = 0.0
            g_neg[triggered] = 0.0

    results = []
    for series_idx in range(n_series):
        cps = changepoint_lists[series_idx]
        if not cps:
            results.append(np.array([], dtype=int))
            continue

        series_length = lengths[series_idx]
        eff_min = effective_min_distance[series_idx]
        series_values = series[series_idx, :series_length]
        
        # Vectorized validation of changepoints
        if len(cps) > 0:
            cps_array = np.array(cps, dtype=int)
            window_size = min(30, eff_min, series_length // 10) if series_length > 0 else 3
            
            validated = []
            for cp in cps_array:
                ws = min(window_size, cp, series_length - cp - 1)
                if ws < 3:
                    validated.append(cp)
                    continue

                before_window = series_values[max(0, cp - ws):cp]
                after_window = series_values[cp:min(series_length, cp + ws)]

                if before_window.size == 0 or after_window.size == 0:
                    validated.append(cp)
                    continue

                mean_diff = abs(np.mean(after_window) - np.mean(before_window))
                std_before = np.std(before_window) if before_window.size > 1 else 1.0
                std_after = np.std(after_window) if after_window.size > 1 else 1.0
                avg_std = (std_before + std_after) / 2
                threshold_val = max(0.2, 0.3 * avg_std)

                if mean_diff > threshold_val:
                    validated.append(cp)

            if validated:
                # Vectorized min_distance filtering
                validated_array = np.array(validated, dtype=int)
                if len(validated_array) > 1:
                    diffs = np.diff(validated_array)
                    keep_mask = np.concatenate([[True], diffs >= eff_min])
                    filtered = validated_array[keep_mask]
                else:
                    filtered = validated_array
                results.append(filtered)
            else:
                results.append(np.array([], dtype=int))
        else:
            results.append(np.array([], dtype=int))

    return results


def _vectorized_ewma_changepoints(
    series_list,
    lambda_param,
    control_limit,
    min_distance,
    normalize,
    two_sided,
    adaptive,
    min_segment_length,
):
    """Vectorized EWMA detection across multiple series."""
    if not series_list:
        return []

    matrix, lengths = _build_series_matrix(series_list)
    max_len = matrix.shape[1]
    mask = np.arange(max_len)[None, :] < lengths[:, None]
    centered = matrix - ((matrix * mask).sum(axis=1) / np.maximum(lengths, 1))[:, None]
    centered *= mask

    n_series = len(series_list)
    sigma = np.sqrt((centered ** 2).sum(axis=1) / np.maximum(lengths, 1))
    valid_mask = sigma > 1e-8

    if normalize:
        normalized = np.zeros_like(centered)
        safe_idx = np.nonzero(valid_mask)[0]
        if safe_idx.size:
            normalized[safe_idx] = centered[safe_idx] / sigma[safe_idx, None]
        normalized *= mask
        centered = normalized
        sigma[:] = 1.0
        valid_mask = valid_mask
    else:
        valid_mask &= sigma > 1e-8

    centered[~valid_mask, :] = 0.0
    sigma[~valid_mask] = 1.0

    eff_min_distance = np.maximum(
        min_distance, np.maximum(1, (lengths * 0.02).astype(int))
    )
    last_cp = -eff_min_distance.astype(int)

    t_index = np.arange(max_len)
    if adaptive:
        factor = 1 - (1 - lambda_param) ** (2 * (t_index + 1))
        scale = np.sqrt(lambda_param / (2 - lambda_param))
        std_errors = sigma[:, None] * scale * np.sqrt(factor)[None, :]
    else:
        steady_state = sigma * np.sqrt(lambda_param / (2 - lambda_param))
        std_errors = np.broadcast_to(steady_state[:, None], (n_series, max_len))

    z = np.zeros_like(centered)
    if max_len > 0:
        z[:, 0] = centered[:, 0]

    in_signal = np.zeros(n_series, dtype=bool)
    signal_start = np.full(n_series, -1, dtype=int)
    changepoint_lists = [[] for _ in range(n_series)]

    for t in range(1, max_len):
        z[:, t] = lambda_param * centered[:, t] + (1 - lambda_param) * z[:, t - 1]
        active = (t < lengths) & valid_mask
        if not np.any(active):
            continue

        ucl = control_limit * std_errors[:, t]
        if two_sided:
            signal = ((z[:, t] > ucl) | (z[:, t] < -ucl)) & active
        else:
            signal = (z[:, t] > ucl) & active

        prev_signal = in_signal.copy()
        start_mask = signal & ~prev_signal
        signal_start[start_mask] = t

        end_mask = prev_signal & ~signal
        if np.any(end_mask):
            for idx in np.nonzero(end_mask)[0]:
                cp_idx = signal_start[idx]
                if cp_idx >= 0 and (cp_idx - last_cp[idx]) >= eff_min_distance[idx]:
                    changepoint_lists[idx].append(cp_idx)
                    last_cp[idx] = cp_idx
                signal_start[idx] = -1

        in_signal = signal

    # Handle ongoing signals
    if np.any(in_signal):
        ongoing_idx = np.nonzero(in_signal)[0]
        for idx in ongoing_idx:
            cp_idx = signal_start[idx]
            if cp_idx >= 0 and (cp_idx - last_cp[idx]) >= eff_min_distance[idx]:
                changepoint_lists[idx].append(cp_idx)
                last_cp[idx] = cp_idx

    results = []
    for idx in range(n_series):
        cps = changepoint_lists[idx]
        if not cps:
            results.append(np.array([], dtype=int))
            continue

        n = lengths[idx]
        eff_min = eff_min_distance[idx]
        series_values = centered[idx, :n]
        
        # Vectorized validation
        if len(cps) > 0:
            cps_array = np.array(cps, dtype=int)
            window_size = min(30, eff_min, max(1, n // 10))

            validated = []
            for cp in cps_array:
                before_start = max(0, cp - window_size)
                after_end = min(n, cp + window_size)

                before_window = series_values[before_start:cp]
                after_window = series_values[cp:after_end]

                if before_window.size == 0 or after_window.size == 0:
                    validated.append(cp)
                    continue

                mean_diff = abs(np.mean(after_window) - np.mean(before_window))
                std_before = np.std(before_window) if before_window.size > 1 else 1.0
                std_after = np.std(after_window) if after_window.size > 1 else 1.0
                avg_std = (std_before + std_after) / 2
                threshold_val = max(0.15, 0.2 * avg_std)
                ewma_signal_strength = abs(z[idx, cp]) if cp < n else 0.0

                if mean_diff > threshold_val or ewma_signal_strength > control_limit * 0.6:
                    validated.append(cp)

            if validated:
                # Vectorized min_distance filtering
                validated_array = np.array(validated, dtype=int)
                if len(validated_array) > 1:
                    diffs = np.diff(validated_array)
                    keep_mask = np.concatenate([[True], diffs >= eff_min])
                    filtered = validated_array[keep_mask]
                else:
                    filtered = validated_array
                results.append(filtered)
            else:
                results.append(np.array([], dtype=int))
        else:
            results.append(np.array([], dtype=int))

    return results


def _approximate_l1_trend_filter_batch(data_block, method, lambda_reg):
    """Vectorized approximate L1 trend filtering for multiple series."""
    batch_size, n = data_block.shape
    if method == 'fused_lasso':
        if n < 2:
            return data_block.astype(float)
        D = np.eye(n - 1, n, dtype=float) - np.eye(n - 1, n, k=1, dtype=float)
    else:
        if n < 3:
            return data_block.astype(float)
        D = (
            np.eye(n - 2, n, dtype=float)
            - 2 * np.eye(n - 2, n, k=1, dtype=float)
            + np.eye(n - 2, n, k=2, dtype=float)
        )

    identity = np.eye(n, dtype=float)
    results = np.empty_like(data_block, dtype=float)
    m = D.shape[0]
    max_elements = 4_000_000
    chunk_size = max(1, max_elements // max(1, m * n))

    for start in range(0, batch_size, chunk_size):
        end = min(batch_size, start + chunk_size)
        chunk = data_block[start:end].astype(float, copy=True)
        x_chunk = chunk.copy()

        for _ in range(3):
            DX = x_chunk @ D.T
            weights = 1.0 / (np.abs(DX) + 1e-6)
            sqrt_weights = np.sqrt(lambda_reg * weights)
            D_weighted = D[None, :, :] * sqrt_weights[:, :, None]
            A = identity[None, :, :] + np.matmul(
                np.transpose(D_weighted, (0, 2, 1)), D_weighted
            )
            try:
                x_chunk = np.linalg.solve(A, chunk)
            except np.linalg.LinAlgError:
                x_chunk = np.array([_simple_smooth(row, lambda_reg) for row in chunk])
                break

        results[start:end] = x_chunk

    return results


def _extract_changepoints_from_trend_batch(fitted_trends, method, min_segment_length):
    """Extract changepoints from fitted trends for multiple series."""
    if fitted_trends.size == 0:
        return []

    n_series, n = fitted_trends.shape
    if method == 'fused_lasso':
        if n < 2:
            return [np.array([], dtype=int) for _ in range(n_series)]
        differences = np.abs(np.diff(fitted_trends, axis=1))
        shift = 1
    else:
        if n < 3:
            return [np.array([], dtype=int) for _ in range(n_series)]
        differences = np.abs(np.diff(fitted_trends, n=2, axis=1))
        shift = 2

    mean_diff = differences.mean(axis=1)
    std_diff = differences.std(axis=1)

    results = []
    for idx in range(n_series):
        if differences.shape[1] == 0 or std_diff[idx] == 0:
            results.append(np.array([], dtype=int))
            continue

        threshold = mean_diff[idx] + 1.5 * std_diff[idx]
        candidates = np.where(differences[idx] > threshold)[0] + shift
        candidates = candidates[(candidates > 2) & (candidates < n - 2)]

        if candidates.size == 0:
            results.append(np.array([], dtype=int))
            continue

        # Vectorized min_distance filtering
        if candidates.size > 1:
            diffs = np.diff(candidates)
            keep_mask = np.concatenate([[True], diffs >= min_segment_length])
            filtered = candidates[keep_mask]
        else:
            filtered = candidates

        results.append(np.array(sorted(set(filtered.tolist())), dtype=int))

    return results


def _vectorized_l1_detection(names, series_list, lambda_reg, method_key, min_segment_length):
    """Vectorized L1 (fused lasso / total variation) detection across series."""
    results = {}
    if not series_list:
        return results

    method = 'fused_lasso' if method_key == 'l1_fused_lasso' else 'total_variation'
    min_required = 4 if method == 'total_variation' else 3
    
    # Vectorized length calculation
    lengths = np.array([len(arr) for arr in series_list], dtype=int)
    
    # Handle short series
    short_mask = lengths < min_required
    if np.any(short_mask):
        short_indices = np.where(short_mask)[0]
        for idx in short_indices:
            arr = series_list[idx]
            if len(arr) >= 3:
                cp = _simple_threshold_changepoints(arr)
            else:
                cp = np.array([], dtype=int)
            results[names[idx]] = {
                'changepoints': cp,
                'fitted': arr.astype(float, copy=False),
            }
    
    # Group eligible series by length
    eligible_mask = lengths >= min_required
    if not np.any(eligible_mask):
        return results
        
    eligible_indices = np.where(eligible_mask)[0]
    eligible_lengths = lengths[eligible_indices]
    
    # Group by unique lengths
    unique_lengths, inverse_indices = np.unique(eligible_lengths, return_inverse=True)
    
    for group_idx, length in enumerate(unique_lengths):
        # Get all series indices with this length
        indices_in_group = eligible_indices[inverse_indices == group_idx]
        
        data_block = np.vstack([series_list[i].astype(float, copy=False) for i in indices_in_group])
        try:
            fitted_block = _approximate_l1_trend_filter_batch(
                data_block, method, lambda_reg
            )
        except Exception:
            fitted_block = data_block.copy()

        cp_lists = _extract_changepoints_from_trend_batch(
            fitted_block, method, min_segment_length
        )

        for row, series_idx in enumerate(indices_in_group):
            name = names[series_idx]
            results[name] = {
                'changepoints': cp_lists[row],
                'fitted': fitted_block[row],
            }

    return results


def _prepare_autoencoder_windows(data, window_size):
    """Create overlapping windows for autoencoder training."""
    series = np.asarray(data, dtype=float)
    n = len(series)
    if n == 0:
        return np.empty((0, window_size)), np.array([], dtype=int)

    window_size = max(1, min(int(window_size), n))
    if window_size == 1:
        return series.reshape(-1, 1), np.arange(n, dtype=int)

    windows = chunk_reshape(
        series,
        window_size=window_size,
        chunk_size=max(1, min(128, series.shape[0])),
        dtype=np.float64,
    )

    if windows.size == 0:
        return np.empty((0, window_size)), np.array([], dtype=int)

    indices = np.arange(window_size - 1, window_size - 1 + windows.shape[0], dtype=int)
    return windows, indices


def _detect_autoencoder_changepoints(
    data,
    method_params=None,
    min_distance=5,
):
    """
    Detect changepoints using an autoencoder-based anomaly signal.

    Parameters:
    data (array-like): Time series values.
    method_params (dict): Parameters for autoencoder training and post-processing.
        - window_size (int): Size of sliding windows for autoencoder input. 
          Default: min(30, n//5). Larger windows capture more context but require more data.
        - smoothing_window (int): Window size for smoothing anomaly scores. Default: 3.
        - contamination (float): Expected proportion of anomalies. Default: 0.1.
        - epochs (int): Training epochs for autoencoder. Default: 40.
        - normalize_scores (bool): Whether to normalize scores. Default: True.
        - use_anomaly_flags (bool): Use binary anomaly flags vs continuous scores. Default: True.
        - score_threshold (float): Manual threshold for scores (overrides quantile). Default: None.
        - score_quantile (float): Quantile for automatic threshold. Default: 0.95.
    min_distance (int): Minimum separation between detected changepoints.

    Returns:
    tuple: (np.ndarray changepoints, np.ndarray anomaly_scores)
    """
    if method_params is None:
        method_params = {}
    
    try:
        from autots.tools.autoencoder import vae_outliers, torch_available
    except ImportError as exc:
        raise ImportError("Autoencoder changepoint detection requires the autoencoder tools.") from exc
    
    if not torch_available:
        raise ImportError("PyTorch is required for autoencoder-based changepoint detection.")
    
    series = np.asarray(data, dtype=float)
    n = len(series)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    # Validate and set window size
    window_size = method_params.get('window_size', max(3, min(30, n // 5 or 3)))
    window_size = max(3, min(int(window_size), n))  # Ensure at least 3, at most n
    
    # Check if we have enough data
    if n < window_size:
        # Not enough data for autoencoder - return no changepoints
        import warnings
        warnings.warn(
            f"Data length ({n}) is less than window_size ({window_size}). "
            "Autoencoder changepoint detection requires more data.",
            UserWarning
        )
        return np.array([], dtype=int), np.zeros(n, dtype=float)
    
    smoothing_window = int(method_params.get('smoothing_window', 3))
    use_flags = method_params.get('use_anomaly_flags', True)
    normalize_scores = method_params.get('normalize_scores', True)
    score_threshold = method_params.get('score_threshold', None)
    score_quantile = method_params.get('score_quantile', 0.95)
    
    windows, indices = _prepare_autoencoder_windows(series, window_size)
    if windows.size == 0:
        return np.array([], dtype=int), np.zeros(n, dtype=float)
    
    df_windows = pd.DataFrame(windows)
    
    detection_keys = {
        'window_size',
        'smoothing_window',
        'use_anomaly_flags',
        'normalize_scores',
        'score_threshold',
        'score_quantile',
        'min_distance',
    }
    vae_params = {k: v for k, v in method_params.items() if k not in detection_keys}
    if 'contamination' not in vae_params:
        vae_params['contamination'] = method_params.get('contamination', 0.1)
    
    anomalies_df, scores_df = vae_outliers(df_windows, method_params=vae_params)
    scores = scores_df.iloc[:, 0].values.astype(float)
    
    if smoothing_window > 1:
        scores = pd.Series(scores).rolling(window=smoothing_window, min_periods=1, center=False).mean().values
    
    # Improved normalization with better numerical stability
    if normalize_scores:
        score_std = np.std(scores)
        if score_std > 1e-6:  # Use larger threshold for stability
            # Use robust statistics for better outlier handling
            score_median = np.median(scores)
            # MAD (Median Absolute Deviation) for robust scaling
            mad = np.median(np.abs(scores - score_median))
            if mad > 1e-6:
                # Use MAD-based normalization (more robust to outliers)
                scores = (scores - score_median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std for normal dist
            else:
                # Fall back to mean/std if MAD is too small
                scores = (scores - np.mean(scores)) / (score_std + 1e-6)
        else:
            # Very low variance - likely constant or near-constant scores
            # Keep original scores but centered
            scores = scores - np.mean(scores)
    
    if use_flags and 'anomaly' in anomalies_df.columns:
        anomaly_mask = anomalies_df.iloc[:, 0].values == -1
    else:
        if score_threshold is None:
            score_threshold = np.quantile(scores, score_quantile)
        anomaly_mask = scores >= score_threshold
    
    candidate_indices = indices[anomaly_mask]
    candidate_scores = scores[anomaly_mask]
    
    min_distance = max(1, int(min_distance))
    filtered = []
    for idx, sc in zip(candidate_indices, candidate_scores):
        if not filtered:
            filtered.append((idx, sc))
            continue
        last_idx, last_sc = filtered[-1]
        if idx - last_idx < min_distance:
            if sc > last_sc:
                filtered[-1] = (idx, sc)
        else:
            filtered.append((idx, sc))
    
    changepoints = np.array([idx for idx, _ in filtered], dtype=int) if filtered else np.array([], dtype=int)
    
    anomaly_scores = np.zeros(n, dtype=float)
    if len(indices) > 0:
        anomaly_scores[indices] = scores
        first_fill = scores[0] if len(scores) > 0 else 0.0
        anomaly_scores[:indices[0]] = first_fill
    
    return changepoints, anomaly_scores


def _detect_autoencoder_changepoints_vectorized(
    series_names,
    series_list,
    method_params,
    fallback_min_distance,
):
    """Run autoencoder changepoint detection once for multiple series."""
    results = {}
    if method_params is None:
        method_params = {}

    try:
        from autots.tools.autoencoder import vae_outliers, torch_available
    except ImportError as exc:
        raise ImportError("Autoencoder changepoint detection requires the autoencoder tools.") from exc

    if not torch_available:
        raise ImportError("PyTorch is required for autoencoder-based changepoint detection.")

    lengths = [len(arr) for arr in series_list]
    if not lengths:
        return results

    if 'window_size' in method_params:
        window_size = int(method_params.get('window_size', 3))
    else:
        default_sizes = [
            max(3, min(30, (length // 5) if (length // 5) > 0 else 3))
            for length in lengths
            if length > 0
        ]
        window_size = min(default_sizes) if default_sizes else 3
    window_size = max(3, window_size)

    smoothing_window = int(method_params.get('smoothing_window', 3))
    use_flags = method_params.get('use_anomaly_flags', True)
    normalize_scores = method_params.get('normalize_scores', True)
    score_threshold = method_params.get('score_threshold', None)
    score_quantile = method_params.get('score_quantile', 0.95)
    min_distance = max(1, int(method_params.get('min_distance', fallback_min_distance)))

    detection_keys = {
        'window_size',
        'smoothing_window',
        'use_anomaly_flags',
        'normalize_scores',
        'score_threshold',
        'score_quantile',
        'min_distance',
    }
    vae_params = {k: v for k, v in method_params.items() if k not in detection_keys}
    if 'contamination' not in vae_params:
        vae_params['contamination'] = method_params.get('contamination', 0.1)

    all_windows = []
    series_window_indices = {}

    for idx, arr in enumerate(series_list):
        name = series_names[idx]
        if len(arr) < window_size:
            results[name] = {
                'changepoints': np.array([], dtype=int),
                'fitted': np.zeros(len(arr), dtype=float),
            }
            continue

        windows, indices = _prepare_autoencoder_windows(arr, window_size)
        if windows.size == 0 or indices.size == 0:
            results[name] = {
                'changepoints': np.array([], dtype=int),
                'fitted': np.zeros(len(arr), dtype=float),
            }
            continue

        all_windows.append(windows)
        series_window_indices[idx] = indices

    if not all_windows:
        return results

    window_matrix = np.vstack(all_windows)
    df_windows = pd.DataFrame(window_matrix)

    anomalies_df, scores_df = vae_outliers(df_windows, method_params=vae_params)
    scores_array = scores_df.iloc[:, 0].values.astype(float)
    if use_flags and 'anomaly' in anomalies_df.columns:
        anomaly_flags = anomalies_df.iloc[:, 0].values
    else:
        anomaly_flags = None

    offset = 0
    for idx, arr in enumerate(series_list):
        name = series_names[idx]
        if idx not in series_window_indices:
            # Already populated (insufficient data)
            if name not in results:
                results[name] = {
                    'changepoints': np.array([], dtype=int),
                    'fitted': arr.astype(float, copy=False),
                }
            continue

        count = len(series_window_indices[idx])
        series_scores = scores_array[offset:offset + count].astype(float, copy=True)
        if anomaly_flags is not None:
            series_flags = anomaly_flags[offset:offset + count]
        else:
            series_flags = None
        offset += count

        if smoothing_window > 1 and series_scores.size > 0:
            series_scores = (
                pd.Series(series_scores)
                .rolling(window=smoothing_window, min_periods=1)
                .mean()
                .values
            )

        if normalize_scores and series_scores.size > 0:
            score_std = np.std(series_scores)
            if score_std > 1e-6:
                score_median = np.median(series_scores)
                mad = np.median(np.abs(series_scores - score_median))
                if mad > 1e-6:
                    series_scores = (series_scores - score_median) / (1.4826 * mad)
                else:
                    series_scores = (series_scores - np.mean(series_scores)) / (score_std + 1e-6)
            else:
                series_scores = series_scores - np.mean(series_scores)

        if use_flags and series_flags is not None:
            anomaly_mask = series_flags == -1
        else:
            if series_scores.size == 0:
                anomaly_mask = np.array([], dtype=bool)
            else:
                threshold_value = (
                    score_threshold
                    if score_threshold is not None
                    else np.quantile(series_scores, score_quantile)
                )
                anomaly_mask = series_scores >= threshold_value

        indices = series_window_indices[idx]
        candidate_indices = indices[anomaly_mask] if anomaly_mask.size else np.array([], dtype=int)
        candidate_scores = series_scores[anomaly_mask] if anomaly_mask.size else np.array([], dtype=float)

        filtered = []
        for cp_idx, sc in zip(candidate_indices, candidate_scores):
            if not filtered:
                filtered.append((cp_idx, sc))
                continue
            last_idx, last_score = filtered[-1]
            if cp_idx - last_idx < min_distance:
                if sc > last_score:
                    filtered[-1] = (cp_idx, sc)
            else:
                filtered.append((cp_idx, sc))

        changepoints = (
            np.array([item[0] for item in filtered], dtype=int) if filtered else np.array([], dtype=int)
        )

        anomaly_scores = np.zeros(len(arr), dtype=float)
        if indices.size > 0:
            anomaly_scores[indices] = series_scores
            first_fill = series_scores[0]
            anomaly_scores[: indices[0]] = first_fill

        results[name] = {
            'changepoints': changepoints,
            'fitted': anomaly_scores,
        }

    return results


def create_changepoint_features(
    DTindex, 
    changepoint_spacing=60, 
    changepoint_distance_end=120,
    method='basic',
    params=None,
    data=None
):
    """
    Creates a feature set for estimating trend changepoints using various algorithms.

    Parameters:
    DTindex (pd.DatetimeIndex): a datetimeindex
    changepoint_spacing (int): Distance between consecutive changepoints (legacy, for basic method).
    changepoint_distance_end (int): Number of rows that belong to the final changepoint (legacy, for basic method).
    method (str): Method for changepoint detection ('basic', 'pelt', 'l1_fused_lasso', 'l1_total_variation', 'cusum', 'ewma', 'autoencoder')
    params (dict): Additional parameters for the chosen method
    data (array-like): Time series data (required for advanced methods)

    Returns:
    pd.DataFrame: DataFrame containing changepoint features for linear regression.
    """
    if params is None:
        params = {}
    
    if method == 'basic':
        return _create_basic_changepoints(DTindex, changepoint_spacing, changepoint_distance_end)
    
    elif method == 'pelt':
        if data is None:
            raise ValueError("Data is required for PELT changepoint detection")
        penalty = params.get('penalty', 10)
        loss_function = params.get('loss_function', 'l2')
        min_segment_length = params.get('min_segment_length', 1)
        return _create_pelt_changepoint_features(DTindex, data, penalty, loss_function, min_segment_length)
    
    elif method in ['l1_fused_lasso', 'l1_total_variation']:
        if data is None:
            raise ValueError("Data is required for L1 trend filtering")
        lambda_reg = params.get('lambda_reg', 1.0)
        l1_method = 'fused_lasso' if method == 'l1_fused_lasso' else 'total_variation'
        return _create_l1_changepoint_features(DTindex, data, lambda_reg, l1_method)
    
    elif method == 'cusum':
        if data is None:
            raise ValueError("Data is required for CUSUM changepoint detection")
        threshold = params.get('threshold', 5.0)
        drift = params.get('drift', 0.0)
        min_distance = params.get('min_distance', 5)
        normalize = params.get('normalize', True)
        return _create_cusum_changepoint_features(
            DTindex,
            data,
            threshold=threshold,
            drift=drift,
            min_distance=min_distance,
            normalize=normalize,
        )
    
    elif method == 'ewma':
        if data is None:
            raise ValueError("Data is required for EWMA changepoint detection")
        lambda_param = params.get('lambda_param', 0.2)
        control_limit = params.get('control_limit', 3.0)
        min_distance = params.get('min_distance', 5)
        normalize = params.get('normalize', True)
        two_sided = params.get('two_sided', True)
        adaptive = params.get('adaptive', False)
        return _create_ewma_changepoint_features(
            DTindex,
            data,
            lambda_param=lambda_param,
            control_limit=control_limit,
            min_distance=min_distance,
            normalize=normalize,
            two_sided=two_sided,
            adaptive=adaptive,
        )
    
    elif method == 'autoencoder':
        if data is None:
            raise ValueError("Data is required for autoencoder changepoint detection")
        return _create_autoencoder_changepoint_features(DTindex, data, params)
    
    else:
        raise ValueError(f"Unknown changepoint detection method: {method}")


def changepoint_fcst_from_last_row(x_t_last_row, n_forecast=10):
    last_values = (
        x_t_last_row.values.reshape(1, -1) + 1
    )  # Shape it as 1 row, multiple columns

    # Create a 2D array where each column starts from the corresponding value in last_values
    forecast_steps = np.arange(n_forecast).reshape(
        -1, 1
    )  # Shape it as multiple rows, 1 column
    extended_features = np.maximum(0, last_values + forecast_steps)
    return pd.DataFrame(extended_features, columns=x_t_last_row.index)


def half_yr_spacing(df):
    return int(df.shape[0] / ((df.index.max().year - df.index.min().year + 1) * 2))

def _create_pelt_changepoint_features(DTindex, data, penalty=10, loss_function='l2', min_segment_length=1):
    """Create changepoint features using PELT algorithm."""
    changepoints = _detect_pelt_changepoints(data, penalty, loss_function, min_segment_length)
    
    if len(changepoints) == 0:
        # Return at least one changepoint in the middle if none detected
        changepoints = np.array([len(DTindex) // 2])
    
    n = len(DTindex)
    res = []
    for i, cp in enumerate(changepoints):
        feature_name = f'pelt_changepoint_{i+1}'
        res.append(pd.Series(np.maximum(0, np.arange(n) - cp), name=feature_name))
    
    changepoint_features = pd.concat(res, axis=1)
    changepoint_features.index = DTindex
    return changepoint_features


def _create_l1_changepoint_features(DTindex, data, lambda_reg=1.0, method='fused_lasso'):
    """Create changepoint features using L1 trend filtering."""
    changepoints, _ = _detect_l1_trend_changepoints(data, lambda_reg, method)
    
    if len(changepoints) == 0:
        # Return at least one changepoint in the middle if none detected
        changepoints = np.array([len(DTindex) // 2])
    
    n = len(DTindex)
    res = []
    for i, cp in enumerate(changepoints):
        feature_name = f'l1_{method}_changepoint_{i+1}'
        res.append(pd.Series(np.maximum(0, np.arange(n) - cp), name=feature_name))
    
    changepoint_features = pd.concat(res, axis=1)
    changepoint_features.index = DTindex
    return changepoint_features


def _create_cusum_changepoint_features(
    DTindex,
    data,
    threshold=5.0,
    drift=0.0,
    min_distance=5,
    normalize=True,
):
    """Create changepoint features using the CUSUM algorithm."""
    changepoints = _detect_cusum_changepoints(
        data,
        threshold=threshold,
        drift=drift,
        min_distance=min_distance,
        normalize=normalize,
    )

    if len(changepoints) == 0:
        changepoints = np.array([len(DTindex) // 2])

    n = len(DTindex)
    res = []
    for i, cp in enumerate(changepoints):
        feature_name = f'cusum_changepoint_{i+1}'
        res.append(pd.Series(np.maximum(0, np.arange(n) - cp), name=feature_name))

    changepoint_features = pd.concat(res, axis=1)
    changepoint_features.index = DTindex
    return changepoint_features


def _create_ewma_changepoint_features(
    DTindex,
    data,
    lambda_param=0.2,
    control_limit=3.0,
    min_distance=5,
    normalize=True,
    two_sided=True,
    adaptive=False,
):
    """Create changepoint features using the EWMA algorithm."""
    changepoints = _detect_ewma_changepoints(
        data,
        lambda_param=lambda_param,
        control_limit=control_limit,
        min_distance=min_distance,
        normalize=normalize,
        two_sided=two_sided,
        adaptive=adaptive,
    )

    if len(changepoints) == 0:
        changepoints = np.array([len(DTindex) // 2])

    n = len(DTindex)
    res = []
    for i, cp in enumerate(changepoints):
        feature_name = f'ewma_changepoint_{i+1}'
        res.append(pd.Series(np.maximum(0, np.arange(n) - cp), name=feature_name))

    changepoint_features = pd.concat(res, axis=1)
    changepoint_features.index = DTindex
    return changepoint_features


def _create_autoencoder_changepoint_features(
    DTindex,
    data,
    params=None,
):
    """Create changepoint features using autoencoder-based detection."""
    if params is None:
        params = {}
    min_distance = params.get('min_distance', 5)
    changepoints, _ = _detect_autoencoder_changepoints(
        data,
        method_params=params,
        min_distance=min_distance,
    )
    
    if len(changepoints) == 0:
        changepoints = np.array([len(DTindex) // 2])
    
    n = len(DTindex)
    res = []
    for i, cp in enumerate(changepoints):
        feature_name = f'autoencoder_changepoint_{i+1}'
        res.append(pd.Series(np.maximum(0, np.arange(n) - cp), name=feature_name))
    
    changepoint_features = pd.concat(res, axis=1)
    changepoint_features.index = DTindex
    return changepoint_features

def _approximate_l1_trend_filter(data, D, lambda_reg):
    """Approximate L1 trend filtering using iterative reweighting."""
    n = len(data)
    x = data.copy()  # Initialize with data
    
    # Iterative reweighted least squares approximation to L1
    for _ in range(3):  # Limited iterations for efficiency
        try:
            # Weights for reweighting (avoid division by zero)
            weights = 1.0 / (np.abs(D @ x) + 1e-6)
            
            # Weighted least squares problem: minimize ||data - x||^2 + lambda * ||W * D * x||^2
            # This approximates the L1 penalty with a weighted L2 penalty
            W = np.diag(np.sqrt(weights * lambda_reg))
            WD = W @ D
            
            # Solve: (I + (WD)^T * WD) * x = data
            A = np.eye(n) + WD.T @ WD
            x = np.linalg.solve(A, data)
            
        except (np.linalg.LinAlgError, ValueError):
            # If solve fails, use a simpler smoothing approach
            x = _simple_smooth(data, lambda_reg)
            break
    
    return x

def _simple_smooth(data, lambda_reg):
    """Simple smoothing as fallback."""
    try:
        from scipy.ndimage import gaussian_filter1d
        # Use Gaussian smoothing as a simple alternative
        sigma = max(1.0, lambda_reg / 10.0)  # Convert lambda to smoothing parameter
        return gaussian_filter1d(data.astype(float), sigma=sigma, mode='nearest')
    except ImportError:
        # Final fallback: simple moving average
        window = max(3, min(len(data) // 4, int(lambda_reg)))
        if window >= len(data):
            return data.astype(float)
        
        smoothed = data.astype(float).copy()
        for i in range(window // 2, len(data) - window // 2):
            smoothed[i] = np.mean(data[i - window // 2:i + window // 2 + 1])
        return smoothed

def _extract_changepoints_from_trend(fitted_trend, method):
    """Extract changepoints from fitted trend."""
    n = len(fitted_trend)
    
    if method == 'fused_lasso':
        # Look for level changes (first-order differences)
        differences = np.abs(np.diff(fitted_trend))
    else:  # total_variation
        # Look for trend changes (second-order differences)
        if n < 3:
            return np.array([])
        differences = np.abs(np.diff(fitted_trend, n=2))
    
    if len(differences) == 0:
        return np.array([])
    
    # Adaptive threshold based on data characteristics
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    if std_diff == 0:
        return np.array([])
    
    # Use a more conservative threshold
    threshold = mean_diff + 1.5 * std_diff
    
    if method == 'fused_lasso':
        changepoints = np.where(differences > threshold)[0] + 1
    else:  # total_variation
        changepoints = np.where(differences > threshold)[0] + 2  # Adjust for second-order diff
    
    # Filter out changepoints too close to boundaries
    changepoints = changepoints[(changepoints > 2) & (changepoints < n - 2)]
    
    return changepoints


class ChangepointDetector(object):
    """
    Advanced changepoint detection class for time series data.
    
    Supports multiple algorithms for detecting changepoints and level shifts in 
    wide-format time series data, similar to HolidayDetector.
    """
    
    def __init__(
        self,
        method='pelt',
        method_params=None,
        aggregate_method='mean',
        min_segment_length=5,
        probabilistic_output=False,
        n_jobs=1,
    ):
        """
        Initialize ChangePointDetector.
        
        Args:
            method (str): Changepoint detection method ('basic', 'pelt', 'l1_fused_lasso',
                'l1_total_variation', 'cusum', 'ewma', 'autoencoder', 'composite_fused_lasso')
            method_params (dict): Parameters specific to the chosen method
            aggregate_method (str): How to aggregate across series ('mean', 'median', 'individual')
            min_segment_length (int): Minimum length of segments between changepoints
            probabilistic_output (bool): Whether to output probability distributions for changepoints
            n_jobs (int): Number of parallel jobs for processing multiple series
        """
        self.method = method
        self.method_params = method_params if method_params is not None else {}
        self.aggregate_method = aggregate_method
        self.min_segment_length = min_segment_length
        self.probabilistic_output = probabilistic_output
        self.n_jobs = n_jobs
        
        # Results storage
        self.changepoints_ = None
        self.changepoint_probabilities_ = None
        self.fitted_trends_ = None
        self.df = None
        self.segment_breaks_ = None
        self.segment_values_ = None
        self.default_breaks_ = None
        self.default_values_ = None
        self.trend_df_ = None
        
    def _detect_series_individual(self, series_name, series):
        """Helper to run detection for a single series when using individual aggregation."""
        data = series.dropna().values
        changepoints = np.array([], dtype=int)
        fitted_trend = data
        probabilities = None

        if len(data) < 2 * self.min_segment_length:
            return {
                'name': series_name,
                'changepoints': changepoints,
                'fitted': fitted_trend,
                'probabilities': probabilities,
            }

        method = self.method
        params = self.method_params

        if method == 'pelt':
            penalty = params.get('penalty', 10)
            loss_function = params.get('loss_function', 'l2')
            changepoints = _detect_pelt_changepoints(
                data, penalty, loss_function, self.min_segment_length
            )
            fitted_trend = data

        elif method in ['l1_fused_lasso', 'l1_total_variation']:
            lambda_reg = params.get('lambda_reg', 1.0)
            l1_method = 'fused_lasso' if method == 'l1_fused_lasso' else 'total_variation'
            changepoints, fitted_trend = _detect_l1_trend_changepoints(
                data, lambda_reg, l1_method
            )

        elif method == 'cusum':
            threshold = params.get('threshold', 5.0)
            drift = params.get('drift', 0.0)
            normalize = params.get('normalize', True)
            min_distance = params.get('min_distance', self.min_segment_length)
            changepoints = _detect_cusum_changepoints(
                data,
                threshold=threshold,
                drift=drift,
                min_distance=min_distance,
                normalize=normalize,
            )
            fitted_trend = data

        elif method == 'ewma':
            lambda_param = params.get('lambda_param', 0.2)
            control_limit = params.get('control_limit', 3.0)
            normalize = params.get('normalize', True)
            two_sided = params.get('two_sided', True)
            adaptive = params.get('adaptive', False)
            min_distance = params.get('min_distance', self.min_segment_length)
            changepoints = _detect_ewma_changepoints(
                data,
                lambda_param=lambda_param,
                control_limit=control_limit,
                min_distance=min_distance,
                normalize=normalize,
                two_sided=two_sided,
                adaptive=adaptive,
            )
            fitted_trend = data

        elif method == 'autoencoder':
            min_distance = params.get('min_distance', self.min_segment_length)
            changepoints, fitted_trend = _detect_autoencoder_changepoints(
                data,
                method_params=params,
                min_distance=min_distance,
            )

        elif method == 'composite_fused_lasso':
            changepoints, fitted_trend = self._detect_composite_fused_lasso(data)

        elif method == 'basic':
            changepoint_spacing = params.get('changepoint_spacing', 60)
            changepoint_distance_end = params.get('changepoint_distance_end', 120)
            n = len(data)
            changepoint_range_end = max(1, min(n - changepoint_distance_end, n - 1))
            if changepoint_range_end <= 0:
                changepoints = np.array([0])
            else:
                changepoints = np.arange(0, changepoint_range_end, changepoint_spacing)
                changepoints = np.append(changepoints, changepoint_range_end)
            fitted_trend = data

        else:
            raise ValueError(f"Unknown method: {method}")

        if self.probabilistic_output:
            prob_method = params.get('probabilistic_method', 'bootstrap')
            probabilities, prob_cps = self._detect_probabilistic_changepoints(data, prob_method)
            if params.get('use_probabilistic_changepoints', False):
                changepoints = prob_cps

        return {
            'name': series_name,
            'changepoints': changepoints,
            'fitted': fitted_trend,
            'probabilities': probabilities,
        }

    def _detect_individual_sequential(self, df):
        """Fallback per-series detection path."""
        self.changepoints_ = {}
        self.fitted_trends_ = {}
        prob_results = {} if self.probabilistic_output else None

        for col in df.columns:
            result = self._detect_series_individual(col, df[col])
            self.changepoints_[col] = result['changepoints']
            self.fitted_trends_[col] = result['fitted']
            if self.probabilistic_output and result['probabilities'] is not None:
                prob_results[col] = result['probabilities']

        if self.probabilistic_output:
            self.changepoint_probabilities_ = prob_results if prob_results is not None else {}

    def _detect_individual_vectorized(self, df):
        """Vectorized detection path for supported methods."""
        series_names = list(df.columns)
        series_arrays = [df[name].dropna().to_numpy(dtype=float) for name in series_names]

        results = {}
        method_params = self.method_params

        if self.method == 'cusum':
            threshold = method_params.get('threshold', 5.0)
            drift = method_params.get('drift', 0.0)
            normalize = method_params.get('normalize', True)
            min_distance = method_params.get('min_distance', self.min_segment_length)

            eligible_arrays = []
            eligible_names = []
            for name, arr in zip(series_names, series_arrays):
                if len(arr) < max(1, 2 * self.min_segment_length):
                    results[name] = {
                        'changepoints': np.array([], dtype=int),
                        'fitted': arr,
                    }
                else:
                    eligible_names.append(name)
                    eligible_arrays.append(arr)

            if eligible_arrays:
                cps_list = _vectorized_cusum_changepoints(
                    eligible_arrays,
                    threshold,
                    drift,
                    min_distance,
                    normalize,
                    self.min_segment_length,
                )
                for name, arr, cps in zip(eligible_names, eligible_arrays, cps_list):
                    results[name] = {'changepoints': cps, 'fitted': arr}

        elif self.method == 'ewma':
            lambda_param = method_params.get('lambda_param', 0.2)
            control_limit = method_params.get('control_limit', 3.0)
            normalize = method_params.get('normalize', True)
            two_sided = method_params.get('two_sided', True)
            adaptive = method_params.get('adaptive', False)
            min_distance = method_params.get('min_distance', self.min_segment_length)

            eligible_arrays = []
            eligible_names = []
            for name, arr in zip(series_names, series_arrays):
                if len(arr) < max(2, 2 * self.min_segment_length):
                    results[name] = {
                        'changepoints': np.array([], dtype=int),
                        'fitted': arr,
                    }
                else:
                    eligible_names.append(name)
                    eligible_arrays.append(arr)

            if eligible_arrays:
                cps_list = _vectorized_ewma_changepoints(
                    eligible_arrays,
                    lambda_param,
                    control_limit,
                    min_distance,
                    normalize,
                    two_sided,
                    adaptive,
                    self.min_segment_length,
                )
                for name, arr, cps in zip(eligible_names, eligible_arrays, cps_list):
                    results[name] = {'changepoints': cps, 'fitted': arr}

        elif self.method in {'l1_fused_lasso', 'l1_total_variation'}:
            lambda_reg = method_params.get('lambda_reg', 1.0)
            l1_results = _vectorized_l1_detection(
                series_names,
                series_arrays,
                lambda_reg,
                self.method,
                self.min_segment_length,
            )
            results.update(l1_results)

        elif self.method == 'autoencoder':
            auto_results = _detect_autoencoder_changepoints_vectorized(
                series_names,
                series_arrays,
                method_params,
                self.min_segment_length,
            )
            results.update(auto_results)

        else:
            raise ValueError(f"Vectorized detection not implemented for method: {self.method}")

        self.changepoints_ = {}
        self.fitted_trends_ = {}
        prob_results = {} if self.probabilistic_output else None

        for name, arr in zip(series_names, series_arrays):
            outcome = results.get(name, None)
            if outcome is None:
                outcome = {
                    'changepoints': np.array([], dtype=int),
                    'fitted': arr,
                }

            changepoints = np.asarray(outcome['changepoints'], dtype=int)
            fitted = np.asarray(outcome['fitted'], dtype=float)

            self.changepoints_[name] = changepoints
            self.fitted_trends_[name] = fitted

            if self.probabilistic_output and len(arr) >= max(1, 2 * self.min_segment_length):
                prob_method = method_params.get('probabilistic_method', 'bootstrap')
                probabilities, prob_cps = self._detect_probabilistic_changepoints(arr, prob_method)
                if method_params.get('use_probabilistic_changepoints', False):
                    self.changepoints_[name] = prob_cps
                prob_results[name] = probabilities

        if self.probabilistic_output:
            self.changepoint_probabilities_ = prob_results if prob_results is not None else {}

    def detect(self, df):
        """
        Run changepoint detection on wide-format time series data.
        
        Args:
            df (pd.DataFrame): Wide-format time series with DatetimeIndex
        """
        self.df = df.copy()
        self.df_cols = df.columns
        self.changepoint_probabilities_ = None
        
        if self.aggregate_method == 'individual':
            if self.method in {'cusum', 'ewma', 'l1_fused_lasso', 'l1_total_variation', 'autoencoder'}:
                self._detect_individual_vectorized(df)
            else:
                self._detect_individual_sequential(df)
        else:
            # Aggregate data across series first
            if self.aggregate_method == 'mean':
                aggregated_data = df.mean(axis=1).values
            elif self.aggregate_method == 'median':
                aggregated_data = df.median(axis=1).values
            else:
                raise ValueError(f"Unknown aggregate_method: {self.aggregate_method}")
            
            aggregated_data = aggregated_data[~np.isnan(aggregated_data)]
            
            if self.method == 'pelt':
                penalty = self.method_params.get('penalty', 10)
                loss_function = self.method_params.get('loss_function', 'l2')
                self.changepoints_ = _detect_pelt_changepoints(
                    aggregated_data, penalty, loss_function, self.min_segment_length
                )
                
            elif self.method in ['l1_fused_lasso', 'l1_total_variation']:
                lambda_reg = self.method_params.get('lambda_reg', 1.0)
                l1_method = 'fused_lasso' if self.method == 'l1_fused_lasso' else 'total_variation'
                self.changepoints_, self.fitted_trends_ = _detect_l1_trend_changepoints(
                    aggregated_data, lambda_reg, l1_method
                )
                
            elif self.method == 'cusum':
                threshold = self.method_params.get('threshold', 5.0)
                drift = self.method_params.get('drift', 0.0)
                normalize = self.method_params.get('normalize', True)
                min_distance = self.method_params.get('min_distance', self.min_segment_length)
                self.changepoints_ = _detect_cusum_changepoints(
                    aggregated_data,
                    threshold=threshold,
                    drift=drift,
                    min_distance=min_distance,
                    normalize=normalize,
                )
                self.fitted_trends_ = aggregated_data
            
            elif self.method == 'ewma':
                lambda_param = self.method_params.get('lambda_param', 0.2)
                control_limit = self.method_params.get('control_limit', 3.0)
                normalize = self.method_params.get('normalize', True)
                two_sided = self.method_params.get('two_sided', True)
                adaptive = self.method_params.get('adaptive', False)
                min_distance = self.method_params.get('min_distance', self.min_segment_length)
                self.changepoints_ = _detect_ewma_changepoints(
                    aggregated_data,
                    lambda_param=lambda_param,
                    control_limit=control_limit,
                    min_distance=min_distance,
                    normalize=normalize,
                    two_sided=two_sided,
                    adaptive=adaptive,
                )
                self.fitted_trends_ = aggregated_data
            
            elif self.method == 'autoencoder':
                min_distance = self.method_params.get('min_distance', self.min_segment_length)
                self.changepoints_, self.fitted_trends_ = _detect_autoencoder_changepoints(
                    aggregated_data,
                    method_params=self.method_params,
                    min_distance=min_distance,
                )
                
            elif self.method == 'composite_fused_lasso':
                self.changepoints_, self.fitted_trends_ = self._detect_composite_fused_lasso(aggregated_data)
            
            elif self.method == 'basic':
                # Basic evenly-spaced changepoints (legacy method)
                changepoint_spacing = self.method_params.get('changepoint_spacing', 60)
                changepoint_distance_end = self.method_params.get('changepoint_distance_end', 120)
                # Convert to changepoint indices
                n = len(aggregated_data)
                changepoint_range_end = max(1, min(n - changepoint_distance_end, n - 1))
                if changepoint_range_end <= 0:
                    self.changepoints_ = np.array([0])
                else:
                    self.changepoints_ = np.arange(0, changepoint_range_end, changepoint_spacing)
                    self.changepoints_ = np.append(self.changepoints_, changepoint_range_end)
                self.fitted_trends_ = aggregated_data
            
            # Handle probabilistic output for aggregated data
            if self.probabilistic_output:
                prob_method = self.method_params.get('probabilistic_method', 'bootstrap')
                self.changepoint_probabilities_, prob_cps = self._detect_probabilistic_changepoints(aggregated_data, prob_method)
                # Update changepoints with probabilistic results if requested
                if self.method_params.get('use_probabilistic_changepoints', False):
                    self.changepoints_ = prob_cps
        
        # Prepare transformer-related data structures
        self._prepare_transform_support()

    def _aggregate_series(self, df):
        """Aggregate wide-format data according to the configured method."""
        if self.aggregate_method == 'mean':
            return df.mean(axis=1)
        elif self.aggregate_method == 'median':
            return df.median(axis=1)
        elif self.aggregate_method == 'individual':
            raise ValueError("_aggregate_series should not be called with 'individual' method.")
        else:
            raise ValueError(f"Unknown aggregate_method: {self.aggregate_method}")

    def _prepare_transform_support(self):
        """Create segment metadata and cached trend for transformer operations."""
        if self.df is None or self.changepoints_ is None:
            self.segment_breaks_ = None
            self.segment_values_ = None
            self.trend_df_ = None
            return

        self.segment_breaks_ = {}
        self.segment_values_ = {}

        if self.aggregate_method == 'individual' and isinstance(self.changepoints_, dict):
            for col in self.df.columns:
                series = self.df[col]
                changepoints = self.changepoints_.get(col, np.array([]))
                breaks, values = _compute_segment_statistics(series, changepoints)
                self.segment_breaks_[col] = breaks
                self.segment_values_[col] = values
        else:
            aggregated_series = self._aggregate_series(self.df)
            breaks, values = _compute_segment_statistics(
                aggregated_series, self.changepoints_
            )
            for col in self.df.columns:
                self.segment_breaks_[col] = breaks
                self.segment_values_[col] = values

        if self.segment_breaks_:
            first_key = next(iter(self.segment_breaks_))
            self.default_breaks_ = self.segment_breaks_[first_key]
            self.default_values_ = self.segment_values_[first_key]
        else:
            self.default_breaks_ = pd.Index([])
            self.default_values_ = np.array([], dtype=float)

        self.trend_df_ = self._generate_trend_for_df(self.df)

    def _generate_trend_for_df(self, df):
        """Generate a trend DataFrame aligned to df using stored segment metadata."""
        if self.segment_breaks_ is None or self.segment_values_ is None:
            raise ValueError("Must call detect() or fit() before generating trends.")

        trend_dict = {}
        for col in df.columns:
            breaks = self.segment_breaks_.get(col, self.default_breaks_)
            values = self.segment_values_.get(col, self.default_values_)
            trend_values = _evaluate_segment_trend(df.index, breaks, values)
            trend_dict[col] = trend_values

        return pd.DataFrame(trend_dict, index=df.index, dtype=float)

    def fit(self, df):
        """
        Fit the changepoint detector and prepare transformer artifacts.

        Args:
            df (pd.DataFrame): Training data with DatetimeIndex.
        """
        self.detect(df)
        return self

    def fit_transform(self, df):
        """
        Fit the detector and immediately transform the input data.

        Args:
            df (pd.DataFrame): Training data with DatetimeIndex.
        """
        self.fit(df)
        return self.transform(df)

    def transform(self, df):
        """
        Apply changepoint-based detrending to the provided data.

        Args:
            df (pd.DataFrame): Data to transform.
        """
        try:
            df_numeric = df.astype(float)
        except Exception as exc:
            raise ValueError("Data Cannot Be Converted to Numeric Float") from exc

        trend = self._generate_trend_for_df(df_numeric)
        return df_numeric - trend

    def inverse_transform(self, df, trans_method="forecast"):
        """
        Restore data to the original scale using stored changepoint trends.

        Args:
            df (pd.DataFrame): Data to inverse transform.
            trans_method (str): Compatibility argument for transformer interface.
        """
        try:
            df_numeric = df.astype(float)
        except Exception as exc:
            raise ValueError("Data Cannot Be Converted to Numeric Float") from exc

        trend = self._generate_trend_for_df(df_numeric)
        return df_numeric + trend
    
    def _detect_composite_fused_lasso(self, data):
        """
        Composite fused lasso for joint level + slope changepoint detection.
        
        Args:
            data (array-like): Time series data
            
        Returns:
            tuple: (changepoints, fitted_trend)
        """
        try:
            from scipy.optimize import minimize
            from scipy.sparse import diags
        except ImportError:
            raise ImportError("scipy is required for composite fused lasso")
        
        n = len(data)
        if n < 3:
            return np.array([]), data.copy()
        
        lambda_level = self.method_params.get('lambda_level', 1.0)
        lambda_slope = self.method_params.get('lambda_slope', 1.0)
        
        # Create difference matrices
        D1 = diags([1, -1], [0, 1], shape=(n-1, n)).toarray()  # First differences (level)
        D2 = diags([1, -2, 1], [0, 1, 2], shape=(n-2, n)).toarray()  # Second differences (slope)
        
        # Objective function
        def objective(x):
            data_fit = 0.5 * np.sum((data - x) ** 2)
            level_penalty = lambda_level * np.sum(np.abs(D1 @ x))
            slope_penalty = lambda_slope * np.sum(np.abs(D2 @ x))
            return data_fit + level_penalty + slope_penalty
        
        # Solve optimization
        result = minimize(objective, data, method='L-BFGS-B')
        fitted_trend = result.x
        
        # Find changepoints from both level and slope changes
        level_changes = np.abs(np.diff(fitted_trend))
        slope_changes = np.abs(np.diff(fitted_trend, n=2))
        
        level_threshold = np.std(level_changes) * 2
        slope_threshold = np.std(slope_changes) * 2
        
        level_cps = np.where(level_changes > level_threshold)[0] + 1
        slope_cps = np.where(slope_changes > slope_threshold)[0] + 1
        
        # Combine and remove duplicates
        all_changepoints = np.unique(np.concatenate([level_cps, slope_cps]))
        
        # Apply minimum segment length constraint
        if len(all_changepoints) > 0:
            filtered_cps = [all_changepoints[0]]
            for cp in all_changepoints[1:]:
                if cp - filtered_cps[-1] >= self.min_segment_length:
                    filtered_cps.append(cp)
            all_changepoints = np.array(filtered_cps)
        
        return all_changepoints, fitted_trend
    
    def _detect_probabilistic_changepoints(self, data, method='bayesian_online'):
        """
        Detect changepoints with probability distributions.
        
        Args:
            data (array-like): Time series data
            method (str): Probabilistic method ('bayesian_online', 'bootstrap')
            
        Returns:
            tuple: (changepoint_probabilities, most_likely_changepoints)
        """
        n = len(data)
        
        if method == 'bayesian_online':
            # Simple Bayesian online changepoint detection
            hazard_rate = self.method_params.get('hazard_rate', 1/100)  # Prior belief about changepoint frequency
            
            # Initialize
            R = np.zeros((n, n))  # Run length probabilities
            R[0, 0] = 1.0
            
            changepoint_probs = np.zeros(n)
            
            # Online updates
            for t in range(1, n):
                # Calculate predictive probabilities
                pred_probs = np.zeros(t)
                for r in range(t):
                    if R[t-1, r] > 1e-10:  # Only compute for non-zero probabilities
                        # Simple Gaussian model for segments
                        if r == 0:
                            segment_data = data[:t]
                        else:
                            segment_data = data[t-r-1:t]
                        
                        if len(segment_data) > 0:
                            mu = np.mean(segment_data)
                            sigma = np.std(segment_data) + 1e-6  # Add small constant for stability
                            pred_probs[r] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data[t] - mu) / sigma) ** 2)
                
                # Update run length probabilities
                evidence = 0
                for r in range(t):
                    # Growth probability (no changepoint)
                    R[t, r+1] = R[t-1, r] * pred_probs[r] * (1 - hazard_rate)
                    evidence += R[t, r+1]
                    
                    # Changepoint probability
                    R[t, 0] += R[t-1, r] * pred_probs[r] * hazard_rate
                    evidence += R[t-1, r] * pred_probs[r] * hazard_rate
                
                # Normalize
                if evidence > 0:
                    R[t, :] /= evidence
                    changepoint_probs[t] = R[t, 0]
                
        elif method == 'bootstrap':
            # Bootstrap-based uncertainty estimation
            max_bootstrap = int(self.method_params.get('probabilistic_max_bootstrap', 30))
            default_bootstrap = min(max_bootstrap, max(10, n // 2 if n // 2 > 0 else 10))
            n_bootstrap = int(self.method_params.get('n_bootstrap', default_bootstrap))
            n_bootstrap = max(1, n_bootstrap)
            bootstrap_changepoints = []
            rng = np.random.default_rng()
            
            for _ in range(n_bootstrap):
                # Resample data with replacement
                bootstrap_indices = rng.choice(n, size=n, replace=True)
                bootstrap_data = data[bootstrap_indices]
                
                # Detect changepoints on bootstrap sample
                if self.method == 'pelt':
                    penalty = self.method_params.get('penalty', 10)
                    loss_function = self.method_params.get('loss_function', 'l2')
                    cps = _detect_pelt_changepoints(bootstrap_data, penalty, loss_function, self.min_segment_length)
                else:
                    # Use L1 trend filtering as fallback
                    lambda_reg = self.method_params.get('lambda_reg', 1.0)
                    cps, _ = _detect_l1_trend_changepoints(bootstrap_data, lambda_reg, 'fused_lasso')
                
                bootstrap_changepoints.extend(cps)
            
            # Convert to probabilities
            changepoint_probs = np.zeros(n)
            for cp in bootstrap_changepoints:
                if 0 <= cp < n:
                    changepoint_probs[cp] += 1
            changepoint_probs /= n_bootstrap
            
        else:
            raise ValueError(f"Unknown probabilistic method: {method}")
        
        # Find most likely changepoints (above threshold)
        threshold = self.method_params.get('probability_threshold', 0.5)
        most_likely_cps = np.where(changepoint_probs > threshold)[0]
        
        return changepoint_probs, most_likely_cps
    
    def get_market_changepoints(self, method='dbscan', params=None):
        """
        Find common changepoints across multiple time series using clustering.
        
        Args:
            method (str): Clustering method ('dbscan', 'kmeans', 'hierarchical')
            params (dict): Parameters for clustering algorithm
            
        Returns:
            np.ndarray: Array of market-wide changepoint indices
        """
        if self.changepoints_ is None:
            raise ValueError("Must run detect() first")
        
        if params is None:
            params = {}
        
        if isinstance(self.changepoints_, dict):
            # Collect all changepoints from individual series
            all_changepoints = []
            for col, cps in self.changepoints_.items():
                all_changepoints.extend(cps)
            all_changepoints = np.array(all_changepoints)
        else:
            all_changepoints = self.changepoints_
        
        if len(all_changepoints) == 0:
            return np.array([])
        
        # Reshape for clustering (each changepoint is a 1D point)
        X = all_changepoints.reshape(-1, 1)
        
        if method == 'dbscan':
            try:
                from sklearn.cluster import DBSCAN
            except ImportError:
                raise ImportError("scikit-learn is required for DBSCAN clustering")
            
            eps = params.get('eps', 5)  # 5 time steps tolerance
            min_samples = params.get('min_samples', 2)
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = clustering.labels_
            
            # Find cluster centers (market changepoints)
            market_changepoints = []
            for label in set(labels):
                if label != -1:  # Ignore noise points
                    cluster_points = all_changepoints[labels == label]
                    market_changepoints.append(int(np.median(cluster_points)))
            
            return np.array(sorted(market_changepoints))
        
        elif method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                raise ImportError("scikit-learn is required for KMeans clustering")
            
            n_clusters = params.get('n_clusters', max(1, len(all_changepoints) // 3))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            return np.array(sorted(kmeans.cluster_centers_.flatten().astype(int)))
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def plot(self, series_name=None, figsize=(12, 8)):
        """
        Plot time series with detected changepoints.
        
        Args:
            series_name (str): Name of series to plot (for individual detection)
            figsize (tuple): Figure size
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        
        if self.df is None:
            raise ValueError("Must run detect() first")
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        if isinstance(self.changepoints_, dict):
            if series_name is None:
                series_name = self.df.columns[0]
            data = self.df[series_name]
            changepoints = self.changepoints_[series_name]
            fitted_trend = self.fitted_trends_.get(series_name, None)
        else:
            data = self.df.mean(axis=1)
            changepoints = self.changepoints_
            fitted_trend = self.fitted_trends_
        
        # Plot original data
        axes[0].plot(data.index, data.values, label='Original Data', alpha=0.7)
        if fitted_trend is not None:
            axes[0].plot(data.index, fitted_trend, label='Fitted Trend', linewidth=2)
        
        # Mark changepoints
        for cp in changepoints:
            if cp < len(data):
                axes[0].axvline(data.index[cp], color='red', linestyle='--', alpha=0.7)
        
        axes[0].set_title(f'Changepoint Detection - {series_name if isinstance(self.changepoints_, dict) else "Aggregated"}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot changepoint locations or probabilities
        if self.probabilistic_output and self.changepoint_probabilities_ is not None:
            if isinstance(self.changepoint_probabilities_, dict):
                if series_name is None:
                    series_name = list(self.changepoint_probabilities_.keys())[0]
                probs = self.changepoint_probabilities_[series_name]
            else:
                probs = self.changepoint_probabilities_
            
            axes[1].plot(range(len(probs)), probs, color='blue', linewidth=2)
            axes[1].set_xlabel('Time Index')
            axes[1].set_ylabel('Changepoint Probability')
            axes[1].set_title('Changepoint Probabilities')
            axes[1].axhline(y=self.method_params.get('probability_threshold', 0.5), 
                          color='red', linestyle='--', alpha=0.7, label='Threshold')
            axes[1].legend()
        else:
            axes[1].scatter(range(len(changepoints)), changepoints, color='red', s=50)
            axes[1].set_xlabel('Changepoint Number')
            axes[1].set_ylabel('Time Index')
            axes[1].set_title('Changepoint Locations')
        
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_features(self, forecast_length=0):
        """
        Create changepoint features for regression modeling.
        
        Args:
            forecast_length (int): Number of future periods to extend features
            
        Returns:
            pd.DataFrame: Changepoint features
        """
        if self.changepoints_ is None:
            raise ValueError("Must run detect() first")
        
        extended_index = self.df.index
        if forecast_length > 0:
            freq = infer_frequency(self.df.index)
            try:
                future_index = pd.date_range(
                    start=self.df.index[-1] + pd.Timedelta(freq), 
                    periods=forecast_length, 
                    freq=freq
                )
            except (ValueError, TypeError):
                # Fallback: use the median time difference
                time_diff = pd.Series(self.df.index).diff().dropna().median()
                future_index = pd.date_range(
                    start=self.df.index[-1] + time_diff,
                    periods=forecast_length,
                    freq=time_diff
                )
            extended_index = self.df.index.append(future_index)
        
        if isinstance(self.changepoints_, dict):
            # Use market changepoints or the first series' changepoints
            try:
                changepoints = self.get_market_changepoints()
            except:
                changepoints = list(self.changepoints_.values())[0]
        else:
            changepoints = self.changepoints_
        
        if len(changepoints) == 0:
            changepoints = np.array([len(self.df) // 2])  # Default middle changepoint
        
        # Convert changepoint indices to actual timestamps
        changepoint_dates = []
        for cp in changepoints:
            if cp < len(self.df):
                changepoint_dates.append(self.df.index[int(cp)])
            else:
                # Handle edge case where changepoint is beyond training data
                changepoint_dates.append(self.df.index[-1])
        
        # Create time-based features
        res = []
        for i, cp_date in enumerate(changepoint_dates):
            feature_name = f'{self.method}_changepoint_{i+1}'
            
            # Calculate time-based differences
            if isinstance(extended_index, pd.DatetimeIndex):
                # For datetime index, calculate differences in periods
                time_diffs = (extended_index - cp_date).total_seconds()
                
                # Infer the time unit from the data frequency
                try:
                    freq_str = infer_frequency(self.df.index)
                    if freq_str:
                        freq_seconds = pd.Timedelta(freq_str).total_seconds()
                    else:
                        # Fallback: use median time difference
                        freq_seconds = pd.Series(self.df.index).diff().dropna().median().total_seconds()
                except (ValueError, TypeError):
                    # Fallback: use median time difference
                    freq_seconds = pd.Series(self.df.index).diff().dropna().median().total_seconds()
                
                time_periods = time_diffs / freq_seconds
                
                # Create feature as "periods since changepoint"
                feature_values = np.maximum(0, time_periods)
            else:
                # Fallback for non-datetime indices (shouldn't happen in practice)
                feature_values = np.maximum(0, np.arange(len(extended_index)) - changepoints[i])
            
            res.append(pd.Series(feature_values, name=feature_name))
        
        changepoint_features = pd.concat(res, axis=1)
        changepoint_features.index = extended_index
        
        return changepoint_features
    
    @staticmethod
    def get_new_params(method="random"):
        """
        Generate new random parameters for changepoint detection.
        
        Args:
            method (str): Method for parameter selection
                - 'fast': All methods but with fastest parameter configurations for PELT and composite_fused_lasso
                - Or specify a method name directly: 'basic', 'pelt', 'l1_fused_lasso', 
                  'l1_total_variation', 'cusum', 'autoencoder', 'composite_fused_lasso'
        
        Returns:
            dict: Complete parameter dictionary for ChangepointDetector initialization
        """
        # List of all valid method names
        valid_methods = ['basic', 'pelt', 'l1_fused_lasso', 'l1_total_variation', 
                        'cusum', 'ewma', 'autoencoder', 'composite_fused_lasso']

        selection_mode = "fast"  # default to fast
        if method in valid_methods:
            new_method = method
        elif method == "fast":
            # Include all methods but will use fast parameters for potentially slow ones
            method_options = ['basic', 'cusum', 'ewma', 'l1_fused_lasso', 'l1_total_variation', 'pelt', 'composite_fused_lasso', 'autoencoder']
            method_weights = [0.3, 0.2, 0.2, 0.08, 0.08, 0.08, 0.03, 0.03]
            new_method = random.choices(method_options, weights=method_weights, k=1)[0]
        elif method in ["default", "random"]:
            # Heavily weight basic method for compatibility, with EWMA and CUSUM as good alternatives
            method_options = ['basic', 'cusum', 'ewma', 'pelt', 'l1_fused_lasso', 'l1_total_variation', 'autoencoder']
            method_weights = [0.5, 0.15, 0.15, 0.08, 0.04, 0.04, 0.04]
            new_method = random.choices(method_options, weights=method_weights, k=1)[0]
            selection_mode = "random"
        else:  # random
            new_method = random.choices(valid_methods, k=1)[0]
            selection_mode = "random"

        # Generate method-specific parameters with weighted choices
        if new_method == 'basic':
            # Basic method parameters (legacy style)
            spacing_options = [6, 28, 60, 90, 120, 180, 360, 5040]
            spacing_weights = [0.05, 0.1, 0.1, 0.1, 0.05, 0.2, 0.1, 0.3]
            distance_end_options = [6, 28, 60, 90, 180, 360, 520, 5040]
            distance_end_weights = [0.05, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.3]
            
            new_params = {
                'changepoint_spacing': random.choices(spacing_options, weights=spacing_weights, k=1)[0],
                'changepoint_distance_end': random.choices(distance_end_options, weights=distance_end_weights, k=1)[0],
            }
            
        elif new_method == 'pelt':
            # PELT method parameters
            if selection_mode == "fast":
                # Fast mode: Use only fastest parameters
                # Higher penalties = fewer changepoints = faster computation
                # L2 loss is fastest
                penalty_options = [20, 50, 100]
                penalty_weights = [0.4, 0.4, 0.2]
                loss_functions = ['l2']  # L2 is fastest
                loss_weights = [1.0]
                min_segment_options = [5, 10]  # Larger segments = faster
                min_segment_weights = [0.6, 0.4]
            else:
                # Normal mode: Full range of parameters
                penalty_options = [1, 5, 10, 20, 50, 100]
                penalty_weights = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]
                loss_functions = ['l1', 'l2', 'huber']
                loss_weights = [0.3, 0.5, 0.2]
                min_segment_options = [1, 2, 5, 10]
                min_segment_weights = [0.4, 0.3, 0.2, 0.1]
            
            new_params = {
                'penalty': random.choices(penalty_options, weights=penalty_weights, k=1)[0],
                'loss_function': random.choices(loss_functions, weights=loss_weights, k=1)[0],
                'min_segment_length': random.choices(min_segment_options, weights=min_segment_weights, k=1)[0],
            }
            
        elif new_method in ['l1_fused_lasso', 'l1_total_variation']:
            # L1 trend filtering parameters
            lambda_options = [0.01, 0.1, 1.0, 10.0, 100.0]
            lambda_weights = [0.1, 0.2, 0.4, 0.2, 0.1]
            
            new_params = {
                'lambda_reg': random.choices(lambda_options, weights=lambda_weights, k=1)[0],
            }
            
        elif new_method == 'composite_fused_lasso':
            # Composite fused lasso parameters
            if selection_mode == "fast":
                # Fast mode: Use smaller lambda values for faster convergence
                # Smaller lambda = less regularization = faster optimization
                lambda_level_options = [0.1, 0.5, 1.0]
                lambda_level_weights = [0.4, 0.4, 0.2]
                lambda_slope_options = [0.1, 0.5, 1.0]
                lambda_slope_weights = [0.4, 0.4, 0.2]
            else:
                # Normal mode: Full range including higher lambda values
                lambda_level_options = [0.1, 0.5, 1.0, 2.0, 5.0]
                lambda_level_weights = [0.2, 0.3, 0.3, 0.15, 0.05]
                lambda_slope_options = [0.1, 0.5, 1.0, 2.0, 5.0]
                lambda_slope_weights = [0.2, 0.3, 0.3, 0.15, 0.05]
            
            new_params = {
                'lambda_level': random.choices(lambda_level_options, weights=lambda_level_weights, k=1)[0],
                'lambda_slope': random.choices(lambda_slope_options, weights=lambda_slope_weights, k=1)[0],
            }
            
        elif new_method == 'cusum':
            # CUSUM parameters - updated with better defaults
            threshold_options = [5.0, 10.0, 15.0, 20.0, 30.0]
            threshold_weights = [0.1, 0.3, 0.3, 0.2, 0.1]
            drift_options = [0.0, 0.25, 0.5, 1.0]
            drift_weights = [0.2, 0.3, 0.3, 0.2]
            normalize_options = [True, False]
            normalize_weights = [0.8, 0.2]
            min_distance_options = [5, 10, 15, 20]
            min_distance_weights = [0.2, 0.4, 0.3, 0.1]
            
            new_params = {
                'threshold': random.choices(threshold_options, weights=threshold_weights, k=1)[0],
                'drift': random.choices(drift_options, weights=drift_weights, k=1)[0],
                'normalize': random.choices(normalize_options, weights=normalize_weights, k=1)[0],
                'min_distance': random.choices(min_distance_options, weights=min_distance_weights, k=1)[0],
            }
        
        elif new_method == 'ewma':
            # EWMA parameters with industry standard defaults
            # Lambda: smaller = more smoothing, better for small persistent shifts
            lambda_options = [0.1, 0.2, 0.3, 0.4, 0.6]
            lambda_weights = [0.2, 0.35, 0.2, 0.15, 0.1]  # Favor standard 0.2
            
            # Control limits: standard is 3-sigma
            control_limit_options = [2.5, 3.0, 3.5, 4.0]
            control_limit_weights = [0.15, 0.5, 0.25, 0.1]  # Favor standard 3.0
            
            normalize_options = [True, False]
            normalize_weights = [0.8, 0.2]
            
            two_sided_options = [True, False]
            two_sided_weights = [0.8, 0.2]  # Usually want both directions
            
            adaptive_options = [True, False]
            adaptive_weights = [0.6, 0.4]  # Adaptive (FIR) generally better
            
            min_distance_options = [5, 10, 15, 20]
            min_distance_weights = [0.2, 0.4, 0.3, 0.1]
            
            new_params = {
                'lambda_param': random.choices(lambda_options, weights=lambda_weights, k=1)[0],
                'control_limit': random.choices(control_limit_options, weights=control_limit_weights, k=1)[0],
                'normalize': random.choices(normalize_options, weights=normalize_weights, k=1)[0],
                'two_sided': random.choices(two_sided_options, weights=two_sided_weights, k=1)[0],
                'adaptive': random.choices(adaptive_options, weights=adaptive_weights, k=1)[0],
                'min_distance': random.choices(min_distance_options, weights=min_distance_weights, k=1)[0],
            }
            
        elif new_method == 'autoencoder':
            # Autoencoder parameters
            window_options = [5, 10, 20, 30]
            window_weights = [0.2, 0.3, 0.3, 0.2]
            smoothing_options = [1, 3, 5]
            smoothing_weights = [0.3, 0.5, 0.2]
            contamination_options = [0.05, 0.1, 0.15, 0.2]
            contamination_weights = [0.25, 0.4, 0.2, 0.15]
            epochs_options = [20, 40, 60]
            epochs_weights = [0.4, 0.4, 0.2]
            normalize_scores_options = [True, False]
            normalize_scores_weights = [0.7, 0.3]
            use_flags_options = [True, False]
            use_flags_weights = [0.7, 0.3]
            
            new_params = {
                'window_size': random.choices(window_options, weights=window_weights, k=1)[0],
                'smoothing_window': random.choices(smoothing_options, weights=smoothing_weights, k=1)[0],
                'contamination': random.choices(contamination_options, weights=contamination_weights, k=1)[0],
                'epochs': random.choices(epochs_options, weights=epochs_weights, k=1)[0],
                'normalize_scores': random.choices(normalize_scores_options, weights=normalize_scores_weights, k=1)[0],
                'use_anomaly_flags': random.choices(use_flags_options, weights=use_flags_weights, k=1)[0],
            }
        else:
            new_params = {}
        
        # Generate common parameters with weighted choices
        if selection_mode == "fast":
            aggregate_options = ['mean', 'median', 'individual']
            aggregate_weights = [0.45, 0.35, 0.20]
            min_segment_options = [5, 10, 15]
            min_segment_weights = [0.5, 0.3, 0.2]
            probabilistic_options = [False, True]
            probabilistic_weights = [0.9, 0.1]
            # Encourage faster probability methods when they are requested
            new_params.setdefault('probabilistic_method', 'bayesian_online')
            new_params.setdefault('probabilistic_max_bootstrap', 20)
            new_params.setdefault('n_bootstrap', 15)
        else:
            aggregate_options = ['mean', 'median', 'individual']
            aggregate_weights = [0.45, 0.30, 0.25]
            min_segment_options = [3, 5, 10, 15]
            min_segment_weights = [0.25, 0.4, 0.25, 0.1]
            probabilistic_options = [True, False]
            probabilistic_weights = [0.15, 0.85]
        
        return {
            'method': new_method,
            'method_params': new_params,
            'aggregate_method': random.choices(aggregate_options, weights=aggregate_weights, k=1)[0],
            'min_segment_length': random.choices(min_segment_options, weights=min_segment_weights, k=1)[0],
            'probabilistic_output': random.choices(probabilistic_options, weights=probabilistic_weights, k=1)[0],
        }


def generate_random_changepoint_params(method='random'):
    """
    Generate random parameters for changepoint detection methods.
    
    This function creates appropriately weighted random parameters for different
    changepoint detection algorithms, supporting the flexible method/params system.
    
    DEPRECATED: This function now delegates to ChangepointDetector.get_new_params()
    for consistency. Use ChangepointDetector.get_new_params() directly for new code.
    
    Args:
        method (str): Method for parameter selection
            - 'random': All methods with balanced weights
            - 'fast': All methods but with fastest parameter configurations for PELT and composite_fused_lasso
            - 'default'/'basic_weighted': Basic method heavily weighted
        
    Returns:
        tuple: (changepoint_method, changepoint_params) where
            - changepoint_method (str): Selected method name
            - changepoint_params (dict): Method-specific parameters
    """
    # Delegate to the unified implementation
    result = ChangepointDetector.get_new_params(method=method)
    
    # Return in the legacy format (method, params) instead of full dict
    return result['method'], result['method_params']


def find_market_changepoints_multivariate(
    df, 
    detector_params=None, 
    clustering_method='dbscan',
    clustering_params=None,
    min_series_agreement=0.3
):
    """
    Find common changepoints across multivariate time series data.
    
    Args:
        df (pd.DataFrame): Wide-format time series data
        detector_params (dict): Parameters for ChangePointDetector
        clustering_method (str): Method for clustering changepoints ('dbscan', 'kmeans', 'agreement')
        clustering_params (dict): Parameters for clustering
        min_series_agreement (float): Minimum fraction of series that must agree on a changepoint
        
    Returns:
        dict: Dictionary with market changepoints and individual series changepoints
    """
    if detector_params is None:
        detector_params = {'method': 'pelt', 'aggregate_method': 'individual'}
    
    if clustering_params is None:
        clustering_params = {}
    
    # Detect changepoints for each series individually
    detector = ChangepointDetector(**detector_params)
    detector.detect(df)
    
    if clustering_method == 'agreement':
        # Find changepoints that appear in multiple series
        all_changepoints = []
        series_weights = []
        
        for col, cps in detector.changepoints_.items():
            all_changepoints.extend(cps)
            series_weights.extend([col] * len(cps))
        
        if len(all_changepoints) == 0:
            return {
                'market_changepoints': np.array([]), 
                'individual_changepoints': detector.changepoints_,
                'detector': detector
            }
        
        # Count occurrences of nearby changepoints
        tolerance = clustering_params.get('tolerance', 3)  # Time periods
        market_changepoints = []
        
        unique_cps = np.unique(all_changepoints)
        for cp in unique_cps:
            # Count how many series have a changepoint within tolerance
            nearby_count = 0
            nearby_series = set()
            
            for col, cps in detector.changepoints_.items():
                if any(abs(other_cp - cp) <= tolerance for other_cp in cps):
                    nearby_count += 1
                    nearby_series.add(col)
            
            agreement_ratio = nearby_count / len(df.columns)
            if agreement_ratio >= min_series_agreement:
                market_changepoints.append(cp)
        
        return {
            'market_changepoints': np.array(sorted(market_changepoints)),
            'individual_changepoints': detector.changepoints_,
            'detector': detector
        }
    
    else:
        # Use clustering-based approach
        market_cps = detector.get_market_changepoints(method=clustering_method, params=clustering_params)
        return {
            'market_changepoints': market_cps,
            'individual_changepoints': detector.changepoints_,
            'detector': detector
        }
