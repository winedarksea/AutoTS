import random
import numpy as np
import pandas as pd
from autots.tools.shaping import infer_frequency


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
    data = np.asarray(data)
    n = len(data)
    if n < 2 * min_segment_length:
        return np.array([])
    
    # Initialize cost and optimal segmentation arrays
    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    cp = np.zeros(n + 1, dtype=int)
    
    # PELT algorithm
    R = [0]  # Set of potential changepoints
    
    for t in range(1, n + 1):
        candidates = []
        for s in R:
            if t - s >= min_segment_length:
                cost = _calculate_segment_cost(data[s:t], loss_function)
                total_cost = F[s] + cost + penalty
                candidates.append((total_cost, s))
        
        if candidates:
            F[t], cp[t] = min(candidates)
            
            # Pruning step - keep only competitive changepoints
            R_new = []
            for s in R:
                # A changepoint s is kept if it could potentially be optimal for future t
                if F[s] <= F[t]:  # Simplified pruning condition
                    R_new.append(s)
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
    method (str): Method for changepoint detection ('basic', 'pelt', 'l1_fused_lasso', 'l1_total_variation')
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


class ChangePointDetector(object):
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
            method (str): Changepoint detection method ('pelt', 'l1_fused_lasso', 'l1_total_variation', 'composite_fused_lasso')
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
        
    def detect(self, df):
        """
        Run changepoint detection on wide-format time series data.
        
        Args:
            df (pd.DataFrame): Wide-format time series with DatetimeIndex
        """
        self.df = df.copy()
        self.df_cols = df.columns
        
        if self.aggregate_method == 'individual':
            # Detect changepoints for each series individually
            self.changepoints_ = {}
            self.fitted_trends_ = {}
            
            for col in df.columns:
                data = df[col].dropna().values
                if len(data) < 2 * self.min_segment_length:
                    self.changepoints_[col] = np.array([])
                    self.fitted_trends_[col] = data
                    continue
                    
                if self.method == 'pelt':
                    penalty = self.method_params.get('penalty', 10)
                    loss_function = self.method_params.get('loss_function', 'l2')
                    changepoints = _detect_pelt_changepoints(
                        data, penalty, loss_function, self.min_segment_length
                    )
                    self.changepoints_[col] = changepoints
                    self.fitted_trends_[col] = data  # Could fit segment means
                    
                elif self.method in ['l1_fused_lasso', 'l1_total_variation']:
                    lambda_reg = self.method_params.get('lambda_reg', 1.0)
                    l1_method = 'fused_lasso' if self.method == 'l1_fused_lasso' else 'total_variation'
                    changepoints, fitted_trend = _detect_l1_trend_changepoints(
                        data, lambda_reg, l1_method
                    )
                    self.changepoints_[col] = changepoints
                    self.fitted_trends_[col] = fitted_trend
                    
                elif self.method == 'composite_fused_lasso':
                    self.changepoints_[col], self.fitted_trends_[col] = self._detect_composite_fused_lasso(data)
                
                elif self.method == 'basic':
                    # Basic evenly-spaced changepoints (legacy method)
                    changepoint_spacing = self.method_params.get('changepoint_spacing', 60)
                    changepoint_distance_end = self.method_params.get('changepoint_distance_end', 120)
                    # Convert to changepoint indices
                    n = len(data)
                    changepoint_range_end = max(1, min(n - changepoint_distance_end, n - 1))
                    if changepoint_range_end <= 0:
                        changepoints = np.array([0])
                    else:
                        changepoints = np.arange(0, changepoint_range_end, changepoint_spacing)
                        changepoints = np.append(changepoints, changepoint_range_end)
                    self.changepoints_[col] = changepoints
                    self.fitted_trends_[col] = data
                
                # Handle probabilistic output
                if self.probabilistic_output:
                    if self.changepoint_probabilities_ is None:
                        self.changepoint_probabilities_ = {}
                    prob_method = self.method_params.get('probabilistic_method', 'bootstrap')
                    self.changepoint_probabilities_[col], prob_cps = self._detect_probabilistic_changepoints(data, prob_method)
                    # Update changepoints with probabilistic results if requested
                    if self.method_params.get('use_probabilistic_changepoints', False):
                        self.changepoints_[col] = prob_cps
                    
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
            n_bootstrap = self.method_params.get('n_bootstrap', 100)
            bootstrap_changepoints = []
            
            for _ in range(n_bootstrap):
                # Resample data with replacement
                bootstrap_indices = np.random.choice(n, size=n, replace=True)
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
    
    def get_new_params(self, method="random"):
        """Generate new random parameters for changepoint detection."""
        method_options = ['pelt', 'l1_fused_lasso', 'l1_total_variation', 'composite_fused_lasso']
        
        new_method = random.choice(method_options)
        
        if new_method == 'pelt':
            new_params = {
                'penalty': random.choice([1, 5, 10, 20, 50]),
                'loss_function': random.choice(['l2', 'l1', 'huber']),
            }
        elif new_method in ['l1_fused_lasso', 'l1_total_variation']:
            new_params = {
                'lambda_reg': random.choice([0.1, 0.5, 1.0, 2.0, 5.0]),
            }
        elif new_method == 'composite_fused_lasso':
            new_params = {
                'lambda_level': random.choice([0.1, 0.5, 1.0, 2.0]),
                'lambda_slope': random.choice([0.1, 0.5, 1.0, 2.0]),
            }
        else:
            new_params = {}
        
        return {
            'method': new_method,
            'method_params': new_params,
            'aggregate_method': random.choice(['mean', 'median', 'individual']),
            'min_segment_length': random.choice([3, 5, 10, 15]),
            'probabilistic_output': random.choice([True, False]),
        }


def generate_random_changepoint_params(method='random'):
    """
    Generate random parameters for changepoint detection methods.
    
    This function creates appropriately weighted random parameters for different
    changepoint detection algorithms, supporting the flexible method/params system.
    
    Args:
        method (str): Method for parameter selection (currently only 'random' supported)
        
    Returns:
        tuple: (changepoint_method, changepoint_params) where
            - changepoint_method (str): Selected method name
            - changepoint_params (dict): Method-specific parameters
    """
    import random
    
    # Changepoint method options - balanced weights now that all methods work
    changepoint_methods = ['basic', 'pelt', 'l1_fused_lasso', 'l1_total_variation']
    changepoint_method_weights = [0.5, 0.3, 0.1, 0.1]
    
    # Select method
    selected_method = random.choices(changepoint_methods, weights=changepoint_method_weights, k=1)[0]
    
    # Generate method-specific parameters
    changepoint_params = {}
    
    if selected_method == "basic":
        # Basic method parameters (legacy style)
        spacing_options = [6, 28, 60, 90, 120, 180, 360, 5040]
        spacing_weights = [0.05, 0.1, 0.1, 0.1, 0.05, 0.2, 0.1, 0.3]
        distance_end_options = [6, 28, 60, 90, 180, 360, 520, 5040]
        distance_end_weights = [0.05, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.3]
        
        changepoint_params = {
            "changepoint_spacing": random.choices(spacing_options, weights=spacing_weights, k=1)[0],
            "changepoint_distance_end": random.choices(distance_end_options, weights=distance_end_weights, k=1)[0],
        }
        
    elif selected_method == "pelt":
        # PELT method parameters
        penalty_options = [1, 5, 10, 20, 50, 100]
        penalty_weights = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]
        loss_functions = ['l1', 'l2', 'huber']
        loss_weights = [0.3, 0.5, 0.2]
        min_segment_options = [1, 2, 5, 10]
        min_segment_weights = [0.4, 0.3, 0.2, 0.1]
        
        changepoint_params = {
            "penalty": random.choices(penalty_options, weights=penalty_weights, k=1)[0],
            "loss_function": random.choices(loss_functions, weights=loss_weights, k=1)[0],
            "min_segment_length": random.choices(min_segment_options, weights=min_segment_weights, k=1)[0],
        }
        
    elif selected_method in ["l1_fused_lasso", "l1_total_variation"]:
        # L1 trend filtering parameters
        lambda_options = [0.01, 0.1, 1.0, 10.0, 100.0]
        lambda_weights = [0.1, 0.2, 0.4, 0.2, 0.1]
        
        changepoint_params = {
            "lambda_reg": random.choices(lambda_options, weights=lambda_weights, k=1)[0],
        }
    
    return selected_method, changepoint_params


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
    detector = ChangePointDetector(**detector_params)
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
