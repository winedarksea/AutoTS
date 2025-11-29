import pandas as pd
import numpy as np
import datetime

from autots.tools.seasonal import date_part
from autots.models.base import ModelObject, PredictionObject
from autots.tools.seasonal import (
    date_part,
    random_datepart,
)
from autots.tools.changepoints import (
    create_changepoint_features,
    changepoint_fcst_from_last_row,
    half_yr_spacing,
    generate_random_changepoint_params,
    find_market_changepoints_multivariate,
    ChangepointDetector,
)
from autots.tools.window_functions import window_maker, last_window, sliding_window_view
from autots.tools.shaping import infer_frequency

# this is done to allow users to use the rest of AutoTS without these libraries installed
try:
    from scipy.stats import norm
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
except Exception:
    from autots.tools.mocks import norm, tqdm, StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Module
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    HAS_TORCH = True
except Exception:
    from autots.tools.mocks import Module, Dataset, DataLoader, TensorDataset
    HAS_TORCH = False


# TODO: consider instead of passing changepoint features, pass the fit changepoint (from TimeSeriesFeatureDetector)
# TODO: take advantage of the 2d feature shape with convolutional designs


def build_series_feature_mapping(feature_columns, series_names, changepoint_features_cols):
    """
    Build a mapping from series index to their corresponding feature column indices.

    This handles per-series feature columns (changepoints, naive features, etc.)
    so that each series only receives its own feature subset while shared features
    remain available to all series.
    
    Args:
        feature_columns: All feature column names
        series_names: List of series names in order
        changepoint_features_cols: Changepoint feature column names
        
    Returns:
        tuple: (series_feat_mapping, has_per_series_features)
            - series_feat_mapping: dict mapping series_idx -> list of feature column indices
            - has_per_series_features: bool indicating if per-series features were detected
    """
    if len(feature_columns) == 0:
        return None, False

    # Normalize to strings for reliable comparisons
    feature_columns_list = list(feature_columns)
    feature_columns_str = np.array([str(col) for col in feature_columns_list])
    series_names_str = [str(name) for name in series_names]
    n_features = len(feature_columns_str)
    n_series = len(series_names_str)

    # Vectorized per-series feature detection
    # Create boolean matrix: [n_features, n_series] where True means feature belongs to series
    feature_series_matrix = np.zeros((n_features, n_series), dtype=bool)
    
    # Process all series at once for each pattern type
    for series_idx, series_name in enumerate(series_names_str):
        # Pattern 1: Naive feature - exact match (vectorized)
        naive_match = feature_columns_str == f"naive_last_{series_name}"
        
        # Pattern 2: Prefix pattern - starts with series_name followed by non-alphanumeric (vectorized)
        # Use vectorized string operations where possible
        prefix_match = np.zeros(n_features, dtype=bool)
        starts_with_series = np.char.startswith(feature_columns_str.astype(str), series_name)
        # Check for exact match or non-alphanumeric following character
        for feat_idx in np.where(starts_with_series)[0]:
            col_str = feature_columns_str[feat_idx]
            if len(col_str) == len(series_name) or not col_str[len(series_name)].isalnum():
                prefix_match[feat_idx] = True
        
        # Pattern 3: Suffix pattern - ends with _series_name (vectorized)
        suffix_pattern = f"_{series_name}"
        suffix_match = np.char.endswith(feature_columns_str.astype(str), suffix_pattern)
        
        # Combine all patterns for this series
        feature_series_matrix[:, series_idx] = naive_match | prefix_match | suffix_match
    
    # Detect if any per-series features exist
    has_per_series = np.any(feature_series_matrix)
    
    # Fallback: inspect changepoint columns if detection fails (legacy naming safeguards)
    if (
        not has_per_series
        and changepoint_features_cols is not None
        and len(changepoint_features_cols) > 0
    ):
        cp_cols_str = [str(col) for col in changepoint_features_cols]
        for series_name in series_names_str:
            for col_str in cp_cols_str:
                if (col_str == f"naive_last_{series_name}" or 
                    col_str.endswith(f"_{series_name}") or
                    (col_str.startswith(series_name) and 
                     (len(col_str) == len(series_name) or not col_str[len(series_name)].isalnum()))):
                    has_per_series = True
                    break
            if has_per_series:
                break

    if not has_per_series:
        return None, False

    # Build the mapping efficiently using vectorized operations
    # Shared features: those that don't belong to any specific series
    is_shared = ~np.any(feature_series_matrix, axis=1)
    shared_indices = np.where(is_shared)[0]
    
    series_feat_mapping = {}
    for series_idx in range(n_series):
        # Get series-specific feature indices
        series_specific_indices = np.where(feature_series_matrix[:, series_idx])[0]
        
        # Combine shared and series-specific features
        # Use concatenate for efficiency, then convert to list
        series_feature_indices = np.concatenate([shared_indices, series_specific_indices])
        series_feature_indices.sort()  # Maintain order for consistency
        
        series_feat_mapping[series_idx] = series_feature_indices.tolist()

    return series_feat_mapping, True


def create_training_batches(
    y_train_scaled, 
    train_feats_scaled, 
    prediction_batch_size, 
    max_samples=50000, 
    random_seed=2023,
    series_feat_mapping=None,
    series_names=None,
    naive_feature_indices=None,
    use_naive_window=False,
    y_train_raw=None,
    feature_means=None,
    feature_scales=None,
):
    """
    Create training batches for models that predict multiple future steps at once.
    
    Each sample contains features for prediction_batch_size future timesteps,
    with naive features properly handled to avoid data leakage.
    
    Args:
        y_train_scaled: Scaled time series data (n_timesteps, n_series)
        train_feats_scaled: Scaled feature data (n_timesteps, n_features)
        prediction_batch_size: Number of timesteps to predict in each batch
        max_samples: Maximum number of training samples to generate
        random_seed: Random seed for sampling
        series_feat_mapping: Optional dict mapping series index to feature column indices
        series_names: Optional list of series names (for per-series features)
        naive_feature_indices: Indices of naive features (if single value) or window features (if use_naive_window=True)
        use_naive_window: If True, naive features are windows that are already lagged in train_feats_scaled
        y_train_raw: Optional raw target values (unscaled) for building naive windows without the shift bug
        feature_means: Optional per-feature mean from the fitted feature scaler
        feature_scales: Optional per-feature scale from the fitted feature scaler
        
    Returns:
        tuple: (X_data, Y_data) where:
            - X_data has shape (n_samples, prediction_batch_size, n_features)
            - Y_data has shape (n_samples, prediction_batch_size, 1)
    """
    n_timesteps, n_series = y_train_scaled.shape
    n_features = train_feats_scaled.shape[1]
    
    # Validate that we have enough data
    # When using naive windows, we need prediction_batch_size extra for the initial window
    min_required = (2 * prediction_batch_size) if use_naive_window else (prediction_batch_size + 1)
    if n_timesteps < min_required:
        raise ValueError(
            f"Training data has {n_timesteps} timesteps but need at least {min_required}. "
            f"Either increase training data length or decrease prediction_batch_size."
        )
    
    # When using naive windows, exclude first prediction_batch_size timesteps
    # since they don't have complete windows before them
    start_offset = prediction_batch_size if use_naive_window else 0
    
    # Calculate maximum possible samples (excluding the offset)
    max_possible_samples = (n_timesteps - prediction_batch_size - start_offset) * n_series
    max_samples = min(max_samples, max_possible_samples)
    
    # Calculate maximum possible samples (excluding the offset)
    max_possible_samples = (n_timesteps - prediction_batch_size - start_offset) * n_series
    max_samples = min(max_samples, max_possible_samples)
    
    # Generate sample indices
    rng = np.random.default_rng(random_seed)
    if max_possible_samples > max_samples:
        selected_indices = rng.choice(max_possible_samples, size=max_samples, replace=False)
    else:
        selected_indices = np.arange(max_possible_samples)
    
    # Calculate which series and starting timestep for each sample
    # Add start_offset to ensure we skip the first prediction_batch_size timesteps when using naive windows
    series_indices = selected_indices % n_series
    start_indices = (selected_indices // n_series) + start_offset
    
    if series_feat_mapping is not None and series_names is not None:
        # Per-series features: different feature subsets per series
        max_feats = max(len(feat_cols) for feat_cols in series_feat_mapping.values())
        X_data = np.zeros((len(selected_indices), prediction_batch_size, max_feats), dtype=np.float32)
    else:
        X_data = np.zeros((len(selected_indices), prediction_batch_size, n_features), dtype=np.float32)
    
    Y_data = np.zeros((len(selected_indices), prediction_batch_size, 1), dtype=np.float32)
    
    # Fill in data for each sample
    for i, (start_idx, series_idx) in enumerate(zip(start_indices, series_indices)):
        end_idx = start_idx + prediction_batch_size
        
        # Get target values for this series and window
        Y_data[i, :, 0] = y_train_scaled[start_idx:end_idx, series_idx]
        
        if series_feat_mapping is not None and series_names is not None:
            # Get feature indices for this series
            feat_indices = series_feat_mapping.get(series_idx, [])
            if len(feat_indices) > 0:
                # Get features for this window
                batch_features = train_feats_scaled[start_idx:end_idx, feat_indices].copy()
                
                # For window-based naive features, populate with proper historical window
                if use_naive_window and naive_feature_indices is not None:
                    for naive_idx in naive_feature_indices:
                        if naive_idx in feat_indices:
                            # Find position in the subset
                            local_idx = feat_indices.index(naive_idx)
                            # Fill this feature with window values: [t-1, t-2, ..., t-prediction_batch_size]
                            # where t is start_idx (the beginning of the prediction window)
                            # REVERSED so most recent value comes first
                            if start_idx >= prediction_batch_size:
                                # Get window from [start_idx-prediction_batch_size : start_idx]
                                if (
                                    y_train_raw is not None
                                    and feature_means is not None
                                    and feature_scales is not None
                                ):
                                    # Use true targets, then scale to feature space to avoid shifted naive column
                                    window_raw = y_train_raw[
                                        start_idx - prediction_batch_size : start_idx,
                                        series_idx,
                                    ]
                                    window_values = (
                                        window_raw - feature_means[naive_idx]
                                    ) / feature_scales[naive_idx]
                                else:
                                    window_values = train_feats_scaled[
                                        start_idx - prediction_batch_size : start_idx,
                                        naive_idx,
                                    ]
                                # Reverse so most recent (t-1) is at index 0
                                batch_features[:, local_idx] = window_values[::-1]
                            else:
                                # Edge case: not enough history, pad with earliest available values
                                if (
                                    y_train_raw is not None
                                    and feature_means is not None
                                    and feature_scales is not None
                                ):
                                    available = (
                                        y_train_raw[:start_idx, series_idx]
                                        - feature_means[naive_idx]
                                    ) / feature_scales[naive_idx]
                                else:
                                    available = train_feats_scaled[:start_idx, naive_idx]
                                if len(available) > 0:
                                    padded = np.pad(
                                        available, 
                                        (prediction_batch_size - len(available), 0), 
                                        mode='edge'
                                    )
                                    # Reverse so most recent is first
                                    batch_features[:, local_idx] = padded[::-1]
                
                X_data[i, :, :len(feat_indices)] = batch_features
        else:
            # Shared features across all series
            batch_features = train_feats_scaled[start_idx:end_idx, :].copy()
            
            # For window-based naive features, populate with proper historical window
            if use_naive_window and naive_feature_indices is not None:
                for naive_idx in naive_feature_indices:
                    # Fill this feature with window values for the corresponding series
                    # Since this is shared features, we need to fill with THIS series' historical values
                    if start_idx >= prediction_batch_size:
                        # Get window from [start_idx-prediction_batch_size : start_idx]
                        if (
                            y_train_raw is not None
                            and feature_means is not None
                            and feature_scales is not None
                        ):
                            window_raw = y_train_raw[
                                start_idx - prediction_batch_size : start_idx,
                                series_idx,
                            ]
                            window_values = (
                                window_raw - feature_means[naive_idx]
                            ) / feature_scales[naive_idx]
                        else:
                            window_values = train_feats_scaled[
                                start_idx - prediction_batch_size : start_idx,
                                naive_idx,
                            ]
                        # Reverse so most recent (t-1) is at index 0
                        batch_features[:, naive_idx] = window_values[::-1]
                    else:
                        # Edge case: not enough history, pad with earliest available values
                        if (
                            y_train_raw is not None
                            and feature_means is not None
                            and feature_scales is not None
                        ):
                            available = (
                                y_train_raw[:start_idx, series_idx]
                                - feature_means[naive_idx]
                            ) / feature_scales[naive_idx]
                        else:
                            available = train_feats_scaled[:start_idx, naive_idx]
                        if len(available) > 0:
                            padded = np.pad(
                                available, 
                                (prediction_batch_size - len(available), 0), 
                                mode='edge'
                            )
                            # Reverse so most recent is first
                            batch_features[:, naive_idx] = padded[::-1]
            elif not use_naive_window and naive_feature_indices is not None:
                # Legacy single-value naive features
                for naive_idx in naive_feature_indices:
                    if start_idx > 0:
                        naive_val = train_feats_scaled[start_idx - 1, naive_idx]
                    else:
                        naive_val = train_feats_scaled[0, naive_idx]
                    batch_features[:, naive_idx] = naive_val
            
            X_data[i, :, :] = batch_features
    
    return X_data, Y_data


class CombinedNLLWassersteinLoss(Module):
    def __init__(self, nll_weight=1.0, wasserstein_weight=0.1):
        super().__init__()
        self.nll_weight = nll_weight
        self.wasserstein_weight = wasserstein_weight
        self.nll_loss = nn.GaussianNLLLoss(reduction="mean")

    def _wasserstein_distance(self, p, q):
        """Compute 1-Wasserstein distance between flattened tensors."""
        p_sorted, _ = torch.sort(p.flatten(), dim=0)
        q_sorted, _ = torch.sort(q.flatten(), dim=0)
        return torch.abs(p_sorted - q_sorted).mean()

    def forward(self, mu, sigma, y_true):
        """
        Combined Gaussian NLL and Wasserstein distance loss.

        Args:
            mu: Mean predictions (any shape)
            sigma: Standard deviation predictions (same shape as mu)
            y_true: True values (same shape as mu)

        Returns:
            Scalar loss value combining NLL and Wasserstein distance
        """
        sigma = torch.clamp(sigma, min=1e-6)
        nll = self.nll_loss(mu, y_true, sigma.pow(2))
        wasserstein = self._wasserstein_distance(mu, y_true)
        return self.nll_weight * nll + self.wasserstein_weight * wasserstein


class QuantileLoss(Module):
    def __init__(self, quantiles=None, prediction_interval=0.95):
        super().__init__()

        # Auto-construct quantiles from prediction interval if not provided
        if quantiles is None:
            alpha = 1 - prediction_interval
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2
            self.quantiles = [lower_q, 0.5, upper_q]
        else:
            self.quantiles = quantiles

    def forward(self, mu, sigma, y_true):
        """
        Quantile loss (pinball loss) for probabilistic forecasting.

        Args:
            mu: Mean predictions (any shape)
            sigma: Standard deviation predictions (same shape as mu)
            y_true: True values (same shape as mu)

        Returns:
            Scalar loss value
        """
        # Convert Gaussian params to quantiles
        sigma = torch.clamp(sigma, min=1e-6)
        q_tensor = torch.as_tensor(self.quantiles, device=mu.device, dtype=mu.dtype)
        z_tensor = torch.erfinv(q_tensor.mul(2).sub(1)) * np.sqrt(2)

        mu_expanded = mu.unsqueeze(-1)
        sigma_expanded = sigma.unsqueeze(-1)
        y_expanded = y_true.unsqueeze(-1)

        q_pred = mu_expanded + sigma_expanded * z_tensor
        error = y_expanded - q_pred
        losses = torch.maximum(q_tensor * error, (q_tensor - 1) * error)
        return losses.mean()


class CRPSLoss(Module):
    def __init__(self):
        super().__init__()
        self.sqrt_pi = np.sqrt(np.pi)
        self.sqrt_2 = np.sqrt(2)

    def forward(self, mu, sigma, y_true):
        """
        Continuous Ranked Probability Score (CRPS) for Gaussian distributions.

        CRPS is a proper scoring rule that measures the difference between
        the predicted cumulative distribution and the empirical distribution
        of the observations. Lower values are better.

        Args:
            mu: Mean predictions (any shape)
            sigma: Standard deviation predictions (same shape as mu)
            y_true: True values (same shape as mu)

        Returns:
            Scalar CRPS loss (mean over all predictions)
        """
        # Ensure positive sigma
        sigma = torch.clamp(sigma, min=1e-6)
        standardized = (y_true - mu) / sigma

        # Normal PDF and CDF using error function
        phi = torch.exp(-0.5 * standardized.pow(2)) / np.sqrt(2 * np.pi)
        Phi = 0.5 * (1 + torch.erf(standardized / self.sqrt_2))

        # Analytical CRPS formula for Gaussian distribution
        crps = sigma * (standardized * (2 * Phi - 1) + 2 * phi - 1 / self.sqrt_pi)
        return crps.mean()


class EnergyScore(Module):
    def __init__(self, n_samples=30, beta=1.0):
        super().__init__()
        self.n_samples = n_samples
        self.beta = beta  # Power parameter for generalized energy score

    def forward(self, mu, sigma, y_true):
        """
        Energy Score for probabilistic forecasting.

        The energy score is a proper scoring rule that generalizes CRPS.
        It's more robust to outliers than CRPS but computationally more expensive
        due to Monte Carlo sampling.

        Args:
            mu: Mean predictions (any shape)
            sigma: Standard deviation predictions (same shape as mu)
            y_true: True values (same shape as mu)

        Returns:
            Scalar energy score (lower is better)
        """
        sigma = torch.clamp(sigma, min=1e-6)

        # Sample from predicted Gaussian distribution
        eps = torch.randn(self.n_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
        samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

        # Flatten samples/targets for distance computations
        samples_flat = samples.reshape(self.n_samples, -1)
        target_flat = y_true.reshape(1, -1)

        # Energy score: E[||X - y||^β] - 0.5 * E[||X - X'||^β]
        distances = torch.norm(samples_flat - target_flat, dim=1)
        if self.beta != 1.0:
            distances = distances.pow(self.beta)
        term1 = distances.mean()

        pairwise_distances = torch.pdist(samples_flat, p=2)
        if pairwise_distances.numel() == 0:
            term2 = torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
        else:
            if self.beta != 1.0:
                pairwise_distances = pairwise_distances.pow(self.beta)
            term2 = 0.5 * pairwise_distances.mean()

        return term1 - term2


class RegularizedGaussianNLL(Module):
    def __init__(self, sigma_reg_weight=0.01, reg_type="l1"):
        super().__init__()
        self.sigma_reg_weight = sigma_reg_weight
        self.reg_type = reg_type
        self.nll_loss = nn.GaussianNLLLoss(reduction="mean")

    def forward(self, mu, sigma, y_true):
        """
        Gaussian Negative Log-Likelihood with sigma regularization.

        Adds regularization on sigma to prevent overconfident predictions
        (very small sigma values). This encourages the model to maintain
        reasonable uncertainty estimates.

        Args:
            mu: Mean predictions (any shape)
            sigma: Standard deviation predictions (same shape as mu)
            y_true: True values (same shape as mu)

        Returns:
            Scalar loss combining NLL and sigma regularization
        """
        sigma = torch.clamp(sigma, min=1e-6)

        # Standard Gaussian NLL
        nll = self.nll_loss(mu, y_true, sigma.pow(2))

        # Sigma regularization to prevent overconfident predictions
        if self.reg_type == "l1":
            sigma_reg = self.sigma_reg_weight * sigma.mean()
        elif self.reg_type == "l2":
            sigma_reg = self.sigma_reg_weight * sigma.pow(2).mean()
        elif self.reg_type == "inv":
            # Inverse regularization: penalize very small sigma
            sigma_reg = self.sigma_reg_weight * (1.0 / sigma).mean()
        else:
            raise ValueError(f"Unknown regularization type: {self.reg_type}")

        return nll + sigma_reg


class RankedSharpeLoss(Module):
    def __init__(self, nll_weight=0.7, rank_weight=0.3, temperature=1.0, use_kendall=False):
        """
        Loss function optimized for ranked Sharpe metric performance.
        
        Combines Gaussian NLL for probabilistic anchoring with rank correlation
        objectives to optimize cross-series relative performance.
        
        Args:
            nll_weight: Weight for Gaussian NLL anchoring term
            rank_weight: Weight for rank correlation term
            temperature: Temperature parameter for soft ranking (higher = softer)
            use_kendall: If True, uses Kendall's tau approximation; if False, uses Spearman
        """
        super().__init__()
        self.nll_weight = nll_weight
        self.rank_weight = rank_weight
        self.temperature = temperature
        self.use_kendall = use_kendall
        self.nll_loss = nn.GaussianNLLLoss(reduction="mean")
        
    def _soft_rank(self, x, temperature=1.0):
        """
        Compute differentiable soft ranks using temperature-scaled sigmoid.
        
        For short horizon forecasting, we need stable gradients that preserve
        the relative ordering information critical for ranked Sharpe.
        """
        # x shape: (batch_size,) - predictions for all series at one timestep
        n = x.shape[0]
        
        # Create pairwise comparison matrix
        # diff[i,j] = x[i] - x[j]
        x_expanded = x.unsqueeze(1)  # (n, 1)
        diff = x_expanded - x_expanded.T  # (n, n)
        
        # Soft ranks using sigmoid: higher values get lower ranks (ranks start from 1)
        # sigmoid(diff/temp) ≈ 1 when x[i] > x[j], ≈ 0 when x[i] < x[j]
        soft_comparison = torch.sigmoid(diff / temperature)
        
        # Sum gives approximate rank (higher values = higher ranks)
        # Add 1 to make ranks start from 1 instead of 0
        soft_ranks = soft_comparison.sum(dim=1) + 1
        
        return soft_ranks
    
    def _kendall_tau_loss(self, pred_ranks, true_ranks):
        """
        Compute differentiable approximation of Kendall's tau correlation.
        
        Kendall's tau is more robust for short sequences and focuses on
        pairwise concordance, which aligns well with ranked Sharpe objectives.
        """
        n = pred_ranks.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=pred_ranks.device)
        
        # Create all pairwise differences
        pred_diff = pred_ranks.unsqueeze(1) - pred_ranks.unsqueeze(0)  # (n, n)
        true_diff = true_ranks.unsqueeze(1) - true_ranks.unsqueeze(0)  # (n, n)
        
        # Concordant pairs: same sign in both differences
        # Use tanh to make it differentiable
        concordant = torch.tanh(pred_diff / self.temperature) * torch.tanh(true_diff / self.temperature)
        
        # Sum over upper triangle (avoid diagonal and duplicates)
        mask = torch.triu(torch.ones_like(concordant), diagonal=1)
        kendall = (concordant * mask).sum() / mask.sum()
        
        # Return negative for minimization (we want to maximize correlation)
        return -kendall
    
    def _spearman_loss(self, pred_ranks, true_ranks):
        """
        Compute differentiable Spearman correlation using soft ranks.
        
        This matches the Spearman correlation used in your ranked Sharpe metric.
        """
        # Center the ranks
        pred_centered = pred_ranks - pred_ranks.mean()
        true_centered = true_ranks - true_ranks.mean()
        
        # Compute correlation
        numerator = (pred_centered * true_centered).sum()
        pred_var = (pred_centered ** 2).sum()
        true_var = (true_centered ** 2).sum()
        
        # Add epsilon for numerical stability
        epsilon = 1e-8
        correlation = numerator / (torch.sqrt(pred_var * true_var) + epsilon)
        
        # Return negative for minimization
        return -correlation
    
    def forward(self, mu, sigma, y_true):
        """
        Forward pass combining NLL anchoring with rank correlation optimization.
        
        Args:
            mu: Mean predictions, shape (batch_size, 1) or (batch_size,)
            sigma: Std predictions, same shape as mu
            y_true: True values, same shape as mu
            
        Returns:
            Combined loss value (scalar)
        """
        # Ensure proper shapes
        if mu.dim() > 1:
            mu = mu.squeeze(-1)
        if sigma.dim() > 1:
            sigma = sigma.squeeze(-1)
        if y_true.dim() > 1:
            y_true = y_true.squeeze(-1)
            
        sigma = torch.clamp(sigma, min=1e-6)
        
        # 1. Probabilistic anchoring term (Gaussian NLL)
        nll = self.nll_loss(mu.unsqueeze(-1), y_true.unsqueeze(-1), sigma.pow(2).unsqueeze(-1))
        
        # 2. Rank correlation term
        # For short horizon forecasting, we focus on cross-series ranking at each timestep
        if mu.shape[0] >= 2:  # Need at least 2 series for ranking
            # Compute soft ranks for predictions and actuals
            pred_ranks = self._soft_rank(mu, self.temperature)
            true_ranks = self._soft_rank(y_true, self.temperature)
            
            # Choose correlation method
            if self.use_kendall:
                rank_loss = self._kendall_tau_loss(pred_ranks, true_ranks)
            else:
                rank_loss = self._spearman_loss(pred_ranks, true_ranks)
        else:
            rank_loss = torch.tensor(0.0, device=mu.device)
        
        # Combine losses
        total_loss = self.nll_weight * nll + self.rank_weight * rank_loss
        
        return total_loss


class ShortHorizonRankLoss(Module):
    def __init__(self, nll_weight=0.6, rank_weight=0.3, consistency_weight=0.1, 
                 temperature=0.5, horizon_steps=4):
        """
        Specialized loss for short horizon (≤4 steps) ranked Sharpe optimization.
        
        Adds temporal consistency term to ensure predictions maintain relative
        rankings across the short forecast horizon.
        
        Args:
            nll_weight: Weight for probabilistic anchoring
            rank_weight: Weight for cross-series rank correlation
            consistency_weight: Weight for temporal ranking consistency
            temperature: Temperature for soft ranking
            horizon_steps: Number of forecast steps to optimize for
        """
        super().__init__()
        self.nll_weight = nll_weight
        self.rank_weight = rank_weight
        self.consistency_weight = consistency_weight
        self.temperature = temperature
        self.horizon_steps = horizon_steps
        self.nll_loss = nn.GaussianNLLLoss(reduction="mean")
        
        # Store recent predictions for temporal consistency
        self.register_buffer('recent_predictions', torch.zeros(horizon_steps, 1))
        self.register_buffer('recent_actuals', torch.zeros(horizon_steps, 1))
        self.step_count = 0
        
    def _soft_rank(self, x, temperature=1.0):
        """Compute differentiable soft ranks."""
        n = x.shape[0]
        if n < 2:
            return x  # Return original if can't rank
            
        x_expanded = x.unsqueeze(1)
        diff = x_expanded - x_expanded.T
        soft_comparison = torch.sigmoid(diff / temperature)
        soft_ranks = soft_comparison.sum(dim=1) + 1
        return soft_ranks
        
    def _rank_correlation_loss(self, pred_ranks, true_ranks):
        """Compute Spearman correlation loss."""
        pred_centered = pred_ranks - pred_ranks.mean()
        true_centered = true_ranks - true_ranks.mean()
        
        numerator = (pred_centered * true_centered).sum()
        pred_var = (pred_centered ** 2).sum()
        true_var = (true_centered ** 2).sum()
        
        epsilon = 1e-8
        correlation = numerator / (torch.sqrt(pred_var * true_var) + epsilon)
        return -correlation
        
    def _temporal_consistency_loss(self, current_pred_ranks):
        """
        Penalize sudden changes in relative rankings across timesteps.
        
        For short horizons, maintaining ranking consistency helps achieve
        better Sharpe ratios by reducing ranking volatility.
        """
        if self.step_count < 2:
            return torch.tensor(0.0, device=current_pred_ranks.device)
            
        # Get the most recent stored prediction ranks
        if hasattr(self, '_prev_pred_ranks'):
            prev_ranks = self._prev_pred_ranks
            
            # Compute ranking change penalty
            rank_diff = torch.abs(current_pred_ranks - prev_ranks)
            consistency_loss = rank_diff.mean()
            
            return consistency_loss
        
        return torch.tensor(0.0, device=current_pred_ranks.device)
    
    def forward(self, mu, sigma, y_true):
        """
        Forward pass with temporal consistency for short horizon optimization.
        """
        # Ensure proper shapes
        if mu.dim() > 1:
            mu = mu.squeeze(-1)
        if sigma.dim() > 1:
            sigma = sigma.squeeze(-1)
        if y_true.dim() > 1:
            y_true = y_true.squeeze(-1)
            
        sigma = torch.clamp(sigma, min=1e-6)
        
        # 1. Probabilistic anchoring
        nll = self.nll_loss(mu.unsqueeze(-1), y_true.unsqueeze(-1), sigma.pow(2).unsqueeze(-1))
        
        # 2. Cross-series rank correlation
        rank_loss = torch.tensor(0.0, device=mu.device)
        consistency_loss = torch.tensor(0.0, device=mu.device)
        
        if mu.shape[0] >= 2:
            pred_ranks = self._soft_rank(mu, self.temperature)
            true_ranks = self._soft_rank(y_true, self.temperature)
            
            rank_loss = self._rank_correlation_loss(pred_ranks, true_ranks)
            
            # 3. Temporal consistency (only during training)
            if self.training:
                consistency_loss = self._temporal_consistency_loss(pred_ranks)
                # Store current ranks for next timestep
                self._prev_pred_ranks = pred_ranks.detach()
                self.step_count += 1
        
        # Combine all terms
        total_loss = (self.nll_weight * nll + 
                     self.rank_weight * rank_loss + 
                     self.consistency_weight * consistency_loss)
        
        return total_loss


# Self-contained Mamba Block and Core Model
class MambaMinimalBlock(Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_extra_gating=False):
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = (
            d_model,
            d_state,
            d_conv,
            expand,
        )
        self.use_extra_gating = use_extra_gating
        self.d_inner = int(self.expand * self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Optional extra gating mechanism
        if self.use_extra_gating:
            self.gate_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A_log = (
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(1)
            .repeat(1, self.d_inner)
            .T
        )
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # More efficient conv1d with fewer operations
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = F.silu(x)
        y = self.ssm(x)

        # Fuse gating and output projection to reduce operations
        y = y * F.silu(z)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        B, L, d_inner = x.shape
        A = -torch.exp(self.A_log.float())
        delta = F.softplus(self.dt_proj(x))

        # Optional extra gating mechanism
        if self.use_extra_gating:
            gate = torch.sigmoid(self.gate_proj(x))

        bc = self.x_proj(x)
        B_val, C_val = torch.split(bc, self.d_state, dim=-1)

        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y = torch.zeros(B, L, d_inner, device=x.device, dtype=x.dtype)

        for i in range(L):
            delta_i = delta[:, i, :].unsqueeze(-1)
            A_bar = torch.exp(delta_i * A)
            B_bar = delta_i * B_val[:, i, :].unsqueeze(1)

            if self.use_extra_gating:
                # Additional gating on state transition
                gate_i = gate[:, i, :].unsqueeze(-1)
                h = (gate_i * A_bar) * h + B_bar * x[:, i, :].unsqueeze(-1)
            else:
                # Standard Mamba SSM update (already has implicit gating via delta/A_bar)
                h = A_bar * h + B_bar * x[:, i, :].unsqueeze(-1)

            y[:, i, :] = torch.sum(h * C_val[:, i, :].unsqueeze(1), dim=-1)

        y = y + x * self.D
        return y


class MambaCore(Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        n_layers=4,
        d_state=16,
        d_conv=4,
        use_extra_gating=False,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.LayerNorm(d_model),
                        MambaMinimalBlock(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            use_extra_gating=use_extra_gating,
                        ),
                    ]
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.out_mu = nn.Linear(d_model, 1)
        self.out_sigma = nn.Linear(d_model, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Process sequence through Mamba blocks
        x = self.input_projection(x)
        x = self.dropout(x)

        for norm, block in self.layers:
            # Residual connection with in-place addition when possible
            x = x + block(norm(x))

        x = self.norm_f(x)
        # Output for all timesteps in sequence
        mu = self.out_mu(x)
        sigma = self.softplus(self.out_sigma(x))
        sigma = torch.clamp(sigma, min=1e-6)
        return mu, sigma


class MLPCore(Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[512, 256],  # Wide but shallow by default
        dropout_rate=0.2,
        use_batch_norm=True,
        activation='relu',
        preserve_temporal=False,  # New: whether to preserve temporal dimension
    ):
        super().__init__()
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        self.preserve_temporal = preserve_temporal
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build layers
        if preserve_temporal:
            # Use LayerNorm for temporal data (works on last dim, preserves seq structure)
            layers = []
            prev_dim = input_dim
            
            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_batch_norm:
                    # LayerNorm normalizes over the feature dimension
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(self.activation)
                if i < len(hidden_dims) - 1:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            
            self.backbone = nn.Sequential(*layers)
        else:
            # Original BatchNorm1d path for backwards compatibility
            layers = []
            prev_dim = input_dim
            
            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(self.activation)
                if i < len(hidden_dims) - 1:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            
            self.backbone = nn.Sequential(*layers)
        
        # Output heads for probabilistic forecasting
        self.out_mu = nn.Linear(prev_dim, 1)
        self.out_sigma = nn.Linear(prev_dim, 1)
        self.softplus = nn.Softplus()
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass for MLP.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            mu: Mean predictions, shape matches input: (batch_size, seq_len, 1) or (batch_size, 1)
            sigma: Standard deviation predictions, same shape as mu
        """
        original_shape = x.shape
        
        if self.preserve_temporal and x.dim() == 3:
            # Keep temporal structure intact
            # x: (batch_size, seq_len, input_dim)
            hidden = self.backbone(x)  # LayerNorm and Linear work on last dim
            
            # Generate outputs
            mu = self.out_mu(hidden)
            sigma = self.softplus(self.out_sigma(hidden))
            sigma = torch.clamp(sigma, min=1e-6)
            # Output: (batch_size, seq_len, 1)
            
        elif x.dim() == 3:
            # Original flattening path for backwards compatibility
            batch_size, seq_len, input_dim = x.shape
            # Reshape to (batch_size * seq_len, input_dim) to process all timesteps
            x = x.reshape(-1, input_dim)
            
            # Process through backbone - no change needed, BatchNorm1d works on flattened
            hidden = self.backbone(x)
            
            # Generate outputs
            mu = self.out_mu(hidden)
            sigma = self.softplus(self.out_sigma(hidden))
            sigma = torch.clamp(sigma, min=1e-6)
            
            # Reshape back to (batch_size, seq_len, 1)
            mu = mu.reshape(batch_size, seq_len, 1)
            sigma = sigma.reshape(batch_size, seq_len, 1)
        else:
            # Single timestep: (batch_size, input_dim)
            hidden = self.backbone(x)
            mu = self.out_mu(hidden)
            sigma = self.softplus(self.out_sigma(hidden))
            sigma = torch.clamp(sigma, min=1e-6)
        
        return mu, sigma


class SqueezeExcitation(Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)

    def forward(self, x):
        """
        Apply squeeze-and-excitation weighting.
        
        Args:
            x: Tensor of shape (batch, channels, time)
        """
        batch, channels, _ = x.shape
        weights = F.adaptive_avg_pool1d(x, 1).view(batch, channels)
        weights = torch.sigmoid(self.fc2(F.silu(self.fc1(weights)))).view(batch, channels, 1)
        return x * weights


class DepthwiseSeparableTemporal(Module):
    def __init__(
        self,
        in_channels,
        mid_channels=None,
        kernel_size=5,
        dilation=1,
        use_se=True,
        dropout=0.1,
        use_batch_norm=True,
    ):
        super().__init__()
        mid_channels = mid_channels or min(128, max(16, in_channels // 2))
        padding = ((kernel_size - 1) // 2) * dilation

        self.pointwise_in = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm1d(mid_channels) if use_batch_norm else nn.Identity()
        self.depthwise = nn.Conv1d(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=mid_channels,
            bias=False,
        )
        self.bn_depth = nn.BatchNorm1d(mid_channels) if use_batch_norm else nn.Identity()
        self.se = SqueezeExcitation(mid_channels) if use_se else nn.Identity()
        self.pointwise_out = nn.Conv1d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Residual depthwise-separable temporal convolution.
        
        Processes temporal patterns within each feature independently (depthwise),
        then mixes features (pointwise). This captures:
        - Within-feature temporal patterns (e.g., "weekday_sin" oscillating over forecast horizon)
        - Cross-feature interactions via pointwise convolutions
        - Global feature importance via squeeze-excitation
        
        Args:
            x: Tensor of shape (batch, time, features)
            
        Returns:
            Tensor of shape (batch, time, features) with residual connection
        """
        residual = x
        y = x.transpose(1, 2)  # (batch, features, time) for Conv1d
        y = F.silu(self.bn_in(self.pointwise_in(y)))
        y = F.silu(self.bn_depth(self.depthwise(y)))
        y = self.se(y)
        y = self.pointwise_out(y)
        y = self.dropout(y).transpose(1, 2)  # Back to (batch, time, features)
        return residual + y


class CNNMLPHead(Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout_rate=0.2,
        use_batch_norm=True,
        activation="relu",
        num_blocks=1,
        kernel_size=5,
        use_se=True,
        cnn_batch_norm=False,  # Separate control for CNN batch norm (can be unstable with small sequences)
    ):
        super().__init__()
        num_blocks = max(0, int(num_blocks)) if num_blocks is not None else 0

        blocks = []
        if num_blocks > 0:
            blocks.append(
                DepthwiseSeparableTemporal(
                    input_dim,
                    kernel_size=kernel_size,
                    dilation=1,
                    use_se=use_se,
                    dropout=dropout_rate,
                    use_batch_norm=cnn_batch_norm,
                )
            )
        if num_blocks > 1:
            blocks.append(
                DepthwiseSeparableTemporal(
                    input_dim,
                    kernel_size=kernel_size,
                    dilation=2,
                    use_se=use_se,
                    dropout=dropout_rate,
                    use_batch_norm=cnn_batch_norm,
                )
            )
        self.cnn = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.mlp = MLPCore(
            input_dim,
            hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation=activation,
            preserve_temporal=True,  # KEY: Don't flatten after CNN!
        )

    def forward(self, x):
        """
        Apply optional CNN front-end followed by MLP.
        
        Args:
            x: Tensor of shape (batch, time, features)
            
        Returns:
            mu, sigma: Predictions with shape (batch, time, 1)
        """
        x = self.cnn(x)
        return self.mlp(x)


class MambaSSM(ModelObject):
    def __init__(
        self,
        name: str = "MambaSSM",
        frequency: str = "infer",
        prediction_interval: float = 0.9,
        holiday_country: str = "US",
        random_seed: int = 2023,
        verbose: int = 1,
        context_length: int = 120,
        d_model: int = 32,
        n_layers: int = 2,
        d_state: int = 8,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        loss_function: str = "combined_nll_wasserstein",
        nll_weight: float = 1.0,
        wasserstein_weight: float = 0.1,
        prediction_batch_size: int = 60,
        datepart_method: str = "expanded",
        holiday_countries_used: bool = False,
        use_extra_gating: bool = False,
        use_naive_feature: bool = True,
        changepoint_method: str = "basic",
        changepoint_params: dict = None,
        regression_type: str = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            regression_type=regression_type,
        )
        self.regression_type = regression_type
        self.datepart_method = datepart_method
        self.holiday_countries_used = holiday_countries_used
        self.use_extra_gating = use_extra_gating
        normalized_method = changepoint_method if changepoint_method is not None else "none"
        if isinstance(normalized_method, str):
            normalized_method = normalized_method.lower()
        self.changepoint_method = normalized_method
        self.changepoint_params = changepoint_params if changepoint_params is not None else {}
        self.context_length = context_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.use_naive_feature = use_naive_feature

        # Loss function parameters
        self.loss_function = loss_function
        self.nll_weight = nll_weight
        self.wasserstein_weight = wasserstein_weight

        self.prediction_batch_size = prediction_batch_size
        self.model = None
        
        # Device selection: CUDA > MPS > CPU
        # Check CUDA first (highest priority for performance)
        if torch.cuda.is_available():
            self.device = "cuda"
        # Only check MPS on macOS to avoid false positives on Linux
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            import platform
            if platform.system() == "Darwin":  # macOS only
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

        try:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        except Exception:
            pass

        self.changepoint_features = None
        self.changepoint_features_columns = pd.Index([])
        self.naive_feature_columns = pd.Index([])
        self.naive_feature_values = None

    def _get_loss_function(self):
        """Create the appropriate loss function based on the loss_function parameter."""
        if self.loss_function == "combined_nll_wasserstein":
            return CombinedNLLWassersteinLoss(self.nll_weight, self.wasserstein_weight)
        elif self.loss_function == "quantile":
            return QuantileLoss(
                prediction_interval=self.prediction_interval
            )
        elif self.loss_function == "crps":
            return CRPSLoss()
        elif self.loss_function == "energy":
            return EnergyScore()
        elif self.loss_function == "regularized_nll":
            return RegularizedGaussianNLL()
        elif self.loss_function == "ranked_sharpe":
            return RankedSharpeLoss(
                nll_weight=getattr(self, 'nll_weight', 0.7),
                rank_weight=getattr(self, 'rank_weight', 0.3),
                temperature=getattr(self, 'temperature', 1.0),
                use_kendall=getattr(self, 'use_kendall', False)
            )
        elif self.loss_function == "short_horizon_rank":
            return ShortHorizonRankLoss(
                nll_weight=getattr(self, 'nll_weight', 0.6),
                rank_weight=getattr(self, 'rank_weight', 0.3),
                consistency_weight=getattr(self, 'consistency_weight', 0.1),
                temperature=getattr(self, 'temperature', 0.5),
                horizon_steps=getattr(self, 'horizon_steps', 4)
            )
        else:
            raise ValueError(
                f"Unknown loss function: {self.loss_function}. "
                "Options: 'combined_nll_wasserstein', 'quantile', 'crps', 'energy', 'regularized_nll', "
                "'ranked_sharpe', 'short_horizon_rank'"
            )

    def fit(self, df, future_regressor=None, **kwargs):
        """Train the Mamba SSM forecaster."""
        fit_start_time = datetime.datetime.now()

        # 1. Book-keeping
        df = self.basic_profile(df)  # saves col names
        self.df_train = df  # Store training data for predict method
        y_train = df.to_numpy(dtype=np.float32)
        train_index = df.index

        if self.regression_type is not None:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but not future_regressor supplied."
                )

        # 2. Date-part + changepoint features + optional regressors
        date_feats_train = date_part(
            train_index,
            method=self.datepart_method,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )
        
        use_changepoints = self.changepoint_method not in {None, "none"}

        # Set default changepoint parameters for basic method if not provided
        if (
            use_changepoints
            and self.changepoint_method == 'basic'
            and not self.changepoint_params
        ):
            half_yr_space = half_yr_spacing(df)
            self.changepoint_params = {
                'changepoint_spacing': int(half_yr_space),
                'changepoint_distance_end': int(half_yr_space / 2)
            }
        
        # Changepoint features
        if use_changepoints:
            # Default to individual per-series changepoints unless overridden in kwargs
            if 'aggregate_method' not in kwargs:
                kwargs['aggregate_method'] = 'individual'
            
            self.changepoint_detector = ChangepointDetector(
                method=self.changepoint_method,
                method_params=self.changepoint_params,
                **kwargs,
            )
            self.changepoint_detector.detect(df)
            changepoint_features = self.changepoint_detector.create_features(
                forecast_length=0
            )
        else:
            self.changepoint_detector = None
            changepoint_features = pd.DataFrame(index=df.index)

        self.changepoint_features = changepoint_features
        self.changepoint_features_columns = changepoint_features.columns

        # Single naive feature per series (will be populated with window values during batch creation)
        if self.use_naive_feature:
            # Create ONE naive feature column per series
            # During training batch creation, this will be filled with the appropriate window values
            naive_feature_data = {}
            for col in df.columns:
                feature_name = f"naive_last_{col}"
                # Initialize with shifted values (will be replaced during batch creation with proper windows)
                shifted_values = df[col].shift(1).bfill().fillna(0.0).astype(np.float32)
                naive_feature_data[feature_name] = shifted_values
            
            naive_features = pd.DataFrame(naive_feature_data, index=df.index, dtype=np.float32)
            self.naive_feature_columns = naive_features.columns
            # Store last window for each series (for prediction)
            self.naive_window_size = self.prediction_batch_size
            self.naive_feature_windows = {}
            for col in df.columns:
                # Get the last prediction_batch_size values
                last_window = df[col].ffill().iloc[-self.prediction_batch_size:].fillna(0.0).values.astype(np.float32)
                # Reverse so most recent is first: [t-1, t-2, ..., t-prediction_batch_size]
                self.naive_feature_windows[col] = last_window[::-1]
        else:
            naive_features = None
            self.naive_feature_columns = pd.Index([])
            self.naive_feature_windows = None
            self.naive_window_size = 0
        
        # Combine all features
        feature_list = [date_feats_train, changepoint_features]
        if naive_features is not None:
            feature_list.append(naive_features)
        if future_regressor is not None:
            feature_list.append(future_regressor)
            
        feat_df_train = pd.concat(feature_list, axis=1).reindex(train_index)

        feat_df_train = feat_df_train.ffill().bfill().astype(np.float32)
        feat_df_train.columns = [str(c) for c in feat_df_train.columns]
        self.features = feat_df_train  # <- *** stored for predict ***
        self.feature_columns = feat_df_train.columns

        # 3. Scaling
        self.scaler_means = np.mean(y_train, axis=0)
        self.scaler_stds = np.std(y_train, axis=0)
        self.scaler_stds[self.scaler_stds == 0.0] = 1.0

        y_train_scaled = (y_train - self.scaler_means) / self.scaler_stds

        self.feature_scaler = StandardScaler()
        train_feats_scaled = self.feature_scaler.fit_transform(feat_df_train.values)

        # 4. Build per-series feature mapping if needed
        series_names = list(df.columns)
        series_feat_mapping, has_per_series = build_series_feature_mapping(
            self.feature_columns, 
            series_names,
            self.changepoint_features_columns
        )
        self.series_feat_mapping = series_feat_mapping
        self.has_per_series_features = has_per_series
        
        # 4b. Create training batches using shared efficient function
        # Get naive feature indices for anti-leakage
        if self.use_naive_feature:
            naive_feature_indices = [
                i for i, col in enumerate(self.feature_columns)
                if col in self.naive_feature_columns
            ]
        else:
            naive_feature_indices = []
        
        X_data, Y_data = create_training_batches(
            y_train_scaled, 
            train_feats_scaled, 
            self.prediction_batch_size, 
            max_samples=50000,
            random_seed=self.random_seed,
            series_feat_mapping=series_feat_mapping,
            series_names=series_names if has_per_series else None,
            naive_feature_indices=naive_feature_indices,
            use_naive_window=self.use_naive_feature,  # Window-based naive features
            y_train_raw=y_train,
            feature_means=self.feature_scaler.mean_,
            feature_scales=self.feature_scaler.scale_,
        )

        # 5. Torch plumbing
        # Account for potentially different feature dimensions per series
        if has_per_series:
            # Use max features across all series
            num_features = X_data.shape[2]  # Features only (no y value in X anymore)
        else:
            num_features = train_feats_scaled.shape[1]
        input_dim = num_features
        self.model = MambaCore(
            input_dim,
            self.d_model,
            self.n_layers,
            self.d_state,
            use_extra_gating=self.use_extra_gating,
        ).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = self._get_loss_function()

        # Create tensors and dataset
        try:
            X_tensor = torch.tensor(X_data, dtype=torch.float32)
            Y_tensor = torch.tensor(Y_data, dtype=torch.float32)
        except (AttributeError, NameError):
            # Fallback when torch is not available
            X_tensor = X_data
            Y_tensor = Y_data
        
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device == "cuda"),
        )

        if self.verbose:
            print(f"Training on {self.device} • {len(dataset):,} samples/epoch")

        # 5. Training loop
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.model.train()
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            
            for x_batch, y_batch in tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                disable=(self.verbose == 0),
            ):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                mu, sigma = self.model(x_batch)
                
                # mu and sigma now have shape (batch_size, prediction_batch_size, 1)
                # y_batch has shape (batch_size, prediction_batch_size, 1)
                # Calculate loss across all timesteps in the batch
                loss = criterion(mu, sigma, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if self.verbose:
                print(
                    f"Epoch {epoch+1}  avg-loss: {running_loss / len(dataloader):.4f}"
                )

        self.fit_runtime = datetime.datetime.now() - fit_start_time
        return self

    # ----------------------------------------------------------------------

    def predict(self, forecast_length: int, future_regressor=None, **kwargs):
        """Batch prediction using future features."""
        predict_start_time = datetime.datetime.now()
        self.model.eval()

        # 1. Forecast index
        forecast_index = self.create_forecast_index(forecast_length)

        # 2. Future feature frame
        future_date_feats = date_part(
            forecast_index,
            method=self.datepart_method,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )
        
        # Create future changepoint features without re-fitting detector (single fit at start)
        if (
            getattr(self, "changepoint_detector", None) is not None
            and self.changepoint_detector is not None
        ):
            # Use the changepoint detector's create_features method for consistent feature generation
            # This properly continues "periods since changepoint" values into the future
            all_features = self.changepoint_detector.create_features(forecast_length=forecast_length)
            future_changepoint_feats = all_features.iloc[-forecast_length:].copy()
            future_changepoint_feats.index = forecast_index
            future_changepoint_feats = future_changepoint_feats.reindex(
                columns=self.changepoint_features_columns
            )
        else:
            future_changepoint_feats = pd.DataFrame(
                index=forecast_index,
                columns=getattr(self, "changepoint_features_columns", []),
            )

        # Get last observed window for naive features (before prediction starts)
        if self.use_naive_feature and self.naive_feature_windows is not None:
            # Start with last training window for each series
            current_naive_windows = {
                col: window.copy() for col, window in self.naive_feature_windows.items()
            }
        else:
            current_naive_windows = None
        
        num_series = self.df_train.shape[1]
        forecast_mu_scaled = np.zeros((forecast_length, num_series), dtype=np.float32)
        forecast_sigma_scaled = np.zeros((forecast_length, num_series), dtype=np.float32)

        # 3. Predict in batches of prediction_batch_size
        num_batches = int(np.ceil(forecast_length / self.prediction_batch_size))
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_step = batch_idx * self.prediction_batch_size
                end_step = min(start_step + self.prediction_batch_size, forecast_length)
                actual_batch_size = end_step - start_step
                
                if self.verbose and batch_idx % 5 == 0:
                    print(f"Predicting batch {batch_idx+1}/{num_batches} (steps {start_step+1}-{end_step})")
                
                # Build window-based naive features for this batch
                if current_naive_windows is not None:
                    # Create ONE naive feature per series
                    # For each series, the naive feature at each prediction timestep gets the corresponding window value
                    # If predicting steps t, t+1, ..., t+prediction_batch_size-1:
                    #   - At timestep t: use window value at position corresponding to t-1 (most recent)
                    #   - At timestep t+1: use window value at position corresponding to t (which is now in window)
                    #   - etc.
                    naive_feature_data = {}
                    for col in self.df_train.columns:
                        feature_name = f"naive_last_{col}"
                        window = current_naive_windows[col]  # Current window for this series
                        # The window contains [t-1, t-2, ..., t-prediction_batch_size] (REVERSED, most recent first)
                        # For prediction batch, we use these values in order
                        naive_feature_data[feature_name] = window[:actual_batch_size].tolist()
                    
                    naive_future_feats = pd.DataFrame(
                        naive_feature_data,
                        index=forecast_index[start_step:end_step],
                        dtype=np.float32,
                    )
                    naive_future_feats = naive_future_feats.reindex(columns=self.naive_feature_columns)
                else:
                    naive_future_feats = None
                
                # Combine all future features for this batch
                feature_list = [
                    future_date_feats.iloc[start_step:end_step],
                    future_changepoint_feats.iloc[start_step:end_step]
                ]
                if naive_future_feats is not None:
                    feature_list.append(naive_future_feats)
                if future_regressor is not None:
                    feature_list.append(future_regressor.iloc[start_step:end_step])
                    
                feat_batch_df = pd.concat(feature_list, axis=1)
                feat_batch_df = (
                    feat_batch_df.reindex(columns=self.feature_columns)
                    .ffill()
                    .bfill()
                    .fillna(0.0)
                    .astype(np.float32)
                )
                batch_feats_scaled = self.feature_scaler.transform(feat_batch_df.values)
                
                # Handle per-series vs shared features
                if getattr(self, 'has_per_series_features', False):
                    # Per-series features: process each series individually
                    max_feats_from_mapping = max(len(feat_indices) for feat_indices in self.series_feat_mapping.values())
                    
                    for series_idx in range(num_series):
                        # Get feature indices for this series
                        feat_indices = self.series_feat_mapping.get(series_idx, [])
                        
                        # Extract features for this series
                        series_feats = batch_feats_scaled[:, feat_indices]
                        
                        # Pad if needed
                        if len(feat_indices) < max_feats_from_mapping:
                            pad_size = max_feats_from_mapping - len(feat_indices)
                            padding = np.zeros((actual_batch_size, pad_size), dtype=np.float32)
                            series_feats = np.concatenate([series_feats, padding], axis=1)
                        
                        # Create input tensor: (1, actual_batch_size, n_features)
                        model_in = torch.tensor(
                            series_feats, 
                            device=self.device, 
                            dtype=torch.float32
                        ).unsqueeze(0)
                        
                        # Predict for this series
                        mu, sigma = self.model(model_in)
                        
                        # Store predictions: (actual_batch_size, 1)
                        # Take only actual_batch_size elements in case model outputs more
                        forecast_mu_scaled[start_step:end_step, series_idx] = mu[0, :actual_batch_size, 0].cpu().numpy()
                        forecast_sigma_scaled[start_step:end_step, series_idx] = sigma[0, :actual_batch_size, 0].cpu().numpy()
                else:
                    # Shared features: broadcast for all series
                    # batch_feats_scaled shape: (actual_batch_size, n_features)
                    # Expand to (num_series, actual_batch_size, n_features)
                    batch_feats_tensor = torch.tensor(
                        batch_feats_scaled,
                        device=self.device,
                        dtype=torch.float32
                    )
                    model_in = batch_feats_tensor.unsqueeze(0).expand(num_series, -1, -1)
                    
                    # Predict for all series at once
                    mu, sigma = self.model(model_in)
                    
                    # Store predictions: (num_series, actual_batch_size, 1)
                    # Take only actual_batch_size elements in case model outputs more
                    forecast_mu_scaled[start_step:end_step, :] = mu[:, :actual_batch_size, 0].T.cpu().numpy()
                    forecast_sigma_scaled[start_step:end_step, :] = sigma[:, :actual_batch_size, 0].T.cpu().numpy()
                
                # Update naive window with predictions from this batch for next batch
                if current_naive_windows is not None:
                    # Get all predictions from this batch (unscaled to match stored window)
                    batch_predictions_scaled = forecast_mu_scaled[start_step:end_step, :]
                    batch_predictions_unscaled = (
                        batch_predictions_scaled * self.scaler_stds + self.scaler_means
                    )
                    
                    # Update window for each series
                    for col_idx, col in enumerate(self.df_train.columns):
                        current_window = current_naive_windows[col]
                        new_values = batch_predictions_unscaled[:, col_idx]
                        
                        # Prepend new predictions in reverse order so most recent is first
                        updated_window = np.concatenate(
                            [new_values[::-1], current_window]
                        )[: self.naive_window_size]
                        current_naive_windows[col] = updated_window

        # 4. Un-scale & wrap
        mu_unscaled = forecast_mu_scaled * self.scaler_stds + self.scaler_means
        sigma_unscaled = forecast_sigma_scaled * self.scaler_stds

        z = norm.ppf(1 - (1 - self.prediction_interval) / 2)
        lower = mu_unscaled - z * sigma_unscaled
        upper = mu_unscaled + z * sigma_unscaled

        forecast_df = pd.DataFrame(
            mu_unscaled, index=forecast_index, columns=self.column_names
        )
        lower_forecast_df = pd.DataFrame(
            lower, index=forecast_index, columns=self.column_names
        )
        upper_forecast_df = pd.DataFrame(
            upper, index=forecast_index, columns=self.column_names
        )

        predict_runtime = datetime.datetime.now() - predict_start_time
        return PredictionObject(
            model_name=self.name,
            forecast_length=forecast_length,
            forecast_index=forecast_df.index,
            forecast_columns=forecast_df.columns,
            lower_forecast=lower_forecast_df,
            forecast=forecast_df,
            upper_forecast=upper_forecast_df,
            prediction_interval=self.prediction_interval,
            predict_runtime=predict_runtime,
            fit_runtime=self.fit_runtime,
            model_parameters=self.get_params(),
        )

    def get_params(self):
        return {
            "context_length": self.context_length,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_state": self.d_state,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "loss_function": self.loss_function,
            "nll_weight": self.nll_weight,
            "wasserstein_weight": self.wasserstein_weight,
            "prediction_batch_size": self.prediction_batch_size,
            "datepart_method": self.datepart_method,
            "holiday_countries_used": self.holiday_countries_used,
            "use_extra_gating": self.use_extra_gating,
            "use_naive_feature": self.use_naive_feature,
            "changepoint_method": self.changepoint_method,
            "changepoint_params": self.changepoint_params,
            "regression_type": self.regression_type,
        }

    @staticmethod
    def get_new_params(method: str = "random"):
        """
        Generate new random parameters for MambaSSM model.
        
        Parameters are weighted based on:
        - Accuracy: Higher weights for parameters likely to improve accuracy
        - Speed: Lower weights for parameters that are slower on larger datasets
        - Stability: Higher weights for more stable/reliable options
        
        Args:
            method: Method for parameter selection (currently only 'random' supported)
            
        Returns:
            dict: Dictionary of randomly selected parameters
        """
        import random
        
        # Context length options - longer contexts can be more accurate but slower
        context_lengths = [30, 60, 90, 120, 180, 240]
        context_weights = [0.1, 0.2, 0.3, 0.25, 0.1, 0.05]  # Prefer medium lengths
        
        # Model dimension - larger models more accurate but slower
        d_models = [16, 24, 32, 48, 64, 96]
        d_model_weights = [0.1, 0.15, 0.25, 0.25, 0.2, 0.05]  # Prefer 32-64 range
        
        # Number of layers - more layers can be more accurate but much slower
        n_layers_options = [1, 2, 3, 4, 6]
        n_layers_weights = [0.15, 0.35, 0.25, 0.2, 0.05]  # Prefer 2-3 layers
        
        # State dimension - affects model capacity
        d_states = [4, 8, 12, 16, 24, 32]
        d_state_weights = [0.1, 0.25, 0.2, 0.25, 0.15, 0.05]  # Prefer 8-16 range
        
        # Training epochs - more epochs more accurate but slower
        epochs_options = [5, 8, 10, 15, 20, 30]
        epochs_weights = [0.1, 0.2, 0.25, 0.25, 0.15, 0.05]  # Prefer 10-15 range
        
        # Batch size - affects training stability and speed
        batch_sizes = [16, 24, 32, 48, 64, 96]
        batch_weights = [0.15, 0.2, 0.25, 0.2, 0.15, 0.05]  # Prefer 24-48 range
        
        # Learning rate - critical for convergence
        learning_rates = [5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
        lr_weights = [0.2, 0.35, 0.25, 0.15, 0.05]  # Prefer lower learning rates
        
        # Loss functions - weight by expected accuracy and stability
        loss_functions = [
            "combined_nll_wasserstein", 
            "crps", 
            "quantile", 
            "regularized_nll", 
            "energy",
            "short_horizon_rank",      # Best for short horizon ranked Sharpe
            "ranked_sharpe",           # General ranked Sharpe optimization
        ]
        loss_weights = [0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.03]
        
        # NLL weight for combined loss
        nll_weights = [0.5, 0.8, 1.0, 1.2, 1.5]
        nll_weight_probs = [0.1, 0.2, 0.4, 0.2, 0.1]  # Prefer around 1.0
        
        # Wasserstein weight for combined loss
        wasserstein_weights = [0.05, 0.1, 0.15, 0.2, 0.3]
        wasserstein_weight_probs = [0.15, 0.3, 0.25, 0.2, 0.1]  # Prefer lower values
        
        # Datepart method - use existing random_datepart function for proper weighting
        # This leverages the sophisticated weighting already built into random_datepart
        
        # Boolean options with appropriate weights
        holiday_used_options = [True, False]
        holiday_weights = [0.3, 0.7]  # Prefer False for simplicity unless needed
        
        extra_gating_options = [True, False]  
        gating_weights = [0.25, 0.75]  # Prefer False for stability
        
        naive_feature_options = [True, False]
        naive_feature_weights = [0.5, 0.5]  # Default to using naive features

        # Generate changepoint method and parameters using dedicated function
        changepoint_method, changepoint_params = generate_random_changepoint_params()

        # Regression type logic
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices([None, 'User'], [0.7, 0.3])[0]

        # Generate random selections
        selected_params = {
            "context_length": random.choices(context_lengths, weights=context_weights, k=1)[0],
            "d_model": random.choices(d_models, weights=d_model_weights, k=1)[0],
            "n_layers": random.choices(n_layers_options, weights=n_layers_weights, k=1)[0],
            "d_state": random.choices(d_states, weights=d_state_weights, k=1)[0],
            "epochs": random.choices(epochs_options, weights=epochs_weights, k=1)[0],
            "batch_size": random.choices(batch_sizes, weights=batch_weights, k=1)[0],
            "lr": random.choices(learning_rates, weights=lr_weights, k=1)[0],
            "loss_function": random.choices(loss_functions, weights=loss_weights, k=1)[0],
            "nll_weight": random.choices(nll_weights, weights=nll_weight_probs, k=1)[0],
            "wasserstein_weight": random.choices(wasserstein_weights, weights=wasserstein_weight_probs, k=1)[0],
            "datepart_method": random_datepart(),
            "holiday_countries_used": random.choices(holiday_used_options, weights=holiday_weights, k=1)[0],
            "use_extra_gating": random.choices(extra_gating_options, weights=gating_weights, k=1)[0],
            "use_naive_feature": random.choices(naive_feature_options, weights=naive_feature_weights, k=1)[0],
            "changepoint_method": changepoint_method,
            "changepoint_params": changepoint_params,
            "regression_type": regression_choice,
        }
        
        # Add prediction_batch_size based on context_length (longer contexts need smaller batches)
        if selected_params["context_length"] >= 180:
            selected_params["prediction_batch_size"] = random.choices([30, 45, 60], weights=[0.4, 0.4, 0.2], k=1)[0]
        else:
            selected_params["prediction_batch_size"] = random.choices([45, 60, 90], weights=[0.3, 0.4, 0.3], k=1)[0]
        
        return selected_params


class pMLP(ModelObject):
    """
    Probabilistic MLP for time series forecasting.
    
    Uses a wide, shallow MLP architecture with optional CNN front-end for
    capturing temporal patterns in future seasonal features.
    
    Args:
        name: Model name
        frequency: Time series frequency
        prediction_interval: Prediction interval (e.g., 0.9 for 90% interval)
        holiday_country: Country code for holiday features
        random_seed: Random seed for reproducibility
        verbose: Verbosity level (0=silent, 1=progress)
        hidden_dims: List of hidden layer dimensions (default: [768, 256])
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        activation: Activation function ('relu', 'gelu', 'silu')
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        loss_function: Loss function to use
        nll_weight: Weight for NLL component in combined loss
        wasserstein_weight: Weight for Wasserstein component in combined loss
        prediction_batch_size: Number of timesteps to predict per batch
        datepart_method: Method for date feature extraction
        holiday_countries_used: Whether to use holiday features
        use_naive_feature: Whether to use naive/last-value features
        changepoint_method: Method for changepoint detection
        changepoint_params: Parameters for changepoint detection
        regression_type: Type of regression (None or 'User')
        num_cnn_blocks: Number of CNN blocks (0=no CNN, 1-2=CNN layers)
        cnn_params: Dictionary of CNN parameters:
            - kernel_size (int): Temporal kernel size (3, 5, 7, 9)
            - use_se (bool): Use squeeze-and-excitation for feature importance
    """
    # TODO: improve the changepoint features / future trend modeling for MLP
    # TODO: add a holiday detector integration
    def __init__(
        self,
        name: str = "pMLP",  # Probabilistic MLP
        frequency: str = "infer",
        prediction_interval: float = 0.9,
        holiday_country: str = "US",
        random_seed: int = 2023,
        verbose: int = 1,
        hidden_dims: list = None,  # Will default to wide, shallow architecture
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        epochs: int = 15,  # Slightly more epochs since MLPs train faster
        batch_size: int = 64,  # Larger batch size for efficiency
        lr: float = 2e-3,  # Slightly higher LR for MLPs
        loss_function: str = "combined_nll_wasserstein",
        nll_weight: float = 1.0,
        wasserstein_weight: float = 0.1,
        prediction_batch_size: int = 100,  # Larger for efficiency
        datepart_method: str = "expanded",
        holiday_countries_used: bool = False,
        use_naive_feature: bool = True,
        changepoint_method: str = "basic",
        changepoint_params: dict = None,
        regression_type: str = None,
        num_cnn_blocks: int | None = 0,
        cnn_params: dict = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            regression_type=regression_type,
        )
        self.regression_type = regression_type
        self.datepart_method = datepart_method
        self.holiday_countries_used = holiday_countries_used
        normalized_method = changepoint_method if changepoint_method is not None else "none"
        if isinstance(normalized_method, str):
            normalized_method = normalized_method.lower()
        self.changepoint_method = normalized_method
        self.changepoint_params = changepoint_params if changepoint_params is not None else {}
        
        # Default to wide, shallow architecture if not specified
        if hidden_dims is None:
            hidden_dims = [768, 256]  # Wide first layer, narrower second
        self.hidden_dims = hidden_dims
        
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.use_naive_feature = use_naive_feature
        self.num_cnn_blocks = None if num_cnn_blocks is None else max(0, int(num_cnn_blocks))
        
        # CNN parameters: kernel_size, use_se, etc.
        self.cnn_params = cnn_params if cnn_params is not None else {}

        # Loss function parameters
        self.loss_function = loss_function
        self.nll_weight = nll_weight
        self.wasserstein_weight = wasserstein_weight

        self.prediction_batch_size = prediction_batch_size
        self.model = None

        self.changepoint_features = None

    def _get_loss_function(self):
        """Create the appropriate loss function based on the loss_function parameter."""
        if self.loss_function == "combined_nll_wasserstein":
            return CombinedNLLWassersteinLoss(self.nll_weight, self.wasserstein_weight)
        elif self.loss_function == "quantile":
            return QuantileLoss(
                prediction_interval=self.prediction_interval
            )
        elif self.loss_function == "crps":
            return CRPSLoss()
        elif self.loss_function == "energy":
            return EnergyScore()
        elif self.loss_function == "regularized_nll":
            return RegularizedGaussianNLL()
        elif self.loss_function == "ranked_sharpe":
            return RankedSharpeLoss(
                nll_weight=getattr(self, 'nll_weight', 0.7),
                rank_weight=getattr(self, 'rank_weight', 0.3),
                temperature=getattr(self, 'temperature', 1.0),
                use_kendall=getattr(self, 'use_kendall', False)
            )
        elif self.loss_function == "short_horizon_rank":
            return ShortHorizonRankLoss(
                nll_weight=getattr(self, 'nll_weight', 0.6),
                rank_weight=getattr(self, 'rank_weight', 0.3),
                consistency_weight=getattr(self, 'consistency_weight', 0.1),
                temperature=getattr(self, 'temperature', 0.5),
                horizon_steps=getattr(self, 'horizon_steps', 4)
            )
        else:
            raise ValueError(
                f"Unknown loss function: {self.loss_function}. "
                "Options: 'combined_nll_wasserstein', 'quantile', 'crps', 'energy', 'regularized_nll', "
                "'ranked_sharpe', 'short_horizon_rank'"
            )

    def fit(self, df, future_regressor=None, **kwargs):
        """Train the pMLP forecaster."""
        fit_start_time = datetime.datetime.now()

        # 1. Book-keeping
        df = self.basic_profile(df)  # saves col names
        self.df_train = df  # Store training data for predict method
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        y_train = df.to_numpy(dtype=np.float32)
        train_index = df.index

        if self.regression_type is not None:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but not future_regressor supplied."
                )

        # 2. Date-part + changepoint features + optional regressors
        date_feats_train = date_part(
            train_index,
            method=self.datepart_method,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )
        
        use_changepoints = self.changepoint_method not in {None, "none"}

        # Set default changepoint parameters for basic method if not provided
        if (
            use_changepoints
            and self.changepoint_method == 'basic'
            and not self.changepoint_params
        ):
            half_yr_space = half_yr_spacing(df)
            self.changepoint_params = {
                'changepoint_spacing': int(half_yr_space),
                'changepoint_distance_end': int(half_yr_space / 2)
            }
        
        # Changepoint features
        if use_changepoints:
            # Default to individual per-series changepoints unless overridden in kwargs
            if 'aggregate_method' not in kwargs:
                kwargs['aggregate_method'] = 'individual'
            
            self.changepoint_detector = ChangepointDetector(
                method=self.changepoint_method,
                method_params=self.changepoint_params,
                **kwargs,
            )
            self.changepoint_detector.detect(df)
            changepoint_features = self.changepoint_detector.create_features(
                forecast_length=0
            )
        else:
            self.changepoint_detector = None
            changepoint_features = pd.DataFrame(index=df.index)

        self.changepoint_features = changepoint_features
        self.changepoint_features_columns = changepoint_features.columns

        # Single naive feature per series (will be populated with window values during batch creation) - pMLP version
        if self.use_naive_feature:
            # Create ONE naive feature column per series
            # During training batch creation, this will be filled with the appropriate window values
            naive_feature_data = {}
            for col in df.columns:
                feature_name = f"naive_last_{col}"
                # Initialize with shifted values (will be replaced during batch creation with proper windows)
                shifted_values = df[col].shift(1).bfill().fillna(0.0).astype(np.float32)
                naive_feature_data[feature_name] = shifted_values
            
            naive_features = pd.DataFrame(naive_feature_data, index=df.index, dtype=np.float32)
            self.naive_feature_columns = naive_features.columns
            # Store last window for each series (for prediction)
            self.naive_window_size = self.prediction_batch_size
            self.naive_feature_windows = {}
            for col in df.columns:
                # Get the last prediction_batch_size values
                last_window = df[col].ffill().iloc[-self.prediction_batch_size:].fillna(0.0).values.astype(np.float32)
                # Reverse so most recent is first: [t-1, t-2, ..., t-prediction_batch_size]
                self.naive_feature_windows[col] = last_window[::-1]
        else:
            naive_features = None
            self.naive_feature_columns = pd.Index([])
            self.naive_feature_windows = None
            self.naive_window_size = 0
        
        # Combine all features
        feature_list = [date_feats_train, changepoint_features]
        if naive_features is not None:
            feature_list.append(naive_features)
        if future_regressor is not None:
            feature_list.append(future_regressor)
            
        feat_df_train = pd.concat(feature_list, axis=1).reindex(train_index)

        feat_df_train = feat_df_train.ffill().bfill().astype(np.float32)
        feat_df_train.columns = [str(c) for c in feat_df_train.columns]
        self.features = feat_df_train  # <- *** stored for predict ***
        self.feature_columns = feat_df_train.columns

        # 3. Scaling
        self.scaler_means = np.mean(y_train, axis=0)
        self.scaler_stds = np.std(y_train, axis=0)
        self.scaler_stds[self.scaler_stds == 0.0] = 1.0

        y_train_scaled = (y_train - self.scaler_means) / self.scaler_stds

        self.feature_scaler = StandardScaler()
        train_feats_scaled = self.feature_scaler.fit_transform(feat_df_train.values)

        # 4. Build per-series feature mapping if needed
        series_names = list(df.columns)
        series_feat_mapping, has_per_series = build_series_feature_mapping(
            self.feature_columns, 
            series_names,
            self.changepoint_features_columns
        )
        self.series_feat_mapping = series_feat_mapping
        self.has_per_series_features = has_per_series
        
        # 4b. Create training batches using shared efficient function
        # Get naive feature indices for anti-leakage
        if self.use_naive_feature:
            naive_feature_indices = [
                i for i, col in enumerate(self.feature_columns)
                if col in self.naive_feature_columns
            ]
        else:
            naive_feature_indices = []
        
        self.X_data, self.Y_data = create_training_batches(
            y_train_scaled, 
            train_feats_scaled, 
            self.prediction_batch_size, 
            max_samples=100000,  # More samples for MLP efficiency
            random_seed=self.random_seed,
            series_feat_mapping=series_feat_mapping,
            series_names=series_names if has_per_series else None,
            naive_feature_indices=naive_feature_indices,
            use_naive_window=self.use_naive_feature,  # Window-based naive features
            y_train_raw=y_train,
            feature_means=self.feature_scaler.mean_,
            feature_scales=self.feature_scaler.scale_,
        )

        # 5. Torch setup - optimized for MLP
        # # Device selection: CUDA > MPS > CPU
        # Check CUDA first (highest priority for performance)
        if torch.cuda.is_available():
            self.device = "cuda"
        # Only check MPS on macOS to avoid false positives on Linux
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            import platform
            if platform.system() == "Darwin":  # macOS only
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

        # Account for potentially different feature dimensions per series
        if has_per_series:
            # Use max features across all series
            num_features = self.X_data.shape[2]  # Features only (no y value in X anymore)
        else:
            num_features = train_feats_scaled.shape[1]
        input_dim = num_features
        if self.num_cnn_blocks and self.num_cnn_blocks > 0:
            self.model = CNNMLPHead(
                input_dim,
                hidden_dims=self.hidden_dims,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
                activation=self.activation,
                num_blocks=self.num_cnn_blocks,
                **self.cnn_params  # Pass through CNN parameters
            ).to(self.device)
        else:
            self.model = MLPCore(
                input_dim,  # Input per timestep
                self.hidden_dims,
                self.dropout_rate,
                self.use_batch_norm,
                self.activation,
            ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = self._get_loss_function()

        # Create tensors and dataset - keep sequence structure for MLP
        try:
            X_tensor = torch.tensor(self.X_data, dtype=torch.float32)
            Y_tensor = torch.tensor(self.Y_data, dtype=torch.float32)
        except (AttributeError, NameError):
            # Fallback when torch is not available
            X_tensor = self.X_data
            Y_tensor = self.Y_data

        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device == "cuda"),
        )

        if self.verbose:
            print(f"Training pMLP on {self.device} • {len(dataset):,} samples/epoch")

        # 6. Training loop - optimized for efficiency
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.model.train()
        
        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            num_batches = 0
            
            for x_batch, y_batch in tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                disable=(self.verbose == 0),
            ):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                mu, sigma = self.model(x_batch)
                
                # mu and sigma now have shape (batch_size, prediction_batch_size, 1)
                # y_batch has shape (batch_size, prediction_batch_size, 1)
                # Calculate loss across all timesteps in the batch
                loss = criterion(mu, sigma, y_batch)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
            
            avg_loss = running_loss / num_batches
            scheduler.step(avg_loss)
            
            if self.verbose:
                print(f"Epoch {epoch+1}  avg-loss: {avg_loss:.4f}")

        self.fit_runtime = datetime.datetime.now() - fit_start_time
        return self

    def predict(self, forecast_length: int, future_regressor=None, **kwargs):
        """Batch prediction using future features for pMLP."""
        predict_start_time = datetime.datetime.now()
        self.model.eval()

        # 1. Forecast index
        forecast_index = self.create_forecast_index(forecast_length)

        # 2. Future feature frame
        future_date_feats = date_part(
            forecast_index,
            method=self.datepart_method,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )
        
        # Create future changepoint features without re-fitting detector (single fit at start)
        if (
            getattr(self, "changepoint_detector", None) is not None
            and self.changepoint_detector is not None
        ):
            # Use the changepoint detector's create_features method for consistent feature generation
            # This properly continues "periods since changepoint" values into the future
            all_features = self.changepoint_detector.create_features(forecast_length=forecast_length)
            future_changepoint_feats = all_features.iloc[-forecast_length:].copy()
            future_changepoint_feats.index = forecast_index
            future_changepoint_feats = future_changepoint_feats.reindex(
                columns=self.changepoint_features_columns
            )
        else:
            future_changepoint_feats = pd.DataFrame(
                index=forecast_index,
                columns=getattr(self, "changepoint_features_columns", []),
            )

        # Get last observed window for naive features (before prediction starts) - pMLP version
        if self.use_naive_feature and self.naive_feature_windows is not None:
            # Start with last training window for each series
            current_naive_windows = {
                col: window.copy() for col, window in self.naive_feature_windows.items()
            }
        else:
            current_naive_windows = None
        
        num_series = self.df_train.shape[1]
        forecast_mu_scaled = np.zeros((forecast_length, num_series), dtype=np.float32)
        forecast_sigma_scaled = np.zeros((forecast_length, num_series), dtype=np.float32)

        # 3. Predict in batches of prediction_batch_size
        num_batches = int(np.ceil(forecast_length / self.prediction_batch_size))
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_step = batch_idx * self.prediction_batch_size
                end_step = min(start_step + self.prediction_batch_size, forecast_length)
                actual_batch_size = end_step - start_step
                
                if self.verbose and batch_idx % 5 == 0:
                    print(f"Predicting batch {batch_idx+1}/{num_batches} (steps {start_step+1}-{end_step})")
                
                # Build window-based naive features for this batch - pMLP version
                if current_naive_windows is not None:
                    # Create ONE naive feature per series
                    # For each series, the naive feature at each prediction timestep gets the corresponding window value
                    # If predicting steps t, t+1, ..., t+prediction_batch_size-1:
                    #   - At timestep t: use window value at position 0 (t-1, most recent)
                    #   - At timestep t+1: use window value at position 1 (t-2)
                    #   - etc.
                    naive_feature_data = {}
                    for col in self.df_train.columns:
                        feature_name = f"naive_last_{col}"
                        window = current_naive_windows[col]  # Current window for this series
                        # The window contains [t-1, t-2, ..., t-prediction_batch_size] (REVERSED, most recent first)
                        # For prediction batch, we use these values in order
                        naive_feature_data[feature_name] = window[:actual_batch_size].tolist()
                    
                    naive_future_feats = pd.DataFrame(
                        naive_feature_data,
                        index=forecast_index[start_step:end_step],
                        dtype=np.float32,
                    )
                    naive_future_feats = naive_future_feats.reindex(columns=self.naive_feature_columns)
                else:
                    naive_future_feats = None
                
                # Combine all future features for this batch
                feature_list = [
                    future_date_feats.iloc[start_step:end_step],
                    future_changepoint_feats.iloc[start_step:end_step]
                ]
                if naive_future_feats is not None:
                    feature_list.append(naive_future_feats)
                if future_regressor is not None:
                    feature_list.append(future_regressor.iloc[start_step:end_step])
                    
                feat_batch_df = pd.concat(feature_list, axis=1)
                feat_batch_df = (
                    feat_batch_df.reindex(columns=self.feature_columns)
                    .ffill()
                    .bfill()
                    .fillna(0.0)
                    .astype(np.float32)
                )
                batch_feats_scaled = self.feature_scaler.transform(feat_batch_df.values)
                
                # Handle per-series vs shared features
                if getattr(self, 'has_per_series_features', False):
                    # Per-series features: process each series individually
                    max_feats_from_mapping = max(len(feat_indices) for feat_indices in self.series_feat_mapping.values())
                    
                    for series_idx in range(num_series):
                        # Get feature indices for this series
                        feat_indices = self.series_feat_mapping.get(series_idx, [])
                        
                        # Extract features for this series
                        series_feats = batch_feats_scaled[:, feat_indices]
                        
                        # Pad if needed
                        if len(feat_indices) < max_feats_from_mapping:
                            pad_size = max_feats_from_mapping - len(feat_indices)
                            padding = np.zeros((actual_batch_size, pad_size), dtype=np.float32)
                            series_feats = np.concatenate([series_feats, padding], axis=1)
                        
                        # Create input tensor: (1, actual_batch_size, n_features)
                        model_in = torch.tensor(
                            series_feats, 
                            device=self.device, 
                            dtype=torch.float32
                        ).unsqueeze(0)
                        
                        # Predict for this series
                        mu, sigma = self.model(model_in)
                        
                        # Store predictions: (actual_batch_size, 1)
                        # Take only actual_batch_size elements in case model outputs more
                        forecast_mu_scaled[start_step:end_step, series_idx] = mu[0, :actual_batch_size, 0].cpu().numpy()
                        forecast_sigma_scaled[start_step:end_step, series_idx] = sigma[0, :actual_batch_size, 0].cpu().numpy()
                else:
                    # Shared features: broadcast for all series
                    # batch_feats_scaled shape: (actual_batch_size, n_features)
                    # Expand to (num_series, actual_batch_size, n_features)
                    batch_feats_tensor = torch.tensor(
                        batch_feats_scaled,
                        device=self.device,
                        dtype=torch.float32
                    )
                    model_in = batch_feats_tensor.unsqueeze(0).expand(num_series, -1, -1)
                    
                    # Predict for all series at once
                    mu, sigma = self.model(model_in)
                    
                    # Store predictions: (num_series, actual_batch_size, 1)
                    # Take only actual_batch_size elements in case model outputs more
                    forecast_mu_scaled[start_step:end_step, :] = mu[:, :actual_batch_size, 0].T.cpu().numpy()
                    forecast_sigma_scaled[start_step:end_step, :] = sigma[:, :actual_batch_size, 0].T.cpu().numpy()
                
                # Update naive window with predictions from this batch for next batch - pMLP version
                if current_naive_windows is not None:
                    # Get all predictions from this batch (unscaled to match stored window)
                    batch_predictions_scaled = forecast_mu_scaled[start_step:end_step, :]
                    batch_predictions_unscaled = (
                        batch_predictions_scaled * self.scaler_stds + self.scaler_means
                    )
                    
                    # Update window for each series
                    for col_idx, col in enumerate(self.df_train.columns):
                        current_window = current_naive_windows[col]
                        new_values = batch_predictions_unscaled[:, col_idx]
                        
                        # Prepend new predictions in reverse order so most recent is first
                        updated_window = np.concatenate(
                            [new_values[::-1], current_window]
                        )[: self.naive_window_size]
                        current_naive_windows[col] = updated_window

        # 4. Un-scale & wrap
        mu_unscaled = forecast_mu_scaled * self.scaler_stds + self.scaler_means
        sigma_unscaled = forecast_sigma_scaled * self.scaler_stds

        z = norm.ppf(1 - (1 - self.prediction_interval) / 2)
        lower = mu_unscaled - z * sigma_unscaled
        upper = mu_unscaled + z * sigma_unscaled

        forecast_df = pd.DataFrame(
            mu_unscaled, index=forecast_index, columns=self.column_names
        )
        lower_forecast_df = pd.DataFrame(
            lower, index=forecast_index, columns=self.column_names
        )
        upper_forecast_df = pd.DataFrame(
            upper, index=forecast_index, columns=self.column_names
        )

        predict_runtime = datetime.datetime.now() - predict_start_time
        return PredictionObject(
            model_name=self.name,
            forecast_length=forecast_length,
            forecast_index=forecast_df.index,
            forecast_columns=forecast_df.columns,
            lower_forecast=lower_forecast_df,
            forecast=forecast_df,
            upper_forecast=upper_forecast_df,
            prediction_interval=self.prediction_interval,
            predict_runtime=predict_runtime,
            fit_runtime=self.fit_runtime,
            model_parameters=self.get_params(),
        )

    def get_params(self):
        return {
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "loss_function": self.loss_function,
            "nll_weight": self.nll_weight,
            "wasserstein_weight": self.wasserstein_weight,
            "prediction_batch_size": self.prediction_batch_size,
            "num_cnn_blocks": self.num_cnn_blocks,
            "cnn_params": self.cnn_params,
            "datepart_method": self.datepart_method,
            "holiday_countries_used": self.holiday_countries_used,
            "use_naive_feature": self.use_naive_feature,
            "changepoint_method": self.changepoint_method,
            "changepoint_params": self.changepoint_params,
            "regression_type": self.regression_type,
        }

    @staticmethod
    def get_new_params(method: str = "random"):
        """
        Generate new random parameters for pMLP model.
        
        Focuses on wide, shallow architectures and efficient training.
        
        Args:
            method: Method for parameter selection (currently only 'random' supported)
            
        Returns:
            dict: Dictionary of randomly selected parameters
        """
        import random

        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices([None, 'User'], [0.7, 0.3])[0]

        # Hidden dimensions - focus on wide, shallow architectures
        hidden_dim_options = [
            [512],              # Single wide layer
            [768],              # Single very wide layer  
            [1024],             # Single extremely wide layer
            [2056],             # Single ultra-wide layer
            [512, 256],         # Wide -> Medium
            [768, 256],         # Very wide -> Medium
            [1024, 256],        # Extremely wide -> Medium
            [512, 256, 128],    # Wide -> Medium -> Narrow (3 layers max)
            [512, 256, 512],    # Wide -> Medium -> Wide (3 layers max)
            [768, 384],         # Very wide -> Medium-wide
            [1024, 512],        # Extremely wide -> Wide
            [256, 128],         # Medium -> Narrow (for smaller datasets)
            [768, 128],         # Very wide -> Narrow
        ]
        hidden_weights = [0.15, 0.2, 0.15, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02, 0.01]
        
        # Dropout rates - moderate values for regularization
        dropout_rates = [0.1, 0.15, 0.2, 0.25, 0.3]
        dropout_weights = [0.15, 0.25, 0.3, 0.25, 0.05]  # Prefer 0.15-0.25 range
        
        # Batch normalization - usually helps with wide networks
        batch_norm_options = [True, False]
        batch_norm_weights = [0.8, 0.2]  # Strongly prefer batch norm
        
        # Activation functions
        activations = ['relu', 'gelu', 'silu']
        activation_weights = [0.5, 0.3, 0.2]  # ReLU still dominant, but GELU/SiLU competitive
        
        # Training epochs - MLPs train faster, can use more epochs
        epochs_options = [10, 15, 20, 25, 30]
        epochs_weights = [0.15, 0.3, 0.3, 0.2, 0.05]  # Prefer 15-20 range
        
        # Batch size - larger for efficiency with MLPs
        batch_sizes = [32, 48, 64, 96, 128]
        batch_weights = [0.1, 0.2, 0.35, 0.25, 0.1]  # Prefer 64-96 range
        
        # Learning rate - MLPs can handle slightly higher learning rates
        learning_rates = [1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3]
        lr_weights = [0.2, 0.25, 0.3, 0.2, 0.05]  # Prefer 1.5e-3 to 2e-3
        
        # Loss functions - weight by expected performance
        loss_functions = [
            "combined_nll_wasserstein", 
            "crps", 
            "quantile", 
            "regularized_nll", 
            "energy",
            "short_horizon_rank",      # Best for short horizon ranked Sharpe with MLPs
            "ranked_sharpe",           # General ranked Sharpe optimization
        ]
        loss_weights = [0.3, 0.25, 0.25, 0.15, 0.1, 0.1, 0.1]
        
        # NLL weight for combined loss
        nll_weights = [0.5, 0.8, 1.0, 1.2, 1.5]
        nll_weight_probs = [0.1, 0.2, 0.4, 0.2, 0.1]
        
        # Wasserstein weight for combined loss
        wasserstein_weights = [0.05, 0.1, 0.15, 0.2, 0.3]
        wasserstein_weight_probs = [0.15, 0.3, 0.25, 0.2, 0.1]

        # CNN blocks - allow optional temporal convolutions
        cnn_block_options = [0, 1, 2]
        cnn_block_weights = [0.55, 0.3, 0.15]
        
        # CNN parameters - only matters when num_cnn_blocks > 0
        # Kernel size options - affects temporal receptive field
        cnn_kernel_sizes = [3, 5, 7, 9]
        cnn_kernel_weights = [0.2, 0.4, 0.3, 0.1]  # Prefer 5-7 for daily data
        
        # Squeeze-and-excitation - feature importance weighting
        cnn_use_se_options = [True, False]
        cnn_use_se_weights = [0.7, 0.3]  # Usually helps
        
        # Datepart method
        datepart_method = random_datepart()
        
        # Boolean options
        holiday_used_options = [True, False]
        holiday_weights = [0.3, 0.7]
        
        naive_feature_options = [True, False]
        naive_feature_weights = [0.75, 0.25]
        
        # Generate changepoint method and parameters
        changepoint_method, changepoint_params = generate_random_changepoint_params()

        # Generate CNN parameters dict
        selected_num_cnn_blocks = random.choices(cnn_block_options, weights=cnn_block_weights, k=1)[0]
        
        # Only generate detailed CNN params if we're using CNN blocks
        if selected_num_cnn_blocks > 0:
            cnn_params_dict = {
                "kernel_size": random.choices(cnn_kernel_sizes, weights=cnn_kernel_weights, k=1)[0],
                "use_se": random.choices(cnn_use_se_options, weights=cnn_use_se_weights, k=1)[0],
            }
        else:
            # No CNN, so params don't matter but include defaults for consistency
            cnn_params_dict = {}

        # Generate random selections
        selected_params = {
            "hidden_dims": random.choices(hidden_dim_options, weights=hidden_weights, k=1)[0],
            "dropout_rate": random.choices(dropout_rates, weights=dropout_weights, k=1)[0],
            "use_batch_norm": random.choices(batch_norm_options, weights=batch_norm_weights, k=1)[0],
            "activation": random.choices(activations, weights=activation_weights, k=1)[0],
            "epochs": random.choices(epochs_options, weights=epochs_weights, k=1)[0],
            "batch_size": random.choices(batch_sizes, weights=batch_weights, k=1)[0],
            "lr": random.choices(learning_rates, weights=lr_weights, k=1)[0],
            "loss_function": random.choices(loss_functions, weights=loss_weights, k=1)[0],
            "nll_weight": random.choices(nll_weights, weights=nll_weight_probs, k=1)[0],
            "wasserstein_weight": random.choices(wasserstein_weights, weights=wasserstein_weight_probs, k=1)[0],
            "num_cnn_blocks": selected_num_cnn_blocks,
            "cnn_params": cnn_params_dict,
            "datepart_method": datepart_method,
            "holiday_countries_used": random.choices(holiday_used_options, weights=holiday_weights, k=1)[0],
            "use_naive_feature": random.choices(naive_feature_options, weights=naive_feature_weights, k=1)[0],
            "changepoint_method": changepoint_method,
            "changepoint_params": changepoint_params,
            "regression_type": regression_choice,
        }
        
        # Add prediction_batch_size - larger for MLP efficiency
        prediction_batch_options = [60, 100, 150, 200]
        prediction_batch_weights = [0.2, 0.4, 0.3, 0.1]
        selected_params["prediction_batch_size"] = random.choices(
            prediction_batch_options, weights=prediction_batch_weights, k=1
        )[0]
        
        return selected_params


if False:
    from autots import load_daily

    df_train = load_daily(long=False).ffill().bfill()
    # Instantiate and run the model
    mamba_model = MambaSSM(
        context_length=90,
        epochs=10,  # Reduced epochs for demo
        batch_size=32,
        verbose=1,
    )

    # Fit the model
    mamba_model.fit(df_train)

    # Make a forecast
    forecast_horizon = 90
    prediction = mamba_model.predict(forecast_length=forecast_horizon)

    # Inspect the results
    print("\n--- Forecast Results ---")
    print(f"Loss function used: {mamba_model.loss_function}")
    print(f"Fit runtime: {prediction.fit_runtime}")
    print(f"Predict runtime: {prediction.predict_runtime}")

    print("\nPoint Forecast:")
    print(prediction.forecast.iloc[:5, :5])
    pd.concat(
        [
            df_train["SP500"].rename("actual"),
            prediction.forecast["SP500"].rename("forecast"),
        ],
        axis=1,
    ).plot()
