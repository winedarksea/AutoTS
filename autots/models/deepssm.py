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


# Custom Loss Functions Only - Dataset creation now uses window_maker


def build_series_feature_mapping(feature_columns, series_names, changepoint_features_cols):
    """
    Build a mapping from series index to their corresponding feature column indices.
    
    This handles per-series changepoint features where each series has its own set
    of changepoint features (e.g., 'SeriesA_basic_changepoint_1', 'SeriesB_basic_changepoint_1').
    
    Args:
        feature_columns: All feature column names
        series_names: List of series names in order
        changepoint_features_cols: Changepoint feature column names
        
    Returns:
        tuple: (series_feat_mapping, has_per_series_features)
            - series_feat_mapping: dict mapping series_idx -> list of feature column indices
            - has_per_series_features: bool indicating if per-series features were detected
    """
    if len(changepoint_features_cols) == 0:
        return None, False
    
    # Check if we have per-series changepoint features
    # Per-series features have pattern: {series_name}_{method}_changepoint_{num}
    has_per_series = any(
        any(str(col).startswith(f"{series_name}_") for series_name in series_names)
        for col in changepoint_features_cols
    )
    
    if not has_per_series:
        return None, False
    
    # Build the mapping
    series_feat_mapping = {}
    feature_columns_list = list(feature_columns)
    
    for series_idx, series_name in enumerate(series_names):
        # Find all feature columns for this series
        # Includes: date features (shared), series-specific changepoint features, series-specific naive features
        series_feature_indices = []
        
        for feat_idx, feat_col in enumerate(feature_columns_list):
            feat_col_str = str(feat_col)
            
            # Include if it's a shared feature (doesn't start with any series name)
            is_shared = not any(
                feat_col_str.startswith(f"{sname}_") 
                for sname in series_names
            )
            
            # Or if it's specific to this series
            is_series_specific = feat_col_str.startswith(f"{series_name}_")
            
            # Include naive features for this series
            is_naive_for_series = feat_col_str == f"naive_last_{series_name}"
            
            if is_shared or is_series_specific or is_naive_for_series:
                series_feature_indices.append(feat_idx)
        
        series_feat_mapping[series_idx] = series_feature_indices
    
    return series_feat_mapping, True


def create_training_windows(
    y_train_scaled, 
    train_feats_scaled, 
    context_length, 
    max_windows=50000, 
    random_seed=2023,
    series_feat_mapping=None,
    series_names=None
):
    """
    Efficient shared function to create training windows for both MambaSSM and pMLP.
    
    Args:
        y_train_scaled: Scaled time series data (n_timesteps, n_series)
        train_feats_scaled: Scaled feature data (n_timesteps, n_features) or 
                           dict mapping series names to (n_timesteps, n_features_per_series)
        context_length: Length of context window
        max_windows: Maximum number of windows to generate
        random_seed: Random seed for sampling
        series_feat_mapping: Optional dict mapping series index to feature column indices
        series_names: Optional list of series names (for per-series features)
        
    Returns:
        tuple: (X_data, Y_data) where X_data has shape (n_windows, context_length, 1+n_features)
               and Y_data has shape (n_windows, 1)
    """
    # Calculate maximum possible windows for memory efficiency
    max_possible_windows = (y_train_scaled.shape[0] - context_length) * y_train_scaled.shape[1]
    max_windows = min(max_windows, max_possible_windows)
    
    # Use numpy for efficient window generation via stride tricks
    y_windows = sliding_window_view(y_train_scaled, context_length + 1, axis=0)
    
    # Handle per-series features
    if series_feat_mapping is not None and series_names is not None:
        # Per-series features: each series has its own set of feature columns
        num_windows_per_series = y_windows.shape[0]
        total_windows = num_windows_per_series * y_train_scaled.shape[1]
        
        # Sample windows if needed
        if total_windows > max_windows:
            rng = np.random.default_rng(random_seed)
            selected_indices = rng.choice(total_windows, size=max_windows, replace=False)
        else:
            selected_indices = np.arange(total_windows)
        
        # Calculate series and window indices
        series_indices = selected_indices % y_train_scaled.shape[1]
        window_indices = selected_indices // y_train_scaled.shape[1]
        
        # Determine max features across all series
        max_feats = max(len(feat_cols) for feat_cols in series_feat_mapping.values())

        # Pre-allocate output arrays
        X_data = np.zeros((len(selected_indices), context_length, 1 + max_feats), dtype=np.float32)
        Y_data = np.empty((len(selected_indices), 1), dtype=np.float32)

        # Cache feature windows once; avoid recomputing per series
        feat_windows_all = (
            sliding_window_view(train_feats_scaled, context_length, axis=0)
            if train_feats_scaled.shape[1] > 0
            else None
        )

        # Process each unique series to minimize feature extraction overhead
        for series_idx in range(y_train_scaled.shape[1]):
            # Find all samples for this series
            mask = series_indices == series_idx
            if not np.any(mask):
                continue
                
            series_window_indices = window_indices[mask]
            
            # Extract y values for this series
            X_data[mask, :, 0] = y_windows[series_window_indices, series_idx, :context_length]
            Y_data[mask, 0] = y_windows[series_window_indices, series_idx, context_length]
            
            # Extract series-specific features
            series_name = series_names[series_idx]
            feat_cols = series_feat_mapping.get(series_idx, [])
            
            if len(feat_cols) > 0:
                if feat_windows_all is None:
                    continue
                # Assign to X_data (features start at index 1)
                # feat_windows_all shape: (n_windows, n_features, context_length)
                # We need: (n_windows, context_length, n_features)
                series_feat_windows = feat_windows_all[series_window_indices][:, feat_cols, :]
                X_data[mask, :, 1:1+len(feat_cols)] = series_feat_windows.transpose(0, 2, 1)
        
        return X_data, Y_data
    else:
        # Original path: shared features across all series
        feat_windows = sliding_window_view(train_feats_scaled, context_length, axis=0) 
        
        # Flatten across all series and time windows
        num_windows_per_series = y_windows.shape[0]
        total_windows = num_windows_per_series * y_train_scaled.shape[1]
        
        # Sample windows if we exceed max_windows
        if total_windows > max_windows:
            rng = np.random.default_rng(random_seed)
            selected_indices = rng.choice(total_windows, size=max_windows, replace=False)
        else:
            selected_indices = np.arange(total_windows)
            
        # Generate training samples efficiently using vectorized operations
        num_features = train_feats_scaled.shape[1]
        
        # Vectorized index calculations (much faster than Python loop)
        series_indices = selected_indices % y_train_scaled.shape[1]
        window_indices = selected_indices // y_train_scaled.shape[1]
        
        # Pre-allocate output arrays
        X_data = np.empty((len(selected_indices), context_length, 1 + num_features), dtype=np.float32)
        Y_data = np.empty((len(selected_indices), 1), dtype=np.float32)
        
        # Vectorized data extraction using advanced indexing
        # Time series values: y_windows[window_indices, series_indices, :context_length]
        X_data[:, :, 0] = y_windows[window_indices, series_indices, :context_length]
        
        # Feature values: broadcast feat_windows across all selected samples
        # feat_windows[window_indices] has shape (n_samples, n_features, context_length)
        # We need (n_samples, context_length, n_features) so transpose the last two dimensions
        X_data[:, :, 1:] = feat_windows[window_indices].transpose(0, 2, 1)
        
        # Target values: y_windows[window_indices, series_indices, context_length]
        Y_data[:, 0] = y_windows[window_indices, series_indices, context_length]

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

        # Combine gating and output projection
        z_gated = F.silu(z)
        y = y * z_gated
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

        h = torch.zeros(B, d_inner, self.d_state, device=x.device)
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
        x = self.dropout(self.input_projection(x))

        for norm, block in self.layers:
            # Cache normalized input to avoid recomputation
            x_norm = norm(x)
            x = block(x_norm) + x

        x = self.norm_f(x)
        mu = self.out_mu(x)
        sigma = self.softplus(self.out_sigma(x))
        sigma = torch.clamp(sigma, min=1e-6)  # More efficient than addition
        return mu, sigma


class MLPCore(Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[512, 256],  # Wide but shallow by default
        dropout_rate=0.2,
        use_batch_norm=True,
        activation='relu',
    ):
        super().__init__()
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        
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
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional, but often helps with wide networks)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout (skip on last hidden layer to avoid over-regularization before output)
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
            mu: Mean predictions 
            sigma: Standard deviation predictions
        """
        # Handle both sequence and non-sequence inputs
        if x.dim() == 3:
            # For sequence input, we can either:
            # 1. Take the last timestep (like RNN)
            # 2. Global average pooling
            # 3. Flatten the sequence
            # For time series, taking last timestep often works well
            x = x[:, -1, :]  # Take last timestep: (batch_size, input_dim)
        
        # Process through backbone
        hidden = self.backbone(x)
        
        # Generate probabilistic outputs
        mu = self.out_mu(hidden)
        sigma = self.softplus(self.out_sigma(hidden))
        sigma = torch.clamp(sigma, min=1e-6)  # Ensure positive sigma
        
        return mu, sigma


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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"  # Added for MacOS users
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
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

        # Last value naive features (constant per series)
        if self.use_naive_feature:
            last_values = df.ffill().iloc[-1].fillna(0.0).astype(np.float32)
            naive_feature_mapping = {
                f"naive_last_{col}": float(last_values[col]) for col in df.columns
            }
            naive_features = pd.DataFrame(
                naive_feature_mapping,
                index=df.index,
                dtype=np.float32,
            )
            self.naive_feature_columns = naive_features.columns
            self.naive_feature_values = last_values
        else:
            naive_features = None
            self.naive_feature_columns = pd.Index([])
            self.naive_feature_values = None
        
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
        
        # 4b. Create training windows using shared efficient function
        X_data, Y_data = create_training_windows(
            y_train_scaled, 
            train_feats_scaled, 
            self.context_length, 
            max_windows=50000,
            random_seed=self.random_seed,
            series_feat_mapping=series_feat_mapping,
            series_names=series_names if has_per_series else None
        )

        # 5. Torch plumbing
        # Account for potentially different feature dimensions per series
        if has_per_series:
            # Use max features across all series
            num_features = X_data.shape[2] - 1  # Subtract 1 for the y value
        else:
            num_features = train_feats_scaled.shape[1]
        input_dim = 1 + num_features
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
                
                # Extract only the last timestep prediction for loss calculation
                mu_last = mu[:, -1, :]  # Shape: (batch_size, 1)
                sigma_last = sigma[:, -1, :]  # Shape: (batch_size, 1)
                
                loss = criterion(mu_last, sigma_last, y_batch)
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
        """Roll-forward prediction."""
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
        
        # Create future changepoint features without re-fitting detector
        if (
            getattr(self, "changepoint_features", None) is not None
            and not self.changepoint_features.empty
        ):
            future_changepoint_feats = changepoint_fcst_from_last_row(
                self.changepoint_features.iloc[-1], n_forecast=forecast_length
            )
            future_changepoint_feats.index = forecast_index
            future_changepoint_feats = future_changepoint_feats.reindex(
                columns=self.changepoint_features_columns
            )
        else:
            future_changepoint_feats = pd.DataFrame(
                index=forecast_index,
                columns=getattr(self, "changepoint_features_columns", []),
            )

        # Naive last value features extended as constant
        if self.use_naive_feature and self.naive_feature_values is not None:
            naive_future_feats = pd.DataFrame(
                {
                    f"naive_last_{col}": float(self.naive_feature_values[col])
                    for col in self.df_train.columns
                },
                index=forecast_index,
                dtype=np.float32,
            )
            naive_future_feats = naive_future_feats.reindex(
                columns=self.naive_feature_columns
            )
        else:
            naive_future_feats = None
        
        # Combine all future features
        feature_list = [future_date_feats, future_changepoint_feats]
        if naive_future_feats is not None:
            feature_list.append(naive_future_feats)
        if future_regressor is not None:
            feature_list.append(future_regressor)
            
        feat_future_df = pd.concat(feature_list, axis=1)

        feat_future_df = (
            feat_future_df.reindex(columns=self.feature_columns)
            .ffill()
            .bfill()
            .fillna(0.0)
            .astype(np.float32)
        )
        future_feats_scaled = self.feature_scaler.transform(feat_future_df.values)

        # 3. Context tensors (scaled) - cache the scaled training data
        if not hasattr(self, "_cached_y_train_scaled"):
            self._cached_y_train_scaled = (
                self.df_train.to_numpy(dtype=np.float32) - self.scaler_means
            ) / self.scaler_stds

        ctx_ts = torch.tensor(
            self._cached_y_train_scaled[-self.context_length :, :],
            device=self.device,
            dtype=torch.float32,
        )

        ctx_feat_np = self.feature_scaler.transform(
            self.features.iloc[-self.context_length :].values.astype(np.float32)
        )
        ctx_feat = torch.tensor(ctx_feat_np, device=self.device, dtype=torch.float32)

        num_series = self.df_train.shape[1]
        forecast_mu_scaled = np.zeros((forecast_length, num_series), dtype=np.float32)
        forecast_sigma_scaled = np.zeros(
            (forecast_length, num_series), dtype=np.float32
        )

        # 4. Roll forward with per-series feature support
        with torch.no_grad():
            if getattr(self, 'has_per_series_features', False):
                # Per-series features: process each series individually
                series_names = list(self.df_train.columns)
                
                # Determine the max feature dimension used during training
                # This should match what was used in create_training_windows
                max_feats_from_mapping = max(len(feat_indices) for feat_indices in self.series_feat_mapping.values())
                
                for series_idx in range(num_series):
                    if self.verbose and series_idx % 10 == 0:
                        print(f"Predicting series {series_idx+1}/{num_series}")
                    
                    # Get feature indices for this series
                    feat_indices = self.series_feat_mapping.get(series_idx, [])
                    
                    # Initialize context for this series
                    series_ctx_ts = ctx_ts[:, series_idx].unsqueeze(1)  # (context_length, 1)
                    
                    # Extract features for this series from the full feature array
                    series_ctx_feat_full = ctx_feat_np[:, feat_indices]  # (context_length, n_feats_for_series)
                    
                    # Pad to match model input dimension if needed
                    if len(feat_indices) < max_feats_from_mapping:
                        pad_size = max_feats_from_mapping - len(feat_indices)
                        padding = np.zeros((self.context_length, pad_size), dtype=np.float32)
                        series_ctx_feat_full = np.concatenate([series_ctx_feat_full, padding], axis=1)
                    
                    series_ctx_feat = torch.tensor(series_ctx_feat_full, device=self.device, dtype=torch.float32)
                    
                    for step in range(forecast_length):
                        # Build model input for this series
                        model_in = torch.cat([series_ctx_ts, series_ctx_feat], dim=-1).unsqueeze(0)
                        mu, sigma = self.model(model_in)
                        
                        # Extract prediction
                        mu_next = mu[0, -1, 0]
                        sigma_next = sigma[0, -1, 0]
                        
                        # Store predictions
                        forecast_mu_scaled[step, series_idx] = mu_next.cpu().numpy()
                        forecast_sigma_scaled[step, series_idx] = sigma_next.cpu().numpy()
                        
                        # Update context
                        series_ctx_ts = torch.cat([series_ctx_ts[1:], mu_next.unsqueeze(0).unsqueeze(1)], dim=0)
                        
                        # Update features for next step
                        next_feat_full = future_feats_scaled[step, feat_indices]
                        
                        # Pad if needed
                        if len(feat_indices) < max_feats_from_mapping:
                            pad_size = max_feats_from_mapping - len(feat_indices)
                            padding = np.zeros(pad_size, dtype=np.float32)
                            next_feat_full = np.concatenate([next_feat_full, padding])
                        
                        next_feat_tensor = torch.tensor(next_feat_full, device=self.device, dtype=torch.float32)
                        series_ctx_feat = torch.cat([series_ctx_feat[1:], next_feat_tensor.unsqueeze(0)], dim=0)
            else:
                # Original path: shared features across all series
                # Pre-allocate broadcast tensor shape to avoid repeated allocations
                feat_broadcast_shape = (num_series, self.context_length, ctx_feat.shape[1])

                for step in range(forecast_length):
                    if self.verbose and step % self.prediction_batch_size == 0:
                        print(f"Predicting step {step+1}/{forecast_length}")

                    # More efficient broadcasting without repeat()
                    feat_broadcast = ctx_feat.unsqueeze(0).expand(feat_broadcast_shape)
                    model_in = torch.cat([ctx_ts.T.unsqueeze(-1), feat_broadcast], dim=-1)
                    mu, sigma = self.model(model_in)

                    # Extract next predictions (already on correct device)
                    mu_next = mu[:, -1, 0]  # Remove unnecessary squeeze operations
                    sigma_next = sigma[:, -1, 0]

                    # Store predictions (convert to numpy once)
                    forecast_mu_scaled[step] = mu_next.cpu().numpy()
                    forecast_sigma_scaled[step] = sigma_next.cpu().numpy()

                    # Update context more efficiently
                    ctx_ts = torch.cat([ctx_ts[1:], mu_next.unsqueeze(0)], dim=0)
                    # Update features by shifting and replacing last element
                    ctx_feat[:-1] = ctx_feat[1:]
                    ctx_feat[-1] = torch.tensor(
                        future_feats_scaled[step], device=self.device, dtype=ctx_feat.dtype
                    )

        # 5. Un-scale & wrap
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
    def __init__(
        self,
        name: str = "pMLP",  # Probabilistic MLP
        frequency: str = "infer",
        prediction_interval: float = 0.9,
        holiday_country: str = "US",
        random_seed: int = 2023,
        verbose: int = 1,
        context_length: int = 60,  # Shorter default for MLP efficiency
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
        self.context_length = context_length
        
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

        # Loss function parameters
        self.loss_function = loss_function
        self.nll_weight = nll_weight
        self.wasserstein_weight = wasserstein_weight

        self.prediction_batch_size = prediction_batch_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"  # Added for MacOS users
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
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

        if self.use_naive_feature:
            last_values = df.ffill().iloc[-1].fillna(0.0).astype(np.float32)
            naive_feature_mapping = {
                f"naive_last_{col}": float(last_values[col]) for col in df.columns
            }
            naive_features = pd.DataFrame(
                naive_feature_mapping,
                index=df.index,
                dtype=np.float32,
            )
            self.naive_feature_columns = naive_features.columns
            self.naive_feature_values = last_values
        else:
            naive_features = None
            self.naive_feature_columns = pd.Index([])
            self.naive_feature_values = None
        
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
        
        # 4b. Create training windows using shared efficient function
        X_data, Y_data = create_training_windows(
            y_train_scaled, 
            train_feats_scaled, 
            self.context_length, 
            max_windows=100000,  # More windows for MLP efficiency
            random_seed=self.random_seed,
            series_feat_mapping=series_feat_mapping,
            series_names=series_names if has_per_series else None
        )

        # 5. Torch setup - optimized for MLP
        # Account for potentially different feature dimensions per series
        if has_per_series:
            # Use max features across all series
            num_features = X_data.shape[2] - 1  # Subtract 1 for the y value
        else:
            num_features = train_feats_scaled.shape[1]
        input_dim = 1 + num_features
        self.model = MLPCore(
            input_dim * self.context_length,  # Flatten the input for MLP
            self.hidden_dims,
            self.dropout_rate,
            self.use_batch_norm,
            self.activation,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = self._get_loss_function()

        # Create tensors and dataset - flatten inputs for MLP
        try:
            # Flatten the sequence dimension for MLP
            X_flattened = X_data.reshape(X_data.shape[0], -1)
            X_tensor = torch.tensor(X_flattened, dtype=torch.float32)
            Y_tensor = torch.tensor(Y_data, dtype=torch.float32)
        except (AttributeError, NameError):
            # Fallback when torch is not available
            X_tensor = X_data.reshape(X_data.shape[0], -1)
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
            print(f"Training pMLP on {self.device} • {len(dataset):,} samples/epoch")

        # 6. Training loop - optimized for efficiency
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.model.train()
        
        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            epoch_losses = []
            
            for x_batch, y_batch in tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                disable=(self.verbose == 0),
            ):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                mu, sigma = self.model(x_batch)
                
                loss = criterion(mu, sigma, y_batch)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                loss_item = loss.item()
                running_loss += loss_item
                epoch_losses.append(loss_item)
            
            avg_loss = running_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if self.verbose:
                print(f"Epoch {epoch+1}  avg-loss: {avg_loss:.4f}")

        self.fit_runtime = datetime.datetime.now() - fit_start_time
        return self

    def predict(self, forecast_length: int, future_regressor=None, **kwargs):
        """Roll-forward prediction for pMLP."""
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
        
        # Create future changepoint features without re-fitting detector
        if (
            getattr(self, "changepoint_features", None) is not None
            and not self.changepoint_features.empty
        ):
            future_changepoint_feats = changepoint_fcst_from_last_row(
                self.changepoint_features.iloc[-1], n_forecast=forecast_length
            )
            future_changepoint_feats.index = forecast_index
            future_changepoint_feats = future_changepoint_feats.reindex(
                columns=self.changepoint_features_columns
            )
        else:
            future_changepoint_feats = pd.DataFrame(
                index=forecast_index,
                columns=getattr(self, "changepoint_features_columns", []),
            )

        if self.use_naive_feature and self.naive_feature_values is not None:
            naive_future_feats = pd.DataFrame(
                {
                    f"naive_last_{col}": float(self.naive_feature_values[col])
                    for col in self.df_train.columns
                },
                index=forecast_index,
                dtype=np.float32,
            )
            naive_future_feats = naive_future_feats.reindex(
                columns=self.naive_feature_columns
            )
        else:
            naive_future_feats = None
        
        # Combine all future features
        feature_list = [future_date_feats, future_changepoint_feats]
        if naive_future_feats is not None:
            feature_list.append(naive_future_feats)
        if future_regressor is not None:
            feature_list.append(future_regressor)
            
        feat_future_df = pd.concat(feature_list, axis=1)

        feat_future_df = (
            feat_future_df.reindex(columns=self.feature_columns)
            .ffill()
            .bfill()
            .fillna(0.0)
            .astype(np.float32)
        )
        future_feats_scaled = self.feature_scaler.transform(feat_future_df.values)

        # 3. Context tensors (scaled) - cache the scaled training data
        if not hasattr(self, "_cached_y_train_scaled"):
            self._cached_y_train_scaled = (
                self.df_train.to_numpy(dtype=np.float32) - self.scaler_means
            ) / self.scaler_stds

        # Get context window
        ctx_ts = self._cached_y_train_scaled[-self.context_length :, :]
        ctx_feat_np = self.feature_scaler.transform(
            self.features.iloc[-self.context_length :].values.astype(np.float32)
        )

        num_series = self.df_train.shape[1]
        forecast_mu_scaled = np.zeros((forecast_length, num_series), dtype=np.float32)
        forecast_sigma_scaled = np.zeros((forecast_length, num_series), dtype=np.float32)

        # 4. Roll forward prediction with per-series feature support
        with torch.no_grad():
            if getattr(self, 'has_per_series_features', False):
                # Per-series features: process each series individually
                series_names = list(self.df_train.columns)
                
                # Determine the max feature dimension used during training
                max_feats_from_mapping = max(len(feat_indices) for feat_indices in self.series_feat_mapping.values())
                
                for series_idx in range(num_series):
                    if self.verbose and series_idx % 10 == 0:
                        print(f"Predicting series {series_idx+1}/{num_series}")
                    
                    # Get feature indices for this series
                    feat_indices = self.series_feat_mapping.get(series_idx, [])
                    
                    # Initialize context for this series
                    series_ctx_ts = ctx_ts[:, series_idx:series_idx+1]  # (context_length, 1)
                    series_ctx_feat = ctx_feat_np[:, feat_indices]  # (context_length, n_feats_for_series)
                    
                    # Pad to match model input dimension if needed
                    if len(feat_indices) < max_feats_from_mapping:
                        pad_size = max_feats_from_mapping - len(feat_indices)
                        padding = np.zeros((self.context_length, pad_size), dtype=np.float32)
                        series_ctx_feat = np.concatenate([series_ctx_feat, padding], axis=1)
                    
                    for step in range(forecast_length):
                        # Combine and flatten for MLP
                        combined = np.concatenate([series_ctx_ts, series_ctx_feat], axis=1)
                        flattened = combined.flatten()
                        
                        # Predict
                        input_tensor = torch.tensor(
                            flattened, 
                            device=self.device, 
                            dtype=torch.float32
                        ).unsqueeze(0)
                        
                        mu, sigma = self.model(input_tensor)
                        
                        # Store predictions
                        forecast_mu_scaled[step, series_idx] = mu.cpu().numpy().flatten()[0]
                        forecast_sigma_scaled[step, series_idx] = sigma.cpu().numpy().flatten()[0]
                        
                        # Update context
                        new_value = mu.cpu().numpy().flatten()[0]
                        series_ctx_ts = np.concatenate([series_ctx_ts[1:], [[new_value]]], axis=0)
                        
                        # Update features for next step
                        if step < len(future_feats_scaled):
                            next_feat_full = future_feats_scaled[step, feat_indices]
                            
                            # Pad if needed
                            if len(feat_indices) < max_feats_from_mapping:
                                pad_size = max_feats_from_mapping - len(feat_indices)
                                padding = np.zeros(pad_size, dtype=np.float32)
                                next_feat_full = np.concatenate([next_feat_full, padding])
                            
                            series_ctx_feat = np.concatenate([series_ctx_feat[1:], [next_feat_full]], axis=0)
            else:
                # Original path: shared features across all series
                for step in range(forecast_length):
                    if self.verbose and step % self.prediction_batch_size == 0:
                        print(f"Predicting step {step+1}/{forecast_length}")

                    # Prepare input for all series at once
                    batch_inputs = []
                    for series_idx in range(num_series):
                        # Combine time series values with features
                        ts_values = ctx_ts[:, series_idx:series_idx+1]  # Shape: (context_length, 1)
                        features = ctx_feat_np  # Shape: (context_length, n_features)
                        combined = np.concatenate([ts_values, features], axis=1)  # Shape: (context_length, 1+n_features)
                        flattened = combined.flatten()  # Flatten for MLP
                        batch_inputs.append(flattened)
                    
                    # Convert to tensor and predict
                    input_tensor = torch.tensor(
                        np.array(batch_inputs), 
                        device=self.device, 
                        dtype=torch.float32
                    )
                    
                    mu, sigma = self.model(input_tensor)
                    
                    # Store predictions
                    forecast_mu_scaled[step] = mu.cpu().numpy().flatten()
                    forecast_sigma_scaled[step] = sigma.cpu().numpy().flatten()

                    # Update context for next step
                    new_values = mu.cpu().numpy().flatten()
                    ctx_ts = np.concatenate([ctx_ts[1:], new_values.reshape(1, -1)], axis=0)
                    
                    # Update features
                    if step < len(future_feats_scaled):
                        new_feat = future_feats_scaled[step].reshape(1, -1)
                        ctx_feat_np = np.concatenate([ctx_feat_np[1:], new_feat], axis=0)

        # 5. Un-scale & wrap
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

        # Context length options - shorter for MLP efficiency
        context_lengths = [20, 30, 45, 60, 90, 120]
        context_weights = [0.15, 0.25, 0.25, 0.2, 0.1, 0.05]  # Prefer shorter contexts
        
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
        ]
        hidden_weights = [0.15, 0.2, 0.15, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]
        
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
        
        # Datepart method
        datepart_method = random_datepart()
        
        # Boolean options
        holiday_used_options = [True, False]
        holiday_weights = [0.3, 0.7]
        
        naive_feature_options = [True, False]
        naive_feature_weights = [0.75, 0.25]
        
        # Generate changepoint method and parameters
        changepoint_method, changepoint_params = generate_random_changepoint_params()

        # Generate random selections
        selected_params = {
            "context_length": random.choices(context_lengths, weights=context_weights, k=1)[0],
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
