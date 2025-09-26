import pandas as pd
import numpy as np
import datetime

from autots.tools.seasonal import date_part
from autots.models.base import ModelObject, PredictionObject
from autots.tools.seasonal import date_part, random_datepart
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
    from torch.utils.data import DataLoader, Dataset
except Exception:
    from autots.tools.mocks import Module, Dataset, DataLoader


# Custom Dataset and Loss
class TimeSeriesDataset(Dataset):
    def __init__(self, ts_data, feature_data, context_length):
        super().__init__()
        self.ts_data = ts_data
        self.feature_data = feature_data
        self.context_length = context_length
        self.n_timesteps, self.n_series = ts_data.shape
        self.feature_dim = feature_data.shape[1]

    def __len__(self):
        return (self.n_timesteps - self.context_length) * self.n_series

    def __getitem__(self, idx):
        series_idx = idx % self.n_series
        start_idx = idx // self.n_series
        end_idx = start_idx + self.context_length

        # Pre-allocate arrays to reduce memory allocations
        x = np.empty((self.context_length, 1 + self.feature_dim), dtype=np.float32)
        x[:, 0] = self.ts_data[start_idx:end_idx, series_idx]
        x[:, 1:] = self.feature_data[start_idx:end_idx]

        y = self.ts_data[
            start_idx + 1 : end_idx + 1, series_idx : series_idx + 1
        ].astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


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
        losses = []
        for q in self.quantiles:
            # Convert quantile to standard normal quantile
            z_q = torch.erfinv(2 * q - 1) * np.sqrt(2)
            q_pred = mu + sigma * z_q
            error = y_true - q_pred
            loss_q = torch.max(q * error, (q - 1) * error)
            losses.append(loss_q)
        return torch.stack(losses).mean()


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

        # Energy score: E[||X - y||^β] - 0.5 * E[||X - X'||^β]
        # where X, X' are independent samples from predicted distribution

        # Term 1: Expected distance between samples and observations
        if self.beta == 1.0:
            term1 = torch.norm(samples - y_true.unsqueeze(0), dim=-1).mean(0)
        else:
            term1 = (
                torch.norm(samples - y_true.unsqueeze(0), dim=-1).pow(self.beta).mean(0)
            )

        # Term 2: Expected distance between sample pairs
        if self.beta == 1.0:
            # More efficient computation for β=1
            pairwise_diffs = samples.unsqueeze(0) - samples.unsqueeze(1)
            term2 = 0.5 * torch.norm(pairwise_diffs, dim=-1).mean()
        else:
            pairwise_diffs = samples.unsqueeze(0) - samples.unsqueeze(1)
            term2 = 0.5 * torch.norm(pairwise_diffs, dim=-1).pow(self.beta).mean()

        return (term1 - term2).mean()


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
        )
        self.datepart_method = datepart_method
        self.holiday_countries_used = holiday_countries_used
        self.use_extra_gating = use_extra_gating
        self.context_length = context_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

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
        else:
            raise ValueError(
                f"Unknown loss function: {self.loss_function}. "
                "Options: 'combined_nll_wasserstein', 'quantile', 'crps', 'energy', 'regularized_nll'"
            )

    def fit(self, df, future_regressor=None, **kwargs):
        """Train the Mamba SSM forecaster."""
        fit_start_time = datetime.datetime.now()

        # 1. Book-keeping
        df = self.basic_profile(df)  # saves col names
        self.df_train = df  # Store training data for predict method
        y_train = df.to_numpy(dtype=np.float32)
        train_index = df.index

        # 2. Date-part + optional regressors
        date_feats_train = date_part(
            train_index,
            method=self.datepart_method,
            holiday_country=self.holiday_country,
            holiday_countries_used=self.holiday_countries_used,
        )

        if future_regressor is not None:
            feat_df_train = pd.concat(
                [date_feats_train, future_regressor], axis=1
            ).reindex(train_index)
        else:
            feat_df_train = date_feats_train

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

        # 4. Torch plumbing
        input_dim = 1 + train_feats_scaled.shape[1]
        self.model = MambaCore(
            input_dim,
            self.d_model,
            self.n_layers,
            self.d_state,
            use_extra_gating=self.use_extra_gating,
        ).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = self._get_loss_function()

        dataset = TimeSeriesDataset(
            y_train_scaled, train_feats_scaled, self.context_length
        )
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
        if future_regressor is not None:
            feat_future_df = pd.concat([future_date_feats, future_regressor], axis=1)
        else:
            feat_future_df = future_date_feats

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

        # 4. Roll forward
        with torch.no_grad():
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
        }

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
            "energy"
        ]
        loss_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Prefer combined and crps
        
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
        }
        
        # Add prediction_batch_size based on context_length (longer contexts need smaller batches)
        if selected_params["context_length"] >= 180:
            selected_params["prediction_batch_size"] = random.choices([30, 45, 60], weights=[0.4, 0.4, 0.2], k=1)[0]
        else:
            selected_params["prediction_batch_size"] = random.choices([45, 60, 90], weights=[0.3, 0.4, 0.3], k=1)[0]
        
        return selected_params


if False:
    from autots import load_daily

    df_train = load_daily(long=False).ffill().bfill()
    # Instantiate and run the model
    mamba_model = MambaSSM(
        context_length=90,
        epochs=3,  # Reduced epochs for demo
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
