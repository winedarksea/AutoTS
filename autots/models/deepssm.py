import pandas as pd
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from scipy.stats import norm
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Corrected import path
from autots.tools.seasonal import date_part

# Placeholder for the base class (unchanged)
class ModelObject:
    def __init__(self, name, frequency, prediction_interval, **kwargs):
        self.name = name
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.fit_runtime = datetime.timedelta(0)
        self.verbose = kwargs.get('verbose', 0)
        self.random_seed = kwargs.get('random_seed', 2023)
        self.holiday_country = kwargs.get('holiday_country', 'US') # Added for consistency
        self.column_names = []
        self.df_train = None

    def basic_profile(self, df):
        self.column_names = df.columns
        self.df_train = df
        return df

    def create_forecast_index(self, forecast_length):
        if self.df_train.index.freq is None:
            freq = pd.infer_freq(self.df_train.index)
            if freq is None: raise ValueError("Could not infer frequency from DatetimeIndex.")
        else:
            freq = self.df_train.index.freq
        return pd.date_range(start=self.df_train.index[-1], periods=forecast_length + 1, freq=freq)[1:]

# Placeholder for the output object
@dataclass
class PredictionObject:
    model_name: str; forecast_length: int; forecast_index: pd.DatetimeIndex; forecast_columns: pd.Index
    lower_forecast: pd.DataFrame; forecast: pd.DataFrame; upper_forecast: pd.DataFrame
    prediction_interval: float; predict_runtime: datetime.timedelta; fit_runtime: datetime.timedelta; model_parameters: dict

# Custom Dataset and Loss
class TimeSeriesDataset(Dataset):
    def __init__(self, ts_data, feature_data, context_length):
        super().__init__(); self.ts_data = ts_data; self.feature_data = feature_data
        self.context_length = context_length; self.n_timesteps, self.n_series = ts_data.shape
    def __len__(self): return (self.n_timesteps - self.context_length) * self.n_series
    def __getitem__(self, idx):
        series_idx = idx % self.n_series; start_idx = idx // self.n_series
        end_idx = start_idx + self.context_length
        x_ts = self.ts_data[start_idx:end_idx, series_idx]
        x_features = self.feature_data[start_idx:end_idx]
        x = np.concatenate([x_ts[:, None], x_features], axis=1)
        y_ts = self.ts_data[start_idx + 1 : end_idx + 1, series_idx]
        y = y_ts[:, None]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

class CombinedNLLWassersteinLoss(nn.Module):
    def __init__(self, nll_weight=1.0, wasserstein_weight=0.1):
        super().__init__(); self.nll_weight = nll_weight; self.wasserstein_weight = wasserstein_weight
        self.nll_loss = nn.GaussianNLLLoss(reduction='mean')
    def _wasserstein_distance(self, p, q):
        p_sorted, _ = torch.sort(p, dim=0); q_sorted, _ = torch.sort(q, dim=0)
        return torch.abs(p_sorted - q_sorted).mean()
    def forward(self, mu, sigma, y_true):
        sigma = torch.clamp(sigma, min=1e-6); nll = self.nll_loss(mu, y_true, sigma.pow(2))
        wasserstein = self._wasserstein_distance(mu.flatten(), y_true.flatten())
        return self.nll_weight * nll + self.wasserstein_weight * wasserstein

# Self-contained Mamba Block and Core Model
class MambaMinimalBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = d_model, d_state, d_conv, expand
        self.d_inner = int(self.expand * self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=d_conv, bias=True, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(1).repeat(1, self.d_inner).T
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))
    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x); x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2); x = self.conv1d(x)[:, :, :L]; x = x.transpose(1, 2)
        x = F.silu(x); y = self.ssm(x)
        y = y * F.silu(z); output = self.out_proj(y)
        return output
    def ssm(self, x):
        B, L, d_inner = x.shape; A = -torch.exp(self.A_log.float()); delta = F.softplus(self.dt_proj(x))
        bc = self.x_proj(x); B_val, C_val = torch.split(bc, self.d_state, dim=-1)
        h = torch.zeros(B, d_inner, self.d_state, device=x.device); ys = []
        for i in range(L):
            delta_i = delta[:, i, :].unsqueeze(-1); A_bar = torch.exp(delta_i * A)
            B_bar = delta_i * B_val[:, i, :].unsqueeze(1)
            h = A_bar * h + B_bar * x[:, i, :].unsqueeze(-1)
            y_i = torch.sum(h * C_val[:, i, :].unsqueeze(1), dim=-1)
            ys.append(y_i)
        y = torch.stack(ys, dim=1); y = y + x * self.D
        return y
class MambaCore(nn.Module):
    def __init__(self, input_dim, d_model=64, n_layers=4, d_state=16, d_conv=4):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model); self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([nn.ModuleList([nn.LayerNorm(d_model), MambaMinimalBlock(d_model=d_model, d_state=d_state, d_conv=d_conv)]) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model); self.out_mu = nn.Linear(d_model, 1); self.out_sigma = nn.Linear(d_model, 1); self.softplus = nn.Softplus()
    def forward(self, x):
        x = self.input_projection(x); x = self.dropout(x)
        for norm, block in self.layers: x = block(norm(x)) + x
        x = self.norm_f(x); mu = self.out_mu(x); sigma = self.softplus(self.out_sigma(x)) + 1e-6
        return mu, sigma

# Main Forecaster Class with all fixes
class MambaSSMForecaster(ModelObject):
    def __init__(
        self, name: str = "MambaSSM_local", frequency: str = 'infer', prediction_interval: float = 0.9, holiday_country: str = 'US',
        random_seed: int = 2023, verbose: int = 1, context_length: int = 120, d_model: int = 32, n_layers: int = 2,
        d_state: int = 8, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, nll_weight: float = 1.0,
        wasserstein_weight: float = 0.1, prediction_batch_size: int = 60, datepart_method: str = 'expanded',
        holiday_countries_used: bool = False, **kwargs,
    ):
        ModelObject.__init__(
            self, name, frequency, prediction_interval, holiday_country=holiday_country,
            random_seed=random_seed, verbose=verbose
        )
        self.datepart_method = datepart_method
        self.holiday_countries_used = holiday_countries_used
        self.context_length = context_length; self.d_model = d_model; self.n_layers = n_layers
        self.d_state = d_state; self.epochs = epochs; self.batch_size = batch_size; self.lr = lr
        self.nll_weight = nll_weight; self.wasserstein_weight = wasserstein_weight
        self.prediction_batch_size = prediction_batch_size
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available(): self.device = 'mps' # Added for MacOS users
        torch.manual_seed(self.random_seed); np.random.seed(self.random_seed)

    def fit(self, df, future_regressor=None, **kwargs):
        """Train the Mamba SSM forecaster."""
        fit_start_time = datetime.datetime.now()
    
        # 1. Book-keeping
        df = self.basic_profile(df)                       # saves col names / df_train
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
            feat_df_train = pd.concat([date_feats_train, future_regressor], axis=1).reindex(train_index)
        else:
            feat_df_train = date_feats_train
    
        feat_df_train = (
            feat_df_train.ffill().bfill().astype(np.float32)
        )
        feat_df_train.columns = [str(c) for c in feat_df_train.columns]
        self.features = feat_df_train                       # <- *** stored for predict ***
        self.feature_columns = feat_df_train.columns
    
        # 3. Scaling
        self.scaler_means = np.mean(y_train, axis=0)
        self.scaler_stds  = np.std(y_train,  axis=0)
        self.scaler_stds[self.scaler_stds == 0.0] = 1.0
    
        y_train_scaled = (y_train - self.scaler_means) / self.scaler_stds
    
        self.feature_scaler = StandardScaler()
        train_feats_scaled  = self.feature_scaler.fit_transform(feat_df_train.values)
    
        # 4. Torch plumbing
        input_dim = 1 + train_feats_scaled.shape[1]
        self.model  = MambaCore(input_dim, self.d_model, self.n_layers, self.d_state).to(self.device)
        optimizer   = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion   = CombinedNLLWassersteinLoss(self.nll_weight, self.wasserstein_weight)
    
        dataset   = TimeSeriesDataset(y_train_scaled, train_feats_scaled, self.context_length)
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
        torch.manual_seed(self.random_seed); np.random.seed(self.random_seed)
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for x_batch, y_batch in tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", disable=(self.verbose == 0)
            ):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                mu, sigma = self.model(x_batch)
                loss = criterion(mu, sigma, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if self.verbose:
                print(f"Epoch {epoch+1}  avg-loss: {running_loss / len(dataloader):.4f}")
    
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
                         .ffill().bfill().fillna(0.0).astype(np.float32)
        )
        future_feats_scaled = self.feature_scaler.transform(feat_future_df.values)
    
        # 3. Context tensors (scaled)
        y_train_scaled = (self.df_train.to_numpy(dtype=np.float32) - self.scaler_means) / self.scaler_stds
        ctx_ts   = torch.tensor(y_train_scaled[-self.context_length:, :], device=self.device)
    
        ctx_feat_np = self.feature_scaler.transform(
            self.features.iloc[-self.context_length:].values.astype(np.float32)
        )
        ctx_feat = torch.tensor(ctx_feat_np, device=self.device)
    
        num_series = self.df_train.shape[1]
        forecast_mu_scaled    = np.zeros((forecast_length, num_series), dtype=np.float32)
        forecast_sigma_scaled = np.zeros((forecast_length, num_series), dtype=np.float32)
    
        # 4. Roll forward
        with torch.no_grad():
            for step in range(forecast_length):
                if self.verbose and step % self.prediction_batch_size == 0:
                    print(f"Predicting step {step+1}/{forecast_length}")
    
                feat_broadcast = ctx_feat.unsqueeze(0).repeat(num_series, 1, 1)
                model_in       = torch.cat([ctx_ts.T.unsqueeze(-1), feat_broadcast], dim=-1)
                mu, sigma      = self.model(model_in)
    
                mu_next    = mu[:, -1, :].squeeze(-1)
                sigma_next = sigma[:, -1, :].squeeze(-1)
    
                forecast_mu_scaled[step]    = mu_next.cpu().numpy()
                forecast_sigma_scaled[step] = sigma_next.cpu().numpy()
    
                ctx_ts = torch.cat([ctx_ts[1:], mu_next.unsqueeze(0)], dim=0).detach()
                ctx_feat = torch.roll(ctx_feat, shifts=-1, dims=0)
                ctx_feat[-1] = torch.tensor(future_feats_scaled[step], device=self.device)
    
        # 5. Un-scale & wrap
        mu_unscaled    = forecast_mu_scaled * self.scaler_stds + self.scaler_means
        sigma_unscaled = forecast_sigma_scaled * self.scaler_stds
    
        z = norm.ppf(1 - (1 - self.prediction_interval) / 2)
        lower = mu_unscaled - z * sigma_unscaled
        upper = mu_unscaled + z * sigma_unscaled
    
        forecast_df       = pd.DataFrame(mu_unscaled, index=forecast_index, columns=self.column_names)
        lower_forecast_df = pd.DataFrame(lower,        index=forecast_index, columns=self.column_names)
        upper_forecast_df = pd.DataFrame(upper,        index=forecast_index, columns=self.column_names)
    
        predict_runtime = datetime.datetime.now() - predict_start_time
        return PredictionObject(
            model_name         = self.name,
            forecast_length    = forecast_length,
            forecast_index     = forecast_df.index,
            forecast_columns   = forecast_df.columns,
            lower_forecast     = lower_forecast_df,
            forecast           = forecast_df,
            upper_forecast     = upper_forecast_df,
            prediction_interval= self.prediction_interval,
            predict_runtime    = predict_runtime,
            fit_runtime        = self.fit_runtime,
            model_parameters   = self.get_params(),
        )

    def get_params(self):
        return {
            "context_length": self.context_length, "d_model": self.d_model, "n_layers": self.n_layers, "d_state": self.d_state,
            "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr, "nll_weight": self.nll_weight,
            "wasserstein_weight": self.wasserstein_weight, "prediction_batch_size": self.prediction_batch_size,
            "datepart_method": self.datepart_method, "holiday_countries_used": self.holiday_countries_used,
        }


# Create some dummy data for demonstration
num_series = 100  # Reduced for faster demonstration
num_timesteps = 500
print(f"Generating dummy data for {num_series} series...")

dates = pd.date_range(start='2020-01-01', periods=num_timesteps, freq='D')
data = np.random.randn(num_timesteps, num_series).astype(np.float32)
time_factor = np.linspace(0, 5, num_timesteps)[:, None]
seasonality = np.sin(np.arange(num_timesteps) * 2 * np.pi / 365.25)[:, None]
data += time_factor + seasonality * 5

df_train = pd.DataFrame(data, index=dates, columns=[f'series_{i}' for i in range(num_series)])
print("Data shape:", df_train.shape)

# Instantiate and run the model
mamba_model = MambaSSMForecaster(
    context_length=90,
    epochs=3, # Reduced epochs for demo
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
print(f"Fit runtime: {prediction.fit_runtime}")
print(f"Predict runtime: {prediction.predict_runtime}")

print("\nPoint Forecast:")
print(prediction.forecast.iloc[:5, :5])
