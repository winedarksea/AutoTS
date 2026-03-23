# -*- coding: utf-8 -*-
"""
TVA — Time Variant Architecture.

Top-level orchestrator for the TVA forecasting graph. Ties together
decomposition, priors, trend network, fusion, losses, reconciliation,
and scenario planning into a single fit/predict interface.
"""

import numpy as np
import pandas as pd
from typing import Optional

from autots.evaluator.tva.decomposition import NornDecomposer
from autots.evaluator.tva.priors import YggdrasilPriors, SeriesMetadata
from autots.evaluator.tva.reconciliation import ReconciliationBridge

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


class TVA:
    """Time Variant Architecture — end-to-end coherent forecasting graph.

    The Common Operating Picture for time series. Produces structurally
    consistent forecasts across related series by routing all trends through
    shared composite prototypes.

    Args:
        detector_params: Dict passed to TimeSeriesFeatureDetector.
        trend_network: 'v1' (hierarchical latent) or 'v2' (learned directed).
        fusion: 'attention' (DigitalTwinFusion) or 'additive' (AdditiveFusion).
        series_metadata: List of SeriesMetadata for prior construction.
        prior_adjacency: (N, N) explicit prior adjacency matrix (optional).
        prior_confidence: Weight of priors (0=ignore, 1=rigid).
        d_token: Token/latent dimension.
        n_meso: Number of meso latent nodes.
        n_global: Number of global latent nodes.
        n_prototypes: Number of prototype trend signatures.
        n_heads: Attention heads.
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        window_size: Input trend window length.
        forecast_horizon: Output forecast length.
        loss_weights: Dict overriding loss component weights.
        reconciliation_method: None, 'mint', 'erm', etc.
        min_anchor_history: Minimum periods for a series to be an anchor.
        device: 'cpu' or 'cuda'.
        random_seed: Reproducibility seed.
        verbose: 0=silent, 1=progress bar, 2=per-epoch loss.
    """

    def __init__(
        self,
        detector_params: dict = None,
        trend_network: str = 'v2',
        fusion: str = 'attention',
        series_metadata: list = None,
        prior_adjacency: np.ndarray = None,
        prior_confidence: float = 0.3,
        d_token: int = 64,
        n_meso: int = 8,
        n_global: int = 4,
        n_prototypes: int = 4,
        n_heads: int = 4,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        window_size: int = 90,
        forecast_horizon: int = 30,
        loss_weights: dict = None,
        reconciliation_method: str = None,
        min_anchor_history: int = 180,
        device: str = 'cpu',
        random_seed: int = 42,
        verbose: int = 1,
    ):
        if not HAS_TORCH:
            raise ImportError("TVA requires PyTorch. Install with: pip install torch")

        self.detector_params = detector_params
        self.trend_network_type = trend_network
        self.fusion_type = fusion
        self.series_metadata = series_metadata
        self.prior_adjacency = prior_adjacency
        self.prior_confidence = prior_confidence
        self.d_token = d_token
        self.n_meso = n_meso
        self.n_global = n_global
        self.n_prototypes = n_prototypes
        self.n_heads = n_heads
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.loss_weights = loss_weights
        self.reconciliation_method = reconciliation_method
        self.min_anchor_history = min_anchor_history
        self.device = device
        self.random_seed = random_seed
        self.verbose = verbose

        # fitted state
        self._decomposer = None
        self._priors = None
        self._network = None
        self._fusion_layer = None
        self._loss_fn = None
        self._reconciler = None
        self._components = None
        self._df_original = None
        self._anchor_mask = None
        self._metadata_embeddings = None
        self._prior_adj = None

    def fit(self, df: pd.DataFrame) -> 'TVA':
        """Full TVA pipeline: decompose, build priors, train network.

        Args:
            df: Wide DataFrame with DatetimeIndex and numeric columns.

        Returns:
            self
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self._df_original = df

        # Step 1: Decompose
        if self.verbose:
            print("TVA: Decomposing time series (Norns weaving fate)...")
        self._decomposer = NornDecomposer(self.detector_params)
        self._decomposer.fit(df)
        self._components = self._decomposer.get_components()

        n_series = len(df.columns)

        # Step 2: Build priors
        if self.verbose:
            print("TVA: Building priors (Yggdrasil connecting realms)...")
        self._priors = YggdrasilPriors(
            series_metadata=self.series_metadata,
            relationship_matrix=self.prior_adjacency,
            prior_confidence=self.prior_confidence,
        )

        if self.series_metadata:
            self._prior_adj = self._priors.build_prior_adjacency()
            self._metadata_embeddings = self._priors.build_metadata_embeddings()
            self._anchor_mask = self._priors.get_anchor_mask(self.min_anchor_history)
        else:
            self._prior_adj = None
            self._metadata_embeddings = None
            self._anchor_mask = np.ones(n_series, dtype=bool)

        d_meta = self._metadata_embeddings.shape[1] if self._metadata_embeddings is not None else 0

        # Step 3: Prepare training data
        trend_data = self._components['trend'].values  # (T, N)
        T_total, N = trend_data.shape

        # create sliding windows
        windows, targets = self._create_windows(trend_data)
        if len(windows) == 0:
            raise ValueError(
                f"Not enough data for window_size={self.window_size} + "
                f"forecast_horizon={self.forecast_horizon}. "
                f"Need at least {self.window_size + self.forecast_horizon} periods, "
                f"have {T_total}."
            )

        # prepare other component windows for loss computation
        seasonal_targets = self._create_target_windows(self._components['seasonality'].values)
        holiday_targets = self._create_target_windows(self._components['holidays'].values)
        anomaly_targets = self._create_target_windows(self._components['anomalies'].values)

        # to tensors
        device = torch.device(self.device)
        X = torch.tensor(windows, dtype=torch.float32, device=device)  # (B, N, T_win)
        Y = torch.tensor(targets, dtype=torch.float32, device=device)  # (B, N, T_fc)
        S_sea = torch.tensor(seasonal_targets, dtype=torch.float32, device=device)
        S_hol = torch.tensor(holiday_targets, dtype=torch.float32, device=device)
        S_ano = torch.tensor(anomaly_targets, dtype=torch.float32, device=device)

        anchor_mask_t = torch.tensor(self._anchor_mask, dtype=torch.bool, device=device)

        # metadata tensor
        meta_t = None
        if self._metadata_embeddings is not None and d_meta > 0:
            meta_np = np.tile(self._metadata_embeddings, (len(windows), 1, 1))
            meta_t = torch.tensor(meta_np, dtype=torch.float32, device=device)

        # Step 4: Instantiate network
        if self.verbose:
            print(f"TVA: Building {self.trend_network_type.upper()} trend network...")

        from autots.evaluator.tva.trend_network import (
            CompositeTrendNetworkV1,
            CompositeTrendNetworkV2,
        )

        network_kwargs = dict(
            n_series=N,
            window_size=self.window_size,
            forecast_horizon=self.forecast_horizon,
            d_token=self.d_token,
            n_meso=self.n_meso,
            n_global=self.n_global,
            n_prototypes=self.n_prototypes,
            n_heads=self.n_heads,
            d_meta=d_meta,
            prior_adjacency=self._prior_adj,
        )

        if self.trend_network_type == 'v2':
            self._network = CompositeTrendNetworkV2(**network_kwargs).to(device)
        else:
            self._network = CompositeTrendNetworkV1(**network_kwargs).to(device)

        # Step 5: Instantiate fusion
        from autots.evaluator.tva.fusion import DigitalTwinFusion, AdditiveFusion

        if self.fusion_type == 'attention':
            self._fusion_layer = DigitalTwinFusion().to(device)
        else:
            self._fusion_layer = AdditiveFusion().to(device)

        # Step 6: Instantiate loss
        from autots.evaluator.tva.losses import TemporalLossComposite

        self._loss_fn = TemporalLossComposite(weights=self.loss_weights)

        # Step 7: Training loop
        if self.verbose:
            print("TVA: Training (establishing the Sacred Timeline)...")

        all_params = list(self._network.parameters())
        if hasattr(self._fusion_layer, 'parameters'):
            all_params += list(self._fusion_layer.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=self.lr)

        dataset = TensorDataset(X, Y, S_sea, S_hol, S_ano)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._network.train()
        if isinstance(self._fusion_layer, nn.Module):
            self._fusion_layer.train()

        epoch_iter = tqdm(range(self.epochs), desc="TVA Training") if self.verbose else range(self.epochs)

        for epoch in epoch_iter:
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                x_b, y_b, sea_b, hol_b, ano_b = batch

                # expand metadata for batch
                meta_b = None
                if meta_t is not None:
                    idx = slice(0, x_b.shape[0])
                    meta_b = meta_t[:x_b.shape[0]]

                outputs = self._network(x_b, meta_b, anchor_mask_t)

                targets = {
                    'true_trend': y_b,
                    'seasonal': sea_b,
                    'holidays': hol_b,
                    'anomalies': ano_b,
                }
                if self._prior_adj is not None:
                    targets['prior_adjacency'] = self._prior_adj

                loss, breakdown = self._loss_fn(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if self.verbose >= 2:
                print(f"  Epoch {epoch+1}/{self.epochs} — loss: {avg_loss:.6f}")

        self._network.eval()
        if isinstance(self._fusion_layer, nn.Module):
            self._fusion_layer.eval()

        # Step 8: Setup reconciliation
        if self.reconciliation_method:
            self._reconciler = ReconciliationBridge(method=self.reconciliation_method)

        if self.verbose:
            print("TVA: Training complete.")

        return self

    def predict(self, forecast_length: int = None) -> pd.DataFrame:
        """Generate forecasts for all series.

        Args:
            forecast_length: Number of future periods. Defaults to forecast_horizon.

        Returns:
            Wide DataFrame (forecast_length, N) with forecasted values.
        """
        if self._network is None:
            raise RuntimeError("TVA must be fit before calling predict.")

        if forecast_length is None:
            forecast_length = self.forecast_horizon

        device = torch.device(self.device)

        # get last window of trend data
        trend_data = self._components['trend'].values  # (T, N)
        last_window = trend_data[-self.window_size:]  # (window_size, N)

        # to tensor: (1, N, T_window)
        x = torch.tensor(
            last_window.T[np.newaxis, :, :], dtype=torch.float32, device=device
        )

        # metadata
        meta = None
        if self._metadata_embeddings is not None and self._metadata_embeddings.shape[1] > 0:
            meta = torch.tensor(
                self._metadata_embeddings[np.newaxis, :, :],
                dtype=torch.float32, device=device,
            )

        anchor_mask_t = torch.tensor(self._anchor_mask, dtype=torch.bool, device=device)

        # forward pass
        with torch.no_grad():
            outputs = self._network(x, meta, anchor_mask_t)

        trend_forecast = outputs['trend_forecast'].cpu().numpy()[0]  # (N, T_fc)

        # get forecast components from decomposer
        fc_length = min(forecast_length, self.forecast_horizon)
        forecast_comps = self._decomposer.get_forecast_components(fc_length)

        # prepare component arrays (N, T)
        seasonal = forecast_comps['seasonality'].values.T  # (N, T)
        holidays = forecast_comps['holidays'].values.T
        level_shifts = forecast_comps['level_shifts'].values.T

        # truncate trend forecast if needed
        trend_fc = trend_forecast[:, :fc_length]

        # fuse components
        if isinstance(self._fusion_layer, nn.Module):
            with torch.no_grad():
                t_trend = torch.tensor(trend_fc[np.newaxis], dtype=torch.float32, device=device)
                t_sea = torch.tensor(seasonal[np.newaxis, :, :fc_length], dtype=torch.float32, device=device)
                t_hol = torch.tensor(holidays[np.newaxis, :, :fc_length], dtype=torch.float32, device=device)
                t_ls = torch.tensor(level_shifts[np.newaxis, :, :fc_length], dtype=torch.float32, device=device)
                fused = self._fusion_layer(t_trend, t_sea, t_hol, t_ls)
                forecast_values = fused.cpu().numpy()[0].T  # (T, N)
        else:
            forecast_values = (trend_fc + seasonal[:, :fc_length] +
                               holidays[:, :fc_length] + level_shifts[:, :fc_length]).T

        # build output DataFrame
        future_index = forecast_comps['trend'].index[:fc_length]
        result = pd.DataFrame(
            forecast_values, index=future_index, columns=self._df_original.columns
        )

        return result

    def what_if(self, **constraints) -> pd.DataFrame:
        """Scenario planning via BifrostOptimizer.

        Args:
            **constraints: Passed to BifrostOptimizer methods.
                Supported keys:
                - series_name, timestep, target_value -> apply_constraint
                - series_name, growth_rate -> apply_growth_constraint
                - level_name, target_value -> apply_hierarchical_constraint

        Returns:
            Adjusted forecast DataFrame.
        """
        from autots.evaluator.tva.scenario import BifrostOptimizer

        optimizer = BifrostOptimizer(self)

        if 'growth_rate' in constraints:
            return optimizer.apply_growth_constraint(
                constraints['series_name'], constraints['growth_rate']
            )
        elif 'level_name' in constraints:
            S = self._priors.build_hierarchy_matrix() if self._priors else None
            return optimizer.apply_hierarchical_constraint(
                constraints['level_name'], constraints['target_value'],
                hierarchy_matrix=S,
            )
        else:
            return optimizer.apply_constraint(
                constraints.get('series_name'),
                constraints.get('timestep', 0),
                constraints.get('target_value', 0.0),
            )

    def reconcile(self, forecasts: pd.DataFrame = None) -> pd.DataFrame:
        """Apply hierarchical reconciliation if configured.

        Args:
            forecasts: Forecast DataFrame. If None, generates fresh predictions.

        Returns:
            Reconciled DataFrame.
        """
        if forecasts is None:
            forecasts = self.predict()

        if self._reconciler is None:
            if self.reconciliation_method:
                self._reconciler = ReconciliationBridge(method=self.reconciliation_method)
            else:
                return forecasts

        if self._priors is None:
            return forecasts

        S = self._priors.build_hierarchy_matrix()
        if S.shape[0] == S.shape[1] and np.allclose(S, np.eye(S.shape[0])):
            return forecasts  # no hierarchy defined

        # mint_reconcile requires y_all for ALL hierarchy levels (aggregates + bottom).
        # TVA predicts only bottom-level series, so compute aggregate forecasts
        # by summing bottom series per the S matrix top rows.
        n_bottom = S.shape[1]
        n_all = S.shape[0]
        n_agg = n_all - n_bottom

        if forecasts.shape[1] == n_bottom and n_agg > 0:
            agg_values = (S[:n_agg] @ forecasts.values.T).T  # (T, n_agg)
            agg_cols = [f'_agg_{i}' for i in range(n_agg)]
            full_df = pd.concat([
                pd.DataFrame(agg_values, index=forecasts.index, columns=agg_cols),
                forecasts,
            ], axis=1)
        else:
            full_df = forecasts

        reconciled = self._reconciler.reconcile(full_df, S)
        # Return only the original bottom-level columns
        return reconciled[list(forecasts.columns)]

    def get_composite_trends(self) -> dict:
        """Return learned composite/prototype trends for inspection.

        Returns:
            Dict with 'prototypes' (K, D), 'composite_trend' (n_global, T_forecast),
            and 'prototype_weights' (N, K).
        """
        if self._network is None:
            raise RuntimeError("TVA must be fit first.")

        device = torch.device(self.device)
        trend_data = self._components['trend'].values
        last_window = trend_data[-self.window_size:]

        x = torch.tensor(
            last_window.T[np.newaxis, :, :], dtype=torch.float32, device=device
        )

        meta = None
        if self._metadata_embeddings is not None and self._metadata_embeddings.shape[1] > 0:
            meta = torch.tensor(
                self._metadata_embeddings[np.newaxis, :, :],
                dtype=torch.float32, device=device,
            )

        anchor_mask_t = torch.tensor(self._anchor_mask, dtype=torch.bool, device=device)

        with torch.no_grad():
            outputs = self._network(x, meta, anchor_mask_t)

        result = {
            'prototypes': self._network.prototype._sacred_timeline_prototypes.detach().cpu().numpy(),
            'composite_trend': outputs['composite_trend'].cpu().numpy()[0],
        }
        if 'prototype_weights' in outputs:
            result['prototype_weights'] = outputs['prototype_weights'].cpu().numpy()[0]
        return result

    def get_graph(self) -> np.ndarray:
        """Return the adjacency matrix (learned for V2, prior for V1).

        Returns:
            (M, M) numpy array.
        """
        if self._network is None:
            raise RuntimeError("TVA must be fit first.")

        if hasattr(self._network, 'learned_adjacency'):
            return self._network.learned_adjacency.detach().cpu().numpy()
        elif self._prior_adj is not None:
            return self._prior_adj.copy()
        else:
            n = self.n_global
            return np.ones((n, n), dtype=np.float32)

    # ---- internal helpers ----

    def _create_windows(self, data: np.ndarray) -> tuple:
        """Create sliding windows and targets from (T, N) trend data.

        Returns:
            windows: (n_windows, N, window_size) — transposed for network input.
            targets: (n_windows, N, forecast_horizon)
        """
        T, N = data.shape
        total_len = self.window_size + self.forecast_horizon
        n_windows = max(T - total_len + 1, 0)

        if n_windows == 0:
            return np.array([]), np.array([])

        windows = np.zeros((n_windows, N, self.window_size), dtype=np.float32)
        targets = np.zeros((n_windows, N, self.forecast_horizon), dtype=np.float32)

        for i in range(n_windows):
            windows[i] = data[i : i + self.window_size].T
            targets[i] = data[i + self.window_size : i + total_len].T

        return windows, targets

    def _create_target_windows(self, data: np.ndarray) -> np.ndarray:
        """Create target-only windows (aligned with _create_windows targets).

        Returns:
            (n_windows, N, forecast_horizon)
        """
        T, N = data.shape
        total_len = self.window_size + self.forecast_horizon
        n_windows = max(T - total_len + 1, 0)

        if n_windows == 0:
            return np.array([])

        targets = np.zeros((n_windows, N, self.forecast_horizon), dtype=np.float32)
        for i in range(n_windows):
            targets[i] = data[i + self.window_size : i + total_len].T

        return targets

    def _he_who_remains(self):
        """Hidden: the one who holds the TVA together."""
        return {
            'network_type': self.trend_network_type,
            'n_series': len(self._df_original.columns) if self._df_original is not None else 0,
            'n_anchors': int(self._anchor_mask.sum()) if self._anchor_mask is not None else 0,
            'n_prototypes': self.n_prototypes,
        }
