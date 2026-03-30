# -*- coding: utf-8 -*-
"""
TVA — Time Variant Architecture.

Top-level orchestrator for the TVA forecasting graph. Ties together
decomposition, priors, trend network, fusion, losses, reconciliation,
and scenario planning into a single fit/predict interface.

Note a wrapper exists in /autots/models/tva_model.py for the AutoTS search integration.

Some reference papers:
https://www.mdpi.com/2227-7390/13/20/3288
https://openreview.net/pdf?id=GYSG2vF6z5
https://arxiv.org/html/2409.10996v2
https://arxiv.org/html/2507.15119v2#S4
https://www.researchgate.net/profile/Jawad-Chowdhury-6/publication/379087074_CD-_NOTEAR[…]N0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19
https://openreview.net/pdf?id=80g3Yqlo1a
https://openreview.net/pdf?id=WjDjem8mWE
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional

from autots.evaluator.tva.decomposition import NornDecomposer
from autots.evaluator.tva.priors import YggdrasilPriors, SeriesMetadata
from autots.evaluator.tva.reconciliation import ReconciliationBridge
from autots.evaluator.tva.structure import (
    StructureLearningConfig,
    build_graph_snapshot,
    plot_graph_snapshot,
)

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
    shared composite prototypes. For all Time. Always.

    Args:
        detector_params: Dict passed to TimeSeriesFeatureDetector.
        trend_network: 'v1' (hierarchical latent) or 'v2' (learned directed).
        fusion: How stochastic components (trend, seasonality, holidays) are
            recombined before level shifts are added additively.
            'attention' (default): DigitalTwinFusion — self-attention contextualizes
                component embeddings; sigmoid gates independently fade each component
                in/out (gate in [0,1]) applied to the original values.
            'direct': DirectAttentionFusion — self-attention contextualizes component
                embeddings; attended representations are projected directly to scalar
                contributions summed as a residual over the originals. More purely
                attention-driven; zero-init ensures pure-additive start.
            'additive': AdditiveFusion — plain sum, no learned parameters.
        series_metadata: List of SeriesMetadata for prior construction.
        prior_adjacency: (N, N) explicit prior adjacency matrix (optional).
        prior_confidence: Weight of priors (0=ignore, 1=rigid).
        causal_prior: Optional soft causal edge prior for V2 adjacency regularization.
        prior_construction_config: Dict configuring automatic structural prior
            construction from event clusters and metadata. Defaults to blending
            detected changepoints/anomalies with metadata similarity
            ({'sources': ['event', 'metadata'], ...}). Pass {} to disable.
        causal_prior_construction_config: Dict configuring automatic Granger-causal
            prior construction from decomposed trend components (requires
            statsmodels). Defaults to {'max_lag': 3, 'min_history': 90,
            'top_k': 8, 'alpha': 0.05, 'difference': True}. Pass {} to disable.
        d_token: Token/latent dimension.
        n_meso: Number of meso latent nodes, or 'auto' (default) to set as
            2 * n_global derived from N series.
        n_global: Number of global latent nodes, or 'auto' (default) to set as
            max(2, ceil(sqrt(N_anchors))). Controls the DAG size.
        n_prototypes: Number of prototype trend signatures, or 'auto' (default)
            to set as max(2, round(log2(N_anchors + 1))). Capped at 8.
        n_heads: Attention heads.
        epochs: Training epochs. Default 50 (needed for DAG warmup convergence).
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
        prototype_assignment_method: Prototype assignment method for bottleneck
            ('cosine', 'l2', 'linear'). Defaults to 'cosine'.
        prototype_assignment_temperature: Temperature for prototype assignment logits.
        structure_learning_config: Dict enabling DAG and dynamic hierarchy learning
            in the V2 trend network. Defaults to enabled with conservative penalties
            ({'enabled': True, 'learn_hierarchy': True, 'learn_dag': True, ...}).
            Pass {'enabled': False} to disable.
    """

    def __init__(
        self,
        detector_params: dict = None,
        trend_network: str = 'v2',
        fusion: str = 'attention',
        series_metadata: list = None,
        prior_adjacency: np.ndarray = None,
        prior_confidence: float = 0.3,
        causal_prior: np.ndarray = None,
        prior_construction_config: dict = None,
        causal_prior_construction_config: dict = None,
        d_token: int = 64,
        n_meso: int | str = 'auto',
        n_global: int | str = 'auto',
        n_prototypes: int | str = 'auto',
        n_heads: int = 4,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        window_size: int = 91,
        forecast_horizon: int = 28,
        loss_weights: dict = None,
        reconciliation_method: str = None,
        min_anchor_history: int = 180,
        holiday_country=None,
        holiday_countries: dict = None,
        device: str = 'cpu',
        random_seed: int = 42,
        verbose: int = 1,
        prototype_assignment_method: str = 'cosine',
        prototype_assignment_temperature: float = 1.0,
        structure_learning_config: dict = None,
    ):
        if not HAS_TORCH:
            raise ImportError("TVA requires PyTorch. Install with: pip install torch")

        # Apply defaults for config dicts using sentinel pattern to allow explicit
        # override with {} to disable. These replace None-only defaults.
        if structure_learning_config is None:
            structure_learning_config = {
                'enabled': True,
                'learn_hierarchy': True,
                'learn_dag': True,
                'dag_penalty': 0.1,
                'sparsity_weight': 0.01,
                'assignment_entropy_weight': 0.01,
            }
        if prior_construction_config is None:
            prior_construction_config = {
                'sources': ['event', 'metadata'],
                'source_weights': {'event': 0.7, 'metadata': 0.3},
                'max_distance_days': 7,
            }
        if causal_prior_construction_config is None:
            causal_prior_construction_config = {
                'max_lag': 3,
                'min_history': 90,
                'top_k': 8,
                'alpha': 0.05,
                'difference': True,
            }

        self.detector_params = detector_params
        self.trend_network_type = trend_network
        self.fusion_type = fusion
        self.series_metadata = series_metadata
        self.prior_adjacency = prior_adjacency
        self.prior_confidence = prior_confidence
        self.causal_prior = causal_prior
        self.prior_construction_config = prior_construction_config
        self.causal_prior_construction_config = causal_prior_construction_config
        self.d_token = d_token
        self.n_meso = n_meso
        self.n_global = n_global
        self.n_prototypes = n_prototypes
        # resolved values stored after fit() calls _resolve_network_sizes()
        self._n_meso_fit: Optional[int] = None
        self._n_global_fit: Optional[int] = None
        self._n_prototypes_fit: Optional[int] = None
        self.prototype_assignment_method = prototype_assignment_method
        self.prototype_assignment_temperature = prototype_assignment_temperature
        self._structure_config = StructureLearningConfig.from_dict(
            structure_learning_config
        )
        self.n_heads = n_heads
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.loss_weights = loss_weights
        self.reconciliation_method = reconciliation_method
        self.min_anchor_history = min_anchor_history
        self.holiday_country = holiday_country
        self.holiday_countries = holiday_countries
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
            print("TVA: Decomposing time series...")
        self._decomposer = NornDecomposer(
            self.detector_params,
            holiday_country=self.holiday_country,
            holiday_countries=self.holiday_countries,
        )
        self._decomposer.fit(df)
        self._components = self._decomposer.get_components()
        detected_features = self._decomposer.get_features()

        n_series = len(df.columns)

        # Step 2: Build priors
        if self.verbose:
            print("TVA: Building priors (Connecting domain knowledge)...")
        should_build_priors = any(
            [
                self.series_metadata,
                self.prior_adjacency is not None,
                self.causal_prior is not None,
                self.prior_construction_config,
                self.causal_prior_construction_config,
            ]
        )

        if should_build_priors:
            self._priors = YggdrasilPriors(
                series_metadata=self.series_metadata,
                relationship_matrix=self.prior_adjacency,
                prior_confidence=self.prior_confidence,
                detected_features=detected_features,
                trend_data=self._components['trend'],
                observed_history=df.notna().sum().to_dict(),
                prior_construction_config=self.prior_construction_config,
                causal_prior_construction_config=self.causal_prior_construction_config,
                series_names=list(df.columns),
            )
        else:
            self._priors = None

        if self._priors is not None:
            self._prior_adj = self._priors.build_structural_prior_adjacency()
            if self.series_metadata:
                self._metadata_embeddings = self._priors.build_metadata_embeddings()
                self._anchor_mask = self._priors.get_anchor_mask(
                    self.min_anchor_history
                )
            else:
                self._metadata_embeddings = None
                self._anchor_mask = np.ones(n_series, dtype=bool)
        else:
            self._prior_adj = None
            self._metadata_embeddings = None
            self._anchor_mask = np.ones(n_series, dtype=bool)

        if self._anchor_mask is not None and not np.any(self._anchor_mask):
            warnings.warn(
                "TVA found no anchor series under the current history threshold. "
                "Falling back to treating all series as anchors for this fit.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._anchor_mask = np.ones(n_series, dtype=bool)

        d_meta = (
            self._metadata_embeddings.shape[1]
            if self._metadata_embeddings is not None
            else 0
        )

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
        seasonal_targets = self._create_target_windows(
            self._components['seasonality'].values
        )
        holiday_targets = self._create_target_windows(
            self._components['holidays'].values
        )
        level_shift_targets = self._create_target_windows(
            self._components['level_shifts'].values
        )
        anomaly_targets = self._create_target_windows(
            self._components['anomalies'].values
        )

        # to tensors
        device = torch.device(self.device)
        X = torch.tensor(windows, dtype=torch.float32, device=device)  # (B, N, T_win)
        Y = torch.tensor(targets, dtype=torch.float32, device=device)  # (B, N, T_fc)
        S_sea = torch.tensor(seasonal_targets, dtype=torch.float32, device=device)
        S_hol = torch.tensor(holiday_targets, dtype=torch.float32, device=device)
        S_lvl = torch.tensor(level_shift_targets, dtype=torch.float32, device=device)
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

        n_anchors = int(self._anchor_mask.sum())
        self._n_global_fit, self._n_meso_fit, self._n_prototypes_fit = (
            self._resolve_network_sizes(
                n_anchors, self.n_global, self.n_meso, self.n_prototypes
            )
        )
        if self.verbose >= 2:
            print(
                f"TVA: network sizes — n_global={self._n_global_fit}, "
                f"n_meso={self._n_meso_fit}, n_prototypes={self._n_prototypes_fit}"
            )

        network_kwargs = dict(
            n_series=N,
            window_size=self.window_size,
            forecast_horizon=self.forecast_horizon,
            d_token=self.d_token,
            n_meso=self._n_meso_fit,
            n_global=self._n_global_fit,
            n_prototypes=self._n_prototypes_fit,
            prototype_assignment_method=self.prototype_assignment_method,
            prototype_assignment_temperature=self.prototype_assignment_temperature,
            n_heads=self.n_heads,
            d_meta=d_meta,
            prior_adjacency=self._prior_adj,
        )

        if self.trend_network_type == 'v2':
            network_kwargs['n_anchor_series'] = int(self._anchor_mask.sum())
            network_kwargs['structure_learning_config'] = (
                self._structure_config.to_dict()
            )
            if self.causal_prior is not None:
                network_kwargs['causal_prior'] = self.causal_prior
            elif self._priors is not None:
                network_kwargs['causal_prior'] = (
                    self._priors.build_causal_prior_adjacency()
                )
            self._network = CompositeTrendNetworkV2(**network_kwargs).to(device)
        else:
            self._network = CompositeTrendNetworkV1(**network_kwargs).to(device)

        # Step 5: Instantiate fusion
        from autots.evaluator.tva.fusion import (
            DigitalTwinFusion,
            DirectAttentionFusion,
            AdditiveFusion,
        )

        if self.fusion_type == 'attention':
            self._fusion_layer = DigitalTwinFusion().to(device)
        elif self.fusion_type == 'direct':
            self._fusion_layer = DirectAttentionFusion().to(device)
        else:
            self._fusion_layer = AdditiveFusion().to(device)

        # Step 6: Instantiate loss
        from autots.evaluator.tva.losses import TemporalLossComposite

        self._loss_fn = TemporalLossComposite(
            weights=self.loss_weights,
            structure_config=self._structure_config,
        )

        # Step 7: Training loop
        if self.verbose:
            print("TVA: Training...")

        all_params = list(self._network.parameters())
        if hasattr(self._fusion_layer, 'parameters'):
            all_params += list(self._fusion_layer.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=self.lr)

        dataset = TensorDataset(X, Y, S_sea, S_hol, S_lvl, S_ano)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._network.train()
        if isinstance(self._fusion_layer, nn.Module):
            self._fusion_layer.train()
        fusion_trainable = isinstance(self._fusion_layer, nn.Module) and any(
            p.requires_grad for p in self._fusion_layer.parameters()
        )
        fusion_forecast_weight = float(
            (self.loss_weights or {}).get('fusion_forecast', 0.25)
        )

        epoch_iter = (
            tqdm(range(self.epochs), desc="TVA Training")
            if self.verbose
            else range(self.epochs)
        )

        for epoch in epoch_iter:
            epoch_loss = 0.0
            n_batches = 0
            structure_loss_scale = self._structure_config.structure_scale(
                epoch_index=epoch,
                total_epochs=self.epochs,
            )
            structure_regularization_enabled = True

            for batch in loader:
                x_b, y_b, sea_b, hol_b, lvl_b, ano_b = batch

                # expand metadata for batch
                meta_b = None
                if meta_t is not None:
                    meta_b = meta_t[: x_b.shape[0]]

                outputs = self._network(x_b, meta_b, anchor_mask_t)

                targets = {
                    'true_trend': y_b,
                    'seasonal': sea_b,
                    'holidays': hol_b,
                    'anomalies': ano_b,
                }
                if self._prior_adj is not None:
                    targets['prior_adjacency'] = self._prior_adj
                if self.prior_confidence is not None:
                    targets['prior_confidence'] = self.prior_confidence
                targets['structure_loss_scale'] = (
                    structure_loss_scale if structure_regularization_enabled else 0.0
                )
                targets['structure_prior_weight'] = (
                    self._structure_config.prior_tether_weight
                )
                targets['structure_config'] = self._structure_config

                loss, breakdown = self._loss_fn(outputs, targets)
                if (
                    self._structure_config.enabled
                    and structure_regularization_enabled
                    and not torch.isfinite(loss)
                ):
                    structure_regularization_enabled = False
                    warnings.warn(
                        "TVA structure regularization produced a non-finite loss. "
                        "Continuing training with structure penalties disabled.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    targets['structure_loss_scale'] = 0.0
                    loss, breakdown = self._loss_fn(outputs, targets)

                if not torch.isfinite(loss):
                    raise RuntimeError("TVA encountered a non-finite training loss.")

                # Train fusion explicitly on reconstructed full signal. Without
                # this term, learned fusion parameters receive no gradients.
                if fusion_trainable and fusion_forecast_weight > 0:
                    # level_shifts are always additive; fusion gates only the
                    # stochastic components (trend, seasonality, holidays).
                    fused_forecast = (
                        self._fusion_layer(
                            outputs['trend_forecast'],
                            sea_b,
                            hol_b,
                        )
                        + lvl_b
                    )
                    full_signal_target = y_b + sea_b + hol_b + lvl_b
                    fusion_loss = fusion_forecast_weight * torch.nn.functional.mse_loss(
                        fused_forecast,
                        full_signal_target,
                    )
                    if not torch.isfinite(fusion_loss):
                        raise RuntimeError(
                            "TVA encountered a non-finite fusion training loss."
                        )
                    breakdown['fusion_forecast'] = fusion_loss.item()
                    loss = loss + fusion_loss

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
        last_window = trend_data[-self.window_size :]  # (window_size, N)

        # to tensor: (1, N, T_window)
        x = torch.tensor(
            last_window.T[np.newaxis, :, :], dtype=torch.float32, device=device
        )

        # metadata
        meta = None
        if (
            self._metadata_embeddings is not None
            and self._metadata_embeddings.shape[1] > 0
        ):
            meta = torch.tensor(
                self._metadata_embeddings[np.newaxis, :, :],
                dtype=torch.float32,
                device=device,
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

        # fuse stochastic components; level_shifts always added additively afterward
        if isinstance(self._fusion_layer, nn.Module):
            with torch.no_grad():
                t_trend = torch.tensor(
                    trend_fc[np.newaxis], dtype=torch.float32, device=device
                )
                t_sea = torch.tensor(
                    seasonal[np.newaxis, :, :fc_length],
                    dtype=torch.float32,
                    device=device,
                )
                t_hol = torch.tensor(
                    holidays[np.newaxis, :, :fc_length],
                    dtype=torch.float32,
                    device=device,
                )
                t_ls = torch.tensor(
                    level_shifts[np.newaxis, :, :fc_length],
                    dtype=torch.float32,
                    device=device,
                )
                fused = self._fusion_layer(t_trend, t_sea, t_hol) + t_ls
                forecast_values = fused.cpu().numpy()[0].T  # (T, N)
        else:
            forecast_values = (
                trend_fc
                + seasonal[:, :fc_length]
                + holidays[:, :fc_length]
                + level_shifts[:, :fc_length]
            ).T

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
            if 'series_name' not in constraints:
                raise ValueError(
                    "what_if() with 'growth_rate' requires 'series_name' in constraints."
                )
            return optimizer.apply_growth_constraint(
                constraints['series_name'], constraints['growth_rate']
            )
        elif 'level_name' in constraints:
            S = self._priors.build_hierarchy_matrix() if self._priors else None
            return optimizer.apply_hierarchical_adjustment(
                constraints['level_name'],
                constraints['target_value'],
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
                self._reconciler = ReconciliationBridge(
                    method=self.reconciliation_method
                )
            else:
                return forecasts

        if self._priors is None:
            warnings.warn(
                "reconcile() called but no priors/hierarchy matrix is available. "
                "Returning unreconciled forecasts.",
                UserWarning,
                stacklevel=2,
            )
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
            full_df = pd.concat(
                [
                    pd.DataFrame(agg_values, index=forecasts.index, columns=agg_cols),
                    forecasts,
                ],
                axis=1,
            )
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
        outputs = self._get_last_window_outputs()

        result = {
            'prototypes': self._network.prototype._sacred_timeline_prototypes.detach()
            .cpu()
            .numpy(),
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
            n = (
                self._n_global_fit
                if self._n_global_fit is not None
                else max(2, int(self.n_global) if self.n_global != 'auto' else 4)
            )
            return np.ones((n, n), dtype=np.float32)

    def get_graph_snapshot(
        self,
        threshold: float = None,
        include_priors: bool = True,
    ) -> dict:
        """Return a serializable snapshot of the learned graph and hierarchy."""
        if self._network is None:
            raise RuntimeError("TVA must be fit first.")

        outputs = self._get_last_window_outputs()
        assignment_matrices = [
            matrix.detach().cpu().numpy()
            for matrix in outputs.get('assignment_matrices', [])
        ]
        anchor_names = list(np.asarray(self._df_original.columns)[self._anchor_mask])
        snapshot = build_graph_snapshot(
            adjacency_dense=self.get_graph(),
            assignment_matrices=assignment_matrices,
            threshold=(
                self._structure_config.threshold_for_export
                if threshold is None
                else float(threshold)
            ),
            prior_adjacency=self._prior_adj if include_priors else None,
            anchor_names=anchor_names,
        )
        return snapshot.to_dict()

    def plot_graph(
        self,
        view: str = 'dag',
        threshold: float = None,
        max_edges: int = 50,
        show_priors: bool = False,
        ax=None,
    ):
        """Plot the learned graph, hierarchy, or adjacency heatmap."""
        snapshot_dict = self.get_graph_snapshot(
            threshold=threshold,
            include_priors=show_priors,
        )
        snapshot = build_graph_snapshot(
            adjacency_dense=snapshot_dict['adjacency_dense'],
            assignment_matrices=snapshot_dict['assignment_matrices'],
            threshold=(
                threshold
                if threshold is not None
                else self._structure_config.threshold_for_export
            ),
            prior_adjacency=snapshot_dict.get('prior_adjacency'),
            anchor_names=[
                node['node_id']
                for node in snapshot_dict['node_table']
                if node.get('level') == 0
            ],
        )
        return plot_graph_snapshot(
            snapshot=snapshot,
            view=view,
            max_edges=max_edges,
            show_priors=show_priors,
            ax=ax,
        )

    # ---- internal helpers ----

    @staticmethod
    def _resolve_network_sizes(
        n_anchors: int,
        n_global,
        n_meso,
        n_prototypes,
    ) -> tuple:
        """Resolve 'auto' sentinels to concrete integers based on anchor count.

        n_global  = max(2, ceil(sqrt(n_anchors)))
        n_meso    = max(n_global, 2 * n_global)  — interpolates between global and series
        n_prototypes = max(2, round(log2(n_anchors + 1))), capped at 8

        Integer values are passed through unchanged with a floor of 2.
        """
        import math

        n = max(int(n_anchors), 1)

        resolved_global = (
            max(2, int(np.ceil(np.sqrt(n))))
            if n_global == 'auto'
            else max(2, int(n_global))
        )
        resolved_meso = (
            max(resolved_global, 2 * resolved_global)
            if n_meso == 'auto'
            else max(2, int(n_meso))
        )
        resolved_prototypes = (
            min(8, max(2, round(math.log2(n + 1))))
            if n_prototypes == 'auto'
            else max(2, int(n_prototypes))
        )
        return resolved_global, resolved_meso, resolved_prototypes

    def _get_last_window_outputs(self) -> dict:
        if self._network is None:
            raise RuntimeError("TVA must be fit first.")

        device = torch.device(self.device)
        trend_data = self._components['trend'].values
        last_window = trend_data[-self.window_size :]
        x = torch.tensor(
            last_window.T[np.newaxis, :, :], dtype=torch.float32, device=device
        )

        meta = None
        if (
            self._metadata_embeddings is not None
            and self._metadata_embeddings.shape[1] > 0
        ):
            meta = torch.tensor(
                self._metadata_embeddings[np.newaxis, :, :],
                dtype=torch.float32,
                device=device,
            )

        anchor_mask_t = torch.tensor(self._anchor_mask, dtype=torch.bool, device=device)
        with torch.no_grad():
            return self._network(x, meta, anchor_mask_t)

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

    def _get_metadata(self):
        return {
            'network_type': self.trend_network_type,
            'n_series': (
                len(self._df_original.columns) if self._df_original is not None else 0
            ),
            'n_anchors': (
                int(self._anchor_mask.sum()) if self._anchor_mask is not None else 0
            ),
            'n_prototypes': (
                self._n_prototypes_fit
                if self._n_prototypes_fit is not None
                else self.n_prototypes
            ),
            'prototype_assignment_method': self.prototype_assignment_method,
        }
