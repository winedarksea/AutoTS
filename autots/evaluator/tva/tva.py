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
from autots.evaluator.tva.priors import (
    YggdrasilPriors,
    SeriesMetadata,
    coerce_prior_adjacency,
)
from autots.evaluator.tva.reconciliation import ReconciliationBridge
from autots.evaluator.tva.structure import (
    GraphSnapshot,
    StructureLearningConfig,
    build_graph_snapshot,
    plot_graph_snapshot,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


# What ``reconciliation_covariance='auto'`` resolves to. Before this, the
# factor and torch-free modes reached MinT with no residual matrix at all, so W
# fell back to the identity and "MinT" was plain OLS reconciliation.
#
# Set to 'structural' on the strength of examples/tva_reconciliation_gate.py:
# over 10 seeds of a factor-grouped synthetic hierarchy, the structural arm
# scored an aggregate MASE ratio of 0.846 against the identity arm (winning on
# 8 of 10 seeds, worst seed 1.043) at identical, exact coherence. A
# variance-only control (the same W with its off-diagonals deleted) scored
# 1.409 against structural's 1.385, so most of the gain is ordinary variance
# weighting and roughly 1.7% is the cross-series covariance itself.
#
# Note what this does *not* change: when ``reconcile`` synthesizes the
# aggregate rows by summing the bottom level -- what every in-library caller
# does -- the input already lies in the coherent subspace MinT projects onto,
# so no W moves any number and the structural W is not even assembled. This
# setting bites only for a caller supplying independently-produced
# aggregate-level forecasts.
RECONCILIATION_COVARIANCE_AUTO = 'structural'


class TVA:
    """Time Variant Architecture — end-to-end coherent forecasting graph.

    The Common Operating Picture for time series. Produces structurally
    consistent forecasts across related series by routing all trends through
    shared composite prototypes. For all Time. Always.

    Args:
        detector_params: Dict passed to TimeSeriesFeatureDetector.
        trend_network: 'v2' (learned directed, default), 'v1' (hierarchical
            latent), 'factor' (learned latent-factor trend), or 'none'
            with a damped rolling-trend extrapolation while still running
            factor/edge discovery, MinT reconciliation, and residual-sigma
            intervals. 'none' is the Phase-4 kill-rule configuration: fast,
            interpretable, and a strong baseline the network must beat.
        fusion: How stochastic components (trend, seasonality, holidays) are
            recombined before level shifts are added additively.

            - 'attention' (default): DigitalTwinFusion — self-attention contextualizes
              component embeddings; sigmoid gates independently fade each component
              in/out (gate in [0,1]) applied to the original values.
            - 'direct': DirectAttentionFusion — self-attention contextualizes component
              embeddings; attended representations are projected directly to scalar
              contributions summed as a residual over the originals. More purely
              attention-driven; zero-init ensures pure-additive start.
            - 'additive': AdditiveFusion — plain sum, no learned parameters.

        series_metadata: List of SeriesMetadata for prior construction.
        prior_adjacency: Optional co-movement graph prior. Accepts an (N, N)
            matrix, a labelled DataFrame, a list of
            ``{'source', 'target', 'weight', 'directed'}`` edge dicts, or a
            list of name groups (each group a clique) -- see
            ``priors.coerce_prior_adjacency``. Signed: a negative weight is a
            substitution claim and pushes the pair apart.
        prior_confidence: Weight of priors (0=ignore, 1=rigid). Sets the
            merge weight of the 'business'/'causal' edge families (against
            1.0 for the data-derived ones) and the coherence blend weight.
        causal_prior: Optional directed prior in the same forms, used for V2
            adjacency regularization and kept as its own edge family.
        prior_construction_config: Dict configuring automatic structural prior
            construction from event clusters and metadata. Defaults to blending
            detected changepoints/anomalies with metadata similarity
            ({'sources': ['event', 'metadata'], ...}). Pass {} to disable.
        causal_prior_construction_config: Deprecated and ignored — series-level
            causal structure now comes from factor-residual discovery
            (autots.evaluator.tva.discovery). Use discovery_config instead.
        discovery_config: Dict overriding discovery.DEFAULT_DISCOVERY_CONFIG
            (factor extraction, conditional screening, stability selection,
            lead-lag scan, falsification). Pass {'enabled': False} to skip
            structure discovery entirely.
        d_token: Token/latent dimension.
        n_meso: Number of meso latent nodes, or 'auto' (default) to set as
            2 * n_global derived from N series.
        n_global: Number of global latent nodes, or 'auto' (default) to set as
            max(2, ceil(sqrt(N_anchors))). Controls the DAG size.
        n_prototypes: Number of prototype trend signatures, or 'auto' (default)
            to set as max(2, round(log2(N_anchors + 1))). Capped at 8.
        n_heads: Attention heads.
        epochs: Maximum training epochs. Default 200; training normally stops
            earlier via time-ordered validation early stopping (patience 8).
        lr: Learning rate.
        batch_size: Training batch size.
        window_size: Input trend window length.
        forecast_horizon: Output forecast length.
        loss_weights: Dict overriding loss component weights.
        reconciliation_method: None, 'mint', 'erm', etc.
        reconciliation_covariance: how MinT's W is obtained when no per-node
            residual matrix is available (the 'factor' and 'none' modes, where
            the network-based residual estimator returns None and W therefore
            falls back to the identity). 'auto' (default) follows the module
            constant ``RECONCILIATION_COVARIANCE_AUTO``, 'structural' uses
            ``S @ forecast_covariance() @ S.T``, 'identity' forces the
            previous behaviour.
        min_anchor_history: Minimum periods for a series to be an anchor.
        device: 'cpu', 'cuda', or None (auto-detects cuda if available, else cpu).
        random_seed: Reproducibility seed.
        verbose: 0=silent, 1=progress bar, 2=per-epoch loss.
        prototype_assignment_method: Prototype assignment method for bottleneck
            ('cosine', 'l2', 'linear'). Defaults to 'cosine'.
        prototype_assignment_temperature: Temperature for prototype assignment logits.
        n_factors: Number of latent factors for trend_network='factor', or
            'auto' (default) to pick a rank from the smoothed level-panel
            singular spectrum.
        factor_knot_spacing: Candidate-knot spacing (steps) of the factor
            trend-filter basis in 'factor' mode.
        factor_max_lag: Maximum learned per-series response lag in 'factor'
            mode. Defaults to 0 (contemporaneous)
        factor_config: Dict overriding factor_network.DEFAULT_FACTOR_CONFIG.
        coherence_config: Post-forecast coherence-shrink settings. Merged
            over ``factor_config['coherence_config']``; supplying it also
            enables the shrink unless ``factor_config`` set 'coherence'
            explicitly. None (default) leaves the shrink off.
        derived_definitions: Declared ratio identities {column: (num, den)};
            never inferred from column names.
        factor_deconfound_edges: Re-run series edge discovery deconfounded
            against the learned factors ('factor' mode). Off by default
        structure_learning_config: Dict enabling DAG and dynamic hierarchy learning
            in the V2 trend network.
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
        discovery_config: dict = None,
        d_token: int = 32,
        n_meso: int | str = 'auto',
        n_global: int | str = 'auto',
        n_prototypes: int | str = 'auto',
        n_heads: int = 2,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        window_size: int = 91,
        forecast_horizon: int = 28,
        recency_halflife_days: Optional[float] = None,
        loss_weights: dict = None,
        reconciliation_method: str = None,
        min_anchor_history: int = 180,
        holiday_country=None,
        holiday_countries: dict = None,
        device: str = None,
        random_seed: int = 42,
        verbose: int = 1,
        prototype_assignment_method: str = 'cosine',
        prototype_assignment_temperature: float = 1.0,
        structure_learning_config: dict = None,
        n_factors: int | str = 'auto',
        factor_knot_spacing: int = 7,
        factor_max_lag: int = 0,
        factor_config: dict = None,
        coherence_config: dict = None,
        derived_definitions: dict = None,
        factor_deconfound_edges: bool = False,
        reconciliation_covariance: str = 'auto',
    ):
        if not HAS_TORCH and str(trend_network) != 'none':
            raise ImportError(
                "TVA requires PyTorch for trend_network 'v1'/'v2'/'factor'. "
                "Install with: "
                "pip install torch — or use trend_network='none' for the "
                "torch-free damped-trend + factor-discovery mode."
            )

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
        self.detector_params = detector_params
        self.trend_network_type = trend_network
        self.fusion_type = fusion
        self.series_metadata = series_metadata
        self.prior_adjacency = prior_adjacency
        self.prior_confidence = prior_confidence
        self.causal_prior = causal_prior
        self.prior_construction_config = prior_construction_config
        self.causal_prior_construction_config = causal_prior_construction_config
        self.discovery_config = discovery_config
        self.n_factors = n_factors
        self.factor_knot_spacing = factor_knot_spacing
        self.factor_max_lag = factor_max_lag
        self.factor_config = factor_config
        self.coherence_config = coherence_config
        self.derived_definitions = derived_definitions
        self.factor_deconfound_edges = factor_deconfound_edges
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
        self.recency_halflife_days = recency_halflife_days
        self.loss_weights = loss_weights
        self.reconciliation_method = reconciliation_method
        self.reconciliation_covariance = reconciliation_covariance
        self.min_anchor_history = min_anchor_history
        self.holiday_country = holiday_country
        self.holiday_countries = holiday_countries
        if device is not None:
            self.device = device
        elif HAS_TORCH:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            # torch-free mode (trend_network='none') never moves tensors
            self.device = 'cpu'
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
        self._trend_scale = None
        self._sigma_calibration = None
        self._last_sigma = None
        self._discovery = None
        self._reconciliation_method_effective = None
        self._factor_network = None
        self._factor_info = None
        self._structure_loadings = None
        self._shift_returned = None
        self._structure_agreement = None
        self._factor_scale = None
        self._prior_input = None
        self._causal_prior_input = None

    def fit(self, df: pd.DataFrame) -> 'TVA':
        """Full TVA pipeline: decompose, build priors, train network.

        Args:
            df: Wide DataFrame with DatetimeIndex and numeric columns.

        Returns:
            self
        """
        if HAS_TORCH:
            torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self._df_original = df

        # Step 0: normalize whatever prior form the user supplied into one
        # signed (N, N) matrix over this panel's column order. Every
        # downstream consumer reads these, not the raw attributes.
        self._prior_input = coerce_prior_adjacency(
            self.prior_adjacency, list(df.columns)
        )
        self._causal_prior_input = coerce_prior_adjacency(
            self.causal_prior, list(df.columns)
        )
        if (
            self._prior_input is not None or self._causal_prior_input is not None
        ) and str(self.trend_network_type).lower() == 'v1':
            warnings.warn(
                "TVA trend_network='v1' ignores graph priors entirely: the "
                "(N, N) prior fails v1's attention-mask shape check and v1 "
                "emits no adjacency for the soft-prior loss. Use "
                "trend_network='factor' or 'v2' for the prior to have effect.",
                RuntimeWarning,
                stacklevel=2,
            )

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
                self._prior_input is not None,
                self._causal_prior_input is not None,
                self.prior_construction_config,
                self.causal_prior_construction_config,
            ]
        )

        if should_build_priors:
            self._priors = YggdrasilPriors(
                series_metadata=self.series_metadata,
                relationship_matrix=self._prior_input,
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

        # per-series scale: robust std of trend first differences, floored so
        # near-constant trends do not explode when normalized (P1-1)
        self._trend_scale = self._compute_trend_scale(trend_data)

        if self.trend_network_type == 'factor':
            self._run_structure_discovery()
            self._fit_factor_network(df)
            self._setup_reconciliation()
            return self

        # create sliding windows (window-local delta targets, per-series scaled)
        windows, targets = self._create_windows(trend_data)
        if len(windows) == 0:
            raise ValueError(
                f"Not enough data for window_size={self.window_size} + "
                f"forecast_horizon={self.forecast_horizon}. "
                f"Need at least {self.window_size + self.forecast_horizon} periods, "
                f"have {T_total}."
            )

        # Step 3.5: Structure discovery — factors first, conditional edges
        # second (Phase 2). Torch-free; the network only learns small deltas
        # on top of this data-anchored structure.
        self._run_structure_discovery()

        # Arm A mode (Phase 4 kill rule): no torch network at all — detector
        # decomposition + damped rolling trend + discovery for explainability
        # + MinT + residual-sigma intervals. Fast, interpretable, torch-free.
        if self.trend_network_type == 'none':
            self._network = None
            self._fusion_layer = None
            self._loss_fn = None
            self._setup_reconciliation()
            if self.verbose:
                print("TVA: torch-free mode ('none') — no network training.")
            return self

        # Torch-only from here: build the training tensors. Kept below the
        # torch-free early return so trend_network='none' never touches torch.
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

        device = torch.device(self.device)
        X = torch.tensor(windows, dtype=torch.float32, device=device)  # (B, N, T_win)
        Y = torch.tensor(targets, dtype=torch.float32, device=device)  # (B, N, T_fc)
        S_sea = torch.tensor(seasonal_targets, dtype=torch.float32, device=device)
        S_hol = torch.tensor(holiday_targets, dtype=torch.float32, device=device)
        S_lvl = torch.tensor(level_shift_targets, dtype=torch.float32, device=device)
        S_ano = torch.tensor(anomaly_targets, dtype=torch.float32, device=device)

        anchor_mask_t = torch.tensor(self._anchor_mask, dtype=torch.bool, device=device)

        # metadata tensor — one copy, expanded per batch (identical per window)
        meta_t = None
        if self._metadata_embeddings is not None and d_meta > 0:
            meta_t = torch.tensor(
                self._metadata_embeddings[np.newaxis, :, :],
                dtype=torch.float32,
                device=device,
            )

        # Step 4: Instantiate network
        if self.verbose:
            print(f"TVA: Building {self.trend_network_type.upper()} trend network...")

        from autots.evaluator.tva.trend_network import (
            CompositeTrendNetworkV1,
            CompositeTrendNetworkV2,
        )

        n_anchors = int(self._anchor_mask.sum())
        # n_global = r: the latent width follows the number of discovered
        # factors instead of the arbitrary ceil(sqrt(n_anchors)) (D-2)
        n_global_resolved = self.n_global
        if (
            n_global_resolved == 'auto'
            and self._discovery is not None
            and self._discovery['loadings'].shape[1] > 0
        ):
            n_global_resolved = max(2, int(self._discovery['loadings'].shape[1]))
        self._n_global_fit, self._n_meso_fit, self._n_prototypes_fit = (
            self._resolve_network_sizes(
                n_anchors, n_global_resolved, self.n_meso, self.n_prototypes
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
            if self._causal_prior_input is not None:
                network_kwargs['causal_prior'] = self._causal_prior_input
            network_kwargs['discovery'] = self._build_network_discovery_inputs()
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
        optimizer = torch.optim.AdamW(all_params, lr=self.lr, weight_decay=0.05)

        # Time-ordered validation split (P1-5). Windows overlap by W-1, so a
        # random split would leak the target; the holdout must be the LAST
        # windows by time. Validation monitors forecast loss only.
        n_windows_total = X.shape[0]
        n_val = 0
        if n_windows_total >= 12:
            n_val = int(
                min(
                    max(2 * self.forecast_horizon, np.ceil(0.1 * n_windows_total)),
                    n_windows_total // 3,
                )
            )
        n_train = n_windows_total - n_val
        X_val, Y_val = X[n_train:], Y[n_train:]
        X_tr = X[:n_train]

        dataset = TensorDataset(
            X_tr,
            Y[:n_train],
            S_sea[:n_train],
            S_hol[:n_train],
            S_lvl[:n_train],
            S_ano[:n_train],
        )
        # sample weights are computed AFTER the split so the sampler indices
        # align with the training subset rather than the full window set
        sample_weights = self._compute_window_sample_weights(
            self._components['trend'].index
        )
        if sample_weights is not None:
            sample_weights = sample_weights[:n_train]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
            )
        else:
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

        # early stopping state (P1-5): monitor validation forecast loss only —
        # the penalty terms are regularizers, not the objective
        patience = 8
        best_val = float('inf')
        epochs_since_best = 0
        best_network_state = None
        best_fusion_state = None

        def _validation_forecast_loss() -> float:
            if n_val == 0:
                return float('nan')
            self._network.eval()
            total_val = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for start in range(0, n_val, self.batch_size):
                    x_v = X_val[start : start + self.batch_size]
                    y_v = Y_val[start : start + self.batch_size]
                    meta_v = None
                    if meta_t is not None:
                        meta_v = meta_t.expand(x_v.shape[0], -1, -1)
                    out_v = self._network(x_v, meta_v, anchor_mask_t)
                    total_val += self._loss_fn.forecast_loss(
                        out_v['trend_forecast'], y_v
                    ).item()
                    n_val_batches += 1
            self._network.train()
            return total_val / max(n_val_batches, 1)

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

                # expand metadata for batch (identical embedding per window)
                meta_b = None
                if meta_t is not None:
                    meta_b = meta_t.expand(x_b.shape[0], -1, -1)

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

            val_loss = _validation_forecast_loss()
            if self.verbose >= 2:
                print(
                    f"  Epoch {epoch+1}/{self.epochs} — loss: {avg_loss:.6f}"
                    + (f" val_forecast: {val_loss:.6f}" if n_val else "")
                )
            if n_val:
                if val_loss < best_val - 1e-8:
                    best_val = val_loss
                    epochs_since_best = 0
                    best_network_state = {
                        k: v.detach().clone()
                        for k, v in self._network.state_dict().items()
                    }
                    if isinstance(self._fusion_layer, nn.Module):
                        best_fusion_state = {
                            k: v.detach().clone()
                            for k, v in self._fusion_layer.state_dict().items()
                        }
                else:
                    epochs_since_best += 1
                    if epochs_since_best >= patience:
                        if self.verbose >= 2:
                            print(
                                f"  Early stopping at epoch {epoch+1} "
                                f"(best val {best_val:.6f})"
                            )
                        break

        # restore best weights found on the validation holdout
        if best_network_state is not None:
            self._network.load_state_dict(best_network_state)
        if best_fusion_state is not None and isinstance(self._fusion_layer, nn.Module):
            self._fusion_layer.load_state_dict(best_fusion_state)

        self._network.eval()
        if isinstance(self._fusion_layer, nn.Module):
            self._fusion_layer.eval()

        # store validation residual statistics for sigma calibration (P3-2)
        self._calibrate_sigma(X_val, Y_val, meta_t, anchor_mask_t, n_val)

        # Step 8: Setup reconciliation
        self._setup_reconciliation()

        if self.verbose:
            print("TVA: Training complete.")

        return self

    def _build_adjusted_panel(self, smooth_window: int = 365) -> np.ndarray:
        """Level-space panel with the detector's *validated* components removed.

        Args:
            smooth_window: window of the low-frequency part retained from the
                high-frequency components. Defaults to a full year (clamped to
                T // 3) so a yearly seasonal cycle averages to ~zero and is
                still removed, while multi-year drift is kept.

        Returns:
            (T, N) float array aligned with the training index and columns.
        """
        adjusted = self._df_original.ffill().bfill().astype(float)
        window = max(int(min(smooth_window, max(len(adjusted) // 3, 1))), 1)
        for key in ('seasonality', 'holidays', 'anomalies'):
            comp = self._components.get(key)
            if comp is None:
                continue
            comp = comp.reindex(index=adjusted.index, columns=adjusted.columns).fillna(
                0.0
            )
            slow = comp.rolling(window, center=True, min_periods=1).mean()
            adjusted = adjusted - (comp - slow)
        shifts = self._components.get('level_shifts')
        if shifts is not None:
            shifts = shifts.reindex(
                index=adjusted.index, columns=adjusted.columns
            ).fillna(0.0)
            cfg = self.factor_config or {}
            if cfg.get('level_shift_veto'):
                vetoed = self._veto_shared_shifts(shifts, cfg)
                # Whatever the veto keeps in the panel is now part of the
                # fitted trend, so the reconstruction must NOT add it back a
                # second time. Record it once here and correct every add-back
                # site against it; the two paths have to stay symmetric or the
                # forecast double-counts every shared step.
                self._shift_returned = (shifts - vetoed).to_numpy(dtype=float)
                shifts = vetoed
            else:
                self._shift_returned = None
            adjusted = adjusted - shifts
        return adjusted.to_numpy(dtype=float)

    @staticmethod
    def _veto_shared_shifts(shifts, cfg):
        """Keep the part of a level shift that the whole panel made together.

        The changepoint detector is univariate: it sees each series alone, so a
        single shared factor move gets charged to N series as N independent
        level shifts and is then subtracted out of the panel entirely. That is
        the largest measured loss in the factor pipeline -- removing
        ``level_shifts`` more than halves the variance the true factors explain
        (R^2 0.338 -> 0.156), because the movement being removed *is* the
        signal the factor model exists to find.

        This keeps the per-series *excess* over what the co-stepping group did
        and returns the shared part to the panel. A step that only one series
        takes is untouched, so genuine idiosyncratic shifts are still removed.

        Args:
            shifts: (T, N) frame of per-series step functions.
            cfg: factor config, read for ``level_shift_min_share``,
                ``level_shift_min_agreement`` and ``level_shift_window``.

        Returns:
            A (T, N) frame to subtract instead of ``shifts``.
        """
        import pandas as pd

        step = shifts.diff().fillna(0.0)
        values = step.to_numpy(dtype=float)
        n_time, n_series = values.shape
        if n_series < 3 or n_time < 3:
            return shifts

        window = max(int(cfg.get('level_shift_window', 31) or 31), 1)
        min_share = float(cfg.get('level_shift_min_share', 0.15) or 0.15)
        min_agree = float(cfg.get('level_shift_min_agreement', 0.60) or 0.60)
        shrink = float(cfg.get('level_shift_veto_shrink', 1.0))
        if str(cfg.get('level_shift_veto')).lower() == 'all':
            # the unconditional control: subtract no level shift at all. This
            # is what isolates how much the subtraction costs the factor fit
            # from how much of it is genuinely shared movement.
            return shifts * (1.0 - shrink)

        moved = np.abs(values) > 0
        # a shared move is rarely dated identically across series, so pool the
        # vote over a short window before deciding whether it was one event
        pooled = (
            pd.DataFrame(moved.astype(float))
            .rolling(window, center=True, min_periods=1)
            .max()
            .to_numpy()
            > 0
        )
        keep = np.zeros_like(values)
        for t in range(n_time):
            members = np.where(pooled[t] & moved[t])[0]
            if members.size < 2 or members.size < min_share * n_series:
                continue
            signs = np.sign(values[t, members])
            nz = signs[signs != 0]
            if nz.size == 0:
                continue
            if abs(float(nz.sum())) / float(nz.size) < min_agree:
                continue  # the panel disagrees on direction: not one event
            # Return the step to the panel rather than trying to split it into
            # shared and idiosyncratic parts: a series' share of a factor move
            # is proportional to its loading, which is exactly what has not
            # been estimated yet at this point in the pipeline. Whatever excess
            # a series had over the group stays in the panel too, where the
            # factor model reads it as idiosyncratic -- which is what it is.
            keep[t, members] = shrink * values[t, members]
        # subtract only the shared part from the original panel. Rebuilding by
        # cumsum of the differences instead would silently drop shifts.iloc[0]
        # -- turning the whole veto into a per-series constant, which the
        # median centering in robust_level_scale then removes exactly, so the
        # fit would be bit-identical while looking like it had changed.
        vetoed = shifts.to_numpy(dtype=float) - np.cumsum(keep, axis=0)
        return pd.DataFrame(vetoed, index=shifts.index, columns=shifts.columns)

    def _robust_adjusted_panel(self) -> np.ndarray:
        """Trend-isolated panel from the joint robust estimator (2b).

        Uses the ORIGINAL missing-data mask rather than ffill/bfill: a filled
        run is not an observation and must not carry estimation weight.

        Falls back to ``_build_adjusted_panel`` on any failure.
        """
        from autots.evaluator.tva.robust_input import robust_adjusted_panel

        try:
            frame = self._df_original
            values = frame.to_numpy(dtype=float)
            mask = np.isfinite(values)
            components = {
                key: (
                    self._components[key]
                    .reindex(index=frame.index, columns=frame.columns)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                for key in ('seasonality', 'holidays', 'anomalies', 'level_shifts')
                if self._components.get(key) is not None
            }
            cfg = (self.factor_config or {}).get('input_config') or {}
            result = robust_adjusted_panel(
                values,
                mask=mask,
                components=components,
                shrink=cfg.get('shrink'),
                config=cfg.get('config'),
            )
            adjusted = np.asarray(result['adjusted'], dtype=float)
            if adjusted.shape != values.shape or not np.isfinite(adjusted).any():
                raise ValueError('robust input returned an unusable panel')
            self._robust_input_info = result.get('diagnostics')
            return adjusted
        except Exception as exc:  # pragma: no cover - never fail a fit
            warnings.warn(
                f"TVA: robust trend-isolation input failed; falling back to the "
                f"detector-adjusted panel. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._build_adjusted_panel()

    def _factor_input_panel(self) -> np.ndarray:
        """The adjusted panel the factor stage is fit on, per config (2b)."""
        estimator = str(
            (self.factor_config or {}).get('input_estimator', 'detector') or 'detector'
        )
        if estimator == 'robust':
            return self._robust_adjusted_panel()
        return self._build_adjusted_panel()

    def _fit_factor_network(self, df: pd.DataFrame):
        """Fit the level-space latent-factor trend model ('factor' mode)."""
        from autots.evaluator.tva.factor_network import (
            fit_anchor_factor_model,
            robust_level_scale,
            select_n_factors,
        )

        adjusted_raw = self._factor_input_panel()
        space = str((self.factor_config or {}).get('space', 'level') or 'level')
        adjusted = self._to_factor_space(adjusted_raw, space)
        center, scale = robust_level_scale(adjusted)
        normalized = (adjusted - center[np.newaxis, :]) / scale[np.newaxis, :]
        self._factor_scale = {'center': center, 'scale': scale, 'space': space}
        self._adjusted_raw = adjusted_raw

        n_factors = self.n_factors
        if n_factors == 'auto' or n_factors is None:
            n_factors = select_n_factors(normalized, cap=6)
            # a discovered rank is a useful second opinion when it is larger
            if (
                self._discovery is not None
                and self._discovery['loadings'].shape[1] > n_factors
            ):
                n_factors = min(int(self._discovery['loadings'].shape[1]), 6)
        n_factors = int(np.clip(int(n_factors), 1, max(len(df.columns), 1)))

        if self.verbose:
            print(f"TVA: fitting latent-factor trend model (K={n_factors})...")
        cfg = dict(self.factor_config or {})
        cfg.setdefault(
            'min_observed_multiple',
            max(float(self.min_anchor_history) / max(self.forecast_horizon, 1), 1.0),
        )
        # D: make the user's graph prior available to the loading penalty.
        # w_prior_loadings stays at its 0.0 default, so this changes nothing
        # until the user (or the AutoTS search) asks for it.
        if self._prior_input is not None:
            cfg.setdefault('prior_adjacency', self._prior_input)
        observed_mask = self._df_original.notna().to_numpy()
        self._factor_network, self._factor_info = fit_anchor_factor_model(
            normalized,
            n_factors=n_factors,
            knot_spacing=self.factor_knot_spacing,
            max_lag=self.factor_max_lag,
            horizon=self.forecast_horizon,
            config=cfg,
            seed=self.random_seed,
            device=self.device,
            verbose=max(self.verbose - 1, 0),
            mask=observed_mask,
        )
        self._select_factor_space(adjusted_raw, n_factors)
        self._fit_structure_loadings(n_factors)
        self._network = None
        self._fusion_layer = None
        self._loss_fn = None
        if self.factor_deconfound_edges:
            self._rediscover_edges_with_factors()

    def _fit_structure_loadings(self, n_factors: int):
        """C5: a second, structure-only factor fit on the robust input.

        The robust trend-isolation estimator recovers the factor *span* far
        better than the detector-adjusted panel (canonical correlation 0.77 vs
        0.53) but is a measured dead end for the forecast path. The two uses
        are separable: the forecast needs a well-extrapolating trend model, the
        coherence graph needs only a good loading matrix. So this fits a
        torch-free identification on the robust panel purely to build the
        graph, and leaves ``self._factor_network`` — and therefore every
        forecast value — untouched.

        Guarded twice. The result is only trusted when its factors match the
        forecast model's (``match_factors`` mean ``|corr|`` >= 0.4); a graph built
        on a *different* set of factors than the one being shrunk would
        assert relationships the shrink cannot act on coherently. And any
        failure sets the attribute to None, falling back to
        ``fitted_loadings()`` rather than failing the fit.
        """
        self._structure_loadings = None
        requested = str((self.factor_config or {}).get('structure_input') or '').lower()
        if requested != 'robust' or self._factor_network is None:
            return
        try:
            from autots.evaluator.tva.discovery import match_factors
            from autots.evaluator.tva.factor_network import (
                DEFAULT_FACTOR_CONFIG,
                identify_factors,
                robust_level_scale,
            )

            cfg = dict(self._factor_info.get('config') or {})
            space = self._factor_scale.get('space', 'level')
            adjusted = self._to_factor_space(self._robust_adjusted_panel(), space)
            center, scale = robust_level_scale(adjusted)
            normalized = (adjusted - center[np.newaxis, :]) / scale[np.newaxis, :]

            # Share the identification dispatcher rather than re-deriving the
            # estimate+rotate sequence here, which had already drifted from
            # DEFAULT_FACTOR_CONFIG's values -- and which would otherwise leave
            # the structure-only track unable to use the sparse backends.
            merged = dict(DEFAULT_FACTOR_CONFIG)
            merged.update({k: v for k, v in cfg.items() if v is not None})
            merged.setdefault('knot_spacing', self.factor_knot_spacing)
            ident = identify_factors(
                normalized,
                int(n_factors),
                merged,
                seed=self.random_seed,
                device=self.device,
            )

            reference = self._factor_network.fitted_factors(
                cfg.get('prune_share', 0.02)
            )
            agreement = float(
                match_factors(reference, ident['factors'])['mean_abs_corr']
            )
            self._structure_agreement = agreement
            if agreement >= float(cfg.get('structure_min_agreement', 0.4) or 0.4):
                self._structure_loadings = np.asarray(ident['loadings'], dtype=float)
        except Exception as exc:  # pragma: no cover - never fail a fit
            warnings.warn(
                f"TVA structure-only factor fit failed; the coherence graph "
                f"falls back to the forecast model's loadings. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._structure_loadings = None

    def _rediscover_edges_with_factors(self):
        """Re-run edge discovery deconfounded against the *learned* factors.

        Discovery's own factors come from the lag-1 differenced detector
        trend, where slow factors have poor SNR; the learned level-space paths
        are a better confounder estimate. Opt-in: the improved confounder
        estimate is still not good enough to make series-level edges
        trustworthy on its own.
        """
        if self._discovery is None or self._factor_network is None:
            return
        from autots.evaluator.tva.discovery import discover_structure

        prune = self._factor_info['config']['prune_share']
        paths = self._factor_network.fitted_factors(prune)
        discovery_cfg = dict(self.discovery_config or {})
        discovery_cfg.pop('enabled', None)
        discovery_cfg = self._apply_prior_family_weights(discovery_cfg)
        try:
            rediscovered = discover_structure(
                self._components['trend'],
                anchor_mask=None,
                series_metadata=self.series_metadata,
                config=discovery_cfg or None,
                seed=self.random_seed,
                extra_adjacencies=self._build_extra_adjacencies() or None,
                external_factors=paths,
            )
        except Exception as exc:
            warnings.warn(
                "TVA factor-deconfounded edge discovery failed; keeping the "
                f"original edge table. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self._discovery['edges'] = rediscovered['edges']
        self._discovery['adjacency'] = rediscovered['adjacency']
        self._discovery['family_adjacencies'] = rediscovered['family_adjacencies']
        self._discovery['factors'] = paths
        self._discovery['loadings'] = self._factor_network.fitted_loadings(prune)
        self._discovery['factor_names'] = [
            f'factor_{i + 1}' for i in range(paths.shape[1])
        ]
        self._discovery['communities'] = (
            np.abs(self._discovery['loadings']).argmax(axis=1).astype(int)
            if self._discovery['loadings'].size
            else np.full(len(self._df_original.columns), -1, dtype=int)
        )

    def _apply_prior_family_weights(self, discovery_cfg: dict) -> dict:
        """Trust user priors at ``prior_confidence``, not at 1.0.

        Prior edges used to enter the merged adjacency at the same weight as
        the data-derived ``factor``/``lasso`` families while bypassing their
        held-out delta-MSE falsification. ``prior_confidence`` (default 0.3)
        is the user's own statement of how much they trust the guess, so it
        is what the merge should use -- unless they named the weights
        explicitly in ``discovery_config['family_weights']``.
        """
        if self._prior_input is None and self._causal_prior_input is None:
            return discovery_cfg
        confidence = self.prior_confidence
        if confidence is None:
            return discovery_cfg
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return discovery_cfg
        cfg = dict(discovery_cfg or {})
        weights = dict(cfg.get('family_weights') or {})
        for family in ('business', 'causal'):
            weights.setdefault(family, confidence)
        cfg['family_weights'] = weights
        return cfg

    def _build_extra_adjacencies(self) -> dict:
        """Event / metadata / business / causal adjacencies for discovery."""
        extra_adjacencies = {}
        if self._priors is not None:
            structural_config = self._priors._resolve_structural_config()
            if structural_config:
                event_adj = self._priors._build_event_prior_adjacency(structural_config)
                if event_adj is not None:
                    extra_adjacencies['event'] = event_adj
            metadata_adj = self._priors._build_metadata_prior_adjacency()
            if metadata_adj is not None:
                extra_adjacencies['metadata'] = metadata_adj
        # 'business' (co-movement) and 'causal' (directed) stay separate
        # families: merging them destroyed the distinction and let a directed
        # prior masquerade as a symmetric one.
        if self._prior_input is not None:
            extra_adjacencies['business'] = np.asarray(
                self._prior_input, dtype=np.float32
            )
        if self._causal_prior_input is not None:
            extra_adjacencies['causal'] = np.asarray(
                self._causal_prior_input, dtype=np.float32
            )
        return extra_adjacencies

    def _run_structure_discovery(self):
        """Factor and edge discovery (torch-free, Phase 2)."""
        self._discovery = None
        discovery_cfg = dict(self.discovery_config or {})
        discovery_enabled = discovery_cfg.pop('enabled', True)
        if not discovery_enabled:
            return
        if self.verbose:
            print("TVA: Discovering factor and edge structure...")
        from autots.evaluator.tva.discovery import discover_structure

        extra_adjacencies = self._build_extra_adjacencies()
        discovery_cfg = self._apply_prior_family_weights(discovery_cfg)
        try:
            self._discovery = discover_structure(
                self._components['trend'],
                anchor_mask=self._anchor_mask,
                series_metadata=self.series_metadata,
                config=discovery_cfg or None,
                seed=self.random_seed,
                extra_adjacencies=extra_adjacencies or None,
            )
        except Exception as exc:
            warnings.warn(
                f"TVA structure discovery failed; continuing without it. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._discovery = None

    def _setup_reconciliation(self):
        """P3-4: post-hoc MinT gives an auditable guarantee (S·b − a = 0);
        default to it whenever metadata yields a real hierarchy and the user
        did not choose otherwise."""
        self._reconciliation_method_effective = self.reconciliation_method
        if self._reconciliation_method_effective is None and self._priors is not None:
            S = self._priors.build_hierarchy_matrix()
            if S.shape[0] > S.shape[1]:
                self._reconciliation_method_effective = 'mint'
        if self._reconciliation_method_effective:
            self._reconciler = ReconciliationBridge(
                method=self._reconciliation_method_effective
            )

    def predict(self, forecast_length: int = None) -> pd.DataFrame:
        """Generate forecasts for all series.

        Args:
            forecast_length: Number of future periods. Defaults to forecast_horizon.

        Returns:
            Wide DataFrame (forecast_length, N) with forecasted values.
        """
        if self._components is None:
            raise RuntimeError("TVA must be fit before calling predict.")

        if forecast_length is None:
            forecast_length = self.forecast_horizon

        if self.trend_network_type == 'factor':
            return self._predict_factor(int(forecast_length))

        # torch-free mode: damped rolling-trend extrapolation
        if self._network is None:
            return self._predict_numpy(int(forecast_length))

        device = torch.device(self.device)

        # get last window of trend data, normalized to network input space
        trend_data = self._components['trend'].values  # (T, N)
        last_window = trend_data[-self.window_size :]  # (window_size, N)
        x_np, anchor = self._normalized_last_window(last_window)
        scale = self._get_trend_scale()

        x = torch.tensor(x_np, dtype=torch.float32, device=device)

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

        # (N, T_fc) — normalized deltas relative to the last observed trend value
        trend_forecast_norm = outputs['trend_forecast'].cpu().numpy()[0]

        # get forecast components from decomposer
        fc_length = min(forecast_length, self.forecast_horizon)
        forecast_comps = self._decomposer.get_forecast_components(fc_length)

        # prepare component arrays (N, T)
        seasonal = forecast_comps['seasonality'].values.T  # (N, T)
        holidays = forecast_comps['holidays'].values.T
        level_shifts = forecast_comps['level_shifts'].values.T

        # truncate trend forecast if needed
        trend_fc_norm = trend_forecast_norm[:, :fc_length]
        anchor_col = anchor[:, np.newaxis]
        scale_col = scale[:, np.newaxis]
        trend_fc = trend_fc_norm * scale_col + anchor_col

        # fuse stochastic components in the SAME normalized space used during
        # training (trend deltas + scale-normalized components), then invert.
        # Level shifts are always added additively after fusion.
        if isinstance(self._fusion_layer, nn.Module):
            with torch.no_grad():
                t_trend = torch.tensor(
                    trend_fc_norm[np.newaxis], dtype=torch.float32, device=device
                )
                t_sea = torch.tensor(
                    (seasonal[:, :fc_length] / scale_col)[np.newaxis],
                    dtype=torch.float32,
                    device=device,
                )
                t_hol = torch.tensor(
                    (holidays[:, :fc_length] / scale_col)[np.newaxis],
                    dtype=torch.float32,
                    device=device,
                )
                t_ls = torch.tensor(
                    (level_shifts[:, :fc_length] / scale_col)[np.newaxis],
                    dtype=torch.float32,
                    device=device,
                )
                fused_norm = self._fusion_layer(t_trend, t_sea, t_hol) + t_ls
                fused = fused_norm.cpu().numpy()[0] * scale_col + anchor_col
                forecast_values = fused.T  # (T, N)
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

        # Route the trained sigma head to a usable, calibrated raw-scale sigma
        # (P3-1/P3-2): de-normalize, apply per-series validation calibration,
        # then add the decomposition residual sigma in quadrature.
        self._last_sigma = None
        sigma = outputs.get('sigma')
        if sigma is not None:
            sigma_raw = sigma.cpu().numpy()[0][:, :fc_length] * scale_col
            if self._sigma_calibration is not None:
                sigma_raw = sigma_raw * self._sigma_calibration[:, np.newaxis]
            resid_sigma = None
            if hasattr(self._decomposer, 'get_residual_sigma'):
                resid_sigma = self._decomposer.get_residual_sigma()
            if resid_sigma is not None:
                sigma_raw = np.sqrt(
                    sigma_raw**2 + np.asarray(resid_sigma)[:, np.newaxis] ** 2
                )
            self._last_sigma = pd.DataFrame(
                sigma_raw.T, index=future_index, columns=self._df_original.columns
            )

        # coherence via reconciliation is primary (P3-4): when a hierarchy is
        # configured (or defaulted from metadata), reconcile before returning
        if (
            getattr(self, '_reconciliation_method_effective', None)
            and self._priors is not None
        ):
            try:
                result = self.reconcile(result)
            except Exception as exc:
                warnings.warn(
                    f"TVA reconciliation failed; returning unreconciled forecast. {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return result

    def _predict_numpy(self, forecast_length: int) -> pd.DataFrame:
        """Torch-free forecast: damped local trend + detector components.

        The Arm A configuration from the Phase-4 ablation: per-series OLS
        slope over the last min(window_size, 4H) trend steps, damped with
        phi = 0.9, anchored at the last observed trend value; detector
        seasonality/holidays/level shifts added additively; intervals from
        the decomposition residual sigma; MinT reconciliation when a
        hierarchy is configured.
        """
        trend = self._components['trend'].values
        T, N = trend.shape
        L = max(min(self.window_size, 4 * self.forecast_horizon), 2)
        L = min(L, T)
        x = trend[-L:]
        t_idx = np.arange(L) - (L - 1) / 2.0
        denom = max(float(np.sum(t_idx**2)), 1e-8)
        slope = (x * t_idx[:, np.newaxis]).sum(axis=0) / denom  # (N,)
        damp = np.cumsum(0.9 ** np.arange(1, forecast_length + 1))
        trend_fc = trend[-1][np.newaxis, :] + damp[:, np.newaxis] * slope[np.newaxis, :]

        forecast_comps = self._decomposer.get_forecast_components(forecast_length)
        forecast_values = (
            trend_fc
            + forecast_comps['seasonality'].values
            + forecast_comps['holidays'].values
            + forecast_comps['level_shifts'].values
        )
        future_index = forecast_comps['trend'].index[:forecast_length]
        result = pd.DataFrame(
            forecast_values,
            index=future_index,
            columns=self._df_original.columns,
        )

        self._last_sigma = None
        resid_sigma = None
        if hasattr(self._decomposer, 'get_residual_sigma'):
            resid_sigma = self._decomposer.get_residual_sigma()
        if resid_sigma is not None:
            self._last_sigma = pd.DataFrame(
                np.tile(np.asarray(resid_sigma)[np.newaxis, :], (forecast_length, 1)),
                index=future_index,
                columns=self._df_original.columns,
            )

        if (
            getattr(self, '_reconciliation_method_effective', None)
            and self._priors is not None
        ):
            try:
                result = self.reconcile(result)
            except Exception as exc:
                warnings.warn(
                    f"TVA reconciliation failed; returning unreconciled forecast. {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return result

    def _select_factor_space(self, adjusted_raw: np.ndarray, n_factors: int):
        """Pick level vs log modeling space by inner-origin validation (2c).

        Only runs when ``factor_config['space'] == 'auto'``. Data-driven
        rather than assumed from column semantics; costs one extra
        factor-stage fit, and the winner is left installed.
        """
        from autots.evaluator.tva.factor_network import (
            fit_latent_factor_model,
            robust_level_scale,
        )

        requested = str((self.factor_config or {}).get('space', 'level') or 'level')
        if requested != 'auto':
            return

        horizon = max(int(self.forecast_horizon or 28), 1)
        best = None
        for space in ('level', 'log'):
            try:
                adjusted = self._to_factor_space(adjusted_raw, space)
                center, scale = robust_level_scale(adjusted)
                normalized = (adjusted - center[np.newaxis, :]) / scale[np.newaxis, :]
                cfg = dict(self.factor_config or {})
                cfg['space'] = space
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model, info = fit_latent_factor_model(
                        normalized,
                        n_factors=n_factors,
                        knot_spacing=self.factor_knot_spacing,
                        max_lag=self.factor_max_lag,
                        horizon=horizon,
                        config=cfg,
                        seed=self.random_seed,
                        device=self.device,
                        verbose=0,
                    )
                self._factor_network, self._factor_info = model, info
                self._factor_scale = {'center': center, 'scale': scale, 'space': space}
                folds = self._factor_inner_folds(horizon, 2)
                if not folds:
                    score = np.inf
                else:
                    errors = [
                        np.abs(folds['tva_folds'][i] - folds['actual_folds'][i])
                        for i in range(len(folds['tva_folds']))
                    ]
                    with np.errstate(invalid='ignore'):
                        score = float(
                            np.nanmean(
                                np.nanmean(np.concatenate(errors, axis=0), axis=0)
                                / folds['scale']
                            )
                        )
                if not np.isfinite(score):
                    score = np.inf
            except Exception:  # pragma: no cover - a failed space is not a crash
                continue
            state = (
                score,
                space,
                self._factor_network,
                self._factor_info,
                dict(self._factor_scale),
            )
            if best is None or score < best[0]:
                best = state
        if best is None:
            return
        _score, space, model, info, factor_scale = best
        self._factor_network, self._factor_info = model, info
        self._factor_scale = factor_scale
        if self.verbose:
            print(f"TVA: factor modeling space selected by validation: {space}")

    def _to_factor_space(self, values: np.ndarray, space: str) -> np.ndarray:
        """Map the adjusted panel into the factor model's modeling space (2c).

        Signed log1p so the transform is defined even where the adjusted
        panel goes negative after component subtraction. Only the trend block
        round-trips through this space; components stay in level space.
        """
        values = np.asarray(values, dtype=float)
        if str(space) != 'log':
            return values
        return np.sign(values) * np.log1p(np.abs(values))

    def _from_factor_space(self, values: np.ndarray, space: str) -> np.ndarray:
        """Inverse of ``_to_factor_space``."""
        values = np.asarray(values, dtype=float)
        if str(space) != 'log':
            return values
        return np.sign(values) * np.expm1(np.clip(np.abs(values), None, 700.0))

    def _selected_continuation(self, forecast_length: int, origin: int = None):
        """(1, H, K) deltas from the validation-selected continuation, or None.

        ``None`` means use the model's own damped local-linear rule (the
        pre-Phase-1 default).
        """
        info = getattr(self, '_factor_info', None)
        if not info or not info.get('continuation'):
            return None
        from autots.evaluator.tva.factor_network import (
            selected_continuation_deltas,
        )

        try:
            return selected_continuation_deltas(
                self._factor_network,
                forecast_length,
                info['continuation'],
                origin=origin,
                config=(info.get('config') or {}).get('continuation_config'),
            )
        except Exception as exc:  # pragma: no cover - never fail a forecast
            warnings.warn(
                f"TVA: selected factor continuation could not be applied; "
                f"falling back to the default rule. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _factor_trend_at_origin(self, origin: int, horizon: int) -> np.ndarray:
        """(H, N) level-space trend forecast made from a historical origin.

        Re-evaluates the fitted model at ``origin`` rather than refitting —
        every candidate sees the same mildly-optimistic model, so the
        selection ranking is unaffected and a full fit per fold is avoided.
        """
        deltas = self._selected_continuation(horizon, origin=origin)
        with torch.no_grad():
            normalized = (
                self._factor_network.forecast(horizon, origin=origin, deltas=deltas)
                .cpu()
                .numpy()
            )  # (N, H)
        center = self._factor_scale['center']
        scale = self._factor_scale['scale']
        return self._from_factor_space(
            normalized.T * scale[np.newaxis, :] + center[np.newaxis, :],
            self._factor_scale.get('space', 'level'),
        )

    def _refit_factor_trend(self, adjusted: np.ndarray, origin: int, horizon: int):
        """(H, N) level trend from a factor model refit on history <= origin.

        Honest counterpart of ``_factor_trend_at_origin``: only the factor
        stage is refit (components reused, truncated), so the extrapolation
        the safety layer grades is genuinely out of sample.

        Returns ``None`` on any failure; caller falls back to in-sample eval.
        """
        from autots.evaluator.tva.factor_network import (
            fit_latent_factor_model,
            robust_level_scale,
        )

        try:
            history = np.asarray(adjusted, dtype=float)[: int(origin) + 1]
            if history.shape[0] < max(3 * horizon, 120):
                return None
            center, scale = robust_level_scale(history)
            normalized = (history - center[np.newaxis, :]) / scale[np.newaxis, :]
            cfg = dict((self._factor_info or {}).get('config') or {})
            # the refit exists to *score* the incumbent, so it must not run the
            # selections that depend on it -- that would be circular
            for key in ('continuation_select', 'inner_refit'):
                cfg[key] = False
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model, _info = fit_latent_factor_model(
                    normalized,
                    n_factors=self._factor_network.K,
                    knot_spacing=self.factor_knot_spacing,
                    max_lag=self.factor_max_lag,
                    horizon=horizon,
                    config=cfg,
                    seed=self.random_seed,
                    verbose=0,
                )
                with torch.no_grad():
                    normalized_fc = model.forecast(horizon).cpu().numpy()
            return normalized_fc.T * scale[np.newaxis, :] + center[np.newaxis, :]
        except Exception:  # pragma: no cover - diagnostics only, never fatal
            return None

    def _factor_inner_folds(self, horizon: int, n_folds: int = 3) -> dict:
        """Inner rolling origins, in level space, for the safety selections.

        Returns the model forecast, a SeasonalNaive forecast, actuals, the
        MASE scale, and level anchors, all aligned on the same origins.
        Empty dict when history can't support even one origin.
        """
        from autots.evaluator.tva.safety import seasonal_naive_forecast

        df = self._df_original
        values = df.to_numpy(dtype=float)
        T = values.shape[0]
        H = max(int(horizon), 1)
        season = int(getattr(self, 'season_length', 7) or 7)

        adjusted = self._factor_input_panel()
        cfg_all = (self._factor_info or {}).get('config') or {}
        window = int(((self.factor_config or {}).get('reanchor_window')) or 14)

        origins = []
        for i in range(max(int(n_folds), 0)):
            origin = T - 1 - H * (i + 1)
            if origin < max(H, season) + 1:
                break
            origins.append(origin)
        origins = sorted(origins)
        if not origins:
            return {}

        refit = bool(cfg_all.get('inner_refit'))
        tva_folds, sn_folds, actual_folds = [], [], []
        anchor_levels, origin_levels = [], []
        comps = {
            key: (
                self._components.get(key)
                .reindex(index=df.index, columns=df.columns)
                .fillna(0.0)
                .to_numpy(dtype=float)
                if self._components.get(key) is not None
                else np.zeros_like(values)
            )
            for key in ('seasonality', 'holidays', 'level_shifts', 'trend')
        }
        from autots.evaluator.tva import seasonal as sn_mod

        for origin in origins:
            trend = (
                self._refit_factor_trend(adjusted, origin, H)
                if refit
                else self._factor_trend_at_origin(origin, H)
            )
            if trend is None:
                trend = self._factor_trend_at_origin(origin, H)
            window_slice = slice(origin + 1, origin + 1 + H)
            # must be a FORECAST of periodic components, not their in-sample
            # fitted values -- fitted values would hand this fold an oracle
            # the SeasonalNaive fold doesn't get, biasing the blend selection
            returned = getattr(self, '_shift_returned', None)
            shift_level = comps['level_shifts']
            if returned is not None and returned.shape == shift_level.shape:
                shift_level = shift_level - returned
            residual = (
                values[: origin + 1]
                - comps['trend'][: origin + 1]
                - shift_level[: origin + 1]
            )
            periodic = sn_mod.tile_profile(
                sn_mod.empirical_profile(residual, season, 2), H
            )
            if periodic is None:
                periodic = np.zeros((H, values.shape[1]), dtype=float)
            # a level shift is a step: the only thing knowable at the origin is
            # that it persists
            add_back = periodic + shift_level[origin][np.newaxis, :]
            tva_folds.append(trend + add_back)
            sn_folds.append(seasonal_naive_forecast(values[: origin + 1], H, season))
            actual_folds.append(values[window_slice])
            recent = adjusted[max(origin + 1 - window, 0) : origin + 1]
            anchor_levels.append(np.nanmedian(recent, axis=0))
            origin_levels.append(trend[0])

        train = values[: origins[0] + 1]
        if train.shape[0] > season:
            scale = np.nanmean(np.abs(train[season:] - train[:-season]), axis=0)
        else:
            scale = np.nanmean(np.abs(np.diff(train, axis=0)), axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.nan)

        return {
            'origins': [int(o) for o in origins],
            'tva_folds': tva_folds,
            'sn_folds': sn_folds,
            'actual_folds': actual_folds,
            'scale': scale,
            'anchor_levels': np.asarray(anchor_levels, dtype=float),
            'origin_levels': np.asarray(origin_levels, dtype=float),
            'season_m': season,
        }

    def _seasonal_candidates(
        self, horizon: int, datepart_future: np.ndarray, upto: int = None
    ) -> dict:
        """{name: (H, N)} seasonal-path candidates for one forecast window.

        ``upto`` restricts history used for the empirical/amplitude
        candidates, keeping selection honest at a historical origin.
        """
        from autots.evaluator.tva import seasonal as sn

        cfg = ((self._factor_info or {}).get('config') or {}).get(
            'seasonal_config'
        ) or {}
        season = int(cfg.get('season_m', getattr(self, 'season_length', 7) or 7))
        df = self._df_original
        values = df.to_numpy(dtype=float)
        end = len(values) if upto is None else int(upto) + 1

        def comp(key):
            frame = self._components.get(key)
            if frame is None:
                return np.zeros_like(values)
            return (
                frame.reindex(index=df.index, columns=df.columns)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

        residual = values[:end] - comp('trend')[:end] - comp('level_shifts')[:end]
        candidates = {'datepart': np.asarray(datepart_future, dtype=float)}

        n_cycles = int(cfg.get('n_cycles', 2))
        profile = sn.empirical_profile(residual, season, n_cycles)
        weekly = sn.tile_profile(profile, horizon)
        # a yearly cycle only earns a candidate when at least two exist to
        # average; one noisy year tiled forward is worse than the datepart fit
        yearly_m = int(cfg.get('yearly_m', 365))
        min_years = int(cfg.get('min_years_for_yearly', 2))
        if weekly is not None and end >= yearly_m * min_years:
            yearly = sn.tile_profile(
                sn.empirical_profile(residual, yearly_m, n_cycles), horizon
            )
            if yearly is not None:
                candidates['empirical_yearly'] = yearly
        if weekly is not None:
            candidates['empirical'] = weekly

        window = season * int(cfg.get('amplitude_window_cycles', 2))
        fitted_in = comp('seasonality')[:end]
        if fitted_in.shape[0] >= window:
            alpha = sn.amplitude_scale(fitted_in[-window:], residual[-window:], cfg)
            candidates['amplitude'] = (
                np.asarray(datepart_future, dtype=float) * alpha[np.newaxis, :]
            )
        return candidates

    def _select_seasonal_paths(self, horizon: int, datepart_future: np.ndarray):
        """(H, N) seasonal path plus the selection diagnostics (2a).

        Scored on one held-out window at origin ``T - H - 1``, each candidate
        assembled into a complete forecast so the comparison isolates just the
        seasonal path. Incumbent datepart path wins ties.
        """
        datepart_future = np.asarray(datepart_future, dtype=float)
        cfg_all = (self._factor_info or {}).get('config') or {}
        if not cfg_all.get('seasonal_arbitration'):
            return datepart_future, {}

        from autots.evaluator.tva import seasonal as sn

        try:
            df = self._df_original
            values = df.to_numpy(dtype=float)
            T = values.shape[0]
            H = max(int(horizon), 1)
            origin = T - H - 1
            season = int(getattr(self, 'season_length', 7) or 7)
            if origin < max(3 * H, 2 * season):
                return datepart_future, {'reason': 'insufficient history'}

            def comp(key):
                frame = self._components.get(key)
                if frame is None:
                    return np.zeros_like(values)
                return (
                    frame.reindex(index=df.index, columns=df.columns)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )

            window = slice(origin + 1, origin + 1 + H)
            trend = self._factor_trend_at_origin(origin, H)
            fixed = comp('holidays')[window] + comp('level_shifts')[window]
            hist_candidates = self._seasonal_candidates(
                H, comp('seasonality')[window], upto=origin
            )
            assembled = {
                name: trend + path + fixed
                for name, path in hist_candidates.items()
                if path is not None
            }
            actual = values[window]
            train = values[: origin + 1]
            if train.shape[0] > season:
                scale = np.nanmean(np.abs(train[season:] - train[:-season]), axis=0)
            else:
                scale = np.nanmean(np.abs(np.diff(train, axis=0)), axis=0)
            selection = sn.select_seasonal_paths(
                assembled,
                actual,
                scale,
                default='datepart',
                config=cfg_all.get('seasonal_config'),
            )
            future_candidates = self._seasonal_candidates(H, datepart_future)
            path = sn.assemble_choice(
                future_candidates, selection['choice'], 'datepart'
            )
            selection.pop('scores', None)
            return path, selection
        except Exception as exc:  # pragma: no cover - never fail a forecast
            warnings.warn(
                f"TVA seasonal arbitration failed; keeping the datepart path. "
                f"{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return datepart_future, {'error': str(exc)}

    def _apply_coherence(self, normalized_fc: np.ndarray, horizon: int):
        """Coherence-shrink the standardized trend block (Phase 4).

        Applied to standardized trend only, before denormalization and before
        seasonality/holidays/anomalies/level-shifts are added back, so local
        features aren't smeared toward a cross-series consensus.

        Default-off: useful once the factor-loading identification is good
        enough to trust, gated on loading recovery rather than retired.

        Args:
            normalized_fc: (N, H) standardized-space trend forecast.
            horizon: H.

        Returns:
            (adjusted (N, H), info dict).
        """
        cfg = (self._factor_info or {}).get('config') or {}
        # E: the top-level ``coherence_config`` argument was stored and never
        # read. Passing it is an explicit user act, so it also enables the
        # shrink when factor_config left ``coherence`` unset -- the global
        # default stays off.
        top_level = dict(self.coherence_config or {})
        enabled = cfg.get('coherence')
        if not enabled and top_level and 'coherence' not in (self.factor_config or {}):
            enabled = True
        if not enabled:
            return normalized_fc, {}
        try:
            from autots.evaluator.tva import coherence as ch

            cconf = dict(cfg.get('coherence_config') or {})
            cconf.update(top_level)
            cconf.setdefault('gated', self._factor_info.get('gated_series'))
            cconf.setdefault('sigma', self._factor_info.get('sigma'))
            # C4: per-factor split-half stability, when the fit computed it.
            # resolve_signs multiplies it into the sign confidence, so an
            # unreplicable factor falls below min_sign_confidence and drops
            # out of the graph rather than partitioning the panel on noise.
            cconf.setdefault('stability', self._factor_info.get('factor_stability'))
            loadings = self._factor_network.fitted_loadings(
                cfg.get('prune_share', 0.02)
            )
            # C5: prefer the structure-only fit's loadings when one was
            # computed and cleared its agreement guard (see
            # _fit_structure_loadings). Shape must still line up with the
            # panel the shrink will act on.
            structure = getattr(self, '_structure_loadings', None)
            if structure is not None and structure.shape[0] == loadings.shape[0]:
                loadings = structure
            if self._prior_input is not None:
                cconf.setdefault('prior_weight', self.prior_confidence)
            graphs, _meta = ch.build_candidates(
                loadings, cconf, prior=self._prior_input
            )
            folds = self._coherence_inner_folds(horizon)
            if not folds:
                return normalized_fc, {'reason': 'insufficient history'}
            selection = ch.select_coherence(
                folds['trend_folds'],
                folds['actual_folds'],
                graphs,
                folds['scale'],
                cconf,
            )
            trend = np.asarray(normalized_fc, dtype=float).T  # (H, N)
            adjusted, info = ch.apply_selection(trend, selection, graphs, cconf)
            return np.asarray(adjusted, dtype=float).T, info
        except Exception as exc:  # pragma: no cover - never fail a forecast
            warnings.warn(
                f"TVA coherence shrink failed; returning the unshrunk trend. " f"{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return normalized_fc, {'error': str(exc)}

    def _coherence_inner_folds(self, horizon: int, n_folds: int = 3) -> dict:
        """Inner folds in STANDARDIZED trend space, for the coherence selector.

        Distinct from ``_factor_inner_folds`` (level space, components added
        back): coherence is a property of the trend block alone.
        """
        try:
            adjusted = self._to_factor_space(
                self._factor_input_panel(), self._factor_scale.get('space', 'level')
            )
            center = self._factor_scale['center']
            scale = self._factor_scale['scale']
            std_panel = (adjusted - center[np.newaxis, :]) / scale[np.newaxis, :]
            T = std_panel.shape[0]
            H = max(int(horizon), 1)
            season = int(getattr(self, 'season_length', 7) or 7)
            origins = []
            for i in range(max(int(n_folds), 0)):
                origin = T - 1 - H * (i + 1)
                if origin < max(H, season) + 1:
                    break
                origins.append(origin)
            if not origins:
                return {}
            trend_folds, actual_folds = [], []
            for origin in sorted(origins):
                deltas = self._selected_continuation(H, origin=origin)
                with torch.no_grad():
                    path = (
                        self._factor_network.forecast(H, origin=origin, deltas=deltas)
                        .cpu()
                        .numpy()
                        .T
                    )
                trend_folds.append(path)
                actual_folds.append(std_panel[origin + 1 : origin + 1 + H])
            train = std_panel[: min(origins) + 1]
            if train.shape[0] > season:
                denom = np.nanmean(np.abs(train[season:] - train[:-season]), axis=0)
            else:
                denom = np.nanmean(np.abs(np.diff(train, axis=0)), axis=0)
            return {
                'trend_folds': trend_folds,
                'actual_folds': actual_folds,
                'scale': np.where(np.isfinite(denom) & (denom > 1e-12), denom, np.nan),
            }
        except Exception:  # pragma: no cover
            return {}

    def _apply_derived_identities(self, result: pd.DataFrame) -> pd.DataFrame:
        """Rebuild declared ratio columns from their parents' forecasts (6d).

        Forecasting ``Z = X / Y`` directly lets X, Y, Z tell inconsistent
        stories; replacing it with ``forecast(X) / forecast(Y)`` makes it
        coherent by construction. Definitions are declared, never inferred
        from column names.
        """
        definitions = self.derived_definitions
        if not definitions:
            return result
        history = self._df_original
        for column, spec in definitions.items():
            try:
                numerator, denominator = spec[0], spec[1]
            except Exception:
                continue
            if any(
                name not in result.columns for name in (column, numerator, denominator)
            ):
                continue
            den_fc = result[denominator].to_numpy(dtype=float)
            reference = float(
                np.nanmedian(np.abs(history[denominator].to_numpy(dtype=float)))
            )
            floor = max(reference * 1e-3, 1e-12)
            if not np.isfinite(reference) or np.nanmin(np.abs(den_fc)) < floor:
                # near-zero denominator would amplify error; keep direct forecast
                continue
            result[column] = result[numerator].to_numpy(dtype=float) / den_fc
        return result

    def _apply_safety_layer(self, forecast_values: np.ndarray, horizon: int):
        """Blend, re-anchor and cap the reconstructed factor forecast (1b/1b'/1e).

        Each piece is selected on the same inner rolling origins and can
        settle on a no-op by selection, not just by config; with all four
        flags off this returns the input unchanged, bit for bit.

        Order: re-anchor (fixes level offset) before blend (fixes path shape)
        so blending doesn't spend budget correcting an offset a shift would
        fix for free; cap runs last as the final bound.

        Returns:
            (adjusted forecast (H, N), diagnostics dict).
        """
        cfg = (self._factor_info or {}).get('config') or {}
        wants = any(
            cfg.get(key)
            for key in ('sn_blend', 'reanchor', 'error_cap', 'conformal_sigma')
        )
        if not wants:
            return forecast_values, {}

        from autots.evaluator.tva import safety as sf

        sconf = cfg.get('safety_config') or {}
        folds = self._factor_inner_folds(horizon, int(cfg.get('inner_folds', 3) or 3))
        if not folds:
            return forecast_values, {'reason': 'insufficient history for inner folds'}

        values = self._df_original.to_numpy(dtype=float)
        season = folds['season_m']
        scale = folds['scale']
        tva_folds = folds['tva_folds']
        info = {'origins': folds['origins']}

        # ---- 1e: re-anchor the level ---------------------------------------
        offset = np.zeros(values.shape[1], dtype=float)
        alpha = np.zeros(values.shape[1], dtype=float)
        if cfg.get('reanchor'):
            alpha = sf.select_reanchor_alpha(
                tva_folds,
                folds['actual_folds'],
                folds['anchor_levels'],
                folds['origin_levels'],
                scale,
                sconf,
            )
            window = int(sconf.get('reanchor_window', 14) or 14)
            adjusted = self._factor_input_panel()
            anchor_now = np.nanmedian(adjusted[-window:], axis=0)
            offset = (
                anchor_now
                - self._factor_trend_at_origin(len(values) - 1 - horizon, horizon)[0]
            )
            offsets_by_fold = folds['anchor_levels'] - folds['origin_levels']
            tva_folds = [
                sf.apply_reanchor(f, offsets_by_fold[i], alpha)
                for i, f in enumerate(tva_folds)
            ]
            forecast_values = sf.apply_reanchor(forecast_values, offset, alpha)
            info['reanchor_alpha'] = alpha.tolist()

        # ---- 1b: SeasonalNaive blend ---------------------------------------
        weights = np.ones(values.shape[1], dtype=float)
        sn_fc = sf.seasonal_naive_forecast(values, horizon, season)
        if cfg.get('sn_blend'):
            weights = sf.select_blend_weights(
                tva_folds, folds['sn_folds'], folds['actual_folds'], scale, sconf
            )
            forecast_values = sf.blend_forecasts(forecast_values, sn_fc, weights)
            info['blend_weights'] = weights.tolist()

        # ---- 1b': horizon-bucketed conformal scales -------------------------
        bucket_scales = None
        if cfg.get('conformal_sigma') or cfg.get('error_cap'):
            residuals = np.stack(
                [tva_folds[i] - folds['actual_folds'][i] for i in range(len(tva_folds))]
            )
            bucket_scales = sf.horizon_bucket_scales(residuals, sconf)

        # ---- 1b: error cap --------------------------------------------------
        capped = 0
        if cfg.get('error_cap'):
            abs_errors = [
                np.abs(folds['sn_folds'][i] - folds['actual_folds'][i])
                for i in range(len(tva_folds))
            ]
            lower, upper = sf.error_cap_bounds(
                sn_fc,
                abs_errors,
                horizon,
                sconf,
                bucket_scales=bucket_scales if cfg.get('conformal_sigma') else None,
            )
            capped = sf.count_capped(forecast_values, lower, upper)
            forecast_values = sf.apply_error_cap(forecast_values, lower, upper)
            info['capped_cells'] = int(capped)

        info['summary'] = sf.summarize(
            blend_weights=weights,
            reanchor_alphas=alpha,
            bucket_scales=bucket_scales,
            capped_cells=capped,
            scale=scale,
            n_series=values.shape[1],
            horizon=horizon,
        )
        # only hand a bucket profile to sigma when the flag asked for it; the
        # JSON-safe copy of it lives in info['summary']
        info['bucket_scales'] = bucket_scales if cfg.get('conformal_sigma') else None
        return forecast_values, info

    def _predict_factor(self, forecast_length: int) -> pd.DataFrame:
        """Forecast from the learned latent-factor trend model.

        The factor paths are continued with their calibrated per-factor damping
        and recombined through each series' loadings and response lag (a series
        with lag d reads the leaders' already-observed path for its first d
        steps); the idiosyncratic line is continued with its own damping. The
        detector's seasonality, holidays and level shifts are added back
        additively, and sigma combines the model's rolling-origin residual with
        the decomposition residual in quadrature.
        """
        if self._factor_network is None:
            raise RuntimeError("TVA must be fit before calling predict.")

        deltas = self._selected_continuation(forecast_length)
        with torch.no_grad():
            normalized_fc = (
                self._factor_network.forecast(forecast_length, deltas=deltas)
                .cpu()
                .numpy()
            )  # (N, H)
        normalized_fc, self._coherence_info = self._apply_coherence(
            normalized_fc, forecast_length
        )
        center = self._factor_scale['center']
        scale = self._factor_scale['scale']
        space = self._factor_scale.get('space', 'level')
        trend_fc = self._from_factor_space(
            normalized_fc.T * scale[np.newaxis, :] + center[np.newaxis, :], space
        )

        forecast_comps = self._decomposer.get_forecast_components(forecast_length)
        future_index = forecast_comps['trend'].index[:forecast_length]
        seasonal_path, seasonal_info = self._select_seasonal_paths(
            forecast_length, forecast_comps['seasonality'].values
        )
        self._seasonal_info = seasonal_info
        shift_add_back = forecast_comps['level_shifts'].values
        returned = getattr(self, '_shift_returned', None)
        if returned is not None and returned.shape[1] == shift_add_back.shape[1]:
            # the last row is the standing level of the shifts the veto handed
            # back to the trend, which the trend forecast already carries
            shift_add_back = shift_add_back - returned[-1][np.newaxis, :]
        forecast_values = (
            trend_fc
            + seasonal_path
            + forecast_comps['holidays'].values
            + shift_add_back
        )

        forecast_values, safety = self._apply_safety_layer(
            forecast_values, forecast_length
        )
        self._safety_info = safety

        result = pd.DataFrame(
            forecast_values, index=future_index, columns=self._df_original.columns
        )

        model_sigma = np.asarray(self._factor_info['sigma'], dtype=float) * scale
        resid_sigma = None
        if hasattr(self._decomposer, 'get_residual_sigma'):
            resid_sigma = self._decomposer.get_residual_sigma()
        if resid_sigma is not None:
            model_sigma = np.sqrt(
                model_sigma**2 + np.asarray(resid_sigma, dtype=float) ** 2
            )
        from autots.evaluator.tva.safety import conformal_sigma

        self._last_sigma = pd.DataFrame(
            conformal_sigma(model_sigma, forecast_length, safety.get('bucket_scales')),
            index=future_index,
            columns=self._df_original.columns,
        )

        if (
            getattr(self, '_reconciliation_method_effective', None)
            and self._priors is not None
        ):
            try:
                result = self.reconcile(result)
            except Exception as exc:
                warnings.warn(
                    f"TVA reconciliation failed; returning unreconciled forecast. {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        # declared ratio identities are enforced last, after blending, shrinkage
        # and reconciliation, so nothing downstream can break them again
        result = self._apply_derived_identities(result)
        return result

    # ---- forecast covariance -------------------------------------------
    def forecast_covariance(self, horizon: int = None):
        """(N, N) raw-unit cross-series forecast-error covariance, or None.

        One covariance object for the callers that each otherwise invent their
        own weighting: MinT's ``W``, the closed-form what-if solver, and (later)
        interval coherence. It is built from the rolling-origin residual matrix
        the trend model already forms, blended toward a low-rank
        ``Lambda Sigma_f Lambda' + diag(psi)`` target, and floored at the same
        per-series sigma the shipped prediction intervals use.

        Only the modes whose trend model is linear in a small set of factors
        are served: ``trend_network='factor'`` (loadings from the fitted factor
        model) and ``'none'`` (loadings from structure discovery, residuals
        from rolling the damped-trend rule). ``'v1'``/``'v2'`` return None —
        those already have :meth:`_compute_recent_reconciliation_residuals` for
        reconciliation and the backprop scenario solver for what-if, and a
        covariance built from a *different* model than the one forecasting
        would misdescribe them.

        Args:
            horizon: forecast horizon the covariance should describe. Defaults
                to ``forecast_horizon``.

        Returns:
            ``(sigma, info)`` with ``sigma`` an (N, N) covariance in the raw
            units of the input data and ``info`` a diagnostics dict carrying
            ``alpha`` (shrinkage intensity), ``beta`` (structural scale),
            ``psi``, ``floor_binding`` and ``n_samples``. Returns ``None`` when
            unfitted, when the mode is not served, or when no usable residual
            matrix or loadings exist — callers degrade to their previous
            behaviour rather than seeing an exception.
        """
        if self._components is None or self._df_original is None:
            return None
        H = int(horizon) if horizon else int(self.forecast_horizon)
        H = max(H, 1)
        try:
            if self.trend_network_type == 'factor' and self._factor_network is not None:
                return self._factor_forecast_covariance(H)
            if self.trend_network_type == 'none':
                return self._numpy_forecast_covariance(H)
        except Exception as exc:  # pragma: no cover - never fail a forecast
            warnings.warn(
                f"TVA forecast covariance could not be assembled; callers will "
                f"fall back to their uncorrelated behaviour. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        return None

    def _factor_residual_matrix(self, horizon: int) -> Optional[np.ndarray]:
        """(n_samples, N) rolling-origin residuals in the factor model's space.

        Prefers the matrix the fit already produced (``info['residual_matrix']``,
        kept rather than reduced to its per-series std) and recomputes only when
        that one does not describe this panel at this horizon — a different
        horizon than the fit's, or an anchor/responder fit whose stored matrix
        covers the anchor sub-panel only.
        """
        N = self._df_original.shape[1]
        info = self._factor_info or {}
        stored = info.get('residual_matrix')
        if stored is not None and int(horizon) == int(self.forecast_horizon):
            stored = np.asarray(stored, dtype=float)
            if stored.ndim == 2 and stored.shape[1] == N and stored.shape[0] >= 2:
                return np.where(np.isfinite(stored), stored, 0.0)

        model = self._factor_network
        space = self._factor_scale.get('space', 'level')
        center = self._factor_scale['center']
        scale = self._factor_scale['scale']
        panel = self._to_factor_space(self._factor_input_panel(), space)
        norm = (panel - center[np.newaxis, :]) / scale[np.newaxis, :]
        T = norm.shape[0]
        H = max(int(horizon), 1)

        span = max(min(300, T // 3), 1)
        lo = max(T - span, int(getattr(model, 'slope_window', 2)))
        hi = T - H - 1
        if hi < lo:
            return None
        origins = np.arange(lo, hi + 1)
        if origins.size > 128:
            origins = origins[np.linspace(0, origins.size - 1, 128).astype(int)]

        with torch.no_grad():
            pred = (
                model.rolling_forecast(
                    torch.tensor(origins, device=self.device, dtype=torch.long), H
                )
                .cpu()
                .numpy()
            )  # (O, H, N)
        target = norm[origins[:, None] + np.arange(1, H + 1)[None, :]]  # (O, H, N)
        resid = (pred - target).reshape(-1, N)
        return np.where(np.isfinite(resid), resid, 0.0)

    def _factor_sigma_floor(self, jacobian: np.ndarray) -> np.ndarray:
        """(N,) raw-unit sigma floor, matching ``_predict_factor``'s intervals.

        The factor model's idiosyncratic term is capped at 2 degrees of freedom
        by design, so its own residual spread is structurally over-confident
        (this is why ``factor_variance_share`` reads 1.00 even on factor-free
        panels). Floor at the sigma the prediction intervals are already built
        from: the model residual combined in quadrature with the
        decomposition's.
        """
        scale = self._factor_scale['scale']
        model_sigma = (
            np.asarray(self._factor_info['sigma'], dtype=float) * scale * jacobian
        )
        resid_sigma = None
        if hasattr(self._decomposer, 'get_residual_sigma'):
            resid_sigma = self._decomposer.get_residual_sigma()
        if resid_sigma is not None:
            model_sigma = np.sqrt(
                model_sigma**2 + np.asarray(resid_sigma, dtype=float) ** 2
            )
        return model_sigma

    def _factor_forecast_covariance(self, horizon: int):
        """Forecast covariance for ``trend_network='factor'``."""
        from autots.evaluator.tva.covariance import (
            assemble_covariance,
            damped_accumulated_variance,
        )

        resid = self._factor_residual_matrix(horizon)
        if resid is None or resid.shape[0] < 2:
            return None

        model = self._factor_network
        loadings = model.fitted_loadings()  # (N, k), pruned + sign-fixed
        keep = model.live_factors()
        with torch.no_grad():
            phi = model.phi().detach().cpu().numpy().ravel()[keep]
        factor_var = damped_accumulated_variance(phi, horizon)

        scale = self._factor_scale['scale']
        space = self._factor_scale.get('space', 'level')
        jacobian = np.ones_like(np.asarray(scale, dtype=float))
        if str(space) == 'log':
            # signed log1p: x = sign(y) expm1(|y|), so d x / d y = exp(|y|).
            # Evaluated at the horizon-mean forecast, which is where the
            # covariance is being asked about.
            deltas = self._selected_continuation(horizon)
            with torch.no_grad():
                fc_norm = model.forecast(horizon, deltas=deltas).cpu().numpy()  # (N, H)
            y_model = fc_norm.T * scale[np.newaxis, :] + self._factor_scale['center']
            jacobian = np.exp(np.clip(np.abs(y_model).mean(axis=0), None, 700.0))

        sigma, info = assemble_covariance(
            resid,
            loadings=loadings,
            factor_var=factor_var,
            scale=scale,
            jacobian=jacobian,
            floor_sd=self._factor_sigma_floor(jacobian),
        )
        info['loadings'] = loadings
        info['space'] = space
        info['horizon'] = int(horizon)
        info['source'] = 'factor'
        return sigma, info

    def _numpy_residual_matrix(self, horizon: int) -> Optional[np.ndarray]:
        """(n_samples, N) rolling-origin residuals of the torch-free trend rule.

        Rolls exactly the rule :meth:`_predict_numpy` ships — OLS slope over the
        last ``L`` trend steps, damped 0.9, anchored at the last trend value —
        so the covariance describes the forecast that is actually produced.
        """
        trend = self._components['trend'].values
        T, N = trend.shape
        H = max(int(horizon), 1)
        L = max(min(self.window_size, 4 * self.forecast_horizon), 2)
        L = min(L, T)
        if T - H < L:
            return None
        origins = np.arange(L, T - H + 1)
        if origins.size > 128:
            origins = origins[np.linspace(0, origins.size - 1, 128).astype(int)]

        t_idx = np.arange(L) - (L - 1) / 2.0
        denom = max(float(np.sum(t_idx**2)), 1e-8)
        damp = np.cumsum(0.9 ** np.arange(1, H + 1))[:, np.newaxis]
        rows = []
        for o in origins:
            x = trend[o - L : o]
            slope = (x * t_idx[:, np.newaxis]).sum(axis=0) / denom
            fc = trend[o - 1][np.newaxis, :] + damp * slope[np.newaxis, :]
            rows.append(fc - trend[o : o + H])
        resid = np.concatenate(rows, axis=0)
        return np.where(np.isfinite(resid), resid, 0.0)

    def _numpy_forecast_covariance(self, horizon: int):
        """Forecast covariance for ``trend_network='none'`` (torch-free)."""
        from autots.evaluator.tva.covariance import (
            assemble_covariance,
            damped_accumulated_variance,
        )

        disc = self._discovery
        if not disc:
            return None
        loadings = np.asarray(disc.get('loadings'), dtype=float)
        factors = np.asarray(disc.get('factors'), dtype=float)
        if (
            loadings.ndim != 2
            or loadings.size == 0
            or not np.any(loadings)
            or factors.ndim != 2
            or factors.shape[1] != loadings.shape[1]
            or loadings.shape[0] != self._df_original.shape[1]
        ):
            return None

        resid = self._numpy_residual_matrix(horizon)
        if resid is None or resid.shape[0] < 2:
            return None

        # discovery loadings are in standardized-difference units; the scalar
        # beta fitted in structural_target absorbs the unit mismatch, so only
        # the *relative* per-factor variance has to be right here.
        inc_var = np.nanvar(np.diff(factors, axis=0), axis=0)
        inc_var = np.where(np.isfinite(inc_var) & (inc_var > 0), inc_var, 1.0)
        factor_var = (
            damped_accumulated_variance(np.full(loadings.shape[1], 0.9), horizon)
            * inc_var
        )

        floor = None
        if hasattr(self._decomposer, 'get_residual_sigma'):
            floor = self._decomposer.get_residual_sigma()

        sigma, info = assemble_covariance(
            resid,
            loadings=loadings,
            factor_var=factor_var,
            floor_sd=floor,
        )
        info['loadings'] = loadings
        info['space'] = 'level'
        info['horizon'] = int(horizon)
        info['source'] = 'discovery'
        return sigma, info

    def _structural_reconciliation_W(
        self, S: np.ndarray, aggregate_sigma=None
    ) -> Optional[np.ndarray]:
        """(L, L) MinT weighting from the forecast covariance, or None.

        The model produces a bottom-level covariance; MinT wants one over every
        hierarchy node, which for a summing matrix S is ``S Sigma S'`` plus
        whatever error the *aggregate-level* base forecasts carry on their own.

        That second term is not optional. ``S Sigma S'`` has rank M < L, and
        ``mint_reconcile`` needs a positive-definite W, so the obvious fix is a
        numerical ridge — but with only a ridge the estimator collapses::

            G = (S' W^-1 S)^-1 S' W^-1  ->  (S' S)^-1 S'    as ridge -> 0

        i.e. exactly the OLS reconciler that ``W = I`` already gives. Sigma
        cancels out entirely (verified numerically to 1e-12). Sigma changes the
        answer only when the aggregate nodes carry variance that is *not* the
        aggregated bottom variance — which is the case precisely when the
        aggregate forecasts were made independently rather than summed up from
        the bottom level.

        Args:
            S: (L, M) summing matrix.
            aggregate_sigma: optional (n_agg,) standard deviations of the
                independently-produced aggregate-level base forecasts. Without
                it the returned W is the rank-deficient form plus a numerical
                ridge, which reconciles identically to ``W = I``.

        Returns:
            (L, L) covariance, or None when the mode is off or no forecast
            covariance is available.
        """
        mode = str(getattr(self, 'reconciliation_covariance', 'auto') or 'auto').lower()
        if mode == 'auto':
            mode = RECONCILIATION_COVARIANCE_AUTO
        if mode != 'structural':
            return None
        cov = self.forecast_covariance()
        if cov is None:
            return None
        sigma = np.asarray(cov[0], dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        if sigma.shape[0] != S.shape[1]:
            return None
        W = S @ sigma @ S.T
        L, M = S.shape
        n_agg = L - M
        if aggregate_sigma is not None and n_agg > 0:
            extra = np.zeros(L)
            agg = np.asarray(aggregate_sigma, dtype=np.float64).ravel()
            if agg.shape[0] != n_agg:
                raise ValueError(
                    f"aggregate_sigma must have {n_agg} entries; got {agg.shape[0]}"
                )
            extra[:n_agg] = np.maximum(agg, 0.0) ** 2
            W = W + np.diag(extra)
        trace = float(np.trace(W))
        ridge = 1e-6 * (trace / L if trace > 0 else 1.0)
        return W + np.eye(L) * ridge

    def what_if(self, **constraints) -> pd.DataFrame:
        """Scenario planning: adjust the forecast to satisfy user constraints.

        Two solvers behind one signature, picked by what the fit produced.
        ``trend_network='v1'``/``'v2'`` backpropagate a latent perturbation
        through the trend network (:class:`~.scenario.BifrostOptimizer`).
        ``'factor'`` and ``'none'`` have no network to backprop through — the
        optimizer used to dereference ``self._network`` and raise there — but
        their forecast is linear in the factor paths, so
        :class:`~.scenario.ClosedFormScenario` solves the same
        minimum-disruption problem in closed form under
        :meth:`forecast_covariance`.

        Args:
            **constraints: Passed to the solver's methods.
                Supported keys:
                - series_name, timestep, target_value -> apply_constraint
                - series_name, growth_rate -> apply_growth_constraint
                - level_name, target_value -> apply_hierarchical_constraint

        Returns:
            Adjusted forecast DataFrame.
        """
        if self._network is None:
            from autots.evaluator.tva.scenario import ClosedFormScenario

            optimizer = ClosedFormScenario(self)
        else:
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

    def reconcile(
        self,
        forecasts: pd.DataFrame = None,
        residuals: np.ndarray = None,
        aggregate_sigma=None,
    ) -> pd.DataFrame:
        """Apply hierarchical reconciliation if configured.

        .. note::
            When ``forecasts`` holds only the bottom level, this method builds
            the aggregate rows by summing it through S — which places the input
            *exactly* in the coherent subspace MinT projects onto, so
            reconciliation returns it unchanged for any W, in any trend mode.
            Reconciliation can only move a number when the aggregate levels
            carry a forecast that was not derived from the bottom one; pass a
            full (L-column) ``forecasts`` frame, ordered aggregates-then-bottom,
            to supply one.

        Args:
            forecasts: Forecast DataFrame. If None, generates fresh predictions.
                Bottom-level columns only, or all L hierarchy levels with the
                aggregate columns first (see the note above).
            residuals: Optional historical residual matrix for all hierarchy levels.
                If omitted, TVA estimates recent one-step residuals from the last
                30-90 training windows so covariance-aware reconciliation methods
                can use a meaningful W matrix.
            aggregate_sigma: Optional (n_agg,) forecast-error standard deviations
                for independently-produced aggregate forecasts. Only consulted on
                the structural-covariance path, where it is what stops Sigma from
                cancelling out of MinT's estimator (see
                :meth:`_structural_reconciliation_W`).

        Returns:
            Reconciled DataFrame.
        """
        if forecasts is None:
            forecasts = self.predict()

        if self._reconciler is None:
            method = getattr(
                self, '_reconciliation_method_effective', self.reconciliation_method
            )
            if method:
                self._reconciler = ReconciliationBridge(method=method)
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

        synthesized_aggregates = forecasts.shape[1] == n_bottom and n_agg > 0
        if synthesized_aggregates:
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

        W = None
        if residuals is None:
            residuals = self._compute_recent_reconciliation_residuals(S)
            if residuals is None and not synthesized_aggregates:
                # the network-based estimator has nothing to offer in the
                # factor and torch-free modes; without this the bridge falls
                # back to W = I, which makes "MinT" plain OLS reconciliation.
                # Skipped entirely when the aggregate rows were synthesized
                # from the bottom level: that input is already coherent, so no
                # W can move it and assembling one would be pure cost.
                W = self._structural_reconciliation_W(
                    S, aggregate_sigma=aggregate_sigma
                )
        elif isinstance(residuals, pd.DataFrame):
            residuals = residuals.values

        if W is None:
            reconciled = self._reconciler.reconcile(full_df, S, residuals=residuals)
        else:
            reconciled = self._reconciler.reconcile(full_df, S, W=W)
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
        if self._network is None and self._components is None:
            raise RuntimeError("TVA must be fit first.")

        if self._network is None:
            # torch-free mode: the discovered merged adjacency IS the graph
            if self._discovery is not None:
                return np.asarray(self._discovery['adjacency'], dtype=np.float32)
            n = len(self._df_original.columns)
            return np.zeros((n, n), dtype=np.float32)

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
        prototype_weights = None
        if 'prototype_weights' in outputs:
            prototype_weights = outputs['prototype_weights'].detach().cpu().numpy()[0]
        global_prototype_weights = None
        if 'global_prototype_weights' in outputs:
            global_prototype_weights = (
                outputs['global_prototype_weights'].detach().cpu().numpy()[0]
            )
        decoded_top_trends = None
        if 'composite_trend' in outputs:
            decoded_top_trends = outputs['composite_trend'].detach().cpu().numpy()[0]
        graph = self.get_graph()
        # V2's graph is series-level (N x N): draw the DAG over real series
        # names (W-7). Legacy latent-sized graphs keep the top-level view.
        dag_level = (
            'series' if graph.shape[0] == len(self._df_original.columns) else 'top'
        )
        snapshot = build_graph_snapshot(
            adjacency_dense=graph,
            assignment_matrices=assignment_matrices,
            threshold=(
                self._structure_config.threshold_for_export
                if threshold is None
                else float(threshold)
            ),
            prior_adjacency=self._prior_adj if include_priors else None,
            anchor_names=anchor_names,
            full_series_names=list(self._df_original.columns),
            anchor_mask=self._anchor_mask,
            series_metadata=self.series_metadata,
            prototype_weights=prototype_weights,
            global_prototype_weights=global_prototype_weights,
            decoded_top_trends=decoded_top_trends,
            dag_level=dag_level,
        )
        return snapshot.to_dict()

    def plot_graph(
        self,
        view: str = 'dag',
        threshold: float = None,
        max_edges: int = 50,
        show_priors: bool = False,
        ax=None,
        metadata_color_by: str = 'auto',
    ):
        """Plot the learned graph, hierarchy, or adjacency heatmap."""
        snapshot_dict = self.get_graph_snapshot(
            threshold=threshold,
            include_priors=show_priors,
        )
        snapshot = GraphSnapshot(
            node_table=snapshot_dict['node_table'],
            edge_table=snapshot_dict['edge_table'],
            adjacency_dense=np.asarray(
                snapshot_dict['adjacency_dense'], dtype=np.float32
            ),
            adjacency_thresholded=np.asarray(
                snapshot_dict['adjacency_thresholded'], dtype=np.float32
            ),
            assignment_matrices=[
                np.asarray(matrix, dtype=np.float32)
                for matrix in snapshot_dict['assignment_matrices']
            ],
            topological_order=snapshot_dict['topological_order'],
            prior_adjacency=(
                None
                if snapshot_dict.get('prior_adjacency') is None
                else np.asarray(snapshot_dict['prior_adjacency'], dtype=np.float32)
            ),
            is_acyclic=bool(snapshot_dict['is_acyclic']),
            cycle_score=float(snapshot_dict['cycle_score']),
            series_table=snapshot_dict.get('series_table', []),
            prototype_table=snapshot_dict.get('prototype_table', []),
            affinity_table=snapshot_dict.get('affinity_table', []),
            prototype_edge_table=snapshot_dict.get('prototype_edge_table', []),
        )
        return plot_graph_snapshot(
            snapshot=snapshot,
            view=view,
            max_edges=max_edges,
            show_priors=show_priors,
            ax=ax,
            metadata_color_by=metadata_color_by,
        )

    def get_edges(self) -> pd.DataFrame:
        """Return the discovered edge table over real series names (W-7).

        **These are predictive screening results, not causal claims.** An edge
        means "after removing the shared factors, this series' history helped
        predict that one's, stably across subsamples" — useful for screening,
        not evidence of mechanism. Series edges stay disabled for forecasting
        until they clear the false-positive/lag-recovery/follower-forecast
        gates together; the trustworthy structural output today is the
        factor -> series loading graph (``get_factor_graph`` /
        ``get_factor_diagnostics``'s ``loading_graph``).

        Columns: source, target, lag, sign, weight, family, stability,
        delta_mse. Empty DataFrame when discovery was disabled or found
        nothing. In trend_network='factor' mode the induced factor -> series
        bipartite edges (family 'factor') are appended; see get_factor_graph.
        """
        columns = [
            'source',
            'target',
            'lag',
            'sign',
            'weight',
            'family',
            'stability',
            'delta_mse',
        ]
        series_edges = (
            pd.DataFrame(self._discovery['edges'])[columns]
            if self._discovery is not None and self._discovery.get('edges')
            else pd.DataFrame(columns=columns)
        )
        factor_edges = self.get_factor_graph()
        if factor_edges.empty:
            return series_edges
        factor_edges = factor_edges.assign(stability=np.nan, delta_mse=np.nan)[columns]
        if series_edges.empty:
            return factor_edges.reset_index(drop=True)
        return pd.concat([series_edges, factor_edges], ignore_index=True)

    def get_factor_diagnostics(self) -> dict:
        """Everything the factor mode decided, in one JSON-safe dict.

        Returns:
            dict with the selected rank, factors/loadings, anchor/responder
            masks, stability scores, continuation choice, per-series blend
            weights, re-anchor alphas, gated series (by reason), the learned
            loading graph, validation scores and the coherence adjustment
            magnitude. Keys are absent rather than fabricated when the
            corresponding mechanism was not enabled.
        """
        info = getattr(self, '_factor_info', None) or {}
        safety = getattr(self, '_safety_info', None) or {}
        seasonal = getattr(self, '_seasonal_info', None) or {}
        columns = (
            list(self._df_original.columns) if self._df_original is not None else []
        )

        def names(indices):
            if indices is None:
                return []
            return [columns[i] for i in indices if 0 <= i < len(columns)]

        out = {
            'n_factors_fit': info.get('n_factors_fit'),
            'n_factors_live': info.get('n_factors_live'),
            'space': (self._factor_scale or {}).get('space', 'level'),
            'input_estimator': (self.factor_config or {}).get(
                'input_estimator', 'detector'
            ),
            'gated_series': names(info.get('gated_series')),
            'gated_trendless': names(info.get('gated_trendless')),
            'gated_frozen': names(info.get('gated_frozen')),
            'gated_underperforming': names(info.get('gated_underperforming')),
            'diagnostics': info.get('diagnostics'),
            'stage_a_val': info.get('stage_a_val'),
            'stage_b_loss': info.get('stage_b_loss'),
        }
        if info.get('continuation'):
            cont = info['continuation']
            out['continuation'] = {
                'choice': cont.get('choice'),
                'origins': cont.get('origins'),
                'score': cont.get('score'),
                'baseline_score': cont.get('baseline_score'),
            }
        for key in (
            'anchor_idx',
            'responder_idx',
            'insufficient_overlap',
            'observed_counts',
            'loading_graph',
            'group_assignment',
            'stability',
        ):
            if info.get(key) is not None:
                value = info[key]
                out[key] = (
                    names(value)
                    if key in ('anchor_idx', 'responder_idx', 'insufficient_overlap')
                    else value
                )
        if safety:
            out['safety'] = {
                'origins': safety.get('origins'),
                'blend_weights': dict(zip(columns, safety.get('blend_weights', [])))
                or None,
                'reanchor_alpha': dict(zip(columns, safety.get('reanchor_alpha', [])))
                or None,
                'capped_cells': safety.get('capped_cells'),
                'summary': safety.get('summary'),
            }
        if seasonal:
            out['seasonal_paths'] = {
                columns[int(i)]: name
                for i, name in (seasonal.get('choice') or {}).items()
                if 0 <= int(i) < len(columns)
            }
            out['seasonal_n_changed'] = seasonal.get('n_changed')
        coherence = getattr(self, '_coherence_info', None)
        if coherence:
            out['coherence'] = coherence
        return out

    def get_factors(self) -> dict:
        """Return named composite factors and their sparse signed loadings (W-7).

        Returns:
            Dict with 'factors' (DataFrame, one named column per factor, level
            space, indexed like the training data), 'loadings' (DataFrame,
            series x factor), and 'factor_names'. In trend_network='factor'
            mode these are the *learned* level-space paths, plus 'lags',
            'variance_share', 'phi' and 'diag'.
        """
        if self.trend_network_type == 'factor' and self._factor_network is not None:
            return self._learned_factor_tables()
        if self._discovery is None:
            return {'factors': None, 'loadings': None, 'factor_names': []}
        names = self._discovery['factor_names']
        index = self._components['trend'].index if self._components else None
        factors = pd.DataFrame(self._discovery['factors'], index=index, columns=names)
        loadings = pd.DataFrame(
            self._discovery['loadings'],
            index=list(self._df_original.columns),
            columns=names,
        )
        return {'factors': factors, 'loadings': loadings, 'factor_names': names}

    # ---- internal helpers ----

    def _learned_factor_tables(self) -> dict:
        """Learned factor paths, loadings, lags and diagnostics ('factor' mode)."""
        model = self._factor_network
        prune = self._factor_info['config']['prune_share']
        paths = model.fitted_factors(prune)
        loadings = model.fitted_loadings(prune)
        names = [f'factor_{i + 1}' for i in range(paths.shape[1])]
        index = self._components['trend'].index if self._components else None
        series = list(self._df_original.columns)
        keep = model.live_factors(prune)
        # C5: when a structure-only fit was run and cleared its agreement
        # guard, report the loadings the coherence graph will actually be
        # built on -- otherwise a diagnostic reads the forecast model's basis
        # and silently scores a different object than the shrink consumes.
        structure = getattr(self, '_structure_loadings', None)
        structure_frame = None
        if structure is not None and structure.shape[0] == len(series):
            structure_frame = pd.DataFrame(
                structure,
                index=series,
                columns=[f'factor_{i + 1}' for i in range(structure.shape[1])],
            )
        return {
            'factors': pd.DataFrame(paths, index=index, columns=names),
            'loadings': pd.DataFrame(loadings, index=series, columns=names),
            'structure_loadings': structure_frame,
            'structure_agreement': getattr(self, '_structure_agreement', None),
            'factor_names': names,
            'lags': pd.Series(model.fitted_lags(), index=series, name='lag'),
            'variance_share': pd.Series(
                model.variance_share()[keep], index=names, name='variance_share'
            ),
            'phi': [self._factor_info['diagnostics']['phi'][i] for i in keep],
            'diag': self._factor_info['diagnostics'],
        }

    def get_factor_graph(self) -> pd.DataFrame:
        """Induced factor -> series bipartite graph ('factor' mode).

        One row per (factor, series) pair with a non-negligible loading, giving
        the signed loading weight and the series' learned response lag. This is
        the structure the latent-factor mode actually uses to forecast, as
        opposed to the series->series edges from discovery.

        Returns:
            DataFrame with columns source, target, lag, sign, weight, family.
            Empty when not in 'factor' mode or not fitted.
        """
        columns = ['source', 'target', 'lag', 'sign', 'weight', 'family']
        if self.trend_network_type != 'factor' or self._factor_network is None:
            return pd.DataFrame(columns=columns)
        tables = self._learned_factor_tables()
        loadings = tables['loadings']
        lags = tables['lags']
        threshold = 0.05 * float(np.abs(loadings.values).max() or 1.0)
        rows = []
        for factor in loadings.columns:
            for series, weight in loadings[factor].items():
                if abs(weight) < threshold:
                    continue
                rows.append(
                    {
                        'source': factor,
                        'target': series,
                        'lag': int(lags[series]),
                        'sign': 1 if weight >= 0 else -1,
                        'weight': float(weight),
                        'family': 'factor',
                    }
                )
        rows.sort(key=lambda r: (r['source'], -abs(r['weight'])))
        return pd.DataFrame(rows, columns=columns)

    def _build_network_discovery_inputs(self) -> Optional[dict]:
        """Convert the named discovery result into network-ready buffers."""
        if self._discovery is None:
            return None
        name_to_idx = {c: i for i, c in enumerate(self._df_original.columns)}
        edges_indexed = []
        for edge in self._discovery.get('edges', []):
            src = name_to_idx.get(edge['source'])
            dst = name_to_idx.get(edge['target'])
            if src is None or dst is None:
                continue
            edges_indexed.append(
                {
                    'source': src,
                    'target': dst,
                    'lag': edge['lag'],
                    'sign': edge['sign'],
                    'weight': edge['weight'],
                    'family': edge['family'],
                }
            )
        communities = np.asarray(self._discovery.get('communities'))
        anchor_communities = None
        if communities.size and self._anchor_mask is not None:
            anchor_communities = communities[self._anchor_mask]
        families = sorted({e['family'] for e in edges_indexed}) or None
        return {
            'edges': edges_indexed,
            'families': families,
            'loadings': self._discovery.get('loadings'),
            'anchor_communities': anchor_communities,
            'adjacency': self._discovery.get('adjacency'),
        }

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
        x_np, _anchor = self._normalized_last_window(last_window)
        x = torch.tensor(x_np, dtype=torch.float32, device=device)

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

    def _compute_recent_reconciliation_residuals(
        self, S: np.ndarray, history_periods: int = None
    ) -> Optional[np.ndarray]:
        """Estimate recent one-step-ahead residuals for hierarchical weighting.

        Uses rolling windows from the fitted training history and compares the
        network's first-step predictions against the observed series values. The
        resulting residual matrix is expanded to include aggregate hierarchy nodes
        so MinT/ERM-style reconciliation can estimate a non-identity covariance.
        """
        if (
            self._network is None
            or self._components is None
            or self._df_original is None
        ):
            return None

        trend_data = self._components['trend'].values
        n_targets = max(trend_data.shape[0] - self.window_size, 0)
        if n_targets == 0:
            return None

        if history_periods is None:
            history_periods = min(max(int(self.forecast_horizon), 30), 90)
        try:
            history_periods = int(history_periods)
        except Exception:
            history_periods = 30

        history_periods = max(1, min(n_targets, history_periods))
        target_positions = np.arange(
            trend_data.shape[0] - history_periods, trend_data.shape[0]
        )
        scale = self._get_trend_scale()
        window_anchors = np.stack(
            [trend_data[pos - 1] for pos in target_positions], axis=0
        )  # (H, N)
        recent_windows = np.stack(
            [
                (
                    (trend_data[pos - self.window_size : pos] - trend_data[pos - 1])
                    / scale
                ).T
                for pos in target_positions
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        device = torch.device(self.device)
        x = torch.tensor(recent_windows, dtype=torch.float32, device=device)

        meta = None
        if (
            self._metadata_embeddings is not None
            and self._metadata_embeddings.shape[1] > 0
        ):
            meta = torch.tensor(
                np.repeat(
                    self._metadata_embeddings[np.newaxis, :, :],
                    history_periods,
                    axis=0,
                ),
                dtype=torch.float32,
                device=device,
            )

        anchor_mask_t = torch.tensor(self._anchor_mask, dtype=torch.bool, device=device)
        with torch.no_grad():
            outputs = self._network(x, meta, anchor_mask_t)

        # normalized one-step trend prediction; de-normalize with each window's anchor
        trend_step_norm = outputs['trend_forecast'][:, :, 0].cpu().numpy()  # (H, N)
        trend_step = trend_step_norm * scale[np.newaxis, :] + window_anchors
        seasonal_step = self._components['seasonality'].values[target_positions]
        holiday_step = self._components['holidays'].values[target_positions]
        level_shift_step = self._components['level_shifts'].values[target_positions]

        if isinstance(self._fusion_layer, nn.Module):
            with torch.no_grad():
                fused_norm = self._fusion_layer(
                    torch.tensor(
                        trend_step_norm[:, :, np.newaxis],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.tensor(
                        (seasonal_step / scale)[:, :, np.newaxis],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.tensor(
                        (holiday_step / scale)[:, :, np.newaxis],
                        dtype=torch.float32,
                        device=device,
                    ),
                ) + torch.tensor(
                    (level_shift_step / scale)[:, :, np.newaxis],
                    dtype=torch.float32,
                    device=device,
                )
                predicted_bottom = (
                    fused_norm.cpu().numpy()[:, :, 0] * scale[np.newaxis, :]
                    + window_anchors
                )
        else:
            predicted_bottom = (
                trend_step + seasonal_step + holiday_step + level_shift_step
            )

        actual_bottom = self._df_original.values[target_positions]
        bottom_residuals = actual_bottom - predicted_bottom

        n_bottom = S.shape[1]
        n_all = S.shape[0]
        n_agg = n_all - n_bottom
        if n_agg <= 0:
            return bottom_residuals

        aggregate_actual = (S[:n_agg] @ actual_bottom.T).T
        aggregate_predicted = (S[:n_agg] @ predicted_bottom.T).T
        aggregate_residuals = aggregate_actual - aggregate_predicted
        return np.concatenate([aggregate_residuals, bottom_residuals], axis=1)

    @staticmethod
    def _compute_trend_scale(trend_data: np.ndarray) -> np.ndarray:
        """Per-series robust scale of trend first differences (P1-1).

        Robust std (MAD * 1.4826) of first differences, falling back to plain
        std when the MAD is zero, floored at max(1e-6, 1e-4 * |median level|).
        """
        diffs = np.diff(trend_data, axis=0)
        if diffs.shape[0] == 0:
            diffs = np.zeros_like(trend_data)
        med = np.nanmedian(diffs, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(diffs - med), axis=0)
        std = np.nanstd(diffs, axis=0)
        scale = np.where(mad > 0, mad, std)
        floor = np.maximum(1e-6, 1e-4 * np.abs(np.nanmedian(trend_data, axis=0)))
        scale = np.maximum(np.nan_to_num(scale, nan=0.0), floor)
        return scale.astype(np.float32)

    def _get_trend_scale(self) -> np.ndarray:
        if self._trend_scale is None:
            n = self._components['trend'].shape[1]
            return np.ones(n, dtype=np.float32)
        return self._trend_scale

    def _normalized_last_window(self, window: np.ndarray) -> tuple:
        """Normalize a (T_window, N) trend window to network input space.

        Returns:
            x: (1, N, T_window) float32 array of (window - window[-1]) / scale.
            anchor: (N,) last raw trend value per series.
        """
        scale = self._get_trend_scale()
        anchor = window[-1]
        x = ((window - anchor) / scale).T[np.newaxis, :, :].astype(np.float32)
        return x, anchor

    def _create_windows(self, data: np.ndarray) -> tuple:
        """Create sliding windows and targets from (T, N) trend data.

        Windows and targets are window-local deltas: the per-window last value
        is subtracted (so the network is anchored at the last observed trend
        value) and each series is divided by its robust difference scale (P1-1).

        Returns:
            windows: (n_windows, N, window_size) — transposed for network input.
            targets: (n_windows, N, forecast_horizon)
        """
        T, N = data.shape
        total_len = self.window_size + self.forecast_horizon
        n_windows = max(T - total_len + 1, 0)

        if n_windows == 0:
            return np.array([]), np.array([])

        scale = self._get_trend_scale()
        windows = np.zeros((n_windows, N, self.window_size), dtype=np.float32)
        targets = np.zeros((n_windows, N, self.forecast_horizon), dtype=np.float32)

        for i in range(n_windows):
            anchor = data[i + self.window_size - 1]  # (N,)
            windows[i] = ((data[i : i + self.window_size] - anchor) / scale).T
            targets[i] = (
                (data[i + self.window_size : i + total_len] - anchor) / scale
            ).T

        return windows, targets

    def _create_target_windows(self, data: np.ndarray) -> np.ndarray:
        """Create target-only windows (aligned with _create_windows targets).

        Scaled per series but NOT centered — these components are already
        zero-mean; centering them would break OrthogonalityPenalty and the
        fusion MSE (P1-1).

        Returns:
            (n_windows, N, forecast_horizon)
        """
        T, N = data.shape
        total_len = self.window_size + self.forecast_horizon
        n_windows = max(T - total_len + 1, 0)

        if n_windows == 0:
            return np.array([])

        scale = self._get_trend_scale()
        targets = np.zeros((n_windows, N, self.forecast_horizon), dtype=np.float32)
        for i in range(n_windows):
            targets[i] = (data[i + self.window_size : i + total_len] / scale).T

        return targets

    def _calibrate_sigma(self, X_val, Y_val, meta_t, anchor_mask_t, n_val: int):
        """Per-series sigma calibration from the validation holdout (P3-2).

        r_j = std(validation residual_j) / mean(sigma_pred_j), computed in the
        normalized space (the scale cancels), clipped to a sane range. Applied
        multiplicatively to the predicted sigma at predict time.
        """
        self._sigma_calibration = None
        if n_val <= 1:
            return
        residuals = []
        sigmas = []
        with torch.no_grad():
            for start in range(0, n_val, self.batch_size):
                x_v = X_val[start : start + self.batch_size]
                y_v = Y_val[start : start + self.batch_size]
                meta_v = None
                if meta_t is not None:
                    meta_v = meta_t.expand(x_v.shape[0], -1, -1)
                out = self._network(x_v, meta_v, anchor_mask_t)
                sigma_v = out.get('sigma')
                if sigma_v is None:
                    return
                residuals.append((out['trend_forecast'] - y_v).cpu().numpy())
                sigmas.append(sigma_v.cpu().numpy())
        resid = np.concatenate(residuals, axis=0)  # (B, N, H)
        sig = np.concatenate(sigmas, axis=0)
        ratio = resid.std(axis=(0, 2)) / np.clip(sig.mean(axis=(0, 2)), 1e-8, None)
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=4.0, neginf=1.0)
        self._sigma_calibration = np.clip(ratio, 0.25, 4.0).astype(np.float32)

    def _compute_window_sample_weights(
        self, index: Optional[pd.Index]
    ) -> Optional[np.ndarray]:
        """Compute optional recency weights aligned to _create_windows output."""
        halflife_days = self.recency_halflife_days
        if halflife_days is None:
            return None
        try:
            halflife_days = float(halflife_days)
        except Exception:
            return None
        if not np.isfinite(halflife_days) or halflife_days <= 0:
            return None
        if index is None:
            return None

        total_len = self.window_size + self.forecast_horizon
        n_windows = max(len(index) - total_len + 1, 0)
        if n_windows <= 1:
            return None

        timestamp_index = pd.DatetimeIndex(index)
        target_end_index = timestamp_index[total_len - 1 :]
        if len(target_end_index) != n_windows:
            return None

        latest_timestamp = target_end_index.max()
        ages = (
            np.asarray(
                (latest_timestamp - target_end_index).total_seconds(),
                dtype=np.float64,
            )
            / 86400.0
        )
        ages = np.maximum(ages, 0.0)
        weights = np.power(0.5, ages / halflife_days)
        weights = np.clip(weights, 1e-8, None)
        weights = weights / weights.mean()
        return weights.astype(np.float32)

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
