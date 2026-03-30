# -*- coding: utf-8 -*-
"""
Composite Trend Network — The Sacred Timeline.

V1: Hierarchical Latent Trend Graph (deterministic structure).
V2: Learned Directed Composite Graph (learned adjacency, adaptive prototypes).

The trend network operates only on isolated trend components, not raw series.
It compresses per-series trends into a small set of shared composite prototypes
and reconstructs per-series trend forecasts that are structurally consistent
with those shared states.
"""

import math
import warnings
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

if HAS_TORCH:
    from autots.evaluator.tva.structure import (
        DirectedGraphLearner,
        DynamicHierarchyLearner,
        StructureLearningConfig,
    )

    def _resize_graph_prior_to_square_numpy(
        graph_prior: np.ndarray,
        target_size: int,
        prior_name: str,
    ) -> np.ndarray:
        """Resize an optional square prior matrix into latent graph size."""
        if graph_prior is None:
            return None
        graph_prior = np.asarray(graph_prior, dtype=np.float32)
        if graph_prior.ndim != 2 or graph_prior.shape[0] != graph_prior.shape[1]:
            warnings.warn(
                f"{prior_name} shape {graph_prior.shape} is not square. "
                f"Using a zero prior of shape ({target_size}, {target_size}) instead.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros((target_size, target_size), dtype=np.float32)
        if graph_prior.shape != (target_size, target_size):
            warnings.warn(
                f"{prior_name} shape {graph_prior.shape} does not match "
                f"target_size={target_size}. The prior will be bilinearly interpolated "
                f"to ({target_size}, {target_size}).",
                UserWarning,
                stacklevel=2,
            )
            try:
                graph_prior = (
                    F.interpolate(
                        torch.tensor(graph_prior, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(0),
                        size=(target_size, target_size),
                        mode='bilinear',
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .numpy()
                )
            except Exception:
                graph_prior = np.zeros((target_size, target_size), dtype=np.float32)
        return graph_prior.astype(np.float32)

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for temporal sequences."""

        def __init__(self, d_model: int, max_len: int = 2000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
            self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, D)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1)]

    class TrendTokenizer(nn.Module):
        """Converts each series' trend window into a channel token.

        1D convolution over the time dimension per channel, followed by
        positional encoding and optional metadata concatenation.

        Args:
            window_size: Length of input trend window.
            d_token: Output token dimension.
            d_meta: Dimension of metadata embeddings to concatenate (0 = none).
        """

        def __init__(self, window_size: int, d_token: int, d_meta: int = 0):
            super().__init__()
            self.d_token = d_token
            self.d_meta = d_meta
            d_conv_out = d_token - d_meta if d_meta > 0 else d_token

            # temporal conv: treats each series independently
            self.conv = nn.Conv1d(
                1, d_conv_out, kernel_size=min(7, window_size), padding=3
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.norm = nn.LayerNorm(d_token)

        def forward(
            self, trends: torch.Tensor, metadata: torch.Tensor = None
        ) -> torch.Tensor:
            """
            Args:
                trends: (B, N, T_window) trend values per series.
                metadata: (B, N, D_meta) static metadata embeddings, optional.

            Returns:
                tokens: (B, N, D_token)
            """
            B, N, T = trends.shape
            # process each series through 1D conv
            x = trends.reshape(B * N, 1, T)  # (B*N, 1, T)
            x = F.gelu(self.conv(x))  # (B*N, D_conv, T)
            x = self.pool(x).squeeze(-1)  # (B*N, D_conv)
            x = x.reshape(B, N, -1)  # (B, N, D_conv)

            if metadata is not None and self.d_meta > 0:
                x = torch.cat([x, metadata], dim=-1)  # (B, N, D_token)

            return self.norm(x)

    class HierarchicalLatentEncoder(nn.Module):
        """U-Cast style 2-level latent query encoder.

        Level 1 (meso): cross-attention from learned meso queries to input tokens.
        Level 2 (global): cross-attention from learned global queries to meso latents.

        The _bifrost_bridge connects levels via cross-attention.

        Args:
            d_token: Token dimension.
            n_meso: Number of meso-level latent nodes.
            n_global: Number of global-level latent nodes.
            n_heads: Number of attention heads.
        """

        def __init__(self, d_token: int, n_meso: int, n_global: int, n_heads: int = 4):
            super().__init__()
            self.n_meso = n_meso
            self.n_global = n_global

            # learned latent query vectors
            self.meso_queries = nn.Parameter(torch.randn(1, n_meso, d_token) * 0.02)
            self.global_queries = nn.Parameter(torch.randn(1, n_global, d_token) * 0.02)

            # cross-attention layers
            self._bifrost_bridge_meso = nn.MultiheadAttention(
                d_token, n_heads, batch_first=True
            )
            self._bifrost_bridge_global = nn.MultiheadAttention(
                d_token, n_heads, batch_first=True
            )

            self.norm_meso = nn.LayerNorm(d_token)
            self.norm_global = nn.LayerNorm(d_token)

        def forward(self, tokens: torch.Tensor) -> tuple:
            """
            Args:
                tokens: (B, N, D) input channel tokens.

            Returns:
                meso_latent: (B, n_meso, D)
                global_latent: (B, n_global, D)
                skip: tokens (B, N, D) for skip connection in decoder
            """
            B = tokens.shape[0]

            # level 1: tokens -> meso
            meso_q = self.meso_queries.expand(B, -1, -1)
            meso_latent, _ = self._bifrost_bridge_meso(meso_q, tokens, tokens)
            meso_latent = self.norm_meso(meso_latent)

            # level 2: meso -> global
            global_q = self.global_queries.expand(B, -1, -1)
            global_latent, _ = self._bifrost_bridge_global(
                global_q, meso_latent, meso_latent
            )
            global_latent = self.norm_global(global_latent)

            return meso_latent, global_latent, tokens

    class MaskedSparseAttention(nn.Module):
        """Sparse attention over latent nodes with a connections mask.

        In V1, the mask is fixed from priors. In V2, it becomes learnable.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
        """

        def __init__(self, d_model: int, n_heads: int = 4):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        def forward(
            self, latent: torch.Tensor, mask: torch.Tensor = None
        ) -> torch.Tensor:
            """
            Args:
                latent: (B, M, D) latent nodes.
                mask: (M, M) attention mask. 0 = attend, -inf = block.

            Returns:
                (B, M, D) refined latent nodes.
            """
            out, _ = self.attn(latent, latent, latent, attn_mask=mask)
            return out + latent  # residual

    class PrototypeBottleneck(nn.Module):
        """Projects global latent through a bank of prototype trend signatures.

        K prototypes, each a D-dimensional vector. The global latent is assigned
        to prototypes with a configurable scoring function (distance or projection),
        producing usage weights and prototype-conditioned output.

        Internal name: _sacred_timeline_prototypes

        Args:
            d_latent: Latent dimension.
            n_prototypes: Number of prototypes (K).
            assignment_method: Strategy for prototype logits.
                - 'cosine' (default): cosine similarity in metric space.
                - 'l2': negative squared Euclidean distance.
                - 'linear': legacy learned projection.
            assignment_temperature: Temperature divisor for assignment logits.
        """

        def __init__(
            self,
            d_latent: int,
            n_prototypes: int,
            assignment_method: str = 'cosine',
            assignment_temperature: float = 1.0,
        ):
            super().__init__()
            self.n_prototypes = n_prototypes
            self.assignment_method = str(assignment_method).lower()
            self.assignment_temperature = max(float(assignment_temperature), 1e-6)
            self._sacred_timeline_prototypes = nn.Parameter(
                torch.randn(n_prototypes, d_latent) * 0.02
            )
            self.project = nn.Linear(d_latent, n_prototypes)
            supported = {'cosine', 'l2', 'linear'}
            if self.assignment_method not in supported:
                raise ValueError(
                    f"Unsupported prototype assignment_method='{assignment_method}'. "
                    f"Supported values: {sorted(supported)}"
                )

        def compute_logits(self, latent: torch.Tensor) -> torch.Tensor:
            """Compute prototype assignment logits for latent tokens.

            Args:
                latent: (B, M, D) latent vectors to assign to K prototypes.
            Returns:
                logits: (B, M, K) unnormalized assignment scores.
            """
            if self.assignment_method == 'linear':
                return self.project(latent)

            prototypes = self._sacred_timeline_prototypes  # (K, D)
            if self.assignment_method == 'cosine':
                latent_unit = F.normalize(latent, p=2, dim=-1)
                proto_unit = F.normalize(prototypes, p=2, dim=-1)
                logits = torch.matmul(latent_unit, proto_unit.transpose(0, 1))
                return logits / self.assignment_temperature

            # Negative squared distance yields higher logits for closer prototypes.
            # latent: (B, M, 1, D), prototypes: (1, 1, K, D)
            diffs = latent.unsqueeze(-2) - prototypes.unsqueeze(0).unsqueeze(0)
            sq_l2 = (diffs**2).sum(dim=-1)
            logits = -sq_l2
            return logits / self.assignment_temperature

        def forward(self, global_latent: torch.Tensor) -> tuple:
            """
            Args:
                global_latent: (B, n_global, D)

            Returns:
                conditioned: (B, n_global, D) prototype-conditioned latent.
                usage_weights: (B, n_global, K) soft assignment to prototypes.
            """
            # compute assignment scores to prototypes
            logits = self.compute_logits(global_latent)  # (B, n_global, K)
            usage_weights = F.softmax(logits, dim=-1)  # (B, n_global, K)

            # weighted combination of prototypes
            # (B, n_global, K) @ (K, D) -> (B, n_global, D)
            proto_repr = torch.matmul(usage_weights, self._sacred_timeline_prototypes)

            # combine with original via residual
            conditioned = global_latent + proto_repr
            return conditioned, usage_weights

    class HierarchicalDecoder(nn.Module):
        """Decodes from prototype-conditioned latents back to per-series trend forecasts.

        Reverse of encoder: global -> meso -> per-series, with skip connections.

        Args:
            d_token: Token dimension.
            n_meso: Number of meso latent nodes.
            forecast_horizon: Length of output forecast.
            n_heads: Number of attention heads.
        """

        def __init__(
            self, d_token: int, n_meso: int, forecast_horizon: int, n_heads: int = 4
        ):
            super().__init__()
            self.forecast_horizon = forecast_horizon

            # global -> meso upsampling via cross-attention
            self.upsample_to_meso = nn.MultiheadAttention(
                d_token, n_heads, batch_first=True
            )
            self.norm_meso = nn.LayerNorm(d_token)

            # meso -> per-series upsampling via cross-attention
            self.upsample_to_series = nn.MultiheadAttention(
                d_token, n_heads, batch_first=True
            )
            self.norm_series = nn.LayerNorm(d_token)

            # project token to forecast horizon
            self.forecast_head = nn.Linear(d_token, forecast_horizon)

        def forward(
            self,
            global_latent: torch.Tensor,
            meso_latent: torch.Tensor,
            skip: torch.Tensor,
            anchor_mask: torch.Tensor = None,
        ) -> tuple:
            """
            Args:
                global_latent: (B, n_global, D) prototype-conditioned.
                meso_latent: (B, n_meso, D) from encoder.
                skip: (B, N, D) original tokens for skip connection.
                anchor_mask: (N,) bool, True for anchor series (optional).

            Returns:
                anchor_forecasts: (B, N_anchor, T_forecast) or (B, N, T_forecast)
                composite_trend: (B, n_global, T_forecast) global composite trends
            """
            B = global_latent.shape[0]

            # global -> meso
            meso_upsampled, _ = self.upsample_to_meso(
                meso_latent, global_latent, global_latent
            )
            meso_upsampled = self.norm_meso(meso_upsampled + meso_latent)

            # select anchor tokens for upsampling target
            if anchor_mask is not None:
                skip_anchors = skip[:, anchor_mask]
            else:
                skip_anchors = skip

            # meso -> per-series
            series_tokens, _ = self.upsample_to_series(
                skip_anchors, meso_upsampled, meso_upsampled
            )
            series_tokens = self.norm_series(series_tokens + skip_anchors)

            # project to forecast
            anchor_forecasts = self.forecast_head(
                series_tokens
            )  # (B, N_anchor, T_forecast)

            # composite trends from global latent
            composite_trend = self.forecast_head(
                global_latent
            )  # (B, n_global, T_forecast)

            return anchor_forecasts, composite_trend

    class ResponderHead(nn.Module):
        """Attaches short-history series to derived composite trends.

        Uses cross-attention from responder queries to anchor trend outputs,
        inheriting macro trend direction without perturbing core geometry.

        Args:
            d_token: Token dimension.
            forecast_horizon: Length of output forecast.
            n_heads: Number of attention heads.
        """

        def __init__(self, d_token: int, forecast_horizon: int, n_heads: int = 2):
            super().__init__()
            self.cross_attn = nn.MultiheadAttention(d_token, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(d_token)
            self.forecast_head = nn.Linear(d_token, forecast_horizon)

        def forward(
            self, anchor_tokens: torch.Tensor, responder_tokens: torch.Tensor
        ) -> torch.Tensor:
            """
            Args:
                anchor_tokens: (B, N_anchor, D) anchor series tokens.
                responder_tokens: (B, N_responder, D) responder series tokens.

            Returns:
                (B, N_responder, T_forecast) responder trend forecasts.
            """
            attended, _ = self.cross_attn(
                responder_tokens, anchor_tokens, anchor_tokens
            )
            out = self.norm(attended + responder_tokens)
            return self.forecast_head(out)

    class CompositeTrendNetworkV1(nn.Module):
        """V1: Hierarchical Latent Trend Graph — deterministic structure.

        Full pipeline: tokenize -> encode -> sparse attention -> prototype
        bottleneck -> decode -> responder heads.

        Args:
            n_series: Total number of series.
            window_size: Input trend window length.
            forecast_horizon: Output forecast length.
            d_token: Token/latent dimension.
            n_meso: Number of meso latent nodes.
            n_global: Number of global latent nodes.
            n_prototypes: Number of prototype trend signatures.
            n_heads: Attention heads.
            d_meta: Metadata embedding dimension (0 = none).
            prior_adjacency: Optional (M, M) prior adjacency for sparse attention mask.
            prototype_assignment_method: Prototype assignment function.
            prototype_assignment_temperature: Temperature for assignment logits.
        """

        def __init__(
            self,
            n_series: int,
            window_size: int,
            forecast_horizon: int,
            d_token: int = 64,
            n_meso: int = 8,
            n_global: int = 4,
            n_prototypes: int = 4,
            n_heads: int = 4,
            d_meta: int = 0,
            prior_adjacency: np.ndarray = None,
            prototype_assignment_method: str = 'cosine',
            prototype_assignment_temperature: float = 1.0,
        ):
            super().__init__()
            self.n_series = n_series
            self.window_size = window_size
            self.forecast_horizon = forecast_horizon
            self.n_global = n_global
            self.n_prototypes = n_prototypes

            self.tokenizer = TrendTokenizer(window_size, d_token, d_meta)
            self.encoder = HierarchicalLatentEncoder(d_token, n_meso, n_global, n_heads)
            self.sparse_attn = MaskedSparseAttention(d_token, n_heads)
            self.prototype = PrototypeBottleneck(
                d_token,
                n_prototypes,
                assignment_method=prototype_assignment_method,
                assignment_temperature=prototype_assignment_temperature,
            )
            self.decoder = HierarchicalDecoder(
                d_token, n_meso, forecast_horizon, n_heads
            )
            self.responder = ResponderHead(
                d_token, forecast_horizon, max(n_heads // 2, 1)
            )

            # build fixed attention mask from prior adjacency
            self._register_attention_mask(prior_adjacency)

        def _register_attention_mask(self, prior_adjacency: np.ndarray = None):
            """Convert prior adjacency to attention mask for sparse attention.

            In V1 this is a fixed buffer. In V2 it becomes a parameter.
            """
            if prior_adjacency is not None:
                prior_adjacency = _resize_graph_prior_to_square_numpy(
                    prior_adjacency,
                    target_size=self.n_global,
                    prior_name='V1 prior_adjacency',
                )
                # threshold to binary and convert to attention mask format
                # 0 = attend, -inf = block
                binary = (prior_adjacency > 0.1).astype(np.float32)
                # Always preserve self-attention to avoid fully-masked rows.
                np.fill_diagonal(binary, 1.0)
                mask = np.where(binary > 0, 0.0, float('-inf'))
            else:
                # no mask = full attention over global latent nodes
                mask = np.zeros((self.n_global, self.n_global), dtype=np.float32)

            self.register_buffer('_attn_mask', torch.tensor(mask, dtype=torch.float32))

        def forward(
            self,
            trend_input: torch.Tensor,
            metadata: torch.Tensor = None,
            anchor_mask: torch.Tensor = None,
        ) -> dict:
            """
            Args:
                trend_input: (B, N, T_window) trend components for all series.
                metadata: (B, N, D_meta) static metadata embeddings, optional.
                anchor_mask: (N,) bool tensor, True for anchor (long-history) series.

            Returns dict:
                'trend_forecast': (B, N, T_forecast) per-series trend forecasts.
                'prototype_weights': (B, N, K) prototype usage per series.
                'composite_trend': (B, n_global, T_forecast) global composite trends.
                'composite_trend_per_series': (B, N, T_forecast) composite-only reconstruction.
            """
            B, N, T = trend_input.shape

            # tokenize
            tokens = self.tokenizer(trend_input, metadata)  # (B, N, D)

            # separate anchors for encoder input
            if anchor_mask is not None:
                anchor_tokens = tokens[:, anchor_mask]
            else:
                anchor_tokens = tokens
                anchor_mask_bool = None

            # encode
            meso, glob, skip = self.encoder(anchor_tokens)

            # sparse attention over global latents
            mask = self._attn_mask
            if mask.shape != (glob.shape[1], glob.shape[1]):
                mask = torch.zeros(
                    glob.shape[1],
                    glob.shape[1],
                    device=glob.device,
                    dtype=glob.dtype,
                )
            glob = self.sparse_attn(glob, mask)

            # prototype bottleneck
            glob_conditioned, usage_weights_global = self.prototype(glob)

            # decode to anchor forecasts
            anchor_forecasts, composite_trend = self.decoder(
                glob_conditioned,
                meso,
                skip if anchor_mask is None else tokens[:, anchor_mask],
                anchor_mask=None,  # already filtered
            )

            # responder forecasts for short-history series
            if anchor_mask is not None and not anchor_mask.all():
                responder_mask = ~anchor_mask
                responder_tokens = tokens[:, responder_mask]
                responder_forecasts = self.responder(
                    tokens[:, anchor_mask], responder_tokens
                )
                # reassemble full forecast
                trend_forecast = torch.zeros(
                    B, N, self.forecast_horizon, device=trend_input.device
                )
                trend_forecast[:, anchor_mask] = anchor_forecasts
                trend_forecast[:, responder_mask] = responder_forecasts
            else:
                trend_forecast = anchor_forecasts

            # compute per-series prototype weights by projecting tokens through prototype
            proto_logits = self.prototype.compute_logits(tokens)  # (B, N, K)
            prototype_weights = F.softmax(proto_logits, dim=-1)

            # composite-only reconstruction per series (for local trend penalty)
            # each series' "composite" trend = weighted sum of global composite trends
            # prototype_weights: (B, N, K), we need to map K prototypes to n_global composite trends
            # simplified: use prototype weights as the mapping
            # composite_trend: (B, n_global, T_forecast)
            # map via global usage weights averaged
            with torch.no_grad():
                avg_global_usage = usage_weights_global.mean(dim=1)  # (B, K)
            # composite_trend_per_series approximation
            composite_per_series = torch.matmul(
                prototype_weights, self.prototype._sacred_timeline_prototypes
            )  # (B, N, D)
            composite_per_series = self.decoder.forecast_head(composite_per_series)

            return {
                'trend_forecast': trend_forecast,
                'prototype_weights': prototype_weights,
                'composite_trend': composite_trend,
                'composite_trend_per_series': composite_per_series,
            }

    class CompositeTrendNetworkV2(CompositeTrendNetworkV1):
        """V2: Learned Directed Composite Graph.

        Extends V1 with:
        - Learned adjacency matrix (initialized from priors, regularized by SoftPriorLoss).
        - Adaptive prototypes with regime gating.
        - Probabilistic output head (mu, sigma).
        - Incremental responder-to-anchor promotion.

        "I know what kind of God I Need to be."

        Args:
            causal_prior: (M, M) causal discovery prior for soft edge regularization.
            All other args passed to V1.
        """

        def __init__(self, *args, causal_prior: np.ndarray = None, **kwargs):
            prior_adj = kwargs.get('prior_adjacency', None)
            n_anchor_series = int(
                kwargs.pop('n_anchor_series', kwargs.get('n_series', 1))
            )
            structure_learning_config = kwargs.pop('structure_learning_config', None)
            super().__init__(*args, **kwargs)
            self.structure_config = StructureLearningConfig.from_dict(
                structure_learning_config
            )
            self.n_anchor_series = max(int(n_anchor_series), 1)

            top_latent_size = self.n_global
            self.dynamic_hierarchy = None
            self.structure_level_sizes = []
            if self.structure_config.enabled and self.structure_config.learn_hierarchy:
                self.dynamic_hierarchy = DynamicHierarchyLearner(
                    n_anchor=self.n_anchor_series,
                    d_token=self.tokenizer.d_token,
                    config=self.structure_config,
                )
                self.structure_level_sizes = list(self.dynamic_hierarchy.level_sizes)
                top_latent_size = int(self.structure_level_sizes[-1])

            resized_prior_adj = None
            if prior_adj is not None:
                resized_prior_adj = _resize_graph_prior_to_square_numpy(
                    prior_adj,
                    target_size=top_latent_size,
                    prior_name='V2 prior_adjacency',
                )
            self.graph_learner = DirectedGraphLearner(
                n_nodes=top_latent_size,
                prior_adjacency=resized_prior_adj,
            )
            self.register_buffer(
                '_structure_prior',
                (
                    None
                    if resized_prior_adj is None
                    else torch.tensor(resized_prior_adj, dtype=torch.float32)
                ),
            )

            # causal prior regularizes learned edges as a soft target.
            causal_prior_tensor = None
            if causal_prior is not None:
                causal_prior_tensor = self._resize_graph_prior_to_latent_space(
                    causal_prior,
                    n_latent=top_latent_size,
                    prior_name='causal_prior',
                )
            self.register_buffer('_causal_prior', causal_prior_tensor)

            # adaptive prototypes: regime gating
            self._regime_gate = nn.Sequential(
                nn.Linear(self.tokenizer.d_token, self.n_prototypes),
                nn.Softmax(dim=-1),
            )

            # probabilistic output head
            self._sigma_head = nn.Linear(self.tokenizer.d_token, self.forecast_horizon)
            self._sigma_activation = nn.Softplus()

        @property
        def learned_adjacency(self) -> torch.Tensor:
            """Learned adjacency as a sigmoid-normalized matrix."""
            return self.graph_learner.adjacency

        @staticmethod
        def _resize_graph_prior_to_latent_space(
            graph_prior: np.ndarray, n_latent: int, prior_name: str
        ) -> torch.Tensor:
            """Map optional graph priors into the latent adjacency size."""
            graph_prior = _resize_graph_prior_to_square_numpy(
                graph_prior,
                target_size=n_latent,
                prior_name=f'V2 {prior_name}',
            )
            return torch.tensor(graph_prior, dtype=torch.float32)

        def _register_attention_mask(self, prior_adjacency=None):
            """Override: in V2 the mask is derived from learned adjacency at forward time."""
            # register a dummy buffer; actual mask computed in forward
            self.register_buffer(
                '_attn_mask',
                torch.zeros(self.n_global, self.n_global, dtype=torch.float32),
            )

        def forward(self, trend_input, metadata=None, anchor_mask=None) -> dict:
            """Same interface as V1, plus structure outputs when enabled."""
            B, N, T = trend_input.shape

            # tokenize
            tokens = self.tokenizer(trend_input, metadata)

            if anchor_mask is not None:
                anchor_tokens = tokens[:, anchor_mask]
            else:
                anchor_tokens = tokens

            structure_assignments = []
            assignment_drift = torch.tensor(0.0, device=trend_input.device)
            if self.dynamic_hierarchy is not None:
                hierarchy_outputs = self.dynamic_hierarchy(anchor_tokens)
                structure_assignments = hierarchy_outputs['assignment_matrices']
                structure_dynamic_assignments = hierarchy_outputs.get(
                    'dynamic_assignment_matrices',
                    [],
                )
                assignment_drift = hierarchy_outputs.get(
                    'assignment_drift',
                    assignment_drift,
                )
                latent_levels = hierarchy_outputs['levels']
                glob = hierarchy_outputs['top_latent']
                glob = self.sparse_attn(glob, self.graph_learner.attention_mask())
                glob_conditioned, usage_weights_global = self.prototype(glob)
                regime_weights = self._regime_gate(
                    glob_conditioned.mean(dim=1, keepdim=True)
                )
                usage_weights_global = usage_weights_global * regime_weights
                decoded_levels = self.dynamic_hierarchy.decode(
                    glob_conditioned,
                    latent_levels,
                    assignment_matrices=(
                        structure_dynamic_assignments
                        if structure_dynamic_assignments
                        else structure_assignments
                    ),
                )
                anchor_forecasts = self.decoder.forecast_head(decoded_levels[0])
                composite_trend = self.decoder.forecast_head(glob_conditioned)
            else:
                meso, glob, skip = self.encoder(anchor_tokens)
                glob = self.sparse_attn(glob, self.graph_learner.attention_mask())
                glob_conditioned, usage_weights_global = self.prototype(glob)
                regime_weights = self._regime_gate(
                    glob_conditioned.mean(dim=1, keepdim=True)
                )
                usage_weights_global = usage_weights_global * regime_weights
                anchor_forecasts, composite_trend = self.decoder(
                    glob_conditioned,
                    meso,
                    skip if anchor_mask is None else tokens[:, anchor_mask],
                )

            # responder
            if anchor_mask is not None and not anchor_mask.all():
                responder_mask = ~anchor_mask
                responder_tokens = tokens[:, responder_mask]
                responder_forecasts = self.responder(
                    tokens[:, anchor_mask], responder_tokens
                )
                trend_forecast = torch.zeros(
                    B, N, self.forecast_horizon, device=trend_input.device
                )
                trend_forecast[:, anchor_mask] = anchor_forecasts
                trend_forecast[:, responder_mask] = responder_forecasts
            else:
                trend_forecast = anchor_forecasts
                responder_mask = None

            # prototype weights per series
            proto_logits = self.prototype.compute_logits(tokens)
            prototype_weights = F.softmax(proto_logits, dim=-1)

            # composite per series
            composite_per_series = torch.matmul(
                prototype_weights, self.prototype._sacred_timeline_prototypes
            )
            composite_per_series = self.decoder.forecast_head(composite_per_series)

            # probabilistic output: mu = trend_forecast, sigma from tokens
            mu = trend_forecast
            sigma = self._sigma_activation(self._sigma_head(tokens))

            return {
                'trend_forecast': mu,
                'mu': mu,
                'sigma': sigma,
                'prototype_weights': prototype_weights,
                'composite_trend': composite_trend,
                'composite_trend_per_series': composite_per_series,
                'adjacency': self.learned_adjacency,
                'structure_mode': bool(self.structure_config.enabled),
                'assignment_matrices': structure_assignments,
                'assignment_drift': assignment_drift,
                'structure_prior': self._structure_prior,
                'causal_prior': self._causal_prior,
            }

        def promote_responder(self, series_idx: int):
            """Promote a responder series to anchor status.

            This is a metadata-level operation; the actual anchor_mask is
            managed externally and passed to forward(). This method exists
            as a hook for incremental graph updates.
            """
            pass  # anchor_mask management is external to the network

        def _glorious_purpose(self):
            """Hidden: every variant has a glorious purpose."""
            return self.learned_adjacency.detach().cpu().numpy()
