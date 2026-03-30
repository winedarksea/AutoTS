# -*- coding: utf-8 -*-
"""
Digital Twin Fusion — Component Recombination.

Reconstructs full forecasts by combining trend network output with
seasonality, holidays, level shifts, and anomalies. Different variants
(attention-weighted or additive) serve as the final weaving mechanism.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


if HAS_TORCH:

    class DigitalTwinFusion(nn.Module):
        """Attention-gated fusion of stochastic decomposition components.

        Stacks trend, seasonality, and holidays as (B, N, T, C) and applies a
        small self-attention layer to let components inform each other's gates.
        Each component receives an independent sigmoid gate in [0, 1], allowing
        components to fade in and out per-timestep and per-series rather than
        competing through a fixed-sum softmax.

        Level shifts are structural/deterministic and are always added additively
        by the caller after this layer.

        Bias of output_proj is initialized to 2.0 so the initial gate values are
        sigmoid(2) ≈ 0.88 — close to pure additive at the start of training.

        Internal name: _weave_the_tapestry

        Args:
            n_components: Number of stochastic component channels (default 3:
                trend, seasonality, holidays).
            d_model: Internal attention dimension.
        """

        def __init__(self, n_components: int = 3, d_model: int = 32):
            super().__init__()
            self.n_components = n_components

            # per-component projection
            self.component_proj = nn.Linear(1, d_model)

            # small self-attention over components per timestep
            self._weave_the_tapestry = nn.MultiheadAttention(
                d_model, num_heads=2, batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)

            # output projection: one gate scalar per component
            self.output_proj = nn.Linear(d_model, 1)
            # initialize bias so sigmoid output starts near 1 (near-additive baseline)
            nn.init.constant_(self.output_proj.bias, 2.0)

        def forward(
            self,
            trend: torch.Tensor,
            seasonality: torch.Tensor,
            holidays: torch.Tensor,
            anomalies: torch.Tensor = None,
        ) -> torch.Tensor:
            """
            All inputs: (B, N, T). Level shifts are NOT passed here; the caller
            adds them additively after fusion.

            Returns:
                (B, N, T) gated sum of stochastic components.
            """
            components = [trend, seasonality, holidays]
            if anomalies is not None:
                components.append(anomalies)

            B, N, T = trend.shape
            C = len(components)

            # stack: (B, N, T, C)
            stacked = torch.stack(components, dim=-1)

            # reshape for attention: (B*N*T, C, 1)
            x = stacked.reshape(B * N * T, C, 1)

            # project to d_model: (B*N*T, C, D)
            x = F.gelu(self.component_proj(x))

            # self-attention over components: components can inform each other's gates
            attended, _ = self._weave_the_tapestry(x, x, x)
            x = self.norm(attended + x)

            # independent per-component gate in [0, 1] — sigmoid, not softmax.
            # Each component fades in (gate→1) or out (gate→0) independently;
            # all components can be fully active simultaneously.
            gates = torch.sigmoid(self.output_proj(x).squeeze(-1))  # (B*N*T, C)

            # gated sum of original component values
            original = stacked.reshape(B * N * T, C)  # (B*N*T, C)
            fused = (gates * original).sum(dim=-1)  # (B*N*T,)

            return fused.reshape(B, N, T)

    class DirectAttentionFusion(nn.Module):
        """Direct attention fusion of stochastic decomposition components.

        Each component is projected to a d_model embedding; self-attention lets
        components contextualize each other. The attended representation of each
        component is then projected directly to a scalar contribution and summed,
        with a residual from the original values.

        This is more directly attention-based than DigitalTwinFusion: the
        attention output itself determines what each component contributes,
        rather than being used to gate the original raw values. Fade-out is
        learned by the output projection producing a contribution that cancels
        the original value (contribution_i ≈ -original_i); fade-in is the
        default (zero-init means contributions start at 0, leaving the sum
        purely additive at initialization).

        Level shifts are structural/deterministic and are always added additively
        by the caller after this layer.

        Internal name: _direct_weave

        Args:
            n_components: Number of stochastic component channels (default 3:
                trend, seasonality, holidays).
            d_model: Internal attention dimension.
        """

        def __init__(self, n_components: int = 3, d_model: int = 32):
            super().__init__()
            self.n_components = n_components
            self.component_proj = nn.Linear(1, d_model)
            self._direct_weave = nn.MultiheadAttention(
                d_model, num_heads=2, batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
            # projects attended representation to a scalar contribution per component.
            # zero-initialized: at startup contributions = 0, so output = sum(originals)
            # (pure additive baseline). Training learns deviations from this.
            self.output_proj = nn.Linear(d_model, 1)
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        def forward(
            self,
            trend: torch.Tensor,
            seasonality: torch.Tensor,
            holidays: torch.Tensor,
            anomalies: torch.Tensor = None,
        ) -> torch.Tensor:
            """
            All inputs: (B, N, T). Level shifts are NOT passed here; the caller
            adds them additively after fusion.

            Returns:
                (B, N, T) attention-driven sum of stochastic components.
            """
            components = [trend, seasonality, holidays]
            if anomalies is not None:
                components.append(anomalies)

            B, N, T = trend.shape
            C = len(components)

            stacked = torch.stack(components, dim=-1)  # (B, N, T, C)
            original = stacked.reshape(B * N * T, C)  # (B*N*T, C)

            # project scalar components to d_model embeddings
            x = stacked.reshape(B * N * T, C, 1)
            x = F.gelu(self.component_proj(x))  # (B*N*T, C, D)

            # self-attention: each component's representation is contextualized
            # by all other components before producing its contribution
            attended, _ = self._direct_weave(x, x, x)  # (B*N*T, C, D)
            attended = self.norm(attended + x)  # residual in embedding space

            # project attended representation directly to scalar contribution.
            # unconstrained output: each component can contribute any signed amount.
            contributions = self.output_proj(attended).squeeze(-1)  # (B*N*T, C)

            # sum of (original + learned adjustment): pure additive at init,
            # diverges as training drives contributions toward optimal blend
            fused = (original + contributions).sum(dim=-1)  # (B*N*T,)

            return fused.reshape(B, N, T)

    class AdditiveFusion(nn.Module):
        """Simple additive baseline: sum of stochastic components.

        No learned parameters. Level shifts are added additively by the caller
        after this layer, consistent with DigitalTwinFusion and DirectAttentionFusion.
        """

        def forward(
            self,
            trend: torch.Tensor,
            seasonality: torch.Tensor,
            holidays: torch.Tensor,
            anomalies: torch.Tensor = None,
        ) -> torch.Tensor:
            """All inputs: (B, N, T). Returns (B, N, T)."""
            result = trend + seasonality + holidays
            if anomalies is not None:
                result = result + anomalies
            return result
