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
        """Attention-weighted fusion of decomposition components.

        Stacks components as (B, N, T, C) and applies a small attention layer
        to learn dynamic per-timestep per-series component weighting, rather
        than naive additive or multiplicative combination.

        Internal name: _weave_the_tapestry

        Args:
            n_components: Number of component channels (default 4: trend,
                seasonality, holidays, level_shifts).
            d_model: Internal attention dimension.
        """

        def __init__(self, n_components: int = 4, d_model: int = 32):
            super().__init__()
            self.n_components = n_components

            # per-component projection
            self.component_proj = nn.Linear(1, d_model)

            # small self-attention over components per timestep
            self._weave_the_tapestry = nn.MultiheadAttention(
                d_model, num_heads=2, batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)

            # output projection back to scalar
            self.output_proj = nn.Linear(d_model, 1)

        def forward(
            self,
            trend: torch.Tensor,
            seasonality: torch.Tensor,
            holidays: torch.Tensor,
            level_shifts: torch.Tensor,
            anomalies: torch.Tensor = None,
        ) -> torch.Tensor:
            """
            All inputs: (B, N, T).

            Returns:
                (B, N, T) reconstructed full forecast.
            """
            components = [trend, seasonality, holidays, level_shifts]
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

            # self-attention over components
            attended, _ = self._weave_the_tapestry(x, x, x)
            x = self.norm(attended + x)

            # output: (B*N*T, C, 1)
            weights = self.output_proj(x).squeeze(-1)  # (B*N*T, C)
            weights = F.softmax(weights, dim=-1)  # component weights

            # weighted sum of original component values
            original = stacked.reshape(B * N * T, C)  # (B*N*T, C)
            fused = (weights * original).sum(dim=-1)  # (B*N*T,)

            return fused.reshape(B, N, T)

    class AdditiveFusion(nn.Module):
        """Simple additive baseline: sum of all components.

        No learned parameters. For baseline comparison.
        """

        def forward(
            self,
            trend: torch.Tensor,
            seasonality: torch.Tensor,
            holidays: torch.Tensor,
            level_shifts: torch.Tensor,
            anomalies: torch.Tensor = None,
        ) -> torch.Tensor:
            """All inputs: (B, N, T). Returns (B, N, T)."""
            result = trend + seasonality + holidays + level_shifts
            if anomalies is not None:
                result = result + anomalies
            return result
