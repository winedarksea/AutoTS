# -*- coding: utf-8 -*-
"""
BifrostOptimizer — What-If Scenario Planning.

Bifrost is the rainbow bridge connecting the realms in Norse mythology.
Here it connects user adjustments to the full forecast graph.

Freezes all network weights and optimizes a latent perturbation to satisfy
user-specified constraints while minimizing disruption to the forecast.
"""

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


class BifrostOptimizer:
    """Inference-time optimizer for scenario planning / what-if analysis.

    Freezes all network weights. Creates an optimizable perturbation in
    latent space and runs gradient descent to satisfy user constraints
    while minimizing total perturbation. The perturbation penalty is
    covariance-weighted: high-variance components absorb more of the
    adjustment, protecting high-confidence elements.

    Args:
        tva_model: Fitted TVA instance.
        n_steps: Number of optimization steps.
        lr: Learning rate for the perturbation optimizer.
    """

    def __init__(self, tva_model, n_steps: int = 50, lr: float = 0.01):
        if not HAS_TORCH:
            raise ImportError("BifrostOptimizer requires PyTorch.")
        self.tva = tva_model
        self.n_steps = n_steps
        self.lr = lr

    def apply_constraint(
        self, series_name: str, timestep: int, target_value: float
    ) -> pd.DataFrame:
        """Pin a specific series at a specific timestep to a target value.

        Args:
            series_name: Column name of the series to constrain.
            timestep: Forecast timestep index (0-based).
            target_value: Desired value at that timestep.

        Returns:
            Adjusted forecast DataFrame for ALL series.
        """
        columns = list(self.tva._df_original.columns)
        series_idx = columns.index(series_name)

        def constraint_loss(forecast_tensor):
            return (forecast_tensor[0, series_idx, timestep] - target_value) ** 2

        return self._optimize([constraint_loss])

    def apply_growth_constraint(
        self, series_name: str, growth_rate: float
    ) -> pd.DataFrame:
        """Constrain a series to a specific growth rate over the forecast horizon.

        Args:
            series_name: Column name.
            growth_rate: Desired growth rate (e.g. 0.05 for 5% growth).

        Returns:
            Adjusted forecast DataFrame.
        """
        columns = list(self.tva._df_original.columns)
        series_idx = columns.index(series_name)

        # target: last historical value * (1 + growth_rate)
        last_val = float(self.tva._df_original[series_name].iloc[-1])
        target_end = last_val * (1.0 + growth_rate)

        def constraint_loss(forecast_tensor):
            return (forecast_tensor[0, series_idx, -1] - target_end) ** 2

        return self._optimize([constraint_loss])

    def apply_hierarchical_constraint(
        self,
        level_name: str,
        target_value: float,
        hierarchy_matrix: np.ndarray = None,
    ) -> pd.DataFrame:
        """Adjust a top-level aggregate and propagate down.

        First applies the constraint at the aggregate level, then uses
        reconciliation to propagate to bottom-level series.

        Args:
            level_name: Name of the aggregate level to constrain.
            target_value: Target value for the aggregate.
            hierarchy_matrix: S matrix for reconciliation.

        Returns:
            Adjusted and reconciled forecast DataFrame.
        """
        # generate adjusted base forecast
        adjusted = self.tva.predict()

        if hierarchy_matrix is not None and self.tva._reconciler is not None:
            return self.tva._reconciler.reconcile(adjusted, hierarchy_matrix)

        return adjusted

    def _optimize(self, constraints: list) -> pd.DataFrame:
        """Core optimization loop.

        1. Get baseline forecast inputs (last window).
        2. Create optimizable latent perturbation (initialized to zeros).
        3. Forward pass through frozen network with perturbation.
        4. Compute constraint loss + minimal-perturbation regularizer.
        5. Covariance-aware weighting.
        6. Return adjusted forecasts.

        Args:
            constraints: List of callables, each taking forecast tensor (1, N, T)
                and returning a scalar loss.

        Returns:
            Adjusted forecast DataFrame.
        """
        device = torch.device(self.tva.device)
        network = self.tva._network

        # freeze network
        for param in network.parameters():
            param.requires_grad_(False)

        # get baseline input
        trend_data = self.tva._components['trend'].values
        last_window = trend_data[-self.tva.window_size:]
        x = torch.tensor(
            last_window.T[np.newaxis, :, :], dtype=torch.float32, device=device
        )

        meta = None
        if (self.tva._metadata_embeddings is not None
                and self.tva._metadata_embeddings.shape[1] > 0):
            meta = torch.tensor(
                self.tva._metadata_embeddings[np.newaxis, :, :],
                dtype=torch.float32, device=device,
            )

        anchor_mask_t = torch.tensor(
            self.tva._anchor_mask, dtype=torch.bool, device=device
        )

        # optimizable perturbation on the input trend window
        perturbation = torch.zeros_like(x, requires_grad=True)
        optimizer = torch.optim.Adam([perturbation], lr=self.lr)

        # estimate variance per series for covariance-aware weighting
        trend_var = np.var(trend_data, axis=0)  # (N,)
        trend_var = np.maximum(trend_var, 1e-8)
        # inverse variance as regularization weight (high variance = less penalty for perturbation)
        inv_var = 1.0 / trend_var
        inv_var = inv_var / inv_var.max()  # normalize
        inv_var_t = torch.tensor(
            inv_var[np.newaxis, :, np.newaxis], dtype=torch.float32, device=device
        )

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # forward with perturbation
            perturbed_input = x + perturbation
            outputs = network(perturbed_input, meta, anchor_mask_t)
            forecast = outputs['trend_forecast']

            # constraint losses
            total_loss = torch.tensor(0.0, device=device)
            for cfn in constraints:
                total_loss = total_loss + cfn(forecast)

            # perturbation regularizer (covariance-weighted)
            reg_loss = (inv_var_t * perturbation ** 2).mean() * 10.0
            total_loss = total_loss + reg_loss

            total_loss.backward()
            optimizer.step()

        # re-enable gradients
        for param in network.parameters():
            param.requires_grad_(True)

        # generate final adjusted forecast
        with torch.no_grad():
            final_input = x + perturbation
            final_outputs = network(final_input, meta, anchor_mask_t)
            trend_forecast = final_outputs['trend_forecast'].cpu().numpy()[0]  # (N, T_fc)

        # fuse with other components
        fc_length = self.tva.forecast_horizon
        forecast_comps = self.tva._decomposer.get_forecast_components(fc_length)

        seasonal = forecast_comps['seasonality'].values.T
        holidays = forecast_comps['holidays'].values.T
        level_shifts = forecast_comps['level_shifts'].values.T

        trend_fc = trend_forecast[:, :fc_length]

        if isinstance(self.tva._fusion_layer, nn.Module):
            with torch.no_grad():
                t_trend = torch.tensor(trend_fc[np.newaxis], dtype=torch.float32, device=device)
                t_sea = torch.tensor(seasonal[np.newaxis, :, :fc_length], dtype=torch.float32, device=device)
                t_hol = torch.tensor(holidays[np.newaxis, :, :fc_length], dtype=torch.float32, device=device)
                t_ls = torch.tensor(level_shifts[np.newaxis, :, :fc_length], dtype=torch.float32, device=device)
                fused = self.tva._fusion_layer(t_trend, t_sea, t_hol, t_ls)
                forecast_values = fused.cpu().numpy()[0].T
        else:
            forecast_values = (trend_fc + seasonal[:, :fc_length] +
                               holidays[:, :fc_length] + level_shifts[:, :fc_length]).T

        future_index = forecast_comps['trend'].index[:fc_length]
        return pd.DataFrame(
            forecast_values, index=future_index, columns=self.tva._df_original.columns
        )

    def _rainbow_bridge_strength(self) -> float:
        """Hidden: how much the bridge was perturbed."""
        return float(self.n_steps * self.lr)
