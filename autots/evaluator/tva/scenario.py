# -*- coding: utf-8 -*-
"""
BifrostOptimizer — What-If Scenario Planning.

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

    def apply_hierarchical_adjustment(
        self,
        level_name: str,
        target_value: float,
        hierarchy_matrix: np.ndarray = None,
    ) -> pd.DataFrame:
        """Set an aggregate level to target_value and propagate down proportionally.

        This is an **adjustment** (set to exactly target_value at every timestep),
        not a constraint (threshold/clamp that only fires when exceeded).

        The delta between target_value and the current aggregate sum is distributed
        to constituent bottom-level series proportional to their current share of
        that aggregate. When the current aggregate is near zero, the delta is split
        equally across constituent series.

        Args:
            level_name: Name of the aggregate node to adjust, matched against the
                last component of hierarchy path tuples (e.g. 'global', 'NA', 'EU').
                If multiple nodes share the same last component, the first
                (shallowest) match is used.
            target_value: Desired flat aggregate value applied across all forecast
                timesteps. The constituent series are scaled so their sum equals
                this value at every step.
            hierarchy_matrix: (L, M) summing matrix S. If None, built automatically
                from priors configured at fit time. Required when no series metadata
                was provided.

        Returns:
            Adjusted forecast DataFrame (same shape as predict()), with constituent
            series modified so their aggregate sum equals target_value.
        """
        base_forecast = self.tva.predict()
        columns = list(base_forecast.columns)
        N = len(columns)

        # resolveS matrix
        if hierarchy_matrix is not None:
            S = np.asarray(hierarchy_matrix, dtype=np.float64)
        elif self.tva._priors is not None:
            S = self.tva._priors.build_hierarchy_matrix().astype(np.float64)
        else:
            return base_forecast  # no hierarchy available

        n_bottom = S.shape[1]
        n_all = S.shape[0]
        n_agg = n_all - n_bottom

        if n_agg == 0 or n_bottom != N:
            # flat hierarchy, or S doesn't align with forecast columns
            return base_forecast

        # identify which aggregate row corresponds to level_name by rebuilding
        # the sorted aggregate node list used in build_hierarchy_matrix
        agg_row_idx = None
        if self.tva._priors is not None and self.tva._priors.series_metadata:
            paths = [
                m.hierarchy_path
                for m in self.tva._priors.series_metadata
                if m.hierarchy_path
            ]
            _aggregate_nodes = set()
            for path in paths:
                for depth in range(1, len(path)):
                    _aggregate_nodes.add(tuple(path[:depth]))
            _aggregate_nodes = sorted(_aggregate_nodes, key=lambda x: (len(x), x))

            for i, node in enumerate(_aggregate_nodes):
                if node[-1] == level_name or '/'.join(node) == level_name:
                    agg_row_idx = i
                    break

        if agg_row_idx is None:
            return base_forecast  # level_name not found in hierarchy

        # S row for this aggregate is a binary mask over bottom-level series
        constituent_mask = S[agg_row_idx, :].astype(bool)  # (M,)
        constituent_indices = np.where(constituent_mask)[0]

        if len(constituent_indices) == 0:
            return base_forecast

        # bottom forecasts: (T, N)
        adjusted_values = base_forecast.values.copy().astype(np.float64)

        # current aggregate per timestep
        current_agg = adjusted_values[:, constituent_indices].sum(axis=1)  # (T,)

        for t in range(len(current_agg)):
            delta = target_value - current_agg[t]
            total = current_agg[t]
            if abs(total) > 1e-10:
                # proportional to each series' share of the current aggregate
                shares = adjusted_values[t, constituent_indices] / total
            else:
                # equal split when aggregate is near zero
                shares = np.ones(len(constituent_indices)) / len(constituent_indices)
            adjusted_values[t, constituent_indices] += delta * shares

        return pd.DataFrame(adjusted_values, index=base_forecast.index, columns=columns)

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

        # freeze network, preserving original requires_grad state for each parameter
        original_grad_state = {
            param: param.requires_grad for param in network.parameters()
        }
        for param in network.parameters():
            param.requires_grad_(False)

        # get baseline input
        trend_data = self.tva._components['trend'].values
        last_window = trend_data[-self.tva.window_size :]
        x = torch.tensor(
            last_window.T[np.newaxis, :, :], dtype=torch.float32, device=device
        )

        meta = None
        if (
            self.tva._metadata_embeddings is not None
            and self.tva._metadata_embeddings.shape[1] > 0
        ):
            meta = torch.tensor(
                self.tva._metadata_embeddings[np.newaxis, :, :],
                dtype=torch.float32,
                device=device,
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
            reg_loss = (inv_var_t * perturbation**2).mean() * 10.0
            total_loss = total_loss + reg_loss

            total_loss.backward()
            optimizer.step()

        # restore original requires_grad state per parameter
        for param, grad_state in original_grad_state.items():
            param.requires_grad_(grad_state)

        # generate final adjusted forecast
        with torch.no_grad():
            final_input = x + perturbation
            final_outputs = network(final_input, meta, anchor_mask_t)
            trend_forecast = (
                final_outputs['trend_forecast'].cpu().numpy()[0]
            )  # (N, T_fc)

        # fuse with other components
        fc_length = self.tva.forecast_horizon
        forecast_comps = self.tva._decomposer.get_forecast_components(fc_length)

        seasonal = forecast_comps['seasonality'].values.T
        holidays = forecast_comps['holidays'].values.T
        level_shifts = forecast_comps['level_shifts'].values.T

        trend_fc = trend_forecast[:, :fc_length]

        if isinstance(self.tva._fusion_layer, nn.Module):
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
                fused = self.tva._fusion_layer(t_trend, t_sea, t_hol, t_ls)
                forecast_values = fused.cpu().numpy()[0].T
        else:
            forecast_values = (
                trend_fc
                + seasonal[:, :fc_length]
                + holidays[:, :fc_length]
                + level_shifts[:, :fc_length]
            ).T

        future_index = forecast_comps['trend'].index[:fc_length]
        return pd.DataFrame(
            forecast_values, index=future_index, columns=self.tva._df_original.columns
        )

    def _rainbow_bridge_strength(self) -> float:
        """Hidden: how much the bridge was perturbed."""
        return float(self.n_steps * self.lr)
