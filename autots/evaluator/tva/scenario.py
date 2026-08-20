# -*- coding: utf-8 -*-
"""
What-If Scenario Planning.

Two solvers for one problem — satisfy a user constraint while minimizing
disruption to the rest of the forecast.

:class:`BifrostOptimizer` freezes the trend network's weights and optimizes a
latent perturbation by gradient descent. It needs a network, so it serves
``trend_network='v1'`` and ``'v2'``.

:class:`ClosedFormScenario` serves ``'factor'`` and ``'none'``, where there is
no network. Those forecasts are linear in a few factor paths, so the
minimum-disruption update is a Gaussian conditioning solve against the
forecast covariance — deterministic, torch-free, and cross-series aware
through the same ``Sigma`` that weights MinT reconciliation.
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


# ---------------------------------------------------------------------------
# Shared, torch-free hierarchy helpers
# ---------------------------------------------------------------------------


def _resolve_hierarchy(tva, hierarchy_matrix=None):
    """(L, M) summing matrix from an explicit argument or the fitted priors."""
    if hierarchy_matrix is not None:
        return np.asarray(hierarchy_matrix, dtype=np.float64)
    if tva._priors is not None:
        return tva._priors.build_hierarchy_matrix().astype(np.float64)
    return None


def _aggregate_row_index(tva, level_name):
    """Row of S for ``level_name``, or None when the name is not a node.

    Rebuilds the sorted aggregate-node list ``build_hierarchy_matrix`` uses so
    the row order matches; matches on either the last path component or the
    full slash-joined path, shallowest first.
    """
    if tva._priors is None or not tva._priors.series_metadata:
        return None
    paths = [m.hierarchy_path for m in tva._priors.series_metadata if m.hierarchy_path]
    aggregate_nodes = set()
    for path in paths:
        for depth in range(1, len(path)):
            aggregate_nodes.add(tuple(path[:depth]))
    aggregate_nodes = sorted(aggregate_nodes, key=lambda x: (len(x), x))
    for i, node in enumerate(aggregate_nodes):
        if node[-1] == level_name or '/'.join(node) == level_name:
            return i
    return None


def _proportional_split(values, constituent_indices, target_value):
    """Set the constituent sum to ``target_value``, split by current share.

    The pre-covariance behaviour, kept as the fallback for when no forecast
    covariance is available. When the current aggregate is near zero the delta
    is split equally instead.
    """
    for t in range(values.shape[0]):
        total = values[t, constituent_indices].sum()
        delta = target_value - total
        if abs(total) > 1e-10:
            shares = values[t, constituent_indices] / total
        else:
            shares = np.ones(len(constituent_indices)) / len(constituent_indices)
        values[t, constituent_indices] += delta * shares
    return values


class BifrostOptimizer:
    """Inference-time optimizer for scenario planning and what-if analysis.

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

        S = _resolve_hierarchy(self.tva, hierarchy_matrix)
        if S is None:
            return base_forecast  # no hierarchy available

        n_bottom = S.shape[1]
        n_agg = S.shape[0] - n_bottom
        if n_agg == 0 or n_bottom != N:
            # flat hierarchy, or S doesn't align with forecast columns
            return base_forecast

        agg_row_idx = _aggregate_row_index(self.tva, level_name)
        if agg_row_idx is None:
            return base_forecast  # level_name not found in hierarchy

        # S row for this aggregate is a binary mask over bottom-level series
        constituent_indices = np.where(S[agg_row_idx, :].astype(bool))[0]
        if len(constituent_indices) == 0:
            return base_forecast

        adjusted_values = _proportional_split(
            base_forecast.values.copy().astype(np.float64),
            constituent_indices,
            target_value,
        )
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

        # get baseline input (normalized network space; see TVA P1-1)
        trend_data = self.tva._components['trend'].values
        last_window = trend_data[-self.tva.window_size :]
        x_np, anchor = self.tva._normalized_last_window(last_window)
        scale = self.tva._get_trend_scale()
        anchor_col = anchor[:, np.newaxis]
        scale_col = scale[:, np.newaxis]
        x = torch.tensor(x_np, dtype=torch.float32, device=device)

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
        # (in the same normalized space the perturbation lives in)
        trend_var = np.var(trend_data / scale, axis=0)  # (N,)
        trend_var = np.maximum(trend_var, 1e-8)
        # inverse variance as regularization weight (high variance = less penalty for perturbation)
        inv_var = 1.0 / trend_var
        inv_var = inv_var / inv_var.max()  # normalize
        inv_var_t = torch.tensor(
            inv_var[np.newaxis, :, np.newaxis], dtype=torch.float32, device=device
        )

        scale_t = torch.tensor(
            scale_col[np.newaxis], dtype=torch.float32, device=device
        )
        anchor_t = torch.tensor(
            anchor_col[np.newaxis], dtype=torch.float32, device=device
        )

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # forward with perturbation; constraints operate on RAW trend units
            perturbed_input = x + perturbation
            outputs = network(perturbed_input, meta, anchor_mask_t)
            forecast = outputs['trend_forecast'] * scale_t + anchor_t

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

        # generate final adjusted forecast (normalized network output)
        with torch.no_grad():
            final_input = x + perturbation
            final_outputs = network(final_input, meta, anchor_mask_t)
            trend_forecast_norm = (
                final_outputs['trend_forecast'].cpu().numpy()[0]
            )  # (N, T_fc)

        # fuse with other components — same normalized-space contract as
        # TVA.predict(): level shifts are additive AFTER fusion, never passed
        # as the anomalies argument.
        fc_length = self.tva.forecast_horizon
        forecast_comps = self.tva._decomposer.get_forecast_components(fc_length)

        seasonal = forecast_comps['seasonality'].values.T
        holidays = forecast_comps['holidays'].values.T
        level_shifts = forecast_comps['level_shifts'].values.T

        trend_fc_norm = trend_forecast_norm[:, :fc_length]

        if isinstance(self.tva._fusion_layer, nn.Module):
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
                fused_norm = self.tva._fusion_layer(t_trend, t_sea, t_hol) + t_ls
                forecast_values = (
                    fused_norm.cpu().numpy()[0] * scale_col + anchor_col
                ).T
        else:
            trend_fc = trend_fc_norm * scale_col + anchor_col
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


class ClosedFormScenario:
    """Covariance-weighted what-if solver for the factor and torch-free modes.

    :class:`BifrostOptimizer` backpropagates a perturbation through the trend
    network, which is unavailable in ``trend_network='factor'`` and ``'none'``
    (``tva._network is None`` there, so the optimizer dereferences None and
    raises). Those modes do not need backprop: their forecast is linear in a
    small set of factor paths, so any user constraint is a linear functional of
    the forecast vector and the minimum-disruption update has a closed form —
    the same Gaussian-conditioning / MinT-shaped solve::

        delta = Sigma A' (A Sigma A')^-1 (b - A y_hat)

    This is the minimum-``Sigma``-norm correction satisfying ``A(y+delta) = b``
    exactly, is deterministic, needs no optimizer steps, and reads the *same*
    covariance object as reconciliation — which is what makes the adjustment
    genuinely cross-series aware instead of a proportional top-down split.

    Non-factor components (seasonality, holidays, level shifts) need no special
    handling: they are additive offsets already inside the base forecast, so
    they enter only through the residual ``b - A y_hat``. Declared ratio
    identities are re-applied last, matching ``_predict_factor``'s ordering.

    The solve happens in raw forecast units. When the factor model was fit in
    log space, :meth:`TVA.forecast_covariance` has already carried that
    geometry across with a delta-method Jacobian; solving in raw units keeps
    linear aggregate constraints (which a log-space solve would not preserve)
    exactly satisfied.

    Args:
        tva_model: Fitted TVA instance.
        covariance: Optional precomputed (N, N) covariance. Fetched from
            ``tva_model.forecast_covariance()`` when omitted.
    """

    def __init__(self, tva_model, covariance: np.ndarray = None):
        self.tva = tva_model
        self._sigma = covariance
        self._sigma_resolved = covariance is not None
        self.covariance_info = None

    # -- covariance ---------------------------------------------------------

    def covariance(self, n_series: int = None):
        """(N, N) forecast covariance, or None when unavailable."""
        if not self._sigma_resolved:
            self._sigma_resolved = True
            result = self.tva.forecast_covariance()
            if result is not None:
                self._sigma, self.covariance_info = result
        sigma = self._sigma
        if sigma is None:
            return None
        sigma = np.asarray(sigma, dtype=np.float64)
        if n_series is not None and sigma.shape != (n_series, n_series):
            return None
        return sigma

    # -- core solve ---------------------------------------------------------

    def _solve(self, base_forecast: pd.DataFrame, constraints: list) -> pd.DataFrame:
        """Apply ``(timestep, a_vector, target)`` constraints in closed form.

        Constraints sharing a timestep are solved jointly, so a set of
        simultaneous requirements is satisfied together rather than in
        sequence.
        """
        values = base_forecast.values.astype(np.float64).copy()
        T, N = values.shape
        sigma = self.covariance(N)
        # Sigma = I is the honest degenerate case: with no cross-series
        # information the minimum-norm correction touches only the series the
        # constraint names, which is the pre-existing behaviour.
        S_use = np.eye(N) if sigma is None else sigma

        grouped = {}
        for timestep, a_vec, target in constraints:
            t = int(timestep) % T if T else 0
            grouped.setdefault(t, []).append(
                (np.asarray(a_vec, dtype=np.float64).ravel(), float(target))
            )

        for t, items in grouped.items():
            A = np.stack([item[0] for item in items])  # (m, N)
            b = np.array([item[1] for item in items])  # (m,)
            residual = b - A @ values[t]
            SA = S_use @ A.T  # (N, m)
            M = A @ SA  # (m, m)
            try:
                lam = np.linalg.solve(M, residual)
            except np.linalg.LinAlgError:
                lam = np.linalg.lstsq(M, residual, rcond=None)[0]
            values[t] = values[t] + SA @ lam

        result = pd.DataFrame(
            values, index=base_forecast.index, columns=base_forecast.columns
        )
        return self.tva._apply_derived_identities(result)

    # -- public API (mirrors BifrostOptimizer) ------------------------------

    def apply_constraint(
        self, series_name: str, timestep: int, target_value: float
    ) -> pd.DataFrame:
        """Pin a series at a timestep; other series move along ``Sigma``.

        Args:
            series_name: Column name of the series to constrain.
            timestep: Forecast timestep index (0-based).
            target_value: Desired value at that timestep.

        Returns:
            Adjusted forecast DataFrame for ALL series.
        """
        base = self.tva.predict()
        columns = list(base.columns)
        a = np.zeros(len(columns))
        a[columns.index(series_name)] = 1.0
        return self._solve(base, [(timestep, a, target_value)])

    def apply_growth_constraint(
        self, series_name: str, growth_rate: float
    ) -> pd.DataFrame:
        """Constrain a series to a growth rate over the forecast horizon.

        Args:
            series_name: Column name.
            growth_rate: Desired growth rate (e.g. 0.05 for 5% growth).

        Returns:
            Adjusted forecast DataFrame.
        """
        base = self.tva.predict()
        columns = list(base.columns)
        a = np.zeros(len(columns))
        a[columns.index(series_name)] = 1.0
        last_val = float(self.tva._df_original[series_name].iloc[-1])
        target_end = last_val * (1.0 + growth_rate)
        return self._solve(base, [(len(base) - 1, a, target_end)])

    def apply_hierarchical_adjustment(
        self,
        level_name: str,
        target_value: float,
        hierarchy_matrix: np.ndarray = None,
    ) -> pd.DataFrame:
        """Set an aggregate to ``target_value`` at every timestep.

        With a forecast covariance available the delta is distributed by the
        covariance-weighted solve — series that co-move with the rest of the
        aggregate absorb more of it than their raw share would suggest. Without
        one, this falls back to the proportional split
        :class:`BifrostOptimizer` performs.

        Args:
            level_name: Aggregate node name, matched against the last component
                of hierarchy path tuples (e.g. 'global', 'NA') or the full
                slash-joined path. Shallowest match wins.
            target_value: Desired flat aggregate value across all timesteps.
            hierarchy_matrix: (L, M) summing matrix S. Built from the fitted
                priors when omitted.

        Returns:
            Adjusted forecast DataFrame whose constituent series sum to
            ``target_value`` at every timestep.
        """
        base = self.tva.predict()
        N = base.shape[1]

        S = _resolve_hierarchy(self.tva, hierarchy_matrix)
        if S is None:
            return base
        n_bottom = S.shape[1]
        if S.shape[0] - n_bottom == 0 or n_bottom != N:
            return base

        agg_row_idx = _aggregate_row_index(self.tva, level_name)
        if agg_row_idx is None:
            return base
        a = np.asarray(S[agg_row_idx, :], dtype=np.float64)
        constituent_indices = np.where(a.astype(bool))[0]
        if len(constituent_indices) == 0:
            return base

        if self.covariance(N) is None:
            adjusted = _proportional_split(
                base.values.copy().astype(np.float64),
                constituent_indices,
                target_value,
            )
            return self.tva._apply_derived_identities(
                pd.DataFrame(adjusted, index=base.index, columns=base.columns)
            )

        return self._solve(base, [(t, a, target_value) for t in range(len(base))])
