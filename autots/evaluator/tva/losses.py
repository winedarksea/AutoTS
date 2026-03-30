# -*- coding: utf-8 -*-
"""
TVA Loss Functions — The Allfather's Judgment.

Six loss components that together enforce coherent, high-quality composite trends.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


if HAS_TORCH:
    from autots.evaluator.tva.structure import (
        StructureLearningConfig,
        adjacency_sparsity_penalty,
        assignment_entropy_penalty,
        assignment_full_rank_penalty,
        dag_cycle_penalty,
    )

    class ForecastLoss(nn.Module):
        """Standard forecast loss on trend outputs.

        Supports 'mse', 'mae', and 'huber' base losses.
        """

        def __init__(self, base_loss: str = 'mse'):
            super().__init__()
            self.base_loss = base_loss

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """
            Args:
                pred: (B, N, T) predicted trend.
                target: (B, N, T) true trend.
            """
            if self.base_loss == 'mae':
                return F.l1_loss(pred, target)
            elif self.base_loss == 'huber':
                return F.smooth_l1_loss(pred, target)
            return F.mse_loss(pred, target)

    class OrthogonalityPenalty(nn.Module):
        """Penalizes correlation between predicted trend and deterministic components.

        Prevents seasonality, holidays, and anomalies from bleeding into the trend.
        Computes mean squared Pearson correlation across series and components.
        """

        def __init__(self, penalty_weight: float = 1.0):
            super().__init__()
            self.penalty_weight = penalty_weight

        def forward(
            self,
            pred_trend: torch.Tensor,
            seasonal: torch.Tensor = None,
            holidays: torch.Tensor = None,
            anomalies: torch.Tensor = None,
        ) -> torch.Tensor:
            """
            All inputs: (B, N, T). Returns scalar penalty.
            """
            penalty = torch.tensor(0.0, device=pred_trend.device)
            components = [c for c in [seasonal, holidays, anomalies] if c is not None]
            if not components:
                return penalty

            # flatten batch and series: (B*N, T)
            trend_flat = pred_trend.reshape(-1, pred_trend.shape[-1])
            for comp in components:
                comp_flat = comp.reshape(-1, comp.shape[-1])
                # pearson correlation per series
                t_centered = trend_flat - trend_flat.mean(dim=-1, keepdim=True)
                c_centered = comp_flat - comp_flat.mean(dim=-1, keepdim=True)
                num = (t_centered * c_centered).sum(dim=-1)
                den = (t_centered.norm(dim=-1) * c_centered.norm(dim=-1)).clamp(
                    min=1e-8
                )
                corr = num / den
                penalty = penalty + (corr**2).mean()

            return self.penalty_weight * penalty

    class LocalTrendPenalty(nn.Module):
        """Forces individual series trends to rely on shared composite nodes.

        Penalizes the magnitude of deviation between per-series trend and
        the trend reconstructed purely from shared prototypes.

        Internal name: _variant_pruning_penalty
        """

        def __init__(self, penalty_weight: float = 1.0):
            super().__init__()
            self.penalty_weight = penalty_weight

        def forward(
            self, per_series_trend: torch.Tensor, composite_trend: torch.Tensor
        ) -> torch.Tensor:
            """
            Args:
                per_series_trend: (B, N, T) actual per-series trend output.
                composite_trend: (B, N, T) trend reconstructed from prototypes only.
            """
            return self.penalty_weight * F.mse_loss(per_series_trend, composite_trend)

    class SmoothnessPenalty(nn.Module):
        """Encourages derived composite tracks toward smoothness.

        Penalizes second-order differences (curvature) of the composite trend,
        reinforcing their role as slow-moving macro indicators.
        """

        def __init__(self, penalty_weight: float = 0.1):
            super().__init__()
            self.penalty_weight = penalty_weight

        def forward(self, composite_trend: torch.Tensor) -> torch.Tensor:
            """
            Args:
                composite_trend: (B, K, T) or (B, N_latent, T) composite trends.
            """
            if composite_trend.shape[-1] < 3:
                return torch.tensor(0.0, device=composite_trend.device)
            # second-order differences
            d2 = (
                composite_trend[..., 2:]
                - 2 * composite_trend[..., 1:-1]
                + composite_trend[..., :-2]
            )
            return self.penalty_weight * (d2**2).mean()

    class SoftPriorLoss(nn.Module):
        """Keeps learned adjacency near the event-cluster / business-prior structure.

        Uses MSE between learned and prior adjacency matrices, scaled by prior_confidence.

        Internal name: _sacred_timeline_divergence
        """

        def __init__(self, penalty_weight: float = 0.5):
            super().__init__()
            self.penalty_weight = penalty_weight

        def forward(
            self,
            learned_adjacency: torch.Tensor,
            prior_adjacency: torch.Tensor,
            prior_confidence: float = 1.0,
        ) -> torch.Tensor:
            """
            Args:
                learned_adjacency: (M, M) learned graph edges.
                prior_adjacency: (M, M) prior graph edges (may be series-level,
                    larger than latent-space adjacency in V2).
                prior_confidence: Runtime multiplier for optional prior strength.
            """
            if prior_confidence <= 0:
                return torch.tensor(0.0, device=learned_adjacency.device)
            # learned adjacency is (n_global, n_global); prior may be (N, N).
            # When sizes differ, resize prior via interpolation rather than
            # broadcasting incorrectly, or skip if ambiguous.
            if learned_adjacency.shape != prior_adjacency.shape:
                m = learned_adjacency.shape[0]
                if prior_adjacency.ndim == 2 and m > 0:
                    # resize prior to latent space by adaptive avg pooling
                    # unsqueeze to (1, 1, M_prior, M_prior) for interpolation
                    prior_resized = (
                        F.interpolate(
                            prior_adjacency.unsqueeze(0).unsqueeze(0).float(),
                            size=(m, m),
                            mode='bilinear',
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    prior_adjacency = prior_resized
                else:
                    return torch.tensor(0.0, device=learned_adjacency.device)
            return (
                self.penalty_weight
                * float(prior_confidence)
                * F.mse_loss(learned_adjacency, prior_adjacency)
            )

    class ProbabilisticLoss(nn.Module):
        """Gaussian negative log-likelihood on V2 probabilistic output head.

        Uses mu and sigma emitted by CompositeTrendNetworkV2 to compute NLL
        over the trend targets, teaching the network to calibrate its uncertainty.
        Sigma is kept positive by Softplus in the network head.

        Default weight is low (0.2) because calibration is secondary to the
        structural accuracy goals of the other loss components.
        """

        def __init__(self, penalty_weight: float = 0.2):
            super().__init__()
            self.penalty_weight = penalty_weight

        def forward(
            self,
            mu: torch.Tensor,
            sigma: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                mu: (B, N, T) predicted mean — same as trend_forecast.
                sigma: (B, N, T) predicted std dev, strictly positive.
                target: (B, N, T) true trend values.

            Returns:
                Scalar Gaussian NLL (mean over all elements).
            """
            # Gaussian NLL = log(sigma) + 0.5 * ((target - mu) / sigma)^2
            # Constant 0.5 * log(2π) omitted — does not affect gradients.
            sigma = sigma.clamp(min=1e-6)
            nll = torch.log(sigma) + 0.5 * ((target - mu) / sigma) ** 2
            return self.penalty_weight * nll.mean()

    class CrossSeriesCoherenceLoss(nn.Module):
        """Ensures related metrics produce consistent macro direction.

        THE MAIN VALUE PROPOSITION of TVA.

        For each prototype, computes the consensus trend direction (growth/decline)
        among all series weighted by that prototype. Penalizes series whose slope
        disagrees with the weighted consensus. Uses soft sign (tanh) for
        differentiability.

        The penalty is scaled by |consensus| so that it only enforces consistency
        when there is a genuine shared direction signal. When a prototype group is
        heterogeneous (mixed growth/decline), consensus ≈ 0 and the penalty
        vanishes, preventing the degenerate case where the loss pulls all series
        toward zero slope (flat forecast). Series with a strong shared driver
        (consensus near ±1) are held tightly together; genuinely independent
        series are left to follow their own data-driven history.
        """

        def __init__(self, penalty_weight: float = 2.0, temperature: float = 1.0):
            super().__init__()
            self.penalty_weight = penalty_weight
            self.temperature = temperature

        def _trend_direction(self, x: torch.Tensor) -> torch.Tensor:
            """Multi-scale soft direction signal.

            Averages tanh-slope across three windows (full, first-half,
            second-half) so the signal captures acceleration and deceleration,
            not just net endpoint displacement.

            Args:
                x: (..., T) trend tensor.
            Returns:
                (...) direction signal in [-1, 1].
            """
            T = x.shape[-1]
            mid = max(T // 2, 1)
            s_full = (x[..., -1] - x[..., 0]) / T
            s_first = (x[..., mid] - x[..., 0]) / mid
            s_second = (x[..., -1] - x[..., mid]) / max(T - mid, 1)
            t = self.temperature
            return (
                torch.tanh(s_full / t)
                + torch.tanh(s_first / t)
                + torch.tanh(s_second / t)
            ) / 3.0

        def forward(
            self,
            trend_forecasts: torch.Tensor,
            prototype_weights: torch.Tensor,
            composite_per_series: torch.Tensor = None,
        ) -> torch.Tensor:
            """
            Args:
                trend_forecasts: (B, N, T) per-series trend forecasts.
                prototype_weights: (B, N, K) how much each series uses each prototype.
                composite_per_series: (B, N, T) optional trend reconstructed purely
                    from shared prototypes (composite_trend_per_series from network
                    output). When provided, each series is compared against its own
                    structural anchor instead of an emergent group consensus. The
                    penalty is gated by anchor_dir.abs() so a flat composite anchor
                    never drives the model toward flat forecasts.
            """
            B, N, T = trend_forecasts.shape
            K = prototype_weights.shape[-1]

            if T < 2:
                return torch.tensor(0.0, device=trend_forecasts.device)

            soft_dir = self._trend_direction(trend_forecasts)  # (B, N)

            # --- Composite-anchored mode ---
            # When composite_per_series is available it is the structurally derived
            # "what the shared driver says this series should do" signal.  Each
            # series is compared against its own anchor rather than an emergent
            # group average, which avoids the consensus-toward-flat collapse and
            # is grounded in the network's latent structure rather than shape
            # similarity.
            if (
                composite_per_series is not None
                and composite_per_series.shape == trend_forecasts.shape
            ):
                # Detach the composite anchor so gradients only flow through
                # trend_forecasts here.  LocalTrendPenalty already trains the
                # composite (via MSE); allowing gradients through anchor_strength
                # would let the optimizer collapse _sacred_timeline_prototypes → 0
                # to zero-out the gate — the same degenerate flat solution.
                anchor_dir = self._trend_direction(
                    composite_per_series.detach()
                )  # (B, N)
                # Gate by anchor strength: when the composite is near-flat
                # (anchor_dir ≈ 0) the penalty vanishes, preventing the degenerate
                # case where a flat anchor actively suppresses trend diversity.
                # Only a composite with a genuine directional signal enforces
                # alignment — same contract as consensus_strength in the fallback.
                anchor_strength = anchor_dir.abs()  # (B, N), no gradient
                # Also gate by assignment confidence: how committed each series is
                # to any one prototype.  A series spread uniformly across prototypes
                # has no clear structural home and gets a lower penalty.
                assignment_confidence = prototype_weights.max(dim=-1).values  # (B, N)
                deviation = (soft_dir - anchor_dir) ** 2  # (B, N)
                penalty = (anchor_strength * assignment_confidence * deviation).mean()
                return self.penalty_weight * penalty

            # --- Fallback: prototype-consensus mode ---
            # Original approach, retained for V1 / cases where composite_per_series
            # is not available.  Uses multi-scale direction signal in place of the
            # previous endpoint-only slope.
            penalty = torch.tensor(0.0, device=trend_forecasts.device)
            for k in range(K):
                w_k = prototype_weights[:, :, k]  # (B, N)
                w_sum = w_k.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                consensus = (w_k * soft_dir).sum(dim=-1, keepdim=True) / w_sum
                consensus_strength = consensus.abs()
                deviation = (soft_dir - consensus) ** 2
                weighted_dev = (w_k * deviation).sum(dim=-1)
                penalty = (
                    penalty + (consensus_strength.squeeze(-1) * weighted_dev).mean()
                )
            return self.penalty_weight * penalty / max(K, 1)

    class TemporalLossComposite(nn.Module):
        """Combines all TVA losses with configurable weights.

        Internal name: _allfather_judgment

        Args:
            weights: Dict overriding default weights for each loss component.
                Keys: 'forecast', 'orthogonality', 'local_trend', 'smoothness',
                      'soft_prior', 'causal_prior', 'coherence'.
            base_forecast_loss: 'mse', 'mae', or 'huber'.
        """

        def __init__(
            self,
            weights: dict = None,
            base_forecast_loss: str = 'mse',
            structure_config: dict = None,
        ):
            super().__init__()
            w = weights or {}
            self.structure_config = StructureLearningConfig.from_dict(structure_config)
            self.forecast_loss = ForecastLoss(base_loss=base_forecast_loss)
            self.orthogonality = OrthogonalityPenalty(w.get('orthogonality', 1.0))
            self.local_trend = LocalTrendPenalty(w.get('local_trend', 1.0))
            self.smoothness = SmoothnessPenalty(w.get('smoothness', 0.1))
            self.soft_prior = SoftPriorLoss(w.get('soft_prior', 0.5))
            self.causal_prior = SoftPriorLoss(
                w.get('causal_prior', w.get('soft_prior', 0.5))
            )
            self.structure_prior = SoftPriorLoss(1.0)
            self.coherence = CrossSeriesCoherenceLoss(w.get('coherence', 2.0))
            self.probabilistic = ProbabilisticLoss(w.get('probabilistic', 0.2))
            self._forecast_weight = w.get('forecast', 1.0)

        def forward(self, outputs: dict, targets: dict) -> tuple:
            """
            Args:
                outputs: Dict from CompositeTrendNetwork.forward():
                    - 'trend_forecast': (B, N, T) — same as 'mu' for V2.
                    - 'mu': (B, N, T) V2 only; used for probabilistic loss alongside sigma.
                    - 'sigma': (B, N, T) V2 only; predicted std dev for NLL.
                    - 'prototype_weights': (B, N, K) or (B, n_global, K)
                    - 'composite_trend': (B, K, T) or (B, n_latent, T)
                    - 'adjacency': (M, M) (V2 only, optional)
                targets: Dict with:
                    - 'true_trend': (B, N, T)
                    - 'seasonal': (B, N, T) optional
                    - 'holidays': (B, N, T) optional
                    - 'anomalies': (B, N, T) optional
                    - 'prior_adjacency': (M, M) optional
                    - 'prior_confidence': scalar multiplier for optional priors

            Returns:
                (total_loss, breakdown_dict)
            """
            device = outputs['trend_forecast'].device
            breakdown = {}
            prior_confidence = float(targets.get('prior_confidence', 1.0))
            structure_loss_scale = float(targets.get('structure_loss_scale', 0.0))
            structure_config = targets.get('structure_config', self.structure_config)
            structure_prior_weight = float(
                targets.get(
                    'structure_prior_weight',
                    getattr(structure_config, 'prior_tether_weight', 0.0),
                )
            )

            # forecast loss
            loss_fc = self._forecast_weight * self.forecast_loss(
                outputs['trend_forecast'], targets['true_trend']
            )
            breakdown['forecast'] = loss_fc.item()
            total = loss_fc

            # orthogonality
            loss_orth = self.orthogonality(
                outputs['trend_forecast'],
                targets.get('seasonal'),
                targets.get('holidays'),
                targets.get('anomalies'),
            )
            breakdown['orthogonality'] = loss_orth.item()
            total = total + loss_orth

            # local trend penalty
            if 'composite_trend_per_series' in outputs:
                loss_lt = self.local_trend(
                    outputs['trend_forecast'], outputs['composite_trend_per_series']
                )
                breakdown['local_trend'] = loss_lt.item()
                total = total + loss_lt

            # smoothness
            if 'composite_trend' in outputs:
                loss_sm = self.smoothness(outputs['composite_trend'])
                breakdown['smoothness'] = loss_sm.item()
                total = total + loss_sm

            # soft prior (V2 only)
            if (
                'adjacency' in outputs
                and 'prior_adjacency' in targets
                and not outputs.get('structure_mode', False)
            ):
                prior = targets['prior_adjacency']
                if not isinstance(prior, torch.Tensor):
                    prior = torch.tensor(prior, dtype=torch.float32, device=device)
                loss_sp = self.soft_prior(
                    outputs['adjacency'], prior, prior_confidence=prior_confidence
                )
                breakdown['soft_prior'] = loss_sp.item()
                total = total + loss_sp

            # causal prior (V2 only)
            if 'adjacency' in outputs and outputs.get('causal_prior') is not None:
                causal_prior = outputs['causal_prior']
                if not isinstance(causal_prior, torch.Tensor):
                    causal_prior = torch.tensor(
                        causal_prior, dtype=torch.float32, device=device
                    )
                loss_cp = self.causal_prior(
                    outputs['adjacency'],
                    causal_prior,
                    prior_confidence=prior_confidence,
                )
                breakdown['causal_prior'] = loss_cp.item()
                total = total + loss_cp

            if (
                structure_loss_scale > 0
                and 'adjacency' in outputs
                and outputs.get('structure_mode', False)
            ):
                if getattr(structure_config, 'learn_dag', True):
                    loss_dag = (
                        structure_loss_scale
                        * float(getattr(structure_config, 'dag_penalty', 0.0))
                        * dag_cycle_penalty(outputs['adjacency'])
                    )
                    breakdown['dag'] = loss_dag.item()
                    total = total + loss_dag

                    loss_sparse = (
                        structure_loss_scale
                        * float(getattr(structure_config, 'sparsity_weight', 0.0))
                        * adjacency_sparsity_penalty(outputs['adjacency'])
                    )
                    breakdown['structure_sparsity'] = loss_sparse.item()
                    total = total + loss_sparse

                assignments = outputs.get('assignment_matrices') or []
                if assignments:
                    loss_entropy = (
                        structure_loss_scale
                        * float(
                            getattr(structure_config, 'assignment_entropy_weight', 0.0)
                        )
                        * assignment_entropy_penalty(assignments)
                    )
                    breakdown['assignment_entropy'] = loss_entropy.item()
                    total = total + loss_entropy

                    loss_rank = (
                        structure_loss_scale
                        * float(
                            getattr(
                                structure_config, 'assignment_full_rank_weight', 0.0
                            )
                        )
                        * assignment_full_rank_penalty(assignments)
                    )
                    breakdown['assignment_full_rank'] = loss_rank.item()
                    total = total + loss_rank

                if (
                    outputs.get('structure_prior') is not None
                    and structure_prior_weight > 0
                ):
                    loss_structure_prior = structure_loss_scale * self.structure_prior(
                        outputs['adjacency'],
                        outputs['structure_prior'],
                        prior_confidence=structure_prior_weight,
                    )
                    breakdown['structure_prior'] = loss_structure_prior.item()
                    total = total + loss_structure_prior

                drift_value = outputs.get('assignment_drift')
                if drift_value is not None:
                    if not isinstance(drift_value, torch.Tensor):
                        drift_value = torch.tensor(
                            drift_value, dtype=torch.float32, device=device
                        )
                    loss_drift = (
                        structure_loss_scale
                        * float(getattr(structure_config, 'temporal_drift_weight', 0.0))
                        * drift_value.mean()
                    )
                    breakdown['assignment_drift'] = loss_drift.item()
                    total = total + loss_drift

            # coherence
            if 'prototype_weights' in outputs:
                loss_ch = self.coherence(
                    outputs['trend_forecast'],
                    outputs['prototype_weights'],
                    outputs.get('composite_trend_per_series'),
                )
                breakdown['coherence'] = loss_ch.item()
                total = total + loss_ch

            # probabilistic loss (V2 only — requires both mu and sigma)
            if 'mu' in outputs and 'sigma' in outputs:
                loss_prob = self.probabilistic(
                    outputs['mu'], outputs['sigma'], targets['true_trend']
                )
                breakdown['probabilistic'] = loss_prob.item()
                total = total + loss_prob

            breakdown['total'] = total.item()
            return total, breakdown

        def _allfather_judgment(self):
            """Hidden: the Allfather sees all losses."""
            return {
                name: getattr(mod, 'penalty_weight', 1.0)
                for name, mod in self.named_modules()
                if name
            }
