# -*- coding: utf-8 -*-
"""AutoTS wrapper for TVA — Time Variant Architecture deep learning forecaster."""

import random
import datetime
import numpy as np
import pandas as pd
import warnings

from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability


class TVAModel(ModelObject):
    """Time Variant Architecture (TVA) — deep learning multivariate forecasting.

    Routes all series trends through shared composite prototypes in a learned
    directed graph, producing structurally coherent cross-series forecasts.
    Wraps autots.evaluator.tva.TVA for use within AutoTS search.

    Requires PyTorch. Multivariate — uses shared information across all series.

    Args:
        name: String to identify class.
        frequency: String alias of datetime index frequency or 'infer'.
        prediction_interval: Confidence interval for probabilistic forecast.
        forecast_length: Number of periods to forecast (sets TVA forecast_horizon).
        n_factors: latent factors for trend_network='factor' ('auto' default).
        factor_config: overrides for factor_network.DEFAULT_FACTOR_CONFIG.
        coherence_config: post-forecast coherence-shrink settings (None = off).
        derived_definitions: declared ratio identities {column: (num, den)}
            used for post-reconciliation; never inferred from column names.
        factor_knot_spacing: trend-filter knot spacing in 'factor' mode.
        factor_max_lag: max learned response lag in 'factor' mode (0 = off).
        trend_network: 'v2' (learned directed graph, default), 'factor'
            (learned latent-factor trend), 'v1'
            (hierarchical latent), or 'none' (torch-free damped rolling trend
            + factor/edge discovery + MinT + residual-sigma intervals).
        fusion: 'attention' (DigitalTwinFusion) or 'additive' (AdditiveFusion).
        d_token: Token/latent dimension.
        n_meso: Meso latent width, integer or 'auto'.
        n_global: Global latent width, integer or 'auto'.
        n_prototypes: Number of shared trend prototypes, integer or 'auto'.
        n_heads: Number of attention heads.
        epochs: Training epochs.
        lr: Learning rate for AdamW optimizer.
        batch_size: Training batch size.
        window_size: Input trend window length (must be <= training length - forecast_length).
        series_metadata: Optional list of SeriesMetadata (or plain dicts with
            the same keys, so it round-trips as JSON) describing each series'
            attributes and hierarchy path. A real hierarchy here is what lets
            the MinT reconciliation auto-enable.
        prior_adjacency: Optional graph prior. Accepts an (N, N) matrix, a
            labelled DataFrame, a list of ``{'source', 'target', 'weight'}``
            edge dicts, or a list of name groups (each group a clique). See
            ``tva.priors.coerce_prior_adjacency``.
        causal_prior: Optional directed prior in the same forms, kept as its
            own edge family.
        prior_confidence: Weight of prior adjacency matrix (0=ignore priors, 1=rigid).
            Also the merge weight the prior edge families get in discovery,
            against 1.0 for the data-derived families.
        prototype_assignment_method: Bottleneck assignment ('cosine', 'l2', or 'linear').
        prototype_assignment_temperature: Softmax temperature for prototype assignment.
        loss_weights: Dict of TVA loss component weights.
        reconciliation_method: Hierarchical reconciliation — None, 'mint', or 'erm'.
        reconciliation_covariance: how MinT's W is obtained in the 'factor'
            and 'none' modes, which produce no per-node residual matrix.
            'auto' (default) keeps the shipped behaviour, 'structural' uses
            the forecast covariance, 'identity' forces W = I.
        min_anchor_history: Minimum history length for anchor selection.
        structure_learning_enabled: Whether to learn DAG structure and hierarchy in V2 network.
        structure_learning_config: Additional structure-learning penalties and options.
        holiday_country: Shared default country for TVA calendar holiday fusion.
        holiday_countries: Optional per-series country override map.
        random_seed: Reproducibility seed.
        verbose: 0=silent, 1=progress bar, 2=per-epoch loss.
        n_jobs: Unused, kept for API compatibility.
    """

    def __init__(
        self,
        name: str = "TVAModel",
        frequency: str = "infer",
        prediction_interval: float = 0.9,
        forecast_length: int = 28,
        trend_network: str = "v2",
        fusion: str = "attention",
        n_factors="auto",
        factor_knot_spacing: int = 7,
        factor_max_lag: int = 0,
        factor_config: dict = None,
        coherence_config: dict = None,
        derived_definitions: dict = None,
        d_token: int = 32,
        n_meso="auto",
        n_global="auto",
        n_prototypes="auto",
        n_heads: int = 2,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        window_size: int = 91,
        series_metadata: list = None,
        prior_adjacency=None,
        causal_prior=None,
        prior_confidence: float = 0.3,
        prototype_assignment_method: str = "cosine",
        prototype_assignment_temperature: float = 1.0,
        loss_weights: dict = None,
        reconciliation_method: str = None,
        reconciliation_covariance: str = "auto",
        min_anchor_history: int = 180,
        structure_learning_enabled: bool = True,
        structure_learning_config: dict = None,
        detector_params: dict = None,
        holiday_country: str = "US",
        holiday_countries: dict = None,
        random_seed: int = 42,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.forecast_length = forecast_length
        self.trend_network = trend_network
        self.fusion = fusion
        self.n_factors = n_factors
        self.factor_knot_spacing = factor_knot_spacing
        self.factor_max_lag = factor_max_lag
        self.factor_config = factor_config
        self.coherence_config = coherence_config
        self.derived_definitions = derived_definitions
        self.d_token = d_token
        self.n_meso = n_meso
        self.n_global = n_global
        self.n_prototypes = n_prototypes
        self.n_heads = n_heads
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.window_size = window_size
        self.series_metadata = series_metadata
        self.prior_adjacency = prior_adjacency
        self.causal_prior = causal_prior
        self.prior_confidence = prior_confidence
        self.prototype_assignment_method = prototype_assignment_method
        self.prototype_assignment_temperature = prototype_assignment_temperature
        self.loss_weights = loss_weights
        self.reconciliation_method = reconciliation_method
        self.reconciliation_covariance = reconciliation_covariance
        self.min_anchor_history = min_anchor_history
        self.structure_learning_enabled = structure_learning_enabled
        self.structure_learning_config = structure_learning_config
        self.detector_params = detector_params
        self.holiday_countries = holiday_countries
        self._tva = None

    def fit(self, df, future_regressor=None):
        """Train TVA on wide-format time series DataFrame.

        Args:
            df: Wide DataFrame with DatetimeIndex, each column a series.
            future_regressor: Unused. Kept for API compatibility.

        Returns:
            self
        """
        from autots.evaluator.tva.tva import TVA

        df = self.basic_profile(df)
        self.df_train = df

        # warn if training data is very short relative to window_size + forecast_length
        min_required = self.window_size + self.forecast_length + 1
        if len(df) < min_required:
            effective_window = max(10, len(df) - self.forecast_length - 1)
            warnings.warn(
                f"TVAModel: training length {len(df)} is less than window_size "
                f"({self.window_size}) + forecast_length ({self.forecast_length}) + 1. "
                f"Reducing window_size to {effective_window}.",
                UserWarning,
                stacklevel=2,
            )
        else:
            effective_window = self.window_size

        # anchors need min_anchor_history observations behind them; asking for
        # more than the data holds leaves nothing to anchor on and the forecast
        # comes back all NaN
        effective_min_anchor = min(
            self.min_anchor_history, max(10, len(df) - self.forecast_length - 1)
        )

        structure_learning_config = {"enabled": self.structure_learning_enabled}
        if isinstance(self.structure_learning_config, dict):
            # preserve explicit overrides while ensuring enabled flag follows wrapper arg
            structure_learning_config.update(self.structure_learning_config)
            structure_learning_config["enabled"] = self.structure_learning_enabled

        self._tva = TVA(
            detector_params=self.detector_params,
            trend_network=self.trend_network,
            fusion=self.fusion,
            n_factors=self.n_factors,
            factor_knot_spacing=self.factor_knot_spacing,
            factor_max_lag=self.factor_max_lag,
            factor_config=self.factor_config,
            coherence_config=self.coherence_config,
            derived_definitions=self.derived_definitions,
            d_token=self.d_token,
            n_meso=self.n_meso,
            n_global=self.n_global,
            n_prototypes=self.n_prototypes,
            n_heads=self.n_heads,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            window_size=effective_window,
            forecast_horizon=self.forecast_length,
            loss_weights=self.loss_weights,
            series_metadata=self._coerced_series_metadata(),
            prior_adjacency=self.prior_adjacency,
            causal_prior=self.causal_prior,
            prior_confidence=self.prior_confidence,
            prototype_assignment_method=self.prototype_assignment_method,
            prototype_assignment_temperature=self.prototype_assignment_temperature,
            reconciliation_method=self.reconciliation_method,
            reconciliation_covariance=self.reconciliation_covariance,
            min_anchor_history=effective_min_anchor,
            holiday_country=self.holiday_country,
            holiday_countries=self.holiday_countries,
            structure_learning_config=structure_learning_config,
            random_seed=self.random_seed,
            verbose=self.verbose,
        )
        self._tva.fit(df)

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generate forecast following training data.

        Args:
            forecast_length: Number of future periods to forecast.
            future_regressor: Unused.
            just_point_forecast: If True, return only point forecast DataFrame.

        Returns:
            PredictionObject, or DataFrame if just_point_forecast is True.
        """
        predict_start_time = datetime.datetime.now()

        if self._tva is None:
            raise RuntimeError("TVAModel must be fit before predict().")

        test_index = self.create_forecast_index(forecast_length=forecast_length)

        raw_forecast = self._tva.predict(forecast_length=forecast_length)

        # Align to AutoTS-computed index and ensure correct columns
        n_returned = len(raw_forecast)
        if n_returned >= forecast_length:
            forecast = raw_forecast.iloc[:forecast_length].copy()
        else:
            # TVA clipped to forecast_horizon < forecast_length; pad with last value
            pad_block = pd.concat(
                [raw_forecast.iloc[[-1]]] * (forecast_length - n_returned),
                ignore_index=True,
            )
            pad_block.index = test_index[n_returned:]
            forecast = pd.concat([raw_forecast, pad_block])

        forecast.index = test_index
        forecast = forecast[self.column_names]

        if just_point_forecast:
            return forecast

        # P3-3: use the trained, calibrated sigma (network NLL head, scaled
        # back to raw units, validation-calibrated, decomposition residual
        # added in quadrature) for intervals; fall back to the generic
        # Point_to_Probability heuristic only when sigma is unavailable.
        sigma_df = getattr(self._tva, "_last_sigma", None)
        if sigma_df is not None and not sigma_df.empty:
            sigma = sigma_df.reindex(columns=self.column_names)
            n_sigma = len(sigma)
            if n_sigma >= forecast_length:
                sigma = sigma.iloc[:forecast_length]
            else:
                pad = pd.concat(
                    [sigma.iloc[[-1]]] * (forecast_length - n_sigma),
                    ignore_index=True,
                )
                sigma = pd.concat([sigma, pad], ignore_index=True)
            sigma.index = test_index
            z_score = self._interval_z(self.prediction_interval)
            upper_forecast = forecast + z_score * sigma
            lower_forecast = forecast - z_score * sigma
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method="inferred_normal",
                prediction_interval=self.prediction_interval,
            )

        predict_runtime = datetime.datetime.now() - predict_start_time
        prediction = PredictionObject(
            model_name=self.name,
            forecast_length=forecast_length,
            forecast_index=test_index,
            forecast_columns=forecast.columns,
            lower_forecast=lower_forecast,
            forecast=forecast,
            upper_forecast=upper_forecast,
            prediction_interval=self.prediction_interval,
            predict_runtime=predict_runtime,
            fit_runtime=self.fit_runtime,
            model_parameters=self.get_params(),
        )
        return prediction

    @staticmethod
    def _interval_z(prediction_interval: float) -> float:
        """Two-sided normal z for a central prediction interval."""
        p = 1.0 - (1.0 - float(prediction_interval)) / 2.0
        try:
            from scipy.stats import norm

            return float(norm.ppf(p))
        except Exception:
            import torch

            return float(
                torch.erfinv(torch.tensor(2.0 * p - 1.0, dtype=torch.float64)).item()
                * (2.0**0.5)
            )

    def _coerced_series_metadata(self):
        """SeriesMetadata objects from whatever the user supplied.

        Plain dicts are accepted so a template stays JSON-serializable;
        ``SeriesMetadata`` already takes exactly these as kwargs.
        """
        metadata = self.series_metadata
        if not metadata:
            return None
        from autots.evaluator.tva.priors import SeriesMetadata

        coerced = []
        for entry in metadata:
            if isinstance(entry, SeriesMetadata):
                coerced.append(entry)
            elif isinstance(entry, dict):
                coerced.append(SeriesMetadata(**entry))
            else:
                warnings.warn(
                    "TVAModel: series_metadata entries must be SeriesMetadata "
                    f"or dicts; dropping {type(entry).__name__}.",
                    UserWarning,
                    stacklevel=3,
                )
        return coerced or None

    def get_params(self):
        """Return dict of current parameters."""
        return {
            "trend_network": self.trend_network,
            "fusion": self.fusion,
            "n_factors": self.n_factors,
            "factor_knot_spacing": self.factor_knot_spacing,
            "factor_max_lag": self.factor_max_lag,
            "factor_config": self.factor_config,
            "coherence_config": self.coherence_config,
            "derived_definitions": self.derived_definitions,
            "d_token": self.d_token,
            "n_meso": self.n_meso,
            "n_global": self.n_global,
            "n_prototypes": self.n_prototypes,
            "n_heads": self.n_heads,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "series_metadata": self.series_metadata,
            "prior_adjacency": self.prior_adjacency,
            "causal_prior": self.causal_prior,
            "prior_confidence": self.prior_confidence,
            "prototype_assignment_method": self.prototype_assignment_method,
            "prototype_assignment_temperature": self.prototype_assignment_temperature,
            "loss_weights": self.loss_weights,
            "reconciliation_method": self.reconciliation_method,
            "reconciliation_covariance": self.reconciliation_covariance,
            "min_anchor_history": self.min_anchor_history,
            "structure_learning_enabled": self.structure_learning_enabled,
            "structure_learning_config": self.structure_learning_config,
            "detector_params": self.detector_params,
            "holiday_countries": self.holiday_countries,
        }

    @staticmethod
    def _new_factor_config(method: str = "random"):
        """Sample ``factor_network.DEFAULT_FACTOR_CONFIG`` overrides for search.

        Profile-anchored rather than independent per key, for two measured
        reasons. Several knobs are inert alone — a blend tie tolerance does
        nothing without ``sn_blend``, and ``space`` only pays once the blend
        is choosing between two sane paths. And the stack matters far more
        than any single key: on the LRP iteration folds plain factor defaults
        scored an aggregate base MASE ratio of 2.22 against SeasonalNaive,
        while the assembled "arm F" stack (safety blend + tie tolerance 0.25 +
        seasonal arbitration + log space) scored 1.011. Independent per-key
        sampling would reach that corner of the space only rarely.

        Library defaults are deliberately left unchanged. Arm F clears the
        aggregate MASE, per-fold MASE and coherence gates but fails both
        per-series gates, so it is offered to the optimizer to win on a given
        dataset rather than asserted as a default.

        Excluded on measured evidence: ``space='auto'`` (1.166 vs 1.011 for
        forced log — the selector fails, not the option), ``blend_risk_weight``
        (moves the worst-series ratio the wrong way), ``coherence`` (shrink
        gain 0.000) and ``level_shift_veto`` (never fires on real detector
        output). Enabling any of them is still possible by hand.
        """
        fast_mode = "fast" in str(method).lower()

        # The safety layer refits the factor stage once per inner fold, so it
        # is the dominant cost knob here; 'default' and 'explore' keep cheap
        # arms in the pool.
        profile = random.choices(
            ["arm_f", "arm_f_sparse", "safety", "explore", "default"],
            weights=(
                [0.2, 0.15, 0.15, 0.2, 0.3]
                if fast_mode
                else [0.32, 0.23, 0.15, 0.15, 0.15]
            ),
        )[0]
        if profile == "default":
            # today's behavior, bit for bit — worth keeping reachable as the
            # comparator every other profile is measured against
            return None

        cfg = {}
        if profile in ("arm_f", "arm_f_sparse", "safety"):
            # 1b/1e: blend the factor path against SeasonalNaive on inner
            # rolling origins. inner_refit rides along unconditionally — the
            # factor stage must be refit on truncated history or the folds it
            # is graded on are in-sample, which is the bug that let the blend
            # hand the network a weight it had not earned.
            cfg["sn_blend"] = True
            cfg["inner_refit"] = True
            cfg["inner_folds"] = random.choices([3, 5], weights=[0.8, 0.2])[0]
            cfg["safety_config"] = {
                "blend_tie_tolerance": random.choices(
                    [0.01, 0.1, 0.25], weights=[0.15, 0.3, 0.55]
                )[0]
            }
            cfg["reanchor"] = random.choices([True, False], weights=[0.4, 0.6])[0]
            cfg["error_cap"] = random.choices([True, False], weights=[0.3, 0.7])[0]
            cfg["conformal_sigma"] = random.choices([True, False], weights=[0.3, 0.7])[
                0
            ]

        if profile in ("arm_f", "arm_f_sparse"):
            cfg["seasonal_arbitration"] = True
            # 2c: the single largest measured lever on LRP (aggregate 1.123 ->
            # 1.011, worst series 6.14 -> 1.93) on a panel spanning ~1e-2 to
            # 2e8 whose metrics comove in growth-rate terms
            cfg["space"] = "log"
        else:
            cfg["space"] = random.choices(["level", "log"], weights=[0.65, 0.35])[0]
            cfg["seasonal_arbitration"] = random.choices(
                [True, False], weights=[0.3, 0.7]
            )[0]

        # ---- C9: sparse-code identification / autoencoder ------------------
        # Both tiers initialize from and fall back to the alternating
        # estimator, so sampling one cannot leave a panel worse identified.
        # 'sparse_ae' degrades to 'sparse_alt' without torch.
        if profile == "arm_f_sparse":
            identification = random.choices(
                ["sparse_ae", "sparse_alt"], weights=[0.55, 0.45]
            )[0]
        else:
            identification = random.choices(
                ["alternating", "sparse_alt", "sparse_ae"],
                weights=[0.7, 0.15, 0.15],
            )[0]
        cfg["identification"] = identification
        if identification != "alternating":
            cfg["sparse_config"] = {
                # k stays at 1: code_topk=2 is on the ladder's do-not-retry
                # list (pair precision 0.267 vs 0.290, and it re-enables the
                # abstention knobs at the cost of the sparsity that made the
                # basis identifiable). Still settable by hand.
                "code_topk": 1,
                # the only abstention channel left at code_topk == 1, where
                # dominance_margin and min_loading_share pass unconditionally
                "min_code_share": random.choices(
                    [0.0, 0.1, 0.2], weights=[0.5, 0.3, 0.2]
                )[0],
                # three restarts cost 3x the identification; a single varimax
                # start recovers most of the anti-collapse benefit
                "init_rotate": random.choices(
                    [["varimax"], [None, "varimax", "quartimax"]],
                    weights=[0.8, 0.2] if fast_mode else [0.5, 0.5],
                )[0],
            }

        # ---- estimator knobs: cheap (no extra fits) and never yet swept ----
        # The trendless-series gate zeroes loadings of series with no
        # low-frequency structure. It is the direct lever on the largest
        # measured factor-mode failure: on load_daily both factor arms run
        # ~8x SeasonalNaive against ~3.66x for 'none', which is exactly a
        # trendless series being handed factor exposure. Pinned at 0.6 since
        # the mode was written and never swept; 0.0 disables the gate.
        cfg["min_trend_to_noise"] = random.choices(
            [0.0, 0.3, 0.6, 1.0, 1.5], weights=[0.1, 0.2, 0.35, 0.25, 0.1]
        )[0]
        # Exposure-share floor below which a factor is dropped. Fitted rank
        # came out 4.0 against a true 3 in 7/7 ladder cells, and bad rank
        # selection is what blocked the robust input estimator, so the pruning
        # threshold is worth searching rather than assuming.
        cfg["prune_share"] = random.choices(
            [0.0, 0.01, 0.02, 0.05, 0.1], weights=[0.1, 0.2, 0.35, 0.25, 0.1]
        )[0]
        # l1 trend-filter smoothness on the factor paths. Each score column is
        # standardized before the Lasso and rescaled after, so this is a pure
        # smoothness knob and does not need retuning per panel scale — which
        # is what makes it safe to sample across arbitrary datasets. Controls
        # how many knots a factor path spends, i.e. how smooth the thing being
        # extrapolated is, and extrapolation is the LRP failure mode.
        cfg["alpha"] = random.choices(
            [3e-4, 1e-3, 3e-3, 1e-2], weights=[0.2, 0.4, 0.25, 0.15]
        )[0]

        # ---- lower-weight knobs, each measured and each still plausible ----
        # 1d: quarantine series whose recent tail is numerically constant
        cfg["frozen_tail_gate"] = random.choices([True, False], weights=[0.35, 0.65])[0]
        # 1a: pick the factor continuation rule by held-out reconstruction
        cfg["continuation_select"] = random.choices([True, False], weights=[0.3, 0.7])[
            0
        ]
        # 1c: zero loadings of series the factor model forecasts worse than a
        # damped local-linear baseline on their own raw target (None == off)
        cfg["gate_forecast_margin"] = random.choices(
            [None, 1.0, 1.25], weights=[0.7, 0.2, 0.1]
        )[0]
        # C1: rotation moves only the basis the coherence graph reads, never
        # the reconstruction. Clears every gate on a clean panel and hurts on
        # the detector-adjusted one, so it stays low weight.
        cfg["rotate"] = random.choices([None, "varimax"], weights=[0.85, 0.15])[0]
        # 2b: recovers the factor span far better (canonical correlation 0.77
        # vs 0.53) while making level-space scale and rank selection worse;
        # on LRP the damage slightly outweighed the gain (1.021 vs 1.011)
        cfg["input_estimator"] = random.choices(
            ["detector", "robust"], weights=[0.8, 0.2]
        )[0]
        # C5: the same robust panel used for the *graph only*, leaving every
        # forecast value untouched. A second identification fit, so it costs.
        if random.random() < (0.1 if fast_mode else 0.15):
            cfg["structure_input"] = "robust"
        # 3b: fit shared factors on long-history anchors, project the rest
        cfg["anchor_selection"] = random.choices([True, False], weights=[0.3, 0.7])[0]
        # 3c: data-derived group factors underneath the global ones; costs
        # group_refits extra fits, so it is rare and never sampled in fast mode
        if not fast_mode and random.random() < 0.12:
            cfg["group_factors"] = True
        # D: pull prior-linked series toward a shared loading profile. Inert
        # unless the caller also supplied `prior_adjacency`, which is a dataset
        # fact the search never invents -- so this is only ever a live knob on
        # panels where a prior exists, and costs nothing everywhere else.
        if random.random() < 0.1:
            cfg["w_prior_loadings"] = random.choices(
                [0.01, 0.05, 0.2], weights=[0.4, 0.4, 0.2]
            )[0]

        return cfg

    @staticmethod
    def get_new_params(method: str = "random"):
        """Return randomly sampled parameters for AutoTS optimizer.

        Weights favor faster/more reliable configurations while also exposing
        structure and loss regularization knobs for broader TVA search.
        ``trend_network`` is weighted toward 'factor' and never samples 'v1'
        (see the comment below); in 'factor' mode a ``factor_config`` bundle
        is drawn by :meth:`_new_factor_config`, which is where the Phase 1/2
        safety-layer and sparse-identification knobs enter the search.
        """
        fast_mode = "fast" in str(method).lower()

        # Weighted by the five-arm benchmark (5 synthetic datasets x 2 horizons
        # x 4 folds, plus the LRP iteration folds). 'factor' is the only arm
        # that separates from the torch-free 'none' baseline on either
        # benchmark; 'v1' and 'v2' scored identically to 'none' on synthetic
        # (geometric-mean skill 0.338 for all three) while costing 663.7s and
        # 336.8s mean fit on LRP against 19.2s for 'factor'. So 'v1' is no
        # longer sampled at all and 'v2' is kept only as a rare escape hatch.
        trend_network = random.choices(
            ["factor", "none", "v2"], weights=[0.6, 0.3, 0.1]
        )[0]
        # only read in 'factor' mode; left None elsewhere so templates don't
        # carry inert nesting
        factor_config = (
            TVAModel._new_factor_config(method) if trend_network == "factor" else None
        )
        # only sampled meaningfully in 'factor' mode; harmless elsewhere.
        # 'auto' (select_n_factors, an eigenvalue ratio) loses weight on
        # measured evidence: it returned rank 4.0 against a true 3 in 7 of 7
        # ladder cells, and rank misspecification is what blocked the robust
        # input estimator from promotion. Explicit small K gets the mass.
        n_factors = random.choices(
            ["auto", 1, 2, 3, 4, 6], weights=[0.35, 0.05, 0.15, 0.2, 0.15, 0.1]
        )[0]
        factor_knot_spacing = random.choices([7, 14, 28], weights=[0.6, 0.25, 0.15])[0]
        factor_max_lag = random.choices([0, 7, 14], weights=[0.8, 0.1, 0.1])[0]
        fusion = random.choices(
            ["attention", "additive", "direct"], weights=[0.35, 0.45, 0.15]
        )[0]
        d_token = random.choices(
            [16, 24, 32, 48, 64], weights=[0.1, 0.2, 0.4, 0.2, 0.1]
        )[0]
        n_heads = random.choices([2, 4, 8], weights=[0.55, 0.35, 0.1])[0]
        # keep n_heads as divisor of d_token
        if d_token % n_heads != 0:
            n_heads = 4 if d_token % 4 == 0 else 2
        if fast_mode:
            epochs = random.choices([30, 50, 80, 120], weights=[0.2, 0.4, 0.3, 0.1])[0]
            batch_size = random.choices([16, 32, 64], weights=[0.1, 0.45, 0.45])[0]
            window_size = random.choices(
                [30, 45, 60, 91], weights=[0.2, 0.3, 0.3, 0.2]
            )[0]
        else:
            # early stopping bounds the effective epochs; sample high maxima
            epochs = random.choices(
                [50, 100, 200, 300], weights=[0.15, 0.3, 0.4, 0.15]
            )[0]
            batch_size = random.choices([16, 32, 64], weights=[0.2, 0.5, 0.3])[0]
            window_size = random.choices(
                [45, 60, 91, 120, 180], weights=[0.1, 0.2, 0.4, 0.2, 0.1]
            )[0]
        lr = random.choices(
            [3e-4, 5e-4, 1e-3, 2e-3, 5e-3], weights=[0.1, 0.25, 0.35, 0.2, 0.1]
        )[0]
        prior_confidence = random.choices(
            [0.05, 0.1, 0.2, 0.3, 0.5, 0.7], weights=[0.05, 0.15, 0.25, 0.3, 0.15, 0.1]
        )[0]
        n_global = random.choices(
            ["auto", 2, 3, 4, 5, 6], weights=[0.5, 0.08, 0.14, 0.14, 0.08, 0.06]
        )[0]
        n_meso = random.choices(
            ["auto", 4, 6, 8, 10, 12], weights=[0.45, 0.1, 0.15, 0.15, 0.1, 0.05]
        )[0]
        n_prototypes = random.choices(
            ["auto", 2, 3, 4, 5, 6, 8],
            weights=[0.55, 0.12, 0.14, 0.1, 0.06, 0.03, 0.03],
        )[0]
        if n_meso != "auto" and n_global != "auto":
            n_meso = max(int(n_meso), int(n_global))
        prototype_assignment_method = random.choices(
            ["cosine", "l2", "linear"], weights=[0.6, 0.3, 0.1]
        )[0]
        prototype_assignment_temperature = random.choices(
            [0.5, 0.75, 1.0, 1.25, 1.5], weights=[0.1, 0.2, 0.4, 0.2, 0.1]
        )[0]
        reconciliation_method = random.choices(
            [None, "mint", "erm"], weights=[0.6, 0.3, 0.1]
        )[0]
        structure_learning_enabled = random.choices([True, False], weights=[0.7, 0.3])[
            0
        ]
        min_anchor_history = random.choices(
            [90, 120, 180, 365], weights=[0.15, 0.25, 0.45, 0.15]
        )[0]
        structure_learning_config = {
            "learn_hierarchy": random.choices([True, False], weights=[0.7, 0.3])[0],
            "learn_dag": random.choices([True, False], weights=[0.8, 0.2])[0],
            "dag_penalty": random.choices(
                [0.02, 0.05, 0.1, 0.2], weights=[0.2, 0.35, 0.3, 0.15]
            )[0],
            "sparsity_weight": random.choices(
                [0.001, 0.005, 0.01, 0.02], weights=[0.2, 0.35, 0.3, 0.15]
            )[0],
            "assignment_entropy_weight": random.choices(
                [0.0, 0.002, 0.005, 0.01], weights=[0.15, 0.25, 0.35, 0.25]
            )[0],
            "assignment_full_rank_weight": random.choices(
                [0.0, 0.005, 0.01, 0.02], weights=[0.1, 0.25, 0.45, 0.2]
            )[0],
            "prior_tether_weight": random.choices(
                [0.0, 0.02, 0.05, 0.1], weights=[0.1, 0.3, 0.45, 0.15]
            )[0],
            "temporal_drift_weight": random.choices(
                [0.0, 0.001, 0.005, 0.01], weights=[0.6, 0.2, 0.15, 0.05]
            )[0],
        }
        loss_weights = {
            "forecast": random.choices([0.75, 1.0, 1.25], weights=[0.15, 0.7, 0.15])[0],
            "orthogonality": random.choices(
                [0.0, 0.1, 0.25, 0.5], weights=[0.15, 0.25, 0.4, 0.2]
            )[0],
            "local_trend": random.choices(
                [0.0, 0.05, 0.1, 0.25, 0.5],
                weights=[0.15, 0.2, 0.35, 0.2, 0.1],
            )[0],
            "smoothness": random.choices(
                [0.0, 0.01, 0.02, 0.05, 0.1],
                weights=[0.15, 0.2, 0.35, 0.2, 0.1],
            )[0],
            "soft_prior": random.choices(
                [0.0, 0.1, 0.25, 0.5], weights=[0.2, 0.25, 0.4, 0.15]
            )[0],
            "causal_prior": random.choices(
                [0.0, 0.1, 0.25, 0.5], weights=[0.2, 0.25, 0.4, 0.15]
            )[0],
            "coherence": random.choices(
                [0.0, 0.1, 0.25, 0.5, 1.0],
                weights=[0.2, 0.2, 0.35, 0.15, 0.1],
            )[0],
            "probabilistic": random.choices(
                [0.0, 0.1, 0.25, 0.5, 0.75],
                weights=[0.1, 0.15, 0.25, 0.35, 0.15],
            )[0],
            "fusion_forecast": random.choices(
                [0.0, 0.25, 0.5, 0.75, 1.0],
                weights=[0.1, 0.2, 0.25, 0.2, 0.25],
            )[0],
            "trend_phi": random.choices(
                [0.0, 0.1, 0.25, 0.5, 1.0],
                weights=[0.6, 0.12, 0.12, 0.1, 0.06],
            )[0],
        }

        from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

        detector_params = TimeSeriesFeatureDetector.get_new_params(method=method)

        return {
            "trend_network": trend_network,
            "fusion": fusion,
            "n_factors": n_factors,
            "factor_config": factor_config,
            "factor_knot_spacing": factor_knot_spacing,
            "factor_max_lag": factor_max_lag,
            "d_token": d_token,
            "n_meso": n_meso,
            "n_global": n_global,
            "n_prototypes": n_prototypes,
            "n_heads": n_heads,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "window_size": window_size,
            "prior_confidence": prior_confidence,
            "prototype_assignment_method": prototype_assignment_method,
            "prototype_assignment_temperature": prototype_assignment_temperature,
            "loss_weights": loss_weights,
            "reconciliation_method": reconciliation_method,
            "min_anchor_history": min_anchor_history,
            "structure_learning_enabled": structure_learning_enabled,
            "structure_learning_config": structure_learning_config,
            "detector_params": detector_params,
        }
