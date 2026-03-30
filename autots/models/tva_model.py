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
        trend_network: 'v2' (learned directed graph, default) or 'v1' (hierarchical latent).
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
        prior_confidence: Weight of prior adjacency matrix (0=ignore priors, 1=rigid).
        prototype_assignment_method: Bottleneck assignment ('cosine', 'l2', or 'linear').
        prototype_assignment_temperature: Softmax temperature for prototype assignment.
        loss_weights: Dict of TVA loss component weights.
        reconciliation_method: Hierarchical reconciliation — None, 'mint', or 'erm'.
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
        d_token: int = 64,
        n_meso="auto",
        n_global="auto",
        n_prototypes="auto",
        n_heads: int = 4,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        window_size: int = 91,
        prior_confidence: float = 0.3,
        prototype_assignment_method: str = "cosine",
        prototype_assignment_temperature: float = 1.0,
        loss_weights: dict = None,
        reconciliation_method: str = None,
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
        self.d_token = d_token
        self.n_meso = n_meso
        self.n_global = n_global
        self.n_prototypes = n_prototypes
        self.n_heads = n_heads
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.window_size = window_size
        self.prior_confidence = prior_confidence
        self.prototype_assignment_method = prototype_assignment_method
        self.prototype_assignment_temperature = prototype_assignment_temperature
        self.loss_weights = loss_weights
        self.reconciliation_method = reconciliation_method
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

        structure_learning_config = {"enabled": self.structure_learning_enabled}
        if isinstance(self.structure_learning_config, dict):
            # preserve explicit overrides while ensuring enabled flag follows wrapper arg
            structure_learning_config.update(self.structure_learning_config)
            structure_learning_config["enabled"] = self.structure_learning_enabled

        self._tva = TVA(
            detector_params=self.detector_params,
            trend_network=self.trend_network,
            fusion=self.fusion,
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
            prior_confidence=self.prior_confidence,
            prototype_assignment_method=self.prototype_assignment_method,
            prototype_assignment_temperature=self.prototype_assignment_temperature,
            reconciliation_method=self.reconciliation_method,
            min_anchor_history=self.min_anchor_history,
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

    def get_params(self):
        """Return dict of current parameters."""
        return {
            "trend_network": self.trend_network,
            "fusion": self.fusion,
            "d_token": self.d_token,
            "n_meso": self.n_meso,
            "n_global": self.n_global,
            "n_prototypes": self.n_prototypes,
            "n_heads": self.n_heads,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "prior_confidence": self.prior_confidence,
            "prototype_assignment_method": self.prototype_assignment_method,
            "prototype_assignment_temperature": self.prototype_assignment_temperature,
            "loss_weights": self.loss_weights,
            "reconciliation_method": self.reconciliation_method,
            "min_anchor_history": self.min_anchor_history,
            "structure_learning_enabled": self.structure_learning_enabled,
            "structure_learning_config": self.structure_learning_config,
            "detector_params": self.detector_params,
            "holiday_countries": self.holiday_countries,
        }

    @staticmethod
    def get_new_params(method: str = "random"):
        """Return randomly sampled parameters for AutoTS optimizer.

        Weights favor faster/more reliable configurations while also exposing
        structure and loss regularization knobs for broader TVA search.
        """
        fast_mode = "fast" in str(method).lower()

        trend_network = random.choices(["v2", "v1"], weights=[0.75, 0.25])[0]
        fusion = random.choices(
            ["attention", "additive", "direct"], weights=[0.35, 0.45, 0.15]
        )[0]
        d_token = random.choices(
            [32, 48, 64, 96, 128], weights=[0.15, 0.2, 0.35, 0.2, 0.1]
        )[0]
        n_heads = random.choices([2, 4, 8, 10], weights=[0.2, 0.6, 0.2, 0.1])[0]
        # keep n_heads as divisor of d_token
        if d_token % n_heads != 0:
            n_heads = 4 if d_token % 4 == 0 else 2
        if fast_mode:
            epochs = random.choices([15, 20, 30, 40], weights=[0.2, 0.4, 0.3, 0.1])[0]
            batch_size = random.choices([16, 32, 64], weights=[0.1, 0.45, 0.45])[0]
            window_size = random.choices(
                [30, 45, 60, 91], weights=[0.2, 0.3, 0.3, 0.2]
            )[0]
        else:
            epochs = random.choices(
                [20, 30, 50, 75, 100], weights=[0.2, 0.3, 0.3, 0.15, 0.05]
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
                [0.0, 0.25, 0.5, 1.0], weights=[0.1, 0.2, 0.3, 0.4]
            )[0],
            "local_trend": random.choices(
                [0.0, 0.05, 0.1, 0.25, 0.5, 1.0],
                weights=[0.12, 0.15, 0.2, 0.22, 0.18, 0.13],
            )[0],
            "smoothness": random.choices(
                [0.0, 0.01, 0.03, 0.1, 0.2, 0.3],
                weights=[0.1, 0.2, 0.25, 0.3, 0.15, 0.05],
            )[0],
            "soft_prior": random.choices(
                [0.0, 0.1, 0.25, 0.5, 0.75], weights=[0.15, 0.25, 0.3, 0.3, 0.1]
            )[0],
            "causal_prior": random.choices(
                [0.0, 0.1, 0.25, 0.5], weights=[0.2, 0.25, 0.3, 0.25]
            )[0],
            "coherence": random.choices(
                [0.0, 0.05, 0.25, 0.5, 1.0, 2.0],
                weights=[0.2, 0.1, 0.2, 0.1, 0.25, 0.15],
            )[0],
            "probabilistic": random.choices(
                [0.0, 0.05, 0.1, 0.2, 0.3],
                weights=[0.1, 0.2, 0.35, 0.25, 0.1],
            )[0],
            "fusion_forecast": random.choices(
                [0.0, 0.25, 0.5, 0.75, 1.0],
                weights=[0.1, 0.2, 0.25, 0.2, 0.25],
            )[0],
        }

        from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

        detector_params = TimeSeriesFeatureDetector.get_new_params(method=method)

        return {
            "trend_network": trend_network,
            "fusion": fusion,
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
