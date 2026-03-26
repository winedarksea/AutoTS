# -*- coding: utf-8 -*-
"""DecompositionMixin for TimeSeriesFeatureDetector."""

import numpy as np
import pandas as pd
from autots.tools.transform import DatepartRegressionTransformer


class DecompositionMixin:
    """Mixin providing initial decomposition (rough seasonality removal)."""

    def _initial_decomposition(self, df_work):
        """
        Perform initial decomposition: rough seasonality, holidays, and anomalies.

        Returns
        -------
        tuple
            (rough_residual, rough_seasonality)
        """
        # Rough seasonality removal
        (
            rough_residual,
            rough_seasonality,
            self.rough_seasonality_model,
        ) = self._compute_rough_seasonality(df_work)

        # Holiday detection should see the rough residual directly because the
        # holiday detector itself relies on anomaly-like spikes to identify dates.
        holiday_dates, holiday_splash_dates, holiday_regressors = self._detect_holidays(
            rough_residual
        )
        self._holiday_dates_temp = holiday_dates
        self._holiday_regressors_temp = holiday_regressors
        # Persist regressor column names separately so forecast() can align future
        # regressors to the fitted schema even if _holiday_regressors_temp is lost
        # (e.g., across a serialization round-trip). Calendar holiday columns are
        # type-stable (e.g., "calendar_US__Christmas Day"), not date-specific, so
        # schema alignment from stored names is safe across forecast horizons.
        self._holiday_regressor_columns = (
            list(holiday_regressors.columns)
            if holiday_regressors is not None
            and not getattr(holiday_regressors, "empty", True)
            else []
        )

        anomaly_params = self._build_anomaly_params()
        if self.global_holiday_anomaly_suppression:
            combined_holiday_dates = self._flatten_holiday_dates(holiday_dates)
            anomaly_params = self._build_anomaly_params(
                holiday_dates=combined_holiday_dates
            )

        # Anomaly detection
        residual_without_anomalies, anomaly_records = self._detect_anomalies(
            rough_residual,
            anomaly_params=anomaly_params,
        )
        self._anomaly_records_temp = anomaly_records

        return rough_residual, rough_seasonality

    def _compute_rough_seasonality(self, df):
        model = DatepartRegressionTransformer(**self.rough_seasonality_params)
        residual = model.fit_transform(df)
        seasonal = df - residual
        return residual, seasonal, model
