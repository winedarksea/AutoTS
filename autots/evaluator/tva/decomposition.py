# -*- coding: utf-8 -*-
"""
NornDecomposer — Wraps TimeSeriesFeatureDetector for TVA decomposition.

In the spirit of structural weaving, this class decomposes time series into
constituent threads (trend, seasonality, noise, etc.) for joint forecasting.
"""

import numpy as np
import pandas as pd


class NornDecomposer:
    """Wraps TimeSeriesFeatureDetector to extract structured component DataFrames.

    Produces trend, seasonality, holidays, level_shifts, anomalies, and noise
    as aligned DataFrames sharing the same index and columns as the input.

    Args:
        detector_params: Dict of kwargs passed to TimeSeriesFeatureDetector.__init__.
    """

    def __init__(
        self,
        detector_params: dict = None,
        holiday_country=None,
        holiday_countries: dict = None,
    ):
        self.detector_params = dict(detector_params or {})
        if holiday_country is not None:
            self.detector_params['holiday_country'] = holiday_country
        if holiday_countries is not None:
            self.detector_params['holiday_countries'] = holiday_countries
        self.detector = None
        self._df_original = None
        self._components = None

    def fit(self, df: pd.DataFrame) -> 'NornDecomposer':
        """Fit the feature detector on a wide DataFrame.

        Args:
            df: Wide DataFrame with DatetimeIndex and one numeric column per series.

        Returns:
            self
        """
        from autots.evaluator.feature_detector import TimeSeriesFeatureDetector

        self._df_original = df
        self.detector = TimeSeriesFeatureDetector(**self.detector_params)
        self.detector.fit(df)
        self._components = None  # clear cache
        return self

    def get_components(self) -> dict:
        """Extract decomposition components as DataFrames.

        Returns:
            Dict with keys 'trend', 'seasonality', 'holidays', 'level_shifts',
            'anomalies', 'noise'. Each value is a DataFrame (T, N) aligned with
            the original input index and columns.
        """
        if self._components is not None:
            return self._components

        if self.detector is None:
            raise RuntimeError(
                "NornDecomposer must be fit before calling get_components."
            )

        index = self.detector.date_index
        columns = self._df_original.columns

        _component_keys = {
            'trend': 'trend',
            'seasonality': 'seasonality',
            'holidays': 'holidays',
            'level_shifts': 'level_shift',
            'anomalies': 'anomalies',
            'noise': 'noise',
        }

        result = {}
        for output_key, internal_key in _component_keys.items():
            data = {}
            for col in columns:
                comp = self.detector.components.get(col, {})
                arr = comp.get(internal_key, None)
                if arr is not None:
                    data[col] = np.asarray(arr, dtype=np.float64)
                else:
                    data[col] = np.zeros(len(index), dtype=np.float64)
            result[output_key] = pd.DataFrame(data, index=index, columns=columns)

        self._components = result
        return result

    def get_features(self) -> dict:
        """Return the full detected features dict from the underlying detector.

        Includes changepoints, holiday impacts, anomaly records, etc.
        Useful for constructing priors or inspecting detection results.
        """
        if self.detector is None:
            raise RuntimeError(
                "NornDecomposer must be fit before calling get_features."
            )
        return self.detector.get_detected_features()

    def get_forecast_components(self, forecast_length: int) -> dict:
        """Forward-project seasonality, holidays, level shifts, and trend.

        Uses the detector's built-in forecast method to project known
        deterministic components into the future.

        Args:
            forecast_length: Number of future periods to project.

        Returns:
            Dict with same keys as get_components(), each a DataFrame
            of shape (forecast_length, N).
        """
        if self.detector is None:
            raise RuntimeError("NornDecomposer must be fit before forecasting.")

        pred = self.detector.forecast(forecast_length)
        future_index = pred.forecast.index
        columns = self._df_original.columns

        # the detector forecast returns a PredictionObject; its components
        # include 'trend', 'seasonality', 'holidays', 'level_shift'
        result = {}
        component_map = {
            'trend': 'trend',
            'seasonality': 'seasonality',
            'holidays': 'holidays',
            'level_shifts': 'level_shift',
        }

        # extract from prediction components if available.
        # pred.components is a MultiIndex DataFrame with columns (series, component_name)
        # produced by stack_component_frames — never a plain dict.
        pred_components = getattr(pred, 'components', None)
        if pred_components is not None and isinstance(pred_components, pd.DataFrame):
            available_components = set(pred_components.columns.get_level_values(1))
            for output_key, comp_key in component_map.items():
                if comp_key in available_components:
                    df = pred_components.xs(comp_key, axis=1, level=1)
                    result[output_key] = df.reindex(columns=columns, fill_value=0.0)
                else:
                    result[output_key] = pd.DataFrame(
                        0.0, index=future_index, columns=columns
                    )
        else:
            for output_key in component_map:
                result[output_key] = pd.DataFrame(
                    0.0, index=future_index, columns=columns
                )

        # anomalies and noise are not projected
        result['anomalies'] = pd.DataFrame(0.0, index=future_index, columns=columns)
        result['noise'] = pd.DataFrame(0.0, index=future_index, columns=columns)

        return result

    def _urd_verdict(self, series_name: str) -> float:
        """Hidden: Urd (Norn of the past) judges the noise-to-signal ratio."""
        if self.detector is None:
            return float('nan')
        return self.detector.noise_to_signal_ratios.get(series_name, float('nan'))
