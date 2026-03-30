# -*- coding: utf-8 -*-
"""SeasonalityMixin for TimeSeriesFeatureDetector."""

import numpy as np
import pandas as pd
import copy
import warnings
from autots.tools.transform import DatepartRegressionTransformer
from autots.tools.seasonal import date_part, build_adaptive_fourier_features
from autots.tools.fft import FFT


class SeasonalityMixin:
    """Mixin providing seasonality fitting, strength estimation, and changepoint detection."""

    def _final_seasonality_fit(self, df_work, rough_residual, rough_seasonality):
        """
        Fit final seasonality model including holiday effects.

        Fits on original data (df_work) with only anomalies removed.
        This ensures final seasonality captures the full seasonal pattern,
        and holidays are fit simultaneously as regressors.

        Returns
        -------
        tuple
            (final_residual, final_seasonality, seasonality_strength, holiday_component,
             holiday_coefficients, holiday_splash_impacts)
        """
        # Reconstruct original data with anomalies removed
        # df_work = original standardized data
        # We need to remove anomalies from df_work, not from rough_residual
        df_without_anomalies = self.anomaly_detector.transform(df_work)

        # Fit final seasonality on original data (with anomalies removed)
        # Holiday effects are captured as regressors during this fit
        (
            final_residual,
            final_seasonality,
            seasonality_strength,
            self.seasonality_model,
            holiday_component_scaled,
            holiday_coefficients,
            holiday_splash_impacts_scaled,
        ) = self._fit_final_seasonality(
            df_without_anomalies, self._holiday_regressors_temp
        )

        return (
            final_residual,
            final_seasonality,
            seasonality_strength,
            holiday_component_scaled,
            holiday_coefficients,
            holiday_splash_impacts_scaled,
        )

    def _fit_final_seasonality(self, df, holiday_regressors=None):
        """
        Fit final seasonality model and decompose holiday effects.

        Returns
        -------
        tuple
            (residual, seasonal_component, strength, model, holiday_component, holiday_coefficients, holiday_splash_impacts)
        """
        regressor = None
        if holiday_regressors is not None and not holiday_regressors.empty:
            regressor = holiday_regressors.reindex(df.index).fillna(0.0)

        seasonality_params = copy.deepcopy(self.seasonality_params)

        # Adaptive Fourier mode: detect dominant periods with FFT, then augment
        # regressors with period-specific Fourier features.
        self.detected_seasonal_periods = None
        if seasonality_params.get('datepart_method') == 'adaptive_fourier':
            try:
                fft_model = FFT(n_harm=None, detrend='linear')
                fft_input = df.dropna(how='all').to_numpy(dtype=float)
                nan_mask = np.isfinite(fft_input).all(axis=1)
                if nan_mask.sum() >= 14:
                    fft_model.fit(fft_input[nan_mask])
                    detected = fft_model.detect_dominant_periods(
                        min_period=3, max_periods=5, power_threshold=0.1
                    )
                else:
                    detected = []

                if len(detected) >= 2:
                    self.detected_seasonal_periods = detected
                    adaptive_features = build_adaptive_fourier_features(
                        df.index, detected, max_order=12
                    )
                    if regressor is not None:
                        regressor = pd.concat([regressor, adaptive_features], axis=1)
                    else:
                        regressor = adaptive_features
                    seasonality_params['datepart_method'] = 'simple_3'
                else:
                    seasonality_params['datepart_method'] = 'common_fourier'
            except Exception:
                seasonality_params['datepart_method'] = 'common_fourier'

        model = DatepartRegressionTransformer(**seasonality_params)
        regressor_full = regressor
        df_fit = df.dropna(how='all')
        if df_fit.empty:
            df_fit = df
        regressor_fit = None
        if regressor_full is not None:
            regressor_fit = regressor_full.loc[df_fit.index]
        model.fit(df_fit, regressor=regressor_fit)
        residual = model.transform(df, regressor=regressor_full)
        seasonal_total = df - residual

        holiday_component = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        seasonal_component = seasonal_total
        holiday_coefficients = {col: {} for col in df.columns}
        holiday_splash_impacts = {col: {} for col in df.columns}

        if regressor is not None:
            zero_regressor = regressor.copy()
            zero_regressor.loc[:, :] = 0.0
            zeros_df = pd.DataFrame(0.0, index=df.index, columns=df.columns)
            try:
                baseline_pred = model.inverse_transform(
                    zeros_df, regressor=zero_regressor
                )
                seasonal_component = baseline_pred
                holiday_component = seasonal_total - seasonal_component

                # Detect splash/bridge days: days with holiday impact but not in core holiday list
                # Splash days are typically adjacent to core holidays with reduced impact
                for col in df.columns:
                    holiday_series = holiday_component[col]
                    significant_impacts = holiday_series[abs(holiday_series) > 1e-9]
                    for date, impact in significant_impacts.items():
                        # This will be refined by checking against core holiday dates
                        # For now, store all non-zero holiday impacts
                        # The core vs. splash distinction will be made during template building
                        holiday_splash_impacts[col][pd.Timestamp(date)] = float(impact)

            except Exception as exc:
                warnings.warn(
                    f"Failed to isolate holiday contribution during seasonality fit: {exc}",
                    RuntimeWarning,
                )
                holiday_component = pd.DataFrame(
                    0.0, index=df.index, columns=df.columns
                )
                seasonal_component = seasonal_total
            holiday_coefficients = self._solve_holiday_coefficients(
                regressor, holiday_component
            )

        strength = self._compute_seasonality_strength(df, residual, seasonal_component)
        return (
            residual,
            seasonal_component,
            strength,
            model,
            holiday_component,
            holiday_coefficients,
            holiday_splash_impacts,
        )

    def _compute_seasonality_strength(self, original_df, residual_df, seasonal_df):
        strength = {}
        for col in original_df.columns:
            y = original_df[col].to_numpy(dtype=float)
            resid = residual_df[col].to_numpy(dtype=float)
            seasonal = seasonal_df[col].to_numpy(dtype=float)
            mask = ~(np.isnan(y) | np.isnan(resid) | np.isnan(seasonal))
            if mask.sum() < 2:
                strength[col] = 0.0
                continue
            y_clean = y[mask]
            resid_clean = resid[mask]
            seasonal_clean = seasonal[mask]
            total_var = np.var(y_clean)
            resid_var = np.var(resid_clean)
            r_squared = (
                0.0 if total_var == 0 else max(0.0, min(1.0, 1 - resid_var / total_var))
            )
            if len(seasonal_clean) > 1:
                corr = self._safe_correlation(y_clean, seasonal_clean)
                corr_strength = max(0.0, corr**2) if np.isfinite(corr) else 0.0
            else:
                corr_strength = 0.0
            variance_ratio = (
                0.0 if total_var == 0 else min(1.0, np.var(seasonal_clean) / total_var)
            )
            combined = 0.6 * r_squared + 0.3 * corr_strength + 0.1 * variance_ratio
            strength[col] = max(0.0, min(1.0, combined))
        return strength

    def _estimate_seasonality_profile(self, seasonal_series, series_scale):
        """
        Estimate relative strength of weekly, yearly, and detected periodic signatures.

        Parameters
        ----------
        seasonal_series : pd.Series
            Estimated seasonal component for a single series.
        series_scale : float
            Standard deviation of the original series used for normalization.

        Returns
        -------
        dict
            Dictionary containing combined, weekly, yearly, and optional
            `period_{n}` strength estimates when adaptive Fourier periods are available.
        """
        if series_scale is None or not np.isfinite(series_scale) or series_scale == 0:
            series_scale = 1.0

        if not isinstance(seasonal_series, pd.Series):
            seasonal_series = pd.Series(seasonal_series, index=self.date_index)
        seasonal_series = seasonal_series.astype(float)

        valid = seasonal_series.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            return {'combined': 0.0, 'weekly': 0.0, 'yearly': 0.0}

        combined_strength = float(np.nanstd(valid)) / series_scale

        weekly_strength = 0.0
        if valid.size >= 7:
            weekly_groups = valid.groupby(valid.index.dayofweek).mean()
            if len(weekly_groups) > 1:
                weekly_strength = float(np.nanstd(weekly_groups)) / series_scale

        yearly_strength = 0.0
        date_range_days = (
            (valid.index[-1] - valid.index[0]).days if len(valid.index) > 1 else 0
        )
        if date_range_days >= 180:
            yearly_groups = valid.groupby(valid.index.dayofyear).mean()
            if len(yearly_groups) > 1:
                yearly_strength = float(np.nanstd(yearly_groups)) / series_scale

        profile = {
            'combined': combined_strength,
            'weekly': weekly_strength,
            'yearly': yearly_strength,
        }

        if self.detected_seasonal_periods:
            total_var = float(np.nanvar(valid))
            for period, _ in self.detected_seasonal_periods:
                period_int = int(round(period))
                if period_int < 2 or period_int >= len(valid):
                    continue
                groups = valid.groupby(np.arange(len(valid)) % period_int).mean()
                if len(groups) > 1:
                    period_var = float(np.nanvar(groups))
                    period_strength = period_var / (total_var + 1e-12)
                    profile[f'period_{period_int}'] = min(1.0, period_strength)

        return profile

    def _detect_seasonality_changepoints(self, df, seasonal_component):
        """Detect dates where seasonality strength changes materially over time."""
        n = len(df)
        window_size = min(365, max(60, n // 3))
        stride = max(1, window_size // 2)

        changepoints = {}
        for col in df.columns:
            strengths = []
            dates = []
            for start in range(0, n - window_size, stride):
                end = start + window_size
                window_df = df.iloc[start:end]
                window_seasonal = seasonal_component.iloc[start:end]
                window_residual = window_df - window_seasonal
                strength = self._compute_seasonality_strength(
                    window_df[[col]], window_residual[[col]], window_seasonal[[col]]
                )
                strengths.append(strength.get(col, 0.0))
                dates.append(df.index[start + window_size // 2])

            col_cps = []
            for i in range(1, len(strengths)):
                prev_strength = strengths[i - 1]
                curr_strength = strengths[i]
                abs_change = abs(curr_strength - prev_strength)
                rel_denom = max(prev_strength, curr_strength, 0.02)
                rel_change = abs_change / rel_denom
                # Require meaningful absolute and relative deltas to avoid
                # near-zero baseline windows over-triggering changepoints.
                if abs_change >= 0.03 and rel_change > 0.3:
                    col_cps.append(
                        (
                            dates[i],
                            f"strength_change_{prev_strength:.2f}_to_{curr_strength:.2f}",
                        )
                    )
            changepoints[col] = col_cps
        return changepoints

    @staticmethod
    def _infer_season_type(seasonality_profile, seasonality_changepoints):
        weekly = float((seasonality_profile or {}).get('weekly', 0.0) or 0.0)
        yearly = float((seasonality_profile or {}).get('yearly', 0.0) or 0.0)
        combined = float((seasonality_profile or {}).get('combined', 0.0) or 0.0)
        cp_count = len(seasonality_changepoints or [])

        if cp_count >= 2 and combined >= 0.005:
            return 'seasonality_changepoints'
        if cp_count == 1 and combined >= 0.005:
            return 'time_varying_seasonality'
        if weekly >= 0.01 and yearly >= 0.02:
            return 'weekly_yearly'
        if weekly >= 0.01:
            return 'weekly'
        if yearly >= 0.02:
            return 'yearly'
        return 'none'
