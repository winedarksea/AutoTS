# -*- coding: utf-8 -*-
"""HolidayMixin for TimeSeriesFeatureDetector."""

import numpy as np
import pandas as pd
import copy
import warnings
from autots.evaluator.anomaly_detector import HolidayDetector
from autots.tools.holiday import holiday_flag


class HolidayMixin:
    """Mixin providing holiday detection, coefficient solving, and parameter sanitization."""

    @staticmethod
    def _normalize_holiday_country_value(value):
        """Normalize holiday country config to None, string, or list of strings."""
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            cleaned = [
                str(x).upper() for x in value if x is not None and str(x).strip()
            ]
            return cleaned if cleaned else None
        if isinstance(value, dict):
            cleaned = [
                str(x).upper() for x in value.keys() if x is not None and str(x).strip()
            ]
            return cleaned if cleaned else None
        value = str(value).strip()
        if not value:
            return None
        return value.upper()

    @staticmethod
    def _holiday_country_label(country_value):
        """Create a stable label for calendar holiday feature column prefixes."""
        normalized = HolidayMixin._normalize_holiday_country_value(country_value)
        if normalized is None:
            return None
        if isinstance(normalized, list):
            return "_".join(normalized)
        return normalized

    def _resolve_holiday_country_map(self, columns):
        """Resolve per-series holiday countries from shared default and overrides."""
        default_country = self._normalize_holiday_country_value(
            getattr(self, 'holiday_country', None)
        )
        override_map = getattr(self, 'holiday_countries', None) or {}
        resolved = {}
        for col in columns:
            if col in override_map:
                resolved[col] = self._normalize_holiday_country_value(override_map[col])
            else:
                resolved[col] = default_country
        self._resolved_holiday_countries = resolved
        return resolved

    def _build_calendar_holiday_features(self, dates, columns):
        """Build shared calendar regressors and per-series flags from holiday countries."""
        index = pd.DatetimeIndex(dates)
        columns = list(columns)
        resolved = self._resolve_holiday_country_map(columns)
        calendar_flags = pd.DataFrame(0.0, index=index, columns=columns)
        regressor_frames = []
        feature_cache = {}

        for series_name in columns:
            country_value = resolved.get(series_name)
            if country_value is None:
                continue
            label = self._holiday_country_label(country_value)
            if not label:
                continue
            if label not in feature_cache:
                try:
                    country_features = holiday_flag(
                        index,
                        country=country_value,
                        encode_holiday_type=True,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Calendar holiday fetch failed for country '{country_value}': {exc}. "
                        "Holiday features for this country will be skipped.",
                        RuntimeWarning,
                    )
                    country_features = pd.DataFrame(index=index)
                if isinstance(country_features, pd.Series):
                    country_features = country_features.to_frame()
                if country_features is None or country_features.empty:
                    country_features = pd.DataFrame(index=index)
                else:
                    country_features = (
                        country_features.reindex(index)
                        .fillna(0.0)
                        .astype(float)
                        .loc[:, ~country_features.columns.duplicated()]
                    )
                    country_features = country_features.rename(
                        columns=lambda x: f"calendar_{label}__{x}"
                    )
                feature_cache[label] = country_features
                if not country_features.empty:
                    regressor_frames.append(country_features)
            country_features = feature_cache.get(label)
            if country_features is not None and not country_features.empty:
                calendar_flags[series_name] = (
                    country_features.sum(axis=1).to_numpy(dtype=float) > 0
                ).astype(float)

        if regressor_frames:
            calendar_regressors = pd.concat(regressor_frames, axis=1)
            calendar_regressors = calendar_regressors.loc[
                :, ~calendar_regressors.columns.duplicated()
            ]
        else:
            calendar_regressors = pd.DataFrame(index=index)
        return calendar_flags, calendar_regressors

    @staticmethod
    def _merge_holiday_regressors(*frames):
        """Concatenate holiday regressor blocks into one aligned design matrix."""
        valid_frames = []
        for frame in frames:
            if frame is None:
                continue
            if isinstance(frame, pd.Series):
                frame = frame.to_frame()
            if not isinstance(frame, pd.DataFrame):
                continue
            if frame.empty and frame.columns.empty:
                continue
            valid_frames.append(frame)
        if not valid_frames:
            return pd.DataFrame()
        merged = pd.concat(valid_frames, axis=1)
        merged = merged.fillna(0.0).astype(float)
        return merged.loc[:, ~merged.columns.duplicated()]

    @staticmethod
    def _merge_holiday_dates_by_series(base_dates, added_flags):
        """Union anomaly-derived and calendar-derived holiday dates by series."""
        merged = {
            series_name: [pd.Timestamp(x) for x in dates]
            for series_name, dates in (base_dates or {}).items()
        }
        if added_flags is None or added_flags.empty:
            return merged
        for series_name in added_flags.columns:
            existing = {pd.Timestamp(x) for x in merged.get(series_name, [])}
            series_flags = added_flags[series_name]
            existing.update(
                pd.Timestamp(x) for x in series_flags[series_flags > 0].index
            )
            merged[series_name] = sorted(existing)
        return merged

    def _build_holiday_regressors_for_index(
        self, dates, columns, include_anomaly_rules=True
    ):
        """Build the merged holiday regressor design matrix for any date index."""
        index = pd.DatetimeIndex(dates)
        anomaly_regressors = pd.DataFrame(index=index)
        if (
            include_anomaly_rules
            and getattr(self, 'holiday_detector', None) is not None
        ):
            try:
                anomaly_regressors = self.holiday_detector.dates_to_holidays(
                    index, style='flag'
                )
                if anomaly_regressors is None:
                    anomaly_regressors = pd.DataFrame(index=index)
                else:
                    anomaly_regressors = (
                        anomaly_regressors.reindex(index)
                        .fillna(0.0)
                        .astype(float)
                        .loc[:, ~anomaly_regressors.columns.duplicated()]
                    )
            except Exception:
                anomaly_regressors = pd.DataFrame(index=index)

        _, calendar_regressors = self._build_calendar_holiday_features(index, columns)
        merged = self._merge_holiday_regressors(anomaly_regressors, calendar_regressors)
        if merged.empty:
            return pd.DataFrame(index=index)
        return merged.reindex(index).fillna(0.0)

    def _sanitize_holiday_params(self, holiday_params):
        """Return holiday detector parameters filtered to supported keys."""
        default_params = {
            'anomaly_detector_params': {
                'method': 'mad',
                'transform_dict': None,
                'forecast_params': None,
                'method_params': {'distribution': 'uniform', 'alpha': 0.05},
            },
            'threshold': 0.8,
            'min_occurrences': 2,
            'splash_threshold': None,
            'use_dayofmonth_holidays': True,
            'use_wkdom_holidays': True,
            'use_wkdeom_holidays': False,
            'use_lunar_holidays': False,
            'use_lunar_weekday': False,
            'use_islamic_holidays': True,
            'use_hebrew_holidays': False,
            'use_hindu_holidays': False,
            'auto_relax': False,
            'relax_threshold_floor': 0.55,
            'relax_splash_threshold': 0.55,
            'relax_rounds': 2,
            'min_holidays_per_series': 1,
            'max_holidays_per_series': None,
            'holiday_selection_strategy': 'score',
            'output': self.detection_mode,  # Use instance's detection_mode
            'n_jobs': 1,
        }

        if holiday_params is None:
            return default_params

        allowed_keys = set(default_params.keys())
        sanitized = default_params.copy()
        unsupported = []

        for key, value in holiday_params.items():
            if key in allowed_keys:
                sanitized[key] = value
            else:
                unsupported.append(key)

        if unsupported:
            warnings.warn(
                f"Ignoring unsupported holiday_params keys: {sorted(set(unsupported))}",
                RuntimeWarning,
            )

        # Override output to match detection_mode
        sanitized['output'] = self.detection_mode

        return sanitized

    def _detect_holidays(self, residual_df):
        """
        Detect holidays using HolidayDetector.

        Returns both core holiday dates and splash/bridge impacts in separate structures
        to align with synthetic generator output.
        """
        self.holiday_detector = HolidayDetector(**self.holiday_params)
        holiday_regressors = pd.DataFrame(index=residual_df.index)
        try:
            self.holiday_detector.detect(residual_df)
            holiday_flags = self.holiday_detector.dates_to_holidays(
                residual_df.index, style='series_flag'
            )
            holiday_regressors = self.holiday_detector.dates_to_holidays(
                residual_df.index, style='flag'
            )
            if holiday_regressors is None:
                holiday_regressors = pd.DataFrame(index=residual_df.index)
            else:
                holiday_regressors = (
                    holiday_regressors.reindex(residual_df.index)
                    .fillna(0.0)
                    .astype(float)
                )
                holiday_regressors = holiday_regressors.loc[
                    :, ~holiday_regressors.columns.duplicated()
                ]
        except Exception:
            holiday_flags = pd.DataFrame(
                0, index=residual_df.index, columns=residual_df.columns
            )
            holiday_regressors = pd.DataFrame(index=residual_df.index)

        calendar_flags, calendar_regressors = self._build_calendar_holiday_features(
            residual_df.index, residual_df.columns
        )
        holiday_regressors = self._merge_holiday_regressors(
            holiday_regressors, calendar_regressors
        )
        if holiday_regressors.empty:
            holiday_regressors = pd.DataFrame(index=residual_df.index)
        else:
            holiday_regressors = holiday_regressors.reindex(residual_df.index).fillna(
                0.0
            )

        holiday_dates = {}
        holiday_splash_dates = {}  # For storing splash/bridge days separately

        # Handle both multivariate and univariate outputs
        if self.detection_mode == 'univariate':
            # Univariate mode: single column of holiday flags for all series
            if holiday_flags.shape[1] > 0:
                holiday_col = holiday_flags.iloc[:, 0]
                flagged = holiday_col[holiday_col > 0].index
                holiday_list = [pd.Timestamp(ix) for ix in flagged]
                # In univariate mode, all series share the same holidays
                for col in residual_df.columns:
                    holiday_dates[col] = holiday_list
                    holiday_splash_dates[col] = (
                        []
                    )  # Will be populated during final seasonality fit
            else:
                for col in residual_df.columns:
                    holiday_dates[col] = []
                    holiday_splash_dates[col] = []
        else:
            # Multivariate mode: each series has its own holiday flags
            for col in residual_df.columns:
                series_flags = (
                    holiday_flags[col]
                    if col in holiday_flags
                    else pd.Series(0, index=residual_df.index)
                )
                flagged = series_flags[series_flags > 0].index
                holiday_dates[col] = [pd.Timestamp(ix) for ix in flagged]
                holiday_splash_dates[col] = (
                    []
                )  # Will be populated during final seasonality fit

        holiday_dates = self._merge_holiday_dates_by_series(
            holiday_dates, calendar_flags
        )

        return holiday_dates, holiday_splash_dates, holiday_regressors

    @staticmethod
    def _flatten_holiday_dates(holiday_dates_dict):
        """Flatten per-series holiday date lists into a single Timestamp set."""
        all_dates = set()
        if not holiday_dates_dict:
            return all_dates
        for dates in holiday_dates_dict.values():
            for date in dates:
                all_dates.add(pd.Timestamp(date))
        return all_dates

    def _solve_holiday_coefficients(self, regressor_df, holiday_component_df):
        coefficients = {col: {} for col in holiday_component_df.columns}
        if regressor_df is None or regressor_df.empty:
            return coefficients
        X = regressor_df.to_numpy(dtype=float)
        if X.size == 0:
            return coefficients
        regressor_columns = list(regressor_df.columns)
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.pinv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = None

        for idx, series_name in enumerate(holiday_component_df.columns):
            y = holiday_component_df.iloc[:, idx].to_numpy(dtype=float)
            if np.allclose(y, 0.0):
                continue
            try:
                if XtX_inv is not None:
                    beta = XtX_inv @ X.T @ y
                else:
                    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            series_coeffs = {}
            for j, value in enumerate(beta):
                if np.isfinite(value) and abs(value) > 1e-9:
                    series_coeffs[regressor_columns[j]] = float(value)
            if series_coeffs:
                coefficients[series_name] = series_coeffs
        return coefficients

    def _extract_splash_impacts(self, holiday_splash_impacts_scaled, holiday_dates):
        """
        Extract splash/bridge day impacts by filtering out core holiday dates.
        TODO: move this to HolidayDetector and use full functionality there.

        Splash days are days with holiday impacts that are NOT in the core holiday list.
        This aligns with the synthetic generator's distinction between direct holidays
        and their splash/bridge effects.
        """
        splash_impacts = {}
        for series_name, impacts_dict in holiday_splash_impacts_scaled.items():
            core_dates = set(holiday_dates.get(series_name, []))
            splash_dict = {}
            for date, impact in impacts_dict.items():
                date_ts = pd.Timestamp(date)
                # If this date has an impact but is NOT a core holiday, it's a splash day
                if date_ts not in core_dates and abs(impact) > 1e-9:
                    # Rescale to original scale
                    scale = float(self.scale_series.get(series_name, 1.0))
                    splash_dict[date_ts] = float(impact * scale)
            splash_impacts[series_name] = splash_dict
        return splash_impacts
