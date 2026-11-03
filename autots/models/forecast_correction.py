"""Per-series forecast corrections fit on backtest hindcasts.

Adapts the two corrections of probabilistic bias correction (Guan et al. 2026,
Nature, "Advancing subseasonal forecasting with machine learning") from quantile
CDFs to point forecasts of wide data:

- debias (Debias++ analog): subtract the shrunk historical bias of the base
  forecast average, as a constant or linear-in-horizon curve per series.
- persistence (Persistence++ analog): per-series ridge regression of the
  residual from the base average on recent level, recent-cycle climatology and
  each base model's deviation from the average.

Both are solved in closed form for all series at once. Everything is expressed
in scale-free units, (x - last_value) / scale, so folds at different levels pool
cleanly and one penalty means the same thing on every series. Zero coefficients
reproduce the simple base-model average, so with too little hindcast data the
corrections shrink back to that average rather than overfit.

Array conventions: hindcasts are stacked as (fold, model, horizon, series);
contexts are dicts of per-fold arrays from `forecast_context`.
"""

import numpy as np
import pandas as pd

# Points of recent history used for the seasonal-key climatology, by key.
# Long enough to average out noise, short enough to track the current level.
CLIMATOLOGY_LOOKBACK_POINTS = {
    "hour": 24 * 28,
    "dayofweek": 7 * 12,
    "week": 52 * 3,
    "month": 12 * 4,
}


def _seasonal_key_name(datetime_index):
    """Pick a calendar key from the median sampling interval."""
    if len(datetime_index) < 2:
        return None
    median_step = pd.Series(datetime_index).diff().median()
    if pd.isna(median_step):
        return None
    if median_step <= pd.Timedelta(hours=1.5):
        return "hour"
    if median_step <= pd.Timedelta(days=1.5):
        return "dayofweek"
    if median_step <= pd.Timedelta(days=8):
        return "week"
    if median_step <= pd.Timedelta(days=32):
        return "month"
    return None


def _seasonal_keys(datetime_index, key_name):
    if key_name == "week":
        return np.asarray(datetime_index.isocalendar().week, dtype=int)
    return np.asarray(getattr(datetime_index, key_name), dtype=int)


def climatology_over_horizon(history, forecast_index):
    """Recent-cycle seasonal means mapped onto forecast dates, shape (H, N).

    Falls back to the recent mean where the frequency has no calendar key or a
    forecast key never occurs in the lookback window.
    """
    key_name = _seasonal_key_name(history.index)
    horizon = len(forecast_index)
    lookback = CLIMATOLOGY_LOOKBACK_POINTS.get(key_name, max(horizon, 28))
    recent = history.iloc[-max(lookback, horizon) :]
    recent_mean = recent.mean().to_numpy(dtype=float)
    if key_name is None:
        return np.broadcast_to(recent_mean, (horizon, history.shape[1])).copy()
    seasonal_means = recent.groupby(_seasonal_keys(recent.index, key_name)).mean()
    climatology = seasonal_means.reindex(
        _seasonal_keys(forecast_index, key_name)
    ).to_numpy(dtype=float)
    return np.where(np.isnan(climatology), recent_mean, climatology)


def forecast_context(history, forecast_index):
    """Origin-time quantities shared by both corrections for one fold.

    `scale` is the median absolute one-step change: robust to level and
    outliers, and the natural unit of forecast error. Flat series fall back to
    the standard deviation, then to 1, so the transform stays invertible.
    """
    horizon = len(forecast_index)
    values = history.to_numpy(dtype=float)
    last_value = history.ffill().iloc[-1].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        scale = np.nanmedian(np.abs(np.diff(values, axis=0)), axis=0)
        fallback_scale = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback_scale)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    # A gap at the tail must not NaN the whole series' features; only a fully
    # missing series (NaN last_value) stays NaN.
    recent_mean = history.iloc[-horizon:].mean().to_numpy(dtype=float)
    recent_mean = np.where(np.isnan(recent_mean), last_value, recent_mean)
    climatology = climatology_over_horizon(history, forecast_index)
    climatology = np.where(np.isnan(climatology), recent_mean, climatology)
    return {
        "last_value": last_value,
        "scale": scale,
        "recent_mean": recent_mean,
        "climatology": climatology,
    }


def _standardize(values, context):
    return (values - context["last_value"]) / context["scale"]


def _destandardize(values, context):
    return values * context["scale"] + context["last_value"]


def batched_ridge_per_series(design, target, row_mask, penalty):
    """Solve one ridge regression per series in a single batched call.

    Args:
        design: (N, R, P) per-series rows; target: (N, R); row_mask: (N, R) bool.
        penalty: (N,) or scalar ridge strength applied to every coefficient.

    Masked rows are zeroed before any product, since NaN * 0 is still NaN. A
    series without valid rows solves to zero coefficients (the prior).
    """
    design = np.where(row_mask[..., None], design, 0.0)
    target = np.where(row_mask, target, 0.0)
    n_features = design.shape[2]
    gram = np.einsum("nrp,nrq->npq", design, design)
    ridge = np.broadcast_to(np.asarray(penalty, dtype=float), (design.shape[0],))
    # Floor keeps empty or rank-deficient series solvable; it is far below any
    # meaningful penalty on unit-RMS features.
    gram = gram + (ridge[:, None, None] + 1e-9) * np.eye(n_features)
    moment = np.einsum("nrp,nr->np", design, target)
    return np.linalg.solve(gram, moment[..., None])[..., 0]


def _persistence_design(base_forecasts_z, context_z):
    """Features (..., H, N, P) relative to the base average, so 0 = average."""
    base_average = base_forecasts_z.mean(axis=-3)
    horizon_shape = base_average.shape
    features = [
        np.ones(horizon_shape),
        np.broadcast_to(context_z["recent_mean"][..., None, :], horizon_shape)
        - base_average,
        context_z["climatology"] - base_average,
    ]
    model_deviation = base_forecasts_z - base_average[..., None, :, :]
    features.extend(np.moveaxis(model_deviation, -3, 0))
    return np.stack(features, axis=-1), base_average


def _context_in_z(context):
    """Context quantities of a single origin in standardized units."""
    return {
        "recent_mean": _standardize(context["recent_mean"], context),
        "climatology": _standardize(context["climatology"], context),
    }


def _stack_contexts(contexts):
    return {k: np.stack([c[k] for c in contexts]) for k in contexts[0]}


class ForecastCorrector:
    """Fit debias and/or persistence corrections on hindcasts, apply to new forecasts.

    Args:
        correction_method: 'pbc' (average of both), 'debias', or 'persistence'.
        bias_shape: 'constant' or 'linear_horizon' bias curve for debias.
        bias_shrinkage: pseudo-fold count k; bias is scaled by V / (V + k).
        ridge_lambda: persistence ridge strength per row on unit-RMS features.
    """

    def __init__(
        self,
        correction_method="pbc",
        bias_shape="linear_horizon",
        bias_shrinkage=1.0,
        ridge_lambda=1.0,
    ):
        if correction_method not in ("pbc", "debias", "persistence"):
            raise ValueError(f"unknown correction_method {correction_method}")
        if bias_shape not in ("constant", "linear_horizon"):
            raise ValueError(f"unknown bias_shape {bias_shape}")
        self.correction_method = correction_method
        self.bias_shape = bias_shape
        self.bias_shrinkage = float(bias_shrinkage)
        self.ridge_lambda = float(ridge_lambda)

    def _bias_design(self, horizon):
        relative_step = (np.arange(horizon) + 1) / horizon
        if self.bias_shape == "constant":
            return np.ones((horizon, 1))
        return np.column_stack([np.ones(horizon), relative_step])

    def fit(self, base_forecasts, actuals, contexts):
        """Args: base_forecasts (V, M, H, N), actuals (V, H, N), contexts list of V."""
        context = _stack_contexts(contexts)
        # (V, N) origin quantities broadcast over the horizon (and model) axes.
        per_horizon = {
            "last_value": context["last_value"][:, None, :],
            "scale": context["scale"][:, None, :],
        }
        per_model = {k: v[:, None] for k, v in per_horizon.items()}
        forecasts_z = _standardize(base_forecasts, per_model)
        actuals_z = _standardize(actuals, per_horizon)
        context_z = {
            "recent_mean": _standardize(context["recent_mean"], context),
            "climatology": _standardize(context["climatology"], per_horizon),
        }
        design, base_average = _persistence_design(forecasts_z, context_z)
        residual = actuals_z - base_average
        valid = np.isfinite(residual) & np.isfinite(design).all(axis=-1)
        n_folds, horizon, n_series = residual.shape

        if self.correction_method in ("pbc", "debias"):
            # Count folds, not rows: errors within one fold are strongly correlated.
            folds_with_data = valid.any(axis=1).sum(axis=0)
            shrink = folds_with_data / (folds_with_data + self.bias_shrinkage)
            curve = self._bias_design(horizon)
            bias_design = np.broadcast_to(
                curve[None, :, None, :], (n_folds, horizon, n_series, curve.shape[1])
            )
            # Tiny ridge only for solvability; shrinkage is applied explicitly.
            coefficients = batched_ridge_per_series(
                np.moveaxis(bias_design, 2, 0).reshape(n_series, n_folds * horizon, -1),
                np.moveaxis(residual, 2, 0).reshape(n_series, -1),
                np.moveaxis(valid, 2, 0).reshape(n_series, -1),
                0.0,
            )
            self.bias_coefficients = coefficients * shrink[:, None]

        if self.correction_method in ("pbc", "persistence"):
            rows = np.moveaxis(design, 2, 0).reshape(n_series, n_folds * horizon, -1)
            row_valid = np.moveaxis(valid, 2, 0).reshape(n_series, -1)
            safe_rows = np.where(row_valid[..., None], rows, 0.0)
            n_valid = row_valid.sum(axis=1)
            # Unit-RMS columns make ridge_lambda comparable across features and series.
            feature_rms = np.sqrt(
                (safe_rows**2).sum(axis=1) / np.maximum(n_valid, 1)[:, None]
            )
            feature_rms = np.where(feature_rms > 0, feature_rms, 1.0)
            standardized_coefficients = batched_ridge_per_series(
                rows / feature_rms[:, None, :],
                np.moveaxis(residual, 2, 0).reshape(n_series, -1),
                row_valid,
                self.ridge_lambda * n_valid,
            )
            self.persistence_coefficients = standardized_coefficients / feature_rms
        return self

    def predict(self, base_forecasts, context):
        """Args: base_forecasts (M, H, N) for the live origin; context from forecast_context."""
        forecasts_z = _standardize(base_forecasts, context)
        context_z = _context_in_z(context)
        design, base_average = _persistence_design(forecasts_z, context_z)
        corrected = []
        if self.correction_method in ("pbc", "debias"):
            bias = self._bias_design(base_average.shape[0]) @ self.bias_coefficients.T
            corrected.append(base_average + bias)
        if self.correction_method in ("pbc", "persistence"):
            adjustment = np.einsum("hnp,np->hn", design, self.persistence_coefficients)
            corrected.append(base_average + adjustment)
        return _destandardize(np.mean(corrected, axis=0), context)


def base_model_intervals(point, base_forecasts, base_upper, base_lower):
    """Recentre the base models' average interval widths on the corrected point."""
    upper_width = np.clip(np.nanmean(base_upper - base_forecasts, axis=0), 0, None)
    lower_width = np.clip(np.nanmean(base_forecasts - base_lower, axis=0), 0, None)
    return point + upper_width, point - lower_width
