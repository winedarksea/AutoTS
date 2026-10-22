"""
Profiling
"""

import warnings

import numpy as np
import pandas as pd
from autots.tools.seasonal import date_part
from autots.tools.changepoints import (
    create_changepoint_features,
    half_yr_spacing,
)
from autots.models.basics import BasicLinearModel


class ProfileWarning(UserWarning):
    """A profiling operation failed and its existing fallback was used."""


def _lag_one_autocorrelation(df):
    """Match the legacy back-filled Pearson correlations without a pairwise matrix."""
    values = df.bfill().to_numpy(dtype=np.float64, copy=True)
    lagged_values = df.shift(1).bfill().to_numpy(dtype=np.float64, copy=True)
    # Ordinary means preserve NaN propagation from unfilled trailing observations.
    values -= values.mean(axis=0)
    lagged_values -= lagged_values.mean(axis=0)
    cross_products = np.einsum('ij,ij->j', values, lagged_values)
    squared_values = np.einsum('ij,ij->j', values, values)
    squared_lagged_values = np.einsum('ij,ij->j', lagged_values, lagged_values)
    with np.errstate(divide='ignore', invalid='ignore'):
        correlations = cross_products / np.sqrt(squared_values)
        correlations /= np.sqrt(squared_lagged_values)
    return np.clip(correlations, -1, 1)


def data_profile(df):
    """Legacy profiler.
    Input: a pd DataFrame of columns which are time series, and a datetime index

    Output: a pd DataFrame of column per time series, with rows which are statistics
    """

    a = pd.DataFrame(df.min(skipna=True)).transpose()
    b = pd.DataFrame(df.mean(skipna=True)).transpose()
    c = pd.DataFrame(df.median(skipna=True)).transpose()
    d = pd.DataFrame(df.max(skipna=True)).transpose()
    e = pd.DataFrame(df.notna().idxmax()).transpose()
    f = pd.DataFrame(df.notna()[::-1].idxmax()).transpose()
    g = f - e
    h = pd.DataFrame(df.isnull().sum() * 100 / len(df)).transpose()
    profile_df = pd.concat([a, b, c, d, e, f, g, h], ignore_index=True, sort=True)
    profile_df.index = [
        'min',
        'mean',
        'median',
        'max',
        'FirstDate',
        'LastDate',
        'LengthDays',
        "PercentNA",
    ]

    return profile_df


def summarize_series(df):
    """Return statistics as rows and input time series as columns.

    Failed autocorrelation or decomposition operations emit ProfileWarning and
    retain their existing batch-wide fallbacks.

    Args:
        df (pd.DataFrame): wide style data with datetimeindex
    """
    df_sum = df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    df_filled = df.ffill().bfill()
    df_sum.loc["count_non_zero"] = ((df != 0) & df.notna()).sum()
    df_sum.loc["cv_squared"] = (df_sum.loc["std"] / df.abs().mean().replace(0, 1)) ** 2
    df_sum.loc["adi"] = (
        df_sum.loc["count"] / df_sum.loc["count_non_zero"].replace(0, 1)
    ) ** 2

    # Calculate ADI for first and second halves to avoid false positives from leading/trailing zeros
    mid_point = df.shape[0] // 2
    df_first_half = df.iloc[:mid_point]
    df_second_half = df.iloc[mid_point:]

    for half_df, suffix in [
        (df_first_half, "_first_half"),
        (df_second_half, "_second_half"),
    ]:
        count_valid = half_df.notna().sum()
        count_non_zero_half = ((half_df != 0) & half_df.notna()).sum()
        # If entire half is NaN, set adi to 0 (will pass any threshold check)
        adi_half = np.where(
            count_valid == 0, 0, (count_valid / count_non_zero_half.replace(0, 1)) ** 2
        )
        df_sum.loc[f"adi{suffix}"] = adi_half

    first_non_nan_index = df.replace(0, np.nan).reset_index(drop=True).notna().idxmax()
    try:
        df_sum.loc["autocorr_1"] = _lag_one_autocorrelation(df)
    except Exception as e:
        warnings.warn(
            f"summarize_series autocorr_1 failed with {repr(e)}",
            ProfileWarning,
            stacklevel=2,
        )
    df_sum.loc["null_percentage"] = (first_non_nan_index / df.shape[0]).fillna(1)
    diffs = df.diff().iloc[1:]
    valid_pairs = (df.notna() & df.shift(1).notna()).iloc[1:]
    zero_diffs = ((diffs == 0) & valid_pairs).sum()
    total_valid_diffs = valid_pairs.sum()
    zero_diff_proportions = zero_diffs / total_valid_diffs.replace(0, np.nan)
    zero_diff_proportions = zero_diff_proportions.fillna(0)
    df_sum.loc['zero_diff_proportion'] = zero_diff_proportions
    df_sum.loc["deseasonalized_mean"] = df_sum.loc["mean"]
    try:
        mod = BasicLinearModel(changepoint_spacing=None)
        mod.fit(df_filled)
        summary = mod.coefficient_summary(df_filled)
        seasonal_contrib, _, _ = mod.return_components(df_filled)
        seasonal_df = pd.DataFrame(
            seasonal_contrib, index=df_filled.index, columns=df_filled.columns
        )
        # Remove explicit seasonal structure so volatility reflects underlying noise
        deseasonalized = df_filled - seasonal_df
        deseasonalized_mean = deseasonalized.mean()
        df_sum.loc["deseasonalized_mean"] = deseasonalized_mean
        deseasonalized_std = deseasonalized.std()
        df_sum.loc["cv_squared"] = (
            deseasonalized_std / deseasonalized.abs().mean().replace(0, 1)
        ) ** 2
        df_sum = pd.concat([df_sum, summary.transpose()])
    except Exception as e:
        df_sum.loc["season_trend_percent"] = 0
        warnings.warn(
            f"summarize_series BasicLinearModel decomposition failed with {repr(e)}",
            ProfileWarning,
            stacklevel=2,
        )
    return df_sum


def profile_time_series(
    df,
    adi_threshold=1.2,
    cvar_threshold=0.5,
    flat_threshold=0.95,
    new_product_threshold='auto',
    seasonal_threshold=0.46,
    drift_trend_threshold=0.6,
    drift_autocorr_threshold=0.9,
):
    """
    Profiles time series data into categories:
        smooth: series driven mostly by trend, generally series that don't fit other categories
        intermittent: occasional demand, one-sided spikes around median, usually many zeroes
        binary: only two values, often 0 and 1 (also includes trinary, three states)
        smooth_drift: smooth series whose dynamics are predominantly changepoint/trend driven with strong first-order autocorrelation
        erratic: high volatility series
        flat: generally constant with only occasional movements
        new_product: limited data, series just started receiving data
        highly_seasonal: strong seasonal component dominates series

    Args:
        df (pd.DataFrame): Wide format DataFrame with datetime index and each column as a time series.
        new_product_threshold (float or str): leading null-or-zero fraction required
            for new_product. 'auto' resolves from the index length and elapsed span.
        drift_trend_threshold (float): minimum proportion of trend/changepoint contribution required to flag a smooth series as smooth_drift
        drift_autocorr_threshold (float): minimum lag-1 autocorrelation required to flag a smooth series as smooth_drift
    Returns:
        pd.DataFrame: DataFrame with 'SERIES' and 'PROFILE' columns.

    Precedence, highest first: highly_seasonal, new_product, flat, binary,
    erratic, intermittent, smooth_drift, then smooth. Use summarize_series for
    the underlying metrics. ProfileWarning reports batch-wide fallback use.
    """

    metrics_df = summarize_series(df).transpose()
    value_counts = df.nunique(dropna=True)

    # Initialize demand profile as 'smooth'
    metrics_df['PROFILE'] = 'smooth'

    if new_product_threshold == "auto":
        half_yr_space = half_yr_spacing(df)
        new_product_threshold = 1 - (half_yr_space * 0.65 / df.shape[0])
        if new_product_threshold < 0.85:
            new_product_threshold = 0.85
        if new_product_threshold > 0.99:
            new_product_threshold = 0.99
    # Apply conditions to classify the demand profiles
    # Series is intermittent if overall ADI meets threshold AND both non-empty halves meet threshold
    # (halves with all NaN are ignored, not required to pass)
    metrics_df.loc[
        (metrics_df['adi'] >= adi_threshold)
        & (
            (metrics_df['adi_first_half'] == 0)
            | (metrics_df['adi_first_half'] >= adi_threshold)
        )
        & (
            (metrics_df['adi_second_half'] == 0)
            | (metrics_df['adi_second_half'] >= adi_threshold)
        ),
        'PROFILE',
    ] = 'intermittent'
    metrics_df.loc[
        (metrics_df['adi'] < adi_threshold)
        & (metrics_df['cv_squared'] >= cvar_threshold),
        'PROFILE',
    ] = 'erratic'
    metrics_df.loc[
        (value_counts <= 3) & (value_counts > 1),
        'PROFILE',
    ] = 'binary'
    metrics_df.loc[metrics_df['zero_diff_proportion'] >= flat_threshold, 'PROFILE'] = (
        'flat'
    )
    metrics_df.loc[
        metrics_df['null_percentage'] >= new_product_threshold, 'PROFILE'
    ] = 'new_product'
    metrics_df.loc[
        metrics_df['season_trend_percent'] > seasonal_threshold, 'PROFILE'
    ] = "highly_seasonal"

    # Identify smooth series dominated by drift-like trend contributions
    trend_strength = 1 - metrics_df.get(
        'season_trend_percent', pd.Series(0, index=metrics_df.index)
    ).fillna(0)
    autocorr = metrics_df.get(
        'autocorr_1', pd.Series(0, index=metrics_df.index)
    ).fillna(0)
    drift_mask = (
        (metrics_df['PROFILE'] == 'smooth')
        & (trend_strength >= drift_trend_threshold)
        & (autocorr >= drift_autocorr_threshold)
    )
    metrics_df.loc[drift_mask, 'PROFILE'] = 'smooth_drift'

    # Reset index to get 'SERIES' column
    intermittence_df = metrics_df[['PROFILE']].rename_axis('SERIES').reset_index()

    return intermittence_df
