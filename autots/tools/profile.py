"""
Profiling
"""

import numpy as np
import pandas as pd
from autots.tools.seasonal import (
    date_part,
    create_changepoint_features,
    half_yr_spacing,
)
from autots.models.basics import BasicLinearModel


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
    """Summarize time series data.

    Args:
        df (pd.DataFrame): wide style data with datetimeindex
    """
    df_sum = df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    df_sum.loc["count_non_zero"] = (df != 0).sum()
    df_sum.loc["cv_squared"] = (
        df_sum.loc["std"] / df_sum.loc["mean"].replace(0, 1)
    ) ** 2
    df_sum.loc["adi"] = (df.shape[0] / df_sum.loc["count_non_zero"].replace(0, 1)) ** 2
    first_non_nan_index = df.replace(0, np.nan).reset_index(drop=True).notna().idxmax()
    try:
        df_sum.loc["autocorr_1"] = np.diag(
            np.corrcoef(df.bfill().T, df.shift(1).bfill().T)[
                : df.shape[1], df.shape[1] :
            ]
        )
    except Exception as e:
        print(f"summarize_series autocorr_1 failed with {repr(e)}")
    df_sum.loc["null_percentage"] = (first_non_nan_index / df.shape[0]).fillna(1)
    diffs = df.diff().iloc[1:]  # Exclude the first row with NaN
    zero_diffs = (diffs == 0).sum()
    total_diffs = df.shape[0] - 1  # Number of differences per series
    total_diffs = total_diffs if total_diffs > 0 else 1
    zero_diff_proportions = zero_diffs / total_diffs
    df_sum.loc['zero_diff_proportion'] = zero_diff_proportions
    try:
        mod = BasicLinearModel(changepoint_spacing=None)
        mod.fit(df.ffill().bfill())
        summary = mod.coefficient_summary(df.ffill().bfill())
        df_sum = pd.concat([df_sum, summary.transpose()])
    except Exception as e:
        df_sum.loc["season_trend_percent"] = 0
        print(f"summarize_series BasicLinearModel decomposition failed with {repr(e)}")
    return df_sum


def profile_time_series(
    df,
    adi_threshold=1.3,
    cvar_threshold=0.5,
    flat_threshold=0.92,
    new_product_threshold='auto',
    seasonal_threshold=0.5,
):
    """
    Profiles time series data into categories:
        smooth, intermittent, erratic, lumpy, flat, new_product

    Args:
        df (pd.DataFrame): Wide format DataFrame with datetime index and each column as a time series.
        new_product_threshold (float): one of the more finiky thresholds, percent of null or zero data from beginning to declare new product
        new_product_correct (bool): use dt index to correct
    Returns:
        pd.DataFrame: DataFrame with 'SERIES' and 'DEMAND_PROFILE' columns.
    """

    metrics_df = summarize_series(df).transpose()

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
    metrics_df.loc[
        (metrics_df['adi'] >= adi_threshold)
        & (metrics_df['cv_squared'] < cvar_threshold),
        'PROFILE',
    ] = 'intermittent'
    metrics_df.loc[
        (metrics_df['adi'] < adi_threshold)
        & (metrics_df['cv_squared'] >= cvar_threshold),
        'PROFILE',
    ] = 'erratic'
    metrics_df.loc[
        (metrics_df['adi'] >= adi_threshold)
        & (metrics_df['cv_squared'] >= cvar_threshold),
        'PROFILE',
    ] = 'lumpy'
    metrics_df.loc[metrics_df['zero_diff_proportion'] >= flat_threshold, 'PROFILE'] = (
        'flat'
    )
    metrics_df.loc[
        metrics_df['null_percentage'] >= new_product_threshold, 'PROFILE'
    ] = 'new_product'
    metrics_df.loc[
        metrics_df['season_trend_percent'] > seasonal_threshold, 'PROFILE'
    ] = "seasonal"

    # Reset index to get 'SERIES' column
    intermittence_df = (
        metrics_df[['PROFILE']].reset_index().rename(columns={'index': 'SERIES'})
    )

    return intermittence_df


# burst, stationary, seasonality
