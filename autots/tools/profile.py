"""
Profiling
"""

import numpy as np
import pandas as pd


def data_profile(df):
    """
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


def profile_time_series(df, adi_threshold=1.3, cvar_threshold=0.5, flat_threshold=0.9, new_product_threshold=0.9):
    """
    Profiles time series data into categories: 
        smooth, intermittent, erratic, lumpy, flat, new_product

    Parameters:
    df (pd.DataFrame): Wide format DataFrame with datetime index and each column as a time series.

    Returns:
    pd.DataFrame: DataFrame with 'SERIES' and 'DEMAND_PROFILE' columns.
    """

    # Total number of time periods (e.g., weeks)
    num_weeks = df.index.nunique()

    # Compute mean and standard deviation for each series
    means = df.mean()
    stds = df.std()

    # Count of non-zero observations for each series
    non_zero_counts = (df != 0).sum()

    # Coefficient of variation squared for each series
    cv_squared = (stds / means) ** 2

    # Average Demand Interval (ADI) for each series
    adi = num_weeks / non_zero_counts

    # Create a DataFrame with all metrics
    metrics_df = pd.DataFrame({
        'mean': means,
        'std': stds,
        'cv_squared': cv_squared,
        'adi': adi
    })

    # Find non zero or NaN index for new product estimation
    first_non_nan_index = df.replace(0, np.nan).reset_index(drop=True).apply(lambda row: row.first_valid_index(), axis=0)
    metrics_df["percentage"] = (first_non_nan_index / df.shape[0]).fillna(1)

    # Initialize demand profile as 'smooth'
    metrics_df['PROFILE'] = 'smooth'

    # Compute the differences for each series
    diffs = df.diff().iloc[1:]  # Exclude the first row with NaN
    zero_diffs = (diffs == 0).sum()
    total_diffs = df.shape[0] - 1  # Number of differences per series
    zero_diff_proportions = zero_diffs / total_diffs
    metrics_df['zero_diff_proportion'] = zero_diff_proportions

    # Apply conditions to classify the demand profiles
    metrics_df.loc[(metrics_df['adi'] >= adi_threshold) & (metrics_df['cv_squared'] < cvar_threshold), 'PROFILE'] = 'intermittent'
    metrics_df.loc[(metrics_df['adi'] < adi_threshold) & (metrics_df['cv_squared'] >= cvar_threshold), 'PROFILE'] = 'erratic'
    metrics_df.loc[(metrics_df['adi'] >= adi_threshold) & (metrics_df['cv_squared'] >= cvar_threshold), 'PROFILE'] = 'lumpy'
    metrics_df.loc[metrics_df['zero_diff_proportion'] >= flat_threshold, 'PROFILE'] = 'flat'
    metrics_df.loc[metrics_df['percentage'] >= new_product_threshold, 'PROFILE'] = 'new_product'

    # Reset index to get 'SERIES' column
    intermittence_df = metrics_df[['PROFILE']].reset_index().rename(columns={'index': 'SERIES'})

    return intermittence_df

# burst, stationary, seasonality
