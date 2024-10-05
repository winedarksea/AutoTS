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
        'adi': adi,
        'autocorr_1': np.diag(np.corrcoef(df.T, df.shift(1).bfill().T)[:df.shape[1], df.shape[1]:]),
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
"""
from autots.tools.seasonal import date_part

x_s = date_part(df.index, method="common_fourier")
x_t = create_changepoint_features(df.index, changepoint_spacing=60, changepoint_distance_end=120)
X = pd.concat([x_s, x_t], axis=1)
X["constant"] = 1

# Convert X and df (Y) to NumPy arrays for linear regression
X_values = X.values
Y_values = df.values

# Perform linear regression using the normal equation: (X.T @ X)^(-1) @ X.T @ Y
beta = np.linalg.inv(X_values.T @ X_values) @ X_values.T @ Y_values

# Calculate predicted values for Y
Y_pred = X_values @ beta

# Calculate the contribution of each feature group
contribution_seasonality = X[x_s.columns].values @ beta[:len(x_s.columns)]
contribution_changepoints = X[x_t.columns].values @ beta[len(x_s.columns):-1]
contribution_constant = X["constant"].values.reshape(-1, 1) @ beta[-1:]

# Total contribution (sum of absolute contributions for each time step)
total_contribution = np.abs(contribution_seasonality) + np.abs(contribution_changepoints) + np.abs(contribution_constant)

# Normalize each contribution by the total contribution
contrib_seasonality_pct = np.abs(contribution_seasonality) / total_contribution
contrib_changepoints_pct = np.abs(contribution_changepoints) / total_contribution
contrib_constant_pct = np.abs(contribution_constant) / total_contribution

# Calculate the average percentage contribution for each group
avg_contrib_seasonality = np.mean(contrib_seasonality_pct, axis=0)
avg_contrib_changepoints = np.mean(contrib_changepoints_pct, axis=0)
avg_contrib_constant = np.mean(contrib_constant_pct, axis=0)

# Create a DataFrame to summarize the percentage contributions
feature_contributions = pd.DataFrame({
    "seasonality_contribution": avg_contrib_seasonality,
    "changepoint_contribution": avg_contrib_changepoints,
    "constant_contribution": avg_contrib_constant
}, index=df.columns)
feature_contributions['largest_contributor'] = feature_contributions[[
    'seasonality_contribution', 
    'changepoint_contribution',
    'constant_contribution'
]].idxmax(axis=1)
feature_contributions["season_trend_percent"] = feature_contributions["seasonality_contribution"] / (feature_contributions["changepoint_contribution"] + feature_contributions["seasonality_contribution"])

autocorr_lag1 = np.diag(np.corrcoef(df.T, df.shift(1).bfill().T)[:df.shape[1], df.shape[1]:])


window_size = 20  # Define the rolling window size

# Calculate rolling means for the beginning and end of the series
rolling_mean_start = df.rolling(window=window_size).mean().iloc[:window_size]
rolling_mean_end = df.rolling(window=window_size).mean().iloc[-window_size:]

# Compute the absolute difference between start and end rolling means
rolling_diff = np.abs(rolling_mean_start.mean() - rolling_mean_end.mean())

# Define threshold for stationarity (e.g., small difference means stationary)
threshold = 0.01
stationarity_labels = np.where(rolling_diff < threshold, 'Stationary', 'Non-Stationary')




from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Create the design matrix for linear regression (time index as X)
X = np.arange(len(df)).reshape(-1, 1)

# Perform linear regression in a fully vectorized way using NumPy
# Here we're taking the transpose to fit across all columns (time series) simultaneously
reg = LinearRegression(fit_intercept=True)
reg.fit(X, scaler.fit_transform(df.rolling(3).mean().bfill().values))

# Get the slopes (coefficients for each time series)
slopes = reg.coef_

# Define threshold for slope to indicate non-stationarity
slope_threshold = 0.01
stationarity_labels = np.where(np.abs(slopes) < slope_threshold, 'Stationary', 'Non-Stationary')

# Create a DataFrame with stationarity labels
stationarity_df = pd.DataFrame(stationarity_labels, index=df.columns, columns=['Stationarity'])

print(stationarity_df)
"""
