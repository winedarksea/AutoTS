"""
Point to Probabilistic
"""
import pandas as pd
import numpy as np
from autots.tools.impute import fake_date_fill
from autots.tools.percentile import nan_quantile

try:
    from scipy.stats import percentileofscore
except Exception:
    pass


def percentileofscore_appliable(x, a, kind='rank'):
    return percentileofscore(a, score=x, kind=kind)


def historic_quantile(df_train, prediction_interval: float = 0.9, nan_flag=None):
    """
    Computes the difference between the median and the prediction interval range in historic data.

    Args:
        df_train (pd.DataFrame): a dataframe of training data
        prediction_interval (float): the desired forecast interval range

    Returns:
        lower, upper (np.array): two 1D arrays
    """
    quantiles = [0, 1 - prediction_interval, 0.5, prediction_interval, 1]
    # save compute time by using the non-nan verison if possible
    if not isinstance(nan_flag, bool):
        if isinstance(df_train, pd.DataFrame):
            nan_flag = np.isnan(np.min(df_train.to_numpy()))
        else:
            nan_flag = np.isnan(np.min(np.array(df_train)))
    if nan_flag:
        bins = nan_quantile(df_train.astype(float), quantiles, axis=0)
    else:
        bins = np.quantile(df_train.astype(float), quantiles, axis=0, keepdims=False)
    upper = bins[3] - bins[2]
    if 0 in upper:
        np.where(upper != 0, upper, (bins[4] - bins[2]) / 4)
    lower = bins[2] - bins[1]
    if 0 in lower:
        np.where(lower != 0, lower, (bins[2] - bins[0]) / 4)
    return lower, upper


def inferred_normal(train, forecast, n: int = 5, prediction_interval: float = 0.9):
    """A corruption of Bayes theorem.
    It will be sensitive to the transformations of the data."""
    prior_mu = train.mean().values
    prior_sigma = train.std().values
    idx = forecast.index
    columns = forecast.columns
    from scipy.stats import norm

    p_int = 1 - ((1 - prediction_interval) / 2)
    adj = norm.ppf(p_int)
    upper_forecast, lower_forecast = [], []

    for row in forecast.values:
        data_mu = row
        reshape_row = data_mu.reshape(1, -1)
        post_mu = (
            (prior_mu / prior_sigma**2) + ((n * data_mu) / prior_sigma**2)
        ) / ((1 / prior_sigma**2) + (n / prior_sigma**2))
        lower = pd.DataFrame(post_mu - adj * prior_sigma, index=columns).transpose()
        lower = lower.where(lower <= data_mu, reshape_row, axis=1)
        upper = pd.DataFrame(post_mu + adj * prior_sigma, index=columns).transpose()
        upper = upper.where(upper >= data_mu, reshape_row, axis=1)
        lower_forecast.append(lower)
        upper_forecast.append(upper)
    lower_forecast = pd.concat(lower_forecast, axis=0)
    upper_forecast = pd.concat(upper_forecast, axis=0)
    lower_forecast.index = idx
    upper_forecast.index = idx
    return upper_forecast, lower_forecast


"""
post_mu = ((prior_mu/prior_sigma ** 2) + ((n * data_mu)/data_sigma ** 2))/
      ((1/prior_sigma ** 2) + (n/data_sigma ** 2))
post_sigma = sqrt(1/((1/prior_sigma ** 2) + (n/data_sigma ** 2)))
"""


def Variable_Point_to_Probability(train, forecast, alpha=0.3, beta=1):
    """Data driven placeholder for model error estimation.

    ErrorRange = beta * (En + alpha * En-1 [cum sum of En])
    En = abs(0.5 - QTP) * D
    D = abs(Xn - ((Avg % Change of Train * Xn-1) + Xn-1))
    Xn = Forecast Value
    QTP = Percentile of Score in All Percent Changes of Train
    Score = Percent Change (from Xn-1 to Xn)

    Args:
        train (pandas.DataFrame): DataFrame of time series where index is DatetimeIndex
        forecast (pandas.DataFrame): DataFrame of forecast time series
            in which the index is a DatetimeIndex and columns/series aligned with train.
            Forecast must be > 1 in length.
        alpha (float): parameter which effects the broadening of error range over time
            Usually 0 < alpha < 1 (although it can be larger than 1)
        beta (float): parameter which effects the general width of the error bar
            Usually 0 < beta < 1 (although it can be larger than 1)

    Returns:
        ErrorRange (pandas.DataFrame): error width for each value of forecast.
    """
    column_order = train.columns.intersection(forecast.columns)
    intial_length = len(forecast.columns)
    forecast = forecast[column_order]  # align columns
    aligned_length = len(forecast.columns)
    train = train[column_order]
    if aligned_length != intial_length:
        print("Forecast columns do not match train, some series may be lost")

    train = train.replace(0, np.nan)

    train = fake_date_fill(train, back_method='keepNA')

    percent_changes = train.pct_change()

    median_change = percent_changes.median()
    # median_change = (1  + median_change)
    # median_change[median_change <= 0 ] = 0.01  # HANDLE GOING BELOW ZERO

    diffs = abs(
        forecast - (forecast + forecast * median_change).fillna(method='ffill').shift(1)
    )

    forecast_percent_changes = forecast.replace(0, np.nan).pct_change()

    quantile_differences = pd.DataFrame()
    for column in forecast.columns:
        percentile_distribution = percent_changes[column].dropna()

        quantile_difference = abs(
            (
                50
                - forecast_percent_changes[column].apply(
                    percentileofscore_appliable, a=percentile_distribution, kind='rank'
                )
            )
            / 100
        )
        quantile_differences = pd.concat(
            [quantile_differences, quantile_difference], axis=1
        )

    En = quantile_differences * diffs
    Enneg1 = En.cumsum().shift(1).fillna(0)
    ErrorRange = beta * (En + alpha * Enneg1)
    ErrorRange = ErrorRange.fillna(method='bfill').fillna(method='ffill')

    return ErrorRange


def Point_to_Probability(
    train, forecast, prediction_interval=0.9, method: str = 'historic_quantile'
):
    """Data driven placeholder for model error estimation.

    Catlin Point to Probability method ('a mixture of dark magic and gum disease')

    Args:
        train (pandas.DataFrame): DataFrame of time series where index is DatetimeIndex
        forecast (pandas.DataFrame): DataFrame of forecast time series
            in which the index is a DatetimeIndex and columns/series aligned with train.
            Forecast must be > 1 in length.
        prediction_interval (float): confidence or perhaps credible interval
        method (str): spell to cast to create dark magic.
            'historic_quantile', 'inferred_normal', 'variable_pct_change'
            gum disease available separately upon request.

    Returns:
        upper_error, lower_error (two pandas.DataFrames for upper and lower bound respectively)
    """
    if method == 'historic_quantile':
        lower, upper = historic_quantile(train, prediction_interval)
        upper_forecast = forecast.astype(float) + upper
        lower_forecast = forecast.astype(float) - lower
        return upper_forecast, lower_forecast
    if method == 'inferred_normal':
        return inferred_normal(
            train, forecast, n=5, prediction_interval=prediction_interval
        )
    if method == 'variable_pct_change':
        beta = np.exp(prediction_interval * 10)
        alpha = 0.3
        errorranges = Variable_Point_to_Probability(
            train, forecast, alpha=alpha, beta=beta
        )
        # make symmetric error ranges
        errorranges = errorranges / 2

        upper_forecast = forecast + errorranges
        lower_forecast = forecast - errorranges
        return upper_forecast, lower_forecast
