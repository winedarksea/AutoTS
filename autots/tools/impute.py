"""Fill NA."""
import numpy as np
import pandas as pd


def fill_zero(df):
    """Fill NaN with zero."""
    df = df.fillna(0)
    return df


def fill_forward(df):
    """Fill NaN with previous values."""
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df


def fill_mean(df):
    """Fill NaN with mean."""
    df = df.fillna(df.mean().to_dict())
    return df


def fill_median(df):
    """Fill NaN with median."""
    df = df.fillna(df.median().to_dict())
    return df


def rolling_mean(df, window: int = 10):
    """Fill NaN with mean of last window values."""
    df = df.fillna(df.rolling(
        window=window, min_periods=1).mean()).fillna(df.mean().to_dict())
    return df


def biased_ffill(df, mean_weight: float = 1):
    """Fill NaN with average of last value and mean."""
    df_mean = fill_mean(df)
    df_ffill = fill_forward(df)
    df = ((df_mean * mean_weight) + df_ffill)/(1 + mean_weight)
    return df


def fake_date_fill(df, back_method: str = 'slice'):
    """
    Return a dataframe where na values are removed and values shifted forward.

    Warnings:
        Thus, values will likely have incorrect timestamps!

    Args:
        back_method (str): how to deal with tails left by shifting NaN
            - 'bfill' -back fill the last value
            - 'slice' - drop any rows with any na
            - 'keepNA' - keep the lagging na
    """
    df_index = df.index.to_series().copy()
    df = df.sort_index(ascending=False)
    df = df.apply(lambda x: pd.Series(x.dropna().values))
    df = df.sort_index(ascending=False)
    df.index = df_index.tail(len(df.index))
    df = df.dropna(how='all', axis=0)

    if back_method == 'bfill':
        df = df.fillna(method='bfill')
        return df
    elif back_method == 'slice':
        df = df.dropna(how='any', axis=0)
        return df
    elif back_method == 'keepNA':
        return df
    else:
        print('back_method not recognized in fake_date_fill')
        return df


def FillNA(df, method: str = 'ffill', window: int = 10):
    """Fill NA values using different methods.

    Args:
        method (str):
            'ffill' - fill most recent non-na value forward until another non-na value is reached
            'zero' - fill with zero. Useful for sales and other data where NA does usually mean $0.
            'mean' - fill all missing values with the series' overall average value
            'median' - fill all missing values with the series' overall median value
            'rolling mean' - fill with last n (window) values
            'ffill mean biased' - simple avg of ffill and mean
            'fake date' - shifts forward data over nan, thus values will have incorrect timestamps
        window (int): length of rolling windows for filling na, for rolling methods
    """
    if method == 'zero':
        df = fill_zero(df)
        return df

    if method == 'ffill':
        df = fill_forward(df)
        return df

    if method == 'mean':
        df = fill_mean(df)
        return df

    if method == 'median':
        df = fill_median(df)
        return df

    if method == 'rolling mean':
        df = rolling_mean(df, window=window)
        return df

    if method == 'ffill mean biased':
        df = biased_ffill(df)
        return df

    if method == 'fake date':
        df = fake_date_fill(df, back_method='slice')
        return df

    if method is None:
        return df

    else:
        print("FillNA method not known, returning original")
        return df
