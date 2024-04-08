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
