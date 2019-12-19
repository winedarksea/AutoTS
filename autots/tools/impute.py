"""
Fill NA
"""
import numpy as np
import pandas as pd

def fill_zero(df):
    df = df.fillna(0)
    return df

def fill_forward(df):
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df
    
def fill_mean(df):
    df = df.fillna(df.mean().to_dict())
    return df

def fill_median(df):
    df = df.fillna(df.median().to_dict())
    return df
    
def rolling_mean(df, window: int = 10):
    df= df.fillna(df.rolling(window = window, min_periods=1).mean())
    return df
    
def biased_ffill(df, mean_weight: float = 1):
    df_mean = fill_mean(df)
    df_ffill = fill_forward(df)
    df = ((df_mean * mean_weight) + df_ffill)/(1 + mean_weight)
    return df

def fake_date_fill(df, back_method: str = 'slice'):
    """
    Returns a dataframe where na values are removed and values shifted forward.
    Thus, values will likely have incorrect timestamps!
    
    :param back_method: - how to deal with tails due to different length series
        - 'bfill' -back fill the last value
        - 'slice' - drop any rows with any na
        - 'keepNA' - keep the lagging na
    :type back_method: str
    """    
    df_index = df.index.to_series().copy()
    df = df.sort_index(ascending=False)
    df = df.apply(lambda x: pd.Series(x.dropna().values))
    df = df.sort_index(ascending=False)
    df.index = df_index.tail(len(df.index))    
    df = df.dropna(how = 'all', axis = 0)
    
    if back_method == 'bfill':
        df = df.fillna(method = 'bfill')
    if back_method == 'slice':
        df = df.dropna(how = 'any', axis = 0)
    if back_method == 'keepNA':
        pass
    return df

def fill_na(df, method: str = 'ffill', window: int = 10):
    """
    Fill NA values using different methods
    
    args:
    ======
    
        :param method:
            'ffill' - fill most recent non-na value forward until another non-na value is reached
            'zero' - fill with zero. Useful for sales and other data where NA does usually mean $0.
            'mean' - fill all missing values with the series' overall average value
            'median' - fill all missing values with the series' overall median value
            'rolling mean' - fill with last n (window) values
            'ffill mean biased' - simple avg of ffill and mean
            'fake date' - shifts forward data over nan, thus values will have incorrect timestamps
        :type method: str
        
        :param window: length of rolling windows for filling na, for rolling methods
        :type window: int
    """
    if method == 'zero':
        df = fill_zero(df)
    
    if method == 'ffill':
        df = fill_forward(df)
        
    if method == 'mean':
        df = fill_mean(df)
        
    if method == 'median':
        df = fill_median(df)
        
    if method == 'rolling mean':
        df = rolling_mean(df, window = window)
        
    if method == 'ffill mean biased':
        df = biased_ffill(df)
    
    if method == 'fake date':
        df = fake_date_fill(df, back_method = 'slice')
    
    return df
    
