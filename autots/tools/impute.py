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
    
def biased_ffill(df):
    df_mean = fill_mean(df)
    df_ffill = fill_forward(df)
    df = (df_mean + df_ffill)/2
    return df

def fill_na(df, method: str = 'ffill'):
    """
    Fill NA values using different methods
    
    args:
        method:
            'ffill' - fill most recent non-na value forward until another non-na value is reached
            'zero' - fill with zero. Useful for sales and other data where NA does usually mean $0.
            'mean' - fill all missing values with the series' overall average value
            'median' - fill all missing values with the series' overall median value
            'rolling mean' - fill with last n values (10)
            'ffill mean biased' - simple avg of ffill and mean
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
        df = rolling_mean(df)
        
    if method == 'ffill mean biased':
        df = biased_ffill(df)
    
    return df
    
