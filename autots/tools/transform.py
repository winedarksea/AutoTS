import numpy as np
import pandas as pd

def remove_outliers(df, std_threshold: float = 3):
    """
    Replace outliers with np.nan
    :param df: DataFrame containing numeric data
    :type df: pandas.DataFrame
    
    :param std_threshold: The number of standard deviations away from mean to count as outlier.
    :type std_threshold: float
    
    https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
    """
    df = df[np.abs(df - df.mean()) <= (std_threshold * df.std())]
    return df

def clip_outliers(df, std_threshold: float = 3):
    """
    Replace outliers above threshold with that threshold.
    
    :param df: DataFrame containing numeric data
    :type df: pandas.DataFrame
    
    :param std_threshold: The number of standard deviations away from mean to count as outlier.
    :type std_threshold: float
    """
    df_std = df.std(axis = 0, skipna = True)
    df_mean = df.mean(axis = 0, skipna = True)
    
    lower = df_mean - (df_std * std_threshold)
    upper = df_mean + (df_std * std_threshold)
    df2 = df.clip(lower =  lower, upper = upper, axis = 1)
    return df2
def context_slicer(method: str = 'max', fraction: float = 2, forecast_length: int = 30):
    """
    Crop the training dataset to reduce the context given. Keeps newest.
    Usually faster and occasionally more accurate.
    
    args
    ======
    
        :param method: - amount of data to retain
            'max' - all data, none removed
            'fraction max' - a fraction (< 1) of the total data based on total length.
            'fraction forecast' - fraction (usually > 1) based on forecast length
        :type method: str
        
        :param fraction:
        :type fraction: float
        
        :param forecast_length: - forecast length, if 'fraction forecast' used
        :type
    """
    

from autots.tools.impute import fill_na

class GeneralTransformer(object):
    """
    Results of values_to_numeric
        categorical_features - list of columns (series_ids) which were encoded
    """
    def __init__(self, outlier = None, fillNA = [], transformation = False,  = None):
        self.dataframe = dataframe
        self.categorical_features = categorical_features
        self.categorical_transformed = categorical_transformed
        self.encoder = encoder

df = pd.DataFrame({'Data':np.random.normal(size=200),
                   'Data1':np.random.normal(size=200),
                   'Data2':np.random.normal(size=200)})

df2 = clip_outliers(df, std_threshold = 1)
df3 = pd.concat([df, df2], axis = 1)