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

def context_slicer(df, method: str = 'max', fraction: float = 2, forecast_length: int = 30):
    """
    Crop the training dataset to reduce the context given. Keeps newest.
    Usually faster and occasionally more accurate.
    
    args
    ======
    
        :param df: DataFrame with a datetime index of training data sample
        :type df: pandas.DataFrame
        
        :param method: - amount of data to retain
            'max' - all data, none removed
            'fraction max' - a fraction (< 1) of the total data based on total length.
            'fraction forecast' - fraction (usually > 1) based on forecast length
        :type method: str
        
        :param fraction: - percent of data to retain, either of total length or forecast length
        :type fraction: float
        
        :param forecast_length: - forecast length, if 'fraction forecast' used
        :type forecast_length: int
    """
    if method == 'max':
        return df
    
    df = df.sort_index(ascending=True)
    
    if method == 'fraction max':
        assert fraction > 0, "fraction must be > 0 with 'fraction max'"
        
        df = df.tail(int(len(df.index) * fraction))
        return df
    
    if method == 'fraction forecast':
        assert forecast_length > 0, "forecast_length must be greater than 0"
        assert fraction > 0, "fraction must be > 0 with 'fraction forecast'"
        
        df = df.tail(int(forecast_length * fraction))
        return df
    
    else:
        return df


class Detrend(object):
    """Remove a linear trend from the data
    
    """
    def fit(self, df):
        """Fits trend for later detrending
        Args:
            df (pandas.DataFrame): input dataframe
        """
        from statsmodels.regression.linear_model import GLS
        try:
            df = df.astype(float)
        except:
            raise ValueError ("Data Cannot Be Converted to Numeric Float")
            
        self.model = GLS(df.values, (df.index.astype( int ).values), missing = 'drop').fit()
        self.shape = df.shape
        return self        
        
    def fit_transform(self, df):
        """Fits and Returns Detrended DataFrame
        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)
        

    def transform(self, df):
        """Returns detrended data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except:
            raise ValueError ("Data Cannot Be Converted to Numeric Float")
        df = df.astype(float) - self.model.predict(df.index.astype( int ).values)
        return df
    
    def inverse_transform(self, df):
        """Returns data to original form
        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except:
            raise ValueError ("Data Cannot Be Converted to Numeric Float")
        df = df.astype(float) + self.model.predict(df.index.astype( int ).values)
        return df
        
class RollingMeanTransformer(object):
    """Attempt at Rolling Mean with built-in inverse_transform for time series
    inverse_transform can only be applied to the original series, or an immediately following forecast
    Does not play well with data with NaNs
    
    Args:
        window (int): number of periods to roll over
    """
    def __init__(self, window: int = 10):
        self.window = window
        
    def fit(self, df):
        """Fits
        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.shape = df.shape
        self.last_values = df.tail(self.window)
        self.first_values = df.head(self.window)
        return self        
    def transform(self, df):
        """Returns rolling data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = df.rolling(window = self.window, min_periods  = 1).mean()
        self.last_rolling = df.tail(1)
        return df
    def fit_transform(self, df):
        """Fits and Returns Magical DataFrame
        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Returns data to original *or* forecast form
        Args:
            df (pandas.DataFrame): input dataframe
            trans_method (str): whether to inverse on original data, or on a following sequence
                - 'original' return original data to original numbers 
                - 'forecast' inverse the transform on a dataset immediately following the original
        """
        window = self.window
        if trans_method == 'original':
            staged = self.first_values
            diffed = ((df.astype(float) - df.shift(1).astype(float)) * window).tail(len(df.index) - window)
            temp_cols = diffed.columns
            for n in range(len(diffed.index)):
                temp_index = diffed.index[n]
                temp_row = diffed.iloc[n].reset_index(drop=True) + staged.iloc[n].reset_index(drop=True).astype(float)
                temp_row = pd.DataFrame(temp_row.values.reshape(1,len(temp_row)), columns = temp_cols)
                temp_row.index = pd.DatetimeIndex([temp_index])
                staged = pd.concat([staged, temp_row], axis = 0)
            return staged

        if trans_method == 'forecast':
            staged = self.last_values
            df = pd.concat([self.last_rolling, df], axis = 0)
            diffed = ((df.astype(float) - df.shift(1).astype(float)) * window).tail(len(df.index))
            temp_cols = diffed.columns
            for n in range(len(diffed.index)):
                temp_index = diffed.index[n]
                temp_row = diffed.iloc[n].reset_index(drop=True) + staged.iloc[n].reset_index(drop=True).astype(float)
                temp_row = pd.DataFrame(temp_row.values.reshape(1,len(temp_row)), columns = temp_cols)
                temp_row.index = pd.DatetimeIndex([temp_index])
                staged = pd.concat([staged, temp_row], axis = 0)
            staged = staged.tail(len(df.index))
            return staged
            
    

from autots.tools.impute import fill_na

class GeneralTransformer(object):
    """
    Transform data.
    Remove outliers, fillNA, then transform.
    
    Args:       
        outlier (str): - level of outlier removal, if any, per series
            'None'
            'clip2std' - replace values > 2 stdev with the value of 2 st dev
            'clip3std' - replace values > 3 stdev with the value of 2 st dev
            'clip4std' - replace values > 4 stdev with the value of 2 st dev
            'remove3std' - replace values > 3 stdev with NaN

        fillNA (str): - method to fill NA
            'ffill' - fill most recent non-na value forward until another non-na value is reached
            'zero' - fill with zero. Useful for sales and other data where NA does usually mean $0.
            'mean' - fill all missing values with the series' overall average value
            'median' - fill all missing values with the series' overall median value
            'rolling mean' - fill with last n (window) values
            'ffill mean biased' - simple avg of ffill and mean
            'fake date' - shifts forward data over nan, thus values will have incorrect timestamps
        
        transformation (str): - transformation to apply
            'MinMaxScaler'
            'PowerTransformer'
            'detrend'
            
            'PctChange'
            'RollingMean'

    """
    def __init__(self, outlier = None, fillNA = [], transformation = False):
        self.dataframe = dataframe
        self.categorical_features = categorical_features
        self.categorical_transformed = categorical_transformed
        self.encoder = encoder
    
    def _check_is_fitted(self, X):
        """Check the inputs before transforming"""
        check_is_fitted(self)
        # check that the dimension of X are adequate with the fitted data
        if X.shape[1] != self.quantiles_.shape[1]:
            raise ValueError('X does not have the same number of features as'
                             ' the previously fitted data. Got {} instead of'
                             ' {}.'.format(X.shape[1],
                                           self.quantiles_.shape[1]))

    def fit(self, X, y=None):
        """Estimate the optimal parameter lambda for each feature.
        The optimal lambda parameter for minimizing skewness is estimated on
        each feature independently using maximum likelihood.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to estimate the optimal transformation parameters.
        y : Ignored
        Returns
        -------
        self : object
        """
        self._fit(X, y=y, force_transform=False)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X, y, force_transform=True)


    def transform(self, X):

        return X
    def inverse_transform(self, X):
        return X


