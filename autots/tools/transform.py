import numpy as np
import pandas as pd

def remove_outliers(df, std_threshold: float = 3):
    """Replace outliers with np.nan
    https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame

    Args:
        df (pandas.DataFrame): DataFrame containing numeric data, DatetimeIndex
        std_threshold (float): The number of standard deviations away from mean to count as outlier.
    
    """
    df = df[np.abs(df - df.mean()) <= (std_threshold * df.std())]
    return df

def clip_outliers(df, std_threshold: float = 3):
    """Replace outliers above threshold with that threshold. Axis = 0
    
    Args:
        df (pandas.DataFrame): DataFrame containing numeric data
        std_threshold (float): The number of standard deviations away from mean to count as outlier.
    """
    df_std = df.std(axis = 0, skipna = True)
    df_mean = df.mean(axis = 0, skipna = True)
    
    lower = df_mean - (df_std * std_threshold)
    upper = df_mean + (df_std * std_threshold)
    df2 = df.clip(lower =  lower, upper = upper, axis = 1)
    
    return df2

def context_slicer(df, method: str = 'max', fraction: float = 2, forecast_length: int = 30):
    """Crop the training dataset to reduce the context given. Keeps newest.
    Usually faster and occasionally more accurate.
    
    Args:
        df (pandas.DataFrame): DataFrame with a datetime index of training data sample
        
        method (str): - amount of data to retain
            'max' - all data, none removed
            'fraction max' - a fraction (< 1) of the total data based on total length.
            'fraction forecast' - fraction (usually > 1) based on forecast length
        
        fraction (float): - percent of data to retain, either of total length or forecast length
        
        forecast_length (int): - forecast length, if 'fraction forecast' used
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

def simple_context_slicer(df, method: str = 'None', forecast_length: int = 30):
    """Condensed version of context_slicer with more limited options.
    
    Args:
        df (pandas.DataFrame): training data frame to slice
        method (str): Option to slice dataframe
            'None' - return unaltered dataframe
            'HalfMax' - return half of dataframe
            '2ForecastLength' - return dataframe equal to twice length of forecast
    """
    if (method == 'None') or (method == None):
        return df
    
    df = df.sort_index(ascending=True)
    
    if method == 'HalfMax':
        return df.tail(int(len(df.index)/2))
    if method == '2ForecastLength':
        return df.tail(2 * forecast_length)
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
        window (int): number of periods to take mean over
    """
    def __init__(self, window: int = 10):
        self.window = window
        
    def fit(self, df):
        """Fits
        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.shape = df.shape
        self.last_values = df.tail(self.window).fillna(method = 'ffill').fillna(method = 'bfill')
        self.first_values = df.head(self.window).fillna(method = 'ffill').fillna(method = 'bfill')
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
            diffed = diffed.tail(len(diffed.index) -1)
            temp_cols = diffed.columns
            for n in range(len(diffed.index)):
                temp_index = diffed.index[n]
                temp_row = diffed.iloc[n].reset_index(drop=True) + staged.iloc[n].reset_index(drop=True).astype(float)
                temp_row = pd.DataFrame(temp_row.values.reshape(1,len(temp_row)), columns = temp_cols)
                temp_row.index = pd.DatetimeIndex([temp_index])
                staged = pd.concat([staged, temp_row], axis = 0)
            staged = staged.tail(len(diffed.index))
            return staged
            
class EmptyTransformer(object):
    def fit(self, df):
        return self
    def transform(self, df):
        return df
    def inverse_transform(self, df):
        return df
    def fit_transform(self, df):
        return df

from autots.tools.impute import FillNA

class GeneralTransformer(object):
    """Remove outliers, fillNA, then mathematical transformations.
    
    Args:       
        outlier (str): - level of outlier removal, if any, per series
            'None'
            'clip2std' - replace values > 2 stdev with the value of 2 st dev
            'clip3std' - replace values > 3 stdev with the value of 2 st dev
            'clip4std' - replace values > 4 stdev with the value of 2 st dev
            'remove3std' - replace values > 3 stdev with NaN

        fillNA (str): - method to fill NA, passed through to FillNA()
            'ffill' - fill most recent non-na value forward until another non-na value is reached
            'zero' - fill with zero. Useful for sales and other data where NA does usually mean $0.
            'mean' - fill all missing values with the series' overall average value
            'median' - fill all missing values with the series' overall median value
            'rolling mean' - fill with last n (window = 10) values
            'ffill mean biased' - simple avg of ffill and mean
            'fake date' - shifts forward data over nan, thus values will have incorrect timestamps
        
        transformation (str): - transformation to apply
            'None'
            'MinMaxScaler' - Sklearn MinMaxScaler
            'PowerTransformer' - Sklearn PowerTransformer
            'Detrend' - fit then remove a linear regression from the data
            'RollingMean10' - 10 period rolling average (smoothing)
            'RollingMean100thN' - Rolling mean of periods of len(train)/100 (minimum 2)

    """
    def __init__(self, outlier: str = "None", fillNA: str = 'ffill', transformation: str = 'None'):
        self.outlier = outlier
        self.fillNA = fillNA
        self.transformation = transformation

    def outlier_treatment(self, df):
        """
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        outlier = self.outlier
        
        if (outlier =='None') or (outlier == None):
            return df
        
        if (outlier == 'clip2std'):
            df = clip_outliers(df, std_threshold = 2)
            return df
        if (outlier == 'clip3std'):
            df = clip_outliers(df, std_threshold = 3)
            return df
        if (outlier == 'clip4std'):
            df = clip_outliers(df, std_threshold = 4)
            return df
        if (outlier == 'remove3std'):
            df = remove_outliers(df, std_threshold = 3)
            return df
        else:
            return df
    
    def fill_na(self, df, window: int = 10):
        """
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = FillNA(df, method = self.fillNA, window = window)
        return df
        
    def _transformation_fit(self, df):
        """
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        transformation = self.transformation
        if (transformation =='None') or (transformation == None):
            transformer = EmptyTransformer.fit(df)
            return transformer
        
        if (transformation =='MinMaxScaler'):
            from sklearn.preprocessing import MinMaxScaler
            transformer = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df)
            #df = transformer.transform(df)
            return transformer
        
        if (transformation =='PowerTransformer'):
            from sklearn.preprocessing import PowerTransformer
            transformer = PowerTransformer(method = 'yeo-johnson', standardize=True, copy=True).fit(df)
            #df = transformer.transform(df)
            return transformer
        
        if (transformation =='Detrend'):
            transformer = Detrend().fit(df)
            #df = transformer.transform(df)
            return transformer
        
        if (transformation == 'RollingMean10'):
            self.window = 10
            transformer = RollingMeanTransformer(window = self.window).fit(df)
            #df = transformer.transform(df)
            return transformer
        
        if (transformation == 'RollingMean100thN'):
            window = int(len(df.index)/100)
            window = 2 if window < 2 else window
            self.window = window
            transformer = RollingMeanTransformer(window = self.window).fit(df)
            #df = transformer.transform(df)
            return transformer
        else:
            print("Transformation method not known or improperly entered, returning untransformed df")
            transformer = EmptyTransformer.fit(df)
            return transformer
        
    def _fit(self, df):
        df = df.copy()
        df = self.outlier_treatment(df)
        df = self.fill_na(df)
        self.column_names = df.columns
        self.index = df.index
        self.transformer = self._transformation_fit(df)
        df = self.transformer.transform(df)
        df = pd.DataFrame(df, index = self.index, columns = self.column_names)
        return df
    
    def fit(self, df):
        """Apply transformations and return transformer object
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        self._fit(df)
        return self

    def fit_transform(self, df):
        return self._fit(df)

    def transform(self, df):
        df = df.copy()
        df = self.outlier_treatment(df)
        df = self.fill_na(df)
        self.column_names = df.columns
        self.index = df.index
        df = self.transformer.transform(df)
        df = pd.DataFrame(df, index = self.index, columns = self.column_names)
        return df
    
    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Apply transformations and return transformer object
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
            trans_method (str): passed through to RollingTransformer, if used
        """
        if (self.transformation == 'RollingMean100thN') or (self.transformation == 'RollingMean10'):
            df = self.transformer.inverse_transform(df, trans_method = trans_method)
        else:
            df = self.transformer.inverse_transform(df)
        
        return df


def RandomTransform():
    """
    Returns a dict of randomly choosen transformation selections
    """
    outlier_choice = np.random.choice(a = [None, 'clip3std', 'clip2std','clip4std','remove3std'], size = 1, p = [0.4, 0.3, 0.1, 0.1, 0.1]).item()
    na_choice = np.random.choice(a = ['ffill', 'fake date', 'rolling mean','mean','zero', 'ffill mean biased', 'median'], size = 1, p = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]).item()
    transformation_choice = np.random.choice(a = [None, 'PowerTransformer', 'RollingMean100thN','MinMaxScaler','Detrend', 'RollingMean10'], size = 1, p = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]).item()
    context_choice = np.random.choice(a = [None, 'HalfMax', '2ForecastLength'], size = 1, p = [0.8, 0.1, 0.1]).item()
    param_dict = {
            'outlier': outlier_choice,
            'fillNA' : na_choice, 
           'transformation' : transformation_choice,
           'context_slicer' : context_choice
            }
    return param_dict