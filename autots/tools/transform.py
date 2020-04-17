"""Preprocessing data methods."""
import numpy as np
import pandas as pd
from autots.tools.impute import FillNA
# pylint disable=W293,E251


def remove_outliers(df, std_threshold: float = 3):
    """Replace outliers with np.nan.
    https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame

    Args:
        df (pandas.DataFrame): DataFrame containing numeric data, DatetimeIndex
        std_threshold (float): The number of standard deviations away from mean to count as outlier.

    """

    df = df[np.abs(df - df.mean()) <= (std_threshold * df.std())]
    return df

def clip_outliers(df, std_threshold: float = 3):
    """Replace outliers above threshold with that threshold. Axis = 0.
    
    Args:
        df (pandas.DataFrame): DataFrame containing numeric data
        std_threshold (float): The number of standard deviations away from mean to count as outlier.
    """
    df_std = df.std(axis=0, skipna=True)
    df_mean = df.mean(axis=0, skipna=True)
    
    lower = df_mean - (df_std * std_threshold)
    upper = df_mean + (df_std * std_threshold)
    df2 = df.clip(lower=lower, upper=upper, axis=1)
    
    return df2

def simple_context_slicer(df, method: str = 'None', forecast_length: int = 30):
    """Condensed version of context_slicer with more limited options.

    Args:
        df (pandas.DataFrame): training data frame to slice
        method (str): Option to slice dataframe
            'None' - return unaltered dataframe
            'HalfMax' - return half of dataframe
            'ForecastLength' - return dataframe equal to length of forecast
            '2ForecastLength' - return dataframe equal to twice length of forecast
                (also takes 4, 6, 8, 10 in addition to 2)
    """
    if method in [None, "None"]:
        return df

    df = df.sort_index(ascending=True)

    if 'forecastlength' in str(method).lower():
        len_int = int([x for x in str(method) if x.isdigit()][0])
        return df.tail(len_int * forecast_length)
    elif method == 'HalfMax':
        return df.tail(int(len(df.index)/2))
    elif str(method).isdigit():
        return df.tail(int(method))
    else:
        print("Context Slicer Method not recognized")
        return df
    """
    if method == '2ForecastLength':
        return df.tail(2 * forecast_length)
    elif method == '6ForecastLength':
        return df.tail(6 * forecast_length)
    elif method == '12ForecastLength':
        return df.tail(12 * forecast_length)
    elif method == 'ForecastLength':
        return df.tail(forecast_length)
    elif method == '4ForecastLength':
        return df.tail(4 * forecast_length)
    elif method == '8ForecastLength':
        return df.tail(8 * forecast_length)
    elif method == '10ForecastLength':
        return df.tail(10 * forecast_length)
    """


class Detrend(object):
    """Remove a linear trend from the data."""

    def __init__(self):
        self.name = 'Detrend'

    def fit(self, df):
        """Fits trend for later detrending.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        from statsmodels.regression.linear_model import GLS
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")

        # formerly df.index.astype( int ).values
        y = df.values
        X = (pd.to_numeric(df.index,
                           errors='coerce', downcast='integer').values)
        # from statsmodels.tools import add_constant
        # X = add_constant(X, has_constant='add')
        self.model = GLS(y, X, missing='drop').fit()
        self.shape = df.shape
        return self

    def fit_transform(self, df):
        """Fit and Return Detrended DataFrame.
        
        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)        

    def transform(self, df):
        """Return detrended data.
        
        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")
        # formerly X = df.index.astype( int ).values
        X = (pd.to_numeric(df.index, errors = 'coerce',downcast='integer').values)
        # from statsmodels.tools import add_constant
        # X = add_constant(X, has_constant='add')
        df = df.astype(float) - self.model.predict(X)
        return df
    
    def inverse_transform(self, df):
        """Return data to original form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError ("Data Cannot Be Converted to Numeric Float")
        X = pd.to_numeric(df.index, errors = 'coerce',downcast='integer').values
        # from statsmodels.tools import add_constant
        # X = add_constant(X, has_constant='add')
        df = df.astype(float) + self.model.predict(X)
        return df

class SinTrend(object):
    """Modelling sin"""
    
    def __init__(self):
        self.name = 'SinTrend'
    def fit_sin(self, tt, yy):
        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
        
        from user unsym @ https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
        '''
        import scipy.optimize
        tt = np.array(tt)
        yy = np.array(yy)
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    
        def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=10000)
        A, w, p, c = popt
        # f = w/(2.*np.pi)
        # fitfunc = lambda t: A * np.sin(w*t + p) + c
        return {"amp": A, "omega": w, "phase": p, "offset": c} #, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

    def fit(self, df):
        """Fits trend for later detrending
        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError ("Data Cannot Be Converted to Numeric Float")
        
        X = (pd.to_numeric(df.index, errors = 'coerce',downcast='integer').values)
        self.sin_params = pd.DataFrame()
        # make this faster
        for column in df.columns:
            try:
                y = df[column].values
                vals = self.fit_sin(X, y)
                current_param = pd.DataFrame(vals, index = [column])
            except Exception as e:
                print(e)
                current_param = pd.DataFrame({"amp": 0, "omega": 1, "phase": 1, "offset": 1}, index = [column])
            self.sin_params = pd.concat([self.sin_params, current_param], axis = 0)
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
        except Exception:
            raise ValueError ("Data Cannot Be Converted to Numeric Float")
        X = (pd.to_numeric(df.index, errors = 'coerce',downcast='integer').values)
        
        sin_df = pd.DataFrame()
        # make this faster
        for index, row in self.sin_params.iterrows():
            yy = pd.DataFrame(row['amp']*np.sin(row['omega']*X + row['phase']) + row['offset'], columns = [index])
            sin_df = pd.concat([sin_df, yy], axis = 1)
        df_index = df.index
        df = df.astype(float).reset_index(drop = True) - sin_df.reset_index(drop = True)
        df.index = df_index
        return df
    
    def inverse_transform(self, df):
        """Returns data to original form
        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError ("Data Cannot Be Converted to Numeric Float")
        X = pd.to_numeric(df.index, errors = 'coerce',downcast='integer').values
        
        sin_df = pd.DataFrame()
        # make this faster
        for index, row in self.sin_params.iterrows():
            yy = pd.DataFrame(row['amp']*np.sin(row['omega']*X + row['phase']) + row['offset'], columns = [index])
            sin_df = pd.concat([sin_df, yy], axis = 1)
        df_index = df.index
        df = df.astype(float).reset_index(drop = True) + sin_df.reset_index(drop = True)
        df.index = df_index
        return df

class RollingMeanTransformer(object):
    """Attempt at Rolling Mean with built-in inverse_transform for time series
    inverse_transform can only be applied to the original series, or an immediately following forecast
    Does not play well with data with NaNs
    Inverse transformed values returned will also not return as 'exactly' equals due to floating point imprecision.
    
    Args:
        window (int): number of periods to take mean over
    """
    def __init__(self, window: int = 10, fixed: bool = False):
        self.window = window
        self.fixed = fixed
        
    def fit(self, df):
        """Fits.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.shape = df.shape
        self.last_values = df.tail(self.window).fillna(method = 'ffill').fillna(method = 'bfill')
        self.first_values = df.head(self.window).fillna(method = 'ffill').fillna(method = 'bfill')
        
        df = df.tail(self.window + 1).rolling(window = self.window, min_periods  = 1).mean()
        self.last_rolling = df.tail(1)
        return self        
    def transform(self, df):
        """Returns rolling data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = df.rolling(window = self.window, min_periods  = 1).mean()
        # self.last_rolling = df.tail(1)
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
        if self.fixed:
            return df
        else:
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
    
            # current_inversed = current * window - cumsum(window-1 to previous)
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
"""
df = df_wide_numeric.tail(60).head(50)
df_forecast = (df_wide_numeric).tail(10)
forecats = transformed.tail(10)
test = RollingMeanTransformer().fit(df)
transformed = test.transform(df)
inverse = test.inverse_transform(forecats, trans_method = 'forecast')
df == test.inverse_transform(test.transform(df), trans_method = 'original')
inverse == df_wide_numeric.tail(10)
"""
class DifferencedTransformer(object):
    """Difference from lag n value
    inverse_transform can only be applied to the original series, or an immediately following forecast
    
    Args:
        lag (int): number of periods to shift (not implemented, default = 1)
    """
    def __init__(self):
        self.lag = 1
        
    def fit(self, df):
        """Fits
        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.last_values = df.tail(self.lag)
        self.first_values = df.head(self.lag)
        return self        
    def transform(self, df):
        """Returns differenced data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        # df = df_wide_numeric.tail(60).head(50)
        # df_forecast = (df_wide_numeric - df_wide_numeric.shift(1)).tail(10)
        df = (df - df.shift(self.lag)).fillna(method = 'bfill')
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
        lag = self.lag
        # add last values, group by lag, cumsum
        if trans_method == 'original':
            df = pd.concat([self.first_values, df.tail(df.shape[0] - lag)])
            return df.cumsum()
        else:
            df_len = df.shape[0]
            df = pd.concat([self.last_values, df], axis = 0)
            return df.cumsum().tail(df_len)

class PctChangeTransformer(object):
    """% Change of Data.
    
    Warning:
        Because % change doesn't play well with zeroes, zeroes are replaced by positive of the lowest non-zero value.
        Inverse transformed values returned will also not return as 'exactly' equals due to floating point imprecision.
        inverse_transform can only be applied to the original series, or an immediately following forecast

    """
    def __init__(self):
        self.name = 'PctChangeTransformer'
    
    def fit(self, df):
        """Fits.
        
        Args:
            df (pandas.DataFrame): input dataframe
        """
        temp = df.replace([0], np.nan).fillna((df[df != 0]).abs().min(axis = 0)).fillna(0.1)
        self.last_values = temp.tail(1)
        self.first_values = temp.head(1)
        return self
    def transform(self, df):
        """Returns changed data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = df.replace([0], np.nan)
        df = df.fillna((df[df != 0]).abs().min(axis = 0)).fillna(0.1)
        df = df.pct_change(periods=1, fill_method='ffill').fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        return df
    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame
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
        df = (df + 1).replace([0], np.nan)
        df = df.fillna((df[df != 0]).abs().min()).fillna(0.1)

        # add last values, group by lag, cumprod
        if trans_method == 'original':
            df = pd.concat([self.first_values, df.tail(df.shape[0] - 1)], axis = 0)
            return df.cumprod()
        else:
            df_len = df.shape[0]
            df = pd.concat([self.last_values, df], axis = 0)
            return df.cumprod().tail(df_len)

class CumSumTransformer(object):
    """Cumulative Sum of Data.
    
    Warning:
        Inverse transformed values returned will also not return as 'exactly' equals due to floating point imprecision.
        inverse_transform can only be applied to the original series, or an immediately following forecast
    """
    def fit(self, df):
        """Fits.
        
        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.last_values = df.tail(1)
        self.first_values = df.head(1)
        return self
    def transform(self, df):
        """Returns changed data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = df.cumsum(skipna = True)
        return df
    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame
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

        if trans_method == 'original':
            df = pd.concat([self.first_values, (df - df.shift(1)).tail(df.shape[0] - 1)], axis = 0)
            return df
        else:
            df_len = df.shape[0]
            df = pd.concat([self.last_values, df], axis = 0)
            df = df - df.shift(1)
            return df.tail(df_len)

class EmptyTransformer(object):
    def fit(self, df):
        return self
    def transform(self, df):
        return df
    def inverse_transform(self, df):
        return df
    def fit_transform(self, df):
        return df


class GeneralTransformer(object):
    """Remove outliers, fillNA, then mathematical transformations.
    
    Expects a chronologically sorted pandas.DataFrame with a DatetimeIndex, only numeric data, and a 'wide' (one column per series) shape.
    
    Warning:
        - inverse_transform will not fully return the original data under some conditions
            * outliers removed or clipped will be returned in the clipped or filled na form
            * NAs filled will be returned with the filled value
            * Discretization cannot be inversed
            * RollingMean, PctChange, CumSum, and DifferencedTransformer will only return original or an immediately following forecast
                - by default 'forecast' is expected, 'original' can be set in trans_method
    
    Args:       
        outlier_method (str): - level of outlier removal, if any, per series
            'None'
            'clip' - replace outliers with the highest value allowed by threshold
            'remove' - remove outliers and replace with np.nan
        
        outlier_threshold (float): number of std deviations from mean to consider an outlier. Default 3.
        
        outlier_position (str): when to remove outliers
            'first' - remove outliers before other transformations
            'middle' - remove outliers after first_transformation
            'last' - remove outliers after third_transformation

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
            'QuantileTransformer' - Sklearn
            'MaxAbsScaler' - Sklearn
            'StandardScaler' - Sklearn
            'RobustScaler' - Sklearn
            'PCA, 'FastICA' - performs sklearn decomposition and returns n-cols worth of n_components
            'Detrend' - fit then remove a linear regression from the data
            'RollingMean' - 10 period rolling average, can receive a custom window by transformation_param if used as second_transformation
            'FixedRollingMean' - same as RollingMean, but with inverse_transform disabled, so smoothed forecasts are maintained.
            'RollingMean10' - 10 period rolling average (smoothing)
            'RollingMean100thN' - Rolling mean of periods of len(train)/100 (minimum 2)
            'DifferencedTransformer' - makes each value the difference of that value and the previous value
            'PctChangeTransformer' - converts to pct_change, not recommended if lots of zeroes in data
            'SinTrend' - removes a sin trend (fitted to each column) from the data
            'CumSumTransformer' - makes value sum of all previous
        
        second_transformation (str): second transformation to apply. Same options as transformation, but with transformation_param passed in if used

        transformation_param (str): passed to second_transformation, not used by most transformers.
        
        third_transformation (str): third transformation to apply. Sames options as transformation.
        
        discretization (str): method of binning to apply
            None - no discretization
            'center' - values are rounded to center value of each bin
            'lower' - values are rounded to lower range of closest bin
            'upper' - values are rounded up to upper edge of closest bin
            'sklearn-quantile', 'sklearn-uniform', 'sklearn-kmeans' - sklearn kbins discretizer
            
        n_bins (int): number of quantile bins to split data into
        
        random_seed (int): random state passed through where applicable
    """
    
    def __init__(self,
                 outlier_method: str = None, outlier_threshold: float = 3,
                 outlier_position: str = 'first',
                 fillna: str = 'ffill',
                 transformation: str = None, detrend: bool = False,
                 second_transformation: str = None,
                 transformation_param: str = None,
                 third_transformation: str = None,
                 discretization: str = 'center', n_bins: int = None,
                 random_seed: int = 2020):
        
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.outlier_position = outlier_position
        self.fillna = fillna
        self.transformation = transformation
        self.detrend = detrend
        self.second_transformation = second_transformation
        self.transformation_param = transformation_param
        self.third_transformation = third_transformation
        self.discretization = discretization 
        self.n_bins = n_bins
        self.random_seed = random_seed

    def outlier_treatment(self, df):
        """
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        
        Returns:
            pandas.DataFrame
        """
        outlier_method = self.outlier_method
        
        if (outlier_method in [None, 'None']):
            return df
        elif (outlier_method == 'clip'):
            df = clip_outliers(df, std_threshold = self.outlier_threshold)
            return df
        elif (outlier_method == 'remove'):
            df = remove_outliers(df, std_threshold = self.outlier_threshold)
            return df
        else:
            self.outlier_method = None
            return df
    
    def fill_na(self, df, window: int = 10):
        """
        Args:
            df (pandas.DataFrame): Datetime Indexed
            window (int): passed through to rolling mean fill technique
        
        Returns:
            pandas.DataFrame
        """
        df = FillNA(df, method = self.fillna, window = window)
        return df
        
    def _retrieve_transformer(self, df = None, transformation: str = None, param: str = None):
        """
        Args:
            df (pandas.DataFrame): Datetime Indexed - required to set params for some transformers
            transformation (str): name of desired method
            
        Returns:
            transformer object
        """
        
        if transformation in [None, 'None', 'Detrend', 'SinTrend',
                              'DifferencedTransformer', 'RollingMean10',
                              'PctChangeTransformer', 'CumSumTransformer']:
            return {'None': EmptyTransformer(), 
                    None: EmptyTransformer(),
                    'RollingMean10': RollingMeanTransformer(window = 10),
                    'Detrend':Detrend(),
                    'DifferencedTransformer': DifferencedTransformer(),
                    'PctChangeTransformer': PctChangeTransformer(),
                    'SinTrend': SinTrend(),
                    'CumSumTransformer': CumSumTransformer()
                    }[transformation]
        
        elif (transformation == 'MinMaxScaler'):
            from sklearn.preprocessing import MinMaxScaler
            transformer = MinMaxScaler(feature_range=(0, 1), copy=True)
            return transformer
        
        elif (transformation == 'PowerTransformer'):
            from sklearn.preprocessing import PowerTransformer
            transformer = PowerTransformer(method = 'yeo-johnson', standardize=True, copy=True)
            return transformer
        
        elif (transformation == 'QuantileTransformer'):
            from sklearn.preprocessing import QuantileTransformer
            quants = 1000 if df.shape[0] > 1000 else int(df.shape[0] / 3)
            transformer = QuantileTransformer(n_quantiles = quants, copy=True)
            return transformer
        
        elif (transformation == 'StandardScaler'):
            from sklearn.preprocessing import StandardScaler
            transformer = StandardScaler(copy=True)
            return transformer
        
        elif (transformation == 'MaxAbsScaler'):
            from sklearn.preprocessing import MaxAbsScaler
            transformer = MaxAbsScaler(copy=True)
            return transformer
        
        elif (transformation == 'RobustScaler'):
            from sklearn.preprocessing import RobustScaler
            transformer = RobustScaler(copy=True)
            return transformer
        
        elif (transformation in ['RollingMean', 'FixedRollingMean']):
            param = 10 if param is None else param
            if not str(param).isdigit():
                window = int(''.join([s for s in str(param) if s.isdigit()]))
                window = int(df.shape[0]/window)
            else:
                window = int(param)
            window = 2 if window < 2 else window
            self.window = window
            if transformation == 'FixedRollingMean':
                transformer = RollingMeanTransformer(window = self.window, fixed = True)
            else:
                transformer = RollingMeanTransformer(window = self.window, fixed = False)
            return transformer
        
        elif (transformation == 'RollingMean100thN'):
            window = int(df.shape[0]/100)
            window = 2 if window < 2 else window
            self.window = window
            transformer = RollingMeanTransformer(window = self.window)
            return transformer
        
        elif (transformation == 'RollingMean10thN'):
            window = int(df.shape[0]/10)
            window = 2 if window < 2 else window
            self.window = window
            transformer = RollingMeanTransformer(window = self.window)
            return transformer
        
        elif (transformation =='PCA'):
            from sklearn.decomposition import PCA
            transformer = PCA(n_components=df.shape[1], whiten = False, random_state = self.random_seed)
            return transformer
        
        elif (transformation =='FastICA'):
            from sklearn.decomposition import FastICA
            transformer = FastICA(n_components=df.shape[1], whiten = True, random_state = self.random_seed)
            return transformer
        
        else:
            print("Transformation method not known or improperly entered, returning untransformed df")
            transformer = EmptyTransformer
            return transformer
        """
        elif (transformation =='Detrend'):
            transformer = Detrend()
            return transformer
        
        elif (transformation =='DifferencedTransformer'):
            transformer = DifferencedTransformer()
            return transformer
        
        elif (transformation =='SinTrend'):
            transformer = SinTrend()
            return transformer
        
        elif (transformation == 'RollingMean10'):
            self.window = 10
            transformer = RollingMeanTransformer(window = self.window)
            return transformer
        """
        
    def _fit(self, df):
        # clean up outliers
        if 'first' in str(self.outlier_position):
            df = self.outlier_treatment(df)
        
        # fill NaN
        df = self.fill_na(df)
        
        self.df_index = df.index
        self.df_colnames = df.columns
        
        # the first transformation!
        self.transformer = self._retrieve_transformer(df, transformation = self.transformation)
        self.transformer = self.transformer.fit(df)
        df = pd.DataFrame(self.transformer.transform(df))
        df.index = self.df_index
        df.columns = self.df_colnames
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        if self.detrend:
            # Note that currently this is slightly different than the detrend in the separate transformers
            from sklearn.linear_model import LinearRegression
            X = (pd.to_numeric(self.df_index, errors = 'coerce',downcast='integer').values).reshape((-1, 1))
            self.model = LinearRegression(fit_intercept=True).fit(X, df.values)
            df = df - self.model.predict(X)
        
        # clean up outliers
        if 'middle' in str(self.outlier_position):
            df = self.outlier_treatment(df)
            if self.outlier_method == 'remove':
                df = self.fill_na(df)
                if self.fillna in ['fake date']:
                    self.df_index = df.index
        
        # the second transformation! This one has an optional parameter passed through
        self.second_transformer = self._retrieve_transformer(df, transformation = self.second_transformation, param = self.transformation_param)
        self.second_transformer = self.second_transformer.fit(df)
        df = pd.DataFrame(self.second_transformer.transform(df))
        df.index = self.df_index
        df.columns = self.df_colnames
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # the third transformation!
        self.third_transformer = self._retrieve_transformer(df, transformation = self.third_transformation, param = self.transformation_param)
        self.third_transformer = self.third_transformer.fit(df)
        df = pd.DataFrame(self.third_transformer.transform(df))
        df.index = self.df_index
        df.columns = self.df_colnames
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # clean up outliers
        if 'last' in str(self.outlier_position):
            df = self.outlier_treatment(df)
            if self.outlier_method == 'remove':
                df = self.fill_na(df)
                if self.fillna in ['fake date']:
                    self.df_index = df.index
            
        # discretization
        if self.discretization not in [None, 'None']:
            if self.discretization in ['sklearn-quantile', 'sklearn-uniform', 'sklearn-kmeans']:
                from sklearn.preprocessing import KBinsDiscretizer
                self.kbins_discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.discretization.split('-')[1]).fit(df)
                df = pd.DataFrame(self.kbins_discretizer.transform(df))
                df.index = self.df_index
                df.columns = self.df_colnames
            else:
                steps = 1/self.n_bins
                quantiles = np.arange(0, 1 + steps, steps)
                bins = np.nanquantile(df, quantiles, axis=0, keepdims=True)
                if self.discretization == 'center':
                    bins = np.cumsum(bins, dtype=float, axis = 0)
                    bins[2:] = bins[2:] - bins[:-2]
                    bins = bins[2 - 1:] / 2
                elif self.discretization == 'lower':
                    bins = np.delete(bins, (-1), axis=0)
                elif self.discretization == 'upper':
                    bins = np.delete(bins, (0), axis=0)
                self.bins = bins
                binned = (np.abs(df.values - self.bins)).argmin(axis = 0)
                indices = np.indices(binned.shape)[1]
                bins_reshaped = self.bins.reshape((self.n_bins, len(df.columns)))
                df = pd.DataFrame(bins_reshaped[binned, indices], index = self.df_index, columns = self.df_colnames)
        
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
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
        
        # clean up outliers
        if 'first' in str(self.outlier_position):
            df = self.outlier_treatment(df)
        
        # fill NaN
        df = self.fill_na(df)
        
        self.df_index = df.index
        self.df_colnames = df.columns
        
        # first transformation
        df = pd.DataFrame(self.transformer.transform(df))
        df.index = self.df_index
        df.columns = self.df_colnames
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # detrend
        if self.detrend == True:
            X = (pd.to_numeric(self.df_index, errors = 'coerce',downcast='integer').values).reshape((-1, 1))
            df = df - self.model.predict(X)
            # df = pd.DataFrame(df, index = self.df_index, columns = self.df_colnames)
        
        # clean up outliers
        if 'middle' in str(self.outlier_position):
            df = self.outlier_treatment(df)
            if self.outlier_method == 'remove':
                df = self.fill_na(df)
                if self.fillna in ['fake date']:
                    self.df_index = df.index
        
        # second transformation
        df = pd.DataFrame(self.second_transformer.transform(df))
        df.index = self.df_index
        df.columns = self.df_colnames
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # third transformation
        df = pd.DataFrame(self.third_transformer.transform(df))
        df.index = self.df_index
        df.columns = self.df_colnames
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # clean up outliers
        if 'last' in str(self.outlier_position):
            df = self.outlier_treatment(df)
            if self.outlier_method == 'remove':
                df = self.fill_na(df)
                if self.fillna in ['fake date']:
                    self.df_index = df.index
        
        #discretization
        if self.discretization not in [None, 'None']:
            if self.discretization in ['sklearn-quantile', 'sklearn-uniform', 'sklearn-kmeans']:
                df = pd.DataFrame(self.kbins_discretizer.transform(df))
                df.index = self.df_index
                df.columns = self.df_colnames
            else:
                binned = (np.abs(df.values - self.bins)).argmin(axis = 0)
                indices = np.indices(binned.shape)[1]
                bins_reshaped = self.bins.reshape((self.n_bins, df.shape[1]))
                df = pd.DataFrame(bins_reshaped[binned, indices], index = self.df_index, columns = self.df_colnames)
        
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        return df
    
    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Undo the madness
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
            trans_method (str): 'forecast' or 'original' passed through to RollingTransformer, DifferencedTransformer, if used
        """
        self.df_index = df.index
        self.df_colnames = df.columns
        oddities_list = ['DifferencedTransformer', 'RollingMean100thN',
                         'RollingMean10thN', 'RollingMean10', 'RollingMean',
                         'PctChangeTransformer', 'CumSumTransformer']
        
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        # discretization (only needed inverse for sklearn)
        if self.discretization in ['sklearn-quantile', 'sklearn-uniform', 'sklearn-kmeans']:
            df = df.astype(int).clip(lower = 0, upper = (self.n_bins - 1))
            df = pd.DataFrame(self.kbins_discretizer.inverse_transform(df))
            df.index = self.df_index
            df.columns = self.df_colnames
        
        if self.third_transformation in oddities_list:
            df = pd.DataFrame(self.third_transformer.inverse_transform(df,trans_method = trans_method))
        else:
            df = pd.DataFrame(self.third_transformer.inverse_transform(df))
        df.index = self.df_index
        df.columns = self.df_colnames
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        if self.second_transformation in oddities_list:
            df = pd.DataFrame(self.second_transformer.inverse_transform(df, trans_method = trans_method))
        else:
            df = pd.DataFrame(self.second_transformer.inverse_transform(df))
        df.index = self.df_index
        df.columns = self.df_colnames
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        if self.detrend == True:
            X = (pd.to_numeric(self.df_index, errors = 'coerce',
                               downcast='integer').values).reshape((-1, 1))
            df = df + self.model.predict(X)
            # df = pd.DataFrame(df, index = self.df_index, columns = self.df_colnames)
        
        if self.transformation in oddities_list:
            df = pd.DataFrame(self.transformer.inverse_transform(df, trans_method = trans_method))
        else:
            df = pd.DataFrame(self.transformer.inverse_transform(df))
        df.index = self.df_index
        df.columns = self.df_colnames
        
        # since inf just causes trouble. Feel free to debate my choice of replacing with zero.
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        return df


def RandomTransform():
    """Return a dict of randomly choosen transformation selections."""
    transformer_list = [None, 'MinMaxScaler', 'PowerTransformer', 'QuantileTransformer',
                        'MaxAbsScaler', 'StandardScaler', 'RobustScaler', 'PCA',
                        'FastICA', 'Detrend', 'RollingMean10', 'RollingMean100thN',
                        'DifferencedTransformer', 'SinTrend', 'PctChangeTransformer',
                        'CumSumTransformer']
    first_transformer_prob = [0.3, 0.05, 0.22, 0.05,
                              0.05, 0.05, 0.05, 0.01,
                              0.01, 0.01, 0.03, 0.02,
                              0.06, 0.01, 0.05,
                              0.03]
    third_transformer_prob = [0.22, 0.05, 0.1, 0.05,
                              0.05, 0.1, 0.05, 0.05,
                              0.05, 0.05, 0.03, 0.02,
                              0.1, 0.01, 0.04,
                              0.03]
    outlier_method_choice = np.random.choice(a=[None, 'clip', 'remove'],
                                             size=1, p=[0.5, 0.3, 0.2]).item()
    if outlier_method_choice is not None:
        outlier_threshold_choice = np.random.choice(a=[2, 3, 4, 6], size=1,
                                                    p=[0.2, 0.5, 0.2, 0.1]
                                                    ).item()
        outlier_position_choice = np.random.choice(a=['first', 'middle',
                                                      'last', 'first;last',
                                                      'first;middle'],
                                                   size=1,
                                                   p=[0.3, 0.4, 0.1,
                                                      0.1, 0.1]).item()
    else:
        outlier_threshold_choice = None
        outlier_position_choice = None

    na_choice = np.random.choice(a=['ffill', 'fake date', 'rolling mean',
                                    'mean', 'zero',
                                    'ffill mean biased', 'median'],
                                 size=1, p=[0.2, 0.2, 0.2,
                                            0.1, 0.1, 0.1, 0.1]).item()
    transformation_choice = np.random.choice(a=transformer_list, size=1,
                                             p=first_transformer_prob).item()
    detrend_choice = np.random.choice(a=[True, False], size=1,
                                      p=[0.2, 0.8]).item()
    second_transformation_choice = np.random.choice(
        a=[None, 'RollingMean', 'FixedRollingMean', 'other'], size=1,
        p=[0.3, 0.4, 0.1, 0.2]).item()
    if second_transformation_choice == 'other':
        second_transformation_choice = np.random.choice(
            a=transformer_list, size=1,
            p=first_transformer_prob).item()
    if second_transformation_choice in ['RollingMean', 'FixedRollingMean']:
        transformation_param_choice = np.random.choice(
            a=[3, 10, 14, 28, '10thN', '25thN', '100thN'], size=1,
            p=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]).item()
    else:
        transformation_param_choice = None
    third_transformation_choice = np.random.choice(a=transformer_list, size=1,
                                                   p=third_transformer_prob
                                                   ).item()
    discretization_choice = np.random.choice(
        a=[None, 'center', 'lower', 'upper', 'sklearn-quantile',
           'sklearn-uniform', 'sklearn-kmeans'], size=1,
        p=[0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03]).item()
    if discretization_choice is not None:
        n_bins_choice = np.random.choice(a=[5, 10, 25], size=1,
                                         p=[0.8, 0.1, 0.1]).item()
    else:
        n_bins_choice = None
    context_choice = np.random.choice(
        a=[None, 'HalfMax', '2ForecastLength', '6ForecastLength',
           10, '12ForecastLength'], size=1,
        p=[0.7, 0.05, 0.1, 0.05, 0.05, 0.05]).item()
    param_dict = {
            'outlier_method': outlier_method_choice,
            'outlier_threshold': outlier_threshold_choice,
            'outlier_position': outlier_position_choice,
            'fillna': na_choice,
            'transformation': transformation_choice,
            'detrend' : detrend_choice,
            'second_transformation' : second_transformation_choice,
            'transformation_param' : transformation_param_choice,
            'third_transformation' : third_transformation_choice,
            'discretization' : discretization_choice,
            'n_bins' : n_bins_choice,
            'context_slicer' : context_choice
            }
    return param_dict