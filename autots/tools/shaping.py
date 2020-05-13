"""
Reshape
"""
import numpy as np
import pandas as pd
# import typing
# id_col: typing.Optional[str]=None

def long_to_wide(df, date_col: str = 'datetime', value_col: str = 'value', id_col: str = 'series_id', 
                 frequency: str = "infer", na_tolerance: float = 0.95,
                 drop_data_older_than_periods: int = 100000, 
                 drop_most_recent: int = 0, aggfunc: str ='first',
                 verbose: int = 1):
    """
    Takes long data and converts into wide, cleaner data
    
    args:
    ========
    
        :param df: - a pandas dataframe having three columns:
        :type df: pandas.DataFrame
            
            :param date_col: - the name of the column containing dates, preferrably already in pandas datetime format
            :type date_col: str
            
            :param value_col: - the name of the column with the values of the time series (ie sales $)
            :type value_col: str
            
            :param id_col: - name of the id column, unique for each time series
            :type id_col: str
        
        :param frequency: - frequency in string of alias for DateOffset object, normally "1D" -daily, "MS" -month start etc.
            currently, aliases are listed somewhere in here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        :type frequency: str
        
        :param na_tolerance: - allow up to this percent of values to be NaN, else drop the entire series
            the default of 0.95 means a series can be 95% NaN values and still be included.
        :type na_tolerance: float
        
        :param drop_data_older_than_periods: - cut off older data because eventually you just get too much
            10,000 is meant to be rather high, normally for daily data I'd use only the last couple of years, say 1500 samples
        :type drop_data_older_than_periods: int
        
        :param drop_most_recent: - if to drop the most recent data point
            useful if you pull monthly data before month end, and you don't want an incomplete month appearing complete
        :type drop_most_recent: int
        
        :param aggfunc: - passed to pd.pivot_table, determines how to aggregate duplicates for series_id and datetime
            other options include "mean" and other numpy functions
        :type aggfunc: str
    """
    df_long = df.copy()

    # Attempt to convert to datetime format if not already
    try:
        df_long[date_col] = pd.to_datetime(df_long[date_col], infer_datetime_format = True)
    except Exception:
        raise ValueError("Could not convert date to datetime format. Incorrect column name or preformat with pandas to_datetime")
    
    # handle no id_col for if only one time series
    # this isn't particularly robust, hence an id_col is required
    if (id_col is None) or (id_col == 'None'):
        # print("No id_col passed, using only first time series...")
        df_long[id_col] = 'First'
        df_long.drop_duplicates(subset = date_col, keep = 'first', inplace = True)
    
    # drop any unnecessary columns
    df_long = df_long[[date_col,id_col,value_col]]
    
    # pivot to different wide shape
    df_wide = df_long.pivot_table(values=value_col, index=date_col, columns=id_col, aggfunc = aggfunc)
    df_wide = df_wide.sort_index(ascending=True)
    
    # drop older data, because too much of a good thing...
    # okay, so technically it may not be periods until after asfreq, but whateva
    if str(drop_data_older_than_periods).isdigit():
        df_wide = df_wide.tail(int(drop_data_older_than_periods))
    
    # infer frequency
    if frequency == 'infer':
        frequency = pd.infer_freq(df_wide.index, warn = True)
        if frequency == None:
            # hack to get around data which has a few oddities
            frequency = pd.infer_freq(df_wide.head(10).index, warn = True)
        if frequency == None:
            # hack to get around data which has a few oddities
            frequency = pd.infer_freq(df_wide.tail(10).index, warn = True)
        if verbose > 0:
            print("Inferred frequency is: {}".format(str(frequency)))
    if frequency == None:
        print("Achtung! Frequency is 'None'. Input frequency not recognized. Defaulting to daily.")
    
    # fill missing dates in index with NaN
    try:
        df_wide = df_wide.resample(frequency).apply(aggfunc)
    except Exception:
        df_wide = df_wide.asfreq(frequency, fill_value=np.nan)
    
    # remove series with way too many NaNs - probably those of a different frequency, or brand new
    na_threshold = int(len(df_wide.index) * (1 - na_tolerance))
    initial_length = len(df_wide.columns)
    df_wide = df_wide.dropna(axis = 1, thresh=na_threshold)
    if initial_length != len(df_wide.columns):
        print("Some columns dropped as having too many NaN (greater than na_tolerance)")
    
    if len(df_wide.columns) < 1:
        raise ValueError("All series filtered! Probably the na_tolerance is too low or frequency is incorrect")
    
    # drop most recent value when desired
    if drop_most_recent > 0:
        df_wide.drop(df_wide.tail(drop_most_recent).index, inplace = True)
    
    return pd.DataFrame(df_wide)


class NumericTransformer(object):
    """Test numeric conversion."""

    def __init__(self,
                 na_strings: list = ['', ' ', 'NULL', 'NA', 'NaN', 'na', 'nan'],
                 categorical_impute_strategy: str = 'constant',
                 verbose: int = 0):
        self.na_strings = na_strings
        self.categorical_impute_strategy = categorical_impute_strategy
        self.verbose = verbose
        self.categorical_flag = False

    def fit(self, df):
        """Fit categorical to numeric."""
        # replace some common nan datatypes from strings to np.nan
        df.replace(self.na_strings, np.nan, inplace=True)

        # convert series to numeric which can be readily converted.
        df = df.apply(pd.to_numeric, errors='ignore')

        # record which columns are which dtypes
        self.column_order = df.columns
        # df_datatypes = df.dtypes
        self.numeric_features = (df.select_dtypes(include=[np.number]).columns.tolist())
        self.categorical_features = list(set(df.columns.tolist()) - set(self.numeric_features))

        if len(self.categorical_features) > 0:
            self.categorical_flag = True
        if self.categorical_flag:
            from sklearn.preprocessing import OrdinalEncoder
            from sklearn.impute import SimpleImputer
            imp_enc = SimpleImputer(strategy=self.categorical_impute_strategy)
            df_enc = imp_enc.fit_transform(df[self.categorical_features])
            self.cat_transformer = OrdinalEncoder()
            self.cat_transformer.fit(df_enc)

            df_enc = self.cat_transformer.transform(df_enc)
            self.cat_max = df_enc.max(axis=0)
            self.cat_min = df_enc.min(axis=0)
            if self.verbose >= 0:
                print("Categorical features converted to numeric")
        return self

    def transform(self, df):
        """Convert categorical dataset to numeric."""
        df.replace(self.na_strings, np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='ignore')
        if self.categorical_flag:
            from sklearn.impute import SimpleImputer
            imp_enc = SimpleImputer(strategy=self.categorical_impute_strategy)
            df_enc = imp_enc.fit_transform(df[self.categorical_features])
            df_enc = self.cat_transformer.transform(df_enc)
            df = pd.concat([
                pd.DataFrame(df[self.numeric_features],
                             columns=self.numeric_features),
                pd.DataFrame(df_enc, columns=self.categorical_features,
                             index=df.index)
                   ], axis=1)[self.column_order]
        return df.astype(float)

    def inverse_transform(self, df):
        """Convert numeric back to categorical."""
        if self.categorical_flag:
            df_enc = df[self.categorical_features].clip(
                upper=self.cat_max, lower=self.cat_min, axis=1)
            df_enc = self.cat_transformer.inverse_transform(df_enc)
            df = pd.concat([
                pd.DataFrame(df[self.numeric_features],
                             columns=self.numeric_features),
                pd.DataFrame(df_enc, columns=self.categorical_features,
                             index=df.index)
                   ], axis=1)[self.column_order]
        return df

def subset_series(df, weights, n: int = 1000,
                  random_state: int = 2020):
    """Expects a pandas DataFrame in format of output from long_to_wide().
    That is, in the format where the Index is a Date
    and Columns are each a unique time series

    Args:
        n (int): - number of unique time series to keep
        na_tolerance (float): - allow up to this percent of values to be NaN, else drop the entire series
            default is 1.0, allow all NaNs, primarily handled instead by long_to_wide
        random_state (int): - random seed
    """
    # remove series with way too many NaNs - probably those of a different frequency, or brand new
    # na_threshold = int(len(df.index) * (1 - na_tolerance))
    # df = df.dropna(axis = 1, thresh=na_threshold)

    if isinstance(n, (int, float, complex)) and not isinstance(n, bool):
        if n > len(df.shape[1]):
            return df
        else:
            df = df.sample(n, axis=1, random_state=random_state,
                           replace=False, weights=weights)
            return df


def simple_train_test_split(df, forecast_length: int = 10,
                            min_allowed_train_percent: float = 0.3,
                            verbose: int = 1):
    """
    Uses the last periods of forecast_length as the test set, the rest as train
    
    Args:
        forecast_length (int): number of future periods to predict
        
        min_allowed_train_percent (float): - forecast length cannot be greater than 1 - this
            constrains the forecast length from being much larger than than the training data
            note this includes NaNs in current configuration
    
    Returns: 
        train, test  (both pd DataFrames)
    """
    assert forecast_length > 0, "forecast_length must be greater than 0"
    
    if forecast_length > int(len(df.index) * (min_allowed_train_percent)):
        raise ValueError("forecast_length is too large, not enough training data, alter min_allowed_train_percent to override, or reduce validation number, if applicable")
    
    train = df.head(len(df.index) - forecast_length)
    test = df.tail(forecast_length)
    
    if (verbose > 0) and ((train.isnull().sum(axis=0)/train.shape[0]).max() > 0.9):
        print("One or more series is 90% or more NaN in this train split")
    if (verbose > 0) and ((test.isnull().sum(axis=0)/test.shape[0]).max() > 0.9):
        print("One or more series is 90% or more NaN in this test split")
    return train, test
