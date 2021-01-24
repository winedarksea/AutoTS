"""Reshape data."""
import numpy as np
import pandas as pd


def df_cleanup(
    df_wide,
    frequency: str = "infer",
    na_tolerance: float = 0.999,
    drop_data_older_than_periods: int = 100000,
    drop_most_recent: int = 0,
    aggfunc: str = 'first',
    verbose: int = 1,
):
    """Pass cleaning functions through to dataframe.

    Args:
        df_wide (pd.DataFrame): input dataframe to clean.
        frequency (str, optional): frequency in string of alias for DateOffset object, normally "1D" -daily, "MS" -month start etc. Currently, aliases are listed somewhere in here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html. Defaults to "infer".
        na_tolerance (float, optional): allow up to this percent of values to be NaN, else drop the entire series. The default of 0.95 means a series can be 95% NaN values and still be included. Defaults to 0.999.
        drop_data_older_than_periods (int, optional): cut off older data because eventually you just get too much. Defaults to 100000.
        drop_most_recent (int, optional): number of most recent data points to remove. Useful if you pull monthly data before month end, and you don't want an incomplete month appearing complete. Defaults to 0.
        aggfunc (str, optional): passed to pd.pivot_table, determines how to aggregate duplicates for upsampling. Other options include "mean" and other numpy functions, beware data *must* already be input as numeric type for these to work. If categorical data is provided, `aggfunc='first'` is recommended. Defaults to 'first'.
        verbose (int, optional): 0 for silence, higher values for more noise. Defaults to 1.

    Returns:
        pd.DataFrame: original dataframe, now possibly shorter.

    """
    # check to make sure column names are unique
    if verbose > 0:
        dupes = df_wide.columns.duplicated()
        if sum(dupes) > 0:
            print("Warning, series ids are not unique: {df_wide.columns[dupes]}")

    # infer frequency
    if frequency == 'infer':
        frequency = pd.infer_freq(df_wide.index, warn=True)
        if frequency is None:
            # hack to get around data which has a few oddities
            frequency = pd.infer_freq(df_wide.tail(10).index, warn=True)
        if frequency is None:
            # hack to get around data which has a few oddities
            frequency = pd.infer_freq(df_wide.head(10).index, warn=True)
        if verbose > 0:
            print("Inferred frequency is: {}".format(str(frequency)))
    if (frequency is None) and (verbose >= 0):
        print("Frequency is 'None'! Input frequency not recognized.")

    # fill missing dates in index with NaN, resample to freq as necessary
    try:
        df_wide = df_wide.resample(frequency).apply(aggfunc)
    except Exception:
        df_wide = df_wide.asfreq(frequency, fill_value=np.nan)

    # drop older data, because too much of a good thing...
    if str(drop_data_older_than_periods).isdigit():
        if int(drop_data_older_than_periods) < df_wide.shape[0]:
            if verbose >= 0:
                print("Old data dropped by `drop_data_older_than_periods`.")
            df_wide = df_wide.tail(int(drop_data_older_than_periods))

    # remove series with way too many NaNs
    na_tolerance = abs(float(na_tolerance))
    na_tolerance = 1 if na_tolerance > 1 else na_tolerance
    if na_tolerance < 1:
        na_threshold = int(np.floor(df_wide.shape[0] * (1 - na_tolerance)))
        initial_length = df_wide.shape[1]
        df_wide = df_wide.dropna(axis=1, thresh=na_threshold)
        if initial_length != df_wide.shape[1] and verbose >= 0:
            print("Series dropped having too many NaN (see: `na_tolerance`)")

    if (df_wide.shape[1]) < 1:
        raise ValueError("All series filtered! Frequency may be incorrect")

    # drop most recent value when desired
    if drop_most_recent > 0:
        df_wide.drop(df_wide.tail(drop_most_recent).index, inplace=True)

    return pd.DataFrame(df_wide)


def long_to_wide(
    df,
    date_col: str = 'datetime',
    value_col: str = 'value',
    id_col: str = 'series_id',
    aggfunc: str = 'first',
):
    """
    Take long data and convert into wide, cleaner data.

    Args:
        df (pd.DataFrame) - a pandas dataframe having three columns:
        date_col (str) - the name of the column containing dates, preferrably already in pandas datetime format
        value_col (str): - the name of the column with the values of the time series (ie sales $)
        id_col (str): - name of the id column, unique for each time series
        aggfunc (str): - passed to pd.pivot_table, determines how to aggregate duplicates for series_id and datetime
            other options include "mean" and other numpy functions, beware data *must* already be input as numeric type for these to work.
            if categorical data is provided, `aggfunc='first'` is recommended
    """
    df_long = df.copy()

    # Attempt to convert to datetime format if not already
    try:
        df_long[date_col] = pd.to_datetime(
            df_long[date_col], infer_datetime_format=True
        )
    except Exception:
        raise ValueError(
            "Could not convert date to datetime format. Incorrect column name or preformat with pandas to_datetime"
        )

    # handle no id_col for if only one time series
    if id_col in [None, 'None']:
        df_long[id_col] = 'First'
        df_long.drop_duplicates(subset=date_col, keep='first', inplace=True)

    # drop any unnecessary columns
    df_long = df_long[[date_col, id_col, value_col]]

    # pivot to different wide shape
    df_wide = df_long.pivot_table(
        values=value_col, index=date_col, columns=id_col, aggfunc=aggfunc
    )
    df_wide = df_wide.sort_index(ascending=True)

    return pd.DataFrame(df_wide)


class NumericTransformer(object):
    """Test numeric conversion."""

    def __init__(
        self,
        na_strings: list = ['', ' ', 'NULL', 'NA', 'NaN', 'na', 'nan'],
        categorical_impute_strategy: str = 'constant',
        verbose: int = 0,
    ):
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
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = list(
            set(df.columns.tolist()) - set(self.numeric_features)
        )

        if len(self.categorical_features) > 0:
            self.categorical_flag = True
        if self.categorical_flag:
            from sklearn.preprocessing import OrdinalEncoder

            df_enc = (df[self.categorical_features]).fillna(method='ffill')
            df_enc = df_enc.fillna(method='bfill').fillna('missing_value')
            self.cat_transformer = OrdinalEncoder()
            self.cat_transformer.fit(df_enc)

            # the + 1 makes it compatible with remove_leading_zeroes
            df_enc = self.cat_transformer.transform(df_enc) + 1
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
            df_enc = (df[self.categorical_features]).fillna(method='ffill')
            df_enc = df_enc.fillna(method='bfill').fillna('missing_value')
            df_enc = self.cat_transformer.transform(df_enc) + 1
            df = pd.concat(
                [
                    pd.DataFrame(
                        df[self.numeric_features], columns=self.numeric_features
                    ),
                    pd.DataFrame(
                        df_enc, columns=self.categorical_features, index=df.index
                    ),
                ],
                axis=1,
            )[self.column_order]
        return df.astype(float)

    def inverse_transform(self, df):
        """Convert numeric back to categorical."""
        if self.categorical_flag:
            df_enc = (
                df[self.categorical_features].clip(
                    upper=self.cat_max, lower=self.cat_min, axis=1
                )
                - 1
            )
            df_enc = self.cat_transformer.inverse_transform(df_enc)
            df = pd.concat(
                [
                    pd.DataFrame(
                        df[self.numeric_features], columns=self.numeric_features
                    ),
                    pd.DataFrame(
                        df_enc, columns=self.categorical_features, index=df.index
                    ),
                ],
                axis=1,
            )[self.column_order]
        return df


def subset_series(df, weights, n: int = 1000, random_state: int = 2020):
    """Return a sample of time series.

    Args:
        df (pd.DataFrame): wide df with series as columns and DT index
        n (int): number of unique time series to keep, or None
        random_state (int): random seed
    """
    if n is None:
        return df
    else:
        n = int(n)

    if n > df.shape[1]:
        return df
    else:
        df = df.sample(
            n, axis=1, random_state=random_state, replace=False, weights=weights
        )
        return df


def simple_train_test_split(
    df,
    forecast_length: int = 10,
    min_allowed_train_percent: float = 0.3,
    verbose: int = 1,
):
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

    if (forecast_length * min_allowed_train_percent) > int(
        (df.shape[0]) - forecast_length
    ):
        raise ValueError(
            "forecast_length is too large, not enough training data, alter min_allowed_train_percent to override, or reduce validation number, if applicable"
        )

    train = df.head((df.shape[0]) - forecast_length)
    test = df.tail(forecast_length)

    if (verbose > 0) and ((train.isnull().sum(axis=0) / train.shape[0]).max() > 0.9):
        print("One or more series is 90% or more NaN in this train split")
    if (verbose >= 0) and ((test.isnull().sum(axis=0) / test.shape[0]).max() > 0.9):
        print("One or more series is 90% or more NaN in this test split")
    return train, test
