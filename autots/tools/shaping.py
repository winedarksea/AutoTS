"""Reshape data."""

import re
import numpy as np
import pandas as pd


def infer_frequency(df_wide, warn=True, **kwargs):
    """Infer the frequency in a slightly more robust way.

    Args:
        df_wide (pd.Dataframe or pd.DatetimeIndex): input to pull frequency from
        warn (bool): unused, here to make swappable with pd.infer_freq
    """
    if isinstance(df_wide, pd.DataFrame):
        DTindex = df_wide.index
    elif isinstance(df_wide, pd.DatetimeIndex):
        DTindex = df_wide
    else:
        raise ValueError(
            "infer_frequency failed due to input not being pandas DF or DT index"
        )
    # 'warn' arg removed in pandas 2.0.0
    frequency = pd.infer_freq(DTindex)
    if frequency is None:
        # hack to get around data which has a few oddities
        frequency = pd.infer_freq(DTindex[-10:])
    if frequency is None:
        # hack to get around data which has a few oddities
        frequency = pd.infer_freq(DTindex[:10])
    return frequency


def split_digits_and_non_digits(s):
    # Find all digit and non-digit sequences
    all_parts = re.findall(r'\d+|\D+', s)

    # Separate digit and non-digit parts
    digits = ''.join(part for part in all_parts if part.isdigit())
    nondigits = ''.join(part for part in all_parts if not part.isdigit())

    return digits, nondigits


def freq_to_timedelta(freq):
    """Working around pandas limitations."""
    freq = str(freq).split("-")[0]
    digits, nondigits = split_digits_and_non_digits(freq)
    # 'month start' is being recognized as miliseconds here
    if freq == "MS":
        return pd.to_timedelta(28, units='D')
    if not digits:
        freq = "1" + freq
    try:
        new_freq = pd.to_timedelta(freq)
    except Exception:
        if "M" in freq:
            new_freq = pd.to_timedelta(28, unit='D')
        elif 'y' in freq.lower():
            new_freq = pd.to_timedelta(364, unit='D')
        else:
            raise ValueError(
                f"freq {freq} not recognized for to_timedelta. Please report this issue to AutoTS if found"
            )
    return new_freq


def df_cleanup(
    df_wide,
    frequency: str = "infer",
    prefill_na: str = None,
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
    if drop_most_recent is None:
        drop_most_recent = 0
    elif not isinstance(drop_most_recent, (float, int)):
        raise ValueError(
            f"drop_most_recent must be an integer or None, not {drop_most_recent}"
        )
    # check to make sure column names are unique
    if verbose > 0:
        dupes = df_wide.columns.duplicated()
        if sum(dupes) > 0:
            print(f"Warning, series ids are not unique: {df_wide.columns[dupes]}")

    # infer frequency
    if frequency == 'infer':
        inferred_freq = infer_frequency(df_wide)
        frequency = inferred_freq

    # trying to avoid resampling if necessary because it can cause unexpected data changes
    # test if dates are missing from index
    expected_index = pd.date_range(df_wide.index[0], df_wide.index[-1], freq=frequency)
    # or at least if lengths are the same, which should be a 'good enough' test for likely issues
    if len(df_wide.index) != len(expected_index):
        # fill missing dates in index with NaN, resample to freq as necessary
        try:
            df_wide = df_wide.resample(frequency).apply(aggfunc)
        except Exception:
            df_wide = df_wide.asfreq(frequency, fill_value=np.nan)

    # drop older data, because too much of a good thing...
    if str(drop_data_older_than_periods).isdigit():
        if int(drop_data_older_than_periods) < df_wide.shape[0]:
            if verbose >= 1:
                print("Old data dropped by `drop_data_older_than_periods`.")
            df_wide = df_wide.tail(int(drop_data_older_than_periods))

    # fill NaN now if asked:
    if prefill_na is not None:
        if str(prefill_na).isdigit():
            df_wide = df_wide.fillna(float(prefill_na))
        elif prefill_na == "mean":
            df_wide = df_wide.fillna(df_wide.mean(axis=0))
        elif prefill_na == "median":
            df_wide = df_wide.fillna(df_wide.median(axis=0))
        else:
            if verbose >= 0:
                print("WARNING: prefill_na method {prefill_na} not recognized.")

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

    return df_wide


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
        df_long[date_col] = pd.to_datetime(df_long[date_col])
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
    """General purpose numeric conversion for pandas dataframes.

    All categorical data and levels must be passed to .fit().
    If new categorical series or levels are present in .transform() it won't work!

    Currently datetimes cannot be inverse_transformed back to datetime

    Args:
        na_strings (list): list of strings to replace as pd.NA
        categorical_fillna (str): how to fill NaN for categorical variables (numeric NaN are unaltered)
            "ffill" - uses forward and backward filling to supply na values
            "indicator" or anything else currently results in all missing replaced with str "missing_value"
        handle_unknown (str): passed through to scikit-learn OrdinalEncoder
        downcast (str): passed to pd.to_numeric, use None or 'float'
        verbose (int): greater than 0 to print some messages
    """

    def __init__(
        self,
        na_strings: list = ['', ' '],  # 'NULL', 'NA', 'NaN', 'na', 'nan'
        categorical_fillna: str = "ffill",
        handle_unknown: str = 'use_encoded_value',
        downcast: str = None,
        verbose: int = 0,
    ):
        self.na_strings = na_strings
        self.verbose = verbose
        self.categorical_fillna = categorical_fillna
        self.handle_unknown = handle_unknown
        self.categorical_flag = False
        self.needs_transformation = True
        self.downcast = downcast

    def _fit(self, df):
        """Fit categorical to numeric."""
        # test if any columns aren't numeric
        if not isinstance(df, pd.DataFrame):  # basically just Series inputs
            df = pd.DataFrame(df)

        if df.shape[1] == df.select_dtypes(include=np.number).shape[1]:
            self.needs_transformation = False
            if self.verbose > 2:
                print("All data is numeric, skipping NumericTransformer")

        if self.needs_transformation:
            # replace some common nan datatypes from strings to nan
            df.replace(self.na_strings, np.nan, inplace=True)  # pd.NA in future

            # convert series to numeric which can be readily converted.
            df = df.apply(pd.to_numeric, errors='ignore', downcast=self.downcast)

            # record which columns are which dtypes
            self.column_order = df.columns
            self.numeric_features = df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            self.categorical_features = list(
                set(df.columns.tolist()) - set(self.numeric_features)
            )

            if len(self.categorical_features) > 0:
                self.categorical_flag = True
            if self.categorical_flag:
                from sklearn.preprocessing import OrdinalEncoder

                df_enc = df[self.categorical_features]
                if self.categorical_fillna == "ffill":
                    df_enc = df_enc.ffill().bfill()
                df_enc = df_enc.fillna('missing_value')
                self.cat_transformer = OrdinalEncoder(
                    handle_unknown=self.handle_unknown, unknown_value=np.nan
                )
                # the + 1 makes it compatible with remove_leading_zeroes
                df_enc = self.cat_transformer.fit_transform(df_enc) + 1
                # df_enc = self.cat_transformer.transform(df_enc) + 1

                self.cat_max = df_enc.max(axis=0)
                self.cat_min = df_enc.min(axis=0)
                if self.verbose > 0:
                    print("Categorical features converted to numeric")
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

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self._fit(df)
        return self

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)

    def transform(self, df):
        """Convert categorical dataset to numeric."""
        if self.needs_transformation:
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            df.replace(self.na_strings, np.nan, inplace=True)
            df = df.apply(pd.to_numeric, errors='ignore')
            if self.categorical_flag:
                df_enc = (df[self.categorical_features]).ffill()
                df_enc = df_enc.bfill().fillna('missing_value')
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
        try:
            df = df.astype(float)
        except ValueError as e:
            raise ValueError(
                f"NumericTransformer.transform() could not convert data to float. {str(e)}."
            )
        return df

    def inverse_transform(self, df, convert_dtypes: bool = False):
        """Convert numeric back to categorical.
        Args:
            df (pandas.DataFrame): df
            convert_dtypes (bool): whether to use pd.convert_dtypes after inverse
        """
        if self.categorical_flag:
            if not isinstance(df, pd.DataFrame):  # basically just Series inputs
                df = pd.DataFrame(df)
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
        if convert_dtypes:
            df = df.convert_dtypes()
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
            """forecast_length is too large for training data.
What this means is you don't have enough history to support cross validation with your forecast_length.
Various solutions include bringing in more data, alter min_allowed_train_percent to something smaller,
and also setting a shorter forecast_length to class init for cross validation which you can then override with a longer value in .predict()
This error is also often caused by errors in inputing of or preshaping the data.
Check model.df_wide_numeric to make sure data was imported correctly.
            """
        )

    train = df.head((df.shape[0]) - forecast_length)
    test = df.tail(forecast_length)

    # not failing on the rational that it effects all models, and therefore all metrics equally
    if (verbose > 0) and ((train.isnull().sum(axis=0) / train.shape[0]).max() > 0.9):
        print("One or more series is 90% or more NaN in this train split")
    if (verbose >= 0) and ((test.isnull().sum(axis=0) / test.shape[0]).max() > 0.9):
        print("One or more series is 90% or more NaN in this test split")
    return train.copy(), test.copy()


def clean_weights(weights, series, verbose=0):
    """Polish up series weighting information

    Args:
        weights (dict): dictionary of series_id: weight (float or int)
        series (iterable): list of series_ids in the dataset
    """
    if not bool(weights):
        weights = {x: 1 for x in series}
    else:
        # handle not all weights being provided
        if verbose > 1:
            key_count = 0
            for col in series:
                if col in weights:
                    key_count += 1
            key_count = len(series) - key_count
            if key_count > 0:
                print(f"{key_count} series_id not in weights. Inferring 1.")
            else:
                print("All series_id present in weighting.")
        weights = {col: (weights[col] if col in weights else 1) for col in series}
        # handle non-numeric inputs, somewhat slower than desired
        for key in weights:
            try:
                weights[key] = abs(float(weights[key]))
            except Exception:
                weights[key] = 1
    return weights


def wide_to_3d(wide_arr, seasonality=7, output_shape="gst"):
    """Generates 3d (groups/seasonality, series, time steps) from wide (time step, series) numpy array.

    Args:
        wide_arr (np.array): wide style (timesteps, series) numpy time series
        seasonality (int): seasonality of the series to use, avoid really large values
        output_shape (str): either 'gst' or 'sgt' which is output shape
            gst: (groups/seasonality, series, time steps)
            sgt: (series, groups/seasonality, time steps)
    """
    excess = wide_arr.shape[0] % seasonality
    cuts = np.arange(seasonality, wide_arr.shape[0] - excess, step=seasonality)
    if output_shape == "sgt":
        shifted = np.array(np.vsplit(wide_arr[excess:], cuts)).T
    else:
        shifted = np.array(np.vsplit(wide_arr[excess:], cuts))
        shifted = np.moveaxis(shifted, 0, -1)
    return shifted
