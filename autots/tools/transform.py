"""Preprocessing data methods."""
import random
import numpy as np
import pandas as pd
from autots.tools.impute import FillNA, df_interpolate
from autots.tools.seasonal import date_part, seasonal_int


class EmptyTransformer(object):
    """Base transformer returning raw data."""

    def __init__(self, name: str = 'EmptyTransformer', **kwargs):
        self.name = name

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self._fit(df)
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)

    def __repr__(self):
        """Print."""
        return 'Transformer ' + str(self.name) + ', uses standard .fit/.transform'

    @staticmethod
    def get_new_params(method: str = 'random'):
        """Generate new random parameters"""
        if method == 'test':
            return {'test': random.choice([1, 2])}
        else:
            return {}


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
            'n' - any integer length to slice by
            '-n' - full length less this amount
            "0.n" - this percent of the full data
    """
    if method in [None, "None"]:
        return df

    df = df.sort_index(ascending=True)

    if 'forecastlength' in str(method).lower():
        len_int = int([x for x in str(method) if x.isdigit()][0])
        return df.tail(len_int * forecast_length)
    elif method == 'HalfMax':
        return df.tail(int(len(df.index) / 2))
    elif str(method).replace("-", "").replace(".", "").isdigit():
        method = float(method)
        if method >= 1:
            return df.tail(int(method))
        elif method > -1:
            return df.tail(int(df.shape[0] * abs(method)))
        else:
            return df.tail(int(df.shape[0] + method))
    else:
        print("Context Slicer Method not recognized")
        return df


class Detrend(EmptyTransformer):
    """Remove a linear trend from the data."""

    def __init__(self, model: str = 'GLS', **kwargs):
        super().__init__(name='Detrend')
        self.model = model
        self.need_positive = ['Poisson', 'Gamma', 'Tweedie']

    @staticmethod
    def get_new_params(method: str = 'random'):
        if method == "fast":
            choice = random.choices(
                [
                    "GLS",
                    "Linear",
                ],
                [
                    0.5,
                    0.5,
                ],
                k=1,
            )[0]
        else:
            choice = random.choices(
                [
                    "GLS",
                    "Linear",
                    "Poisson",
                    "Tweedie",
                    "Gamma",
                    "TheilSen",
                    "RANSAC",
                    "ARD",
                ],
                [0.24, 0.2, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02],
                k=1,
            )[0]
        return {
            "model": choice,
        }

    def _retrieve_detrend(self, detrend: str = "Linear"):
        if detrend == 'Linear':
            from sklearn.linear_model import LinearRegression

            return LinearRegression(fit_intercept=True)
        elif detrend == "Poisson":
            from sklearn.linear_model import PoissonRegressor
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(
                PoissonRegressor(fit_intercept=True, max_iter=200)
            )
        elif detrend == 'Tweedie':
            from sklearn.linear_model import TweedieRegressor
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(TweedieRegressor(power=1.5, max_iter=200))
        elif detrend == 'Gamma':
            from sklearn.linear_model import GammaRegressor
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(
                GammaRegressor(fit_intercept=True, max_iter=200)
            )
        elif detrend == 'TheilSen':
            from sklearn.linear_model import TheilSenRegressor
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(TheilSenRegressor())
        elif detrend == 'RANSAC':
            from sklearn.linear_model import RANSACRegressor

            return RANSACRegressor()
        elif detrend == 'ARD':
            from sklearn.linear_model import ARDRegression
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(ARDRegression())
        else:
            from sklearn.linear_model import LinearRegression

            return LinearRegression()

    def fit(self, df):
        """Fits trend for later detrending.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")

        Y = df.values
        X = pd.to_numeric(df.index, errors='coerce', downcast='integer').values
        if self.model == 'GLS':
            from statsmodels.regression.linear_model import GLS

            self.trained_model = GLS(Y, X, missing='drop').fit()
        else:
            self.trained_model = self._retrieve_detrend(detrend=self.model)
            if self.model in self.need_positive:
                self.trnd_trans = PositiveShift(
                    log=False, center_one=True, squared=False
                )
                Y = pd.DataFrame(self.trnd_trans.fit_transform(df)).values
            X = X.reshape((-1, 1))
            self.trained_model.fit(X, Y)
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
        X = pd.to_numeric(df.index, errors='coerce', downcast='integer').values
        if self.model != "GLS":
            X = X.reshape((-1, 1))
        # df = df.astype(float) - self.model.predict(X)
        if self.model in self.need_positive:
            temp = pd.DataFrame(
                self.trained_model.predict(X), index=df.index, columns=df.columns
            )
            temp = self.trnd_trans.inverse_transform(temp)
            df = df - temp
        else:
            df = df - self.trained_model.predict(X)
        return df

    def inverse_transform(self, df):
        """Return data to original form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")
        X = pd.to_numeric(df.index, errors='coerce', downcast='integer').values
        if self.model != "GLS":
            X = X.reshape((-1, 1))
        if self.model in self.need_positive:
            temp = pd.DataFrame(
                self.trained_model.predict(X), index=df.index, columns=df.columns
            )
            df = df + self.trnd_trans.inverse_transform(temp)
        else:
            df = df + self.trained_model.predict(X)
        # df = df.astype(float) + self.trained_model.predict(X)
        return df


class StatsmodelsFilter(EmptyTransformer):
    """Irreversible filters.

    Args:
        method (str): bkfilter or cffilter
    """

    def __init__(self, method: str = 'bkfilter', **kwargs):
        super().__init__(name="StatsmodelsFilter")
        self.method = method

    def fit(self, df):
        """Fits filter.

        Args:
            df (pandas.DataFrame): input dataframe
        """
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

        if self.method == 'bkfilter':
            from statsmodels.tsa.filters import bk_filter

            cycles = bk_filter.bkfilter(df, K=1)
            cycles.columns = df.columns
            df = (df - cycles).fillna(method='ffill').fillna(method='bfill')
        elif self.method == 'cffilter':
            from statsmodels.tsa.filters import cf_filter

            cycle, trend = cf_filter.cffilter(df)
            cycle.columns = df.columns
            df = df - cycle
        return df

    def inverse_transform(self, df):
        """Return data to original form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df


class SinTrend(EmptyTransformer):
    """Modelling sin."""

    def __init__(self, **kwargs):
        super().__init__(name="SinTrend")

    def fit_sin(self, tt, yy):
        """Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"

        from user unsym @ https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
        """
        import scipy.optimize

        tt = np.array(tt)
        yy = np.array(yy)
        ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = abs(
            ff[np.argmax(Fyy[1:]) + 1]
        )  # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2.0 ** 0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

        def sinfunc(t, A, w, p, c):
            return A * np.sin(w * t + p) + c

        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=10000)
        A, w, p, c = popt
        # f = w/(2.*np.pi)
        # fitfunc = lambda t: A * np.sin(w*t + p) + c
        return {
            "amp": A,
            "omega": w,
            "phase": p,
            "offset": c,
        }  # , "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

    def fit(self, df):
        """Fits trend for later detrending
        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")

        X = pd.to_numeric(df.index, errors='coerce', downcast='integer').values
        self.sin_params = pd.DataFrame()
        # make this faster
        for column in df.columns:
            try:
                y = df[column].values
                vals = self.fit_sin(X, y)
                current_param = pd.DataFrame(vals, index=[column])
            except Exception as e:
                print(e)
                current_param = pd.DataFrame(
                    {"amp": 0, "omega": 1, "phase": 1, "offset": 1}, index=[column]
                )
            self.sin_params = pd.concat([self.sin_params, current_param], axis=0)
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
            raise ValueError("Data Cannot Be Converted to Numeric Float")
        X = pd.to_numeric(df.index, errors='coerce', downcast='integer').values

        sin_df = pd.DataFrame()
        # make this faster
        for index, row in self.sin_params.iterrows():
            yy = pd.DataFrame(
                row['amp'] * np.sin(row['omega'] * X + row['phase']) + row['offset'],
                columns=[index],
            )
            sin_df = pd.concat([sin_df, yy], axis=1)
        df_index = df.index
        df = df.astype(float).reset_index(drop=True) - sin_df.reset_index(drop=True)
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
            raise ValueError("Data Cannot Be Converted to Numeric Float")
        X = pd.to_numeric(df.index, errors='coerce', downcast='integer').values

        sin_df = pd.DataFrame()
        # make this faster
        for index, row in self.sin_params.iterrows():
            yy = pd.DataFrame(
                row['amp'] * np.sin(row['omega'] * X + row['phase']) + row['offset'],
                columns=[index],
            )
            sin_df = pd.concat([sin_df, yy], axis=1)
        df_index = df.index
        df = df.astype(float).reset_index(drop=True) + sin_df.reset_index(drop=True)
        df.index = df_index
        return df


class PositiveShift(EmptyTransformer):
    """Shift each series if necessary to assure all values >= 1.

    Args:
        log (bool): whether to include a log transform.
        center_one (bool): whether to shift to 1 instead of 0.
        squared (bool): whether to square (**2) values after shift.
    """

    def __init__(
        self, log: bool = False, center_one: bool = True, squared=False, **kwargs
    ):
        super().__init__(name="PositiveShift")
        self.log = log
        self.center_one = center_one
        self.squared = squared

    def fit(self, df):
        """Fits shift interval.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.log or self.center_one:
            shift_amount = df.min(axis=0) - 1
        else:
            shift_amount = df.min(axis=0)
        self.shift_amount = shift_amount.where(shift_amount < 0, 0).abs()

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
        df = df + self.shift_amount
        if self.squared:
            df = df ** 2
        if self.log:
            df_log = pd.DataFrame(np.log(df))
            return df_log
        else:
            return df

    def inverse_transform(self, df):
        """Return data to original form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.log:
            df = pd.DataFrame(np.exp(df))
        if self.squared:
            df = df ** 0.5
        df = df - self.shift_amount
        return df


class IntermittentOccurrence(EmptyTransformer):
    """Intermittent inspired binning predicts probability of not center.

    Does not inverse to original values!

    Args:
        center (str): one of "mean", "median", "midhinge"
    """

    def __init__(self, center: str = "median", **kwargs):
        super().__init__(name="IntermittentOccurrence")
        self.center = center

    @staticmethod
    def get_new_params(method: str = 'random'):
        if method == "fast":
            choice = "mean"
        else:
            choice = random.choices(
                [
                    "mean",
                    "median",
                    "midhinge",
                ],
                [0.4, 0.3, 0.3],
                k=1,
            )[0]
        return {
            "center": choice,
        }

    def fit(self, df):
        """Fits shift interval.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.center == "mean":
            self.df_med = df.mean(axis=0)
        elif self.center == "midhinge":
            self.df_med = (df.quantile(0.75, axis=0) + df.quantile(0.25, axis=0)) / 2
        else:
            self.df_med = df.median(axis=0, skipna=True)
        self.upper_mean = df[df > self.df_med].mean(axis=0) - self.df_med
        self.lower_mean = df[df < self.df_med].mean(axis=0) - self.df_med
        self.lower_mean.fillna(0, inplace=True)
        self.upper_mean.fillna(0, inplace=True)
        return self

    def fit_transform(self, df):
        """Fit and Return Detrended DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)

    def transform(self, df):
        """0 if Median. 1 if > Median, -1 if less.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        temp = df.where(df >= self.df_med, -1)
        temp = temp.where(df <= self.df_med, 1).where(df != self.df_med, 0)
        return temp

    def inverse_transform(self, df):
        """Return data to original form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        invtrans_df = df.copy()

        invtrans_df = invtrans_df.where(df <= 0, self.upper_mean * df, axis=1)
        invtrans_df = invtrans_df.where(
            df >= 0, (self.lower_mean * df).abs() * -1, axis=1
        )
        invtrans_df = invtrans_df + self.df_med
        invtrans_df = invtrans_df.where(df != 0, self.df_med, axis=1)
        return invtrans_df


class RollingMeanTransformer(EmptyTransformer):
    """Attempt at Rolling Mean with built-in inverse_transform for time series
    inverse_transform can only be applied to the original series, or an immediately following forecast
    Does not play well with data with NaNs
    Inverse transformed values returned will also not return as 'exactly' equals due to floating point imprecision.

    Args:
        window (int): number of periods to take mean over
    """

    def __init__(self, window: int = 10, fixed: bool = False, **kwargs):
        super().__init__(name="RollingMeanTransformer")
        self.window = window
        self.fixed = fixed

    @staticmethod
    def get_new_params(method: str = 'random'):
        bool_c = bool(random.getrandbits(1))
        if method == "fast":
            choice = random.choice([3, 7, 10, 12])
        else:
            choice = seasonal_int(include_one=False)
        return {"fixed": bool_c, "window": choice}

    def fit(self, df):
        """Fits.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.shape = df.shape
        self.last_values = (
            df.tail(self.window).fillna(method='ffill').fillna(method='bfill')
        )
        self.first_values = (
            df.head(self.window).fillna(method='ffill').fillna(method='bfill')
        )

        df = df.tail(self.window + 1).rolling(window=self.window, min_periods=1).mean()
        self.last_rolling = df.tail(1)
        return self

    def transform(self, df):
        """Returns rolling data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = df.rolling(window=self.window, min_periods=1).mean()
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
                diffed = ((df.astype(float) - df.shift(1).astype(float)) * window).tail(
                    len(df.index) - window
                )
                temp_cols = diffed.columns
                for n in range(len(diffed.index)):
                    temp_index = diffed.index[n]
                    temp_row = diffed.iloc[n].reset_index(drop=True) + staged.iloc[
                        n
                    ].reset_index(drop=True).astype(float)
                    temp_row = pd.DataFrame(
                        temp_row.values.reshape(1, len(temp_row)), columns=temp_cols
                    )
                    temp_row.index = pd.DatetimeIndex([temp_index])
                    staged = pd.concat([staged, temp_row], axis=0)
                return staged

            # current_inversed = current * window - cumsum(window-1 to previous)
            if trans_method == 'forecast':
                staged = self.last_values
                df = pd.concat([self.last_rolling, df], axis=0)
                diffed = ((df.astype(float) - df.shift(1).astype(float)) * window).tail(
                    len(df.index)
                )
                diffed = diffed.tail(len(diffed.index) - 1)
                temp_cols = diffed.columns
                for n in range(len(diffed.index)):
                    temp_index = diffed.index[n]
                    temp_row = diffed.iloc[n].reset_index(drop=True) + staged.iloc[
                        n
                    ].reset_index(drop=True).astype(float)
                    temp_row = pd.DataFrame(
                        temp_row.values.reshape(1, len(temp_row)), columns=temp_cols
                    )
                    temp_row.index = pd.DatetimeIndex([temp_index])
                    staged = pd.concat([staged, temp_row], axis=0)
                staged = staged.tail(len(diffed.index))
                return staged


"""
df = df_wide_numeric.tail(60).head(50).fillna(0)
df_forecast = (df_wide_numeric).tail(10).fillna(0)
forecats = transformed.tail(10)
test = RollingMeanTransformer().fit(df)
transformed = test.transform(df)
inverse = test.inverse_transform(forecats, trans_method = 'forecast')
df == test.inverse_transform(test.transform(df), trans_method = 'original')
inverse == df_wide_numeric.tail(10)
"""
"""
df = df_wide_numeric.tail(60).fillna(0)
test = SeasonalDifference().fit(df)
transformed = test.transform(df)
forecats = transformed.tail(10)
df == test.inverse_transform(transformed, trans_method = 'original')

df = df_wide_numeric.tail(60).head(50).fillna(0)
test = SeasonalDifference().fit(df)
inverse = test.inverse_transform(forecats, trans_method = 'forecast')
inverse == df_wide_numeric.tail(10).fillna(0)
"""


class SeasonalDifference(EmptyTransformer):
    """Remove seasonal component.

    Args:
        lag_1 (int): length of seasonal period to remove.
        method (str): 'LastValue', 'Mean', 'Median' to construct seasonality
    """

    def __init__(self, lag_1: int = 7, method: str = 'LastValue', **kwargs):
        super().__init__(name="SeasonalDifference")
        self.lag_1 = int(abs(lag_1))
        self.method = method

    @staticmethod
    def get_new_params(method: str = 'random'):
        method_c = random.choice(['LastValue', 'Mean', "Median"])
        if method == "fast":
            choice = random.choice([7, 12])
        else:
            choice = seasonal_int(include_one=False)
        return {"lag_1": choice, "method": method_c}

    def fit(self, df):
        """Fits.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        df_length = df.shape[0]

        if self.method in ['Mean', 'Median']:
            df2 = df.copy()
            tile_index = np.tile(
                np.arange(self.lag_1), int(np.ceil(df_length / self.lag_1))
            )
            tile_index = tile_index[len(tile_index) - (df_length) :]
            df2.index = tile_index
            if self.method == "Median":
                self.tile_values_lag_1 = df2.groupby(level=0, axis=0).median()
            else:
                self.tile_values_lag_1 = df2.groupby(level=0, axis=0).mean()
        else:
            self.method == 'LastValue'
            self.tile_values_lag_1 = df.tail(self.lag_1)
        return self

    def transform(self, df):
        """Returns rolling data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        tile_len = len(self.tile_values_lag_1.index)  # self.lag_1
        df_len = df.shape[0]
        sdf = pd.DataFrame(
            np.tile(self.tile_values_lag_1, (int(np.ceil(df_len / tile_len)), 1))
        )
        #
        sdf = sdf.tail(df_len)
        sdf.index = df.index
        sdf.columns = df.columns
        return df - sdf

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
        tile_len = len(self.tile_values_lag_1.index)
        df_len = df.shape[0]
        sdf = pd.DataFrame(
            np.tile(self.tile_values_lag_1, (int(np.ceil(df_len / tile_len)), 1))
        )
        if trans_method == 'original':
            sdf = sdf.tail(df_len)
        else:
            sdf = sdf.head(df_len)
        sdf.index = df.index
        sdf.columns = df.columns
        return df + sdf


class DatepartRegressionTransformer(EmptyTransformer):
    """Remove a regression on datepart from the data."""

    def __init__(
        self,
        regression_model: dict = {
            "model": 'DecisionTree',
            "model_params": {"max_depth": 5, "min_samples_split": 2},
        },
        datepart_method: str = 'expanded',
        **kwargs,
    ):
        super().__init__(name="DatepartRegressionTransformer")
        self.regression_model = regression_model
        self.datepart_method = datepart_method

    @staticmethod
    def get_new_params(method: str = 'random'):
        method_c = random.choice(["simple", "expanded", "recurring"])
        from autots.models.sklearn import generate_regressor_params

        if method == "all":
            choice = generate_regressor_params()
        else:
            choice = generate_regressor_params(
                model_dict={
                    'ElasticNet': 0.25,
                    'DecisionTree': 0.25,
                    'KNN': 0.1,
                    'MLP': 0.2,
                    'RandomForest': 0.2,
                }
            )

        return {"regression_model": choice, "datepart_method": method_c}

    def fit(self, df):
        """Fits trend for later detrending.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")

        y = df.values
        X = date_part(df.index, method=self.datepart_method)
        from autots.models.sklearn import retrieve_regressor

        self.model = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=0,
            verbose_bool=False,
            random_seed=2020,
        )
        self.model = self.model.fit(X, y)
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

        X = date_part(df.index, method=self.datepart_method)
        y = pd.DataFrame(self.model.predict(X))
        y.columns = df.columns
        y.index = df.index
        df = df - y
        return df

    def inverse_transform(self, df):
        """Return data to original form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")

        X = date_part(df.index, method=self.datepart_method)
        y = pd.DataFrame(self.model.predict(X))
        y.columns = df.columns
        y.index = df.index
        df = df + y
        return df


DatepartRegression = DatepartRegressionTransformer


class DifferencedTransformer(EmptyTransformer):
    """Difference from lag n value.
    inverse_transform can only be applied to the original series, or an immediately following forecast

    Args:
        lag (int): number of periods to shift (not implemented, default = 1)
    """

    def __init__(self, **kwargs):
        super().__init__(name="DifferencedTransformer")
        self.lag = 1
        self.beta = 1

    def fit(self, df):
        """Fit.
        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.last_values = df.tail(self.lag)
        self.first_values = df.head(self.lag)
        return self

    def transform(self, df):
        """Return differenced data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        # df = df_wide_numeric.tail(60).head(50)
        # df_forecast = (df_wide_numeric - df_wide_numeric.shift(1)).tail(10)
        df = (df - df.shift(self.lag)).fillna(method='bfill')
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
            df = pd.concat([self.last_values, df], axis=0)
            return df.cumsum().tail(df_len)


class PctChangeTransformer(EmptyTransformer):
    """% Change of Data.

    Warning:
        Because % change doesn't play well with zeroes, zeroes are replaced by positive of the lowest non-zero value.
        Inverse transformed values returned will also not return as 'exactly' equals due to floating point imprecision.
        inverse_transform can only be applied to the original series, or an immediately following forecast
    """

    def __init__(self, **kwargs):
        super().__init__(name="PctChangeTransformer")

    def fit(self, df):
        """Fits.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        temp = (
            df.replace([0], np.nan).fillna((df[df != 0]).abs().min(axis=0)).fillna(0.1)
        )
        self.last_values = temp.tail(1)
        self.first_values = temp.head(1)
        return self

    def transform(self, df):
        """Returns changed data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = df.replace([0], np.nan)
        df = df.fillna((df[df != 0]).abs().min(axis=0)).fillna(0.1)
        df = df.pct_change(periods=1, fill_method='ffill').fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        return df

    def fit_transform(self, df):
        """Fit and Return *Magical* DataFrame.
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
            df = pd.concat([self.first_values, df.tail(df.shape[0] - 1)], axis=0)
            return df.cumprod()
        else:
            df_len = df.shape[0]
            df = pd.concat([self.last_values, df], axis=0)
            return df.cumprod().tail(df_len)


class CumSumTransformer(EmptyTransformer):
    """Cumulative Sum of Data.

    Warning:
        Inverse transformed values returned will also not return as 'exactly' equals due to floating point imprecision.
        inverse_transform can only be applied to the original series, or an immediately following forecast
    """

    def __init__(self, **kwargs):
        super().__init__(name="CumSumTransformer")

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
        df = df.cumsum(skipna=True)
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
            df = pd.concat(
                [self.first_values, (df - df.shift(1)).tail(df.shape[0] - 1)], axis=0
            )
            return df
        else:
            df_len = df.shape[0]
            df = pd.concat([self.last_values, df], axis=0)
            df = df - df.shift(1)
            return df.tail(df_len)


class ClipOutliers(EmptyTransformer):
    """PURGE THE OUTLIERS.

    Args:
        method (str): "clip" or "remove"
        std_threshold (float): number of std devs from mean to call an outlier
        fillna (str): fillna method to use per tools.impute.FillNA
    """

    def __init__(
        self,
        method: str = "clip",
        std_threshold: float = 4,
        fillna: str = None,
        **kwargs,
    ):
        super().__init__(name="ClipOutliers")
        self.method = method
        self.std_threshold = std_threshold
        self.fillna = fillna

    @staticmethod
    def get_new_params(method: str = 'random'):
        fillna_c = None
        if method == "fast":
            method_c = "clip"
            choice = random.choices(
                [
                    "GLS",
                    "Linear",
                ],
                [
                    0.5,
                    0.5,
                ],
                k=1,
            )[0]
        else:
            method_c = random.choice(["clip", "remove"])
            if method_c == "remove":
                fillna_c = random.choice(["ffill", "mean", "rolling_mean_24"])
        choice = random.choices(
            [1, 2, 3, 3.5, 4, 5], [0.1, 0.2, 0.2, 0.2, 0.2, 0.1], k=1
        )[0]
        return {
            "method": method_c,
            "std_threshold": choice,
            "fillna": fillna_c,
        }

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.df_std = df.std(axis=0, skipna=True)
        self.df_mean = df.mean(axis=0, skipna=True)
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.method == "remove":
            df2 = df[np.abs(df - self.df_mean) <= (self.std_threshold * self.df_std)]
        else:
            lower = self.df_mean - (self.df_std * self.std_threshold)
            upper = self.df_mean + (self.df_std * self.std_threshold)
            df2 = df.clip(lower=lower, upper=upper, axis=1)

        if self.fillna is not None:
            df2 = FillNA(df2, method=self.fillna, window=10)
        return df2

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)


class Round(EmptyTransformer):
    """Round all values. Convert into Integers if decimal <= 0.

    Inverse_transform will not undo the transformation!

    Args:
        method (str): only "middle", in future potentially up/ceiling floor/down
        decimals (int): number of decimal places to round to.
        on_transform (bool): perform rounding on transformation
        on_inverse (bool): perform rounding on inverse transform
    """

    def __init__(
        self,
        method: str = "middle",
        decimals: int = 0,
        on_transform: bool = False,
        on_inverse: bool = True,
        **kwargs,
    ):
        super().__init__(name="Round")
        self.method = method
        self.decimals = decimals
        self.on_transform = on_transform
        self.on_inverse = on_inverse

        self.force_int = False
        if decimals <= 0:
            self.force_int = True

    @staticmethod
    def get_new_params(method: str = 'random'):
        on_inverse_c = bool(random.getrandbits(1))
        on_transform_c = bool(random.getrandbits(1))
        if not on_inverse_c and not on_transform_c:
            on_inverse_c = True
        choice = random.choices([-2, -1, 0, 1, 2], [0.1, 0.2, 0.4, 0.2, 0.1], k=1)[0]
        return {
            "model": "middle",
            "decimals": choice,
            "on_transform": on_transform_c,
            "on_inverse": on_inverse_c,
        }

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.on_transform:
            df = df.round(decimals=self.decimals)
            if self.force_int:
                df = df.astype(int)
        return df

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.on_inverse:
            df = df.round(decimals=self.decimals)
            if self.force_int:
                df = df.astype(int)
        return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)


class Slice(EmptyTransformer):
    """Take the .tail() of the data returning only most recent values.

    Inverse_transform will not undo the transformation!

    Args:
        method (str): only "middle", in future potentially up/ceiling floor/down
        forecast_length (int): forecast horizon, scales some slice windows
    """

    def __init__(
        self,
        method: str = "100",
        forecast_length: int = 30,
        **kwargs,
    ):
        super().__init__(name="Slice")
        self.method = method
        self.forecast_length = forecast_length

    @staticmethod
    def get_new_params(method: str = 'random'):
        if method == "fast":
            choice = random.choices([100, 0.5, 0.2], [0.3, 0.5, 0.2], k=1)[0]
        else:
            choice = random.choices(
                [100, 0.5, 0.8, 0.9, 0.3], [0.2, 0.2, 0.2, 0.2, 0.2], k=1
            )[0]
        return {
            "method": choice,
        }

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = simple_context_slicer(
            df,
            method=self.method,
            forecast_length=self.forecast_length,
        )
        return df

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)


class Discretize(EmptyTransformer):
    """Round/convert data to bins.

    Args:
        discretization (str): method of binning to apply
            None - no discretization
            'center' - values are rounded to center value of each bin
            'lower' - values are rounded to lower range of closest bin
            'upper' - values are rounded up to upper edge of closest bin
            'sklearn-quantile', 'sklearn-uniform', 'sklearn-kmeans' - sklearn kbins discretizer
        n_bins (int): number of bins to group data into.
    """

    def __init__(self, discretization: str = "center", n_bins: int = 10, **kwargs):
        super().__init__(name="Discretize")
        self.discretization = discretization
        self.n_bins = n_bins

    @staticmethod
    def get_new_params(method: str = 'random'):
        if method == "fast":
            choice = random.choice(["center", "upper", "lower"])
            n_bin_c = random.choice([5, 10, 20])
        else:
            choice = random.choices(
                [
                    "center",
                    "upper",
                    "lower",
                    'sklearn-quantile',
                    'sklearn-uniform',
                    'sklearn-kmeans',
                ],
                [0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
                k=1,
            )[0]
            n_bin_c = random.choice([5, 10, 20, 50])
        return {
            "discretization": choice,
            "n_bins": n_bin_c,
        }

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.discretization not in [None, 'None']:
            self.df_index = df.index
            self.df_colnames = df.columns
            if self.discretization in [
                'sklearn-quantile',
                'sklearn-uniform',
                'sklearn-kmeans',
            ]:
                from sklearn.preprocessing import KBinsDiscretizer

                self.kbins_discretizer = KBinsDiscretizer(
                    n_bins=self.n_bins,
                    encode='ordinal',
                    strategy=self.discretization.split('-')[1],
                )
                df = pd.DataFrame(self.kbins_discretizer.fit_transform(df))
                df.index = self.df_index
                df.columns = self.df_colnames
                self.bin_min = df.min(axis=0)
                self.bin_max = df.max(axis=0)
            else:
                steps = 1 / self.n_bins
                quantiles = np.arange(0, 1 + steps, steps)
                bins = np.nanquantile(df, quantiles, axis=0, keepdims=True)
                if self.discretization == 'center':
                    bins = np.cumsum(bins, dtype=float, axis=0)
                    bins[2:] = bins[2:] - bins[:-2]
                    bins = bins[2 - 1 :] / 2
                elif self.discretization == 'lower':
                    bins = np.delete(bins, (-1), axis=0)
                elif self.discretization == 'upper':
                    bins = np.delete(bins, (0), axis=0)
                self.bins = bins
                binned = (np.abs(df.values - self.bins)).argmin(axis=0)
                indices = np.indices(binned.shape)[1]
                bins_reshaped = self.bins.reshape((self.n_bins, len(df.columns)))
                df = pd.DataFrame(
                    bins_reshaped[binned, indices],
                    index=self.df_index,
                    columns=self.df_colnames,
                )
        return df

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self._fit(df)
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.discretization not in [None, 'None']:
            if self.discretization in [
                'sklearn-quantile',
                'sklearn-uniform',
                'sklearn-kmeans',
            ]:
                df = pd.DataFrame(self.kbins_discretizer.transform(df))
                df.index = self.df_index
                df.columns = self.df_colnames
            else:
                binned = (np.abs(df.values - self.bins)).argmin(axis=0)
                indices = np.indices(binned.shape)[1]
                bins_reshaped = self.bins.reshape((self.n_bins, df.shape[1]))
                df = pd.DataFrame(
                    bins_reshaped[binned, indices],
                    index=self.df_index,
                    columns=self.df_colnames,
                )
        return df

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """

        if self.discretization in [
            'sklearn-quantile',
            'sklearn-uniform',
            'sklearn-kmeans',
        ]:
            df_index = df.index
            df_colnames = df.columns9
            df = df.clip(upper=self.bin_max, lower=self.bin_min, axis=1)
            df = df.astype(int).clip(lower=0, upper=(self.n_bins - 1))
            df = pd.DataFrame(self.kbins_discretizer.inverse_transform(df))
            df.index = df_index
            df.columns = df_colnames
        return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)


class CenterLastValue(EmptyTransformer):
    """Scale all data relative to the last value(s) of the series.

    Args:
        rows (int): number of rows to average from most recent data
    """

    def __init__(self, rows: int = 1, **kwargs):
        super().__init__(name="CenterLastValue")
        self.rows = rows

    @staticmethod
    def get_new_params(method: str = 'random'):
        choice = random.randint(1, 6)
        return {
            "rows": choice,
        }

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.center = df.tail(self.rows).mean()
        self.center = self.center.replace(0, np.nan)
        if self.center.isnull().any():
            surrogate = df.replace(0, np.nan).median().fillna(1)
            self.center = self.center.fillna(surrogate)
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = df / self.center
        return df

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = df * self.center
        return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)


# lookup dict for all non-parameterized transformers
trans_dict = {
    'None': EmptyTransformer(),
    None: EmptyTransformer(),
    'RollingMean10': RollingMeanTransformer(window=10),
    'DifferencedTransformer': DifferencedTransformer(),
    'PctChangeTransformer': PctChangeTransformer(),
    'SinTrend': SinTrend(),
    'SineTrend': SinTrend(),
    'PositiveShift': PositiveShift(squared=False),
    'Log': PositiveShift(log=True),
    'CumSumTransformer': CumSumTransformer(),
    'SeasonalDifference7': SeasonalDifference(lag_1=7, method='LastValue'),
    'SeasonalDifference12': SeasonalDifference(lag_1=12, method='Mean'),
    'SeasonalDifference28': SeasonalDifference(lag_1=28, method='Mean'),
    'bkfilter': StatsmodelsFilter(method='bkfilter'),
    'cffilter': StatsmodelsFilter(method='cffilter'),
    "Discretize": Discretize(discretization="center", n_bins=10),
    'DatepartRegressionLtd': DatepartRegressionTransformer(
        regression_model={
            "model": 'DecisionTree',
            "model_params": {"max_depth": 4, "min_samples_split": 2},
        },
        datepart_method='recurring',
    ),
    'DatepartRegressionElasticNet': DatepartRegressionTransformer(
        regression_model={"model": 'ElasticNet', "model_params": {}}
    ),
    'DatepartRegressionRandForest': DatepartRegressionTransformer(
        regression_model={"model": 'RandomForest', "model_params": {}}
    ),
}
# transformers with parameter pass through (internal only)
have_params = {
    'RollingMeanTransformer': RollingMeanTransformer,
    'SeasonalDifference': SeasonalDifference,
    'Discretize': Discretize,
    'CenterLastValue': CenterLastValue,
    'IntermittentOccurrence': IntermittentOccurrence,
    'ClipOutliers': ClipOutliers,
    'DatepartRegression': DatepartRegression,
    'Round': Round,
    'Slice': Slice,
    'Detrend': Detrend,
}
# where will results will vary if not all series are included together
shared_trans = ['PCA', 'FastICA']
# transformers not defined in AutoTS
external_transformers = [
    'MinMaxScaler',
    'PowerTransformer',
    'QuantileTransformer',
    'MaxAbsScaler',
    'StandardScaler',
    'RobustScaler',
    "PCA",
    "FastICA",
]


class GeneralTransformer(object):
    """Remove fillNA and then mathematical transformations.

    Expects a chronologically sorted pandas.DataFrame with a DatetimeIndex, only numeric data, and a 'wide' (one column per series) shape.

    Warning:
        - inverse_transform will not fully return the original data under many conditions
            * the primary intention of inverse_transform is to inverse for forecast (immediately following the historical time period) data from models, not to return original data
            * NAs filled will be returned with the filled value
            * Discretization, statsmodels filters, Round, Slice, ClipOutliers cannot be inversed
            * RollingMean, PctChange, CumSum, Seasonal Difference, and DifferencedTransformer will only return original or an immediately following forecast
                - by default 'forecast' is expected, 'original' can be set in trans_method

    Args:
        fillNA (str): - method to fill NA, passed through to FillNA()
            'ffill' - fill most recent non-na value forward until another non-na value is reached
            'zero' - fill with zero. Useful for sales and other data where NA does usually mean $0.
            'mean' - fill all missing values with the series' overall average value
            'median' - fill all missing values with the series' overall median value
            'rolling_mean' - fill with last n (window = 10) values
            'rolling_mean_24' - fill with avg of last 24
            'ffill_mean_biased' - simple avg of ffill and mean
            'fake_date' - shifts forward data over nan, thus values will have incorrect timestamps
            'IterativeImputer' - sklearn iterative imputer
            most of the interpolate methods from pandas.interpolate

        transformations (dict): - transformations to apply {0: "MinMaxScaler", 1: "Detrend", ...}
            'None'
            'MinMaxScaler' - Sklearn MinMaxScaler
            'PowerTransformer' - Sklearn PowerTransformer
            'QuantileTransformer' - Sklearn
            'MaxAbsScaler' - Sklearn
            'StandardScaler' - Sklearn
            'RobustScaler' - Sklearn
            'PCA, 'FastICA' - performs sklearn decomposition and returns n-cols worth of n_components
            'Detrend' - fit then remove a linear regression from the data
            'RollingMeanTransformer' - 10 period rolling average, can receive a custom window by transformation_param if used as second_transformation
            'FixedRollingMean' - same as RollingMean, but with inverse_transform disabled, so smoothed forecasts are maintained.
            'RollingMean10' - 10 period rolling average (smoothing)
            'RollingMean100thN' - Rolling mean of periods of len(train)/100 (minimum 2)
            'DifferencedTransformer' - makes each value the difference of that value and the previous value
            'PctChangeTransformer' - converts to pct_change, not recommended if lots of zeroes in data
            'SinTrend' - removes a sin trend (fitted to each column) from the data
            'CumSumTransformer' - makes value sum of all previous
            'PositiveShift' - makes all values >= 1
            'Log' - log transform (uses PositiveShift first as necessary)
            'IntermittentOccurrence' - -1, 1 for non median values
            'SeasonalDifference' - remove the last lag values from all values
            'SeasonalDifferenceMean' - remove the average lag values from all
            'SeasonalDifference7','12','28' - non-parameterized version of Seasonal
            'CenterLastValue' - center data around tail of dataset
            'Round' - round values on inverse or transform
            'Slice' - use only recent records
            'ClipOutliers' - remove outliers
            'Discretize' - bin or round data into groups
            'DatepartRegression' - move a trend trained on datetime index

        transformation_params (dict): params of transformers {0: {}, 1: {'model': 'Poisson'}, ...}
            pass through dictionary of empty dictionaries to utilize defaults

        random_seed (int): random state passed through where applicable
    """

    def __init__(
        self,
        fillna: str = 'ffill',
        transformations: dict = {},
        transformation_params: dict = {},
        grouping: str = None,
        reconciliation: str = None,
        grouping_ids=None,
        random_seed: int = 2020,
    ):

        self.fillna = fillna
        self.transformations = transformations
        # handle users passing in no params
        if transformation_params is None or not transformation_params:
            keys = transformations.keys()
            transformation_params = {x: {} for x in keys}
        self.transformation_params = transformation_params

        self.grouping = grouping
        self.reconciliation = reconciliation
        self.grouping_ids = grouping_ids

        self.random_seed = random_seed
        self.transformers = {}
        self.oddities_list = [
            'DifferencedTransformer',
            'RollingMean100thN',
            'RollingMean10thN',
            'RollingMean10',
            'RollingMean',
            'RollingMeanTransformer',
            'PctChangeTransformer',
            'CumSumTransformer',
            'SeasonalDifference',
            'SeasonalDifferenceMean',
            'SeasonalDifference7',
            'SeasonalDifference12',
            'SeasonalDifference28',
        ]

    def fill_na(self, df, window: int = 10):
        """
        Args:
            df (pandas.DataFrame): Datetime Indexed
            window (int): passed through to rolling mean fill technique

        Returns:
            pandas.DataFrame
        """
        df = FillNA(df, method=self.fillna, window=window)
        return df

    @classmethod
    def retrieve_transformer(
        self,
        transformation: str = None,
        param: dict = {},
        df=None,
        random_seed: int = 2020,
    ):
        """Retrieves a specific transformer object from a string.

        Args:
            df (pandas.DataFrame): Datetime Indexed - required to set params for some transformers
            transformation (str): name of desired method
            param (dict): dict of kwargs to pass (legacy: an actual param)

        Returns:
            transformer object
        """

        if transformation in (trans_dict.keys()):
            return trans_dict[transformation]

        elif transformation in list(have_params.keys()):
            return have_params[transformation](**param)

        elif transformation == 'MinMaxScaler':
            from sklearn.preprocessing import MinMaxScaler

            return MinMaxScaler()

        elif transformation == 'PowerTransformer':
            from sklearn.preprocessing import PowerTransformer

            transformer = PowerTransformer(
                method='yeo-johnson', standardize=True, copy=True
            )
            return transformer

        elif transformation == 'QuantileTransformer':
            from sklearn.preprocessing import QuantileTransformer

            quants = param["n_quantiles"]
            quants = quants if df.shape[0] > quants else int(df.shape[0] / 3)
            param["n_quantiles"] = quants
            return QuantileTransformer(copy=True, **param)

        elif transformation == 'StandardScaler':
            from sklearn.preprocessing import StandardScaler

            return StandardScaler(copy=True)

        elif transformation == 'MaxAbsScaler':
            from sklearn.preprocessing import MaxAbsScaler

            return MaxAbsScaler(copy=True)

        elif transformation == 'RobustScaler':
            from sklearn.preprocessing import RobustScaler

            return RobustScaler(copy=True)

        elif transformation == 'PCA':
            from sklearn.decomposition import PCA

            transformer = PCA(
                n_components=df.shape[1], whiten=False, random_state=random_seed
            )
            return transformer

        elif transformation == 'FastICA':
            from sklearn.decomposition import FastICA

            transformer = FastICA(
                n_components=df.shape[1],
                whiten=True,
                random_state=random_seed,
                **param,
            )
            return transformer

        elif transformation in ['RollingMean', 'FixedRollingMean']:
            param = 10 if param is None else param
            if not str(param).isdigit():
                window = int(''.join([s for s in str(param) if s.isdigit()]))
                window = int(df.shape[0] / window)
            else:
                window = int(param)
            window = 2 if window < 2 else window
            self.window = window
            if transformation == 'FixedRollingMean':
                transformer = RollingMeanTransformer(window=self.window, fixed=True)
            else:
                transformer = RollingMeanTransformer(window=self.window, fixed=False)
            return transformer

        elif transformation in ['SeasonalDifference', 'SeasonalDifferenceMean']:
            if transformation == 'SeasonalDifference':
                return SeasonalDifference(lag_1=param, method='LastValue')
            else:
                return SeasonalDifference(lag_1=param, method='Mean')

        elif transformation == 'RollingMean100thN':
            window = int(df.shape[0] / 100)
            window = 2 if window < 2 else window
            self.window = window
            return RollingMeanTransformer(window=self.window)

        elif transformation == 'RollingMean10thN':
            window = int(df.shape[0] / 10)
            window = 2 if window < 2 else window
            self.window = window
            return RollingMeanTransformer(window=self.window)

        else:
            print(
                f"Transformation {transformation} not known or improperly entered, returning untransformed df"
            )
            return EmptyTransformer()

    def _fit(self, df):
        """
        if self.grouping is not None:
            from autots.tools.hierarchial import hierarchial

            if 'kmeans' in self.grouping:
                n_groups = int(''.join([s for s in str(self.grouping) if s.isdigit()]))
            else:
                n_groups = 3
            self.hier = hierarchial(
                n_groups=n_groups,
                grouping_method=self.grouping,
                grouping_ids=self.grouping_ids,
                reconciliation=self.reconciliation,
            ).fit(df)
            df = self.hier.transform(df)
        """
        # fill NaN
        df = self.fill_na(df)

        self.df_index = df.index
        self.df_colnames = df.columns
        for i in sorted(self.transformations.keys()):
            transformation = self.transformations[i]
            self.transformers[i] = self.retrieve_transformer(
                transformation=transformation,
                df=df,
                param=self.transformation_params[i],
                random_seed=self.random_seed,
            )
            df = self.transformers[i].fit_transform(df)
            # convert to DataFrame only if it isn't already
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
                df.index = self.df_index
                df.columns = self.df_colnames
            # update index reference if sliced
            if transformation in ['Slice']:
                self.df_index = df.index
                self.df_colnames = df.columns
            # df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)

        df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)
        return df

    def fit(self, df):
        """Apply transformations and return transformer object.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        self._fit(df)
        return self

    def fit_transform(self, df):
        """Directly fit and apply transformations to convert df."""
        return self._fit(df)

    def transform(self, df):
        """Apply transformations to convert df."""
        df = df.copy()
        """
        if self.grouping is not None:
            df = self.hier.transform(df)
        """
        # fill NaN
        df = self.fill_na(df)

        self.df_index = df.index
        self.df_colnames = df.columns
        # transformations
        for i in sorted(self.transformations.keys()):
            transformation = self.transformations[i]
            df = self.transformers[i].transform(df)
            # convert to DataFrame only if it isn't already
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
                df.index = self.df_index
                df.columns = self.df_colnames
            # update index reference if sliced
            if transformation in ['Slice']:
                self.df_index = df.index
                self.df_colnames = df.columns
        df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)
        return df

    def inverse_transform(
        self, df, trans_method: str = "forecast", fillzero: bool = False
    ):
        """Undo the madness.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            trans_method (str): 'forecast' or 'original' passed through
            fillzero (bool): if inverse returns NaN, fill with zero
        """
        self.df_index = df.index
        self.df_colnames = df.columns
        df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)

        for i in sorted(self.transformations.keys(), reverse=True):
            if self.transformations[i] in self.oddities_list:
                df = self.transformers[i].inverse_transform(
                    df, trans_method=trans_method
                )
            else:
                df = self.transformers[i].inverse_transform(df)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
                df.index = self.df_index
                df.columns = self.df_colnames
            df = df.replace([np.inf, -np.inf], 0)

        if fillzero:
            df = df.fillna(0)

        """
        if self.grouping is not None:
            df = self.hier.reconcile(df)
        """
        return df


def get_transformer_params(transformer: str = "EmptyTransformer", method: str = None):
    """Retrieve new random params for new Transformers."""
    if transformer in list(have_params.keys()):
        return have_params[transformer].get_new_params(method=method)
    elif transformer == "FastICA":
        return {
            "algorithm": random.choice(["parallel", "deflation"]),
            "fun": random.choice(["logcosh", "exp", "cube"]),
        }
    elif transformer == "QuantileTransformer":
        return {
            "output_distribution": random.choices(
                ["uniform", "normal"], [0.8, 0.2], k=1
            )[0],
            "n_quantiles": random.choices([1000, 100, 20], [0.7, 0.2, 0.1], k=1)[0],
        }
    else:
        return {}


# dictionary of probabilities for randomly choosen transformers
transformer_dict = {
    None: 0.0,
    'MinMaxScaler': 0.05,
    'PowerTransformer': 0.1,
    'QuantileTransformer': 0.1,
    'MaxAbsScaler': 0.05,
    'StandardScaler': 0.04,
    'RobustScaler': 0.05,
    'PCA': 0.01,
    'FastICA': 0.01,
    'Detrend': 0.05,
    'RollingMeanTransformer': 0.02,
    'RollingMean100thN': 0.01,  # old
    'DifferencedTransformer': 0.1,
    'SinTrend': 0.01,
    'PctChangeTransformer': 0.01,
    'CumSumTransformer': 0.02,
    'PositiveShift': 0.02,
    'Log': 0.01,
    'IntermittentOccurrence': 0.01,
    'SeasonalDifference7': 0.0,  # old
    'SeasonalDifference': 0.08,
    'SeasonalDifference28': 0.0,  # old
    'cffilter': 0.01,
    'bkfilter': 0.05,
    'DatepartRegression': 0.02,
    'DatepartRegressionElasticNet': 0.0,  # old
    'DatepartRegressionLtd': 0.0,  # old
    "ClipOutliers": 0.05,
    "Discretize": 0.05,
    "CenterLastValue": 0.01,
    "Round": 0.05,
    "Slice": 0.01,
}
# remove any slow transformers
fast_transformer_dict = transformer_dict.copy()
del fast_transformer_dict['DatepartRegression']
del fast_transformer_dict['SinTrend']
del fast_transformer_dict['FastICA']

# probability dictionary of FillNA methods
na_probs = {
    'ffill': 0.1,
    'fake_date': 0.1,
    'rolling_mean': 0.1,
    'rolling_mean_24': 0.099,
    'IterativeImputer': 0.1,
    'mean': 0.1,
    'zero': 0.1,
    'ffill_mean_biased': 0.1,
    'median': 0.1,
    None: 0.001,
    "interpolate": 0.1,
}


def transformer_list_to_dict(transformer_list):
    """Convert various possibilities to dict."""
    if not transformer_list or transformer_list == "all":
        transformer_list = transformer_dict
    elif transformer_list == "fast":
        transformer_list = fast_transformer_dict

    if isinstance(transformer_list, dict):
        transformer_prob = list(transformer_list.values())
        transformer_list = [*transformer_list]
        # xsx = sum(transformer_prob)
        # if xsx != 1:
        #     transformer_prob = [float(i) / xsx for i in transformer_prob]
    elif isinstance(transformer_list, list):
        trs_len = len(transformer_list)
        transformer_prob = [1 / trs_len] * trs_len
    else:
        raise ValueError("transformer_list alias not recognized.")
    return transformer_list, transformer_prob


def RandomTransform(
    transformer_list: dict = transformer_dict,
    transformer_max_depth: int = 4,
    na_prob_dict: dict = na_probs,
    fast_params: bool = None,
    traditional_order: bool = False,
):
    """Return a dict of randomly choosen transformation selections.

    DatepartRegression is used as a signal that slow parameters are allowed.
    """
    transformer_list, transformer_prob = transformer_list_to_dict(transformer_list)

    # adjust fast/slow based on Transformers allowed
    if fast_params is None:
        fast_params = True
        slow_flags = ["DatepartRegression"]
        intersects = [i for i in slow_flags if i in transformer_list]
        if intersects:
            fast_params = False

    # filter na_probs if Fast
    params_method = None
    if fast_params:
        params_method = "fast"
        throw_away = na_prob_dict.pop('IterativeImputer', None)
        throw_away = na_prob_dict.pop('interpolate', None)  # NOQA

    # clean na_probs dict
    na_probabilities = list(na_prob_dict.values())
    na_probs_list = [*na_prob_dict]
    # sum_nas = sum(na_probabilities)
    # if sum_nas != 1:
    #     na_probabilities = [float(i) / sum_nas for i in na_probabilities]

    # choose FillNA
    na_choice = random.choices(na_probs_list, na_probabilities)[0]
    if na_choice == "interpolate":
        na_choice = random.choice(df_interpolate)

    # choose length of transformers
    num_trans = random.randint(1, transformer_max_depth)
    # sometimes return no transformation
    if num_trans == 1:
        test = random.choices(["None", "Some"], [0.1, 0.9])[0]
        if test == "None":
            return {
                "fillna": na_choice,
                "transformations": {0: None},
                "transformation_params": {0: {}},
            }
    if traditional_order:
        # handle these not being in TransformerList
        randos = random.choices(transformer_list, transformer_prob, k=5)
        clip = "ClipOutliers" if "ClipOutliers" in transformer_list else randos[0]
        detrend = "Detrend" if "Detrend" in transformer_list else randos[1]
        discretize = "Discretize" if "Discretize" in transformer_list else randos[2]
        # create new dictionary in fixed order
        trans = [clip, randos[3], detrend, randos[4], discretize]
        trans = trans[0:num_trans]
        num_trans = len(trans)
    else:
        trans = random.choices(transformer_list, transformer_prob, k=num_trans)

    keys = list(range(num_trans))
    params = [get_transformer_params(x, method=params_method) for x in trans]
    return {
        "fillna": na_choice,
        "transformations": dict(zip(keys, trans)),
        "transformation_params": dict(zip(keys, params)),
    }
