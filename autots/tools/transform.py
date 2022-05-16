"""Preprocessing data methods."""
import random
import warnings
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

    def __init__(
        self, model: str = 'GLS', phi: float = 1.0, window: int = None, **kwargs
    ):
        super().__init__(name='Detrend')
        self.model = model
        self.need_positive = ['Poisson', 'Gamma', 'Tweedie']
        self.phi = phi
        self.window = window

    @staticmethod
    def get_new_params(method: str = 'random'):
        window = random.choices(
            [None, 365, 900, 30, 90, 10], [2.0, 0.1, 0.1, 0.1, 0.1, 0.1]
        )[0]
        if method == "fast":
            choice = random.choices(["GLS", "Linear"], [0.5, 0.5], k=1)[0]
            phi = random.choices([1, 0.999, 0.998, 0.99], [0.9, 0.05, 0.01, 0.01])[0]
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
                [0.3, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                k=1,
            )[0]
            phi = random.choices([1, 0.999, 0.998, 0.99], [0.9, 0.1, 0.05, 0.05])[0]
        return {
            "model": choice,
            "phi": phi,
            "window": window,
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

        Y = df.to_numpy()
        X = pd.to_numeric(df.index, errors='coerce', downcast='integer').to_numpy()
        if self.window is not None:
            Y = Y[-self.window :]
            X = X[-self.window :]
        if self.model == 'GLS':
            from statsmodels.regression.linear_model import GLS

            self.trained_model = GLS(Y, X, missing='drop').fit()
        else:
            self.trained_model = self._retrieve_detrend(detrend=self.model)
            if self.model in self.need_positive:
                self.trnd_trans = PositiveShift(
                    log=False, center_one=True, squared=False
                )
                Y = pd.DataFrame(self.trnd_trans.fit_transform(df)).to_numpy()
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
        # pd.Series([phi] * 10).pow(range(10))
        if self.model in self.need_positive:
            temp = pd.DataFrame(
                self.trained_model.predict(X), index=df.index, columns=df.columns
            )
            temp = self.trnd_trans.inverse_transform(temp)
            df = df - temp
        else:
            if self.model == "GLS" and df.shape[1] == 1:
                pred = self.trained_model.predict(X)
                pred = pred.reshape(-1, 1)
                df = df - pred
            else:
                df = df - self.trained_model.predict(X)
        return df

    def inverse_transform(self, df):
        """Return data to original form.
        Will only match original if phi==1

        Args:
            df (pandas.DataFrame): input dataframe
        """
        # try:
        #     df = df.astype(float)
        # except Exception:
        #     raise ValueError("Data Cannot Be Converted to Numeric Float")
        x_in = df.index
        if not isinstance(x_in, pd.DatetimeIndex):
            x_in = pd.DatetimeIndex(x_in)
        X = pd.to_numeric(x_in, errors='coerce', downcast='integer').values
        if self.model != "GLS":
            X = X.reshape((-1, 1))
        if self.model in self.need_positive:
            temp = self.trnd_trans.inverse_transform(
                pd.DataFrame(
                    self.trained_model.predict(X), index=x_in, columns=df.columns
                )
            )
            if self.phi != 1:
                temp = temp.mul(
                    pd.Series([self.phi] * df.shape[0], index=temp.index).pow(
                        range(df.shape[0])
                    ),
                    axis=0,
                )
            df = df + temp
        else:
            pred = pd.DataFrame(
                self.trained_model.predict(X), index=x_in, columns=df.columns
            )
            if self.phi != 1:
                pred = pred.mul(
                    pd.Series([self.phi] * df.shape[0], index=pred.index).pow(
                        range(df.shape[0])
                    ),
                    axis=0,
                )
            df = df + pred
        return df


class StatsmodelsFilter(EmptyTransformer):
    """Irreversible filters.

    Args:
        method (str): bkfilter or cffilter or convolution_filter
    """

    def __init__(self, method: str = 'bkfilter', **kwargs):
        super().__init__(name="StatsmodelsFilter")
        self.method = method

    def fit_transform(self, df):
        """Fit and Return Detrended DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self.transform(df)

    def transform(self, df):
        """Return detrended data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
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
        elif "convolution_filter" in self.method:
            from statsmodels.tsa.filters.filtertools import convolution_filter

            df = convolution_filter(df, [[0.75] * df.shape[1], [0.25] * df.shape[1]])
            df = df.fillna(method='ffill').fillna(method='bfill')
        return df


class HPFilter(EmptyTransformer):
    """Irreversible filters.

    Args:
        lamb (int): lambda for hpfilter
    """

    def __init__(self, part: str = 'trend', lamb: float = 1600, **kwargs):
        super().__init__(name="HPFilter")
        self.part = part
        self.lamb = lamb

    def fit_transform(self, df):
        """Fit and Return Detrended DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self.transform(df)

    def transform(self, df):
        """Return detrended data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        from statsmodels.tsa.filters.hp_filter import hpfilter

        def _hpfilter_one_return(series, lamb=1600, part="trend"):
            """Convert filter to apply on pd DataFrame."""
            hp_cycle, hp_trend = hpfilter(series, lamb)
            if part == "cycle":
                return hp_cycle
            else:
                return hp_trend

        if df.isnull().values.any():
            raise ValueError("hpfilter does not handle null values.")
        df = df.apply(_hpfilter_one_return, lamb=self.lamb, part=self.part)
        return df

    @staticmethod
    def get_new_params(method: str = 'random'):
        part = random.choices(['trend', 'cycle'], weights=[0.98, 0.02])[0]
        lamb = random.choices(
            [1600, 6.25, 129600, 104976000000], weights=[0.5, 0.2, 0.2, 0.1]
        )[0]
        return {"part": part, "lamb": lamb}


class STLFilter(EmptyTransformer):
    """Irreversible filters.

    Args:
        decomp_type (str): which decomposition to use
        part (str): which part of decomposition to return
        seaonal (int): seaonsal component of STL
    """

    def __init__(
        self, decomp_type="STL", part: str = 'trend', seasonal: int = 7, **kwargs
    ):
        super().__init__(name="STLFilter")
        self.part = part
        self.seasonal = seasonal
        self.decomp_type = decomp_type

    def fit_transform(self, df):
        """Fit and Return Detrended DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self.transform(df)

    def transform(self, df):
        """Return detrended data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        from statsmodels.tsa.seasonal import STL, seasonal_decompose

        def _stl_one_return(series, decomp_type="STL", seasonal=7, part="trend"):
            """Convert filter to apply on pd DataFrame."""
            if str(decomp_type).lower() == 'stl':
                result = STL(series, seasonal=seasonal).fit()
            else:
                result = seasonal_decompose(series)
            if part == "seasonal":
                return result.seasonal
            elif part == "resid":
                return result.resid
            else:
                return result.trend

        if df.isnull().values.any():
            raise ValueError("STLFilter does not handle null values.")

        df = df.apply(
            _stl_one_return,
            decomp_type=self.decomp_type,
            seasonal=self.seasonal,
            part=self.part,
        )
        return df.fillna(method='ffill').fillna(method='bfill')

    @staticmethod
    def get_new_params(method: str = 'random'):
        decomp_type = random.choices(['STL', 'seasonal_decompose'], weights=[0.5, 0.5])[
            0
        ]
        part = random.choices(
            ['trend', 'seasonal', "resid"], weights=[0.98, 0.02, 0.001]
        )[0]
        if decomp_type == "STL":
            seasonal = seasonal_int()
            if seasonal < 7 or method == "fast":
                seasonal = 7
            elif seasonal % 2 == 0:
                seasonal = seasonal - 1
            return {"decomp_type": decomp_type, "part": part, "seasonal": seasonal}
        else:
            return {"decomp_type": decomp_type, "part": part}


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
        guess_amp = np.std(yy) * 2.0**0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

        def sinfunc(t, A, w, p, c):
            return A * np.sin(w * t + p) + c

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = scipy.optimize.curve_fit(
                sinfunc, tt, yy, p0=guess, maxfev=10000
            )
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
        # make this faster (250 columns in 2.5 seconds isn't bad, though)
        fail_count = 0
        for column in df.columns:
            vals = 0
            try:
                y = df[column].values
                vals = self.fit_sin(X, y)
                current_param = pd.DataFrame(vals, index=[column])
            except Exception as e:
                print(f"SinTrend failed with {repr(e)} for {column} with {vals}")
                current_param = pd.DataFrame(
                    {"amp": 0, "omega": 1, "phase": 1, "offset": 1}, index=[column]
                )
                fail_count += 1
            if fail_count >= df.shape[1]:
                raise ValueError("SinTrend Transformer failed on all series.")
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

        sin_df = []
        # make this faster
        for index, row in self.sin_params.iterrows():
            sin_df.append(
                pd.DataFrame(
                    row['amp'] * np.sin(row['omega'] * X + row['phase'])
                    + row['offset'],
                    columns=[index],
                )
            )
        sin_df = pd.concat(sin_df, axis=1)
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
            df = df**2
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
            df = df**0.5
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
    """Remove a regression on datepart from the data. See tools.seasonal.date_part"""

    def __init__(
        self,
        regression_model: dict = {
            "model": 'DecisionTree',
            "model_params": {"max_depth": 5, "min_samples_split": 2},
        },
        datepart_method: str = 'expanded',
        polynomial_degree: int = None,
        **kwargs,
    ):
        super().__init__(name="DatepartRegressionTransformer")
        self.regression_model = regression_model
        self.datepart_method = datepart_method
        self.polynomial_degree = polynomial_degree

    @staticmethod
    def get_new_params(method: str = 'random'):
        datepart_choice = random.choice(["simple", "expanded", "recurring", "simple_2"])
        if datepart_choice in ["simple", "simple_2", "recurring"]:
            polynomial_choice = random.choices([None, 2], [0.5, 0.2])[0]
        else:
            polynomial_choice = None
        from autots.models.sklearn import generate_regressor_params

        if method == "all":
            choice = generate_regressor_params()
        elif method == "fast":
            choice = generate_regressor_params(
                model_dict={
                    'ElasticNet': 0.5,
                    'DecisionTree': 0.5,
                    # 'ExtraTrees': 0.25,
                }
            )
        else:
            choice = generate_regressor_params(
                model_dict={
                    'ElasticNet': 0.25,
                    'DecisionTree': 0.25,
                    'KNN': 0.1,
                    'MLP': 0.2,
                    'RandomForest': 0.2,
                    'ExtraTrees': 0.25,
                    "SVM": 0.1,
                    "RadiusRegressor": 0.1,
                }
            )

        return {
            "regression_model": choice,
            "datepart_method": datepart_choice,
            "polynomial_degree": polynomial_choice,
        }

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
        if y.shape[1] == 1:
            y = y.ravel()
        X = date_part(
            df.index,
            method=self.datepart_method,
            polynomial_degree=self.polynomial_degree,
        )
        from autots.models.sklearn import retrieve_regressor

        multioutput = True
        if y.ndim < 2:
            multioutput = False
        elif y.shape[1] < 2:
            multioutput = False
        self.model = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=0,
            verbose_bool=False,
            random_seed=2020,
            multioutput=multioutput,
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

        X = date_part(
            df.index,
            method=self.datepart_method,
            polynomial_degree=self.polynomial_degree,
        )
        # X.columns = [str(xc) for xc in X.columns]
        y = pd.DataFrame(self.model.predict(X), columns=df.columns, index=df.index)
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

        X = date_part(
            df.index,
            method=self.datepart_method,
            polynomial_degree=self.polynomial_degree,
        )
        y = pd.DataFrame(self.model.predict(X), columns=df.columns, index=df.index)
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
            if df.isnull().values.any():
                raise ValueError("NaN in DifferencedTransformer.inverse_transform")
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
        decimals: int = 0,
        on_transform: bool = False,
        on_inverse: bool = True,
        force_int: bool = False,
        **kwargs,
    ):
        super().__init__(name="Round")
        self.decimals = int(decimals)
        self.on_transform = on_transform
        self.on_inverse = on_inverse
        self.force_int = force_int

    @staticmethod
    def get_new_params(method: str = 'random'):
        on_inverse_c = bool(random.getrandbits(1))
        on_transform_c = bool(random.getrandbits(1))
        if not on_inverse_c and not on_transform_c:
            on_inverse_c = True
        choice = random.choices([-2, -1, 0, 1, 2], [0.1, 0.2, 0.4, 0.2, 0.1], k=1)[0]
        return {
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
        nan_flag (bool): set to True if this has to run on NaN values
    """

    def __init__(
        self, discretization: str = "center", n_bins: int = 10, nan_flag=False, **kwargs
    ):
        super().__init__(name="Discretize")
        self.discretization = discretization
        self.n_bins = n_bins
        self.nan_flag = nan_flag

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
                if self.nan_flag:
                    bins = np.nanquantile(df, quantiles, axis=0, keepdims=True)
                else:
                    bins = np.quantile(df, quantiles, axis=0, keepdims=True)
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


class ScipyFilter(EmptyTransformer):
    """Irreversible filters from Scipy

    Args:
        method (str): "hilbert", "wiener", "savgol_filter", "butter", "cheby1", "cheby2", "ellip", "bessel",
        method_args (list): passed to filter as appropriate
    """

    def __init__(self, method: str = 'hilbert', method_args: list = None, **kwargs):
        super().__init__(name="ScipyFilter")
        self.method = method
        self.method_args = method_args

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

    @staticmethod
    def get_new_params(method: str = 'random'):
        method = random.choices(
            [
                "hilbert",
                "wiener",
                "savgol_filter",
                "butter",
                # "cheby1",
                # "cheby2",
                # "ellip",
                # "bessel",
            ],
            [0.1, 0.2, 0.2, 0.1],
            k=1,
        )[0]
        # analog_choice = bool(random.randint(0, 1))
        analog_choice = False
        xn = random.randint(1, 99)
        btype = random.choice(["lowpass", "highpass"])  # "bandpass", "bandstop"
        if method in ['wiener', 'hilbert']:
            method_args = None
        elif method == "savgol_filter":
            method_args = [random.randrange(5, 11, 2), random.randint(1, 3)]
        elif method in ["butter", "bessel"]:
            if btype in ["bandpass", "bandstop"]:
                Wn = [xn / 100, random.randint(1, 99) / 100]
            else:
                Wn = xn / 100 if not analog_choice else xn
            method_args = [
                random.randint(1, 5),
                Wn,
                btype,
                analog_choice,
            ]
        elif method in ["cheby1", "cheby2"]:
            if btype in ["bandpass", "bandstop"]:
                Wn = [xn / 100, random.randint(1, 99) / 100]
            else:
                Wn = xn / 100 if not analog_choice else xn
            method_args = [
                random.randint(1, 5),
                random.randint(1, 10),
                Wn,
                btype,
                analog_choice,
            ]
        elif method in ["ellip"]:
            if btype in ["bandpass", "bandstop"]:
                Wn = [xn / 100, random.randint(1, 99) / 100]
            else:
                Wn = xn / 100 if not analog_choice else xn
            method_args = [
                random.randint(1, 5),
                random.randint(1, 10),
                random.randint(1, 10),
                Wn,
                btype,
                analog_choice,
            ]
        return {
            "method": method,
            "method_args": method_args,
        }

    def transform(self, df):
        """Return detrended data.

        Args:
            df (pandas.DataFrame): input dataframe
        """

        if self.method == 'hilbert':
            from scipy.signal import hilbert

            test = pd.DataFrame(hilbert(df.values), columns=df.columns, index=df.index)
            return np.abs(test)
        elif self.method == 'wiener':
            from scipy.signal import wiener

            return pd.DataFrame(wiener(df.values), columns=df.columns, index=df.index)
        elif self.method == 'savgol_filter':
            from scipy.signal import savgol_filter

            # args = [5, 2]
            return pd.DataFrame(
                savgol_filter(df.values, *self.method_args, axis=0, mode='nearest'),
                columns=df.columns,
                index=df.index,
            )
        elif self.method == 'butter':
            from scipy.signal import butter, sosfiltfilt

            # args = [4, 0.125]
            # [4, 100, 'lowpass'], [1, 0.125, "highpass"]
            sos = butter(*self.method_args, output='sos')
            return pd.DataFrame(
                sosfiltfilt(sos, df.values, axis=0), columns=df.columns, index=df.index
            )
        elif self.method == "cheby1":
            from scipy.signal import cheby1, sosfiltfilt

            # args = [4, 5, 100, 'lowpass', True]
            # args = [10, 1, 15, 'highpass']
            sos = cheby1(*self.method_args, output='sos')
            return pd.DataFrame(
                sosfiltfilt(sos, df.values, axis=0), columns=df.columns, index=df.index
            )
        elif self.method == "cheby2":
            from scipy.signal import cheby2, sosfiltfilt

            # args = [4, 40, 100, 'lowpass', True]
            # args = [12, 20, 17, 'highpass']
            sos = cheby2(*self.method_args, output='sos')
            return pd.DataFrame(
                sosfiltfilt(sos, df.values, axis=0), columns=df.columns, index=df.index
            )
        elif self.method == "ellip":
            from scipy.signal import ellip, sosfiltfilt

            # args = [4, 5, 40, 100, 'lowpass', True]
            # args = [8, 1, 100, 17, 'highpass']
            sos = ellip(*self.method_args, output='sos')
            return pd.DataFrame(
                sosfiltfilt(sos, df.values, axis=0), columns=df.columns, index=df.index
            )
        elif self.method == "bessel":
            from scipy.signal import bessel, sosfiltfilt

            # args = [4, 100, 'lowpass', True]
            # args = [3, 10, 'highpass']
            sos = bessel(*self.method_args, output='sos')
            return pd.DataFrame(
                sosfiltfilt(sos, df.values, axis=0), columns=df.columns, index=df.index
            )
        else:
            raise ValueError(f"ScipyFilter method {self.method} not found.")

    def inverse_transform(self, df):
        """Return data to original form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df


class EWMAFilter(EmptyTransformer):
    """Irreversible filters of Exponential Weighted Moving Average

    Args:
        span (int): span of exponetial period to convert to alpha
    """

    def __init__(self, span: int = 7, **kwargs):
        super().__init__(name="EWMAFilter")
        self.span = span

    def fit_transform(self, df):
        """Fit and Return Detrended DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self.transform(df)

    def transform(self, df):
        """Return detrended data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df.ewm(span=self.span).mean()

    @staticmethod
    def get_new_params(method: str = 'random'):
        if method == "fast":
            choice = random.choice([3, 7, 10, 12])
        else:
            choice = seasonal_int(include_one=False)
        return {"span": choice}


class FastICA(EmptyTransformer):
    """sklearn FastICA for signal decomposition. But need to store columns.

    Args:
        span (int): span of exponetial period to convert to alpha
    """

    def __init__(self, **kwargs):
        super().__init__(name="FastICA")
        self.kwargs = kwargs

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        from sklearn.decomposition import FastICA

        self.columns = df.columns
        self.index = df.index
        self.transformer = FastICA(**self.kwargs)
        return_df = self.transformer.fit_transform(df)
        return pd.DataFrame(return_df, index=self.index)

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
        return_df = self.transformer.transform(df)
        return pd.DataFrame(return_df, index=df.index)

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return_df = self.transformer.inverse_transform(df)
        return pd.DataFrame(return_df, index=df.index, columns=self.columns)

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)

    @staticmethod
    def get_new_params(method: str = 'random'):
        return {
            "algorithm": random.choice(["parallel", "deflation"]),
            "fun": random.choice(["logcosh", "exp", "cube"]),
            "max_iter": random.choices([100, 250, 500], [0.2, 0.7, 0.1])[0],
            "whiten": random.choices([True, False], [0.9, 0.1])[0],
        }


class PCA(EmptyTransformer):
    """sklearn PCA for signal decomposition. But need to store columns.

    Args:
        span (int): span of exponetial period to convert to alpha
    """

    def __init__(self, **kwargs):
        super().__init__(name="PCA")
        self.kwargs = kwargs

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        from sklearn.decomposition import PCA

        self.columns = df.columns
        self.index = df.index
        self.transformer = PCA(**self.kwargs)
        return_df = self.transformer.fit_transform(df)
        return pd.DataFrame(return_df, index=self.index)

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
        return_df = self.transformer.transform(df)
        return pd.DataFrame(return_df, index=df.index)

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return_df = self.transformer.inverse_transform(df)
        return pd.DataFrame(return_df, index=df.index, columns=self.columns)

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)

    @staticmethod
    def get_new_params(method: str = 'random'):
        return {
            "whiten": random.choices([True, False], [0.2, 0.8])[0],
        }


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
    'convolution_filter': StatsmodelsFilter(method='convolution_filter'),
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
    'ScipyFilter': ScipyFilter,
    'HPFilter': HPFilter,
    'STLFilter': STLFilter,
    "EWMAFilter": EWMAFilter,
    "FastICA": FastICA,
    "PCA": PCA,
}
# where will results will vary if not all series are included together
shared_trans = ['PCA', 'FastICA', "DatepartRegression"]
# transformers not defined in AutoTS
external_transformers = [
    'MinMaxScaler',
    'PowerTransformer',
    'QuantileTransformer',
    'MaxAbsScaler',
    'StandardScaler',
    'RobustScaler',
    # "PCA",
    # "FastICA",
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
                * by default 'forecast' is expected, 'original' can be set in trans_method

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
            "ScipyFilter" - filter data (lose information but smoother!) from scipy
            "HPFilter" - statsmodels hp_filter
            "STLFilter" - seasonal decompose and keep just one part of decomposition
            "EWMAFilter" - use an exponential weighted moving average to smooth data

        transformation_params (dict): params of transformers {0: {}, 1: {'model': 'Poisson'}, ...}
            pass through dictionary of empty dictionaries to utilize defaults

        random_seed (int): random state passed through where applicable
    """

    def __init__(
        self,
        fillna: str = None,
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
        # so much faster not to try to fill NaN if there aren't any NaN
        if isinstance(df, pd.DataFrame):
            self.nan_flag = np.isnan(np.min(df.to_numpy()))
        else:
            self.nan_flag = np.isnan(np.min(np.array(df)))
        if self.nan_flag:
            return FillNA(df, method=self.fillna, window=window)
        else:
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
            if quants == "quarter":
                quants = int(df.shape[0] / 4)
            elif quants == "fifth":
                quants = int(df.shape[0] / 5)
            elif quants == "tenth":
                quants = int(df.shape[0] / 10)
            else:
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

            # could probably may it work, but this is simpler
            if df.shape[1] > df.shape[0]:
                raise ValueError("PCA fails when n series > n observations")
            transformer = PCA(
                n_components=min(df.shape), whiten=False, random_state=random_seed
            )
            return transformer

        elif transformation == 'FastICA':
            from sklearn.decomposition import FastICA

            if df.shape[1] > 500:
                raise ValueError("FastICA fails with > 500 series")
            transformer = FastICA(
                n_components=df.shape[1],
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
        # fill NaN
        df = self.fill_na(df)

        self.df_index = df.index
        self.df_colnames = df.columns
        try:
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
                    df = pd.DataFrame(df, index=self.df_index, columns=self.df_colnames)
                # update index reference if sliced
                if transformation in ['Slice', "FastICA", "PCA"]:
                    self.df_index = df.index
                    self.df_colnames = df.columns
                # df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)
        except Exception as e:
            raise Exception(
                f"Transformer {self.transformations[i]} failed on fit"
            ) from e
        # df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)
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
        i = 0
        for i in sorted(self.transformations.keys()):
            transformation = self.transformations[i]
            df = self.transformers[i].transform(df)
            # convert to DataFrame only if it isn't already
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df, index=self.df_index, columns=self.df_colnames)
            # update index reference if sliced
            if transformation in ['Slice', "FastICA", "PCA"]:
                self.df_index = df.index
                self.df_colnames = df.columns
        # df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)
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
        # df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)
        try:
            for i in sorted(self.transformations.keys(), reverse=True):
                if self.transformations[i] in self.oddities_list:
                    df = self.transformers[i].inverse_transform(
                        df, trans_method=trans_method
                    )
                else:
                    df = self.transformers[i].inverse_transform(df)
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df, index=self.df_index, columns=self.df_colnames)
                elif self.transformations[i] in ["FastICA", "PCA"]:
                    self.df_colnames = df.columns
                # df = df.replace([np.inf, -np.inf], 0)
        except Exception as e:
            raise Exception(
                f"Transformer {self.transformations[i]} failed on inverse"
            ) from e

        if fillzero:
            df = df.fillna(0)

        return df


def get_transformer_params(transformer: str = "EmptyTransformer", method: str = None):
    """Retrieve new random params for new Transformers."""
    if transformer in list(have_params.keys()):
        return have_params[transformer].get_new_params(method=method)
    elif transformer == "QuantileTransformer":
        return {
            "output_distribution": random.choices(
                ["uniform", "normal"], [0.8, 0.2], k=1
            )[0],
            "n_quantiles": random.choices(
                ["quarter", "fifth", "tenth", 1000, 100, 20],
                [0.05, 0.05, 0.05, 0.7, 0.1, 0.05],
                k=1,
            )[0],
        }
    else:
        return {}


# dictionary of probabilities for randomly choosen transformers
transformer_dict = {
    None: 0.0,
    'MinMaxScaler': 0.05,
    'PowerTransformer': 0.02,  # is noticeably slower at scale, if not tons
    'QuantileTransformer': 0.05,
    'MaxAbsScaler': 0.05,
    'StandardScaler': 0.04,
    'RobustScaler': 0.05,
    'PCA': 0.01,
    'FastICA': 0.01,
    'Detrend': 0.1,  # slow with some params, but that's handled in get_params
    'RollingMeanTransformer': 0.02,
    'RollingMean100thN': 0.01,  # old
    'DifferencedTransformer': 0.07,
    'SinTrend': 0.01,
    'PctChangeTransformer': 0.01,
    'CumSumTransformer': 0.02,
    'PositiveShift': 0.02,
    'Log': 0.01,
    'IntermittentOccurrence': 0.01,
    'SeasonalDifference': 0.1,
    'cffilter': 0.01,
    'bkfilter': 0.05,
    'convolution_filter': 0.001,
    "HPFilter": 0.01,
    'DatepartRegression': 0.01,
    "ClipOutliers": 0.05,
    "Discretize": 0.03,
    "CenterLastValue": 0.01,
    "Round": 0.02,
    "Slice": 0.02,
    "ScipyFilter": 0.02,
    "STLFilter": 0.01,
    "EWMAFilter": 0.02,
}
# remove any slow transformers
fast_transformer_dict = transformer_dict.copy()
del fast_transformer_dict['SinTrend']
del fast_transformer_dict['FastICA']
del fast_transformer_dict['ScipyFilter']

# and even more, not just removing slow but also less commonly useful ones
superfast_transformer_dict = {
    None: 0.0,
    'MinMaxScaler': 0.05,
    'MaxAbsScaler': 0.05,
    'StandardScaler': 0.04,
    'RobustScaler': 0.05,
    'Detrend': 0.1,
    'RollingMeanTransformer': 0.02,
    'DifferencedTransformer': 0.1,
    'PositiveShift': 0.02,
    'Log': 0.01,
    'SeasonalDifference': 0.1,
    'bkfilter': 0.05,
    "ClipOutliers": 0.05,
    "Discretize": 0.03,
    "Slice": 0.02,
    "EWMAFilter": 0.01,
}

# probability dictionary of FillNA methods
na_probs = {
    'ffill': 0.4,
    'fake_date': 0.1,
    'rolling_mean': 0.1,
    'rolling_mean_24': 0.1,
    'IterativeImputer': 0.05,  # this parallelizes, uses much memory
    'mean': 0.06,
    'zero': 0.05,
    'ffill_mean_biased': 0.1,
    'median': 0.03,
    None: 0.001,
    "interpolate": 0.4,
    "KNNImputer": 0.05,
    "IterativeImputerExtraTrees": 0.0001,  # and this one is even slower
}


def transformer_list_to_dict(transformer_list):
    """Convert various possibilities to dict."""
    if not transformer_list or transformer_list == "all":
        transformer_list = transformer_dict
    elif transformer_list in ["fast", "default", "Fast"]:
        transformer_list = fast_transformer_dict
    elif transformer_list == "superfast":
        transformer_list = superfast_transformer_dict

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
    superfast_params: bool = None,
    traditional_order: bool = False,
):
    """Return a dict of randomly choosen transformation selections.

    SinTrend is used as a signal that slow parameters are allowed.
    """
    transformer_list, transformer_prob = transformer_list_to_dict(transformer_list)

    # adjust fast/slow based on Transformers allowed
    if fast_params is None:
        fast_params = True
        slow_flags = ["SinTrend"]
        intersects = [i for i in slow_flags if i in transformer_list]
        if intersects:
            fast_params = False
    if superfast_params is None:
        superfast_params = False
        slow_flags = ["DatepartRegression", "ScipyFilter", "QuantileTransformer"]
        intersects = [i for i in slow_flags if i in transformer_list]
        if not intersects:
            superfast_params = True

    # filter na_probs if Fast
    params_method = None
    if fast_params:
        params_method = "fast"
        throw_away = na_prob_dict.pop('IterativeImputer', None)
        throw_away = df_interpolate.pop('spline', None)  # noqa
        throw_away = na_prob_dict.pop('IterativeImputerExtraTrees', None)  # noqa
    if superfast_params:
        params_method = "fast"
        throw_away = na_prob_dict.pop('KNNImputer', None)  # noqa

    # clean na_probs dict
    na_probabilities = list(na_prob_dict.values())
    na_probs_list = [*na_prob_dict]
    # sum_nas = sum(na_probabilities)
    # if sum_nas != 1:
    #     na_probabilities = [float(i) / sum_nas for i in na_probabilities]

    # choose FillNA
    na_choice = random.choices(na_probs_list, na_probabilities)[0]
    if na_choice == "interpolate":
        na_choice = random.choices(
            list(df_interpolate.keys()), list(df_interpolate.values())
        )[0]

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
