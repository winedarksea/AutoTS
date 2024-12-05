"""Preprocessing data methods."""

import random
import warnings
import numpy as np
import pandas as pd
from autots.tools.impute import FillNA, df_interpolate
from autots.tools.seasonal import (
    date_part,
    seasonal_int,
    random_datepart,
    half_yr_spacing,
)
from autots.tools.cointegration import coint_johansen, btcd_decompose
from autots.tools.constraint import (
    fit_constraint,
    apply_fit_constraint,
    constraint_new_params,
)
from autots.models.sklearn import (
    generate_regressor_params,
    retrieve_regressor,
    retrieve_classifier,
    generate_classifier_params,
)
from autots.tools.anomaly_utils import (
    anomaly_new_params,
    detect_anomalies,
    anomaly_df_to_holidays,
    holiday_new_params,
    dates_to_holidays,
)
from autots.tools.window_functions import window_lin_reg_mean_no_nan, np_2d_arange
from autots.tools.fast_kalman import KalmanFilter, new_kalman_params
from autots.tools.shaping import infer_frequency
from autots.tools.holiday import holiday_flag
from autots.tools.fft import FFT as fft_class
from autots.tools.percentile import nan_quantile
from autots.tools.fir_filter import (
    generate_random_fir_params,
    fft_fir_filter_to_timeseries,
)

try:
    from scipy.signal import butter, sosfiltfilt, savgol_filter
    from scipy.optimize import curve_fit
    from scipy.stats import norm
    from scipy.signal import fftconvolve
except Exception:
    norm = lambda x: 0.05
    curve_fit = lambda x: "scipy import failed"
    butter = lambda x: "scipy import failed"
    sosfiltfilt = lambda x: "scipy import failed"
    savgol_filter = lambda x: "scipy import failed"
    fftconvolve = lambda x: "scipy import failed"

try:
    from joblib import Parallel, delayed
except Exception:
    pass


class EmptyTransformer(object):
    """Base transformer returning raw data."""

    def __init__(self, name: str = "EmptyTransformer", **kwargs):
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
        return "Transformer " + str(self.name) + ", uses standard .fit/.transform"

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        if method == "test":
            return {"test": random.choice([1, 2])}
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


def clip_outliers(df, std_threshold: float = 4):
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


def simple_context_slicer(df, method: str = "None", forecast_length: int = 30):
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

    if not df.index.is_monotonic_increasing:
        df = df.sort_index(ascending=True)

    if "forecastlength" in str(method).lower():
        len_int = int([x for x in str(method) if x.isdigit()][0])
        return df.tail(len_int * forecast_length)
    elif method == "HalfMax":
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
        raise ValueError(f"context_slicer method `{method}` not recognized")
        return df


class Detrend(EmptyTransformer):
    """Remove a linear trend from the data."""

    def __init__(
        self,
        model: str = "GLS",
        phi: float = 1.0,
        window: int = None,
        transform_dict=None,
        **kwargs,
    ):
        super().__init__(name="Detrend")
        self.model = model
        self.need_positive = ["Poisson", "Gamma", "Tweedie"]
        self.phi = phi
        self.window = window
        self.transform_dict = transform_dict

    @staticmethod
    def get_new_params(method: str = "random"):
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
                ],
                [0.3, 0.4, 0.1, 0.1, 0.1],
                k=1,
            )[0]
            phi = random.choices([1, 0.999, 0.998, 0.99], [0.9, 0.1, 0.05, 0.05])[0]
        return {
            "model": choice,
            "phi": phi,
            "window": window,
            "transform_dict": random_cleaners(),
        }

    def _retrieve_detrend(self, detrend: str = "Linear", multioutput=True):
        if detrend == "Linear":
            from sklearn.linear_model import LinearRegression

            return LinearRegression(fit_intercept=True)
        elif detrend == "Poisson":
            from sklearn.linear_model import PoissonRegressor
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(
                PoissonRegressor(fit_intercept=True, max_iter=200)
            )
        elif detrend == "Tweedie":
            from sklearn.linear_model import TweedieRegressor
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(TweedieRegressor(power=1.5, max_iter=200))
        elif detrend == "Gamma":
            from sklearn.linear_model import GammaRegressor
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(
                GammaRegressor(fit_intercept=True, max_iter=200)
            )
        elif detrend == "TheilSen":
            from sklearn.linear_model import TheilSenRegressor
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(TheilSenRegressor())
        elif detrend == "RANSAC":
            from sklearn.linear_model import RANSACRegressor

            return RANSACRegressor()
        elif detrend == "ARD":
            from sklearn.linear_model import ARDRegression
            from sklearn.multioutput import MultiOutputRegressor

            return MultiOutputRegressor(ARDRegression())
        elif detrend == 'ElasticNet':
            if multioutput:
                from sklearn.linear_model import MultiTaskElasticNet

                regr = MultiTaskElasticNet(alpha=1.0)
            else:
                from sklearn.linear_model import ElasticNet

                regr = ElasticNet(alpha=1.0)
            return regr
        elif detrend == 'Ridge':
            from sklearn.linear_model import Ridge

            return Ridge()
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

        # this is meant mostly for filling in some smoothing we don't want kept on original
        if self.transform_dict is not None:
            model = GeneralTransformer(**self.transform_dict)
            Y = model.fit_transform(df)
        else:
            Y = df.to_numpy()
        X = pd.to_numeric(df.index, errors="coerce", downcast="integer").to_numpy()
        if self.window is not None:
            Y = Y[-self.window :]
            X = X[-self.window :]
        if self.model == "GLS":
            from statsmodels.regression.linear_model import GLS

            self.trained_model = GLS(Y, X, missing="drop").fit()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.trained_model = self._retrieve_detrend(
                    detrend=self.model, multioutput=df.shape[1] > 1
                )
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
        X = pd.to_numeric(df.index, errors="coerce", downcast="integer").values
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
        X = pd.to_numeric(x_in, errors="coerce", downcast="integer").values
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

    def __init__(self, method: str = "bkfilter", **kwargs):
        super().__init__(name="StatsmodelsFilter")
        self.method = method
        self.filters = {
            "bkfilter": self.bkfilter,
            "cffilter": self.cffilter,
            "convolution_filter": self.convolution_filter,
        }

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
        return self.filters[self.method](df)

    def bkfilter(self, df):
        from statsmodels.tsa.filters import bk_filter

        cycles = bk_filter.bkfilter(df, K=1)
        cycles.columns = df.columns
        return (df - cycles).ffill().bfill()

    def cffilter(self, df):
        from statsmodels.tsa.filters import cf_filter

        cycle, trend = cf_filter.cffilter(df)
        if isinstance(cycle, pd.Series):
            cycle = cycle.to_frame()
        cycle.columns = df.columns
        return df - cycle

    def convolution_filter(self, df):
        from statsmodels.tsa.filters.filtertools import convolution_filter

        df = convolution_filter(df, [[0.75] * df.shape[1], [0.25] * df.shape[1]])
        return df.ffill().bfill()


class HPFilter(EmptyTransformer):
    """Irreversible filters.

    Args:
        lamb (int): lambda for hpfilter
    """

    def __init__(self, part: str = "trend", lamb: float = 1600, **kwargs):
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
    def get_new_params(method: str = "random"):
        part = random.choices(["trend", "cycle"], weights=[0.98, 0.02])[0]
        lamb = random.choices(
            [1600, 6.25, 129600, 104976000000, 4, 16, 62.5, 1049760000000],
            weights=[0.5, 0.2, 0.2, 0.1, 0.025, 0.025, 0.025, 0.025],
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
        self, decomp_type="STL", part: str = "trend", seasonal: int = 7, **kwargs
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

        if df.index.freq is None:
            freq = infer_frequency(df)
            df = df.asfreq(freq).ffill()

        def _stl_one_return(series, decomp_type="STL", seasonal=7, part="trend"):
            """Convert filter to apply on pd DataFrame."""
            if str(decomp_type).lower() == "stl":
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
        return df.ffill().bfill()

    @staticmethod
    def get_new_params(method: str = "random"):
        decomp_type = random.choices(["STL", "seasonal_decompose"], weights=[0.5, 0.5])[
            0
        ]
        part = random.choices(
            ["trend", "seasonal", "resid"], weights=[0.98, 0.02, 0.001]
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

    def __init__(self, n_jobs=1, method='lm', **kwargs):
        super().__init__(name="SinTrend")
        self.n_jobs = n_jobs
        self.method = method

    @staticmethod
    def get_new_params(method: str = "random"):
        return {
            'method': random.choices(['lm', 'trf', 'dogbox'], [0.95, 0.025, 0.025])[0]
        }

    @staticmethod
    def fit_sin(tt, yy, method="lm"):
        """Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"

        from user unsym @ https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
        """

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

        popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess, maxfev=10000, method=method)
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

        X = pd.to_numeric(df.index, errors="coerce", downcast="integer").values
        # make this faster (250 columns in 2.5 seconds isn't bad, though)
        cols = df.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 100:
            parallel = False
        # joblib multiprocessing to loop through series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if parallel:
                df_list = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.fit_sin)(X, df[col].to_numpy(), method=self.method)
                    for col in cols
                )
            else:
                df_list = []
                for col in cols:
                    df_list.append(
                        self.fit_sin(X, df[col].to_numpy(), method=self.method)
                    )
        self.sin_params = pd.DataFrame(df_list)

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
        X = pd.to_numeric(df.index, errors="coerce", downcast="integer").to_numpy()

        """
        sin_df = []
        for index, row in self.sin_params.iterrows():
            sin_df.append(
                pd.DataFrame(
                    row["amp"] * np.sin(row["omega"] * X + row["phase"])
                    + row["offset"],
                    columns=[index],
                )
            )
        sin_df = pd.concat(sin_df, axis=1)
        """
        X = np.repeat(X[..., np.newaxis], df.shape[1], axis=1)
        sin_df = pd.DataFrame(
            self.sin_params['amp'].to_frame().to_numpy().T
            * np.sin(
                self.sin_params['omega'].to_frame().to_numpy().T * X
                + self.sin_params['phase'].to_frame().to_numpy().T
            ),
            columns=df.columns,
        )
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
        X = pd.to_numeric(df.index, errors="coerce", downcast="integer").to_numpy()
        X = np.repeat(X[..., np.newaxis], df.shape[1], axis=1)
        sin_df = pd.DataFrame(
            self.sin_params['amp'].to_frame().to_numpy().T
            * np.sin(
                self.sin_params['omega'].to_frame().to_numpy().T * X
                + self.sin_params['phase'].to_frame().to_numpy().T
            ),
            columns=df.columns,
        )

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
    def get_new_params(method: str = "random"):
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
        fixed (bool): if True, don't inverse to volatile state
        macro_micro (bool): if True, split on rolling trend vs remainder and later recombine. Overrides fixed arg.
    """

    def __init__(
        self,
        window: int = 10,
        fixed: bool = False,
        macro_micro: bool = False,
        suffix: str = "_lltmicro",
        center: bool = False,
        **kwargs,
    ):
        super().__init__(name="RollingMeanTransformer")
        self.window = window
        self.fixed = fixed
        self.macro_micro = macro_micro
        self.suffix = suffix
        self.center = center

    @staticmethod
    def get_new_params(method: str = "random"):
        bool_c = random.choices([True, False], [0.7, 0.3])[0]
        center = random.choices([True, False], [0.5, 0.5])[0]
        macro_micro = random.choices([True, False], [0.2, 0.8])[0]
        if macro_micro:
            choice = random.choice([3, 7, 10, 12, 28, 90, 180, 360])
        elif method == "fast":
            choice = random.choice([3, 7, 10, 12, 90])
        else:
            choice = seasonal_int(include_one=False)
        return {
            "fixed": bool_c,
            "window": choice,
            "macro_micro": macro_micro,
            "center": center,
        }

    def fit(self, df):
        """Fits.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.shape = df.shape
        self.last_values = df.tail(self.window).ffill().bfill()
        self.first_values = df.head(self.window).ffill().bfill()

        df = (
            df.tail(self.window + 1)
            .rolling(window=self.window, min_periods=1, center=self.center)
            .mean()
        )
        self.last_rolling = df.tail(1)
        return self

    def transform(self, df):
        """Returns rolling data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        macro = df.rolling(window=self.window, min_periods=1, center=self.center).mean()
        if self.macro_micro:
            self.columns = df.columns
            micro = (df - macro).rename(columns=lambda x: str(x) + self.suffix)
            return pd.concat([macro, micro], axis=1)
        else:
            return macro

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
        if self.macro_micro:
            macro = df[self.columns]
            micro = df[df.columns.difference(self.columns)]
            micro = micro.rename(columns=lambda x: str(x)[: -len(self.suffix)])[
                self.columns
            ]
            return macro + micro
        elif self.fixed:
            return df
        else:
            window = self.window
            if trans_method == "original":
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
            elif trans_method == "forecast":
                staged = self.last_values
                # these are all rolling values at first (forecast rolling and historic)
                df = pd.concat([self.last_rolling, df], axis=0).astype(float)
                diffed = (df - df.shift(1)) * window
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
                return staged.tail(len(diffed.index))


class SeasonalDifference(EmptyTransformer):
    """Remove seasonal component.

    "Oh that's nice - ash on my tomatoes!" - Pippin

    Args:
        lag_1 (int): length of seasonal period to remove.
        method (str): 'LastValue', 'Mean', 'Median' to construct seasonality
    """

    def __init__(self, lag_1: int = 7, method: str = "LastValue", **kwargs):
        super().__init__(name="SeasonalDifference")
        self.lag_1 = int(abs(lag_1))
        self.method = method

    @staticmethod
    def get_new_params(method: str = "random"):
        method_c = random.choices(
            ["LastValue", "Mean", "Median", 2, 5, 20], [0.5, 0.2, 0.2, 0.1, 0.1, 0.1]
        )[0]
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

        if self.method in ["Mean", "Median"]:
            df2 = df.copy()
            tile_index = np.tile(
                np.arange(self.lag_1), int(np.ceil(df_length / self.lag_1))
            )
            tile_index = tile_index[len(tile_index) - (df_length) :]
            df2.index = tile_index
            if self.method == "Median":
                self.tile_values_lag_1 = df2.groupby(level=0).median()
            else:
                self.tile_values_lag_1 = df2.groupby(level=0).mean()
        elif isinstance(self.method, (int, float)):
            # do an exponential weighting with given span
            arr = np.asarray(df)
            N = self.lag_1
            span = self.method
            num_slices = arr.shape[0] // N
            # (slices, lag size, num seriesd)
            arr_3d = arr[-(num_slices * N) :].reshape(num_slices, N, -1)
            alpha = 2 / (span + 1)
            weights = (alpha * (1 - alpha) ** np.arange(num_slices))[::-1]
            self.tile_values_lag_1 = np.sum(
                arr_3d * weights[:, np.newaxis, np.newaxis], axis=0
            ) / np.sum(weights)
        elif self.method in ['lastvalue', 'LastValue', 'last_value']:
            self.tile_values_lag_1 = df.tail(self.lag_1)
        else:
            raise ValueError(
                f"SeasonalDifference method '{self.method}' not recognized"
            )
        return self

    def transform(self, df):
        """Returns rolling data
        Args:
            df (pandas.DataFrame): input dataframe
        """
        tile_len = self.tile_values_lag_1.shape[0]  # self.lag_1
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
        tile_len = self.tile_values_lag_1.shape[0]
        df_len = df.shape[0]
        sdf = pd.DataFrame(
            np.tile(self.tile_values_lag_1, (int(np.ceil(df_len / tile_len)), 1))
        )
        if trans_method == "original":
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
            "model": "DecisionTree",
            "model_params": {"max_depth": 5, "min_samples_split": 2},
        },
        datepart_method: str = "expanded",
        polynomial_degree: int = None,
        transform_dict: dict = None,
        holiday_country: list = None,
        holiday_countries_used: bool = False,
        lags: int = None,
        forward_lags: int = None,
        n_jobs: int = 1,
        **kwargs,
    ):
        super().__init__(name="DatepartRegressionTransformer")
        self.regression_model = regression_model
        self.datepart_method = datepart_method
        self.polynomial_degree = polynomial_degree
        self.transform_dict = transform_dict
        self.holiday_country = holiday_country
        self.holiday_countries_used = holiday_countries_used
        self.lags = lags
        self.forward_lags = forward_lags
        self.n_jobs = n_jobs

    @staticmethod
    def get_new_params(method: str = "random", holiday_countries_used=None):
        datepart_choice = random_datepart(method=method)
        if datepart_choice in ["simple", "simple_2", "recurring"]:
            polynomial_choice = random.choices([None, 2], [0.5, 0.2])[0]
        else:
            polynomial_choice = None

        if method in ["all", "deep"]:
            choice = generate_regressor_params()
        elif method == "fast":
            choice = generate_regressor_params(
                model_dict={
                    "ElasticNet": 0.5,
                    "DecisionTree": 0.25,
                    # "KNN": 0.002,  # simply uses too much memory at scale
                }
            )
        else:
            choice = generate_regressor_params(
                model_dict={
                    "ElasticNet": 0.25,
                    "DecisionTree": 0.25,
                    "KNN": 0.1,
                    "MLP": 0.2,
                    "RandomForest": 0.2,
                    "ExtraTrees": 0.25,
                    "SVM": 0.1,
                    "RadiusRegressor": 0.1,
                    'MultioutputGPR': 0.0001,
                    "ElasticNetwork": 0.05,
                }
            )
        if holiday_countries_used is None:
            holiday_countries_used = random.choices([True, False], [0.3, 0.7])[0]

        return {
            "regression_model": choice,
            "datepart_method": datepart_choice,
            "polynomial_degree": polynomial_choice,
            "transform_dict": random_cleaners(),
            "holiday_countries_used": holiday_countries_used,
            'lags': random.choices([None, 1, 2, 3, 4], [0.9, 0.05, 0.05, 0.02, 0.02])[
                0
            ],
            'forward_lags': random.choices(
                [None, 1, 2, 3, 4], [0.9, 0.05, 0.05, 0.02, 0.02]
            )[0],
        }

    def fit(self, df, regressor=None):
        """Fits trend for later detrending.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        df_local = df.copy()
        try:
            df_local = df_local.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")
        # handle NaN
        nan_mask = np.isnan(df_local)
        nan_mask_all = nan_mask.all(axis=1)
        nan_mask_any = nan_mask.any(axis=1)
        sum_nan_all = np.sum(nan_mask_all)
        self.full_nan_rows = sum_nan_all > 0
        self.partial_nan_rows = np.sum(nan_mask_any) > sum_nan_all
        if self.full_nan_rows and not self.partial_nan_rows:
            # this is the simplest if just all NaN rows, drop
            df_local = df_local[~nan_mask_all]

        if self.transform_dict is not None:
            model = GeneralTransformer(**self.transform_dict)
            y = model.fit_transform(df_local)
        else:
            y = df_local.to_numpy()
        if y.shape[1] == 1:
            y = np.asarray(y).ravel()
        X = date_part(
            df_local.index,
            method=self.datepart_method,
            polynomial_degree=self.polynomial_degree,
            lags=self.lags,
            forward_lags=self.forward_lags,
        )
        if self.holiday_country is not None and self.holiday_countries_used:
            X = pd.concat(
                [
                    X,
                    holiday_flag(
                        df_local.index,
                        country=self.holiday_country,
                        encode_holiday_type=True,
                    ),
                ],
                axis=1,
            )
        if regressor is not None:
            X = pd.concat([X, regressor], axis=1)
        multioutput = True
        if self.partial_nan_rows:
            # process into a single output approach
            X = np.repeat(X.to_numpy(), y.shape[1], axis=0)
            # add a (hopefully not too massive) series ID encoder. This will not scale well to large multivariate
            # X = np.concatenate([X, np.repeat(np.eye(y.shape[1]), df_local.shape[0], axis=0)], axis=1)
            X = np.concatenate(
                [X, np.tile(np.eye(y.shape[1]), df_local.shape[0]).T], axis=1
            )
            y = np.asarray(y).flatten()
            y_mask = np.isnan(y)
            y = y[~y_mask]
            X = X[~y_mask]
            multioutput = False
        else:
            if y.ndim < 2:
                multioutput = False
            elif y.shape[1] < 2:
                multioutput = False
        self.X = X  # diagnostic
        self.model = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=0,
            verbose_bool=False,
            random_seed=2020,
            multioutput=multioutput,
            n_jobs=self.n_jobs,
        )
        self.model = self.model.fit(
            X.fillna(0) if isinstance(X, pd.DataFrame) else np.nan_to_num(X),
            y.fillna(0) if isinstance(y, pd.DataFrame) else np.nan_to_num(y),
        )
        self.shape = df_local.shape
        return self

    def impute(self, df, regressor=None):
        """Fill Missing. Needs to have same general pattern of missingness (full rows of NaN only or scattered NaN) as was present during .fit()"""
        X = date_part(
            df.index,
            method=self.datepart_method,
            polynomial_degree=self.polynomial_degree,
            lags=self.lags,
            forward_lags=self.forward_lags,
        )
        if self.holiday_country is not None and self.holiday_countries_used:
            X = pd.concat(
                [
                    X,
                    holiday_flag(
                        df.index, country=self.holiday_country, encode_holiday_type=True
                    ),
                ],
                axis=1,
            )
        if regressor is not None:
            X = pd.concat([X, regressor], axis=1)
        if self.partial_nan_rows:
            # process into a single output approach
            X = np.repeat(X.to_numpy(), self.shape[1], axis=0)
            X = np.concatenate(
                [X, np.tile(np.eye(self.shape[1]), df.shape[0]).T], axis=1
            )

        pred = self.model.predict(X)
        if self.partial_nan_rows:
            pred = pred.reshape(-1, df.shape[1])
        y = pd.DataFrame(pred, columns=df.columns, index=df.index)
        # np.where(np.isnan(df), y, df)
        return df.where(~df.isnull(), y)

    def fit_transform(self, df, regressor=None):
        """Fit and Return Detrended DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df, regressor=regressor)
        return self.transform(df, regressor=regressor)

    def transform(self, df, regressor=None):
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
            lags=self.lags,
            forward_lags=self.forward_lags,
        )
        if self.holiday_country is not None and self.holiday_countries_used:
            X = pd.concat(
                [
                    X,
                    holiday_flag(
                        df.index, country=self.holiday_country, encode_holiday_type=True
                    ),
                ],
                axis=1,
            )
        if regressor is not None:
            X = pd.concat([X, regressor], axis=1)
        if self.partial_nan_rows:
            # process into a single output approach
            X = np.repeat(X.to_numpy(), self.shape[1], axis=0)
            X = np.concatenate(
                [X, np.tile(np.eye(self.shape[1]), df.shape[0]).T], axis=1
            )
        # X.columns = [str(xc) for xc in X.columns]
        pred = self.model.predict(X)
        if self.partial_nan_rows:
            pred = pred.reshape(-1, df.shape[1])
        y = pd.DataFrame(pred, columns=df.columns, index=df.index)
        df = df - y
        return df

    def inverse_transform(self, df, regressor=None):
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
            lags=self.lags,
            forward_lags=self.forward_lags,
        )
        if self.holiday_country is not None and self.holiday_countries_used:
            X = pd.concat(
                [
                    X,
                    holiday_flag(
                        df.index, country=self.holiday_country, encode_holiday_type=True
                    ),
                ],
                axis=1,
            )
        if regressor is not None:
            X = pd.concat([X, regressor], axis=1)
        if self.partial_nan_rows:
            # process into a single output approach
            X = np.repeat(X.to_numpy(), self.shape[1], axis=0)
            X = np.concatenate(
                [X, np.tile(np.eye(self.shape[1]), df.shape[0]).T], axis=1
            )
        pred = self.model.predict(X)
        if self.partial_nan_rows:
            pred = pred.reshape(-1, df.shape[1])
        y = pd.DataFrame(pred, columns=df.columns, index=df.index)
        df = df + y
        return df


DatepartRegression = DatepartRegressionTransformer


class DifferencedTransformer:
    """Difference from lag n value.
    inverse_transform can only be applied to the original series, or an immediately following forecast.

    Args:
        lag (int): number of periods to shift.
        fill (str): method to fill NaN values created by differencing, options: 'bfill', 'zero'.
    """

    def __init__(self, lag=1, fill='bfill'):
        self.name = "DifferencedTransformer"
        self.lag = lag
        self.fill = fill
        self.last_values = None
        self.first_values = None

    @staticmethod
    def get_new_params(method: str = "random"):
        method_c = random.choices(["bfill", "zero", "one"], [0.5, 0.2, 0.01])[0]
        choice = random.choices([1, 2, 7], [0.8, 0.1, 0.1])[0]
        return {"lag": choice, "fill": method_c}

    def fit(self, df):
        """Fit.
        Args:
            df (pandas.DataFrame): input dataframe.
        """
        self.last_values = df.iloc[-self.lag :]
        self.first_values = df.iloc[: self.lag]
        return self

    def transform(self, df):
        """Return differenced data.

        Args:
            df (pandas.DataFrame): input dataframe.
        """
        differenced = df.diff(self.lag)
        if self.fill == 'bfill':
            return differenced.bfill()
        elif self.fill == 'zero':
            return differenced.fillna(0)
        elif self.fill == 'one':
            return differenced.fillna(1)
        else:
            raise ValueError(
                f"DifferencedTransformer fill method {self.fill} not recognized"
            )

    def fit_transform(self, df):
        """Fits and returns differenced DataFrame.
        Args:
            df (pandas.DataFrame): input dataframe.
        """
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df, trans_method="forecast"):
        """Returns data to original *or* forecast form

        Args:
            df (pandas.DataFrame): input dataframe.
            trans_method (str): whether to inverse on original data, or on a following sequence
                - 'original' return original data to original numbers
                - 'forecast' inverse the transform on a dataset immediately following the original.
        """
        if trans_method == "original":
            df_with_first = pd.concat(
                [self.first_values, df.tail(df.shape[0] - self.lag)]
            )
            return df_with_first.cumsum()
        elif trans_method == "forecast":
            df_len = df.shape[0]
            df_with_last = pd.concat([self.last_values, df])
            return df_with_last.cumsum().tail(df_len)
        else:
            raise ValueError("Invalid transformation method specified.")


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
        df = df.pct_change(periods=1).fillna(0)
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
        if trans_method == "original":
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
        return df.cumsum(skipna=True)

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

        if trans_method == "original":
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
    def get_new_params(method: str = "random"):
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
            [1, 2, 3, 3.5, 4, 4.5, 5], [0.1, 0.2, 0.2, 0.2, 0.4, 0.1, 0.1], k=1
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
            # splitting it is a bit more memory efficienty if slightly slower
            lower = self.df_mean - (self.df_std * self.std_threshold)
            df2 = df.where(df >= lower, lower, axis=1)
            upper = self.df_mean + (self.df_std * self.std_threshold)
            df2 = df2.where(df2 <= upper, upper, axis=1)
            # df2 = df.clip(lower=lower, upper=upper, axis=1)

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
    def get_new_params(method: str = "random"):
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
    def get_new_params(method: str = "random"):
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
    def get_new_params(method: str = "random"):
        if method == "fast":
            choice = random.choice(["center", "upper", "lower"])
            n_bin_c = random.choice([5, 10, 20])
        else:
            choice = random.choices(
                [
                    "center",  # more memory intensive
                    "upper",
                    "lower",
                    "sklearn-quantile",
                    "sklearn-uniform",
                    "sklearn-kmeans",
                ],
                [0.1, 0.3, 0.3, 0.1, 0.1, 0.1],
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
        if self.discretization not in [None, "None"]:
            self.df_index = df.index
            self.df_colnames = df.columns
            if self.discretization in [
                "sklearn-quantile",
                "sklearn-uniform",
                "sklearn-kmeans",
            ]:
                from sklearn.preprocessing import KBinsDiscretizer

                self.kbins_discretizer = KBinsDiscretizer(
                    n_bins=self.n_bins,
                    encode="ordinal",
                    strategy=self.discretization.split("-")[1],
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
                if self.discretization == "center":
                    bins = np.cumsum(bins, dtype=float, axis=0)
                    bins[2:] = bins[2:] - bins[:-2]
                    bins = bins[2 - 1 :] / 2
                    binned = (np.abs(df.to_numpy() - bins)).argmin(axis=0)
                elif self.discretization == "lower":
                    bins = np.delete(bins, (-1), axis=0)
                    binned = (
                        np.sum(
                            df.to_numpy()[:, np.newaxis, :] < bins[:, np.newaxis, :],
                            axis=0,
                        )[:, 0, :]
                        - 1
                    )
                elif self.discretization == "upper":
                    bins = np.delete(bins, (0), axis=0)
                    binned = np.sum(
                        df.to_numpy()[:, np.newaxis, :] <= bins[:, np.newaxis, :],
                        axis=0,
                    )[:, 0, :]
                self.bins = bins
                # binned = (np.abs(df.to_numpy() - self.bins)).argmin(axis=0)
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
        if self.discretization not in [None, "None"]:
            if self.discretization in [
                "sklearn-quantile",
                "sklearn-uniform",
                "sklearn-kmeans",
            ]:
                df2 = pd.DataFrame(self.kbins_discretizer.transform(df))
                df2.index = df.index
                df2.columns = self.df_colnames
            else:
                binned = (np.abs(df.to_numpy() - self.bins)).argmin(axis=0)
                indices = np.indices(binned.shape)[1]
                bins_reshaped = self.bins.reshape((self.n_bins, df.shape[1]))
                df2 = pd.DataFrame(
                    bins_reshaped[binned, indices],
                    index=df.index,
                    columns=self.df_colnames,
                )
        return df2

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """

        if self.discretization in [
            "sklearn-quantile",
            "sklearn-uniform",
            "sklearn-kmeans",
        ]:
            df_index = df.index
            df_colnames = df.columns
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
    def get_new_params(method: str = "random"):
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


class ShiftFirstValue(EmptyTransformer):
    """Shift all data relative to the first value(s) of the series.

    Args:
        rows (int): number of rows to average from beginning of data
    """

    def __init__(self, rows: int = 1, **kwargs):
        super().__init__(name="ShiftFirstValue")
        self.rows = rows

    @staticmethod
    def get_new_params(method: str = "random"):
        choice = random.choices([1, 2, 7, 28], [0.2, 0.2, 0.2, 0.2])[0]
        return {
            "rows": choice,
        }

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.center = df.bfill().head(self.rows).mean()
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = df - self.center
        return df

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        df = df + self.center
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

    def __init__(self, method: str = "hilbert", method_args: list = None, **kwargs):
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
    def get_new_params(method: str = "random"):
        if method == "fast":
            method = random.choice(['butter', 'savgol_filter'])
            polyorder = random.choices([1, 2, 3, 4], [0.5, 0.5, 0.25, 0.2])[0]
        else:
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
                [0.1, 0.1, 0.9, 0.9],
                k=1,
            )[0]
            polyorder = random.choice([1, 2, 3, 4])
        # analog_choice = bool(random.randint(0, 1))
        analog_choice = False
        xn = random.randint(1, 99)
        btype = random.choice(["lowpass", "highpass"])  # "bandpass", "bandstop"
        if method in ["wiener", "hilbert"]:
            method_args = None
        elif method == "savgol_filter":
            method_args = {
                'window_length': random.choices([7, 31, 91], [0.4, 0.3, 0.3])[0],
                'polyorder': polyorder,
                'deriv': random.choices([0, 1], [0.8, 0.2])[0],
                'mode': random.choice(['mirror', 'nearest', 'interp']),
            }
        elif method in ["butter"]:
            if btype in ["bandpass", "bandstop"]:
                Wn = [xn / 100, random.randint(1, 99) / 100]
            else:
                Wn = xn / 100 if not analog_choice else xn
            method_args = {
                'N': random.randint(1, 8),
                # 'Wn': Wn,  # cutoff
                'window_size': seasonal_int(small=True),
                'btype': btype,
                'analog': analog_choice,
                'output': 'sos',
            }
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

        if self.method == "hilbert":
            from scipy.signal import hilbert

            test = pd.DataFrame(hilbert(df.values), columns=df.columns, index=df.index)
            return np.abs(test)
        elif self.method == "wiener":
            from scipy.signal import wiener

            return pd.DataFrame(wiener(df.values), columns=df.columns, index=df.index)
        elif self.method == "savgol_filter":
            # args = [5, 2]
            return pd.DataFrame(
                savgol_filter(df.ffill().bfill().values, **self.method_args, axis=0),
                columns=df.columns,
                index=df.index,
            )
        elif self.method == "butter":
            # args = [4, 0.125]
            # [4, 100, 'lowpass'], [1, 0.125, "highpass"]
            if 'window_size' in self.method_args:
                self.method_args['Wn'] = 1 / self.method_args.pop("window_size")
            sos = butter(**self.method_args)
            return pd.DataFrame(
                sosfiltfilt(sos, df.values, axis=0), columns=df.columns, index=df.index
            )
        elif self.method == "cheby1":
            from scipy.signal import cheby1

            # args = [4, 5, 100, 'lowpass', True]
            # args = [10, 1, 15, 'highpass']
            sos = cheby1(*self.method_args, output="sos")
            return pd.DataFrame(
                sosfiltfilt(sos, df.values, axis=0), columns=df.columns, index=df.index
            )
        elif self.method == "cheby2":
            from scipy.signal import cheby2

            sos = cheby2(*self.method_args, output="sos")
            return pd.DataFrame(
                sosfiltfilt(sos, df.values, axis=0), columns=df.columns, index=df.index
            )
        elif self.method == "ellip":
            from scipy.signal import ellip

            sos = ellip(*self.method_args, output="sos")
            return pd.DataFrame(
                sosfiltfilt(sos, df.values, axis=0), columns=df.columns, index=df.index
            )
        elif self.method == "bessel":
            from scipy.signal import bessel

            # args = [4, 100, 'lowpass', True]
            # args = [3, 10, 'highpass']
            sos = bessel(*self.method_args, output="sos")
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
    def get_new_params(method: str = "random"):
        if method == "fast":
            choice = random.choice([3, 7, 10, 12, 28])
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
    def get_new_params(method: str = "random"):
        return {
            "algorithm": random.choice(["parallel", "deflation"]),
            "fun": random.choice(["logcosh", "exp", "cube"]),
            "max_iter": random.choices([100, 250, 500], [0.2, 0.7, 0.1])[0],
            "whiten": random.choices(
                ['unit-variance', 'arbitrary-variance', False], [0.9, 0.1, 0.1]
            )[0],
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
        if isinstance(return_df, pd.DataFrame):
            if return_df.shape[1] == len(self.columns):
                return_df.columns = self.columns
            return return_df
        else:
            if return_df.shape[1] == len(self.columns):
                return pd.DataFrame(return_df, index=self.index, columns=self.columns)
            else:
                return pd.DataFrame(return_df, index=self.index).rename(
                    columns=lambda x: "pca_" + str(x)
                )

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
        if isinstance(return_df, pd.DataFrame):
            if return_df.shape[1] == len(self.columns):
                return_df.columns = self.columns
            return return_df
        else:
            if return_df.shape[1] == len(self.columns):
                return pd.DataFrame(return_df, index=self.index, columns=self.columns)
            else:
                return pd.DataFrame(return_df, index=df.index).rename(
                    columns=lambda x: "pca_" + str(x)
                )

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return_df = self.transformer.inverse_transform(df)
        if isinstance(return_df, pd.DataFrame):
            return_df.columns = self.columns
            return return_df
        else:
            return pd.DataFrame(return_df, index=df.index, columns=self.columns)

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        return {
            "whiten": random.choices([True, False], [0.2, 0.8])[0],
            "n_components": random.choices(
                [None, 4, 10, 24, 100, 0.3], [0.8, 0.05, 0.1, 0.05, 0.1, 0.05]
            )[0],
        }


class MeanDifference(EmptyTransformer):
    """Difference from lag n value, but differenced by mean of all series.
    inverse_transform can only be applied to the original series, or an immediately following forecast

    Args:
        lag (int): number of periods to shift (not implemented, default = 1)
    """

    def __init__(self, **kwargs):
        super().__init__(name="MeanDifference")
        self.lag = 1
        self.beta = 1

    def fit(self, df):
        """Fit.
        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.means = df.mean(axis=1)

        self.last_values = self.means.iloc[-self.lag]
        self.first_values = self.means.iloc[self.lag - 1]
        return self

    def transform(self, df):
        """Return differenced data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return (df - df.mean(axis=1).shift(self.lag).values[..., None]).fillna(
            method="bfill"
        )

    def fit_transform(self, df):
        """Fits and Returns Magical DataFrame
        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return (df - self.means.shift(self.lag).values[..., None]).bfill()

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
        if trans_method == "original":
            return df + self.means.shift(lag).values[..., None]
        else:
            last_vals = self.last_values
            for indx in range(df.shape[0]):
                df.iloc[indx] = df.iloc[indx] + last_vals
                last_vals = df.iloc[indx].mean()
            if df.isnull().values.any():
                raise ValueError("NaN in MeanDifference.inverse_transform")
            return df


class Cointegration(EmptyTransformer):
    """Johansen Cointegration Decomposition."""

    def __init__(
        self,
        det_order: int = -1,
        k_ar_diff: int = 1,
        name: str = "Cointegration",
        **kwargs,
    ):
        self.name = name
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if df.shape[1] < 2:
            raise ValueError("Coint only works on multivarate series")
        # might be helpful to add a fast test for correlation?
        self.components_ = coint_johansen(df.values, self.det_order, self.k_ar_diff)
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return pd.DataFrame(
            np.matmul(self.components_, (df.values).T).T,
            index=df.index,
            columns=df.columns,
        )

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original space.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return pd.DataFrame(
            # np.dot(np.linalg.pinv(df), self.components_),
            np.linalg.lstsq(self.components_, df.T, rcond=1)[0].T,
            # np.linalg.solve(self.components_, df.T).T,
            index=df.index,
            columns=df.columns,
        ).astype(float)

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self.fit(df).transform(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        return {
            "det_order": random.choice([-1, 0, 1]),
            "k_ar_diff": random.choice([0, 1, 2]),
        }


class BTCD(EmptyTransformer):
    """Box and Tiao Canonical Decomposition."""

    def __init__(
        self,
        regression_model: dict = {
            "model": "LinearRegression",
            "model_params": {},
        },
        max_lags: int = 1,
        name: str = "BTCD",
        **kwargs,
    ):
        self.name = name
        self.regression_model = regression_model
        self.max_lags = max_lags

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if df.shape[1] < 2:
            raise ValueError("BTCD only works on multivarate series")

        self.components_ = btcd_decompose(
            df.values,
            retrieve_regressor(
                regression_model=self.regression_model,
                verbose=0,
                verbose_bool=False,
                random_seed=2020,
                multioutput=False,
            ),
            self.max_lags,
        )
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return pd.DataFrame(
            np.matmul(self.components_, (df.values).T).T,
            index=df.index,
            columns=df.columns,
        ).astype(float)

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original space.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return pd.DataFrame(
            # np.linalg.lstsq(self.components_, df.T)[0].T,
            np.linalg.solve(self.components_, df.T).T,
            index=df.index,
            columns=df.columns,
        ).astype(float)

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self.fit(df).transform(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        if method == "deep":
            choice = generate_regressor_params(
                model_dict={
                    "ElasticNet": 0.05,
                    "DecisionTree": 0.05,
                    "LinearRegression": 0.9,
                }
            )
        else:
            choice1 = random.choice(["LinearRegression", "FastRidge"])
            choice = {
                "model": choice1,
                "model_params": {},
            }
        return {"regression_model": choice, "max_lags": random.choice([1, 2])}


class AlignLastValue(EmptyTransformer):
    """Shift all data relative to the last value(s) of the series.

    Args:
        rows (int): number of rows to average as last record
        lag (int): use last value as this lag back, 1 is no shift, 2 is lag one from end, ie second to last
        method (str): 'additive', 'multiplicative'
        strength (float): softening parameter [0, 1], 1.0 for full difference
    """

    def __init__(
        self,
        rows: int = 1,
        lag: int = 1,
        method: str = "additive",
        strength: float = 1.0,
        first_value_only: bool = False,
        threshold: int = None,
        threshold_method: str = "max",
        **kwargs,
    ):
        super().__init__(name="AlignLastValue")
        self.rows = rows
        self.lag = lag
        self.method = method
        self.strength = strength
        self.first_value_only = first_value_only
        self.adjustment = None
        self.threshold = threshold
        self.threshold_method = threshold_method

    @staticmethod
    def get_new_params(method: str = "random"):
        return {
            "rows": random.choices(
                [1, 2, 4, 7, 24, 84, 168], [0.83, 0.02, 0.05, 0.1, 0.01, 0.05, 0.05]
            )[0],
            "lag": random.choices(
                [1, 2, 7, 28, 84, 168], [0.8, 0.05, 0.1, 0.05, 0.05, 0.01]
            )[0],
            "method": random.choices(["additive", "multiplicative"], [0.9, 0.1])[0],
            "strength": random.choices(
                [1.0, 0.9, 0.7, 0.5, 0.2], [0.8, 0.05, 0.05, 0.05, 0.05]
            )[0],
            'first_value_only': random.choices([True, False], [0.1, 0.9])[0],
            "threshold": random.choices([None, 1, 3, 10], [0.8, 0.9, 0.2, 0.9])[0],
            "threshold_method": random.choices(['max', 'mean'], [0.5, 0.5])[0],
        }

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.rows is None:
            self.rows = df.shape[0]
        # fill NaN if present (up to a limit for slight speedup)
        if np.isnan(np.sum(np.array(df)[-50:])):
            self.center = self.find_centerpoint(df.ffill(axis=0), self.rows, self.lag)
        else:
            self.center = self.find_centerpoint(df, self.rows, self.lag)
        if self.threshold is not None:
            if self.method == "multiplicative":
                if self.threshold_method == "max":
                    self.threshold = df.iloc[-self.threshold :].pct_change().abs().max()
                else:
                    self.threshold = df.pct_change().abs().mean() * self.threshold
            else:
                if self.threshold_method == "max":
                    self.threshold = df.iloc[-self.threshold :].diff().abs().max()
                else:
                    self.threshold = df.diff().abs().mean() * self.threshold
        return self

    @staticmethod
    def find_centerpoint(df, rows, lag):
        if rows <= 1:
            if lag > 1:
                center = df.iloc[-lag, :]
            else:
                center = df.iloc[-1, :]
        else:
            if lag > 1:
                center = df.iloc[-(lag + rows - 1) : -(lag - 1), :].mean()
            else:
                center = df.tail(rows).mean()
        return center

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def inverse_transform(self, df, trans_method: str = "forecast", adjustment=None):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
            adjustment (float): size of shift, utilized for adjusting the upper and lower bounds to match point forecast
        """
        if self.adjustment is not None:
            self.adjustment = adjustment
        if trans_method == "original":
            return df
        else:
            if self.first_value_only:
                if self.method == "multiplicative":
                    if self.adjustment is None:
                        self.adjustment = (
                            1
                            + ((self.center / df.iloc[0].replace(0, 1)) - 1)
                            * self.strength
                        )
                    return pd.concat(
                        [
                            df.iloc[0:1] * self.adjustment,
                            df.iloc[1:],
                        ],
                        axis=0,
                    )
                else:
                    if self.adjustment is None:
                        self.adjustment = self.strength * (self.center - df.iloc[0])
                    return pd.concat(
                        [
                            df.iloc[0:1] + self.adjustment,
                            df.iloc[1:],
                        ],
                        axis=0,
                    )
            else:
                if self.method == "multiplicative":
                    if self.adjustment is None:
                        self.adjustment = (
                            1
                            + ((self.center / df.iloc[0].replace(0, 1)) - 1)
                            * self.strength
                        )
                    if self.threshold is not None:
                        return df.where(
                            self.adjustment.abs() <= self.threshold,
                            df * self.adjustment,
                        )
                    else:
                        return df * self.adjustment
                else:
                    if self.adjustment is None:
                        self.adjustment = self.strength * (self.center - df.iloc[0])
                    if self.threshold is not None:
                        return df.where(
                            self.adjustment.abs() <= self.threshold,
                            df + self.adjustment,
                        )
                    else:
                        return df + self.adjustment

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)


class AnomalyRemoval(EmptyTransformer):
    def __init__(
        self,
        output="multivariate",
        method="zscore",
        transform_dict={  # also  suggest DifferencedTransformer
            "transformations": {0: "DatepartRegression"},
            "transformation_params": {
                0: {
                    "datepart_method": "simple_3",
                    "regression_model": {
                        "model": "ElasticNet",
                        "model_params": {},
                    },
                }
            },
        },
        method_params={},
        fillna=None,
        isolated_only=False,
        on_inverse=False,
        n_jobs=1,
    ):
        """Detect anomalies on a historic dataset. No inverse_transform available.

        Args:
            output (str): 'multivariate' (each series unique outliers), or 'univariate' (all series together for one outlier flag per timestamp)
            method (str): method choosen, from sklearn, AutoTS, and basic stats. Use `.get_new_params()` to see potential models
            transform_dict (dict): option but helpful, often datepart, differencing, or other standard AutoTS transformer params
            method_params (dict): parameters specific to the method, use `.get_new_params()` to see potential models
            fillna (str): how to fill anomaly values removed
            isolated_only (bool): if True, only removal standalone anomalies
            n_jobs (int): multiprocessing jobs, used by some methods
        """
        super().__init__(name="AnomalyRemoval")
        self.output = output
        self.method = method
        self.transform_dict = transform_dict
        self.method_params = method_params
        self.n_jobs = n_jobs
        self.fillna = fillna
        self.isolated_only = isolated_only
        self.anomaly_classifier = None
        self.on_inverse = False

    def fit(self, df):
        """All will return -1 for anomalies.

        Args:
            df (pd.DataFrame): pandas wide-style data
        Returns:
            pd.DataFrame (classifications, -1 = outlier, 1 = not outlier), pd.DataFrame s(scores)
        """
        self.df_anomaly = df.copy()
        if self.transform_dict is not None:
            model = GeneralTransformer(**self.transform_dict)
            self.df_anomaly = model.fit_transform(self.df_anomaly)

        self.anomalies, self.scores = detect_anomalies(
            self.df_anomaly,
            output=self.output,
            method=self.method,
            transform_dict=self.transform_dict,
            method_params=self.method_params,
            n_jobs=self.n_jobs,
        )
        if self.isolated_only:
            # replace all anomalies (-1) except those which are isolated (1 before and after)
            mask_minus_one = self.anomalies == -1
            mask_prev_one = self.anomalies.shift(1) == 1
            mask_next_one = self.anomalies.shift(-1) == 1
            mask_replace = mask_minus_one & ~(mask_prev_one & mask_next_one)
            self.anomalies[mask_replace] = 1
        return self

    def transform(self, df):
        df2 = df[self.anomalies != -1]
        if self.fillna is not None:
            df2 = FillNA(df2, method=self.fillna, window=10)
        return df2

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def fit_anomaly_classifier(self):
        """Fit a model to predict if a score is an anomaly."""
        # Using DecisionTree as it should almost handle nonparametric anomalies
        from sklearn.tree import DecisionTreeClassifier

        scores_flat = self.scores.melt(var_name='series', value_name="value")
        categor = pd.Categorical(scores_flat['series'])
        self.score_categories = categor.categories
        scores_flat['series'] = categor
        scores_flat = pd.concat(
            [pd.get_dummies(scores_flat['series'], dtype=float), scores_flat['value']],
            axis=1,
        )
        anomalies_flat = self.anomalies.melt(var_name='series', value_name="value")
        self.anomaly_classifier = DecisionTreeClassifier(max_depth=None).fit(
            scores_flat, anomalies_flat['value']
        )
        # anomaly_classifier.score(scores_flat, anomalies_flat['value'])

    def score_to_anomaly(self, scores):
        """A DecisionTree model, used as models are nonstandard (and nonparametric)."""
        if self.anomaly_classifier is None:
            self.fit_anomaly_classifier()
        scores.index.name = 'date'
        scores_flat = scores.reset_index(drop=False).melt(
            id_vars="date", var_name='series', value_name="value"
        )
        scores_flat['series'] = pd.Categorical(
            scores_flat['series'], categories=self.score_categories
        )
        res = self.anomaly_classifier.predict(
            pd.concat(
                [
                    pd.get_dummies(scores_flat['series'], dtype=float),
                    scores_flat['value'],
                ],
                axis=1,
            )
        )
        res = pd.concat(
            [scores_flat[['date', "series"]], pd.Series(res, name='value')], axis=1
        ).pivot_table(index='date', columns='series', values="value")
        return res[scores.columns]

    def inverse_transform(self, df, trans_method: str = "forecast"):
        if self.on_inverse:
            return self.fit_transform(df)
        else:
            return df

    @staticmethod
    def get_new_params(method="random"):
        method_choice, method_params, transform_dict = anomaly_new_params(method=method)
        if transform_dict == "random":
            transform_dict = RandomTransform(
                transformer_list="scalable", transformer_max_depth=2
            )

        return {
            "method": method_choice,
            "method_params": method_params,
            "fillna": random.choices(
                [None, "ffill", "mean", "rolling_mean_24", "linear", "fake_date"],
                [0.01, 0.39, 0.1, 0.3, 0.15, 0.05],
            )[0],
            "transform_dict": transform_dict,
            "isolated_only": random.choices([True, False], [0.2, 0.8])[0],
            "on_inverse": random.choices([True, False], [0.05, 0.95])[0],
        }


class HolidayTransformer(EmptyTransformer):
    def __init__(
        self,
        anomaly_detector_params={},
        threshold=0.8,
        min_occurrences=2,
        splash_threshold=0.65,
        use_dayofmonth_holidays=True,
        use_wkdom_holidays=True,
        use_wkdeom_holidays=True,
        use_lunar_holidays=True,
        use_lunar_weekday=False,
        use_islamic_holidays=False,
        use_hebrew_holidays=False,
        use_hindu_holidays=False,
        remove_excess_anomalies=True,
        impact=None,
        regression_params=None,
        n_jobs: int = 1,
        output="multivariate",  # really can only be this for preprocessing
        verbose: int = 1,
    ):
        """Detect anomalies, then mark as holidays (events, festivals, etc) any that reoccur to a calendar.

        Args:
            anomaly_detector_params (dict): anomaly detection params passed to detector class
            threshold (float): percent of date occurrences that must be anomalous (0 - 1)
            splash_threshold (float): None, or % required, avg of nearest 2 neighbors to point
            use* (bool): whether to use these calendars for holiday detection
        """
        super().__init__(name="HolidayTransformer")
        self.anomaly_detector_params = anomaly_detector_params
        self.threshold = threshold
        self.min_occurrences = min_occurrences
        self.splash_threshold = splash_threshold
        self.use_dayofmonth_holidays = use_dayofmonth_holidays
        self.use_wkdom_holidays = use_wkdom_holidays
        self.use_wkdeom_holidays = use_wkdeom_holidays
        self.use_lunar_holidays = use_lunar_holidays
        self.use_lunar_weekday = use_lunar_weekday
        self.use_islamic_holidays = use_islamic_holidays
        self.use_hebrew_holidays = use_hebrew_holidays
        self.use_hindu_holidays = use_hindu_holidays
        self.output = output
        self.anomaly_model = AnomalyRemoval(
            output=output, **self.anomaly_detector_params, n_jobs=n_jobs
        )
        self.remove_excess_anomalies = remove_excess_anomalies
        self.fillna = anomaly_detector_params.get("fillna", None)
        self.impact = impact
        self.regression_params = regression_params
        self.n_jobs = n_jobs
        self.holiday_count = 0
        self.df_cols = None
        self.verbose = verbose

    def dates_to_holidays(
        self, dates, style="flag", holiday_impacts=False, max_features=365
    ):
        """
        dates (pd.DatetimeIndex): list of dates
        style (str): option for how to return information
            "long" - return date, name, series for all holidays in a long style dataframe
            "impact" - returns dates, series with values of sum of impacts (if given) or joined string of holiday names
            'flag' - return dates, holidays flag, (is not 0-1 but rather sum of input series impacted for that holiday and day)
            'prophet' - return format required for prophet. Will need to be filtered on `series` for multivariate case
            'series_flag' - dates, series 0/1 for if holiday occurred in any calendar
        """
        if self.df_cols is None:
            return ValueError("HolidayTransformer has not yet been .fit()")
        return dates_to_holidays(
            dates,
            self.df_cols,
            style=style,
            holiday_impacts=holiday_impacts,
            day_holidays=self.day_holidays,
            wkdom_holidays=self.wkdom_holidays,
            wkdeom_holidays=self.wkdeom_holidays,
            lunar_holidays=self.lunar_holidays,
            lunar_weekday=self.lunar_weekday,
            islamic_holidays=self.islamic_holidays,
            hebrew_holidays=self.hebrew_holidays,
            hindu_holidays=self.hindu_holidays,
            max_features=max_features,
        )

    def fit(self, df):
        """Run holiday detection. Input wide-style pandas time series."""
        self.anomaly_model.fit(df)
        if np.min(self.anomaly_model.anomalies.values) != -1:
            if self.verbose > 1:
                print("HolidayTransformer: no anomalies detected.")
        (
            self.day_holidays,
            self.wkdom_holidays,
            self.wkdeom_holidays,
            self.lunar_holidays,
            self.lunar_weekday,
            self.islamic_holidays,
            self.hebrew_holidays,
            self.hindu_holidays,
        ) = anomaly_df_to_holidays(
            self.anomaly_model.anomalies,
            splash_threshold=self.splash_threshold,
            threshold=self.threshold,
            actuals=df if self.output != "univariate" else None,
            anomaly_scores=(
                self.anomaly_model.scores if self.output != "univariate" else None
            ),
            # actuals=df,
            # anomaly_scores=self.anomaly_model.scores,
            use_dayofmonth_holidays=self.use_dayofmonth_holidays,
            use_wkdom_holidays=self.use_wkdom_holidays,
            use_wkdeom_holidays=self.use_wkdeom_holidays,
            use_lunar_holidays=self.use_lunar_holidays,
            use_lunar_weekday=self.use_lunar_weekday,
            use_islamic_holidays=self.use_islamic_holidays,
            use_hebrew_holidays=self.use_hebrew_holidays,
            use_hindu_holidays=self.use_hindu_holidays,
        )
        self.df_cols = df.columns
        return self

    def transform(self, df):
        if self.remove_excess_anomalies:
            holidays = self.dates_to_holidays(df.index, style='series_flag')
            df2 = df[~((self.anomaly_model.anomalies == -1) & (holidays != 1))]
            self.holiday_count = np.count_nonzero(holidays)
        else:
            df2 = df.copy()

        if self.fillna is not None:
            df2 = FillNA(df2, method=self.fillna, window=10)

        if self.impact == "datepart_regression":
            self.holidays = self.dates_to_holidays(df.index, style='flag')
            self.regression_model = DatepartRegression(**self.regression_params)
            return self.regression_model.fit_transform(
                df2, regressor=self.holidays.astype(float)
            )
        if self.impact == "regression":
            self.holidays = self.dates_to_holidays(df.index, style='flag').clip(upper=1)
            self.holidays['intercept'] = 1
            weights = (np.arange(df2.shape[0]) ** 0.6)[..., None]
            self.model_coef = np.linalg.lstsq(
                self.holidays.to_numpy() * weights, df2.to_numpy() * weights, rcond=None
            )[0]
            return df2 - np.dot(self.holidays.iloc[:, 0:-1], self.model_coef[0:-1])
        elif self.impact == "median_value":
            holidays = self.dates_to_holidays(
                df.index, style='impact', holiday_impacts="value"
            )
            self.medians = df2.median()
            holidays = holidays.where(holidays == 0, holidays - self.medians)
            return df2 - holidays
        elif self.impact == "anomaly_score":
            holidays = self.dates_to_holidays(
                df.index, style='impact', holiday_impacts="anomaly_score"
            )
            self.medians = self.anomaly_model.scores.median().fillna(1)
            return df2 * (holidays / self.medians).replace(0.0, 1.0)
        elif self.impact is None or self.impact == "create_feature":
            return df2
        else:
            raise ValueError("`impact` arg not recognized in HolidayTransformer")

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df):
        if self.impact == "datepart_regression":
            holidays = self.dates_to_holidays(df.index, style='flag')
            return self.regression_model.inverse_transform(
                df, regressor=holidays.astype(float)
            )
        elif self.impact == "regression":
            holidays = self.dates_to_holidays(df.index, style='flag').clip(upper=1)
            holidays['intercept'] = 1
            return df + np.dot(holidays.iloc[:, 0:-1], self.model_coef[0:-1])
        elif self.impact == "median_value":
            holidays = self.dates_to_holidays(
                df.index, style='impact', holiday_impacts="value"
            )
            holidays = holidays.where(holidays == 0, holidays - self.medians)
            return df + holidays
        elif self.impact == "anomaly_score":
            holidays = self.dates_to_holidays(
                df.index, style='impact', holiday_impacts="anomaly_score"
            )
            return df / (holidays / self.medians).replace(0.0, 1.0)
        return df

    @staticmethod
    def get_new_params(method="random"):
        holiday_params = holiday_new_params(method=method)
        holiday_params['anomaly_detector_params'] = AnomalyRemoval.get_new_params(
            method="fast"
        )
        holiday_params['remove_excess_anomalies'] = random.choices(
            [True, False], [0.9, 0.1]
        )[0]
        if method in ['fast', 'superfast']:
            # regressions are actually faster at scale due to it making the same flag for all
            holiday_params['impact'] = random.choices(
                [
                    None,
                    'datepart_regression',
                    'regression',
                ],
                [0.05, 0.4, 0.4],
            )[0]
        else:
            holiday_params['impact'] = random.choices(
                [
                    None,
                    'median_value',
                    'anomaly_score',
                    'datepart_regression',
                    'regression',
                ],
                [0.1, 0.3, 0.3, 0.1, 0.1],
            )[0]
        if holiday_params['impact'] == 'datepart_regression':
            holiday_params['regression_params'] = DatepartRegression.get_new_params(
                method=method,
                holiday_countries_used=False,
            )
        else:
            holiday_params['regression_params'] = {}
        return holiday_params


class LocalLinearTrend(EmptyTransformer):
    """Remove a rolling linear trend.
    Note this will probably perform poorly with long forecast horizons as forecast trend is simply the tail (n_future) of data's trend.

    Args:
        rolling_window (int): width of window to take trend on
        n_future (int): amount of data for the trend to be used extending beyond the edges of history.
        macro_micro (bool): when True, splits the data into separate parts (trend and residual) and later combines together in inverse
    """

    def __init__(
        self,
        rolling_window: float = 0.1,
        # n_tails: float = 0.1,
        n_future: float = 0.2,
        method: str = "mean",
        macro_micro: bool = False,
        suffix: str = "_lltmicro",
        **kwargs,
    ):
        super().__init__(name="LocalLinearTrend")
        self.rolling_window = rolling_window
        # self.n_tails = n_tails
        self.n_future = n_future
        self.method = method
        self.macro_micro = macro_micro
        self.suffix = suffix

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.dates = df.index.to_julian_date()
        if self.rolling_window <= 0:
            raise ValueError(f"rolling_window {self.rolling_window} arg is not valid")
        elif self.rolling_window < 1:
            self.rolling_window = int(self.rolling_window * len(self.dates))

        """
        if self.n_tails <= 0:
            raise ValueError(f"n_tails {self.n_tails} arg is not valid")
        elif self.n_tails < 1:
            self.n_tails = int(self.n_tails * len(self.dates))
        """

        if self.n_future <= 0:
            raise ValueError(f"n_future {self.n_future} arg is not valid")
        elif self.n_future < 1:
            self.n_future = int(self.n_future * len(self.dates))

        self.dates_2d = np.repeat(self.dates.to_numpy()[..., None], df.shape[1], axis=1)
        w_1 = self.rolling_window - 1
        """
        slope, intercept = window_lin_reg(
            self.dates_2d,
            df.values,
            w=self.rolling_window,
        )
        self.slope = np.concatenate(
            [
                np.repeat(slope[0:self.n_tails].mean(axis=0)[None, ...], int(w_1 / 2), axis=0),
                slope,
                np.repeat(slope[-self.n_tails:].mean(axis=0)[None, ...], w_1 - int(w_1 / 2), axis=0)
            ]
        )
        self.intercept = np.concatenate([
            np.repeat(intercept[0:self.n_tails].mean(axis=0)[None, ...], int(w_1 / 2), axis=0),
            intercept,
            np.repeat(intercept[-self.n_tails:].mean(axis=0)[None, ...], w_1 - int(w_1 / 2), axis=0)
        ])
        """
        # rolling trend
        steps_ahd = int(w_1 / 2)
        y0 = np.repeat(np.array(df[0:1]), steps_ahd, axis=0)
        # d0 = -1 * self.dates_2d[1 : y0.shape[0] + 1][::-1]
        start_pt = self.dates_2d[0, 0]
        step = self.dates_2d[1, 0] - start_pt
        d0 = np_2d_arange(
            start_pt,
            stop=start_pt - ((y0.shape[0] + 1) * step),
            step=-step,
            num_columns=self.dates_2d.shape[1],
        )[1:][::-1]
        shape2 = (w_1 - steps_ahd, y0.shape[1])
        # end_point = self.dates_2d[-1, 0]
        # d2 = np_2d_arange(start=end_point, stop=end_point+)
        y2 = np.concatenate(
            [
                y0,
                df.to_numpy(),
                np.full(shape2, np.nan),
            ]
        )
        d = np.concatenate(
            [
                d0,
                self.dates_2d,
                np.full(shape2, np.nan),
            ]
        )
        self.slope, self.intercept = window_lin_reg_mean_no_nan(
            d, y2, w=self.rolling_window
        )

        if self.method == "mean":
            futslp = np.array([np.mean(self.slope[-self.n_future :], axis=0)])
            self.full_slope = np.concatenate(
                [
                    [np.mean(self.slope[0 : self.n_future], axis=0)],
                    self.slope,
                    futslp,
                    futslp,  # twice to have an N+1 size
                ]
            )
            futinc = np.array([np.mean(self.intercept[-self.n_future :], axis=0)])
            self.full_intercept = np.concatenate(
                [
                    [np.mean(self.intercept[0 : self.n_future], axis=0)],
                    self.intercept,
                    futinc,
                    futinc,
                ]
            )
            # self.greater_slope = self.slope[-self.n_future:].mean()
            # self.greater_intercept = self.intercept[-self.n_future:].mean()
            # self.lesser_slope = self.slope[0:self.n_future].mean()
            # self.lesser_intercept = self.intercept[0:self.n_future].mean()
        elif self.method == "median":
            futslp = np.array([np.median(self.slope[-self.n_future :], axis=0)])
            self.full_slope = np.concatenate(
                [
                    [np.median(self.slope[0 : self.n_future], axis=0)],
                    self.slope,
                    futslp,
                    futslp,  # twice to have an N+1 size
                ]
            )
            futinc = np.array([np.median(self.intercept[-self.n_future :], axis=0)])
            self.full_intercept = np.concatenate(
                [
                    [np.median(self.intercept[0 : self.n_future], axis=0)],
                    self.intercept,
                    futinc,
                    futinc,
                ]
            )
        self.full_dates = np.concatenate(
            [
                [self.dates.min() - 0.01],
                self.dates,
                [self.dates.max() + 0.01],
            ]
        )
        if self.macro_micro:
            macro = pd.DataFrame(
                self.slope * self.dates_2d + self.intercept,
                index=df.index,
                columns=df.columns,
            )
            self.columns = df.columns
            micro = (df - macro).rename(columns=lambda x: str(x) + self.suffix)
            return pd.concat([macro, micro], axis=1)
        else:
            return df - (self.slope * self.dates_2d + self.intercept)

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
        dates = df.index.to_julian_date()
        dates_2d = np.repeat(dates.to_numpy()[..., None], df.shape[1], axis=1)
        idx = self.full_dates.searchsorted(dates)
        # return df - (self.full_slope[idx] * dates_2d + self.full_intercept[idx])

        if self.macro_micro:
            macro = pd.DataFrame(
                self.full_slope[idx] * dates_2d + self.full_intercept[idx],
                index=df.index,
                columns=df.columns,
            )
            micro = (df - macro).rename(columns=lambda x: str(x) + self.suffix)
            return pd.concat([macro, micro], axis=1)
        else:
            return df - (self.full_slope[idx] * dates_2d + self.full_intercept[idx])

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.macro_micro:
            macro = df[self.columns]
            micro = df[df.columns.difference(self.columns)]
            micro = micro.rename(columns=lambda x: str(x)[: -len(self.suffix)])[
                self.columns
            ]
            return macro + micro
        else:
            dates = df.index.to_julian_date()
            n_cols = df.shape[1]
            dates_2d = np.repeat(dates.to_numpy()[..., None], n_cols, axis=1)
            idx = self.full_dates.searchsorted(dates)
            return df + (self.full_slope[idx] * dates_2d + self.full_intercept[idx])

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        return {
            "rolling_window": random.choices(
                [0.1, 90, 30, 180, 360, 0.05], [0.5, 0.1, 0.1, 0.1, 0.1, 0.2]
            )[0],
            "n_tails": random.choices(
                [0.1, 90, 30, 180, 360, 0.05], [0.5, 0.1, 0.1, 0.1, 0.1, 0.2]
            )[0],
            "n_future": random.choices(
                [0.2, 90, 360, 0.1, 0.05], [0.5, 0.1, 0.1, 0.1, 0.2]
            )[0],
            "method": random.choice(["mean", "median"]),
            "macro_micro": random.choice([True, False]),
        }


class KalmanSmoothing(EmptyTransformer):
    """Apply a Kalman Filter to smooth data given a transition matrix.

    Args:
        rows (int): number of rows to average as last record
        lag (int): use last value as this lag back, 1 is no shift, 2 is lag one from end, ie second to last
        method (str): 'additive', 'multiplicative'
        strength (float): softening parameter [0, 1], 1.0 for full difference
    """

    def __init__(
        self,
        state_transition=[[1, 1], [0, 1]],
        process_noise=[[0.1, 0.0], [0.0, 0.01]],
        observation_model=[[1, 0]],
        observation_noise: float = 1.0,
        em_iter: int = None,
        on_transform: bool = True,
        on_inverse: bool = False,
        **kwargs,
    ):
        super().__init__(name="KalmanSmoothing")
        self.state_transition = state_transition
        self.process_noise = process_noise
        self.observation_model = observation_model
        self.observation_noise = observation_noise
        self.em_iter = em_iter
        self.on_transform = on_transform
        self.on_inverse = on_inverse

    @staticmethod
    def get_new_params(method: str = "random"):
        params = new_kalman_params(method=method, allow_auto=False)
        selection = random.choices([True, False], [0.8, 0.2])[0]
        params['on_transform'] = selection
        params["on_inverse"] = not selection
        return params

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.kf = KalmanFilter(
            state_transition=self.state_transition,  # matrix A
            process_noise=self.process_noise,  # Q
            observation_model=self.observation_model,  # H
            observation_noise=self.observation_noise,  # R
        )
        if self.em_iter is not None:
            self.kf = self.kf.em(df.to_numpy().T, n_iter=self.em_iter)
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.on_transform:
            result = self.kf.smooth(df.to_numpy().T, covariances=False)
            return pd.DataFrame(
                result.observations.mean.T, index=df.index, columns=df.columns
            )
        else:
            return df

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.on_inverse:
            result = self.kf.smooth(df.to_numpy().T, covariances=False)
            return pd.DataFrame(
                result.observations.mean.T, index=df.index, columns=df.columns
            )
        else:
            return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)


class RegressionFilter(EmptyTransformer):
    """Models seasonal and local linear trend, and clips std dvs from this fit."""

    def __init__(
        self,
        name: str = "RegressionFilter",
        sigma: float = 2.0,
        rolling_window: int = 90,
        run_order: str = "season_first",
        regression_params: dict = None,
        holiday_params: dict = None,
        holiday_country: str = "US",
        trend_method: str = "local_linear",
        **kwargs,
    ):
        super().__init__(name=name)
        self.sigma = sigma
        self.rolling_window = rolling_window
        self.run_order = run_order
        self.regression_params = regression_params
        self.holiday_params = holiday_params
        self.holiday_country = holiday_country
        self.trend_method = trend_method

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.holiday_params is not None:
            self.holiday_params["regression_params"] = self.regression_params
            self.holiday_params["impact"] = 'datepart_regression'
            self.seasonal = HolidayTransformer(**self.holiday_params)
        else:
            if self.regression_params is None:
                self.regression_params = {
                    "regression_model": {
                        "model": "DecisionTree",
                        "model_params": {"max_depth": 4, "min_samples_split": 2},
                    },
                    "datepart_method": "common_fourier",
                    "holiday_countries_used": True,
                }
            self.seasonal = DatepartRegressionTransformer(
                **self.regression_params, holiday_country=self.holiday_country
            )
        if self.trend_method == "local_linear":
            self.trend = LocalLinearTrend(
                rolling_window=self.rolling_window
            )  # memory intensive at times
        # self.trend = RollingMeanTransformer(rolling_window=self.rolling_window, fixed=False, center=True)

        if self.run_order == 'season_first':
            deseason = self.seasonal.fit_transform(df)
            if self.trend_method == "local_linear":
                result = self.trend.fit_transform(deseason)
            else:
                trend_diff = deseason.rolling(
                    self.rolling_window, center=True, min_periods=1
                ).mean()
                result = deseason - trend_diff
        else:
            if self.trend_method == "local_linear":
                detrend = self.trend.fit_transform(df)
            else:
                trend_diff = df.rolling(
                    self.rolling_window, center=True, min_periods=1
                ).mean()
                detrend = df - trend_diff
            result = self.seasonal.fit_transform(detrend)

        std_dev = result.std() * self.sigma
        clipped = result.clip(upper=std_dev, lower=-1 * std_dev, axis=1)

        if self.run_order == 'season_first':
            if self.trend_method == "local_linear":
                retrend = self.trend.inverse_transform(clipped, trans_method='original')
            else:
                retrend = clipped + trend_diff
            original = self.seasonal.inverse_transform(retrend)
        else:
            reseason = self.seasonal.inverse_transform(clipped)
            if self.trend_method == "local_linear":
                original = self.trend.inverse_transform(
                    reseason, trans_method='original'
                )
            else:
                original = reseason + trend_diff
        return original

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
        if self.run_order == 'season_first':
            deseason = self.seasonal.transform(df)
            if self.trend_method == "local_linear":
                result = self.trend.transform(deseason)
            else:
                trend_diff = deseason.rolling(
                    self.rolling_window, center=True, min_periods=1
                ).mean()
                result = deseason - trend_diff
        else:
            if self.trend_method == "local_linear":
                detrend = self.trend.transform(df)
            else:
                trend_diff = deseason.rolling(
                    self.rolling_window, center=True, min_periods=1
                ).mean()
                detrend = df - trend_diff
            result = self.seasonal.transform(detrend)

        std_dev = result.std() * self.sigma
        clipped = result.clip(upper=std_dev, lower=-1 * std_dev, axis=1)

        if self.run_order == 'season_first':
            if self.trend_method == "local_linear":
                retrend = self.trend.inverse_transform(clipped, trans_method='original')
            else:
                retrend = clipped + trend_diff
            original = self.seasonal.inverse_transform(retrend)
        else:
            reseason = self.seasonal.inverse_transform(clipped)
            if self.trend_method == "local_linear":
                original = self.trend.inverse_transform(
                    reseason, trans_method='original'
                )
            else:
                original = reseason + trend_diff
        return original

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """No changes made.

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
        return "Transformer " + str(self.name) + ", uses standard .fit/.transform"

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        regression_params = DatepartRegressionTransformer.get_new_params(method="fast")

        if method in ["fast", 'superfast', 'scalable']:
            holiday_trans_use = False
            trend_method = "rolling_mean"
        else:
            holiday_trans_use = random.choices([True, False], [0.3, 0.7])[0]
            trend_method = random.choices(["rolling_mean", "local_linear"], [0.3, 0.7])[
                0
            ]
        if holiday_trans_use:
            holiday_params = HolidayTransformer.get_new_params(method="fast")
            holiday_params["regression_params"] = regression_params
            holiday_params["impact"] = 'datepart_regression'
        else:
            holiday_params = None

        return {
            "sigma": random.choices(
                [0.5, 1, 1.5, 2, 2.5, 3], [0.1, 0.4, 0.1, 0.3, 0.05, 0.1]
            )[0],
            "rolling_window": 90,
            "run_order": random.choices(["season_first", "trend_first"], [0.7, 0.3])[0],
            "regression_params": regression_params,
            "holiday_params": holiday_params,
            "trend_method": trend_method,
        }


class LevelShiftMagic(EmptyTransformer):
    """Detects and corrects for level shifts. May seriously alter trend.

    Args:
        method (str): "clip" or "remove"
        std_threshold (float): number of std devs from mean to call an outlier
        fillna (str): fillna method to use per tools.impute.FillNA
    """

    def __init__(
        self,
        window_size: int = 90,
        alpha: float = 2.5,
        grouping_forward_limit: int = 3,
        max_level_shifts: int = 20,
        alignment: str = "average",
        old_way: bool = False,
        **kwargs,
    ):
        super().__init__(name="LevelShiftMagic")
        self.window_size = window_size
        self.alpha = alpha
        self.grouping_forward_limit = grouping_forward_limit
        self.max_level_shifts = max_level_shifts
        self.alignment = alignment
        self.old_way = old_way

    @staticmethod
    def get_new_params(method: str = "random"):
        return {
            "window_size": random.choices(
                [4, 7, 14, 30, 70, 90, 120, 364],
                [0.05, 0.1, 0.05, 0.4, 0.05, 0.4, 0.05, 0.1],
                k=1,
            )[0],
            "alpha": random.choices(
                [1.0, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0],
                [0.05, 0.02, 0.2, 0.02, 0.3, 0.2, 0.15, 0.1],
                k=1,
            )[0],
            "grouping_forward_limit": random.choice([2, 3, 4, 5, 6]),
            "max_level_shifts": random.choices(
                [3, 5, 8, 10, 30, 40], [0.05, 0.3, 0.05, 0.2, 0.2, 0.05]
            )[0],
            "alignment": random.choices(
                [
                    "average",
                    "last_value",
                    "rolling_diff",
                    "rolling_diff_3nn",
                    "rolling_diff_5nn",
                ],
                [0.5, 0.2, 0.15, 0.25, 0.05],
            )[0],
        }

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        rolling = df.rolling(self.window_size, center=False, min_periods=1).mean()
        # pandas 1.1.0 introduction, or thereabouts
        try:
            indexer = pd.api.indexers.FixedForwardWindowIndexer(
                window_size=self.window_size
            )
            rolling_forward = df.rolling(window=indexer, min_periods=1).mean()
        except Exception:
            rolling_forward = (
                df.loc[::-1]
                .rolling(self.window_size, center=False, min_periods=1)
                .mean()[::-1]
            )
        # compare rolling forward and backwards diffs
        diff = rolling - rolling_forward
        threshold = diff.std() * self.alpha
        diff_abs = diff.abs()
        diff_mask_0 = diff_abs > threshold  #  | (diff < -threshold)
        # merge nearby groups
        diff_smoothed = diff_abs.where(diff_mask_0, np.nan).ffill(
            limit=self.grouping_forward_limit
        )
        diff_mask = (diff_smoothed > threshold) | (diff_smoothed < -threshold)
        # the max of each changepoint group is the chosen changepoint of the level shift
        # doesn't handle identical maxes although that is unlikely with these floating point averages
        maxes = diff_abs.where(diff_mask, np.nan).max()
        # group ids are needed so that we can progressively remove
        range_arr = pd.DataFrame(
            np.mgrid[0 : df.shape[0] : 1, 0 : df.shape[1]][0],
            index=df.index,
            columns=df.columns,
        )
        group_ids = range_arr[~diff_mask].ffill()  # [diff_mask]
        max_mask = diff_abs == maxes
        if not self.old_way:
            # new way is from  gpt o1-preview which thought that the old way was leading to
            # errors when doing the mean of the group ids
            # I am too sleepy to be convinced for sure
            # but this is undeniably faster, and still passes the previous unittest
            # yielding only some (more) level shifts on a few edge cases it seems
            group_ids_np = group_ids.to_numpy()
            diff_abs_np = diff_abs.to_numpy()
            diff_mask_np = diff_mask.to_numpy()
            max_mask_np = max_mask.to_numpy()

            # Initialize curr_diff_np
            used_groups_np = np.where(max_mask_np, group_ids_np, np.nan)
            used_groups_flat = used_groups_np[~np.isnan(used_groups_np)]
            used_groups_unique = np.unique(used_groups_flat)
            curr_diff_np = np.where(
                (~np.isin(group_ids_np, used_groups_unique)) & diff_mask_np,
                diff_abs_np,
                np.nan,
            )
            curr_diff_sum = np.nansum(curr_diff_np)
            count = 0

            while curr_diff_sum != 0 and count < self.max_level_shifts:
                curr_maxes_np = np.nanmax(np.nan_to_num(curr_diff_np), axis=0)
                mask = curr_diff_np == curr_maxes_np
                max_mask_np |= mask
                used_groups_np = np.where(mask, group_ids_np, np.nan)
                used_groups_flat = used_groups_np[~np.isnan(used_groups_np)]
                used_groups_unique = np.unique(used_groups_flat)
                mask_to_update = (
                    np.isin(group_ids_np, used_groups_unique) & diff_mask_np
                )
                curr_diff_np = np.where(~mask_to_update, curr_diff_np, np.nan)
                curr_diff_sum = np.sum(~np.isnan(curr_diff_np))
                count += 1
        else:
            self.used_groups = group_ids[max_mask].mean()
            curr_diff = diff_abs.where(
                ((group_ids != self.used_groups) & diff_mask), np.nan
            )
            curr_diff_sum = np.nansum(curr_diff.to_numpy())
            count = 0
            while curr_diff_sum != 0 and count < self.max_level_shifts:
                curr_maxes = curr_diff.max()
                max_mask = max_mask | (curr_diff == curr_maxes)
                used_groups = group_ids[curr_diff == curr_maxes].mean()
                curr_diff = curr_diff.where(
                    ((group_ids != used_groups) & diff_mask), np.nan
                )
                curr_diff_sum = np.sum(~curr_diff.isnull().to_numpy())
                count += 1
        # alignment is tricky, especially when level shifts are a mix of gradual and instantaneous
        if self.alignment == "last_value":
            self.lvlshft = (
                (df[max_mask] - df[max_mask.shift(1)].shift(-1))
                .fillna(0)
                .loc[::-1]
                .cumsum()[::-1]
            )
        elif self.alignment == "average":
            # average the two other approaches
            lvlshft1 = (
                (df[max_mask] - df[max_mask.shift(1)].shift(-1))
                .fillna(0)
                .loc[::-1]
                .cumsum()[::-1]
            )
            lvlshft2 = diff[max_mask].fillna(0).loc[::-1].cumsum()[::-1]
            self.lvlshft = (lvlshft1 + lvlshft2) / 2
        elif self.alignment == "rolling_diff_3nn":
            self.lvlshft = (
                (
                    (
                        diff[max_mask.shift(1)].shift(-1).fillna(0)
                        + diff[max_mask]
                        + diff[max_mask.shift(-1)].shift(1).fillna(0)
                    )
                    / 3
                )
                .fillna(0)
                .loc[::-1]
                .cumsum()[::-1]
            )
        elif self.alignment == "rolling_diff_5nn":
            self.lvlshft = (
                (
                    (
                        diff[max_mask.shift(2)].shift(-2).fillna(0)
                        + diff[max_mask.shift(1)].shift(-1).fillna(0)
                        + diff[max_mask]
                        + diff[max_mask.shift(-1)].shift(1).fillna(0)
                        + diff[max_mask.shift(-2)].shift(2).fillna(0)
                    )
                    / 5
                )
                .fillna(0)
                .loc[::-1]
                .cumsum()[::-1]
            )
        else:
            self.lvlshft = diff[max_mask].fillna(0).loc[::-1].cumsum()[::-1]

        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df - self.lvlshft.reindex(
            index=df.index, columns=df.columns
        ).bfill().fillna(0)

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df + self.lvlshft.reindex(
            index=df.index, columns=df.columns
        ).bfill().fillna(0)

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)


LevelShiftTransformer = LevelShiftMagic


class CenterSplit(EmptyTransformer):
    """Vaguely Croston inspired approach separating occurrence from magnitude.

    Args:
        center (str): 'zero' or 'median', the value to use as most the intermittent gap
        fillna (str): a fillna method, see standard fillna methods
    """

    def __init__(
        self,
        center: str = "zero",
        fillna="linear",
        suffix: str = "_lltmicro",
        **kwargs,
    ):
        super().__init__(name="CenterSplit")
        self.center = center
        self.fillna = fillna
        self.suffix = suffix

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.center in ["zero", "0", 0]:
            mask = df != 0
            use_df = df
        elif self.center == "median":
            self.median = df.median(axis=0)
            mask = df != self.median
            use_df = df - self.median
        elif isinstance(self.center, int):
            mask = df != self.center
            use_df = df
        else:
            raise ValueError(f"CenterSplit arg center `{self.center}` not recognized")

        macro = use_df.where(mask, np.nan)

        self.columns = df.columns
        micro = use_df.where(~mask, 1).rename(columns=lambda x: str(x) + self.suffix)
        return FillNA(macro, method=self.fillna, window=10).join(micro)

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
        if self.center in ["zero", "0", 0]:
            mask = df != 0
            use_df = df
        elif self.center == "median":
            # self.median = df.median(axis=0)
            mask = df != self.median
            use_df = df - self.median
        else:
            raise ValueError(
                f"ModifiedCroston arg center `{self.center}` not recognized"
            )

        macro = use_df.where(mask, np.nan)

        # self.columns = df.columns
        micro = use_df.where(~mask, 1).rename(columns=lambda x: str(x) + self.suffix)
        # return pd.concat([macro, micro], axis=1)
        return FillNA(macro, method=self.fillna, window=10).join(micro)

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        macro = df[self.columns]
        micro = df[df.columns.difference(self.columns)]
        micro = micro.rename(columns=lambda x: str(x)[: -len(self.suffix)])[
            self.columns
        ]
        if self.center == "median":
            return macro * micro + self.median
        else:
            return macro * micro

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        return {
            "fillna": random.choices(
                [
                    "linear",  # a bit RAM heavier than the others
                    'pchip',
                    'akima',
                    'mean',
                    'ffill',
                    "one",
                    # "SeasonalityMotifImputer1K",
                ],
                [0.1, 0.02, 0.2, 0.2, 0.3, 0.1],
            )[0],
            "center": random.choices(["zero", "median"], [0.7, 0.3])[0],
        }


class FFTFilter(EmptyTransformer):
    """Fit Fourier Transform and keep only lowest frequencies below cutoff

    Args:
        cutoff (float): smoothing value
        reverse (bool): if True, keep highest frequencies only
    """

    def __init__(
        self,
        cutoff: float = 0.1,
        reverse: bool = False,
        on_transform: bool = True,
        on_inverse: bool = False,
        **kwargs,
    ):
        super().__init__(name="FFTFilter")
        self.cutoff = cutoff
        self.reverse = reverse
        self.on_transform = on_transform
        self.on_inverse = on_inverse

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

    def _filter(self, df):
        data = df.to_numpy()
        spectrum = np.fft.fft(data, axis=0)
        frequencies = np.fft.fftfreq(data.shape[0])

        # Zero out components beyond the cutoff
        if self.reverse:
            spectrum[np.abs(frequencies) < self.cutoff] = 0
        else:
            spectrum[np.abs(frequencies) > self.cutoff] = 0

        # Inverse FFT to get the smoothed data
        smoothed_data = np.real(np.fft.ifft(spectrum, axis=0))
        return pd.DataFrame(smoothed_data, index=df.index, columns=df.columns)

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.on_transform:
            return self._filter(df)
        else:
            return df

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.on_inverse:
            return self._filter(df)
        else:
            return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self.transform(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        return {
            "cutoff": random.choices(
                [0.005, 0.01, 0.05, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 7, 365],
                [0.1, 0.2, 0.1, 0.24, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
            )[0],
            "reverse": random.choices([False, True], [0.9, 0.1])[0],
            "on_transform": random.choices([False, True], [0.1, 0.9])[0],
            "on_inverse": random.choices([False, True], [0.9, 0.1])[0],
        }


class FFTDecomposition(EmptyTransformer):
    """FFT decomposition, then removal, then extrapolation and addition.

    Args:
        n_harmnonics (float): number of frequencies to include
        detrend (str): None, 'linear', or 'quadratic'
    """

    def __init__(
        self,
        n_harmonics: float = 0.1,
        detrend: str = "linear",
        **kwargs,
    ):
        super().__init__(name="FFTDecomposition")
        self.n_harmonics = n_harmonics
        self.detrend = detrend

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.df_columns = df.columns
        self.df_index = df.index
        self.freq = infer_frequency(df)
        self.fft = fft_class(n_harm=self.n_harmonics, detrend=self.detrend)
        self.fft.fit(df.to_numpy())
        self.start_forecast_len = df.shape[0]
        self._predict(forecast_length=self.start_forecast_len)
        return df - self.predicted.reindex(df.index)

    def _predict(self, forecast_length):
        self.predicted = pd.DataFrame(
            self.fft.predict(forecast_length), columns=self.df_columns
        )
        self.predicted.index = self.df_index.union(
            pd.date_range(
                self.df_index[-1], periods=forecast_length + 1, freq=self.freq
            )
        )
        return self

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
        return df - self.predicted.reindex(df.index)

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if df.shape[0] > self.start_forecast_len:
            self._predict(forecast_length=df.shape[0])
        return df + self.predicted.reindex(df.index)

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        return {
            "n_harmonics": random.choices(
                [None, 10, 20, 0.5, -0.5, -0.95, "mid10", "mid20"],
                [0.1, 0.3, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
            )[0],
            "detrend": random.choices(
                [None, "linear", "quadratic", "cubic", "quartic"],
                [0.4, 0.3, 0.3, 0.1, 0.05],
            )[0],
        }


class ReplaceConstant(EmptyTransformer):
    """Replace constant, filling the NaN, then possibly reintroducing.
    If reintroducion is used, it is unlikely inverse_transform will match original exactly.

    Args:
        constant (float): target to replace
        fillna (str): None, and standard fillna methods of AutoTS
        reintroduction_model (dict): if given, attempts to predict occurrence of constant and reintroduce
    """

    def __init__(
        self,
        constant: float = 0,
        fillna: str = "linear",
        reintroduction_model: str = None,
        n_jobs: int = 1,
        **kwargs,
    ):
        super().__init__(name="ReplaceConstant")
        self.constant = constant
        self.fillna = fillna
        self.reintroduction_model = reintroduction_model
        self.n_jobs = n_jobs

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.reintroduction_model is None:
            return FillNA(
                df.replace(self.constant, np.nan), method=self.fillna, window=10
            )
        else:
            # goal is for y to be 0 for constant and 1 for everything else
            y = 1 - np.where(df != self.constant, 0, 1)
            X = date_part(
                df.index,
                method=self.reintroduction_model.get(
                    "datepart_method", "simple_binarized"
                ),
            )
            if y.ndim < 2:
                multioutput = False
            elif y.shape[1] < 2:
                multioutput = False
            else:
                multioutput = True

            self.model = retrieve_classifier(
                regression_model=self.reintroduction_model,
                verbose=0,
                verbose_bool=False,
                random_seed=2023,
                multioutput=multioutput,
                n_jobs=self.n_jobs,
            )
            self.model.fit(X, y)
            if False:
                print(self.model.score(X, y))
            return FillNA(
                df.replace(self.constant, np.nan), method=self.fillna, window=10
            )

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
        return FillNA(df.replace(self.constant, np.nan), method=self.fillna, window=10)

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.reintroduction_model is None:
            return df
        else:
            X = date_part(
                df.index,
                method=self.reintroduction_model.get(
                    "datepart_method", "simple_binarized"
                ),
            )
            pred = pd.DataFrame(
                self.model.predict(X), index=df.index, columns=df.columns
            )
            if self.constant == 0:
                return df * pred
            else:
                return df.where(pred != 0, self.constant)

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        reintroduction_model = random.choices([None, True], [0.3, 0.7])[0]
        if reintroduction_model:
            reintroduction_model = generate_classifier_params(method='fast')
            reintroduction_model['datepart_method'] = random_datepart(method=method)
        return {
            "constant": random.choices([0, 1], [0.9, 0.1])[0],
            "reintroduction_model": reintroduction_model,
            "fillna": random.choices(
                [
                    None,
                    "linear",
                    'pchip',
                    'akima',
                    'mean',
                    'ffill',
                    "SeasonalityMotifImputer1K",
                ],
                [0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.0001],
            )[0],
        }


def exponential_decay(n, span=None, halflife=None):
    assert not (
        (span is not None) and (halflife is not None)
    ), "Only one of span or halflife should be provided"

    t = np.arange(n)

    if span is not None:
        decay_values = np.exp(-t / span)
    else:
        decay_values = np.exp(-np.log(2) * t / halflife)

    return decay_values


class AlignLastDiff(EmptyTransformer):
    """Shift all data relative to the last value(s) of the series.
    This version aligns based on historic diffs rather than direct values.

    Args:
        rows (int): number of rows to average as diff history. rows=1 rather different from others
        quantile (float): quantile of historic diffs to use as allowed [0, 1]
        decay_span (int): span of exponential decay which softens adjustment to no adjustment
    """

    def __init__(
        self,
        rows: int = 1,
        quantile: float = 0.5,
        decay_span: float = None,
        displacement_rows: int = 1,
        **kwargs,
    ):
        super().__init__(name="AlignLastDiff")
        self.rows = rows
        self.quantile = quantile
        self.decay_span = decay_span
        self.displacement_rows = displacement_rows
        self.adjustment = None

    @staticmethod
    def get_new_params(method: str = "random"):
        return {
            "rows": random.choices(
                [1, 2, 4, 7, 28, 90, 364, None],
                [0.2, 0.05, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1],
            )[0],
            "displacement_rows": random.choices(
                [1, 2, 4, 7, 21], [0.8, 0.05, 0.05, 0.05, 0.05]
            )[0],
            "quantile": random.choices(
                [1.0, 0.9, 0.7, 0.5, 0.2, 0], [0.8, 0.05, 0.05, 0.05, 0.05, 0.05]
            )[0],
            "decay_span": random.choices(
                [None, 2, 3, 4, 90, 365], [0.6, 0.1, 0.1, 0.05, 0.1, 0.1]
            )[0],
        }

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.rows is None:
            self.rows = df.shape[0]
        # fill NaN if present (up to a limit for slight speedup)
        if np.isnan(np.sum(np.array(df)[-self.rows :])):
            local_df = df.ffill(axis=0)
        else:
            local_df = df

        self.center = df.iloc[-1, :]

        if self.rows <= 1:
            self.diff = np.abs(((local_df - local_df.shift(1)).iloc[-1:]).to_numpy())
        else:
            # positive diff = growing
            diff = (local_df - local_df.shift(1)).iloc[1:].iloc[-self.rows :]
            mask = diff > 0
            self.growth_diff = nan_quantile(diff[mask], q=self.quantile, axis=0)
            self.decline_diff = nan_quantile(diff[~mask], q=self.quantile, axis=0)

        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def inverse_transform(self, df, trans_method: str = "forecast", adjustment=None):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
            adjustment (float): size of shift, utilized for adjusting the upper and lower bounds to match point forecast
        """
        if self.adjustment is not None:
            self.adjustment = adjustment
        if trans_method == "original":
            return df

        if self.adjustment is None:
            if self.displacement_rows == 1 or self.displacement_rows is None:
                displacement = df.iloc[0] - self.center  # positive is growth
            else:
                displacement = (
                    df.iloc[0 : self.displacement_rows].mean() - self.center
                )  # positive is growth

            if self.rows <= 1:
                self.adjustment = np.where(
                    np.abs(displacement) > self.diff.flatten(),
                    displacement - (self.diff.flatten() * np.sign(displacement)),
                    0,
                )
            else:
                self.adjustment = np.where(
                    displacement > 0,
                    np.where(
                        displacement > self.growth_diff,
                        displacement - self.growth_diff,
                        0,
                    ),
                    np.where(
                        displacement < self.decline_diff,
                        displacement - self.decline_diff,
                        0,
                    ),
                )
            if self.decay_span is not None:
                self.adjustment = np.repeat(
                    self.adjustment[..., np.newaxis], df.shape[0], axis=1
                ).T
                self.adjustment = (
                    self.adjustment
                    * exponential_decay(self.adjustment.shape[0], span=self.decay_span)[
                        ..., np.newaxis
                    ]
                )

        return df - self.adjustment

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)


class DiffSmoother(EmptyTransformer):
    def __init__(
        self,
        output="multivariate",
        method=None,
        transform_dict=None,
        method_params=None,
        fillna=2.0,
        n_jobs=1,
        adjustment: int = 2,
        reverse_alignment=True,
        isolated_only=False,
    ):
        """Detect anomalies on a historic dataset in the DIFFS then cumsum back to origin space.
        No inverse_transform available.

        Args:
            output (str): 'multivariate' (each series unique outliers), or 'univariate' (all series together for one outlier flag per timestamp)
            method (str): method choosen, from sklearn, AutoTS, and basic stats. Use `.get_new_params()` to see potential models
            transform_dict (dict): option but helpful, often datepart, differencing, or other standard AutoTS transformer params
            method_params (dict): parameters specific to the method, use `.get_new_params()` to see potential models
            fillna (str): how to fill anomaly values removed
            reverse_alighment (bool): if True, remove diffs then cumsum
            isolated_only (bool): if True, only standalone anomalies are used
            n_jobs (int): multiprocessing jobs, used by some methods
        """
        super().__init__(name="DiffSmoother")
        self.output = output
        self.method = method
        self.transform_dict = transform_dict
        self.method_params = method_params
        self.n_jobs = n_jobs
        self.fillna = fillna
        self.adjustment = adjustment
        self.reverse_alignment = reverse_alignment
        self.isolated_only = isolated_only

    def fit(self, df):
        """Fit.
        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self

    def transform(self, df):
        """Return differenced data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        diffs = df.diff(-1).head(df.shape[0] - 1)
        if self.transform_dict is not None:
            model = GeneralTransformer(**self.transform_dict)
            diffs = model.fit_transform(diffs)

        # integer or float is a clip based approach, others are anomaly detection
        if isinstance(self.fillna, (float, int)) and self.reverse_alignment:
            diffs = diffs.clip(
                upper=diffs[diffs > 0].std() * self.fillna,
                lower=-(diffs[diffs < 0].std() * self.fillna),
                axis=1,
            )
        else:
            # for fillna float and not reverse (not in default get_new_params) but just in case
            if isinstance(self.fillna, (float, int)):
                self.method = "rolling_zscore"
                self.method_parmas = {
                    'distribution': 'norm',
                    'alpha': norm.sf(self.fillna, 1),
                    'rolling_periods': 30,
                    'center': False,
                }
                self.transform_dict = None
                fillna = 'ffill'
            else:
                fillna = self.fillna
            self.anomalies, self.scores = detect_anomalies(
                diffs.copy().abs(),
                output=self.output,
                method=self.method,
                transform_dict=self.transform_dict,
                method_params=self.method_params,
                n_jobs=self.n_jobs,
            )
            if self.isolated_only:
                # replace all anomalies (-1) except those which are isolated (for diffs, 1, -1, -1, 1 pattern)
                # this rather assumes the anomaly is (up anom, down anom) on the diff, but that's not actually enforced
                mask = self.anomalies == -1
                mask = mask & (self.anomalies.shift(1) == 1)
                mask = mask & (self.anomalies.shift(-1) == -1)
                mask = mask & (self.anomalies.shift(-2) == 1)
                self.anomalies[mask] = 1
            if self.reverse_alignment:
                diffs = diffs.where(self.anomalies != -1, np.nan)
            else:
                # also removes any -1 in your first row of data, for giggles
                full_anom = pd.concat([df.iloc[0:1], self.anomalies])
                full_anom.index = df.index
                full_anom.columns = df.columns
                diffs = df.where(full_anom != -1, np.nan)
            if self.fillna is not None:
                diffs = FillNA(diffs, method=fillna, window=10)

        if self.reverse_alignment:
            return pd.concat([diffs, df.tail(1)]).iloc[::-1].cumsum().iloc[::-1]
        else:
            return diffs

    def fit_transform(self, df):
        """Fits and Returns Magical DataFrame
        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self.transform(df)

    @staticmethod
    def get_new_params(method="fast"):
        fillna = random.choices(
            [
                None,
                "ffill",
                "mean",
                "rolling_mean_24",
                "linear",
                'time',
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
            ],
            [0.0, 0.3, 0.1, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )[0]
        if isinstance(fillna, (float, int)):
            method_choice = None
            method_params = None
            transform_dict = None
            reverse_alignment = True
            isolated_only = False
        else:
            method_choice, method_params, transform_dict = anomaly_new_params(
                method=method
            )
            reverse_alignment = random.choices([True, False], [0.3, 0.7])[0]
            isolated_only = random.choice([True, False])

        return {
            "method": method_choice,
            "method_params": method_params,
            "transform_dict": None,
            "reverse_alignment": reverse_alignment,
            "isolated_only": isolated_only,
            "fillna": fillna,
        }


class HistoricValues(EmptyTransformer):
    """Overwrite (align) all forecast values with the nearest actual value in window (tail) of history.
    (affected by upstream transformers, as usual)

    Args:
        window (int): or None, the most recent n history to use for alignment
    """

    def __init__(
        self,
        window: int = None,
        **kwargs,
    ):
        super().__init__(name="HistoricValues")
        self.window = window

    def _fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """

        # I am not sure a copy is necessary, but certainly is safer
        if self.window is None:
            self.df = df
        else:
            self.df = df.tail(self.window).copy()

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
        # using loop because experience with vectorized has been high memory usage
        # also usually forecast length is relatively short
        result = []
        m_arr = np.asarray(self.df)
        for row in np.asarray(df):
            # find the closest historic value and select those values
            result.append(
                m_arr[np.abs(m_arr - row).argmin(axis=0), range(df.shape[1])][
                    ..., np.newaxis
                ]
            )

        return pd.DataFrame(
            np.concatenate(result, axis=1).T, index=df.index, columns=df.columns
        )

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self._fit(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        return {
            "window": random.choices(
                [None, 10, 28, 100, 364, 730],
                [0.6, 0.1, 0.1, 0.1, 0.1, 0.1],
            )[0],
        }


def bkfilter_st(x, low=6, high=32, K=12, lanczos_factor=False):
    """This code is mostly from Statsmodel's bkfilter function."""
    # input is array
    omega_1 = 2.0 * np.pi / high  # convert from freq. to periodicity
    omega_2 = 2.0 * np.pi / low
    bweights = np.zeros(2 * K + 1)
    bweights[K] = (omega_2 - omega_1) / np.pi  # weight at zero freq.
    j = np.arange(1, int(K) + 1)
    weights = 1 / (np.pi * j) * (np.sin(omega_2 * j) - np.sin(omega_1 * j))
    if lanczos_factor:
        lanczos_factors = np.sinc(2 * j / (2.0 * K + 1))
        weights *= lanczos_factors
    bweights[K + j] = weights  # j is an idx
    bweights[:K] = weights[::-1]  # make symmetric weights
    bweights -= bweights.mean()  # make sure weights sum to zero
    if x.ndim == 2:
        bweights = bweights[:, None]
    return fftconvolve(x, bweights, mode='valid')


class BKBandpassFilter(EmptyTransformer):
    """More complete implentation of Baxter King Bandpass Filter
    based off the successful but somewhat confusing statmodelsfilter transformer.

    Args:
        window (int): or None, the most recent n history to use for alignment
    """

    def __init__(
        self,
        low: int = 6,
        high: int = 32,
        K: int = 1,
        lanczos_factor: int = False,
        return_diff: int = True,
        on_transform: bool = True,
        on_inverse: bool = False,
        **kwargs,
    ):
        super().__init__(name="BKBandpassFilter")
        self.low = low
        self.high = high
        self.K = K
        self.lanczos_factor = lanczos_factor
        self.return_diff = return_diff
        self.on_transform = on_transform
        self.on_inverse = on_inverse

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

    def filter(self, df):
        cycles = bkfilter_st(
            np.asarray(df),
            low=self.low,
            high=self.high,
            K=self.K,
            lanczos_factor=self.lanczos_factor,
        )
        if self.return_diff:
            N = cycles.shape[0]
            start_index = (df.shape[0] - N) // 2
            end_index = start_index + N
            cycles = pd.DataFrame(
                cycles, index=df.index[start_index:end_index], columns=df.columns
            ).reindex(df.index, fill_value=0)
            return (df - cycles).ffill().bfill()
        else:
            # so the output is actually centered but using the tail axis for forecasting effectiveness
            return pd.DataFrame(
                cycles, columns=df.columns, index=df.index[-cycles.shape[0] :]
            )

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.on_transform:
            return self.filter(df)
        else:
            return df

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.on_inverse:
            return self.filter(df)
        else:
            return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self.transform(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        selection = random.choices([True, False], [0.8, 0.2])[0]
        return {
            "low": random.choices(
                [6, 4, 7, 12, 8, 28], [0.6, 0.1, 0.1, 0.1, 0.05, 0.05]
            )[0],
            "high": random.choices(
                [32, 40, 90, 28, 364, 728], [0.5, 0.1, 0.1, 0.1, 0.15, 0.05]
            )[0],
            "K": random.choices([1, 3, 6, 12, 25], [0.6, 0.1, 0.1, 0.1, 0.1])[0],
            "lanczos_factor": random.choices(
                [True, False],
                [0.2, 0.8],
            )[0],
            "return_diff": random.choices([True, False], [0.7, 0.3])[0],
            'on_transform': selection,
            "on_inverse": not selection,
        }


class Constraint(EmptyTransformer):
    """Apply constraints (caps on values based on history).

    See base.py constraints function for argument documentation
    """

    def __init__(
        self,
        constraint_method: int = "historic_growth",
        constraint_value: int = 1.0,
        constraint_direction: str = "upper",
        constraint_regularization: int = 1.0,
        forecast_length: int = None,
        bounds_only: bool = False,
        fillna: str = None,
        **kwargs,
    ):
        super().__init__(name="Constraint")
        self.constraint_method = constraint_method
        self.constraint_value = constraint_value
        self.constraint_direction = constraint_direction
        self.constraint_regularization = constraint_regularization
        self.forecast_length = forecast_length
        self.bounds_only = bounds_only
        self.fillna = fillna
        self.adjustment = None

    def fit(self, df):
        """Learn behavior of data to change.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        (
            self.lower_constraint,
            self.upper_constraint,
            self.train_min,
            self.train_max,
        ) = fit_constraint(
            constraint_method=self.constraint_method,
            constraint_value=self.constraint_value,
            constraint_direction=self.constraint_direction,
            constraint_regularization=self.constraint_regularization,
            bounds=False,
            df_train=df,
            forecast_length=self.forecast_length,
        )
        return self

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return df

    def inverse_transform(self, df, trans_method: str = "forecast", adjustment=None):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if trans_method == "original":
            return df
        # reusing the adjustments style arg from alignlastvalue for determining if bounds
        if not self.bounds_only or (self.bounds_only and adjustment is not None):
            forecast, up, low = apply_fit_constraint(
                forecast=df,
                lower_forecast=0,
                upper_forecast=0,
                constraint_method=self.constraint_method,
                constraint_value=self.constraint_value,
                constraint_direction=self.constraint_direction,
                constraint_regularization=self.constraint_regularization,
                bounds=False,
                lower_constraint=self.lower_constraint,
                upper_constraint=self.upper_constraint,
                train_min=self.train_min,
                train_max=self.train_max,
                fillna=self.fillna,
            )
            return forecast
        else:
            # if point forecast, don't do anything for bounds_only
            self.adjustment = True
            return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return df

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        params = constraint_new_params(method=method)
        params["bounds_only"] = random.choices([True, False], [0.2, 0.8])[0]
        params['fillna'] = random.choices(
            [None, "ffill", "linear"], [0.95, 0.05, 0.05]
        )[0]
        return params


class FIRFilter(EmptyTransformer):
    """Scipy firwin"""

    def __init__(
        self,
        sampling_frequency: int = 365,
        numtaps: int = 512,
        cutoff_hz: float = 30,
        window: str = "hamming",
        on_transform: bool = True,
        on_inverse: bool = False,
        **kwargs,
    ):
        super().__init__(name="FIRFilter")
        self.sampling_frequency = sampling_frequency
        self.numtaps = numtaps
        self.cutoff_hz = cutoff_hz
        self.window = window
        self.on_transform = on_transform
        self.on_inverse = on_inverse

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

    def filter(self, df):
        return pd.DataFrame(
            fft_fir_filter_to_timeseries(
                df.to_numpy(),
                sampling_frequency=self.sampling_frequency,
                numtaps=self.numtaps,
                cutoff_hz=self.cutoff_hz,
                window=self.window,
            ),
            index=df.index,
            columns=df.columns,
        )

    def transform(self, df):
        """Return changed data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.on_transform:
            return self.filter(df)
        else:
            return df

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """Return data to original *or* forecast form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        if self.on_inverse:
            return self.filter(df)
        else:
            return df

    def fit_transform(self, df):
        """Fits and Returns *Magical* DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        return self.transform(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        selection = random.choices([True, False], [0.8, 0.2])[0]
        params = generate_random_fir_params(method=method)
        params["sampling_frequency"] = seasonal_int(include_one=False)
        params["on_transform"] = selection
        params["on_inverse"] = not selection
        return params


class ThetaTransformer:
    def __init__(self, theta_values=[0, 2], regularization=1e-3, verbose=0):
        """
        ThetaTransformer decomposes a time series into theta lines based on the Theta method.

        Args:
            theta_values (list): List of theta coefficients to use in transformation.
            regularization (float): Regularization term for numerical stability.
            verbose (int): Verbosity level for logging.
        """
        self.theta_values = theta_values
        self.regularization = regularization
        self.verbose = verbose
        self.beta = None
        self.columns = None
        self.index = None
        self.t0 = None
        self.t_scale = None

    def _compute_time_variables(self, index):
        """
        Compute the time variable t based on the index.

        Parameters:
            index (pd.DatetimeIndex): Index for which to compute t.

        Returns:
            np.ndarray: Time variables scaled between 0 and 1.
        """
        t = (index - self.t0).total_seconds().values.reshape(-1, 1)
        # Handle the case where total_seconds is not available
        if not hasattr(t, 'reshape'):
            t = ((index.astype('int64') - self.t0.value) / 1e9).reshape(-1, 1)
        # Scale t using the original scaling factor
        t = t / self.t_scale
        return t

    def fit(self, df):
        """
        Fit the transformer to the data.

        Parameters:
            df (pd.DataFrame): DataFrame with DatetimeIndex and columns representing time series.

        Returns:
            self
        """
        self.columns = df.columns
        self.index = df.index
        n = len(df)
        self.t0 = df.index[0]
        self.t_scale = (df.index[-1] - self.t0).total_seconds()
        if self.t_scale == 0:
            self.t_scale = 1  # Avoid division by zero for constant time index

        # Compute time variable t
        t = self._compute_time_variables(df.index)
        X = np.hstack((np.ones((n, 1)), t))  # n x 2
        self.X = X

        # Compute beta coefficients for the LRL using least squares
        y = df.values  # n x m
        self.beta = np.linalg.lstsq(X, y, rcond=None)[0]  # 2 x m

        return self

    def transform(self, df):
        """
        Transform the data into theta lines.

        Parameters:
            df (pd.DataFrame): DataFrame with same index and columns as fitted.

        Returns:
            pd.DataFrame: Transformed DataFrame containing theta lines.
        """
        y = df.values  # n x m
        n = len(df)

        # Compute time variable t for the given index
        t = self._compute_time_variables(df.index)
        X = np.hstack((np.ones((n, 1)), t))  # n x 2

        # Compute LRL_t (linear regression line) for the given index
        LRL_t = X @ self.beta  # (n x 2) @ (2 x m) = n x m

        # Compute Theta lines for each theta in theta_values
        theta_lines = []
        transformed_columns = []

        for theta in self.theta_values:
            # Corrected theta line formula
            theta_line = LRL_t + theta * (y - LRL_t)  # n x m
            theta_lines.append(theta_line)
            # Create column names for this theta
            columns_theta = [f"{col}_theta{theta}" for col in self.columns]
            transformed_columns.extend(columns_theta)

        # Stack all theta lines horizontally
        transformed_data = np.hstack(theta_lines)  # n x (m * len(theta_values))

        transformed_df = pd.DataFrame(
            transformed_data, index=df.index, columns=transformed_columns
        )
        return transformed_df

    def fit_transform(self, df):
        """
        Fit the transformer to the data and then transform it.

        Parameters:
            df (pd.DataFrame): DataFrame to fit and transform.

        Returns:
            pd.DataFrame: Transformed DataFrame containing theta lines.
        """
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df, trans_method: str = "forecast"):
        """
        Reconstruct the original data from theta lines.

        Parameters:
            df (pd.DataFrame): DataFrame with theta lines.

        Returns:
            pd.DataFrame: Reconstructed DataFrame in the original feature space.
        """
        m = len(self.columns)
        # n_theta = len(self.theta_values)

        # Extract theta lines from the transformed data
        theta_lines = []
        for i, theta in enumerate(self.theta_values):
            start_col = i * m
            end_col = (i + 1) * m
            theta_line = df.iloc[:, start_col:end_col].values  # n x m
            theta_lines.append(theta_line)

        y_reconstructed = np.mean(theta_lines, axis=0)  # n x m
        # Use weights to reconstruct the original data
        # weights = np.ones(n_theta) / n_theta
        # y_reconstructed = np.tensordot(weights, theta_lines, axes=([0], [0]))  # n x m

        reconstructed_df = pd.DataFrame(
            y_reconstructed, index=df.index, columns=self.columns
        )
        return reconstructed_df

    @staticmethod
    def get_new_params(method: str = "random"):
        return {
            "theta_values": random.choice(
                [
                    [0, 2],
                    [0.5, 1.5],
                    [0.2, 1.8],
                    [0.4, 1.6],
                    [0.6, 1.4],
                    [0.8, 1.2],
                    [0, 1, 2],
                    [0, 0.5, 1.5, 2],
                ]
            ),
        }


class ChangepointDetrend(Detrend):
    """Remove trend using changepoint features linked to a specific datetime origin."""

    def __init__(
        self,
        model: str = "Linear",
        changepoint_spacing: int = 60,
        changepoint_distance_end: int = 120,
        datepart_method: str = None,
        **kwargs,
    ):
        super().__init__(name="ChangepointDetrend")
        self.model = model
        self.changepoint_spacing = changepoint_spacing
        self.changepoint_distance_end = changepoint_distance_end
        self.datepart_method = datepart_method

    @staticmethod
    def get_new_params(method: str = "random"):
        if method == "fast":
            choice = random.choices(
                ["Linear", "Ridge", "ElasticNet"], [0.5, 0.2, 0.2], k=1
            )[0]
            # phi = random.choices([1, 0.999, 0.998, 0.99], [0.9, 0.05, 0.01, 0.01])[0]
        else:
            choice = random.choices(
                [
                    "Linear",
                    "Poisson",
                    "Tweedie",
                    "Gamma",
                    "Ridge",
                    "ElasticNet",
                ],
                [0.4, 0.1, 0.1, 0.1, 0.1, 0.1],
                k=1,
            )[0]
            # phi = random.choices([1, 0.999, 0.998, 0.99], [0.9, 0.1, 0.05, 0.05])[0]
        datepart_method = random.choices([None, "something"], [0.5, 0.5])[0]
        if datepart_method is not None:
            datepart_method = random_datepart(method=method)
        return {
            "model": choice,
            # "phi": phi,
            "changepoint_spacing": random.choices(
                [None, 6, 28, 60, 90, 120, 180, 360, 5040],
                [0.1, 0.05, 0.1, 0.1, 0.1, 0.05, 0.2, 0.1, 0.2],
            )[0],
            "changepoint_distance_end": random.choices(
                [None, 6, 28, 60, 90, 180, 360, 520, 5040],
                [0.1, 0.05, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.2],
            )[0],
            "datepart_method": datepart_method,
        }

    def _create_X(self, DTindex, datepart_method=None):
        """
        Create changepoint features for given datetime index and stored changepoint dates.

        Parameters:
        DTindex (pd.DatetimeIndex): datetime index of the data

        Returns:
        pd.DataFrame: DataFrame containing changepoint features for linear regression.
        """
        # Compute time differences between DTindex and each changepoint date
        DTindex_values = DTindex.values.astype('datetime64[ns]')
        cp_dates_values = self.changepoint_dates.values.astype('datetime64[ns]')
        # Compute time differences in days
        time_diffs = (DTindex_values[:, None] - cp_dates_values[None, :]).astype(
            'timedelta64[s]'
        ) / np.timedelta64(1, 'D')
        # Apply np.maximum(0, time_diff)
        features = np.maximum(0, time_diffs)
        # Create DataFrame
        feature_names = [
            f'changepoint_{i+1}' for i in range(len(self.changepoint_dates))
        ]
        changepoint_features = pd.DataFrame(
            features, index=DTindex, columns=feature_names
        )
        if datepart_method is not None:
            x_s = date_part(DTindex, method=datepart_method, set_index=True)
            return pd.concat([changepoint_features, x_s], axis=1)
        return changepoint_features

    def fit(self, df):
        """Fits trend for later detrending using changepoint features.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")
        Y = df.copy()
        DTindex = df.index
        n = len(DTindex)
        self.origin_datetime = DTindex[0]
        if self.changepoint_spacing is None or self.changepoint_distance_end is None:
            half_yr_space = half_yr_spacing(df)
            if self.changepoint_spacing is None:
                self.changepoint_spacing = int(half_yr_space)
            if self.changepoint_distance_end is None:
                self.changepoint_distance_end = int(half_yr_space / 2)
        # Compute changepoint positions as indices
        changepoint_range_end = n - self.changepoint_distance_end
        # Adjust for cases where changepoint_distance_end >= n
        if changepoint_range_end <= 0:
            # Set changepoint_positions to include only the start position
            self.changepoint_positions = np.array([0])
        else:
            self.changepoint_positions = np.arange(
                0, changepoint_range_end, self.changepoint_spacing
            )
            self.changepoint_positions = np.append(
                self.changepoint_positions, changepoint_range_end
            )
        # Get the datetime values at the changepoint positions
        self.changepoint_dates = DTindex[self.changepoint_positions]
        # Generate changepoint features
        x_t = self._create_X(DTindex, datepart_method=self.datepart_method)
        # Fit the regression model
        self.trained_model = self._retrieve_detrend(
            detrend=self.model, multioutput=df.shape[1] > 1
        )
        self.trained_model.fit(x_t, Y)
        self.shape = df.shape
        return self

    def transform(self, df):
        """Return detrended data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")
        DTindex = df.index
        x_t = self._create_X(DTindex, datepart_method=self.datepart_method)
        # Predict trend
        trend = pd.DataFrame(
            self.trained_model.predict(x_t), index=DTindex, columns=df.columns
        )
        # Detrend data
        df_detrended = df - trend
        return df_detrended

    def inverse_transform(self, df):
        """Return data to original form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        DTindex = df.index
        x_t = self._create_X(DTindex, datepart_method=self.datepart_method)
        # Predict trend
        trend = pd.DataFrame(
            self.trained_model.predict(x_t), index=DTindex, columns=df.columns
        )
        # Add trend back to data
        df_original = df + trend
        return df_original

    def fit_transform(self, df):
        """Fit and return detrended data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)


class MeanPercentSplitter(EmptyTransformer):
    """Splits data into rolling means and percentages. Designed to help with intermittent demand forecasting.

    Args:
        window (int or str): Rolling window size. If 'forecast_length', uses forecast_length as window.
        forecast_length (int): The forecast length to use if window == 'forecast_length'.
    """

    def __init__(self, window=10, forecast_length=None, **kwargs):
        super().__init__(name="MeanPercentSplitter")
        self.window = window
        self.forecast_length = forecast_length

    def fit(self, df):
        """Fit the transformer to the data.

        Args:
            df (pandas.DataFrame): Input DataFrame with pd.DatetimeIndex.
        """
        self.columns = df.columns

        # Determine the rolling window size
        if self.window == "forecast_length":
            if self.forecast_length is None:
                raise ValueError(
                    "forecast_length must be provided when window == 'forecast_length'"
                )
            self.window_size = self.forecast_length
        else:
            self.window_size = int(self.window)

        # Store the index for comparison in inverse_transform
        self.fit_index_max = df.index.max()

        return self

    def transform(self, df):
        """Transform the data by splitting into rolling means and percentages.

        Args:
            df (pandas.DataFrame): Input DataFrame with pd.DatetimeIndex.
        """
        if self.window == "forecast_length":
            window_size = self.forecast_length
        else:
            window_size = int(self.window)

        rolling_means = df.rolling(window=window_size, min_periods=1).mean()
        percentages = df / rolling_means.replace(0, 1)

        # Rename columns to distinguish between means and percentages
        # the X in there is to try to assure uniqueness from input column names
        mean_cols = [f"{col}_Xmean" for col in df.columns]
        percentage_cols = [f"{col}_Xpercentage" for col in df.columns]

        rolling_means.columns = mean_cols
        percentages.columns = percentage_cols

        return pd.concat([rolling_means, percentages], axis=1)

    def inverse_transform(self, df):
        """Inverse transform the data back to original space.

        Args:
            df (pandas.DataFrame): Transformed DataFrame with rolling means and percentages.
        """
        mean_cols = [f"{col}_Xmean" for col in self.columns]
        percentage_cols = [f"{col}_Xpercentage" for col in self.columns]

        rolling_means = df[mean_cols]
        percentages = df[percentage_cols]

        original_values = rolling_means.to_numpy() * percentages.to_numpy()

        original_df = pd.DataFrame(
            original_values, index=df.index, columns=self.columns
        )

        # Normalize by the final value if that is the mean of full forecast
        if self.window == "forecast_length":
            # Determine if inverse_transform is on future data
            if original_df.index.min() > self.fit_index_max:
                # Additional normalization step
                # Use the final value of the rolling mean components (from fit) to normalize the final value,
                # so that the mean of the inverse transformed components for each time series is equal to that final rolling mean value.

                # Compute mean of the inverse transformed components for each time series
                mean_values = original_df.mean()

                # Compute normalization factor
                finale = rolling_means.iloc[-1, :]
                finale.index = self.columns
                normalization_factor = finale / mean_values.replace(0, 1)

                # Multiply original_df by normalization_factor
                original_df = original_df.multiply(normalization_factor, axis=1)

        return original_df

    def fit_transform(self, df):
        """Fit to data, then transform it.

        Args:
            df (pandas.DataFrame): Input DataFrame with pd.DatetimeIndex.
        """
        self.fit(df)
        return self.transform(df)

    @staticmethod
    def get_new_params(method: str = "random"):
        """Generate new random parameters"""
        params = {
            "window": random.choice([3, 7, 10, 24, "forecast_length"]),
        }
        return params


class StandardScaler:
    def __init__(self):
        self.means = None
        self.stds = None
        self.skip_columns = None

    def fit(self, df: pd.DataFrame):
        """Compute the mean and standard deviation for each feature."""
        self.means = df.mean()
        self.stds = df.std(ddof=0).replace(
            0, 1
        )  # Use population standard deviation (ddof=0)
        # Identify columns to skip (constant or zero std)
        self.skip_columns = (
            self.stds == 1
        )  # 0 replace with 1, exact 1 unlikely in real data

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale the dataset using the stored mean and standard deviation."""
        X_copy = df.copy()  # Create a safe copy of the DataFrame
        # print(self.means.index.difference(df.columns))
        # print(df.columns.difference(self.stds.index))
        X_scaled = (X_copy - self.means) / self.stds
        # Restore original values for columns that should not be scaled
        X_scaled.loc[:, self.skip_columns] = X_copy.loc[:, self.skip_columns]
        return X_scaled

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Revert the scaled data back to the original scale."""
        X_copy = df.copy()  # Create a safe copy of the DataFrame
        X_original = (X_copy * self.stds) + self.means
        # Restore original values for columns that were not scaled
        X_original.loc[:, self.skip_columns] = X_copy.loc[:, self.skip_columns]
        return X_original

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler and transform the dataset."""
        self.fit(df)
        return self.transform(df)


# lookup dict for all non-parameterized transformers
trans_dict = {
    "None": EmptyTransformer(),
    None: EmptyTransformer(),
    "RollingMean10": RollingMeanTransformer(window=10),
    # "DifferencedTransformer": DifferencedTransformer(),
    "PctChangeTransformer": PctChangeTransformer(),
    "SinTrend": SinTrend(),
    "SineTrend": SinTrend(),
    "PositiveShift": PositiveShift(squared=False),
    "Log": PositiveShift(log=True),
    "CumSumTransformer": CumSumTransformer(),
    "SeasonalDifference7": SeasonalDifference(lag_1=7, method="LastValue"),
    "SeasonalDifference12": SeasonalDifference(lag_1=12, method="Mean"),
    "SeasonalDifference28": SeasonalDifference(lag_1=28, method="Mean"),
    "bkfilter": StatsmodelsFilter(method="bkfilter"),
    "cffilter": StatsmodelsFilter(method="cffilter"),
    "convolution_filter": StatsmodelsFilter(method="convolution_filter"),
    "Discretize": Discretize(discretization="center", n_bins=10),
    "DatepartRegressionLtd": DatepartRegressionTransformer(
        regression_model={
            "model": "DecisionTree",
            "model_params": {"max_depth": 4, "min_samples_split": 2},
        },
        datepart_method="recurring",
    ),
    "DatepartRegressionElasticNet": DatepartRegressionTransformer(
        regression_model={"model": "ElasticNet", "model_params": {}}
    ),
    "DatepartRegressionRandForest": DatepartRegressionTransformer(
        regression_model={"model": "RandomForest", "model_params": {}}
    ),
    "MeanDifference": MeanDifference(),
}
# have n_jobs
n_jobs_trans = {
    # datepart not included for fears it will slow it down sometimes
    "SinTrend": SinTrend(),
    "SineTrend": SinTrend(),
    "AnomalyRemoval": AnomalyRemoval,
    'HolidayTransformer': HolidayTransformer,
    'ReplaceConstant': ReplaceConstant,
    # 'DiffSmoother': DiffSmoother,
}
# transformers with parameter pass through (internal only) MUST be here
have_params = {
    "RollingMeanTransformer": RollingMeanTransformer,
    "SeasonalDifference": SeasonalDifference,
    "Discretize": Discretize,
    "CenterLastValue": CenterLastValue,
    "ShiftFirstValue": ShiftFirstValue,
    "IntermittentOccurrence": IntermittentOccurrence,
    "ClipOutliers": ClipOutliers,
    "Round": Round,
    "Slice": Slice,
    "Detrend": Detrend,
    "ScipyFilter": ScipyFilter,
    "HPFilter": HPFilter,
    "STLFilter": STLFilter,
    "EWMAFilter": EWMAFilter,
    "FastICA": FastICA,
    "PCA": PCA,
    "BTCD": BTCD,
    "Cointegration": Cointegration,
    "AlignLastValue": AlignLastValue,
    "AnomalyRemoval": AnomalyRemoval,  # not shared as long as output is 'multivariate'
    "HolidayTransformer": HolidayTransformer,
    "LocalLinearTrend": LocalLinearTrend,
    "KalmanSmoothing": KalmanSmoothing,
    "RegressionFilter": RegressionFilter,
    "DatepartRegression": DatepartRegressionTransformer,
    "DatepartRegressionTransformer": DatepartRegressionTransformer,
    "LevelShiftMagic": LevelShiftMagic,
    "LevelShiftTransformer": LevelShiftTransformer,
    "CenterSplit": CenterSplit,
    "FFTFilter": FFTFilter,
    "FFTDecomposition": FFTDecomposition,
    "ReplaceConstant": ReplaceConstant,
    "AlignLastDiff": AlignLastDiff,
    "DiffSmoother": DiffSmoother,
    "HistoricValues": HistoricValues,
    "BKBandpassFilter": BKBandpassFilter,
    "DifferencedTransformer": DifferencedTransformer,
    "Constraint": Constraint,
    "FIRFilter": FIRFilter,
    "ThetaTransformer": ThetaTransformer,
    "ChangepointDetrend": ChangepointDetrend,
    "MeanPercentSplitter": MeanPercentSplitter,
}
# where results will vary if not all series are included together
shared_trans = [
    "PCA",
    "FastICA",
    "DatepartRegression",
    "MeanDifference",
    "BTCD",
    "Cointegration",
    "HolidayTransformer",  # confirmed
    "RegressionFilter",
]
# transformers not defined in AutoTS
external_transformers = [
    "MinMaxScaler",
    "PowerTransformer",
    "QuantileTransformer",
    "MaxAbsScaler",
    "StandardScaler",
    "RobustScaler",
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
            'ClipOutliers' - simple remove outliers
            'Discretize' - bin or round data into groups
            'DatepartRegression' - move a trend trained on datetime index
            "ScipyFilter" - filter data (lose information but smoother!) from scipy
            "HPFilter" - statsmodels hp_filter
            "STLFilter" - seasonal decompose and keep just one part of decomposition
            "EWMAFilter" - use an exponential weighted moving average to smooth data
            "MeanDifference" - joint version of differencing
            "Cointegration" - VECM but just the vectors
            "BTCD" - Box Tiao decomposition
            'AlignLastValue': align forecast start to end of training data
            'AnomalyRemoval': more tailored anomaly removal options
            'HolidayTransformer': detects holidays and wishes good cheer to all
            'LocalLinearTrend': rolling local trend, using tails for future and past trend
            'KalmanSmoothing': smooth using a state space model
            'RegressionFilter': fit seasonal removal and local linear trend, clip std devs away from this fit
            'LevelShiftTransformer': automatically compensate for historic level shifts in data.
            'CenterSplit': Croston inspired magnitude/occurrence split for intermittent
            "FFTFilter": filter using a fast fourier transform
            "FFTDecomposition": remove FFT harmonics, later add back
            "ReplaceConstant": replace a value with NaN, optionally fillna then later reintroduce
            "AlignLastDiff": shift forecast to be within range of historical diffs
            "DiffSmoother": smooth diffs then return to original space
            "HistoricValues": match predictions to most similar historic value and overwrite
            "BKBandpassFilter": another version of the Baxter King bandpass filter
            "Constraint": apply constraints (caps) on values
            "FIRFilter": apply a FIR filter (firwin)
            "ShiftFirstValue": similar to positive shift but uses the first values as the basis of zero
            "ThetaTransformer": decomposes into theta lines, then recombines
            "ChangepointDetrend": detrend but with changepoints, and seasonality thrown in for fun
            "MeanPercentSplitter": split data into rolling mean and percent of rolling mean

        transformation_params (dict): params of transformers {0: {}, 1: {'model': 'Poisson'}, ...}
            pass through dictionary of empty dictionaries to utilize defaults

        random_seed (int): random state passed through where applicable
        forecast_length (int): length of forecast, not needed as argument for most transformers/params
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
        n_jobs: int = 1,
        holiday_country: list = None,
        verbose: int = 0,
        forecast_length: int = 30,
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
        self.n_jobs = n_jobs
        self.holiday_country = holiday_country
        self.verbose = verbose
        self.forecast_length = forecast_length
        self.transformers = {}
        self.adjustments = {}
        # upper/lower forecast inverses are different
        self.bounded_oddities = ["AlignLastValue", "AlignLastDiff", "Constraint"]
        # trans methods are different
        self.oddities_list = [
            "DifferencedTransformer",
            "RollingMean100thN",
            "RollingMean10thN",
            "RollingMean10",
            "RollingMean",
            "RollingMeanTransformer",
            "PctChangeTransformer",
            "CumSumTransformer",
            "SeasonalDifference",
            "SeasonalDifferenceMean",
            "SeasonalDifference7",
            "SeasonalDifference12",
            "SeasonalDifference28",
            "MeanDifference",
            "AlignLastValue",
            "AlignLastDiff",
            "Constraint",
        ]

    @staticmethod
    def get_new_params(method="fast"):
        return RandomTransform(transformer_list=method)

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
            # attempting to find a bug, zero size array to min
            try:
                self.nan_flag = np.isnan(np.min(df.to_numpy()))
            except Exception as e:
                self.nan_flag = True
                print(repr(e))
                # print(df)
                # print(df.shape)
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
        n_jobs: int = 1,
        holiday_country: list = None,
        forecast_length: int = 30,
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

        elif transformation in n_jobs_trans.keys():
            return n_jobs_trans[transformation](n_jobs=n_jobs, **param)

        # these need holiday_country
        elif transformation in ["DatepartRegression", "DatepartRegressionTransformer"]:
            return DatepartRegression(
                holiday_country=holiday_country, **param
            )  # n_jobs=n_jobs,

        elif transformation in ["RegressionFilter"]:
            return RegressionFilter(
                holiday_country=holiday_country, n_jobs=n_jobs, **param
            )
        elif transformation in ["Constraint", "MeanPercentSplitter"]:
            return Constraint(forecast_length=forecast_length, **param)

        elif transformation == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler

            return MinMaxScaler()

        elif transformation == "PowerTransformer":
            from sklearn.preprocessing import PowerTransformer

            return PowerTransformer(method="yeo-johnson", standardize=True, copy=True)

        elif transformation == "QuantileTransformer":
            from sklearn.preprocessing import QuantileTransformer

            quants = param.get("n_quantiles", 1000)
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

        elif transformation == "StandardScaler":
            try:
                from sklearn.preprocessing import StandardScaler as SS

                return SS(copy=True)
            except Exception as e:
                print(f"sklearn standardscaler import failed with {repr(e)}")
                return StandardScaler()

        elif transformation == "MaxAbsScaler":
            from sklearn.preprocessing import MaxAbsScaler

            return MaxAbsScaler(copy=True)

        elif transformation == "RobustScaler":
            from sklearn.preprocessing import RobustScaler

            return RobustScaler(copy=True)

        elif False:  #  OLD  for transformation == "PCA"
            from sklearn.decomposition import PCA

            # could probably may it work, but this is simpler
            if df.shape[1] > df.shape[0]:
                raise ValueError("PCA fails when n series > n observations")
            return PCA(
                n_components=min(df.shape), whiten=False, random_state=random_seed
            )

        elif transformation == "FastICA":
            from sklearn.decomposition import FastICA

            if df.shape[1] > 5000:
                raise ValueError("FastICA fails with > 5000 series for speed reasons")
            return FastICA(
                n_components=df.shape[1],
                random_state=random_seed,
                **param,
            )

        elif transformation in ["RollingMean", "FixedRollingMean"]:
            param = 10 if param is None else param
            if not str(param).isdigit():
                window = int("".join([s for s in str(param) if s.isdigit()]))
                window = int(df.shape[0] / window)
            else:
                window = int(param)
            window = 2 if window < 2 else window
            self.window = window
            if transformation == "FixedRollingMean":
                transformer = RollingMeanTransformer(window=self.window, fixed=True)
            else:
                transformer = RollingMeanTransformer(window=self.window, fixed=False)
            return transformer

        elif transformation in ["SeasonalDifferenceMean"]:
            return SeasonalDifference(lag_1=param, method="Mean")

        elif transformation == "RollingMean100thN":
            window = int(df.shape[0] / 100)
            window = 2 if window < 2 else window
            self.window = window
            return RollingMeanTransformer(window=self.window)

        elif transformation == "RollingMean10thN":
            window = int(df.shape[0] / 10)
            window = 2 if window < 2 else window
            self.window = window
            return RollingMeanTransformer(window=self.window)

        # must be at bottom as it has duplicates of above inside
        elif transformation in list(have_params.keys()):
            return have_params[transformation](**param)

        else:
            print(
                f"Transformation {transformation} not known or improperly entered, returning untransformed df"
            )
            return EmptyTransformer()

    def _first_fit(self, df):
        # fill NaN
        df = self.fill_na(df)

        self.df_index = df.index
        self.df_colnames = df.columns
        return df

    def _fit_one(self, df, i):
        transformation = self.transformations[i]
        self.transformers[i] = self.retrieve_transformer(
            transformation=transformation,
            df=df,
            param=self.transformation_params[i],
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            holiday_country=self.holiday_country,
            forecast_length=self.forecast_length,
        )
        df = self.transformers[i].fit_transform(df)
        # convert to DataFrame only if it isn't already
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, index=self.df_index, columns=self.df_colnames)
        # update index reference if sliced
        if transformation in expanding_transformers:
            self.df_index = df.index
            self.df_colnames = df.columns
        # df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)
        return df

    def _fit(self, df):
        df = self._first_fit(df)

        try:
            for i in sorted(self.transformations.keys()):
                df = self._fit_one(df, i)
        except Exception as e:
            err_str = f"Transformer {self.transformations[i]} failed on fit"
            if self.verbose >= 1:
                err_str += f" from params {self.fillna} {self.transformation_params} with error {repr(e)}"
            raise Exception(err_str) from e
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

    def _transform_one(self, df, i):
        transformation = self.transformations[i]
        df = self.transformers[i].transform(df)
        # convert to DataFrame only if it isn't already
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, index=self.df_index, columns=self.df_colnames)
        # update index reference if sliced
        if transformation in expanding_transformers:
            self.df_index = df.index
            self.df_colnames = df.columns
        return df

    def transform(self, df):
        """Apply transformations to convert df."""
        df = df.copy()
        """
        if self.grouping is not None:
            df = self.hier.transform(df)
        """
        # fill NaN
        df = self._first_fit(df)
        # transformations
        i = 0
        for i in sorted(self.transformations.keys()):
            df = self._transform_one(df, i)
        # df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)
        return df

    def _inverse_one(self, df, i, trans_method='forecast', bounds=False):
        self.c_trans_n = self.transformations[i]
        if self.c_trans_n in self.oddities_list:
            if self.c_trans_n in self.bounded_oddities:
                if not bounds:
                    adjustment = None
                else:
                    adjustment = self.adjustments.get(i, None)
                df = self.transformers[i].inverse_transform(
                    df,
                    trans_method=trans_method,
                    adjustment=adjustment,
                )
                if not bounds:
                    self.adjustments[i] = self.transformers[i].adjustment
            else:
                df = self.transformers[i].inverse_transform(
                    df, trans_method=trans_method
                )
        else:
            df = self.transformers[i].inverse_transform(df)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, index=self.df_index, columns=self.df_colnames)
        elif self.c_trans_n in expanding_transformers:
            self.df_colnames = df.columns
        # df = df.replace([np.inf, -np.inf], 0)
        return df

    def inverse_transform(
        self,
        df,
        trans_method: str = "forecast",
        fillzero: bool = False,
        bounds: bool = False,
    ):
        """Undo the madness.

        Args:
            df (pandas.DataFrame): Datetime Indexed
            trans_method (str): 'forecast' or 'original' passed through
            fillzero (bool): if inverse returns NaN, fill with zero
            bounds (bool): currently ignores AlignLastValue transform if True (also used in process_components of Cassandra)
        """
        self.df_index = df.index
        self.df_colnames = df.columns
        # df = df.replace([np.inf, -np.inf], 0)  # .fillna(0)
        try:
            for i in sorted(self.transformations.keys(), reverse=True):
                df = self._inverse_one(df, i, trans_method=trans_method, bounds=bounds)
        except Exception as e:
            err_str = f"Transformer {self.c_trans_n} failed on inverse"
            if self.verbose >= 1:
                err_str += f" from params {self.fillna} {self.transformation_params} with {repr(e)}"
            raise Exception(err_str) from e

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
    "MinMaxScaler": 0.03,
    "PowerTransformer": 0.01,  # is noticeably slower at scale, if not tons
    "QuantileTransformer": 0.03,
    "MaxAbsScaler": 0.03,
    "StandardScaler": 0.04,
    "RobustScaler": 0.03,
    "PCA": 0.01,
    "FastICA": 0.01,
    "Detrend": 0.02,  # slow with some params, but that's handled in get_params
    "RollingMeanTransformer": 0.02,
    "RollingMean100thN": 0.01,  # old
    "DifferencedTransformer": 0.05,
    "SinTrend": 0.01,
    "PctChangeTransformer": 0.01,
    "CumSumTransformer": 0.02,
    "PositiveShift": 0.02,
    "Log": 0.01,
    "IntermittentOccurrence": 0.01,
    "SeasonalDifference": 0.06,
    "cffilter": 0.01,
    "bkfilter": 0.05,
    "convolution_filter": 0.001,
    "HPFilter": 0.01,
    "DatepartRegression": 0.01,
    "ClipOutliers": 0.03,
    "Discretize": 0.01,
    "CenterLastValue": 0.01,
    "ShiftFirstValue": 0.01,
    "Round": 0.02,
    "Slice": 0.02,
    "ScipyFilter": 0.02,
    "STLFilter": 0.01,
    "EWMAFilter": 0.02,
    "MeanDifference": 0.002,
    "BTCD": 0.01,
    "Cointegration": 0.01,
    "AlignLastValue": 0.2,
    "AnomalyRemoval": 0.03,
    'HolidayTransformer': 0.01,
    'LocalLinearTrend': 0.01,
    'KalmanSmoothing': 0.02,
    'RegressionFilter': 0.01,
    "LevelShiftTransformer": 0.03,
    "CenterSplit": 0.01,
    "FFTFilter": 0.01,
    "FFTDecomposition": 0.01,
    "ReplaceConstant": 0.02,
    "AlignLastDiff": 0.01,
    "DiffSmoother": 0.005,
    "HistoricValues": 0.01,
    "BKBandpassFilter": 0.01,
    "Constraint": 0.01,  # 52
    "FIRFilter": 0.01,
    "ThetaTransformer": 0.01,
    "ChangepointDetrend": 0.01,
    "MeanPercentSplitter": 0.01,
}

# and even more, not just removing slow but also less commonly useful ones
# also there should be no 'shared' transformers in this list to make h-ensembles faster
superfast_transformer_dict = {
    None: 0.0,
    "MinMaxScaler": 0.05,
    "MaxAbsScaler": 0.05,
    "StandardScaler": 0.04,
    "RobustScaler": 0.05,
    "Detrend": 0.05,
    "RollingMeanTransformer": 0.02,
    "DifferencedTransformer": 0.05,
    "PositiveShift": 0.02,
    "Log": 0.01,
    "SeasonalDifference": 0.05,
    "bkfilter": 0.05,
    "ClipOutliers": 0.05,
    # "Discretize": 0.01,  # excessive memory use for some of this
    "Slice": 0.02,
    "EWMAFilter": 0.01,
    "AlignLastValue": 0.05,
    "AlignLastDiff": 0.05,
    "HistoricValues": 0.005,  # need to test more
    "CenterSplit": 0.0005,  # need to test more
    "Round": 0.01,
    "CenterLastValue": 0.01,
    "ShiftFirstValue": 0.005,
    "Constraint": 0.005,  # not well tested yet on speed/ram
    "BKBandpassFilter": 0.01,  # seems feasible, untested
    "DiffSmoother": 0.005,  # seems feasible, untested
    # "FIRFilter": 0.005,  # seems feasible, untested
    # "FFTFilter": 0.01,  # seems feasible, untested
    # "FFTDecomposition": 0.01,  # seems feasible, untested
}
# Split tranformers by type
# filters that remain near original space most of the time
filters = {
    'ScipyFilter': 0.1,
    # "RollingMeanTransformer": 0.1,  # makes for a mess with components if fixed=False
    "EWMAFilter": 0.1,
    "bkfilter": 0.1,
    "Slice": 0.01,  # sorta horizontal filter
    "AlignLastValue": 0.15,
    "KalmanSmoothing": 0.05,
    "ClipOutliers": 0.1,
    "RegressionFilter": 0.005,
    "FFTFilter": 0.01,
    "BKBandpassFilter": 0.005,
    "FIRFilter": 0.01,
    "AnomalyRemoval": 0.01,
    "RollingMeanTransformer": 0.005,
    "cffilter": 0.005,
    "HPFilter": 0.005,
    "RollingMean100thN": 0.005,
    "DiffSmoother": 0.005,
    "convolution_filter": 0.005,
}
scalers = {
    "MinMaxScaler": 0.05,
    "MaxAbsScaler": 0.05,
    "StandardScaler": 0.05,
    "RobustScaler": 0.05,
    "Log": 0.03,
    "Discretize": 0.01,
    "QuantileTransformer": 0.1,
    "PowerTransformer": 0.02,
    "PctChangeTransformer": 0.005,
    "CenterLastValue": 0.005,
}
# intended to clean up external regressors
decompositions = {
    "STLFilter": 0.05,
    "Detrend": 0.05,
    "DifferencedTransformer": 0.05,  # not really a decomposition
    "DatepartRegression": 0.05,
    "LocalLinearTrend": 0.03,
    "FFTDecomposition": 0.02,
    "SeasonalDifference": 0.01,
    "CenterSplit": 0.001,
    "HolidayTransformer": 0.01,
    "IntermittentOccurrence": 0.005,
    "PCA": 0.005,
    "ThetaTransformer": 0.005,
    "ChangepointDetrend": 0.01,
    "MeanPercentSplitter": 0.01,
}
postprocessing = {
    "Round": 0.1,
    "HistoricValues": 0.1,
    "BKBandpassFilter": 0.1,
    "KalmanSmoothing": 0.1,
    "AlignLastDiff": 0.1,
    "AlignLastValue": 0.1,
    "Constraint": 0.1,
    "FIRFilter": 0.1,
}
# transformers that may change the number of columns/index
expanding_transformers = [
    "Slice",
    "FastICA",
    "PCA",
    "CenterSplit",
    "RollingMeanTransformer",
    "LocalLinearTrend",
    "ThetaTransformer",
    "MeanPercentSplitter",
]  # note there is also prob_trans below for preventing reuse of these in one transformer

transformer_class = {}

# probability dictionary of FillNA methods
na_probs = {
    "ffill": 0.4,
    "fake_date": 0.1,
    "rolling_mean": 0.1,
    "rolling_mean_24": 0.1,
    "IterativeImputer": 0.025,  # this parallelizes, uses much memory
    "mean": 0.06,
    "zero": 0.05,
    "ffill_mean_biased": 0.1,
    "median": 0.03,
    None: 0.001,
    "interpolate": 0.4,
    "KNNImputer": 0.02,  # can get a bit slow
    "IterativeImputerExtraTrees": 0.0001,  # and this one is even slower
    "SeasonalityMotifImputer": 0.005,  # apparently this is too memory hungry at scale
    "SeasonalityMotifImputerLinMix": 0.005,  # apparently this is too memory hungry at scale
    "SeasonalityMotifImputer1K": 0.005,  # apparently this is too memory hungry at scale
    "DatepartRegressionImputer": 0.01,  # also slow
}


def transformer_list_to_dict(transformer_list):
    """Convert various possibilities to dict."""
    if transformer_list in ["fast", "default", "Fast", "auto", 'scalable']:
        # remove any slow transformers
        fast_transformer_dict = transformer_dict.copy()
        # downweight some
        fast_transformer_dict['ReplaceConstant'] = 0.002
        # del fast_transformer_dict["SinTrend"]
        del fast_transformer_dict["FastICA"]
        del fast_transformer_dict["Cointegration"]
        del fast_transformer_dict["BTCD"]
        del fast_transformer_dict["LocalLinearTrend"]
        del fast_transformer_dict["KalmanSmoothing"]  # potential kernel/RAM issues

    if transformer_list is None:
        transformer_list = "superfast"
    if not transformer_list or transformer_list == "all":
        transformer_list = transformer_dict
    elif transformer_list in ["fast", "default", "Fast", "auto", "fast_no_slice"]:
        transformer_list = fast_transformer_dict
    elif transformer_list in ["superfast", "superfast_no_slice"]:
        transformer_list = superfast_transformer_dict
    elif transformer_list in ["scalable", "scalable_no_slice"]:
        # "scalable" meant to be even smaller than "fast" subset of transformers
        transformer_list = fast_transformer_dict.copy()
        del transformer_list["SinTrend"]  # no observed issues, but for efficiency
        # del transformer_list["HolidayTransformer"]  # improved, should be good enough
        del transformer_list["ReplaceConstant"]
        del transformer_list["ThetaTransformer"]  # just haven't tested it enough yet
    elif "no_slice" in transformer_list:
        # slice can be a problem child in some cases, so can remove by adding this
        del transformer_list["Slice"]

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
    transformer_min_depth: int = 1,
    allow_none: bool = True,
    no_nan_fill: bool = False,
):
    """Return a dict of randomly choosen transformation selections.

    BTCD is used as a signal that slow parameters are allowed.
    """
    transformer_list, transformer_prob = transformer_list_to_dict(transformer_list)
    if transformer_max_depth <= 0:
        transformer_max_depth = 0
        transformer_min_depth = 0

    # adjust fast/slow based on Transformers allowed
    if fast_params is None:
        fast_params = True
        slow_flags = ["BTCD"]
        intersects = [i for i in slow_flags if i in transformer_list]
        if intersects:
            fast_params = False
    if superfast_params is None:
        superfast_params = False
        slow_flags = [
            "DatepartRegression",
            "ScipyFilter",
            "QuantileTransformer",
            "KalmanSmoothing",
        ]
        intersects = [i for i in slow_flags if i in transformer_list]
        if not intersects:
            superfast_params = True

    # filter na_probs if Fast
    params_method = None
    if fast_params or superfast_params:
        params_method = "fast"
        throw_away = na_prob_dict.pop("IterativeImputer", None)
        throw_away = df_interpolate.pop("spline", None)  # noqa
        throw_away = na_prob_dict.pop("IterativeImputerExtraTrees", None)  # noqa
        # throw_away = na_prob_dict.pop("SeasonalityMotifImputer1K", None)  # noqa
        # throw_away = na_prob_dict.pop("SeasonalityMotifImputerLinMix", None)  # noqa
        throw_away = na_prob_dict.pop("SeasonalityMotifImputer", None)  # noqa
        throw_away = na_prob_dict.pop("DatepartRegressionImputer", None)  # noqa
    # in addition to the above, also remove
    if superfast_params:
        params_method = "fast"
        throw_away = na_prob_dict.pop("KNNImputer", None)  # noqa
        throw_away = na_prob_dict.pop("SeasonalityMotifImputer1K", None)  # noqa
        throw_away = na_prob_dict.pop("SeasonalityMotifImputerLinMix", None)  # noqa

    # clean na_probs dict
    na_probabilities = list(na_prob_dict.values())
    na_probs_list = [*na_prob_dict]
    # sum_nas = sum(na_probabilities)
    # if sum_nas != 1:
    #     na_probabilities = [float(i) / sum_nas for i in na_probabilities]

    # choose FillNA
    if no_nan_fill:
        na_choice = None
    else:
        na_choice = random.choices(na_probs_list, na_probabilities)[0]
        if na_choice == "interpolate":
            na_choice = random.choices(
                list(df_interpolate.keys()), list(df_interpolate.values())
            )[0]

    # choose length of transformers
    num_trans = random.randint(transformer_min_depth, transformer_max_depth)
    # sometimes return no transformation
    if num_trans == 1 and allow_none:
        test = random.choices(["None", "Some"], [0.1, 0.9])[0]
        if test == "None":
            return {
                "fillna": na_choice,
                "transformations": {0: None},
                "transformation_params": {0: {}},
            }
    if traditional_order:
        # handle these not being in TransformerList
        randos = random.choices(transformer_list, transformer_prob, k=4)
        clip = "ClipOutliers" if "ClipOutliers" in transformer_list else randos[0]
        detrend = "Detrend" if "Detrend" in transformer_list else randos[1]
        # formerly Discretize
        discretize = (
            "AlignLastValue" if "AlignLastValue" in transformer_list else randos[2]
        )
        # create new dictionary in fixed order
        trans = [clip, detrend, randos[3], discretize]
        trans = trans[0:num_trans]
        num_trans = len(trans)
    else:
        trans = random.choices(transformer_list, transformer_prob, k=num_trans)

    # remove duplication of some which scale memory exponentially
    # only allow one of these
    prob_trans = {
        "CenterSplit",
        "RollingMeanTransformer",
        "LocalLinearTrend",
        "ThetaTransformer",
        "MeanPercentSplitter",
    }
    if any(x in prob_trans for x in trans):
        # for loop, only way I saw to do this right now
        seen = False
        result = []
        for item in trans:
            if item in prob_trans:
                if not seen:
                    seen = True
                    result.append(item)
            else:
                result.append(item)
        trans = result
        keys = list(range(len(trans)))
    else:
        keys = list(range(num_trans))
    # now get the parameters for the specified transformers
    params = [get_transformer_params(x, method=params_method) for x in trans]
    return {
        "fillna": na_choice,
        "transformations": dict(zip(keys, trans)),
        "transformation_params": dict(zip(keys, params)),
    }


def random_cleaners():
    """Returns transformation params that clean data without shifting."""
    transform_dict = random.choices(
        [
            None,
            "random",
            {
                "fillna": None,
                "transformations": {"0": "EWMAFilter"},
                "transformation_params": {
                    "0": {"span": 7},
                },
            },
            {
                "fillna": None,
                "transformations": {"0": "EWMAFilter"},
                "transformation_params": {
                    "0": {"span": 2},
                },
            },
            {
                "fillna": None,
                "transformations": {"0": "ScipyFilter"},
                "transformation_params": {
                    "0": {
                        'method': 'savgol_filter',
                        'method_args': {
                            'window_length': 31,
                            'polyorder': 3,
                            'deriv': 0,
                            'mode': 'interp',
                        },
                    },
                },
            },
            {
                "fillna": None,
                "transformations": {"0": "ClipOutliers"},
                "transformation_params": {
                    "0": {"method": "clip", "std_threshold": 4},
                },
            },
            {
                "fillna": None,
                "transformations": {"0": "bkfilter"},
                "transformation_params": {"0": {}},
            },
            {
                "fillna": None,
                "transformations": {"0": "Discretize"},
                "transformation_params": {
                    "0": {"discretization": "center", "n_bins": 20},
                },
            },
            {
                "fillna": None,
                "transformations": {"0": "AnomalyRemoval"},
                "transformation_params": {
                    "0": {
                        "method": "zscore",
                        "transform_dict": {
                            "transformations": {0: "DatepartRegression"},
                            "transformation_params": {
                                0: {
                                    "datepart_method": "simple_3",
                                    "regression_model": {
                                        "model": "ElasticNet",
                                        "model_params": {},
                                    },
                                }
                            },
                        },
                        "method_params": {
                            "distribution": "uniform",
                            "alpha": 0.05,
                        },
                    },
                },
            },
        ],
        [0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
    )[0]
    if transform_dict == "random":
        transform_dict = RandomTransform(
            transformer_list="scalable", transformer_max_depth=2
        )
    return transform_dict
