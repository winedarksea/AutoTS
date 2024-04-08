"""Arch Models from arch package."""

import datetime
import random
import warnings
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.seasonal import seasonal_int
from autots.tools.holiday import holiday_flag

try:
    from arch import arch_model

    arch_present = True
except Exception:
    arch_present = False
try:
    from joblib import Parallel, delayed

    joblib_present = True
except Exception:
    joblib_present = False

"""
from autots import load_artificial

random_seed = 12345
forecast_length = 5
verbose_bool = True

df = load_artificial(long=False)
am = arch_model(df.iloc[:, 0], vol='Garch', p=1, o=0, q=1, dist='Normal')

y = df["linear"].head(df.shape[0] - forecast_length)
x = df.iloc[:, 1: 3].head(df.shape[0] - forecast_length)
x_fore = df.iloc[:, 1: 3].tail(forecast_length)

am = arch_model(y)
res = am.fit(update_freq=0, disp="off", show_warning=verbose_bool)  # , starting_values=params)
params = res.params

# TARCH
am = arch_model(y, p=1, o=1, q=1, power=1.0)
res = am.fit(update_freq=5)

# different dist (from Normal)
am = arch_model(y, p=1, o=1, q=1, power=1.0, dist="StudentsT")
am = arch_model(y, x=x, mean="ARX", lags=1, power=1.0, q=0)

forecasts = res.forecast(horizon=1, reindex=False, random_state=random_seed)
forecasts = res.forecast(
    horizon=forecast_length, method='simulation',
    reindex=False,
    simulations=1000,
    x=x_fore.to_dict(orient="list")
)
forecasts.mean
forecasts.variance
np.percentile(forecasts.simulations.values, 50, axis=1)
"""


class ARCH(ModelObject):
    """ARCH model family from arch package. See arch package for arg details.
    Not to be confused with a linux distro.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): type of regression (None, 'User', or 'Holiday')
        n_jobs (int): passed to joblib for multiprocessing. Set to none for context manager.

    """

    def __init__(
        self,
        name: str = "ARCH",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        mean: str = "Constant",
        lags: int = 2,
        vol: str = "GARCH",
        p: int = 1,
        o: int = 0,
        q: int = 1,
        power: float = 2.0,
        dist: str = "normal",
        rescale: bool = False,
        maxiter: int = 200,
        simulations: int = 1000,
        regression_type: str = None,
        return_result_windows: bool = False,
        holiday_country: str = 'US',
        random_seed: int = 2022,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            regression_type=regression_type,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.mean = mean
        self.lags = lags
        self.vol = vol
        self.p = p
        self.o = o
        self.q = q
        self.power = power
        self.dist = dist
        self.rescale = rescale
        self.simulations = simulations
        self.maxiter = maxiter
        self.return_result_windows = return_result_windows

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied .

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        if not arch_present:
            raise ImportError("`arch` package must be installed from pip")

        df = self.basic_profile(df)
        self.regressor_train = None
        self.verbose_bool = False
        if self.verbose > 2:
            print("\U0001F3F9 ARCHers, draw")
            self.verbose_bool = True
        elif self.verbose > 1:
            self.verbose_bool = True
        if self.regression_type == 'holiday':
            self.regressor_train = pd.DataFrame(
                holiday_flag(df.index, country=self.holiday_country)
            )
        elif self.regression_type in ["User", "user"]:
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but future_regressor not supplied"
                )
            else:
                self.regressor_train = future_regressor.reindex(df.index)
        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()

        # defining this internally seems to make multiprocessing run better
        def arch_per_column(current_series, args):
            with warnings.catch_warnings():
                if args['verbose'] < 2:
                    warnings.simplefilter("ignore")
                if args['regression_type'] in ["User", "user", "holiday"]:
                    am = arch_model(
                        current_series,
                        self.regressor_train,
                        mean=self.mean,
                        lags=self.lags,
                        vol=self.vol,
                        p=self.p,
                        o=self.o,
                        q=self.q,
                        dist=self.dist,
                        power=self.power,
                        rescale=self.rescale,
                    )
                    res = am.fit(
                        update_freq=0,
                        disp="off",
                        show_warning=self.verbose_bool,
                        options={'maxiter': self.maxiter},
                    )
                    forecasts = res.forecast(
                        horizon=args['forecast_length'],
                        method='simulation',
                        reindex=False,
                        simulations=self.simulations,
                        x=args['future_regressor'].to_dict(orient="list"),
                        random_state=self.random_seed,
                    )
                else:
                    am = arch_model(
                        current_series,
                        mean=self.mean,
                        lags=self.lags,
                        vol=self.vol,
                        p=self.p,
                        o=self.o,
                        q=self.q,
                        dist=self.dist,
                        power=self.power,
                        rescale=self.rescale,
                    )
                    res = am.fit(
                        update_freq=0,
                        disp="off",
                        show_warning=self.verbose_bool,
                        options={'maxiter': self.maxiter},
                    )
                    forecasts = res.forecast(
                        horizon=args['forecast_length'],
                        method='simulation',
                        reindex=False,
                        simulations=self.simulations,
                        random_state=self.random_seed,
                    )

            if forecasts.mean.shape[0] == 1 and forecasts.mean.ndim > 1:
                cforecast = forecasts.mean.iloc[0]
            else:
                cforecast = forecasts.mean.iloc[:, 0]
            clower_forecast = pd.Series(
                np.quantile(forecasts.simulations.values, pred_range, axis=1).flatten(),
                name=current_series.name,
                index=args['test_index'],
            )
            cupper_forecast = pd.Series(
                np.quantile(
                    forecasts.simulations.values, 1 - pred_range, axis=1
                ).flatten(),
                name=current_series.name,
                index=args['test_index'],
            )
            cforecast.name = current_series.name
            cforecast.index = args['test_index']
            if self.return_result_windows:
                return (
                    cforecast,
                    clower_forecast,
                    cupper_forecast,
                    forecasts.simulations.values,
                )
            else:
                return (cforecast, clower_forecast, cupper_forecast)

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        pred_range = round((1 - self.prediction_interval) / 2, 3)
        if self.regression_type == 'holiday':
            future_regressor = pd.DataFrame(
                holiday_flag(test_index, country=self.holiday_country)
            )
        if self.regression_type is not None:
            assert (
                future_regressor.shape[0] == forecast_length
            ), "regressor not equal to forecast length"

        args = {
            'pred_range': pred_range,
            'forecast_length': forecast_length,
            'verbose': self.verbose,
            'test_index': test_index,
            'regression_type': self.regression_type,
            'future_regressor': future_regressor,
        }
        parallel = True
        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 5:
            parallel = False
        elif not joblib_present:
            parallel = False
        # joblib multiprocessing to loop through series
        if self.verbose > 2:
            print("\U0001F3F9 ARCHers, loose!")
        if parallel:
            verbs = 0 if self.verbose < 1 else self.verbose - 1
            df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                delayed(arch_per_column)(
                    current_series=self.df_train[col],
                    args=args,
                )
                for col in cols
            )
            complete = list(map(list, zip(*df_list)))
        else:
            df_list = []
            for col in cols:
                df_list.append(arch_per_column(self.df_train[col], args))
            complete = list(map(list, zip(*df_list)))
        forecast = pd.concat(complete[0], axis=1)
        lower_forecast = pd.concat(complete[1], axis=1)
        upper_forecast = pd.concat(complete[2], axis=1)
        if self.return_result_windows:
            self.result_windows = dict(zip(forecast.columns, complete[3]))

        if just_point_forecast:
            return forecast
        else:
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=test_index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        vol = random.choice(['GARCH', 'ARCH', 'EGARCH', 'FIGARCH', 'HARCH'])
        if vol == "FIGARCH":
            p = random.choice([0, 1])
            q = random.choice([0, 1])
        else:
            p = seasonal_int(include_one=True, very_small=True)
            q = random.choice([0, 1, 2])
        power = random.choice([1.0, 2.0, 1.5])
        dist = random.choice(['normal', 'studentst', 'skewt', 'ged'])
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_list = [None, 'User', 'holiday']
            regression_probability = [0.7, 0.2, 0.1]
            regression_choice = random.choices(regression_list, regression_probability)[
                0
            ]
        if regression_choice is None:
            mean_choice = random.choice(
                ['Constant', 'Zero', 'LS', 'AR', 'ARX', 'HAR', 'HARX']
            )
        else:
            mean_choice = random.choice(['ARX', 'HARX'])

        return {
            'mean': mean_choice,
            'lags': random.choice([0, 1, 2]),
            'vol': vol,
            'p': p,
            'o': random.choice([0, 1, 2]),
            'q': q,
            'power': power,
            'dist': dist,
            'rescale': bool(random.getrandbits(1)),
            'simulations': 1000,
            'maxiter': 200,
            'regression_type': regression_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'mean': self.mean,
            'lags': self.lags,
            'vol': self.vol,
            'p': self.p,
            'o': self.o,
            'q': self.q,
            'power': self.power,
            'dist': self.dist,
            'rescale': self.rescale,
            'simulations': self.simulations,
            'maxiter': self.maxiter,
            'regression_type': self.regression_type,
        }
