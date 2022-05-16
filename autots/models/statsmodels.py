"""Statsmodels based forecasting models.

Statsmodels documentation can be a bit confusing.
And it seems standard at first, but each model likes to do things differently.
For example: exog, exog_oos, and exog_fc all sometimes mean the same thing
"""
import datetime
import warnings
import random
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.tools.seasonal import date_part, seasonal_int
from autots.tools.holiday import holiday_flag

# these are optional packages
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.api import GLM as SM_GLM
except Exception:
    pass
try:
    from joblib import Parallel, delayed

    joblib_present = True
except Exception:
    joblib_present = False


class GLS(ModelObject):
    """Simple linear regression from statsmodels

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """

    def __init__(
        self,
        name: str = "GLS",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
        )

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        from statsmodels.regression.linear_model import GLS

        df = self.basic_profile(df)
        self.df_train = df
        Xf = pd.to_numeric(df.index, errors='coerce', downcast='integer').values
        self.model = GLS(df.values, Xf, missing='drop').fit()
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        index = self.create_forecast_index(forecast_length=forecast_length)
        Xf = pd.to_numeric(index, errors='coerce', downcast='integer').values
        forecast = self.model.predict(Xf)
        df = pd.DataFrame(forecast, columns=self.column_names, index=index)
        if just_point_forecast:
            return df
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                df,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=df.index,
                forecast_columns=df.columns,
                lower_forecast=lower_forecast,
                forecast=df,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning"""
        return {}

    def get_params(self):
        """Return dict of current parameters"""
        return {}


def glm_forecast_by_column(current_series, X, Xf, args):
    """Run one series of GLM and return prediction."""
    series_name = current_series.name
    family = args['family']
    verbose = args['verbose']
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category='RuntimeWarning')
    if str(family).lower() == 'poisson':
        from statsmodels.genmod.families.family import Poisson

        model = SM_GLM(current_series.values, X, family=Poisson(), missing='drop').fit(
            disp=verbose
        )
    elif str(family).lower() == 'binomial':
        from statsmodels.genmod.families.family import Binomial

        model = SM_GLM(current_series.values, X, family=Binomial(), missing='drop').fit(
            disp=verbose
        )
    elif str(family).lower() == 'negativebinomial':
        from statsmodels.genmod.families.family import NegativeBinomial

        model = SM_GLM(
            current_series.values, X, family=NegativeBinomial(), missing='drop'
        ).fit(disp=verbose)
    elif str(family).lower() == 'tweedie':
        from statsmodels.genmod.families.family import Tweedie

        model = SM_GLM(current_series.values, X, family=Tweedie(), missing='drop').fit(
            disp=verbose
        )
    elif str(family).lower() == 'gamma':
        from statsmodels.genmod.families.family import Gamma

        model = SM_GLM(current_series.values, X, family=Gamma(), missing='drop').fit(
            disp=verbose
        )
    else:
        family = 'Gaussian'
        model = SM_GLM(current_series.values, X, missing='drop').fit(disp=verbose)
    Pred = model.predict((Xf))
    Pred = pd.Series(Pred)
    Pred.name = series_name
    return Pred


class GLM(ModelObject):
    """Simple linear regression from statsmodels

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): type of regression (None, 'User')

    """

    def __init__(
        self,
        name: str = "GLM",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        regression_type: str = None,
        family='Gaussian',
        constant: bool = False,
        verbose: int = 1,
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
        self.family = family
        self.constant = constant

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """

        df = self.basic_profile(df)
        self.df_train = df
        if self.verbose > 1:
            self.verbose = True
        else:
            self.verbose = False
        if self.regression_type in ['User', 'user']:
            if future_regressor is None:
                raise ValueError("regression_type=user and no future_regressor passed")
            else:
                self.future_regressor_train = future_regressor
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        test_index = self.create_forecast_index(forecast_length=forecast_length)

        if self.regression_type == 'datepart':
            X = date_part(self.df_train.index, method='expanded').values
        else:
            X = pd.to_numeric(
                self.df_train.index, errors='coerce', downcast='integer'
            ).values
        if self.constant in [True, 'True', 'true']:
            from statsmodels.tools import add_constant

            X = add_constant(X, has_constant='add')
        if self.regression_type in ['User', 'user']:
            if self.future_regressor_train.ndim == 1:
                self.future_regressor_train = np.array(
                    self.future_regressor_train
                ).reshape(-1, 1)
            X = np.concatenate((X.reshape(-1, 1), self.future_regressor_train), axis=1)

        self.df_train = self.df_train.replace(0, np.nan)
        fill_vals = self.df_train.abs().min(axis=0, skipna=True)
        self.df_train = self.df_train.fillna(fill_vals).fillna(0.1)
        if self.regression_type == 'datepart':
            Xf = date_part(test_index, method='expanded').values
        else:
            Xf = pd.to_numeric(test_index, errors='coerce', downcast='integer').values
        if self.constant or self.constant == 'True':
            Xf = add_constant(Xf, has_constant='add')
        if self.regression_type == 'User':
            if future_regressor.ndim == 1:
                future_regressor = np.array(future_regressor).reshape(-1, 1)
            Xf = np.concatenate((Xf.reshape(-1, 1), future_regressor), axis=1)

        parallel = True
        cols = self.df_train.columns.tolist()
        df = self.df_train
        args = {
            'family': self.family,
            'verbose': self.verbose,
        }
        if self.verbose:
            pool_verbose = 1
        else:
            pool_verbose = 0

        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        elif not joblib_present:
            parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            df_list = Parallel(n_jobs=self.n_jobs, verbose=pool_verbose)(
                delayed(glm_forecast_by_column)(
                    current_series=df[col],
                    X=X,
                    Xf=Xf,
                    args=args,
                )
                for col in cols
            )
            df_forecast = pd.concat(df_list, axis=1)
        else:
            df_list = []
            for col in cols:
                df_list.append(glm_forecast_by_column(df[col], X, Xf, args))
            df_forecast = pd.concat(df_list, axis=1)
        df_forecast.index = test_index

        if just_point_forecast:
            return df_forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                df_forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=df_forecast.index,
                forecast_columns=df_forecast.columns,
                lower_forecast=lower_forecast,
                forecast=df_forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        family_choice = random.choices(
            [
                'Gaussian',
                'Poisson',
                'Binomial',
                'NegativeBinomial',
                'Tweedie',
                'Gamma',
            ],
            [0.1, 0.3, 0.1, 0.3, 0.1, 0.1],
        )[0]
        constant_choice = random.choices([False, True], [0.95, 0.05])[0]
        if "regressor" in method:
            regression_type_choice = "User"
        else:
            regression_type_choice = random.choices(
                [None, 'datepart', 'User'], [0.4, 0.4, 0.2]
            )[0]
        return {
            'family': family_choice,
            'constant': constant_choice,
            'regression_type': regression_type_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'family': self.family,
            'constant': self.constant,
            'regression_type': self.regression_type,
        }


class ETS(ModelObject):
    """Exponential Smoothing from Statsmodels

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        damped_trend (bool): passed through to statsmodel ETS (formerly just 'damped')
        trend (str): passed through to statsmodel ETS
        seasonal (bool): passed through to statsmodel ETS
        seasonal_periods (int): passed through to statsmodel ETS

    """

    def __init__(
        self,
        name: str = "ETS",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        damped_trend: bool = False,
        trend: str = None,
        seasonal: str = None,
        seasonal_periods: int = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.damped_trend = damped_trend
        self.trend = trend
        self.seasonal = seasonal
        if (seasonal not in ["additive", "multiplicative"]) or (
            seasonal_periods is None
        ):
            self.seasonal = None
            self.seasonal_periods = None
        else:
            self.seasonal_periods = abs(int(seasonal_periods))

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)

        self.df_train = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        parallel = True
        args = {
            'damped_trend': self.damped_trend,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'freq': self.frequency,
            'forecast_length': forecast_length,
            'verbose': self.verbose,
        }

        def ets_forecast_by_column(current_series, args):
            """Run one series of ETS and return prediction."""
            series_name = current_series.name
            with warnings.catch_warnings():
                if args['verbose'] < 2:
                    warnings.simplefilter("ignore")
                try:
                    # handle statsmodels 0.13 method changes
                    try:
                        esModel = ExponentialSmoothing(
                            current_series,
                            damped_trend=args['damped_trend'],
                            trend=args['trend'],
                            seasonal=args['seasonal'],
                            seasonal_periods=args['seasonal_periods'],
                            # initialization_method=None,
                            # freq=args['freq'],
                        )
                    except Exception as e:
                        if args['verbose'] > 0:
                            print(f"ETS error {repr(e)}")
                        esModel = ExponentialSmoothing(
                            current_series,
                            damped=args['damped_trend'],
                            trend=args['trend'],
                            seasonal=args['seasonal'],
                            seasonal_periods=args['seasonal_periods'],
                            # initialization_method='heuristic',  # estimated
                            freq=args['freq'],
                        )
                    esResult = esModel.fit()
                    srt = current_series.shape[0]
                    esPred = esResult.predict(
                        start=srt, end=srt + args['forecast_length'] - 1
                    )
                    esPred = pd.Series(esPred)
                except Exception as e:
                    # this error handling is meant for horizontal ensembles where it will only then be needed for select series
                    if args['verbose'] > 0:
                        print(f"ETS failed on {series_name} with {repr(e)}")
                    esPred = pd.Series((np.zeros((forecast_length,))), index=test_index)
            esPred.name = series_name
            return esPred

        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        elif not joblib_present:
            parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            df_list = Parallel(n_jobs=self.n_jobs)(
                delayed(ets_forecast_by_column)(self.df_train[col], args)
                for (col) in cols
            )
            forecast = pd.concat(df_list, axis=1)
        else:
            df_list = []
            for col in cols:
                df_list.append(ets_forecast_by_column(self.df_train[col], args))
            forecast = pd.concat(df_list, axis=1)
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )

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
        trend_list = ["additive", "multiplicative", None]
        trend_probability = [0.2, 0.2, 0.6]
        trend_choice = random.choices(trend_list, trend_probability)[0]
        if trend_choice in ["additive", "multiplicative"]:
            damped_choice = random.choice([True, False])
        else:
            damped_choice = False
        seasonal_list = ["additive", "multiplicative", None]
        seasonal_probability = [0.2, 0.2, 0.6]
        seasonal_choice = random.choices(seasonal_list, seasonal_probability)[0]
        if seasonal_choice in ["additive", "multiplicative"]:
            seasonal_period_choice = seasonal_int()
        else:
            seasonal_period_choice = None
        parameter_dict = {
            'damped_trend': damped_choice,
            'trend': trend_choice,
            'seasonal': seasonal_choice,
            'seasonal_periods': seasonal_period_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'damped_trend': self.damped_trend,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
        }
        return parameter_dict


def arima_seek_the_oracle(current_series, args, series):
    try:
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore", category='ConvergenceWarning')
            warnings.simplefilter("ignore")
            if args['regression_type'] in ["User", "Holiday"]:
                maModel = SARIMAX(
                    current_series,
                    order=args['order'],
                    freq=args['frequency'],
                    exog=args['regressor_train'],
                ).fit(maxiter=600, disp=args['verbose'])
            else:
                maModel = SARIMAX(
                    current_series, order=args['order'], freq=args['frequency']
                ).fit(maxiter=400, disp=args['verbose'])
            if args['regression_type'] in ["User", "Holiday"]:
                outer_forecasts = maModel.get_forecast(
                    steps=args['forecast_length'], exog=args['exog']
                )
            else:
                outer_forecasts = maModel.get_forecast(steps=args['forecast_length'])
            outer_forecasts_df = outer_forecasts.conf_int(alpha=args['alpha'])
            cforecast = outer_forecasts.summary_frame()['mean']
            clower_forecast = outer_forecasts_df.iloc[:, 0]
            cupper_forecast = outer_forecasts_df.iloc[:, 1]
    except Exception:
        cforecast = pd.Series(
            np.zeros((args['forecast_length'],)), index=args['test_index']
        )
        clower_forecast = pd.Series(
            np.zeros((args['forecast_length'],)), index=args['test_index']
        )
        cupper_forecast = pd.Series(
            np.zeros((args['forecast_length'],)), index=args['test_index']
        )
    cforecast.name = current_series.name
    clower_forecast.name = current_series.name
    cupper_forecast.name = current_series.name
    return (cforecast, clower_forecast, cupper_forecast)


class ARIMA(ModelObject):
    """ARIMA from Statsmodels.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        p (int): is the number of autoregressive steps,
        d (int): is the number of differences needed for stationarity
        q (int): is the number of lagged forecast errors in the prediction.
        regression_type (str): type of regression (None, 'User', or 'Holiday')
        n_jobs (int): passed to joblib for multiprocessing. Set to none for context manager.

    """

    def __init__(
        self,
        name: str = "ARIMA",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        p: int = 0,
        d: int = 1,
        q: int = 0,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
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
        self.p = p
        self.d = d
        self.q = q
        self.order = (p, d, q)

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied .

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.regressor_train = None
        method_str = str(self.regression_type).lower()
        if method_str == 'holiday':
            self.regressor_train = holiday_flag(
                df.index, country=self.holiday_country
            ).values
        elif method_str == "user":
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but future_regressor not supplied"
                )
            else:
                self.regressor_train = future_regressor
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

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        alpha = 1 - self.prediction_interval
        if self.regression_type == 'Holiday':

            future_regressor = holiday_flag(test_index, country=self.holiday_country)
        if self.regression_type is not None:
            assert (
                future_regressor.shape[0] == forecast_length
            ), "regressor not equal to forecast length"
        if self.regression_type in ["User", "Holiday"]:
            if future_regressor.values.ndim == 1:
                exog = pd.DataFrame(future_regressor).values.reshape(-1, 1)
            else:
                exog = future_regressor.values
        else:
            exog = None

        args = {
            'order': self.order,
            'regression_type': self.regression_type,
            'regressor_train': self.regressor_train,
            'exog': exog,
            'frequency': self.frequency,
            'alpha': alpha,
            'verbose': self.verbose,
            'test_index': test_index,
            'forecast_length': forecast_length,
        }
        parallel = True
        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        elif not joblib_present:
            parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            verbs = 0 if self.verbose < 1 else self.verbose - 1
            df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                delayed(arima_seek_the_oracle)(
                    current_series=self.df_train[col], args=args, series=col
                )
                for col in cols
            )
            complete = list(map(list, zip(*df_list)))
        else:
            df_list = []
            for col in cols:
                df_list.append(arima_seek_the_oracle(self.df_train[col], args, col))
            complete = list(map(list, zip(*df_list)))
        forecast = pd.concat(complete[0], axis=1)
        lower_forecast = pd.concat(complete[1], axis=1)
        upper_forecast = pd.concat(complete[2], axis=1)

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
        """Return dict of new parameters for parameter tuning.

        large p,d,q can be very slow (a p of 30 can take hours)
        """
        p_choice = random.choices(
            [0, 1, 2, 3, 4, 5, 7, 12],
            [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )[0]
        d_choice = random.choices([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1])[0]
        q_choice = random.choices(
            [0, 1, 2, 3, 4, 5, 7, 12],
            [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )[0]
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_list = [None, 'User', 'Holiday']
            regression_probability = [0.5, 0.3, 0.2]
            regression_choice = random.choices(regression_list, regression_probability)[
                0
            ]

        parameter_dict = {
            'p': p_choice,
            'd': d_choice,
            'q': q_choice,
            'regression_type': regression_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'p': self.p,
            'd': self.d,
            'q': self.q,
            'regression_type': self.regression_type,
        }
        return parameter_dict


class UnobservedComponents(ModelObject):
    """UnobservedComponents from Statsmodels.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        model_kwargs (dict): additional model params to pass through underlying statsmodel

        regression_type (str): type of regression (None, 'User', or 'Holiday')

    """

    def __init__(
        self,
        name: str = "UnobservedComponents",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = 1,
        level: str = "smooth trend",  # bool with other method
        trend: bool = False,
        cycle: bool = False,
        damped_cycle: bool = False,
        irregular: bool = False,
        autoregressive: int = None,
        stochastic_cycle: bool = False,
        stochastic_trend: bool = False,
        stochastic_level: bool = False,
        maxiter: int = 100,
        cov_type: str = "opg",
        method: str = "lbfgs",
        model_kwargs: dict = None,
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
        self.level = level
        self.trend = trend
        self.cycle = cycle
        self.damped_cycle = damped_cycle
        self.irregular = irregular
        self.stochastic_level = stochastic_level
        self.stochastic_cycle = stochastic_cycle
        self.stochastic_trend = stochastic_trend
        self.maxiter = maxiter
        self.cov_type = cov_type
        self.method = method
        self.autoregressive = autoregressive
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.regressor_train = None

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        if self.regression_type == 'Holiday':
            self.regressor_train = holiday_flag(
                df.index, country=self.holiday_country
            ).values
        else:
            if self.regression_type is not None:
                if future_regressor is None:
                    raise ValueError(
                        "regression_type='User' but no future_regressor supplied"
                    )
                elif (np.array(future_regressor).shape[0]) != (df.shape[0]):
                    self.regression_type = None
                else:
                    self.regressor_train = np.array(future_regressor)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=None,
        just_point_forecast: bool = False,
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
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        if self.regression_type == 'Holiday':
            future_regressor = holiday_flag(
                test_index, country=self.holiday_country
            ).values
        if self.regression_type is not None:
            assert (
                len(future_regressor) == forecast_length
            ), "regressor not equal to forecast length"
            future_regressor = np.array(future_regressor)

        parallel = True
        alpha = 1 - self.prediction_interval
        args = {
            'freq': self.frequency,
            'regressor_train': self.regressor_train,
            'alpha': alpha,
            'level': self.level,
            'forecast_length': forecast_length,
            'regression_type': self.regression_type,
            'verbose_bool': False,
            'exog': future_regressor,
            'maxiter': self.maxiter,
            'cov_type': self.cov_type,
            'method': self.method,
            'autoregressive': self.autoregressive,
            'model_kwargs': self.model_kwargs,
        }

        def uc_forecast_by_column(current_series, args):
            """Run one series of Unobserved Components and return prediction."""
            series_name = current_series.name
            with warnings.catch_warnings():
                if not args['verbose_bool']:
                    warnings.simplefilter("ignore")
                if args['regression_type'] in ["User", "Holiday"]:
                    maModel = UnobservedComponents(
                        current_series,
                        freq=args['freq'],
                        exog=args['regressor_train'],
                        level=args['level'],
                        autoregressive=args['autoregressive'],
                        **args['model_kwargs'],
                    ).fit(
                        disp=args['verbose_bool'],
                        maxiter=args['maxiter'],
                        cov_type=args['cov_type'],
                        method=args['method'],
                    )
                else:
                    maModel = UnobservedComponents(
                        current_series,
                        freq=args['freq'],
                        level=args['level'],
                        # trend=args['trend'],
                        # cycle=args['cycle'],
                        # damped_cycle=args['damped_cycle'],
                        # irregular=args['irregular'],
                        autoregressive=args['autoregressive'],
                        **args['model_kwargs']
                        # stochastic_cycle=args['stochastic_cycle'],
                        # stochastic_level=args['stochastic_level'],
                        # stochastic_trend=args['stochastic_trend'],
                    ).fit(
                        disp=args['verbose_bool'],
                        maxiter=args['maxiter'],
                        cov_type=args['cov_type'],
                        method=args['method'],
                    )
                series_len = current_series.shape[0]
                if args['regression_type'] in ["User", "Holiday"]:
                    outer_forecasts = maModel.get_prediction(
                        start=series_len,
                        end=series_len + args['forecast_length'] - 1,
                        exog=args['exog'],
                    )
                else:
                    outer_forecasts = maModel.get_forecast(args['forecast_length'])
                outer_forecasts_df = outer_forecasts.conf_int(alpha=args['alpha'])
            cforecast = outer_forecasts.summary_frame()['mean']
            clower_forecast = outer_forecasts_df.iloc[:, 0]
            cupper_forecast = outer_forecasts_df.iloc[:, 1]
            cforecast.name = series_name
            clower_forecast.name = series_name
            cupper_forecast.name = series_name
            return (cforecast, clower_forecast, cupper_forecast)

        parallel = True
        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        elif not joblib_present:
            parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            verbs = 0 if self.verbose < 1 else self.verbose - 1
            df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                delayed(uc_forecast_by_column)(
                    current_series=self.df_train[col],
                    args=args,
                )
                for col in cols
            )
            complete = list(map(list, zip(*df_list)))
        else:
            df_list = []
            for col in cols:
                df_list.append(uc_forecast_by_column(self.df_train[col], args))
            complete = list(map(list, zip(*df_list)))
        forecast = pd.concat(complete[0], axis=1)
        lower_forecast = pd.concat(complete[1], axis=1)
        upper_forecast = pd.concat(complete[2], axis=1)

        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='historic_quantile',
                prediction_interval=self.prediction_interval,
            )

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
        levels = [
            'irregular',
            'fixed intercept',
            'deterministic constant',
            'local level',
            'random walk',
            'fixed slope',
            'deterministic trend',
            'local linear deterministic trend',
            'random walk with drift',
            'local linear trend',
            'smooth trend',
            'random trend',
        ]
        level_choice = random.choice(levels)
        """
        level_choice = random.choice([True, False])
        if level_choice:
            trend_choice = random.choice([True, False])
        else:
            trend_choice = False
        cycle_choice = random.choice([True, False])
        if cycle_choice:
            damped_cycle_choice = random.choice([True, False])
        else:
            damped_cycle_choice = False
        irregular_choice = random.choice([True, False])
        stochastic_trend_choice = random.choice([True, False])
        stochastic_level_choice = random.choice([True, False])
        stochastic_cycle_choice = random.choice([True, False])
        """
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_list = [None, 'User', 'Holiday']
            regression_probability = [0.6, 0.2, 0.2]
            regression_choice = random.choices(regression_list, regression_probability)[
                0
            ]

        return {
            'level': level_choice,
            'maxiter': random.choice([50, 100, 250]),
            'cov_type': random.choices(
                ["opg", "oim", "approx", 'robust'], [0.8, 0.1, 0.1, 0.1]
            )[0],
            'method': random.choices(
                ["lbfgs", "bfgs", "powell", "cg", "newton", "nm"],
                [0.8, 0.1, 0.1, 0.1, 0.1, 0.1],
            )[0],
            'autoregressive': random.choices([None, 1, 2], [0.8, 0.2, 0.1])[0],
            'regression_type': regression_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'level': self.level,
            'maxiter': self.maxiter,
            'cov_type': self.cov_type,
            'method': self.method,
            'autoregressive': self.autoregressive,
            'regression_type': self.regression_type,
        }


class DynamicFactor(ModelObject):
    """DynamicFactor from Statsmodels

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        regression_type (str): type of regression (None, 'User', or 'Holiday')

    """

    def __init__(
        self,
        name: str = "DynamicFactor",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        k_factors: int = 1,
        factor_order: int = 0,
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
        )
        self.k_factors = k_factors
        self.factor_order = factor_order

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        if self.verbose > 1:
            self.verbose = True
        else:
            self.verbose = False
        self.df_train = df

        if self.regression_type == 'Holiday':
            self.regressor_train = holiday_flag(
                df.index, country=self.holiday_country
            ).values
        else:
            if self.regression_type is not None:
                if future_regressor is None:
                    raise ValueError(
                        "regression_type='User' but future_regressor not passed"
                    )
                else:
                    self.regressor_train = future_regressor

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts

         maModel = DynamicFactor(df_train, freq = 'MS', k_factors = 2, factor_order=2).fit()
         maPred = maModel.predict()
        """
        predictStartTime = datetime.datetime.now()
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        if self.regression_type == 'Holiday':
            future_regressor = holiday_flag(
                test_index, country=self.holiday_country
            ).values
        if self.regression_type is not None:
            assert (
                len(future_regressor) == forecast_length
            ), "regressor not equal to forecast length"

        if self.regression_type in ["User", "Holiday", 'user']:
            maModel = DynamicFactor(
                self.df_train,
                freq=self.frequency,
                exog=self.regressor_train,
                k_factors=self.k_factors,
                factor_order=self.factor_order,
            ).fit(disp=self.verbose, maxiter=100)
            if future_regressor.values.ndim == 1:
                exog = future_regressor.values.reshape(-1, 1)
            else:
                exog = future_regressor.values
            forecast = maModel.predict(
                start=test_index[0], end=test_index[-1], exog=exog
            )
        else:
            maModel = DynamicFactor(
                self.df_train,
                freq=self.frequency,
                k_factors=self.k_factors,
                factor_order=self.factor_order,
            ).fit(disp=self.verbose, maxiter=100)
            forecast = maModel.predict(start=test_index[0], end=test_index[-1])

        if just_point_forecast:
            return forecast
        else:
            # outer forecasts
            alpha = 1 - self.prediction_interval
            # predict_results = maModel.get_prediction(start='2020',end='2021')
            if self.regression_type in ["User", "Holiday", 'user']:
                outer_forecasts = maModel.get_forecast(steps=forecast_length, exog=exog)
            else:
                outer_forecasts = maModel.get_forecast(steps=forecast_length)
            outer_forecasts_df = outer_forecasts.conf_int(alpha=alpha)
            df_size = int(outer_forecasts_df.shape[1] / 2)
            lower_df = outer_forecasts_df.iloc[:, 0:df_size]
            lower_df = lower_df.rename(columns=lambda x: x[6:])
            upper_df = outer_forecasts_df.iloc[:, df_size:]
            upper_df = upper_df.rename(columns=lambda x: x[6:])

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=test_index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_df,
                forecast=forecast,
                upper_forecast=upper_df,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        k_factors_choice = random.choices([0, 1, 2, 3, 10], [0.1, 0.4, 0.2, 0.2, 0.1])[
            0
        ]
        factor_order_choice = random.choices([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1])[0]

        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_list = [None, 'User', 'Holiday']
            regression_probability = [0.6, 0.2, 0.2]
            regression_choice = random.choices(regression_list, regression_probability)[
                0
            ]

        parameter_dict = {
            'k_factors': k_factors_choice,
            'factor_order': factor_order_choice,
            'regression_type': regression_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'k_factors': self.k_factors,
            'factor_order': self.factor_order,
            'regression_type': self.regression_type,
        }
        return parameter_dict


class VECM(ModelObject):
    """VECM from Statsmodels

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        regression_type (str): type of regression (None, 'User', or 'Holiday')

    """

    def __init__(
        self,
        name: str = "VECM",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        deterministic: str = 'n',
        k_ar_diff: int = 1,
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
        )
        self.deterministic = deterministic
        self.k_ar_diff = k_ar_diff

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        type_str = str(self.regression_type).lower()
        if type_str == 'holiday':
            self.regressor_train = holiday_flag(df.index, country=self.holiday_country)
        elif type_str == "user":
            if future_regressor is None:
                raise ValueError(
                    "regression_type='User' but no future_regressor supplied"
                )
            else:
                self.regressor_train = future_regressor

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        from statsmodels.tsa.vector_ar.vecm import VECM

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        if self.regression_type == 'Holiday':
            future_regressor = holiday_flag(test_index, country=self.holiday_country)
        elif self.regression_type is not None:
            assert (
                future_regressor.shape[0] == forecast_length
            ), "regressor row count not equal to forecast length"

        # LinAlgError: SVD did not converge (occurs when NaN in train data)
        if self.regression_type in ["User", "Holiday", 'user']:
            maModel = VECM(
                self.df_train,
                freq=self.frequency,
                exog=np.array(self.regressor_train),
                deterministic=self.deterministic,
                k_ar_diff=self.k_ar_diff,
            ).fit()
            # don't ask me why it is exog_fc here and not exog like elsewhere
            forecast = maModel.predict(
                steps=forecast_length, exog_fc=np.array(future_regressor)
            )
        else:
            maModel = VECM(
                self.df_train,
                freq=self.frequency,
                deterministic=self.deterministic,
                k_ar_diff=self.k_ar_diff,
            ).fit()
            forecast = maModel.predict(steps=forecast_length)
        forecast = pd.DataFrame(forecast, index=test_index, columns=self.column_names)

        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='historic_quantile',
                prediction_interval=self.prediction_interval,
            )

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
        deterministic_choice = np.random.choice(
            a=["n", "co", "ci", "lo", "li", "cili", "colo"],
            size=1,
            p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ).item()
        k_ar_diff_choice = np.random.choice(
            a=[0, 1, 2, 3], size=1, p=[0.1, 0.5, 0.2, 0.2]
        ).item()

        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_list = [None, 'User', 'Holiday']
            regression_probability = [0.8, 0.15, 0.05]
            regression_choice = np.random.choice(
                a=regression_list, size=1, p=regression_probability
            ).item()

        parameter_dict = {
            'deterministic': deterministic_choice,
            'k_ar_diff': k_ar_diff_choice,
            'regression_type': regression_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'deterministic': self.deterministic,
            'k_ar_diff': self.k_ar_diff,
            'regression_type': self.regression_type,
        }
        return parameter_dict


class VARMAX(ModelObject):
    """VARMAX from Statsmodels

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        regression_type (str): type of regression (None, 'User', or 'Holiday')
    """

    def __init__(
        self,
        name: str = "VARMAX",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        order: tuple = (1, 0),
        trend: str = 'c',
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
        )
        self.order = order
        self.trend = trend

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)

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
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        from statsmodels.tsa.statespace.varmax import VARMAX

        maModel = VARMAX(
            self.df_train, freq=self.frequency, order=self.order, trend=self.trend
        ).fit(disp=self.verbose_bool)
        forecast = maModel.predict(start=test_index[0], end=test_index[-1])
        if just_point_forecast:
            return forecast
        else:
            # outer forecasts
            alpha = 1 - self.prediction_interval
            # predict_results = maModel.get_prediction(start='2020',end='2021')
            outer_forecasts = maModel.get_forecast(steps=forecast_length)
            outer_forecasts_df = outer_forecasts.conf_int(alpha=alpha)
            df_size = int(outer_forecasts_df.shape[1] / 2)
            lower_df = outer_forecasts_df.iloc[:, 0:df_size]
            lower_df = lower_df.rename(columns=lambda x: x[6:])
            upper_df = outer_forecasts_df.iloc[:, df_size:]
            upper_df = upper_df.rename(columns=lambda x: x[6:])

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=test_index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_df,
                forecast=forecast,
                upper_forecast=upper_df,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        # make these big and it's REAL slow, and if both p and q are non zero
        ar_choice = random.choices(
            [0, 1, 2, 5, 7, 10], [0.5, 0.5, 0.2, 0.01, 0.01, 0.001]
        )[0]
        if ar_choice == 0 or "deep" in method:
            ma_choice = random.choices([1, 2, 5, 7, 10], [0.8, 0.2, 0.01, 0.01, 0.001])[
                0
            ]
        else:
            ma_choice = 0
        trend_choice = random.choices(
            ['n', 'c', 't', 'ct', 'poly'], [0.2, 0.4, 0.1, 0.2, 0.1]
        )[0]
        if trend_choice == 'poly':
            trend_choice = [
                random.randint(0, 2),
                random.randint(0, 2),
                random.randint(0, 2),
                random.randint(0, 2),
            ]
        return {'order': (ar_choice, ma_choice), 'trend': trend_choice}

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {'order': self.order, 'trend': self.trend}
        return parameter_dict


class VAR(ModelObject):
    """VAR from Statsmodels.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        regression_type (str): type of regression (None, 'User', or 'Holiday')
    """

    def __init__(
        self,
        name: str = "VAR",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        maxlags: int = 15,
        ic: str = 'fpe',
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
        )
        self.maxlags = maxlags
        self.ic = ic

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        if self.regression_type == 'Holiday':
            self.regressor_train = holiday_flag(
                df.index, country=self.holiday_country
            ).values
        else:
            if self.regression_type is not None:
                if (np.array(future_regressor).shape[0]) != (df.shape[0]):
                    self.regression_type = None
                else:
                    self.regressor_train = future_regressor

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        from statsmodels.tsa.api import VAR

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        if self.regression_type == 'Holiday':
            future_regressor = holiday_flag(
                test_index, country=self.holiday_country
            ).values
        if self.regression_type is not None:
            assert (
                len(future_regressor) == forecast_length
            ), "regressor not equal to forecast length"
        if (self.df_train < 0).any(axis=None):
            from autots.tools.transform import PositiveShift

            transformer = PositiveShift(center_one=True)
            self.df_train = transformer.fit_transform(self.df_train)
        else:
            from autots.tools.transform import EmptyTransformer

            transformer = EmptyTransformer()

        if self.regression_type in ["User", "Holiday"]:
            maModel = VAR(
                self.df_train, freq=self.frequency, exog=self.regressor_train
            ).fit(maxlags=self.maxlags, ic=self.ic, trend='n')
            forecast, lower_forecast, upper_forecast = maModel.forecast_interval(
                steps=len(test_index),
                exog_future=future_regressor,
                y=self.df_train.values,
            )
        else:
            maModel = VAR(self.df_train, freq=self.frequency).fit(
                ic=self.ic, maxlags=self.maxlags
            )
            forecast, lower_forecast, upper_forecast = maModel.forecast_interval(
                steps=len(test_index),
                y=self.df_train.values,
                alpha=1 - self.prediction_interval,
            )
        forecast = pd.DataFrame(forecast)
        forecast.index = test_index
        forecast.columns = self.column_names
        forecast = transformer.inverse_transform(forecast)
        lower_forecast = pd.DataFrame(lower_forecast)
        lower_forecast.index = test_index
        lower_forecast.columns = self.column_names
        lower_forecast = transformer.inverse_transform(lower_forecast)
        upper_forecast = pd.DataFrame(upper_forecast)
        upper_forecast.index = test_index
        upper_forecast.columns = self.column_names
        upper_forecast = transformer.inverse_transform(upper_forecast)

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
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_list = [None, 'User', 'Holiday']
            regression_probability = [0.9, 0.05, 0.05]
            regression_choice = np.random.choice(
                a=regression_list, size=1, p=regression_probability
            ).item()
        maxlags_choice = np.random.choice([None, 5, 15], size=1).item()
        ic_choice = np.random.choice(['fpe', 'aic', 'bic', 'hqic'], size=1).item()

        parameter_dict = {
            'regression_type': regression_choice,
            'maxlags': maxlags_choice,
            'ic': ic_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'regression_type': self.regression_type,
            'maxlags': self.maxlags,
            'ic': self.ic,
        }
        return parameter_dict


class Theta(ModelObject):
    """Theta Model from Statsmodels

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        params from Theta Model as per statsmodels

    """

    def __init__(
        self,
        name: str = "Theta",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        deseasonalize: bool = True,
        use_test: bool = True,
        difference: bool = False,
        period: int = None,
        theta: float = 2,
        use_mle: bool = False,
        method: str = "auto",
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.deseasonalize = deseasonalize
        self.difference = difference
        self.use_test = use_test
        self.period = period
        self.method = method
        self.theta = theta
        self.use_mle = use_mle

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)

        self.df_train = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        from statsmodels.tsa.forecasting.theta import ThetaModel

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        args = {
            'deseasonalize': self.deseasonalize,
            'difference': self.difference,
            'use_test': self.use_test,
            'method': self.method,
            'period': self.period,
            'forecast_length': forecast_length,
            'prediction_interval': self.prediction_interval,
            'theta': self.theta,
            'use_mle': self.use_mle,
            'verbose': self.verbose,
        }

        def theta_forecast_by_column(current_series, args):
            """Run one series of Theta and return prediction."""
            series_name = current_series.name
            esModel = ThetaModel(
                current_series,
                deseasonalize=args['deseasonalize'],
                difference=args['difference'],
                use_test=args['use_test'],
                method=args['method'],
                period=args['period'],
            )
            with warnings.catch_warnings():
                # warnings.simplefilter("ignore", category='ConvergenceWarning')
                if args['verbose'] < 2:
                    warnings.simplefilter("ignore")
                # fit model
                modelResult = esModel.fit(use_mle=args['use_mle'])
                # generate forecasts
                esPred = modelResult.forecast(
                    steps=args['forecast_length'], theta=args['theta']
                )
                bound_predict = modelResult.prediction_intervals(
                    steps=args['forecast_length'],
                    theta=args['theta'],
                    alpha=(1 - args['prediction_interval']),
                )
                # overly clever identification of which is lower and upper
                sumz = bound_predict.sum()
                lower_forecast = bound_predict[sumz.idxmin()]
                upper_forecast = bound_predict[sumz.idxmax()]
            esPred = pd.Series(esPred)
            esPred.name = series_name
            lower_forecast.name = series_name
            upper_forecast.name = series_name
            return (esPred, lower_forecast, upper_forecast)

        parallel = True
        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 5:
            parallel = False
        elif not joblib_present:
            parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            verbs = 0 if self.verbose < 1 else self.verbose - 1
            df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                delayed(theta_forecast_by_column)(
                    current_series=self.df_train[col], args=args
                )
                for col in cols
            )
            complete = list(map(list, zip(*df_list)))
        else:
            df_list = []
            for col in cols:
                df_list.append(theta_forecast_by_column(self.df_train[col], args))
            complete = list(map(list, zip(*df_list)))
        forecast = pd.concat(complete[0], axis=1)
        lower_forecast = pd.concat(complete[1], axis=1)
        upper_forecast = pd.concat(complete[2], axis=1)
        forecast.index = test_index
        lower_forecast.index = test_index
        upper_forecast.index = test_index

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
        return {
            'deseasonalize': random.choices([True, False], [0.8, 0.2])[0],
            'difference': random.choice([True, False]),
            'use_test': random.choices([True, False], [0.8, 0.2])[0],
            'method': "auto",
            'period': None,
            'theta': random.choice([1.2, 1.4, 1.6, 2, 2.5, 3, 4]),
            'use_mle': random.choices([True, False], [0.2, 0.8])[0],
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'deseasonalize': self.deseasonalize,
            'difference': self.difference,
            'use_test': self.use_test,
            'method': self.method,
            'period': self.period,
            'theta': self.theta,
            'use_mle': self.use_mle,
        }


class ARDL(ModelObject):
    """ARDL from Statsmodels.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        lags (int): lags 1 to max
        trend (str): n/c/t/ct
        order (int): 0 to max
        regression_type (str): type of regression (None, 'User', or 'Holiday')
        n_jobs (int): passed to joblib for multiprocessing. Set to none for context manager.

    """

    def __init__(
        self,
        name: str = "ARDL",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        lags: int = 2,
        trend: str = "c",
        order: int = 0,
        regression_type: str = "holiday",
        holiday_country: str = 'US',
        random_seed: int = 2020,
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
        self.lags = lags
        self.trend = trend
        self.order = order

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied .

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.regressor_train = None
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
        from statsmodels.tsa.api import ARDL

        def ardl_per_column(current_series, args):
            with warnings.catch_warnings():
                # warnings.simplefilter("ignore", category='ConvergenceWarning')
                # warnings.simplefilter("ignore", category='ValueWarning')
                if args['verbose'] < 2:
                    warnings.simplefilter("ignore", category=UserWarning)
                if args['regression_type'] in ["User", "user", "holiday"]:
                    maModel = ARDL(
                        current_series,
                        lags=args['lags'],
                        trend=args['trend'],
                        order=args['order'],
                        exog=args['regressor_train'],
                    ).fit()
                else:
                    maModel = ARDL(
                        current_series,
                        lags=args['lags'],
                        trend=args['trend'],
                        order=args['order'],
                    ).fit()
                series_len = current_series.shape[0]
                if args['regression_type'] in ["User", "user", "holiday"]:
                    outer_forecasts = maModel.get_prediction(
                        start=series_len,
                        end=series_len + args['forecast_length'] - 1,
                        exog_oos=args['exog'],
                    )
                else:
                    outer_forecasts = maModel.get_prediction(
                        start=series_len, end=series_len + args['forecast_length'] - 1
                    )
                outer_forecasts_df = outer_forecasts.conf_int(alpha=args['alpha'])
                cforecast = outer_forecasts.summary_frame()['mean']
                clower_forecast = outer_forecasts_df.iloc[:, 0]
                cupper_forecast = outer_forecasts_df.iloc[:, 1]
            cforecast.name = current_series.name
            clower_forecast.name = current_series.name
            cupper_forecast.name = current_series.name
            return (cforecast, clower_forecast, cupper_forecast)

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        alpha = 1 - self.prediction_interval
        if self.regression_type == 'holiday':
            future_regressor = pd.DataFrame(
                holiday_flag(test_index, country=self.holiday_country)
            )
        if self.regression_type is not None:
            assert (
                future_regressor.shape[0] == forecast_length
            ), "regressor not equal to forecast length"

        args = {
            'lags': self.lags,
            'order': self.order,
            'regression_type': self.regression_type,
            'regressor_train': self.regressor_train,
            'exog': future_regressor,
            'trend': self.trend,
            'alpha': alpha,
            'forecast_length': forecast_length,
            'verbose': self.verbose,
        }
        parallel = True
        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        elif not joblib_present:
            parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            verbs = 0 if self.verbose < 1 else self.verbose - 1
            df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                delayed(ardl_per_column)(
                    current_series=self.df_train[col],
                    args=args,
                )
                for col in cols
            )
            complete = list(map(list, zip(*df_list)))
        else:
            df_list = []
            for col in cols:
                df_list.append(ardl_per_column(self.df_train[col], args))
            complete = list(map(list, zip(*df_list)))
        forecast = pd.concat(complete[0], axis=1)
        lower_forecast = pd.concat(complete[1], axis=1)
        upper_forecast = pd.concat(complete[2], axis=1)

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
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_list = [None, 'User', 'holiday']
            regression_probability = [0.3, 0.5, 0.5]
            regression_choice = random.choices(regression_list, regression_probability)[
                0
            ]
        if regression_choice is None:
            order_choice = 0
        else:
            order_choice = random.choices([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1])[0]

        return {
            'lags': random.choices([1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1])[0],
            'trend': random.choice(['n', 'c', 't', 'ct']),
            'order': order_choice,
            'regression_type': regression_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            'lags': self.lags,
            'trend': self.trend,
            'order': self.order,
            'regression_type': self.regression_type,
        }


class DynamicFactorMQ(ModelObject):
    """DynamicFactorMQ from Statsmodels

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """

    def __init__(
        self,
        name: str = "DynamicFactorMQ",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        factors: int = 1,
        factor_orders: int = 2,
        factor_multiplicities: int = None,
        idiosyncratic_ar1: bool = False,
        maxiter: int = 1000,
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
        )
        self.factors = factors
        self.factor_orders = factor_orders
        self.factor_multiplicities = factor_multiplicities
        self.idiosyncratic_ar1 = idiosyncratic_ar1
        self.maxiter = maxiter

    def fit(self, df, future_regressor=None):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        if self.verbose > 2:
            self.verbose = True
        else:
            self.verbose = False
        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=None, just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

        test_index = self.create_forecast_index(forecast_length=forecast_length)

        maModel = DynamicFactorMQ(
            self.df_train,
            freq=self.frequency,
            factors=self.factors,
            factor_orders=self.factor_orders,
            factor_multiplicities=self.factor_multiplicities,
            idiosyncratic_ar1=self.idiosyncratic_ar1,
        ).fit(disp=self.verbose, maxiter=self.maxiter)
        forecast = maModel.predict(start=test_index[0], end=test_index[-1])

        if just_point_forecast:
            return forecast
        else:
            # outer forecasts
            alpha = 1 - self.prediction_interval
            # predict_results = maModel.get_prediction(start='2020',end='2021')
            outer_forecasts = maModel.get_forecast(steps=forecast_length)
            outer_forecasts_df = outer_forecasts.conf_int(alpha=alpha)
            df_size = int(outer_forecasts_df.shape[1] / 2)
            lower_df = outer_forecasts_df.iloc[:, 0:df_size]
            lower_df = lower_df.rename(columns=lambda x: x[6:])
            upper_df = outer_forecasts_df.iloc[:, df_size:]
            upper_df = upper_df.rename(columns=lambda x: x[6:])

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=test_index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_df,
                forecast=forecast,
                upper_forecast=upper_df,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        k_factors_choice = random.choices([1, 2, 3, 10], [0.4, 0.2, 0.2, 0.1])[0]
        factor_order_choice = random.choices([1, 2, 3, 4], [0.3, 0.2, 0.1, 0.02])[0]

        parameter_dict = {
            'factors': k_factors_choice,
            'factor_orders': factor_order_choice,
            "factor_multiplicities": random.choice([None, 2]),
            'idiosyncratic_ar1': random.choice([True, False]),
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'factors': self.factors,
            'factor_orders': self.factor_orders,
            'factor_multiplicities': self.factor_multiplicities,
            'idiosyncratic_ar1': self.idiosyncratic_ar1,
        }
        return parameter_dict
