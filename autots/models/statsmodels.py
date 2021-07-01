"""Statsmodels based forecasting models."""
import datetime
import random
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability
from autots.tools.seasonal import date_part, seasonal_int


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

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        from statsmodels.regression.linear_model import GLS

        df = self.basic_profile(df)
        self.df_train = df
        self.model = GLS(df.values, (df.index.astype(int).values), missing='drop').fit()
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=[], just_point_forecast=False
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
        forecast = self.model.predict(index.astype(int).values)
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

    def fit(self, df, future_regressor=[]):
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
        if self.regression_type == 'User':
            if (np.array(future_regressor).shape[0]) != (df.shape[0]):
                self.regression_type = None
            else:
                self.future_regressor_train = future_regressor
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=[], just_point_forecast=False
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
        from statsmodels.api import GLM
        from autots.tools.seasonal import date_part

        if self.regression_type == 'datepart':
            X = date_part(self.df_train.index, method='expanded').values
        else:
            X = pd.to_numeric(
                self.df_train.index, errors='coerce', downcast='integer'
            ).values
        if self.constant in [True, 'True', 'true']:
            from statsmodels.tools import add_constant

            X = add_constant(X, has_constant='add')
        if self.regression_type == 'User':
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
        """
        forecast = pd.DataFrame()
        for y in self.df_train.columns:
            current_series = self.df_train[y]
            if str(self.family).lower() == 'poisson':
                from statsmodels.genmod.families.family import Poisson

                model = GLM(
                    current_series.values, X, family=Poisson(), missing='drop'
                ).fit(disp=self.verbose)
            elif str(self.family).lower() == 'binomial':
                from statsmodels.genmod.families.family import Binomial

                model = GLM(
                    current_series.values, X, family=Binomial(), missing='drop'
                ).fit(disp=self.verbose)
            elif str(self.family).lower() == 'negativebinomial':
                from statsmodels.genmod.families.family import NegativeBinomial

                model = GLM(
                    current_series.values, X, family=NegativeBinomial(), missing='drop'
                ).fit(disp=self.verbose)
            elif str(self.family).lower() == 'tweedie':
                from statsmodels.genmod.families.family import Tweedie

                model = GLM(
                    current_series.values, X, family=Tweedie(), missing='drop'
                ).fit(disp=self.verbose)
            elif str(self.family).lower() == 'gamma':
                from statsmodels.genmod.families.family import Gamma

                model = GLM(
                    current_series.values, X, family=Gamma(), missing='drop'
                ).fit(disp=self.verbose)
            else:
                self.family = 'Gaussian'
                model = GLM(current_series.values, X, missing='drop').fit()
            current_forecast = model.predict((Xf))
            forecast = pd.concat([forecast, pd.Series(current_forecast)], axis=1)
        df_forecast = pd.DataFrame(forecast)
        df_forecast.columns = self.column_names
        
        # handle weird pickling errors for multiprocessing
        try:
            from joblib import wrap_non_picklable_objects
        except Exception:
            def wrap_non_picklable_objects(ob):
                return ob
            parallel = False
        """
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

        def forecast_by_column(df, X, Xf, args, col):
            """Run one series of ETS and return prediction."""
            current_series = df[col]
            series_name = current_series.name
            family = args['family']
            verbose = args['verbose']
            if str(family).lower() == 'poisson':
                from statsmodels.genmod.families.family import Poisson

                model = GLM(
                    current_series.values, X, family=Poisson(), missing='drop'
                ).fit(disp=verbose)
            elif str(family).lower() == 'binomial':
                from statsmodels.genmod.families.family import Binomial

                model = GLM(
                    current_series.values, X, family=Binomial(), missing='drop'
                ).fit(disp=verbose)
            elif str(family).lower() == 'negativebinomial':
                from statsmodels.genmod.families.family import NegativeBinomial

                model = GLM(
                    current_series.values, X, family=NegativeBinomial(), missing='drop'
                ).fit(disp=verbose)
            elif str(family).lower() == 'tweedie':
                from statsmodels.genmod.families.family import Tweedie

                model = GLM(
                    current_series.values, X, family=Tweedie(), missing='drop'
                ).fit(disp=verbose)
            elif str(family).lower() == 'gamma':
                from statsmodels.genmod.families.family import Gamma

                model = GLM(
                    current_series.values, X, family=Gamma(), missing='drop'
                ).fit(disp=verbose)
            else:
                family = 'Gaussian'
                model = GLM(current_series.values, X, missing='drop').fit(disp=verbose)
            Pred = model.predict((Xf))
            Pred = pd.Series(Pred)
            Pred.name = series_name
            return Pred

        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        else:
            try:
                from joblib import Parallel, delayed
            except Exception:
                parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            df_list = Parallel(n_jobs=self.n_jobs, verbose=pool_verbose)(
                delayed(forecast_by_column)(df=df, X=X, Xf=Xf, args=args, col=col)
                for col in cols
            )
            df_forecast = pd.concat(df_list, axis=1)
        else:
            df_list = []
            for col in cols:
                df_list.append(forecast_by_column(df, X, Xf, args, col))
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

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)

        self.df_train = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=[], just_point_forecast=False
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

        def forecast_by_column(df, args, col):
            """Run one series of ETS and return prediction."""
            current_series = df[col]
            series_name = current_series.name
            try:
                # handle statsmodels 0.13 method changes
                try:
                    esModel = ExponentialSmoothing(
                        current_series,
                        damped_trend=args['damped_trend'],
                        trend=args['trend'],
                        seasonal=args['seasonal'],
                        seasonal_periods=args['seasonal_periods'],
                        initialization_method=None,
                        freq=args['freq'],
                    )
                except Exception:
                    esModel = ExponentialSmoothing(
                        current_series,
                        damped=args['damped_trend'],
                        trend=args['trend'],
                        seasonal=args['seasonal'],
                        seasonal_periods=args['seasonal_periods'],
                        # initialization_method='heuristic',  # estimated
                        freq=args['freq'],
                    )
                esModel = esModel.fit()
                srt = current_series.shape[0]
                esPred = esModel.predict(
                    start=srt, end=srt + args['forecast_length'] - 1
                )
                esPred = pd.Series(esPred)
            except Exception as e:
                if args['verbose'] > 1:
                    print(f"ETS failed on {series_name} with {repr(e)}")
                esPred = pd.Series((np.zeros((forecast_length,))), index=test_index)
            esPred.name = series_name
            return esPred

        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        else:
            try:
                from joblib import Parallel, delayed
            except Exception:
                parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            df_list = Parallel(n_jobs=self.n_jobs)(
                delayed(forecast_by_column)(self.df_train, args, col) for (col) in cols
            )
            forecast = pd.concat(df_list, axis=1)
        else:
            df_list = []
            for col in cols:
                df_list.append(forecast_by_column(self.df_train, args, col))
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

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied .

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.regressor_train = None
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag

            self.regressor_train = holiday_flag(
                df.index, country=self.holiday_country
            ).values
        else:
            if self.regression_type is not None:
                if (np.array(future_regressor).shape[0]) != (df.shape[0]):
                    self.regression_type = None
                else:
                    self.regressor_train = future_regressor
        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=[], just_point_forecast=False
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
        # from statsmodels.tsa.arima_model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        test_index = self.create_forecast_index(forecast_length=forecast_length)
        alpha = 1 - self.prediction_interval
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag

            future_regressor = holiday_flag(test_index, country=self.holiday_country)
        if self.regression_type != None:
            assert (
                len(future_regressor) == forecast_length
            ), "regressor not equal to forecast length"
        if self.regression_type in ["User", "Holiday"]:
            if future_regressor.values.ndim == 1:
                exog = pd.DataFrame(future_regressor).values.reshape(-1, 1)
            else:
                exog = future_regressor.values
        else:
            exog = None
        """
        forecast = []
        upper_forecast = []
        lower_forecast = []
        for series in self.df_train.columns:
            current_series = self.df_train[series]
            try:
                if self.regression_type in ["User", "Holiday"]:
                    maModel = SARIMAX(
                        current_series,
                        order=self.order,
                        freq=self.frequency,
                        exog=self.regressor_train,
                    ).fit(maxiter=600)
                else:
                    maModel = SARIMAX(
                        current_series, order=self.order, freq=self.frequency
                    ).fit(maxiter=400, disp=self.verbose)
                if self.regression_type in ["User", "Holiday"]:
                    outer_forecasts = maModel.get_forecast(steps=forecast_length, exog=exog)
                else:
                    outer_forecasts = maModel.get_forecast(steps=forecast_length)
                outer_forecasts_df = outer_forecasts.conf_int(alpha=alpha)
                cforecast = outer_forecasts.summary_frame()['mean']
                clower_forecast = outer_forecasts_df.iloc[:, 0]
                cupper_forecast = outer_forecasts_df.iloc[:, 1]
            except Exception:
                cforecast = pd.Series(np.zeros((forecast_length,)))
                clower_forecast = pd.Series(np.zeros((forecast_length,)))
                cupper_forecast = pd.Series(np.zeros((forecast_length,)))
            cforecast.name = current_series.name
            clower_forecast.name = current_series.name
            cupper_forecast.name = current_series.name
            forecast.append(cforecast)
            lower_forecast.append(clower_forecast)
            upper_forecast.append(cupper_forecast)
        forecast = pd.concat(forecast, axis=1)
        lower_forecast = pd.concat(lower_forecast, axis=1)
        upper_forecast = pd.concat(upper_forecast, axis=1)
        """

        def seek_the_oracle(df, args, series):
            current_series = df[series]
            try:
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
                    outer_forecasts = maModel.get_forecast(
                        steps=args['forecast_length']
                    )
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
        else:
            try:
                from joblib import Parallel, delayed
            except Exception:
                parallel = False
        # joblib multiprocessing to loop through series
        if parallel:
            verbs = 0 if self.verbose < 1 else self.verbose - 1
            df_list = Parallel(n_jobs=self.n_jobs, verbose=(verbs))(
                delayed(seek_the_oracle)(df=self.df_train, args=args, series=col)
                for col in cols
            )
            complete = list(map(list, zip(*df_list)))
        else:
            df_list = []
            for col in cols:
                df_list.append(seek_the_oracle(self.df_train, args, col))
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
        p_choice = np.random.choice(
            a=[0, 1, 2, 3, 4, 5, 7, 12],
            size=1,
            p=[0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ).item()
        d_choice = np.random.choice(
            a=[0, 1, 2, 3], size=1, p=[0.4, 0.3, 0.2, 0.1]
        ).item()
        q_choice = np.random.choice(
            a=[0, 1, 2, 3, 4, 5, 7, 12],
            size=1,
            p=[0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ).item()
        regression_list = [None, 'User', 'Holiday']
        regression_probability = [0.4, 0.4, 0.2]
        regression_choice = np.random.choice(
            a=regression_list, size=1, p=regression_probability
        ).item()

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
        level: bool = False,
        trend: bool = False,
        cycle: bool = False,
        damped_cycle: bool = False,
        irregular: bool = False,
        stochastic_cycle: bool = False,
        stochastic_trend: bool = False,
        stochastic_level: bool = False,
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

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df
        self.regressor_train = None

        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag

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
        self,
        forecast_length: int,
        future_regressor=[],
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
            from autots.tools.holiday import holiday_flag

            future_regressor = holiday_flag(
                test_index, country=self.holiday_country
            ).values
        if self.regression_type is not None:
            assert (
                len(future_regressor) == forecast_length
            ), "regressor not equal to forecast length"

        parallel = True
        args = {
            'freq': self.frequency,
            'exog': self.regressor_train,
            'level': self.level,
            'trend': self.trend,
            'cycle': self.cycle,
            'damped_cycle': self.damped_cycle,
            'irregular': self.irregular,
            'stochastic_cycle': self.stochastic_cycle,
            'stochastic_level': self.stochastic_level,
            'stochastic_trend': self.stochastic_trend,
            'forecast_length': forecast_length,
            'regression_type': self.regression_type,
            'verbose_bool': self.verbose_bool,
            'test_index': test_index,
            'future_regressor': future_regressor,
        }

        def forecast_by_column(df, args, col):
            """Run one series of ETS and return prediction."""
            current_series = df[col]
            series_name = current_series.name
            test_index = args['test_index']
            try:
                if args['regression_type'] in ["User", "Holiday"]:
                    maModel = UnobservedComponents(
                        current_series,
                        freq=args['frequency'],
                        exog=args['regressor_train'],
                        level=args['level'],
                        trend=args['trend'],
                        cycle=args['cycle'],
                        damped_cycle=args['damped_cycle'],
                        irregular=args['irregular'],
                        stochastic_cycle=args['stochastic_cycle'],
                        stochastic_level=args['stochastic_level'],
                        stochastic_trend=args['stochastic_trend'],
                    ).fit(disp=args['verbose_bool'])
                    maPred = maModel.predict(
                        start=test_index[0],
                        end=test_index[-1],
                        exog=args['future_regressor'],
                    )
                else:
                    maModel = UnobservedComponents(
                        current_series,
                        freq=args['frequency'],
                        level=args['level'],
                        trend=args['trend'],
                        cycle=args['cycle'],
                        damped_cycle=args['damped_cycle'],
                        irregular=args['irregular'],
                        stochastic_cycle=args['stochastic_cycle'],
                        stochastic_level=args['stochastic_level'],
                        stochastic_trend=args['stochastic_trend'],
                    ).fit(disp=args['verbose_bool'])
                    maPred = maModel.predict(start=test_index[0], end=test_index[-1])
            except Exception as e:
                if args['verbose_bool']:
                    print(f"Uncomp failed on {series_name} with {repr(e)}")
                maPred = pd.Series((np.zeros((forecast_length,))), index=test_index)

            maPred = pd.Series(maPred)
            maPred.name = series_name
            return maPred

        cols = self.df_train.columns.tolist()
        if self.n_jobs in [0, 1] or len(cols) < 4:
            parallel = False
        else:
            try:
                from joblib import Parallel, delayed
            except Exception:
                if self.verbose > 1:
                    print("Joblib import failed, not parallel...")
                parallel = False
        # joblib multiprocessing to loop through series
        # print(f"parallel is {parallel} and n_jobs is {self.n_jobs}")
        if parallel:
            df_list = Parallel(n_jobs=self.n_jobs)(
                delayed(forecast_by_column)(self.df_train, args, col) for (col) in cols
            )
            forecast = pd.concat(df_list, axis=1)
        else:
            df_list = []
            for col in cols:
                df_list.append(forecast_by_column(self.df_train, args, col))
            forecast = pd.concat(df_list, axis=1)
        """
        forecast = pd.DataFrame()
        for series in self.df_train.columns:
            current_series = self.df_train[series].copy()
            try:
                if self.regression_type in ["User", "Holiday"]:
                    maModel = UnobservedComponents(
                        current_series,
                        freq=self.frequency,
                        exog=self.regressor_train,
                        level=self.level,
                        trend=self.trend,
                        cycle=self.cycle,
                        damped_cycle=self.damped_cycle,
                        irregular=self.irregular,
                        stochastic_cycle=self.stochastic_cycle,
                        stochastic_level=self.stochastic_level,
                        stochastic_trend=self.stochastic_trend,
                    ).fit(disp=self.verbose_bool)
                    maPred = maModel.predict(
                        start=test_index[0], end=test_index[-1], exog=future_regressor
                    )
                else:
                    maModel = UnobservedComponents(
                        current_series,
                        freq=self.frequency,
                        level=self.level,
                        trend=self.trend,
                        cycle=self.cycle,
                        damped_cycle=self.damped_cycle,
                        irregular=self.irregular,
                        stochastic_cycle=self.stochastic_cycle,
                        stochastic_level=self.stochastic_level,
                        stochastic_trend=self.stochastic_trend,
                    ).fit(disp=self.verbose_bool)
                    maPred = maModel.predict(start=test_index[0], end=test_index[-1])
            except Exception:
                maPred = pd.Series((np.zeros((forecast_length,))), index=test_index)
            forecast = pd.concat([forecast, maPred], axis=1)
        forecast.columns = self.column_names
        """

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
        level_choice = np.random.choice(a=[True, False], size=1, p=[0.5, 0.5]).item()
        if level_choice:
            trend_choice = np.random.choice(
                a=[True, False], size=1, p=[0.5, 0.5]
            ).item()
        else:
            trend_choice = False
        cycle_choice = np.random.choice(a=[True, False], size=1, p=[0.5, 0.5]).item()
        if cycle_choice:
            damped_cycle_choice = np.random.choice(
                a=[True, False], size=1, p=[0.5, 0.5]
            ).item()
        else:
            damped_cycle_choice = False
        irregular_choice = np.random.choice(
            a=[True, False], size=1, p=[0.5, 0.5]
        ).item()
        stochastic_trend_choice = np.random.choice(
            a=[True, False], size=1, p=[0.5, 0.5]
        ).item()
        stochastic_level_choice = np.random.choice(
            a=[True, False], size=1, p=[0.5, 0.5]
        ).item()
        stochastic_cycle_choice = np.random.choice(
            a=[True, False], size=1, p=[0.5, 0.5]
        ).item()
        regression_list = [None, 'User', 'Holiday']
        regression_probability = [0.6, 0.2, 0.2]
        regression_choice = np.random.choice(
            a=regression_list, size=1, p=regression_probability
        ).item()

        parameter_dict = {
            'level': level_choice,
            'trend': trend_choice,
            'cycle': cycle_choice,
            'damped_cycle': damped_cycle_choice,
            'irregular': irregular_choice,
            'stochastic_trend': stochastic_trend_choice,
            'stochastic_level': stochastic_level_choice,
            'stochastic_cycle': stochastic_cycle_choice,
            'regression_type': regression_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'level': self.level,
            'trend': self.trend,
            'cycle': self.cycle,
            'damped_cycle': self.damped_cycle,
            'irregular': self.irregular,
            'stochastic_trend': self.stochastic_trend,
            'stochastic_level': self.stochastic_level,
            'stochastic_cycle': self.stochastic_cycle,
            'regression_type': self.regression_type,
        }
        return parameter_dict


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

    def fit(self, df, future_regressor=[]):
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
            from autots.tools.holiday import holiday_flag

            self.regressor_train = holiday_flag(
                df.index, country=self.holiday_country
            ).values
        else:
            if self.regression_type != None:
                if (np.array(future_regressor).shape[0]) != (df.shape[0]):
                    self.regression_type = None
                else:
                    self.regressor_train = future_regressor

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=[], just_point_forecast=False
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
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
            from autots.tools.holiday import holiday_flag

            future_regressor = holiday_flag(
                test_index, country=self.holiday_country
            ).values
        if self.regression_type is not None:
            assert (
                len(future_regressor) == forecast_length
            ), "regressor not equal to forecast length"

        if self.regression_type in ["User", "Holiday"]:
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
            if self.regression_type in ["User", "Holiday"]:
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
        k_factors_choice = np.random.choice(
            a=[0, 1, 2, 3, 10], size=1, p=[0.1, 0.4, 0.2, 0.2, 0.1]
        ).item()
        factor_order_choice = np.random.choice(
            a=[0, 1, 2, 3], size=1, p=[0.4, 0.3, 0.2, 0.1]
        ).item()

        regression_list = [None, 'User', 'Holiday']
        regression_probability = [0.6, 0.2, 0.2]
        regression_choice = np.random.choice(
            a=regression_list, size=1, p=regression_probability
        ).item()

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
        deterministic: str = 'nc',
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

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag

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
        self, forecast_length: int, future_regressor=[], just_point_forecast=False
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
            from autots.tools.holiday import holiday_flag

            future_regressor = holiday_flag(
                test_index, country=self.holiday_country
            ).values
        if self.regression_type is not None:
            assert (
                len(future_regressor) == forecast_length
            ), "regressor not equal to forecast length"

        if self.regression_type in ["User", "Holiday"]:
            maModel = VECM(
                self.df_train,
                freq=self.frequency,
                exog=self.regressor_train,
                deterministic=self.deterministic,
                k_ar_diff=self.k_ar_diff,
            ).fit()
            # forecast = maModel.predict(start=test_index[0], end=test_index[-1], exog = future_regressor)
            forecast = maModel.predict(steps=len(test_index), exog=future_regressor)
        else:
            maModel = VECM(
                self.df_train,
                freq=self.frequency,
                deterministic=self.deterministic,
                k_ar_diff=self.k_ar_diff,
            ).fit()
            forecast = maModel.predict(steps=len(test_index))
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
            a=["nc", "co", "ci", "lo", "li", "cili", "colo"],
            size=1,
            p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ).item()
        k_ar_diff_choice = np.random.choice(
            a=[0, 1, 2, 3], size=1, p=[0.1, 0.5, 0.2, 0.2]
        ).item()

        regression_list = [None, 'User', 'Holiday']
        regression_probability = [0.9, 0.05, 0.05]
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

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)

        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self, forecast_length: int, future_regressor=[], just_point_forecast=False
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
        ar_choice = random.choices([0, 1, 2], [0.3, 0.5, 0.2])[0]
        if ar_choice == 0:
            ma_choice = random.choices([1, 2], [0.8, 0.2])[0]
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

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag

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
        self, forecast_length: int, future_regressor=[], just_point_forecast=False
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
            from autots.tools.holiday import holiday_flag

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
            ).fit(maxlags=15, ic='fpe', trend='nc')
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
