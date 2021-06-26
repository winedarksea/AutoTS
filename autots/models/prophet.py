"""
Facebook's Prophet

Since Prophet install can be finicky on Windows, it will be an optional dependency.
"""
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject

import logging

# https://stackoverflow.com/questions/27361427/how-to-properly-deal-with-optional-features-in-python
try:
    try:  # no idea when they switched
        from fbprophet import Prophet
    except Exception:
        from prophet import Prophet
except Exception:  # except ImportError
    _has_prophet = False
else:
    _has_prophet = True


class FBProphet(ModelObject):
    """Facebook's Prophet

    'thou shall count to 3, no more, no less, 3 shall be the number thou shall count, and the number of the counting
    shall be 3. 4 thou shall not count, neither count thou 2, excepting that thou then preceed to 3.' -Python

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holidays
        regression_type (str): type of regression (None, 'User')

    """

    def __init__(
        self,
        name: str = "FBProphet",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday: bool = False,
        regression_type: str = None,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = None,
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
        self.holiday = holiday

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        if not _has_prophet:
            raise ImportError("Package fbprophet is required")

        df = self.basic_profile(df)
        self.regressor_train = None
        self.dimensionality_reducer = None
        if self.regression_type == 'User':
            """
            print("the shape of the input is: {}".format(str(((np.array(future_regressor).shape[0])))))
            print("the shape of the training data is: {}".format(str(df.shape[0])))
            """
            if (np.array(future_regressor).shape[0]) != (df.shape[0]):
                self.regression_type = None
            else:
                if future_regressor.ndim > 1:
                    from sklearn.decomposition import PCA

                    self.dimensionality_reducer = PCA(n_components=1).fit(
                        future_regressor
                    )
                    self.regressor_train = self.dimensionality_reducer.transform(
                        future_regressor
                    )
                else:
                    self.regressor_train = future_regressor.copy()

        # this is a hack to utilize regressors with a name unlikely to exist
        random_two = "n9032380gflljWfu8233koWQop3"
        random_one = "nJIOVxgQ0vZGC7nx_"
        self.regressor_name = random_one if random_one not in df.columns else random_two
        self.df_train = df

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=[],
        just_point_forecast: bool = False,
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
        if not _has_prophet:
            raise ImportError("Package fbprophet is required")
        predictStartTime = datetime.datetime.now()
        # if self.regression_type != None:
        #   assert len(future_regressor) == forecast_length, "regressor not equal to forecast length"
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        forecast = pd.DataFrame()
        lower_forecast = pd.DataFrame()
        upper_forecast = pd.DataFrame()
        if self.verbose <= 0:
            logging.getLogger('fbprophet').setLevel(logging.WARNING)
        if self.regression_type == 'User':
            self.df_train[self.regressor_name] = self.regressor_train

        """
        for series in self.df_train.columns:
            current_series = self.df_train.copy()
            current_series['y'] = current_series[series]
            current_series['ds'] = current_series.index
            m = Prophet(interval_width=self.prediction_interval)
            if self.holiday:
                m.add_country_holidays(country_name=self.holiday_country)
            if self.regression_type == 'User':
                m.add_regressor(self.regressor_name)
            m = m.fit(current_series)
            future = m.make_future_dataframe(periods=forecast_length)
            if self.regression_type == 'User':
                if future_regressor.ndim > 1:
                    a = self.dimensionality_reducer.transform(future_regressor)
                    a = np.append(self.regressor_train, a)
                else:
                    a = np.append(self.regressor_train, future_regressor.values)
                future[self.regressor_name] = a
            fcst = m.predict(future)
            fcst = fcst.tail(forecast_length)  # remove the backcast
            forecast = pd.concat([forecast, fcst['yhat']], axis=1)
            lower_forecast = pd.concat([lower_forecast, fcst['yhat_lower']], axis=1)
            upper_forecast = pd.concat([upper_forecast, fcst['yhat_upper']], axis=1)
        forecast.columns = self.column_names
        forecast.index = test_index
        lower_forecast.columns = self.column_names
        lower_forecast.index = test_index
        upper_forecast.columns = self.column_names
        upper_forecast.index = test_index
        """

        def seek_the_oracle(df, args, series):
            current_series = df
            current_series['y'] = current_series[series]
            current_series['ds'] = current_series.index
            m = Prophet(interval_width=args['prediction_interval'])
            if args['holiday']:
                m.add_country_holidays(country_name=args['holiday_country'])
            if args['regression_type'] == 'User':
                m.add_regressor(args['regressor_name'])
            m = m.fit(current_series)
            future = m.make_future_dataframe(periods=forecast_length)
            if args['regression_type'] == 'User':
                if future_regressor.ndim > 1:
                    a = args['dimensionality_reducer'].transform(future_regressor)
                    a = np.append(args['regressor_train'], a)
                else:
                    a = np.append(args['regressor_train'], future_regressor.values)
                future[args['regressor_name']] = a
            fcst = m.predict(future)
            fcst = fcst.tail(forecast_length)  # remove the backcast
            forecast = fcst['yhat']
            forecast.name = series
            lower_forecast = fcst['yhat_lower']
            lower_forecast.name = series
            upper_forecast = fcst['yhat_upper']
            upper_forecast.name = series
            return (forecast, lower_forecast, upper_forecast)

        args = {
            'holiday': self.holiday,
            'holiday_country': self.holiday_country,
            'regression_type': self.regression_type,
            'regressor_name': self.regressor_name,
            'regressor_train': self.regressor_train,
            'dimensionality_reducer': self.dimensionality_reducer,
            'prediction_interval': self.prediction_interval,
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
                forecast_index=forecast.index,
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
        holiday_choice = np.random.choice(a=[True, False], size=1, p=[0.8, 0.2]).item()
        regression_list = [None, 'User']
        regression_probability = [0.8, 0.2]
        regression_choice = np.random.choice(
            a=regression_list, size=1, p=regression_probability
        ).item()

        parameter_dict = {
            'holiday': holiday_choice,
            'regression_type': regression_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'holiday': self.holiday,
            'regression_type': self.regression_type,
        }
        return parameter_dict
