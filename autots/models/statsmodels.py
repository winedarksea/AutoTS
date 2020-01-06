"""
Statsmodels based Models
"""
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject
from autots.evaluator.auto_model import PredictionObject
from autots.tools.probabilistic import Point_to_Probability


class GLM(ModelObject):
    """Simple linear regression from statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "GLM", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country = holiday_country, random_seed = random_seed)
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        from statsmodels.regression.linear_model import GLS
        df = self.basic_profile(df)
        self.df_train = df
        self.model = GLS(df.values, (df.index.astype( int ).values), missing = 'drop').fit()
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, preord_regressor = [], just_point_forecast = False):
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
        forecast = self.model.predict(index.astype( int ).values)
        df = pd.DataFrame(forecast, columns = self.column_names, index = index)
        if just_point_forecast:
            return df
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, df, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = df.index,
                                          forecast_columns = df.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=df, upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self,method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {}


class ETS(ModelObject):
    """Exponential Smoothing from Statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        damped (bool): passed through to statsmodel ETS
        trend (str): passed through to statsmodel ETS
        seasonal (bool): passed through to statsmodel ETS
        seasonal_periods (int): passed through to statsmodel ETS

    """
    def __init__(self, name: str = "ETS", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, damped: bool = False, 
                 trend: str = None, seasonal: str=None, seasonal_periods:int=None, 
                 holiday_country: str = 'US',random_seed: int = 2020):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country = holiday_country, random_seed = random_seed)
        self.damped = damped
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.df_train = df
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, preord_regressor = [], just_point_forecast = False):
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
        forecast = pd.DataFrame()
        for series in self.df_train.columns:
            current_series = self.df_train[series].copy()
            esModel = ExponentialSmoothing(current_series, damped = self.damped, trend = self.trend, seasonal=self.seasonal,seasonal_periods=self.seasonal_periods, freq = self.frequency).fit()
            esPred = esModel.predict(start=test_index[0], end=test_index[-1])
            forecast = pd.concat([forecast, esPred], axis = 1)
        forecast.columns = self.column_names
        if forecast.isnull().all(axis = 0).astype(int).sum() > 0:
            print("One or more series have failed to optimize with ETS model")
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, forecast, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = test_index,
                                          forecast_columns = forecast.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=forecast, 
                                          upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        trend_list = ["additive", "multiplicative", None]
        trend_probability = [0.2, 0.2, 0.6]
        trend_choice = np.random.choice(a = trend_list, size = 1, p = trend_probability).item()
        if trend_choice in ["additive", "multiplicative"]:
            damped_choice = np.random.choice([True, False], size = 1).item()
        else:
            damped_choice = False
        seasonal_list = ["additive", "multiplicative", None]
        seasonal_probability = [0.2, 0.2, 0.6]
        seasonal_choice = np.random.choice(a = seasonal_list, size = 1, p = seasonal_probability).item()
        if seasonal_choice in ["additive", "multiplicative"]:
            seasonal_period_choice = np.random.choice([4,7,12,30], size = 1).item()
        else:
            seasonal_period_choice = None
        parameter_dict = {
                        'damped' : damped_choice,
                        'trend': trend_choice,
                        'seasonal': seasonal_choice,
                        'seasonal_periods': seasonal_period_choice
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'damped' : self.damped,
                        'trend': self.trend,
                        'seasonal': self.seasonal,
                        'seasonal_periods': self.seasonal_periods
                        }
        return parameter_dict

        
class ARIMA(ModelObject):
    """ARIMA from Statsmodels
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        p (int): is the number of autoregressive steps,
        d (int): is the number of differences needed for stationarity
        q (int): is the number of lagged forecast errors in the prediction . 
        regression_type (str): type of regression (None, 'User', or 'Holiday')

    """
    def __init__(self, name: str = "ARIMA", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, p:int = 0, d:int = 1,
                 q:int = 0, regression_type: str = None, holiday_country: str = 'US',
                 random_seed: int = 2020):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type = regression_type, 
                             holiday_country = holiday_country, random_seed = random_seed)
        self.p = p
        self.d = d
        self.q = q
        self.order = (p, d, q)
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag
            self.regressor_train = holiday_flag(df.index, country = self.holiday_country).values
        else:
            if self.regression_type != None:
                if (len(preord_regressor) != len(df.index)):
                    self.regression_type = None
            self.regressor_train = preord_regressor
        self.df_train = df
        
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, preord_regressor = [], just_point_forecast = False):
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
        from statsmodels.tsa.arima_model import ARIMA
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        if self.regression_type == 'Holiday':
            from autots.tools.holiday import holiday_flag
            preord_regressor = holiday_flag(test_index, country = self.holiday_country).values
        if self.regression_type != None:
            assert len(preord_regressor) == forecast_length, "regressor not equal to forecast length"
        forecast = pd.DataFrame()
        for series in self.df_train.columns:
            current_series = self.df_train[series].copy()
            try:
                if (self.regression_type == "User") or (self.regression_type == "Holiday"):
                    maModel = ARIMA(current_series, order = self.order, freq = self.frequency, exog = self.regressor_train).fit(maxiter = 600)
                    maPred = maModel.predict(start=test_index[0], end=test_index[-1], exog = preord_regressor)
                else:
                    maModel = ARIMA(current_series, order = self.order, freq = self.frequency).fit(maxiter = 400)
                    maPred = maModel.predict(start=test_index[0], end=test_index[-1])
            except Exception:
                maPred = pd.Series((np.zeros((forecast_length,))), index = test_index)
            forecast = pd.concat([forecast, maPred], axis = 1)
        forecast.columns = self.column_names
        
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, forecast, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = test_index,
                                          forecast_columns = forecast.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=forecast, 
                                          upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self, method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        
        large p,d,q can be very slow (a p of 30 can take hours, whereas 5 takes seconds)
        """
        p_choice = np.random.choice(a = [0,1,2,3,4,5,7,10], size = 1, p = [0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1]).item()
        d_choice = np.random.choice(a = [0,1,2,3], size = 1, p = [0.4, 0.3, 0.2, 0.1]).item()
        q_choice = np.random.choice(a = [0,1,2,3,4,5,7,10], size = 1, p = [0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1]).item()
        regression_list = [None, 'User', 'Holiday']
        regression_probability = [0.2, 0.6, 0.2]
        regression_choice = np.random.choice(a = regression_list, size = 1, p = regression_probability).item()

        parameter_dict = {
                        'p' : p_choice,
                        'd': d_choice,
                        'q': q_choice,
                        'regression_type': regression_choice
                        }
        return parameter_dict
    
    def get_params(self):
        """Return dict of current parameters
        """
        parameter_dict = {
                        'p' : self.p,
                        'd': self.d,
                        'q': self.q,
                        'regression_type': self.regression_type
                        }
        return parameter_dict